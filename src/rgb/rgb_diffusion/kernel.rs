/*
For copyright information see the COPYRIGHT file included in the top-level
directory of this distribution.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

     1. Redistributions of source code must retain the included copyright notice,
        this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the included copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. Neither the names of the copyright holders nor the names of their
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Low-level per-pixel anisotropic (heat-PDE) diffusion kernel math. Pure
//! neighbourhood math with no orchestration or allocation policy: the
//! bit-exact heavy worker shared by the full-scan and scoped paths lives here.

use ndarray::{ArrayView2, ArrayViewMut2, NdFloat};
use std::cmp;

/// Matrix representing rotations
struct RotationMatrix<T: NdFloat + Default>([[T; 2]; 2]);

/// Flattened 3x3 matrix
struct Flat3Matrix<T: NdFloat + Default>([T; 9]);

/// Types of anisotropic diffusion behavior
///
/// Determines how diffusion responds to image gradients and edges.
///
/// * `Isotrope` - Uniform diffusion in all directions (isotropic)
/// * `Isophote` - Diffusion perpendicular to isophotes (edges)
/// * `Gradient` - Diffusion aligned with gradient direction
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum IsotropyType {
    Isotrope,
    Isophote,
    Gradient,
}

/// Neighbourhood offset used by [`diffusion_compute`] (mult*H is the 3x3 window
/// step about the current cell).
const H: usize = 1;

#[inline]
fn find_gradients<T: NdFloat + Default>(pixels: Flat3Matrix<T>) -> [T; 2] {
    let pixels = pixels.0;
    [
        (pixels[7] - pixels[1]) / T::from(2.0).unwrap(),
        (pixels[5] - pixels[3]) / T::from(2.0).unwrap(),
    ]
}

/// Generate isotropic Laplacian kernel for diffusion
///
/// Returns a 3×3 kernel (9 values in row-major order) that computes
/// the Laplacian with edge-preserving properties. The center coefficient
/// is negative to implement the Laplacian operator: Σ(neighbor - center)
///
/// # Returns
/// * `Flat3Matrix` - Kernel coefficients in row-major order
#[inline]
fn isotrop_laplacian<T: NdFloat + Default>() -> Flat3Matrix<T> {
    Flat3Matrix([
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(-3.0).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.25).unwrap(),
    ])
}

/// Compute rotation matrix for isophote-based anisotropic diffusion
///
/// Generates a 2×2 rotation matrix that aligns diffusion perpendicular
/// to image isophotes (lines of constant intensity). This preserves edges
/// while smoothing along the isophote direction.
///
/// # Arguments
/// * `c2` - Anisotropy factor squared (exp(-|∇u| * anisotropy))
/// * `cos_theta_sin_theta` - Product of normalized gradient components
/// * `cos_theta2` - Square of normalized x-gradient
/// * `sin_theta2` - Square of normalized y-gradient
///
/// # Returns
/// * `[[T; 2]; 2]` - 2×2 rotation matrix
#[inline]
fn rotation_matrix_isophote<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
) -> RotationMatrix<T> {
    let mut a: [[T; 2]; 2] = [[T::default(); 2]; 2];
    a[0][0] = cos_theta2 + c2 * sin_theta2;
    a[1][1] = c2 * cos_theta2 + sin_theta2;
    a[0][1] = (c2 - T::from(1.0).unwrap()) * cos_theta_sin_theta;
    a[1][0] = a[0][1];
    RotationMatrix(a)
}

/// Compute rotation matrix for gradient-based anisotropic diffusion
///
/// Generates a 2×2 rotation matrix that aligns diffusion with the image
/// gradient direction. This smooths in the direction of greatest change
/// while preserving perpendicular features.
///
/// # Arguments
/// * `c2` - Anisotropy factor squared (exp(-|∇u| * anisotropy))
/// * `cos_theta_sin_theta` - Product of normalized gradient components
/// * `cos_theta2` - Square of normalized x-gradient
/// * `sin_theta2` - Square of normalized y-gradient
///
/// # Returns
/// * `[[T; 2]; 2]` - 2×2 rotation matrix
#[inline]
fn rotation_matrix_gradient<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
) -> RotationMatrix<T> {
    let mut a: [[T; 2]; 2] = [[T::default(); 2]; 2];
    a[0][0] = c2 * cos_theta2 + sin_theta2;
    a[1][1] = cos_theta2 + c2 * sin_theta2;
    a[0][1] = (T::from(1.0).unwrap() - c2) * cos_theta_sin_theta;
    a[1][0] = a[0][1];
    RotationMatrix(a)
}

/// Build 3×3 diffusion kernel from 2×2 rotation matrix
///
/// Converts a rotation matrix derived from image gradients into a
/// 3×3 convolution kernel for anisotropic diffusion. The kernel
/// incorporates the rotation information to create directionally-
/// dependent diffusion behavior.
///
/// # Arguments
/// * `a` - 2×2 rotation matrix encoding gradient information
///
/// # Returns
/// * `Flat3Matrix<T>` - 3×3 kernel coefficients in row-major order
#[inline]
fn build_matrix<T: NdFloat + Default>(a: RotationMatrix<T>) -> Flat3Matrix<T> {
    let a = a.0;
    let b11 = a[0][1] / T::from(2.0).unwrap();
    let b13 = -b11;
    let b22 = T::from(-2.0).unwrap() * (a[0][0] + a[1][1]);

    Flat3Matrix([b11, a[1][1], b13, a[0][0], b22, a[0][0], b13, a[1][1], b11])
}

/// Compute 3×3 diffusion kernel based on isotropy type
///
/// Selects and constructs the appropriate kernel for anisotropic diffusion
/// based on the specified isotropy type. The kernel encodes directional
/// diffusion behavior derived from image structure tensors.
///
/// # Arguments
/// * `c2` - Anisotropy factor squared (exp(-|∇u| * anisotropy))
/// * `cos_theta_sin_theta` - Cross product of normalized gradient
/// * `cos_theta2` - Square of normalized x-gradient
/// * `sin_theta2` - Square of normalized y-gradient
/// * `isotropy_type` - Type of anisotropic behavior to apply
///
/// # Returns
/// * `Flat3Matrix<T>` - 3×3 kernel coefficients in row-major order
#[inline]
fn compute_kernel<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
    isotropy_type: &IsotropyType,
) -> Flat3Matrix<T> {
    match isotropy_type {
        IsotropyType::Isotrope => isotrop_laplacian(),
        IsotropyType::Isophote => {
            let iso_matrix =
                rotation_matrix_isophote(c2, cos_theta_sin_theta, cos_theta2, sin_theta2);
            build_matrix(iso_matrix)
        }
        IsotropyType::Gradient => {
            let rot_matrix =
                rotation_matrix_gradient(c2, cos_theta_sin_theta, cos_theta2, sin_theta2);
            build_matrix(rot_matrix)
        }
    }
}

/// Apply anisotropic diffusion PDE to image subregions using four-derivative approach
///
/// Implements the heat equation with variable diffusion coefficients:
///
///     ∂u/∂t = ∇·(c(x,y,∇u)∇u)
///
/// where c(x,y,∇u) = exp(-|∇u| * anisotropy) controls edge preservation.
/// High gradient regions (edges) have low diffusion coefficients, while
/// flat regions diffuse more strongly.
///
/// The four-derivative approach captures directional information:
/// [0,2] - Gradient-based diffusion (horizontal/vertical components)
/// [1,3] - Laplacian-based diffusion (diagonal components)
///
/// # Arguments
/// * `hf_input` - High-frequency component (wavelet detail coefficients)
/// * `lf_input` - Low-frequency component (wavelet approximation)
/// * `output` - Output array, modified in-place
/// * `mult` - Scale multiplier (1<<scale), determines neighborhood size
/// * `anisotropy` - Anisotropy parameters for four diffusion terms
/// * `isotropy_type` - Isotropy mode for each of four terms
/// * `variance_threshold` - Minimum variance for numerical stability
/// * `regularization` - Regularization parameter
/// * `current_radius_sq` - Current scale radius squared
/// * `abcd` - Four diffusion coefficients weighted by position
/// * `strength` - Overall diffusion strength multiplier
/// * `mask` - Optional boolean mask for selective pixel processing
/// * `scoped` - Optional linear indices (row-major, idx=r*width+c) of the cells
///   to process for the current scale (`masked ∪ read-halo`). `Some` scopes the
///   loop to just those cells instead of the full image; `None` falls back to
///   the full-image loop (exactly preserving the legacy mask=None and
///   full-scan-masked behavior). The heavy masked branch and the unmasked
///   `hf + lf` passthrough are shared by both paths, so the two produce
///   identical numerics for the cells they have in common.
///
/// # Notes
/// The diffusion coefficient computation: c = exp(-|∇u| * anisotropy)
/// ensures edge preservation: high gradient = low diffusion.
///
/// Yields the `(row, col)` cells that [`heat_pde_diffusion`] must touch.
///
/// `Scoped` decodes a slice of linear indices (`idx = r*width + c`); `Full`
/// iterates the whole `h x w` Cartesian grid in row-major order. Both produce
/// the exact same cell order as the pre-refactor dual loop, so numerics are
/// bit-identical.
enum CellIter<'a> {
    Scoped {
        idxs: &'a [usize],
        width: usize,
        pos: usize,
    },
    Full {
        row: usize,
        col: usize,
        h: usize,
        w: usize,
    },
}

impl<'a> Iterator for CellIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        match self {
            CellIter::Scoped { idxs, width, pos } => {
                let &idx = idxs.get(*pos)?;
                *pos += 1;
                let width = *width;
                Some((idx / width, idx % width))
            }
            CellIter::Full { row, col, h, w } => {
                if *row >= *h {
                    return None;
                }
                let out = (*row, *col);
                *col += 1;
                if *col >= *w {
                    *col = 0;
                    *row += 1;
                }
                Some(out)
            }
        }
    }
}

pub(super) fn heat_pde_diffusion<T: NdFloat + Default>(
    hf_input: ArrayView2<T>,
    lf_input: ArrayView2<T>,
    output: ArrayViewMut2<T>,
    mult: usize,
    anisotropy: [T; 4],
    isotropy_type: [IsotropyType; 4],
    variance_threshold: T,
    regularization: T,
    current_radius_sq: T,
    abcd: [T; 4],
    strength: T,
    mask: &Option<ArrayView2<bool>>,
    scoped: Option<&[usize]>,
    clip_negatives: bool,
) {
    let mut output = output;
    let regularization_factor = regularization * current_radius_sq / T::from(9.0).unwrap();

    let (height, width) = output.dim();

    // The masked-fill reconstruction path is unclipped (`clip_negatives ==
    // false`), leaving values free to go negative; the legacy mask=None
    // diffusion clamps to non-negative (`clip_negatives == true`).
    let clip = clip_negatives;

    // The heavy masked branch uses a 3x3 neighborhood of `buffer_in` (the
    // `lf_input` argument here) at offsets ±mult*H. Neighboring unmasked values
    // are NOT constant across scales (they equal original - finer hf detail), so
    // every cell a masked pixel can read must hold a correct value. Hence the
    // loop must cover `masked ∪ read-halo` (halo = cells within Chebyshev
    // distance mult of a masked cell). Far-field unmasked cells are never read by
    // a masked pixel, so they need not be written here (they were pre-seeded by
    // cloning the input in process_image).
    let cells = match scoped {
        Some(idxs) => CellIter::Scoped {
            idxs,
            width,
            pos: 0,
        },
        None => CellIter::Full {
            row: 0,
            col: 0,
            h: height,
            w: width,
        },
    };
    for (row, col) in cells {
        let do_pixel = match mask {
            Some(m) => m[(row, col)],
            None => true,
        };
        if do_pixel {
            output[(row, col)] = diffusion_compute(
                hf_input,
                lf_input,
                row,
                col,
                mult,
                height,
                width,
                anisotropy,
                isotropy_type,
                variance_threshold,
                regularization_factor,
                abcd,
                strength,
                clip,
            );
        } else {
            output[(row, col)] = hf_input[(row, col)] + lf_input[(row, col)];
        }
    }
}

/// Shared per-pixel heavy-branch worker for [`heat_pde_diffusion`], written so
/// it `#[inline(always)]` into both the full-scan and the scoped loops, keeping
/// the exact same expression ordering (hence bit-identical numerics) across
/// paths while avoiding per-pixel closure/indirection overhead. `clip` selects
/// the legacy mask=None clamp; the masked-fill path (`clip == false`) leaves
/// values unclipped.
#[inline(always)]
fn diffusion_compute<T: NdFloat + Default>(
    hf_input: ArrayView2<T>,
    lf_input: ArrayView2<T>,
    row: usize,
    col: usize,
    mult: usize,
    height: usize,
    width: usize,
    anisotropy: [T; 4],
    isotropy_type: [IsotropyType; 4],
    variance_threshold: T,
    regularization_factor: T,
    abcd: [T; 4],
    strength: T,
    clip: bool,
) -> T {
    let mut i_neighbours: [usize; 3] = [0, 0, 0];
    let mut j_neighbours: [usize; 3] = [0, 0, 0];
    let mut neighbour_pixel_hf = [T::default(); 9];
    let mut neighbour_pixel_lf = [T::default(); 9];
    let mut c2 = [T::default(); 4];

    i_neighbours[0] = (cmp::max(row as i32 - (mult * H) as i32, 0) as i32) as usize;
    i_neighbours[1] = row;
    i_neighbours[2] = cmp::min((row + mult * H) as i32, height as i32 - 1) as usize;
    j_neighbours[0] = cmp::max(col as i32 - (mult * H) as i32, 0) as usize;
    j_neighbours[1] = col;
    j_neighbours[2] = cmp::min((col + mult * H) as i32, width as i32 - 1) as usize;

    for ii in 0..3 {
        for jj in 0..3 {
            neighbour_pixel_hf[3 * ii + jj] =
                hf_input[(i_neighbours[ii], j_neighbours[jj])];
            neighbour_pixel_lf[3 * ii + jj] =
                lf_input[(i_neighbours[ii], j_neighbours[jj])];
        }
    }

    let mut gradient = find_gradients(Flat3Matrix(neighbour_pixel_lf));
    let mut laplace = find_gradients(Flat3Matrix(neighbour_pixel_hf));

    let magnitude_grad = (gradient[0].powi(2) + gradient[1].powi(2)).sqrt();
    c2[0] = -magnitude_grad * anisotropy[0];
    c2[2] = -magnitude_grad * anisotropy[2];

    if magnitude_grad != T::default() {
        gradient[0] /= magnitude_grad;
        gradient[1] /= magnitude_grad;
    } else {
        gradient[0] = T::from(1.0).unwrap();
        gradient[1] = T::default();
    }
    let cos_theta_grad_sq = gradient[0].powi(2);
    let sin_theta_grad_sq = gradient[1].powi(2);
    let cos_theta_sin_theta_grad = gradient[0] * gradient[1];

    let magnitude_lapl = (laplace[0].powi(2) + laplace[1].powi(2)).sqrt();
    c2[1] = -magnitude_lapl * anisotropy[1];
    c2[3] = -magnitude_lapl * anisotropy[3];

    if magnitude_lapl != T::default() {
        laplace[0] /= magnitude_lapl;
        laplace[1] /= magnitude_lapl;
    } else {
        laplace[0] = T::from(1.0).unwrap();
        laplace[1] = T::default();
    }

    let cos_theta_lapl_sq = laplace[0].powi(2);
    let sin_theta_lapl_sq = laplace[1].powi(2);
    let cos_theta_sin_theta_lapl = laplace[0] * laplace[1];

    for k in 0..4 {
        c2[k] = c2[k].exp();
    }
    let kern_first = compute_kernel(
        c2[0],
        cos_theta_sin_theta_grad,
        cos_theta_grad_sq,
        sin_theta_grad_sq,
        &isotropy_type[0],
    );
    let kern_second = compute_kernel(
        c2[1],
        cos_theta_sin_theta_lapl,
        cos_theta_lapl_sq,
        sin_theta_lapl_sq,
        &isotropy_type[1],
    );
    let kern_third = compute_kernel(
        c2[2],
        cos_theta_sin_theta_grad,
        cos_theta_grad_sq,
        sin_theta_grad_sq,
        &isotropy_type[2],
    );
    let kern_fourth = compute_kernel(
        c2[3],
        cos_theta_sin_theta_lapl,
        cos_theta_lapl_sq,
        sin_theta_lapl_sq,
        &isotropy_type[3],
    );

    let mut derivatives: [T; 4] = [T::default(); 4];
    let mut variance = T::default();
    for k in 0..9 {
        derivatives[0] += kern_first.0[k] * neighbour_pixel_lf[k];
        derivatives[1] += kern_second.0[k] * neighbour_pixel_lf[k];
        derivatives[2] += kern_third.0[k] * neighbour_pixel_hf[k];
        derivatives[3] += kern_fourth.0[k] * neighbour_pixel_hf[k];
        variance += neighbour_pixel_hf[k].powi(2);
    }

    variance = variance_threshold + variance * regularization_factor;

    let mut acc = T::default();
    for k in 0..4 {
        acc += derivatives[k] * abcd[k];
    }

    acc = hf_input[(row, col)] * strength + acc / variance;

    // The masked-fill reconstruction is `lf + acc` with NO clipping to
    // non-negative values. The surrounding (unmasked) pixels are not
    // clipped either (they take the `hf + lf` branch below), so values
    // legitimately go negative; clamping the fill to zero was flooring
    // the negative half of its distribution and manufacturing a
    // homogeneous constant ring. With the clamp removed and `lf`
    // spanning the mask (v1.1), the operator reproduces the nearby
    // structure/multi-scale content. The general (mask=None) diffusion
    // keeps the legacy clamp to preserve its existing behavior.
    let v = lf_input[(row, col)] + acc;
    if clip {
        v.max(T::default())
    } else {
        v
    }
}

#[inline]
pub(super) fn compute_anisotropy_factor<T: NdFloat + Default>(user_param: T) -> T {
    user_param.powi(2)
}

#[inline]
pub(super) fn check_isotropy_mode<T: NdFloat + Default>(anisotropy: T) -> IsotropyType {
    if anisotropy == T::default() {
        IsotropyType::Isotrope
    } else if anisotropy > T::default() {
        IsotropyType::Isophote
    } else {
        IsotropyType::Gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::rgb::rgb_diffusion::regions::build_scoped_regions;
    use crate::test_utils::assert_delta;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_find_gradients_flat() {
        let pixels = Flat3Matrix([0.0; 9]);
        let grad = find_gradients(pixels);
        assert_delta!(grad[0], 0.0, 1e-10);
        assert_delta!(grad[1], 0.0, 1e-10);
    }

    #[test]
    fn test_find_gradients_slope_x() {
        let pixels = Flat3Matrix([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let grad = find_gradients(pixels);
        // Gradient in x direction (columns 3,4,5)
        assert_delta!(grad[1], 0.0, 1e-10);
    }

    #[test]
    fn test_find_gradients_slope_y() {
        let pixels = Flat3Matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let grad = find_gradients(pixels);
        // Gradient in y direction (rows 0,1,2)
        // pixels[7] - pixels[1] = 1.0 - 0.0 = 1.0
        // grad[0] = 1.0 / 2.0 = 0.5
        assert_delta!(grad[0], 0.5, 1e-10);
    }

    #[test]
    fn test_isotrop_laplacian_sum() {
        let lap = isotrop_laplacian::<f64>();
        let sum: f64 = lap.0.iter().map(|&x| x as f64).sum();
        assert_delta!(sum, 0.0, 1e-10);
    }

    #[test]
    fn test_isotrop_laplacian_center() {
        let lap = isotrop_laplacian::<f64>();
        assert_delta!(lap.0[4], -3.0, 1e-10);
    }

    #[test]
    fn test_compute_anisotropy_factor() {
        assert_delta!(compute_anisotropy_factor(1.0), 1.0, 1e-10);
        assert_delta!(compute_anisotropy_factor(0.0), 0.0, 1e-10);
        assert_delta!(compute_anisotropy_factor(2.0), 4.0, 1e-10);
    }

    #[test]
    fn test_check_isotropy_mode() {
        assert_eq!(check_isotropy_mode(0.0), IsotropyType::Isotrope);
        assert_eq!(check_isotropy_mode(1.0), IsotropyType::Isophote);
        assert_eq!(check_isotropy_mode(-1.0), IsotropyType::Gradient);
    }

    // --- Scoped diffusion tests (Option C, Stage 1) ---

    #[test]
    fn test_heat_pde_scoped_equals_fullscan() {
        // The key golden test: running heat_pde_diffusion with the mask-Some
        // full-scan path (scoped=None) and with the scoped path (scoped=Some)
        // on IDENTICAL pre-seeded buffers must produce bit-for-bit identical
        // output, because both share the same per-pixel worker. The mask
        // combines a border-touching cell, an isolated pixel, and a central
        // blob.
        let (h, w) = (9usize, 7usize);
        let mut mask = Array2::<bool>::from_elem((h, w), false);
        mask[(0, 0)] = true; // border-touching corner
        mask[(8, 6)] = true; // isolated pixel (opposite corner)
        for r in 3..=5 {
            for c in 3..=5 {
                mask[(r, c)] = true; // central blob
            }
        }

        // Deterministic pseudo-random hf/lf buffers.
        let mut hf = Array2::<f64>::zeros((h, w));
        let mut lf = Array2::<f64>::zeros((h, w));
        let mut rng = StdRng::seed_from_u64(7);
        let n = Normal::new(0.0, 1.0).unwrap();
        for i in 0..h {
            for j in 0..w {
                hf[(i, j)] = n.sample(&mut rng);
                lf[(i, j)] = n.sample(&mut rng);
            }
        }

        let anisotropy = [0.5, 0.7, -0.3, 1.2];
        let isotropy_type = [
            IsotropyType::Isotrope,
            IsotropyType::Isophote,
            IsotropyType::Gradient,
            IsotropyType::Isophote,
        ];
        let abcd = [0.0065, -0.25, -0.25, -0.2774];
        let strength = 1.0;
        let the_mask: Option<ArrayView2<bool>> = Some(mask.view());

        // Full-scan masked path. Both buffers are pre-seeded with the identical
        // passthrough `lf + hf` (mirroring real usage where the pre-seed equals
        // the input and `lf + hf == input` at unmasked cells), so far-field
        // unmasked cells hold the same value in both paths.
        let mut preseed = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                preseed[(i, j)] = hf[(i, j)] + lf[(i, j)];
            }
        }
        let mut out_full = preseed.clone();
        heat_pde_diffusion(
            hf.view(),
            lf.view(),
            out_full.view_mut(),
            1,
            anisotropy,
            isotropy_type,
            1.0,
            0.0,
            1.0,
            abcd,
            strength,
            &the_mask,
            None,
            false,
        );

        // Scoped path (mult = 1<<0 = 1).
        let regions = build_scoped_regions(&mask.view(), 1);
        let mut out_scoped = preseed.clone();
        heat_pde_diffusion(
            hf.view(),
            lf.view(),
            out_scoped.view_mut(),
            1,
            anisotropy,
            isotropy_type,
            1.0,
            0.0,
            1.0,
            abcd,
            strength,
            &the_mask,
            Some(&regions.halo[0][..]),
            false,
        );

        // Exact equality (identical expression ordering via the shared worker).
        assert_eq!(out_full, out_scoped, "scoped and full-scan paths diverged");
        // Sanity: the heavy branch actually ran (a masked blob pixel changed).
        assert!((out_full[(4, 4)] - preseed[(4, 4)]).abs() > 1e-12);

        // Far-field unmasked cells stay at their pre-seed value in the scoped
        // path (they are never written), matching the full-scan passthrough.
        assert_eq!(out_scoped[(0, 3)], preseed[(0, 3)]);
        assert_eq!(out_full[(0, 3)], preseed[(0, 3)]);
    }

    #[test]
    fn test_heat_pde_fully_masked_scoped_finite() {
        // A fully-masked image must run without panicking, stay finite, and take
        // the unclipped masked branch (values may legitimately go negative).
        let (h, w) = (8usize, 8usize);
        let mask = Array2::<bool>::from_elem((h, w), true);
        // Smooth gradients guarantee a non-zero variance in every 3x3
        // neighborhood, so the heavy branch stays finite (no div-by-zero).
        let mut hf = Array2::<f64>::zeros((h, w));
        let mut lf = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                hf[(i, j)] = 0.001 * i as f64 + 0.002 * j as f64 + 0.5;
                lf[(i, j)] = 0.003 * i as f64 - 0.001 * j as f64 + 1.0;
            }
        }
        let anisotropy = [0.0, 0.0, 0.0, 2.0];
        let isotropy_type = [IsotropyType::Isotrope; 4];
        let abcd = [0.0, 0.0, 0.0, 1.0];
        let the_mask: Option<ArrayView2<bool>> = Some(mask.view());

        let regions = build_scoped_regions(&mask.view(), 2);
        let mut out = Array2::<f64>::zeros((h, w));
        heat_pde_diffusion(
            hf.view(),
            lf.view(),
            out.view_mut(),
            2,
            anisotropy,
            isotropy_type,
            1.0,
            0.0,
            1.0,
            abcd,
            1.0,
            &the_mask,
            Some(&regions.halo[1][..]),
            false,
        );
        assert!(out.iter().all(|&x| x.is_finite()));
    }
}
