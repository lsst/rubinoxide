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

/// Compute the three independent second-difference "basis" values (V, W, C)
/// of a 3x3 neighborhood `n` in row-major order.
///
/// Every rotation-derived (Isophote/Gradient) direction kernel reduces to a
/// linear combination of these three values, so they are computed once per
/// source field and shared by all four diffusion directions. The values are:
/// * `V = n[3] + n[5] - 2*n[4]` - horizontal second difference
/// * `W = n[1] + n[7] - 2*n[4]` - vertical second difference
/// * `C = n[0] + n[8] - n[2] - n[6]` - diagonal second difference
#[inline]
fn compute_basis<T: NdFloat + Default>(n: &[T; 9]) -> [T; 3] {
    let two = T::from(2.0).unwrap();
    [
        n[3] + n[5] - two * n[4],
        n[1] + n[7] - two * n[4],
        n[0] + n[8] - n[2] - n[6],
    ]
}

/// Derivative from the fixed isotropic Laplacian kernel.
///
/// Edge-preserving and independent of `c2` and the local orientation, so it
/// needs no per-direction kernel construction.
#[inline]
fn isotrope_derivative<T: NdFloat + Default>(n: &[T; 9]) -> T {
    let lap = isotrop_laplacian::<T>();
    let mut acc = T::default();
    for k in 0..9 {
        acc += lap.0[k] * n[k];
    }
    acc
}

/// Evaluate one diffusion direction against a source field's precomputed
/// basis.
///
/// For Isophote/Gradient the 3x3 kernel is entirely determined by the rotation
/// triple `(a00, a11, a01)` (see `build_matrix`), so the derivative collapses
/// to the 3-term dot product `a00*V + a11*W + (a01/2)*C` over the shared
/// basis. For Isotrope it is the fixed Laplacian (no `c2`/orientation use).
#[inline(always)]
fn direction_derivative<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
    isotropy_type: &IsotropyType,
    basis: [T; 3],
    n: &[T; 9],
) -> T {
    match isotropy_type {
        IsotropyType::Isotrope => isotrope_derivative(n),
        IsotropyType::Isophote => {
            let a00 = cos_theta2 + c2 * sin_theta2;
            let a11 = c2 * cos_theta2 + sin_theta2;
            let a01 = (c2 - T::from(1.0).unwrap()) * cos_theta_sin_theta;
            basis[0] * a00 + basis[1] * a11 + basis[2] * (T::from(0.5).unwrap() * a01)
        }
        IsotropyType::Gradient => {
            let a00 = c2 * cos_theta2 + sin_theta2;
            let a11 = cos_theta2 + c2 * sin_theta2;
            let a01 = (T::from(1.0).unwrap() - c2) * cos_theta_sin_theta;
            basis[0] * a00 + basis[1] * a11 + basis[2] * (T::from(0.5).unwrap() * a01)
        }
    }
}

/// Which cells [`heat_pde_diffusion`] iterates over: a scoped slice of linear
/// indices (`idx = r*width + c`) or the whole `h x w` grid in row-major order.
/// Both yield the exact same cell order as the pre-refactor dual loop, so
/// numerics are bit-identical.
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

/// Apply anisotropic diffusion to the chosen cells via [`diffusion_compute`].
///
/// This is the bit-exact heavy worker shared by the full-scan and scoped paths.
/// See the module landing doc in `mod.rs` for the heat-PDE model and the
/// four-derivative breakdown driving the four directions.
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

    // Normalize the gradient to a unit direction, branchlessly selecting the
    // (1, 0) fallback for a zero gradient: `s` is 1.0 when the magnitude is
    // non-zero, and `safe` is the magnitude (or 1.0 when it is zero, avoiding
    // a 0/0 NaN). The result is bit-identical to the former `if`/`else`.
    let s = T::from((magnitude_grad != T::default()) as u8).unwrap();
    let safe = s * magnitude_grad + (T::from(1.0).unwrap() - s);
    gradient[0] = s * (gradient[0] / safe) + (T::from(1.0).unwrap() - s);
    gradient[1] = s * (gradient[1] / safe);
    let cos_theta_grad_sq = gradient[0].powi(2);
    let sin_theta_grad_sq = gradient[1].powi(2);
    let cos_theta_sin_theta_grad = gradient[0] * gradient[1];

    let magnitude_lapl = (laplace[0].powi(2) + laplace[1].powi(2)).sqrt();
    c2[1] = -magnitude_lapl * anisotropy[1];
    c2[3] = -magnitude_lapl * anisotropy[3];

    let s = T::from((magnitude_lapl != T::default()) as u8).unwrap();
    let safe = s * magnitude_lapl + (T::from(1.0).unwrap() - s);
    laplace[0] = s * (laplace[0] / safe) + (T::from(1.0).unwrap() - s);
    laplace[1] = s * (laplace[1] / safe);

    let cos_theta_lapl_sq = laplace[0].powi(2);
    let sin_theta_lapl_sq = laplace[1].powi(2);
    let cos_theta_sin_theta_lapl = laplace[0] * laplace[1];

    for k in 0..4 {
        // `c2` is only used by Isophote/Gradient directions; an Isotrope
        // direction uses the fixed Laplacian kernel and discards it, so skip
        // the (expensive) exponentiation there. `isotropy_type` is fixed for
        // the whole call, so this branch is decided once, not per cell.
        if isotropy_type[k] != IsotropyType::Isotrope {
            c2[k] = c2[k].exp();
        }
    }

    // All rotation-derived (Isophote/Gradient) direction kernels reduce to a
    // 3-term dot product over these per-source basis values, computed once
    // here and shared by the four directions.
    let basis_lf = compute_basis(&neighbour_pixel_lf);
    let basis_hf = compute_basis(&neighbour_pixel_hf);

    let mut derivatives: [T; 4] = [T::default(); 4];
    derivatives[0] = direction_derivative(
        c2[0],
        cos_theta_sin_theta_grad,
        cos_theta_grad_sq,
        sin_theta_grad_sq,
        &isotropy_type[0],
        basis_lf,
        &neighbour_pixel_lf,
    );
    derivatives[1] = direction_derivative(
        c2[1],
        cos_theta_sin_theta_lapl,
        cos_theta_lapl_sq,
        sin_theta_lapl_sq,
        &isotropy_type[1],
        basis_lf,
        &neighbour_pixel_lf,
    );
    derivatives[2] = direction_derivative(
        c2[2],
        cos_theta_sin_theta_grad,
        cos_theta_grad_sq,
        sin_theta_grad_sq,
        &isotropy_type[2],
        basis_hf,
        &neighbour_pixel_hf,
    );
    derivatives[3] = direction_derivative(
        c2[3],
        cos_theta_sin_theta_lapl,
        cos_theta_lapl_sq,
        sin_theta_lapl_sq,
        &isotropy_type[3],
        basis_hf,
        &neighbour_pixel_hf,
    );

    let mut variance = T::default();
    for k in 0..9 {
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
    fn test_find_gradients() {
        // Flat image: zero gradient.
        let grad = find_gradients(Flat3Matrix([0.0; 9]));
        assert_delta!(grad[0], 0.0, 1e-10);
        assert_delta!(grad[1], 0.0, 1e-10);
        // Horizontal slope (columns 3,4,5): vertical gradient is zero.
        let grad = find_gradients(Flat3Matrix([
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ]));
        assert_delta!(grad[1], 0.0, 1e-10);
        // Vertical slope (rows 0,1,2): pixels[7]-pixels[1] = 1.0, so grad[0] = 0.5.
        let grad = find_gradients(Flat3Matrix([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]));
        assert_delta!(grad[0], 0.5, 1e-10);
    }

    #[test]
    fn test_isotrop_laplacian() {
        let lap = isotrop_laplacian::<f64>();
        // A Laplacian kernel should sum to zero, with a negative center.
        let sum: f64 = lap.0.iter().map(|&x| x as f64).sum();
        assert_delta!(sum, 0.0, 1e-10);
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

    #[test]
    fn test_direction_derivative_matches_nine_tap() {
        // The factored form `a00*V + a11*W + (a01/2)*C` must agree (to f64
        // rounding) with the original explicit 9-tap kernel application.
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(2024);
        for _ in 0..500 {
            let mut n = [0.0_f64; 9];
            for e in n.iter_mut() {
                *e = rng.gen_range(-2.0..2.0);
            }
            let c2 = rng.gen_range(0.05..3.0);
            let cos_sin = rng.gen_range(-1.0..1.0);
            let cos2 = rng.gen_range(0.0..1.0);
            let sin2 = rng.gen_range(0.0..1.0);
            // Recompute direction_derivative and the direct 9-tap dot and
            // compare for both Isophote and Gradient.
            for iso in [IsotropyType::Isophote, IsotropyType::Gradient] {
                let basis = compute_basis(&n);
                let factored = direction_derivative(c2, cos_sin, cos2, sin2, &iso, basis, &n);
                let (a00, a11, a01) = match iso {
                    IsotropyType::Isophote => (
                        cos2 + c2 * sin2,
                        c2 * cos2 + sin2,
                        (c2 - 1.0) * cos_sin,
                    ),
                    IsotropyType::Gradient => (
                        c2 * cos2 + sin2,
                        cos2 + c2 * sin2,
                        (1.0 - c2) * cos_sin,
                    ),
                    _ => unreachable!(),
                };
                let half = a01 / 2.0;
                let kernel = [
                    half, a11, -half, a00, -2.0 * (a00 + a11), a00, -half, a11, half,
                ];
                let direct: f64 = kernel.iter().zip(n.iter()).map(|(k, &v)| k * v).sum();
                let delta = (factored - direct).abs();
                let scale = 1.0 + direct.abs();
                assert!(delta < 1e-9 * scale, "iso={iso:?} factored={factored} direct={direct}");
            }
        }
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
