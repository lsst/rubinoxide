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
extern crate openblas_src;
use log;
use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, NdFloat};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::rgb::color_spaces::{linear_rgb_to_oklab, oklab_to_linear_rgb};
use rand;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::cmp;
use std::collections::VecDeque;

const MAX_NUM_SCALES: usize = 10;
const B_SPLINE_SIGMA: f64 = 2.0553651328015339;
const H: usize = 1;
const KAPPA: f64 = 0.25;

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
enum IsotropyType {
    Isotrope,
    Isophote,
    Gradient,
}

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
///
/// # Notes
/// The diffusion coefficient computation: c = exp(-|∇u| * anisotropy)
/// ensures edge preservation: high gradient = low diffusion.
fn heat_pde_diffusion<T: NdFloat + Default>(
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
) {
    let mut output = output;
    let regularization_factor = regularization * current_radius_sq / T::from(9.0).unwrap();
    let mut i_neighbours: [usize; 3] = [0, 0, 0];
    let mut j_neighbours: [usize; 3] = [0, 0, 0];

    let mut neighbour_pixel_hf = [T::default(); 9];
    let mut neighbour_pixel_lf = [T::default(); 9];

    let mut c2 = [T::default(); 4];

    let (height, width) = output.dim();

    // for row in &process_points.0 {
    for row in 0..height {
        i_neighbours[0] = (cmp::max(row as i32 - (mult * H) as i32, 0) as i32) as usize;
        i_neighbours[1] = row;
        i_neighbours[2] = cmp::min((row + mult * H) as i32, height as i32 - 1) as usize;
        // for col in &process_points.1 {
        for col in 0..width {
            j_neighbours[0] = cmp::max(col as i32 - (mult * H) as i32, 0) as usize;
            j_neighbours[1] = col;
            j_neighbours[2] = cmp::min((col + mult * H) as i32, width as i32 - 1) as usize;

            let do_pixel = match mask {
                Some(m) => m[(row, col)],
                None => true,
            };

            if do_pixel {
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
                output[(row, col)] = if mask.is_some() {
                    v
                } else {
                    v.max(T::default())
                };
            } else {
                output[(row, col)] = hf_input[(row, col)] + lf_input[(row, col)];
            }
        }
    }
}

#[inline]
fn compute_anisotropy_factor<T: NdFloat + Default>(user_param: T) -> T {
    user_param.powi(2)
}

#[inline]
fn check_isotropy_mode<T: NdFloat + Default>(anisotropy: T) -> IsotropyType {
    if anisotropy == T::default() {
        IsotropyType::Isotrope
    } else if anisotropy > T::default() {
        IsotropyType::Isophote
    } else {
        IsotropyType::Gradient
    }
}

/// Perform vertical B-spline convolution pass on image
///
/// Applies a 5-tap binomial filter [1,4,6,4,1]/16 to convolve the image
/// vertically at a multi-scale level determined by `mult`. The results
/// are written to the output buffer.
///
/// The B-spline filter approximates Gaussian convolution with:
/// [1/16, 4/16, 6/16, 4/16, 1/16]
///
/// Every pixel (masked and unmasked) contributes to the low-pass, so `lf` is a
/// smooth, boundary-continuous field that spans the mask region.
///
/// # Arguments
/// * `in_array` - Input image array
/// * `row` - Current row being processed
/// * `width` - Image width
/// * `height` - Image height
/// * `mult` - Multiplier for filter support (1<<scale level)
/// * `clip_negatives` - If true, clamp negative results to zero
/// * `out_buf` - Output buffer (length = width), receives filtered row
#[inline]
fn _bspline_vertical_pass<T: NdFloat + Default>(
    in_array: ArrayViewMut2<T>,
    row: usize,
    width: usize,
    height: usize,
    mult: i32,
    clip_negatives: bool,
    out_buf: &mut [T],
) {
    let irow = row as i32;
    let indicies: [usize; 5] = [
        cmp::max(irow - 2 * mult, 0) as usize,
        cmp::max(irow - mult, 0) as usize,
        row,
        cmp::min((irow + mult) as usize, height - 1),
        cmp::min((irow + 2 * mult) as usize, height - 1),
    ];

    let filter: [T; 5] = [
        T::from(1.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(6.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(1.0 / 16.0).unwrap(),
    ];

    for index in 0..width {
        // The 5-tap filter weights sum to 1, so the weighted sum is the mean.
        let val_sum = filter.iter().enumerate().fold(T::default(), |acc, (k, &f)| {
            acc + in_array[(indicies[k], index)] * f
        });
        out_buf[index] = if clip_negatives {
            val_sum.max(T::default())
        } else {
            val_sum
        };
    }
}

/// Perform horizontal B-spline convolution on a 1D slice
///
/// Applies a 5-tap binomial filter to convolve horizontally.
/// Complements `_bspline_vertical_pass` for 2D decomposition.
///
/// # Arguments
/// * `in_slice` - 1D input array (row to filter)
/// * `col` - Current column position
/// * `width` - Array width
/// * `mult` - Multiplier for filter support (1<<scale)
/// * `clip_negatives` - If true, clamp negative results to zero
///
/// # Returns
/// * `T` - Filtered value at column position
#[inline]
fn _bspline_horizontal<T: NdFloat + Default>(
    in_slice: &[T],
    col: usize,
    width: usize,
    mult: i32,
    clip_negatives: bool,
) -> T {
    let icol = col as i32;
    let indicies: [usize; 5] = [
        cmp::max(icol - 2 * mult, 0) as usize,
        cmp::max(icol - mult, 0) as usize,
        col,
        cmp::min((icol + mult) as usize, width - 1),
        cmp::min((icol + 2 * mult) as usize, width - 1),
    ];

    let filter: [T; 5] = [
        T::from(1.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(6.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(1.0 / 16.0).unwrap(),
    ];

    // The 5-tap filter weights sum to 1, so the weighted sum is the mean.
    let val_sum = filter.iter().enumerate().fold(T::default(), |acc, (k, &f)| {
        acc + in_slice[indicies[k]] * f
    });
    if clip_negatives {
        val_sum.max(T::default())
    } else {
        val_sum
    }
}

/// Decompose image into high/low frequency components using 2D B-spline
///
/// Performs separable B-spline wavelet decomposition by applying
/// vertical then horizontal passes. Produces high-frequency (detail)
/// and low-frequency (approximation) components.
///
/// The low-pass includes every pixel (masked and unmasked), so `lf` is a smooth
/// field spanning the mask region with the (filled) values rather than going to
/// zero deep inside the mask.
///
/// # Arguments
/// * `in_array` - Input image (modified in-place for efficiency)
/// * `hf` - High-frequency output array (details)
/// * `lf` - Low-frequency output array (approximation)
/// * `width` - Image width
/// * `height` - Image height
/// * `mult` - Scale multiplier (1<<scale)
/// * `row_buf` - Reusable buffer for vertical pass results
#[inline]
fn decompose_2d_bspline<T: NdFloat + Default>(
    in_array: ArrayViewMut2<T>,
    hf: ArrayViewMut2<T>,
    lf: ArrayViewMut2<T>,
    width: usize,
    height: usize,
    mult: i32,
    row_buf: &mut [T],
) {
    let mut hf = hf;
    let mut lf = lf;
    let mut in_array = in_array;
    for row in 0..height {
        _bspline_vertical_pass(in_array.view_mut(), row, width, height, mult, true, row_buf);
        for col in 0..width {
            let blur = _bspline_horizontal(row_buf, col, width, mult, true);
            let index = (row, col);
            lf[index] = blur;
            hf[index] = in_array[index] - blur;
        }
    }
}

/// Compute equivalent standard deviation at wavelet decomposition step
///
/// Calculates the cumulative Gaussian width after `s` steps of B-spline
/// decomposition. Each step doubles the effective scale.
///
/// # Arguments
/// * `sigma` - Base standard deviation (B_SPLINE_SIGMA)
/// * `s` - Decomposition step (0 = base scale)
///
/// # Returns
/// * `T` - Cumulative equivalent sigma at step s
#[inline]
fn equivalent_sigma_at_step<T: NdFloat + Default>(sigma: T, s: usize) -> T {
    if s == 0 {
        T::from(sigma).unwrap()
    } else {
        ((equivalent_sigma_at_step(sigma, s - 1)).powi(2)
            + (T::from(s).unwrap().exp2() * sigma).powi(2))
        .sqrt()
    }
}

/// Calculate number of wavelet decomposition steps for target sigma
///
/// Determines how many B-spline decomposition levels are needed
/// to achieve a specified effective smoothing scale.
///
/// The accumulated effective sigma follows the same squared-additive semantics
/// as [`equivalent_sigma_at_step`]: each step adds the *square* `(2^s * sigma)^2`
/// of the additional scale (not the linear `2^s * sigma`). Keeping the two
/// functions consistent means the same `s` steps produce the same equivalent
/// sigma in both.
///
/// # Arguments
/// * `sigma_filter` - Base filter standard deviation
/// * `sigma_final` - Target equivalent sigma
///
/// # Returns
/// * `usize` - Number of decomposition steps required
#[inline]
fn num_steps_to_reach_equivalent_sigma<T: NdFloat + Default>(
    sigma_filter: T,
    sigma_final: T,
) -> usize {
    let mut s: usize = 0;
    let mut radius = sigma_filter;
    while radius < sigma_final {
        s += 1;
        radius = (radius.powi(2) + (T::from(1 << s).unwrap() * sigma_filter).powi(2)).sqrt();
    }
    s + 1
}
/// Process image through multi-scale wavelet decomposition and diffusion
///
/// Orchestrates the complete diffusion pipeline:
/// 1. Decomposes image into high/low frequency components at each scale
/// 2. Applies anisotropic diffusion to high-frequency components
/// 3. Reconstructs image by combining processed components
///
/// Uses ping-pong buffering between `lf_odd` and `lf_even` arrays to
/// avoid excessive allocations during multi-scale decomposition.
///
/// # Arguments
/// * `process_args` - All diffusion algorithm parameters
/// * `scales` - Number of wavelet decomposition levels
/// * `input` - Input image, modified during processing
/// * `reconstructed` - Final output image
/// * `lf_odd` - Low-frequency buffer for odd scales
/// * `lf_even` - Low-frequency buffer for even scales
/// * `hf` - High-frequency components for each scale
/// * `zoom` - Scaling factor for radius computation
/// * `mask` - Optional boolean mask for selective processing
/// * `row_buf` - Pre-allocated buffer for B-spline passes
fn wavelets_process<T: NdFloat + Default>(
    process_args: &ProcessArgs<T>,
    scales: usize,
    input: &mut ArrayViewMut2<T>,
    reconstructed: &mut ArrayViewMut2<T>,
    lf_odd: &mut Array2<T>,
    lf_even: &mut Array2<T>,
    hf: &mut Vec<Array2<T>>,
    zoom: T,
    mask: &Option<ArrayView2<bool>>,
    row_buf: &mut [T],
) {
    let anisotropy = [
        compute_anisotropy_factor(process_args.anisotropy_first),
        compute_anisotropy_factor(process_args.anisotropy_second),
        compute_anisotropy_factor(process_args.anisotropy_third),
        compute_anisotropy_factor(process_args.anisotropy_fourth),
    ];

    let isotropy_type = [
        check_isotropy_mode(process_args.anisotropy_first),
        check_isotropy_mode(process_args.anisotropy_second),
        check_isotropy_mode(process_args.anisotropy_third),
        check_isotropy_mode(process_args.anisotropy_fourth),
    ];

    let regularization =
        T::from(10.0).unwrap().powf(process_args.regularization) - T::from(1.0).unwrap();

    let variance_threshold = T::from(10.0).unwrap().powf(process_args.variance_threshold);

    let mut buffer_in: ArrayViewMut2<T>;
    let mut buffer_out: ArrayViewMut2<T>;
    let residual: &mut Array2<T>;
    let temp: &mut Array2<T>;

    let (height, width) = input.dim();

    let mut final_scale: usize = 0;
    for sc in 0..scales {
        let mult = 1 << sc;

        if sc == 0 {
            buffer_in = input.view_mut();
            buffer_out = lf_odd.view_mut();
        } else if (sc % 2) != 0 {
            buffer_in = lf_odd.view_mut();
            buffer_out = lf_even.view_mut();
        } else {
            buffer_in = lf_even.view_mut();
            buffer_out = lf_odd.view_mut();
        }

        decompose_2d_bspline(
            buffer_in,
            hf[sc].view_mut(),
            buffer_out,
            width,
            height,
            mult,
            row_buf,
        );

        final_scale = sc;
    }

    if final_scale == 0 {
        residual = lf_odd;
        temp = lf_even;
    } else if (final_scale % 2) != 0 {
        residual = lf_even;
        temp = lf_odd;
    } else {
        residual = lf_odd;
        temp = lf_even;
    }

    let kappa = T::from(KAPPA).unwrap();

    let mut buffer_in: ArrayView2<T>;
    for (count, scale) in (0..=scales - 1).rev().enumerate() {
        let mult = 1 << scale;
        let current_radius = equivalent_sigma_at_step(T::from(B_SPLINE_SIGMA).unwrap(), scale);
        let real_radius = current_radius * zoom;
        let norm =
            (-(real_radius - process_args.radius_center).powi(2) / process_args.radius).exp();
        let abcd = [
            process_args.first * kappa * norm,
            process_args.second * kappa * norm,
            process_args.third * kappa * norm,
            process_args.fourth * kappa * norm,
        ];
        let strength = process_args.sharpness * norm + T::from(1.0).unwrap();

        if count == 0 {
            buffer_in = residual.view();
            buffer_out = temp.view_mut();
        } else if (count % 2) != 0 {
            buffer_in = temp.view();
            buffer_out = residual.view_mut();
        } else {
            buffer_in = residual.view();
            buffer_out = temp.view_mut();
        }

        if scale == 0 {
            buffer_out = reconstructed.view_mut();
        }

        heat_pde_diffusion(
            hf[scale].view(),
            buffer_in,
            buffer_out,
            mult,
            anisotropy,
            isotropy_type,
            variance_threshold,
            regularization,
            current_radius.powi(2),
            abcd,
            strength,
            mask,
        );
    }
}

/// Parameters for diffusion algorithm configuration
///
/// Encapsulates all tunable parameters for the anisotropic diffusion
/// algorithm, including anisotropy weights, diffusion coefficients,
/// and scale parameters.
struct ProcessArgs<T: NdFloat + Default> {
    iterations: usize,
    anisotropy_first: T,
    anisotropy_second: T,
    anisotropy_third: T,
    anisotropy_fourth: T,
    regularization: T,
    variance_threshold: T,
    radius_center: T,
    radius: T,
    first: T,
    second: T,
    third: T,
    fourth: T,
    sharpness: T,
}

/// Apply complete anisotropic diffusion pipeline to image
///
/// This is the top-level function that processes an image through
/// multiple iterations of wavelet decomposition and diffusion:
///
/// 1. Allocates working buffers (temp arrays, low-freq buffers, high-freq buffers)
/// 2. Computes required wavelet decomposition scales based on radius
/// 3. Iterates diffusion process, alternating between buffers
/// 4. Returns final diffused image
///
/// # Arguments
/// * `process_args` - Complete diffusion configuration
/// * `image_in` - Input image (modified during processing)
/// * `mask` - Optional boolean mask for selective pixel processing
///
/// # Returns
/// * `Array2<T>` - Diffused image
///
/// # Notes
/// The number of iterations is clamped to minimum 1 to ensure
/// at least one diffusion pass is always performed.
fn process_image<T: NdFloat + Default>(
    process_args: ProcessArgs<T>,
    image_in: &mut ArrayViewMut2<T>,
    mask: Option<ArrayView2<bool>>,
) -> Array2<T> {
    let mut image_out = Array2::<T>::zeros(image_in.dim());
    let mut temp_1 = Array2::<T>::zeros(image_in.dim());
    let mut temp_2 = Array2::<T>::zeros(image_in.dim());
    let mut lf_odd = Array2::<T>::zeros(image_in.dim());
    let mut lf_even = Array2::<T>::zeros(image_in.dim());

    let final_radius = process_args.radius + process_args.radius_center * T::from(2.0).unwrap();

    let iterations = cmp::max(process_args.iterations, 1);
    let diffusion_scales =
        num_steps_to_reach_equivalent_sigma(T::from(B_SPLINE_SIGMA).unwrap(), final_radius);
    let scales = diffusion_scales.clamp(1, MAX_NUM_SCALES);

    let mut hf = (0..scales)
        .map(|_| Array2::<T>::zeros(image_in.dim()))
        .collect::<Vec<_>>();

    let zoom = T::from(1.0).unwrap();

    // Get image dimensions for buffer allocation
    let (_, width) = image_in.dim();

    // Pre-allocate buffer for B-spline vertical pass to avoid heap allocations
    let mut row_buf = vec![T::default(); width];

    let temp_1_ref = &mut temp_1.view_mut();
    let temp_2_ref = &mut temp_2.view_mut();
    let mut temp_out: &mut ArrayViewMut2<T>;
    let image_out_ref = &mut image_out.view_mut();
    for it in 0..iterations {
        if it == 0 {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_2_ref;
            }
            wavelets_process(
                &process_args,
                scales,
                image_in,
                temp_out,
                &mut lf_odd,
                &mut lf_even,
                &mut hf,
                zoom,
                &mask,
                &mut row_buf,
            );
        } else if (it % 2) == 0 {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_2_ref;
            }
            wavelets_process(
                &process_args,
                scales,
                temp_1_ref,
                temp_out,
                &mut lf_odd,
                &mut lf_even,
                &mut hf,
                zoom,
                &mask,
                &mut row_buf,
            );
        } else {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_1_ref;
            }
            wavelets_process(
                &process_args,
                scales,
                temp_2_ref,
                temp_out,
                &mut lf_odd,
                &mut lf_even,
                &mut hf,
                zoom,
                &mask,
                &mut row_buf,
            );
        }
    }
    image_out
}

/// Apply anisotropic diffusion to a grayscale image using multi-scale
/// B-spline wavelet decomposition and PDE-based diffusion.
///
/// This function implements the heat equation with variable diffusion
/// coefficients controlled by image structure tensors:
///
///     ∂u/∂t = ∇·(c(x,y,∇u)∇u)
///
/// where c(x,y,∇u) = exp(-|∇u| * anisotropy) is the diffusion coefficient.
/// High gradient regions (edges) have low diffusion coefficients, preserving
/// edges while smoothing flat regions.
///
/// The algorithm uses a four-derivative approach to capture directional
/// information in both gradient and Laplacian domains, providing superior
/// edge preservation compared to isotropic diffusion.
///
/// Parameters
/// ----------
/// image : `NDArray`
///     Input grayscale image as 2D numpy array of dtype float64.
/// iterations : `int`, optional
///     Number of diffusion iterations. Higher values produce more
///     diffusion. Default is 3.
/// anisotropy_first : `float`, optional
///     Diffusion strength for gradient-based first directional derivative.
///     Values > 0 enable edge-preserving anisotropic diffusion.
///     Default is 1.0.
/// anisotropy_second : `float`, optional
///     Diffusion strength for Laplacian-based second directional derivative.
///     Default is 1.0.
/// anisotropy_third : `float`, optional
///     Diffusion strength for third directional derivative term.
///     Default is 1.0.
/// anisotropy_fourth : `float`, optional
///     Diffusion strength for fourth directional derivative term.
///     Default is 1.0.
/// regularization : `float`, optional
///     Regularization parameter for numerical stability. Controls minimum
///     variance threshold. Default is 2.94.
/// variance_threshold : `float`, optional
///     Minimum variance threshold added to diffusion computation.
///     Prevents division by zero. Default is 0.0.
/// radius_center : `float`, optional
///     Center radius for diffusion weighting. Determines scale of diffusion
///     effects. Default is 0.0.
/// first : `float`, optional
///     First diffusion coefficient (weighted by position). Default is 0.0065.
/// second : `float`, optional
///     Second diffusion coefficient. Default is -0.25.
/// third : `float`, optional
///     Third diffusion coefficient. Default is -0.25.
/// fourth : `float`, optional
///     Fourth diffusion coefficient. Default is -0.2774.
/// radius : `float`, optional
///     Diffusion radius parameter. Larger values increase diffusion scale.
///     Default is 5.0.
/// sharpness : `float`, optional
///     Sharpness enhancement parameter. Positive values preserve peaks,
///     negative values smooth them. Default is 0.0.
///
/// Returns
/// -------
/// results : `NDArray`
///     Diffused image as 2D numpy array of dtype float64 with same shape
///     as input.
///
/// Raises
/// ------
/// `ValueError`
///     If image dimensions are not positive.
///     If image contains NaN or infinite values.
///
/// Notes
/// -----
/// The B-spline wavelet decomposition uses a 5-tap binomial filter
/// approximating Gaussian convolution:
///
///     [1/16, 4/16, 6/16, 4/16, 1/16]
///
/// This provides multi-scale analysis where high-frequency components
/// capture details and low-frequency components capture smooth variations.
///
/// The diffusion coefficient computation from structure tensor eigenvalues
/// ensures edge preservation:
///
///     c = exp(-|∇u| * anisotropy)
///
/// See Also
/// --------
/// inpaint_mask : Inpaint masked regions using diffusion
#[pyfunction]
#[pyo3(signature = (image,
    iterations= 3,
    anisotropy_first= 1.0,
    anisotropy_second= 1.0,
    anisotropy_third= 1.0,
    anisotropy_fourth= 1.0,
    regularization= 2.94,
    variance_threshold= 0.0,
    radius_center= 0.0,
    first= 0.0065,
    second= -0.25,
    third= -0.25,
    fourth= -0.2774,
    radius= 5.0,
    sharpness= 0.0,
))]
pub fn diffuse_gray_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f64>,
    iterations: usize,
    anisotropy_first: f64,
    anisotropy_second: f64,
    anisotropy_third: f64,
    anisotropy_fourth: f64,
    regularization: f64,
    variance_threshold: f64,
    radius_center: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
    radius: f64,
    sharpness: f64,
) -> Bound<'py, PyArray2<f64>> {
    // `image` is a read-only numpy array. Copy it into a mutable local Array2
    // and run the diffusion on that copy, so we never mutate a
    // PyReadonlyArray2 through `unsafe`.
    let mut array = image.as_array().to_owned();

    let process_args = ProcessArgs {
        iterations,
        anisotropy_first,
        anisotropy_second,
        anisotropy_third,
        anisotropy_fourth,
        regularization,
        variance_threshold,
        radius_center,
        first,
        second,
        third,
        fourth,
        radius,
        sharpness,
    };
    let result = process_image(process_args, &mut array.view_mut(), None);
    result.to_pyarray(py)
}

/// Replace masked pixels with Gaussian noise for inpainting initialization
///
/// Substitutes masked region pixels with values sampled from a Gaussian
/// distribution centered at the original pixel value with standard
/// deviation equal to the original value. This provides a stochastic
/// starting point for subsequent diffusion-based inpainting.
///
/// # Arguments
/// * `image` - Input image (contains original pixel values)
/// * `mask` - Boolean mask indicating pixels to replace (True = replace)
///
/// # Returns
/// * `Array2<T>` - Image with masked pixels replaced by noise
///
/// # Panics
/// Panics if any masked pixel has value <= 0, as this would make
/// the standard deviation non-positive for `Normal::new()`. Callers using
/// `init_method="noise"` should validate via [`validate_noise_masked`] (which
/// raises a `PyValueError`) before calling this.
fn replace_masked_with_noise<T: NdFloat + Default>(
    image: ArrayView2<T>,
    mask: &ArrayView2<bool>,
    random_seed: Option<u64>,
) -> Array2<T>
where
    StandardNormal: Distribution<T>,
{
    let im_dim = image.dim();
    let mut result = Array2::<T>::zeros(im_dim);
    for i in 0..im_dim.0 {
        for j in 0..im_dim.1 {
            let val = image[(i, j)];
            result[(i, j)] = if mask[(i, j)] {
                let normal = Normal::new(val, val).unwrap();

                let mut rng = match random_seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => StdRng::from_rng(&mut rand::rng()),
                };
                normal.sample(&mut rng)
            } else {
                image[(i, j)]
            }
        }
    }
    result
}

/// A connected component of masked pixels, together with its bounding box.
struct Component {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    coords: Vec<(usize, usize)>,
}

/// 8-connected neighbour offsets (Chebyshev neighbourhood), shared by the
/// component flood-fill and the nearest-value BFS.
const NEIGHBORS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Multi-source 8-connected BFS that fills every cell whose `dist == u32::MAX`
/// with the value and (Chebyshev) distance of its nearest seed cell.
///
/// Callers pre-initialize `dist[(i, j)] = 0` and `values[(i, j)]` for the seed
/// cells (e.g. unmasked pixels), then call this once to propagate the seed
/// values inward. Cells with no reachable seed keep `dist == u32::MAX`.
fn bfs_nearest_value<T: NdFloat + Default>(values: &mut Array2<T>, dist: &mut Array2<u32>) {
    let (h, w) = dist.dim();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for i in 0..h {
        for j in 0..w {
            if dist[(i, j)] == 0 {
                queue.push_back((i, j));
            }
        }
    }
    while let Some((ci, cj)) = queue.pop_front() {
        let d = dist[(ci, cj)];
        for (di, dj) in NEIGHBORS {
            let ni = ci as i32 + di;
            let nj = cj as i32 + dj;
            if ni < 0 || nj < 0 || ni >= h as i32 || nj >= w as i32 {
                continue;
            }
            let (ni, nj) = (ni as usize, nj as usize);
            if dist[(ni, nj)] == u32::MAX {
                dist[(ni, nj)] = d + 1;
                values[(ni, nj)] = values[(ci, cj)];
                queue.push_back((ni, nj));
            }
        }
    }
}

/// Re-initialize a nearest-value BFS distance map for a fresh propagation.
///
/// Resets every cell to `u32::MAX` and marks each unmasked (seed) cell with
/// distance 0, so a subsequent [`bfs_nearest_value`] starts from the same
/// distance semantics every time. Call this before each channel's BFS so a
/// previously populated `dist` is never reused as if it were unseeded (which
/// would leave later channels unwarmed because their masked cells no longer
/// hold `u32::MAX`).
fn seed_nearest_dist(dist: &mut Array2<u32>, mask: &ArrayView2<bool>) {
    let (h, w) = mask.dim();
    dist.fill(u32::MAX);
    for i in 0..h {
        for j in 0..w {
            if !mask[(i, j)] {
                dist[(i, j)] = 0;
            }
        }
    }
}

/// Find all 8-connected components of masked (True) pixels.
///
/// Returns one `Component` per connected masked region including its tight
/// bounding box. This reads the whole mask (O(H*W)) but stores only the masked
/// pixel coordinates, so downstream work can be scoped to each component.
fn find_components(mask: &ArrayView2<bool>) -> Vec<Component> {
    let (h, w) = mask.dim();
    let mut visited = Array2::<bool>::from_elem((h, w), false);
    let mut comps = Vec::new();
    for i in 0..h {
        for j in 0..w {
            if !mask[(i, j)] || visited[(i, j)] {
                continue;
            }
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            let mut coords: Vec<(usize, usize)> = Vec::new();
            let mut r0 = i;
            let mut r1 = i;
            let mut c0 = j;
            let mut c1 = j;
            visited[(i, j)] = true;
            queue.push_back((i, j));
            while let Some((ci, cj)) = queue.pop_front() {
                coords.push((ci, cj));
                r0 = r0.min(ci);
                r1 = r1.max(ci);
                c0 = c0.min(cj);
                c1 = c1.max(cj);
                for (di, dj) in NEIGHBORS {
                    let ni = ci as i32 + di;
                    let nj = cj as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= h as i32 || nj >= w as i32 {
                        continue;
                    }
                    let (ni, nj) = (ni as usize, nj as usize);
                    if mask[(ni, nj)] && !visited[(ni, nj)] {
                        visited[(ni, nj)] = true;
                        queue.push_back((ni, nj));
                    }
                }
            }
            comps.push(Component {
                r0,
                r1,
                c0,
                c1,
                coords,
            });
        }
    }
    comps
}

/// Fill the masked pixels of `comp` in `result`.
///
/// Builds summed-area tables (count/sum/sum-of-squares) and a multi-source BFS
/// of the nearest unmasked value only over the component's bounding box padded
/// by `radius`. Each masked pixel's mean and std come from the equal-weight
/// window statistics; a window containing no unmasked neighbor falls back to
/// the BFS nearest value. Cost scales with the component's bounding box, not
/// the full image, so many small regions stay cheap.
fn fill_component<T: NdFloat + Default>(
    result: &mut Array2<T>,
    image: ArrayView2<T>,
    mask: &ArrayView2<bool>,
    comp: &Component,
    radius: usize,
    rng: &mut StdRng,
) where
    StandardNormal: Distribution<T>,
{
    let (h, w) = image.dim();
    let eps = T::from(1e-6).unwrap();

    let sr0 = comp.r0.saturating_sub(radius);
    let sr1 = (comp.r1 + radius).min(h - 1);
    let sc0 = comp.c0.saturating_sub(radius);
    let sc1 = (comp.c1 + radius).min(w - 1);
    let sh = sr1 - sr0 + 1;
    let sw = sc1 - sc0 + 1;

    // Summed-area tables over valid pixels only in the padded bbox.
    let mut cnt = Array2::<i64>::zeros((sh + 1, sw + 1));
    let mut sum = Array2::<T>::zeros((sh + 1, sw + 1));
    let mut sq = Array2::<T>::zeros((sh + 1, sw + 1));
    for li in 0..sh {
        for lj in 0..sw {
            let gi = sr0 + li;
            let gj = sc0 + lj;
            let m = mask[(gi, gj)];
            let v = if m { T::default() } else { image[(gi, gj)] };
            cnt[(li + 1, lj + 1)] =
                cnt[(li, lj + 1)] + cnt[(li + 1, lj)] - cnt[(li, lj)] + if m { 0 } else { 1 };
            sum[(li + 1, lj + 1)] =
                sum[(li, lj + 1)] + sum[(li + 1, lj)] - sum[(li, lj)] + v;
            sq[(li + 1, lj + 1)] =
                sq[(li, lj + 1)] + sq[(li + 1, lj)] - sq[(li, lj)] + v * v;
        }
    }

    // Multi-source BFS of the nearest unmasked value over the padded bbox.
    let mut nearest = Array2::<T>::from_elem((sh, sw), T::default());
    let mut dist = Array2::<u32>::from_elem((sh, sw), u32::MAX);
    for li in 0..sh {
        for lj in 0..sw {
            let gi = sr0 + li;
            let gj = sc0 + lj;
            if !mask[(gi, gj)] {
                nearest[(li, lj)] = image[(gi, gj)];
                dist[(li, lj)] = 0;
            }
        }
    }
    bfs_nearest_value(&mut nearest, &mut dist);

    // Edge blend: scale down the added noise only for the outermost few pixels
    // so the fill blends seamlessly into the surroundings, ramping to full noise
    // quickly away from the edge.
    let ramp_dist = T::from(3.0).unwrap();

    // The per-local-window standard deviation is inflated near the mask boundary
    // where the window holds few valid unmasked samples, which would inject too
    // much noise there. Instead fill every masked pixel with the local mean plus
    // a noise term whose magnitude is a single robust (median) standard
    // deviation for the component, so the fill reproduces the background's
    // natural per-pixel noise character throughout.
    let n = comp.coords.len();
    let mut means = Vec::<T>::with_capacity(n);
    let mut sigmas = Vec::<T>::with_capacity(n);
    for &(i, j) in &comp.coords {
        let li = i - sr0;
        let lj = j - sc0;
        let lr0 = li.saturating_sub(radius);
        let lr1 = (li + radius).min(sh - 1);
        let lc0 = lj.saturating_sub(radius);
        let lc1 = (lj + radius).min(sw - 1);
        let (i0, i1) = (lr0, lr1 + 1);
        let (j0, j1) = (lc0, lc1 + 1);
        let ncnt = cnt[(i1, j1)] - cnt[(i0, j1)] - cnt[(i1, j0)] + cnt[(i0, j0)];
        let (mean, sigma) = if ncnt > 0 {
            let nn = T::from(ncnt as f64).unwrap();
            let s = sum[(i1, j1)] - sum[(i0, j1)] - sum[(i1, j0)] + sum[(i0, j0)];
            let q = sq[(i1, j1)] - sq[(i0, j1)] - sq[(i1, j0)] + sq[(i0, j0)];
            let m0 = s / nn;
            let var = (q / nn - m0 * m0).max(T::default());
            (m0, var.sqrt())
        } else {
            let mean = if dist[(li, lj)] == u32::MAX {
                image[(i, j)]
            } else {
                nearest[(li, lj)]
            };
            (mean, eps)
        };
        means.push(mean);
        sigmas.push(sigma);
    }

    let mut sorted: Vec<T> = sigmas.iter().copied().filter(|&s| s > eps).collect();
    let robust_sigma = if sorted.is_empty() {
        eps
    } else {
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) * T::from(0.5).unwrap()
        } else {
            sorted[mid]
        }
    }
    .max(eps);

    for (k, &(i, j)) in comp.coords.iter().enumerate() {
        let li = i - sr0;
        let lj = j - sc0;
        // Fraction of texture noise to keep, 0 at the boundary -> 1 away from it.
        let dd = dist[(li, lj)];
        let keep = if dd == u32::MAX {
            T::from(1.0).unwrap()
        } else {
            let d = T::from(dd as f64).unwrap();
            let u = (d / ramp_dist).min(T::from(1.0).unwrap());
            u * u * (T::from(3.0).unwrap() - T::from(2.0).unwrap() * u)
        };

        let mean = means[k];
        let normal = Normal::new(mean, robust_sigma).unwrap();
        let sample = normal.sample(rng);
        result[(i, j)] = mean + keep * (sample - mean);
    }
}

/// Fill masked pixels from boundary-consistent values plus texture noise.
///
/// For each masked pixel, the mean is the equal-weight average of unmasked
/// neighbors within `radius`, computed in `O(1)` per pixel via summed-area
/// tables. Masked pixels whose `radius` window contains no unmasked neighbor
/// fall back to the nearest unmasked value obtained from a multi-source BFS.
///
/// The added texture noise reproduces the background's natural per-pixel noise:
/// every masked pixel is filled with its local mean plus a Gaussian sample whose
/// standard deviation is a single robust (median) estimate of the local texture
/// spread for the component, so the fill has the same noise character as the
/// surroundings without a mottled rim. Only the outermost few pixels are blended
/// smoothly into the boundary (the noise is scaled to zero there and ramps back
/// to full strength within a couple of pixels).
///
/// Work is scoped per connected component of the mask, so the cost scales with
/// the masked footprint (bounding boxes) rather than the full image.
///
/// Values are sampled as `Normal(mean, max(sigma, EPS))` where `EPS` is a
/// small positive floor that prevents a non-positive standard deviation.
/// Unlike `replace_masked_with_noise`, values may be negative without
/// panicking and a single RNG is used so the whole mask gets varied texture.
///
/// # Arguments
/// * `image` - Input image (contains original pixel values)
/// * `mask` - Boolean mask indicating pixels to replace (True = replace)
/// * `radius` - Search radius (in pixels) for the local mean/std window
/// * `random_seed` - Optional seed for reproducible output
///
/// # Returns
/// * `Array2<T>` - Image with masked pixels replaced by boundary-fill values
fn fill_masked_from_boundary<T: NdFloat + Default>(
    image: ArrayView2<T>,
    mask: &ArrayView2<bool>,
    radius: usize,
    random_seed: Option<u64>,
) -> Array2<T>
where
    StandardNormal: Distribution<T>,
{
    let mut result = image.to_owned();

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut rand::rng()),
    };

    for comp in find_components(mask) {
        fill_component(&mut result, image, mask, &comp, radius, &mut rng);
    }

    result
}

/// Fill masked pixels with a radial stellar brightness profile.
///
/// Intended for the L (lightness) channel of a Lab image when reconstructing
/// saturated stars, where the true core is brighter than the surrounding
/// unsaturated wings but is hidden by the mask.
///
/// Seed = the value of the nearest unmasked pixel (so every masked pixel is
/// contiguous with its local boundary value at the edge — no dip, no halo) plus
/// a radial brightness lift growing toward the core:
///
/// ```text
/// L(p) = nearest_unmasked_value(p) + peak * depth * ( (1-a)*(d'/depth') + a*(d'/depth')^2 )
/// ```
///
/// where `d = distance_to_edge(p)` (8-connected Chebyshev distance) and `depth`
/// is the maximum `d` over the component (how deep / how large the masked core
/// is). The lift uses `d' = d - 1` and `depth' = depth - 1`, so the outermost
/// masked pixels (`d = 1`) get zero lift and the fill is exactly contiguous
/// with the surrounding value at the boundary; it then rises smoothly and
/// monotonically to a brighter core. If a component has no reachable unmasked
/// pixel (a fully-masked / no-boundary-data mask) the lift ramps are skipped and
/// those pixels are left unchanged, instead of blowing up from the sentinel
/// depth. Larger/deeper masks get a brighter core (`peak * depth`) — matching the
/// rule that a bigger saturated area corresponds to a brighter star. `peak` is a
/// tunable amplitude (signal per pixel of mask depth). The quadratic (squared)
/// term gives the dome curvature / brighter core while the linear term gives a
/// nonzero slope right at the seam (d'=0) so the fill continues the surrounding
/// star's radial gradient instead of a flat shoulder.
///
/// Anchoring to the *nearest* boundary value (rather than a mean over a local
/// window) is what removes the dark dip that a window-averaged base produces on
/// a steep stellar wing. Only pixels outside the mask contribute to the base; the
/// core brightness is set by the model. Used only for the L channel via
/// `init_method="radial_rise"`; the a/b color channels keep `boundary_fill`.
fn fill_masked_radial_rise<T: NdFloat + Default>(
    image: ArrayView2<T>,
    mask: &ArrayView2<bool>,
    peak: T,
) -> Array2<T> {
    let (h, w) = image.dim();
    let one = T::from(1.0).unwrap();
    // Tuning knob: alpha=1 is a pure quadratic (zero initial seam slope),
    // alpha=0 a pure cone (constant slope). A small value (~0.25) matches the
    // outer wing gradient. Core brightness stays peak*depth for any alpha.
    let a = T::from(0.25).unwrap();

    // Multi-source BFS seeded at all unmasked pixels: propagate both the
    // distance-to-edge and the value of the nearest unmasked pixel inward.
    let mut result = image.to_owned();
    let mut dist = Array2::<u32>::from_elem((h, w), u32::MAX);
    for i in 0..h {
        for j in 0..w {
            if !mask[(i, j)] {
                dist[(i, j)] = 0;
            }
        }
    }
    bfs_nearest_value(&mut result, &mut dist);

    // Add the radial brightness lift, scaled per component by its depth. The
    // ramp uses d-1 (not d), so the outermost masked pixels (d == 1) get zero
    // lift and the fill is exactly contiguous with the surrounding value at the
    // boundary.
    for comp in find_components(mask) {
        // Maximum Chebyshev distance-to-edge over the component.
        let mut depth = 1u32;
        for &(i, j) in &comp.coords {
            depth = depth.max(dist[(i, j)]);
        }
        // No reachable unmasked pixel (all-masked / no boundary data): the
        // sentinel u32::MAX remains and treating it as a depth would explode the
        // lift (peak * 4.29e9). Leave those pixels unchanged — `result` already
        // holds the original image value wherever no seed was reachable.
        if depth == u32::MAX {
            continue;
        }
        let depth_t = T::from(depth).unwrap();
        let inv_depth = if depth > 1 {
            T::from(1.0).unwrap() / T::from(depth - 1).unwrap()
        } else {
            T::from(1.0).unwrap()
        };
        for &(i, j) in &comp.coords {
            let d = dist[(i, j)];
            // d >= 1 for masked pixels; d-1 reaches 0 at the boundary and the
            // deepest cell (d == depth) reaches 1, so core brightness stays
            // peak*depth while the edge gets no lift.
            let dnorm = if depth > 1 {
                (T::from(d - 1).unwrap() * inv_depth).min(one)
            } else {
                T::default()
            };
            let lift = peak * depth_t * ((one - a) * dnorm + a * dnorm * dnorm);
            let v = result[(i, j)] + lift;
            result[(i, j)] = v.max(T::default());
        }
    }
    result
}

/// Estimate the star's intrinsic colour for `comp` by integrating conserved
/// per-band flux in linear RGB.
///
/// Explanation: for a telescope whose chromatic diffraction spikes are rotated /
/// smeared into a starburst pattern (e.g. multi-exposure, rotating mount), the
/// per-pixel `a`/`b` (and thus hue) is noise-chromatic and useless. But the
/// misalignment only *redistributes* each band's light — it conserves total
/// flux. So integrating the background-subtracted linear-RGB flux over a region
/// that encloses the whole starburst recovers the star's intrinsic colour ratio.
///
/// The region is a disk of `radius` around the component centroid; the
/// background is the median linear-RGB in the ring `[bg_inner, bg_outer]` around
/// the same centre. Only unmasked pixels are used. Returns `None` if the
/// integrated flux is too small to trust (faint star / dominated by noise).
///
/// This consumes a precomputed linear-RGB buffer (`lin`, shape (h, w, 3)) owned
/// by the caller (materialised once per `reconstruct_star_color` call rather than
/// once per component), instead of deriving linear RGB from Oklab here.
fn estimate_star_linear_colour(
    lin: &Array3<f64>,
    mask: &ArrayView2<bool>,
    comp: &Component,
    radius: f64,
    bg_inner: f64,
    bg_outer: f64,
) -> Option<[f64; 3]> {
    let (h, w, _) = lin.dim();
    let cy = comp.coords.iter().map(|&(i, _)| i as f64).sum::<f64>() / comp.coords.len() as f64;
    let cx = comp.coords.iter().map(|&(_, j)| j as f64).sum::<f64>() / comp.coords.len() as f64;

    let in_disk = |i: usize, j: usize| {
        let dy = i as f64 - cy;
        let dx = j as f64 - cx;
        dy * dy + dx * dx <= radius * radius
    };
    let in_bg = |i: usize, j: usize| {
        let dy = i as f64 - cy;
        let dx = j as f64 - cx;
        let d2 = dy * dy + dx * dx;
        d2 >= bg_inner * bg_inner && d2 < bg_outer * bg_outer
    };

    // Background median linear RGB over the ring.
    let mut bg_r: Vec<f64> = Vec::new();
    let mut bg_g: Vec<f64> = Vec::new();
    let mut bg_b: Vec<f64> = Vec::new();
    for i in 0..h {
        for j in 0..w {
            if mask[(i, j)] || !in_bg(i, j) {
                continue;
            }
            bg_r.push(lin[(i, j, 0)]);
            bg_g.push(lin[(i, j, 1)]);
            bg_b.push(lin[(i, j, 2)]);
        }
    }
    let bg = if bg_r.is_empty() {
        [0.0, 0.0, 0.0]
    } else {
        let p = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        [p(bg_r), p(bg_g), p(bg_b)]
    };

    // Integrated background-subtracted linear flux over the disk.
    let mut s = [0.0f64; 3];
    for i in 0..h {
        for j in 0..w {
            if mask[(i, j)] || !in_disk(i, j) {
                continue;
            }
            s[0] += (lin[(i, j, 0)] - bg[0]).max(0.0);
            s[1] += (lin[(i, j, 1)] - bg[1]).max(0.0);
            s[2] += (lin[(i, j, 2)] - bg[2]).max(0.0);
        }
    }

    // If the total flux is negligible the "colour" is just noise; refuse.
    if s[0] + s[1] + s[2] < 1e-6 {
        return None;
    }
    Some(s)
}

/// Build a sorted-by-L lookup curve `(L, a, b)` for light of fixed chromaticity
/// `u` (a linear-RGB direction), i.e. colours `t * u` traced through Oklab.
///
/// Each point is a uniform-chromaticity colour; `L` is monotone in `t`, so the
/// returned slice is sorted by `L` and can be searched for a target lightness.
fn build_colour_curve(u: [f64; 3], l_min: f64, l_max: f64) -> Vec<[f64; 3]> {
    let n = 512;
    // Rough scan range in t; cover a wide range of brightness. The per-pixel
    // query interpolates, so exact extent mostly just needs to bound L.
    let t_min = 1e-9_f64;
    let t_max = 1e3_f64;
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(n);
    for k in 0..n {
        let t = t_min * (t_max / t_min).powf(k as f64 / (n as f64 - 1.0));
        let rgb = [u[0] * t, u[1] * t, u[2] * t];
        pts.push(linear_rgb_to_oklab(rgb));
    }
    // Keep only points spanning the queried L range to tighten the lookup.
    let mut pts: Vec<[f64; 3]> = pts
        .into_iter()
        .filter(|p| p[0] >= l_min - 0.05 && p[0] <= l_max + 0.05)
        .collect();
    if pts.is_empty() {
        pts.push(linear_rgb_to_oklab([u[0] * 0.001, u[1] * 0.001, u[2] * 0.001]));
    }
    pts.sort_by(|x, y| x[0].partial_cmp(&y[0]).unwrap_or(std::cmp::Ordering::Equal));
    pts
}

/// Look up the `(a, b)` of the constant-chromaticity curve at lightness `lp`,
/// linear-interpolating between the two bracketing samples.
fn lookup_ab_curve(curve: &[[f64; 3]], lp: f64) -> [f64; 2] {
    if curve.len() == 1 || lp <= curve[0][0] {
        return [curve[0][1], curve[0][2]];
    }
    let last = curve[curve.len() - 1];
    if lp >= last[0] {
        return [last[1], last[2]];
    }
    // Binary search for the first point with L > lp.
    let (mut lo, mut hi) = (0usize, curve.len() - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if curve[mid][0] <= lp {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (p0, p1) = (curve[lo], curve[hi]);
    let f = (lp - p0[0]) / (p1[0] - p0[0]).max(1e-12);
    [p0[1] + f * (p1[1] - p0[1]), p0[2] + f * (p1[2] - p0[2])]
}

/// Reconstruct the `a`/`b` channels of saturated stars.
///
/// See the `reconstruct_star_color` pyo3 wrapper for the full docstring. This
/// fills each masked component with a constant-chromaticity colour: the colour
/// is estimated by integrating conserved per-band linear-RGB flux, and each
/// masked pixel's `(a, b)` is set from that colour at the pixel's reconstructed
/// lightness `L`.
fn fill_star_colour(
    a_out: &mut Array2<f64>,
    b_out: &mut Array2<f64>,
    l_img: &ArrayView2<f64>,
    a_img: &ArrayView2<f64>,
    b_img: &ArrayView2<f64>,
    lin: &Array3<f64>,
    mask: &ArrayView2<bool>,
    radius: f64,
    bg_inner: f64,
    bg_outer: f64,
    blend: f64,
) {
    let (h, w) = mask.dim();

    // Nearest-unmasked a/b values for the optional boundary blend. Each channel
    // must be propagated from a freshly-seeded (identical) distance map; the
    // first BFS populates `dist` with finite distances at masked cells, so it
    // has to be re-seeded before the second channel's BFS, otherwise `near_b`
    // would never be written and the b blend would fall back to the original
    // saturated values.
    let mut near_a = a_img.to_owned();
    let mut near_b = b_img.to_owned();
    let mut dist = Array2::<u32>::from_elem((h, w), u32::MAX);
    seed_nearest_dist(&mut dist, mask);
    bfs_nearest_value(&mut near_a, &mut dist);
    seed_nearest_dist(&mut dist, mask);
    bfs_nearest_value(&mut near_b, &mut dist);

    for comp in find_components(mask) {
        let Some(u) = estimate_star_linear_colour(lin, mask, &comp, radius, bg_inner, bg_outer)
        else {
            continue; // too faint / no signal: leave the masked pixels untouched
        };
        let mut l_min = f64::INFINITY;
        let mut l_max = f64::NEG_INFINITY;
        for &(i, j) in &comp.coords {
            l_min = l_min.min(l_img[(i, j)]);
            l_max = l_max.max(l_img[(i, j)]);
        }
        let curve = build_colour_curve(u, l_min, l_max);

        for &(i, j) in &comp.coords {
            let lp = l_img[(i, j)];
            let [am, bm] = lookup_ab_curve(&curve, lp);
            let dd = dist[(i, j)];
            let w = if blend <= 0.0 {
                1.0
            } else {
                let dd = dd as f64;
                let uu = (dd / blend).min(1.0);
                uu * uu * (3.0 - 2.0 * uu) // smoothstep: 0 at edge -> 1 inside
            };
            a_out[(i, j)] = w * am + (1.0 - w) * near_a[(i, j)];
            b_out[(i, j)] = w * bm + (1.0 - w) * near_b[(i, j)];
        }
    }
}

/// Validate that every masked pixel has a positive value, as required by the
/// `init_method="noise"` seeding (which uses the masked value both as the mean
/// and the standard deviation of a Gaussian). Returns a `PyValueError` naming
/// the offending pixel instead of panicking inside `Normal::new(...).unwrap()`.
fn validate_noise_masked(image: ArrayView2<f64>, mask: &ArrayView2<bool>) -> PyResult<()> {
    let (h, w) = mask.dim();
    for i in 0..h {
        for j in 0..w {
            if mask[(i, j)] && image[(i, j)] <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "init_method='noise' requires all masked pixel values to be positive, \
                     but masked pixel ({i}, {j}) has value {}",
                    image[(i, j)]
                )));
            }
        }
    }
    Ok(())
}

/// Inpaint masked regions in a grayscale image using anisotropic diffusion.
///
/// First replaces masked regions with Gaussian noise (mean=original pixel
/// value, std=original pixel value), then applies anisotropic diffusion
/// while respecting mask boundaries. The diffusion process smooths the
/// inpainted region while maintaining consistency with surrounding pixels.
///
/// Parameters
/// ----------
/// image : `NDArray`
///     Input grayscale image as 2D numpy array of dtype float64.
/// mask : `NDArray`
///     Boolean mask where True indicates regions to inpaint. Must have
///     same shape as image. Pixels with True are replaced with a seed
///     and then diffused.
/// init_method : `str`, optional
///     How masked pixels are initialized before diffusion. One of:
///
///     - ``"boundary_fill"`` (default): fill each masked pixel with the
///       equal-weight mean of unmasked neighbors within ``radius`` plus Gaussian
///       noise whose standard deviation is a single robust (median) estimate of
///       the surrounding texture (preserving realistic noise character).
///       Converges in far fewer iterations and tolerates negative values.
///     - ``"noise"``: the legacy behavior, filling each masked pixel with
///       Gaussian noise of mean and standard deviation equal to the original
///       pixel value. Simpler, but offers no convergence benefit and requires
///       positive pixel values.
///     - ``"none"``: leave the masked pixels exactly as supplied in
///       ``image`` (no re-fill). Use this to pre-seed the mask with your own
///       model (e.g. a radial stellar brightness profile) before the
///       diffusion refines it.
///     - ``"radial_rise"``: seed the mask with a radial stellar brightness
///       profile (higher in the core, brighter for larger/deeper masks,
///       contiguous at the boundary). Use for the L channel when
///       reconstructing saturated stars; see ``peak_amp``.
/// iterations : `int`, optional
///     Number of diffusion iterations. Higher values produce more
///     complete inpainting. Default is 32.
/// anisotropy_first : `float`, optional
///     Diffusion strength for gradient-based first directional derivative.
///     Default is 0.0 (isotropic diffusion for inpainting).
/// anisotropy_second : `float`, optional
///     Diffusion strength for Laplacian-based second directional derivative.
///     Default is 0.0.
/// anisotropy_third : `float`, optional
///     Diffusion strength for third directional derivative term.
///     Default is 0.0.
/// anisotropy_fourth : `float`, optional
///     Diffusion strength for fourth directional derivative term.
///     Default is 2.0 (edge-preserving).
/// regularization : `float`, optional
///     Regularization parameter for numerical stability. Default is 0.0.
/// variance_threshold : `float`, optional
///     Minimum variance threshold. Default is 0.0.
/// radius_center : `float`, optional
///     Center radius for diffusion weighting. Default is 0.0.
/// first : `float`, optional
///     First diffusion coefficient. Default is 0.0.
/// second : `float`, optional
///     Second diffusion coefficient. Default is 0.0.
/// third : `float`, optional
///     Third diffusion coefficient. Default is 0.0.
/// fourth : `float`, optional
///     Fourth diffusion coefficient. Default is 1.0.
/// radius : `float`, optional
///     Diffusion radius parameter. Default is 5.0.
/// sharpness : `float`, optional
///     Sharpness enhancement parameter. Default is 0.0.
/// random_seed : `int`, optional
///     An optional positive int that is used to set the random seed. If
///     None, no seed will be set.
/// peak_amp : `float`, optional
///     Amplitude (in signal per pixel of mask depth) of the radial-rise core
///     used by ``init_method="radial_rise"``. Larger values produce a brighter
///     reconstructed star core. Ignored by other init methods. Default is 0.02.
///
/// Notes
/// -----
/// The ``"boundary_fill"`` initialization starts each masked pixel near the
/// smooth boundary-consistent solution, so the anisotropic diffusion largely
/// refines fine-scale detail rather than tearing down large noise. This both
/// reduces the number of iterations required and produces cleaner mask
/// boundaries. With ``"noise"`` the initialization matches the original
/// behavior.
///
/// The diffusion process ensures:
/// - Values at mask boundaries match the surrounding image
/// - Interior values are smoothly interpolated
/// - Edge preservation properties are maintained
///
/// Returns
/// -------
/// result : `NDArray`
///     Inpainted image as 2D numpy array of dtype float64 with same shape
///     as input.
///
/// Raises
/// ------
/// `ValueError`
///     If image and mask dimensions do not match.
///     If mask contains no True pixels (nothing to inpaint).
///
/// See Also
/// --------
/// diffuse_gray_image : Apply anisotropic diffusion to entire image
#[pyfunction]
#[pyo3(signature = (image,
    mask,
    iterations= 32,
    anisotropy_first= 0.0,
    anisotropy_second= 0.0,
    anisotropy_third= 0.0,
    anisotropy_fourth= 2.0,
    regularization= 0.0,
    variance_threshold= 0.0,
    radius_center= 0.0,
    first= 0.0,
    second= 0.0,
    third= 0.0,
    fourth= 1.0,
    radius= 5.0,
    sharpness= 0.0,
    random_seed = None,
    init_method = "boundary_fill",
    peak_amp = 0.02
))]
pub fn inpaint_mask<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f64>,
    mask: PyReadonlyArray2<bool>,
    iterations: usize,
    anisotropy_first: f64,
    anisotropy_second: f64,
    anisotropy_third: f64,
    anisotropy_fourth: f64,
    regularization: f64,
    variance_threshold: f64,
    radius_center: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
    radius: f64,
    sharpness: f64,
    random_seed: Option<u64>,
    init_method: &str,
    peak_amp: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = image.as_array();
    let mask_array = mask.as_array();
    log::debug!(
        "Inpainting mask with {} pixels",
        mask_array.iter().fold(0, |acc, &b| acc + b as usize)
    );
    if array.dim() != mask_array.dim() {
        return Err(PyValueError::new_err(format!(
            "image {:?} and mask {:?} must have the same dimensions",
            array.dim(),
            mask_array.dim()
        )));
    }
    if !matches!(init_method, "noise" | "none" | "radial_rise" | "boundary_fill") {
        return Err(PyValueError::new_err(format!(
            "unsupported init_method {init_method:?}; expected one of \
             \"noise\", \"none\", \"radial_rise\", \"boundary_fill\""
        )));
    }

    let process_args = ProcessArgs {
        iterations,
        anisotropy_first,
        anisotropy_second,
        anisotropy_third,
        anisotropy_fourth,
        regularization,
        variance_threshold,
        radius_center,
        first,
        second,
        third,
        fourth,
        radius,
        sharpness,
    };
    let init_start = std::time::Instant::now();
    let mut masked = match init_method {
        "noise" => {
            validate_noise_masked(array, &mask_array)?;
            replace_masked_with_noise(array, &mask_array, random_seed)
        }
        // "none": keep the caller-supplied masked values as the diffusion seed
        // (no statistical re-fill). Lets an application pre-seed the mask with a
        // model profile (e.g. a radial stellar PSF extrapolation) and then let
        // the diffusion refine it while keeping the surrounding pixels fixed.
        "none" => array.to_owned(),
        // "radial_rise": seed the mask with a radial stellar brightness profile
        // for saturated-star reconstruction (typically the L channel). Brighter
        // cores for larger/deeper masks, contiguous at the boundary.
        "radial_rise" => fill_masked_radial_rise(array, &mask_array, peak_amp),
        _ => fill_masked_from_boundary(array, &mask_array, radius.max(1.0) as usize, random_seed),
    };
    let init_elapsed = init_start.elapsed();

    let diff_start = std::time::Instant::now();
    let result = process_image(process_args, &mut masked.view_mut(), Some(mask_array));
    let diff_elapsed = diff_start.elapsed();

    log::debug!(
        "[inpaint_mask] init_method={} radius={} init={:.3?} diffusion={:.3?} total={:.3?}",
        init_method,
        radius,
        init_elapsed,
        diff_elapsed,
        init_elapsed + diff_elapsed
    );

    Ok(result.to_pyarray(py))
}

/// Reconstruct the `a`/`b` colour channels of saturated stars (Oklab image).
///
/// Parameters
/// ----------
/// L : `NDArray`
///     Lightness channel (2D float64). Should be the *reconstructed* L from
///     `inpaint_mask(..., init_method="radial_rise")`, since the masked core is
///     what gets displayed and its lightness drives the colour lookup.
/// a, b : `NDArray`
///     The original a and b channels of the Oklab image (2D float64).
/// mask : `NDArray`
///     Boolean mask where True marks the saturated star cores to reconstruct.
/// linear_rgb : `NDArray`, optional
///     Precomputed linear-RGB image cube of shape (h, w, 3) with channels in RGB
///     order, holding the raw linear float data that was fed to `RGB_to_Oklab`
///     with the DEFAULT D65 illuminant (so it is in the crate's D65 working
///     space). When supplied, it is used directly for the per-band flux
///     integration that estimates each star's colour, instead of re-deriving
///     linear RGB from the L/a/b arrays. When `None` (default), linear RGB is
///     derived once from L/a/b.
/// radius : `float`, optional
///     Outer radius (px) of the disk around each star over which per-band
///     linear flux is integrated to estimate the star's colour. Must enclose the
///     whole chromatic-spike / starburst pattern. Default is 90.0.
/// bg_inner : `float`, optional
///     Inner radius (px) of the background ring used for sky subtraction.
///     Default is 200.0.
/// bg_outer : `float`, optional
///     Outer radius (px) of the background ring. Default is 300.0.
/// blend : `float`, optional
///     Width (px) over which the reconstructed colour is blended toward the
///     nearest unmasked pixel at the mask boundary, for a seamless seam. 0
///     disables blending. Default is 2.0.
///
/// Returns
/// -------
/// (a, b) : tuple of `NDArray`
///     Two float64 arrays, the reconstructed a and b channels (masked pixels
///     filled; unmasked pixels unchanged). Components too faint to estimate a
///     colour are left with their original masked values.
///
/// Raises
/// ------
/// `ValueError`
///     If L, a, b, or mask shapes do not all match.
/// `ValueError`
///     If `linear_rgb` is provided but is not (h, w, 3) matching L/a/b/mask, or
///     contains NaN/inf.
///
/// Notes
/// -----
/// This does NOT run the structural diffusion operator on a/b. The presence of
/// chromatic diffraction spikes (rotated between bands, or smeared by multiple
/// exposures) means that per-pixel a/b hue is noise-chromatic and that letting a
/// structural fill reproduce boundary chroma would scatter the spikes into the
/// reconstructed core. Instead the star's colour is estimated in a
/// flux-additive (linear-RGB) space — integrated per-band flux is conserved
/// under spike misalignment — and each masked pixel is painted with that single
/// colour at its own reconstructed lightness. The Oklab convention used here is
/// the library's D65 default; pass data produced with the same convention. In
/// particular, when supplying `linear_rgb` it must be in the crate's D65 working
/// space: providing data in a different whitepoint space biases the result.
#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (L, a, b, mask, linear_rgb=None, radius=90.0, bg_inner=200.0, bg_outer=300.0, blend=2.0))]
pub fn reconstruct_star_color<'py>(
    py: Python<'py>,
    L: PyReadonlyArray2<f64>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    mask: PyReadonlyArray2<bool>,
    linear_rgb: Option<PyReadonlyArray3<f64>>,
    radius: f64,
    bg_inner: f64,
    bg_outer: f64,
    blend: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let l_img = L.as_array();
    let a_img = a.as_array();
    let b_img = b.as_array();
    let mask_array = mask.as_array();
    let (h, w) = l_img.dim();
    let (ah, aw) = a_img.dim();
    let (bh, bw) = b_img.dim();
    let (mh, mw) = mask_array.dim();
    if (ah, aw) != (h, w) || (bh, bw) != (h, w) || (mh, mw) != (h, w) {
        return Err(PyValueError::new_err(format!(
            "L({h}x{w}), a({ah}x{aw}), b({bh}x{bw}), mask({mh}x{mw}) shapes must match"
        )));
    }
    // Validate the geometry parameters: all finite, positive, and strictly
    // ordered. NaN comparisons are false so NaN would slip past a bare
    // `radius < bg_inner` check, hence the explicit is_finite() rejection.
    if !(radius.is_finite() && bg_inner.is_finite() && bg_outer.is_finite()) {
        return Err(PyValueError::new_err(
            "radius, bg_inner, and bg_outer must all be finite (rejecting NaN/inf)",
        ));
    }
    if radius <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "radius ({radius}) must be positive"
        )));
    }
    if bg_inner <= radius {
        return Err(PyValueError::new_err(format!(
            "bg_inner ({bg_inner}) must be greater than radius ({radius})"
        )));
    }
    if bg_outer <= bg_inner {
        return Err(PyValueError::new_err(format!(
            "bg_outer ({bg_outer}) must be greater than bg_inner ({bg_inner})"
        )));
    }

    // Build the shared linear-RGB buffer once per call (per pixel), so the
    // flux integration per component reuses it instead of re-converting.
    let mut lin: Array3<f64>;
    if let Some(cube) = linear_rgb {
        let (ch, cw, cc) = cube.as_array().dim();
        if cc != 3 {
            return Err(PyValueError::new_err(format!(
                "linear_rgb must have exactly 3 channels (R,G,B) along the last axis; got {cc}"
            )));
        }
        if (ch, cw) != (h, w) {
            return Err(PyValueError::new_err(format!(
                "linear_rgb shape ({ch}x{cw}x{cc}) must match L/a/b/mask shape ({h}x{w})"
            )));
        }
        if cube.as_array().iter().any(|&x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "linear_rgb must be finite (rejecting NaN/inf)",
            ));
        }
        lin = cube.as_array().to_owned();
    } else {
        lin = Array3::<f64>::zeros((h, w, 3));
        for i in 0..h {
            for j in 0..w {
                let [r, g, b] = oklab_to_linear_rgb([l_img[(i, j)], a_img[(i, j)], b_img[(i, j)]]);
                lin[[i, j, 0]] = r;
                lin[[i, j, 1]] = g;
                lin[[i, j, 2]] = b;
            }
        }
    }

    let mut a_out = a_img.to_owned();
    let mut b_out = b_img.to_owned();
    fill_star_colour(
        &mut a_out,
        &mut b_out,
        &l_img,
        &a_img,
        &b_img,
        &lin,
        &mask_array,
        radius,
        bg_inner,
        bg_outer,
        blend,
    );

    Ok((a_out.to_pyarray(py), b_out.to_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_delta;

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

    // --- Mask-aware B-spline decomposition tests ---

    fn make_center_mask() -> Array2<bool> {
        // Mask the 3x3 center (rows 2-4, cols 2-4)
        let mut mask = Array2::<bool>::from_elem((7, 7), false);
        for r in 2..=4 {
            for c in 2..=4 {
                mask[(r, c)] = true;
            }
        }
        mask
    }

    #[test]
    fn test_bspline_vertical_spans_mask() {
        // v1.1: the vertical pass includes every pixel in the low-pass (masked
        // or not), so `lf` spans the mask. A masked 1000.0 block now contributes
        // to lf above it and dominates lf deep inside it (not forced to 0).
        let mut test_img = Array2::<f64>::zeros((7, 7));
        test_img.fill(1.0);
        for r in 2..=4 {
            for c in 2..=4 {
                test_img[(r, c)] = 1000.0;
            }
        }
        let mut out_buf = vec![0.0_f64; 7];

        // Row 1, col 3 (inside the block's columns): blurred above by the 1000s.
        _bspline_vertical_pass(test_img.view_mut(), 1, 7, 7, 1, false, &mut out_buf);
        assert!(out_buf[3] > 1.0, "masked values should contribute to lf, got {}", out_buf[3]);

        // Deep in the block (row 3, col 3) lf reflects the high fill values.
        _bspline_vertical_pass(test_img.view_mut(), 3, 7, 7, 1, false, &mut out_buf);
        assert!(out_buf[3] > 700.0, "deep-masked lf should reflect fill values, got {}", out_buf[3]);
    }

    #[test]
    fn test_bspline_horizontal_includes_all() {
        // v1.1: the horizontal pass includes every column (masked or not), so a
        // constant row returns the constant at any position.
        let mut row = vec![0.0_f64; 7];
        for c in 0..7 {
            row[c] = 1.0;
        }

        let result1 = _bspline_horizontal(&row, 1, 7, 1, false);
        assert_delta!(result1, 1.0, 1e-10);

        let result2 = _bspline_horizontal(&row, 3, 7, 1, false);
        assert_delta!(result2, 1.0, 1e-10);
    }

    #[test]
    fn test_bspline_vertical_includes_masked() {
        // v1.1: all pixels are included in the vertical low-pass (masked pixels
        // were previously excluded, giving 0 when every neighbor was masked).
        // A constant image returns the constant.
        let mut img = Array2::<f64>::zeros((3, 3));
        img.fill(42.0);
        let mut out_buf = vec![0.0_f64; 3];

        _bspline_vertical_pass(img.view_mut(), 1, 3, 3, 1, false, &mut out_buf);
        assert_delta!(out_buf[0], 42.0, 1e-10);
    }

    #[test]
    fn test_decompose_lowpass_spans_mask() {
        // v1.1: masked pixels are included in the low-pass, so lf spans the mask
        // region instead of going to zero deep inside it. A deep-masked pixel
        // must have a non-zero lf reflecting the surrounding (filled) values,
        // and lf + hf reconstructs the input exactly everywhere.
        let mut img = Array2::<f64>::zeros((7, 7));
        for r in 2..=4 {
            for c in 2..=4 {
                img[(r, c)] = 1000.0;
            }
        }

        let mut hf = Array2::<f64>::zeros((7, 7));
        let mut lf = Array2::<f64>::zeros((7, 7));
        let mut row_buf = vec![0.0_f64; 7];

        decompose_2d_bspline(
            img.view_mut(),
            hf.view_mut(),
            lf.view_mut(),
            7, 7, 1, &mut row_buf,
        );

        assert!(lf[(3, 3)] > 700.0, "lf at deep masked pixel should be non-zero, got {}", lf[(3, 3)]);
        for r in 0..7 {
            for c in 0..7 {
                let delta = (lf[(r, c)] + hf[(r, c)] - img[(r, c)]).abs();
                assert!(delta < 1e-9, "reconstruction failed at {},{}: {}", r, c, delta);
            }
        }
    }

    #[test]
    fn test_decompose_reconstruction_exact_unmasked() {
        // A constant field reconstructs exactly and lf equals the constant
        // (the mask never forces lf to 0).
        let mut img = Array2::<f64>::from_elem((10, 10), 2.0);
        let mut hf = Array2::<f64>::zeros((10, 10));
        let mut lf = Array2::<f64>::zeros((10, 10));
        let mut row_buf = vec![0.0_f64; 10];

        decompose_2d_bspline(
            img.view_mut(),
            hf.view_mut(),
            lf.view_mut(),
            10, 10, 1, &mut row_buf,
        );

        assert_delta!(lf[(5, 5)], 2.0, 1e-9);
        assert_delta!(hf[(5, 5)], 0.0, 1e-9);
        assert!(lf.iter().all(|&v| (v - 2.0).abs() < 1e-9), "lf must span the whole field");
    }

    #[test]
    fn test_inpaint_mask_preserves_unmasked_values() {
        // Verify that the full inpainting pipeline preserves unmasked values
        // and produces finite output for offset (non-negative) image values.
        let mut img = Array2::<f64>::zeros((10, 10));
        // Linear gradient, all values >= 0 so noise init doesn't panic
        for r in 0..10 {
            for c in 0..10 {
                img[(r, c)] = 0.1 + (r as f64) * 0.1 + (c as f64) * 0.05;
            }
        }

        let mut mask = Array2::<bool>::from_elem((10, 10), false);
        for r in 3..=6 {
            for c in 3..=6 {
                mask[(r, c)] = true;
            }
        }

        let process_args = ProcessArgs {
            iterations: 10,
            anisotropy_first: 0.0,
            anisotropy_second: 0.0,
            anisotropy_third: 0.0,
            anisotropy_fourth: 2.0,
            regularization: 0.0,
            variance_threshold: 0.0,
            radius_center: 0.0,
            first: 0.0,
            second: 0.0,
            third: 0.0,
            fourth: 1.0,
            radius: 3.0,
            sharpness: 0.0,
        };

        let mut masked = replace_masked_with_noise(img.view(), &mask.view(), Some(42));
        let result = process_image(process_args, &mut masked.view_mut(), Some(mask.view()));

        assert_eq!(result.shape(), &[10, 10]);
        assert!(result.iter().all(|&x| x.is_finite()),
            "Result should have all finite values");

        // Unmasked pixels should be preserved exactly (diffusion only operates
        // on masked pixels, and with mask-aware decomposition, HF at boundary
        // should not be contaminated)
        for r in 0..3 {
            for c in 0..3 {
                assert_delta!(result[(r, c)], img[(r, c)], 1e-6);
            }
        }
    }

    // --- boundary_fill initialization tests ---

    #[test]
    fn test_boundary_fill_single_pixel_approximates_neighbors() {
        // Constant image (value 5.0) with one masked pixel.
        // The masked pixel should be filled close to the boundary value 5.0.
        let img = Array2::<f64>::from_elem((7, 7), 5.0);
        let mut mask = Array2::<bool>::from_elem((7, 7), false);
        mask[(3, 3)] = true;

        let filled = fill_masked_from_boundary(img.view(), &mask.view(), 3, Some(1));
        // Constant neighbors give zero local std (floored to ~1e-6), so the
        // fill stays within a few sigma of the boundary value.
        assert_delta!(filled[(3, 3)], 5.0, 1e-3);
    }

    #[test]
    fn test_boundary_fill_matches_local_mean() {
        // Image where boundary around the mask is all value 2.5.
        let mut img = Array2::<f64>::from_elem((9, 9), 2.5);
        // Mask the 3x3 center.
        let mut mask = Array2::<bool>::from_elem((9, 9), false);
        for r in 3..=5 {
            for c in 3..=5 {
                mask[(r, c)] = true;
            }
        }
        // Pretend the masked/raw values are garbage (1000).
        for r in 3..=5 {
            for c in 3..=5 {
                img[(r, c)] = 1000.0;
            }
        }

        let filled = fill_masked_from_boundary(img.view(), &mask.view(), 2, Some(1));
        // Interior masked pixels derive mean from surrounding 2.5 neighbors,
        // so they should be close to 2.5 (not 1000).
        assert_delta!(filled[(4, 4)], 2.5, 1.0);
    }

    #[test]
    fn test_boundary_fill_negative_values_no_panic() {
        // Values centered around 0 (Lab a/b channel) must not panic.
        let mut img = Array2::<f64>::zeros((10, 10));
        for r in 0..10 {
            for c in 0..10 {
                img[(r, c)] = (r as f64 - 5.0) * 0.5 + (c as f64 - 5.0) * 0.3;
            }
        }
        let mut mask = Array2::<bool>::from_elem((10, 10), false);
        for r in 3..=6 {
            for c in 3..=6 {
                mask[(r, c)] = true;
            }
        }

        let filled = fill_masked_from_boundary(img.view(), &mask.view(), 3, Some(42));
        assert!(filled.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_boundary_fill_fully_masked_finite() {
        // Fully masked image: no valid neighbors, must not panic and stay finite.
        let img = Array2::<f64>::from_elem((10, 10), 3.0);
        let mask = Array2::<bool>::from_elem((10, 10), true);

        let filled = fill_masked_from_boundary(img.view(), &mask.view(), 3, Some(7));
        assert!(filled.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_boundary_fill_varied_across_seeds() {
        // Texture should produce run-to-run variation across seeds.
        let mut img = Array2::<f64>::from_elem((15, 15), 10.0);
        let mut mask = Array2::<bool>::from_elem((15, 15), false);
        for r in 3..=11 {
            for c in 3..=11 {
                mask[(r, c)] = true;
            }
        }
        // Give the boundary some texture variation so local std > 0.
        for r in 0..15 {
            for c in 0..15 {
                if !mask[(r, c)] {
                    img[(r, c)] = 10.0 + ((r * 7 + c * 3) % 5) as f64;
                }
            }
        }

        let filled_a = fill_masked_from_boundary(img.view(), &mask.view(), 3, Some(1));
        let filled_b = fill_masked_from_boundary(img.view(), &mask.view(), 3, Some(2));
        // Center pixel should differ between seeds due to texture noise.
        assert!(
            (filled_a[(7, 7)] - filled_b[(7, 7)]).abs() > 1e-9,
            "boundary_fill should vary across seeds"
        );
    }

    #[test]
    fn test_find_components_scattered() {
        // Two separated masked blobs should be detected as two components.
        let mut mask = Array2::<bool>::from_elem((10, 10), false);
        // Blob A (2x2) and blob B (diagonally connected pair), separated.
        for r in 1..=2 {
            for c in 1..=2 {
                mask[(r, c)] = true;
            }
        }
        mask[(6, 6)] = true;
        mask[(7, 6)] = true;

        let comps = find_components(&mask.view());
        assert_eq!(comps.len(), 2);
        let total: usize = comps.iter().map(|c| c.coords.len()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_boundary_fill_deep_interior_uses_nearest() {
        // A mask larger than the search window: the deep interior has no
        // unmasked neighbor within the window, so it must fall back to the
        // nearest unmasked value from the BFS.
        let img = Array2::<f64>::from_elem((15, 15), 3.0);
        let mut mask = Array2::<bool>::from_elem((15, 15), false);
        // 5x5 central mask with radius=1 -> window is 3x3, fully masked at
        // the center (7,7), so it uses the nearest (3.0) fallback.
        for r in 5..=9 {
            for c in 5..=9 {
                mask[(r, c)] = true;
            }
        }

        let filled = fill_masked_from_boundary(img.view(), &mask.view(), 1, Some(4));
        assert_delta!(filled[(7, 7)], 3.0, 1e-3);
    }

    #[test]
    fn test_radial_rise_monotonic_contiguous() {
        // A uniform 0.4 field with a 5x5 square mask. radial_rise must be
        // brighter toward the (deepest) center, match the boundary (0.4) exactly
        // at the very edge (d == 1 gives zero lift with the d-1 ramp), and stay
        // non-negative.
        let img = Array2::<f64>::from_elem((15, 15), 0.4);
        let mut mask = Array2::<bool>::from_elem((15, 15), false);
        for r in 5..=9 {
            for c in 5..=9 {
                mask[(r, c)] = true;
            }
        }
        let filled = fill_masked_radial_rise(img.view(), &mask.view(), 0.02);

        let center = filled[(7, 7)];
        let edge = filled[(5, 5)];
        assert!(center > edge, "center {} should be brighter than edge {}", center, edge);
        assert!(filled.iter().all(|&v| v >= 0.0));
        // Edge (d == 1) gets zero lift: it must equal the 0.4 boundary exactly.
        assert_delta!(edge, 0.4, 1e-12);
        // The lift ramps in one pixel: the cell one step inside the boundary
        // (row 6) is already brighter than the edge.
        assert!(filled[(6, 6)] > edge, "fill should ramp to brightness just inside the edge");
    }

    #[test]
    fn test_radial_rise_fully_masked_stays_finite() {
        // A fully-masked image has no unmasked seed, so every dist stays u32::MAX.
        // The depth sentinel must NOT explode the lift: the fill must remain the
        // original (finite, reasonable) values rather than peak * 4.29e9.
        let img = Array2::<f64>::from_elem((10, 10), 0.4);
        let mask = Array2::<bool>::from_elem((10, 10), true);

        let filled = fill_masked_radial_rise(img.view(), &mask.view(), 0.02);
        assert!(filled.iter().all(|&v| v.is_finite()));
        assert!(filled.iter().all(|&v| (v - 0.4).abs() < 1e-12),
            "fully-masked radial_rise must leave pixels unchanged, got non-uniform fill");
    }

    #[test]
    fn test_bfs_nearest_value_propagates_chebyshev() {
        // Seeds are marked with dist == 0; unseeded cells get the nearest seed
        // value and an 8-connected (Chebyshev) distance.
        let mut values = Array2::<f64>::from_elem((3, 3), 0.0);
        let mut dist = Array2::<u32>::from_elem((3, 3), u32::MAX);
        values[(0, 0)] = 7.0;
        dist[(0, 0)] = 0; // single seed at the corner
        bfs_nearest_value(&mut values, &mut dist);

        // All cells reachable, holding the seed value.
        assert!(dist.iter().all(|&d| d != u32::MAX));
        assert!(values.iter().all(|&v| v == 7.0));
        // Chebyshev distance grows by 1 per step, so (2,2) is 2 away.
        assert_eq!(dist[(2, 2)], 2);
        assert_eq!(dist[(0, 2)], 2);

        // An isolated second seed: DBZ... nearest source decides which wins.
        let mut v2 = Array2::<f64>::from_elem((3, 3), 0.0);
        let mut d2 = Array2::<u32>::from_elem((3, 3), u32::MAX);
        v2[(0, 0)] = 1.0;
        v2[(2, 2)] = 9.0;
        d2[(0, 0)] = 0;
        d2[(2, 2)] = 0;
        bfs_nearest_value(&mut v2, &mut d2);
        // Tie point (1,1) takes whichever seed was relaxed first (origin seed).
        assert!((v2[(1, 1)] - 1.0).abs() < 1e-12 || (v2[(1, 1)] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_star_colour_curve_is_hue_stable() {
        // A uniform-chromaticity curve must have a single hue (constant a/b
        // direction) across all lightnesses, and lightness must be monotone.
        // Use a bluish chromaticity: more blue than red in linear RGB.
        let u = [0.8, 1.0, 1.2];
        let curve = build_colour_curve(u, 0.0, 1.0);
        let hue0 = curve[0][2].atan2(curve[0][1]);
        for p in curve.iter() {
            let hue = p[2].atan2(p[1]);
            let _ = (p[0], hue); // hue consistency checked below
            let delta = (hue - hue0).abs().min((hue - hue0 + std::f64::consts::TAU).abs())
                .min((hue0 - hue + std::f64::consts::TAU).abs());
            assert!(delta < 1e-1, "hue should be ~stable on a constant-chromaticity curve");
        }
        // Monotone L.
        for w in curve.windows(2) {
            assert!(w[1][0] >= w[0][0] - 1e-9, "L must be monotone on the curve");
        }
    }

    #[test]
    fn test_lookup_ab_curve_interpolates() {
        // Build a curve from a neutral chromaticity: (a, b) ~ 0 for all L, so a
        // lookup returns ~achromatic values whatever the target lightness.
        let curve = build_colour_curve([1.0, 1.0, 1.0], 0.1, 0.9);
        for lp in [0.15, 0.3, 0.5, 0.75] {
            let ab = lookup_ab_curve(&curve, lp);
            assert!(ab[0].abs() < 1e-4, "neutral star should stay achromatic, got a={}", ab[0]);
            assert!(ab[1].abs() < 1e-4, "neutral star should stay achromatic, got b={}", ab[1]);
        }
    }

}
