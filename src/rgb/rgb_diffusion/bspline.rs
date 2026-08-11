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

//! Self-contained separable B-spline wavelet decomposition and its scale math,
//! with all filter constants colocated.

use ndarray::{ArrayViewMut2, NdFloat};
use std::cmp;

/// 5-tap binomial B-spline filter approximating Gaussian convolution.
const B_SPLINE_FILTER_F64: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Base standard deviation of a single B-spline decomposition step.
pub(super) const B_SPLINE_SIGMA: f64 = 2.0553651328015339;

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
/// * `height` - Image height
/// * `mult` - Multiplier for filter support (1<<scale level)
/// * `clip_negatives` - If true, clamp negative results to zero
/// * `out_buf` - Output buffer (length = width), receives filtered row
/// * `filter` - The 5-tap binomial filter coefficients
/// * `col_lo`, `col_hi` - Inclusive range of output columns to fill (the
///   vertical pass may be limited to a bounding region of the row).
#[inline]
fn _bspline_vertical_pass<T: NdFloat + Default>(
    in_array: ArrayViewMut2<T>,
    row: usize,
    height: usize,
    mult: i32,
    clip_negatives: bool,
    out_buf: &mut [T],
    filter: &[T; 5],
    col_lo: usize,
    col_hi: usize,
) {
    let irow = row as i32;
    let indicies: [usize; 5] = [
        cmp::max(irow - 2 * mult, 0) as usize,
        cmp::max(irow - mult, 0) as usize,
        row,
        cmp::min((irow + mult) as usize, height - 1),
        cmp::min((irow + 2 * mult) as usize, height - 1),
    ];

    for index in col_lo..=col_hi {
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

/// Perform one horizontal B-spline convolution tap on a 1D slice
///
/// Applies a 5-tap binomial filter to convolve horizontally. This is the
/// single-tap counterpart to [`_bspline_vertical_pass`]: the vertical pass
/// writes a whole filtered row into `out_buf`, whereas this computes one
/// filtered value at `col` from an already-vertical-filtered row slice.
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
fn _bspline_horizontal_tap<T: NdFloat + Default>(
    in_slice: &[T],
    col: usize,
    width: usize,
    mult: i32,
    clip_negatives: bool,
    filter: &[T; 5],
) -> T {
    let icol = col as i32;
    let indicies: [usize; 5] = [
        cmp::max(icol - 2 * mult, 0) as usize,
        cmp::max(icol - mult, 0) as usize,
        col,
        cmp::min((icol + mult) as usize, width - 1),
        cmp::min((icol + 2 * mult) as usize, width - 1),
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
/// * `region` - Optional inclusive `(r0, r1, c0, c1)` bounding region over which
///   to run the decomposition. `None` runs the legacy full-image path (all rows
///   and cols, bit-for-bit identical to before). `Some` computes lf/hf only
///   within `[r0, r1]` rows and `[c0, c1]` cols (with the vertical pass extended
///   to `c0-2m..c1+2m` so every horizontal tap's `row_buf` reads are valid);
///   cells outside stay at whatever the buffers held.
#[inline]
pub(super) fn decompose_2d_bspline<T: NdFloat + Default>(
    in_array: ArrayViewMut2<T>,
    hf: ArrayViewMut2<T>,
    lf: ArrayViewMut2<T>,
    width: usize,
    height: usize,
    mult: i32,
    row_buf: &mut [T],
    region: Option<(usize, usize, usize, usize)>,
) {
    let mut hf = hf;
    let mut lf = lf;
    let mut in_array = in_array;

    let (r0, r1, c0, c1) = match region {
        Some((r0, r1, c0, c1)) => (r0, r1, c0, c1),
        None => (0, height - 1, 0, width - 1),
    };

    // Build the 5-tap filter ONCE per scale (numeric values identical to the
    // old per-call construction), and reuse it for every vertical/horizontal tap.
    let filter: [T; 5] = std::array::from_fn(|i| T::from(B_SPLINE_FILTER_F64[i]).unwrap());

    let m = mult as usize;
    // The vertical pass must fill row_buf over `[c0-2m, c1+2m]` so every
    // horizontal tap at columns c0..=c1 (which reads row_buf[col±2m]) is valid.
    let buf_col_lo = c0.saturating_sub(2 * m);
    let buf_col_hi = (c1 + 2 * m).min(width - 1);

    for row in r0..=r1 {
        _bspline_vertical_pass(
            in_array.view_mut(),
            row,
            height,
            mult,
            true,
            row_buf,
            &filter,
            buf_col_lo,
            buf_col_hi,
        );
        for col in c0..=c1 {
            let blur = _bspline_horizontal_tap(row_buf, col, width, mult, true, &filter);
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
pub(super) fn equivalent_sigma_at_step<T: NdFloat + Default>(sigma: T, s: usize) -> T {
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
pub(super) fn num_steps_to_reach_equivalent_sigma<T: NdFloat + Default>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::rgb::rgb_diffusion::regions::compute_decomp_region;
    use crate::test_utils::assert_delta;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    // --- Mask-aware B-spline decomposition tests ---

    #[allow(dead_code)]
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
        _bspline_vertical_pass(test_img.view_mut(), 1, 7, 1, false, &mut out_buf, &B_SPLINE_FILTER_F64, 0, 6);
        assert!(out_buf[3] > 1.0, "masked values should contribute to lf, got {}", out_buf[3]);

        // Deep in the block (row 3, col 3) lf reflects the high fill values.
        _bspline_vertical_pass(test_img.view_mut(), 3, 7, 1, false, &mut out_buf, &B_SPLINE_FILTER_F64, 0, 6);
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

        let result1 = _bspline_horizontal_tap(&row, 1, 7, 1, false, &B_SPLINE_FILTER_F64);
        assert_delta!(result1, 1.0, 1e-10);

        let result2 = _bspline_horizontal_tap(&row, 3, 7, 1, false, &B_SPLINE_FILTER_F64);
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

        _bspline_vertical_pass(img.view_mut(), 1, 3, 1, false, &mut out_buf, &B_SPLINE_FILTER_F64, 0, 2);
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
            None,
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
            None,
        );

        assert_delta!(lf[(5, 5)], 2.0, 1e-9);
        assert_delta!(hf[(5, 5)], 0.0, 1e-9);
        assert!(lf.iter().all(|&v| (v - 2.0).abs() < 1e-9), "lf must span the whole field");
    }

    /// Run a full (region=None) and a region-restricted decompose on identical
    /// inputs and assert hf/lf are f64-bit-exact inside the region.
    fn assert_decomp_eq_full_in_region(img: &Array2<f64>, mask: &Array2<bool>, scales: usize, mult: i32) {
        let (h, w) = img.dim();
        let region = compute_decomp_region(&mask.view(), scales, h, w)
            .expect("region should be Some for a small mask");
        let (r0, r1, c0, c1) = region;

        // Full path.
        let mut hf_full = Array2::<f64>::zeros((h, w));
        let mut lf_full = Array2::<f64>::zeros((h, w));
        let mut img_full = img.clone();
        let mut row_buf = vec![0.0_f64; w];
        decompose_2d_bspline(
            img_full.view_mut(),
            hf_full.view_mut(),
            lf_full.view_mut(),
            w, h, mult, &mut row_buf, None,
        );

        // Restricted path (same region for every scale).
        let mut hf_res = Array2::<f64>::zeros((h, w));
        let mut lf_res = Array2::<f64>::zeros((h, w));
        let mut img_res = img.clone();
        let mut row_buf2 = vec![0.0_f64; w];
        decompose_2d_bspline(
            img_res.view_mut(),
            hf_res.view_mut(),
            lf_res.view_mut(),
            w, h, mult, &mut row_buf2, Some(region),
        );

        for i in r0..=r1 {
            for j in c0..=c1 {
                assert_eq!(
                    hf_full[(i, j)].to_bits(),
                    hf_res[(i, j)].to_bits(),
                    "hf bit-exactness failed at ({i},{j}) mult={mult}"
                );
                assert_eq!(
                    lf_full[(i, j)].to_bits(),
                    lf_res[(i, j)].to_bits(),
                    "lf bit-exactness failed at ({i},{j}) mult={mult}"
                );
            }
        }
    }

    #[test]
    fn test_restricted_decompose_equals_full_in_region() {
        // Pseudo-random non-negative image with masked blocks. Covers three
        // scenarios: a central multi-scale mask (cumulative region, scales=3),
        // a border-touching mask, and a tiny mask. Use a large image so the
        // cumulative pad (16 for scales=3) leaves a region under half the frame.
        let (h, w) = (100usize, 100usize);
        let mut rng = StdRng::seed_from_u64(99);
        let n = Normal::new(5.0, 2.0).unwrap();
        let mut img = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                let v: f64 = n.sample(&mut rng); img[(i, j)] = v.abs();
            }
        }

        // Case A: central masked block, multi-scale region (scales=3).
        let mut mask = Array2::<bool>::from_elem((h, w), false);
        for i in 40..=49 {
            for j in 42..=50 {
                mask[(i, j)] = true;
            }
        }
        for mult in [1i32, 2, 4] {
            assert_decomp_eq_full_in_region(&img, &mask, 3, mult);
        }

        // Case B: border-touching mask (clamping of r0/c0/r1/c1).
        let mut mask_b = Array2::<bool>::from_elem((h, w), false);
        for i in 0..=4 {
            for j in 0..=4 {
                mask_b[(i, j)] = true;
            }
        }
        for mult in [1i32, 4] {
            assert_decomp_eq_full_in_region(&img, &mask_b, 3, mult);
        }

        // Case C: tiny region (a handful of masked pixels clustered so the wide
        // cumulative padding dominates the region, yet stays under half-frame).
        let mut mask_t = Array2::<bool>::from_elem((h, w), false);
        mask_t[(48, 48)] = true;
        mask_t[(48, 49)] = true;
        mask_t[(49, 48)] = true;
        for mult in [1i32, 4] {
            assert_decomp_eq_full_in_region(&img, &mask_t, 3, mult);
        }
    }

    #[test]
    fn test_decompose_region_none_full_unchanged() {
        // region=None must reproduce the legacy full decomposition: on a random
        // image hf+lf reconstructs `in` exactly (bit-for-bit) everywhere, and on
        // a constant image the full blur equals the constant.
        let (h, w) = (12usize, 14usize);
        let mut rng = StdRng::seed_from_u64(5);
        let n = Normal::new(4.0, 2.0).unwrap();
        let mut img = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                // Non-negative so the clamped low-pass never clips (and the
                // exact hf+lf == in reconstruction holds bit-for-bit).
                let v: f64 = n.sample(&mut rng); img[(i, j)] = v.abs();
            }
        }
        let mut hf = Array2::<f64>::zeros((h, w));
        let mut lf = Array2::<f64>::zeros((h, w));
        let mut row_buf = vec![0.0_f64; w];
        let mut img_c = img.clone();
        decompose_2d_bspline(
            img_c.view_mut(),
            hf.view_mut(),
            lf.view_mut(),
            w, h, 3, &mut row_buf, None,
        );
        for i in 0..h {
            for j in 0..w {
                assert_eq!(
                    (lf[(i, j)] + hf[(i, j)]).to_bits(),
                    img[(i, j)].to_bits(),
                    "region=None reconstruction must be exact at ({i},{j})"
                );
            }
        }

        // Constant image: lf equals the constant blur everywhere (hoist-safe).
        let cimg = Array2::<f64>::from_elem((10, 10), 2.0);
        let mut hf2 = Array2::<f64>::zeros((10, 10));
        let mut lf2 = Array2::<f64>::zeros((10, 10));
        let mut row_buf2 = vec![0.0_f64; 10];
        let mut cimg_c = cimg.clone();
        decompose_2d_bspline(
            cimg_c.view_mut(),
            hf2.view_mut(),
            lf2.view_mut(),
            10, 10, 2, &mut row_buf2, None,
        );
        assert!(lf2.iter().all(|&v| (v - 2.0).abs() < 1e-12), "full blur equals constant");
        assert!(hf2.iter().all(|&v| v == 0.0), "hf is zero on a constant image");
    }
}
