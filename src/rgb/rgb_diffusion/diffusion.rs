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

//! Multi-scale anisotropic diffusion over B-spline wavelets (heat PDE):
//! high-level pipeline orchestration (buffer ping-pong, scale derivation, and
//! the iteration loop). The actual per-pixel math lives in `kernel` and the
//! wavelet step in `bspline`.

use ndarray::{Array2, ArrayView2, ArrayViewMut2, NdFloat};
use std::cmp;
use super::kernel::{compute_anisotropy_factor, check_isotropy_mode, heat_pde_diffusion};
use super::bspline::{decompose_2d_bspline, equivalent_sigma_at_step, num_steps_to_reach_equivalent_sigma, B_SPLINE_SIGMA};
use super::regions::{decide_scoping, ScopedPlan};

const MAX_NUM_SCALES: usize = 10;
const KAPPA: f64 = 0.25;

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
/// * `plan` - Optional scoping plan (per-scale masked∪halo regions plus an
///   optional region-restricted forward decomposition). `None` keeps the legacy
///   full-image diffusion and the full-image decomposition (bit-for-bit).
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
    plan: Option<&ScopedPlan>,
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
            plan.and_then(|p| p.decomp_region),
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
            plan.map(|p| &p.regions.halo[scale][..]),
            mask.is_none(),
        );
    }
}

/// Parameters for diffusion algorithm configuration
///
/// Encapsulates all tunable parameters for the anisotropic diffusion
/// algorithm, including anisotropy weights, diffusion coefficients,
/// and scale parameters.
pub(super) struct ProcessArgs<T: NdFloat + Default> {
    pub(super) iterations: usize,
    pub(super) anisotropy_first: T,
    pub(super) anisotropy_second: T,
    pub(super) anisotropy_third: T,
    pub(super) anisotropy_fourth: T,
    pub(super) regularization: T,
    pub(super) variance_threshold: T,
    pub(super) radius_center: T,
    pub(super) radius: T,
    pub(super) first: T,
    pub(super) second: T,
    pub(super) third: T,
    pub(super) fourth: T,
    pub(super) sharpness: T,
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
pub(super) fn process_image<T: NdFloat + Default>(
    process_args: ProcessArgs<T>,
    image_in: &mut ArrayViewMut2<T>,
    mask: Option<ArrayView2<bool>>,
) -> Array2<T> {
    let mut image_out: Array2<T>;
    let mut temp_1: Array2<T>;
    let mut temp_2: Array2<T>;
    let mut lf_odd = Array2::<T>::zeros(image_in.dim());
    let mut lf_even = Array2::<T>::zeros(image_in.dim());

    let final_radius = process_args.radius + process_args.radius_center * T::from(2.0).unwrap();

    let iterations = cmp::max(process_args.iterations, 1);
    let diffusion_scales =
        num_steps_to_reach_equivalent_sigma(T::from(B_SPLINE_SIGMA).unwrap(), final_radius);
    let scales = diffusion_scales.clamp(1, MAX_NUM_SCALES);

    // Precompute the per-scale masked∪halo cell lists once. When a mask is
    // present, the working buffers are pre-seeded with a CLONE of the input:
    // unmasked pixels are immortal (= original) and, under scoping, far-field
    // unmasked pixels are never re-written by diffusion, so seeding once keeps
    // the final output (and next-iteration inputs) correct. When there is no
    // mask, keep the exact legacy zeros init (bit-for-bit unchanged).
    //
    // Scoping only pays off when the masked∪halo footprint is a small part of
    // the image. A mask that already covers much of the image (dense coverage,
    // or one whose halos merge to fill the frame) reduces no work, so we fall
    // back to the (fast) full-scan path and the legacy zeros init (bit-for-bit
    // the original behavior). `decide_scoping` bundles the masked-fraction and
    // largest-halo pre-checks together with the region-restricted forward
    // decomposition selection.
    let (h, w) = image_in.dim();
    let plan = decide_scoping(&mask, scales, h, w);

    if plan.is_some() {
        let base = image_in.to_owned();
        image_out = base.clone();
        temp_1 = base.clone();
        temp_2 = base;
    } else {
        image_out = Array2::<T>::zeros(image_in.dim());
        temp_1 = Array2::<T>::zeros(image_in.dim());
        temp_2 = Array2::<T>::zeros(image_in.dim());
    }

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
        // Ping-pong between ping-pong buffers. The source and destination must
        // be concrete and disjoint within each arm for the borrow checker: the
        // src is read once per iteration (input, then alternating buffers) while
        // the dest is written (image_out on the final pass, else the other
        // temp). Each arm independently selects both so their borrows are
        // provably non-overlapping (see the reference bindings above).
        if it == 0 {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_2_ref;
            }
            wavelets_process(
                &process_args, scales, image_in, temp_out,
                &mut lf_odd, &mut lf_even, &mut hf, zoom, &mask,
                plan.as_ref(), &mut row_buf,
            );
        } else if (it % 2) == 0 {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_2_ref;
            }
            wavelets_process(
                &process_args, scales, temp_1_ref, temp_out,
                &mut lf_odd, &mut lf_even, &mut hf, zoom, &mask,
                plan.as_ref(), &mut row_buf,
            );
        } else {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_1_ref;
            }
            wavelets_process(
                &process_args, scales, temp_2_ref, temp_out,
                &mut lf_odd, &mut lf_even, &mut hf, zoom, &mask,
                plan.as_ref(), &mut row_buf,
            );
        }
    }
    image_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rgb::rgb_diffusion::regions::{build_scoped_regions, compute_decomp_region};
    use crate::rgb::rgb_diffusion::fills::replace_masked_with_noise;
    use crate::test_utils::assert_delta;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

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

    #[test]
    fn test_process_image_empty_mask_returns_input() {
        // With an empty mask the scoped regions are empty and process_image
        // must return a copy of the input unchanged (the clone pre-seed).
        let mut img = Array2::<f64>::zeros((6, 6));
        for r in 0..6 {
            for c in 0..6 {
                img[(r, c)] = (r * 7 + c) as f64 * 0.1;
            }
        }
        let mask = Array2::<bool>::from_elem((6, 6), false);
        let pa = ProcessArgs {
            iterations: 3,
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
        let out = process_image(pa, &mut img.view_mut(), Some(mask.view()));
        assert_eq!(out, img);
    }

    #[test]
    fn test_wavelets_scoped_restricted_decomp() {
        // End-to-end: running wavelets_process with a SCOPED inverse pass and a
        // RESTRICTED forward decomposition must produce bit-identical output on
        // the masked∪halo cells compared with the same scoped pass but a FULL
        // (region=None) decomposition, while far-field unmasked cells keep their
        // pre-seed (input) value exactly.
        let (h, w) = (96usize, 96usize);
        let mut rng = StdRng::seed_from_u64(31);
        let n = Normal::new(2.0, 1.0).unwrap();
        let mut img = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                let v: f64 = n.sample(&mut rng); img[(i, j)] = v.abs();
            }
        }

        // Sparse mask: a central compact blob (so the cumulative region stays
        // well under half the frame, leaving a sizeable far field).
        let mut mask = Array2::<bool>::from_elem((h, w), false);
        for i in 40..=46 {
            for j in 38..=44 {
                mask[(i, j)] = true;
            }
        }

        let scales = 3usize;
        let the_mask: Option<ArrayView2<bool>> = Some(mask.view());

        let pa = ProcessArgs {
            iterations: 1,
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
            radius: 5.0,
            sharpness: 0.0,
        };

        // Run wavelets_process twice from identical pre-seeded buffers: once
        // with the forward decomposition restricted to the region and once with
        // a full (plan.decomp_region = None) decomposition.
        fn run(
            pa: &ProcessArgs<f64>,
            scales: usize,
            img: &Array2<f64>,
            the_mask: &Option<ArrayView2<bool>>,
            plan: &ScopedPlan,
        ) -> Array2<f64> {
            let (h, w) = img.dim();
            let mut lf_odd = Array2::<f64>::zeros((h, w));
            let mut lf_even = Array2::<f64>::zeros((h, w));
            let mut hf = (0..scales).map(|_| Array2::<f64>::zeros((h, w))).collect::<Vec<_>>();
            let mut reconstructed = img.clone(); // pre-seed === input (scoped semantics)
            let mut row_buf = vec![0.0_f64; w];
            wavelets_process(
                pa, scales,
                &mut img.clone().view_mut(),
                &mut reconstructed.view_mut(),
                &mut lf_odd, &mut lf_even, &mut hf,
                f64::from(1.0),
                the_mask,
                Some(plan),
                &mut row_buf,
            );
            reconstructed
        }
        // Both plans share identical regions; only the decomp region differs.
        let regions = build_scoped_regions(&mask.view(), scales);
        let decomp_region = compute_decomp_region(&mask.view(), scales, h, w).expect("some region");
        let plan_full = ScopedPlan {
            regions: build_scoped_regions(&mask.view(), scales),
            decomp_region: None,
        };
        let plan_restr = ScopedPlan {
            regions,
            decomp_region: Some(decomp_region),
        };
        let out_full = run(&pa, scales, &img, &the_mask, &plan_full);
        let out_restr = run(&pa, scales, &img, &the_mask, &plan_restr);

        // masked∪halo cells: bit-identical (restriction reproduces full decomp there).
        for &idx in plan_restr.regions.halo.iter().flat_map(|r| r.iter()) {
            let i = idx / w;
            let j = idx % w;
            assert_eq!(
                out_full[(i, j)].to_bits(),
                out_restr[(i, j)].to_bits(),
                "scoped masked∪halo diverged at ({i},{j})"
            );
        }

        // Far-field unmasked (outside the halo): stays at the input pre-seed.
        let in_halo = |i: usize, j: usize| {
            plan_restr.regions.halo.iter().any(|r| r.contains(&(i * w + j)))
        };
        for i in 0..h {
            for j in 0..w {
                if !in_halo(i, j) {
                    assert_eq!(out_restr[(i, j)].to_bits(), img[(i, j)].to_bits(),
                        "far-field unmasked must equal input at ({i},{j})");
                }
            }
        }
        // Sanity: the heavy masked branch actually ran (a masked blob pixel moved).
        assert!(mask[(43, 41)]);
        let delta = (out_restr[(43, 41)] - img[(43, 41)]).abs();
        assert!(delta > 1e-12, "masked pixel should have been diffused, delta={}", delta);
    }
}
