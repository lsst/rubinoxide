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

//! High-level inpainting pipeline: split the image into scales, diffuse the
//! high-frequency parts, and rebuild it, reusing two working buffers across
//! scales so nothing is re-allocated. The per-pixel diffusion math lives in
//! `kernel` and the wavelet decomposition step in `bspline`.

use ndarray::{Array2, ArrayView2, ArrayViewMut2, NdFloat};
use std::cmp;
use super::kernel::{compute_anisotropy_factor, check_isotropy_mode, heat_pde_diffusion};
use super::bspline::{decompose_2d_bspline, equivalent_sigma_at_step, num_steps_to_reach_equivalent_sigma, B_SPLINE_SIGMA};
use super::regions::{decide_scoping, ScopedPlan};

const MAX_NUM_SCALES: usize = 10;
const KAPPA: f64 = 0.25;

/// Diffuse one round of the multi-scale decomposition.
///
/// Decomposes the image into high/low frequency parts one scale at a time,
/// then, in reverse order, diffuses each high-frequency part and rebuilds the
/// image by combining it with the low-frequency residual. Work alternates
/// between two low-frequency buffers (`lf_odd`/`lf_even`) across the forward
/// scales and between two reconstruction buffers (`residual`/`temp`) across the
/// reverse passes, so no new buffers are allocated per scale.
///
/// # Arguments
/// * `process_args` - Diffusion parameters (see [`ProcessArgs`]).
/// * `scales` - Number of decomposition levels.
/// * `input` - Image to diffuse; read at the start (first scale).
/// * `reconstructed` - Final diffused image.
/// * `lf_odd`, `lf_even` - The two low-frequency buffers, alternated per scale.
/// * `hf` - High-frequency components, one array per scale.
/// * `zoom` - Extra radius scale factor per scale.
/// * `mask` - Optional boolean mask; when present only these pixels diffuse.
/// * `plan` - Optional scoping plan (see `regions`); `None` keeps the legacy
///   full-image behavior, identical bit-for-bit.
/// * `row_buf` - Scratch buffer for the B-spline passes.
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

/// Tunable parameters for the diffusion algorithm.
///
/// The raw internal knobs; each public Python function maps one of its
/// arguments onto each field (see `mod.rs` for the defaults and meanings).
pub(super) struct ProcessArgs<T: NdFloat + Default> {
    /// Number of whole diffusion passes over the image.
    pub(super) iterations: usize,
    /// Edge-preservation strength for the first directional derivative.
    pub(super) anisotropy_first: T,
    /// Strength for the Laplacian-based second directional derivative.
    pub(super) anisotropy_second: T,
    /// Strength for the third directional derivative.
    pub(super) anisotropy_third: T,
    /// Strength for the fourth directional derivative.
    pub(super) anisotropy_fourth: T,
    /// Regularization exponent: sets the minimum variance used for numerical
    /// stability.
    pub(super) regularization: T,
    /// Minimum variance added to the diffusion computation to avoid division
    /// by zero.
    pub(super) variance_threshold: T,
    /// Radius at which diffusion is strongest.
    pub(super) radius_center: T,
    /// Diffusion radius; wider values use more decomposition scales and diffuse
    /// more.
    pub(super) radius: T,
    /// Diffusion coefficient 1 (weighted by position).
    pub(super) first: T,
    /// Diffusion coefficient 2.
    pub(super) second: T,
    /// Diffusion coefficient 3.
    pub(super) third: T,
    /// Diffusion coefficient 4.
    pub(super) fourth: T,
    /// Sharpness adjustment: positive keeps peaks, negative flattens them.
    pub(super) sharpness: T,
}

/// Run the full diffusion pipeline on an image.
///
/// This is the top-level entry point. It works out how many decomposition
/// scales are needed from the requested radius, then runs [`wavelets_process`]
/// once per iteration (at least once), alternating between two working buffers
/// between iterations and writing the final result into `image_out`.
///
/// # Buffer initialization
/// When scoping is used or the mask is empty (`decide_scoping` returns a plan),
/// the working buffers are pre-filled with a copy of the input, so unmasked
/// pixels keep their original values (an empty mask then returns the input
/// unchanged). Otherwise — no mask, or a mask too dense to make scoping
/// worthwhile — the buffers start at zero, matching the original behavior
/// exactly.
///
/// # Scoping
/// When a mask is present, `decide_scoping` picks whether restricting the work
/// to the mask (plus a surrounding halo) is worthwhile. If it is not — a mask
/// that already covers most of the image saves no work — the plain full-image
/// path is used instead, leaving results unchanged.
///
/// # Arguments
/// * `process_args` - Diffusion parameters (see [`ProcessArgs`]).
/// * `image_in` - Input image (read at the first iteration).
/// * `mask` - Optional boolean mask of pixels to diffuse.
///
/// # Returns
/// The diffused image.
///
/// # Notes
/// `iterations` is clamped to at least 1 so one diffusion pass always runs.
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

    // Decide the scoping plan once. Precompute the per-scale masked∪halo cell
    // lists so the scoped path reuses them each iteration (see `regions`).
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
        // Alternate between the two working buffers so the source and
        // destination are always distinct arrays for the borrow checker: the
        // source is read once per iteration (input, then alternating buffers)
        // while the destination is written (`image_out` on the final pass,
        // else the other working buffer). Each arm picks both explicitly so
        // their borrows never overlap (see the reference bindings above).
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
