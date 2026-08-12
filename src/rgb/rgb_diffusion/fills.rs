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

//! All mask-seeding / initialization strategies for the inpainting pipeline and
//! their validation: noise seeding, boundary fill, and the radial stellar rise.

use ndarray::{Array2, ArrayView2, NdFloat};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, StandardNormal};
use super::regions::{bfs_nearest_value, find_components, Component};

/// Alpha weighting between the linear and quadratic terms of the radial-rise
/// lift: 1 is a pure quadratic (zero seam slope), 0 a pure cone (constant slope).
const RADIAL_RISE_ALPHA: f64 = 0.25;
/// Radius (px) over which the boundary-fill texture noise ramps to full strength.
const EDGE_BLEND_RAMP: f64 = 3.0;
/// Small positive floor for a non-positive standard deviation / noise samples.
const EPS: f64 = 1e-6;

/// Seed masked pixels with Gaussian noise.
///
/// Each masked pixel is replaced with a sample from a Gaussian whose mean and
/// standard deviation are both the original pixel value, giving the diffusion a
/// starting value. Unmasked pixels are left unchanged.
///
/// # Panics
/// Panics if any masked pixel has value `<= 0`, since that would give a
/// non-positive standard deviation for `Normal::new()`. Unlike
/// [`fill_masked_from_boundary`], this does not tolerate negative values, so
/// callers should first check with [`validate_noise_masked`], which raises a
/// `PyValueError` instead of panicking.
pub(super) fn replace_masked_with_noise<T: NdFloat + Default>(
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

/// Fill one connected component of masked pixels; see [`fill_masked_from_boundary`]
/// for the algorithm. Builds summed-area tables (count/sum/sum-of-squares) and a
/// multi-source BFS of the nearest unmasked value, both over the component's
/// bounding box padded by `radius`. Cost scales with the bounding box, not the
/// full image, so many small regions stay cheap.
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
    let eps = T::from(EPS).unwrap();

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
    let ramp_dist = T::from(EDGE_BLEND_RAMP).unwrap();

    // The per-local-window standard deviation is inflated near the mask boundary
    // where the window holds few valid unmasked samples, which would inject too
    // much noise there. Instead fill every masked pixel with the local mean plus
    // a noise term whose magnitude is a single robust (median) standard
    // deviation for the component (see [`fill_masked_from_boundary`]).
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
/// Each masked pixel's mean is the equal-weight average of the unmasked
/// neighbors within `radius`, computed in O(1) per pixel via summed-area tables.
/// Masked pixels whose `radius` window holds no unmasked neighbor fall back to
/// the nearest unmasked value from a multi-source BFS.
///
/// The added texture reproduces the background's natural per-pixel noise: every
/// masked pixel is filled with its local mean plus a Gaussian sample. Rather
/// than each pixel's own local standard deviation (which is inflated near the
/// mask edge, where the window holds few unmasked samples, and would leave a
/// mottled rim), a single robust median of the component's local standard
/// deviations is used for every pixel in that component. Only the outermost few
/// pixels are blended smoothly into the boundary — the noise is scaled to zero
/// there and ramps back to full strength within a couple of pixels.
///
/// Values are sampled as `Normal(mean, max(sigma, EPS))`, where `EPS` is a small
/// positive floor that prevents a non-positive standard deviation. Unlike
/// [`replace_masked_with_noise`], values may be negative without panicking, and
/// a single RNG is shared across the whole mask so the texture varies across the
/// region. Work is scoped per connected component, so cost grows with the masked
/// footprint rather than the whole image.
pub(super) fn fill_masked_from_boundary<T: NdFloat + Default>(
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
pub(super) fn fill_masked_radial_rise<T: NdFloat + Default>(
    image: ArrayView2<T>,
    mask: &ArrayView2<bool>,
    peak: T,
) -> Array2<T> {
    let (h, w) = image.dim();
    let one = T::from(1.0).unwrap();
    // Tuning knob: alpha=1 is a pure quadratic (zero initial seam slope),
    // alpha=0 a pure cone (constant slope). A small value (~0.25) matches the
    // outer wing gradient. Core brightness stays peak*depth for any alpha.
    let a = T::from(RADIAL_RISE_ALPHA).unwrap();

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

/// Validate that every masked pixel has a positive value, as required by the
/// `init_method="noise"` seeding (which uses the masked value both as the mean
/// and the standard deviation of a Gaussian). Returns a `PyValueError` naming
/// the offending pixel instead of panicking inside `Normal::new(...).unwrap()`.
pub(super) fn validate_noise_masked(image: ArrayView2<f64>, mask: &ArrayView2<bool>) -> PyResult<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_delta;

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
}
