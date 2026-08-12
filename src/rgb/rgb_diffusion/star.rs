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

//! Saturated-star colour reconstruction: flux-conserving per-band colour
//! estimation plus a chromaticity->Oklab lookup curve.

use ndarray::{Array2, Array3, ArrayView2};
use crate::rgb::color_spaces::linear_rgb_to_oklab;
use super::regions::{bfs_nearest_value, find_components, seed_nearest_dist, Component};

/// Number of samples in the constant-chromaticity lookup curve.
const COLOUR_CURVE_N: usize = 512;
/// Lower bound of the `t` scan range for building the colour curve.
const COLOUR_CURVE_T_MIN: f64 = 1e-9;
/// Upper bound of the `t` scan range for building the colour curve.
const COLOUR_CURVE_T_MAX: f64 = 1e3;
/// Minimum integrated linear-RGB flux below which a star's colour is "just noise".
const MIN_FLUX: f64 = 1e-6;
/// Small denominator floor when interpolating the colour lookup curve.
const LOOKUP_DENOM_EPS: f64 = 1e-12;
/// How far the colour curve is kept beyond the queried L range (to tighten the lookup).
const COLOUR_CURVE_L_PAD: f64 = 0.05;

/// Estimate a star's intrinsic colour by integrating its per-band linear-RGB
/// flux over a disk around the component centroid.
///
/// The disk has radius `radius` around the centroid. The background sky is the
/// per-channel median of the ring between `bg_inner` and `bg_outer` about the
/// same centre, and this median is subtracted from every pixel before summing;
/// only unmasked pixels contribute. Returns `None` if the total subtracted flux
/// is below `MIN_FLUX`, meaning the region is too faint to trust (dominated by
/// noise) and the component should be left untouched.
///
/// `linear_rgb` is a precomputed (h, w, 3) linear-RGB buffer, materialised once
/// per `reconstruct_star_color` call (not once per component) by the caller and
/// shared here across all stars.
fn estimate_star_linear_colour(
    linear_rgb: &Array3<f64>,
    mask: &ArrayView2<bool>,
    comp: &Component,
    radius: f64,
    bg_inner: f64,
    bg_outer: f64,
) -> Option<[f64; 3]> {
    let (h, w, _) = linear_rgb.dim();
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
            bg_r.push(linear_rgb[(i, j, 0)]);
            bg_g.push(linear_rgb[(i, j, 1)]);
            bg_b.push(linear_rgb[(i, j, 2)]);
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
            s[0] += (linear_rgb[(i, j, 0)] - bg[0]).max(0.0);
            s[1] += (linear_rgb[(i, j, 1)] - bg[1]).max(0.0);
            s[2] += (linear_rgb[(i, j, 2)] - bg[2]).max(0.0);
        }
    }

    // If the total flux is negligible the "colour" is just noise; refuse.
    if s[0] + s[1] + s[2] < MIN_FLUX {
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
    let n = COLOUR_CURVE_N;
    // Rough scan range in t; cover a wide range of brightness. The per-pixel
    // query interpolates, so exact extent mostly just needs to bound L.
    let t_min = COLOUR_CURVE_T_MIN;
    let t_max = COLOUR_CURVE_T_MAX;
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(n);
    for k in 0..n {
        let t = t_min * (t_max / t_min).powf(k as f64 / (n as f64 - 1.0));
        let rgb = [u[0] * t, u[1] * t, u[2] * t];
        pts.push(linear_rgb_to_oklab(rgb));
    }
    // Keep only points spanning the queried L range to tighten the lookup.
    let mut pts: Vec<[f64; 3]> = pts
        .into_iter()
        .filter(|p| {
            p[0] >= l_min - COLOUR_CURVE_L_PAD && p[0] <= l_max + COLOUR_CURVE_L_PAD
        })
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
    let f = (lp - p0[0]) / (p1[0] - p0[0]).max(LOOKUP_DENOM_EPS);
    [p0[1] + f * (p1[1] - p0[1]), p0[2] + f * (p1[2] - p0[2])]
}

/// Reconstruct the `a`/`b` channels of saturated stars.
///
/// See the `reconstruct_star_color` pyo3 wrapper for the full docstring. This
/// fills each masked component with a constant-chromaticity colour: the colour
/// is estimated by integrating conserved per-band linear-RGB flux, and each
/// masked pixel's `(a, b)` is set from that colour at the pixel's reconstructed
/// lightness `L`.
pub(super) fn fill_star_colour(
    a_out: &mut Array2<f64>,
    b_out: &mut Array2<f64>,
    l_img: &ArrayView2<f64>,
    a_img: &ArrayView2<f64>,
    b_img: &ArrayView2<f64>,
    linear_rgb: &Array3<f64>,
    mask: &ArrayView2<bool>,
    radius: f64,
    bg_inner: f64,
    bg_outer: f64,
    blend: f64,
) {
    let (h, w) = mask.dim();
    // `blend <= 0.0` disables the boundary blend entirely (each masked pixel is
    // painted with the model colour, w = 1.0); a positive value is the distance
    // over which the model colour blends toward the nearest unmasked pixel.
    let blend_enabled = blend > 0.0;

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
        let Some(u) = estimate_star_linear_colour(linear_rgb, mask, &comp, radius, bg_inner, bg_outer)
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
            let w = if !blend_enabled {
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

#[cfg(test)]
mod tests {
    use super::*;

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
