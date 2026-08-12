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

//! Region and connectivity primitives that let the diffusion and B-spline
//! decomposition touch only the image cells they actually need.
//!
//! # Scoping
//!
//! "Scoping" means restricting expensive work to the masked image cells plus the
//! cells around them. A masked cell runs the heavy diffusion branch; the
//! unmasked cells next to it (within a radius called `mult`) just supply the
//! correct neighbour values that masked cell reads. Together these two groups
//! are a **scoped region** (a mask "plus its halo"). Cells far from any masked
//! cell are never read and are left at their pre-seeded values, so leaving them
//! untouched is exact. The helpers in this module find masked regions
//! (`find_components`) and build these scoped regions at every scale
//! (`build_scoped_regions` / `decide_scoping`).
//!
//! # Distances and neighbourhoods
//!
//! Everything here works on the 8-connected neighbourhood ([`NEIGHBORS`]): a
//! cell's eight neighbours (up/down, left/right, and the four diagonals). The
//! distance between two cells is the smallest number of 8-neighbour steps needed
//! to reach one from the other (a Chebyshev / max-norm distance). A multi-source
//! BFS seeded at every masked cell computes these distances for all cells at
//! once, growing the distance by 1 per neighbour step.

use ndarray::{Array2, ArrayView2, NdFloat};
use std::collections::VecDeque;

/// Masked-fraction above which mask scoping is deemed not to reduce work, so
/// `process_image` falls back to the legacy full-scan / zeros-seed path.
const MAX_MASKED_FRACTION: f64 = 0.2;
/// Largest masked-plus-halo fraction above which scoping is deemed not to
/// reduce work (halos merge to fill the frame), so we fall back to the
/// full-scan path.
const MAX_HALO_FRACTION: f64 = 0.5;
/// Decomposition-region area (as a fraction of the image) above which scoping
/// the forward B-spline does not pay, so it falls back to a full decomposition.
const MAX_DECOMP_REGION_FRACTION: f64 = 0.5;

/// 8-connected neighbour offsets (Chebyshev neighbourhood), shared by the
/// component flood-fill and the nearest-value BFS.
pub(super) const NEIGHBORS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// A connected component of masked pixels, together with its bounding box.
pub(super) struct Component {
    pub(super) r0: usize,
    pub(super) r1: usize,
    pub(super) c0: usize,
    pub(super) c1: usize,
    pub(super) coords: Vec<(usize, usize)>,
}

/// Per-scale scoped regions for diffusion: the masked cells plus their halo.
///
/// See the module doc for what a scoped region (mask plus halo) is. For each
/// scale `s`, `halo[s]` lists, as linear indices (`idx = r*width + c`), every
/// cell within distance `mult = 1<<s` of some masked cell — which includes the
/// masked cells themselves. Far-field unmasked cells appear in no halo because
/// no masked pixel reads them.
pub(super) struct ScopedRegions {
    pub(super) width: usize,
    /// `halo[scale]` = masked cells plus their distance-`1<<scale` (Chebyshev /
    /// max-norm) halo, as linear indices (`idx = r*width + c`).
    pub(super) halo: Vec<Vec<usize>>,
}

/// Bounding box over which the forward B-spline decomposition must run so every
/// cell the diffusion reads reproduces the full-image result exactly.
///
/// Each scale's input is a blurred version of the previous one, so a cell's
/// value at the coarsest scale depends on cells whose distance accumulates
/// across scales. The separable 5-tap B-spline spans `2 * mult_s = 2 * 2^s`
/// pixels on each side at scale `s`, so the total half-width a cell's value can
/// depend on is `max_mult + Σ_{s=1}^{scales-1} 2*mult_s` (with
/// `max_mult = 1<<(scales-1)`). Padding the masked bounding box by this
/// cumulative radius reproduces the full-image low/high frequencies at every
/// cell the diffusion reads, so the inverse stage and output are correct.
///
/// Returns `None` (fall back to the full decomposition) when the resulting
/// region would cover more than half the image (scoping would not reduce work),
/// or when the mask is empty (nothing to inpaint).
///
/// # Arguments
/// * `mask` - Boolean mask; True cells are the masked (read) region.
/// * `scales` - Number of wavelet decomposition levels.
/// * `h`, `w` - Image height and width.
///
/// # Returns
/// * `Option<(usize, usize, usize, usize)>` - inclusive `(r0, r1, c0, c1)`.
pub(super) fn compute_decomp_region(
    mask: &ArrayView2<bool>,
    scales: usize,
    h: usize,
    w: usize,
) -> Option<(usize, usize, usize, usize)> {
    // Cumulative padding: max_mult + Σ_{s=1}^{scales-1} 2*mult_s.
    let max_mult = 1usize << scales.saturating_sub(1);
    let mut pad = max_mult;
    for s in 1..scales {
        pad += 2usize << s;
    }

    // Bounding box of masked cells.
    let mut mr0 = h;
    let mut mr1 = 0usize;
    let mut mc0 = w;
    let mut mc1 = 0usize;
    let mut any = false;
    for i in 0..h {
        for j in 0..w {
            if mask[(i, j)] {
                any = true;
                mr0 = mr0.min(i);
                mr1 = mr1.max(i);
                mc0 = mc0.min(j);
                mc1 = mc1.max(j);
            }
        }
    }
    if !any {
        return None;
    }

    let r0 = mr0.saturating_sub(pad);
    let r1 = (mr1 + pad).min(h - 1);
    let c0 = mc0.saturating_sub(pad);
    let c1 = (mc1 + pad).min(w - 1);

    let area = (r1 - r0 + 1) * (c1 - c0 + 1);
    if (area as f64) > MAX_DECOMP_REGION_FRACTION * (h * w) as f64 {
        return None;
    }
    Some((r0, r1, c0, c1))
}

/// Build the per-scale scoped regions (masked cells plus halo) for diffusion.
///
/// Uses a single multi-source 8-connected BFS seeded at every masked cell
/// (distance 0) to compute each cell's distance to the nearest masked cell, as
/// described in the module doc. For each scale `s`, `halo[s]` holds every cell
/// with `dist <= 1<<s`, which necessarily includes the masked cells and their
/// halo.
///
/// The BFS is bounded to `max_mult = 1 << (scales-1)`, the largest radius any
/// halo needs, so cells beyond it are never expanded. This keeps the cost
/// proportional to the (usually tiny) scoped footprint rather than the full
/// image, while producing the exact same halo sets as a whole-image scan.
///
/// An empty mask yields empty halo rows (the caller short-circuits before
/// reaching the diffusion, but the structure stays valid for tests).
pub(super) fn build_scoped_regions(mask: &ArrayView2<bool>, scales: usize) -> ScopedRegions {
    let (height, width) = mask.dim();
    let max_mult = 1usize << (scales.saturating_sub(1));

    let mut dist = Array2::<u32>::from_elem((height, width), u32::MAX);
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..height {
        for c in 0..width {
            if mask[(r, c)] {
                dist[(r, c)] = 0;
                queue.push_back((r, c));
            }
        }
    }

    let mut halo = Vec::with_capacity(scales);
    if queue.is_empty() {
        // Empty mask: no masked cells -> every halo row is empty.
        for _ in 0..scales {
            halo.push(Vec::new());
        }
        return ScopedRegions { width, halo };
    }

    // Multi-source BFS over 8-neighbors tracking the Chebyshev distance to the
    // nearest masked cell, bounded to distances <= max_mult. Each cell in range
    // is recorded once with its distance.
    let mut in_range: Vec<(usize, u32)> = Vec::new();
    while let Some((ci, cj)) = queue.pop_front() {
        let d = dist[(ci, cj)];
        in_range.push((ci * width + cj, d));
        if d >= max_mult as u32 {
            continue; // beyond the largest needed radius: never in any halo
        }
        for (di, dj) in NEIGHBORS {
            let ni = ci as i32 + di;
            let nj = cj as i32 + dj;
            if ni < 0 || nj < 0 || ni >= height as i32 || nj >= width as i32 {
                continue;
            }
            let (ni, nj) = (ni as usize, nj as usize);
            if dist[(ni, nj)] == u32::MAX {
                dist[(ni, nj)] = d + 1;
                queue.push_back((ni, nj));
            }
        }
    }

    for s in 0..scales {
        let mult = 1usize << s;
        halo.push(
            in_range
                .iter()
                .filter(|&&(_, d)| (d as usize) <= mult)
                .map(|&(idx, _)| idx)
                .collect(),
        );
    }

    ScopedRegions { width, halo }
}

/// A single scoping decision bundling the per-scale scoped regions with the
/// (optional) bounding box for a region-restricted forward decomposition.
pub(super) struct ScopedPlan {
    pub(super) regions: ScopedRegions,
    pub(super) decomp_region: Option<(usize, usize, usize, usize)>,
}

/// Decide whether masked scoping is worthwhile and, if so, build the plan.
///
/// Returns `None` (fall back to the legacy full-scan / full-decomposition path)
/// when masking is not present, covers too much of the image (more than
/// [`MAX_MASKED_FRACTION`]), or yields halos that merge to fill the frame (more
/// than [`MAX_HALO_FRACTION`]). An *empty* (all-false) mask returns `Some` with
/// empty regions and `decomp_region = None` so `process_image` keeps the
/// clone-vs-zeros seeding branch (empty mask -> clone pre-seed -> output equals
/// the input).
///
/// # Arguments
/// * `mask` - Optional boolean mask to scope against; `None` disables scoping.
/// * `scales` - Number of wavelet decomposition levels.
/// * `h`, `w` - Image height and width.
///
/// # Returns
/// * `Option<ScopedPlan>` - `None` when scoping is not worthwhile; otherwise the
///   bundled regions + optional region-restricted decomposition.
pub(super) fn decide_scoping(
    mask: &Option<ArrayView2<bool>>,
    scales: usize,
    h: usize,
    w: usize,
) -> Option<ScopedPlan> {
    let m = mask.as_ref()?;

    // Cheap masked-fraction pre-check: dense coverage means scoping cannot help.
    let masked_fraction = m.iter().filter(|&&b| b).count() as f64 / (h * w) as f64;
    if masked_fraction > MAX_MASKED_FRACTION {
        return None;
    }

    let regions = build_scoped_regions(m, scales);
    let total = (h * w) as f64;
    let largest = regions.halo.iter().map(|r| r.len()).max().unwrap_or(0) as f64;
    if total > 0.0 && largest / total > MAX_HALO_FRACTION {
        return None;
    }

    let decomp_region = compute_decomp_region(m, scales, h, w);
    Some(ScopedPlan { regions, decomp_region })
}

/// Find all 8-connected components of masked (True) pixels.
///
/// Returns one `Component` per connected masked region including its tight
/// bounding box. This reads the whole mask (O(H*W)) but stores only the masked
/// pixel coordinates, so downstream work can be scoped to each component.
pub(super) fn find_components(mask: &ArrayView2<bool>) -> Vec<Component> {
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

/// Multi-source 8-connected BFS that fills every cell whose `dist == u32::MAX`
/// with the value and (Chebyshev) distance of its nearest seed cell.
///
/// Callers pre-initialize `dist[(i, j)] = 0` and `values[(i, j)]` for the seed
/// cells (e.g. unmasked pixels), then call this once to propagate the seed
/// values inward. Cells with no reachable seed keep `dist == u32::MAX`.
pub(super) fn bfs_nearest_value<T: NdFloat + Default>(values: &mut Array2<T>, dist: &mut Array2<u32>) {
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
pub(super) fn seed_nearest_dist(dist: &mut Array2<u32>, mask: &ArrayView2<bool>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
    fn test_build_scoped_regions() {
        // A single masked pixel on a non-square (5x3) buffer: check that the
        // mult=1 halo includes the masked cell and a Chebyshev distance-1
        // neighbor, excludes a distance-2 cell, and the mult=2 halo re-includes
        // it. Also verify the linear-index round trip decodes correctly.
        let (h, w) = (5usize, 3usize);
        let mut mask = Array2::<bool>::from_elem((h, w), false);
        mask[(2, 1)] = true; // single masked cell
        let regions = build_scoped_regions(&mask.view(), 2);

        assert_eq!(regions.width, w);
        assert_eq!(regions.halo.len(), 2);

        // Round trip: decode(encode(r, c)) == (r, c).
        for &idx in regions.halo[0].iter().chain(regions.halo[1].iter()) {
            let r = idx / w;
            let c = idx % w;
            assert_eq!(r * w + c, idx, "linear-index round trip failed for {idx}");
        }

        // Masked cell (dist 0) always included.
        assert!(regions.halo[0].contains(&(2 * w + 1)));
        // Chebyshev distance-1 neighbor (e.g. (1,1)) included at mult=1.
        assert!(regions.halo[0].contains(&(1 * w + 1)));
        // Distance-2 cell (4,1) excluded at mult=1, re-included at mult=2.
        assert!(!regions.halo[0].contains(&(4 * w + 1)));
        assert!(regions.halo[1].contains(&(4 * w + 1)));

        // An all-False mask must produce empty halo rows and a valid width.
        let empty = Array2::<bool>::from_elem((4, 6), false);
        let eregions = build_scoped_regions(&empty.view(), 3);
        assert_eq!(eregions.width, 6);
        assert_eq!(eregions.halo.len(), 3);
        assert!(eregions.halo.iter().all(|h| h.is_empty()));
    }

    #[test]
    fn test_decide_scoping_gates() {
        let (h, w) = (20usize, 20usize);

        // No mask -> None.
        assert!(decide_scoping(&None, 3, h, w).is_none());

        // Empty (all-false) mask -> Some(empty plan) with no decomp region
        // (must NOT be None, to keep the clone-vs-zeros seeding branch).
        let empty = Array2::<bool>::from_elem((h, w), false);
        let plan = decide_scoping(&Some(empty.view()), 3, h, w).expect("empty mask must yield a plan");
        assert!(plan.regions.halo.iter().all(|r| r.is_empty()));
        assert_eq!(plan.decomp_region, None);

        // Fully-masked -> None (masked fraction exceeds the gate).
        let full = Array2::<bool>::from_elem((h, w), true);
        assert!(decide_scoping(&Some(full.view()), 3, h, w).is_none());

        // Sparse central mask on a large-enough image (the cumulative
        // decomposition pad must stay well under half the frame): Some with a
        // decomp region.
        let (bh, bw) = (80usize, 80usize);
        let mut sparse = Array2::<bool>::from_elem((bh, bw), false);
        for r in 36..=43 {
            for c in 36..=43 {
                sparse[(r, c)] = true;
            }
        }
        let plan = decide_scoping(&Some(sparse.view()), 3, bh, bw).expect("sparse mask must yield a plan");
        assert!(plan.decomp_region.is_some());
    }
}
