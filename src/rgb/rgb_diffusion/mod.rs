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

/*!
# `rgb_diffusion` — inpainting & saturated-star colour reconstruction

## Overview

This subtree implements a Rust+PyO3 diffusion-based inpainting engine and a
flux-conserving saturated-star colour reconstructor exposed to Python as three
functions on the `rgb` extension module. The heavy lifting is split into six
submodules by strict concern: low-level PDE kernel math, B-spline wavelet
decomposition, high-level pipeline orchestration, masked-region connectivity
scoping, mask-value seeding, and colour reconstruction. All modules below except
`regions` are internal (`pub(super)`); only this file exposes `#[pyfunction]`
wrappers, and `src/rgb.rs` re-exports exactly the three functions below.

## Public API (the only external surface)

- `diffuse_gray_image` — apply multi-scale anisotropic (heat-PDE) diffusion over
  B-spline wavelets to an entire grayscale image.
- `inpaint_mask` — replace masked pixels with a seed (boundary-fill / noise /
  none / radial-rise), then diffuse the mask to seamlessly inpaint the region.
- `reconstruct_star_color` — estimate each saturated star's colour by integrating
  conserved per-band linear-RGB flux and paint the masked a/b channels at the
  reconstructed lightness.

## The two high-level pipelines

1. **Inpainting pipeline** (`inpaint_mask`):
   `inpaint_mask` -> `fills::fill_*` / `replace_masked_with_noise` / `validate_noise_masked`
   (seed the mask) -> `diffusion::process_image` -> `diffusion::wavelets_process`
   -> `kernel::heat_pde_diffusion` (the diffusion step) and
   `bspline::decompose_2d_bspline` (the wavelet step), with
   `regions::decide_scoping` deciding whether to restrict work to the mask.

2. **Star-colour pipeline** (`reconstruct_star_color`):
   `reconstruct_star_color` -> `star::fill_star_colour` -> `star::estimate_star_linear_colour`
   (flux integration) + `star::build_colour_curve` / `star::lookup_ab_curve`
   (chromaticity->Oklab lookup), reusing `regions::find_components` /
   `bfs_nearest_value` / `seed_nearest_dist`.

## File-by-file map

| File | What lives here | Why it is separate |
|------|-----------------|--------------------|
| `kernel.rs` | `IsotropyType`, `RotationMatrix`, `Flat3Matrix`, `find_gradients`, `isotrop_laplacian`, `rotation_matrix_*`, `build_matrix`, `compute_kernel`, `diffusion_compute`, `CellIter`, `heat_pde_diffusion`, `compute_anisotropy_factor`, `check_isotropy_mode`, `H` | Pure per-pixel/neighbourhood PDE math; no orchestration, no allocation policy, no filter logic. The bit-exact scoped/full-scan worker lives here. |
| `bspline.rs` | `_bspline_vertical_pass`, `_bspline_horizontal_tap`, `decompose_2d_bspline`, `equivalent_sigma_at_step`, `num_steps_to_reach_equivalent_sigma`, `B_SPLINE_FILTER_F64`, `B_SPLINE_SIGMA` | Self-contained separable B-spline wavelet decomposition and its scale math; all filter constants colocated. |
| `diffusion.rs` | `ProcessArgs`, `wavelets_process`, `process_image`, `MAX_NUM_SCALES`, `KAPPA` | The high-level inpainting pipeline: buffer ping-pong, scale derivation, iteration loop, and the call into kernel/bspline. |
| `regions.rs` | `Component`, `ScopedRegions`, `ScopedPlan`, `find_components`, `bfs_nearest_value`, `seed_nearest_dist`, `compute_decomp_region`, `build_scoped_regions`, `decide_scoping`, `NEIGHBORS` | Connected-component / BFS / scoping primitives shared by fills, star and diffusion. |
| `fills.rs` | `fill_component`, `replace_masked_with_noise`, `fill_masked_from_boundary`, `fill_masked_radial_rise`, `validate_noise_masked` + their tuning constants | All mask-seeding / initialization strategies and their validation. |
| `star.rs` | `fill_star_colour`, `estimate_star_linear_colour`, `build_colour_curve`, `lookup_ab_curve` + their constants | Chromaticity-curve-based saturated-star colour reconstruction only. |

## Where to start reading

Start in `mod.rs`: read the three wrapper signatures (the public contract), then
the inpainting docstrings. Next read `diffusion.rs::process_image` (the
orchestration loop), then `kernel.rs::heat_pde_diffusion` / `diffusion_compute`
(the bit-exact heavy branch), then `bspline.rs::decompose_2d_bspline` (the
wavelet step), and finally `regions.rs` for scoping. For star colour, read
`star.rs::fill_star_colour` and its three helpers.

## Dependency / call-graph diagram

```
                  mod.rs  (pyfunction wrappers)
        ┌───────────┼───────────────┬──────────────────┐
        │           │               │                  │
 diff_gray_image   inpaint_mask   (seed)        reconstruct_star_color
        │           │    │          │                  │
        │           │    │   fills::{fill_masked_*,   │
        │           │    │   replace_masked_with_noise,│
        │           │    │   validate_noise_masked}   │
        └───────────┴───► diffusion::{                star::fill_star_colour
          diffusion::process_image   process_image,   ├─ estimate_star_linear_colour
               │                     wavelets_process}├─ build_colour_curve
               │                        │             └─ lookup_ab_curve
               │                  ┌─────┴───────┐
               │             kernel::        bspline::
               │             heat_pde_       decompose_2d_bspline /
               │             diffusion       equivalent_sigma* /
               │                            num_steps_to_reach*
               │         regions::{decide_scoping, ScopedPlan, compute_decomp_region}
               │                   (shared by diffusion, fills, star)
               └────────────────────────────────────────────────

Submodule dependency edges:
  mod.rs  -> diffusion, fills, star, regions (via use)
  diffusion -> kernel, bspline, regions
  kernel  -> (leaf: no deps)
  bspline -> (leaf: no deps)
  fills   -> regions
  star    -> regions
  regions -> (leaf: shared primitives)
```
*/
extern crate openblas_src;
mod kernel;
mod bspline;
mod diffusion;
mod regions;
mod fills;
mod star;

use log;
use ndarray::Array3;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use crate::rgb::color_spaces::oklab_to_linear_rgb;
use diffusion::{process_image, ProcessArgs};
use fills::{fill_masked_from_boundary, fill_masked_radial_rise, replace_masked_with_noise, validate_noise_masked};
use star::fill_star_colour;

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

    // Empty-mask short-circuit: nothing to inpaint. The unmasked passthrough is
    // the exact identity (a no-op reconstruction), so returning an owned copy of
    // the input is cheaper and matches the previous behavior.
    if !mask_array.iter().any(|&b| b) {
        return Ok(array.to_pyarray(py));
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
    // flux integration per component reuses it instead of re-converting. The
    // caller-supplied cube (pyo3 keyword `linear_rgb`) is consumed first here;
    // the owned working buffer is `linear_rgb_buf` to avoid shadowing it.
    let mut linear_rgb_buf: Array3<f64>;
    if let Some(cube) = linear_rgb {
        let (ch, cw, channels) = cube.as_array().dim();
        if channels != 3 {
            return Err(PyValueError::new_err(format!(
                "linear_rgb must have exactly 3 channels (R,G,B) along the last axis; got {channels}"
            )));
        }
        if (ch, cw) != (h, w) {
            return Err(PyValueError::new_err(format!(
                "linear_rgb shape ({ch}x{cw}x{channels}) must match L/a/b/mask shape ({h}x{w})"
            )));
        }
        if cube.as_array().iter().any(|&x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "linear_rgb must be finite (rejecting NaN/inf)",
            ));
        }
        linear_rgb_buf = cube.as_array().to_owned();
    } else {
        linear_rgb_buf = Array3::<f64>::zeros((h, w, 3));
        for i in 0..h {
            for j in 0..w {
                let [r, g, b] = oklab_to_linear_rgb([l_img[(i, j)], a_img[(i, j)], b_img[(i, j)]]);
                linear_rgb_buf[[i, j, 0]] = r;
                linear_rgb_buf[[i, j, 1]] = g;
                linear_rgb_buf[[i, j, 2]] = b;
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
        &linear_rgb_buf,
        &mask_array,
        radius,
        bg_inner,
        bg_outer,
        blend,
    );

    Ok((a_out.to_pyarray(py), b_out.to_pyarray(py)))
}
