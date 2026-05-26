# For copyright information see the COPYRIGHT file included in the top-level
# directory of this distribution.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#      1. Redistributions of source code must retain the included copyright
#         notice, this list of conditions and the following disclaimer.

#      2. Redistributions in binary form must reproduce the included copyright
#         notice, this list of conditions and the following disclaimer in the
#         documentation and/or other materials provided with the distribution.

#      3. Neither the names of the copyright holders nor the names of their
#         contributors may be used to endorse or promote products derived from
#         this software without specific prior written permission.

#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

"""
Comprehensive comparison tests for the Rust diff kernel (`DiffKernel` and
`DiffKernelF32`) against a Python reference implementation.

The Python implementations below faithfully reproduce the steps of the Rust
`solve_diff_kernel` and `apply_kernel` methods so we can assert numerical
equivalence between the two code paths.
"""

import unittest

import numpy as np
from scipy.signal import fftconvolve

from lsst.utils.tests import TestCase, init

# ---------------------------------------------------------------------------
# Import the Rust-compiled symbols
# ---------------------------------------------------------------------------
from lsst.rubinoxide._rubinoxide import difference_kernel as dk

DiffKernel = dk.DiffKernel
generate_gauss_hermite_basis = dk.generate_gauss_hermite_basis

# DiffKernelF32 may not be exported in all builds; guard gracefully
try:
    DiffKernelF32 = dk.DiffKernelF32
except AttributeError:
    DiffKernelF32 = None

# deserialize_diff_kernel may not be exported in all builds; guard gracefully
try:
    deserialize_diff_kernel = dk.deserialize_diff_kernel
except AttributeError:
    deserialize_diff_kernel = None

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def rms(arr: np.ndarray) -> float:
    """Return RMS of a (flattened) array."""
    return float(np.sqrt(np.mean(arr.ravel() ** 2)))


def generate_stars(image: np.ndarray, num_stars: int, psf_sigma: float = 1.2) -> np.ndarray:
    """Populate *image* with *num_stars* random 2-D Gaussian point sources.

    Stars are placed with integer-pixel centres and amplitudes drawn uniformly
    from [1.0, 1000.0].  A square patch of radius ``star_radius`` is used; the
    function clamps each patch to valid image boundaries.
    """
    star_radius = int(np.ceil(4.0 * psf_sigma))
    offsets = np.arange(-star_radius, star_radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    profile = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * psf_sigma ** 2)) / (2.0 * np.pi * psf_sigma ** 2)

    edge = star_radius
    y_cens = np.random.randint(edge, image.shape[0] - edge, size=num_stars)
    x_cens = np.random.randint(edge, image.shape[1] - edge, size=num_stars)
    amplitudes = np.random.uniform(1.0, 1000.0, size=num_stars)
    pr, pc = profile.shape

    for i in range(num_stars):
        yc, xc = int(y_cens[i]), int(x_cens[i])
        y_start = max(0, yc - star_radius)
        x_start = max(0, xc - star_radius)
        y_end = min(image.shape[0], yc + star_radius + 1)
        x_end = min(image.shape[1], xc + star_radius + 1)
        pry = y_end - y_start
        prc = x_end - x_start
        yr = y_start - (yc - star_radius)
        xr = x_start - (xc - star_radius)
        image[y_start:y_end, x_start:x_end] += (
            amplitudes[i] * profile[yr : yr + pry, xr : xr + prc]
        )
    return image


# ---------------------------------------------------------------------------
# Python reference implementations  —  MUST match the Rust algorithm exactly
# ---------------------------------------------------------------------------


def _compute_spatial_terms(poly_x_pos: float, poly_y_pos: float, spatial_order: int) -> np.ndarray:
    """Compute lower-triangular Chebyshev spatial terms.

    Returns a 1-D array of length ``(order+1)*(order+2)//2`` containing the
    products ``T_i(x_pos) * T_j(y_pos)`` for ``i + j <= order``, in the same
    row-major ordering used by the Rust code.
    """
    cheb_size = max(1, spatial_order)
    x_cheb = np.zeros(cheb_size + 1, dtype=np.float64)
    y_cheb = np.zeros(cheb_size + 1, dtype=np.float64)
    x_cheb[0] = 1.0
    y_cheb[0] = 1.0
    x_cheb[1] = poly_x_pos
    y_cheb[1] = poly_y_pos

    for i in range(2, spatial_order + 1):
        x_cheb[i] = 2.0 * poly_x_pos * x_cheb[i - 1] - x_cheb[i - 2]
        y_cheb[i] = 2.0 * poly_y_pos * y_cheb[i - 1] - y_cheb[i - 2]

    size = (spatial_order + 1) * (spatial_order + 2) // 2
    spatial_terms = np.zeros(size, dtype=np.float64)
    idx = 0
    for i in range(spatial_order + 1):
        for j in range(spatial_order - i + 1):
            spatial_terms[idx] = x_cheb[i] * y_cheb[j]
            idx += 1
    return spatial_terms


def _convolve_at_one_point(
    x: int,
    y: int,
    R: int,
    basis_functions: list,
    image: np.ndarray,
) -> np.ndarray:
    """Correlate the template with each separable basis function at pixel (x, y).

    This is a *correlation* (no kernel flip), matching the Rust implementation.
    For basis ``b``:

        C[b] = Σ_i Σ_j  image[y-R+i, x-R+j] * basis_y[i] * basis_x[j]

    Returns a 1-D array of length ``len(basis_functions)``.
    """
    basis_len = len(basis_functions)
    kernel_size = 2 * R + 1
    result = np.zeros(basis_len, dtype=np.float64)

    y_start = y - R
    x_start = x - R

    for b in range(basis_len):
        basis_y = basis_functions[b][0]
        basis_x = basis_functions[b][1]
        acc = 0.0
        for i in range(kernel_size):
            for j in range(kernel_size):
                acc += image[y_start + i, x_start + j] * basis_y[i] * basis_x[j]
        result[b] = acc
    return result


def py_solve_diff_kernel(
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    basis_functions: list,
    spatial_order: int,
    template_image: np.ndarray,
    target_image: np.ndarray,
) -> dict:
    """Pure-Python reference for ``DiffKernel.solve_diff_kernel``.

    Returns a dict with keys ``'coefficients'``, ``'basis_radius'``, and
    ``'spatial_order'``.
    """
    R = len(basis_functions[0][0]) // 2  # half_width
    template_H, template_W = template_image.shape
    x_mid = template_W // 2
    y_mid = template_H // 2

    # -- position filtering (identical to Rust) --
    valid = (
        (x_indices.astype(np.int64) > R)
        & (x_indices.astype(np.int64) < template_W - (R + 2))
        & (y_indices.astype(np.int64) > R)
        & (y_indices.astype(np.int64) < template_H - (R + 2))
    )
    x_filtered = x_indices[valid]
    y_filtered = y_indices[valid]

    basis_len = len(basis_functions)
    spatial_size = (spatial_order + 1) * (spatial_order + 2) // 2
    num_params = basis_len * spatial_size

    M = np.zeros((num_params, num_params), dtype=np.float64)
    rhs = np.zeros(num_params, dtype=np.float64)
    terms = np.zeros(num_params, dtype=np.float64)

    for idx in range(len(x_filtered)):
        x = int(x_filtered[idx])
        y = int(y_filtered[idx])

        # Normalised Chebyshev coordinates
        poly_x_pos = (x - x_mid) / x_mid
        poly_y_pos = (y - y_mid) / y_mid

        # Chebyshev spatial terms
        spatial_terms = _compute_spatial_terms(poly_x_pos, poly_y_pos, spatial_order)

        # Convolve template with each separable basis
        conv_vals = _convolve_at_one_point(x, y, R, basis_functions, template_image)

        # Build terms vector: terms[bas * spatial_size + spat] = conv * spatial
        for b in range(basis_len):
            terms[b * spatial_size : (b + 1) * spatial_size] = (
                conv_vals[b] * spatial_terms
            )

        # Normal equations (accumulate symmetric outer product)
        M += np.outer(terms, terms)
        target_value = target_image[y - R, x - R]
        rhs += terms * target_value

    coefficients = np.linalg.solve(M, rhs)

    return {
        "coefficients": coefficients,
        "basis_radius": R,
        "spatial_order": spatial_order,
    }


def py_apply_kernel(
    input_image: np.ndarray,
    basis_functions: list,
    coefficients: np.ndarray,
    basis_radius: int,
    spatial_order: int,
) -> np.ndarray:
    """Pure-Python reference for ``DiffKernel.apply_kernel``.

    Vectorized implementation that:
    1. Correlates the image with each separable basis function via FFT
    2. Computes Chebyshev spatial terms for all output pixels at once
    3. Accumulates the weighted sum ``convolved[b] * spat[t] * coeffs[b,t]``

    This replaces a pixel-by-pixel pure Python loop that scaled as O(H*W*basis*spatial)
    with a vectorised approach using scipy.signal.fftconvolve.
    """
    R = basis_radius
    H, W = input_image.shape
    x_mid = W // 2
    y_mid = H // 2
    basis_len = len(basis_functions)
    spatial_size = (spatial_order + 1) * (spatial_order + 2) // 2
    coeffs = coefficients.reshape(basis_len, spatial_size)

    # 1. Correlate image with each separable basis (using FFT, flipping kernel = correlation)
    convolved = np.zeros((basis_len, H, W), dtype=np.float64)
    for b in range(basis_len):
        kernel_2d = np.outer(basis_functions[b][0], basis_functions[b][1])
        convolved[b] = fftconvolve(input_image, kernel_2d[::-1, ::-1], mode="same")

    # 2. Compute Chebyshev spatial terms for all output pixels at once
    y_indices = np.arange(R, H - R)
    x_indices = np.arange(R, W - R)
    y_pos, x_pos = np.meshgrid(x_indices, y_indices, indexing="ij")

    poly_x_pos = (x_pos.astype(np.float64) - x_mid) / x_mid
    poly_y_pos = (y_pos.astype(np.float64) - y_mid) / y_mid

    # Chebyshev vectors for each pixel
    cheb_size = max(1, spatial_order)
    x_cheb = np.zeros((cheb_size + 1, H - 2 * R, W - 2 * R), dtype=np.float64)
    y_cheb = np.zeros((cheb_size + 1, H - 2 * R, W - 2 * R), dtype=np.float64)
    x_cheb[0] = 1.0
    y_cheb[0] = 1.0
    x_cheb[1] = poly_x_pos
    y_cheb[1] = poly_y_pos
    for i in range(2, spatial_order + 1):
        x_cheb[i] = 2.0 * poly_x_pos * x_cheb[i - 1] - x_cheb[i - 2]
        y_cheb[i] = 2.0 * poly_y_pos * y_cheb[i - 1] - y_cheb[i - 2]

    # Build spatial terms: shape (spatial_size, H-2R, W-2R)
    spat = np.zeros((spatial_size, H - 2 * R, W - 2 * R), dtype=np.float64)
    idx = 0
    for i in range(spatial_order + 1):
        for j in range(spatial_order - i + 1):
            spat[idx] = x_cheb[i] * y_cheb[j]
            idx += 1

    # 3. Weighted sum: output[oy, ox] = sum_b sum_t convolved[b, oy+R, ox+R] * spat[t, oy, ox] * coeffs[b, t]
    out_H, out_W = H - 2 * R, W - 2 * R
    output = np.zeros((out_H, out_W), dtype=np.float64)
    for b in range(basis_len):
        for t in range(spatial_size):
            output += convolved[b, R : R + out_H, R : R + out_W] * spat[t] * coeffs[b, t]

    return output


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class DiffKernelConvolveTestCase(TestCase):
    """Validation of the Rust diff kernel against a Python reference."""

    # ------------------------------------------------------------------
    # Common synthetic-image setup (used by all tests)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_synthetic_images(
        img_size: int = 500,
        num_stars: int = 70,
        noise_sigma: float = 1.0,
        seed: int = 42,
    ):
        """Return ``(template_sci, target_sci, xind, yind, noise_sigma)``.

        *template_sci*  — noisy template image
        *target_sci*    — convolved-then-cropped "science" image (ground truth)
        *xind*, *yind*  — int32 arrays of star-pixel positions
        """
        np.random.seed(seed)

        # --- 1. Build template image with random Gaussian stars ---
        template = np.zeros((img_size, img_size), dtype=np.float64)
        template = generate_stars(template, num_stars, psf_sigma=1.2)

        # --- 2. Build separable anisotropic Gaussian kernel ---
        sigma_x, sigma_y = 2.0, 1.0
        kernel_size = 21
        center = kernel_size // 2
        y_off = np.arange(kernel_size, dtype=np.float64) - center + 1.0
        x_off = np.arange(kernel_size, dtype=np.float64) - center - 0.5
        gauss_y = (
            np.exp(-(y_off ** 2) / (2.0 * sigma_y ** 2))
            / (np.sqrt(2.0 * np.pi) * sigma_y)
        )
        gauss_x = (
            np.exp(-(x_off ** 2) / (2.0 * sigma_x ** 2))
            / (np.sqrt(2.0 * np.pi) * sigma_x)
        )
        kernel_2d = np.outer(gauss_y, gauss_x)
        kernel_2d /= kernel_2d.sum()

        # --- 3. Convolve + crop to create "science" image ---
        convolved = fftconvolve(template, kernel_2d, mode="same").astype(np.float64)
        target_sci = convolved[10:-10, 10:-10].copy()

        # --- 4. Add noise to the template ---
        noise = np.random.normal(0, noise_sigma * np.sqrt(template.size), template.size)
        template += noise.reshape(template.shape)

        # --- 5. Identify star positions ---
        threshold = np.percentile(template[template > 0], 85) if np.any(template > 0) else 0.0
        yind_all, xind_all = np.where(template > threshold)

        # Sub-sample to manageable density
        if len(xind_all) > 200:
            rng = np.random.RandomState(seed + 1)
            pick = rng.choice(len(xind_all), size=200, replace=False)
            xind_all = xind_all[pick]
            yind_all = yind_all[pick]

        xind = xind_all.astype(np.int32)
        yind = yind_all.astype(np.int32)

        return template, target_sci, xind, yind, noise_sigma

    # ------------------------------------------------------------------
    # f64 test
    # ------------------------------------------------------------------
    def test_apply_kernel_f64_vs_python(self):
        """Isolated test of ``DiffKernel.apply_kernel`` vs Python reference.

        Fits coefficients with the **Rust** solver, then applies those same
        coefficients with both the Rust :py:meth:`apply_kernel` and the Python
        ``py_apply_kernel``.  The comparison therefore exercises **only** the
        apply logic, not the solver.
        """
        template, target_sci, xind, yind, noise_sigma = self._build_synthetic_images(
            img_size=500, num_stars=70, noise_sigma=1.0, seed=42
        )

        basis = generate_gauss_hermite_basis(
            half_width=10.0, widths=[1.0, 2.0], orders=[1, 2]
        )
        spatial_order = 2

        # --- Rust solver (provides coefficients) ---
        rust_kernel = DiffKernel.solve_diff_kernel(
            xind, yind, basis, spatial_order, template, target_sci
        )
        rust_coeffs = rust_kernel.get_basis_coefficients()
        R = len(basis[0][0]) // 2

        # --- Rust apply ---
        rust_applied = rust_kernel.apply_kernel(template)

        # --- Python apply **with Rust coefficients** (isolates apply) ---
        py_applied = py_apply_kernel(
            template, basis, rust_coeffs, R, spatial_order
        )

        # Shape
        self.assertEqual(rust_applied.shape, py_applied.shape)
        self.assertEqual(rust_applied.shape, target_sci.shape)

        # The two apply paths should agree to f64 precision
        np.testing.assert_allclose(rust_applied, py_applied, rtol=1e-6, atol=1e-8)

        # Sanity: residual RMS vs target is bounded
        diff_rust = target_sci - rust_applied
        self.assertLess(rms(diff_rust), 2.0 * noise_sigma * np.sqrt(template.size))

    def test_apply_kernel_f32_vs_python(self):
        """Isolated test of ``DiffKernelF32.apply_kernel`` vs Python reference.

        Same approach as the f64 test but with float32 inputs.  The Rust f32
        apply result is compared against a float32-cast of the Python f64
        apply result (computed with the f64 Rust coefficients, cast to f32).
        """
        if DiffKernelF32 is None:
            self.skipTest("DiffKernelF32 not exported in current build")

        template, target_sci, xind, yind, noise_sigma = self._build_synthetic_images(
            img_size=500, num_stars=70, noise_sigma=1.0, seed=42
        )

        template_f32 = template.astype(np.float32)
        target_f32 = target_sci.astype(np.float32)

        basis_f64 = generate_gauss_hermite_basis(
            half_width=10.0, widths=[1.0, 2.0], orders=[1, 2]
        )
        basis_f32 = [
            (y_k.astype(np.float32), x_k.astype(np.float32))
            for y_k, x_k in basis_f64
        ]
        spatial_order = 2

        # --- Rust f32 solver ---
        rust_kernel = DiffKernelF32.solve_diff_kernel(
            xind, yind, basis_f32, spatial_order, template_f32, target_f32
        )
        rust_coeffs = rust_kernel.get_basis_coefficients()
        R = len(basis_f32[0][0]) // 2

        # --- Rust f32 apply ---
        rust_applied = rust_kernel.apply_kernel(template_f32)

        # --- Python apply **with Rust f32 coefficients, f32 basis and image** ---
        py_applied = py_apply_kernel(
            template_f32, basis_f32, rust_coeffs, R, spatial_order
        )

        # Shape
        self.assertEqual(rust_applied.shape, py_applied.shape)

        # f32: relaxed tolerance
        np.testing.assert_allclose(
            rust_applied,
            py_applied.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_solve_coefficients_f64_consistent(self):
        """Verify Rust and Python **solver** coefficients agree closely.

        Both solvers are run on the same inputs and their coefficient vectors
        are compared element-wise.
        """
        np.random.seed(99)

        img_size = 200
        template = np.zeros((img_size, img_size), dtype=np.float64)
        template = generate_stars(template, num_stars=20, psf_sigma=1.0)

        sigma_x, sigma_y = 1.5, 0.8
        kernel_size = 13
        center = kernel_size // 2
        y_off = np.arange(kernel_size, dtype=np.float64) - center
        x_off = np.arange(kernel_size, dtype=np.float64) - center
        gauss_y = np.exp(-(y_off ** 2) / (2.0 * sigma_y ** 2))
        gauss_x = np.exp(-(x_off ** 2) / (2.0 * sigma_x ** 2))
        kernel_2d = np.outer(gauss_y, gauss_x)
        kernel_2d /= kernel_2d.sum()

        convolved = fftconvolve(template, kernel_2d, mode="same")
        target_sci = convolved[6:-6, 6:-6].copy()

        template += np.random.normal(0, 0.5, template.size).reshape(template.shape)

        threshold = np.percentile(template[template > 0], 80)
        yind_all, xind_all = np.where(template > threshold)
        if len(xind_all) > 100:
            rng = np.random.RandomState(99)
            pick = rng.choice(len(xind_all), size=100, replace=False)
            xind_all = xind_all[pick]
            yind_all = yind_all[pick]
        xind = xind_all.astype(np.int32)
        yind = yind_all.astype(np.int32)

        spatial_order = 2
        basis = generate_gauss_hermite_basis(
            half_width=6.0, widths=[1.0, 2.0], orders=[1, 2]
        )

        # Rust solver
        rust_kernel = DiffKernel.solve_diff_kernel(
            xind, yind, basis, spatial_order, template, target_sci
        )
        rust_coeffs = rust_kernel.get_basis_coefficients()

        # Python solver
        py_result = py_solve_diff_kernel(
            xind, yind, basis, spatial_order, template, target_sci
        )
        py_coeffs = py_result["coefficients"]

        self.assertEqual(rust_coeffs.shape[0], py_coeffs.shape[0])
        np.testing.assert_allclose(rust_coeffs, py_coeffs, rtol=1e-4, atol=1e-8)

    def test_json_roundtrip_f64(self):
        """Test that DiffKernel json/from_json round-trips correctly (f64)."""
        np.random.seed(42)
        template, target_sci, xind, yind, _ = self._build_synthetic_images()

        basis = generate_gauss_hermite_basis(half_width=10.0, widths=[1.0, 2.0], orders=[1, 2])
        spatial_order = 2

        kernel = DiffKernel.solve_diff_kernel(
            xind, yind, basis, spatial_order, template, target_sci
        )

        # Serialize to JSON
        json_str = kernel.json()
        self.assertIsInstance(json_str, str)
        self.assertGreater(len(json_str), 0)

        # Verify JSON contains the "dtype" field
        import json
        parsed = json.loads(json_str)
        self.assertEqual(parsed.get("dtype"), "DiffKernel")

        # Deserialize back
        restored = DiffKernel.from_json(json_str)
        self.assertIsInstance(restored, DiffKernel)

        # Verify coefficients match
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

        # Verify apply_kernel produces same output on a test image
        test_image = template.astype(np.float64)
        out_original = kernel.apply_kernel(test_image)
        out_restored = restored.apply_kernel(test_image)
        np.testing.assert_allclose(out_original, out_restored, rtol=1e-8, atol=1e-10)

        # Test deserialize_diff_kernel dispatch
        if deserialize_diff_kernel is not None:
            dispatched = deserialize_diff_kernel(json_str)
            self.assertIsInstance(dispatched, DiffKernel)
            dispatched_coeffs = dispatched.get_basis_coefficients()
            np.testing.assert_allclose(original_coeffs, dispatched_coeffs, rtol=1e-10, atol=1e-12)

    def test_json_roundtrip_invalid_json(self):
        """Test that DiffKernel.from_json raises on invalid JSON."""
        with self.assertRaises(ValueError):
            DiffKernel.from_json("this is not valid json")
        with self.assertRaises(ValueError):
            DiffKernel.from_json("{}")
        with self.assertRaises(ValueError):
            DiffKernel.from_json("[]")

    def test_deserialize_diff_kernel_dispatch_f64(self):
        """Test that deserialize_diff_kernel correctly dispatches an f64 kernel."""
        if deserialize_diff_kernel is None:
            self.skipTest("deserialize_diff_kernel not available in current build")

        np.random.seed(42)
        template, target_sci, xind, yind, _ = self._build_synthetic_images()

        basis = generate_gauss_hermite_basis(half_width=10.0, widths=[1.0, 2.0], orders=[1, 2])
        spatial_order = 2

        kernel = DiffKernel.solve_diff_kernel(
            xind, yind, basis, spatial_order, template, target_sci
        )

        json_str = kernel.json()

        # Dispatch should return a DiffKernel instance
        restored = deserialize_diff_kernel(json_str)
        self.assertIsInstance(restored, DiffKernel)

        # Verify coefficients match
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

        # Verify apply_kernel produces same output
        out_original = kernel.apply_kernel(template.astype(np.float64))
        out_restored = restored.apply_kernel(template.astype(np.float64))
        np.testing.assert_allclose(out_original, out_restored, rtol=1e-8, atol=1e-10)

    def test_deserialize_diff_kernel_dispatch_f32(self):
        """Test that deserialize_diff_kernel correctly dispatches an f32 kernel."""
        if DiffKernelF32 is None:
            self.skipTest("DiffKernelF32 not exported in current build")
        if deserialize_diff_kernel is None:
            self.skipTest("deserialize_diff_kernel not available in current build")

        np.random.seed(42)
        template, target_sci, xind, yind, _ = self._build_synthetic_images()
        template_f32 = template.astype(np.float32)
        target_f32 = target_sci.astype(np.float32)

        basis_f64 = generate_gauss_hermite_basis(half_width=10.0, widths=[1.0, 2.0], orders=[1, 2])
        basis_f32 = [
            (y_k.astype(np.float32), x_k.astype(np.float32))
            for y_k, x_k in basis_f64
        ]
        spatial_order = 2

        kernel = DiffKernelF32.solve_diff_kernel(
            xind, yind, basis_f32, spatial_order, template_f32, target_f32
        )

        json_str = kernel.json()

        # Dispatch should return a DiffKernelF32 instance
        restored = deserialize_diff_kernel(json_str)
        self.assertIsInstance(restored, DiffKernelF32)

        # Verify apply_kernel produces same output (f32 tolerance)
        out_original = kernel.apply_kernel(template_f32)
        out_restored = restored.apply_kernel(template_f32)
        np.testing.assert_allclose(out_original, out_restored, rtol=1e-4, atol=1e-4)

    def test_deserialize_diff_kernel_invalid(self):
        """Test that deserialize_diff_kernel raises on invalid input."""
        if deserialize_diff_kernel is None:
            self.skipTest("deserialize_diff_kernel not available in current build")

        # Invalid JSON
        with self.assertRaises(ValueError):
            deserialize_diff_kernel("not valid json")

        # Missing dtype field (legacy format not supported by dispatcher)
        with self.assertRaises(ValueError):
            deserialize_diff_kernel("{}")

        # Wrong dtype value
        with self.assertRaises(ValueError):
            import json as _json
            bad = {"dtype": "DiffKernelF16", "basis_arrays": [], "basis_radius": 0,
                   "spatial_order": 0, "basis_coefficients": []}
            deserialize_diff_kernel(_json.dumps(bad))


# ---------------------------------------------------------------------------
# Module boilerplate
# ---------------------------------------------------------------------------


def setup_module(module):
    """Initialise the lsst.utils test harness before any tests run."""
    init()


if __name__ == "__main__":
    init()
    unittest.main()
