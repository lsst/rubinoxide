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
Tests for the Pydantic v2 integration methods on ``DiffKernel``.

The unified ``DiffKernel`` class supports both f64 and f32 data.  The
following methods are exercised:

  - ``__get_pydantic_core_schema__``
  - ``__get_pydantic_json_schema__``
  - ``model_validate``
  - ``model_dump``

Pydantic-specific integration tests (embedding DiffKernel inside a Pydantic
``BaseModel``) are gated behind a runtime import and skip gracefully.
"""

import json
import jsonschema
import unittest

import numpy as np

from lsst.utils.tests import TestCase, init

# ---------------------------------------------------------------------------
# Import the Rust-compiled symbols
# ---------------------------------------------------------------------------
from lsst.rubinoxide._rubinoxide import difference_kernel as dk

DiffKernel = dk.DiffKernel
generate_gauss_hermite_basis = dk.generate_gauss_hermite_basis

# ---------------------------------------------------------------------------
# Optional pydantic import — tests skip cleanly if absent
# ---------------------------------------------------------------------------
try:
    import pydantic

    PYDANTIC_AVAILABLE = True
except ImportError:
    pydantic = None
    PYDANTIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

SEED = 42
HALF_WIDTH = 10.0
WIDTHS = [1.0, 2.0]
ORDERS = [1, 2]
SPATIAL_ORDER = 2


def _build_synthetic_images():
    """Return ``(template, target_sci, xind, yind)`` suitable for solving a
    diff kernel.  This is a minimal copy of the setup used in
    ``test_diff_kernel.py`` so this module stays self-contained.

    Uses a fixed seed for reproducibility.
    """
    np.random.seed(SEED)
    img_size = 200

    # Build template with a few synthetic stars
    template = np.zeros((img_size, img_size), dtype=np.float64)
    star_radius = int(np.ceil(4.0 * 1.2))
    offsets = np.arange(-star_radius, star_radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    profile = np.exp(-(dx**2 + dy**2) / (2.0 * 1.2**2)) / (2.0 * np.pi * 1.2**2)

    edge = star_radius
    num_stars = 20
    y_cens = np.random.randint(edge, img_size - edge, size=num_stars)
    x_cens = np.random.randint(edge, img_size - edge, size=num_stars)
    amplitudes = np.random.uniform(100.0, 1000.0, size=num_stars)

    for i in range(num_stars):
        yc, xc = int(y_cens[i]), int(x_cens[i])
        y_start = max(0, yc - star_radius)
        x_start = max(0, xc - star_radius)
        y_end = min(img_size, yc + star_radius + 1)
        x_end = min(img_size, xc + star_radius + 1)
        pry = y_end - y_start
        prc = x_end - x_start
        yr = y_start - (yc - star_radius)
        xr = x_start - (xc - star_radius)
        template[y_start:y_end, x_start:x_end] += amplitudes[i] * profile[yr : yr + pry, xr : xr + prc]

    # Apply a simple Gaussian PSF change to simulate different telescope
    # conditions between template and science
    from scipy.signal import fftconvolve

    sigma_x, sigma_y = 1.5, 0.8
    ks = 13
    center = ks // 2
    y_off = np.arange(ks, dtype=np.float64) - center
    x_off = np.arange(ks, dtype=np.float64) - center
    gauss_y = np.exp(-(y_off**2) / (2.0 * sigma_y**2))
    gauss_x = np.exp(-(x_off**2) / (2.0 * sigma_x**2))
    kernel_2d = np.outer(gauss_y, gauss_x)
    kernel_2d /= kernel_2d.sum()
    convolved = fftconvolve(template, kernel_2d, mode="same")
    target_sci = convolved[5:-5, 5:-5].copy()

    # Add a bit of noise
    template += np.random.normal(0, 0.5, template.size).reshape(template.shape)

    # Extract star positions
    threshold = np.percentile(template[template > 0], 80)
    yind_all, xind_all = np.where(template > threshold)
    if len(xind_all) > 100:
        rng = np.random.RandomState(SEED + 1)
        pick = rng.choice(len(xind_all), size=100, replace=False)
        xind_all = xind_all[pick]
        yind_all = yind_all[pick]
    xind = xind_all.astype(np.int32)
    yind = yind_all.astype(np.int32)

    return template, target_sci, xind, yind


def _make_kernel():
    """Build a real ``DiffKernel`` (f64) using ``solve_diff_kernel``."""
    template, target_sci, xind, yind = _build_synthetic_images()
    basis = generate_gauss_hermite_basis(half_width=HALF_WIDTH, widths=WIDTHS, orders=ORDERS)
    return DiffKernel.solve_diff_kernel(xind, yind, basis, SPATIAL_ORDER, template, target_sci)


def _make_kernel_f32():
    """Build a real ``DiffKernel`` with f32 data using ``solve_diff_kernel``."""
    template, target_sci, xind, yind = _build_synthetic_images()
    basis_f64 = generate_gauss_hermite_basis(half_width=HALF_WIDTH, widths=WIDTHS, orders=ORDERS)
    basis_f32 = [(yk.astype(np.float32), xk.astype(np.float32)) for yk, xk in basis_f64]
    template_f32 = template.astype(np.float32)
    target_f32 = target_sci.astype(np.float32)
    return DiffKernel.solve_diff_kernel(xind, yind, basis_f32, SPATIAL_ORDER, template_f32, target_f32)


def _get_dict(kernel):
    """Return a Python dict from a kernel's JSON serialisation."""
    return json.loads(kernel.json())


EXPECTED_DUMP_KEYS = frozenset(
    {"dtype", "basis_arrays", "basis_radius", "spatial_order", "basis_coefficients", "reduced_chisq"}
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class DiffKernelPydanticTestCase(TestCase):
    """Exhaustive tests for Pydantic v2 integration on DiffKernel.

    Includes both f64 and f32 variants exercising the unified class.
    """

    # ########################################################################
    # Test 1 — model_validate from existing instance (pass-through)
    # ########################################################################

    def test_model_validate_from_instance(self):
        """``DiffKernel.model_validate(kernel)`` reconstructs an equivalent
        kernel from an existing instance.

        Note: the method creates a fresh instance (does not short-circuit),
        but the coefficients are identical.
        """
        kernel = _make_kernel()
        result = DiffKernel.model_validate(kernel)
        self.assertIsInstance(result, DiffKernel)
        self.assertIsNot(result, kernel)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = result.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

    def test_model_validate_from_instance_f32(self):
        """Unified DiffKernel model_validate from an f32 instance."""
        kernel = _make_kernel_f32()
        result = DiffKernel.model_validate(kernel)
        self.assertIsInstance(result, DiffKernel)
        self.assertIsNot(result, kernel)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = result.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-5, atol=1e-7)

    # ########################################################################
    # Test 2 — model_validate from dict
    # ########################################################################

    def test_model_validate_from_dict(self):
        """``DiffKernel.model_validate(dict)`` reconstructs the kernel."""
        kernel = _make_kernel()
        kernel_dict = _get_dict(kernel)
        restored = DiffKernel.model_validate(kernel_dict)
        self.assertIsInstance(restored, DiffKernel)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

    def test_model_validate_from_dict_f32(self):
        """Unified DiffKernel model_validate from an f32 dict."""
        kernel = _make_kernel_f32()
        kernel_dict = _get_dict(kernel)
        restored = DiffKernel.model_validate(kernel_dict)
        self.assertIsInstance(restored, DiffKernel)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-5, atol=1e-7)

    # ########################################################################
    # Test 3 — model_validate from JSON string
    # ########################################################################

    def test_model_validate_from_json_string(self):
        """``DiffKernel.model_validate(json_str)`` deserialises correctly."""
        kernel = _make_kernel()
        json_str = kernel.json()
        restored = DiffKernel.model_validate(json_str)
        self.assertIsInstance(restored, DiffKernel)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

    def test_model_validate_from_json_string_f32(self):
        """Unified DiffKernel model_validate from an f32 JSON string."""
        kernel = _make_kernel_f32()
        json_str = kernel.json()
        restored = DiffKernel.model_validate(json_str)
        self.assertIsInstance(restored, DiffKernel)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-5, atol=1e-7)

    # ########################################################################
    # Test 4 — model_validate invalid type
    # ########################################################################

    def test_model_validate_invalid_type(self):
        """``model_validate`` with unsupported types raises ``TypeError``."""
        for bad_value in [[1, 2, 3], 42, None]:
            with self.assertRaises(TypeError):
                DiffKernel.model_validate(bad_value)

    # ########################################################################
    # Test 5 — model_validate invalid JSON string
    # ########################################################################

    def test_model_validate_invalid_json_string(self):
        """``model_validate("not json")`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            DiffKernel.model_validate("not json")

    # ########################################################################
    # Test 6 — model_dump returns dict with expected keys
    # ########################################################################

    def test_model_dump_returns_dict(self):
        """``kernel.model_dump()`` returns a dict with the expected schema."""
        kernel = _make_kernel()
        dumped = kernel.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(set(dumped.keys()), EXPECTED_DUMP_KEYS)
        self.assertEqual(dumped["dtype"], "DiffKernel")
        # basis_coefficients is serialised as a numpy-compatible payload
        # (either a plain list or an ndarray dict {"v": 1, "dim": [...], "data": [...]})
        bc = dumped["basis_coefficients"]
        self.assertIsInstance(bc, (list, dict))
        self.assertIsInstance(dumped["basis_arrays"], list)
        self.assertIsInstance(dumped["basis_radius"], int)
        self.assertIsInstance(dumped["spatial_order"], int)

    def test_model_dump_returns_dict_f32(self):
        """Unified DiffKernel model_dump from an f32 kernel preserves dtype."""
        kernel = _make_kernel_f32()
        dumped = kernel.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(set(dumped.keys()), EXPECTED_DUMP_KEYS)
        self.assertEqual(dumped["dtype"], "DiffKernelF32")

    # ########################################################################
    # Test 7 — model_dump → model_validate round-trip
    # ########################################################################

    def test_model_dump_round_trip(self):
        """Coefficients survive a ``model_dump`` → ``model_validate`` round-trip."""
        kernel = _make_kernel()
        dumped = kernel.model_dump()
        restored = DiffKernel.model_validate(dumped)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

    def test_model_dump_round_trip_f32(self):
        """Unified DiffKernel f32 round-trip via model_dump/model_validate."""
        kernel = _make_kernel_f32()
        dumped = kernel.model_dump()
        restored = DiffKernel.model_validate(dumped)
        original_coeffs = kernel.get_basis_coefficients()
        restored_coeffs = restored.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-5, atol=1e-7)

    # ########################################################################
    # Test 8 — model_dump matches json().loads()
    # ########################################################################

    def test_model_dump_json_matches(self):
        """``model_dump()`` is identical to ``json.loads(kernel.json())``."""
        kernel = _make_kernel()
        dumped = kernel.model_dump()
        json_dict = json.loads(kernel.json())
        self.assertEqual(json_dict, dumped)

    def test_model_dump_json_matches_f32(self):
        """Unified DiffKernel f32: model_dump() matches json().loads()."""
        kernel = _make_kernel_f32()
        dumped = kernel.model_dump()
        json_dict = json.loads(kernel.json())
        self.assertEqual(json_dict, dumped)

    # ########################################################################
    # Test 9 — __get_pydantic_core_schema__ exists and is callable
    # ########################################################################

    def test_pydantic_core_schema_exists(self):
        """``__get_pydantic_core_schema__`` is a callable class attribute."""
        self.assertTrue(hasattr(DiffKernel, "__get_pydantic_core_schema__"))
        self.assertTrue(callable(getattr(DiffKernel, "__get_pydantic_core_schema__")))

    # ########################################################################
    # Test 10 — __get_pydantic_json_schema__ exists and returns sensible dict
    # ########################################################################

    def test_pydantic_json_schema_exists(self):
        """``__get_pydantic_json_schema__`` returns a dict with 'type': 'object'."""
        handler = lambda schema, _outer_handler: {}  # noqa: E731
        result = DiffKernel.__get_pydantic_json_schema__({}, handler)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("type"), "object")
        self.assertIn("properties", result)
        props = result["properties"]
        self.assertIn("dtype", props)
        self.assertIn("basis_coefficients", props)
        self.assertIn("basis_arrays", props)
        self.assertIn("basis_radius", props)
        self.assertIn("spatial_order", props)

    # ########################################################################
    # Test 10.1 — JSON schema validates real serialized output
    # ########################################################################

    def test_json_schema_validates_real_output(self):
        """Serialized kernel dict validates against the Pydantic JSON Schema."""
        kernel = _make_kernel()
        kernel_dict = json.loads(kernel.json())
        handler = lambda schema, _outer_handler: {}  # noqa: E731
        schema = DiffKernel.__get_pydantic_json_schema__({}, handler)
        jsonschema.validate(kernel_dict, schema)

    # ########################################################################
    # Tests 11–15 — Full Pydantic BaseModel integration (require pydantic)
    # ########################################################################

    def test_pydantic_basemodel_integration(self):
        """DiffKernel can be embedded in a ``pydantic.BaseModel``."""
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed")
        from pydantic import BaseModel

        class MyModel(BaseModel):
            kernel: DiffKernel

        existing_kernel = _make_kernel()

        # Construct from existing instance
        m = MyModel(kernel=existing_kernel)
        self.assertIsInstance(m.kernel, DiffKernel)

        # Construct from dict
        kernel_dict = json.loads(existing_kernel.json())
        m2 = MyModel(kernel=kernel_dict)
        self.assertIsInstance(m2.kernel, DiffKernel)

    def test_pydantic_model_dump_json(self):
        """A ``BaseModel`` embedding DiffKernel serialises to JSON."""
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed")
        from pydantic import BaseModel

        class MyModel(BaseModel):
            kernel: DiffKernel

        existing_kernel = _make_kernel()
        m = MyModel(kernel=existing_kernel)
        json_str = m.model_dump_json()
        self.assertIsInstance(json_str, str)
        # Should be valid JSON
        data = json.loads(json_str)
        self.assertIn("kernel", data)

    def test_pydantic_model_validate_json(self):
        """A ``BaseModel`` embedding DiffKernel validates from JSON."""
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed")
        from pydantic import BaseModel

        class MyModel(BaseModel):
            kernel: DiffKernel

        existing_kernel = _make_kernel()
        m1 = MyModel(kernel=existing_kernel)
        json_str = m1.model_dump_json()
        m2 = MyModel.model_validate_json(json_str)
        self.assertIsInstance(m2.kernel, DiffKernel)
        original_coeffs = existing_kernel.get_basis_coefficients()
        restored_coeffs = m2.kernel.get_basis_coefficients()
        np.testing.assert_allclose(original_coeffs, restored_coeffs, rtol=1e-10, atol=1e-12)

    def test_pydantic_json_schema(self):
        """``BaseModel.model_json_schema()`` includes the kernel sub-schema."""
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed")
        from pydantic import BaseModel

        class MyModel(BaseModel):
            kernel: DiffKernel

        schema = MyModel.model_json_schema()
        self.assertIsInstance(schema, dict)
        # The kernel field should appear in properties
        self.assertIn("properties", schema)
        self.assertIn("kernel", schema["properties"])

    def test_pydantic_validation_error(self):
        """Assigning an invalid value to a DiffKernel field raises an error.

        Values that cause ``ValueError`` (e.g. unparseable JSON string) are
        caught by Pydantic as ``ValidationError``.  Types rejected with
        ``TypeError`` propagate as ``TypeError``.
        """
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed")
        from pydantic import BaseModel, ValidationError

        class MyModel(BaseModel):
            kernel: DiffKernel

        # Unparseable JSON string → ValueError → ValidationError
        with self.assertRaises(ValidationError):
            MyModel(kernel="invalid")

        # Plain int → TypeError (not caught as ValidationError)
        with self.assertRaises(TypeError):
            MyModel(kernel=42)

        # List → TypeError (not caught as ValidationError)
        with self.assertRaises(TypeError):
            MyModel(kernel=[1, 2, 3])


# ---------------------------------------------------------------------------
# Module boilerplate
# ---------------------------------------------------------------------------


def setup_module(module):
    """Initialise the lsst.utils test harness before any tests run."""
    init()


if __name__ == "__main__":
    init()
    unittest.main()
