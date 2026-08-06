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

import unittest

import numpy as np

from lsst.rubinoxide import rgb
from lsst.utils.tests import MemoryTestCase, TestCase, init


class RGBTestCase(TestCase):
    """Tests the capability of the rgb submodule"""

    def test_pixel_conversion(self):
        """Test the colorspace conversion code"""
        test_image = np.array([[[0.2, 0.42, 0.81]]])

        whitepoint = (0.31, 0.32)

        expected = np.array([[[0.73695252, -0.03689747, -0.11662942]]])

        oklab_values = rgb.RGB_to_Oklab(test_image, whitepoint)
        np.testing.assert_allclose(oklab_values, expected, atol=1e-4, rtol=0)

        round_trip = rgb.Oklab_to_RGB(oklab_values, whitepoint)

        np.testing.assert_allclose(round_trip, test_image, atol=1e-4, rtol=0)


class DiffusionTestCase(TestCase):
    """Tests the diffusion and inpainting functionality"""

    def test_diffuse_constant_image_invariant(self):
        """Diffusion should preserve constant images exactly"""
        test_image = np.ones((50, 50), dtype=np.float64) * 0.73
        result = rgb.diffuse_gray_image(test_image, iterations=10)
        np.testing.assert_allclose(result, 0.73, atol=1e-10)

    def test_diffuse_zero_radius_identity(self):
        """Zero radius should return unchanged image"""
        np.random.seed(42)
        test_image = np.random.rand(30, 30).astype(np.float64)
        result = rgb.diffuse_gray_image(test_image, iterations=1, radius=0.0)
        np.testing.assert_allclose(result, test_image, atol=1e-10)

    def test_diffuse_single_pixel_invariant(self):
        """Single pixel should be unchanged regardless of iterations"""
        test_image = np.array([[0.5]], dtype=np.float64)
        result = rgb.diffuse_gray_image(test_image, iterations=10)
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_diffuse_all_masked_unchanged(self):
        """No masked pixels should return exact copy"""
        np.random.seed(42)
        test_image = np.random.rand(30, 30).astype(np.float64)
        mask = np.zeros((30, 30), dtype=bool)
        result = rgb.inpaint_mask(test_image, mask, iterations=10, random_seed=20)
        np.testing.assert_allclose(result, test_image, atol=1e-10)

    def test_diffuse_linear_gradient_preserved(self):
        """Diffusion should preserve linear gradients with minimal
        distortion.
        """
        test_image = np.linspace(0.1, 0.9, 50).reshape(1, 50).repeat(50, axis=0)
        result = rgb.diffuse_gray_image(test_image, iterations=3, radius=1.0)

        original_grad = test_image[25, 30] - test_image[25, 20]
        result_grad = result[25, 30] - result[25, 20]
        np.testing.assert_allclose(original_grad, result_grad, rtol=0.05)

    def test_diffuse_edge_location_preserved(self):
        """Edge positions should be preserved within sub-pixel accuracy"""
        test_image = np.zeros((50, 50), dtype=np.float64)
        test_image[:, 25:] = 1.0
        result = rgb.diffuse_gray_image(test_image, iterations=5)

        edge_crossings = np.where(result[25, :] > 0.5)[0]
        self.assertGreater(len(edge_crossings), 0)
        self.assertAlmostEqual(edge_crossings[0], 25, delta=1)

    def test_diffuse_anisotropic_sharpens_edges(self):
        """Anisotropic diffusion should preserve edges better than isotropic"""
        test_image = np.zeros((50, 50), dtype=np.float64)
        test_image[15:35, 15:35] = 1.0

        result_iso = rgb.diffuse_gray_image(test_image, iterations=5, anisotropy_first=0.0)

        result_aniso = rgb.diffuse_gray_image(test_image, iterations=5, anisotropy_first=1.0)

        iso_grad = abs(result_iso[15, 16] - result_iso[15, 14])
        aniso_grad = abs(result_aniso[15, 16] - result_aniso[15, 14])
        self.assertGreater(aniso_grad, iso_grad)

    def test_diffuse_variance_non_increase(self):
        """With zero sharpness, diffusion should not increase variance
        significantly.
        """
        np.random.seed(42)
        test_image = np.random.rand(50, 50).astype(np.float64) * 0.8 + 0.1
        result = rgb.diffuse_gray_image(test_image, iterations=5, sharpness=0.0)
        self.assertLessEqual(result.var(), test_image.var() * 1.05)

    def test_inpaint_gradient_context_preserved(self):
        """Inpainting should respect background gradients"""
        y, x = np.mgrid[0:50, 0:50]
        test_image = (x + y) / 100.0

        mask = np.zeros((50, 50), dtype=bool)
        mask[15:35, 15:35] = True

        result = rgb.inpaint_mask(test_image, mask, iterations=20, random_seed=20)

        masked_values = result[18:32, 18:32]
        self.assertGreater(masked_values.min(), 0.16)
        self.assertLess(masked_values.max(), 0.98)

    def test_inpaint_boundary_consistency(self):
        """Values at mask boundaries should be consistent with surroundings"""
        test_image = np.ones((50, 50), dtype=np.float64) * 0.5
        test_image[:, 25:] += 0.2

        mask = np.zeros((50, 50), dtype=bool)
        mask[:, 20:30] = True

        result = rgb.inpaint_mask(test_image, mask, iterations=20, random_seed=20)

        left_mean = result[20, 20].mean()
        np.testing.assert_allclose(left_mean, 0.5, atol=0.1)

        right_mean = result[20, 29].mean()
        np.testing.assert_allclose(right_mean, 0.7, atol=0.1)

    def test_inpaint_no_boundary_overshoot_ring(self):
        """The hf-sharpening taper (2b) removes the boundary overshoot/dip ring.

        A constant image with a rectangular mask must not exhibit a boundary
        ring or two-pixel bright/dark dipole (values rising above then dipping
        below the background) that the fourth-order diffusion term would
        otherwise amplify from the decomposition step at the mask edge.
        """
        test_image = np.ones((60, 60), dtype=np.float64) * 0.5
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True

        result = rgb.inpaint_mask(test_image, mask, iterations=32, random_seed=42, radius=5.0)

        row = result[30, :]
        interior = row[24:36]
        self.assertLess(interior.max(), 0.5 + 0.03)
        self.assertGreater(interior.min(), 0.5 - 0.03)
        # The ring/dipole manifests as values well above/below the background
        # near the mask edge, so check the whole masked region stays near 0.5.
        masked = result[mask]
        self.assertLess(masked.max(), 0.5 + 0.03)
        self.assertGreater(masked.min(), 0.5 - 0.03)

    def test_inpaint_noise_init_inverse(self):
        """init_method='noise' seeds the masked pixels stochastically, so a
        single masked pixel produces seed-to-seed variation (the stochastic
        structure is preserved, not flattened out by a smoothing band)."""
        test_image = np.ones((50, 50), dtype=np.float64) * 0.5
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True

        results = []
        for seed in range(10):
            result = rgb.inpaint_mask(
                test_image, mask, iterations=5, random_seed=seed, init_method="noise"
            )
            results.append(result[25, 25])

        # The noise init contributes real per-seed structure.
        self.assertGreater(np.std(results), 0.01)
        self.assertTrue(np.all(np.isfinite(results)))

    def test_inpaint_all_masked_valid(self):
        """All pixels masked should still return valid result"""
        test_image = np.ones((20, 20), dtype=np.float64) * 0.5
        mask = np.ones((20, 20), dtype=bool)

        result = rgb.inpaint_mask(test_image, mask, iterations=10, random_seed=20)
        self.assertEqual(result.shape, test_image.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_diffuse_small_images(self):
        """Diffusion should handle 3x3 and larger images"""
        for size in [(3, 3), (5, 5), (10, 10)]:
            test_image = np.random.rand(*size).astype(np.float64)
            result = rgb.diffuse_gray_image(test_image, iterations=1)
            self.assertEqual(result.shape, size)
            self.assertTrue(np.isfinite(result).all())

    def test_diffuse_large_iterations_converge(self):
        """More iterations should lead to more diffusion"""
        test_image = np.zeros((50, 50), dtype=np.float64)
        test_image[20:30, 20:30] = 1.0

        results = []
        for iters in [1, 5, 10, 20]:
            result = rgb.diffuse_gray_image(test_image, iterations=iters)
            results.append(result[25, 25])

        self.assertLess(results[-1], results[0] * 1.001)

    def test_diffuse_sharpness_preserves_peaks(self):
        """Positive sharpness should preserve peak values better than
        negative.
        """
        test_image = np.zeros((50, 50), dtype=np.float64)
        test_image[20:30, 20:30] = 1.0

        result_negative = rgb.diffuse_gray_image(test_image, iterations=10, sharpness=-1.0)
        result_positive = rgb.diffuse_gray_image(test_image, iterations=10, sharpness=1.0)

        self.assertGreater(result_positive[25, 25], result_negative[25, 25])

    def test_diffuse_radius_effects(self):
        """Larger radius should produce more diffusion"""
        test_image = np.zeros((50, 50), dtype=np.float64)
        test_image[20:30, 20:30] = 1.0

        result_small = rgb.diffuse_gray_image(test_image, iterations=5, radius=1.0)
        result_large = rgb.diffuse_gray_image(test_image, iterations=5, radius=5.0)

        self.assertLess(result_large[25, 25], result_small[25, 25])

    def test_inpaint_mask_decomposition_no_contamination(self):
        """The mask-aware decomposition should prevent masked-region values
        from contaminating HF components at nearby unmasked pixels.

        Uses a constant image with a distinctive masked region value.
        Unmasked pixels just outside the mask should NOT be altered.
        """
        # Constant image with a block of different value in the mask
        test_image = np.ones((30, 30), dtype=np.float64) * 10.0
        mask = np.zeros((30, 30), dtype=bool)
        # Mask a 10x10 center
        mask[10:20, 10:20] = True

        # Set masked pixels to a very different value (simulating saturated star)
        test_image[10:20, 10:20] = 1000.0

        result = rgb.inpaint_mask(
            test_image, mask, iterations=10, random_seed=42, radius=3.0
        )

        # Unmasked pixels far from mask should be preserved exactly
        np.testing.assert_allclose(
            result[0:8, 0:8], test_image[0:8, 0:8], atol=1e-10
        )

        # Unmasked pixels where the B-spline filter crosses the mask boundary
        # should also be preserved (the mask-aware decomposition excludes
        # masked neighbors from the filter).
        # Position (9, 15): unmasked, but vertical filter at row 9 includes
        #   rows 10,11 which ARE masked → with fix, those are excluded
        np.testing.assert_allclose(
            result[9, 15], test_image[9, 15], atol=1e-6,
        )
        # Position (15, 9): unmasked, but horizontal filter at col 9 includes
        #   cols 10,11 which ARE masked → with fix, those are excluded
        np.testing.assert_allclose(
            result[15, 9], test_image[15, 9], atol=1e-6,
        )

        # All values should be finite
        self.assertTrue(np.isfinite(result).all())

    def test_inpaint_saturated_star_ab_channel(self):
        """Simulate a saturated star in the a/b channel: extreme values
        in the mask that should not contaminate the decomposition of
        surrounding pixels.
        """
        np.random.seed(42)
        # Background with smooth gradient (offset to non-negative)
        y, x = np.mgrid[0:40, 0:40]
        background = 128.0 + (x + y) * 0.1

        # Saturated star in center: extreme value
        test_image = background.astype(np.float64)
        test_image[15:25, 15:25] = 1000.0  # extreme saturated value

        mask = np.zeros((40, 40), dtype=bool)
        mask[15:25, 15:25] = True

        result = rgb.inpaint_mask(
            test_image, mask, iterations=32, random_seed=42, radius=5.0
        )

        # Pixels far from the mask should be unaffected
        np.testing.assert_allclose(
            result[0:10, 0:10], test_image[0:10, 0:10], atol=1e-10
        )

        # Pixels just outside the mask should NOT show contamination
        # from the extreme 1000.0 value inside the mask
        boundary_band = result[13:15, 13:27]  # 2 rows just above mask
        self.assertLess(
            boundary_band.max(), 200.0,
            "Boundary pixels should not be contaminated by extreme masked values"
        )

        # Result should be finite everywhere
        self.assertTrue(np.isfinite(result).all())

    def test_inpaint_boundary_fill_no_offset_negative(self):
        """boundary_fill (default) should handle Lab a/b-like values centered
        around zero with NO offsetting required, and produce a smooth boundary.
        """
        np.random.seed(42)
        # Smooth star-like profile in an a-channel, centered around 0
        y, x = np.mgrid[0:40, 0:40]
        r2 = (x - 20)**2 + (y - 20)**2
        profile = 30.0 * np.exp(-r2 / 60.0)  # can be positive or negative-ish
        # Add a mild tilt so the field is not perfectly symmetric
        test_image = (profile + (x - 20) * 0.2).astype(np.float64)

        mask = np.zeros((40, 40), dtype=bool)
        mask[14:26, 14:26] = True  # mask the star core

        # No offsetting applied. Should not panic with negative values.
        result = rgb.inpaint_mask(
            test_image, mask, iterations=32, random_seed=42, radius=5.0
        )

        self.assertTrue(np.isfinite(result).all())

        # Unmasked pixels preserved exactly
        np.testing.assert_allclose(
            result[0:12, 0:12], test_image[0:12, 0:12], atol=1e-10
        )

        # Smooth boundary: no large jump between masked (col 14) and
        # unmasked (col 13) pixels at the mask edge.
        row = 20
        jump = abs(result[row, 14] - result[row, 13])
        self.assertLess(
            jump, 20.0,
            "boundary_fill should not create a large discontinuity at the mask edge"
        )

    def test_inpaint_boundary_fill_converges_faster_than_noise(self):
        """boundary_fill should reach near the true solution in far fewer
        iterations than the legacy noise initialization.
        """
        # Constant image; the true inpainted interior value is 0.5
        test_image = np.ones((50, 50), dtype=np.float64) * 0.5
        mask = np.zeros((50, 50), dtype=bool)
        mask[18:32, 18:32] = True

        # At a LOW iteration count, boundary_fill should already be close to
        # the true value, while the noise init lags behind.
        bf = rgb.inpaint_mask(
            test_image, mask, iterations=3, random_seed=1, init_method="boundary_fill"
        )[25, 25]
        noise = rgb.inpaint_mask(
            test_image, mask, iterations=3, random_seed=1, init_method="noise"
        )[25, 25]

        self.assertLess(
            abs(bf - 0.5), abs(noise - 0.5),
            "boundary_fill should be closer to the true value than noise at low iterations"
        )
        self.assertLess(abs(bf - 0.5), 0.05)

    def test_inpaint_boundary_fill_retains_texture(self):
        """boundary_fill should retain spatial structure from the surrounding
        image rather than flattening the masked region to a constant.
        A constant image yields a constant fill, but a textured image should
        produce a textured (spatially varying) inpainted region.
        """
        # Image with a sinusoidal texture overlaying a base gradient.
        np.random.seed(7)
        y, x = np.mgrid[0:50, 0:50]
        base = 10.0 + (x + y) * 0.02
        texture = 0.3 * np.sin(x * 1.7) * np.cos(y * 2.3)
        test_image = (base + texture).astype(np.float64)

        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True

        result = rgb.inpaint_mask(
            test_image, mask, iterations=5, random_seed=1,
            init_method="boundary_fill", radius=3.0
        )

        # The inpainted region should retain spatial variation (not be a
        # flat constant), following the boundary texture.
        inpainted = result[20:30, 20:30]
        self.assertGreater(
            inpainted.std(), 0.05,
            "boundary_fill should retain spatial structure in the masked region"
        )

        # The result stays bounded (finite, sensible range), not exploding
        # from the seed noise.
        self.assertTrue(np.isfinite(inpainted).all())
        self.assertLess(np.abs(inpainted).max(), 100.0)

    def test_inpaint_noise_method_still_works(self):
        """init_method='noise' remains functional: results stay finite and
        bounded near the background level for a single masked pixel (no runaway
        overshoot once the fill is no longer clipped)."""
        test_image = np.ones((50, 50), dtype=np.float64) * 0.5
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True

        results = []
        for seed in range(10):
            result = rgb.inpaint_mask(
                test_image, mask, iterations=5, random_seed=seed, init_method="noise"
            )
            results.append(result[25, 25])

        self.assertTrue(np.isfinite(np.array(results)).all())
        self.assertLess(np.abs(np.array(results) - 0.5).max(), 0.2)


class MemoryTestCase(MemoryTestCase):
    """Test for memory leaks"""

    pass


def setup_module(module):
    init()


if __name__ == "__main__":
    init()
    unittest.main()
