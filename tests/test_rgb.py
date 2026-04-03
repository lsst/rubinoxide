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
        result = rgb.inpaint_mask(test_image, mask, iterations=10)
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

        result = rgb.inpaint_mask(test_image, mask, iterations=20)

        masked_values = result[18:32, 18:32]
        self.assertGreater(masked_values.min(), 0.16)
        self.assertLess(masked_values.max(), 0.98)

    def test_inpaint_boundary_consistency(self):
        """Values at mask boundaries should be consistent with surroundings"""
        test_image = np.ones((50, 50), dtype=np.float64) * 0.5
        test_image[:, 25:] += 0.2

        mask = np.zeros((50, 50), dtype=bool)
        mask[:, 20:30] = True

        result = rgb.inpaint_mask(test_image, mask, iterations=20)

        left_mean = result[20, 20].mean()
        np.testing.assert_allclose(left_mean, 0.5, atol=0.1)

        right_mean = result[20, 29].mean()
        np.testing.assert_allclose(right_mean, 0.7, atol=0.1)

    def test_inpaint_noise_varies(self):
        """Multiple inpainting runs should show noise variation"""
        test_image = np.ones((50, 50), dtype=np.float64) * 0.5

        results = []
        for seed in range(10):
            np.random.seed(seed)
            mask = np.zeros((50, 50), dtype=bool)
            mask[25, 25] = True
            result = rgb.inpaint_mask(test_image, mask, iterations=5)
            results.append(result[25, 25])

        self.assertGreater(np.std(results), 0.01)

    def test_inpaint_all_masked_valid(self):
        """All pixels masked should still return valid result"""
        test_image = np.ones((20, 20), dtype=np.float64) * 0.5
        mask = np.ones((20, 20), dtype=bool)

        result = rgb.inpaint_mask(test_image, mask, iterations=10)
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


class MemoryTestCase(MemoryTestCase):
    """Test for memory leaks"""

    pass


def setup_module(module):
    init()


if __name__ == "__main__":
    init()
    unittest.main()
