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
from scipy.interpolate import RBFInterpolator

from lsst.rubinoxide.rbf_interpolator import fast_rbf_interpolation_on_grid
from lsst.utils.tests import MemoryTestCase, TestCase, init


class RBFInterpolatorTestCase(TestCase):
    """Tests the RBF interpolator Python wrapper functionality"""

    def test_comparison_with_scipy(self):
        """Compare fast interpolator with scipy RBFInterpolator"""
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = np.array([0.0, 1.0, 1.0, 0.0])

        rbf = RBFInterpolator(centers, values, neighbors=None)

        grid_shape = (5, 5)

        fast_result = fast_rbf_interpolation_on_grid(rbf, grid_shape)

        y_grid, x_grid = np.mgrid[0 : grid_shape[0], 0 : grid_shape[1]]
        scipy_result = rbf(np.column_stack([y_grid.ravel(), x_grid.ravel()])).reshape(grid_shape)

        np.testing.assert_allclose(fast_result, scipy_result, rtol=1e-5, atol=1e-5)

    def test_quadratic_polynomial(self):
        """Test with quadratic polynomial terms"""
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.5, 1.5]])
        values = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 0.5])

        rbf = RBFInterpolator(centers, values, neighbors=None, degree=2)
        grid_shape = (5, 5)

        result = fast_rbf_interpolation_on_grid(rbf, grid_shape)

        self.assertEqual(result.shape, grid_shape)
        self.assertTrue(np.isfinite(result).all())

    def test_error_conditions(self):
        """Test that proper errors are raised for invalid inputs"""
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = np.array([1.0, 1.0, 1.0, 1.0])

        rbf = RBFInterpolator(centers, values, neighbors=3)

        grid_shape = (5, 5)

        with self.assertRaises(ValueError) as cm:
            fast_rbf_interpolation_on_grid(rbf, grid_shape)

        self.assertIn("neighbors", str(cm.exception).lower())

    def test_small_grid(self):
        """Test small grid edge case"""
        centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
        values = np.array([1.0, 0.8, 0.8, 0.6])

        rbf = RBFInterpolator(centers, values, neighbors=None)

        result = fast_rbf_interpolation_on_grid(rbf, (3, 3))

        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.isfinite(result).all())


class MemoryTestCase(MemoryTestCase):
    """Test for memory leaks"""

    pass


def setup_module(module):
    init()


if __name__ == "__main__":
    init()
    unittest.main()
