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
        # setup an rgb image with a single pixel
        test_image = np.array([[[0.2, 0.42, 0.81]]])

        # declare an xy whitepoint
        whitepoint = (0.31, 0.32)

        # expected oklab value
        expected = np.array([[[0.73695252, -0.03689747, -0.11662942]]])

        oklab_values = rgb.RGB_to_Oklab(test_image, whitepoint)
        np.testing.assert_allclose(oklab_values, expected, atol=1e-4, rtol=0)

        round_trip = rgb.Oklab_to_RGB(oklab_values, whitepoint)

        np.testing.assert_allclose(round_trip, test_image, atol=1e-4, rtol=0)


class MemoryTestCase(MemoryTestCase):
    """Test for memory leaks"""

    pass


if __name__ == "__main__":
    init()
    unittest.main()
