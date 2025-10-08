import numpy as np

import unittest

from lsst.rubinoxide import rgb
from lsst.utils.tests import TestCase


class RGBTestCase(TestCase):
    def testPixelConversion(self):
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
