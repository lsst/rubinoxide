# For copyright information see the COPYRIGHT file included in the top-level
# directory of this distribution.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#      1. Redistributions of source code must retain the included copyright
#         notice, this list of conditions and the following disclaimer.
#
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

import logging
import time

import numpy as np

from ._rubinoxide import _rbf_interpolator


def fast_rbf_interpolation_on_grid(rbf_interp, grid_shape: tuple):
    """Evaluate a fitted RBFInterpolator on a regular grid using a
    high-performance Rust backend via PyO3.

    This function is optimized for massive regular grids and should be
    significantly faster than calling the standard RBFInterpolator.__call__
    method.

    Parameters
    ----------
        rbf_interp : `scipy.interpolate.RBFInterpolator`
            A fitted interpolator. IMPORTANT: This must be an instance that
            was fitted with neighbors=None.
        grid_shape : `tuple` of `int, int`
            The (height, width) of the regular grid to evaluate on.

    Returns
    -------
        result : `np.ndarray`
            np.ndarray: A 2D array of shape `grid_shape` containing the
            interpolated values.
    """
    if rbf_interp.neighbors is not None:
        raise ValueError(
            "This optimized function only works when the RBFInterpolator was created with neighbors=None."
        )

    height, width = grid_shape

    # --- 1. Extract parameters from the SciPy object ---
    # Input data points (centers)
    # Shape: (n_centers, 2)
    centers = rbf_interp.y

    # Solved coefficients from the linear system
    # Shape: (n_centers + n_poly_terms, 1)
    # Flatten it to a 1D array.
    coeffs_flat = rbf_interp._coeffs.ravel()

    n_centers = centers.shape[0]

    # Split the coefficients into RBF weights and polynomial coefficients
    weights = coeffs_flat[:n_centers]
    poly_coeffs = coeffs_flat[n_centers:]

    # Normalization parameters
    shift = rbf_interp._shift
    scale = rbf_interp._scale

    # Polynomial coefficients
    powers = np.ascontiguousarray(rbf_interp.powers, dtype=np.int64)
    powers = powers[:, ::-1]

    # --- 2. Prepare data for Rust ---
    # PyO3 works best with Vec<f64> or similar contiguous arrays.
    # Flatten the 2D center array to a 1D array: [y0, x0, y1, x1, ...]
    centers_flat = centers.ravel()

    # --- 3. Call the Rust function ---
    logging.debug("Calling Rust backend for interpolation...")
    start_time = time.time()

    results_flat = _rbf_interpolator.fast_rbf_grid_ndarray(
        height,
        width,
        centers_flat,
        weights,
        poly_coeffs,
        shift,
        scale,
        powers,
    )

    end_time = time.time()
    logging.debug(f"Rust backend finished in {end_time - start_time:.4f} seconds.")

    # Reshape to the desired 2D grid shape
    results_grid = results_flat.reshape((height, width))

    return results_grid
