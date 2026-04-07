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

"""
# Radial Basis Function (RBF) Interpolation Module

This module provides high-performance RBF interpolation for regular grids
using a Rust backend with PyO3 bindings.

## Mathematical Background

The RBF interpolant is defined as:

    ``s(x) = Σᵢ wᵢ ϕ(||x - cᵢ||) + Σⱼ pⱼ(x)``

where:
- ϕ is the thin plate spline (TPS) kernel: ϕ(r) = r² ln(r)
- wᵢ are coefficients solved from the linear system
- cᵢ are center coordinates (data points)
- pⱼ(x) are polynomial terms ensuring uniqueness

The polynomial ensures uniqueness as described in the scipy documenation on
radial basis functions:
`Radial Basis Functions <https://docs.scipy.org/doc/scipy/reference/generated/
scipy.interpolate.RBFInterpolator.html`>__.

### Thin Plate Spline Kernel

The TPS kernel ϕ(r) = r² ln(r) would require the use of a sqrt opertation at
every point which can get expensive. Instead the mathematical equivalence:

    ``ln(r) = ln(r²) / 2```

allows computing the kernel in a single step. A threshold (1e-100) prevents
logarithm of zero for numerical stability.

### Normalization

Coordinates are normalized to improve numerical conditioning:

    ``x_norm = (x - shift_x) / scale_x``
    ``y_norm = (y - shift_y) / scale_y``

This centers data around the origin and scales to unit variance.

### Polynomial Terms

The polynomial terms pⱼ(x) ensure the interpolant is unique. The powers
matrix defines which monomials are included:
- [[0, 0]] → constant term (1)
- [[1, 0]] → linear x term
- [[0, 1]] → linear y term
- [[2, 0]] → quadratic x² term
- etc.

## Algorithm

For each grid point (x, y):

1. Compute RBF contribution: Σᵢ wᵢ ϕ(||x - cᵢ||²)
2. Compute polynomial contribution: Σⱼ βⱼ x^exp_x y^exp_y
3. Return sum: s(x, y) = RBF + polynomial

The algorithm is O(n_centers × height × width) and optimized for regular
grids by pre-computing powers and normalization.
"""

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._rubinoxide import _rbf_interpolator


def fast_rbf_interpolation_on_grid(rbf_interp, grid_shape: tuple[int, int]) -> NDArray[np.floating[Any]]:
    """Evaluate a fitted RBFInterpolator on a regular grid using a
    high-performance Rust backend via PyO3.

    This function is optimized for massive regular grids and should be
    significantly faster than calling the standard RBFInterpolator.__call__
    method.

    Parameters
    ----------
    rbf_interp : `scipy.interpolate.RBFInterpolator`
        A fitted interpolator. Must be created with ``neighbors=None``
        (this is required for the optimized Rust backend).
    grid_shape : `tuple` of `int`, `int`
        The (height, width) of the regular grid to evaluate on.

    Returns
    -------
    return : `numpy.ndarray`
        A 2D array of shape ``grid_shape`` containing the interpolated values.

    Raises
    ------
    `ValueError`
        If `rbf_interp.neighbors` is not `None` (Rust backend requires
        neighbors=None for the RBFInterpolator).

    Notes
    -----
    The Rust backend computes the RBF interpolant:

        ``s(x) = Σᵢ wᵢ ϕ(||x - cᵢ||) + Σⱼ pⱼ(x)``

    where ϕ is the thin plate spline kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import RBFInterpolator
    >>> from lsst.rubinoxide import fast_rbf_interpolation_on_grid
    >>> centers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> values = np.array([1.0, 0.5, 0.5])
    >>> rbf = RBFInterpolator(centers, values, neighbors=None)
    >>> result = fast_rbf_interpolation_on_grid(rbf, (10, 10))
    """
    if rbf_interp.neighbors is not None:
        raise ValueError(
            "The Rust backend requires RBFInterpolator to be created with neighbors=None. "
            "Consider using RBFInterpolator.__call__ instead for neighbor-based interpolation."
        )

    height, width = grid_shape

    # --- 1. Extract parameters from the SciPy object ---
    # Input data points (centers) - shape: (n_centers, 2)
    centers = rbf_interp.y

    # Solved coefficients from the linear system
    # Shape: (n_centers + n_poly_terms, 1) - contains both RBF weights and
    # polynomial coeffs
    # Because we are using scipy's implementation of the solver we need
    # to reach in to get the coefficents. In the future we will look into
    # either upstreaming public acess, upstreaming this evaluator, or
    # implementing our own fitter.
    coeffs_flat = rbf_interp._coeffs.ravel()

    n_centers = centers.shape[0]

    # Split the coefficients into RBF weights and polynomial coefficients
    weights = coeffs_flat[:n_centers]
    poly_coeffs = coeffs_flat[n_centers:]

    # Normalization parameters
    shift = rbf_interp._shift
    scale = rbf_interp._scale

    # Polynomial coefficients
    # The powers matrix from scipy has shape (n_poly_terms, 2) with columns
    # [exp_x, exp_y]
    # We reverse columns to get [exp_y, exp_x] to match Rust's expected layout
    powers = np.ascontiguousarray(rbf_interp.powers, dtype=np.int64)
    powers = powers[:, ::-1]

    # --- 2. Prepare data for Rust ---
    # PyO3 works best with contiguous arrays. Flatten the 2D center array
    # to a 1D array in [y0, x0, y1, x1, ...] order (Rust expects this layout).
    centers_flat = centers.ravel()

    # --- 3. Call the Rust function ---
    # Data ownership is transferred to Rust via PyO3 bindings.
    # The Rust function returns a new array which we reshape to 2D.
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
