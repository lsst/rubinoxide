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
# Radial Basis Function (RBF) Interpolation Module

This module provides high-performance RBF interpolation for regular grids
using a Rust backend with PyO3 bindings.

## Mathematical Background

The RBF interpolant is defined as:

    s(x) = Σᵢ wᵢ ϕ(||x - cᵢ||) + Σⱼ pⱼ(x)

where:
- ϕ is the thin plate spline (TPS) kernel: ϕ(r) = r² ln(r²)
- wᵢ are coefficients solved from the linear system
- cᵢ are center coordinates (data points)
- pⱼ(x) are polynomial terms ensuring uniqueness

### Thin Plate Spline Kernel

The TPS kernel ϕ(r) = r² ln(r²) is "branchless" because it operates on
the squared distance r² instead of r. This avoids expensive sqrt operations.
The mathematical equivalence:

    ln(r) = ln(r²) / 2

allows computing the kernel in a single step. A threshold (1e-100) prevents
logarithm of zero for numerical stability.

### Normalization

Coordinates are normalized to improve numerical conditioning:

    x_norm = (x - shift_x) / scale_x
    y_norm = (y - shift_y) / scale_y

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
*/

use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::fmt;

/// Error type for RBF interpolation operations.
#[derive(Debug)]
pub enum RbfError {
    /// Dimension mismatch between powers matrix and polynomial coefficients
    DimensionMismatch {
        powers_rows: usize,
        poly_coeffs_len: usize,
    },
    /// Invalid powers array shape (expected 2D)
    InvalidPowersShape,
}

impl fmt::Display for RbfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RbfError::DimensionMismatch {
                powers_rows,
                poly_coeffs_len,
            } => write!(
                f,
                "Mismatch: powers rows ({}) != poly_coeffs length ({})",
                powers_rows, poly_coeffs_len
            ),
            RbfError::InvalidPowersShape => {
                write!(f, "Expected 2D powers array for 2D interpolation.")
            }
        }
    }
}

impl std::error::Error for RbfError {}

impl From<RbfError> for PyErr {
    fn from(err: RbfError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

// The branchless, inlineable kernel logic.
// We use a constant to avoid magic numbers and a clamp to avoid NaNs.
const TPS_LOG_THRESHOLD: f64 = 1e-100;

/// Evaluate the RBF interpolant on a regular grid.
///
/// This function computes the RBF interpolant:
///
///     s(x, y) = Σᵢ wᵢ ϕ(||x - cᵢ||²) + Σⱼ βⱼ x^exp_x y^exp_y
///
/// where ϕ(r²) = r² ln(r²) is the thin plate spline kernel.
///
/// # Arguments
/// * height: Grid height (output array rows)
/// * width: Grid width (output array columns)
/// * centers: 1D view of flattened center coordinates [y0, x0, y1, x1, ...]
/// * weights: 1D view of RBF weights (one per center)
/// * poly_coeffs: 1D view of polynomial coefficients (length = num_poly_terms)
/// * shift: 1D view of normalization parameters [shift_y, shift_x]
/// * scale: 1D view of normalization parameters [scale_y, scale_x]
/// * powers: 2D view of exponents [[exp_y, exp_x], ...] defining polynomial terms
///
/// # Returns
/// * Result<Array2<f64>, RbfError>: 2D array of shape (height, width) with interpolated values
///
/// # Errors
/// * RbfError::DimensionMismatch: If powers.shape()[0] != poly_coeffs.len()
/// * RbfError::InvalidPowersShape: If powers.shape()[1] != 2
///
/// # Notes
/// * The TPS kernel uses branchless formulation: ϕ(r²) = r² ln(r²)
/// * Normalization improves numerical stability: x_norm = (x - shift) / scale
/// * Polynomial powers are pre-computed for each grid row/column
/// * Algorithm complexity: O(n_centers × height × width)
fn compute_rbf_grid_dynamic(
    height: usize,
    width: usize,
    centers: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    poly_coeffs: ArrayView1<f64>,
    shift: ArrayView1<f64>,
    scale: ArrayView1<f64>,
    powers: ArrayView2<i64>,
) -> Result<Array2<f64>, RbfError> {
    let num_centers = weights.len();
    let num_poly_terms = poly_coeffs.len();

    // Verify dimensions
    if powers.shape()[0] != num_poly_terms {
        return Err(RbfError::DimensionMismatch {
            powers_rows: powers.shape()[0],
            poly_coeffs_len: num_poly_terms,
        });
    }
    if powers.shape()[1] != 2 {
        return Err(RbfError::InvalidPowersShape);
    }

    // Determine max degrees to size our power arrays
    let mut max_x_exp: usize = 0;
    let mut max_y_exp: usize = 0;
    for j in 0..num_poly_terms {
        let exp_x = powers[[j, 0]];
        let exp_y = powers[[j, 1]];
        if exp_x as usize > max_x_exp {
            max_x_exp = exp_x as usize;
        }
        if exp_y as usize > max_y_exp {
            max_y_exp = exp_y as usize;
        }
    }

    // Pre-compute normalized centers
    // Normalization improves numerical stability: x_norm = (x - shift) / scale
    let mut centers_norm = Array2::zeros((num_centers, 2));
    for i in 0..num_centers {
        let y_c = centers[2 * i];
        let x_c = centers[2 * i + 1];
        centers_norm[[i, 0]] = (y_c - shift[0]) / scale[0];
        centers_norm[[i, 1]] = (x_c - shift[1]) / scale[1];
    }

    // Pre-allocate power arrays
    // These buffers are reused for every row/column to avoid reallocation
    // x_powers[k] = x_norm^k, y_powers[k] = y_norm^k (Horner's method)
    let mut x_powers = vec![0.0; max_x_exp + 1];
    let mut y_powers = vec![0.0; max_y_exp + 1];

    // Pre-allocate y component buffers
    // y_r_comp: normalized y distance squared (y_norm - y_center_norm)^2
    // y_r_comp_nn: original y distance squared (y_coord - y_center)^2
    let mut y_r_comp = vec![0.0; num_centers];
    let mut y_r_comp_nn = vec![0.0; num_centers];

    let mut results = Array2::zeros((height, width));

    for y_coord in 0..height {
        let y_norm = (y_coord as f64 - shift[0]) / scale[0];

        // Pre-compute y^0, y^1, ..., y^max_y_exp using Horner's method
        // y^k = y^(k-1) * y_norm
        y_powers[0] = 1.0;
        for k in 1..=max_y_exp {
            y_powers[k] = y_powers[k - 1] * y_norm;
        }

        for r_y in 0..num_centers {
            // Pre-compute squared y distances for this row
            // y_r_comp: normalized distance, y_r_comp_nn: original distance
            y_r_comp[r_y] = (y_norm - centers_norm[[r_y, 0]]).powf(2.0);
            y_r_comp_nn[r_y] = (y_coord as f64 - centers[2 * r_y]).powf(2.0);
        }

        for x_coord in 0..width {
            // Normalize grid point
            let x_norm = (x_coord as f64 - shift[1]) / scale[1];

            // Pre-compute x^0, x^1, ..., x^max_x_exp using Horner's method
            x_powers[0] = 1.0;
            for k in 1..=max_x_exp {
                x_powers[k] = x_powers[k - 1] * x_norm;
            }

            // 1. RBF Contribution: Σᵢ wᵢ ϕ(||x - cᵢ||²)
            // The TPS kernel ϕ(r²) = r² ln(r²) is "branchless" - no sqrt needed
            // Mathematical equivalence: ln(r) = ln(r²) / 2
            // Division by 2 is part of the TPS formulation
            let mut rbf_sum = 0.0;
            for i in 0..num_centers {
                let c_x = centers[2 * i + 1];
                let dx = x_coord as f64 - c_x;
                let r_sq = dx * dx + y_r_comp_nn[i];

                // Branchless TPS: use r² directly to avoid sqrt
                // Clamp to avoid ln(0) for numerical stability
                let clamped_r_sq = r_sq.max(TPS_LOG_THRESHOLD);
                let kernel_val = clamped_r_sq * clamped_r_sq.ln();

                rbf_sum += weights[i] * kernel_val;
            }

            // TPS formulation includes division by 2
            rbf_sum /= 2.0;

            // 2. Polynomial Contribution: Σⱼ βⱼ x^exp_x y^exp_y
            // Uses pre-computed powers for efficiency (Horner's method)
            let mut poly_sum = 0.0;
            for j in 0..num_poly_terms {
                let exp_x = powers[[j, 0]] as usize;
                let exp_y = powers[[j, 1]] as usize;

                // Lookup pre-computed powers
                let term_val = x_powers[exp_x] * y_powers[exp_y];

                poly_sum += poly_coeffs[j] * term_val;
            }

            results[[y_coord, x_coord]] = rbf_sum + poly_sum;
        }
    }

    Ok(results)
}

/// Evaluate the RBF interpolant on a regular grid.
///
/// This is the PyO3 wrapper function that exposes the Rust backend to Python.
/// It extracts views from Python arrays and passes them to the core computation.
///
/// Parameters
/// ----------
/// height : `int`
///     Grid height (output array rows).
/// width : `int`
///     Grid width (output array columns).
/// centers : `NDArray[float]`
///     1D array of flattened center coordinates [y0, x0, y1, x1, ...].
/// weights : `NDArray[float]`
///     1D array of RBF weights (one per center).
/// poly_coeffs : `NDArray[float]`
///     1D array of polynomial coefficients.
/// shift : `NDArray[float]`
///     1D array of normalization shifts [shift_y, shift_x].
/// scale`` : `NDArray[float]`
///     1D array of normalization scales [scale_y, scale_x].
/// powers : `NDArray[float]`
///     2D array of exponents [[exp_y, exp_x], ...] defining polynomial terms.
///
/// Returns
/// -------
/// result : `NDArray[float]`
///     2D numpy array of shape (height, width) with interpolated values.
///
/// Raises
/// ------
/// ``ValueError``
///     If `powers.shape[0] != len(poly_coeffs)` (dimension mismatch).
///     If `powers.shape[1] != 2` (expected 2D powers array).
///
/// Notes
/// -----
/// This function calls the Rust backend `compute_rbf_grid_dynamic` which
/// computes the RBF interpolant:
///
///     s(x, y) = Σᵢ wᵢ ϕ(||x - cᵢ||²) + Σⱼ βⱼ x^exp_x y^exp_y
///
/// where ϕ(r²) = r² ln(r²) is the thin plate spline kernel.
#[pyfunction]
fn fast_rbf_grid_ndarray<'py>(
    py: Python<'py>,
    height: usize,
    width: usize,
    centers: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    poly_coeffs: PyReadonlyArray1<f64>,
    shift: PyReadonlyArray1<f64>,
    scale: PyReadonlyArray1<f64>,
    powers: PyReadonlyArray2<i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Extract views from the Python arrays
    let centers_view = centers.as_array();
    let weights_view = weights.as_array();
    let poly_coeffs_view = poly_coeffs.as_array();
    let shift_view = shift.as_array();
    let scale_view = scale.as_array();
    let powers_view = powers.as_array();

    // Perform the computation and propagate errors
    let result = compute_rbf_grid_dynamic(
        height,
        width,
        centers_view,
        weights_view,
        poly_coeffs_view,
        shift_view,
        scale_view,
        powers_view,
    )?;

    Ok(result.to_pyarray(py))
}

pub fn create_rbf_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let rbf_module = PyModule::new(parent_module.py(), "_rbf_interpolator")?;
    rbf_module.add_function(wrap_pyfunction!(fast_rbf_grid_ndarray, &rbf_module)?)?;
    parent_module.add_submodule(&rbf_module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_delta;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_tps_log_threshold() {
        let centers = arr1(&[0.0, 0.0]);
        let weights = arr1(&[1.0]);
        let poly_coeffs = arr1(&[0.0]);
        let shift = arr1(&[0.0, 0.0]);
        let scale = arr1(&[1.0, 1.0]);
        let powers = arr2(&[[0i64, 0i64]]);

        let result = compute_rbf_grid_dynamic(
            5,
            5,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        )
        .unwrap();

        assert!(result[[2, 2]] > 0.0);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_compute_rbf_grid_multiple_centers() {
        let centers = arr1(&[1.0, 1.0, 3.0, 3.0]);
        let weights = arr1(&[1.0, 1.0]);
        let poly_coeffs = arr1(&[0.0]);
        let shift = arr1(&[0.0, 0.0]);
        let scale = arr1(&[1.0, 1.0]);
        let powers = arr2(&[[0i64, 0i64]]);

        let result = compute_rbf_grid_dynamic(
            5,
            5,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        )
        .unwrap();

        assert!(result[[1, 1]] > 0.0);
        assert!(result[[3, 3]] > 0.0);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_normalization() {
        let centers = arr1(&[1.0, 1.0, 3.0, 3.0]);
        let weights = arr1(&[1.0, 1.0]);
        let poly_coeffs = arr1(&[0.0]);
        let shift = arr1(&[2.0, 2.0]);
        let scale = arr1(&[2.0, 2.0]);
        let powers = arr2(&[[0i64, 0i64]]);

        let result = compute_rbf_grid_dynamic(
            4,
            4,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        )
        .unwrap();

        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_powers_matrix() {
        let centers = arr1(&[0.0, 0.0]);
        let weights = arr1(&[0.0]);
        let poly_coeffs = arr1(&[1.0, 2.0, 3.0]);
        let shift = arr1(&[0.0, 0.0]);
        let scale = arr1(&[1.0, 1.0]);
        let powers = arr2(&[[0i64, 0i64], [1i64, 0i64], [0i64, 1i64]]);

        let result = compute_rbf_grid_dynamic(
            2,
            2,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        )
        .unwrap();

        assert_delta!(result[[0, 0]], 1.0, 1e-15);
        assert_delta!(result[[0, 1]], 1.0 + 2.0 * 1.0, 1e-15);
        assert_delta!(result[[1, 0]], 1.0 + 3.0 * 1.0, 1e-15);
        assert_delta!(result[[1, 1]], 1.0 + 2.0 * 1.0 + 3.0 * 1.0, 1e-15);
    }

    #[test]
    fn test_empty_centers() {
        let centers = arr1(&[]);
        let weights = arr1(&[]);
        let poly_coeffs = arr1(&[1.0]);
        let shift = arr1(&[0.0, 0.0]);
        let scale = arr1(&[1.0, 1.0]);
        let powers = arr2(&[[0i64, 0i64]]);

        let result = compute_rbf_grid_dynamic(
            2,
            2,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        )
        .unwrap();

        assert_delta!(result[[0, 0]], 1.0, 1e-15);
        assert_delta!(result[[0, 1]], 1.0, 1e-15);
        assert_delta!(result[[1, 0]], 1.0, 1e-15);
        assert_delta!(result[[1, 1]], 1.0, 1e-15);
    }

    #[test]
    fn test_invalid_powers_shape() {
        let centers = arr1(&[0.0, 0.0]);
        let weights = arr1(&[1.0]);
        let poly_coeffs = arr1(&[1.0]);
        let shift = arr1(&[0.0, 0.0]);
        let scale = arr1(&[1.0, 1.0]);
        let powers = arr2(&[[0i64, 0i64, 0i64]]);

        let result = compute_rbf_grid_dynamic(
            2,
            2,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        );

        assert!(matches!(result, Err(RbfError::InvalidPowersShape)));
    }

    #[test]
    fn test_mismatched_powers_poly() {
        let centers = arr1(&[0.0, 0.0]);
        let weights = arr1(&[1.0]);
        let poly_coeffs = arr1(&[1.0, 2.0]);
        let shift = arr1(&[0.0, 0.0]);
        let scale = arr1(&[1.0, 1.0]);
        let powers = arr2(&[[0i64, 0i64]]);

        let result = compute_rbf_grid_dynamic(
            2,
            2,
            centers.view(),
            weights.view(),
            poly_coeffs.view(),
            shift.view(),
            scale.view(),
            powers.view(),
        );

        assert!(matches!(result, Err(RbfError::DimensionMismatch { .. })));
    }
}
