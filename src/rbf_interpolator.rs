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

use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// The branchless, inlineable kernel logic.
// We use a constant to avoid magic numbers and a clamp to avoid NaNs.
const TPS_LOG_THRESHOLD: f64 = 1e-100;

/// The core computation function.
///
/// # Arguments
/// * `height`, `width`: Grid dimensions
/// * `centers`: 1D view of flattened center coordinates [y0, x0, ...]
/// * `weights`: 1D view of RBF weights
/// * `poly_coeffs`: 1D view of ALL polynomial coefficients (length = num_terms)
/// * `shift`, `scale`: 1D view of normalization parameters
/// * `powers`: 2D view of exponents [[e_x, e_y], ...] defining the polynomial terms
fn compute_rbf_grid_dynamic(
    height: usize,
    width: usize,
    centers: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    poly_coeffs: ArrayView1<f64>,
    shift: ArrayView1<f64>,
    scale: ArrayView1<f64>,
    powers: ArrayView2<i64>, // Exponents are integers
) -> Array2<f64> {
    let num_centers = weights.len();
    let num_poly_terms = poly_coeffs.len();

    // Verify dimensions
    if powers.shape()[0] != num_poly_terms {
        panic!(
            "Mismatch: powers rows ({}) != poly_coeffs length ({})",
            powers.shape()[0],
            num_poly_terms
        );
    }
    if powers.shape()[1] != 2 {
        panic!("Expected 2D powers array for 2D interpolation.");
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
    let mut centers_norm = Array2::zeros((num_centers, 2));
    for i in 0..num_centers {
        let y_c = centers[2 * i];
        let x_c = centers[2 * i + 1];
        centers_norm[[i, 0]] = (y_c - shift[0]) / scale[0];
        centers_norm[[i, 1]] = (x_c - shift[1]) / scale[1];
    }

    // Pre-allocate power arrays
    // We will reuse these buffers for every row/column to avoid reallocation
    let mut x_powers = vec![0.0; max_x_exp + 1];
    let mut y_powers = vec![0.0; max_y_exp + 1];

    // Pre-allocate y component of r array
    let mut y_r_comp = vec![0.0; num_centers];
    let mut y_r_comp_nn = vec![0.0; num_centers];

    let mut results = Array2::zeros((height, width));

    for y_coord in 0..height {
        let y_norm = (y_coord as f64 - shift[0]) / scale[0];

        // Pre-compute y^0, y^1, ..., y^max_y_exp
        // y^0 = 1.0
        y_powers[0] = 1.0;
        for k in 1..=max_y_exp {
            y_powers[k] = y_powers[k - 1] * y_norm;
        }

        for r_y in 0..num_centers {
            y_r_comp[r_y] = (y_norm - centers_norm[[r_y, 0]]).powf(2.0);
            y_r_comp_nn[r_y] = (y_coord as f64 - centers[2 * r_y]).powf(2.0);
        }

        for x_coord in 0..width {
            // Normalize grid point
            let x_norm = (x_coord as f64 - shift[1]) / scale[1];

            // Pre-compute x^0, x^1, ..., x^max_x_exp
            x_powers[0] = 1.0;
            for k in 1..=max_x_exp {
                x_powers[k] = x_powers[k - 1] * x_norm;
            }

            // 1. RBF Contribution
            let mut rbf_sum = 0.0;
            for i in 0..num_centers {
                let c_x = centers[2 * i + 1];
                let dx = x_coord as f64 - c_x;
                let r_sq = dx * dx + y_r_comp_nn[i];

                // Branchless TPS
                let clamped_r_sq = r_sq.max(TPS_LOG_THRESHOLD);
                // let r = clamped_r_sq.sqrt();
                // need ln(r) but that is the same as ln(r^2)/2 which is
                // faster than calculating sqrt, more so this is linear
                // so the division can happen after the sum to reduce
                // operations
                let kernel_val = clamped_r_sq * clamped_r_sq.ln();

                rbf_sum += weights[i] * kernel_val;
            }

            rbf_sum /= 2.0;

            // 2. Polynomial Contribution (DYNAMIC)
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

    results
}

#[pyfunction]
fn fast_rbf_grid_ndarray<'py>(
    py: Python<'py>,
    height: usize,
    width: usize,
    // Use PyReadonlyArray1 to ensure we don't modify the Python data and get a view
    centers: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    poly_coeffs: PyReadonlyArray1<f64>,
    shift: PyReadonlyArray1<f64>,
    scale: PyReadonlyArray1<f64>,
    powers: PyReadonlyArray2<i64>,
) -> Bound<'py, PyArray2<f64>> {
    // Extract views from the Python arrays
    let centers_view = centers.as_array();
    let weights_view = weights.as_array();
    let poly_coeffs_view = poly_coeffs.as_array();
    let shift_view = shift.as_array();
    let scale_view = scale.as_array();
    let powers_view = powers.as_array();

    // Perform the computation
    let result = compute_rbf_grid_dynamic(
        height,
        width,
        centers_view,
        weights_view,
        poly_coeffs_view,
        shift_view,
        scale_view,
        powers_view,
    );

    result.to_pyarray(py)
}

pub fn create_rbf_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let rbf_module = PyModule::new(parent_module.py(), "_rbf_interpolator")?;
    rbf_module.add_function(wrap_pyfunction!(fast_rbf_grid_ndarray, &rbf_module)?)?;
    parent_module.add_submodule(&rbf_module)
}
