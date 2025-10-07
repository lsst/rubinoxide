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

// many of the names in this file follow established capitalization practices.
// Compile warnings should be suppressed for this module.
#![allow(non_snake_case)]

extern crate openblas_src;
use log;
use ndarray::linalg::general_mat_mul;
use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

// from CIE RGB to XYZ
// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static RGB_TO_XYZ_MATRIX: [f64; 9] = [
    4.86570949e-01, 2.65667693e-01, 1.98217285e-01,
    2.28974564e-01, 6.91738522e-01, 7.92869141e-02,
    -3.97207552e-17, 4.51133819e-02, 1.04394437e+00,
];

// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static XYZ_TO_RGB_MATRIX: [f64; 9] = [
    2.49349691, -0.93138362, -0.40271078,
    -0.82948897, 1.76266406, 0.02362469,
    0.03584583, -0.07617239, 0.95688452,
];

// transfer function to shift whitepoint
// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static CAT_BRADFORD: [f64; 9] = [
    0.8951, 0.2664, -0.1614,
    -0.7502, 1.7135, 0.0367,
    0.0389, -0.0685, 1.0296,
];

// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static XYZ_TO_LMS: [f64; 9] = [
    0.81893301, 0.36186674, -0.12885971,
    0.03298454, 0.92931187, 0.03614564,
    0.0482003, 0.26436627, 0.63385171,
];

// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static LMS_TO_LAB: [f64; 9] = [
    0.21045426, 0.79361779, -0.00407205,
    1.9779985, -2.42859221, 0.45059371,
    0.02590404, 0.78277177, -0.80867577,
];

// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static LAB_TO_LMS: [f64; 9] = [
    1., 0.39633779, 0.21580376,
    1.00000001, -0.10556134, -0.06385417,
    1.00000005, -0.08948418, -1.29148554,
];

// keep rust fmt from re-formatting what is a 3x3 matrix
#[rustfmt::skip]
static LMS_TO_XYZ: [f64; 9] = [
    1.22701385, -0.55779998, 0.28125615,
    -0.04058018, 1.11225687, -0.07167668,
    -0.07638128, -0.42148198, 1.58616322,
];

// Transform coordinates in xy coordinates to XYZ coordinates
fn xy_to_XYZ(white_point: [f64; 2]) -> Array1<f64> {
    let cie_Y_y = 1.0 / white_point[1];
    array![
        white_point[0] * cie_Y_y,
        1.0,
        (1.0 - (white_point[0] + white_point[1])) * cie_Y_y
    ]
}

// Transfrom from one whitepoint to another
fn transform_whitepoint(from: &Array1<f64>, to: &Array1<f64>) -> Array2<f64> {
    let M = ArrayView::from_shape((3, 3), &CAT_BRADFORD).unwrap();
    let RGB_w = M.dot(from);
    let RGB_wr = M.dot(to);
    let div = RGB_wr / RGB_w;
    let D = Array2::from_diag(&div);
    let M_CAT = M.inv().unwrap().dot(&D);
    M_CAT.dot(&M)
}

/// Convert an image of RGB values into OKlab values
///
/// The image format must be a 3D image with the rgb values along the third
/// axis. The values must be double precision.
///
/// Parameters
/// ----------
/// image : array-like
///     A numpy array of 3 dimensions with the third dimension corresponding
///     to the rgb values in 0-1 range in double precision.
/// illuminant_xyz : tuple of double
///     the illuminant of the xyz space in xy coordinates. The first element is
///     x, and the second is y.
///
/// Returns
/// -------
/// image : array like
///    A numpy array of 3 dimensions with the third dimension corresponding
///    to the converted Oklab values in double precision
#[pyfunction]
fn RGB_to_Oklab<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    cie_whitepoint: [f64; 2],
) -> Bound<'py, PyArray3<f64>> {
    log::debug!("Converting RGB image to Oklab colorspace\n");
    let illuminant_RGB = [0.31272, 0.32903];

    let working_array = image.as_array();
    let (height, width, depth) = working_array.dim();
    let mut output: Array2<f64> = Array2::zeros([height * width, depth]);
    let mut scratch: Array2<f64> = Array2::zeros([height * width, depth]);

    let cie_XYZ = xy_to_XYZ(illuminant_RGB);
    let wp_XYZ = xy_to_XYZ(cie_whitepoint);
    let new_whitepoint = transform_whitepoint(&cie_XYZ, &wp_XYZ);

    let rgb_transformation_matrix = ArrayView::from_shape((3, 3), &RGB_TO_XYZ_MATRIX).unwrap();
    let xyz_to_lms = ArrayView::from_shape((3, 3), &XYZ_TO_LMS).unwrap();
    let lms_to_lab = ArrayView::from_shape((3, 3), &LMS_TO_LAB).unwrap();

    // convert RGB to XYZ
    // shift the whitepoint
    // turn to intermediate representation
    let combined = xyz_to_lms.dot(&(new_whitepoint.dot(&rgb_transformation_matrix)));
    let pow = 1.0 / 3.0;

    // Because the transformation we are interested in is over color, and
    // parallel over pixels, the operation is faster if we collapse the 2
    // spatial dimensions into one such that transformation operation
    // becomes a matrix product. The array is later restored to the correct
    // shape.
    let reshaped = working_array
        .into_shape_with_order([height * width, depth])
        .unwrap();
    general_mat_mul(1.0, &reshaped, &(combined.t()), 0.0, &mut scratch);
    scratch.map_inplace(|x| *x = x.signum() * x.abs().powf(pow));
    general_mat_mul(1.0, &scratch, &(lms_to_lab.t()), 0.0, &mut output);
    output
        .into_shape_with_order((height, width, depth))
        .unwrap()
        .into_pyarray(py)
}

/// Convert an image of Oklab values into RGB
///
/// The image format must be a 3D image with the oklab values along the third
/// axis. The values must be double precision.
///
/// Parameters
/// ----------
/// image : array-like
///     A numpy array of 3 dimensions with the third dimension corresponding
///     to the oklab values
/// illuminant_xyz : tuple of double
///     the illuminant of the xyz space in xy coordinates. The first element is
///     x, and the second is y.
///
/// Returns
/// -------
/// image : array like
///    A numpy array of 3 dimensions with the third dimension corresponding
///    to the converted RGB values
#[pyfunction]
fn Oklab_to_RGB<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<f64>,
    illuminant_xyz: [f64; 2],
) -> Bound<'py, PyArray3<f64>> {
    log::debug!("Converting Oklab image to RGB colorspace\n");
    let working_array = image.as_array();
    let (height, width, depth) = working_array.dim();

    // pre allocate array
    let mut output: Array2<f64> = Array2::zeros([height * width, depth]);
    let mut scratch: Array2<f64> = Array2::zeros([height * width, depth]);

    let illuminant_D65 = [0.31272, 0.32903];
    let cie_whitepiont = xy_to_XYZ(illuminant_xyz);
    let cie_d65 = xy_to_XYZ(illuminant_D65);
    let whitepoint_trans_matrix = transform_whitepoint(&cie_whitepiont, &cie_d65);

    let lab_to_lms = ArrayView::from_shape((3, 3), &LAB_TO_LMS).unwrap();
    let lms_to_xyz = ArrayView::from_shape((3, 3), &LMS_TO_XYZ).unwrap();

    let xyz_to_rgb = ArrayView::from_shape((3, 3), &XYZ_TO_RGB_MATRIX).unwrap();

    let combined = xyz_to_rgb.dot(&(whitepoint_trans_matrix.dot(&lms_to_xyz)));

    // Because the transformation we are interested in is over color, and
    // parallel over pixels, the operation is faster if we collapse the 2
    // spatial dimensions into one such that transformation operation
    // becomes a matrix product. The array is later restored to the correct
    // shape.
    let reshaped = working_array
        .into_shape_with_order([height * width, depth])
        .expect("first reshape failed\n");
    general_mat_mul(1.0, &reshaped, &(lab_to_lms.t()), 0.0, &mut scratch);
    scratch.mapv_inplace(|x| x.powf(3.0));
    general_mat_mul(1.0, &scratch, &(combined.t()), 0.0, &mut output);
    output
        .into_shape_with_order([height, width, depth])
        .expect("second reshape failed\n")
        .into_pyarray(py)
}

// This function is called by the main python module rubinoxide. Its job
// is to create a new sub moduled named rgb, and bind the declared
// pyfunctions to that module. Finally it adds this submodule into the main
// module.
pub fn create_rgb_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let rgb_module = PyModule::new(parent_module.py(), "rgb")?;
    rgb_module.add_function(wrap_pyfunction!(Oklab_to_RGB, &rgb_module)?)?;
    rgb_module.add_function(wrap_pyfunction!(RGB_to_Oklab, &rgb_module)?)?;
    parent_module.add_submodule(&rgb_module)
}
