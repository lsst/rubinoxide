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

extern crate openblas_src;
use std::ptr;

use ndarray::NdFloat;
use ndarray::{prelude::*, Zip};
use ndarray::{Array1, Array2};
use num_traits::{NumCast, One};
use ndarray_conv::{ConvFFTExt, ConvMode, FftProcessor, PaddingMode};
use numpy::Element;
use ndarray_linalg::Solve;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use serde::{Serialize, Deserialize};

struct ConvolveCache<T: NdFloat + Default> {
    array: Array2<T>,
    pixel_size: usize,
    basis_size: usize,
    current_index: usize,
    y_hop_cache: Array1<T>,
}

impl<T: NdFloat + Default> ConvolveCache<T> {
    fn new(pixel_size: usize, basis_size: usize) -> Self {
        ConvolveCache {
            array: Array2::<T>::zeros((basis_size, pixel_size)),
            pixel_size,
            basis_size,
            current_index: 0,
            y_hop_cache: Array1::<T>::zeros(pixel_size),
        }
    }

    #[inline]
    fn increment(&mut self) {
        self.current_index = (self.current_index + 1) % self.pixel_size;
    }

    #[inline]
    fn reset(&mut self) {
        self.array.fill(T::default());
        self.current_index = 0;
    }
}

/// Convolve the separable basis kernel with a region of the input image at one output pixel.
///
/// For each basis function, the kernel is separable: `K(i, j) = basis_y[i] * basis_x[j]`.
/// This makes the 2D convolution `output[y, x] = Σ_i Σ_j input[y+i, x+j] * basis_y[i] * basis_x[j]`
/// decomposable into two 1D steps:
///
/// 1. **Y-direction** — for each column `c` in the kernel window, compute
///    `col_conv[c] = Σ_i input[y+i, c] * basis_y[i]`
/// 2. **X-direction** — combine the column convolutions with the x-kernel:
///    `basis_values[bas] = Σ_j basis_x[j] * col_conv[j]`
///
/// This function handles both cache-miss and cache-hit cases for a sliding-window
/// traversal across the image.
///
/// ## Cache miss — full recomputation (new row or non-sequential jump)
///
/// Triggered when the previous pixel was on a different row or was not adjacent
/// (`y != prev_y` or `abs(x - prev_x) != 1`). The entire 2D convolution for all
/// kernel window columns is recomputed from scratch.
///
/// The algorithm uses a **row-major inner loop** to maximize memory locality: for each
/// row `y_start + y_v`, it reads `kernel_size` contiguous input pixels (one for each
/// kernel column) and multiplies them by `basis_y[y_v]`. This means each input row is
/// touched exactly once, rather than the naive column-by-column approach that would
/// read the same row multiple times (once per column). After all rows are accumulated
/// into the ring buffer (`basis_y_cache`), the x-kernel dot product is applied.
///
/// ## Cache hit — sliding window update (consecutive pixel to the right)
///
/// Triggered when moving to the immediately adjacent pixel on the same row. One new
/// column convolution (the rightmost edge entering the kernel window) is computed,
/// then one existing ring buffer slot is overwritten with it. The output is assembled
/// by reading ring slots in the correct order: slots `(cache_offset+1) mod N` through
/// `(N-1)` give the contiguous block, while slots `0` through `cache_offset` give the
/// wrapped-around block. This produces `basis_x[0]*CC[1] + ... + basis_x[(N-1)]*CC[new]`.
///
/// ## Ring buffer design
///
/// The `ConvolveCache` ring has `pixel_size = 2*radius + 1` slots, exactly matching
/// the kernel width. After a cache miss fills all slots sequentially, `current_index`
/// is initialized to 0. On each cache hit:
///
/// 1. The new column convolution is written to `ring[current_index]`, overwriting the
///    oldest value in the ring.
/// 2. The output is assembled by reading from `(current_index+1) % N` to the end of
///    the ring, then wrapping around from 0 to `current_index`. This correctly maps
///    the ring positions to kernel window columns because the overwrite happens at the
///    slot whose value has already been absorbed into the cumulative output.
/// 3. `current_index` is incremented modulo `N` for the next hit.
///
/// After the function returns, regardless of which path was taken, `prev_x` and `prev_y`
/// are updated so the next call can determine whether it is a cache hit or miss.
#[inline(always)]
fn convolve_at_one_point<T: NdFloat + Default>(
    x: &i32,
    y: &i32,
    prev_x: &mut i32,
    prev_y: &mut i32,
    kernel_radius: i32,
    basis_len: usize,
    basis_values: &mut Array1<T>,
    basis_arrays: &Vec<(ArrayView1<T>, ArrayView1<T>)>,
    basis_y_cache: &mut ConvolveCache<T>,
    input_array: &ArrayView2<T>,
) {
    let y_start = (*y - kernel_radius) as usize;
    let x_start = (*x - kernel_radius) as usize;
    let x_stop = (*x + kernel_radius + 1) as usize;
    let kernel_size = (2 * kernel_radius + 1) as usize;
    let n_cols = input_array.strides()[0] as usize;

    // need to zero of the basis_value to start as it will be set from previous loop
    basis_values.fill(T::default());
    if (*y != *prev_y) || (*x - *prev_x) != 1 {
        // Since this is a new pixel jump, need to reset this cache var
        basis_y_cache.reset();
        unsafe {
            let basis_values_ptr = basis_values.as_mut_ptr();
            let input_base = input_array.as_ptr();
            let cache_base = basis_y_cache.array.as_mut_ptr();
            // loop over each basis function number
            for basis_idx in 0..basis_len {
                let basis_y_ptr = basis_arrays[basis_idx].0.as_ptr();
                let basis_x_ptr = basis_arrays[basis_idx].1.as_ptr();
                // intermediate container for x kernel multiplied by template summed for each x
                let basis_y_cache_ptr = cache_base.add(basis_idx * kernel_size);
                for kv in 0..kernel_size {
                    let input_ptr = input_base.add((y_start + kv) * n_cols + x_start);
                    let basis_y_val = *basis_y_ptr.add(kv);
                    for kv in 0..kernel_size {
                        *basis_y_cache_ptr.add(kv) += *input_ptr.add(kv) * basis_y_val;
                    }
                }

                let mut acc: T = T::default();
                for kv in 0..kernel_size {
                    acc += *basis_x_ptr.add(kv) * *basis_y_cache_ptr.add(kv);
                }
                *basis_values_ptr.add(basis_idx) = acc;
            }
        }
    } else {
        unsafe {
            let basis_values_ptr = basis_values.as_mut_ptr();

            let cache_offset = basis_y_cache.current_index;
            let first_valid_slot = (cache_offset + 1) % basis_y_cache.pixel_size;

            let row_offset = input_array.strides()[0] as usize;

            let cache_base = basis_y_cache.array.as_mut_ptr();

            // grab the cache unfriendly code once up front instead of each loop
            let hop_cache = basis_y_cache.y_hop_cache.as_mut_ptr();

            let mut input_ptr = input_array
                .as_ptr()
                .add(y_start * n_cols + (x_stop - 1));
            for hop in 0..kernel_size {
                *hop_cache.add(hop) = *input_ptr;
                input_ptr = input_ptr.add(row_offset);
            }

            for basis_idx in 0..basis_len {
                let basis_y_ptr = basis_arrays[basis_idx].0.as_ptr();
                let basis_x_ptr = basis_arrays[basis_idx].1.as_ptr();
                let basis_y_cache_ptr = cache_base.add(basis_idx * kernel_size);

                // fill in the new y cache column
                let mut acc: T = T::default();
                for kv in 0..kernel_size {
                    acc += *basis_y_ptr.add(kv) * *hop_cache.add(kv);
                }
                *basis_y_cache_ptr.add(cache_offset) = acc;

                // need to get cleaver to use the cache since the beginning is overwritten
                let basis_y_cache_ptr_offset = basis_y_cache_ptr.add(first_valid_slot);
                let mut basis_values_acc: T = T::default();
                for kv in 0..(kernel_size - (cache_offset + 1)) {
                    basis_values_acc += *basis_x_ptr.add(kv) * *basis_y_cache_ptr_offset.add(kv);
                }

                let basis_x_ptr_offset = basis_x_ptr.add(kernel_size - (cache_offset + 1));
                for kv in 0..(cache_offset + 1) {
                    basis_values_acc += *basis_x_ptr_offset.add(kv) * *basis_y_cache_ptr.add(kv);
                }
                *basis_values_ptr.add(basis_idx) = basis_values_acc;
            }
            basis_y_cache.increment();
        }
    }

    *prev_y = *y;
    *prev_x = *x;
}

#[derive(Serialize, Deserialize)]
#[pyclass(name = "DiffKernel")]
pub struct DiffKernelF64 {
    basis_arrays: Vec<(Array1<f64>, Array1<f64>)>,
    basis_radius: usize,
    spatial_order: u32,
    basis_coefficients: Array1<f64>,
}

#[derive(Serialize, Deserialize)]
#[pyclass(name = "DiffKernelF32")]
pub struct DiffKernelF32 {
    basis_arrays: Vec<(Array1<f32>, Array1<f32>)>,
    basis_radius: usize,
    spatial_order: u32,
    basis_coefficients: Array1<f32>,
}

// Macro to generate the pure-Rust impl block for both DiffKernelF64 and DiffKernelF32
macro_rules! impl_diff_kernel_methods {
    ($struct_name:ident, $T:ty) => {
        impl $struct_name {
            fn _draw_unweighted_basis(&self, index: usize) -> Array2<$T> {
                let basis_len = self.basis_radius * 2 + 1;
                let y_column = self.basis_arrays[index].0.to_shape((basis_len, 1)).unwrap();
                let x_row = self.basis_arrays[index].1.to_shape((1, basis_len)).unwrap();

                y_column.dot(&x_row)
            }

            fn _draw_weighted_basis(&self, index: usize, y_pos: $T, x_pos: $T) -> Array2<$T> {
                let spatial_order = self.spatial_order as usize + 1;
                let basis_len = self.basis_radius * 2 + 1;

                let spatial_size = (spatial_order * (spatial_order + 1) / 2) as usize;

                let mut spatial_terms = Array1::<$T>::zeros(spatial_size);
                let cheb_size = if self.spatial_order < 1 {
                    1
                } else {
                    self.spatial_order
                };

                let mut y_cheb = Array1::<$T>::zeros((cheb_size + 1) as usize);
                let mut x_cheb = Array1::<$T>::zeros((cheb_size + 1) as usize);
                self._populate_spatial_terms(
                    y_pos,
                    x_pos,
                    &mut spatial_terms.view_mut(),
                    &mut y_cheb.view_mut(),
                    &mut x_cheb.view_mut(),
                );

                // get indexes into the component parts
                let basis_index = index / spatial_size;
                let spatial_index = index % spatial_size;

                let weight = self.basis_coefficients[index];
                let spatial_weight = spatial_terms[spatial_index];

                let y_column = self.basis_arrays[basis_index]
                    .0
                    .to_shape((basis_len, 1))
                    .unwrap();
                let x_row = self.basis_arrays[basis_index]
                    .1
                    .to_shape((1, basis_len))
                    .unwrap();

                y_column.dot(&x_row) * weight * spatial_weight
            }

            fn _populate_spatial_terms(
                &self,
                y_pos: $T,
                x_pos: $T,
                spatial_terms: &mut ArrayViewMut1<$T>,
                y_cheb: &mut ArrayViewMut1<$T>,
                x_cheb: &mut ArrayViewMut1<$T>,
            ) {
                y_cheb[0] = <$T>::one();
                x_cheb[0] = <$T>::one();
                y_cheb[1] = y_pos;
                x_cheb[1] = x_pos;

                for i in 2..self.spatial_order + 1 {
                    let i = i as usize;
                    y_cheb[i] = <$T as NumCast>::from(2).unwrap() * y_pos * y_cheb[i - 1] - y_cheb[i - 2];
                    x_cheb[i] = <$T as NumCast>::from(2).unwrap() * x_pos * x_cheb[i - 1] - x_cheb[i - 2];
                }

                let mut index: usize = 0;
                for i in 0..(self.spatial_order as usize + 1) {
                    for j in 0..(self.spatial_order as usize - i + 1) {
                        spatial_terms[index] = x_cheb[i] * y_cheb[j];
                        index += 1;
                    }
                }
            }
        }
    };
}

impl_diff_kernel_methods!(DiffKernelF64, f64);
impl_diff_kernel_methods!(DiffKernelF32, f32);

// Macro to generate #[pymethods] blocks for both DiffKernelF64 and DiffKernelF32.
// Method names are not suffixed — the type name (DiffKernel vs DiffKernelF32)
// already disambiguates.
//
// Invocation:
//   impl_diff_kernel_pymethods!(DiffKernelF64, f64);
//   impl_diff_kernel_pymethods!(DiffKernelF32, f32);
macro_rules! impl_diff_kernel_pymethods {
    ($struct_name:ident, $T:ty) => {
        #[pymethods]
        impl $struct_name {
            fn get_basis_coefficients<'py>(
                &self,
                py: Python<'py>,
            ) -> Bound<'py, PyArray1<$T>> {
                self.basis_coefficients.to_owned().into_pyarray(py)
            }

            fn apply_kernel<'py>(
                &self,
                py: Python<'py>,
                input_image: PyReadonlyArray2<$T>,
            ) -> Bound<'py, PyArray2<$T>> {
                let input_array = input_image.as_array();
                let input_shape = input_array.dim();
                let output_shape = (
                    input_shape.0 - 2 * self.basis_radius,
                    input_shape.1 - 2 * self.basis_radius,
                );
                // setup chebichev polynomial stuff
                let x_mid = input_shape.1 / 2;
                let y_mid = input_shape.0 / 2;

                let cheb_size = if self.spatial_order < 1 {
                    1
                } else {
                    self.spatial_order
                };

                let mut y_cheb = Array1::<$T>::zeros((cheb_size + 1) as usize);
                y_cheb[0] = <$T>::one();
                let mut x_cheb = Array1::<$T>::zeros((cheb_size + 1) as usize);
                x_cheb[0] = <$T>::one();

                let mut output_array = Array2::<$T>::zeros(output_shape);

                let order = self.spatial_order as usize;
                let size = (order + 1) * (order + 2) / 2;
                let mut spatial_terms_filtered = Array1::<$T>::zeros(size);

                // setup variables used in convolution
                let basis_len = self.basis_arrays.len();
                let kernel_size = 2 * self.basis_radius + 1;
                let mut basis_values = Array1::<$T>::zeros(basis_len);
                let mut basis_y_cache = ConvolveCache::new(kernel_size, basis_len);
                let mut prev_y = i32::MAX;
                let mut prev_x = i32::MAX;

                let basis_views = self
                    .basis_arrays
                    .iter()
                    .map(|(y, x)| (y.view(), x.view()))
                    .collect::<Vec<(ArrayView1<$T>, ArrayView1<$T>)>>();

                // loop over the array, calculating all the outputs
                for y_pos in self.basis_radius..input_shape.0 - self.basis_radius {
                    for x_pos in self.basis_radius..input_shape.1 - self.basis_radius {
                        // calculate the cheb poly
                        let poly_y_pos =
                            (<$T as NumCast>::from(y_pos).unwrap()
                                - <$T as NumCast>::from(y_mid).unwrap())
                                / <$T as NumCast>::from(y_mid).unwrap();
                        let poly_x_pos =
                            (<$T as NumCast>::from(x_pos).unwrap()
                                - <$T as NumCast>::from(x_mid).unwrap())
                                / <$T as NumCast>::from(x_mid).unwrap();
                        self._populate_spatial_terms(
                            poly_y_pos,
                            poly_x_pos,
                            &mut (spatial_terms_filtered.view_mut()),
                            &mut y_cheb.view_mut(),
                            &mut x_cheb.view_mut(),
                        );
                        convolve_at_one_point(
                            &(x_pos as i32),
                            &(y_pos as i32),
                            &mut prev_x,
                            &mut prev_y,
                            self.basis_radius as i32,
                            basis_len,
                            &mut basis_values,
                            &basis_views,
                            &mut basis_y_cache,
                            &input_array,
                        );

                        let mut accu: $T = <$T>::default();

                        unsafe {
                            let mut coeff_ptr = self.basis_coefficients.as_ptr();
                            for basis_value in &basis_values {
                                for sp_term in &spatial_terms_filtered {
                                    accu += *basis_value * *sp_term * *coeff_ptr;
                                    coeff_ptr = coeff_ptr.add(1);
     
                                }
                            }
                        }

                        unsafe {
                            *output_array.uget_mut([
                                y_pos - self.basis_radius,
                                x_pos - self.basis_radius,
                            ]) = accu;
                        }
                    }
                }

                output_array.into_pyarray(py)
            }

            fn draw_unweighted_basis<'py>(
                &self,
                py: Python<'py>,
                index: usize,
            ) -> Bound<'py, PyArray2<$T>> {
                self._draw_unweighted_basis(index).into_pyarray(py)
            }

            fn draw_weighted_basis<'py>(
                &self,
                py: Python<'py>,
                index: usize,
                y_pos: $T,
                x_pos: $T,
            ) -> Bound<'py, PyArray2<$T>> {
                self._draw_weighted_basis(index, y_pos, x_pos).into_pyarray(py)
            }

            fn draw_kernel<'py>(
                &self,
                py: Python<'py>,
                y_pos: $T,
                x_pos: $T,
            ) -> Bound<'py, PyArray2<$T>> {
                let mut output =
                    Array2::<$T>::zeros((self.basis_radius * 2 + 1, self.basis_radius * 2 + 1));
                for index in 0..self.basis_coefficients.len() {
                    output += &self._draw_weighted_basis(index, y_pos, x_pos);
                }
                output.into_pyarray(py)
            }

            /// Serialize this kernel to a JSON string.
            fn to_json(&self) -> PyResult<String> {
                serde_json::to_string(self)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }

            /// Deserialize a kernel from a JSON string.
            #[staticmethod]
            fn from_json(json_str: &str) -> PyResult<Self> {
                serde_json::from_str(json_str)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }


            /// of note, the template_image dimensions are larger than the target image dimensions by
            /// the width of the basis function -1. I.e. if the target image is 4000x4000 and the
            /// basis function is len 21, template_image will have dimensions of 4020x4020 so there will
            /// always be pixels to convolve with
            #[staticmethod]
            fn solve_diff_kernel(
                x_values: PyReadonlyArray1<i32>,
                y_values: PyReadonlyArray1<i32>,
                // basis_functions: PyReadonlyArray3<$T>,
                basis_functions: Vec<(PyReadonlyArray1<$T>, PyReadonlyArray1<$T>)>,
                // basis_functions: Vec<PyReadonlyArray2<f32>>,
                spatial_order: u32,
                template_image: PyReadonlyArray2<$T>,
                target_image: PyReadonlyArray2<$T>,
            ) -> PyResult<$struct_name> {
                // get ndarray views
                let basis_arrays: Vec<(ArrayView1<$T>, ArrayView1<$T>)> = basis_functions
                    .iter()
                    .map(|(y_pyarr, x_pyarr)| (y_pyarr.as_array(), x_pyarr.as_array()))
                    .collect();
                let template_array = template_image.as_array();
                let target_array = target_image.as_array();
                let x_values_array = x_values.as_array();
                let y_values_array = y_values.as_array();

                // get needed shapes
                let kernel_radius = (&basis_arrays[0].0.dim() / 2) as i32;
                // let kernel_width = (basis_arrays.dim().1 / 2) as i32;
                let template_shape = template_array.dim();

                let x_mid = template_shape.1 / 2;
                let y_mid = template_shape.0 / 2;

                let cheb_size = if spatial_order < 1 { 1 } else { spatial_order };

                let mut y_cheb = Array1::<$T>::zeros((cheb_size + 1) as usize);
                y_cheb[0] = <$T>::one();
                let mut x_cheb = Array1::<$T>::zeros((cheb_size + 1) as usize);
                x_cheb[0] = <$T>::one();

                // filter out any x or y that is too close to bounds
                let xy_positions: Vec<(&i32, &i32)> = x_values_array
                    .iter()
                    .zip(y_values_array.iter())
                    .filter(|(x, y)| {
                        **x > kernel_radius as i32
                            && **x < (template_shape.1 as i32 - (kernel_radius + 2))
                            && **y > kernel_radius as i32
                            && **y < (template_shape.0 as i32 - (kernel_radius + 2))
                    })
                    .collect();

                let order = (spatial_order) as usize;
                let size = (order + 1) * (order + 2) / 2;
                let basis_len = basis_arrays.len();
                let num_parameters = size * basis_len;

                let mut basis_accumulator = Array2::<$T>::zeros((num_parameters, num_parameters));
                let mut basis_accumulator_vec =
                    Array1::<$T>::zeros(num_parameters * (num_parameters + 1) / 2);
                let mut target_accumulator = Array1::<$T>::zeros(num_parameters);

                let mut spatial_terms_filtered = Array1::<$T>::zeros(size);
                let x_len = basis_arrays[0].0.dim();
                let mut basis_values = Array1::<$T>::zeros(basis_len);
                let mut basis_y_cache = ConvolveCache::new(x_len, basis_len);

                let mut prev_y = i32::MAX;
                let mut prev_x = i32::MAX;

                let mut terms = Array1::<$T>::zeros(num_parameters);
                let terms_len = num_parameters;

                for (x, y) in &xy_positions {
                    convolve_at_one_point(
                        *x,
                        *y,
                        &mut prev_x,
                        &mut prev_y,
                        kernel_radius,
                         basis_len,
                        &mut basis_values,
                        &basis_arrays,
                        &mut basis_y_cache,
                        &template_array,
                    );
     
                    let poly_y_pos = (<$T as NumCast>::from(**y).unwrap()
                        - <$T as NumCast>::from(y_mid).unwrap())
                        / <$T as NumCast>::from(y_mid).unwrap();
                    let poly_x_pos = (<$T as NumCast>::from(**x).unwrap()
                        - <$T as NumCast>::from(x_mid).unwrap())
                        / <$T as NumCast>::from(x_mid).unwrap();

                    y_cheb[1] = poly_y_pos;
                    x_cheb[1] = poly_x_pos;
                    for i in 2..spatial_order + 1 {
                        let i = i as usize;
                        y_cheb[i] =
                            <$T as NumCast>::from(2).unwrap() * poly_y_pos * y_cheb[i - 1]
                                - y_cheb[i - 2];
                        x_cheb[i] =
                            <$T as NumCast>::from(2).unwrap() * poly_x_pos * x_cheb[i - 1]
                                - x_cheb[i - 2];
                    }

                    let mut index: usize = 0;
                    for i in 0..(order + 1) {
                        for j in 0..(order - i + 1) {
                            spatial_terms_filtered[index] = x_cheb[i] * y_cheb[j];
                            index += 1;
                        }
                    }


                    unsafe {
                        let bv_ptr = basis_values.as_ptr();
                        let sp_term_filt_ptr = spatial_terms_filtered.as_ptr();
                        let terms_ptr = terms.as_mut_ptr();

                        let mut terms_offset = 0usize;
                        for bas in 0..basis_len {
                            // let bv_val = (basis_values[bas]) as f32;
                            let bv_val = *bv_ptr.add(bas);
                            let terms_sub_ptr = terms_ptr.add(terms_offset);
                            for sp in 0..size {
                                *terms_sub_ptr.add(sp) = bv_val * *sp_term_filt_ptr.add(sp);
                            }
                            terms_offset += size;
                        }
                    }

                    unsafe {
                        let n = terms_len;
                        let basis_ptr_nn =
                            ptr::NonNull::new_unchecked(basis_accumulator_vec.as_mut_ptr());
                        let terms_ptr_nn = ptr::NonNull::new_unchecked(terms.as_mut_ptr());
                        let basis_ptr = basis_ptr_nn.as_ptr();
                        let terms_ptr = terms_ptr_nn.as_ptr();

                        let mut offset = 0usize;
                        for i in 0..n {
                            let row_len = n - i;
                            let term = *terms_ptr.add(i);
                            let local_basis_ptr = basis_ptr.add(offset);
                            let local_term = terms_ptr.add(i);
                            for k in 0..row_len {
                                *local_basis_ptr.add(k) += term * *local_term.add(k);
                            }
                            offset += row_len;
                        }
                    }

                    let target_value = target_array[[
                        (*y - kernel_radius) as usize,
                        (**x - kernel_radius) as usize,
                    ]];
                    target_accumulator += &(&terms * target_value);
                }

                let mut incrementor: usize = 0;
                unsafe {
                    let basis_accumulator_vec_ptr = basis_accumulator_vec.as_ptr();
                    for i in 0..basis_accumulator.dim().0 {
                        for j in i..basis_accumulator.dim().1 {
                            let basis_value = *basis_accumulator_vec_ptr.add(incrementor);
     
                            basis_accumulator[[i, j]] = basis_value;
                            basis_accumulator[[j, i]] = basis_value;
                            incrementor += 1;
                        }
                    }
                }

                let coefficients = basis_accumulator.solve(&target_accumulator).unwrap();

                Ok($struct_name {
                    basis_arrays: basis_arrays
                        .iter()
                        .map(|(y, x)| (y.to_owned(), x.to_owned()))
                        .collect(),
                    basis_radius: kernel_radius as usize,
                    spatial_order,
                    basis_coefficients: coefficients,
                })
            }
        }
    };
}

impl_diff_kernel_pymethods!(DiffKernelF64, f64);
impl_diff_kernel_pymethods!(DiffKernelF32, f32);
/// Fast convolution between an image and a kernel using FFT.
///
/// # Parameters
/// input_image : numpy.ndarray  (float32)
///     The input image to convolve.
/// input_kernel : numpy.ndarray  (float32)
///     The convolution kernel (float32).
///
/// # Returns
/// numpy.ndarray  (float32)
///     The convolved output with the same shape as ``input_image``.
#[pyfunction]
#[pyo3(name = "my_convolve")]
pub fn my_convolve_f32<'py>(
    py: Python<'py>,
    input_image: PyReadonlyArray2<f32>,
    input_kernel: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let mut processor = FftProcessor::<f32>::default();

    input_image
        .as_array()
        .conv_fft_with_processor(
            &input_kernel.as_array(),
            ConvMode::Same,
            PaddingMode::Zeros,
            &mut processor,
        )
        .unwrap()
        .into_pyarray(py)
}

/// Fast convolution between an image and a kernel using FFT (f64 variant).
///
/// # Parameters
/// input_image : numpy.ndarray  (float64)
///     The input image to convolve.
/// input_kernel : numpy.ndarray  (float64)
///     The convolution kernel (float64).
///
/// # Returns
/// numpy.ndarray  (float64)
///     The convolved output with the same shape as ``input_image``.
#[pyfunction]
pub fn my_convolve_f64<'py>(
    py: Python<'py>,
    input_image: PyReadonlyArray2<f64>,
    input_kernel: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let mut processor = FftProcessor::<f64>::default();

    input_image
        .as_array()
        .conv_fft_with_processor(
            &input_kernel.as_array(),
            ConvMode::Same,
            PaddingMode::Zeros,
            &mut processor,
        )
        .unwrap()
        .into_pyarray(py)
}

/// Computes the nth Hermite polynomial at x using recurrence.
fn hermite_polynomial<T: NdFloat + Default>(x: T, n: usize, amplitude: T) -> T {
    match n {
        0 => amplitude,
        1 => <T as NumCast>::from(2).unwrap() * x * amplitude,
        _ => {
            let two = <T as NumCast>::from(2).unwrap();
            let one = <T>::one();
            let mut h0 = amplitude;
            let mut h1 = two * x * amplitude;
            for i in 2..=n {
                let is = <T as NumCast>::from(i).unwrap();
                let h2 = (two / is).sqrt() * x * h1
                    - ((is - one) / is).sqrt() * h0;
                h0 = h1;
                h1 = h2;
            }
            h1
        }
    }
}

fn _generate_gauss_hermite_basis_inner<'py, T>(
    py: Python<'py>,
    half_width: f64,
    widths: Vec<f64>,
    orders: Vec<usize>,
) -> Vec<(Bound<'py, PyArray1<T>>, Bound<'py, PyArray1<T>>)>
where
    T: NdFloat + Default + NumCast + Element,
{
    let num_points = 2 * (half_width as isize) as usize + 1;
    let hw = T::from(half_width).unwrap();
    let x_values: Array1<T> =
        (0..num_points).map(|i| -hw + T::from(i as f64).unwrap()).collect();

    let mut basis_kernels = Vec::new();

    for (i, &sigma_f64) in widths.iter().enumerate() {
        let order = orders[i] + 1;
        let sigma = T::from(sigma_f64).unwrap();

        // Compute Gaussian envelope in f64 (where exp/sqrt are natively available),
        // then cast to T. This preserves exact numerical results across f32/f64.
        let norm_factor = sigma_f64 * (2.0_f64 * std::f64::consts::PI).sqrt();
        let x_values_f64: Array1<f64> =
            (0..num_points).map(|i| -half_width + (i as f64)).collect();
        let gauss_term_f64 = x_values_f64.mapv(|x| {
            ((-x.powi(2) / (2.0 * sigma_f64.powi(2))).exp()) / norm_factor
        });
        let gauss_term: Array1<T> = gauss_term_f64.mapv(|v| T::from(v).unwrap());

        for j in 0..order {
            // do the y part
            let y_hermite = Zip::from(x_values.view())
                .and(&gauss_term)
                .map_collect(|ind_val, g| hermite_polynomial::<T>(*ind_val / sigma, j, *g));
            let y_kernel = y_hermite;
            for k in 0..(order - j) {
                let x_hermite = Zip::from(x_values.view())
                    .and(&gauss_term)
                    .map_collect(|ind_val, g| hermite_polynomial::<T>(*ind_val / sigma, k, *g));
                let x_kernel = x_hermite;
                basis_kernels.push((y_kernel.to_owned(), x_kernel));
            }
        }
    }
    basis_kernels
        .into_iter()
        .map(|(y_k, x_k)| (y_k.into_pyarray(py), x_k.into_pyarray(py)))
        .collect()
}

/// Generate separable Gaussian-Hermite basis functions for image differencing.
///
/// Creates a list of separable 1D basis function pairs (y, x) by combining
/// a Gaussian envelope with Hermite polynomials. Each basis spans
/// ``2 * half_width + 1`` integer grid points from -half_width to +half_width.
///
/// The basis uses a triangular ordering: for each (width, order) pair
/// the y-direction generates Hermite polynomials of orders ``0..order``,
/// and for each such y-order *j*, the x-direction generates polynomials
/// of orders ``0..(order - j)``. This mirrors the lower-triangular
/// spatial polynomial scheme used by the difference kernel solver.
///
/// Parameters
/// ----------
/// half_width : float
///     Half-width of the 1D grid in pixels. Yields ``2 * half_width + 1``
///     evenly-spaced integer points spanning ``[-half_width, +half_width]``.
/// widths : list of float
///     Standard deviation (sigma) of the Gaussian envelope for each basis
///     family.
/// orders : list of int
///     Maximum Hermite order for each corresponding width (same length as
///     ``widths``). Each width-ordered pair generates a triangular set of
///     y- and x-Hermite polynomials as described above.
///
/// Returns
/// -------
/// list of tuple of numpy.ndarray
///     Each element is a ``(y_kernel, x_kernel)`` pair. Each kernel is a
///     1-D NumPy array of length ``2 * half_width + 1`` representing a
///     Gaussian-weighted Hermite polynomial.
///
/// See Also
/// --------
/// solve_diff_kernel
///     Fits optimal kernel coefficients against template/target images.
/// DiffKernel
///     Applies the fitted basis to difference images.
///
/// Examples
/// --------
/// >>> basis = generate_gauss_hermite_basis(2.0, [0.5, 1.0], [1, 2])
/// >>> len(basis)
/// 9
/// >>> y_k, x_k = basis[0]
/// >>> y_k.shape
/// (5,)
#[pyfunction]
#[pyo3(name = "generate_gauss_hermite_basis")]
pub fn generate_gauss_hermite_basis_f64<'py>(
    py: Python<'py>,
    half_width: f64,
    widths: Vec<f64>,
    orders: Vec<usize>,
) -> Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    _generate_gauss_hermite_basis_inner(py, half_width, widths, orders)
}

/// Generate separable Gaussian-Hermite basis functions for image differencing (f32 variant).
///
/// Creates a list of separable 1D basis function pairs (y, x) by combining
/// a Gaussian envelope with Hermite polynomials. Each basis spans
/// ``2 * half_width + 1`` integer grid points from -half_width to +half_width.
///
/// The basis uses a triangular ordering: for each (width, order) pair
/// the y-direction generates Hermite polynomials of orders ``0..order``,
/// and for each such y-order *j*, the x-direction generates polynomials
/// of orders ``0..(order - j)``. This mirrors the lower-triangular
/// spatial polynomial scheme used by the difference kernel solver.
///
/// Parameters
/// ----------
/// half_width : float
///     Half-width of the 1D grid in pixels. Yields ``2 * half_width + 1``
///     evenly-spaced integer points spanning ``[-half_width, +half_width]``.
/// widths : list of float
///     Standard deviation (sigma) of the Gaussian envelope for each basis
///     family.
/// orders : list of int
///     Maximum Hermite order for each corresponding width (same length as
///     ``widths``). Each width-ordered pair generates a triangular set of
///     y- and x-Hermite polynomials as described above.
///
/// Returns
/// -------
/// list of tuple of numpy.ndarray
///     Each element is a ``(y_kernel, x_kernel)`` pair. Each kernel is a
///     1-D NumPy array of length ``2 * half_width + 1`` representing a
///     Gaussian-weighted Hermite polynomial.
#[pyfunction]
pub fn generate_gauss_hermite_basis_f32<'py>(
    py: Python<'py>,
    half_width: f64,
    widths: Vec<f64>,
    orders: Vec<usize>,
) -> Vec<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    _generate_gauss_hermite_basis_inner(py, half_width, widths, orders)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_roundtrip_f64() {
        let basis5 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let basis6 = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let coeff6 = Array1::from_vec(vec![0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6]);

        let kernel = DiffKernelF64 {
            basis_arrays: vec![(basis5.clone(), basis6.clone()), (basis6, basis5)],
            basis_radius: 2,
            spatial_order: 1,
            basis_coefficients: coeff6,
        };

        let json = serde_json::to_string(&kernel).unwrap();
        let restored: DiffKernelF64 = serde_json::from_str(&json).unwrap();

        assert_eq!(kernel.basis_radius, restored.basis_radius);
        assert_eq!(kernel.spatial_order, restored.spatial_order);
        assert!(
            kernel.basis_coefficients
                .iter()
                .zip(restored.basis_coefficients.iter())
                .all(|(a, b)| (a - b).abs() < f64::EPSILON * 10.0)
        );
        assert_eq!(kernel.basis_arrays.len(), restored.basis_arrays.len());
        for ((oy, ox), (ry, rx)) in kernel
            .basis_arrays
            .iter()
            .zip(restored.basis_arrays.iter())
        {
            assert!(oy.iter().zip(ry.iter()).all(|(a, b)| (a - b).abs() < f64::EPSILON * 10.0));
            assert!(ox.iter().zip(rx.iter()).all(|(a, b)| (a - b).abs() < f64::EPSILON * 10.0));
        }
    }

    #[test]
    fn test_json_roundtrip_f32() {
        let basis5 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let basis6 = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let coeff6 = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]);

        let kernel = DiffKernelF32 {
            basis_arrays: vec![(basis5.clone(), basis6.clone()), (basis6, basis5)],
            basis_radius: 2,
            spatial_order: 1,
            basis_coefficients: coeff6,
        };

        let json = serde_json::to_string(&kernel).unwrap();
        let restored: DiffKernelF32 = serde_json::from_str(&json).unwrap();

        assert_eq!(kernel.basis_radius, restored.basis_radius);
        assert_eq!(kernel.spatial_order, restored.spatial_order);
        assert!(
            kernel.basis_coefficients
                .iter()
                .zip(restored.basis_coefficients.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0)
        );
        assert_eq!(kernel.basis_arrays.len(), restored.basis_arrays.len());
        for ((oy, ox), (ry, rx)) in kernel
            .basis_arrays
            .iter()
            .zip(restored.basis_arrays.iter())
        {
            assert!(oy.iter().zip(ry.iter()).all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0));
            assert!(ox.iter().zip(rx.iter()).all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0));
        }
    }

    #[test]
    fn test_from_json_invalid() {
        let res: Result<DiffKernelF64, _> = serde_json::from_str("not valid json");
        assert!(res.is_err());
    }
}
