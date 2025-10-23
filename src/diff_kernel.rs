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
use std::f64;
use std::ptr;

use log;
use ndarray::{prelude::*, Zip};
use ndarray::{Array1, Array2};
use ndarray_conv::{ConvFFTExt, ConvMode, FftProcessor, PaddingMode};
use ndarray_linalg::Inverse;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::types::IntoPyDict;
use pyo3::{prelude::*, BoundObject};

struct ConvolveCache {
    array: Array2<f64>,
    pixel_size: usize,
    basis_size: usize,
    current_index: usize,
}

impl ConvolveCache {
    fn new(pixel_size: usize, basis_size: usize) -> Self {
        ConvolveCache {
            array: Array2::<f64>::zeros((basis_size, pixel_size)),
            pixel_size,
            basis_size,
            current_index: 0,
        }
    }

    #[inline]
    fn increment(&mut self) {
        self.current_index = (self.current_index + 1) % self.pixel_size;
    }

    #[inline]
    fn reset(&mut self) {
        self.array.fill(0.0);
        self.current_index = 0;
    }
}

#[inline]
fn convolve_at_one_point(
    x: &i32,
    y: &i32,
    prev_x: &mut i32,
    prev_y: &mut i32,
    kernel_radius: i32,
    basis_len: usize,
    basis_values: &mut Array1<f64>,
    basis_arrays: &Vec<(ArrayView1<f64>, ArrayView1<f64>)>,
    basis_y_cache: &mut ConvolveCache,
    input_array: &Array2<f64>,
) {
    let y_start = *y - kernel_radius;

    let x_start = *x - kernel_radius;
    let x_stop = *x + kernel_radius + 1;

    // need to zero of the basis_value to start as it will be set from previous loop
    basis_values.fill(0.0);

    if (*y != *prev_y) || (*x - *prev_x) != 1 {
        // Since this is a new pixel jump, need to reset this cache var
        basis_y_cache.reset();
        unsafe {
            //basis_values are the result of the convolution with each basis function
            let basis_values_ptr = basis_values.as_mut_ptr();
            // loop over each basis function number
            for bas in 0..basis_len {
                let basis_y_ptr = basis_arrays[bas].0.as_ptr();
                let basis_x_ptr = basis_arrays[bas].1.as_ptr();
                // intermediate container for x kernel multiplied by template summed for each x
                let basis_y_cache_ptr = basis_y_cache.array.get_mut_ptr((bas, 0)).unwrap();
                for (x_v, x_ind) in (x_start..x_stop).into_iter().enumerate() {
                    let input_ptr = input_array
                        .get_ptr((y_start as usize, x_ind as usize))
                        .unwrap();
                    for y_v in 0..2 * kernel_radius + 1 {
                        *basis_y_cache_ptr.add(x_v as usize) +=
                            *input_ptr.add(y_v as usize * input_array.dim().1) as f64
                                * *basis_y_ptr.add(y_v as usize);
                    }
                    *basis_values_ptr.add(bas) +=
                        *basis_x_ptr.add(x_v as usize) * *basis_y_cache_ptr.add(x_v as usize);
                }
            }
        }
    } else {
        unsafe {
            let mut basis_values_ptr = basis_values.as_mut_ptr();

            let x_len = basis_arrays[0].0.dim();
            let cache_offset = basis_y_cache.current_index;
            let existing_column_offset = (cache_offset + 1) % basis_y_cache.pixel_size;

            let row_offset = input_array.dim().1;
            for bas in 0..basis_len {
                // calculate the last y colum
                let mut input_ptr = input_array
                    .get_ptr((y_start as usize, (x_stop - 1) as usize))
                    .unwrap();
                let mut basis_y_ptr = basis_arrays[bas].0.as_ptr();
                let basis_x_ptr = basis_arrays[bas].1.as_ptr();
                let basis_y_cache_ptr = basis_y_cache.array.get_mut_ptr((bas, 0)).unwrap();

                // set the current cache location to zero
                let tmp_ptr = basis_y_cache_ptr.add(cache_offset);
                *tmp_ptr = 0.0;
                let mut acc = 0.0;
                for _ in 0..x_len {
                    acc += *basis_y_ptr * *input_ptr;
                    // *tmp_ptr += *basis_y_ptr.add(y_v) * *input_ptr;
                    basis_y_ptr = basis_y_ptr.add(1);
                    input_ptr = input_ptr.add(row_offset);
                }
                *tmp_ptr = acc;

                // need to get cleaver to use the cache since the beginning is overwritten
                let basis_y_cache_ptr_offset = basis_y_cache_ptr.add(existing_column_offset);
                for x_v in 0..(x_len - (cache_offset + 1)) {
                    *basis_values_ptr += *basis_x_ptr.add(x_v) * *basis_y_cache_ptr_offset.add(x_v);
                }

                let basis_x_ptr_offset = basis_x_ptr.add(x_len - (cache_offset + 1));
                // let basis_values_ptr_offset = basis_values_ptr.add(x_len - existing_column_offset);
                for x_v in 0..(cache_offset + 1) {
                    *basis_values_ptr += *basis_x_ptr_offset.add(x_v) * *basis_y_cache_ptr.add(x_v);
                }
                basis_values_ptr = basis_values_ptr.add(1);
            }
            basis_y_cache.increment();
        }
    }

    *prev_y = *y;
    *prev_x = *x;
}

#[pyclass]
struct DiffKernel {
    basis_arrays: Vec<(Array1<f64>, Array1<f64>)>,
    basis_radius: usize,
    spatial_order: u32,
    basis_coeffients: Array1<f32>,
}

// Implement the pure rust methods that will not be used directly from python
impl DiffKernel {
    fn _draw_unweighted_basis(&self, index: usize) -> Array2<f64> {
        let basis_len = self.basis_radius * 2 + 1;
        let y_column = self.basis_arrays[index].0.to_shape((basis_len, 1)).unwrap();
        let x_row = self.basis_arrays[index].1.to_shape((1, basis_len)).unwrap();

        y_column.dot(&x_row)
    }

    fn _draw_weighted_basis(&self, index: usize, y_pos: f64, x_pos: f64) -> Array2<f64> {
        let spatial_order = self.spatial_order as usize + 1;
        let basis_len = self.basis_radius * 2 + 1;

        let spatial_size = (spatial_order * (spatial_order + 1) / 2) as usize;

        let mut spatial_terms = Array1::<f64>::zeros(spatial_size);
        let cheb_size = if self.spatial_order < 1 {
            1
        } else {
            self.spatial_order
        };

        let mut y_cheb = Array1::<f64>::zeros((cheb_size + 1) as usize);
        let mut x_cheb = Array1::<f64>::zeros((cheb_size + 1) as usize);
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

        let weight = self.basis_coeffients[index] as f64;
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
        y_pos: f64,
        x_pos: f64,
        spatial_terms: &mut ArrayViewMut1<f64>,
        y_cheb: &mut ArrayViewMut1<f64>,
        x_cheb: &mut ArrayViewMut1<f64>,
    ) {
        y_cheb[0] = 1.0;
        x_cheb[0] = 1.0;
        y_cheb[1] = y_pos;
        x_cheb[1] = x_pos;

        for i in 2..self.spatial_order + 1 {
            let i = i as usize;
            y_cheb[i] = 2.0 * y_pos * y_cheb[i - 1] - y_cheb[i - 2];
            x_cheb[i] = 2.0 * x_pos * x_cheb[i - 1] - x_cheb[i - 2];
        }

        let mut index: usize = 0;
        for i in 0..(self.spatial_order as usize + 1) {
            for j in 0..(self.spatial_order as usize - i + 1) {
                spatial_terms[index] = y_cheb[i] * x_cheb[j];
                index += 1;
            }
        }
    }
}

// Implement all the methods that will be called from python
#[pymethods]
impl DiffKernel {
    fn apply_kernel<'py>(
        &self,
        py: Python<'py>,
        input_image: PyReadonlyArray2<f32>,
    ) -> Bound<'py, PyArray2<f64>> {
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

        let mut y_cheb = Array1::<f64>::zeros((cheb_size + 1) as usize);
        y_cheb[0] = 1.0;
        let mut x_cheb = Array1::<f64>::zeros((cheb_size + 1) as usize);
        x_cheb[0] = 1.0;

        let mut output_array = Array2::<f64>::zeros(output_shape);

        // let mut processor = FftProcessor::default();
        // let convolved_basis: Vec<Array2<f64>> = (0..self.basis_arrays.len())
        //     .map(|index| {
        //         input_image
        //             .as_array()
        //             .mapv(|v| v as f64)
        //             .conv_fft_with_processor(
        //                 &self._draw_unweighted_basis(index),
        //                 ConvMode::Same,
        //                 PaddingMode::Zeros,
        //                 &mut processor,
        //             )
        //             .unwrap()
        //     })
        //     .collect();

        // println!(
        //     "The basis rad is {:?} the input shape is {input_shape:?}",
        //     self.basis_radius
        // );

        let order = self.spatial_order as usize;
        let size = (order + 1) * (order + 2) / 2;
        let mut spatial_terms_filtered = Array1::<f64>::zeros(size);

        // setup variables used in convolution
        let basis_len = self.basis_arrays.len();
        let kernel_size = 2 * self.basis_radius + 1;
        let mut basis_values = Array1::<f64>::zeros(basis_len);
        // let mut basis_y_cache = Array2::<f64>::zeros((basis_len, kernel_size));
        let mut basis_y_cache = ConvolveCache::new(kernel_size, basis_len);
        let mut prev_y = i32::MAX;
        let mut prev_x = i32::MAX;

        let basis_views = self
            .basis_arrays
            .iter()
            .map(|(y, x)| (y.view(), x.view()))
            .collect::<Vec<(ArrayView1<f64>, ArrayView1<f64>)>>();

        let input_f64 = input_array.mapv(|v| v as f64);

        // loop over the array, calculating all the outputs
        for y_pos in self.basis_radius..input_shape.0 - self.basis_radius {
            for x_pos in self.basis_radius..input_shape.1 - self.basis_radius {
                // calculate the cheb poly
                let poly_y_pos = ((y_pos as f32) - y_mid as f32) / y_mid as f32;
                let poly_x_pos = ((x_pos as f32) - x_mid as f32) / x_mid as f32;
                self._populate_spatial_terms(
                    poly_y_pos as f64,
                    poly_x_pos as f64,
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
                    &input_f64,
                );

                let mut forward_iterator = 0;
                let mut accu: f64 = 0.0;

                unsafe {
                    let mut coeff_ptr = self.basis_coeffients.as_ptr();
                    for basis_value in &basis_values {
                        for sp_term in &spatial_terms_filtered {
                            accu += *basis_value * *sp_term as f64 * *coeff_ptr as f64;
                            coeff_ptr = coeff_ptr.add(1);
                            //     * *self.basis_coeffients.uget(forward_iterator) as f64;
                            // forward_iterator += 1;
                        }
                    }
                }

                unsafe {
                    *output_array
                        .uget_mut([y_pos - self.basis_radius, x_pos - self.basis_radius]) =
                        accu as f64;
                }
            }
        }

        output_array.into_pyarray(py)
    }

    fn daw_unweighted_basis<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        self._draw_unweighted_basis(index).into_pyarray(py)
    }

    fn draw_weighted_basis<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        y_pos: f64,
        x_pos: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        self._draw_weighted_basis(index, y_pos, x_pos)
            .into_pyarray(py)
    }

    fn draw_kernel<'py>(
        &self,
        py: Python<'py>,
        y_pos: f64,
        x_pos: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let mut output =
            Array2::<f64>::zeros((self.basis_radius * 2 + 1, self.basis_radius * 2 + 1));
        for index in 0..self.basis_coeffients.len() {
            output += &self._draw_weighted_basis(index, y_pos, x_pos);
        }
        output.into_pyarray(py)
    }

    // #[staticmethod]
    // fn solve_diff_kernel_full(
    //     x_values: PyReadonlyArray1<i32>,
    //     y_values: PyReadonlyArray1<i32>,
    //     basis_functions: Vec<PyReadonlyArray2<f32>>,
    //     spatial_order: u32,
    //     template_image: PyReadonlyArray2<f32>,
    //     target_image: PyReadonlyArray2<f32>,
    // ) -> PyResult<DiffKernel> {
    //     // get ndarray views
    //     let basis_arrays: Vec<ArrayView2<f32>> =
    //         basis_functions.iter().map(|x| x.as_array()).collect();
    //     let template_array = template_image.as_array();
    //     let target_array = target_image.as_array();
    //     let x_values_array = x_values.as_array();
    //     let y_values_array = y_values.as_array();

    //     // get needed shapes
    //     let kernel_width = (&basis_arrays[0].dim().0 / 2) as i32;
    //     let template_shape = template_array.dim();

    //     let x_positions: Vec<&i32> = x_values_array
    //         .iter()
    //         .filter(|x| {
    //             **x > (kernel_width - 1) as i32
    //                 && **x < (template_shape.1 as i32 - (kernel_width) - 1)
    //         })
    //         .collect();
    //     let y_positions: Vec<&i32> = y_values_array
    //         .iter()
    //         .filter(|y| {
    //             **y > (kernel_width - 1) as i32
    //                 && **y < (template_shape.0 as i32 - (kernel_width) - 1)
    //         })
    //         .collect();

    //     // need to allocate accumulators
    //     let num_parameters =
    //         basis_arrays.len() * (spatial_order + 1) as usize * (spatial_order + 1) as usize;
    //     let mut basis_accumulator = Array2::<f32>::zeros((num_parameters, num_parameters));
    //     let mut target_accumulator = Array1::<f32>::zeros(num_parameters);

    //     let mut terms: Array1<f32> = Array1::zeros(num_parameters);
    //     let mut spatial_terms: Array1<f32> =
    //         Array1::zeros(((spatial_order + 1) * (spatial_order + 1)) as usize);

    //     // pre convolve all the arrays
    //     let mut processor = FftProcessor::default();
    //     let convolved_basis: Vec<Array2<f32>> = basis_arrays
    //         .iter()
    //         .map(|arr| {
    //             template_array
    //                 .conv_fft_with_processor(
    //                     arr,
    //                     ConvMode::Same,
    //                     PaddingMode::Zeros,
    //                     &mut processor,
    //                 )
    //                 .unwrap()
    //         })
    //         .collect();
    //     // need to allocate array
    //     let mut x_params = Array2::<f32>::zeros((x_positions.len(), (spatial_order + 1) as usize));

    //     for (i, x) in (&x_positions).iter().enumerate() {
    //         x_params.index_axis_mut(Axis(0), i).assign(
    //             &(0..spatial_order + 1)
    //                 .rev()
    //                 .map(|x_pow| x.pow(x_pow) as f32)
    //                 .collect::<Array1<_>>(),
    //         )
    //     }

    //     println!("looping positions");
    //     for y in y_positions {
    //         println!("doing {y:?}");
    //         let y_terms = (0..spatial_order + 1)
    //             .rev()
    //             .map(|y_pow| y.pow(y_pow) as f32)
    //             .collect::<Array1<f32>>();
    //         for (x_index, x) in (&x_positions).iter().enumerate() {
    //             // let mut terms_iter = terms.iter_mut();

    //             let mut forward_spatial = 0;
    //             for x_t in x_params.slice(s![x_index, ..]) {
    //                 for y_t in &y_terms {
    //                     unsafe {
    //                         *spatial_terms.uget_mut(forward_spatial) = x_t * y_t;
    //                     }
    //                     forward_spatial += 1;
    //                 }
    //             }

    //             let mut forward_terms = 0;
    //             for basis_func in &convolved_basis {
    //                 unsafe {
    //                     let basis_val = basis_func.uget([*y as usize, (**x) as usize]);
    //                     for st in &spatial_terms {
    //                         *terms.uget_mut(forward_terms) = (basis_val * st);
    //                         forward_terms += 1;
    //                     }
    //                 }
    //             }
    //             let terms_column = terms.to_shape((terms.len(), 1)).unwrap();
    //             let terms_row = terms.to_shape((1, terms.len())).unwrap();
    //             // basis_accumulator += &(terms_column.dot(&terms_row));
    //             general_mat_mul(1.0, &terms_column, &terms_row, 1.0, &mut basis_accumulator);

    //             unsafe {
    //                 let target_value = *target_array
    //                     .uget([(*y - kernel_width) as usize, (**x - kernel_width) as usize]);
    //                 target_accumulator += &(&terms * target_value);
    //             }
    //         }
    //     }
    //     let coefficients = basis_accumulator.inv().unwrap().dot(&target_accumulator);
    //     println!("The coefficients are {coefficients:?}");

    //     Ok(DiffKernel {
    //         basis_arrays: basis_arrays.iter().map(|x| x.to_owned()).collect(),
    //         basis_radius: kernel_width as usize,
    //         spatial_order,
    //         basis_coeffients: coefficients,
    //     })
    // }
    #[staticmethod]
    fn solve_diff_kernel(
        x_values: PyReadonlyArray1<i32>,
        y_values: PyReadonlyArray1<i32>,
        // basis_functions: PyReadonlyArray3<f64>,
        basis_functions: Vec<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)>,
        // basis_functions: Vec<PyReadonlyArray2<f32>>,
        spatial_order: u32,
        template_image: PyReadonlyArray2<f32>,
        target_image: PyReadonlyArray2<f32>,
    ) -> PyResult<DiffKernel> {
        // get ndarray views
        let basis_arrays = basis_functions
            .iter()
            .map(|(y_pyarr, x_pyarr)| (y_pyarr.as_array(), x_pyarr.as_array()))
            .collect::<Vec<_>>();
        let template_array = template_image.as_array();
        let target_array = target_image.as_array();
        let x_values_array = x_values.as_array();
        let y_values_array = y_values.as_array();

        // get needed shapes
        let kernel_width = (&basis_arrays[0].0.dim() / 2) as i32;
        // let kernel_width = (basis_arrays.dim().1 / 2) as i32;
        let template_shape = template_array.dim();

        let x_mid = template_shape.1 / 2;
        let y_mid = template_shape.0 / 2;

        let cheb_size = if spatial_order < 1 { 1 } else { spatial_order };

        let mut y_cheb = Array1::<f32>::zeros((cheb_size + 1) as usize);
        y_cheb[0] = 1.0;
        let mut x_cheb = Array1::<f32>::zeros((cheb_size + 1) as usize);
        x_cheb[0] = 1.0;

        // filter out any x or y that is too close to bounds
        let xy_positions: Vec<(&i32, &i32)> = x_values_array
            .iter()
            .zip(y_values_array.iter())
            .filter(|(x, y)| {
                **x > kernel_width as i32
                    && **x < (template_shape.1 as i32 - (kernel_width + 2))
                    && **y > kernel_width as i32
                    && **y < (template_shape.0 as i32 - (kernel_width + 2))
            })
            .collect();

        let order = (spatial_order) as usize;
        let size = (order + 1) * (order + 2) / 2;
        let basis_len = basis_arrays.len();
        let num_parameters = size * basis_len;

        let mut basis_accumulator = Array2::<f32>::zeros((num_parameters, num_parameters));
        let mut basis_accumulator_vec =
            Array1::<f32>::zeros(num_parameters * (num_parameters + 1) / 2);
        let mut target_accumulator = Array1::<f32>::zeros(num_parameters);

        let mut spatial_terms_filtered = Array1::<f64>::zeros(size);
        let x_len = basis_arrays[0].0.dim();
        // let x_len = basis_arrays.dim().1;
        let mut basis_values = Array1::<f64>::zeros(basis_len);
        // let mut basis_y_arrays = Array2::<f64>::zeros((basis_len, x_len));
        let mut basis_y_cache = ConvolveCache::new(x_len, basis_len);

        println!("looping positions");
        let mut prev_y = i32::MAX;
        let mut prev_x = i32::MAX;

        // let mut new_basis_arrays = Array3::<f64>::zeros((basis_dims, x_len, x_len));
        // for bas in 0..basis_len {
        //     let y_col = basis_arrays[bas].0.to_shape((x_len, 1)).unwrap();
        //     let x_row = basis_arrays[bas].1.to_shape((1, x_len)).unwrap();
        //     new_basis_arrays
        //         .slice_mut(s![bas, .., ..])
        //         .assign(&y_col.dot(&x_row));
        // }
        // let new_basis_arrays = basis_arrays;
        // let element_size = std::mem::size_of::<f64>();
        // let num_elements = size * basis_len;
        // let layout = Layout::from_size_align(element_size * num_elements, 16).unwrap();
        // let manual_terms_ptr = unsafe { alloc(layout) as *mut f64 };

        // let mut stack_array: [f64; 250] = [0.0; 250];

        // let mut terms = unsafe {
        //     ArrayViewMut1::<f64>::from_shape_ptr(
        //         (num_elements).strides(1),
        //         stack_array.as_mut_ptr(),
        //     )
        // };
        let mut terms = Array1::<f32>::zeros(num_parameters);

        let template_f64 = template_array.mapv(|v| v as f64);

        for (x, y) in &xy_positions {
            /*
            let y_start = *y - kernel_width;
            // let y_stop = *y + kernel_width + 1;

            let x_start = **x - kernel_width;
            let x_stop = **x + kernel_width + 1;

            // let template_view = template_array
            //     .slice(s![y_start..y_stop, x_start..x_stop])
            //     .mapv(|v| v as f64);

            // let basis_values = (&new_basis_arrays * &template_view)
            //     .axis_iter(Axis(0))
            //     .map(|x| x.sum())
            //     .collect::<Array1<f64>>();

            // need to zero of the basis_value to start as it will be set from previous loop
            basis_values.fill(0.0);

            if (**y != prev_y) || (**x - prev_x) != 1 {
                // Since this is a new pixel jump, need to reset this cache var
                basis_y_arrays.fill(0.0);
                unsafe {
                    //basis_values are the result of the convolution with each basis function
                    let basis_values_ptr = basis_values.as_mut_ptr();
                    // loop over each basis function number
                    for bas in 0..basis_len {
                        let basis_y_ptr = basis_arrays[bas].0.as_ptr();
                        let basis_x_ptr = basis_arrays[bas].1.as_ptr();
                        // intermediate container for x kernel multiplied by template summed for each x
                        let basis_y_array_ptr = basis_y_arrays.get_mut_ptr((bas, 0)).unwrap();
                        for (x_v, x_ind) in (x_start..x_stop).into_iter().enumerate() {
                            let template_ptr = template_array
                                .get_ptr((y_start as usize, x_ind as usize))
                                .unwrap();
                            for y_v in 0..2 * kernel_width + 1 {
                                *basis_y_array_ptr.add(x_v as usize) +=
                                    *template_ptr.add(y_v as usize * template_array.dim().1) as f64
                                        * *basis_y_ptr.add(y_v as usize);
                            }
                            *basis_values_ptr.add(bas) += *basis_x_ptr.add(x_v as usize)
                                * *basis_y_array_ptr.add(x_v as usize);
                        }
                    }
                }
            } else {
                unsafe {
                    let basis_values_ptr = basis_values.as_mut_ptr();

                    for bas in 0..basis_len {
                        let basis_y_ptr = basis_arrays[bas].0.as_ptr();
                        let basis_x_ptr = basis_arrays[bas].1.as_ptr();
                        let basis_y_array_ptr = basis_y_arrays.get_mut_ptr((bas, 0)).unwrap();

                        // need to shift down all the y values by 1
                        for i in 0..x_len - 1 {
                            *basis_y_array_ptr.add(i) = *basis_y_array_ptr.add(i + 1);
                        }

                        // calculate the last y colum
                        let template_ptr = template_array
                            .get_ptr((y_start as usize, (x_stop - 1) as usize))
                            .unwrap();
                        // set the last value to zero
                        *basis_y_array_ptr.add(x_len - 1) = 0.0;
                        for y_v in 0..x_len {
                            *basis_y_array_ptr.add(x_len - 1) += *basis_y_ptr.add(y_v)
                                * *template_ptr.add(y_v * template_array.dim().1) as f64;
                        }
                        for (x_v, x_ind) in (x_start..x_stop).into_iter().enumerate() {
                            *basis_values_ptr.add(bas) +=
                                *basis_x_ptr.add(x_v) * *basis_y_array_ptr.add(x_v);
                        }
                    }
                }
            }

            prev_y = **y;
            prev_x = **x;
            */
            convolve_at_one_point(
                *x,
                *y,
                &mut prev_x,
                &mut prev_y,
                kernel_width,
                basis_len,
                &mut basis_values,
                &basis_arrays,
                &mut basis_y_cache,
                &template_f64,
            );

            let basis_values_colum = basis_values.to_shape((basis_values.len(), 1)).unwrap();

            let poly_y_pos = ((**y as f32) - y_mid as f32) / y_mid as f32;
            let poly_x_pos = ((**x as f32) - x_mid as f32) / x_mid as f32;

            y_cheb[1] = poly_y_pos;
            x_cheb[1] = poly_x_pos;
            for i in 2..spatial_order + 1 {
                let i = i as usize;
                y_cheb[i] = 2.0 * poly_y_pos * y_cheb[i - 1] - y_cheb[i - 2];
                x_cheb[i] = 2.0 * poly_x_pos * x_cheb[i - 1] - x_cheb[i - 2];
            }

            let mut index: usize = 0;
            for i in 0..(order + 1) {
                for j in 0..(order - i + 1) {
                    spatial_terms_filtered[index] = (y_cheb[i] * x_cheb[j]) as f64;
                    index += 1;
                }
            }

            // let y_cheb_row = y_cheb.to_shape((1, y_cheb.len())).unwrap();
            // let x_cheb_column = x_cheb.to_shape((x_cheb.len(), 1)).unwrap();

            // let spatial_terms = x_cheb_column.dot(&y_cheb_row);
            // let spatial_term_filtered = spatial_terms
            //     .rows()
            //     .into_iter()
            //     .enumerate()
            //     .map(|(index, row)| row.slice(s![0..row.len() - index]).to_owned())
            //     .flat_map(|f| f.mapv(|v| v))
            //     .collect::<Array1<f32>>();

            // let spatial_terms_row = spatial_erms.to_shape((1, spatial_terms.len())).unwrap();
            let spatial_terms_row = spatial_terms_filtered
                .to_shape((1, spatial_terms_filtered.len()))
                .unwrap();

            let terms_m = &(basis_values_colum.dot(&spatial_terms_row));
            terms.assign(&terms_m.flatten().mapv(|v| v as f32));

            let terms_len = terms.len();
            unsafe {
                let mut basis_ptr = basis_accumulator_vec.as_mut_ptr();
                let terms_ptr = terms.as_ptr();
                // let mut incrementor: usize = 0;
                for i in 0..terms_len {
                    let term = *terms_ptr.add(i);
                    // incrementor += i;
                    for j in i..terms_len {
                        *basis_ptr += (term * *terms_ptr.add(j));
                        basis_ptr = basis_ptr.add(1);
                    }
                }
            }

            let target_value =
                target_array[[(*y - kernel_width) as usize, (**x - kernel_width) as usize]];
            target_accumulator += &(&terms * target_value);
        }
        let mut incrementor: usize = 0;
        for i in 0..basis_accumulator.dim().0 {
            for j in i..basis_accumulator.dim().1 {
                let basis_value = basis_accumulator_vec[incrementor];
                // basis_accumulator[[i, j]] = basis_accumulator[[j, i]];
                basis_accumulator[[i, j]] = basis_value;
                basis_accumulator[[j, i]] = basis_value;
                incrementor += 1;
            }
        }

        let coefficients = basis_accumulator.inv().unwrap().dot(&target_accumulator);
        // let coefficients = target_accumulator;
        println!("The coefficients are {coefficients:?}");

        // let outer: Vec<Array2<f32>> = basis_arrays
        //     .iter()
        //     .map(|(y_basis, x_basis)| {
        //         let y_colum = y_basis.to_shape((y_basis.len(), 1)).unwrap();
        //         let x_row = x_basis.to_shape((1, x_basis.len())).unwrap();
        //         y_colum.to_owned().dot(&x_row.to_owned()).mapv(|v| v as f32)
        //     })
        //     .collect();
        Ok(DiffKernel {
            basis_arrays: basis_arrays
                .iter()
                .map(|(y, x)| (y.to_owned(), x.to_owned()))
                .collect(),
            basis_radius: kernel_width as usize,
            spatial_order,
            basis_coeffients: coefficients.mapv(|v| v as f32),
        })
    }
}

#[pyfunction]
fn my_convolve<'py>(
    py: Python<'py>,
    input_image: PyReadonlyArray2<f32>,
    input_kernel: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let mut processor = FftProcessor::<f32>::default();

    for _ in 0..25 {
        input_image
            .as_array()
            .conv_fft_with_processor(
                &input_kernel.as_array(),
                ConvMode::Same,
                PaddingMode::Zeros,
                &mut processor,
            )
            .unwrap();
    }

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
fn hermite_polynomial(x: f64, n: usize, amplitude: f64) -> f64 {
    match n {
        0 => amplitude,
        1 => 2.0 * x * amplitude,
        _ => {
            let mut h0 = amplitude;
            let mut h1 = 2.0 * x * amplitude;
            for i in 2..=n {
                let h2 =
                    (2.0 / i as f64).sqrt() * x * h1 - ((i as f64 - 1.0) / i as f64).sqrt() * h0;
                h0 = h1;
                h1 = h2;
            }
            h1
        }
    }
}

#[pyfunction]
pub fn generate_gauss_hermite_basis<'py>(
    py: Python<'py>,
    half_width: f64,
    widths: Vec<f64>,
    orders: Vec<usize>,
) -> Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let num_points = 2 * (half_width as usize) + 1;
    let x_values: Array1<f64> = (0..num_points).map(|i| -half_width + (i as f64)).collect();

    let mut basis_kernels = Vec::new();

    for (i, &sigma) in widths.iter().enumerate() {
        let order = orders[i] + 1;

        let norm_factor = sigma * (2.0 * f64::consts::PI).sqrt();
        let gauss_term =
            x_values.mapv(|x| ((-x.powi(2) / (2.0 * sigma.powi(2))).exp()) / norm_factor);
        for j in 0..order {
            // do the y part
            let y_hermite = Zip::from(x_values.view())
                .and(&gauss_term)
                .map_collect(|ind_val, g| hermite_polynomial(*ind_val / sigma, j, *g));
            let y_kernel = y_hermite;
            for k in 0..(order - j) {
                let x_hermite = Zip::from(x_values.view())
                    .and(&gauss_term)
                    .map_collect(|ind_val, g| hermite_polynomial(*ind_val, k, *g));
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

pub fn create_diff_kernel_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let diff_module = PyModule::new(parent_module.py(), "difference_kernel")?;
    diff_module.add_class::<DiffKernel>()?;
    diff_module.add_function(wrap_pyfunction!(my_convolve, &diff_module)?)?;
    diff_module.add_function(wrap_pyfunction!(
        generate_gauss_hermite_basis,
        &diff_module
    )?)?;
    parent_module.add_submodule(&diff_module)
}
