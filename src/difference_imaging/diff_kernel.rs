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
use ndarray_linalg::Solve;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::types::IntoPyDict;
use pyo3::{prelude::*, BoundObject};

struct ConvolveCache {
    array: Array2<f64>,
    pixel_size: usize,
    basis_size: usize,
    current_index: usize,
    y_hop_cache: Array1<f64>,
}

impl ConvolveCache {
    fn new(pixel_size: usize, basis_size: usize) -> Self {
        ConvolveCache {
            array: Array2::<f64>::zeros((basis_size, pixel_size)),
            pixel_size,
            basis_size,
            current_index: 0,
            y_hop_cache: Array1::<f64>::zeros(pixel_size),
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
    input_array: &ArrayView2<f64>,
) {
    let y_start = (*y - kernel_radius) as usize;
    let x_start = (*x - kernel_radius) as usize;
    let x_stop = (*x + kernel_radius + 1) as usize;
    let kernel_size = (2 * kernel_radius + 1) as usize;
    let input_num_col = input_array.ncols();

    // need to zero of the basis_value to start as it will be set from previous loop
    basis_values.fill(0.0);
    // println!(
    //     "The template view is {:?}",
    //     input_array.slice(s![
    //         y_start..y_start + (2 * kernel_radius as usize + 1),
    //         x_start..x_stop
    //     ])
    // );

    if (*y != *prev_y) || (*x - *prev_x) != 1 {
        // Since this is a new pixel jump, need to reset this cache var
        basis_y_cache.reset();
        unsafe {
            //basis_values are the result of the convolution with each basis function
            let basis_values_ptr = basis_values.as_mut_ptr();
            let input_base = input_array.as_ptr();
            let cache_base = basis_y_cache.array.as_mut_ptr();
            // loop over each basis function number
            for bas in 0..basis_len {
                let basis_y_ptr = basis_arrays[bas].0.as_ptr();
                let basis_x_ptr = basis_arrays[bas].1.as_ptr();
                // intermediate container for x kernel multiplied by template summed for each x
                let basis_y_cache_ptr = cache_base.add(bas * kernel_size);
                for y_v in 0..kernel_size {
                    let input_ptr = input_base.add((y_start + y_v) * input_num_col + x_start);
                    let basis_y_val = *basis_y_ptr.add(y_v);
                    for x_v in 0..kernel_size {
                        *basis_y_cache_ptr.add(x_v) += *input_ptr.add(x_v) * basis_y_val;
                    }
                }

                let mut acc: f64 = 0.0;
                for x_v in 0..kernel_size {
                    acc += *basis_x_ptr.add(x_v) * *basis_y_cache_ptr.add(x_v);
                }
                *basis_values_ptr.add(bas) = acc;
            }
        }
    } else {
        unsafe {
            let basis_values_ptr = basis_values.as_mut_ptr();

            let x_len = basis_arrays[0].0.dim();
            let cache_offset = basis_y_cache.current_index;
            let existing_column_offset = (cache_offset + 1) % basis_y_cache.pixel_size;

            let row_offset = input_array.dim().1;

            let cache_base = basis_y_cache.array.as_mut_ptr();

            // grab the cache unfriendly code once up front instead of each loop
            let hop_cache = basis_y_cache.y_hop_cache.as_mut_ptr();

            let mut input_ptr = input_array
                .as_ptr()
                .add(y_start * input_num_col + (x_stop - 1));
            for hop in 0..x_len {
                *hop_cache.add(hop) = *input_ptr;
                input_ptr = input_ptr.add(row_offset);
            }

            // let hop_cache = basis_y_cache.y_hop_cache.as_mut_ptr();
            for bas in 0..basis_len {
                let basis_y_ptr = basis_arrays[bas].0.as_ptr();
                let basis_x_ptr = basis_arrays[bas].1.as_ptr();
                // let basis_y_cache_ptr = basis_y_cache.array.get_mut_ptr((bas, 0)).unwrap();
                let basis_y_cache_ptr = cache_base.add(bas * x_len);

                // fill in the new y cache column
                let mut acc: f64 = 0.0;
                for count in 0..x_len {
                    acc += *basis_y_ptr.add(count) * *hop_cache.add(count);
                }
                *basis_y_cache_ptr.add(cache_offset) = acc;

                // need to get cleaver to use the cache since the beginning is overwritten
                let basis_y_cache_ptr_offset = basis_y_cache_ptr.add(existing_column_offset);
                let mut basis_values_acc: f64 = 0.0;
                for x_v in 0..(x_len - (cache_offset + 1)) {
                    basis_values_acc += *basis_x_ptr.add(x_v) * *basis_y_cache_ptr_offset.add(x_v);
                }

                let basis_x_ptr_offset = basis_x_ptr.add(x_len - (cache_offset + 1));
                for x_v in 0..(cache_offset + 1) {
                    basis_values_acc += *basis_x_ptr_offset.add(x_v) * *basis_y_cache_ptr.add(x_v);
                }
                *basis_values_ptr.add(bas) = basis_values_acc as f64;
            }
            basis_y_cache.increment();
        }
    }

    *prev_y = *y;
    *prev_x = *x;
}

#[pyclass]
pub struct DiffKernel {
    basis_arrays: Vec<(Array1<f64>, Array1<f64>)>,
    basis_radius: usize,
    spatial_order: u32,
    basis_coeffients: Array1<f64>,
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

        let weight = self.basis_coeffients[index];
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
                spatial_terms[index] = x_cheb[i] * y_cheb[j];
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
        input_image: PyReadonlyArray2<f64>,
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

        // loop over the array, calculating all the outputs
        for y_pos in self.basis_radius..input_shape.0 - self.basis_radius {
            for x_pos in self.basis_radius..input_shape.1 - self.basis_radius {
                // calculate the cheb poly
                let poly_y_pos = ((y_pos as f64) - y_mid as f64) / y_mid as f64;
                let poly_x_pos = ((x_pos as f64) - x_mid as f64) / x_mid as f64;
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

                let mut accu: f64 = 0.0;

                unsafe {
                    let mut coeff_ptr = self.basis_coeffients.as_ptr();
                    for basis_value in &basis_values {
                        for sp_term in &spatial_terms_filtered {
                            accu += *basis_value * *sp_term * *coeff_ptr;
                            coeff_ptr = coeff_ptr.add(1);
                            //     * *self.basis_coeffients.uget(forward_iterator) as f64;
                            // forward_iterator += 1;
                        }
                    }
                }

                unsafe {
                    *output_array
                        .uget_mut([y_pos - self.basis_radius, x_pos - self.basis_radius]) = accu;
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

    /// of note, the template_image dimensions are larger than the target image dimensions by
    /// the width of the basis function -1. I.e. if the target image is 4000x4000 and the
    /// basis function is len 21, template_image will have dimensions of 4020x4020 so there will
    /// always be pixels to convolve with
    #[staticmethod]
    fn solve_diff_kernel(
        x_values: PyReadonlyArray1<i32>,
        y_values: PyReadonlyArray1<i32>,
        // basis_functions: PyReadonlyArray3<f64>,
        basis_functions: Vec<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)>,
        // basis_functions: Vec<PyReadonlyArray2<f32>>,
        spatial_order: u32,
        template_image: PyReadonlyArray2<f64>,
        target_image: PyReadonlyArray2<f64>,
    ) -> PyResult<DiffKernel> {
        // get ndarray views
        let basis_arrays: Vec<(ArrayView1<f64>, ArrayView1<f64>)> = basis_functions
            .iter()
            .map(|(y_pyarr, x_pyarr)| (y_pyarr.as_array(), x_pyarr.as_array()))
            .collect();
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

        let mut y_cheb = Array1::<f64>::zeros((cheb_size + 1) as usize);
        y_cheb[0] = 1.0;
        let mut x_cheb = Array1::<f64>::zeros((cheb_size + 1) as usize);
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

        let mut basis_accumulator = Array2::<f64>::zeros((num_parameters, num_parameters));
        let mut basis_accumulator_vec =
            Array1::<f64>::zeros(num_parameters * (num_parameters + 1) / 2);
        let mut target_accumulator = Array1::<f64>::zeros(num_parameters);

        let mut spatial_terms_filtered = Array1::<f64>::zeros(size);
        let x_len = basis_arrays[0].0.dim();
        let mut basis_values = Array1::<f64>::zeros(basis_len);
        let mut basis_y_cache = ConvolveCache::new(x_len, basis_len);

        let mut prev_y = i32::MAX;
        let mut prev_x = i32::MAX;

        let mut terms = Array1::<f64>::zeros(num_parameters);
        let terms_len = num_parameters;

        for (x, y) in &xy_positions {
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
                &template_array,
            );
            // println!("The values arrays are {:?}", &basis_values);
            // println!("\n");
            // println!("\n");

            let poly_y_pos = ((**y as f64) - y_mid as f64) / y_mid as f64;
            let poly_x_pos = ((**x as f64) - x_mid as f64) / x_mid as f64;

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
                    spatial_terms_filtered[index] = x_cheb[i] * y_cheb[j];
                    index += 1;
                }
            }
            // println!("Spatial terms filtered are {:?}", spatial_terms_filtered);
            // println!("\n");
            // println!("\n");

            unsafe {
                let bv_ptr = basis_values.as_ptr();
                let sp_term_filt_ptr = spatial_terms_filtered.as_ptr();
                let terms_ptr = terms.as_mut_ptr();

                let mut terms_base_count = 0usize;
                for bas in 0..basis_len {
                    // let bv_f32 = (basis_values[bas]) as f32;
                    let bv_f32 = *bv_ptr.add(bas);
                    let terms_sub_ptr = terms_ptr.add(terms_base_count);
                    for sp in 0..size {
                        *terms_sub_ptr.add(sp) = bv_f32 * *sp_term_filt_ptr.add(sp);
                    }
                    terms_base_count += size;
                }
            }

            unsafe {
                let n = terms_len;
                let basis_ptr_nn = ptr::NonNull::new_unchecked(basis_accumulator_vec.as_mut_ptr());
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

            let target_value =
                target_array[[(*y - kernel_width) as usize, (**x - kernel_width) as usize]] as f64;
            target_accumulator += &(&terms * target_value);
        }

        let mut incrementor: usize = 0;
        unsafe {
            let basis_accumulator_vec_ptr = basis_accumulator_vec.as_ptr();
            for i in 0..basis_accumulator.dim().0 {
                for j in i..basis_accumulator.dim().1 {
                    let basis_value = *basis_accumulator_vec_ptr.add(incrementor);
                    // basis_accumulator[[i, j]] = basis_accumulator[[j, i]];
                    basis_accumulator[[i, j]] = basis_value;
                    basis_accumulator[[j, i]] = basis_value;
                    incrementor += 1;
                }
            }
        }

        println!("the basis accumulator is {:?}", &basis_accumulator);
        println!("\n");
        println!("the target accumuator is {:?}", &target_accumulator);
        // let coefficients = basis_accumulator
        //     .mapv(|v| v as f64)
        //     .inv()
        //     .unwrap()
        //     .dot(&target_accumulator.mapv(|v| v as f64));
        // solve is maringally slower, will need to understand why, or switch to it if
        // there is numerical stability issues with inv
        let coefficients = basis_accumulator.solve(&target_accumulator).unwrap();
        // let coefficients = target_accumulator;

        println!("The coefficients are {coefficients:?}");

        Ok(DiffKernel {
            basis_arrays: basis_arrays
                .iter()
                .map(|(y, x)| (y.to_owned(), x.to_owned()))
                .collect(),
            basis_radius: kernel_width as usize,
            spatial_order,
            basis_coeffients: coefficients,
        })
    }
}

#[pyfunction]
pub fn my_convolve<'py>(
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
                let h2 = (2.0_f64 / i as f64).sqrt() * x * h1
                    - ((i as f64 - 1.0) / i as f64).sqrt() * h0;
                h0 = h1;
                h1 = h2;
            }
            h1
        }
    }
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
///     Standard deviation (σ) of the Gaussian envelope for each basis
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
                    .map_collect(|ind_val, g| hermite_polynomial(*ind_val / sigma, k, *g));
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
