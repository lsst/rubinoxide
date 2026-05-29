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
use ndarray_linalg::{Scalar, Solve};
use num_traits::NumCast;
use numpy::Element;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

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

            let mut input_ptr = input_array.as_ptr().add(y_start * n_cols + (x_stop - 1));
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

// ---------------------------------------------------------------------------
// Inner data structures — plain Rust, no #[pyclass]
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffKernelData<T: NdFloat + Default + NumCast> {
    pub basis_arrays: Vec<(Array1<T>, Array1<T>)>,
    pub basis_radius: usize,
    pub spatial_order: u32,
    pub basis_coefficients: Array1<T>,
    pub reduced_chisq: T,
}

// ---------------------------------------------------------------------------
// Tagged union enum — uses #[serde(tag = "dtype")] for internal tagging
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "dtype")]
pub enum DiffKernelInner {
    #[serde(rename = "DiffKernel")]
    F64(DiffKernelData<f64>),
    #[serde(rename = "DiffKernelF32")]
    F32(DiffKernelData<f32>),
}

// ---------------------------------------------------------------------------
// Pure-Rust methods on DiffKernelData (generic impl)
// ---------------------------------------------------------------------------

impl<T> DiffKernelData<T>
where
    T: NdFloat + Default + NumCast,
{
    fn _draw_unweighted_basis(&self, index: usize) -> Array2<T> {
        let basis_len = self.basis_radius * 2 + 1;
        let y_column = self.basis_arrays[index].0.to_shape((basis_len, 1)).unwrap();
        let x_row = self.basis_arrays[index].1.to_shape((1, basis_len)).unwrap();

        y_column.dot(&x_row)
    }

    fn _draw_weighted_basis(&self, index: usize, y_pos: T, x_pos: T) -> Array2<T> {
        let spatial_order = self.spatial_order as usize + 1;
        let basis_len = self.basis_radius * 2 + 1;

        let spatial_size = (spatial_order * (spatial_order + 1) / 2) as usize;

        let mut spatial_terms = Array1::<T>::zeros(spatial_size);
        let cheb_size = if self.spatial_order < 1 {
            1
        } else {
            self.spatial_order
        };

        let mut y_cheb = Array1::<T>::zeros((cheb_size + 1) as usize);
        let mut x_cheb = Array1::<T>::zeros((cheb_size + 1) as usize);
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
        y_pos: T,
        x_pos: T,
        spatial_terms: &mut ArrayViewMut1<T>,
        y_cheb: &mut ArrayViewMut1<T>,
        x_cheb: &mut ArrayViewMut1<T>,
    ) {
        y_cheb[0] = <T>::one();
        x_cheb[0] = <T>::one();
        y_cheb[1] = y_pos;
        x_cheb[1] = x_pos;

        for i in 2..self.spatial_order + 1 {
            let i = i as usize;
            y_cheb[i] = <T as NumCast>::from(2).unwrap() * y_pos * y_cheb[i - 1] - y_cheb[i - 2];
            x_cheb[i] = <T as NumCast>::from(2).unwrap() * x_pos * x_cheb[i - 1] - x_cheb[i - 2];
        }

        let mut index: usize = 0;
        for i in 0..(self.spatial_order as usize + 1) {
            for j in 0..(self.spatial_order as usize - i + 1) {
                spatial_terms[index] = x_cheb[i] * y_cheb[j];
                index += 1;
            }
        }
    }

    fn apply_kernel(&self, input_array: ArrayView2<T>) -> Array2<T> {
        let input_shape = input_array.dim();
        let output_shape = (
            input_shape.0 - 2 * self.basis_radius,
            input_shape.1 - 2 * self.basis_radius,
        );
        let x_mid = input_shape.1 / 2;
        let y_mid = input_shape.0 / 2;

        let cheb_size = if self.spatial_order < 1 {
            1
        } else {
            self.spatial_order
        };

        let mut y_cheb = Array1::<T>::zeros((cheb_size + 1) as usize);
        y_cheb[0] = <T>::one();
        let mut x_cheb = Array1::<T>::zeros((cheb_size + 1) as usize);
        x_cheb[0] = <T>::one();

        let mut output_array = Array2::<T>::zeros(output_shape);

        let order = self.spatial_order as usize;
        let size = (order + 1) * (order + 2) / 2;
        let mut spatial_terms_filtered = Array1::<T>::zeros(size);

        let basis_len = self.basis_arrays.len();
        let kernel_size = 2 * self.basis_radius + 1;
        let mut basis_values = Array1::<T>::zeros(basis_len);
        let mut basis_y_cache = ConvolveCache::new(kernel_size, basis_len);
        let mut prev_y = i32::MAX;
        let mut prev_x = i32::MAX;

        let basis_views = self
            .basis_arrays
            .iter()
            .map(|(y, x)| (y.view(), x.view()))
            .collect::<Vec<(ArrayView1<T>, ArrayView1<T>)>>();

        for y_pos in self.basis_radius..input_shape.0 - self.basis_radius {
            for x_pos in self.basis_radius..input_shape.1 - self.basis_radius {
                let poly_y_pos = (<T as NumCast>::from(y_pos).unwrap()
                    - <T as NumCast>::from(y_mid).unwrap())
                    / <T as NumCast>::from(y_mid).unwrap();
                let poly_x_pos = (<T as NumCast>::from(x_pos).unwrap()
                    - <T as NumCast>::from(x_mid).unwrap())
                    / <T as NumCast>::from(x_mid).unwrap();

                self._populate_spatial_terms(
                    poly_y_pos,
                    poly_x_pos,
                    &mut spatial_terms_filtered.view_mut(),
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

                let mut accu: T = T::default();

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
                    *output_array
                        .uget_mut([y_pos - self.basis_radius, x_pos - self.basis_radius]) = accu;
                }
            }
        }

        output_array
    }
}

// ---------------------------------------------------------------------------
// Generic solve_diff_kernel (pure Rust, no Python types)
// ---------------------------------------------------------------------------

fn solve_diff_kernel_impl<T>(
    x_values: ArrayView1<i32>,
    y_values: ArrayView1<i32>,
    basis_functions: Vec<(ArrayView1<T>, ArrayView1<T>)>,
    spatial_order: u32,
    template_image: ArrayView2<T>,
    target_image: ArrayView2<T>,
) -> DiffKernelData<T>
where
    T: NdFloat + Default + NumCast + Clone + Scalar,
    Array2<T>: ndarray_linalg::Solve<T>,
{
    let template_shape = template_image.dim();

    let kernel_radius = (&basis_functions[0].0.dim() / 2) as i32;

    let x_mid = template_shape.1 / 2;
    let y_mid = template_shape.0 / 2;

    let cheb_size = if spatial_order < 1 { 1 } else { spatial_order };

    let mut y_cheb = Array1::<T>::zeros((cheb_size + 1) as usize);
    y_cheb[0] = <T>::one();
    let mut x_cheb = Array1::<T>::zeros((cheb_size + 1) as usize);
    x_cheb[0] = <T>::one();

    let xy_positions: Vec<(&i32, &i32)> = x_values
        .iter()
        .zip(y_values.iter())
        .filter(|(x, y)| {
            **x > kernel_radius
                && **x < (template_shape.1 as i32 - (kernel_radius + 2))
                && **y > kernel_radius
                && **y < (template_shape.0 as i32 - (kernel_radius + 2))
        })
        .collect();

    let order = (spatial_order) as usize;
    let size = (order + 1) * (order + 2) / 2;
    let basis_len = basis_functions.len();
    let num_parameters = size * basis_len;

    let mut basis_accumulator = Array2::<T>::zeros((num_parameters, num_parameters));
    let mut basis_accumulator_vec = Array1::<T>::zeros(num_parameters * (num_parameters + 1) / 2);
    let mut target_accumulator = Array1::<T>::zeros(num_parameters);

    let mut spatial_terms_filtered = Array1::<T>::zeros(size);
    let x_len = basis_functions[0].0.dim();
    let mut basis_values = Array1::<T>::zeros(basis_len);
    let mut basis_y_cache = ConvolveCache::new(x_len, basis_len);

    let mut prev_y = i32::MAX;
    let mut prev_x = i32::MAX;

    let mut terms = Array1::<T>::zeros(num_parameters);
    let terms_len = num_parameters;

    let mut pixel_counter = 0usize;
    let mut sum_sq_response = <T>::default();

    for (x, y) in &xy_positions {
        pixel_counter += 1;
        convolve_at_one_point(
            x,
            y,
            &mut prev_x,
            &mut prev_y,
            kernel_radius,
            basis_len,
            &mut basis_values,
            &basis_functions
                .iter()
                .map(|(y, x)| (y.view(), x.view()))
                .collect(),
            &mut basis_y_cache,
            &template_image,
        );

        let poly_y_pos = (<T as NumCast>::from(**y).unwrap()
            - <T as NumCast>::from(y_mid).unwrap())
            / <T as NumCast>::from(y_mid).unwrap();
        let poly_x_pos = (<T as NumCast>::from(**x).unwrap()
            - <T as NumCast>::from(x_mid).unwrap())
            / <T as NumCast>::from(x_mid).unwrap();

        y_cheb[1] = poly_y_pos;
        x_cheb[1] = poly_x_pos;
        for i in 2..spatial_order + 1 {
            let i = i as usize;
            y_cheb[i] =
                <T as NumCast>::from(2).unwrap() * poly_y_pos * y_cheb[i - 1] - y_cheb[i - 2];
            x_cheb[i] =
                <T as NumCast>::from(2).unwrap() * poly_x_pos * x_cheb[i - 1] - x_cheb[i - 2];
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

        let target_value: T = target_image[[
            (*y - kernel_radius) as usize,
            (**x - kernel_radius) as usize,
        ]];
        target_accumulator += &(&terms * target_value);
        sum_sq_response += target_value * target_value;
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

    let fit_power = coefficients.dot(&target_accumulator);
    let chisq = sum_sq_response - fit_power;
    let reduced_chisq = chisq / T::from(pixel_counter - num_parameters).unwrap();

    DiffKernelData {
        basis_arrays: basis_functions
            .iter()
            .map(|(y, x)| (y.to_owned(), x.to_owned()))
            .collect(),
        basis_radius: kernel_radius as usize,
        spatial_order,
        basis_coefficients: coefficients,
        reduced_chisq,
    }
}

// ---------------------------------------------------------------------------
// The single Python-visible type
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[pyclass(name = "DiffKernel")]
pub struct DiffKernel {
    inner: DiffKernelInner,
}

#[pymethods]
impl DiffKernel {
    /// Return the basis coefficients array fitted during ``solve_diff_kernel``.
    ///
    /// Returns the 1-D array of learned kernel coefficients that weight each
    /// basis function expanded over the spatial Chebyshev polynomial grid.
    ///
    /// Returns
    /// -------
    /// ``numpy.ndarray`` of float
    ///     1-D array of basis coefficients. The dtype matches the kernel type
    ///     (``float64`` for f64 kernels, ``float32`` for f32 kernels).
    ///
    /// Examples
    /// --------
    /// >>> coeffs = kernel.get_basis_coefficients()
    /// >>> print(coeffs.shape)
    /// (6,)
    fn get_basis_coefficients<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.inner {
            DiffKernelInner::F64(k) => Ok(k.basis_coefficients.clone().into_pyarray(py).into()),
            DiffKernelInner::F32(k) => Ok(k.basis_coefficients.clone().into_pyarray(py).into()),
        }
    }

    /// Apply the learned difference kernel to an input image.
    ///
    /// Convolves ``input_image`` with the spatially-varying difference kernel
    /// using the basis functions and coefficients learned by
    /// ``solve_diff_kernel``. The kernel position is normalized relative
    /// to the image center using Chebyshev polynomials.
    ///
    /// Parameters
    /// ----------
    /// input_image : ``numpy.ndarray``
    ///     2-D input image to process. The dtype must match the kernel type
    ///     (``float64`` for f64 kernels, ``float32`` for f32 kernels).
    ///
    /// Returns
    /// -------
    /// ``numpy.ndarray``
    ///     2-D convolved output array. Its shape is shrunk by
    ///     ``2 * basis_radius`` in each dimension compared to ``input_image``,
    ///     i.e. ``(H - 2*R, W - 2*R)`` where ``R`` is the basis radius.
    ///
    /// Examples
    /// --------
    /// >>> from rubinoxide import DiffKernel
    /// >>> kernel = DiffKernel.solve_diff_kernel(...)
    /// >>> result = kernel.apply_kernel(science_image)
    /// >>> print(result.shape)
    /// (3960, 3960)
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If *input_image* dtype does not match the kernel's internal dtype.
    /// ValueError
    ///     If *input_image* is smaller than ``2 * basis_radius`` in either
    ///     dimension.
    fn apply_kernel<'py>(
        &self,
        py: Python<'py>,
        input_image: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        let input_dtype = input_image.getattr("dtype")?;

        match &self.inner {
            DiffKernelInner::F64(k) => {
                let dtype_name: String = input_dtype.getattr("name")?.extract()?;
                if dtype_name != "float64" {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Kernel is float64 but input image is not float64.",
                    ));
                }
                let arr: PyReadonlyArray2<f64> = input_image.extract()?;
                let result = k.apply_kernel(arr.as_array());
                Ok(result.into_pyarray(py).into())
            }
            DiffKernelInner::F32(k) => {
                let dtype_name: String = input_dtype.getattr("name")?.extract()?;
                if dtype_name != "float32" {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Kernel is float32 but input image is not float32.",
                    ));
                }
                let arr: PyReadonlyArray2<f32> = input_image.extract()?;
                let result = k.apply_kernel(arr.as_array());
                Ok(result.into_pyarray(py).into())
            }
        }
    }

    /// Return a single basis function (without spatial weighting or learned coefficients).
    ///
    /// Returns the 2-D outer product of the y- and x-components of the
    /// basis function at the given index, with no multiplication by the
    /// learned coefficient or any spatial Chebyshev weighting.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Index of the basis function to draw (0-based).
    ///
    /// Returns
    /// -------
    /// ``numpy.ndarray``
    ///     2-D array of shape ``(2 * basis_radius + 1, 2 * basis_radius + 1)``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If *index* is out of range for the stored basis functions.
    fn draw_unweighted_basis<'py>(&self, py: Python<'py>, index: usize) -> PyResult<PyObject> {
        match &self.inner {
            DiffKernelInner::F64(k) => Ok(k._draw_unweighted_basis(index).into_pyarray(py).into()),
            DiffKernelInner::F32(k) => Ok(k._draw_unweighted_basis(index).into_pyarray(py).into()),
        }
    }

    /// Return a single basis function weighted by spatial position and learned coefficients.
    ///
    /// Returns the 2-D outer product of the y- and x-components of the
    /// basis function at the given index, scaled by the corresponding
    /// learned coefficient and the spatial Chebyshev polynomial term
    /// evaluated at ``(y_pos, x_pos)``.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Index of the weighted basis function to draw (0-based).
    /// y_pos : float
    ///     Y position (normalized pixel coordinate) for spatial weighting.
    ///     Should be in ``[-1, +1]`` relative to image center for correct
    ///     Chebyshev polynomial evaluation.
    /// x_pos : float
    ///     X position (normalized pixel coordinate) for spatial weighting.
    ///     Should be in ``[-1, +1]`` relative to image center for correct
    ///     Chebyshev polynomial evaluation.
    ///
    /// Returns
    /// -------
    /// ``numpy.ndarray``
    ///     2-D weighted basis function array of shape
    ///     ``(2 * basis_radius + 1, 2 * basis_radius + 1)``.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If *index* is out of range for the stored basis functions.
    fn draw_weighted_basis<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        y_pos: f64,
        x_pos: f64,
    ) -> PyResult<PyObject> {
        match &self.inner {
            DiffKernelInner::F64(k) => Ok(k
                ._draw_weighted_basis(index, y_pos, x_pos)
                .into_pyarray(py)
                .into()),
            DiffKernelInner::F32(k) => Ok(k
                ._draw_weighted_basis(index, y_pos as f32, x_pos as f32)
                .into_pyarray(py)
                .into()),
        }
    }

    /// Return the composite difference kernel at a given spatial position.
    ///
    /// Sums all weighted basis functions evaluated at ``(y_pos, x_pos)`` to
    /// produce the full spatially-varying difference kernel. Each basis
    /// function is scaled by its learned coefficient and the corresponding
    /// Chebyshev spatial polynomial evaluated at the given position.
    ///
    /// Parameters
    /// ----------
    /// y_pos : float
    ///     Y position (normalized pixel coordinate) for spatial weighting.
    ///     Should be in ``[-1, +1]`` relative to image center for correct
    ///     Chebyshev polynomial evaluation.
    /// x_pos : float
    ///     X position (normalized pixel coordinate) for spatial weighting.
    ///     Should be in ``[-1, +1]`` relative to image center for correct
    ///     Chebyshev polynomial evaluation.
    ///
    /// Returns
    /// -------
    /// ``numpy.ndarray``
    ///     2-D composite kernel of shape ``(2 * basis_radius + 1, 2 * basis_radius + 1)``.
    ///
    /// See Also
    /// --------
    /// draw_weighted_basis
    ///     Return an individual weighted basis function instead of the sum.
    fn draw_kernel<'py>(&self, py: Python<'py>, y_pos: f64, x_pos: f64) -> PyResult<PyObject> {
        match &self.inner {
            DiffKernelInner::F64(k) => {
                let mut output =
                    Array2::<f64>::zeros((k.basis_radius * 2 + 1, k.basis_radius * 2 + 1));
                for index in 0..k.basis_coefficients.len() {
                    output += &k._draw_weighted_basis(index, y_pos, x_pos);
                }
                Ok(output.into_pyarray(py).into())
            }
            DiffKernelInner::F32(k) => {
                let y_f32 = y_pos as f32;
                let x_f32 = x_pos as f32;
                let mut output =
                    Array2::<f32>::zeros((k.basis_radius * 2 + 1, k.basis_radius * 2 + 1));
                for index in 0..k.basis_coefficients.len() {
                    output += &k._draw_weighted_basis(index, y_f32, x_f32);
                }
                Ok(output.into_pyarray(py).into())
            }
        }
    }

    /// Serialize this kernel to a JSON string with a type discriminator.
    ///
    /// Embeds a ``"dtype"`` field in the JSON output to identify the kernel
    /// variant (``DiffKernel`` for float64 or ``DiffKernelF32`` for float32),
    /// enabling correct type dispatch during deserialization.
    ///
    /// Returns
    /// -------
    /// str
    ///     JSON string representation of the kernel including the dtype tag.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If serialization fails.
    fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Deserialize a kernel from a JSON string.
    ///
    /// Parses the JSON string and reconstructs the kernel instance. The JSON
    /// must contain a ``"dtype"`` field matching the kernel type.
    ///
    /// Parameters
    /// ----------
    /// json_str : str
    ///     JSON string containing serialized kernel data.
    ///
    /// Returns
    /// -------
    /// ``DiffKernel``
    ///     Reconstructed kernel of the matching type.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the JSON is invalid, deserialization fails, or the ``dtype`` is
    ///     unrecognized.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let inner: DiffKernelInner = serde_json::from_str(json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to deserialize kernel: {}", e))
        })?;
        Ok(DiffKernel { inner })
    }

    /// Provide Pydantic v2 integration — builds a
    /// `pydantic_core.core_schema.CoreSchema` so that this type
    /// can be used as a field type in `pydantic.BaseModel` classes.
    ///
    /// Called by Pydantic v2 with the signature:
    ///   `__get_pydantic_core_schema__(cls, handler, source_type)`
    /// where `handler` is the Pydantic handler (unused here — we build
    /// the schema directly via pydantic_core.core_schema).
    #[classmethod]
    fn __get_pydantic_core_schema__(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        _handler: &Bound<'_, PyAny>,
        _source_type: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let cs = py.import("pydantic_core.core_schema")?;
        let json_mod = py.import("json")?;

        let code = r#"
def _build_schema(cls, cs, json_mod, type_err):
    def _validate(v):
        if isinstance(v, cls):
            return v
        if isinstance(v, dict):
            return cls.from_json(json_mod.dumps(v))
        if isinstance(v, str):
            return cls.from_json(v)
        raise type_err(
            f"Cannot convert {type(v).__name__} to {cls.__name__}. "
            f"Expected a {cls.__name__} instance, dict, or JSON string."
        )

    def _serialize(v):
        if isinstance(v, cls):
            return json_mod.loads(v.json())
        return v

    return cs.json_or_python_schema(
        json_schema=cs.no_info_plain_validator_function(_validate),
        python_schema=cs.union_schema([
            cs.is_instance_schema(cls),
            cs.no_info_plain_validator_function(_validate),
        ]),
        serialization=cs.plain_serializer_function_ser_schema(
            _serialize,
            return_schema=cs.dict_schema(),
            when_used="always",
        ),
    )
"#;

        let builtins = py.import("builtins")?;
        let locals_dict = pyo3::types::PyDict::new(py);
        let cls_ref = _cls;
        locals_dict.set_item("cls", cls_ref)?;
        locals_dict.set_item("cs", &cs)?;
        locals_dict.set_item("json_mod", &json_mod)?;
        locals_dict.set_item("type_err", py.get_type::<pyo3::exceptions::PyTypeError>())?;

        builtins.call_method1("exec", (code, &locals_dict))?;

        let builder = locals_dict.get_item("_build_schema")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to define _build_schema in exec")
        })?;

        let schema = builder.call1((
            cls_ref,
            &cs,
            &json_mod,
            py.get_type::<pyo3::exceptions::PyTypeError>(),
        ))?;

        Ok(schema.into())
    }

    /// Provide JSON Schema for OpenAPI / Pydantic documentation.
    ///
    /// Called by Pydantic v2 with:
    ///   `__get_pydantic_json_schema__(cls, core_schema, handler)`
    #[classmethod]
    fn __get_pydantic_json_schema__(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        _core_schema: &Bound<'_, PyAny>,
        _handler: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let ndarray_obj_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "v": {"type": "integer"},
                "dim": {"type": "array", "items": {"type": "integer"}},
                "data": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["v", "dim", "data"]
        });

        let json_schema_value = serde_json::json!({
            "type": "object",
            "title": "DiffKernel",
            "description": "Serialized DiffKernel kernel (dict format)",
            "properties": {
                "dtype": {"type": "string"},
                "basis_arrays": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": ndarray_obj_schema,
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "basis_radius": {"type": "integer", "minimum": 0},
                "spatial_order": {"type": "integer", "minimum": 0},
                "basis_coefficients": ndarray_obj_schema.clone(),
                "reduced_chisq": {"type": "number"}
            },
            "required": ["dtype", "basis_arrays", "basis_radius", "spatial_order", "basis_coefficients"]
        });

        let json_str = serde_json::to_string(&json_schema_value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to serialize JSON schema: {}",
                e
            ))
        })?;

        let json_mod = py.import("json")?;
        let py_dict = json_mod.call_method1("loads", (&json_str,))?;
        Ok(py_dict.into())
    }

    /// Validate and construct a kernel instance from data.
    ///
    /// Pydantic-style classmethod that accepts a kernel instance (pass-through),
    /// a dict, or a JSON string. Mirrors ``pydantic.BaseModel.model_validate``.
    ///
    /// Parameters
    /// ----------
    /// data : one of ``DiffKernel``, ``dict``, ``str``
    ///     The data to validate. If already a DiffKernel instance, it is
    ///     extracted into a new instance. If a dict, it is serialized to
    ///     JSON then deserialized. If a string, it is treated as a JSON
    ///     string.
    ///
    /// Returns
    /// -------
    /// ``DiffKernel``
    ///     The constructed or extracted kernel object.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If *data* type is not supported (not a kernel instance, dict, or
    ///     string).
    #[classmethod]
    fn model_validate(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if data.is_instance_of::<DiffKernel>() {
            return data.extract::<Self>();
        }
        if data.is_instance_of::<pyo3::types::PyString>() {
            let json_str: &str = data.extract()?;
            return Self::from_json(json_str);
        }
        if data.is_instance_of::<pyo3::types::PyDict>() {
            let json_mod = py.import("json")?;
            let json_str_obj = json_mod.call_method1("dumps", (data,))?;
            let json_str: String = json_str_obj.extract()?;
            return Self::from_json(&json_str);
        }
        let type_name = data.get_type().name()?;
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "model_validate expects a DiffKernel instance, dict, or JSON string. Got {}.",
            type_name
        )))
    }

    /// Serialize kernel to a Python dict.
    ///
    /// Pydantic-style method that returns a dictionary representation of the
    /// kernel, including all fields and the dtype tag. Mirrors
    /// ``pydantic.BaseModel.model_dump``.
    ///
    /// Returns
    /// -------
    /// ``dict``
    ///     Dictionary representation of the kernel with all fields including
    ///     the dtype tag.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If serialization fails.
    fn model_dump(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = self.json()?;
        let json_mod = py.import("json")?;
        let py_dict = json_mod.call_method1("loads", (&json_str,))?;
        Ok(py_dict.into())
    }

    /// Fit difference kernel coefficients by solving a linear system.
    ///
    /// Given point source positions and template/target image pairs, solves for
    /// the set of basis coefficients that best model the difference between the
    /// template and target images. Each basis function is expanded over spatial
    /// Chebyshev polynomial terms up to the specified order, allowing the kernel
    /// to vary across the image.
    ///
    /// The template image dimensions must be larger than the target image
    /// dimensions by ``basis_function_width - 1``. For example, if the target
    /// image is 4000x4000 and the basis function has length 21, the template
    /// image should have dimensions 4020x4020 so that there are always pixels
    /// available to convolve at every valid position.
    ///
    /// Parameters
    /// ----------
    /// x_values : ``numpy.ndarray`` of int
    ///     X coordinates of point sources in pixel coordinates.
    /// y_values : ``numpy.ndarray`` of int
    ///     Y coordinates of point sources in pixel coordinates.
    /// basis_functions : list of tuple of ``numpy.ndarray``, ``numpy.ndarray``
    ///     Separable (y, x) basis function pairs. Each tuple contains two 1-D
    ///     arrays representing the y-axis and x-axis components of a Gaussian
    ///     Hermite basis function.
    /// spatial_order : int
    ///     Maximum order of the Chebyshev spatial polynomial model. Controls
    ///     the spatial variability of the difference kernel.
    /// template_image : ``numpy.ndarray`` of float
    ///     Reference/template image. Must be at least ``basis_function_width - 1``
    ///     pixels larger in each dimension than ``target_image``. The dtype must
    ///     be float32 or float64.
    /// target_image : ``numpy.ndarray`` of float
    ///     Science/target image to be difference-imaged against the template.
    ///     The dtype must match ``template_image`` (both float32 or both float64).
    ///
    /// Returns
    /// -------
    /// ``DiffKernel``
    ///     The fitted kernel object containing the learned basis coefficients,
    ///     basis functions, spatial order, and kernel radius.
    ///
    /// See Also
    /// --------
    /// generate_gauss_hermite_basis
    ///     Precompute Gaussian-Hermite basis functions for use here.
    /// DiffKernel.apply_kernel
    ///     Apply the fitted kernel to difference an image.
    ///
    /// Examples
    /// --------
    /// >>> from rubinoxide import DiffKernel, generate_gauss_hermite_basis
    /// >>> basis = generate_gauss_hermite_basis(10, [0.5, 1.0, 2.0], [12, 12, 12])
    /// >>> kernel = DiffKernel.solve_diff_kernel(
    /// ...     psf_x, psf_y, basis, 3, template, target
    /// ... )
    /// >>> coeffs = kernel.get_basis_coefficients()
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If template_image or target_image are not float32 or float64.
    /// ValueError
    ///     If the linear system is singular (e.g., insufficient or
    ///     degenerate point sources). Also raised if all point sources
    ///     are filtered out due to being too close to boundaries.
    #[staticmethod]
    fn solve_diff_kernel(
        _py: Python<'_>,
        x_values: &Bound<'_, PyAny>,
        y_values: &Bound<'_, PyAny>,
        basis_functions: &Bound<'_, PyAny>,
        spatial_order: u32,
        template_image: &Bound<'_, PyAny>,
        target_image: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let template_dtype = template_image.getattr("dtype")?;
        let template_dtype_name: String = template_dtype.getattr("name")?.extract()?;
        let target_dtype = target_image.getattr("dtype")?;
        let target_dtype_name: String = target_dtype.getattr("name")?.extract()?;
        if template_dtype_name != target_dtype_name {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "template_image (dtype={}) and target_image (dtype={}) must have the same dtype.",
                template_dtype_name, target_dtype_name
            )));
        }

        if template_dtype_name == "float64" {
            let xv: PyReadonlyArray1<i32> = x_values.extract()?;
            let yv: PyReadonlyArray1<i32> = y_values.extract()?;
            let bf: Vec<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)> =
                basis_functions.extract()?;
            let ti: PyReadonlyArray2<f64> = template_image.extract()?;
            let tr: PyReadonlyArray2<f64> = target_image.extract()?;
            let basis_views: Vec<(ArrayView1<f64>, ArrayView1<f64>)> = bf
                .iter()
                .map(|(y, x)| (y.as_array(), x.as_array()))
                .collect();
            let inner = solve_diff_kernel_impl(
                xv.as_array(),
                yv.as_array(),
                basis_views,
                spatial_order,
                ti.as_array(),
                tr.as_array(),
            );
            Ok(DiffKernel {
                inner: DiffKernelInner::F64(inner),
            })
        } else if template_dtype_name == "float32" {
            let xv: PyReadonlyArray1<i32> = x_values.extract()?;
            let yv: PyReadonlyArray1<i32> = y_values.extract()?;
            let bf: Vec<(PyReadonlyArray1<f32>, PyReadonlyArray1<f32>)> =
                basis_functions.extract()?;
            let ti: PyReadonlyArray2<f32> = template_image.extract()?;
            let tr: PyReadonlyArray2<f32> = target_image.extract()?;
            let basis_views: Vec<(ArrayView1<f32>, ArrayView1<f32>)> = bf
                .iter()
                .map(|(y, x)| (y.as_array(), x.as_array()))
                .collect();
            let inner = solve_diff_kernel_impl(
                xv.as_array(),
                yv.as_array(),
                basis_views,
                spatial_order,
                ti.as_array(),
                tr.as_array(),
            );
            Ok(DiffKernel {
                inner: DiffKernelInner::F32(inner),
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "template_image must be float32 or float64.",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Basis generation helpers (unchanged)
// ---------------------------------------------------------------------------

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
                let h2 = (two / is).sqrt() * x * h1 - ((is - one) / is).sqrt() * h0;
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
    let x_values: Array1<T> = (0..num_points)
        .map(|i| -hw + T::from(i as f64).unwrap())
        .collect();

    let mut basis_kernels = Vec::new();

    for (i, &sigma_f64) in widths.iter().enumerate() {
        let order = orders[i] + 1;
        let sigma = T::from(sigma_f64).unwrap();

        // Compute Gaussian envelope in f64 (where exp/sqrt are natively available),
        // then cast to T. This preserves exact numerical results across f32/f64.
        let norm_factor = sigma_f64 * (2.0_f64 * std::f64::consts::PI).sqrt();
        let x_values_f64: Array1<f64> = (0..num_points).map(|i| -half_width + (i as f64)).collect();
        let gauss_term_f64 =
            x_values_f64.mapv(|x| ((-x.powi(2) / (2.0 * sigma_f64.powi(2))).exp()) / norm_factor);
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
///     Maximum Hermite order for each corresponding width (must be the same
///     length as ``widths``). Each width-ordered pair generates a triangular
///     set of y- and x-Hermite polynomials as described above.
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
/// All returned arrays use ``float32`` dtype.
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
///     Maximum Hermite order for each corresponding width (must be the same
///     length as ``widths``). Each width-ordered pair generates a triangular
///     set of y- and x-Hermite polynomials as described above.
///
/// Returns
/// -------
/// list of tuple of ``numpy.ndarray`` of float32
///     Each element is a ``(y_kernel, x_kernel)`` pair. Each kernel is a
///     1-D NumPy array of length ``2 * half_width + 1`` with dtype ``float32``
///     representing a Gaussian-weighted Hermite polynomial.
///
/// See Also
/// --------
/// generate_gauss_hermite_basis
///     Float64 variant, also aliased as ``generate_gauss_hermite_basis``.
/// DiffKernel.solve_diff_kernel
///     Fits optimal kernel coefficients against template/target images.
///
/// Examples
/// --------
/// >>> basis = generate_gauss_hermite_basis_f32(2.0, [0.5, 1.0], [1, 2])
/// >>> len(basis)
/// 9
/// >>> y_k, x_k = basis[0]
/// >>> y_k.shape
/// (5,)
/// >>> y_k.dtype
/// dtype('float32')
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

    fn create_test_kernel_f64() -> DiffKernelData<f64> {
        let basis1 = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let basis2 = Array1::from_vec(vec![5.0f64, 4.0, 3.0, 2.0, 1.0]);
        let coeffs = Array1::from_vec(vec![0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6]);
        DiffKernelData {
            basis_arrays: vec![(basis1, basis2)],
            basis_radius: 2,
            spatial_order: 1,
            basis_coefficients: coeffs,
        }
    }

    fn create_test_kernel_f32() -> DiffKernelData<f32> {
        let basis1 = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let basis2 = Array1::from_vec(vec![5.0f32, 4.0, 3.0, 2.0, 1.0]);
        let coeffs = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]);
        DiffKernelData {
            basis_arrays: vec![(basis1, basis2)],
            basis_radius: 2,
            spatial_order: 1,
            basis_coefficients: coeffs,
        }
    }

    #[test]
    fn test_json_roundtrip_inner_f64() {
        let original = create_test_kernel_f64();
        let tagged = DiffKernelInner::F64(original);
        let json = serde_json::to_string(&tagged).unwrap();
        let restored: DiffKernelInner = serde_json::from_str(&json).unwrap();

        match (&tagged, &restored) {
            (DiffKernelInner::F64(o), DiffKernelInner::F64(r)) => {
                assert_eq!(o.basis_radius, r.basis_radius);
                assert_eq!(o.spatial_order, r.spatial_order);
                assert!(o
                    .basis_coefficients
                    .iter()
                    .zip(r.basis_coefficients.iter())
                    .all(|(a, b)| (a - b).abs() < f64::EPSILON * 10.0));
                assert_eq!(o.basis_arrays.len(), r.basis_arrays.len());
                for ((oy, ox), (ry, rx)) in o.basis_arrays.iter().zip(r.basis_arrays.iter()) {
                    assert!(oy
                        .iter()
                        .zip(ry.iter())
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON * 10.0));
                    assert!(ox
                        .iter()
                        .zip(rx.iter())
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON * 10.0));
                }
            }
            _ => panic!("Expected F64 variant"),
        }
    }

    #[test]
    fn test_json_roundtrip_inner_f32() {
        let original = create_test_kernel_f32();
        let tagged = DiffKernelInner::F32(original);
        let json = serde_json::to_string(&tagged).unwrap();
        let restored: DiffKernelInner = serde_json::from_str(&json).unwrap();

        match (&tagged, &restored) {
            (DiffKernelInner::F32(o), DiffKernelInner::F32(r)) => {
                assert_eq!(o.basis_radius, r.basis_radius);
                assert_eq!(o.spatial_order, r.spatial_order);
                assert!(o
                    .basis_coefficients
                    .iter()
                    .zip(r.basis_coefficients.iter())
                    .all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0));
                assert_eq!(o.basis_arrays.len(), r.basis_arrays.len());
                for ((oy, ox), (ry, rx)) in o.basis_arrays.iter().zip(r.basis_arrays.iter()) {
                    assert!(oy
                        .iter()
                        .zip(ry.iter())
                        .all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0));
                    assert!(ox
                        .iter()
                        .zip(rx.iter())
                        .all(|(a, b)| (a - b).abs() < f32::EPSILON * 10.0));
                }
            }
            _ => panic!("Expected F32 variant"),
        }
    }

    #[test]
    fn test_from_json_invalid() {
        let res: Result<DiffKernelInner, _> = serde_json::from_str("not valid json");
        assert!(res.is_err());
    }

    #[test]
    fn test_deserialize_f64_by_dtype() {
        // Serialize a real f64 kernel, then deserialize and verify
        let original = create_test_kernel_f64();
        let tagged = DiffKernelInner::F64(original);
        let json_str = serde_json::to_string(&tagged).unwrap();
        // Verify the JSON has the correct dtype tag
        let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(
            value.get("dtype").and_then(|v| v.as_str()),
            Some("DiffKernel")
        );
        // Deserialize back
        let result: DiffKernelInner = serde_json::from_str(&json_str).unwrap();
        match result {
            DiffKernelInner::F64(k) => {
                assert_eq!(k.basis_radius, 2);
                assert_eq!(k.spatial_order, 1);
                assert_eq!(k.basis_coefficients.len(), 6);
            }
            _ => panic!("Expected F64 variant"),
        }
    }

    #[test]
    fn test_deserialize_f32_by_dtype() {
        // Serialize a real f32 kernel, then deserialize and verify
        let original = create_test_kernel_f32();
        let tagged = DiffKernelInner::F32(original);
        let json_str = serde_json::to_string(&tagged).unwrap();
        // Verify the JSON has the correct dtype tag
        let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(
            value.get("dtype").and_then(|v| v.as_str()),
            Some("DiffKernelF32")
        );
        // Deserialize back
        let result: DiffKernelInner = serde_json::from_str(&json_str).unwrap();
        match result {
            DiffKernelInner::F32(k) => {
                assert_eq!(k.basis_radius, 2);
                assert_eq!(k.spatial_order, 1);
                assert_eq!(k.basis_coefficients.len(), 6);
            }
            _ => panic!("Expected F32 variant"),
        }
    }

    #[test]
    fn test_deserialize_unknown_dtype() {
        let json = serde_json::json!({
            "dtype": "UnknownType",
            "basis_arrays": [],
            "basis_radius": 0,
            "spatial_order": 0,
            "basis_coefficients": []
        });
        let json_str = serde_json::to_string(&json).unwrap();
        let result: Result<DiffKernelInner, _> = serde_json::from_str(&json_str);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("UnknownType"));
    }

    #[test]
    fn test_deserialize_missing_dtype() {
        let json = serde_json::json!({
            "basis_arrays": [],
            "basis_radius": 0,
            "spatial_order": 0,
            "basis_coefficients": []
        });
        let json_str = serde_json::to_string(&json).unwrap();
        let result: Result<DiffKernelInner, _> = serde_json::from_str(&json_str);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("dtype"));
    }

    /// Verify serialization of DiffKernelInner produces the expected "dtype" field.
    #[test]
    fn test_serialize_f64_has_dtype_tag() {
        let original = create_test_kernel_f64();
        let tagged = DiffKernelInner::F64(original);
        let json = serde_json::to_string(&tagged).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("dtype").and_then(|v| v.as_str()),
            Some("DiffKernel")
        );
        assert!(value.get("basis_radius").is_some());
        assert!(value.get("basis_arrays").is_some());
    }

    #[test]
    fn test_serialize_f32_has_dtype_tag() {
        let original = create_test_kernel_f32();
        let tagged = DiffKernelInner::F32(original);
        let json = serde_json::to_string(&tagged).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("dtype").and_then(|v| v.as_str()),
            Some("DiffKernelF32")
        );
        assert!(value.get("basis_radius").is_some());
    }

    /// Test DiffKernel wrapper construction and clone.
    #[test]
    fn test_diff_kernel_wrapper_f64() {
        let inner = create_test_kernel_f64();
        let dk = DiffKernel {
            inner: DiffKernelInner::F64(inner.clone()),
        };
        let dk2 = dk.clone();
        // Round-trip via json
        match &dk2.inner {
            DiffKernelInner::F64(k) => {
                assert_eq!(k.basis_radius, 2);
                assert_eq!(k.basis_coefficients.len(), 6);
            }
            _ => panic!("Expected F64"),
        }
    }

    #[test]
    fn test_diff_kernel_wrapper_f32() {
        let inner = create_test_kernel_f32();
        let dk = DiffKernel {
            inner: DiffKernelInner::F32(inner.clone()),
        };
        let dk2 = dk.clone();
        match &dk2.inner {
            DiffKernelInner::F32(k) => {
                assert_eq!(k.basis_radius, 2);
                assert_eq!(k.basis_coefficients.len(), 6);
            }
            _ => panic!("Expected F32"),
        }
    }

    /// Test apply_kernel method on DiffKernelData<f64>
    #[test]
    fn test_apply_kernel_f64() {
        let kernel = create_test_kernel_f64();
        // Create a simple 10x10 test image
        let input = Array2::<f64>::from_shape_fn((10, 10), |(_, _)| 1.0);
        let result = kernel.apply_kernel(input.view());
        // Output should be (10 - 2*2, 10 - 2*2) = (6, 6)
        assert_eq!(result.shape(), &[6, 6]);
    }

    /// Test apply_kernel method on DiffKernelData<f32>
    #[test]
    fn test_apply_kernel_f32() {
        let kernel = create_test_kernel_f32();
        // Create a simple 10x10 test image
        let input = Array2::<f32>::from_shape_fn((10, 10), |(_, _)| 1.0f32);
        let result = kernel.apply_kernel(input.view());
        // Output should be (10 - 2*2, 10 - 2*2) = (6, 6)
        assert_eq!(result.shape(), &[6, 6]);
    }

    /// Test _draw_unweighted_basis method
    #[test]
    fn test_draw_unweighted_basis() {
        let kernel = create_test_kernel_f64();
        let basis = kernel._draw_unweighted_basis(0);
        // Should be (2*radius+1, 2*radius+1) = (5, 5)
        assert_eq!(basis.shape(), &[5, 5]);
    }

    /// Verify ndarray serde serialization format matches the JSON schema.
    /// Array1<T> serializes as {"v": 1, "dim": [N], "data": [...]}.
    /// Vec<(Array1, Array1)> serializes as [[obj, obj], ...].
    #[test]
    fn test_json_schema_structure_matches_serialization() {
        let original = create_test_kernel_f64();
        let tagged = DiffKernelInner::F64(original);
        let json = serde_json::to_string(&tagged).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        // dtype is a string
        assert!(value.get("dtype").is_some());
        assert_eq!(value["dtype"], "DiffKernel");

        // basis_coefficients has ndarray format: {v, dim, data}
        let bc = &value["basis_coefficients"];
        assert!(bc.get("v").is_some(), "basis_coefficients missing 'v'");
        assert!(bc.get("dim").is_some(), "basis_coefficients missing 'dim'");
        assert!(
            bc.get("data").is_some(),
            "basis_coefficients missing 'data'"
        );
        assert_eq!(bc["v"], 1);
        assert!(bc["dim"].is_array());
        assert!(bc["data"].is_array());
        assert_eq!(bc["data"].as_array().unwrap().len(), 6);

        // basis_arrays is an array of 2-element arrays of ndarray objects
        let ba = &value["basis_arrays"];
        assert!(ba.is_array());
        assert_eq!(ba.as_array().unwrap().len(), 1);
        let first_pair = &ba[0];
        assert!(first_pair.is_array());
        assert_eq!(first_pair.as_array().unwrap().len(), 2);
        // Both elements have ndarray object format
        for elem in first_pair.as_array().unwrap() {
            assert!(elem.get("v").is_some(), "basis_arrays element missing 'v'");
            assert!(
                elem.get("dim").is_some(),
                "basis_arrays element missing 'dim'"
            );
            assert!(
                elem.get("data").is_some(),
                "basis_arrays element missing 'data'"
            );
            assert_eq!(elem["v"], 1);
        }
    }

    /// Test solve_diff_kernel_impl produces a valid kernel
    #[test]
    fn test_solve_impl_f64() {
        // Create minimal 5x5 basis functions (identity-like kernel)
        let basis_y = Array1::<f64>::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let basis_x = Array1::<f64>::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let basis_functions = vec![(basis_y.view(), basis_x.view())];

        // Create 20x20 template with known values
        let mut template = Array2::<f64>::zeros((20, 20));
        template[[8, 8]] = 1.0;
        template[[10, 10]] = 2.0;
        template[[12, 12]] = 3.0;

        // Create 16x16 target with known values
        let mut target = Array2::<f64>::zeros((16, 16));
        target[[6, 6]] = 1.5;
        target[[8, 8]] = 2.5;
        target[[10, 10]] = 3.5;

        // Point sources well inside the image (offset from edges by radius=2)
        let x_values = Array1::from_vec(vec![8i32, 10i32, 12i32]);
        let y_values = Array1::from_vec(vec![8i32, 10i32, 12i32]);

        let result = solve_diff_kernel_impl(
            x_values.view(),
            y_values.view(),
            basis_functions,
            0u32,
            template.view(),
            target.view(),
        );

        assert_eq!(result.basis_radius, 2);
        assert_eq!(result.spatial_order, 0);
        // With spatial_order=0, size=1, 1 basis function: num_parameters = 1 * 1 = 1
        assert_eq!(result.basis_coefficients.len(), 1);
    }
}
