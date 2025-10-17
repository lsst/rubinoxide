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
use log;
use ndarray::linalg::general_mat_mul;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, NdFloat};
use ndarray_linalg::Inverse;
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::cmp;

const MAX_NUM_SCALES: usize = 10;
const B_SPLINE_SIGMA: f64 = 2.0553651328015339;
const H: usize = 1;
const KAPPA: f64 = 0.25;

#[derive(Clone, Copy)]
enum IsotropyType {
    Isotrope,
    Isophote,
    Gradient,
}

#[inline]
fn find_gradients<T: NdFloat + Default>(pixels: [T; 9]) -> [T; 2] {
    [
        (pixels[7] - pixels[1]) / T::from(2.0).unwrap(),
        (pixels[5] - pixels[3]) / T::from(2.0).unwrap(),
    ]
}

#[inline]
fn isotrop_laplacian<T: NdFloat + Default>() -> [T; 9] {
    [
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(-3.0).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.25).unwrap(),
        T::from(0.5).unwrap(),
        T::from(0.25).unwrap(),
    ]
}

#[inline]
fn rotation_matrix_isophote<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
) -> [[T; 2]; 2] {
    let mut a: [[T; 2]; 2] = [[T::default(); 2]; 2];
    a[0][0] = cos_theta2 + c2 * sin_theta2;
    a[1][1] = c2 * cos_theta2 + sin_theta2;
    a[0][1] = (c2 - T::from(1.0).unwrap()) * cos_theta_sin_theta;
    a[1][0] = a[0][1];
    a
}

#[inline]
fn rotation_matrix_gradient<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
) -> [[T; 2]; 2] {
    let mut a: [[T; 2]; 2] = [[T::default(); 2]; 2];
    a[0][0] = c2 * cos_theta2 + sin_theta2;
    a[1][1] = cos_theta2 + c2 * sin_theta2;
    a[0][1] = (T::from(1.0).unwrap() - c2) * cos_theta_sin_theta;
    a[1][0] = a[0][1];
    a
}

#[inline]
fn build_matrix<T: NdFloat + Default>(a: [[T; 2]; 2]) -> [T; 9] {
    let b11 = a[0][1] / T::from(2.0).unwrap();
    let b13 = -b11;
    let b22 = T::from(-2.0).unwrap() * (a[0][0] + a[1][1]);

    [b11, a[1][1], b13, a[0][0], b22, a[0][0], b13, a[1][1], b11]
}

#[inline]
fn compute_kernel<T: NdFloat + Default>(
    c2: T,
    cos_theta_sin_theta: T,
    cos_theta2: T,
    sin_theta2: T,
    isotropy_type: &IsotropyType,
) -> [T; 9] {
    match isotropy_type {
        IsotropyType::Isotrope => isotrop_laplacian(),
        IsotropyType::Isophote => {
            let iso_matrix =
                rotation_matrix_isophote(c2, cos_theta_sin_theta, cos_theta2, sin_theta2);
            build_matrix(iso_matrix)
        }
        IsotropyType::Gradient => {
            let rot_matrix =
                rotation_matrix_gradient(c2, cos_theta_sin_theta, cos_theta2, sin_theta2);
            build_matrix(rot_matrix)
        }
    }
}

fn heat_PDE_diffusion<T: NdFloat + Default>(
    hf_input: ArrayView2<T>,
    lf_input: ArrayView2<T>,
    output: ArrayViewMut2<T>,
    mult: usize,
    anisotropy: [T; 4],
    isotropy_type: [IsotropyType; 4],
    variance_threshold: T,
    regularization: T,
    current_radius_sq: T,
    ABCD: [T; 4],
    strength: T,
    mask: &Option<ArrayView2<bool>>,
) {
    let mut output = output;
    let regularization_factor = regularization * current_radius_sq / T::from(9.0).unwrap();
    let mut i_neighbours: [usize; 3] = [0, 0, 0];
    let mut j_neighbours: [usize; 3] = [0, 0, 0];

    let mut neighbour_pixel_hf = [T::default(); 9];
    let mut neighbour_pixel_lf = [T::default(); 9];

    let mut c2 = [T::default(); 4];

    let (height, width) = output.dim();

    // for row in &process_points.0 {
    for row in 0..height {
        i_neighbours[0] = (cmp::max(row as i32 - (mult * H) as i32, 0) as i32) as usize;
        i_neighbours[1] = row;
        i_neighbours[2] = cmp::min((row + mult * H) as i32, height as i32 - 1) as usize;
        // for col in &process_points.1 {
        for col in 0..width {
            j_neighbours[0] = cmp::max(col as i32 - (mult * H) as i32, 0) as usize;
            j_neighbours[1] = col;
            j_neighbours[2] = cmp::min((col + mult * H) as i32, width as i32 - 1) as usize;

            let do_pixel = match mask {
                Some(m) => m[(row, col)],
                None => true,
            };

            if do_pixel {
                for ii in 0..3 {
                    for jj in 0..3 {
                        neighbour_pixel_hf[3 * ii + jj] =
                            hf_input[(i_neighbours[ii], j_neighbours[jj])];
                        neighbour_pixel_lf[3 * ii + jj] =
                            lf_input[(i_neighbours[ii], j_neighbours[jj])];
                    }
                }

                let mut gradient = find_gradients(neighbour_pixel_lf);
                let mut laplace = find_gradients(neighbour_pixel_hf);

                let magnitude_grad = (gradient[0].powi(2) + gradient[1].powi(2)).sqrt();
                c2[0] = -magnitude_grad * anisotropy[0];
                c2[2] = -magnitude_grad * anisotropy[2];

                if magnitude_grad != T::default() {
                    gradient[0] /= magnitude_grad;
                    gradient[1] /= magnitude_grad;
                } else {
                    gradient[0] = T::from(1.0).unwrap();
                    gradient[1] = T::default();
                }
                let cos_theta_grad_sq = gradient[0].powi(2);
                let sin_theta_grad_sq = gradient[1].powi(2);
                let cos_theta_sin_theta_grad = gradient[0] * gradient[1];

                let magnitude_lapl = (laplace[0].powi(2) + laplace[1].powi(2)).sqrt();
                c2[1] = -magnitude_lapl * anisotropy[1];
                c2[3] = -magnitude_lapl * anisotropy[3];

                if magnitude_lapl != T::default() {
                    laplace[0] /= magnitude_lapl;
                    laplace[1] /= magnitude_lapl;
                } else {
                    laplace[0] = T::from(1.0).unwrap();
                    laplace[1] = T::default();
                }

                let cos_theta_lapl_sq = laplace[0].powi(2);
                let sin_theta_lapl_sq = laplace[1].powi(2);
                let cos_theta_sin_theta_lapl = laplace[0] * laplace[1];

                for k in 0..4 {
                    c2[k] = c2[k].exp();
                }
                let kern_first = compute_kernel(
                    c2[0],
                    cos_theta_sin_theta_grad,
                    cos_theta_grad_sq,
                    sin_theta_grad_sq,
                    &isotropy_type[0],
                );
                let kern_second = compute_kernel(
                    c2[1],
                    cos_theta_sin_theta_lapl,
                    cos_theta_lapl_sq,
                    sin_theta_lapl_sq,
                    &isotropy_type[1],
                );
                let kern_third = compute_kernel(
                    c2[2],
                    cos_theta_sin_theta_grad,
                    cos_theta_grad_sq,
                    sin_theta_grad_sq,
                    &isotropy_type[2],
                );
                let kern_fourth = compute_kernel(
                    c2[3],
                    cos_theta_sin_theta_lapl,
                    cos_theta_lapl_sq,
                    sin_theta_lapl_sq,
                    &isotropy_type[3],
                );

                let mut derivatives: [T; 4] = [T::default(); 4];
                let mut variance = T::default();
                for k in 0..9 {
                    derivatives[0] += kern_first[k] * neighbour_pixel_lf[k];
                    derivatives[1] += kern_second[k] * neighbour_pixel_lf[k];
                    derivatives[2] += kern_third[k] * neighbour_pixel_hf[k];
                    derivatives[3] += kern_fourth[k] * neighbour_pixel_hf[k];
                    variance += neighbour_pixel_hf[k].powi(2);
                }

                variance = variance_threshold + variance * regularization_factor;

                let mut acc = T::default();
                for k in 0..4 {
                    acc += derivatives[k] * ABCD[k];
                }

                acc = hf_input[(row, col)] * strength + acc / variance;
                output[(row, col)] = (acc + lf_input[(row, col)]).max(T::default());
            } else {
                output[(row, col)] = hf_input[(row, col)] + lf_input[(row, col)];
            }
        }
    }
}

#[inline]
fn compute_anisotropy_factor<T: NdFloat + Default>(user_param: T) -> T {
    user_param.powi(2)
}

#[inline]
fn check_isotropy_mode<T: NdFloat + Default>(anisotropy: T) -> IsotropyType {
    if anisotropy == T::default() {
        IsotropyType::Isotrope
    } else if anisotropy > T::default() {
        IsotropyType::Isophote
    } else {
        IsotropyType::Gradient
    }
}

// #[inline]
// fn sparse_scalar_product()

#[inline]
fn _bspline_vertical_pass<T: NdFloat + Default>(
    in_array: ArrayViewMut2<T>,
    row: usize,
    width: usize,
    height: usize,
    mult: i32,
    clip_negatives: bool,
) -> Array1<T> {
    let irow = row as i32;
    let indicies: [usize; 5] = [
        cmp::max(irow - 2 * mult, 0) as usize,
        cmp::max(irow - mult, 0) as usize,
        row,
        cmp::min((irow + mult) as usize, height - 1),
        cmp::min((irow + 2 * mult) as usize, height - 1),
    ];

    let filter: [T; 5] = [
        T::from(1.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(6.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(1.0 / 16.0).unwrap(),
    ];

    (0..width)
        .map(|index| {
            let val_sum = (0..5).fold(T::default(), |acc, k| {
                acc + in_array[(indicies[k], index)] * filter[k]
            });
            if clip_negatives {
                val_sum.max(T::default())
            } else {
                val_sum
            }
        })
        .collect::<Array1<T>>()
}

#[inline]
fn _bspline_horizontal<T: NdFloat + Default>(
    in_array: &ArrayViewMut1<T>,
    col: usize,
    width: usize,
    mult: i32,
    clip_negatives: bool,
) -> T {
    let icol = col as i32;
    let indicies: [usize; 5] = [
        cmp::max(icol - 2 * mult, 0) as usize,
        cmp::max(icol - mult, 0) as usize,
        col,
        cmp::min((icol + mult) as usize, width - 1),
        cmp::min((icol + 2 * mult) as usize, width - 1),
    ];

    let filter: [T; 5] = [
        T::from(1.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(6.0 / 16.0).unwrap(),
        T::from(4.0 / 16.0).unwrap(),
        T::from(1.0 / 16.0).unwrap(),
    ];

    let val_sum = (0..5).fold(T::default(), |acc, k| {
        acc + in_array[indicies[k]] * filter[k]
    });
    if clip_negatives {
        val_sum.max(T::default())
    } else {
        val_sum
    }
}

#[inline]
fn decompose_2D_Bspline<T: NdFloat + Default>(
    in_array: ArrayViewMut2<T>,
    hf: ArrayViewMut2<T>,
    lf: ArrayViewMut2<T>,
    width: usize,
    height: usize,
    mult: i32,
) {
    let mut hf = hf;
    let mut lf = lf;
    let mut in_array = in_array;
    for row in 0..height {
        let mut row_conv =
            _bspline_vertical_pass(in_array.view_mut(), row, width, height, mult, true);
        for col in 0..width {
            let blur = _bspline_horizontal(&row_conv.view_mut(), col, width, mult, true);
            let index = (row, col);
            lf[index] = blur;
            hf[index] = in_array[index] - blur;
        }
    }
}

#[inline]
fn equivalent_sigma_at_step<T: NdFloat + Default>(sigma: T, s: usize) -> T {
    if s == 0 {
        T::from(sigma).unwrap()
    } else {
        ((equivalent_sigma_at_step(sigma, s - 1)).powi(2)
            + (T::from(s).unwrap().exp2() * sigma).powi(2))
        .sqrt()
    }
}

#[inline]
fn num_steps_to_reach_equivalent_sigma<T: NdFloat + Default>(
    sigma_filter: T,
    sigma_final: T,
) -> usize {
    let mut s: usize = 0;
    let mut radius = sigma_filter;
    while radius < sigma_final {
        s += 1;
        radius = (radius.powi(2) + T::from(1 << s).unwrap() * sigma_filter).sqrt();
    }
    s + 1
}
fn wavelets_process<T: NdFloat + Default>(
    process_args: &ProcessArgs<T>,
    scales: usize,
    input: &mut ArrayViewMut2<T>,
    reconstructed: &mut ArrayViewMut2<T>,
    lf_odd: &mut Array2<T>,
    lf_even: &mut Array2<T>,
    hf: &mut Vec<Array2<T>>,
    zoom: T,
    mask: &Option<ArrayView2<bool>>,
) {
    let anisotropy = [
        compute_anisotropy_factor(process_args.anisotropy_first),
        compute_anisotropy_factor(process_args.anisotropy_second),
        compute_anisotropy_factor(process_args.anisotropy_third),
        compute_anisotropy_factor(process_args.anisotropy_fourth),
    ];

    let isotropy_type = [
        check_isotropy_mode(process_args.anisotropy_first),
        check_isotropy_mode(process_args.anisotropy_second),
        check_isotropy_mode(process_args.anisotropy_third),
        check_isotropy_mode(process_args.anisotropy_fourth),
    ];

    let regularization =
        T::from(10.0).unwrap().powf(process_args.regularization) - T::from(1.0).unwrap();

    let variance_threshold = T::from(10.0).unwrap().powf(process_args.variance_threshold);

    let mut buffer_in: ArrayViewMut2<T>;
    let mut buffer_out: ArrayViewMut2<T>;
    let residual: &mut Array2<T>;
    let temp: &mut Array2<T>;

    let (height, width) = input.dim();

    let mut final_scale: usize = 0;
    for sc in 0..scales {
        let mult = 1 << sc;

        if sc == 0 {
            buffer_in = input.view_mut();
            buffer_out = lf_odd.view_mut();
        } else if (sc % 2) != 0 {
            buffer_in = lf_odd.view_mut();
            buffer_out = lf_even.view_mut();
        } else {
            buffer_in = lf_even.view_mut();
            buffer_out = lf_odd.view_mut();
        }

        decompose_2D_Bspline(
            buffer_in,
            hf[sc].view_mut(),
            buffer_out,
            width,
            height,
            mult,
        );

        final_scale = sc;

        // needed for second borrow
    }

    if final_scale == 0 {
        residual = lf_odd;
        temp = lf_even;
    } else if (final_scale % 2) != 0 {
        residual = lf_even;
        temp = lf_odd;
    } else {
        residual = lf_odd;
        temp = lf_even;
    }

    let kappa = T::from(KAPPA).unwrap();

    let mut buffer_in: ArrayView2<T>;
    for (count, scale) in (0..=scales - 1).rev().enumerate() {
        let mult = 1 << scale;
        let current_radius = equivalent_sigma_at_step(T::from(B_SPLINE_SIGMA).unwrap(), scale);
        let real_radius = current_radius * zoom;
        let norm =
            (-(real_radius - process_args.radius_center).powi(2) / process_args.radius).exp();
        let ABCD = [
            process_args.first * kappa * norm,
            process_args.second * kappa * norm,
            process_args.third * kappa * norm,
            process_args.fourth * kappa * norm,
        ];
        let strength = process_args.sharpness * norm + T::from(1.0).unwrap();

        if count == 0 {
            buffer_in = residual.view();
            buffer_out = temp.view_mut();
        } else if (count % 2) != 0 {
            buffer_in = temp.view();
            buffer_out = residual.view_mut();
        } else {
            buffer_in = residual.view();
            buffer_out = temp.view_mut();
        }

        if scale == 0 {
            buffer_out = reconstructed.view_mut();
        }

        heat_PDE_diffusion(
            hf[scale].view(),
            buffer_in,
            buffer_out,
            mult,
            anisotropy,
            isotropy_type,
            variance_threshold,
            regularization,
            current_radius.powi(2),
            ABCD,
            strength,
            mask,
        );
    }
}

struct ProcessArgs<T: NdFloat + Default> {
    // mask: Option<(Array1<usize>, Array1<usize>)>,
    iterations: usize,
    anisotropy_first: T,
    anisotropy_second: T,
    anisotropy_third: T,
    anisotropy_fourth: T,
    regularization: T,
    variance_threshold: T,
    radius_center: T,
    radius: T,
    first: T,
    second: T,
    third: T,
    fourth: T,
    sharpness: T,
}

fn process_image<T: NdFloat + Default>(
    process_args: ProcessArgs<T>,
    image_in: &mut ArrayViewMut2<T>,
    mask: Option<ArrayView2<bool>>,
) -> Array2<T> {
    let im_dim = image_in.dim();
    let mut image_out = Array2::<T>::zeros(image_in.dim());
    let mut temp_1 = Array2::<T>::zeros(image_in.dim());
    let mut temp_2 = Array2::<T>::zeros(image_in.dim());
    let mut lf_odd = Array2::<T>::zeros(image_in.dim());
    let mut lf_even = Array2::<T>::zeros(image_in.dim());

    let final_radius = process_args.radius + process_args.radius_center * T::from(2.0).unwrap();

    let iterations = cmp::max(process_args.iterations, 1);
    let diffusion_scales =
        num_steps_to_reach_equivalent_sigma(T::from(B_SPLINE_SIGMA).unwrap(), final_radius);
    let scales = diffusion_scales.clamp(1, MAX_NUM_SCALES);

    let mut hf = (0..scales)
        .map(|_| Array2::<T>::zeros(image_in.dim()))
        .collect::<Vec<_>>();

    let zoom = T::from(1.0).unwrap();

    // let passthrough_points = (Array1::<usize>::zeros(0), Array1::<usize>::zeros(0));
    // let process_points = (
    //     Array1::<usize>::from_iter(0..im_dim.0),
    //     Array1::<usize>::from_iter(0..im_dim.1),
    // );

    let temp_1_ref = &mut temp_1.view_mut();
    let temp_2_ref = &mut temp_2.view_mut();
    let mut temp_out: &mut ArrayViewMut2<T>;
    let image_out_ref = &mut image_out.view_mut();
    for it in 0..iterations {
        if it == 0 {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_2_ref;
            }
            wavelets_process(
                &process_args,
                scales,
                image_in,
                temp_out,
                &mut lf_odd,
                &mut lf_even,
                &mut hf,
                zoom,
                &mask,
            );
        } else if (it % 2) == 0 {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_2_ref;
            }
            wavelets_process(
                &process_args,
                scales,
                temp_1_ref,
                temp_out,
                &mut lf_odd,
                &mut lf_even,
                &mut hf,
                zoom,
                &mask,
            );
        } else {
            if it == (iterations - 1) {
                temp_out = image_out_ref;
            } else {
                temp_out = temp_1_ref;
            }
            wavelets_process(
                &process_args,
                scales,
                temp_2_ref,
                temp_out,
                &mut lf_odd,
                &mut lf_even,
                &mut hf,
                zoom,
                &mask,
            );
        }
    }
    image_out
}

#[pyfunction]
#[pyo3(signature = (image,
    iterations= 3,
    anisotropy_first= 1.0,
    anisotropy_second= 1.0,
    anisotropy_third= 1.0,
    anisotropy_fourth= 1.0,
    regularization= 2.94,
    variance_threshold= 0.0,
    radius_center= 0.0,
    first= 0.0065,
    second= -0.25,
    third= -0.25,
    fourth= -0.2774,
    radius= 5.0,
    sharpness= 0.0,
))]
fn diffuse_gray_image<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f64>,
    iterations: usize,
    anisotropy_first: f64,
    anisotropy_second: f64,
    anisotropy_third: f64,
    anisotropy_fourth: f64,
    regularization: f64,
    variance_threshold: f64,
    radius_center: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
    radius: f64,
    sharpness: f64,
) -> Bound<'py, PyArray2<f64>> {
    unsafe {
        let mut array = image.as_array_mut();

        let process_args = ProcessArgs {
            iterations,
            anisotropy_first,
            anisotropy_second,
            anisotropy_third,
            anisotropy_fourth,
            regularization,
            variance_threshold,
            radius_center,
            first,
            second,
            third,
            fourth,
            radius,
            sharpness,
        };
        let result = process_image(process_args, &mut array, None);
        result.to_pyarray(py)
    }
}

fn replace_masked_with_noise<T: NdFloat + Default>(
    image: ArrayView2<T>,
    mask: &ArrayView2<bool>,
) -> Array2<T>
where
    StandardNormal: Distribution<T>,
{
    let im_dim = image.dim();
    let mut result = Array2::<T>::zeros(im_dim);
    for i in 0..im_dim.0 {
        for j in 0..im_dim.1 {
            let val = image[(i, j)];
            result[(i, j)] = if mask[(i, j)] {
                let normal = Normal::new(val, val).unwrap();

                normal.sample(&mut rand::rng())
            } else {
                image[(i, j)]
            }
        }
    }
    result
}

#[pyfunction]
#[pyo3(signature = (image,
    mask,
    iterations= 32,
    anisotropy_first= 0.0,
    anisotropy_second= 0.0,
    anisotropy_third= 0.0,
    anisotropy_fourth= 2.0,
    regularization= 0.0,
    variance_threshold= 0.0,
    radius_center= 0.0,
    first= 0.0,
    second= 0.0,
    third= 0.0,
    fourth= 1.0,
    radius= 5.0,
    sharpness= 0.0,
))]
fn inpaint_mask<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f64>,
    mask: PyReadonlyArray2<bool>,
    iterations: usize,
    anisotropy_first: f64,
    anisotropy_second: f64,
    anisotropy_third: f64,
    anisotropy_fourth: f64,
    regularization: f64,
    variance_threshold: f64,
    radius_center: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
    radius: f64,
    sharpness: f64,
) -> Bound<'py, PyArray2<f64>> {
    let array = image.as_array();
    let mask_array = mask.as_array();

    let process_args = ProcessArgs {
        iterations,
        anisotropy_first,
        anisotropy_second,
        anisotropy_third,
        anisotropy_fourth,
        regularization,
        variance_threshold,
        radius_center,
        first,
        second,
        third,
        fourth,
        radius,
        sharpness,
    };
    let mut masked = replace_masked_with_noise(array, &mask_array);
    let result = process_image(process_args, &mut masked.view_mut(), Some(mask_array));
    result.to_pyarray(py)
}

pub fn create_rgb_diffusion_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let rgb_diffusion_module = PyModule::new(parent_module.py(), "rgb_diffusion")?;
    rgb_diffusion_module
        .add_function(wrap_pyfunction!(diffuse_gray_image, &rgb_diffusion_module)?)?;
    rgb_diffusion_module.add_function(wrap_pyfunction!(inpaint_mask, &rgb_diffusion_module)?)?;
    parent_module.add_submodule(&rgb_diffusion_module)
}
