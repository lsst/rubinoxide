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
mod diff_kernel;
extern crate openblas_src;

use diff_kernel::{generate_gauss_hermite_basis, my_convolve, DiffKernel};
use pyo3::exceptions::PyValueError;
use pyo3::types::IntoPyDict;
use pyo3::{prelude::*, BoundObject};

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
