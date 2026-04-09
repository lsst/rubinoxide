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
This is a module for code related to the production of RGB images.
*/

mod color_spaces;
mod rgb_diffusion;
pub use color_spaces::{Oklab_to_RGB, RGB_to_Oklab};
use pyo3::prelude::*;
pub use rgb_diffusion::{diffuse_gray_image, inpaint_mask};

// This function is called by the main python module rubinoxide. Its job
// is to create a new sub moduled named rgb, and bind the declared
// pyfunctions to that module. Finally it adds this submodule into the main
// module.
pub fn create_rgb_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let rgb_module = PyModule::new(parent_module.py(), "rgb")?;
    rgb_module.add_function(wrap_pyfunction!(Oklab_to_RGB, &rgb_module)?)?;
    rgb_module.add_function(wrap_pyfunction!(RGB_to_Oklab, &rgb_module)?)?;
    rgb_module.add_function(wrap_pyfunction!(diffuse_gray_image, &rgb_module)?)?;
    rgb_module.add_function(wrap_pyfunction!(inpaint_mask, &rgb_module)?)?;
    parent_module.add_submodule(&rgb_module)
}
