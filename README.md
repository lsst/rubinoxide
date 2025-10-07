# lsst-rubin-oxide

Utility code for the [Rubin Science Pipelines](https://pipelines.lsst.io) written in Rust and bound to Python with PyO3. The public API of the code is intended to be Python only; the fact that most of the package is written in Rust is an implementation detail for performance reasons.

When this library is used through eups for local development developers should:
* Clone the repo
* Setup the repo through eups
* Edit the source code, either python code in the python directory, or rust code in the src directory
* run make build and re-deploy the code, this must be done even if only the python code changed. (alternatively type make install to skip tests)
