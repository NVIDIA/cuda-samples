Introduction
============

This sample demonstrates linking a libnvvm-generated module with an existing
CUDA C library. The LLVM C++ API is used to generate an LLVM IR module that
conforms to the NVVM IR specification and contains a call to an externally-
defined function, and this module is compiled to PTX with libnvvm. The JIT
linker (part of the CUDA Driver API) is then used to assemble the PTX and link
it with the math library, creating a linked CUBIN image. This image is then
executed on the first CUDA device on the system.

Files
-----

- cuda-c-linking.cpp    - Main source file demonstrating the generated of a
                          PTX file using libnvvm and linking it with a CUDA C
                          device library

- math-funcs            - CUDA C device library source file

- CMakeLists.txt        - CMake build script

Building
--------

This sample is optionally built as part of the libnvvm samples from the CUDA
samples tree.  Please see the README file at the root of the libnvvm samples
for build instructions.

Usage
-----

Once built, the sample can be executed by running the "cuda-c-linking" binary.

Linux:

    $ cd $SAMPLES_INSTALL_DIR
    $ ./cuda-c-linking

Windows:

    $ cd %SAMPLES_INSTALL_DIR%
    $ cuda-c-linking.exe

For inspection purposes, the following command-line options are available:

- -save-ptx     - Write generated PTX kernel to cuda-c-linking.kernel.ptx
- -save-ir      - Write generated LLVM IR to cuda-c-linking.kernel.ll
- -save-cubin   - Write linked CUBIN image to cuda-c-linking.linked.cubin
