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

It requires the LLVM development headers and libraries, version 7 or newer.
LLVM 15 and newer emit opaque pointers, which libNVVM accepts only for
Blackwell and later architectures, so such a build cannot run on an older GPU:

    $ ./cuda-c-linking
    Using CUDA Device [0]: NVIDIA L4
    Device Compute Capability: 8.9
    This sample was built against LLVM 18, which emits opaque pointers, but
    libNVVM accepts only LLVM 7 IR for compute_89. Build against LLVM 14 or
    older to run on this device, or run on a Blackwell or later device.

The sample exits with code 2 in that case, reporting an unmet requirement
rather than a failure.  Building against LLVM 7 to 14 produces a binary that
runs on any device this sample supports.

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
