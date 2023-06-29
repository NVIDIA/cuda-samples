Introduction
============

ptxgen is a simple IR compiler that generates PTX code in stdout with the
input NVVM IR files. It generates warnings, errors, and other messages in
stderr. When multiple input files are given, it links them into a single
module before the compilation, and generates a single PTX module.

ptxgen always links the libDevice library with the input NVVM IR program.

Before compiling the input IR, ptxgen will verify the IR for conformance
to the NVVM IR specification.

Usage
-----

The command-line options, except for the program name and the input file
names, are directly passed to nvvmCompileProgram without modification.
Each input NVVM IR file can be either in the bitcode representation or
in the text representation. Input file names and command-line options can be
interleaved.

For example,

    $ ptxgen a.ll -arch=compute_50 b.bc

links a.ll and b.bc, and generates PTX code for the compute_50 architecture.
