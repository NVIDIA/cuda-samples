/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" bool runTest(const int argc, const char **argv, char *data,
                        int2 *data_int2, unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // input data
  int len = 16;
  // the data has some zero padding at the end so that the size is a multiple of
  // four, this simplifies the processing as each thread can process four
  // elements (which is necessary to avoid bank conflicts) but no branching is
  // necessary to avoid out of bounds reads
  char str[] = {82,  111, 118, 118, 121, 42, 97, 121,
                124, 118, 110, 56,  10,  10, 10, 10};

  // Use int2 showing that CUDA vector types can be used in cpp code
  int2 i2[16];

  for (int i = 0; i < len; i++) {
    i2[i].x = str[i];
    i2[i].y = 10;
  }

  bool bTestResult;

  // run the device part of the program
  bTestResult = runTest(argc, (const char **)argv, str, i2, len);

  std::cout << str << std::endl;

  char str_device[16];

  for (int i = 0; i < len; i++) {
    str_device[i] = (char)(i2[i].x);
  }

  std::cout << str_device << std::endl;

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
