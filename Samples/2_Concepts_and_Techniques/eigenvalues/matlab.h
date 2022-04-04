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

/* Header for utility functionality.
* Host code.
*/

#ifndef _MATLAB_H_
#define _MATLAB_H_

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project

////////////////////////////////////////////////////////////////////////////////
//! Write a tridiagonal, symmetric matrix in vector representation and
//! it's eigenvalues
//! @param  filename  name of output file
//! @param  d  diagonal entries of the matrix
//! @param  s  superdiagonal entries of the matrix (len = n - 1)
//! @param  eigenvals  eigenvalues of the matrix
//! @param  indices  vector of len n containing the position of the eigenvalues
//!                  if these are sorted in ascending order
//! @param  n  size of the matrix
////////////////////////////////////////////////////////////////////////////////
extern "C" void writeTridiagSymMatlab(const char *filename, float *d, float *s,
                                      float *eigenvals, const unsigned int n);

////////////////////////////////////////////////////////////////////////////////
//! Write matrix to a file in Matlab format
//! @param  file  file handle to which to write he matrix
//! @param  mat_name  name of matrix in Matlab
//! @param  mat  matrix to write to the file
//! @param  mat_size  size of the (square) matrix \a mat
////////////////////////////////////////////////////////////////////////////////
template <class T, class S>
void writeMatrixMatlab(T &file, const char *mat_name, S *&mat,
                       const unsigned int mat_size);

////////////////////////////////////////////////////////////////////////////////
//! Write vector to a file in Matlab format
//! @param  file  file handle to which to write he matrix
//! @param  vec_name  name of vector in Matlab
//! @param  vec  matrix to write to the file
//! @param  vec_len  length of the vector
////////////////////////////////////////////////////////////////////////////////
template <class T, class S>
void writeVectorMatlab(T &file, const char *vec_name, S *&vec,
                       const unsigned int vec_len);

// implementations

////////////////////////////////////////////////////////////////////////////////
//! Write matrix to a file in Matlab format
//! @param  file  file handle to which to write he matrix
//! @param  mat_name  name of matrix in Matlab
//! @param  mat  matrix to write to the file
//! @param  mat_size  size of the (square) matrix \a mat
////////////////////////////////////////////////////////////////////////////////
template <class T, class S>
void writeMatrixMatlab(T &file, const char *mat_name, S *&mat,
                       const unsigned int mat_size) {
  const unsigned int pitch = sizeof(S) * mat_size;

  file << mat_name << " = [";

  for (unsigned int i = 0; i < mat_size; ++i) {
    for (unsigned int j = 0; j < mat_size; ++j) {
      file << getMatrix(mat, pitch, i, j) << " ";
    }

    if (i != mat_size - 1) {
      file << "; ";
    }
  }

  file << "];\n";
}

////////////////////////////////////////////////////////////////////////////////
//! Write vector to a file in Matlab format
//! @param  file  file handle to which to write he matrix
//! @param  vec_name  name of vector in Matlab
//! @param  vec  matrix to write to the file
//! @param  vec_len  length of the vector
////////////////////////////////////////////////////////////////////////////////
template <class T, class S>
void writeVectorMatlab(T &file, const char *vec_name, S *&vec,
                       const unsigned int vec_len) {
  file << vec_name << " = [";

  for (unsigned int i = 0; i < vec_len; ++i) {
    file << vec[i] << " ";
  }

  file << "];\n";
}

#endif  // _MATLAB_H_
