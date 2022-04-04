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

/* Helper structures to simplify variable handling */

#ifndef _STRUCTS_H_
#define _STRUCTS_H_

struct InputData {
  //! host side representation of diagonal
  float *a;
  //! host side representation superdiagonal
  float *b;

  //! device side representation of diagonal
  float *g_a;
  //! device side representation of superdiagonal
  float *g_b;
  //! helper variable pointing to the mem allocated for g_b which provides
  //! space for one additional element of padding at the beginning
  float *g_b_raw;
};

struct ResultDataSmall {
  //! eigenvalues (host side)
  float *eigenvalues;

  // left interval limits at the end of the computation
  float *g_left;

  // right interval limits at the end of the computation
  float *g_right;

  // number of eigenvalues smaller than the left interval limit
  unsigned int *g_left_count;

  // number of eigenvalues bigger than the right interval limit
  unsigned int *g_right_count;

  //! flag if algorithm converged
  unsigned int *g_converged;

  // helper variables

  unsigned int mat_size_f;
  unsigned int mat_size_ui;

  float *zero_f;
  unsigned int *zero_ui;
};

struct ResultDataLarge {
  // number of intervals containing one eigenvalue after the first step
  unsigned int *g_num_one;

  // number of (thread) blocks of intervals containing multiple eigenvalues
  // after the first step
  unsigned int *g_num_blocks_mult;

  //! left interval limits of intervals containing one eigenvalue after the
  //! first iteration step
  float *g_left_one;

  //! right interval limits of intervals containing one eigenvalue after the
  //! first iteration step
  float *g_right_one;

  //! interval indices (position in sorted listed of eigenvalues)
  //! of intervals containing one eigenvalue after the first iteration step
  unsigned int *g_pos_one;

  //! left interval limits of intervals containing multiple eigenvalues
  //! after the first iteration step
  float *g_left_mult;

  //! right interval limits of intervals containing multiple eigenvalues
  //! after the first iteration step
  float *g_right_mult;

  //! number of eigenvalues less than the left limit of the eigenvalue
  //! intervals containing multiple eigenvalues
  unsigned int *g_left_count_mult;

  //! number of eigenvalues less than the right limit of the eigenvalue
  //! intervals containing multiple eigenvalues
  unsigned int *g_right_count_mult;

  //! start addresses in g_left_mult etc. of blocks of intervals containing
  //! more than one eigenvalue after the first step
  unsigned int *g_blocks_mult;

  //! accumulated number of intervals in g_left_mult etc. of blocks of
  //! intervals containing more than one eigenvalue after the first step
  unsigned int *g_blocks_mult_sum;

  //! eigenvalues that have been generated in the second step from intervals
  //! that still contained multiple eigenvalues after the first step
  float *g_lambda_mult;

  //! eigenvalue index of intervals that have been generated in the second
  //! processing step
  unsigned int *g_pos_mult;
};

#endif  // #ifndef _STRUCTS_H_
