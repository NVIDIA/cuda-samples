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

/* Utility functions */

#ifndef _UTIL_H_
#define _UTIL_H_

////////////////////////////////////////////////////////////////////////////////
//! Safely free() for pointer
////////////////////////////////////////////////////////////////////////////////
template <class T>
inline void freePtr(T *&ptr) {
  if (NULL != ptr) {
    free(ptr);
    ptr = NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Minimum
////////////////////////////////////////////////////////////////////////////////
template <class T>
#ifdef __CUDACC__
__host__
    __device__
#endif
        T
        min(const T &lhs, const T &rhs) {

  return (lhs < rhs) ? lhs : rhs;
}

////////////////////////////////////////////////////////////////////////////////
//! Maximum
////////////////////////////////////////////////////////////////////////////////
template <class T>
#ifdef __CUDACC__
__host__
    __device__
#endif
        T
        max(const T &lhs, const T &rhs) {

  return (lhs < rhs) ? rhs : lhs;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (integer data type)
////////////////////////////////////////////////////////////////////////////////
template <class T>
#ifdef __CUDACC__
__host__
    __device__
#endif
        T
        sign_i(const T &val) {
  return (val < 0) ? -1 : 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (float)
////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
__host__ __device__
#endif
    inline float
    sign_f(const float &val) {
  return (val < 0.0f) ? -1.0f : 1.0f;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (double)
////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
__host__ __device__
#endif
    inline double
    sign_d(const double &val) {
  return (val < 0.0) ? -1.0 : 1.0;
}

////////////////////////////////////////////////////////////////////////////////
//! Swap \a lhs and \a rhs
////////////////////////////////////////////////////////////////////////////////
template <class T>
#ifdef __CUDACC__
__host__ __device__
#endif
    void
    swap(T &lhs, T &rhs) {

  T temp = rhs;
  rhs = lhs;
  lhs = temp;
}

///////////////////////////////////////////////////////////////////////////////
//! Get the number of blocks that are required to process \a num_threads with
//! \a num_threads_blocks threads per block
///////////////////////////////////////////////////////////////////////////////
extern "C" inline unsigned int getNumBlocksLinear(
    const unsigned int num_threads, const unsigned int num_threads_block) {
  const unsigned int block_rem =
      ((num_threads % num_threads_block) != 0) ? 1 : 0;
  return (num_threads / num_threads_block) + block_rem;
}

#endif  // #ifndef _UTIL_H_
