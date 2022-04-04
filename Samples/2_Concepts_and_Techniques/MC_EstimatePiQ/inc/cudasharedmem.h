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

#ifndef CUDASHAREDMEM_H
#define CUDASHAREDMEM_H

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          extern __shared__  T sdata[];
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//
// With this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          SharedMemory<T> sdata;
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//****************************************************************************

// This is the un-specialized struct.  Note that we prevent instantiation of
// this struct by making it abstract (i.e. with pure virtual methods).
template <typename T>
struct SharedMemory {
  // Ensure that we won't compile any un-specialized types
  virtual __device__ T &operator*() = 0;
  virtual __device__ T &operator[](int i) = 0;
};

#define BUILD_SHAREDMEMORY_TYPE(t, n) \
  template <>                         \
  struct SharedMemory<t> {            \
    __device__ t &operator*() {       \
      extern __shared__ t n[];        \
      return *n;                      \
    }                                 \
    __device__ t &operator[](int i) { \
      extern __shared__ t n[];        \
      return n[i];                    \
    }                                 \
  }

BUILD_SHAREDMEMORY_TYPE(int, s_int);
BUILD_SHAREDMEMORY_TYPE(unsigned int, s_uint);
BUILD_SHAREDMEMORY_TYPE(char, s_char);
BUILD_SHAREDMEMORY_TYPE(unsigned char, s_uchar);
BUILD_SHAREDMEMORY_TYPE(short, s_short);
BUILD_SHAREDMEMORY_TYPE(unsigned short, s_ushort);
BUILD_SHAREDMEMORY_TYPE(long, s_long);
BUILD_SHAREDMEMORY_TYPE(unsigned long, s_ulong);
BUILD_SHAREDMEMORY_TYPE(bool, s_bool);
BUILD_SHAREDMEMORY_TYPE(float, s_float);
BUILD_SHAREDMEMORY_TYPE(double, s_double);

#endif
