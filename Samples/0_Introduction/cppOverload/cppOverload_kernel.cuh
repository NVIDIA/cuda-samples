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

__global__ void simple_kernel(const int *pIn, int *pOut, int a) {
  __shared__ int sData[THREAD_N];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  sData[threadIdx.x] = pIn[tid];
  __syncthreads();

  pOut[tid] = sData[threadIdx.x] * a + tid;
  ;
}

__global__ void simple_kernel(const int2 *pIn, int *pOut, int a) {
  __shared__ int2 sData[THREAD_N];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  sData[threadIdx.x] = pIn[tid];
  __syncthreads();

  pOut[tid] = (sData[threadIdx.x].x + sData[threadIdx.x].y) * a + tid;
  ;
}

__global__ void simple_kernel(const int *pIn1, const int *pIn2, int *pOut,
                              int a) {
  __shared__ int sData1[THREAD_N];
  __shared__ int sData2[THREAD_N];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  sData1[threadIdx.x] = pIn1[tid];
  sData2[threadIdx.x] = pIn2[tid];
  __syncthreads();

  pOut[tid] = (sData1[threadIdx.x] + sData2[threadIdx.x]) * a + tid;
}
