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

#ifndef _VOLUMEFILTER_KERNEL_CU_
#define _VOLUMEFILTER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include "volumeFilter.h"

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

__constant__ float4 c_filterData[VOLUMEFILTER_MAXWEIGHTS];

__global__ void d_filter_surface3d(int filterSize, float filter_offset,
                                   cudaExtent volumeSize,
                                   cudaTextureObject_t volumeTexIn,
                                   cudaSurfaceObject_t volumeTexOut) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= volumeSize.width || y >= volumeSize.height ||
      z >= volumeSize.depth) {
    return;
  }

  float filtered = 0;
  float4 basecoord = make_float4(x, y, z, 0);

  for (int i = 0; i < filterSize; i++) {
    float4 coord = basecoord + c_filterData[i];
    filtered += tex3D<float>(volumeTexIn, coord.x, coord.y, coord.z) *
                c_filterData[i].w;
  }

  filtered += filter_offset;

  VolumeType output = VolumeTypeInfo<VolumeType>::convert(filtered);

  // surface writes need byte offsets for x!
  surf3Dwrite(output, volumeTexOut, x * sizeof(VolumeType), y, z);
}

static unsigned int iDivUp(size_t a, size_t b) {
  size_t val = (a % b != 0) ? (a / b + 1) : (a / b);
  if (val > UINT_MAX) {
    fprintf(stderr, "\nUINT_MAX limit exceeded in iDivUp() exiting.....\n");
    exit(EXIT_FAILURE);  // val exceeds limit
  }

  return static_cast<unsigned int>(val);
}

extern "C" Volume *VolumeFilter_runFilter(Volume *input, Volume *output0,
                                          Volume *output1, int iterations,
                                          int numWeights, float4 *weights,
                                          float postWeightOffset) {
  Volume *swap = 0;
  cudaExtent size = input->size;
  unsigned int dim = 32 / sizeof(VolumeType);
  dim3 blockSize(dim, dim, 1);
  dim3 gridSize(iDivUp(size.width, blockSize.x),
                iDivUp(size.height, blockSize.y),
                iDivUp(size.depth, blockSize.z));

  // set weights
  checkCudaErrors(
      cudaMemcpyToSymbol(c_filterData, weights, sizeof(float4) * numWeights));

  for (int i = 0; i < iterations; i++) {
    d_filter_surface3d<<<gridSize, blockSize>>>(numWeights, postWeightOffset,
                                                size, input->volumeTex,
                                                output0->volumeSurf);

    getLastCudaError("filter kernel failed");

    swap = input;
    input = output0;
    output0 = swap;

    if (i == 0) {
      output0 = output1;
    }
  }

  return input;
}
#endif
