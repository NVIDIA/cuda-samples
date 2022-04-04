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

#ifndef _VOLUME_H_
#define _VOLUME_H_

#include <cuda_runtime.h>

typedef unsigned char VolumeType;

extern "C" {

struct Volume {
  cudaArray *content;
  cudaExtent size;
  cudaChannelFormatDesc channelDesc;
  cudaTextureObject_t volumeTex;
  cudaSurfaceObject_t volumeSurf;
};

void Volume_init(Volume *vol, cudaExtent size, void *data, int allowStore);
void Volume_deinit(Volume *vol);
};

//////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

/* Helper class to do popular integer storage to float conversions if required
 */

template <typename T>
struct VolumeTypeInfo {};

template <>
struct VolumeTypeInfo<unsigned char> {
  static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;
  static __inline__ __device__ unsigned char convert(float sampled) {
    return (unsigned char)(saturate(sampled) * 255.0);
  }
};

template <>
struct VolumeTypeInfo<unsigned short> {
  static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;
  static __inline__ __device__ unsigned short convert(float sampled) {
    return (unsigned short)(saturate(sampled) * 65535.0);
  }
};

template <>
struct VolumeTypeInfo<float> {
  static const cudaTextureReadMode readMode = cudaReadModeElementType;
  static __inline__ __device__ float convert(float sampled) { return sampled; }
};

#endif

#endif
