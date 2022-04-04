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
* CUDA Device code for particle simulation.
*/

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

cudaTextureObject_t noiseTex;
// simulation parameters
__constant__ SimParams params;

// look up in 3D noise texture
__device__ float3 noise3D(float3 p, cudaTextureObject_t noiseTex) {
  float4 n = tex3D<float4>(noiseTex, p.x, p.y, p.z);
  return make_float3(n.x, n.y, n.z);
}

// integrate particle attributes
struct integrate_functor {
  float deltaTime;
  cudaTextureObject_t noiseTex;

  __host__ __device__ integrate_functor(float delta_time,
                                        cudaTextureObject_t noise_Tex)
      : deltaTime(delta_time), noiseTex(noise_Tex) {}

  template <typename Tuple>
  __device__ void operator()(Tuple t) {
    volatile float4 posData = thrust::get<2>(t);
    volatile float4 velData = thrust::get<3>(t);

    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);

    // update particle age
    float age = posData.w;
    float lifetime = velData.w;

    if (age < lifetime) {
      age += deltaTime;
    } else {
      age = lifetime;
    }

    // apply accelerations
    vel += params.gravity * deltaTime;

    // apply procedural noise
    float3 noise = noise3D(
        pos * params.noiseFreq + params.time * params.noiseSpeed, noiseTex);
    vel += noise * params.noiseAmp;

    // new position = old position + velocity * deltaTime
    pos += vel * deltaTime;

    vel *= params.globalDamping;

    // store new position and velocity
    thrust::get<0>(t) = make_float4(pos, age);
    thrust::get<1>(t) = make_float4(vel, velData.w);
  }
};

struct calcDepth_functor {
  float3 sortVector;

  __host__ __device__ calcDepth_functor(float3 sort_vector)
      : sortVector(sort_vector) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    volatile float4 p = thrust::get<0>(t);
    float key = -dot(make_float3(p.x, p.y, p.z),
                     sortVector);  // project onto sort vector
    thrust::get<1>(t) = key;
  }
};

#endif
