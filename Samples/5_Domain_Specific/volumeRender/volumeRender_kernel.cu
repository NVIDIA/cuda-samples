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

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
// typedef unsigned short VolumeType;

cudaTextureObject_t texObject;    // For 3D texture
cudaTextureObject_t transferTex;  // For 1D transfer function texture

typedef struct { float4 m[3]; } float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
  float3 o;  // origin
  float3 d;  // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear,
                            float *tfar) {
  // compute intersection of ray with all six bbox planes
  float3 invR = make_float3(1.0f) / r.d;
  float3 tbot = invR * (boxmin - r.o);
  float3 ttop = invR * (boxmax - r.o);

  // re-order intersections to find smallest and largest on each axis
  float3 tmin = fminf(ttop, tbot);
  float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v) {
  float3 r;
  r.x = dot(v, make_float3(M.m[0]));
  r.y = dot(v, make_float3(M.m[1]));
  r.z = dot(v, make_float3(M.m[2]));
  return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v) {
  float4 r;
  r.x = dot(v, M.m[0]);
  r.y = dot(v, M.m[1]);
  r.z = dot(v, M.m[2]);
  r.w = 1.0f;
  return r;
}

__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
         (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__global__ void d_render(uint *d_output, uint imageW, uint imageH,
                         float density, float brightness, float transferOffset,
                         float transferScale, cudaTextureObject_t tex,
                         cudaTextureObject_t transferTex) {
  const int maxSteps = 500;
  const float tstep = 0.01f;
  const float opacityThreshold = 0.95f;
  const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
  const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= imageW) || (y >= imageH)) return;

  float u = (x / (float)imageW) * 2.0f - 1.0f;
  float v = (y / (float)imageH) * 2.0f - 1.0f;

  // calculate eye ray in world space
  Ray eyeRay;
  eyeRay.o =
      make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
  eyeRay.d = normalize(make_float3(u, v, -2.0f));
  eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

  // find intersection with box
  float tnear, tfar;
  int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

  if (!hit) return;

  if (tnear < 0.0f) tnear = 0.0f;  // clamp to near plane

  // march along ray from front to back, accumulating color
  float4 sum = make_float4(0.0f);
  float t = tnear;
  float3 pos = eyeRay.o + eyeRay.d * tnear;
  float3 step = eyeRay.d * tstep;

  for (int i = 0; i < maxSteps; i++) {
    // read from 3D texture
    // remap position to [0, 1] coordinates
    float sample = tex3D<float>(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f,
                                pos.z * 0.5f + 0.5f);
    // sample *= 64.0f;    // scale for 10-bit data

    // lookup in transfer function texture
    float4 col =
        tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
    col.w *= density;

    // "under" operator for back-to-front blending
    // sum = lerp(sum, col, col.w);

    // pre-multiply alpha
    col.x *= col.w;
    col.y *= col.w;
    col.z *= col.w;
    // "over" operator for front-to-back blending
    sum = sum + col * (1.0f - sum.w);

    // exit early if opaque
    if (sum.w > opacityThreshold) break;

    t += tstep;

    if (t > tfar) break;

    pos += step;
  }

  sum *= brightness;

  // write output color
  d_output[y * imageW + x] = rgbaFloatToInt(sum);
}

extern "C" void setTextureFilterMode(bool bLinearFilter) {
  if (texObject) {
    checkCudaErrors(cudaDestroyTextureObject(texObject));
  }
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_volumeArray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode =
      bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;

  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.addressMode[2] = cudaAddressModeWrap;

  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(
      cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

extern "C" void initCuda(void *h_volume, cudaExtent volumeSize) {
  // create 3D array
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
  checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType),
                          volumeSize.width, volumeSize.height);
  copyParams.dstArray = d_volumeArray;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_volumeArray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords =
      true;  // access with normalized texture coordinates
  texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

  texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;

  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(
      cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

  // create transfer function texture
  float4 transferFunc[] = {
    {  0.0, 0.0, 0.0, 0.0, },
    {  1.0, 0.0, 0.0, 1.0, },
    {  1.0, 0.5, 0.0, 1.0, },
    {  1.0, 1.0, 0.0, 1.0, },
    {  0.0, 1.0, 0.0, 1.0, },
    {  0.0, 1.0, 1.0, 1.0, },
    {  0.0, 0.0, 1.0, 1.0, },
    {  1.0, 0.0, 1.0, 1.0, },
    {  0.0, 0.0, 0.0, 0.0, },
  };

  cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
  cudaArray *d_transferFuncArray;
  checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2,
                                  sizeof(transferFunc) / sizeof(float4), 1));
  checkCudaErrors(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, transferFunc,
                                      0, sizeof(transferFunc), 1,
                                      cudaMemcpyHostToDevice));

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_transferFuncArray;

  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords =
      true;  // access with normalized texture coordinates
  texDescr.filterMode = cudaFilterModeLinear;

  texDescr.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates

  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));
}

extern "C" void freeCudaBuffers() {
  checkCudaErrors(cudaDestroyTextureObject(texObject));
  checkCudaErrors(cudaDestroyTextureObject(transferTex));
  checkCudaErrors(cudaFreeArray(d_volumeArray));
  checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output,
                              uint imageW, uint imageH, float density,
                              float brightness, float transferOffset,
                              float transferScale) {
  d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                    brightness, transferOffset, transferScale,
                                    texObject, transferTex);
}

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) {
  checkCudaErrors(
      cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif  // #ifndef _VOLUMERENDER_KERNEL_CU_
