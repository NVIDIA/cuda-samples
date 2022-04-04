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

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>  // CUDA device initialization helper functions

__constant__ float cGaussian[64];  // gaussian array in device side

cudaTextureObject_t rgbaTexdImage;
cudaTextureObject_t rgbaTexdTemp;

uint *dImage = NULL;  // original image
uint *dTemp = NULL;  // temp array for iterations
size_t pitch;

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/

// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d) {
  float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
              (b.z - a.z) * (b.z - a.z);

  return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(fabs(rgba.x));  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(fabs(rgba.y));
  rgba.z = __saturatef(fabs(rgba.z));
  rgba.w = __saturatef(fabs(rgba.w));
  return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
         (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c) {
  float4 rgba;
  rgba.x = (c & 0xff) * 0.003921568627f;          //  /255.0f;
  rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;   //  /255.0f;
  rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;  //  /255.0f;
  rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;  //  /255.0f;
  return rgba;
}

// column pass using coalesced global memory reads
__global__ void d_bilateral_filter(uint *od, int w, int h, float e_d, int r,
                                   cudaTextureObject_t rgbaTex) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= w || y >= h) {
    return;
  }

  float sum = 0.0f;
  float factor;
  float4 t = {0.f, 0.f, 0.f, 0.f};
  float4 center = tex2D<float4>(rgbaTex, x, y);

  for (int i = -r; i <= r; i++) {
    for (int j = -r; j <= r; j++) {
      float4 curPix = tex2D<float4>(rgbaTex, x + j, y + i);
      factor = cGaussian[i + r] * cGaussian[j + r] *  // domain factor
               euclideanLen(curPix, center, e_d);  // range factor

      t += factor * curPix;
      sum += factor;
    }
  }

  od[y * w + x] = rgbaFloatToInt(t / sum);
}

extern "C" void initTexture(int width, int height, uint *hImage) {
  // copy image data to array
  checkCudaErrors(
      cudaMallocPitch(&dImage, &pitch, sizeof(uint) * width, height));
  checkCudaErrors(
      cudaMallocPitch(&dTemp, &pitch, sizeof(uint) * width, height));
  checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint) * width,
                               sizeof(uint) * width, height,
                               cudaMemcpyHostToDevice));

  // texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = dImage;
  texRes.res.pitch2D.desc = desc;
  texRes.res.pitch2D.width = width;
  texRes.res.pitch2D.height = height;
  texRes.res.pitch2D.pitchInBytes = pitch;
  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(
      cudaCreateTextureObject(&rgbaTexdImage, &texRes, &texDescr, NULL));

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = dTemp;
  texRes.res.pitch2D.desc = desc;
  texRes.res.pitch2D.width = width;
  texRes.res.pitch2D.height = height;
  texRes.res.pitch2D.pitchInBytes = pitch;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeNormalizedFloat;

  checkCudaErrors(
      cudaCreateTextureObject(&rgbaTexdTemp, &texRes, &texDescr, NULL));
}

extern "C" void freeTextures() {
  checkCudaErrors(cudaDestroyTextureObject(rgbaTexdImage));
  checkCudaErrors(cudaDestroyTextureObject(rgbaTexdTemp));
  checkCudaErrors(cudaFree(dImage));
  checkCudaErrors(cudaFree(dTemp));
}

/*
    Because a 2D gaussian mask is symmetry in row and column,
    here only generate a 1D mask, and use the product by row
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
extern "C" void updateGaussian(float delta, int radius) {
  float fGaussian[64];

  for (int i = 0; i < 2 * radius + 1; ++i) {
    float x = (float)(i - radius);
    fGaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }

  checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian,
                                     sizeof(float) * (2 * radius + 1)));
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations
*/

// RGBA version
extern "C" double bilateralFilterRGBA(uint *dDest, int width, int height,
                                      float e_d, int radius, int iterations,
                                      StopWatchInterface *timer) {
  // var for kernel computation timing
  double dKernelTime;

  for (int i = 0; i < iterations; i++) {
    // sync host and start kernel computation timer
    dKernelTime = 0.0;
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&timer);

    dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
    dim3 blockSize(16, 16);

    if (iterations > 1) {
      d_bilateral_filter<<<gridSize, blockSize>>>(dDest, width, height, e_d,
                                                  radius, rgbaTexdTemp);
    } else {
      d_bilateral_filter<<<gridSize, blockSize>>>(dDest, width, height, e_d,
                                                  radius, rgbaTexdImage);
    }

    // sync host and stop computation timer
    checkCudaErrors(cudaDeviceSynchronize());
    dKernelTime += sdkGetTimerValue(&timer);

    if (iterations > 1) {
      // copy result back from global memory to array
      checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int) * width,
                                   sizeof(int) * width, height,
                                   cudaMemcpyDeviceToDevice));
    }
  }

  return ((dKernelTime / 1000.) / (double)iterations);
}
