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

// Utilities and system includes

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>

cudaTextureObject_t inTexObject;

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  return (int(b) << 16) | (int(g) << 8) | int(r);
}

// get pixel from 2D image, with clamping to border
__device__ uchar4 getPixel(int x, int y, cudaTextureObject_t inTex) {
#ifndef USE_TEXTURE_RGBA8UI
  float4 res = tex2D<float4>(inTex, x, y);
  uchar4 ucres = make_uchar4(res.x * 255.0f, res.y * 255.0f, res.z * 255.0f,
                             res.w * 255.0f);
#else
  uchar4 ucres = tex2D<uchar4>(inTex, x, y);
#endif
  return ucres;
}

// macros to make indexing shared memory easier
#define SMEM(X, Y) sdata[(Y)*tilew + (X)]

/*
    2D convolution using shared memory
    - operates on 8-bit RGB data stored in 32-bit int
    - assumes kernel radius is less than or equal to block size
    - not optimized for performance
     _____________
    |   :     :   |
    |_ _:_____:_ _|
    |   |     |   |
    |   |     |   |
    |_ _|_____|_ _|
  r |   :     :   |
    |___:_____:___|
      r    bw   r
    <----tilew---->
*/

__global__ void cudaProcess(unsigned int *g_odata, int imgw, int imgh,
                            int tilew, int r, float threshold, float highlight,
                            cudaTextureObject_t inTex) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ uchar4 sdata[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bw = blockDim.x;
  int bh = blockDim.y;
  int x = blockIdx.x * bw + tx;
  int y = blockIdx.y * bh + ty;

#if 0
    uchar4 c4 = getPixel(x, y);
    g_odata[y*imgw+x] = rgbToInt(c4.z, c4.y, c4.x);
#else
  // copy tile to shared memory
  // center region
  SMEM(r + tx, r + ty) = getPixel(x, y, inTex);

  // borders
  if (threadIdx.x < r) {
    // left
    SMEM(tx, r + ty) = getPixel(x - r, y, inTex);
    // right
    SMEM(r + bw + tx, r + ty) = getPixel(x + bw, y, inTex);
  }

  if (threadIdx.y < r) {
    // top
    SMEM(r + tx, ty) = getPixel(x, y - r, inTex);
    // bottom
    SMEM(r + tx, r + bh + ty) = getPixel(x, y + bh, inTex);
  }

  // load corners
  if ((threadIdx.x < r) && (threadIdx.y < r)) {
    // tl
    SMEM(tx, ty) = getPixel(x - r, y - r, inTex);
    // bl
    SMEM(tx, r + bh + ty) = getPixel(x - r, y + bh, inTex);
    // tr
    SMEM(r + bw + tx, ty) = getPixel(x + bh, y - r, inTex);
    // br
    SMEM(r + bw + tx, r + bh + ty) = getPixel(x + bw, y + bh, inTex);
  }

  // wait for loads to complete
  cg::sync(cta);

  // perform convolution
  float rsum = 0.0f;
  float gsum = 0.0f;
  float bsum = 0.0f;
  float samples = 0.0f;

  for (int dy = -r; dy <= r; dy++) {
    for (int dx = -r; dx <= r; dx++) {
#if 0
            // try this to see the benefit of using shared memory
            uchar4 pixel = getPixel(x+dx, y+dy);
#else
      uchar4 pixel = SMEM(r + tx + dx, r + ty + dy);
#endif

      // only sum pixels within disc-shaped kernel
      float l = dx * dx + dy * dy;

      if (l <= r * r) {
        float r = float(pixel.x);
        float g = float(pixel.y);
        float b = float(pixel.z);
#if 1
        // brighten highlights
        float lum = (r + g + b) / (255 * 3);

        if (lum > threshold) {
          r *= highlight;
          g *= highlight;
          b *= highlight;
        }

#endif
        rsum += r;
        gsum += g;
        bsum += b;
        samples += 1.0f;
      }
    }
  }

  rsum /= samples;
  gsum /= samples;
  bsum /= samples;
  // ABGR
  g_odata[y * imgw + x] = rgbToInt(rsum, gsum, bsum);
// g_odata[y*imgw+x] = rgbToInt(x,y,0);
#endif
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                   cudaArray *g_data_array,
                                   unsigned int *g_odata, int imgw, int imgh,
                                   int tilew, int radius, float threshold,
                                   float highlight) {
  struct cudaChannelFormatDesc desc;
  checkCudaErrors(cudaGetChannelDesc(&desc, g_data_array));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = g_data_array;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&inTexObject, &texRes, &texDescr, NULL));

#if 0
    printf("CUDA Array channel descriptor, bits per component:\n");
    printf("X %d Y %d Z %d W %d, kind %d\n",
           desc.x,desc.y,desc.z,desc.w,desc.f);

    printf("Possible values for channel format kind: i %d, u%d, f%d:\n",
           cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned,
           cudaChannelFormatKindFloat);
#endif

// printf("\n");
#ifdef GPU_PROFILING
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  int nIter = 30;

  for (int i = -1; i < nIter; ++i) {
    if (i == 0) {
      sdkStartTimer(&timer);
    }

#endif

    cudaProcess<<<grid, block, sbytes>>>(g_odata, imgw, imgh,
                                         block.x + (2 * radius), radius, 0.8f,
                                         4.0f, inTexObject);

#ifdef GPU_PROFILING
  }

  cudaDeviceSynchronize();
  sdkStopTimer(&timer);
  double dSeconds = sdkGetTimerValue(&timer) / ((double)nIter * 1000.0);
  double dNumTexels = (double)imgw * (double)imgh;
  double mtexps = 1.0e-6 * dNumTexels / dSeconds;

  if (radius == 4) {
    printf("\n");
    printf(
        "postprocessGL, Throughput = %.4f MTexels/s, Time = %.5f s, Size = "
        "%.0f Texels, NumDevsUsed = %d, Workgroup = %u\n",
        mtexps, dSeconds, dNumTexels, 1, block.x * block.y);
  }

#endif
}
