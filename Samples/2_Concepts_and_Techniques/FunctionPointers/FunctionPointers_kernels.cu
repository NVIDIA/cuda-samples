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

#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>

#include "FunctionPointers_kernels.h"

// Texture object for reading image
cudaTextureObject_t tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

// pixel value used for thresholding function,
// works well with sample image 'teapot512'
#define THRESHOLD 150.0f

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

// A function pointer can be declared explicitly like this line:
//__device__ unsigned char (*pointFunction)(unsigned char, float ) = NULL;
// or by using typedef's like below:

typedef unsigned char (*blockFunction_t)(unsigned char, unsigned char,
                                         unsigned char, unsigned char,
                                         unsigned char, unsigned char,
                                         unsigned char, unsigned char,
                                         unsigned char, float);

typedef unsigned char (*pointFunction_t)(unsigned char, float);

__device__ blockFunction_t blockFunction;

__device__ unsigned char ComputeSobel(unsigned char ul,  // upper left
                                      unsigned char um,  // upper middle
                                      unsigned char ur,  // upper right
                                      unsigned char ml,  // middle left
                                      unsigned char mm,  // middle (unused)
                                      unsigned char mr,  // middle right
                                      unsigned char ll,  // lower left
                                      unsigned char lm,  // lower middle
                                      unsigned char lr,  // lower right
                                      float fScale) {
  short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
  short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
  short Sum = (short)(fScale * (abs((int)Horz) + abs((int)Vert)));
  return (unsigned char)((Sum < 0) ? 0 : ((Sum > 255) ? 255 : Sum));
}

// define a function pointer and initialize to NULL
__device__ unsigned char (*varFunction)(unsigned char, unsigned char,
                                        unsigned char, unsigned char,
                                        unsigned char, unsigned char,
                                        unsigned char, unsigned char,
                                        unsigned char, float x) = NULL;

__device__ unsigned char ComputeBox(unsigned char ul,  // upper left
                                    unsigned char um,  // upper middle
                                    unsigned char ur,  // upper right
                                    unsigned char ml,  // middle left
                                    unsigned char mm,  // middle...middle
                                    unsigned char mr,  // middle right
                                    unsigned char ll,  // lower left
                                    unsigned char lm,  // lower middle
                                    unsigned char lr,  // lower right
                                    float fscale) {
  short Sum = (short)(ul + um + ur + ml + mm + mr + ll + lm + lr) / 9;
  Sum *= fscale;
  return (unsigned char)((Sum < 0) ? 0 : ((Sum > 255) ? 255 : Sum));
}
__device__ unsigned char Threshold(unsigned char in, float thresh) {
  if (in > thresh) {
    return 0xFF;
  } else {
    return 0;
  }
}

// Declare function tables, one for the point function chosen, one for the
// block function chosen.  The number of entries is determined by the
// enum in FunctionPointers_kernels.h
__device__ blockFunction_t blockFunction_table[LAST_BLOCK_FILTER];
__device__ pointFunction_t pointFunction_table[LAST_POINT_FILTER];

// Declare device side function pointers.  We retrieve them later with
// cudaMemcpyFromSymbol to set our function tables above in some
// particular order specified at runtime.
__device__ blockFunction_t pComputeSobel = ComputeSobel;
__device__ blockFunction_t pComputeBox = ComputeBox;
__device__ pointFunction_t pComputeThreshold = Threshold;

// Allocate host side tables to mirror the device side, and later, we
// fill these tables with the function pointers.  This lets us send
// the pointers to the kernel on invocation, as a method of choosing
// which function to run.
blockFunction_t h_blockFunction_table[2];
pointFunction_t h_pointFunction_table[2];

// Perform a filter operation on the data, using shared memory
// The actual operation performed is
// determined by the function pointer "blockFunction" and selected
// by the integer argument "blockOperation" and has access
// to an apron around the current pixel being processed.
// Following the block operation, a per-pixel operation,
// pointed to by pPointFunction is performed before the final
// pixel is produced.
__global__ void SobelShared(uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
                            short BlockWidth, short SharedPitch,
#endif
                            short w, short h, float fScale, int blockOperation,
                            pointFunction_t pPointFunction,
                            cudaTextureObject_t tex) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  short u = 4 * blockIdx.x * BlockWidth;
  short v = blockIdx.y * blockDim.y + threadIdx.y;
  short ib;

  int SharedIdx = threadIdx.y * SharedPitch;

  for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
    LocalBlock[SharedIdx + 4 * ib + 0] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 0), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 1] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 1), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 2] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 2), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 3] = tex2D<unsigned char>(
        tex, (float)(u + 4 * ib - RADIUS + 3), (float)(v - RADIUS));
  }

  if (threadIdx.y < RADIUS * 2) {
    //
    // copy trailing RADIUS*2 rows of pixels into shared
    //
    SharedIdx = (blockDim.y + threadIdx.y) * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
      LocalBlock[SharedIdx + 4 * ib + 0] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 0),
                               (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 1] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 1),
                               (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 2] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 2),
                               (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 3] =
          tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 3),
                               (float)(v + blockDim.y - RADIUS));
    }
  }

  cg::sync(cta);

  u >>= 2;  // index as uchar4 from here
  uchar4 *pSobel = (uchar4 *)(((char *)pSobelOriginal) + v * SobelPitch);
  SharedIdx = threadIdx.y * SharedPitch;

  blockFunction = blockFunction_table[blockOperation];

  for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x) {
    uchar4 out;

    unsigned char pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 0];
    unsigned char pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 1];
    unsigned char pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 2];
    unsigned char pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 0];
    unsigned char pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 1];
    unsigned char pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 2];
    unsigned char pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 0];
    unsigned char pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 1];
    unsigned char pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 2];

    out.x = (*blockFunction)(pix00, pix01, pix02, pix10, pix11, pix12, pix20,
                             pix21, pix22, fScale);

    pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 3];
    pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 3];
    pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 3];
    out.y = (*blockFunction)(pix01, pix02, pix00, pix11, pix12, pix10, pix21,
                             pix22, pix20, fScale);

    pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 4];
    pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 4];
    pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 4];
    out.z = (*blockFunction)(pix02, pix00, pix01, pix12, pix10, pix11, pix22,
                             pix20, pix21, fScale);

    pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 5];
    pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 5];
    pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 5];
    out.w = (*blockFunction)(pix00, pix01, pix02, pix10, pix11, pix12, pix20,
                             pix21, pix22, fScale);

    if (pPointFunction != NULL) {
      out.x = (*pPointFunction)(out.x, THRESHOLD);
      out.y = (*pPointFunction)(out.y, THRESHOLD);
      out.z = (*pPointFunction)(out.z, THRESHOLD);
      out.w = (*pPointFunction)(out.w, THRESHOLD);
    }

    if (u + ib < w / 4 && v < h) {
      pSobel[u + ib] = out;
    }
  }

  cg::sync(cta);
}

__global__ void SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch, int w,
                               int h, float fscale, cudaTextureObject_t tex) {
  unsigned char *pSobel =
      (unsigned char *)(((char *)pSobelOriginal) + blockIdx.x * Pitch);

  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    pSobel[i] = min(
        max((tex2D<unsigned char>(tex, (float)i, (float)blockIdx.x) * fscale),
            0.f),
        255.f);
  }
}

// Perform block and pointer filtering using texture lookups.
// The block and point operations are determined by the
// input argument (see comment above for "SobelShared" function)
__global__ void SobelTex(Pixel *pSobelOriginal, unsigned int Pitch, int w,
                         int h, float fScale, int blockOperation,
                         pointFunction_t pPointOperation,
                         cudaTextureObject_t tex) {
  unsigned char *pSobel =
      (unsigned char *)(((char *)pSobelOriginal) + blockIdx.x * Pitch);
  unsigned char tmp = 0;

  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    unsigned char pix00 =
        tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x - 1);
    unsigned char pix01 =
        tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x - 1);
    unsigned char pix02 =
        tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x - 1);
    unsigned char pix10 =
        tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 0);
    unsigned char pix11 =
        tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 0);
    unsigned char pix12 =
        tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 0);
    unsigned char pix20 =
        tex2D<unsigned char>(tex, (float)i - 1, (float)blockIdx.x + 1);
    unsigned char pix21 =
        tex2D<unsigned char>(tex, (float)i + 0, (float)blockIdx.x + 1);
    unsigned char pix22 =
        tex2D<unsigned char>(tex, (float)i + 1, (float)blockIdx.x + 1);
    tmp = (*(blockFunction_table[blockOperation]))(
        pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22, fScale);

    if (pPointOperation != NULL) {
      tmp = (*pPointOperation)(tmp, 150.0);
    }

    pSobel[i] = tmp;
  }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp) {
  cudaChannelFormatDesc desc;

  if (Bpp == 1) {
    desc = cudaCreateChannelDesc<unsigned char>();
  } else {
    desc = cudaCreateChannelDesc<uchar4>();
  }

  checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
  checkCudaErrors(cudaMemcpy2DToArray(
      array, 0, 0, data, iw * Bpp * sizeof(Pixel), iw * Bpp * sizeof(Pixel), ih,
      cudaMemcpyHostToDevice));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = array;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
}

extern "C" void deleteTexture(void) {
  checkCudaErrors(cudaFreeArray(array));
  checkCudaErrors(cudaDestroyTextureObject(tex));
}

// Copy the pointers from the function tables to the host side
void setupFunctionTables() {
  // Dynamically assign the function table.
  // Copy the function pointers to their appropriate locations according to the
  // enum
  checkCudaErrors(cudaMemcpyFromSymbol(&h_blockFunction_table[SOBEL_FILTER],
                                       pComputeSobel, sizeof(blockFunction_t)));
  checkCudaErrors(cudaMemcpyFromSymbol(&h_blockFunction_table[BOX_FILTER],
                                       pComputeBox, sizeof(blockFunction_t)));

  // do the same for the point function, where the 2nd function is NULL ("no-op"
  // filter, skipped in kernel code)
  checkCudaErrors(cudaMemcpyFromSymbol(&h_pointFunction_table[THRESHOLD_FILTER],
                                       pComputeThreshold,
                                       sizeof(pointFunction_t)));
  h_pointFunction_table[NULL_FILTER] = NULL;

  // now copy the function tables back to the device, so if we wish we can use
  // an index into the table to choose them
  // We have now set the order in the function table according to our enum.
  checkCudaErrors(
      cudaMemcpyToSymbol(blockFunction_table, h_blockFunction_table,
                         sizeof(blockFunction_t) * LAST_BLOCK_FILTER));
  checkCudaErrors(
      cudaMemcpyToSymbol(pointFunction_table, h_pointFunction_table,
                         sizeof(pointFunction_t) * LAST_POINT_FILTER));
}

// Wrapper for the __global__ call that sets up the texture and threads
// Below two methods for selecting the image processing function to run are
// shown.
// BlockOperation is an integer kernel argument used as an index into the
// blockFunction_table on the device side
// pPointOp is itself a function pointer passed as a kernel argument, retrieved
// from a host side copy of the function table
extern "C" void sobelFilter(Pixel *odata, int iw, int ih,
                            enum SobelDisplayMode mode, float fScale,
                            int blockOperation, int pointOperation) {
  pointFunction_t pPointOp = h_pointFunction_table[pointOperation];

  switch (mode) {
    case SOBELDISPLAY_IMAGE:
      SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale, tex);
      break;

    case SOBELDISPLAY_SOBELTEX:
      SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale, blockOperation, pPointOp,
                            tex);
      break;

    case SOBELDISPLAY_SOBELSHARED: {
      dim3 threads(16, 4);
#ifndef FIXED_BLOCKWIDTH
      int BlockWidth = 80;  // must be divisible by 16 for coalescing
#endif
      dim3 blocks = dim3(iw / (4 * BlockWidth) + (0 != iw % (4 * BlockWidth)),
                         ih / threads.y + (0 != ih % threads.y));
      int SharedPitch = ~0x3f & (4 * (BlockWidth + 2 * RADIUS) + 0x3f);
      int sharedMem = SharedPitch * (threads.y + 2 * RADIUS);

      // for the shared kernel, width must be divisible by 4
      iw &= ~3;

      SobelShared<<<blocks, threads, sharedMem>>>(
          (uchar4 *)odata, iw,
#ifndef FIXED_BLOCKWIDTH
          BlockWidth, SharedPitch,
#endif
          iw, ih, fScale, blockOperation, pPointOp, tex);
    } break;
  }
}
