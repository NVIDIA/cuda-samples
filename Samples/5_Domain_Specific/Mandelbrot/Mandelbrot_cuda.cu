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
#include "helper_cuda.h"
#include "Mandelbrot_kernel.h"
#include "Mandelbrot_kernel.cuh"

// The Mandelbrot CUDA GPU thread function

template <class T>
__global__ void Mandelbrot0(uchar4 *dst, const int imageW, const int imageH,
                            const int crunch, const T xOff, const T yOff,
                            const T xJP, const T yJP, const T scale,
                            const uchar4 colors, const int frame,
                            const int animationFrame, const int gridWidth,
                            const int numBlocks, const bool isJ) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // Calculate the location
      const T xPos = (T)ix * scale + xOff;
      const T yPos = (T)iy * scale + yOff;

      // Calculate the Mandelbrot index for the current location
      int m = CalcMandelbrot<T>(xPos, yPos, xJP, yJP, crunch, isJ);
      //            int m = blockIdx.x;         // uncomment to see scheduling
      //            order
      m = m > 0 ? crunch - m : 0;

      // Convert the Mandelbrot index into a color
      uchar4 color;

      if (m) {
        m += animationFrame;
        color.x = m * colors.x;
        color.y = m * colors.y;
        color.z = m * colors.z;
      } else {
        color.x = 0;
        color.y = 0;
        color.z = 0;
      }

      // Output the pixel
      int pixel = imageW * iy + ix;

      if (frame == 0) {
        color.w = 0;
        dst[pixel] = color;
      } else {
        int frame1 = frame + 1;
        int frame2 = frame1 / 2;
        dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
        dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
        dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
      }
    }
  }

}  // Mandelbrot0

// The Mandelbrot CUDA GPU thread function (double single version)
__global__ void MandelbrotDS0(uchar4 *dst, const int imageW, const int imageH,
                              const int crunch, const float xOff0,
                              const float xOff1, const float yOff0,
                              const float yOff1, const float xJP,
                              const float yJP, const float scale,
                              const uchar4 colors, const int frame,
                              const int animationFrame, const int gridWidth,
                              const int numBlocks, const bool isJ) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // Calculate the location
      float xPos0 = (float)ix * scale;
      float xPos1 = 0.0f;
      float yPos0 = (float)iy * scale;
      float yPos1 = 0.0f;
      dsadd(xPos0, xPos1, xPos0, xPos1, xOff0, xOff1);
      dsadd(yPos0, yPos1, yPos0, yPos1, yOff0, yOff1);

      // Calculate the Mandelbrot index for the current location
      int m =
          CalcMandelbrotDS(xPos0, xPos1, yPos0, yPos1, xJP, yJP, crunch, isJ);
      m = m > 0 ? crunch - m : 0;

      // Convert the Mandelbrot index into a color
      uchar4 color;

      if (m) {
        m += animationFrame;
        color.x = m * colors.x;
        color.y = m * colors.y;
        color.z = m * colors.z;
      } else {
        color.x = 0;
        color.y = 0;
        color.z = 0;
      }

      // Output the pixel
      int pixel = imageW * iy + ix;

      if (frame == 0) {
        color.w = 0;
        dst[pixel] = color;
      } else {
        int frame1 = frame + 1;
        int frame2 = frame1 / 2;
        dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
        dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
        dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
      }
    }
  }
}  // MandelbrotDS0

// The Mandelbrot secondary AA pass CUDA GPU thread function
template <class T>
__global__ void Mandelbrot1(uchar4 *dst, const int imageW, const int imageH,
                            const int crunch, const T xOff, const T yOff,
                            const T xJP, const T yJP, const T scale,
                            const uchar4 colors, const int frame,
                            const int animationFrame, const int gridWidth,
                            const int numBlocks, const bool isJ) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // Get the current pixel color
      int pixel = imageW * iy + ix;
      uchar4 pixelColor = dst[pixel];
      int count = 0;

      // Search for pixels out of tolerance surrounding the current pixel
      if (ix > 0) {
        count += CheckColors(pixelColor, dst[pixel - 1]);
      }

      if (ix + 1 < imageW) {
        count += CheckColors(pixelColor, dst[pixel + 1]);
      }

      if (iy > 0) {
        count += CheckColors(pixelColor, dst[pixel - imageW]);
      }

      if (iy + 1 < imageH) {
        count += CheckColors(pixelColor, dst[pixel + imageW]);
      }

      if (count) {
        // Calculate the location
        const T xPos = (T)ix * scale + xOff;
        const T yPos = (T)iy * scale + yOff;

        // Calculate the Mandelbrot index for the current location
        int m = CalcMandelbrot(xPos, yPos, xJP, yJP, crunch, isJ);
        m = m > 0 ? crunch - m : 0;

        // Convert the Mandelbrot index into a color
        uchar4 color;

        if (m) {
          m += animationFrame;
          color.x = m * colors.x;
          color.y = m * colors.y;
          color.z = m * colors.z;
        } else {
          color.x = 0;
          color.y = 0;
          color.z = 0;
        }

        // Output the pixel
        int frame1 = frame + 1;
        int frame2 = frame1 / 2;
        dst[pixel].x = (pixelColor.x * frame + color.x + frame2) / frame1;
        dst[pixel].y = (pixelColor.y * frame + color.y + frame2) / frame1;
        dst[pixel].z = (pixelColor.z * frame + color.z + frame2) / frame1;
      }
    }
  }

}  // Mandelbrot1

// The Mandelbrot secondary AA pass CUDA GPU thread function (double single
// version)
__global__ void MandelbrotDS1(uchar4 *dst, const int imageW, const int imageH,
                              const int crunch, const float xOff0,
                              const float xOff1, const float yOff0,
                              const float yOff1, const float xJP,
                              const float yJP, const float scale,
                              const uchar4 colors, const int frame,
                              const int animationFrame, const int gridWidth,
                              const int numBlocks, const bool isJ) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // Get the current pixel color
      int pixel = imageW * iy + ix;
      uchar4 pixelColor = dst[pixel];
      int count = 0;

      // Search for pixels out of tolerance surrounding the current pixel
      if (ix > 0) {
        count += CheckColors(pixelColor, dst[pixel - 1]);
      }

      if (ix + 1 < imageW) {
        count += CheckColors(pixelColor, dst[pixel + 1]);
      }

      if (iy > 0) {
        count += CheckColors(pixelColor, dst[pixel - imageW]);
      }

      if (iy + 1 < imageH) {
        count += CheckColors(pixelColor, dst[pixel + imageW]);
      }

      if (count) {
        // Calculate the location
        float xPos0 = (float)ix * scale;
        float xPos1 = 0.0f;
        float yPos0 = (float)iy * scale;
        float yPos1 = 0.0f;
        dsadd(xPos0, xPos1, xPos0, xPos1, xOff0, xOff1);
        dsadd(yPos0, yPos1, yPos0, yPos1, yOff0, yOff1);

        // Calculate the Mandelbrot index for the current location
        int m =
            CalcMandelbrotDS(xPos0, xPos1, yPos0, yPos1, xJP, yJP, crunch, isJ);
        m = m > 0 ? crunch - m : 0;

        // Convert the Mandelbrot index into a color
        uchar4 color;

        if (m) {
          m += animationFrame;
          color.x = m * colors.x;
          color.y = m * colors.y;
          color.z = m * colors.z;
        } else {
          color.x = 0;
          color.y = 0;
          color.z = 0;
        }

        // Output the pixel
        int frame1 = frame + 1;
        int frame2 = frame1 / 2;
        dst[pixel].x = (pixelColor.x * frame + color.x + frame2) / frame1;
        dst[pixel].y = (pixelColor.y * frame + color.y + frame2) / frame1;
        dst[pixel].z = (pixelColor.z * frame + color.z + frame2) / frame1;
      }
    }
  }

}  // MandelbrotDS1

// The host CPU Mandelbrot thread spawner
void RunMandelbrot0(uchar4 *dst, const int imageW, const int imageH,
                    const int crunch, const double xOff, const double yOff,
                    const double xjp, const double yjp, const double scale,
                    const uchar4 colors, const int frame,
                    const int animationFrame, const int mode, const int numSMs,
                    const bool isJ, int version) {
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

  int numWorkerBlocks = numSMs;

  switch (mode) {
    default:
    case 0:
      Mandelbrot0<float><<<numWorkerBlocks, threads>>>(
          dst, imageW, imageH, crunch, (float)xOff, (float)yOff, (float)xjp,
          (float)yjp, (float)scale, colors, frame, animationFrame, grid.x,
          grid.x * grid.y, isJ);
      break;
    case 1:
      float x0, x1, y0, y1;
      dsdeq(x0, x1, xOff);
      dsdeq(y0, y1, yOff);
      MandelbrotDS0<<<numWorkerBlocks, threads>>>(
          dst, imageW, imageH, crunch, x0, x1, y0, y1, (float)xjp, (float)yjp,
          (float)scale, colors, frame, animationFrame, grid.x, grid.x * grid.y,
          isJ);
      break;
    case 2:
      Mandelbrot0<double><<<numWorkerBlocks, threads>>>(
          dst, imageW, imageH, crunch, xOff, yOff, xjp, yjp, scale, colors,
          frame, animationFrame, grid.x, grid.x * grid.y, isJ);
      break;
  }

  getLastCudaError("Mandelbrot0 kernel execution failed.\n");
}  // RunMandelbrot0

// The host CPU Mandelbrot thread spawner
void RunMandelbrot1(uchar4 *dst, const int imageW, const int imageH,
                    const int crunch, const double xOff, const double yOff,
                    const double xjp, const double yjp, const double scale,
                    const uchar4 colors, const int frame,
                    const int animationFrame, const int mode, const int numSMs,
                    const bool isJ, int version) {
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

  int numWorkerBlocks = numSMs;

  switch (mode) {
    default:
    case 0:
      Mandelbrot1<float><<<numWorkerBlocks, threads>>>(
          dst, imageW, imageH, crunch, (float)xOff, (float)yOff, (float)xjp,
          (float)yjp, (float)scale, colors, frame, animationFrame, grid.x,
          grid.x * grid.y, isJ);
      break;
    case 1:
      float x0, x1, y0, y1;
      dsdeq(x0, x1, xOff);
      dsdeq(y0, y1, yOff);
      MandelbrotDS1<<<numWorkerBlocks, threads>>>(
          dst, imageW, imageH, crunch, x0, x1, y0, y1, (float)xjp, (float)yjp,
          (float)scale, colors, frame, animationFrame, grid.x, grid.x * grid.y,
          isJ);
      break;
    case 2:
      Mandelbrot1<double><<<numWorkerBlocks, threads>>>(
          dst, imageW, imageH, crunch, xOff, yOff, xjp, yjp, scale, colors,
          frame, animationFrame, grid.x, grid.x * grid.y, isJ);
      break;
  }

  getLastCudaError("Mandelbrot1 kernel execution failed.\n");
}  // RunMandelbrot1
