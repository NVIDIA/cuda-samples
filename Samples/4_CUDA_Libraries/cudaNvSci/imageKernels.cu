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

#include <cuda.h>
#include <helper_cuda.h>
#include <helper_image.h>

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return ((unsigned int)(rgba.w * 255.0f) << 24) |
         ((unsigned int)(rgba.z * 255.0f) << 16) |
         ((unsigned int)(rgba.y * 255.0f) << 8) |
         ((unsigned int)(rgba.x * 255.0f));
}

////////////////////////////////////////////////////////////////////////////////
//! Rotate an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
static __global__ void transformKernel(unsigned int *outputData, int width,
                                       int height, float theta,
                                       cudaTextureObject_t tex) {
  // calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  float u = (float)x - (float)width / 2;
  float v = (float)y - (float)height / 2;
  float tu = u * cosf(theta) - v * sinf(theta);
  float tv = v * cosf(theta) + u * sinf(theta);

  tu /= (float)width;
  tv /= (float)height;

  // read from texture and write to global memory
  float4 pix = tex2D<float4>(tex, tu + 0.5f, tv + 0.5f);
  unsigned int pixelInt = rgbaFloatToInt(pix);
  outputData[y * width + x] = pixelInt;
}

static __global__ void rgbToGrayscaleKernel(unsigned int *rgbaImage,
                                            size_t imageWidth,
                                            size_t imageHeight) {
  size_t gidX = blockDim.x * blockIdx.x + threadIdx.x;

  uchar4 *pixArray = (uchar4 *)rgbaImage;

  for (int pixId = gidX; pixId < imageWidth * imageHeight;
       pixId += gridDim.x * blockDim.x) {
    uchar4 dataA = pixArray[pixId];
    unsigned char grayscale =
        (unsigned char)(dataA.x * 0.3 + dataA.y * 0.59 + dataA.z * 0.11);
    uchar4 dataB = make_uchar4(grayscale, grayscale, grayscale, 0);
    pixArray[pixId] = dataB;
  }
}

void launchGrayScaleKernel(unsigned int *d_rgbaImage,
                           std::string image_filename, size_t imageWidth,
                           size_t imageHeight, cudaStream_t stream) {
  int numThreadsPerBlock = 1024;
  int numOfBlocks = (imageWidth * imageHeight) / numThreadsPerBlock;

  rgbToGrayscaleKernel<<<numOfBlocks, numThreadsPerBlock, 0, stream>>>(
      d_rgbaImage, imageWidth, imageHeight);

  unsigned int *outputData;
  checkCudaErrors(cudaMallocHost((void**)&outputData, sizeof(unsigned int) * imageWidth * imageHeight));
  checkCudaErrors(cudaMemcpyAsync(
      outputData, d_rgbaImage, sizeof(unsigned int) * imageWidth * imageHeight,
      cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  char outputFilename[1024];
  strcpy(outputFilename, image_filename.c_str());
  strcpy(outputFilename + image_filename.length() - 4, "_out.ppm");
  sdkSavePPM4ub(outputFilename, (unsigned char *)outputData, imageWidth,
                imageHeight);
  printf("Wrote '%s'\n", outputFilename);

  checkCudaErrors(cudaFreeHost(outputData));
}

void rotateKernel(cudaTextureObject_t &texObj, const float angle,
                  unsigned int *d_outputData, const int imageWidth,
                  const int imageHeight, cudaStream_t stream) {
  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(imageWidth / dimBlock.x, imageHeight / dimBlock.y, 1);

  transformKernel<<<dimGrid, dimBlock, 0, stream>>>(d_outputData, imageWidth,
                                                    imageHeight, angle, texObj);
}
