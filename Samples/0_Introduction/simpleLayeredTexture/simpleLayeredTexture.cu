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
* This sample demonstrates how to use texture fetches from layered 2D textures
* in CUDA C
*
* This sample first generates a 3D input data array for the layered texture
* and the expected output. Then it starts CUDA C kernels, one for each layer,
* which fetch their layer's texture data (using normalized texture coordinates)
* transform it to the expected output, and write it to a 3D output data array.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

static const char *sSDKname = "simpleLayeredTexture";

////////////////////////////////////////////////////////////////////////////////
//! Transform a layer of a layered 2D texture using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float *g_odata, int width, int height,
                                int layer, cudaTextureObject_t tex) {
  // calculate this thread's data point
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 0.5f offset and division are necessary to access the original data points
  // in the texture (such that bilinear interpolation will not be activated).
  // For details, see also CUDA Programming Guide, Appendix D
  float u = (x + 0.5f) / (float)width;
  float v = (y + 0.5f) / (float)height;

  // read from texture, do expected transformation and write to global memory
  g_odata[layer * width * height + y * width + x] =
      -tex2DLayered<float>(tex, u, v, layer) + layer;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", sSDKname);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  int devID = findCudaDevice(argc, (const char **)argv);

  bool bResult = true;

  // get number of SMs on this GPU
  cudaDeviceProp deviceProps;

  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name,
         deviceProps.multiProcessorCount);
  printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

  // generate input data for layered texture
  unsigned int width = 512, height = 512, num_layers = 5;
  unsigned int size = width * height * num_layers * sizeof(float);
  float *h_data = (float *)malloc(size);

  for (unsigned int layer = 0; layer < num_layers; layer++)
    for (int i = 0; i < (int)(width * height); i++) {
      h_data[layer * width * height + i] = (float)i;
    }

  // this is the expected transformation of the input data (the expected output)
  float *h_data_ref = (float *)malloc(size);

  for (unsigned int layer = 0; layer < num_layers; layer++)
    for (int i = 0; i < (int)(width * height); i++) {
      h_data_ref[layer * width * height + i] =
          -h_data[layer * width * height + i] + layer;
    }

  // allocate device memory for result
  float *d_data = NULL;
  checkCudaErrors(cudaMalloc((void **)&d_data, size));

  // allocate array and copy image data
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cu_3darray;
  checkCudaErrors(cudaMalloc3DArray(&cu_3darray, &channelDesc,
                                    make_cudaExtent(width, height, num_layers),
                                    cudaArrayLayered));
  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0, 0, 0);
  myparms.dstPos = make_cudaPos(0, 0, 0);
  myparms.srcPtr =
      make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
  myparms.dstArray = cu_3darray;
  myparms.extent = make_cudaExtent(width, height, num_layers);
  myparms.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&myparms));

  cudaTextureObject_t tex;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = cu_3darray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  printf(
      "Covering 2D data array of %d x %d: Grid size is %d x %d, each block has "
      "8 x 8 threads\n",
      width, height, dimGrid.x, dimGrid.y);

  transformKernel<<<dimGrid, dimBlock>>>(d_data, width, height, 0,
                                         tex);  // warmup (for better timing)

  // check if kernel execution generated an error
  getLastCudaError("warmup Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // execute the kernel
  for (unsigned int layer = 0; layer < num_layers; layer++)
    transformKernel<<<dimGrid, dimBlock, 0>>>(d_data, width, height, layer,
                                              tex);

  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  printf("Processing time: %.3f msec\n", sdkGetTimerValue(&timer));
  printf("%.2f Mtexlookups/sec\n",
         (width * height * num_layers / (sdkGetTimerValue(&timer) / 1000.0f) /
          1e6));
  sdkDeleteTimer(&timer);

  // allocate mem for the result on host side
  float *h_odata = (float *)malloc(size);
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(h_odata, d_data, size, cudaMemcpyDeviceToHost));

  // write regression file if necessary
  if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
    // write file for regression test
    sdkWriteFile<float>("./data/regression.dat", h_odata, width * height, 0.0f,
                        false);
  } else {
    printf("Comparing kernel output to expected data\n");

#define MIN_EPSILON_ERROR 5e-3f
    bResult = compareData(h_odata, h_data_ref, width * height * num_layers,
                          MIN_EPSILON_ERROR, 0.0f);
  }

  // cleanup memory
  free(h_data);
  free(h_data_ref);
  free(h_odata);

  checkCudaErrors(cudaDestroyTextureObject(tex));
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFreeArray(cu_3darray));

  exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
