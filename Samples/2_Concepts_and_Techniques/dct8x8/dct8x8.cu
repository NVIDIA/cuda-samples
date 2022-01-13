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

/**
**************************************************************************
* \file dct8x8.cu
* \brief Contains entry point, wrappers to host and device code and benchmark.
*
* This sample implements forward and inverse Discrete Cosine Transform to blocks
* of image pixels (of 8x8 size), as in JPEG standard. The typical work flow is
*as
* follows:
* 1. Run CPU version (Host code) and measure execution time;
* 2. Run CUDA version (Device code) and measure execution time;
* 3. Output execution timings and calculate CUDA speedup.
*/

#include "Common.h"
#include "DCT8x8_Gold.h"
#include "BmpUtil.h"

/**
*  The number of DCT kernel calls
*/
#define BENCHMARK_SIZE 10

/**
*  The PSNR values over this threshold indicate images equality
*/
#define PSNR_THRESHOLD_EQUAL 40

// includes kernels
#include "dct8x8_kernel1.cuh"
#include "dct8x8_kernel2.cuh"
#include "dct8x8_kernel_short.cuh"
#include "dct8x8_kernel_quantization.cuh"

/**
**************************************************************************
*  Wrapper function for 1st gold version of DCT, quantization and IDCT
*implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperGold1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size) {
  // allocate float buffers for DCT and other data
  int StrideF;
  float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
  float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

  // convert source image to float representation
  CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
  AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

  // create and start CUDA timer
  StopWatchInterface *timerGold = 0;
  sdkCreateTimer(&timerGold);
  sdkResetTimer(&timerGold);

  // perform block-wise DCT processing and benchmarking
  for (int i = 0; i < BENCHMARK_SIZE; i++) {
    sdkStartTimer(&timerGold);
    computeDCT8x8Gold1(ImgF1, ImgF2, StrideF, Size);
    sdkStopTimer(&timerGold);
  }

  // stop and destroy CUDA timer
  float TimerGoldSpan = sdkGetAverageTimerValue(&timerGold);
  sdkDeleteTimer(&timerGold);

  // perform quantization
  quantizeGoldFloat(ImgF2, StrideF, Size);

  // perform block-wise IDCT processing
  computeIDCT8x8Gold1(ImgF2, ImgF1, StrideF, Size);

  // convert image back to byte representation
  AddFloatPlane(128.0f, ImgF1, StrideF, Size);
  CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

  // free float buffers
  FreePlane(ImgF1);
  FreePlane(ImgF2);

  // return time taken by the operation
  return TimerGoldSpan;
}

/**
**************************************************************************
*  Wrapper function for 2nd gold version of DCT, quantization and IDCT
*implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperGold2(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size) {
  // allocate float buffers for DCT and other data
  int StrideF;
  float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
  float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

  // convert source image to float representation
  CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
  AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

  // create and start CUDA timer
  StopWatchInterface *timerGold = 0;
  sdkCreateTimer(&timerGold);
  sdkResetTimer(&timerGold);

  // perform block-wise DCT processing and benchmarking
  for (int i = 0; i < BENCHMARK_SIZE; i++) {
    sdkStartTimer(&timerGold);
    computeDCT8x8Gold2(ImgF1, ImgF2, StrideF, Size);
    sdkStopTimer(&timerGold);
  }

  // stop and destroy CUDA timer
  float TimerGoldSpan = sdkGetAverageTimerValue(&timerGold);
  sdkDeleteTimer(&timerGold);

  // perform quantization
  quantizeGoldFloat(ImgF2, StrideF, Size);

  // perform block-wise IDCT processing
  computeIDCT8x8Gold2(ImgF2, ImgF1, StrideF, Size);

  // convert image back to byte representation
  AddFloatPlane(128.0f, ImgF1, StrideF, Size);
  CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

  // free float buffers
  FreePlane(ImgF1);
  FreePlane(ImgF2);

  // return time taken by the operation
  return TimerGoldSpan;
}

/**
**************************************************************************
*  Wrapper function for 1st CUDA version of DCT, quantization and IDCT
*implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperCUDA1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size) {
  // prepare channel format descriptor for passing texture into kernels
  cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

  // allocate device memory
  cudaArray *Src;
  float *Dst;
  size_t DstStride;
  checkCudaErrors(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
  checkCudaErrors(cudaMallocPitch((void **)(&Dst), &DstStride,
                                  Size.width * sizeof(float), Size.height));
  DstStride /= sizeof(float);

  // convert source image to float representation
  int ImgSrcFStride;
  float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
  CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
  AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

  // copy from host memory to device
  checkCudaErrors(cudaMemcpy2DToArray(
      Src, 0, 0, ImgSrcF, ImgSrcFStride * sizeof(float),
      Size.width * sizeof(float), Size.height, cudaMemcpyHostToDevice));

  // setup execution parameters
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

  // create and start CUDA timer
  StopWatchInterface *timerCUDA = 0;
  sdkCreateTimer(&timerCUDA);
  sdkResetTimer(&timerCUDA);

  // execute DCT kernel and benchmark
  cudaTextureObject_t TexSrc;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = Src;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&TexSrc, &texRes, &texDescr, NULL));

  for (int i = 0; i < BENCHMARK_SIZE; i++) {
    sdkStartTimer(&timerCUDA);
    CUDAkernel1DCT<<<grid, threads>>>(Dst, (int)DstStride, 0, 0, TexSrc);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timerCUDA);
  }

  getLastCudaError("Kernel execution failed");

  // finalize CUDA timer
  float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);
  sdkDeleteTimer(&timerCUDA);

  // execute Quantization kernel
  CUDAkernelQuantizationFloat<<<grid, threads>>>(Dst, (int)DstStride);
  getLastCudaError("Kernel execution failed");

  // copy quantized coefficients from host memory to device array
  checkCudaErrors(cudaMemcpy2DToArray(Src, 0, 0, Dst, DstStride * sizeof(float),
                                      Size.width * sizeof(float), Size.height,
                                      cudaMemcpyDeviceToDevice));

  // execute IDCT kernel
  CUDAkernel1IDCT<<<grid, threads>>>(Dst, (int)DstStride, 0, 0, TexSrc);
  getLastCudaError("Kernel execution failed");

  // copy quantized image block to host
  checkCudaErrors(cudaMemcpy2D(
      ImgSrcF, ImgSrcFStride * sizeof(float), Dst, DstStride * sizeof(float),
      Size.width * sizeof(float), Size.height, cudaMemcpyDeviceToHost));

  // convert image back to byte representation
  AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
  CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

  // clean up memory
  checkCudaErrors(cudaDestroyTextureObject(TexSrc));
  checkCudaErrors(cudaFreeArray(Src));
  checkCudaErrors(cudaFree(Dst));
  FreePlane(ImgSrcF);

  // return time taken by the operation
  return TimerCUDASpan;
}

/**
**************************************************************************
*  Wrapper function for 2nd CUDA version of DCT, quantization and IDCT
*implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/

float WrapperCUDA2(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size) {
  // allocate host buffers for DCT and other data
  int StrideF;
  float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

  // convert source image to float representation
  CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
  AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

  // allocate device memory
  float *src, *dst;
  size_t DeviceStride;
  checkCudaErrors(cudaMallocPitch((void **)&src, &DeviceStride,
                                  Size.width * sizeof(float), Size.height));
  checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride,
                                  Size.width * sizeof(float), Size.height));
  DeviceStride /= sizeof(float);

  // copy from host memory to device
  checkCudaErrors(cudaMemcpy2D(
      src, DeviceStride * sizeof(float), ImgF1, StrideF * sizeof(float),
      Size.width * sizeof(float), Size.height, cudaMemcpyHostToDevice));

  // create and start CUDA timer
  StopWatchInterface *timerCUDA = 0;
  sdkCreateTimer(&timerCUDA);

  // setup execution parameters
  dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH,
                     Size.height / KER2_BLOCK_HEIGHT, 1);
  dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH / 8, KER2_BLOCK_HEIGHT / 8);

  // perform block-wise DCT processing and benchmarking
  const int numIterations = 100;

  for (int i = -1; i < numIterations; i++) {
    if (i == 0) {
      checkCudaErrors(cudaDeviceSynchronize());
      sdkResetTimer(&timerCUDA);
      sdkStartTimer(&timerCUDA);
    }

    CUDAkernel2DCT<<<GridFullWarps, ThreadsFullWarps>>>(dst, src,
                                                        (int)DeviceStride);
    getLastCudaError("Kernel execution failed");
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timerCUDA);

  // finalize timing of CUDA Kernels
  float avgTime = (float)sdkGetTimerValue(&timerCUDA) / (float)numIterations;
  sdkDeleteTimer(&timerCUDA);
  printf("%f MPix/s //%f ms\n",
         (1E-6 * (float)Size.width * (float)Size.height) / (1E-3 * avgTime),
         avgTime);

  // setup execution parameters for quantization
  dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

  // execute Quantization kernel
  CUDAkernelQuantizationFloat<<<GridSmallBlocks, ThreadsSmallBlocks>>>(
      dst, (int)DeviceStride);
  getLastCudaError("Kernel execution failed");

  // perform block-wise IDCT processing
  CUDAkernel2IDCT<<<GridFullWarps, ThreadsFullWarps>>>(src, dst,
                                                       (int)DeviceStride);
  checkCudaErrors(cudaDeviceSynchronize());
  getLastCudaError("Kernel execution failed");

  // copy quantized image block to host
  checkCudaErrors(cudaMemcpy2D(
      ImgF1, StrideF * sizeof(float), src, DeviceStride * sizeof(float),
      Size.width * sizeof(float), Size.height, cudaMemcpyDeviceToHost));

  // convert image back to byte representation
  AddFloatPlane(128.0f, ImgF1, StrideF, Size);
  CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

  // clean up memory
  checkCudaErrors(cudaFree(dst));
  checkCudaErrors(cudaFree(src));
  FreePlane(ImgF1);

  // return time taken by the operation
  return avgTime;
}

/**
**************************************************************************
*  Wrapper function for short CUDA version of DCT, quantization and IDCT
*implementations
*
* \param ImgSrc         [IN] - Source byte image plane
* \param ImgDst         [IN] - Quantized result byte image plane
* \param Stride         [IN] - Stride for both source and result planes
* \param Size           [IN] - Size of both planes
*
* \return Execution time in milliseconds
*/
float WrapperCUDAshort(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size) {
  // allocate host buffers for DCT and other data
  int StrideS;
  short *ImgS1 = MallocPlaneShort(Size.width, Size.height, &StrideS);

  // convert source image to short representation centered at 128
  for (int i = 0; i < Size.height; i++) {
    for (int j = 0; j < Size.width; j++) {
      ImgS1[i * StrideS + j] = (short)ImgSrc[i * Stride + j] - 128;
    }
  }

  // allocate device memory
  short *SrcDst;
  size_t DeviceStride;
  checkCudaErrors(cudaMallocPitch((void **)(&SrcDst), &DeviceStride,
                                  Size.width * sizeof(short), Size.height));
  DeviceStride /= sizeof(short);

  // copy from host memory to device
  checkCudaErrors(cudaMemcpy2D(
      SrcDst, DeviceStride * sizeof(short), ImgS1, StrideS * sizeof(short),
      Size.width * sizeof(short), Size.height, cudaMemcpyHostToDevice));

  // create and start CUDA timer
  StopWatchInterface *timerLibJpeg = 0;
  sdkCreateTimer(&timerLibJpeg);
  sdkResetTimer(&timerLibJpeg);

  // setup execution parameters
  dim3 GridShort(Size.width / KERS_BLOCK_WIDTH, Size.height / KERS_BLOCK_HEIGHT,
                 1);
  dim3 ThreadsShort(8, KERS_BLOCK_WIDTH / 8, KERS_BLOCK_HEIGHT / 8);

  // perform block-wise DCT processing and benchmarking
  sdkStartTimer(&timerLibJpeg);
  CUDAkernelShortDCT<<<GridShort, ThreadsShort>>>(SrcDst, (int)DeviceStride);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timerLibJpeg);
  getLastCudaError("Kernel execution failed");

  // stop and destroy CUDA timer
  float TimerLibJpegSpan16b = sdkGetAverageTimerValue(&timerLibJpeg);
  sdkDeleteTimer(&timerLibJpeg);

  // setup execution parameters for quantization
  dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

  // execute Quantization kernel
  CUDAkernelQuantizationShort<<<GridSmallBlocks, ThreadsSmallBlocks>>>(
      SrcDst, (int)DeviceStride);
  getLastCudaError("Kernel execution failed");

  // perform block-wise IDCT processing
  CUDAkernelShortIDCT<<<GridShort, ThreadsShort>>>(SrcDst, (int)DeviceStride);
  checkCudaErrors(cudaDeviceSynchronize());
  getLastCudaError("Kernel execution failed");

  // copy quantized image block to host
  checkCudaErrors(cudaMemcpy2D(
      ImgS1, StrideS * sizeof(short), SrcDst, DeviceStride * sizeof(short),
      Size.width * sizeof(short), Size.height, cudaMemcpyDeviceToHost));

  // convert image back to byte representation
  for (int i = 0; i < Size.height; i++) {
    for (int j = 0; j < Size.width; j++) {
      ImgDst[i * Stride + j] = clamp_0_255(ImgS1[i * StrideS + j] + 128);
    }
  }

  // free float buffers
  checkCudaErrors(cudaFree(SrcDst));
  FreePlane(ImgS1);

  // return time taken by the operation
  return TimerLibJpegSpan16b;
}

/**
**************************************************************************
*  Program entry point
*
* \param argc       [IN] - Number of command-line arguments
* \param argv       [IN] - Array of command-line arguments
*
* \return Status code
*/

int main(int argc, char **argv) {
  //
  // Sample initialization
  //
  printf("%s Starting...\n\n", argv[0]);

  // initialize CUDA
  findCudaDevice(argc, (const char **)argv);

  // source and results image filenames
  char SampleImageFname[] = "teapot512.bmp";
  char SampleImageFnameResGold1[] = "teapot512_gold1.bmp";
  char SampleImageFnameResGold2[] = "teapot512_gold2.bmp";
  char SampleImageFnameResCUDA1[] = "teapot512_cuda1.bmp";
  char SampleImageFnameResCUDA2[] = "teapot512_cuda2.bmp";
  char SampleImageFnameResCUDAshort[] = "teapot512_cuda_short.bmp";

  char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

  if (pSampleImageFpath == NULL) {
    printf("dct8x8 could not locate Sample Image <%s>\nExiting...\n",
           pSampleImageFpath);
    exit(EXIT_FAILURE);
  }

  // preload image (acquire dimensions)
  int ImgWidth, ImgHeight;
  ROI ImgSize;
  int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
  ImgSize.width = ImgWidth;
  ImgSize.height = ImgHeight;

  // CONSOLE INFORMATION: saying hello to user
  printf("CUDA sample DCT/IDCT implementation\n");
  printf("===================================\n");
  printf("Loading test image: %s... ", SampleImageFname);

  if (res) {
    printf("\nError: Image file not found or invalid!\n");
    exit(EXIT_FAILURE);
    return 1;
  }

  // check image dimensions are multiples of BLOCK_SIZE
  if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0) {
    printf("\nError: Input image dimensions must be multiples of 8!\n");
    exit(EXIT_FAILURE);
    return 1;
  }

  printf("[%d x %d]... ", ImgWidth, ImgHeight);

  // allocate image buffers
  int ImgStride;
  byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
  byte *ImgDstGold1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
  byte *ImgDstGold2 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
  byte *ImgDstCUDA1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
  byte *ImgDstCUDA2 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
  byte *ImgDstCUDAshort = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

  // load sample image
  LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);

  //
  // RUNNING WRAPPERS
  //

  // compute Gold 1 version of DCT/quantization/IDCT
  printf("Success\nRunning Gold 1 (CPU) version... ");
  float TimeGold1 = WrapperGold1(ImgSrc, ImgDstGold1, ImgStride, ImgSize);

  // compute Gold 2 version of DCT/quantization/IDCT
  printf("Success\nRunning Gold 2 (CPU) version... ");
  float TimeGold2 = WrapperGold2(ImgSrc, ImgDstGold2, ImgStride, ImgSize);

  // compute CUDA 1 version of DCT/quantization/IDCT
  printf("Success\nRunning CUDA 1 (GPU) version... ");
  float TimeCUDA1 = WrapperCUDA1(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);

  // compute CUDA 2 version of DCT/quantization/IDCT
  printf("Success\nRunning CUDA 2 (GPU) version... ");
  float TimeCUDA2 = WrapperCUDA2(ImgSrc, ImgDstCUDA2, ImgStride, ImgSize);

  // compute CUDA short version of DCT/quantization/IDCT
  printf("Success\nRunning CUDA short (GPU) version... ");
  float TimeCUDAshort =
      WrapperCUDAshort(ImgSrc, ImgDstCUDAshort, ImgStride, ImgSize);
  //
  // Execution statistics, result saving and validation
  //

  // dump result of Gold 1 processing
  printf("Success\nDumping result to %s... ", SampleImageFnameResGold1);
  DumpBmpAsGray(SampleImageFnameResGold1, ImgDstGold1, ImgStride, ImgSize);

  // dump result of Gold 2 processing
  printf("Success\nDumping result to %s... ", SampleImageFnameResGold2);
  DumpBmpAsGray(SampleImageFnameResGold2, ImgDstGold2, ImgStride, ImgSize);

  // dump result of CUDA 1 processing
  printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA1);
  DumpBmpAsGray(SampleImageFnameResCUDA1, ImgDstCUDA1, ImgStride, ImgSize);

  // dump result of CUDA 2 processing
  printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA2);
  DumpBmpAsGray(SampleImageFnameResCUDA2, ImgDstCUDA2, ImgStride, ImgSize);

  // dump result of CUDA short processing
  printf("Success\nDumping result to %s... ", SampleImageFnameResCUDAshort);
  DumpBmpAsGray(SampleImageFnameResCUDAshort, ImgDstCUDAshort, ImgStride,
                ImgSize);
  // print speed info
  printf("Success\n");

  printf("Processing time (CUDA 1)    : %f ms \n", TimeCUDA1);
  printf("Processing time (CUDA 2)    : %f ms \n", TimeCUDA2);
  printf("Processing time (CUDA short): %f ms \n", TimeCUDAshort);

  // calculate PSNR between each pair of images
  float PSNR_Src_DstGold1 =
      CalculatePSNR(ImgSrc, ImgDstGold1, ImgStride, ImgSize);
  float PSNR_Src_DstGold2 =
      CalculatePSNR(ImgSrc, ImgDstGold2, ImgStride, ImgSize);
  float PSNR_Src_DstCUDA1 =
      CalculatePSNR(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);
  float PSNR_Src_DstCUDA2 =
      CalculatePSNR(ImgSrc, ImgDstCUDA2, ImgStride, ImgSize);
  float PSNR_Src_DstCUDAshort =
      CalculatePSNR(ImgSrc, ImgDstCUDAshort, ImgStride, ImgSize);
  float PSNR_DstGold1_DstCUDA1 =
      CalculatePSNR(ImgDstGold1, ImgDstCUDA1, ImgStride, ImgSize);
  float PSNR_DstGold2_DstCUDA2 =
      CalculatePSNR(ImgDstGold2, ImgDstCUDA2, ImgStride, ImgSize);
  float PSNR_DstGold2_DstCUDA16b =
      CalculatePSNR(ImgDstGold2, ImgDstCUDAshort, ImgStride, ImgSize);

  printf("PSNR Original    <---> CPU(Gold 1)    : %f\n", PSNR_Src_DstGold1);
  printf("PSNR Original    <---> CPU(Gold 2)    : %f\n", PSNR_Src_DstGold2);
  printf("PSNR Original    <---> GPU(CUDA 1)    : %f\n", PSNR_Src_DstCUDA1);
  printf("PSNR Original    <---> GPU(CUDA 2)    : %f\n", PSNR_Src_DstCUDA2);
  printf("PSNR Original    <---> GPU(CUDA short): %f\n", PSNR_Src_DstCUDAshort);
  printf("PSNR CPU(Gold 1) <---> GPU(CUDA 1)    : %f\n",
         PSNR_DstGold1_DstCUDA1);
  printf("PSNR CPU(Gold 2) <---> GPU(CUDA 2)    : %f\n",
         PSNR_DstGold2_DstCUDA2);
  printf("PSNR CPU(Gold 2) <---> GPU(CUDA short): %f\n",
         PSNR_DstGold2_DstCUDA16b);

  bool bTestResult = (PSNR_DstGold1_DstCUDA1 > PSNR_THRESHOLD_EQUAL &&
                      PSNR_DstGold2_DstCUDA2 > PSNR_THRESHOLD_EQUAL &&
                      PSNR_DstGold2_DstCUDA16b > PSNR_THRESHOLD_EQUAL);

  //
  // Finalization
  //

  // release byte planes
  FreePlane(ImgSrc);
  FreePlane(ImgDstGold1);
  FreePlane(ImgDstGold2);
  FreePlane(ImgDstCUDA1);
  FreePlane(ImgDstCUDA2);
  FreePlane(ImgDstCUDAshort);

  // finalize
  printf("\nTest Summary...\n");

  if (!bTestResult) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
