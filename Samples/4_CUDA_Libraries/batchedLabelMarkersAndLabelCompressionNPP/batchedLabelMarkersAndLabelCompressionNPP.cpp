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

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <stdio.h>
#include <string.h>
#include <fstream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>

// Note:  If you want to view these images we HIGHLY recommend using imagej
//        which is free on the internet and works on most platforms
//        because it is one of the few image viewing apps that can display 32
//        bit integer image data.  While it normalizes the data to floating
//        point values for viewing it still provides a good representation of
//        the relative brightness of each label value. Note that label
//        compression output results in smaller differences between label values
//        making it visually more difficult to detect differences in labeled
//        regions.  If you have an editor that can display hex values you can
//        see what the exact values of each label is, every 4 bytes represents 1
//        32 bit integer label value.
//
//        The files read and written by this sample app use RAW image format,
//        that is, only the image data itself exists in the files with no image
//        format information.   When viewing RAW files with imagej just enter
//        the image size and bit depth values that are part of the file name
//        when requested by imagej.
//
//        This sample app works in 2 stages, first it processes all of the
//        images individually then it processes them all again in 1 batch using
//        the Batch_Advanced versions of the NPP batch functions which allow
//        each image to have it's own ROI.  The 2 stages are completely
//        separable but in this sample the second stage takes advantage of some
//        of the data that has already been initialized.
//
//        Note that there is a small amount of variability in the number of
//        unique label markers generated from one run to the next by the UF
//        algorithm.
//
//        Performance of ALL NPP image batch functions is limited by the maximum
//        ROI height in the list of images.

// Batched label compression support is only available on NPP versions > 11.0,
// comment out if using NPP 11.0
#define USE_BATCHED_LABEL_COMPRESSION 1

#define NUMBER_OF_IMAGES 5

Npp8u *pInputImageDev[NUMBER_OF_IMAGES];
Npp8u *pInputImageHost[NUMBER_OF_IMAGES];
Npp8u *pUFGenerateLabelsScratchBufferDev[NUMBER_OF_IMAGES];
Npp8u *pUFCompressedLabelsScratchBufferDev[NUMBER_OF_IMAGES];
Npp32u *pUFLabelDev[NUMBER_OF_IMAGES];
Npp32u *pUFLabelHost[NUMBER_OF_IMAGES];
NppiImageDescriptor *pUFBatchSrcImageListDev = 0;
NppiImageDescriptor *pUFBatchSrcDstImageListDev = 0;
NppiImageDescriptor *pUFBatchSrcImageListHost = 0;
NppiImageDescriptor *pUFBatchSrcDstImageListHost = 0;
NppiBufferDescriptor *pUFBatchSrcDstScratchBufferListDev =
    0;  // from nppi_filtering_functions.h
NppiBufferDescriptor *pUFBatchSrcDstScratchBufferListHost = 0;
Npp32u *pUFBatchPerImageCompressedCountListDev = 0;
Npp32u *pUFBatchPerImageCompressedCountListHost = 0;

void tearDown()  // Clean up and tear down
{
  if (pUFBatchPerImageCompressedCountListDev != 0)
    cudaFree(pUFBatchPerImageCompressedCountListDev);
  if (pUFBatchSrcDstScratchBufferListDev != 0)
    cudaFree(pUFBatchSrcDstScratchBufferListDev);
  if (pUFBatchSrcDstImageListDev != 0) cudaFree(pUFBatchSrcDstImageListDev);
  if (pUFBatchSrcImageListDev != 0) cudaFree(pUFBatchSrcImageListDev);
  if (pUFBatchPerImageCompressedCountListHost != 0)
    cudaFreeHost(pUFBatchPerImageCompressedCountListHost);
  if (pUFBatchSrcDstScratchBufferListHost != 0)
    cudaFreeHost(pUFBatchSrcDstScratchBufferListHost);
  if (pUFBatchSrcDstImageListHost != 0)
    cudaFreeHost(pUFBatchSrcDstImageListHost);
  if (pUFBatchSrcImageListHost != 0) cudaFreeHost(pUFBatchSrcImageListHost);

  for (int j = 0; j < NUMBER_OF_IMAGES; j++) {
    if (pUFCompressedLabelsScratchBufferDev[j] != 0)
      cudaFree(pUFCompressedLabelsScratchBufferDev[j]);
    if (pUFGenerateLabelsScratchBufferDev[j] != 0)
      cudaFree(pUFGenerateLabelsScratchBufferDev[j]);
    if (pUFLabelDev[j] != 0) cudaFree(pUFLabelDev[j]);
    if (pInputImageDev[j] != 0) cudaFree(pInputImageDev[j]);
    if (pUFLabelHost[j] != 0) cudaFreeHost(pUFLabelHost[j]);
    if (pInputImageHost[j] != 0) cudaFreeHost(pInputImageHost[j]);
  }
}

const std::string &LabelMarkersOutputFile0 =
    "teapot_LabelMarkersUF_8Way_512x512_32u.raw";
const std::string &LabelMarkersOutputFile1 =
    "CT_skull_LabelMarkersUF_8Way_512x512_32u.raw";
const std::string &LabelMarkersOutputFile2 =
    "PCB_METAL_LabelMarkersUF_8Way_509x335_32u.raw";
const std::string &LabelMarkersOutputFile3 =
    "PCB2_LabelMarkersUF_8Way_1024x683_32u.raw";
const std::string &LabelMarkersOutputFile4 =
    "PCB_LabelMarkersUF_8Way_1280x720_32u.raw";

const std::string &CompressedMarkerLabelsOutputFile0 =
    "teapot_CompressedMarkerLabelsUF_8Way_512x512_32u.raw";
const std::string &CompressedMarkerLabelsOutputFile1 =
    "CT_skull_CompressedMarkerLabelsUF_8Way_512x512_32u.raw";
const std::string &CompressedMarkerLabelsOutputFile2 =
    "PCB_METAL_CompressedMarkerLabelsUF_8Way_509x335_32u.raw";
const std::string &CompressedMarkerLabelsOutputFile3 =
    "PCB2_CompressedMarkerLabelsUF_8Way_1024x683_32u.raw";
const std::string &CompressedMarkerLabelsOutputFile4 =
    "PCB_CompressedMarkerLabelsUF_8Way_1280x720_32u.raw";

const std::string &LabelMarkersBatchOutputFile0 =
    "teapot_LabelMarkersUFBatch_8Way_512x512_32u.raw";
const std::string &LabelMarkersBatchOutputFile1 =
    "CT_skull_LabelMarkersUFBatch_8Way_512x512_32u.raw";
const std::string &LabelMarkersBatchOutputFile2 =
    "PCB_METAL_LabelMarkersUFBatch_8Way_509x335_32u.raw";
const std::string &LabelMarkersBatchOutputFile3 =
    "PCB2_LabelMarkersUFBatch_8Way_1024x683_32u.raw";
const std::string &LabelMarkersBatchOutputFile4 =
    "PCB_LabelMarkersUFBatch_8Way_1280x720_32u.raw";

const std::string &CompressedMarkerLabelsBatchOutputFile0 =
    "teapot_CompressedMarkerLabelsUFBatch_8Way_512x512_32u.raw";
const std::string &CompressedMarkerLabelsBatchOutputFile1 =
    "CT_skull_CompressedMarkerLabelsUFBatch_8Way_512x512_32u.raw";
const std::string &CompressedMarkerLabelsBatchOutputFile2 =
    "PCB_METAL_CompressedMarkerLabelsUFBatch_8Way_509x335_32u.raw";
const std::string &CompressedMarkerLabelsBatchOutputFile3 =
    "PCB2_CompressedMarkerLabelsUFBatch_8Way_1024x683_32u.raw";
const std::string &CompressedMarkerLabelsBatchOutputFile4 =
    "PCB_CompressedMarkerLabelsUFBatch_8Way_1280x720_32u.raw";

int loadRaw8BitImage(Npp8u *pImage, int nWidth, int nHeight, int nImage) {
  FILE *bmpFile;
  size_t nSize;

  if (nImage == 0) {
    if (nWidth != 512 || nHeight != 512) return -1;
    const char *fileName = "teapot_512x512_8u.raw";
    const char *InputFile = sdkFindFilePath(fileName, ".");
    if (InputFile == NULL) {
      printf("%s file not found.. exiting\n", fileName);
      exit(EXIT_WAIVED);
    }

    FOPEN(bmpFile, InputFile, "rb");
  } else if (nImage == 1) {
    if (nWidth != 512 || nHeight != 512) return -1;
    const char *fileName = "CT_skull_512x512_8u.raw";
    const char *InputFile = sdkFindFilePath(fileName, ".");
    if (InputFile == NULL) {
      printf("%s file not found.. exiting\n", fileName);
      exit(EXIT_WAIVED);
    }

    FOPEN(bmpFile, InputFile, "rb");
  } else if (nImage == 2) {
    if (nWidth != 509 || nHeight != 335) return -1;
    const char *fileName = "PCB_METAL_509x335_8u.raw";
    const char *InputFile = sdkFindFilePath(fileName, ".");
    if (InputFile == NULL) {
      printf("%s file not found.. exiting\n", fileName);
      exit(EXIT_WAIVED);
    }

    FOPEN(bmpFile, InputFile, "rb");
  } else if (nImage == 3) {
    if (nWidth != 1024 || nHeight != 683) return -1;
    const char *fileName = "PCB2_1024x683_8u.raw";
    const char *InputFile = sdkFindFilePath(fileName, ".");
    if (InputFile == NULL) {
      printf("%s file not found.. exiting\n", fileName);
      exit(EXIT_WAIVED);
    }

    FOPEN(bmpFile, InputFile, "rb");
  } else if (nImage == 4) {
    if (nWidth != 1280 || nHeight != 720) return -1;
    const char *fileName = "PCB_1280x720_8u.raw";
    const char *InputFile = sdkFindFilePath(fileName, ".");
    if (InputFile == NULL) {
      printf("%s file not found.. exiting\n", fileName);
      exit(EXIT_WAIVED);
    }

    FOPEN(bmpFile, InputFile, "rb");
  } else {
    printf("Input file load failed.\n");
    return -1;
  }

  if (bmpFile == NULL) return -1;
  nSize = fread(pImage, 1, nWidth * nHeight, bmpFile);
  if (nSize < nWidth * nHeight) {
    fclose(bmpFile);
    return -1;
  }
  fclose(bmpFile);

  printf("Input file load succeeded.\n");

  return 0;
}

int main(int argc, char **argv) {
  int aGenerateLabelsScratchBufferSize[NUMBER_OF_IMAGES];
  int aCompressLabelsScratchBufferSize[NUMBER_OF_IMAGES];

  int nCompressedLabelCount = 0;
  cudaError_t cudaError;
  NppStatus nppStatus;
  NppStreamContext nppStreamCtx;
  FILE *bmpFile;

  for (int j = 0; j < NUMBER_OF_IMAGES; j++) {
    pInputImageDev[j] = 0;
    pInputImageHost[j] = 0;
    pUFGenerateLabelsScratchBufferDev[j] = 0;
    pUFCompressedLabelsScratchBufferDev[j] = 0;
    pUFLabelDev[j] = 0;
    pUFLabelHost[j] = 0;
  }

  nppStreamCtx.hStream = 0;  // The NULL stream by default, set this to whatever
                             // your stream ID is if not the NULL stream.

  cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
  if (cudaError != cudaSuccess) {
    printf("CUDA error: no devices supporting CUDA.\n");
    return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
  }

  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("CUDA Runtime Version: %d.%d\n\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  cudaError = cudaDeviceGetAttribute(
      &nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
      cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId);
  if (cudaError != cudaSuccess) return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

  cudaError = cudaDeviceGetAttribute(
      &nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
      cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId);
  if (cudaError != cudaSuccess) return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

  cudaError =
      cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);

  cudaDeviceProp oDeviceProperties;

  cudaError =
      cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId);

  nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor =
      oDeviceProperties.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

  NppiSize oSizeROI[NUMBER_OF_IMAGES];

  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    if (nImage == 0) {
      oSizeROI[nImage].width = 512;
      oSizeROI[nImage].height = 512;
    } else if (nImage == 1) {
      oSizeROI[nImage].width = 512;
      oSizeROI[nImage].height = 512;
    } else if (nImage == 2) {
      oSizeROI[nImage].width = 509;
      oSizeROI[nImage].height = 335;
    } else if (nImage == 3) {
      oSizeROI[nImage].width = 1024;
      oSizeROI[nImage].height = 683;
    } else if (nImage == 4) {
      oSizeROI[nImage].width = 1280;
      oSizeROI[nImage].height = 720;
    }

    // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can
    // significantly improve the performance of many NPP functions, for UF
    // function label markers generation or compression DO NOT USE
    // cudaMallocPitch().  Doing so could result in incorrect output.

    cudaError = cudaMalloc(
        (void **)&pInputImageDev[nImage],
        oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height);
    if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

    // For images processed with UF label markers functions ROI width and height
    // for label markers generation output AND marker compression functions MUST
    // be the same AND line pitch MUST be equal to ROI.width * sizeof(Npp32u).
    // Also the image pointer used for label markers generation output must
    // start at the same position in the image as it does in the marker
    // compression function.  Also note that actual input image size and ROI do
    // not necessarily need to be related other than ROI being less than or
    // equal to image size and image starting position does not necessarily have
    // to be at pixel 0 in the input image.

    cudaError = cudaMalloc(
        (void **)&pUFLabelDev[nImage],
        oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height);
    if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

    checkCudaErrors(cudaMallocHost(
        &(pInputImageHost[nImage]),
        oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height));
    checkCudaErrors(cudaMallocHost(
        &(pUFLabelHost[nImage]),
        oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height));

    // Use UF functions throughout this sample.

    nppStatus = nppiLabelMarkersUFGetBufferSize_32u_C1R(
        oSizeROI[nImage], &aGenerateLabelsScratchBufferSize[nImage]);

    // One at a time image processing

    cudaError = cudaMalloc((void **)&pUFGenerateLabelsScratchBufferDev[nImage],
                           aGenerateLabelsScratchBufferSize[nImage]);
    if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

    if (loadRaw8BitImage(pInputImageHost[nImage],
                         oSizeROI[nImage].width * sizeof(Npp8u),
                         oSizeROI[nImage].height, nImage) == 0) {
      cudaError = cudaMemcpy2DAsync(
          pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
          pInputImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

      nppStatus = nppiLabelMarkersUF_8u32u_C1R_Ctx(
          pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u),
          pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
          oSizeROI[nImage], nppiNormInf,
          pUFGenerateLabelsScratchBufferDev[nImage], nppStreamCtx);

      if (nppStatus != NPP_SUCCESS) {
        if (nImage == 0)
          printf("teapot_LabelMarkersUF_8Way_512x512_32u failed.\n");
        else if (nImage == 1)
          printf("CT_skull_LabelMarkersUF_8Way_512x512_32u failed.\n");
        else if (nImage == 2)
          printf("PCB_METAL_LabelMarkersUF_8Way_509x335_32u failed.\n");
        else if (nImage == 3)
          printf("PCB2_LabelMarkersUF_8Way_1024x683_32u failed.\n");
        else if (nImage == 4)
          printf("PCB_LabelMarkersUF_8Way_1280x720_32u failed.\n");
        tearDown();
        return -1;
      }

      cudaError = cudaMemcpy2DAsync(
          pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
          pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
          oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

      // Wait host image read backs to complete, not necessary if no need to
      // synchronize
      if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) !=
          cudaSuccess) {
        printf("Post label generation cudaStreamSynchronize failed\n");
        tearDown();
        return -1;
      }

      if (nImage == 0)
        FOPEN(bmpFile, LabelMarkersOutputFile0.c_str(), "wb");
      else if (nImage == 1)
        FOPEN(bmpFile, LabelMarkersOutputFile1.c_str(), "wb");
      else if (nImage == 2)
        FOPEN(bmpFile, LabelMarkersOutputFile2.c_str(), "wb");
      else if (nImage == 3)
        FOPEN(bmpFile, LabelMarkersOutputFile3.c_str(), "wb");
      else if (nImage == 4)
        FOPEN(bmpFile, LabelMarkersOutputFile4.c_str(), "wb");

      if (bmpFile == NULL) return -1;
      size_t nSize = 0;
      for (int j = 0; j < oSizeROI[nImage].height; j++) {
        nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width],
                        sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
      }
      fclose(bmpFile);

      nppStatus = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
          oSizeROI[nImage].width * oSizeROI[nImage].height,
          &aCompressLabelsScratchBufferSize[nImage]);
      if (nppStatus != NPP_NO_ERROR) return nppStatus;

      cudaError =
          cudaMalloc((void **)&pUFCompressedLabelsScratchBufferDev[nImage],
                     aCompressLabelsScratchBufferSize[nImage]);
      if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

      nCompressedLabelCount = 0;

      nppStatus = nppiCompressMarkerLabelsUF_32u_C1IR(
          pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
          oSizeROI[nImage], oSizeROI[nImage].width * oSizeROI[nImage].height,
          &nCompressedLabelCount, pUFCompressedLabelsScratchBufferDev[nImage]);

      if (nppStatus != NPP_SUCCESS) {
        if (nImage == 0)
          printf("teapot_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
        else if (nImage == 1)
          printf(
              "CT_Skull_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
        else if (nImage == 2)
          printf(
              "PCB_METAL_CompressedLabelMarkersUF_8Way_509x335_32u failed.\n");
        else if (nImage == 3)
          printf("PCB2_CompressedLabelMarkersUF_8Way_1024x683_32u failed.\n");
        else if (nImage == 4)
          printf("PCB_CompressedLabelMarkersUF_8Way_1280x720_32u failed.\n");
        tearDown();
        return -1;
      }

      cudaError = cudaMemcpy2DAsync(
          pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
          pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
          oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

      // Wait for host image read backs to finish, not necessary if no need to
      // synchronize
      if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) !=
              cudaSuccess ||
          nCompressedLabelCount == 0) {
        printf("Post label compression cudaStreamSynchronize failed\n");
        tearDown();
        return -1;
      }

      if (nImage == 0)
        FOPEN(bmpFile, CompressedMarkerLabelsOutputFile0.c_str(), "wb");
      else if (nImage == 1)
        FOPEN(bmpFile, CompressedMarkerLabelsOutputFile1.c_str(), "wb");
      else if (nImage == 2)
        FOPEN(bmpFile, CompressedMarkerLabelsOutputFile2.c_str(), "wb");
      else if (nImage == 3)
        FOPEN(bmpFile, CompressedMarkerLabelsOutputFile3.c_str(), "wb");
      else if (nImage == 4)
        FOPEN(bmpFile, CompressedMarkerLabelsOutputFile4.c_str(), "wb");

      if (bmpFile == NULL) return -1;
      nSize = 0;
      for (int j = 0; j < oSizeROI[nImage].height; j++) {
        nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width],
                        sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
      }
      fclose(bmpFile);

      if (nImage == 0)
        printf(
            "teapot_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, "
            "compressed label count is %d.\n",
            nCompressedLabelCount);
      else if (nImage == 1)
        printf(
            "CT_Skull_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, "
            "compressed label count is %d.\n",
            nCompressedLabelCount);
      else if (nImage == 2)
        printf(
            "PCB_METAL_CompressedMarkerLabelsUF_8Way_509x335_32u succeeded, "
            "compressed label count is %d.\n",
            nCompressedLabelCount);
      else if (nImage == 3)
        printf(
            "PCB2_CompressedMarkerLabelsUF_8Way_1024x683_32u succeeded, "
            "compressed label count is %d.\n",
            nCompressedLabelCount);
      else if (nImage == 4)
        printf(
            "PCB_CompressedMarkerLabelsUF_8Way_1280x720_32u succeeded, "
            "compressed label count is %d.\n",
            nCompressedLabelCount);
    }
  }

  // Batch image processing

  // We want to allocate scratch buffers more efficiently for batch processing
  // so first we free up the scratch buffers for image 0 and reallocate them.
  // This is not required but helps cudaMalloc to work more efficiently.

  cudaFree(pUFCompressedLabelsScratchBufferDev[0]);

  int nTotalBatchedUFCompressLabelsScratchBufferDevSize = 0;

  for (int k = 0; k < NUMBER_OF_IMAGES; k++)
    nTotalBatchedUFCompressLabelsScratchBufferDevSize +=
        aCompressLabelsScratchBufferSize[k];

  cudaError = cudaMalloc((void **)&pUFCompressedLabelsScratchBufferDev[0],
                         nTotalBatchedUFCompressLabelsScratchBufferDevSize);
  if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

  // Now allocate batch lists

  int nBatchImageListBytes = NUMBER_OF_IMAGES * sizeof(NppiImageDescriptor);

  cudaError =
      cudaMalloc((void **)&pUFBatchSrcImageListDev, nBatchImageListBytes);
  if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

  cudaError =
      cudaMalloc((void **)&pUFBatchSrcDstImageListDev, nBatchImageListBytes);
  if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

  checkCudaErrors(
      cudaMallocHost((void **)&pUFBatchSrcImageListHost, nBatchImageListBytes));

  checkCudaErrors(cudaMallocHost((void **)&pUFBatchSrcDstImageListHost,
                                 nBatchImageListBytes));

  NppiSize oMaxROISize = {0, 0};

  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    pUFBatchSrcImageListHost[nImage].pData = pInputImageDev[nImage];
    pUFBatchSrcImageListHost[nImage].nStep =
        oSizeROI[nImage].width * sizeof(Npp8u);
    // src image oSize parameter is ignored in these NPP functions
    pUFBatchSrcDstImageListHost[nImage].pData = pUFLabelDev[nImage];
    pUFBatchSrcDstImageListHost[nImage].nStep =
        oSizeROI[nImage].width * sizeof(Npp32u);
    pUFBatchSrcDstImageListHost[nImage].oSize = oSizeROI[nImage];
    if (oSizeROI[nImage].width > oMaxROISize.width)
      oMaxROISize.width = oSizeROI[nImage].width;
    if (oSizeROI[nImage].height > oMaxROISize.height)
      oMaxROISize.height = oSizeROI[nImage].height;
  }

  // Copy label generation batch lists from CPU to GPU
  cudaError = cudaMemcpyAsync(pUFBatchSrcImageListDev, pUFBatchSrcImageListHost,
                              nBatchImageListBytes, cudaMemcpyHostToDevice,
                              nppStreamCtx.hStream);
  if (cudaError != cudaSuccess) return NPP_MEMCPY_ERROR;

  cudaError = cudaMemcpyAsync(pUFBatchSrcDstImageListDev,
                              pUFBatchSrcDstImageListHost, nBatchImageListBytes,
                              cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaError != cudaSuccess) return NPP_MEMCPY_ERROR;

  // We use 8-way neighbor search throughout this example
  nppStatus = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(
      pUFBatchSrcImageListDev, pUFBatchSrcDstImageListDev, NUMBER_OF_IMAGES,
      oMaxROISize, nppiNormInf, nppStreamCtx);

  if (nppStatus != NPP_SUCCESS) {
    printf("LabelMarkersUFBatch_8Way_8u32u failed.\n");
    tearDown();
    return -1;
  }

  // Now read back generated device images to the host

  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    cudaError = cudaMemcpy2DAsync(
        pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
        pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
        oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
        cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
  }

  // Wait for host image read backs to complete, not necessary if no need to
  // synchronize
  if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) !=
      cudaSuccess) {
    printf("Post label generation cudaStreamSynchronize failed\n");
    tearDown();
    return -1;
  }

  // Save output to files
  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    if (nImage == 0)
      FOPEN(bmpFile, LabelMarkersBatchOutputFile0.c_str(), "wb");
    else if (nImage == 1)
      FOPEN(bmpFile, LabelMarkersBatchOutputFile1.c_str(), "wb");
    else if (nImage == 2)
      FOPEN(bmpFile, LabelMarkersBatchOutputFile2.c_str(), "wb");
    else if (nImage == 3)
      FOPEN(bmpFile, LabelMarkersBatchOutputFile3.c_str(), "wb");
    else if (nImage == 4)
      FOPEN(bmpFile, LabelMarkersBatchOutputFile4.c_str(), "wb");

    if (bmpFile == NULL) return -1;
    size_t nSize = 0;
    for (int j = 0; j < oSizeROI[nImage].height; j++) {
      nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width],
                      sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
    }
    fclose(bmpFile);
  }

#ifdef USE_BATCHED_LABEL_COMPRESSION

  // Now allocate scratch buffer memory for batched label compression
  cudaError = cudaMalloc((void **)&pUFBatchSrcDstScratchBufferListDev,
                         NUMBER_OF_IMAGES * sizeof(NppiBufferDescriptor));
  if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

  cudaError = cudaMalloc((void **)&pUFBatchPerImageCompressedCountListDev,
                         NUMBER_OF_IMAGES * sizeof(Npp32u));
  if (cudaError != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;

  // Allocate host side scratch buffer point and size list and initialize with
  // device scratch buffer pointers
  checkCudaErrors(
      cudaMallocHost((void **)&pUFBatchSrcDstScratchBufferListHost,
                     NUMBER_OF_IMAGES * sizeof(NppiBufferDescriptor)));

  checkCudaErrors(
      cudaMallocHost((void **)&pUFBatchPerImageCompressedCountListHost,
                     +NUMBER_OF_IMAGES * sizeof(Npp32u)));

  // Start buffer pointer at beginning of full per image buffer list sized
  // pUFCompressedLabelsScratchBufferDev[0]
  Npp32u *pCurUFCompressedLabelsScratchBufferDev =
      reinterpret_cast<Npp32u *>(pUFCompressedLabelsScratchBufferDev[0]);

  int nMaxUFCompressedLabelsScratchBufferSize = 0;

  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    // This particular function works on in-place data and SrcDst image batch
    // list has already been initialized in batched label generation function
    // setup

    //  Initialize each per image buffer descriptor
    pUFBatchSrcDstScratchBufferListHost[nImage].pData =
        reinterpret_cast<void *>(pCurUFCompressedLabelsScratchBufferDev);
    pUFBatchSrcDstScratchBufferListHost[nImage].nBufferSize =
        aCompressLabelsScratchBufferSize[nImage];

    if (aCompressLabelsScratchBufferSize[nImage] >
        nMaxUFCompressedLabelsScratchBufferSize)
      nMaxUFCompressedLabelsScratchBufferSize =
          aCompressLabelsScratchBufferSize[nImage];

    // Offset buffer pointer to next per image buffer
    Npp8u *pTempBuffer =
        reinterpret_cast<Npp8u *>(pCurUFCompressedLabelsScratchBufferDev);
    pTempBuffer += aCompressLabelsScratchBufferSize[nImage];
    pCurUFCompressedLabelsScratchBufferDev =
        reinterpret_cast<Npp32u *>((void *)(pTempBuffer));
  }

  // Copy compression batch scratch buffer list from CPU to GPU
  cudaError = cudaMemcpyAsync(pUFBatchSrcDstScratchBufferListDev,
                              pUFBatchSrcDstScratchBufferListHost,
                              NUMBER_OF_IMAGES * sizeof(NppiBufferDescriptor),
                              cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (cudaError != cudaSuccess) return NPP_MEMCPY_ERROR;

  nppStatus = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(
      pUFBatchSrcDstImageListDev, pUFBatchSrcDstScratchBufferListDev,
      pUFBatchPerImageCompressedCountListDev, NUMBER_OF_IMAGES, oMaxROISize,
      nMaxUFCompressedLabelsScratchBufferSize, nppStreamCtx);
  if (nppStatus != NPP_SUCCESS) {
    printf("BatchCompressedLabelMarkersUF_8Way_32u failed.\n");
    tearDown();
    return -1;
  }

  // Copy output compressed label images back to host
  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    cudaError = cudaMemcpy2DAsync(
        pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
        pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u),
        oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
        cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
  }

  // Wait for host image read backs to complete, not necessary if no need to
  // synchronize
  if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) !=
      cudaSuccess) {
    printf("Post label compression cudaStreamSynchronize failed\n");
    tearDown();
    return -1;
  }

  // Save compressed label images into files
  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    if (nImage == 0)
      FOPEN(bmpFile, CompressedMarkerLabelsBatchOutputFile0.c_str(), "wb");
    else if (nImage == 1)
      FOPEN(bmpFile, CompressedMarkerLabelsBatchOutputFile1.c_str(), "wb");
    else if (nImage == 2)
      FOPEN(bmpFile, CompressedMarkerLabelsBatchOutputFile2.c_str(), "wb");
    else if (nImage == 3)
      FOPEN(bmpFile, CompressedMarkerLabelsBatchOutputFile3.c_str(), "wb");
    else if (nImage == 4)
      FOPEN(bmpFile, CompressedMarkerLabelsBatchOutputFile4.c_str(), "wb");

    if (bmpFile == NULL) return -1;
    size_t nSize = 0;
    for (int j = 0; j < oSizeROI[nImage].height; j++) {
      nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width],
                      sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
    }
    fclose(bmpFile);
  }

  // Read back per image compressed label count.
  cudaError = cudaMemcpyAsync(pUFBatchPerImageCompressedCountListHost,
                              pUFBatchPerImageCompressedCountListDev,
                              NUMBER_OF_IMAGES * sizeof(Npp32u),
                              cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
  if (cudaError != cudaSuccess) {
    tearDown();
    return NPP_MEMCPY_ERROR;
  }

  // Wait for host read back to complete
  cudaError = cudaStreamSynchronize(nppStreamCtx.hStream);

  printf("\n\n");

  for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++) {
    if (nImage == 0)
      printf(
          "teapot_CompressedMarkerLabelsUFBatch_8Way_512x512_32u succeeded, "
          "compressed label count is %d.\n",
          pUFBatchPerImageCompressedCountListHost[nImage]);
    else if (nImage == 1)
      printf(
          "CT_Skull_CompressedMarkerLabelsUFBatch_8Way_512x512_32u succeeded, "
          "compressed label count is %d.\n",
          pUFBatchPerImageCompressedCountListHost[nImage]);
    else if (nImage == 2)
      printf(
          "PCB_METAL_CompressedMarkerLabelsUFBatch_8Way_509x335_32u succeeded, "
          "compressed label count is %d.\n",
          pUFBatchPerImageCompressedCountListHost[nImage]);
    else if (nImage == 3)
      printf(
          "PCB2_CompressedMarkerLabelsUFBatch_8Way_1024x683_32u succeeded, "
          "compressed label count is %d.\n",
          pUFBatchPerImageCompressedCountListHost[nImage]);
    else if (nImage == 4)
      printf(
          "PCB_CompressedMarkerLabelsUFBatch_8Way_1280x720_32u succeeded, "
          "compressed label count is %d.\n",
          pUFBatchPerImageCompressedCountListHost[nImage]);
  }

#endif  // USE_BATCHED_LABEL_COMPRESSION

  tearDown();

  return 0;
}
