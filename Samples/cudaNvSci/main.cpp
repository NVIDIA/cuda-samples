/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include "cudaNvSci.h"
#include <helper_cuda.h>
#include <helper_image.h>

void loadImageData(const std::string &filename, const char **argv,
                   unsigned char **image_data, uint32_t &imageWidth,
                   uint32_t &imageHeight) {
  // load image (needed so we can get the width and height before we create
  // the window
  char *image_path = sdkFindFilePath(filename.c_str(), argv[0]);

  if (image_path == 0) {
    printf("Error finding image file '%s'\n", filename.c_str());
    exit(EXIT_FAILURE);
  }

  sdkLoadPPM4(image_path, image_data, &imageWidth, &imageHeight);

  if (!image_data) {
    printf("Error opening file '%s'\n", image_path);
    exit(EXIT_FAILURE);
  }

  printf("Loaded '%s', %d x %d pixels\n", image_path, imageWidth, imageHeight);
}

int main(int argc, const char **argv) {
  int numOfGPUs = 0;
  std::vector<int> deviceIds;
  checkCudaErrors(cudaGetDeviceCount(&numOfGPUs));

  printf("%d GPUs found\n", numOfGPUs);
  if (!numOfGPUs) {
    exit(EXIT_WAIVED);
  } else {
    for (int devID = 0; devID < numOfGPUs; devID++) {
      int major = 0, minor = 0;
      checkCudaErrors(cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor, devID));
      checkCudaErrors(cudaDeviceGetAttribute(
          &minor, cudaDevAttrComputeCapabilityMinor, devID));
      if (major >= 6) {
        deviceIds.push_back(devID);
      }
    }
    if (deviceIds.size() == 0) {
      printf(
          "cudaNvSci requires one or more GPUs of Pascal(SM 6.0) or higher "
          "archs\nWaiving..\n");
      exit(EXIT_WAIVED);
    }
  }

  std::string image_filename = "lenaRGB.ppm";

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    getCmdLineArgumentString(argc, (const char **)argv, "file",
                             (char **)&image_filename);
  }

  uint32_t imageWidth = 0;
  uint32_t imageHeight = 0;
  unsigned char *image_data = NULL;

  loadImageData(image_filename, argv, &image_data, imageWidth, imageHeight);

  cudaNvSci cudaNvSciApp(deviceIds.size() > 1, deviceIds, image_data,
                         imageWidth, imageHeight);
  cudaNvSciApp.runCudaNvSci(image_filename);

  return EXIT_SUCCESS;
}