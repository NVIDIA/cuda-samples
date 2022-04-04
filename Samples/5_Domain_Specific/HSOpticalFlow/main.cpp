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

const static char *const sSDKsample = "HSOpticalFlow";

// CPU-GPU discrepancy threshold for self-test
const float THRESHOLD = 0.05f;

#include <cuda_runtime.h>

#include "common.h"
#include "flowGold.h"
#include "flowCUDA.h"

#include <helper_functions.h>

///////////////////////////////////////////////////////////////////////////////
/// \brief save optical flow in format described on vision.middlebury.edu/flow
/// \param[in] name output file name
/// \param[in] w    optical flow field width
/// \param[in] h    optical flow field height
/// \param[in] s    optical flow field row stride
/// \param[in] u    horizontal displacement
/// \param[in] v    vertical displacement
///////////////////////////////////////////////////////////////////////////////
void WriteFloFile(const char *name, int w, int h, int s, const float *u,
                  const float *v) {
  FILE *stream;
  stream = fopen(name, "wb");

  if (stream == 0) {
    printf("Could not save flow to \"%s\"\n", name);
    return;
  }

  float data = 202021.25f;
  fwrite(&data, sizeof(float), 1, stream);
  fwrite(&w, sizeof(w), 1, stream);
  fwrite(&h, sizeof(h), 1, stream);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int pos = j + i * s;
      fwrite(u + pos, sizeof(float), 1, stream);
      fwrite(v + pos, sizeof(float), 1, stream);
    }
  }

  fclose(stream);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief
/// load 4-channel unsigned byte image
/// and convert it to single channel FP32 image
/// \param[out] img_data pointer to raw image data
/// \param[out] img_w    image width
/// \param[out] img_h    image height
/// \param[out] img_s    image row stride
/// \param[in]  name     image file name
/// \param[in]  exePath  executable file path
/// \return true if image is successfully loaded or false otherwise
///////////////////////////////////////////////////////////////////////////////
bool LoadImageAsFP32(float *&img_data, int &img_w, int &img_h, int &img_s,
                     const char *name, const char *exePath) {
  printf("Loading \"%s\" ...\n", name);
  char *name_ = sdkFindFilePath(name, exePath);

  if (!name_) {
    printf("File not found\n");
    return false;
  }

  unsigned char *data = 0;
  unsigned int w = 0, h = 0;
  bool result = sdkLoadPPM4ub(name_, &data, &w, &h);

  if (result == false) {
    printf("Invalid file format\n");
    return false;
  }

  img_w = w;
  img_h = h;
  img_s = iAlignUp(img_w);

  img_data = new float[img_s * h];

  // source is 4 channel image
  const int widthStep = 4 * img_w;

  for (int i = 0; i < img_h; ++i) {
    for (int j = 0; j < img_w; ++j) {
      img_data[j + i * img_s] = ((float)data[j * 4 + i * widthStep]) / 255.0f;
    }
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compare given flow field with gold (L1 norm)
/// \param[in] width    optical flow field width
/// \param[in] height   optical flow field height
/// \param[in] stride   optical flow field row stride
/// \param[in] h_uGold  horizontal displacement, gold
/// \param[in] h_vGold  vertical displacement, gold
/// \param[in] h_u      horizontal displacement
/// \param[in] h_v      vertical displacement
/// \return true if discrepancy is lower than a given threshold
///////////////////////////////////////////////////////////////////////////////
bool CompareWithGold(int width, int height, int stride, const float *h_uGold,
                     const float *h_vGold, const float *h_u, const float *h_v) {
  float error = 0.0f;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int pos = j + i * stride;
      error += fabsf(h_u[pos] - h_uGold[pos]) + fabsf(h_v[pos] - h_vGold[pos]);
    }
  }

  error /= (float)(width * height);

  printf("L1 error : %.6f\n", error);

  return (error < THRESHOLD);
}

///////////////////////////////////////////////////////////////////////////////
/// application entry point
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // welcome message
  printf("%s Starting...\n\n", sSDKsample);

  // pick GPU
  findCudaDevice(argc, (const char **)argv);

  // find images
  const char *const sourceFrameName = "frame10.ppm";
  const char *const targetFrameName = "frame11.ppm";

  // image dimensions
  int width;
  int height;
  // row access stride
  int stride;

  // flow is computed from source image to target image
  float *h_source;  // source image, host memory
  float *h_target;  // target image, host memory

  // load image from file
  if (!LoadImageAsFP32(h_source, width, height, stride, sourceFrameName,
                       argv[0])) {
    exit(EXIT_FAILURE);
  }

  if (!LoadImageAsFP32(h_target, width, height, stride, targetFrameName,
                       argv[0])) {
    exit(EXIT_FAILURE);
  }

  // allocate host memory for CPU results
  float *h_uGold = new float[stride * height];
  float *h_vGold = new float[stride * height];

  // allocate host memory for GPU results
  float *h_u = new float[stride * height];
  float *h_v = new float[stride * height];

  // smoothness
  // if image brightness is not within [0,1]
  // this paramter should be scaled appropriately
  const float alpha = 0.2f;

  // number of pyramid levels
  const int nLevels = 5;

  // number of solver iterations on each level
  const int nSolverIters = 500;

  // number of warping iterations
  const int nWarpIters = 3;

  ComputeFlowGold(h_source, h_target, width, height, stride, alpha, nLevels,
                  nWarpIters, nSolverIters, h_uGold, h_vGold);

  ComputeFlowCUDA(h_source, h_target, width, height, stride, alpha, nLevels,
                  nWarpIters, nSolverIters, h_u, h_v);

  // compare results (L1 norm)
  bool status =
      CompareWithGold(width, height, stride, h_uGold, h_vGold, h_u, h_v);

  WriteFloFile("FlowGPU.flo", width, height, stride, h_u, h_v);

  WriteFloFile("FlowCPU.flo", width, height, stride, h_uGold, h_vGold);

  // free resources
  delete[] h_uGold;
  delete[] h_vGold;

  delete[] h_u;
  delete[] h_v;

  delete[] h_source;
  delete[] h_target;

  // report self-test status
  exit(status ? EXIT_SUCCESS : EXIT_FAILURE);
}
