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
NVIDIA HW Decoder, both dGPU and Tegra, normally outputs NV12 pitch format
frames. For the inference using TensorRT, the input frame needs to be BGR planar
format with possibly different size. So, conversion and resizing from NV12 to
BGR planar is usually required for the inference following decoding.
This CUDA code is to provide a reference implementation for conversion and
resizing.

Limitaion
=========
    NV12resize needs the height to be a even value.

Note
====
    Resize function needs the pitch of image buffer to be 32 alignment.

Run
====
./NV12toBGRandResize
   OR
./NV12toBGRandResize -input=data/test1920x1080.nv12 -width=1920 -height=1080 \
-dst_width=640 -dst_height=480 -batch=40 -device=0

*/

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include "resize_convert.h"
#include "utils.h"

#define TEST_LOOP 20

typedef struct _nv12_to_bgr24_context_t {
  int width;
  int height;
  int pitch;

  int dst_width;
  int dst_height;
  int dst_pitch;

  int batch;
  int device;  // cuda device ID

  char *input_nv12_file;

  int ctx_pitch;    // the value will be suitable for Texture memroy.
  int ctx_heights;  // the value will be even.

} nv12_to_bgr24_context;

nv12_to_bgr24_context g_ctx;

static void printHelp(const char *app_name) {
  std::cout << "Usage:" << app_name << " [options]\n\n";
  std::cout << "OPTIONS:\n";
  std::cout << "\t-h,--help\n\n";
  std::cout << "\t-input=nv12file             nv12 input file\n";
  std::cout
      << "\t-width=width                input nv12 image width, <1 -- 4096>\n";
  std::cout
      << "\t-height=height              input nv12 image height, <1 -- 4096>\n";
  std::cout
      << "\t-pitch=pitch(optional)      input nv12 image pitch, <0 -- 4096>\n";
  std::cout
      << "\t-dst_width=width            output BGR image width, <1 -- 4096>\n";
  std::cout
      << "\t-dst_height=height          output BGR image height, <1 -- 4096>\n";
  std::cout
      << "\t-dst_pitch=pitch(optional)  output BGR image pitch, <0 -- 4096>\n";
  std::cout
      << "\t-batch=batch                process frames count, <1 -- 4096>\n\n";
  std::cout
      << "\t-device=device_num(optional)   cuda device number, <0 -- 4096>\n\n";

  return;
}

int parseCmdLine(int argc, char *argv[]) {
  char **argp = (char **)argv;
  char *arg = (char *)argv[0];

  memset(&g_ctx, 0, sizeof(g_ctx));

  if ((arg && (!strcmp(arg, "-h") || !strcmp(arg, "--help")))) {
    printHelp(argv[0]);
    return -1;
  }

  if (argc == 1) {
    // Run using default arguments

    g_ctx.input_nv12_file = sdkFindFilePath("test1920x1080.nv12", argv[0]);
    if (g_ctx.input_nv12_file == NULL) {
      printf("Cannot find input file test1920x1080.nv12\n Exiting\n");
      return EXIT_FAILURE;
    }
    g_ctx.width = 1920;
    g_ctx.height = 1080;
    g_ctx.dst_width = 640;
    g_ctx.dst_height = 480;
    g_ctx.batch = 24;
  } else if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
      g_ctx.width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
      g_ctx.height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "pitch")) {
      g_ctx.pitch = getCmdLineArgumentInt(argc, (const char **)argv, "pitch");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input",
                               (char **)&g_ctx.input_nv12_file);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dst_width")) {
      g_ctx.dst_width =
          getCmdLineArgumentInt(argc, (const char **)argv, "dst_width");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dst_height")) {
      g_ctx.dst_height =
          getCmdLineArgumentInt(argc, (const char **)argv, "dst_height");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dst_pitch")) {
      g_ctx.dst_pitch =
          getCmdLineArgumentInt(argc, (const char **)argv, "dst_pitch");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "batch")) {
      g_ctx.batch = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
    }
  }

  g_ctx.device = findCudaDevice(argc, (const char **)argv);

  if ((g_ctx.width == 0) || (g_ctx.height == 0) || (g_ctx.dst_width == 0) ||
      (g_ctx.dst_height == 0) || !g_ctx.input_nv12_file) {
    printHelp(argv[0]);
    return -1;
  }

  if (g_ctx.pitch == 0) g_ctx.pitch = g_ctx.width;
  if (g_ctx.dst_pitch == 0) g_ctx.dst_pitch = g_ctx.dst_width;

  return 0;
}

/*
  load nv12 yuvfile data into GPU device memory with batch of copy
 */
static int loadNV12Frame(unsigned char *d_inputNV12) {
  unsigned char *pNV12FrameData;
  unsigned char *d_nv12;
  int frameSize;
  std::ifstream nv12File(g_ctx.input_nv12_file, std::ifstream::in | std::ios::binary);

  if (!nv12File.is_open()) {
    std::cerr << "Can't open files\n";
    return -1;
  }

  frameSize = g_ctx.pitch * g_ctx.ctx_heights;

#if USE_UVM_MEM
  pNV12FrameData = d_inputNV12;
#else
  pNV12FrameData = (unsigned char *)malloc(frameSize);
  if (pNV12FrameData == NULL) {
    std::cerr << "Failed to malloc pNV12FrameData\n";
    return -1;
  }
#endif

  nv12File.read((char *)pNV12FrameData, frameSize);

  if (nv12File.gcount() < frameSize) {
    std::cerr << "can't get one frame!\n";
    return -1;
  }

#if USE_UVM_MEM
  // Prefetch to GPU for following GPU operation
  cudaStreamAttachMemAsync(NULL, pNV12FrameData, 0, cudaMemAttachGlobal);
#endif

  // expand one frame to multi frames for batch processing
  d_nv12 = d_inputNV12;
  for (int i = 0; i < g_ctx.batch; i++) {
    checkCudaErrors(cudaMemcpy2D((void *)d_nv12, g_ctx.ctx_pitch,
                                 pNV12FrameData, g_ctx.width, g_ctx.width,
                                 g_ctx.ctx_heights, cudaMemcpyHostToDevice));

    d_nv12 += g_ctx.ctx_pitch * g_ctx.ctx_heights;
  }

#if (USE_UVM_MEM == 0)
  free(pNV12FrameData);
#endif
  nv12File.close();

  return 0;
}

/*
  1. resize interlace nv12 to target size
  2. convert nv12 to bgr 3 progressive planars
 */
void nv12ResizeAndNV12ToBGR(unsigned char *d_inputNV12) {
  unsigned char *d_resizedNV12;
  float *d_outputBGR;
  int size;
  char filename[40];

  /* allocate device memory for resized nv12 output */
  size = g_ctx.dst_width * ceil(g_ctx.dst_height * 3.0f / 2.0f) * g_ctx.batch *
         sizeof(unsigned char);
  checkCudaErrors(cudaMalloc((void **)&d_resizedNV12, size));

  /* allocate device memory for bgr output */
  size = g_ctx.dst_pitch * g_ctx.dst_height * 3 * g_ctx.batch * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_outputBGR, size));

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  /* create cuda event handles */
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsedTime = 0.0f;

  /* resize interlace nv12 */

  cudaEventRecord(start, 0);
  for (int i = 0; i < TEST_LOOP; i++) {
    resizeNV12Batch(d_inputNV12, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height,
                    d_resizedNV12, g_ctx.dst_width, g_ctx.dst_width,
                    g_ctx.dst_height, g_ctx.batch);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(
      "  CUDA resize nv12(%dx%d --> %dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.width, g_ctx.height, g_ctx.dst_width, g_ctx.dst_height, g_ctx.batch,
      (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  sprintf(filename, "resized_nv12_%dx%d", g_ctx.dst_width, g_ctx.dst_height);

  /* convert nv12 to bgr 3 progressive planars */
  cudaEventRecord(start, 0);
  for (int i = 0; i < TEST_LOOP; i++) {
    nv12ToBGRplanarBatch(d_resizedNV12, g_ctx.dst_pitch,  // intput
                         d_outputBGR,
                         g_ctx.dst_pitch * sizeof(float),    // output
                         g_ctx.dst_width, g_ctx.dst_height,  // output
                         g_ctx.batch, 0);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);

  printf(
      "  CUDA convert nv12(%dx%d) to bgr(%dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.dst_width, g_ctx.dst_height, g_ctx.dst_width, g_ctx.dst_height,
      g_ctx.batch, (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  sprintf(filename, "converted_bgr_%dx%d", g_ctx.dst_width, g_ctx.dst_height);
  dumpBGR(d_outputBGR, g_ctx.dst_pitch, g_ctx.dst_width, g_ctx.dst_height,
          g_ctx.batch, (char *)"t1", filename);

  /* release resources */
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_resizedNV12));
  checkCudaErrors(cudaFree(d_outputBGR));
}

/*
  1. convert nv12 to bgr 3 progressive planars
  2. resize bgr 3 planars to target size
*/
void nv12ToBGRandBGRresize(unsigned char *d_inputNV12) {
  float *d_bgr;
  float *d_resizedBGR;
  int size;
  char filename[40];

  /* allocate device memory for bgr output */
  size = g_ctx.ctx_pitch * g_ctx.height * 3 * g_ctx.batch * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_bgr, size));

  /* allocate device memory for resized bgr output */
  size = g_ctx.dst_width * g_ctx.dst_height * 3 * g_ctx.batch * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_resizedBGR, size));

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  /* create cuda event handles */
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsedTime = 0.0f;

  /* convert interlace nv12 to bgr 3 progressive planars */
  cudaEventRecord(start, 0);
  cudaDeviceSynchronize();
  for (int i = 0; i < TEST_LOOP; i++) {
    nv12ToBGRplanarBatch(d_inputNV12, g_ctx.ctx_pitch, d_bgr,
                         g_ctx.ctx_pitch * sizeof(float), g_ctx.width,
                         g_ctx.height, g_ctx.batch, 0);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(
      "  CUDA convert nv12(%dx%d) to bgr(%dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.width, g_ctx.height, g_ctx.width, g_ctx.height, g_ctx.batch,
      (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  sprintf(filename, "converted_bgr_%dx%d", g_ctx.width, g_ctx.height);

  /* resize bgr 3 progressive planars */
  cudaEventRecord(start, 0);
  for (int i = 0; i < TEST_LOOP; i++) {
    resizeBGRplanarBatch(d_bgr, g_ctx.ctx_pitch, g_ctx.width, g_ctx.height,
                         d_resizedBGR, g_ctx.dst_width, g_ctx.dst_width,
                         g_ctx.dst_height, g_ctx.batch);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(
      "  CUDA resize bgr(%dx%d --> %dx%d), batch: %d,"
      " average time: %.3f ms ==> %.3f ms/frame\n",
      g_ctx.width, g_ctx.height, g_ctx.dst_width, g_ctx.dst_height, g_ctx.batch,
      (elapsedTime / (TEST_LOOP * 1.0f)),
      (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

  memset(filename, 0, sizeof(filename));
  sprintf(filename, "resized_bgr_%dx%d", g_ctx.dst_width, g_ctx.dst_height);
  dumpBGR(d_resizedBGR, g_ctx.dst_pitch, g_ctx.dst_width, g_ctx.dst_height,
          g_ctx.batch, (char *)"t2", filename);

  /* release resources */
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_bgr));
  checkCudaErrors(cudaFree(d_resizedBGR));
}

int main(int argc, char *argv[]) {
  unsigned char *d_inputNV12;

  if (parseCmdLine(argc, argv) < 0) return EXIT_FAILURE;

  g_ctx.ctx_pitch = g_ctx.width;
  int ctx_alignment = 32;
  g_ctx.ctx_pitch += (g_ctx.ctx_pitch % ctx_alignment != 0)
                         ? (ctx_alignment - g_ctx.ctx_pitch % ctx_alignment)
                         : 0;

  g_ctx.ctx_heights = ceil(g_ctx.height * 3.0f / 2.0f);

  /* load nv12 yuv data into d_inputNV12 with batch of copies */
#if USE_UVM_MEM
  checkCudaErrors(cudaMallocManaged(
      (void **)&d_inputNV12,
      (g_ctx.ctx_pitch * g_ctx.ctx_heights * g_ctx.batch), cudaMemAttachHost));
  printf("\nUSE_UVM_MEM\n");
#else
  checkCudaErrors(
      cudaMalloc((void **)&d_inputNV12,
                 (g_ctx.ctx_pitch * g_ctx.ctx_heights * g_ctx.batch)));
#endif
  if (loadNV12Frame(d_inputNV12)) {
    std::cerr << "failed to load batch data!\n";
    return EXIT_FAILURE;
  }

  /* firstly resize nv12, then convert nv12 to bgr */
  printf("\nTEST#1:\n");
  nv12ResizeAndNV12ToBGR(d_inputNV12);

  /* first convert nv12 to bgr, then resize bgr */
  printf("\nTEST#2:\n");
  nv12ToBGRandBGRresize(d_inputNV12);

  checkCudaErrors(cudaFree(d_inputNV12));

  return EXIT_SUCCESS;
}
