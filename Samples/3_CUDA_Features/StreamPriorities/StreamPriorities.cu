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

// std::system includes
#include <cstdio>

// CUDA-C includes
#include <cuda_runtime.h>

#include <helper_cuda.h>

#define TOTAL_SIZE 256 * 1024 * 1024
#define EACH_SIZE 128 * 1024 * 1024

// # threadblocks
#define TBLOCKS 1024
#define THREADS 512

// throw error on equality
#define ERR_EQ(X, Y)                                                           \
  do {                                                                         \
    if ((X) == (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

// throw error on difference
#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

// copy from source -> destination arrays
__global__ void memcpy_kernel(int *dst, int *src, size_t n) {
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = id; i < n / sizeof(int); i += num) {
    dst[i] = src[i];
  }
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

int main(int argc, char **argv) {
  cudaDeviceProp device_prop;
  int dev_id;

  printf("Starting [%s]...\n", argv[0]);

  // set device
  dev_id = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

  if ((device_prop.major << 4) + device_prop.minor < 0x35) {
    fprintf(stderr,
            "%s requires Compute Capability of SM 3.5 or higher to "
            "run.\nexiting...\n",
            argv[0]);
    exit(EXIT_WAIVED);
  }

  // get the range of priorities available
  // [ greatest_priority, lowest_priority ]
  int priority_low;
  int priority_hi;
  checkCudaErrors(
      cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));

  printf("CUDA stream priority range: LOW: %d to HIGH: %d\n", priority_low,
         priority_hi);

  // create streams with highest and lowest available priorities
  cudaStream_t st_low;
  cudaStream_t st_hi;
  checkCudaErrors(cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking,
                                               priority_low));
  checkCudaErrors(
      cudaStreamCreateWithPriority(&st_hi, cudaStreamNonBlocking, priority_hi));

  size_t size;
  size = TOTAL_SIZE;

  // initialise host data
  int *h_src_low;
  int *h_src_hi;
  ERR_EQ(h_src_low = (int *)malloc(size), NULL);
  ERR_EQ(h_src_hi = (int *)malloc(size), NULL);
  mem_init(h_src_low, size);
  mem_init(h_src_hi, size);

  // initialise device data
  int *h_dst_low;
  int *h_dst_hi;
  ERR_EQ(h_dst_low = (int *)malloc(size), NULL);
  ERR_EQ(h_dst_hi = (int *)malloc(size), NULL);
  memset(h_dst_low, 0, size);
  memset(h_dst_hi, 0, size);

  // copy source data -> device
  int *d_src_low;
  int *d_src_hi;
  checkCudaErrors(cudaMalloc(&d_src_low, size));
  checkCudaErrors(cudaMalloc(&d_src_hi, size));
  checkCudaErrors(
      cudaMemcpy(d_src_low, h_src_low, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_src_hi, h_src_hi, size, cudaMemcpyHostToDevice));

  // allocate memory for memcopy destination
  int *d_dst_low;
  int *d_dst_hi;
  checkCudaErrors(cudaMalloc(&d_dst_low, size));
  checkCudaErrors(cudaMalloc(&d_dst_hi, size));

  // create some events
  cudaEvent_t ev_start_low;
  cudaEvent_t ev_start_hi;
  cudaEvent_t ev_end_low;
  cudaEvent_t ev_end_hi;
  checkCudaErrors(cudaEventCreate(&ev_start_low));
  checkCudaErrors(cudaEventCreate(&ev_start_hi));
  checkCudaErrors(cudaEventCreate(&ev_end_low));
  checkCudaErrors(cudaEventCreate(&ev_end_hi));

  /* */

  // call pair of kernels repeatedly (with different priority streams)
  checkCudaErrors(cudaEventRecord(ev_start_low, st_low));
  checkCudaErrors(cudaEventRecord(ev_start_hi, st_hi));

  for (int i = 0; i < TOTAL_SIZE; i += EACH_SIZE) {
    int j = i / sizeof(int);
    memcpy_kernel<<<TBLOCKS, THREADS, 0, st_low>>>(d_dst_low + j, d_src_low + j,
                                                   EACH_SIZE);
    memcpy_kernel<<<TBLOCKS, THREADS, 0, st_hi>>>(d_dst_hi + j, d_src_hi + j,
                                                  EACH_SIZE);
  }

  checkCudaErrors(cudaEventRecord(ev_end_low, st_low));
  checkCudaErrors(cudaEventRecord(ev_end_hi, st_hi));

  checkCudaErrors(cudaEventSynchronize(ev_end_low));
  checkCudaErrors(cudaEventSynchronize(ev_end_hi));

  /* */

  size = TOTAL_SIZE;
  checkCudaErrors(
      cudaMemcpy(h_dst_low, d_dst_low, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_dst_hi, d_dst_hi, size, cudaMemcpyDeviceToHost));

  // check results of kernels
  ERR_NE(memcmp(h_dst_low, h_src_low, size), 0);
  ERR_NE(memcmp(h_dst_hi, h_src_hi, size), 0);

  // check timings
  float ms_low;
  float ms_hi;
  checkCudaErrors(cudaEventElapsedTime(&ms_low, ev_start_low, ev_end_low));
  checkCudaErrors(cudaEventElapsedTime(&ms_hi, ev_start_hi, ev_end_hi));

  printf("elapsed time of kernels launched to LOW priority stream: %.3lf ms\n",
         ms_low);
  printf("elapsed time of kernels launched to HI  priority stream: %.3lf ms\n",
         ms_hi);

  exit(EXIT_SUCCESS);
}
