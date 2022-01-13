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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

extern "C" {

void cudaInit(int argc, char **argv) {
  int devID;

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  devID = findCudaDevice(argc, (const char **)argv);

  if (devID < 0) {
    printf("No CUDA Capable devices found, exiting...\n");
    exit(EXIT_SUCCESS);
  }
}

void allocateArray(void **devPtr, size_t size) {
  checkCudaErrors(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr) { checkCudaErrors(cudaFree(devPtr)); }

void threadSync() { checkCudaErrors(cudaDeviceSynchronize()); }

void copyArrayToDevice(void *device, const void *host, int offset, int size) {
  checkCudaErrors(
      cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo,
                            struct cudaGraphicsResource **cuda_vbo_resource) {
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                               cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource) {
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource) {
  void *ptr;
  checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&ptr, &num_bytes, *cuda_vbo_resource));
  return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource) {
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void copyArrayFromDevice(void *host, const void *device,
                         struct cudaGraphicsResource **cuda_vbo_resource,
                         int size) {
  if (cuda_vbo_resource) {
    device = mapGLBufferObject(cuda_vbo_resource);
  }

  checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

  if (cuda_vbo_resource) {
    unmapGLBufferObject(*cuda_vbo_resource);
  }
}

void setParameters(SimParams *hostParams) {
  // copy parameters to constant memory
  checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
}

// Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks,
                     uint &numThreads) {
  numThreads = min(blockSize, n);
  numBlocks = iDivUp(n, numThreads);
}

void integrateSystem(float *pos, float *vel, float deltaTime,
                     uint numParticles) {
  thrust::device_ptr<float4> d_pos4((float4 *)pos);
  thrust::device_ptr<float4> d_vel4((float4 *)vel);

  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
      thrust::make_zip_iterator(
          thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles)),
      integrate_functor(deltaTime));
}

void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos,
              int numParticles) {
  uint numThreads, numBlocks;
  computeGridSize(numParticles, 256, numBlocks, numThreads);

  // execute the kernel
  calcHashD<<<numBlocks, numThreads>>>(gridParticleHash, gridParticleIndex,
                                       (float4 *)pos, numParticles);

  // check if kernel invocation generated an error
  getLastCudaError("Kernel execution failed");
}

void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd,
                                 float *sortedPos, float *sortedVel,
                                 uint *gridParticleHash,
                                 uint *gridParticleIndex, float *oldPos,
                                 float *oldVel, uint numParticles,
                                 uint numCells) {
  uint numThreads, numBlocks;
  computeGridSize(numParticles, 256, numBlocks, numThreads);

  // set all cells to empty
  checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

  uint smemSize = sizeof(uint) * (numThreads + 1);
  reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(
      cellStart, cellEnd, (float4 *)sortedPos, (float4 *)sortedVel,
      gridParticleHash, gridParticleIndex, (float4 *)oldPos, (float4 *)oldVel,
      numParticles);
  getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

void collide(float *newVel, float *sortedPos, float *sortedVel,
             uint *gridParticleIndex, uint *cellStart, uint *cellEnd,
             uint numParticles, uint numCells) {
  // thread per particle
  uint numThreads, numBlocks;
  computeGridSize(numParticles, 64, numBlocks, numThreads);

  // execute the kernel
  collideD<<<numBlocks, numThreads>>>((float4 *)newVel, (float4 *)sortedPos,
                                      (float4 *)sortedVel, gridParticleIndex,
                                      cellStart, cellEnd, numParticles);

  // check if kernel invocation generated an error
  getLastCudaError("Kernel execution failed");
}

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex,
                   uint numParticles) {
  thrust::sort_by_key(
      thrust::device_ptr<uint>(dGridParticleHash),
      thrust::device_ptr<uint>(dGridParticleHash + numParticles),
      thrust::device_ptr<uint>(dGridParticleIndex));
}

}  // extern "C"
