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

#pragma once
#ifndef __PISIM_H__
#define __PISIM_H__

#include <vector>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

#include "helper_multiprocess.h"

typedef float vec2[2];

class MonteCarloPiSimulation {
  size_t m_numPoints;

  // Pointers to Cuda allocated buffers which are imported and used by vulkan as
  // vertex buffer
  vec2 *m_xyVector;
  float *m_pointsInsideCircle;

  // Pointers to device and host allocated memories storing number of points
  // that are inside the unit circle
  float *m_numPointsInCircle;
  float *m_hostNumPointsInCircle;

  int m_blocks, m_threads;

  // Total size of allocations created by cuMemMap Apis. This size is the sum of
  // sizes of m_xyVector and m_pointsInsideCircle buffers.
  size_t m_totalAllocationSize;

  // Shareable Handles(a file descriptor on Linux and NT Handle on Windows),
  // used for sharing cuda
  // allocated memory with Vulkan
  ShareableHandle m_posShareableHandle, m_inCircleShareableHandle;

  // Cuda Device corresponding to the Vulkan Physical device
  int m_cudaDevice;

  // Track and accumulate total points that have been simulated since start of
  // the sample. The idea is to get a closer approximation to PI with time.
  size_t m_totalPointsInsideCircle;
  size_t m_totalPointsSimulated;

  void setupSimulationAllocations();
  void cleanupSimulationAllocations();
  void getIdealExecutionConfiguration();

 public:
  MonteCarloPiSimulation(size_t num_points);
  ~MonteCarloPiSimulation();
  void initSimulation(int cudaDevice, cudaStream_t stream = 0);
  void stepSimulation(float time, cudaStream_t stream = 0);
  static void computePiCallback(void *args);

  size_t getNumPoints() const { return m_numPoints; }

  float getNumPointsInCircle() const { return *m_hostNumPointsInCircle; }

  ShareableHandle &getPositionShareableHandle() { return m_posShareableHandle; }
  ShareableHandle &getInCircleShareableHandle() {
    return m_inCircleShareableHandle;
  }
};

#endif  // __PISIM_H__
