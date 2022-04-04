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

#include "../inc/piestimator.h"

#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <curand.h>

using std::string;
using std::vector;

__device__ unsigned int reduce_sum(unsigned int in, cg::thread_block cta) {
  extern __shared__ unsigned int sdata[];

  // Perform first level of reduction:
  // - Write to shared memory
  unsigned int ltid = threadIdx.x;

  sdata[ltid] = in;
  cg::sync(cta);

  // Do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (ltid < s) {
      sdata[ltid] += sdata[ltid + s];
    }

    cg::sync(cta);
  }

  return sdata[0];
}

// Estimator kernel
template <typename Real>
__global__ void computeValue(unsigned int *const results,
                             const Real *const points,
                             const unsigned int numSims) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Determine thread ID
  unsigned int bid = blockIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = gridDim.x * blockDim.x;

  // Shift the input/output pointers
  const Real *pointx = points + tid;
  const Real *pointy = pointx + numSims;

  // Count the number of points which lie inside the unit quarter-circle
  unsigned int pointsInside = 0;

  for (unsigned int i = tid; i < numSims;
       i += step, pointx += step, pointy += step) {
    Real x = *pointx;
    Real y = *pointy;
    Real l2norm2 = x * x + y * y;

    if (l2norm2 < static_cast<Real>(1)) {
      pointsInside++;
    }
  }

  // Reduce within the block
  pointsInside = reduce_sum(pointsInside, cta);

  // Store the result
  if (threadIdx.x == 0) {
    results[bid] = pointsInside;
  }
}

template <typename Real>
PiEstimator<Real>::PiEstimator(unsigned int numSims, unsigned int device,
                               unsigned int threadBlockSize)
    : m_numSims(numSims),
      m_device(device),
      m_threadBlockSize(threadBlockSize) {}

template <typename Real>
Real PiEstimator<Real>::operator()() {
  cudaError_t cudaResult = cudaSuccess;
  struct cudaDeviceProp deviceProperties;
  struct cudaFuncAttributes funcAttributes;

  // Get device properties
  cudaResult = cudaGetDeviceProperties(&deviceProperties, m_device);

  if (cudaResult != cudaSuccess) {
    string msg("Could not get device properties: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Check precision is valid
  if (typeid(Real) == typeid(double) &&
      (deviceProperties.major < 1 ||
       (deviceProperties.major == 1 && deviceProperties.minor < 3))) {
    throw std::runtime_error("Device does not have double precision support");
  }

  // Attach to GPU
  cudaResult = cudaSetDevice(m_device);

  if (cudaResult != cudaSuccess) {
    string msg("Could not set CUDA device: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Determine how to divide the work between cores
  dim3 block;
  dim3 grid;
  block.x = m_threadBlockSize;
  grid.x = (m_numSims + m_threadBlockSize - 1) / m_threadBlockSize;

  // Aim to launch around ten or more times as many blocks as there
  // are multiprocessors on the target device.
  unsigned int blocksPerSM = 10;
  unsigned int numSMs = deviceProperties.multiProcessorCount;

  while (grid.x > 2 * blocksPerSM * numSMs) {
    grid.x >>= 1;
  }

  // Get computeValue function properties and check the maximum block size
  cudaResult = cudaFuncGetAttributes(&funcAttributes, computeValue<Real>);

  if (cudaResult != cudaSuccess) {
    string msg("Could not get function attributes: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for computeValue kernel");
  }

  // Check the dimensions are valid
  if (block.x > (unsigned int)deviceProperties.maxThreadsDim[0]) {
    throw std::runtime_error("Block X dimension is too large for device");
  }

  if (grid.x > (unsigned int)deviceProperties.maxGridSize[0]) {
    throw std::runtime_error("Grid X dimension is too large for device");
  }

  // Allocate memory for points
  // Each simulation has two random numbers to give X and Y coordinate
  Real *d_points = 0;
  cudaResult = cudaMalloc((void **)&d_points, 2 * m_numSims * sizeof(Real));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for random numbers: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Allocate memory for result
  // Each thread block will produce one result
  unsigned int *d_results = 0;
  cudaResult = cudaMalloc((void **)&d_results, grid.x * sizeof(unsigned int));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for partial results: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Generate random points in unit square
  curandStatus_t curandResult;
  curandGenerator_t qrng;

  if (typeid(Real) == typeid(float)) {
    curandResult = curandCreateGenerator(&qrng, CURAND_RNG_QUASI_SOBOL32);
  } else if (typeid(Real) == typeid(double)) {
    curandResult = curandCreateGenerator(&qrng, CURAND_RNG_QUASI_SOBOL64);
  } else {
    string msg("Could not create random number generator of specified type");
    throw std::runtime_error(msg);
  }

  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not create quasi-random number generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  curandResult = curandSetQuasiRandomGeneratorDimensions(qrng, 2);

  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg(
        "Could not set number of dimensions for quasi-random number "
        "generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  curandResult =
      curandSetGeneratorOrdering(qrng, CURAND_ORDERING_QUASI_DEFAULT);

  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not set order for quasi-random number generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  if (typeid(Real) == typeid(float)) {
    curandResult =
        curandGenerateUniform(qrng, (float *)d_points, 2 * m_numSims);
  } else if (typeid(Real) == typeid(double)) {
    curandResult =
        curandGenerateUniformDouble(qrng, (double *)d_points, 2 * m_numSims);
  } else {
    string msg("Could not generate random numbers of specified type");
    throw std::runtime_error(msg);
  }

  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not generate quasi-random numbers: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  curandResult = curandDestroyGenerator(qrng);

  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not destroy quasi-random number generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  // Count the points inside unit quarter-circle
  computeValue<Real><<<grid, block, block.x * sizeof(unsigned int)>>>(
      d_results, d_points, m_numSims);

  // Copy partial results back
  vector<unsigned int> results(grid.x);
  cudaResult = cudaMemcpy(&results[0], d_results, grid.x * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost);

  if (cudaResult != cudaSuccess) {
    string msg("Could not copy partial results to host: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Complete sum-reduction on host
  Real value =
      static_cast<Real>(std::accumulate(results.begin(), results.end(), 0));

  // Determine the proportion of points inside the quarter-circle,
  // i.e. the area of the unit quarter-circle
  value /= m_numSims;

  // Value is currently an estimate of the area of a unit quarter-circle, so we
  // can scale to a full circle by multiplying by four. Now since the area of a
  // circle is pi * r^2, and r is one, the value will be an estimate for the
  // value of pi.
  value *= 4;

  // Cleanup
  if (d_points) {
    cudaFree(d_points);
    d_points = 0;
  }

  if (d_results) {
    cudaFree(d_results);
    d_results = 0;
  }

  return value;
}

// Explicit template instantiation
template class PiEstimator<float>;
template class PiEstimator<double>;
