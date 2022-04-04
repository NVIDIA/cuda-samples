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
#include <curand_kernel.h>

#include "../inc/cudasharedmem.h"

using std::string;
using std::vector;

// Helper templates to support float and double in same code
template <typename L, typename R>
struct TYPE_IS {
  static const bool test = false;
};
template <typename L>
struct TYPE_IS<L, L> {
  static const bool test = true;
};
template <bool, class L, class R>
struct IF {
  typedef R type;
};
template <class L, class R>
struct IF<true, L, R> {
  typedef L type;
};

// RNG init kernel
template <typename rngState_t, typename rngDirectionVectors_t>
__global__ void initRNG(rngState_t *const rngStates,
                        rngDirectionVectors_t *const rngDirections,
                        unsigned int numDrawsPerDirection) {
  // Determine thread ID
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = gridDim.x * blockDim.x;

  // Determine offset to avoid overlapping sub-sequences
  unsigned int offset = tid * ((numDrawsPerDirection + step - 1) / step);

  // Initialise the RNG
  curand_init(rngDirections[0], offset, &rngStates[tid]);
  curand_init(rngDirections[1], offset, &rngStates[tid + step]);
}

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

__device__ inline void getPoint(float &x, float &y, curandStateSobol32 &state1,
                                curandStateSobol32 &state2) {
  x = curand_uniform(&state1);
  y = curand_uniform(&state2);
}
__device__ inline void getPoint(double &x, double &y,
                                curandStateSobol64 &state1,
                                curandStateSobol64 &state2) {
  x = curand_uniform_double(&state1);
  y = curand_uniform_double(&state2);
}

// Estimator kernel
template <typename Real, typename rngState_t>
__global__ void computeValue(unsigned int *const results,
                             rngState_t *const rngStates,
                             const unsigned int numSims) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Determine thread ID
  unsigned int bid = blockIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = gridDim.x * blockDim.x;

  // Initialise the RNG
  rngState_t localState1 = rngStates[tid];
  rngState_t localState2 = rngStates[tid + step];

  // Count the number of points which lie inside the unit quarter-circle
  unsigned int pointsInside = 0;

  for (unsigned int i = tid; i < numSims; i += step) {
    Real x;
    Real y;
    getPoint(x, y, localState1, localState2);
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

  // Determine type of generator to use (32- or 64-bit)
  typedef typename IF<TYPE_IS<Real, double>::test, curandStateSobol64_t,
                      curandStateSobol32_t>::type curandStateSobol_sz;
  typedef
      typename IF<TYPE_IS<Real, double>::test, curandDirectionVectors64_t,
                  curandDirectionVectors32_t>::type curandDirectionVectors_sz;

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

  // Get initRNG function properties and check the maximum block size
  cudaResult = cudaFuncGetAttributes(
      &funcAttributes, initRNG<curandStateSobol_sz, curandDirectionVectors_sz>);

  if (cudaResult != cudaSuccess) {
    string msg("Could not get function attributes: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for initRNG kernel");
  }

  // Get computeValue function properties and check the maximum block size
  cudaResult = cudaFuncGetAttributes(&funcAttributes,
                                     computeValue<Real, curandStateSobol_sz>);

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

  // Allocate memory for RNG states and direction vectors
  curandStateSobol_sz *d_rngStates = 0;
  curandDirectionVectors_sz *d_rngDirections = 0;
  cudaResult = cudaMalloc((void **)&d_rngStates,
                          2 * grid.x * block.x * sizeof(curandStateSobol_sz));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for RNG states: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  cudaResult = cudaMalloc((void **)&d_rngDirections,
                          2 * sizeof(curandDirectionVectors_sz));

  if (cudaResult != cudaSuccess) {
    string msg(
        "Could not allocate memory on device for RNG direction vectors: ");
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

  // Generate direction vectors on the host and copy to the device
  if (typeid(Real) == typeid(float)) {
    curandDirectionVectors32_t *rngDirections;
    curandStatus_t curandResult = curandGetDirectionVectors32(
        &rngDirections, CURAND_DIRECTION_VECTORS_32_JOEKUO6);

    if (curandResult != CURAND_STATUS_SUCCESS) {
      string msg(
          "Could not get direction vectors for quasi-random number "
          "generator: ");
      msg += curandResult;
      throw std::runtime_error(msg);
    }

    cudaResult = cudaMemcpy(d_rngDirections, rngDirections,
                            2 * sizeof(curandDirectionVectors32_t),
                            cudaMemcpyHostToDevice);

    if (cudaResult != cudaSuccess) {
      string msg("Could not copy direction vectors to device: ");
      msg += cudaGetErrorString(cudaResult);
      throw std::runtime_error(msg);
    }
  } else if (typeid(Real) == typeid(double)) {
    curandDirectionVectors64_t *rngDirections;
    curandStatus_t curandResult = curandGetDirectionVectors64(
        &rngDirections, CURAND_DIRECTION_VECTORS_64_JOEKUO6);

    if (curandResult != CURAND_STATUS_SUCCESS) {
      string msg(
          "Could not get direction vectors for quasi-random number "
          "generator: ");
      msg += curandResult;
      throw std::runtime_error(msg);
    }

    cudaResult = cudaMemcpy(d_rngDirections, rngDirections,
                            2 * sizeof(curandDirectionVectors64_t),
                            cudaMemcpyHostToDevice);

    if (cudaResult != cudaSuccess) {
      string msg("Could not copy direction vectors to device: ");
      msg += cudaGetErrorString(cudaResult);
      throw std::runtime_error(msg);
    }
  } else {
    string msg(
        "Could not get direction vectors for random number generator of "
        "specified type");
    throw std::runtime_error(msg);
  }

  // Initialise RNG
  initRNG<<<grid, block>>>(d_rngStates, d_rngDirections, m_numSims);

  // Count the points inside unit quarter-circle
  computeValue<Real><<<grid, block, block.x * sizeof(unsigned int)>>>(
      d_results, d_rngStates, m_numSims);

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
  if (d_rngStates) {
    cudaFree(d_rngStates);
    d_rngStates = 0;
  }

  if (d_rngDirections) {
    cudaFree(d_rngDirections);
    d_rngDirections = 0;
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
