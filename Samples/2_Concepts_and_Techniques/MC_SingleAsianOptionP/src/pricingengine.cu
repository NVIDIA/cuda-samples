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

#include "../inc/pricingengine.h"

#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <curand_kernel.h>

#include "../inc/asianoption.h"
#include "../inc/cudasharedmem.h"

using std::string;
using std::vector;

// RNG init kernel
__global__ void initRNG(curandState *const rngStates, const unsigned int seed) {
  // Determine thread ID
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialise the RNG
  curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ inline float getPathStep(float &drift, float &diffusion,
                                    curandState &state) {
  return expf(drift + diffusion * curand_normal(&state));
}
__device__ inline double getPathStep(double &drift, double &diffusion,
                                     curandState &state) {
  return exp(drift + diffusion * curand_normal_double(&state));
}

// Path generation kernel
template <typename Real>
__global__ void generatePaths(Real *const paths, curandState *const rngStates,
                              const AsianOption<Real> *const option,
                              const unsigned int numSims,
                              const unsigned int numTimesteps) {
  // Determine thread ID
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = gridDim.x * blockDim.x;

  // Compute parameters
  Real drift =
      (option->r - static_cast<Real>(0.5) * option->sigma * option->sigma) *
      option->dt;
  Real diffusion = option->sigma * sqrt(option->dt);

  // Initialise the RNG
  curandState localState = rngStates[tid];

  for (unsigned int i = tid; i < numSims; i += step) {
    // Shift the output pointer
    Real *output = paths + i;

    // Simulate the path
    Real s = static_cast<Real>(1);

    for (unsigned int t = 0; t < numTimesteps; t++, output += numSims) {
      s *= getPathStep(drift, diffusion, localState);
      *output = s;
    }
  }
}

template <typename Real>
__device__ Real reduce_sum(Real in, cg::thread_block cta) {
  SharedMemory<Real> sdata;

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

// Valuation kernel
template <typename Real>
__global__ void computeValue(Real *const values, const Real *const paths,
                             const AsianOption<Real> *const option,
                             const unsigned int numSims,
                             const unsigned int numTimesteps) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Determine thread ID
  unsigned int bid = blockIdx.x;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = gridDim.x * blockDim.x;

  Real sumPayoffs = static_cast<Real>(0);

  for (unsigned int i = tid; i < numSims; i += step) {
    // Shift the input pointer
    const Real *path = paths + i;
    // Compute the arithmetic average
    Real avg = static_cast<Real>(0);

    for (unsigned int t = 0; t < numTimesteps; t++, path += numSims) {
      avg += *path;
    }

    avg = avg * option->spot / numTimesteps;
    // Compute the payoff
    Real payoff = avg - option->strike;

    if (option->type == AsianOption<Real>::Put) {
      payoff = -payoff;
    }

    payoff = max(static_cast<Real>(0), payoff);
    // Accumulate payoff locally
    sumPayoffs += payoff;
  }

  // Reduce within the block
  sumPayoffs = reduce_sum<Real>(sumPayoffs, cta);

  // Store the result
  if (threadIdx.x == 0) {
    values[bid] = sumPayoffs;
  }
}

template <typename Real>
PricingEngine<Real>::PricingEngine(unsigned int numSims, unsigned int device,
                                   unsigned int threadBlockSize,
                                   unsigned int seed)
    : m_numSims(numSims),
      m_device(device),
      m_threadBlockSize(threadBlockSize),
      m_seed(seed) {}

template <typename Real>
void PricingEngine<Real>::operator()(AsianOption<Real> &option) {
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
  unsigned int deviceVersion =
      deviceProperties.major * 10 + deviceProperties.minor;

  if (typeid(Real) == typeid(double) && deviceVersion < 13) {
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
  cudaResult = cudaFuncGetAttributes(&funcAttributes, initRNG);

  if (cudaResult != cudaSuccess) {
    string msg("Could not get function attributes: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for initRNG kernel");
  }

  // Get generatePaths function properties and check the maximum block size
  cudaResult = cudaFuncGetAttributes(&funcAttributes, generatePaths<Real>);

  if (cudaResult != cudaSuccess) {
    string msg("Could not get function attributes: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error(
        "Block X dimension is too large for generatePaths kernel");
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

  // Setup problem on GPU
  AsianOption<Real> *d_option = 0;
  cudaResult = cudaMalloc((void **)&d_option, sizeof(AsianOption<Real>));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for option data: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  cudaResult = cudaMemcpy(d_option, &option, sizeof(AsianOption<Real>),
                          cudaMemcpyHostToDevice);

  if (cudaResult != cudaSuccess) {
    string msg("Could not copy data to device: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Allocate memory for paths
  Real *d_paths = 0;
  int numTimesteps = static_cast<int>(option.tenor / option.dt);
  cudaResult =
      cudaMalloc((void **)&d_paths, m_numSims * numTimesteps * sizeof(Real));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for paths: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Allocate memory for RNG states
  curandState *d_rngStates = 0;
  cudaResult =
      cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for RNG state: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Allocate memory for result
  Real *d_values = 0;
  cudaResult = cudaMalloc((void **)&d_values, grid.x * sizeof(Real));

  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for partial results: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Initialise RNG
  initRNG<<<grid, block>>>(d_rngStates, m_seed);

  // Generate paths
  generatePaths<Real><<<grid, block>>>(d_paths, d_rngStates, d_option,
                                       m_numSims, numTimesteps);

  // Compute value
  computeValue<<<grid, block, block.x * sizeof(Real)>>>(
      d_values, d_paths, d_option, m_numSims, numTimesteps);

  // Copy partial results back
  vector<Real> values(grid.x);
  cudaResult = cudaMemcpy(&values[0], d_values, grid.x * sizeof(Real),
                          cudaMemcpyDeviceToHost);

  if (cudaResult != cudaSuccess) {
    string msg("Could not copy partial results to host: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Complete sum-reduction on host
  option.value =
      std::accumulate(values.begin(), values.end(), static_cast<Real>(0));

  // Compute the mean
  option.value /= m_numSims;

  // Discount to present value
  option.value *= exp(-option.r * option.tenor);

  // Cleanup
  if (d_option) {
    cudaFree(d_option);
    d_option = 0;
  }

  if (d_paths) {
    cudaFree(d_paths);
    d_paths = 0;
  }

  if (d_rngStates) {
    cudaFree(d_rngStates);
    d_rngStates = 0;
  }

  if (d_values) {
    cudaFree(d_values);
    d_values = 0;
  }
}

// Explicit template instantiation
template class PricingEngine<float>;
template class PricingEngine<double>;
