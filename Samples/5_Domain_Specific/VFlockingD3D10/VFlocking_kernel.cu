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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <VFlockingD3D10.h>

#define PI 3.1415926536f

typedef unsigned int uint;

__device__ bool isInsideQuad_D(float2 pos0, float2 pos1, float width,
                               float height) {
  if (fabs(pos0.x - pos1.x) < 0.5f * width &&
      fabs(pos0.y - pos1.y) < 0.5f * height) {
    return true;
  } else {
    return false;
  }
}

__device__ bool isInsideBird(float2 pixel, float2 pos, float width,
                             float height, float radius) {
  if (abs(pixel.x - pos.x) < 0.5f * width &&
          abs(pixel.y - pos.y) < 0.5f * height ||
      (pixel.x - pos.x) * (pixel.x - pos.x) +
              (pixel.y - pos.y) * (pixel.y - pos.y) <
          radius * radius) {
    return true;
  } else {
    return false;
  }
}

__global__ void cuda_kernel_update(float2 *newPos, float2 *curPos,
                                   uint numBirds, bool *hasproxy,
                                   bool *neighbors, bool *rightgoals,
                                   bool *leftgoals, Params *params) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= numBirds) {
    return;
  }

  float minDist = 50000.f;
  float2 dij = make_float2(0.f);

  if (!hasproxy[i]) {
    for (uint j = 0; j < numBirds; j++) {
      if (j == i) {
        continue;
      }

      if (leftgoals[i * numBirds + j]) {
        dij = params->dX * normalize(curPos[j] - curPos[i]);
        break;
      }
    }
  } else {
    bool collision = false;

    for (uint j = 0; j < numBirds; j++) {
      float d;

      if (leftgoals[i * numBirds + j]) {
        d = curPos[j].x - (params->wingspan + params->lambda) - curPos[i].x;

        if (fabs(d) < fabs(minDist)) {
          minDist = d;
        }
      }

      if (rightgoals[i * numBirds + j]) {
        d = curPos[j].x + (params->wingspan + params->lambda) - curPos[i].x;

        if (fabs(d) < fabs(minDist)) {
          minDist = d;
        }
      }

      if (neighbors[i * numBirds + j] && !collision) {
        if (curPos[j].y >= curPos[i].y &&
            curPos[j].y < curPos[i].y + params->epsilon) {
          dij.y = -params->dY;
          collision = true;
        }
      }
    }

    if (fabs(minDist) <= params->dX) {
      return;
    }

    dij.x = minDist > 0 ? params->dX : -params->dX;
  }

  newPos[i].x = curPos[i].x + dij.x;
  newPos[i].y = curPos[i].y + dij.y;
}

__global__ void cuda_kernel_checktriples(float2 *pos, uint numBirds,
                                         bool *hasproxy, bool *neighbors,
                                         bool *rightgoals, bool *leftgoals,
                                         uint3 *triples, Params *params) {
  uint ith = blockIdx.x * blockDim.x + threadIdx.x;

  if (ith >= numBirds * (numBirds - 1) * (numBirds - 2) / 6) {
    return;
  }

  uint a[3];
  a[0] = triples[ith].x;
  a[1] = triples[ith].y;
  a[2] = triples[ith].z;

  uint i, j, x;

  for (i = 0; i < 3; i++) {
    for (j = 2; j > i; j--) {
      if (pos[a[j - 1]].y > pos[a[j]].y) {
        x = a[j - 1];
        a[j - 1] = a[j];
        a[j] = x;
      }
    }
  }

  if (hasproxy[a[0]]) {
    float a2a1 = pos[a[2]].x - pos[a[1]].x;

    if (fabs(a2a1) < 2.f * (params->wingspan + params->lambda))
      if (a2a1 >= 0) {
        if (leftgoals[a[0] * numBirds + a[2]]) {
          leftgoals[a[0] * numBirds + a[2]] = false;
        }

        if (rightgoals[a[0] * numBirds + a[1]]) {
          rightgoals[a[0] * numBirds + a[1]] = false;
        }
      } else {
        if (leftgoals[a[0] * numBirds + a[1]]) {
          leftgoals[a[0] * numBirds + a[1]] = false;
        }

        if (rightgoals[a[0] * numBirds + a[2]]) {
          rightgoals[a[0] * numBirds + a[2]] = false;
        }
      }
  } else {
    if ((leftgoals[a[0] * numBirds + a[2]]) &&
        (leftgoals[a[0] * numBirds + a[1]]))
      if ((length(pos[a[1]] - pos[a[0]]) < length(pos[a[2]] - pos[a[0]]))) {
        leftgoals[a[0] * numBirds + a[2]] = false;
      } else {
        leftgoals[a[0] * numBirds + a[1]] = false;
      }
  }
}

__global__ void cuda_kernel_checkpairs(float2 *pos, uint numBirds,
                                       bool *hasproxy, bool *neighbors,
                                       bool *rightgoals, bool *leftgoals,
                                       uint2 *pairs, Params *params) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= numBirds * (numBirds - 1) / 2) {
    return;
  }

  uint front, back;

  if (pos[pairs[i].y].y > pos[pairs[i].x].y) {
    front = pairs[i].y;
    back = pairs[i].x;
  } else {
    front = pairs[i].x;
    back = pairs[i].y;
  }

  leftgoals[back * numBirds + front] = true;
  rightgoals[back * numBirds + front] = true;

  float2 stepback;
  stepback.x = pos[front].x;
  stepback.y = pos[front].y - 0.5f * params->upwashY;

  if (isInsideQuad_D(
          pos[back], stepback,
          2.f * (params->wingspan + params->lambda + params->upwashX),
          params->upwashY)) {
    neighbors[back * numBirds + front] = true;

    if (!hasproxy[back]) {
      hasproxy[back] = true;
    }
  }
}

extern "C" void cuda_simulate(float2 *newPos, float2 *curPos, uint numBirds,
                              bool *d_hasproxy, bool *d_neighbors,
                              bool *d_leftgoals, bool *d_rightgoals,
                              uint2 *d_pairs, uint3 *d_triples,
                              Params *d_params) {
  cudaError_t error = cudaSuccess;
  float tempms;
  static float ms = 0.f;
  static uint step = 0;
  int smallblockSize = 32, midblockSize = 128, bigblockSize = 32;

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, 0);

  cudaMemset(d_leftgoals, 0, numBirds * numBirds * sizeof(bool));
  cudaMemset(d_rightgoals, 0, numBirds * numBirds * sizeof(bool));
  cudaMemset(d_hasproxy, 0, numBirds * sizeof(bool));
  cudaMemset(d_neighbors, 0, numBirds * numBirds * sizeof(bool));

  dim3 Db = dim3(bigblockSize);
  dim3 Dg =
      dim3((numBirds * (numBirds - 1) / 2 + bigblockSize - 1) / bigblockSize);
  cuda_kernel_checkpairs<<<Dg, Db>>>(curPos, numBirds, d_hasproxy, d_neighbors,
                                     d_rightgoals, d_leftgoals, d_pairs,
                                     d_params);

  Db = dim3(midblockSize);
  Dg =
      dim3((numBirds * (numBirds - 1) * (numBirds - 2) / 6 + bigblockSize - 1) /
           bigblockSize);
  cuda_kernel_checktriples<<<Dg, Db>>>(curPos, numBirds, d_hasproxy,
                                       d_neighbors, d_rightgoals, d_leftgoals,
                                       d_triples, d_params);

  Db = dim3(smallblockSize);
  Dg = dim3((numBirds + smallblockSize - 1) / smallblockSize);
  cuda_kernel_update<<<Dg, Db>>>(newPos, curPos, numBirds, d_hasproxy,
                                 d_neighbors, d_rightgoals, d_leftgoals,
                                 d_params /*, d_pWingTips */);

  cudaDeviceSynchronize();

  cudaEventRecord(e_stop, 0);
  cudaEventSynchronize(e_stop);
  cudaEventElapsedTime(&tempms, e_start, e_stop);
  ms += tempms;

  if (!(step % 100) && step) {
    printf("GPU, step %d \ntime per step %6.3f ms \n", step, ms / 100.f);
    ms = 0.f;
  }

  step++;

  error = cudaGetLastError();

  if (error != cudaSuccess) {
    printf("one of the cuda kernels failed to launch, error = %d\n", error);
  }
}
