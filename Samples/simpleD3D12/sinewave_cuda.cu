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

#include "ShaderStructs.h"

__global__ void sinewave_gen_kernel(Vertex *vertices, unsigned int width,
                                    unsigned int height, float time) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // calculate uv coordinates
  float u = x / (float)width;
  float v = y / (float)height;
  u = u * 2.0f - 1.0f;
  v = v * 2.0f - 1.0f;

  // calculate simple sine wave pattern
  float freq = 4.0f;
  float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

  if (y < height && x < width) {
    // write output vertex
    vertices[y * width + x].position.x = u;
    vertices[y * width + x].position.y = w;
    vertices[y * width + x].position.z = v;
    // vertices[y*width+x].position[3] = 1.0f;
    vertices[y * width + x].color.x = 1.0f;
    vertices[y * width + x].color.y = 0.0f;
    vertices[y * width + x].color.z = 0.0f;
    vertices[y * width + x].color.w = 0.0f;
  }
}

// The host CPU Sinewave thread spawner
void RunSineWaveKernel(size_t mesh_width, size_t mesh_height,
                       Vertex *cudaDevVertptr, cudaStream_t streamToRun,
                       float AnimTime) {
  dim3 block(16, 16, 1);
  dim3 grid(mesh_width / 16, mesh_height / 16, 1);
  Vertex *vertices = (Vertex *)cudaDevVertptr;
  sinewave_gen_kernel<<<grid, block, 0, streamToRun>>>(vertices, mesh_width,
                                                       mesh_height, AnimTime);

  getLastCudaError("sinewave_gen_kernel execution failed.\n");
}
