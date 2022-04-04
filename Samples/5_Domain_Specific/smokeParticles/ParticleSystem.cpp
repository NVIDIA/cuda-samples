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

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include "particles_kernel.cuh"

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

/*
    This handles the particle simulation using CUDA
*/

ParticleSystem::ParticleSystem(uint numParticles, bool bUseVBO, bool bUseGL)
    : m_bInitialized(false),
      m_bUseVBO(bUseVBO),
      m_numParticles(numParticles),
      m_particleRadius(0.1f),
      m_doDepthSort(false),
      m_timer(NULL),
      m_time(0.0f) {
  m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);
  m_params.globalDamping = 1.0f;
  m_params.noiseSpeed = make_float3(0.0f, 0.0f, 0.0f);

  _initialize(numParticles, bUseGL);
}

ParticleSystem::~ParticleSystem() {
  _free();
  m_numParticles = 0;
}

void ParticleSystem::_initialize(int numParticles, bool bUseGL) {
  assert(!m_bInitialized);

  createNoiseTexture(64, 64, 64);

  m_numParticles = numParticles;

  // allocate GPU arrays
  m_pos.alloc(m_numParticles, m_bUseVBO, true);  // create as VBO
  m_vel.alloc(m_numParticles, m_bUseVBO, true);

  m_sortKeys.alloc(m_numParticles);
  m_indices.alloc(m_numParticles, m_bUseVBO, false,
                  true);  // create as index buffer

  sdkCreateTimer(&m_timer);
  setParameters(&m_params);

  m_bInitialized = true;
}

void ParticleSystem::_free() { assert(m_bInitialized); }

// step the simulation
void ParticleSystem::step(float deltaTime) {
  assert(m_bInitialized);

  m_params.time = m_time;
  setParameters(&m_params);

  m_pos.map();
  m_vel.map();

  // integrate particles
  integrateSystem(m_pos.getDevicePtr(), m_pos.getDeviceWritePtr(),
                  m_vel.getDevicePtr(), m_vel.getDeviceWritePtr(), deltaTime,
                  m_numParticles);

  m_pos.unmap();
  m_vel.unmap();

  m_pos.swap();
  m_vel.swap();

  m_time += deltaTime;
}

// depth sort the particles
void ParticleSystem::depthSort() {
  if (!m_doDepthSort) {
    return;
  }

  m_pos.map();
  m_indices.map();

  // calculate depth
  calcDepth(m_pos.getDevicePtr(), m_sortKeys.getDevicePtr(),
            m_indices.getDevicePtr(), m_sortVector, m_numParticles);

  // radix sort
  sortParticles(m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(),
                m_numParticles);

  m_pos.unmap();
  m_indices.unmap();
}

uint *ParticleSystem::getSortedIndices() {
  // copy sorted indices back to CPU
  m_indices.copy(GpuArray<uint>::DEVICE_TO_HOST);
  return m_indices.getHostPtr();
}

// random float [0, 1]
inline float frand() { return rand() / (float)RAND_MAX; }

// signed random float [-1, 1]
inline float sfrand() { return frand() * 2.0f - 1.0f; }

// random signed vector
inline vec3f svrand() { return vec3f(sfrand(), sfrand(), sfrand()); }

// random point in circle
inline vec2f randCircle() {
  vec2f r;

  do {
    r = vec2f(sfrand(), sfrand());
  } while (length(r) > 1.0f);

  return r;
}

// random point in sphere
inline vec3f randSphere() {
  vec3f r;

  do {
    r = svrand();
  } while (length(r) > 1.0f);

  return r;
}

// initialize in regular grid
void ParticleSystem::initGrid(vec3f start, uint3 size, vec3f spacing,
                              float jitter, vec3f vel, uint numParticles,
                              float lifetime) {
  srand(1973);

  float4 *posPtr = m_pos.getHostPtr();
  float4 *velPtr = m_vel.getHostPtr();

  for (uint z = 0; z < size.z; z++) {
    for (uint y = 0; y < size.y; y++) {
      for (uint x = 0; x < size.x; x++) {
        uint i = (z * size.y * size.x) + (y * size.x) + x;

        if (i < numParticles) {
          vec3f pos = start + spacing * vec3f((float)x, (float)y, (float)z) +
                      svrand() * jitter;

          posPtr[i] = make_float4(pos.x, pos.y, pos.z, 0.0f);
          velPtr[i] = make_float4(vel.x, vel.y, vel.z, lifetime);
        }
      }
    }
  }
}

// initialize in random positions within cube
void ParticleSystem::initCubeRandom(vec3f origin, vec3f size, vec3f vel,
                                    float lifetime) {
  float4 *posPtr = m_pos.getHostPtr();
  float4 *velPtr = m_vel.getHostPtr();

  for (uint i = 0; i < m_numParticles; i++) {
    vec3f pos = origin + svrand() * size;
    posPtr[i] = make_float4(pos.x, pos.y, pos.z, 0.0f);
    velPtr[i] = make_float4(vel.x, vel.y, vel.z, lifetime);
  }
}

// add sphere on regular grid
void ParticleSystem::addSphere(uint &index, vec3f pos, vec3f vel, int r,
                               float spacing, float jitter, float lifetime) {
  float4 *posPtr = m_pos.getHostPtr();
  float4 *velPtr = m_vel.getHostPtr();

  uint start = index;
  uint count = 0;

  for (int z = -r; z <= r; z++) {
    for (int y = -r; y <= r; y++) {
      for (int x = -r; x <= r; x++) {
        vec3f delta = vec3f((float)x, (float)y, (float)z) * spacing;
        float dist = length(delta);

        if ((dist <= spacing * r) && (index < m_numParticles)) {
          // vec3f p = pos + delta + svrand()*jitter;

          posPtr[index] = make_float4(pos.x, pos.y, pos.z, 0.0f);
          velPtr[index] = make_float4(vel.x, vel.y, vel.z, lifetime);

          index++;
          count++;
        }
      }
    }
  }

  m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
  m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
}

void ParticleSystem::reset(ParticleConfig config) {
  switch (config) {
    default:
    case CONFIG_RANDOM:
      initCubeRandom(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 1.0), vec3f(0.0f),
                     100.0);
      break;

    case CONFIG_GRID: {
      float jitter = m_particleRadius * 0.01f;
      uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
      uint gridSize[3];
      gridSize[0] = gridSize[1] = gridSize[2] = s;
      initGrid(vec3f(-1.0, 0.0, -1.0), make_uint3(s, s, s),
               vec3f(m_particleRadius * 2.0f), jitter, vec3f(0.0),
               m_numParticles, 100.0);
    } break;
  }

  m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE);
  m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE);
}

// particle emitters
void ParticleSystem::discEmitter(uint &index, vec3f pos, vec3f vel, vec3f vx,
                                 vec3f vy, float r, int n, float lifetime,
                                 float lifetimeVariance) {
  float4 *posPtr = m_pos.getHostPtr();
  float4 *velPtr = m_vel.getHostPtr();

  uint start = index;
  uint count = 0;

  for (int i = 0; i < n; i++) {
    vec2f delta = randCircle() * r;

    if (index < m_numParticles) {
      vec3f p = pos + delta.x * vx + delta.y * vy;
      float lt = lifetime + frand() * lifetimeVariance;

      posPtr[index] = make_float4(p.x, p.y, p.z, 0.0f);
      velPtr[index] = make_float4(vel.x, vel.y, vel.z, lt);

      index++;
      count++;
    }
  }

  m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
  m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
}

void ParticleSystem::sphereEmitter(uint &index, vec3f pos, vec3f vel,
                                   vec3f spread, float r, int n, float lifetime,
                                   float lifetimeVariance) {
  float4 *posPtr = m_pos.getHostPtr();
  float4 *velPtr = m_vel.getHostPtr();

  uint start = index;
  uint count = 0;

  for (int i = 0; i < n; i++) {
    vec3f x = randSphere();

    // float dist = length(x);
    if (index < m_numParticles) {
      vec3f p = pos + x * r;
      float age = 0.0;

      float lt = lifetime + frand() * lifetimeVariance;

      vec3f dir = randSphere();
      dir.y = fabs(dir.y);
      vec3f v = vel + dir * spread;

      posPtr[index] = make_float4(p.x, p.y, p.z, age);
      velPtr[index] = make_float4(v.x, v.y, v.z, lt);

      index++;
      count++;
    }
  }

  m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
  m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
}

void ParticleSystem::setModelView(float *m) {
  for (int i = 0; i < 16; i++) {
    m_modelView.m[i] = m[i];
  }
}

// dump particles to stdout for debugging
void ParticleSystem::dumpParticles(uint start, uint count) {
  m_pos.copy(GpuArray<float4>::DEVICE_TO_HOST);
  float4 *pos = m_pos.getHostPtr();

  m_vel.copy(GpuArray<float4>::DEVICE_TO_HOST);
  float4 *vel = m_vel.getHostPtr();

  for (uint i = start; i < start + count; i++) {
    printf("%d: ", i);
    printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", pos[i].x, pos[i].y, pos[i].z,
           pos[i].w);
    printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", vel[i].x, vel[i].y, vel[i].z,
           vel[i].w);
  }
}

// dump particles to a system memory host
void ParticleSystem::dumpBin(float4 **posData, float4 **velData) {
  m_pos.copy(GpuArray<float4>::DEVICE_TO_HOST);
  *posData = m_pos.getHostPtr();

  m_vel.copy(GpuArray<float4>::DEVICE_TO_HOST);
  *velData = m_vel.getHostPtr();
}
