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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem {
 public:
  ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
  ~ParticleSystem();

  enum ParticleConfig { CONFIG_RANDOM, CONFIG_GRID, _NUM_CONFIGS };

  enum ParticleArray {
    POSITION,
    VELOCITY,
  };

  void update(float deltaTime);
  void reset(ParticleConfig config);

  float *getArray(ParticleArray array);
  void setArray(ParticleArray array, const float *data, int start, int count);

  int getNumParticles() const { return m_numParticles; }

  unsigned int getCurrentReadBuffer() const { return m_posVbo; }
  unsigned int getColorBuffer() const { return m_colorVBO; }

  void *getCudaPosVBO() const { return (void *)m_cudaPosVBO; }
  void *getCudaColorVBO() const { return (void *)m_cudaColorVBO; }

  void dumpGrid();
  void dumpParticles(uint start, uint count);

  void setIterations(int i) { m_solverIterations = i; }

  void setDamping(float x) { m_params.globalDamping = x; }
  void setGravity(float x) { m_params.gravity = make_float3(0.0f, x, 0.0f); }

  void setCollideSpring(float x) { m_params.spring = x; }
  void setCollideDamping(float x) { m_params.damping = x; }
  void setCollideShear(float x) { m_params.shear = x; }
  void setCollideAttraction(float x) { m_params.attraction = x; }

  void setColliderPos(float3 x) { m_params.colliderPos = x; }

  float getParticleRadius() { return m_params.particleRadius; }
  float3 getColliderPos() { return m_params.colliderPos; }
  float getColliderRadius() { return m_params.colliderRadius; }
  uint3 getGridSize() { return m_params.gridSize; }
  float3 getWorldOrigin() { return m_params.worldOrigin; }
  float3 getCellSize() { return m_params.cellSize; }

  void addSphere(int index, float *pos, float *vel, int r, float spacing);

 protected:  // methods
  ParticleSystem() {}
  uint createVBO(uint size);

  void _initialize(int numParticles);
  void _finalize();

  void initGrid(uint *size, float spacing, float jitter, uint numParticles);

 protected:  // data
  bool m_bInitialized, m_bUseOpenGL;
  uint m_numParticles;

  // CPU data
  float *m_hPos;  // particle positions
  float *m_hVel;  // particle velocities

  uint *m_hParticleHash;
  uint *m_hCellStart;
  uint *m_hCellEnd;

  // GPU data
  float *m_dPos;
  float *m_dVel;

  float *m_dSortedPos;
  float *m_dSortedVel;

  // grid data for sorting method
  uint *m_dGridParticleHash;   // grid hash value for each particle
  uint *m_dGridParticleIndex;  // particle index for each particle
  uint *m_dCellStart;          // index of start of each cell in sorted list
  uint *m_dCellEnd;            // index of end of cell

  uint m_gridSortBits;

  uint m_posVbo;    // vertex buffer object for particle positions
  uint m_colorVBO;  // vertex buffer object for colors

  float *m_cudaPosVBO;    // these are the CUDA deviceMem Pos
  float *m_cudaColorVBO;  // these are the CUDA deviceMem Color

  struct cudaGraphicsResource
      *m_cuda_posvbo_resource;  // handles OpenGL-CUDA exchange
  struct cudaGraphicsResource
      *m_cuda_colorvbo_resource;  // handles OpenGL-CUDA exchange

  // params
  SimParams m_params;
  uint3 m_gridSize;
  uint m_numGridCells;

  StopWatchInterface *m_timer;

  uint m_solverIterations;
};

#endif  // __PARTICLESYSTEM_H__
