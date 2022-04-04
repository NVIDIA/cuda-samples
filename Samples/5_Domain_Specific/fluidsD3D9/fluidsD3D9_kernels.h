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

#ifndef __STABLEFLUIDS_KERNELS_H_
#define __STABLEFLUIDS_KERNELS_H_

#define DIM 512              // Square size of solver domain
#define DS (DIM * DIM)       // Total domain size
#define CPADW (DIM / 2 + 1)  // Padded width for real->complex in-place FFT
#define RPADW \
  (2 * (DIM / 2 + 1))      // Padded width for real->complex in-place FFT
#define PDS (DIM * CPADW)  // Padded total domain size

#define DT 0.09f            // Delta T for interative solver
#define VIS 0.0025f         // Viscosity constant
#define FORCE (5.8f * DIM)  // Force scale factor
#define FR 4                // Force update radius

#define TILEX 64  // Tile width
#define TILEY 64  // Tile height
#define TIDSX 64  // Tids in X
#define TIDSY 4   // Tids in Y

typedef unsigned long DWORD;

typedef struct vertex {
  float x, y, z;
  DWORD c;
} Vertex;

// Vector data type used to velocity and force fields
typedef float2 cData;

extern "C" void setupTexture(int x, int y);
extern "C" void updateTexture(cData *data, size_t w, size_t h, size_t pitch);
extern "C" void deleteTexture(void);

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void addForces_k(cData *v, int dx, int dy, int spx, int spy,
                            float fx, float fy, int r, size_t pitch);

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void advectVelocity_k(cData *v, float *vx, float *vy, int dx,
                                 int pdx, int dy, float dt, int lb,
                                 cudaTextureObject_t tex);

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the wave wave vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt,
                                 float visc, int lb);

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.
__global__ void updateVelocity_k(cData *v, float *vx, float *vy, int dx,
                                 int pdx, int dy, int lb, size_t pitch);

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void advectParticles_k(Vertex *part, cData *v, int dx, int dy,
                                  float dt, int lb, size_t pitch);

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx,
                          float fy, int r, size_t tPitch);
extern "C" void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx,
                               int dy, float dt, size_t tPitch);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt,
                               float visc, size_t tPitch);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx,
                               int dy, size_t tPitch);
extern "C" void advectParticles(Vertex *p, cData *v, int dx, int dy, float dt,
                                size_t tPitch);

#endif
