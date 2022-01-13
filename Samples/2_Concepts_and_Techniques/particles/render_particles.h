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

#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer {
 public:
  ParticleRenderer();
  ~ParticleRenderer();

  void setPositions(float *pos, int numParticles);
  void setVertexBuffer(unsigned int vbo, int numParticles);
  void setColorBuffer(unsigned int vbo) { m_colorVBO = vbo; }

  enum DisplayMode { PARTICLE_POINTS, PARTICLE_SPHERES, PARTICLE_NUM_MODES };

  void display(DisplayMode mode = PARTICLE_POINTS);
  void displayGrid();

  void setPointSize(float size) { m_pointSize = size; }
  void setParticleRadius(float r) { m_particleRadius = r; }
  void setFOV(float fov) { m_fov = fov; }
  void setWindowSize(int w, int h) {
    m_window_w = w;
    m_window_h = h;
  }

 protected:  // methods
  void _initGL();
  void _drawPoints();
  GLuint _compileProgram(const char *vsource, const char *fsource);

 protected:  // data
  float *m_pos;
  int m_numParticles;

  float m_pointSize;
  float m_particleRadius;
  float m_fov;
  int m_window_w, m_window_h;

  GLuint m_program;

  GLuint m_vbo;
  GLuint m_colorVBO;
};

#endif  //__ RENDER_PARTICLES__
