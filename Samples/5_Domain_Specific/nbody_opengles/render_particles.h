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

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <cstdio>

typedef float matrix4[4][4];
typedef float vector3[3];

// check for OpenGL errors
inline void checkGLErrors(const char *s) {
  EGLenum error;

  while ((error = glGetError()) != GL_NO_ERROR) {
    fprintf(stderr, "%s: error - %d\n", s, error);
  }
}

class ParticleRenderer {
 public:
  ParticleRenderer(unsigned int windowWidth = 720,
                   unsigned int windowHeight = 480);
  ~ParticleRenderer();

  void setPositions(float *pos, int numParticles);
  void setPositions(double *pos, int numParticles);
  void setBaseColor(float color[4]);
  void setColors(float *color, int numParticles);
  void setPBO(unsigned int pbo, int numParticles, bool fp64);

  enum DisplayMode {
    PARTICLE_POINTS,
    PARTICLE_SPRITES,
    PARTICLE_SPRITES_COLOR,
    PARTICLE_NUM_MODES
  };

  void display();

  void setPointSize(float size) { m_pointSize = size; }
  void setSpriteSize(float size) { m_spriteSize = size; }

  void setCameraPos(vector3 camera_pos) {
    m_camera[0] = camera_pos[0];
    m_camera[1] = camera_pos[1];
    m_camera[2] = camera_pos[2];
  }

  void resetPBO();

 protected:  // methods
  void _initGL();
  void _createTexture(int resolution);

 protected:  // data
  float *m_pos;
  double *m_pos_fp64;
  int m_numParticles;

  float m_pointSize;
  float m_spriteSize;
  vector3 m_camera;

  unsigned int m_vertexShader;
  unsigned int m_vertexShaderPoints;
  unsigned int m_fragmentShader;
  unsigned int m_programPoints;
  unsigned int m_programSprites;
  unsigned int m_texture;
  unsigned int m_pbo;
  unsigned int m_vboColor;
  unsigned int m_windowWidth;
  unsigned int m_windowHeight;

  float m_baseColor[4];

  bool m_bFp64Positions;
};

#endif  //__ RENDER_PARTICLES__
