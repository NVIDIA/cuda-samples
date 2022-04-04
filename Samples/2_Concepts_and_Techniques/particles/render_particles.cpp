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

#include <math.h>
#include <assert.h>
#include <stdio.h>

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_particleRadius(0.125f * 0.5f),
      m_program(0),
      m_vbo(0),
      m_colorVBO(0) {
  _initGL();
}

ParticleRenderer::~ParticleRenderer() { m_pos = 0; }

void ParticleRenderer::setPositions(float *pos, int numParticles) {
  m_pos = pos;
  m_numParticles = numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles) {
  m_vbo = vbo;
  m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints() {
  if (!m_vbo) {
    glBegin(GL_POINTS);
    {
      int k = 0;

      for (int i = 0; i < m_numParticles; ++i) {
        glVertex3fv(&m_pos[k]);
        k += 4;
      }
    }
    glEnd();
  } else {
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    if (m_colorVBO) {
      glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
      glColorPointer(4, GL_FLOAT, 0, 0);
      glEnableClientState(GL_COLOR_ARRAY);
    }

    glDrawArrays(GL_POINTS, 0, m_numParticles);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
  }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */) {
  switch (mode) {
    case PARTICLE_POINTS:
      glColor3f(1, 1, 1);
      glPointSize(m_pointSize);
      _drawPoints();
      break;

    default:
    case PARTICLE_SPHERES:
      glEnable(GL_POINT_SPRITE_ARB);
      glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
      glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);

      glUseProgram(m_program);
      glUniform1f(glGetUniformLocation(m_program, "pointScale"),
                  m_window_h / tanf(m_fov * 0.5f * (float)M_PI / 180.0f));
      glUniform1f(glGetUniformLocation(m_program, "pointRadius"),
                  m_particleRadius);

      glColor3f(1, 1, 1);
      _drawPoints();

      glUseProgram(0);
      glDisable(GL_POINT_SPRITE_ARB);
      break;
  }
}

GLuint ParticleRenderer::_compileProgram(const char *vsource,
                                         const char *fsource) {
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

  glShaderSource(vertexShader, 1, &vsource, 0);
  glShaderSource(fragmentShader, 1, &fsource, 0);

  glCompileShader(vertexShader);
  glCompileShader(fragmentShader);

  GLuint program = glCreateProgram();

  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);

  glLinkProgram(program);

  // check if program linked
  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);

  if (!success) {
    char temp[256];
    glGetProgramInfoLog(program, 256, 0, temp);
    printf("Failed to link program:\n%s\n", temp);
    glDeleteProgram(program);
    program = 0;
  }

  return program;
}

void ParticleRenderer::_initGL() {
  m_program = _compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
  glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
  glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
