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

#include "render_particles.h"

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <math.h>
#include <assert.h>

#define GL_POINT_SPRITE_ARB 0x8861
#define GL_COORD_REPLACE_ARB 0x8862
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_spriteSize(2.0f),
      m_vertexShader(0),
      m_vertexShaderPoints(0),
      m_pixelShader(0),
      m_programPoints(0),
      m_programSprites(0),
      m_texture(0),
      m_pbo(0),
      m_vboColor(0),
      m_bFp64Positions(false) {
  _initGL();
}

ParticleRenderer::~ParticleRenderer() { m_pos = 0; }

void ParticleRenderer::resetPBO() { glDeleteBuffers(1, (GLuint *)&m_pbo); }

void ParticleRenderer::setPositions(float *pos, int numParticles) {
  m_pos = pos;
  m_numParticles = numParticles;

  if (!m_pbo) {
    glGenBuffers(1, (GLuint *)&m_pbo);
  }

  glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
  glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(float), pos,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  SDK_CHECK_ERROR_GL();
}

void ParticleRenderer::setPositions(double *pos, int numParticles) {
  m_bFp64Positions = true;
  m_pos_fp64 = pos;
  m_numParticles = numParticles;

  if (!m_pbo) {
    glGenBuffers(1, (GLuint *)&m_pbo);
  }

  glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
  glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(double), pos,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  SDK_CHECK_ERROR_GL();
}

void ParticleRenderer::setColors(float *color, int numParticles) {
  glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
  glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(float), color,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleRenderer::setBaseColor(float color[4]) {
  for (int i = 0; i < 4; i++) m_baseColor[i] = color[i];
}

void ParticleRenderer::setPBO(unsigned int pbo, int numParticles, bool fp64) {
  m_pbo = pbo;
  m_numParticles = numParticles;

  if (fp64) m_bFp64Positions = true;
}

void ParticleRenderer::_drawPoints(bool color) {
  if (!m_pbo) {
    glBegin(GL_POINTS);
    {
      int k = 0;

      for (int i = 0; i < m_numParticles; ++i) {
        if (m_bFp64Positions)
          glVertex3dv(&m_pos_fp64[k]);
        else {
          glVertex3fv(&m_pos[k]);
        }

        k += 4;
      }
    }
    glEnd();
  } else {
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);

    if (m_bFp64Positions)
      glVertexPointer(4, GL_DOUBLE, 0, 0);
    else
      glVertexPointer(4, GL_FLOAT, 0, 0);

    if (color) {
      glEnableClientState(GL_COLOR_ARRAY);
      glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
      // glActiveTexture(GL_TEXTURE1);
      // glTexCoordPointer(4, GL_FLOAT, 0, 0);
      glColorPointer(4, GL_FLOAT, 0, 0);
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
      glUseProgram(m_programPoints);
      _drawPoints();
      glUseProgram(0);
      break;

    case PARTICLE_SPRITES:
    default: {
      // setup point sprites
      glEnable(GL_POINT_SPRITE_ARB);
      glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
      glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
      glPointSize(m_spriteSize);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);

      glUseProgram(m_programSprites);
      GLuint texLoc = glGetUniformLocation(m_programSprites, "splatTexture");
      glUniform1i(texLoc, 0);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_texture);

      glColor3f(1, 1, 1);
      glSecondaryColor3fv(m_baseColor);

      _drawPoints();

      glUseProgram(0);

      glDisable(GL_POINT_SPRITE_ARB);
      glDisable(GL_BLEND);
      glDepthMask(GL_TRUE);
    }

    break;

    case PARTICLE_SPRITES_COLOR: {
      // setup point sprites
      glEnable(GL_POINT_SPRITE_ARB);
      glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
      glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
      glPointSize(m_spriteSize);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);

      glUseProgram(m_programSprites);
      GLuint texLoc = glGetUniformLocation(m_programSprites, "splatTexture");
      glUniform1i(texLoc, 0);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_texture);

      glColor3f(1, 1, 1);
      glSecondaryColor3fv(m_baseColor);

      _drawPoints(true);

      glUseProgram(0);

      glDisable(GL_POINT_SPRITE_ARB);
      glDisable(GL_BLEND);
      glDepthMask(GL_TRUE);
    }

    break;
  }

  SDK_CHECK_ERROR_GL();
}

const char vertexShaderPoints[] = {
    "void main()                                                            \n"
    "{                                                                      \n"
    "    vec4 vert = vec4(gl_Vertex.xyz, 1.0);  			       "
    " "
    "              \n"
    "    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;        "
    "                   \n"
    "    gl_FrontColor = gl_Color;                                          \n"
    "}                                                                      "
    "\n"};

const char vertexShader[] = {
    "void main()                                                            \n"
    "{                                                                      \n"
    "    float pointSize = 500.0 * gl_Point.size;                           \n"
    "    vec4 vert = gl_Vertex;						"
    "						\n"
    "    vert.w = 1.0;							"
    "	"
    "						\n"
    "    vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);                   \n"
    "    gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));            \n"
    "    gl_TexCoord[0] = gl_MultiTexCoord0;                                \n"
    //"    gl_TexCoord[1] = gl_MultiTexCoord1; \n"
    "    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;     \n"
    "    gl_FrontColor = gl_Color;                                          \n"
    "    gl_FrontSecondaryColor = gl_SecondaryColor;                        \n"
    "}                                                                      "
    "\n"};

const char pixelShader[] = {
    "uniform sampler2D splatTexture;                                        \n"

    "void main()                                                            \n"
    "{                                                                      \n"
    "    vec4 color2 = gl_SecondaryColor;                                   \n"
    "    vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, "
    "gl_TexCoord[0].st); \n"
    "    gl_FragColor =                                                     \n"
    "         color * color2;\n"  // mix(vec4(0.1, 0.0, 0.0, color.w), color2,
                                  // color.w);\n"
    "}                                                                      "
    "\n"};

void ParticleRenderer::_initGL() {
  m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
  m_vertexShaderPoints = glCreateShader(GL_VERTEX_SHADER);
  m_pixelShader = glCreateShader(GL_FRAGMENT_SHADER);

  const char *v = vertexShader;
  const char *p = pixelShader;
  glShaderSource(m_vertexShader, 1, &v, 0);
  glShaderSource(m_pixelShader, 1, &p, 0);
  const char *vp = vertexShaderPoints;
  glShaderSource(m_vertexShaderPoints, 1, &vp, 0);

  glCompileShader(m_vertexShader);
  glCompileShader(m_vertexShaderPoints);
  glCompileShader(m_pixelShader);

  m_programSprites = glCreateProgram();
  glAttachShader(m_programSprites, m_vertexShader);
  glAttachShader(m_programSprites, m_pixelShader);
  glLinkProgram(m_programSprites);

  m_programPoints = glCreateProgram();
  glAttachShader(m_programPoints, m_vertexShaderPoints);
  glLinkProgram(m_programPoints);

  _createTexture(32);

  glGenBuffers(1, (GLuint *)&m_vboColor);
  glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
  glBufferData(GL_ARRAY_BUFFER, m_numParticles * 4 * sizeof(float), 0,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//------------------------------------------------------------------------------
// Function           : EvalHermite
// Description      :
//------------------------------------------------------------------------------
/**
* EvalHermite(float pA, float pB, float vA, float vB, float u)
* @brief Evaluates Hermite basis functions for the specified coefficients.
*/
inline float evalHermite(float pA, float pB, float vA, float vB, float u) {
  float u2 = (u * u), u3 = u2 * u;
  float B0 = 2 * u3 - 3 * u2 + 1;
  float B1 = -2 * u3 + 3 * u2;
  float B2 = u3 - 2 * u2 + u;
  float B3 = u3 - u;
  return (B0 * pA + B1 * pB + B2 * vA + B3 * vB);
}

unsigned char *createGaussianMap(int N) {
  float *M = new float[2 * N * N];
  unsigned char *B = new unsigned char[4 * N * N];
  float X, Y, Y2, Dist;
  float Incr = 2.0f / N;
  int i = 0;
  int j = 0;
  Y = -1.0f;

  // float mmax = 0;
  for (int y = 0; y < N; y++, Y += Incr) {
    Y2 = Y * Y;
    X = -1.0f;

    for (int x = 0; x < N; x++, X += Incr, i += 2, j += 4) {
      Dist = (float)sqrtf(X * X + Y2);

      if (Dist > 1) Dist = 1;

      M[i + 1] = M[i] = evalHermite(1.0f, 0, 0, 0, Dist);
      B[j + 3] = B[j + 2] = B[j + 1] = B[j] = (unsigned char)(M[i] * 255);
    }
  }

  delete[] M;
  return (B);
}

void ParticleRenderer::_createTexture(int resolution) {
  unsigned char *data = createGaussianMap(resolution);
  glGenTextures(1, (GLuint *)&m_texture);
  glBindTexture(GL_TEXTURE_2D, m_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, data);
}
