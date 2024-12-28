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

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <math.h>
#include <assert.h>

void mat_identity(matrix4 m) {
  m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
      m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.0f;
  m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
}

void mat_multiply(matrix4 m0, matrix4 m1) {
  float m[4];

  for (int r = 0; r < 4; r++) {
    m[0] = m[1] = m[2] = m[3] = 0.0f;

    for (int c = 0; c < 4; c++) {
      for (int i = 0; i < 4; i++) {
        m[c] += m0[i][r] * m1[c][i];
      }
    }

    for (int c = 0; c < 4; c++) {
      m0[c][r] = m[c];
    }
  }
}

void mat_translate(matrix4 m, vector3 v) {
  matrix4 m2;
  m2[0][0] = m2[1][1] = m2[2][2] = m2[3][3] = 1.0f;
  m2[0][1] = m2[0][2] = m2[0][3] = m2[1][0] = m2[1][2] = m2[1][3] = m2[2][0] =
      m2[2][1] = m2[2][3] = 0.0f;
  m2[3][0] = v[0];
  m2[3][1] = v[1];
  m2[3][2] = v[2];
  mat_multiply(m, m2);
}

void mat_perspective(matrix4 m, GLfloat fovy, GLfloat aspect, GLfloat znear,
                     GLfloat zfar) {
  matrix4 m2;
  m2[1][0] = m2[2][0] = m2[3][0] = m2[0][1] = m2[2][1] = m2[3][1] = m2[0][2] =
      m2[1][2] = m2[0][3] = m2[1][3] = m2[3][3] = 0.0f;
  m2[2][3] = -1.0f;

  float f = 1 / tan((fovy * M_PI / 180) / 2);
  m2[0][0] = f / aspect;
  m2[1][1] = f;

  m2[2][2] = ((znear + zfar) / (znear - zfar));
  m2[3][2] = ((2 * znear * zfar) / (znear - zfar));

  mat_multiply(m, m2);
}

ParticleRenderer::ParticleRenderer(unsigned int windowWidth,
                                   unsigned int windowHeight)
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_spriteSize(2.0f),
      m_vertexShader(0),
      m_vertexShaderPoints(0),
      m_fragmentShader(0),
      m_programPoints(0),
      m_programSprites(0),
      m_texture(0),
      m_pbo(0),
      m_vboColor(0),
      m_windowWidth(windowWidth),
      m_windowHeight(windowHeight),
      m_bFp64Positions(false) {
  m_camera[0] = 0;
  m_camera[1] = 0;
  m_camera[2] = 0;
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
  checkGLErrors("Setting particle float position");
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
  checkGLErrors("Setting particle double position");
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

void ParticleRenderer::display() {
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  glDepthMask(GL_FALSE);

  glUseProgram(m_programSprites);

  // Set modelview and projection matrices
  GLint h_ModelViewMatrix = glGetUniformLocation(m_programSprites, "modelview");
  GLint h_ProjectionMatrix =
      glGetUniformLocation(m_programSprites, "projection");
  matrix4 modelview;
  matrix4 projection;
  mat_identity(modelview);
  mat_identity(projection);
  mat_translate(modelview, m_camera);
  mat_perspective(projection, 60, (float)m_windowWidth / (float)m_windowHeight,
                  0.1, 1000.0);
  glUniformMatrix4fv(h_ModelViewMatrix, 1, GL_FALSE, (GLfloat *)modelview);
  glUniformMatrix4fv(h_ProjectionMatrix, 1, GL_FALSE, (GLfloat *)projection);

  // Set point size
  GLint h_PointSize = glGetUniformLocation(m_programSprites, "size");
  glUniform1f(h_PointSize, m_spriteSize);

  // Set base and secondary colors
  GLint h_BaseColor = glGetUniformLocation(m_programSprites, "baseColor");
  GLint h_SecondaryColor =
      glGetUniformLocation(m_programSprites, "secondaryColor");
  glUniform4f(h_BaseColor, 1.0, 1.0, 1.0, 1.0);
  glUniform4f(h_SecondaryColor, m_baseColor[0], m_baseColor[1], m_baseColor[2],
              m_baseColor[3]);

  // Set position coords
  GLint h_position = glGetAttribLocation(m_programSprites, "a_position");
  glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
  glEnableVertexAttribArray(h_position);
  glVertexAttribPointer(h_position, 4, GL_FLOAT, GL_FALSE, 0, 0);

  GLuint texLoc = glGetUniformLocation(m_programSprites, "splatTexture");
  glUniform1i(texLoc, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_texture);

  glDrawArrays(GL_POINTS, 0, m_numParticles);

  glDisableVertexAttribArray(h_position);

  glUseProgram(0);

  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);
}

const char vertexShader[] = {
    "attribute vec4 a_position;"

    "uniform mat4 projection;"
    "uniform mat4 modelview;"
    "uniform float size;"

    "void main()"
    "{"
    "float pointSize = 500.0 * size;"
    "vec4 vert = a_position;"
    "vert.w = 1.0;"
    "vec3 pos_eye = vec3(modelview * vert);"
    "gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));"
    "gl_Position = projection * modelview * a_position;"
    "}"};

const char fragmentShader[] = {
    "uniform sampler2D splatTexture;"
    "uniform lowp vec4 baseColor;"
    "uniform lowp vec4 secondaryColor;"

    "void main()"
    "{"
    "lowp vec4 textureColor = (0.6 + 0.4 * baseColor) * "
    "texture2D(splatTexture, gl_PointCoord);"
    "gl_FragColor = textureColor * secondaryColor;"
    "}"};

// Checks if the shader is compiled.
static int CheckCompiled(GLuint shader) {
  GLint isCompiled = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

  if (!isCompiled) {
    GLint infoLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);

    if (infoLen > 1) {
      char *infoLog = (char *)malloc(sizeof(char) * infoLen);

      glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
      printf("Error compiling program:\n%s\n", infoLog);
      free(infoLog);
    }

    return 0;
  }

  return 1;
}

void ParticleRenderer::_initGL() {
  m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
  m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

  const char *v = vertexShader;
  const char *f = fragmentShader;
  glShaderSource(m_vertexShader, 1, &v, 0);
  glShaderSource(m_fragmentShader, 1, &f, 0);

  checkGLErrors("Shader Source");

  glCompileShader(m_vertexShader);
  glCompileShader(m_fragmentShader);

  if (!CheckCompiled(m_vertexShader) || !CheckCompiled(m_fragmentShader)) {
    printf("A shader failed to compile.\n");
    exit(1);
  }

  m_programSprites = glCreateProgram();

  checkGLErrors("create program");

  glAttachShader(m_programSprites, m_vertexShader);
  glAttachShader(m_programSprites, m_fragmentShader);

  checkGLErrors("attaching shaders");

  glLinkProgram(m_programSprites);

  checkGLErrors("linking program");

  EGLint linked;
  glGetProgramiv(m_programSprites, GL_LINK_STATUS, &linked);
  if (!linked) {
    printf("A shader failed to link.\n");
    exit(1);
  }

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
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR);  //_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, data);
}
