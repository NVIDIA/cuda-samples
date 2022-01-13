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

#include <stdlib.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include "GLSLProgram.h"

GLSLProgram::GLSLProgram(const char *vsource, const char *fsource) {
  mProg = compileProgram(vsource, 0, fsource);
}

GLSLProgram::GLSLProgram(const char *vsource, const char *gsource,
                         const char *fsource, GLenum gsInput, GLenum gsOutput) {
  mProg = compileProgram(vsource, gsource, fsource, gsInput, gsOutput);
}

GLSLProgram::~GLSLProgram() {
  if (mProg) {
    glDeleteProgram(mProg);
  }
}

void GLSLProgram::enable() { glUseProgram(mProg); }

void GLSLProgram::disable() { glUseProgram(0); }

void GLSLProgram::setUniform1f(const char *name, float value) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc >= 0) {
    glUniform1f(loc, value);
  } else {
#if _DEBUG
    fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
  }
}

void GLSLProgram::setUniform2f(const char *name, float x, float y) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc >= 0) {
    glUniform2f(loc, x, y);
  } else {
#if _DEBUG
    fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
  }
}

void GLSLProgram::setUniform3f(const char *name, float x, float y, float z) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc >= 0) {
    glUniform3f(loc, x, y, z);
  } else {
#if _DEBUG
    fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
  }
}

void GLSLProgram::setUniform4f(const char *name, float x, float y, float z,
                               float w) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc >= 0) {
    glUniform4f(loc, x, y, z, w);
  } else {
#if _DEBUG
    fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
  }
}

void GLSLProgram::setUniformMatrix4fv(const GLchar *name, GLfloat *m,
                                      bool transpose) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc >= 0) {
    glUniformMatrix4fv(loc, 1, transpose, m);
  } else {
#if _DEBUG
    fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
  }
}

void GLSLProgram::setUniformfv(const GLchar *name, GLfloat *v, int elementSize,
                               int count) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc == -1) {
#ifdef _DEBUG
    fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
    return;
  }

  switch (elementSize) {
    case 1:
      glUniform1fv(loc, count, v);
      break;

    case 2:
      glUniform2fv(loc, count, v);
      break;

    case 3:
      glUniform3fv(loc, count, v);
      break;

    case 4:
      glUniform4fv(loc, count, v);
      break;
  }
}

void GLSLProgram::bindTexture(const char *name, GLuint tex, GLenum target,
                              GLint unit) {
  GLint loc = glGetUniformLocation(mProg, name);

  if (loc >= 0) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(target, tex);
    glUseProgram(mProg);
    glUniform1i(loc, unit);
    glActiveTexture(GL_TEXTURE0);
  } else {
#if _DEBUG
    fprintf(stderr, "Error binding texture '%s'\n", name);
#endif
  }
}

GLuint GLSLProgram::checkCompileStatus(GLuint shader, GLint *status) {
  glGetShaderiv(shader, GL_COMPILE_STATUS, status);

  if (!(*status)) {
    char log[2048];
    int len;
    glGetShaderInfoLog(shader, 2048, (GLsizei *)&len, log);
    printf("Error: shader(%04d), Info log: %s\n", (int)shader, log);
    glDeleteShader(shader);
    return 0;
  }

  return 1;
}

GLuint GLSLProgram::compileProgram(const char *vsource, const char *gsource,
                                   const char *fsource, GLenum gsInput,
                                   GLenum gsOutput) {
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

  GLint compiled = 0;

  glShaderSource(vertexShader, 1, &vsource, 0);
  glShaderSource(fragmentShader, 1, &fsource, 0);

  glCompileShader(vertexShader);

  if (checkCompileStatus(vertexShader, &compiled) == 0) {
    printf("<compileProgram compilation error with vertexShader>:\n");
    printf("%s\n", vsource);
    return 0;
  }

  glCompileShader(fragmentShader);

  if (checkCompileStatus(fragmentShader, &compiled) == 0) {
    printf("<compileProgram compilation error with fragmentShader>:\n");
    printf("%s\n", fsource);
    return 0;
  }

  GLuint program = glCreateProgram();

  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);

  if (gsource) {
    GLuint geomShader = glCreateShader(GL_GEOMETRY_SHADER_EXT);
    glShaderSource(geomShader, 1, &gsource, 0);
    glCompileShader(geomShader);
    glGetShaderiv(geomShader, GL_COMPILE_STATUS, (GLint *)&compiled);

    if (checkCompileStatus(geomShader, &compiled) == 0) {
      printf("<compileProgram compilation error with geomShader>:\n");
      printf("%s\n", gsource);
      return 0;
    }

    glAttachShader(program, geomShader);

    glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, gsInput);
    glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, gsOutput);
    glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_EXT, 4);
  }

  glLinkProgram(program);

  // check if program linked
  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);

  if (!success) {
    char temp[1024];
    glGetProgramInfoLog(program, 1024, 0, temp);
    fprintf(stderr, "Failed to link program:\n%s\n", temp);
    glDeleteProgram(program);
    program = 0;
    exit(EXIT_FAILURE);
  }

  return program;
}
