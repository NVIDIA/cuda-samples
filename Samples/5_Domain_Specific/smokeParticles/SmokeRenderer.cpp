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

/*
    This class renders particles using OpenGL and GLSL shaders
*/

#include <math.h>
#include <stdlib.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include "SmokeRenderer.h"
#include "SmokeShaders.h"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#define USE_MBLUR 1
#define COLOR_ATTENUATION 1

SmokeRenderer::SmokeRenderer(int maxParticles)
    : mMaxParticles(maxParticles),
      mNumParticles(0),
      mPosVbo(0),
      mVelVbo(0),
      mColorVbo(0),
      mIndexBuffer(0),
      mParticleRadius(0.005f),
      mDisplayMode(VOLUMETRIC),
      mWindowW(800),
      mWindowH(600),
      mFov(40.0f),
      m_downSample(2),
      m_numSlices(32),
      m_numDisplayedSlices(32),
      m_sliceNo(0),
      m_shadowAlpha(0.005f),
      m_spriteAlpha(0.1f),
      m_doBlur(false),
      m_blurRadius(1.0f),
      m_displayLightBuffer(false),
      m_lightPos(5.0f, 5.0f, -5.0f),
      m_lightTarget(0.0f, 0.0f, 0.0f),
      m_lightColor(1.0f, 1.0f, 0.5f),
      m_colorAttenuation(0.1f, 0.2f, 0.3f),
      m_lightBufferSize(256),
      m_srcLightTexture(0),
      m_lightDepthTexture(0),
      m_lightFbo(0),
      m_imageTex(0),
      m_depthTex(0),
      m_imageFbo(0) {
  // load shader programs
  m_simpleProg = new GLSLProgram(particleVS, simplePS);
  m_particleProg = new GLSLProgram(mblurVS, mblurGS, particlePS);
  m_particleShadowProg = new GLSLProgram(mblurVS, mblurGS, particleShadowPS);

  m_blurProg = new GLSLProgram(passThruVS, blurPS);
  m_displayTexProg = new GLSLProgram(passThruVS, texture2DPS);

  // create buffer for light shadows
  createLightBuffer();

  glutReportErrors();
}

SmokeRenderer::~SmokeRenderer() {
  delete m_particleProg;
  delete m_particleShadowProg;
  delete m_blurProg;
  delete m_displayTexProg;
  delete m_simpleProg;

  delete m_lightFbo;
  glDeleteTextures(2, m_lightTexture);
  glDeleteTextures(1, &m_lightDepthTexture);

  delete m_imageFbo;
  glDeleteTextures(1, &m_imageTex);
  glDeleteTextures(1, &m_depthTex);
}

// draw points from vertex buffer objects
void SmokeRenderer::drawPoints(int start, int count, bool sort) {
  glBindBuffer(GL_ARRAY_BUFFER, mPosVbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  if (mColorVbo) {
    glBindBuffer(GL_ARRAY_BUFFER, mColorVbo);
    glColorPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
  }

  if (mVelVbo) {
    glBindBuffer(GL_ARRAY_BUFFER, mVelVbo);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  if (sort) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndexBuffer);
    glDrawElements(GL_POINTS, count, GL_UNSIGNED_INT,
                   (void *)(start * sizeof(unsigned int)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
  } else {
    glDrawArrays(GL_POINTS, start, count);
  }

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glClientActiveTexture(GL_TEXTURE0);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

// draw points using given shader program
void SmokeRenderer::drawPointSprites(GLSLProgram *prog, int start, int count,
                                     bool shadowed) {
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);  // don't write depth
  glEnable(GL_BLEND);

  prog->enable();
  prog->setUniform1f("pointRadius", mParticleRadius);

  if (shadowed) {
    prog->bindTexture("shadowTex", m_lightTexture[m_srcLightTexture],
                      GL_TEXTURE_2D, 0);
  }

  // draw points
  drawPoints(start, count, true);

  prog->disable();

  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
}

// calculate vectors for half-angle slice rendering
void SmokeRenderer::calcVectors() {
  // get model view matrix
  glGetFloatv(GL_MODELVIEW_MATRIX, (float *)m_modelView.get_value());

  // calculate eye space light vector
  m_lightVector = normalize(m_lightPos);
  m_lightPosEye = m_modelView * vec4f(m_lightPos, 1.0);

  // calculate half-angle vector between view and light
  m_viewVector = -vec3f(m_modelView.get_row(2));

  if (dot(m_viewVector, m_lightVector) > 0) {
    m_halfVector = normalize(m_viewVector + m_lightVector);
    m_invertedView = false;
  } else {
    m_halfVector = normalize(-m_viewVector + m_lightVector);
    m_invertedView = true;
  }

  // calculate light view matrix
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  gluLookAt(m_lightPos[0], m_lightPos[1], m_lightPos[2], m_lightTarget[0],
            m_lightTarget[1], m_lightTarget[2], 0.0, 1.0, 0.0);

  // calculate light projection matrix
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluPerspective(45.0, 1.0, 1.0, 200.0);

  glGetFloatv(GL_MODELVIEW_MATRIX, (float *)m_lightView.get_value());
  glGetFloatv(GL_PROJECTION_MATRIX, (float *)m_lightProj.get_value());

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // construct shadow matrix
  matrix4f scale;
  scale.set_scale(vec3f(0.5, 0.5, 0.5));
  matrix4f translate;
  translate.set_translate(vec3f(0.5, 0.5, 0.5));

  m_shadowMatrix =
      translate * scale * m_lightProj * m_lightView * inverse(m_modelView);

  // calc object space eye position
  m_eyePos = inverse(m_modelView) * vec4f(0.0, 0.0, 0.0, 1.0);

  // calc half vector in eye space
  m_halfVectorEye = m_modelView * vec4f(m_halfVector, 0.0);
}

// draw slice of particles from camera view
void SmokeRenderer::drawSlice(int i) {
  m_imageFbo->Bind();
  glViewport(0, 0, m_imageW, m_imageH);

  glColor4f(1.0, 1.0, 1.0, m_spriteAlpha);

  if (m_invertedView) {
    // front-to-back
    glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE);
  } else {
    // back-to-front
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  }

  drawPointSprites(m_particleShadowProg, i * m_batchSize, m_batchSize, true);

  m_imageFbo->Disable();
}

// draw slice of particles from light's point of view
void SmokeRenderer::drawSliceLightView(int i) {
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadMatrixf((GLfloat *)m_lightView.get_value());

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadMatrixf((GLfloat *)m_lightProj.get_value());

  m_lightFbo->Bind();
  glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

  glColor4f(m_colorAttenuation[0] * m_shadowAlpha,
            m_colorAttenuation[1] * m_shadowAlpha,
            m_colorAttenuation[2] * m_shadowAlpha, 1.0);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);

  drawPointSprites(m_particleProg, i * m_batchSize, m_batchSize, false);

  m_lightFbo->Disable();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

// draw particles as slices with shadowing
void SmokeRenderer::drawSlices() {
  m_batchSize = mNumParticles / m_numSlices;

  // clear light buffer
  m_srcLightTexture = 0;
  m_lightFbo->Bind();
  m_lightFbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture],
                            GL_COLOR_ATTACHMENT0_EXT);
  glClearColor(1.0f - m_lightColor[0], 1.0f - m_lightColor[1],
               1.0f - m_lightColor[2], 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  m_lightFbo->Disable();

  // clear volume image
  m_imageFbo->Bind();
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
  m_imageFbo->Disable();

  glActiveTexture(GL_TEXTURE0);
  glMatrixMode(GL_TEXTURE);
  glLoadMatrixf((GLfloat *)m_shadowMatrix.get_value());

  // render slices
  if (m_numDisplayedSlices > m_numSlices) m_numDisplayedSlices = m_numSlices;

  for (int i = 0; i < m_numDisplayedSlices; i++) {
    // draw slice from camera view, sampling light buffer
    drawSlice(i);
    // draw slice from light view to light buffer, accumulating shadows
    drawSliceLightView(i);

    if (m_doBlur) {
      blurLightBuffer();
    }
  }

  glActiveTexture(GL_TEXTURE0);
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
}

// blur light buffer to simulate scattering effects
void SmokeRenderer::blurLightBuffer() {
  m_lightFbo->Bind();
  m_lightFbo->AttachTexture(GL_TEXTURE_2D,
                            m_lightTexture[1 - m_srcLightTexture],
                            GL_COLOR_ATTACHMENT0_EXT);
  glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

  m_blurProg->enable();
  m_blurProg->bindTexture("tex", m_lightTexture[m_srcLightTexture],
                          GL_TEXTURE_2D, 0);
  m_blurProg->setUniform2f("texelSize", 1.0f / (float)m_lightBufferSize,
                           1.0f / (float)m_lightBufferSize);
  m_blurProg->setUniform1f("blurRadius", m_blurRadius);
  glDisable(GL_DEPTH_TEST);
  drawQuad();
  m_blurProg->disable();

  m_srcLightTexture = 1 - m_srcLightTexture;

  m_lightFbo->Disable();
}

// display texture to screen
void SmokeRenderer::displayTexture(GLuint tex) {
  m_displayTexProg->enable();
  m_displayTexProg->bindTexture("tex", tex, GL_TEXTURE_2D, 0);
  drawQuad();
  m_displayTexProg->disable();
}

// composite final volume image on top of scene
void SmokeRenderer::compositeResult() {
  glViewport(0, 0, mWindowW, mWindowH);
  glDisable(GL_DEPTH_TEST);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
  displayTexture(m_imageTex);
  glDisable(GL_BLEND);
}

void SmokeRenderer::render() {
  switch (mDisplayMode) {
    case POINTS:
      glColor3f(1.0, 1.0, 1.0);
      m_simpleProg->enable();
      drawPoints(0, mNumParticles, false);
      m_simpleProg->disable();
      break;

    case SPRITES:
      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
      glColor4f(1.0, 1.0, 1.0, m_spriteAlpha);
      drawPointSprites(m_particleProg, 0, mNumParticles, false);
      break;

    case VOLUMETRIC:
      drawSlices();
      compositeResult();
      break;

    case NUM_MODES:
      break;
  }

  if (m_displayLightBuffer) {
    // display light buffer to screen
    glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);
    glDisable(GL_DEPTH_TEST);
    displayTexture(m_lightTexture[m_srcLightTexture]);
    glViewport(0, 0, mWindowW, mWindowH);
  }

  glutReportErrors();
}

// render scene depth to texture
// (this is to ensure that particle are correctly occluded in the low-resolution
// render buffer)
void SmokeRenderer::beginSceneRender(Target target) {
  if (target == LIGHT_BUFFER) {
    m_lightFbo->Bind();
    glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf((GLfloat *)m_lightView.get_value());

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf((GLfloat *)m_lightProj.get_value());
  } else {
    m_imageFbo->Bind();
    glViewport(0, 0, m_imageW, m_imageH);
  }

  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glDepthMask(GL_TRUE);
  glClear(GL_DEPTH_BUFFER_BIT);
}

void SmokeRenderer::endSceneRender(Target target) {
  if (target == LIGHT_BUFFER) {
    m_lightFbo->Disable();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
  } else {
    m_imageFbo->Disable();
  }

  glViewport(0, 0, mWindowW, mWindowH);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}

// create an OpenGL texture
GLuint SmokeRenderer::createTexture(GLenum target, int w, int h,
                                    GLint internalformat, GLenum format) {
  GLuint texid;
  glGenTextures(1, &texid);
  glBindTexture(target, texid);

  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
  return texid;
}

// create buffers for off-screen rendering
void SmokeRenderer::createBuffers(int w, int h) {
  if (m_imageFbo) {
    glDeleteTextures(1, &m_imageTex);
    glDeleteTextures(1, &m_depthTex);
    delete m_imageFbo;
  }

  mWindowW = w;
  mWindowH = h;

  m_imageW = w / m_downSample;
  m_imageH = h / m_downSample;

  // create fbo for image buffer
  GLint format = GL_RGBA16F_ARB;
  // GLint format = GL_LUMINANCE16F_ARB;
  // GLint format = GL_RGBA8;
  m_imageTex =
      createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
  m_depthTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH,
                             GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  m_imageFbo = new FramebufferObject();
  m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_imageTex,
                            GL_COLOR_ATTACHMENT0_EXT);
  m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
  m_imageFbo->IsValid();
}

void SmokeRenderer::setLightColor(vec3f c) {
  m_lightColor = c;

  // set light texture border color
  GLfloat borderColor[4] = {1.0f - m_lightColor[0], 1.0f - m_lightColor[1],
                            1.0f - m_lightColor[2], 0.0f};

  glBindTexture(GL_TEXTURE_2D, m_lightTexture[0]);
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

  glBindTexture(GL_TEXTURE_2D, m_lightTexture[1]);
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

  glBindTexture(GL_TEXTURE_2D, 0);
}

// create FBOs for light buffer
void SmokeRenderer::createLightBuffer() {
  GLint format = GL_RGBA16F_ARB;
  // GLint format = GL_RGBA8;
  // GLint format = GL_LUMINANCE16F_ARB;

  m_lightTexture[0] = createTexture(GL_TEXTURE_2D, m_lightBufferSize,
                                    m_lightBufferSize, format, GL_RGBA);
  // make shadows clamp to light color at edges
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

  m_lightTexture[1] = createTexture(GL_TEXTURE_2D, m_lightBufferSize,
                                    m_lightBufferSize, format, GL_RGBA);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

  m_lightDepthTexture =
      createTexture(GL_TEXTURE_2D, m_lightBufferSize, m_lightBufferSize,
                    GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);

  m_lightFbo = new FramebufferObject();
  m_lightFbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture],
                            GL_COLOR_ATTACHMENT0_EXT);
  m_lightFbo->AttachTexture(GL_TEXTURE_2D, m_lightDepthTexture,
                            GL_DEPTH_ATTACHMENT_EXT);
  m_lightFbo->IsValid();
}

void SmokeRenderer::setWindowSize(int w, int h) {
  mAspect = (float)mWindowW / (float)mWindowH;
  mInvFocalLen = tan(mFov * 0.5f * NV_PI / 180.0f);

  createBuffers(w, h);
}

void SmokeRenderer::drawQuad() {
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, -1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, 1.0f);
  glEnd();
}

void SmokeRenderer::drawVector(vec3f v) {
  glBegin(GL_LINES);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3fv((float *)&v[0]);
  glEnd();
}

// render vectors to screen for debugging
void SmokeRenderer::debugVectors() {
  glColor3f(1.0f, 1.0f, 0.0f);
  drawVector(m_lightVector);

  glColor3f(0.0f, 1.0f, 0.0f);
  drawVector(m_viewVector);

  glColor3f(0.0f, 0.0f, 1.0f);
  drawVector(-m_viewVector);

  glColor3f(1.0f, 0.0f, 0.0f);
  drawVector(m_halfVector);
}
