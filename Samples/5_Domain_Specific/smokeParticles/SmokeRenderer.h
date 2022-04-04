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

// Smoke particle renderer with volumetric shadows

#ifndef SMOKE_RENDERER_H
#define SMOKE_RENDERER_H

#include "framebufferObject.h"
#include "GLSLProgram.h"
#include "nvMath.h"

using namespace nv;

class SmokeRenderer {
 public:
  SmokeRenderer(int maxParticles);
  ~SmokeRenderer();

  enum DisplayMode { POINTS, SPRITES, VOLUMETRIC, NUM_MODES };

  enum Target { LIGHT_BUFFER, SCENE_BUFFER };

  void setDisplayMode(DisplayMode mode) { mDisplayMode = mode; }

  void setNumParticles(unsigned int x) { mNumParticles = x; }
  void setPositionBuffer(GLuint vbo) { mPosVbo = vbo; }
  void setVelocityBuffer(GLuint vbo) { mVelVbo = vbo; }
  void setColorBuffer(GLuint vbo) { mColorVbo = vbo; }
  void setIndexBuffer(GLuint ib) { mIndexBuffer = ib; }

  void setParticleRadius(float x) { mParticleRadius = x; }
  void setWindowSize(int w, int h);
  void setFOV(float fov) { mFov = fov; }

  void setNumSlices(int x) { m_numSlices = x; }
  void setNumDisplayedSlices(int x) { m_numDisplayedSlices = x; }

  void setAlpha(float x) { m_spriteAlpha = x; }
  void setShadowAlpha(float x) { m_shadowAlpha = x; }
  void setColorAttenuation(vec3f c) { m_colorAttenuation = c; }
  void setLightColor(vec3f c);

  void setDoBlur(bool b) { m_doBlur = b; }
  void setBlurRadius(float x) { m_blurRadius = x; }
  void setDisplayLightBuffer(bool b) { m_displayLightBuffer = b; }

  void beginSceneRender(Target target);
  void endSceneRender(Target target);

  void setLightPosition(vec3f v) { m_lightPos = v; }
  void setLightTarget(vec3f v) { m_lightTarget = v; }

  vec4f getLightPositionEyeSpace() { return m_lightPosEye; }
  matrix4f getShadowMatrix() { return m_shadowMatrix; }

  GLuint getShadowTexture() { return m_lightTexture[m_srcLightTexture]; }

  void calcVectors();
  vec3f getSortVector() { return m_halfVector; }

  void render();
  void debugVectors();

 private:
  void drawPoints(int start, int count, bool sort);
  void drawPointSprites(GLSLProgram *prog, int start, int count, bool shadowed);

  void drawSlice(int i);
  void drawSliceLightView(int i);
  void drawSlices();
  void displayTexture(GLuint tex);
  void compositeResult();
  void blurLightBuffer();
  void depthSort();

  GLuint createTexture(GLenum target, int w, int h, GLint internalformat,
                       GLenum format);
  void createBuffers(int w, int h);
  void createLightBuffer();

  void drawQuad();
  void drawVector(vec3f v);

  // particle data
  unsigned int mMaxParticles;
  unsigned int mNumParticles;

  GLuint mPosVbo;
  GLuint mVelVbo;
  GLuint mColorVbo;
  GLuint mIndexBuffer;

  float mParticleRadius;
  DisplayMode mDisplayMode;

  // window
  unsigned int mWindowW, mWindowH;
  float mAspect, mInvFocalLen;
  float mFov;

  int m_downSample;
  int m_imageW, m_imageH;

  int m_numSlices;
  int m_numDisplayedSlices;
  int m_batchSize;
  int m_sliceNo;

  float m_shadowAlpha;
  float m_spriteAlpha;
  bool m_doBlur;
  float m_blurRadius;
  bool m_displayLightBuffer;

  vec3f m_lightVector, m_lightPos, m_lightTarget;
  vec3f m_lightColor;
  vec3f m_colorAttenuation;
  float m_lightDistance;

  matrix4f m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
  vec3f m_viewVector, m_halfVector;
  bool m_invertedView;
  vec4f m_eyePos;
  vec4f m_halfVectorEye;
  vec4f m_lightPosEye;

  // programs
  GLSLProgram *m_simpleProg;
  GLSLProgram *m_particleProg, *m_particleShadowProg;
  GLSLProgram *m_displayTexProg, *m_blurProg;

  // image buffers
  int m_lightBufferSize;
  GLuint m_lightTexture[2];
  int m_srcLightTexture;
  GLuint m_lightDepthTexture;
  FramebufferObject *m_lightFbo;

  GLuint m_imageTex, m_depthTex;
  FramebufferObject *m_imageFbo;
};

#endif
