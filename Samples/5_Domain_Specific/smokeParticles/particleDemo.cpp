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
    CUDA particle system with volumetric shadows
    sgreen 11/2008

    This sample demonstrates a technique for rendering realistic volumetric
    shadows through a cloud of particles. It uses CUDA for the simulation and
    depth sorting of the particles, and OpenGL for rendering.

    See the accompanying documentation for more details on the algorithm.

    This file handles OpenGL initialization and the user interface.
*/

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>

#include <helper_gl.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#if defined(WIN32)
#include <GL/wglew.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include "SmokeRenderer.h"
#include "paramgl.h"
#include "GLSLProgram.h"
#include "SmokeShaders.h"

uint numParticles = 1 << 16;

ParticleSystem *psystem = 0;
SmokeRenderer *renderer = 0;
GLSLProgram *floorProg = 0;

int winWidth = 1280, winHeight = 1024;
int g_TotalErrors = 0;

// view params
int ox, oy;
int buttonState = 0;
bool keyDown[256];

vec3f cameraPos(0, -1, -4);
vec3f cameraRot(0, 0, 0);
vec3f cameraPosLag(cameraPos);
vec3f cameraRotLag(cameraRot);
vec3f cursorPos(0, 1, 0);
vec3f cursorPosLag(cursorPos);

vec3f lightPos(5.0, 5.0, -5.0);

const float inertia = 0.1f;
const float translateSpeed = 0.002f;
const float cursorSpeed = 0.01f;
const float rotateSpeed = 0.2f;
const float walkSpeed = 0.05f;

enum { M_VIEW = 0, M_MOVE_CURSOR, M_MOVE_LIGHT };
int mode = 0;
int displayMode = (int)SmokeRenderer::VOLUMETRIC;

// QA AutoTest
bool g_bQAReadback = false;

// toggles
bool displayEnabled = true;
bool paused = false;
bool displaySliders = false;
bool wireframe = false;
bool animateEmitter = true;
bool emitterOn = true;
bool sort = true;
bool displayLightBuffer = false;
bool drawVectors = false;
bool doBlur = false;

float emitterVel = 0.0f;
uint emitterRate = 1000;
float emitterRadius = 0.25;
float emitterSpread = 0.0;
uint emitterIndex = 0;

// simulation parameters
float timestep = 0.5f;
float currentTime = 0.0f;
float spriteSize = 0.05f;
float alpha = 0.1f;
float shadowAlpha = 0.02f;
float particleLifetime = (float)numParticles / (float)emitterRate;
vec3f lightColor(1.0f, 1.0f, 0.8f);
vec3f colorAttenuation(0.5f, 0.75f, 1.0f);
float blurRadius = 2.0f;

int numSlices = 64;
int numDisplayedSlices = numSlices;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

float modelView[16];
ParamListGL *params;

GLuint floorTex = 0;

// CheckRender object for verification
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD 0.40f

// Define the files that are to be saved and the reference images for validation
const char *sSDKsample = "CUDA Smoke Particles";

const char *sRefBin[] = {"ref_smokePart_pos.bin", "ref_smokePart_vel.bin",
                         NULL};

void runEmitter();

// initialize particle system
void initParticles(int numParticles, bool bUseVBO, bool bUseGL) {
  psystem = new ParticleSystem(numParticles, bUseVBO, bUseGL);
  psystem->reset(ParticleSystem::CONFIG_RANDOM);

  if (bUseVBO) {
    renderer = new SmokeRenderer(numParticles);
    renderer->setLightTarget(vec3f(0.0, 1.0, 0.0));

    sdkCreateTimer(&timer);
  }
}

void cleanup() {
  if (psystem) {
    delete psystem;
  }

  if (renderer) {
    delete renderer;
  }

  if (floorProg) {
    delete floorProg;
  }

  sdkDeleteTimer(&timer);

  if (params) {
    delete params;
  }

  if (floorTex) {
    glDeleteTextures(1, &floorTex);
  }
}

void renderScene() {
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);

  // draw floor
  floorProg->enable();
  floorProg->bindTexture("tex", floorTex, GL_TEXTURE_2D, 0);
  floorProg->bindTexture("shadowTex", renderer->getShadowTexture(),
                         GL_TEXTURE_2D, 1);
  floorProg->setUniformfv("lightPosEye", renderer->getLightPositionEyeSpace(),
                          3);
  floorProg->setUniformfv("lightColor", lightColor, 3);

  // set shadow matrix as texture matrix
  matrix4f shadowMatrix = renderer->getShadowMatrix();
  glActiveTexture(GL_TEXTURE0);
  glMatrixMode(GL_TEXTURE);
  glLoadMatrixf((GLfloat *)shadowMatrix.get_value());

  glColor3f(1.0, 1.0, 1.0);
  glNormal3f(0.0, 1.0, 0.0);
  glBegin(GL_QUADS);
  {
    float s = 20.f;
    float rep = 20.f;
    glTexCoord2f(0.f, 0.f);
    glVertex3f(-s, 0, -s);
    glTexCoord2f(rep, 0.f);
    glVertex3f(s, 0, -s);
    glTexCoord2f(rep, rep);
    glVertex3f(s, 0, s);
    glTexCoord2f(0.f, rep);
    glVertex3f(-s, 0, s);
  }
  glEnd();
  floorProg->disable();

  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();

  // draw light
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(lightPos.x, lightPos.y, lightPos.z);
  glColor3fv(&lightColor[0]);
  glutSolidSphere(0.1, 10, 5);
  glPopMatrix();
}

// main rendering loop
void display() {
  sdkStartTimer(&timer);

  // move camera
  if (cameraPos[1] > 0.0f) {
    cameraPos[1] = 0.0f;
  }

  cameraPosLag += (cameraPos - cameraPosLag) * inertia;
  cameraRotLag += (cameraRot - cameraRotLag) * inertia;
  cursorPosLag += (cursorPos - cursorPosLag) * inertia;

  // view transform
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(cameraRotLag[0], 1.0, 0.0, 0.0);
  glRotatef(cameraRotLag[1], 0.0, 1.0, 0.0);
  glTranslatef(cameraPosLag[0], cameraPosLag[1], cameraPosLag[2]);

  glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

  // update the simulation
  if (!paused) {
    if (emitterOn) {
      runEmitter();
    }

    SimParams &p = psystem->getParams();
    p.cursorPos = make_float3(cursorPosLag.x, cursorPosLag.y, cursorPosLag.z);

    psystem->step(timestep);
    currentTime += timestep;
  }

  renderer->calcVectors();
  vec3f sortVector = renderer->getSortVector();

  psystem->setSortVector(make_float3(sortVector.x, sortVector.y, sortVector.z));
  psystem->setModelView(modelView);
  psystem->setSorting(sort);
  psystem->depthSort();

  // render
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  renderScene();

  // draw particles
  if (displayEnabled) {
    // render scene to offscreen buffers to get correct occlusion
    renderer->beginSceneRender(SmokeRenderer::LIGHT_BUFFER);
    renderScene();
    renderer->endSceneRender(SmokeRenderer::LIGHT_BUFFER);

    renderer->beginSceneRender(SmokeRenderer::SCENE_BUFFER);
    renderScene();
    renderer->endSceneRender(SmokeRenderer::SCENE_BUFFER);

    renderer->setPositionBuffer(psystem->getPosBuffer());
    renderer->setVelocityBuffer(psystem->getVelBuffer());
    renderer->setIndexBuffer(psystem->getSortedIndexBuffer());

    renderer->setNumParticles(psystem->getNumParticles());
    renderer->setParticleRadius(spriteSize);
    renderer->setDisplayLightBuffer(displayLightBuffer);
    renderer->setAlpha(alpha);
    renderer->setShadowAlpha(shadowAlpha);
    renderer->setLightPosition(lightPos);
    renderer->setColorAttenuation(colorAttenuation);
    renderer->setLightColor(lightColor);
    renderer->setNumSlices(numSlices);
    renderer->setNumDisplayedSlices(numDisplayedSlices);
    renderer->setBlurRadius(blurRadius);

    renderer->render();

    if (drawVectors) {
      renderer->debugVectors();
    }
  }

  // display sliders
  if (displaySliders) {
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);  // invert color
    glEnable(GL_BLEND);
    params->Render(0, 0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
  }

  glutSwapBuffers();
  glutReportErrors();
  sdkStopTimer(&timer);

  fpsCount++;

  // this displays the frame rate updated every second (independent of frame
  // rate)
  if (fpsCount >= fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "CUDA Smoke Particles (%d particles): %3.1f fps", numParticles,
            ifps);
    glutSetWindowTitle(fps);
    fpsCount = 0;
    fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

    if (paused) {
      fpsLimit = 0;
    }

    sdkResetTimer(&timer);
  }
}

// GLUT callback functions
void reshape(int w, int h) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)w / (float)h, 0.01, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);

  renderer->setFOV(60.0);
  renderer->setWindowSize(w, h);
}

void mouse(int button, int state, int x, int y) {
  int mods;

  if (state == GLUT_DOWN) {
    buttonState |= 1 << button;
  } else if (state == GLUT_UP) {
    buttonState = 0;
  }

  mods = glutGetModifiers();

  if (mods & GLUT_ACTIVE_SHIFT) {
    buttonState = 2;
  } else if (mods & GLUT_ACTIVE_CTRL) {
    buttonState = 3;
  }

  ox = x;
  oy = y;

  if (displaySliders) {
    if (params->Mouse(x, y, button, state)) {
      glutPostRedisplay();
      return;
    }
  }

  glutPostRedisplay();
}

// transform vector by matrix
void xform(vec3f &v, vec3f &r, float *m) {
  r.x = v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12];
  r.y = v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13];
  r.z = v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14];
}

// transform vector by transpose of matrix (assuming orthonormal)
void ixform(vec3f &v, vec3f &r, float *m) {
  r.x = v.x * m[0] + v.y * m[1] + v.z * m[2];
  r.y = v.x * m[4] + v.y * m[5] + v.z * m[6];
  r.z = v.x * m[8] + v.y * m[9] + v.z * m[10];
}

void ixformPoint(vec3f &v, vec3f &r, float *m) {
  vec3f x;
  x.x = v.x - m[12];
  x.y = v.y - m[13];
  x.z = v.z - m[14];
  ixform(x, r, m);
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);

  if (displaySliders) {
    if (params->Motion(x, y)) {
      ox = x;
      oy = y;
      glutPostRedisplay();
      return;
    }
  }

  switch (mode) {
    case M_VIEW: {
      if (buttonState == 1) {
        // left = rotate
        cameraRot[0] += dy * rotateSpeed;
        cameraRot[1] += dx * rotateSpeed;
      }

      if (buttonState == 2) {
        // middle = translate
        vec3f v = vec3f(dx * translateSpeed, -dy * translateSpeed, 0.0f);
        vec3f r;
        ixform(v, r, modelView);
        cameraPos += r;
      }

      if (buttonState == 3) {
        // left+middle = zoom
        vec3f v = vec3f(0.0, 0.0, dy * translateSpeed);
        vec3f r;
        ixform(v, r, modelView);
        cameraPos += r;
      }
    } break;

    case M_MOVE_CURSOR: {
      if (buttonState == 1) {
        vec3f v = vec3f(dx * cursorSpeed, -dy * cursorSpeed, 0.0f);
        vec3f r;
        ixform(v, r, modelView);
        cursorPos += r;
      } else if (buttonState == 2) {
        vec3f v = vec3f(0.0f, 0.0f, dy * cursorSpeed);
        vec3f r;
        ixform(v, r, modelView);
        cursorPos += r;
      }
    } break;

    case M_MOVE_LIGHT:
      if (buttonState == 1) {
        vec3f v = vec3f(dx * cursorSpeed, -dy * cursorSpeed, 0.0f);
        vec3f r;
        ixform(v, r, modelView);
        lightPos += r;
      } else if (buttonState == 2) {
        vec3f v = vec3f(0.0f, 0.0f, dy * cursorSpeed);
        vec3f r;
        ixform(v, r, modelView);
        lightPos += r;
      }

      break;
  }

  ox = x;
  oy = y;
  glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
    case ' ':
      paused = !paused;
      break;

    case 13:
      psystem->step(timestep);
      renderer->setPositionBuffer(psystem->getPosBuffer());
      renderer->setVelocityBuffer(psystem->getVelBuffer());
      break;

    case '\033':
      cleanup();
      exit(EXIT_SUCCESS);
      break;

    case 'v':
      mode = M_VIEW;
      animateEmitter = true;
      break;

    case 'm':
      mode = M_MOVE_CURSOR;
      animateEmitter = false;
      break;

    case 'l':
      mode = M_MOVE_LIGHT;
      break;

    case 'r':
      displayEnabled = !displayEnabled;
      break;

    case '1':
      psystem->reset(ParticleSystem::CONFIG_RANDOM);
      break;

    case '2':
      emitterOn ^= 1;
      break;

    case 'W':
      wireframe = !wireframe;
      break;

    case 'h':
      displaySliders = !displaySliders;
      break;

    case 'o':
      sort ^= 1;
      psystem->setSorting(sort);
      break;

    case 'D':
      displayLightBuffer ^= 1;
      break;

    case 'p':
      displayMode = (displayMode + 1) % SmokeRenderer::NUM_MODES;
      renderer->setDisplayMode((SmokeRenderer::DisplayMode)displayMode);
      break;

    case 'P':
      displayMode--;

      if (displayMode < 0) {
        displayMode = SmokeRenderer::NUM_MODES - 1;
      }

      renderer->setDisplayMode((SmokeRenderer::DisplayMode)displayMode);
      break;

    case 'V':
      drawVectors ^= 1;
      break;

    case '=':
      numSlices *= 2;

      if (numSlices > 256) {
        numSlices = 256;
      }

      numDisplayedSlices = numSlices;
      break;

    case '-':
      if (numSlices > 1) {
        numSlices /= 2;
      }

      numDisplayedSlices = numSlices;
      break;

    case 'b':
      doBlur ^= 1;
      renderer->setDoBlur(doBlur);
      break;
  }

  printf("numSlices = %d\n", numSlices);
  keyDown[key] = true;

  glutPostRedisplay();
}

void keyUp(unsigned char key, int /*x*/, int /*y*/) { keyDown[key] = false; }

void runEmitter() {
  vec3f vel = vec3f(0, emitterVel, 0);
  vec3f vx(1, 0, 0);
  vec3f vy(0, 0, 1);
  vec3f spread(emitterSpread, 0.0f, emitterSpread);

  psystem->sphereEmitter(emitterIndex, cursorPosLag, vel, spread, emitterRadius,
                         ftoi(emitterRate * timestep), particleLifetime,
                         particleLifetime * 0.1f);

  if (emitterIndex > numParticles - 1) {
    emitterIndex = 0;
  }
}

void special(int k, int x, int y) {
  if (displaySliders) {
    params->Special(k, x, y);
  }
}

void idle(void) {
  // move camera in view direction
  /*
      0   4   8   12  x
      1   5   9   13  y
      2   6   10  14  z
  */
  if (keyDown['w']) {
    cameraPos[0] += modelView[2] * walkSpeed;
    cameraPos[1] += modelView[6] * walkSpeed;
    cameraPos[2] += modelView[10] * walkSpeed;
  }

  if (keyDown['s']) {
    cameraPos[0] -= modelView[2] * walkSpeed;
    cameraPos[1] -= modelView[6] * walkSpeed;
    cameraPos[2] -= modelView[10] * walkSpeed;
  }

  if (keyDown['a']) {
    cameraPos[0] += modelView[0] * walkSpeed;
    cameraPos[1] += modelView[4] * walkSpeed;
    cameraPos[2] += modelView[8] * walkSpeed;
  }

  if (keyDown['d']) {
    cameraPos[0] -= modelView[0] * walkSpeed;
    cameraPos[1] -= modelView[4] * walkSpeed;
    cameraPos[2] -= modelView[8] * walkSpeed;
  }

  if (keyDown['e']) {
    cameraPos[0] += modelView[1] * walkSpeed;
    cameraPos[1] += modelView[5] * walkSpeed;
    cameraPos[2] += modelView[9] * walkSpeed;
  }

  if (keyDown['q']) {
    cameraPos[0] -= modelView[1] * walkSpeed;
    cameraPos[1] -= modelView[5] * walkSpeed;
    cameraPos[2] -= modelView[9] * walkSpeed;
  }

  if (animateEmitter) {
    const float speed = 0.02f;
    cursorPos.x = sin(currentTime * speed) * 1.5f;
    cursorPos.y = 1.5f + sin(currentTime * speed * 1.3f);
    cursorPos.z = cos(currentTime * speed) * 1.5f;
  }

  glutPostRedisplay();
}

// initialize sliders
void initParams() {
  // create a new parameter list
  params = new ParamListGL("misc");

  params->AddParam(new Param<int>("displayed slices", numDisplayedSlices, 0,
                                  256, 1, &numDisplayedSlices));

  params->AddParam(
      new Param<float>("time step", timestep, 0.0f, 1.0f, 0.001f, &timestep));

  SimParams &p = psystem->getParams();
  params->AddParam(
      new Param<float>("damping", 0.99f, 0.0f, 1.0f, 0.001f, &p.globalDamping));
  params->AddParam(
      new Param<float>("gravity", 0.0f, 0.01f, -0.01f, 0.0001f, &p.gravity.y));

  params->AddParam(
      new Param<float>("noise freq", 0.1f, 0.0f, 1.0f, 0.001f, &p.noiseFreq));
  params->AddParam(new Param<float>("noise strength", 0.001f, 0.0f, 0.01f,
                                    0.001f, &p.noiseAmp));
  params->AddParam(new Param<float>("noise anim", 0.0f, -0.001f, 0.001f,
                                    0.0001f, &p.noiseSpeed.y));

  params->AddParam(new Param<float>("sprite size", spriteSize, 0.0f, 0.1f,
                                    0.001f, &spriteSize));
  params->AddParam(
      new Param<float>("alpha", alpha, 0.0f, 1.0f, 0.001f, &alpha));

  params->AddParam(new Param<float>("light color r", lightColor[0], 0.0f, 1.0f,
                                    0.01f, &lightColor[0]));
  params->AddParam(new Param<float>("light color g", lightColor[1], 0.0f, 1.0f,
                                    0.01f, &lightColor[1]));
  params->AddParam(new Param<float>("light color b", lightColor[2], 0.0f, 1.0f,
                                    0.01f, &lightColor[2]));

  params->AddParam(new Param<float>("atten color r", colorAttenuation[0], 0.0f,
                                    1.0f, 0.01f, &colorAttenuation[0]));
  params->AddParam(new Param<float>("atten color g", colorAttenuation[1], 0.0f,
                                    1.0f, 0.01f, &colorAttenuation[1]));
  params->AddParam(new Param<float>("atten color b", colorAttenuation[2], 0.0f,
                                    1.0f, 0.01f, &colorAttenuation[2]));
  params->AddParam(new Param<float>("shadow alpha", shadowAlpha, 0.0f, 0.1f,
                                    0.001f, &shadowAlpha));

  params->AddParam(new Param<float>("blur radius", blurRadius, 0.0f, 10.0f,
                                    0.1f, &blurRadius));

  params->AddParam(new Param<float>("emitter radius", emitterRadius, 0.0f, 2.0f,
                                    0.01f, &emitterRadius));
  params->AddParam(
      new Param<uint>("emitter rate", emitterRate, 0, 10000, 1, &emitterRate));
  params->AddParam(new Param<float>("emitter velocity", emitterVel, 0.0f, 0.1f,
                                    0.001f, &emitterVel));
  params->AddParam(new Param<float>("emitter spread", emitterSpread, 0.0f, 0.1f,
                                    0.001f, &emitterSpread));

  params->AddParam(new Param<float>("particle lifetime", particleLifetime, 0.0f,
                                    1000.0f, 1.0f, &particleLifetime));
}

void mainMenu(int i) { key((unsigned char)i, 0, 0); }

void initMenus() {
  glutCreateMenu(mainMenu);
  glutAddMenuEntry("Reset block [1]", '1');
  glutAddMenuEntry("Toggle emitter [2]", '2');
  glutAddMenuEntry("Toggle animation [ ]", ' ');
  glutAddMenuEntry("Step animation [ret]", 13);
  glutAddMenuEntry("View mode [v]", 'v');
  glutAddMenuEntry("Move cursor mode [m]", 'm');
  glutAddMenuEntry("Move light mode [l]", 'l');
  glutAddMenuEntry("Toggle point rendering [p]", 'p');
  glutAddMenuEntry("Toggle sliders [h]", 'h');
  glutAddMenuEntry("Toggle sorting [o]", 'o');
  glutAddMenuEntry("Toggle vectors [V]", 'V');
  glutAddMenuEntry("Display light buffer [D]", 'D');
  glutAddMenuEntry("Toggle shadow blur [b]", 'b');
  glutAddMenuEntry("Increase no. slices [=]", '=');
  glutAddMenuEntry("Decrease no. slices [-]", '-');
  glutAddMenuEntry("Quit (esc)", '\033');
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w,
                     int h, void *data) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(target, tex);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE,
               data);
  return tex;
}

GLuint loadTexture(char *filename) {
  unsigned char *data = 0;
  unsigned int width, height;
  sdkLoadPPM4ub(filename, &data, &width, &height);

  if (!data) {
    printf("Error opening file '%s'\n", filename);
    return 0;
  }

  printf("Loaded '%s', %d x %d pixels\n", filename, width, height);

  return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
}

// initialize OpenGL
void initGL(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(winWidth, winHeight);
  glutCreateWindow("CUDA Smoke Particles");

  if (!isGLVersionSupported(2, 0)) {
    fprintf(stderr,
            "The following required OpenGL extensions "
            "missing:\n\tGL_VERSION_2_0\n\tGL_VERSION_1_5\n");
    exit(EXIT_SUCCESS);
  }

  if (!areGLExtensionsSupported("GL_ARB_multitexture "
                                "GL_ARB_vertex_buffer_object "
                                "GL_EXT_geometry_shader4")) {
    fprintf(stderr,
            "The following required OpenGL extensions "
            "missing:\n\tGL_ARB_multitexture\n\tGL_ARB_vertex_buffer_"
            "object\n\tGL_EXT_geometry_shader4.\n");
    exit(EXIT_SUCCESS);
  }

#if defined(WIN32)

  if (wglewIsSupported("WGL_EXT_swap_control")) {
    // disable vertical sync
    wglSwapIntervalEXT(0);
  }

#endif

  glEnable(GL_DEPTH_TEST);

  // load floor texture
  char *imagePath = sdkFindFilePath("floortile.ppm", argv[0]);

  if (imagePath == NULL) {
    fprintf(stderr, "Error finding floor image file\n");
    exit(EXIT_FAILURE);
  }

  floorTex = loadTexture(imagePath);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);

  floorProg = new GLSLProgram(floorVS, floorPS);

  glutReportErrors();
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  printf("%s Starting...\n\n", sSDKsample);

  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
      numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
      g_bQAReadback = true;
    }
  }

  if (g_bQAReadback) {
    // For Automated testing, we do not use OpenGL/CUDA interop
    findCudaDevice(argc, (const char **)argv);

    // This code path is used for Automated Testing
    initParticles(numParticles, false, false);
    initParams();

    if (emitterOn) {
      runEmitter();
    }

    SimParams &params = psystem->getParams();
    params.cursorPos =
        make_float3(cursorPosLag.x, cursorPosLag.y, cursorPosLag.z);

    psystem->step(timestep);

    float4 *pos = NULL, *vel = NULL;

    psystem->dumpBin(&pos, &vel);

    sdkDumpBin(pos, numParticles * sizeof(float4), "smokeParticles_pos.bin");
    sdkDumpBin(vel, numParticles * sizeof(float4), "smokeParticles_vel.bin");

    if (!sdkCompareBin2BinFloat("smokeParticles_pos.bin", sRefBin[0],
                                numParticles * sizeof(float4),
                                MAX_EPSILON_ERROR, THRESHOLD, argv[0])) {
      g_TotalErrors++;
    }

    if (!sdkCompareBin2BinFloat("smokeParticles_vel.bin", sRefBin[1],
                                numParticles * sizeof(float4),
                                MAX_EPSILON_ERROR, THRESHOLD, argv[0])) {
      g_TotalErrors++;
    }

    delete psystem;
  } else {
    // Normal smokeParticles rendering path
    // 1st initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is needed to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);

    findCudaDevice(argc, (const char **)argv);

    // This is the normal code path for SmokeParticles
    initParticles(numParticles, true, true);
    initParams();
    initMenus();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutKeyboardUpFunc(keyUp);
    glutSpecialFunc(special);
    glutIdleFunc(idle);

    glutMainLoop();
  }

  exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
