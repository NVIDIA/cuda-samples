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

#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <paramgl.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bodysystemcuda.h"
#include "bodysystemcpu.h"
#include "render_particles.h"
#include "cuda_runtime.h"

// view params
int ox = 0, oy = 0;
int buttonState = 0;
float camera_trans[] = {0, -2, -150};
float camera_rot[] = {0, 0, 0};
float camera_trans_lag[] = {0, -2, -150};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;

ParticleRenderer::DisplayMode displayMode =
    ParticleRenderer::PARTICLE_SPRITES_COLOR;

bool benchmark = false;
bool compareToCPU = false;
bool QATest = false;
int blockSize = 256;
bool useHostMem = false;
bool useP2P = true;  // this is always optimal to use P2P path when available
bool fp64 = false;
bool useCpu = false;
int numDevsRequested = 1;
bool displayEnabled = true;
bool bPause = false;
bool bFullscreen = false;
bool bDispInteractions = false;
bool bSupportDouble = false;
int flopsPerInteraction = 20;

char deviceName[100];

enum { M_VIEW = 0, M_MOVE };

int numBodies = 16384;

std::string tipsyFile = "";

int numIterations = 0;  // run until exit

void computePerfStats(double &interactionsPerSecond, double &gflops,
                      float milliseconds, int iterations) {
  // double precision uses intrinsic operation followed by refinement,
  // resulting in higher operation count per interaction.
  // (Note Astrophysicists use 38 flops per interaction no matter what,
  // based on "historical precedent", but they are using FLOP/s as a
  // measure of "science throughput". We are using it as a measure of
  // hardware throughput.  They should really use interactions/s...
  // const int flopsPerInteraction = fp64 ? 30 : 20;
  interactionsPerSecond = (float)numBodies * (float)numBodies;
  interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
  gflops = interactionsPerSecond * (float)flopsPerInteraction;
}

////////////////////////////////////////
// Demo Parameters
////////////////////////////////////////
struct NBodyParams {
  float m_timestep;
  float m_clusterScale;
  float m_velocityScale;
  float m_softening;
  float m_damping;
  float m_pointSize;
  float m_x, m_y, m_z;

  void print() {
    printf("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n", m_timestep,
           m_clusterScale, m_velocityScale, m_softening, m_damping, m_pointSize,
           m_x, m_y, m_z);
  }
};

NBodyParams demoParams[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    {0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0,
     -50},
};

int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
bool cycleDemo = true;
int activeDemo = 0;
float demoTime = 10000.0f;  // ms
StopWatchInterface *demoTimer = NULL, *timer = NULL;

// run multiple iterations to compute an average sort time

NBodyParams activeParams = demoParams[activeDemo];

// The UI.
ParamListGL *paramlist;  // parameter list
bool bShowSliders = true;

// fps
static int fpsCount = 0;
static int fpsLimit = 5;
cudaEvent_t startEvent, stopEvent;
cudaEvent_t hostMemSyncEvent;

template <typename T>
class NBodyDemo {
 public:
  static void Create() { m_singleton = new NBodyDemo; }
  static void Destroy() { delete m_singleton; }

  static void init(int numBodies, int numDevices, int blockSize, bool usePBO,
                   bool useHostMem, bool useP2P, bool useCpu, int devID) {
    m_singleton->_init(numBodies, numDevices, blockSize, usePBO, useHostMem,
                       useP2P, useCpu, devID);
  }

  static void reset(int numBodies, NBodyConfig config) {
    m_singleton->_reset(numBodies, config);
  }

  static void selectDemo(int index) { m_singleton->_selectDemo(index); }

  static bool compareResults(int numBodies) {
    return m_singleton->_compareResults(numBodies);
  }

  static void runBenchmark(int iterations) {
    m_singleton->_runBenchmark(iterations);
  }

  static void updateParams() {
    m_singleton->m_nbody->setSoftening(activeParams.m_softening);
    m_singleton->m_nbody->setDamping(activeParams.m_damping);
  }

  static void updateSimulation() {
    m_singleton->m_nbody->update(activeParams.m_timestep);
  }

  static void display() {
    m_singleton->m_renderer->setSpriteSize(activeParams.m_pointSize);

    if (useHostMem) {
      // This event sync is required because we are rendering from the host
      // memory that CUDA is
      // writing.  If we don't wait until CUDA is done updating it, we will
      // render partially
      // updated data, resulting in a jerky frame rate.
      if (!useCpu) {
        cudaEventSynchronize(hostMemSyncEvent);
      }

      m_singleton->m_renderer->setPositions(
          m_singleton->m_nbody->getArray(BODYSYSTEM_POSITION),
          m_singleton->m_nbody->getNumBodies());
    } else {
      m_singleton->m_renderer->setPBO(
          m_singleton->m_nbody->getCurrentReadBuffer(),
          m_singleton->m_nbody->getNumBodies(), (sizeof(T) > 4));
    }

    // display particles
    m_singleton->m_renderer->display(displayMode);
  }

  static void getArrays(T *pos, T *vel) {
    T *_pos = m_singleton->m_nbody->getArray(BODYSYSTEM_POSITION);
    T *_vel = m_singleton->m_nbody->getArray(BODYSYSTEM_VELOCITY);
    memcpy(pos, _pos, m_singleton->m_nbody->getNumBodies() * 4 * sizeof(T));
    memcpy(vel, _vel, m_singleton->m_nbody->getNumBodies() * 4 * sizeof(T));
  }

  static void setArrays(const T *pos, const T *vel) {
    if (pos != m_singleton->m_hPos) {
      memcpy(m_singleton->m_hPos, pos, numBodies * 4 * sizeof(T));
    }

    if (vel != m_singleton->m_hVel) {
      memcpy(m_singleton->m_hVel, vel, numBodies * 4 * sizeof(T));
    }

    m_singleton->m_nbody->setArray(BODYSYSTEM_POSITION, m_singleton->m_hPos);
    m_singleton->m_nbody->setArray(BODYSYSTEM_VELOCITY, m_singleton->m_hVel);

    if (!benchmark && !useCpu && !compareToCPU) {
      m_singleton->_resetRenderer();
    }
  }

 private:
  static NBodyDemo *m_singleton;

  BodySystem<T> *m_nbody;
  BodySystemCUDA<T> *m_nbodyCuda;
  BodySystemCPU<T> *m_nbodyCpu;

  ParticleRenderer *m_renderer;

  T *m_hPos;
  T *m_hVel;
  float *m_hColor;

 private:
  NBodyDemo()
      : m_nbody(0),
        m_nbodyCuda(0),
        m_nbodyCpu(0),
        m_renderer(0),
        m_hPos(0),
        m_hVel(0),
        m_hColor(0) {}

  ~NBodyDemo() {
    if (m_nbodyCpu) {
      delete m_nbodyCpu;
    }

    if (m_nbodyCuda) {
      delete m_nbodyCuda;
    }

    if (m_hPos) {
      delete[] m_hPos;
    }

    if (m_hVel) {
      delete[] m_hVel;
    }

    if (m_hColor) {
      delete[] m_hColor;
    }

    sdkDeleteTimer(&demoTimer);

    if (!benchmark && !compareToCPU) delete m_renderer;
  }

  void _init(int numBodies, int numDevices, int blockSize, bool bUsePBO,
             bool useHostMem, bool useP2P, bool useCpu, int devID) {
    if (useCpu) {
      m_nbodyCpu = new BodySystemCPU<T>(numBodies);
      m_nbody = m_nbodyCpu;
      m_nbodyCuda = 0;
    } else {
      m_nbodyCuda = new BodySystemCUDA<T>(numBodies, numDevices, blockSize,
                                          bUsePBO, useHostMem, useP2P, devID);
      m_nbody = m_nbodyCuda;
      m_nbodyCpu = 0;
    }

    // allocate host memory
    m_hPos = new T[numBodies * 4];
    m_hVel = new T[numBodies * 4];
    m_hColor = new float[numBodies * 4];

    m_nbody->setSoftening(activeParams.m_softening);
    m_nbody->setDamping(activeParams.m_damping);

    if (useCpu) {
      sdkCreateTimer(&timer);
      sdkStartTimer(&timer);
    } else {
      checkCudaErrors(cudaEventCreate(&startEvent));
      checkCudaErrors(cudaEventCreate(&stopEvent));
      checkCudaErrors(cudaEventCreate(&hostMemSyncEvent));
    }

    if (!benchmark && !compareToCPU) {
      m_renderer = new ParticleRenderer;
      _resetRenderer();
    }

    sdkCreateTimer(&demoTimer);
    sdkStartTimer(&demoTimer);
  }

  void _reset(int numBodies, NBodyConfig config) {
    if (tipsyFile == "") {
      randomizeBodies(config, m_hPos, m_hVel, m_hColor,
                      activeParams.m_clusterScale, activeParams.m_velocityScale,
                      numBodies, true);
      setArrays(m_hPos, m_hVel);
    } else {
      m_nbody->loadTipsyFile(tipsyFile);
      ::numBodies = m_nbody->getNumBodies();
    }
  }

  void _resetRenderer() {
    if (fp64) {
      float color[4] = {0.4f, 0.8f, 0.1f, 1.0f};
      m_renderer->setBaseColor(color);
    } else {
      float color[4] = {1.0f, 0.6f, 0.3f, 1.0f};
      m_renderer->setBaseColor(color);
    }

    m_renderer->setColors(m_hColor, m_nbody->getNumBodies());
    m_renderer->setSpriteSize(activeParams.m_pointSize);
  }

  void _selectDemo(int index) {
    assert(index < numDemos);

    activeParams = demoParams[index];
    camera_trans[0] = camera_trans_lag[0] = activeParams.m_x;
    camera_trans[1] = camera_trans_lag[1] = activeParams.m_y;
    camera_trans[2] = camera_trans_lag[2] = activeParams.m_z;
    reset(numBodies, NBODY_CONFIG_SHELL);
    sdkResetTimer(&demoTimer);
  }

  bool _compareResults(int numBodies) {
    assert(m_nbodyCuda);

    bool passed = true;

    m_nbody->update(0.001f);

    {
      m_nbodyCpu = new BodySystemCPU<T>(numBodies);

      m_nbodyCpu->setArray(BODYSYSTEM_POSITION, m_hPos);
      m_nbodyCpu->setArray(BODYSYSTEM_VELOCITY, m_hVel);

      m_nbodyCpu->update(0.001f);

      T *cudaPos = m_nbodyCuda->getArray(BODYSYSTEM_POSITION);
      T *cpuPos = m_nbodyCpu->getArray(BODYSYSTEM_POSITION);

      T tolerance = 0.0005f;

      for (int i = 0; i < numBodies; i++) {
        if (fabs(cpuPos[i] - cudaPos[i]) > tolerance) {
          passed = false;
          printf("Error: (host)%f != (device)%f\n", cpuPos[i], cudaPos[i]);
        }
      }
    }
    if (passed) {
      printf("  OK\n");
    }
    return passed;
  }

  void _runBenchmark(int iterations) {
    // once without timing to prime the device
    if (!useCpu) {
      m_nbody->update(activeParams.m_timestep);
    }

    if (useCpu) {
      sdkCreateTimer(&timer);
      sdkStartTimer(&timer);
    } else {
      checkCudaErrors(cudaEventRecord(startEvent, 0));
    }

    for (int i = 0; i < iterations; ++i) {
      m_nbody->update(activeParams.m_timestep);
    }

    float milliseconds = 0;

    if (useCpu) {
      sdkStopTimer(&timer);
      milliseconds = sdkGetTimerValue(&timer);
      sdkStartTimer(&timer);
    } else {
      checkCudaErrors(cudaEventRecord(stopEvent, 0));
      checkCudaErrors(cudaEventSynchronize(stopEvent));
      checkCudaErrors(
          cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    }

    double interactionsPerSecond = 0;
    double gflops = 0;
    computePerfStats(interactionsPerSecond, gflops, milliseconds, iterations);

    printf("%d bodies, total time for %d iterations: %.3f ms\n", numBodies,
           iterations, milliseconds);
    printf("= %.3f billion interactions per second\n", interactionsPerSecond);
    printf("= %.3f %s-precision GFLOP/s at %d flops per interaction\n", gflops,
           (sizeof(T) > 4) ? "double" : "single", flopsPerInteraction);
  }
};

void finalize() {
  if (!useCpu) {
    checkCudaErrors(cudaEventDestroy(startEvent));
    checkCudaErrors(cudaEventDestroy(stopEvent));
    checkCudaErrors(cudaEventDestroy(hostMemSyncEvent));
  }

  NBodyDemo<float>::Destroy();

  if (bSupportDouble) NBodyDemo<double>::Destroy();
}

template <>
NBodyDemo<double> *NBodyDemo<double>::m_singleton = 0;
template <>
NBodyDemo<float> *NBodyDemo<float>::m_singleton = 0;

template <typename T_new, typename T_old>
void switchDemoPrecision() {
  cudaDeviceSynchronize();

  fp64 = !fp64;
  flopsPerInteraction = fp64 ? 30 : 20;

  T_old *oldPos = new T_old[numBodies * 4];
  T_old *oldVel = new T_old[numBodies * 4];

  NBodyDemo<T_old>::getArrays(oldPos, oldVel);

  // convert float to double
  T_new *newPos = new T_new[numBodies * 4];
  T_new *newVel = new T_new[numBodies * 4];

  for (int i = 0; i < numBodies * 4; i++) {
    newPos[i] = (T_new)oldPos[i];
    newVel[i] = (T_new)oldVel[i];
  }

  NBodyDemo<T_new>::setArrays(newPos, newVel);

  cudaDeviceSynchronize();

  delete[] oldPos;
  delete[] oldVel;
  delete[] newPos;
  delete[] newVel;
}

// check for OpenGL errors
inline void checkGLErrors(const char *s) {
  GLenum error;

  while ((error = glGetError()) != GL_NO_ERROR) {
    fprintf(stderr, "%s: error - %s\n", s, (char *)gluErrorString(error));
  }
}

void initGL(int *argc, char **argv) {
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(720, 480);
  glutCreateWindow("CUDA n-body system");

  if (bFullscreen) {
    glutFullScreen();
  }

  else if (!isGLVersionSupported(2, 0) ||
           !areGLExtensionsSupported("GL_ARB_multitexture "
                                     "GL_ARB_vertex_buffer_object")) {
    fprintf(stderr, "Required OpenGL extensions missing.");
    exit(EXIT_FAILURE);
  } else {
#if defined(WIN32)
    wglSwapIntervalEXT(0);
#elif defined(LINUX)
    glxSwapIntervalSGI(0);
#endif
  }

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  checkGLErrors("initGL");
}

void initParameters() {
  // create a new parameter list
  paramlist = new ParamListGL("sliders");
  paramlist->SetBarColorInner(0.8f, 0.8f, 0.0f);

  // add some parameters to the list

  // Point Size
  paramlist->AddParam(new Param<float>("Point Size", activeParams.m_pointSize,
                                       0.001f, 10.0f, 0.01f,
                                       &activeParams.m_pointSize));

  // Velocity Damping
  paramlist->AddParam(new Param<float>("Velocity Damping",
                                       activeParams.m_damping, 0.5f, 1.0f,
                                       .0001f, &(activeParams.m_damping)));
  // Softening Factor
  paramlist->AddParam(new Param<float>("Softening Factor",
                                       activeParams.m_softening, 0.001f, 1.0f,
                                       .0001f, &(activeParams.m_softening)));
  // Time step size
  paramlist->AddParam(new Param<float>("Time Step", activeParams.m_timestep,
                                       0.0f, 1.0f, .0001f,
                                       &(activeParams.m_timestep)));
  // Cluster scale (only affects starting configuration
  paramlist->AddParam(new Param<float>("Cluster Scale",
                                       activeParams.m_clusterScale, 0.0f, 10.0f,
                                       0.01f, &(activeParams.m_clusterScale)));

  // Velocity scale (only affects starting configuration)
  paramlist->AddParam(
      new Param<float>("Velocity Scale", activeParams.m_velocityScale, 0.0f,
                       1000.0f, 0.1f, &activeParams.m_velocityScale));
}

void selectDemo(int activeDemo) {
  if (fp64) {
    NBodyDemo<double>::selectDemo(activeDemo);
  } else {
    NBodyDemo<float>::selectDemo(activeDemo);
  }
}

void updateSimulation() {
  if (fp64) {
    NBodyDemo<double>::updateSimulation();
  } else {
    NBodyDemo<float>::updateSimulation();
  }
}

void displayNBodySystem() {
  if (fp64) {
    NBodyDemo<double>::display();
  } else {
    NBodyDemo<float>::display();
  }
}

void display() {
  static double gflops = 0;
  static double ifps = 0;
  static double interactionsPerSecond = 0;

  // update the simulation
  if (!bPause) {
    if (cycleDemo && (sdkGetTimerValue(&demoTimer) > demoTime)) {
      activeDemo = (activeDemo + 1) % numDemos;
      selectDemo(activeDemo);
    }

    updateSimulation();

    if (!useCpu) {
      cudaEventRecord(hostMemSyncEvent,
                      0);  // insert an event to wait on before rendering
    }
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (displayEnabled) {
    // view transform
    {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      for (int c = 0; c < 3; ++c) {
        camera_trans_lag[c] +=
            (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
      }

      glTranslatef(camera_trans_lag[0], camera_trans_lag[1],
                   camera_trans_lag[2]);
      glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
      glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
    }

    displayNBodySystem();

    // display user interface
    if (bShowSliders) {
      glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);  // invert color
      glEnable(GL_BLEND);
      paramlist->Render(0, 0);
      glDisable(GL_BLEND);
    }

    if (bFullscreen) {
      beginWinCoords();
      char msg0[256], msg1[256], msg2[256];

      if (bDispInteractions) {
        sprintf(msg1, "%0.2f billion interactions per second",
                interactionsPerSecond);
      } else {
        sprintf(msg1, "%0.2f GFLOP/s", gflops);
      }

      sprintf(msg0, "%s", deviceName);
      sprintf(msg2, "%0.2f FPS [%s | %d bodies]", ifps,
              fp64 ? "double precision" : "single precision", numBodies);

      glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);  // invert color
      glEnable(GL_BLEND);
      glColor3f(0.46f, 0.73f, 0.0f);
      glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 122, msg0,
              GLUT_BITMAP_TIMES_ROMAN_24);
      glColor3f(1.0f, 1.0f, 1.0f);
      glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 96, msg2,
              GLUT_BITMAP_TIMES_ROMAN_24);
      glColor3f(1.0f, 1.0f, 1.0f);
      glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 70, msg1,
              GLUT_BITMAP_TIMES_ROMAN_24);
      glDisable(GL_BLEND);

      endWinCoords();
    }

    glutSwapBuffers();
  }

  fpsCount++;

  // this displays the frame rate updated every second (independent of frame
  // rate)
  if (fpsCount >= fpsLimit) {
    char fps[256];

    float milliseconds = 1;

    // stop timer
    if (useCpu) {
      milliseconds = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
    } else {
      checkCudaErrors(cudaEventRecord(stopEvent, 0));
      checkCudaErrors(cudaEventSynchronize(stopEvent));
      checkCudaErrors(
          cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    }

    milliseconds /= (float)fpsCount;
    computePerfStats(interactionsPerSecond, gflops, milliseconds, 1);

    ifps = 1.f / (milliseconds / 1000.f);
    sprintf(fps,
            "CUDA N-Body (%d bodies): "
            "%0.1f fps | %0.1f BIPS | %0.1f GFLOP/s | %s",
            numBodies, ifps, interactionsPerSecond, gflops,
            fp64 ? "double precision" : "single precision");

    glutSetWindowTitle(fps);
    fpsCount = 0;
    fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

    if (bPause) {
      fpsLimit = 0;
    }

    // restart timer
    if (!useCpu) {
      checkCudaErrors(cudaEventRecord(startEvent, 0));
    }
  }

  glutReportErrors();
}

void reshape(int w, int h) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)w / (float)h, 0.1, 1000.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);
}

void updateParams() {
  if (fp64) {
    NBodyDemo<double>::updateParams();
  } else {
    NBodyDemo<float>::updateParams();
  }
}

void mouse(int button, int state, int x, int y) {
  if (bShowSliders) {
    // call list mouse function
    if (paramlist->Mouse(x, y, button, state)) {
      updateParams();
    }
  }

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

  glutPostRedisplay();
}

void motion(int x, int y) {
  if (bShowSliders) {
    // call parameter list motion function
    if (paramlist->Motion(x, y)) {
      updateParams();
      glutPostRedisplay();
      return;
    }
  }

  float dx = (float)(x - ox);
  float dy = (float)(y - oy);

  if (buttonState == 3) {
    // left+middle = zoom
    camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
  } else if (buttonState & 2) {
    // middle = translate
    camera_trans[0] += dx / 100.0f;
    camera_trans[1] -= dy / 100.0f;
  } else if (buttonState & 1) {
    // left = rotate
    camera_rot[0] += dy / 5.0f;
    camera_rot[1] += dx / 5.0f;
  }

  ox = x;
  oy = y;
  glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
    case ' ':
      bPause = !bPause;
      break;

    case 27:  // escape
    case 'q':
    case 'Q':
      finalize();
      exit(EXIT_SUCCESS);
      break;

    case 13:  // return
      if (bSupportDouble) {
        if (fp64) {
          switchDemoPrecision<float, double>();
        } else {
          switchDemoPrecision<double, float>();
        }

        printf("> %s precision floating point simulation\n",
               fp64 ? "Double" : "Single");
      }

      break;

    case '`':
      bShowSliders = !bShowSliders;
      break;

    case 'g':
    case 'G':
      bDispInteractions = !bDispInteractions;
      break;

    case 'p':
    case 'P':
      displayMode = (ParticleRenderer::DisplayMode)(
          (displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
      break;

    case 'c':
    case 'C':
      cycleDemo = !cycleDemo;
      printf("Cycle Demo Parameters: %s\n", cycleDemo ? "ON" : "OFF");
      break;

    case '[':
      activeDemo =
          (activeDemo == 0) ? numDemos - 1 : (activeDemo - 1) % numDemos;
      selectDemo(activeDemo);
      break;

    case ']':
      activeDemo = (activeDemo + 1) % numDemos;
      selectDemo(activeDemo);
      break;

    case 'd':
    case 'D':
      displayEnabled = !displayEnabled;
      break;

    case 'o':
    case 'O':
      activeParams.print();
      break;

    case '1':
      if (fp64) {
        NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_SHELL);
      } else {
        NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_SHELL);
      }

      break;

    case '2':
      if (fp64) {
        NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_RANDOM);
      } else {
        NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_RANDOM);
      }

      break;

    case '3':
      if (fp64) {
        NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_EXPAND);
      } else {
        NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_EXPAND);
      }

      break;
  }

  glutPostRedisplay();
}

void special(int key, int x, int y) {
  paramlist->Special(key, x, y);
  glutPostRedisplay();
}

void idle(void) { glutPostRedisplay(); }

void showHelp() {
  printf("\t-fullscreen       (run n-body simulation in fullscreen mode)\n");
  printf(
      "\t-fp64             (use double precision floating point values for "
      "simulation)\n");
  printf("\t-hostmem          (stores simulation data in host memory)\n");
  printf("\t-benchmark        (run benchmark to measure performance) \n");
  printf(
      "\t-numbodies=<N>    (number of bodies (>= 1) to run in simulation) \n");
  printf(
      "\t-device=<d>       (where d=0,1,2.... for the CUDA device to use)\n");
  printf(
      "\t-numdevices=<i>   (where i=(number of CUDA devices > 0) to use for "
      "simulation)\n");
  printf(
      "\t-compare          (compares simulation results running once on the "
      "default GPU and once on the CPU)\n");
  printf("\t-cpu              (run n-body simulation on the CPU)\n");
  printf("\t-tipsy=<file.bin> (load a tipsy model file for simulation)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  bool bTestResults = true;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("\n> Command line options\n");
    showHelp();
    return 0;
  }

  printf(
      "Run \"nbody -benchmark [-numbodies=<numBodies>]\" to measure "
      "performance.\n");
  showHelp();

  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  bFullscreen =
      (checkCmdLineFlag(argc, (const char **)argv, "fullscreen") != 0);

  if (bFullscreen) {
    bShowSliders = false;
  }

  benchmark = (checkCmdLineFlag(argc, (const char **)argv, "benchmark") != 0);

  compareToCPU =
      ((checkCmdLineFlag(argc, (const char **)argv, "compare") != 0) ||
       (checkCmdLineFlag(argc, (const char **)argv, "qatest") != 0));

  QATest = (checkCmdLineFlag(argc, (const char **)argv, "qatest") != 0);
  useHostMem = (checkCmdLineFlag(argc, (const char **)argv, "hostmem") != 0);
  fp64 = (checkCmdLineFlag(argc, (const char **)argv, "fp64") != 0);

  flopsPerInteraction = fp64 ? 30 : 20;

  useCpu = (checkCmdLineFlag(argc, (const char **)argv, "cpu") != 0);

  if (checkCmdLineFlag(argc, (const char **)argv, "numdevices")) {
    numDevsRequested =
        getCmdLineArgumentInt(argc, (const char **)argv, "numdevices");

    if (numDevsRequested < 1) {
      printf(
          "Error: \"number of CUDA devices\" specified %d is invalid.  Value "
          "should be >= 1\n",
          numDevsRequested);
      exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
    } else {
      printf("number of CUDA devices  = %d\n", numDevsRequested);
    }
  }

  int numDevsAvailable = 0;
  bool customGPU = false;
  cudaGetDeviceCount(&numDevsAvailable);

  if (numDevsAvailable < numDevsRequested) {
    printf("Error: only %d Devices available, %d requested.  Exiting.\n",
           numDevsAvailable, numDevsRequested);
    exit(EXIT_FAILURE);
  }

  if (numDevsRequested > 1) {
    // If user did not explicitly request host memory to be used, we default to
    // P2P.
    // We fallback to host memory, if any of GPUs does not support P2P.
    bool allGPUsSupportP2P = true;
    if (!useHostMem) {
      // Enable P2P only in one direction, as every peer will access gpu0
      for (int i = 1; i < numDevsRequested; ++i) {
        int canAccessPeer;
        checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, i, 0));

        if (canAccessPeer != 1) {
          allGPUsSupportP2P = false;
        }
      }

      if (!allGPUsSupportP2P) {
        useHostMem = true;
        useP2P = false;
      }
    }
  }

  printf("> %s mode\n", bFullscreen ? "Fullscreen" : "Windowed");
  printf("> Simulation data stored in %s memory\n",
         useHostMem ? "system" : "video");
  printf("> %s precision floating point simulation\n",
         fp64 ? "Double" : "Single");
  printf("> %d Devices used for simulation\n", numDevsRequested);

  int devID;
  cudaDeviceProp props;

  if (useCpu) {
    useHostMem = true;
    compareToCPU = false;
    bSupportDouble = true;

#ifdef OPENMP
    printf("> Simulation with CPU using OpenMP\n");
#else
    printf("> Simulation with CPU\n");
#endif
  }

  // Initialize GL and GLUT if necessary
  if (!benchmark && !compareToCPU) {
    initGL(&argc, argv);
    initParameters();
  }

  if (!useCpu) {
    // Now choose the CUDA Device
    // Either without GL interop:
    if (benchmark || compareToCPU || useHostMem) {
      // Note if we are using host memory for the body system, we
      // don't use CUDA-GL interop.

      if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        customGPU = true;
      }

      devID = findCudaDevice(argc, (const char **)argv);
    } else  // or with GL interop:
    {
      if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        customGPU = true;
      }

      devID = findCudaDevice(argc, (const char **)argv);
    }

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    bSupportDouble = true;

#if CUDART_VERSION < 4000

    if (numDevsRequested > 1) {
      printf("MultiGPU n-body requires CUDA 4.0 or later\n");
      exit(EXIT_SUCCESS);
    }

#endif

    // Initialize devices
    if (numDevsRequested > 1 && customGPU) {
      printf("You can't use --numdevices and --device at the same time.\n");
      exit(EXIT_SUCCESS);
    }

    if (customGPU || numDevsRequested == 1) {
      cudaDeviceProp props;
      checkCudaErrors(cudaGetDeviceProperties(&props, devID));
      printf("> Compute %d.%d CUDA device: [%s]\n", props.major, props.minor,
             props.name);
    } else {
      for (int i = 0; i < numDevsRequested; i++) {
        cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, i));

        printf("> Compute %d.%d CUDA device: [%s]\n", props.major, props.minor,
               props.name);

        if (useHostMem) {
#if CUDART_VERSION >= 2020

          if (!props.canMapHostMemory) {
            fprintf(stderr, "Device %d cannot map host memory!\n", devID);
            exit(EXIT_SUCCESS);
          }

          if (numDevsRequested > 1) {
            checkCudaErrors(cudaSetDevice(i));
          }

          checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#else
          fprintf(stderr,
                  "This CUDART version does not support "
                  "<cudaDeviceProp.canMapHostMemory> field\n");
          exit(EXIT_SUCCESS);
#endif
        }
      }

      // CC 1.2 and earlier do not support double precision
      if (props.major * 10 + props.minor <= 12) {
        bSupportDouble = false;
      }
    }

    // if(numDevsRequested > 1)
    //    checkCudaErrors(cudaSetDevice(devID));

    if (fp64 && !bSupportDouble) {
      fprintf(stderr,
              "One or more of the requested devices does not support double "
              "precision floating-point\n");
      exit(EXIT_SUCCESS);
    }
  }

  numIterations = 0;
  blockSize = 0;

  if (checkCmdLineFlag(argc, (const char **)argv, "i")) {
    numIterations = getCmdLineArgumentInt(argc, (const char **)argv, "i");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "blockSize")) {
    blockSize = getCmdLineArgumentInt(argc, (const char **)argv, "blockSize");
  }

  if (blockSize == 0)  // blockSize not set on command line
    blockSize = 256;

  // default number of bodies is #SMs * 4 * CTA size
  if (useCpu)
#ifdef OPENMP
    numBodies = 8192;

#else
    numBodies = 4096;
#endif
  else if (numDevsRequested == 1) {
    numBodies = compareToCPU ? 4096 : blockSize * 4 * props.multiProcessorCount;
  } else {
    numBodies = 0;

    for (int i = 0; i < numDevsRequested; i++) {
      cudaDeviceProp props;
      checkCudaErrors(cudaGetDeviceProperties(&props, i));
      numBodies +=
          blockSize * (props.major >= 2 ? 4 : 1) * props.multiProcessorCount;
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "numbodies")) {
    numBodies = getCmdLineArgumentInt(argc, (const char **)argv, "numbodies");

    if (numBodies < 1) {
      printf(
          "Error: \"number of bodies\" specified %d is invalid.  Value should "
          "be >= 1\n",
          numBodies);
      exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
    } else if (numBodies % blockSize) {
      int newNumBodies = ((numBodies / blockSize) + 1) * blockSize;
      printf(
          "Warning: \"number of bodies\" specified %d is not a multiple of "
          "%d.\n",
          numBodies, blockSize);
      printf("Rounding up to the nearest multiple: %d.\n", newNumBodies);
      numBodies = newNumBodies;
    } else {
      printf("number of bodies = %d\n", numBodies);
    }
  }

  char *fname;

  if (getCmdLineArgumentString(argc, (const char **)argv, "tipsy", &fname)) {
    tipsyFile.assign(fname, strlen(fname));
    cycleDemo = false;
    bShowSliders = false;
  }

  if (numBodies <= 1024) {
    activeParams.m_clusterScale = 1.52f;
    activeParams.m_velocityScale = 2.f;
  } else if (numBodies <= 2048) {
    activeParams.m_clusterScale = 1.56f;
    activeParams.m_velocityScale = 2.64f;
  } else if (numBodies <= 4096) {
    activeParams.m_clusterScale = 1.68f;
    activeParams.m_velocityScale = 2.98f;
  } else if (numBodies <= 8192) {
    activeParams.m_clusterScale = 1.98f;
    activeParams.m_velocityScale = 2.9f;
  } else if (numBodies <= 16384) {
    activeParams.m_clusterScale = 1.54f;
    activeParams.m_velocityScale = 8.f;
  } else if (numBodies <= 32768) {
    activeParams.m_clusterScale = 1.44f;
    activeParams.m_velocityScale = 11.f;
  }

  // Create the demo -- either double (fp64) or float (fp32, default)
  // implementation
  NBodyDemo<float>::Create();

  NBodyDemo<float>::init(numBodies, numDevsRequested, blockSize,
                         !(benchmark || compareToCPU || useHostMem), useHostMem,
                         useP2P, useCpu, devID);
  NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_SHELL);

  if (bSupportDouble) {
    NBodyDemo<double>::Create();
    NBodyDemo<double>::init(numBodies, numDevsRequested, blockSize,
                            !(benchmark || compareToCPU || useHostMem),
                            useHostMem, useP2P, useCpu, devID);
    NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_SHELL);
  }

  if (fp64) {
    if (benchmark) {
      if (numIterations <= 0) {
        numIterations = 10;
      } else if (numIterations > 10) {
        printf("Advisory: setting a high number of iterations\n");
        printf("in benchmark mode may cause failure on Windows\n");
        printf("Vista and Win7. On these OSes, set iterations <= 10\n");
      }

      NBodyDemo<double>::runBenchmark(numIterations);
    } else if (compareToCPU) {
      bTestResults = NBodyDemo<double>::compareResults(numBodies);
    } else {
      glutDisplayFunc(display);
      glutReshapeFunc(reshape);
      glutMouseFunc(mouse);
      glutMotionFunc(motion);
      glutKeyboardFunc(key);
      glutSpecialFunc(special);
      glutIdleFunc(idle);

      if (!useCpu) {
        checkCudaErrors(cudaEventRecord(startEvent, 0));
      }

      glutMainLoop();
    }

  } else {
    if (benchmark) {
      if (numIterations <= 0) {
        numIterations = 10;
      }

      NBodyDemo<float>::runBenchmark(numIterations);
    } else if (compareToCPU) {
      bTestResults = NBodyDemo<float>::compareResults(numBodies);
    } else {
      glutDisplayFunc(display);
      glutReshapeFunc(reshape);
      glutMouseFunc(mouse);
      glutMotionFunc(motion);
      glutKeyboardFunc(key);
      glutSpecialFunc(special);
      glutIdleFunc(idle);

      if (!useCpu) {
        checkCudaErrors(cudaEventRecord(startEvent, 0));
      }

      glutMainLoop();
    }
  }

  finalize();
  exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
