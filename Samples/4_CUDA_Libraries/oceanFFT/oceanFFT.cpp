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
  FFT-based Ocean simulation
  based on original code by Yury Uralsky and Calvin Lin

  This sample demonstrates how to use CUFFT to synthesize and
  render an ocean surface in real-time.

  See Jerry Tessendorf's Siggraph course notes for more details:
  http://tessendorf.org/reports.html

  It also serves as an example of how to generate multiple vertex
  buffer streams from CUDA and render them using GLSL shaders.
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <math_constants.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <rendercheck_gl.h>

const char *sSDKsample = "CUDA FFT Ocean Simulation";

#define MAX_EPSILON 0.10f
#define THRESHOLD 0.15f
#define REFRESH_DELAY 10  // ms

////////////////////////////////////////////////////////////////////////////////
// constants
unsigned int windowW = 512, windowH = 512;

const unsigned int meshSize = 256;
const unsigned int spectrumW = meshSize + 4;
const unsigned int spectrumH = meshSize + 1;

const int frameCompare = 4;

// OpenGL vertex buffers
GLuint posVertexBuffer;
GLuint heightVertexBuffer, slopeVertexBuffer;
struct cudaGraphicsResource *cuda_posVB_resource, *cuda_heightVB_resource,
    *cuda_slopeVB_resource;  // handles OpenGL-CUDA exchange

GLuint indexBuffer;
GLuint shaderProg;
char *vertShaderPath = 0, *fragShaderPath = 0;

// mouse controls
int mouseOldX, mouseOldY;
int mouseButtons = 0;
float rotateX = 20.0f, rotateY = 0.0f;
float translateX = 0.0f, translateY = 0.0f, translateZ = -2.0f;

bool animate = true;
bool drawPoints = false;
bool wireFrame = false;
bool g_hasDouble = false;

// FFT data
cufftHandle fftPlan;
float2 *d_h0 = 0;  // heightfield at time 0
float2 *h_h0 = 0;
float2 *d_ht = 0;  // heightfield at time t
float2 *d_slope = 0;

// pointers to device object
float *g_hptr = NULL;
float2 *g_sptr = NULL;

// simulation parameters
const float g = 9.81f;        // gravitational constant
const float A = 1e-7f;        // wave scale factor
const float patchSize = 100;  // patch size
float windSpeed = 100.0f;
float windDir = CUDART_PI_F / 3.0f;
float dirDepend = 0.07f;

StopWatchInterface *timer = NULL;
float animTime = 0.0f;
float prevTime = 0.0f;
float animationRate = -0.001f;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

////////////////////////////////////////////////////////////////////////////////
// kernels
//#include <oceanFFT_kernel.cu>

extern "C" void cudaGenerateSpectrumKernel(float2 *d_h0, float2 *d_ht,
                                           unsigned int in_width,
                                           unsigned int out_width,
                                           unsigned int out_height,
                                           float animTime, float patchSize);

extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap, float2 *d_ht,
                                          unsigned int width,
                                          unsigned int height, bool autoTest);

extern "C" void cudaCalculateSlopeKernel(float *h, float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height);

////////////////////////////////////////////////////////////////////////////////
// forward declarations
void runAutoTest(int argc, char **argv);
void runGraphicsTest(int argc, char **argv);

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, int size);
void deleteVBO(GLuint *vbo);
void createMeshIndexBuffer(GLuint *id, int w, int h);
void createMeshPositionVBO(GLuint *id, int w, int h);
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int w, int h);
void timerEvent(int value);

// Cuda functionality
void runCuda();
void runCudaTest(char *exec_path);
void generate_h0(float2 *h0);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  // check for command line arguments
  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    animate = false;
    fpsLimit = frameCheckNumber;
    runAutoTest(argc, argv);
  } else {
    printf(
        "[%s]\n\n"
        "Left mouse button          - rotate\n"
        "Middle mouse button        - pan\n"
        "Right mouse button         - zoom\n"
        "'w' key                    - toggle wireframe\n",
        sSDKsample);

    runGraphicsTest(argc, argv);
  }

  exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);

  // Cuda init
  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Compute capability %d.%d\n", deviceProp.major, deviceProp.minor);

  // create FFT plan
  checkCudaErrors(cufftPlan2d(&fftPlan, meshSize, meshSize, CUFFT_C2C));

  // allocate memory
  int spectrumSize = spectrumW * spectrumH * sizeof(float2);
  checkCudaErrors(cudaMalloc((void **)&d_h0, spectrumSize));
  h_h0 = (float2 *)malloc(spectrumSize);
  generate_h0(h_h0);
  checkCudaErrors(cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice));

  int outputSize = meshSize * meshSize * sizeof(float2);
  checkCudaErrors(cudaMalloc((void **)&d_ht, outputSize));
  checkCudaErrors(cudaMalloc((void **)&d_slope, outputSize));

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  prevTime = sdkGetTimerValue(&timer);

  runCudaTest(argv[0]);

  checkCudaErrors(cudaFree(d_ht));
  checkCudaErrors(cudaFree(d_slope));
  checkCudaErrors(cudaFree(d_h0));
  checkCudaErrors(cufftDestroy(fftPlan));
  free(h_h0);

  exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void runGraphicsTest(int argc, char **argv) {
#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  printf("[%s] ", sSDKsample);
  printf("\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
    printf("[%s]\n", argv[0]);
    printf("   Does not explicitly support -device=n in OpenGL mode\n");
    printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
    printf(" > %s -device=n -qatest\n", argv[0]);
    printf("exiting...\n");

    exit(EXIT_SUCCESS);
  }

  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  if (false == initGL(&argc, argv)) {
    return;
  }

  findCudaDevice(argc, (const char **)argv);

  // create FFT plan
  checkCudaErrors(cufftPlan2d(&fftPlan, meshSize, meshSize, CUFFT_C2C));

  // allocate memory
  int spectrumSize = spectrumW * spectrumH * sizeof(float2);
  checkCudaErrors(cudaMalloc((void **)&d_h0, spectrumSize));
  h_h0 = (float2 *)malloc(spectrumSize);
  generate_h0(h_h0);
  checkCudaErrors(cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice));

  int outputSize = meshSize * meshSize * sizeof(float2);
  checkCudaErrors(cudaMalloc((void **)&d_ht, outputSize));
  checkCudaErrors(cudaMalloc((void **)&d_slope, outputSize));

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  prevTime = sdkGetTimerValue(&timer);

  // create vertex buffers and register with CUDA
  createVBO(&heightVertexBuffer, meshSize * meshSize * sizeof(float));
  checkCudaErrors(
      cudaGraphicsGLRegisterBuffer(&cuda_heightVB_resource, heightVertexBuffer,
                                   cudaGraphicsMapFlagsWriteDiscard));

  createVBO(&slopeVertexBuffer, outputSize);
  checkCudaErrors(
      cudaGraphicsGLRegisterBuffer(&cuda_slopeVB_resource, slopeVertexBuffer,
                                   cudaGraphicsMapFlagsWriteDiscard));

  // create vertex and index buffer for mesh
  createMeshPositionVBO(&posVertexBuffer, meshSize, meshSize);
  createMeshIndexBuffer(&indexBuffer, meshSize, meshSize);

  runCuda();

  // register callbacks
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

  // start rendering mainloop
  glutMainLoop();
}

float urand() { return rand() / (float)RAND_MAX; }

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss() {
  float u1 = urand();
  float u2 = urand();

  if (u1 < 1e-6f) {
    u1 = 1e-6f;
  }

  return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
}

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A,
               float dir_depend) {
  float k_squared = Kx * Kx + Ky * Ky;

  if (k_squared == 0.0f) {
    return 0.0f;
  }

  // largest possible wave from constant wind of velocity v
  float L = V * V / g;

  float k_x = Kx / sqrtf(k_squared);
  float k_y = Ky / sqrtf(k_squared);
  float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

  float phillips = A * expf(-1.0f / (k_squared * L * L)) /
                   (k_squared * k_squared) * w_dot_k * w_dot_k;

  // filter out waves moving opposite to wind
  if (w_dot_k < 0.0f) {
    phillips *= dir_depend;
  }

  // damp out waves with very small length w << l
  // float w = L / 10000;
  // phillips *= expf(-k_squared * w * w);

  return phillips;
}

// Generate base heightfield in frequency space
void generate_h0(float2 *h0) {
  for (unsigned int y = 0; y <= meshSize; y++) {
    for (unsigned int x = 0; x <= meshSize; x++) {
      float kx = (-(int)meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
      float ky = (-(int)meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

      float P = sqrtf(phillips(kx, ky, windDir, windSpeed, A, dirDepend));

      if (kx == 0.0f && ky == 0.0f) {
        P = 0.0f;
      }

      // float Er = urand()*2.0f-1.0f;
      // float Ei = urand()*2.0f-1.0f;
      float Er = gauss();
      float Ei = gauss();

      float h0_re = Er * P * CUDART_SQRT_HALF_F;
      float h0_im = Ei * P * CUDART_SQRT_HALF_F;

      int i = y * spectrumW + x;
      h0[i].x = h0_re;
      h0[i].y = h0_im;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda kernels
////////////////////////////////////////////////////////////////////////////////
void runCuda() {
  size_t num_bytes;

  // generate wave spectrum in frequency domain
  cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize,
                             animTime, patchSize);

  // execute inverse FFT to convert to spatial domain
  checkCudaErrors(cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE));

  // update heightmap values in vertex buffer
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_heightVB_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&g_hptr, &num_bytes, cuda_heightVB_resource));

  cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, false);

  // calculate slope for shading
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_slopeVB_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&g_sptr, &num_bytes, cuda_slopeVB_resource));

  cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, 0));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_slopeVB_resource, 0));
}

void runCudaTest(char *exec_path) {
  checkCudaErrors(
      cudaMalloc((void **)&g_hptr, meshSize * meshSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&g_sptr, meshSize * meshSize * sizeof(float2)));

  // generate wave spectrum in frequency domain
  cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize,
                             animTime, patchSize);

  // execute inverse FFT to convert to spatial domain
  checkCudaErrors(cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE));

  // update heightmap values
  cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, true);

  {
    float *hptr = (float *)malloc(meshSize * meshSize * sizeof(float));
    cudaMemcpy((void *)hptr, (void *)g_hptr,
               meshSize * meshSize * sizeof(float), cudaMemcpyDeviceToHost);
    sdkDumpBin((void *)hptr, meshSize * meshSize * sizeof(float),
               "spatialDomain.bin");

    if (!sdkCompareBin2BinFloat("spatialDomain.bin", "ref_spatialDomain.bin",
                                meshSize * meshSize, MAX_EPSILON, THRESHOLD,
                                exec_path)) {
      g_TotalErrors++;
    }

    free(hptr);
  }

  // calculate slope for shading
  cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

  {
    float2 *sptr = (float2 *)malloc(meshSize * meshSize * sizeof(float2));
    cudaMemcpy((void *)sptr, (void *)g_sptr,
               meshSize * meshSize * sizeof(float2), cudaMemcpyDeviceToHost);
    sdkDumpBin(sptr, meshSize * meshSize * sizeof(float2), "slopeShading.bin");

    if (!sdkCompareBin2BinFloat("slopeShading.bin", "ref_slopeShading.bin",
                                meshSize * meshSize * 2, MAX_EPSILON, THRESHOLD,
                                exec_path)) {
      g_TotalErrors++;
    }

    free(sptr);
  }

  checkCudaErrors(cudaFree(g_hptr));
  checkCudaErrors(cudaFree(g_sptr));
}

// void computeFPS()
//{
//    frameCount++;
//    fpsCount++;
//
//    if (fpsCount == fpsLimit) {
//        fpsCount = 0;
//    }
//}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display() {
  // run CUDA kernel to generate vertex positions
  if (animate) {
    runCuda();
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(translateX, translateY, translateZ);
  glRotatef(rotateX, 1.0, 0.0, 0.0);
  glRotatef(rotateY, 0.0, 1.0, 0.0);

  // render from the vbo
  glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
  glClientActiveTexture(GL_TEXTURE0);
  glTexCoordPointer(1, GL_FLOAT, 0, 0);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
  glClientActiveTexture(GL_TEXTURE1);
  glTexCoordPointer(2, GL_FLOAT, 0, 0);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);

  glUseProgram(shaderProg);

  // Set default uniform variables parameters for the vertex shader
  GLuint uniHeightScale, uniChopiness, uniSize;

  uniHeightScale = glGetUniformLocation(shaderProg, "heightScale");
  glUniform1f(uniHeightScale, 0.5f);

  uniChopiness = glGetUniformLocation(shaderProg, "chopiness");
  glUniform1f(uniChopiness, 1.0f);

  uniSize = glGetUniformLocation(shaderProg, "size");
  glUniform2f(uniSize, (float)meshSize, (float)meshSize);

  // Set default uniform variables parameters for the pixel shader
  GLuint uniDeepColor, uniShallowColor, uniSkyColor, uniLightDir;

  uniDeepColor = glGetUniformLocation(shaderProg, "deepColor");
  glUniform4f(uniDeepColor, 0.0f, 0.1f, 0.4f, 1.0f);

  uniShallowColor = glGetUniformLocation(shaderProg, "shallowColor");
  glUniform4f(uniShallowColor, 0.1f, 0.3f, 0.3f, 1.0f);

  uniSkyColor = glGetUniformLocation(shaderProg, "skyColor");
  glUniform4f(uniSkyColor, 1.0f, 1.0f, 1.0f, 1.0f);

  uniLightDir = glGetUniformLocation(shaderProg, "lightDir");
  glUniform3f(uniLightDir, 0.0f, 1.0f, 0.0f);
  // end of uniform settings

  glColor3f(1.0, 1.0, 1.0);

  if (drawPoints) {
    glDrawArrays(GL_POINTS, 0, meshSize * meshSize);
  } else {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

    glPolygonMode(GL_FRONT_AND_BACK, wireFrame ? GL_LINE : GL_FILL);
    glDrawElements(GL_TRIANGLE_STRIP, ((meshSize * 2) + 2) * (meshSize - 1),
                   GL_UNSIGNED_INT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }

  glDisableClientState(GL_VERTEX_ARRAY);
  glClientActiveTexture(GL_TEXTURE0);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glClientActiveTexture(GL_TEXTURE1);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);

  glUseProgram(0);

  glutSwapBuffers();

  // computeFPS();
}

void timerEvent(int value) {
  float time = sdkGetTimerValue(&timer);

  if (animate) {
    animTime += (time - prevTime) * animationRate;
  }

  glutPostRedisplay();
  prevTime = time;

  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup() {
  sdkDeleteTimer(&timer);
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_heightVB_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_slopeVB_resource));

  deleteVBO(&posVertexBuffer);
  deleteVBO(&heightVertexBuffer);
  deleteVBO(&slopeVertexBuffer);

  checkCudaErrors(cudaFree(d_h0));
  checkCudaErrors(cudaFree(d_slope));
  checkCudaErrors(cudaFree(d_ht));
  free(h_h0);
  cufftDestroy(fftPlan);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
    case (27):
      cleanup();
      exit(EXIT_SUCCESS);

    case 'w':
      wireFrame = !wireFrame;
      break;

    case 'p':
      drawPoints = !drawPoints;
      break;

    case ' ':
      animate = !animate;
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    mouseButtons |= 1 << button;
  } else if (state == GLUT_UP) {
    mouseButtons = 0;
  }

  mouseOldX = x;
  mouseOldY = y;
  glutPostRedisplay();
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - mouseOldX);
  dy = (float)(y - mouseOldY);

  if (mouseButtons == 1) {
    rotateX += dy * 0.2f;
    rotateY += dx * 0.2f;
  } else if (mouseButtons == 2) {
    translateX += dx * 0.01f;
    translateY -= dy * 0.01f;
  } else if (mouseButtons == 4) {
    translateZ += dy * 0.01f;
  }

  mouseOldX = x;
  mouseOldY = y;
}

void reshape(int w, int h) {
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (double)w / (double)h, 0.1, 10.0);

  windowW = w;
  windowH = h;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv) {
  // Create GL context
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(windowW, windowH);
  glutCreateWindow("CUDA FFT Ocean Simulation");

  vertShaderPath = sdkFindFilePath("ocean.vert", argv[0]);
  fragShaderPath = sdkFindFilePath("ocean.frag", argv[0]);

  if (vertShaderPath == NULL || fragShaderPath == NULL) {
    fprintf(stderr, "Error unable to find GLSL vertex and fragment shaders!\n");
    exit(EXIT_FAILURE);
  }

  // initialize necessary OpenGL extensions

  if (!isGLVersionSupported(2, 0)) {
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    return false;
  }

  if (!areGLExtensionsSupported(
          "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
    fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
    fprintf(stderr, "This sample requires:\n");
    fprintf(stderr, "  OpenGL version 1.5\n");
    fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
    fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
    cleanup();
    exit(EXIT_FAILURE);
  }

  // default initialization
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glEnable(GL_DEPTH_TEST);

  // load shader
  shaderProg = loadGLSLProgram(vertShaderPath, fragShaderPath);

  SDK_CHECK_ERROR_GL();
  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, int size) {
  // create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo) {
  glDeleteBuffers(1, vbo);
  *vbo = 0;
}

// create index buffer for rendering quad mesh
void createMeshIndexBuffer(GLuint *id, int w, int h) {
  int size = ((w * 2) + 2) * (h - 1) * sizeof(GLuint);

  // create index buffer
  glGenBuffers(1, id);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

  // fill with indices for rendering mesh as triangle strips
  GLuint *indices =
      (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

  if (!indices) {
    return;
  }

  for (int y = 0; y < h - 1; y++) {
    for (int x = 0; x < w; x++) {
      *indices++ = y * w + x;
      *indices++ = (y + 1) * w + x;
    }

    // start new strip with degenerate triangle
    *indices++ = (y + 1) * w + (w - 1);
    *indices++ = (y + 1) * w;
  }

  glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void createMeshPositionVBO(GLuint *id, int w, int h) {
  createVBO(id, w * h * 4 * sizeof(float));

  glBindBuffer(GL_ARRAY_BUFFER, *id);
  float *pos = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

  if (!pos) {
    return;
  }

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float u = x / (float)(w - 1);
      float v = y / (float)(h - 1);
      *pos++ = u * 2.0f - 1.0f;
      *pos++ = 0.0f;
      *pos++ = v * 2.0f - 1.0f;
      *pos++ = 1.0f;
    }
  }

  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Attach shader to a program
int attachShader(GLuint prg, GLenum type, const char *name) {
  GLuint shader;
  FILE *fp;
  int size, compiled;
  char *src;

  fp = fopen(name, "rb");

  if (!fp) {
    return 0;
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  src = (char *)malloc(size);

  fseek(fp, 0, SEEK_SET);
  fread(src, sizeof(char), size, fp);
  fclose(fp);

  shader = glCreateShader(type);
  glShaderSource(shader, 1, (const char **)&src, (const GLint *)&size);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint *)&compiled);

  if (!compiled) {
    char log[2048];
    int len;

    glGetShaderInfoLog(shader, 2048, (GLsizei *)&len, log);
    printf("Info log: %s\n", log);
    glDeleteShader(shader);
    return 0;
  }

  free(src);

  glAttachShader(prg, shader);
  glDeleteShader(shader);

  return 1;
}

// Create shader program from vertex shader and fragment shader files
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName) {
  GLint linked;
  GLuint program;

  program = glCreateProgram();

  if (!attachShader(program, GL_VERTEX_SHADER, vertFileName)) {
    glDeleteProgram(program);
    fprintf(stderr, "Couldn't attach vertex shader from file %s\n",
            vertFileName);
    return 0;
  }

  if (!attachShader(program, GL_FRAGMENT_SHADER, fragFileName)) {
    glDeleteProgram(program);
    fprintf(stderr, "Couldn't attach fragment shader from file %s\n",
            fragFileName);
    return 0;
  }

  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &linked);

  if (!linked) {
    glDeleteProgram(program);
    char temp[256];
    glGetProgramInfoLog(program, 256, 0, temp);
    fprintf(stderr, "Failed to link program: %s\n", temp);
    return 0;
  }

  return program;
}
