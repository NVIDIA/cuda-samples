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
  Mandelbrot sample
  submitted by Mark Granger, NewTek

  CUDA 2.0 SDK - updated with double precision support
  CUDA 2.1 SDK - updated to demonstrate software block scheduling using
  atomics
  CUDA 2.2 SDK - updated with drawing of Julia sets by Konstantin Kolchin,
  NVIDIA
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include "Mandelbrot_kernel.h"
#include "Mandelbrot_gold.h"

#define MAX_EPSILON_ERROR 5.0f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] = {"mandelbrot.ppm", "julia.ppm", NULL};

const char *sReference[] = {"Mandelbrot_fp32.ppm", "Mandelbrot_fp64.ppm", NULL};

const char *sReferenceJulia[] = {"referenceJulia_fp32.ppm",
                                 "referenceJulia_fp64.ppm", NULL};

bool g_isJuliaSet = false;
bool g_isMoving = true;
bool g_runCPU = false;

FILE *stream;
char g_ExecPath[300];

// Set to 1 to run on the CPU instead of the GPU for timing comparison.
#define RUN_CPU 0

// Set to 1 to time frame generation
#define RUN_TIMING 0

// Random number macros
#define RANDOMSEED(seed) ((seed) = ((seed)*1103515245 + 12345))
#define RANDOMBITS(seed, bits) ((unsigned int)RANDOMSEED(seed) >> (32 - (bits)))

// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange

// Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4 *d_dst = NULL;

// Original image width and height
int imageW = 800, imageH = 600;

// Starting iteration limit
int crunch = 512;

// Starting position and scale
double xOff = -0.5;
double yOff = 0.0;
double scale = 3.2;

// Starting stationary position and scale motion
double xdOff = 0.0;
double ydOff = 0.0;
double dscale = 1.0;

// Julia parameter
double xJParam = 0.0;
double yJParam = 0.0;

// Precision mode
// 0=single precision, 1=double single, 2=double
int precisionMode = 0;

// Starting animation frame and anti-aliasing pass
int animationFrame = 0;
int animationStep = 0;
int pass = 0;

// Starting color multipliers and random seed
int colorSeed = 0;
uchar4 colors;

// Timer ID
StopWatchInterface *hTimer = NULL;

// User interface variables
int lastx = 0;
int lasty = 0;
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

bool haveDoubles = true;
int numSMs = 0;   // number of multiprocessors
int version = 1;  // Compute Capability

// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;   // FPS count for averaging
int fpsLimit = 15;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

const char *sSDKsample = "CUDA Mandelbrot/Julia Set";

#define MAX_EPSILON 50
#define REFRESH_DELAY 10  // ms

#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define BUFFER_DATA(i) ((char *)0 + i)

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// This is specifically to enable the application to enable/disable vsync
typedef BOOL(WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

void setVSync(int interval) {
  if (WGL_EXT_swap_control) {
    wglSwapIntervalEXT =
        (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
    wglSwapIntervalEXT(interval);
  }
}
#endif

void computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
    sprintf(fps, "<CUDA %s Set> %3.1f fps",
            g_isJuliaSet ? "Julia" : "Mandelbrot", ifps);
    glutSetWindowTitle(fps);
    fpsCount = 0;

    fpsLimit = (int)MAX(1.f, (float)ifps);
    sdkResetTimer(&hTimer);
  }
}

void startJulia(const char *path) {
  g_isJuliaSet = true;
  g_isMoving = false;

  if ((path == NULL) || (stream = fopen(path, "r")) == NULL) {
    printf(
        "JuliaSet: params.txt could not be opened.  Using default "
        "parameters\n");
    xOff = -0.085760;
    yOff = 0.007040;
    scale = 3.200000;
    xJParam = -0.172400;
    yJParam = -0.652693;
  } else {
    fseek(stream, 0L, SEEK_SET);
    fscanf(stream, "%lf %lf %lf %lf %lf", &xOff, &yOff, &scale, &xJParam,
           &yJParam);
    fclose(stream);
  }

  xdOff = 0.0;
  ydOff = 0.0;
  dscale = 1.0;
  pass = 0;
}

// Get a sub-pixel sample location
void GetSample(int sampleIndex, float &x, float &y) {
  static const unsigned char pairData[128][2] = {
      {64, 64},  {0, 0},     {1, 63},    {63, 1},    {96, 32},  {97, 95},
      {36, 96},  {30, 31},   {95, 127},  {4, 97},    {33, 62},  {62, 33},
      {31, 126}, {67, 99},   {99, 65},   {2, 34},    {81, 49},  {19, 80},
      {113, 17}, {112, 112}, {80, 16},   {115, 81},  {46, 15},  {82, 79},
      {48, 78},  {16, 14},   {49, 113},  {114, 48},  {45, 45},  {18, 47},
      {20, 109}, {79, 115},  {65, 82},   {52, 94},   {15, 124}, {94, 111},
      {61, 18},  {47, 30},   {83, 100},  {98, 50},   {110, 2},  {117, 98},
      {50, 59},  {77, 35},   {3, 114},   {5, 77},    {17, 66},  {32, 13},
      {127, 20}, {34, 76},   {35, 110},  {100, 12},  {116, 67}, {66, 46},
      {14, 28},  {23, 93},   {102, 83},  {86, 61},   {44, 125}, {76, 3},
      {109, 36}, {6, 51},    {75, 89},   {91, 21},   {60, 117}, {29, 43},
      {119, 29}, {74, 70},   {126, 87},  {93, 75},   {71, 24},  {106, 102},
      {108, 58}, {89, 9},    {103, 23},  {72, 56},   {120, 8},  {88, 40},
      {11, 88},  {104, 120}, {57, 105},  {118, 122}, {53, 6},   {125, 44},
      {43, 68},  {58, 73},   {24, 22},   {22, 5},    {40, 86},  {122, 108},
      {87, 90},  {56, 42},   {70, 121},  {8, 7},     {37, 52},  {25, 55},
      {69, 11},  {10, 106},  {12, 38},   {26, 69},   {27, 116}, {38, 25},
      {59, 54},  {107, 72},  {121, 57},  {39, 37},   {73, 107}, {85, 123},
      {28, 103}, {123, 74},  {55, 85},   {101, 41},  {42, 104}, {84, 27},
      {111, 91}, {9, 19},    {21, 39},   {90, 53},   {41, 60},  {54, 26},
      {92, 119}, {51, 71},   {124, 101}, {68, 92},   {78, 10},  {13, 118},
      {7, 84},   {105, 4}};

  x = (1.0f / 128.0f) * (0.5f + (float)pairData[sampleIndex][0]);
  y = (1.0f / 128.0f) * (0.5f + (float)pairData[sampleIndex][1]);
}  // GetSample

// render Mandelbrot image using CUDA or CPU
void renderImage(bool bUseOpenGL, bool fp64, int mode) {
#if RUN_TIMING
  pass = 0;
#endif

  if (pass < 128) {
    if (g_runCPU) {
      int startPass = pass;
      float xs, ys;
      sdkResetTimer(&hTimer);

      if (bUseOpenGL) {
        // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst,
        // gl_PBO));
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void **)&d_dst, &num_bytes, cuda_pbo_resource));
      }

      // Get the anti-alias sub-pixel sample location
      GetSample(pass & 127, xs, ys);

      // Get the pixel scale and offset
      double s = scale / (double)imageW;
      double x = (xs - (double)imageW * 0.5f) * s + xOff;
      double y = (ys - (double)imageH * 0.5f) * s + yOff;

      // Run the mandelbrot generator
      // Use the adaptive sampling version when animating.
      if (pass && !startPass) {
        if (precisionMode)
          RunMandelbrotDSGold1(h_Src, imageW, imageH, crunch, x, y, xJParam,
                               yJParam, s, colors, pass++, animationFrame,
                               g_isJuliaSet);
        else
          RunMandelbrotGold1(h_Src, imageW, imageH, crunch, (float)x, (float)y,
                             (float)xJParam, (float)yJParam, (float)s, colors,
                             pass++, animationFrame, g_isJuliaSet);
      } else {
        if (precisionMode)
          RunMandelbrotDSGold0(h_Src, imageW, imageH, crunch, x, y, xJParam,
                               yJParam, s, colors, pass++, animationFrame,
                               g_isJuliaSet);
        else
          RunMandelbrotGold0(h_Src, imageW, imageH, crunch, (float)x, (float)y,
                             (float)xJParam, (float)yJParam, (float)s, colors,
                             pass++, animationFrame, g_isJuliaSet);
      }

      checkCudaErrors(cudaMemcpy(d_dst, h_Src, imageW * imageH * sizeof(uchar4),
                                 cudaMemcpyHostToDevice));

      if (bUseOpenGL) {
        // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
      }

#if RUN_TIMING
      printf("CPU = %5.8f\n", 0.001f * sdkGetTimerValue(&hTimer));
#endif
    } else {  // this is the GPU Path
      float timeEstimate;
      int startPass = pass;
      sdkResetTimer(&hTimer);

      if (bUseOpenGL) {
        // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst,
        // gl_PBO));
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void **)&d_dst, &num_bytes, cuda_pbo_resource));
      }

      // Render anti-aliasing passes until we run out time (60fps approximately)
      do {
        float xs, ys;

        // Get the anti-alias sub-pixel sample location
        GetSample(pass & 127, xs, ys);

        // Get the pixel scale and offset
        double s = scale / (float)imageW;
        double x = (xs - (double)imageW * 0.5f) * s + xOff;
        double y = (ys - (double)imageH * 0.5f) * s + yOff;

        // Run the mandelbrot generator
        // Use the adaptive sampling version when animating.
        if (pass && !startPass)
          RunMandelbrot1(d_dst, imageW, imageH, crunch, x, y, xJParam, yJParam,
                         s, colors, pass++, animationFrame, precisionMode,
                         numSMs, g_isJuliaSet, version);
        else
          RunMandelbrot0(d_dst, imageW, imageH, crunch, x, y, xJParam, yJParam,
                         s, colors, pass++, animationFrame, precisionMode,
                         numSMs, g_isJuliaSet, version);

        // Estimate the total time of the frame if one more pass is rendered
        timeEstimate =
            0.1f * sdkGetTimerValue(&hTimer) *
            ((float)(pass + 1 - startPass) / (float)(pass - startPass));
      } while ((pass < 128) && (timeEstimate < 1.0f / 60.0f) && !RUN_TIMING);

      if (bUseOpenGL) {
        // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
      }

#if RUN_TIMING
            printf("GPU = %5.8f\n", 0.001f * sdkGetTimerValue(&hTimer);
#endif
    }
  }
}

// OpenGL display function
void displayFunc(void) {
  sdkStartTimer(&hTimer);

  if ((xdOff != 0.0) || (ydOff != 0.0)) {
    if (g_isMoving || !g_isJuliaSet) {
      xOff += xdOff;
      yOff += ydOff;
    } else {
      xJParam += xdOff;
      yJParam += ydOff;
    }

    pass = 0;
  }

  if (dscale != 1.0) {
    scale *= dscale;
    pass = 0;
  }

  if (animationStep) {
    animationFrame -= animationStep;
    pass = 0;
  }

  // render the Mandelbrot image
  renderImage(true, g_isJuliaSet, precisionMode);

  // load texture from PBO
  //  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
                  GL_UNSIGNED_BYTE, BUFFER_DATA(0));
  //  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  // fragment program is required to display floating point texture
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glDisable(GL_DEPTH_TEST);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(0.0f, 0.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, 0.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(0.0f, 1.0f);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);

  sdkStopTimer(&hTimer);
  glutSwapBuffers();

  computeFPS();
}  // displayFunc

void cleanup() {
  if (h_Src) {
    free(h_Src);
    h_Src = 0;
  }

  sdkStopTimer(&hTimer);
  sdkDeleteTimer(&hTimer);

  // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  glDeleteBuffers(1, &gl_PBO);
  glDeleteTextures(1, &gl_Tex);
  glDeleteProgramsARB(1, &gl_Shader);
}

void initMenus();

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int) {
  int seed;

  switch (k) {
    case '\033':
    case 'q':
    case 'Q':
      printf("Shutting down...\n");

#if defined(__APPLE__) || defined(MACOSX)
      exit(EXIT_SUCCESS);
#else
      glutDestroyWindow(glutGetWindow());
      return;
#endif
      break;

    case '?':
      printf("xOff = %5.8f\n", xOff);
      printf("yOff = %5.8f\n", yOff);
      printf("scale = %e\n", scale);
      printf("detail = %d\n", crunch);
      printf("color = %d\n", colorSeed);
      printf("xJParam = %5.8f\n", xJParam);
      printf("yJParam = %5.8f\n", yJParam);
      printf("\n");
      break;

    case 'e':
    case 'E':
      // Reset all values to their defaults
      g_isJuliaSet = false;
      g_isMoving = true;
      g_runCPU = false;
      printf(
          "All parameters are reset to defaults. GPU implementation is "
          "used.\n");
      xOff = -0.5;
      yOff = 0.0;
      scale = 3.2;
      xdOff = 0.0;
      ydOff = 0.0;
      dscale = 1.0;
      colorSeed = 0;
      colors.x = 3;
      colors.y = 5;
      colors.z = 7;
      crunch = 512;
      animationFrame = 0;
      animationStep = 0;
      xJParam = 0.0;
      yJParam = 0.0;
      pass = 0;
      break;

    case 'c':
      seed = ++colorSeed;

      if (seed) {
        colors.x = RANDOMBITS(seed, 4);
        colors.y = RANDOMBITS(seed, 4);
        colors.z = RANDOMBITS(seed, 4);
      } else {
        colors.x = 3;
        colors.y = 5;
        colors.z = 7;
      }

      pass = 0;
      break;

    case 'C':
      seed = --colorSeed;

      if (seed) {
        colors.x = RANDOMBITS(seed, 4);
        colors.y = RANDOMBITS(seed, 4);
        colors.z = RANDOMBITS(seed, 4);
      } else {
        colors.x = 3;
        colors.y = 5;
        colors.z = 7;
      }

      pass = 0;
      break;

    case 'a':
      if (animationStep < 0) {
        animationStep = 0;
      } else {
        animationStep++;

        if (animationStep > 8) {
          animationStep = 8;
        }
      }

      break;

    case 'A':
      if (animationStep > 0) {
        animationStep = 0;
      } else {
        animationStep--;

        if (animationStep < -8) {
          animationStep = -8;
        }
      }

      break;

    case 'd':
      if (2 * crunch <= MIN(numSMs * (version < 20 ? 512 : 2048), 0x4000)) {
        crunch *= 2;
        pass = 0;
      }

      printf("detail = %d\n", crunch);
      break;

    case 'D':
      if (crunch > 2) {
        crunch /= 2;
        pass = 0;
      }

      printf("detail = %d\n", crunch);
      break;

    case 'r':
      colors.x -= 1;
      pass = 0;
      break;

    case 'R':
      colors.x += 1;
      pass = 0;
      break;

    case 'g':
      colors.y -= 1;
      pass = 0;
      break;

    case 'G':
      colors.y += 1;
      pass = 0;
      break;

    case 'b':
      colors.z -= 1;
      pass = 0;
      break;

    case 'B':
      colors.z += 1;
      pass = 0;
      break;

    case 's':
    case 'S':
      if (g_runCPU) {
        g_runCPU = false;
        printf("GPU implementation\n");
      } else {
        g_runCPU = true;
        printf("CPU implementation\n");
      }

      pass = 0;
      glutDestroyMenu(glutGetMenu());
      initMenus();
      break;

    case 'j':
    case 'J':

      // toggle between Mandelbrot and Julia sets and reset all parameters
      if (!g_isJuliaSet) {  // settings for Julia
        g_isJuliaSet = true;
        startJulia("params.txt");
      } else {  // settings for Mandelbrot
        g_isJuliaSet = false;
        g_isMoving = true;
        xOff = -0.5;
        yOff = 0.0;
        scale = 3.2;
        xdOff = 0.0;
        ydOff = 0.0;
        dscale = 1.0;
        colorSeed = 0;
        colors.x = 3;
        colors.y = 5;
        colors.z = 7;
        crunch = 512;
        animationFrame = 0;
        animationStep = 0;
        pass = 0;
      }

      char fps[30];
      sprintf(fps, "<CUDA %s Set>", g_isJuliaSet ? "Julia" : "Mandelbrot");
      glutSetWindowTitle(fps);

      break;

    case 'm':
    case 'M':
      if (g_isJuliaSet) {
        g_isMoving = !g_isMoving;
        pass = 0;
      }

      break;

    case 'p':
    case 'P':
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
      if (fopen_s(&stream, "params.txt", "w") != 0)
#else
      if ((stream = fopen("params.txt", "w")) == NULL)
#endif
      {
        printf("The file params.txt was not opened\n");
        break;
      }

      fprintf(stream, "%f %f %f %f %f\n", xOff, yOff, scale, xJParam, yJParam);
      fclose(stream);
      break;

    case 'o':
    case 'O':
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
      if (fopen_s(&stream, "params.txt", "r") != 0)
#else
      if ((stream = fopen("params.txt", "r")) == NULL)
#endif
      {
        printf("The file params.txt was not opened\n");

        break;
      }

      fseek(stream, 0L, SEEK_SET);
      fscanf(stream, "%lf %lf %lf %lf %lf", &xOff, &yOff, &scale, &xJParam,
             &yJParam);
      xdOff = 0.0;
      ydOff = 0.0;
      dscale = 1.0;
      fclose(stream);
      pass = 0;
      break;

    case '4':  // Left arrow key
      xOff -= 0.05f * scale;
      pass = 0;
      break;

    case '8':  // Up arrow key
      yOff += 0.05f * scale;
      pass = 0;
      break;

    case '6':  // Right arrow key
      xOff += 0.05f * scale;
      pass = 0;
      break;

    case '2':  // Down arrow key
      yOff -= 0.05f * scale;
      pass = 0;
      break;

    case '+':
      scale /= 1.1f;
      pass = 0;
      break;

    case '-':
      scale *= 1.1f;
      pass = 0;
      break;

    default:
      break;
  }

}  // keyboardFunc

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y) {
  if (button == 0) {
    leftClicked = !leftClicked;
  }

  if (button == 1) {
    middleClicked = !middleClicked;
  }

  if (button == 2) {
    rightClicked = !rightClicked;
  }

  int modifiers = glutGetModifiers();

  if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT)) {
    leftClicked = 0;
    middleClicked = 1;
  }

  if (state == GLUT_UP) {
    leftClicked = 0;
    middleClicked = 0;
  }

  lastx = x;
  lasty = y;
  xdOff = 0.0;
  ydOff = 0.0;
  dscale = 1.0;
}  // clickFunc

// OpenGL mouse motion function
void motionFunc(int x, int y) {
  double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
  double fy = (double)(lasty - y) / 50.0 / (double)(imageH);

  if (leftClicked) {
    xdOff = fx * scale;
    ydOff = fy * scale;
  } else {
    xdOff = 0.0f;
    ydOff = 0.0f;
  }

  if (middleClicked)
    if (fy > 0.0f) {
      dscale = 1.0 - fy;
      dscale = dscale < 1.05 ? dscale : 1.05;
    } else {
      dscale = 1.0 / (1.0 + fy);
      dscale = dscale > (1.0 / 1.05) ? dscale : (1.0 / 1.05);
    }
  else {
    dscale = 1.0;
  }
}  // motionFunc

void timerEvent(int value) {
  if (glutGetWindow()) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  }
}

void mainMenu(int i) {
  precisionMode = i;
  pass = 0;
}

void initMenus() {
  glutCreateMenu(mainMenu);

  if (!g_runCPU) {
    glutAddMenuEntry("Hardware single precision", 0);

    if (numSMs > 2) {
      glutAddMenuEntry("Emulated double-single precision", 1);
    }

    if (haveDoubles) {
      glutAddMenuEntry("Hardware double precision", 2);
    }
  } else {
    glutAddMenuEntry("Software single precision", 0);
    glutAddMenuEntry("Software double precision", 1);
  }

  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

// gl_Shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code) {
  GLuint program_id;
  glGenProgramsARB(1, &program_id);
  glBindProgramARB(program_type, program_id);
  glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
                     (GLsizei)strlen(code), (GLubyte *)code);

  GLint error_pos;
  glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

  if (error_pos != -1) {
    const GLubyte *error_string;
    error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
    fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos,
            error_string);
    return 0;
  }

  return program_id;
}

void initOpenGLBuffers(int w, int h) {
  // delete old buffers
  if (h_Src) {
    free(h_Src);
    h_Src = 0;
  }

  if (gl_Tex) {
    glDeleteTextures(1, &gl_Tex);
    gl_Tex = 0;
  }

  if (gl_PBO) {
    // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &gl_PBO);
    gl_PBO = 0;
  }

  // allocate new buffers
  h_Src = (uchar4 *)malloc(w * h * 4);

  printf("Creating GL texture...\n");
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_Tex);
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               h_Src);
  printf("Texture created.\n");

  printf("Creating PBO...\n");
  glGenBuffers(1, &gl_PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
  // While a PBO is registered to CUDA, it can't be used
  // as the destination for OpenGL drawing calls.
  // But in our particular case OpenGL is only used
  // to display the content of the PBO, specified by CUDA kernels,
  // so we need to register/unregister it only once.

  // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
      &cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
  printf("PBO created.\n");

  // load shader program
  gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void reshapeFunc(int w, int h) {
  glViewport(0, 0, w, h);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  if (w != 0 && h != 0)  // Do not call when window is minimized that is when
                         // width && height == 0
    initOpenGLBuffers(w, h);

  imageW = w;
  imageH = h;
  pass = 0;

  glutPostRedisplay();
}

void initGL(int *argc, char **argv) {
  printf("Initializing GLUT...\n");
  glutInit(argc, argv);

  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(imageW, imageH);
  glutInitWindowPosition(0, 0);
  glutCreateWindow(argv[0]);

  glutDisplayFunc(displayFunc);
  glutKeyboardFunc(keyboardFunc);
  glutMouseFunc(clickFunc);
  glutMotionFunc(motionFunc);
  glutReshapeFunc(reshapeFunc);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  initMenus();

  if (!isGLVersionSupported(1, 5) ||
      !areGLExtensionsSupported(
          "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
    fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
    fprintf(stderr, "This sample requires:\n");
    fprintf(stderr, "  OpenGL version 1.5\n");
    fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
    fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
    exit(EXIT_SUCCESS);
  }

  printf("OpenGL window created.\n");
}

void initData(int argc, char **argv) {
  // check for hardware double precision support
  int dev = 0;
  dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  version = deviceProp.major * 10 + deviceProp.minor;

  numSMs = deviceProp.multiProcessorCount;

  // initialize some of the arguments
  if (checkCmdLineFlag(argc, (const char **)argv, "xOff")) {
    xOff = getCmdLineArgumentFloat(argc, (const char **)argv, "xOff");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "yOff")) {
    yOff = getCmdLineArgumentFloat(argc, (const char **)argv, "yOff");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "scale")) {
    scale = getCmdLineArgumentFloat(argc, (const char **)argv, "xOff");
  }

  colors.w = 0;
  colors.x = 3;
  colors.y = 5;
  colors.z = 7;
  printf("Data initialization done.\n");
}

////////////////////////////////////////////////////////////////////////////////
// runAutoTest validates the Mandelbrot and Julia sets without using OpenGL
////////////////////////////////////////////////////////////////////////////////
int runSingleTest(int argc, char **argv) {
  char dump_file[256], *ref_file = NULL;
  bool haveDouble = false;

  printf("* Running Automatic Test: <%s>\n", sSDKsample);

  strcpy(dump_file, (const char *)"rendered_image.ppm");
  // We've already determined that file has been passed in as input, we can grab
  // the file here
  getCmdLineArgumentString(argc, (const char **)argv, "file",
                           (char **)&ref_file);

  if (checkCmdLineFlag(argc, (const char **)argv, "fp64")) {
    haveDouble = true;
  }

  // initialize Data for CUDA
  initData(argc, argv);

  // Allocate memory for renderImage (to be able to render into a CUDA memory
  // buffer)
  checkCudaErrors(
      cudaMalloc((void **)&d_dst, (imageW * imageH * sizeof(uchar4))));

  // Allocate memory for cpu buffer
  unsigned char *h_dst =
      (unsigned char *)malloc(sizeof(uchar4) * imageH * imageW);

  if (g_isJuliaSet) {
    char *ref_path = sdkFindFilePath("params.txt", argv[0]);
    startJulia(ref_path);

    for (int i = 0; i < 50; i++) {
      renderImage(false, haveDouble, 0);
    }

    checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW * imageH * sizeof(uchar4),
                               cudaMemcpyDeviceToHost));
    sdkSavePPM4ub(dump_file, h_dst, imageW, imageH);
  } else {
    // Mandelbrot Set
    for (int i = 0; i < 50; i++) {
      renderImage(false, haveDouble, 0);
    }

    checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW * imageH * sizeof(uchar4),
                               cudaMemcpyDeviceToHost));
    sdkSavePPM4ub(dump_file, h_dst, imageW, imageH);
  }

  printf("\n[%s], %s Set, %s -> Saved File\n", dump_file,
         (g_isJuliaSet ? "Julia" : "Mandelbrot"),
         (haveDouble ? "(fp64 double precision)" : "(fp32 single precision)"));

  if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, argv[0]),
                     MAX_EPSILON_ERROR, 0.15f, false)) {
    printf("Images \"%s\", \"%s\" are different\n", ref_file, dump_file);
    g_TotalErrors++;
  } else {
    printf("Images \"%s\", \"%s\" are matching\n", ref_file, dump_file);
  }

  checkCudaErrors(cudaFree(d_dst));
  free(h_dst);

  return true;
}

// Performance Test
void runBenchmark(int argc, char **argv) {
  int N = 1000;
  // initialize Data for CUDA
  initData(argc, argv);

  printf("\n* Run Performance Test\n");
  printf("Image Size %d x %d\n", imageW, imageH);
  printf("Double Precision\n");
  printf("%d Iterations\n", N);

  // Allocate memory for renderImage (to be able to render into a CUDA memory
  // buffer)
  checkCudaErrors(
      cudaMalloc((void **)&d_dst, (imageW * imageH * sizeof(uchar4))));

  float xs, ys;

  // Get the anti-alias sub-pixel sample location
  GetSample(0, xs, ys);

  double s = scale / (float)imageW;
  double x = (xs - (double)imageW * 0.5f) * s + xOff;
  double y = (ys - (double)imageH * 0.5f) * s + yOff;

  // Create Timers
  StopWatchInterface *kernel_timer = NULL;
  sdkCreateTimer(&kernel_timer);
  sdkStartTimer(&kernel_timer);

  // render Mandelbrot set and verify
  for (int i = 0; i < N; i++) {
    RunMandelbrot0(d_dst, imageW, imageH, crunch, x, y, xJParam, yJParam, s,
                   colors, pass++, animationFrame, 2, numSMs, g_isJuliaSet,
                   version);
    cudaDeviceSynchronize();
  }

  sdkStopTimer(&hTimer);
  float ExecutionTime = sdkGetTimerValue(&kernel_timer);

  float PixelsPerSecond =
      (float)imageW * (float)imageH * N / (ExecutionTime / 1000.0f);

  printf("\nMegaPixels Per Second %.4f\n", PixelsPerSecond / 1e6);

  checkCudaErrors(cudaFree(d_dst));
  sdkDeleteTimer(&kernel_timer);
}

void printHelp() {
  printf("[Mandelbrot]\n");
  printf("\tUsage Parameters\n");
  printf("\t-device=n        (requires to be in non-graphics mode)\n");
  printf("\t-file=output.ppm (output file for image testing)\n");
  printf("\t-mode=0,1        (0=Mandelbrot Set, 1=Julia Set)\n");
  printf("\t-fp64            (run in double precision mode)\n");
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  printf("[%s] - Starting...\n", sSDKsample);

  // parse command line arguments
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp();
    exit(EXIT_SUCCESS);
  }

  int mode = 0;

  if (checkCmdLineFlag(argc, (const char **)argv, "mode")) {
    mode = getCmdLineArgumentInt(argc, (const char **)argv, "mode");
    g_isJuliaSet = mode;

  } else {
    g_isJuliaSet = 0;
  }

  // Set the initial parameters for either Mandelbrot and Julia sets and reset
  // all parameters
  if (g_isJuliaSet)  // settings for Julia
  {
    char *ref_path = sdkFindFilePath("params.txt", argv[0]);
    startJulia(ref_path);
  } else  // settings for Mandelbrot
  {
    g_isMoving = true;
    xOff = -0.5;
    yOff = 0.0;
    scale = 3.2;
    xdOff = 0.0;
    ydOff = 0.0;
    dscale = 1.0;
    colorSeed = 0;
    colors.x = 3;
    colors.y = 5;
    colors.z = 7;
    crunch = 512;
    animationFrame = 0;
    animationStep = 0;
    pass = 0;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    fpsLimit = frameCheckNumber;

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    findCudaDevice(argc, (const char **)argv);  // no OpenGL usage

    // We run the Automated Testing code path
    runSingleTest(argc, argv);

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
  } else if (checkCmdLineFlag(argc, (const char **)argv, "benchmark")) {
    // run benchmark
    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    findCudaDevice(argc, (const char **)argv);

    // We run the Automated Performance Test
    runBenchmark(argc, argv);

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  else if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
    printf("[%s]\n", argv[0]);
    printf("   Does not explicitly support -device=n in OpenGL mode\n");
    printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
    printf(" > %s -device=n -file=<image_name>.ppm\n", argv[0]);
    printf("exiting...\n");
    exit(EXIT_SUCCESS);
  }

  // Otherwise it succeeds, we will continue to run this sample
  initData(argc, argv);

  // Initialize OpenGL context first before the CUDA context is created.  This
  // is needed
  // to achieve optimal performance with OpenGL/CUDA interop.
  initGL(&argc, argv);
  initOpenGLBuffers(imageW, imageH);

  printf("Starting GLUT main loop...\n");
  printf("\n");
  printf("Press [s] to toggle between GPU and CPU implementations\n");
  printf("Press [j] to toggle between Julia and Mandelbrot sets\n");
  printf("Press [r] or [R] to decrease or increase red color channel\n");
  printf("Press [g] or [G] to decrease or increase green color channel\n");
  printf("Press [b] or [B] to decrease or increase blue color channel\n");
  printf("Press [e] to reset\n");
  printf("Press [a] or [A] to animate colors\n");
  printf("Press [c] or [C] to change colors\n");
  printf("Press [d] or [D] to increase or decrease the detail\n");
  printf("Press [p] to record main parameters to file params.txt\n");
  printf("Press [o] to read main parameters from file params.txt\n");
  printf(
      "Left mouse button + drag = move (Mandelbrot or Julia) or animate "
      "(Julia)\n");
  printf(
      "Press [m] to toggle between move and animate (Julia) for left mouse "
      "button\n");
  printf("Middle mouse button + drag = Zoom\n");
  printf("Right mouse button = Menu\n");
  printf("Press [?] to print location and scale\n");
  printf("Press [q] to exit\n");
  printf("\n");

  sdkCreateTimer(&hTimer);
  sdkStartTimer(&hTimer);

#if defined(__APPLE__) || defined(MACOSX)
  atexit(cleanup);
#else
  glutCloseFunc(cleanup);
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  setVSync(0);
#endif

  glutMainLoop();
}  // main
