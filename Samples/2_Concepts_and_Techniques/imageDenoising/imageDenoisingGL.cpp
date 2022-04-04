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
 * This sample demonstrates two adaptive image denoising techniques:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter technique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageDenoising.h"

// includes, project
#include <helper_functions.h>  // includes for helper utility functions
#include <helper_cuda.h>  // includes for cuda error checking and initialization

const char *sSDKsample = "CUDA ImageDenoising";

const char *filterMode[] = {"Passthrough", "KNN method", "NLM method",
                            "Quick NLM(NLM2) method", NULL};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] = {"image_passthru.ppm", "image_knn.ppm",
                           "image_nlm.ppm", "image_nlm2.ppm", NULL};

const char *sReference[] = {"ref_passthru.ppm", "ref_knn.ppm", "ref_nlm.ppm",
                            "ref_nlm2.ppm", NULL};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
// Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int g_Kernel = 0;
bool g_FPS = false;
bool g_Diag = false;
StopWatchInterface *timer = NULL;

// Algorithms global parameters
const float noiseStep = 0.025f;
const float lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float lerpC = 0.2f;

const int frameN = 24;
int frameCounter = 0;

#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY 10  // ms

void cleanup();

void computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "<%s>: %3.1f fps", filterMode[g_Kernel], ifps);

    glutSetWindowTitle(fps);
    fpsCount = 0;

    // fpsLimit = (int)MAX(ifps, 1.f);
    sdkResetTimer(&timer);
  }
}

void runImageFilters(TColor *d_dst) {
  switch (g_Kernel) {
    case 0:
      cuda_Copy(d_dst, imageW, imageH, texImage);
      break;

    case 1:
      if (!g_Diag) {
        cuda_KNN(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC,
                 texImage);
      } else {
        cuda_KNNdiag(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC,
                     texImage);
      }

      break;

    case 2:
      if (!g_Diag) {
        cuda_NLM(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC,
                 texImage);
      } else {
        cuda_NLMdiag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC,
                     texImage);
      }

      break;

    case 3:
      if (!g_Diag) {
        cuda_NLM2(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC,
                  texImage);
      } else {
        cuda_NLM2diag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise),
                      lerpC, texImage);
      }

      break;
  }

  getLastCudaError("Filtering kernel execution failed.\n");
}

void displayFunc(void) {
  sdkStartTimer(&timer);
  TColor *d_dst = NULL;
  size_t num_bytes;

  if (frameCounter++ == 0) {
    sdkResetTimer(&timer);
  }

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  getLastCudaError("cudaGraphicsMapResources failed");
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_dst, &num_bytes, cuda_pbo_resource));
  getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

  runImageFilters(d_dst);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

  // Common display code path
  {
    glClear(GL_COLOR_BUFFER_BIT);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
                    GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(2, 0);
    glVertex2f(+3, -1);
    glTexCoord2f(0, 2);
    glVertex2f(-1, +3);
    glEnd();
    glFinish();
  }

  if (frameCounter == frameN) {
    frameCounter = 0;

    if (g_FPS) {
      printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001));
      g_FPS = false;
    }
  }

  glutSwapBuffers();
  glutReportErrors();

  sdkStopTimer(&timer);

  computeFPS();
}

void timerEvent(int value) {
  if (glutGetWindow()) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  }
}

void keyboard(unsigned char k, int /*x*/, int /*y*/) {
  switch (k) {
    case 27:
    case 'q':
    case 'Q':
#if defined(__APPLE__) || defined(MACOSX)
      exit(EXIT_SUCCESS);
#else
      glutDestroyWindow(glutGetWindow());
      return;
#endif

    case '1':
      printf("Passthrough.\n");
      g_Kernel = 0;
      break;

    case '2':
      printf("KNN method \n");
      g_Kernel = 1;
      break;

    case '3':
      printf("NLM method\n");
      g_Kernel = 2;
      break;

    case '4':
      printf("Quick NLM(NLM2) method\n");
      g_Kernel = 3;
      break;

    case '*':
      printf(g_Diag ? "LERP highlighting mode.\n" : "Normal mode.\n");
      g_Diag = !g_Diag;
      break;

    case 'n':
      printf("Decrease noise level.\n");
      knnNoise -= noiseStep;
      nlmNoise -= noiseStep;
      break;

    case 'N':
      printf("Increase noise level.\n");
      knnNoise += noiseStep;
      nlmNoise += noiseStep;
      break;

    case 'l':
      printf("Decrease LERP quotient.\n");
      lerpC = MAX(lerpC - lerpStep, 0.0f);
      break;

    case 'L':
      printf("Increase LERP quotient.\n");
      lerpC = MIN(lerpC + lerpStep, 1.0f);
      break;

    case 'f':
    case 'F':
      g_FPS = true;
      break;

    case '?':
      printf("lerpC = %5.5f\n", lerpC);
      printf("knnNoise = %5.5f\n", knnNoise);
      printf("nlmNoise = %5.5f\n", nlmNoise);
      break;
  }
}

int initGL(int *argc, char **argv) {
  printf("Initializing GLUT...\n");
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(imageW, imageH);
  glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
  glutCreateWindow(argv[0]);
  glutDisplayFunc(displayFunc);
  glutKeyboardFunc(keyboard);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  printf("OpenGL window created.\n");

#if defined(__APPLE__) || defined(MACOSX)
  atexit(cleanup);
#else
  glutCloseFunc(cleanup);
#endif

  if (!isGLVersionSupported(1, 5) ||
      !areGLExtensionsSupported(
          "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
    fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
    fprintf(stderr, "This sample requires:\n");
    fprintf(stderr, "  OpenGL version 1.5\n");
    fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
    fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
    fflush(stderr);
    return false;
  }

  return 0;
}

// shader for displaying floating-point texture
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

void initOpenGLBuffers() {
  printf("Creating GL texture...\n");
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_Tex);
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, h_Src);
  printf("Texture created.\n");

  printf("Creating PBO...\n");
  glGenBuffers(1, &gl_PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src,
               GL_STREAM_COPY);
  // While a PBO is registered to CUDA, it can't be used
  // as the destination for OpenGL drawing calls.
  // But in our particular case OpenGL is only used
  // to display the content of the PBO, specified by CUDA kernels,
  // so we need to register/unregister it only once.
  // DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(gl_PBO) );
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
      &cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
  GLenum gl_error = glGetError();

  if (gl_error != GL_NO_ERROR) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    char tmpStr[512];
    // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at
    // the right line when the user double clicks on the error line in the
    // Output pane. Like any compile error.
    sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", __FILE__, __LINE__,
              gluErrorString(gl_error));
    OutputDebugString(tmpStr);
#endif
    fprintf(stderr, "GL Error in file '%s' in line %d :\n", __FILE__, __LINE__);
    fprintf(stderr, "%s\n", gluErrorString(gl_error));
    exit(EXIT_FAILURE);
  }

  printf("PBO created.\n");

  // load shader program
  shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void cleanup() {
  free(h_Src);
  checkCudaErrors(CUDA_FreeArray());
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

  glDeleteProgramsARB(1, &shader);

  sdkDeleteTimer(&timer);
}

void runAutoTest(int argc, char **argv, const char *filename,
                 int kernel_param) {
  printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

  int devID = findCudaDevice(argc, (const char **)argv);

  // First load the image, so we know what the size of the image (imageW and
  // imageH)
  printf("Allocating host and CUDA memory and loading image file...\n");
  const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);

  if (image_path == NULL) {
    printf(
        "imageDenoisingGL was unable to find and load image file "
        "<portrait_noise.bmp>.\nExiting...\n");
    exit(EXIT_FAILURE);
  }

  LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
  printf("Data init done.\n");

  checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

  TColor *d_dst = NULL;
  unsigned char *h_dst = NULL;
  checkCudaErrors(
      cudaMalloc((void **)&d_dst, imageW * imageH * sizeof(TColor)));
  h_dst = (unsigned char *)malloc(imageH * imageW * 4);

  {
    g_Kernel = kernel_param;
    printf("[AutoTest]: %s <%s>\n", sSDKsample, filterMode[g_Kernel]);

    runImageFilters(d_dst);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW * imageH * sizeof(TColor),
                               cudaMemcpyDeviceToHost));
    sdkSavePPM4ub(filename, h_dst, imageW, imageH);
  }

  checkCudaErrors(CUDA_FreeArray());
  free(h_Src);

  checkCudaErrors(cudaFree(d_dst));
  free(h_dst);

  printf("\n[%s] -> Kernel %d, Saved: %s\n", sSDKsample, kernel_param,
         filename);

  exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char **argv) {
  char *dump_file = NULL;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", sSDKsample);

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    getCmdLineArgumentString(argc, (const char **)argv, "file",
                             (char **)&dump_file);

    int kernel = 1;

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
      kernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    runAutoTest(argc, argv, dump_file, kernel);
  } else {
    printf("[%s]\n", sSDKsample);

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
      printf("[%s]\n", argv[0]);
      printf("   Does not explicitly support -device=n in OpenGL mode\n");
      printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
      printf(" > %s -device=n -qatest\n", argv[0]);
      printf("exiting...\n");
      exit(EXIT_SUCCESS);
    }

    // First load the image, so we know what the size of the image (imageW and
    // imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
    const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);

    if (image_path == NULL) {
      printf(
          "imageDenoisingGL was unable to find and load image file "
          "<portrait_noise.bmp>.\nExiting...\n");
      exit(EXIT_FAILURE);
    }

    LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
    printf("Data init done.\n");

    initGL(&argc, argv);
    findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

    initOpenGLBuffers();
  }

  printf("Starting GLUT main loop...\n");
  printf("Press [1] to view noisy image\n");
  printf("Press [2] to view image restored with knn filter\n");
  printf("Press [3] to view image restored with nlm filter\n");
  printf("Press [4] to view image restored with modified nlm filter\n");
  printf(
      "Press [*] to view smooth/edgy areas [RED/BLUE] Ct's when a filter is "
      "active\n");
  printf("Press [f] to print frame rate\n");
  printf("Press [?] to print Noise and Lerp Ct's\n");
  printf("Press [q] to exit\n");

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  glutMainLoop();
}
