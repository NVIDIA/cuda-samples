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
  Image box filtering example

  This sample uses CUDA to perform a simple box filter on an image
  and uses OpenGL to display the results.

  It processes rows and columns of the image in parallel.

  The box filter is implemented such that it has a constant cost,
  regardless of the filter width.

  Press '=' to increment the filter radius, '-' to decrease it

  Version 1.1 - modified to process 8-bit RGBA images
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY 10  // ms

const static char *sSDKsample = "CUDA Iterative Box Filter";

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] = {"teapot1024_14.ppm", "teapot1024_22.ppm", NULL};

const char *sReference[] = {"ref_14.ppm", "ref_22.ppm", NULL};

const char *image_filename = "teapot1024.ppm";
int iterations = 1;
int filter_radius = 14;
int nthreads = 64;
unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
unsigned int *d_temp = NULL;

GLuint pbo;                                      // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
GLuint texid;                                    // Texture
GLuint shader;

StopWatchInterface *timer = NULL, *kernel_timer = NULL;

// Auto-Verification Code
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 8;  // FPS limit for sampling
int g_Index = 0;
int g_nFilterSign = 1;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bInteractive = false;

int *pArgc = NULL;
char **pArgv = NULL;

extern "C" int runSingleTest(char *ref_file, char *exec_path);
extern "C" int runBenchmark();
extern "C" void loadImageData(int argc, char **argv);
extern "C" void computeGold(float *id, float *od, int w, int h, int n);

// These are CUDA functions to handle allocation and launching the kernels
extern "C" void initTexture(int width, int height, void *pImage, bool useRGBA);
extern "C" void freeTextures();
extern "C" double boxFilter(float *d_src, float *d_temp, float *d_dest,
                            int width, int height, int radius, int iterations,
                            int nthreads, StopWatchInterface *timer);

extern "C" double boxFilterRGBA(unsigned int *d_src, unsigned int *d_temp,
                                unsigned int *d_dest, int width, int height,
                                int radius, int iterations, int nthreads,
                                StopWatchInterface *timer);

// This varies the filter radius, so we can see automatic animation
void varySigma() {
  filter_radius += g_nFilterSign;

  if (filter_radius > 64) {
    filter_radius = 64;  // clamp to 64 and then negate sign
    g_nFilterSign = -1;
  } else if (filter_radius < 0) {
    filter_radius = 0;
    g_nFilterSign = 1;
  }
}

// Calculate the Frames per second and print in the title bar
void computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    avgFPS = 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
    fpsCount = 0;
    fpsLimit = (int)MAX(avgFPS, 1.0f);
    sdkResetTimer(&timer);
  }

  char fps[256];
  sprintf(fps,
          "CUDA Rolling Box Filter <Animation=%s> (radius=%d, passes=%d): "
          "%3.1f fps",
          (!g_bInteractive ? "ON" : "OFF"), filter_radius, iterations, avgFPS);
  glutSetWindowTitle(fps);

  if (!g_bInteractive) {
    varySigma();
  }
}

// display results using OpenGL
void display() {
  sdkStartTimer(&timer);

  // execute filter, writing results to pbo
  unsigned int *d_result;

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_result, &num_bytes, cuda_pbo_resource));
  boxFilterRGBA(d_img, d_temp, d_result, width, height, filter_radius,
                iterations, nthreads, kernel_timer);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

  // OpenGL display code path
  {
    glClear(GL_COLOR_BUFFER_BIT);

    // load texture from pbo
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(0.0f, 0.0f);
      glTexCoord2f(1.0f, 0.0f);
      glVertex2f(1.0f, 0.0f);
      glTexCoord2f(1.0f, 1.0f);
      glVertex2f(1.0f, 1.0f);
      glTexCoord2f(0.0f, 1.0f);
      glVertex2f(0.0f, 1.0f);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
  }

  glutSwapBuffers();
  glutReportErrors();

  sdkStopTimer(&timer);

  computeFPS();
}

// Keyboard callback function for OpenGL (GLUT)
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
    case 27:
#if defined(__APPLE__) || defined(MACOSX)
      exit(EXIT_SUCCESS);
#else
      glutDestroyWindow(glutGetWindow());
      return;
#endif
      break;

    case 'a':
    case 'A':
      g_bInteractive = !g_bInteractive;
      printf("> Animation is %s\n", !g_bInteractive ? "ON" : "OFF");
      break;

    case '=':
    case '+':
      if (filter_radius < (int)width - 1 && filter_radius < (int)height - 1) {
        filter_radius++;
      }

      break;

    case '-':
      if (filter_radius > 1) {
        filter_radius--;
      }

      break;

    case ']':
      iterations++;
      break;

    case '[':
      if (iterations > 1) {
        iterations--;
      }

      break;

    default:
      break;
  }

  printf("radius = %d, iterations = %d\n", filter_radius, iterations);
}

// Timer Event so we can refresh the display
void timerEvent(int value) {
  if (glutGetWindow()) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  }
}

// Resizing the window
void reshape(int x, int y) {
  glViewport(0, 0, x, y);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCuda(bool useRGBA) {
  // allocate device memory
  checkCudaErrors(
      cudaMalloc((void **)&d_img, (width * height * sizeof(unsigned int))));
  checkCudaErrors(
      cudaMalloc((void **)&d_temp, (width * height * sizeof(unsigned int))));

  // Refer to boxFilter_kernel.cu for implementation
  initTexture(width, height, h_img, useRGBA);

  sdkCreateTimer(&timer);
  sdkCreateTimer(&kernel_timer);
}

void cleanup() {
  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&kernel_timer);

  if (h_img) {
    free(h_img);
    h_img = NULL;
  }

  if (d_img) {
    cudaFree(d_img);
    d_img = NULL;
  }

  if (d_temp) {
    cudaFree(d_temp);
    d_temp = NULL;
  }

  // Refer to boxFilter_kernel.cu for implementation
  freeTextures();

  cudaGraphicsUnregisterResource(cuda_pbo_resource);

  glDeleteBuffers(1, &pbo);
  glDeleteTextures(1, &texid);
  glDeleteProgramsARB(1, &shader);
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
    printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
    return 0;
  }

  return program_id;
}

// This is where we create the OpenGL PBOs, FBOs, and texture resources
void initGLResources() {
  // create pixel buffer object
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4,
               h_img, GL_STREAM_DRAW_ARB);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
      &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

  // create texture for display
  glGenTextures(1, &texid);
  glBindTexture(GL_TEXTURE_2D, texid);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);

  // load shader program
  shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void initGL(int *argc, char **argv) {
  // initialize GLUT
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(768, 768);
  glutCreateWindow("CUDA Rolling Box Filter");
  glutDisplayFunc(display);

  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

  if (!isGLVersionSupported(2, 0) ||
      !areGLExtensionsSupported(
          "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
    printf("Error: failed to get minimal extensions for demo\n");
    printf("This sample requires:\n");
    printf("  OpenGL version 2.0\n");
    printf("  GL_ARB_vertex_buffer_object\n");
    printf("  GL_ARB_pixel_buffer_object\n");
    exit(EXIT_FAILURE);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple benchmark test for CUDA
////////////////////////////////////////////////////////////////////////////////
int runBenchmark() {
  printf("[runBenchmark]: [%s]\n", sSDKsample);

  initCuda(true);

  unsigned int *d_result;
  checkCudaErrors(
      cudaMalloc((void **)&d_result, width * height * sizeof(unsigned int)));

  // warm-up
  boxFilterRGBA(d_img, d_temp, d_temp, width, height, filter_radius, iterations,
                nthreads, kernel_timer);
  checkCudaErrors(cudaDeviceSynchronize());

  sdkStartTimer(&kernel_timer);
  // Start round-trip timer and process iCycles loops on the GPU
  iterations = 1;  // standard 1-pass filtering
  const int iCycles = 150;
  double dProcessingTime = 0.0;
  printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);

  for (int i = 0; i < iCycles; i++) {
    dProcessingTime +=
        boxFilterRGBA(d_img, d_temp, d_img, width, height, filter_radius,
                      iterations, nthreads, kernel_timer);
  }

  // check if kernel execution generated an error and sync host
  getLastCudaError("Error: boxFilterRGBA Kernel execution FAILED");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&kernel_timer);

  // Get average computation time
  dProcessingTime /= (double)iCycles;

  // log testname, throughput, timing and config info to sample and master logs
  printf(
      "boxFilter-texture, Throughput = %.4f M RGBA Pixels/s, Time = %.5f s, "
      "Size = %u RGBA Pixels, NumDevsUsed = %u, Workgroup = %u\n",
      (1.0e-6 * width * height) / dProcessingTime, dProcessingTime,
      (width * height), 1, nthreads);
  printf("\n");

  return 0;
}

// This test specifies a single test (where you specify radius and/or
// iterations)
int runSingleTest(char *ref_file, char *exec_path) {
  int nTotalErrors = 0;
  char dump_file[256];

  printf("[runSingleTest]: [%s]\n", sSDKsample);

  initCuda(true);

  unsigned int *d_result;
  unsigned int *h_result =
      (unsigned int *)malloc(width * height * sizeof(unsigned int));
  checkCudaErrors(
      cudaMalloc((void **)&d_result, width * height * sizeof(unsigned int)));

  // run the sample radius
  {
    printf("%s (radius=%d) (passes=%d) ", sSDKsample, filter_radius,
           iterations);
    boxFilterRGBA(d_img, d_temp, d_result, width, height, filter_radius,
                  iterations, nthreads, kernel_timer);

    // check if kernel execution generated an error
    getLastCudaError("Error: boxFilterRGBA Kernel execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());

    // readback the results to system memory
    cudaMemcpy((unsigned char *)h_result, (unsigned char *)d_result,
               width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    sprintf(dump_file, "teapot1024_%02d.ppm", filter_radius);

    sdkSavePPM4ub((const char *)dump_file, (unsigned char *)h_result, width,
                  height);

    if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, exec_path),
                       MAX_EPSILON_ERROR, 0.15f, false)) {
      printf("Image is Different ");
      nTotalErrors++;
    } else {
      printf("Image is Matching ");
    }

    printf(" <%s>\n", ref_file);
  }
  printf("\n");

  free(h_result);
  checkCudaErrors(cudaFree(d_result));

  return nTotalErrors;
}

void loadImageData(int argc, char **argv) {
  // load image (needed so we can get the width and height before we create the
  // window
  char *image_path = NULL;

  if (argc >= 1) {
    image_path = sdkFindFilePath(image_filename, argv[0]);
  }

  if (image_path == 0) {
    printf("Error finding image file '%s'\n", image_filename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPPM4(image_path, (unsigned char **)&h_img, &width, &height);

  if (!h_img) {
    printf("Error opening file '%s'\n", image_path);
    exit(EXIT_FAILURE);
  }

  printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
}

void printHelp() {
  printf("boxFilter usage\n");
  printf("    -threads=n (specify the # of of threads to use)\n");
  printf("    -radius=n  (specify the filter radius n to use)\n");
  printf("    -passes=n  (specify the number of passes n to use)\n");
  printf("    -file=name (specify reference file for comparison)\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  int devID = 0;
  char *ref_file = NULL;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  pArgc = &argc;
  pArgv = argv;

  // start logs
  printf("%s Starting...\n\n", argv[0]);

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp();
    exit(EXIT_SUCCESS);
  }

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
      nthreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "radius")) {
      filter_radius =
          getCmdLineArgumentInt(argc, (const char **)argv, "radius");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "passes")) {
      iterations = getCmdLineArgumentInt(argc, (const char **)argv, "passes");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
      getCmdLineArgumentString(argc, (const char **)argv, "file",
                               (char **)&ref_file);
    }
  }

  // load image to process
  loadImageData(argc, argv);
  devID = findCudaDevice(argc, (const char **)argv);

  if (checkCmdLineFlag(argc, (const char **)argv, "benchmark")) {
    // This is a separate mode of the sample, where we are benchmark the kernels
    // for performance
    // Running CUDA kernels (boxfilter) in Benchmarking mode
    g_TotalErrors += runBenchmark();
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
  } else if (checkCmdLineFlag(argc, (const char **)argv, "radius") ||
             checkCmdLineFlag(argc, (const char **)argv, "passes")) {
    // This overrides the default mode.  Users can specify the radius used by
    // the filter kernel
    g_TotalErrors += runSingleTest(ref_file, argv[0]);
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
  } else {
    // Default mode running with OpenGL visualization and in automatic mode
    // the output automatically changes animation
    printf("\n");

    initGL(&argc, argv);

    initCuda(true);
    initGLResources();

// sets the callback function so it will call cleanup upon exit
#if defined(__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    printf("Running Standard Demonstration with GLUT loop...\n\n");
    printf(
        "Press '+' and '-' to change filter width\n"
        "Press ']' and '[' to change number of iterations\n"
        "Press 'a' or  'A' to change animation ON/OFF\n\n");

    // Main OpenGL loop that will run visualization for every vsync
    glutMainLoop();
  }
}
