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
  Image bilateral filtering example

  This sample uses CUDA to perform a simple bilateral filter on an image
  and uses OpenGL to display the results.

  Bilateral filter is an edge-preserving nonlinear smoothing filter. There
  are three parameters distribute to the filter: gaussian delta, euclidean
  delta and iterations.

  When the euclidean delta increases, most of the fine texture will be
  filtered away, yet all contours are as crisp as in the original image.
  If the euclidean delta approximates to ∞, the filter becomes a normal
  gaussian filter. Fine texture will blur more with larger gaussian delta.
  Multiple iterations have the effect of flattening the colors in an
  image considerably, but without blurring edges, which produces a cartoon
  effect.

  To learn more details about this filter, please view C. Tomasi's "Bilateral
  Filtering for Gray and Color Images".

*/

#include <math.h>

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

#include <helper_cuda.h>  // CUDA device initialization helper functions

// Shared Library Test Functions
#include <helper_functions.h>  // CUDA SDK Helper functions

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY 10  // ms
#define MIN_EUCLIDEAN_D 0.01f
#define MAX_EUCLIDEAN_D 5.f
#define MAX_FILTER_RADIUS 25

const static char *sSDKsample = "CUDA Bilateral Filter";

const char *image_filename = "nature_monte.bmp";
int iterations = 1;
float gaussian_delta = 4;
float euclidean_delta = 0.1f;
int filter_radius = 5;

unsigned int width, height;
unsigned int *hImage = NULL;

GLuint pbo;                                      // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
GLuint texid;                                    // texture
GLuint shader;

int *pArgc = NULL;
char **pArgv = NULL;

StopWatchInterface *timer = NULL;
StopWatchInterface *kernel_timer = NULL;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
unsigned int g_TotalErrors = 0;
bool g_bInteractive = false;

//#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void loadImageData(int argc, char **argv);

// These are CUDA functions to handle allocation and launching the kernels
extern "C" void initTexture(int width, int height, void *pImage);
extern "C" void freeTextures();

extern "C" double bilateralFilterRGBA(unsigned int *d_dest, int width,
                                      int height, float e_d, int radius,
                                      int iterations,
                                      StopWatchInterface *timer);
extern "C" void updateGaussian(float delta, int radius);
extern "C" void updateGaussianGold(float delta, int radius);
extern "C" void bilateralFilterGold(unsigned int *pSrc, unsigned int *pDest,
                                    float e_d, int w, int h, int r);
extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width,
                            unsigned int *height, const char *name);

void varyEuclidean() {
  static float factor = 1.02f;

  if (euclidean_delta > MAX_EUCLIDEAN_D) {
    factor = 1 / 1.02f;
  }

  if (euclidean_delta < MIN_EUCLIDEAN_D) {
    factor = 1.02f;
  }

  euclidean_delta *= factor;
}

void computeFPS() {
  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
    sprintf(fps,
            "CUDA Bilateral Filter: %3.f fps (radius=%d, iter=%d, "
            "euclidean=%.2f, gaussian=%.2f)",
            ifps, filter_radius, iterations, (double)euclidean_delta,
            (double)gaussian_delta);

    glutSetWindowTitle(fps);
    fpsCount = 0;
    fpsLimit = (int)MAX(ifps, 1.0f);

    sdkResetTimer(&timer);
  }

  if (!g_bInteractive) {
    varyEuclidean();
  }
}

// display results using OpenGL
void display() {
  sdkStartTimer(&timer);

  // execute filter, writing results to pbo
  unsigned int *dResult;

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&dResult, &num_bytes, cuda_pbo_resource));
  bilateralFilterRGBA(dResult, width, height, euclidean_delta, filter_radius,
                      iterations, kernel_timer);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

  // Common display code path
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
      glTexCoord2f(0, 0);
      glVertex2f(0, 0);
      glTexCoord2f(1, 0);
      glVertex2f(1, 0);
      glTexCoord2f(1, 1);
      glVertex2f(1, 1);
      glTexCoord2f(0, 1);
      glVertex2f(0, 1);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_TYPE, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
  }

  glutSwapBuffers();
  glutReportErrors();

  sdkStopTimer(&timer);

  computeFPS();
}

/*
    right arrow to increase the gaussian delta
    left arrow to decrease the gaussian delta
    up arrow to increase the euclidean delta
    down arrow to decrease the euclidean delta
*/
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

    case ']':
      iterations++;
      break;

    case '[':
      iterations--;

      if (iterations < 1) {
        iterations = 1;
      }

      break;

    case '=':
    case '+':
      filter_radius++;

      if (filter_radius > MAX_FILTER_RADIUS) {
        filter_radius = MAX_FILTER_RADIUS;
      }

      updateGaussian(gaussian_delta, filter_radius);
      break;

    case '-':
      filter_radius--;

      if (filter_radius < 1) {
        filter_radius = 1;
      }

      updateGaussian(gaussian_delta, filter_radius);
      break;

    case 'E':
      euclidean_delta *= 1.5;
      break;

    case 'e':
      euclidean_delta /= 1.5;
      break;

    case 'g':
      if (gaussian_delta > 0.1) {
        gaussian_delta /= 2;
      }

      // updateGaussianGold(gaussian_delta, filter_radius);
      updateGaussian(gaussian_delta, filter_radius);
      break;

    case 'G':
      gaussian_delta *= 2;
      // updateGaussianGold(gaussian_delta, filter_radius);
      updateGaussian(gaussian_delta, filter_radius);
      break;

    default:
      break;
  }

  printf(
      "filter radius = %d, iterations = %d, gaussian delta = %.2f, euclidean "
      "delta = %.2f\n",
      filter_radius, iterations, gaussian_delta, euclidean_delta);
  glutPostRedisplay();
}

void timerEvent(int value) {
  if (glutGetWindow()) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  }
}

void reshape(int x, int y) {
  glViewport(0, 0, x, y);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCuda() {
  // initialize gaussian mask
  updateGaussian(gaussian_delta, filter_radius);

  initTexture(width, height, hImage);
  sdkCreateTimer(&timer);
  sdkCreateTimer(&kernel_timer);
}

void cleanup() {
  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&kernel_timer);

  if (hImage) {
    free(hImage);
  }

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

void initGLResources() {
  // create pixel buffer object
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4,
               hImage, GL_STREAM_DRAW_ARB);

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

////////////////////////////////////////////////////////////////////////////////
//! Run a simple benchmark test for CUDA
////////////////////////////////////////////////////////////////////////////////
int runBenchmark(int argc, char **argv) {
  printf("[runBenchmark]: [%s]\n", sSDKsample);

  loadImageData(argc, argv);
  initCuda();

  unsigned int *dResult;
  size_t pitch;
  checkCudaErrors(cudaMallocPitch((void **)&dResult, &pitch,
                                  width * sizeof(unsigned int), height));
  sdkStartTimer(&kernel_timer);

  // warm-up
  bilateralFilterRGBA(dResult, width, height, euclidean_delta, filter_radius,
                      iterations, kernel_timer);
  checkCudaErrors(cudaDeviceSynchronize());

  // Start round-trip timer and process iCycles loops on the GPU
  iterations = 1;  // standard 1-pass filtering
  const int iCycles = 150;
  double dProcessingTime = 0.0;
  printf("\nRunning BilateralFilterGPU for %d cycles...\n\n", iCycles);

  for (int i = 0; i < iCycles; i++) {
    dProcessingTime +=
        bilateralFilterRGBA(dResult, width, height, euclidean_delta,
                            filter_radius, iterations, kernel_timer);
  }

  // check if kernel execution generated an error and sync host
  getLastCudaError("Error: bilateralFilterRGBA Kernel execution FAILED");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&kernel_timer);

  // Get average computation time
  dProcessingTime /= (double)iCycles;

  // log testname, throughput, timing and config info to sample and master logs
  printf(
      "bilateralFilter-texture, Throughput = %.4f M RGBA Pixels/s, Time = %.5f "
      "s, Size = %u RGBA Pixels, NumDevsUsed = %u\n",
      (1.0e-6 * width * height) / dProcessingTime, dProcessingTime,
      (width * height), 1);
  printf("\n");

  return 0;
}

void initGL(int argc, char **argv) {
  // initialize GLUT
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(width, height);

  glutCreateWindow("CUDA Bilateral Filter");
  glutDisplayFunc(display);

  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);
  // glutIdleFunc(idle);
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

// This test specifies a single test (where you specify radius and/or
// iterations)
int runSingleTest(char *ref_file, char *exec_path) {
  int nTotalErrors = 0;
  char dump_file[256];

  printf("[runSingleTest]: [%s]\n", sSDKsample);

  initCuda();

  unsigned int *dResult;
  unsigned int *hResult =
      (unsigned int *)malloc(width * height * sizeof(unsigned int));
  size_t pitch;
  checkCudaErrors(cudaMallocPitch((void **)&dResult, &pitch,
                                  width * sizeof(unsigned int), height));

  // run the sample radius
  {
    printf("%s (radius=%d) (passes=%d) ", sSDKsample, filter_radius,
           iterations);
    bilateralFilterRGBA(dResult, width, height, euclidean_delta, filter_radius,
                        iterations, kernel_timer);

    // check if kernel execution generated an error
    getLastCudaError("Error: bilateralFilterRGBA Kernel execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());

    // readback the results to system memory
    cudaMemcpy2D(hResult, sizeof(unsigned int) * width, dResult, pitch,
                 sizeof(unsigned int) * width, height, cudaMemcpyDeviceToHost);

    sprintf(dump_file, "nature_%02d.ppm", filter_radius);

    sdkSavePPM4ub((const char *)dump_file, (unsigned char *)hResult, width,
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

  free(hResult);
  checkCudaErrors(cudaFree(dResult));
  freeTextures();

  return nTotalErrors;
}

void loadImageData(int argc, char **argv) {
  // load image (needed so we can get the width and height before we create the
  // window
  char *image_path = NULL;

  if (argc >= 1) {
    image_path = sdkFindFilePath(image_filename, argv[0]);
  }

  if (image_path == NULL) {
    fprintf(stderr, "Error finding image file '%s'\n", image_filename);
    exit(EXIT_FAILURE);
  }

  LoadBMPFile((uchar4 **)&hImage, &width, &height, image_path);

  if (!hImage) {
    fprintf(stderr, "Error opening file '%s'\n", image_path);
    exit(EXIT_FAILURE);
  }

  printf("Loaded '%s', %d x %d pixels\n\n", image_path, width, height);
}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute) {
  int runtimeVersion = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  cudaRuntimeGetVersion(&runtimeVersion);
  fprintf(stderr, "  CUDA Runtime Version     :\t%d.%d\n",
          runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  fprintf(stderr, "  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major,
          deviceProp.minor);

  if (runtimeVersion >= min_runtime &&
      ((deviceProp.major << 4) + deviceProp.minor) >= min_compute) {
    return true;
  } else {
    return false;
  }
}

void printHelp() {
  printf("bilateralFilter usage\n");
  printf("    -radius=n  (specify the filter radius n to use)\n");
  printf("    -passes=n  (specify the number of passes n to use)\n");
  printf("    -file=name (specify reference file for comparison)\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // start logs
  int devID;
  char *ref_file = NULL;
  printf("%s Starting...\n\n", argv[0]);

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  if (argc > 1) {
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
    // Running CUDA kernels (bilateralfilter) in Benchmarking mode
    g_TotalErrors += runBenchmark(argc, argv);

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

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with
    // OpenGL/CUDA interop.
    initGL(argc, (char **)argv);

    initCuda();
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
        "Press 'e' and 'E' to change Euclidean delta\n"
        "Press 'g' and 'G' to change Gaussian delta\n"
        "Press 'a' or  'A' to change Animation mode ON/OFF\n\n");

    // Main OpenGL loop that will run visualization for every vsync
    glutMainLoop();
  }
}
