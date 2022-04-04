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
    Bicubic texture filtering sample
    sgreen 6/2008

    This sample demonstrates how to efficiently implement bicubic texture
    filtering in CUDA.

    Bicubic filtering is a higher order interpolation method that produces
    smoother results than bilinear interpolation:
    http://en.wikipedia.org/wiki/Bicubic

    It requires reading a 4 x 4 pixel neighbourhood rather than the
    2 x 2 area required by bilinear filtering.

    Current graphics hardware doesn't support bicubic filtering natively,
    but it is possible to compose a bicubic filter using just 4 bilinear
    lookups by offsetting the sample position within each texel and weighting
    the samples correctly. The only disadvantage to this method is that the
    hardware only maintains 9-bits of filtering precision within each texel.

    See "Fast Third-Order Texture Filtering", Sigg & Hadwiger, GPU Gems 2:
    https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering

    v1.1 - updated to include the brute force method using 16 texture lookups.
    v1.2 - added Catmull-Rom interpolation

    Example performance results from GeForce 8800 GTS:
    Bilinear     - 5500 MPixels/sec
    Bicubic      - 1400 MPixels/sec
    Fast Bicubic - 2100 MPixels/sec
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA system and GL includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions

typedef unsigned int uint;
typedef unsigned char uchar;

#define USE_BUFFER_TEX 0
#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 4;  // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
StopWatchInterface *timer = 0;
bool g_Verify = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY 10  // ms

static const char *sSDKsample = "CUDA BicubicTexture";

// Define the files that are to be save and the reference images for validation
const char *sFilterMode[] = {"Nearest",      "Bilinear",    "Bicubic",
                             "Fast Bicubic", "Catmull-Rom", NULL};

const char *sOriginal[] = {"0_nearest.ppm",     "1_bilinear.ppm",
                           "2_bicubic.ppm",     "3_fastbicubic.ppm",
                           "4_catmull-rom.ppm", NULL};

const char *sReference[] = {"0_nearest.ppm",     "1_bilinear.ppm",
                            "2_bicubic.ppm",     "3_fastbicubic.ppm",
                            "4_catmull-rom.ppm", NULL};

const char *srcImageFilename = "teapot512.pgm";
char *dumpFilename = NULL;

uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

enum eFilterMode {
  MODE_NEAREST,
  MODE_BILINEAR,
  MODE_BICUBIC,
  MODE_FAST_BICUBIC,
  MODE_CATMULL_ROM,
  NUM_MODES
};

eFilterMode g_FilterMode = MODE_FAST_BICUBIC;

bool drawCurves = false;

GLuint pbo = 0;                                  // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
GLuint displayTex = 0;
GLuint bufferTex = 0;
GLuint fprog;  // fragment program (shader)

float tx = -27.75f, ty = -189.0f;  // image translation
float scale = 0.125f;   // image scale
float cx, cy;                 // image centre

void display();
void initGLBuffers();
void runBenchmark(int iterations);
void cleanup();

#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
//#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void initGL(int *argc, char **argv);
extern "C" void loadImageData(int argc, char **argv);

extern "C" void initTexture(int imageWidth, int imageHeight, uchar *h_data);
extern "C" void freeTexture();
extern "C" void render(int width, int height, float tx, float ty, float scale,
                       float cx, float cy, dim3 blockSize, dim3 gridSize,
                       eFilterMode filter_mode, uchar4 *output);

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float bspline_w0(float a) {
  return (1.0f / 6.0f) * (-a * a * a + 3.0f * a * a - 3.0f * a + 1.0f);
}

float bspline_w1(float a) {
  return (1.0f / 6.0f) * (3.0f * a * a * a - 6.0f * a * a + 4.0f);
}

float bspline_w2(float a) {
  return (1.0f / 6.0f) * (-3.0f * a * a * a + 3.0f * a * a + 3.0f * a + 1.0f);
}

__host__ __device__ float bspline_w3(float a) {
  return (1.0f / 6.0f) * (a * a * a);
}

void computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit - 1) {
    g_Verify = true;
  }

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "%s %s <%s>: %3.1f fps", "", sSDKsample,
            sFilterMode[g_FilterMode], ifps);

    glutSetWindowTitle(fps);
    fpsCount = 0;

    sdkResetTimer(&timer);
  }
}

void plotCurve(float (*func)(float)) {
  const int steps = 100;
  glBegin(GL_LINE_STRIP);

  for (int i = 0; i < steps; i++) {
    float x = i / (float)(steps - 1);
    glVertex2f(x, func(x));
  }

  glEnd();
}

// display results using OpenGL (called by GLUT)
void display() {
  sdkStartTimer(&timer);

  // map PBO to get CUDA device pointer
  uchar4 *d_output;
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_output, &num_bytes, cuda_pbo_resource));
  render(imageWidth, imageHeight, tx, ty, scale, cx, cy, blockSize, gridSize,
         g_FilterMode, d_output);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

  // Common display path
  {
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

#if USE_BUFFER_TEX
    // display using buffer texture
    glBindTexture(GL_TEXTURE_BUFFER_EXT, bufferTex);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fprog);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glProgramLocalParameterI4iNV(GL_FRAGMENT_PROGRAM_ARB, 0, width, 0, 0, 0);
#else
    // download image from PBO to OpenGL texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_TYPE, displayTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_TYPE, 0, 0, 0, width, height, GL_BGRA,
                    GL_UNSIGNED_BYTE, 0);
    glEnable(GL_TEXTURE_TYPE);
#endif

    // draw textured quad
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, (GLfloat)height);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f((GLfloat)width, (GLfloat)height);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f((GLfloat)width, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();
    glDisable(GL_TEXTURE_TYPE);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    if (drawCurves) {
      // draw spline curves
      glPushMatrix();
      glScalef(0.25, 0.25, 1.0);

      glTranslatef(0.0, 2.0, 0.0);
      glColor3f(1.0, 0.0, 0.0);
      plotCurve(bspline_w3);

      glTranslatef(1.0, 0.0, 0.0);
      glColor3f(0.0, 1.0, 0.0);
      plotCurve(bspline_w2);

      glTranslatef(1.0, 0.0, 0.0);
      glColor3f(0.0, 0.0, 1.0);
      plotCurve(bspline_w1);

      glTranslatef(1.0, 0.0, 0.0);
      glColor3f(1.0, 0.0, 1.0);
      plotCurve(bspline_w0);

      glPopMatrix();
      glColor3f(1.0, 1.0, 1.0);
    }
  }

  glutSwapBuffers();
  glutReportErrors();

  sdkStopTimer(&timer);

  computeFPS();
}

// GLUT callback functions
void timerEvent(int value) {
  if (glutGetWindow()) {
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  }
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
    case 27:
#if defined(__APPLE__) || defined(MACOSX)
      exit(EXIT_SUCCESS);
#else
      glutDestroyWindow(glutGetWindow());
      return;
#endif

    case '1':
      g_FilterMode = MODE_NEAREST;
      break;

    case '2':
      g_FilterMode = MODE_BILINEAR;
      break;

    case '3':
      g_FilterMode = MODE_BICUBIC;
      break;

    case '4':
      g_FilterMode = MODE_FAST_BICUBIC;
      break;

    case '5':
      g_FilterMode = MODE_CATMULL_ROM;
      break;

    case '=':
    case '+':
      scale *= 0.5f;
      break;

    case '-':
      scale *= 2.0f;
      break;

    case 'r':
      scale = 1.0f;
      tx = ty = 0.0f;
      break;

    case 'd':
      printf("%f, %f, %f\n", tx, ty, scale);
      break;

    case 'b':
      runBenchmark(500);
      break;

    case 'c':
      drawCurves ^= 1;
      break;

    default:
      break;
  }

  if (key >= '1' && key <= '5') {
    printf("> FilterMode[%d] = %s\n", g_FilterMode + 1,
           sFilterMode[g_FilterMode]);
  }
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    buttonState |= 1 << button;
  } else if (state == GLUT_UP) {
    buttonState = 0;
  }

  ox = x;
  oy = y;
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);

  if (buttonState & 1) {
    // left = translate
    tx -= dx * scale;
    ty -= dy * scale;
  } else if (buttonState & 2) {
    // middle = zoom
    scale -= dy / 1000.0f;
  }

  ox = x;
  oy = y;
}

void reshape(int x, int y) {
  width = x;
  height = y;
  imageWidth = width;
  imageHeight = height;

  initGLBuffers();

  glViewport(0, 0, x, y);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup() {
  freeTexture();
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

  glDeleteBuffers(1, &pbo);

#if USE_BUFFER_TEX
  glDeleteTextures(1, &bufferTex);
  glDeleteProgramsARB(1, &fprog);
#else
  glDeleteTextures(1, &displayTex);
#endif

  sdkDeleteTimer(&timer);
}

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void initGLBuffers() {
  if (pbo) {
    // delete old buffer
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1, &pbo);
  }

  // create pixel buffer object for display
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), 0,
               GL_STREAM_DRAW_ARB);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  checkCudaErrors(cudaGraphicsGLRegisterBuffer(
      &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

#if USE_BUFFER_TEX

  // create buffer texture, attach to pbo
  if (bufferTex) {
    glDeleteTextures(1, &bufferTex);
  }

  glGenTextures(1, &bufferTex);
  glBindTexture(GL_TEXTURE_BUFFER_EXT, bufferTex);
  glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA8, pbo);
  glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
#else

  // create texture for display
  if (displayTex) {
    glDeleteTextures(1, &displayTex);
  }

  glGenTextures(1, &displayTex);
  glBindTexture(GL_TEXTURE_TYPE, displayTex);
  glTexImage2D(GL_TEXTURE_TYPE, 0, GL_RGBA8, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_TYPE, 0);
#endif

  // calculate new grid size
  gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

void mainMenu(int i) { keyboard(i, 0, 0); }

void initMenus() {
  glutCreateMenu(mainMenu);
  glutAddMenuEntry("Nearest      [1]", '1');
  glutAddMenuEntry("Bilinear     [2]", '2');
  glutAddMenuEntry("Bicubic      [3]", '3');
  glutAddMenuEntry("Fast Bicubic [4]", '4');
  glutAddMenuEntry("Catmull-Rom  [5]", '5');
  glutAddMenuEntry("Zoom in      [=]", '=');
  glutAddMenuEntry("Zoom out     [-]", '-');
  glutAddMenuEntry("Benchmark    [b]", 'b');
  glutAddMenuEntry("DrawCurves   [c]", 'c');
  glutAddMenuEntry("Quit       [esc]", 27);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void runBenchmark(int iterations) {
  printf("[%s] (Benchmark Mode)\n", sSDKsample);

  sdkCreateTimer(&timer);

  uchar4 *d_output;
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_output, &num_bytes, cuda_pbo_resource));

  sdkStartTimer(&timer);

  for (int i = 0; i < iterations; ++i) {
    render(imageWidth, imageHeight, tx, ty, scale, cx, cy, blockSize, gridSize,
           g_FilterMode, d_output);
  }

  cudaDeviceSynchronize();
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer) / (float)iterations;

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

  printf("time: %0.3f ms, %f Mpixels/sec\n", time,
         (width * height / (time * 0.001f)) / 1e6);
}

void runAutoTest(int argc, char **argv, const char *dump_filename,
                 eFilterMode filter_mode) {
  cudaDeviceProp deviceProps;

  int devID = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));

  printf("[%s] (automated testing w/ readback)\n", sSDKsample);
  printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name,
         deviceProps.multiProcessorCount);

  loadImageData(argc, argv);

  uchar4 *d_output;
  checkCudaErrors(cudaMalloc((void **)&d_output, imageWidth * imageHeight * 4));
  unsigned int *h_result =
      (unsigned int *)malloc(width * height * sizeof(unsigned int));

  printf("AutoTest: %s Filter Mode: <%s>\n", sSDKsample,
         sFilterMode[g_FilterMode]);

  render(imageWidth, imageHeight, tx, ty, scale, cx, cy, blockSize, gridSize,
         filter_mode, d_output);

  // check if kernel execution generated an error
  getLastCudaError("Error: render (bicubicTexture) Kernel execution FAILED");
  checkCudaErrors(cudaDeviceSynchronize());

  cudaMemcpy(h_result, d_output, imageWidth * imageHeight * 4,
             cudaMemcpyDeviceToHost);

  sdkSavePPM4ub(dump_filename, (unsigned char *)h_result, imageWidth,
                imageHeight);

  checkCudaErrors(cudaFree(d_output));
  free(h_result);
}

#if USE_BUFFER_TEX
// fragment program for reading from buffer texture
static const char *shaderCode =
    "!!NVfp4.0\n"
    "INT PARAM width = program.local[0];\n"
    "INT TEMP index;\n"
    "FLR.S index, fragment.texcoord;\n"
    "MAD.S index.x, index.y, width, index.x;\n"  // compute 1D index from 2D
                                                 // coords
    "TXF result.color, index.x, texture[0], BUFFER;\n"
    "END";
#endif

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

void initialize(int argc, char **argv) {
  printf("[%s] (OpenGL Mode)\n", sSDKsample);

  initGL(&argc, argv);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  int devID = findCudaDevice(argc, (const char **)argv);

  // get number of SMs on this GPU
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name,
         deviceProps.multiProcessorCount);

  // Create the timer (for fps measurement)
  sdkCreateTimer(&timer);

  // load image from disk
  loadImageData(argc, argv);

  printf(
      "\n"
      "\tControls\n"
      "\t=/- : Zoom in/out\n"
      "\tb   : Run Benchmark g_FilterMode\n"
      "\tc   : Draw Bicubic Spline Curve\n"
      "\t[esc] - Quit\n\n"

      "\tPress number keys to change filtering g_FilterMode:\n\n"
      "\t1 : nearest filtering\n"
      "\t2 : bilinear filtering\n"
      "\t3 : bicubic filtering\n"
      "\t4 : fast bicubic filtering\n"
      "\t5 : Catmull-Rom filtering\n\n");

  initGLBuffers();

#if USE_BUFFER_TEX
  fprog = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shaderCode);

  if (!fprog) {
    exit(EXIT_SUCCESS);
  }

#endif
}

void initGL(int *argc, char **argv) {
  // initialize GLUT callback functions
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(width, height);
  glutCreateWindow("CUDA bicubic texture filtering");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

#if defined(__APPLE__) || defined(MACOSX)
  atexit(cleanup);
#else
  glutCloseFunc(cleanup);
#endif

  initMenus();

  if (!isGLVersionSupported(2, 0) ||
      !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
    fprintf(stderr, "Required OpenGL extensions are missing.");
    exit(EXIT_FAILURE);
  }

#if USE_BUFFER_TEX

  if (!areGLExtensionsSupported("GL_EXT_texture_buffer_object")) {
    fprintf(stderr,
            "OpenGL extension: GL_EXT_texture_buffer_object missing.\n");
    exit(EXIT_FAILURE);
  }

  if (!areGLExtensionsSupported("GL_NV_gpu_program4")) {
    fprintf(stderr, "OpenGL extension: GL_NV_gpu_program4 missing.\n");
    exit(EXIT_FAILURE);
  }

#endif
}

void loadImageData(int argc, char **argv) {
  // load image from disk
  uchar *h_data = NULL;
  char *srcImagePath = NULL;

  if ((srcImagePath = sdkFindFilePath(srcImageFilename, argv[0])) == NULL) {
    printf("bicubicTexture loadImageData() could not find <%s>\nExiting...\n",
           srcImageFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM<unsigned char>(srcImagePath, &h_data, &imageWidth, &imageHeight);

  printf("Loaded '%s', %d x %d pixels\n", srcImageFilename, imageWidth,
         imageHeight);

  cx = imageWidth * 0.5f;
  cy = imageHeight * 0.5f;

  // initialize texture
  initTexture(imageWidth, imageHeight, h_data);
}

void printHelp() {
  printf("bicubicTexture Usage:\n");
  printf("\t-file=output.ppm (output file to save to disk)\n");
  printf(
      "\t-mode=n (0=Nearest, 1=Bilinear, 2=Bicubic, 3=Fast-Bicubic, "
      "4=Catmull-Rom\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  // parse arguments
  char *filename;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  printf("Starting bicubicTexture\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp();
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "mode")) {
    g_FilterMode =
        (eFilterMode)getCmdLineArgumentInt(argc, (const char **)argv, "mode");

    if (g_FilterMode < 0 || g_FilterMode >= NUM_MODES) {
      printf("Invalid Mode setting %d\n", g_FilterMode);
      exit(EXIT_FAILURE);
    }
  }

  if (getCmdLineArgumentString(argc, (const char **)argv, "file", &filename)) {
    dumpFilename = filename;
    fpsLimit = frameCheckNumber;

    // Running CUDA kernel (bicubicFiltering) without visualization (QA
    // Testing/Verification)
    runAutoTest(argc, argv, (const char *)dumpFilename, g_FilterMode);
  } else {
    // This runs the CUDA kernel (bicubicFiltering) + OpenGL visualization
    initialize(argc, argv);
    glutMainLoop();
  }

  exit(EXIT_SUCCESS);
}
