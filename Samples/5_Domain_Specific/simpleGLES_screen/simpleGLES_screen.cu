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
   This example demonstrates how to use the CUDA C bindings to OpenGL ES to
   dynamically modify a vertex buffer using a CUDA C kernel.

   The steps are:
   1. Create an empty vertex buffer object (VBO)
   2. Register the VBO with CUDA C
   3. Map the VBO for writing from CUDA C
   4. Run CUDA C kernel to modify the vertex positions
   5. Unmap the VBO
   6. Render the results using OpenGL ES

   Host code
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <stdarg.h>
#include <unistd.h>
#include <screen/screen.h>

#include "graphics_interface.c"

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 0.0f
#define THRESHOLD 0.0f
#define REFRESH_DELAY 1  // ms

#define GUI_IDLE 0x100
#define GUI_ROTATE 0x101
#define GUI_TRANSLATE 0x102

int gui_mode;

////////////////////////////////////////////////////////////////////////////////
// Default configuration
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int dispno = 0;

// constants
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// OpenGL ES variables and interop with CUDA C
GLuint mesh_vao, mesh_vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// UI / mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Frame statistics
int frame;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

// The default number of seconds after which the test will end.
#define TIME_LIMIT 10.0  // 10 secs

// Flag indicating it is time to shut down
static GLboolean shutdown = GL_FALSE;

// Callback to close window
static void closeCB_app(void) { shutdown = GL_TRUE; }

// Callback to handle key presses
static void keyCB_app(char key, int state) {
  // Ignoring releases
  if (!state) return;

  if ((key == 'q') || (key == 'Q') || (key == NvGlDemoKeyCode_Escape))
    shutdown = GL_TRUE;
}

// Auto-Verification Code
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a, b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

// CUDA functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGLES on Screen (VBO)";

void computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    fpsCount = 0;
    fpsLimit = (int)MAX(avgFPS, 1.f);

    sdkResetTimer(&timer);
  }

  char fps[256];
  sprintf(fps, "Cuda/OpenGL ES Interop (VBO): %3.1f fps (Max 1000 fps)",
          avgFPS);
  graphics_set_windowtitle(fps);
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width,
                                  unsigned int height, float time) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // calculate uv coordinates
  float u = x / (float)width;
  float v = y / (float)height;
  u = u * 2.0f - 1.0f;
  v = v * 2.0f - 1.0f;

  // calculate simple sine wave pattern
  float freq = 4.0f;
  float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

  // write output vertex
  pos[y * width + x] = make_float4(u, w, v, 1.0f);
}

void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time) {
  // execute the kernel
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  simple_vbo_kernel<<<grid, block>>>(pos, mesh_width, mesh_height, time);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource) {
  // map OpenGL buffer object for writing from CUDA
  float4 *dptr;
  cudaGraphicsMapResources(1, vbo_resource, 0);
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                       *vbo_resource);

  launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

  // unmap buffer object
  cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename) {
  printf("sdkDumpBin: <%s>\n", filename);
  FILE *fp;
  FOPEN(fp, filename, "wb");
  fwrite(data, bytes, 1, fp);
  fflush(fp);
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file) {
  char *reference_file = NULL;
  void *imageData = malloc(mesh_width * mesh_height * sizeof(float));

  // execute the kernel
  launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

  cudaDeviceSynchronize();
  getLastCudaError("launch_kernel failed");

  cudaMemcpy(imageData, d_vbo_buffer, mesh_width * mesh_height * sizeof(float),
             cudaMemcpyDeviceToHost);

  sdkDumpBin2(imageData, mesh_width * mesh_height * sizeof(float),
              "simpleGLES_screen.bin");
  reference_file = sdkFindFilePath(ref_file, argv[0]);

  if (reference_file &&
      !sdkCompareBin2BinFloat("simpleGLES_screen.bin", reference_file,
                              mesh_width * mesh_height * sizeof(float),
                              MAX_EPSILON_ERROR, THRESHOLD, pArgv[0])) {
    g_TotalErrors++;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display_thisframe(float time_delta) {
  sdkStartTimer(&timer);

  // run CUDA kernel to generate vertex positions
  runCuda(&cuda_vbo_resource);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);

  glFinish();

  g_fAnim += time_delta;

  sdkStopTimer(&timer);
  computeFPS();
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo) {
  if (!d_vbo_buffer) {
    printf("%s: Mapping result buffer from OpenGL ES\n", __FUNCTION__);

    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    // map buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float *data = (float *)glMapBufferRange(
        GL_ARRAY_BUFFER, 0, mesh_width * mesh_height * 4 * sizeof(float),
        GL_READ_ONLY);

    // check result
    if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
      // write file for regression test
      sdkWriteFile<float>("./data/regression.dat", data,
                          mesh_width * mesh_height * 3, 0.0, false);
    }

    // unmap GL buffer object
    if (!glUnmapBuffer(GL_ARRAY_BUFFER)) {
      fprintf(stderr, "Unmap buffer failed.\n");
      fflush(stderr);
    }

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

    CHECK_GLERROR();
  }
}

GLuint mesh_shader = 0;

void readAndCompileShaderFromGLSLFile(GLuint new_shaderprogram,
                                      const char *filename, GLenum shaderType) {
  FILE *file = fopen(filename, "rb");  // open shader text file
  if (!file) {
    error_exit("Filename %s does not exist\n", filename);
  }

  // get the size of the file and read it
  fseek(file, 0, SEEK_END);
  GLint size = ftell(file);
  char *data = (char *)malloc(sizeof(char) * (size + 1));
  memset(data, 0, sizeof(char) * (size + 1));
  fseek(file, 0, SEEK_SET);
  size_t res = fread(data, 1, size, file);
  fclose(file);

  GLuint shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, (const GLchar **)&data, &size);
  glCompileShader(shader);

  CHECK_GLERROR();
  GLint compile_success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_success);
  CHECK_GLERROR();

  if (compile_success == GL_FALSE) {
    printf("Compilation of %s failed!\n Reason:\n", filename);

    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

    char errorLog[maxLength];
    glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

    printf("%s", errorLog);

    glDeleteShader(shader);
    exit(1);
  }

  glAttachShader(new_shaderprogram, shader);
  glDeleteShader(shader);

  free(data);
}

GLuint ShaderCreate(const char *vshader_filename,
                    const char *fshader_filename) {
  printf("Loading GLSL shaders %s %s\n", vshader_filename, fshader_filename);

  GLuint new_shaderprogram = glCreateProgram();

  CHECK_GLERROR();
  if (vshader_filename) {
    readAndCompileShaderFromGLSLFile(new_shaderprogram, vshader_filename,
                                     GL_VERTEX_SHADER);
  }

  CHECK_GLERROR();
  if (fshader_filename) {
    readAndCompileShaderFromGLSLFile(new_shaderprogram, fshader_filename,
                                     GL_FRAGMENT_SHADER);
  }

  CHECK_GLERROR();

  glLinkProgram(new_shaderprogram);

  CHECK_GLERROR();
  GLint link_success;
  glGetProgramiv(new_shaderprogram, GL_LINK_STATUS, &link_success);

  if (link_success == GL_FALSE) {
    printf("Linking of %s with %s failed!\n Reason:\n", vshader_filename,
           fshader_filename);

    GLint maxLength = 0;
    glGetShaderiv(new_shaderprogram, GL_INFO_LOG_LENGTH, &maxLength);

    char errorLog[maxLength];
    glGetShaderInfoLog(new_shaderprogram, maxLength, &maxLength, &errorLog[0]);

    printf("%s", errorLog);

    exit(EXIT_FAILURE);
  }

  return new_shaderprogram;
}

//===========================================================================
// InitGraphicsState() - initialize OpenGL
//===========================================================================
static void InitGraphicsState(void) {
  char *GL_version = (char *)glGetString(GL_VERSION);
  char *GL_vendor = (char *)glGetString(GL_VENDOR);
  char *GL_renderer = (char *)glGetString(GL_RENDERER);

  printf("Version: %s\n", GL_version);
  printf("Vendor: %s\n", GL_vendor);
  printf("Renderer: %s\n", GL_renderer);

  // RENDERING SETUP (OpenGL ES or OpenGL Core Profile!)
  glGenVertexArrays(1, &mesh_vao);  // Features' Vertex Array Object allocation
  glBindVertexArray(mesh_vao);      // bind VAO

  // initialize buffer object
  glGenBuffers(1, &mesh_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo);

  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer((GLuint)0, 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, mesh_vbo,
                                               cudaGraphicsMapFlagsNone));

  // GLSL stuff
  char *vertex_shader_path = sdkFindFilePath("mesh.vert.glsl", pArgv[0]);
  char *fragment_shader_path = sdkFindFilePath("mesh.frag.glsl", pArgv[0]);

  if (vertex_shader_path == NULL || fragment_shader_path == NULL) {
    printf("Error finding shader file\n");
    exit(EXIT_FAILURE);
  }

  mesh_shader = ShaderCreate(vertex_shader_path, fragment_shader_path);
  CHECK_GLERROR();

  free(vertex_shader_path);
  free(fragment_shader_path);

  glUseProgram(mesh_shader);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file) {
  // command line mode only
  if (ref_file != NULL) {
    // This will pick the best possible CUDA capable device
    // int devID = findCudaDevice(argc, (const char **)argv);
#if defined(__aarch64__) || defined(__arm__)
    // find iGPU on the system which is compute capable which will perform
    // GLES-CUDA interop
    int devID = findIntegratedGPU();
#else
    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);
#endif

    // create VBO
    checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer,
                               mesh_width * mesh_height * 4 * sizeof(float)));

    // run the cuda part
    runAutoTest(devID, argv, ref_file);

    // check result of Cuda step
    checkResultCuda(argc, argv, mesh_vbo);

    cudaFree(d_vbo_buffer);
    d_vbo_buffer = NULL;
  } else {
    double endTime = TIME_LIMIT;

    // this would use command-line specified CUDA device, note that CUDA
    // defaults to highest Gflops/s device
    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
      error_exit("Device setting not yet implemented!\n");
    }

    // display selection
    if (checkCmdLineFlag(argc, (const char **)argv, "dispno")) {
      dispno = getCmdLineArgumentInt(argc, (const char **)argv, "dispno");
    }

    // Window width
    if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
      window_width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    }

    // Window Height
    if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
      window_height =
          getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }

    // Determine how long to run for in secs: default is 10s
    if (checkCmdLineFlag(argc, (const char **)argv, "runtime")) {
      endTime = getCmdLineArgumentInt(argc, (const char **)argv, "runtime");
    }

    SetCloseCB(closeCB_app);
    SetKeyCB(keyCB_app);

    // create QNX screen window and set up associated OpenGL ES context
    graphics_setup_window(0, 0, window_width, window_height, sSDKsample,
                          dispno);

#if defined(__aarch64__) || defined(__arm__)
    // find iGPU on the system which is compute capable which will perform
    // GLES-CUDA interop
    int devID = findIntegratedGPU();
#else
    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);
#endif
    InitGraphicsState();  // set up GLES stuff

    glClearColor(0, 0.5, 1, 1);  // blue-ish background
    glClear(GL_COLOR_BUFFER_BIT);

    graphics_swap_buffers();

    int frame = 0;

    struct timeval begin, end;
    gettimeofday(&begin, NULL);

    // Print runtime
    if (endTime < 0.0) {
      endTime = TIME_LIMIT;
      printf(" running forever...\n");
    } else {
      printf(" running for %f seconds...\n", endTime);
    }

    while (!shutdown) {
      frame++;
      display_thisframe(0.010);
      usleep(1000);
      graphics_swap_buffers();
      CheckEvents();

      gettimeofday(&end, 0);
      double elapsed = (end.tv_sec - begin.tv_sec) +
                       ((end.tv_usec - begin.tv_usec) / 1000000.0);

      // Check whether time limit has been exceeded
      if (!shutdown) shutdown = (endTime <= elapsed);
    }

    // NOTE: Before destroying OpenGL ES context, must unregister all shared
    // resources from CUDA !
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

    graphics_close_window();  // close window and destroy OpenGL ES context
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif

  printf("%s starting...\n", sSDKsample);

  if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
      // In this mode, we run without OpenGL and see if VBO is generated
      // correctly
      getCmdLineArgumentString(argc, (const char **)argv, "file",
                               (char **)&ref_file);
    }
  }

  printf("\n");

  runTest(argc, argv, ref_file);

  printf("%s completed, returned %s\n", sSDKsample,
         (g_TotalErrors == 0) ? "OK" : "ERROR!");

  exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
