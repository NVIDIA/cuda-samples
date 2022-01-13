/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <unistd.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

void error_exit(const char* format, ... )
{
    va_list args;
    va_start( args, format );
    vfprintf( stderr, format, args );
    va_end( args );
    exit(1);
}

// GLES related includes and Xlib and EGL stuff
#include "graphics_interface.h"

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gles.h>
#include <helper_cuda.h>

#include "defines.h"
#include "fluidsGLES_kernels.h"

typedef float matrix4[4][4];
typedef float vector3[3];

#define MAX_EPSILON_ERROR 1.0f

const char *sSDKname = "fluidsGLES";
// CUDA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// CUDA FFT library (CUFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// CUDA-OpenGLES interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

void cleanup(void);
void reshape(int x, int y);

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth  = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

static int clicked  = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

int gui_mode; // For X window
// Rotate & translate variable temp., will remove and use shaders.
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;


// Particle data
GLuint vbo = 0,vao = 0;                 // OpenGLES vertex buffer object
GLuint m_texture = 0;
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGLES-CUDA exchange
static cData *particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

char *ref_file         = NULL;
bool g_bQAAddTestForce = true;
int  g_iFrameToCompare = 100;
int  g_TotalErrors     = 0;

bool g_bExitESC = false;

const unsigned int window_width  = 512;
const unsigned int window_height = 512;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

void autoTest(char **);
void displayFrame();
void keyboard(unsigned char key, int x, int y, int argc, char **argv);

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt);

void simulateFluids(void)
{
    // simulate fluid
    advectVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM, DT);
    diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
    updateVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
    advectParticles(vbo, dvfield, DIM, DIM, DT);
}

GLuint mesh_shader = 0;

void mat_identity(matrix4 m)
{
    m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] = 
    m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.0f;

    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
}

void mat_multiply(matrix4 m0, matrix4 m1)
{
    float m[4];
    for(int r = 0; r < 4; r++)
    {
        m[0] = m[1] = m[2] = m[3] = 0.0f;
        for(int c = 0; c < 4; c++)
        {
            for(int i = 0; i < 4; i++)
            {
                m[c] += m0[i][r] * m1[c][i];
            }
        }
        for(int c = 0; c < 4; c++)
        {
            m0[c][r] = m[c];
        }
    }
}

void mat4f_Ortho(float left, float right, float bottom, float top, float near, float far, matrix4 m)
{
    float r_l = right - left;
    float t_b = top - bottom;
    float f_n = far - near;
    float tx = - (right + left) / (right - left);
    float ty = - (top + bottom) / (top - bottom);
    float tz = - (far + near) / (far - near);

    matrix4 m2;

    m2[0][0] = 2.0f/ r_l;
    m2[0][1] = 0.0f;
    m2[0][2] = 0.0f;
    m2[0][3] = 0.0f;

    m2[1][0] = 0.0f; 
    m2[1][1] = 2.0f / t_b;
    m2[1][2] = 0.0f;
    m2[1][3] = 0.0f;

    m2[2][0] = 0.0f;
    m2[2][1] = 0.0f; 
    m2[2][2] = -2.0f / f_n;
    m2[2][3] = 0.0f;

    m2[3][0] = tx;
    m2[3][1] = ty; 
    m2[3][2] = tz;
    m2[3][3] = 1.0f;

    mat_multiply(m, m2); 
}

void readAndCompileShaderFromGLSLFile(GLuint new_shaderprogram, const char *filename, GLenum shaderType)
{
    FILE *file = fopen(filename,"rb"); // open shader text file
    if (!file) 
        error_exit("Filename %s does not exist\n", filename);

    /* get the size of the file and read it */
    fseek(file,0,SEEK_END);
    GLint size = ftell(file);
    char *data = (char*)malloc(sizeof(char)*(size + 1));
    memset(data, 0, sizeof(char)*(size + 1));
    fseek(file,0,SEEK_SET);
    size_t res = fread(data,1,size,file);
    fclose(file);

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const GLchar**)&data, &size);
    glCompileShader(shader);

    GET_GLERROR(0);
    GLint compile_success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_success);
    GET_GLERROR(0);

    if (compile_success == GL_FALSE)
    {
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

    free(data);
}

GLuint ShaderCreate(const char *vshader_filename, const char *fshader_filename)
{
    printf("Loading GLSL shaders %s %s\n", vshader_filename, fshader_filename);

    GLuint new_shaderprogram = glCreateProgram();

    GET_GLERROR(0);
    if (vshader_filename)
        readAndCompileShaderFromGLSLFile(new_shaderprogram, vshader_filename, GL_VERTEX_SHADER);

    GET_GLERROR(0);
    if (fshader_filename)
        readAndCompileShaderFromGLSLFile(new_shaderprogram, fshader_filename, GL_FRAGMENT_SHADER);

    GET_GLERROR(0);

    glLinkProgram(new_shaderprogram);

    GET_GLERROR(0);
    GLint link_success;
    glGetProgramiv(new_shaderprogram, GL_LINK_STATUS, &link_success);

    if (link_success == GL_FALSE)
    {
        printf("Linking of %s with %s failed!\n Reason:\n", vshader_filename, fshader_filename);

        GLint maxLength = 0;
        glGetShaderiv(new_shaderprogram, GL_INFO_LOG_LENGTH, &maxLength);

        char errorLog[maxLength];
        glGetShaderInfoLog(new_shaderprogram, maxLength, &maxLength, &errorLog[0]);

        printf("%s", errorLog);

        exit(EXIT_FAILURE);
    }

    return new_shaderprogram;
}

void motion(int x, int y)
{
    // Convert motion coordinates to domain
    float fx = (lastx / (float)wWidth);
    float fy = (lasty / (float)wHeight);
    int nx = (int)(fx * DIM);
    int ny = (int)(fy * DIM);

    if (clicked && nx < DIM-FR && nx > FR-1 && ny < DIM-FR && ny > FR-1)
    {
        int ddx = x - lastx;
        int ddy = y - lasty;
        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny-FR;
        int spx = nx-FR;
        addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
        lastx = x;
        lasty = y;
    }
}

//===========================================================================
// InitGraphicsState() - initialize OpenGLES
//===========================================================================
static void InitGraphicsState(int argc, char** argv)
{
    char *GL_version  = (char *)glGetString(GL_VERSION);
    char *GL_vendor   = (char *)glGetString(GL_VENDOR);
    char *GL_renderer = (char *)glGetString(GL_RENDERER);
  
    printf("Version: %s\n", GL_version);
    printf("Vendor: %s\n", GL_vendor);
    printf("Renderer: %s\n", GL_renderer);

    // Allocate and initialize host data
    GLint bsize;

    // initialize buffer object
    glGenBuffers(1, &vbo);  
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS, particles, GL_DYNAMIC_DRAW);

    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

    if (bsize != (sizeof(cData) * DS))
    {
        printf("Failed to initialize GL extensions.\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
   
    // GLSL stuff
    char *vertex_shader_path = sdkFindFilePath("mesh.vert.glsl", argv[0]);
    char *fragment_shader_path = sdkFindFilePath("mesh.frag.glsl", argv[0]);

    if (vertex_shader_path == NULL || fragment_shader_path == NULL)
    {
        printf("Error finding shader file\n");
        exit(EXIT_FAILURE);
    }

    mesh_shader = ShaderCreate(vertex_shader_path, fragment_shader_path);
    GET_GLERROR(0);

    free(vertex_shader_path);
    free(fragment_shader_path);
  
    glUseProgram(mesh_shader);
}

void displayFrame(void)
{
    if (!ref_file)
    {
        sdkStartTimer(&timer);
        simulateFluids();
    }

    GLint view_arr[4];	

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glUseProgram(mesh_shader);
   
    // Set modelview and projection matricies
    GLint h_ModelViewMatrix = glGetUniformLocation(mesh_shader, "modelview");
    GLint h_ProjectionMatrix = glGetUniformLocation(mesh_shader, "projection");
    matrix4 modelview;
    matrix4 projection;
    mat_identity(modelview);
    mat_identity(projection);

// (float left, float right, float bottom, float top, float near, float far, matrix4 m)
    mat4f_Ortho(0.0, 1.0, 1.0, 0.0, 0.0, 1.0, projection);
 
    glUniformMatrix4fv(h_ModelViewMatrix, 1, GL_FALSE, (GLfloat*)modelview);
    glUniformMatrix4fv(h_ProjectionMatrix, 1, GL_FALSE, (GLfloat*)projection);

    // Set position coords
    GLint h_position = glGetAttribLocation(mesh_shader, "a_position");
    glEnableVertexAttribArray(h_position);
    glVertexAttribPointer(h_position, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glDrawArrays(GL_POINTS, 0, DS*sizeof(cData));
    glDisableVertexAttribArray(h_position);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    if (ref_file)
    {
        return;
    }

    glUseProgram(0);
    // Finish timing before swap buffers to avoid refresh sync
    sdkStopTimer(&timer);
    graphics_swap_buffers();

    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
        graphics_set_windowtitle(fps);
        fpsCount = 0;
        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void autoTest(char **argv)
{
    CFrameBufferObject *fbo = new CFrameBufferObject(wWidth, wHeight, 4, false, GL_TEXTURE_2D);
    g_CheckRender = new CheckFBO(wWidth, wHeight, 4, fbo);

    g_CheckRender->setPixelFormat(GL_RGBA);
    g_CheckRender->setExecPath(argv[0]);
    g_CheckRender->EnableQAReadback(true);

    fbo->bindRenderPath();

    for (int count=0; count<g_iFrameToCompare; count++)
    {
        simulateFluids();

        // add in a little force so the automated testing is interesing.
        if (ref_file)
        {
            int x = wWidth/(count+1);
            int y = wHeight/(count+1);
            float fx = (x / (float)wWidth);
            float fy = (y / (float)wHeight);
            int nx = (int)(fx * DIM);
            int ny = (int)(fy * DIM);

            int ddx = 35;
            int ddy = 35;
            fx = ddx / (float)wWidth;
            fy = ddy / (float)wHeight;
            int spy = ny-FR;
            int spx = nx-FR;

            addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);

            lastx = x;
            lasty = y;
            getLastCudaError("addForces kernel failed");
        }
    }

    displayFrame();

    fbo->unbindRenderPath();

    // compare to offical reference image, printing PASS or FAIL.
    printf("> (Frame %d) Readback BackBuffer\n", 100);
    g_CheckRender->readback(wWidth, wHeight);
    g_CheckRender->savePPM("fluidsGLES.ppm", true, NULL);

    if (!g_CheckRender->PPMvsPPM("fluidsGLES.ppm", ref_file, MAX_EPSILON_ERROR, 0.25f))
    {
        g_TotalErrors++;
    }
}

// Run fluids Simulation
bool runFluidsSimulation(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);


    if (ref_file != NULL)
    {
    // command line mode only - auto test
        graphics_setup_window(0,0, wWidth, wHeight, sSDKname);
        InitGraphicsState(argc, argv); // set up GLES stuff
        autoTest(argv);
        cleanup();
    }
    else
    {
        // create X11 window and set up associated OpenGL ES context
        graphics_setup_window(0,0, wWidth, wHeight, sSDKname);

        InitGraphicsState(argc, argv); // set up GLES stuff

        glClear(GL_COLOR_BUFFER_BIT);
        graphics_swap_buffers();
        XEvent event;
        KeySym key;
        char text[255];

        while (1)
        {
            while (XPending(display) > 0)
            {
                XNextEvent(display, &event);

                if (event.type==Expose && event.xexpose.count==0)
                {
                    printf("Redraw requested!\n");
                }    

                if (event.type==KeyPress && XLookupString(&event.xkey,text,255,&key,0)==1)
                {
                    if (text[0] == 27 || text[0] == 'q' || text[0] == 'Q')
                    {
                        keyboard(text[0], 0, 0, argc, argv);
                        return true; 
                    }

                    if (text[0] == 114)
                    {
                        keyboard(text[0], 0, 0, argc, argv);
                    }
                    
                    printf("You pressed the %c key!\n",text[0]);
                }

                if (event.type==ButtonPress)
                {
                    lastx = event.xbutton.x;
                    lasty = event.xbutton.y;
                    clicked = !clicked;
                }

                if (event.type==ButtonRelease)
                {
                    lastx = event.xbutton.x;
                    lasty = event.xbutton.y;
                    clicked = !clicked;
                }

                if (event.type == MotionNotify)
                {
                    motion(event.xmotion.x, event.xmotion.y);               
                }
                else
                {
                    XFlush(display);
                }
            }
            displayFrame();
            usleep(1000);  // need not take full CPU and GPU
        }
    }

    return true;
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
    static int seed = 72191;
    char sq[22];

    if (ref_file)
    {
        seed *= seed;
        sprintf(sq, "%010d", seed);
        // pull the middle 5 digits out of sq
        sq[8] = 0;
        seed = atoi(&sq[3]);

        return seed/99999.f;
    }
    else
    {
        return rand()/(float)RAND_MAX;
    }
}

void initParticles(cData *p, int dx, int dy)
{
    int i, j;
    for (i = 0; i < dy; i++)
    {
        for (j = 0; j < dx; j++)
        {
            p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
            p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
        }
    }
}

void keyboard(unsigned char key, int x, int y, int argc, char **argv)
{
    switch (key)
    {
        case 'q':
        case 'Q':
        case  27:
            g_bExitESC = true;
            cleanup();
            graphics_close_window(); // close window and destroy OpenGL ES context
            return;
            break;
        case 'r':
            printf("\nResetting\n");
            memset(hvfield, 0, sizeof(cData) * DS);
            cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);

            initParticles(particles, DIM, DIM);

            checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

            getLastCudaError("cudaGraphicsUnregisterBuffer failed");

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDeleteBuffers(1, &vbo);
            InitGraphicsState(argc, argv); // set up GLES stuff
            graphics_swap_buffers();

            getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
            break;

        default:
            break;
    }
}

void cleanup(void)
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

    deleteTexture();

    // Free all host and device resources
    free(hvfield);
    free(particles);
    checkCudaErrors(cudaFree(dvfield));
    checkCudaErrors(cudaFree(vxfield));
    checkCudaErrors(cudaFree(vyfield));
    checkCudaErrors(cufftDestroy(planr2c));
    checkCudaErrors(cufftDestroy(planc2r));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);

    sdkDeleteTimer(&timer);
}

int main(int argc, char **argv)
{
    int devID;
    cudaDeviceProp deviceProps;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s Starting...\n\n", sSDKname);

    printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

#if defined (__aarch64__) || defined(__arm__)
    // find iGPU on the system which is compute capable which will perform GLES-CUDA interop
    devID = findIntegratedGPU();
#else
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);
#endif

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n",
           deviceProps.name, deviceProps.multiProcessorCount);

    // automated build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
    }

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    hvfield = (cData *)malloc(sizeof(cData) * DS);
    memset(hvfield, 0, sizeof(cData) * DS);

    // Allocate and initialize device data
    checkCudaErrors(cudaMallocPitch((void **)&dvfield, &tPitch, sizeof(cData)*DIM, DIM));

    checkCudaErrors(cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice));
    // Temporary complex velocity field data
    checkCudaErrors(cudaMalloc((void **)&vxfield, sizeof(cData) * PDS));
    checkCudaErrors(cudaMalloc((void **)&vyfield, sizeof(cData) * PDS));

    setupTexture(DIM, DIM);

    // Create particle array
    particles = (cData *)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);

    initParticles(particles, DIM, DIM);

    // Create CUFFT transform plan configuration
    checkCudaErrors(cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R));

    runFluidsSimulation(argc, argv, ref_file);

    if (ref_file)
    {
        printf("[fluidsGLES] - Test Results: %d Failures\n", g_TotalErrors);
        exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    sdkDeleteTimer(&timer);

    if (!ref_file)
    {
        exit(EXIT_SUCCESS);
    }

    return 0;
}
