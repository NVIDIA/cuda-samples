/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * CUDA 3D Volume Filtering sample
 *
 * This sample loads a 3D volume from disk and displays it using
 * ray marching and 3D textures.
 *
 * Note - this is intended to be an example of using 3D textures
 * in CUDA, not an optimized volume renderer.
 *
 * Changes
 * sgg 22/3/2010
 * - updated to use texture for display instead of glDrawPixels.
 * - changed to render from front-to-back rather than back-to-front.
 */

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime and Interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>
#include <helper_timer.h>

// CUDA utilities and system includes
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

const char *sSDKsample = "CUDA 3D Volume Filtering";


#include "volume.h"
#include "volumeFilter.h"
#include "volumeRender.h"

const char *volumeFilename = "Bucky.raw";
cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool  linearFiltering = true;
bool  preIntegrated = true;
StopWatchInterface *animationTimer = NULL;

float   filterFactor = 0.0f;
bool    filterAnimation = true;
int     filterIterations = 2;
float   filterTimeScale = 0.001f;
float   filterBias = 0.0f;
float4  filterWeights[3*3*3];

Volume  volumeOriginal;
Volume  volumeFilter0;
Volume  volumeFilter1;

GLuint pbo = 0;           // OpenGL pixel buffer object
GLuint volumeTex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc;
char **pArgv;

#define MAX(a,b) ((a > b) ? a : b)

//////////////////////////////////////////////////////////////////////////
// QA RELATED

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA 3D Volume Filtering: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = ftoi(MAX(1.f, ifps));
        sdkResetTimer(&timer);
    }
}

//////////////////////////////////////////////////////////////////////////
// 3D FILTER

static float filteroffsets[3*3*3][3] =
{
    {-1,-1,-1},{ 0,-1,-1},{ 1,-1,-1},
    {-1, 0,-1},{ 0, 0,-1},{ 1, 0,-1},
    {-1, 1,-1},{ 0, 1,-1},{ 1, 1,-1},

    {-1,-1, 0},{ 0,-1, 0},{ 1,-1, 0},
    {-1, 0, 0},{ 0, 0, 0},{ 1, 0, 0},
    {-1, 1, 0},{ 0, 1, 0},{ 1, 1, 0},

    {-1,-1, 1},{ 0,-1, 1},{ 1,-1, 1},
    {-1, 0, 1},{ 0, 0, 1},{ 1, 0, 1},
    {-1, 1, 1},{ 0, 1, 1},{ 1, 1, 1},
};

static float filterblur[3*3*3] =
{
    0,1,0,
    1,2,1,
    0,1,0,

    1,2,1,
    2,4,2,
    1,2,1,

    0,1,0,
    1,2,1,
    0,1,0,
};
static float filtersharpen[3*3*3] =
{
    0,0,0,
    0,-2,0,
    0,0,0,

    0,-2,0,
    -2,15,-2,
    0,-2,0,

    0,0,0,
    0,-2,0,
    0,0,0,
};

static float filterpassthru[3*3*3] =
{
    0,0,0,
    0,0,0,
    0,0,0,

    0,0,0,
    0,1,0,
    0,0,0,

    0,0,0,
    0,0,0,
    0,0,0,
};

void FilterKernel_init()
{
    float sumblur = 0.0f;
    float sumsharpen = 0.0f;

    for (int i = 0; i < 3*3*3; i++)
    {
        sumblur += filterblur[i];
        sumsharpen += filtersharpen[i];
    }

    for (int i = 0; i < 3*3*3; i++)
    {
        filterblur[i] /= sumblur;
        filtersharpen[i] /= sumsharpen;

        filterWeights[i].x = filteroffsets[i][0];
        filterWeights[i].y = filteroffsets[i][1];
        filterWeights[i].z = filteroffsets[i][2];
    }
}

void FilterKernel_update(float blurfactor)
{
    if (blurfactor > 0.0f)
    {
        for (int i = 0; i < 3*3*3; i++)
        {
            filterWeights[i].w = filterblur[i] * blurfactor + filterpassthru[i] * (1.0f - blurfactor);
        }
    }
    else
    {
        blurfactor = -blurfactor;

        for (int i = 0; i < 3*3*3; i++)
        {
            filterWeights[i].w = filtersharpen[i] * blurfactor + filterpassthru[i] * (1.0f - blurfactor);
        }
    }

}

void filter()
{
    if (filterAnimation)
    {
        filterFactor = cosf(sdkGetTimerValue(&animationTimer) * filterTimeScale);
    }

    FilterKernel_update(filterFactor);

    Volume *volumeRender = VolumeFilter_runFilter(&volumeOriginal,&volumeFilter0,&volumeFilter1,
                                                  filterIterations, 3*3*3,filterWeights,filterBias);

}

//////////////////////////////////////////////////////////////////////////
// RENDERING

// render image using CUDA
void render()
{

    VolumeRender_copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    VolumeRender_render(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, volumeOriginal.volumeTex);

    getLastCudaError("render kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    filter();
    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // draw using texture
    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, volumeTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

//////////////////////////////////////////////////////////////////////////
// LOGIC

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            #if defined (__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            break;

        case ' ':
            filterAnimation = !filterAnimation;

            if (!filterAnimation)
            {
                sdkStopTimer(&animationTimer);
            }
            else
            {
                sdkStartTimer(&animationTimer);
            }

            break;

        case 'f':
            linearFiltering = !linearFiltering;
            VolumeRender_setTextureFilterMode(linearFiltering, &volumeOriginal);
            break;

        case 'p':
            preIntegrated = !preIntegrated;
            VolumeRender_setPreIntegrated(preIntegrated);
            break;

        case '+':
            density += 0.01f;
            break;

        case '-':
            density -= 0.01f;
            break;

        case ']':
            brightness += 0.1f;
            break;

        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;

        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;

        case ',':
            transferScale -= 0.01f;
            break;

        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

//////////////////////////////////////////////////////////////////////////
// SAMPLE INIT/DEINIT

static int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void initPixelBuffer();
void reshape(int w, int h)
{
    width = w;
    height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}


void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA 3D Volume Filtering");

    if (!isGLVersionSupported(2,0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions are missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &volumeTex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &volumeTex);
    glBindTexture(GL_TEXTURE_2D, volumeTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

//////////////////////////////////////////////////////////////////////////

void cleanup()
{
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&animationTimer);

    Volume_deinit(&volumeOriginal);
    Volume_deinit(&volumeFilter0);
    Volume_deinit(&volumeFilter1);
    VolumeRender_deinit();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &volumeTex);
    }
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %d bytes\n", filename, (int)read);

    return data;
}

void initData(int argc, char **argv)
{
    // parse arguments
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "file", &filename))
    {
        volumeFilename = filename;
    }

    int n;

    if (checkCmdLineFlag(argc, (const char **) argv, "size"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "size");
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "xsize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "xsize");
        volumeSize.width = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "ysize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "ysize");
        volumeSize.height = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "zsize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "zsize");
        volumeSize.depth = n;
    }

    char *path = sdkFindFilePath(volumeFilename, argv[0]);

    if (path == 0)
    {
        printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    void *h_volume = loadRawFile(path, size);

    FilterKernel_init();
    Volume_init(&volumeOriginal,volumeSize, h_volume, 0);
    free(h_volume);
    Volume_init(&volumeFilter0, volumeSize, NULL, 1);
    Volume_init(&volumeFilter1, volumeSize, NULL, 1);
    VolumeRender_init();
    VolumeRender_setPreIntegrated(preIntegrated);

    sdkCreateTimer(&timer);
    sdkCreateTimer(&animationTimer);
    sdkStartTimer(&animationTimer);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

//////////////////////////////////////////////////////////////////////////
// AUTOMATIC TESTING
void runSingleTest(const char *ref_file, const char *exec_path)
{
    uint *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, width*height*sizeof(uint)));
    checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(uint)));

    float modelView[16] =
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 4.0f, 1.0f
    };

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    // call CUDA kernel, writing results to PBO
    VolumeRender_copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
    filterAnimation = false;

    // Start timer 0 and process n loops on the GPU
    int nIter = 10;
    float scale = 2.0f/float(nIter-1);

    for (int i = -1; i < nIter; i++)
    {
        if (i == 0)
        {
            cudaDeviceSynchronize();
            sdkStartTimer(&timer);
        }

        filterFactor = (float(i) * scale) - 1.0f;
        filterFactor = -filterFactor;
        filter();
        VolumeRender_render(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, volumeOriginal.volumeTex);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = sdkGetTimerValue(&timer)/(nIter * 1000.0);
    printf("volumeFiltering, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y);


    getLastCudaError("Error: kernel execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char *h_output = (unsigned char *)malloc(width*height*4);
    checkCudaErrors(cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost));

    sdkSavePPM4ub("volumefilter.ppm", h_output, width, height);
    bool bTestResult = sdkComparePPM("volumefilter.ppm", sdkFindFilePath(ref_file, exec_path),
                                     MAX_EPSILON_ERROR, THRESHOLD, true);

    checkCudaErrors(cudaFree(d_output));
    free(h_output);
    cleanup();

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void printHelp()
{
    printf("\nUsage: volumeFiltering <options>\n");
    printf("\t\t-file = filename.raw (volume file for input)\n\n");
    printf("\t\t-size = 64 (volume size, isotropic)\n\n");
    printf("\t\t-xsize = 128 (volume size, anisotropic)\n\n");
    printf("\t\t-ysize = 128 (volume size, anisotropic)\n\n");
    printf("\t\t-zsize = 32 (volume size, anisotropic)\n\n");
}

int
main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    char *ref_file = NULL;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s Starting...\n\n", sSDKsample);

    //start logs

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printHelp();
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        fpsLimit = frameCheckNumber;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
    }

    int device = findCudaDevice(argc, (const char **)argv);

    if (!ref_file)
    {
        initGL(&argc, argv);
    }

    // load volume data
    initData(argc, argv);

    printf(
        "Press \n"
        "  'SPACE'     to toggle animation\n"
        "  'p'         to toggle pre-integrated transfer function\n"
        "  '+' and '-' to change density (0.01 increments)\n"
        "  ']' and '[' to change brightness\n"
        "  ';' and ''' to modify transfer function offset\n"
        "  '.' and ',' to modify transfer function scale\n\n");

    if (ref_file)
    {
        runSingleTest(ref_file, argv[0]);
    }
    else
    {
        // This is the normal rendering path for VolumeRender
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutReshapeFunc(reshape);
        glutIdleFunc(idle);

        initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        glutMainLoop();
    }
}
