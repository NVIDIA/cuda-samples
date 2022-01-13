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

#pragma warning(disable : 4312)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#endif

// including CUDA headers and helper functions
#include <builtin_types.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_d3d9_interop.h>

// SDK helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include "fluidsD3D9_kernels.h"
#include <rendercheck_d3d9.h>
#include <DirectXMath.h>
using namespace DirectX;

#define MAX_EPSILON 10

static char *SDK_name = "fluidsD3D9";

int *pArgc = NULL;
char **pArgv = NULL;

// CUDA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// CUDA FFT library (CUFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

HWND hWnd;                                // Window handle
LPDIRECT3D9EX g_pD3D = NULL;              // Used to create the D3DDevice
unsigned int g_iAdapter = NULL;           // Adapter
LPDIRECT3DDEVICE9EX g_pD3DDevice = NULL;  // Rendering device
LPDIRECT3DVERTEXBUFFER9 g_pVB = NULL;     // Buffer to hold particles
LPDIRECT3DTEXTURE9 g_pTexture = NULL;     // Texture to render points

struct cudaGraphicsResource *cuda_VB_resource;  // handles D3D9-CUDA exchange

HRESULT InitD3D9(HWND hWnd);
HRESULT InitD3D9RenderState();
HRESULT InitCUDA();
HRESULT InitCUFFT();
HRESULT InitVertexBuffer();
HRESULT FreeVertexBuffer();
HRESULT InitPointTexture();
HRESULT RestoreContextResources();

#define D3DFVF_CUSTOMVERTEX (D3DFVF_XYZ | D3DFVF_DIFFUSE)
void updateVB(void);
void initParticles(cData *p, int dx, int dy);

// CUFFT plan handle
static cufftHandle g_planr2c;
static cufftHandle g_planc2r;
static cData *g_vxfield = NULL;
static cData *g_vyfield = NULL;

cData *g_hvfield = NULL;
cData *g_dvfield = NULL;
static int wWidth = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// Particle data
static Vertex *g_mparticles = NULL;
static cData *g_particles = NULL;
static int lastx = 0, lasty = 0;

// Texture pitch
// unsigned int g_tPitch = 0;
size_t g_tPitch = 0;

D3DDISPLAYMODEEX g_d3ddm;
D3DPRESENT_PARAMETERS g_d3dpp;

bool g_bWindowed = true;
bool g_bDeviceLost = false;
bool g_bPassed = true;
int g_iFrameToCompare = 100;
bool g_bQAAddTestForce = true;
char *ref_file = NULL;

#define NAME_LEN 512

char device_name[NAME_LEN];

VOID Cleanup() {
  // Unregister vertex buffer
  FreeVertexBuffer();

  deleteTexture();

  // Free all host and device resources
  free(g_hvfield);
  free(g_particles);
  cudaFree(g_dvfield);
  cudaFree(g_vxfield);
  cudaFree(g_vyfield);

  cufftDestroy(g_planr2c);
  cufftDestroy(g_planc2r);

  if (g_pTexture != NULL) {
    g_pTexture->Release();
    g_pTexture = NULL;
  }

  if (g_pD3DDevice != NULL) {
    g_pD3DDevice->Release();
    g_pD3DDevice = NULL;
  }

  if (g_pD3D != NULL) {
    g_pD3D->Release();
    g_pD3D = NULL;
  }

  sdkDeleteTimer(&timer);
}

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_DESTROY:
      Cleanup();
      PostQuitMessage(0);
      exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
      return 0;

    case WM_KEYDOWN:
      switch (wParam) {
        case 27:
          Cleanup();
          PostQuitMessage(0);
          break;

        case 0x52:
          memset(g_hvfield, 0, sizeof(cData) * DS);
          cudaMemcpy(g_dvfield, g_hvfield, sizeof(cData) * DS,
                     cudaMemcpyHostToDevice);

          initParticles(g_particles, DIM, DIM);
          cudaGraphicsUnregisterResource(cuda_VB_resource);

          updateVB();

          cudaGraphicsD3D9RegisterResource(&cuda_VB_resource, g_pVB,
                                           cudaD3D9RegisterFlagsNone);
          getLastCudaError("cudaGraphicsD3D9RegisterResource failed");
          break;

        default:
          break;
      }

      break;

    case WM_SIZE:
      wWidth = LOWORD(lParam);
      wHeight = HIWORD(lParam);
      break;

    case WM_MOUSEMOVE:
      if (wParam == MK_LBUTTON) {
        clicked = 1;
      } else {
        clicked = 0;
      }

      int x = LOWORD(lParam), y = HIWORD(lParam);

      // Convert motion coordinates to domain
      float fx = (x / (float)wWidth);
      float fy = (y / (float)wHeight);
      int nx = (int)(fx * DIM);
      int ny = (int)(fy * DIM);

      if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR &&
          ny > FR - 1) {
        int ddx = LOWORD(lParam) - lastx;
        int ddy = HIWORD(lParam) - lasty;

        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny - FR;
        int spx = nx - FR;
        addForces(g_dvfield, DIM, DIM, spx, spy, FORCE * DT * fx,
                  FORCE * DT * fy, FR, g_tPitch);
        lastx = x;
        lasty = y;
      }

      break;
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}

HRESULT InitVertexBuffer() {
  // Create the vertex buffer.
  if (FAILED(g_pD3DDevice->CreateVertexBuffer(DS * sizeof(Vertex), 0,
                                              D3DFVF_CUSTOMVERTEX,
                                              D3DPOOL_DEFAULT, &g_pVB, NULL))) {
    return E_FAIL;
  }

  // Initialize the Vertex Buffer with the particles
  updateVB();

  cudaGraphicsD3D9RegisterResource(&cuda_VB_resource, g_pVB,
                                   cudaD3D9RegisterFlagsNone);
  getLastCudaError("cudaGraphicsD3D9RegisterResource failed");

  return S_OK;
}

HRESULT InitPointTexture() {
  // Create the texture.
  int width = 64;
  int height = width;

  if (FAILED(g_pD3DDevice->CreateTexture(
          width, height, 0, D3DUSAGE_AUTOGENMIPMAP | D3DUSAGE_DYNAMIC,
          D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_pTexture, NULL))) {
    return E_FAIL;
  }

  // Fill in top level
  D3DLOCKED_RECT rect;

  if (FAILED(g_pTexture->LockRect(0, &rect, 0, 0))) {
    return E_FAIL;
  }

  typedef unsigned int TexelType;
  TexelType *texel = (TexelType *)rect.pBits;

  for (int y = -height / 2; y < height / 2; ++y) {
    float yf = y + 0.5f;
    TexelType *t = texel;

    for (int x = -width / 2; x < width / 2; ++x) {
      float xf = x + 0.5f;
      float radius = (float)width / 32;
      float dist = sqrtf(xf * xf + yf * yf) / radius;
      float n = 0.1f;
      float value;

      if (dist < 1) {
        value = 1 - 0.5f * powf(dist, n);
      } else if (dist < 2) {
        value = 0.5f * powf(2 - dist, n);
      } else {
        value = 0;
      }

      value *= 75;
      unsigned char *c = (unsigned char *)t;
      c[0] = c[1] = c[2] = c[3] = (unsigned char)value;
      ++t;
    }

    texel += rect.Pitch / sizeof(TexelType);
  }

  if (FAILED(g_pTexture->UnlockRect(0))) {
    return E_FAIL;
  }

  // Set sampler state
  if (FAILED(g_pD3DDevice->SetSamplerState(0, D3DSAMP_MINFILTER,
                                           D3DTEXF_LINEAR))) {
    return E_FAIL;
  }

  if (FAILED(g_pD3DDevice->SetSamplerState(0, D3DSAMP_MAGFILTER,
                                           D3DTEXF_LINEAR))) {
    return E_FAIL;
  }

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: FreeVertexBuffer()
// Desc: Free's the Vertex Buffer resource
//-----------------------------------------------------------------------------
HRESULT FreeVertexBuffer() {
  if (g_pVB != NULL) {
    // Unregister vertex buffer
    cudaGraphicsUnregisterResource(cuda_VB_resource);
    getLastCudaError("cudaGraphicsUnregisterResource failed");

    g_pVB->Release();
  }

  return S_OK;
}

void updateVB(void) {
  Vertex *data = new Vertex[DS];
  g_pVB->Lock(0, DS * sizeof(Vertex), (void **)&data, 0);

  for (int i = 0; i < DS; i++) {
    data[i].x = g_particles[i].x;
    data[i].y = g_particles[i].y;
    data[i].z = 0.f;
    data[i].c = 0xff00ff00;
  }

  g_pVB->Unlock();
}

HRESULT InitD3D9(HWND hWnd) {
  // Create the D3D object.
  if (S_OK != Direct3DCreate9Ex(D3D_SDK_VERSION, &g_pD3D)) {
    return E_FAIL;
  }

  D3DADAPTER_IDENTIFIER9 adapterId;
  int device;
  bool bDeviceFound = false;
  printf("\n");

  cudaError cuStatus;

  for (g_iAdapter = 0; g_iAdapter < g_pD3D->GetAdapterCount(); g_iAdapter++) {
    HRESULT hr = g_pD3D->GetAdapterIdentifier(g_iAdapter, 0, &adapterId);

    if (FAILED(hr)) {
      continue;
    }

    // clear any errors we got while querying invalid compute devices
    cuStatus = cudaGetLastError();
    cuStatus = cudaD3D9GetDevice(&device, adapterId.DeviceName);
    printLastCudaError("cudaD3D9GetDevice failed");  // This prints and resets
                                                     // the cudaError to
                                                     // cudaSuccess

    printf("> Display Device #%d: \"%s\" %s Direct3D9\n", g_iAdapter,
           adapterId.Description,
           (cuStatus == cudaSuccess) ? "supports" : "does not support");

    if (cudaSuccess == cuStatus) {
      bDeviceFound = true;
      STRCPY(device_name, NAME_LEN, adapterId.Description);
      break;
    }
  }

  // we check to make sure we have found a cuda-compatible D3D device to work on
  if (!bDeviceFound) {
    printf("\nNo CUDA-compatible Direct3D9 device available\n");
    // Release the D3D device
    g_pD3D->Release();
    exit(EXIT_SUCCESS);
  }

  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  strcpy(device_name, deviceProp.name);

  RECT rc;
  GetClientRect(hWnd, &rc);
  g_pD3D->GetAdapterDisplayModeEx(g_iAdapter, &g_d3ddm, NULL);

  // Set up the structure used to create the D3DDevice
  D3DPRESENT_PARAMETERS d3dpp;
  ZeroMemory(&d3dpp, sizeof(d3dpp));
  d3dpp.Windowed = TRUE;
  d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  d3dpp.BackBufferFormat = g_d3ddm.Format;  // D3DFMT_UNKNOWN;

  // Create the D3DDevice
  if (FAILED(g_pD3D->CreateDeviceEx(g_iAdapter, D3DDEVTYPE_HAL, hWnd,
                                    D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp,
                                    NULL, &g_pD3DDevice))) {
    return E_FAIL;
  } else {
    return S_OK;
  }
}

// Initialize the D3D Rendering State
HRESULT InitD3D9RenderState() {
  // Set projection matrix
  XMMATRIX matProj;
  XMFLOAT4X4 matProjFloat;
  matProj = XMMatrixOrthographicOffCenterLH(0, 1, 1, 0, 0, 1);
  XMStoreFloat4x4(&matProjFloat, matProj);
  g_pD3DDevice->SetTransform(D3DTS_PROJECTION, (D3DMATRIX *)&matProjFloat);

  // Turn off D3D lighting, since we are providing our own vertex colors
  if (FAILED(g_pD3DDevice->SetRenderState(D3DRS_LIGHTING, FALSE))) {
    return E_FAIL;
  }

  return S_OK;
}

HRESULT InitCUDA() {
  printf("InitCUDA() g_pD3DDevice = %p\n", g_pD3DDevice);

  // Now we need to bind a CUDA context to the DX9 device
  // This is the CUDA 2.0 DX9 interface (required for Windows XP and Vista)
  cudaD3D9SetDirect3DDevice(g_pD3DDevice);
  getLastCudaError("cudaD3D9SetDirect3DDevice failed");

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! RestoreContextResourcess
//    - this function restores all of the CUDA/D3D resources and contexts
////////////////////////////////////////////////////////////////////////////////
HRESULT RestoreContextResources() {
  // Reinitialize D3D9 resources, CUDA resources/contexts
  InitCUDA();
  InitD3D9RenderState();
  InitCUFFT();
  InitVertexBuffer();
  InitPointTexture();

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! DeviceLostHandler
//    - this function handles reseting and initialization of the D3D device
//      in the event this Device gets Lost
////////////////////////////////////////////////////////////////////////////////
HRESULT DeviceLostHandler() {
  HRESULT hr = S_OK;

  // test the cooperative level to see if it's okay
  // to render
  if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel())) {
    // if the device was truly lost, (i.e., a fullscreen device just lost
    // focus), wait
    // until we g_et it back
    if (hr == D3DERR_DEVICELOST) {
      return S_OK;
    }

    // eventually, we will g_et this return value,
    // indicating that we can now reset the device
    if (hr == D3DERR_DEVICENOTRESET) {
      // if we are windowed, read the desktop mode and use the same format for
      // the back buffer; this effectively turns off color conversion

      if (g_bWindowed) {
        g_pD3D->GetAdapterDisplayModeEx(g_iAdapter, &g_d3ddm, NULL);
        g_d3dpp.BackBufferFormat = g_d3ddm.Format;
      }

      // now try to reset the device
      if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp))) {
        return hr;
      } else {
        // This is a common function we use to restore all hardware
        // resources/state
        RestoreContextResources();

        // we have acquired the device
        g_bDeviceLost = false;
      }
    }
  }

  return hr;
}

HRESULT InitCUFFT() {
  // You can only call CUDA D3D9 device has been bound to the CUDA
  // context, otherwise it will not work
  g_hvfield = (cData *)malloc(sizeof(cData) * DS);
  memset(g_hvfield, 0, sizeof(cData) * DS);

  // Allocate and initialize device data
  cudaMallocPitch((void **)&g_dvfield, &g_tPitch, sizeof(cData) * DIM, DIM);

  cudaMemcpy(g_dvfield, g_hvfield, sizeof(cData) * DS, cudaMemcpyHostToDevice);

  // Temporary complex velocity field data
  cudaMalloc((void **)&g_vxfield, sizeof(cData) * PDS);
  cudaMalloc((void **)&g_vyfield, sizeof(cData) * PDS);

  setupTexture(DIM, DIM);

  // Create particle array
  g_particles = (cData *)malloc(sizeof(cData) * DS);
  memset(g_particles, 0, sizeof(cData) * DS);

  initParticles(g_particles, DIM, DIM);

  // Create CUFFT transform plan configuration
  cufftPlan2d(&g_planr2c, DIM, DIM, CUFFT_R2C);
  cufftPlan2d(&g_planc2r, DIM, DIM, CUFFT_C2R);

  return S_OK;
}

HRESULT Render(void) {
  HRESULT hr = S_OK;

  // Normal case where CUDA Device is not lost
  if (!g_bDeviceLost) {
    sdkStartTimer(&timer);

    advectVelocity(g_dvfield, (float *)g_vxfield, (float *)g_vyfield, DIM,
                   RPADW, DIM, DT, g_tPitch);
    {
      // Forward FFT
      cufftExecR2C(g_planr2c, (cufftReal *)g_vxfield,
                   (cufftComplex *)g_vxfield);
      cufftExecR2C(g_planr2c, (cufftReal *)g_vyfield,
                   (cufftComplex *)g_vyfield);

      diffuseProject(g_vxfield, g_vyfield, CPADW, DIM, DT, VIS, g_tPitch);

      // Inverse FFT
      cufftExecC2R(g_planc2r, (cufftComplex *)g_vxfield,
                   (cufftReal *)g_vxfield);
      cufftExecC2R(g_planc2r, (cufftComplex *)g_vyfield,
                   (cufftReal *)g_vyfield);
    }
    updateVelocity(g_dvfield, (float *)g_vxfield, (float *)g_vyfield, DIM,
                   RPADW, DIM, g_tPitch);

    // Map D3D9 vertex buffer to CUDA
    {
      size_t num_bytes;
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_VB_resource, 0));
      getLastCudaError("cudaGraphicsMapResources failed");
      // This gets a pointer from the Vertex Buffer
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
          (void **)&g_mparticles, &num_bytes, cuda_VB_resource));
      getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

      advectParticles(g_mparticles, g_dvfield, DIM, DIM, DT, g_tPitch);

      // Unmap vertex buffer
      checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_VB_resource, 0));
      getLastCudaError("cudaGraphicsUnmapResource failed");
    }

    g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 0, 0), 1.0f,
                        0);
    g_pD3DDevice->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    g_pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    g_pD3DDevice->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE);
    g_pD3DDevice->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ONE);
    g_pD3DDevice->SetRenderState(D3DRS_POINTSPRITEENABLE, TRUE);
    float size = 16;
    g_pD3DDevice->SetRenderState(D3DRS_POINTSIZE, *((DWORD *)&size));
    g_pD3DDevice->SetTexture(0, g_pTexture);

    if (SUCCEEDED(g_pD3DDevice->BeginScene())) {
      // Draw particles
      g_pD3DDevice->SetStreamSource(0, g_pVB, 0, sizeof(Vertex));
      g_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);
      g_pD3DDevice->DrawPrimitive(D3DPT_POINTLIST, 0, DS);

      g_pD3DDevice->EndScene();
    }

    // Finish timing before swap buffers to avoid refresh sync
    sdkStopTimer(&timer);
    // Present the backbuffer contents to the display
    hr = g_pD3DDevice->Present(NULL, NULL, NULL, NULL);

    if (hr == D3DERR_DEVICELOST) {
      fprintf(stderr, "drawScene Present = %08x detected D3D DeviceLost\n", hr);
      g_bDeviceLost = true;

      FreeVertexBuffer();
    }

    fpsCount++;

    if (fpsCount == fpsLimit) {
      char fps[256];
      float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
      sprintf(fps, "CUDA/D3D9 Stable Fluids (%d x %d): %3.1f fps", DIM, DIM,
              ifps);
      SetWindowText(hWnd, fps);
      fpsCount = 0;
      fpsLimit = (int)MAX(ifps, 1.f);
      sdkResetTimer(&timer);
    }
  } else {
    // Begin code to handle case where the D3D9 device is lost
    if (FAILED(hr = DeviceLostHandler())) {
      fprintf(stderr, "DeviceLostHandler FAILED returned %08x\n", hr);
      return hr;
    }

    fprintf(stderr, "Render DeviceLost handler\n");

    // test the cooperative level to see if it's okay
    // to render
    if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel())) {
      fprintf(stderr,
              "TestCooperativeLevel = %08x failed, will attempt to reset\n",
              hr);

      // if the device was truly lost, (i.e., a fullscreen device just lost
      // focus), wait
      // until we g_et it back

      if (hr == D3DERR_DEVICELOST) {
        fprintf(
            stderr,
            "TestCooperativeLevel = %08x DeviceLost, will retry next call\n",
            hr);
        return S_OK;
      }

      // eventually, we will g_et this return value,
      // indicating that we can now reset the device
      if (hr == D3DERR_DEVICENOTRESET) {
        fprintf(stderr,
                "TestCooperativeLevel = %08x will try to RESET the device\n",
                hr);
        // if we are windowed, read the desktop mode and use the same format for
        // the back buffer; this effectively turns off color conversion

        if (g_bWindowed) {
          g_pD3D->GetAdapterDisplayModeEx(g_iAdapter, &g_d3ddm, NULL);
          g_d3dpp.BackBufferFormat = g_d3ddm.Format;
        }

        // now try to reset the device
        if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp))) {
          fprintf(stderr, "TestCooperativeLevel = %08x RESET device FAILED\n",
                  hr);
          return hr;
        } else {
          fprintf(stderr, "TestCooperativeLevel = %08x RESET device SUCCESS!\n",
                  hr);

          // Reinitialize D3D9 resources, CUDA resources/contexts
          RestoreContextResources();

          fprintf(stderr, "TestCooperativeLevel = %08x INIT device SUCCESS!\n",
                  hr);

          // we have acquired the device
          g_bDeviceLost = false;
        }
      }
    }
  }

  return hr;
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void) {
  static int seed = 72191;
  char sq[22];

  if (ref_file) {
    seed *= seed;
    sprintf(sq, "%010d", seed);
    // pull the middle 5 digits out of sq
    sq[8] = 0;
    seed = atoi(&sq[3]);

    return seed / 99999.f;
  } else {
    return rand() / (float)RAND_MAX;
  }
}

void initParticles(cData *p, int dx, int dy) {
  int i, j;

  for (i = 0; i < dy; i++) {
    for (j = 0; j < dx; j++) {
      p[i * dx + j].x = (j + 0.5f + (myrand() - 0.5f)) / dx;
      p[i * dx + j].y = (i + 0.5f + (myrand() - 0.5f)) / dy;
    }
  }
}

int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", argv[0]);

  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  // command line options
  // automated build testing harness
  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

  HINSTANCE hInst = GetModuleHandle(NULL);
  // Register the window class
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),    CS_CLASSDC, MsgProc, 0L,   0L,
                   GetModuleHandle(NULL), NULL,       NULL,    NULL, NULL,
                   "fluidsD3D9",          NULL};
  RegisterClassEx(&wc);

  // Create the application's window
  int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
  int yCaption = ::GetSystemMetrics(SM_CYCAPTION);
  int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
  hWnd = CreateWindow("fluidsD3D9", "CUDA/D3D9 Stable Fluids",
                      WS_OVERLAPPEDWINDOW, 100, 100, wWidth + 2 * xBorder,
                      wHeight + 2 * yBorder + yCaption, NULL, NULL,
                      wc.hInstance, NULL);

  if (SUCCEEDED(InitD3D9(hWnd)) && SUCCEEDED(InitCUDA()) &&
      SUCCEEDED(InitD3D9RenderState()) && SUCCEEDED(InitCUFFT()) &&
      SUCCEEDED(InitVertexBuffer()) && SUCCEEDED(InitPointTexture())) {
    ShowWindow(hWnd, SW_SHOWDEFAULT);
    UpdateWindow(hWnd);

    // Rendering loop
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    while (msg.message != WM_QUIT) {
      if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      } else {
        Render();

        if (ref_file) {
          for (int count = 0; count < g_iFrameToCompare; count++) {
            // add in a little force so the automated testing is interesing.
            int x = wWidth / (count + 1);
            int y = wHeight / (count + 1);
            float fx = (x / (float)wWidth);
            float fy = (y / (float)wHeight);
            int nx = (int)(fx * DIM);
            int ny = (int)(fy * DIM);

            int ddx = 35;
            int ddy = 35;
            fx = ddx / (float)wWidth;
            fy = ddy / (float)wHeight;
            int spy = ny - FR;
            int spx = nx - FR;

            addForces(g_dvfield, DIM, DIM, spx, spy, FORCE * DT * fx,
                      FORCE * DT * fy, FR, g_tPitch);
            // g_bQAAddTestForce = false; // only add it once

            Render();
          }

          const char *cur_image_path = "qatest_fluidsD3D9.ppm";

          // Save a reference of our current test run image
          CheckRenderD3D9::BackbufferToPPM(g_pD3DDevice, cur_image_path);

          // compare to official reference image, printing PASS or FAIL.
          g_bPassed = CheckRenderD3D9::PPMvsPPM(cur_image_path, ref_file,
                                                argv[0], MAX_EPSILON, 0.30f);

          PostQuitMessage(0);
        }
      }
    }
  }

  UnregisterClass("fluidsD3D9", wc.hInstance);

  //
  // and exit
  //
  printf("> %s running on %s exiting...\n", SDK_name, device_name);
  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}
