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

/* This example demonstrates how to use the CUDA Direct3D bindings to
 * transfer data between CUDA and DX9 2D, CubeMap, and Volume Textures.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#endif

// This header inclues all the necessary D3D10 and CUDA includes
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>

// includes, project
#include <rendercheck_d3d9.h>
#include <helper_cuda.h>
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

#include <cassert>

#define MAX_EPSILON 10

static char *SDK_name = "simpleD3D9Texture";

bool g_bDone = false;
bool g_bPassed = true;
IDirect3D9Ex *g_pD3D;  // Used to create the D3DDevice
unsigned int g_iAdapter;
IDirect3DDevice9Ex *g_pD3DDevice;

D3DDISPLAYMODEEX g_d3ddm;
D3DPRESENT_PARAMETERS g_d3dpp;

bool g_bWindowed = true;
bool g_bDeviceLost = false;

const unsigned int g_WindowWidth = 720;
const unsigned int g_WindowHeight = 720;

int g_iFrameToCompare = 10;

int *pArgc = NULL;
char **pArgv = NULL;

// Data structure for 2D texture shared between DX9 and CUDA
struct {
  IDirect3DTexture9 *pTexture;
  cudaGraphicsResource *cudaResource;
  void *cudaLinearMemory;
  size_t pitch;
  int width;
  int height;
} g_texture_2d;

// Data structure for cube texture shared between DX9 and CUDA
struct {
  IDirect3DCubeTexture9 *pTexture;
  cudaGraphicsResource *cudaResource;
  void *cudaLinearMemory;
  size_t pitch;
  int size;
} g_texture_cube;

// Data structure for volume textures shared between DX9 and CUDA
struct {
  IDirect3DVolumeTexture9 *pTexture;
  cudaGraphicsResource *cudaResource;
  void *cudaLinearMemory;
  size_t pitch;
  int width;
  int height;
  int depth;
} g_texture_vol;

// The CUDA kernel launchers that get called
extern "C" {
bool cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch,
                     float t);
bool cuda_texture_cube(void *surface, int width, int height, size_t pitch,
                       int face, float t);
bool cuda_texture_volume(void *surface, int width, int height, int depth,
                         size_t pitch, size_t pitchslice, float t);
}

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D9(HWND hWnd);
HRESULT InitCUDA();
HRESULT InitTextures();
HRESULT ReleaseTextures();
HRESULT RegisterD3D9ResourceWithCUDA();
HRESULT DeviceLostHandler();

void RunKernels();
HRESULT DrawScene();
void Cleanup();
void RunCUDA();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#define NAME_LEN 512

char device_name[NAME_LEN];

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

  printf("[%s] - Starting...\n", SDK_name);

  // command line options
  if (argc > 1) {
    // automatied build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
      getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

//
// create window
//
// Register the window class
#if 1
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),          CS_CLASSDC, MsgProc, 0L,   0L,
                   GetModuleHandle(NULL),       NULL,       NULL,    NULL, NULL,
                   "CUDA/D3D9 Texture InterOP", NULL};
  RegisterClassEx(&wc);

  int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
  int yMenu = ::GetSystemMetrics(SM_CYMENU);
  int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);

  // Create the application's window (padding by window border for uniform BB
  // sizes across OSs)
  HWND hWnd = CreateWindow(
      wc.lpszClassName, "CUDA/D3D9 Texture InterOP", WS_OVERLAPPEDWINDOW, 0, 0,
      g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu, NULL,
      NULL, wc.hInstance, NULL);

#else
  static WNDCLASSEX wc = {
      sizeof(WNDCLASSEX),    CS_CLASSDC, MsgProc, 0L,   0L,
      GetModuleHandle(NULL), NULL,       NULL,    NULL, NULL,
      "CudaD3D9Tex",         NULL};
  RegisterClassEx(&wc);
  HWND hWnd = CreateWindow("CudaD3D9Tex", "CUDA D3D9 Texture Interop",
                           WS_OVERLAPPEDWINDOW, 0, 0, 800, 320,
                           GetDesktopWindow(), NULL, wc.hInstance, NULL);
#endif

  ShowWindow(hWnd, SW_SHOWDEFAULT);
  UpdateWindow(hWnd);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D9(hWnd)) && SUCCEEDED(InitCUDA()) &&
      SUCCEEDED(InitTextures())) {
    if (!g_bDeviceLost) {
      RegisterD3D9ResourceWithCUDA();
    }
  } else {
    printf("\n");
    printf("  No CUDA-compatible Direct3D9 device available\n");
    printf("WAIVED\n");
    exit(EXIT_WAIVED);
  }

  //
  // the main loop
  //
  while (false == g_bDone) {
    RunCUDA();
    DrawScene();

    //
    // handle I/O
    //
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    while (msg.message != WM_QUIT) {
      if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      } else {
        RunCUDA();
        DrawScene();

        if (ref_file) {
          for (int count = 0; count < g_iFrameToCompare; count++) {
            RunCUDA();
            DrawScene();
          }

          const char *cur_image_path = "simpleD3D9Texture.ppm";

          // Save a reference of our current test run image
          CheckRenderD3D9::BackbufferToPPM(g_pD3DDevice, cur_image_path);

          // compare to offical reference image, printing PASS or FAIL.
          g_bPassed = CheckRenderD3D9::PPMvsPPM(cur_image_path, ref_file,
                                                argv[0], MAX_EPSILON, 0.15f);

          g_bDone = true;

          Cleanup();
          PostQuitMessage(0);
        }
      }
    }
  };

  // Unregister windows class
  UnregisterClass(wc.lpszClassName, wc.hInstance);

  //
  // and exit
  //
  printf("> %s running on %s exiting...\n", SDK_name, device_name);

  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

//-----------------------------------------------------------------------------
// Name: InitD3D9()
// Desc: Initializes Direct3D9
//-----------------------------------------------------------------------------
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

    cuStatus = cudaD3D9GetDevice(&device, adapterId.DeviceName);
    // This prints and resets the cudaError to cudaSuccess
    printLastCudaError("cudaD3D9GetDevice failed");

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
    printf("\n");
    printf("  No CUDA-compatible Direct3D9 device available\n");
    printf("PASSED\n");
    // destroy the D3D device
    g_pD3D->Release();
    exit(EXIT_SUCCESS);
  }

  // Create the D3D Display Device
  RECT rc;
  GetClientRect(hWnd, &rc);
  D3DDISPLAYMODE d3ddm;
  g_pD3D->GetAdapterDisplayMode(g_iAdapter, &d3ddm);
  D3DPRESENT_PARAMETERS d3dpp;
  ZeroMemory(&d3dpp, sizeof(d3dpp));
  d3dpp.Windowed = TRUE;
  d3dpp.BackBufferCount = 1;
  d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  d3dpp.hDeviceWindow = hWnd;
  // d3dpp.BackBufferWidth = g_bQAReadback?g_WindowWidth:(rc.right - rc.left);
  // d3dpp.BackBufferHeight = g_bQAReadback?g_WindowHeight:(rc.bottom - rc.top);
  d3dpp.BackBufferWidth = g_WindowWidth;
  d3dpp.BackBufferHeight = g_WindowHeight;

  d3dpp.BackBufferFormat = d3ddm.Format;

  if (FAILED(g_pD3D->CreateDeviceEx(g_iAdapter, D3DDEVTYPE_HAL, hWnd,
                                    D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp,
                                    NULL, &g_pD3DDevice))) {
    return E_FAIL;
  }

  // We clear the back buffer
  g_pD3DDevice->BeginScene();
  g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);
  g_pD3DDevice->EndScene();

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

HRESULT RegisterD3D9ResourceWithCUDA() {
  // 2D
  // register the Direct3D resources that we'll use
  // we'll read to and write from g_texture_2d, so don't set any special map
  // flags for it
  cudaGraphicsD3D9RegisterResource(&g_texture_2d.cudaResource,
                                   g_texture_2d.pTexture,
                                   cudaGraphicsRegisterFlagsNone);
  getLastCudaError("cudaGraphicsD3D9RegisterResource (g_texture_2d) failed");
  // cuda cannot write into the texture directly : the texture is seen as a
  // cudaArray and can only be mapped as a texture
  // Create a buffer so that cuda can write into it
  // pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
  cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch,
                  g_texture_2d.width * sizeof(float) * 4, g_texture_2d.height);
  getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
  cudaMemset(g_texture_2d.cudaLinearMemory, 1,
             g_texture_2d.pitch * g_texture_2d.height);

  // CUBE
  cudaGraphicsD3D9RegisterResource(&g_texture_cube.cudaResource,
                                   g_texture_cube.pTexture,
                                   cudaGraphicsRegisterFlagsNone);
  getLastCudaError("cudaGraphicsD3D9RegisterResource (g_texture_cube) failed");
  // create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
  cudaMallocPitch(&g_texture_cube.cudaLinearMemory, &g_texture_cube.pitch,
                  g_texture_cube.size * 4, g_texture_cube.size);
  getLastCudaError("cudaMallocPitch (g_texture_cube) failed");
  cudaMemset(g_texture_cube.cudaLinearMemory, 1,
             g_texture_cube.pitch * g_texture_cube.size);
  getLastCudaError("cudaMemset (g_texture_cube) failed");

  // 3D
  cudaGraphicsD3D9RegisterResource(&g_texture_vol.cudaResource,
                                   g_texture_vol.pTexture,
                                   cudaGraphicsRegisterFlagsNone);
  getLastCudaError("cudaGraphicsD3D9RegisterResource (g_texture_vol) failed");
  // create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
  // cudaMallocPitch(&g_texture_vol.cudaLinearMemory, &g_texture_vol.pitch,
  // g_texture_vol.width * 4, g_texture_vol.height * g_texture_vol.depth);
  cudaMalloc(
      &g_texture_vol.cudaLinearMemory,
      g_texture_vol.width * 4 * g_texture_vol.height * g_texture_vol.depth);
  g_texture_vol.pitch = g_texture_vol.width * 4;
  getLastCudaError("cudaMallocPitch (g_texture_vol) failed");
  cudaMemset(g_texture_vol.cudaLinearMemory, 1,
             g_texture_vol.pitch * g_texture_vol.height * g_texture_vol.depth);
  getLastCudaError("cudaMemset (g_texture_vol) failed");

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: InitTextures()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
HRESULT InitTextures() {
  //
  // create the D3D resources we'll be using
  //

  // 2D texture
  g_texture_2d.width = 256;
  g_texture_2d.height = 256;

  if (FAILED(g_pD3DDevice->CreateTexture(
          g_texture_2d.width, g_texture_2d.height, 1, 0, D3DFMT_A32B32G32R32F,
          D3DPOOL_DEFAULT, &g_texture_2d.pTexture, NULL))) {
    return E_FAIL;
  }

  // cube texture
  g_texture_cube.size = 64;

  if (FAILED(g_pD3DDevice->CreateCubeTexture(g_texture_cube.size, 1, 0,
                                             D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT,
                                             &g_texture_cube.pTexture, NULL))) {
    return E_FAIL;
  }

  // 3D texture
  g_texture_vol.width = 64;
  g_texture_vol.height = 64;
  g_texture_vol.depth = 32;

  if (FAILED(g_pD3DDevice->CreateVolumeTexture(
          g_texture_vol.width, g_texture_vol.height, g_texture_vol.depth, 1, 0,
          D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_texture_vol.pTexture, NULL))) {
    return E_FAIL;
  }

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: ReleaseTextures()
// Desc: Release Direct3D Textures (free-ing)
//-----------------------------------------------------------------------------
HRESULT ReleaseTextures() {
  // unregister the Cuda resources
  cudaGraphicsUnregisterResource(g_texture_2d.cudaResource);
  getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
  cudaFree(g_texture_2d.cudaLinearMemory);
  getLastCudaError("cudaFree (g_texture_2d) failed");

  cudaGraphicsUnregisterResource(g_texture_cube.cudaResource);
  getLastCudaError("cudaGraphicsUnregisterResource (g_texture_cube) failed");
  cudaFree(g_texture_cube.cudaLinearMemory);
  getLastCudaError("cudaFree (g_texture_2d) failed");

  cudaGraphicsUnregisterResource(g_texture_vol.cudaResource);
  getLastCudaError("cudaGraphicsUnregisterResource (g_texture_vol) failed");
  cudaFree(g_texture_vol.cudaLinearMemory);
  getLastCudaError("cudaFree (g_texture_vol) failed");

  //
  // clean up Direct3D
  //
  {
    // release the resources we created
    g_texture_2d.pTexture->Release();
    g_texture_cube.pTexture->Release();
    g_texture_vol.pTexture->Release();
  }

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void RunKernels() {
  static float t = 0.0f;

  // populate the 2d texture
  {
    cudaArray *cuArray;
    cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_2d.cudaResource,
                                          0, 0);
    getLastCudaError(
        "cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

    // kick off the kernel and send the staging buffer cudaLinearMemory as an
    // argument to allow the kernel to write to it
    cuda_texture_2d(g_texture_2d.cudaLinearMemory, g_texture_2d.width,
                    g_texture_2d.height, g_texture_2d.pitch, t);
    getLastCudaError("cuda_texture_2d failed");

    // then we want to copy cudaLinearMemory to the D3D texture, via its mapped
    // form : cudaArray
    cudaMemcpy2DToArray(
        cuArray,                                            // dst array
        0, 0,                                               // offset
        g_texture_2d.cudaLinearMemory, g_texture_2d.pitch,  // src
        g_texture_2d.width * 4 * sizeof(float), g_texture_2d.height,  // extent
        cudaMemcpyDeviceToDevice);                                    // kind
    getLastCudaError("cudaMemcpy2DToArray failed");
  }
  // populate the volume texture
  {
    size_t pitchSlice = g_texture_vol.pitch * g_texture_vol.height;
    cudaArray *cuArray;
    cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_vol.cudaResource,
                                          0, 0);
    getLastCudaError(
        "cudaGraphicsSubResourceGetMappedArray (cuda_texture_3d) failed");

    // kick off the kernel and send the staging buffer cudaLinearMemory as an
    // argument to allow the kernel to write to it
    cuda_texture_volume(g_texture_vol.cudaLinearMemory, g_texture_vol.width,
                        g_texture_vol.height, g_texture_vol.depth,
                        g_texture_vol.pitch, pitchSlice, t);
    getLastCudaError("cuda_texture_3d failed");

    // then we want to copy cudaLinearMemory to the D3D texture, via its mapped
    // form : cudaArray
    struct cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.dstArray = cuArray;
    memcpyParams.srcPtr.ptr = g_texture_vol.cudaLinearMemory;
    memcpyParams.srcPtr.pitch = g_texture_vol.pitch;
    memcpyParams.srcPtr.xsize = g_texture_vol.width;
    memcpyParams.srcPtr.ysize = g_texture_vol.height;
    memcpyParams.extent.width = g_texture_vol.width;
    memcpyParams.extent.height = g_texture_vol.height;
    memcpyParams.extent.depth = g_texture_vol.depth;
    memcpyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&memcpyParams);
    getLastCudaError("cudaMemcpy3D failed");
  }

  // populate the faces of the cube map
  for (int face = 0; face < 6; ++face) {
    cudaArray *cuArray;
    cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_cube.cudaResource,
                                          face, 0);
    getLastCudaError(
        "cudaGraphicsSubResourceGetMappedArray (cuda_texture_cube) failed");

    // kick off the kernel and send the staging buffer cudaLinearMemory as an
    // argument to allow the kernel to write to it
    cuda_texture_cube(g_texture_cube.cudaLinearMemory, g_texture_cube.size,
                      g_texture_cube.size, g_texture_cube.pitch, face, t);
    getLastCudaError("cuda_texture_cube failed");

    // then we want to copy cudaLinearMemory to the D3D texture, via its mapped
    // form : cudaArray
    cudaMemcpy2DToArray(cuArray,  // dst array
                        0, 0,     // offset
                        g_texture_cube.cudaLinearMemory,
                        g_texture_cube.pitch,                          // src
                        g_texture_cube.size * 4, g_texture_cube.size,  // extent
                        cudaMemcpyDeviceToDevice);                     // kind
    getLastCudaError("cudaMemcpy2DToArray failed");
  }

  t += 0.1f;
}
/*{
    static float t = 0.0f;

    // populate the 2d texture
    {
        void* pData;
        size_t pitch;
        checkCudaErrorsNoSync ( cudaD3D9ResourceGetMappedPointer(&pData,
g_texture_2d.pTexture, 0, 0) );
        checkCudaErrorsNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL,
g_texture_2d.pTexture, 0, 0) );
        cuda_texture_2d(pData, g_texture_2d.width, g_texture_2d.height, pitch,
t);
    }

    // populate the faces of the cube map
    for (int face = 0; face < 6; ++face)
    {
        void* pData;
        size_t pitch;
        checkCudaErrorsNoSync ( cudaD3D9ResourceGetMappedPointer(&pData,
g_texture_cube.pTexture, face, 0) );
        checkCudaErrorsNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL,
g_texture_cube.pTexture, face, 0) );
        cuda_texture_cube(pData, g_texture_cube.size, g_texture_cube.size,
pitch, face, t);
    }

    // populate the volume texture
    {
        void* pData;
        size_t pitch;
        size_t pitchSlice;
        checkCudaErrorsNoSync ( cudaD3D9ResourceGetMappedPointer(&pData,
g_texture_vol.pTexture, 0, 0) );
        checkCudaErrorsNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch,
&pitchSlice, g_texture_vol.pTexture, 0, 0) );
        cuda_texture_volume(pData, g_texture_vol.width, g_texture_vol.height,
g_texture_vol.depth, pitch, pitchSlice);
    }

    t += 0.1f;
}*/

////////////////////////////////////////////////////////////////////////////////
//! RestoreContextResources
//    - this function restores all of the CUDA/D3D resources and contexts
////////////////////////////////////////////////////////////////////////////////
HRESULT RestoreContextResources() {
  // Reinitialize D3D9 resources, CUDA resources/contexts
  InitCUDA();
  InitTextures();
  RegisterD3D9ResourceWithCUDA();

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! DeviceLostHandler
//    - this function handles reseting and initialization of the D3D device
//      in the event this Device gets Lost
////////////////////////////////////////////////////////////////////////////////
HRESULT DeviceLostHandler() {
  HRESULT hr = S_OK;

  fprintf(stderr, "-> Starting DeviceLostHandler() \n");

  // test the cooperative level to see if it's okay
  // to render
  if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel())) {
    fprintf(stderr,
            "TestCooperativeLevel = %08x failed, will attempt to reset\n", hr);

    // if the device was truly lost, (i.e., a fullscreen device just lost
    // focus), wait
    // until we g_et it back

    if (hr == D3DERR_DEVICELOST) {
      fprintf(stderr,
              "TestCooperativeLevel = %08x DeviceLost, will retry next call\n",
              hr);
      return S_OK;
    }

    // eventually, we will g_et this return value,
    // indicating that we can now reset the device
    if (hr == D3DERR_DEVICENOTRESET) {
      fprintf(stderr,
              "TestCooperativeLevel = %08x will try to RESET the device\n", hr);
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

        // This is a common function we use to restore all hardware
        // resources/state
        RestoreContextResources();

        fprintf(stderr, "TestCooperativeLevel = %08x INIT device SUCCESS!\n",
                hr);

        // we have acquired the device
        g_bDeviceLost = false;
      }
    }
  }

  return hr;
}

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
HRESULT DrawScene() {
  HRESULT hr = S_OK;

  if (g_bDeviceLost) {
    if (FAILED(hr = DeviceLostHandler())) {
      fprintf(stderr, "DeviceLostHandler FAILED returned %08x\n", hr);
      return hr;
    }
  }

  if (!g_bDeviceLost) {
    //
    // we will use this index and vertex data throughout
    //
    unsigned int IB[6] = {
        0, 1, 2, 0, 2, 3,
    };
    struct VertexStruct {
      float position[3];
      float texture[3];
    };

    //
    // initialize the scene
    //
    D3DVIEWPORT9 viewport_window = {0, 0, 672, 192, 0, 1};
    g_pD3DDevice->SetViewport(&viewport_window);
    g_pD3DDevice->BeginScene();
    g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);
    g_pD3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    g_pD3DDevice->SetRenderState(D3DRS_LIGHTING, FALSE);
    g_pD3DDevice->SetFVF(D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE3(0));

    //
    // draw the 2d texture
    //
    VertexStruct VB[4] = {
      {  {-1,-1,0,}, {0,0,0,},  },
      {  { 1,-1,0,}, {1,0,0,},  },
      {  { 1, 1,0,}, {1,1,0,},  },
      {  {-1, 1,0,}, {0,1,0,},  },
    };
    D3DVIEWPORT9 viewport = {32, 32, 256, 256, 0, 1};
    g_pD3DDevice->SetViewport(&viewport);
    g_pD3DDevice->SetTexture(0, g_texture_2d.pTexture);
    g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB,
                                         D3DFMT_INDEX32, VB,
                                         sizeof(VertexStruct));

    //
    // draw the Z-positive side of the cube texture
    //
    VertexStruct VB_Zpos[4] = {
      {  {-1,-1,0,}, {-1,-1, 0.5f,},  },
      {  { 1,-1,0,}, { 1,-1, 0.5f,},  },
      {  { 1, 1,0,}, { 1, 1, 0.5f,},  },
      {  {-1, 1,0,}, {-1, 1, 0.5f,},  },
    };
    viewport.Y += viewport.Height + 32;
    g_pD3DDevice->SetViewport(&viewport);
    g_pD3DDevice->SetTexture(0, g_texture_cube.pTexture);
    g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB,
                                         D3DFMT_INDEX32, VB_Zpos,
                                         sizeof(VertexStruct));

    //
    // draw the Z-negative side of the cube texture
    //
    VertexStruct VB_Zneg[4] = {
      {  {-1,-1,0,}, { 1,-1,-0.5f,},  },
      {  { 1,-1,0,}, {-1,-1,-0.5f,},  },
      {  { 1, 1,0,}, {-1, 1,-0.5f,},  },
      {  {-1, 1,0,}, { 1, 1,-0.5f,},  },
    };
    viewport.X += viewport.Width + 32;
    g_pD3DDevice->SetViewport(&viewport);
    g_pD3DDevice->SetTexture(0, g_texture_cube.pTexture);
    g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB,
                                         D3DFMT_INDEX32, VB_Zneg,
                                         sizeof(VertexStruct));

    //
    // draw a slice the volume texture
    //
    VertexStruct VB_Zslice[4] = {
      {  {-1,-1,0,}, {0,0,0,},  },
      {  { 1,-1,0,}, {1,0,0,},  },
      {  { 1, 1,0,}, {1,1,1,},  },
      {  {-1, 1,0,}, {0,1,1,},  },
    };
    viewport.Y -= viewport.Height + 32;
    g_pD3DDevice->SetViewport(&viewport);
    g_pD3DDevice->SetTexture(0, g_texture_vol.pTexture);
    g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB,
                                         D3DFMT_INDEX32, VB_Zslice,
                                         sizeof(VertexStruct));

    //
    // end the scene
    //
    g_pD3DDevice->EndScene();
    hr = g_pD3DDevice->Present(NULL, NULL, NULL, NULL);

    if (hr == D3DERR_DEVICELOST) {
      fprintf(stderr, "DrawScene Present = %08x detected D3D DeviceLost\n", hr);
      g_bDeviceLost = true;

      ReleaseTextures();
    }
  }

  return hr;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup() {
  ReleaseTextures();

  {
    // destroy the D3D device
    g_pD3DDevice->Release();
    g_pD3D->Release();
  }
}

//-----------------------------------------------------------------------------
// Name: RunCUDA()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void RunCUDA() {
  //
  // map the resources we've registered so we can access them in Cuda
  // - it is most efficient to map and unmap all resources in a single call,
  //   and to have the map/unmap calls be the boundary between using the GPU
  //   for Direct3D and Cuda
  //

  if (!g_bDeviceLost) {
    cudaStream_t stream = 0;
    const int nbResources = 3;
    cudaGraphicsResource *ppResources[nbResources] = {
        g_texture_2d.cudaResource, g_texture_vol.cudaResource,
        g_texture_cube.cudaResource,
    };
    cudaGraphicsMapResources(nbResources, ppResources, stream);
    getLastCudaError("cudaGraphicsMapResources(3) failed");

    //
    // run kernels which will populate the contents of those textures
    //
    RunKernels();

    //
    // unmap the resources
    //
    cudaGraphicsUnmapResources(nbResources, ppResources, stream);
    getLastCudaError("cudaGraphicsUnmapResources(3) failed");
  }
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam,
                              LPARAM lParam) {
  switch (msg) {
    case WM_KEYDOWN:
      if (wParam == VK_ESCAPE) {
        g_bDone = true;
        Cleanup();
        PostQuitMessage(0);
        return 0;
      }

      break;

    case WM_DESTROY:
      g_bDone = true;
      Cleanup();
      PostQuitMessage(0);
      return 0;

    case WM_PAINT:
      ValidateRect(hWnd, NULL);
      return 0;
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
