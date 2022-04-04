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

//
// This example demonstrates interoperability of SLI with a Direct3D10 texture
// and CUDA. The program creates a D3D10 texture which is written from
// a CUDA kernel. Direct3D then renders the result on the screen.
// A Direct3D Capable device is required.
//

#pragma warning(disable : 4312)

#include <windows.h>
#include <mmsystem.h>

#pragma warning(disable : 4996)  // disable deprecated warning
#include <strsafe.h>
#pragma warning(default : 4996)

// this header includes all the necessary D3D10 includes
#include <dynlink_d3d10.h>

// includes, cuda
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime.h>
#include <cuda_d3d10_interop.h>
#include <d3d10.h>

// includes, project
#include <rendercheck_d3d10.h>  // automated testing
#include <helper_cuda.h>  // helper for CUDA error checking and initialization

#define MAX_EPSILON 10

static char *SDK_name = "SLID3D10Texture";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D10Device *g_pd3dDevice = NULL;           // Our rendering device
IDXGISwapChain *g_pSwapChain = NULL;         // The swap chain of the window
ID3D10RenderTargetView *g_pSwapChainRTV =
    NULL;  // The Render target view on the swap chain ( used for clear)
ID3D10RasterizerState *g_pRasterState = NULL;

ID3D10InputLayout *g_pInputLayout = NULL;
ID3D10Effect *g_pSimpleEffect = NULL;
ID3D10EffectTechnique *g_pSimpleTechnique = NULL;
ID3D10EffectVectorVariable *g_pvQuadRect = NULL;
ID3D10EffectShaderResourceVariable *g_pTexture2D = NULL;

static const char g_simpleEffectSrc[] =
    "float4 g_vQuadRect; \n"
    "Texture2D g_Texture2D; \n"
    "\n"
    "SamplerState samLinear{ \n"
    "    Filter = MIN_MAG_LINEAR_MIP_POINT; \n"
    "};\n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float3 Tex : TEXCOORD0; };\n"
    "\n"
    "Fragment VS( uint vertexId : SV_VertexID )\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Tex = float3( 0.f, 0.f, 0.f); \n"
    "    if (vertexId == 1) f.Tex.x = 1.f; \n"
    "    else if (vertexId == 2) f.Tex.y = 1.f; \n"
    "    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f); \n"
    "    \n"
    "    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n"
    "    \n"
    "    return f;\n"
    "}\n"
    "\n"
    "float4 PS( Fragment f ) : SV_Target\n"
    "{\n"
    "    // return g_Texture2D.Sample( samLinear, f.Tex.xy ); \n"
    "    // return float4(f.Tex, 1);\n"
    "    float4 g = g_Texture2D.Sample( samLinear, f.Tex.xy );"
    "    for (int i = 0; i < 1024; ++i) { "
    "        g.x = sqrt(g.x);"
    "        g.x += 0.0001;"
    "        g.x = g.x * g.x;"
    "    }"
    "    return g;\n"
    "}\n"
    "\n"
    "technique10 Render\n"
    "{\n"
    "    pass P0\n"
    "    {\n"
    "        SetVertexShader( CompileShader( vs_4_0, VS() ) );\n"
    "        SetGeometryShader( NULL );\n"
    "        SetPixelShader( CompileShader( ps_4_0, PS() ) );\n"
    "    }\n"
    "}\n"
    "\n";

// testing/tracing function used pervasively in tests. If the condition is
// unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x)                                                  \
  if (!(x)) {                                                            \
    fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, \
            __FILE__, __LINE__);                                         \
    return 1;                                                            \
  }

bool g_bDone = false;
bool g_bPassed = true;

int *pArgc = NULL;
char **pArgv = NULL;

const unsigned int g_WindowWidth = 720;
const unsigned int g_WindowHeight = 720;

bool g_bQAReadback = false;
int g_iFrameToCompare = 1;

struct CudaContextData {
  UINT index;
  CUcontext context;
  int deviceOrdinal;
  cudaGraphicsResource *cudaResource;
  void *cudaLinearMemory;
};

UINT g_ContextCount = 0;
CudaContextData g_ContextData[32];

// Data structure for 2D texture shared between DX10 and CUDA
struct {
  ID3D10Texture2D *pTexture;
  ID3D10ShaderResourceView *pSRView;
  size_t pitch;
  int width;
  int height;
} g_texture_2d;

// The CUDA kernel launchers that get called
extern "C" {
bool cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch,
                     float t);
}

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);
HRESULT InitTextures();

int RunKernels(CudaContextData *currentContextData);
void DrawScene();
int Cleanup();
int Render();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#define NAME_LEN 512

bool findCUDADevice() {
  int nGraphicsGPU = 0;
  int deviceCount = 0;
  bool bFoundGraphics = false;
  char devname[NAME_LEN];

  // This function call returns 0 if there are no CUDA capable devices.
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0) {
    printf("> There are no device(s) supporting CUDA\n");
    return false;
  } else {
    printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
  }

  // Get CUDA device properties
  cudaDeviceProp deviceProp;

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaGetDeviceProperties(&deviceProp, dev);
    STRCPY(devname, NAME_LEN, deviceProp.name);
    printf("> GPU %d: %s\n", dev, devname);
  }

  return true;
}

bool findDXDevice(char *dev_name) {
  HRESULT hr = S_OK;
  cudaError cuStatus;

  // Iterate through the candidate adapters
  IDXGIFactory *pFactory;
  hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory), (void **)(&pFactory));

  if (!SUCCEEDED(hr)) {
    printf("> No DXGI Factory created.\n");
    return false;
  }

  UINT adapter = 0;

  for (; !g_pCudaCapableAdapter; ++adapter) {
    // Get a candidate DXGI adapter
    IDXGIAdapter *pAdapter = NULL;
    hr = pFactory->EnumAdapters(adapter, &pAdapter);

    if (FAILED(hr)) {
      break;  // no compatible adapters found
    }

    // Query to see if there exists a corresponding compute device
    int cuDevice;
    cuStatus = cudaD3D10GetDevice(&cuDevice, pAdapter);
    printLastCudaError("cudaD3D10GetDevice failed");  // This prints and resets
                                                      // the cudaError to
                                                      // cudaSuccess

    if (cudaSuccess == cuStatus) {
      // If so, mark it as the one against which to create our d3d10 device
      g_pCudaCapableAdapter = pAdapter;
      g_pCudaCapableAdapter->AddRef();
    }

    pAdapter->Release();
  }

  printf("> Found %d D3D10 Adapater(s).\n", (int)adapter);

  pFactory->Release();

  if (!g_pCudaCapableAdapter) {
    printf("> Found 0 D3D10 Adapater(s) /w Compute capability.\n");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  g_pCudaCapableAdapter->GetDesc(&adapterDesc);
  wcstombs_s(NULL, dev_name, 256, adapterDesc.Description, 128);

  printf("> Found 1 D3D10 Adapater(s) /w Compute capability.\n");
  printf("> %s\n", dev_name);

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
  char device_name[256];
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", SDK_name);

  if (!findCUDADevice())  // Search for CUDA GPU
  {
    printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
    exit(EXIT_SUCCESS);
  }

  if (!dynlinkLoadD3D10API())  // Search for D3D API (locate drivers, does not
                               // mean device is found)
  {
    printf("> D3D10 API libraries NOT found on.. Exiting.\n");
    dynlinkUnloadD3D10API();
    exit(EXIT_SUCCESS);
  }

  if (!findDXDevice(device_name))  // Search for D3D Hardware Device
  {
    printf("> D3D10 Graphics Device NOT found.. Exiting.\n");
    dynlinkUnloadD3D10API();
    exit(EXIT_SUCCESS);
  }

  // command line options
  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    g_bQAReadback = true;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

  //
  // create window
  //
  // Register the window class
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),
                   CS_CLASSDC,
                   MsgProc,
                   0L,
                   0L,
                   GetModuleHandle(NULL),
                   NULL,
                   NULL,
                   NULL,
                   NULL,
                   "CUDA SDK",
                   NULL};
  RegisterClassEx(&wc);

  // Create the application's window
  HWND hWnd = CreateWindow(wc.lpszClassName, "CUDA-SLI Interop, D3D10",
                           WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth,
                           g_WindowHeight, NULL, NULL, wc.hInstance, NULL);

  ShowWindow(hWnd, SW_SHOWDEFAULT);
  UpdateWindow(hWnd);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D(hWnd)) && SUCCEEDED(InitTextures())) {
    CUresult result = CUDA_SUCCESS;
    cudaError_t error = cudaSuccess;

    // get the list of interop devices
    {
      unsigned int interopDeviceCount = 0;
      int interopDevices[32];
      error = cudaD3D10GetDevices(&interopDeviceCount, interopDevices, 32,
                                  g_pd3dDevice, cudaD3D10DeviceListAll);
      printLastCudaError("cudaD3D10GetDevices failed");  // This prints and
                                                         // resets the cudaError
                                                         // to cudaSuccess
      AssertOrQuit(cudaSuccess == error);

      g_ContextCount = interopDeviceCount;

      for (UINT i = 0; i < interopDeviceCount; ++i) {
        g_ContextData[i].index = i;
        g_ContextData[i].deviceOrdinal = interopDevices[i];
      }
    }

    // Initialize g_ContextCount interop contexts on the device,
    // striping across AFR groups
    for (UINT i = 0; i < g_ContextCount; ++i) {
      printf("Creating context %d on device %d\n", g_ContextData[i].index,
             g_ContextData[i].deviceOrdinal);

      // create a context
      error = cudaD3D10SetDirect3DDevice(g_pd3dDevice,
                                         g_ContextData[i].deviceOrdinal);
      AssertOrQuit(cudaSuccess == error);
      error = cudaFree(0);
      AssertOrQuit(cudaSuccess == error);

      // allocate a buffer
      // error = cudaMalloc((void**)&g_ContextData[i].buffer, BYTES_PER_PIXEL);
      cudaMallocPitch(&g_ContextData[i].cudaLinearMemory, &g_texture_2d.pitch,
                      g_texture_2d.width * sizeof(float) * 4,
                      g_texture_2d.height);
      getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
      cudaMemset(g_ContextData[i].cudaLinearMemory, 1,
                 g_texture_2d.pitch * g_texture_2d.height);

      AssertOrQuit(cudaSuccess == error);

      // pop the context
      result = cuCtxPopCurrent(&g_ContextData[i].context);
      AssertOrQuit(CUDA_SUCCESS == result);
    }

    // Register the texture with all contexts
    for (UINT i = 0; i < g_ContextCount; ++i) {
      printf("Registering texture with context %d\n", i);
      result = cuCtxPushCurrent(g_ContextData[i].context);
      AssertOrQuit(CUDA_SUCCESS == result);
      {
        // Register the resource
        error = cudaGraphicsD3D10RegisterResource(
            &g_ContextData[i].cudaResource, g_texture_2d.pTexture,
            cudaGraphicsRegisterFlagsNone);
        getLastCudaError(
            "cudaGraphicsD3D10RegisterResource (g_texture_2d) failed");

        error = cudaGraphicsResourceSetMapFlags(g_ContextData[i].cudaResource,
                                                cudaD3D10MapFlagsWriteDiscard);
        getLastCudaError(
            "cudaGraphicsResourceSetMapFlags (g_texture_2d) failed");

        AssertOrQuit(cudaSuccess == error);
      }
      result = cuCtxPopCurrent(&g_ContextData[i].context);
      AssertOrQuit(CUDA_SUCCESS == result);
    }
  }

  //
  // the main loop
  //
  while (false == g_bDone) {
    Render();

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
        Render();

        if (ref_file) {
          for (int count = 0; count < g_iFrameToCompare; count++) {
            Render();
          }

          const char *cur_image_path = "SLID3D10Texture.ppm";

          // Save a reference of our current test run image
          CheckRenderD3D10::ActiveRenderTargetToPPM(g_pd3dDevice,
                                                    cur_image_path);

          // compare to offical reference image, printing PASS or FAIL.
          g_bPassed = CheckRenderD3D10::PPMvsPPM(cur_image_path, ref_file,
                                                 argv[0], MAX_EPSILON, 0.15f);

          g_bDone = true;

          Cleanup();

          PostQuitMessage(0);
        } else {
          g_bPassed = true;
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

  printf(g_bPassed ? "Test images compared OK\n"
                   : "Test images are Different!\n");
}

//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd) {
  // Set up the structure used to create the device and swapchain
  DXGI_SWAP_CHAIN_DESC sd;
  ZeroMemory(&sd, sizeof(sd));
  sd.BufferCount = 1;
  sd.BufferDesc.Width = g_WindowWidth;
  sd.BufferDesc.Height = g_WindowHeight;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferDesc.RefreshRate.Numerator = 60;
  sd.BufferDesc.RefreshRate.Denominator = 1;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd.OutputWindow = hWnd;
  sd.SampleDesc.Count = 1;
  sd.SampleDesc.Quality = 0;
  sd.Windowed = TRUE;

  // Create device and swapchain
  HRESULT hr = sFnPtr_D3D10CreateDeviceAndSwapChain(
      g_pCudaCapableAdapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0,
      D3D10_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice);
  AssertOrQuit(SUCCEEDED(hr));
  g_pCudaCapableAdapter->Release();

  // Create a render target view of the swapchain
  ID3D10Texture2D *pBuffer;
  hr =
      g_pSwapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID *)&pBuffer);
  AssertOrQuit(SUCCEEDED(hr));

  hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
  AssertOrQuit(SUCCEEDED(hr));
  pBuffer->Release();

  g_pd3dDevice->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

  // Setup the viewport
  D3D10_VIEWPORT vp;
  vp.Width = g_WindowWidth;
  vp.Height = g_WindowHeight;
  vp.MinDepth = 0.0f;
  vp.MaxDepth = 1.0f;
  vp.TopLeftX = 0;
  vp.TopLeftY = 0;
  g_pd3dDevice->RSSetViewports(1, &vp);

  // Setup the effect
  {
    ID3D10Blob *pCompiledEffect;
    ID3D10Blob *pErrors = NULL;
    hr = sFnPtr_D3D10CompileEffectFromMemory((void *)g_simpleEffectSrc,
                                             sizeof(g_simpleEffectSrc), NULL,
                                             NULL,  // pDefines
                                             NULL,  // pIncludes
                                             0,     // HLSL flags
                                             0,     // FXFlags
                                             &pCompiledEffect, &pErrors);

    if (pErrors) {
      LPVOID l_pError = NULL;
      l_pError = pErrors->GetBufferPointer();  // then cast to a char* to see it
                                               // in the locals window
      fprintf(stdout, "Compilation error: \n %s", (char *)l_pError);
    }

    AssertOrQuit(SUCCEEDED(hr));

    hr = sFnPtr_D3D10CreateEffectFromMemory(
        pCompiledEffect->GetBufferPointer(), pCompiledEffect->GetBufferSize(),
        0,  // FXFlags
        g_pd3dDevice, NULL, &g_pSimpleEffect);
    pCompiledEffect->Release();

    g_pSimpleTechnique = g_pSimpleEffect->GetTechniqueByName("Render");

    g_pvQuadRect =
        g_pSimpleEffect->GetVariableByName("g_vQuadRect")->AsVector();

    g_pTexture2D =
        g_pSimpleEffect->GetVariableByName("g_Texture2D")->AsShaderResource();

    // Setup  no Input Layout
    g_pd3dDevice->IASetInputLayout(0);
    g_pd3dDevice->IASetPrimitiveTopology(
        D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
  }

  D3D10_RASTERIZER_DESC rasterizerState;
  rasterizerState.FillMode = D3D10_FILL_SOLID;
  rasterizerState.CullMode = D3D10_CULL_FRONT;
  rasterizerState.FrontCounterClockwise = false;
  rasterizerState.DepthBias = false;
  rasterizerState.DepthBiasClamp = 0;
  rasterizerState.SlopeScaledDepthBias = 0;
  rasterizerState.DepthClipEnable = false;
  rasterizerState.ScissorEnable = false;
  rasterizerState.MultisampleEnable = false;
  rasterizerState.AntialiasedLineEnable = false;
  g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
  g_pd3dDevice->RSSetState(g_pRasterState);

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
  {
    g_texture_2d.width = 768;
    g_texture_2d.height = 768;

    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D10_TEXTURE2D_DESC));
    desc.Width = g_texture_2d.width;
    desc.Height = g_texture_2d.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;

    if (FAILED(
            g_pd3dDevice->CreateTexture2D(&desc, NULL, &g_texture_2d.pTexture)))
      return E_FAIL;

    if (FAILED(g_pd3dDevice->CreateShaderResourceView(
            g_texture_2d.pTexture, NULL, &g_texture_2d.pSRView)))
      return E_FAIL;

    g_pTexture2D->SetResource(g_texture_2d.pSRView);
  }

  return S_OK;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
int RunKernels(CudaContextData *currentContextData) {
  static float t = 0.0f;

  // populate the 2d texture
  {
    cudaArray *cuArray;
    cudaGraphicsSubResourceGetMappedArray(
        &cuArray, currentContextData->cudaResource, 0, 0);
    getLastCudaError(
        "cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

    // kick off the kernel and send the staging buffer cudaLinearMemory as an
    // argument to allow the kernel to write to it
    cuda_texture_2d(currentContextData->cudaLinearMemory, g_texture_2d.width,
                    g_texture_2d.height, g_texture_2d.pitch,
                    g_bQAReadback ? 0.2f : t);
    getLastCudaError("cuda_texture_2d failed");

    // then we want to copy cudaLinearMemory to the D3D texture, via its mapped
    // form : cudaArray
    cudaMemcpy2DToArray(
        cuArray,                                                   // dst array
        0, 0,                                                      // offset
        currentContextData->cudaLinearMemory, g_texture_2d.pitch,  // src
        g_texture_2d.width * 4 * sizeof(float), g_texture_2d.height,  // extent
        cudaMemcpyDeviceToDevice);                                    // kind
    getLastCudaError("cudaMemcpy2DToArray failed");
  }
  t += 0.1f;

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
void DrawScene() {
  // Clear the backbuffer to a black color
  float ClearColor[4] = {0.5f, 0.5f, 0.6f, 1.0f};
  g_pd3dDevice->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

  //
  // draw the 2d texture
  //
  float quadRect[4] = {-0.98f, -0.98f, 1.96f, 1.96f};
  g_pvQuadRect->SetFloatVector((float *)&quadRect);
  g_pSimpleTechnique->GetPassByIndex(0)->Apply(0);
  g_pd3dDevice->Draw(4, 0);

  // Present the backbuffer contents to the display
  g_pSwapChain->Present(0, 0);
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
int Cleanup() {
  // unregister the Cuda resources
  CUresult result = CUDA_SUCCESS;
  cudaError_t error = cudaSuccess;

  // Drop the D3D resources' refcounts
  // Unregister the texture with all contexts
  for (UINT i = 0; i < g_ContextCount; ++i) {
    printf("Unregistering texture with context %d\n", i);
    result = cuCtxPushCurrent(g_ContextData[i].context);
    AssertOrQuit(CUDA_SUCCESS == result);
    {
      // Register the resource
      error = cudaGraphicsUnregisterResource(g_ContextData[i].cudaResource);
      AssertOrQuit(cudaSuccess == error);
    }
    result = cuCtxPopCurrent(&g_ContextData[i].context);
    AssertOrQuit(CUDA_SUCCESS == result);
  }

  // Destroy all contexts
  for (UINT i = 0; i < g_ContextCount; ++i) {
    printf("Destroying context %d\n", i);
    result = cuCtxPushCurrent(g_ContextData[i].context);
    AssertOrQuit(CUDA_SUCCESS == result);
  }

  //
  // clean up Direct3D
  //
  {
    // release the resources we created
    g_texture_2d.pSRView->Release();
    g_texture_2d.pTexture->Release();

    if (g_pInputLayout != NULL) g_pInputLayout->Release();

    if (g_pSimpleEffect != NULL) g_pSimpleEffect->Release();

    if (g_pSwapChainRTV != NULL) g_pSwapChainRTV->Release();

    if (g_pSwapChain != NULL) g_pSwapChain->Release();

    if (g_pd3dDevice != NULL) g_pd3dDevice->Release();
  }

  return 0;
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
int Render() {
  //
  // map the resources we've registered so we can access them in Cuda
  // - it is most efficient to map and unmap all resources in a single call,
  //   and to have the map/unmap calls be the boundary between using the GPU
  //   for Direct3D and Cuda
  //
  {
    cudaStream_t stream = 0;

    cudaError_t error = cudaSuccess;
    CudaContextData *currentContextData = NULL;

    // get the current device ordinal
    static int currentDevice = -1;
    error = cudaD3D10GetDevices(NULL, &currentDevice, 1, g_pd3dDevice,
                                cudaD3D10DeviceListCurrentFrame);
    printLastCudaError("cudaD3D10GetDevices failed");  // This prints and resets
                                                       // the cudaError to
                                                       // cudaSuccess
    AssertOrQuit(cudaSuccess == error);

    static int nextDevice = -1;
    // assert that querying the next device in AFR isn't broken
    AssertOrQuit(nextDevice == -1 || nextDevice == currentDevice);
    error = cudaD3D10GetDevices(NULL, &nextDevice, 1, g_pd3dDevice,
                                cudaD3D10DeviceListNextFrame);
    printLastCudaError("cudaD3D10GetDevices failed");  // This prints and resets
                                                       // the cudaError to
                                                       // cudaSuccess
    AssertOrQuit(cudaSuccess == error);

    // choose context data corresponding to the current device ordinal
    for (UINT i = 0; i < g_ContextCount; ++i) {
      if (currentDevice == g_ContextData[i].deviceOrdinal) {
        currentContextData = &g_ContextData[i];
      }
    }

    AssertOrQuit(currentContextData);

    CUresult result;
    result = cuCtxPushCurrent(currentContextData->context);
    AssertOrQuit(CUDA_SUCCESS == result);

    cudaGraphicsMapResources(1, &currentContextData->cudaResource, stream);
    getLastCudaError("cudaGraphicsMapResources(3) failed");

    //
    // run kernels which will populate the contents of those textures
    //
    RunKernels(currentContextData);

    //
    // unmap the resources
    //
    cudaGraphicsUnmapResources(1, &currentContextData->cudaResource, stream);
    getLastCudaError("cudaGraphicsUnmapResources(3) failed");

    CUcontext poppedContext;
    result = cuCtxPopCurrent(&poppedContext);
    AssertOrQuit(CUDA_SUCCESS == result);
    AssertOrQuit(poppedContext == currentContextData->context);
  }

  //
  // draw the scene using them
  //
  DrawScene();

  return 0;
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
