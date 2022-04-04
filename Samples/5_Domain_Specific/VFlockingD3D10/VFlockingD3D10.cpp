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

/* This sample program models formation of V-shaped flocks by big birds,
 * such as geese and cranes, as an example of simple AI. It demonstrates
 * that the CUDA-based implementation is much faster than a CPU-based one.
 */

#pragma warning(disable : 4312)

#include <windows.h>
#include <mmsystem.h>

#pragma warning(disable : 4996)  // disable deprecated warning
#include <strsafe.h>
#pragma warning(default : 4996)

#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>

// This header includes all the necessary D3D10 includes
#include <dynlink_d3d10.h>
#include <cuda_runtime.h>
#include <cuda_d3d10_interop.h>

// includes, project
#include <rendercheck_d3d10.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "VFlockingD3D10.h"

#define MAX_EPSILON 10

static char *SDK_name = "VFlockingD3D10";

bool g_bPassed = true;
int g_iFrameToCompare = 1300;

int *pArgc = NULL;
char **pArgv = NULL;

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D10Device *g_pd3dDevice = NULL;           // Our rendering device
IDXGISwapChain *g_pSwapChain = NULL;         // The swap chain of the window
ID3D10RenderTargetView *g_pSwapChainRTV =
    NULL;  // The Render target view on the swap chain ( used for clear)
ID3D10RasterizerState *g_pRasterState = NULL;

ID3D10Buffer *g_pPositions = NULL;
cudaGraphicsResource *g_pCudaResourcePos;
ID3D10Buffer *g_pNewPositions = NULL;
cudaGraphicsResource *g_pCudaResourceNewPos;
ID3D10ShaderResourceView *g_pPositionsSRV = NULL;

ID3D10InputLayout *g_pInputLayout = NULL;
ID3D10Effect *g_pSimpleEffect = NULL;
ID3D10EffectTechnique *g_pDrawQuadTechnique = NULL;
ID3D10EffectTechnique *g_pDrawBirdsTechnique = NULL;
ID3D10EffectVectorVariable *g_pvQuadRect = NULL;
ID3D10EffectShaderResourceVariable *g_pTexture2D = NULL;

static const char g_simpleEffectSrc[] =
    "Buffer<float2> g_BirdsPositions   : register(t0); \n"
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
    "    return g_Texture2D.Sample( samLinear, f.Tex.xy ); \n"
    "    // return float4(f.Tex, 1);\n"
    "}\n"
    "\n"
    "float4 VSBird( uint VertexID : SV_VertexID ) : SV_Position \n"
    "{ \n"
    "	float4 position = float4( 0, 0, 0.5, 1 ) ; \n"
    " \n"
    "	int bn = VertexID / 3 ; \n"
    "	int vn = VertexID % 3 ;  \n"
    " \n"
    "	float2 birdcenter = 0.0014 * g_BirdsPositions.Load(bn) - float2(-0.15, "
    "0.15) ;  \n"
    " \n"
    "	float wing = 0.12 ;  \n"
    "	switch(vn)  \n"
    "	{ \n"
    "	case 0 : \n"
    "		position.x = birdcenter.x - wing ;  \n"
    "		position.y = birdcenter.y - 0.01 ;     \n"
    "		break ; \n"
    "	case 1 :  \n"
    "		position.x = birdcenter.x + wing ;  \n"
    "		position.y = birdcenter.y - 0.01 ;     \n"
    "		break ;  \n"
    "	case 2 : \n"
    "		position.x = birdcenter.x ; \n"
    "		position.y = birdcenter.y + 0.005 ;     \n"
    "		break ;  \n"
    "	} \n"
    " \n"
    "	position.z = 0.5; \n"
    "	position.w = 1.0; \n"
    " \n"
    "	return position ; \n"
    "} \n"
    " \n"
    "float4 PSBird( float4 input : SV_Position ) : SV_Target \n"
    "{ \n"
    "	return float4( 1, 1, 1, 1 ); \n"
    "} \n"
    "RasterizerState NoCull \n"
    "{ \n"
    "    CullMode = None; \n"
    "}; \n"
    "BlendState Opaque \n"
    "{ \n"
    "    BlendEnable[0] = false; \n"
    "}; \n"
    " \n"
    "DepthStencilState DisableDepthTestWrite \n"
    "{ \n"
    "    DepthEnable = FALSE; \n"
    "    DepthWriteMask = 0; \n"
    "}; \n"
    " \n"
    "technique10 DrawBirds \n"
    "{ \n"
    "    pass P0 \n"
    "    { \n"
    "        SetVertexShader( CompileShader( vs_4_0, VSBird() ) ); \n"
    "        SetGeometryShader( NULL ); \n"
    "        SetPixelShader( CompileShader( ps_4_0, PSBird() ) ); \n"
    "         \n"
    "        SetDepthStencilState( DisableDepthTestWrite, 1 ); \n"
    "        SetBlendState(Opaque,float4(0,0,0,0),0xffffffff); \n"
    "        SetRasterizerState(NoCull); \n"
    "    } \n"
    "} \n"
    " \n"
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

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p) \
  {                     \
    if (p) {            \
      (p)->Release();   \
      (p) = NULL;       \
    }                   \
  }
#endif

bool g_runCPU = false;
bool g_bDone = false;

int g_seed = 126832513;  //  1247227967

typedef unsigned int uint;
const uint g_WindowWidth = 784;
const uint g_WindowHeight = 784;

// simulation parameters
float alpha = 90.f;
float upwashX = 30.f;
float upwashY = 50.f;
float wingspan = 50.f;
float dX = .5f;
float dY = .5f;
float epsilon = 30.f;
float lambda = -0.1073f * wingspan;

// number of birds
const uint nBirds = 25;
// positions on host
float2 *positions = NULL;
float2 *new_positions = NULL;

struct WingTip {
  float x;  // x coordinate
  float y;  // y coordinate
  int lr;   // 1 if left, -1 if right
};

struct Gap {
  float2 left;   // left bordering point
  float2 right;  // right bordering point
};

struct ViewGoal {
  float2 pos;  // x coordinate of a bird's goal when pursuing unobstructed view
  float dist;  // distance
};

WingTip *g_wingTips = NULL;
uint2 *pairs = NULL;
uint2 *d_pairs = NULL;

uint3 *triples = NULL;
uint3 *d_triples = NULL;

bool *hasproxy = NULL;
bool *d_hasproxy = NULL;
bool *d_neighbors = NULL;
bool *leftgoals = NULL;
bool *d_leftgoals = NULL;
bool *rightgoals = NULL;
bool *d_rightgoals = NULL;

Params *params = NULL;
Params *d_params = NULL;

// The CUDA kernel launchers that get called
extern "C" void cuda_simulate(float2 *newPos, float2 *curPos, uint numBirds,
                              bool *d_hasproxy, bool *d_neighbors,
                              bool *d_leftgoals, bool *d_rightgoals,
                              uint2 *d_pairs, uint3 *d_triples,
                              Params *m_params);

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);

void DrawScene();
void Cleanup();
void Render();

float2 diff(float2 pos0, float2 pos1);
float norm(float2 pos);
float dist(float2 pos0, float2 pos1);
bool isInsideQuad(float2 pos0, float2 pos1, float width, float height);
void initialize(uint numBirds);
void simulate(float2 *newPos, float2 *curPos, uint numBirds);

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

bool findCUDADevice() {
  int nGraphicsGPU = 0;
  int deviceCount = 0;
  bool bFoundGraphics = false;
  char devname[256];

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
    strcpy(devname, deviceProp.name);
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
    // This prints and resets the cudaError to cudaSuccess
    printLastCudaError("cudaD3D10GetDevice failed");  

    if (cudaSuccess == cuStatus) {
      // If so, mark it as the one against which to create our d3d10 device
      g_pCudaCapableAdapter = pAdapter;
      g_pCudaCapableAdapter->AddRef();
    }

    pAdapter->Release();
  }

  printf("> Found %d D3D10 Adapter(s).\n", (int)adapter);

  pFactory->Release();

  if (!g_pCudaCapableAdapter) {
    printf("> Found 0 D3D10 Adapter(s) /w Compute capability.\n");
    return false;
  }

  DXGI_ADAPTER_DESC adapterDesc;
  g_pCudaCapableAdapter->GetDesc(&adapterDesc);
  wcstombs(dev_name, adapterDesc.Description, 128);

  printf("> Found 1 D3D10 Adapter(s) /w Compute capability.\n");
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

  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

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
  if (argc > 1) {
    // automated build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
      getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
    }
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
  HWND hWnd = CreateWindow(
      wc.lpszClassName, "VFlocking", WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth,
      g_WindowHeight, GetDesktopWindow() /*NULL*/, NULL, wc.hInstance, NULL);

  ShowWindow(hWnd, SW_SHOWDEFAULT);
  UpdateWindow(hWnd);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D(hWnd))) {
    {
      // register the Direct3D resources that we'll use
      // we'll read to and write from g_texture_2d, so don't set any special map
      // flags for it
      cudaError_t error = cudaSuccess;
      error = cudaGraphicsD3D10RegisterResource(
          &g_pCudaResourcePos, g_pPositions, cudaGraphicsRegisterFlagsNone);
      getLastCudaError(
          "cudaGraphicsD3D10RegisterResource (g_texture_2d) failed");

      error = cudaGraphicsResourceSetMapFlags(g_pCudaResourcePos,
                                              cudaD3D10MapFlagsWriteDiscard);
      getLastCudaError("cudaGraphicsResourceSetMapFlags (g_texture_2d) failed");

      cudaGraphicsD3D10RegisterResource(&g_pCudaResourceNewPos, g_pNewPositions,
                                        cudaGraphicsRegisterFlagsNone);
      getLastCudaError(
          "cudaGraphicsD3D10RegisterResource (g_texture_2d) failed");

      error = cudaGraphicsResourceSetMapFlags(g_pCudaResourceNewPos,
                                              cudaD3D10MapFlagsWriteDiscard);
      getLastCudaError("cudaGraphicsResourceSetMapFlags (g_texture_2d) failed");
    }
  }

  srand(g_seed);

  // allocate device memory for positions
  checkCudaErrors(
      cudaMalloc((void **)&d_pairs, nBirds * (nBirds - 1) * sizeof(uint2) / 2));
  checkCudaErrors(
      cudaMalloc((void **)&d_triples,
                 nBirds * (nBirds - 1) * (nBirds - 2) * sizeof(uint3) / 6));

  checkCudaErrors(
      cudaMalloc((void **)&d_neighbors, nBirds * nBirds * sizeof(bool)));
  checkCudaErrors(
      cudaMalloc((void **)&d_leftgoals, nBirds * nBirds * sizeof(bool)));
  checkCudaErrors(
      cudaMalloc((void **)&d_rightgoals, nBirds * nBirds * sizeof(bool)));

  checkCudaErrors(cudaMalloc((void **)&d_hasproxy, nBirds * sizeof(bool)));
  checkCudaErrors(cudaMalloc((void **)&d_params, sizeof(Params)));

  initialize(nBirds);

  g_pd3dDevice->UpdateSubresource(g_pPositions, 0, NULL, positions, 0, 0);

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

          const char *cur_image_path = "VFlockingD3D10.ppm";

          // Save a reference of our current test run image
          CheckRenderD3D10::ActiveRenderTargetToPPM(g_pd3dDevice,
                                                    cur_image_path);

          // compare to official reference image, printing PASS or FAIL.
          g_bPassed = CheckRenderD3D10::PPMvsPPM(cur_image_path, ref_file,
                                                 argv[0], MAX_EPSILON, 0.15f);

          g_bDone = true;
          PostQuitMessage(0);
        } else {
          g_bPassed = true;
        }
      }
    }
  };

  // Unregister windows class
  UnregisterClass(wc.lpszClassName, wc.hInstance);

  // clean
  delete[] positions;

  delete[] new_positions;

  delete[] g_wingTips;

  //
  // and exit
  //
  printf("> %s running on %s exiting...\n", SDK_name, device_name);

  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
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

  // birds' buffer
  D3D10_BUFFER_DESC bdesc;
  memset(&bdesc, 0, sizeof(bdesc));
  bdesc.Usage = D3D10_USAGE_DEFAULT;
  bdesc.ByteWidth = nBirds * sizeof(float2);  // sizeof(D3DXVECTOR2) ; //
  bdesc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
  bdesc.CPUAccessFlags = 0;
  bdesc.MiscFlags = 0;

  g_pd3dDevice->CreateBuffer(&bdesc, NULL, &g_pPositions);
  g_pd3dDevice->CreateBuffer(&bdesc, NULL, &g_pNewPositions);

  D3D10_SHADER_RESOURCE_VIEW_DESC rsvdesc;
  memset(&rsvdesc, 0, sizeof(rsvdesc));
  rsvdesc.Buffer.ElementOffset = 0;
  rsvdesc.Buffer.ElementWidth = nBirds;
  rsvdesc.Format = DXGI_FORMAT_R32G32_FLOAT;
  rsvdesc.ViewDimension = D3D10_SRV_DIMENSION_BUFFER;
  g_pd3dDevice->CreateShaderResourceView(g_pPositions, &rsvdesc,
                                         &g_pPositionsSRV);

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

    g_pDrawQuadTechnique = g_pSimpleEffect->GetTechniqueByName("Render");

    g_pDrawBirdsTechnique = g_pSimpleEffect->GetTechniqueByName("DrawBirds");

    g_pvQuadRect =
        g_pSimpleEffect->GetVariableByName("g_vQuadRect")->AsVector();

    g_pTexture2D =
        g_pSimpleEffect->GetVariableByName("g_Texture2D")->AsShaderResource();

    g_pd3dDevice->IASetInputLayout(NULL);
    g_pd3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
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

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
void DrawScene() {
  // Clear the backbuffer to a black color
  float ClearColor[4] = {0.18f, 0.63f, 1.f, 1.0f};
  g_pd3dDevice->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

  //
  // draw the 2d texture
  //
  float quadRect[4] = {-0.98f, -0.98f, 1.96f, 1.96f};
  g_pvQuadRect->SetFloatVector((float *)&quadRect);

#if 0
    g_pDrawQuadTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->Draw(4, 0);
#else
  g_pDrawBirdsTechnique->GetPassByIndex(0)->Apply(0);

  ID3D10ShaderResourceView *pSRVViews[1];
  pSRVViews[0] = g_pPositionsSRV;
  g_pd3dDevice->VSSetShaderResources(0, 1, pSRVViews);

  g_pd3dDevice->Draw(3 * nBirds, 0);

#endif

  // Present the backbuffer contents to the display
  g_pSwapChain->Present(0, 0);
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup() {
  // unregister the Cuda resources
  cudaGraphicsUnregisterResource(g_pCudaResourcePos);
  getLastCudaError(
      "cudaGraphicsUnregisterResource (g_pCudaResourcePos) failed");
  cudaGraphicsUnregisterResource(g_pCudaResourceNewPos);
  getLastCudaError(
      "cudaGraphicsUnregisterResource (g_pCudaResourceNewPos) failed");

  //
  // clean up Direct3D
  //
  {
    if (g_pInputLayout != NULL) {
      g_pInputLayout->Release();
    }

    if (g_pSimpleEffect != NULL) {
      g_pSimpleEffect->Release();
    }

    if (g_pSwapChainRTV != NULL) {
      g_pSwapChainRTV->Release();
    }

    if (g_pSwapChain != NULL) {
      g_pSwapChain->Release();
    }

    if (g_pd3dDevice != NULL) {
      g_pd3dDevice->Release();
    }
  }

  // Uninitialize CUDA
  checkCudaErrors(cudaFree(d_pairs));
  checkCudaErrors(cudaFree(d_triples));

  checkCudaErrors(cudaFree(d_neighbors));
  checkCudaErrors(cudaFree(d_leftgoals));
  checkCudaErrors(cudaFree(d_rightgoals));

  checkCudaErrors(cudaFree(d_hasproxy));
  checkCudaErrors(cudaFree(d_params));

  SAFE_RELEASE(g_pPositions);
  SAFE_RELEASE(g_pNewPositions);
  SAFE_RELEASE(g_pPositionsSRV);
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void Render() {
  //
  // map the resources we've registered so we can access them in Cuda
  // - it is most efficient to map and unmap all resources in a single call,
  //   and to have the map/unmap calls be the boundary between using the GPU
  //   for Direct3D and Cuda
  cudaGraphicsMapResources(1, &g_pCudaResourcePos, 0);
  getLastCudaError("cudaGraphicsMapResources(3) failed");
  cudaGraphicsMapResources(1, &g_pCudaResourceNewPos, 0);
  getLastCudaError("cudaGraphicsMapResources(3) failed");

  getLastCudaError("cudaD3D10MapResources(3) failed");

  float2 *mappedpositions, *new_mappedpositions;
  static clock_t start, nextstart, end, end2, end3;
  static DWORD tick_start, next_tick_start = 0, tick_end;
  static uint step = 0;

  if (g_runCPU) {
    if (!step) {
      std::cout << "CPU simulation \n";
    }

    if (!(step % 100)) {
      tick_start = next_tick_start;
      next_tick_start = GetTickCount();
    }

    simulate(new_positions, positions, nBirds);
    std::swap(positions, new_positions);
    g_pd3dDevice->UpdateSubresource(g_pPositions, 0, NULL, positions, 0, 0);

    if (!(step % 100) && step) {
      tick_end = GetTickCount();
      std::cout << "CPU, step " << step << " \n";
      std::cout << "time per step " << float(tick_end - tick_start) / 100.f
                << " ms \n";
    }

    step++;
  } else {
    if (!step) {
      std::cout << "CUDA simulation \n";
    }

    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void **)&mappedpositions, &num_bytes, g_pCudaResourcePos));
    getLastCudaError("cudaGraphicsResourceGetMappedPointer 1 failed \n");
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void **)&new_mappedpositions, &num_bytes, g_pCudaResourceNewPos));
    getLastCudaError("cudaGraphicsResourceGetMappedPointer 2 failed \n");

    cuda_simulate(new_mappedpositions, mappedpositions, nBirds, d_hasproxy,
                  d_neighbors, d_leftgoals, d_rightgoals, d_pairs, d_triples,
                  d_params);
    std::swap(g_pCudaResourceNewPos, g_pCudaResourcePos);
    step++;
  }

  //
  // unmap the resources
  //
  cudaGraphicsUnmapResources(1, &g_pCudaResourcePos, 0);
  getLastCudaError("cudaGraphicsUnmapResources(3) failed");
  cudaGraphicsUnmapResources(1, &g_pCudaResourceNewPos, 0);
  getLastCudaError("cudaGraphicsUnmapResources(3) failed");

  // draw the scene using them
  //
  DrawScene();
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

      if (wParam == 'r' || wParam == 'R') {
        g_seed = (unsigned)time(NULL);
        srand(g_seed);

        for (uint i = 0; i < nBirds; i++) {
          positions[i].x = (float)rand() / (RAND_MAX + 1) * 768 - 500;
          positions[i].y = (float)rand() / (RAND_MAX + 1) * 768 - 300;
        }

        g_pd3dDevice->UpdateSubresource(g_pPositions, 0, NULL, positions, 0, 0);
      }

      if (wParam == 'g' || wParam == 'G') {
        g_runCPU = !g_runCPU;
        srand(g_seed);

        for (uint i = 0; i < nBirds; i++) {
          positions[i].x = (float)rand() / (RAND_MAX + 1) * 768 - 500;
          positions[i].y = (float)rand() / (RAND_MAX + 1) * 768 - 300;
        }

        g_pd3dDevice->UpdateSubresource(g_pPositions, 0, NULL, positions, 0, 0);
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

void initForCUDA(uint numBirds) {
  uint i, j, k = 0, l = 0;
  uint2 *p = pairs;

  for (i = 0; i < numBirds; i++)
    for (j = i + 1; j < numBirds; j++) {
      p->x = i;
      p->y = j;
      p++;
    }

  checkCudaErrors(cudaMemcpy(d_pairs, pairs,
                             nBirds * (nBirds - 1) * sizeof(uint2) / 2,
                             cudaMemcpyHostToDevice));

  for (i = 0; i < numBirds; i++)
    for (j = i + 1; j < numBirds; j++)
      for (k = j + 1; k < numBirds; k++) {
        triples[l].x = i;
        triples[l].y = j;
        triples[l].z = k;
        l++;
      }

  checkCudaErrors(
      cudaMemcpy(d_triples, triples,
                 nBirds * (nBirds - 1) * (nBirds - 2) * sizeof(uint3) / 6,
                 cudaMemcpyHostToDevice));
  params->alpha = 90.f;
  params->upwashX = 30.f;
  params->upwashY = 50.f;
  params->wingspan = 50.f;
  params->dX = .5f;
  params->dY = .5f;
  params->epsilon = 30.f;
  params->lambda = -0.1073f * params->wingspan;

  checkCudaErrors(
      cudaMemcpy(d_params, params, sizeof(Params), cudaMemcpyHostToDevice));

  memset(leftgoals, 0, nBirds * nBirds * sizeof(bool));
  memset(rightgoals, 0, nBirds * nBirds * sizeof(bool));

  cudaMemset(d_neighbors, 0, nBirds * nBirds * sizeof(bool));
  cudaMemset(d_leftgoals, 0, nBirds * nBirds * sizeof(bool));
  cudaMemset(d_rightgoals, 0, nBirds * nBirds * sizeof(bool));

  cudaMemset(d_hasproxy, 0, nBirds * sizeof(bool));
}

void initialize(uint numBirds) {
  positions = new float2[numBirds];
  new_positions = new float2[numBirds];
  pairs = new uint2[numBirds * (numBirds - 1) / 2];
  triples = new uint3[numBirds * (numBirds - 1) * (numBirds - 2) / 6];

  params = new Params;
  leftgoals = new bool[numBirds * numBirds];
  rightgoals = new bool[numBirds * numBirds];

  for (uint i = 0; i < numBirds; i++) {
    positions[i].x = (float)rand() / (RAND_MAX + 1) * 768 - 500;
    positions[i].y = (float)rand() / (RAND_MAX + 1) * 768 - 300;
  }

  if (!g_runCPU) {
    initForCUDA(numBirds);
  }

  g_wingTips = new WingTip[2 * numBirds];
}

float2 diff(float2 pos0, float2 pos1) {
  float2 ret;
  ret.x = pos1.x - pos0.x;
  ret.y = pos1.y - pos0.y;
  return ret;
}

float cross(float2 vec0, float2 vec1) {
  return vec0.x * vec1.y - vec0.y * vec1.x;
}

float norm(float2 pos) { return sqrt(pos.x * pos.x + pos.y * pos.y); }

float dist(float2 pos0, float2 pos1) {
  return sqrt((pos0.x - pos1.x) * (pos0.x - pos1.x) +
              (pos0.y - pos1.y) * (pos0.y - pos1.y));
}

bool isInsideQuad(float2 pos0, float2 pos1, float width, float height) {
  if (fabs(pos0.x - pos1.x) < 0.5f * width &&
      fabs(pos0.y - pos1.y) < 0.5f * height) {
    return true;
  } else {
    return false;
  }
}

bool compare(WingTip &t1, WingTip &t2) { return t1.x < t2.x ? true : false; }

bool compareGoals(ViewGoal &g1, ViewGoal &g2) {
  return g1.dist < g2.dist ? true : false;
}

float sign(float x) {
  if (x > 0.f) {
    return 1.f;
  } else if (x < 0.f) {
    return -1.f;
  }

  return 0.f;
}

bool isVisible(float2 pos, float2 goal) {
  float2 leftBorder, rightBorder;
  leftBorder.x = goal.x - (0.5f * wingspan + lambda) - pos.x;
  leftBorder.y = goal.y - pos.y;
  rightBorder.x = goal.x + (0.5f * wingspan + lambda) - pos.x;
  rightBorder.y = goal.y - pos.y;

  for (uint j = 0; j < nBirds; j++) {
    if (positions[j].y >= goal.y || positions[j].y <= pos.y) {
      continue;
    }

    float2 dirl, dirr;
    dirl.x = positions[j].x - pos.x + 0.5f * wingspan;
    dirl.y = positions[j].y - pos.y;
    dirr.x = positions[j].x - pos.x - 0.5f * wingspan;
    dirr.y = positions[j].y - pos.y;

    if (cross(leftBorder, dirl) < 0 && cross(rightBorder, dirr) > 0) {
      return false;
    }
  }

  return true;
}

void simulate(float2 *newPos, float2 *curPos, uint numBirds) {
  uint i, j, k, nGaps;

  std::vector<WingTip> vWingTips;
  Gap g_gaps[nBirds + 1];

  for (i = 0; i < numBirds; i++) {
    WingTip tip;
    tip.x = curPos[i].x - 0.5f * wingspan;
    tip.y = curPos[i].y;
    tip.lr = 1;
    vWingTips.push_back(tip);
    tip.x = curPos[i].x + 0.5f * wingspan;
    tip.y = curPos[i].y;
    tip.lr = -1;
    vWingTips.push_back(tip);
  }

  bool isSorted = false;

  for (i = 0; i < numBirds; i++) {
    std::vector<ViewGoal> vViewGoals;
    bool useRule1 = true;

    newPos[i].x = curPos[i].x;
    newPos[i].y = curPos[i].y;

    uint upwashCount = 0;
    std::vector<uint> vNeighbors;

    for (j = 0; j < numBirds; j++) {
      if (j == i || curPos[j].y < curPos[i].y) {
        continue;
      }

      float2 curPosShiftedBack;
      curPosShiftedBack.x = curPos[j].x;
      curPosShiftedBack.y = curPos[j].y - 0.5f * upwashY;

      if (isInsideQuad(curPos[i], curPosShiftedBack,
                       2.f * (wingspan + lambda + upwashX), upwashY)) {
        uint neighbor = j;
        vNeighbors.push_back(neighbor);

        if (useRule1) {
          useRule1 = false;
        }

        if (curPos[i].x > curPos[j].x + (wingspan + lambda + 0.5f * upwashX) ||
            curPos[i].x < curPos[j].x - (wingspan + lambda + 0.5f * upwashX)) {
          upwashCount++;
        }
      }
    }

    // if rule 1 is valid, find nearest bird and move to it
    float d = 0.f, minDist = 1000.f;
    float2 dij;
    dij.x = 0.f;
    dij.y = 0.f;
    uint nearest = 1000;

    if (useRule1) {
      for (j = 0; j < numBirds; j++) {
        if (j == i || curPos[j].y < curPos[i].y) {
          continue;
        }

        if ((d = norm(diff(curPos[i], curPos[j]))) < minDist) {
          minDist = d;
          nearest = j;
          dij = diff(curPos[i], curPos[j]);
        }
      }

      if (!d) {
        continue;
      }

      d ? dij.x = dij.x / d : dij.x = 0.f;
      d ? dij.y = dij.y / d : dij.y = 0.f;
      newPos[i].x = curPos[i].x + dX * dij.x;
      newPos[i].y = curPos[i].y + dY * dij.y;
    } else {
      if (!isSorted) {
        std::sort(vWingTips.begin(), vWingTips.end(), compare);
        isSorted = true;
      }

      // find all gaps that are big enough
      int count = 0;
      bool gapBegin = true;
      g_gaps[0].left.x = -1000.f;
      g_gaps[0].left.y = 0.f;
      j = 0;
      ViewGoal goal;

      for (k = 0; k < 2 * numBirds; k++) {
        if (vWingTips[k].y <= curPos[i].y)  // look for gaps only ahead
        {
          continue;
        }

        count += vWingTips[k].lr;

        if (gapBegin && 1 == count) {
          gapBegin = false;
          g_gaps[j].right.x = vWingTips[k].x;
          g_gaps[j].right.y = vWingTips[k].y;

          if (g_gaps[j].right.x - g_gaps[j].left.x > wingspan + 2.f * lambda) {
            goal.pos.x = g_gaps[j].right.x - (0.5f * wingspan + lambda);
            goal.pos.y = g_gaps[j].right.y;
            goal.dist = fabs(goal.pos.x - curPos[i].x);
            vViewGoals.push_back(goal);

            if (j) {
              goal.pos.x = g_gaps[j].left.x + (0.5f * wingspan + lambda);
              goal.pos.y = g_gaps[j].left.y;
              goal.dist = fabs(goal.pos.x - curPos[i].x);
              vViewGoals.push_back(goal);
            }
          }
        } else if (!count) {
          j++;
          gapBegin = true;
          g_gaps[j].left.x = vWingTips[k].x;  // + 0.5f * wingspan + lambda;
          g_gaps[j].left.y = vWingTips[k].y;
        }
      }

      g_gaps[j].right.x = 1000.f;
      g_gaps[j].right.y = 0.f;
      goal.pos.x = g_gaps[j].left.x + (0.5f * wingspan + lambda);
      goal.pos.y = g_gaps[j].left.y;
      goal.dist = fabs(goal.pos.x - curPos[i].x);
      vViewGoals.push_back(goal);
      nGaps = j + 1;

      // search the closest gap for unobstructed view
      minDist = 1000.f;
      dij.x = 0.f;
      dij.y = 0.f;

      for (j = 0; j < nGaps; j++) {
        if ((d = norm(diff(curPos[i], g_gaps[j].left))) < minDist) {
          minDist = d;
          dij = diff(curPos[i], g_gaps[j].left);
        }

        if ((d = norm(diff(curPos[i], g_gaps[j].right))) < minDist) {
          minDist = d;
          dij = diff(curPos[i], g_gaps[j].right);
        }
      }

      std::sort(vViewGoals.begin(), vViewGoals.end(), compareGoals);

      if (vViewGoals.size()) {
        if (vViewGoals[0].dist <= dX) {
          continue;
        }

        for (j = 0; j < vViewGoals.size(); j++) {
          if (!isVisible(curPos[i], vViewGoals[j].pos)) {
            continue;
          }

          newPos[i].x =
              curPos[i].x + sign(vViewGoals[j].pos.x - curPos[i].x) * dX;

          for (k = 0; k < vNeighbors.size(); k++) {
            if (curPos[vNeighbors[k]].y >= curPos[i].y &&
                curPos[vNeighbors[k]].y < curPos[i].y + epsilon) {
              newPos[i].y = curPos[i].y - dY;
            } else if (curPos[vNeighbors[k]].y < curPos[i].y &&
                       curPos[vNeighbors[k]].y > curPos[i].y - epsilon) {
              newPos[i].y = curPos[i].y + dY;
            }
          }

          break;
        }
      }
    }

    vNeighbors.clear();
    vViewGoals.clear();
  }

  vWingTips.clear();
}
