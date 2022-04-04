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

/* This example demonstrates how to use the CUDA Direct3D bindings to fill
 * a vertex buffer with CUDA and use Direct3D to render the data.
 * Host code.
 */

#pragma warning(disable : 4312)

#include <windows.h>
#include <mmsystem.h>

// this header inclues all the necessary D3D10 includes
#include <dynlink_d3d10.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d10_interop.h>

// includes, project
#include <rendercheck_d3d10.h>
#include <helper_cuda.h>
#include <helper_functions.h>

int g_iFrameToCompare = 10;

bool g_bDone = false;
bool g_bPassed = true;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX_EPSILON 10

static char *SDK_name = "simpleD3D10RenderTarget";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D10Device *g_pd3dDevice = NULL;           // Our rendering device
IDXGISwapChain *g_pSwapChain = NULL;         // The swap chain of the window
ID3D10RenderTargetView *g_pSwapChainRTV =
    NULL;  // The Render target view on the swap chain ( used for clear)
ID3D10RasterizerState *g_pRasterState = NULL;

struct Color {
  ID3D10Texture2D *pBuffer;  // The color buffer
  ID3D10RenderTargetView
      *pBufferRTV;  // The Render target view on the color buffer
  ID3D10ShaderResourceView
      *pBufferSRV;  // The shader resource view on the color buffer
  cudaGraphicsResource *cudaResource;  // resource of the Buffer on cuda side
  int pitch;
  cudaArray *pCudaArray;  // the data in a cuda view
} g_color;

struct Histogram {
  ID3D10Buffer *pBuffer;                 // Buffer to hold histogram
  ID3D10ShaderResourceView *pBufferSRV;  // View on the histogram buffer
  cudaGraphicsResource *cudaResource;    // resource of the Buffer on cuda side
  unsigned int *cudaBuffer;  // staging buffer to allow cuda to write results
  // cudaArray*                    pCudaArray; // the data in a cuda view
  size_t size;
} g_histogram;

ID3D10Effect *g_pDisplayEffect = NULL;
ID3D10EffectTechnique *g_pDisplayTechnique = NULL;
ID3D10EffectScalarVariable *g_pTime = NULL;

static const char g_displayEffectSrc[] =
    "float   g_Time; \n"
    "uint2   g_vGrid = uint2(20,20); \n"
    "float4 g_vGridSize = float4(0.05f, 0.05f, 0.046f, 0.046f); \n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float2 Tex : TEXCOORD0; \n"
    "    float4 Col : TEXCOORD1; };\n"
    "\n"
    "Fragment VS( uint instanceId : SV_InstanceID, uint vertexId : SV_VertexID "
    ")\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Tex = float2( 1.f*((vertexId == 1) || (vertexId == 3)), 1.f*( "
    "vertexId >= 2)); \n"
    "    \n"
    "    uint2 cellId = uint2(instanceId % g_vGrid.x, instanceId / "
    "g_vGrid.x);\n"
    "    f.Pos = float4( g_vGridSize.xy*cellId + 0.5f*(g_vGridSize.xy - "
    "g_vGridSize.zw) + f.Tex * g_vGridSize.zw, 0, 1);\n"
    "    f.Pos.xy = (f.Pos.xy*2.f - 1.f);\n"
    "    \n"
    "    f.Col = float4( ((g_vGrid.x-1.f) - cellId.x) / (g_vGrid.x-1.f), "
    "(cellId.x + (g_vGrid.y-1.f) - cellId.y) / (g_vGrid.x+g_vGrid.y-1.f), "
    "cellId.y / (g_vGrid.y-1.f), 1.f);\n"
    "    f.Col *= float4( 0.5 + 0.5*sin(g_Time), 0.5 + "
    "0.5*sin(g_Time)*cos(g_Time), 0.5 + 0.5*cos(g_Time), 1.f);\n"
    "    return f;\n"
    "}\n"
    "\n"
    "float4 PS( Fragment f ) : SV_Target\n"
    "{\n"
    "    return f.Col;\n"
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

ID3D10Effect *g_pCompositeEffect = NULL;
ID3D10EffectTechnique *g_pCompositeTechnique = NULL;
ID3D10EffectVectorVariable *g_pvQuadRect = NULL;
ID3D10EffectScalarVariable *g_pUseCase = NULL;
ID3D10EffectShaderResourceVariable *g_pTexture2D = NULL;
ID3D10EffectShaderResourceVariable *g_pHistogram = NULL;

static const char g_compositeEffectSrc[] =
    "float4 g_vQuadRect; \n"
    "int g_UseCase; \n"
    "Texture2D g_Texture2D; \n"
    "Buffer<uint> g_Histogram; \n"
    "\n"
    "SamplerState samLinear{ \n"
    "    Filter = MIN_MAG_LINEAR_MIP_POINT; \n"
    "};\n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float3 Tex : TEXCOORD0; \n"
    "    float2 uv : TEXCOORD1; };\n"
    "\n"
    "Fragment VS( uint vertexId : SV_VertexID )\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Tex = float3( 1.f*((vertexId == 1) || (vertexId == 3)), 1.f*( "
    "vertexId >= 2), 0.f); \n"
    "    \n"
    "    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n"
    "    \n"
    "    f.uv = float2( f.Tex.x*255.f, f.Tex.y*50000.f ); \n"
    "    return f;\n"
    "}\n"
    "\n"
    "float4 PS( Fragment f ) : SV_Target\n"
    "{\n"
    "    if (g_UseCase == 0) \n"
    "        return g_Texture2D.Sample( samLinear, f.Tex.xy ); \n"
    "    else if (g_UseCase == 1) { \n"
    "        uint index = f.uv.x; \n"
    "        float value = g_Histogram.Load( index ); \n"
    "        //float value = index * 1000; \n"
    "        float red = ( value >= f.uv.y ? (0.5f * f.uv.y / value) + 0.5f : "
    "0.f ); \n"
    "        return float4(red, 0, 0, 1); \n"
    "    } else return float4(f.Tex, 1);\n"
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

// testing/tracing function used pervasively in tests.  if the condition is
// unsatisfied then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x)                                                  \
  if (!(x)) {                                                            \
    fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, \
            __FILE__, __LINE__);                                         \
    return 1;                                                            \
  }

const unsigned int g_WindowWidth = 800;
const unsigned int g_WindowHeight = 800;
const unsigned int g_HistogramSize = 256;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
void runTest(int argc, char **argv, char *ref_file);
void runCuda();
HRESULT InitD3D(HWND hWnd);
VOID Cleanup();
VOID Render();
LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// CUDA/D3D10 kernels
extern "C" void checkCudaError();
extern "C" void createHistogramTex(unsigned int *histogram, unsigned int width,
                                   unsigned int height, cudaArray *colorArray);

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
  wcstombs(dev_name, adapterDesc.Description, 128);

  printf("> Found 1 D3D10 Adapater(s) /w Compute capability.\n");
  printf("> %s\n", dev_name);

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  char device_name[256];
  char *ref_file = NULL;

  pArgc = &argc;
  pArgv = argv;

  printf("[%s] - Starting...\n", SDK_name);

  if (!findCUDADevice())  // Search for CUDA GPU
  {
    printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
    exit(EXIT_SUCCESS);
  }

  if (!dynlinkLoadD3D10API())  // Search for D3D API (locate drivers, does not
                               // mean device is found)
  {
    printf("> D3D10 API libraries NOT found.. Exiting.\n");
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
    // automatied build testing harness
    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
      getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

  // run D3D10/CUDA test
  runTest(argc, argv, ref_file);

  //
  // and exit
  //
  printf("%s running on %s exiting...\n", SDK_name, device_name);

  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv, char *ref_file) {
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
  HWND hWnd = CreateWindow(wc.lpszClassName, "CUDA/D3D10 RenderTarget InterOP",
                           WS_OVERLAPPEDWINDOW, 100, 100, g_WindowWidth,
                           g_WindowHeight, NULL, NULL, wc.hInstance, NULL);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D(hWnd))) {
    // Initialize interoperability between CUDA and Direct3D
    // Register vertex buffer with CUDA
    cudaGraphicsD3D10RegisterResource(&g_histogram.cudaResource,
                                      g_histogram.pBuffer,
                                      cudaGraphicsMapFlagsNone);
    getLastCudaError("cudaGraphicsD3D10RegisterResource (g_pHistogram) failed");

    // Register color buffer with CUDA
    cudaGraphicsD3D10RegisterResource(&g_color.cudaResource, g_color.pBuffer,
                                      cudaGraphicsMapFlagsNone);
    getLastCudaError(
        "cudaGraphicsD3D10RegisterResource (g_color.pBuffer) failed");

    // Show the window
    ShowWindow(hWnd, SW_SHOWDEFAULT);
    UpdateWindow(hWnd);
  }

  //
  // The main loop
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

          const char *cur_image_path = "simpleD3D10RenderTarget.ppm";

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
  }

  // Release D3D Library (after message loop)
  dynlinkUnloadD3D10API();

  // Unregister windows class
  UnregisterClass(wc.lpszClassName, wc.hInstance);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda() {
  cudaStream_t stream = 0;
  const int nbResources = 2;
  cudaGraphicsResource *ppResources[nbResources] = {
      g_histogram.cudaResource, g_color.cudaResource,
  };
  // Map resources for Cuda
  checkCudaErrors(cudaGraphicsMapResources(nbResources, ppResources, stream));
  getLastCudaError("cudaGraphicsMapResources(2) failed");
  // Get pointers
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&g_histogram.cudaBuffer, &g_histogram.size,
      g_histogram.cudaResource));
  getLastCudaError(
      "cudaGraphicsResourceGetMappedPointer (g_color.pBuffer) failed");
  cudaGraphicsSubResourceGetMappedArray(&g_color.pCudaArray,
                                        g_color.cudaResource, 0, 0);
  getLastCudaError(
      "cudaGraphicsSubResourceGetMappedArray (g_color.pBuffer) failed");

  // Execute kernel
  createHistogramTex(g_histogram.cudaBuffer, g_WindowWidth, g_WindowHeight,
                     g_color.pCudaArray);
  checkCudaError();
  //
  // unmap the resources
  //
  checkCudaErrors(cudaGraphicsUnmapResources(nbResources, ppResources, stream));
  getLastCudaError("cudaGraphicsUnmapResources(2) failed");
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
      g_pCudaCapableAdapter, D3D10_DRIVER_TYPE_HARDWARE, NULL,
      0,  // D3D10_CREATE_DEVICE_DEBUG,
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

  // Create a color buffer, corresponding render target view and shader resource
  // view
  D3D10_TEXTURE2D_DESC tex2Ddesc;
  ZeroMemory(&tex2Ddesc, sizeof(D3D10_TEXTURE2D_DESC));
  tex2Ddesc.Width = g_WindowWidth;
  tex2Ddesc.Height = g_WindowHeight;
  tex2Ddesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  tex2Ddesc.MipLevels = 1;
  tex2Ddesc.ArraySize = 1;
  tex2Ddesc.SampleDesc.Count = 1;
  tex2Ddesc.Usage = D3D10_USAGE_DEFAULT;
  tex2Ddesc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;

  hr = g_pd3dDevice->CreateTexture2D(&tex2Ddesc, NULL, &g_color.pBuffer);
  AssertOrQuit(SUCCEEDED(hr));

  hr = g_pd3dDevice->CreateShaderResourceView(g_color.pBuffer, NULL,
                                              &g_color.pBufferSRV);
  AssertOrQuit(SUCCEEDED(hr));

  hr = g_pd3dDevice->CreateRenderTargetView(g_color.pBuffer, NULL,
                                            &g_color.pBufferRTV);
  AssertOrQuit(SUCCEEDED(hr));

  // Create a buffer which will contain the resulting histogram and the SRV to
  // plug it
  D3D10_BUFFER_DESC bufferDesc;
  bufferDesc.Usage = D3D10_USAGE_DEFAULT;
  // NOTE: allocation of more than what is needed to display in the shader
  // but this 64 factor is required for CUDA to work with this buffer (see
  // BLOCK_N in .cu code...)
  bufferDesc.ByteWidth =
      sizeof(unsigned int) * g_HistogramSize * 64 /*BLOCK_N*/;
  bufferDesc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
  bufferDesc.CPUAccessFlags = 0;
  bufferDesc.MiscFlags = 0;
  // useless values... we could remove this...
  unsigned int values[256 * 64];

  for (int i = 0; i < 256 * 64; i++) {
    values[i] = i;
  }

  D3D10_SUBRESOURCE_DATA data;
  data.pSysMem = values;
  data.SysMemPitch = 0;
  data.SysMemSlicePitch = 0;

  hr = g_pd3dDevice->CreateBuffer(&bufferDesc, &data, &g_histogram.pBuffer);
  AssertOrQuit(SUCCEEDED(hr));

  D3D10_SHADER_RESOURCE_VIEW_DESC bufferSRVDesc;
  bufferSRVDesc.Format = DXGI_FORMAT_R32_UINT;
  bufferSRVDesc.ViewDimension = D3D10_SRV_DIMENSION_BUFFER;
  bufferSRVDesc.Buffer.ElementOffset = 0;
  bufferSRVDesc.Buffer.ElementWidth =
      g_HistogramSize;  // 4*sizeof(unsigned int);

  hr = g_pd3dDevice->CreateShaderResourceView(
      g_histogram.pBuffer, &bufferSRVDesc, &g_histogram.pBufferSRV);
  AssertOrQuit(SUCCEEDED(hr));
  // Create the equivalent as a cuda staging buffer that we'll use to write from
  // Cuda. Then we'll copy it to the texture
  // cudaMalloc(g_histogram.cudaBuffer, sizeof(float) * g_HistogramSize;
  // getLastCudaError("cudaMallocPitch (g_histogram) failed");

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
    ID3D10Blob *pErrors = NULL;
    ID3D10Blob *pCompiledEffect;
    hr = sFnPtr_D3D10CompileEffectFromMemory((void *)g_displayEffectSrc,
                                             sizeof(g_displayEffectSrc), NULL,
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
        g_pd3dDevice, NULL, &g_pDisplayEffect);
    pCompiledEffect->Release();

    g_pDisplayTechnique = g_pDisplayEffect->GetTechniqueByName("Render");

    g_pTime = g_pDisplayEffect->GetVariableByName("g_Time")->AsScalar();
  }

  // Setup the effect
  {
    ID3D10Blob *pCompiledEffect;
    ID3D10Blob *pErrors = NULL;
    hr = sFnPtr_D3D10CompileEffectFromMemory((void *)g_compositeEffectSrc,
                                             sizeof(g_compositeEffectSrc), NULL,
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
        g_pd3dDevice, NULL, &g_pCompositeEffect);
    pCompiledEffect->Release();

    g_pCompositeTechnique = g_pCompositeEffect->GetTechniqueByName("Render");

    g_pvQuadRect =
        g_pCompositeEffect->GetVariableByName("g_vQuadRect")->AsVector();
    g_pUseCase = g_pCompositeEffect->GetVariableByName("g_UseCase")->AsScalar();

    g_pTexture2D = g_pCompositeEffect->GetVariableByName("g_Texture2D")
                       ->AsShaderResource();
    g_pTexture2D->SetResource(g_color.pBufferSRV);

    g_pHistogram = g_pCompositeEffect->GetVariableByName("g_Histogram")
                       ->AsShaderResource();
    g_pHistogram->SetResource(g_histogram.pBufferSRV);
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
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
VOID Cleanup() {
  if (g_histogram.pBuffer != NULL) {
    // Unregister vertex buffer
    cudaGraphicsUnregisterResource(g_histogram.cudaResource);
    getLastCudaError("cudaGraphicsUnregisterResource failed");
    g_histogram.pBuffer->Release();
  }

  if (g_histogram.pBufferSRV != NULL) {
    g_histogram.pBufferSRV->Release();
  }

  if (g_pDisplayEffect != NULL) {
    g_pDisplayEffect->Release();
  }

  if (g_pCompositeEffect != NULL) {
    g_pCompositeEffect->Release();
  }

  if (g_color.pBufferSRV != NULL) {
    g_color.pBufferSRV->Release();
  }

  if (g_color.pBufferRTV != NULL) {
    g_color.pBufferRTV->Release();
  }

  if (g_color.pBuffer != NULL) {
    // Unregister vertex buffer
    cudaGraphicsUnregisterResource(g_color.cudaResource);
    getLastCudaError("cudaD3D10UnregisterResource failed");
    g_color.pBuffer->Release();
  }

  if (g_pRasterState != NULL) {
    g_pRasterState->Release();
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

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Draws the scene
//-----------------------------------------------------------------------------
VOID Render() {
  g_pd3dDevice->RSSetState(g_pRasterState);

  // Draw frame
  {
    static float time = 0.f;
    time += 0.001f;
    g_pTime->SetFloat(time);

    // Clear the Color to a black color
    float ClearColor[4] = {0.f, 0.1f, 0.1f, 1.f};
    g_pd3dDevice->ClearRenderTargetView(g_color.pBufferRTV, ClearColor);
    g_pd3dDevice->OMSetRenderTargets(1, &g_color.pBufferRTV, NULL);

    g_pd3dDevice->IASetInputLayout(0);
    g_pd3dDevice->IASetPrimitiveTopology(
        D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    g_pDisplayTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->DrawInstanced(4, 400, 0, 0);
  }

  // Run CUDA to compute the histogram
  runCuda();

  // draw the 2d texture
  {
    // Clear the Color to a black color
    float ClearColor[4] = {0, 0, 0, 1.f};
    g_pd3dDevice->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);
    g_pd3dDevice->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

    g_pd3dDevice->IASetInputLayout(0);
    g_pd3dDevice->IASetPrimitiveTopology(
        D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    g_pTexture2D->SetResource(g_color.pBufferSRV);
    g_pHistogram->SetResource(g_histogram.pBufferSRV);

    g_pUseCase->SetInt(0);
    float quadRect1[4] = {-1.0f, -0.8f, 2.0f, 1.8f};
    g_pvQuadRect->SetFloatVector((float *)&quadRect1);

    g_pCompositeTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->Draw(4, 0);

    g_pUseCase->SetInt(1);
    float quadRect2[4] = {-0.8f, -0.99f, 1.6f, 0.19f};
    g_pvQuadRect->SetFloatVector((float *)&quadRect2);

    g_pCompositeTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->Draw(4, 0);

    g_pTexture2D->SetResource(NULL);
    g_pHistogram->SetResource(NULL);
    g_pCompositeTechnique->GetPassByIndex(0)->Apply(0);
  }

  // Present the backbuffer contents to the display
  g_pSwapChain->Present(0, 0);
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_DESTROY:
    case WM_KEYDOWN:
      if (msg != WM_KEYDOWN || wParam == 27) {
        g_bDone = true;
        Cleanup();
        PostQuitMessage(0);
        return 0;
      }
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
