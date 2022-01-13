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

// Windows System include files
#include <windows.h>
#include <mmsystem.h>

// This header includes all the necessary D3D10 and CUDA includes
#include <dynlink_d3d10.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d10_interop.h>

// includes, project
#include <rendercheck_d3d10.h>
#include <helper_functions.h>  // Helper functions for other non-cuda utilities
#include <helper_cuda.h>       // CUDA Helper Functions for initialization

#define MAX_EPSILON 10

static char *sSDKSample = "simpleD3D10";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D10Device *g_pd3dDevice = NULL;           // Our rendering device
IDXGISwapChain *g_pSwapChain = NULL;         // The swap chain of the window
ID3D10RenderTargetView *g_pSwapChainRTV =
    NULL;  // The Render target view on the swap chain ( used for clear)

ID3D10InputLayout *g_pInputLayout = NULL;
ID3D10Effect *g_pSimpleEffect = NULL;
ID3D10EffectTechnique *g_pSimpleTechnique = NULL;
ID3D10EffectMatrixVariable *g_pmWorld = NULL;
ID3D10EffectMatrixVariable *g_pmView = NULL;
ID3D10EffectMatrixVariable *g_pmProjection = NULL;

static const char g_simpleEffectSrc[] =
    "matrix g_mWorld;\n"
    "matrix g_mView;\n"
    "matrix g_mProjection;\n"
    "\n"
    "struct Fragment{ \n"
    "    float4 Pos : SV_POSITION;\n"
    "    float4 Col : TEXCOORD0; };\n"
    "\n"
    "Fragment VS( float4 Pos : POSITION, float4 Col : COLOR )\n"
    "{\n"
    "    Fragment f;\n"
    "    f.Pos = mul(Pos, g_mWorld);\n"
    "    f.Pos = mul(f.Pos, g_mView);\n"
    "    f.Pos = mul(f.Pos, g_mProjection);\n"
    "    f.Col = Col;\n"
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

ID3D10Buffer *g_pVB = NULL;  // Buffer to hold vertices

struct cudaGraphicsResource *cuda_VB_resource;  // handles D3D10-CUDA exchange

// testing/tracing function used pervasively in tests.  if the condition is
// unsatisfied then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x)                                                  \
  if (!(x)) {                                                            \
    fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, \
            __FILE__, __LINE__);                                         \
    return 1;                                                            \
  }

// A structure for our custom vertex type
struct CUSTOMVERTEX {
  FLOAT x, y, z;  // The untransformed, 3D position for the vertex
  DWORD color;    // The vertex color
};

const unsigned int g_WindowWidth = 1024;
const unsigned int g_WindowHeight = 1024;

const unsigned int g_MeshWidth = 512;
const unsigned int g_MeshHeight = 512;

const unsigned int g_NumVertices = g_MeshWidth * g_MeshHeight;

bool g_bPassed = true;
int g_iFrameToCompare = 10;

int *pArgc = NULL;
char **pArgv = NULL;

float anim;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
void runTest(int argc, char **argv, char *ref_file);
void runCuda();
bool SaveResult(int argc, char **argv);
HRESULT InitD3D(HWND hWnd);
HRESULT InitGeometry();
VOID Cleanup();
VOID SetupMatrices();
VOID Render();
LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// CUDA/D3D10 kernel
extern "C" void simpleD3DKernel(float4 *pos, unsigned int width,
                                unsigned int height, float time);

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
    // This prints and resets  the cudaError to cudaSuccess
    printLastCudaError("cudaD3D10GetDevice failed");

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

  printf("> %s starting...\n", sSDKSample);

  if (!findCUDADevice()) {  // Search for CUDA GPU
    printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
    exit(EXIT_SUCCESS);
  }

  // Search for D3D API (locate drivers, does not mean device is found)
  if (!dynlinkLoadD3D10API()) {
    printf("> D3D10 API libraries NOT found on.. Exiting.\n");
    dynlinkUnloadD3D10API();
    exit(EXIT_SUCCESS);
  }

  if (!findDXDevice(device_name)) {  // Search for D3D Hardware Device
    printf("> D3D10 Graphics Device NOT found.. Exiting.\n");
    dynlinkUnloadD3D10API();
    exit(EXIT_SUCCESS);
  }

  if (argc > 1) {
    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
      getCmdLineArgumentString(argc, (const char **)argv, "file",
                               (char **)&ref_file);
    }
  }

  runTest(argc, argv, ref_file);

  //
  // and exit
  //
  printf("%s running on %s exiting...\n", sSDKSample, device_name);
  printf("%s sample finished returned: %s\n", sSDKSample,
         (g_bPassed ? "OK" : "ERROR!"));
  exit(g_bPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv, char *ref_file) {
  // Register the window class
  WNDCLASSEX wc = {sizeof(WNDCLASSEX),       CS_CLASSDC, MsgProc, 0L,   0L,
                   GetModuleHandle(NULL),    NULL,       NULL,    NULL, NULL,
                   "CUDA/D3D10 simpleD3D10", NULL};
  RegisterClassEx(&wc);

  // Create the application's window
  int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
  int yMenu = ::GetSystemMetrics(SM_CYMENU);
  int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
  HWND hWnd = CreateWindow(
      wc.lpszClassName, "CUDA/D3D10 simpleD3D10", WS_OVERLAPPEDWINDOW, 0, 0,
      g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu, NULL,
      NULL, wc.hInstance, NULL);

  // Initialize Direct3D
  if (SUCCEEDED(InitD3D(hWnd))) {
    // Create the scene geometry
    if (SUCCEEDED(InitGeometry())) {
      // Initialize interoperability between CUDA and Direct3D
      // Register vertex buffer with CUDA
      // DEPRECATED: cudaD3D10RegisterResource(g_pVB,
      // cudaD3D10RegisterFlagsNone);
      cudaGraphicsD3D10RegisterResource(&cuda_VB_resource, g_pVB,
                                        cudaD3D10RegisterFlagsNone);
      getLastCudaError("cudaGraphicsD3D10RegisterResource (g_pVB) failed");

      // Initialize vertex buffer with CUDA
      runCuda();

      // Save result
      SaveResult(argc, argv);

      // Show the window
      ShowWindow(hWnd, SW_SHOWDEFAULT);
      UpdateWindow(hWnd);

      // Enter the message loop
      MSG msg;
      ZeroMemory(&msg, sizeof(msg));

      while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
          TranslateMessage(&msg);
          DispatchMessage(&msg);
        } else {
          Render();

          if (ref_file != NULL) {
            for (int count = 0; count < g_iFrameToCompare; count++) {
              Render();
            }

            const char *cur_image_path = "simpleD3D10.ppm";

            // Save a reference of our current test run image
            CheckRenderD3D10::ActiveRenderTargetToPPM(g_pd3dDevice,
                                                      cur_image_path);

            // compare to offical reference image, printing PASS or FAIL.
            g_bPassed = CheckRenderD3D10::PPMvsPPM(cur_image_path, ref_file,
                                                   argv[0], MAX_EPSILON, 0.15f);

            Cleanup();

            PostQuitMessage(0);
          }
        }
      }
    }
  }

  // Release D3D Library (after message loop)
  dynlinkUnloadD3D10API();

  UnregisterClass(wc.lpszClassName, wc.hInstance);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda() {
  // Map vertex buffer to Cuda
  float4 *d_ptr;

  // CUDA Map call to the Vertex Buffer and return a pointer
  // DEPRECATED: cudaD3D10MapResources(1, (ID3D10Resource **)&g_pVB);
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_VB_resource, 0));
  getLastCudaError("cudaGraphicsMapResources failed");

  // DEPRECATED: cudaD3D10ResourceGetMappedPointer( (void **)&dptr, g_pVB, 0);
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&d_ptr, &num_bytes, cuda_VB_resource));
  getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

  // Execute kernel
  simpleD3DKernel(d_ptr, g_MeshWidth, g_MeshHeight, anim);

  // CUDA Map Unmap vertex buffer
  // DEPRECATED: cudaD3D10UnmapResources(1, (ID3D10Resource **)&g_pVB);
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_VB_resource, 0));
  getLastCudaError("cudaGraphicsUnmapResource failed");
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
bool SaveResult(int argc, char **argv) {
  // Map vertex buffer
  float *data;

  if (FAILED(g_pVB->Map(D3D10_MAP_READ, 0,
                        (void **)&data)))  // Lock(0, 0, (void**)&data, 0)))
    return false;

  // Unmap
  g_pVB->Unmap();

  // Save result
  if (checkCmdLineFlag(argc, (const char **)argv, "regression")) {
    // write file for regression test
    sdkWriteFile<float>("./data/regression.dat", data, sizeof(CUSTOMVERTEX),
                        0.0f, false);
  }

  return true;
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
    hr = sFnPtr_D3D10CompileEffectFromMemory((void *)g_simpleEffectSrc,
                                             sizeof(g_simpleEffectSrc), NULL,
                                             NULL,  // pDefines
                                             NULL,  // pIncludes
                                             0,     // HLSL flags
                                             0,     // FXFlags
                                             &pCompiledEffect, NULL);
    AssertOrQuit(SUCCEEDED(hr));

    hr = sFnPtr_D3D10CreateEffectFromMemory(
        pCompiledEffect->GetBufferPointer(), pCompiledEffect->GetBufferSize(),
        0,  // FXFlags
        g_pd3dDevice, NULL, &g_pSimpleEffect);
    pCompiledEffect->Release();

    g_pSimpleTechnique = g_pSimpleEffect->GetTechniqueByName("Render");

    // g_pmWorldViewProjection =
    // g_pSimpleEffect->GetVariableByName("g_mWorldViewProjection")->AsMatrix();
    g_pmWorld = g_pSimpleEffect->GetVariableByName("g_mWorld")->AsMatrix();
    g_pmView = g_pSimpleEffect->GetVariableByName("g_mView")->AsMatrix();
    g_pmProjection =
        g_pSimpleEffect->GetVariableByName("g_mProjection")->AsMatrix();

    // Define the input layout
    D3D10_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D10_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 12,
         D3D10_INPUT_PER_VERTEX_DATA, 0},
    };
    UINT numElements = sizeof(layout) / sizeof(layout[0]);

    // Create the input layout
    D3D10_PASS_DESC PassDesc;
    g_pSimpleTechnique->GetPassByIndex(0)->GetDesc(&PassDesc);
    hr = g_pd3dDevice->CreateInputLayout(
        layout, numElements, PassDesc.pIAInputSignature,
        PassDesc.IAInputSignatureSize, &g_pInputLayout);
    AssertOrQuit(SUCCEEDED(hr));

    // Setup Input Layout, apply effect and draw points
    g_pd3dDevice->IASetInputLayout(g_pInputLayout);
    g_pSimpleTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_POINTLIST);
  }

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: InitGeometry()
// Desc: Creates the scene geometry
//-----------------------------------------------------------------------------
HRESULT InitGeometry() {
  // Setup buffer desc
  D3D10_BUFFER_DESC bufferDesc;
  bufferDesc.Usage = D3D10_USAGE_DEFAULT;
  bufferDesc.ByteWidth = sizeof(CUSTOMVERTEX) * g_NumVertices;
  bufferDesc.BindFlags = D3D10_BIND_VERTEX_BUFFER;
  bufferDesc.CPUAccessFlags = 0;
  bufferDesc.MiscFlags = 0;

  // Create the buffer, no need for sub resource data struct since everything
  // will be defined from cuda
  if (FAILED(g_pd3dDevice->CreateBuffer(&bufferDesc, NULL, &g_pVB)))
    return E_FAIL;

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
VOID Cleanup() {
  if (g_pVB != NULL) {
    // Unregister vertex buffer
    // DEPRECATED: checkCudaErrors(cudaD3D10UnregisterResource(g_pVB));
    cudaGraphicsUnregisterResource(cuda_VB_resource);
    getLastCudaError("cudaGraphicsUnregisterResource failed");

    g_pVB->Release();
  }

  if (g_pInputLayout != NULL) g_pInputLayout->Release();

  if (g_pSimpleEffect != NULL) g_pSimpleEffect->Release();

  if (g_pSwapChainRTV != NULL) g_pSwapChainRTV->Release();

  if (g_pSwapChain != NULL) g_pSwapChain->Release();

  if (g_pd3dDevice != NULL) g_pd3dDevice->Release();
}

//-----------------------------------------------------------------------------
// Name: SetupMatrices()
// Desc: Sets up the world, view, and projection transform matrices.
//-----------------------------------------------------------------------------
VOID SetupMatrices() {
  XMMATRIX matWorld;
  matWorld = XMMatrixIdentity();

  XMVECTOR vEyePt = {0.0f, 3.0f, -2.0f};
  XMVECTOR vLookatPt = {0.0f, 0.0f, 0.0f};
  XMVECTOR vUpVec = {0.0f, 1.0f, 0.0f};
  XMMATRIX matView;
  matView = XMMatrixLookAtLH(vEyePt, vLookatPt, vUpVec);

  XMMATRIX matProj;
  matProj = XMMatrixPerspectiveFovLH((float)XM_PI / 4.f, 1.0f, 0.01f, 10.0f);

  g_pmWorld->SetMatrix((float *)&matWorld);
  g_pmView->SetMatrix((float *)&matView);
  g_pmProjection->SetMatrix((float *)&matProj);
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Draws the scene
//-----------------------------------------------------------------------------
VOID Render() {
  // Clear the backbuffer to a black color
  float ClearColor[4] = {0, 0, 0, 0};
  g_pd3dDevice->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

  // Run CUDA to update vertex positions
  runCuda();

  // Draw frame
  {
    // Setup the world, view, and projection matrices
    SetupMatrices();

    // Render the vertex buffer contents
    UINT stride = sizeof(CUSTOMVERTEX);
    UINT offset = 0;
    g_pd3dDevice->IASetVertexBuffers(0, 1, &g_pVB, &stride, &offset);

    g_pSimpleTechnique->GetPassByIndex(0)->Apply(0);
    g_pd3dDevice->Draw(g_NumVertices, 0);
  }

  // Present the backbuffer contents to the display
  g_pSwapChain->Present(0, 0);

  anim += 0.01f;
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
        Cleanup();
        PostQuitMessage(0);
        return 0;
      }
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
