/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

/* This example demonstrates how to use the CUDA-D3D11 External Resource Interoperability APIs
*  to update D3D11 buffers from CUDA and synchronize between D3D11 and CUDA with Keyed Mutexes.
 */

#pragma warning(disable: 4312)

#include <windows.h>
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <dxgi1_2.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>

// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include "ShaderStructs.h"
#include "sinewave_cuda.h"

#define MAX_EPSILON 10

static char *SDK_name = "simpleD3D11";

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter1          *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D11Device           *g_pd3dDevice = NULL; // Our rendering device
ID3D11DeviceContext    *g_pd3dDeviceContext = NULL;
IDXGISwapChain         *g_pSwapChain = NULL; // The swap chain of the window
ID3D11RenderTargetView *g_pSwapChainRTV = NULL; //The Render target view on the swap chain ( used for clear)
ID3D11RasterizerState  *g_pRasterState = NULL;
ID3D11InputLayout      *g_pInputLayout = NULL;
ID3D11VertexShader     *g_pVertexShader;
ID3D11PixelShader      *g_pPixelShader;
ID3D11InputLayout      *g_pLayout;
ID3D11Buffer           *g_VertexBuffer;
IDXGIKeyedMutex        *g_pKeyedMutex11;

Vertex *d_VertexBufPtr = NULL;
cudaExternalMemory_t extMemory;
cudaExternalSemaphore_t extSemaphore;

//
// Vertex and Pixel shaders here : VSMain() & PSMain()
//
static const char g_simpleShaders[] =
"struct PSInput\n" \
"{ \n" \
"    float4 position : SV_POSITION;\n" \
"    float4 color : COLOR; \n" \
"};\n" \
"PSInput VSMain(float3 position : POSITION, float4 color : COLOR)\n" \
"{ \n" \
"    PSInput result;\n" \
"    result.position = float4(position, 1.0f); \n" \
"    // Pass the color through without modification. \n" \
"    result.color = color; \n" \
"    return result; \n" \
"} \n" \
"float4 PSMain(PSInput input) : SV_TARGET \n" \
"{ \n" \
"    return input.color; \n" \
"} \n" \
;

// testing/tracing function used pervasively in tests.  if the condition is unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }

bool g_bDone   = false;
bool g_bPassed = true;

int *pArgc = NULL;
char **pArgv = NULL;

const unsigned int g_WindowWidth = 720;
const unsigned int g_WindowHeight = 720;

int g_iFrameToCompare = 10;

cudaStream_t cuda_stream;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);

bool DrawScene();
void Cleanup();
void Render();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#define NAME_LEN    512

bool findCUDADevice()
{
    int deviceCount = 0;
    // This function call returns 0 if there are no CUDA capable devices.
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    return true;
}

bool findDXDevice(char *dev_name)
{
    HRESULT hr = S_OK;
    cudaError cuStatus;
    int cuda_dev = -1;

    // Iterate through the candidate adapters
    IDXGIFactory1 *pFactory;
    hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory1), (void **)(&pFactory));

    if (! SUCCEEDED(hr))
    {
        printf("> No DXGI Factory created.\n");
        return false;
    }

    UINT adapter = 0;

    for (; !g_pCudaCapableAdapter; ++adapter)
    {
        // Get a candidate DXGI adapter
        IDXGIAdapter1 *pAdapter = NULL;

        hr = pFactory->EnumAdapters1(adapter, &pAdapter);

        if (FAILED(hr))
        {
            break;    // no compatible adapters found
        }

        // Query to see if there exists a corresponding compute device
        int cuDevice;
        cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
        printLastCudaError("cudaD3D11GetDevice failed"); //This prints and resets the cudaError to cudaSuccess

        if (cudaSuccess == cuStatus)
        {
            // If so, mark it as the one against which to create our d3d11 device
            g_pCudaCapableAdapter = pAdapter;
            g_pCudaCapableAdapter->AddRef();
            cuda_dev = cuDevice;
            printf("\ncuda device id selected = %d\n", cuda_dev);
        }

        pAdapter->Release();
    }

    printf("> Found %d D3D11 Adapater(s).\n", (int) adapter);

    pFactory->Release();

    if (!g_pCudaCapableAdapter)
    {
        printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
        return false;
    }

    DXGI_ADAPTER_DESC adapterDesc;
    g_pCudaCapableAdapter->GetDesc(&adapterDesc);
    wcstombs(dev_name, adapterDesc.Description, 128);

    checkCudaErrors(cudaSetDevice(cuda_dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

    printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
    printf("> %s\n", dev_name);

    return true;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    char device_name[256];
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

    printf("[%s] - Starting...\n", SDK_name);

    if (!findCUDADevice())                   // Search for CUDA GPU
    {
        printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
        exit(EXIT_SUCCESS);
    }

    if (!dynlinkLoadD3D11API())                  // Search for D3D API (locate drivers, does not mean device is found)
    {
        printf("> D3D11 API libraries NOT found on.. Exiting.\n");
        dynlinkUnloadD3D11API();
        exit(EXIT_SUCCESS);
    }

    if (!findDXDevice(device_name))           // Search for D3D Hardware Device
    {
        printf("> D3D11 Graphics Device NOT found.. Exiting.\n");
        dynlinkUnloadD3D11API();
        exit(EXIT_SUCCESS);
    }

    // command line options
    if (argc > 1)
    {
        // automatied build testing harness
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
            getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
    }

    //
    // create window
    //
    // Register the window class
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      "CUDA SDK", NULL
                    };
    RegisterClassEx(&wc);

    // Create the application's window
    int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
    int yMenu = ::GetSystemMetrics(SM_CYMENU);
    int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
    HWND hWnd = CreateWindow(wc.lpszClassName, "CUDA/D3D11 InterOP",
                             WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth + 2*xBorder, g_WindowHeight+ 2*yBorder+yMenu,
                             NULL, NULL, wc.hInstance, NULL);

    ShowWindow(hWnd, SW_SHOWDEFAULT);
    UpdateWindow(hWnd);

    // Initialize Direct3D
    if (!SUCCEEDED(InitD3D(hWnd)))
    {
        printf("InitD3D Failed.. Exiting..\n");
        exit(EXIT_FAILURE);
    }

    //
    // the main loop
    //
    while (false == g_bDone)
    {
        Render();

        //
        // handle I/O
        //
        MSG msg;
        ZeroMemory(&msg, sizeof(msg));

        while (msg.message!=WM_QUIT)
        {
            if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            else
            {
                Render();

                if (ref_file)
                {
                    for (int count=0; count<g_iFrameToCompare; count++)
                    {
                        Render();
                    }

                    const char *cur_image_path = "simpleD3D11.ppm";

                    // Save a reference of our current test run image
                    CheckRenderD3D11::ActiveRenderTargetToPPM(g_pd3dDevice,cur_image_path);

                    // compare to offical reference image, printing PASS or FAIL.
                    g_bPassed = CheckRenderD3D11::PPMvsPPM(cur_image_path, ref_file, argv[0], MAX_EPSILON, 0.15f);

                    g_bDone = true;

                    Cleanup();

                    PostQuitMessage(0);
                }
                else
                {
                    g_bPassed = true;
                }
            }
        }

    };

    // Release D3D Library (after message loop)
    dynlinkUnloadD3D11API();

    // Unregister windows class
    UnregisterClass(wc.lpszClassName, wc.hInstance);

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
HRESULT InitD3D(HWND hWnd)
{
    HRESULT hr = S_OK;
    cudaError cuStatus;

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

    D3D_FEATURE_LEVEL tour_fl[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    D3D_FEATURE_LEVEL flRes;
    // Create device and swapchain
    hr = sFnPtr_D3D11CreateDeviceAndSwapChain(
             g_pCudaCapableAdapter,
             D3D_DRIVER_TYPE_UNKNOWN,//D3D_DRIVER_TYPE_HARDWARE,
             NULL, //HMODULE Software
             0, //UINT Flags
             tour_fl, // D3D_FEATURE_LEVEL* pFeatureLevels
             2, //FeatureLevels
             D3D11_SDK_VERSION, //UINT SDKVersion
             &sd, // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
             &g_pSwapChain, //IDXGISwapChain** ppSwapChain
             &g_pd3dDevice, //ID3D11Device** ppDevice
             &flRes, //D3D_FEATURE_LEVEL* pFeatureLevel
             &g_pd3dDeviceContext//ID3D11DeviceContext** ppImmediateContext
         );

    AssertOrQuit(SUCCEEDED(hr));

    g_pCudaCapableAdapter->Release();

    // Get the immediate DeviceContext
    g_pd3dDevice->GetImmediateContext(&g_pd3dDeviceContext);

    // Create a render target view of the swapchain
    ID3D11Texture2D *pBuffer;
    hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID *)&pBuffer);
    AssertOrQuit(SUCCEEDED(hr));

    hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
    AssertOrQuit(SUCCEEDED(hr));
    pBuffer->Release();

    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = g_WindowWidth;
    vp.Height = g_WindowHeight;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pd3dDeviceContext->RSSetViewports(1, &vp);


    ID3DBlob *VS;
    ID3DBlob *PS;
    ID3DBlob *pErrorMsgs;
    // Vertex shader
    {
        hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL, NULL,"VSMain", "vs_4_0", 0/*Flags1*/, 0/*Flags2*/, &VS, &pErrorMsgs);

        if (FAILED(hr))
        {
            const char *pStr = (const char *)pErrorMsgs->GetBufferPointer();
            printf(pStr);
        }

        AssertOrQuit(SUCCEEDED(hr));
        hr = g_pd3dDevice->CreateVertexShader(VS->GetBufferPointer(), VS->GetBufferSize(), NULL, &g_pVertexShader);
        AssertOrQuit(SUCCEEDED(hr));
        // Let's bind it now : no other vtx shader will replace it...
        g_pd3dDeviceContext->VSSetShader(g_pVertexShader, NULL, 0);
    }
    // Pixel shader
    {
        hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL, NULL, "PSMain", "ps_4_0", 0/*Flags1*/, 0/*Flags2*/, &PS, &pErrorMsgs);

        AssertOrQuit(SUCCEEDED(hr));
        hr = g_pd3dDevice->CreatePixelShader(PS->GetBufferPointer(), PS->GetBufferSize(), NULL, &g_pPixelShader);
        AssertOrQuit(SUCCEEDED(hr));
        // Let's bind it now : no other pix shader will replace it...
        g_pd3dDeviceContext->PSSetShader(g_pPixelShader, NULL, 0);
    }

    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(Vertex) * g_WindowWidth * g_WindowHeight;
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    hr = g_pd3dDevice->CreateBuffer(&bufferDesc, NULL, &g_VertexBuffer);
    AssertOrQuit(SUCCEEDED(hr));

    hr = g_VertexBuffer->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&g_pKeyedMutex11);

    AssertOrQuit(SUCCEEDED(hr));

    D3D11_INPUT_ELEMENT_DESC inputElementDescs[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };

    hr =  g_pd3dDevice->CreateInputLayout(inputElementDescs, 2, VS->GetBufferPointer(), VS->GetBufferSize(), &g_pLayout);
    AssertOrQuit(SUCCEEDED(hr));
    // Setup  Input Layout
    g_pd3dDeviceContext->IASetInputLayout(g_pLayout);
    AssertOrQuit(SUCCEEDED(hr));
    g_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
    AssertOrQuit(SUCCEEDED(hr));

    IDXGIResource1* pResource;
    HANDLE sharedHandle;
    g_VertexBuffer->QueryInterface(__uuidof(IDXGIResource1), (void**)&pResource);
    hr = pResource->GetSharedHandle(&sharedHandle);
    if (!SUCCEEDED(hr))
    {
        std::cout << "Failed GetSharedHandle hr= " << hr << std::endl;
    }
    // Import the D3D11 Vertex Buffer into CUDA
    d_VertexBufPtr = cudaImportVertexBuffer(sharedHandle, extMemory, g_WindowWidth, g_WindowHeight);
    pResource->Release();

    g_pKeyedMutex11->QueryInterface(__uuidof(IDXGIResource1), (void**)&pResource);
    pResource->GetSharedHandle(&sharedHandle);
    // Import the D3D11 Keyed Mutex into CUDA
    cudaImportKeyedMutex(sharedHandle, extSemaphore);
    pResource->Release();

    D3D11_RASTERIZER_DESC rasterizerState;
    rasterizerState.FillMode = D3D11_FILL_SOLID;
    rasterizerState.CullMode = D3D11_CULL_FRONT;
    rasterizerState.FrontCounterClockwise = false;
    rasterizerState.DepthBias = false;
    rasterizerState.DepthBiasClamp = 0;
    rasterizerState.SlopeScaledDepthBias = 0;
    rasterizerState.DepthClipEnable = false;
    rasterizerState.ScissorEnable = false;
    rasterizerState.MultisampleEnable = false;
    rasterizerState.AntialiasedLineEnable = false;
    g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
    g_pd3dDeviceContext->RSSetState(g_pRasterState);

    return S_OK;
}


////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
bool DrawScene(uint64_t &key)
{

    HRESULT hr = S_OK;
    // Clear the backbuffer
    float ClearColor[4] = { 0.5f, 0.5f, 0.6f, 1.0f };

    g_pd3dDeviceContext->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

    hr = g_pKeyedMutex11->AcquireSync(key++, INFINITE);
    AssertOrQuit(SUCCEEDED(hr));
    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    g_pd3dDeviceContext->IASetVertexBuffers(0, 1, &g_VertexBuffer, &stride, &offset);
    g_pd3dDeviceContext->Draw(g_WindowHeight*g_WindowWidth, 0);
    hr = g_pKeyedMutex11->ReleaseSync(key);
    AssertOrQuit(SUCCEEDED(hr));

    // Present the backbuffer contents to the display
    g_pSwapChain->Present(0, 0);

    return true;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup()
{
    checkCudaErrors(cudaFree(d_VertexBufPtr));
    checkCudaErrors(cudaDestroyExternalMemory(extMemory));
    checkCudaErrors(cudaDestroyExternalSemaphore(extSemaphore));
    //
    // clean up Direct3D
    //
    // release the resources we created
    if (g_pInputLayout != NULL)
    {
        g_pInputLayout->Release();
    }

    if (g_pVertexShader)
    {
        g_pVertexShader->Release();
    }

    if (g_pPixelShader)
    {
        g_pPixelShader->Release();
    }

    if (g_VertexBuffer)
    {
        g_VertexBuffer->Release();
    }

    if (g_pSwapChainRTV != NULL)
    {
        g_pSwapChainRTV->Release();
    }

    if (g_pSwapChain != NULL)
    {
        g_pSwapChain->Release();
    }

    if (g_pd3dDevice != NULL)
    {
        g_pd3dDevice->Release();
    }
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the CUDA kernels to fill in the vertex buffer
//-----------------------------------------------------------------------------
void Render()
{
    static uint64_t key = 0;

    // Launch cuda kernel to generate sinewave in vertex buffer
    RunSineWaveKernel(extSemaphore, key, INFINITE, g_WindowWidth, g_WindowWidth, d_VertexBufPtr, cuda_stream);

    // Draw the scene using them
    DrawScene(key);
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
        case WM_KEYDOWN:
            if (wParam==VK_ESCAPE)
            {
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

