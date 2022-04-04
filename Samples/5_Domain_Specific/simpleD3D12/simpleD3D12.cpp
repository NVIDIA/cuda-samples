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

#include <windows.h>

#include "d3dx12.h"

#include <string>
#include <wrl.h>
#include <shellapi.h>

#include <cuda_runtime.h>
#include "ShaderStructs.h"
#include "simpleD3D12.h"
#include <aclapi.h>

//////////////////////////////////////////////
// WindowsSecurityAttributes implementation //
//////////////////////////////////////////////

class WindowsSecurityAttributes {
 protected:
  SECURITY_ATTRIBUTES m_winSecurityAttributes;
  PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

 public:
  WindowsSecurityAttributes();
  ~WindowsSecurityAttributes();
  SECURITY_ATTRIBUTES *operator&();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
  m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
      1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
  assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

  PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor +
                         SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

  InitializeSecurityDescriptor(m_winPSecurityDescriptor,
                               SECURITY_DESCRIPTOR_REVISION);

  SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
      SECURITY_WORLD_SID_AUTHORITY;
  AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
                           0, 0, 0, 0, 0, ppSID);

  EXPLICIT_ACCESS explicitAccess;
  ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
  explicitAccess.grfAccessPermissions =
      STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
  explicitAccess.grfAccessMode = SET_ACCESS;
  explicitAccess.grfInheritance = INHERIT_ONLY;
  explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
  explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

  SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

  SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

  m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
  m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
  m_winSecurityAttributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
  PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor +
                         SECURITY_DESCRIPTOR_MIN_LENGTH);
  PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

  if (*ppSID) {
    FreeSid(*ppSID);
  }
  if (*ppACL) {
    LocalFree(*ppACL);
  }
  free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
  return &m_winSecurityAttributes;
}

DX12CudaInterop::DX12CudaInterop(UINT width, UINT height, std::string name)
    : DX12CudaSample(width, height, name),
      m_frameIndex(0),
      m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
      m_fenceValues{},
      m_rtvDescriptorSize(0) {
  m_viewport = {0.0f, 0.0f, static_cast<float>(width),
                static_cast<float>(height)};
  m_AnimTime = 1.0f;
}

void DX12CudaInterop::OnInit() {
  LoadPipeline();
  InitCuda();
  LoadAssets();
}

// Load the rendering pipeline dependencies.
void DX12CudaInterop::LoadPipeline() {
  UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
  // Enable the debug layer (requires the Graphics Tools "optional feature").
  // NOTE: Enabling the debug layer after device creation will invalidate the
  // active device.
  {
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
      debugController->EnableDebugLayer();

      // Enable additional debug layers.
      dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
  }
#endif

  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  if (m_useWarpDevice) {
    ComPtr<IDXGIAdapter> warpAdapter;
    ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

    ThrowIfFailed(D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&m_device)));
  } else {
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(),
                                    D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(&m_device)));
    DXGI_ADAPTER_DESC1 desc;
    hardwareAdapter->GetDesc1(&desc);
    m_dx12deviceluid = desc.AdapterLuid;
  }

  // Describe and create the command queue.
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

  ThrowIfFailed(
      m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

  // Describe and create the swap chain.
  DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
  swapChainDesc.BufferCount = FrameCount;
  swapChainDesc.Width = m_width;
  swapChainDesc.Height = m_height;
  swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapChainDesc.SampleDesc.Count = 1;

  ComPtr<IDXGISwapChain1> swapChain;
  ThrowIfFailed(factory->CreateSwapChainForHwnd(
      m_commandQueue.Get(),  // Swap chain needs the queue so that it can force
                             // a flush on it.
      Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr,
      &swapChain));

  // This sample does not support fullscreen transitions.
  ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(),
                                               DXGI_MWA_NO_ALT_ENTER));

  ThrowIfFailed(swapChain.As(&m_swapChain));
  m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

  // Create descriptor heaps.
  {
    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = FrameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(
        m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

    m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  }

  // Create frame resources.
  {
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
        m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

    // Create a RTV and a command allocator for each frame.
    for (UINT n = 0; n < FrameCount; n++) {
      ThrowIfFailed(
          m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
      m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr,
                                       rtvHandle);
      rtvHandle.Offset(1, m_rtvDescriptorSize);

      ThrowIfFailed(m_device->CreateCommandAllocator(
          D3D12_COMMAND_LIST_TYPE_DIRECT,
          IID_PPV_ARGS(&m_commandAllocators[n])));
    }
  }
}

void DX12CudaInterop::InitCuda() {
  int num_cuda_devices = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

  if (!num_cuda_devices) {
    throw std::exception("No CUDA Devices found");
  }
  for (UINT devId = 0; devId < num_cuda_devices; devId++) {
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));

    if ((memcmp(&m_dx12deviceluid.LowPart, devProp.luid,
                sizeof(m_dx12deviceluid.LowPart)) == 0) &&
        (memcmp(&m_dx12deviceluid.HighPart,
                devProp.luid + sizeof(m_dx12deviceluid.LowPart),
                sizeof(m_dx12deviceluid.HighPart)) == 0)) {
      checkCudaErrors(cudaSetDevice(devId));
      m_cudaDeviceID = devId;
      m_nodeMask = devProp.luidDeviceNodeMask;
      checkCudaErrors(cudaStreamCreate(&m_streamToRun));
      printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
      break;
    }
  }
}
// Load the sample assets.
void DX12CudaInterop::LoadAssets() {
  // Create a root signature.
  {
    CD3DX12_DESCRIPTOR_RANGE range;
    CD3DX12_ROOT_PARAMETER parameter;

    range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
    parameter.InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_VERTEX);

    D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        // Only the input assembler stage needs access to the constant buffer.
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;

    CD3DX12_ROOT_SIGNATURE_DESC descRootSignature;
    descRootSignature.Init(1, &parameter, 0, nullptr, rootSignatureFlags);
    ComPtr<ID3DBlob> pSignature;
    ComPtr<ID3DBlob> pError;
    ThrowIfFailed(D3D12SerializeRootSignature(
        &descRootSignature, D3D_ROOT_SIGNATURE_VERSION_1,
        pSignature.GetAddressOf(), pError.GetAddressOf()));
    ThrowIfFailed(m_device->CreateRootSignature(
        0, pSignature->GetBufferPointer(), pSignature->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSignature)));
  }
  // Create the pipeline state, which includes compiling and loading shaders.
  {
    ComPtr<ID3DBlob> vertexShader;
    ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif
    std::wstring filePath = GetAssetFullPath("shaders.hlsl");
    LPCWSTR result = filePath.c_str();
    ThrowIfFailed(D3DCompileFromFile(result, nullptr, nullptr, "VSMain",
                                     "vs_5_0", compileFlags, 0, &vertexShader,
                                     nullptr));
    ThrowIfFailed(D3DCompileFromFile(result, nullptr, nullptr, "PSMain",
                                     "ps_5_0", compileFlags, 0, &pixelShader,
                                     nullptr));

    // Define the vertex input layout.
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

    // Describe and create the graphics pipeline state object (PSO).
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = {inputElementDescs, _countof(inputElementDescs)};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    ThrowIfFailed(m_device->CreateGraphicsPipelineState(
        &psoDesc, IID_PPV_ARGS(&m_pipelineState)));
  }

  // Create the command list.
  ThrowIfFailed(m_device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_DIRECT,
      m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get(),
      IID_PPV_ARGS(&m_commandList)));

  // Command lists are created in the recording state, but there is nothing
  // to record yet. The main loop expects it to be closed, so close it now.
  ThrowIfFailed(m_commandList->Close());

  // Create the vertex buffer.
  {
    // Define the geometry for a triangle.
    vertBufWidth = m_width / 2;
    vertBufHeight = m_height / 2;
    const UINT vertexBufferSize = sizeof(Vertex) * vertBufWidth * vertBufHeight;

    ThrowIfFailed(m_device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_SHARED,
        &CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, nullptr,
        IID_PPV_ARGS(&m_vertexBuffer)));

    // Initialize the vertex buffer view.
    m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
    m_vertexBufferView.StrideInBytes = sizeof(Vertex);
    m_vertexBufferView.SizeInBytes = vertexBufferSize;

    HANDLE sharedHandle;
    WindowsSecurityAttributes windowsSecurityAttributes;
    LPCWSTR name = NULL;
    ThrowIfFailed(m_device->CreateSharedHandle(
        m_vertexBuffer.Get(), &windowsSecurityAttributes, GENERIC_ALL, name,
        &sharedHandle));

    D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
    d3d12ResourceAllocationInfo = m_device->GetResourceAllocationInfo(
        m_nodeMask, 1, &CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize));
    size_t actualSize = d3d12ResourceAllocationInfo.SizeInBytes;
    size_t alignment = d3d12ResourceAllocationInfo.Alignment;

    cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
    memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
    externalMemoryHandleDesc.size = actualSize;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

    checkCudaErrors(
        cudaImportExternalMemory(&m_externalMemory, &externalMemoryHandleDesc));
    CloseHandle(sharedHandle);

    cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
    memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
    externalMemoryBufferDesc.offset = 0;
    externalMemoryBufferDesc.size = vertexBufferSize;
    externalMemoryBufferDesc.flags = 0;

    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
        &m_cudaDevVertptr, m_externalMemory, &externalMemoryBufferDesc));
    RunSineWaveKernel(vertBufWidth, vertBufHeight, (Vertex *)m_cudaDevVertptr,
                      m_streamToRun, 1.0f);
    checkCudaErrors(cudaStreamSynchronize(m_streamToRun));
    
  }

  // Create synchronization objects and wait until assets have been uploaded to
  // the GPU.
  {
    ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex],
                                        D3D12_FENCE_FLAG_SHARED,
                                        IID_PPV_ARGS(&m_fence)));

    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;

    memset(&externalSemaphoreHandleDesc, 0,
           sizeof(externalSemaphoreHandleDesc));
    WindowsSecurityAttributes windowsSecurityAttributes;
    LPCWSTR name = NULL;
    HANDLE sharedHandle;
    externalSemaphoreHandleDesc.type =
        cudaExternalSemaphoreHandleTypeD3D12Fence;
    m_device->CreateSharedHandle(m_fence.Get(), &windowsSecurityAttributes,
                                 GENERIC_ALL, name, &sharedHandle);
    externalSemaphoreHandleDesc.handle.win32.handle = (void *)sharedHandle;
    externalSemaphoreHandleDesc.flags = 0;

    checkCudaErrors(cudaImportExternalSemaphore(&m_externalSemaphore,
                                                &externalSemaphoreHandleDesc));

    m_fenceValues[m_frameIndex]++;

    // Create an event handle to use for frame synchronization.
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (m_fenceEvent == nullptr) {
      ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
    }

    // Wait for the command list to execute; we are reusing the same command
    // list in our main loop but for now, we just want to wait for setup to
    // complete before continuing.
    WaitForGpu();
  }
}

// Render the scene.
void DX12CudaInterop::OnRender() {
  // Record all the commands we need to render the scene into the command list.
  PopulateCommandList();

  // Execute the command list.
  ID3D12CommandList *ppCommandLists[] = {m_commandList.Get()};
  m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // Present the frame.
  ThrowIfFailed(m_swapChain->Present(1, 0));

  // Schedule a Signal command in the queue.
  const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
  ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

  MoveToNextFrame();
}

void DX12CudaInterop::OnDestroy() {
  // Ensure that the GPU is no longer referencing resources that are about to be
  // cleaned up by the destructor.
  WaitForGpu();
  checkCudaErrors(cudaDestroyExternalSemaphore(m_externalSemaphore));
  checkCudaErrors(cudaDestroyExternalMemory(m_externalMemory));
  checkCudaErrors(cudaFree(m_cudaDevVertptr));
  CloseHandle(m_fenceEvent);
}

void DX12CudaInterop::PopulateCommandList() {
  // Command list allocators can only be reset when the associated
  // command lists have finished execution on the GPU; apps should use
  // fences to determine GPU execution progress.
  ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

  // However, when ExecuteCommandList() is called on a particular command
  // list, that command list can then be reset at any time and must be before
  // re-recording.
  ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(),
                                     m_pipelineState.Get()));

  m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());

  // Set necessary state.
  m_commandList->RSSetViewports(1, &m_viewport);
  m_commandList->RSSetScissorRects(1, &m_scissorRect);

  // Indicate that the back buffer will be used as a render target.
  m_commandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT,
             D3D12_RESOURCE_STATE_RENDER_TARGET));

  CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
      m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex,
      m_rtvDescriptorSize);
  m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

  // Record commands.
  const float clearColor[] = {0.0f, 0.2f, 0.4f, 1.0f};
  m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
  m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
  m_commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
  m_commandList->DrawInstanced(vertBufHeight * vertBufWidth, 1, 0, 0);

  // Indicate that the back buffer will now be used to present.
  m_commandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             m_renderTargets[m_frameIndex].Get(),
             D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

  ThrowIfFailed(m_commandList->Close());
}

// Wait for pending GPU work to complete.
void DX12CudaInterop::WaitForGpu() {
  // Schedule a Signal command in the queue.
  ThrowIfFailed(
      m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

  // Wait until the fence has been processed.
  ThrowIfFailed(
      m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
  WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

  // Increment the fence value for the current frame.
  m_fenceValues[m_frameIndex]++;
}

// Prepare to render the next frame.
void DX12CudaInterop::MoveToNextFrame() {
  const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
  cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
  memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

  externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
  externalSemaphoreWaitParams.flags = 0;

  checkCudaErrors(cudaWaitExternalSemaphoresAsync(
      &m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun));

  m_AnimTime += 0.01f;
  RunSineWaveKernel(vertBufWidth, vertBufHeight, (Vertex *)m_cudaDevVertptr,
                    m_streamToRun, m_AnimTime);

  cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
  memset(&externalSemaphoreSignalParams, 0,
         sizeof(externalSemaphoreSignalParams));
  m_fenceValues[m_frameIndex] = currentFenceValue + 1;
  externalSemaphoreSignalParams.params.fence.value =
      m_fenceValues[m_frameIndex];
  externalSemaphoreSignalParams.flags = 0;

  checkCudaErrors(cudaSignalExternalSemaphoresAsync(
      &m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));

  // Update the frame index.
  m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

  // If the next frame is not ready to be rendered yet, wait until it is ready.
  if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex]) {
    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex],
                                                m_fenceEvent));
    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
  }

  // Set the fence value for the next frame.
  m_fenceValues[m_frameIndex] = currentFenceValue + 2;
}
