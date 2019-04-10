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

#pragma once

#include "DX12CudaSample.h"
#include "ShaderStructs.h"

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the
// CPU, it has no understanding of the lifetime of resources on the GPU. Apps
// must account for the GPU lifetime of resources to avoid destroying objects
// that may still be referenced by the GPU. An example of this can be found in
// the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

static const char *shaderstr =
    " struct PSInput \n"
    " { \n"
    "  float4 position : SV_POSITION; \n"
    "  float4 color : COLOR; \n"
    " } \n"
    " PSInput VSMain(float3 position : POSITION, float4 color : COLOR) \n"
    " { \n"
    "  PSInput result;\n"
    "  result.position = float4(position, 1.0f);\n"
    "  result.color = color;\n"
    "  return result; \n"
    " } \n"
    " float4 PSMain(PSInput input) : SV_TARGET \n"
    " { \n"
    "  return input.color;\n"
    " } \n";

class DX12CudaInterop : public DX12CudaSample {
 public:
  DX12CudaInterop(UINT width, UINT height, std::string name);

  virtual void OnInit();
  virtual void OnRender();
  virtual void OnDestroy();

 private:
  // In this sample we overload the meaning of FrameCount to mean both the
  // maximum number of frames that will be queued to the GPU at a time, as well
  // as the number of back buffers in the DXGI swap chain. For the majority of
  // applications, this is convenient and works well. However, there will be
  // certain cases where an application may want to queue up more frames than
  // there are back buffers available. It should be noted that excessive
  // buffering of frames dependent on user input may result in noticeable
  // latency in your app.
  static const UINT FrameCount = 2;
  std::string shadersSrc = shaderstr;
#if 0
		" struct PSInput \n" \
		" { \n" \
		"  float4 position : SV_POSITION; \n" \
		"  float4 color : COLOR; \n" \
		" } \n" \
		" PSInput VSMain(float3 position : POSITION, float4 color : COLOR) \n" \
		" { \n" \
		"  PSInput result;\n" \
		"  result.position = float4(position, 1.0f);\n" \
		"  result.color = color;\n"	\
		"  return result; \n" \
		" } \n" \
		" float4 PSMain(PSInput input) : SV_TARGET \n" \
		" { \n" \
		"  return input.color;\n" \
		" } \n";
#endif

  // Vertex Buffer dimension
  size_t vertBufHeight, vertBufWidth;

  // Pipeline objects.
  D3D12_VIEWPORT m_viewport;
  CD3DX12_RECT m_scissorRect;
  ComPtr<IDXGISwapChain3> m_swapChain;
  ComPtr<ID3D12Device> m_device;
  ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
  ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
  ComPtr<ID3D12CommandQueue> m_commandQueue;
  ComPtr<ID3D12RootSignature> m_rootSignature;
  ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
  ComPtr<ID3D12PipelineState> m_pipelineState;
  ComPtr<ID3D12GraphicsCommandList> m_commandList;
  UINT m_rtvDescriptorSize;

  // App resources.
  ComPtr<ID3D12Resource> m_vertexBuffer;
  D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

  // Synchronization objects.
  UINT m_frameIndex;
  HANDLE m_fenceEvent;
  ComPtr<ID3D12Fence> m_fence;
  UINT64 m_fenceValues[FrameCount];

  // CUDA objects
  cudaExternalMemoryHandleType m_externalMemoryHandleType;
  cudaExternalMemory_t m_externalMemory;
  cudaExternalSemaphore_t m_externalSemaphore;
  cudaStream_t m_streamToRun;
  LUID m_dx12deviceluid;
  UINT m_cudaDeviceID;
  UINT m_nodeMask;
  float m_AnimTime;
  void *m_cudaDevVertptr = NULL;

  void LoadPipeline();
  void InitCuda();
  void LoadAssets();
  void PopulateCommandList();
  void MoveToNextFrame();
  void WaitForGpu();
};
