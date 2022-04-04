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

/*
 * This sample demonstrates CUDA Interop with Vulkan using cuMemMap APIs.
 * Allocating device memory and updating values in those allocations are
 * performed by CUDA and the contents of the allocation are visualized by
 * Vulkan.
 */

#include "VulkanBaseApp.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "MonteCarloPi.h"
#include <helper_cuda.h>
#include <cuda.h>

#include "helper_multiprocess.h"

//#define DEBUG
#ifndef DEBUG
#define ENABLE_VALIDATION (false)
#else
#define ENABLE_VALIDATION (true)
#endif

#define NUM_SIMULATION_POINTS 50000

std::string execution_path;

class VulkanCudaPi : public VulkanBaseApp {
  typedef struct UniformBufferObject_st { float frame; } UniformBufferObject;

  VkBuffer m_inCircleBuffer, m_xyPositionBuffer;
  VkDeviceMemory m_inCircleMemory, m_xyPositionMemory;
  VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
  MonteCarloPiSimulation m_sim;
  UniformBufferObject m_ubo;
  cudaStream_t m_stream;
  cudaExternalSemaphore_t m_cudaWaitSemaphore, m_cudaSignalSemaphore;
  using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  chrono_tp m_lastTime;
  size_t m_lastFrame;

 public:
  VulkanCudaPi(size_t num_points)
      : VulkanBaseApp("simpleVulkanMMAP", ENABLE_VALIDATION),
        m_inCircleBuffer(VK_NULL_HANDLE),
        m_xyPositionBuffer(VK_NULL_HANDLE),
        m_inCircleMemory(VK_NULL_HANDLE),
        m_xyPositionMemory(VK_NULL_HANDLE),
        m_sim(num_points),
        m_ubo(),
        m_stream(0),
        m_vkWaitSemaphore(VK_NULL_HANDLE),
        m_vkSignalSemaphore(VK_NULL_HANDLE),
        m_cudaWaitSemaphore(),
        m_cudaSignalSemaphore(),
        m_lastFrame(0) {
    // Add our compiled vulkan shader files
    char* vertex_shader_path =
        sdkFindFilePath("vert.spv", execution_path.c_str());
    char* fragment_shader_path =
        sdkFindFilePath("frag.spv", execution_path.c_str());
    m_shaderFiles.push_back(
        std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path));
    m_shaderFiles.push_back(
        std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path));
  }

  ~VulkanCudaPi() {
    if (m_stream) {
      // Make sure there's no pending work before we start tearing down
      checkCudaErrors(cudaStreamSynchronize(m_stream));
      checkCudaErrors(cudaStreamDestroy(m_stream));
    }

    if (m_vkSignalSemaphore != VK_NULL_HANDLE) {
      checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaSignalSemaphore));
      vkDestroySemaphore(m_device, m_vkSignalSemaphore, nullptr);
    }
    if (m_vkWaitSemaphore != VK_NULL_HANDLE) {
      checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaWaitSemaphore));
      vkDestroySemaphore(m_device, m_vkWaitSemaphore, nullptr);
    }
    if (m_xyPositionBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_xyPositionBuffer, nullptr);
    }
    if (m_xyPositionMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_xyPositionMemory, nullptr);
    }
    if (m_inCircleBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_inCircleBuffer, nullptr);
    }
    if (m_inCircleMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_inCircleMemory, nullptr);
    }
  }

  void fillRenderingCommandBuffer(VkCommandBuffer& commandBuffer) {
    VkBuffer vertexBuffers[] = {m_inCircleBuffer, m_xyPositionBuffer};
    VkDeviceSize offsets[] = {0, 0};
    vkCmdBindVertexBuffers(commandBuffer, 0,
                           sizeof(vertexBuffers) / sizeof(vertexBuffers[0]),
                           vertexBuffers, offsets);
    vkCmdDraw(commandBuffer, (uint32_t)(m_sim.getNumPoints()), 1, 0, 0);
  }

  void getVertexDescriptions(
      std::vector<VkVertexInputBindingDescription>& bindingDesc,
      std::vector<VkVertexInputAttributeDescription>& attribDesc) {
    bindingDesc.resize(2);
    attribDesc.resize(2);

    bindingDesc[0].binding = 0;
    bindingDesc[0].stride = sizeof(float);
    bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    bindingDesc[1].binding = 1;
    bindingDesc[1].stride = sizeof(vec2);
    bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    attribDesc[0].binding = 0;
    attribDesc[0].location = 0;
    attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
    attribDesc[0].offset = 0;

    attribDesc[1].binding = 1;
    attribDesc[1].location = 1;
    attribDesc[1].format = VK_FORMAT_R32G32_SFLOAT;
    attribDesc[1].offset = 0;
  }

  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info) {
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    info.primitiveRestartEnable = VK_FALSE;
  }

  void getWaitFrameSemaphores(
      std::vector<VkSemaphore>& wait,
      std::vector<VkPipelineStageFlags>& waitStages) const {
    if (m_currentFrame != 0) {
      // Have vulkan wait until cuda is done with the vertex buffer before
      // rendering
      // We don't do this on the first frame, as the wait semaphore hasn't been
      // initialized yet
      wait.push_back(m_vkWaitSemaphore);
      // We want to wait until all the pipeline commands are complete before
      // letting cuda work
      waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
  }

  void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const {
    // Add this semaphore for vulkan to signal once the vertex buffer is ready
    // for cuda to modify
    signal.push_back(m_vkSignalSemaphore);
  }

  void initVulkanApp() {
    const size_t nVerts = m_sim.getNumPoints();

    // Obtain cuda device id for the device corresponding to the Vulkan physical
    // device
    int deviceCount;
    int cudaDevice = cudaInvalidDeviceId;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    for (int dev = 0; dev < deviceCount; ++dev) {
      cudaDeviceProp devProp = {};
      checkCudaErrors(cudaGetDeviceProperties(&devProp, dev));
      if (isVkPhysicalDeviceUuid(&devProp.uuid)) {
        cudaDevice = dev;
        break;
      }
    }
    if (cudaDevice == cudaInvalidDeviceId) {
      throw std::runtime_error("No Suitable device found!");
    }

    // On the corresponding cuda device, create the cuda stream we'll using
    checkCudaErrors(cudaSetDevice(cudaDevice));
    checkCudaErrors(
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
    m_sim.initSimulation(cudaDevice, m_stream);

    importExternalBuffer(
        (void*)(uintptr_t)m_sim.getPositionShareableHandle(),
        getDefaultMemHandleType(), nVerts * sizeof(vec2),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_xyPositionBuffer,
        m_xyPositionMemory);

    importExternalBuffer(
        (void*)(uintptr_t)m_sim.getInCircleShareableHandle(),
        getDefaultMemHandleType(), nVerts * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_inCircleBuffer,
        m_inCircleMemory);

    // Create the semaphore vulkan will signal when it's done with the vertex
    // buffer
    createExternalSemaphore(m_vkSignalSemaphore,
                            getDefaultSemaphoreHandleType());
    // Create the semaphore vulkan will wait for before using the vertex buffer
    createExternalSemaphore(m_vkWaitSemaphore, getDefaultSemaphoreHandleType());
    // Import the semaphore cuda will use -- vulkan's signal will be cuda's wait
    importCudaExternalSemaphore(m_cudaWaitSemaphore, m_vkSignalSemaphore,
                                getDefaultSemaphoreHandleType());
    // Import the semaphore cuda will use -- cuda's signal will be vulkan's wait
    importCudaExternalSemaphore(m_cudaSignalSemaphore, m_vkWaitSemaphore,
                                getDefaultSemaphoreHandleType());
  }

  void importCudaExternalSemaphore(
      cudaExternalSemaphore_t& cudaSem, VkSemaphore& vkSem,
      VkExternalSemaphoreHandleTypeFlagBits handleType) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      externalSemaphoreHandleDesc.type =
          cudaExternalSemaphoreHandleTypeOpaqueWin32;
    } else if (handleType &
               VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      externalSemaphoreHandleDesc.type =
          cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      externalSemaphoreHandleDesc.type =
          cudaExternalSemaphoreHandleTypeOpaqueFd;
    } else {
      throw std::runtime_error("Unknown handle type requested!");
    }

#ifdef _WIN64
    externalSemaphoreHandleDesc.handle.win32.handle =
        (HANDLE)getSemaphoreHandle(vkSem, handleType);
#else
    externalSemaphoreHandleDesc.handle.fd =
        (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
#endif

    externalSemaphoreHandleDesc.flags = 0;

    checkCudaErrors(
        cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
  }

  VkDeviceSize getUniformSize() const { return sizeof(UniformBufferObject); }

  void updateUniformBuffer(uint32_t imageIndex, size_t globalFrame) {
    m_ubo.frame = (float)globalFrame;
    void* data;
    vkMapMemory(m_device, m_uniformMemory[imageIndex], 0, getUniformSize(), 0,
                &data);
    memcpy(data, &m_ubo, sizeof(m_ubo));
    vkUnmapMemory(m_device, m_uniformMemory[imageIndex]);
  }

  std::vector<const char*> getRequiredExtensions() const {
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    return extensions;
  }

  std::vector<const char*> getRequiredDeviceExtensions() const {
    std::vector<const char*> extensions;

    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif /* _WIN64 */
    return extensions;
  }

  void drawFrame() {
    static chrono_tp startTime = std::chrono::high_resolution_clock::now();

    chrono_tp currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    if (m_currentFrame == 0) {
      m_lastTime = startTime;
    }

    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = 0;

    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = 0;

    // Have vulkan draw the current frame...
    VulkanBaseApp::drawFrame();
    // Wait for vulkan to complete it's work
    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore,
                                                    &waitParams, 1, m_stream));
    // Now step the simulation
    m_sim.stepSimulation(time, m_stream);

    // Signal vulkan to continue with the updated buffers
    checkCudaErrors(cudaSignalExternalSemaphoresAsync(
        &m_cudaSignalSemaphore, &signalParams, 1, m_stream));
  }
};

int main(int argc, char** argv) {
  execution_path = argv[0];
  VulkanCudaPi app(NUM_SIMULATION_POINTS);
  app.init();
  app.mainLoop();
  return 0;
}
