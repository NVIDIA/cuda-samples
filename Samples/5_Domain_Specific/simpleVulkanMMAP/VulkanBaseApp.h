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

#pragma once
#ifndef __VULKANBASEAPP_H__
#define __VULKANBASEAPP_H__

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif /* _WIN64 */

struct GLFWwindow;

class VulkanBaseApp {
 public:
  VulkanBaseApp(const std::string& appName, bool enableValidation = false);
  static VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType();
  static VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType();
  virtual ~VulkanBaseApp();
  void init();
  void* getMemHandle(VkDeviceMemory memory,
                     VkExternalMemoryHandleTypeFlagBits handleType);
  void* getSemaphoreHandle(VkSemaphore semaphore,
                           VkExternalSemaphoreHandleTypeFlagBits handleType);
  bool isVkPhysicalDeviceUuid(void* Uuid);
  void createExternalSemaphore(
      VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer& buffer,
                    VkDeviceMemory& bufferMemory);
  void createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties,
                            VkExternalMemoryHandleTypeFlagsKHR extMemHandleType,
                            VkBuffer& buffer, VkDeviceMemory& bufferMemory);
  void importExternalBuffer(void* handle,
                            VkExternalMemoryHandleTypeFlagBits handleType,
                            size_t size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties, VkBuffer& buffer,
                            VkDeviceMemory& memory);
  void copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize size);
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer);
  void mainLoop();

 protected:
  const std::string m_appName;
  const bool m_enableValidation;
  VkInstance m_instance;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkSurfaceKHR m_surface;
  VkPhysicalDevice m_physicalDevice;
  uint8_t m_deviceUUID[VK_UUID_SIZE];
  VkDevice m_device;
  VkQueue m_graphicsQueue;
  VkQueue m_presentQueue;
  VkSwapchainKHR m_swapChain;
  std::vector<VkImage> m_swapChainImages;
  VkFormat m_swapChainFormat;
  VkExtent2D m_swapChainExtent;
  std::vector<VkImageView> m_swapChainImageViews;
  std::vector<std::pair<VkShaderStageFlagBits, std::string> > m_shaderFiles;
  VkRenderPass m_renderPass;
  VkPipelineLayout m_pipelineLayout;
  VkPipeline m_graphicsPipeline;
  std::vector<VkFramebuffer> m_swapChainFramebuffers;
  VkCommandPool m_commandPool;
  std::vector<VkCommandBuffer> m_commandBuffers;
  std::vector<VkSemaphore> m_imageAvailableSemaphores;
  std::vector<VkSemaphore> m_renderFinishedSemaphores;
  std::vector<VkFence> m_inFlightFences;
  std::vector<VkBuffer> m_uniformBuffers;
  std::vector<VkDeviceMemory> m_uniformMemory;
  VkDescriptorSetLayout m_descriptorSetLayout;
  VkDescriptorPool m_descriptorPool;
  std::vector<VkDescriptorSet> m_descriptorSets;

  VkImage m_depthImage;
  VkDeviceMemory m_depthImageMemory;
  VkImageView m_depthImageView;
  size_t m_currentFrame;
  bool m_framebufferResized;

  virtual void initVulkanApp() {}
  virtual void fillRenderingCommandBuffer(VkCommandBuffer& buffer) {}
  virtual std::vector<const char*> getRequiredExtensions() const;
  virtual std::vector<const char*> getRequiredDeviceExtensions() const;
  virtual void getVertexDescriptions(
      std::vector<VkVertexInputBindingDescription>& bindingDesc,
      std::vector<VkVertexInputAttributeDescription>& attribDesc);
  virtual void getAssemblyStateInfo(
      VkPipelineInputAssemblyStateCreateInfo& info);
  virtual void getWaitFrameSemaphores(
      std::vector<VkSemaphore>& wait,
      std::vector<VkPipelineStageFlags>& waitStages) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
  virtual VkDeviceSize getUniformSize() const;
  virtual void updateUniformBuffer(uint32_t imageIndex, size_t globalFrame);
  virtual void drawFrame();

 private:
  GLFWwindow* m_window;

  void initWindow();
  void initVulkan();
  void createInstance();
  void createSurface();
  void createDevice();
  void createSwapChain();
  void createImageViews();
  void createRenderPass();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandPool();
  void createDepthResources();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void createCommandBuffers();
  void createSyncObjects();

  void cleanupSwapChain();
  void recreateSwapChain();

  bool isSuitableDevice(VkPhysicalDevice dev) const;
  static void resizeCallback(GLFWwindow* window, int width, int height);
};

void readFile(std::istream& s, std::vector<char>& data);

#endif /* __VULKANBASEAPP_H__ */
