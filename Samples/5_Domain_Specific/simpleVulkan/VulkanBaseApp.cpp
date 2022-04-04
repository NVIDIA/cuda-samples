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
 * This file contains basic cross-platform setup paths in working with Vulkan
 * and rendering window.  It is largely based off of tutorials provided here:
 * https://vulkan-tutorial.com/
*/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <set>
#include <string.h>

#include "VulkanBaseApp.h"

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#ifdef _WIN64
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <aclapi.h>
#endif /* _WIN64 */

#ifndef countof
#define countof(x) (sizeof(x) / sizeof(*(x)))
#endif

static const char *validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
static const size_t MAX_FRAMES_IN_FLIGHT = 5;

void VulkanBaseApp::resizeCallback(GLFWwindow *window, int width, int height) {
  VulkanBaseApp *app =
      reinterpret_cast<VulkanBaseApp *>(glfwGetWindowUserPointer(window));
  app->m_framebufferResized = true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

VulkanBaseApp::VulkanBaseApp(const std::string &appName, bool enableValidation)
    : m_appName(appName),
      m_enableValidation(enableValidation),
      m_instance(VK_NULL_HANDLE),
      m_window(nullptr),
      m_debugMessenger(VK_NULL_HANDLE),
      m_surface(VK_NULL_HANDLE),
      m_physicalDevice(VK_NULL_HANDLE),
      m_device(VK_NULL_HANDLE),
      m_graphicsQueue(VK_NULL_HANDLE),
      m_presentQueue(VK_NULL_HANDLE),
      m_swapChain(VK_NULL_HANDLE),
      m_vkDeviceUUID(),
      m_swapChainImages(),
      m_swapChainFormat(),
      m_swapChainExtent(),
      m_swapChainImageViews(),
      m_shaderFiles(),
      m_renderPass(),
      m_pipelineLayout(VK_NULL_HANDLE),
      m_graphicsPipeline(VK_NULL_HANDLE),
      m_swapChainFramebuffers(),
      m_commandPool(VK_NULL_HANDLE),
      m_commandBuffers(),
      m_imageAvailableSemaphores(),
      m_renderFinishedSemaphores(),
      m_inFlightFences(),
      m_uniformBuffers(),
      m_uniformMemory(),
      m_descriptorSetLayout(VK_NULL_HANDLE),
      m_descriptorPool(VK_NULL_HANDLE),
      m_descriptorSets(),
      m_depthImage(VK_NULL_HANDLE),
      m_depthImageMemory(VK_NULL_HANDLE),
      m_depthImageView(VK_NULL_HANDLE),
      m_currentFrame(0),
      m_framebufferResized(false) {}

VkExternalSemaphoreHandleTypeFlagBits
VulkanBaseApp::getDefaultSemaphoreHandleType() {
#ifdef _WIN64
  return IsWindows8OrGreater()
             ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
             : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}

VkExternalMemoryHandleTypeFlagBits VulkanBaseApp::getDefaultMemHandleType() {
#ifdef _WIN64
  return IsWindows8Point1OrGreater()
             ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
             : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
}

VulkanBaseApp::~VulkanBaseApp() {
  cleanupSwapChain();

  if (m_descriptorSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  }

#ifdef _VK_TIMELINE_SEMAPHORE
  if (m_vkPresentationSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_device, m_vkPresentationSemaphore, nullptr);
  }
#endif /* _VK_TIMELINE_SEMAPHORE */

  for (size_t i = 0; i < m_renderFinishedSemaphores.size(); i++) {
    vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
    vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
  }
  if (m_commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
  }

  if (m_device != VK_NULL_HANDLE) {
    vkDestroyDevice(m_device, nullptr);
  }

  if (m_enableValidation) {
    PFN_vkDestroyDebugUtilsMessengerEXT func =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(m_instance, m_debugMessenger, nullptr);
    }
  }

  if (m_surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
  }

  if (m_instance != VK_NULL_HANDLE) {
    vkDestroyInstance(m_instance, nullptr);
  }

  if (m_window) {
    glfwDestroyWindow(m_window);
  }

  glfwTerminate();
}

void VulkanBaseApp::init() {
  initWindow();
  initVulkan();
}

VkCommandBuffer VulkanBaseApp::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void VulkanBaseApp::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphicsQueue);

  vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
}

void VulkanBaseApp::initWindow() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  m_window = glfwCreateWindow(1280, 800, m_appName.c_str(), nullptr, nullptr);
  glfwSetWindowUserPointer(m_window, this);
  glfwSetFramebufferSizeCallback(m_window, resizeCallback);
}

std::vector<const char *> VulkanBaseApp::getRequiredExtensions() const {
  return std::vector<const char *>();
}

std::vector<const char *> VulkanBaseApp::getRequiredDeviceExtensions() const {
  return std::vector<const char *>();
}

void VulkanBaseApp::initVulkan() {
  createInstance();
  createSurface();
  createDevice();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCommandPool();
  createDepthResources();
  createFramebuffers();
  initVulkanApp();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  createSyncObjects();
}

#ifdef _WIN64
class WindowsSecurityAttributes {
 protected:
  SECURITY_ATTRIBUTES m_winSecurityAttributes;
  PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

 public:
  WindowsSecurityAttributes();
  SECURITY_ATTRIBUTES *operator&();
  ~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
  m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
      1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
  if (!m_winPSecurityDescriptor) {
    throw std::runtime_error(
        "Failed to allocate memory for security descriptor");
  }

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

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() {
  return &m_winSecurityAttributes;
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
#endif /* _WIN64 */

static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                                    const std::vector<VkFormat> &candidates,
                                    VkImageTiling tiling,
                                    VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  throw std::runtime_error("Failed to find supported format!");
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice,
                               uint32_t typeFilter,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if (typeFilter & (1 << i) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }
  return ~0;
}

static bool supportsValidationLayers() {
  std::vector<VkLayerProperties> availableLayers;
  uint32_t layerCount;

  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  availableLayers.resize(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

void VulkanBaseApp::createInstance() {
  if (m_enableValidation && !supportsValidationLayers()) {
    throw std::runtime_error("Validation requested, but not supported!");
  }

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = m_appName.c_str();
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  std::vector<const char *> exts = getRequiredExtensions();

  {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    exts.insert(exts.begin(), glfwExtensions,
                glfwExtensions + glfwExtensionCount);

    if (m_enableValidation) {
      exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
  }

  createInfo.enabledExtensionCount = static_cast<uint32_t>(exts.size());
  createInfo.ppEnabledExtensionNames = exts.data();
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
  if (m_enableValidation) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(countof(validationLayers));
    createInfo.ppEnabledLayerNames = validationLayers;

    debugCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;

    createInfo.pNext = &debugCreateInfo;
  } else {
    createInfo.enabledLayerCount = 0;
    createInfo.pNext = nullptr;
  }

  if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance!");
  }

  if (m_enableValidation) {
    PFN_vkCreateDebugUtilsMessengerEXT func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (func == nullptr ||
        func(m_instance, &debugCreateInfo, nullptr, &m_debugMessenger) !=
            VK_SUCCESS) {
      throw std::runtime_error("Failed to set up debug messenger!");
    }
  }
}

void VulkanBaseApp::createSurface() {
  if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}

static bool findGraphicsQueueIndicies(VkPhysicalDevice device,
                                      VkSurfaceKHR surface,
                                      uint32_t &graphicsFamily,
                                      uint32_t &presentFamily) {
  uint32_t queueFamilyCount = 0;

  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  graphicsFamily = presentFamily = ~0;

  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueCount > 0) {
      if (graphicsFamily == ~0 &&
          queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        graphicsFamily = i;
      }
      uint32_t presentSupport = 0;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
      if (presentFamily == ~0 && presentSupport) {
        presentFamily = i;
      }
      if (presentFamily != ~0 && graphicsFamily != ~0) {
        break;
      }
    }
  }

  return graphicsFamily != ~0 && presentFamily != ~0;
}

static bool hasAllExtensions(
    VkPhysicalDevice device,
    const std::vector<const char *> &deviceExtensions) {
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                           deviceExtensions.end());

  for (const auto &extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}

static void getSwapChainProperties(
    VkPhysicalDevice device, VkSurfaceKHR surface,
    VkSurfaceCapabilitiesKHR &capabilities,
    std::vector<VkSurfaceFormatKHR> &formats,
    std::vector<VkPresentModeKHR> &presentModes) {
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
  if (formatCount != 0) {
    formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         formats.data());
  }
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            nullptr);
  if (presentModeCount != 0) {
    presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount, presentModes.data());
  }
}

bool VulkanBaseApp::isSuitableDevice(VkPhysicalDevice dev) const {
  uint32_t graphicsQueueIndex, presentQueueIndex;
  std::vector<const char *> deviceExtensions = getRequiredDeviceExtensions();
  VkSurfaceCapabilitiesKHR caps;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
  deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  getSwapChainProperties(dev, m_surface, caps, formats, presentModes);
  return hasAllExtensions(dev, deviceExtensions) && !formats.empty() &&
         !presentModes.empty() &&
         findGraphicsQueueIndicies(dev, m_surface, graphicsQueueIndex,
                                   presentQueueIndex);
}

void VulkanBaseApp::createDevice() {
  {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      throw std::runtime_error("Failed to find Vulkan capable GPUs!");
    }
    std::vector<VkPhysicalDevice> phyDevs(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, phyDevs.data());
    std::vector<VkPhysicalDevice>::iterator it =
        std::find_if(phyDevs.begin(), phyDevs.end(),
                     std::bind(&VulkanBaseApp::isSuitableDevice, this,
                               std::placeholders::_1));
    if (it == phyDevs.end()) {
      throw std::runtime_error("No suitable device found!");
    }
    m_physicalDevice = *it;
  }

  uint32_t graphicsQueueIndex, presentQueueIndex;
  findGraphicsQueueIndicies(m_physicalDevice, m_surface, graphicsQueueIndex,
                            presentQueueIndex);

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueFamilyIndices = {graphicsQueueIndex,
                                            presentQueueIndex};

  float queuePriority = 1.0f;

  for (uint32_t queueFamily : uniqueFamilyIndices) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures = {};
  deviceFeatures.fillModeNonSolid = true;

  VkDeviceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

#ifdef _VK_TIMELINE_SEMAPHORE
  VkPhysicalDeviceVulkan12Features vk12features = {};
  vk12features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vk12features.timelineSemaphore = true;
  createInfo.pNext = &vk12features;
#endif

  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());

  createInfo.pEnabledFeatures = &deviceFeatures;

  std::vector<const char *> deviceExtensions = getRequiredDeviceExtensions();
  deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  if (m_enableValidation) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(countof(validationLayers));
    createInfo.ppEnabledLayerNames = validationLayers;
  } else {
    createInfo.enabledLayerCount = 0;
  }

  if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(m_device, graphicsQueueIndex, 0, &m_graphicsQueue);
  vkGetDeviceQueue(m_device, presentQueueIndex, 0, &m_presentQueue);

  VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
  vkPhysicalDeviceIDProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  vkPhysicalDeviceIDProperties.pNext = NULL;

  VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
  vkPhysicalDeviceProperties2.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

  PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
  fpGetPhysicalDeviceProperties2 =
      (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
          m_instance, "vkGetPhysicalDeviceProperties2");
  if (fpGetPhysicalDeviceProperties2 == NULL) {
    throw std::runtime_error(
        "Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not "
        "found.\n");
  }

  fpGetPhysicalDeviceProperties2(m_physicalDevice,
                                 &vkPhysicalDeviceProperties2);

  memcpy(m_vkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE);
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  if (availableFormats.size() == 1 &&
      availableFormats[0].format == VK_FORMAT_UNDEFINED) {
    return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  for (const auto &availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
        availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }

  return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes) {
  VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      bestMode = availablePresentMode;
    }
  }

  return bestMode;
}

static VkExtent2D chooseSwapExtent(
    GLFWwindow *window, const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                               static_cast<uint32_t>(height)};

    actualExtent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(capabilities.maxImageExtent.height, actualExtent.height));

    return actualExtent;
  }
}

void VulkanBaseApp::createSwapChain() {
  VkSurfaceCapabilitiesKHR capabilities;
  VkSurfaceFormatKHR format;
  VkPresentModeKHR presentMode;
  VkExtent2D extent;
  uint32_t imageCount;

  {
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    getSwapChainProperties(m_physicalDevice, m_surface, capabilities, formats,
                           presentModes);
    format = chooseSwapSurfaceFormat(formats);
    presentMode = chooseSwapPresentMode(presentModes);
    extent = chooseSwapExtent(m_window, capabilities);
    imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 &&
        imageCount > capabilities.maxImageCount) {
      imageCount = capabilities.maxImageCount;
    }
  }

  VkSwapchainCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = m_surface;

  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = format.format;
  createInfo.imageColorSpace = format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queueFamilyIndices[2];
  findGraphicsQueueIndicies(m_physicalDevice, m_surface, queueFamilyIndices[0],
                            queueFamilyIndices[1]);

  if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = countof(queueFamilyIndices);
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapChain) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr);
  m_swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount,
                          m_swapChainImages.data());

  m_swapChainFormat = format.format;
  m_swapChainExtent = extent;
}

static VkImageView createImageView(VkDevice dev, VkImage image, VkFormat format,
                                   VkImageAspectFlags aspectFlags) {
  VkImageView imageView;
  VkImageViewCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.image = image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format = format;
  createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.subresourceRange.aspectMask = aspectFlags;
  createInfo.subresourceRange.baseMipLevel = 0;
  createInfo.subresourceRange.levelCount = 1;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount = 1;
  if (vkCreateImageView(dev, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image views!");
  }

  return imageView;
}

static void createImage(VkPhysicalDevice physicalDevice, VkDevice device,
                        uint32_t width, uint32_t height, VkFormat format,
                        VkImageTiling tiling, VkImageUsageFlags usage,
                        VkMemoryPropertyFlags properties, VkImage &image,
                        VkDeviceMemory &imageMemory) {
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      physicalDevice, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  vkBindImageMemory(device, image, imageMemory, 0);
}

void VulkanBaseApp::createImageViews() {
  m_swapChainImageViews.resize(m_swapChainImages.size());

  for (uint32_t i = 0; i < m_swapChainImages.size(); i++) {
    m_swapChainImageViews[i] =
        createImageView(m_device, m_swapChainImages[i], m_swapChainFormat,
                        VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

void VulkanBaseApp::createRenderPass() {
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = m_swapChainFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription depthAttachment = {};
  depthAttachment.format = findSupportedFormat(
      m_physicalDevice, {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                         VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef = {};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkAttachmentDescription attachments[] = {colorAttachment, depthAttachment};
  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = countof(attachments);
  renderPassInfo.pAttachments = attachments;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void VulkanBaseApp::createDescriptorSetLayout() {
  VkDescriptorSetLayoutBinding uboLayoutBinding = {};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.pImmutableSamplers = nullptr;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &uboLayoutBinding;

  if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr,
                                  &m_descriptorSetLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

VkShaderModule createShaderModule(VkDevice device, const char *filename) {
  std::vector<char> shaderContents;
  std::ifstream shaderFile(filename, std::ios_base::in | std::ios_base::binary);
  VkShaderModuleCreateInfo createInfo = {};
  VkShaderModule shaderModule;

  if (!shaderFile.good()) {
    throw std::runtime_error("Failed to load shader contents");
  }
  readFile(shaderFile, shaderContents);

  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = shaderContents.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(shaderContents.data());

  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module!");
  }

  return shaderModule;
}

void VulkanBaseApp::getVertexDescriptions(
    std::vector<VkVertexInputBindingDescription> &bindingDesc,
    std::vector<VkVertexInputAttributeDescription> &attribDesc) {}

void VulkanBaseApp::getAssemblyStateInfo(
    VkPipelineInputAssemblyStateCreateInfo &info) {}

void VulkanBaseApp::createGraphicsPipeline() {
  std::vector<VkPipelineShaderStageCreateInfo> shaderStageInfos(
      m_shaderFiles.size());
  for (size_t i = 0; i < m_shaderFiles.size(); i++) {
    shaderStageInfos[i] = {};
    shaderStageInfos[i].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfos[i].stage = m_shaderFiles[i].first;
    shaderStageInfos[i].module =
        createShaderModule(m_device, m_shaderFiles[i].second.c_str());
    shaderStageInfos[i].pName = "main";
  }

  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};

  std::vector<VkVertexInputBindingDescription> vertexBindingDescriptions;
  std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions;

  getVertexDescriptions(vertexBindingDescriptions, vertexAttributeDescriptions);

  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount =
      static_cast<uint32_t>(vertexBindingDescriptions.size());
  vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptions.data();
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(vertexAttributeDescriptions.size());
  vertexInputInfo.pVertexAttributeDescriptions =
      vertexAttributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  getAssemblyStateInfo(inputAssembly);

  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)m_swapChainExtent.width;
  viewport.height = (float)m_swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = m_swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;           // Optional
  multisampling.pSampleMask = nullptr;             // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE;  // Optional
  multisampling.alphaToOneEnable = VK_FALSE;       // Optional

  VkPipelineDepthStencilStateCreateInfo depthStencil = {};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;                    // Optional
  pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;  // Optional
  pipelineLayoutInfo.pushConstantRangeCount = 0;            // Optional
  pipelineLayoutInfo.pPushConstantRanges = nullptr;         // Optional

  if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                             &m_pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(shaderStageInfos.size());
  pipelineInfo.pStages = shaderStageInfos.data();

  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;  // Optional
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr;  // Optional

  pipelineInfo.layout = m_pipelineLayout;

  pipelineInfo.renderPass = m_renderPass;
  pipelineInfo.subpass = 0;

  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
  pipelineInfo.basePipelineIndex = -1;               // Optional

  if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                nullptr, &m_graphicsPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  for (size_t i = 0; i < shaderStageInfos.size(); i++) {
    vkDestroyShaderModule(m_device, shaderStageInfos[i].module, nullptr);
  }
}

void VulkanBaseApp::createFramebuffers() {
  m_swapChainFramebuffers.resize(m_swapChainImageViews.size());
  for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
    VkImageView attachments[] = {m_swapChainImageViews[i], m_depthImageView};

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_renderPass;
    framebufferInfo.attachmentCount = countof(attachments);
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = m_swapChainExtent.width;
    framebufferInfo.height = m_swapChainExtent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr,
                            &m_swapChainFramebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

void VulkanBaseApp::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo = {};
  uint32_t graphicsIndex, presentIndex;

  findGraphicsQueueIndicies(m_physicalDevice, m_surface, graphicsIndex,
                            presentIndex);

  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = graphicsIndex;
  poolInfo.flags = 0;  // Optional

  if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool!");
  }
}

static void transitionImageLayout(VulkanBaseApp *app, VkImage image,
                                  VkFormat format, VkImageLayout oldLayout,
                                  VkImageLayout newLayout) {
  VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;

  if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        format == VK_FORMAT_D24_UNORM_S8_UINT) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }

  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);

  app->endSingleTimeCommands(commandBuffer);
}

void VulkanBaseApp::createDepthResources() {
  VkFormat depthFormat = findSupportedFormat(
      m_physicalDevice, {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                         VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  createImage(m_physicalDevice, m_device, m_swapChainExtent.width,
              m_swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthImage,
              m_depthImageMemory);
  m_depthImageView = createImageView(m_device, m_depthImage, depthFormat,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);
  transitionImageLayout(this, m_depthImage, depthFormat,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void VulkanBaseApp::createUniformBuffers() {
  VkDeviceSize size = getUniformSize();
  if (size > 0) {
    m_uniformBuffers.resize(m_swapChainImages.size());
    m_uniformMemory.resize(m_swapChainImages.size());
    for (size_t i = 0; i < m_uniformBuffers.size(); i++) {
      createBuffer(getUniformSize(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   m_uniformBuffers[i], m_uniformMemory[i]);
    }
  }
}

void VulkanBaseApp::createDescriptorPool() {
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(m_swapChainImages.size());
  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = static_cast<uint32_t>(m_swapChainImages.size());
  if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void VulkanBaseApp::createDescriptorSets() {
  std::vector<VkDescriptorSetLayout> layouts(m_swapChainImages.size(),
                                             m_descriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = m_descriptorPool;
  allocInfo.descriptorSetCount =
      static_cast<uint32_t>(m_swapChainImages.size());
  allocInfo.pSetLayouts = layouts.data();
  m_descriptorSets.resize(m_swapChainImages.size());

  if (vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptorSets.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  VkDescriptorBufferInfo bufferInfo = {};
  bufferInfo.offset = 0;
  bufferInfo.range = VK_WHOLE_SIZE;
  VkWriteDescriptorSet descriptorWrite = {};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &bufferInfo;
  descriptorWrite.pImageInfo = nullptr;        // Optional
  descriptorWrite.pTexelBufferView = nullptr;  // Optional

  for (size_t i = 0; i < m_swapChainImages.size(); i++) {
    bufferInfo.buffer = m_uniformBuffers[i];
    descriptorWrite.dstSet = m_descriptorSets[i];
    vkUpdateDescriptorSets(m_device, 1, &descriptorWrite, 0, nullptr);
  }
}

void VulkanBaseApp::createCommandBuffers() {
  m_commandBuffers.resize(m_swapChainFramebuffers.size());
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = m_commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)m_commandBuffers.size();

  if (vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  for (size_t i = 0; i < m_commandBuffers.size(); i++) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = nullptr;  // Optional

    if (vkBeginCommandBuffer(m_commandBuffers[i], &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_renderPass;
    renderPassInfo.framebuffer = m_swapChainFramebuffers[i];

    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_swapChainExtent;

    VkClearValue clearColors[2];
    clearColors[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clearColors[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = countof(clearColors);
    renderPassInfo.pClearValues = clearColors;

    vkCmdBeginRenderPass(m_commandBuffers[i], &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(m_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_graphicsPipeline);

    vkCmdBindDescriptorSets(m_commandBuffers[i],
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_descriptorSets[i], 0, nullptr);

    fillRenderingCommandBuffer(m_commandBuffers[i]);

    vkCmdEndRenderPass(m_commandBuffers[i]);

    if (vkEndCommandBuffer(m_commandBuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }
}

void VulkanBaseApp::createSyncObjects() {
  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
  m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
  m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr,
                          &m_imageAvailableSemaphores[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create image available semaphore!");
    }
    if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr,
                          &m_renderFinishedSemaphores[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create image available semaphore!");
    }
    if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create image available semaphore!");
    }
  }

#ifdef _VK_TIMELINE_SEMAPHORE
  if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr,
                        &m_vkPresentationSemaphore) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create binary semaphore!");
  }
#endif /* _VK_TIMELINE_SEMAPHORE */
}

void VulkanBaseApp::getWaitFrameSemaphores(
    std::vector<VkSemaphore> &wait,
    std::vector<VkPipelineStageFlags> &waitStages) const {}

void VulkanBaseApp::getSignalFrameSemaphores(
    std::vector<VkSemaphore> &signal) const {}

VkDeviceSize VulkanBaseApp::getUniformSize() const { return VkDeviceSize(0); }

void VulkanBaseApp::updateUniformBuffer(uint32_t imageIndex) {}

void VulkanBaseApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties,
                                 VkBuffer &buffer,
                                 VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      m_physicalDevice, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
}

void VulkanBaseApp::createExternalBuffer(
    VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer &buffer,
    VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
  externalMemoryBufferInfo.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  externalMemoryBufferInfo.handleTypes = extMemHandleType;
  bufferInfo.pNext = &externalMemoryBufferInfo;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

#ifdef _WIN64
  WindowsSecurityAttributes winSecurityAttributes;

  VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
  vulkanExportMemoryWin32HandleInfoKHR.sType =
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
  vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
  vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
      DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif /* _WIN64 */
  VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
  vulkanExportMemoryAllocateInfoKHR.sType =
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
  vulkanExportMemoryAllocateInfoKHR.pNext =
      extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
          ? &vulkanExportMemoryWin32HandleInfoKHR
          : NULL;
  vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
#else
  vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
  vulkanExportMemoryAllocateInfoKHR.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */
  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      m_physicalDevice, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate external buffer memory!");
  }

  vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
}

void *VulkanBaseApp::getMemHandle(
    VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType) {
#ifdef _WIN64
  HANDLE handle = 0;

  VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
  vkMemoryGetWin32HandleInfoKHR.sType =
      VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
  vkMemoryGetWin32HandleInfoKHR.memory = memory;
  vkMemoryGetWin32HandleInfoKHR.handleType = handleType;

  PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
  fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
          m_device, "vkGetMemoryWin32HandleKHR");
  if (!fpGetMemoryWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR,
                                &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void *)handle;
#else
  int fd = -1;

  VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
  vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  vkMemoryGetFdInfoKHR.pNext = NULL;
  vkMemoryGetFdInfoKHR.memory = memory;
  vkMemoryGetFdInfoKHR.handleType = handleType;

  PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
  fpGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdKHR");
  if (!fpGetMemoryFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetMemoryFdKHR(m_device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void *)(uintptr_t)fd;
#endif /* _WIN64 */
}

void *VulkanBaseApp::getSemaphoreHandle(
    VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType) {
#ifdef _WIN64
  HANDLE handle;

  VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
  semaphoreGetWin32HandleInfoKHR.sType =
      VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  semaphoreGetWin32HandleInfoKHR.pNext = NULL;
  semaphoreGetWin32HandleInfoKHR.semaphore = semaphore;
  semaphoreGetWin32HandleInfoKHR.handleType = handleType;

  PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
  fpGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
          m_device, "vkGetSemaphoreWin32HandleKHR");
  if (!fpGetSemaphoreWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetSemaphoreWin32HandleKHR(m_device, &semaphoreGetWin32HandleInfoKHR,
                                   &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }

  return (void *)handle;
#else
  int fd;

  VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
  semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  semaphoreGetFdInfoKHR.pNext = NULL;
  semaphoreGetFdInfoKHR.semaphore = semaphore;
  semaphoreGetFdInfoKHR.handleType = handleType;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
  fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
      m_device, "vkGetSemaphoreFdKHR");
  if (!fpGetSemaphoreFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
  }
  if (fpGetSemaphoreFdKHR(m_device, &semaphoreGetFdInfoKHR, &fd) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }

  return (void *)(uintptr_t)fd;
#endif /* _WIN64 */
}

void VulkanBaseApp::createExternalSemaphore(
    VkSemaphore &semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType) {
  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
  exportSemaphoreCreateInfo.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

#ifdef _VK_TIMELINE_SEMAPHORE
  VkSemaphoreTypeCreateInfo timelineCreateInfo;
  timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timelineCreateInfo.pNext = NULL;
  timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timelineCreateInfo.initialValue = 0;
  exportSemaphoreCreateInfo.pNext = &timelineCreateInfo;
#else
  exportSemaphoreCreateInfo.pNext = NULL;
#endif /* _VK_TIMELINE_SEMAPHORE */
  exportSemaphoreCreateInfo.handleTypes = handleType;
  semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

  if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &semaphore) !=
      VK_SUCCESS) {
    throw std::runtime_error(
        "failed to create synchronization objects for a CUDA-Vulkan!");
  }
}

void VulkanBaseApp::importExternalBuffer(
    void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    VkBuffer &buffer, VkDeviceMemory &memory) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

#ifdef _WIN64
  VkImportMemoryWin32HandleInfoKHR handleInfo = {};
  handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  handleInfo.pNext = NULL;
  handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  handleInfo.handle = handle;
  handleInfo.name = NULL;
#else
  VkImportMemoryFdInfoKHR handleInfo = {};
  handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
  handleInfo.pNext = NULL;
  handleInfo.fd = (int)(uintptr_t)handle;
  handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif /* _WIN64 */

  VkMemoryAllocateInfo memAllocation = {};
  memAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAllocation.pNext = (void *)&handleInfo;
  memAllocation.allocationSize = size;
  memAllocation.memoryTypeIndex = findMemoryType(
      m_physicalDevice, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(m_device, &memAllocation, nullptr, &memory) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to import allocation!");
  }

  vkBindBufferMemory(m_device, buffer, memory, 0);
}

void VulkanBaseApp::copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize size) {
  VkCommandBuffer commandBuffer = beginSingleTimeCommands();

  VkBufferCopy copyRegion = {};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

  endSingleTimeCommands(commandBuffer);
}
#ifdef _VK_TIMELINE_SEMAPHORE
void VulkanBaseApp::drawFrame() {
  static uint64_t waitValue = 0;
  static uint64_t signalValue = 1;

  VkSemaphoreWaitInfo semaphoreWaitInfo = {};
  semaphoreWaitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  semaphoreWaitInfo.pSemaphores = &m_vkTimelineSemaphore;
  semaphoreWaitInfo.semaphoreCount = 1;
  semaphoreWaitInfo.pValues = &waitValue;
  vkWaitSemaphores(m_device, &semaphoreWaitInfo,
                   std::numeric_limits<uint64_t>::max());

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(
      m_device, m_swapChain, std::numeric_limits<uint64_t>::max(),
      m_vkPresentationSemaphore, VK_NULL_HANDLE, &imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain();
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image!");
  }

  updateUniformBuffer(imageIndex);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  std::vector<VkSemaphore> waitSemaphores;
  std::vector<VkPipelineStageFlags> waitStages;

  waitSemaphores.push_back(m_vkTimelineSemaphore);
  waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

  submitInfo.waitSemaphoreCount = (uint32_t)waitSemaphores.size();
  submitInfo.pWaitSemaphores = waitSemaphores.data();
  submitInfo.pWaitDstStageMask = waitStages.data();

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

  std::vector<VkSemaphore> signalSemaphores;
  signalSemaphores.push_back(m_vkTimelineSemaphore);
  submitInfo.signalSemaphoreCount = (uint32_t)signalSemaphores.size();
  submitInfo.pSignalSemaphores = signalSemaphores.data();

  VkTimelineSemaphoreSubmitInfo timelineInfo = {};
  timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timelineInfo.waitSemaphoreValueCount = 1;
  timelineInfo.pWaitSemaphoreValues = &waitValue;
  timelineInfo.signalSemaphoreValueCount = 1;
  timelineInfo.pSignalSemaphoreValues = &signalValue;

  submitInfo.pNext = &timelineInfo;

  if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &m_vkPresentationSemaphore;

  VkSwapchainKHR swapChains[] = {m_swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;

  result = vkQueuePresentKHR(m_presentQueue, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      m_framebufferResized) {
    recreateSwapChain();
    m_framebufferResized = false;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to acquire swap chain image!");
  }

  m_currentFrame++;

  waitValue += 2;
  signalValue += 2;
}
#else
void VulkanBaseApp::drawFrame() {
  size_t currentFrameIdx = m_currentFrame % MAX_FRAMES_IN_FLIGHT;
  vkWaitForFences(m_device, 1, &m_inFlightFences[currentFrameIdx], VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(
      m_device, m_swapChain, std::numeric_limits<uint64_t>::max(),
      m_imageAvailableSemaphores[currentFrameIdx], VK_NULL_HANDLE, &imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain();
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("Failed to acquire swap chain image!");
  }

  updateUniformBuffer(imageIndex);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  std::vector<VkSemaphore> waitSemaphores;
  std::vector<VkPipelineStageFlags> waitStages;

  waitSemaphores.push_back(m_imageAvailableSemaphores[currentFrameIdx]);
  waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  getWaitFrameSemaphores(waitSemaphores, waitStages);

  submitInfo.waitSemaphoreCount = (uint32_t)waitSemaphores.size();
  submitInfo.pWaitSemaphores = waitSemaphores.data();
  submitInfo.pWaitDstStageMask = waitStages.data();

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

  std::vector<VkSemaphore> signalSemaphores;
  getSignalFrameSemaphores(signalSemaphores);
  signalSemaphores.push_back(m_renderFinishedSemaphores[currentFrameIdx]);
  submitInfo.signalSemaphoreCount = (uint32_t)signalSemaphores.size();
  submitInfo.pSignalSemaphores = signalSemaphores.data();

  vkResetFences(m_device, 1, &m_inFlightFences[currentFrameIdx]);

  if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo,
                    m_inFlightFences[currentFrameIdx]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[currentFrameIdx];

  VkSwapchainKHR swapChains[] = {m_swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;

  result = vkQueuePresentKHR(m_presentQueue, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      m_framebufferResized) {
    recreateSwapChain();
    m_framebufferResized = false;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to acquire swap chain image!");
  }

  m_currentFrame++;
}
#endif /* _VK_TIMELINE_SEMAPHORE */

void VulkanBaseApp::cleanupSwapChain() {
  if (m_depthImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(m_device, m_depthImageView, nullptr);
  }
  if (m_depthImage != VK_NULL_HANDLE) {
    vkDestroyImage(m_device, m_depthImage, nullptr);
  }
  if (m_depthImageMemory != VK_NULL_HANDLE) {
    vkFreeMemory(m_device, m_depthImageMemory, nullptr);
  }

  for (size_t i = 0; i < m_uniformBuffers.size(); i++) {
    vkDestroyBuffer(m_device, m_uniformBuffers[i], nullptr);
    vkFreeMemory(m_device, m_uniformMemory[i], nullptr);
  }

  if (m_descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
  }

  for (size_t i = 0; i < m_swapChainFramebuffers.size(); i++) {
    vkDestroyFramebuffer(m_device, m_swapChainFramebuffers[i], nullptr);
  }

  if (m_graphicsPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  }

  if (m_pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  }

  if (m_renderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);
  }

  for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
    vkDestroyImageView(m_device, m_swapChainImageViews[i], nullptr);
  }

  if (m_swapChain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
  }
}

void VulkanBaseApp::recreateSwapChain() {
  int width, height;

  glfwGetFramebufferSize(m_window, &width, &height);
  while (width == 0 || height == 0) {
    glfwWaitEvents();
    glfwGetFramebufferSize(m_window, &width, &height);
  }

  vkDeviceWaitIdle(m_device);

  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createDepthResources();
  createFramebuffers();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
}

void VulkanBaseApp::mainLoop() {
  while (!glfwWindowShouldClose(m_window)) {
    glfwPollEvents();
    drawFrame();
  }
  vkDeviceWaitIdle(m_device);
}

void readFile(std::istream &s, std::vector<char> &data) {
  s.seekg(0, std::ios_base::end);
  data.resize(s.tellg());
  s.clear();
  s.seekg(0, std::ios_base::beg);
  s.read(data.data(), data.size());
}
