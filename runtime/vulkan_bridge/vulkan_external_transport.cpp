/*
 * NeuralStudio Vulkan <-> D3D12 external-memory transport
 * SPDX-License-Identifier: MIT
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR

#include "vulkan_external_transport.h"

#include <windows.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace neuralstudio::vulkan_bridge {
namespace {

[[noreturn]] void Fail(std::string message) {
  throw std::runtime_error(std::move(message));
}

void RequireVk(VkResult result, std::string_view operation) {
  if (result != VK_SUCCESS) {
    Fail(std::string(operation) + " failed with VkResult " +
         std::to_string(static_cast<int>(result)));
  }
}

template <typename T>
T RequireProc(PFN_vkGetInstanceProcAddr get, VkInstance instance,
              const char* name) {
  const auto function = reinterpret_cast<T>(get(instance, name));
  if (function == nullptr) Fail(std::string("Vulkan function missing: ") + name);
  return function;
}

template <typename T>
T RequireDeviceProc(PFN_vkGetDeviceProcAddr get, VkDevice device,
                    const char* name) {
  const auto function = reinterpret_cast<T>(get(device, name));
  if (function == nullptr) {
    Fail(std::string("Vulkan device function missing: ") + name);
  }
  return function;
}

class Module {
 public:
  Module() : value_(LoadLibraryExW(L"vulkan-1.dll", nullptr,
                                   LOAD_LIBRARY_SEARCH_SYSTEM32)) {
    if (value_ == nullptr) Fail("Unable to load the system Vulkan loader");
  }
  ~Module() {
    if (value_ != nullptr) FreeLibrary(value_);
  }
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  [[nodiscard]] HMODULE get() const noexcept { return value_; }

 private:
  HMODULE value_ = nullptr;
};

class WinHandle {
 public:
  WinHandle() = default;
  explicit WinHandle(HANDLE value) : value_(value) {}
  ~WinHandle() { reset(); }
  WinHandle(const WinHandle&) = delete;
  WinHandle& operator=(const WinHandle&) = delete;
  WinHandle(WinHandle&& other) noexcept : value_(other.release()) {}
  WinHandle& operator=(WinHandle&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }
  [[nodiscard]] HANDLE get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
  }
  HANDLE release() noexcept {
    HANDLE result = value_;
    value_ = nullptr;
    return result;
  }
  void reset(HANDLE value = nullptr) noexcept {
    if (*this) CloseHandle(value_);
    value_ = value;
  }

 private:
  HANDLE value_ = nullptr;
};

WinHandle DuplicateLocal(HandleToken token) {
  if (token == 0) Fail("External Win32 handle is null");
  HANDLE duplicate = nullptr;
  const HANDLE source = reinterpret_cast<HANDLE>(token);
  if (!DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(),
                       &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    Fail("DuplicateHandle failed for Vulkan import");
  }
  return WinHandle(duplicate);
}

struct Functions {
  PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
  PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
  PFN_vkDestroyInstance destroy_instance = nullptr;
  PFN_vkEnumeratePhysicalDevices enumerate_physical_devices = nullptr;
  PFN_vkGetPhysicalDeviceProperties2 get_properties2 = nullptr;
  PFN_vkGetPhysicalDeviceFeatures2 get_features2 = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_families = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties = nullptr;
  PFN_vkGetPhysicalDeviceImageFormatProperties2 get_image_properties = nullptr;
  PFN_vkGetPhysicalDeviceExternalSemaphoreProperties get_semaphore_properties =
      nullptr;
  PFN_vkDestroyDevice destroy_device = nullptr;
  PFN_vkGetDeviceQueue get_device_queue = nullptr;
  PFN_vkCreateImage create_image = nullptr;
  PFN_vkDestroyImage destroy_image = nullptr;
  PFN_vkGetImageMemoryRequirements get_image_requirements = nullptr;
  PFN_vkAllocateMemory allocate_memory = nullptr;
  PFN_vkFreeMemory free_memory = nullptr;
  PFN_vkBindImageMemory bind_image_memory = nullptr;
  PFN_vkGetMemoryWin32HandlePropertiesKHR get_handle_properties = nullptr;
  PFN_vkCreateSemaphore create_semaphore = nullptr;
  PFN_vkDestroySemaphore destroy_semaphore = nullptr;
  PFN_vkImportSemaphoreWin32HandleKHR import_semaphore = nullptr;
  PFN_vkWaitSemaphores wait_semaphores = nullptr;
  PFN_vkGetSemaphoreCounterValue get_semaphore_value = nullptr;
  PFN_vkCreateBuffer create_buffer = nullptr;
  PFN_vkDestroyBuffer destroy_buffer = nullptr;
  PFN_vkGetBufferMemoryRequirements get_buffer_requirements = nullptr;
  PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
  PFN_vkMapMemory map_memory = nullptr;
  PFN_vkUnmapMemory unmap_memory = nullptr;
  PFN_vkFlushMappedMemoryRanges flush_memory = nullptr;
  PFN_vkInvalidateMappedMemoryRanges invalidate_memory = nullptr;
  PFN_vkCreateCommandPool create_command_pool = nullptr;
  PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
  PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
  PFN_vkResetCommandPool reset_command_pool = nullptr;
  PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
  PFN_vkEndCommandBuffer end_command_buffer = nullptr;
  PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
  PFN_vkCmdCopyImageToBuffer cmd_copy_image_to_buffer = nullptr;
  PFN_vkCmdCopyBufferToImage cmd_copy_buffer_to_image = nullptr;
  PFN_vkQueueSubmit queue_submit = nullptr;
  PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  bool coherent = false;
};

struct ImportedImage {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct Slot {
  ImportedImage input{};
  ImportedImage output{};
  Buffer upload{};
  Buffer readback{};
  VkCommandPool command_pool = VK_NULL_HANDLE;
  std::array<VkCommandBuffer, 2> commands{};
};

bool HasExtension(const std::unordered_set<std::string>& available,
                  const char* name) {
  return available.contains(name);
}

std::uint64_t LuidValue(const std::uint8_t luid[VK_LUID_SIZE]) {
  static_assert(VK_LUID_SIZE == sizeof(std::uint64_t));
  std::uint64_t value = 0;
  std::memcpy(&value, luid, sizeof(value));
  return value;
}

}  // namespace

struct VulkanExternalTransport::Impl {
  explicit Impl(const ExternalTransportDescription& requested)
      : description(requested),
        format(requested.bytes_per_pixel == 8
                   ? VK_FORMAT_R16G16B16A16_SFLOAT
                   : VK_FORMAT_R8G8B8A8_UNORM) {}

  void Initialize() {
    ValidateDescription();
    InitializeVulkan();
    ValidateExternalSupport();
    ImportTimeline();
    for (std::size_t index = 0; index < slots.size(); ++index) {
      CreateSlot(index);
    }
  }

  ~Impl() {
    if (device != VK_NULL_HANDLE && function.device_wait_idle != nullptr) {
      (void)function.device_wait_idle(device);
    }
    for (Slot& slot : slots) DestroySlot(slot);
    if (timeline != VK_NULL_HANDLE && function.destroy_semaphore != nullptr) {
      function.destroy_semaphore(device, timeline, nullptr);
    }
    if (device != VK_NULL_HANDLE && function.destroy_device != nullptr) {
      function.destroy_device(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE && function.destroy_instance != nullptr) {
      function.destroy_instance(instance, nullptr);
    }
  }

  void ValidateDescription() const {
    if (description.adapter_luid == 0 || description.timeline_fence == 0 ||
        description.width == 0 || description.height == 0 ||
        (description.bytes_per_pixel != 4 &&
         description.bytes_per_pixel != 8)) {
      Fail("Invalid Vulkan external-transport description");
    }
    for (const ExternalImagePair& pair : description.slots) {
      if (pair.input == 0 || pair.output == 0) {
        Fail("Vulkan external-transport slot handle is null");
      }
    }
    const std::uint64_t byte_count =
        static_cast<std::uint64_t>(description.width) * description.height *
        description.bytes_per_pixel;
    if (byte_count == 0 ||
        byte_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max())) {
      Fail("Vulkan external-transport byte count overflow");
    }
  }

  void InitializeVulkan() {
    function.get_instance_proc =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetProcAddress(module.get(), "vkGetInstanceProcAddr"));
    if (function.get_instance_proc == nullptr) {
      Fail("Vulkan loader lacks vkGetInstanceProcAddr");
    }
    const auto create_instance = RequireProc<PFN_vkCreateInstance>(
        function.get_instance_proc, nullptr, "vkCreateInstance");
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "NeuralStudio external transport";
    application.applicationVersion = 1;
    application.pEngineName = "NeuralStudio";
    application.engineVersion = 1;
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    RequireVk(create_instance(&create, nullptr, &instance),
              "vkCreateInstance");

    auto& f = function;
    f.get_device_proc = RequireProc<PFN_vkGetDeviceProcAddr>(
        f.get_instance_proc, instance, "vkGetDeviceProcAddr");
    f.destroy_instance = RequireProc<PFN_vkDestroyInstance>(
        f.get_instance_proc, instance, "vkDestroyInstance");
    f.enumerate_physical_devices = RequireProc<PFN_vkEnumeratePhysicalDevices>(
        f.get_instance_proc, instance, "vkEnumeratePhysicalDevices");
    f.get_properties2 = RequireProc<PFN_vkGetPhysicalDeviceProperties2>(
        f.get_instance_proc, instance, "vkGetPhysicalDeviceProperties2");
    f.get_features2 = RequireProc<PFN_vkGetPhysicalDeviceFeatures2>(
        f.get_instance_proc, instance, "vkGetPhysicalDeviceFeatures2");
    f.get_queue_families =
        RequireProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            f.get_instance_proc, instance,
            "vkGetPhysicalDeviceQueueFamilyProperties");
    f.enumerate_extensions =
        RequireProc<PFN_vkEnumerateDeviceExtensionProperties>(
            f.get_instance_proc, instance,
            "vkEnumerateDeviceExtensionProperties");
    f.get_memory_properties =
        RequireProc<PFN_vkGetPhysicalDeviceMemoryProperties>(
            f.get_instance_proc, instance,
            "vkGetPhysicalDeviceMemoryProperties");
    f.get_image_properties =
        RequireProc<PFN_vkGetPhysicalDeviceImageFormatProperties2>(
            f.get_instance_proc, instance,
            "vkGetPhysicalDeviceImageFormatProperties2");
    f.get_semaphore_properties =
        RequireProc<PFN_vkGetPhysicalDeviceExternalSemaphoreProperties>(
            f.get_instance_proc, instance,
            "vkGetPhysicalDeviceExternalSemaphoreProperties");
    const auto create_device = RequireProc<PFN_vkCreateDevice>(
        f.get_instance_proc, instance, "vkCreateDevice");

    std::uint32_t device_count = 0;
    RequireVk(f.enumerate_physical_devices(instance, &device_count, nullptr),
              "vkEnumeratePhysicalDevices(count)");
    if (device_count == 0) Fail("No Vulkan physical devices found");
    std::vector<VkPhysicalDevice> devices(device_count);
    RequireVk(f.enumerate_physical_devices(instance, &device_count,
                                           devices.data()),
              "vkEnumeratePhysicalDevices(list)");
    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceIDProperties id{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 properties{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      properties.pNext = &id;
      f.get_properties2(candidate, &properties);
      if (id.deviceLUIDValid &&
          LuidValue(id.deviceLUID) == description.adapter_luid) {
        physical_device = candidate;
        break;
      }
    }
    if (physical_device == VK_NULL_HANDLE) {
      Fail("No Vulkan physical device matched the D3D12 adapter LUID");
    }

    std::uint32_t family_count = 0;
    f.get_queue_families(physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    f.get_queue_families(physical_device, &family_count, families.data());
    bool found_family = false;
    for (std::uint32_t index = 0; index < family_count; ++index) {
      const VkQueueFlags required =
          VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
      if ((families[index].queueFlags & required) == required) {
        queue_family = index;
        found_family = true;
        break;
      }
    }
    if (!found_family) Fail("No Vulkan graphics/transfer queue family found");

    std::uint32_t extension_count = 0;
    RequireVk(f.enumerate_extensions(physical_device, nullptr,
                                     &extension_count, nullptr),
              "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> extensions(extension_count);
    RequireVk(f.enumerate_extensions(physical_device, nullptr,
                                     &extension_count, extensions.data()),
              "vkEnumerateDeviceExtensionProperties(list)");
    std::unordered_set<std::string> available;
    for (const VkExtensionProperties& extension : extensions) {
      available.emplace(extension.extensionName);
    }
    const std::array<const char*, 2> required_extensions{
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    for (const char* extension : required_extensions) {
      if (!HasExtension(available, extension)) {
        Fail(std::string("Required Vulkan extension unavailable: ") +
             extension);
      }
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_support{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceFeatures2 supported{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &timeline_support;
    f.get_features2(physical_device, &supported);
    if (!timeline_support.timelineSemaphore) {
      Fail("Vulkan timeline semaphores are unavailable");
    }

    VkPhysicalDeviceExternalSemaphoreInfo semaphore_info{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
    semaphore_info.handleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    VkExternalSemaphoreProperties semaphore_properties{
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    f.get_semaphore_properties(physical_device, &semaphore_info,
                               &semaphore_properties);
    if ((semaphore_properties.externalSemaphoreFeatures &
         VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) == 0) {
      Fail("D3D12 fence import is unsupported by the Vulkan adapter");
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_create.queueFamilyIndex = queue_family;
    queue_create.queueCount = 1;
    queue_create.pQueuePriorities = &priority;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_enable{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline_enable.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo device_create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create.pNext = &timeline_enable;
    device_create.queueCreateInfoCount = 1;
    device_create.pQueueCreateInfos = &queue_create;
    device_create.enabledExtensionCount =
        static_cast<std::uint32_t>(required_extensions.size());
    device_create.ppEnabledExtensionNames = required_extensions.data();
    RequireVk(create_device(physical_device, &device_create, nullptr, &device),
              "vkCreateDevice");

    auto load = [&](auto& target, const char* name) {
      using Function = std::remove_reference_t<decltype(target)>;
      target = RequireDeviceProc<Function>(f.get_device_proc, device, name);
    };
    load(f.destroy_device, "vkDestroyDevice");
    load(f.get_device_queue, "vkGetDeviceQueue");
    load(f.create_image, "vkCreateImage");
    load(f.destroy_image, "vkDestroyImage");
    load(f.get_image_requirements, "vkGetImageMemoryRequirements");
    load(f.allocate_memory, "vkAllocateMemory");
    load(f.free_memory, "vkFreeMemory");
    load(f.bind_image_memory, "vkBindImageMemory");
    load(f.get_handle_properties, "vkGetMemoryWin32HandlePropertiesKHR");
    load(f.create_semaphore, "vkCreateSemaphore");
    load(f.destroy_semaphore, "vkDestroySemaphore");
    load(f.import_semaphore, "vkImportSemaphoreWin32HandleKHR");
    load(f.wait_semaphores, "vkWaitSemaphores");
    load(f.get_semaphore_value, "vkGetSemaphoreCounterValue");
    load(f.create_buffer, "vkCreateBuffer");
    load(f.destroy_buffer, "vkDestroyBuffer");
    load(f.get_buffer_requirements, "vkGetBufferMemoryRequirements");
    load(f.bind_buffer_memory, "vkBindBufferMemory");
    load(f.map_memory, "vkMapMemory");
    load(f.unmap_memory, "vkUnmapMemory");
    load(f.flush_memory, "vkFlushMappedMemoryRanges");
    load(f.invalidate_memory, "vkInvalidateMappedMemoryRanges");
    load(f.create_command_pool, "vkCreateCommandPool");
    load(f.destroy_command_pool, "vkDestroyCommandPool");
    load(f.allocate_command_buffers, "vkAllocateCommandBuffers");
    load(f.reset_command_pool, "vkResetCommandPool");
    load(f.begin_command_buffer, "vkBeginCommandBuffer");
    load(f.end_command_buffer, "vkEndCommandBuffer");
    load(f.cmd_pipeline_barrier, "vkCmdPipelineBarrier");
    load(f.cmd_copy_image_to_buffer, "vkCmdCopyImageToBuffer");
    load(f.cmd_copy_buffer_to_image, "vkCmdCopyBufferToImage");
    load(f.queue_submit, "vkQueueSubmit");
    load(f.device_wait_idle, "vkDeviceWaitIdle");
    f.get_device_queue(device, queue_family, 0, &queue);
    f.get_memory_properties(physical_device, &memory_properties);
  }

  void ValidateExternalSupport() const {
    VkPhysicalDeviceExternalImageFormatInfo external{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    external.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkPhysicalDeviceImageFormatInfo2 image{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    image.pNext = &external;
    image.format = format;
    image.type = VK_IMAGE_TYPE_2D;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkExternalImageFormatProperties external_properties{
        VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 properties{
        VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    properties.pNext = &external_properties;
    RequireVk(function.get_image_properties(physical_device, &image,
                                            &properties),
              "vkGetPhysicalDeviceImageFormatProperties2");
    const VkExternalMemoryProperties& memory =
        external_properties.externalMemoryProperties;
    if ((memory.externalMemoryFeatures &
         VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0 ||
        (memory.compatibleHandleTypes &
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT) == 0) {
      Fail("Vulkan cannot import the requested D3D12 image format");
    }
    if (description.width > properties.imageFormatProperties.maxExtent.width ||
        description.height >
            properties.imageFormatProperties.maxExtent.height) {
      Fail("External image size exceeds Vulkan limits");
    }
  }

  std::uint32_t PickMemoryType(std::uint32_t bits,
                               VkMemoryPropertyFlags required,
                               VkMemoryPropertyFlags preferred = 0) const {
    std::optional<std::uint32_t> fallback;
    for (std::uint32_t index = 0;
         index < memory_properties.memoryTypeCount; ++index) {
      if ((bits & (1u << index)) == 0) continue;
      const VkMemoryPropertyFlags flags =
          memory_properties.memoryTypes[index].propertyFlags;
      if ((flags & required) != required) continue;
      if ((flags & preferred) == preferred) return index;
      if (!fallback) fallback = index;
    }
    if (!fallback) Fail("No compatible Vulkan memory type found");
    return *fallback;
  }

  void ImportTimeline() {
    VkSemaphoreTypeCreateInfo type{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    create.pNext = &type;
    RequireVk(function.create_semaphore(device, &create, nullptr, &timeline),
              "vkCreateSemaphore(timeline)");
    WinHandle duplicate = DuplicateLocal(description.timeline_fence);
    VkImportSemaphoreWin32HandleInfoKHR import{
        VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    import.semaphore = timeline;
    import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    import.handle = duplicate.get();
    RequireVk(function.import_semaphore(device, &import),
              "vkImportSemaphoreWin32HandleKHR");
  }

  ImportedImage ImportImage(HandleToken token) {
    ImportedImage imported;
    try {
      VkExternalMemoryImageCreateInfo external{
          VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
      external.handleTypes =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.pNext = &external;
      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = format;
      image.extent = {description.width, description.height, 1};
      image.mipLevels = 1;
      image.arrayLayers = 1;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      RequireVk(
          function.create_image(device, &image, nullptr, &imported.image),
          "vkCreateImage(import alias)");
      VkMemoryRequirements requirements{};
      function.get_image_requirements(device, imported.image, &requirements);
      WinHandle duplicate = DuplicateLocal(token);
      VkMemoryWin32HandlePropertiesKHR handle_properties{
          VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
      RequireVk(function.get_handle_properties(
                    device,
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                    duplicate.get(), &handle_properties),
                "vkGetMemoryWin32HandlePropertiesKHR");
      const std::uint32_t memory_bits =
          requirements.memoryTypeBits & handle_properties.memoryTypeBits;
      if (memory_bits == 0) {
        Fail("D3D12 resource and Vulkan image share no memory type");
      }
      VkMemoryDedicatedAllocateInfo dedicated{
          VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
      dedicated.image = imported.image;
      VkImportMemoryWin32HandleInfoKHR import{
          VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
      import.pNext = &dedicated;
      import.handleType =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
      import.handle = duplicate.get();
      VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocate.pNext = &import;
      allocate.allocationSize = requirements.size;
      allocate.memoryTypeIndex = PickMemoryType(
          memory_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      RequireVk(function.allocate_memory(device, &allocate, nullptr,
                                         &imported.memory),
                "vkAllocateMemory(import D3D12 resource)");
      RequireVk(function.bind_image_memory(device, imported.image,
                                           imported.memory, 0),
                "vkBindImageMemory(import D3D12 resource)");
      return imported;
    } catch (...) {
      DestroyImage(imported);
      throw;
    }
  }

  Buffer CreateBuffer(VkBufferUsageFlags usage) {
    Buffer buffer;
    try {
      buffer.size = static_cast<VkDeviceSize>(description.width) *
                    description.height * description.bytes_per_pixel;
      VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      create.size = buffer.size;
      create.usage = usage;
      create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      RequireVk(
          function.create_buffer(device, &create, nullptr, &buffer.buffer),
          "vkCreateBuffer");
      VkMemoryRequirements requirements{};
      function.get_buffer_requirements(device, buffer.buffer, &requirements);
      const std::uint32_t memory_type = PickMemoryType(
          requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      buffer.coherent =
          (memory_properties.memoryTypes[memory_type].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocate.allocationSize = requirements.size;
      allocate.memoryTypeIndex = memory_type;
      RequireVk(function.allocate_memory(device, &allocate, nullptr,
                                         &buffer.memory),
                "vkAllocateMemory(buffer)");
      RequireVk(function.bind_buffer_memory(device, buffer.buffer,
                                            buffer.memory, 0),
                "vkBindBufferMemory");
      return buffer;
    } catch (...) {
      DestroyBuffer(buffer);
      throw;
    }
  }

  void CreateSlot(std::size_t index) {
    Slot& slot = slots[index];
    slot.input = ImportImage(description.slots[index].input);
    slot.output = ImportImage(description.slots[index].output);
    slot.upload = CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    slot.readback = CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkCommandPoolCreateInfo pool{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = queue_family;
    RequireVk(function.create_command_pool(device, &pool, nullptr,
                                           &slot.command_pool),
              "vkCreateCommandPool");
    VkCommandBufferAllocateInfo allocate{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = slot.command_pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount =
        static_cast<std::uint32_t>(slot.commands.size());
    RequireVk(function.allocate_command_buffers(device, &allocate,
                                                slot.commands.data()),
              "vkAllocateCommandBuffers");
    InitializeLayouts(slot);
  }

  void InitializeLayouts(Slot& slot) {
    VkCommandBuffer command = slot.commands[0];
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    RequireVk(function.begin_command_buffer(command, &begin),
              "vkBeginCommandBuffer(layout)");
    std::array<VkImageMemoryBarrier, 2> barriers{};
    const std::array<VkImage, 2> images{slot.input.image, slot.output.image};
    for (std::size_t index = 0; index < barriers.size(); ++index) {
      barriers[index].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[index].srcAccessMask = 0;
      barriers[index].dstAccessMask =
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      barriers[index].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[index].newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].image = images[index];
      barriers[index].subresourceRange =
          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    function.cmd_pipeline_barrier(
        command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
        static_cast<std::uint32_t>(barriers.size()), barriers.data());
    RequireVk(function.end_command_buffer(command),
              "vkEndCommandBuffer(layout)");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    RequireVk(function.queue_submit(queue, 1, &submit, VK_NULL_HANDLE),
              "vkQueueSubmit(layout)");
    RequireVk(function.device_wait_idle(device), "vkDeviceWaitIdle(layout)");
  }

  void DestroyBuffer(Buffer& buffer) noexcept {
    if (buffer.buffer != VK_NULL_HANDLE) {
      function.destroy_buffer(device, buffer.buffer, nullptr);
    }
    if (buffer.memory != VK_NULL_HANDLE) {
      function.free_memory(device, buffer.memory, nullptr);
    }
    buffer = {};
  }

  void DestroyImage(ImportedImage& image) noexcept {
    if (image.image != VK_NULL_HANDLE) {
      function.destroy_image(device, image.image, nullptr);
    }
    if (image.memory != VK_NULL_HANDLE) {
      function.free_memory(device, image.memory, nullptr);
    }
    image = {};
  }

  void DestroySlot(Slot& slot) noexcept {
    if (device == VK_NULL_HANDLE) return;
    if (slot.command_pool != VK_NULL_HANDLE) {
      function.destroy_command_pool(device, slot.command_pool, nullptr);
    }
    DestroyBuffer(slot.upload);
    DestroyBuffer(slot.readback);
    DestroyImage(slot.input);
    DestroyImage(slot.output);
    slot = {};
  }

  bool Wait(std::uint64_t value, std::uint32_t timeout_ms,
            std::string& error) const {
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &timeline;
    wait.pValues = &value;
    const std::uint64_t timeout_ns =
        static_cast<std::uint64_t>(timeout_ms) * 1000000ull;
    const VkResult result =
        function.wait_semaphores(device, &wait, timeout_ns);
    if (result == VK_SUCCESS) return true;
    if (result == VK_TIMEOUT) {
      error = "Vulkan external timeline wait timed out";
      return false;
    }
    error = "vkWaitSemaphores failed with VkResult " +
            std::to_string(static_cast<int>(result));
    return false;
  }

  void WriteBuffer(Buffer& buffer, std::span<const std::uint8_t> pixels) {
    if (pixels.size() != static_cast<std::size_t>(buffer.size)) {
      Fail("Vulkan upload size mismatch");
    }
    void* mapped = nullptr;
    RequireVk(function.map_memory(device, buffer.memory, 0, buffer.size, 0,
                                  &mapped),
              "vkMapMemory(upload)");
    std::memcpy(mapped, pixels.data(), pixels.size());
    if (!buffer.coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      RequireVk(function.flush_memory(device, 1, &range),
                "vkFlushMappedMemoryRanges");
    }
    function.unmap_memory(device, buffer.memory);
  }

  std::vector<std::uint8_t> ReadBuffer(Buffer& buffer) {
    void* mapped = nullptr;
    RequireVk(function.map_memory(device, buffer.memory, 0, buffer.size, 0,
                                  &mapped),
              "vkMapMemory(readback)");
    if (!buffer.coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffer.memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      RequireVk(function.invalidate_memory(device, 1, &range),
                "vkInvalidateMappedMemoryRanges");
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(buffer.size));
    std::memcpy(result.data(), mapped, result.size());
    function.unmap_memory(device, buffer.memory);
    return result;
  }

  void Begin(Slot& slot, VkCommandBuffer command) {
    RequireVk(function.reset_command_pool(device, slot.command_pool, 0),
              "vkResetCommandPool");
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    RequireVk(function.begin_command_buffer(command, &begin),
              "vkBeginCommandBuffer");
  }

  void Transition(VkCommandBuffer command, VkImage image,
                  VkAccessFlags source_access, VkAccessFlags destination_access,
                  VkImageLayout old_layout, VkImageLayout new_layout,
                  VkPipelineStageFlags source_stage,
                  VkPipelineStageFlags destination_stage) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = source_access;
    barrier.dstAccessMask = destination_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    function.cmd_pipeline_barrier(command, source_stage, destination_stage, 0,
                                  0, nullptr, 0, nullptr, 1, &barrier);
  }

  void Submit(VkCommandBuffer command, std::optional<std::uint64_t> wait_value,
              std::uint64_t signal_value) {
    VkTimelineSemaphoreSubmitInfo values{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    const std::uint64_t wait_storage = wait_value.value_or(0);
    if (wait_value) {
      values.waitSemaphoreValueCount = 1;
      values.pWaitSemaphoreValues = &wait_storage;
    }
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &signal_value;
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &values;
    if (wait_value) {
      submit.waitSemaphoreCount = 1;
      submit.pWaitSemaphores = &timeline;
      submit.pWaitDstStageMask = &wait_stage;
    }
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &timeline;
    RequireVk(function.queue_submit(queue, 1, &submit, VK_NULL_HANDLE),
              "vkQueueSubmit(external timeline)");
  }

  bool UploadInput(std::size_t slot_index,
                   std::span<const std::uint8_t> pixels,
                   std::uint64_t signal_value, std::uint32_t timeout_ms,
                   std::string& error) {
    try {
      if (slot_index >= slots.size() || signal_value == 0) {
        Fail("Invalid Vulkan input submission");
      }
      Slot& slot = slots[slot_index];
      WriteBuffer(slot.upload, pixels);
      const VkCommandBuffer command = slot.commands[0];
      Begin(slot, command);
      VkBufferMemoryBarrier host_write{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      host_write.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      host_write.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      host_write.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      host_write.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      host_write.buffer = slot.upload.buffer;
      host_write.offset = 0;
      host_write.size = VK_WHOLE_SIZE;
      function.cmd_pipeline_barrier(
          command, VK_PIPELINE_STAGE_HOST_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &host_write, 0,
          nullptr);
      Transition(command, slot.input.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
      VkBufferImageCopy copy{};
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copy.imageExtent = {description.width, description.height, 1};
      function.cmd_copy_buffer_to_image(
          command, slot.upload.buffer, slot.input.image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
      Transition(command, slot.input.image, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      RequireVk(function.end_command_buffer(command), "vkEndCommandBuffer");
      Submit(command, std::nullopt, signal_value);
      return Wait(signal_value, timeout_ms, error);
    } catch (const std::exception& exception) {
      error = exception.what();
      return false;
    }
  }

  bool ReadOutput(std::size_t slot_index, std::uint64_t wait_value,
                  std::uint64_t signal_value, std::uint32_t timeout_ms,
                  std::vector<std::uint8_t>& pixels, std::string& error) {
    try {
      if (slot_index >= slots.size() || wait_value == 0 ||
          signal_value <= wait_value) {
        Fail("Invalid Vulkan output submission");
      }
      Slot& slot = slots[slot_index];
      const VkCommandBuffer command = slot.commands[1];
      Begin(slot, command);
      Transition(command, slot.output.image, 0, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
      VkBufferImageCopy copy{};
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copy.imageExtent = {description.width, description.height, 1};
      function.cmd_copy_image_to_buffer(
          command, slot.output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          slot.readback.buffer, 1, &copy);
      Transition(command, slot.output.image, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      VkBufferMemoryBarrier host_read{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      host_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      host_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      host_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      host_read.buffer = slot.readback.buffer;
      host_read.offset = 0;
      host_read.size = VK_WHOLE_SIZE;
      function.cmd_pipeline_barrier(
          command, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host_read, 0,
          nullptr);
      RequireVk(function.end_command_buffer(command), "vkEndCommandBuffer");
      Submit(command, wait_value, signal_value);
      if (!Wait(signal_value, timeout_ms, error)) return false;
      pixels = ReadBuffer(slot.readback);
      return true;
    } catch (const std::exception& exception) {
      error = exception.what();
      return false;
    }
  }

  bool Healthy(std::string& error) const {
    if (device == VK_NULL_HANDLE || timeline == VK_NULL_HANDLE) {
      error = "Vulkan external transport is not initialized";
      return false;
    }
    std::uint64_t value = 0;
    const VkResult result =
        function.get_semaphore_value(device, timeline, &value);
    if (result != VK_SUCCESS) {
      error = "vkGetSemaphoreCounterValue failed with VkResult " +
              std::to_string(static_cast<int>(result));
      return false;
    }
    error.clear();
    return true;
  }

  ExternalTransportDescription description{};
  Module module{};
  Functions function{};
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkSemaphore timeline = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::array<Slot, FrameRing::kDefaultSlotCount> slots{};
};

std::unique_ptr<VulkanExternalTransport> VulkanExternalTransport::Create(
    const ExternalTransportDescription& description, std::string& error) {
  try {
    auto implementation = std::make_unique<Impl>(description);
    implementation->Initialize();
    error.clear();
    return std::unique_ptr<VulkanExternalTransport>(
        new VulkanExternalTransport(std::move(implementation)));
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
}

VulkanExternalTransport::VulkanExternalTransport(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

VulkanExternalTransport::~VulkanExternalTransport() = default;

bool VulkanExternalTransport::UploadInput(
    std::size_t slot, std::span<const std::uint8_t> pixels,
    std::uint64_t signal_value, std::uint32_t timeout_ms,
    std::string& error) {
  return implementation_->UploadInput(slot, pixels, signal_value, timeout_ms,
                                      error);
}

bool VulkanExternalTransport::ReadOutput(
    std::size_t slot, std::uint64_t wait_value, std::uint64_t signal_value,
    std::uint32_t timeout_ms, std::vector<std::uint8_t>& pixels,
    std::string& error) {
  return implementation_->ReadOutput(slot, wait_value, signal_value,
                                     timeout_ms, pixels, error);
}

bool VulkanExternalTransport::Healthy(std::string& error) const {
  return implementation_->Healthy(error);
}

std::uint64_t VulkanExternalTransport::adapter_luid() const noexcept {
  return implementation_->description.adapter_luid;
}

}  // namespace neuralstudio::vulkan_bridge
