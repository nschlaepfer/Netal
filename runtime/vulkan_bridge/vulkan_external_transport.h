/*
 * NeuralStudio Vulkan <-> D3D12 external-memory transport
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "bridge_core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace neuralstudio::vulkan_bridge {

struct ExternalImagePair {
  HandleToken input = 0;
  HandleToken output = 0;
};

struct ExternalTransportDescription {
  std::uint64_t adapter_luid = 0;
  HandleToken timeline_fence = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t bytes_per_pixel = 0;
  std::array<ExternalImagePair, FrameRing::kDefaultSlotCount> slots{};
};

class VulkanExternalTransport final {
 public:
  static std::unique_ptr<VulkanExternalTransport> Create(
      const ExternalTransportDescription& description, std::string& error);

  ~VulkanExternalTransport();
  VulkanExternalTransport(const VulkanExternalTransport&) = delete;
  VulkanExternalTransport& operator=(const VulkanExternalTransport&) = delete;

  [[nodiscard]] bool UploadInput(std::size_t slot,
                                 std::span<const std::uint8_t> pixels,
                                 std::uint64_t signal_value,
                                 std::uint32_t timeout_ms,
                                 std::string& error);

  [[nodiscard]] bool ReadOutput(std::size_t slot,
                                std::uint64_t wait_value,
                                std::uint64_t signal_value,
                                std::uint32_t timeout_ms,
                                std::vector<std::uint8_t>& pixels,
                                std::string& error);

  [[nodiscard]] bool Healthy(std::string& error) const;
  [[nodiscard]] std::uint64_t adapter_luid() const noexcept;

 private:
  struct Impl;
  explicit VulkanExternalTransport(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace neuralstudio::vulkan_bridge
