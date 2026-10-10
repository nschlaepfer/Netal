/* NeuralStudio guide-complete diagnostic frame source - SPDX-License-Identifier: MIT */

#pragma once

#include "neuralstudio_adapter.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct ID3D12Device;

namespace neuralstudio {

class DiagnosticD3D12 {
 public:
  DiagnosticD3D12();
  ~DiagnosticD3D12();
  DiagnosticD3D12(const DiagnosticD3D12&) = delete;
  DiagnosticD3D12& operator=(const DiagnosticD3D12&) = delete;
  DiagnosticD3D12(DiagnosticD3D12&&) noexcept;
  DiagnosticD3D12& operator=(DiagnosticD3D12&&) noexcept;

  bool Initialize(std::uint32_t width, std::uint32_t height,
                  std::string& error);
  bool WriteFrame(std::uint64_t frame_index, std::string& error);
  bool ReadOutput(std::vector<std::uint16_t>& output, std::string& error);
  bool SaveBmp(const std::filesystem::path& path,
               std::span<const std::uint16_t> rgba16f,
               std::string& error) const;

  [[nodiscard]] ID3D12Device* device() const noexcept;
  [[nodiscard]] ns_device_desc device_desc() const;
  [[nodiscard]] ns_session_desc session_desc(
      const std::filesystem::path& package_root) const;
  [[nodiscard]] std::vector<ns_resource> inputs() const;
  [[nodiscard]] ns_resource output() const;
  [[nodiscard]] std::span<const std::uint16_t> input_pixels() const noexcept;
  [[nodiscard]] std::span<const std::uint16_t> output_canary() const noexcept;
  [[nodiscard]] std::uint32_t width() const noexcept;
  [[nodiscard]] std::uint32_t height() const noexcept;
  [[nodiscard]] std::uint32_t severe_debug_messages() const;
  [[nodiscard]] std::uint64_t current_process_handle_count() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::uint64_t Fnv1aBytes(std::span<const std::uint8_t> bytes) noexcept;
std::size_t ChangedBytes(std::span<const std::uint8_t> left,
                         std::span<const std::uint8_t> right) noexcept;

}  // namespace neuralstudio
