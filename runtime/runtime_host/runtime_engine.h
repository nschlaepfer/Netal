/* NeuralStudio Runtime v0 diagnostic engine - SPDX-License-Identifier: MIT */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace neuralstudio {

enum class DiagnosticTestFailure { None, Crash, Timeout, DeviceLoss };

struct DiagnosticOptions {
  std::filesystem::path package_root;
  std::filesystem::path output_path;
  std::uint32_t width = 640;
  std::uint32_t height = 360;
  std::uint32_t evaluation_frames = 1;
  bool inference_armed = false;
  DiagnosticTestFailure test_failure = DiagnosticTestFailure::None;
};

struct DiagnosticResult {
  bool ok = false;
  std::string state = "Standby";
  std::string detail;
  std::string package_id;
  std::string adapter_version;
  std::filesystem::path output_path;
  std::uint64_t evaluations = 0;
  std::uint64_t output_hash = 0;
  std::uint64_t input_hash = 0;
  std::size_t changed_from_canary_bytes = 0;
  std::size_t changed_from_input_bytes = 0;
  double last_inference_ms = 0.0;
  std::uint64_t start_handle_count = 0;
  std::uint64_t end_handle_count = 0;
  std::uint32_t severe_d3d_messages = 0;
};

DiagnosticResult RunDiagnostic(const DiagnosticOptions& options);
std::string DiagnosticResultJson(const DiagnosticResult& result);
std::string JsonEscape(const std::string& value);

}  // namespace neuralstudio
