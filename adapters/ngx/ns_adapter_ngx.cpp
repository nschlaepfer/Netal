/* NeuralStudio NGX DLSSNR adapter - SPDX-License-Identifier: MIT */

#define NS_ADAPTER_IMPLEMENTATION
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "neuralstudio_adapter.h"
#include "client.h"

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <string>

using Microsoft::WRL::ComPtr;
namespace worker = neuralstudio::dlssnr_worker;
namespace protocol = neuralstudio::vulkan_bridge::dlssnr_worker;

struct ns_session_t {
  ns_extent2d extent{};
  ns_format color_format = ns_format_unknown;
  ns_format output_format = ns_format_unknown;
  ns_conventions conventions{};
  protocol::Options options{};
  worker::Client worker;
  std::array<std::uint64_t, 4> handles{};
  std::uint32_t generation = 0;
  std::uint32_t consecutive_failures = 0;
  bool configured = false;
  bool reset_required = true;
  bool circuit_open = false;
  std::uint32_t pending_test_failure = 0;
  std::chrono::steady_clock::time_point retry_after{};
};

namespace {

constexpr std::uint32_t kNvidiaVendor = 0x10DEu;
constexpr std::uint32_t kControlTimeoutMs = 15000;
constexpr std::uint32_t kCircuitFailureLimit = 3;

struct AdapterState {
  std::mutex mutex;
  ns_host_v1 host{};
  ComPtr<ID3D12Device> device;
  std::filesystem::path package_root;
  std::filesystem::path runtime_directory;
  std::filesystem::path worker_binary;
  bool initialized = false;
};

AdapterState g_state;

void Detail(char* destination, std::size_t capacity, const char* text) {
  if (destination == nullptr || capacity == 0) return;
  std::snprintf(destination, capacity, "%s", text == nullptr ? "" : text);
}

void Log(ns_log_level level, const std::string& message) {
  if (g_state.host.log != nullptr) {
    g_state.host.log(g_state.host.user_data, level, message.c_str());
  }
}

const ns_resource* FindInput(const ns_frame* frame, ns_guide_kind kind) {
  if (frame == nullptr || frame->inputs == nullptr) return nullptr;
  for (std::uint32_t index = 0; index < frame->input_count; ++index) {
    if (frame->inputs[index].kind == kind) return &frame->inputs[index];
  }
  return nullptr;
}

bool ValidSharedResource(const ns_resource* resource, ns_format format,
                         ns_extent2d extent) {
  return resource != nullptr &&
         resource->struct_size >= sizeof(ns_resource) &&
         resource->api == ns_api_d3d12 && resource->format == format &&
         resource->extent.width == extent.width &&
         resource->extent.height == extent.height &&
         resource->shared_handle != 0;
}

void RecordFailure(ns_session_t& session) {
  session.worker.Abort();
  session.configured = false;
  session.reset_required = true;
  ++session.consecutive_failures;
  if (session.consecutive_failures >= kCircuitFailureLimit) {
    session.circuit_open = true;
    return;
  }
  const std::uint32_t delay_ms =
      100u << std::min(session.consecutive_failures - 1, 5u);
  session.retry_after = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(delay_ms);
}

bool ConfigureWorker(ns_session_t& session,
                     const std::array<std::uint64_t, 4>& handles,
                     std::string& error) {
  if (session.circuit_open ||
      std::chrono::steady_clock::now() < session.retry_after) {
    error = session.circuit_open ? "DLSSNR worker circuit breaker is open"
                                 : "DLSSNR worker restart is backing off";
    return false;
  }
  ++session.generation;
  if (!session.worker.Start(g_state.worker_binary, g_state.runtime_directory,
                            session.generation, error)) {
    RecordFailure(session);
    return false;
  }
  std::array<HANDLE, 4> native{};
  for (std::size_t index = 0; index < native.size(); ++index) {
    native[index] = reinterpret_cast<HANDLE>(handles[index]);
  }
  if (!session.worker.Configure(g_state.device.Get(), session.extent.width,
                                session.extent.height, native,
                                session.options, error)) {
    RecordFailure(session);
    return false;
  }
  session.handles = handles;
  session.configured = true;
  session.reset_required = true;
  return true;
}

ns_status NS_CALL GetInfo(ns_adapter_info* out_info) {
  if (out_info == nullptr || out_info->struct_size < sizeof(ns_adapter_info)) {
    return ns_status_invalid_argument;
  }
  out_info->package_id = "com.nvidia.ngx.dlss-nr";
  out_info->display_name = "NVIDIA DLSS Neural Rendering";
  out_info->adapter_version = "0.1.0";
  out_info->backend_name = "NGX feature 18 isolated worker";
  out_info->abi_version_major = NS_ADAPTER_ABI_VERSION_MAJOR;
  out_info->abi_version_minor = NS_ADAPTER_ABI_VERSION_MINOR;
  out_info->supports_sandboxed_hosting = 1;
  out_info->requires_state_guard = 0;
  return ns_status_ok;
}

ns_status NS_CALL Probe(const ns_device_desc* device,
                        ns_capability_report* out_report) {
  if (device == nullptr || out_report == nullptr ||
      device->struct_size < sizeof(ns_device_desc) ||
      out_report->struct_size < sizeof(ns_capability_report)) {
    return ns_status_invalid_argument;
  }
  out_report->supported = 0;
  out_report->requires_signature_check = 1;
  out_report->estimated_vram_bytes = 0;
  if (device->api != ns_api_d3d12 || device->device == nullptr) {
    Detail(out_report->detail, sizeof(out_report->detail),
           "A D3D12 source/output device is required.");
    return ns_status_ok;
  }
  if (device->vendor_id != kNvidiaVendor) {
    Detail(out_report->detail, sizeof(out_report->detail),
           "The experimental DLSSNR runtime requires an NVIDIA GPU.");
    return ns_status_ok;
  }
  out_report->supported = 1;
  out_report->estimated_vram_bytes = 2400ull * 1024ull * 1024ull;
  Detail(out_report->detail, sizeof(out_report->detail),
         "D3D12 NVIDIA device accepted; manifest and runtime signature checks remain mandatory.");
  return ns_status_ok;
}

ns_status NS_CALL Initialize(const ns_host_v1* host,
                             const ns_init_desc* desc) {
  if (host == nullptr || desc == nullptr ||
      host->struct_size < sizeof(ns_host_v1) ||
      desc->struct_size < sizeof(ns_init_desc) ||
      desc->device.api != ns_api_d3d12 || desc->device.device == nullptr ||
      desc->package_root == nullptr) {
    return ns_status_invalid_argument;
  }
  std::scoped_lock lock(g_state.mutex);
  if (g_state.initialized) return ns_status_invalid_argument;
  try {
    g_state.host = *host;
    g_state.device = static_cast<ID3D12Device*>(desc->device.device);
    g_state.package_root = std::filesystem::weakly_canonical(desc->package_root);
    g_state.runtime_directory = g_state.package_root / L"runtime";
    g_state.worker_binary =
        g_state.runtime_directory / L"neuralstudio-dlssnr-bridge-worker.exe";
    if (!std::filesystem::is_regular_file(g_state.worker_binary) ||
        !std::filesystem::is_regular_file(g_state.runtime_directory /
                                          L"sl.interposer.dll")) {
      g_state.device.Reset();
      g_state.host = {};
      return ns_status_runtime_missing;
    }
    g_state.initialized = true;
    Log(ns_log_info, "NGX adapter initialized; model execution remains isolated.");
    return ns_status_ok;
  } catch (...) {
    g_state.device.Reset();
    g_state.host = {};
    return ns_status_internal_error;
  }
}

void NS_CALL Shutdown() {
  std::scoped_lock lock(g_state.mutex);
  g_state.initialized = false;
  g_state.device.Reset();
  g_state.host = {};
  g_state.package_root.clear();
  g_state.runtime_directory.clear();
  g_state.worker_binary.clear();
}

ns_status NS_CALL CreateSession(const ns_session_desc* desc,
                                ns_session_handle* out_session) {
  if (desc == nullptr || out_session == nullptr ||
      desc->struct_size < sizeof(ns_session_desc)) {
    return ns_status_invalid_argument;
  }
  std::scoped_lock lock(g_state.mutex);
  if (!g_state.initialized) return ns_status_not_initialized;
  if (desc->input_extent.width < 64 || desc->input_extent.height < 64 ||
      desc->input_extent.width != desc->output_extent.width ||
      desc->input_extent.height != desc->output_extent.height ||
      desc->color_format != ns_format_r16g16b16a16_float ||
      desc->output_format != ns_format_r16g16b16a16_float) {
    return ns_status_unsupported;
  }
  auto session = std::unique_ptr<ns_session_t>(new (std::nothrow) ns_session_t);
  if (!session) return ns_status_out_of_memory;
  session->extent = desc->input_extent;
  session->color_format = desc->color_format;
  session->output_format = desc->output_format;
  session->conventions = desc->conventions;
  *out_session = session.release();
  return ns_status_ok;
}

ns_status NS_CALL DestroySession(ns_session_handle session) {
  if (session == nullptr) return ns_status_invalid_argument;
  std::scoped_lock lock(g_state.mutex);
  std::unique_ptr<ns_session_t> owned(session);
  std::string ignored;
  (void)owned->worker.Shutdown(2000, ignored);
  return ns_status_ok;
}

ns_status NS_CALL ResetHistory(ns_session_handle session,
                               ns_reset_reason reason) {
  if (session == nullptr || reason == ns_reset_none) {
    return ns_status_invalid_argument;
  }
  std::scoped_lock lock(g_state.mutex);
  session->reset_required = true;
  session->circuit_open = false;
  session->consecutive_failures = 0;
  session->retry_after = {};
  return ns_status_ok;
}

ns_status NS_CALL Evaluate(ns_session_handle session, const ns_frame* frame,
                           ns_evaluate_result* out_result) {
  if (session == nullptr || frame == nullptr || out_result == nullptr ||
      frame->struct_size < sizeof(ns_frame) ||
      out_result->struct_size < sizeof(ns_evaluate_result)) {
    return ns_status_invalid_argument;
  }
  std::scoped_lock lock(g_state.mutex);
  out_result->evaluated = 0;
  out_result->output_fresh = 0;
  out_result->used_fallback_path = 0;
  out_result->inference_ms = 0.0;
  out_result->requested_reset = ns_reset_none;
  out_result->submission = {};
  out_result->submission.struct_size = sizeof(ns_submission);
  Detail(out_result->detail, sizeof(out_result->detail), "Not evaluated");

  if (!g_state.initialized) return ns_status_not_initialized;
  const ns_resource* color = FindInput(frame, ns_guide_color);
  const ns_resource* depth = FindInput(frame, ns_guide_depth);
  const ns_resource* motion = FindInput(frame, ns_guide_motion_vectors);
  if (!ValidSharedResource(color, ns_format_r16g16b16a16_float,
                           session->extent) ||
      !ValidSharedResource(depth, ns_format_r32_float, session->extent) ||
      !ValidSharedResource(motion, ns_format_r16g16_float, session->extent) ||
      !ValidSharedResource(&frame->output, ns_format_r16g16b16a16_float,
                           session->extent)) {
    Detail(out_result->detail, sizeof(out_result->detail),
           "Color, depth, motion vectors, and output must be matching shared D3D12 textures.");
    return ns_status_guide_missing;
  }
  std::array<std::uint64_t, 4> handles{
      color->shared_handle, frame->output.shared_handle,
      depth->shared_handle, motion->shared_handle};
  std::string error;
  if (!session->configured || session->handles != handles ||
      !session->worker.alive()) {
    session->worker.Abort();
    session->configured = false;
    if (!ConfigureWorker(*session, handles, error)) {
      Detail(out_result->detail, sizeof(out_result->detail), error.c_str());
      return session->circuit_open ? ns_status_runtime_rejected
                                   : ns_status_internal_error;
    }
  }
  const bool reset = frame->reset != ns_reset_none;
  if (session->reset_required && !reset) {
    out_result->requested_reset = ns_reset_session_start;
    Detail(out_result->detail, sizeof(out_result->detail),
           "First evaluation after creation or recovery requires a reset.");
    return ns_status_needs_reset;
  }

  if (session->pending_test_failure != 0) {
    const std::uint32_t requested = session->pending_test_failure;
    session->pending_test_failure = 0;
    worker::FailureInjection injection = worker::FailureInjection::crash;
    if (requested == 2) injection = worker::FailureInjection::timeout;
    if (requested == 3) injection = worker::FailureInjection::device_loss;
    std::string injection_error;
    (void)session->worker.InjectFailure(injection, 500, injection_error);
    RecordFailure(*session);
    Detail(out_result->detail, sizeof(out_result->detail),
           injection_error.empty() ? "Injected isolated-worker failure"
                                   : injection_error.c_str());
    return ns_status_internal_error;
  }

  const auto started = std::chrono::steady_clock::now();
  protocol::FrameConstants constants;
  constants.motion_scale_x = session->conventions.motion_vector_scale.x;
  constants.motion_scale_y = session->conventions.motion_vector_scale.y;
  constants.jitter_x = frame->jitter.x;
  constants.jitter_y = frame->jitter.y;
  constants.depth_inverted =
      session->conventions.depth_direction == ns_depth_inverse ? 1u : 0u;
  constants.camera_aspect =
      static_cast<float>(session->extent.width) /
      static_cast<float>(session->extent.height);
  if (frame->camera.valid != 0) {
    constants.camera_near = frame->camera.near_plane;
    constants.camera_far = frame->camera.far_plane;
    constants.camera_fov_radians = frame->camera.vertical_fov_radians;
  }
  const worker::Evaluation evaluated = session->worker.Evaluate(
      frame->frame_index + 1, static_cast<std::uint32_t>(frame->frame_index % 3),
      reset, constants, kControlTimeoutMs, error);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  out_result->inference_ms =
      std::chrono::duration<double, std::milli>(elapsed).count();
  if (evaluated.status != protocol::Status::ok ||
      evaluated.debug_severe_messages != 0) {
    RecordFailure(*session);
    Detail(out_result->detail, sizeof(out_result->detail),
           error.empty() ? "Isolated DLSSNR evaluation failed" : error.c_str());
    /* The worker owns a private model device. Its removal does not mean the
       Runtime's source/output device was removed, so surface this as an
       isolated adapter-host failure and let the broker pass the source through. */
    return ns_status_internal_error;
  }
  session->configured = true;
  session->reset_required = false;
  session->consecutive_failures = 0;
  session->circuit_open = false;
  out_result->evaluated = 1;
  out_result->output_fresh = 1;
  Detail(out_result->detail, sizeof(out_result->detail),
         "DLSSNR worker produced a fresh output");
  return ns_status_ok;
}

ns_status ApplySetting(ns_session_t& session, const char* key,
                       const ns_value* value, bool& requires_recreate) {
  if (key == nullptr || value == nullptr) return ns_status_invalid_argument;
  requires_recreate = false;
  const std::string name(key);
  auto set_float = [&](float& target) {
    if (value->type != ns_value_float || !std::isfinite(value->float_value) ||
        value->float_value < 0.0 || value->float_value > 4.0) {
      return false;
    }
    target = static_cast<float>(value->float_value);
    return true;
  };
  if (name == "NRIntensity") {
    if (!set_float(session.options.intensity)) return ns_status_invalid_argument;
  } else if (name == "NRLocalTone") {
    if (!set_float(session.options.local_tone_strength)) return ns_status_invalid_argument;
  } else if (name == "NRLocalStructure") {
    if (!set_float(session.options.local_structure_strength)) return ns_status_invalid_argument;
  } else if (name == "NRGlobalTone") {
    if (!set_float(session.options.global_tone_strength)) return ns_status_invalid_argument;
  } else if (name == "NRSkinStructure") {
    if (!set_float(session.options.skin_structure_strength)) return ns_status_invalid_argument;
  } else if (name == "NRAutoMask") {
    if (value->type != ns_value_bool) return ns_status_invalid_argument;
    session.options.use_auto_mask = value->int_value != 0 ? 1u : 0u;
  } else if (name == "NRStyle" || name == "NRPreset" ||
             name == "NRPerformanceMode") {
    if (value->type != ns_value_int || value->int_value < 0 ||
        value->int_value > 3) return ns_status_invalid_argument;
    if (name == "NRStyle") session.options.style = static_cast<std::uint32_t>(value->int_value);
    if (name == "NRPreset") session.options.render_preset = static_cast<std::uint32_t>(value->int_value);
    if (name == "NRPerformanceMode") session.options.performance_mode = static_cast<std::uint32_t>(value->int_value);
    requires_recreate = name != "NRStyle";
  } else {
    return ns_status_unsupported;
  }
  return ns_status_ok;
}

ns_status NS_CALL SetSetting(ns_session_handle session, const char* key,
                             const ns_value* value) {
  if (session == nullptr) return ns_status_invalid_argument;
  std::scoped_lock lock(g_state.mutex);
  if (key != nullptr && std::strcmp(key, "NeuralStudioTestFailure") == 0) {
    char enabled[8]{};
    if (GetEnvironmentVariableA("NEURALSTUDIO_TEST_MODE", enabled,
                                static_cast<DWORD>(sizeof(enabled))) == 0 ||
        std::strcmp(enabled, "1") != 0 || value == nullptr ||
        value->type != ns_value_int || value->int_value < 1 ||
        value->int_value > 3) {
      return ns_status_unsupported;
    }
    session->pending_test_failure =
        static_cast<std::uint32_t>(value->int_value);
    return ns_status_ok;
  }
  bool recreate = false;
  const ns_status status = ApplySetting(*session, key, value, recreate);
  if (status != ns_status_ok) return status;
  if (recreate) {
    session->worker.Abort();
    session->configured = false;
    session->reset_required = true;
    return ns_status_needs_recreate;
  }
  if (session->configured) {
    std::string error;
    if (!session->worker.SetOptions(session->options, kControlTimeoutMs,
                                    error)) {
      RecordFailure(*session);
      Log(ns_log_error, error);
      return ns_status_internal_error;
    }
  }
  return ns_status_ok;
}

ns_status NS_CALL GetSetting(ns_session_handle session, const char* key,
                             ns_value* out_value) {
  if (session == nullptr || key == nullptr || out_value == nullptr) {
    return ns_status_invalid_argument;
  }
  std::scoped_lock lock(g_state.mutex);
  const std::string name(key);
  if (name == "NRAutoMask") {
    out_value->type = ns_value_bool;
    out_value->int_value = session->options.use_auto_mask != 0;
    return ns_status_ok;
  }
  if (name == "NRStyle" || name == "NRPreset" ||
      name == "NRPerformanceMode") {
    out_value->type = ns_value_int;
    if (name == "NRStyle") out_value->int_value = session->options.style;
    if (name == "NRPreset") out_value->int_value = session->options.render_preset;
    if (name == "NRPerformanceMode") out_value->int_value = session->options.performance_mode;
    return ns_status_ok;
  }
  out_value->type = ns_value_float;
  if (name == "NRIntensity") out_value->float_value = session->options.intensity;
  else if (name == "NRLocalTone") out_value->float_value = session->options.local_tone_strength;
  else if (name == "NRLocalStructure") out_value->float_value = session->options.local_structure_strength;
  else if (name == "NRGlobalTone") out_value->float_value = session->options.global_tone_strength;
  else if (name == "NRSkinStructure") out_value->float_value = session->options.skin_structure_strength;
  else return ns_status_unsupported;
  return ns_status_ok;
}

ns_status NS_CALL QueryVram(ns_session_handle, std::uint64_t* out_bytes) {
  if (out_bytes == nullptr) return ns_status_invalid_argument;
  *out_bytes = 0;
  return ns_status_not_implemented;
}

const char* NS_CALL StatusString(ns_status status) {
  switch (status) {
    case ns_status_ok: return "ok";
    case ns_status_unsupported: return "unsupported";
    case ns_status_invalid_argument: return "invalid_argument";
    case ns_status_out_of_memory: return "out_of_memory";
    case ns_status_device_lost: return "device_lost";
    case ns_status_runtime_missing: return "runtime_missing";
    case ns_status_runtime_rejected: return "runtime_rejected";
    case ns_status_not_initialized: return "not_initialized";
    case ns_status_needs_reset: return "needs_reset";
    case ns_status_needs_recreate: return "needs_recreate";
    case ns_status_guide_missing: return "guide_missing";
    case ns_status_not_implemented: return "not_implemented";
    default: return "internal_error";
  }
}

const ns_adapter_v1 kAdapter{
    sizeof(ns_adapter_v1), NS_ADAPTER_ABI_VERSION_MAJOR,
    NS_ADAPTER_ABI_VERSION_MINOR, GetInfo, Probe, Initialize, Shutdown,
    CreateSession, DestroySession, ResetHistory, Evaluate, SetSetting,
    GetSetting, QueryVram, StatusString};

}  // namespace

extern "C" NS_EXPORT const ns_adapter_v1* NS_CALL
nsGetAdapterV1(std::uint32_t requested_major) {
  return requested_major == NS_ADAPTER_ABI_VERSION_MAJOR ? &kAdapter : nullptr;
}
