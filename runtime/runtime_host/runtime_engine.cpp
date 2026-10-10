/* NeuralStudio Runtime v0 diagnostic engine - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "runtime_engine.h"

#include "../broker/pipeline_state.h"
#include "../frame_pipeline/diagnostic_d3d12.h"
#include "../package/package_loader.h"
#include "../../spec/adapter/v1/neuralstudio_adapter.h"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace neuralstudio {
namespace {

class Module {
 public:
  ~Module() { reset(); }
  Module() = default;
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  HMODULE get() const noexcept { return value_; }
  void reset(HMODULE value = nullptr) noexcept {
    if (value_ != nullptr) FreeLibrary(value_);
    value_ = value;
  }

 private:
  HMODULE value_ = nullptr;
};

class FileLock {
 public:
  ~FileLock() { if (value_ != INVALID_HANDLE_VALUE) CloseHandle(value_); }
  FileLock() = default;
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  FileLock(FileLock&& other) noexcept : value_(other.value_) {
    other.value_ = INVALID_HANDLE_VALUE;
  }
  FileLock& operator=(FileLock&& other) noexcept {
    if (this != &other) {
      if (value_ != INVALID_HANDLE_VALUE) CloseHandle(value_);
      value_ = other.value_;
      other.value_ = INVALID_HANDLE_VALUE;
    }
    return *this;
  }
  bool Open(const std::filesystem::path& path) {
    value_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    return value_ != INVALID_HANDLE_VALUE;
  }

 private:
  HANDLE value_ = INVALID_HANDLE_VALUE;
};

struct AdapterScope {
  const ns_adapter_v1* adapter = nullptr;
  ns_session_handle session = nullptr;
  bool initialized = false;
  ~AdapterScope() {
    if (adapter != nullptr && session != nullptr) {
      (void)adapter->destroy_session(session);
    }
    if (adapter != nullptr && initialized) adapter->shutdown();
  }
};

void NS_CALL Log(void*, ns_log_level level, const char* message) {
  if (level < ns_log_warning) return;
  std::fprintf(stderr, "[NeuralStudio adapter] %s\n",
               message == nullptr ? "" : message);
}

void NS_CALL Metric(void*, const char*, double) {}

ns_status NS_CALL RequestReset(void*, ns_reset_reason) {
  return ns_status_ok;
}

ns_status NS_CALL AllocateTexture(void*, const ns_texture_desc*, ns_resource*) {
  return ns_status_not_implemented;
}

void NS_CALL ReleaseTexture(void*, ns_resource*) {}

std::span<const std::uint8_t> Bytes(std::span<const std::uint16_t> values) {
  return {reinterpret_cast<const std::uint8_t*>(values.data()),
          values.size_bytes()};
}

std::string LastErrorText(const char* operation) {
  char buffer[160]{};
  std::snprintf(buffer, sizeof(buffer), "%s failed with Win32 error %lu",
                operation, static_cast<unsigned long>(GetLastError()));
  return buffer;
}

void Fail(DiagnosticResult& result, std::string detail,
          PipelineStateMachine* state = nullptr) {
  result.ok = false;
  result.detail = std::move(detail);
  if (state != nullptr) {
    (void)state->Fail(result.detail);
    result.state = ToString(state->state());
  } else {
    result.state = "Failed";
  }
}

bool SameVerifiedFiles(const VerifiedPackage& left,
                       const VerifiedPackage& right) {
  if (left.verified_files.size() != right.verified_files.size()) return false;
  std::unordered_set<std::wstring> paths;
  for (const auto& file : left.verified_files) {
    paths.insert(std::filesystem::weakly_canonical(file).wstring());
  }
  for (const auto& file : right.verified_files) {
    if (!paths.contains(std::filesystem::weakly_canonical(file).wstring())) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string JsonEscape(const std::string& value) {
  std::string result;
  result.reserve(value.size() + 8);
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (byte < 0x20) {
          char escaped[7]{};
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
          result += escaped;
        } else {
          result.push_back(static_cast<char>(byte));
        }
    }
  }
  return result;
}

std::string DiagnosticResultJson(const DiagnosticResult& result) {
  std::ostringstream out;
  out << "{\"ok\":" << (result.ok ? "true" : "false")
      << ",\"state\":\"" << JsonEscape(result.state) << "\""
      << ",\"detail\":\"" << JsonEscape(result.detail) << "\""
      << ",\"packageId\":\"" << JsonEscape(result.package_id) << "\""
      << ",\"adapterVersion\":\"" << JsonEscape(result.adapter_version) << "\""
      << ",\"outputPath\":\"" << JsonEscape(result.output_path.string()) << "\""
      << ",\"evaluations\":" << result.evaluations
      << ",\"inputHash\":\"" << std::hex << result.input_hash << "\""
      << ",\"outputHash\":\"" << result.output_hash << "\"" << std::dec
      << ",\"changedFromCanaryBytes\":" << result.changed_from_canary_bytes
      << ",\"changedFromInputBytes\":" << result.changed_from_input_bytes
      << ",\"lastInferenceMs\":" << result.last_inference_ms
      << ",\"startHandleCount\":" << result.start_handle_count
      << ",\"endHandleCount\":" << result.end_handle_count
      << ",\"severeD3dMessages\":" << result.severe_d3d_messages << "}";
  return out.str();
}

DiagnosticResult RunDiagnostic(const DiagnosticOptions& options) {
  DiagnosticResult result;
  result.output_path = options.output_path;
  PipelineStateMachine state(Thresholds{3, 1.0, 0.0, true});
  if (!options.inference_armed) {
    Fail(result, "Inference is disarmed. Pass the explicit arm flag to execute a proprietary model.", &state);
    return result;
  }
  if (options.evaluation_frames == 0 || options.evaluation_frames > 10000) {
    Fail(result, "evaluation_frames must be in the range 1..10000", &state);
    return result;
  }

  VerifiedPackage package;
  std::string error;
  if (!VerifyPackage(options.package_root, package, error)) {
    Fail(result, "Package verification failed before load: " + error, &state);
    return result;
  }
  const std::filesystem::path expected_runtime =
      std::filesystem::weakly_canonical(package.root / "runtime");
  if (package.runtime_directory != expected_runtime) {
    Fail(result,
         "Runtime v0 requires verified provider files beside the isolated worker",
         &state);
    return result;
  }
  std::vector<FileLock> package_locks;
  package_locks.reserve(package.verified_files.size());
  for (const auto& file : package.verified_files) {
    FileLock lock;
    if (!lock.Open(file)) {
      Fail(result, "Unable to lock a verified package file against replacement: " +
                       file.string(), &state);
      return result;
    }
    package_locks.push_back(std::move(lock));
  }
  VerifiedPackage locked_package;
  if (!VerifyPackage(options.package_root, locked_package, error) ||
      locked_package.adapter_library != package.adapter_library ||
      locked_package.manifest.summary.package_id !=
          package.manifest.summary.package_id ||
      !SameVerifiedFiles(package, locked_package)) {
    Fail(result, "Package changed while entering its execution lock: " + error,
         &state);
    return result;
  }
  result.package_id = package.manifest.summary.package_id;

  DiagnosticD3D12 source;
  if (!source.Initialize(options.width, options.height, error)) {
    Fail(result, "Diagnostic source initialization failed: " + error, &state);
    return result;
  }
  result.start_handle_count = source.current_process_handle_count();

  if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 |
                                LOAD_LIBRARY_SEARCH_USER_DIRS)) {
    Fail(result, LastErrorText("SetDefaultDllDirectories"), &state);
    return result;
  }
  Module module;
  module.reset(LoadLibraryExW(package.adapter_library.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32));
  if (module.get() == nullptr) {
    Fail(result, LastErrorText("LoadLibraryExW(adapter)"), &state);
    return result;
  }
  const auto entry = reinterpret_cast<ns_get_adapter_v1_fn>(
      GetProcAddress(module.get(), "nsGetAdapterV1"));
  if (entry == nullptr) {
    Fail(result, "Verified adapter does not export nsGetAdapterV1", &state);
    return result;
  }
  AdapterScope scope;
  scope.adapter = entry(NS_ADAPTER_ABI_VERSION_MAJOR);
  if (scope.adapter == nullptr ||
      scope.adapter->struct_size < sizeof(ns_adapter_v1) ||
      scope.adapter->abi_version_major != NS_ADAPTER_ABI_VERSION_MAJOR) {
    Fail(result, "Adapter rejected Runtime ABI v1", &state);
    return result;
  }
  ns_adapter_info info{};
  info.struct_size = sizeof(info);
  if (scope.adapter->get_info(&info) != ns_status_ok ||
      info.package_id == nullptr || info.adapter_version == nullptr ||
      result.package_id != info.package_id) {
    Fail(result, "Adapter identity does not match the verified manifest", &state);
    return result;
  }
  result.adapter_version = info.adapter_version;

  ns_device_desc device = source.device_desc();
  ns_capability_report capability{};
  capability.struct_size = sizeof(capability);
  const ns_status probe_status = scope.adapter->probe(&device, &capability);
  if (probe_status != ns_status_ok || capability.supported == 0) {
    Fail(result, std::string("Adapter capability probe rejected this device: ") +
                     capability.detail, &state);
    return result;
  }

  ns_host_v1 host{};
  host.struct_size = sizeof(host);
  host.log = Log;
  host.report_metric = Metric;
  host.request_reset = RequestReset;
  host.allocate_texture = AllocateTexture;
  host.release_texture = ReleaseTexture;
  const std::string package_root = package.root.string();
  const std::filesystem::path cache_path =
      package.root.parent_path().parent_path() / "cache" / result.package_id;
  std::error_code filesystem_error;
  std::filesystem::create_directories(cache_path, filesystem_error);
  if (filesystem_error) {
    Fail(result, "Creating the adapter cache failed: " + filesystem_error.message(), &state);
    return result;
  }
  const std::string cache_root = cache_path.string();
  ns_init_desc init{};
  init.struct_size = sizeof(init);
  init.device = device;
  init.package_root = package_root.c_str();
  init.cache_root = cache_root.c_str();
  init.sandboxed = 0;
  ns_status status = scope.adapter->initialize(&host, &init);
  if (status != ns_status_ok) {
    Fail(result, std::string("Adapter initialization failed: ") +
                     scope.adapter->status_string(status), &state);
    return result;
  }
  scope.initialized = true;

  ns_session_desc session_desc = source.session_desc(package.root);
  session_desc.package_root = package_root.c_str();
  status = scope.adapter->create_session(&session_desc, &scope.session);
  if (status != ns_status_ok || scope.session == nullptr) {
    Fail(result, std::string("Session creation failed: ") +
                     scope.adapter->status_string(status), &state);
    return result;
  }

  ActivationGate gate(4);
  for (std::uint64_t warmup = 0; warmup < 4; ++warmup) {
    if (!source.WriteFrame(warmup, error)) {
      Fail(result, "Warmup frame upload failed: " + error, &state);
      return result;
    }
    gate.ObserveResolution(options.width, options.height);
  }
  if (!gate.IsReady()) {
    Fail(result, "Stable-frame activation gate did not arm", &state);
    return result;
  }

  std::vector<std::uint16_t> output;
  for (std::uint64_t index = 0; index < options.evaluation_frames; ++index) {
    const std::uint64_t frame_index = index + 4;
    if (!source.WriteFrame(frame_index, error)) {
      Fail(result, "Frame upload failed: " + error, &state);
      return result;
    }
    const auto inputs = source.inputs();
    ns_frame frame{};
    frame.struct_size = sizeof(frame);
    frame.frame_index = frame_index;
    frame.timestamp_seconds = static_cast<double>(frame_index) / 60.0;
    frame.reset = index == 0 ? ns_reset_session_start : ns_reset_none;
    frame.inputs = inputs.data();
    frame.input_count = static_cast<std::uint32_t>(inputs.size());
    frame.output = source.output();
    frame.camera.struct_size = sizeof(ns_camera);
    frame.camera.valid = 1;
    frame.camera.near_plane = 0.1f;
    frame.camera.far_plane = 1000.0f;
    frame.camera.vertical_fov_radians = 1.04719755f;
    frame.exposure = 1.0f;

    ns_evaluate_result evaluated{};
    evaluated.struct_size = sizeof(evaluated);
    status = scope.adapter->evaluate(scope.session, &frame, &evaluated);
    if (status != ns_status_ok) {
      Observation observation{};
      observation.now_seconds = frame.timestamp_seconds;
      observation.device_lost = status == ns_status_device_lost;
      observation.adapter_host_lost = status == ns_status_internal_error;
      (void)state.Observe(observation);
      /* The source remains valid when its model worker fails. Preserve a real
         pass-through frame and the Bypassed state instead of escalating an
         isolated model fault into a failed media pipeline. */
      const auto input = source.input_pixels();
      std::string output_error;
      if (!source.SaveBmp(options.output_path, input, output_error)) {
        (void)state.Fail("Model and pass-through output both failed");
        result.state = ToString(state.state());
        result.detail = "Frame evaluation failed and pass-through output could not be written: " +
                        output_error;
        return result;
      }
      result.input_hash = Fnv1aBytes(Bytes(input));
      result.output_hash = result.input_hash;
      result.state = ToString(state.state());
      result.detail = std::string("Model worker failed; source was safely passed through: ") +
                      scope.adapter->status_string(status) + " (" +
                      evaluated.detail + ")";
      return result;
    }
    if (!source.ReadOutput(output, error)) {
      Fail(result, "Output readback failed: " + error, &state);
      return result;
    }
    const auto output_bytes = Bytes(output);
    const auto input_bytes = Bytes(source.input_pixels());
    const auto canary_bytes = Bytes(source.output_canary());
    result.changed_from_canary_bytes = ChangedBytes(output_bytes, canary_bytes);
    result.changed_from_input_bytes = ChangedBytes(output_bytes, input_bytes);
    const bool fresh = evaluated.evaluated != 0 &&
                       evaluated.output_fresh != 0 &&
                       result.changed_from_canary_bytes != 0;
    Observation observation{};
    observation.now_seconds = frame.timestamp_seconds;
    observation.evaluated = evaluated.evaluated != 0;
    observation.output_fresh = fresh;
    observation.used_fallback_path = evaluated.used_fallback_path != 0;
    observation.inference_ms = evaluated.inference_ms;
    (void)state.Observe(observation);
    if (!fresh) {
      Fail(result, "Adapter returned without demonstrably fresh output", &state);
      return result;
    }
    ++result.evaluations;
    result.last_inference_ms = evaluated.inference_ms;
    result.input_hash = Fnv1aBytes(input_bytes);
    result.output_hash = Fnv1aBytes(output_bytes);
    if (index == 0 && options.test_failure != DiagnosticTestFailure::None) {
      ns_value injected{};
      injected.type = ns_value_int;
      injected.int_value =
          options.test_failure == DiagnosticTestFailure::Crash ? 1 :
          options.test_failure == DiagnosticTestFailure::Timeout ? 2 : 3;
      const ns_status injection_status = scope.adapter->set_setting(
          scope.session, "NeuralStudioTestFailure", &injected);
      if (injection_status != ns_status_ok) {
        Fail(result, "The test-only fault injection gate was refused", &state);
        return result;
      }
    }
  }

  if (result.changed_from_input_bytes == 0) {
    Fail(result, "Output was fresh but byte-identical to the synthetic input", &state);
    return result;
  }
  if (!source.SaveBmp(options.output_path, output, error)) {
    Fail(result, "Writing diagnostic output failed: " + error, &state);
    return result;
  }
  result.severe_d3d_messages = source.severe_debug_messages();
  if (result.severe_d3d_messages != 0) {
    Fail(result, "D3D12 validation reported a severe message", &state);
    return result;
  }
  /* Teardown is part of the acceptance contract. Sample the handle count only
     after the adapter session, isolated worker, and adapter module are gone;
     otherwise live worker IPC handles look like a leak. */
  (void)scope.adapter->destroy_session(scope.session);
  scope.session = nullptr;
  scope.adapter->shutdown();
  scope.initialized = false;
  scope.adapter = nullptr;
  module.reset();
  result.end_handle_count = source.current_process_handle_count();
  result.state = ToString(state.state());
  result.ok = state.state() == PipelineState::Active ||
              state.state() == PipelineState::Degraded;
  result.detail = result.ok
      ? "Verified package produced a fresh, changed, file-backed output"
      : "Pipeline did not reach an evidence-backed running state";
  return result;
}

}  // namespace neuralstudio
