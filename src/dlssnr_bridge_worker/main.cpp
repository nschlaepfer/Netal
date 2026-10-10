/* NeuralStudio crash-isolated D3D12 DLSSNR worker - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <sl.h>

#include "dlssnr_private_contract.h"
#include "dlssnr_worker_protocol.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace protocol = neuralstudio::vulkan_bridge::dlssnr_worker;

namespace {

constexpr DWORD kGpuTimeoutMs = 10000;
std::atomic<std::uint32_t> g_feature_creations{0};

[[noreturn]] void Fail(std::string message) {
  throw std::runtime_error(std::move(message));
}

void RequireHr(HRESULT result, std::string_view operation) {
  if (FAILED(result)) {
    char code[16]{};
    std::snprintf(code, sizeof(code), "%08X",
                  static_cast<unsigned>(result));
    Fail(std::string(operation) + " failed HRESULT=0x" + code);
  }
}

class Handle {
 public:
  Handle() = default;
  explicit Handle(HANDLE value) : value_(value) {}
  ~Handle() { reset(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept : value_(other.release()) {}
  Handle& operator=(Handle&& other) noexcept {
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

struct LibraryCloser {
  void operator()(HMODULE module) const noexcept {
    if (module != nullptr) FreeLibrary(module);
  }
};
using Library =
    std::unique_ptr<std::remove_pointer_t<HMODULE>, LibraryCloser>;

template <typename Function>
Function* LoadFunction(HMODULE module, const char* name) {
  auto* result = reinterpret_cast<Function*>(GetProcAddress(module, name));
  if (result == nullptr) Fail(std::string("Missing Streamline export: ") + name);
  return result;
}

std::wstring Argument(int argc, wchar_t** argv, std::wstring_view name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (name == argv[index]) return argv[index + 1];
  }
  return {};
}

bool ReadPacket(HANDLE pipe, protocol::Packet& packet) {
  DWORD read = 0;
  return ReadFile(pipe, &packet, sizeof(packet), &read, nullptr) &&
         read == sizeof(packet) && protocol::Valid(packet);
}

bool WritePacket(HANDLE pipe, const protocol::Packet& packet) {
  DWORD written = 0;
  return WriteFile(pipe, &packet, sizeof(packet), &written, nullptr) &&
         written == sizeof(packet);
}

void CaptureSlLog(sl::LogType type, const char* message) {
  const std::string_view text = message == nullptr ? "" : message;
  if (text.find("Created DLSS-NR NGX feature") != std::string_view::npos) {
    ++g_feature_creations;
  }
  if (type == sl::LogType::eError ||
      text.find("device removed") != std::string_view::npos) {
    std::cerr << "sl.error=\"" << text << "\"\n";
  }
}

void SetIdentity(sl::float4x4& matrix) {
  matrix.setRow(0, sl::float4(1.0f, 0.0f, 0.0f, 0.0f));
  matrix.setRow(1, sl::float4(0.0f, 1.0f, 0.0f, 0.0f));
  matrix.setRow(2, sl::float4(0.0f, 0.0f, 1.0f, 0.0f));
  matrix.setRow(3, sl::float4(0.0f, 0.0f, 0.0f, 1.0f));
}

struct Slot {
  ComPtr<ID3D12Resource> input;
  ComPtr<ID3D12Resource> output;
  ComPtr<ID3D12Resource> depth;
  ComPtr<ID3D12Resource> motion;
};

class ModelSession {
 public:
  explicit ModelSession(std::filesystem::path runtime)
      : runtime_(std::move(runtime)) {}

  ~ModelSession() { Shutdown(); }
  ModelSession(const ModelSession&) = delete;
  ModelSession& operator=(const ModelSession&) = delete;

  protocol::Status Configure(const protocol::Packet& packet) {
    protocol::Status failure_status = protocol::Status::internal_error;
    try {
      if (packet.width < 64 || packet.height < 64 || packet.width > 8192 ||
          packet.height > 8192 || packet.adapter_luid == 0) {
        return protocol::Status::invalid_packet;
      }
      ReleaseResources();
      if (device_ == nullptr) InitializeDevice(packet.adapter_luid);
      if (adapter_luid_ != packet.adapter_luid) {
        return protocol::Status::adapter_mismatch;
      }
      failure_status = protocol::Status::streamline_init_failed;
      if (interposer_ == nullptr) InitializeStreamline();
      failure_status = protocol::Status::shared_handle_open_failed;
      OpenSharedObjects(packet);
      width_ = packet.width;
      height_ = packet.height;
      ApplyOptions(packet.options);
      configured_ = true;
      requires_reset_ = true;
      last_retired_ = 0;
      return protocol::Status::ok;
    } catch (const std::exception& error) {
      std::cerr << "worker.configure_error=\"" << error.what() << "\"\n";
      ReleaseResources();
      return failure_status;
    }
  }

  protocol::Status Evaluate(const protocol::Packet& packet) {
    if (!configured_ || packet.slot >= protocol::kSlotCount ||
        packet.width != width_ || packet.height != height_ ||
        !protocol::ValidTimeline(packet) || packet.frame_id == 0 ||
        !protocol::ValidConstants(packet.constants) ||
        packet.reset > 1 || (requires_reset_ && packet.reset == 0) ||
        packet.input_ready <= last_retired_) {
      return protocol::Status::invalid_state;
    }
    try {
      const HRESULT health = device_->GetDeviceRemovedReason();
      if (FAILED(health)) return protocol::Status::device_lost;

      RequireHr(allocator_->Reset(), "CommandAllocator::Reset");
      RequireHr(list_->Reset(allocator_.Get(), nullptr),
                "CommandList::Reset");
      Slot& slot = slots_[packet.slot];

      std::array<D3D12_RESOURCE_BARRIER, 4> before{};
      ID3D12Resource* resources[]{slot.input.Get(), slot.output.Get(),
                                  slot.depth.Get(), slot.motion.Get()};
      const D3D12_RESOURCE_STATES states[]{
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
      for (std::size_t index = 0; index < before.size(); ++index) {
        before[index].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        before[index].Transition.pResource = resources[index];
        before[index].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        before[index].Transition.StateAfter = states[index];
        before[index].Transition.Subresource =
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      }
      list_->ResourceBarrier(static_cast<UINT>(before.size()), before.data());

      const sl::ViewportHandle viewport(0u);
      sl::Constants constants{};
      SetIdentity(constants.cameraViewToClip);
      SetIdentity(constants.clipToCameraView);
      SetIdentity(constants.clipToLensClip);
      SetIdentity(constants.clipToPrevClip);
      SetIdentity(constants.prevClipToClip);
      constants.jitterOffset =
          sl::float2(packet.constants.jitter_x, packet.constants.jitter_y);
      constants.mvecScale = sl::float2(packet.constants.motion_scale_x,
                                       packet.constants.motion_scale_y);
      constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
      constants.cameraPos = sl::float3(0.0f, 0.0f, 0.0f);
      constants.cameraUp = sl::float3(0.0f, 1.0f, 0.0f);
      constants.cameraRight = sl::float3(1.0f, 0.0f, 0.0f);
      constants.cameraFwd = sl::float3(0.0f, 0.0f, 1.0f);
      constants.cameraNear = packet.constants.camera_near;
      constants.cameraFar = packet.constants.camera_far;
      constants.cameraFOV = packet.constants.camera_fov_radians;
      constants.cameraAspectRatio = packet.constants.camera_aspect;
      constants.motionVectorsInvalidValue = 0.0f;
      constants.depthInverted = packet.constants.depth_inverted
                                    ? sl::Boolean::eTrue
                                    : sl::Boolean::eFalse;
      constants.cameraMotionIncluded =
          packet.constants.camera_motion_included ? sl::Boolean::eTrue
                                                  : sl::Boolean::eFalse;
      constants.motionVectors3D = packet.constants.motion_vectors_3d
                                      ? sl::Boolean::eTrue
                                      : sl::Boolean::eFalse;
      constants.reset =
          packet.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;

      sl::FrameToken* frame = nullptr;
      const std::uint32_t frame_index =
          static_cast<std::uint32_t>(packet.frame_id & 0xffffffffu);
      if (get_frame_(frame, &frame_index) != sl::Result::eOk || frame == nullptr ||
          set_constants_(constants, *frame, viewport) != sl::Result::eOk) {
        (void)list_->Close();
        return protocol::Status::evaluate_failed;
      }

      sl::Resource input(sl::ResourceType::eTex2d, slot.input.Get(), states[0]);
      sl::Resource output(sl::ResourceType::eTex2d, slot.output.Get(), states[1]);
      sl::Resource depth(sl::ResourceType::eTex2d, slot.depth.Get(), states[2]);
      sl::Resource motion(sl::ResourceType::eTex2d, slot.motion.Get(), states[3]);
      const sl::Extent extent{0, 0, width_, height_};
      sl::ResourceTag input_tag(
          &input, neuralstudio::dlssnr::kBufferTypeUpliftInputColor,
          sl::ResourceLifecycle::eValidUntilEvaluate, &extent);
      sl::ResourceTag output_tag(
          &output, neuralstudio::dlssnr::kBufferTypeUpliftOutputColor,
          sl::ResourceLifecycle::eValidUntilEvaluate, &extent);
      sl::ResourceTag depth_tag(&depth, sl::kBufferTypeDepth,
                                sl::ResourceLifecycle::eValidUntilEvaluate,
                                &extent);
      sl::ResourceTag motion_tag(&motion, sl::kBufferTypeMotionVectors,
                                 sl::ResourceLifecycle::eValidUntilEvaluate,
                                 &extent);
      const sl::BaseStructure* inputs[]{&viewport, &input_tag, &output_tag,
                                        &motion_tag, &depth_tag};
      const sl::Result evaluated = evaluate_(
          neuralstudio::dlssnr::kFeature, *frame, inputs,
          static_cast<std::uint32_t>(std::size(inputs)),
          reinterpret_cast<sl::CommandBuffer*>(list_.Get()));
      if (evaluated != sl::Result::eOk) {
        list_->Close();
        return protocol::Status::evaluate_failed;
      }
      feature_live_ = true;

      D3D12_RESOURCE_BARRIER uav{};
      uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      uav.UAV.pResource = slot.output.Get();
      list_->ResourceBarrier(1, &uav);
      std::array<D3D12_RESOURCE_BARRIER, 4> after = before;
      for (D3D12_RESOURCE_BARRIER& barrier : after) {
        std::swap(barrier.Transition.StateBefore,
                  barrier.Transition.StateAfter);
      }
      list_->ResourceBarrier(static_cast<UINT>(after.size()), after.data());
      RequireHr(list_->Close(), "CommandList::Close");
      RequireHr(queue_->Wait(timeline_.Get(), packet.input_ready),
                "CommandQueue::Wait(input_ready)");
      ID3D12CommandList* lists[]{list_.Get()};
      queue_->ExecuteCommandLists(1, lists);
      RequireHr(queue_->Signal(timeline_.Get(), packet.model_complete),
                "CommandQueue::Signal(model_complete)");
      if (!WaitTimeline(packet.model_complete, kGpuTimeoutMs)) {
        return protocol::Status::gpu_timeout;
      }
      if (FAILED(device_->GetDeviceRemovedReason())) {
        return protocol::Status::device_lost;
      }
      if (CollectDebugErrors() != 0) {
        return protocol::Status::internal_error;
      }
      requires_reset_ = false;
      last_retired_ = packet.retired;
      ++successful_evaluations_;
      return protocol::Status::ok;
    } catch (const std::exception& error) {
      std::cerr << "worker.evaluate_error=\"" << error.what() << "\"\n";
      return FAILED(device_->GetDeviceRemovedReason())
                 ? protocol::Status::device_lost
                 : protocol::Status::internal_error;
    }
  }

  protocol::Status SetOptions(const protocol::Packet& packet) {
    if (!configured_ || !protocol::ValidOptions(packet.options)) {
      return protocol::Status::invalid_state;
    }
    try {
      ApplyOptions(packet.options);
      return protocol::Status::ok;
    } catch (const std::exception& error) {
      std::cerr << "worker.options_error=\"" << error.what() << "\"\n";
      return protocol::Status::internal_error;
    }
  }

  bool InduceDeviceLoss() {
    if (device_ == nullptr) return false;
    ComPtr<ID3D12Device5> device5;
    if (FAILED(device_.As(&device5))) return false;
    device5->RemoveDevice();
    return FAILED(device_->GetDeviceRemovedReason());
  }

  void ReleaseResources() noexcept {
    if (feature_live_ && free_ != nullptr) {
      (void)free_(neuralstudio::dlssnr::kFeature, sl::ViewportHandle(0u));
    }
    feature_live_ = false;
    configured_ = false;
    timeline_.Reset();
    for (Slot& slot : slots_) slot = {};
    width_ = 0;
    height_ = 0;
    requires_reset_ = true;
    last_retired_ = 0;
  }

  [[nodiscard]] std::uint32_t feature_creations() const noexcept {
    return g_feature_creations.load();
  }
  [[nodiscard]] std::uint32_t successful_evaluations() const noexcept {
    return successful_evaluations_;
  }
  [[nodiscard]] bool debug_layer_available() const noexcept {
    return info_queue_ != nullptr;
  }
  [[nodiscard]] std::uint32_t debug_severe_messages() const noexcept {
    return debug_severe_messages_;
  }

 private:
  void InitializeDevice(std::uint64_t luid_value) {
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
      debug->EnableDebugLayer();
    }
    ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
      dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
      dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
      dred->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    }
    LUID requested{};
    static_assert(sizeof(requested) == sizeof(luid_value));
    std::memcpy(&requested, &luid_value, sizeof(requested));
    ComPtr<IDXGIFactory6> factory;
    RequireHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
              "CreateDXGIFactory2");
    ComPtr<IDXGIAdapter1> adapter;
    RequireHr(factory->EnumAdapterByLuid(requested, IID_PPV_ARGS(&adapter)),
              "EnumAdapterByLuid");
    DXGI_ADAPTER_DESC1 description{};
    RequireHr(adapter->GetDesc1(&description), "IDXGIAdapter::GetDesc1");
    if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
      Fail("Refusing a software adapter");
    }
    RequireHr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                IID_PPV_ARGS(&device_)),
              "D3D12CreateDevice");
    (void)device_.As(&info_queue_);
    if (info_queue_ != nullptr) info_queue_->ClearStoredMessages();
    adapter_luid_ = luid_value;
    D3D12_COMMAND_QUEUE_DESC queue{};
    queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    RequireHr(device_->CreateCommandQueue(&queue, IID_PPV_ARGS(&queue_)),
              "CreateCommandQueue");
    RequireHr(device_->CreateCommandAllocator(
                  D3D12_COMMAND_LIST_TYPE_DIRECT,
                  IID_PPV_ARGS(&allocator_)),
              "CreateCommandAllocator");
    RequireHr(device_->CreateCommandList(
                  0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                  IID_PPV_ARGS(&list_)),
              "CreateCommandList");
    RequireHr(list_->Close(), "Initial command-list close");
    completion_event_.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
    if (!completion_event_) Fail("CreateEvent failed");
  }

  void InitializeStreamline() {
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                             LOAD_LIBRARY_SEARCH_USER_DIRS);
    dll_cookie_ = AddDllDirectory(runtime_.c_str());
    if (dll_cookie_ == nullptr) Fail("AddDllDirectory failed");
    interposer_.reset(LoadLibraryExW(
        (runtime_ / L"sl.interposer.dll").c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
            LOAD_LIBRARY_SEARCH_USER_DIRS));
    if (interposer_ == nullptr) Fail("Loading sl.interposer.dll failed");
    init_ = LoadFunction<PFun_slInit>(interposer_.get(), "slInit");
    shutdown_ = LoadFunction<PFun_slShutdown>(interposer_.get(), "slShutdown");
    set_device_ =
        LoadFunction<PFun_slSetD3DDevice>(interposer_.get(), "slSetD3DDevice");
    get_feature_function_ = LoadFunction<PFun_slGetFeatureFunction>(
        interposer_.get(), "slGetFeatureFunction");
    get_frame_ = LoadFunction<PFun_slGetNewFrameToken>(
        interposer_.get(), "slGetNewFrameToken");
    set_constants_ =
        LoadFunction<PFun_slSetConstants>(interposer_.get(), "slSetConstants");
    evaluate_ = LoadFunction<PFun_slEvaluateFeature>(
        interposer_.get(), "slEvaluateFeature");
    free_ = LoadFunction<PFun_slFreeResources>(interposer_.get(),
                                               "slFreeResources");

    const wchar_t* paths[]{runtime_.c_str()};
    const sl::Feature features[]{neuralstudio::dlssnr::kFeature};
    sl::Preferences preferences{};
    preferences.logLevel = sl::LogLevel::eDefault;
    preferences.pathsToPlugins = paths;
    preferences.numPathsToPlugins = 1;
    preferences.pathToLogsAndData = runtime_.c_str();
    preferences.logMessageCallback = CaptureSlLog;
    preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking;
    preferences.featuresToLoad = features;
    preferences.numFeaturesToLoad = 1;
    preferences.applicationId = 0x0E658703u;
    preferences.renderAPI = sl::RenderAPI::eD3D12;
    if (init_(preferences, sl::kSDKVersion) != sl::Result::eOk) {
      Fail("slInit rejected DLSSNR worker");
    }
    streamline_initialized_ = true;
    if (set_device_(device_.Get()) != sl::Result::eOk) {
      Fail("slSetD3DDevice rejected worker device");
    }
  }

  void ApplyOptions(const protocol::Options& requested) {
    if (!protocol::ValidOptions(requested)) {
      Fail("DLSSNR options are outside the audited range");
    }
    void* address = nullptr;
    if (get_feature_function_(neuralstudio::dlssnr::kFeature,
                              "slDLSSNRSetOptions", address) !=
            sl::Result::eOk ||
        address == nullptr) {
      Fail("slDLSSNRSetOptions unavailable");
    }
    auto* set_options =
        reinterpret_cast<neuralstudio::dlssnr::PFunSetOptions*>(address);
    neuralstudio::dlssnr::OptionsV3 options{};
    options.mode = neuralstudio::dlssnr::Mode::kOn;
    options.intensity = requested.intensity;
    options.local_tone_strength = requested.local_tone_strength;
    options.local_structure_strength = requested.local_structure_strength;
    options.global_tone_strength = requested.global_tone_strength;
    options.style = requested.style;
    options.render_preset = requested.render_preset;
    options.use_auto_mask = requested.use_auto_mask
                                ? sl::Boolean::eTrue
                                : sl::Boolean::eFalse;
    options.skin_structure_strength = requested.skin_structure_strength;
    options.performance_mode = requested.performance_mode;
    if (set_options(sl::ViewportHandle(0u), options) != sl::Result::eOk) {
      Fail("slDLSSNRSetOptions rejected worker options");
    }
  }

  template <typename Object>
  ComPtr<Object> Open(HANDLE handle) {
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
      Fail("Null duplicated shared handle");
    }
    ComPtr<Object> object;
    RequireHr(device_->OpenSharedHandle(handle, IID_PPV_ARGS(&object)),
              "OpenSharedHandle");
    return object;
  }

  void OpenSharedObjects(const protocol::Packet& packet) {
    std::array<Handle, protocol::kSharedHandleCount> handles;
    for (std::size_t index = 0; index < handles.size(); ++index) {
      if (packet.shared_handles[index] == 0) {
        Fail("Null duplicated shared handle");
      }
      handles[index].reset(
          reinterpret_cast<HANDLE>(packet.shared_handles[index]));
    }
    timeline_ = Open<ID3D12Fence>(handles[0].get());
    for (std::uint32_t slot = 0; slot < protocol::kSlotCount; ++slot) {
      Slot& resources = slots_[slot];
      resources.input = Open<ID3D12Resource>(
          handles[protocol::ResourceHandleIndex(slot, 0)].get());
      resources.output = Open<ID3D12Resource>(
          handles[protocol::ResourceHandleIndex(slot, 1)].get());
      resources.depth = Open<ID3D12Resource>(
          handles[protocol::ResourceHandleIndex(slot, 2)].get());
      resources.motion = Open<ID3D12Resource>(
          handles[protocol::ResourceHandleIndex(slot, 3)].get());
      ValidateTexture(resources.input.Get(), packet.width, packet.height,
                      DXGI_FORMAT_R16G16B16A16_FLOAT, false);
      ValidateTexture(resources.output.Get(), packet.width, packet.height,
                      DXGI_FORMAT_R16G16B16A16_FLOAT, true);
      ValidateTexture(resources.depth.Get(), packet.width, packet.height,
                      DXGI_FORMAT_R32_FLOAT, false);
      ValidateTexture(resources.motion.Get(), packet.width, packet.height,
                      DXGI_FORMAT_R16G16_FLOAT, false);
    }
  }

  static void ValidateTexture(ID3D12Resource* resource, std::uint32_t width,
                              std::uint32_t height, DXGI_FORMAT format,
                              bool require_uav) {
    if (resource == nullptr) Fail("Shared texture is null");
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Width != width || desc.Height != height ||
        desc.DepthOrArraySize != 1 || desc.MipLevels != 1 ||
        desc.Format != format || desc.SampleDesc.Count != 1 ||
        (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) == 0 ||
        (require_uav &&
         (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)) {
      Fail("Shared texture contract mismatch");
    }
  }

  bool WaitTimeline(std::uint64_t value, DWORD timeout_ms) {
    if (timeline_->GetCompletedValue() >= value) return true;
    RequireHr(timeline_->SetEventOnCompletion(value, completion_event_.get()),
              "Fence::SetEventOnCompletion");
    return WaitForSingleObject(completion_event_.get(), timeout_ms) ==
           WAIT_OBJECT_0;
  }

  std::uint32_t CollectDebugErrors() {
    if (info_queue_ == nullptr) return 0;
    std::uint32_t severe = 0;
    const std::uint64_t count =
        info_queue_->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (std::uint64_t index = 0; index < count; ++index) {
      SIZE_T size = 0;
      if (FAILED(info_queue_->GetMessage(index, nullptr, &size)) ||
          size == 0) {
        continue;
      }
      std::vector<std::uint8_t> bytes(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
      if (FAILED(info_queue_->GetMessage(index, message, &size))) continue;
      if (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
          message->Severity == D3D12_MESSAGE_SEVERITY_ERROR) {
        ++severe;
        std::cerr << "worker.d3d12_error=\""
                  << (message->pDescription == nullptr
                          ? "unknown"
                          : message->pDescription)
                  << "\"\n";
      }
    }
    info_queue_->ClearStoredMessages();
    debug_severe_messages_ += severe;
    return severe;
  }

  void Shutdown() noexcept {
    ReleaseResources();
    if (streamline_initialized_ && shutdown_ != nullptr) (void)shutdown_();
    streamline_initialized_ = false;
    interposer_.reset();
    if (dll_cookie_ != nullptr) RemoveDllDirectory(dll_cookie_);
    dll_cookie_ = nullptr;
  }

  std::filesystem::path runtime_;
  DLL_DIRECTORY_COOKIE dll_cookie_ = nullptr;
  Library interposer_;
  PFun_slInit* init_ = nullptr;
  PFun_slShutdown* shutdown_ = nullptr;
  PFun_slSetD3DDevice* set_device_ = nullptr;
  PFun_slGetFeatureFunction* get_feature_function_ = nullptr;
  PFun_slGetNewFrameToken* get_frame_ = nullptr;
  PFun_slSetConstants* set_constants_ = nullptr;
  PFun_slEvaluateFeature* evaluate_ = nullptr;
  PFun_slFreeResources* free_ = nullptr;
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12InfoQueue> info_queue_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  ComPtr<ID3D12Fence> timeline_;
  Handle completion_event_;
  std::array<Slot, protocol::kSlotCount> slots_{};
  std::uint64_t adapter_luid_ = 0;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  bool streamline_initialized_ = false;
  bool configured_ = false;
  bool feature_live_ = false;
  bool requires_reset_ = true;
  std::uint64_t last_retired_ = 0;
  std::uint32_t successful_evaluations_ = 0;
  std::uint32_t debug_severe_messages_ = 0;
};

protocol::Packet Response(const protocol::Packet& request,
                          protocol::MessageType type,
                          protocol::Status status,
                          const ModelSession& session) {
  protocol::Packet response;
  response.type = type;
  response.status = status;
  response.nonce = request.nonce;
  response.sequence = request.sequence;
  response.process_id = GetCurrentProcessId();
  response.slot = request.slot;
  response.width = request.width;
  response.height = request.height;
  response.reset = request.reset;
  response.feature_creations = session.feature_creations();
  response.successful_evaluations = session.successful_evaluations();
  response.debug_layer_available = session.debug_layer_available() ? 1u : 0u;
  response.debug_severe_messages = session.debug_severe_messages();
  response.adapter_luid = request.adapter_luid;
  response.frame_id = request.frame_id;
  response.input_ready = request.input_ready;
  response.model_complete = request.model_complete;
  response.copyback_complete = request.copyback_complete;
  response.retired = request.retired;
  response.options = request.options;
  response.constants = request.constants;
  return response;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    const std::wstring pipe_name = Argument(argc, argv, L"--pipe");
    const std::wstring runtime_arg = Argument(argc, argv, L"--streamline-dir");
    if (pipe_name.empty() || runtime_arg.empty()) return 2;
    const std::filesystem::path runtime =
        std::filesystem::weakly_canonical(runtime_arg);
    if (!std::filesystem::is_regular_file(runtime / L"sl.interposer.dll")) {
      Fail("Streamline runtime is incomplete");
    }
    if (!WaitNamedPipeW(pipe_name.c_str(), 5000)) {
      const DWORD error = GetLastError();
      Fail(error == ERROR_SEM_TIMEOUT ? "WaitNamedPipe timed out"
                                      : "WaitNamedPipe failed");
    }
    Handle pipe(CreateFileW(pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                            nullptr, OPEN_EXISTING, 0, nullptr));
    if (!pipe) Fail("Opening worker control pipe failed");
    DWORD mode = PIPE_READMODE_MESSAGE;
    if (!SetNamedPipeHandleState(pipe.get(), &mode, nullptr, nullptr)) {
      Fail("SetNamedPipeHandleState failed");
    }

    protocol::Packet packet;
    if (!ReadPacket(pipe.get(), packet) ||
        packet.type != protocol::MessageType::hello || packet.nonce == 0 ||
        packet.process_id == 0) {
      Fail("Invalid supervisor handshake");
    }
    const std::uint64_t nonce = packet.nonce;
    const std::uint32_t supervisor_id = packet.process_id;
    ULONG pipe_server_id = 0;
    if (!GetNamedPipeServerProcessId(pipe.get(), &pipe_server_id) ||
        pipe_server_id != supervisor_id) {
      Fail("Named-pipe server identity mismatch");
    }
    std::uint64_t last_sequence = packet.sequence;
    ModelSession session(runtime);
    auto hello = Response(packet, protocol::MessageType::hello_ack,
                          protocol::Status::ok, session);
    if (!WritePacket(pipe.get(), hello)) Fail("Handshake response failed");

    while (ReadPacket(pipe.get(), packet)) {
      if (packet.nonce != nonce || packet.process_id != supervisor_id ||
          packet.sequence <= last_sequence) {
        auto invalid = Response(packet, packet.type,
                                protocol::Status::invalid_packet, session);
        (void)WritePacket(pipe.get(), invalid);
        break;
      }
      last_sequence = packet.sequence;
      if (packet.type == protocol::MessageType::configure) {
        const auto status = session.Configure(packet);
        auto response = Response(packet, protocol::MessageType::configured,
                                 status, session);
        if (!WritePacket(pipe.get(), response)) break;
      } else if (packet.type == protocol::MessageType::evaluate) {
        const auto status = session.Evaluate(packet);
        auto response = Response(packet, protocol::MessageType::evaluated,
                                 status, session);
        if (!WritePacket(pipe.get(), response)) break;
        if (status == protocol::Status::gpu_timeout ||
            status == protocol::Status::device_lost) {
          TerminateProcess(GetCurrentProcess(),
                           status == protocol::Status::gpu_timeout
                               ? 0xE104u
                               : 0xE105u);
          return 9;
        }
      } else if (packet.type == protocol::MessageType::release) {
        session.ReleaseResources();
        auto response = Response(packet, protocol::MessageType::released,
                                 protocol::Status::ok, session);
        if (!WritePacket(pipe.get(), response)) break;
      } else if (packet.type == protocol::MessageType::set_options) {
        const auto status = session.SetOptions(packet);
        auto response = Response(packet, protocol::MessageType::options_set,
                                 status, session);
        if (!WritePacket(pipe.get(), response)) break;
      } else if (packet.type == protocol::MessageType::inject_crash) {
        FlushFileBuffers(pipe.get());
        TerminateProcess(GetCurrentProcess(), 0xE101u);
        return 5;
      } else if (packet.type == protocol::MessageType::inject_timeout) {
        Sleep(60000);
      } else if (packet.type == protocol::MessageType::inject_device_loss) {
        const bool removed = session.InduceDeviceLoss();
        auto response = Response(
            packet, protocol::MessageType::device_lost,
            removed ? protocol::Status::device_lost
                    : protocol::Status::internal_error,
            session);
        (void)WritePacket(pipe.get(), response);
        // WriteFile has already copied this fixed-size reply into the pipe.
        // Do not call FlushFileBuffers here: on a device-loss path it can wait
        // indefinitely for the supervisor and defeat the restart deadline.
        // Device removal can leave vendor-runtime teardown blocked in a DLL
        // detach callback. This process owns no application state, so force a
        // process boundary exit after the supervisor has received the fault.
        // The parent keeps rendering through its fail-open bypass and starts a
        // fresh worker instead of waiting on poisoned driver/runtime state.
        TerminateProcess(GetCurrentProcess(),
                         removed ? 0xE102u : 0xE103u);
        return 8;
      } else if (packet.type == protocol::MessageType::shutdown) {
        session.ReleaseResources();
        auto response = Response(packet, protocol::MessageType::shutdown_ack,
                                 protocol::Status::ok, session);
        const bool sent = WritePacket(pipe.get(), response);
        FlushFileBuffers(pipe.get());
        return sent ? 0 : 6;
      } else {
        auto response = Response(packet, packet.type,
                                 protocol::Status::invalid_packet, session);
        (void)WritePacket(pipe.get(), response);
        break;
      }
    }
    return 7;
  } catch (const std::exception& error) {
    std::cerr << "worker.fatal=\"" << error.what() << "\"\n";
    return 1;
  }
}
