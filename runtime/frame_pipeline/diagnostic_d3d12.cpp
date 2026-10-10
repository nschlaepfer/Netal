/* NeuralStudio guide-complete diagnostic frame source - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "diagnostic_d3d12.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

using Microsoft::WRL::ComPtr;

namespace neuralstudio {
namespace {

constexpr DWORD kGpuTimeoutMs = 15000;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

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
  HANDLE get() const noexcept { return value_; }
  explicit operator bool() const noexcept {
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

std::string HrText(const char* operation, HRESULT result) {
  char buffer[160]{};
  std::snprintf(buffer, sizeof(buffer), "%s failed HRESULT=0x%08X",
                operation, static_cast<unsigned>(result));
  return buffer;
}

std::uint16_t FloatToHalf(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa = (mantissa | 0x800000u) >> (1 - exponent);
    return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10) |
      ((mantissa + 0x1000u) >> 13));
}

float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000u)) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      const std::uint32_t adjusted = static_cast<std::uint32_t>(127 - 15 - shift);
      bits = sign | (adjusted << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    exponent += 127 - 15;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, std::uint64_t size,
                                    D3D12_HEAP_TYPE heap_type,
                                    D3D12_RESOURCE_STATES state,
                                    std::string& error) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = heap_type;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> result;
  const HRESULT hr = device->CreateCommittedResource(
      &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
      IID_PPV_ARGS(&result));
  if (FAILED(hr)) error = HrText("CreateCommittedResource(buffer)", hr);
  return result;
}

struct Texture {
  ComPtr<ID3D12Resource> resource;
  ComPtr<ID3D12Resource> upload;
  ComPtr<ID3D12Resource> readback;
  Handle shared;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  std::uint64_t footprint_size = 0;
  std::uint32_t bytes_per_pixel = 0;
};

}  // namespace

class DiagnosticD3D12::Impl {
 public:
  bool Initialize(std::uint32_t width, std::uint32_t height,
                  std::string& error) {
    if (width < 64 || height < 64 || width > 8192 || height > 8192) {
      error = "Diagnostic extent is outside 64..8192";
      return false;
    }
    width_ = width;
    height_ = height;
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
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory_));
    if (FAILED(hr)) { error = HrText("CreateDXGIFactory2", hr); return false; }
    for (UINT index = 0;; ++index) {
      ComPtr<IDXGIAdapter1> candidate;
      hr = factory_->EnumAdapterByGpuPreference(
          index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
          IID_PPV_ARGS(&candidate));
      if (hr == DXGI_ERROR_NOT_FOUND) break;
      if (FAILED(hr)) { error = HrText("EnumAdapterByGpuPreference", hr); return false; }
      DXGI_ADAPTER_DESC1 desc{};
      if (FAILED(candidate->GetDesc1(&desc)) ||
          (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
          desc.VendorId != 0x10DEu) continue;
      ComPtr<ID3D12Device> device;
      if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0,
                                      IID_PPV_ARGS(&device)))) {
        if (FAILED(candidate.As(&adapter_))) continue;
        device_ = std::move(device);
        break;
      }
    }
    if (!device_) { error = "No NVIDIA D3D12 device was found"; return false; }
    if (FAILED(adapter_->GetDesc3(&description_))) {
      error = "Reading adapter description failed";
      return false;
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_))) ||
        FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&allocator_))) ||
        FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           allocator_.Get(), nullptr,
                                           IID_PPV_ARGS(&list_))) ||
        FAILED(list_->Close()) ||
        FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                    IID_PPV_ARGS(&fence_)))) {
      error = "Creating the diagnostic D3D12 queue failed";
      return false;
    }
    event_.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
    if (!event_) { error = "CreateEvent failed"; return false; }
    if (!CreateTexture(input_, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, false, error) ||
        !CreateTexture(output_, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, true, error) ||
        !CreateTexture(depth_, DXGI_FORMAT_R32_FLOAT, 4, false, error) ||
        !CreateTexture(motion_, DXGI_FORMAT_R16G16_FLOAT, 4, false, error)) {
      return false;
    }
    return true;
  }

  bool WriteFrame(std::uint64_t frame_index, std::string& error) {
    const std::size_t pixels = static_cast<std::size_t>(width_) * height_;
    input_pixels_.assign(pixels * 4, 0);
    output_canary_.assign(pixels * 4,
                          static_cast<std::uint16_t>(0x3400u + (frame_index & 0x1ffu)));
    std::vector<float> depth_values(pixels);
    std::vector<std::uint16_t> motion_values(pixels * 2);
    for (std::uint32_t y = 0; y < height_; ++y) {
      for (std::uint32_t x = 0; x < width_; ++x) {
        const std::size_t p = static_cast<std::size_t>(y) * width_ + x;
        const float fx = static_cast<float>(x) / static_cast<float>(width_ - 1);
        const float fy = static_cast<float>(y) / static_cast<float>(height_ - 1);
        input_pixels_[p * 4 + 0] = FloatToHalf(std::clamp(fx + static_cast<float>(frame_index % 17) * 0.002f, 0.0f, 1.0f));
        input_pixels_[p * 4 + 1] = FloatToHalf(fy);
        input_pixels_[p * 4 + 2] = FloatToHalf(0.12f + 0.72f * (1.0f - fx) * fy);
        input_pixels_[p * 4 + 3] = FloatToHalf(1.0f);
        depth_values[p] = 1.0f - (0.1f + 0.8f * fy);
        motion_values[p * 2 + 0] =
            FloatToHalf(frame_index == 0 ? 0.0f : 0.25f);
        motion_values[p * 2 + 1] = FloatToHalf(0.0f);
      }
    }
    const auto bytes = [](const auto& values) {
      return std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(values.data()),
          values.size() * sizeof(values[0]));
    };
    if (!Upload(input_, bytes(input_pixels_), error) ||
        !Upload(output_, bytes(output_canary_), error) ||
        !Upload(depth_, bytes(depth_values), error) ||
        !Upload(motion_, bytes(motion_values), error)) return false;
    if (!Begin(error)) return false;
    CopyUpload(input_);
    CopyUpload(output_);
    CopyUpload(depth_);
    CopyUpload(motion_);
    return ExecuteAndWait(error);
  }

  bool ReadOutput(std::vector<std::uint16_t>& result, std::string& error) {
    if (!Begin(error)) return false;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = output_.resource.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list_->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = output_.resource.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = output_.readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = output_.footprint;
    list_->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list_->ResourceBarrier(1, &barrier);
    if (!ExecuteAndWait(error)) return false;
    const D3D12_RANGE read{0, static_cast<SIZE_T>(output_.footprint_size)};
    const std::uint8_t* mapped = nullptr;
    if (FAILED(output_.readback->Map(
            0, &read,
            reinterpret_cast<void**>(const_cast<std::uint8_t**>(&mapped))))) {
      error = "Mapping output readback failed";
      return false;
    }
    result.assign(static_cast<std::size_t>(width_) * height_ * 4, 0);
    for (std::uint32_t row = 0; row < height_; ++row) {
      std::memcpy(result.data() + static_cast<std::size_t>(row) * width_ * 4,
                  mapped + output_.footprint.Offset +
                      static_cast<std::size_t>(row) *
                          output_.footprint.Footprint.RowPitch,
                  static_cast<std::size_t>(width_) * 8);
    }
    const D3D12_RANGE no_write{0, 0};
    output_.readback->Unmap(0, &no_write);
    return true;
  }

  bool SaveBmp(const std::filesystem::path& path,
               std::span<const std::uint16_t> rgba16f,
               std::string& error) const {
    const std::size_t expected = static_cast<std::size_t>(width_) * height_ * 4;
    if (rgba16f.size() != expected) {
      error = "BMP source size does not match the diagnostic extent";
      return false;
    }
    const std::uint32_t row_bytes = width_ * 4;
    const std::uint32_t pixels_bytes = row_bytes * height_;
#pragma pack(push, 1)
    struct FileHeader { std::uint16_t type; std::uint32_t size; std::uint16_t r1; std::uint16_t r2; std::uint32_t offset; };
    struct InfoHeader { std::uint32_t size; std::int32_t width; std::int32_t height; std::uint16_t planes; std::uint16_t bits; std::uint32_t compression; std::uint32_t image_size; std::int32_t xppm; std::int32_t yppm; std::uint32_t used; std::uint32_t important; };
#pragma pack(pop)
    FileHeader file{0x4D42u, static_cast<std::uint32_t>(sizeof(FileHeader) + sizeof(InfoHeader) + pixels_bytes), 0, 0, static_cast<std::uint32_t>(sizeof(FileHeader) + sizeof(InfoHeader))};
    InfoHeader info{sizeof(InfoHeader), static_cast<std::int32_t>(width_), -static_cast<std::int32_t>(height_), 1, 32, 0, pixels_bytes, 2835, 2835, 0, 0};
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) { error = "Opening BMP output failed"; return false; }
    stream.write(reinterpret_cast<const char*>(&file), sizeof(file));
    stream.write(reinterpret_cast<const char*>(&info), sizeof(info));
    std::vector<std::uint8_t> row(row_bytes);
    for (std::uint32_t y = 0; y < height_; ++y) {
      for (std::uint32_t x = 0; x < width_; ++x) {
        const std::size_t source = (static_cast<std::size_t>(y) * width_ + x) * 4;
        const auto channel = [&](std::size_t at) {
          const float linear = std::clamp(HalfToFloat(rgba16f[at]), 0.0f, 1.0f);
          return static_cast<std::uint8_t>(std::lround(std::pow(linear, 1.0f / 2.2f) * 255.0f));
        };
        const std::size_t target = static_cast<std::size_t>(x) * 4;
        row[target + 0] = channel(source + 2);
        row[target + 1] = channel(source + 1);
        row[target + 2] = channel(source + 0);
        row[target + 3] = 255;
      }
      stream.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
    if (!stream) { error = "Writing BMP output failed"; return false; }
    return true;
  }

  ns_device_desc DeviceDesc() const {
    ns_device_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.api = ns_api_d3d12;
    desc.device = device_.Get();
    desc.queue = queue_.Get();
    desc.vendor_id = description_.VendorId;
    desc.device_id = description_.DeviceId;
    desc.dedicated_vram_bytes = description_.DedicatedVideoMemory;
    desc.driver_version = "unknown";
    return desc;
  }

  ns_session_desc SessionDesc(const std::filesystem::path& package_root) const {
    (void)package_root;
    ns_session_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.input_extent = {width_, height_};
    desc.output_extent = {width_, height_};
    desc.color_format = ns_format_r16g16b16a16_float;
    desc.output_format = ns_format_r16g16b16a16_float;
    desc.conventions.struct_size = sizeof(ns_conventions);
    desc.conventions.depth_direction = ns_depth_inverse;
    desc.conventions.motion_vector_space = ns_mv_screen;
    desc.conventions.motion_vector_scale = {1.0f, 1.0f};
    desc.conventions.color_space = ns_color_scrgb;
    desc.conventions.paper_white_nits = 200.0f;
    desc.enable_hdr = 1;
    return desc;
  }

  std::vector<ns_resource> Inputs() const {
    std::vector<ns_resource> result(3);
    FillResource(result[0], ns_guide_color, ns_format_r16g16b16a16_float, input_);
    FillResource(result[1], ns_guide_depth, ns_format_r32_float, depth_);
    FillResource(result[2], ns_guide_motion_vectors, ns_format_r16g16_float, motion_);
    return result;
  }
  ns_resource Output() const {
    ns_resource result{};
    FillResource(result, ns_guide_color, ns_format_r16g16b16a16_float, output_);
    return result;
  }

  std::uint32_t SevereDebugMessages() const {
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(device_.As(&queue))) return std::numeric_limits<std::uint32_t>::max();
    std::uint32_t severe = 0;
    const std::uint64_t count = queue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (std::uint64_t index = 0; index < count; ++index) {
      SIZE_T size = 0;
      if (FAILED(queue->GetMessage(index, nullptr, &size)) || size == 0) continue;
      std::vector<std::uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (SUCCEEDED(queue->GetMessage(index, message, &size)) &&
          (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
           message->Severity == D3D12_MESSAGE_SEVERITY_ERROR)) ++severe;
    }
    return severe;
  }

  std::uint64_t HandleCount() const {
    DWORD count = 0;
    return GetProcessHandleCount(GetCurrentProcess(), &count) ? count : 0;
  }

  ID3D12Device* device() const noexcept { return device_.Get(); }
  std::uint32_t width() const noexcept { return width_; }
  std::uint32_t height() const noexcept { return height_; }
  std::span<const std::uint16_t> input_pixels() const noexcept { return input_pixels_; }
  std::span<const std::uint16_t> output_canary() const noexcept { return output_canary_; }

 private:
  bool CreateTexture(Texture& texture, DXGI_FORMAT format,
                     std::uint32_t bytes_per_pixel, bool output,
                     std::string& error) {
    texture.bytes_per_pixel = bytes_per_pixel;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width_;
    desc.Height = height_;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                 (output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                         : D3D12_RESOURCE_FLAG_NONE);
    HRESULT hr = device_->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&texture.resource));
    if (FAILED(hr)) { error = HrText("CreateCommittedResource(texture)", hr); return false; }
    HANDLE shared = nullptr;
    hr = device_->CreateSharedHandle(texture.resource.Get(), nullptr,
                                     GENERIC_ALL, nullptr, &shared);
    if (FAILED(hr)) { error = HrText("CreateSharedHandle(texture)", hr); return false; }
    texture.shared.reset(shared);
    UINT rows = 0;
    UINT64 row_size = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &texture.footprint,
                                   &rows, &row_size, &texture.footprint_size);
    if (rows != height_ || row_size != static_cast<UINT64>(width_) * bytes_per_pixel) {
      error = "Unexpected D3D12 texture footprint";
      return false;
    }
    texture.upload = CreateBuffer(device_.Get(), texture.footprint_size,
                                  D3D12_HEAP_TYPE_UPLOAD,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, error);
    if (!texture.upload) return false;
    if (output) {
      texture.readback = CreateBuffer(device_.Get(), texture.footprint_size,
                                      D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_STATE_COPY_DEST, error);
      if (!texture.readback) return false;
    }
    return true;
  }

  bool Upload(Texture& texture, std::span<const std::uint8_t> bytes,
              std::string& error) {
    const std::size_t expected = static_cast<std::size_t>(width_) * height_ *
                                 texture.bytes_per_pixel;
    if (bytes.size() != expected) { error = "Upload size mismatch"; return false; }
    void* mapped = nullptr;
    const D3D12_RANGE no_read{0, 0};
    const HRESULT hr = texture.upload->Map(0, &no_read, &mapped);
    if (FAILED(hr)) { error = HrText("Upload::Map", hr); return false; }
    for (std::uint32_t row = 0; row < height_; ++row) {
      std::memcpy(static_cast<std::uint8_t*>(mapped) + texture.footprint.Offset +
                      static_cast<std::size_t>(row) *
                          texture.footprint.Footprint.RowPitch,
                  bytes.data() + static_cast<std::size_t>(row) * width_ *
                                     texture.bytes_per_pixel,
                  static_cast<std::size_t>(width_) * texture.bytes_per_pixel);
    }
    texture.upload->Unmap(0, nullptr);
    return true;
  }

  bool Begin(std::string& error) {
    HRESULT hr = allocator_->Reset();
    if (SUCCEEDED(hr)) hr = list_->Reset(allocator_.Get(), nullptr);
    if (FAILED(hr)) { error = HrText("Reset command list", hr); return false; }
    return true;
  }

  void CopyUpload(Texture& texture) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture.resource.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list_->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = texture.resource.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = texture.upload.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = texture.footprint;
    list_->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list_->ResourceBarrier(1, &barrier);
  }

  bool ExecuteAndWait(std::string& error) {
    HRESULT hr = list_->Close();
    if (FAILED(hr)) { error = HrText("CommandList::Close", hr); return false; }
    ID3D12CommandList* lists[]{list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    const std::uint64_t value = ++fence_value_;
    hr = queue_->Signal(fence_.Get(), value);
    if (SUCCEEDED(hr)) hr = fence_->SetEventOnCompletion(value, event_.get());
    if (FAILED(hr)) { error = HrText("Queue completion signal", hr); return false; }
    if (WaitForSingleObject(event_.get(), kGpuTimeoutMs) != WAIT_OBJECT_0) {
      error = "Diagnostic D3D12 queue timed out";
      return false;
    }
    hr = device_->GetDeviceRemovedReason();
    if (FAILED(hr)) { error = HrText("D3D12 device removed", hr); return false; }
    return true;
  }

  void FillResource(ns_resource& result, ns_guide_kind kind,
                    ns_format format, const Texture& texture) const {
    result = {};
    result.struct_size = sizeof(result);
    result.kind = kind;
    result.format = format;
    result.api = ns_api_d3d12;
    result.extent = {width_, height_};
    result.shared_handle = reinterpret_cast<std::uint64_t>(texture.shared.get());
    result.resource_state = D3D12_RESOURCE_STATE_COMMON;
  }

  ComPtr<IDXGIFactory6> factory_;
  ComPtr<IDXGIAdapter4> adapter_;
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  ComPtr<ID3D12Fence> fence_;
  Handle event_;
  DXGI_ADAPTER_DESC3 description_{};
  Texture input_;
  Texture output_;
  Texture depth_;
  Texture motion_;
  std::vector<std::uint16_t> input_pixels_;
  std::vector<std::uint16_t> output_canary_;
  std::uint64_t fence_value_ = 0;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
};

DiagnosticD3D12::DiagnosticD3D12() : impl_(std::make_unique<Impl>()) {}
DiagnosticD3D12::~DiagnosticD3D12() = default;
DiagnosticD3D12::DiagnosticD3D12(DiagnosticD3D12&&) noexcept = default;
DiagnosticD3D12& DiagnosticD3D12::operator=(DiagnosticD3D12&&) noexcept = default;
bool DiagnosticD3D12::Initialize(std::uint32_t width, std::uint32_t height, std::string& error) { return impl_->Initialize(width, height, error); }
bool DiagnosticD3D12::WriteFrame(std::uint64_t frame_index, std::string& error) { return impl_->WriteFrame(frame_index, error); }
bool DiagnosticD3D12::ReadOutput(std::vector<std::uint16_t>& output, std::string& error) { return impl_->ReadOutput(output, error); }
bool DiagnosticD3D12::SaveBmp(const std::filesystem::path& path, std::span<const std::uint16_t> pixels, std::string& error) const { return impl_->SaveBmp(path, pixels, error); }
ID3D12Device* DiagnosticD3D12::device() const noexcept { return impl_->device(); }
ns_device_desc DiagnosticD3D12::device_desc() const { return impl_->DeviceDesc(); }
ns_session_desc DiagnosticD3D12::session_desc(const std::filesystem::path& root) const { return impl_->SessionDesc(root); }
std::vector<ns_resource> DiagnosticD3D12::inputs() const { return impl_->Inputs(); }
ns_resource DiagnosticD3D12::output() const { return impl_->Output(); }
std::span<const std::uint16_t> DiagnosticD3D12::input_pixels() const noexcept { return impl_->input_pixels(); }
std::span<const std::uint16_t> DiagnosticD3D12::output_canary() const noexcept { return impl_->output_canary(); }
std::uint32_t DiagnosticD3D12::width() const noexcept { return impl_->width(); }
std::uint32_t DiagnosticD3D12::height() const noexcept { return impl_->height(); }
std::uint32_t DiagnosticD3D12::severe_debug_messages() const { return impl_->SevereDebugMessages(); }
std::uint64_t DiagnosticD3D12::current_process_handle_count() const { return impl_->HandleCount(); }

std::uint64_t Fnv1aBytes(std::span<const std::uint8_t> bytes) noexcept {
  std::uint64_t hash = kFnvOffset;
  for (std::uint8_t byte : bytes) { hash ^= byte; hash *= kFnvPrime; }
  return hash;
}

std::size_t ChangedBytes(std::span<const std::uint8_t> left,
                         std::span<const std::uint8_t> right) noexcept {
  const std::size_t count = std::min(left.size(), right.size());
  std::size_t changed = left.size() > right.size() ? left.size() - right.size()
                                                  : right.size() - left.size();
  for (std::size_t index = 0; index < count; ++index) {
    if (left[index] != right[index]) ++changed;
  }
  return changed;
}

}  // namespace neuralstudio
