#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 180;
constexpr std::uint32_t kInputWidth = 160;
constexpr std::uint32_t kInputHeight = 96;
constexpr std::uint32_t kChannels = 16;
constexpr std::size_t kInputBytes =
    std::size_t{kInputWidth} * kInputHeight * kChannels * 2u;
constexpr std::size_t kOutputVectorBytes =
    std::size_t{kWidth} * kHeight * 4u;
constexpr std::size_t kOutputScalarBytes =
    std::size_t{kWidth} * kHeight * 2u;
constexpr std::size_t kDepthBytes = std::size_t{kWidth} * kHeight;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kModuleSha[] =
    "d45333f43f343d1f38bd304fe76214dc7064f2a7d24ce1ea06bc791867783264";
constexpr char kRawParameterSha[] =
    "b85b7b8f373002d0518a784cfa9e9d8511979be6dd2641106b5c5acfb4a2a01d";
constexpr char kGenerator[] =
    "splitmix64-fg-mid-l52-half16-depth-v1";

using NvStatus = std::int32_t;
using NvObjectHandle = void*;
constexpr NvStatus kNvapiOk = 0;
constexpr std::uint32_t kIdInitialize = 0x0150e828;
constexpr std::uint32_t kIdGetMergedTextureSampler = 0x329fe6e0;
constexpr std::uint32_t kIdGetIndependentDescriptor = 0x0ddac234;
constexpr std::uint32_t kIdCreateCuModule = 0xad1a677d;
constexpr std::uint32_t kIdCreateCuFunction = 0xe2436e22;
constexpr std::uint32_t kIdLaunchKernelChain = 0x24973538;
constexpr std::uint32_t kIdDestroyCuModule = 0x41c65285;
constexpr std::uint32_t kIdDestroyCuFunction = 0xdf295ea6;

struct CpuDescriptorHandle { std::uintptr_t ptr; };
struct MergedTextureSamplerParams {
  std::size_t struct_size_in;
  std::size_t struct_size_out;
  void* device;
  CpuDescriptorHandle texture_descriptor;
  CpuDescriptorHandle sampler_descriptor;
  std::uint64_t texture_handle;
};
enum class IndependentDescriptorType : std::int32_t {
  kSurface = 0,
  kTexture = 1,
  kSampler = 2,
};
struct IndependentDescriptorParams {
  std::size_t struct_size_in;
  std::size_t struct_size_out;
  void* device;
  IndependentDescriptorType type;
  CpuDescriptorHandle descriptor;
  std::uint64_t handle;
};
struct Dim3 { std::uint32_t x, y, z; };
struct KernelLaunchParams {
  NvObjectHandle function;
  Dim3 grid;
  Dim3 block;
  std::uint32_t dynamic_shared_bytes;
  const void* params;
  std::uint32_t param_size;
};
static_assert(sizeof(MergedTextureSamplerParams) == 48);
static_assert(sizeof(IndependentDescriptorParams) == 48);
static_assert(sizeof(KernelLaunchParams) == 56);

using QueryInterface = void*(__cdecl*)(std::uint32_t);
using Initialize = NvStatus(__cdecl*)();
using GetMergedTextureSampler = NvStatus(__cdecl*)(MergedTextureSamplerParams*);
using GetIndependentDescriptor = NvStatus(__cdecl*)(IndependentDescriptorParams*);
using CreateCuModule = NvStatus(__cdecl*)(void*, const void*, std::uint32_t,
                                          NvObjectHandle*);
using CreateCuFunction = NvStatus(__cdecl*)(void*, NvObjectHandle, const char*,
                                            NvObjectHandle*);
using LaunchKernelChain = NvStatus(__cdecl*)(void*, const KernelLaunchParams*,
                                             std::uint32_t);
using DestroyCuModule = NvStatus(__cdecl*)(void*, NvObjectHandle);
using DestroyCuFunction = NvStatus(__cdecl*)(void*, NvObjectHandle);

struct LibraryCloser {
  void operator()(HMODULE library) const noexcept {
    if (library) FreeLibrary(library);
  }
};
using UniqueLibrary =
    std::unique_ptr<std::remove_pointer_t<HMODULE>, LibraryCloser>;
struct HandleCloser {
  void operator()(HANDLE handle) const noexcept {
    if (handle && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
  }
};
using UniqueHandle =
    std::unique_ptr<std::remove_pointer_t<HANDLE>, HandleCloser>;

void RequireHr(HRESULT value, const char* operation) {
  if (SUCCEEDED(value)) return;
  std::ostringstream message;
  message << operation << " failed: 0x" << std::hex
          << static_cast<std::uint32_t>(value);
  throw std::runtime_error(message.str());
}

void RequireNvapi(NvStatus value, const char* operation) {
  if (value != kNvapiOk)
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(value));
}

template <typename Function>
Function Resolve(QueryInterface query, std::uint32_t id, const char* name) {
  auto function = reinterpret_cast<Function>(query(id));
  if (!function)
    throw std::runtime_error(std::string("NvAPI interface unavailable: ") + name);
  return function;
}


std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t FeatureHalf(std::uint64_t pixel, std::uint32_t channel,
                          std::uint64_t salt) {
  static constexpr std::array<std::uint16_t, 16> kAnchors = {
      0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
      0x3400, 0xb400, 0x3800, 0xb800, 0x3a00, 0xba00, 0x3c00, 0xbc00};
  const auto mixed = SplitMix64(salt ^ pixel * 0xD6E8FEB86659FD93ull ^
                                std::uint64_t{channel + 1u} *
                                    0xA0761D6478BD642Full);
  if (pixel < 32u) return kAnchors[(pixel + channel) & 15u];
  const std::uint16_t sign = (mixed >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(9u + ((mixed >> 17u) % 8u));
  return static_cast<std::uint16_t>(
      sign | (exponent << 10u) | (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateInput(const std::string& fixture) {
  const std::uint64_t salt =
      fixture == "binary-depth" ? 0xE7037ED1A0B428DBull :
      fixture == "ramp-depth" ? 0x8EBC6AF09C88C6E3ull : 0ull;
  if (!salt) throw std::runtime_error("unsupported fixture");
  std::vector<std::uint8_t> bytes(kInputBytes);
  for (std::size_t pixel = 0;
       pixel < std::size_t{kInputWidth} * kInputHeight; ++pixel) {
    for (std::uint32_t channel = 0; channel < kChannels; ++channel)
      Store16(&bytes, (pixel * kChannels + channel) * 2u,
              FeatureHalf(pixel, channel, salt));
  }
  return bytes;
}

std::vector<std::uint8_t> GenerateDepth(const std::string& fixture) {
  std::vector<std::uint8_t> bytes(kDepthBytes);
  for (std::size_t pixel = 0; pixel < bytes.size(); ++pixel) {
    if (fixture == "binary-depth") {
      bytes[pixel] = ((pixel / kWidth + pixel % kWidth) & 1u) ? 0xffu : 0u;
    } else if (fixture == "ramp-depth") {
      const auto mixed = SplitMix64(0x589965CC75374CC3ull ^ pixel);
      bytes[pixel] = static_cast<std::uint8_t>(
          pixel < 256u ? pixel : (mixed >> 24u));
    } else {
      throw std::runtime_error("unsupported fixture");
    }
  }
  return bytes;
}

std::vector<std::uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void Write(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0, digest_bytes = 0, written = 0;
  auto require = [](NTSTATUS status, const char* operation) {
    if (status < 0) throw std::runtime_error(std::string(operation) + " failed");
  };
  require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                      nullptr, 0),
          "BCryptOpenAlgorithmProvider");
  require(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                            reinterpret_cast<PUCHAR>(&object_bytes),
                            sizeof(object_bytes), &written, 0),
          "BCryptGetProperty(object)");
  require(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                            reinterpret_cast<PUCHAR>(&digest_bytes),
                            sizeof(digest_bytes), &written, 0),
          "BCryptGetProperty(digest)");
  std::vector<std::uint8_t> object(object_bytes), digest(digest_bytes);
  require(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes,
                           nullptr, 0, 0),
          "BCryptCreateHash");
  require(BCryptHashData(hash, const_cast<PUCHAR>(bytes.data()),
                         static_cast<ULONG>(bytes.size()), 0),
          "BCryptHashData");
  require(BCryptFinishHash(hash, digest.data(), digest_bytes, 0),
          "BCryptFinishHash");
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}

struct GpuContext {
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  UniqueHandle event;
  std::uint64_t next_fence = 1;
  std::vector<ComPtr<ID3D12Resource>> pending;
};

GpuContext CreateGpuContext() {
  GpuContext gpu;
  ComPtr<ID3D12Debug> debug;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
    debug->EnableDebugLayer();
  ComPtr<IDXGIFactory6> factory;
  RequireHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
  for (UINT index = 0;; ++index) {
    ComPtr<IDXGIAdapter1> adapter;
    const HRESULT status = factory->EnumAdapterByGpuPreference(
        index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
    if (status == DXGI_ERROR_NOT_FOUND) break;
    RequireHr(status, "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC1 description{};
    RequireHr(adapter->GetDesc1(&description), "GetDesc1");
    if (description.VendorId != 0x10deu ||
        (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
      continue;
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&gpu.device)))) {
      gpu.adapter = adapter;
      break;
    }
  }
  if (!gpu.device) throw std::runtime_error("NVIDIA D3D12 device unavailable");
  D3D12_COMMAND_QUEUE_DESC queue{};
  queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  RequireHr(gpu.device->CreateCommandQueue(&queue, IID_PPV_ARGS(&gpu.queue)),
            "CreateCommandQueue");
  RequireHr(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&gpu.allocator)),
            "CreateCommandAllocator");
  RequireHr(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           gpu.allocator.Get(), nullptr,
                                           IID_PPV_ARGS(&gpu.list)),
            "CreateCommandList");
  RequireHr(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                     IID_PPV_ARGS(&gpu.fence)),
            "CreateFence");
  gpu.event.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
  if (!gpu.event) throw std::runtime_error("CreateEventW failed");
  return gpu;
}

void ExecuteAndReset(GpuContext& gpu) {
  RequireHr(gpu.list->Close(), "Close");
  ID3D12CommandList* lists[] = {gpu.list.Get()};
  gpu.queue->ExecuteCommandLists(1, lists);
  const auto value = gpu.next_fence++;
  RequireHr(gpu.queue->Signal(gpu.fence.Get(), value), "Signal");
  if (gpu.fence->GetCompletedValue() < value) {
    RequireHr(gpu.fence->SetEventOnCompletion(value, gpu.event.get()),
              "SetEventOnCompletion");
    if (WaitForSingleObject(gpu.event.get(), 120000) != WAIT_OBJECT_0)
      throw std::runtime_error("GPU fence wait timed out");
  }
  gpu.pending.clear();
  RequireHr(gpu.allocator->Reset(), "allocator Reset");
  RequireHr(gpu.list->Reset(gpu.allocator.Get(), nullptr), "list Reset");
}

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, std::uint64_t bytes,
                                    D3D12_HEAP_TYPE heap_type,
                                    D3D12_RESOURCE_FLAGS flags,
                                    D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = heap_type;
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  description.Width = bytes;
  description.Height = 1;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  description.Flags = flags;
  ComPtr<ID3D12Resource> result;
  RequireHr(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                                             &description, state, nullptr,
                                             IID_PPV_ARGS(&result)),
            "CreateCommittedResource(buffer)");
  return result;
}

void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  if (before == after) return;
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  list->ResourceBarrier(1, &barrier);
}

struct GpuBuffer {
  ComPtr<ID3D12Resource> resource;
  std::uint64_t size = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};
struct GpuTexture {
  ComPtr<ID3D12Resource> resource;
  std::uint32_t bytes_per_pixel = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

GpuBuffer UploadBuffer(GpuContext& gpu,
                       const std::vector<std::uint8_t>& bytes) {
  GpuBuffer result;
  result.size = bytes.size();
  result.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  result.resource = CreateBuffer(gpu.device.Get(), bytes.size(),
                                 D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COPY_DEST);
  auto upload = CreateBuffer(gpu.device.Get(), bytes.size(),
                             D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(buffer upload)");
  std::memcpy(mapped, bytes.data(), bytes.size());
  upload->Unmap(0, nullptr);
  gpu.list->CopyBufferRegion(result.resource.Get(), 0, upload.Get(), 0,
                             bytes.size());
  Transition(gpu.list.Get(), result.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST, result.state);
  gpu.pending.push_back(upload);
  return result;
}

GpuTexture UploadTexture(GpuContext& gpu, DXGI_FORMAT format,
                         std::uint32_t bytes_per_pixel,
                         D3D12_RESOURCE_STATES final_state,
                         const std::vector<std::uint8_t>& bytes) {
  GpuTexture result;
  result.bytes_per_pixel = bytes_per_pixel;
  result.state = final_state;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  description.Width = kWidth;
  description.Height = kHeight;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.Format = format;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.resource)),
            "CreateCommittedResource(texture)");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0, upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &upload_size);
  const std::size_t source_row = std::size_t{kWidth} * bytes_per_pixel;
  if (bytes.size() != source_row * kHeight)
    throw std::runtime_error("texture byte size mismatch");
  auto upload = CreateBuffer(gpu.device.Get(), upload_size,
                             D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(texture upload)");
  for (std::uint32_t row = 0; row < kHeight; ++row)
    std::memcpy(static_cast<std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                bytes.data() + std::size_t{row} * source_row, source_row);
  upload->Unmap(0, nullptr);
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = result.resource.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = upload.Get();
  source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  source.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  Transition(gpu.list.Get(), result.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST, final_state);
  gpu.pending.push_back(upload);
  return result;
}

std::vector<std::uint8_t> ReadBuffer(GpuContext& gpu, GpuBuffer& source) {
  auto readback = CreateBuffer(gpu.device.Get(), source.size,
                               D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), source.resource.Get(), source.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  source.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  gpu.list->CopyBufferRegion(readback.Get(), 0, source.resource.Get(), 0,
                             source.size);
  ExecuteAndReset(gpu);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(source.size));
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(source.size)};
  RequireHr(readback->Map(0, &read_range, &mapped), "Map(buffer readback)");
  std::memcpy(bytes.data(), mapped, bytes.size());
  const D3D12_RANGE no_write{0, 0};
  readback->Unmap(0, &no_write);
  return bytes;
}

std::vector<std::uint8_t> ReadTexture(GpuContext& gpu, GpuTexture& source) {
  const auto description = source.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0, readback_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &readback_size);
  auto readback = CreateBuffer(gpu.device.Get(), readback_size,
                               D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), source.resource.Get(), source.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  source.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  D3D12_TEXTURE_COPY_LOCATION source_location{};
  source_location.pResource = source.resource.Get();
  source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = readback.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  destination.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&destination, 0, 0, 0, &source_location, nullptr);
  ExecuteAndReset(gpu);
  const std::size_t output_row = std::size_t{kWidth} * source.bytes_per_pixel;
  std::vector<std::uint8_t> bytes(output_row * kHeight);
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(readback_size)};
  RequireHr(readback->Map(0, &read_range, &mapped), "Map(texture readback)");
  for (std::uint32_t row = 0; row < kHeight; ++row)
    std::memcpy(bytes.data() + std::size_t{row} * output_row,
                static_cast<const std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                output_row);
  const D3D12_RANGE no_write{0, 0};
  readback->Unmap(0, &no_write);
  return bytes;
}


std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t payload_bytes) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < kGuardBytes; ++i) {
    count += guarded[i] != kGuardValue;
    count += guarded[kGuardBytes + payload_bytes + i] != kGuardValue;
  }
  return count;
}

void Pack64(std::array<std::uint8_t, 168>* bytes, std::size_t offset,
            std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (i * 8u));
}
void Pack32(std::array<std::uint8_t, 168>* bytes, std::size_t offset,
            std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (i * 8u));
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison");
  return static_cast<std::uint8_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fg-mid-l52-nvapi <module> <output-dir> <run-id> <fixture> <poison-hex>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_root = argv[2];
    const std::string run_id = argv[3];
    const std::string fixture = argv[4];
    const std::uint8_t poison = ParsePoison(argv[5]);
    fs::create_directories(output_root);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != 9064u || Sha256(module_bytes) != kModuleSha)
      throw std::runtime_error("selected module identity mismatch");
    const auto input = GenerateInput(fixture);
    const auto depth = GenerateDepth(fixture);
    std::vector<std::uint8_t> guarded_input(kInputBytes + 2u * kGuardBytes,
                                            kGuardValue);
    std::memcpy(guarded_input.data() + kGuardBytes, input.data(), input.size());
    const std::vector<std::uint8_t> initial_vector(kOutputVectorBytes, poison);
    const std::vector<std::uint8_t> initial_scalar(kOutputScalarBytes, poison);

    GpuContext gpu = CreateGpuContext();
    GpuBuffer input_buffer = UploadBuffer(gpu, guarded_input);
    GpuTexture depth_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, depth);
    GpuTexture output_a_texture = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_vector);
    GpuTexture output_b_texture = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_vector);
    GpuTexture confidence_texture = UploadTexture(
        gpu, DXGI_FORMAT_R16_FLOAT, 2,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_scalar);
    ExecuteAndReset(gpu);

    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (!query) throw std::runtime_error("nvapi_QueryInterface unavailable");
    auto initialize = Resolve<Initialize>(query, kIdInitialize, "Initialize");
    auto get_texture = Resolve<GetMergedTextureSampler>(
        query, kIdGetMergedTextureSampler, "GetCudaMergedTextureSamplerObject");
    auto get_surface = Resolve<GetIndependentDescriptor>(
        query, kIdGetIndependentDescriptor, "GetCudaIndependentDescriptorObject");
    auto create_module =
        Resolve<CreateCuModule>(query, kIdCreateCuModule, "CreateCuModule");
    auto create_function = Resolve<CreateCuFunction>(
        query, kIdCreateCuFunction, "CreateCuFunction");
    auto launch_chain = Resolve<LaunchKernelChain>(
        query, kIdLaunchKernelChain, "LaunchKernelChain");
    auto destroy_module =
        Resolve<DestroyCuModule>(query, kIdDestroyCuModule, "DestroyCuModule");
    auto destroy_function = Resolve<DestroyCuFunction>(
        query, kIdDestroyCuFunction, "DestroyCuFunction");
    RequireNvapi(initialize(), "NvAPI_Initialize");

    D3D12_DESCRIPTOR_HEAP_DESC resource_heap_desc{};
    resource_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_heap_desc.NumDescriptors = 2048;
    resource_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_heap_desc.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_heap_desc,
                                                IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{};
    sampler_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_heap_desc.NumDescriptors = 3;
    sampler_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_heap_desc.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_heap_desc,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(samplers)");
    const UINT ri = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto rs = resource_heap->GetCPUDescriptorHandleForHeapStart();
    auto resource_slot = [&](UINT slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          rs.ptr + std::uintptr_t{slot} * ri};
    };
    D3D12_SHADER_RESOURCE_VIEW_DESC depth_srv{};
    depth_srv.Format = DXGI_FORMAT_R8_UNORM;
    depth_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    depth_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    depth_srv.Texture2D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(depth_texture.resource.Get(),
                                         &depth_srv, resource_slot(45));
    auto create_uav = [&](GpuTexture& texture, DXGI_FORMAT format, UINT slot) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
      view.Format = format;
      view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr,
                                            &view, resource_slot(slot));
    };
    create_uav(confidence_texture, DXGI_FORMAT_R16_FLOAT, 0);
    create_uav(output_b_texture, DXGI_FORMAT_R16G16_FLOAT, 25);
    create_uav(output_a_texture, DXGI_FORMAT_R16G16_FLOAT, 46);
    D3D12_SAMPLER_DESC sampler_desc{};
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    const auto sampler_handle =
        sampler_heap->GetCPUDescriptorHandleForHeapStart();
    gpu.device->CreateSampler(&sampler_desc, sampler_handle);

    MergedTextureSamplerParams texture_params{};
    texture_params.struct_size_in = sizeof(texture_params);
    texture_params.struct_size_out = sizeof(texture_params);
    texture_params.device = gpu.device.Get();
    texture_params.texture_descriptor.ptr = resource_slot(45).ptr;
    texture_params.sampler_descriptor.ptr = sampler_handle.ptr;
    RequireNvapi(get_texture(&texture_params),
                 "GetCudaMergedTextureSamplerObject");
    auto surface_object = [&](UINT slot) {
      IndependentDescriptorParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.type = IndependentDescriptorType::kSurface;
      params.descriptor.ptr = resource_slot(slot).ptr;
      RequireNvapi(get_surface(&params),
                   "GetCudaIndependentDescriptorObject");
      return params.handle;
    };
    const auto confidence_surface = surface_object(0);
    const auto output_b_surface = surface_object(25);
    const auto output_a_surface = surface_object(46);

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module),
                 "CreateCuModule");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function),
                 "CreateCuFunction");
    std::array<std::uint8_t, 168> packed{};
    Pack64(&packed, 0,
           input_buffer.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack32(&packed, 8, 1u);
    Pack64(&packed, 16, texture_params.texture_handle);
    Pack32(&packed, 24, 0u);
    Pack64(&packed, 48, output_a_surface);
    Pack64(&packed, 56, output_b_surface);
    Pack64(&packed, 64, confidence_surface);
    Pack32(&packed, 136, kWidth);
    Pack32(&packed, 140, kHeight);
    Pack32(&packed, 144, kInputWidth);
    Pack32(&packed, 148, kInputHeight);
    Pack32(&packed, 152, 0u);
    Pack32(&packed, 156, 0u);
    Pack32(&packed, 160, 3u);
    Pack32(&packed, 164, 3u);
    const KernelLaunchParams launch{
        function, {20, 12, 1}, {16, 16, 1}, 0, packed.data(),
        static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto depth_after = ReadTexture(gpu, depth_texture);
    const auto output_a = ReadTexture(gpu, output_a_texture);
    const auto output_b = ReadTexture(gpu, output_b_texture);
    const auto confidence = ReadTexture(gpu, confidence_texture);
    const auto guarded_input_after = ReadBuffer(gpu, input_buffer);
    const std::vector<std::uint8_t> input_after(
        guarded_input_after.begin() + kGuardBytes,
        guarded_input_after.begin() + kGuardBytes + kInputBytes);
    const std::size_t input_mismatches =
        Mismatches(input, input_after) + Mismatches(depth, depth_after);
    const std::size_t canary_mismatches =
        CanaryMismatches(guarded_input_after, kInputBytes);
    Write(output_root / "input-after.bin", input_after);
    Write(output_root / "depth-after.bin", depth_after);
    Write(output_root / "output-a.rg16f", output_a);
    Write(output_root / "output-b.rg16f", output_b);
    Write(output_root / "confidence.r16f", confidence);
    std::vector<std::uint8_t> inputs = input;
    inputs.insert(inputs.end(), depth.begin(), depth.end());
    std::vector<std::uint8_t> outputs = output_a;
    outputs.insert(outputs.end(), output_b.begin(), output_b.end());
    outputs.insert(outputs.end(), confidence.begin(), confidence.end());
    const bool passed = input_mismatches == 0 && canary_mismatches == 0;
    std::ofstream receipt(output_root / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.mid-l52-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"fixture\": \"" << fixture << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"selected_launch_token\": 52,\n"
            << "  \"module_token\": 6,\n"
            << "  \"module_bytes\": 9064,\n"
            << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
            << "  \"raw_live_parameter_sha256\": \"" << kRawParameterSha << "\",\n"
            << "  \"packed_parameter_bytes\": 168,\n"
            << "  \"grid\": [20, 12, 1],\n"
            << "  \"block\": [16, 16, 1],\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"input_root_sha256\": \"" << Sha256(inputs) << "\",\n"
            << "  \"output_root_sha256\": \"" << Sha256(outputs) << "\",\n"
            << "  \"output_a_sha256\": \"" << Sha256(output_a) << "\",\n"
            << "  \"output_b_sha256\": \"" << Sha256(output_b) << "\",\n"
            << "  \"confidence_sha256\": \"" << Sha256(confidence) << "\",\n"
            << "  \"output_bytes\": " << outputs.size() << ",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"buffer_canary_mismatched_bytes\": " << canary_mismatches << ",\n"
            << "  \"resource_backend\": \"live-format-d3d12-descriptors-through-nvapi\",\n"
            << "  \"expected_output_fed_to_execution\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
            << "}\n";
    std::cout << "probe=fg_mid_l52_nvapi\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "fixture=" << fixture << '\n'
              << "output_root_sha256=" << Sha256(outputs) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "buffer_canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
