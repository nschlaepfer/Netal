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

#include "compositor_l87_contract.h"

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using namespace fg_compositor_l87;

namespace {

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
    if (library != nullptr) FreeLibrary(library);
  }
};
using UniqueLibrary =
    std::unique_ptr<std::remove_pointer_t<HMODULE>, LibraryCloser>;
struct HandleCloser {
  void operator()(HANDLE handle) const noexcept {
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
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
  if (value == kNvapiOk) return;
  throw std::runtime_error(std::string(operation) + " failed: " +
                           std::to_string(value));
}

template <typename Function>
Function Resolve(QueryInterface query, std::uint32_t id, const char* name) {
  auto function = reinterpret_cast<Function>(query(id));
  if (function == nullptr)
    throw std::runtime_error(std::string("NvAPI interface unavailable: ") + name);
  return function;
}

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t RgbHalf(std::uint64_t pixel, std::uint32_t channel) {
  const auto bits = SplitMix64(0x6a09e667f3bcc909ull ^
                               pixel * 0x9e3779b97f4a7c15ull ^
                               std::uint64_t{channel + 1u} *
                                   0xbf58476d1ce4e5b9ull);
  const std::uint16_t sign = (bits >> 63u) != 0u ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((bits >> 17u) & 3u));
  return static_cast<std::uint16_t>(sign | exponent << 10u |
                                    (bits & 0x03ffu));
}

std::uint16_t AlphaHalf(std::uint64_t pixel) {
  const auto bits = SplitMix64(0xbb67ae8584caa73bull ^
                               pixel * 0x94d049bb133111ebull);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((bits >> 23u) & 3u));
  return static_cast<std::uint16_t>(exponent << 10u | (bits & 0x03ffu));
}

std::vector<std::uint8_t> GenerateHalf4(std::size_t bytes,
                                        std::uint64_t salt) {
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t pixel = 0; pixel < bytes / 8u; ++pixel) {
    const auto identity = pixel ^ salt;
    Store16(&result, pixel * 8u, RgbHalf(identity, 0));
    Store16(&result, pixel * 8u + 2u, RgbHalf(identity, 1));
    Store16(&result, pixel * 8u + 4u, RgbHalf(identity, 2));
    Store16(&result, pixel * 8u + 6u, AlphaHalf(identity));
  }
  return result;
}

std::vector<std::uint8_t> GenerateMask(bool sparse) {
  std::vector<std::uint8_t> mask(kMaskBytes);
  for (std::size_t i = 0; i < mask.size(); ++i) {
    const auto value = SplitMix64(0x3c6ef372fe94f82bull ^ i);
    mask[i] = sparse && ((i % 17u) == 0u || (i % 101u) == 7u)
                  ? 0u
                  : static_cast<std::uint8_t>(1u + value % 255u);
  }
  return mask;
}

std::vector<std::uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size " + path.string());
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
                                      nullptr, 0), "BCryptOpenAlgorithmProvider");
  std::vector<std::uint8_t> object;
  std::vector<std::uint8_t> digest;
  try {
    require(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                              reinterpret_cast<PUCHAR>(&object_bytes),
                              sizeof(object_bytes), &written, 0),
            "BCryptGetProperty(object)");
    require(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                              reinterpret_cast<PUCHAR>(&digest_bytes),
                              sizeof(digest_bytes), &written, 0),
            "BCryptGetProperty(digest)");
    object.resize(object_bytes);
    digest.resize(digest_bytes);
    require(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes,
                             nullptr, 0, 0), "BCryptCreateHash");
    require(BCryptHashData(hash, const_cast<PUCHAR>(bytes.data()),
                           static_cast<ULONG>(bytes.size()), 0),
            "BCryptHashData");
    require(BCryptFinishHash(hash, digest.data(), digest_bytes, 0),
            "BCryptFinishHash");
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
  } catch (...) {
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw;
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
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
        (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
      continue;
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&gpu.device)))) {
      gpu.adapter = adapter;
      break;
    }
  }
  if (gpu.device == nullptr) throw std::runtime_error("NVIDIA D3D12 device unavailable");
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
                                     IID_PPV_ARGS(&gpu.fence)), "CreateFence");
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
  std::uint32_t width = 0, height = 0, bytes_per_pixel = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

GpuBuffer UploadBuffer(GpuContext& gpu, const std::vector<std::uint8_t>& bytes) {
  GpuBuffer result;
  result.size = bytes.size();
  result.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  result.resource = CreateBuffer(gpu.device.Get(), bytes.size(),
                                 D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COPY_DEST);
  auto upload = CreateBuffer(gpu.device.Get(), bytes.size(), D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(buffer upload)");
  std::memcpy(mapped, bytes.data(), bytes.size());
  upload->Unmap(0, nullptr);
  gpu.list->CopyBufferRegion(result.resource.Get(), 0, upload.Get(), 0, bytes.size());
  Transition(gpu.list.Get(), result.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
             result.state);
  gpu.pending.push_back(upload);
  return result;
}

GpuTexture UploadTexture(GpuContext& gpu, DXGI_FORMAT format,
                         std::uint32_t bytes_per_pixel,
                         D3D12_RESOURCE_FLAGS flags,
                         D3D12_RESOURCE_STATES final_state,
                         const std::vector<std::uint8_t>& bytes) {
  GpuTexture result;
  result.width = kWidth;
  result.height = kHeight;
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
  description.Flags = flags;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.resource)), "CreateCommittedResource(texture)");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0, upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &upload_size);
  const std::size_t source_row = std::size_t{kWidth} * bytes_per_pixel;
  auto upload = CreateBuffer(gpu.device.Get(), upload_size, D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
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
  Transition(gpu.list.Get(), result.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
             final_state);
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
  gpu.list->CopyBufferRegion(readback.Get(), 0, source.resource.Get(), 0, source.size);
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

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
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

std::size_t PoisonPixels(const std::vector<std::uint8_t>& output,
                         std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < output.size(); i += 4u)
    count += output[i] == poison && output[i + 1u] == poison &&
             output[i + 2u] == poison && output[i + 3u] == poison;
  return count;
}

std::size_t ZeroMaskPixels(const std::vector<std::uint8_t>& mask) {
  std::size_t count = 0;
  for (const auto value : mask) count += value == 0;
  return count;
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

void Pack64(std::array<std::uint8_t, kPackedParameterBytes>* bytes,
            std::size_t offset, std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (i * 8u));
}
void Pack32(std::array<std::uint8_t, kPackedParameterBytes>* bytes,
            std::size_t offset, std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (i * 8u));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fg-compositor-l87-nvapi <module> <output-dir> <run-id> <poison-hex> <full|sparse>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    const std::string mask_mode = argv[5];
    if (mask_mode != "full" && mask_mode != "sparse")
      throw std::runtime_error("unsupported mask mode");
    const bool sparse = mask_mode == "sparse";
    fs::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != kModuleBytes || Sha256(module_bytes) != kModuleSha256)
      throw std::runtime_error("selected module identity mismatch");

    std::array<std::vector<std::uint8_t>, 4> pyramid;
    std::array<std::vector<std::uint8_t>, 4> guarded_pyramid;
    for (std::size_t i = 0; i < pyramid.size(); ++i) {
      pyramid[i] = GenerateHalf4(kPyramidBytes[i], 0x100000001b3ull * (i + 1));
      guarded_pyramid[i].assign(pyramid[i].size() + 2u * kGuardBytes, kGuardValue);
      std::memcpy(guarded_pyramid[i].data() + kGuardBytes, pyramid[i].data(),
                  pyramid[i].size());
    }
    const auto mask = GenerateMask(sparse);
    const std::vector<std::uint8_t> initial_output(kOutputBytes, poison);
    const std::vector<std::uint8_t> initial_float_output(
        std::size_t{kWidth} * kHeight * 16u, poison);

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading real nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (query == nullptr) throw std::runtime_error("nvapi_QueryInterface unavailable");
    const auto initialize = Resolve<Initialize>(query, kIdInitialize, "NvAPI_Initialize");
    const auto get_texture = Resolve<GetMergedTextureSampler>(
        query, kIdGetMergedTextureSampler, "GetCudaMergedTextureSamplerObject");
    const auto get_surface = Resolve<GetIndependentDescriptor>(
        query, kIdGetIndependentDescriptor, "GetCudaIndependentDescriptorObject");
    const auto create_module = Resolve<CreateCuModule>(query, kIdCreateCuModule,
                                                        "CreateCuModule");
    const auto create_function = Resolve<CreateCuFunction>(
        query, kIdCreateCuFunction, "CreateCuFunction");
    const auto launch_chain = Resolve<LaunchKernelChain>(
        query, kIdLaunchKernelChain, "LaunchKernelChain");
    const auto destroy_module = Resolve<DestroyCuModule>(
        query, kIdDestroyCuModule, "DestroyCuModule");
    const auto destroy_function = Resolve<DestroyCuFunction>(
        query, kIdDestroyCuFunction, "DestroyCuFunction");
    RequireNvapi(initialize(), "NvAPI_Initialize");

    std::array<GpuBuffer, 4> buffers;
    for (std::size_t i = 0; i < buffers.size(); ++i)
      buffers[i] = UploadBuffer(gpu, guarded_pyramid[i]);
    GpuTexture mask_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 1, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mask);
    GpuTexture output_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_output);
    GpuTexture float_output_texture = UploadTexture(
        gpu, DXGI_FORMAT_R32G32B32A32_FLOAT, 16,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_float_output);
    ExecuteAndReset(gpu);

    D3D12_DESCRIPTOR_HEAP_DESC resource_heap_description{};
    resource_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_heap_description.NumDescriptors = 2048;
    resource_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_heap_description,
                                                IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_description{};
    sampler_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_heap_description.NumDescriptors = 3;
    sampler_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_heap_description,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(sampler)");
    const UINT resource_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto resource_start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uav{resource_start.ptr + 54ull * resource_increment};
    D3D12_CPU_DESCRIPTOR_HANDLE float_uav{
        resource_start.ptr + 55ull * resource_increment};
    D3D12_CPU_DESCRIPTOR_HANDLE srv{resource_start.ptr + 59ull * resource_increment};
    D3D12_CPU_DESCRIPTOR_HANDLE sampler =
        sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_description{};
    srv_description.Format = DXGI_FORMAT_R8_UNORM;
    srv_description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_description.Texture2D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(mask_texture.resource.Get(),
                                         &srv_description, srv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_description{};
    uav_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav_description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(output_texture.resource.Get(), nullptr,
                                          &uav_description, uav);
    D3D12_UNORDERED_ACCESS_VIEW_DESC float_uav_description{};
    float_uav_description.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    float_uav_description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(float_output_texture.resource.Get(),
                                          nullptr, &float_uav_description,
                                          float_uav);
    D3D12_SAMPLER_DESC sampler_description{};
    sampler_description.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_description.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&sampler_description, sampler);

    MergedTextureSamplerParams texture_params{};
    texture_params.struct_size_in = sizeof(texture_params);
    texture_params.struct_size_out = sizeof(texture_params);
    texture_params.device = gpu.device.Get();
    texture_params.texture_descriptor.ptr = srv.ptr;
    texture_params.sampler_descriptor.ptr = sampler.ptr;
    RequireNvapi(get_texture(&texture_params), "GetCudaMergedTextureSamplerObject");
    IndependentDescriptorParams surface_params{};
    surface_params.struct_size_in = sizeof(surface_params);
    surface_params.struct_size_out = sizeof(surface_params);
    surface_params.device = gpu.device.Get();
    surface_params.type = IndependentDescriptorType::kSurface;
    surface_params.descriptor.ptr = uav.ptr;
    RequireNvapi(get_surface(&surface_params), "GetCudaIndependentDescriptorObject");
    IndependentDescriptorParams float_surface_params{};
    float_surface_params.struct_size_in = sizeof(float_surface_params);
    float_surface_params.struct_size_out = sizeof(float_surface_params);
    float_surface_params.device = gpu.device.Get();
    float_surface_params.type = IndependentDescriptorType::kSurface;
    float_surface_params.descriptor.ptr = float_uav.ptr;
    RequireNvapi(get_surface(&float_surface_params),
                 "GetCudaIndependentDescriptorObject(float)");
    if (surface_params.handle == 0 || float_surface_params.handle == 0)
      throw std::runtime_error("NvAPI returned a null surface handle");

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module), "CreateCuModule");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel", &function),
                 "CreateCuFunction");
    std::array<std::uint8_t, kPackedParameterBytes> packed{};
    Pack64(&packed, 0, texture_params.texture_handle);
    Pack64(&packed, 8, surface_params.handle);
    Pack32(&packed, 16, 0);
    Pack32(&packed, 20, 0);
    for (std::size_t i = 0; i < buffers.size(); ++i)
      Pack64(&packed, 24u + i * 8u,
             buffers[i].resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack32(&packed, 56, kSelectedLevel);
    Pack32(&packed, 60, kWidth);
    Pack32(&packed, 64, kHeight);
    Pack32(&packed, 68, 0);
    const KernelLaunchParams launch{
        function, {kGridX, kGridY, 1}, {kBlockX, 1, 1}, 0,
        packed.data(), static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1), "LaunchKernelChain");
    ExecuteAndReset(gpu);
    Pack64(&packed, 8, float_surface_params.handle);
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(float diagnostic)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function), "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto actual_mask = ReadTexture(gpu, mask_texture);
    const auto output = ReadTexture(gpu, output_texture);
    const auto float_output = ReadTexture(gpu, float_output_texture);
    std::size_t input_mismatches = Mismatches(mask, actual_mask);
    std::size_t canary_mismatches = 0;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      const auto guarded = ReadBuffer(gpu, buffers[i]);
      input_mismatches += Mismatches(guarded_pyramid[i], guarded);
      canary_mismatches += CanaryMismatches(guarded, pyramid[i].size());
    }
    const std::size_t zero_mask_pixels = ZeroMaskPixels(mask);
    const std::size_t poison_pixels = PoisonPixels(output, poison);
    const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                        poison_pixels == zero_mask_pixels;
    Write(output_directory / "input-level0-after.rgba16f", pyramid[0]);
    Write(output_directory / "input-mask-after.r8", actual_mask);
    Write(output_directory / "output.rgba8", output);
    Write(output_directory / "diagnostic-output.rgba32f", float_output);
    std::ofstream receipt(output_directory / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    if (!receipt) throw std::runtime_error("unable to write receipt");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.compositor-l87-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"mask_mode\": \"" << mask_mode << "\",\n"
            << "  \"selected_launch_token\": 87,\n"
            << "  \"descriptor_trace_launch_alias\": 85,\n"
            << "  \"module_token\": 20,\n"
            << "  \"module_bytes\": " << module_bytes.size() << ",\n"
            << "  \"module_sha256\": \"" << kModuleSha256 << "\",\n"
            << "  \"raw_parameter_sha256\": \"" << kRawParameterSha256 << "\",\n"
            << "  \"address_free_parameter_sha256\": \""
            << kAddressFreeParameterSha256 << "\",\n"
            << "  \"packed_parameter_bytes\": 72,\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"numerical_domain\": \"finite-normal-half4-positive-normal-alpha\",\n"
            << "  \"level0_bytes\": " << pyramid[0].size() << ",\n"
            << "  \"level0_sha256\": \"" << Sha256(pyramid[0]) << "\",\n"
            << "  \"mask_bytes\": " << actual_mask.size() << ",\n"
            << "  \"mask_sha256\": \"" << Sha256(actual_mask) << "\",\n"
            << "  \"output_bytes\": " << output.size() << ",\n"
            << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
            << "  \"diagnostic_float_output_bytes\": "
            << float_output.size() << ",\n"
            << "  \"diagnostic_float_output_sha256\": \""
            << Sha256(float_output) << "\",\n"
            << "  \"zero_mask_pixels\": " << zero_mask_pixels << ",\n"
            << "  \"poison_pixels_after\": " << poison_pixels << ",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
            << "  \"external_expected_tensors_consumed\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n}\n";
    std::cout << "probe=fg_compositor_l87_nvapi\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "texture_handle_nonzero=" << (texture_params.texture_handle != 0) << '\n'
              << "surface_handle_nonzero=" << (surface_params.handle != 0) << '\n'
              << "output_sha256=" << Sha256(output) << '\n'
              << "poison_pixels_after=" << poison_pixels << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
