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

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 360;
constexpr std::uint32_t kPaddedHeight = 368;
constexpr std::size_t kMotionBytes =
    static_cast<std::size_t>(kWidth) * kPaddedHeight * 16u;
constexpr std::size_t kColorBytes =
    static_cast<std::size_t>(kWidth) * kHeight * 8u;
constexpr std::size_t kMaskBytes =
    static_cast<std::size_t>(kWidth) * kHeight;
constexpr std::array<std::size_t, 4> kPyramidBytes = {
    320u * 180u * 8u, 160u * 90u * 8u, 80u * 45u * 8u, 40u * 23u * 8u};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kModuleSha[] =
    "3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76";
constexpr char kRawParameterSha[] =
    "31cfc2016f76ad79fe92717d4bb3b38ceb6c8b3f14968f6b5f1e2f66503e548a";
constexpr char kGenerator[] =
    "splitmix64-fg-postprocess-l84-independent-textures-motion-v1";

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

std::uint16_t PositiveUnitHalf(std::uint64_t key) {
  const auto mixed = SplitMix64(key);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((mixed >> 17u) % 4u));
  return static_cast<std::uint16_t>((exponent << 10u) | (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateColor(std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(kColorBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * kWidth + x;
      for (std::uint32_t channel = 0; channel < 3; ++channel)
        Store16(&bytes, pixel * 8u + channel * 2u,
                PositiveUnitHalf(seed ^ pixel * 0xD6E8FEB86659FD93ull ^
                                 channel * 0xA0761D6478BD642Full));
      Store16(&bytes, pixel * 8u + 6u, 0x3c00u);
    }
  }
  return bytes;
}

std::uint16_t SmallSignedHalf(std::uint64_t key) {
  const auto mixed = SplitMix64(key);
  const std::uint16_t sign = (mixed >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(3u + ((mixed >> 11u) % 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateMotion(std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(kMotionBytes);
  for (std::uint32_t y = 0; y < kPaddedHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * kWidth + x;
      for (std::uint32_t channel = 0; channel < 4; ++channel)
        Store16(&bytes, pixel * 16u + channel * 2u,
                SmallSignedHalf(seed ^ pixel * 0xE7037ED1A0B428DBull ^
                                channel * 0x8EBC6AF09C88C6E3ull));
      const auto mixed = SplitMix64(seed ^ pixel * 0x589965CC75374CC3ull);
      const std::uint16_t logit = static_cast<std::uint16_t>(
          ((mixed >> 63u) ? 0x8000u : 0u) | (14u << 10u) |
          (mixed & 0x03ffu));
      Store16(&bytes, pixel * 16u + 8u, logit);
      Store16(&bytes, pixel * 16u + 10u, SmallSignedHalf(mixed ^ 0x11u));
      Store16(&bytes, pixel * 16u + 12u, SmallSignedHalf(mixed ^ 0x22u));
      Store16(&bytes, pixel * 16u + 14u, SmallSignedHalf(mixed ^ 0x33u));
    }
  }
  return bytes;
}

std::vector<std::uint8_t> GenerateMask(const std::string& fixture) {
  std::vector<std::uint8_t> bytes(kMaskBytes, 0u);
  if (fixture == "open") return bytes;
  if (fixture != "islands") throw std::runtime_error("unsupported fixture");
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const bool island =
          ((x >= 104u && x < 224u && y >= 72u && y < 184u) ||
           (x >= 376u && x < 552u && y >= 196u && y < 320u));
      const bool holes = ((x * 13u + y * 7u) % 47u) == 0u;
      bytes[static_cast<std::size_t>(y) * kWidth + x] =
          island && !holes ? 0xffu : 0u;
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

void Pack64(std::array<std::uint8_t, 152>* bytes, std::size_t offset,
            std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (i * 8u));
}
void Pack32(std::array<std::uint8_t, 152>* bytes, std::size_t offset,
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
    std::cerr << "usage: fg-postprocess-l84-nvapi <module> <output-dir> <run-id> <fixture> <poison-hex>\n";
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
    if (module_bytes.size() != 33936u || Sha256(module_bytes) != kModuleSha)
      throw std::runtime_error("selected module identity mismatch");
    const std::uint64_t fixture_seed =
        fixture == "open" ? 0xA0761D6478BD642Full :
        fixture == "islands" ? 0xE7037ED1A0B428DBull : 0ull;
    if (!fixture_seed) throw std::runtime_error("unsupported fixture");
    const auto color_a = GenerateColor(fixture_seed ^ 0x11111111ull);
    const auto color_b = GenerateColor(fixture_seed ^ 0x22222222ull);
    const auto mask = GenerateMask(fixture);
    const auto motion = GenerateMotion(fixture_seed ^ 0x33333333ull);
    const std::vector<std::uint8_t> initial_color(kWidth * kHeight * 4u,
                                                  poison);
    const std::vector<std::uint8_t> initial_mask(kMaskBytes, poison);
    std::vector<std::uint8_t> guarded_motion(kMotionBytes + 2u * kGuardBytes,
                                             kGuardValue);
    std::memcpy(guarded_motion.data() + kGuardBytes, motion.data(), motion.size());
    std::array<std::vector<std::uint8_t>, 4> guarded_pyramid;
    for (std::size_t i = 0; i < guarded_pyramid.size(); ++i)
      guarded_pyramid[i].assign(kPyramidBytes[i] + 2u * kGuardBytes, poison);
    for (auto& bytes : guarded_pyramid) {
      std::fill(bytes.begin(), bytes.begin() + kGuardBytes, kGuardValue);
      std::fill(bytes.end() - kGuardBytes, bytes.end(), kGuardValue);
    }

    GpuContext gpu = CreateGpuContext();
    GpuTexture color_a_texture = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, color_a);
    GpuTexture color_b_texture = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, color_b);
    GpuTexture mask_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mask);
    GpuTexture output_color_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_color);
    GpuTexture output_mask_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initial_mask);
    GpuBuffer motion_buffer = UploadBuffer(gpu, guarded_motion);
    std::array<GpuBuffer, 4> pyramid_buffers;
    for (std::size_t i = 0; i < pyramid_buffers.size(); ++i)
      pyramid_buffers[i] = UploadBuffer(gpu, guarded_pyramid[i]);
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
    const UINT resource_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT sampler_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    const auto resource_start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto sampler_start = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    auto resource_slot = [&](UINT slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          resource_start.ptr + std::uintptr_t{slot} * resource_increment};
    };
    auto sampler_slot = [&](UINT slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          sampler_start.ptr + std::uintptr_t{slot} * sampler_increment};
    };
    D3D12_SHADER_RESOURCE_VIEW_DESC color_srv{};
    color_srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    color_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    color_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    color_srv.Texture2D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(color_a_texture.resource.Get(),
                                         &color_srv, resource_slot(55));
    gpu.device->CreateShaderResourceView(color_b_texture.resource.Get(),
                                         &color_srv, resource_slot(56));
    D3D12_SHADER_RESOURCE_VIEW_DESC mask_srv{};
    mask_srv.Format = DXGI_FORMAT_R8_UNORM;
    mask_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    mask_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    mask_srv.Texture2D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(mask_texture.resource.Get(),
                                         &mask_srv, resource_slot(57));
    D3D12_UNORDERED_ACCESS_VIEW_DESC color_uav{};
    color_uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    color_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(output_color_texture.resource.Get(),
                                          nullptr, &color_uav,
                                          resource_slot(54));
    D3D12_UNORDERED_ACCESS_VIEW_DESC mask_uav{};
    mask_uav.Format = DXGI_FORMAT_R8_UNORM;
    mask_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(output_mask_texture.resource.Get(),
                                          nullptr, &mask_uav,
                                          resource_slot(58));
    D3D12_SAMPLER_DESC sampler_desc{};
    sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    for (UINT i = 0; i < 3; ++i)
      gpu.device->CreateSampler(&sampler_desc, sampler_slot(i));

    auto texture_object = [&](UINT resource_index, UINT sampler_index) {
      MergedTextureSamplerParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.texture_descriptor.ptr = resource_slot(resource_index).ptr;
      params.sampler_descriptor.ptr = sampler_slot(sampler_index).ptr;
      RequireNvapi(get_texture(&params), "GetCudaMergedTextureSamplerObject");
      return params.texture_handle;
    };
    auto surface_object = [&](UINT resource_index) {
      IndependentDescriptorParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.type = IndependentDescriptorType::kSurface;
      params.descriptor.ptr = resource_slot(resource_index).ptr;
      RequireNvapi(get_surface(&params), "GetCudaIndependentDescriptorObject");
      return params.handle;
    };
    const auto output_color_surface = surface_object(54);
    const auto color_a_object = texture_object(55, 0);
    const auto color_b_object = texture_object(56, 1);
    const auto mask_object = texture_object(57, 2);
    const auto output_mask_surface = surface_object(58);

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module),
                 "CreateCuModule");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function),
                 "CreateCuFunction");
    std::array<std::uint8_t, 152> packed{};
    Pack64(&packed, 0, output_color_surface);
    Pack64(&packed, 16, color_a_object);
    Pack64(&packed, 24, color_b_object);
    Pack64(&packed, 64,
           motion_buffer.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack64(&packed, 72, mask_object);
    for (std::size_t i = 0; i < pyramid_buffers.size(); ++i)
      Pack64(&packed, 80u + i * 8u,
             pyramid_buffers[i].resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack64(&packed, 112, output_mask_surface);
    Pack32(&packed, 128, kWidth);
    Pack32(&packed, 132, kPaddedHeight);
    Pack32(&packed, 136, kWidth);
    Pack32(&packed, 140, kHeight);
    Pack32(&packed, 144, 0x3f000000u);
    Pack32(&packed, 148, 1u);
    const KernelLaunchParams launch{
        function, {40, 23, 1}, {256, 1, 1}, 0, packed.data(),
        static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto color_a_after = ReadTexture(gpu, color_a_texture);
    const auto color_b_after = ReadTexture(gpu, color_b_texture);
    const auto mask_after = ReadTexture(gpu, mask_texture);
    const auto output_color = ReadTexture(gpu, output_color_texture);
    const auto output_mask = ReadTexture(gpu, output_mask_texture);
    const auto guarded_motion_after = ReadBuffer(gpu, motion_buffer);
    std::array<std::vector<std::uint8_t>, 4> guarded_pyramid_after;
    std::array<std::vector<std::uint8_t>, 4> pyramid_after;
    std::size_t canary_mismatches =
        CanaryMismatches(guarded_motion_after, kMotionBytes);
    for (std::size_t i = 0; i < pyramid_buffers.size(); ++i) {
      guarded_pyramid_after[i] = ReadBuffer(gpu, pyramid_buffers[i]);
      canary_mismatches +=
          CanaryMismatches(guarded_pyramid_after[i], kPyramidBytes[i]);
      pyramid_after[i] = {
          guarded_pyramid_after[i].begin() + kGuardBytes,
          guarded_pyramid_after[i].begin() + kGuardBytes + kPyramidBytes[i]};
    }
    const std::vector<std::uint8_t> motion_after(
        guarded_motion_after.begin() + kGuardBytes,
        guarded_motion_after.begin() + kGuardBytes + kMotionBytes);
    const std::size_t input_mismatches =
        Mismatches(color_a, color_a_after) + Mismatches(color_b, color_b_after) +
        Mismatches(mask, mask_after) + Mismatches(motion, motion_after);
    Write(output_root / "color-a-after.bin", color_a_after);
    Write(output_root / "color-b-after.bin", color_b_after);
    Write(output_root / "mask-after.bin", mask_after);
    Write(output_root / "motion-after.bin", motion_after);
    Write(output_root / "output-color.bin", output_color);
    Write(output_root / "output-mask.bin", output_mask);
    for (std::size_t i = 0; i < pyramid_after.size(); ++i)
      Write(output_root / ("pyramid-" + std::to_string(i) + ".bin"),
            pyramid_after[i]);
    std::vector<std::uint8_t> input_combined = color_a;
    input_combined.insert(input_combined.end(), color_b.begin(), color_b.end());
    input_combined.insert(input_combined.end(), mask.begin(), mask.end());
    input_combined.insert(input_combined.end(), motion.begin(), motion.end());
    std::vector<std::uint8_t> output_combined = output_color;
    output_combined.insert(output_combined.end(), output_mask.begin(),
                           output_mask.end());
    for (const auto& level : pyramid_after)
      output_combined.insert(output_combined.end(), level.begin(), level.end());
    const bool passed = input_mismatches == 0 && canary_mismatches == 0;
    std::ofstream receipt(output_root / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.postprocess-l84-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"fixture\": \"" << fixture << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"selected_launch_token\": 84,\n"
            << "  \"module_token\": 17,\n"
            << "  \"module_bytes\": 33936,\n"
            << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
            << "  \"raw_live_parameter_sha256\": \"" << kRawParameterSha
            << "\",\n"
            << "  \"packed_parameter_bytes\": 152,\n"
            << "  \"grid\": [40, 23, 1],\n"
            << "  \"block\": [256, 1, 1],\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"resource_backend\": \"live-format-d3d12-descriptors-through-nvapi\",\n"
            << "  \"input_root_sha256\": \"" << Sha256(input_combined)
            << "\",\n"
            << "  \"output_root_sha256\": \"" << Sha256(output_combined)
            << "\",\n"
            << "  \"output_color_sha256\": \"" << Sha256(output_color)
            << "\",\n"
            << "  \"output_mask_sha256\": \"" << Sha256(output_mask)
            << "\",\n"
            << "  \"pyramid_sha256\": [\"" << Sha256(pyramid_after[0])
            << "\", \"" << Sha256(pyramid_after[1]) << "\", \""
            << Sha256(pyramid_after[2]) << "\", \""
            << Sha256(pyramid_after[3]) << "\"],\n"
            << "  \"output_bytes\": " << output_combined.size() << ",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"buffer_canary_mismatched_bytes\": "
            << canary_mismatches << ",\n"
            << "  \"expected_output_fed_to_execution\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail")
            << "\"\n}\n";
    std::cout << "probe=fg_postprocess_l84_nvapi\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "fixture=" << fixture << '\n'
              << "output_root_sha256=" << Sha256(output_combined) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "buffer_canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
