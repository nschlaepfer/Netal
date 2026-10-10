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

#include "fg_mid_l55_contract.h"
#include "fg_mid_l55_inputs.h"

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using namespace fg_mid_l55;

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
  kSurface = 0, kTexture = 1, kSampler = 2,
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
  if (gpu.device == nullptr)
    throw std::runtime_error("NVIDIA D3D12 device unavailable");
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
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
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
                         std::uint32_t width, std::uint32_t height,
                         std::uint32_t bytes_per_pixel,
                         D3D12_RESOURCE_STATES final_state,
                         const std::vector<std::uint8_t>& bytes) {
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != source_row * height)
    throw std::runtime_error("texture upload size drift");
  GpuTexture result;
  result.width = width;
  result.height = height;
  result.bytes_per_pixel = bytes_per_pixel;
  result.format = format;
  result.state = final_state;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap.CreationNodeMask = 1;
  heap.VisibleNodeMask = 1;
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  description.Width = width;
  description.Height = height;
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
  auto upload = CreateBuffer(gpu.device.Get(), upload_size,
                             D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(texture upload)");
  for (std::uint32_t row = 0; row < height; ++row)
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
  D3D12_TEXTURE_COPY_LOCATION src{};
  src.pResource = source.resource.Get();
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION dst{};
  dst.pResource = readback.Get();
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  ExecuteAndReset(gpu);
  const std::size_t output_row =
      std::size_t{source.width} * source.bytes_per_pixel;
  std::vector<std::uint8_t> bytes(output_row * source.height);
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(readback_size)};
  RequireHr(readback->Map(0, &read_range, &mapped), "Map(texture readback)");
  for (std::uint32_t row = 0; row < source.height; ++row)
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
  for (std::size_t index = 0; index < a.size(); ++index)
    count += a[index] != b[index];
  return count;
}

std::vector<std::uint8_t> Guarded(const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> bytes(payload.size() + 2u * kGuardBytes,
                                  kGuardValue);
  std::memcpy(bytes.data() + kGuardBytes, payload.data(), payload.size());
  return bytes;
}

std::vector<std::uint8_t> PoisonGuarded(std::size_t payload_bytes,
                                        std::uint8_t poison) {
  std::vector<std::uint8_t> bytes(payload_bytes + 2u * kGuardBytes,
                                  kGuardValue);
  std::memset(bytes.data() + kGuardBytes, poison, payload_bytes);
  return bytes;
}

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t payload_bytes) {
  return {guarded.begin() + kGuardBytes,
          guarded.begin() + kGuardBytes + payload_bytes};
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t payload_bytes) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    count += guarded[index] != kGuardValue;
    count += guarded[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return count;
}

std::size_t UntouchedWords(const std::vector<std::uint8_t>& bytes,
                           std::uint8_t poison, std::size_t word_bytes) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += word_bytes) {
    bool untouched = true;
    for (std::size_t lane = 0; lane < word_bytes; ++lane)
      untouched = untouched && bytes[offset + lane] == poison;
    count += untouched;
  }
  return count;
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

void Pack64(std::array<std::uint8_t, kPackedParameterBytes>* bytes,
            std::size_t offset, std::uint64_t value) {
  for (std::size_t index = 0; index < 8; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (8u * index));
}
void Pack32(std::array<std::uint8_t, kPackedParameterBytes>* bytes,
            std::size_t offset, std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (8u * index));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: fg-mid-l55-rtx <module> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    fs::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != kModuleBytes ||
        Sha256(module_bytes) != kModuleSha256)
      throw std::runtime_error("selected module identity mismatch");
    const GeneratedInputs inputs;

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading real nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (query == nullptr) throw std::runtime_error("nvapi_QueryInterface unavailable");
    const auto initialize = Resolve<Initialize>(query, kIdInitialize,
                                                "NvAPI_Initialize");
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

    const auto candidate_a_guarded = Guarded(inputs.candidate_a);
    const auto candidate_b_guarded = Guarded(inputs.candidate_b);
    const auto linear_a_initial = PoisonGuarded(kLinearRgbaBytes, poison);
    const auto linear_b_initial = PoisonGuarded(kLinearRgbaBytes, poison);
    const auto linear_mask_initial = PoisonGuarded(kLinearMaskBytes, poison);
    GpuBuffer candidate_a = UploadBuffer(gpu, candidate_a_guarded);
    GpuBuffer candidate_b = UploadBuffer(gpu, candidate_b_guarded);
    GpuBuffer linear_a = UploadBuffer(gpu, linear_a_initial);
    GpuBuffer linear_b = UploadBuffer(gpu, linear_b_initial);
    GpuBuffer linear_mask = UploadBuffer(gpu, linear_mask_initial);
    GpuTexture rgba_a = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, inputs.rgba_a);
    GpuTexture rgba_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, inputs.rgba_b);
    GpuTexture motion = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, kCandidateWidth, kCandidateHeight, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, inputs.motion);
    GpuTexture blend = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, kCandidateWidth, kCandidateHeight, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, inputs.blend);
    const std::vector<std::uint8_t> poison_surface_rgba(kSurfaceRgbaBytes,
                                                        poison);
    const std::vector<std::uint8_t> poison_surface_mask(kSurfaceMaskBytes,
                                                        poison);
    GpuTexture surface_a = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, poison_surface_rgba);
    GpuTexture surface_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, poison_surface_rgba);
    GpuTexture surface_mask = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, kActiveWidth, kActiveHeight, 1,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, poison_surface_mask);
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
              "CreateDescriptorHeap(samplers)");
    const UINT increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto resource_start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto sampler_handle = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC sampler_description{};
    sampler_description.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_description.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;

    auto make_texture = [&](GpuTexture& texture, UINT slot) {
      D3D12_CPU_DESCRIPTOR_HANDLE srv{resource_start.ptr +
                                      std::uint64_t{slot} * increment};
      D3D12_SHADER_RESOURCE_VIEW_DESC description{};
      description.Format = texture.format;
      description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      description.Texture2D.MipLevels = 1;
      gpu.device->CreateShaderResourceView(texture.resource.Get(), &description,
                                           srv);
      gpu.device->CreateSampler(&sampler_description, sampler_handle);
      MergedTextureSamplerParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.texture_descriptor.ptr = srv.ptr;
      params.sampler_descriptor.ptr = sampler_handle.ptr;
      RequireNvapi(get_texture(&params), "GetCudaMergedTextureSamplerObject");
      if (params.texture_handle == 0)
        throw std::runtime_error("NvAPI returned null texture handle");
      return params.texture_handle;
    };
    auto make_surface = [&](GpuTexture& texture, UINT slot) {
      D3D12_CPU_DESCRIPTOR_HANDLE uav{resource_start.ptr +
                                      std::uint64_t{slot} * increment};
      D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
      description.Format = texture.format;
      description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr,
                                            &description, uav);
      IndependentDescriptorParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.type = IndependentDescriptorType::kSurface;
      params.descriptor.ptr = uav.ptr;
      RequireNvapi(get_surface(&params), "GetCudaIndependentDescriptorObject");
      if (params.handle == 0)
        throw std::runtime_error("NvAPI returned null surface handle");
      return params.handle;
    };

    const std::uint64_t rgba_a_texture = make_texture(rgba_a, 40);
    const std::uint64_t rgba_b_texture = make_texture(rgba_b, 41);
    const std::uint64_t motion_texture = make_texture(motion, 42);
    const std::uint64_t blend_texture = make_texture(blend, 43);
    const std::uint64_t surface_a_handle = make_surface(surface_a, 50);
    const std::uint64_t surface_b_handle = make_surface(surface_b, 51);
    const std::uint64_t surface_mask_handle = make_surface(surface_mask, 52);

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module), "CreateCuModule");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel", &function),
                 "CreateCuFunction");
    std::array<std::uint8_t, kPackedParameterBytes> packed{};
    Pack64(&packed, 0, rgba_a_texture);
    Pack64(&packed, 8, rgba_b_texture);
    Pack64(&packed, 16, motion_texture);
    Pack64(&packed, 24,
           candidate_a.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack64(&packed, 32,
           candidate_b.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack64(&packed, 40, blend_texture);
    Pack64(&packed, 48, surface_a_handle);
    Pack64(&packed, 56, surface_b_handle);
    Pack64(&packed, 64, surface_mask_handle);
    Pack64(&packed, 72, linear_a.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack64(&packed, 80, linear_b.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack64(&packed, 88,
           linear_mask.resource->GetGPUVirtualAddress() + kGuardBytes);
    Pack32(&packed, 184, kPhysicalWidth);
    Pack32(&packed, 188, kPhysicalHeight);
    Pack32(&packed, 192, kAccumulationMode);
    Pack32(&packed, 196, kCandidateWidth);
    Pack32(&packed, 200, kCandidateHeight);
    Pack32(&packed, 204, kCandidateWidth);
    Pack32(&packed, 208, kCandidateHeight);
    Pack32(&packed, 212, kActiveWidth);
    Pack32(&packed, 216, kActiveHeight);
    packed[220] = static_cast<std::uint8_t>(kOptionalUiEnabled);
    const KernelLaunchParams launch{
        function, {kGridX, kGridY, 1}, {kBlockX, 1, 1}, 0,
        packed.data(), static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1), "LaunchKernelChain");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto actual_surface_a = ReadTexture(gpu, surface_a);
    const auto actual_surface_b = ReadTexture(gpu, surface_b);
    const auto actual_surface_mask = ReadTexture(gpu, surface_mask);
    const auto actual_rgba_a = ReadTexture(gpu, rgba_a);
    const auto actual_rgba_b = ReadTexture(gpu, rgba_b);
    const auto actual_motion = ReadTexture(gpu, motion);
    const auto actual_blend = ReadTexture(gpu, blend);
    const auto candidate_a_after = ReadBuffer(gpu, candidate_a);
    const auto candidate_b_after = ReadBuffer(gpu, candidate_b);
    const auto linear_a_after = ReadBuffer(gpu, linear_a);
    const auto linear_b_after = ReadBuffer(gpu, linear_b);
    const auto linear_mask_after = ReadBuffer(gpu, linear_mask);
    const auto actual_linear_a = Payload(linear_a_after, kLinearRgbaBytes);
    const auto actual_linear_b = Payload(linear_b_after, kLinearRgbaBytes);
    const auto actual_linear_mask = Payload(linear_mask_after, kLinearMaskBytes);
    std::size_t input_mismatches =
        Mismatches(inputs.rgba_a, actual_rgba_a) +
        Mismatches(inputs.rgba_b, actual_rgba_b) +
        Mismatches(inputs.motion, actual_motion) +
        Mismatches(inputs.blend, actual_blend) +
        Mismatches(candidate_a_guarded, candidate_a_after) +
        Mismatches(candidate_b_guarded, candidate_b_after);
    const std::size_t canary_mismatches =
        CanaryMismatches(candidate_a_after, kCandidateBufferBytes) +
        CanaryMismatches(candidate_b_after, kCandidateBufferBytes) +
        CanaryMismatches(linear_a_after, kLinearRgbaBytes) +
        CanaryMismatches(linear_b_after, kLinearRgbaBytes) +
        CanaryMismatches(linear_mask_after, kLinearMaskBytes);
    const std::size_t poison_pattern_words =
        UntouchedWords(actual_surface_a, poison, 2) +
        UntouchedWords(actual_surface_b, poison, 2) +
        UntouchedWords(actual_surface_mask, poison, 1) +
        UntouchedWords(actual_linear_a, poison, 2) +
        UntouchedWords(actual_linear_b, poison, 2) +
        UntouchedWords(actual_linear_mask, poison, 2);
    std::vector<std::uint8_t> combined;
    for (const auto* value : {&actual_surface_a, &actual_surface_b,
                              &actual_surface_mask, &actual_linear_a,
                              &actual_linear_b, &actual_linear_mask})
      combined.insert(combined.end(), value->begin(), value->end());
    const bool passed = input_mismatches == 0 && canary_mismatches == 0;
    Write(output_directory / "surface-a.rgba16f", actual_surface_a);
    Write(output_directory / "surface-b.rgba16f", actual_surface_b);
    Write(output_directory / "surface-mask.r8", actual_surface_mask);
    Write(output_directory / "linear-a.rgba16f", actual_linear_a);
    Write(output_directory / "linear-b.rgba16f", actual_linear_b);
    Write(output_directory / "linear-mask.r16f", actual_linear_mask);
    Write(output_directory / "input-rgba-a.rgba16f", actual_rgba_a);
    Write(output_directory / "input-rgba-b.rgba16f", actual_rgba_b);
    Write(output_directory / "input-motion.rg16f", actual_motion);
    Write(output_directory / "input-blend.rg16f", actual_blend);
    std::ofstream receipt(output_directory / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    if (!receipt) throw std::runtime_error("unable to write receipt");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.mid-l55-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"selected_launch_token\": 55,\n"
            << "  \"descriptor_trace_launch_alias\": 53,\n"
            << "  \"module_token\": 3,\n"
            << "  \"module_bytes\": " << module_bytes.size() << ",\n"
            << "  \"module_sha256\": \"" << kModuleSha256 << "\",\n"
            << "  \"raw_parameter_sha256\": \"" << kRawParameterSha256 << "\",\n"
            << "  \"address_free_parameter_sha256\": \""
            << kAddressFreeParameterSha256 << "\",\n"
            << "  \"packed_parameter_bytes\": 240,\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"input_hashes\": {\"rgba_a\": \"" << Sha256(actual_rgba_a)
            << "\", \"rgba_b\": \"" << Sha256(actual_rgba_b)
            << "\", \"motion\": \"" << Sha256(actual_motion)
            << "\", \"blend\": \"" << Sha256(actual_blend)
            << "\", \"candidate_a\": \"" << Sha256(inputs.candidate_a)
            << "\", \"candidate_b\": \"" << Sha256(inputs.candidate_b)
            << "\"},\n"
            << "  \"output_hashes\": {\"surface_a\": \"" << Sha256(actual_surface_a)
            << "\", \"surface_b\": \"" << Sha256(actual_surface_b)
            << "\", \"surface_mask\": \"" << Sha256(actual_surface_mask)
            << "\", \"linear_a\": \"" << Sha256(actual_linear_a)
            << "\", \"linear_b\": \"" << Sha256(actual_linear_b)
            << "\", \"linear_mask\": \"" << Sha256(actual_linear_mask)
            << "\"},\n"
            << "  \"combined_output_bytes\": " << combined.size() << ",\n"
            << "  \"combined_output_sha256\": \"" << Sha256(combined) << "\",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
            << "  \"poison_pattern_output_words\": "
            << poison_pattern_words << ",\n"
            << "  \"external_expected_tensors_consumed\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n}\n";
    std::cout << "probe=fg_mid_l55_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "combined_output_sha256=" << Sha256(combined) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "poison_pattern_output_words=" << poison_pattern_words << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
