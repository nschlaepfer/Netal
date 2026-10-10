#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <d3d12.h>
#include <wrl/client.h>

#include "runtime_allocation_capture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace neuralstudio::fg {

namespace {

RuntimeAllocationCapture* g_active_capture = nullptr;
SRWLOCK g_active_capture_lock = SRWLOCK_INIT;
volatile LONG64 g_next_allocation_token = 0;

class ExclusiveLock final {
 public:
  explicit ExclusiveLock(SRWLOCK* lock) : lock_(lock) {
    AcquireSRWLockExclusive(lock_);
  }
  ~ExclusiveLock() { ReleaseSRWLockExclusive(lock_); }
  ExclusiveLock(const ExclusiveLock&) = delete;
  ExclusiveLock& operator=(const ExclusiveLock&) = delete;

 private:
  SRWLOCK* lock_;
};

std::uint64_t NextToken() {
  return static_cast<std::uint64_t>(
      InterlockedIncrement64(&g_next_allocation_token));
}

std::string Sha256(const void* data, std::size_t size) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_size = 0;
  DWORD hash_size = 0;
  DWORD copied = 0;
  std::vector<std::uint8_t> object;
  std::vector<std::uint8_t> digest;
  bool success = false;
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                  0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                        reinterpret_cast<PUCHAR>(&object_size),
                        sizeof(object_size), &copied, 0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                        reinterpret_cast<PUCHAR>(&hash_size), sizeof(hash_size),
                        &copied, 0) < 0 ||
      hash_size != 32u) {
    goto cleanup;
  }
  object.resize(object_size);
  digest.resize(hash_size);
  if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0,
                       0) < 0) {
    goto cleanup;
  }
  {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t remaining = size;
    while (remaining != 0) {
      const ULONG chunk = static_cast<ULONG>((std::min<std::size_t>)(
          remaining, (std::numeric_limits<ULONG>::max)()));
      if (BCryptHashData(hash, const_cast<PUCHAR>(cursor), chunk, 0) < 0) {
        goto cleanup;
      }
      cursor += chunk;
      remaining -= chunk;
    }
  }
  if (BCryptFinishHash(hash, digest.data(), hash_size, 0) < 0) goto cleanup;
  success = true;

cleanup:
  if (hash != nullptr) BCryptDestroyHash(hash);
  if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
  if (!success) throw std::runtime_error("SHA-256 calculation failed");
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    encoded << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return encoded.str();
}

void WriteAll(HANDLE file, const void* data, std::size_t size) {
  const auto* cursor = static_cast<const std::uint8_t*>(data);
  std::size_t remaining = size;
  while (remaining != 0) {
    const DWORD chunk = static_cast<DWORD>((std::min<std::size_t>)(
        remaining, (std::numeric_limits<DWORD>::max)()));
    DWORD written = 0;
    if (WriteFile(file, cursor, chunk, &written, nullptr) == FALSE ||
        written != chunk) {
      throw std::runtime_error("immutable upload dump write failed");
    }
    cursor += chunk;
    remaining -= chunk;
  }
}

void WriteImmutableFile(const std::filesystem::path& path, const void* data,
                        std::size_t size) {
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("immutable upload dump collision or open failure");
  }
  bool success = false;
  try {
    WriteAll(file, data, size);
    if (FlushFileBuffers(file) == FALSE) {
      throw std::runtime_error("immutable upload dump flush failed");
    }
    success = true;
  } catch (...) {
    CloseHandle(file);
    throw;
  }
  if (CloseHandle(file) == FALSE || !success ||
      SetFileAttributesW(path.c_str(), FILE_ATTRIBUTE_READONLY) == FALSE) {
    throw std::runtime_error("immutable upload dump close/seal failed");
  }
}

std::string JsonLineForAllocation(
    std::uint64_t token, const D3D12_RESOURCE_DESC& description,
    const D3D12_HEAP_PROPERTIES& heap, int initial_state,
    const D3D12_RESOURCE_ALLOCATION_INFO& info) {
  char line[2048]{};
  const int length = sprintf_s(
      line,
      "{\"event\":\"allocation\",\"token\":%llu,"
      "\"dimension\":%u,\"alignment\":%llu,\"width\":%llu,"
      "\"height\":%u,\"depth_or_array\":%u,\"mip_levels\":%u,"
      "\"format\":%u,\"sample_count\":%u,\"sample_quality\":%u,"
      "\"layout\":%u,\"resource_flags\":%u,\"heap_type\":%u,"
      "\"heap_flags\":0,"
      "\"cpu_page_property\":%u,\"memory_pool_preference\":%u,"
      "\"creation_node_mask\":%u,\"visible_node_mask\":%u,"
      "\"initial_state\":%u,\"allocation_size\":%llu,"
      "\"allocation_alignment\":%llu}\n",
      static_cast<unsigned long long>(token),
      static_cast<unsigned int>(description.Dimension),
      static_cast<unsigned long long>(description.Alignment),
      static_cast<unsigned long long>(description.Width), description.Height,
      static_cast<unsigned int>(description.DepthOrArraySize),
      static_cast<unsigned int>(description.MipLevels),
      static_cast<unsigned int>(description.Format),
      description.SampleDesc.Count, description.SampleDesc.Quality,
      static_cast<unsigned int>(description.Layout),
      static_cast<unsigned int>(description.Flags),
      static_cast<unsigned int>(heap.Type),
      static_cast<unsigned int>(heap.CPUPageProperty),
      static_cast<unsigned int>(heap.MemoryPoolPreference),
      heap.CreationNodeMask, heap.VisibleNodeMask,
      static_cast<unsigned int>(initial_state),
      static_cast<unsigned long long>(info.SizeInBytes),
      static_cast<unsigned long long>(info.Alignment));
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    return {};
  }
  return std::string(line, static_cast<std::size_t>(length));
}

std::string JsonLineForVolatileAllocation(std::uint64_t token,
                                          D3D12_GPU_VIRTUAL_ADDRESS address,
                                          std::uint64_t size) {
  char line[384]{};
  const int length = sprintf_s(
      line,
      "{\"event\":\"allocation\",\"token\":%llu,"
      "\"gpu_virtual_address\":%llu,\"allocation_size\":%llu}\n",
      static_cast<unsigned long long>(token),
      static_cast<unsigned long long>(address),
      static_cast<unsigned long long>(size));
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    return {};
  }
  return std::string(line, static_cast<std::size_t>(length));
}

std::string JsonLineForTokenEvent(const char* event, std::uint64_t token) {
  char line[192]{};
  const int length = sprintf_s(
      line, "{\"event\":\"%s\",\"token\":%llu}\n", event,
      static_cast<unsigned long long>(token));
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    return {};
  }
  return std::string(line, static_cast<std::size_t>(length));
}

}  // namespace

struct RuntimeAllocationCapture::Impl {
  struct Record {
    IUnknown* identity = nullptr;
    ID3D12Resource* resource = nullptr;
    std::uint64_t token = 0;
    D3D12_RESOURCE_DESC description{};
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_RESOURCE_ALLOCATION_INFO allocation_info{};
    int initial_state = 0;
    bool released = false;
  };
  struct DefaultReadback {
    std::uint64_t source_token = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t mapped_bytes = 0;
    D3D12_RESOURCE_DESC source_description{};
    D3D12_HEAP_PROPERTIES source_heap{};
    int source_initial_state = 0;
    bool texture = false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    std::uint32_t row_count = 0;
    std::uint64_t row_bytes = 0;
    ComPtr<ID3D12Resource> resource;
  };

  ComPtr<ID3D12Device> device;
  HANDLE semantic = INVALID_HANDLE_VALUE;
  HANDLE volatile_addresses = INVALID_HANDLE_VALUE;
  SRWLOCK lock{};
  std::vector<Record> records;
  std::size_t releases = 0;
  std::size_t leaks = 0;
  std::size_t upload_dumps = 0;
  std::uint64_t upload_bytes = 0;
  bool upload_capture_completed = false;
  std::vector<DefaultReadback> default_readbacks;
  std::size_t default_dumps = 0;
  std::uint64_t default_bytes = 0;
  bool default_readbacks_prepared = false;
  bool default_capture_completed = false;
  bool error = false;
  bool finalized = false;

  bool Write(HANDLE file, const std::string& line) {
    if (file == INVALID_HANDLE_VALUE || line.empty() ||
        line.size() > static_cast<std::size_t>(MAXDWORD)) {
      error = true;
      return false;
    }
    DWORD written = 0;
    const bool success =
        WriteFile(file, line.data(), static_cast<DWORD>(line.size()), &written,
                  nullptr) != FALSE &&
        written == static_cast<DWORD>(line.size()) &&
        FlushFileBuffers(file) != FALSE;
    if (!success) error = true;
    return success;
  }

  void RecordError(const char* reason) {
    error = true;
    char line[256]{};
    const int length = sprintf_s(
        line, "{\"event\":\"error\",\"reason\":\"%s\"}\n", reason);
    if (length > 0 && static_cast<std::size_t>(length) < sizeof(line)) {
      (void)Write(semantic,
                  std::string(line, static_cast<std::size_t>(length)));
    }
  }
};

RuntimeAllocationCapture::RuntimeAllocationCapture(
    ID3D12Device* device, const std::filesystem::path& semantic_ledger,
    const std::filesystem::path& volatile_ledger)
    : impl_(std::make_unique<Impl>()) {
  if (device == nullptr) {
    throw std::invalid_argument("runtime allocation capture requires a device");
  }
  impl_->device = device;
  impl_->semantic = CreateFileW(
      semantic_ledger.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
      CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl_->semantic == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("creating semantic allocation ledger failed");
  }
  impl_->volatile_addresses = CreateFileW(
      volatile_ledger.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
      CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl_->volatile_addresses == INVALID_HANDLE_VALUE) {
    CloseHandle(impl_->semantic);
    impl_->semantic = INVALID_HANDLE_VALUE;
    throw std::runtime_error("creating volatile allocation ledger failed");
  }

  AcquireSRWLockExclusive(&g_active_capture_lock);
  if (g_active_capture != nullptr) {
    ReleaseSRWLockExclusive(&g_active_capture_lock);
    CloseHandle(impl_->volatile_addresses);
    CloseHandle(impl_->semantic);
    impl_->volatile_addresses = INVALID_HANDLE_VALUE;
    impl_->semantic = INVALID_HANDLE_VALUE;
    throw std::runtime_error("runtime allocation capture is already active");
  }
  g_active_capture = this;
  ReleaseSRWLockExclusive(&g_active_capture_lock);
}

RuntimeAllocationCapture::~RuntimeAllocationCapture() {
  if (impl_ && !impl_->finalized) Finalize();
}

PFN_NVSDK_NGX_D3D12_ResourceAllocCallback
RuntimeAllocationCapture::allocation_callback() {
  return &Allocate;
}

PFN_NVSDK_NGX_ResourceReleaseCallback
RuntimeAllocationCapture::release_callback() {
  return &Release;
}

void NVSDK_CONV RuntimeAllocationCapture::Allocate(
    D3D12_RESOURCE_DESC* description, int initial_state,
    CD3DX12_HEAP_PROPERTIES* heap, ID3D12Resource** output_resource) {
  if (output_resource != nullptr) *output_resource = nullptr;
  AcquireSRWLockShared(&g_active_capture_lock);
  RuntimeAllocationCapture* capture = g_active_capture;
  if (capture == nullptr || !capture->impl_) {
    ReleaseSRWLockShared(&g_active_capture_lock);
    return;
  }
  Impl* impl = capture->impl_.get();
  if (description == nullptr || heap == nullptr || output_resource == nullptr) {
    AcquireSRWLockExclusive(&impl->lock);
    impl->RecordError("invalid_allocation_callback_argument");
    ReleaseSRWLockExclusive(&impl->lock);
    ReleaseSRWLockShared(&g_active_capture_lock);
    return;
  }
  ComPtr<ID3D12Device> device = impl->device;

  // CD3DX12_HEAP_PROPERTIES publicly derives from D3D12_HEAP_PROPERTIES and
  // adds no storage; the NGX callback ABI passes it by pointer.
  const auto* heap_properties =
      reinterpret_cast<const D3D12_HEAP_PROPERTIES*>(heap);
  const D3D12_RESOURCE_ALLOCATION_INFO allocation_info =
      device->GetResourceAllocationInfo(0, 1, description);
  ID3D12Resource* resource = nullptr;
  const HRESULT result = device->CreateCommittedResource(
      heap_properties, D3D12_HEAP_FLAG_NONE, description,
      static_cast<D3D12_RESOURCE_STATES>(initial_state), nullptr,
      IID_PPV_ARGS(&resource));

  AcquireSRWLockExclusive(&impl->lock);
  if (FAILED(result) || resource == nullptr) {
    impl->RecordError("CreateCommittedResource_failed");
    ReleaseSRWLockExclusive(&impl->lock);
    ReleaseSRWLockShared(&g_active_capture_lock);
    return;
  }
  const std::uint64_t token = NextToken();
  const D3D12_GPU_VIRTUAL_ADDRESS gpu_address =
      resource->GetGPUVirtualAddress();
  impl->records.push_back({static_cast<IUnknown*>(resource), resource, token,
                           *description, *heap_properties, allocation_info,
                           initial_state, false});
  const bool semantic_written = impl->Write(
      impl->semantic,
      JsonLineForAllocation(token, *description, *heap_properties,
                            initial_state, allocation_info));
  const bool volatile_written = impl->Write(
      impl->volatile_addresses,
      JsonLineForVolatileAllocation(token, gpu_address,
                                    allocation_info.SizeInBytes));
  if (!semantic_written || !volatile_written) {
    impl->RecordError("allocation_ledger_write_failed");
    impl->records.back().released = true;
    resource->Release();
    ReleaseSRWLockExclusive(&impl->lock);
    ReleaseSRWLockShared(&g_active_capture_lock);
    return;
  }
  *output_resource = resource;
  ReleaseSRWLockExclusive(&impl->lock);
  ReleaseSRWLockShared(&g_active_capture_lock);
}

void NVSDK_CONV RuntimeAllocationCapture::Release(IUnknown* resource) {
  AcquireSRWLockShared(&g_active_capture_lock);
  RuntimeAllocationCapture* capture = g_active_capture;
  if (capture == nullptr || !capture->impl_) {
    ReleaseSRWLockShared(&g_active_capture_lock);
    return;
  }
  Impl* impl = capture->impl_.get();
  if (resource == nullptr) {
    AcquireSRWLockExclusive(&impl->lock);
    impl->RecordError("null_release_callback_argument");
    ReleaseSRWLockExclusive(&impl->lock);
    ReleaseSRWLockShared(&g_active_capture_lock);
    return;
  }

  ID3D12Resource* release_target = nullptr;
  AcquireSRWLockExclusive(&impl->lock);
  const auto found = std::find_if(
      impl->records.begin(), impl->records.end(),
      [resource](const Impl::Record& record) {
        return record.identity == resource;
      });
  if (found == impl->records.end()) {
    impl->RecordError("release_of_unknown_resource");
  } else if (found->released) {
    impl->RecordError("double_release");
  } else {
    found->released = true;
    release_target = found->resource;
    ++impl->releases;
    (void)impl->Write(impl->semantic,
                      JsonLineForTokenEvent("release", found->token));
    (void)impl->Write(impl->volatile_addresses,
                      JsonLineForTokenEvent("release", found->token));
  }
  ReleaseSRWLockExclusive(&impl->lock);
  if (release_target != nullptr) release_target->Release();
  ReleaseSRWLockShared(&g_active_capture_lock);
}

void RuntimeAllocationCapture::CaptureLiveCpuVisibleBuffers(
    const std::filesystem::path& dump_directory,
    const std::filesystem::path& semantic_manifest) {
  if (!impl_) throw std::runtime_error("allocation capture is unavailable");
  ExclusiveLock lock(&impl_->lock);
  try {
    if (impl_->finalized || impl_->upload_capture_completed) {
      throw std::runtime_error(
          "runtime upload capture was finalized or already executed");
    }
    const std::size_t eligible = static_cast<std::size_t>(std::count_if(
        impl_->records.begin(), impl_->records.end(),
        [](const Impl::Record& record) {
          return !record.released &&
                 ClassifyCpuVisibleBuffer(record.description, record.heap) !=
                     CpuVisibleBufferClass::kNotSafelyReadable;
        }));
    if (eligible == 0) {
      throw std::runtime_error("runtime upload capture found no live buffers");
    }
    if (CreateDirectoryW(dump_directory.c_str(), nullptr) == FALSE) {
      throw std::runtime_error(
          "runtime upload dump directory collision or creation failure");
    }

    std::ostringstream manifest;
    manifest << "{\n"
             << "  \"schema\": \"neuralstudio.dlss_fg.runtime_uploads.v1\",\n"
             << "  \"capture_point\": \"post_create_fence_pre_evaluate\",\n"
             << "  \"default_heap_read\": false,\n"
             << "  \"privacy\": \"private_local_evidence\",\n"
             << "  \"entries\": [\n";
    std::size_t captured = 0;
    std::uint64_t captured_bytes = 0;
    for (Impl::Record& record : impl_->records) {
      const CpuVisibleBufferClass buffer_class =
          ClassifyCpuVisibleBuffer(record.description, record.heap);
      if (record.released ||
          buffer_class == CpuVisibleBufferClass::kNotSafelyReadable) {
        continue;
      }
      if (record.description.Width >
          static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)())) {
        throw std::runtime_error("runtime upload logical size is not addressable");
      }
      const std::size_t logical_bytes =
          static_cast<std::size_t>(record.description.Width);
      if (logical_bytes == 0 ||
          captured_bytes >
              (std::numeric_limits<std::uint64_t>::max)() - logical_bytes) {
        throw std::runtime_error("runtime upload byte count is invalid");
      }
      std::vector<std::uint8_t> bytes(logical_bytes);
      void* mapped = nullptr;
      const D3D12_RANGE read_range{0, static_cast<SIZE_T>(logical_bytes)};
      const HRESULT map_result = record.resource->Map(0, &read_range, &mapped);
      if (FAILED(map_result) || mapped == nullptr) {
        throw std::runtime_error("mapping a live CPU-visible buffer failed");
      }
      std::memcpy(bytes.data(), mapped, logical_bytes);
      const D3D12_RANGE no_write{0, 0};
      record.resource->Unmap(0, &no_write);

      const std::string sha256 = Sha256(bytes.data(), bytes.size());
      char file_name[96]{};
      if (sprintf_s(file_name, "upload-token-%06llu.bin",
                    static_cast<unsigned long long>(record.token)) <= 0) {
        throw std::runtime_error("formatting upload dump name failed");
      }
      const std::filesystem::path dump_path = dump_directory / file_name;
      WriteImmutableFile(dump_path, bytes.data(), bytes.size());

      if (captured != 0) manifest << ",\n";
      manifest << "    {\"token\": " << record.token
               << ", \"dump_file\": \"" << file_name
               << "\", \"heap_class\": \""
               << CpuVisibleBufferClassName(buffer_class)
               << "\", \"heap_type\": "
               << static_cast<unsigned int>(record.heap.Type)
               << ", \"cpu_page_property\": "
               << static_cast<unsigned int>(record.heap.CPUPageProperty)
               << ", \"memory_pool_preference\": "
               << static_cast<unsigned int>(record.heap.MemoryPoolPreference)
               << ", \"descriptor_dimension\": "
               << static_cast<unsigned int>(record.description.Dimension)
               << ", \"descriptor_width\": " << record.description.Width
               << ", \"initial_state\": "
               << static_cast<unsigned int>(record.initial_state)
               << ", \"logical_bytes\": " << logical_bytes
               << ", \"allocation_bytes\": "
               << record.allocation_info.SizeInBytes
               << ", \"sha256\": \"" << sha256 << "\"}";
      ++captured;
      captured_bytes += logical_bytes;
    }
    manifest << "\n  ],\n"
             << "  \"captured_buffers\": " << captured << ",\n"
             << "  \"captured_logical_bytes\": " << captured_bytes << "\n"
             << "}\n";
    const std::string manifest_text = manifest.str();
    WriteImmutableFile(semantic_manifest, manifest_text.data(),
                       manifest_text.size());
    impl_->upload_dumps = captured;
    impl_->upload_bytes = captured_bytes;
    impl_->upload_capture_completed = true;
  } catch (...) {
    impl_->RecordError("runtime_upload_capture_failed");
    throw;
  }
}

void RuntimeAllocationCapture::PrepareDefaultBufferReadbacks(
    const std::vector<std::uint64_t>& allocation_tokens,
    ID3D12GraphicsCommandList* command_list) {
  if (!impl_ || command_list == nullptr || allocation_tokens.empty()) {
    throw std::invalid_argument(
        "default-buffer readback preparation requires tokens and command list");
  }
  ExclusiveLock lock(&impl_->lock);
  try {
    if (impl_->finalized || impl_->default_readbacks_prepared ||
        impl_->default_capture_completed) {
      throw std::runtime_error(
          "default-buffer readback was finalized or already prepared");
    }
    std::vector<std::uint64_t> sorted_tokens = allocation_tokens;
    std::sort(sorted_tokens.begin(), sorted_tokens.end());
    if (std::adjacent_find(sorted_tokens.begin(), sorted_tokens.end()) !=
        sorted_tokens.end()) {
      throw std::invalid_argument("default-buffer token list contains duplicates");
    }

    for (const std::uint64_t token : allocation_tokens) {
      const auto found = std::find_if(
          impl_->records.begin(), impl_->records.end(),
          [token](const Impl::Record& record) {
            return record.token == token;
          });
      const bool is_buffer = found != impl_->records.end() && !found->released &&
          IsImplicitCopySourceDefaultBuffer(
              found->description, found->heap, found->initial_state);
      const bool is_texture = found != impl_->records.end() && !found->released &&
          found->heap.Type == D3D12_HEAP_TYPE_DEFAULT &&
          found->description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
          found->description.DepthOrArraySize == 1 &&
          found->description.MipLevels == 1 &&
          found->description.SampleDesc.Count == 1;
      if (!is_buffer && !is_texture) {
        throw std::runtime_error(
            "requested token is not a supported live DEFAULT resource");
      }

      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      UINT row_count = 0;
      UINT64 row_bytes = 0;
      UINT64 mapped_bytes = found->description.Width;
      UINT64 logical_bytes = found->description.Width;
      if (is_texture) {
        impl_->device->GetCopyableFootprints(
            &found->description, 0, 1, 0, &footprint, &row_count,
            &row_bytes, &mapped_bytes);
        if (row_count == 0 || row_bytes == 0 || mapped_bytes == 0 ||
            row_bytes > (std::numeric_limits<UINT64>::max)() / row_count) {
          throw std::runtime_error("DEFAULT texture footprint is invalid");
        }
        logical_bytes = row_bytes * row_count;
      }

      D3D12_HEAP_PROPERTIES readback_heap{};
      readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
      readback_heap.CreationNodeMask = 1;
      readback_heap.VisibleNodeMask = 1;
      D3D12_RESOURCE_DESC readback_description{};
      readback_description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      readback_description.Width = mapped_bytes;
      readback_description.Height = 1;
      readback_description.DepthOrArraySize = 1;
      readback_description.MipLevels = 1;
      readback_description.SampleDesc.Count = 1;
      readback_description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      ComPtr<ID3D12Resource> readback;
      const HRESULT result = impl_->device->CreateCommittedResource(
          &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_description,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
      if (FAILED(result) || readback == nullptr) {
        throw std::runtime_error(
            "creating a DEFAULT-buffer readback destination failed");
      }

      if (is_texture) {
        D3D12_RESOURCE_BARRIER to_copy{};
        to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_copy.Transition.pResource = found->resource;
        to_copy.Transition.Subresource = 0;
        to_copy.Transition.StateBefore =
            static_cast<D3D12_RESOURCE_STATES>(found->initial_state);
        to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        command_list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = found->resource;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;
        command_list->CopyTextureRegion(&destination, 0, 0, 0, &source,
                                        nullptr);
        std::swap(to_copy.Transition.StateBefore,
                  to_copy.Transition.StateAfter);
        command_list->ResourceBarrier(1, &to_copy);
      } else {
        // Deliberately no transition: the source was created in COMMON, and
        // buffers support implicit promotion to COPY_SOURCE and decay after
        // this dedicated ExecuteCommandLists submission.
        command_list->CopyBufferRegion(readback.Get(), 0, found->resource, 0,
                                       found->description.Width);
      }
      impl_->default_readbacks.push_back(
          {found->token, logical_bytes, mapped_bytes, found->description,
           found->heap, found->initial_state, is_texture, footprint,
           row_count, row_bytes, std::move(readback)});
    }
    impl_->default_readbacks_prepared = true;
  } catch (...) {
    impl_->RecordError("default_buffer_readback_prepare_failed");
    throw;
  }
}

void RuntimeAllocationCapture::CompleteDefaultBufferReadbacks(
    const std::filesystem::path& dump_directory,
    const std::filesystem::path& semantic_manifest,
    std::string_view capture_point) {
  if (!impl_) throw std::runtime_error("allocation capture is unavailable");
  ExclusiveLock lock(&impl_->lock);
  try {
    if (impl_->finalized || !impl_->default_readbacks_prepared ||
        impl_->default_capture_completed || impl_->default_readbacks.empty()) {
      throw std::runtime_error(
          "default-buffer readback completion has no prepared batch");
    }
    if (capture_point != "post_create_fence_pre_evaluate" &&
        capture_point != "post_evaluate_0_fence_pre_evaluate_1" &&
        capture_point != "post_evaluate_1_fence_pre_evaluate_2") {
      throw std::invalid_argument("unknown DEFAULT-buffer capture point");
    }
    if (CreateDirectoryW(dump_directory.c_str(), nullptr) == FALSE) {
      throw std::runtime_error(
          "default-buffer dump directory collision or creation failure");
    }

    const bool has_texture = std::any_of(
        impl_->default_readbacks.begin(), impl_->default_readbacks.end(),
        [](const Impl::DefaultReadback& pending) { return pending.texture; });
    std::ostringstream manifest;
    manifest << "{\n"
             << "  \"schema\": \""
             << (has_texture
                     ? "neuralstudio.dlss_fg.default_resource_readback.v2"
                     : "neuralstudio.dlss_fg.default_buffer_readback.v1")
             << "\",\n"
             << "  \"capture_point\": \"" << capture_point << "\",\n"
             << "  \"source_heap\": \"default\",\n"
             << "  \"source_state_strategy\": \""
             << (has_texture
                     ? "texture_original_to_copy_source_and_restore"
                     : "implicit_common_to_copy_source_promotion_and_decay")
             << "\",\n"
             << "  \"explicit_source_transition\": "
             << (has_texture ? "true" : "false") << ",\n"
             << "  \"direct_default_map\": false,\n"
             << "  \"copy_submission\": \"dedicated\",\n"
             << "  \"bytes_fed_back_to_execution\": false,\n"
             << "  \"privacy\": \"private_local_evidence\",\n"
             << "  \"entries\": [\n";
    std::size_t captured = 0;
    std::uint64_t captured_bytes = 0;
    for (Impl::DefaultReadback& pending : impl_->default_readbacks) {
      if (pending.logical_bytes == 0 ||
          pending.logical_bytes >
              static_cast<std::uint64_t>(
                  (std::numeric_limits<std::size_t>::max)()) ||
          captured_bytes > (std::numeric_limits<std::uint64_t>::max)() -
                               pending.logical_bytes) {
        throw std::runtime_error("default-buffer logical size is invalid");
      }
      const std::size_t logical_bytes =
          static_cast<std::size_t>(pending.logical_bytes);
      std::vector<std::uint8_t> bytes(logical_bytes);
      void* mapped = nullptr;
      if (pending.mapped_bytes == 0 ||
          pending.mapped_bytes > static_cast<std::uint64_t>(
              (std::numeric_limits<SIZE_T>::max)())) {
        throw std::runtime_error("DEFAULT readback mapped size is invalid");
      }
      const D3D12_RANGE read_range{
          0, static_cast<SIZE_T>(pending.mapped_bytes)};
      const HRESULT map_result = pending.resource->Map(0, &read_range, &mapped);
      if (FAILED(map_result) || mapped == nullptr) {
        throw std::runtime_error("mapping DEFAULT-buffer readback failed");
      }
      if (pending.texture) {
        if (pending.row_count == 0 || pending.row_bytes == 0 ||
            pending.footprint.Footprint.RowPitch < pending.row_bytes) {
          pending.resource->Unmap(0, nullptr);
          throw std::runtime_error("DEFAULT texture row metadata is invalid");
        }
        const auto* source = static_cast<const std::uint8_t*>(mapped) +
                             pending.footprint.Offset;
        for (std::uint32_t row = 0; row < pending.row_count; ++row) {
          std::memcpy(bytes.data() + std::size_t{row} * pending.row_bytes,
                      source + std::size_t{row} *
                                   pending.footprint.Footprint.RowPitch,
                      static_cast<std::size_t>(pending.row_bytes));
        }
      } else {
        std::memcpy(bytes.data(), mapped, logical_bytes);
      }
      const D3D12_RANGE no_write{0, 0};
      pending.resource->Unmap(0, &no_write);

      const std::string sha256 = Sha256(bytes.data(), bytes.size());
      char file_name[96]{};
      if (sprintf_s(file_name, "default-token-%06llu.bin",
                    static_cast<unsigned long long>(pending.source_token)) <=
          0) {
        throw std::runtime_error("formatting DEFAULT-buffer dump name failed");
      }
      WriteImmutableFile(dump_directory / file_name, bytes.data(), bytes.size());
      if (captured != 0) manifest << ",\n";
      manifest << "    {\"source_token\": " << pending.source_token
               << ", \"dump_file\": \"" << file_name
               << "\", \"descriptor_dimension\": "
               << static_cast<unsigned int>(pending.source_description.Dimension)
               << ", \"descriptor_width\": "
               << pending.source_description.Width
               << ", \"source_heap_type\": "
               << static_cast<unsigned int>(pending.source_heap.Type)
               << ", \"source_initial_state\": "
               << static_cast<unsigned int>(pending.source_initial_state)
               << ", \"logical_bytes\": " << logical_bytes
               << ", \"resource_kind\": \""
               << (pending.texture ? "texture2d" : "buffer") << "\""
               << (pending.texture
                       ? ", \"row_bytes\": " +
                             std::to_string(pending.row_bytes) +
                             ", \"row_count\": " +
                             std::to_string(pending.row_count) +
                             ", \"readback_row_pitch\": " +
                             std::to_string(
                                 pending.footprint.Footprint.RowPitch)
                       : "")
               << ", \"sha256\": \"" << sha256 << "\"}";
      ++captured;
      captured_bytes += pending.logical_bytes;
    }
    manifest << "\n  ],\n"
             << "  \""
             << (has_texture ? "captured_resources" : "captured_buffers")
             << "\": " << captured << ",\n"
             << "  \"captured_logical_bytes\": " << captured_bytes << "\n"
             << "}\n";
    const std::string manifest_text = manifest.str();
    WriteImmutableFile(semantic_manifest, manifest_text.data(),
                       manifest_text.size());
    impl_->default_dumps = captured;
    impl_->default_bytes = captured_bytes;
    impl_->default_capture_completed = true;
  } catch (...) {
    impl_->RecordError("default_buffer_readback_complete_failed");
    throw;
  }
}

void RuntimeAllocationCapture::Finalize() {
  if (!impl_ || impl_->finalized) return;
  AcquireSRWLockExclusive(&g_active_capture_lock);
  if (g_active_capture == this) g_active_capture = nullptr;
  ReleaseSRWLockExclusive(&g_active_capture_lock);

  AcquireSRWLockExclusive(&impl_->lock);
  impl_->leaks = static_cast<std::size_t>(std::count_if(
      impl_->records.begin(), impl_->records.end(),
      [](const Impl::Record& record) { return !record.released; }));
  if (impl_->leaks != 0 || impl_->releases != impl_->records.size()) {
    impl_->RecordError("allocation_leak_or_release_count_mismatch");
  }
  if (impl_->default_readbacks_prepared &&
      !impl_->default_capture_completed) {
    impl_->RecordError("default_buffer_readback_incomplete");
  }
  char summary[384]{};
  const int length = sprintf_s(
      summary,
      "{\"event\":\"summary\",\"allocations\":%llu,"
      "\"releases\":%llu,\"leaks\":%llu,\"healthy\":%s}\n",
      static_cast<unsigned long long>(impl_->records.size()),
      static_cast<unsigned long long>(impl_->releases),
      static_cast<unsigned long long>(impl_->leaks),
      impl_->error ? "false" : "true");
  if (length > 0 && static_cast<std::size_t>(length) < sizeof(summary)) {
    (void)impl_->Write(
        impl_->semantic,
        std::string(summary, static_cast<std::size_t>(length)));
  } else {
    impl_->error = true;
  }
  if (impl_->semantic != INVALID_HANDLE_VALUE) {
    const bool flushed = FlushFileBuffers(impl_->semantic) != FALSE;
    const bool closed = CloseHandle(impl_->semantic) != FALSE;
    if (!flushed || !closed) {
      impl_->error = true;
    }
    impl_->semantic = INVALID_HANDLE_VALUE;
  }
  if (impl_->volatile_addresses != INVALID_HANDLE_VALUE) {
    const bool flushed =
        FlushFileBuffers(impl_->volatile_addresses) != FALSE;
    const bool closed = CloseHandle(impl_->volatile_addresses) != FALSE;
    if (!flushed || !closed) {
      impl_->error = true;
    }
    impl_->volatile_addresses = INVALID_HANDLE_VALUE;
  }
  impl_->finalized = true;
  ReleaseSRWLockExclusive(&impl_->lock);
}

bool RuntimeAllocationCapture::healthy() const {
  return impl_ && impl_->finalized && !impl_->error && impl_->leaks == 0 &&
         impl_->releases == impl_->records.size();
}

std::size_t RuntimeAllocationCapture::allocation_count() const {
  return impl_ ? impl_->records.size() : 0;
}

std::size_t RuntimeAllocationCapture::release_count() const {
  return impl_ ? impl_->releases : 0;
}

std::size_t RuntimeAllocationCapture::leak_count() const {
  return impl_ ? impl_->leaks : 0;
}

std::size_t RuntimeAllocationCapture::upload_dump_count() const {
  return impl_ ? impl_->upload_dumps : 0;
}

std::uint64_t RuntimeAllocationCapture::upload_dump_bytes() const {
  return impl_ ? impl_->upload_bytes : 0;
}

std::size_t RuntimeAllocationCapture::default_dump_count() const {
  return impl_ ? impl_->default_dumps : 0;
}

std::uint64_t RuntimeAllocationCapture::default_dump_bytes() const {
  return impl_ ? impl_->default_bytes : 0;
}

}  // namespace neuralstudio::fg
