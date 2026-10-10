#pragma once

#include <windows.h>
#include <d3d12.h>

#include <nvsdk_ngx_defs.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace neuralstudio::fg {

enum class CpuVisibleBufferClass : std::uint32_t {
  kNotSafelyReadable = 0,
  kUpload = 1,
  kReadback = 2,
  kCustomWriteCombine = 3,
  kCustomWriteBack = 4,
};

inline CpuVisibleBufferClass ClassifyCpuVisibleBuffer(
    const D3D12_RESOURCE_DESC& description,
    const D3D12_HEAP_PROPERTIES& heap) noexcept {
  if (description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
      description.Width == 0) {
    return CpuVisibleBufferClass::kNotSafelyReadable;
  }
  if (heap.Type == D3D12_HEAP_TYPE_UPLOAD) {
    return CpuVisibleBufferClass::kUpload;
  }
  if (heap.Type == D3D12_HEAP_TYPE_READBACK) {
    return CpuVisibleBufferClass::kReadback;
  }
  if (heap.Type == D3D12_HEAP_TYPE_CUSTOM &&
      heap.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE) {
    return CpuVisibleBufferClass::kCustomWriteCombine;
  }
  if (heap.Type == D3D12_HEAP_TYPE_CUSTOM &&
      heap.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK) {
    return CpuVisibleBufferClass::kCustomWriteBack;
  }
  return CpuVisibleBufferClass::kNotSafelyReadable;
}

inline const char* CpuVisibleBufferClassName(
    CpuVisibleBufferClass buffer_class) noexcept {
  switch (buffer_class) {
    case CpuVisibleBufferClass::kUpload: return "upload";
    case CpuVisibleBufferClass::kReadback: return "readback";
    case CpuVisibleBufferClass::kCustomWriteCombine:
      return "custom_write_combine";
    case CpuVisibleBufferClass::kCustomWriteBack:
      return "custom_write_back";
    case CpuVisibleBufferClass::kNotSafelyReadable:
    default: return "not_safely_readable";
  }
}

inline bool IsImplicitCopySourceDefaultBuffer(
    const D3D12_RESOURCE_DESC& description,
    const D3D12_HEAP_PROPERTIES& heap, int initial_state) noexcept {
  return description.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
         description.Width != 0 && heap.Type == D3D12_HEAP_TYPE_DEFAULT &&
         initial_state == static_cast<int>(D3D12_RESOURCE_STATE_COMMON);
}

class RuntimeAllocationCapture final {
 public:
  RuntimeAllocationCapture(ID3D12Device* device,
                           const std::filesystem::path& semantic_ledger,
                           const std::filesystem::path& volatile_ledger);
  ~RuntimeAllocationCapture();

  RuntimeAllocationCapture(const RuntimeAllocationCapture&) = delete;
  RuntimeAllocationCapture& operator=(const RuntimeAllocationCapture&) =
      delete;

  static PFN_NVSDK_NGX_D3D12_ResourceAllocCallback allocation_callback();
  static PFN_NVSDK_NGX_ResourceReleaseCallback release_callback();

  void CaptureLiveCpuVisibleBuffers(
      const std::filesystem::path& dump_directory,
      const std::filesystem::path& semantic_manifest);
  void PrepareDefaultBufferReadbacks(
      const std::vector<std::uint64_t>& allocation_tokens,
      ID3D12GraphicsCommandList* command_list);
  void CompleteDefaultBufferReadbacks(
      const std::filesystem::path& dump_directory,
      const std::filesystem::path& semantic_manifest,
      std::string_view capture_point);
  void Finalize();
  bool healthy() const;
  std::size_t allocation_count() const;
  std::size_t release_count() const;
  std::size_t leak_count() const;
  std::size_t upload_dump_count() const;
  std::uint64_t upload_dump_bytes() const;
  std::size_t default_dump_count() const;
  std::uint64_t default_dump_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  static void NVSDK_CONV Allocate(D3D12_RESOURCE_DESC* description,
                                  int initial_state,
                                  CD3DX12_HEAP_PROPERTIES* heap,
                                  ID3D12Resource** output_resource);
  static void NVSDK_CONV Release(IUnknown* resource);
};

}  // namespace neuralstudio::fg
