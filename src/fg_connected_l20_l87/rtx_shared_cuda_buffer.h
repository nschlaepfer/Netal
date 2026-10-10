#pragma once

#include <cuda.h>

namespace fg_connected_l20_l87 {

inline void CudaCheck(CUresult value, const char* operation) {
  if (value == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(value, &name);
  cuGetErrorString(value, &description);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name == nullptr ? "?" : name) + " / " +
                           (description == nullptr ? "?" : description));
}

struct SharedCudaBuffer {
  GpuBuffer d3d;
  CUdevice device = 0;
  bool retained = false;
  HANDLE shared_handle = nullptr;
  CUexternalMemory external = nullptr;
  CUdeviceptr cuda_ptr = 0;

  void Release() {
    if (cuda_ptr != 0) {
      CudaCheck(cuMemFree(cuda_ptr), "free external mapping");
      cuda_ptr = 0;
    }
    if (external != nullptr) {
      CudaCheck(cuDestroyExternalMemory(external), "destroy external memory");
      external = nullptr;
    }
    if (shared_handle != nullptr) {
      CloseHandle(shared_handle);
      shared_handle = nullptr;
    }
    if (retained) {
      CudaCheck(cuDevicePrimaryCtxRelease(device),
                "release external-memory context");
      retained = false;
    }
  }
};

inline SharedCudaBuffer UploadSharedCudaBuffer(
    GpuContext& gpu, const std::vector<std::uint8_t>& bytes) {
  SharedCudaBuffer result;
  result.d3d.size = bytes.size();
  result.d3d.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  description.Width = bytes.size();
  description.Height = 1;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_SHARED, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.d3d.resource)),
            "CreateCommittedResource(shared buffer)");
  auto upload = CreateBuffer(gpu.device.Get(), bytes.size(),
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(shared upload)");
  std::memcpy(mapped, bytes.data(), bytes.size());
  upload->Unmap(0, nullptr);
  gpu.list->CopyBufferRegion(result.d3d.resource.Get(), 0, upload.Get(), 0,
                             bytes.size());
  Transition(gpu.list.Get(), result.d3d.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST, result.d3d.state);
  gpu.pending.push_back(upload);
  RequireHr(gpu.device->CreateSharedHandle(result.d3d.resource.Get(), nullptr,
                                            GENERIC_ALL, nullptr,
                                            &result.shared_handle),
            "CreateSharedHandle");
  CudaCheck(cuInit(0), "cuInit external memory");
  CudaCheck(cuDeviceGet(&result.device, 0), "cuDeviceGet external memory");
  CUcontext context = nullptr;
  CudaCheck(cuDevicePrimaryCtxRetain(&context, result.device),
            "retain external-memory context");
  result.retained = true;
  CudaCheck(cuCtxSetCurrent(context), "set external-memory context");
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC external_description{};
  external_description.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  external_description.handle.win32.handle = result.shared_handle;
  external_description.size =
      gpu.device->GetResourceAllocationInfo(0, 1, &description).SizeInBytes;
  external_description.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
  CudaCheck(cuImportExternalMemory(&result.external, &external_description),
            "import D3D12 resource");
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC map{};
  map.offset = 0;
  map.size = bytes.size();
  CudaCheck(cuExternalMemoryGetMappedBuffer(&result.cuda_ptr, result.external,
                                            &map),
            "map D3D12 resource in CUDA");
  return result;
}

}  // namespace fg_connected_l20_l87
