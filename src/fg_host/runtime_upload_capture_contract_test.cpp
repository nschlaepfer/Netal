#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include "runtime_allocation_capture.h"

#include <cassert>
#include <cstring>
#include <iostream>

int main() {
  using neuralstudio::fg::ClassifyCpuVisibleBuffer;
  using neuralstudio::fg::CpuVisibleBufferClass;
  using neuralstudio::fg::CpuVisibleBufferClassName;
  using neuralstudio::fg::IsImplicitCopySourceDefaultBuffer;

  D3D12_RESOURCE_DESC buffer{};
  buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  buffer.Width = 56320;
  buffer.Height = 1;
  buffer.DepthOrArraySize = 1;
  buffer.MipLevels = 1;
  buffer.SampleDesc.Count = 1;
  buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_UPLOAD;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kUpload);
  assert(std::strcmp(CpuVisibleBufferClassName(
                         ClassifyCpuVisibleBuffer(buffer, heap)),
                     "upload") == 0);

  heap.Type = D3D12_HEAP_TYPE_READBACK;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kReadback);

  heap.Type = D3D12_HEAP_TYPE_CUSTOM;
  heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kCustomWriteCombine);
  heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kCustomWriteBack);
  heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kNotSafelyReadable);

  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kNotSafelyReadable);
  assert(IsImplicitCopySourceDefaultBuffer(
      buffer, heap, static_cast<int>(D3D12_RESOURCE_STATE_COMMON)));
  assert(!IsImplicitCopySourceDefaultBuffer(
      buffer, heap, static_cast<int>(D3D12_RESOURCE_STATE_COPY_SOURCE)));
  heap.Type = D3D12_HEAP_TYPE_UPLOAD;
  assert(!IsImplicitCopySourceDefaultBuffer(
      buffer, heap, static_cast<int>(D3D12_RESOURCE_STATE_COMMON)));

  buffer.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kNotSafelyReadable);
  buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  buffer.Width = 0;
  assert(ClassifyCpuVisibleBuffer(buffer, heap) ==
         CpuVisibleBufferClass::kNotSafelyReadable);

  std::cout << "runtime_upload_capture.classification_contract=passed\n";
  std::cout << "runtime_upload_capture.default_heap_read=false\n";
  return 0;
}
