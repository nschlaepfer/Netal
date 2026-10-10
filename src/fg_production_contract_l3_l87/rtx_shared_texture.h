#pragma once

namespace fg_production_contract_l3_l87 {

struct SharedTexture {
  GpuTexture d3d;
  HANDLE shared_handle = nullptr;

  void Release() {
    if (shared_handle != nullptr) {
      CloseHandle(shared_handle);
      shared_handle = nullptr;
    }
  }
};

inline SharedTexture UploadSharedTexture(
    GpuContext& gpu, DXGI_FORMAT format, std::uint32_t width,
    std::uint32_t height, std::uint32_t bytes_per_pixel,
    D3D12_RESOURCE_STATES final_state,
    const std::vector<std::uint8_t>& bytes) {
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != source_row * height)
    throw std::runtime_error("shared texture size drift");
  SharedTexture result;
  result.d3d.width = width;
  result.d3d.height = height;
  result.d3d.bytes_per_pixel = bytes_per_pixel;
  result.d3d.format = format;
  result.d3d.state = final_state;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
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
                &heap, D3D12_HEAP_FLAG_SHARED, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.d3d.resource)),
            "CreateCommittedResource(shared texture)");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint,
                                    &rows, &row_size, &upload_size);
  auto upload = CreateBuffer(gpu.device.Get(), upload_size,
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(shared texture upload)");
  for (std::uint32_t row = 0; row < height; ++row)
    std::memcpy(static_cast<std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                bytes.data() + std::size_t{row} * source_row, source_row);
  upload->Unmap(0, nullptr);
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = result.d3d.resource.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = upload.Get();
  source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  source.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  Transition(gpu.list.Get(), result.d3d.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST, final_state);
  gpu.pending.push_back(upload);
  RequireHr(gpu.device->CreateSharedHandle(result.d3d.resource.Get(), nullptr,
                                            GENERIC_ALL, nullptr,
                                            &result.shared_handle),
            "CreateSharedHandle(allocation29)");
  return result;
}

}  // namespace fg_production_contract_l3_l87
