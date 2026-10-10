#define main fg_compositor_l87_embedded_main
#include "../fg_compositor_l87/rtx_nvapi_oracle.cpp"
#undef main

#include "contract.h"

namespace {
namespace l19 = fg_early_l19;

constexpr std::size_t kL19GuardBytes = 256;

struct TextureSpec {
  DXGI_FORMAT format;
  std::uint32_t bytes_per_pixel;
  std::uint32_t descriptor_slot;
  const char* role;
};

constexpr std::array<TextureSpec, 10> kTextureSpecs = {{
    {DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 24, "allocation24"},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 36, "allocation25"},
    {DXGI_FORMAT_R8_UNORM, 1, 37, "allocation32"},
    {DXGI_FORMAT_R8_UNORM, 1, 38, "allocation31"},
    {DXGI_FORMAT_R16G16_FLOAT, 4, 39, "allocation26"},
    {DXGI_FORMAT_R16_FLOAT, 2, 40, "allocation30"},
    {DXGI_FORMAT_R16_FLOAT, 2, 41, "allocation29"},
    {DXGI_FORMAT_R32G32_FLOAT, 8, 14, "allocation7"},
    {DXGI_FORMAT_R16G16_FLOAT, 4, 22, "allocation8"},
    {DXGI_FORMAT_R8_UNORM, 1, 42, "allocation17"},
}};

GpuTexture UploadL19Texture(GpuContext& gpu, std::uint32_t width,
                            std::uint32_t height, DXGI_FORMAT format,
                            std::uint32_t bytes_per_pixel,
                            D3D12_RESOURCE_FLAGS flags,
                            D3D12_RESOURCE_STATES final_state,
                            const std::vector<std::uint8_t>& bytes) {
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != source_row * height) {
    throw std::runtime_error("L19 texture upload size mismatch");
  }
  GpuTexture result;
  result.width = width;
  result.height = height;
  result.bytes_per_pixel = bytes_per_pixel;
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
  description.Flags = flags;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.resource)),
            "CreateCommittedResource(L19 texture)");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &upload_size);
  auto upload = CreateBuffer(gpu.device.Get(), upload_size,
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(L19 texture upload)");
  for (std::uint32_t row = 0; row < height; ++row) {
    std::memcpy(static_cast<std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                bytes.data() + std::size_t{row} * source_row, source_row);
  }
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

std::vector<std::uint8_t> ReadL19Texture(GpuContext& gpu,
                                         GpuTexture& source) {
  const auto description = source.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 readback_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &readback_size);
  auto readback = CreateBuffer(gpu.device.Get(), readback_size,
                               D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), source.resource.Get(), source.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  source.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  D3D12_TEXTURE_COPY_LOCATION input{};
  input.pResource = source.resource.Get();
  input.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION output{};
  output.pResource = readback.Get();
  output.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  output.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&output, 0, 0, 0, &input, nullptr);
  ExecuteAndReset(gpu);
  const std::size_t output_row =
      std::size_t{source.width} * source.bytes_per_pixel;
  std::vector<std::uint8_t> bytes(output_row * source.height);
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(readback_size)};
  RequireHr(readback->Map(0, &read_range, &mapped),
            "Map(L19 texture readback)");
  for (std::uint32_t row = 0; row < source.height; ++row) {
    std::memcpy(bytes.data() + std::size_t{row} * output_row,
                static_cast<const std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                output_row);
  }
  const D3D12_RANGE no_write{0, 0};
  readback->Unmap(0, &no_write);
  return bytes;
}

template <std::size_t Size>
void L19Pack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
               std::uint32_t value) {
  if (offset + 4u > Size) throw std::runtime_error("L19Pack32 OOB");
  for (std::size_t index = 0; index < 4u; ++index) {
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
  }
}

template <std::size_t Size>
void L19Pack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
               std::uint64_t value) {
  if (offset + 8u > Size) throw std::runtime_error("L19Pack64 OOB");
  for (std::size_t index = 0; index < 8u; ++index) {
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
  }
}

template <std::size_t Size>
std::string AddressFreeHash(std::array<std::uint8_t, Size> packed) {
  for (const std::size_t offset : {std::size_t{0}, std::size_t{8},
                                   std::size_t{16}, std::size_t{24},
                                   std::size_t{32}, std::size_t{40},
                                   std::size_t{48}, std::size_t{56},
                                   std::size_t{64}, std::size_t{72},
                                   std::size_t{80}, std::size_t{88},
                                   std::size_t{96}}) {
    for (std::size_t byte = 0; byte < 8u; ++byte) packed[offset + byte] = 0;
  }
  return Sha256(std::vector<std::uint8_t>(packed.begin(), packed.end()));
}

std::size_t PoisonPixels(const std::vector<std::uint8_t>& bytes,
                         std::size_t pixel_bytes, std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += pixel_bytes) {
    bool same = true;
    for (std::size_t byte = 0; byte < pixel_bytes; ++byte) {
      same = same && bytes[offset + byte] == poison;
    }
    count += same;
  }
  return count;
}

std::size_t L19CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                                std::size_t payload_bytes,
                                std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < kL19GuardBytes; ++index) {
    count += guarded[index] != poison;
    count += guarded[kL19GuardBytes + payload_bytes + index] != poison;
  }
  return count;
}

void WriteL19Receipt(
    const fs::path& path, const std::string& run_id, l19::Fixture fixture,
    std::uint8_t poison, const l19::Inputs& inputs,
    const std::array<std::vector<std::uint8_t>, 10>& inputs_after,
    const std::vector<std::uint8_t>& tensor,
    const std::vector<std::uint8_t>& mask,
    const std::vector<std::uint8_t>& vector,
    std::size_t input_mismatches, std::size_t buffer_canary_mismatches,
    std::size_t texture_canary_mismatches,
    const std::string& fixture_parameter_sha) {
  const bool pass = input_mismatches == 0 && buffer_canary_mismatches == 0 &&
                    texture_canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write L19 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.early-l19-run.v1\",\n"
          << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"fixture\": \"" << l19::FixtureName(fixture) << "\",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"selected_launch\": 19,\n"
          << "  \"module_token\": 5,\n"
          << "  \"module_bytes\": " << l19::kModuleBytes << ",\n"
          << "  \"module_sha256\": \"" << l19::kModuleSha256 << "\",\n"
          << "  \"module_transport\": \"NvAPI_D3D12_CreateCuModule selected fatbin\",\n"
          << "  \"function_name\": \"main_kernel\",\n"
          << "  \"grid\": [20, 12, 1],\n"
          << "  \"block\": [16, 16, 1],\n"
          << "  \"packed_parameter_bytes\": 192,\n"
          << "  \"frozen_selected_raw_parameter_sha256\": \""
          << l19::kFrozenRawParameterSha256 << "\",\n"
          << "  \"frozen_selected_address_free_parameter_sha256\": \""
          << l19::kFrozenAddressFreeParameterSha256 << "\",\n"
          << "  \"fixture_address_free_parameter_sha256\": \""
          << fixture_parameter_sha << "\",\n"
          << "  \"input_generator\": \"" << l19::kInputGenerator
          << "\",\n"
          << "  \"allocation29_first_frame_evidence_root_sha256\": \""
          << l19::kAllocation29FirstFrameEvidenceRootSha256 << "\",\n"
          << "  \"allocation29_temporal_evidence_root_sha256\": \""
          << l19::kAllocation29TemporalEvidenceRootSha256 << "\",\n"
          << "  \"allocation29_runtime_source\": \"independently-generated\",\n"
          << "  \"allocation29_capture_bytes_consumed\": false,\n"
          << "  \"input_pre_sha256\": [";
  for (std::size_t index = 0; index < inputs.textures.size(); ++index) {
    if (index != 0) receipt << ", ";
    receipt << "\"" << Sha256(inputs.textures[index]) << "\"";
  }
  receipt << "],\n  \"input_post_sha256\": [";
  for (std::size_t index = 0; index < inputs_after.size(); ++index) {
    if (index != 0) receipt << ", ";
    receipt << "\"" << Sha256(inputs_after[index]) << "\"";
  }
  receipt << "],\n"
          << "  \"tensor_output_bytes\": " << tensor.size() << ",\n"
          << "  \"tensor_output_sha256\": \"" << Sha256(tensor) << "\",\n"
          << "  \"mask_output_bytes\": " << mask.size() << ",\n"
          << "  \"mask_output_sha256\": \"" << Sha256(mask) << "\",\n"
          << "  \"vector_output_bytes\": " << vector.size() << ",\n"
          << "  \"vector_output_sha256\": \"" << Sha256(vector) << "\",\n"
          << "  \"tensor_poison_pixels\": "
          << PoisonPixels(tensor, 2u, poison) << ",\n"
          << "  \"mask_poison_pixels\": "
          << PoisonPixels(mask, 1u, poison) << ",\n"
          << "  \"vector_poison_pixels\": "
          << PoisonPixels(vector, 4u, poison) << ",\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"buffer_canary_mismatched_bytes\": "
          << buffer_canary_mismatches << ",\n"
          << "  \"texture_canary_mismatched_bytes\": "
          << texture_canary_mismatches << ",\n"
          << "  \"texture_resource_bounds_enforced\": true,\n"
          << "  \"complete_write_coverage_checked_by_dual_poison\": true,\n"
          << "  \"expected_output_fed_to_execution\": false,\n"
          << "  \"archived_intermediates_consumed\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage=fg-early-l19-nvapi <module> <output-dir> <run-id> "
                 "<poison-hex> <first-frame|fan-in>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    const auto fixture = l19::ParseFixture(argv[5]);
    fs::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != l19::kModuleBytes ||
        Sha256(module_bytes) != l19::kModuleSha256) {
      throw std::runtime_error("selected module5 identity mismatch");
    }
    const auto inputs = l19::GenerateInputs(fixture);
    for (std::size_t index = 0; index < inputs.textures.size(); ++index) {
      if (inputs.textures[index].size() != l19::kInputBytes[index]) {
        throw std::runtime_error("generated L19 input size mismatch");
      }
    }
    std::vector<std::uint8_t> guarded_tensor(
        l19::kTensorBytes + 2u * kL19GuardBytes, poison);
    const std::vector<std::uint8_t> mask_poison(l19::kMaskOutputBytes,
                                                 poison);
    const std::vector<std::uint8_t> vector_poison(l19::kVectorOutputBytes,
                                                   poison);
    const std::vector<std::uint8_t> mask_canary_before(
        l19::kMaskOutputBytes, static_cast<std::uint8_t>(poison ^ 0xffu));
    const std::vector<std::uint8_t> vector_canary_before(
        l19::kVectorOutputBytes, static_cast<std::uint8_t>(poison ^ 0xffu));

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (query == nullptr) throw std::runtime_error("nvapi_QueryInterface missing");
    const auto initialize = Resolve<Initialize>(query, kIdInitialize,
                                                 "NvAPI_Initialize");
    const auto get_texture = Resolve<GetMergedTextureSampler>(
        query, kIdGetMergedTextureSampler, "GetCudaMergedTextureSamplerObject");
    const auto get_surface = Resolve<GetIndependentDescriptor>(
        query, kIdGetIndependentDescriptor,
        "GetCudaIndependentDescriptorObject");
    const auto create_module = Resolve<CreateCuModule>(
        query, kIdCreateCuModule, "CreateCuModule");
    const auto create_function = Resolve<CreateCuFunction>(
        query, kIdCreateCuFunction, "CreateCuFunction");
    const auto launch_chain = Resolve<LaunchKernelChain>(
        query, kIdLaunchKernelChain, "LaunchKernelChain");
    const auto destroy_module = Resolve<DestroyCuModule>(
        query, kIdDestroyCuModule, "DestroyCuModule");
    const auto destroy_function = Resolve<DestroyCuFunction>(
        query, kIdDestroyCuFunction, "DestroyCuFunction");
    RequireNvapi(initialize(), "NvAPI_Initialize");

    std::array<GpuTexture, 10> input_textures;
    for (std::size_t index = 0; index < input_textures.size(); ++index) {
      input_textures[index] = UploadL19Texture(
          gpu, l19::kInputWidth, l19::kInputHeight,
          kTextureSpecs[index].format, kTextureSpecs[index].bytes_per_pixel,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, inputs.textures[index]);
    }
    GpuBuffer tensor_buffer = UploadBuffer(gpu, guarded_tensor);
    GpuTexture mask_output = UploadL19Texture(
        gpu, l19::kInputWidth, l19::kInputHeight, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, mask_poison);
    GpuTexture vector_output = UploadL19Texture(
        gpu, l19::kInputWidth, l19::kInputHeight,
        DXGI_FORMAT_R16G16_FLOAT, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, vector_poison);
    GpuTexture mask_canary = UploadL19Texture(
        gpu, l19::kInputWidth, l19::kInputHeight, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, mask_canary_before);
    GpuTexture vector_canary = UploadL19Texture(
        gpu, l19::kInputWidth, l19::kInputHeight,
        DXGI_FORMAT_R16G16_FLOAT, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, vector_canary_before);
    ExecuteAndReset(gpu);

    D3D12_DESCRIPTOR_HEAP_DESC resource_heap_description{};
    resource_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_heap_description.NumDescriptors = 2048;
    resource_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_heap_description,
                                                IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(L19 resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_description{};
    sampler_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_heap_description.NumDescriptors = 1;
    sampler_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_heap_description,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(L19 sampler)");
    const UINT increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto resource_start =
        resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto descriptor = [&](std::uint32_t slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          resource_start.ptr + std::uint64_t{slot} * increment};
    };
    const auto sampler = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC sampler_description{};
    sampler_description.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_description.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&sampler_description, sampler);

    std::array<std::uint64_t, 10> texture_handles{};
    for (std::size_t index = 0; index < input_textures.size(); ++index) {
      D3D12_SHADER_RESOURCE_VIEW_DESC view{};
      view.Format = kTextureSpecs[index].format;
      view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      view.Texture2D.MipLevels = 1;
      const auto srv = descriptor(kTextureSpecs[index].descriptor_slot);
      gpu.device->CreateShaderResourceView(input_textures[index].resource.Get(),
                                           &view, srv);
      MergedTextureSamplerParams parameters{};
      parameters.struct_size_in = sizeof(parameters);
      parameters.struct_size_out = sizeof(parameters);
      parameters.device = gpu.device.Get();
      parameters.texture_descriptor.ptr = srv.ptr;
      parameters.sampler_descriptor.ptr = sampler.ptr;
      RequireNvapi(get_texture(&parameters), "GetCudaMergedTextureSamplerObject");
      if (parameters.texture_handle == 0) {
        throw std::runtime_error("null L19 texture handle");
      }
      texture_handles[index] = parameters.texture_handle;
    }
    const auto mask_uav = descriptor(43);
    const auto vector_uav = descriptor(44);
    D3D12_UNORDERED_ACCESS_VIEW_DESC mask_view{};
    mask_view.Format = DXGI_FORMAT_R8_UNORM;
    mask_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(mask_output.resource.Get(), nullptr,
                                          &mask_view, mask_uav);
    D3D12_UNORDERED_ACCESS_VIEW_DESC vector_view{};
    vector_view.Format = DXGI_FORMAT_R16G16_FLOAT;
    vector_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(vector_output.resource.Get(), nullptr,
                                          &vector_view, vector_uav);
    IndependentDescriptorParams mask_surface{};
    mask_surface.struct_size_in = sizeof(mask_surface);
    mask_surface.struct_size_out = sizeof(mask_surface);
    mask_surface.device = gpu.device.Get();
    mask_surface.type = IndependentDescriptorType::kSurface;
    mask_surface.descriptor.ptr = mask_uav.ptr;
    RequireNvapi(get_surface(&mask_surface),
                 "GetCudaIndependentDescriptorObject(mask)");
    IndependentDescriptorParams vector_surface{};
    vector_surface.struct_size_in = sizeof(vector_surface);
    vector_surface.struct_size_out = sizeof(vector_surface);
    vector_surface.device = gpu.device.Get();
    vector_surface.type = IndependentDescriptorType::kSurface;
    vector_surface.descriptor.ptr = vector_uav.ptr;
    RequireNvapi(get_surface(&vector_surface),
                 "GetCudaIndependentDescriptorObject(vector)");
    if (mask_surface.handle == 0 || vector_surface.handle == 0) {
      throw std::runtime_error("null L19 surface handle");
    }

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module),
                 "CreateCuModule(module5)");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function),
                 "CreateCuFunction(main_kernel)");
    std::array<std::uint8_t, 192> packed{};
    for (std::size_t index = 0; index < texture_handles.size(); ++index) {
      L19Pack64(&packed, index * 8u, texture_handles[index]);
    }
    L19Pack64(&packed, 80,
              tensor_buffer.resource->GetGPUVirtualAddress() +
                  kL19GuardBytes);
    L19Pack64(&packed, 88, mask_surface.handle);
    L19Pack64(&packed, 96, vector_surface.handle);
    L19Pack32(&packed, 152, l19::kInputWidth);
    L19Pack32(&packed, 156, l19::kInputHeight);
    L19Pack32(&packed, 160, l19::kInputWidth);
    L19Pack32(&packed, 164, l19::kInputHeight);
    L19Pack32(&packed, 168, l19::kTensorWidth);
    L19Pack32(&packed, 172, l19::kTensorHeight);
    L19Pack32(&packed, 176, 0);
    L19Pack32(&packed, 180, 0);
    L19Pack32(&packed, 184, 3);
    L19Pack32(&packed, 188, 3);
    const std::string fixture_parameter_sha = AddressFreeHash(packed);
    if (fixture_parameter_sha != l19::kFixtureAddressFreeParameterSha256) {
      throw std::runtime_error("L19 fixture parameter identity mismatch");
    }
    const KernelLaunchParams launch{
        function, {20, 12, 1}, {16, 16, 1}, 0, packed.data(),
        static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(module5/main_kernel)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    std::array<std::vector<std::uint8_t>, 10> inputs_after;
    std::size_t input_mismatches = 0;
    for (std::size_t index = 0; index < input_textures.size(); ++index) {
      inputs_after[index] = ReadL19Texture(gpu, input_textures[index]);
      input_mismatches +=
          Mismatches(inputs.textures[index], inputs_after[index]);
    }
    const auto mask_after = ReadL19Texture(gpu, mask_output);
    const auto vector_after = ReadL19Texture(gpu, vector_output);
    const auto mask_canary_after = ReadL19Texture(gpu, mask_canary);
    const auto vector_canary_after = ReadL19Texture(gpu, vector_canary);
    const auto tensor_guarded_after = ReadBuffer(gpu, tensor_buffer);
    const std::vector<std::uint8_t> tensor_after(
        tensor_guarded_after.begin() + kL19GuardBytes,
        tensor_guarded_after.begin() + kL19GuardBytes + l19::kTensorBytes);
    const std::size_t buffer_canary_mismatches = L19CanaryMismatches(
        tensor_guarded_after, l19::kTensorBytes, poison);
    const std::size_t texture_canary_mismatches =
        Mismatches(mask_canary_before, mask_canary_after) +
        Mismatches(vector_canary_before, vector_canary_after);
    Write(output_directory / "tensor-output.fp16", tensor_after);
    Write(output_directory / "mask-output.r8", mask_after);
    Write(output_directory / "vector-output.rg16f", vector_after);
    WriteL19Receipt(output_directory / "run-receipt.json", run_id, fixture,
                    poison, inputs, inputs_after, tensor_after, mask_after,
                    vector_after, input_mismatches, buffer_canary_mismatches,
                    texture_canary_mismatches, fixture_parameter_sha);
    const bool pass = input_mismatches == 0 &&
                      buffer_canary_mismatches == 0 &&
                      texture_canary_mismatches == 0;
    std::cout << "probe=fg_early_l19_rtx_nvapi\n"
              << "run_id=" << run_id << '\n'
              << "tensor_output_sha256=" << Sha256(tensor_after) << '\n'
              << "mask_output_sha256=" << Sha256(mask_after) << '\n'
              << "vector_output_sha256=" << Sha256(vector_after) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
