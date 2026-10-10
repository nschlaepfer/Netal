#define main FgSubrectEmbeddedCompositorMain
#include "../fg_compositor_l87/rtx_nvapi_oracle.cpp"
#undef main

#include "contract.h"

namespace subrect = fg_connected_subrect_l3_l83;

namespace {

GpuTexture UploadSubrectTexture(
    GpuContext& gpu, std::uint32_t width, std::uint32_t height,
    DXGI_FORMAT format, std::uint32_t bytes_per_pixel,
    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES final_state,
    const std::vector<std::uint8_t>& bytes) {
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != source_row * height)
    throw std::runtime_error("subrect texture upload size mismatch");
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
            "CreateCommittedResource(subrect texture)");
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
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(subrect upload)");
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

std::vector<std::uint8_t> ReadSubrectTexture(GpuContext& gpu,
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
            "Map(subrect readback)");
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
void SubrectPack32(std::array<std::uint8_t, Size>* bytes,
                   std::size_t offset, std::uint32_t value) {
  if (offset + 4u > Size) throw std::runtime_error("pack32 OOB");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void SubrectPack64(std::array<std::uint8_t, Size>* bytes,
                   std::size_t offset, std::uint64_t value) {
  if (offset + 8u > Size) throw std::runtime_error("pack64 OOB");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::size_t SubrectCanaryMismatches(
    const std::vector<std::uint8_t>& bytes, std::size_t payload_bytes) {
  std::size_t result = 0;
  for (std::size_t index = 0; index < subrect::kGuardBytes; ++index) {
    result += bytes[index] != subrect::kGuardValue;
    result += bytes[subrect::kGuardBytes + payload_bytes + index] !=
              subrect::kGuardValue;
  }
  return result;
}

std::uint32_t SubrectFixture(const std::string& value) {
  if (value == "mode1-a") return 0;
  if (value == "mode1-b") return 1;
  throw std::runtime_error("fixture must be mode1-a or mode1-b");
}

struct SubrectCoverage {
  std::size_t positive_mask_pixels = 0;
  std::size_t changed_positive_pixels = 0;
  std::size_t zero_mask_pixels = 0;
  std::size_t zero_mask_mismatched_bytes = 0;
  std::size_t outside_mismatched_bytes = 0;
};

SubrectCoverage MeasureSubrect(
    const std::vector<std::uint8_t>& before,
    const std::vector<std::uint8_t>& after,
    const std::vector<std::uint8_t>& mask) {
  SubrectCoverage result;
  for (std::uint32_t y = 0; y < subrect::kResourceHeight; ++y) {
    for (std::uint32_t x = 0; x < subrect::kResourceWidth; ++x) {
      const std::size_t offset =
          (std::size_t{y} * subrect::kResourceWidth + x) * 4u;
      const bool inside =
          x >= subrect::kOriginX && x < subrect::kOriginX + subrect::kWidth &&
          y >= subrect::kOriginY && y < subrect::kOriginY + subrect::kHeight;
      if (!inside) {
        for (std::size_t channel = 0; channel < 4; ++channel)
          result.outside_mismatched_bytes +=
              before[offset + channel] != after[offset + channel];
        continue;
      }
      const std::size_t local =
          std::size_t{y - subrect::kOriginY} * subrect::kWidth +
          x - subrect::kOriginX;
      if (mask[local] == 0) {
        ++result.zero_mask_pixels;
        for (std::size_t channel = 0; channel < 4; ++channel)
          result.zero_mask_mismatched_bytes +=
              before[offset + channel] != after[offset + channel];
      } else {
        ++result.positive_mask_pixels;
        bool changed = false;
        for (std::size_t channel = 0; channel < 4; ++channel)
          changed = changed || before[offset + channel] != after[offset + channel];
        result.changed_positive_pixels += changed;
      }
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-subrect-module19-rtx <module> <output> <run-id> "
                 "<fixture> <poison> <mode>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output = argv[2];
    const std::string run_id = argv[3];
    const std::string fixture_name = argv[4];
    const auto fixture = SubrectFixture(fixture_name);
    const auto poison = ParsePoison(argv[5]);
    const auto mode = std::stoul(argv[6], nullptr, 10);
    if (mode != 1u) {
      std::cerr << "error=mode-1 specialization rejects mode " << mode << '\n';
      return 4;
    }
    fs::create_directories(output);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != subrect::kModule19Bytes ||
        Sha256(module_bytes) != subrect::kModule19Sha256)
      throw std::runtime_error("selected module19 identity mismatch");
    const auto levels = subrect::Pyramid(fixture);
    const auto mask_before = subrect::Mask(fixture);
    const auto output_before =
        subrect::InitialOutput(fixture, poison, mask_before);
    std::array<std::vector<std::uint8_t>, 4> guarded_levels;
    for (std::size_t index = 0; index < levels.size(); ++index) {
      guarded_levels[index].assign(
          levels[index].size() + 2u * subrect::kGuardBytes,
          subrect::kGuardValue);
      std::memcpy(guarded_levels[index].data() + subrect::kGuardBytes,
                  levels[index].data(), levels[index].size());
    }

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (query == nullptr) throw std::runtime_error("nvapi query missing");
    const auto initialize = Resolve<Initialize>(query, kIdInitialize,
                                                 "NvAPI_Initialize");
    const auto get_texture = Resolve<GetMergedTextureSampler>(
        query, kIdGetMergedTextureSampler,
        "GetCudaMergedTextureSamplerObject");
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

    std::array<GpuBuffer, 4> level_buffers;
    for (std::size_t index = 0; index < level_buffers.size(); ++index)
      level_buffers[index] = UploadBuffer(gpu, guarded_levels[index]);
    auto mask = UploadSubrectTexture(
        gpu, subrect::kWidth, subrect::kHeight, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mask_before);
    auto target = UploadSubrectTexture(
        gpu, subrect::kResourceWidth, subrect::kResourceHeight,
        DXGI_FORMAT_R8G8B8A8_UNORM, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output_before);
    const std::vector<std::uint8_t> canary_before(
        subrect::kOutputBytes, static_cast<std::uint8_t>(poison ^ 0xffu));
    auto target_canary = UploadSubrectTexture(
        gpu, subrect::kResourceWidth, subrect::kResourceHeight,
        DXGI_FORMAT_R8G8B8A8_UNORM, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, canary_before);
    ExecuteAndReset(gpu);

    D3D12_DESCRIPTOR_HEAP_DESC heap_description{};
    heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_description.NumDescriptors = 2048;
    heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(
                  &heap_description, IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(resources)");
    heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    heap_description.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(
                  &heap_description, IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(sampler)");
    const auto start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto descriptor = [&](std::uint32_t slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          start.ptr + std::uint64_t{slot} * increment};
    };
    const auto srv = descriptor(59);
    const auto uav = descriptor(54);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_description{};
    srv_description.Format = DXGI_FORMAT_R8_UNORM;
    srv_description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_description.Shader4ComponentMapping =
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_description.Texture2D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(mask.resource.Get(),
                                         &srv_description, srv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_description{};
    uav_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav_description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(target.resource.Get(), nullptr,
                                          &uav_description, uav);
    D3D12_SAMPLER_DESC sampler_description{};
    sampler_description.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_description.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    const auto sampler = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    gpu.device->CreateSampler(&sampler_description, sampler);
    MergedTextureSamplerParams texture{};
    texture.struct_size_in = texture.struct_size_out = sizeof(texture);
    texture.device = gpu.device.Get();
    texture.texture_descriptor.ptr = srv.ptr;
    texture.sampler_descriptor.ptr = sampler.ptr;
    RequireNvapi(get_texture(&texture), "GetCudaMergedTextureSamplerObject");
    IndependentDescriptorParams surface{};
    surface.struct_size_in = surface.struct_size_out = sizeof(surface);
    surface.device = gpu.device.Get();
    surface.type = IndependentDescriptorType::kSurface;
    surface.descriptor.ptr = uav.ptr;
    RequireNvapi(get_surface(&surface),
                 "GetCudaIndependentDescriptorObject");
    if (texture.texture_handle == 0 || surface.handle == 0)
      throw std::runtime_error("NvAPI returned null descriptor object");

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module),
                 "CreateCuModule");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function),
                 "CreateCuFunction");
    std::array<std::uint8_t, 80> parameters{};
    for (std::size_t index = 0; index < level_buffers.size(); ++index)
      SubrectPack64(&parameters, 8u + index * 8u,
                    level_buffers[index].resource->GetGPUVirtualAddress() +
                        subrect::kGuardBytes);
    SubrectPack32(&parameters, 40, subrect::kWidth);
    SubrectPack32(&parameters, 44, subrect::kHeight);
    SubrectPack32(&parameters, 48, 1);
    SubrectPack64(&parameters, 56, texture.texture_handle);
    SubrectPack64(&parameters, 64, surface.handle);
    SubrectPack32(&parameters, 72, subrect::kOriginX);
    SubrectPack32(&parameters, 76, subrect::kOriginY);
    const KernelLaunchParams launch{
        function, {subrect::kGridX, subrect::kGridY, 1}, {256, 1, 1}, 0,
        parameters.data(), static_cast<std::uint32_t>(parameters.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(module19 mode1)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto mask_after = ReadSubrectTexture(gpu, mask);
    const auto output_after = ReadSubrectTexture(gpu, target);
    const auto target_canary_after = ReadSubrectTexture(gpu, target_canary);
    std::size_t input_mismatches = Mismatches(mask_before, mask_after);
    std::size_t canary_mismatches = 0;
    std::array<std::vector<std::uint8_t>, 4> level_after;
    for (std::size_t index = 0; index < level_buffers.size(); ++index) {
      const auto guarded = ReadBuffer(gpu, level_buffers[index]);
      level_after[index] = {
          guarded.begin() + subrect::kGuardBytes,
          guarded.begin() + subrect::kGuardBytes + levels[index].size()};
      input_mismatches += Mismatches(levels[index], level_after[index]);
      canary_mismatches +=
          SubrectCanaryMismatches(guarded, levels[index].size());
    }
    const auto coverage =
        MeasureSubrect(output_before, output_after, mask_before);
    const auto region = subrect::Region(output_after);
    const std::size_t texture_canary_mismatches =
        Mismatches(canary_before, target_canary_after);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                      texture_canary_mismatches == 0 &&
                      coverage.positive_mask_pixels != 0 &&
                      coverage.zero_mask_pixels != 0 &&
                      coverage.changed_positive_pixels ==
                          coverage.positive_mask_pixels &&
                      coverage.zero_mask_mismatched_bytes == 0 &&
                      coverage.outside_mismatched_bytes == 0;
    Write(output / "output-full.rgba8", output_after);
    Write(output / "output-region.rgba8", region);
    Write(output / "input-mask-after.r8", mask_after);
    for (std::size_t index = 0; index < level_after.size(); ++index)
      Write(output / ("input-level" + std::to_string(index) + "-after.bin"),
            level_after[index]);
    std::ofstream receipt(output / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.subrect-module19-mode1-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"fixture\": \"" << fixture_name << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"source_module_token\": 19,\n"
            << "  \"source_module_sha256\": \""
            << subrect::kModule19Sha256 << "\",\n"
            << "  \"module_transport\": \"NvAPI D3D12 selected fatbin\",\n"
            << "  \"raw_parameter_sha256\": \""
            << subrect::kRawL83ParameterSha256 << "\",\n"
            << "  \"address_free_parameter_sha256\": "
               "\"2ebebd84a2aa74f814428d545d870c9e1a25c0fea16309c5e38b3fb97881d956\",\n"
            << "  \"packed_parameter_bytes\": 80,\n"
            << "  \"mode\": 1,\n"
            << "  \"origin\": [32, 24],\n"
            << "  \"internal_extent\": [512, 288],\n"
            << "  \"resource_extent\": [640, 360],\n"
            << "  \"grid\": [32, 18, 1],\n"
            << "  \"block\": [256, 1, 1],\n"
            << "  \"input_generator\": \"" << subrect::kGenerator << "\",\n"
            << "  \"input_level_sha256\": [";
    for (std::size_t index = 0; index < level_after.size(); ++index)
      receipt << (index ? ", \"" : "\"") << Sha256(level_after[index])
              << "\"";
    receipt << "],\n"
            << "  \"input_mask_sha256\": \"" << Sha256(mask_after)
            << "\",\n"
            << "  \"output_full_sha256\": \"" << Sha256(output_after)
            << "\",\n"
            << "  \"output_region_bytes\": " << region.size() << ",\n"
            << "  \"output_region_sha256\": \"" << Sha256(region)
            << "\",\n"
            << "  \"positive_mask_pixels\": "
            << coverage.positive_mask_pixels << ",\n"
            << "  \"changed_positive_pixels\": "
            << coverage.changed_positive_pixels << ",\n"
            << "  \"zero_mask_pixels\": " << coverage.zero_mask_pixels
            << ",\n"
            << "  \"zero_mask_mismatched_bytes\": "
            << coverage.zero_mask_mismatched_bytes << ",\n"
            << "  \"outside_mismatched_bytes\": "
            << coverage.outside_mismatched_bytes << ",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches
            << ",\n"
            << "  \"buffer_canary_mismatched_bytes\": "
            << canary_mismatches << ",\n"
            << "  \"texture_canary_mismatched_bytes\": "
            << texture_canary_mismatches << ",\n"
            << "  \"typed_d3d12_resources\": true,\n"
            << "  \"expected_or_captured_output_ingress\": false,\n"
            << "  \"archived_intermediate_ingress\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (pass ? "pass" : "fail")
            << "\"\n}\n";
    std::cout << "output_region_sha256=" << Sha256(region) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
