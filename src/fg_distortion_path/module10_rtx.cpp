#define main fg_compositor_embedded_main
#include "../fg_compositor_l87/rtx_nvapi_oracle.cpp"
#undef main

#include "module10_contract.h"

namespace {

namespace m10 = fg_distortion_module10;

GpuTexture UploadModule10Texture(GpuContext &gpu, std::uint32_t width,
                                 std::uint32_t height, DXGI_FORMAT format,
                                 std::uint32_t bytes_per_pixel,
                                 D3D12_RESOURCE_FLAGS flags,
                                 D3D12_RESOURCE_STATES final_state,
                                 const std::vector<std::uint8_t> &bytes) {
  GpuTexture result;
  result.width = width;
  result.height = height;
  result.bytes_per_pixel = bytes_per_pixel;
  result.state = final_state;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap.CreationNodeMask = heap.VisibleNodeMask = 1;
  D3D12_RESOURCE_DESC descriptor{};
  descriptor.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  descriptor.Width = width;
  descriptor.Height = height;
  descriptor.DepthOrArraySize = 1;
  descriptor.MipLevels = 1;
  descriptor.Format = format;
  descriptor.SampleDesc.Count = 1;
  descriptor.Flags = flags;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &descriptor,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.resource)),
            "create texture");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 upload_size = 0;
  gpu.device->GetCopyableFootprints(&descriptor, 0, 1, 0, &footprint, &rows,
                                    &row_size, &upload_size);
  auto upload =
      CreateBuffer(gpu.device.Get(), upload_size, D3D12_HEAP_TYPE_UPLOAD,
                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  void *mapped = nullptr;
  D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "map upload");
  for (std::uint32_t y = 0; y < height; ++y) {
    std::memcpy(static_cast<std::uint8_t *>(mapped) +
                    std::size_t{y} * footprint.Footprint.RowPitch,
                bytes.data() + std::size_t{y} * width * bytes_per_pixel,
                std::size_t{width} * bytes_per_pixel);
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

std::vector<std::uint8_t> ReadModule10Texture(GpuContext &gpu,
                                              GpuTexture &source) {
  const auto descriptor = source.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 readback_size = 0;
  gpu.device->GetCopyableFootprints(&descriptor, 0, 1, 0, &footprint, &rows,
                                    &row_size, &readback_size);
  auto readback =
      CreateBuffer(gpu.device.Get(), readback_size, D3D12_HEAP_TYPE_READBACK,
                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), source.resource.Get(), source.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  source.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = readback.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  destination.PlacedFootprint = footprint;
  D3D12_TEXTURE_COPY_LOCATION input{};
  input.pResource = source.resource.Get();
  input.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  gpu.list->CopyTextureRegion(&destination, 0, 0, 0, &input, nullptr);
  ExecuteAndReset(gpu);
  std::vector<std::uint8_t> result(std::size_t{source.width} * source.height *
                                   source.bytes_per_pixel);
  void *mapped = nullptr;
  D3D12_RANGE range{0, static_cast<SIZE_T>(readback_size)};
  RequireHr(readback->Map(0, &range, &mapped), "map readback");
  for (std::uint32_t y = 0; y < source.height; ++y) {
    std::memcpy(result.data() +
                    std::size_t{y} * source.width * source.bytes_per_pixel,
                static_cast<std::uint8_t *>(mapped) +
                    std::size_t{y} * footprint.Footprint.RowPitch,
                std::size_t{source.width} * source.bytes_per_pixel);
  }
  readback->Unmap(0, nullptr);
  return result;
}

template <std::size_t Size>
void Pack64(std::array<std::uint8_t, Size> &bytes, std::size_t offset,
            std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    bytes[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
  }
}

template <std::size_t Size>
void Pack32(std::array<std::uint8_t, Size> &bytes, std::size_t offset,
            std::uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) {
    bytes[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
  }
}

std::size_t CountPoisonPixels(const std::vector<std::uint8_t> &bytes,
                              std::size_t pixel_bytes, std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += pixel_bytes) {
    bool untouched = true;
    for (std::size_t byte = 0; byte < pixel_bytes; ++byte)
      untouched &= bytes[offset + byte] == poison;
    result += untouched;
  }
  return result;
}

} // namespace

#ifndef FG_DISTORTION_MODULE10_NO_MAIN
int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  try {
    fs::path module_path = argv[1];
    fs::path output_root = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    fs::create_directories(output_root);

    const auto module = Read(module_path);
    if (module.size() != m10::kModuleBytes ||
        Sha256(module) != m10::kModuleSha256) {
      throw std::runtime_error("module10 identity");
    }
    const auto input = m10::GenerateInput();
    std::vector<std::uint8_t> initialized(m10::kRgba16Bytes, poison);
    std::vector<std::uint8_t> canary_bytes(m10::kRgba16Bytes, poison ^ 0xffu);

    GpuContext gpu = CreateGpuContext();
    auto input_texture = UploadModule10Texture(
        gpu, m10::kWidth, m10::kHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, input);
    auto output_texture = UploadModule10Texture(
        gpu, m10::kWidth, m10::kHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, initialized);
    auto canary_texture = UploadModule10Texture(
        gpu, m10::kWidth, m10::kHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, canary_bytes);
    ExecuteAndReset(gpu);

    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    auto initialize = Resolve<Initialize>(query, kIdInitialize, "init");
    auto get_texture = Resolve<GetMergedTextureSampler>(
        query, kIdGetMergedTextureSampler, "texture");
    auto get_surface = Resolve<GetIndependentDescriptor>(
        query, kIdGetIndependentDescriptor, "surface");
    auto create_module =
        Resolve<CreateCuModule>(query, kIdCreateCuModule, "module");
    auto create_function =
        Resolve<CreateCuFunction>(query, kIdCreateCuFunction, "function");
    auto launch =
        Resolve<LaunchKernelChain>(query, kIdLaunchKernelChain, "launch");
    RequireNvapi(initialize(), "init");

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 32;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heap_desc.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)),
              "heap");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_desc{};
    sampler_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_desc.NumDescriptors = 1;
    sampler_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_desc.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_desc,
                                               IID_PPV_ARGS(&sampler_heap)),
              "sampler heap");
    const UINT increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto heap_start = heap->GetCPUDescriptorHandleForHeapStart();
    const auto descriptor = [&](UINT slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{heap_start.ptr +
                                         std::uint64_t{slot} * increment};
    };
    const auto sampler = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC sampler_value{};
    sampler_value.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_value.AddressU = sampler_value.AddressV = sampler_value.AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&sampler_value, sampler);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    const auto input_descriptor = descriptor(14);
    gpu.device->CreateShaderResourceView(input_texture.resource.Get(), &srv,
                                         input_descriptor);
    MergedTextureSamplerParams texture_params{};
    texture_params.struct_size_in = texture_params.struct_size_out =
        sizeof(texture_params);
    texture_params.device = gpu.device.Get();
    texture_params.texture_descriptor.ptr = input_descriptor.ptr;
    texture_params.sampler_descriptor.ptr = sampler.ptr;
    RequireNvapi(get_texture(&texture_params), "texture handle");

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    const auto output_descriptor = descriptor(15);
    gpu.device->CreateUnorderedAccessView(output_texture.resource.Get(),
                                          nullptr, &uav, output_descriptor);
    IndependentDescriptorParams surface_params{};
    surface_params.struct_size_in = surface_params.struct_size_out =
        sizeof(surface_params);
    surface_params.device = gpu.device.Get();
    surface_params.type = IndependentDescriptorType::kSurface;
    surface_params.descriptor.ptr = output_descriptor.ptr;
    RequireNvapi(get_surface(&surface_params), "surface handle");

    NvObjectHandle module_object = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module.data(),
                               static_cast<std::uint32_t>(module.size()),
                               &module_object),
                 "create module");
    RequireNvapi(create_function(gpu.device.Get(), module_object, "main_kernel",
                                 &function),
                 "create function");
    std::array<std::uint8_t, 56> packed{};
    Pack64(packed, 0, texture_params.texture_handle);
    Pack32(packed, 8, 0);
    Pack32(packed, 12, 0);
    Pack32(packed, 16, m10::kWidth);
    Pack32(packed, 20, m10::kHeight);
    Pack32(packed, 24, 0x3b4ccccdu);
    Pack32(packed, 28, 0x3bb60b61u);
    Pack64(packed, 32, surface_params.handle);
    Pack32(packed, 40, m10::kWidth);
    Pack32(packed, 44, m10::kHeight);
    Pack32(packed, 48, 0x00000000u);
    Pack32(packed, 52, 0x3f800000u);
    auto address_free = packed;
    Pack64(address_free, 0, 0);
    Pack64(address_free, 32, 0);
    if (Sha256(std::vector<std::uint8_t>(address_free.begin(),
                                         address_free.end())) !=
        m10::kFixtureAddressFreeParameterSha256) {
      throw std::runtime_error("module10 ABI");
    }
    KernelLaunchParams launch_params{
        function, {20, 12, 1},   {16, 16, 1},
        0,        packed.data(), static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch(gpu.list.Get(), &launch_params, 1), "launch");
    ExecuteAndReset(gpu);

    const auto input_after = ReadModule10Texture(gpu, input_texture);
    const auto output = ReadModule10Texture(gpu, output_texture);
    const auto canary_after = ReadModule10Texture(gpu, canary_texture);
    const std::size_t input_mismatches = Mismatches(input, input_after);
    const std::size_t canary_mismatches =
        Mismatches(canary_bytes, canary_after);
    Write(output_root / "output.rgba16f", output);
    std::ofstream receipt(output_root / "run-receipt.json");
    receipt << "{\n\"schema\":\"neuralstudio.fg.distortion-module10-run.v1\","
               "\n\"backend\":\"rtx-d3d12-nvapi\",\n\"run_id\":\""
            << run_id << "\",\n\"process_id\":" << GetCurrentProcessId()
            << ",\n\"poison_byte\":" << unsigned(poison)
            << ",\n\"module_sha256\":\"" << m10::kModuleSha256
            << "\",\n\"module_bytes\":" << m10::kModuleBytes
            << ",\n\"selected_launch_tokens\":[\"launch-000005\",\"launch-"
               "000016\"],"
               "\n\"selected_phase_raw_parameter_sha256\":[\""
            << m10::kResetRawParameterSha256 << "\",\""
            << m10::kGeneratedRawParameterSha256
            << "\"],\n\"selected_phase_address_free_parameter_sha256\":[\""
            << m10::kResetAddressFreeParameterSha256 << "\",\""
            << m10::kGeneratedAddressFreeParameterSha256
            << "\"],\n\"packed_parameter_bytes\":56,"
               "\n\"fixture_address_free_parameter_sha256\":\""
            << m10::kFixtureAddressFreeParameterSha256
            << "\",\n\"input_generator\":\"" << m10::kInputGenerator
            << "\",\n\"input_pre_sha256\":\"" << Sha256(input)
            << "\",\n\"input_post_sha256\":\"" << Sha256(input_after)
            << "\",\n\"input_mismatched_bytes\":" << input_mismatches
            << ",\n\"texture_canary_mismatched_bytes\":" << canary_mismatches
            << ",\n\"output_bytes\":" << output.size()
            << ",\n\"output_sha256\":\"" << Sha256(output)
            << "\",\n\"output_poison_pixels\":"
            << CountPoisonPixels(output, 8, poison)
            << ",\n\"typed_d3d12_resources\":true,"
               "\n\"complete_write_coverage_checked_by_dual_poison\":true,"
               "\n\"expected_output_fed_to_execution\":false,"
               "\n\"archived_intermediates_consumed\":false,"
               "\n\"address_fields_recorded\":false,\n\"result\":\""
            << ((input_mismatches | canary_mismatches) ? "fail" : "pass")
            << "\"\n}\n";
    return (input_mismatches | canary_mismatches) ? 3 : 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
#endif
