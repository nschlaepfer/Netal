#define main fg_compositor_l87_embedded_main
#include "../fg_compositor_l87/rtx_nvapi_oracle.cpp"
#undef main

#include "contract.h"

namespace {
namespace l11 = fg_early_l11;

template <std::size_t Size>
void L11Pack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
               std::uint32_t value) {
  if (offset + 4u > Size) throw std::runtime_error("L11Pack32 OOB");
  for (std::size_t index = 0; index < 4u; ++index) {
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
  }
}

template <std::size_t Size>
void L11Pack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
               std::uint64_t value) {
  if (offset + 8u > Size) throw std::runtime_error("L11Pack64 OOB");
  for (std::size_t index = 0; index < 8u; ++index) {
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
  }
}

std::size_t L11OutsideMismatches(const std::vector<std::uint8_t>& output,
                                 std::uint8_t poison) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kTextureHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kTextureWidth; ++x) {
      if (l11::InOutputRegion(x, y)) continue;
      const std::size_t offset =
          (std::size_t{y} * l11::kTextureWidth + x) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        count += output[offset + channel] != poison;
      }
    }
  }
  return count;
}

std::size_t L11RegionMismatches(const std::vector<std::uint8_t>& input,
                                const std::vector<std::uint8_t>& output) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kCopyWidth; ++x) {
      const std::size_t source =
          (std::size_t{y + l11::kInputOriginY} * l11::kTextureWidth +
           x + l11::kInputOriginX) * 4u;
      const std::size_t destination =
          (std::size_t{y + l11::kOutputOriginY} * l11::kTextureWidth +
           x + l11::kOutputOriginX) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        count += input[source + channel] != output[destination + channel];
      }
    }
  }
  return count;
}

std::size_t L11RegionPoisonPixels(const std::vector<std::uint8_t>& output,
                                  std::uint8_t poison) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kCopyWidth; ++x) {
      const std::size_t offset =
          (std::size_t{y + l11::kOutputOriginY} * l11::kTextureWidth +
           x + l11::kOutputOriginX) * 4u;
      count += output[offset] == poison && output[offset + 1u] == poison &&
               output[offset + 2u] == poison &&
               output[offset + 3u] == poison;
    }
  }
  return count;
}

void WriteL11Receipt(const fs::path& path, const std::string& run_id,
                     std::uint8_t poison,
                     const std::vector<std::uint8_t>& input_before,
                     const std::vector<std::uint8_t>& input_after,
                     const std::vector<std::uint8_t>& output,
                     const std::vector<std::uint8_t>& region,
                     std::size_t input_mismatches,
                     std::size_t region_mismatches,
                     std::size_t outside_mismatches,
                     std::size_t expected_mismatches,
                     std::size_t region_poison_pixels) {
  const bool pass = input_mismatches == 0 && region_mismatches == 0 &&
                    outside_mismatches == 0 && expected_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write L11 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.early-l11-run.v1\",\n"
          << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"selected_launch\": 11,\n"
          << "  \"module_token\": 4,\n"
          << "  \"module_bytes\": " << l11::kModuleBytes << ",\n"
          << "  \"module_sha256\": \"" << l11::kModuleSha256 << "\",\n"
          << "  \"module_transport\": \"NvAPI_D3D12_CreateCuModule exact selected fatbin\",\n"
          << "  \"function_name\": \"main_kernel\",\n"
          << "  \"grid\": [40, 23, 1],\n"
          << "  \"block\": [16, 16, 1],\n"
          << "  \"packed_parameter_bytes\": 40,\n"
          << "  \"frozen_selected_raw_parameter_sha256\": \""
          << l11::kFrozenRawParameterSha256 << "\",\n"
          << "  \"frozen_selected_address_free_parameter_sha256\": \""
          << l11::kFrozenAddressFreeParameterSha256 << "\",\n"
          << "  \"fixture_address_free_parameter_sha256\": \""
          << l11::kFixtureAddressFreeParameterSha256 << "\",\n"
          << "  \"resource_format\": \"DXGI_FORMAT_R8G8B8A8_UNORM\",\n"
          << "  \"resource_extent\": [640, 360],\n"
          << "  \"input_origin\": [17, 11],\n"
          << "  \"output_origin\": [29, 19],\n"
          << "  \"copy_extent\": [587, 317],\n"
          << "  \"input_generator\": \"" << l11::kInputGenerator << "\",\n"
          << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
          << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
          << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
          << "  \"region_output_sha256\": \"" << Sha256(region) << "\",\n"
          << "  \"output_bytes\": " << output.size() << ",\n"
          << "  \"declared_region_bytes\": " << region.size() << ",\n"
          << "  \"outside_region_pixels\": " << l11::kOutsidePixels << ",\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"declared_region_mismatched_bytes\": "
          << region_mismatches << ",\n"
          << "  \"outside_region_mismatched_bytes\": "
          << outside_mismatches << ",\n"
          << "  \"canary_mismatched_bytes\": " << outside_mismatches << ",\n"
          << "  \"post_execution_expected_mismatched_bytes\": "
          << expected_mismatches << ",\n"
          << "  \"declared_region_poison_pixels\": "
          << region_poison_pixels << ",\n"
          << "  \"input_immutable\": "
          << (input_mismatches == 0 ? "true" : "false") << ",\n"
          << "  \"expected_bytes_fed_to_execution\": false,\n"
          << "  \"archived_intermediates_consumed\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-early-l11-nvapi <module.bin> <output-dir> "
                 "<run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    fs::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != l11::kModuleBytes ||
        Sha256(module_bytes) != l11::kModuleSha256) {
      throw std::runtime_error("selected module4 identity mismatch");
    }
    const auto input_before = l11::GenerateInput();
    const std::vector<std::uint8_t> output_before(l11::kTextureBytes, poison);

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading real nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (query == nullptr) throw std::runtime_error("nvapi_QueryInterface missing");
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

    GpuTexture input_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, input_before);
    GpuTexture output_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output_before);
    ExecuteAndReset(gpu);

    D3D12_DESCRIPTOR_HEAP_DESC resource_heap_description{};
    resource_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_heap_description.NumDescriptors = 4;
    resource_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_heap_description,
                                                IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_description{};
    sampler_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_heap_description.NumDescriptors = 1;
    sampler_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_heap_description,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(sampler)");
    const UINT increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto resource_start =
        resource_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE srv{resource_start.ptr};
    D3D12_CPU_DESCRIPTOR_HANDLE uav{resource_start.ptr + increment};
    D3D12_CPU_DESCRIPTOR_HANDLE sampler =
        sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_description{};
    srv_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv_description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_description.Shader4ComponentMapping =
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_description.Texture2D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(input_texture.resource.Get(),
                                         &srv_description, srv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_description{};
    uav_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav_description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(output_texture.resource.Get(), nullptr,
                                          &uav_description, uav);
    D3D12_SAMPLER_DESC sampler_description{};
    sampler_description.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_description.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&sampler_description, sampler);

    MergedTextureSamplerParams texture_parameters{};
    texture_parameters.struct_size_in = sizeof(texture_parameters);
    texture_parameters.struct_size_out = sizeof(texture_parameters);
    texture_parameters.device = gpu.device.Get();
    texture_parameters.texture_descriptor.ptr = srv.ptr;
    texture_parameters.sampler_descriptor.ptr = sampler.ptr;
    RequireNvapi(get_texture(&texture_parameters),
                 "GetCudaMergedTextureSamplerObject");
    IndependentDescriptorParams surface_parameters{};
    surface_parameters.struct_size_in = sizeof(surface_parameters);
    surface_parameters.struct_size_out = sizeof(surface_parameters);
    surface_parameters.device = gpu.device.Get();
    surface_parameters.type = IndependentDescriptorType::kSurface;
    surface_parameters.descriptor.ptr = uav.ptr;
    RequireNvapi(get_surface(&surface_parameters),
                 "GetCudaIndependentDescriptorObject");
    if (texture_parameters.texture_handle == 0 ||
        surface_parameters.handle == 0) {
      throw std::runtime_error("NvAPI returned null L11 resource handle");
    }

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module),
                 "CreateCuModule(module4)");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function),
                 "CreateCuFunction(main_kernel)");
    std::array<std::uint8_t, l11::kPackedParameterBytes> packed{};
    L11Pack64(&packed, 0, texture_parameters.texture_handle);
    L11Pack32(&packed, 8, l11::kInputOriginX);
    L11Pack32(&packed, 12, l11::kInputOriginY);
    L11Pack64(&packed, 16, surface_parameters.handle);
    L11Pack32(&packed, 24, l11::kOutputOriginX);
    L11Pack32(&packed, 28, l11::kOutputOriginY);
    L11Pack32(&packed, 32, l11::kCopyWidth);
    L11Pack32(&packed, 36, l11::kCopyHeight);
    const KernelLaunchParams launch{
        function, {40, 23, 1}, {16, 16, 1}, 0, packed.data(),
        static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(module4/main_kernel)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto input_after = ReadTexture(gpu, input_texture);
    const auto output_after = ReadTexture(gpu, output_texture);
    const auto expected = l11::ExpectedOutput(input_before, poison);
    const auto region = l11::ExtractRegion(output_after);
    const std::size_t input_mismatches = Mismatches(input_before, input_after);
    const std::size_t region_mismatches =
        L11RegionMismatches(input_before, output_after);
    const std::size_t outside_mismatches =
        L11OutsideMismatches(output_after, poison);
    const std::size_t expected_mismatches = Mismatches(expected, output_after);
    const std::size_t region_poison_pixels =
        L11RegionPoisonPixels(output_after, poison);
    Write(output_directory / "input-after.rgba8", input_after);
    Write(output_directory / "output.rgba8", output_after);
    Write(output_directory / "region-output.rgba8", region);
    WriteL11Receipt(output_directory / "run-receipt.json", run_id, poison,
                    input_before, input_after, output_after, region,
                    input_mismatches, region_mismatches, outside_mismatches,
                    expected_mismatches, region_poison_pixels);
    const bool pass = input_mismatches == 0 && region_mismatches == 0 &&
                      outside_mismatches == 0 && expected_mismatches == 0;
    std::cout << "probe=fg_early_l11_nvapi\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "region_output_sha256=" << Sha256(region) << '\n'
              << "output_sha256=" << Sha256(output_after) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "declared_region_mismatched_bytes=" << region_mismatches
              << '\n'
              << "outside_region_mismatched_bytes=" << outside_mismatches
              << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
