#define main FgDynamicModule11EmbeddedMain
#include "../fg_mid_l55/rtx_oracle.cpp"
#undef main

#include <algorithm>

#include "contract.h"

namespace {
namespace dynamic_contract = fg_dynamic_module11;

GpuTexture DynamicUploadTypedTexture(
    GpuContext& gpu, DXGI_FORMAT format, std::uint32_t width,
    std::uint32_t height, std::uint32_t bytes_per_pixel,
    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES final_state,
    const std::vector<std::uint8_t>& bytes) {
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != source_row * height)
    throw std::runtime_error("typed texture byte size mismatch");
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
  description.Flags = flags;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.resource)),
            "CreateCommittedResource(typed texture)");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0, upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &upload_size);
  auto upload = CreateBuffer(gpu.device.Get(), upload_size,
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(typed texture upload)");
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

template <std::size_t Size>
void DynamicPack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
                   std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void DynamicPack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
                   std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> DynamicJoin(
    std::initializer_list<const std::vector<std::uint8_t>*> values) {
  std::vector<std::uint8_t> result;
  std::size_t bytes = 0;
  for (const auto* value : values) bytes += value->size();
  result.reserve(bytes);
  for (const auto* value : values)
    result.insert(result.end(), value->begin(), value->end());
  return result;
}

std::size_t DynamicPoisonHalfwords(const std::vector<std::uint8_t>& bytes,
                                   std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 2u)
    result += bytes[offset] == poison && bytes[offset + 1u] == poison;
  return result;
}

void DynamicWriteReceipt(
    const fs::path& path, const std::string& run_id,
    const dynamic_contract::Resolution& resolution, bool generated,
    std::uint8_t poison, bool artifacts_written,
    const std::vector<std::uint8_t>& source_before,
    const std::vector<std::uint8_t>& history_before,
    const std::vector<std::uint8_t>& half_color,
    const std::vector<std::uint8_t>& half_metric,
    const std::vector<std::uint8_t>& next_history,
    const std::vector<std::uint8_t>& combined,
    std::size_t input_mismatches, std::size_t canary_mismatches) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write module11 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.dynamic-module11-run.v1\",\n"
          << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"phase\": \"" << (generated ? "generated" : "reset")
          << "\",\n"
          << "  \"large_trace_launch\": " << (generated ? 15 : 3) << ",\n"
          << "  \"canonical_launch\": " << (generated ? 13 : 3) << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module_token\": 11,\n"
          << "  \"module_sha256\": \"" << dynamic_contract::kModuleSha256
          << "\",\n"
          << "  \"packed_parameter_bytes\": 168,\n"
          << "  \"selected_mode\": " << resolution.mode << ",\n"
          << "  \"source_extent\": [" << resolution.source_width << ", "
          << resolution.source_height << "],\n"
          << "  \"output_extent\": [" << resolution.output_width << ", "
          << resolution.output_height << "],\n"
          << "  \"grid\": [" << resolution.grid_x << ", "
          << resolution.grid_y << ", 1],\n"
          << "  \"block\": [256, 1, 1],\n"
          << "  \"trace_raw_parameter_sha256\": \""
          << (generated ? resolution.generated_raw_parameter_sha256
                        : resolution.reset_raw_parameter_sha256)
          << "\",\n"
          << "  \"trace_address_free_parameter_sha256\": \""
          << resolution.address_free_parameter_sha256 << "\",\n"
          << "  \"history_read_bank\": \""
          << dynamic_contract::HistoryReadBank(generated) << "\",\n"
          << "  \"history_write_bank\": \""
          << dynamic_contract::HistoryWriteBank(generated) << "\",\n"
          << "  \"input_generator\": \"" << dynamic_contract::kGenerator
          << "\",\n"
          << "  \"source_pre_sha256\": \"" << Sha256(source_before)
          << "\",\n"
          << "  \"history_pre_sha256\": \"" << Sha256(history_before)
          << "\",\n"
          << "  \"half_color_sha256\": \"" << Sha256(half_color) << "\",\n"
          << "  \"half_metric_sha256\": \"" << Sha256(half_metric) << "\",\n"
          << "  \"next_history_sha256\": \"" << Sha256(next_history)
          << "\",\n"
          << "  \"combined_output_bytes\": " << combined.size() << ",\n"
          << "  \"combined_output_sha256\": \"" << Sha256(combined)
          << "\",\n"
          << "  \"poison_pattern_halfwords\": "
          << DynamicPoisonHalfwords(combined, poison) << ",\n"
          << "  \"artifacts_written\": "
          << (artifacts_written ? "true" : "false") << ",\n"
          << "  \"typed_resource_backend\": true,\n"
          << "  \"surface_half_rounding\": \"toward-zero\",\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
          << "  \"expected_or_captured_output_ingress\": false,\n"
          << "  \"archived_intermediate_ingress\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage=fg-dynamic-module11-rtx <module> <output-dir> "
                 "<run-id> <poison-hex> <resolution> <reset|generated> "
                 "<full|hash-only>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != dynamic_contract::kModuleBytes ||
        Sha256(module_bytes) != dynamic_contract::kModuleSha256)
      throw std::runtime_error("selected module11 identity mismatch");
    const fs::path output_root = argv[2];
    const std::string run_id = argv[3];
    const auto poison = ParsePoison(argv[4]);
    const auto& resolution = dynamic_contract::ParseResolution(argv[5]);
    const std::string phase = argv[6];
    if (phase != "reset" && phase != "generated")
      throw std::runtime_error("invalid phase");
    const bool generated = phase == "generated";
    const std::string artifact_mode = argv[7];
    if (artifact_mode != "full" && artifact_mode != "hash-only")
      throw std::runtime_error("invalid artifact mode");
    const bool artifacts_written = artifact_mode == "full";
    fs::create_directories(output_root);
    const auto source_before = dynamic_contract::Source(resolution, generated);
    const auto history_before = dynamic_contract::History(resolution, generated);
    const auto half_color_initial = dynamic_contract::Poison(
        dynamic_contract::HalfColorBytes(resolution), poison);
    const auto half_metric_initial = dynamic_contract::Poison(
        dynamic_contract::HalfMetricBytes(resolution), poison);
    const auto next_history_initial = dynamic_contract::Poison(
        dynamic_contract::HistoryBytes(resolution), poison);
    const std::vector<std::uint8_t> canary_before(
        dynamic_contract::kCanaryBytes, dynamic_contract::kCanaryValue);

    GpuContext gpu = CreateGpuContext();
    auto source = DynamicUploadTypedTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, source_before);
    auto history = DynamicUploadTypedTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.source_width,
        resolution.source_height, 8, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, history_before);
    auto half_color = DynamicUploadTypedTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.output_width,
        resolution.output_height, 8,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, half_color_initial);
    auto half_metric = DynamicUploadTypedTexture(
        gpu, DXGI_FORMAT_R16_FLOAT, resolution.output_width,
        resolution.output_height, 2,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, half_metric_initial);
    auto next_history = DynamicUploadTypedTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.source_width,
        resolution.source_height, 8,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, next_history_initial);
    GpuBuffer canary_left = UploadBuffer(gpu, canary_before);
    GpuBuffer canary_right = UploadBuffer(gpu, canary_before);
    ExecuteAndReset(gpu);

    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading real nvapi64.dll failed");
    const auto query = reinterpret_cast<QueryInterface>(
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

    D3D12_DESCRIPTOR_HEAP_DESC resource_heap_description{};
    resource_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_heap_description.NumDescriptors = 5;
    resource_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_heap_description,
                                                IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_description{};
    sampler_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_heap_description.NumDescriptors = 2;
    sampler_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_heap_description,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(samplers)");
    const auto resource_start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto resource_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto sampler_start = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    const auto sampler_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    auto resource_slot = [&](UINT index) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          resource_start.ptr + std::uint64_t{index} * resource_increment};
    };
    auto sampler_slot = [&](UINT index) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          sampler_start.ptr + std::uint64_t{index} * sampler_increment};
    };
    auto make_srv = [&](GpuTexture& texture, UINT index) {
      D3D12_SHADER_RESOURCE_VIEW_DESC description{};
      description.Format = texture.format;
      description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      description.Texture2D.MipLevels = 1;
      gpu.device->CreateShaderResourceView(texture.resource.Get(), &description,
                                           resource_slot(index));
    };
    auto make_uav = [&](GpuTexture& texture, UINT index) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
      description.Format = texture.format;
      description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr,
                                            &description, resource_slot(index));
    };
    make_srv(source, 0);
    make_srv(history, 1);
    make_uav(half_color, 2);
    make_uav(half_metric, 3);
    make_uav(next_history, 4);
    D3D12_SAMPLER_DESC source_sampler{};
    source_sampler.Filter = generated ? D3D12_FILTER_MIN_MAG_MIP_POINT
                                      : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    source_sampler.AddressU = source_sampler.AddressV =
        source_sampler.AddressW = generated ? D3D12_TEXTURE_ADDRESS_MODE_BORDER
                                            : D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    source_sampler.MaxAnisotropy = generated ? 0u : 1u;
    gpu.device->CreateSampler(&source_sampler, sampler_slot(0));
    D3D12_SAMPLER_DESC history_sampler{};
    history_sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    history_sampler.AddressU = history_sampler.AddressV =
        history_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&history_sampler, sampler_slot(1));
    auto texture_handle = [&](UINT resource_index, UINT sampler_index) {
      MergedTextureSamplerParams parameters{};
      parameters.struct_size_in = sizeof(parameters);
      parameters.struct_size_out = sizeof(parameters);
      parameters.device = gpu.device.Get();
      parameters.texture_descriptor.ptr = resource_slot(resource_index).ptr;
      parameters.sampler_descriptor.ptr = sampler_slot(sampler_index).ptr;
      RequireNvapi(get_texture(&parameters), "GetCudaMergedTextureSamplerObject");
      return parameters.texture_handle;
    };
    auto surface_handle = [&](UINT resource_index) {
      IndependentDescriptorParams parameters{};
      parameters.struct_size_in = sizeof(parameters);
      parameters.struct_size_out = sizeof(parameters);
      parameters.device = gpu.device.Get();
      parameters.type = IndependentDescriptorType::kSurface;
      parameters.descriptor.ptr = resource_slot(resource_index).ptr;
      RequireNvapi(get_surface(&parameters),
                   "GetCudaIndependentDescriptorObject");
      return parameters.handle;
    };
    const auto source_handle = texture_handle(0, 0);
    const auto history_handle = texture_handle(1, 1);
    const auto half_color_handle = surface_handle(2);
    const auto half_metric_handle = surface_handle(3);
    const auto next_history_handle = surface_handle(4);

    NvObjectHandle module = nullptr, function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module), "CreateCuModule(module11)");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function), "CreateCuFunction(module11)");
    std::array<std::uint8_t, dynamic_contract::kPackedParameterBytes> packed{};
    DynamicPack64(&packed, 0, source_handle);
    DynamicPack64(&packed, 80, history_handle);
    DynamicPack64(&packed, 96, half_color_handle);
    DynamicPack64(&packed, 104, half_metric_handle);
    DynamicPack64(&packed, 120, next_history_handle);
    DynamicPack32(&packed, 128, resolution.mode);
    DynamicPack32(&packed, 136, resolution.source_width);
    DynamicPack32(&packed, 140, resolution.source_height);
    DynamicPack32(&packed, 144, resolution.output_width);
    DynamicPack32(&packed, 148, resolution.output_height);
    const KernelLaunchParams launch{
        function, {resolution.grid_x, resolution.grid_y, 1}, {256, 1, 1}, 0,
        packed.data(), static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(module11)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction(module11)");
    RequireNvapi(destroy_module(gpu.device.Get(), module),
                 "DestroyCuModule(module11)");

    const auto source_after = ReadTexture(gpu, source);
    const auto history_after = ReadTexture(gpu, history);
    const auto half_color_after = ReadTexture(gpu, half_color);
    const auto half_metric_after = ReadTexture(gpu, half_metric);
    const auto next_history_after = ReadTexture(gpu, next_history);
    const auto canary_left_after = ReadBuffer(gpu, canary_left);
    const auto canary_right_after = ReadBuffer(gpu, canary_right);
    const auto combined = DynamicJoin(
        {&half_color_after, &half_metric_after, &next_history_after});
    const std::size_t input_mismatches =
        Mismatches(source_before, source_after) +
        Mismatches(history_before, history_after);
    const std::size_t canary_mismatches =
        Mismatches(canary_before, canary_left_after) +
        Mismatches(canary_before, canary_right_after);
    if (artifacts_written) {
      Write(output_root / "half-color.rgba16f", half_color_after);
      Write(output_root / "half-metric.r16f", half_metric_after);
      Write(output_root / "next-history.rgba16f", next_history_after);
      Write(output_root / "output.bin", combined);
    }
    DynamicWriteReceipt(output_root / "run-receipt.json", run_id, resolution,
                        generated, poison, artifacts_written, source_before,
                        history_before, half_color_after, half_metric_after,
                        next_history_after, combined, input_mismatches,
                        canary_mismatches);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_dynamic_module11_rtx\n"
              << "run_id=" << run_id << '\n'
              << "combined_output_sha256=" << Sha256(combined) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
