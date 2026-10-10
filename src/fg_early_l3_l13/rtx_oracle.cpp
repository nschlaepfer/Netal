#define main FgMidL55EmbeddedMain
#include "../fg_mid_l55/rtx_oracle.cpp"
#undef main

#include "contract.h"

namespace {

namespace contract = fg_early_l3_l13;

template <std::size_t N>
void L3Pack64(std::array<std::uint8_t, N>* bytes, std::size_t offset,
              std::uint64_t value) {
  for (std::size_t index = 0; index < 8u; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
}

template <std::size_t N>
void L3Pack32(std::array<std::uint8_t, N>* bytes, std::size_t offset,
              std::uint32_t value) {
  for (std::size_t index = 0; index < 4u; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
}

GpuTexture UploadTypedTexture(GpuContext& gpu, DXGI_FORMAT format,
                              std::uint32_t width, std::uint32_t height,
                              std::uint32_t bytes_per_pixel,
                              D3D12_RESOURCE_FLAGS flags,
                              D3D12_RESOURCE_STATES final_state,
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

std::vector<std::uint8_t> Join(
    std::initializer_list<const std::vector<std::uint8_t>*> values) {
  std::vector<std::uint8_t> result;
  for (const auto* value : values)
    result.insert(result.end(), value->begin(), value->end());
  return result;
}

std::size_t PoisonWords(const std::vector<std::uint8_t>& bytes,
                        std::uint8_t poison, std::size_t word_bytes) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += word_bytes) {
    bool match = true;
    for (std::size_t index = 0; index < word_bytes; ++index)
      match = match && bytes[offset + index] == poison;
    result += match;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-early-l3-l13-rtx <module> <output-dir> <run-id> "
                 "<poison-hex> <l3-evaluate-0|l13-evaluate-1> "
                 "<reset-history|generated-history>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const auto poison = ParsePoison(argv[4]);
    const std::string phase = argv[5];
    const std::string fixture = argv[6];
    const auto selected_launch = contract::SelectedLaunch(phase);
    fs::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256)
      throw std::runtime_error("selected module11 identity mismatch");
    const auto source_before = contract::GenerateSource();
    const auto history_before = contract::GenerateHistory(fixture);
    const auto half_color_initial =
        contract::Poison(contract::kHalfColorBytes, poison);
    const auto half_metric_initial =
        contract::Poison(contract::kHalfMetricBytes, poison);
    const auto next_history_initial =
        contract::Poison(contract::kHistoryBytes, poison);
    const std::vector<std::uint8_t> canary_before(
        contract::kCanaryBytes, contract::kCanaryValue);

    GpuContext gpu = CreateGpuContext();
    auto source = UploadTypedTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, contract::kWidth,
        contract::kHeight, 4, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, source_before);
    auto history = UploadTypedTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, contract::kWidth,
        contract::kHeight, 8, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, history_before);
    auto half_color = UploadTypedTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, contract::kHalfWidth,
        contract::kHalfHeight, 8,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, half_color_initial);
    auto half_metric = UploadTypedTexture(
        gpu, DXGI_FORMAT_R16_FLOAT, contract::kHalfWidth,
        contract::kHalfHeight, 2,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, half_metric_initial);
    auto next_history = UploadTypedTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, contract::kWidth,
        contract::kHeight, 8,
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
    resource_heap_description.NumDescriptors = 8;
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
    const auto resource_start =
        resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto resource_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto sampler_start =
        sampler_heap->GetCPUDescriptorHandleForHeapStart();
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
    source_sampler.Filter = phase == "l3-evaluate-0"
        ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
        : D3D12_FILTER_MIN_MAG_MIP_POINT;
    source_sampler.AddressU = source_sampler.AddressV =
        source_sampler.AddressW = phase == "l3-evaluate-0"
            ? D3D12_TEXTURE_ADDRESS_MODE_CLAMP
            : D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    source_sampler.MaxAnisotropy = phase == "l3-evaluate-0" ? 1u : 0u;
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
      RequireNvapi(get_texture(&parameters),
                   "GetCudaMergedTextureSamplerObject");
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

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module),
                 "CreateCuModule(module11)");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function),
                 "CreateCuFunction(module11)");
    std::array<std::uint8_t, contract::kPackedParameterBytes> packed{};
    L3Pack64(&packed, 0, source_handle);
    L3Pack64(&packed, 80, history_handle);
    L3Pack64(&packed, 96, half_color_handle);
    L3Pack64(&packed, 104, half_metric_handle);
    L3Pack64(&packed, 120, next_history_handle);
    L3Pack32(&packed, 128, 2);
    L3Pack32(&packed, 136, contract::kWidth);
    L3Pack32(&packed, 140, contract::kHeight);
    L3Pack32(&packed, 144, contract::kHalfWidth);
    L3Pack32(&packed, 148, contract::kHalfHeight);
    const KernelLaunchParams launch{
        function, {40, 23, 1}, {256, 1, 1}, 0, packed.data(),
        static_cast<std::uint32_t>(packed.size())};
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
    const auto combined = Join(
        {&half_color_after, &half_metric_after, &next_history_after});
    const std::size_t input_mismatches =
        Mismatches(source_before, source_after) +
        Mismatches(history_before, history_after);
    const std::size_t canary_mismatches =
        Mismatches(canary_before, canary_left_after) +
        Mismatches(canary_before, canary_right_after);
    const std::size_t poison_words =
        PoisonWords(half_color_after, poison, 2) +
        PoisonWords(half_metric_after, poison, 2) +
        PoisonWords(next_history_after, poison, 2);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    Write(output_directory / "half-color.rgba16f", half_color_after);
    Write(output_directory / "half-metric.r16f", half_metric_after);
    Write(output_directory / "next-history.rgba16f", next_history_after);
    Write(output_directory / "output.bin", combined);
    std::ofstream receipt(output_directory / "run-receipt.json");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.early-l3-l13-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"phase\": \"" << phase << "\",\n"
            << "  \"fixture\": \"" << fixture << "\",\n"
            << "  \"selected_launch\": " << selected_launch << ",\n"
            << "  \"module_token\": 11,\n"
            << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
            << "  \"raw_parameter_sha256\": \""
            << (selected_launch == 3 ? contract::kL3RawParameterSha256
                                     : contract::kL13RawParameterSha256)
            << "\",\n"
            << "  \"address_free_parameter_sha256\": \""
            << (selected_launch == 3 ? contract::kL3AddressFreeParameterSha256
                                     : contract::kL13AddressFreeParameterSha256)
            << "\",\n"
            << "  \"history_read_bank\": \""
            << contract::HistoryReadBank(phase) << "\",\n"
            << "  \"history_write_bank\": \""
            << contract::HistoryWriteBank(phase) << "\",\n"
            << "  \"input_generator\": \"" << contract::kGenerator << "\",\n"
            << "  \"input_sha256\": [\"" << Sha256(source_before)
            << "\", \"" << Sha256(history_before) << "\"],\n"
            << "  \"output_sha256\": {\"half_color\": \""
            << Sha256(half_color_after) << "\", \"half_metric\": \""
            << Sha256(half_metric_after) << "\", \"next_history\": \""
            << Sha256(next_history_after) << "\"},\n"
            << "  \"combined_output_bytes\": " << combined.size() << ",\n"
            << "  \"combined_output_sha256\": \"" << Sha256(combined)
            << "\",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
            << "  \"poison_pattern_halfwords\": " << poison_words << ",\n"
            << "  \"typed_resource_backend\": true,\n"
            << "  \"expected_or_archived_tensor_ingress\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (pass ? "pass" : "fail")
            << "\"\n}\n";
    std::cout << "phase=" << phase << '\n'
              << "fixture=" << fixture << '\n'
              << "combined_output_sha256=" << Sha256(combined) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
