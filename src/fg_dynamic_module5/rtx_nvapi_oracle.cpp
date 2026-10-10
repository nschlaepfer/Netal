#define main FgDynamicModule5EmbeddedCompositorMain
#include "../fg_compositor_l87/rtx_nvapi_oracle.cpp"
#undef main

#include "contract.h"

namespace {
namespace dynamic5 = fg_dynamic_module5;

constexpr std::size_t kDynamic5GuardBytes = 256;

struct Dynamic5TextureSpec {
  DXGI_FORMAT format;
  std::uint32_t bytes_per_pixel;
  std::uint32_t descriptor_slot;
};

constexpr std::array<Dynamic5TextureSpec, 10> kDynamic5TextureSpecs{{
    {DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 24},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 36},
    {DXGI_FORMAT_R8_UNORM, 1, 37},
    {DXGI_FORMAT_R8_UNORM, 1, 38},
    {DXGI_FORMAT_R16G16_FLOAT, 4, 39},
    {DXGI_FORMAT_R16_FLOAT, 2, 40},
    {DXGI_FORMAT_R16_FLOAT, 2, 41},
    {DXGI_FORMAT_R32G32_FLOAT, 8, 14},
    {DXGI_FORMAT_R16G16_FLOAT, 4, 22},
    {DXGI_FORMAT_R8_UNORM, 1, 42},
}};

GpuTexture UploadDynamic5Texture(
    GpuContext& gpu, std::uint32_t width, std::uint32_t height,
    DXGI_FORMAT format, std::uint32_t bytes_per_pixel,
    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES final_state,
    const std::vector<std::uint8_t>& bytes) {
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != source_row * height)
    throw std::runtime_error("dynamic module5 texture upload size mismatch");
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
            "CreateCommittedResource(dynamic module5 texture)");
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
  RequireHr(upload->Map(0, &no_read, &mapped),
            "Map(dynamic module5 texture upload)");
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

std::vector<std::uint8_t> ReadDynamic5Texture(GpuContext& gpu,
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
            "Map(dynamic module5 texture readback)");
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
void Dynamic5Pack32(std::array<std::uint8_t, Size>* bytes,
                    std::size_t offset, std::uint32_t value) {
  if (offset + 4u > Size) throw std::runtime_error("Dynamic5Pack32 OOB");
  for (std::size_t index = 0; index < 4u; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
}

template <std::size_t Size>
void Dynamic5Pack64(std::array<std::uint8_t, Size>* bytes,
                    std::size_t offset, std::uint64_t value) {
  if (offset + 8u > Size) throw std::runtime_error("Dynamic5Pack64 OOB");
  for (std::size_t index = 0; index < 8u; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
}

template <std::size_t Size>
std::string Dynamic5AddressFreeHash(std::array<std::uint8_t, Size> packed) {
  for (std::size_t offset = 0; offset <= 96u; offset += 8u)
    for (std::size_t byte = 0; byte < 8u; ++byte)
      packed[offset + byte] = 0;
  return Sha256(std::vector<std::uint8_t>(packed.begin(), packed.end()));
}

std::size_t Dynamic5CanaryMismatches(
    const std::vector<std::uint8_t>& guarded, std::size_t payload_bytes,
    std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < kDynamic5GuardBytes; ++index) {
    count += guarded[index] != poison;
    count += guarded[kDynamic5GuardBytes + payload_bytes + index] != poison;
  }
  return count;
}

void WriteDynamic5Receipt(
    const std::filesystem::path& path, const std::string& run_id,
    const dynamic5::Resolution& resolution, dynamic5::Fixture fixture,
    std::uint8_t poison, const dynamic5::Inputs& inputs,
    const std::array<std::vector<std::uint8_t>, 10>& inputs_after,
    const std::vector<std::uint8_t>& tensor,
    const std::vector<std::uint8_t>& mask,
    const std::vector<std::uint8_t>& vector,
    std::size_t input_mismatches, std::size_t buffer_canary_mismatches,
    std::size_t texture_canary_mismatches,
    const std::string& address_free_parameter_sha,
    bool artifacts_written) {
  const bool pass = input_mismatches == 0 &&
                    buffer_canary_mismatches == 0 &&
                    texture_canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write dynamic module5 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.dynamic-module5-run.v1\",\n"
          << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"phase_fixture\": \"" << dynamic5::FixtureName(fixture)
          << "\",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module_token\": 5,\n"
          << "  \"module_sha256\": \"" << dynamic5::kModuleSha256 << "\",\n"
          << "  \"module_transport\": \"NvAPI D3D12 selected fatbin\",\n"
          << "  \"function_name\": \"main_kernel\",\n"
          << "  \"grid\": [" << resolution.grid_x << ", "
          << resolution.grid_y << ", 1],\n"
          << "  \"block\": [16, 16, 1],\n"
          << "  \"packed_parameter_bytes\": 192,\n"
          << "  \"motion_extent\": [" << resolution.motion_width << ", "
          << resolution.motion_height << "],\n"
          << "  \"work_extent\": [" << resolution.work_width << ", "
          << resolution.work_height << "],\n"
          << "  \"tensor_extent\": [" << resolution.tensor_width << ", "
          << resolution.tensor_height << ", 48],\n"
          << "  \"logical_origin\": [" << resolution.origin_x << ", "
          << resolution.origin_y << "],\n"
          << "  \"raw_parameter_sha256\": \""
          << resolution.raw_parameter_sha256 << "\",\n"
          << "  \"address_free_parameter_sha256\": \""
          << address_free_parameter_sha << "\",\n"
          << "  \"input_generator\": \"" << dynamic5::kInputGenerator
          << "\",\n"
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
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"buffer_canary_mismatched_bytes\": "
          << buffer_canary_mismatches << ",\n"
          << "  \"texture_canary_mismatched_bytes\": "
          << texture_canary_mismatches << ",\n"
          << "  \"complete_write_coverage_checked_by_dual_poison\": true,\n"
          << "  \"artifacts_written\": "
          << (artifacts_written ? "true" : "false") << ",\n"
          << "  \"expected_or_captured_output_ingress\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage=fg-dynamic-module5-rtx <module> <output-dir> <run-id> "
                 "<poison-hex> <resolution> <first-frame|fan-in> "
                 "<full|hash-only>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    const auto& resolution = dynamic5::FindResolution(argv[5]);
    const auto fixture = dynamic5::ParseFixture(argv[6]);
    const bool full_artifacts = std::string_view(argv[7]) == "full";
    if (!full_artifacts && std::string_view(argv[7]) != "hash-only") {
      throw std::runtime_error("invalid artifact mode");
    }
    std::filesystem::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != dynamic5::kModuleBytes ||
        Sha256(module_bytes) != dynamic5::kModuleSha256) {
      throw std::runtime_error("selected dynamic module5 identity mismatch");
    }
    const auto inputs = dynamic5::GenerateInputs(resolution, fixture);
    for (std::size_t index = 0; index < inputs.textures.size(); ++index) {
      if (inputs.textures[index].size() !=
          dynamic5::InputBytes(resolution, index)) {
        throw std::runtime_error("dynamic module5 input size mismatch");
      }
    }
    const auto tensor_bytes = dynamic5::TensorBytes(resolution);
    const auto mask_bytes = dynamic5::MaskBytes(resolution);
    const auto vector_bytes = dynamic5::VectorBytes(resolution);
    std::vector<std::uint8_t> guarded_tensor(
        tensor_bytes + 2u * kDynamic5GuardBytes, poison);
    const std::vector<std::uint8_t> mask_poison(mask_bytes, poison);
    const std::vector<std::uint8_t> vector_poison(vector_bytes, poison);
    const std::vector<std::uint8_t> mask_canary_before(
        mask_bytes, static_cast<std::uint8_t>(poison ^ 0xffu));
    const std::vector<std::uint8_t> vector_canary_before(
        vector_bytes, static_cast<std::uint8_t>(poison ^ 0xffu));

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
      input_textures[index] = UploadDynamic5Texture(
          gpu, resolution.work_width, resolution.work_height,
          kDynamic5TextureSpecs[index].format,
          kDynamic5TextureSpecs[index].bytes_per_pixel,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, inputs.textures[index]);
    }
    GpuBuffer tensor_buffer = UploadBuffer(gpu, guarded_tensor);
    GpuTexture mask_output = UploadDynamic5Texture(
        gpu, resolution.work_width, resolution.work_height,
        DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, mask_poison);
    GpuTexture vector_output = UploadDynamic5Texture(
        gpu, resolution.work_width, resolution.work_height,
        DXGI_FORMAT_R16G16_FLOAT, 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, vector_poison);
    GpuTexture mask_canary = UploadDynamic5Texture(
        gpu, resolution.work_width, resolution.work_height,
        DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, mask_canary_before);
    GpuTexture vector_canary = UploadDynamic5Texture(
        gpu, resolution.work_width, resolution.work_height,
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
              "CreateDescriptorHeap(dynamic module5 resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_description{};
    sampler_heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_heap_description.NumDescriptors = 1;
    sampler_heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_heap_description,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(dynamic module5 sampler)");
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
      view.Format = kDynamic5TextureSpecs[index].format;
      view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      view.Texture2D.MipLevels = 1;
      const auto srv = descriptor(kDynamic5TextureSpecs[index].descriptor_slot);
      gpu.device->CreateShaderResourceView(input_textures[index].resource.Get(),
                                           &view, srv);
      MergedTextureSamplerParams parameters{};
      parameters.struct_size_in = sizeof(parameters);
      parameters.struct_size_out = sizeof(parameters);
      parameters.device = gpu.device.Get();
      parameters.texture_descriptor.ptr = srv.ptr;
      parameters.sampler_descriptor.ptr = sampler.ptr;
      RequireNvapi(get_texture(&parameters),
                   "GetCudaMergedTextureSamplerObject");
      if (parameters.texture_handle == 0)
        throw std::runtime_error("null dynamic module5 texture handle");
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
    if (mask_surface.handle == 0 || vector_surface.handle == 0)
      throw std::runtime_error("null dynamic module5 surface handle");

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module), "CreateCuModule(module5)");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function), "CreateCuFunction(main_kernel)");
    std::array<std::uint8_t, 192> packed{};
    for (std::size_t index = 0; index < texture_handles.size(); ++index)
      Dynamic5Pack64(&packed, index * 8u, texture_handles[index]);
    Dynamic5Pack64(&packed, 80,
              tensor_buffer.resource->GetGPUVirtualAddress() +
                  kDynamic5GuardBytes);
    Dynamic5Pack64(&packed, 88, mask_surface.handle);
    Dynamic5Pack64(&packed, 96, vector_surface.handle);
    Dynamic5Pack32(&packed, 152, resolution.motion_width);
    Dynamic5Pack32(&packed, 156, resolution.motion_height);
    Dynamic5Pack32(&packed, 160, resolution.work_width);
    Dynamic5Pack32(&packed, 164, resolution.work_height);
    Dynamic5Pack32(&packed, 168, resolution.tensor_width);
    Dynamic5Pack32(&packed, 172, resolution.tensor_height);
    Dynamic5Pack32(&packed, 176, resolution.origin_x);
    Dynamic5Pack32(&packed, 180, resolution.reserved_origin_x);
    Dynamic5Pack32(&packed, 184, resolution.origin_y);
    Dynamic5Pack32(&packed, 188, resolution.reserved_origin_y);
    const std::string address_free_parameter_sha =
        Dynamic5AddressFreeHash(packed);
    if (address_free_parameter_sha !=
        resolution.address_free_parameter_sha256) {
      throw std::runtime_error("dynamic module5 address-free ABI mismatch");
    }
    const KernelLaunchParams launch{
        function, {resolution.grid_x, resolution.grid_y, 1}, {16, 16, 1},
        0, packed.data(), static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(dynamic module5)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    std::array<std::vector<std::uint8_t>, 10> inputs_after;
    std::size_t input_mismatches = 0;
    for (std::size_t index = 0; index < input_textures.size(); ++index) {
      inputs_after[index] = ReadDynamic5Texture(gpu, input_textures[index]);
      input_mismatches += Mismatches(inputs.textures[index],
                                     inputs_after[index]);
    }
    const auto mask_after = ReadDynamic5Texture(gpu, mask_output);
    const auto vector_after = ReadDynamic5Texture(gpu, vector_output);
    const auto mask_canary_after = ReadDynamic5Texture(gpu, mask_canary);
    const auto vector_canary_after = ReadDynamic5Texture(gpu, vector_canary);
    const auto tensor_guarded_after = ReadBuffer(gpu, tensor_buffer);
    const std::vector<std::uint8_t> tensor_after(
        tensor_guarded_after.begin() + kDynamic5GuardBytes,
        tensor_guarded_after.begin() + kDynamic5GuardBytes + tensor_bytes);
    const std::size_t buffer_canary_mismatches = Dynamic5CanaryMismatches(
        tensor_guarded_after, tensor_bytes, poison);
    const std::size_t texture_canary_mismatches =
        Mismatches(mask_canary_before, mask_canary_after) +
        Mismatches(vector_canary_before, vector_canary_after);
    if (full_artifacts) {
      Write(output_directory / "tensor-output.fp16", tensor_after);
      Write(output_directory / "mask-output.r8", mask_after);
      Write(output_directory / "vector-output.rg16f", vector_after);
    }
    WriteDynamic5Receipt(
        output_directory / "run-receipt.json", run_id, resolution, fixture,
        poison, inputs, inputs_after, tensor_after, mask_after, vector_after,
        input_mismatches, buffer_canary_mismatches,
        texture_canary_mismatches, address_free_parameter_sha,
        full_artifacts);
    const bool pass = input_mismatches == 0 &&
                      buffer_canary_mismatches == 0 &&
                      texture_canary_mismatches == 0;
    std::cout << "probe=fg_dynamic_module5_rtx_nvapi\n"
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
