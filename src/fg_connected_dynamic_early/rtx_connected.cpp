#define main FgConnectedDynamicEarlyEmbeddedBaseMain
#include "../fg_mid_l55/rtx_oracle.cpp"
#undef main

using uint = std::uint32_t;
#include "contract.h"

namespace {
namespace dyn = fg_connected_dynamic_early;

GpuTexture DynamicUploadTexture(
    GpuContext& gpu, DXGI_FORMAT format, std::uint32_t width,
    std::uint32_t height, std::uint32_t bytes_per_pixel,
    D3D12_RESOURCE_STATES final_state,
    const std::vector<std::uint8_t>& bytes) {
  const std::size_t row_bytes = std::size_t{width} * bytes_per_pixel;
  if (bytes.size() != row_bytes * height)
    throw std::runtime_error("dynamic texture size drift");
  GpuTexture result;
  result.width = width;
  result.height = height;
  result.bytes_per_pixel = bytes_per_pixel;
  result.format = format;
  result.state = final_state;
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
                &heap, D3D12_HEAP_FLAG_NONE, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&result.resource)),
            "CreateCommittedResource(dynamic texture)");
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
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(dynamic texture upload)");
  for (std::uint32_t row = 0; row < height; ++row)
    std::memcpy(static_cast<std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                bytes.data() + std::size_t{row} * row_bytes, row_bytes);
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
void DPack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
             std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void DPack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
             std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> DJoin(
    std::initializer_list<std::vector<std::uint8_t>> values) {
  std::vector<std::uint8_t> result;
  std::size_t total = 0;
  for (const auto& value : values) total += value.size();
  result.reserve(total);
  for (const auto& value : values)
    result.insert(result.end(), value.begin(), value.end());
  return result;
}

void DRestore(GpuContext& gpu, GpuTexture& texture,
              D3D12_RESOURCE_STATES state) {
  Transition(gpu.list.Get(), texture.resource.Get(), texture.state, state);
  texture.state = state;
}

void DRestore(GpuContext& gpu, GpuBuffer& buffer,
              D3D12_RESOURCE_STATES state) {
  Transition(gpu.list.Get(), buffer.resource.Get(), buffer.state, state);
  buffer.state = state;
}

void DOverwriteBuffer(GpuContext& gpu, GpuBuffer& buffer,
                      const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != buffer.size)
    throw std::runtime_error("dynamic buffer overwrite size drift");
  auto upload = CreateBuffer(gpu.device.Get(), bytes.size(),
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(dynamic overwrite)");
  std::memcpy(mapped, bytes.data(), bytes.size());
  upload->Unmap(0, nullptr);
  Transition(gpu.list.Get(), buffer.resource.Get(), buffer.state,
             D3D12_RESOURCE_STATE_COPY_DEST);
  gpu.list->CopyBufferRegion(buffer.resource.Get(), 0, upload.Get(), 0,
                             bytes.size());
  Transition(gpu.list.Get(), buffer.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  buffer.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  gpu.pending.push_back(upload);
  ExecuteAndReset(gpu);
}

std::string DynamicModuleName(std::uint32_t token) {
  std::ostringstream stream;
  stream << "cu-module-" << std::setfill('0') << std::setw(6)
         << token << ".bin";
  return stream.str();
}

struct DynamicModuleSpec {
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
};

constexpr std::array<DynamicModuleSpec, 12> kDynamicModules{{
    {11, 53920, "06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206"},
    {15, 12632, "9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381"},
    {13, 13088, "fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1"},
    {21, 38472, "11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"},
    {22, 15120, "7d6602bbed88409e9ad2ef71293869dd22f7153f6387339ed65e91c12341c67a"},
    {23, 56480, "d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d"},
    {24, 25880, "5a41f9d65d1f6f933ae42f36b493f60687a646b73929291bafbe9d00297b6bdf"},
    {25, 6640, "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824"},
    {14, 4584, "540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c"},
    {27, 4024, "c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7"},
    {4, 4512, "c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840"},
    {5, 14448, "3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"},
}};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-connected-dynamic-early-rtx <modules> <output> "
                 "<run-id> <poison> <resolution> <full|hash-only>\n";
    return 2;
  }
  try {
    const fs::path modules = argv[1];
    const fs::path output_root = argv[2];
    fs::create_directories(output_root);
    const std::string run_id = argv[3];
    const auto poison = ParsePoison(argv[4]);
    const auto& resolution = dyn::Find(argv[5]);
    const std::string artifact_mode = argv[6];
    if (artifact_mode != "full" && artifact_mode != "hash-only")
      throw std::runtime_error("invalid artifact mode");
    const bool full_artifacts = artifact_mode == "full";
    const auto& r11 = dyn::Module11(resolution);
    const auto& r15 = dyn::Module15(resolution);
    const auto& r13 = dyn::Module13(resolution);
    const auto& r21 = dyn::Module21(resolution);
    const auto& r23 = dyn::Module23(resolution);
    const auto& r5 = dyn::Module5(resolution);
    const auto external = dyn::GenerateExternal(resolution);
    const auto pyramid_bytes = dyn::PyramidBytes(resolution);
    const auto extra_bytes = dyn::ExtraPyramidBytes(resolution);
    const auto motion_pixels = dyn::MotionPixels(resolution);
    const auto color_pixels =
        std::size_t{resolution.color_width} * resolution.color_height;

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("nvapi64.dll unavailable");
    const auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    const auto initialize = Resolve<Initialize>(query, kIdInitialize,
                                                 "Initialize");
    const auto get_texture = Resolve<GetMergedTextureSampler>(
        query, kIdGetMergedTextureSampler, "GetTexture");
    const auto get_surface = Resolve<GetIndependentDescriptor>(
        query, kIdGetIndependentDescriptor, "GetSurface");
    const auto create_module = Resolve<CreateCuModule>(
        query, kIdCreateCuModule, "CreateModule");
    const auto create_function = Resolve<CreateCuFunction>(
        query, kIdCreateCuFunction, "CreateFunction");
    const auto launch_chain = Resolve<LaunchKernelChain>(
        query, kIdLaunchKernelChain, "Launch");
    const auto destroy_module = Resolve<DestroyCuModule>(
        query, kIdDestroyCuModule, "DestroyModule");
    const auto destroy_function = Resolve<DestroyCuFunction>(
        query, kIdDestroyCuFunction, "DestroyFunction");
    RequireNvapi(initialize(), "NvAPI_Initialize");

    D3D12_DESCRIPTOR_HEAP_DESC resource_description{};
    resource_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_description.NumDescriptors = 256;
    resource_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(
                  &resource_description, IID_PPV_ARGS(&resource_heap)),
              "dynamic resource heap");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_description{};
    sampler_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_description.NumDescriptors = 3;
    sampler_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(
                  &sampler_description, IID_PPV_ARGS(&sampler_heap)),
              "dynamic sampler heap");
    const auto resource_start =
        resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto sampler_start = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    const UINT resource_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT sampler_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    auto resource_slot = [&](UINT index) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          resource_start.ptr + std::uint64_t{index} * resource_increment};
    };
    auto sampler_slot = [&](UINT index) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          sampler_start.ptr + std::uint64_t{index} * sampler_increment};
    };
    D3D12_SAMPLER_DESC point_border{};
    point_border.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    point_border.AddressU = point_border.AddressV = point_border.AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&point_border, sampler_slot(0));
    auto point_clamp = point_border;
    point_clamp.AddressU = point_clamp.AddressV = point_clamp.AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    gpu.device->CreateSampler(&point_clamp, sampler_slot(1));
    auto linear_clamp = point_clamp;
    linear_clamp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    linear_clamp.MaxAnisotropy = 1;
    gpu.device->CreateSampler(&linear_clamp, sampler_slot(2));
    UINT descriptor_index = 0;
    auto texture_handle = [&](GpuTexture& texture, UINT sampler_index) {
      const auto slot = descriptor_index++;
      D3D12_SHADER_RESOURCE_VIEW_DESC view{};
      view.Format = texture.format;
      view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      view.Texture2D.MipLevels = 1;
      gpu.device->CreateShaderResourceView(texture.resource.Get(), &view,
                                           resource_slot(slot));
      MergedTextureSamplerParams request{};
      request.struct_size_in = request.struct_size_out = sizeof(request);
      request.device = gpu.device.Get();
      request.texture_descriptor.ptr = resource_slot(slot).ptr;
      request.sampler_descriptor.ptr = sampler_slot(sampler_index).ptr;
      RequireNvapi(get_texture(&request), "GetTexture");
      return request.texture_handle;
    };
    auto surface_handle = [&](GpuTexture& texture) {
      const auto slot = descriptor_index++;
      D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
      view.Format = texture.format;
      view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr,
                                            &view, resource_slot(slot));
      IndependentDescriptorParams request{};
      request.struct_size_in = request.struct_size_out = sizeof(request);
      request.device = gpu.device.Get();
      request.type = IndependentDescriptorType::kSurface;
      request.descriptor.ptr = resource_slot(slot).ptr;
      RequireNvapi(get_surface(&request), "GetSurface");
      return request.handle;
    };

    auto texture = [&](DXGI_FORMAT format, std::uint32_t width,
                       std::uint32_t height, std::uint32_t pixel_bytes,
                       D3D12_RESOURCE_STATES state,
                       const std::vector<std::uint8_t>& bytes) {
      return DynamicUploadTexture(gpu, format, width, height, pixel_bytes,
                                  state, bytes);
    };
    auto source_reset = texture(
        DXGI_FORMAT_R8G8B8A8_UNORM, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        external.source_reset);
    auto source_generated = texture(
        DXGI_FORMAT_R8G8B8A8_UNORM, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        external.source_generated);
    auto history_a = texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.source_width,
        resolution.source_height, 8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        dyn::History(resolution, 0));
    auto history_b = texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.source_width,
        resolution.source_height, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        dyn::History(resolution, poison));
    auto reset_color = texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.color_width,
        resolution.color_height, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(dyn::ColorBytes(resolution), poison));
    auto generated_color = texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT, resolution.color_width,
        resolution.color_height, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(dyn::ColorBytes(resolution), poison));
    auto shared_metric = texture(
        DXGI_FORMAT_R16_FLOAT, resolution.color_width, resolution.color_height,
        2, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(dyn::MetricBytes(resolution), poison));
    auto vector_reset = texture(
        DXGI_FORMAT_R16G16_FLOAT, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        external.vector_reset);
    auto vector_generated = texture(
        DXGI_FORMAT_R16G16_FLOAT, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        external.vector_generated);
    auto scalar_reset = texture(
        DXGI_FORMAT_R32_FLOAT, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        external.scalar_reset);
    auto scalar_generated = texture(
        DXGI_FORMAT_R32_FLOAT, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        external.scalar_generated);
    auto flow = texture(
        DXGI_FORMAT_R32G32_FLOAT, resolution.motion_width,
        resolution.motion_height, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels * 8u, poison));
    auto reset_depth = texture(
        DXGI_FORMAT_R32_UINT, resolution.motion_width,
        resolution.motion_height, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels * 4u, poison));
    auto generated_depth = texture(
        DXGI_FORMAT_R32_UINT, resolution.motion_width,
        resolution.motion_height, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        dyn::InitialGeneratedDepth(resolution));
    auto reset_mask = texture(
        DXGI_FORMAT_R8_UNORM, resolution.motion_width,
        resolution.motion_height, 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels, poison));
    auto generated_mask = texture(
        DXGI_FORMAT_R8_UNORM, resolution.motion_width,
        resolution.motion_height, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        dyn::InitialGeneratedMask(resolution));
    auto working_vector = texture(
        DXGI_FORMAT_R16G16_FLOAT, resolution.motion_width,
        resolution.motion_height, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels * 4u, poison));
    auto working_mask = texture(
        DXGI_FORMAT_R8_UNORM, resolution.motion_width,
        resolution.motion_height, 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels, poison));
    auto scatter_reset = texture(
        DXGI_FORMAT_R8_UNORM, resolution.motion_width,
        resolution.motion_height, 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels, 255u));
    auto scatter_generated = texture(
        DXGI_FORMAT_R8_UNORM, resolution.motion_width,
        resolution.motion_height, 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(motion_pixels, 255u));
    auto reset_projection = texture(
        DXGI_FORMAT_R16G16_FLOAT, resolution.color_width,
        resolution.color_height, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(color_pixels * 4u, poison));
    auto allocation29 = texture(
        DXGI_FORMAT_R16_FLOAT, resolution.color_width,
        resolution.color_height, 2,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        dyn::InitialAllocation29(resolution));
    auto auxiliary_mask = texture(
        DXGI_FORMAT_R8_UNORM, resolution.color_width,
        resolution.color_height, 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(color_pixels, poison));
    auto auxiliary_vector = texture(
        DXGI_FORMAT_R16G16_FLOAT, resolution.color_width,
        resolution.color_height, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(color_pixels * 4u, poison));
    auto reset_present = texture(
        DXGI_FORMAT_R8G8B8A8_UNORM, resolution.source_width,
        resolution.source_height, 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(
            std::size_t{resolution.source_width} * resolution.source_height * 4u,
            poison));
    auto atomic = UploadBuffer(gpu, Guarded(dyn::InitialAtomic(resolution)));
    std::array<GpuBuffer, 5> pyramid;
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      pyramid[index] = UploadBuffer(gpu, PoisonGuarded(pyramid_bytes[index], poison));
    std::array<GpuBuffer, 4> extra_pyramid;
    for (std::size_t index = 0; index < extra_pyramid.size(); ++index)
      extra_pyramid[index] = UploadBuffer(gpu, PoisonGuarded(extra_bytes[index], poison));
    auto model_input = UploadBuffer(
        gpu, PoisonGuarded(dyn::ModelInputBytes(resolution), poison));
    auto status_buffer = UploadBuffer(gpu, PoisonGuarded(4, poison));
    ExecuteAndReset(gpu);

    const auto source_reset_tex = texture_handle(source_reset, 2);
    const auto source_generated_tex = texture_handle(source_generated, 0);
    const auto history_a_tex = texture_handle(history_a, 0);
    const auto history_a_surf = surface_handle(history_a);
    const auto history_b_tex = texture_handle(history_b, 0);
    const auto history_b_surf = surface_handle(history_b);
    const auto reset_color_tex = texture_handle(reset_color, 0);
    const auto reset_color_surf = surface_handle(reset_color);
    const auto generated_color_tex = texture_handle(generated_color, 0);
    const auto generated_color_surf = surface_handle(generated_color);
    const auto shared_metric_tex = texture_handle(shared_metric, 0);
    const auto shared_metric_surf = surface_handle(shared_metric);
    const auto vector_reset_tex = texture_handle(vector_reset, 1);
    const auto vector_generated_tex = texture_handle(vector_generated, 1);
    const auto scalar_reset_tex = texture_handle(scalar_reset, 0);
    const auto scalar_generated_tex = texture_handle(scalar_generated, 0);
    const auto flow_tex = texture_handle(flow, 0);
    const auto flow_surf = surface_handle(flow);
    const auto reset_depth_tex = texture_handle(reset_depth, 0);
    const auto reset_depth_surf = surface_handle(reset_depth);
    const auto generated_depth_tex = texture_handle(generated_depth, 0);
    const auto generated_depth_surf = surface_handle(generated_depth);
    const auto reset_mask_tex = texture_handle(reset_mask, 0);
    const auto reset_mask_surf = surface_handle(reset_mask);
    const auto generated_mask_tex = texture_handle(generated_mask, 0);
    const auto generated_mask_surf = surface_handle(generated_mask);
    const auto working_vector_tex = texture_handle(working_vector, 0);
    const auto working_vector_surf = surface_handle(working_vector);
    const auto working_mask_tex = texture_handle(working_mask, 0);
    const auto working_mask_surf = surface_handle(working_mask);
    const auto scatter_reset_tex = texture_handle(scatter_reset, 0);
    const auto scatter_reset_surf = surface_handle(scatter_reset);
    const auto scatter_generated_tex = texture_handle(scatter_generated, 0);
    const auto scatter_generated_surf = surface_handle(scatter_generated);
    const auto reset_projection_tex = texture_handle(reset_projection, 0);
    const auto reset_projection_surf = surface_handle(reset_projection);
    const auto allocation29_tex = texture_handle(allocation29, 0);
    const auto auxiliary_mask_surf = surface_handle(auxiliary_mask);
    const auto auxiliary_vector_surf = surface_handle(auxiliary_vector);
    const auto reset_present_surf = surface_handle(reset_present);

    struct Loaded { NvObjectHandle module = nullptr; NvObjectHandle function = nullptr; };
    std::array<Loaded, kDynamicModules.size()> loaded{};
    for (std::size_t index = 0; index < kDynamicModules.size(); ++index) {
      const auto bytes = Read(modules / DynamicModuleName(kDynamicModules[index].token));
      if (bytes.size() != kDynamicModules[index].bytes ||
          Sha256(bytes) != kDynamicModules[index].sha256)
        throw std::runtime_error("dynamic early module identity mismatch");
      RequireNvapi(create_module(gpu.device.Get(), bytes.data(),
          static_cast<std::uint32_t>(bytes.size()), &loaded[index].module),
          "CreateModule");
      RequireNvapi(create_function(gpu.device.Get(), loaded[index].module,
          "main_kernel", &loaded[index].function), "CreateFunction");
    }
    auto function_for = [&](std::uint32_t token) {
      for (std::size_t index = 0; index < kDynamicModules.size(); ++index)
        if (kDynamicModules[index].token == token) return loaded[index].function;
      throw std::runtime_error("dynamic module function missing");
    };
    auto launch = [&](std::uint32_t token, const void* parameters,
                      std::size_t parameter_bytes, Dim3 grid, Dim3 block) {
      const KernelLaunchParams descriptor{
          function_for(token), grid, block, 0, parameters,
          static_cast<std::uint32_t>(parameter_bytes)};
      RequireNvapi(launch_chain(gpu.list.Get(), &descriptor, 1),
                   "LaunchKernelChain");
      ExecuteAndReset(gpu);
    };
    auto snapshot_texture = [&](GpuTexture& item,
                                D3D12_RESOURCE_STATES next) {
      auto bytes = ReadTexture(gpu, item);
      DRestore(gpu, item, next);
      ExecuteAndReset(gpu);
      return bytes;
    };
    auto snapshot_buffer = [&](GpuBuffer& item, std::size_t bytes) {
      auto guarded = ReadBuffer(gpu, item);
      DRestore(gpu, item, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);
      return Payload(guarded, bytes);
    };
    std::vector<std::pair<std::string, std::string>> checkpoints;
    auto checkpoint = [&](const char* label,
                          std::initializer_list<std::vector<std::uint8_t>> parts) {
      checkpoints.emplace_back(label, Sha256(DJoin(parts)));
      std::cout << "checkpoint=" << label << '\n' << std::flush;
    };

    auto encode_m11 = [&](std::uint64_t source, std::uint64_t history,
                          std::uint64_t color, std::uint64_t next_history) {
      std::array<std::uint8_t, 168> parameters{};
      DPack64(&parameters, 0, source);
      DPack64(&parameters, 80, history);
      DPack64(&parameters, 96, color);
      DPack64(&parameters, 104, shared_metric_surf);
      DPack64(&parameters, 120, next_history);
      DPack32(&parameters, 128, resolution.module11_mode);
      DPack32(&parameters, 136, resolution.source_width);
      DPack32(&parameters, 140, resolution.source_height);
      DPack32(&parameters, 144, resolution.color_width);
      DPack32(&parameters, 148, resolution.color_height);
      launch(11, parameters.data(), parameters.size(),
             {r11.grid_x, r11.grid_y, 1}, {256, 1, 1});
    };
    auto encode_m15 = [&](bool reset, std::uint64_t vector,
                          std::uint64_t scalar, std::uint64_t depth) {
      std::array<std::uint8_t, 160> parameters{};
      DPack64(&parameters, 0, vector);
      DPack32(&parameters, 16, resolution.source_width);
      DPack32(&parameters, 20, resolution.source_height);
      DPack32(&parameters, 24, r15.inverse_x_bits);
      DPack32(&parameters, 28, r15.inverse_y_bits);
      DPack64(&parameters, 32, flow_surf);
      DPack64(&parameters, 40, scalar);
      DPack32(&parameters, 56, resolution.source_width);
      DPack32(&parameters, 60, resolution.source_height);
      DPack32(&parameters, 64, r15.inverse_x_bits);
      DPack32(&parameters, 68, r15.inverse_y_bits);
      DPack32(&parameters, 72, 0x42200000u);
      DPack32(&parameters, 76, 0x3f800000u);
      DPack32(&parameters, 80, 0x44160000u);
      parameters[84] = 0;
      parameters[85] = static_cast<std::uint8_t>(reset ? 127u : 0u);
      DPack64(&parameters, 88, depth);
      DPack32(&parameters, 144, resolution.motion_width);
      DPack32(&parameters, 148, resolution.motion_height);
      DPack32(&parameters, 152, r15.output_scale_x_bits);
      DPack32(&parameters, 156, r15.output_scale_y_bits);
      launch(15, parameters.data(), parameters.size(),
             {r15.grid_x, r15.grid_y, 1}, {16, 16, 1});
    };
    auto encode_m13 = [&](std::uint64_t depth, std::uint64_t mask) {
      std::array<std::uint8_t, 144> parameters{};
      DPack64(&parameters, 0, flow_tex);
      DPack64(&parameters, 8, depth);
      DPack64(&parameters, 16,
              atomic.resource->GetGPUVirtualAddress() + kGuardBytes);
      DPack64(&parameters, 24, mask);
      DPack32(&parameters, 40, r13.input_scale_x_bits);
      DPack32(&parameters, 44, r13.input_scale_y_bits);
      DPack32(&parameters, 48, resolution.motion_width);
      DPack32(&parameters, 52, resolution.motion_height);
      DPack32(&parameters, 56, r13.motion_divisor_bits);
      DPack32(&parameters, 60, r13.neighbor_divisor_bits);
      DPack32(&parameters, 64, 0x42200000u);
      DPack32(&parameters, 68, 0x3f800000u);
      DPack32(&parameters, 72, 0x44160000u);
      const std::array<float, 16> identity = {
          1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      std::memcpy(parameters.data() + 80, identity.data(), 64);
      launch(13, parameters.data(), parameters.size(),
             {r13.grid_x, r13.grid_y, 1}, {16, 16, 1});
    };
    auto encode_m21 = [&](std::uint64_t depth, std::uint64_t mask) {
      std::array<std::uint8_t, 224> parameters{};
      DPack64(&parameters, 0, flow_tex);
      DPack64(&parameters, 8, depth);
      DPack64(&parameters, 16, mask);
      DPack64(&parameters, 24,
              atomic.resource->GetGPUVirtualAddress() + kGuardBytes);
      DPack64(&parameters, 32, working_vector_surf);
      DPack64(&parameters, 40, working_mask_surf);
      for (std::size_t index = 0; index < pyramid.size(); ++index)
        DPack64(&parameters, 48 + index * 8,
            pyramid[index].resource->GetGPUVirtualAddress() + kGuardBytes);
      DPack32(&parameters, 128, r21.invx);
      DPack32(&parameters, 132, r21.invy);
      DPack32(&parameters, 136, resolution.motion_width);
      DPack32(&parameters, 140, resolution.motion_height);
      DPack32(&parameters, 144, 0x42200000u);
      DPack32(&parameters, 148, 0x3f800000u);
      DPack32(&parameters, 152, 0x44160000u);
      parameters[156] = 0;
      parameters[157] = static_cast<std::uint8_t>(r21.padding157);
      const std::array<float, 16> identity = {
          1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      std::memcpy(parameters.data() + 160, identity.data(), 64);
      launch(21, parameters.data(), parameters.size(),
             {r21.gx, r21.gy, 1}, {256, 1, 1});
    };
    auto encode_m22 = [&] {
      const auto input_width = dyn::CeilDiv(resolution.motion_width, 16u);
      const auto input_height = dyn::CeilDiv(resolution.motion_height, 16u);
      std::array<std::uint8_t, 56> parameters{};
      DPack64(&parameters, 0,
          pyramid[4].resource->GetGPUVirtualAddress() + kGuardBytes);
      for (std::size_t index = 0; index < extra_pyramid.size(); ++index)
        DPack64(&parameters, 8 + index * 8,
            extra_pyramid[index].resource->GetGPUVirtualAddress() + kGuardBytes);
      DPack32(&parameters, 40, resolution.extra_level_count);
      DPack32(&parameters, 44, input_width);
      DPack32(&parameters, 48, input_height);
      std::memset(parameters.data() + 52, poison, 4);
      launch(22, parameters.data(), parameters.size(),
          {dyn::CeilDiv(input_width, 16u), dyn::CeilDiv(input_height, 16u), 1},
          {256, 1, 1});
    };
    std::array<GpuBuffer*, 5> selected_levels{};
    if (resolution.selected_level == 10) {
      selected_levels = {&pyramid[1], &pyramid[2], &pyramid[3], &pyramid[4],
                         &extra_pyramid[0]};
    } else {
      selected_levels = {&pyramid[2], &pyramid[3], &pyramid[4],
                         &extra_pyramid[0], &extra_pyramid[1]};
    }
    auto encode_m23 = [&] {
      std::array<std::uint8_t, 72> parameters{};
      for (std::size_t index = 0; index < selected_levels.size(); ++index)
        DPack64(&parameters, index * 8,
            selected_levels[index]->resource->GetGPUVirtualAddress() +
            kGuardBytes);
      DPack32(&parameters, 40, resolution.module23_width);
      DPack32(&parameters, 44, resolution.module23_height);
      DPack32(&parameters, 48, 0);
      launch(23, parameters.data(), parameters.size(),
             {r23.gx, r23.gy, 1}, {256, 1, 1});
    };
    auto encode_m24 = [&] {
      std::array<std::uint8_t, 64> parameters{};
      DPack64(&parameters, 0, working_vector_surf);
      DPack64(&parameters, 8,
              pyramid[0].resource->GetGPUVirtualAddress() + kGuardBytes);
      for (std::size_t index = 0; index < 4; ++index)
        DPack64(&parameters, 16 + index * 8,
            pyramid[index + 1].resource->GetGPUVirtualAddress() +
            kGuardBytes);
      DPack32(&parameters, 48, resolution.selected_level);
      DPack32(&parameters, 52, resolution.motion_width);
      DPack32(&parameters, 56, resolution.motion_height);
      std::memset(parameters.data() + 60, poison, 4);
      launch(24, parameters.data(), parameters.size(),
          {dyn::CeilDiv(resolution.motion_width, 16u),
           dyn::CeilDiv(resolution.motion_height, 16u), 1}, {256, 1, 1});
    };
    auto encode_m25 = [&](bool generated, std::uint64_t scatter) {
      std::array<std::uint8_t, 48> parameters{};
      DPack64(&parameters, 0, working_vector_tex);
      DPack64(&parameters, 8, scatter);
      DPack32(&parameters, 16, 0);
      DPack32(&parameters, 20, resolution.motion_width);
      DPack32(&parameters, 24, resolution.motion_height);
      DPack32(&parameters, 28, r21.invx);
      DPack32(&parameters, 32, r21.invy);
      DPack32(&parameters, 36, resolution.motion_width);
      DPack32(&parameters, 40, resolution.motion_height);
      DPack32(&parameters, 44, generated ? 0x43e947aeu : 0x424f5c29u);
      launch(25, parameters.data(), parameters.size(),
          {dyn::CeilDiv(resolution.motion_width, 16u),
           dyn::CeilDiv(resolution.motion_height, 16u), 1}, {16, 16, 1});
    };
    auto encode_status = [&](std::uint32_t value) {
      std::array<std::uint8_t, 16> parameters{};
      DPack64(&parameters, 0,
              status_buffer.resource->GetGPUVirtualAddress() + kGuardBytes);
      DPack32(&parameters, 8, 1);
      DPack32(&parameters, 12, value);
      launch(27, parameters.data(), parameters.size(), {1, 1, 1}, {64, 1, 1});
    };

    encode_m11(source_reset_tex, history_a_tex,
               reset_color_surf, history_b_surf);
    checkpoint("reset-module11", {
        snapshot_texture(reset_color,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        snapshot_texture(shared_metric, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        snapshot_texture(history_b,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    encode_m15(true, vector_reset_tex, scalar_reset_tex, reset_depth_surf);
    checkpoint("reset-module15", {
        snapshot_texture(flow, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        snapshot_texture(reset_depth,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    encode_m13(reset_depth_tex, reset_mask_surf);
    checkpoint("reset-module13", {
        snapshot_buffer(atomic, dyn::InitialAtomic(resolution).size()),
        snapshot_texture(reset_mask,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    encode_m21(generated_depth_tex, generated_mask_tex);
    std::vector<std::uint8_t> reset_working_vector = snapshot_texture(
        working_vector, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::vector<std::uint8_t> reset_working_mask = snapshot_texture(
        working_mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::array<std::vector<std::uint8_t>, 5> reset_pyramid;
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      reset_pyramid[index] = snapshot_buffer(pyramid[index], pyramid_bytes[index]);
    checkpoint("reset-module21", {
        reset_working_vector, reset_working_mask, reset_pyramid[0],
        reset_pyramid[1], reset_pyramid[2], reset_pyramid[3], reset_pyramid[4]});
    encode_m22();
    std::array<std::vector<std::uint8_t>, 2> reset_extra;
    for (std::size_t index = 0; index < resolution.extra_level_count; ++index)
      reset_extra[index] = snapshot_buffer(extra_pyramid[index], extra_bytes[index]);
    if (resolution.extra_level_count == 1)
      checkpoint("reset-module22", {reset_extra[0]});
    else
      checkpoint("reset-module22", {reset_extra[0], reset_extra[1]});
    encode_m23();
    const std::size_t selected_level_bytes =
        resolution.selected_level == 10 ? pyramid_bytes[1] : pyramid_bytes[2];
    checkpoint("reset-module23", {
        snapshot_buffer(*selected_levels[0], selected_level_bytes)});
    encode_m24();
    checkpoint("reset-module24", {
        snapshot_texture(working_vector,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    encode_m25(false, scatter_reset_surf);
    checkpoint("reset-module25", {
        snapshot_texture(scatter_reset,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    std::array<std::uint8_t, 32> module14_parameters{};
    DPack64(&module14_parameters, 0, reset_color_tex);
    DPack64(&module14_parameters, 8, scatter_reset_tex);
    DPack64(&module14_parameters, 16, reset_projection_surf);
    DPack32(&module14_parameters, 24, resolution.color_width);
    DPack32(&module14_parameters, 28, resolution.color_height);
    launch(14, module14_parameters.data(), module14_parameters.size(),
        {dyn::CeilDiv(resolution.color_width, 16u),
         dyn::CeilDiv(resolution.color_height, 16u), 1}, {16, 16, 1});
    checkpoint("reset-module14", {
        snapshot_texture(reset_projection,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    encode_status(1);
    checkpoint("reset-module27", {snapshot_buffer(status_buffer, 4)});
    std::array<std::uint8_t, 40> module4_parameters{};
    DPack64(&module4_parameters, 0, source_reset_tex);
    DPack64(&module4_parameters, 16, reset_present_surf);
    DPack32(&module4_parameters, 32, resolution.source_width);
    DPack32(&module4_parameters, 36, resolution.source_height);
    launch(4, module4_parameters.data(), module4_parameters.size(),
        {dyn::CeilDiv(resolution.source_width, 16u),
         dyn::CeilDiv(resolution.source_height, 16u), 1}, {16, 16, 1});
    const auto reset_present_bytes = snapshot_texture(
        reset_present, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    checkpoint("reset-module4", {reset_present_bytes});

    encode_status(0);
    checkpoint("generated-module27", {snapshot_buffer(status_buffer, 4)});
    DRestore(gpu, history_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m11(source_generated_tex, history_b_tex,
               generated_color_surf, history_a_surf);
    checkpoint("generated-module11", {
        snapshot_texture(generated_color,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        snapshot_texture(shared_metric,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        snapshot_texture(history_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)});
    DRestore(gpu, flow, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DRestore(gpu, generated_depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m15(false, vector_generated_tex, scalar_generated_tex,
               generated_depth_surf);
    checkpoint("generated-module15", {
        snapshot_texture(flow, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        snapshot_texture(generated_depth,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    DOverwriteBuffer(gpu, atomic, Guarded(dyn::InitialAtomic(resolution)));
    DRestore(gpu, generated_mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m13(generated_depth_tex, generated_mask_surf);
    checkpoint("generated-module13", {
        snapshot_buffer(atomic, dyn::InitialAtomic(resolution).size()),
        snapshot_texture(generated_mask,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    DRestore(gpu, working_vector, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m21(reset_depth_tex, reset_mask_tex);
    std::vector<std::uint8_t> generated_working_vector = snapshot_texture(
        working_vector, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::vector<std::uint8_t> generated_working_mask = snapshot_texture(
        working_mask, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    std::array<std::vector<std::uint8_t>, 5> generated_pyramid;
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      generated_pyramid[index] = snapshot_buffer(pyramid[index], pyramid_bytes[index]);
    checkpoint("generated-module21", {
        generated_working_vector, generated_working_mask, generated_pyramid[0],
        generated_pyramid[1], generated_pyramid[2], generated_pyramid[3],
        generated_pyramid[4]});
    encode_m22();
    std::array<std::vector<std::uint8_t>, 2> generated_extra;
    for (std::size_t index = 0; index < resolution.extra_level_count; ++index)
      generated_extra[index] =
          snapshot_buffer(extra_pyramid[index], extra_bytes[index]);
    if (resolution.extra_level_count == 1)
      checkpoint("generated-module22", {generated_extra[0]});
    else
      checkpoint("generated-module22", {generated_extra[0], generated_extra[1]});
    encode_m23();
    checkpoint("generated-module23", {
        snapshot_buffer(*selected_levels[0], selected_level_bytes)});
    encode_m24();
    checkpoint("generated-module24", {
        snapshot_texture(working_vector,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});
    encode_m25(true, scatter_generated_surf);
    checkpoint("generated-module25", {
        snapshot_texture(scatter_generated,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)});

    const std::array<std::uint64_t, 10> module5_inputs = {
        reset_color_tex, generated_color_tex, scatter_generated_tex,
        scatter_reset_tex, reset_projection_tex, shared_metric_tex,
        allocation29_tex, flow_tex, working_vector_tex, working_mask_tex};
    std::array<std::uint8_t, 192> module5_parameters{};
    for (std::size_t index = 0; index < module5_inputs.size(); ++index)
      DPack64(&module5_parameters, index * 8, module5_inputs[index]);
    DPack64(&module5_parameters, 80,
            model_input.resource->GetGPUVirtualAddress() + kGuardBytes);
    DPack64(&module5_parameters, 88, auxiliary_mask_surf);
    DPack64(&module5_parameters, 96, auxiliary_vector_surf);
    DPack32(&module5_parameters, 152, resolution.motion_width);
    DPack32(&module5_parameters, 156, resolution.motion_height);
    DPack32(&module5_parameters, 160, resolution.color_width);
    DPack32(&module5_parameters, 164, resolution.color_height);
    DPack32(&module5_parameters, 168, resolution.tensor_width);
    DPack32(&module5_parameters, 172, resolution.tensor_height);
    DPack32(&module5_parameters, 176, 0);
    DPack32(&module5_parameters, 180, 0);
    DPack32(&module5_parameters, 184, resolution.origin_y);
    DPack32(&module5_parameters, 188, resolution.origin_y);
    launch(5, module5_parameters.data(), module5_parameters.size(),
           {r5.grid_x, r5.grid_y, 1}, {16, 16, 1});
    const auto model_guarded = ReadBuffer(gpu, model_input);
    const auto model_input_bytes = Payload(
        model_guarded, dyn::ModelInputBytes(resolution));
    const auto auxiliary_mask_bytes = ReadTexture(gpu, auxiliary_mask);
    const auto auxiliary_vector_bytes = ReadTexture(gpu, auxiliary_vector);
    checkpoint("generated-module5", {
        model_input_bytes, auxiliary_mask_bytes, auxiliary_vector_bytes});

    std::size_t input_mismatches = 0;
    input_mismatches += Mismatches(external.source_reset,
                                   ReadTexture(gpu, source_reset));
    input_mismatches += Mismatches(external.source_generated,
                                   ReadTexture(gpu, source_generated));
    input_mismatches += Mismatches(external.vector_reset,
                                   ReadTexture(gpu, vector_reset));
    input_mismatches += Mismatches(external.vector_generated,
                                   ReadTexture(gpu, vector_generated));
    input_mismatches += Mismatches(external.scalar_reset,
                                   ReadTexture(gpu, scalar_reset));
    input_mismatches += Mismatches(external.scalar_generated,
                                   ReadTexture(gpu, scalar_generated));
    input_mismatches += Mismatches(dyn::InitialAllocation29(resolution),
                                   ReadTexture(gpu, allocation29));
    std::size_t canary_mismatches =
        CanaryMismatches(model_guarded, dyn::ModelInputBytes(resolution));
    auto atomic_guarded = ReadBuffer(gpu, atomic);
    canary_mismatches += CanaryMismatches(
        atomic_guarded, dyn::InitialAtomic(resolution).size());
    for (std::size_t index = 0; index < pyramid.size(); ++index) {
      const auto guarded = ReadBuffer(gpu, pyramid[index]);
      canary_mismatches += CanaryMismatches(guarded, pyramid_bytes[index]);
    }
    for (std::size_t index = 0; index < extra_pyramid.size(); ++index) {
      const auto guarded = ReadBuffer(gpu, extra_pyramid[index]);
      canary_mismatches += CanaryMismatches(guarded, extra_bytes[index]);
    }
    const auto status_guarded = ReadBuffer(gpu, status_buffer);
    canary_mismatches += CanaryMismatches(status_guarded, 4);
    const auto allocation37_poison_halfwords =
        UntouchedWords(model_input_bytes, poison, 2);
    if (full_artifacts) {
      Write(output_root / "allocation37.fp16", model_input_bytes);
      Write(output_root / "auxiliary-mask.r8", auxiliary_mask_bytes);
      Write(output_root / "auxiliary-vector.rg16f", auxiliary_vector_bytes);
      Write(output_root / "reset-present.rgba8", reset_present_bytes);
    }
    for (auto iterator = loaded.rbegin(); iterator != loaded.rend(); ++iterator) {
      RequireNvapi(destroy_function(gpu.device.Get(), iterator->function),
                   "DestroyFunction");
      RequireNvapi(destroy_module(gpu.device.Get(), iterator->module),
                   "DestroyModule");
    }
    const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                      allocation37_poison_halfwords == 0 &&
                      checkpoints.size() == 21;
    std::ofstream receipt(output_root / "run-receipt.json");
    if (!receipt) throw std::runtime_error("receipt open failed");
    receipt << "{\n  \"schema\": "
               "\"neuralstudio.fg.connected-dynamic-early-run.v1\",\n"
            << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"resolution\": \"" << resolution.name << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"selected_launch_count\": 21,\n"
            << "  \"module22_and_module24_executed_each_phase\": true,\n"
            << "  \"checkpoint_sha256\": {";
    for (std::size_t index = 0; index < checkpoints.size(); ++index)
      receipt << (index ? "," : "") << "\n    \""
              << checkpoints[index].first << "\": \""
              << checkpoints[index].second << "\"";
    receipt << "\n  },\n"
            << "  \"allocation37_bytes\": " << model_input_bytes.size()
            << ",\n  \"allocation37_sha256\": \""
            << Sha256(model_input_bytes) << "\",\n"
            << "  \"allocation37_poison_halfwords\": "
            << allocation37_poison_halfwords << ",\n"
            << "  \"auxiliary_mask_sha256\": \""
            << Sha256(auxiliary_mask_bytes) << "\",\n"
            << "  \"auxiliary_vector_sha256\": \""
            << Sha256(auxiliary_vector_bytes) << "\",\n"
            << "  \"reset_present_sha256\": \""
            << Sha256(reset_present_bytes) << "\",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
            << "  \"artifacts_written\": "
            << (full_artifacts ? "true" : "false") << ",\n"
            << "  \"expected_or_captured_output_ingress\": false,\n"
            << "  \"archived_intermediate_ingress\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n}\n";
    std::cout << "allocation37_sha256=" << Sha256(model_input_bytes)
              << "\ncheckpoint_count=" << checkpoints.size()
              << "\nresult=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
