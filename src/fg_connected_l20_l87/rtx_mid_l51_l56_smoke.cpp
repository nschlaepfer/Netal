#define main FgMidL55UnusedMain
#include "../fg_mid_l55/rtx_oracle.cpp"
#undef main

#include "../fg_mid_l53/contract.h"
#include "../fg_mid_l54/contract.h"
#include "../fg_connected_custom_l56_l83/contract_full.h"
#include "connected_l20_l87_bridge.h"
#include "rtx_live_core_bridge.h"
#include "rtx_shared_cuda_buffer.h"

namespace {

namespace connected = fg_connected_l20_l87;
namespace l53 = fg_mid_l53;

constexpr std::size_t kL51Bytes = 160u * 96u * 16u * 2u;
constexpr std::size_t kL52VectorBytes = 320u * 180u * 4u;
constexpr std::size_t kL52ScalarBytes = 320u * 180u * 2u;
constexpr std::size_t kTailMotionBytes = 640u * 368u * 16u;
constexpr std::size_t kTailMaskBytes = 640u * 360u;
constexpr std::array<std::size_t, 4> kTailPyramidBytes = {
    320u * 180u * 8u, 160u * 90u * 8u, 80u * 45u * 8u,
    40u * 23u * 8u};

template <std::size_t N>
void PackU64(std::array<std::uint8_t, N>* bytes, std::size_t offset,
             std::uint64_t value) {
  for (std::size_t index = 0; index < 8u; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (8u * index));
}

template <std::size_t N>
void PackU32(std::array<std::uint8_t, N>* bytes, std::size_t offset,
             std::uint32_t value) {
  for (std::size_t index = 0; index < 4u; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (8u * index));
}

void Restore(GpuContext& gpu, GpuBuffer& buffer,
             D3D12_RESOURCE_STATES state) {
  Transition(gpu.list.Get(), buffer.resource.Get(), buffer.state, state);
  buffer.state = state;
}

void Restore(GpuContext& gpu, GpuTexture& texture,
             D3D12_RESOURCE_STATES state) {
  Transition(gpu.list.Get(), texture.resource.Get(), texture.state, state);
  texture.state = state;
}

std::vector<std::uint8_t> BinaryDepth() {
  std::vector<std::uint8_t> bytes(320u * 180u);
  for (std::size_t pixel = 0; pixel < bytes.size(); ++pixel)
    bytes[pixel] = ((pixel / 320u + pixel % 320u) & 1u) ? 0xffu : 0u;
  return bytes;
}

std::vector<std::uint8_t> IslandsMask() {
  std::vector<std::uint8_t> bytes(kTailMaskBytes);
  for (std::uint32_t y = 0; y < 360u; ++y) {
    for (std::uint32_t x = 0; x < 640u; ++x) {
      const bool island =
          (x >= 104u && x < 224u && y >= 72u && y < 184u) ||
          (x >= 376u && x < 552u && y >= 196u && y < 320u);
      const bool hole = ((x * 13u + y * 7u) % 47u) == 0u;
      bytes[std::size_t{y} * 640u + x] = island && !hole ? 0xffu : 0u;
    }
  }
  return bytes;
}

std::string ModuleName(std::uint32_t token) {
  std::ostringstream stream;
  stream << "cu-module-" << std::setfill('0') << std::setw(6) << token
         << ".bin";
  return stream.str();
}

struct ModuleSpec {
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
};

struct NvModule {
  NvObjectHandle module = nullptr;
  NvObjectHandle function = nullptr;
};

std::size_t SentinelCount(const std::vector<std::uint8_t>& bytes) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    result += value == l53::kInitialOutput;
  }
  return result;
}

std::vector<std::uint8_t> Join(
    std::initializer_list<const std::vector<std::uint8_t>*> values) {
  std::vector<std::uint8_t> result;
  for (const auto* value : values)
    result.insert(result.end(), value->begin(), value->end());
  return result;
}

void LaunchCudaPacked(CUfunction function, const void* parameters,
                      std::size_t parameter_bytes, unsigned grid_x,
                      unsigned grid_y, unsigned grid_z, unsigned block_x,
                      unsigned block_y, unsigned block_z) {
  void* pointer = const_cast<void*>(parameters);
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, pointer,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                   CU_LAUNCH_PARAM_END};
  connected::CudaCheck(
      cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y,
                     block_z, 0, nullptr, nullptr, extra),
      "launch CUDA tail kernel");
  connected::CudaCheck(cuCtxSynchronize(), "synchronize CUDA tail kernel");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage=fg-connected-mid-smoke <modules> <model> "
                 "<output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const fs::path modules = argv[1];
    const fs::path output_directory = argv[3];
    const std::string run_id = argv[4];
    const auto poison = ParsePoison(argv[5]);
    fs::create_directories(output_directory);
    const GeneratedInputs l55_external;
    const auto l52_depth = BinaryDepth();
    const std::array<std::vector<std::uint8_t>, 4> l53_external = {
        l53::CurrToPrevFlow(), l53::PrevToCurrFlow(false),
        l53::CurrDepth(), l53::PrevDepth()};

    GpuContext gpu = CreateGpuContext();
    auto l51 = connected::UploadSharedCudaBuffer(
        gpu, PoisonGuarded(kL51Bytes, poison));
    GpuTexture l52_depth_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 320, 180, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l52_depth);
    GpuTexture l52_a = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kL52VectorBytes, poison));
    GpuTexture l52_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kL52VectorBytes, poison));
    GpuTexture l52_confidence = UploadTexture(
        gpu, DXGI_FORMAT_R16_FLOAT, 320, 180, 2,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kL52ScalarBytes, poison));
    GpuTexture l53_flow_a = UploadTexture(
        gpu, DXGI_FORMAT_R32G32_FLOAT, 320, 180, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[0]);
    GpuTexture l53_flow_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[1]);
    GpuTexture l53_depth_a = UploadTexture(
        gpu, DXGI_FORMAT_R32_UINT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[2]);
    GpuTexture l53_depth_b = UploadTexture(
        gpu, DXGI_FORMAT_R32_UINT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[3]);
    const auto candidate_initial = Guarded(l53::InitialOutput());
    GpuBuffer l53_a = UploadBuffer(gpu, candidate_initial);
    GpuBuffer l53_b = UploadBuffer(gpu, candidate_initial);
    GpuTexture l54_blend = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kRgTextureBytes, poison));
    GpuTexture rgba_a = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        l55_external.rgba_a);
    GpuTexture rgba_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        l55_external.rgba_b);
    GpuTexture motion = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, kCandidateWidth, kCandidateHeight, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        l55_external.motion);
    GpuTexture surface_a = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kSurfaceRgbaBytes, poison));
    GpuTexture surface_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kSurfaceRgbaBytes, poison));
    GpuTexture surface_mask = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, kActiveWidth, kActiveHeight, 1,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kSurfaceMaskBytes, poison));
    auto linear_a = connected::UploadSharedCudaBuffer(
        gpu, PoisonGuarded(kLinearRgbaBytes, poison));
    auto linear_b = connected::UploadSharedCudaBuffer(
        gpu, PoisonGuarded(kLinearRgbaBytes, poison));
    auto linear_mask = connected::UploadSharedCudaBuffer(
        gpu, PoisonGuarded(kLinearMaskBytes, poison));
    auto l83 = connected::UploadSharedCudaBuffer(
        gpu, PoisonGuarded(fg_connected_custom::kL83Bytes, poison));
    ExecuteAndReset(gpu);

    FgConnectedPrefixResult prefix_result{};
    if (RunFgConnectedPrefixL20L51(
            argv[1], argv[2], poison,
            static_cast<std::uint64_t>(l51.cuda_ptr + kGuardBytes),
            &prefix_result) != 0)
      throw std::runtime_error("connected L20-L51 prefix failed");
    auto l51_guarded = ReadBuffer(gpu, l51.d3d);
    const auto l51_bytes = Payload(l51_guarded, kL51Bytes);
    Restore(gpu, l51.d3d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading real nvapi64.dll failed");
    const auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (query == nullptr) throw std::runtime_error("nvapi_QueryInterface unavailable");
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
    resource_heap_description.NumDescriptors = 64;
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
              "CreateDescriptorHeap(samplers)");
    const auto resource_start =
        resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto sampler = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC sampler_description{};
    sampler_description.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler_description.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler_description.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&sampler_description, sampler);
    auto slot = [&](UINT index) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{
          resource_start.ptr + std::uint64_t{index} * increment};
    };
    auto make_texture = [&](GpuTexture& texture, UINT index) {
      D3D12_SHADER_RESOURCE_VIEW_DESC description{};
      description.Format = texture.format;
      description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      description.Texture2D.MipLevels = 1;
      gpu.device->CreateShaderResourceView(texture.resource.Get(), &description,
                                           slot(index));
      MergedTextureSamplerParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.texture_descriptor.ptr = slot(index).ptr;
      params.sampler_descriptor.ptr = sampler.ptr;
      RequireNvapi(get_texture(&params), "GetCudaMergedTextureSamplerObject");
      return params.texture_handle;
    };
    auto make_surface = [&](GpuTexture& texture, UINT index) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
      description.Format = texture.format;
      description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr,
                                            &description, slot(index));
      IndependentDescriptorParams params{};
      params.struct_size_in = sizeof(params);
      params.struct_size_out = sizeof(params);
      params.device = gpu.device.Get();
      params.type = IndependentDescriptorType::kSurface;
      params.descriptor.ptr = slot(index).ptr;
      RequireNvapi(get_surface(&params),
                   "GetCudaIndependentDescriptorObject");
      return params.handle;
    };

    const auto depth_texture = make_texture(l52_depth_texture, 0);
    const auto l52_a_surface = make_surface(l52_a, 1);
    const auto l52_b_surface = make_surface(l52_b, 2);
    const auto confidence_surface = make_surface(l52_confidence, 3);
    const auto flow_a_texture = make_texture(l53_flow_a, 4);
    const auto flow_b_texture = make_texture(l53_flow_b, 5);
    const auto depth_a_texture = make_texture(l53_depth_a, 6);
    const auto depth_b_texture = make_texture(l53_depth_b, 7);
    const auto l52_a_texture = make_texture(l52_a, 8);
    const auto l52_b_texture = make_texture(l52_b, 9);
    const auto l54_surface = make_surface(l54_blend, 10);
    const auto rgba_a_texture = make_texture(rgba_a, 11);
    const auto rgba_b_texture = make_texture(rgba_b, 12);
    const auto motion_texture = make_texture(motion, 13);
    const auto blend_texture = make_texture(l54_blend, 14);
    const auto surface_a_handle = make_surface(surface_a, 15);
    const auto surface_b_handle = make_surface(surface_b, 16);
    const auto surface_mask_handle = make_surface(surface_mask, 17);

    const std::array<ModuleSpec, 4> specs = {{
        {6, 9064, "d45333f43f343d1f38bd304fe76214dc7064f2a7d24ce1ea06bc791867783264"},
        {12, 98408, "5a8e0284aab8ac14fc82b0504bbeef25d2fce1d13a1c11d8bdb3f91fee8145fc"},
        {26, 7544, "d8d41b87895a3119f24db4a8d4cb1ec4f73613b507e3d1d4de5dea0f36fe5dc5"},
        {3, 43880, "66a9e6bda87ec8b63c5e10752e84d07435dc4392bbe42ec142e78b0d0b624e37"},
    }};
    std::array<NvModule, 4> loaded{};
    for (std::size_t index = 0; index < specs.size(); ++index) {
      const auto bytes = Read(modules / ModuleName(specs[index].token));
      if (bytes.size() != specs[index].bytes ||
          Sha256(bytes) != specs[index].sha256)
        throw std::runtime_error("connected module identity mismatch");
      RequireNvapi(create_module(gpu.device.Get(), bytes.data(),
                                 static_cast<std::uint32_t>(bytes.size()),
                                 &loaded[index].module),
                   "CreateCuModule");
      RequireNvapi(create_function(gpu.device.Get(), loaded[index].module,
                                   "main_kernel", &loaded[index].function),
                   "CreateCuFunction");
    }
    auto launch = [&](std::size_t module_index, const void* parameters,
                      std::uint32_t parameter_bytes, Dim3 grid, Dim3 block) {
      const KernelLaunchParams descriptor{loaded[module_index].function,
                                          grid, block, 0, parameters,
                                          parameter_bytes};
      RequireNvapi(launch_chain(gpu.list.Get(), &descriptor, 1),
                   "LaunchKernelChain");
      ExecuteAndReset(gpu);
    };

    std::array<std::uint8_t, 168> p52{};
    PackU64(&p52, 0, l51.d3d.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU32(&p52, 8, 1);
    PackU64(&p52, 16, depth_texture);
    PackU64(&p52, 48, l52_a_surface);
    PackU64(&p52, 56, l52_b_surface);
    PackU64(&p52, 64, confidence_surface);
    PackU32(&p52, 136, 320);
    PackU32(&p52, 140, 180);
    PackU32(&p52, 144, 160);
    PackU32(&p52, 148, 96);
    PackU32(&p52, 160, 3);
    PackU32(&p52, 164, 3);
    launch(0, p52.data(), static_cast<std::uint32_t>(p52.size()),
           {20, 12, 1}, {16, 16, 1});
    auto l52_a_bytes = ReadTexture(gpu, l52_a);
    auto l52_b_bytes = ReadTexture(gpu, l52_b);
    auto l52_confidence_bytes = ReadTexture(gpu, l52_confidence);
    Restore(gpu, l52_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Restore(gpu, l52_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Restore(gpu, l52_confidence, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    std::array<std::uint8_t, 144> p53{};
    PackU64(&p53, 0, flow_a_texture);
    PackU64(&p53, 8, flow_b_texture);
    PackU64(&p53, 16, depth_a_texture);
    PackU64(&p53, 24, depth_b_texture);
    PackU32(&p53, 32, 0x3f000000u);
    PackU64(&p53, 40,
            l53_a.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p53, 48,
            l53_b.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU32(&p53, 112, 320);
    PackU32(&p53, 116, 180);
    PackU32(&p53, 120, 0x414f5c29u);
    PackU32(&p53, 124, 0x42200000u);
    PackU32(&p53, 128, 0x3f800000u);
    PackU32(&p53, 132, 0x44160000u);
    launch(1, p53.data(), static_cast<std::uint32_t>(p53.size()),
           {20, 12, 1}, {324, 1, 1});
    auto l53_a_guarded = ReadBuffer(gpu, l53_a);
    auto l53_b_guarded = ReadBuffer(gpu, l53_b);
    auto l53_a_bytes = Payload(l53_a_guarded, l53::kOutputBytes);
    auto l53_b_bytes = Payload(l53_b_guarded, l53::kOutputBytes);
    Restore(gpu, l53_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu, l53_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    std::array<std::uint8_t, 88> p54{};
    PackU32(&p54, 0, 320);
    PackU32(&p54, 4, 180);
    PackU32(&p54, 8, 320);
    PackU32(&p54, 12, 180);
    PackU64(&p54, 16, l52_a_texture);
    PackU64(&p54, 24, l52_b_texture);
    PackU64(&p54, 32,
            l53_a.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p54, 40,
            l53_b.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p54, 48, l54_surface);
    launch(2, p54.data(), static_cast<std::uint32_t>(p54.size()),
           {20, 12, 1}, {16, 16, 1});
    auto l54_bytes = ReadTexture(gpu, l54_blend);
    Restore(gpu, l54_blend, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ExecuteAndReset(gpu);

    std::array<std::uint8_t, kPackedParameterBytes> p55{};
    PackU64(&p55, 0, rgba_a_texture);
    PackU64(&p55, 8, rgba_b_texture);
    PackU64(&p55, 16, motion_texture);
    PackU64(&p55, 24,
            l53_a.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p55, 32,
            l53_b.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p55, 40, blend_texture);
    PackU64(&p55, 48, surface_a_handle);
    PackU64(&p55, 56, surface_b_handle);
    PackU64(&p55, 64, surface_mask_handle);
    PackU64(&p55, 72,
            linear_a.d3d.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p55, 80,
            linear_b.d3d.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p55, 88,
            linear_mask.d3d.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU32(&p55, 184, kPhysicalWidth);
    PackU32(&p55, 188, kPhysicalHeight);
    PackU32(&p55, 192, kAccumulationMode);
    PackU32(&p55, 196, kCandidateWidth);
    PackU32(&p55, 200, kCandidateHeight);
    PackU32(&p55, 204, kCandidateWidth);
    PackU32(&p55, 208, kCandidateHeight);
    PackU32(&p55, 212, kActiveWidth);
    PackU32(&p55, 216, kActiveHeight);
    p55[220] = static_cast<std::uint8_t>(kOptionalUiEnabled);
    launch(3, p55.data(), static_cast<std::uint32_t>(p55.size()),
           {kGridX, kGridY, 1}, {kBlockX, 1, 1});
    auto surface_a_bytes = ReadTexture(gpu, surface_a);
    auto surface_b_bytes = ReadTexture(gpu, surface_b);
    auto surface_mask_bytes = ReadTexture(gpu, surface_mask);
    auto linear_a_guarded = ReadBuffer(gpu, linear_a.d3d);
    auto linear_b_guarded = ReadBuffer(gpu, linear_b.d3d);
    auto linear_mask_guarded = ReadBuffer(gpu, linear_mask.d3d);
    auto linear_a_bytes = Payload(linear_a_guarded, kLinearRgbaBytes);
    auto linear_b_bytes = Payload(linear_b_guarded, kLinearRgbaBytes);
    auto linear_mask_bytes = Payload(linear_mask_guarded, kLinearMaskBytes);
    Restore(gpu, surface_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Restore(gpu, surface_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Restore(gpu, linear_a.d3d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu, linear_b.d3d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu, linear_mask.d3d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    FgConnectedCoreBridgeResult core_result{};
    if (RunFgConnectedCoreL56L83Live(
            argv[1], argv[2], poison,
            static_cast<std::uint64_t>(linear_a.cuda_ptr + kGuardBytes),
            static_cast<std::uint64_t>(linear_b.cuda_ptr + kGuardBytes),
            static_cast<std::uint64_t>(linear_mask.cuda_ptr + kGuardBytes),
            static_cast<std::uint64_t>(l83.cuda_ptr + kGuardBytes),
            &core_result) != 0)
      throw std::runtime_error("connected live L56-L83 core failed");
    auto l83_guarded = ReadBuffer(gpu, l83.d3d);
    auto l83_bytes = Payload(l83_guarded, fg_connected_custom::kL83Bytes);

    for (auto item = loaded.rbegin(); item != loaded.rend(); ++item) {
      RequireNvapi(destroy_function(gpu.device.Get(), item->function),
                   "DestroyCuFunction");
      RequireNvapi(destroy_module(gpu.device.Get(), item->module),
                   "DestroyCuModule");
    }
    Restore(gpu, l83.d3d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    const auto tail_mask_input = IslandsMask();
    GpuTexture tail_mask = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 640, 360, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, tail_mask_input);
    GpuTexture final_color = UploadTexture(
        gpu, DXGI_FORMAT_R8G8B8A8_UNORM, 640, 360, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(640u * 360u * 4u, poison));
    GpuTexture tail_output_mask = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 640, 360, 1,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kTailMaskBytes, poison));
    std::array<GpuBuffer, 4> pyramid{};
    std::array<std::vector<std::uint8_t>, 4> pyramid_initial{};
    for (std::size_t index = 0; index < pyramid.size(); ++index) {
      pyramid_initial[index] =
          PoisonGuarded(kTailPyramidBytes[index], poison);
      pyramid[index] = UploadBuffer(gpu, pyramid_initial[index]);
    }
    ExecuteAndReset(gpu);
    const auto tail_mask_texture = make_texture(tail_mask, 18);
    const auto final_color_surface = make_surface(final_color, 19);
    const auto tail_output_mask_surface = make_surface(tail_output_mask, 20);
    const auto tail_output_mask_texture = make_texture(tail_output_mask, 21);
    const auto l55_surface_a_texture = make_texture(surface_a, 22);
    const auto l55_surface_b_texture = make_texture(surface_b, 23);

    const auto module17_bytes = Read(modules / ModuleName(17));
    if (module17_bytes.size() != 33936u ||
        Sha256(module17_bytes) !=
            "3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76")
      throw std::runtime_error("connected module17 identity mismatch");
    NvObjectHandle module17 = nullptr;
    NvObjectHandle function17 = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module17_bytes.data(),
                               static_cast<std::uint32_t>(module17_bytes.size()),
                               &module17),
                 "CreateCuModule(module17)");
    RequireNvapi(create_function(gpu.device.Get(), module17, "main_kernel",
                                 &function17),
                 "CreateCuFunction(module17)");
    std::array<std::uint8_t, 152> p84{};
    PackU64(&p84, 0, final_color_surface);
    PackU64(&p84, 16, l55_surface_a_texture);
    PackU64(&p84, 24, l55_surface_b_texture);
    PackU64(&p84, 64,
            l83.d3d.resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p84, 72, tail_mask_texture);
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      PackU64(&p84, 80u + index * 8u,
              pyramid[index].resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p84, 112, tail_output_mask_surface);
    PackU32(&p84, 128, 640);
    PackU32(&p84, 132, 368);
    PackU32(&p84, 136, 640);
    PackU32(&p84, 140, 360);
    PackU32(&p84, 144, 0x3f000000u);
    PackU32(&p84, 148, 1);
    const KernelLaunchParams launch84{function17, {40, 23, 1},
                                      {256, 1, 1}, 0, p84.data(),
                                      static_cast<std::uint32_t>(p84.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch84, 1),
                 "LaunchKernelChain(L84)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function17),
                 "DestroyCuFunction(module17)");
    RequireNvapi(destroy_module(gpu.device.Get(), module17),
                 "DestroyCuModule(module17)");

    auto color84 = ReadTexture(gpu, final_color);
    auto mask84 = ReadTexture(gpu, tail_output_mask);
    auto surface_a_after84 = ReadTexture(gpu, surface_a);
    auto surface_b_after84 = ReadTexture(gpu, surface_b);
    auto l83_after84_guarded = ReadBuffer(gpu, l83.d3d);
    std::array<std::vector<std::uint8_t>, 4> pyramid84_guarded{};
    std::array<std::vector<std::uint8_t>, 4> pyramid84{};
    for (std::size_t index = 0; index < pyramid.size(); ++index) {
      pyramid84_guarded[index] = ReadBuffer(gpu, pyramid[index]);
      pyramid84[index] =
          Payload(pyramid84_guarded[index], kTailPyramidBytes[index]);
    }
    const auto checkpoint84 =
        Join({&color84, &mask84, &pyramid84[0], &pyramid84[1],
              &pyramid84[2], &pyramid84[3]});
    Restore(gpu, final_color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu, tail_output_mask,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for (auto& item : pyramid)
      Restore(gpu, item, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    constexpr std::array<std::size_t, 4> kL85Bytes = {
        20u * 12u * 8u, 10u * 6u * 8u, 5u * 3u * 8u, 3u * 2u * 8u};
    std::array<GpuBuffer, 4> l85_output{};
    std::array<std::vector<std::uint8_t>, 4> l85_initial{};
    for (std::size_t index = 0; index < l85_output.size(); ++index) {
      l85_initial[index] = PoisonGuarded(kL85Bytes[index], poison);
      l85_output[index] = UploadBuffer(gpu, l85_initial[index]);
    }
    ExecuteAndReset(gpu);
    const auto module18_bytes = Read(modules / ModuleName(18));
    if (module18_bytes.size() != 15128u ||
        Sha256(module18_bytes) !=
            "07877ed3dcec1977a5ddfd7843314a5044ed9559bb2d222713906afdfc609d1c")
      throw std::runtime_error("connected module18 identity mismatch");
    CUmodule module18 = nullptr;
    CUfunction function18 = nullptr;
    connected::CudaCheck(cuModuleLoadData(&module18, module18_bytes.data()),
                         "load module18");
    connected::CudaCheck(
        cuModuleGetFunction(&function18, module18, "main_kernel"),
        "get module18 function");
    std::array<std::uint8_t, 56> p85{};
    PackU64(&p85, 0,
            pyramid[3].resource->GetGPUVirtualAddress() + kGuardBytes);
    for (std::size_t index = 0; index < l85_output.size(); ++index)
      PackU64(&p85, 8u + index * 8u,
              l85_output[index].resource->GetGPUVirtualAddress() +
                  kGuardBytes);
    PackU32(&p85, 40, 1);
    PackU32(&p85, 44, 40);
    PackU32(&p85, 48, 23);
    LaunchCudaPacked(function18, p85.data(), p85.size(), 3, 2, 1, 256, 1,
                     1);
    auto l85_guarded = ReadBuffer(gpu, l85_output[0]);
    auto l85_bytes = Payload(l85_guarded, kL85Bytes[0]);
    for (auto& item : l85_output)
      Restore(gpu, item, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    connected::CudaCheck(cuModuleUnload(module18), "unload module18");

    const auto module19_bytes = Read(modules / ModuleName(19));
    if (module19_bytes.size() != 57792u ||
        Sha256(module19_bytes) !=
            "143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0")
      throw std::runtime_error("connected module19 identity mismatch");
    CUmodule module19 = nullptr;
    CUfunction function19 = nullptr;
    connected::CudaCheck(cuModuleLoadData(&module19, module19_bytes.data()),
                         "load module19");
    connected::CudaCheck(
        cuModuleGetFunction(&function19, module19, "main_kernel"),
        "get module19 function");
    std::array<std::uint8_t, 80> p86{};
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      PackU64(&p86, index * 8u,
              pyramid[index].resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU64(&p86, 32,
            l85_output[0].resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU32(&p86, 40, 320);
    PackU32(&p86, 44, 180);
    LaunchCudaPacked(function19, p86.data(), p86.size(), 20, 12, 1, 256, 1,
                     1);
    connected::CudaCheck(cuModuleUnload(module19), "unload module19");
    std::array<std::vector<std::uint8_t>, 4> pyramid86_guarded{};
    std::array<std::vector<std::uint8_t>, 4> pyramid86{};
    for (std::size_t index = 0; index < pyramid.size(); ++index) {
      pyramid86_guarded[index] = ReadBuffer(gpu, pyramid[index]);
      pyramid86[index] =
          Payload(pyramid86_guarded[index], kTailPyramidBytes[index]);
    }
    std::array<std::vector<std::uint8_t>, 4> l85_after_guarded{};
    std::array<std::vector<std::uint8_t>, 4> l85_after{};
    for (std::size_t index = 0; index < l85_output.size(); ++index) {
      l85_after_guarded[index] = ReadBuffer(gpu, l85_output[index]);
      l85_after[index] = Payload(l85_after_guarded[index], kL85Bytes[index]);
    }
    const auto checkpoint86 =
        Join({&pyramid86[0], &pyramid86[1], &pyramid86[2], &pyramid86[3],
              &l85_after[0]});
    Restore(gpu, pyramid[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    const auto module20_bytes = Read(modules / ModuleName(20));
    if (module20_bytes.size() != 26024u ||
        Sha256(module20_bytes) !=
            "6cc983a76decaed1a8c7f207a95531bece5a1ad092c4db3a0159b590c331c24b")
      throw std::runtime_error("connected module20 identity mismatch");
    NvObjectHandle module20 = nullptr;
    NvObjectHandle function20 = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module20_bytes.data(),
                               static_cast<std::uint32_t>(module20_bytes.size()),
                               &module20),
                 "CreateCuModule(module20)");
    RequireNvapi(create_function(gpu.device.Get(), module20, "main_kernel",
                                 &function20),
                 "CreateCuFunction(module20)");
    std::array<std::uint8_t, 72> p87{};
    PackU64(&p87, 0, tail_output_mask_texture);
    PackU64(&p87, 8, final_color_surface);
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      PackU64(&p87, 24u + index * 8u,
              pyramid[index].resource->GetGPUVirtualAddress() + kGuardBytes);
    PackU32(&p87, 56, 10);
    PackU32(&p87, 60, 640);
    PackU32(&p87, 64, 360);
    const KernelLaunchParams launch87{function20, {40, 23, 1},
                                      {256, 1, 1}, 0, p87.data(),
                                      static_cast<std::uint32_t>(p87.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch87, 1),
                 "LaunchKernelChain(L87)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function20),
                 "DestroyCuFunction(module20)");
    RequireNvapi(destroy_module(gpu.device.Get(), module20),
                 "DestroyCuModule(module20)");
    auto final_bytes = ReadTexture(gpu, final_color);

    const auto l52_checkpoint =
        Join({&l52_a_bytes, &l52_b_bytes, &l52_confidence_bytes});
    const auto l53_checkpoint = Join({&l53_a_bytes, &l53_b_bytes});
    const auto l55_checkpoint =
        Join({&surface_a_bytes, &surface_b_bytes, &surface_mask_bytes,
              &linear_a_bytes, &linear_b_bytes, &linear_mask_bytes});
    const auto l53_a_post = ReadBuffer(gpu, l53_a);
    const auto l53_b_post = ReadBuffer(gpu, l53_b);
    const auto linear_a_post = ReadBuffer(gpu, linear_a.d3d);
    const auto linear_b_post = ReadBuffer(gpu, linear_b.d3d);
    const auto linear_mask_post = ReadBuffer(gpu, linear_mask.d3d);
    std::size_t canaries = prefix_result.canary_mismatched_bytes;
    canaries += CanaryMismatches(l51_guarded, kL51Bytes);
    canaries += CanaryMismatches(l53_a_guarded, l53::kOutputBytes);
    canaries += CanaryMismatches(l53_b_guarded, l53::kOutputBytes);
    canaries += CanaryMismatches(l53_a_post, l53::kOutputBytes);
    canaries += CanaryMismatches(l53_b_post, l53::kOutputBytes);
    canaries += CanaryMismatches(linear_a_guarded, kLinearRgbaBytes);
    canaries += CanaryMismatches(linear_b_guarded, kLinearRgbaBytes);
    canaries += CanaryMismatches(linear_mask_guarded, kLinearMaskBytes);
    canaries += CanaryMismatches(linear_a_post, kLinearRgbaBytes);
    canaries += CanaryMismatches(linear_b_post, kLinearRgbaBytes);
    canaries += CanaryMismatches(linear_mask_post, kLinearMaskBytes);
    canaries += core_result.canary_mismatched_bytes;
    canaries += CanaryMismatches(l83_guarded, fg_connected_custom::kL83Bytes);
    canaries +=
        CanaryMismatches(l83_after84_guarded, fg_connected_custom::kL83Bytes);
    for (std::size_t index = 0; index < pyramid.size(); ++index)
      canaries +=
          CanaryMismatches(pyramid86_guarded[index], kTailPyramidBytes[index]);
    for (std::size_t index = 0; index < l85_output.size(); ++index)
      canaries += CanaryMismatches(l85_after_guarded[index], kL85Bytes[index]);
    std::size_t input_mismatches = prefix_result.input_model_mismatched_bytes;
    input_mismatches += core_result.input_model_mismatched_bytes;
    input_mismatches += Mismatches(l52_depth, ReadTexture(gpu, l52_depth_texture));
    input_mismatches += Mismatches(l52_a_bytes, ReadTexture(gpu, l52_a));
    input_mismatches += Mismatches(l52_b_bytes, ReadTexture(gpu, l52_b));
    input_mismatches += Mismatches(l52_confidence_bytes,
                                    ReadTexture(gpu, l52_confidence));
    input_mismatches += Mismatches(l53_external[0], ReadTexture(gpu,l53_flow_a));
    input_mismatches += Mismatches(l53_external[1], ReadTexture(gpu,l53_flow_b));
    input_mismatches += Mismatches(l53_external[2], ReadTexture(gpu,l53_depth_a));
    input_mismatches += Mismatches(l53_external[3], ReadTexture(gpu,l53_depth_b));
    input_mismatches += Mismatches(l53_a_guarded, l53_a_post);
    input_mismatches += Mismatches(l53_b_guarded, l53_b_post);
    input_mismatches += Mismatches(l54_bytes, ReadTexture(gpu, l54_blend));
    input_mismatches += Mismatches(l55_external.rgba_a, ReadTexture(gpu, rgba_a));
    input_mismatches += Mismatches(l55_external.rgba_b, ReadTexture(gpu, rgba_b));
    input_mismatches += Mismatches(l55_external.motion, ReadTexture(gpu, motion));
    input_mismatches += Mismatches(surface_a_bytes, surface_a_after84);
    input_mismatches += Mismatches(surface_b_bytes, surface_b_after84);
    input_mismatches += Mismatches(linear_a_guarded, linear_a_post);
    input_mismatches += Mismatches(linear_b_guarded, linear_b_post);
    input_mismatches += Mismatches(linear_mask_guarded, linear_mask_post);
    input_mismatches += Mismatches(tail_mask_input, ReadTexture(gpu, tail_mask));
    input_mismatches += Mismatches(
        l83_bytes, Payload(l83_after84_guarded, fg_connected_custom::kL83Bytes));
    std::size_t readonly_tail_mismatches = 0;
    for (std::size_t index = 1; index < pyramid.size(); ++index)
      readonly_tail_mismatches +=
          Mismatches(pyramid84[index], pyramid86[index]);
    readonly_tail_mismatches += Mismatches(l85_bytes, l85_after[0]);
    for (std::size_t index = 1; index < l85_output.size(); ++index)
      readonly_tail_mismatches +=
          Mismatches(l85_initial[index], l85_after_guarded[index]);
    const bool pass = canaries == 0 && input_mismatches == 0 &&
                      readonly_tail_mismatches == 0;
    Write(output_directory / "l51.bin", l51_bytes);
    Write(output_directory / "l52.bin", l52_checkpoint);
    Write(output_directory / "l53.bin", l53_checkpoint);
    Write(output_directory / "l54.bin", l54_bytes);
    Write(output_directory / "l55.bin", l55_checkpoint);
    Write(output_directory / "l83.bin", l83_bytes);
    Write(output_directory / "l84.bin", checkpoint84);
    Write(output_directory / "l85.bin", l85_bytes);
    Write(output_directory / "l86.bin", checkpoint86);
    Write(output_directory / "output.bin", final_bytes);
    std::ofstream receipt(output_directory / "run-receipt.json");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.connected-l20-l87-run.v1\",\n"
            << "  \"backend\": \"rtx-cuda-external-memory-d3d12-nvapi\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"external_flow_scenario\": \"small-flow\",\n"
            << "  \"selected_module_census_root_sha256\": \"324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26\",\n"
            << "  \"allocation_provenance_root_sha256\": \"8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c\",\n"
            << "  \"descriptor_provenance_root_sha256\": \"d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733\",\n"
            << "  \"launch_range\": [20, 87],\n"
            << "  \"launch_count\": 68,\n"
            << "  \"checkpoint_sha256\": {\n";
    for (std::size_t index = 0; index < 32u; ++index)
      receipt << "    \"launch-" << std::setfill('0') << std::setw(6)
              << (20u + index) << "\": \""
              << prefix_result.checkpoint_sha256[index] << "\",\n";
    receipt << "    \"launch-000052\": \"" << Sha256(l52_checkpoint)
            << "\",\n"
            << "    \"launch-000053\": \"" << Sha256(l53_checkpoint)
            << "\",\n"
            << "    \"launch-000054\": \"" << Sha256(l54_bytes)
            << "\",\n"
            << "    \"launch-000055\": \"" << Sha256(l55_checkpoint)
            << "\",\n";
    for (std::size_t index = 0; index < 27u; ++index)
      receipt << "    \"launch-" << std::setfill('0') << std::setw(6)
              << (56u + index) << "\": \""
              << core_result.checkpoint_sha256[index] << "\",\n";
    receipt << "    \"launch-000083\": \"" << Sha256(l83_bytes)
            << "\",\n"
            << "    \"launch-000084\": \"" << Sha256(checkpoint84)
            << "\",\n"
            << "    \"launch-000085\": \"" << Sha256(l85_bytes)
            << "\",\n"
            << "    \"launch-000086\": \"" << Sha256(checkpoint86)
            << "\",\n"
            << "    \"launch-000087\": \"" << Sha256(final_bytes)
            << "\"\n  },\n"
            << "  \"l53_sentinel_u64_count\": ["
            << SentinelCount(l53_a_bytes) << ", "
            << SentinelCount(l53_b_bytes) << "],\n"
            << "  \"l55_linear_input_cuda_ptrs_ready\": true,\n"
            << "  \"live_device_handoff_count\": 67,\n"
            << "  \"input_model_mismatched_bytes\": " << input_mismatches
            << ",\n"
            << "  \"canary_mismatched_bytes\": " << canaries << ",\n"
            << "  \"readonly_tail_mismatched_bytes\": "
            << readonly_tail_mismatches << ",\n"
            << "  \"host_intermediate_ingress\": false,\n"
            << "  \"expected_or_archived_tensor_ingress\": false,\n"
            << "  \"l84_color_edge\": \"l55-surface-a-b-live\",\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"final_output_bytes\": " << final_bytes.size() << ",\n"
            << "  \"final_output_sha256\": \"" << Sha256(final_bytes)
            << "\",\n"
            << "  \"result\": \"" << (pass ? "pass" : "fail")
            << "\"\n}\n";
    std::cout << "l51_sha256=" << Sha256(l51_bytes) << '\n'
              << "l52_sha256=" << Sha256(l52_checkpoint) << '\n'
              << "l53_sha256=" << Sha256(l53_checkpoint) << '\n'
              << "l54_sha256=" << Sha256(l54_bytes) << '\n'
              << "l55_sha256=" << Sha256(l55_checkpoint) << '\n'
              << "l83_sha256=" << Sha256(l83_bytes) << '\n'
              << "l84_sha256=" << Sha256(checkpoint84) << '\n'
              << "l85_sha256=" << Sha256(l85_bytes) << '\n'
              << "l86_sha256=" << Sha256(checkpoint86) << '\n'
              << "l87_sha256=" << Sha256(final_bytes) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    l83.Release();
    linear_mask.Release();
    linear_b.Release();
    linear_a.Release();
    l51.Release();
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
