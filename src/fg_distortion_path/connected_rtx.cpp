#define FG_DISTORTION_MODULE10_NO_MAIN
#include "module10_rtx.cpp"

#include "connected_contract.h"

namespace {

namespace connected = fg_distortion_connected;

void SetState(GpuContext &gpu, GpuTexture &texture,
              D3D12_RESOURCE_STATES state) {
  if (texture.state == state)
    return;
  Transition(gpu.list.Get(), texture.resource.Get(), texture.state, state);
  texture.state = state;
}

std::vector<std::uint8_t> Snapshot(GpuContext &gpu, GpuTexture &texture,
                                   D3D12_RESOURCE_STATES restore) {
  auto bytes = ReadModule10Texture(gpu, texture);
  SetState(gpu, texture, restore);
  ExecuteAndReset(gpu);
  return bytes;
}

std::vector<std::uint8_t> Coverage(const std::vector<std::uint8_t> &bytes,
                                   std::uint8_t poison) {
  std::vector<std::uint8_t> result(bytes.size());
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    if (bytes[index] == 0)
      result[index] = 1;
    else if (bytes[index] == poison)
      result[index] = 0;
    else
      throw std::runtime_error("scatter output escaped zero/poison domain");
  }
  return result;
}

std::size_t CountOnes(const std::vector<std::uint8_t> &bytes) {
  std::size_t result = 0;
  for (const auto value : bytes)
    result += value != 0;
  return result;
}

struct Nvapi {
  UniqueLibrary library{LoadLibraryW(L"nvapi64.dll")};
  QueryInterface query = reinterpret_cast<QueryInterface>(
      GetProcAddress(library.get(), "nvapi_QueryInterface"));
  Initialize initialize = Resolve<Initialize>(query, kIdInitialize, "init");
  GetMergedTextureSampler get_texture = Resolve<GetMergedTextureSampler>(
      query, kIdGetMergedTextureSampler, "texture");
  GetIndependentDescriptor get_surface = Resolve<GetIndependentDescriptor>(
      query, kIdGetIndependentDescriptor, "surface");
  CreateCuModule create_module =
      Resolve<CreateCuModule>(query, kIdCreateCuModule, "module");
  CreateCuFunction create_function =
      Resolve<CreateCuFunction>(query, kIdCreateCuFunction, "function");
  LaunchKernelChain launch =
      Resolve<LaunchKernelChain>(query, kIdLaunchKernelChain, "launch");
};

struct Descriptors {
  GpuContext &gpu;
  Nvapi &nvapi;
  ComPtr<ID3D12DescriptorHeap> resource_heap;
  ComPtr<ID3D12DescriptorHeap> sampler_heap;
  UINT increment = 0;
  D3D12_CPU_DESCRIPTOR_HANDLE resource_start{};
  D3D12_CPU_DESCRIPTOR_HANDLE sampler{};

  Descriptors(GpuContext &gpu_value, Nvapi &nvapi_value)
      : gpu(gpu_value), nvapi(nvapi_value) {
    D3D12_DESCRIPTOR_HEAP_DESC resources{};
    resources.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resources.NumDescriptors = 64;
    resources.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resources.NodeMask = 1;
    RequireHr(gpu.device->CreateDescriptorHeap(&resources,
                                               IID_PPV_ARGS(&resource_heap)),
              "resource heap");
    D3D12_DESCRIPTOR_HEAP_DESC samplers{};
    samplers.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplers.NumDescriptors = 1;
    samplers.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    samplers.NodeMask = 1;
    RequireHr(gpu.device->CreateDescriptorHeap(&samplers,
                                               IID_PPV_ARGS(&sampler_heap)),
              "sampler heap");
    increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    resource_start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    sampler = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC value{};
    value.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    value.AddressU = value.AddressV = value.AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&value, sampler);
  }

  D3D12_CPU_DESCRIPTOR_HANDLE At(UINT slot) const {
    return {resource_start.ptr + std::uint64_t{slot} * increment};
  }

  std::uint64_t Texture(GpuTexture &texture, DXGI_FORMAT format, UINT slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    const auto descriptor = At(slot);
    gpu.device->CreateShaderResourceView(texture.resource.Get(), &view,
                                         descriptor);
    MergedTextureSamplerParams parameters{};
    parameters.struct_size_in = parameters.struct_size_out = sizeof(parameters);
    parameters.device = gpu.device.Get();
    parameters.texture_descriptor.ptr = descriptor.ptr;
    parameters.sampler_descriptor.ptr = sampler.ptr;
    RequireNvapi(nvapi.get_texture(&parameters), "texture handle");
    return parameters.texture_handle;
  }

  std::uint64_t Surface(GpuTexture &texture, DXGI_FORMAT format, UINT slot) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    const auto descriptor = At(slot);
    gpu.device->CreateUnorderedAccessView(texture.resource.Get(), nullptr,
                                          &view, descriptor);
    IndependentDescriptorParams parameters{};
    parameters.struct_size_in = parameters.struct_size_out = sizeof(parameters);
    parameters.device = gpu.device.Get();
    parameters.type = IndependentDescriptorType::kSurface;
    parameters.descriptor.ptr = descriptor.ptr;
    RequireNvapi(nvapi.get_surface(&parameters), "surface handle");
    return parameters.handle;
  }
};

NvObjectHandle LoadFunction(GpuContext &gpu, Nvapi &nvapi,
                            const std::vector<std::uint8_t> &module) {
  NvObjectHandle module_object = nullptr;
  NvObjectHandle function = nullptr;
  RequireNvapi(nvapi.create_module(gpu.device.Get(), module.data(),
                                   static_cast<std::uint32_t>(module.size()),
                                   &module_object),
               "create module");
  RequireNvapi(nvapi.create_function(gpu.device.Get(), module_object,
                                     "main_kernel", &function),
               "create function");
  return function;
}

void Launch(GpuContext &gpu, Nvapi &nvapi, NvObjectHandle function,
            const std::array<std::uint8_t, 56> &parameters) {
  KernelLaunchParams launch{function,
                            {20, 12, 1},
                            {16, 16, 1},
                            0,
                            const_cast<std::uint8_t *>(parameters.data()),
                            56};
  RequireNvapi(nvapi.launch(gpu.list.Get(), &launch, 1), "launch module10");
  ExecuteAndReset(gpu);
}

void Launch(GpuContext &gpu, Nvapi &nvapi, NvObjectHandle function,
            const std::array<std::uint8_t, 72> &parameters) {
  KernelLaunchParams launch{function,
                            {20, 12, 1},
                            {16, 16, 1},
                            0,
                            const_cast<std::uint8_t *>(parameters.data()),
                            72};
  RequireNvapi(nvapi.launch(gpu.list.Get(), &launch, 1), "launch module9");
  ExecuteAndReset(gpu);
}

void Launch(GpuContext &gpu, Nvapi &nvapi, NvObjectHandle function,
            const std::array<std::uint8_t, 48> &parameters) {
  KernelLaunchParams launch{function,
                            {20, 12, 1},
                            {16, 16, 1},
                            0,
                            const_cast<std::uint8_t *>(parameters.data()),
                            48};
  RequireNvapi(nvapi.launch(gpu.list.Get(), &launch, 1), "launch module25");
  ExecuteAndReset(gpu);
}

std::array<std::uint8_t, 56> Module10Parameters(std::uint64_t input,
                                                std::uint64_t output) {
  std::array<std::uint8_t, 56> result{};
  Pack64(result, 0, input);
  Pack32(result, 16, connected::kWidth);
  Pack32(result, 20, connected::kHeight);
  Pack32(result, 24, 0x3b4ccccdu);
  Pack32(result, 28, 0x3bb60b61u);
  Pack64(result, 32, output);
  Pack32(result, 40, connected::kWidth);
  Pack32(result, 44, connected::kHeight);
  Pack32(result, 52, 0x3f800000u);
  return result;
}

std::array<std::uint8_t, 72>
Module9Parameters(std::array<std::uint64_t, 5> inputs,
                  std::array<std::uint64_t, 3> outputs) {
  std::array<std::uint8_t, 72> result{};
  for (std::size_t index = 0; index < inputs.size(); ++index)
    Pack64(result, index * 8u, inputs[index]);
  Pack32(result, 40, connected::kWidth);
  Pack32(result, 44, connected::kHeight);
  for (std::size_t index = 0; index < outputs.size(); ++index)
    Pack64(result, 48u + index * 8u, outputs[index]);
  return result;
}

std::array<std::uint8_t, 48> Module25Parameters(std::uint64_t input,
                                                std::uint64_t output) {
  std::array<std::uint8_t, 48> result{};
  Pack64(result, 0, input);
  Pack64(result, 8, output);
  Pack32(result, 16, 0);
  Pack32(result, 20, connected::kWidth);
  Pack32(result, 24, connected::kHeight);
  Pack32(result, 28, 0x3b4ccccdu);
  Pack32(result, 32, 0x3bb60b61u);
  Pack32(result, 36, connected::kWidth);
  Pack32(result, 40, connected::kHeight);
  Pack32(result, 44, 0x414f5c29u);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 7)
    return 2;
  try {
    const fs::path module10_path = argv[1];
    const fs::path module9_path = argv[2];
    const fs::path module25_path = argv[3];
    const fs::path output_root = argv[4];
    const std::string run_id = argv[5];
    const std::uint8_t poison = ParsePoison(argv[6]);
    fs::create_directories(output_root);
    const auto module10 = Read(module10_path);
    const auto module9 = Read(module9_path);
    const auto module25 = Read(module25_path);
    if (module10.size() != fg_distortion_module10::kModuleBytes ||
        Sha256(module10) != fg_distortion_module10::kModuleSha256 ||
        module9.size() != fg_distortion_module9::kModuleBytes ||
        Sha256(module9) != fg_distortion_module9::kModuleSha256 ||
        module25.size() != 6640 ||
        Sha256(module25) != "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724"
                            "bfe7e70dc4e824") {
      throw std::runtime_error("connected module identity");
    }

    const auto external_reset = connected::Distortion(0);
    const auto previous_generated = connected::Distortion(1);
    const auto external_generated = connected::Distortion(2);
    const auto flow32 = connected::Flow32();
    const auto flow16 = connected::Flow16();
    const auto mask_reset = connected::Mask(0);
    const auto mask_generated = connected::Mask(1);
    std::vector<std::uint8_t> rgba_poison(connected::kRgba16Bytes, poison);
    std::vector<std::uint8_t> rg_poison(connected::kRg16Bytes, poison);
    std::vector<std::uint8_t> r32_poison(connected::kR32Bytes, poison);
    std::vector<std::uint8_t> r8_poison(connected::kR8Bytes, poison);
    std::vector<std::uint8_t> rgba_canary(connected::kRgba16Bytes,
                                          poison ^ 0xffu);
    std::vector<std::uint8_t> rg_canary(connected::kRg16Bytes, poison ^ 0xffu);
    std::vector<std::uint8_t> r32_canary(connected::kR32Bytes, poison ^ 0xffu);
    std::vector<std::uint8_t> r8_canary(connected::kR8Bytes, poison ^ 0xffu);

    GpuContext gpu = CreateGpuContext();
    const auto texture = [&](DXGI_FORMAT format, std::uint32_t bytes,
                             D3D12_RESOURCE_FLAGS flags,
                             D3D12_RESOURCE_STATES state,
                             const std::vector<std::uint8_t> &value) {
      return UploadModule10Texture(gpu, connected::kWidth, connected::kHeight,
                                   format, bytes, flags, state, value);
    };
    auto ext_reset =
        texture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, external_reset);
    auto ext_generated = texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, external_generated);
    auto internal_reset =
        texture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rgba_poison);
    auto internal_generated = texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, previous_generated);
    auto flow32_texture =
        texture(DXGI_FORMAT_R32G32_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, flow32);
    auto flow16_texture =
        texture(DXGI_FORMAT_R16G16_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, flow16);
    auto mask_reset_texture = texture(
        DXGI_FORMAT_R32_UINT, 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mask_reset);
    auto mask_generated_texture =
        texture(DXGI_FORMAT_R32_UINT, 4, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mask_generated);
    auto flow_a = texture(DXGI_FORMAT_R16G16_FLOAT, 4,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rg_poison);
    auto flow_b = texture(DXGI_FORMAT_R16G16_FLOAT, 4,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rg_poison);
    auto reset_mask_output = texture(
        DXGI_FORMAT_R32_UINT, 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, r32_poison);
    auto scatter_reset = texture(
        DXGI_FORMAT_R8_UNORM, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, r8_poison);
    auto scatter_generated = texture(
        DXGI_FORMAT_R8_UNORM, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, r8_poison);
    auto canary_rgba =
        texture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rgba_canary);
    auto canary_rg = texture(DXGI_FORMAT_R16G16_FLOAT, 4,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rg_canary);
    auto canary_r32 = texture(
        DXGI_FORMAT_R32_UINT, 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, r32_canary);
    auto canary_r8 = texture(DXGI_FORMAT_R8_UNORM, 1,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, r8_canary);
    ExecuteAndReset(gpu);

    Nvapi nvapi;
    RequireNvapi(nvapi.initialize(), "initialize");
    Descriptors descriptors(gpu, nvapi);
    const auto ext_reset_t =
        descriptors.Texture(ext_reset, DXGI_FORMAT_R16G16B16A16_FLOAT, 0);
    const auto ext_generated_t =
        descriptors.Texture(ext_generated, DXGI_FORMAT_R16G16B16A16_FLOAT, 1);
    const auto internal_reset_t =
        descriptors.Texture(internal_reset, DXGI_FORMAT_R16G16B16A16_FLOAT, 2);
    const auto internal_generated_t = descriptors.Texture(
        internal_generated, DXGI_FORMAT_R16G16B16A16_FLOAT, 3);
    const auto flow32_t =
        descriptors.Texture(flow32_texture, DXGI_FORMAT_R32G32_FLOAT, 4);
    const auto flow16_t =
        descriptors.Texture(flow16_texture, DXGI_FORMAT_R16G16_FLOAT, 5);
    const auto mask_reset_t =
        descriptors.Texture(mask_reset_texture, DXGI_FORMAT_R32_UINT, 6);
    const auto mask_generated_t =
        descriptors.Texture(mask_generated_texture, DXGI_FORMAT_R32_UINT, 7);
    const auto flow_b_t =
        descriptors.Texture(flow_b, DXGI_FORMAT_R16G16_FLOAT, 8);
    const auto internal_reset_s =
        descriptors.Surface(internal_reset, DXGI_FORMAT_R16G16B16A16_FLOAT, 32);
    const auto internal_generated_s = descriptors.Surface(
        internal_generated, DXGI_FORMAT_R16G16B16A16_FLOAT, 33);
    const auto flow_a_s =
        descriptors.Surface(flow_a, DXGI_FORMAT_R16G16_FLOAT, 34);
    const auto flow_b_s =
        descriptors.Surface(flow_b, DXGI_FORMAT_R16G16_FLOAT, 35);
    const auto reset_mask_s =
        descriptors.Surface(reset_mask_output, DXGI_FORMAT_R32_UINT, 36);
    const auto generated_mask_s =
        descriptors.Surface(mask_reset_texture, DXGI_FORMAT_R32_UINT, 37);
    const auto scatter_reset_s =
        descriptors.Surface(scatter_reset, DXGI_FORMAT_R8_UNORM, 38);
    const auto scatter_generated_s =
        descriptors.Surface(scatter_generated, DXGI_FORMAT_R8_UNORM, 39);

    const auto function10 = LoadFunction(gpu, nvapi, module10);
    const auto function9 = LoadFunction(gpu, nvapi, module9);
    const auto function25 = LoadFunction(gpu, nvapi, module25);

    const auto parameters10_reset =
        Module10Parameters(ext_reset_t, internal_reset_s);
    Launch(gpu, nvapi, function10, parameters10_reset);
    const auto checkpoint10_reset = Snapshot(
        gpu, internal_reset, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    const auto parameters9_reset =
        Module9Parameters({internal_reset_t, internal_generated_t, flow32_t,
                           flow16_t, mask_reset_t},
                          {flow_a_s, flow_b_s, reset_mask_s});
    Launch(gpu, nvapi, function9, parameters9_reset);
    const auto checkpoint9_reset_a =
        Snapshot(gpu, flow_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto checkpoint9_reset_b =
        Snapshot(gpu, flow_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto checkpoint9_reset_mask =
        ReadModule10Texture(gpu, reset_mask_output);

    const auto parameters25_reset =
        Module25Parameters(flow_b_t, scatter_reset_s);
    Launch(gpu, nvapi, function25, parameters25_reset);
    const auto checkpoint25_reset = ReadModule10Texture(gpu, scatter_reset);

    SetState(gpu, internal_generated, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto parameters10_generated =
        Module10Parameters(ext_generated_t, internal_generated_s);
    Launch(gpu, nvapi, function10, parameters10_generated);
    const auto checkpoint10_generated =
        Snapshot(gpu, internal_generated,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    SetState(gpu, flow_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    SetState(gpu, mask_reset_texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto parameters9_generated =
        Module9Parameters({internal_generated_t, internal_reset_t, flow32_t,
                           flow16_t, mask_generated_t},
                          {flow_a_s, flow_b_s, generated_mask_s});
    Launch(gpu, nvapi, function9, parameters9_generated);
    const auto checkpoint9_generated_a = ReadModule10Texture(gpu, flow_a);
    const auto checkpoint9_generated_b =
        Snapshot(gpu, flow_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto checkpoint9_generated_mask =
        ReadModule10Texture(gpu, mask_reset_texture);

    const auto parameters25_generated =
        Module25Parameters(flow_b_t, scatter_generated_s);
    Launch(gpu, nvapi, function25, parameters25_generated);
    const auto checkpoint25_generated =
        ReadModule10Texture(gpu, scatter_generated);

    const auto reset_coverage = Coverage(checkpoint25_reset, poison);
    const auto generated_coverage = Coverage(checkpoint25_generated, poison);
    const auto external_reset_after = ReadModule10Texture(gpu, ext_reset);
    const auto external_generated_after =
        ReadModule10Texture(gpu, ext_generated);
    const auto flow32_after = ReadModule10Texture(gpu, flow32_texture);
    const auto flow16_after = ReadModule10Texture(gpu, flow16_texture);
    const auto mask_generated_after =
        ReadModule10Texture(gpu, mask_generated_texture);
    const std::size_t input_mismatches =
        Mismatches(external_reset, external_reset_after) +
        Mismatches(external_generated, external_generated_after) +
        Mismatches(flow32, flow32_after) + Mismatches(flow16, flow16_after) +
        Mismatches(mask_generated, mask_generated_after);
    const std::size_t handoff_mismatches =
        Mismatches(external_reset, checkpoint10_reset) +
        Mismatches(external_generated, checkpoint10_generated);
    const std::size_t canary_mismatches =
        Mismatches(rgba_canary, ReadModule10Texture(gpu, canary_rgba)) +
        Mismatches(rg_canary, ReadModule10Texture(gpu, canary_rg)) +
        Mismatches(r32_canary, ReadModule10Texture(gpu, canary_r32)) +
        Mismatches(r8_canary, ReadModule10Texture(gpu, canary_r8));

    const std::array<std::pair<const char *, const std::vector<std::uint8_t> *>,
                     10>
        checkpoints = {{{"reset-m10.rgba16f", &checkpoint10_reset},
                        {"reset-m9-a.rg16f", &checkpoint9_reset_a},
                        {"reset-m9-b.rg16f", &checkpoint9_reset_b},
                        {"reset-m9-mask.r32u", &checkpoint9_reset_mask},
                        {"reset-m25.r8", &checkpoint25_reset},
                        {"generated-m10.rgba16f", &checkpoint10_generated},
                        {"generated-m9-a.rg16f", &checkpoint9_generated_a},
                        {"generated-m9-b.rg16f", &checkpoint9_generated_b},
                        {"generated-m9-mask.r32u", &checkpoint9_generated_mask},
                        {"generated-m25.r8", &checkpoint25_generated}}};
    for (const auto &[name, bytes] : checkpoints)
      Write(output_root / name, *bytes);
    Write(output_root / "reset-m25.coverage", reset_coverage);
    Write(output_root / "generated-m25.coverage", generated_coverage);

    const bool pass = input_mismatches == 0 && handoff_mismatches == 0 &&
                      canary_mismatches == 0;
    std::ofstream receipt(output_root / "run-receipt.json");
    receipt << "{\n\"schema\":\"neuralstudio.fg.distortion-six-launch-run.v1\","
               "\n\"backend\":\"rtx-d3d12-nvapi\",\n\"run_id\":\""
            << run_id << "\",\n\"process_id\":" << GetCurrentProcessId()
            << ",\n\"poison_byte\":" << unsigned(poison)
            << ",\n\"module_sha256\":[\""
            << fg_distortion_module10::kModuleSha256 << "\",\""
            << fg_distortion_module9::kModuleSha256
            << "\","
               "\"28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e"
               "824\"],"
               "\n\"launch_sequence\":[10,9,25,10,9,25],"
               "\n\"selected_launch_tokens\":[\"launch-000005\",\"launch-"
               "000009\",\"launch-000010\",\"launch-000016\",\"launch-000020\","
               "\"launch-000021\"],"
               "\n\"selected_raw_parameter_sha256\":["
               "\"e170b309e44af03949713aa49e319aa27448cc33a002f7a415006d2b3acbf"
               "a71\","
               "\"b695e0e6d29050462620ad1f54b593dd57b12a616cb58e2367ffef10a2444"
               "fe2\","
               "\"98f06561b037fdfab9742649c3a82e58c39de0038b5df10f638ede5df5aa6"
               "b55\","
               "\"9c50edeea15f0b497785e9cb6a7e8ca9324992df32f12ce3bba553f55199b"
               "4a5\","
               "\"723e2c4a348298753ace7172a6e1452e22faf4518c10952ab7857f7eaf840"
               "6c3\","
               "\"a765a6a6e229ab59eb65faf1393caf9ccff44d949b23f9af11ad0d957f77c"
               "e4c\"],"
               "\n\"input_generator\":\""
            << connected::kInputGenerator
            << "\",\n\"immutable_input_pre_sha256\":[\""
            << Sha256(external_reset) << "\",\"" << Sha256(external_generated)
            << "\",\"" << Sha256(flow32) << "\",\"" << Sha256(flow16) << "\",\""
            << Sha256(mask_generated)
            << "\"],\n\"immutable_input_post_sha256\":[\""
            << Sha256(external_reset_after) << "\",\""
            << Sha256(external_generated_after) << "\",\""
            << Sha256(flow32_after) << "\",\"" << Sha256(flow16_after)
            << "\",\"" << Sha256(mask_generated_after)
            << "\"],\n\"initial_temporal_state_sha256\":[\""
            << Sha256(previous_generated) << "\",\"" << Sha256(mask_reset)
            << "\"],\n\"checkpoint_sha256\":[";
    for (std::size_t index = 0; index < checkpoints.size(); ++index) {
      if (index)
        receipt << ',';
      receipt << '\"' << Sha256(*checkpoints[index].second) << '\"';
    }
    receipt << "],\n\"checkpoint_bytes\":[460800,230400,230400,230400,57600,"
               "460800,230400,230400,230400,57600],"
               "\n\"reset_coverage_sha256\":\""
            << Sha256(reset_coverage) << "\",\n\"generated_coverage_sha256\":\""
            << Sha256(generated_coverage)
            << "\",\n\"reset_written_pixels\":" << CountOnes(reset_coverage)
            << ",\n\"generated_written_pixels\":"
            << CountOnes(generated_coverage)
            << ",\n\"input_mismatched_bytes\":" << input_mismatches
            << ",\n\"handoff_mismatched_bytes\":" << handoff_mismatches
            << ",\n\"canary_mismatched_bytes\":" << canary_mismatches
            << ",\n\"full_output_poison_pixels\":"
            << (CountPoisonPixels(checkpoint10_reset, 8, poison) +
                CountPoisonPixels(checkpoint9_reset_a, 4, poison) +
                CountPoisonPixels(checkpoint9_reset_b, 4, poison) +
                CountPoisonPixels(checkpoint9_reset_mask, 4, poison) +
                CountPoisonPixels(checkpoint10_generated, 8, poison) +
                CountPoisonPixels(checkpoint9_generated_a, 4, poison) +
                CountPoisonPixels(checkpoint9_generated_b, 4, poison) +
                CountPoisonPixels(checkpoint9_generated_mask, 4, poison))
            << ",\n\"live_device_handoffs\":true,"
               "\n\"host_roundtrip_used_as_launch_input\":false,"
               "\n\"declared_region_coverage_accounted\":true,"
               "\n\"expected_output_fed_to_execution\":false,"
               "\n\"captured_or_archived_intermediate_consumed\":false,"
               "\n\"address_fields_recorded\":false,\n\"result\":\""
            << (pass ? "pass" : "fail") << "\"\n}\n";
    return pass ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
