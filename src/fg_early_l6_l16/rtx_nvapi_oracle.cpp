#define main fg_mid_l52_embedded_main
#include "../fg_mid_l52/rtx_oracle.cpp"
#undef main

#include <algorithm>

#include "contract.h"

namespace {
namespace contract = fg_early_l6_l16;

template <std::size_t Size>
void L6Pack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
              std::uint32_t value) {
  if (offset + 4u > Size) throw std::runtime_error("L6Pack32 OOB");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void L6Pack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
              std::uint64_t value) {
  if (offset + 8u > Size) throw std::runtime_error("L6Pack64 OOB");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> GuardedPayload(
    const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> result(payload.size() + 2u * kGuardBytes,
                                   kGuardValue);
  std::memcpy(result.data() + kGuardBytes, payload.data(), payload.size());
  return result;
}

std::vector<std::uint8_t> L6Payload(
    const std::vector<std::uint8_t>& guarded, std::size_t size) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes + size);
}

std::size_t ByteCount(const std::vector<std::uint8_t>& bytes,
                      std::uint8_t value) {
  return static_cast<std::size_t>(std::count(bytes.begin(), bytes.end(), value));
}

void WriteRunReceipt(
    const fs::path& path, const std::string& run_id, contract::Phase phase,
    std::uint8_t poison, const std::vector<std::uint8_t>& flow_before,
    const std::vector<std::uint8_t>& depth_before,
    const std::vector<std::uint8_t>& input_mask_before,
    const std::vector<std::uint8_t>& candidates_before,
    const std::vector<std::uint8_t>& flow_after,
    const std::vector<std::uint8_t>& depth_after,
    const std::vector<std::uint8_t>& input_mask_after,
    const std::vector<std::uint8_t>& candidates_after,
    const std::vector<std::uint8_t>& vector_surface,
    const std::vector<std::uint8_t>& mask_surface,
    const std::array<std::vector<std::uint8_t>, 5>& output_buffers,
    std::size_t input_mismatches, std::size_t buffer_canary_mismatches) {
  const bool pass = input_mismatches == 0 && buffer_canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write L6/L16 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.early-l6-l16-run.v1\",\n"
          << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"phase\": \"" << contract::PhaseName(phase) << "\",\n"
          << "  \"selected_launch\": "
          << (phase == contract::Phase::kReset ? 6 : 16) << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module_token\": 21,\n"
          << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
          << "  \"function_name\": \"main_kernel\",\n"
          << "  \"grid\": [20, 12, 1],\n"
          << "  \"block\": [256, 1, 1],\n"
          << "  \"packed_parameter_bytes\": 224,\n"
          << "  \"loaded_byte_156\": 0,\n"
          << "  \"padding_157_159\": [" << unsigned(poison) << ", "
          << unsigned(poison) << ", " << unsigned(poison) << "],\n"
          << "  \"input_generator\": \"" << contract::kInputGenerator
          << "\",\n"
          << "  \"flow_pre_sha256\": \"" << Sha256(flow_before) << "\",\n"
          << "  \"flow_post_sha256\": \"" << Sha256(flow_after) << "\",\n"
          << "  \"depth_pre_sha256\": \"" << Sha256(depth_before) << "\",\n"
          << "  \"depth_post_sha256\": \"" << Sha256(depth_after) << "\",\n"
          << "  \"input_mask_pre_sha256\": \""
          << Sha256(input_mask_before) << "\",\n"
          << "  \"input_mask_post_sha256\": \""
          << Sha256(input_mask_after) << "\",\n"
          << "  \"candidates_pre_sha256\": \""
          << Sha256(candidates_before) << "\",\n"
          << "  \"candidates_post_sha256\": \""
          << Sha256(candidates_after) << "\",\n"
          << "  \"vector_surface_sha256\": \"" << Sha256(vector_surface)
          << "\",\n"
          << "  \"mask_surface_sha256\": \"" << Sha256(mask_surface)
          << "\",\n"
          << "  \"output_buffer_sha256\": [";
  for (std::size_t index = 0; index < output_buffers.size(); ++index) {
    receipt << (index ? ", \"" : "\"") << Sha256(output_buffers[index])
            << "\"";
  }
  receipt << "],\n  \"output_buffer_bytes\": [";
  for (std::size_t index = 0; index < output_buffers.size(); ++index) {
    receipt << (index ? ", " : "") << output_buffers[index].size();
  }
  receipt << "],\n"
          << "  \"vector_surface_poison_byte_count\": "
          << ByteCount(vector_surface, poison) << ",\n"
          << "  \"mask_surface_poison_byte_count\": "
          << ByteCount(mask_surface, poison) << ",\n"
          << "  \"output_buffer_poison_byte_count\": [";
  for (std::size_t index = 0; index < output_buffers.size(); ++index) {
    receipt << (index ? ", " : "")
            << ByteCount(output_buffers[index], poison);
  }
  receipt << "],\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"buffer_canary_mismatched_bytes\": "
          << buffer_canary_mismatches << ",\n"
          << "  \"expected_or_captured_output_ingress\": false,\n"
          << "  \"archived_intermediate_ingress\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage=fg-early-l6-l16-rtx <module> <output-dir> <run-id> "
                 "<poison-hex> <reset|generated>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_root = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    const auto phase = contract::ParsePhase(argv[5]);
    fs::create_directories(output_root);
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected module21 identity mismatch");
    }
    const auto flow_before = contract::Flow(phase);
    const auto depth_before = contract::Depth(phase);
    const auto input_mask_before = contract::InputMask(phase);
    const auto candidates_before = contract::Candidates(phase);
    const std::vector<std::uint8_t> vector_before(
        contract::kVectorSurfaceBytes, poison);
    const std::vector<std::uint8_t> mask_surface_before(
        contract::kMaskSurfaceBytes, poison);
    std::array<std::vector<std::uint8_t>, 5> output_before;
    for (std::size_t index = 0; index < output_before.size(); ++index) {
      output_before[index].assign(contract::kOutputBufferBytes[index], poison);
    }

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if (!nvapi) throw std::runtime_error("loading real nvapi64.dll failed");
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (!query) throw std::runtime_error("nvapi_QueryInterface missing");
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

    GpuTexture flow = UploadTexture(
        gpu, DXGI_FORMAT_R32G32_FLOAT, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, flow_before);
    GpuTexture depth = UploadTexture(
        gpu, DXGI_FORMAT_R32_UINT, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, depth_before);
    GpuTexture input_mask = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, input_mask_before);
    GpuTexture vector_surface = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, vector_before);
    GpuTexture mask_surface = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 1,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, mask_surface_before);
    GpuBuffer candidates = UploadBuffer(gpu, GuardedPayload(candidates_before));
    std::array<GpuBuffer, 5> output_buffers;
    for (std::size_t index = 0; index < output_buffers.size(); ++index) {
      output_buffers[index] = UploadBuffer(gpu, GuardedPayload(output_before[index]));
    }
    ExecuteAndReset(gpu);

    D3D12_DESCRIPTOR_HEAP_DESC resource_description{};
    resource_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_description.NumDescriptors = 16;
    resource_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_description,
                                                IID_PPV_ARGS(&resource_heap)),
              "CreateDescriptorHeap(resources)");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_description{};
    sampler_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_description.NumDescriptors = 3;
    sampler_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_description,
                                                IID_PPV_ARGS(&sampler_heap)),
              "CreateDescriptorHeap(samplers)");
    const UINT increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT sampler_increment = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    const auto resource_start = resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto sampler_start = sampler_heap->GetCPUDescriptorHandleForHeapStart();
    auto resource_handle = [&](UINT slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{resource_start.ptr +
                                         std::uintptr_t{slot} * increment};
    };
    auto sampler_handle = [&](UINT slot) {
      return D3D12_CPU_DESCRIPTOR_HANDLE{sampler_start.ptr +
                                         std::uintptr_t{slot} * sampler_increment};
    };
    const std::array<DXGI_FORMAT, 3> input_formats = {
        DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R8_UNORM};
    const std::array<GpuTexture*, 3> input_textures = {
        &flow, &depth, &input_mask};
    std::array<std::uint64_t, 3> texture_objects{};
    for (UINT index = 0; index < 3; ++index) {
      D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
      srv.Format = input_formats[index];
      srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      srv.Texture2D.MipLevels = 1;
      gpu.device->CreateShaderResourceView(input_textures[index]->resource.Get(),
                                           &srv, resource_handle(index));
      D3D12_SAMPLER_DESC sampler{};
      sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
      sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
      sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
      sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
      gpu.device->CreateSampler(&sampler, sampler_handle(index));
      MergedTextureSamplerParams request{};
      request.struct_size_in = sizeof(request);
      request.struct_size_out = sizeof(request);
      request.device = gpu.device.Get();
      request.texture_descriptor.ptr = resource_handle(index).ptr;
      request.sampler_descriptor.ptr = sampler_handle(index).ptr;
      RequireNvapi(get_texture(&request), "GetCudaMergedTextureSamplerObject");
      texture_objects[index] = request.texture_handle;
    }
    const std::array<DXGI_FORMAT, 2> output_formats = {
        DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R8_UNORM};
    const std::array<GpuTexture*, 2> output_textures = {
        &vector_surface, &mask_surface};
    std::array<std::uint64_t, 2> surface_objects{};
    for (UINT index = 0; index < 2; ++index) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
      uav.Format = output_formats[index];
      uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(output_textures[index]->resource.Get(),
                                            nullptr, &uav,
                                            resource_handle(8u + index));
      IndependentDescriptorParams request{};
      request.struct_size_in = sizeof(request);
      request.struct_size_out = sizeof(request);
      request.device = gpu.device.Get();
      request.type = IndependentDescriptorType::kSurface;
      request.descriptor.ptr = resource_handle(8u + index).ptr;
      RequireNvapi(get_surface(&request), "GetCudaIndependentDescriptorObject");
      surface_objects[index] = request.handle;
    }

    NvObjectHandle module = nullptr;
    NvObjectHandle function = nullptr;
    RequireNvapi(create_module(gpu.device.Get(), module_bytes.data(),
                               static_cast<std::uint32_t>(module_bytes.size()),
                               &module), "CreateCuModule(module21)");
    RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                 &function), "CreateCuFunction(main_kernel)");
    std::array<std::uint8_t, 224> packed{};
    L6Pack64(&packed, 0, texture_objects[0]);
    L6Pack64(&packed, 8, texture_objects[1]);
    L6Pack64(&packed, 16, texture_objects[2]);
    L6Pack64(&packed, 24,
             candidates.resource->GetGPUVirtualAddress() + kGuardBytes);
    L6Pack64(&packed, 32, surface_objects[0]);
    L6Pack64(&packed, 40, surface_objects[1]);
    for (std::size_t index = 0; index < output_buffers.size(); ++index) {
      L6Pack64(&packed, 48u + index * 8u,
               output_buffers[index].resource->GetGPUVirtualAddress() +
                   kGuardBytes);
    }
    L6Pack32(&packed, 128, std::bit_cast<std::uint32_t>(1.0f / 640.0f));
    L6Pack32(&packed, 132, std::bit_cast<std::uint32_t>(1.0f / 360.0f));
    L6Pack32(&packed, 136, contract::kWidth);
    L6Pack32(&packed, 140, contract::kHeight);
    L6Pack32(&packed, 144, std::bit_cast<std::uint32_t>(40.0f));
    L6Pack32(&packed, 148, std::bit_cast<std::uint32_t>(1.0f));
    L6Pack32(&packed, 152, std::bit_cast<std::uint32_t>(600.0f));
    packed[156] = 0;
    packed[157] = poison;
    packed[158] = poison;
    packed[159] = poison;
    const std::array<float, 16> identity = {
        1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(packed.data() + 160, identity.data(), 64);
    const KernelLaunchParams launch{
        function, {20, 12, 1}, {256, 1, 1}, 0,
        packed.data(), static_cast<std::uint32_t>(packed.size())};
    RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1),
                 "LaunchKernelChain(module21/main_kernel)");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(), function),
                 "DestroyCuFunction");
    RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");

    const auto flow_after = ReadTexture(gpu, flow);
    const auto depth_after = ReadTexture(gpu, depth);
    const auto input_mask_after = ReadTexture(gpu, input_mask);
    const auto vector_after = ReadTexture(gpu, vector_surface);
    const auto mask_surface_after = ReadTexture(gpu, mask_surface);
    const auto candidates_guarded_after = ReadBuffer(gpu, candidates);
    const auto candidates_after =
        L6Payload(candidates_guarded_after, contract::kCandidateBytes);
    std::array<std::vector<std::uint8_t>, 5> output_after;
    std::size_t buffer_canary_mismatches =
        CanaryMismatches(candidates_guarded_after, contract::kCandidateBytes);
    for (std::size_t index = 0; index < output_buffers.size(); ++index) {
      const auto guarded = ReadBuffer(gpu, output_buffers[index]);
      output_after[index] =
          L6Payload(guarded, contract::kOutputBufferBytes[index]);
      buffer_canary_mismatches +=
          CanaryMismatches(guarded, contract::kOutputBufferBytes[index]);
    }
    const std::size_t input_mismatches =
        Mismatches(flow_before, flow_after) +
        Mismatches(depth_before, depth_after) +
        Mismatches(input_mask_before, input_mask_after) +
        Mismatches(candidates_before, candidates_after);
    Write(output_root / "flow-after.bin", flow_after);
    Write(output_root / "depth-after.bin", depth_after);
    Write(output_root / "input-mask-after.bin", input_mask_after);
    Write(output_root / "candidates-after.bin", candidates_after);
    Write(output_root / "vector-surface.bin", vector_after);
    Write(output_root / "mask-surface.bin", mask_surface_after);
    for (std::size_t index = 0; index < output_after.size(); ++index) {
      Write(output_root / ("output-buffer-" + std::to_string(index) + ".bin"),
            output_after[index]);
    }
    WriteRunReceipt(output_root / "run-receipt.json", run_id, phase, poison,
                    flow_before, depth_before, input_mask_before,
                    candidates_before, flow_after, depth_after,
                    input_mask_after, candidates_after, vector_after,
                    mask_surface_after, output_after, input_mismatches,
                    buffer_canary_mismatches);
    const bool pass = input_mismatches == 0 && buffer_canary_mismatches == 0;
    std::cout << "probe=fg_early_l6_l16_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "vector_surface_sha256=" << Sha256(vector_after) << '\n'
              << "mask_surface_sha256=" << Sha256(mask_surface_after) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
