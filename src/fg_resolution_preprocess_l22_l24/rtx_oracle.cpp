#define main fg_mid_l52_embedded_main
#include "../fg_mid_l52/rtx_oracle.cpp"
#undef main

#include <algorithm>

#include "contract.h"

namespace {
namespace contract = fg_resolution_l22_l24;

template <std::size_t Size>
void RPack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
             std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void RPack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
             std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> GuardedPayload(
    const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> result(payload.size() + 2u * kGuardBytes,
                                   kGuardValue);
  std::memcpy(result.data() + kGuardBytes, payload.data(), payload.size());
  return result;
}

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t bytes) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes + bytes);
}

struct DynamicTexture {
  ComPtr<ID3D12Resource> resource;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t bytes_per_pixel = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

DynamicTexture UploadDynamicTexture(GpuContext& gpu, std::uint32_t width,
                                    std::uint32_t height, DXGI_FORMAT format,
                                    std::uint32_t bytes_per_pixel,
                                    const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != std::size_t{width} * height * bytes_per_pixel) {
    throw std::runtime_error("dynamic texture payload size mismatch");
  }
  DynamicTexture result;
  result.width = width;
  result.height = height;
  result.bytes_per_pixel = bytes_per_pixel;
  result.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
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
  UINT64 row_size = 0, upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint,
                                    &rows, &row_size, &upload_size);
  auto upload = CreateBuffer(gpu.device.Get(), upload_size,
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(dynamic texture upload)");
  const std::size_t source_row = std::size_t{width} * bytes_per_pixel;
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
             D3D12_RESOURCE_STATE_COPY_DEST, result.state);
  gpu.pending.push_back(upload);
  return result;
}

std::vector<std::uint8_t> ReadDynamicTexture(GpuContext& gpu,
                                             DynamicTexture& source) {
  const auto description = source.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0, readback_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint,
                                    &rows, &row_size, &readback_size);
  auto readback = CreateBuffer(gpu.device.Get(), readback_size,
                               D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), source.resource.Get(), source.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  source.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  D3D12_TEXTURE_COPY_LOCATION from{};
  from.pResource = source.resource.Get();
  from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION to{};
  to.pResource = readback.Get();
  to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  to.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
  ExecuteAndReset(gpu);
  const std::size_t output_row =
      std::size_t{source.width} * source.bytes_per_pixel;
  std::vector<std::uint8_t> result(output_row * source.height);
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(readback_size)};
  RequireHr(readback->Map(0, &read_range, &mapped),
            "Map(dynamic texture readback)");
  for (std::uint32_t row = 0; row < source.height; ++row) {
    std::memcpy(result.data() + std::size_t{row} * output_row,
                static_cast<const std::uint8_t*>(mapped) +
                    std::size_t{row} * footprint.Footprint.RowPitch,
                output_row);
  }
  const D3D12_RANGE no_write{0, 0};
  readback->Unmap(0, &no_write);
  return result;
}

std::size_t ByteCount(const std::vector<std::uint8_t>& bytes,
                      std::uint8_t value) {
  return static_cast<std::size_t>(std::count(bytes.begin(), bytes.end(), value));
}

void WriteReceipt(
    const fs::path& path, const std::string& run_id,
    const contract::Resolution& resolution, bool generated,
    std::uint8_t poison, const std::vector<std::uint8_t>& module22_input,
    const std::array<std::vector<std::uint8_t>, 4>& module22_outputs,
    const std::vector<std::uint8_t>& module24_mask,
    const std::array<std::vector<std::uint8_t>, 4>& module24_levels,
    const std::vector<std::uint8_t>& module24_output,
    std::size_t input_mismatches, std::size_t canary_mismatches) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write resolution receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.resolution-l22-l24-run.v1\",\n"
          << "  \"backend\": \"rtx-d3d12-nvapi\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"external_extent\": [" << resolution.external_width << ", "
          << resolution.external_height << "],\n"
          << "  \"working_extent\": [" << resolution.working_width << ", "
          << resolution.working_height << "],\n"
          << "  \"phase\": \"" << (generated ? "generated" : "reset")
          << "\",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module22_sha256\": \"" << contract::kModule22Sha256
          << "\",\n"
          << "  \"module24_sha256\": \"" << contract::kModule24Sha256
          << "\",\n"
          << "  \"module22_abi_bytes\": 56,\n"
          << "  \"module24_abi_bytes\": 64,\n"
          << "  \"module22_grid\": ["
          << contract::CeilDiv(contract::CeilDiv(resolution.working_width, 16), 16)
          << ", "
          << contract::CeilDiv(contract::CeilDiv(resolution.working_height, 16), 16)
          << ", 1],\n"
          << "  \"module24_grid\": ["
          << contract::CeilDiv(resolution.working_width, 16) << ", "
          << contract::CeilDiv(resolution.working_height, 16) << ", 1],\n"
          << "  \"block\": [256, 1, 1],\n"
          << "  \"selected_level\": " << resolution.selected_level << ",\n"
          << "  \"extra_level_count\": " << resolution.extra_level_count
          << ",\n"
          << "  \"input_generator\": \"" << contract::kGenerator << "\",\n"
          << "  \"module22_input_sha256\": \"" << Sha256(module22_input)
          << "\",\n"
          << "  \"module22_output_sha256\": [";
  for (std::size_t index = 0; index < module22_outputs.size(); ++index) {
    receipt << (index ? ", \"" : "\"") << Sha256(module22_outputs[index])
            << "\"";
  }
  receipt << "],\n  \"module22_output_bytes\": [";
  for (std::size_t index = 0; index < module22_outputs.size(); ++index) {
    receipt << (index ? ", " : "") << module22_outputs[index].size();
  }
  receipt << "],\n  \"module22_output_poison_byte_count\": [";
  for (std::size_t index = 0; index < module22_outputs.size(); ++index) {
    receipt << (index ? ", " : "") << ByteCount(module22_outputs[index], poison);
  }
  receipt << "],\n"
          << "  \"module24_mask_sha256\": \"" << Sha256(module24_mask)
          << "\",\n  \"module24_level_sha256\": [";
  for (std::size_t index = 0; index < module24_levels.size(); ++index) {
    receipt << (index ? ", \"" : "\"") << Sha256(module24_levels[index])
            << "\"";
  }
  receipt << "],\n"
          << "  \"module24_output_sha256\": \"" << Sha256(module24_output)
          << "\",\n"
          << "  \"module24_output_bytes\": " << module24_output.size() << ",\n"
          << "  \"module24_output_poison_byte_count\": "
          << ByteCount(module24_output, poison) << ",\n"
          << "  \"true_typed_output\": \"DXGI_FORMAT_R16G16_FLOAT\",\n"
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
    std::cerr << "usage=fg-resolution-l22-l24-rtx <module22> <module24> "
                 "<output-dir> <run-id> <poison-hex> <resolution> "
                 "<reset|generated>\n";
    return 2;
  }
  try {
    const auto module22 = Read(argv[1]);
    const auto module24 = Read(argv[2]);
    if (module22.size() != contract::kModule22Bytes ||
        Sha256(module22) != contract::kModule22Sha256 ||
        module24.size() != contract::kModule24Bytes ||
        Sha256(module24) != contract::kModule24Sha256) {
      throw std::runtime_error("selected module identity mismatch");
    }
    const fs::path output_root = argv[3];
    const std::string run_id = argv[4];
    const std::uint8_t poison = ParsePoison(argv[5]);
    const auto& resolution = contract::ParseResolution(argv[6]);
    const std::string phase_name = argv[7];
    if (phase_name != "reset" && phase_name != "generated") {
      throw std::runtime_error("invalid phase");
    }
    const bool generated = phase_name == "generated";
    fs::create_directories(output_root);
    const auto module22_input =
        contract::Module22Input(resolution, generated);
    const auto module22_initial =
        contract::Module22OutputInitial(resolution, poison);
    const auto module24_mask = contract::Mask(resolution, generated);
    const auto module24_levels = contract::Module24Levels(resolution, generated);
    const std::vector<std::uint8_t> module24_initial(
        std::size_t{resolution.working_width} * resolution.working_height * 4u,
        poison);

    GpuContext gpu = CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    auto query = reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(), "nvapi_QueryInterface"));
    if (!nvapi || !query) throw std::runtime_error("NvAPI unavailable");
    const auto initialize = Resolve<Initialize>(query, kIdInitialize, "initialize");
    const auto get_surface = Resolve<GetIndependentDescriptor>(
        query, kIdGetIndependentDescriptor, "get surface");
    const auto create_module = Resolve<CreateCuModule>(query, kIdCreateCuModule,
                                                        "create module");
    const auto create_function = Resolve<CreateCuFunction>(query, kIdCreateCuFunction,
                                                            "create function");
    const auto launch_chain = Resolve<LaunchKernelChain>(query, kIdLaunchKernelChain,
                                                          "launch");
    const auto destroy_module = Resolve<DestroyCuModule>(query, kIdDestroyCuModule,
                                                          "destroy module");
    const auto destroy_function = Resolve<DestroyCuFunction>(query, kIdDestroyCuFunction,
                                                              "destroy function");
    RequireNvapi(initialize(), "NvAPI_Initialize");

    GpuBuffer m22_input = UploadBuffer(gpu, GuardedPayload(module22_input));
    std::array<GpuBuffer, 4> m22_outputs;
    for (std::size_t index = 0; index < 4; ++index) {
      m22_outputs[index] = UploadBuffer(gpu, GuardedPayload(module22_initial[index]));
    }
    GpuBuffer m24_mask = UploadBuffer(gpu, GuardedPayload(module24_mask));
    std::array<GpuBuffer, 4> m24_levels;
    for (std::size_t index = 0; index < 4; ++index) {
      m24_levels[index] = UploadBuffer(gpu, GuardedPayload(module24_levels[index]));
    }
    DynamicTexture m24_output = UploadDynamicTexture(
        gpu, resolution.working_width, resolution.working_height,
        DXGI_FORMAT_R16G16_FLOAT, 4, module24_initial);
    ExecuteAndReset(gpu);

    D3D12_DESCRIPTOR_HEAP_DESC heap_description{};
    heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_description.NumDescriptors = 1;
    heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heap_description.NodeMask = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&heap_description,
                                                IID_PPV_ARGS(&heap)),
              "CreateDescriptorHeap");
    const auto uav_handle = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R16G16_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(m24_output.resource.Get(), nullptr,
                                          &uav, uav_handle);
    IndependentDescriptorParams surface_request{};
    surface_request.struct_size_in = sizeof(surface_request);
    surface_request.struct_size_out = sizeof(surface_request);
    surface_request.device = gpu.device.Get();
    surface_request.type = IndependentDescriptorType::kSurface;
    surface_request.descriptor.ptr = uav_handle.ptr;
    RequireNvapi(get_surface(&surface_request), "Get independent surface");

    auto execute_module = [&](const std::vector<std::uint8_t>& bytes,
                              std::array<std::uint8_t, 64>& packed,
                              std::uint32_t packed_bytes,
                              Dim3 grid) {
      NvObjectHandle module = nullptr, function = nullptr;
      RequireNvapi(create_module(gpu.device.Get(), bytes.data(),
                                 static_cast<std::uint32_t>(bytes.size()),
                                 &module), "CreateCuModule");
      RequireNvapi(create_function(gpu.device.Get(), module, "main_kernel",
                                   &function), "CreateCuFunction");
      const KernelLaunchParams launch{function, grid, {256, 1, 1}, 0,
                                      packed.data(), packed_bytes};
      RequireNvapi(launch_chain(gpu.list.Get(), &launch, 1), "LaunchKernelChain");
      ExecuteAndReset(gpu);
      RequireNvapi(destroy_function(gpu.device.Get(), function),
                   "DestroyCuFunction");
      RequireNvapi(destroy_module(gpu.device.Get(), module), "DestroyCuModule");
    };
    const std::uint32_t m22_width =
        contract::CeilDiv(resolution.working_width, 16);
    const std::uint32_t m22_height =
        contract::CeilDiv(resolution.working_height, 16);
    std::array<std::uint8_t, 64> packed22{};
    RPack64(&packed22, 0, m22_input.resource->GetGPUVirtualAddress() + kGuardBytes);
    for (std::size_t index = 0; index < 4; ++index) {
      RPack64(&packed22, 8u + index * 8u,
              m22_outputs[index].resource->GetGPUVirtualAddress() + kGuardBytes);
    }
    RPack32(&packed22, 40, resolution.extra_level_count);
    RPack32(&packed22, 44, m22_width);
    RPack32(&packed22, 48, m22_height);
    packed22[52] = packed22[53] = packed22[54] = packed22[55] = poison;
    execute_module(module22, packed22, 56,
                   {contract::CeilDiv(m22_width, 16),
                    contract::CeilDiv(m22_height, 16), 1});

    std::array<std::uint8_t, 64> packed24{};
    RPack64(&packed24, 0, surface_request.handle);
    RPack64(&packed24, 8,
            m24_mask.resource->GetGPUVirtualAddress() + kGuardBytes);
    for (std::size_t index = 0; index < 4; ++index) {
      RPack64(&packed24, 16u + index * 8u,
              m24_levels[index].resource->GetGPUVirtualAddress() + kGuardBytes);
    }
    RPack32(&packed24, 48, resolution.selected_level);
    RPack32(&packed24, 52, resolution.working_width);
    RPack32(&packed24, 56, resolution.working_height);
    packed24[60] = packed24[61] = packed24[62] = packed24[63] = poison;
    execute_module(module24, packed24, 64,
                   {contract::CeilDiv(resolution.working_width, 16),
                    contract::CeilDiv(resolution.working_height, 16), 1});

    const auto m22_input_guarded = ReadBuffer(gpu, m22_input);
    const auto m22_input_after = Payload(m22_input_guarded, module22_input.size());
    std::array<std::vector<std::uint8_t>, 4> m22_after;
    std::size_t canary_mismatches =
        CanaryMismatches(m22_input_guarded, module22_input.size());
    for (std::size_t index = 0; index < 4; ++index) {
      const auto guarded = ReadBuffer(gpu, m22_outputs[index]);
      m22_after[index] = Payload(guarded, module22_initial[index].size());
      canary_mismatches += CanaryMismatches(guarded, module22_initial[index].size());
    }
    const auto m24_mask_guarded = ReadBuffer(gpu, m24_mask);
    const auto m24_mask_after = Payload(m24_mask_guarded, module24_mask.size());
    canary_mismatches += CanaryMismatches(m24_mask_guarded, module24_mask.size());
    std::array<std::vector<std::uint8_t>, 4> m24_levels_after;
    for (std::size_t index = 0; index < 4; ++index) {
      const auto guarded = ReadBuffer(gpu, m24_levels[index]);
      m24_levels_after[index] = Payload(guarded, module24_levels[index].size());
      canary_mismatches += CanaryMismatches(guarded, module24_levels[index].size());
    }
    const auto m24_output_after = ReadDynamicTexture(gpu, m24_output);
    std::size_t input_mismatches =
        Mismatches(module22_input, m22_input_after) +
        Mismatches(module24_mask, m24_mask_after);
    for (std::size_t index = 0; index < 4; ++index) {
      input_mismatches += Mismatches(module24_levels[index],
                                     m24_levels_after[index]);
    }
    Write(output_root / "module22-input-after.bin", m22_input_after);
    for (std::size_t index = 0; index < 4; ++index) {
      Write(output_root / ("module22-output-" + std::to_string(index) + ".bin"),
            m22_after[index]);
      Write(output_root / ("module24-level-" + std::to_string(index) + "-after.bin"),
            m24_levels_after[index]);
    }
    Write(output_root / "module24-mask-after.bin", m24_mask_after);
    Write(output_root / "module24-output.bin", m24_output_after);
    WriteReceipt(output_root / "run-receipt.json", run_id, resolution,
                 generated, poison, module22_input, m22_after, module24_mask,
                 module24_levels, m24_output_after, input_mismatches,
                 canary_mismatches);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_resolution_l22_l24_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "module24_output_sha256=" << Sha256(m24_output_after) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
