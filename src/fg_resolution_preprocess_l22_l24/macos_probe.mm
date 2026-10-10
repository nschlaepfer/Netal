#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include <algorithm>

#include "contract.h"

namespace {
namespace contract = fg_resolution_l22_l24;
constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

std::vector<std::uint8_t> RRead(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short file read");
  return result;
}

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

id<MTLTexture> MakeOutputTexture(id<MTLDevice> device, std::uint32_t width,
                                 std::uint32_t height,
                                 const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != std::size_t{width} * height * 4u)
    throw std::runtime_error("output texture payload size mismatch");
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                MTLPixelFormatRG16Float
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderWrite;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("output texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0 withBytes:bytes.data() bytesPerRow:width * 4u];
  return texture;
}

std::vector<std::uint8_t> ReadOutputTexture(id<MTLTexture> texture,
                                            std::uint32_t width,
                                            std::uint32_t height) {
  std::vector<std::uint8_t> result(std::size_t{width} * height * 4u);
  [texture getBytes:result.data() bytesPerRow:width * 4u
          fromRegion:MTLRegionMake2D(0, 0, width, height)
         mipmapLevel:0];
  return result;
}

std::size_t ByteCount(const std::vector<std::uint8_t>& bytes,
                      std::uint8_t value) {
  return static_cast<std::size_t>(std::count(bytes.begin(), bytes.end(), value));
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const contract::Resolution& resolution, bool generated,
    std::uint8_t poison, const std::string& device_name,
    const std::vector<std::uint8_t>& module22_input,
    const std::array<std::vector<std::uint8_t>, 4>& module22_outputs,
    const std::vector<std::uint8_t>& module24_mask,
    const std::array<std::vector<std::uint8_t>, 4>& module24_levels,
    const std::vector<std::uint8_t>& module24_output,
    std::size_t input_mismatches, std::size_t canary_mismatches) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write Metal receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.resolution-l22-l24-run.v1\",\n"
          << "  \"backend\": \"mac-metal\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << getpid() << ",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"external_extent\": [" << resolution.external_width << ", "
          << resolution.external_height << "],\n"
          << "  \"working_extent\": [" << resolution.working_width << ", "
          << resolution.working_height << "],\n"
          << "  \"phase\": \"" << (generated ? "generated" : "reset")
          << "\",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"device_name\": \"" << device_name << "\",\n"
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
          << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
          << "  \"input_generator\": \"" << contract::kGenerator << "\",\n"
          << "  \"module22_input_sha256\": \"" << Sha256(module22_input)
          << "\",\n"
          << "  \"module22_output_sha256\": [";
  for (std::size_t index = 0; index < module22_outputs.size(); ++index)
    receipt << (index ? ", \"" : "\"") << Sha256(module22_outputs[index])
            << "\"";
  receipt << "],\n  \"module22_output_bytes\": [";
  for (std::size_t index = 0; index < module22_outputs.size(); ++index)
    receipt << (index ? ", " : "") << module22_outputs[index].size();
  receipt << "],\n  \"module22_output_poison_byte_count\": [";
  for (std::size_t index = 0; index < module22_outputs.size(); ++index)
    receipt << (index ? ", " : "") << ByteCount(module22_outputs[index], poison);
  receipt << "],\n"
          << "  \"module24_mask_sha256\": \"" << Sha256(module24_mask)
          << "\",\n  \"module24_level_sha256\": [";
  for (std::size_t index = 0; index < module24_levels.size(); ++index)
    receipt << (index ? ", \"" : "\"") << Sha256(module24_levels[index])
            << "\"";
  receipt << "],\n"
          << "  \"module24_output_sha256\": \"" << Sha256(module24_output)
          << "\",\n"
          << "  \"module24_output_bytes\": " << module24_output.size() << ",\n"
          << "  \"module24_output_poison_byte_count\": "
          << ByteCount(module24_output, poison) << ",\n"
          << "  \"true_typed_output\": \"MTLPixelFormatRG16Float\",\n"
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

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-resolution-l22-l24-mac <metallib> <rcp-rom> "
                   "<output-dir> <run-id> <poison-hex> <resolution> "
                   "<reset|generated>\n";
      return 2;
    }
    try {
      const auto rom = RRead(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha256)
        throw std::runtime_error("RCP ROM identity mismatch");
      char* end = nullptr;
      const auto parsed = std::strtoul(argv[5], &end, 16);
      if (end == argv[5] || *end != '\0' || parsed > 255u)
        throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed);
      const auto& resolution = contract::ParseResolution(argv[6]);
      const std::string phase_name = argv[7];
      if (phase_name != "reset" && phase_name != "generated")
        throw std::runtime_error("invalid phase");
      const bool generated = phase_name == "generated";
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(output_root);
      const auto module22_input = contract::Module22Input(resolution, generated);
      const auto module22_initial =
          contract::Module22OutputInitial(resolution, poison);
      const auto module24_mask = contract::Mask(resolution, generated);
      const auto module24_levels = contract::Module24Levels(resolution, generated);
      const std::vector<std::uint8_t> module24_initial(
          std::size_t{resolution.working_width} * resolution.working_height * 4u,
          poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                error:&error];
      if (library == nil) throw std::runtime_error("Metal library load failed");
      id<MTLFunction> function22 =
          [library newFunctionWithName:@"fg_resolution_module22"];
      NSString* function24_name = resolution.selected_level == 10
          ? @"fg_resolution_module24_level10"
          : @"fg_resolution_module24_level11";
      id<MTLFunction> function24 = [library newFunctionWithName:function24_name];
      id<MTLComputePipelineState> pipeline22 =
          [device newComputePipelineStateWithFunction:function22 error:&error];
      id<MTLComputePipelineState> pipeline24 =
          [device newComputePipelineStateWithFunction:function24 error:&error];
      if (function22 == nil || function24 == nil || pipeline22 == nil ||
          pipeline24 == nil)
        throw std::runtime_error("Metal resolution pipeline failed");

      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      id<MTLBuffer> m22_input =
          Guarded(device, &module22_input, module22_input.size(), 0);
      std::array<id<MTLBuffer>, 4> m22_outputs;
      for (std::size_t index = 0; index < 4; ++index)
        m22_outputs[index] = Guarded(device, &module22_initial[index],
                                     module22_initial[index].size(), 0);
      id<MTLBuffer> m24_mask =
          Guarded(device, &module24_mask, module24_mask.size(), 0);
      std::array<id<MTLBuffer>, 4> m24_levels;
      for (std::size_t index = 0; index < 4; ++index)
        m24_levels[index] = Guarded(device, &module24_levels[index],
                                    module24_levels[index].size(), 0);
      id<MTLTexture> m24_output = MakeOutputTexture(
          device, resolution.working_width, resolution.working_height,
          module24_initial);

      const std::uint32_t m22_width =
          contract::CeilDiv(resolution.working_width, 16);
      const std::uint32_t m22_height =
          contract::CeilDiv(resolution.working_height, 16);
      std::array<std::uint8_t, 56> packed22{};
      RPack64(&packed22, 0, m22_input.gpuAddress + kGuardBytes);
      for (std::size_t index = 0; index < 4; ++index)
        RPack64(&packed22, 8u + index * 8u,
                m22_outputs[index].gpuAddress + kGuardBytes);
      RPack32(&packed22, 40, resolution.extra_level_count);
      RPack32(&packed22, 44, m22_width);
      RPack32(&packed22, 48, m22_height);
      packed22[52] = packed22[53] = packed22[54] = packed22[55] = poison;
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command22 = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder22 =
          [command22 computeCommandEncoder];
      [encoder22 setComputePipelineState:pipeline22];
      [encoder22 setBytes:packed22.data() length:packed22.size() atIndex:0];
      [encoder22 setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [encoder22 useResource:m22_input usage:MTLResourceUsageRead];
      for (id<MTLBuffer> output : m22_outputs)
        [encoder22 useResource:output usage:MTLResourceUsageWrite];
      [encoder22 dispatchThreadgroups:
          MTLSizeMake(contract::CeilDiv(m22_width, 16),
                      contract::CeilDiv(m22_height, 16), 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder22 endEncoding];
      [command22 commit];
      [command22 waitUntilCompleted];
      if (command22.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal module22 command failed: " +
            std::string(command22.error.localizedDescription.UTF8String ?: "unknown"));

      std::array<std::uint8_t, 64> packed24{};
      RPack64(&packed24, 0, 0);
      RPack64(&packed24, 8, m24_mask.gpuAddress + kGuardBytes);
      for (std::size_t index = 0; index < 4; ++index)
        RPack64(&packed24, 16u + index * 8u,
                m24_levels[index].gpuAddress + kGuardBytes);
      RPack32(&packed24, 48, resolution.selected_level);
      RPack32(&packed24, 52, resolution.working_width);
      RPack32(&packed24, 56, resolution.working_height);
      packed24[60] = packed24[61] = packed24[62] = packed24[63] = poison;
      id<MTLCommandBuffer> command24 = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder24 =
          [command24 computeCommandEncoder];
      [encoder24 setComputePipelineState:pipeline24];
      [encoder24 setBytes:packed24.data() length:packed24.size() atIndex:0];
      [encoder24 setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [encoder24 setTexture:m24_output atIndex:0];
      [encoder24 useResource:m24_mask usage:MTLResourceUsageRead];
      for (id<MTLBuffer> level : m24_levels)
        [encoder24 useResource:level usage:MTLResourceUsageRead];
      [encoder24 dispatchThreadgroups:
          MTLSizeMake(contract::CeilDiv(resolution.working_width, 16),
                      contract::CeilDiv(resolution.working_height, 16), 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder24 endEncoding];
      [command24 commit];
      [command24 waitUntilCompleted];
      if (command24.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal module24 command failed: " +
            std::string(command24.error.localizedDescription.UTF8String ?: "unknown"));

      const auto m22_input_after = Payload(m22_input, module22_input.size());
      std::array<std::vector<std::uint8_t>, 4> m22_after;
      std::size_t canary_mismatches =
          CanaryMismatches(m22_input, module22_input.size()) +
          CanaryMismatches(rom_buffer, kRomBytes);
      for (std::size_t index = 0; index < 4; ++index) {
        m22_after[index] =
            Payload(m22_outputs[index], module22_initial[index].size());
        canary_mismatches += CanaryMismatches(
            m22_outputs[index], module22_initial[index].size());
      }
      const auto m24_mask_after = Payload(m24_mask, module24_mask.size());
      canary_mismatches += CanaryMismatches(m24_mask, module24_mask.size());
      std::array<std::vector<std::uint8_t>, 4> m24_levels_after;
      for (std::size_t index = 0; index < 4; ++index) {
        m24_levels_after[index] =
            Payload(m24_levels[index], module24_levels[index].size());
        canary_mismatches += CanaryMismatches(
            m24_levels[index], module24_levels[index].size());
      }
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      const auto m24_output_after = ReadOutputTexture(
          m24_output, resolution.working_width, resolution.working_height);
      std::size_t input_mismatches =
          Mismatches(module22_input, m22_input_after) +
          Mismatches(module24_mask, m24_mask_after) +
          Mismatches(rom, rom_after);
      for (std::size_t index = 0; index < 4; ++index)
        input_mismatches += Mismatches(module24_levels[index],
                                       m24_levels_after[index]);
      Write(output_root / "module22-input-after.bin", m22_input_after);
      for (std::size_t index = 0; index < 4; ++index) {
        Write(output_root / ("module22-output-" + std::to_string(index) + ".bin"),
              m22_after[index]);
        Write(output_root / ("module24-level-" + std::to_string(index) +
                             "-after.bin"), m24_levels_after[index]);
      }
      Write(output_root / "module24-mask-after.bin", m24_mask_after);
      Write(output_root / "module24-output.bin", m24_output_after);
      WriteReceipt(output_root / "run-receipt.json", run_id, resolution,
                   generated, poison, device.name.UTF8String, module22_input,
                   m22_after, module24_mask, module24_levels,
                   m24_output_after, input_mismatches, canary_mismatches);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_resolution_l22_l24_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "module24_output_sha256=" << Sha256(m24_output_after) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
