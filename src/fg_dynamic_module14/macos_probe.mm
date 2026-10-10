#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include "contract.h"

namespace {
namespace contract = fg_dynamic_module14;

constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

struct KernelParameters {
  std::uint32_t width;
  std::uint32_t height;
};

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read");
  return result;
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format,
                           MTLTextureUsage usage, std::uint32_t width,
                           std::uint32_t height, std::size_t row_bytes,
                           const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != row_bytes * height) {
    throw std::runtime_error("texture payload size mismatch");
  }
  auto* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::uint32_t width,
                                      std::uint32_t height,
                                      std::size_t row_bytes) {
  std::vector<std::uint8_t> result(row_bytes * height);
  [texture getBytes:result.data()
         bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, width, height)
         mipmapLevel:0];
  return result;
}

std::vector<std::uint8_t> ActiveOutput(
    const contract::Resolution& resolution,
    const std::vector<std::uint8_t>& guarded) {
  const std::size_t bytes =
      std::size_t{resolution.width} * resolution.height * 4u;
  return {guarded.begin(), guarded.begin() + bytes};
}

std::size_t GuardMismatches(const contract::Resolution& resolution,
                            const std::vector<std::uint8_t>& guarded,
                            std::uint8_t poison) {
  const std::size_t active =
      std::size_t{resolution.width} * resolution.height * 4u;
  std::size_t mismatches = 0;
  for (std::size_t index = active; index < guarded.size(); ++index) {
    mismatches += guarded[index] != poison;
  }
  return mismatches;
}

std::uint32_t Load32(const std::vector<std::uint8_t>& bytes,
                     std::size_t offset) {
  std::uint32_t result = 0;
  std::memcpy(&result, bytes.data() + offset, sizeof(result));
  return result;
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-dynamic-module14-mac <metallib> <rcp-rom> "
                   "<output-dir> <run-id> <poison-hex> <resolution> "
                   "<full|hash-only>\n";
      return 2;
    }
    try {
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha256) {
        throw std::runtime_error("RCP ROM identity mismatch");
      }
      char* poison_end = nullptr;
      const auto parsed_poison = std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' ||
          parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const auto& resolution = contract::FindResolution(argv[6]);
      const bool full_artifacts = std::string_view(argv[7]) == "full";
      if (!full_artifacts && std::string_view(argv[7]) != "hash-only") {
        throw std::runtime_error("invalid artifact mode");
      }
      const std::filesystem::path output_root = argv[3];
      std::filesystem::create_directories(output_root);
      const auto input_before = contract::GuardedInput(resolution, poison);
      const auto input_active = contract::InputActive(resolution);
      const auto dead_before = contract::DeadMask(resolution, poison);
      const auto output_before = contract::PoisonedOutput(resolution, poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_dynamic_module14"];
      id<MTLFunction> probe_function =
          [library newFunctionWithName:@"fg_dynamic_module14_numeric_probe"];
      if (function == nil || probe_function == nil) {
        throw std::runtime_error("Metal function missing");
      }
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      id<MTLComputePipelineState> probe_pipeline =
          [device newComputePipelineStateWithFunction:probe_function error:&error];
      if (pipeline == nil || probe_pipeline == nil) {
        throw std::runtime_error("pipeline creation failed");
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      const auto resource_height = resolution.height + 1u;
      id<MTLTexture> input = MakeTexture(
          device, MTLPixelFormatRGBA16Float, MTLTextureUsageShaderRead,
          resolution.width, resource_height,
          std::size_t{resolution.width} * 8u, input_before);
      id<MTLTexture> dead = MakeTexture(
          device, MTLPixelFormatR8Unorm, MTLTextureUsageShaderRead,
          resolution.width, resource_height, resolution.width, dead_before);
      id<MTLTexture> output = MakeTexture(
          device, MTLPixelFormatRG16Float, MTLTextureUsageShaderWrite,
          resolution.width, resource_height,
          std::size_t{resolution.width} * 4u, output_before);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);

      const std::array<std::uint32_t, 3> probe_inputs{{
          0x3a800200u, 0x3f801001u, 0x40400000u}};
      std::vector<std::uint8_t> probe_input_bytes(sizeof(probe_inputs));
      std::memcpy(probe_input_bytes.data(), probe_inputs.data(),
                  sizeof(probe_inputs));
      std::vector<std::uint8_t> probe_output_before(16u, poison);
      id<MTLBuffer> probe_input = Guarded(
          device, &probe_input_bytes, probe_input_bytes.size(), poison);
      id<MTLBuffer> probe_output = Guarded(
          device, &probe_output_before, probe_output_before.size(), poison);
      const KernelParameters parameters{resolution.width, resolution.height};

      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:input atIndex:0];
      [encoder setTexture:dead atIndex:1];
      [encoder setTexture:output atIndex:2];
      [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(resolution.width,
                                           resolution.height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      id<MTLComputeCommandEncoder> probe_encoder =
          [command computeCommandEncoder];
      [probe_encoder setComputePipelineState:probe_pipeline];
      [probe_encoder setBuffer:probe_input offset:kGuardBytes atIndex:0];
      [probe_encoder setBuffer:probe_output offset:kGuardBytes atIndex:1];
      [probe_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
      [probe_encoder dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [probe_encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal module14 command failed");
      }

      const auto input_after = ReadTexture(
          input, resolution.width, resource_height,
          std::size_t{resolution.width} * 8u);
      const auto dead_after = ReadTexture(
          dead, resolution.width, resource_height, resolution.width);
      const auto output_guarded = ReadTexture(
          output, resolution.width, resource_height,
          std::size_t{resolution.width} * 4u);
      const auto output_after = ActiveOutput(resolution, output_guarded);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      const auto probe_input_after = Payload(
          probe_input, probe_input_bytes.size());
      const auto probe_output_after = Payload(
          probe_output, probe_output_before.size());
      const auto input_mismatches = Mismatches(input_before, input_after) +
                                    Mismatches(dead_before, dead_after) +
                                    Mismatches(rom, rom_after) +
                                    Mismatches(probe_input_bytes,
                                               probe_input_after);
      const auto output_guard_mismatches =
          GuardMismatches(resolution, output_guarded, poison);
      const auto buffer_canary_mismatches =
          CanaryMismatches(rom_buffer, kRomBytes) +
          CanaryMismatches(probe_input, probe_input_bytes.size()) +
          CanaryMismatches(probe_output, probe_output_before.size());
      const auto approx_bits = Load32(probe_output_after, 0);
      const auto precise_bits = Load32(probe_output_after, 4);
      const auto rz_half_bits = Load32(probe_output_after, 8);
      const auto rne_half_bits = Load32(probe_output_after, 12);
      const bool numeric_distinctions = approx_bits != precise_bits &&
                                        rz_half_bits != rne_half_bits;
      if (full_artifacts) {
        Write(output_root / "input-after.bin", input_after);
        Write(output_root / "dead-mask-after.bin", dead_after);
        Write(output_root / "output.bin", output_after);
        Write(output_root / "numeric-probe-output.bin", probe_output_after);
      }
      const bool pass = input_mismatches == 0 &&
                        output_guard_mismatches == 0 &&
                        buffer_canary_mismatches == 0 && numeric_distinctions;
      std::ofstream receipt(output_root / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      if (!receipt) throw std::runtime_error("unable to write run receipt");
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.dynamic-module14-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << argv[4] << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"resolution\": \"" << resolution.name << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"device_name\": \"" << device.name.UTF8String << "\",\n"
              << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
              << "  \"metal_function\": \"fg_dynamic_module14\",\n"
              << "  \"extent\": [" << resolution.width << ", "
              << resolution.height << "],\n"
              << "  \"grid\": [" << resolution.grid_x << ", "
              << resolution.grid_y << ", 1],\n"
              << "  \"block\": [16, 16, 1],\n"
              << "  \"packed_parameter_bytes\": 32,\n"
              << "  \"raw_parameter_sha256\": \""
              << resolution.raw_parameter_sha256 << "\",\n"
              << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
              << "  \"surface_half_rounding\": \"toward-zero\",\n"
              << "  \"input_active_sha256\": \"" << Sha256(input_active)
              << "\",\n"
              << "  \"input_resource_sha256\": \"" << Sha256(input_before)
              << "\",\n"
              << "  \"input_post_sha256\": \"" << Sha256(input_after)
              << "\",\n"
              << "  \"dead_resource_sha256\": \"" << Sha256(dead_before)
              << "\",\n"
              << "  \"dead_post_sha256\": \"" << Sha256(dead_after)
              << "\",\n"
              << "  \"output_bytes\": " << output_after.size() << ",\n"
              << "  \"output_sha256\": \"" << Sha256(output_after) << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"output_guard_mismatched_bytes\": "
              << output_guard_mismatches << ",\n"
              << "  \"buffer_canary_mismatched_bytes\": "
              << buffer_canary_mismatches << ",\n"
              << "  \"approx_division_probe_bits\": " << approx_bits << ",\n"
              << "  \"precise_division_probe_bits\": " << precise_bits << ",\n"
              << "  \"rz_store_probe_half_bits\": " << rz_half_bits << ",\n"
              << "  \"rne_store_probe_half_bits\": " << rne_half_bits << ",\n"
              << "  \"numeric_distinctions_observed\": "
              << (numeric_distinctions ? "true" : "false") << ",\n"
              << "  \"artifacts_written\": "
              << (full_artifacts ? "true" : "false") << ",\n"
              << "  \"expected_or_captured_output_ingress\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
              << "}\n";
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
