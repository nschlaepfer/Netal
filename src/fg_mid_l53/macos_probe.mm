#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include "contract.h"

namespace {
namespace contract = fg_mid_l53;

constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
  float interpolation_factor;
  float motion_threshold;
  float depth_threshold;
  float depth_scale;
  std::uint8_t invert_depth;
  std::uint8_t reserved0;
  std::uint16_t reserved1;
  std::uint32_t low_phase;
};
static_assert(sizeof(Parameters) == 32);

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size " + path.string());
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return result;
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format,
                           std::size_t row_bytes,
                           const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != row_bytes * contract::kHeight) {
    throw std::runtime_error("texture payload size mismatch");
  }
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:contract::kWidth
                                                        height:contract::kHeight
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, contract::kWidth,
                                         contract::kHeight)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::size_t row_bytes) {
  std::vector<std::uint8_t> bytes(row_bytes * contract::kHeight);
  [texture getBytes:bytes.data()
         bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, contract::kWidth,
                                     contract::kHeight)
         mipmapLevel:0];
  return bytes;
}

std::size_t SentinelCount(const std::vector<std::uint8_t>& bytes) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    count += value == contract::kInitialOutput;
  }
  return count;
}

std::size_t PoisonCanaryMismatches(id<MTLBuffer> buffer,
                                   std::size_t payload_bytes,
                                   std::uint8_t poison) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t count = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    count += bytes[index] != poison;
    count += bytes[kGuardBytes + payload_bytes + index] != poison;
  }
  return count;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const std::string& scenario, std::uint8_t poison,
    const std::string& device_name,
    const std::array<std::vector<std::uint8_t>, 4>& inputs_before,
    const std::array<std::vector<std::uint8_t>, 4>& inputs_after,
    const std::vector<std::uint8_t>& output_a,
    const std::vector<std::uint8_t>& output_b,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t sentinel_a, std::size_t sentinel_b) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L53 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.mid-l53-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"scenario\": \"" << scenario << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_launch\": 53,\n"
         << "  \"module_token\": 12,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"raw_live_parameter_sha256\": \""
         << contract::kRawParameterSha256 << "\",\n"
         << "  \"input_generator\": \"" << contract::kInputGenerator
         << "\",\n"
         << "  \"metal_functions\": [\"fg_mid_l53_candidates\", \"fg_mid_l53_pack\"],\n"
         << "  \"atomic_lowering\": \"u32-high-min then equal-high-u32-low-min\",\n"
         << "  \"selected_grid\": [20, 12, 1],\n"
         << "  \"selected_block\": [324, 1, 1],\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"dimensions\": [320, 180],\n"
         << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
         << "  \"candidate_atomic_order\": \"signed-int64-min\",\n"
         << "  \"input_pre_sha256\": [";
  for (std::size_t index = 0; index < inputs_before.size(); ++index) {
    stream << (index == 0 ? "\"" : ", \"")
           << Sha256(inputs_before[index]) << "\"";
  }
  stream << "],\n  \"input_post_sha256\": [";
  for (std::size_t index = 0; index < inputs_after.size(); ++index) {
    stream << (index == 0 ? "\"" : ", \"")
           << Sha256(inputs_after[index]) << "\"";
  }
  stream << "],\n"
         << "  \"output_bytes_each\": " << contract::kOutputBytes << ",\n"
         << "  \"output_sha256\": [\"" << Sha256(output_a) << "\", \""
         << Sha256(output_b) << "\"],\n"
         << "  \"sentinel_u64_count\": [" << sentinel_a << ", "
         << sentinel_b << "],\n"
         << "  \"non_sentinel_u64_count\": ["
         << (contract::kPixelCount - sentinel_a) << ", "
         << (contract::kPixelCount - sentinel_b) << "],\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 7) {
      std::cerr << "usage=fg-mid-l53-mac <metallib> <rcp-rom> <output-dir> "
                   "<run-id> <poison-hex> <small-flow|pixel-flow>\n";
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
      const std::string scenario = argv[6];
      if (scenario != "small-flow" && scenario != "pixel-flow") {
        throw std::runtime_error("invalid L53 scenario");
      }
      const bool pixel_flow = scenario == "pixel-flow";
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(output_root);
      const std::array<std::vector<std::uint8_t>, 4> inputs_before = {
          contract::CurrToPrevFlow(), contract::PrevToCurrFlow(pixel_flow),
          contract::CurrDepth(), contract::PrevDepth()};
      const auto initial_output = contract::InitialOutput();

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> candidate_function =
          [library newFunctionWithName:@"fg_mid_l53_candidates"];
      id<MTLFunction> pack_function =
          [library newFunctionWithName:@"fg_mid_l53_pack"];
      if (candidate_function == nil || pack_function == nil) {
        throw std::runtime_error("Metal function missing");
      }
      id<MTLComputePipelineState> candidate_pipeline =
          [device newComputePipelineStateWithFunction:candidate_function
                                                error:&error];
      id<MTLComputePipelineState> pack_pipeline =
          [device newComputePipelineStateWithFunction:pack_function
                                                error:&error];
      if (candidate_pipeline == nil || pack_pipeline == nil) {
        throw std::runtime_error("pipeline creation failed");
      }
      if (candidate_pipeline.maxTotalThreadsPerThreadgroup < 324u) {
        throw std::runtime_error("L53 threadgroup width unsupported");
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLTexture> curr_to_prev_flow =
          MakeTexture(device, MTLPixelFormatRG32Float,
                      contract::kWidth * 8u, inputs_before[0]);
      id<MTLTexture> prev_to_curr_flow =
          MakeTexture(device, MTLPixelFormatRG16Float,
                      contract::kWidth * 4u, inputs_before[1]);
      id<MTLTexture> curr_depth =
          MakeTexture(device, MTLPixelFormatR32Uint,
                      contract::kWidth * 4u, inputs_before[2]);
      id<MTLTexture> prev_depth =
          MakeTexture(device, MTLPixelFormatR32Uint,
                      contract::kWidth * 4u, inputs_before[3]);
      id<MTLBuffer> output_a = Guarded(
          device, &initial_output, contract::kOutputBytes, 0);
      id<MTLBuffer> output_b = Guarded(
          device, &initial_output, contract::kOutputBytes, 0);
      std::memset(output_a.contents, poison,
                  contract::kOutputBytes + 2u * kGuardBytes);
      std::memset(output_b.contents, poison,
                  contract::kOutputBytes + 2u * kGuardBytes);
      std::memcpy(static_cast<std::uint8_t*>(output_a.contents) + kGuardBytes,
                  initial_output.data(), initial_output.size());
      std::memcpy(static_cast<std::uint8_t*>(output_b.contents) + kGuardBytes,
                  initial_output.data(), initial_output.size());
      std::vector<std::uint8_t> high_initial(
          contract::kPixelCount * sizeof(std::uint32_t));
      for (std::size_t pixel = 0; pixel < contract::kPixelCount; ++pixel) {
        contract::Store32(&high_initial, pixel * sizeof(std::uint32_t),
                          0x7fffffffu);
      }
      const std::vector<std::uint8_t> low_initial(
          contract::kPixelCount * sizeof(std::uint32_t), 0xff);
      id<MTLBuffer> high_a = Guarded(
          device, &high_initial, high_initial.size(), 0);
      id<MTLBuffer> low_a = Guarded(
          device, &low_initial, low_initial.size(), 0);
      id<MTLBuffer> high_b = Guarded(
          device, &high_initial, high_initial.size(), 0);
      id<MTLBuffer> low_b = Guarded(
          device, &low_initial, low_initial.size(), 0);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      Parameters parameters{contract::kWidth,
                            contract::kHeight,
                            0.5f,
                            12.96000003814697265625f,
                            40.0f,
                            1.0f,
                            0,
                            0,
                            0,
                            0};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      for (std::uint32_t phase = 0; phase < 2; ++phase) {
        parameters.low_phase = phase;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:candidate_pipeline];
        [encoder setTexture:curr_to_prev_flow atIndex:0];
        [encoder setTexture:prev_to_curr_flow atIndex:1];
        [encoder setTexture:curr_depth atIndex:2];
        [encoder setTexture:prev_depth atIndex:3];
        [encoder setBuffer:high_a offset:kGuardBytes atIndex:0];
        [encoder setBuffer:low_a offset:kGuardBytes atIndex:1];
        [encoder setBuffer:high_b offset:kGuardBytes atIndex:2];
        [encoder setBuffer:low_b offset:kGuardBytes atIndex:3];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:4];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
                threadsPerThreadgroup:MTLSizeMake(324, 1, 1)];
        [encoder endEncoding];
      }
      id<MTLComputeCommandEncoder> pack_encoder = [command computeCommandEncoder];
      [pack_encoder setComputePipelineState:pack_pipeline];
      [pack_encoder setBuffer:high_a offset:kGuardBytes atIndex:0];
      [pack_encoder setBuffer:low_a offset:kGuardBytes atIndex:1];
      [pack_encoder setBuffer:high_b offset:kGuardBytes atIndex:2];
      [pack_encoder setBuffer:low_b offset:kGuardBytes atIndex:3];
      [pack_encoder setBuffer:output_a offset:kGuardBytes atIndex:4];
      [pack_encoder setBuffer:output_b offset:kGuardBytes atIndex:5];
      [pack_encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
      [pack_encoder dispatchThreads:MTLSizeMake(contract::kPixelCount, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [pack_encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal L53 command failed");
      }

      const std::array<std::vector<std::uint8_t>, 4> inputs_after = {
          ReadTexture(curr_to_prev_flow, contract::kWidth * 8u),
          ReadTexture(prev_to_curr_flow, contract::kWidth * 4u),
          ReadTexture(curr_depth, contract::kWidth * 4u),
          ReadTexture(prev_depth, contract::kWidth * 4u)};
      const auto actual_a = Payload(output_a, contract::kOutputBytes);
      const auto actual_b = Payload(output_b, contract::kOutputBytes);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      std::size_t input_mismatches = Mismatches(rom, rom_after);
      for (std::size_t index = 0; index < inputs_before.size(); ++index) {
        input_mismatches += Mismatches(inputs_before[index], inputs_after[index]);
      }
      const auto canary_mismatches =
          PoisonCanaryMismatches(output_a, contract::kOutputBytes, poison) +
          PoisonCanaryMismatches(output_b, contract::kOutputBytes, poison) +
          CanaryMismatches(high_a, high_initial.size()) +
          CanaryMismatches(low_a, low_initial.size()) +
          CanaryMismatches(high_b, high_initial.size()) +
          CanaryMismatches(low_b, low_initial.size()) +
          CanaryMismatches(rom_buffer, kRomBytes);
      const auto sentinel_a = SentinelCount(actual_a);
      const auto sentinel_b = SentinelCount(actual_b);
      for (std::size_t index = 0; index < inputs_after.size(); ++index) {
        Write(output_root / ("input-" + std::to_string(index) + "-after.bin"),
              inputs_after[index]);
      }
      Write(output_root / "output-a.bin", actual_a);
      Write(output_root / "output-b.bin", actual_b);
      WriteReceipt(output_root / "run-receipt.json", run_id, scenario,
                   poison, device.name.UTF8String, inputs_before, inputs_after,
                   actual_a, actual_b, input_mismatches, canary_mismatches,
                   sentinel_a, sentinel_b);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_mid_l53_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_a_sha256=" << Sha256(actual_a) << '\n'
                << "output_b_sha256=" << Sha256(actual_b) << '\n'
                << "sentinel_a=" << sentinel_a << '\n'
                << "sentinel_b=" << sentinel_b << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
