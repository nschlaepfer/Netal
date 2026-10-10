#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include "contract.h"

namespace {
namespace contract = fg_mid_l54;

constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
};

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

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLTextureUsage usage,
                           const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != contract::kTextureBytes) {
    throw std::runtime_error("texture payload size mismatch");
  }
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float
                                                         width:contract::kWidth
                                                        height:contract::kHeight
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, contract::kWidth,
                                         contract::kHeight)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:contract::kWidth * 4u];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture) {
  std::vector<std::uint8_t> bytes(contract::kTextureBytes);
  [texture getBytes:bytes.data()
         bytesPerRow:contract::kWidth * 4u
          fromRegion:MTLRegionMake2D(0, 0, contract::kWidth,
                                     contract::kHeight)
         mipmapLevel:0];
  return bytes;
}

std::size_t PoisonHalfwords(const std::vector<std::uint8_t>& bytes,
                            std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 2u) {
    count += bytes[offset] == poison && bytes[offset + 1u] == poison;
  }
  return count;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::string& device_name,
    const std::vector<std::uint8_t>& texture_a_before,
    const std::vector<std::uint8_t>& texture_b_before,
    const std::vector<std::uint8_t>& offset_a_before,
    const std::vector<std::uint8_t>& offset_b_before,
    const std::vector<std::uint8_t>& texture_a_after,
    const std::vector<std::uint8_t>& texture_b_after,
    const std::vector<std::uint8_t>& offset_a_after,
    const std::vector<std::uint8_t>& offset_b_after,
    const std::vector<std::uint8_t>& output, std::size_t input_mismatches,
    std::size_t canary_mismatches, std::size_t poison_halfwords) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L54 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.mid-l54-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"metal_function\": \"fg_mid_l54\",\n"
         << "  \"selected_launch\": 54,\n"
         << "  \"module_token\": 26,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"selected_grid\": [20, 12, 1],\n"
         << "  \"selected_block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 88,\n"
         << "  \"dimensions\": [320, 180, 320, 180],\n"
         << "  \"resource_contract\": {\"inputs\": [\"RG16F point-border normalized texture\", \"RG16F point-border normalized texture\", \"u64 offset buffer\", \"u64 offset buffer\"], \"output\": \"RG16F surface\"},\n"
         << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
         << "  \"normalized_point_coordinate_fraction_bits\": 12,\n"
         << "  \"surface_half_rounding\": \"toward-zero\",\n"
         << "  \"texture_pre_sha256\": [\"" << Sha256(texture_a_before)
         << "\", \"" << Sha256(texture_b_before) << "\"],\n"
         << "  \"texture_post_sha256\": [\"" << Sha256(texture_a_after)
         << "\", \"" << Sha256(texture_b_after) << "\"],\n"
         << "  \"offset_pre_sha256\": [\"" << Sha256(offset_a_before)
         << "\", \"" << Sha256(offset_b_before) << "\"],\n"
         << "  \"offset_post_sha256\": [\"" << Sha256(offset_a_after)
         << "\", \"" << Sha256(offset_b_after) << "\"],\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_collision_halfwords\": " << poison_halfwords << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"unused_u64_upper_halves_poisoned\": true,\n"
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
      std::cerr << "usage=fg-mid-l54-mac <metallib> <rcp-rom> <output-dir> "
                   "<run-id> <poison-hex> <threadgroup-width>\n";
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
      char* width_end = nullptr;
      const auto threadgroup_width = std::strtoul(argv[6], &width_end, 10);
      if (width_end == argv[6] || *width_end != '\0' ||
          threadgroup_width == 0 || threadgroup_width > 1024u) {
        throw std::runtime_error("invalid threadgroup width");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(output_root);
      const auto texture_a_before = contract::TextureA();
      const auto texture_b_before = contract::TextureB();
      const auto offset_a_before = contract::OffsetA(poison);
      const auto offset_b_before = contract::OffsetB(poison);
      const std::vector<std::uint8_t> output_poison(contract::kOutputBytes,
                                                     poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function = [library newFunctionWithName:@"fg_mid_l54"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLTexture> texture_a =
          MakeTexture(device, MTLTextureUsageShaderRead, texture_a_before);
      id<MTLTexture> texture_b =
          MakeTexture(device, MTLTextureUsageShaderRead, texture_b_before);
      id<MTLBuffer> offset_a =
          Guarded(device, &offset_a_before, contract::kOffsetBufferBytes, 0);
      id<MTLBuffer> offset_b =
          Guarded(device, &offset_b_before, contract::kOffsetBufferBytes, 0);
      id<MTLTexture> output =
          MakeTexture(device, MTLTextureUsageShaderWrite, output_poison);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      const Parameters parameters{contract::kWidth, contract::kHeight};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:texture_a atIndex:0];
      [encoder setTexture:texture_b atIndex:1];
      [encoder setTexture:output atIndex:2];
      [encoder setBuffer:offset_a offset:kGuardBytes atIndex:0];
      [encoder setBuffer:offset_b offset:kGuardBytes atIndex:1];
      [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(contract::kWidth, contract::kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      const auto texture_a_after = ReadTexture(texture_a);
      const auto texture_b_after = ReadTexture(texture_b);
      const auto offset_a_after = Payload(offset_a, contract::kOffsetBufferBytes);
      const auto offset_b_after = Payload(offset_b, contract::kOffsetBufferBytes);
      const auto output_after = ReadTexture(output);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      const auto input_mismatches =
          Mismatches(texture_a_before, texture_a_after) +
          Mismatches(texture_b_before, texture_b_after) +
          Mismatches(offset_a_before, offset_a_after) +
          Mismatches(offset_b_before, offset_b_after) + Mismatches(rom, rom_after);
      const auto canary_mismatches =
          CanaryMismatches(offset_a, contract::kOffsetBufferBytes) +
          CanaryMismatches(offset_b, contract::kOffsetBufferBytes) +
          CanaryMismatches(rom_buffer, kRomBytes);
      Write(output_root / "texture-a-after.bin", texture_a_after);
      Write(output_root / "texture-b-after.bin", texture_b_after);
      Write(output_root / "offset-a-after.bin", offset_a_after);
      Write(output_root / "offset-b-after.bin", offset_b_after);
      Write(output_root / "output.bin", output_after);
      WriteReceipt(output_root / "run-receipt.json", run_id, poison,
                   device.name.UTF8String, texture_a_before, texture_b_before,
                   offset_a_before, offset_b_before, texture_a_after,
                   texture_b_after, offset_a_after, offset_b_after,
                   output_after, input_mismatches, canary_mismatches,
                   PoisonHalfwords(output_after, poison));
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_mid_l54_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_sha256=" << Sha256(output_after) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
