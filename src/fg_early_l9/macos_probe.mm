#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include "contract.h"

namespace {
namespace contract = fg_early_l9;

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

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format,
                           MTLTextureUsage usage, std::uint32_t height,
                           std::size_t row_bytes,
                           const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != row_bytes * height) {
    throw std::runtime_error("texture payload size mismatch");
  }
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:contract::kWidth
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, contract::kWidth, height)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::uint32_t height,
                                      std::size_t row_bytes) {
  std::vector<std::uint8_t> result(row_bytes * height);
  [texture getBytes:result.data()
         bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, contract::kWidth, height)
         mipmapLevel:0];
  return result;
}

std::vector<std::uint8_t> ActiveRows(
    const std::vector<std::uint8_t>& guarded_output) {
  return std::vector<std::uint8_t>(
      guarded_output.begin(), guarded_output.begin() + contract::kOutputBytes);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& output,
                             std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t index = contract::kOutputBytes; index < output.size(); ++index) {
    count += output[index] != poison;
  }
  return count;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, std::uint8_t poison,
                  const std::string& device_name,
                  const std::vector<std::uint8_t>& input_before,
                  const std::vector<std::uint8_t>& dead_before,
                  const std::vector<std::uint8_t>& input_after,
                  const std::vector<std::uint8_t>& dead_after,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t poison_halfwords) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L9 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l9-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_launch\": 9,\n"
         << "  \"module_token\": 14,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"metal_function\": \"fg_early_l9\",\n"
         << "  \"selected_grid\": [20, 12, 1],\n"
         << "  \"selected_block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 32,\n"
         << "  \"dead_parameter_offset\": 8,\n"
         << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
         << "  \"surface_half_rounding\": \"toward-zero\",\n"
         << "  \"input_sha256\": [\"" << Sha256(input_before) << "\", \""
         << Sha256(dead_before) << "\"],\n"
         << "  \"input_post_sha256\": [\"" << Sha256(input_after) << "\", \""
         << Sha256(dead_after) << "\"],\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_collision_halfwords\": " << poison_halfwords << ",\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-early-l9-mac <metallib> <rcp-rom> <output-dir> "
                   "<run-id> <poison-hex>\n";
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
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(output_root);
      const auto input_before = contract::Input();
      const auto dead_before = contract::DeadMask(poison);
      std::vector<std::uint8_t> output_before(
          contract::kOutputBytes + contract::kWidth * 4u, poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function = [library newFunctionWithName:@"fg_early_l9"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLTexture> input = MakeTexture(
          device, MTLPixelFormatRGBA16Float, MTLTextureUsageShaderRead,
          contract::kHeight, contract::kWidth * 8u, input_before);
      id<MTLTexture> dead = MakeTexture(
          device, MTLPixelFormatR8Unorm, MTLTextureUsageShaderRead,
          contract::kHeight, contract::kWidth, dead_before);
      id<MTLTexture> output = MakeTexture(
          device, MTLPixelFormatRG16Float, MTLTextureUsageShaderWrite,
          contract::kHeight + 1u, contract::kWidth * 4u, output_before);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      const Parameters parameters{contract::kWidth, contract::kHeight};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:input atIndex:0];
      [encoder setTexture:output atIndex:1];
      [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(contract::kWidth,
                                           contract::kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal L9 command failed");
      }
      const auto input_after = ReadTexture(
          input, contract::kHeight, contract::kWidth * 8u);
      const auto dead_after = ReadTexture(
          dead, contract::kHeight, contract::kWidth);
      const auto output_guarded = ReadTexture(
          output, contract::kHeight + 1u, contract::kWidth * 4u);
      const auto output_after = ActiveRows(output_guarded);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      const auto input_mismatches = Mismatches(input_before, input_after) +
                                    Mismatches(dead_before, dead_after) +
                                    Mismatches(rom, rom_after);
      const auto canary_mismatches = CanaryMismatches(output_guarded, poison) +
                                     CanaryMismatches(rom_buffer, kRomBytes);
      const auto poison_halfwords = UntouchedHalfwords(output_after, poison);
      Write(output_root / "input-after.bin", input_after);
      Write(output_root / "dead-mask-after.bin", dead_after);
      Write(output_root / "output.bin", output_after);
      WriteReceipt(output_root / "run-receipt.json", run_id, poison,
                   device.name.UTF8String, input_before, dead_before,
                   input_after, dead_after, output_after, input_mismatches,
                   canary_mismatches, poison_halfwords);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_early_l9_mac\n"
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
