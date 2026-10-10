#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include <algorithm>

#include "contract.h"

namespace {
namespace contract = fg_early_l6_l16;
constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path) {
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

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format,
                           MTLTextureUsage usage, std::size_t row_bytes,
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
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, contract::kWidth,
                                         contract::kHeight)
           mipmapLevel:0 withBytes:bytes.data() bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::size_t row_bytes) {
  std::vector<std::uint8_t> result(row_bytes * contract::kHeight);
  [texture getBytes:result.data() bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, contract::kWidth,
                                     contract::kHeight)
         mipmapLevel:0];
  return result;
}

template <std::size_t Size>
void Pack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
            std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void Pack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
            std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::size_t ByteCount(const std::vector<std::uint8_t>& bytes,
                      std::uint8_t value) {
  return static_cast<std::size_t>(std::count(bytes.begin(), bytes.end(), value));
}

void WriteRunReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    contract::Phase phase, std::uint8_t poison, const std::string& device_name,
    const std::vector<std::uint8_t>& flow_before,
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
          << "  \"backend\": \"mac-metal\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << getpid() << ",\n"
          << "  \"phase\": \"" << contract::PhaseName(phase) << "\",\n"
          << "  \"selected_launch\": "
          << (phase == contract::Phase::kReset ? 6 : 16) << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"device_name\": \"" << device_name << "\",\n"
          << "  \"module_token\": 21,\n"
          << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
          << "  \"metal_function\": \"fg_early_l6_l16\",\n"
          << "  \"grid\": [20, 12, 1],\n"
          << "  \"block\": [256, 1, 1],\n"
          << "  \"packed_parameter_bytes\": 224,\n"
          << "  \"loaded_byte_156\": 0,\n"
          << "  \"padding_157_159\": [" << unsigned(poison) << ", "
          << unsigned(poison) << ", " << unsigned(poison) << "],\n"
          << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
          << "  \"vector_surface_rounding\": \"toward-zero\",\n"
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

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 7) {
      std::cerr << "usage=fg-early-l6-l16-mac <metallib> <rcp-rom> "
                   "<output-dir> <run-id> <poison-hex> <reset|generated>\n";
      return 2;
    }
    try {
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha256) {
        throw std::runtime_error("RCP ROM identity mismatch");
      }
      char* end = nullptr;
      const auto parsed = std::strtoul(argv[5], &end, 16);
      if (end == argv[5] || *end != '\0' || parsed > 255u) {
        throw std::runtime_error("invalid poison");
      }
      const auto poison = static_cast<std::uint8_t>(parsed);
      const auto phase = contract::ParsePhase(argv[6]);
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(output_root);
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

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                error:&error];
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_early_l6_l16"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (library == nil || function == nil || pipeline == nil) {
        throw std::runtime_error("Metal module21 pipeline failed");
      }
      id<MTLTexture> flow = MakeTexture(
          device, MTLPixelFormatRG32Float, MTLTextureUsageShaderRead,
          contract::kWidth * 8u, flow_before);
      id<MTLTexture> depth = MakeTexture(
          device, MTLPixelFormatR32Uint, MTLTextureUsageShaderRead,
          contract::kWidth * 4u, depth_before);
      id<MTLTexture> input_mask = MakeTexture(
          device, MTLPixelFormatR8Unorm, MTLTextureUsageShaderRead,
          contract::kWidth, input_mask_before);
      id<MTLTexture> vector_surface = MakeTexture(
          device, MTLPixelFormatRG16Float, MTLTextureUsageShaderWrite,
          contract::kWidth * 4u, vector_before);
      id<MTLTexture> mask_surface = MakeTexture(
          device, MTLPixelFormatR8Unorm, MTLTextureUsageShaderWrite,
          contract::kWidth, mask_surface_before);
      id<MTLBuffer> candidates = Guarded(
          device, &candidates_before, contract::kCandidateBytes, 0);
      std::array<id<MTLBuffer>, 5> output_buffers;
      for (std::size_t index = 0; index < output_buffers.size(); ++index) {
        output_buffers[index] = Guarded(
            device, &output_before[index], contract::kOutputBufferBytes[index], 0);
      }
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      std::array<std::uint8_t, 224> packed{};
      Pack64(&packed, 24, candidates.gpuAddress + kGuardBytes);
      for (std::size_t index = 0; index < output_buffers.size(); ++index) {
        Pack64(&packed, 48u + index * 8u,
               output_buffers[index].gpuAddress + kGuardBytes);
      }
      Pack32(&packed, 128, std::bit_cast<std::uint32_t>(1.0f / 640.0f));
      Pack32(&packed, 132, std::bit_cast<std::uint32_t>(1.0f / 360.0f));
      Pack32(&packed, 136, contract::kWidth);
      Pack32(&packed, 140, contract::kHeight);
      Pack32(&packed, 144, std::bit_cast<std::uint32_t>(40.0f));
      Pack32(&packed, 148, std::bit_cast<std::uint32_t>(1.0f));
      Pack32(&packed, 152, std::bit_cast<std::uint32_t>(600.0f));
      packed[156] = 0;
      packed[157] = poison;
      packed[158] = poison;
      packed[159] = poison;
      const std::array<float, 16> identity = {
          1, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 1};
      std::memcpy(packed.data() + 160, identity.data(), 64);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBytes:packed.data() length:packed.size() atIndex:0];
      [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [encoder setTexture:flow atIndex:0];
      [encoder setTexture:depth atIndex:1];
      [encoder setTexture:input_mask atIndex:2];
      [encoder setTexture:vector_surface atIndex:3];
      [encoder setTexture:mask_surface atIndex:4];
      [encoder useResource:candidates usage:MTLResourceUsageRead];
      for (id<MTLBuffer> buffer : output_buffers) {
        [encoder useResource:buffer usage:MTLResourceUsageWrite];
      }
      [encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal module21 command failed");
      }

      const auto flow_after = ReadTexture(flow, contract::kWidth * 8u);
      const auto depth_after = ReadTexture(depth, contract::kWidth * 4u);
      const auto input_mask_after = ReadTexture(input_mask, contract::kWidth);
      const auto vector_after =
          ReadTexture(vector_surface, contract::kWidth * 4u);
      const auto mask_surface_after =
          ReadTexture(mask_surface, contract::kWidth);
      const auto candidates_after = Payload(candidates, contract::kCandidateBytes);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      std::array<std::vector<std::uint8_t>, 5> output_after;
      std::size_t canary_mismatches =
          CanaryMismatches(candidates, contract::kCandidateBytes) +
          CanaryMismatches(rom_buffer, kRomBytes);
      for (std::size_t index = 0; index < output_buffers.size(); ++index) {
        output_after[index] =
            Payload(output_buffers[index], contract::kOutputBufferBytes[index]);
        canary_mismatches += CanaryMismatches(
            output_buffers[index], contract::kOutputBufferBytes[index]);
      }
      const std::size_t input_mismatches =
          Mismatches(flow_before, flow_after) +
          Mismatches(depth_before, depth_after) +
          Mismatches(input_mask_before, input_mask_after) +
          Mismatches(candidates_before, candidates_after) +
          Mismatches(rom, rom_after);
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
                      device.name.UTF8String, flow_before, depth_before,
                      input_mask_before, candidates_before, flow_after,
                      depth_after, input_mask_after, candidates_after,
                      vector_after, mask_surface_after, output_after,
                      input_mismatches, canary_mismatches);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_early_l6_l16_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "vector_surface_sha256=" << Sha256(vector_after) << '\n'
                << "mask_surface_sha256=" << Sha256(mask_surface_after) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
