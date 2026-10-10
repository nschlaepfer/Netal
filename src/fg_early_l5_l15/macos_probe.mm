#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include <set>

#include "contract.h"

namespace {
namespace contract = fg_early_l5_l15;

constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
  float motion_divisor;
  float neighbor_divisor;
  float depth_scale;
  std::uint32_t invert_depth;
};
static_assert(sizeof(Parameters) == 24);

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
           mipmapLevel:0 withBytes:bytes.data() bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::uint32_t height,
                                      std::size_t row_bytes) {
  std::vector<std::uint8_t> result(row_bytes * height);
  [texture getBytes:result.data() bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, contract::kWidth, height)
         mipmapLevel:0];
  return result;
}

struct AtomicClassification {
  std::size_t sentinel = 0;
  std::size_t candidate = 0;
  std::size_t negative_high = 0;
  std::set<std::uint32_t> high_words;
  std::set<std::uint32_t> low_words;
};

AtomicClassification ClassifyAtomic(const std::vector<std::uint8_t>& bytes) {
  AtomicClassification result;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    if (value == contract::kAtomicSentinel) {
      ++result.sentinel;
      continue;
    }
    ++result.candidate;
    const auto high = static_cast<std::uint32_t>(value >> 32u);
    const auto low = static_cast<std::uint32_t>(value);
    result.negative_high += (high & 0x80000000u) != 0u;
    result.high_words.insert(high);
    result.low_words.insert(low);
  }
  return result;
}

struct MaskClassification {
  std::size_t zero = 0;
  std::size_t one = 0;
  std::size_t poison = 0;
  std::size_t other = 0;
};

MaskClassification ClassifyMask(const std::vector<std::uint8_t>& bytes,
                                std::uint8_t poison) {
  MaskClassification result;
  for (const auto value : bytes) {
    if (value == 0u) ++result.zero;
    else if (value == 255u) ++result.one;
    else if (value == poison) ++result.poison;
    else ++result.other;
  }
  return result;
}

void EncodePass(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                id<MTLTexture> flow, id<MTLTexture> depth, id<MTLTexture> mask,
                id<MTLBuffer> high, id<MTLBuffer> low, id<MTLBuffer> rom,
                const Parameters& parameters) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:flow atIndex:0];
  [encoder setTexture:depth atIndex:1];
  [encoder setTexture:mask atIndex:2];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:rom offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(contract::kWidth, contract::kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    throw std::runtime_error("Metal candidate pass failed");
  }
}

void EncodePack(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> high, id<MTLBuffer> low, id<MTLBuffer> output,
                const Parameters& parameters) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(contract::kPixels, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    throw std::runtime_error("Metal pack pass failed");
  }
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, contract::Phase phase,
                  contract::Scenario scenario, std::uint8_t poison,
                  const std::string& device_name,
                  const std::vector<std::uint8_t>& flow_before,
                  const std::vector<std::uint8_t>& depth_before,
                  const std::vector<std::uint8_t>& flow_after,
                  const std::vector<std::uint8_t>& depth_after,
                  const std::vector<std::uint8_t>& atomic_output,
                  const std::vector<std::uint8_t>& mask_output,
                  std::size_t input_mismatches,
                  std::size_t atomic_canary_mismatches,
                  std::size_t mask_guard_mismatches,
                  const AtomicClassification& atomic_class,
                  const MaskClassification& mask_class) {
  const bool tie = scenario == contract::Scenario::kEqualHighTie;
  const bool classification_pass =
      atomic_class.candidate != 0 && atomic_class.negative_high == 0 &&
      atomic_class.low_words.size() > 1u &&
      (!tie || atomic_class.high_words.size() <= 2u) &&
      mask_class.zero != 0 && mask_class.one != 0 &&
      mask_class.poison == 0 && mask_class.other == 0;
  const bool pass = input_mismatches == 0 && atomic_canary_mismatches == 0 &&
                    mask_guard_mismatches == 0 && classification_pass;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L5/L15 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l5-l15-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"phase\": \"" << contract::PhaseName(phase) << "\",\n"
         << "  \"scenario\": \"" << contract::ScenarioName(scenario)
         << "\",\n"
         << "  \"selected_launch\": "
         << (phase == contract::Phase::kReset ? 5 : 15) << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"module_token\": 13,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"metal_functions\": [\"fg_early_l5_l15_high\", "
            "\"fg_early_l5_l15_low\", \"fg_early_l5_l15_pack\"],\n"
         << "  \"selected_grid\": [20, 12, 1],\n"
         << "  \"selected_block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"atomic_lowering\": \"signed-high-xor then u32 high-min, "
            "equal-high u32 low-min, GPU pack\",\n"
         << "  \"architecture_rom_sha256\": \"" << kRomSha256 << "\",\n"
         << "  \"input_generator\": \"" << contract::kInputGenerator
         << "\",\n"
         << "  \"flow_pre_sha256\": \"" << Sha256(flow_before) << "\",\n"
         << "  \"flow_post_sha256\": \"" << Sha256(flow_after) << "\",\n"
         << "  \"depth_pre_sha256\": \"" << Sha256(depth_before) << "\",\n"
         << "  \"depth_post_sha256\": \"" << Sha256(depth_after) << "\",\n"
         << "  \"atomic_output_sha256\": \"" << Sha256(atomic_output)
         << "\",\n"
         << "  \"mask_output_sha256\": \"" << Sha256(mask_output) << "\",\n"
         << "  \"atomic_output_bytes\": " << atomic_output.size() << ",\n"
         << "  \"mask_output_bytes\": " << mask_output.size() << ",\n"
         << "  \"atomic_sentinel_u64\": " << atomic_class.sentinel << ",\n"
         << "  \"atomic_candidate_u64\": " << atomic_class.candidate << ",\n"
         << "  \"atomic_negative_high_words\": " << atomic_class.negative_high
         << ",\n"
         << "  \"atomic_unique_high_words\": " << atomic_class.high_words.size()
         << ",\n"
         << "  \"atomic_unique_low_words\": " << atomic_class.low_words.size()
         << ",\n"
         << "  \"mask_zero_pixels\": " << mask_class.zero << ",\n"
         << "  \"mask_one_pixels\": " << mask_class.one << ",\n"
         << "  \"mask_poison_pixels\": " << mask_class.poison << ",\n"
         << "  \"mask_other_pixels\": " << mask_class.other << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"atomic_canary_mismatched_bytes\": "
         << atomic_canary_mismatches << ",\n"
         << "  \"mask_guard_row_mismatched_bytes\": "
         << mask_guard_mismatches << ",\n"
         << "  \"collision_fixture\": true,\n"
         << "  \"equal_high_tie_fixture\": " << (tie ? "true" : "false")
         << ",\n"
         << "  \"expected_or_archived_ingress\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-early-l5-l15-mac <metallib> <rcp-rom> "
                   "<output-dir> <run-id> <poison-hex> <reset|generated> "
                   "<equal-high-tie|priority-collision>\n";
      return 2;
    }
    try {
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha256) {
        throw std::runtime_error("RCP ROM identity mismatch");
      }
      char* poison_end = nullptr;
      const auto parsed_poison = std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' || parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const auto phase = contract::ParsePhase(argv[6]);
      const auto scenario = contract::ParseScenario(argv[7]);
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(output_root);
      const auto flow_before = contract::Flow(phase, scenario);
      const auto depth_before = contract::Depth(phase, scenario);
      std::vector<std::uint8_t> mask_before(
          contract::kMaskBytes + contract::kWidth, poison);
      const std::vector<std::uint8_t> high_before(contract::kPixels * 4u, 0xff);
      const std::vector<std::uint8_t> low_before(contract::kPixels * 4u, 0xff);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      auto pipeline = [&](NSString* name) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function == nil) throw std::runtime_error("Metal function missing");
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (result == nil) throw std::runtime_error("Metal pipeline failed");
        return result;
      };
      id<MTLComputePipelineState> high_pipeline =
          pipeline(@"fg_early_l5_l15_high");
      id<MTLComputePipelineState> low_pipeline =
          pipeline(@"fg_early_l5_l15_low");
      id<MTLComputePipelineState> pack_pipeline =
          pipeline(@"fg_early_l5_l15_pack");
      id<MTLTexture> flow = MakeTexture(
          device, MTLPixelFormatRG32Float, MTLTextureUsageShaderRead,
          contract::kHeight, contract::kWidth * 8u, flow_before);
      id<MTLTexture> depth = MakeTexture(
          device, MTLPixelFormatR32Uint, MTLTextureUsageShaderRead,
          contract::kHeight, contract::kWidth * 4u, depth_before);
      id<MTLTexture> mask = MakeTexture(
          device, MTLPixelFormatR8Unorm, MTLTextureUsageShaderWrite,
          contract::kHeight + 1u, contract::kWidth, mask_before);
      id<MTLBuffer> high = Guarded(device, &high_before, high_before.size(), 0);
      id<MTLBuffer> low = Guarded(device, &low_before, low_before.size(), 0);
      id<MTLBuffer> output =
          Guarded(device, nullptr, contract::kAtomicBytes, poison);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue failed");
      const Parameters parameters{contract::kWidth, contract::kHeight,
                                  144.0f, 12.96f, 1.0f, 0u};
      EncodePass(queue, high_pipeline, flow, depth, mask, high, low,
                 rom_buffer, parameters);
      EncodePass(queue, low_pipeline, flow, depth, mask, high, low,
                 rom_buffer, parameters);
      EncodePack(queue, pack_pipeline, high, low, output, parameters);

      const auto flow_after = ReadTexture(
          flow, contract::kHeight, contract::kWidth * 8u);
      const auto depth_after = ReadTexture(
          depth, contract::kHeight, contract::kWidth * 4u);
      const auto guarded_mask = ReadTexture(
          mask, contract::kHeight + 1u, contract::kWidth);
      const std::vector<std::uint8_t> mask_output(
          guarded_mask.begin(), guarded_mask.begin() + contract::kMaskBytes);
      const auto atomic_output = Payload(output, contract::kAtomicBytes);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      const std::size_t input_mismatches =
          Mismatches(flow_before, flow_after) +
          Mismatches(depth_before, depth_after) + Mismatches(rom, rom_after);
      const std::size_t atomic_canary_mismatches =
          CanaryMismatches(high, high_before.size()) +
          CanaryMismatches(low, low_before.size()) +
          CanaryMismatches(output, contract::kAtomicBytes) +
          CanaryMismatches(rom_buffer, kRomBytes);
      std::size_t mask_guard_mismatches = 0;
      for (std::size_t index = contract::kMaskBytes;
           index < guarded_mask.size(); ++index) {
        mask_guard_mismatches += guarded_mask[index] != poison;
      }
      const auto atomic_class = ClassifyAtomic(atomic_output);
      const auto mask_class = ClassifyMask(mask_output, poison);
      Write(output_root / "flow-after.bin", flow_after);
      Write(output_root / "depth-after.bin", depth_after);
      Write(output_root / "atomic-output.bin", atomic_output);
      Write(output_root / "mask-output.bin", mask_output);
      WriteReceipt(output_root / "run-receipt.json", run_id, phase, scenario,
                   poison, device.name.UTF8String, flow_before, depth_before,
                   flow_after, depth_after, atomic_output, mask_output,
                   input_mismatches, atomic_canary_mismatches,
                   mask_guard_mismatches, atomic_class, mask_class);
      const bool tie = scenario == contract::Scenario::kEqualHighTie;
      const bool pass = input_mismatches == 0 &&
                        atomic_canary_mismatches == 0 &&
                        mask_guard_mismatches == 0 &&
                        atomic_class.candidate != 0 &&
                        atomic_class.negative_high == 0 &&
                        atomic_class.low_words.size() > 1u &&
                        (!tie || atomic_class.high_words.size() <= 2u) &&
                        mask_class.zero != 0 && mask_class.one != 0 &&
                        mask_class.poison == 0 && mask_class.other == 0;
      std::cout << "probe=fg_early_l5_l15_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "atomic_output_sha256=" << Sha256(atomic_output) << '\n'
                << "mask_output_sha256=" << Sha256(mask_output) << '\n'
                << "atomic_sentinel_u64=" << atomic_class.sentinel << '\n'
                << "atomic_unique_high_words=" << atomic_class.high_words.size()
                << '\n'
                << "atomic_unique_low_words=" << atomic_class.low_words.size()
                << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
