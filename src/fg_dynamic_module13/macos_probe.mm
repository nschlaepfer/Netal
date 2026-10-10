#define main FgDynamicModule13EmbeddedMacMain
#include "../fg_early_l5_l15/macos_probe.mm"
#undef main

#include "contract.h"

namespace {
namespace dynamic_contract = fg_dynamic_module13;

id<MTLTexture> DynamicTexture(
    id<MTLDevice> device, MTLPixelFormat format, std::uint32_t width,
    std::uint32_t height, MTLTextureUsage usage,
    const std::vector<std::uint8_t>& bytes, std::size_t row_bytes) {
  if (bytes.size() != row_bytes * height) throw std::runtime_error("texture size");
  auto descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:format width:width height:height
      mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  auto texture = [device newTextureWithDescriptor:descriptor];
  if (!texture) throw std::runtime_error("texture allocation");
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0
             withBytes:bytes.data() bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> DynamicReadTexture(id<MTLTexture> texture,
                                             std::size_t row_bytes) {
  std::vector<std::uint8_t> result(row_bytes * texture.height);
  [texture getBytes:result.data() bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
         mipmapLevel:0];
  return result;
}

void DynamicEncodePass(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
    id<MTLTexture> flow, id<MTLTexture> depth, id<MTLTexture> mask,
    id<MTLBuffer> high, id<MTLBuffer> low, id<MTLBuffer> rom,
    const Parameters& parameters) {
  auto command = [queue commandBuffer];
  auto encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:flow atIndex:0];
  [encoder setTexture:depth atIndex:1];
  [encoder setTexture:mask atIndex:2];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:rom offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(parameters.width, parameters.height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error("candidate pass");
}

void DynamicEncodePack(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> high, id<MTLBuffer> low, id<MTLBuffer> output,
    const Parameters& parameters) {
  auto command = [queue commandBuffer];
  auto encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(
      std::size_t{parameters.width} * parameters.height, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error("pack pass");
}

void DynamicReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const dynamic_contract::Resolution& resolution,
    dynamic_contract::Phase phase, dynamic_contract::Scenario scenario,
    std::uint8_t poison, bool artifacts, const std::string& device,
    const std::vector<std::uint8_t>& flow_before,
    const std::vector<std::uint8_t>& flow_after,
    const std::vector<std::uint8_t>& depth_before,
    const std::vector<std::uint8_t>& depth_after,
    const std::vector<std::uint8_t>& atomic_output,
    const std::vector<std::uint8_t>& mask_output,
    std::size_t input_mismatch, std::size_t atomic_canary,
    std::size_t mask_canary, const AtomicClassification& atomic_class,
    const MaskClassification& mask_class) {
  const bool pass = input_mismatch == 0 && atomic_canary == 0 &&
                    mask_canary == 0 && atomic_class.candidate > 0 &&
                    atomic_class.low_words.size() > 1 && mask_class.zero > 0 &&
                    mask_class.one > 0 && mask_class.poison == 0 &&
                    mask_class.other == 0;
  std::ofstream stream(path);
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.dynamic-module13-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"device_name\": \"" << device << "\",\n"
         << "  \"resolution\": \"" << resolution.name << "\",\n"
         << "  \"phase\": \"" << dynamic_contract::PhaseName(phase) << "\",\n"
         << "  \"scenario\": \"" << dynamic_contract::ScenarioName(scenario) << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_sha256\": \"" << dynamic_contract::kModuleSha256 << "\",\n"
         << "  \"large_trace_launch\": "
         << (phase == dynamic_contract::Phase::kReset ? 5 : 17) << ",\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"extent\": [" << resolution.width << ", " << resolution.height << "],\n"
         << "  \"grid\": [" << resolution.grid_x << ", " << resolution.grid_y << ", 1],\n"
         << "  \"atomic_output_bytes\": " << atomic_output.size() << ",\n"
         << "  \"atomic_output_sha256\": \"" << Sha256(atomic_output) << "\",\n"
         << "  \"mask_output_bytes\": " << mask_output.size() << ",\n"
         << "  \"mask_output_sha256\": \"" << Sha256(mask_output) << "\",\n"
         << "  \"atomic_sentinel_u64\": " << atomic_class.sentinel << ",\n"
         << "  \"atomic_candidate_u64\": " << atomic_class.candidate << ",\n"
         << "  \"atomic_unique_high_words\": " << atomic_class.high_words.size() << ",\n"
         << "  \"atomic_unique_low_words\": " << atomic_class.low_words.size() << ",\n"
         << "  \"mask_zero_pixels\": " << mask_class.zero << ",\n"
         << "  \"mask_one_pixels\": " << mask_class.one << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatch << ",\n"
         << "  \"atomic_canary_mismatched_bytes\": " << atomic_canary << ",\n"
         << "  \"mask_guard_row_mismatched_bytes\": " << mask_canary << ",\n"
         << "  \"artifacts_written\": " << (artifacts ? "true" : "false") << ",\n"
         << "  \"expected_or_captured_output_ingress\": false,\n"
         << "  \"archived_intermediate_ingress\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n}\n";
}
}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 10) return 2;
    try {
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha256)
        throw std::runtime_error("rom");
      char* end = nullptr;
      const auto parsed = std::strtoul(argv[5], &end, 16);
      if (end == argv[5] || *end || parsed > 255) throw std::runtime_error("poison");
      const auto poison = static_cast<std::uint8_t>(parsed);
      const auto& resolution = dynamic_contract::ParseResolution(argv[6]);
      const auto phase = dynamic_contract::ParsePhase(argv[7]);
      const auto scenario = dynamic_contract::ParseScenario(argv[8]);
      const bool artifacts = std::string(argv[9]) == "full";
      const std::filesystem::path root = argv[3];
      std::filesystem::create_directories(root);
      const auto flow_before = dynamic_contract::Flow(resolution, phase, scenario);
      const auto depth_before = dynamic_contract::Depth(resolution, phase, scenario);
      const auto atomic_before = dynamic_contract::InitialAtomic(resolution);
      std::vector<std::uint8_t> mask_before(
          dynamic_contract::Pixels(resolution) + resolution.width, poison);
      const std::vector<std::uint8_t> words(
          dynamic_contract::Pixels(resolution) * 4u, 0xff);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                                         error:&error];
      auto pipeline = [&](NSString* name) {
        auto function = [library newFunctionWithName:name];
        return [device newComputePipelineStateWithFunction:function error:&error];
      };
      auto high_pipeline = pipeline(@"fg_early_l5_l15_high");
      auto low_pipeline = pipeline(@"fg_early_l5_l15_low");
      auto pack_pipeline = pipeline(@"fg_early_l5_l15_pack");
      auto flow = DynamicTexture(device, MTLPixelFormatRG32Float,
          resolution.width, resolution.height, MTLTextureUsageShaderRead,
          flow_before, resolution.width * 8u);
      auto depth = DynamicTexture(device, MTLPixelFormatR32Uint,
          resolution.width, resolution.height, MTLTextureUsageShaderRead,
          depth_before, resolution.width * 4u);
      auto mask = DynamicTexture(device, MTLPixelFormatR8Unorm,
          resolution.width, resolution.height + 1u, MTLTextureUsageShaderWrite,
          mask_before, resolution.width);
      auto high = Guarded(device, &words, words.size(), 0);
      auto low = Guarded(device, &words, words.size(), 0);
      auto output = Guarded(device, nullptr, atomic_before.size(), poison);
      auto rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      Parameters parameters{resolution.width, resolution.height,
          std::bit_cast<float>(resolution.motion_divisor_bits),
          std::bit_cast<float>(resolution.neighbor_divisor_bits), 1.0f, 0};
      auto queue = [device newCommandQueue];
      DynamicEncodePass(queue, high_pipeline, flow, depth, mask, high, low,
                        rom_buffer, parameters);
      DynamicEncodePass(queue, low_pipeline, flow, depth, mask, high, low,
                        rom_buffer, parameters);
      DynamicEncodePack(queue, pack_pipeline, high, low, output, parameters);
      const auto flow_after = DynamicReadTexture(flow, resolution.width * 8u);
      const auto depth_after = DynamicReadTexture(depth, resolution.width * 4u);
      const auto guarded_mask = DynamicReadTexture(mask, resolution.width);
      const auto atomic_output = Payload(output, atomic_before.size());
      const std::vector<std::uint8_t> mask_output(
          guarded_mask.begin(), guarded_mask.begin() + dynamic_contract::Pixels(resolution));
      const auto atomic_class = ClassifyAtomic(atomic_output);
      const auto mask_class = ClassifyMask(mask_output, poison);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      const std::size_t input_mismatch = Mismatches(flow_before, flow_after) +
          Mismatches(depth_before, depth_after) + Mismatches(rom, rom_after);
      const std::size_t atomic_canary = CanaryMismatches(high, words.size()) +
          CanaryMismatches(low, words.size()) +
          CanaryMismatches(output, atomic_before.size()) +
          CanaryMismatches(rom_buffer, kRomBytes);
      std::size_t mask_canary = 0;
      for (std::size_t i = mask_output.size(); i < guarded_mask.size(); ++i)
        mask_canary += guarded_mask[i] != poison;
      if (artifacts) {
        Write(root / "atomic-output.bin", atomic_output);
        Write(root / "mask-output.bin", mask_output);
      }
      DynamicReceipt(root / "run-receipt.json", argv[4], resolution, phase,
          scenario, poison, artifacts, device.name.UTF8String, flow_before,
          flow_after, depth_before, depth_after, atomic_output, mask_output,
          input_mismatch, atomic_canary, mask_canary, atomic_class, mask_class);
      std::cout << "atomic_output_sha256=" << Sha256(atomic_output)
                << "\nmask_output_sha256=" << Sha256(mask_output)
                << "\nresult=pass\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n'; return 1;
    }
  }
}
