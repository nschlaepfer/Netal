#define main FgMidL55EmbeddedMain
#include "../fg_mid_l55/macos_probe.mm"
#undef main

#include "contract.h"

namespace {

namespace contract = fg_early_l3_l13;

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t half_width;
  std::uint32_t half_height;
};
static_assert(sizeof(Parameters) == 16);

std::vector<std::uint8_t> JoinOutputs(
    std::initializer_list<const std::vector<std::uint8_t>*> values) {
  std::vector<std::uint8_t> result;
  for (const auto* value : values)
    result.insert(result.end(), value->begin(), value->end());
  return result;
}

std::size_t PoisonHalfwords(const std::vector<std::uint8_t>& bytes,
                            std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 2u)
    result += bytes[offset] == poison && bytes[offset + 1u] == poison;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-early-l3-l13-mac <metallib> <output-dir> "
                 "<run-id> <poison-hex> <l3-evaluate-0|l13-evaluate-1> "
                 "<reset-history|generated-history>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const fs::path library_path = argv[1];
      const fs::path output_directory = argv[2];
      const std::string run_id = argv[3];
      const auto poison = ParsePoison(argv[4]);
      const std::string phase = argv[5];
      const std::string fixture = argv[6];
      const auto selected_launch = contract::SelectedLaunch(phase);
      fs::create_directories(output_directory);
      const auto source_before = contract::GenerateSource();
      const auto history_before = contract::GenerateHistory(fixture);
      const std::vector<std::uint8_t> canary_before(
          contract::kCanaryBytes, contract::kCanaryValue);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* library_error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(library_path.c_str())]
                      error:&library_error];
      if (library == nil)
        throw std::runtime_error(std::string("metallib load failed: ") +
                                 (library_error
                                      ? library_error.localizedDescription.UTF8String
                                      : "unknown"));
      auto pipeline = Pipeline(device, library, @"fg_early_l3_l13_mode2");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      auto source = Texture(device, MTLPixelFormatRGBA8Unorm,
                            contract::kWidth, contract::kHeight,
                            MTLTextureUsageShaderRead, source_before, 4);
      auto history = Texture(device, MTLPixelFormatRGBA16Float,
                             contract::kWidth, contract::kHeight,
                             MTLTextureUsageShaderRead, history_before, 8);
      auto half_color = Texture(
          device, MTLPixelFormatRGBA16Float, contract::kHalfWidth,
          contract::kHalfHeight, MTLTextureUsageShaderWrite,
          contract::Poison(contract::kHalfColorBytes, poison), 8);
      auto half_metric = Texture(
          device, MTLPixelFormatR16Float, contract::kHalfWidth,
          contract::kHalfHeight, MTLTextureUsageShaderWrite,
          contract::Poison(contract::kHalfMetricBytes, poison), 2);
      auto next_history = Texture(
          device, MTLPixelFormatRGBA16Float, contract::kWidth,
          contract::kHeight, MTLTextureUsageShaderWrite,
          contract::Poison(contract::kHistoryBytes, poison), 8);
      id<MTLBuffer> canary_left =
          GuardedBuffer(device, canary_before);
      id<MTLBuffer> canary_right =
          GuardedBuffer(device, canary_before);
      const Parameters parameters{contract::kWidth, contract::kHeight,
                                  contract::kHalfWidth,
                                  contract::kHalfHeight};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [encoder setTexture:source atIndex:0];
      [encoder setTexture:history atIndex:1];
      [encoder setTexture:half_color atIndex:2];
      [encoder setTexture:half_metric atIndex:3];
      [encoder setTexture:next_history atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(contract::kHalfWidth,
                                           contract::kHalfHeight,1)
          threadsPerThreadgroup:MTLSizeMake(16,8,1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal L3/L13 command failed");

      const auto source_after = ReadTexture(source, 4);
      const auto history_after = ReadTexture(history, 8);
      const auto half_color_after = ReadTexture(half_color, 8);
      const auto half_metric_after = ReadTexture(half_metric, 2);
      const auto next_history_after = ReadTexture(next_history, 8);
      const auto canary_left_after = ReadAll(canary_left);
      const auto canary_right_after = ReadAll(canary_right);
      const auto combined = JoinOutputs(
          {&half_color_after, &half_metric_after, &next_history_after});
      const std::size_t input_mismatches =
          Mismatches(source_before, source_after) +
          Mismatches(history_before, history_after);
      const std::vector<std::uint8_t> expected_canary(
          contract::kCanaryBytes + 2u * kGuardBytes,
          contract::kCanaryValue);
      const std::size_t canary_mismatches =
          Mismatches(expected_canary, canary_left_after) +
          Mismatches(expected_canary, canary_right_after);
      const std::size_t poison_halfwords =
          PoisonHalfwords(half_color_after, poison) +
          PoisonHalfwords(half_metric_after, poison) +
          PoisonHalfwords(next_history_after, poison);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      Write(output_directory / "half-color.rgba16f", half_color_after);
      Write(output_directory / "half-metric.r16f", half_metric_after);
      Write(output_directory / "next-history.rgba16f", next_history_after);
      Write(output_directory / "output.bin", combined);
      std::ofstream receipt(output_directory / "run-receipt.json");
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.early-l3-l13-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"phase\": \"" << phase << "\",\n"
              << "  \"fixture\": \"" << fixture << "\",\n"
              << "  \"selected_launch\": " << selected_launch << ",\n"
              << "  \"module_token\": 11,\n"
              << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
              << "  \"raw_parameter_sha256\": \""
              << (selected_launch == 3 ? contract::kL3RawParameterSha256
                                       : contract::kL13RawParameterSha256)
              << "\",\n"
              << "  \"address_free_parameter_sha256\": \""
              << (selected_launch == 3
                      ? contract::kL3AddressFreeParameterSha256
                      : contract::kL13AddressFreeParameterSha256)
              << "\",\n"
              << "  \"history_read_bank\": \""
              << contract::HistoryReadBank(phase) << "\",\n"
              << "  \"history_write_bank\": \""
              << contract::HistoryWriteBank(phase) << "\",\n"
              << "  \"input_generator\": \"" << contract::kGenerator << "\",\n"
              << "  \"input_sha256\": [\"" << Sha256(source_before)
              << "\", \"" << Sha256(history_before) << "\"],\n"
              << "  \"output_sha256\": {\"half_color\": \""
              << Sha256(half_color_after) << "\", \"half_metric\": \""
              << Sha256(half_metric_after) << "\", \"next_history\": \""
              << Sha256(next_history_after) << "\"},\n"
              << "  \"combined_output_bytes\": " << combined.size() << ",\n"
              << "  \"combined_output_sha256\": \"" << Sha256(combined)
              << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
              << "  \"poison_pattern_halfwords\": " << poison_halfwords << ",\n"
              << "  \"typed_resource_backend\": true,\n"
              << "  \"expected_or_archived_tensor_ingress\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "phase=" << phase << '\n'
                << "fixture=" << fixture << '\n'
                << "combined_output_sha256=" << Sha256(combined) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
