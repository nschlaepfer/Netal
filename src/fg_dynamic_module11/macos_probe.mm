#define main FgDynamicModule11EmbeddedMacMain
#include "../fg_mid_l55/macos_probe.mm"
#undef main

#include <algorithm>

#include "contract.h"

namespace {
namespace dynamic_contract = fg_dynamic_module11;

struct DynamicParameters {
  std::uint32_t source_width;
  std::uint32_t source_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t mode;
};
static_assert(sizeof(DynamicParameters) == 20);

std::vector<std::uint8_t> DynamicJoinOutputs(
    std::initializer_list<const std::vector<std::uint8_t>*> values) {
  std::vector<std::uint8_t> result;
  std::size_t bytes = 0;
  for (const auto* value : values) bytes += value->size();
  result.reserve(bytes);
  for (const auto* value : values)
    result.insert(result.end(), value->begin(), value->end());
  return result;
}

std::size_t DynamicPoisonHalfwords(const std::vector<std::uint8_t>& bytes,
                                   std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 2u)
    result += bytes[offset] == poison && bytes[offset + 1u] == poison;
  return result;
}

void DynamicWriteMacReceipt(
    const fs::path& path, const std::string& run_id,
    const dynamic_contract::Resolution& resolution, bool generated,
    std::uint8_t poison, bool artifacts_written, const std::string& device,
    const std::vector<std::uint8_t>& source_before,
    const std::vector<std::uint8_t>& history_before,
    const std::vector<std::uint8_t>& half_color,
    const std::vector<std::uint8_t>& half_metric,
    const std::vector<std::uint8_t>& next_history,
    const std::vector<std::uint8_t>& combined,
    std::size_t input_mismatches, std::size_t canary_mismatches) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write module11 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.dynamic-module11-run.v1\",\n"
          << "  \"backend\": \"mac-metal\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << getpid() << ",\n"
          << "  \"device_name\": \"" << device << "\",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"phase\": \"" << (generated ? "generated" : "reset")
          << "\",\n"
          << "  \"large_trace_launch\": " << (generated ? 15 : 3) << ",\n"
          << "  \"canonical_launch\": " << (generated ? 13 : 3) << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module_token\": 11,\n"
          << "  \"module_sha256\": \"" << dynamic_contract::kModuleSha256
          << "\",\n"
          << "  \"metal_function\": \"fg_dynamic_module11\",\n"
          << "  \"packed_parameter_bytes\": 168,\n"
          << "  \"selected_mode\": " << resolution.mode << ",\n"
          << "  \"source_extent\": [" << resolution.source_width << ", "
          << resolution.source_height << "],\n"
          << "  \"output_extent\": [" << resolution.output_width << ", "
          << resolution.output_height << "],\n"
          << "  \"selected_rtx_grid\": [" << resolution.grid_x << ", "
          << resolution.grid_y << ", 1],\n"
          << "  \"selected_rtx_block\": [256, 1, 1],\n"
          << "  \"metal_dispatch_threads\": [" << resolution.output_width
          << ", " << resolution.output_height << ", 1],\n"
          << "  \"trace_raw_parameter_sha256\": \""
          << (generated ? resolution.generated_raw_parameter_sha256
                        : resolution.reset_raw_parameter_sha256)
          << "\",\n"
          << "  \"trace_address_free_parameter_sha256\": \""
          << resolution.address_free_parameter_sha256 << "\",\n"
          << "  \"history_read_bank\": \""
          << dynamic_contract::HistoryReadBank(generated) << "\",\n"
          << "  \"history_write_bank\": \""
          << dynamic_contract::HistoryWriteBank(generated) << "\",\n"
          << "  \"input_generator\": \"" << dynamic_contract::kGenerator
          << "\",\n"
          << "  \"source_pre_sha256\": \"" << Sha256(source_before)
          << "\",\n"
          << "  \"history_pre_sha256\": \"" << Sha256(history_before)
          << "\",\n"
          << "  \"half_color_sha256\": \"" << Sha256(half_color) << "\",\n"
          << "  \"half_metric_sha256\": \"" << Sha256(half_metric) << "\",\n"
          << "  \"next_history_sha256\": \"" << Sha256(next_history)
          << "\",\n"
          << "  \"combined_output_bytes\": " << combined.size() << ",\n"
          << "  \"combined_output_sha256\": \"" << Sha256(combined)
          << "\",\n"
          << "  \"poison_pattern_halfwords\": "
          << DynamicPoisonHalfwords(combined, poison) << ",\n"
          << "  \"artifacts_written\": "
          << (artifacts_written ? "true" : "false") << ",\n"
          << "  \"typed_resource_backend\": true,\n"
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

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage=fg-dynamic-module11-mac <metallib> <output-dir> "
                 "<run-id> <poison-hex> <resolution> <reset|generated> "
                 "<full|hash-only>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const fs::path library_path = argv[1];
      const fs::path output_root = argv[2];
      const std::string run_id = argv[3];
      const auto poison = ParsePoison(argv[4]);
      const auto& resolution = dynamic_contract::ParseResolution(argv[5]);
      const std::string phase = argv[6];
      if (phase != "reset" && phase != "generated")
        throw std::runtime_error("invalid phase");
      const bool generated = phase == "generated";
      const std::string artifact_mode = argv[7];
      if (artifact_mode != "full" && artifact_mode != "hash-only")
        throw std::runtime_error("invalid artifact mode");
      const bool artifacts_written = artifact_mode == "full";
      fs::create_directories(output_root);
      const auto source_before = dynamic_contract::Source(resolution, generated);
      const auto history_before = dynamic_contract::History(resolution, generated);
      const std::vector<std::uint8_t> canary_before(
          dynamic_contract::kCanaryBytes, dynamic_contract::kCanaryValue);
      const auto half_color_initial = dynamic_contract::Poison(
          dynamic_contract::HalfColorBytes(resolution), poison);
      const auto half_metric_initial = dynamic_contract::Poison(
          dynamic_contract::HalfMetricBytes(resolution), poison);
      const auto next_history_initial = dynamic_contract::Poison(
          dynamic_contract::HistoryBytes(resolution), poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(library_path.c_str())]
                      error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      auto pipeline = Pipeline(device, library, @"fg_dynamic_module11");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      auto source = Texture(device, MTLPixelFormatRGBA8Unorm,
                            resolution.source_width, resolution.source_height,
                            MTLTextureUsageShaderRead, source_before, 4);
      auto history = Texture(device, MTLPixelFormatRGBA16Float,
                             resolution.source_width, resolution.source_height,
                             MTLTextureUsageShaderRead, history_before, 8);
      auto half_color = Texture(
          device, MTLPixelFormatRGBA16Float, resolution.output_width,
          resolution.output_height, MTLTextureUsageShaderWrite,
          half_color_initial, 8);
      auto half_metric = Texture(
          device, MTLPixelFormatR16Float, resolution.output_width,
          resolution.output_height, MTLTextureUsageShaderWrite,
          half_metric_initial, 2);
      auto next_history = Texture(
          device, MTLPixelFormatRGBA16Float, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderWrite,
          next_history_initial, 8);
      id<MTLBuffer> canary_left = GuardedBuffer(device, canary_before);
      id<MTLBuffer> canary_right = GuardedBuffer(device, canary_before);
      const DynamicParameters parameters{
          resolution.source_width, resolution.source_height,
          resolution.output_width, resolution.output_height, resolution.mode};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [encoder setTexture:source atIndex:0];
      [encoder setTexture:history atIndex:1];
      [encoder setTexture:half_color atIndex:2];
      [encoder setTexture:half_metric atIndex:3];
      [encoder setTexture:next_history atIndex:4];
      [encoder dispatchThreads:
          MTLSizeMake(resolution.output_width, resolution.output_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(std::string("Metal module11 command failed: ") +
            (command.error ? command.error.localizedDescription.UTF8String
                           : "unknown"));

      const auto source_after = ReadTexture(source, 4);
      const auto history_after = ReadTexture(history, 8);
      const auto half_color_after = ReadTexture(half_color, 8);
      const auto half_metric_after = ReadTexture(half_metric, 2);
      const auto next_history_after = ReadTexture(next_history, 8);
      const auto canary_left_after = ReadAll(canary_left);
      const auto canary_right_after = ReadAll(canary_right);
      const auto combined = DynamicJoinOutputs(
          {&half_color_after, &half_metric_after, &next_history_after});
      const std::size_t input_mismatches =
          Mismatches(source_before, source_after) +
          Mismatches(history_before, history_after);
      const std::vector<std::uint8_t> expected_canary(
          dynamic_contract::kCanaryBytes + 2u * kGuardBytes,
          dynamic_contract::kCanaryValue);
      const std::size_t canary_mismatches =
          Mismatches(expected_canary, canary_left_after) +
          Mismatches(expected_canary, canary_right_after);
      if (artifacts_written) {
        Write(output_root / "half-color.rgba16f", half_color_after);
        Write(output_root / "half-metric.r16f", half_metric_after);
        Write(output_root / "next-history.rgba16f", next_history_after);
        Write(output_root / "output.bin", combined);
      }
      DynamicWriteMacReceipt(
          output_root / "run-receipt.json", run_id, resolution, generated,
          poison, artifacts_written, device.name.UTF8String, source_before,
          history_before, half_color_after, half_metric_after,
          next_history_after, combined, input_mismatches, canary_mismatches);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_dynamic_module11_mac\n"
                << "run_id=" << run_id << '\n'
                << "combined_output_sha256=" << Sha256(combined) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
