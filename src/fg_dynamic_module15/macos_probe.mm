#define main FgDynamicModule15EmbeddedMacMain
#include "../fg_early_l4_l14/macos_probe.mm"
#undef main

#include "contract.h"

namespace {
namespace dynamic_contract = fg_dynamic_module15;

struct DynamicSourceParameters {
  std::uint64_t vector_input;
  std::array<std::uint32_t, 2> vector_origin;
  std::array<std::uint32_t, 2> vector_extent;
  std::array<float, 2> vector_inverse_extent;
  std::uint64_t vector_output;
  std::uint64_t scalar_input;
  std::array<std::uint32_t, 2> scalar_origin;
  std::array<std::uint32_t, 2> scalar_extent;
  std::array<float, 2> scalar_inverse_extent;
  float abi_dead_threshold;
  float score_numerator;
  float abi_dead_scale;
  std::uint8_t mode;
  std::array<std::uint8_t, 3> mode_padding;
  std::uint64_t depth_output;
  std::array<std::uint64_t, 6> abi_dead;
  std::array<std::uint32_t, 2> output_extent;
  std::array<float, 2> vector_output_scale;
};
static_assert(sizeof(DynamicSourceParameters) == 160);

struct DynamicMetalParameters {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::array<float, 2> input_inverse;
  std::array<float, 2> output_inverse;
  std::array<float, 2> lower_coordinate;
  std::array<float, 2> upper_coordinate;
  std::array<float, 2> output_scale;
};
static_assert(sizeof(DynamicMetalParameters) == 56);

DynamicSourceParameters DynamicMakeSourceParameters(
    const dynamic_contract::Resolution& resolution, bool reset) {
  return DynamicSourceParameters{
      0,
      {0, 0},
      {resolution.input_width, resolution.input_height},
      {std::bit_cast<float>(resolution.inverse_x_bits),
       std::bit_cast<float>(resolution.inverse_y_bits)},
      0,
      0,
      {0, 0},
      {resolution.input_width, resolution.input_height},
      {std::bit_cast<float>(resolution.inverse_x_bits),
       std::bit_cast<float>(resolution.inverse_y_bits)},
      40.0f,
      1.0f,
      600.0f,
      0,
      {static_cast<std::uint8_t>(reset ? 127u : 0u), 0, 0},
      0,
      {0, 0, 0, 0, 0, 0},
      {resolution.output_width, resolution.output_height},
      {std::bit_cast<float>(resolution.output_scale_x_bits),
       std::bit_cast<float>(resolution.output_scale_y_bits)}};
}

DynamicMetalParameters DynamicMakeMetalParameters(
    const dynamic_contract::Resolution& resolution) {
  return DynamicMetalParameters{
      resolution.input_width,
      resolution.input_height,
      resolution.output_width,
      resolution.output_height,
      {std::bit_cast<float>(resolution.inverse_x_bits),
       std::bit_cast<float>(resolution.inverse_y_bits)},
      {std::bit_cast<float>(resolution.inverse_x_bits + 0x00800000u),
       std::bit_cast<float>(resolution.inverse_y_bits + 0x00800000u)},
      {std::bit_cast<float>(resolution.lower_x_bits),
       std::bit_cast<float>(resolution.lower_y_bits)},
      {std::bit_cast<float>(resolution.upper_x_bits),
       std::bit_cast<float>(resolution.upper_y_bits)},
      {std::bit_cast<float>(resolution.output_scale_x_bits),
       std::bit_cast<float>(resolution.output_scale_y_bits)}};
}

std::vector<std::uint8_t> DynamicSourceBytes(
    const DynamicSourceParameters& parameters) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&parameters);
  return std::vector<std::uint8_t>(begin, begin + sizeof(parameters));
}

void DynamicWriteMacReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const dynamic_contract::Resolution& resolution, bool reset,
    std::uint8_t poison, bool artifacts_written, const std::string& device_name,
    const std::vector<std::uint8_t>& vector_before,
    const std::vector<std::uint8_t>& vector_after,
    const std::vector<std::uint8_t>& scalar_before,
    const std::vector<std::uint8_t>& scalar_after,
    const std::vector<std::uint8_t>& vector_output,
    const std::vector<std::uint8_t>& depth_output,
    const std::string& normalized_parameter_sha,
    std::size_t input_mismatches, std::size_t vector_canary_mismatches,
    std::size_t depth_canary_mismatches, std::size_t vector_untouched,
    std::size_t depth_untouched) {
  const bool pass = input_mismatches == 0 && vector_canary_mismatches == 0 &&
                    depth_canary_mismatches == 0 && vector_untouched == 0 &&
                    depth_untouched == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write dynamic module15 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.dynamic-module15-run.v1\",\n"
          << "  \"backend\": \"mac-metal\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << getpid() << ",\n"
          << "  \"device_name\": \"" << device_name << "\",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"phase\": \"" << (reset ? "reset" : "generated") << "\",\n"
          << "  \"large_trace_launch\": " << (reset ? 4 : 16) << ",\n"
          << "  \"canonical_launch\": " << (reset ? 4 : 14) << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module_token\": 15,\n"
          << "  \"module_sha256\": \"" << dynamic_contract::kModuleSha256
          << "\",\n"
          << "  \"metal_function\": \"fg_dynamic_module15\",\n"
          << "  \"packed_parameter_bytes\": 160,\n"
          << "  \"selected_mode_byte_84\": 0,\n"
          << "  \"reset_marker_byte_85\": " << (reset ? 127 : 0) << ",\n"
          << "  \"input_extent\": [" << resolution.input_width << ", "
          << resolution.input_height << "],\n"
          << "  \"output_extent\": [" << resolution.output_width << ", "
          << resolution.output_height << "],\n"
          << "  \"selected_rtx_grid\": [" << resolution.grid_x << ", "
          << resolution.grid_y << ", 1],\n"
          << "  \"selected_rtx_block\": [16, 16, 1],\n"
          << "  \"metal_dispatch_threads\": [" << resolution.output_width
          << ", " << resolution.output_height << ", 1],\n"
          << "  \"trace_raw_parameter_sha256\": \""
          << (reset ? resolution.reset_raw_sha256
                    : resolution.generated_raw_sha256) << "\",\n"
          << "  \"trace_address_free_parameter_sha256\": \""
          << (reset ? resolution.reset_address_free_sha256
                    : resolution.generated_address_free_sha256) << "\",\n"
          << "  \"fixture_address_free_parameter_sha256\": \""
          << normalized_parameter_sha << "\",\n"
          << "  \"input_generator\": \"" << dynamic_contract::kGenerator
          << "\",\n"
          << "  \"vector_input_pre_sha256\": \"" << HashBytes(vector_before)
          << "\",\n"
          << "  \"vector_input_post_sha256\": \"" << HashBytes(vector_after)
          << "\",\n"
          << "  \"scalar_input_pre_sha256\": \"" << HashBytes(scalar_before)
          << "\",\n"
          << "  \"scalar_input_post_sha256\": \"" << HashBytes(scalar_after)
          << "\",\n"
          << "  \"vector_output_bytes\": " << vector_output.size() << ",\n"
          << "  \"vector_output_sha256\": \"" << HashBytes(vector_output)
          << "\",\n"
          << "  \"depth_output_bytes\": " << depth_output.size() << ",\n"
          << "  \"depth_output_sha256\": \"" << HashBytes(depth_output)
          << "\",\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"vector_canary_mismatched_bytes\": "
          << vector_canary_mismatches << ",\n"
          << "  \"depth_canary_mismatched_bytes\": "
          << depth_canary_mismatches << ",\n"
          << "  \"vector_untouched_pixels\": " << vector_untouched << ",\n"
          << "  \"depth_untouched_pixels\": " << depth_untouched << ",\n"
          << "  \"artifacts_written\": "
          << (artifacts_written ? "true" : "false") << ",\n"
          << "  \"typed_resource_backend\": true,\n"
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
      std::cerr << "usage=fg-dynamic-module15-mac <metallib> <output-dir> "
                   "<run-id> <poison> <resolution> <reset|generated> "
                   "<full|hash-only>\n";
      return 2;
    }
    try {
      const std::filesystem::path library_path = argv[1];
      const std::filesystem::path output_root = argv[2];
      const std::string run_id = argv[3];
      char* poison_end = nullptr;
      const unsigned long poison_value = std::strtoul(argv[4], &poison_end, 16);
      if (poison_end == argv[4] || *poison_end != '\0' || poison_value > 255u)
        throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(poison_value);
      const auto& resolution = dynamic_contract::ParseResolution(argv[5]);
      const std::string phase = argv[6];
      if (phase != "reset" && phase != "generated")
        throw std::runtime_error("invalid phase");
      const bool reset = phase == "reset";
      const std::string artifact_mode = argv[7];
      if (artifact_mode != "full" && artifact_mode != "hash-only")
        throw std::runtime_error("invalid artifact mode");
      const bool artifacts_written = artifact_mode == "full";
      std::filesystem::create_directories(output_root);
      const auto vector_before = dynamic_contract::VectorInput(resolution, !reset);
      const auto scalar_before = dynamic_contract::ScalarInput(resolution, !reset);
      const std::size_t vector_row_bytes =
          std::size_t{resolution.output_width} * 8u;
      const std::size_t depth_row_bytes =
          std::size_t{resolution.output_width} * 4u;
      const std::vector<std::uint8_t> vector_initial(
          vector_row_bytes * (resolution.output_height + 1u), poison);
      const std::vector<std::uint8_t> depth_initial(
          depth_row_bytes * (resolution.output_height + 1u), poison);
      const auto source_parameters =
          DynamicMakeSourceParameters(resolution, reset);
      const std::string normalized_parameter_sha =
          HashBytes(DynamicSourceBytes(source_parameters));
      const char* expected_normalized = reset
          ? resolution.reset_address_free_sha256
          : resolution.generated_address_free_sha256;
      if (normalized_parameter_sha != expected_normalized)
        throw std::runtime_error("address-free parameter mismatch");

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(library_path.c_str())]
                      error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_dynamic_module15"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (function == nil || pipeline == nil)
        throw std::runtime_error("module15 pipeline failed");
      auto vector_input = MakeL4L14Texture(
          device, MTLPixelFormatRG16Float, resolution.input_width,
          resolution.input_height, MTLTextureUsageShaderRead, vector_before,
          std::size_t{resolution.input_width} * 4u);
      auto scalar_input = MakeL4L14Texture(
          device, MTLPixelFormatR32Float, resolution.input_width,
          resolution.input_height, MTLTextureUsageShaderRead, scalar_before,
          std::size_t{resolution.input_width} * 4u);
      auto vector_output = MakeL4L14Texture(
          device, MTLPixelFormatRG32Float, resolution.output_width,
          resolution.output_height + 1u, MTLTextureUsageShaderWrite,
          vector_initial, vector_row_bytes);
      auto depth_output = MakeL4L14Texture(
          device, MTLPixelFormatR32Uint, resolution.output_width,
          resolution.output_height + 1u, MTLTextureUsageShaderWrite,
          depth_initial, depth_row_bytes);
      const auto metal_parameters = DynamicMakeMetalParameters(resolution);
      id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:vector_input atIndex:0];
      [encoder setTexture:scalar_input atIndex:1];
      [encoder setTexture:vector_output atIndex:2];
      [encoder setTexture:depth_output atIndex:3];
      [encoder setBytes:&metal_parameters length:sizeof(metal_parameters)
                atIndex:0];
      [encoder dispatchThreads:
          MTLSizeMake(resolution.output_width, resolution.output_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal module15 command failed");
      const auto vector_after =
          ReadL4L14Texture(vector_input, std::size_t{resolution.input_width} * 4u);
      const auto scalar_after =
          ReadL4L14Texture(scalar_input, std::size_t{resolution.input_width} * 4u);
      const auto vector_guarded = ReadL4L14Texture(vector_output,
                                                    vector_row_bytes);
      const auto depth_guarded = ReadL4L14Texture(depth_output,
                                                  depth_row_bytes);
      const auto vector_payload = SurfacePayload(
          vector_guarded, dynamic_contract::VectorOutputBytes(resolution));
      const auto depth_payload = SurfacePayload(
          depth_guarded, dynamic_contract::DepthOutputBytes(resolution));
      const std::size_t input_mismatches =
          CountMismatches(vector_before, vector_after) +
          CountMismatches(scalar_before, scalar_after);
      const std::size_t vector_canary_mismatches = TailCanaryMismatches(
          vector_guarded, vector_payload.size(), poison);
      const std::size_t depth_canary_mismatches = TailCanaryMismatches(
          depth_guarded, depth_payload.size(), poison);
      const std::size_t vector_untouched =
          UntouchedPixels(vector_payload, 8u, poison);
      const std::size_t depth_untouched =
          UntouchedPixels(depth_payload, 4u, poison);
      if (artifacts_written) {
        WriteBytes(output_root / "vector-output.bin", vector_payload);
        WriteBytes(output_root / "depth-output.bin", depth_payload);
      }
      DynamicWriteMacReceipt(
          output_root / "run-receipt.json", run_id, resolution, reset, poison,
          artifacts_written, device.name.UTF8String, vector_before,
          vector_after, scalar_before, scalar_after, vector_payload,
          depth_payload, normalized_parameter_sha, input_mismatches,
          vector_canary_mismatches, depth_canary_mismatches,
          vector_untouched, depth_untouched);
      const bool pass = input_mismatches == 0 &&
                        vector_canary_mismatches == 0 &&
                        depth_canary_mismatches == 0 &&
                        vector_untouched == 0 && depth_untouched == 0;
      std::cout << "probe=fg_dynamic_module15_mac\n"
                << "run_id=" << run_id << '\n'
                << "vector_output_sha256=" << HashBytes(vector_payload) << '\n'
                << "depth_output_sha256=" << HashBytes(depth_payload) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
