#define FG_CONV_L20_NO_MAIN
#include "macos_probe.mm"
#include "custom_conv_pre_contract.h"

namespace {

namespace cp = fg_custom_conv_pre;

std::vector<std::uint8_t> ReadConvPre(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void WriteConvPreReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::vector<std::uint8_t>& weights_before,
    const std::vector<std::uint8_t>& bias_before,
    const std::vector<std::uint8_t>& input_before,
    const std::vector<std::uint8_t>& weights_after,
    const std::vector<std::uint8_t>& bias_after,
    const std::vector<std::uint8_t>& input_after,
    const std::vector<std::uint8_t>& output,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t poison_collisions, const std::string& device_name) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write convPre receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.custom-conv-pre-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"trace_launch_token\": \"launch-000057\",\n"
         << "  \"module_token\": 54,\n"
         << "  \"module_bytes\": " << cp::kModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << cp::kModuleSha256 << "\",\n"
         << "  \"metal_function\": \"fg_custom_block0_conv_pre\",\n"
         << "  \"selected_rtx_grid\": [80, 92, 1],\n"
         << "  \"selected_rtx_block\": [32, 2, 4],\n"
         << "  \"metal_dispatch_threads\": [320, 184, 32],\n"
         << "  \"metal_threads_per_threadgroup\": [8, 4, 1],\n"
         << "  \"packed_parameter_bytes\": 80,\n"
         << "  \"abi_scalars\": [32, 3, 3, 10, 1, 368, 640, 10, 1, 184, 320, 32],\n"
         << "  \"input_shape_nhwc\": [1, 368, 640, 10],\n"
         << "  \"output_shape_nhwc\": [1, 184, 320, 32],\n"
         << "  \"fused_operations\": [\"conv3x3\", \"leaky_relu\", \"clamp_-6_6\", \"average_pool_2x2\"],\n"
         << "  \"weight_source_token\": 106,\n"
         << "  \"bias_source_token\": 108,\n"
         << "  \"model_capture_point\": \"post_evaluate_0_fence_pre_evaluate_1\",\n"
         << "  \"model_capture_manifest_sha256\": \""
         << cp::kManifestSha256 << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights_before)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(weights_after)
         << "\",\n"
         << "  \"bias_pre_sha256\": \"" << Sha256(bias_before) << "\",\n"
         << "  \"bias_post_sha256\": \"" << Sha256(bias_after) << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-custom-conv-pre-v1\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"per_run_poison_collision_halfwords\": "
         << poison_collisions << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_model_files_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 7) {
      std::cerr << "usage=fg-custom-conv-pre-mac <metallib> <weights> <bias> "
                   "<output-dir> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const auto weights_before = ReadConvPre(argv[2]);
      const auto bias_before = ReadConvPre(argv[3]);
      if (weights_before.size() != cp::kWeightBytes ||
          Sha256(weights_before) != cp::kWeightSha256 ||
          bias_before.size() != cp::kBiasBytes ||
          Sha256(bias_before) != cp::kBiasSha256) {
        throw std::runtime_error("convPre model identity mismatch");
      }
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[6], &poison_end, 16);
      if (poison_end == argv[6] || *poison_end != '\0' ||
          parsed_poison > 255) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const std::filesystem::path output_directory = argv[4];
      std::filesystem::create_directories(output_directory);
      const auto input_before = cp::GenerateInput();
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_custom_block0_conv_pre"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLBuffer> weights =
          Guarded(device, &weights_before, cp::kWeightBytes, 0);
      id<MTLBuffer> bias = Guarded(device, &bias_before, cp::kBiasBytes, 0);
      id<MTLBuffer> input = Guarded(device, &input_before, cp::kInputBytes, 0);
      id<MTLBuffer> output =
          Guarded(device, nullptr, cp::kOutputBytes, poison);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [encoder setBuffer:bias offset:kGuardBytes atIndex:1];
      [encoder setBuffer:input offset:kGuardBytes atIndex:2];
      [encoder setBuffer:output offset:kGuardBytes atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(cp::kOutputWidth, cp::kOutputHeight,
                                           cp::kOutputChannels)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      const auto weights_after = Payload(weights, cp::kWeightBytes);
      const auto bias_after = Payload(bias, cp::kBiasBytes);
      const auto input_after = Payload(input, cp::kInputBytes);
      const auto actual_output = Payload(output, cp::kOutputBytes);
      const std::size_t mismatches =
          Mismatches(weights_before, weights_after) +
          Mismatches(bias_before, bias_after) +
          Mismatches(input_before, input_after);
      const std::size_t canaries =
          CanaryMismatches(weights, cp::kWeightBytes) +
          CanaryMismatches(bias, cp::kBiasBytes) +
          CanaryMismatches(input, cp::kInputBytes) +
          CanaryMismatches(output, cp::kOutputBytes);
      const std::size_t collisions = UntouchedHalfwords(actual_output, poison);
      Write(output_directory / "input-after.bin", input_after);
      Write(output_directory / "output.bin", actual_output);
      WriteConvPreReceipt(output_directory / "run-receipt.json", argv[5],
                          poison, weights_before, bias_before, input_before,
                          weights_after, bias_after, input_after,
                          actual_output, mismatches, canaries, collisions,
                          device.name.UTF8String);
      const bool pass = mismatches == 0 && canaries == 0;
      std::cout << "probe=fg_custom_conv_pre_mac\noutput_sha256="
                << Sha256(actual_output) << "\nresult="
                << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
