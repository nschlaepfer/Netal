#define FG_CONV_L20_NO_MAIN
#include "macos_probe.mm"

#include "custom_block0_bot1_contract.h"

namespace {
namespace contract = fg_custom_block0_bot1;

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("open: " + path.string());
  const auto length = stream.tellg();
  std::vector<std::uint8_t> result(static_cast<std::size_t>(length));
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(result.data()), length);
  return result;
}

void Record(const std::filesystem::path &path, const std::string &run_id,
            std::uint8_t poison, const std::vector<std::uint8_t> &weights0,
            const std::vector<std::uint8_t> &bias0,
            const std::array<std::vector<std::uint8_t>, 3> &inputs0,
            const std::vector<std::uint8_t> &weights1,
            const std::vector<std::uint8_t> &bias1,
            const std::array<std::vector<std::uint8_t>, 3> &inputs1,
            const std::vector<std::uint8_t> &output,
            std::size_t input_mismatches, std::size_t canary_mismatches,
            const std::string &device_name) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path);
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.custom-block0-bot1-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"trace_launch_token\": \"launch-000069\",\n"
         << "  \"module_token\": 59,\n"
         << "  \"module_bytes\": " << contract::kModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"metal_function\": \"fg_custom_block0_bot1\",\n"
         << "  \"selected_rtx_grid\": [40, 23, 3],\n"
         << "  \"selected_rtx_block\": [32, 4, 1],\n"
         << "  \"metal_dispatch_threads\": [320, 184, 8],\n"
         << "  \"metal_threads_per_threadgroup\": [8, 4, 1],\n"
         << "  \"weight_source_token\": 154,\n"
         << "  \"bias_source_token\": 156,\n"
         << "  \"model_manifest_sha256\": \"" << contract::kManifestSha256
         << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights0) << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(weights1) << "\",\n"
         << "  \"bias_pre_sha256\": \"" << Sha256(bias0) << "\",\n"
         << "  \"bias_post_sha256\": \"" << Sha256(bias1) << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-custom-block0-bot1-v1\",\n"
         << "  \"input_pre_sha256\": [\"" << Sha256(inputs0[0]) << "\", \""
         << Sha256(inputs0[1]) << "\", \"" << Sha256(inputs0[2]) << "\"],\n"
         << "  \"input_post_sha256\": [\"" << Sha256(inputs1[0]) << "\", \""
         << Sha256(inputs1[1]) << "\", \"" << Sha256(inputs1[2]) << "\"],\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"external_model_files_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 7) return 2;
    try {
      const auto weights0 = ReadBytes(argv[2]);
      const auto bias0 = ReadBytes(argv[3]);
      if (weights0.size() != contract::kWeightBytes ||
          Sha256(weights0) != contract::kWeightSha256 ||
          bias0.size() != contract::kBiasBytes ||
          Sha256(bias0) != contract::kBiasSha256) {
        throw std::runtime_error("identity");
      }
      char *end = nullptr;
      const auto poison_value = strtoul(argv[6], &end, 16);
      if (end == argv[6] || *end || poison_value > 255)
        throw std::runtime_error("poison");
      const auto poison = static_cast<std::uint8_t>(poison_value);
      const std::filesystem::path output_root = argv[4];
      std::filesystem::create_directories(output_root);
      const std::array<std::vector<std::uint8_t>, 3> inputs0 = {
          contract::Input0(), contract::Input1(), contract::Input2()};

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError *error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL
                                fileURLWithPath:[NSString
                                                    stringWithUTF8String:argv[1]]]
                      error:&error];
      if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_custom_block0_bot1"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);

      id<MTLBuffer> weights =
          Guarded(device, &weights0, contract::kWeightBytes, 0);
      id<MTLBuffer> bias = Guarded(device, &bias0, contract::kBiasBytes, 0);
      id<MTLBuffer> input0 =
          Guarded(device, &inputs0[0], contract::kInputPlaneBytes, 0);
      id<MTLBuffer> input1 =
          Guarded(device, &inputs0[1], contract::kInputPlaneBytes, 0);
      id<MTLBuffer> input2 =
          Guarded(device, &inputs0[2], contract::kInputPlaneBytes, 0);
      id<MTLBuffer> output0 =
          Guarded(device, nullptr, contract::kOutput0Bytes, poison);
      id<MTLBuffer> output1 =
          Guarded(device, nullptr, contract::kOutput1Bytes, poison);
      id<MTLBuffer> output2 =
          Guarded(device, nullptr, contract::kOutput2Bytes, poison);
      id<MTLCommandBuffer> command_buffer =
          [[device newCommandQueue] commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [encoder setBuffer:bias offset:kGuardBytes atIndex:1];
      [encoder setBuffer:input0 offset:kGuardBytes atIndex:2];
      [encoder setBuffer:input1 offset:kGuardBytes atIndex:3];
      [encoder setBuffer:input2 offset:kGuardBytes atIndex:4];
      [encoder setBuffer:output0 offset:kGuardBytes atIndex:5];
      [encoder setBuffer:output1 offset:kGuardBytes atIndex:6];
      [encoder setBuffer:output2 offset:kGuardBytes atIndex:7];
      [encoder dispatchThreads:MTLSizeMake(320, 184, 8)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        throw std::runtime_error(
            command_buffer.error.localizedDescription.UTF8String);
      }

      const auto weights1 = Payload(weights, contract::kWeightBytes);
      const auto bias1 = Payload(bias, contract::kBiasBytes);
      const std::array<std::vector<std::uint8_t>, 3> inputs1 = {
          Payload(input0, contract::kInputPlaneBytes),
          Payload(input1, contract::kInputPlaneBytes),
          Payload(input2, contract::kInputPlaneBytes)};
      const auto result0 = Payload(output0, contract::kOutput0Bytes);
      const auto result1 = Payload(output1, contract::kOutput1Bytes);
      const auto result2 = Payload(output2, contract::kOutput2Bytes);
      std::vector<std::uint8_t> output;
      output.reserve(contract::kOutputBytes);
      output.insert(output.end(), result0.begin(), result0.end());
      output.insert(output.end(), result1.begin(), result1.end());
      output.insert(output.end(), result2.begin(), result2.end());
      const auto mismatches =
          Mismatches(weights0, weights1) + Mismatches(bias0, bias1) +
          Mismatches(inputs0[0], inputs1[0]) +
          Mismatches(inputs0[1], inputs1[1]) +
          Mismatches(inputs0[2], inputs1[2]);
      const auto canaries =
          CanaryMismatches(weights, contract::kWeightBytes) +
          CanaryMismatches(bias, contract::kBiasBytes) +
          CanaryMismatches(input0, contract::kInputPlaneBytes) +
          CanaryMismatches(input1, contract::kInputPlaneBytes) +
          CanaryMismatches(input2, contract::kInputPlaneBytes) +
          CanaryMismatches(output0, contract::kOutput0Bytes) +
          CanaryMismatches(output1, contract::kOutput1Bytes) +
          CanaryMismatches(output2, contract::kOutput2Bytes);
      Record(output_root / "run-receipt.json", argv[5], poison, weights0,
             bias0, inputs0, weights1, bias1, inputs1, output, mismatches,
             canaries, device.name.UTF8String);
      Write(output_root / "output.bin", output);
      const bool pass = mismatches == 0 && canaries == 0;
      std::cout << "output_sha256=" << Sha256(output) << "\n"
                << "result=" << (pass ? "pass" : "fail") << "\n";
      return pass ? 0 : 3;
    } catch (const std::exception &error) {
      std::cerr << "error=" << error.what() << "\n";
      return 1;
    }
  }
}
