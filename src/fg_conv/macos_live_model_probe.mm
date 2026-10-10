#define FG_CONV_L20_NO_MAIN
#include "macos_probe.mm"

namespace {

constexpr char kLiveWeightSha256[] =
    "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9";
constexpr char kLiveManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

std::vector<std::uint8_t> ReadLive(const std::filesystem::path& path) {
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

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::vector<std::uint8_t> GenerateLiveInput() {
  constexpr std::array<std::uint16_t, 7> values = {
      0x0000, 0x3000, 0xb000, 0x3400, 0xb400, 0x3800, 0xb800,
  };
  std::vector<std::uint8_t> input(kInputBytes);
  constexpr std::uint64_t seed = 0xD1B54A32D192ED03ull;
  constexpr std::uint64_t step = 0x9E3779B97F4A7C15ull;
  for (std::size_t index = 0; index < kInputBytes / 2; ++index) {
    StoreHalf(&input, index * 2,
              values[SplitMix64(seed + index * step) % values.size()]);
  }
  return input;
}

void WriteLiveReceipt(const std::filesystem::path& path,
                      const std::string& run_id, std::uint8_t poison,
                      const std::vector<std::uint8_t>& expected_weights,
                      const std::vector<std::uint8_t>& expected_input,
                      const std::vector<std::uint8_t>& actual_weights,
                      const std::vector<std::uint8_t>& actual_input,
                      const std::vector<std::uint8_t>& output,
                      std::size_t input_mismatches,
                      std::size_t canary_mismatches,
                      std::size_t poison_collisions,
                      const std::string& device_name) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write live receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-l20-live-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"metal_function\": \"fg_conv_l20\",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_rtx_grid\": [10, 8, 1],\n"
         << "  \"selected_rtx_block\": [32, 4, 1],\n"
         << "  \"metal_dispatch_threads\": [160, 96, 32],\n"
         << "  \"metal_threads_per_threadgroup\": [8, 4, 1],\n"
         << "  \"abi_scalars\": [160, 96, 162, 98, 160, 96],\n"
         << "  \"weight_source_token\": 63,\n"
         << "  \"weight_capture_point\": \"post_evaluate_0_fence_pre_evaluate_1\",\n"
         << "  \"weight_capture_manifest_sha256\": \""
         << kLiveManifestSha256 << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(expected_weights)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(actual_weights)
         << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-conv-l20-live-v1\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(expected_input)
         << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(actual_input)
         << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"per_run_poison_collision_halfwords\": "
         << poison_collisions << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_weight_file_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-conv-l20-live-mac <metallib> <weights.bin> "
                   "<output-dir> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path weight_path = argv[2];
      const std::filesystem::path output_directory = argv[3];
      const std::string run_id = argv[4];
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' ||
          parsed_poison > 255) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      std::filesystem::create_directories(output_directory);
      const auto expected_weights = ReadLive(weight_path);
      if (expected_weights.size() != kWeightBytes ||
          Sha256(expected_weights) != kLiveWeightSha256) {
        throw std::runtime_error("live weight provenance mismatch");
      }
      const auto expected_input = GenerateLiveInput();

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function = [library newFunctionWithName:@"fg_conv_l20"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLBuffer> weights =
          Guarded(device, &expected_weights, kWeightBytes, 0);
      id<MTLBuffer> input =
          Guarded(device, &expected_input, kInputBytes, 0);
      id<MTLBuffer> output = Guarded(device, nullptr, kOutputBytes, poison);
      const ConvL20Parameters parameters{kWidth, kHeight, 162, 98,
                                         kWidth, kHeight};
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [encoder setBuffer:input offset:kGuardBytes atIndex:1];
      [encoder setBuffer:output offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, kOutputChannels)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      const auto actual_weights = Payload(weights, kWeightBytes);
      const auto actual_input = Payload(input, kInputBytes);
      const auto actual_output = Payload(output, kOutputBytes);
      const std::size_t input_mismatches =
          Mismatches(actual_weights, expected_weights) +
          Mismatches(actual_input, expected_input);
      const std::size_t canary_mismatches =
          CanaryMismatches(weights, kWeightBytes) +
          CanaryMismatches(input, kInputBytes) +
          CanaryMismatches(output, kOutputBytes);
      const std::size_t collisions = UntouchedHalfwords(actual_output, poison);
      Write(output_directory / "input-after.bin", actual_input);
      Write(output_directory / "output.bin", actual_output);
      WriteLiveReceipt(output_directory / "run-receipt.json", run_id,
                       poison, expected_weights, expected_input,
                       actual_weights, actual_input, actual_output,
                       input_mismatches, canary_mismatches, collisions,
                       device.name.UTF8String);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_conv_l20_live_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_sha256=" << Sha256(actual_output) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
