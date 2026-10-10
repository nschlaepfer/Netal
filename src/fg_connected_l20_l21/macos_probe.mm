#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

namespace {

constexpr std::uint32_t kPoolWidth = 80;
constexpr std::uint32_t kPoolHeight = 48;
constexpr std::uint32_t kPoolChannels = 32;
constexpr std::size_t kPoolOutputBytes =
    std::size_t{kPoolWidth} * kPoolHeight * kPoolChannels * 2u;
constexpr char kConvModuleSha256[] =
    "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d";
constexpr char kPoolModuleSha256[] =
    "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977";
constexpr char kLiveWeightSha256[] =
    "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9";
constexpr char kLiveManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

struct PoolingParameters {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
};
static_assert(sizeof(PoolingParameters) == 20);

std::uint64_t ChainSplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::vector<std::uint8_t> GenerateChainInput() {
  constexpr std::array<std::uint16_t, 11> values = {
      0x0000, 0x8000, 0x2400, 0xa400, 0x2c00, 0xac00,
      0x3000, 0xb000, 0x3400, 0xb400, 0x3800,
  };
  std::vector<std::uint8_t> input(kInputBytes);
  constexpr std::uint64_t seed = 0xA0761D6478BD642Full;
  constexpr std::uint64_t step = 0xE7037ED1A0B428DBull;
  for (std::size_t index = 0; index < kInputBytes / 2u; ++index) {
    StoreHalf(&input, index * 2u,
              values[ChainSplitMix64(seed + index * step) % values.size()]);
  }
  return input;
}

std::vector<std::uint8_t> ReadChain(const std::filesystem::path& path) {
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

id<MTLComputePipelineState> ChainPipeline(id<MTLDevice> device,
                                           id<MTLLibrary> library,
                                           NSString* function_name) {
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil) throw std::runtime_error("Metal function missing");
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (pipeline == nil) {
    throw std::runtime_error(
        std::string("pipeline creation failed: ") +
        (error != nil ? error.localizedDescription.UTF8String : "unknown"));
  }
  return pipeline;
}

void WriteChainReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::string& device_name,
    const std::vector<std::uint8_t>& weights,
    const std::vector<std::uint8_t>& expected_input,
    const std::vector<std::uint8_t>& actual_weights,
    const std::vector<std::uint8_t>& actual_input,
    const std::vector<std::uint8_t>& intermediate,
    const std::vector<std::uint8_t>& output, std::size_t immutable_mismatches,
    std::size_t canary_mismatches, std::size_t intermediate_poison_collisions,
    std::size_t output_poison_collisions) {
  const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write chain receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.connected-l20-l21-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_launches\": [\n"
         << "    {\"launch\": 20, \"module\": 28, \"module_sha256\": \""
         << kConvModuleSha256
         << "\", \"function\": \"fg_conv_l20\", \"selected_grid\": [10, 8, 1], \"selected_block\": [32, 4, 1], \"metal_threads\": [160, 96, 32]},\n"
         << "    {\"launch\": 21, \"module\": 29, \"module_sha256\": \""
         << kPoolModuleSha256
         << "\", \"function\": \"fg_pooling\", \"selected_grid\": [20, 12, 1], \"selected_block\": [4, 4, 16]}\n"
         << "  ],\n"
         << "  \"weight_source_token\": 63,\n"
         << "  \"weight_capture_point\": \"post_evaluate_0_fence_pre_evaluate_1\",\n"
         << "  \"weight_capture_manifest_sha256\": \""
         << kLiveManifestSha256 << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights) << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(actual_weights)
         << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-connected-l20-l21-v1\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(expected_input)
         << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(actual_input)
         << "\",\n"
         << "  \"intermediate_bytes\": " << intermediate.size() << ",\n"
         << "  \"intermediate_sha256\": \"" << Sha256(intermediate)
         << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"immutable_input_mismatched_bytes\": "
         << immutable_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"intermediate_poison_collision_halfwords\": "
         << intermediate_poison_collisions << ",\n"
         << "  \"output_poison_collision_halfwords\": "
         << output_poison_collisions << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_model_weight_opened\": true,\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short chain receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-connected-l20-l21-mac <metallib> <weights.bin> "
                   "<output-dir> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path library_path = argv[1];
      const std::filesystem::path weight_path = argv[2];
      const std::filesystem::path output_directory = argv[3];
      const std::string run_id = argv[4];
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' ||
          parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      std::filesystem::create_directories(output_directory);
      const auto expected_weights = ReadChain(weight_path);
      const auto expected_input = GenerateChainInput();
      if (expected_weights.size() != kWeightBytes ||
          Sha256(expected_weights) != kLiveWeightSha256) {
        throw std::runtime_error("live model weight provenance mismatch");
      }

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:
                                                    library_path.c_str()]]
                     error:&error];
      if (library == nil) {
        throw std::runtime_error(
            std::string("metallib load failed: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLComputePipelineState> conv_pipeline =
          ChainPipeline(device, library, @"fg_conv_l20");
      id<MTLComputePipelineState> pool_pipeline =
          ChainPipeline(device, library, @"fg_pooling");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue creation failed");

      id<MTLBuffer> weights =
          Guarded(device, &expected_weights, kWeightBytes, 0);
      id<MTLBuffer> input =
          Guarded(device, &expected_input, kInputBytes, 0);
      id<MTLBuffer> intermediate =
          Guarded(device, nullptr, kOutputBytes, poison);
      id<MTLBuffer> output =
          Guarded(device, nullptr, kPoolOutputBytes, poison);
      const ConvL20Parameters conv_parameters{kWidth, kHeight, 162, 98,
                                               kWidth, kHeight};
      const PoolingParameters pool_parameters{kWidth, kHeight, kPoolWidth,
                                               kPoolHeight, kPoolChannels};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> conv_encoder =
          [command computeCommandEncoder];
      [conv_encoder setComputePipelineState:conv_pipeline];
      [conv_encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [conv_encoder setBuffer:input offset:kGuardBytes atIndex:1];
      [conv_encoder setBuffer:intermediate offset:kGuardBytes atIndex:2];
      [conv_encoder setBytes:&conv_parameters
                      length:sizeof(conv_parameters)
                     atIndex:3];
      [conv_encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, kOutputChannels)
             threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [conv_encoder endEncoding];

      id<MTLComputeCommandEncoder> pool_encoder =
          [command computeCommandEncoder];
      [pool_encoder setComputePipelineState:pool_pipeline];
      [pool_encoder setBuffer:output offset:kGuardBytes atIndex:0];
      [pool_encoder setBuffer:intermediate offset:kGuardBytes atIndex:1];
      [pool_encoder setBytes:&pool_parameters
                      length:sizeof(pool_parameters)
                     atIndex:2];
      [pool_encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 4, 16)];
      [pool_encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(
            std::string("connected Metal command failed: ") +
            (command.error != nil
                 ? command.error.localizedDescription.UTF8String
                 : "unknown"));
      }

      const auto actual_weights = Payload(weights, kWeightBytes);
      const auto actual_input = Payload(input, kInputBytes);
      const auto actual_intermediate = Payload(intermediate, kOutputBytes);
      const auto actual_output = Payload(output, kPoolOutputBytes);
      const std::size_t immutable_mismatches =
          Mismatches(actual_weights, expected_weights) +
          Mismatches(actual_input, expected_input);
      const std::size_t canary_mismatches =
          CanaryMismatches(weights, kWeightBytes) +
          CanaryMismatches(input, kInputBytes) +
          CanaryMismatches(intermediate, kOutputBytes) +
          CanaryMismatches(output, kPoolOutputBytes);
      const std::size_t intermediate_collisions =
          UntouchedHalfwords(actual_intermediate, poison);
      const std::size_t output_collisions =
          UntouchedHalfwords(actual_output, poison);
      Write(output_directory / "input-after.bin", actual_input);
      Write(output_directory / "weights-after.bin", actual_weights);
      Write(output_directory / "intermediate.bin", actual_intermediate);
      Write(output_directory / "output.bin", actual_output);
      WriteChainReceipt(output_directory / "run-receipt.json", run_id, poison,
                        device.name.UTF8String, expected_weights,
                        expected_input, actual_weights, actual_input,
                        actual_intermediate, actual_output,
                        immutable_mismatches, canary_mismatches,
                        intermediate_collisions, output_collisions);
      const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_connected_l20_l21_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "intermediate_sha256=" << Sha256(actual_intermediate)
                << '\n'
                << "output_sha256=" << Sha256(actual_output) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
