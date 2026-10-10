#define FG_CONV_L20_NO_MAIN
#include "macos_probe.mm"

namespace {

constexpr std::size_t kL51InputBytes = 160u * 96u * 32u * 2u;
constexpr std::size_t kL51OutputBytes = 160u * 96u * 16u * 2u;
constexpr std::size_t kL51WeightBytes = 2560;
constexpr char kL51ModuleSha256[] =
    "27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2";
constexpr char kL51WeightSha256[] =
    "477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e";
constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

struct ConvL51Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t output_width;
  std::uint32_t output_height;
};

std::vector<std::uint8_t> ReadL51(const std::filesystem::path& path) {
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

std::uint64_t L51SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::vector<std::uint8_t> GenerateL51Input() {
  constexpr std::array<std::uint16_t, 9> values = {
      0x0000, 0x2c00, 0xac00, 0x3000, 0xb000,
      0x3400, 0xb400, 0x3800, 0xb800,
  };
  std::vector<std::uint8_t> input(kL51InputBytes);
  constexpr std::uint64_t seed = 0x94D049BB133111EBull;
  constexpr std::uint64_t step = 0xD1B54A32D192ED03ull;
  for (std::size_t index = 0; index < input.size() / 2; ++index) {
    StoreHalf(&input, index * 2,
              values[L51SplitMix64(seed + index * step) % values.size()]);
  }
  return input;
}

void WriteL51Receipt(const std::filesystem::path& path,
                     const std::string& run_id, std::uint8_t poison,
                     const std::vector<std::uint8_t>& weights_before,
                     const std::vector<std::uint8_t>& input_before,
                     const std::vector<std::uint8_t>& weights_after,
                     const std::vector<std::uint8_t>& input_after,
                     const std::vector<std::uint8_t>& output,
                     std::size_t input_mismatches,
                     std::size_t canary_mismatches,
                     std::size_t poison_collisions,
                     const std::string& device_name) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L51 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-l51-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 52,\n"
         << "  \"module_bytes\": 3912,\n"
         << "  \"module_sha256\": \"" << kL51ModuleSha256 << "\",\n"
         << "  \"metal_function\": \"fg_conv_l51\",\n"
         << "  \"trace_launch_token\": \"launch-000051\",\n"
         << "  \"selected_rtx_grid\": [10, 24, 1],\n"
         << "  \"selected_rtx_block\": [32, 4, 1],\n"
         << "  \"metal_dispatch_threads\": [160, 96, 16],\n"
         << "  \"metal_threads_per_threadgroup\": [8, 4, 1],\n"
         << "  \"packed_parameter_bytes\": 40,\n"
         << "  \"abi_scalars\": [160, 96, 160, 96],\n"
         << "  \"input_shape_nhwc\": [1, 96, 160, 32],\n"
         << "  \"output_shape_nhwc\": [1, 96, 160, 16],\n"
         << "  \"weight_source_token\": 95,\n"
         << "  \"weight_capture_point\": \"post_evaluate_0_fence_pre_evaluate_1\",\n"
         << "  \"weight_capture_manifest_sha256\": \""
         << kManifestSha256 << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights_before)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(weights_after)
         << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-conv-l51-live-v1\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"per_run_poison_collision_halfwords\": "
         << poison_collisions << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_weight_file_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-conv-l51-mac <metallib> <weights> <output-dir> "
                   "<run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const auto weights_before = ReadL51(argv[2]);
      if (weights_before.size() != kL51WeightBytes ||
          Sha256(weights_before) != kL51WeightSha256) {
        throw std::runtime_error("L51 weight identity mismatch");
      }
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' ||
          parsed_poison > 255) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const std::filesystem::path output_directory = argv[3];
      std::filesystem::create_directories(output_directory);
      const auto input_before = GenerateL51Input();
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function = [library newFunctionWithName:@"fg_conv_l51"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLBuffer> weights =
          Guarded(device, &weights_before, kL51WeightBytes, 0);
      id<MTLBuffer> input =
          Guarded(device, &input_before, kL51InputBytes, 0);
      id<MTLBuffer> output = Guarded(device, nullptr, kL51OutputBytes, poison);
      const ConvL51Parameters parameters{160, 96, 160, 96};
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [encoder setBuffer:input offset:kGuardBytes atIndex:1];
      [encoder setBuffer:output offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(160, 96, 16)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      const auto weights_after = Payload(weights, kL51WeightBytes);
      const auto input_after = Payload(input, kL51InputBytes);
      const auto actual_output = Payload(output, kL51OutputBytes);
      const std::size_t mismatches =
          Mismatches(weights_before, weights_after) +
          Mismatches(input_before, input_after);
      const std::size_t canaries =
          CanaryMismatches(weights, kL51WeightBytes) +
          CanaryMismatches(input, kL51InputBytes) +
          CanaryMismatches(output, kL51OutputBytes);
      const std::size_t collisions = UntouchedHalfwords(actual_output, poison);
      Write(output_directory / "input-after.bin", input_after);
      Write(output_directory / "output.bin", actual_output);
      WriteL51Receipt(output_directory / "run-receipt.json", argv[4], poison,
                      weights_before, input_before, weights_after, input_after,
                      actual_output, mismatches, canaries, collisions,
                      device.name.UTF8String);
      const bool pass = mismatches == 0 && canaries == 0;
      std::cout << "probe=fg_conv_l51_mac\noutput_sha256="
                << Sha256(actual_output) << "\nresult="
                << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
