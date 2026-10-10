#define FG_CONV_L20_NO_MAIN
#include "macos_probe.mm"
#include "conv_1x1_contract.h"

namespace {

constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

struct Conv1x1Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t input_channels;
  std::uint32_t output_channels;
  std::uint32_t active_weight_half_count;
  std::uint32_t apply_leaky_relu;
};
static_assert(sizeof(Conv1x1Parameters) == 24);

std::vector<std::uint8_t> ReadFamily(const std::filesystem::path& path) {
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

void WriteFamilyReceipt(
    const std::filesystem::path& path, const fg_conv_1x1::Layer& layer,
    const std::string& run_id, std::uint8_t poison,
    const std::vector<std::uint8_t>& weights_before,
    const std::vector<std::uint8_t>& input_before,
    const std::vector<std::uint8_t>& weights_after,
    const std::vector<std::uint8_t>& input_after,
    const std::vector<std::uint8_t>& output,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t poison_collisions, const std::string& device_name) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write family receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-1x1-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"layer_id\": \"" << layer.id << "\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": " << layer.module_token << ",\n"
         << "  \"module_bytes\": " << layer.module_bytes << ",\n"
         << "  \"module_sha256\": \"" << layer.module_sha256 << "\",\n"
         << "  \"metal_function\": \"fg_conv_1x1\",\n"
         << "  \"trace_launch_token\": \"launch-" << std::setw(6)
         << std::setfill('0') << layer.launch << "\",\n"
         << std::setfill(' ')
         << "  \"selected_rtx_grid\": [" << layer.grid[0] << ", "
         << layer.grid[1] << ", " << layer.grid[2] << "],\n"
         << "  \"selected_rtx_block\": [" << layer.block[0] << ", "
         << layer.block[1] << ", " << layer.block[2] << "],\n"
         << "  \"metal_dispatch_threads\": [" << layer.width << ", "
         << layer.height << ", " << layer.output_channels << "],\n"
         << "  \"metal_threads_per_threadgroup\": [8, 4, 1],\n"
         << "  \"packed_parameter_bytes\": 40,\n"
         << "  \"abi_scalars\": [" << layer.width << ", " << layer.height
         << ", " << layer.width << ", " << layer.height << "],\n"
         << "  \"input_shape_nhwc\": [1, " << layer.height << ", "
         << layer.width << ", " << layer.input_channels << "],\n"
         << "  \"output_shape_nhwc\": [1, " << layer.height << ", "
         << layer.width << ", " << layer.output_channels << "],\n"
         << "  \"leaky_relu\": " << (layer.leaky_relu ? "true" : "false")
         << ",\n"
         << "  \"weight_source_token\": " << layer.weight_token << ",\n"
         << "  \"weight_capture_point\": \"post_evaluate_0_fence_pre_evaluate_1\",\n"
         << "  \"weight_capture_manifest_sha256\": \""
         << kManifestSha256 << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights_before)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(weights_after)
         << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-conv-1x1-family-v1\",\n"
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
    if (argc != 7) {
      std::cerr << "usage=fg-conv-1x1-mac <layer-id> <metallib> <weights> "
                   "<output-dir> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const auto& layer = fg_conv_1x1::FindLayer(argv[1]);
      const auto weights_before = ReadFamily(argv[3]);
      if (weights_before.size() != layer.weight_bytes ||
          Sha256(weights_before) != layer.weight_sha256) {
        throw std::runtime_error("live weight identity mismatch");
      }
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[6], &poison_end, 16);
      if (poison_end == argv[6] || *poison_end != '\0' ||
          parsed_poison > 255) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const std::filesystem::path output_directory = argv[4];
      std::filesystem::create_directories(output_directory);
      const auto input_before = fg_conv_1x1::GenerateInput(layer);
      const std::size_t output_bytes = fg_conv_1x1::OutputBytes(layer);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[2]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function = [library newFunctionWithName:@"fg_conv_1x1"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLBuffer> weights =
          Guarded(device, &weights_before, layer.weight_bytes, 0);
      id<MTLBuffer> input =
          Guarded(device, &input_before, input_before.size(), 0);
      id<MTLBuffer> output = Guarded(device, nullptr, output_bytes, poison);
      const Conv1x1Parameters parameters{
          layer.width, layer.height, layer.input_channels,
          layer.output_channels,
          layer.input_channels * layer.output_channels,
          layer.leaky_relu ? 1u : 0u,
      };
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [encoder setBuffer:input offset:kGuardBytes atIndex:1];
      [encoder setBuffer:output offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(layer.width, layer.height,
                                           layer.output_channels)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      const auto weights_after = Payload(weights, layer.weight_bytes);
      const auto input_after = Payload(input, input_before.size());
      const auto actual_output = Payload(output, output_bytes);
      const std::size_t mismatches =
          Mismatches(weights_before, weights_after) +
          Mismatches(input_before, input_after);
      const std::size_t canaries =
          CanaryMismatches(weights, layer.weight_bytes) +
          CanaryMismatches(input, input_before.size()) +
          CanaryMismatches(output, output_bytes);
      const std::size_t collisions = UntouchedHalfwords(actual_output, poison);
      Write(output_directory / "input-after.bin", input_after);
      Write(output_directory / "output.bin", actual_output);
      WriteFamilyReceipt(output_directory / "run-receipt.json", layer,
                         argv[5], poison, weights_before, input_before,
                         weights_after, input_after, actual_output,
                         mismatches, canaries, collisions,
                         device.name.UTF8String);
      const bool pass = mismatches == 0 && canaries == 0;
      std::cout << "probe=fg_conv_1x1_mac\nlayer_id=" << layer.id
                << "\noutput_sha256=" << Sha256(actual_output)
                << "\nresult=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
