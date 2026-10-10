#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"
#include "../fg_conv/conv_1x1_contract.h"
#include "../fg_conv/conv_3x3_contract.h"

namespace {

constexpr char kPoolModuleSha256[] =
    "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977";
constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

struct TensorSpec {
  const char* id;
  std::uint32_t launch;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

constexpr std::array<TensorSpec, 11> kTensors = {{
    {"l20", 20, 160, 96, 32}, {"l21", 21, 80, 48, 32},
    {"l22", 22, 80, 48, 64},  {"l23", 23, 40, 24, 64},
    {"l24", 24, 40, 24, 128}, {"l25", 25, 20, 12, 128},
    {"l26", 26, 20, 12, 128}, {"l27", 27, 10, 6, 128},
    {"l28", 28, 10, 6, 256},  {"l29", 29, 5, 3, 256},
    {"l30", 30, 5, 3, 512},
}};

struct Conv3x3Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t input_channels;
  std::uint32_t output_channels;
  std::uint32_t active_weight_half_count;
  std::uint32_t column_stage_schedule;
};

struct Conv1x1Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t input_channels;
  std::uint32_t output_channels;
  std::uint32_t active_weight_half_count;
  std::uint32_t apply_leaky_relu;
};

struct PoolingParameters {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
};

static_assert(sizeof(Conv3x3Parameters) == 24);
static_assert(sizeof(Conv1x1Parameters) == 24);
static_assert(sizeof(PoolingParameters) == 20);

std::size_t TensorBytes(const TensorSpec& tensor) {
  return std::size_t{tensor.width} * tensor.height * tensor.channels * 2u;
}

std::uint64_t ChainSplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::vector<std::uint8_t> GenerateEncoderInput() {
  constexpr std::array<std::uint16_t, 13> values = {
      0x0000, 0x8000, 0x2000, 0xa000, 0x2400, 0xa400, 0x2800,
      0xa800, 0x2c00, 0xac00, 0x3000, 0xb000, 0x3400,
  };
  std::vector<std::uint8_t> input(kInputBytes);
  constexpr std::uint64_t seed = 0x243F6A8885A308D3ull;
  constexpr std::uint64_t step = 0x9E3779B97F4A7C15ull;
  for (std::size_t index = 0; index < kInputBytes / 2u; ++index) {
    const auto mixed = ChainSplitMix64(seed + index * step);
    StoreHalf(&input, index * 2u, values[mixed % values.size()]);
  }
  return input;
}

std::vector<std::uint8_t> ReadEncoder(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size " + path.string());
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return result;
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) throw std::runtime_error("Metal function missing");
  NSError* error = nil;
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (result == nil) {
    throw std::runtime_error(
        std::string("pipeline creation failed: ") +
        (error != nil ? error.localizedDescription.UTF8String : "unknown"));
  }
  return result;
}

void Encode3x3(id<MTLCommandBuffer> command,
               id<MTLComputePipelineState> pipeline,
               const fg_conv_3x3::Layer& layer, id<MTLBuffer> weight,
               id<MTLBuffer> input, id<MTLBuffer> output) {
  const Conv3x3Parameters parameters = {
      layer.width,
      layer.height,
      layer.input_channels,
      layer.output_channels,
      layer.input_channels * layer.output_channels * 9u,
      layer.column_stage_schedule ? 1u : 0u,
  };
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weight offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(layer.width, layer.height,
                                       layer.output_channels)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
  [encoder endEncoding];
}

void Encode1x1(id<MTLCommandBuffer> command,
               id<MTLComputePipelineState> pipeline,
               const fg_conv_1x1::Layer& layer, id<MTLBuffer> weight,
               id<MTLBuffer> input, id<MTLBuffer> output) {
  const Conv1x1Parameters parameters = {
      layer.width,
      layer.height,
      layer.input_channels,
      layer.output_channels,
      layer.input_channels * layer.output_channels,
      layer.leaky_relu ? 1u : 0u,
  };
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weight offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(layer.width, layer.height,
                                       layer.output_channels)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
  [encoder endEncoding];
}

void EncodePool(id<MTLCommandBuffer> command,
                id<MTLComputePipelineState> pipeline, id<MTLBuffer> input,
                id<MTLBuffer> output, std::uint32_t input_width,
                std::uint32_t input_height, std::uint32_t channels) {
  const PoolingParameters parameters = {
      input_width, input_height, (input_width + 1u) / 2u,
      (input_height + 1u) / 2u, channels};
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
  [encoder dispatchThreadgroups:
               MTLSizeMake((parameters.output_width + 3u) / 4u,
                           (parameters.output_height + 3u) / 4u, 1)
             threadsPerThreadgroup:MTLSizeMake(4, 4, 16)];
  [encoder endEncoding];
}

void WriteEncoderReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::string& device_name,
    const std::vector<std::vector<std::uint8_t>>& weights,
    const std::vector<std::vector<std::uint8_t>>& weights_after,
    const std::vector<std::uint8_t>& input,
    const std::vector<std::uint8_t>& input_after,
    const std::vector<std::vector<std::uint8_t>>& tensors,
    const std::vector<std::size_t>& poison_collisions,
    std::size_t immutable_mismatches, std::size_t canary_mismatches) {
  const auto& l20 = fg_conv_3x3::FindLayer("l20");
  const auto& l22 = fg_conv_1x1::FindLayer("l22");
  const auto& l24 = fg_conv_3x3::FindLayer("l24");
  const auto& l26 = fg_conv_3x3::FindLayer("l26");
  const auto& l28 = fg_conv_3x3::FindLayer("l28");
  const auto& l30 = fg_conv_1x1::FindLayer("l30");
  const std::array<const fg_conv_3x3::Layer*, 4> three = {
      &l20, &l24, &l26, &l28};
  const std::array<const fg_conv_1x1::Layer*, 2> one = {&l22, &l30};
  const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write encoder receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.connected-encoder-l20-l30-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_launches\": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],\n"
         << "  \"pool_module\": {\"token\": 29, \"sha256\": \""
         << kPoolModuleSha256 << "\"},\n"
         << "  \"model_capture_manifest_sha256\": \""
         << kManifestSha256 << "\",\n"
         << "  \"weights\": [\n";
  std::size_t weight_index = 0;
  for (const auto* layer : three) {
    stream << "    {\"launch\": " << layer->launch
           << ", \"module\": " << layer->module_token
           << ", \"module_sha256\": \"" << layer->module_sha256
           << "\", \"source_token\": " << layer->weight_token
           << ", \"bytes\": " << layer->weight_bytes
           << ", \"pre_sha256\": \"" << Sha256(weights[weight_index])
           << "\", \"post_sha256\": \""
           << Sha256(weights_after[weight_index]) << "\"},\n";
    ++weight_index;
  }
  for (std::size_t index = 0; index < one.size(); ++index) {
    const auto* layer = one[index];
    stream << "    {\"launch\": " << layer->launch
           << ", \"module\": " << layer->module_token
           << ", \"module_sha256\": \"" << layer->module_sha256
           << "\", \"source_token\": " << layer->weight_token
           << ", \"bytes\": " << layer->weight_bytes
           << ", \"pre_sha256\": \"" << Sha256(weights[weight_index])
           << "\", \"post_sha256\": \""
           << Sha256(weights_after[weight_index]) << "\"}"
           << (index + 1u == one.size() ? "\n" : ",\n");
    ++weight_index;
  }
  stream << "  ],\n"
         << "  \"input_generator\": \"splitmix64-fg-connected-encoder-l20-l30-v1\",\n"
         << "  \"input_bytes\": " << input.size() << ",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after)
         << "\",\n"
         << "  \"tensors\": [\n";
  for (std::size_t index = 0; index < kTensors.size(); ++index) {
    stream << "    {\"id\": \"" << kTensors[index].id
           << "\", \"launch\": " << kTensors[index].launch
           << ", \"bytes\": " << tensors[index].size()
           << ", \"sha256\": \"" << Sha256(tensors[index])
           << "\", \"poison_collision_halfwords\": "
           << poison_collisions[index] << "}"
           << (index + 1u == kTensors.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"immutable_input_mismatched_bytes\": "
         << immutable_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_model_weights_opened\": true,\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short encoder receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-connected-encoder-l20-l30-mac <metallib> "
                   "<model-dir> <output-dir> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path library_path = argv[1];
      const std::filesystem::path model_root = argv[2];
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' ||
          parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      std::filesystem::create_directories(output_root);
      const auto input_before = GenerateEncoderInput();
      const auto& l20 = fg_conv_3x3::FindLayer("l20");
      const auto& l22 = fg_conv_1x1::FindLayer("l22");
      const auto& l24 = fg_conv_3x3::FindLayer("l24");
      const auto& l26 = fg_conv_3x3::FindLayer("l26");
      const auto& l28 = fg_conv_3x3::FindLayer("l28");
      const auto& l30 = fg_conv_1x1::FindLayer("l30");

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:
                                                    library_path.c_str()]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLComputePipelineState> three_pipeline =
          Pipeline(device, library, @"fg_conv_3x3");
      id<MTLComputePipelineState> one_pipeline =
          Pipeline(device, library, @"fg_conv_1x1");
      id<MTLComputePipelineState> pool_pipeline =
          Pipeline(device, library, @"fg_pooling");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue creation failed");

      const std::array<std::pair<std::uint32_t, std::size_t>, 6> weight_specs = {{
          {l20.weight_token, l20.weight_bytes}, {l24.weight_token, l24.weight_bytes},
          {l26.weight_token, l26.weight_bytes}, {l28.weight_token, l28.weight_bytes},
          {l22.weight_token, l22.weight_bytes}, {l30.weight_token, l30.weight_bytes},
      }};
      const std::array<const char*, 6> weight_hashes = {
          l20.weight_sha256, l24.weight_sha256, l26.weight_sha256,
          l28.weight_sha256, l22.weight_sha256, l30.weight_sha256};
      std::vector<std::vector<std::uint8_t>> weights_before;
      std::array<id<MTLBuffer>, 6> weights;
      for (std::size_t index = 0; index < weight_specs.size(); ++index) {
        std::ostringstream name;
        name << "default-token-" << std::setw(6) << std::setfill('0')
             << weight_specs[index].first << ".bin";
        auto bytes = ReadEncoder(model_root / name.str());
        if (bytes.size() != weight_specs[index].second ||
            Sha256(bytes) != weight_hashes[index]) {
          throw std::runtime_error("model-state identity mismatch: " + name.str());
        }
        weights[index] = Guarded(device, &bytes, bytes.size(), 0);
        weights_before.push_back(std::move(bytes));
      }
      id<MTLBuffer> input = Guarded(device, &input_before, kInputBytes, 0);
      std::array<id<MTLBuffer>, 11> tensors;
      for (std::size_t index = 0; index < tensors.size(); ++index) {
        tensors[index] =
            Guarded(device, nullptr, TensorBytes(kTensors[index]), poison);
      }

      id<MTLCommandBuffer> command = [queue commandBuffer];
      Encode3x3(command, three_pipeline, l20, weights[0], input, tensors[0]);
      EncodePool(command, pool_pipeline, tensors[0], tensors[1], 160, 96, 32);
      Encode1x1(command, one_pipeline, l22, weights[4], tensors[1], tensors[2]);
      EncodePool(command, pool_pipeline, tensors[2], tensors[3], 80, 48, 64);
      Encode3x3(command, three_pipeline, l24, weights[1], tensors[3], tensors[4]);
      EncodePool(command, pool_pipeline, tensors[4], tensors[5], 40, 24, 128);
      Encode3x3(command, three_pipeline, l26, weights[2], tensors[5], tensors[6]);
      EncodePool(command, pool_pipeline, tensors[6], tensors[7], 20, 12, 128);
      Encode3x3(command, three_pipeline, l28, weights[3], tensors[7], tensors[8]);
      EncodePool(command, pool_pipeline, tensors[8], tensors[9], 10, 6, 256);
      Encode1x1(command, one_pipeline, l30, weights[5], tensors[9], tensors[10]);
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(
            std::string("connected encoder command failed: ") +
            (command.error != nil
                 ? command.error.localizedDescription.UTF8String
                 : "unknown"));
      }

      const auto input_after = Payload(input, kInputBytes);
      std::size_t immutable_mismatches = Mismatches(input_before, input_after);
      std::size_t canary_mismatches = CanaryMismatches(input, kInputBytes);
      std::vector<std::vector<std::uint8_t>> weights_after;
      for (std::size_t index = 0; index < weights.size(); ++index) {
        auto after = Payload(weights[index], weight_specs[index].second);
        immutable_mismatches += Mismatches(weights_before[index], after);
        canary_mismatches +=
            CanaryMismatches(weights[index], weight_specs[index].second);
        Write(output_root /
                  ("weight-token-" + std::to_string(weight_specs[index].first) +
                   "-after.bin"),
              after);
        weights_after.push_back(std::move(after));
      }
      std::vector<std::vector<std::uint8_t>> tensor_bytes;
      std::vector<std::size_t> collisions;
      for (std::size_t index = 0; index < tensors.size(); ++index) {
        auto bytes = Payload(tensors[index], TensorBytes(kTensors[index]));
        canary_mismatches +=
            CanaryMismatches(tensors[index], TensorBytes(kTensors[index]));
        collisions.push_back(UntouchedHalfwords(bytes, poison));
        Write(output_root / (std::string(kTensors[index].id) + ".bin"), bytes);
        tensor_bytes.push_back(std::move(bytes));
      }
      Write(output_root / "input-after.bin", input_after);
      WriteEncoderReceipt(output_root / "run-receipt.json", run_id, poison,
                          device.name.UTF8String, weights_before, weights_after,
                          input_before, input_after, tensor_bytes, collisions,
                          immutable_mismatches, canary_mismatches);
      const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_connected_encoder_l20_l30_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_sha256=" << Sha256(tensor_bytes.back()) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
