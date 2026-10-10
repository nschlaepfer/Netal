#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"
#include "../fg_conv/conv_1x1_contract.h"
#include "../fg_conv/conv_3x3_contract.h"

namespace {

constexpr char kPoolModuleSha256[] =
    "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977";
constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

struct WeightSpec {
  std::uint32_t launch;
  std::uint32_t module;
  const char* module_sha256;
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
};

constexpr std::array<WeightSpec, 17> kWeights = {{
    {20,28,"7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d",63,56320,"44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"},
    {22,30,"31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785",65,10240,"eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60"},
    {24,31,"e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63",67,299008,"4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55"},
    {26,32,"c74961f6f71f4e7bac5b216cc704b10a0fdd6f0987352fdd315f371467ca7187",69,593920,"c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404"},
    {28,33,"6f2ffbe7faacde8119719d92dacde40f21ddd562878381b81654dad7a66775dd",71,1187840,"fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93"},
    {30,34,"2f490857c6f7aa22f2435057706ec10b28b6b8d3081bd967ebe1d61550ac6419",73,540672,"e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9"},
    {31,35,"331853771cae4aa041038d8e9c74dcb1f8d0d38b4d5603e2e8d079ad1e44fd04",75,532480,"9f20d6589cc9269ad020233e68196d66eba0e8a76497374cf6d445385f94dbc4"},
    {34,38,"8e7abed27b7eae0dd274b2409a13d6dde98277d9e743f0e36a1042022a16ceda",77,270336,"9e5da09b93650c74e89d5262aa8a9a6a3dcbd0d753dc8c9d9aed39731e39b2a0"},
    {35,39,"939d5d9cf7012fb3ed3ce4e4792ccf5c6befec165ae7f7a0d3704cb1103c3030",79,135168,"3969384493583e98dcf357d3046a10a5b836585261f3e584605bf11e23fb8d46"},
    {38,42,"93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc",81,69632,"a0c6690faee2838521c6ad0f9445425d639446c0ef1b7fed9b43d21e7a8ec650"},
    {39,42,"93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc",83,69632,"3a13970b482f1f78ede1bfe531956c203f08f99077c81bda2d1e4d0f453cf9a0"},
    {42,43,"53fdf8d5404c93f2fe64582f271192df3b24c2ef08d9564ffeb9d7990485db0e",85,69632,"47361a699fa6514332dc560cf2416872256e61d1b36b1ca29df323a53ee2be77"},
    {43,44,"7113dfaae4945541276b3e662cba366f9a3fd9666db7a338efa867cfd7f46e31",87,34816,"327bd9b02da4aaac8b251320f60b4aeccb0df7df84bba44d7f1f6a9fdf62c934"},
    {46,47,"8a3efc1e9166ed3ee4a54aea535910c0562b6753b2d0daae0402e1ed36ff19dc",89,18432,"e24f67110bab401ab7f70a861f3f09d2cd3cd6abf8d1aa158ba3300efb64797e"},
    {47,48,"11906664ac548c61262a12b0950bf135926cf269de2d7691d0edfbe4a95e63dd",91,9216,"325d7dfac5620a89edc1d2315731414cf755c50506ea5758a8c3cf563649577c"},
    {50,51,"304d0c0e58958fa7eab6a82fd046ff689c51a986aff88a069f6f509c434a5b96",93,5120,"bb98eb15be7d255a15a46994589705560e0691e273e286a36a68fd679a1890a9"},
    {51,52,"27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2",95,2560,"477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e"},
}};

struct TensorSpec {
  const char* id;
  std::uint32_t launch;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

constexpr std::array<TensorSpec, 32> kTensors = {{
    {"l20", 20, 160, 96, 32}, {"l21", 21, 80, 48, 32},
    {"l22", 22, 80, 48, 64},  {"l23", 23, 40, 24, 64},
    {"l24", 24, 40, 24, 128}, {"l25", 25, 20, 12, 128},
    {"l26", 26, 20, 12, 128}, {"l27", 27, 10, 6, 128},
    {"l28", 28, 10, 6, 256},  {"l29", 29, 5, 3, 256},
    {"l30", 30, 5, 3, 512},
    {"l31", 31, 5, 3, 256}, {"l32", 32, 10, 6, 256},
    {"l33", 33, 10, 6, 256}, {"l34", 34, 10, 6, 256},
    {"l35", 35, 10, 6, 128}, {"l36", 36, 20, 12, 128},
    {"l37", 37, 20, 12, 128}, {"l38", 38, 20, 12, 128},
    {"l39", 39, 20, 12, 128}, {"l40", 40, 40, 24, 128},
    {"l41", 41, 40, 24, 128}, {"l42", 42, 40, 24, 128},
    {"l43", 43, 40, 24, 64}, {"l44", 44, 80, 48, 64},
    {"l45", 45, 80, 48, 64}, {"l46", 46, 80, 48, 64},
    {"l47", 47, 80, 48, 32}, {"l48", 48, 160, 96, 32},
    {"l49", 49, 160, 96, 32}, {"l50", 50, 160, 96, 32},
    {"l51", 51, 160, 96, 16},
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

struct UpscaleParameters {
  std::uint32_t input_height;
  std::uint32_t input_width;
  std::uint32_t output_height;
  std::uint32_t output_width;
  std::uint32_t physical_output_height;
  std::uint32_t physical_output_width;
  std::uint32_t channels;
};

struct ElementwiseParameters {
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
  std::uint32_t input_a_width;
  std::uint32_t input_a_height;
  std::uint32_t input_b_width;
  std::uint32_t input_b_height;
  std::uint32_t vector_thread_stride;
};

static_assert(sizeof(Conv3x3Parameters) == 24);
static_assert(sizeof(Conv1x1Parameters) == 24);
static_assert(sizeof(PoolingParameters) == 20);
static_assert(sizeof(UpscaleParameters) == 28);
static_assert(sizeof(ElementwiseParameters) == 32);

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

void EncodeUpscale(id<MTLCommandBuffer> command,
                   id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> input, id<MTLBuffer> output,
                   std::uint32_t input_width, std::uint32_t input_height,
                   std::uint32_t output_width, std::uint32_t output_height,
                   std::uint32_t channels) {
  const UpscaleParameters parameters = {
      input_height, input_width, output_height, output_width,
      output_height, output_width, channels};
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:kGuardBytes atIndex:0];
  [encoder setBuffer:output offset:kGuardBytes atIndex:1];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(channels / 2u, output_width,
                                       output_height)
          threadsPerThreadgroup:MTLSizeMake(16, 4, 1)];
  [encoder endEncoding];
}

void EncodeAdd(id<MTLCommandBuffer> command,
               id<MTLComputePipelineState> pipeline, id<MTLBuffer> input_a,
               id<MTLBuffer> input_b, id<MTLBuffer> output,
               std::uint32_t width, std::uint32_t height,
               std::uint32_t channels, std::uint32_t grid_x,
               std::uint32_t grid_y, std::uint32_t block_z) {
  const ElementwiseParameters parameters = {
      width, height, channels, width, height, width, height, block_z};
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input_a offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input_b offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
             threadsPerThreadgroup:MTLSizeMake(4, 4, block_z)];
  [encoder endEncoding];
}

void WriteUnetReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::string& device_name,
    const std::vector<std::vector<std::uint8_t>>& weights,
    const std::vector<std::vector<std::uint8_t>>& weights_after,
    const std::vector<std::uint8_t>& input,
    const std::vector<std::uint8_t>& input_after,
    const std::vector<std::vector<std::uint8_t>>& tensors,
    const std::vector<std::size_t>& poison_collisions,
    std::size_t immutable_mismatches, std::size_t canary_mismatches) {
  const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write U-Net receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.connected-unet-l20-l51-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_launch_first\": 20,\n"
         << "  \"selected_launch_last\": 51,\n"
         << "  \"selected_launch_count\": 32,\n"
         << "  \"pool_module\": {\"token\": 29, \"sha256\": \""
         << kPoolModuleSha256 << "\"},\n"
         << "  \"model_capture_manifest_sha256\": \""
         << kManifestSha256 << "\",\n"
         << "  \"weights\": [\n";
  for (std::size_t index = 0; index < kWeights.size(); ++index) {
    const auto& weight = kWeights[index];
    stream << "    {\"launch\": " << weight.launch
           << ", \"module\": " << weight.module
           << ", \"module_sha256\": \"" << weight.module_sha256
           << "\", \"source_token\": " << weight.token
           << ", \"bytes\": " << weight.bytes
           << ", \"pre_sha256\": \"" << Sha256(weights[index])
           << "\", \"post_sha256\": \""
           << Sha256(weights_after[index]) << "\"}"
           << (index + 1u == kWeights.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"input_generator\": \"splitmix64-fg-connected-unet-l20-l51-v1\",\n"
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
  if (!stream) throw std::runtime_error("short U-Net receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-connected-unet-l20-l51-mac <metallib> "
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
      const auto& l31 = fg_conv_1x1::FindLayer("l31");
      const auto& l34 = fg_conv_1x1::FindLayer("l34");
      const auto& l35 = fg_conv_1x1::FindLayer("l35");
      const auto& l38 = fg_conv_1x1::FindLayer("l38");
      const auto& l39 = fg_conv_1x1::FindLayer("l39");
      const auto& l42 = fg_conv_1x1::FindLayer("l42");
      const auto& l43 = fg_conv_1x1::FindLayer("l43");
      const auto& l46 = fg_conv_1x1::FindLayer("l46");
      const auto& l47 = fg_conv_1x1::FindLayer("l47");
      const auto& l50 = fg_conv_1x1::FindLayer("l50");
      const auto& l51 = fg_conv_1x1::FindLayer("l51");

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
      id<MTLComputePipelineState> upscale_pipeline =
          Pipeline(device, library, @"fg_upscale");
      id<MTLComputePipelineState> add_pipeline =
          Pipeline(device, library, @"fg_elementwise_add_family");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue creation failed");

      std::vector<std::vector<std::uint8_t>> weights_before;
      std::array<id<MTLBuffer>, 17> weights;
      for (std::size_t index = 0; index < kWeights.size(); ++index) {
        std::ostringstream name;
        name << "default-token-" << std::setw(6) << std::setfill('0')
             << kWeights[index].token << ".bin";
        auto bytes = ReadEncoder(model_root / name.str());
        if (bytes.size() != kWeights[index].bytes ||
            Sha256(bytes) != kWeights[index].sha256) {
          throw std::runtime_error("model-state identity mismatch: " + name.str());
        }
        weights[index] = Guarded(device, &bytes, bytes.size(), 0);
        weights_before.push_back(std::move(bytes));
      }
      id<MTLBuffer> input = Guarded(device, &input_before, kInputBytes, 0);
      std::array<id<MTLBuffer>, 32> tensors;
      for (std::size_t index = 0; index < tensors.size(); ++index) {
        tensors[index] =
            Guarded(device, nullptr, TensorBytes(kTensors[index]), poison);
      }

      id<MTLCommandBuffer> command = [queue commandBuffer];
      Encode3x3(command, three_pipeline, l20, weights[0], input, tensors[0]);
      EncodePool(command, pool_pipeline, tensors[0], tensors[1], 160, 96, 32);
      Encode1x1(command, one_pipeline, l22, weights[1], tensors[1], tensors[2]);
      EncodePool(command, pool_pipeline, tensors[2], tensors[3], 80, 48, 64);
      Encode3x3(command, three_pipeline, l24, weights[2], tensors[3], tensors[4]);
      EncodePool(command, pool_pipeline, tensors[4], tensors[5], 40, 24, 128);
      Encode3x3(command, three_pipeline, l26, weights[3], tensors[5], tensors[6]);
      EncodePool(command, pool_pipeline, tensors[6], tensors[7], 20, 12, 128);
      Encode3x3(command, three_pipeline, l28, weights[4], tensors[7], tensors[8]);
      EncodePool(command, pool_pipeline, tensors[8], tensors[9], 10, 6, 256);
      Encode1x1(command, one_pipeline, l30, weights[5], tensors[9], tensors[10]);
      Encode1x1(command, one_pipeline, l31, weights[6], tensors[10], tensors[11]);
      EncodeUpscale(command, upscale_pipeline, tensors[11], tensors[12],
                    5, 3, 10, 6, 256);
      EncodeAdd(command, add_pipeline, tensors[12], tensors[8], tensors[13],
                10, 6, 256, 3, 2, 8);
      Encode1x1(command, one_pipeline, l34, weights[7], tensors[13], tensors[14]);
      Encode1x1(command, one_pipeline, l35, weights[8], tensors[14], tensors[15]);
      EncodeUpscale(command, upscale_pipeline, tensors[15], tensors[16],
                    10, 6, 20, 12, 128);
      EncodeAdd(command, add_pipeline, tensors[16], tensors[6], tensors[17],
                20, 12, 128, 5, 3, 8);
      Encode1x1(command, one_pipeline, l38, weights[9], tensors[17], tensors[18]);
      Encode1x1(command, one_pipeline, l39, weights[10], tensors[18], tensors[19]);
      EncodeUpscale(command, upscale_pipeline, tensors[19], tensors[20],
                    20, 12, 40, 24, 128);
      EncodeAdd(command, add_pipeline, tensors[20], tensors[4], tensors[21],
                40, 24, 128, 10, 6, 8);
      Encode1x1(command, one_pipeline, l42, weights[11], tensors[21], tensors[22]);
      Encode1x1(command, one_pipeline, l43, weights[12], tensors[22], tensors[23]);
      EncodeUpscale(command, upscale_pipeline, tensors[23], tensors[24],
                    40, 24, 80, 48, 64);
      EncodeAdd(command, add_pipeline, tensors[24], tensors[2], tensors[25],
                80, 48, 64, 20, 12, 8);
      Encode1x1(command, one_pipeline, l46, weights[13], tensors[25], tensors[26]);
      Encode1x1(command, one_pipeline, l47, weights[14], tensors[26], tensors[27]);
      EncodeUpscale(command, upscale_pipeline, tensors[27], tensors[28],
                    80, 48, 160, 96, 32);
      EncodeAdd(command, add_pipeline, tensors[28], tensors[0], tensors[29],
                160, 96, 32, 40, 24, 4);
      Encode1x1(command, one_pipeline, l50, weights[15], tensors[29], tensors[30]);
      Encode1x1(command, one_pipeline, l51, weights[16], tensors[30], tensors[31]);
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
        auto after = Payload(weights[index], kWeights[index].bytes);
        immutable_mismatches += Mismatches(weights_before[index], after);
        canary_mismatches +=
            CanaryMismatches(weights[index], kWeights[index].bytes);
        Write(output_root /
                  ("weight-token-" + std::to_string(kWeights[index].token) +
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
      WriteUnetReceipt(output_root / "run-receipt.json", run_id, poison,
                       device.name.UTF8String, weights_before, weights_after,
                       input_before, input_after, tensor_bytes, collisions,
                       immutable_mismatches, canary_mismatches);
      const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_connected_unet_l20_l51_mac\n"
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
