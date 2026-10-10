#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"
#include "../fg_conv/conv_1x1_contract.h"
#include "../fg_conv/conv_3x3_contract.h"

#include <algorithm>
#include <memory>

namespace {

constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

struct ModuleSpec {
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
  const char* function;
};

constexpr std::array<ModuleSpec, 25> kModules = {{
    {28, 16512, "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d", "k_conv_fp16_nhwc"},
    {29, 2032, "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977", "k_pooling"},
    {30, 4952, "31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785", "k_conv_fp16_nhwc"},
    {31, 11168, "e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63", "k_conv_fp16_nhwc"},
    {32, 11416, "c74961f6f71f4e7bac5b216cc704b10a0fdd6f0987352fdd315f371467ca7187", "k_conv_fp16_nhwc"},
    {33, 14752, "6f2ffbe7faacde8119719d92dacde40f21ddd562878381b81654dad7a66775dd", "k_conv_fp16_nhwc"},
    {34, 11328, "2f490857c6f7aa22f2435057706ec10b28b6b8d3081bd967ebe1d61550ac6419", "k_conv_fp16_nhwc"},
    {35, 19424, "331853771cae4aa041038d8e9c74dcb1f8d0d38b4d5603e2e8d079ad1e44fd04", "k_conv_fp16_nhwc"},
    {36, 2296, "d6962e058c46f6de9c6c5f8c75dafde64a34f094a49f1719af16c505f3b530fd", "k_upscale"},
    {37, 2152, "1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26", "k_element_wise"},
    {38, 11320, "8e7abed27b7eae0dd274b2409a13d6dde98277d9e743f0e36a1042022a16ceda", "k_conv_fp16_nhwc"},
    {39, 11328, "939d5d9cf7012fb3ed3ce4e4792ccf5c6befec165ae7f7a0d3704cb1103c3030", "k_conv_fp16_nhwc"},
    {40, 2296, "070c7f4987251c28bcad097ac8a00e7b1e2644dae4267ae37b676e280478a2b8", "k_upscale"},
    {41, 2152, "a717783407a1cea5e3d64b4b23edb451f2b791d8cb7980b374433b9f6ded17e1", "k_element_wise"},
    {42, 7368, "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc", "k_conv_fp16_nhwc"},
    {43, 8440, "53fdf8d5404c93f2fe64582f271192df3b24c2ef08d9564ffeb9d7990485db0e", "k_conv_fp16_nhwc"},
    {44, 8496, "7113dfaae4945541276b3e662cba366f9a3fd9666db7a338efa867cfd7f46e31", "k_conv_fp16_nhwc"},
    {45, 2296, "547a05ab275edcb0d413e4b7f89bada85f08035fa0517cd5267960fe695498fc", "k_upscale"},
    {46, 2152, "c733968c9cdca9ba170a9a453d2a831537f467de10e089325aabf8e8ed1cd58f", "k_element_wise"},
    {47, 6088, "8a3efc1e9166ed3ee4a54aea535910c0562b6753b2d0daae0402e1ed36ff19dc", "k_conv_fp16_nhwc"},
    {48, 5992, "11906664ac548c61262a12b0950bf135926cf269de2d7691d0edfbe4a95e63dd", "k_conv_fp16_nhwc"},
    {49, 2296, "2b91bf740602f074acbaa5efba1cd1115c5ee52653a96d087c0d52231de2a74a", "k_upscale"},
    {50, 2144, "dc36c503f9fca1cb42944cab551c268b2d738912927f4aab9db4d53ae66766b6", "k_element_wise"},
    {51, 4840, "304d0c0e58958fa7eab6a82fd046ff689c51a986aff88a069f6f509c434a5b96", "k_conv_fp16_nhwc"},
    {52, 3912, "27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2", "k_conv_fp16_nhwc"},
}};

struct WeightSpec {
  std::uint32_t launch;
  std::uint32_t module;
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
};

constexpr std::array<WeightSpec, 17> kWeights = {{
    {20,28,63,56320,"44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"},
    {22,30,65,10240,"eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60"},
    {24,31,67,299008,"4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55"},
    {26,32,69,593920,"c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404"},
    {28,33,71,1187840,"fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93"},
    {30,34,73,540672,"e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9"},
    {31,35,75,532480,"9f20d6589cc9269ad020233e68196d66eba0e8a76497374cf6d445385f94dbc4"},
    {34,38,77,270336,"9e5da09b93650c74e89d5262aa8a9a6a3dcbd0d753dc8c9d9aed39731e39b2a0"},
    {35,39,79,135168,"3969384493583e98dcf357d3046a10a5b836585261f3e584605bf11e23fb8d46"},
    {38,42,81,69632,"a0c6690faee2838521c6ad0f9445425d639446c0ef1b7fed9b43d21e7a8ec650"},
    {39,42,83,69632,"3a13970b482f1f78ede1bfe531956c203f08f99077c81bda2d1e4d0f453cf9a0"},
    {42,43,85,69632,"47361a699fa6514332dc560cf2416872256e61d1b36b1ca29df323a53ee2be77"},
    {43,44,87,34816,"327bd9b02da4aaac8b251320f60b4aeccb0df7df84bba44d7f1f6a9fdf62c934"},
    {46,47,89,18432,"e24f67110bab401ab7f70a861f3f09d2cd3cd6abf8d1aa158ba3300efb64797e"},
    {47,48,91,9216,"325d7dfac5620a89edc1d2315731414cf755c50506ea5758a8c3cf563649577c"},
    {50,51,93,5120,"bb98eb15be7d255a15a46994589705560e0691e273e286a36a68fd679a1890a9"},
    {51,52,95,2560,"477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e"},
}};

struct TensorSpec {
  const char* id;
  std::uint32_t launch;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

constexpr std::array<TensorSpec, 32> kTensors = {{
    {"l20", 20, 160, 96, 32},
    {"l21", 21, 80, 48, 32},
    {"l22", 22, 80, 48, 64},
    {"l23", 23, 40, 24, 64},
    {"l24", 24, 40, 24, 128},
    {"l25", 25, 20, 12, 128},
    {"l26", 26, 20, 12, 128},
    {"l27", 27, 10, 6, 128},
    {"l28", 28, 10, 6, 256},
    {"l29", 29, 5, 3, 256},
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

struct LoadedKernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

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

LoadedKernel LoadKernel(const std::filesystem::path& path,
                        std::size_t expected_bytes,
                        const std::string& expected_sha,
                        const char* function_name) {
  const auto bytes = Read(path);
  if (bytes.size() != expected_bytes || Sha256(bytes) != expected_sha) {
    throw std::runtime_error("selected module identity mismatch: " +
                             path.string());
  }
  LoadedKernel result;
  Check(cuModuleLoadData(&result.module, bytes.data()), "cuModuleLoadData");
  Check(cuModuleGetFunction(&result.function, result.module, function_name),
        "cuModuleGetFunction");
  return result;
}

std::unique_ptr<GuardedDeviceBuffer> Allocate(std::size_t bytes) {
  auto result = std::make_unique<GuardedDeviceBuffer>();
  result->Allocate(bytes);
  return result;
}

void Launch3x3(const fg_conv_3x3::Layer& layer,
               const LoadedKernel& kernel, GuardedDeviceBuffer& weight,
               GuardedDeviceBuffer& input, GuardedDeviceBuffer& output) {
  CUdeviceptr weight_pointer = weight.payload();
  CUdeviceptr input_pointer = input.payload();
  CUdeviceptr output_pointer = output.payload();
  std::uint32_t width = layer.width;
  std::uint32_t height = layer.height;
  std::uint32_t padded_width = layer.width + 2u;
  std::uint32_t padded_height = layer.height + 2u;
  std::uint32_t output_width = layer.width;
  std::uint32_t output_height = layer.height;
  void* parameters[] = {
      &weight_pointer, &input_pointer, &width,
      &height,         &padded_width,  &padded_height,
      &output_pointer, &output_width,  &output_height,
  };
  Check(cuLaunchKernel(kernel.function, layer.grid[0], layer.grid[1],
                       layer.grid[2], layer.block[0], layer.block[1],
                       layer.block[2], 0, nullptr, parameters, nullptr),
        "cuLaunchKernel(3x3)");
}

void Launch1x1(const fg_conv_1x1::Layer& layer,
               const LoadedKernel& kernel, GuardedDeviceBuffer& weight,
               GuardedDeviceBuffer& input, GuardedDeviceBuffer& output) {
  CUdeviceptr weight_pointer = weight.payload();
  CUdeviceptr input_pointer = input.payload();
  CUdeviceptr output_pointer = output.payload();
  std::uint32_t width = layer.width;
  std::uint32_t height = layer.height;
  std::uint32_t output_width = layer.width;
  std::uint32_t output_height = layer.height;
  void* parameters[] = {&weight_pointer, &input_pointer, &width, &height,
                        &output_pointer, &output_width, &output_height};
  Check(cuLaunchKernel(kernel.function, layer.grid[0], layer.grid[1],
                       layer.grid[2], layer.block[0], layer.block[1],
                       layer.block[2], 0, nullptr, parameters, nullptr),
        "cuLaunchKernel(1x1)");
}

void LaunchPool(const LoadedKernel& kernel, GuardedDeviceBuffer& input,
                GuardedDeviceBuffer& output, std::uint32_t input_width,
                std::uint32_t input_height, std::uint32_t channels) {
  CUdeviceptr input_pointer = input.payload();
  CUdeviceptr output_pointer = output.payload();
  std::uint32_t output_width = (input_width + 1u) / 2u;
  std::uint32_t output_height = (input_height + 1u) / 2u;
  void* parameters[] = {&input_pointer,  &output_pointer, &input_width,
                        &input_height,   &output_width,   &output_height,
                        &channels};
  const std::uint32_t grid_x = (output_width + 3u) / 4u;
  const std::uint32_t grid_y = (output_height + 3u) / 4u;
  Check(cuLaunchKernel(kernel.function, grid_x, grid_y, 1, 4, 4, 16, 0,
                       nullptr, parameters, nullptr),
        "cuLaunchKernel(pooling)");
}

void LaunchUpscale(const LoadedKernel& kernel, GuardedDeviceBuffer& input,
                   GuardedDeviceBuffer& output, std::uint32_t input_width,
                   std::uint32_t input_height, std::uint32_t output_width,
                   std::uint32_t output_height,
                   const std::array<std::uint32_t, 3>& grid,
                   const std::array<std::uint32_t, 3>& block) {
  CUdeviceptr input_pointer = input.payload();
  CUdeviceptr output_pointer = output.payload();
  std::uint32_t physical_height = output_height;
  std::uint32_t physical_width = output_width;
  void* parameters[] = {
      &input_pointer,   &output_pointer, &input_height,  &input_width,
      &output_height,  &output_width,   &physical_height,
      &physical_width,
  };
  Check(cuLaunchKernel(kernel.function, grid[0], grid[1], grid[2], block[0],
                       block[1], block[2], 0, nullptr, parameters, nullptr),
        "cuLaunchKernel(upscale)");
}

void LaunchAdd(const LoadedKernel& kernel, GuardedDeviceBuffer& input_a,
               GuardedDeviceBuffer& input_b, GuardedDeviceBuffer& output,
               std::uint32_t width, std::uint32_t height,
               std::uint32_t channels, std::uint32_t grid_x,
               std::uint32_t grid_y, std::uint32_t block_z) {
  CUdeviceptr input_a_pointer = input_a.payload();
  CUdeviceptr input_b_pointer = input_b.payload();
  CUdeviceptr output_pointer = output.payload();
  void* parameters[] = {
      &input_a_pointer, &input_b_pointer, &output_pointer,
      &width,           &height,          &channels,
      &width,           &height,          &width,
      &height,
  };
  Check(cuLaunchKernel(kernel.function, grid_x, grid_y, 1, 4, 4, block_z, 0,
                       nullptr, parameters, nullptr),
        "cuLaunchKernel(elementwise add)");
}

void WriteUnetReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::vector<std::vector<std::uint8_t>>& weights,
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
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"selected_launch_first\": 20,\n"
         << "  \"selected_launch_last\": 51,\n"
         << "  \"selected_launch_count\": 32,\n"
         << "  \"pool_module\": {\"token\": 29, \"sha256\": \""
         << kModules[1].sha256 << "\"},\n"
         << "  \"model_capture_manifest_sha256\": \""
         << kManifestSha256 << "\",\n"
         << "  \"weights\": [\n";
  for (std::size_t index = 0; index < kWeights.size(); ++index) {
    const auto& weight = kWeights[index];
    const auto module = std::find_if(
        kModules.begin(), kModules.end(),
        [&](const ModuleSpec& value) { return value.token == weight.module; });
    if (module == kModules.end()) {
      throw std::runtime_error("weight module missing from selected table");
    }
    stream << "    {\"launch\": " << weight.launch
           << ", \"module\": " << weight.module
           << ", \"module_sha256\": \"" << module->sha256
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

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-connected-unet-l20-l51-rtx <module-dir> "
                 "<model-dir> <output-dir> <run-id> <poison-hex> "
                 "<device-ordinal>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_root = argv[1];
    const std::filesystem::path model_root = argv[2];
    const std::filesystem::path output_root = argv[3];
    const std::string run_id = argv[4];
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[5], &poison_end, 16);
    if (poison_end == argv[5] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    char* device_end = nullptr;
    const long device_ordinal = std::strtol(argv[6], &device_end, 10);
    if (device_end == argv[6] || *device_end != '\0' || device_ordinal < 0) {
      throw std::runtime_error("invalid device ordinal");
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

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, static_cast<int>(device_ordinal)), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    std::array<LoadedKernel, 53> kernels{};
    for (const auto& module : kModules) {
      std::ostringstream name;
      name << "cu-module-" << std::setw(6) << std::setfill('0')
           << module.token << ".bin";
      kernels[module.token] = LoadKernel(module_root / name.str(), module.bytes,
                                         module.sha256, module.function);
    }

    std::vector<std::vector<std::uint8_t>> weights_before;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> weights;
    for (const auto& weight : kWeights) {
      std::ostringstream name;
      name << "default-token-" << std::setw(6) << std::setfill('0')
           << weight.token << ".bin";
      auto bytes = Read(model_root / name.str());
      if (bytes.size() != weight.bytes || Sha256(bytes) != weight.sha256) {
        throw std::runtime_error("model-state identity mismatch: " + name.str());
      }
      auto buffer = Allocate(bytes.size());
      Check(cuMemcpyHtoD(buffer->payload(), bytes.data(), bytes.size()),
            "copy model state");
      weights_before.push_back(std::move(bytes));
      weights.push_back(std::move(buffer));
    }
    auto input = Allocate(kInputBytes);
    Check(cuMemcpyHtoD(input->payload(), input_before.data(), input_before.size()),
          "copy encoder input");
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> tensors;
    for (const auto& spec : kTensors) {
      auto tensor = Allocate(TensorBytes(spec));
      Check(cuMemsetD8(tensor->payload(), poison, TensorBytes(spec)),
            "poison U-Net tensor");
      tensors.push_back(std::move(tensor));
    }

    Launch3x3(l20, kernels[28], *weights[0], *input, *tensors[0]);
    LaunchPool(kernels[29], *tensors[0], *tensors[1], 160, 96, 32);
    Launch1x1(l22, kernels[30], *weights[1], *tensors[1], *tensors[2]);
    LaunchPool(kernels[29], *tensors[2], *tensors[3], 80, 48, 64);
    Launch3x3(l24, kernels[31], *weights[2], *tensors[3], *tensors[4]);
    LaunchPool(kernels[29], *tensors[4], *tensors[5], 40, 24, 128);
    Launch3x3(l26, kernels[32], *weights[3], *tensors[5], *tensors[6]);
    LaunchPool(kernels[29], *tensors[6], *tensors[7], 20, 12, 128);
    Launch3x3(l28, kernels[33], *weights[4], *tensors[7], *tensors[8]);
    LaunchPool(kernels[29], *tensors[8], *tensors[9], 10, 6, 256);
    Launch1x1(l30, kernels[34], *weights[5], *tensors[9], *tensors[10]);
    Launch1x1(l31, kernels[35], *weights[6], *tensors[10], *tensors[11]);
    LaunchUpscale(kernels[36], *tensors[11], *tensors[12], 5, 3, 10, 6,
                  {1, 2, 2}, {32, 5, 5});
    LaunchAdd(kernels[37], *tensors[12], *tensors[8], *tensors[13], 10, 6,
              256, 3, 2, 8);
    Launch1x1(l34, kernels[38], *weights[7], *tensors[13], *tensors[14]);
    Launch1x1(l35, kernels[39], *weights[8], *tensors[14], *tensors[15]);
    LaunchUpscale(kernels[40], *tensors[15], *tensors[16], 10, 6, 20, 12,
                  {1, 3, 2}, {16, 8, 8});
    LaunchAdd(kernels[41], *tensors[16], *tensors[6], *tensors[17], 20, 12,
              128, 5, 3, 8);
    Launch1x1(l38, kernels[42], *weights[9], *tensors[17], *tensors[18]);
    Launch1x1(l39, kernels[42], *weights[10], *tensors[18], *tensors[19]);
    LaunchUpscale(kernels[40], *tensors[19], *tensors[20], 20, 12, 40, 24,
                  {1, 5, 3}, {16, 8, 8});
    LaunchAdd(kernels[41], *tensors[20], *tensors[4], *tensors[21], 40, 24,
              128, 10, 6, 8);
    Launch1x1(l42, kernels[43], *weights[11], *tensors[21], *tensors[22]);
    Launch1x1(l43, kernels[44], *weights[12], *tensors[22], *tensors[23]);
    LaunchUpscale(kernels[45], *tensors[23], *tensors[24], 40, 24, 80, 48,
                  {1, 8, 5}, {8, 11, 11});
    LaunchAdd(kernels[46], *tensors[24], *tensors[2], *tensors[25], 80, 48,
              64, 20, 12, 8);
    Launch1x1(l46, kernels[47], *weights[13], *tensors[25], *tensors[26]);
    Launch1x1(l47, kernels[48], *weights[14], *tensors[26], *tensors[27]);
    LaunchUpscale(kernels[49], *tensors[27], *tensors[28], 80, 48, 160, 96,
                  {1, 10, 6}, {4, 16, 16});
    LaunchAdd(kernels[50], *tensors[28], *tensors[0], *tensors[29], 160, 96,
              32, 40, 24, 4);
    Launch1x1(l50, kernels[51], *weights[15], *tensors[29], *tensors[30]);
    Launch1x1(l51, kernels[52], *weights[16], *tensors[30], *tensors[31]);
    Check(cuCtxSynchronize(), "synchronize connected U-Net");

    const auto input_guarded = input->ReadAll();
    const auto input_after = Payload(input_guarded, kInputBytes);
    std::size_t immutable_mismatches = Mismatches(input_before, input_after);
    std::size_t canary_mismatches = CanaryMismatches(input_guarded, kInputBytes);
    std::vector<std::vector<std::uint8_t>> weights_after;
    for (std::size_t index = 0; index < weights.size(); ++index) {
      const auto guarded = weights[index]->ReadAll();
      auto after = Payload(guarded, kWeights[index].bytes);
      immutable_mismatches += Mismatches(weights_before[index], after);
      canary_mismatches += CanaryMismatches(guarded, kWeights[index].bytes);
      Write(output_root /
                ("weight-token-" + std::to_string(kWeights[index].token) +
                 "-after.bin"),
            after);
      weights_after.push_back(std::move(after));
    }
    std::vector<std::vector<std::uint8_t>> tensor_bytes;
    std::vector<std::size_t> collisions;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      const auto guarded = tensors[index]->ReadAll();
      auto bytes = Payload(guarded, TensorBytes(kTensors[index]));
      canary_mismatches +=
          CanaryMismatches(guarded, TensorBytes(kTensors[index]));
      collisions.push_back(UntouchedHalfwords(bytes, poison));
      Write(output_root / (std::string(kTensors[index].id) + ".bin"), bytes);
      tensor_bytes.push_back(std::move(bytes));
    }
    Write(output_root / "input-after.bin", input_after);
    WriteUnetReceipt(output_root / "run-receipt.json", run_id, poison,
                     weights_before, weights_after, input_before, input_after,
                     tensor_bytes, collisions, immutable_mismatches,
                     canary_mismatches);
    const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_connected_unet_l20_l51_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(tensor_bytes.back()) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';

    for (auto& tensor : tensors) tensor->Free();
    input->Free();
    for (auto& weight : weights) weight->Free();
    for (auto module = kModules.rbegin(); module != kModules.rend(); ++module) {
      Check(cuModuleUnload(kernels[module->token].module),
            "unload selected module");
    }
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
