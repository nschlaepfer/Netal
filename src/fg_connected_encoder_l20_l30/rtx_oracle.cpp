#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"
#include "../fg_conv/conv_1x1_contract.h"
#include "../fg_conv/conv_3x3_contract.h"

#include <memory>

namespace {

constexpr char kPoolModuleSha256[] =
    "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977";
constexpr std::size_t kPoolModuleBytes = 2032;
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

void WriteEncoderReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::vector<std::vector<std::uint8_t>>& weights,
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
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
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

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-connected-encoder-l20-l30-rtx <module-dir> "
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

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, static_cast<int>(device_ordinal)), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    const LoadedKernel k20 = LoadKernel(
        module_root / "cu-module-000028.bin", l20.module_bytes,
        l20.module_sha256, "k_conv_fp16_nhwc");
    const LoadedKernel pool = LoadKernel(
        module_root / "cu-module-000029.bin", kPoolModuleBytes,
        kPoolModuleSha256, "k_pooling");
    const LoadedKernel k22 = LoadKernel(
        module_root / "cu-module-000030.bin", l22.module_bytes,
        l22.module_sha256, "k_conv_fp16_nhwc");
    const LoadedKernel k24 = LoadKernel(
        module_root / "cu-module-000031.bin", l24.module_bytes,
        l24.module_sha256, "k_conv_fp16_nhwc");
    const LoadedKernel k26 = LoadKernel(
        module_root / "cu-module-000032.bin", l26.module_bytes,
        l26.module_sha256, "k_conv_fp16_nhwc");
    const LoadedKernel k28 = LoadKernel(
        module_root / "cu-module-000033.bin", l28.module_bytes,
        l28.module_sha256, "k_conv_fp16_nhwc");
    const LoadedKernel k30 = LoadKernel(
        module_root / "cu-module-000034.bin", l30.module_bytes,
        l30.module_sha256, "k_conv_fp16_nhwc");

    const std::array<std::pair<std::uint32_t, std::size_t>, 6> weight_specs = {{
        {l20.weight_token, l20.weight_bytes},
        {l24.weight_token, l24.weight_bytes},
        {l26.weight_token, l26.weight_bytes},
        {l28.weight_token, l28.weight_bytes},
        {l22.weight_token, l22.weight_bytes},
        {l30.weight_token, l30.weight_bytes},
    }};
    const std::array<const char*, 6> weight_hashes = {
        l20.weight_sha256, l24.weight_sha256, l26.weight_sha256,
        l28.weight_sha256, l22.weight_sha256, l30.weight_sha256};
    std::vector<std::vector<std::uint8_t>> weights_before;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> weights;
    for (std::size_t index = 0; index < weight_specs.size(); ++index) {
      std::ostringstream name;
      name << "default-token-" << std::setw(6) << std::setfill('0')
           << weight_specs[index].first << ".bin";
      auto bytes = Read(model_root / name.str());
      if (bytes.size() != weight_specs[index].second ||
          Sha256(bytes) != weight_hashes[index]) {
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
            "poison encoder tensor");
      tensors.push_back(std::move(tensor));
    }

    Launch3x3(l20, k20, *weights[0], *input, *tensors[0]);
    LaunchPool(pool, *tensors[0], *tensors[1], 160, 96, 32);
    Launch1x1(l22, k22, *weights[4], *tensors[1], *tensors[2]);
    LaunchPool(pool, *tensors[2], *tensors[3], 80, 48, 64);
    Launch3x3(l24, k24, *weights[1], *tensors[3], *tensors[4]);
    LaunchPool(pool, *tensors[4], *tensors[5], 40, 24, 128);
    Launch3x3(l26, k26, *weights[2], *tensors[5], *tensors[6]);
    LaunchPool(pool, *tensors[6], *tensors[7], 20, 12, 128);
    Launch3x3(l28, k28, *weights[3], *tensors[7], *tensors[8]);
    LaunchPool(pool, *tensors[8], *tensors[9], 10, 6, 256);
    Launch1x1(l30, k30, *weights[5], *tensors[9], *tensors[10]);
    Check(cuCtxSynchronize(), "synchronize connected encoder");

    const auto input_guarded = input->ReadAll();
    const auto input_after = Payload(input_guarded, kInputBytes);
    std::size_t immutable_mismatches = Mismatches(input_before, input_after);
    std::size_t canary_mismatches = CanaryMismatches(input_guarded, kInputBytes);
    std::vector<std::vector<std::uint8_t>> weights_after;
    for (std::size_t index = 0; index < weights.size(); ++index) {
      const auto guarded = weights[index]->ReadAll();
      auto after = Payload(guarded, weight_specs[index].second);
      immutable_mismatches += Mismatches(weights_before[index], after);
      canary_mismatches += CanaryMismatches(guarded, weight_specs[index].second);
      Write(output_root /
                ("weight-token-" + std::to_string(weight_specs[index].first) +
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
    WriteEncoderReceipt(output_root / "run-receipt.json", run_id, poison,
                        weights_before, weights_after, input_before,
                        input_after, tensor_bytes, collisions,
                        immutable_mismatches, canary_mismatches);
    const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_connected_encoder_l20_l30_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(tensor_bytes.back()) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';

    for (auto& tensor : tensors) tensor->Free();
    input->Free();
    for (auto& weight : weights) weight->Free();
    for (const LoadedKernel* kernel :
         std::array<const LoadedKernel*, 7>{&k30, &k28, &k26, &k24,
                                            &k22, &pool, &k20}) {
      Check(cuModuleUnload(kernel->module), "unload selected module");
    }
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
