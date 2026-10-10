#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

namespace {

constexpr std::uint32_t kPoolWidth = 80;
constexpr std::uint32_t kPoolHeight = 48;
constexpr std::uint32_t kPoolChannels = 32;
constexpr std::size_t kPoolOutputBytes =
    std::size_t{kPoolWidth} * kPoolHeight * kPoolChannels * 2u;
constexpr std::size_t kPoolModuleBytes = 2032;
constexpr char kPoolModuleSha256[] =
    "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977";
constexpr char kLiveWeightSha256[] =
    "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9";
constexpr char kLiveManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

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
    const auto mixed = ChainSplitMix64(seed + index * step);
    StoreHalf(&input, index * 2u, values[mixed % values.size()]);
  }
  return input;
}

void WriteChainReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::vector<std::uint8_t>& weights,
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
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"selected_launches\": [\n"
         << "    {\"launch\": 20, \"module\": 28, \"module_sha256\": \""
         << kExpectedModuleSha256
         << "\", \"function\": \"k_conv_fp16_nhwc\", \"grid\": [10, 8, 1], \"block\": [32, 4, 1]},\n"
         << "    {\"launch\": 21, \"module\": 29, \"module_sha256\": \""
         << kPoolModuleSha256
         << "\", \"function\": \"k_pooling\", \"grid\": [20, 12, 1], \"block\": [4, 4, 16]}\n"
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

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage=fg-connected-l20-l21-rtx <conv-module.bin> "
                 "<pool-module.bin> <weights.bin> <output-dir> <run-id> "
                 "<poison-hex> <device-ordinal>\n";
    return 2;
  }
  try {
    const std::filesystem::path conv_module_path = argv[1];
    const std::filesystem::path pool_module_path = argv[2];
    const std::filesystem::path weight_path = argv[3];
    const std::filesystem::path output_directory = argv[4];
    const std::string run_id = argv[5];
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[6], &poison_end, 16);
    if (poison_end == argv[6] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    char* device_end = nullptr;
    const long device_ordinal = std::strtol(argv[7], &device_end, 10);
    if (device_end == argv[7] || *device_end != '\0' || device_ordinal < 0) {
      throw std::runtime_error("invalid CUDA device ordinal");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    std::filesystem::create_directories(output_directory);

    const auto conv_module_bytes = Read(conv_module_path);
    const auto pool_module_bytes = Read(pool_module_path);
    const auto expected_weights = Read(weight_path);
    const auto expected_input = GenerateChainInput();
    if (conv_module_bytes.size() != kExpectedModuleBytes ||
        Sha256(conv_module_bytes) != kExpectedModuleSha256) {
      throw std::runtime_error("selected convolution module identity mismatch");
    }
    if (pool_module_bytes.size() != kPoolModuleBytes ||
        Sha256(pool_module_bytes) != kPoolModuleSha256) {
      throw std::runtime_error("selected pooling module identity mismatch");
    }
    if (expected_weights.size() != kWeightBytes ||
        Sha256(expected_weights) != kLiveWeightSha256) {
      throw std::runtime_error("live model weight provenance mismatch");
    }

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, static_cast<int>(device_ordinal)), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUmodule conv_module = nullptr;
    CUmodule pool_module = nullptr;
    Check(cuModuleLoadData(&conv_module, conv_module_bytes.data()),
          "cuModuleLoadData(convolution)");
    Check(cuModuleLoadData(&pool_module, pool_module_bytes.data()),
          "cuModuleLoadData(pooling)");
    CUfunction conv_function = nullptr;
    CUfunction pool_function = nullptr;
    Check(cuModuleGetFunction(&conv_function, conv_module, "k_conv_fp16_nhwc"),
          "cuModuleGetFunction(convolution)");
    Check(cuModuleGetFunction(&pool_function, pool_module, "k_pooling"),
          "cuModuleGetFunction(pooling)");

    GuardedDeviceBuffer weights;
    GuardedDeviceBuffer input;
    GuardedDeviceBuffer intermediate;
    GuardedDeviceBuffer output;
    weights.Allocate(kWeightBytes);
    input.Allocate(kInputBytes);
    intermediate.Allocate(kOutputBytes);
    output.Allocate(kPoolOutputBytes);
    Check(cuMemcpyHtoD(weights.payload(), expected_weights.data(), kWeightBytes),
          "cuMemcpyHtoD(weights)");
    Check(cuMemcpyHtoD(input.payload(), expected_input.data(), kInputBytes),
          "cuMemcpyHtoD(input)");
    Check(cuMemsetD8(intermediate.payload(), poison, kOutputBytes),
          "cuMemsetD8(intermediate)");
    Check(cuMemsetD8(output.payload(), poison, kPoolOutputBytes),
          "cuMemsetD8(output)");

    CUdeviceptr weight_pointer = weights.payload();
    CUdeviceptr input_pointer = input.payload();
    CUdeviceptr intermediate_pointer = intermediate.payload();
    std::uint32_t width = kWidth;
    std::uint32_t height = kHeight;
    std::uint32_t padded_width = kPaddedWidth;
    std::uint32_t padded_height = kPaddedHeight;
    std::uint32_t conv_output_width = kWidth;
    std::uint32_t conv_output_height = kHeight;
    void* conv_parameters[] = {
        &weight_pointer,       &input_pointer,       &width,
        &height,               &padded_width,        &padded_height,
        &intermediate_pointer, &conv_output_width,   &conv_output_height,
    };
    Check(cuLaunchKernel(conv_function, 10, 8, 1, 32, 4, 1, 0, nullptr,
                         conv_parameters, nullptr),
          "cuLaunchKernel(convolution)");

    CUdeviceptr output_pointer = output.payload();
    std::uint32_t pool_input_width = kWidth;
    std::uint32_t pool_input_height = kHeight;
    std::uint32_t pool_output_width = kPoolWidth;
    std::uint32_t pool_output_height = kPoolHeight;
    std::uint32_t pool_channels = kPoolChannels;
    void* pool_parameters[] = {
        &intermediate_pointer, &output_pointer,      &pool_input_width,
        &pool_input_height,    &pool_output_width,   &pool_output_height,
        &pool_channels,
    };
    Check(cuLaunchKernel(pool_function, 20, 12, 1, 4, 4, 16, 0, nullptr,
                         pool_parameters, nullptr),
          "cuLaunchKernel(pooling)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize(connected group)");

    const auto guarded_weights = weights.ReadAll();
    const auto guarded_input = input.ReadAll();
    const auto guarded_intermediate = intermediate.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto actual_weights = Payload(guarded_weights, kWeightBytes);
    const auto actual_input = Payload(guarded_input, kInputBytes);
    const auto actual_intermediate = Payload(guarded_intermediate, kOutputBytes);
    const auto actual_output = Payload(guarded_output, kPoolOutputBytes);
    const std::size_t immutable_mismatches =
        Mismatches(actual_weights, expected_weights) +
        Mismatches(actual_input, expected_input);
    const std::size_t canary_mismatches =
        CanaryMismatches(guarded_weights, kWeightBytes) +
        CanaryMismatches(guarded_input, kInputBytes) +
        CanaryMismatches(guarded_intermediate, kOutputBytes) +
        CanaryMismatches(guarded_output, kPoolOutputBytes);
    const std::size_t intermediate_collisions =
        UntouchedHalfwords(actual_intermediate, poison);
    const std::size_t output_collisions =
        UntouchedHalfwords(actual_output, poison);

    Write(output_directory / "input-after.bin", actual_input);
    Write(output_directory / "weights-after.bin", actual_weights);
    Write(output_directory / "intermediate.bin", actual_intermediate);
    Write(output_directory / "output.bin", actual_output);
    WriteChainReceipt(output_directory / "run-receipt.json", run_id, poison,
                      expected_weights, expected_input, actual_weights,
                      actual_input, actual_intermediate, actual_output,
                      immutable_mismatches, canary_mismatches,
                      intermediate_collisions, output_collisions);

    const bool pass = immutable_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_connected_l20_l21_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "intermediate_sha256=" << Sha256(actual_intermediate) << '\n'
              << "output_sha256=" << Sha256(actual_output) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';

    weights.Free();
    input.Free();
    intermediate.Free();
    output.Free();
    Check(cuModuleUnload(pool_module), "cuModuleUnload(pooling)");
    Check(cuModuleUnload(conv_module), "cuModuleUnload(convolution)");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
