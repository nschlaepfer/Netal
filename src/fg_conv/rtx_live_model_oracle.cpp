#define FG_CONV_L20_NO_MAIN
#include "rtx_oracle.cpp"

namespace {

constexpr char kLiveWeightSha256[] =
    "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9";
constexpr char kLiveManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::vector<std::uint8_t> GenerateLiveInput() {
  constexpr std::array<std::uint16_t, 7> values = {
      0x0000,  // 0
      0x3000,  // +0.125
      0xb000,  // -0.125
      0x3400,  // +0.25
      0xb400,  // -0.25
      0x3800,  // +0.5
      0xb800,  // -0.5
  };
  std::vector<std::uint8_t> input(kInputBytes);
  constexpr std::uint64_t seed = 0xD1B54A32D192ED03ull;
  constexpr std::uint64_t step = 0x9E3779B97F4A7C15ull;
  for (std::size_t index = 0; index < kInputBytes / 2; ++index) {
    const auto mixed = SplitMix64(seed + index * step);
    StoreHalf(&input, index * 2, values[mixed % values.size()]);
  }
  return input;
}

void WriteLiveReceipt(const std::filesystem::path& path,
                      const std::string& run_id, std::uint8_t poison,
                      const std::string& module_sha,
                      const std::vector<std::uint8_t>& expected_weights,
                      const std::vector<std::uint8_t>& expected_input,
                      const std::vector<std::uint8_t>& actual_weights,
                      const std::vector<std::uint8_t>& actual_input,
                      const std::vector<std::uint8_t>& output,
                      std::size_t input_mismatches,
                      std::size_t canary_mismatches,
                      std::size_t untouched_halfwords) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write live receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-l20-live-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 28,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"function_name\": \"k_conv_fp16_nhwc\",\n"
         << "  \"launch_order\": 20,\n"
         << "  \"grid\": [10, 8, 1],\n"
         << "  \"block\": [32, 4, 1],\n"
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
         << untouched_halfwords << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_weight_file_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage=fg-conv-l20-live-rtx <module.bin> <weights.bin> "
                 "<output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path weight_path = argv[2];
    const std::filesystem::path output_directory = argv[3];
    const std::string run_id = argv[4];
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[5], &poison_end, 16);
    if (poison_end == argv[5] || *poison_end != '\0' || parsed_poison > 255) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    std::filesystem::create_directories(output_directory);
    const auto module_bytes = Read(module_path);
    const auto expected_weights = Read(weight_path);
    if (module_bytes.size() != kExpectedModuleBytes ||
        Sha256(module_bytes) != kExpectedModuleSha256) {
      throw std::runtime_error("selected module identity mismatch");
    }
    if (expected_weights.size() != kWeightBytes ||
        Sha256(expected_weights) != kLiveWeightSha256) {
      throw std::runtime_error("live weight provenance mismatch");
    }
    const auto expected_input = GenerateLiveInput();

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "cuModuleLoadData");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "k_conv_fp16_nhwc"),
          "cuModuleGetFunction");

    GuardedDeviceBuffer weights;
    GuardedDeviceBuffer input;
    GuardedDeviceBuffer output;
    weights.Allocate(kWeightBytes);
    input.Allocate(kInputBytes);
    output.Allocate(kOutputBytes);
    Check(cuMemcpyHtoD(weights.payload(), expected_weights.data(), kWeightBytes),
          "cuMemcpyHtoD(weights)");
    Check(cuMemcpyHtoD(input.payload(), expected_input.data(), kInputBytes),
          "cuMemcpyHtoD(input)");
    Check(cuMemsetD8(output.payload(), poison, kOutputBytes),
          "cuMemsetD8(output)");

    CUdeviceptr weight_pointer = weights.payload();
    CUdeviceptr input_pointer = input.payload();
    CUdeviceptr output_pointer = output.payload();
    std::uint32_t width = kWidth;
    std::uint32_t height = kHeight;
    std::uint32_t padded_width = kPaddedWidth;
    std::uint32_t padded_height = kPaddedHeight;
    std::uint32_t output_width = kWidth;
    std::uint32_t output_height = kHeight;
    void* parameters[] = {
        &weight_pointer, &input_pointer,  &width,
        &height,         &padded_width,   &padded_height,
        &output_pointer, &output_width,   &output_height,
    };
    Check(cuLaunchKernel(function, 10, 8, 1, 32, 4, 1, 0, nullptr,
                         parameters, nullptr),
          "cuLaunchKernel");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");
    const auto guarded_weights = weights.ReadAll();
    const auto guarded_input = input.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto actual_weights = Payload(guarded_weights, kWeightBytes);
    const auto actual_input = Payload(guarded_input, kInputBytes);
    const auto actual_output = Payload(guarded_output, kOutputBytes);
    const std::size_t input_mismatches =
        Mismatches(actual_weights, expected_weights) +
        Mismatches(actual_input, expected_input);
    const std::size_t canary_mismatches =
        CanaryMismatches(guarded_weights, kWeightBytes) +
        CanaryMismatches(guarded_input, kInputBytes) +
        CanaryMismatches(guarded_output, kOutputBytes);
    const std::size_t untouched = UntouchedHalfwords(actual_output, poison);
    Write(output_directory / "input-after.bin", actual_input);
    Write(output_directory / "output.bin", actual_output);
    WriteLiveReceipt(output_directory / "run-receipt.json", run_id, poison,
                     Sha256(module_bytes), expected_weights, expected_input,
                     actual_weights, actual_input, actual_output,
                     input_mismatches, canary_mismatches, untouched);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_conv_l20_live_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(actual_output) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    weights.Free();
    input.Free();
    output.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
