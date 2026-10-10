#define FG_CONV_L20_NO_MAIN
#include "rtx_oracle.cpp"
#include "custom_conv_pre_contract.h"

namespace {

namespace cp = fg_custom_conv_pre;

void WriteConvPreReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::vector<std::uint8_t>& weights_before,
    const std::vector<std::uint8_t>& bias_before,
    const std::vector<std::uint8_t>& input_before,
    const std::vector<std::uint8_t>& weights_after,
    const std::vector<std::uint8_t>& bias_after,
    const std::vector<std::uint8_t>& input_after,
    const std::vector<std::uint8_t>& output,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t poison_collisions) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write convPre receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.custom-conv-pre-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"trace_launch_token\": \"launch-000057\",\n"
         << "  \"module_token\": 54,\n"
         << "  \"module_bytes\": " << cp::kModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << cp::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"custom_block0_convPre_kernel\",\n"
         << "  \"grid\": [80, 92, 1],\n"
         << "  \"block\": [32, 2, 4],\n"
         << "  \"packed_parameter_bytes\": 80,\n"
         << "  \"abi_scalars\": [32, 3, 3, 10, 1, 368, 640, 10, 1, 184, 320, 32],\n"
         << "  \"input_shape_nhwc\": [1, 368, 640, 10],\n"
         << "  \"output_shape_nhwc\": [1, 184, 320, 32],\n"
         << "  \"fused_operations\": [\"conv3x3\", \"leaky_relu\", \"clamp_-6_6\", \"average_pool_2x2\"],\n"
         << "  \"weight_source_token\": 106,\n"
         << "  \"bias_source_token\": 108,\n"
         << "  \"model_capture_point\": \"post_evaluate_0_fence_pre_evaluate_1\",\n"
         << "  \"model_capture_manifest_sha256\": \""
         << cp::kManifestSha256 << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights_before)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(weights_after)
         << "\",\n"
         << "  \"bias_pre_sha256\": \"" << Sha256(bias_before) << "\",\n"
         << "  \"bias_post_sha256\": \"" << Sha256(bias_after) << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-custom-conv-pre-v1\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"per_run_poison_collision_halfwords\": "
         << poison_collisions << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"external_model_files_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-custom-conv-pre-rtx <module> <weights> <bias> "
                 "<output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    const auto weights_before = Read(argv[2]);
    const auto bias_before = Read(argv[3]);
    if (module_bytes.size() != cp::kModuleBytes ||
        Sha256(module_bytes) != cp::kModuleSha256) {
      throw std::runtime_error("convPre module identity mismatch");
    }
    if (weights_before.size() != cp::kWeightBytes ||
        Sha256(weights_before) != cp::kWeightSha256 ||
        bias_before.size() != cp::kBiasBytes ||
        Sha256(bias_before) != cp::kBiasSha256) {
      throw std::runtime_error("convPre model identity mismatch");
    }
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[6], &poison_end, 16);
    if (poison_end == argv[6] || *poison_end != '\0' || parsed_poison > 255) {
      throw std::runtime_error("invalid poison");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::filesystem::path output_directory = argv[4];
    std::filesystem::create_directories(output_directory);
    const auto input_before = cp::GenerateInput();

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module,
                              "custom_block0_convPre_kernel"),
          "get convPre function");

    GuardedDeviceBuffer weights;
    GuardedDeviceBuffer bias;
    GuardedDeviceBuffer input;
    GuardedDeviceBuffer output;
    weights.Allocate(cp::kWeightBytes);
    bias.Allocate(cp::kBiasBytes);
    input.Allocate(cp::kInputBytes);
    output.Allocate(cp::kOutputBytes);
    Check(cuMemcpyHtoD(weights.payload(), weights_before.data(), cp::kWeightBytes),
          "copy weights");
    Check(cuMemcpyHtoD(bias.payload(), bias_before.data(), cp::kBiasBytes),
          "copy bias");
    Check(cuMemcpyHtoD(input.payload(), input_before.data(), cp::kInputBytes),
          "copy input");
    Check(cuMemsetD8(output.payload(), poison, cp::kOutputBytes),
          "poison output");

    CUdeviceptr weight_pointer = weights.payload();
    CUdeviceptr bias_pointer = bias.payload();
    CUdeviceptr input_pointer = input.payload();
    CUdeviceptr output_pointer = output.payload();
    std::uint32_t output_channels = 32;
    std::uint32_t kernel_height = 3;
    std::uint32_t kernel_width = 3;
    std::uint32_t packed_input_channels = 10;
    std::uint32_t input_batch = 1;
    std::uint32_t input_height = 368;
    std::uint32_t input_width = 640;
    std::uint32_t input_channels = 10;
    std::uint32_t output_batch = 1;
    std::uint32_t output_height = 184;
    std::uint32_t output_width = 320;
    std::uint32_t output_stride_channels = 32;
    void* parameters[] = {
        &weight_pointer, &bias_pointer, &output_channels,
        &kernel_height, &kernel_width, &packed_input_channels,
        &input_pointer, &input_batch, &input_height, &input_width,
        &input_channels, &output_pointer, &output_batch, &output_height,
        &output_width, &output_stride_channels,
    };
    Check(cuLaunchKernel(function, 80, 92, 1, 32, 2, 4, 0, nullptr,
                         parameters, nullptr), "launch convPre");
    Check(cuCtxSynchronize(), "synchronize convPre");

    const auto guarded_weights = weights.ReadAll();
    const auto guarded_bias = bias.ReadAll();
    const auto guarded_input = input.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto weights_after = Payload(guarded_weights, cp::kWeightBytes);
    const auto bias_after = Payload(guarded_bias, cp::kBiasBytes);
    const auto input_after = Payload(guarded_input, cp::kInputBytes);
    const auto actual_output = Payload(guarded_output, cp::kOutputBytes);
    const std::size_t mismatches =
        Mismatches(weights_before, weights_after) +
        Mismatches(bias_before, bias_after) +
        Mismatches(input_before, input_after);
    const std::size_t canaries =
        CanaryMismatches(guarded_weights, cp::kWeightBytes) +
        CanaryMismatches(guarded_bias, cp::kBiasBytes) +
        CanaryMismatches(guarded_input, cp::kInputBytes) +
        CanaryMismatches(guarded_output, cp::kOutputBytes);
    const std::size_t collisions = UntouchedHalfwords(actual_output, poison);
    Write(output_directory / "input-after.bin", input_after);
    Write(output_directory / "output.bin", actual_output);
    WriteConvPreReceipt(output_directory / "run-receipt.json", argv[5],
                        poison, weights_before, bias_before, input_before,
                        weights_after, bias_after, input_after, actual_output,
                        mismatches, canaries, collisions);
    const bool pass = mismatches == 0 && canaries == 0;
    std::cout << "probe=fg_custom_conv_pre_rtx\noutput_sha256="
              << Sha256(actual_output) << "\nresult="
              << (pass ? "pass" : "fail") << '\n';
    weights.Free(); bias.Free(); input.Free(); output.Free();
    Check(cuModuleUnload(module), "unload module");
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
