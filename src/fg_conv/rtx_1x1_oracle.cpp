#define FG_CONV_L20_NO_MAIN
#include "rtx_oracle.cpp"
#include "conv_1x1_contract.h"

namespace {

constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

void WriteFamilyReceipt(
    const std::filesystem::path& path, const fg_conv_1x1::Layer& layer,
    const std::string& run_id, std::uint8_t poison,
    const std::vector<std::uint8_t>& weights_before,
    const std::vector<std::uint8_t>& input_before,
    const std::vector<std::uint8_t>& weights_after,
    const std::vector<std::uint8_t>& input_after,
    const std::vector<std::uint8_t>& output,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t poison_collisions) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write family receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-1x1-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"layer_id\": \"" << layer.id << "\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": " << layer.module_token << ",\n"
         << "  \"module_bytes\": " << layer.module_bytes << ",\n"
         << "  \"module_sha256\": \"" << layer.module_sha256 << "\",\n"
         << "  \"function_name\": \"k_conv_fp16_nhwc\",\n"
         << "  \"trace_launch_token\": \"launch-" << std::setw(6)
         << std::setfill('0') << layer.launch << "\",\n"
         << std::setfill(' ')
         << "  \"grid\": [" << layer.grid[0] << ", " << layer.grid[1]
         << ", " << layer.grid[2] << "],\n"
         << "  \"block\": [" << layer.block[0] << ", " << layer.block[1]
         << ", " << layer.block[2] << "],\n"
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
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-conv-1x1-rtx <layer-id> <module> <weights> "
                 "<output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const auto& layer = fg_conv_1x1::FindLayer(argv[1]);
    const auto module_bytes = Read(argv[2]);
    const auto weights_before = Read(argv[3]);
    if (module_bytes.size() != layer.module_bytes ||
        Sha256(module_bytes) != layer.module_sha256) {
      throw std::runtime_error("selected module identity mismatch");
    }
    if (weights_before.size() != layer.weight_bytes ||
        Sha256(weights_before) != layer.weight_sha256) {
      throw std::runtime_error("live weight identity mismatch");
    }
    const std::size_t active_half =
        std::size_t{layer.input_channels} * layer.output_channels;
    if (layer.weight_bytes != active_half * 4 +
                                  std::size_t{layer.output_channels} * 32) {
      throw std::runtime_error("packed weight geometry mismatch");
    }
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[6], &poison_end, 16);
    if (poison_end == argv[6] || *poison_end != '\0' || parsed_poison > 255) {
      throw std::runtime_error("invalid poison");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::filesystem::path output_directory = argv[4];
    std::filesystem::create_directories(output_directory);
    const auto input_before = fg_conv_1x1::GenerateInput(layer);
    const std::size_t output_bytes = fg_conv_1x1::OutputBytes(layer);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "k_conv_fp16_nhwc"),
          "get function");

    GuardedDeviceBuffer weights;
    GuardedDeviceBuffer input;
    GuardedDeviceBuffer output;
    weights.Allocate(layer.weight_bytes);
    input.Allocate(input_before.size());
    output.Allocate(output_bytes);
    Check(cuMemcpyHtoD(weights.payload(), weights_before.data(),
                       weights_before.size()), "copy weights");
    Check(cuMemcpyHtoD(input.payload(), input_before.data(), input_before.size()),
          "copy input");
    Check(cuMemsetD8(output.payload(), poison, output_bytes), "poison output");

    CUdeviceptr weight_pointer = weights.payload();
    CUdeviceptr input_pointer = input.payload();
    CUdeviceptr output_pointer = output.payload();
    std::uint32_t width = layer.width;
    std::uint32_t height = layer.height;
    std::uint32_t output_width = layer.width;
    std::uint32_t output_height = layer.height;
    void* parameters[] = {&weight_pointer, &input_pointer, &width, &height,
                          &output_pointer, &output_width, &output_height};
    Check(cuLaunchKernel(function, layer.grid[0], layer.grid[1], layer.grid[2],
                         layer.block[0], layer.block[1], layer.block[2], 0,
                         nullptr, parameters, nullptr), "launch selected 1x1");
    Check(cuCtxSynchronize(), "synchronize selected 1x1");

    const auto guarded_weights = weights.ReadAll();
    const auto guarded_input = input.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto weights_after = Payload(guarded_weights, layer.weight_bytes);
    const auto input_after = Payload(guarded_input, input_before.size());
    const auto actual_output = Payload(guarded_output, output_bytes);
    const std::size_t mismatches =
        Mismatches(weights_before, weights_after) +
        Mismatches(input_before, input_after);
    const std::size_t canaries =
        CanaryMismatches(guarded_weights, layer.weight_bytes) +
        CanaryMismatches(guarded_input, input_before.size()) +
        CanaryMismatches(guarded_output, output_bytes);
    const std::size_t collisions = UntouchedHalfwords(actual_output, poison);
    Write(output_directory / "input-after.bin", input_after);
    Write(output_directory / "output.bin", actual_output);
    WriteFamilyReceipt(output_directory / "run-receipt.json", layer, argv[5],
                       poison, weights_before, input_before, weights_after,
                       input_after, actual_output, mismatches, canaries,
                       collisions);
    const bool pass = mismatches == 0 && canaries == 0;
    std::cout << "probe=fg_conv_1x1_rtx\nlayer_id=" << layer.id
              << "\noutput_sha256=" << Sha256(actual_output)
              << "\nresult=" << (pass ? "pass" : "fail") << '\n';
    weights.Free(); input.Free(); output.Free();
    Check(cuModuleUnload(module), "unload module");
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
