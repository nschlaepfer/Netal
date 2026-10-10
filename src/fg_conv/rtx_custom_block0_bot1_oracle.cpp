#define FG_CONV_L20_NO_MAIN
#include "rtx_oracle.cpp"

#include "custom_block0_bot1_contract.h"

namespace {
namespace contract = fg_custom_block0_bot1;

struct Parameters {
  CUdeviceptr weights;
  CUdeviceptr bias;
  std::uint32_t output_channels;
  std::uint32_t kernel_height;
  std::uint32_t kernel_width;
  std::uint32_t input_channels;
  CUdeviceptr input0;
  CUdeviceptr input1;
  CUdeviceptr input2;
  std::uint32_t input_batch;
  std::uint32_t input_height;
  std::uint32_t input_width;
  std::uint32_t input_stride;
  CUdeviceptr output0;
  std::uint32_t output0_batch;
  std::uint32_t output0_height;
  std::uint32_t output0_width;
  std::uint32_t output0_stride;
  CUdeviceptr output1;
  std::uint32_t output1_batch;
  std::uint32_t output1_height;
  std::uint32_t output1_width;
  std::uint32_t output1_stride;
  CUdeviceptr output2;
  std::uint32_t output2_batch;
  std::uint32_t output2_height;
  std::uint32_t output2_width;
  std::uint32_t output2_stride;
};
static_assert(sizeof(Parameters) == 144);

void Record(const std::filesystem::path &path, const std::string &run_id,
            std::uint8_t poison, const std::vector<std::uint8_t> &weights0,
            const std::vector<std::uint8_t> &bias0,
            const std::array<std::vector<std::uint8_t>, 3> &inputs0,
            const std::vector<std::uint8_t> &weights1,
            const std::vector<std::uint8_t> &bias1,
            const std::array<std::vector<std::uint8_t>, 3> &inputs1,
            const std::vector<std::uint8_t> &output,
            std::size_t input_mismatches, std::size_t canary_mismatches) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path);
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.custom-block0-bot1-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"trace_launch_token\": \"launch-000069\",\n"
         << "  \"module_token\": 59,\n"
         << "  \"module_bytes\": " << contract::kModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"custom_block0_conv_bot1_hf_kernel\",\n"
         << "  \"grid\": [40, 23, 3],\n"
         << "  \"block\": [32, 4, 1],\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"weight_source_token\": 154,\n"
         << "  \"bias_source_token\": 156,\n"
         << "  \"model_manifest_sha256\": \"" << contract::kManifestSha256
         << "\",\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(weights0) << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(weights1) << "\",\n"
         << "  \"bias_pre_sha256\": \"" << Sha256(bias0) << "\",\n"
         << "  \"bias_post_sha256\": \"" << Sha256(bias1) << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-custom-block0-bot1-v1\",\n"
         << "  \"input_pre_sha256\": [\"" << Sha256(inputs0[0]) << "\", \""
         << Sha256(inputs0[1]) << "\", \"" << Sha256(inputs0[2]) << "\"],\n"
         << "  \"input_post_sha256\": [\"" << Sha256(inputs1[0]) << "\", \""
         << Sha256(inputs1[1]) << "\", \"" << Sha256(inputs1[2]) << "\"],\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"external_model_files_opened\": true,\n"
         << "  \"expected_output_file_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) return 2;
  try {
    const auto module = Read(argv[1]);
    const auto weights0 = Read(argv[2]);
    const auto bias0 = Read(argv[3]);
    if (module.size() != contract::kModuleBytes ||
        Sha256(module) != contract::kModuleSha256 ||
        weights0.size() != contract::kWeightBytes ||
        Sha256(weights0) != contract::kWeightSha256 ||
        bias0.size() != contract::kBiasBytes ||
        Sha256(bias0) != contract::kBiasSha256) {
      throw std::runtime_error("identity");
    }
    char *end = nullptr;
    const auto poison_value = strtoul(argv[6], &end, 16);
    if (end == argv[6] || *end || poison_value > 255)
      throw std::runtime_error("poison");
    const auto poison = static_cast<std::uint8_t>(poison_value);
    const std::filesystem::path output_root = argv[4];
    std::filesystem::create_directories(output_root);
    const std::array<std::vector<std::uint8_t>, 3> inputs0 = {
        contract::Input0(), contract::Input1(), contract::Input2()};

    Check(cuInit(0), "cuInit");
    CUdevice device;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule loaded_module;
    Check(cuModuleLoadData(&loaded_module, module.data()), "load module");
    CUfunction function;
    Check(cuModuleGetFunction(&function, loaded_module,
                              "custom_block0_conv_bot1_hf_kernel"),
          "get function");

    GuardedDeviceBuffer weights, bias, input0, input1, input2, output0,
        output1, output2;
    weights.Allocate(contract::kWeightBytes);
    bias.Allocate(contract::kBiasBytes);
    input0.Allocate(contract::kInputPlaneBytes);
    input1.Allocate(contract::kInputPlaneBytes);
    input2.Allocate(contract::kInputPlaneBytes);
    output0.Allocate(contract::kOutput0Bytes);
    output1.Allocate(contract::kOutput1Bytes);
    output2.Allocate(contract::kOutput2Bytes);
    Check(cuMemcpyHtoD(weights.payload(), weights0.data(), weights0.size()),
          "weights");
    Check(cuMemcpyHtoD(bias.payload(), bias0.data(), bias0.size()), "bias");
    Check(cuMemcpyHtoD(input0.payload(), inputs0[0].data(), inputs0[0].size()),
          "input0");
    Check(cuMemcpyHtoD(input1.payload(), inputs0[1].data(), inputs0[1].size()),
          "input1");
    Check(cuMemcpyHtoD(input2.payload(), inputs0[2].data(), inputs0[2].size()),
          "input2");
    Check(cuMemsetD8(output0.payload(), poison, contract::kOutput0Bytes),
          "poison output0");
    Check(cuMemsetD8(output1.payload(), poison, contract::kOutput1Bytes),
          "poison output1");
    Check(cuMemsetD8(output2.payload(), poison, contract::kOutput2Bytes),
          "poison output2");

    Parameters parameters{weights.payload(), bias.payload(), 8, 3, 3, 32,
                          input0.payload(), input1.payload(), input2.payload(),
                          1, 92, 160, 32,
                          output0.payload(), 1, 184, 320, 4,
                          output1.payload(), 1, 184, 320, 1,
                          output2.payload(), 1, 184, 320, 3};
    std::size_t parameter_bytes = sizeof(parameters);
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 40, 23, 3, 32, 4, 1, 0, nullptr, nullptr,
                         extra),
          "launch");
    Check(cuCtxSynchronize(), "synchronize");

    const auto weights_guarded = weights.ReadAll();
    const auto bias_guarded = bias.ReadAll();
    const std::array<std::vector<std::uint8_t>, 3> inputs_guarded = {
        input0.ReadAll(), input1.ReadAll(), input2.ReadAll()};
    const auto output0_guarded = output0.ReadAll();
    const auto output1_guarded = output1.ReadAll();
    const auto output2_guarded = output2.ReadAll();
    const auto weights1 = Payload(weights_guarded, contract::kWeightBytes);
    const auto bias1 = Payload(bias_guarded, contract::kBiasBytes);
    const std::array<std::vector<std::uint8_t>, 3> inputs1 = {
        Payload(inputs_guarded[0], contract::kInputPlaneBytes),
        Payload(inputs_guarded[1], contract::kInputPlaneBytes),
        Payload(inputs_guarded[2], contract::kInputPlaneBytes)};
    const auto result0 = Payload(output0_guarded, contract::kOutput0Bytes);
    const auto result1 = Payload(output1_guarded, contract::kOutput1Bytes);
    const auto result2 = Payload(output2_guarded, contract::kOutput2Bytes);
    std::vector<std::uint8_t> output;
    output.reserve(contract::kOutputBytes);
    output.insert(output.end(), result0.begin(), result0.end());
    output.insert(output.end(), result1.begin(), result1.end());
    output.insert(output.end(), result2.begin(), result2.end());
    const auto mismatches =
        Mismatches(weights0, weights1) + Mismatches(bias0, bias1) +
        Mismatches(inputs0[0], inputs1[0]) +
        Mismatches(inputs0[1], inputs1[1]) +
        Mismatches(inputs0[2], inputs1[2]);
    const auto canaries =
        CanaryMismatches(weights_guarded, contract::kWeightBytes) +
        CanaryMismatches(bias_guarded, contract::kBiasBytes) +
        CanaryMismatches(inputs_guarded[0], contract::kInputPlaneBytes) +
        CanaryMismatches(inputs_guarded[1], contract::kInputPlaneBytes) +
        CanaryMismatches(inputs_guarded[2], contract::kInputPlaneBytes) +
        CanaryMismatches(output0_guarded, contract::kOutput0Bytes) +
        CanaryMismatches(output1_guarded, contract::kOutput1Bytes) +
        CanaryMismatches(output2_guarded, contract::kOutput2Bytes);
    Record(output_root / "run-receipt.json", argv[5], poison, weights0, bias0,
           inputs0, weights1, bias1, inputs1, output, mismatches, canaries);
    Write(output_root / "output.bin", output);
    const bool pass = mismatches == 0 && canaries == 0;
    std::cout << "output_sha256=" << Sha256(output) << "\n"
              << "result=" << (pass ? "pass" : "fail") << "\n";
    return pass ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << "error=" << error.what() << "\n";
    return 1;
  }
}
