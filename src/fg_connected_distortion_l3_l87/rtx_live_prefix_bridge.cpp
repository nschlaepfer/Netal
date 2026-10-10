#define main FgAcceptedConnectedUnetL20L51UnusedMain
#include "../fg_connected_unet_l20_l51/rtx_oracle.cpp"
#undef main

#include "../fg_connected_l20_l87/connected_l20_l87_bridge.h"
#include "rtx_live_prefix_bridge.h"

namespace {

thread_local CUdeviceptr g_live_prefix_input = 0;

std::vector<std::uint8_t> ReadLivePrefixInput(CUdeviceptr source,
                                              std::size_t bytes) {
  std::vector<std::uint8_t> result(bytes);
  Check(cuMemcpyDtoH(result.data(), source, bytes),
        "read live L19 allocation37");
  return result;
}

}  // namespace

void FgConnectedSetRtxLivePrefixInput(std::uint64_t input_cuda_ptr) {
  g_live_prefix_input = static_cast<CUdeviceptr>(input_cuda_ptr);
}

int RunFgConnectedPrefixL20L51(const char* module_directory,
                               const char* model_directory,
                               std::uint8_t poison,
                               std::uint64_t l51_output_cuda_ptr,
                               FgConnectedPrefixResult* result) {
  try {
    if (module_directory == nullptr || model_directory == nullptr ||
        g_live_prefix_input == 0 || l51_output_cuda_ptr == 0 ||
        result == nullptr) {
      throw std::runtime_error("invalid live connected-prefix argument");
    }
    const std::filesystem::path module_root = module_directory;
    const std::filesystem::path model_root = model_directory;
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

    Check(cuInit(0), "cuInit live prefix");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet live prefix");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "retain live prefix context");
    Check(cuCtxSetCurrent(context), "set live prefix context");
    const auto input_before = ReadLivePrefixInput(g_live_prefix_input,
                                                  kInputBytes);

    std::array<LoadedKernel, 53> kernels{};
    for (const auto& module : kModules) {
      std::ostringstream name;
      name << "cu-module-" << std::setw(6) << std::setfill('0')
           << module.token << ".bin";
      kernels[module.token] = LoadKernel(module_root / name.str(),
                                         module.bytes, module.sha256,
                                         module.function);
    }

    std::vector<std::vector<std::uint8_t>> weights_before;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> weights;
    for (const auto& weight : kWeights) {
      std::ostringstream name;
      name << "default-token-" << std::setw(6) << std::setfill('0')
           << weight.token << ".bin";
      auto bytes = Read(model_root / name.str());
      if (bytes.size() != weight.bytes || Sha256(bytes) != weight.sha256)
        throw std::runtime_error("prefix model-state identity mismatch");
      auto buffer = Allocate(bytes.size());
      Check(cuMemcpyHtoD(buffer->payload(), bytes.data(), bytes.size()),
            "copy prefix model state");
      weights_before.push_back(std::move(bytes));
      weights.push_back(std::move(buffer));
    }
    auto input = Allocate(kInputBytes);
    Check(cuMemcpyDtoD(input->payload(), g_live_prefix_input, kInputBytes),
          "live L19 to L20 device handoff");
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> tensors;
    for (const auto& spec : kTensors) {
      auto tensor = Allocate(TensorBytes(spec));
      Check(cuMemsetD8(tensor->payload(), poison, TensorBytes(spec)),
            "poison prefix tensor");
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
    LaunchAdd(kernels[37], *tensors[12], *tensors[8], *tensors[13],
              10, 6, 256, 3, 2, 8);
    Launch1x1(l34, kernels[38], *weights[7], *tensors[13], *tensors[14]);
    Launch1x1(l35, kernels[39], *weights[8], *tensors[14], *tensors[15]);
    LaunchUpscale(kernels[40], *tensors[15], *tensors[16], 10, 6, 20, 12,
                  {1, 3, 2}, {16, 8, 8});
    LaunchAdd(kernels[41], *tensors[16], *tensors[6], *tensors[17],
              20, 12, 128, 5, 3, 8);
    Launch1x1(l38, kernels[42], *weights[9], *tensors[17], *tensors[18]);
    Launch1x1(l39, kernels[42], *weights[10], *tensors[18], *tensors[19]);
    LaunchUpscale(kernels[40], *tensors[19], *tensors[20], 20, 12, 40, 24,
                  {1, 5, 3}, {16, 8, 8});
    LaunchAdd(kernels[41], *tensors[20], *tensors[4], *tensors[21],
              40, 24, 128, 10, 6, 8);
    Launch1x1(l42, kernels[43], *weights[11], *tensors[21], *tensors[22]);
    Launch1x1(l43, kernels[44], *weights[12], *tensors[22], *tensors[23]);
    LaunchUpscale(kernels[45], *tensors[23], *tensors[24], 40, 24, 80, 48,
                  {1, 8, 5}, {8, 11, 11});
    LaunchAdd(kernels[46], *tensors[24], *tensors[2], *tensors[25],
              80, 48, 64, 20, 12, 8);
    Launch1x1(l46, kernels[47], *weights[13], *tensors[25], *tensors[26]);
    Launch1x1(l47, kernels[48], *weights[14], *tensors[26], *tensors[27]);
    LaunchUpscale(kernels[49], *tensors[27], *tensors[28], 80, 48, 160, 96,
                  {1, 10, 6}, {4, 16, 16});
    LaunchAdd(kernels[50], *tensors[28], *tensors[0], *tensors[29],
              160, 96, 32, 40, 24, 4);
    Launch1x1(l50, kernels[51], *weights[15], *tensors[29], *tensors[30]);
    Launch1x1(l51, kernels[52], *weights[16], *tensors[30], *tensors[31]);
    Check(cuMemcpyDtoD(static_cast<CUdeviceptr>(l51_output_cuda_ptr),
                       tensors[31]->payload(), TensorBytes(kTensors[31])),
          "live L51 device handoff");
    Check(cuCtxSynchronize(), "synchronize live connected prefix");

    const auto input_guarded = input->ReadAll();
    const auto input_after = Payload(input_guarded, kInputBytes);
    const auto live_input_after = ReadLivePrefixInput(g_live_prefix_input,
                                                      kInputBytes);
    std::size_t mismatches = Mismatches(input_before, input_after) +
                             Mismatches(input_before, live_input_after);
    std::size_t canaries = CanaryMismatches(input_guarded, kInputBytes);
    for (std::size_t index = 0; index < weights.size(); ++index) {
      const auto guarded = weights[index]->ReadAll();
      const auto after = Payload(guarded, kWeights[index].bytes);
      mismatches += Mismatches(weights_before[index], after);
      canaries += CanaryMismatches(guarded, kWeights[index].bytes);
    }
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      const auto guarded = tensors[index]->ReadAll();
      const auto bytes = Payload(guarded, TensorBytes(kTensors[index]));
      result->checkpoint_sha256[index] = Sha256(bytes);
      canaries += CanaryMismatches(guarded, TensorBytes(kTensors[index]));
    }
    result->external_input_sha256 = Sha256(input_before);
    result->input_model_mismatched_bytes = mismatches;
    result->canary_mismatched_bytes = canaries;

    for (auto& tensor : tensors) tensor->Free();
    input->Free();
    for (auto& weight : weights) weight->Free();
    for (auto module = kModules.rbegin(); module != kModules.rend(); ++module)
      Check(cuModuleUnload(kernels[module->token].module),
            "unload prefix module");
    Check(cuDevicePrimaryCtxRelease(device), "release live prefix context");
    return mismatches == 0 && canaries == 0 ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "live_prefix_error=" << error.what() << '\n';
    return 1;
  }
}
