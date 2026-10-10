#include <cuda.h>

#include <stdexcept>

#include "../fg_connected_custom_l56_l83/contract_full.h"
#include "../fg_connected_tail_l84_l87/core_bridge.h"

namespace {

struct LiveCopyState {
  bool active = false;
  std::size_t host_to_device_call = 0;
  std::array<CUdeviceptr, 3> input{};
};

thread_local LiveCopyState g_live_copy;
thread_local fg_connected_custom::ExternalInputs g_live_snapshot;

CUresult FgConnectedLiveMemcpyHtoD(CUdeviceptr destination,
                                    const void* source,
                                    std::size_t bytes) {
  const std::size_t call = g_live_copy.host_to_device_call++;
  if (!g_live_copy.active || call < 50u)
    return cuMemcpyHtoD(destination, source, bytes);
  const std::size_t input_index = call - 50u;
  constexpr std::array<std::size_t, 3> kExpectedBytes = {
      fg_connected_custom::kInputABytes,
      fg_connected_custom::kInputBBytes,
      fg_connected_custom::kInputCBytes};
  if (input_index >= g_live_copy.input.size() ||
      bytes != kExpectedBytes[input_index])
    return CUDA_ERROR_INVALID_VALUE;
  return cuMemcpyDtoD(destination, g_live_copy.input[input_index], bytes);
}

std::vector<std::uint8_t> ReadDevice(CUdeviceptr source, std::size_t bytes) {
  std::vector<std::uint8_t> result(bytes);
  if (cuMemcpyDtoH(result.data(), source, bytes) != CUDA_SUCCESS)
    throw std::runtime_error("connected live input readback failed");
  return result;
}

std::size_t MismatchCount(const std::vector<std::uint8_t>& a,
                          const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size())
    throw std::runtime_error("connected live input comparison size drift");
  std::size_t result = 0;
  for (std::size_t index = 0; index < a.size(); ++index)
    result += a[index] != b[index];
  return result;
}

}  // namespace

namespace fg_connected_custom {

inline ExternalInputs FgConnectedLiveGenerate() {
  if (!g_live_copy.active)
    throw std::runtime_error("live connected input generator used out of scope");
  return g_live_snapshot;
}

}  // namespace fg_connected_custom

#define RunFgConnectedCoreL56L83 RunFgAcceptedCoreL56L83
#define Generate FgConnectedLiveGenerate
#undef cuMemcpyHtoD
#define cuMemcpyHtoD FgConnectedLiveMemcpyHtoD
#include "../fg_connected_tail_l84_l87/rtx_core_bridge.cpp"
#undef cuMemcpyHtoD
#undef Generate
#undef RunFgConnectedCoreL56L83

#include "rtx_live_core_bridge.h"

int RunFgConnectedCoreL56L83Live(
    const char* module_directory, const char* model_directory,
    std::uint8_t poison, std::uint64_t input_a_cuda_ptr,
    std::uint64_t input_b_cuda_ptr, std::uint64_t input_c_cuda_ptr,
    std::uint64_t l83_output_cuda_ptr,
    FgConnectedCoreBridgeResult* result) {
  if (module_directory == nullptr || model_directory == nullptr ||
      input_a_cuda_ptr == 0 || input_b_cuda_ptr == 0 ||
      input_c_cuda_ptr == 0 || l83_output_cuda_ptr == 0 || result == nullptr)
    return 2;
  try {
    Check(cuInit(0), "initialize live connected bridge");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "get live connected device");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "retain live connected context");
    Check(cuCtxSetCurrent(context), "set live connected context");
    const std::array<CUdeviceptr, 3> inputs = {
        static_cast<CUdeviceptr>(input_a_cuda_ptr),
        static_cast<CUdeviceptr>(input_b_cuda_ptr),
        static_cast<CUdeviceptr>(input_c_cuda_ptr)};
    g_live_snapshot = {
        ReadDevice(inputs[0], fg_connected_custom::kInputABytes),
        ReadDevice(inputs[1], fg_connected_custom::kInputBBytes),
        ReadDevice(inputs[2], fg_connected_custom::kInputCBytes)};
    g_live_copy = {true, 0, inputs};
    const int status = RunFgAcceptedCoreL56L83(
        module_directory, model_directory, poison, l83_output_cuda_ptr,
        result);
    const auto call_count = g_live_copy.host_to_device_call;
    g_live_copy = {};
    const fg_connected_custom::ExternalInputs after = {
        ReadDevice(inputs[0], fg_connected_custom::kInputABytes),
        ReadDevice(inputs[1], fg_connected_custom::kInputBBytes),
        ReadDevice(inputs[2], fg_connected_custom::kInputCBytes)};
    result->input_model_mismatched_bytes +=
        MismatchCount(g_live_snapshot.a, after.a) +
        MismatchCount(g_live_snapshot.b, after.b) +
        MismatchCount(g_live_snapshot.c, after.c);
    Check(cuDevicePrimaryCtxRelease(device),
          "release live connected context");
    if (call_count != 53u)
      throw std::runtime_error("accepted core HtoD call contract drift");
    if (status != 0 || result->input_model_mismatched_bytes != 0 ||
        result->canary_mismatched_bytes != 0)
      return 3;
    return 0;
  } catch (const std::exception& error) {
    g_live_copy = {};
    std::cerr << "live_core_error=" << error.what() << '\n';
    return 1;
  }
}
