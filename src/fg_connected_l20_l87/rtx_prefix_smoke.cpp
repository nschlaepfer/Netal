#define main FgMidL55UnusedMain
#include "../fg_mid_l55/rtx_oracle.cpp"
#undef main

#include "connected_l20_l87_bridge.h"
#include "rtx_shared_cuda_buffer.h"

using fg_connected_l20_l87::SharedCudaBuffer;
using fg_connected_l20_l87::UploadSharedCudaBuffer;

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-connected-prefix-smoke <modules> <model> "
                 "<output-dir> <poison-hex>\n";
    return 2;
  }
  try {
    const auto poison = ParsePoison(argv[4]);
    const fs::path output_directory = argv[3];
    fs::create_directories(output_directory);
    constexpr std::size_t kL51Bytes = 160u * 96u * 16u * 2u;
    auto initial = PoisonGuarded(kL51Bytes, poison);
    GpuContext gpu = CreateGpuContext();
    SharedCudaBuffer l51 = UploadSharedCudaBuffer(gpu, initial);
    ExecuteAndReset(gpu);
    FgConnectedPrefixResult result{};
    const int status = RunFgConnectedPrefixL20L51(
        argv[1], argv[2], poison,
        static_cast<std::uint64_t>(l51.cuda_ptr + kGuardBytes), &result);
    if (status != 0) throw std::runtime_error("connected prefix failed");
    auto guarded = ReadBuffer(gpu, l51.d3d);
    auto payload = Payload(guarded, kL51Bytes);
    const auto canaries = CanaryMismatches(guarded, kL51Bytes);
    Write(output_directory / "l51.bin", payload);
    std::ofstream receipt(output_directory / "run-receipt.json");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.connected-l20-l51-bridge-smoke.v1\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"l51_bytes\": " << payload.size() << ",\n"
            << "  \"l51_sha256\": \"" << Sha256(payload) << "\",\n"
            << "  \"prefix_input_model_mismatched_bytes\": "
            << result.input_model_mismatched_bytes << ",\n"
            << "  \"prefix_canary_mismatched_bytes\": "
            << result.canary_mismatched_bytes << ",\n"
            << "  \"handoff_canary_mismatched_bytes\": " << canaries
            << ",\n"
            << "  \"live_device_handoff\": true,\n"
            << "  \"host_intermediate_ingress\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \""
            << (canaries == 0 ? "pass" : "fail") << "\"\n}\n";
    std::cout << "l51_sha256=" << Sha256(payload) << "\n"
              << "handoff_canary_mismatched_bytes=" << canaries << "\n"
              << "result=" << (canaries == 0 ? "pass" : "fail") << "\n";
    l51.Release();
    return canaries == 0 ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
