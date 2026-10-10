#include <windows.h>
#include <bcrypt.h>
#include <cuda.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "contract.h"

#pragma comment(lib, "bcrypt.lib")

namespace {
namespace contract = fg_early_status_l10_l12;
constexpr std::size_t kGuardBytes = 256;

void Check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + ": " +
                           (name ? name : "unknown") + " / " +
                           (description ? description : "unknown"));
}

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0, digest_bytes = 0, written = 0;
  auto require = [](NTSTATUS status, const char* operation) {
    if (status < 0) throw std::runtime_error(std::string(operation) + " failed");
  };
  require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                      nullptr, 0),
          "BCryptOpenAlgorithmProvider");
  require(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                            reinterpret_cast<PUCHAR>(&object_bytes),
                            sizeof(object_bytes), &written, 0),
          "BCryptGetProperty(object)");
  require(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                            reinterpret_cast<PUCHAR>(&digest_bytes),
                            sizeof(digest_bytes), &written, 0),
          "BCryptGetProperty(digest)");
  std::vector<std::uint8_t> object(object_bytes), digest(digest_bytes);
  require(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes,
                           nullptr, 0, 0),
          "BCryptCreateHash");
  require(BCryptHashData(hash, const_cast<PUCHAR>(bytes.data()),
                         static_cast<ULONG>(bytes.size()), 0),
          "BCryptHashData");
  require(BCryptFinishHash(hash, digest.data(), digest_bytes, 0),
          "BCryptFinishHash");
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison") ;
  return static_cast<std::uint8_t>(value);
}

struct Parameters {
  CUdeviceptr output;
  std::uint32_t count;
  std::uint32_t value;
};
static_assert(sizeof(Parameters) == 16);

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fg-early-status-rtx <module> <output-dir> <run-id> <poison-hex> <reset|generated>\n";
    return 2;
  }
  try {
    const std::string mode = argv[5];
    if (mode != "reset" && mode != "generated")
      throw std::runtime_error("unsupported status mode");
    const std::uint8_t poison = ParsePoison(argv[4]);
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256)
      throw std::runtime_error("module identity mismatch");
    const std::uint32_t value = mode == "reset" ? contract::kResetValue
                                                  : contract::kGeneratedValue;
    const char* trace_raw_sha =
        mode == "reset" ? contract::kResetTraceRawParameterSha256
                         : contract::kGeneratedTraceRawParameterSha256;
    const char* address_free_sha =
        mode == "reset" ? contract::kResetAddressFreeParameterSha256
                         : contract::kGeneratedAddressFreeParameterSha256;

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module27");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get main_kernel");
    CUdeviceptr allocation = 0;
    constexpr std::size_t payload_bytes = sizeof(std::uint32_t);
    Check(cuMemAlloc(&allocation, payload_bytes + 2u * kGuardBytes),
          "allocate guarded output");
    Check(cuMemsetD8(allocation, poison, payload_bytes + 2u * kGuardBytes),
          "poison guarded output");
    Parameters parameters{allocation + kGuardBytes, contract::kCount, value};
    const Parameters address_free_parameters{0, contract::kCount, value};
    const auto* address_free_begin = reinterpret_cast<const std::uint8_t*>(
        &address_free_parameters);
    const std::vector<std::uint8_t> address_free_parameter_bytes(
        address_free_begin, address_free_begin + sizeof(address_free_parameters));
    const std::string live_address_free_sha =
        Sha256(address_free_parameter_bytes);
    if (live_address_free_sha != address_free_sha)
      throw std::runtime_error("live address-free parameter identity mismatch");
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, nullptr, nullptr,
                         extra),
          "launch module27");
    Check(cuCtxSynchronize(), "synchronize module27");
    std::vector<std::uint8_t> guarded(payload_bytes + 2u * kGuardBytes);
    Check(cuMemcpyDtoH(guarded.data(), allocation, guarded.size()),
          "read guarded output");
    std::vector<std::uint8_t> output(
        guarded.begin() + kGuardBytes,
        guarded.begin() + kGuardBytes + payload_bytes);
    std::uint32_t actual = 0;
    std::memcpy(&actual, output.data(), sizeof(actual));
    std::size_t canary_mismatches = 0;
    for (std::size_t index = 0; index < kGuardBytes; ++index) {
      canary_mismatches += guarded[index] != poison;
      canary_mismatches +=
          guarded[kGuardBytes + payload_bytes + index] != poison;
    }
    std::size_t untouched_payload_bytes = 0;
    for (const auto byte : output) untouched_payload_bytes += byte == poison;
    const bool pass = actual == value && canary_mismatches == 0 &&
                      untouched_payload_bytes == 0;
    Write(output_root / "output-u32.bin", output);
    std::ofstream receipt(output_root / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.early-status-l10-l12-run.v1\",\n"
            << "  \"backend\": \"rtx-cuda-driver\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"mode\": \"" << mode << "\",\n"
            << "  \"canonical_launch\": " << (mode == "reset" ? 10 : 12)
            << ",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"module_token\": 27,\n"
            << "  \"module_bytes\": " << module_bytes.size() << ",\n"
            << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
            << "  \"trace_raw_parameter_sha256\": \"" << trace_raw_sha << "\",\n"
            << "  \"live_address_free_parameter_sha256\": \""
            << live_address_free_sha << "\",\n"
            << "  \"address_free_parameter_sha256\": \"" << address_free_sha
            << "\",\n"
            << "  \"packed_parameter_bytes\": 16,\n"
            << "  \"grid\": [1, 1, 1],\n"
            << "  \"block\": [64, 1, 1],\n"
            << "  \"tensor_input_count\": 0,\n"
            << "  \"input_generator\": \"" << contract::kInputGenerator
            << "\",\n"
            << "  \"element_count\": 1,\n"
            << "  \"requested_value\": " << value << ",\n"
            << "  \"actual_value\": " << actual << ",\n"
            << "  \"output_bytes\": 4,\n"
            << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches
            << ",\n"
            << "  \"untouched_payload_bytes\": " << untouched_payload_bytes
            << ",\n"
            << "  \"expected_output_fed_to_execution\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
            << "}\n";
    std::cout << "probe=fg_early_status_l10_l12_rtx\n"
              << "run_id=" << run_id << '\n'
              << "mode=" << mode << '\n'
              << "actual_value=" << actual << '\n'
              << "output_sha256=" << Sha256(output) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    Check(cuMemFree(allocation), "free output");
    Check(cuModuleUnload(module), "unload module27");
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
