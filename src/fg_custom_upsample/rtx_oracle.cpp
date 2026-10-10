#include <windows.h>
#include <bcrypt.h>
#include <cuda.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace {

constexpr std::uint32_t kInputHeight = 184;
constexpr std::uint32_t kInputWidth = 320;
constexpr std::uint32_t kOutputHeight = 368;
constexpr std::uint32_t kOutputWidth = 640;
constexpr std::array<std::uint32_t, 3> kChannels = {4, 1, 3};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kExpectedModuleBytes = 2112;
constexpr char kExpectedModuleSha256[] =
    "1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6";
constexpr char kAddressFreeParameterSha256[] =
    "80189cac9cec275fc90e1a52f3e814475d063b67ed2fd80b05ae98ee1e9654f6";
constexpr char kRawCapturedParameterSha256[] =
    "2c38d2f149ec90e2e6cac2a843a824485c16421300fafcce34a1acaaeb8a25e4";
constexpr char kGenerator[] =
    "splitmix64-fg-custom-upsample-signed-spatial-v2";

std::size_t TensorBytes(std::uint32_t height, std::uint32_t width,
                        std::uint32_t channels) {
  return static_cast<std::size_t>(height) * width * channels * 2u;
}

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t InputHalfBits(std::size_t plane, std::size_t index,
                            std::uint32_t channels) {
  const std::uint32_t channel = static_cast<std::uint32_t>(index % channels);
  const std::uint64_t mixed = SplitMix64(
      0xD1B54A32D192ED03ull ^
      (static_cast<std::uint64_t>(plane + 1u) * 0x94D049BB133111EBull) ^
      (static_cast<std::uint64_t>(channel) * 0x8EBC6AF09C88C6E3ull) ^
      (static_cast<std::uint64_t>(index) * 0x9E3779B97F4A7C15ull));
  const std::uint16_t sign =
      (mixed >> 63u) != 0u ? 0x8000u : 0x0000u;
  if ((mixed & 63u) == 0u) return sign;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(13u + ((mixed >> 12u) & 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03FFu));
}

std::vector<std::uint8_t> GenerateInput(std::size_t plane) {
  const std::uint32_t channels = kChannels[plane];
  const std::size_t values =
      static_cast<std::size_t>(kInputHeight) * kInputWidth * channels;
  std::vector<std::uint8_t> result(values * 2u);
  for (std::size_t index = 0; index < values; ++index) {
    Store16(&result, index * 2u,
            InputHalfBits(plane, index, channels));
  }
  return result;
}

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size " + path.string());
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
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0;
  DWORD digest_bytes = 0;
  DWORD written = 0;
  auto Require = [](NTSTATUS status, const char* operation) {
    if (status < 0) throw std::runtime_error(std::string(operation) + " failed");
  };
  Require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                      nullptr, 0),
          "BCryptOpenAlgorithmProvider");
  try {
    Require(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                              reinterpret_cast<PUCHAR>(&object_bytes),
                              sizeof(object_bytes), &written, 0),
            "BCryptGetProperty(object)");
    Require(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                              reinterpret_cast<PUCHAR>(&digest_bytes),
                              sizeof(digest_bytes), &written, 0),
            "BCryptGetProperty(digest)");
    std::vector<std::uint8_t> object(object_bytes);
    std::vector<std::uint8_t> digest(digest_bytes);
    Require(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes,
                             nullptr, 0, 0),
            "BCryptCreateHash");
    Require(BCryptHashData(hash, const_cast<PUCHAR>(bytes.data()),
                           static_cast<ULONG>(bytes.size()), 0),
            "BCryptHashData");
    Require(BCryptFinishHash(hash, digest.data(), digest_bytes, 0),
            "BCryptFinishHash");
    BCryptDestroyHash(hash);
    hash = nullptr;
    BCryptCloseAlgorithmProvider(algorithm, 0);
    algorithm = nullptr;
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (std::uint8_t byte : digest) {
      text << std::setw(2) << static_cast<unsigned>(byte);
    }
    return text.str();
  } catch (...) {
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw;
  }
}

void Check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (name != nullptr ? name : "unknown") + " / " +
      (description != nullptr ? description : "unknown"));
}

struct GuardedDeviceBuffer {
  CUdeviceptr base = 0;
  std::size_t payload_bytes = 0;

  CUdeviceptr payload() const { return base + kGuardBytes; }

  void Allocate(std::size_t bytes) {
    payload_bytes = bytes;
    Check(cuMemAlloc(&base, bytes + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, bytes + 2u * kGuardBytes),
          "cuMemsetD8(guarded)");
  }

  std::vector<std::uint8_t> ReadAll() const {
    std::vector<std::uint8_t> result(payload_bytes + 2u * kGuardBytes);
    Check(cuMemcpyDtoH(result.data(), base, result.size()), "cuMemcpyDtoH");
    return result;
  }

  void Free() {
    if (base != 0) {
      Check(cuMemFree(base), "cuMemFree");
      base = 0;
    }
  }

  ~GuardedDeviceBuffer() {
    if (base != 0) cuMemFree(base);
  }
};

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t payload_bytes) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes +
                                       payload_bytes);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t payload_bytes) {
  std::size_t result = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    result += guarded[index] != kGuardValue;
    result += guarded[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return result;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("mismatch comparison size differs");
  }
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] != right[index];
  }
  return result;
}

std::size_t PoisonMatchingHalfValues(const std::vector<std::uint8_t>& output,
                                     std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2u) {
    result += output[offset] == poison && output[offset + 1u] == poison;
  }
  return result;
}

struct TensorRecord {
  CUdeviceptr pointer;
  std::uint32_t batch;
  std::uint32_t height;
  std::uint32_t width;
  std::uint32_t channels;
};
static_assert(sizeof(TensorRecord) == 24);

struct PackedParameters {
  std::array<TensorRecord, 3> inputs;
  std::array<TensorRecord, 3> outputs;
};
static_assert(sizeof(PackedParameters) == 144);

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::string& module_sha,
    const std::array<std::vector<std::uint8_t>, 3>& expected_inputs,
    const std::array<std::vector<std::uint8_t>, 3>& actual_inputs,
    const std::array<std::vector<std::uint8_t>, 3>& outputs,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t poison_matching_half_values) {
  std::vector<std::uint8_t> combined_output;
  for (const auto& output : outputs) {
    combined_output.insert(combined_output.end(), output.begin(), output.end());
  }
  const bool passed = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.custom-upsample-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"module_token\": 60,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadDataEx exact selected sm_89 PTX fatbin\",\n"
         << "  \"function_name\": \"custom_upsample_hf_kernel\",\n"
         << "  \"launch_token\": 70,\n"
         << "  \"grid\": [40, 23, 3],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"address_free_parameter_sha256\": \""
         << kAddressFreeParameterSha256 << "\",\n"
         << "  \"raw_captured_parameter_sha256\": \""
         << kRawCapturedParameterSha256 << "\",\n"
         << "  \"input_generator\": \"" << kGenerator << "\",\n"
         << "  \"planes\": [\n";
  for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
    stream << "    {\"plane\": " << plane << ", \"channels\": "
           << kChannels[plane]
           << ", \"input_height\": 184, \"input_width\": 320"
           << ", \"output_height\": 368, \"output_width\": 640"
           << ", \"input_bytes\": " << expected_inputs[plane].size()
           << ", \"output_bytes\": " << outputs[plane].size()
           << ", \"input_pre_sha256\": \""
           << Sha256(expected_inputs[plane])
           << "\", \"input_post_sha256\": \""
           << Sha256(actual_inputs[plane])
           << "\", \"output_sha256\": \"" << Sha256(outputs[plane])
           << "\"}" << (plane + 1u == kChannels.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"combined_output_sha256\": \"" << Sha256(combined_output)
         << "\",\n"
         << "  \"total_output_bytes\": " << combined_output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_matching_half_values\": "
         << poison_matching_half_values << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"single_poison_full_write_claimed\": false,\n"
         << "  \"independently_generated_inputs\": true,\n"
         << "  \"external_expected_tensors_consumed\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-custom-upsample-rtx <module.bin> <output-dir> "
                 "<run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_directory = argv[2];
    const std::string run_id = argv[3];
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    std::filesystem::create_directories(output_directory);

    const auto module_bytes = Read(module_path);
    const std::string module_sha = Sha256(module_bytes);
    if (module_bytes.size() != kExpectedModuleBytes ||
        module_sha != kExpectedModuleSha256) {
      throw std::runtime_error("selected module identity mismatch");
    }

    std::array<std::vector<std::uint8_t>, 3> expected_inputs;
    for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
      expected_inputs[plane] = GenerateInput(plane);
    }

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    CUmodule module = nullptr;
    std::array<char, 8192> info_log{};
    std::array<char, 8192> error_log{};
    std::array<CUjit_option, 4> options = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    std::array<void*, 4> option_values = {
        info_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(info_log.size())),
        error_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(error_log.size()))};
    Check(cuModuleLoadDataEx(&module, module_bytes.data(),
                             static_cast<unsigned>(options.size()),
                             options.data(), option_values.data()),
          "cuModuleLoadDataEx(exact selected module)");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "custom_upsample_hf_kernel"),
          "cuModuleGetFunction(custom_upsample_hf_kernel)");

    std::array<GuardedDeviceBuffer, 3> inputs;
    std::array<GuardedDeviceBuffer, 3> outputs;
    for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
      inputs[plane].Allocate(expected_inputs[plane].size());
      outputs[plane].Allocate(
          TensorBytes(kOutputHeight, kOutputWidth, kChannels[plane]));
      Check(cuMemcpyHtoD(inputs[plane].payload(), expected_inputs[plane].data(),
                         expected_inputs[plane].size()),
            "cuMemcpyHtoD(input plane)");
      Check(cuMemsetD8(outputs[plane].payload(), poison,
                       outputs[plane].payload_bytes),
            "cuMemsetD8(output poison)");
    }

    PackedParameters parameters{};
    for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
      parameters.inputs[plane] = {inputs[plane].payload(), 1, kInputHeight,
                                  kInputWidth, kChannels[plane]};
      parameters.outputs[plane] = {outputs[plane].payload(), 1, kOutputHeight,
                                   kOutputWidth, kChannels[plane]};
    }
    void* kernel_parameters[] = {&parameters};
    Check(cuLaunchKernel(function, 40, 23, 3, 16, 16, 1, 0, nullptr,
                         kernel_parameters, nullptr),
          "cuLaunchKernel(custom_upsample_hf_kernel)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    std::array<std::vector<std::uint8_t>, 3> actual_inputs;
    std::array<std::vector<std::uint8_t>, 3> actual_outputs;
    std::size_t input_mismatches = 0;
    std::size_t canary_mismatches = 0;
    std::size_t poison_matching_half_values = 0;
    for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
      const auto guarded_input = inputs[plane].ReadAll();
      const auto guarded_output = outputs[plane].ReadAll();
      actual_inputs[plane] =
          Payload(guarded_input, inputs[plane].payload_bytes);
      actual_outputs[plane] =
          Payload(guarded_output, outputs[plane].payload_bytes);
      input_mismatches +=
          Mismatches(expected_inputs[plane], actual_inputs[plane]);
      canary_mismatches +=
          CanaryMismatches(guarded_input, inputs[plane].payload_bytes) +
          CanaryMismatches(guarded_output, outputs[plane].payload_bytes);
      poison_matching_half_values +=
          PoisonMatchingHalfValues(actual_outputs[plane], poison);
      Write(output_directory /
                ("input-plane-" + std::to_string(plane) + "-after.bin"),
            actual_inputs[plane]);
      Write(output_directory /
                ("output-plane-" + std::to_string(plane) + ".bin"),
            actual_outputs[plane]);
    }

    WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                 module_sha, expected_inputs, actual_inputs, actual_outputs,
                 input_mismatches, canary_mismatches,
                 poison_matching_half_values);
    std::ofstream jit_stream(output_directory / "driver-jit.log",
                             std::ios::binary | std::ios::trunc);
    jit_stream << "info_log=" << info_log.data() << "\nerror_log="
               << error_log.data() << '\n';

    std::vector<std::uint8_t> combined_output;
    for (const auto& output : actual_outputs) {
      combined_output.insert(combined_output.end(), output.begin(), output.end());
    }
    const bool passed = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_custom_upsample_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "module_sha256=" << module_sha << '\n'
              << "combined_output_sha256=" << Sha256(combined_output) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "poison_matching_half_values="
              << poison_matching_half_values << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    for (auto& input : inputs) input.Free();
    for (auto& output : outputs) output.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
