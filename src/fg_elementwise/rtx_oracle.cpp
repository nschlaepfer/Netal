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

constexpr std::uint32_t kWidth = 10;
constexpr std::uint32_t kHeight = 6;
constexpr std::uint32_t kChannels = 256;
constexpr std::size_t kValues =
    static_cast<std::size_t>(kWidth) * kHeight * kChannels;
constexpr std::size_t kTensorBytes = kValues * sizeof(std::uint16_t);
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kExpectedModuleBytes = 2152;
constexpr char kExpectedModuleSha256[] =
    "1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26";
constexpr char kExpectedParameterSha256[] =
    "0bb4c1c5e582ea1a7386318d294df1511fdc2e81287b7943f23f9a531a1690bc";
constexpr char kNormalizedParameterSha256[] =
    "60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80";
constexpr char kScalarParameterSha256[] =
    "0f2b3cefe270ff83b105bc583212a894fdaf8671acab257be993fcedab912822";

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t InputHalfBits(std::size_t index, bool input_b) {
  constexpr std::array<std::uint16_t, 8> special_a = {
      0x3C00u, 0x3C01u, 0xBC00u, 0xBC01u,
      0x0400u, 0x8400u, 0x3555u, 0xB555u,
  };
  constexpr std::array<std::uint16_t, 8> special_b = {
      0x1000u, 0x1000u, 0x9000u, 0x9000u,
      0x0400u, 0x8400u, 0x3000u, 0xB000u,
  };
  if (index < special_a.size()) {
    return input_b ? special_b[index] : special_a[index];
  }

  constexpr std::uint64_t seed_a = 0x243F6A8885A308D3ull;
  constexpr std::uint64_t seed_b = 0x13198A2E03707344ull;
  constexpr std::uint64_t seed_sign = 0xA4093822299F31D0ull;
  constexpr std::uint64_t step_a = 0x9E3779B97F4A7C15ull;
  constexpr std::uint64_t step_b = 0xBF58476D1CE4E5B9ull;
  constexpr std::uint64_t step_sign = 0x94D049BB133111EBull;
  const std::uint64_t lane = static_cast<std::uint64_t>(index);
  const std::uint64_t value = SplitMix64(
      (input_b ? seed_b : seed_a) + lane * (input_b ? step_b : step_a));
  const std::uint16_t sign = static_cast<std::uint16_t>(
      ((SplitMix64(seed_sign + lane * step_sign) >> 63u) & 1u) << 15u);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(5u + ((value >> 10u) % 19u));
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>(exponent << 10u) |
      static_cast<std::uint16_t>(value & 0x03FFu));
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

struct Inputs {
  std::vector<std::uint8_t> a;
  std::vector<std::uint8_t> b;
};

Inputs GenerateInputs() {
  Inputs inputs{
      std::vector<std::uint8_t>(kTensorBytes),
      std::vector<std::uint8_t>(kTensorBytes),
  };
  for (std::size_t index = 0; index < kValues; ++index) {
    Store16(&inputs.a, index * 2u, InputHalfBits(index, false));
    Store16(&inputs.b, index * 2u, InputHalfBits(index, true));
  }
  return inputs;
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
    for (const std::uint8_t byte : digest) {
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
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name != nullptr ? name : "unknown") + " / " +
                           (description != nullptr ? description : "unknown"));
}

struct GuardedDeviceBuffer {
  CUdeviceptr base = 0;
  std::size_t payload_bytes = 0;

  CUdeviceptr payload() const { return base + kGuardBytes; }

  void Allocate(std::size_t size) {
    payload_bytes = size;
    Check(cuMemAlloc(&base, size + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, size + 2u * kGuardBytes),
          "cuMemsetD8(guarded allocation)");
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

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded) {
  return std::vector<std::uint8_t>(
      guarded.begin() + kGuardBytes,
      guarded.begin() + kGuardBytes + kTensorBytes);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded) {
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += guarded[index] != kGuardValue;
    mismatches += guarded[kGuardBytes + kTensorBytes + index] != kGuardValue;
  }
  return mismatches;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("comparison size mismatch");
  }
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    mismatches += left[index] != right[index];
  }
  return mismatches;
}

std::size_t UntouchedHalfWords(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  const std::uint16_t marker =
      static_cast<std::uint16_t>(poison | (std::uint16_t(poison) << 8u));
  std::size_t untouched = 0;
  for (std::size_t index = 0; index < kValues; ++index) {
    const std::uint16_t value = static_cast<std::uint16_t>(
        output[index * 2u] | (std::uint16_t(output[index * 2u + 1u]) << 8u));
    untouched += value == marker;
  }
  return untouched;
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& module_sha,
                  const Inputs& inputs,
                  const std::vector<std::uint8_t>& actual_a,
                  const std::vector<std::uint8_t>& actual_b,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t untouched_half_words) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.elementwise-add-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"module_token\": 37,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadDataEx exact selected sm_89 PTX fatbin; driver JIT to active RTX architecture\",\n"
         << "  \"function_name\": \"k_element_wise\",\n"
         << "  \"launch_token\": 33,\n"
         << "  \"evaluate_1_phase_order\": 22,\n"
         << "  \"global_launch_order\": 31,\n"
         << "  \"live_parameter_bytes\": 52,\n"
         << "  \"captured_raw_parameter_sha256\": \""
         << kExpectedParameterSha256 << "\",\n"
         << "  \"address_free_parameter_sha256\": \""
         << kNormalizedParameterSha256 << "\",\n"
         << "  \"scalar_parameter_sha256\": \""
         << kScalarParameterSha256 << "\",\n"
         << "  \"binding_policy\": \"fresh guarded allocations replace captured addresses; scalar ABI remains exact\",\n"
         << "  \"parameter_scalars_u32\": [10, 6, 256, 10, 6, 10, 6],\n"
         << "  \"grid\": [3, 2, 1],\n"
         << "  \"block\": [4, 4, 8],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"shape_nhwc\": [1, 6, 10, 256],\n"
         << "  \"operation\": \"binary16 add.rn, eight channels per vector, NHWC\",\n"
         << "  \"input_generator\": \"splitmix64-fg-elementwise-add-v1\",\n"
         << "  \"input_a_pre_sha256\": \"" << Sha256(inputs.a) << "\",\n"
         << "  \"input_a_post_sha256\": \"" << Sha256(actual_a) << "\",\n"
         << "  \"input_b_pre_sha256\": \"" << Sha256(inputs.b) << "\",\n"
         << "  \"input_b_post_sha256\": \"" << Sha256(actual_b) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"tensor_bytes\": " << kTensorBytes << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_half_words\": " << untouched_half_words << ",\n"
         << "  \"full_write_coverage\": "
         << (untouched_half_words == 0 ? "true" : "false") << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"external_tensor_input_files_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \""
         << ((input_mismatches == 0 && canary_mismatches == 0 &&
              untouched_half_words == 0)
                 ? "pass"
                 : "fail")
         << "\"\n}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-elementwise-rtx <module.bin> <output-dir> "
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
    const Inputs inputs = GenerateInputs();

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
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    std::array<void*, 4> option_values = {
        info_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(info_log.size())),
        error_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(error_log.size())),
    };
    Check(cuModuleLoadDataEx(&module, module_bytes.data(),
                             static_cast<unsigned int>(options.size()),
                             options.data(), option_values.data()),
          "cuModuleLoadDataEx(exact selected module)");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "k_element_wise"),
          "cuModuleGetFunction(k_element_wise)");

    GuardedDeviceBuffer input_a;
    GuardedDeviceBuffer input_b;
    GuardedDeviceBuffer output;
    input_a.Allocate(kTensorBytes);
    input_b.Allocate(kTensorBytes);
    output.Allocate(kTensorBytes);
    Check(cuMemcpyHtoD(input_a.payload(), inputs.a.data(), inputs.a.size()),
          "cuMemcpyHtoD(input a)");
    Check(cuMemcpyHtoD(input_b.payload(), inputs.b.data(), inputs.b.size()),
          "cuMemcpyHtoD(input b)");
    Check(cuMemsetD8(output.payload(), poison, kTensorBytes),
          "cuMemsetD8(output poison)");

    CUdeviceptr input_a_pointer = input_a.payload();
    CUdeviceptr input_b_pointer = input_b.payload();
    CUdeviceptr output_pointer = output.payload();
    std::uint32_t output_width = kWidth;
    std::uint32_t output_height = kHeight;
    std::uint32_t channels = kChannels;
    std::uint32_t input_a_width = kWidth;
    std::uint32_t input_a_height = kHeight;
    std::uint32_t input_b_width = kWidth;
    std::uint32_t input_b_height = kHeight;
    void* kernel_parameters[] = {
        &input_a_pointer, &input_b_pointer, &output_pointer,
        &output_width, &output_height, &channels,
        &input_a_width, &input_a_height, &input_b_width, &input_b_height,
    };
    Check(cuLaunchKernel(function, 3, 2, 1, 4, 4, 8, 0, nullptr,
                         kernel_parameters, nullptr),
          "cuLaunchKernel(k_element_wise)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto guarded_a = input_a.ReadAll();
    const auto guarded_b = input_b.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto actual_a = Payload(guarded_a);
    const auto actual_b = Payload(guarded_b);
    const auto actual_output = Payload(guarded_output);
    const std::size_t input_mismatches =
        Mismatches(actual_a, inputs.a) + Mismatches(actual_b, inputs.b);
    const std::size_t canary_mismatches =
        CanaryMismatches(guarded_a) + CanaryMismatches(guarded_b) +
        CanaryMismatches(guarded_output);
    const std::size_t untouched_half_words =
        UntouchedHalfWords(actual_output, poison);

    Write(output_directory / "input-a-after.bin", actual_a);
    Write(output_directory / "input-b-after.bin", actual_b);
    Write(output_directory / "output.bin", actual_output);
    WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                 module_sha, inputs, actual_a, actual_b, actual_output,
                 input_mismatches, canary_mismatches, untouched_half_words);
    std::ofstream jit_stream(output_directory / "driver-jit.log",
                             std::ios::binary | std::ios::trunc);
    jit_stream << "info_log=" << info_log.data() << "\nerror_log="
               << error_log.data() << '\n';

    const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                        untouched_half_words == 0;
    std::cout << "probe=fg_elementwise_add_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "module_sha256=" << module_sha << '\n'
              << "input_a_sha256=" << Sha256(actual_a) << '\n'
              << "input_b_sha256=" << Sha256(actual_b) << '\n'
              << "output_sha256=" << Sha256(actual_output) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "untouched_half_words=" << untouched_half_words << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    input_a.Free();
    input_b.Free();
    output.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
