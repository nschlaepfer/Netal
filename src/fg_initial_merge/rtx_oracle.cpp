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

#pragma comment(lib, "bcrypt.lib")

namespace {

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 368;
constexpr std::size_t kPixels =
    static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight);
constexpr std::size_t kInputABytes = kPixels * 8;
constexpr std::size_t kInputBBytes = kPixels * 8;
constexpr std::size_t kInputCBytes = kPixels * 2;
constexpr std::size_t kOutputBytes = kPixels * 20;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr float kInterpolation = 0.5f;
constexpr std::uint16_t kInterpolationHalfBits = 0x3800;
constexpr char kExpectedModuleSha256[] =
    "63b7e8105e00b04049e1004a8df23998074fb04b2b6637171da379b4a967d8c3";
constexpr std::size_t kExpectedModuleBytes = 912;

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store64(std::vector<std::uint8_t>* output, std::size_t offset,
             std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    (*output)[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8u));
  }
}

void Store16(std::vector<std::uint8_t>* output, std::size_t offset,
             std::uint16_t value) {
  (*output)[offset] = static_cast<std::uint8_t>(value);
  (*output)[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
}

struct Inputs {
  std::vector<std::uint8_t> a;
  std::vector<std::uint8_t> b;
  std::vector<std::uint8_t> c;
};

Inputs GenerateInputs() {
  Inputs inputs{
      std::vector<std::uint8_t>(kInputABytes),
      std::vector<std::uint8_t>(kInputBBytes),
      std::vector<std::uint8_t>(kInputCBytes),
  };
  constexpr std::uint64_t seed_a = 0x243F6A8885A308D3ull;
  constexpr std::uint64_t seed_b = 0x13198A2E03707344ull;
  constexpr std::uint64_t seed_c = 0xA4093822299F31D0ull;
  constexpr std::uint64_t step_b = 0x9E3779B97F4A7C15ull;
  constexpr std::uint64_t step_c = 0xBF58476D1CE4E5B9ull;
  for (std::size_t index = 0; index < kPixels; ++index) {
    const std::uint64_t lane = static_cast<std::uint64_t>(index);
    Store64(&inputs.a, index * 8, SplitMix64(seed_a ^ lane));
    Store64(&inputs.b, index * 8, SplitMix64(seed_b + lane * step_b));
    Store16(&inputs.c, index * 2,
            static_cast<std::uint16_t>(
                SplitMix64(seed_c ^ (lane * step_c)) >> 19u));
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
  auto RequireStatus = [](NTSTATUS status, const char* operation) {
    if (status < 0) {
      throw std::runtime_error(std::string(operation) + " failed");
    }
  };
  RequireStatus(BCryptOpenAlgorithmProvider(
                    &algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
                "BCryptOpenAlgorithmProvider");
  try {
    RequireStatus(BCryptGetProperty(
                      algorithm, BCRYPT_OBJECT_LENGTH,
                      reinterpret_cast<PUCHAR>(&object_bytes),
                      sizeof(object_bytes), &written, 0),
                  "BCryptGetProperty(object)");
    RequireStatus(BCryptGetProperty(
                      algorithm, BCRYPT_HASH_LENGTH,
                      reinterpret_cast<PUCHAR>(&digest_bytes),
                      sizeof(digest_bytes), &written, 0),
                  "BCryptGetProperty(digest)");
    std::vector<std::uint8_t> object(object_bytes);
    std::vector<std::uint8_t> digest(digest_bytes);
    RequireStatus(BCryptCreateHash(algorithm, &hash, object.data(),
                                   object_bytes, nullptr, 0, 0),
                  "BCryptCreateHash");
    RequireStatus(BCryptHashData(
                      hash, const_cast<PUCHAR>(bytes.data()),
                      static_cast<ULONG>(bytes.size()), 0),
                  "BCryptHashData");
    RequireStatus(BCryptFinishHash(hash, digest.data(), digest_bytes, 0),
                  "BCryptFinishHash");
    BCryptDestroyHash(hash);
    hash = nullptr;
    BCryptCloseAlgorithmProvider(algorithm, 0);
    algorithm = nullptr;
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (std::uint8_t byte : digest) text << std::setw(2) << unsigned(byte);
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

  void Allocate(std::size_t size) {
    payload_bytes = size;
    Check(cuMemAlloc(&base, size + 2 * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, size + 2 * kGuardBytes),
          "cuMemsetD8(guarded allocation)");
  }

  std::vector<std::uint8_t> ReadAll() const {
    std::vector<std::uint8_t> result(payload_bytes + 2 * kGuardBytes);
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
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += guarded[index] != kGuardValue;
    mismatches += guarded[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return mismatches;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("mismatch comparison size differs");
  }
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    mismatches += left[index] != right[index];
  }
  return mismatches;
}

std::size_t UntouchedPixels(const std::vector<std::uint8_t>& output,
                            std::uint8_t poison) {
  std::size_t untouched = 0;
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    bool all_poison = true;
    for (std::size_t byte = 0; byte < 20; ++byte) {
      all_poison &= output[pixel * 20 + byte] == poison;
    }
    untouched += all_poison;
  }
  return untouched;
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& module_sha,
                  const Inputs& inputs,
                  const std::vector<std::uint8_t>& actual_a,
                  const std::vector<std::uint8_t>& actual_b,
                  const std::vector<std::uint8_t>& actual_c,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches, std::size_t canary_mismatches,
                  std::size_t untouched_pixels) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.initial-merge-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 53,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadData exact selected sm_89 PTX fatbin; driver JIT to active RTX architecture\",\n"
         << "  \"function_name\": \"k_initial_merge\",\n"
         << "  \"launch_order\": 56,\n"
         << "  \"grid\": [40, 23, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"height\": " << kHeight << ",\n"
         << "  \"width\": " << kWidth << ",\n"
         << "  \"interpolation_f32\": 0.5,\n"
         << "  \"interpolation_f16_bits\": 14336,\n"
         << "  \"input_generator\": \"splitmix64-fg-initial-merge-v1\",\n"
         << "  \"input_a_pre_sha256\": \"" << Sha256(inputs.a) << "\",\n"
         << "  \"input_a_post_sha256\": \"" << Sha256(actual_a) << "\",\n"
         << "  \"input_b_pre_sha256\": \"" << Sha256(inputs.b) << "\",\n"
         << "  \"input_b_post_sha256\": \"" << Sha256(actual_b) << "\",\n"
         << "  \"input_c_pre_sha256\": \"" << Sha256(inputs.c) << "\",\n"
         << "  \"input_c_post_sha256\": \"" << Sha256(actual_c) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_pixels\": " << untouched_pixels << ",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"full_write_coverage\": "
         << (untouched_pixels == 0 ? "true" : "false") << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \""
         << ((input_mismatches == 0 && canary_mismatches == 0 &&
              untouched_pixels == 0)
                 ? "pass"
                 : "fail")
         << "\"\n}\n";
  if (!stream) throw std::runtime_error("short receipt write");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-initial-merge-rtx <module.bin> <output-dir> "
                 "<run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_directory = argv[2];
    const std::string run_id = argv[3];
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    std::filesystem::create_directories(output_directory);

    const std::vector<std::uint8_t> module_bytes = Read(module_path);
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
    unsigned int info_size = static_cast<unsigned int>(info_log.size());
    unsigned int error_size = static_cast<unsigned int>(error_log.size());
    std::array<CUjit_option, 4> options = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    std::array<void*, 4> option_values = {
        info_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(info_size)),
        error_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(error_size)),
    };
    Check(cuModuleLoadDataEx(&module, module_bytes.data(),
                             static_cast<unsigned int>(options.size()),
                             options.data(), option_values.data()),
          "cuModuleLoadDataEx(exact selected module)");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "k_initial_merge"),
          "cuModuleGetFunction(k_initial_merge)");

    GuardedDeviceBuffer input_a;
    GuardedDeviceBuffer input_b;
    GuardedDeviceBuffer input_c;
    GuardedDeviceBuffer output;
    input_a.Allocate(kInputABytes);
    input_b.Allocate(kInputBBytes);
    input_c.Allocate(kInputCBytes);
    output.Allocate(kOutputBytes);
    Check(cuMemcpyHtoD(input_a.payload(), inputs.a.data(), inputs.a.size()),
          "cuMemcpyHtoD(input a)");
    Check(cuMemcpyHtoD(input_b.payload(), inputs.b.data(), inputs.b.size()),
          "cuMemcpyHtoD(input b)");
    Check(cuMemcpyHtoD(input_c.payload(), inputs.c.data(), inputs.c.size()),
          "cuMemcpyHtoD(input c)");
    Check(cuMemsetD8(output.payload(), poison, kOutputBytes),
          "cuMemsetD8(output poison)");

    CUdeviceptr output_pointer = output.payload();
    CUdeviceptr input_a_pointer = input_a.payload();
    CUdeviceptr input_b_pointer = input_b.payload();
    CUdeviceptr input_c_pointer = input_c.payload();
    std::uint32_t height = kHeight;
    std::uint32_t width = kWidth;
    float interpolation = kInterpolation;
    void* kernel_parameters[] = {
        &output_pointer, &input_a_pointer, &input_b_pointer, &input_c_pointer,
        &height,        &width,           &interpolation,
    };
    Check(cuLaunchKernel(function, 40, 23, 1, 16, 16, 1, 0, nullptr,
                         kernel_parameters, nullptr),
          "cuLaunchKernel(k_initial_merge)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto guarded_a = input_a.ReadAll();
    const auto guarded_b = input_b.ReadAll();
    const auto guarded_c = input_c.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto actual_a = Payload(guarded_a, kInputABytes);
    const auto actual_b = Payload(guarded_b, kInputBBytes);
    const auto actual_c = Payload(guarded_c, kInputCBytes);
    const auto actual_output = Payload(guarded_output, kOutputBytes);
    const std::size_t input_mismatches =
        Mismatches(actual_a, inputs.a) + Mismatches(actual_b, inputs.b) +
        Mismatches(actual_c, inputs.c);
    const std::size_t canary_mismatches =
        CanaryMismatches(guarded_a, kInputABytes) +
        CanaryMismatches(guarded_b, kInputBBytes) +
        CanaryMismatches(guarded_c, kInputCBytes) +
        CanaryMismatches(guarded_output, kOutputBytes);
    const std::size_t untouched_pixels = UntouchedPixels(actual_output, poison);

    Write(output_directory / "input-a-after.bin", actual_a);
    Write(output_directory / "input-b-after.bin", actual_b);
    Write(output_directory / "input-c-after.bin", actual_c);
    Write(output_directory / "output.bin", actual_output);
    WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                 module_sha, inputs, actual_a, actual_b, actual_c,
                 actual_output, input_mismatches, canary_mismatches,
                 untouched_pixels);
    std::ofstream jit_stream(output_directory / "driver-jit.log",
                             std::ios::binary | std::ios::trunc);
    jit_stream << "info_log=" << info_log.data() << "\nerror_log="
               << error_log.data() << '\n';

    const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                        untouched_pixels == 0;
    std::cout << "probe=fg_initial_merge_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "module_sha256=" << module_sha << '\n'
              << "output_sha256=" << Sha256(actual_output) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "untouched_pixels=" << untouched_pixels << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    input_a.Free();
    input_b.Free();
    input_c.Free();
    output.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
