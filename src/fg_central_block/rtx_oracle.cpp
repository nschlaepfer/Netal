#include <windows.h>
#include <bcrypt.h>
#include <cuda.h>

#include <array>
#include <cstdint>
#include <cstdlib>
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
constexpr std::array<std::size_t, 6> kInputBytes = {
    kPixels * 8u, kPixels * 8u, kPixels * 2u,
    kPixels * 8u, kPixels * 2u, kPixels * 6u,
};
constexpr std::size_t kOutputBytes = kPixels * 36u;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr float kFactor = 0.5f;
constexpr char kGenerator[] = "splitmix64-fg-central-block-signed-v1";
constexpr char kExpectedModuleSha256[] =
    "184542107e98be3dc6229251c19cae4ecc44a7cc1b78cbf741b7fb3b8d81f060";
constexpr std::size_t kExpectedModuleBytes = 5752;

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* output, std::size_t offset,
             std::uint16_t value) {
  (*output)[offset] = static_cast<std::uint8_t>(value);
  (*output)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t TensorHalfBits(std::size_t lane, std::size_t index) {
  static constexpr std::array<std::uint16_t, 16> kSpecial = {
      0x0000u, 0x8000u, 0x2C00u, 0xAC00u,
      0x3400u, 0xB400u, 0x3555u, 0xB555u,
      0x3800u, 0xB800u, 0x3A00u, 0xBA00u,
      0x3C00u, 0xBC00u, 0x3E00u, 0xBE00u,
  };
  const std::uint64_t mixed = SplitMix64(
      0xD6E8FEB86659FD93ull ^
      (static_cast<std::uint64_t>(lane + 1u) * 0xA0761D6478BD642Full) ^
      (static_cast<std::uint64_t>(index) * 0xE7037ED1A0B428DBull));
  if ((mixed & 7u) == 0u) {
    return kSpecial[(mixed >> 8u) & 15u];
  }
  const std::uint16_t sign =
      (mixed >> 63u) != 0u ? 0x8000u : 0x0000u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((mixed >> 12u) & 5u));
  return static_cast<std::uint16_t>(
      sign | (exponent << 10u) | ((mixed >> 19u) & 0x03FFu));
}

std::uint16_t MotionHalfBits(std::size_t pixel, std::size_t component) {
  static constexpr std::array<std::uint16_t, 16> kMotion = {
      0x0000u, 0x3400u, 0xB400u, 0x3800u,
      0xB800u, 0x3A00u, 0xBA00u, 0x3D00u,
      0xBD00u, 0x3E00u, 0xBE00u, 0x4080u,
      0xC080u, 0x3555u, 0xB555u, 0xBC00u,
  };
  const std::uint32_t x = static_cast<std::uint32_t>(pixel % kWidth);
  const std::uint32_t y = static_cast<std::uint32_t>(pixel / kWidth);
  if (component == 0u && x == 0u) return 0xBA00u;  // -0.75 -> -1.5
  if (component == 0u && x == 1u) return 0xBA00u;  // -0.75 -> -0.5
  if (component == 1u && y == 0u) return 0xB555u;  // about -0.333
  if (component == 1u && y == 1u) return 0xBE00u;  // -1.5 -> -2
  if (component == 2u && x == 0u) return 0xBD00u;  // -1.25 -> -2.5
  if (component == 3u && y == 0u) return 0xBA00u;  // -0.75 -> -1.5
  if ((component == 0u || component == 2u) && x + 1u == kWidth) {
    return 0x3A00u;
  }
  if ((component == 1u || component == 3u) && y + 1u == kHeight) {
    return 0x3A00u;
  }
  const std::uint64_t mixed = SplitMix64(
      0x8EBC6AF09C88C6E3ull ^
      (static_cast<std::uint64_t>(pixel) * 0x589965CC75374CC3ull) ^
      (static_cast<std::uint64_t>(component + 1u) *
       0x1D8E4E27C47D124Full));
  return kMotion[mixed & 15u];
}

struct Inputs {
  std::array<std::vector<std::uint8_t>, 6> lanes;
};

Inputs GenerateInputs() {
  Inputs result;
  for (std::size_t lane = 0; lane < result.lanes.size(); ++lane) {
    result.lanes[lane].resize(kInputBytes[lane]);
  }
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    for (std::size_t component = 0; component < 4; ++component) {
      Store16(&result.lanes[0], (pixel * 4u + component) * 2u,
              TensorHalfBits(0, pixel * 4u + component));
      Store16(&result.lanes[1], (pixel * 4u + component) * 2u,
              TensorHalfBits(1, pixel * 4u + component));
      Store16(&result.lanes[3], (pixel * 4u + component) * 2u,
              MotionHalfBits(pixel, component));
    }
    Store16(&result.lanes[2], pixel * 2u, TensorHalfBits(2, pixel));
    Store16(&result.lanes[4], pixel * 2u, TensorHalfBits(4, pixel));
    for (std::size_t component = 0; component < 3; ++component) {
      Store16(&result.lanes[5], (pixel * 3u + component) * 2u,
              TensorHalfBits(5 + component, pixel * 3u + component));
    }
  }
  return result;
}

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size " + path.string());
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return result;
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
    if (status < 0) throw std::runtime_error(std::string(operation) + " failed");
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
    RequireStatus(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes,
                                   nullptr, 0, 0),
                  "BCryptCreateHash");
    RequireStatus(BCryptHashData(hash, const_cast<PUCHAR>(bytes.data()),
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

  void Allocate(std::size_t size, std::uint8_t fill) {
    payload_bytes = size;
    Check(cuMemAlloc(&base, size + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, size + 2u * kGuardBytes),
          "cuMemsetD8(allocation)");
    Check(cuMemsetD8(payload(), fill, size), "cuMemsetD8(payload)");
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
                                  std::size_t size) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes + size);
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

std::size_t UntouchedPixels(const std::vector<std::uint8_t>& output,
                            std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    bool untouched = true;
    for (std::size_t byte = 0; byte < 36u; ++byte) {
      untouched &= output[pixel * 36u + byte] == poison;
    }
    result += untouched;
  }
  return result;
}

struct FunctionResult {
  std::string name;
  std::vector<std::uint8_t> output;
  std::size_t canary_mismatches = 0;
  std::size_t untouched_pixels = 0;
};

FunctionResult Execute(CUfunction function, const std::string& name,
                       const std::array<CUdeviceptr, 6>& inputs,
                       std::uint8_t poison) {
  GuardedDeviceBuffer output;
  output.Allocate(kOutputBytes, poison);
  CUdeviceptr output_pointer = output.payload();
  std::array<CUdeviceptr, 6> input_pointers = inputs;
  std::uint32_t width = kWidth;
  std::uint32_t height = kHeight;
  float factor = kFactor;
  void* arguments[] = {
      &output_pointer,
      &input_pointers[0], &input_pointers[1], &input_pointers[2],
      &input_pointers[3], &input_pointers[4], &input_pointers[5],
      &width, &height, &factor,
  };
  Check(cuLaunchKernel(function, 40, 23, 1, 16, 16, 1, 0, nullptr,
                       arguments, nullptr),
        "cuLaunchKernel");
  Check(cuCtxSynchronize(), "cuCtxSynchronize");
  const auto guarded = output.ReadAll();
  FunctionResult result;
  result.name = name;
  result.output = Payload(guarded, kOutputBytes);
  result.canary_mismatches = CanaryMismatches(guarded, kOutputBytes);
  result.untouched_pixels = UntouchedPixels(result.output, poison);
  return result;
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& module_sha,
                  const Inputs& generated,
                  const std::array<std::vector<std::uint8_t>, 6>& actual_inputs,
                  const FunctionResult& original,
                  const FunctionResult& optimized,
                  std::size_t input_mismatches,
                  std::size_t input_canary_mismatches,
                  std::size_t entry_mismatches) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  const std::size_t canary_mismatches =
      input_canary_mismatches + original.canary_mismatches +
      optimized.canary_mismatches;
  const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                      original.untouched_pixels == 0 &&
                      optimized.untouched_pixels == 0 && entry_mismatches == 0;
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.central-block-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 61,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadDataEx exact selected sm_89 PTX fatbin; driver JIT to active RTX architecture\",\n"
         << "  \"launch_order\": 71,\n"
         << "  \"function_names\": [\"k_central_block_original\", \"k_central_block\"],\n"
         << "  \"grid\": [40, 23, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"width\": " << kWidth << ",\n"
         << "  \"height\": " << kHeight << ",\n"
         << "  \"factor_f32\": 0.5,\n"
         << "  \"factor_f16_bits\": 14336,\n"
         << "  \"input_generator\": \"" << kGenerator << "\",\n"
         << "  \"input_bytes\": [";
  for (std::size_t lane = 0; lane < kInputBytes.size(); ++lane) {
    stream << kInputBytes[lane] << (lane + 1u == kInputBytes.size() ? "" : ", ");
  }
  stream << "],\n  \"input_pre_sha256\": [";
  for (std::size_t lane = 0; lane < generated.lanes.size(); ++lane) {
    stream << "\"" << Sha256(generated.lanes[lane]) << "\""
           << (lane + 1u == generated.lanes.size() ? "" : ", ");
  }
  stream << "],\n  \"input_post_sha256\": [";
  for (std::size_t lane = 0; lane < actual_inputs.size(); ++lane) {
    stream << "\"" << Sha256(actual_inputs[lane]) << "\""
           << (lane + 1u == actual_inputs.size() ? "" : ", ");
  }
  stream << "],\n"
         << "  \"output_bytes\": " << kOutputBytes << ",\n"
         << "  \"original_output_sha256\": \"" << Sha256(original.output) << "\",\n"
         << "  \"optimized_output_sha256\": \"" << Sha256(optimized.output) << "\",\n"
         << "  \"entry_mismatched_bytes\": " << entry_mismatches << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"original_untouched_pixels\": " << original.untouched_pixels << ",\n"
         << "  \"optimized_untouched_pixels\": " << optimized.untouched_pixels << ",\n"
         << "  \"input_immutability\": " << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"full_write_coverage\": "
         << ((original.untouched_pixels == 0 && optimized.untouched_pixels == 0)
                 ? "true" : "false") << ",\n"
         << "  \"entry_byte_equality\": " << (entry_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"external_tensor_inputs_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-central-block-rtx <module> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_directory = argv[2];
    const std::string run_id = argv[3];
    char* poison_end = nullptr;
    const unsigned long parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != kExpectedModuleBytes ||
        Sha256(module_bytes) != kExpectedModuleSha256) {
      throw std::runtime_error("selected module identity mismatch");
    }
    std::filesystem::create_directories(output_directory);
    const Inputs inputs = GenerateInputs();

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    Check(cuModuleLoadDataEx(&module, module_bytes.data(), 0, nullptr, nullptr),
          "cuModuleLoadDataEx");
    CUfunction original_function = nullptr;
    CUfunction optimized_function = nullptr;
    Check(cuModuleGetFunction(&original_function, module,
                              "k_central_block_original"),
          "cuModuleGetFunction(original)");
    Check(cuModuleGetFunction(&optimized_function, module, "k_central_block"),
          "cuModuleGetFunction(optimized)");

    std::array<GuardedDeviceBuffer, 6> device_inputs;
    std::array<CUdeviceptr, 6> input_pointers{};
    for (std::size_t lane = 0; lane < device_inputs.size(); ++lane) {
      device_inputs[lane].Allocate(kInputBytes[lane], 0);
      Check(cuMemcpyHtoD(device_inputs[lane].payload(), inputs.lanes[lane].data(),
                         inputs.lanes[lane].size()),
            "cuMemcpyHtoD(input)");
      input_pointers[lane] = device_inputs[lane].payload();
    }

    const FunctionResult original =
        Execute(original_function, "k_central_block_original", input_pointers,
                static_cast<std::uint8_t>(parsed_poison));
    const FunctionResult optimized =
        Execute(optimized_function, "k_central_block", input_pointers,
                static_cast<std::uint8_t>(parsed_poison));

    std::array<std::vector<std::uint8_t>, 6> actual_inputs;
    std::size_t input_mismatches = 0;
    std::size_t input_canary_mismatches = 0;
    for (std::size_t lane = 0; lane < device_inputs.size(); ++lane) {
      const auto guarded = device_inputs[lane].ReadAll();
      actual_inputs[lane] = Payload(guarded, kInputBytes[lane]);
      input_mismatches += Mismatches(inputs.lanes[lane], actual_inputs[lane]);
      input_canary_mismatches += CanaryMismatches(guarded, kInputBytes[lane]);
      Write(output_directory / ("input-" + std::to_string(lane) + "-after.bin"),
            actual_inputs[lane]);
    }
    const std::size_t entry_mismatches =
        Mismatches(original.output, optimized.output);
    Write(output_directory / "output-original.bin", original.output);
    Write(output_directory / "output-optimized.bin", optimized.output);
    WriteReceipt(output_directory / "run-receipt.json", run_id,
                 static_cast<std::uint8_t>(parsed_poison), Sha256(module_bytes),
                 inputs, actual_inputs, original, optimized, input_mismatches,
                 input_canary_mismatches, entry_mismatches);

    for (auto& input : device_inputs) input.Free();

    Check(cuModuleUnload(module), "cuModuleUnload");
    module = nullptr;
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    context = nullptr;

    if (input_mismatches != 0 || input_canary_mismatches != 0 ||
        original.canary_mismatches != 0 || optimized.canary_mismatches != 0 ||
        original.untouched_pixels != 0 || optimized.untouched_pixels != 0 ||
        entry_mismatches != 0) {
      std::cerr << "result=fail entry_mismatches=" << entry_mismatches << "\n";
      return 1;
    }
    std::cout << "result=pass output_sha256=" << Sha256(optimized.output) << "\n";
    return 0;
  } catch (const std::exception& error) {
    if (module != nullptr) cuModuleUnload(module);
    if (context != nullptr) cuDevicePrimaryCtxRelease(0);
    std::cerr << "error=" << error.what() << "\n";
    return 1;
  }
}
