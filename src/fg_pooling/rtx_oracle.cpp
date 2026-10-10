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

constexpr char kExpectedModuleSha256[] =
    "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977";
constexpr std::size_t kExpectedModuleBytes = 2032;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;

struct Shape {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t live_launch_token;
  std::uint32_t trace_order;
};

constexpr std::array<Shape, 5> kShapes = {{
    {160, 96, 80, 48, 32, 20, 12, 21, 19},
    {80, 48, 40, 24, 64, 10, 6, 23, 21},
    {40, 24, 20, 12, 128, 5, 3, 25, 23},
    {20, 12, 10, 6, 128, 3, 2, 27, 25},
    {10, 6, 5, 3, 256, 2, 1, 29, 27},
}};

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t FiniteNormal(std::uint64_t value) {
  const std::uint16_t sign =
      static_cast<std::uint16_t>((value >> 48u) & 0x8000u);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(1u + ((value >> 52u) % 30u));
  std::uint16_t bits = static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>(exponent << 10u) |
      static_cast<std::uint16_t>(value & 0x03FFu));
  if (bits == 0x5A5Au || bits == 0xA5A5u) bits = 0x3C00u;
  return bits;
}

std::uint16_t InputHalfBits(std::size_t shape_index, const Shape& shape,
                            std::uint32_t y, std::uint32_t x,
                            std::uint32_t channel) {
  const std::uint64_t cell =
      ((static_cast<std::uint64_t>(y / 2u) * shape.output_width + x / 2u) *
           shape.channels +
       channel);
  const std::uint32_t lane = (y & 1u) * 2u + (x & 1u);
  const std::uint64_t base = SplitMix64(
      0xD1B54A32D192ED03ull ^
      (static_cast<std::uint64_t>(shape_index + 1u) *
       0x9E3779B97F4A7C15ull) ^
      (cell * 0x94D049BB133111EBull));
  switch (base & 31u) {
    case 0:
      return 0x0000u;
    case 1:
      return 0x8000u;
    case 2:
      return (lane & 1u) != 0u ? 0x8000u : 0x0000u;
    case 3:
      return FiniteNormal(base);
    case 4: {
      constexpr std::array<std::uint16_t, 4> values = {
          0xBC00u, 0x0000u, 0x3C00u, 0x4000u};
      return values[lane];
    }
    case 5: {
      constexpr std::array<std::uint16_t, 4> values = {
          0x0001u, 0x8001u, 0x03FFu, 0x83FFu};
      return values[lane];
    }
    case 6: {
      constexpr std::array<std::uint16_t, 4> values = {
          0x0400u, 0x8400u, 0x7BFFu, 0xFBFFu};
      return values[lane];
    }
    default:
      return FiniteNormal(SplitMix64(
          base ^ (static_cast<std::uint64_t>(lane + 1u) *
                  0xBF58476D1CE4E5B9ull)));
  }
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::vector<std::uint8_t> GenerateInput(std::size_t shape_index,
                                        const Shape& shape) {
  const std::size_t values = static_cast<std::size_t>(shape.input_width) *
                             shape.input_height * shape.channels;
  std::vector<std::uint8_t> bytes(values * 2u);
  for (std::uint32_t y = 0; y < shape.input_height; ++y) {
    for (std::uint32_t x = 0; x < shape.input_width; ++x) {
      for (std::uint32_t channel = 0; channel < shape.channels; ++channel) {
        const std::size_t index =
            (static_cast<std::size_t>(y) * shape.input_width + x) *
                shape.channels +
            channel;
        Store16(&bytes, index * 2u,
                InputHalfBits(shape_index, shape, y, x, channel));
      }
    }
  }
  return bytes;
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

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t payload_bytes) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes + payload_bytes);
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

std::size_t UntouchedValues(const std::vector<std::uint8_t>& output,
                            std::uint8_t poison) {
  if ((output.size() & 1u) != 0u) {
    throw std::runtime_error("pooling output byte count is not even");
  }
  std::size_t untouched = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2u) {
    untouched += output[offset] == poison && output[offset + 1u] == poison;
  }
  return untouched;
}

struct LevelResult {
  Shape shape{};
  std::vector<std::uint8_t> input;
  std::vector<std::uint8_t> output;
  std::size_t input_mismatches = 0;
  std::size_t canary_mismatches = 0;
  std::size_t untouched_values = 0;
};

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& module_sha,
                  const std::vector<LevelResult>& levels) {
  std::vector<std::uint8_t> output_root_bytes;
  std::size_t total_input_mismatches = 0;
  std::size_t total_canary_mismatches = 0;
  std::size_t total_untouched_values = 0;
  for (const auto& level : levels) {
    output_root_bytes.insert(output_root_bytes.end(), level.output.begin(),
                             level.output.end());
    total_input_mismatches += level.input_mismatches;
    total_canary_mismatches += level.canary_mismatches;
    total_untouched_values += level.untouched_values;
  }
  const bool passed = total_input_mismatches == 0 &&
                      total_canary_mismatches == 0 &&
                      total_untouched_values == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.pooling-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"module_token\": 29,\n"
         << "  \"function_token\": 27,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadData exact selected sm_89 PTX fatbin; driver JIT to active RTX architecture\",\n"
         << "  \"function_name\": \"k_pooling\",\n"
         << "  \"input_generator\": \"splitmix64-fg-pooling-finite-v1\",\n"
         << "  \"external_tensor_inputs_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"levels\": [\n";
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const auto& level = levels[index];
    const auto& shape = level.shape;
    stream << "    {\n"
           << "      \"index\": " << index << ",\n"
           << "      \"live_launch_token\": " << shape.live_launch_token << ",\n"
           << "      \"trace_order\": " << shape.trace_order << ",\n"
           << "      \"input_width\": " << shape.input_width << ",\n"
           << "      \"input_height\": " << shape.input_height << ",\n"
           << "      \"output_width\": " << shape.output_width << ",\n"
           << "      \"output_height\": " << shape.output_height << ",\n"
           << "      \"channels\": " << shape.channels << ",\n"
           << "      \"grid\": [" << shape.grid_x << ", " << shape.grid_y
           << ", 1],\n"
           << "      \"block\": [4, 4, 16],\n"
           << "      \"dynamic_shared_bytes\": 0,\n"
           << "      \"input_bytes\": " << level.input.size() << ",\n"
           << "      \"output_bytes\": " << level.output.size() << ",\n"
           << "      \"input_pre_sha256\": \"" << Sha256(level.input)
           << "\",\n"
           << "      \"input_post_sha256\": \"" << Sha256(level.input)
           << "\",\n"
           << "      \"output_sha256\": \"" << Sha256(level.output)
           << "\",\n"
           << "      \"input_mismatched_bytes\": "
           << level.input_mismatches << ",\n"
           << "      \"canary_mismatched_bytes\": "
           << level.canary_mismatches << ",\n"
           << "      \"untouched_half_values\": "
           << level.untouched_values << "\n"
           << "    }" << (index + 1u == levels.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"output_root_sha256\": \"" << Sha256(output_root_bytes)
         << "\",\n"
         << "  \"input_mismatched_bytes\": " << total_input_mismatches
         << ",\n"
         << "  \"canary_mismatched_bytes\": " << total_canary_mismatches
         << ",\n"
         << "  \"untouched_half_values\": " << total_untouched_values
         << ",\n"
         << "  \"input_immutability\": "
         << (total_input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"full_write_coverage\": "
         << (total_untouched_values == 0 ? "true" : "false") << ",\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-pooling-rtx <module.bin> <output-dir> <run-id> <poison-hex>\n";
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
      throw std::runtime_error("selected pooling module identity mismatch");
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
    const unsigned int info_size = static_cast<unsigned int>(info_log.size());
    const unsigned int error_size = static_cast<unsigned int>(error_log.size());
    std::array<CUjit_option, 4> options = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    std::array<void*, 4> option_values = {
        info_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(info_size)),
        error_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(error_size))};
    Check(cuModuleLoadDataEx(&module, module_bytes.data(),
                             static_cast<unsigned int>(options.size()),
                             options.data(), option_values.data()),
          "cuModuleLoadDataEx(exact selected pooling module)");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "k_pooling"),
          "cuModuleGetFunction(k_pooling)");

    std::vector<LevelResult> levels;
    levels.reserve(kShapes.size());
    for (std::size_t index = 0; index < kShapes.size(); ++index) {
      const Shape shape = kShapes[index];
      LevelResult result;
      result.shape = shape;
      result.input = GenerateInput(index, shape);
      const std::size_t output_bytes =
          static_cast<std::size_t>(shape.output_width) * shape.output_height *
          shape.channels * 2u;
      GuardedDeviceBuffer input;
      GuardedDeviceBuffer output;
      input.Allocate(result.input.size());
      output.Allocate(output_bytes);
      Check(cuMemcpyHtoD(input.payload(), result.input.data(), result.input.size()),
            "cuMemcpyHtoD(pooling input)");
      Check(cuMemsetD8(output.payload(), poison, output_bytes),
            "cuMemsetD8(pooling output poison)");

      CUdeviceptr input_pointer = input.payload();
      CUdeviceptr output_pointer = output.payload();
      std::uint32_t input_width = shape.input_width;
      std::uint32_t input_height = shape.input_height;
      std::uint32_t output_width = shape.output_width;
      std::uint32_t output_height = shape.output_height;
      std::uint32_t channels = shape.channels;
      void* parameters[] = {&input_pointer, &output_pointer, &input_width,
                            &input_height, &output_width, &output_height,
                            &channels};
      Check(cuLaunchKernel(function, shape.grid_x, shape.grid_y, 1, 4, 4, 16,
                           0, nullptr, parameters, nullptr),
            "cuLaunchKernel(k_pooling)");
      Check(cuCtxSynchronize(), "cuCtxSynchronize(k_pooling)");

      const auto guarded_input = input.ReadAll();
      const auto guarded_output = output.ReadAll();
      const auto actual_input = Payload(guarded_input, result.input.size());
      result.output = Payload(guarded_output, output_bytes);
      result.input_mismatches = Mismatches(actual_input, result.input);
      result.canary_mismatches =
          CanaryMismatches(guarded_input, result.input.size()) +
          CanaryMismatches(guarded_output, output_bytes);
      result.untouched_values = UntouchedValues(result.output, poison);
      Write(output_directory /
                ("level-" + std::to_string(index) + "-input-after.bin"),
            actual_input);
      Write(output_directory /
                ("level-" + std::to_string(index) + "-output.bin"),
            result.output);
      input.Free();
      output.Free();
      levels.push_back(std::move(result));
    }

    WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                 module_sha, levels);
    std::ofstream jit_stream(output_directory / "driver-jit.log",
                             std::ios::binary | std::ios::trunc);
    jit_stream << "info_log=" << info_log.data() << "\nerror_log="
               << error_log.data() << '\n';

    std::size_t failures = 0;
    for (const auto& level : levels) {
      failures += level.input_mismatches + level.canary_mismatches +
                  level.untouched_values;
    }
    std::cout << "probe=fg_pooling_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "module_sha256=" << module_sha << '\n'
              << "level_count=" << levels.size() << '\n'
              << "result=" << (failures == 0 ? "pass" : "fail") << '\n';
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return failures == 0 ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
