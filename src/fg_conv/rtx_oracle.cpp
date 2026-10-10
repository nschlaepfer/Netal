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

constexpr std::uint32_t kWidth = 160;
constexpr std::uint32_t kHeight = 96;
constexpr std::uint32_t kInputChannels = 48;
constexpr std::uint32_t kOutputChannels = 32;
constexpr std::uint32_t kPaddedWidth = 162;
constexpr std::uint32_t kPaddedHeight = 98;
constexpr std::size_t kInputBytes =
    std::size_t{kWidth} * kHeight * kInputChannels * 2;
constexpr std::size_t kOutputBytes =
    std::size_t{kWidth} * kHeight * kOutputChannels * 2;
constexpr std::size_t kWeightBytes = 56320;
constexpr std::size_t kDenseWeightBytes = 27648;
constexpr std::size_t kUnusedWeightBytes = 27648;
constexpr std::size_t kBiasOffset = 55296;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kExpectedModuleBytes = 16512;
constexpr char kExpectedModuleSha256[] =
    "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d";

static_assert(kDenseWeightBytes == 3 * 3 * kInputChannels *
                                         kOutputChannels * 2);
static_assert(kBiasOffset == kDenseWeightBytes + kUnusedWeightBytes);

enum class Scenario { kSpatialMap, kOneHot, kBias };

std::string ScenarioName(Scenario scenario) {
  switch (scenario) {
    case Scenario::kSpatialMap:
      return "spatial-map";
    case Scenario::kOneHot:
      return "one-hot";
    case Scenario::kBias:
      return "bias";
  }
  throw std::runtime_error("unreachable scenario");
}

Scenario ParseScenario(const std::string& name) {
  if (name == "spatial-map") return Scenario::kSpatialMap;
  if (name == "one-hot") return Scenario::kOneHot;
  if (name == "bias") return Scenario::kBias;
  throw std::runtime_error("unknown scenario: " + name);
}

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size for " + path.string());
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

void StoreHalf(std::vector<std::uint8_t>* bytes, std::size_t offset,
               std::uint16_t value) {
  if (offset + 2 > bytes->size()) throw std::runtime_error("half store OOB");
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
}

void SetInput(std::vector<std::uint8_t>* input, std::uint32_t y,
              std::uint32_t x, std::uint32_t channel,
              std::uint16_t value) {
  const std::size_t element =
      (std::size_t{y} * kWidth + x) * kInputChannels + channel;
  StoreHalf(input, element * 2, value);
}

// PTX m16n8k16 row.col B fragments consume four consecutive half words per
// lane.  lane/4 selects output column n, while lane%4 selects pairs k=(0..7)
// and k=(8..15).  Each 3x3 spatial tile and 16-channel stage is 256 bytes.
void SetPackedWeight(std::vector<std::uint8_t>* weights,
                     std::uint32_t raw_spatial_tile,
                     std::uint32_t input_channel,
                     std::uint32_t output_channel,
                     std::uint16_t value) {
  if (raw_spatial_tile >= 9 || input_channel >= kInputChannels ||
      output_channel >= kOutputChannels) {
    throw std::runtime_error("packed weight coordinate OOB");
  }
  const std::uint32_t output_group = output_channel / 8;
  const std::uint32_t n = output_channel % 8;
  const std::uint32_t channel_stage = input_channel / 16;
  const std::uint32_t k = input_channel % 16;
  const std::uint32_t thread_in_group = (k % 8) / 2;
  const std::uint32_t fragment_half = (k >= 8 ? 2 : 0) + (k & 1u);
  const std::uint32_t lane = n * 4 + thread_in_group;
  const std::size_t tile = std::size_t{output_group} * 6912 +
                           std::size_t{raw_spatial_tile} * 768 +
                           std::size_t{channel_stage} * 256;
  const std::size_t offset = tile + std::size_t{lane} * 8 +
                             std::size_t{fragment_half} * 2;
  if (offset >= kDenseWeightBytes) {
    throw std::runtime_error("packed weight landed outside active region");
  }
  StoreHalf(weights, offset, value);
}

void SetPackedBias(std::vector<std::uint8_t>* weights,
                   std::uint32_t output_channel, std::uint16_t value) {
  if (output_channel >= kOutputChannels) {
    throw std::runtime_error("packed bias coordinate OOB");
  }
  const std::uint32_t group = output_channel / 8;
  const std::uint32_t lane = output_channel % 8;
  for (std::uint32_t repeat = 0; repeat < 16; ++repeat) {
    const std::size_t offset = kBiasOffset + std::size_t{group} * 256 +
                               std::size_t{repeat * 8 + lane} * 2;
    StoreHalf(weights, offset, value);
  }
}

struct Generated {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint8_t> input;
};

Generated Generate(Scenario scenario, std::uint8_t poison) {
  Generated generated{std::vector<std::uint8_t>(kWeightBytes, 0),
                      std::vector<std::uint8_t>(kInputBytes, 0)};
  // The selected PTX never addresses this region.  Poisoning it differently
  // across runs turns the cross-poison output check into a read-dependency
  // test instead of silently assuming the gap is unused.
  std::memset(generated.weights.data() + kDenseWeightBytes, poison,
              kUnusedWeightBytes);

  if (scenario == Scenario::kSpatialMap) {
    SetInput(&generated.input, 48, 80, 0, 0x3c00);  // +1.0
    for (std::uint32_t raw_tile = 0; raw_tile < 9; ++raw_tile) {
      SetPackedWeight(&generated.weights, raw_tile, 0, raw_tile, 0x3c00);
    }
  } else if (scenario == Scenario::kOneHot) {
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        for (std::uint32_t channel = 0; channel < kInputChannels; ++channel) {
          const std::uint16_t bits = static_cast<std::uint16_t>(
              0x3c00u + ((x + 3 * y + 5 * channel) & 7u) * 0x80u);
          SetInput(&generated.input, y, x, channel, bits);
        }
      }
    }
    for (std::uint32_t output_channel = 0;
         output_channel < kOutputChannels; ++output_channel) {
      const std::uint32_t raw_tile = output_channel % 9;
      const std::uint32_t input_channel =
          (output_channel * 13 + 7) % kInputChannels;
      const std::uint16_t weight =
          (output_channel & 1u) == 0 ? 0x3c00 : 0xbc00;  // +/-1.0
      SetPackedWeight(&generated.weights, raw_tile, input_channel,
                      output_channel, weight);
    }
  } else {
    for (std::uint32_t output_channel = 0;
         output_channel < kOutputChannels; ++output_channel) {
      const std::uint16_t bias =
          (output_channel & 1u) == 0 ? 0x3800 : 0xb800;  // +/-0.5
      SetPackedBias(&generated.weights, output_channel, bias);
    }
  }
  return generated;
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
                                  std::size_t size) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes + size);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t size) {
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += guarded[index] != kGuardValue;
    mismatches += guarded[kGuardBytes + size + index] != kGuardValue;
  }
  return mismatches;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("mismatch comparison size differs");
  }
  std::size_t count = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    count += left[index] != right[index];
  }
  return count;
}

std::size_t UntouchedHalfwords(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2) {
    count += output[offset] == poison && output[offset + 1] == poison;
  }
  return count;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, Scenario scenario,
                  std::uint8_t poison, const std::string& module_sha,
                  const Generated& generated,
                  const std::vector<std::uint8_t>& actual_weights,
                  const std::vector<std::uint8_t>& actual_input,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t untouched_halfwords) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                    untouched_halfwords == 0;
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-l20-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"scenario\": \"" << ScenarioName(scenario) << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 28,\n"
         << "  \"module_bytes\": " << kExpectedModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << module_sha << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadDataEx exact selected sm_89 PTX fatbin\",\n"
         << "  \"function_name\": \"k_conv_fp16_nhwc\",\n"
         << "  \"launch_order\": 20,\n"
         << "  \"grid\": [10, 8, 1],\n"
         << "  \"block\": [32, 4, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"abi_scalars\": [160, 96, 162, 98, 160, 96],\n"
         << "  \"input_shape_nhwc\": [1, 96, 160, 48],\n"
         << "  \"output_shape_nhwc\": [1, 96, 160, 32],\n"
         << "  \"packed_weight_bytes\": 56320,\n"
         << "  \"active_weight_bytes\": 27648,\n"
         << "  \"poisoned_gap_offset\": 27648,\n"
         << "  \"poisoned_gap_bytes\": 27648,\n"
         << "  \"bias_offset\": 55296,\n"
         << "  \"generator\": \"fg-conv-l20-synthetic-v1\",\n"
         << "  \"external_data_files_opened\": false,\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(generated.weights)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(actual_weights)
         << "\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(generated.input)
         << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(actual_input)
         << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_halfwords\": " << untouched_halfwords << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

#ifndef FG_CONV_L20_NO_MAIN
int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage=fg-conv-l20-rtx <module.bin> <output-dir> <run-id> "
                 "<poison-hex> <spatial-map|one-hot|bias>\n";
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
    const Scenario scenario = ParseScenario(argv[5]);
    std::filesystem::create_directories(output_directory);

    const auto module_bytes = Read(module_path);
    const std::string module_sha = Sha256(module_bytes);
    if (module_bytes.size() != kExpectedModuleBytes ||
        module_sha != kExpectedModuleSha256) {
      throw std::runtime_error("selected module identity mismatch");
    }
    const Generated generated = Generate(scenario, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()),
          "cuModuleLoadData(exact selected module)");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "k_conv_fp16_nhwc"),
          "cuModuleGetFunction(k_conv_fp16_nhwc)");

    GuardedDeviceBuffer weights;
    GuardedDeviceBuffer input;
    GuardedDeviceBuffer output;
    weights.Allocate(kWeightBytes);
    input.Allocate(kInputBytes);
    output.Allocate(kOutputBytes);
    Check(cuMemcpyHtoD(weights.payload(), generated.weights.data(),
                       generated.weights.size()),
          "cuMemcpyHtoD(weights)");
    Check(cuMemcpyHtoD(input.payload(), generated.input.data(),
                       generated.input.size()),
          "cuMemcpyHtoD(input)");
    Check(cuMemsetD8(output.payload(), poison, kOutputBytes),
          "cuMemsetD8(output poison)");

    CUdeviceptr weight_pointer = weights.payload();
    CUdeviceptr input_pointer = input.payload();
    CUdeviceptr output_pointer = output.payload();
    std::uint32_t width = kWidth;
    std::uint32_t height = kHeight;
    std::uint32_t padded_width = kPaddedWidth;
    std::uint32_t padded_height = kPaddedHeight;
    std::uint32_t output_width = kWidth;
    std::uint32_t output_height = kHeight;
    void* parameters[] = {
        &weight_pointer, &input_pointer,  &width,
        &height,         &padded_width,   &padded_height,
        &output_pointer, &output_width,   &output_height,
    };
    Check(cuLaunchKernel(function, 10, 8, 1, 32, 4, 1, 0, nullptr,
                         parameters, nullptr),
          "cuLaunchKernel(k_conv_fp16_nhwc)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto guarded_weights = weights.ReadAll();
    const auto guarded_input = input.ReadAll();
    const auto guarded_output = output.ReadAll();
    const auto actual_weights = Payload(guarded_weights, kWeightBytes);
    const auto actual_input = Payload(guarded_input, kInputBytes);
    const auto actual_output = Payload(guarded_output, kOutputBytes);
    const std::size_t input_mismatches =
        Mismatches(actual_weights, generated.weights) +
        Mismatches(actual_input, generated.input);
    const std::size_t canary_mismatches =
        CanaryMismatches(guarded_weights, kWeightBytes) +
        CanaryMismatches(guarded_input, kInputBytes) +
        CanaryMismatches(guarded_output, kOutputBytes);
    const std::size_t untouched_halfwords =
        UntouchedHalfwords(actual_output, poison);

    Write(output_directory / "weights-after.bin", actual_weights);
    Write(output_directory / "input-after.bin", actual_input);
    Write(output_directory / "output.bin", actual_output);
    WriteReceipt(output_directory / "run-receipt.json", run_id, scenario,
                 poison, module_sha, generated, actual_weights, actual_input,
                 actual_output, input_mismatches, canary_mismatches,
                 untouched_halfwords);

    const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                      untouched_halfwords == 0;
    std::cout << "probe=fg_conv_l20_rtx\n"
              << "scenario=" << ScenarioName(scenario) << '\n'
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "module_sha256=" << module_sha << '\n'
              << "output_sha256=" << Sha256(actual_output) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "untouched_halfwords=" << untouched_halfwords << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';

    weights.Free();
    input.Free();
    output.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return pass ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
#endif
