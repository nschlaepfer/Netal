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

constexpr std::uint32_t kWidth = 40;
constexpr std::uint32_t kHeight = 23;
constexpr std::uint32_t kLevelCount = 1;
constexpr std::uint32_t kReserved = 0;
constexpr std::size_t kInputBytes = kWidth * kHeight * 8u;
constexpr std::array<std::size_t, 4> kLevelBytes = {1920, 480, 120, 48};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kModuleSha[] =
    "07877ed3dcec1977a5ddfd7843314a5044ed9559bb2d222713906afdfc609d1c";
constexpr char kParameterSha[] =
    "ac5ec30d58d9b7a47fc229acce7e3672af982298ba25e577ec4886bcdd929927";
constexpr char kGenerator[] =
    "splitmix64-fg-postprocess-l85-weighted-half4-v1";

constexpr std::uint16_t kSpecialRed[12][4] = {
    {0x3c00, 0x2d54, 0xc0bb, 0xc6ca},
    {0x2aa8, 0x4477, 0x4844, 0x2987},
    {0xb2ca, 0xc1ed, 0x4198, 0xc2ec},
    {0xafb8, 0xca0f, 0x4697, 0x3822},
    {0x2e20, 0x3d54, 0x3022, 0x4396},
    {0xb5ba, 0xc3eb, 0xc8bb, 0x2adb},
    {0xc499, 0x2aec, 0x4455, 0xabc9},
    {0xc4aa, 0xacee, 0x4976, 0x35fe},
    {0x2aec, 0x35ba, 0xb3c9, 0x32fd},
    {0x2f52, 0x4143, 0xae53, 0xc5fe},
    {0xbdba, 0xbcaa, 0x4afd, 0xbe0f},
    {0xc899, 0x42ec, 0x3987, 0x3965},
};

constexpr std::uint16_t kSpecialAlpha[12][4] = {
    {0x2f74, 0x2c55, 0x2aa8, 0x39dc},
    {0x32ec, 0x3a20, 0x3f85, 0x3796},
    {0x2d65, 0x2fc9, 0x3adb, 0x3aa8},
    {0x2822, 0x32ca, 0x3d65, 0x3400},
    {0x3998, 0x3ba7, 0x2d87, 0x2a75},
    {0x2b52, 0x2ddc, 0x43b8, 0x330e},
    {0x3011, 0x28cc, 0x33da, 0x2cff},
    {0x3231, 0x2e42, 0x3642, 0x3cbb},
    {0x3686, 0x3cbb, 0x3f63, 0x3877},
    {0x2d65, 0x3d65, 0x41a9, 0x3d98},
    {0x2c55, 0x2921, 0x3576, 0x38cc},
    {0x2fc9, 0x3253, 0x3c99, 0x3cee},
};

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

std::uint16_t RegularRgb(std::uint32_t tile, std::uint32_t lane,
                         std::uint32_t channel) {
  const std::uint64_t sign_code =
      SplitMix64(0xD6E8FEB86659FD93ull ^
                 (static_cast<std::uint64_t>(tile) *
                  0x9E3779B97F4A7C15ull) ^
                 (static_cast<std::uint64_t>(channel) *
                  0x94D049BB133111EBull));
  const std::uint64_t mixed =
      SplitMix64(sign_code ^
                 (static_cast<std::uint64_t>(lane + 1u) *
                  0xBF58476D1CE4E5B9ull));
  const std::uint16_t sign = (sign_code >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(12u + ((mixed >> 13u) & 5u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

std::uint16_t RegularAlpha(std::uint32_t tile, std::uint32_t lane) {
  const std::uint64_t mixed =
      SplitMix64(0xA0761D6478BD642Full ^
                 (static_cast<std::uint64_t>(tile + 1u) *
                  0xE7037ED1A0B428DBull) ^
                 (static_cast<std::uint64_t>(lane + 1u) *
                  0x8EBC6AF09C88C6E3ull));
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(12u + ((mixed >> 19u) & 3u));
  return static_cast<std::uint16_t>((exponent << 10u) | (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateInput() {
  std::vector<std::uint8_t> bytes(kInputBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::uint32_t tile = (y / 2u) * 20u + x / 2u;
      const std::uint32_t lane = (y & 1u) * 2u + (x & 1u);
      const std::size_t offset =
          (static_cast<std::size_t>(y) * kWidth + x) * 8u;
      Store16(&bytes, offset,
              tile < 12u ? kSpecialRed[tile][lane]
                         : RegularRgb(tile, lane, 0u));
      Store16(&bytes, offset + 2u, RegularRgb(tile, lane, 1u));
      Store16(&bytes, offset + 4u, RegularRgb(tile, lane, 2u));
      Store16(&bytes, offset + 6u,
              tile < 12u ? kSpecialAlpha[tile][lane]
                         : RegularAlpha(tile, lane));
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
  DWORD object_bytes = 0, digest_bytes = 0, written = 0;
  auto require = [](NTSTATUS status, const char* operation) {
    if (status < 0) throw std::runtime_error(std::string(operation) + " failed");
  };
  require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                      nullptr, 0),
          "BCryptOpenAlgorithmProvider");
  std::vector<std::uint8_t> object;
  std::vector<std::uint8_t> digest;
  try {
    require(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                              reinterpret_cast<PUCHAR>(&object_bytes),
                              sizeof(object_bytes), &written, 0),
            "BCryptGetProperty(object)");
    require(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                              reinterpret_cast<PUCHAR>(&digest_bytes),
                              sizeof(digest_bytes), &written, 0),
            "BCryptGetProperty(digest)");
    object.resize(object_bytes);
    digest.resize(digest_bytes);
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
  } catch (...) {
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw;
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

void Check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name ? name : "unknown") + " / " +
                           (description ? description : "unknown"));
}

struct GuardedDeviceBuffer {
  CUdeviceptr base = 0;
  std::size_t payload_bytes = 0;
  CUdeviceptr payload() const { return base + kGuardBytes; }
  void Allocate(std::size_t bytes, std::uint8_t fill) {
    payload_bytes = bytes;
    Check(cuMemAlloc(&base, bytes + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, bytes + 2u * kGuardBytes),
          "cuMemsetD8(guard)");
    Check(cuMemsetD8(payload(), fill, bytes), "cuMemsetD8(payload)");
  }
  std::vector<std::uint8_t> ReadAll() const {
    std::vector<std::uint8_t> bytes(payload_bytes + 2u * kGuardBytes);
    Check(cuMemcpyDtoH(bytes.data(), base, bytes.size()), "cuMemcpyDtoH");
    return bytes;
  }
  void Free() {
    if (base != 0) Check(cuMemFree(base), "cuMemFree");
    base = 0;
  }
  ~GuardedDeviceBuffer() { if (base != 0) cuMemFree(base); }
};

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t bytes) {
  return {guarded.begin() + kGuardBytes,
          guarded.begin() + kGuardBytes + bytes};
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& bytes,
                             std::size_t payload_bytes) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < kGuardBytes; ++i) {
    count += bytes[i] != kGuardValue;
    count += bytes[kGuardBytes + payload_bytes + i] != kGuardValue;
  }
  return count;
}

std::size_t UntouchedWords(const std::vector<std::uint8_t>& bytes,
                           std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < bytes.size(); i += 2u) {
    count += bytes[i] == poison && bytes[i + 1u] == poison;
  }
  return count;
}

void Pack64(std::array<std::uint8_t, 56>* packed, std::size_t offset,
            std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

void Pack32(std::array<std::uint8_t, 56>* packed, std::size_t offset,
            std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

std::uint8_t ParsePoison(const std::string& text) {
  const unsigned long value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::vector<std::uint8_t>& input,
                  const std::vector<std::uint8_t>& output,
                  const std::array<std::vector<std::uint8_t>, 3>& inactive,
                  std::size_t input_mismatches, std::size_t canary_mismatches,
                  std::size_t untouched_words,
                  std::size_t inactive_mismatches) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("unable to write receipt");
  out << "{\n"
      << "  \"schema\": \"neuralstudio.fg.postprocess-l85-run.v1\",\n"
      << "  \"backend\": \"rtx-cuda-driver\",\n"
      << "  \"run_id\": \"" << run_id << "\",\n"
      << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
      << "  \"poison_byte\": " << unsigned(poison) << ",\n"
      << "  \"function_name\": \"main_kernel\",\n"
      << "  \"selected_launch_token\": 85,\n"
      << "  \"module_token\": 18,\n"
      << "  \"module_bytes\": 15128,\n"
      << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
      << "  \"module_transport\": \"cuModuleLoadDataEx-exact-selected-module\",\n"
      << "  \"address_free_parameter_sha256\": \"" << kParameterSha << "\",\n"
      << "  \"packed_parameter_bytes\": 56,\n"
      << "  \"abi\": {\"buffers\": [\"input_half4\", \"level0_half4\", \"level1_half4\", \"level2_half4\", \"level3_half4\"], \"level_count\": 1, \"width\": 40, \"height\": 23, \"reserved\": 0},\n"
      << "  \"grid\": [3, 2, 1],\n"
      << "  \"block\": [256, 1, 1],\n"
      << "  \"input_generator\": \"" << kGenerator << "\",\n"
      << "  \"numerical_domain\": \"signed-normal-finite-rgb-positive-normal-finite-alpha-normal-denominator-and-reciprocal\",\n"
      << "  \"input_bytes\": " << input.size() << ",\n"
      << "  \"input_sha256\": \"" << Sha256(input) << "\",\n"
      << "  \"output_bytes\": " << output.size() << ",\n"
      << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
      << "  \"inactive_level_bytes\": [" << inactive[0].size() << ", "
      << inactive[1].size() << ", " << inactive[2].size() << "],\n"
      << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
      << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
      << "  \"active_untouched_half_words\": " << untouched_words << ",\n"
      << "  \"inactive_level_mismatched_bytes\": " << inactive_mismatches << ",\n"
      << "  \"external_expected_tensors_consumed\": false,\n"
      << "  \"archived_intermediates_consumed\": false,\n"
      << "  \"address_fields_recorded\": false,\n"
      << "  \"result\": \""
      << ((input_mismatches == 0 && canary_mismatches == 0 &&
           untouched_words == 0 && inactive_mismatches == 0)
              ? "pass"
              : "fail")
      << "\"\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: fg-postprocess-l85-rtx <module> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    std::filesystem::create_directories(output_directory);

    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != 15128u || Sha256(module_bytes) != kModuleSha)
      throw std::runtime_error("selected module identity mismatch");
    const auto input = GenerateInput();

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    std::array<char, 8192> info_log{};
    std::array<char, 8192> error_log{};
    unsigned int info_size = static_cast<unsigned int>(info_log.size());
    unsigned int error_size = static_cast<unsigned int>(error_log.size());
    CUjit_option options[] = {CU_JIT_INFO_LOG_BUFFER,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* option_values[] = {info_log.data(), &info_size, error_log.data(),
                             &error_size};
    CUmodule module = nullptr;
    Check(cuModuleLoadDataEx(&module, module_bytes.data(), 4, options,
                             option_values),
          "cuModuleLoadDataEx");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "cuModuleGetFunction");

    GuardedDeviceBuffer input_buffer;
    std::array<GuardedDeviceBuffer, 4> levels;
    input_buffer.Allocate(kInputBytes, 0);
    for (std::size_t i = 0; i < levels.size(); ++i)
      levels[i].Allocate(kLevelBytes[i], poison);
    Check(cuMemcpyHtoD(input_buffer.payload(), input.data(), input.size()),
          "cuMemcpyHtoD(input)");

    std::array<std::uint8_t, 56> packed{};
    Pack64(&packed, 0, input_buffer.payload());
    for (std::size_t i = 0; i < levels.size(); ++i)
      Pack64(&packed, 8u + i * 8u, levels[i].payload());
    Pack32(&packed, 40, kLevelCount);
    Pack32(&packed, 44, kWidth);
    Pack32(&packed, 48, kHeight);
    Pack32(&packed, 52, kReserved);
    std::size_t packed_size = packed.size();
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 3, 2, 1, 256, 1, 1, 0, nullptr, nullptr,
                         extra),
          "cuLaunchKernel(main_kernel)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto input_guarded = input_buffer.ReadAll();
    const auto actual_input = Payload(input_guarded, kInputBytes);
    std::array<std::vector<std::uint8_t>, 4> level_payloads;
    std::size_t canary_mismatches =
        CanaryMismatches(input_guarded, kInputBytes);
    for (std::size_t i = 0; i < levels.size(); ++i) {
      const auto guarded = levels[i].ReadAll();
      level_payloads[i] = Payload(guarded, kLevelBytes[i]);
      canary_mismatches += CanaryMismatches(guarded, kLevelBytes[i]);
    }
    const std::size_t input_mismatches = Mismatches(input, actual_input);
    const std::size_t untouched_words =
        UntouchedWords(level_payloads[0], poison);
    std::size_t inactive_mismatches = 0;
    for (std::size_t level = 1; level < 4; ++level)
      for (const auto byte : level_payloads[level])
        inactive_mismatches += byte != poison;

    Write(output_directory / "input-after.bin", actual_input);
    Write(output_directory / "output.bin", level_payloads[0]);
    for (std::size_t i = 1; i < levels.size(); ++i)
      Write(output_directory /
                ("inactive-level" + std::to_string(i) + "-after.bin"),
            level_payloads[i]);
    std::array<std::vector<std::uint8_t>, 3> inactive = {
        level_payloads[1], level_payloads[2], level_payloads[3]};
    WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                 actual_input, level_payloads[0], inactive, input_mismatches,
                 canary_mismatches, untouched_words, inactive_mismatches);
    std::ofstream jit(output_directory / "driver-jit.log",
                      std::ios::binary | std::ios::trunc);
    jit << "info_log=" << info_log.data() << "\nerror_log=" << error_log.data()
        << '\n';

    const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                        untouched_words == 0 && inactive_mismatches == 0;
    std::cout << "probe=fg_postprocess_l85_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(level_payloads[0]) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "active_untouched_half_words=" << untouched_words << '\n'
              << "inactive_level_mismatched_bytes=" << inactive_mismatches
              << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    input_buffer.Free();
    for (auto& level : levels) level.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
