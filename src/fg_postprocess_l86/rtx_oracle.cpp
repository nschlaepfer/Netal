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

constexpr std::array<std::uint32_t, 5> kWidths = {320, 160, 80, 40, 20};
constexpr std::array<std::uint32_t, 5> kHeights = {180, 90, 45, 23, 12};
constexpr std::array<std::size_t, 5> kLevelBytes = {
    460800, 115200, 28800, 7360, 1920};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kModuleSha[] =
    "143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0";
constexpr char kRawParameterSha[] =
    "faea46ab664868a3ad3bcdcf66dbb7350fddd82fb8d65031b2ea2cec12eb8925";
constexpr char kCanonicalParameterSha[] =
    "8f2b991ee0a0288d0f79777881720354e691b920393d921ff53a30b7f27d5fc9";
constexpr char kGenerator[] =
    "splitmix64-fg-postprocess-l86-independent-pyramids-v1";

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

std::uint16_t Load16(const std::vector<std::uint8_t>& bytes,
                     std::size_t offset) {
  return static_cast<std::uint16_t>(bytes[offset]) |
         static_cast<std::uint16_t>(bytes[offset + 1u] << 8u);
}

std::uint16_t RgbBits(std::uint32_t fixture, std::uint32_t level,
                      std::uint32_t x, std::uint32_t y,
                      std::uint32_t channel) {
  const auto mixed = SplitMix64(
      0xD6E8FEB86659FD93ull ^
      (static_cast<std::uint64_t>(fixture + 1u) * 0xA0761D6478BD642Full) ^
      (static_cast<std::uint64_t>(level + 1u) * 0xE7037ED1A0B428DBull) ^
      (static_cast<std::uint64_t>(y) * 0x8EBC6AF09C88C6E3ull) ^
      (static_cast<std::uint64_t>(x) * 0x589965CC75374CC3ull) ^
      (static_cast<std::uint64_t>(channel + 1u) *
       0x1D8E4E27C47D124Full));
  const std::uint16_t sign = ((mixed >> 63u) != 0u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(13u + ((mixed >> 19u) & 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

bool HasSupport(std::uint32_t fixture, std::uint32_t level, std::uint32_t x,
                std::uint32_t y) {
  if (fixture == 0u) {
    return ((x * 3u + y * 5u + level * 7u) % 11u) < 4u;
  }
  return ((x * 7u + y * 2u + level * 3u + 1u) % 13u) < 5u;
}

std::uint16_t AlphaBits(std::uint32_t fixture, std::uint32_t level,
                        std::uint32_t x, std::uint32_t y) {
  constexpr std::array<std::uint16_t, 4> kPositive = {
      0x3000u, 0x3400u, 0x3800u, 0x3C00u};
  return kPositive[(fixture + level + x * 3u + y * 5u) & 3u];
}

std::array<std::vector<std::uint8_t>, 5> GeneratePyramid(
    std::uint32_t fixture, std::uint8_t poison) {
  std::array<std::vector<std::uint8_t>, 5> levels;
  for (std::uint32_t level = 0; level < levels.size(); ++level) {
    levels[level].resize(kLevelBytes[level]);
    for (std::uint32_t y = 0; y < kHeights[level]; ++y) {
      for (std::uint32_t x = 0; x < kWidths[level]; ++x) {
        const std::size_t offset =
            (static_cast<std::size_t>(y) * kWidths[level] + x) * 8u;
        const bool support = HasSupport(fixture, level, x, y);
        for (std::uint32_t channel = 0; channel < 3u; ++channel) {
          Store16(&levels[level], offset + channel * 2u,
                  support ? RgbBits(fixture, level, x, y, channel)
                          : static_cast<std::uint16_t>(poison * 0x0101u));
        }
        std::uint16_t alpha = AlphaBits(fixture, level, x, y);
        if (!support) {
          // Level zero deliberately exercises both zero and negative-alpha
          // replacement. Coarser holes stay zero so their poisoned RGB is
          // mathematically unsupported and must not influence the result.
          alpha = (level == 0u && ((x + y + fixture) & 1u) != 0u)
                      ? 0xB400u
                      : 0x0000u;
        }
        Store16(&levels[level], offset + 6u, alpha);
      }
    }
  }
  return levels;
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
  void Allocate(std::size_t bytes) {
    payload_bytes = bytes;
    Check(cuMemAlloc(&base, bytes + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, bytes + 2u * kGuardBytes),
          "cuMemsetD8(guard)");
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
  ~GuardedDeviceBuffer() {
    if (base != 0) cuMemFree(base);
  }
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

void Pack64(std::array<std::uint8_t, 80>* packed, std::size_t offset,
            std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

void Pack32(std::array<std::uint8_t, 80>* packed, std::size_t offset,
            std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

std::uint8_t ParsePoison(const std::string& text) {
  const unsigned long value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

std::uint32_t ParseFixture(const std::string& text) {
  if (text == "pyramid-a") return 0u;
  if (text == "pyramid-b") return 1u;
  throw std::runtime_error("fixture must be pyramid-a or pyramid-b");
}

struct Coverage {
  std::size_t preserved_pixels = 0;
  std::size_t preservation_mismatched_bytes = 0;
  std::size_t hole_pixels = 0;
  std::size_t zero_alpha_holes = 0;
  std::size_t negative_alpha_holes = 0;
  std::size_t changed_hole_pixels = 0;
  std::size_t positive_output_holes = 0;
};

Coverage MeasureCoverage(const std::vector<std::uint8_t>& before,
                         const std::vector<std::uint8_t>& after) {
  Coverage coverage;
  const std::size_t pixels = before.size() / 8u;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const std::size_t offset = pixel * 8u;
    const std::uint16_t alpha = Load16(before, offset + 6u);
    const bool alpha_positive = (alpha & 0x8000u) == 0u &&
                                (alpha & 0x7fffu) != 0u;
    if (alpha_positive) {
      ++coverage.preserved_pixels;
      for (std::size_t byte = 0; byte < 8u; ++byte)
        coverage.preservation_mismatched_bytes +=
            before[offset + byte] != after[offset + byte];
      continue;
    }
    ++coverage.hole_pixels;
    if ((alpha & 0x7fffu) == 0u) {
      ++coverage.zero_alpha_holes;
    } else {
      ++coverage.negative_alpha_holes;
    }
    bool changed = false;
    for (std::size_t byte = 0; byte < 8u; ++byte)
      changed = changed || before[offset + byte] != after[offset + byte];
    coverage.changed_hole_pixels += changed;
    const std::uint16_t output_alpha = Load16(after, offset + 6u);
    const bool output_positive = (output_alpha & 0x8000u) == 0u &&
                                 (output_alpha & 0x7fffu) != 0u;
    coverage.positive_output_holes += output_positive;
  }
  return coverage;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: fg-postprocess-l86-rtx <module> <output-dir> <run-id> <fixture> <poison-hex> <mode>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::string fixture_name = argv[4];
    const std::uint32_t fixture = ParseFixture(fixture_name);
    const std::uint8_t poison = ParsePoison(argv[5]);
    const unsigned long requested_mode = std::stoul(argv[6], nullptr, 10);
    if (requested_mode != 0u) {
      std::cerr << "error=mode-0 specialization rejects requested mode "
                << requested_mode << '\n';
      return 4;
    }
    std::filesystem::create_directories(output_directory);

    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != 57792u || Sha256(module_bytes) != kModuleSha)
      throw std::runtime_error("selected module identity mismatch");
    const auto before = GeneratePyramid(fixture, poison);

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

    std::array<GuardedDeviceBuffer, 5> level_buffers;
    for (std::size_t level = 0; level < level_buffers.size(); ++level) {
      level_buffers[level].Allocate(kLevelBytes[level]);
      Check(cuMemcpyHtoD(level_buffers[level].payload(), before[level].data(),
                         before[level].size()),
            "cuMemcpyHtoD(level)");
    }

    std::array<std::uint8_t, 80> packed{};
    for (std::size_t level = 0; level < level_buffers.size(); ++level)
      Pack64(&packed, level * 8u, level_buffers[level].payload());
    Pack32(&packed, 40, kWidths[0]);
    Pack32(&packed, 44, kHeights[0]);
    Pack32(&packed, 48, 0u);  // strict mode-0 specialization
    Pack64(&packed, 56, 0u);  // dead texture handle in mode 0
    Pack64(&packed, 64, 0u);  // dead surface handle in mode 0
    Pack32(&packed, 72, 0u);
    Pack32(&packed, 76, 0u);
    std::size_t packed_size = packed.size();
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 256, 1, 1, 0, nullptr, nullptr,
                         extra),
          "cuLaunchKernel(main_kernel)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    std::array<std::vector<std::uint8_t>, 5> after;
    std::size_t canary_mismatches = 0;
    for (std::size_t level = 0; level < level_buffers.size(); ++level) {
      const auto guarded = level_buffers[level].ReadAll();
      after[level] = Payload(guarded, kLevelBytes[level]);
      canary_mismatches += CanaryMismatches(guarded, kLevelBytes[level]);
      Write(output_directory /
                ("level" + std::to_string(level) + "-before.bin"),
            before[level]);
      Write(output_directory /
                ("level" + std::to_string(level) + "-after.bin"),
            after[level]);
    }
    std::size_t readonly_mismatches = 0;
    for (std::size_t level = 1; level < level_buffers.size(); ++level)
      readonly_mismatches += Mismatches(before[level], after[level]);
    const Coverage coverage = MeasureCoverage(before[0], after[0]);

    const bool passed =
        canary_mismatches == 0 && readonly_mismatches == 0 &&
        coverage.preserved_pixels != 0 && coverage.hole_pixels != 0 &&
        coverage.zero_alpha_holes != 0 &&
        coverage.negative_alpha_holes != 0 &&
        coverage.preservation_mismatched_bytes == 0 &&
        coverage.changed_hole_pixels == coverage.hole_pixels &&
        coverage.positive_output_holes == coverage.hole_pixels;

    std::ofstream receipt(output_directory / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    if (!receipt) throw std::runtime_error("unable to write receipt");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.postprocess-l86-run.v1\",\n"
            << "  \"backend\": \"rtx-cuda-driver\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"fixture\": \"" << fixture_name << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"function_name\": \"main_kernel\",\n"
            << "  \"selected_launch_token\": 86,\n"
            << "  \"module_token\": 19,\n"
            << "  \"module_bytes\": 57792,\n"
            << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
            << "  \"module_transport\": \"cuModuleLoadDataEx-exact-selected-module\",\n"
            << "  \"raw_parameter_sha256\": \"" << kRawParameterSha << "\",\n"
            << "  \"address_free_parameter_sha256\": \""
            << kCanonicalParameterSha << "\",\n"
            << "  \"packed_parameter_bytes\": 80,\n"
            << "  \"mode\": 0,\n"
            << "  \"texture_handle\": 0,\n"
            << "  \"surface_handle\": 0,\n"
            << "  \"origin\": [0, 0],\n"
            << "  \"grid\": [20, 12, 1],\n"
            << "  \"block\": [256, 1, 1],\n"
            << "  \"static_shared_bytes\": 2752,\n"
            << "  \"level_dimensions\": [[320,180],[160,90],[80,45],[40,23],[20,12]],\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"level0_before_sha256\": \"" << Sha256(before[0])
            << "\",\n"
            << "  \"level0_after_sha256\": \"" << Sha256(after[0])
            << "\",\n"
            << "  \"readonly_level_mismatched_bytes\": "
            << readonly_mismatches << ",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches
            << ",\n"
            << "  \"preserved_pixels\": " << coverage.preserved_pixels
            << ",\n"
            << "  \"preservation_mismatched_bytes\": "
            << coverage.preservation_mismatched_bytes << ",\n"
            << "  \"hole_pixels\": " << coverage.hole_pixels << ",\n"
            << "  \"zero_alpha_holes\": " << coverage.zero_alpha_holes
            << ",\n"
            << "  \"negative_alpha_holes\": "
            << coverage.negative_alpha_holes << ",\n"
            << "  \"changed_hole_pixels\": "
            << coverage.changed_hole_pixels << ",\n"
            << "  \"positive_output_holes\": "
            << coverage.positive_output_holes << ",\n"
            << "  \"external_expected_tensors_consumed\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail")
            << "\"\n}\n";
    std::ofstream jit(output_directory / "driver-jit.log",
                      std::ios::binary | std::ios::trunc);
    jit << "info_log=" << info_log.data() << "\nerror_log=" << error_log.data()
        << '\n';

    std::cout << "probe=fg_postprocess_l86_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "fixture=" << fixture_name << '\n'
              << "level0_after_sha256=" << Sha256(after[0]) << '\n'
              << "readonly_level_mismatched_bytes=" << readonly_mismatches
              << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "preservation_mismatched_bytes="
              << coverage.preservation_mismatched_bytes << '\n'
              << "changed_hole_pixels=" << coverage.changed_hole_pixels
              << '\n'
              << "hole_pixels=" << coverage.hole_pixels << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    for (auto& buffer : level_buffers) buffer.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
