#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<std::uint32_t, 5> kWidths = {320, 160, 80, 40, 20};
constexpr std::array<std::uint32_t, 5> kHeights = {180, 90, 45, 23, 12};
constexpr std::array<std::size_t, 5> kLevelBytes = {
    460800, 115200, 28800, 7360, 1920};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kRcpRomBytes = 2097152;
constexpr char kRcpRomSha[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
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
  if (fixture == 0u)
    return ((x * 3u + y * 5u + level * 7u) % 11u) < 4u;
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
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

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

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>& payload) {
  const std::size_t bytes = payload.size() + 2u * kGuardBytes;
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, bytes);
  std::memcpy(static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes,
              payload.data(), payload.size());
  return buffer;
}

std::vector<std::uint8_t> ReadAll(id<MTLBuffer> buffer) {
  const auto* begin = static_cast<const std::uint8_t*>(buffer.contents);
  return {begin, begin + buffer.length};
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
  for (std::size_t offset = 0; offset < before.size(); offset += 8u) {
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
    if ((alpha & 0x7fffu) == 0u)
      ++coverage.zero_alpha_holes;
    else
      ++coverage.negative_alpha_holes;
    bool changed = false;
    for (std::size_t byte = 0; byte < 8u; ++byte)
      changed = changed || before[offset + byte] != after[offset + byte];
    coverage.changed_hole_pixels += changed;
    const std::uint16_t output_alpha = Load16(after, offset + 6u);
    coverage.positive_output_holes +=
        (output_alpha & 0x8000u) == 0u && (output_alpha & 0x7fffu) != 0u;
  }
  return coverage;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage: fg-postprocess-l86-mac <metallib> <rcp-rom> <output-dir> <run-id> <fixture> <poison-hex> <mode> <kernel-name>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const std::filesystem::path metallib_path = argv[1];
      const std::filesystem::path rom_path = argv[2];
      const std::filesystem::path output_directory = argv[3];
      const std::string run_id = argv[4];
      const std::string fixture_name = argv[5];
      const std::uint32_t fixture = ParseFixture(fixture_name);
      const std::uint8_t poison = ParsePoison(argv[6]);
      const unsigned long requested_mode = std::stoul(argv[7], nullptr, 10);
      const std::string kernel_name = argv[8];
      if (requested_mode != 0u) {
        std::cerr << "error=mode-0 specialization rejects requested mode "
                  << requested_mode << '\n';
        return 4;
      }
      std::filesystem::create_directories(output_directory);
      const auto rom = Read(rom_path);
      if (rom.size() != kRcpRomBytes || Sha256(rom) != kRcpRomSha)
        throw std::runtime_error("SM120 reciprocal ROM identity mismatch");
      const auto before = GeneratePyramid(fixture, poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* library_error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path.c_str())]
                      error:&library_error];
      if (library == nil)
        throw std::runtime_error(std::string("metallib load failed: ") +
                                 (library_error
                                      ? library_error.localizedDescription.UTF8String
                                      : "unknown"));
      id<MTLFunction> function = [library
          newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
      if (function == nil) throw std::runtime_error("Metal function missing");
      NSError* pipeline_error = nil;
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&pipeline_error];
      if (pipeline == nil)
        throw std::runtime_error(std::string("pipeline creation failed: ") +
                                 (pipeline_error
                                      ? pipeline_error.localizedDescription.UTF8String
                                      : "unknown"));
      if (pipeline.maxTotalThreadsPerThreadgroup < 256u)
        throw std::runtime_error("pipeline cannot dispatch selected 256-thread block");

      std::array<id<MTLBuffer>, 5> levels;
      for (std::size_t level = 0; level < levels.size(); ++level)
        levels[level] = Guarded(device, before[level]);
      id<MTLBuffer> rom_buffer = Guarded(device, rom);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      for (NSUInteger level = 0; level < levels.size(); ++level)
        [encoder setBuffer:levels[level] offset:kGuardBytes atIndex:level];
      [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(std::string("Metal command failed: ") +
                                 (command.error
                                      ? command.error.localizedDescription.UTF8String
                                      : "unknown"));

      std::array<std::vector<std::uint8_t>, 5> after;
      std::size_t canary_mismatches = 0;
      for (std::size_t level = 0; level < levels.size(); ++level) {
        const auto guarded = ReadAll(levels[level]);
        after[level] = Payload(guarded, kLevelBytes[level]);
        canary_mismatches += CanaryMismatches(guarded, kLevelBytes[level]);
        Write(output_directory /
                  ("level" + std::to_string(level) + "-before.bin"),
              before[level]);
        Write(output_directory /
                  ("level" + std::to_string(level) + "-after.bin"),
              after[level]);
      }
      const auto guarded_rom = ReadAll(rom_buffer);
      const auto after_rom = Payload(guarded_rom, kRcpRomBytes);
      canary_mismatches += CanaryMismatches(guarded_rom, kRcpRomBytes);
      const std::size_t rom_mismatches = Mismatches(rom, after_rom);
      std::size_t readonly_mismatches = 0;
      for (std::size_t level = 1; level < levels.size(); ++level)
        readonly_mismatches += Mismatches(before[level], after[level]);
      const Coverage coverage = MeasureCoverage(before[0], after[0]);
      const bool passed =
          canary_mismatches == 0 && rom_mismatches == 0 &&
          readonly_mismatches == 0 && coverage.preserved_pixels != 0 &&
          coverage.hole_pixels != 0 && coverage.zero_alpha_holes != 0 &&
          coverage.negative_alpha_holes != 0 &&
          coverage.preservation_mismatched_bytes == 0 &&
          coverage.changed_hole_pixels == coverage.hole_pixels &&
          coverage.positive_output_holes == coverage.hole_pixels;

      std::ofstream receipt(output_directory / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      if (!receipt) throw std::runtime_error("unable to write receipt");
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.postprocess-l86-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"fixture\": \"" << fixture_name << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"metal_function\": \"" << kernel_name << "\",\n"
              << "  \"source_launch_token\": 86,\n"
              << "  \"source_module_token\": 19,\n"
              << "  \"source_module_bytes\": 57792,\n"
              << "  \"source_module_sha256\": \"" << kModuleSha << "\",\n"
              << "  \"raw_parameter_sha256\": \"" << kRawParameterSha
              << "\",\n"
              << "  \"address_free_parameter_sha256\": \""
              << kCanonicalParameterSha << "\",\n"
              << "  \"mode\": 0,\n"
              << "  \"grid\": [20, 12, 1],\n"
              << "  \"block\": [256, 1, 1],\n"
              << "  \"level_dimensions\": [[320,180],[160,90],[80,45],[40,23],[20,12]],\n"
              << "  \"input_generator\": \"" << kGenerator << "\",\n"
              << "  \"architecture_rom\": {\"kind\": \"sm120-mufu-rcp-2bit-correction\", \"bytes\": "
              << rom.size() << ", \"sha256\": \"" << kRcpRomSha
              << "\", \"model_values\": 0, \"archived_intermediates\": 0},\n"
              << "  \"level0_before_sha256\": \"" << Sha256(before[0])
              << "\",\n"
              << "  \"level0_after_sha256\": \"" << Sha256(after[0])
              << "\",\n"
              << "  \"readonly_level_mismatched_bytes\": "
              << readonly_mismatches << ",\n"
              << "  \"architecture_rom_mismatched_bytes\": "
              << rom_mismatches << ",\n"
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
      std::cout << "probe=fg_postprocess_l86_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
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
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
