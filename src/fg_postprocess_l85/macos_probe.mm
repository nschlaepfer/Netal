#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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
#include <unistd.h>

namespace {

constexpr std::uint32_t kWidth = 40;
constexpr std::uint32_t kHeight = 23;
constexpr std::size_t kInputBytes = kWidth * kHeight * 8u;
constexpr std::array<std::size_t, 4> kLevelBytes = {1920, 480, 120, 48};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
constexpr char kModuleSha[] =
    "07877ed3dcec1977a5ddfd7843314a5044ed9559bb2d222713906afdfc609d1c";
constexpr char kParameterSha[] =
    "ac5ec30d58d9b7a47fc229acce7e3672af982298ba25e577ec4886bcdd929927";
constexpr char kGenerator[] =
    "splitmix64-fg-postprocess-l85-weighted-half4-v1";

constexpr std::uint16_t kSpecialRed[12][4] = {
    {0x3c00, 0x2d54, 0xc0bb, 0xc6ca}, {0x2aa8, 0x4477, 0x4844, 0x2987},
    {0xb2ca, 0xc1ed, 0x4198, 0xc2ec}, {0xafb8, 0xca0f, 0x4697, 0x3822},
    {0x2e20, 0x3d54, 0x3022, 0x4396}, {0xb5ba, 0xc3eb, 0xc8bb, 0x2adb},
    {0xc499, 0x2aec, 0x4455, 0xabc9}, {0xc4aa, 0xacee, 0x4976, 0x35fe},
    {0x2aec, 0x35ba, 0xb3c9, 0x32fd}, {0x2f52, 0x4143, 0xae53, 0xc5fe},
    {0xbdba, 0xbcaa, 0x4afd, 0xbe0f}, {0xc899, 0x42ec, 0x3987, 0x3965},
};
constexpr std::uint16_t kSpecialAlpha[12][4] = {
    {0x2f74, 0x2c55, 0x2aa8, 0x39dc}, {0x32ec, 0x3a20, 0x3f85, 0x3796},
    {0x2d65, 0x2fc9, 0x3adb, 0x3aa8}, {0x2822, 0x32ca, 0x3d65, 0x3400},
    {0x3998, 0x3ba7, 0x2d87, 0x2a75}, {0x2b52, 0x2ddc, 0x43b8, 0x330e},
    {0x3011, 0x28cc, 0x33da, 0x2cff}, {0x3231, 0x2e42, 0x3642, 0x3cbb},
    {0x3686, 0x3cbb, 0x3f63, 0x3877}, {0x2d65, 0x3d65, 0x41a9, 0x3d98},
    {0x2c55, 0x2921, 0x3576, 0x38cc}, {0x2fc9, 0x3253, 0x3c99, 0x3cee},
};

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
};
static_assert(sizeof(Parameters) == 8);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t RegularRgb(std::uint32_t tile, std::uint32_t lane,
                         std::uint32_t channel) {
  const auto sign_code = SplitMix64(
      0xD6E8FEB86659FD93ull ^
      (static_cast<std::uint64_t>(tile) * 0x9E3779B97F4A7C15ull) ^
      (static_cast<std::uint64_t>(channel) * 0x94D049BB133111EBull));
  const auto mixed = SplitMix64(
      sign_code ^ (static_cast<std::uint64_t>(lane + 1u) *
                   0xBF58476D1CE4E5B9ull));
  const std::uint16_t sign = (sign_code >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(12u + ((mixed >> 13u) & 5u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

std::uint16_t RegularAlpha(std::uint32_t tile, std::uint32_t lane) {
  const auto mixed = SplitMix64(
      0xA0761D6478BD642Full ^
      (static_cast<std::uint64_t>(tile + 1u) * 0xE7037ED1A0B428DBull) ^
      (static_cast<std::uint64_t>(lane + 1u) * 0x8EBC6AF09C88C6E3ull));
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(12u + ((mixed >> 19u) & 3u));
  return static_cast<std::uint16_t>((exponent << 10u) | (mixed & 0x03ffu));
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
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
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>* payload,
                      std::size_t payload_bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:payload_bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
  auto* contents = static_cast<std::uint8_t*>(buffer.contents);
  std::memset(contents, kGuardValue, buffer.length);
  if (payload != nullptr) {
    if (payload->size() != payload_bytes)
      throw std::runtime_error("guarded payload size mismatch");
    std::memcpy(contents + kGuardBytes, payload->data(), payload_bytes);
  } else {
    std::memset(contents + kGuardBytes, fill, payload_bytes);
  }
  return buffer;
}

std::vector<std::uint8_t> ReadAll(id<MTLBuffer> buffer) {
  const auto* contents = static_cast<const std::uint8_t*>(buffer.contents);
  return {contents, contents + buffer.length};
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

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t bytes) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < kGuardBytes; ++i) {
    count += guarded[i] != kGuardValue;
    count += guarded[kGuardBytes + bytes + i] != kGuardValue;
  }
  return count;
}

std::size_t UntouchedWords(const std::vector<std::uint8_t>& bytes,
                           std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < bytes.size(); i += 2u)
    count += bytes[i] == poison && bytes[i + 1u] == poison;
  return count;
}

std::uint8_t ParsePoison(const std::string& text) {
  const unsigned long value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) throw std::runtime_error("Metal function not found");
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (pipeline == nil)
    throw std::runtime_error(std::string("pipeline creation failed: ") +
                             (error ? error.localizedDescription.UTF8String
                                    : "unknown"));
  return pipeline;
}

void Dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
              const std::array<id<MTLBuffer>, 7>& buffers,
              const std::array<NSUInteger, 7>& offsets,
              std::size_t buffer_count) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  for (std::size_t i = 0; i < buffer_count; ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder dispatchThreads:MTLSizeMake(20, 12, 1)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(std::string("Metal command failed: ") +
                             (command.error
                                  ? command.error.localizedDescription.UTF8String
                                  : "unknown"));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fg-postprocess-l85-mac <metallib> <rcp-rom> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const std::filesystem::path metallib_path = argv[1];
      const std::filesystem::path rom_path = argv[2];
      const std::filesystem::path output_directory = argv[3];
      const std::string run_id = argv[4];
      const std::uint8_t poison = ParsePoison(argv[5]);
      std::filesystem::create_directories(output_directory);
      const auto input = GenerateInput();
      const auto rom = Read(rom_path);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha)
        throw std::runtime_error("SM120 RCP architecture ROM identity mismatch");

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      NSError* library_error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path.c_str())]
                      error:&library_error];
      if (library == nil)
        throw std::runtime_error(std::string("metallib load failed: ") +
                                 (library_error
                                      ? library_error.localizedDescription.UTF8String
                                      : "unknown"));
      auto exact_pipeline = Pipeline(device, library, @"fg_postprocess_l85_exact");
      auto strict_pipeline =
          Pipeline(device, library, @"fg_postprocess_l85_strict_division");

      id<MTLBuffer> input_buffer = Guarded(device, &input, kInputBytes, 0);
      std::array<id<MTLBuffer>, 4> levels;
      for (std::size_t i = 0; i < levels.size(); ++i)
        levels[i] = Guarded(device, nullptr, kLevelBytes[i], poison);
      id<MTLBuffer> strict_output =
          Guarded(device, nullptr, kLevelBytes[0], poison);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      const Parameters parameters{kWidth, kHeight};
      id<MTLBuffer> parameter_buffer =
          [device newBufferWithBytes:&parameters
                              length:sizeof(parameters)
                             options:MTLResourceStorageModeShared];
      if (parameter_buffer == nil)
        throw std::runtime_error("parameter buffer allocation failed");

      const std::array<id<MTLBuffer>, 7> exact_buffers = {
          input_buffer, levels[0], levels[1], levels[2], levels[3], rom_buffer,
          parameter_buffer};
      const std::array<NSUInteger, 7> exact_offsets = {
          kGuardBytes, kGuardBytes, kGuardBytes, kGuardBytes,
          kGuardBytes, kGuardBytes, 0};
      Dispatch(queue, exact_pipeline, exact_buffers, exact_offsets, 7);
      const std::array<id<MTLBuffer>, 7> strict_buffers = {
          input_buffer, strict_output, parameter_buffer, nil, nil, nil, nil};
      const std::array<NSUInteger, 7> strict_offsets = {kGuardBytes,
                                                        kGuardBytes, 0, 0,
                                                        0, 0, 0};
      Dispatch(queue, strict_pipeline, strict_buffers, strict_offsets, 3);

      const auto input_guarded = ReadAll(input_buffer);
      const auto actual_input = Payload(input_guarded, kInputBytes);
      const auto strict_guarded = ReadAll(strict_output);
      const auto strict_payload = Payload(strict_guarded, kLevelBytes[0]);
      const auto rom_guarded = ReadAll(rom_buffer);
      const auto actual_rom = Payload(rom_guarded, kRomBytes);
      std::array<std::vector<std::uint8_t>, 4> level_payloads;
      std::size_t canary_mismatches =
          CanaryMismatches(input_guarded, kInputBytes) +
          CanaryMismatches(strict_guarded, kLevelBytes[0]) +
          CanaryMismatches(rom_guarded, kRomBytes);
      for (std::size_t i = 0; i < levels.size(); ++i) {
        const auto guarded = ReadAll(levels[i]);
        level_payloads[i] = Payload(guarded, kLevelBytes[i]);
        canary_mismatches += CanaryMismatches(guarded, kLevelBytes[i]);
      }
      const std::size_t input_mismatches = Mismatches(input, actual_input);
      const std::size_t rom_mismatches = Mismatches(rom, actual_rom);
      const std::size_t active_untouched =
          UntouchedWords(level_payloads[0], poison);
      const std::size_t strict_untouched =
          UntouchedWords(strict_payload, poison);
      std::size_t inactive_mismatches = 0;
      for (std::size_t level = 1; level < 4; ++level)
        for (const auto byte : level_payloads[level])
          inactive_mismatches += byte != poison;

      Write(output_directory / "input-after.bin", actual_input);
      Write(output_directory / "output.bin", level_payloads[0]);
      Write(output_directory / "strict-division-output.bin", strict_payload);
      for (std::size_t i = 1; i < levels.size(); ++i)
        Write(output_directory /
                  ("inactive-level" + std::to_string(i) + "-after.bin"),
              level_payloads[i]);

      const bool passed = input_mismatches == 0 && rom_mismatches == 0 &&
                          canary_mismatches == 0 && active_untouched == 0 &&
                          strict_untouched == 0 && inactive_mismatches == 0;
      std::ofstream receipt(output_directory / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      if (!receipt) throw std::runtime_error("unable to write receipt");
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.postprocess-l85-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"metal_function\": \"fg_postprocess_l85_exact\",\n"
              << "  \"negative_control_function\": \"fg_postprocess_l85_strict_division\",\n"
              << "  \"source_launch_token\": 85,\n"
              << "  \"source_module_token\": 18,\n"
              << "  \"source_module_bytes\": 15128,\n"
              << "  \"source_module_sha256\": \"" << kModuleSha << "\",\n"
              << "  \"address_free_parameter_sha256\": \"" << kParameterSha << "\",\n"
              << "  \"abi\": {\"buffers\": [\"input_half4\", \"level0_half4\", \"level1_half4\", \"level2_half4\", \"level3_half4\"], \"level_count\": 1, \"width\": 40, \"height\": 23, \"reserved\": 0},\n"
              << "  \"input_generator\": \"" << kGenerator << "\",\n"
              << "  \"numerical_domain\": \"signed-normal-finite-rgb-positive-normal-finite-alpha-normal-denominator-and-reciprocal\",\n"
              << "  \"architecture_rom\": {\"kind\": \"sm120-mufu-rcp-2bit-correction\", \"bytes\": "
              << rom.size() << ", \"sha256\": \"" << kRomSha
              << "\", \"model_values\": 0, \"captured_intermediates\": 0},\n"
              << "  \"input_bytes\": " << actual_input.size() << ",\n"
              << "  \"input_sha256\": \"" << Sha256(actual_input) << "\",\n"
              << "  \"output_bytes\": " << level_payloads[0].size() << ",\n"
              << "  \"output_sha256\": \"" << Sha256(level_payloads[0])
              << "\",\n"
              << "  \"strict_division_output_sha256\": \""
              << Sha256(strict_payload) << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"architecture_rom_mismatched_bytes\": "
              << rom_mismatches << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n"
              << "  \"active_untouched_half_words\": " << active_untouched
              << ",\n"
              << "  \"strict_untouched_half_words\": " << strict_untouched
              << ",\n"
              << "  \"inactive_level_mismatched_bytes\": "
              << inactive_mismatches << ",\n"
              << "  \"external_expected_tensors_consumed\": false,\n"
              << "  \"archived_intermediates_consumed\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (passed ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "probe=fg_postprocess_l85_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_sha256=" << Sha256(level_payloads[0]) << '\n'
                << "strict_division_output_sha256=" << Sha256(strict_payload)
                << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
