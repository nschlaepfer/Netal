#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
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
#include <unistd.h>

namespace {

constexpr std::size_t kIdentityRecords = 1u << 20u;
constexpr std::size_t kIdentityBytes = kIdentityRecords * 16u;
constexpr std::size_t kInputBytes = kIdentityRecords * 4u;
constexpr std::size_t kOutputBytes = kIdentityRecords * 4u;
constexpr std::size_t kUnitCorrectionBytes = 4194305u;
constexpr std::size_t kFractionCorrectionBytes = 2097153u;
constexpr std::size_t kGuardBytes = 256u;
constexpr std::uint8_t kGuardValue = 0xc7u;
constexpr char kIdentitySha256[] =
    "35081520af28773487a37dfd2f8ece07e2fbfe83544ccf1568b8a0569af6c0ee";

struct IdentityRecord {
  std::uint32_t input_bits;
  std::uint32_t direct_bits;
  std::uint32_t normalized_bits;
  std::int32_t binary_scale;
};
static_assert(sizeof(IdentityRecord) == 16);

struct CompactProbeParameters {
  std::uint32_t count;
};

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
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

std::string Sha256(const void* data, std::size_t size) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(data, static_cast<CC_LONG>(size), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest)
    text << std::setw(2) << static_cast<unsigned int>(byte);
  return text.str();
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  return Sha256(bytes.data(), bytes.size());
}

id<MTLBuffer> Guarded(id<MTLDevice> device, const void* source,
                      std::size_t bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  auto* payload = static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes;
  if (source) std::memcpy(payload, source, bytes);
  else std::memset(payload, fill, bytes);
  return buffer;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  return {pointer + kGuardBytes, pointer + kGuardBytes + bytes};
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t result = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    result += pointer[index] != kGuardValue;
    result += pointer[kGuardBytes + bytes + index] != kGuardValue;
  }
  return result;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) return std::max(left.size(), right.size());
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result += left[index] != right[index];
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage=fg-sm120-ex2-compact-mac <metallib> "
                 "<unit-corrections> <fraction-corrections> "
                 "<identity-records> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const auto library_path = std::filesystem::path(argv[1]);
      const auto unit = Read(argv[2]);
      const auto fraction = Read(argv[3]);
      const auto identity = Read(argv[4]);
      const auto output_root = std::filesystem::path(argv[5]);
      const std::string run_id = argv[6];
      char* poison_end = nullptr;
      const auto parsed_poison = std::strtoul(argv[7], &poison_end, 16);
      if (poison_end == argv[7] || *poison_end != '\0' ||
          parsed_poison > 255u)
        throw std::runtime_error("invalid poison byte");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      if (std::filesystem::exists(output_root))
        throw std::runtime_error("output directory exists");
      std::filesystem::create_directories(output_root);
      if (unit.size() != kUnitCorrectionBytes)
        throw std::runtime_error("unit correction extent mismatch");
      if (fraction.size() != kFractionCorrectionBytes)
        throw std::runtime_error("fraction correction extent mismatch");
      if (identity.size() != kIdentityBytes ||
          Sha256(identity) != kIdentitySha256)
        throw std::runtime_error("identity input mismatch");

      std::vector<std::uint8_t> inputs(kInputBytes);
      std::vector<std::uint8_t> expected(kOutputBytes);
      for (std::size_t index = 0; index < kIdentityRecords; ++index) {
        IdentityRecord record{};
        std::memcpy(&record, identity.data() + index * sizeof(record),
                    sizeof(record));
        std::memcpy(inputs.data() + index * 4u, &record.input_bits, 4u);
        std::memcpy(expected.data() + index * 4u, &record.direct_bits, 4u);
      }

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(library_path.c_str())]
                      error:&error];
      if (!library)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_sm120_ex2_compact_probe"];
      if (!function) throw std::runtime_error("probe function unavailable");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLBuffer> input = Guarded(device, inputs.data(), kInputBytes, 0u);
      id<MTLBuffer> output = Guarded(device, nullptr, kOutputBytes, poison);
      id<MTLBuffer> unit_buffer =
          [device newBufferWithBytes:unit.data() length:unit.size()
                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> fraction_buffer =
          [device newBufferWithBytes:fraction.data() length:fraction.size()
                             options:MTLResourceStorageModeShared];
      if (!queue || !input || !output || !unit_buffer || !fraction_buffer)
        throw std::runtime_error("Metal setup failed");
      const CompactProbeParameters parameters{
          static_cast<std::uint32_t>(kIdentityRecords)};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input offset:kGuardBytes atIndex:0];
      [encoder setBuffer:unit_buffer offset:0 atIndex:1];
      [encoder setBuffer:fraction_buffer offset:0 atIndex:2];
      [encoder setBuffer:output offset:kGuardBytes atIndex:3];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(kIdentityRecords, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);

      const auto actual_input = Payload(input, kInputBytes);
      const auto actual_output = Payload(output, kOutputBytes);
      const std::size_t input_mismatches = Mismatches(inputs, actual_input);
      const std::size_t output_mismatches = Mismatches(expected, actual_output);
      const std::size_t canary_mismatches =
          CanaryMismatches(input, kInputBytes) +
          CanaryMismatches(output, kOutputBytes);
      const std::size_t unit_mismatches =
          std::memcmp(unit.data(), unit_buffer.contents, unit.size()) != 0;
      const std::size_t fraction_mismatches =
          std::memcmp(fraction.data(), fraction_buffer.contents,
                      fraction.size()) != 0;
      Write(output_root / "output-u32.bin", actual_output);
      const bool pass = input_mismatches == 0 && output_mismatches == 0 &&
                        canary_mismatches == 0 && unit_mismatches == 0 &&
                        fraction_mismatches == 0;
      std::ofstream receipt(output_root / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.sm120-ex2-compact-mac-run.v1\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"input_records\": " << kIdentityRecords << ",\n"
              << "  \"input_sha256\": \"" << Sha256(inputs) << "\",\n"
              << "  \"output_sha256\": \"" << Sha256(actual_output)
              << "\",\n"
              << "  \"unit_correction_sha256\": \"" << Sha256(unit)
              << "\",\n"
              << "  \"fraction_correction_sha256\": \""
              << Sha256(fraction) << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches
              << ",\n"
              << "  \"output_mismatched_bytes\": " << output_mismatches
              << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n"
              << "  \"unit_correction_mismatched\": " << unit_mismatches
              << ",\n"
              << "  \"fraction_correction_mismatched\": "
              << fraction_mismatches << ",\n"
              << "  \"expected_output_bound_to_gpu\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail")
              << "\"\n"
              << "}\n";
      std::cout << "output_sha256=" << Sha256(actual_output) << '\n'
                << "output_mismatched_bytes=" << output_mismatches << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
