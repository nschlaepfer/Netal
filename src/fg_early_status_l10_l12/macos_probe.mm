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

#include "contract.h"

namespace {
namespace contract = fg_early_status_l10_l12;
constexpr std::size_t kGuardBytes = 256;

struct Parameters {
  std::uint32_t count;
  std::uint32_t value;
};
static_assert(sizeof(Parameters) == 8);

struct SourceParameters {
  std::uint64_t output;
  std::uint32_t count;
  std::uint32_t value;
};
static_assert(sizeof(SourceParameters) == 16);

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison");
  return static_cast<std::uint8_t>(value);
}

}  // namespace

int main(int argc, const char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: fg-early-status-mac <metallib> <output-dir> <run-id> <poison-hex> <reset|generated>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const std::string mode = argv[5];
      if (mode != "reset" && mode != "generated")
        throw std::runtime_error("unsupported status mode");
      const std::uint8_t poison = ParsePoison(argv[4]);
      const std::filesystem::path output_root = argv[2];
      const std::string run_id = argv[3];
      std::filesystem::create_directories(output_root);
      const std::uint32_t value = mode == "reset" ? contract::kResetValue
                                                    : contract::kGeneratedValue;
      const char* trace_raw_sha =
          mode == "reset" ? contract::kResetTraceRawParameterSha256
                           : contract::kGeneratedTraceRawParameterSha256;
      const char* address_free_sha =
          mode == "reset" ? contract::kResetAddressFreeParameterSha256
                           : contract::kGeneratedAddressFreeParameterSha256;
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (!library) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_early_status_write"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) throw std::runtime_error("pipeline creation failed");
      id<MTLBuffer> output =
          [device newBufferWithLength:sizeof(std::uint32_t) + 2u * kGuardBytes
                              options:MTLResourceStorageModeShared];
      if (!output) throw std::runtime_error("output allocation failed");
      std::memset(output.contents, poison, output.length);
      const Parameters parameters{contract::kCount, value};
      const SourceParameters address_free_parameters{0, contract::kCount,
                                                     value};
      const auto* address_free_begin = reinterpret_cast<const std::uint8_t*>(
          &address_free_parameters);
      const std::vector<std::uint8_t> address_free_parameter_bytes(
          address_free_begin,
          address_free_begin + sizeof(address_free_parameters));
      const std::string live_address_free_sha =
          Sha256(address_free_parameter_bytes);
      if (live_address_free_sha != address_free_sha)
        throw std::runtime_error(
            "live address-free parameter identity mismatch");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:output offset:kGuardBytes atIndex:0];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(64, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal command failed");
      const auto* guarded = static_cast<const std::uint8_t*>(output.contents);
      std::vector<std::uint8_t> payload(
          guarded + kGuardBytes,
          guarded + kGuardBytes + sizeof(std::uint32_t));
      std::uint32_t actual = 0;
      std::memcpy(&actual, payload.data(), sizeof(actual));
      std::size_t canary_mismatches = 0;
      for (std::size_t index = 0; index < kGuardBytes; ++index) {
        canary_mismatches += guarded[index] != poison;
        canary_mismatches +=
            guarded[kGuardBytes + sizeof(std::uint32_t) + index] != poison;
      }
      std::size_t untouched_payload_bytes = 0;
      for (const auto byte : payload) untouched_payload_bytes += byte == poison;
      const bool pass = actual == value && canary_mismatches == 0 &&
                        untouched_payload_bytes == 0;
      Write(output_root / "output-u32.bin", payload);
      std::ofstream receipt(output_root / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.early-status-l10-l12-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"mode\": \"" << mode << "\",\n"
              << "  \"canonical_launch\": " << (mode == "reset" ? 10 : 12)
              << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"device_name\": \"" << device.name.UTF8String << "\",\n"
              << "  \"source_module_token\": 27,\n"
              << "  \"source_module_bytes\": " << contract::kModuleBytes
              << ",\n"
              << "  \"source_module_sha256\": \"" << contract::kModuleSha256
              << "\",\n"
              << "  \"trace_raw_parameter_sha256\": \"" << trace_raw_sha
              << "\",\n"
              << "  \"live_address_free_parameter_sha256\": \""
              << live_address_free_sha << "\",\n"
              << "  \"address_free_parameter_sha256\": \""
              << address_free_sha << "\",\n"
              << "  \"packed_parameter_bytes\": 16,\n"
              << "  \"source_grid\": [1, 1, 1],\n"
              << "  \"source_block\": [64, 1, 1],\n"
              << "  \"tensor_input_count\": 0,\n"
              << "  \"input_generator\": \"" << contract::kInputGenerator
              << "\",\n"
              << "  \"element_count\": 1,\n"
              << "  \"requested_value\": " << value << ",\n"
              << "  \"actual_value\": " << actual << ",\n"
              << "  \"output_bytes\": 4,\n"
              << "  \"output_sha256\": \"" << Sha256(payload) << "\",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n"
              << "  \"untouched_payload_bytes\": " << untouched_payload_bytes
              << ",\n"
              << "  \"expected_output_fed_to_execution\": false,\n"
              << "  \"archived_intermediates_consumed\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
              << "}\n";
      std::cout << "probe=fg_early_status_l10_l12_mac\n"
                << "run_id=" << run_id << '\n'
                << "mode=" << mode << '\n'
                << "actual_value=" << actual << '\n'
                << "output_sha256=" << Sha256(payload) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
