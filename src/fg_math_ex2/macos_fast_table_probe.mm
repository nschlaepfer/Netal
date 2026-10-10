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

constexpr std::size_t kUnitRecords = (1u << 24u) + 1u;
constexpr std::size_t kFractionRecords = (1u << 23u) + 1u;
constexpr std::size_t kUnitBytes = kUnitRecords * 4u;
constexpr std::size_t kFractionBytes = kFractionRecords * 4u;
constexpr std::size_t kGuardBytes = 256u;
constexpr std::uint8_t kGuardValue = 0xc7u;

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest)
    text << std::setw(2) << static_cast<unsigned int>(byte);
  return text.str();
}

void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write " + path.string());
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) throw std::runtime_error("Metal function missing");
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return pipeline;
}

id<MTLBuffer> Guarded(id<MTLDevice> device, std::size_t bytes, std::uint8_t poison) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  std::memset(static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes, poison, bytes);
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

void Dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
              id<MTLBuffer> output, std::size_t count) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:kGuardBytes atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage=fg-ex2-fast-table-mac <metallib> <output-dir> <run-id> [poison-hex]\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const auto library_path = std::filesystem::path(argv[1]);
      const auto output_root = std::filesystem::path(argv[2]);
      const std::string run_id = argv[3];
      const unsigned parsed_poison = argc == 5 ? std::stoul(argv[4], nullptr, 16) : 0xa5u;
      if (parsed_poison > 255u) throw std::runtime_error("invalid poison byte");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      if (std::filesystem::exists(output_root))
        throw std::runtime_error("output directory exists");
      std::filesystem::create_directories(output_root);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(library_path.c_str())]
                      error:&error];
      if (!library)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      auto unit_pipeline = Pipeline(device, library,
                                    @"fg_ex2_fast_unit_table");
      auto fraction_pipeline = Pipeline(device, library,
                                        @"fg_ex2_fast_fraction_table");
      id<MTLBuffer> unit = Guarded(device, kUnitBytes, poison);
      id<MTLBuffer> fraction = Guarded(device, kFractionBytes, poison);
      Dispatch(queue, unit_pipeline, unit, kUnitRecords);
      Dispatch(queue, fraction_pipeline, fraction, kFractionRecords);
      const auto unit_bytes = Payload(unit, kUnitBytes);
      const auto fraction_bytes = Payload(fraction, kFractionBytes);
      const std::size_t canary_mismatches =
          CanaryMismatches(unit, kUnitBytes) +
          CanaryMismatches(fraction, kFractionBytes);
      Write(output_root / "fast-unit-u32.bin", unit_bytes);
      Write(output_root / "fast-fraction-u32.bin", fraction_bytes);
      std::ofstream receipt(output_root / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      const bool pass = canary_mismatches == 0;
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.sm120-ex2-fast-table-mac.v1\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"unit_records\": " << kUnitRecords << ",\n"
              << "  \"unit_sha256\": \"" << Sha256(unit_bytes) << "\",\n"
              << "  \"fraction_records\": " << kFractionRecords << ",\n"
              << "  \"fraction_sha256\": \"" << Sha256(fraction_bytes)
              << "\",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail")
              << "\"\n"
              << "}\n";
      std::cout << "unit_sha256=" << Sha256(unit_bytes) << '\n'
                << "fraction_sha256=" << Sha256(fraction_bytes) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}

