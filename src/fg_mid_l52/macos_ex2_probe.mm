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

constexpr std::size_t kDenseRecords = (2u << 21u) + 1u;
constexpr std::size_t kRandomRecords = 1u << 20u;
constexpr std::size_t kDenseBytes = kDenseRecords * 4u;
constexpr std::size_t kRandomBytes = kRandomRecords * 12u;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

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
  stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
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

id<MTLBuffer> Guarded(id<MTLDevice> device, std::size_t bytes) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  return buffer;
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t count = 0;
  for (std::size_t i = 0; i < kGuardBytes; ++i) {
    count += pointer[i] != kGuardValue;
    count += pointer[kGuardBytes + bytes + i] != kGuardValue;
  }
  return count;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  return {pointer + kGuardBytes, pointer + kGuardBytes + bytes};
}

void Dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
              id<MTLBuffer> output, std::size_t count,
              id<MTLBuffer> rom = nil) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:kGuardBytes atIndex:0];
  if (rom) [encoder setBuffer:rom offset:0 atIndex:1];
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
  if (argc != 4) {
    std::cerr << "usage: fg-mid-l52-mac-ex2 <metallib> <rcp-rom> <output-dir>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const auto metallib_path = std::filesystem::path(argv[1]);
      const auto rom = Read(argv[2]);
      const auto output_root = std::filesystem::path(argv[3]);
      if (std::filesystem::exists(output_root))
        throw std::runtime_error("output directory exists");
      std::filesystem::create_directories(output_root);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha)
        throw std::runtime_error("RCP ROM identity mismatch");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path.c_str())]
                      error:&error];
      if (!library)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      auto dense_pipeline = Pipeline(device, library, @"fg_mid_l52_dense_ex2");
      auto random_pipeline = Pipeline(device, library, @"fg_mid_l52_random_ex2");
      id<MTLBuffer> dense_buffer = Guarded(device, kDenseBytes);
      id<MTLBuffer> random_buffer = Guarded(device, kRandomBytes);
      id<MTLBuffer> rom_buffer =
          [device newBufferWithBytes:rom.data() length:rom.size()
                             options:MTLResourceStorageModeShared];
      Dispatch(queue, dense_pipeline, dense_buffer, kDenseRecords);
      Dispatch(queue, random_pipeline, random_buffer, kRandomRecords,
               rom_buffer);
      const auto dense = Payload(dense_buffer, kDenseBytes);
      const auto random = Payload(random_buffer, kRandomBytes);
      const auto canaries = CanaryMismatches(dense_buffer, kDenseBytes) +
                            CanaryMismatches(random_buffer, kRandomBytes);
      Write(output_root / "dense-ex2-minus2-to0-u32.bin", dense);
      Write(output_root / "random-negative-ex2-confidence.bin", random);
      std::ofstream receipt(output_root / "ex2-probe-receipt.json",
                            std::ios::binary | std::ios::trunc);
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.mid-l52-mac-ex2-probe.v1\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"dense_records\": " << kDenseRecords << ",\n"
              << "  \"dense_sha256\": \"" << Sha256(dense) << "\",\n"
              << "  \"random_records\": " << kRandomRecords << ",\n"
              << "  \"random_sha256\": \"" << Sha256(random) << "\",\n"
              << "  \"canary_mismatched_bytes\": " << canaries << ",\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (canaries == 0 ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "dense_sha256=" << Sha256(dense) << '\n'
                << "random_sha256=" << Sha256(random) << '\n'
                << "canary_mismatched_bytes=" << canaries << '\n'
                << "result=" << (canaries == 0 ? "pass" : "fail") << '\n';
      return canaries == 0 ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
