#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unistd.h>

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

#include "contract.h"

namespace fs = std::filesystem;
namespace c = fg_connected_subrect_l3_l83;

namespace {

struct Descriptor {
  std::uint64_t data;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t format;
  std::uint32_t channels;
};
static_assert(sizeof(Descriptor) == 24);

std::vector<std::uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read");
  return result;
}

void Write(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write");
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest) result << std::setw(2) << unsigned(byte);
  return result.str();
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> bytes(
      payload.size() + 2u * c::kGuardBytes, c::kGuardValue);
  std::memcpy(bytes.data() + c::kGuardBytes, payload.data(), payload.size());
  return [device newBufferWithBytes:bytes.data()
                              length:bytes.size()
                             options:MTLResourceStorageModeShared];
}

std::vector<std::uint8_t> All(id<MTLBuffer> buffer) {
  const auto* begin = static_cast<const std::uint8_t*>(buffer.contents);
  return {begin, begin + buffer.length};
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto all = All(buffer);
  return {all.begin() + c::kGuardBytes,
          all.begin() + c::kGuardBytes + bytes};
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) throw std::runtime_error("size drift");
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result += left[index] != right[index];
  return result;
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer,
                             std::size_t payload_bytes) {
  const auto bytes = All(buffer);
  std::size_t result = 0;
  for (std::size_t index = 0; index < c::kGuardBytes; ++index) {
    result += bytes[index] != c::kGuardValue;
    result += bytes[c::kGuardBytes + payload_bytes + index] != c::kGuardValue;
  }
  return result;
}

void Store32(std::array<std::uint8_t, 80>* bytes, std::size_t offset,
             std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

void Store64(std::array<std::uint8_t, 80>* bytes, std::size_t offset,
             std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::uint8_t Poison(const std::string& value) {
  const auto parsed = std::stoul(value, nullptr, 16);
  if (parsed > 255u) throw std::runtime_error("invalid poison");
  return static_cast<std::uint8_t>(parsed);
}

std::uint32_t Fixture(const std::string& value) {
  if (value == "mode1-a") return 0;
  if (value == "mode1-b") return 1;
  throw std::runtime_error("fixture must be mode1-a or mode1-b");
}

struct Coverage {
  std::size_t positive_mask_pixels = 0;
  std::size_t changed_positive_pixels = 0;
  std::size_t zero_mask_pixels = 0;
  std::size_t zero_mask_mismatched_bytes = 0;
  std::size_t outside_mismatched_bytes = 0;
};

Coverage Measure(const std::vector<std::uint8_t>& before,
                 const std::vector<std::uint8_t>& after,
                 const std::vector<std::uint8_t>& mask) {
  Coverage result;
  for (std::uint32_t y = 0; y < c::kResourceHeight; ++y) {
    for (std::uint32_t x = 0; x < c::kResourceWidth; ++x) {
      const std::size_t offset =
          (std::size_t{y} * c::kResourceWidth + x) * 4u;
      const bool inside = x >= c::kOriginX && x < c::kOriginX + c::kWidth &&
                          y >= c::kOriginY && y < c::kOriginY + c::kHeight;
      if (!inside) {
        for (std::size_t channel = 0; channel < 4; ++channel)
          result.outside_mismatched_bytes +=
              before[offset + channel] != after[offset + channel];
        continue;
      }
      const std::size_t local =
          std::size_t{y - c::kOriginY} * c::kWidth + x - c::kOriginX;
      if (mask[local] == 0) {
        ++result.zero_mask_pixels;
        for (std::size_t channel = 0; channel < 4; ++channel)
          result.zero_mask_mismatched_bytes +=
              before[offset + channel] != after[offset + channel];
      } else {
        ++result.positive_mask_pixels;
        bool changed = false;
        for (std::size_t channel = 0; channel < 4; ++channel)
          changed = changed || before[offset + channel] != after[offset + channel];
        result.changed_positive_pixels += changed;
      }
    }
  }
  return result;
}

}  // namespace

int main(int argc, const char* argv[]) {
  if (argc != 8) {
    std::cerr << "usage=fg-subrect-module19-mac <metallib> <rom> <output> "
                 "<run-id> <fixture> <poison> <mode>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const fs::path metallib = argv[1];
      const auto rom = Read(argv[2]);
      const fs::path output = argv[3];
      const std::string run_id = argv[4];
      const std::string fixture_name = argv[5];
      const auto fixture = Fixture(fixture_name);
      const auto poison = Poison(argv[6]);
      const auto mode = std::stoul(argv[7], nullptr, 10);
      if (mode != 1u) {
        std::cerr << "error=mode-1 specialization rejects mode " << mode << '\n';
        return 4;
      }
      if (rom.size() != c::kRcpRomBytes || Sha256(rom) != c::kRcpRomSha256)
        throw std::runtime_error("RCP ROM identity mismatch");
      fs::create_directories(output);

      const auto levels_before = c::Pyramid(fixture);
      const auto mask_before = c::Mask(fixture);
      const auto output_before = c::InitialOutput(fixture, poison, mask_before);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(metallib.string().c_str())]
                      error:&error];
      if (library == nil)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_postprocess_l86_mode1_subrect"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil)
        throw std::runtime_error(error.localizedDescription.UTF8String);

      std::array<id<MTLBuffer>, 4> levels;
      for (std::size_t index = 0; index < levels.size(); ++index)
        levels[index] = Guarded(device, levels_before[index]);
      id<MTLBuffer> mask = Guarded(device, mask_before);
      id<MTLBuffer> target = Guarded(device, output_before);
      id<MTLBuffer> rom_buffer = Guarded(device, rom);
      const Descriptor mask_descriptor{
          mask.gpuAddress + c::kGuardBytes, c::kWidth, c::kHeight, 8, 1};
      const Descriptor target_descriptor{
          target.gpuAddress + c::kGuardBytes, c::kResourceWidth,
          c::kResourceHeight, 8, 4};
      id<MTLBuffer> mask_descriptor_buffer =
          [device newBufferWithBytes:&mask_descriptor
                              length:sizeof(mask_descriptor)
                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> target_descriptor_buffer =
          [device newBufferWithBytes:&target_descriptor
                              length:sizeof(target_descriptor)
                             options:MTLResourceStorageModeShared];

      std::array<std::uint8_t, 80> parameters{};
      Store64(&parameters, 0, 0);
      for (std::size_t index = 0; index < levels.size(); ++index)
        Store64(&parameters, 8u + index * 8u,
                levels[index].gpuAddress + c::kGuardBytes);
      Store32(&parameters, 40, c::kWidth);
      Store32(&parameters, 44, c::kHeight);
      Store32(&parameters, 48, 1);
      Store64(&parameters, 56, mask_descriptor_buffer.gpuAddress);
      Store64(&parameters, 64, target_descriptor_buffer.gpuAddress);
      Store32(&parameters, 72, c::kOriginX);
      Store32(&parameters, 76, c::kOriginY);

      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
      [encoder setBuffer:rom_buffer offset:c::kGuardBytes atIndex:1];
      for (id<MTLBuffer> level : levels)
        [encoder useResource:level usage:MTLResourceUsageRead];
      [encoder useResource:mask usage:MTLResourceUsageRead];
      [encoder useResource:target
                     usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      [encoder useResource:mask_descriptor_buffer usage:MTLResourceUsageRead];
      [encoder useResource:target_descriptor_buffer usage:MTLResourceUsageRead];
      [encoder dispatchThreadgroups:MTLSizeMake(c::kGridX, c::kGridY, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);

      std::array<std::vector<std::uint8_t>, 4> levels_after;
      std::size_t input_mismatches = 0;
      std::size_t canary_mismatches = 0;
      for (std::size_t index = 0; index < levels.size(); ++index) {
        levels_after[index] = Payload(levels[index], c::kLevelBytes[index]);
        input_mismatches +=
            Mismatches(levels_before[index], levels_after[index]);
        canary_mismatches +=
            CanaryMismatches(levels[index], c::kLevelBytes[index]);
      }
      const auto mask_after = Payload(mask, c::kMaskBytes);
      const auto target_after = Payload(target, c::kOutputBytes);
      const auto rom_after = Payload(rom_buffer, c::kRcpRomBytes);
      input_mismatches += Mismatches(mask_before, mask_after);
      input_mismatches += Mismatches(rom, rom_after);
      canary_mismatches += CanaryMismatches(mask, c::kMaskBytes);
      canary_mismatches += CanaryMismatches(target, c::kOutputBytes);
      canary_mismatches += CanaryMismatches(rom_buffer, c::kRcpRomBytes);
      const Coverage coverage = Measure(output_before, target_after, mask_before);
      const auto region = c::Region(target_after);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                        coverage.positive_mask_pixels != 0 &&
                        coverage.zero_mask_pixels != 0 &&
                        coverage.changed_positive_pixels ==
                            coverage.positive_mask_pixels &&
                        coverage.zero_mask_mismatched_bytes == 0 &&
                        coverage.outside_mismatched_bytes == 0;

      Write(output / "output-full.rgba8", target_after);
      Write(output / "output-region.rgba8", region);
      Write(output / "input-mask-after.r8", mask_after);
      for (std::size_t index = 0; index < levels_after.size(); ++index)
        Write(output / ("input-level" + std::to_string(index) + "-after.bin"),
              levels_after[index]);
      std::ofstream receipt(output / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.subrect-module19-mode1-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"fixture\": \"" << fixture_name << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"source_module_token\": 19,\n"
              << "  \"source_module_sha256\": \"" << c::kModule19Sha256
              << "\",\n"
              << "  \"raw_parameter_sha256\": \""
              << c::kRawL83ParameterSha256 << "\",\n"
              << "  \"packed_parameter_bytes\": 80,\n"
              << "  \"mode\": 1,\n"
              << "  \"origin\": [32, 24],\n"
              << "  \"internal_extent\": [512, 288],\n"
              << "  \"resource_extent\": [640, 360],\n"
              << "  \"grid\": [32, 18, 1],\n"
              << "  \"block\": [256, 1, 1],\n"
              << "  \"input_generator\": \"" << c::kGenerator << "\",\n"
              << "  \"input_level_sha256\": [";
      for (std::size_t index = 0; index < levels_after.size(); ++index)
        receipt << (index ? ", \"" : "\"") << Sha256(levels_after[index])
                << "\"";
      receipt << "],\n"
              << "  \"input_mask_sha256\": \"" << Sha256(mask_after)
              << "\",\n"
              << "  \"output_full_sha256\": \"" << Sha256(target_after)
              << "\",\n"
              << "  \"output_region_bytes\": " << region.size() << ",\n"
              << "  \"output_region_sha256\": \"" << Sha256(region)
              << "\",\n"
              << "  \"positive_mask_pixels\": "
              << coverage.positive_mask_pixels << ",\n"
              << "  \"changed_positive_pixels\": "
              << coverage.changed_positive_pixels << ",\n"
              << "  \"zero_mask_pixels\": " << coverage.zero_mask_pixels
              << ",\n"
              << "  \"zero_mask_mismatched_bytes\": "
              << coverage.zero_mask_mismatched_bytes << ",\n"
              << "  \"outside_mismatched_bytes\": "
              << coverage.outside_mismatched_bytes << ",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches
              << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n"
              << "  \"expected_or_captured_output_ingress\": false,\n"
              << "  \"archived_intermediate_ingress\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "output_region_sha256=" << Sha256(region) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
