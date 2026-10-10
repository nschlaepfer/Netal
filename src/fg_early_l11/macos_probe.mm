#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdlib>
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
namespace l11 = fg_early_l11;

struct EarlyL11Parameters {
  std::uint32_t input_origin[2];
  std::uint32_t output_origin[2];
  std::uint32_t copy_extent[2];
};
static_assert(sizeof(EarlyL11Parameters) == 24);

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (unsigned char byte : digest) text << std::setw(2) << unsigned(byte);
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

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) throw std::runtime_error("size mismatch");
  std::size_t count = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    count += left[index] != right[index];
  }
  return count;
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLTextureUsage usage,
                           const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != l11::kTextureBytes) {
    throw std::runtime_error("L11 Metal texture payload size mismatch");
  }
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:l11::kTextureWidth
                                                        height:l11::kTextureHeight
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, l11::kTextureWidth,
                                         l11::kTextureHeight)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:l11::kTextureWidth * 4u];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture) {
  std::vector<std::uint8_t> bytes(l11::kTextureBytes);
  [texture getBytes:bytes.data()
         bytesPerRow:l11::kTextureWidth * 4u
          fromRegion:MTLRegionMake2D(0, 0, l11::kTextureWidth,
                                     l11::kTextureHeight)
         mipmapLevel:0];
  return bytes;
}

std::size_t OutsideMismatches(const std::vector<std::uint8_t>& output,
                              std::uint8_t poison) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kTextureHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kTextureWidth; ++x) {
      if (l11::InOutputRegion(x, y)) continue;
      const std::size_t offset =
          (std::size_t{y} * l11::kTextureWidth + x) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        count += output[offset + channel] != poison;
      }
    }
  }
  return count;
}

std::size_t RegionMismatches(const std::vector<std::uint8_t>& input,
                             const std::vector<std::uint8_t>& output) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kCopyWidth; ++x) {
      const std::size_t source =
          (std::size_t{y + l11::kInputOriginY} * l11::kTextureWidth +
           x + l11::kInputOriginX) * 4u;
      const std::size_t destination =
          (std::size_t{y + l11::kOutputOriginY} * l11::kTextureWidth +
           x + l11::kOutputOriginX) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        count += input[source + channel] != output[destination + channel];
      }
    }
  }
  return count;
}

std::size_t RegionPoisonPixels(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kCopyWidth; ++x) {
      const std::size_t offset =
          (std::size_t{y + l11::kOutputOriginY} * l11::kTextureWidth +
           x + l11::kOutputOriginX) * 4u;
      count += output[offset] == poison && output[offset + 1u] == poison &&
               output[offset + 2u] == poison &&
               output[offset + 3u] == poison;
    }
  }
  return count;
}

void WriteRunReceipt(const std::filesystem::path& path,
                     const std::string& run_id, std::uint8_t poison,
                     const std::string& device_name,
                     const std::vector<std::uint8_t>& input_before,
                     const std::vector<std::uint8_t>& input_after,
                     const std::vector<std::uint8_t>& output,
                     const std::vector<std::uint8_t>& region,
                     std::size_t input_mismatches,
                     std::size_t region_mismatches,
                     std::size_t outside_mismatches,
                     std::size_t expected_mismatches,
                     std::size_t region_poison_pixels) {
  const bool pass = input_mismatches == 0 && region_mismatches == 0 &&
                    outside_mismatches == 0 && expected_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L11 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l11-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"metal_function\": \"fg_early_l11\",\n"
         << "  \"selected_launch\": 11,\n"
         << "  \"source_module_token\": 4,\n"
         << "  \"source_module_bytes\": " << l11::kModuleBytes << ",\n"
         << "  \"source_module_sha256\": \"" << l11::kModuleSha256 << "\",\n"
         << "  \"selected_rtx_grid\": [40, 23, 1],\n"
         << "  \"selected_rtx_block\": [16, 16, 1],\n"
         << "  \"metal_grid\": [640, 368, 1],\n"
         << "  \"metal_threadgroup\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 40,\n"
         << "  \"frozen_selected_raw_parameter_sha256\": \""
         << l11::kFrozenRawParameterSha256 << "\",\n"
         << "  \"frozen_selected_address_free_parameter_sha256\": \""
         << l11::kFrozenAddressFreeParameterSha256 << "\",\n"
         << "  \"fixture_address_free_parameter_sha256\": \""
         << l11::kFixtureAddressFreeParameterSha256 << "\",\n"
         << "  \"resource_format\": \"RGBA8_UNORM\",\n"
         << "  \"resource_extent\": [640, 360],\n"
         << "  \"input_origin\": [17, 11],\n"
         << "  \"output_origin\": [29, 19],\n"
         << "  \"copy_extent\": [587, 317],\n"
         << "  \"input_generator\": \"" << l11::kInputGenerator << "\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"region_output_sha256\": \"" << Sha256(region) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"declared_region_bytes\": " << region.size() << ",\n"
         << "  \"outside_region_pixels\": " << l11::kOutsidePixels << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"declared_region_mismatched_bytes\": " << region_mismatches
         << ",\n"
         << "  \"outside_region_mismatched_bytes\": " << outside_mismatches
         << ",\n"
         << "  \"canary_mismatched_bytes\": " << outside_mismatches << ",\n"
         << "  \"post_execution_expected_mismatched_bytes\": "
         << expected_mismatches << ",\n"
         << "  \"declared_region_poison_pixels\": " << region_poison_pixels
         << ",\n"
         << "  \"input_immutable\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"expected_bytes_fed_to_execution\": false,\n"
         << "  \"archived_intermediates_consumed\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-early-l11-mac <metallib> <output-dir> <run-id> "
                   "<poison-hex>\n";
      return 2;
    }
    try {
      char* poison_end = nullptr;
      const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
      if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const std::filesystem::path output_root = argv[2];
      const std::string run_id = argv[3];
      std::filesystem::create_directories(output_root);
      const auto input_before = l11::GenerateInput();
      const std::vector<std::uint8_t> output_before(l11::kTextureBytes, poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function = [library newFunctionWithName:@"fg_early_l11"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue failed");
      id<MTLTexture> input = MakeTexture(device, MTLTextureUsageShaderRead,
                                         input_before);
      id<MTLTexture> output = MakeTexture(device, MTLTextureUsageShaderWrite,
                                          output_before);
      const EarlyL11Parameters parameters{
          {l11::kInputOriginX, l11::kInputOriginY},
          {l11::kOutputOriginX, l11::kOutputOriginY},
          {l11::kCopyWidth, l11::kCopyHeight},
      };
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:input atIndex:0];
      [encoder setTexture:output atIndex:1];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [encoder dispatchThreadgroups:MTLSizeMake(40, 23, 1)
             threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }

      const auto input_after = ReadTexture(input);
      const auto output_after = ReadTexture(output);
      const auto expected = l11::ExpectedOutput(input_before, poison);
      const auto region = l11::ExtractRegion(output_after);
      const std::size_t input_mismatches =
          Mismatches(input_before, input_after);
      const std::size_t region_mismatches =
          RegionMismatches(input_before, output_after);
      const std::size_t outside_mismatches =
          OutsideMismatches(output_after, poison);
      const std::size_t expected_mismatches =
          Mismatches(expected, output_after);
      const std::size_t region_poison_pixels =
          RegionPoisonPixels(output_after, poison);

      Write(output_root / "input-after.rgba8", input_after);
      Write(output_root / "output.rgba8", output_after);
      Write(output_root / "region-output.rgba8", region);
      WriteRunReceipt(output_root / "run-receipt.json", run_id, poison,
                      device.name.UTF8String, input_before, input_after,
                      output_after, region, input_mismatches, region_mismatches,
                      outside_mismatches, expected_mismatches,
                      region_poison_pixels);

      const bool pass = input_mismatches == 0 && region_mismatches == 0 &&
                        outside_mismatches == 0 && expected_mismatches == 0;
      std::cout << "probe=fg_early_l11_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "region_output_sha256=" << Sha256(region) << '\n'
                << "output_sha256=" << Sha256(output_after) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "declared_region_mismatched_bytes=" << region_mismatches
                << '\n'
                << "outside_region_mismatched_bytes=" << outside_mismatches
                << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
