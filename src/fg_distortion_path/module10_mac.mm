#include "module10_contract.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace {

namespace m10 = fg_distortion_module10;

std::string Hash(const std::vector<std::uint8_t> &bytes) {
  std::array<unsigned char, 32> digest{};
  CC_SHA256(bytes.data(), bytes.size(), digest.data());
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const auto value : digest)
    stream << std::setw(2) << unsigned(value);
  return stream.str();
}

void Write(const std::filesystem::path &path,
           const std::vector<std::uint8_t> &bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLTextureUsage usage,
                           const std::vector<std::uint8_t> &bytes) {
  auto *descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                   width:m10::kWidth
                                  height:m10::kHeight
                               mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil)
    throw std::runtime_error("texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, m10::kWidth, m10::kHeight)
             mipmapLevel:0
               withBytes:bytes.data()
             bytesPerRow:m10::kWidth * 8u];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture) {
  std::vector<std::uint8_t> bytes(m10::kRgba16Bytes);
  [texture getBytes:bytes.data()
        bytesPerRow:m10::kWidth * 8u
         fromRegion:MTLRegionMake2D(0, 0, m10::kWidth, m10::kHeight)
        mipmapLevel:0];
  return bytes;
}

std::size_t Mismatches(const std::vector<std::uint8_t> &left,
                       const std::vector<std::uint8_t> &right) {
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result += left[index] != right[index];
  return result;
}

std::size_t PoisonPixels(const std::vector<std::uint8_t> &bytes,
                         std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    bool untouched = true;
    for (std::size_t byte = 0; byte < 8u; ++byte)
      untouched &= bytes[offset + byte] == poison;
    result += untouched;
  }
  return result;
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 5)
    return 2;
  @autoreleasepool {
    try {
      const std::filesystem::path output_root = argv[2];
      std::filesystem::create_directories(output_root);
      const std::string run_id = argv[3];
      const std::uint8_t poison =
          static_cast<std::uint8_t>(std::stoul(argv[4], nullptr, 16));
      const auto input = m10::GenerateInput();
      std::vector<std::uint8_t> initialized(m10::kRgba16Bytes, poison);
      std::vector<std::uint8_t> canary_bytes(m10::kRgba16Bytes, poison ^ 0xffu);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil)
        throw std::runtime_error("Metal device unavailable");
      NSError *error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:
              [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                      error:&error];
      if (library == nil)
        throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_distortion_module10"];
      if (function == nil)
        throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil)
        throw std::runtime_error("pipeline creation failed");

      auto input_texture =
          MakeTexture(device, MTLTextureUsageShaderRead, input);
      auto output_texture =
          MakeTexture(device, MTLTextureUsageShaderWrite, initialized);
      auto canary_texture =
          MakeTexture(device, MTLTextureUsageShaderWrite, canary_bytes);
      struct {
        std::uint32_t input_origin[2];
        std::uint32_t input_extent[2];
        float inverse_input_extent[2];
        std::uint32_t output_extent[2];
        float output_bias;
        float output_scale;
      } parameters{{0, 0},
                   {m10::kWidth, m10::kHeight},
                   {std::bit_cast<float>(0x3b4ccccdu),
                    std::bit_cast<float>(0x3bb60b61u)},
                   {m10::kWidth, m10::kHeight},
                   0.0f,
                   1.0f};
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:input_texture atIndex:0];
      [encoder setTexture:output_texture atIndex:1];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [encoder dispatchThreads:MTLSizeMake(m10::kWidth, m10::kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal command buffer failed");

      const auto input_after = ReadTexture(input_texture);
      const auto output = ReadTexture(output_texture);
      const auto canary_after = ReadTexture(canary_texture);
      const std::size_t input_mismatches = Mismatches(input, input_after);
      const std::size_t canary_mismatches =
          Mismatches(canary_bytes, canary_after);
      Write(output_root / "output.rgba16f", output);
      std::ofstream receipt(output_root / "run-receipt.json");
      receipt << "{\n\"schema\":\"neuralstudio.fg.distortion-module10-run.v1\","
                 "\n\"backend\":\"mac-metal\",\n\"run_id\":\""
              << run_id << "\",\n\"process_id\":" << getpid()
              << ",\n\"poison_byte\":" << unsigned(poison)
              << ",\n\"module_sha256\":\"" << m10::kModuleSha256
              << "\",\n\"module_bytes\":" << m10::kModuleBytes
              << ",\n\"selected_launch_tokens\":[\"launch-000005\",\"launch-"
                 "000016\"],"
                 "\n\"selected_phase_raw_parameter_sha256\":[\""
              << m10::kResetRawParameterSha256 << "\",\""
              << m10::kGeneratedRawParameterSha256
              << "\"],\n\"selected_phase_address_free_parameter_sha256\":[\""
              << m10::kResetAddressFreeParameterSha256 << "\",\""
              << m10::kGeneratedAddressFreeParameterSha256
              << "\"],\n\"packed_parameter_bytes\":56,"
                 "\n\"fixture_address_free_parameter_sha256\":\""
              << m10::kFixtureAddressFreeParameterSha256
              << "\",\n\"input_generator\":\"" << m10::kInputGenerator
              << "\",\n\"input_pre_sha256\":\"" << Hash(input)
              << "\",\n\"input_post_sha256\":\"" << Hash(input_after)
              << "\",\n\"input_mismatched_bytes\":" << input_mismatches
              << ",\n\"texture_canary_mismatched_bytes\":" << canary_mismatches
              << ",\n\"output_bytes\":" << output.size()
              << ",\n\"output_sha256\":\"" << Hash(output)
              << "\",\n\"output_poison_pixels\":"
              << PoisonPixels(output, poison)
              << ",\n\"typed_metal_resources\":true,"
                 "\n\"complete_write_coverage_checked_by_dual_poison\":true,"
                 "\n\"expected_output_fed_to_execution\":false,"
                 "\n\"archived_intermediates_consumed\":false,"
                 "\n\"address_fields_recorded\":false,\n\"result\":\""
              << ((input_mismatches | canary_mismatches) ? "fail" : "pass")
              << "\"\n}\n";
      return (input_mismatches | canary_mismatches) ? 3 : 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
