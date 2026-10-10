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

#include "fg_mid_l55_contract.h"
#include "fg_mid_l55_inputs.h"

namespace fs = std::filesystem;
using namespace fg_mid_l55;

namespace {

std::vector<std::uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void Write(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
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

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

id<MTLBuffer> GuardedBuffer(id<MTLDevice> device,
                            const std::vector<std::uint8_t>& payload) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:payload.size() + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
  auto* contents = static_cast<std::uint8_t*>(buffer.contents);
  std::memset(contents, kGuardValue, buffer.length);
  std::memcpy(contents + kGuardBytes, payload.data(), payload.size());
  return buffer;
}

id<MTLBuffer> PoisonBuffer(id<MTLDevice> device, std::size_t payload_bytes,
                           std::uint8_t poison) {
  std::vector<std::uint8_t> payload(payload_bytes, poison);
  return GuardedBuffer(device, payload);
}

std::vector<std::uint8_t> ReadAll(id<MTLBuffer> buffer) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  return {bytes, bytes + buffer.length};
}

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t payload_bytes) {
  return {guarded.begin() + kGuardBytes,
          guarded.begin() + kGuardBytes + payload_bytes};
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t index = 0; index < a.size(); ++index)
    count += a[index] != b[index];
  return count;
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t payload_bytes) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    count += guarded[index] != kGuardValue;
    count += guarded[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return count;
}

std::size_t PoisonPatternWords(const std::vector<std::uint8_t>& bytes,
                               std::uint8_t poison,
                               std::size_t word_bytes) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += word_bytes) {
    bool matched = true;
    for (std::size_t lane = 0; lane < word_bytes; ++lane)
      matched = matched && bytes[offset + lane] == poison;
    count += matched;
  }
  return count;
}

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format,
                       NSUInteger width, NSUInteger height,
                       MTLTextureUsage usage,
                       const std::vector<std::uint8_t>& initial,
                       NSUInteger bytes_per_pixel) {
  if (initial.size() != width * height * bytes_per_pixel)
    throw std::runtime_error("Metal texture input size drift");
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
               withBytes:initial.data()
             bytesPerRow:width * bytes_per_pixel];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      NSUInteger bytes_per_pixel) {
  std::vector<std::uint8_t> bytes(texture.width * texture.height *
                                  bytes_per_pixel);
  [texture getBytes:bytes.data()
         bytesPerRow:texture.width * bytes_per_pixel
          fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
         mipmapLevel:0];
  return bytes;
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
              const std::array<id<MTLTexture>, 7>& textures) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  for (NSUInteger index = 0; index < buffers.size(); ++index)
    [encoder setBuffer:buffers[index] offset:offsets[index] atIndex:index];
  for (NSUInteger index = 0; index < textures.size(); ++index)
    [encoder setTexture:textures[index] atIndex:index];
  [encoder dispatchThreads:MTLSizeMake(kPhysicalWidth, kPhysicalHeight, 1)
      threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
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
    std::cerr << "usage: fg-mid-l55-mac <metallib> <rcp-rom> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const fs::path metallib_path = argv[1];
      const fs::path rom_path = argv[2];
      const fs::path output_directory = argv[3];
      const std::string run_id = argv[4];
      const std::uint8_t poison = ParsePoison(argv[5]);
      fs::create_directories(output_directory);
      const GeneratedInputs inputs;
      const auto rom = Read(rom_path);
      if (rom.size() != kRcpRomBytes || Sha256(rom) != kRcpRomSha256)
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
      auto pipeline = Pipeline(device, library, @"fg_mid_l55_exact");
      id<MTLBuffer> candidate_a = GuardedBuffer(device, inputs.candidate_a);
      id<MTLBuffer> candidate_b = GuardedBuffer(device, inputs.candidate_b);
      id<MTLBuffer> linear_a = PoisonBuffer(device, kLinearRgbaBytes, poison);
      id<MTLBuffer> linear_b = PoisonBuffer(device, kLinearRgbaBytes, poison);
      id<MTLBuffer> linear_mask = PoisonBuffer(device, kLinearMaskBytes, poison);
      id<MTLBuffer> rom_buffer = GuardedBuffer(device, rom);
      const MetalParameters parameter_values{
          kPhysicalWidth, kPhysicalHeight, kAccumulationMode,
          kCandidateWidth, kCandidateHeight, kCandidateWidth,
          kCandidateHeight, kActiveWidth, kActiveHeight, kOptionalUiEnabled};
      id<MTLBuffer> parameters =
          [device newBufferWithBytes:&parameter_values
                              length:sizeof(parameter_values)
                             options:MTLResourceStorageModeShared];
      if (parameters == nil) throw std::runtime_error("parameter allocation failed");
      id<MTLTexture> rgba_a = Texture(
          device, MTLPixelFormatRGBA16Float, kActiveWidth, kActiveHeight,
          MTLTextureUsageShaderRead, inputs.rgba_a, 8);
      id<MTLTexture> rgba_b = Texture(
          device, MTLPixelFormatRGBA16Float, kActiveWidth, kActiveHeight,
          MTLTextureUsageShaderRead, inputs.rgba_b, 8);
      id<MTLTexture> motion = Texture(
          device, MTLPixelFormatRG16Float, kCandidateWidth, kCandidateHeight,
          MTLTextureUsageShaderRead, inputs.motion, 4);
      id<MTLTexture> blend = Texture(
          device, MTLPixelFormatRG16Float, kCandidateWidth, kCandidateHeight,
          MTLTextureUsageShaderRead, inputs.blend, 4);
      const std::vector<std::uint8_t> poison_surface_rgba(kSurfaceRgbaBytes,
                                                          poison);
      const std::vector<std::uint8_t> poison_surface_mask(kSurfaceMaskBytes,
                                                          poison);
      id<MTLTexture> surface_a = Texture(
          device, MTLPixelFormatRGBA16Float, kActiveWidth, kActiveHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          poison_surface_rgba, 8);
      id<MTLTexture> surface_b = Texture(
          device, MTLPixelFormatRGBA16Float, kActiveWidth, kActiveHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          poison_surface_rgba, 8);
      id<MTLTexture> surface_mask = Texture(
          device, MTLPixelFormatR8Unorm, kActiveWidth, kActiveHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          poison_surface_mask, 1);
      const std::array<id<MTLBuffer>, 7> buffers = {
          candidate_a, candidate_b, linear_a, linear_b, linear_mask,
          rom_buffer, parameters};
      const std::array<NSUInteger, 7> offsets = {
          kGuardBytes, kGuardBytes, kGuardBytes, kGuardBytes,
          kGuardBytes, kGuardBytes, 0};
      const std::array<id<MTLTexture>, 7> textures = {
          rgba_a, rgba_b, motion, blend, surface_a, surface_b, surface_mask};
      Dispatch(queue, pipeline, buffers, offsets, textures);

      const auto actual_rgba_a = ReadTexture(rgba_a, 8);
      const auto actual_rgba_b = ReadTexture(rgba_b, 8);
      const auto actual_motion = ReadTexture(motion, 4);
      const auto actual_blend = ReadTexture(blend, 4);
      const auto actual_surface_a = ReadTexture(surface_a, 8);
      const auto actual_surface_b = ReadTexture(surface_b, 8);
      const auto actual_surface_mask = ReadTexture(surface_mask, 1);
      const auto candidate_a_after = ReadAll(candidate_a);
      const auto candidate_b_after = ReadAll(candidate_b);
      const auto linear_a_after = ReadAll(linear_a);
      const auto linear_b_after = ReadAll(linear_b);
      const auto linear_mask_after = ReadAll(linear_mask);
      const auto rom_after = ReadAll(rom_buffer);
      const auto actual_linear_a = Payload(linear_a_after, kLinearRgbaBytes);
      const auto actual_linear_b = Payload(linear_b_after, kLinearRgbaBytes);
      const auto actual_linear_mask = Payload(linear_mask_after, kLinearMaskBytes);
      std::size_t input_mismatches =
          Mismatches(inputs.rgba_a, actual_rgba_a) +
          Mismatches(inputs.rgba_b, actual_rgba_b) +
          Mismatches(inputs.motion, actual_motion) +
          Mismatches(inputs.blend, actual_blend) +
          Mismatches(inputs.candidate_a,
                     Payload(candidate_a_after, kCandidateBufferBytes)) +
          Mismatches(inputs.candidate_b,
                     Payload(candidate_b_after, kCandidateBufferBytes)) +
          Mismatches(rom, Payload(rom_after, kRcpRomBytes));
      const std::size_t canary_mismatches =
          CanaryMismatches(candidate_a_after, kCandidateBufferBytes) +
          CanaryMismatches(candidate_b_after, kCandidateBufferBytes) +
          CanaryMismatches(linear_a_after, kLinearRgbaBytes) +
          CanaryMismatches(linear_b_after, kLinearRgbaBytes) +
          CanaryMismatches(linear_mask_after, kLinearMaskBytes) +
          CanaryMismatches(rom_after, kRcpRomBytes);
      const std::size_t poison_pattern_words =
          PoisonPatternWords(actual_surface_a, poison, 2) +
          PoisonPatternWords(actual_surface_b, poison, 2) +
          PoisonPatternWords(actual_surface_mask, poison, 1) +
          PoisonPatternWords(actual_linear_a, poison, 2) +
          PoisonPatternWords(actual_linear_b, poison, 2) +
          PoisonPatternWords(actual_linear_mask, poison, 2);
      std::vector<std::uint8_t> combined;
      for (const auto* value : {&actual_surface_a, &actual_surface_b,
                                &actual_surface_mask, &actual_linear_a,
                                &actual_linear_b, &actual_linear_mask})
        combined.insert(combined.end(), value->begin(), value->end());
      const bool passed = input_mismatches == 0 && canary_mismatches == 0;
      Write(output_directory / "surface-a.rgba16f", actual_surface_a);
      Write(output_directory / "surface-b.rgba16f", actual_surface_b);
      Write(output_directory / "surface-mask.r8", actual_surface_mask);
      Write(output_directory / "linear-a.rgba16f", actual_linear_a);
      Write(output_directory / "linear-b.rgba16f", actual_linear_b);
      Write(output_directory / "linear-mask.r16f", actual_linear_mask);
      Write(output_directory / "input-rgba-a.rgba16f", actual_rgba_a);
      Write(output_directory / "input-rgba-b.rgba16f", actual_rgba_b);
      Write(output_directory / "input-motion.rg16f", actual_motion);
      Write(output_directory / "input-blend.rg16f", actual_blend);
      std::ofstream receipt(output_directory / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      if (!receipt) throw std::runtime_error("unable to write receipt");
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.mid-l55-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"source_launch_token\": 55,\n"
              << "  \"source_module_token\": 3,\n"
              << "  \"source_module_bytes\": " << kModuleBytes << ",\n"
              << "  \"source_module_sha256\": \"" << kModuleSha256 << "\",\n"
              << "  \"raw_parameter_sha256\": \"" << kRawParameterSha256 << "\",\n"
              << "  \"address_free_parameter_sha256\": \""
              << kAddressFreeParameterSha256 << "\",\n"
              << "  \"metal_function\": \"fg_mid_l55_exact\",\n"
              << "  \"input_generator\": \"" << kGenerator << "\",\n"
              << "  \"architecture_rom\": {\"bytes\": " << rom.size()
              << ", \"sha256\": \"" << kRcpRomSha256
              << "\", \"model_values\": 0, \"captured_intermediates\": 0},\n"
              << "  \"input_hashes\": {\"rgba_a\": \"" << Sha256(actual_rgba_a)
              << "\", \"rgba_b\": \"" << Sha256(actual_rgba_b)
              << "\", \"motion\": \"" << Sha256(actual_motion)
              << "\", \"blend\": \"" << Sha256(actual_blend)
              << "\", \"candidate_a\": \"" << Sha256(inputs.candidate_a)
              << "\", \"candidate_b\": \"" << Sha256(inputs.candidate_b)
              << "\"},\n"
              << "  \"output_hashes\": {\"surface_a\": \"" << Sha256(actual_surface_a)
              << "\", \"surface_b\": \"" << Sha256(actual_surface_b)
              << "\", \"surface_mask\": \"" << Sha256(actual_surface_mask)
              << "\", \"linear_a\": \"" << Sha256(actual_linear_a)
              << "\", \"linear_b\": \"" << Sha256(actual_linear_b)
              << "\", \"linear_mask\": \"" << Sha256(actual_linear_mask)
              << "\"},\n"
              << "  \"combined_output_bytes\": " << combined.size() << ",\n"
              << "  \"combined_output_sha256\": \"" << Sha256(combined) << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
              << "  \"poison_pattern_output_words\": "
              << poison_pattern_words << ",\n"
              << "  \"external_expected_tensors_consumed\": false,\n"
              << "  \"archived_intermediates_consumed\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n}\n";
      std::cout << "probe=fg_mid_l55_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "combined_output_sha256=" << Sha256(combined) << '\n'
                << "poison_pattern_output_words=" << poison_pattern_words << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
