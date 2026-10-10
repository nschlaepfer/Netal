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

#include "compositor_l87_contract.h"

namespace fs = std::filesystem;
using namespace fg_compositor_l87;

namespace {

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t RgbHalf(std::uint64_t pixel, std::uint32_t channel) {
  const auto bits = SplitMix64(0x6a09e667f3bcc909ull ^
                               (pixel * 0x9e3779b97f4a7c15ull) ^
                               (std::uint64_t{channel + 1u} *
                                0xbf58476d1ce4e5b9ull));
  const std::uint16_t sign = (bits >> 63u) != 0u ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((bits >> 17u) & 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (bits & 0x03ffu));
}

std::uint16_t AlphaHalf(std::uint64_t pixel) {
  const auto bits = SplitMix64(0xbb67ae8584caa73bull ^
                               (pixel * 0x94d049bb133111ebull));
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((bits >> 23u) & 3u));
  return static_cast<std::uint16_t>((exponent << 10u) |
                                    (bits & 0x03ffu));
}

std::vector<std::uint8_t> GenerateHalf4(std::size_t bytes,
                                        std::uint64_t salt) {
  if ((bytes % 8u) != 0u) throw std::runtime_error("half4 byte drift");
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t pixel = 0; pixel < bytes / 8u; ++pixel) {
    const std::uint64_t identity = pixel ^ salt;
    Store16(&result, pixel * 8u, RgbHalf(identity, 0));
    Store16(&result, pixel * 8u + 2u, RgbHalf(identity, 1));
    Store16(&result, pixel * 8u + 4u, RgbHalf(identity, 2));
    Store16(&result, pixel * 8u + 6u, AlphaHalf(identity));
  }
  return result;
}

std::vector<std::uint8_t> GenerateMask(bool sparse) {
  std::vector<std::uint8_t> mask(kMaskBytes);
  for (std::size_t i = 0; i < mask.size(); ++i) {
    const auto value = SplitMix64(0x3c6ef372fe94f82bull ^ i);
    mask[i] = sparse && ((i % 17u) == 0u || (i % 101u) == 7u)
                  ? 0u
                  : static_cast<std::uint8_t>(1u + (value % 255u));
  }
  return mask;
}

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

id<MTLBuffer> Guarded(id<MTLDevice> device,
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

std::vector<std::uint8_t> ReadAll(id<MTLBuffer> buffer) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  return {bytes, bytes + buffer.length};
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

std::size_t PoisonPixels(const std::vector<std::uint8_t>& output,
                         std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < output.size(); i += 4u)
    count += output[i] == poison && output[i + 1u] == poison &&
             output[i + 2u] == poison && output[i + 3u] == poison;
  return count;
}

std::size_t ZeroMaskPixels(const std::vector<std::uint8_t>& mask) {
  std::size_t count = 0;
  for (const auto value : mask) count += value == 0;
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

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format,
                       MTLTextureUsage usage,
                       const std::vector<std::uint8_t>& initial,
                       NSUInteger bytes_per_pixel) {
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:kWidth
                                                        height:kHeight
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("Metal texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
             mipmapLevel:0
               withBytes:initial.data()
             bytesPerRow:kWidth * bytes_per_pixel];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      NSUInteger bytes_per_pixel) {
  std::vector<std::uint8_t> bytes(kWidth * kHeight * bytes_per_pixel);
  [texture getBytes:bytes.data()
         bytesPerRow:kWidth * bytes_per_pixel
          fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
         mipmapLevel:0];
  return bytes;
}

void Dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
              id<MTLBuffer> level0, id<MTLBuffer> second,
              id<MTLBuffer> parameters, id<MTLTexture> mask,
              id<MTLTexture> output, bool exact) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:level0 offset:kGuardBytes atIndex:0];
  [encoder setBuffer:second offset:(exact ? kGuardBytes : 0) atIndex:1];
  if (exact) {
    [encoder setBuffer:parameters offset:0 atIndex:2];
  }
  [encoder setTexture:mask atIndex:0];
  [encoder setTexture:output atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, 1)
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

void DispatchPacked(id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> level0, id<MTLBuffer> rom,
                    id<MTLBuffer> parameters, id<MTLBuffer> output,
                    id<MTLTexture> mask) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:level0 offset:kGuardBytes atIndex:0];
  [encoder setBuffer:rom offset:kGuardBytes atIndex:1];
  [encoder setBuffer:parameters offset:0 atIndex:2];
  [encoder setBuffer:output offset:kGuardBytes atIndex:3];
  [encoder setTexture:mask atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, 1)
      threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error("packed candidate Metal command failed");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: fg-compositor-l87-mac <metallib> <rcp-rom> <output-dir> <run-id> <poison-hex> <full|sparse>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const fs::path metallib_path = argv[1];
      const fs::path rom_path = argv[2];
      const fs::path output_directory = argv[3];
      const std::string run_id = argv[4];
      const std::uint8_t poison = ParsePoison(argv[5]);
      const std::string mask_mode = argv[6];
      if (mask_mode != "full" && mask_mode != "sparse")
        throw std::runtime_error("unsupported mask mode");
      const bool sparse = mask_mode == "sparse";
      fs::create_directories(output_directory);

      std::array<std::vector<std::uint8_t>, 4> pyramid;
      for (std::size_t i = 0; i < pyramid.size(); ++i)
        pyramid[i] = GenerateHalf4(kPyramidBytes[i], 0x100000001b3ull * (i + 1));
      const auto mask = GenerateMask(sparse);
      const std::vector<std::uint8_t> initial_output(kOutputBytes, poison);
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
      auto exact_pipeline = Pipeline(device, library, @"fg_compositor_l87_exact");
      auto strict_pipeline =
          Pipeline(device, library, @"fg_compositor_l87_strict_division");
      auto diagnostic_pipeline =
          Pipeline(device, library, @"fg_compositor_l87_float_diagnostic");
      auto packed_pipeline =
          Pipeline(device, library, @"fg_compositor_l87_packed_candidate");

      std::array<id<MTLBuffer>, 4> buffers;
      for (std::size_t i = 0; i < buffers.size(); ++i)
        buffers[i] = Guarded(device, pyramid[i]);
      id<MTLBuffer> rom_buffer = Guarded(device, rom);
      const MetalParameters values{kSelectedLevel, kWidth, kHeight, 0};
      id<MTLBuffer> parameter_buffer =
          [device newBufferWithBytes:&values
                              length:sizeof(values)
                             options:MTLResourceStorageModeShared];
      if (parameter_buffer == nil) throw std::runtime_error("parameter allocation failed");
      id<MTLTexture> mask_texture =
          Texture(device, MTLPixelFormatR8Unorm, MTLTextureUsageShaderRead,
                  mask, 1);
      id<MTLTexture> output_texture = Texture(
          device, MTLPixelFormatRGBA8Unorm,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          initial_output, 4);
      id<MTLTexture> strict_texture = Texture(
          device, MTLPixelFormatRGBA8Unorm,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          initial_output, 4);
      const std::vector<std::uint8_t> initial_float_output(
          std::size_t{kWidth} * kHeight * 16u, poison);
      id<MTLTexture> diagnostic_texture = Texture(
          device, MTLPixelFormatRGBA32Float,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          initial_float_output, 16);
      id<MTLBuffer> packed_output_buffer = Guarded(device, initial_output);

      Dispatch(queue, exact_pipeline, buffers[0], rom_buffer, parameter_buffer,
               mask_texture, output_texture, true);
      Dispatch(queue, strict_pipeline, buffers[0], parameter_buffer,
               parameter_buffer, mask_texture, strict_texture, false);
      Dispatch(queue, diagnostic_pipeline, buffers[0], rom_buffer,
               parameter_buffer, mask_texture, diagnostic_texture, true);
      DispatchPacked(queue, packed_pipeline, buffers[0], rom_buffer,
                     parameter_buffer, packed_output_buffer, mask_texture);

      const auto actual_mask = ReadTexture(mask_texture, 1);
      const auto output = ReadTexture(output_texture, 4);
      const auto strict_output = ReadTexture(strict_texture, 4);
      const auto diagnostic_output = ReadTexture(diagnostic_texture, 16);
      const auto packed_guarded = ReadAll(packed_output_buffer);
      const auto packed_output = Payload(packed_guarded, kOutputBytes);
      std::size_t input_mismatches = Mismatches(mask, actual_mask);
      std::size_t canary_mismatches = 0;
      std::array<std::vector<std::uint8_t>, 4> actual_pyramid;
      for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto guarded = ReadAll(buffers[i]);
        canary_mismatches += CanaryMismatches(guarded, pyramid[i].size());
        actual_pyramid[i] = Payload(guarded, pyramid[i].size());
        input_mismatches += Mismatches(pyramid[i], actual_pyramid[i]);
      }
      const auto guarded_rom = ReadAll(rom_buffer);
      canary_mismatches += CanaryMismatches(guarded_rom, rom.size());
      const auto actual_rom = Payload(guarded_rom, rom.size());
      const std::size_t rom_mismatches = Mismatches(rom, actual_rom);
      const std::size_t zero_mask_pixels = ZeroMaskPixels(mask);
      const std::size_t poison_pixels = PoisonPixels(output, poison);
      const std::size_t strict_poison_pixels = PoisonPixels(strict_output, poison);
      const std::size_t strict_mismatched_bytes = Mismatches(output, strict_output);
      canary_mismatches += CanaryMismatches(packed_guarded, kOutputBytes);
      const bool passed = input_mismatches == 0 && rom_mismatches == 0 &&
                          canary_mismatches == 0 &&
                          poison_pixels == zero_mask_pixels &&
                          strict_poison_pixels == zero_mask_pixels;

      Write(output_directory / "input-level0-after.rgba16f", actual_pyramid[0]);
      Write(output_directory / "input-mask-after.r8", actual_mask);
      Write(output_directory / "output.rgba8", output);
      Write(output_directory / "strict-division-output.rgba8", strict_output);
      Write(output_directory / "diagnostic-output.rgba32f", diagnostic_output);
      Write(output_directory / "packed-candidate-output.rgba8", packed_output);
      std::ofstream receipt(output_directory / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      if (!receipt) throw std::runtime_error("unable to write receipt");
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.compositor-l87-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"mask_mode\": \"" << mask_mode << "\",\n"
              << "  \"source_launch_token\": 87,\n"
              << "  \"source_module_token\": 20,\n"
              << "  \"source_module_bytes\": " << kModuleBytes << ",\n"
              << "  \"source_module_sha256\": \"" << kModuleSha256 << "\",\n"
              << "  \"raw_parameter_sha256\": \"" << kRawParameterSha256
              << "\",\n"
              << "  \"address_free_parameter_sha256\": \""
              << kAddressFreeParameterSha256 << "\",\n"
              << "  \"metal_function\": \"fg_compositor_l87_exact\",\n"
              << "  \"negative_control_function\": \"fg_compositor_l87_strict_division\",\n"
              << "  \"input_generator\": \"" << kGenerator << "\",\n"
              << "  \"numerical_domain\": \"finite-normal-half4-positive-normal-alpha\",\n"
              << "  \"architecture_rom\": {\"kind\": \"sm120-mufu-rcp-2bit-correction\", \"bytes\": "
              << rom.size() << ", \"sha256\": \"" << kRcpRomSha256
              << "\", \"model_values\": 0, \"captured_intermediates\": 0},\n"
              << "  \"level0_bytes\": " << actual_pyramid[0].size() << ",\n"
              << "  \"level0_sha256\": \"" << Sha256(actual_pyramid[0])
              << "\",\n"
              << "  \"mask_bytes\": " << actual_mask.size() << ",\n"
              << "  \"mask_sha256\": \"" << Sha256(actual_mask) << "\",\n"
              << "  \"output_bytes\": " << output.size() << ",\n"
              << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
              << "  \"strict_division_output_sha256\": \""
              << Sha256(strict_output) << "\",\n"
              << "  \"strict_division_mismatched_bytes\": "
              << strict_mismatched_bytes << ",\n"
              << "  \"diagnostic_float_output_bytes\": "
              << diagnostic_output.size() << ",\n"
              << "  \"diagnostic_float_output_sha256\": \""
              << Sha256(diagnostic_output) << "\",\n"
              << "  \"packed_candidate_output_sha256\": \""
              << Sha256(packed_output) << "\",\n"
              << "  \"zero_mask_pixels\": " << zero_mask_pixels << ",\n"
              << "  \"poison_pixels_after\": " << poison_pixels << ",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"architecture_rom_mismatched_bytes\": "
              << rom_mismatches << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n"
              << "  \"external_expected_tensors_consumed\": false,\n"
              << "  \"archived_intermediates_consumed\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (passed ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "probe=fg_compositor_l87_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_sha256=" << Sha256(output) << '\n'
                << "strict_division_mismatched_bytes="
                << strict_mismatched_bytes << '\n'
                << "zero_mask_pixels=" << zero_mask_pixels << '\n'
                << "poison_pixels_after=" << poison_pixels << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
