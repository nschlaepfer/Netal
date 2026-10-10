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

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 360;
constexpr std::uint32_t kPaddedHeight = 368;
constexpr std::size_t kMotionBytes =
    static_cast<std::size_t>(kWidth) * kPaddedHeight * 16u;
constexpr std::size_t kColorBytes =
    static_cast<std::size_t>(kWidth) * kHeight * 8u;
constexpr std::size_t kMaskBytes =
    static_cast<std::size_t>(kWidth) * kHeight;
constexpr std::size_t kScratchBytes =
    static_cast<std::size_t>(kWidth) * kPaddedHeight * 16u;
constexpr std::array<std::size_t, 4> kPyramidBytes = {
    320u * 180u * 8u, 160u * 90u * 8u, 80u * 45u * 8u, 40u * 23u * 8u};
constexpr std::array<MTLSize, 4> kPyramidGrids = {
    MTLSize{320, 180, 1}, MTLSize{160, 90, 1}, MTLSize{80, 45, 1},
    MTLSize{40, 23, 1}};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kRcpRomBytes = 2097152;
constexpr std::size_t kSigmoidBytes = 262144;
constexpr char kRcpRomSha[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
constexpr char kSigmoidSha[] =
    "394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850";
constexpr char kModuleSha[] =
    "3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76";
constexpr char kRawParameterSha[] =
    "31cfc2016f76ad79fe92717d4bb3b38ceb6c8b3f14968f6b5f1e2f66503e548a";
constexpr char kGenerator[] =
    "splitmix64-fg-postprocess-l84-independent-textures-motion-v1";

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t padded_height;
  std::uint32_t divisor;
};
static_assert(sizeof(Parameters) == 16);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t PositiveUnitHalf(std::uint64_t key) {
  const auto mixed = SplitMix64(key);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((mixed >> 17u) % 4u));
  return static_cast<std::uint16_t>((exponent << 10u) | (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateColor(std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(kColorBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * kWidth + x;
      for (std::uint32_t channel = 0; channel < 3; ++channel) {
        Store16(&bytes, pixel * 8u + channel * 2u,
                PositiveUnitHalf(seed ^ pixel * 0xD6E8FEB86659FD93ull ^
                                 channel * 0xA0761D6478BD642Full));
      }
      Store16(&bytes, pixel * 8u + 6u, 0x3c00u);
    }
  }
  return bytes;
}

std::uint16_t SmallSignedHalf(std::uint64_t key) {
  const auto mixed = SplitMix64(key);
  const std::uint16_t sign = (mixed >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(3u + ((mixed >> 11u) % 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateMotion(std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(kMotionBytes);
  for (std::uint32_t y = 0; y < kPaddedHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * kWidth + x;
      for (std::uint32_t channel = 0; channel < 4; ++channel) {
        Store16(&bytes, pixel * 16u + channel * 2u,
                SmallSignedHalf(seed ^ pixel * 0xE7037ED1A0B428DBull ^
                                channel * 0x8EBC6AF09C88C6E3ull));
      }
      const auto mixed = SplitMix64(seed ^ pixel * 0x589965CC75374CC3ull);
      const std::uint16_t logit = static_cast<std::uint16_t>(
          ((mixed >> 63u) ? 0x8000u : 0u) | (14u << 10u) |
          (mixed & 0x03ffu));
      Store16(&bytes, pixel * 16u + 8u, logit);
      Store16(&bytes, pixel * 16u + 10u, SmallSignedHalf(mixed ^ 0x11u));
      Store16(&bytes, pixel * 16u + 12u, SmallSignedHalf(mixed ^ 0x22u));
      Store16(&bytes, pixel * 16u + 14u, SmallSignedHalf(mixed ^ 0x33u));
    }
  }
  return bytes;
}

std::vector<std::uint8_t> GenerateMask(const std::string& fixture) {
  std::vector<std::uint8_t> bytes(kMaskBytes, 0u);
  if (fixture == "open") return bytes;
  if (fixture != "islands") throw std::runtime_error("unsupported fixture");
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const bool island =
          ((x >= 104u && x < 224u && y >= 72u && y < 184u) ||
           (x >= 376u && x < 552u && y >= 196u && y < 320u));
      const bool holes = ((x * 13u + y * 7u) % 47u) == 0u;
      bytes[static_cast<std::size_t>(y) * kWidth + x] =
          island && !holes ? 0xffu : 0u;
    }
  }
  return bytes;
}

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

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>* payload,
                      std::size_t bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
  auto* contents = static_cast<std::uint8_t*>(buffer.contents);
  std::memset(contents, kGuardValue, buffer.length);
  if (payload) {
    if (payload->size() != bytes) throw std::runtime_error("payload size");
    std::memcpy(contents + kGuardBytes, payload->data(), bytes);
  } else {
    std::memset(contents + kGuardBytes, fill, bytes);
  }
  return buffer;
}

std::vector<std::uint8_t> BufferPayload(id<MTLBuffer> buffer,
                                        std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  return {pointer + kGuardBytes, pointer + kGuardBytes + bytes};
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

void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(std::string("Metal command failed: ") +
                             (command.error
                                  ? command.error.localizedDescription.UTF8String
                                  : "unknown"));
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison");
  return static_cast<std::uint8_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: fg-postprocess-l84-mac <metallib> <rcp-rom> <sigmoid-lut> <output-dir> <run-id> <fixture> <poison-hex>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const std::filesystem::path metallib_path = argv[1];
      const auto rcp_rom = Read(argv[2]);
      const auto sigmoid_lut = Read(argv[3]);
      const std::filesystem::path output_root = argv[4];
      const std::string run_id = argv[5];
      const std::string fixture = argv[6];
      const std::uint8_t poison = ParsePoison(argv[7]);
      std::filesystem::create_directories(output_root);
      if (rcp_rom.size() != kRcpRomBytes || Sha256(rcp_rom) != kRcpRomSha)
        throw std::runtime_error("RCP ROM identity mismatch");
      if (sigmoid_lut.size() != kSigmoidBytes ||
          Sha256(sigmoid_lut) != kSigmoidSha)
        throw std::runtime_error("sigmoid LUT identity mismatch");
      const std::uint64_t fixture_seed =
          fixture == "open" ? 0xA0761D6478BD642Full :
          fixture == "islands" ? 0xE7037ED1A0B428DBull : 0ull;
      if (!fixture_seed) throw std::runtime_error("unsupported fixture");
      const auto color_a = GenerateColor(fixture_seed ^ 0x11111111ull);
      const auto color_b = GenerateColor(fixture_seed ^ 0x22222222ull);
      const auto mask = GenerateMask(fixture);
      const auto motion = GenerateMotion(fixture_seed ^ 0x33333333ull);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
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
      auto pixel_pipeline =
          Pipeline(device, library, @"fg_postprocess_l84_pixels");
      const std::array<id<MTLComputePipelineState>, 4> pyramid_pipelines = {
          Pipeline(device, library, @"fg_postprocess_l84_level0"),
          Pipeline(device, library, @"fg_postprocess_l84_level1"),
          Pipeline(device, library, @"fg_postprocess_l84_level2"),
          Pipeline(device, library, @"fg_postprocess_l84_level3")};

      id<MTLBuffer> color_a_buffer = Guarded(device, &color_a, kColorBytes, 0);
      id<MTLBuffer> color_b_buffer = Guarded(device, &color_b, kColorBytes, 0);
      id<MTLBuffer> mask_buffer = Guarded(device, &mask, kMaskBytes, 0);
      id<MTLBuffer> motion_buffer = Guarded(device, &motion, kMotionBytes, 0);
      id<MTLBuffer> sigmoid_buffer =
          Guarded(device, &sigmoid_lut, kSigmoidBytes, 0);
      id<MTLBuffer> scratch_buffer =
          Guarded(device, nullptr, kScratchBytes, poison);
      id<MTLBuffer> rcp_buffer = Guarded(device, &rcp_rom, kRcpRomBytes, 0);
      std::array<id<MTLBuffer>, 4> pyramid_buffers;
      for (std::size_t i = 0; i < pyramid_buffers.size(); ++i)
        pyramid_buffers[i] =
            Guarded(device, nullptr, kPyramidBytes[i], poison);
      const Parameters parameters{kWidth, kHeight, kPaddedHeight, 1u};
      id<MTLBuffer> parameter_buffer =
          [device newBufferWithBytes:&parameters
                              length:sizeof(parameters)
                             options:MTLResourceStorageModeShared];

      MTLTextureDescriptor* color_descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      color_descriptor.storageMode = MTLStorageModeShared;
      color_descriptor.usage = MTLTextureUsageShaderWrite;
      id<MTLTexture> output_color_texture = [device newTextureWithDescriptor:color_descriptor];
      id<MTLTexture> native_color_texture = [device newTextureWithDescriptor:color_descriptor];
      MTLTextureDescriptor* mask_descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      mask_descriptor.storageMode = MTLStorageModeShared;
      mask_descriptor.usage = MTLTextureUsageShaderWrite;
      id<MTLTexture> output_mask_texture = [device newTextureWithDescriptor:mask_descriptor];
      if (parameter_buffer == nil || output_color_texture == nil ||
          native_color_texture == nil ||
          output_mask_texture == nil)
        throw std::runtime_error("Metal resource allocation failed");
      std::vector<std::uint8_t> color_poison(kWidth * kHeight * 4u, poison);
      std::vector<std::uint8_t> mask_poison(kMaskBytes, poison);
      [output_color_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                              mipmapLevel:0
                                withBytes:color_poison.data()
                              bytesPerRow:kWidth * 4u];
      [native_color_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                              mipmapLevel:0
                                withBytes:color_poison.data()
                              bytesPerRow:kWidth * 4u];
      [output_mask_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                             mipmapLevel:0
                               withBytes:mask_poison.data()
                             bytesPerRow:kWidth];

      id<MTLCommandBuffer> pixel_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> pixel_encoder =
          [pixel_command computeCommandEncoder];
      [pixel_encoder setComputePipelineState:pixel_pipeline];
      const std::array<id<MTLBuffer>, 7> pixel_buffers = {
          color_a_buffer, color_b_buffer, mask_buffer, motion_buffer,
          sigmoid_buffer, scratch_buffer, parameter_buffer};
      for (NSUInteger i = 0; i < pixel_buffers.size(); ++i)
        [pixel_encoder setBuffer:pixel_buffers[i]
                         offset:i == 6 ? 0 : kGuardBytes
                        atIndex:i];
      [pixel_encoder setTexture:output_color_texture atIndex:0];
      [pixel_encoder setTexture:output_mask_texture atIndex:1];
      [pixel_encoder setTexture:native_color_texture atIndex:2];
      [pixel_encoder dispatchThreads:MTLSizeMake(kWidth, kPaddedHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [pixel_encoder endEncoding];
      Complete(pixel_command);

      for (std::size_t level = 0; level < pyramid_buffers.size(); ++level) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pyramid_pipelines[level]];
        [encoder setBuffer:scratch_buffer offset:kGuardBytes atIndex:0];
        [encoder setBuffer:pyramid_buffers[level] offset:kGuardBytes atIndex:1];
        [encoder setBuffer:rcp_buffer offset:kGuardBytes atIndex:2];
        [encoder dispatchThreads:kPyramidGrids[level]
            threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        [encoder endEncoding];
        Complete(command);
      }

      const auto color_a_after = BufferPayload(color_a_buffer, kColorBytes);
      const auto color_b_after = BufferPayload(color_b_buffer, kColorBytes);
      const auto mask_after = BufferPayload(mask_buffer, kMaskBytes);
      const auto motion_after = BufferPayload(motion_buffer, kMotionBytes);
      const auto sigmoid_after = BufferPayload(sigmoid_buffer, kSigmoidBytes);
      const auto rcp_after = BufferPayload(rcp_buffer, kRcpRomBytes);
      const auto scratch_after = BufferPayload(scratch_buffer, kScratchBytes);
      std::array<std::vector<std::uint8_t>, 4> pyramid_after;
      std::size_t canary_mismatches =
          CanaryMismatches(color_a_buffer, kColorBytes) +
          CanaryMismatches(color_b_buffer, kColorBytes) +
          CanaryMismatches(mask_buffer, kMaskBytes) +
          CanaryMismatches(motion_buffer, kMotionBytes) +
          CanaryMismatches(sigmoid_buffer, kSigmoidBytes) +
          CanaryMismatches(rcp_buffer, kRcpRomBytes) +
          CanaryMismatches(scratch_buffer, kScratchBytes);
      for (std::size_t i = 0; i < pyramid_after.size(); ++i) {
        pyramid_after[i] = BufferPayload(pyramid_buffers[i], kPyramidBytes[i]);
        canary_mismatches +=
            CanaryMismatches(pyramid_buffers[i], kPyramidBytes[i]);
      }
      std::vector<std::uint8_t> output_color(kWidth * kHeight * 4u);
      std::vector<std::uint8_t> native_output_color(kWidth * kHeight * 4u);
      std::vector<std::uint8_t> output_mask(kMaskBytes);
      [output_color_texture getBytes:output_color.data()
                         bytesPerRow:kWidth * 4u
                          fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                         mipmapLevel:0];
      [native_color_texture getBytes:native_output_color.data()
                         bytesPerRow:kWidth * 4u
                          fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                         mipmapLevel:0];
      [output_mask_texture getBytes:output_mask.data()
                        bytesPerRow:kWidth
                         fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                        mipmapLevel:0];
      const std::size_t input_mismatches =
          Mismatches(color_a, color_a_after) + Mismatches(color_b, color_b_after) +
          Mismatches(mask, mask_after) + Mismatches(motion, motion_after) +
          Mismatches(sigmoid_lut, sigmoid_after) + Mismatches(rcp_rom, rcp_after);

      Write(output_root / "color-a-after.bin", color_a_after);
      Write(output_root / "color-b-after.bin", color_b_after);
      Write(output_root / "mask-after.bin", mask_after);
      Write(output_root / "motion-after.bin", motion_after);
      Write(output_root / "pixel-scratch.bin", scratch_after);
      Write(output_root / "output-color.bin", output_color);
      Write(output_root / "native-metal-output-color.bin", native_output_color);
      Write(output_root / "output-mask.bin", output_mask);
      for (std::size_t i = 0; i < pyramid_after.size(); ++i)
        Write(output_root / ("pyramid-" + std::to_string(i) + ".bin"),
              pyramid_after[i]);
      std::vector<std::uint8_t> output_combined = output_color;
      output_combined.insert(output_combined.end(), output_mask.begin(),
                             output_mask.end());
      for (const auto& level : pyramid_after)
        output_combined.insert(output_combined.end(), level.begin(), level.end());
      std::vector<std::uint8_t> input_combined = color_a;
      input_combined.insert(input_combined.end(), color_b.begin(), color_b.end());
      input_combined.insert(input_combined.end(), mask.begin(), mask.end());
      input_combined.insert(input_combined.end(), motion.begin(), motion.end());
      const bool passed = input_mismatches == 0 && canary_mismatches == 0;

      std::ofstream receipt(output_root / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.postprocess-l84-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"fixture\": \"" << fixture << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"selected_launch_token\": 84,\n"
              << "  \"module_token\": 17,\n"
              << "  \"module_bytes\": 33936,\n"
              << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
              << "  \"raw_live_parameter_sha256\": \"" << kRawParameterSha
              << "\",\n"
              << "  \"packed_parameter_bytes\": 152,\n"
              << "  \"grid\": [40, 23, 1],\n"
              << "  \"block\": [256, 1, 1],\n"
              << "  \"input_generator\": \"" << kGenerator << "\",\n"
              << "  \"input_root_sha256\": \"" << Sha256(input_combined)
              << "\",\n"
              << "  \"output_root_sha256\": \"" << Sha256(output_combined)
              << "\",\n"
              << "  \"output_color_sha256\": \"" << Sha256(output_color)
              << "\",\n"
              << "  \"native_metal_output_color_sha256\": \""
              << Sha256(native_output_color) << "\",\n"
              << "  \"output_mask_sha256\": \"" << Sha256(output_mask)
              << "\",\n"
              << "  \"pyramid_sha256\": [\"" << Sha256(pyramid_after[0])
              << "\", \"" << Sha256(pyramid_after[1]) << "\", \""
              << Sha256(pyramid_after[2]) << "\", \""
              << Sha256(pyramid_after[3]) << "\"],\n"
              << "  \"output_bytes\": " << output_combined.size() << ",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"buffer_canary_mismatched_bytes\": "
              << canary_mismatches << ",\n"
              << "  \"sm120_rcp_rom_sha256\": \"" << kRcpRomSha << "\",\n"
              << "  \"sm120_half_logit_sigmoid_lut_sha256\": \""
              << kSigmoidSha << "\",\n"
              << "  \"expected_output_fed_to_execution\": false,\n"
              << "  \"archived_intermediates_consumed\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (passed ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "probe=fg_postprocess_l84_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "fixture=" << fixture << '\n'
                << "output_root_sha256=" << Sha256(output_combined) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "buffer_canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
