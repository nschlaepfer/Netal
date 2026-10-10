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

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 180;
constexpr std::uint32_t kInputWidth = 160;
constexpr std::uint32_t kInputHeight = 96;
constexpr std::uint32_t kChannels = 16;
constexpr std::size_t kInputBytes =
    std::size_t{kInputWidth} * kInputHeight * kChannels * 2u;
constexpr std::size_t kOutputVectorBytes =
    std::size_t{kWidth} * kHeight * 4u;
constexpr std::size_t kOutputScalarBytes =
    std::size_t{kWidth} * kHeight * 2u;
constexpr std::size_t kDepthBytes = std::size_t{kWidth} * kHeight;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr std::size_t kRcpRomBytes = 2097152;
constexpr char kRcpRomSha[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
constexpr char kModuleSha[] =
    "d45333f43f343d1f38bd304fe76214dc7064f2a7d24ce1ea06bc791867783264";
constexpr char kRawParameterSha[] =
    "b85b7b8f373002d0518a784cfa9e9d8511979be6dd2641106b5c5acfb4a2a01d";
constexpr char kGenerator[] =
    "splitmix64-fg-mid-l52-half16-depth-v1";

struct Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t input_offset_x;
  std::uint32_t input_offset_y;
  std::uint32_t crop_top;
  std::uint32_t crop_bottom;
};
static_assert(sizeof(Parameters) == 32);

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

std::uint16_t FeatureHalf(std::uint64_t pixel, std::uint32_t channel,
                          std::uint64_t salt) {
  static constexpr std::array<std::uint16_t, 16> kAnchors = {
      0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
      0x3400, 0xb400, 0x3800, 0xb800, 0x3a00, 0xba00, 0x3c00, 0xbc00};
  const auto mixed = SplitMix64(salt ^ pixel * 0xD6E8FEB86659FD93ull ^
                                std::uint64_t{channel + 1u} *
                                    0xA0761D6478BD642Full);
  if (pixel < 32u) return kAnchors[(pixel + channel) & 15u];
  const std::uint16_t sign = (mixed >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(9u + ((mixed >> 17u) % 8u));
  return static_cast<std::uint16_t>(
      sign | (exponent << 10u) | (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateInput(const std::string& fixture) {
  const std::uint64_t salt =
      fixture == "binary-depth" ? 0xE7037ED1A0B428DBull :
      fixture == "ramp-depth" ? 0x8EBC6AF09C88C6E3ull : 0ull;
  if (!salt) throw std::runtime_error("unsupported fixture");
  std::vector<std::uint8_t> bytes(kInputBytes);
  for (std::size_t pixel = 0;
       pixel < std::size_t{kInputWidth} * kInputHeight; ++pixel) {
    for (std::uint32_t channel = 0; channel < kChannels; ++channel)
      Store16(&bytes, (pixel * kChannels + channel) * 2u,
              FeatureHalf(pixel, channel, salt));
  }
  return bytes;
}

std::vector<std::uint8_t> GenerateDepth(const std::string& fixture) {
  std::vector<std::uint8_t> bytes(kDepthBytes);
  for (std::size_t pixel = 0; pixel < bytes.size(); ++pixel) {
    if (fixture == "binary-depth") {
      bytes[pixel] = ((pixel / kWidth + pixel % kWidth) & 1u) ? 0xffu : 0u;
    } else if (fixture == "ramp-depth") {
      const auto mixed = SplitMix64(0x589965CC75374CC3ull ^ pixel);
      bytes[pixel] = static_cast<std::uint8_t>(
          pixel < 256u ? pixel : (mixed >> 24u));
    } else {
      throw std::runtime_error("unsupported fixture");
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
  if (argc != 7) {
    std::cerr << "usage: fg-mid-l52-mac <metallib> <rcp-rom> <output-dir> <run-id> <fixture> <poison-hex>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const std::filesystem::path metallib_path = argv[1];
      const auto rcp_rom = Read(argv[2]);
      const std::filesystem::path output_root = argv[3];
      const std::string run_id = argv[4];
      const std::string fixture = argv[5];
      const std::uint8_t poison = ParsePoison(argv[6]);
      std::filesystem::create_directories(output_root);
      if (rcp_rom.size() != kRcpRomBytes || Sha256(rcp_rom) != kRcpRomSha)
        throw std::runtime_error("RCP ROM identity mismatch");
      const auto input = GenerateInput(fixture);
      const auto depth = GenerateDepth(fixture);

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
      auto pipeline = Pipeline(device, library, @"fg_mid_l52");
      id<MTLBuffer> input_buffer = Guarded(device, &input, kInputBytes, 0);
      id<MTLBuffer> rcp_buffer = Guarded(device, &rcp_rom, kRcpRomBytes, 0);
      const Parameters parameters{kWidth, kHeight, kInputWidth, kInputHeight,
                                  0u, 0u, 3u, 3u};
      id<MTLBuffer> parameter_buffer =
          [device newBufferWithBytes:&parameters
                              length:sizeof(parameters)
                             options:MTLResourceStorageModeShared];
      MTLTextureDescriptor* depth_descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      depth_descriptor.storageMode = MTLStorageModeShared;
      depth_descriptor.usage = MTLTextureUsageShaderRead;
      id<MTLTexture> depth_texture = [device newTextureWithDescriptor:depth_descriptor];
      MTLTextureDescriptor* vector_descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      vector_descriptor.storageMode = MTLStorageModeShared;
      vector_descriptor.usage = MTLTextureUsageShaderWrite;
      id<MTLTexture> output_a_texture = [device newTextureWithDescriptor:vector_descriptor];
      id<MTLTexture> output_b_texture = [device newTextureWithDescriptor:vector_descriptor];
      MTLTextureDescriptor* scalar_descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      scalar_descriptor.storageMode = MTLStorageModeShared;
      scalar_descriptor.usage = MTLTextureUsageShaderWrite;
      id<MTLTexture> confidence_texture = [device newTextureWithDescriptor:scalar_descriptor];
      if (!parameter_buffer || !depth_texture || !output_a_texture ||
          !output_b_texture || !confidence_texture)
        throw std::runtime_error("Metal resource allocation failed");
      const std::vector<std::uint8_t> initial_vector(kOutputVectorBytes, poison);
      const std::vector<std::uint8_t> initial_scalar(kOutputScalarBytes, poison);
      [depth_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                       mipmapLevel:0 withBytes:depth.data() bytesPerRow:kWidth];
      [output_a_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                          mipmapLevel:0 withBytes:initial_vector.data()
                          bytesPerRow:kWidth * 4u];
      [output_b_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                          mipmapLevel:0 withBytes:initial_vector.data()
                          bytesPerRow:kWidth * 4u];
      [confidence_texture replaceRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                            mipmapLevel:0 withBytes:initial_scalar.data()
                            bytesPerRow:kWidth * 2u];

      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input_buffer offset:kGuardBytes atIndex:0];
      [encoder setBuffer:rcp_buffer offset:kGuardBytes atIndex:1];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:2];
      [encoder setTexture:depth_texture atIndex:0];
      [encoder setTexture:output_a_texture atIndex:1];
      [encoder setTexture:output_b_texture atIndex:2];
      [encoder setTexture:confidence_texture atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      Complete(command);

      const auto input_after = BufferPayload(input_buffer, kInputBytes);
      const auto rcp_after = BufferPayload(rcp_buffer, kRcpRomBytes);
      std::vector<std::uint8_t> depth_after(kDepthBytes);
      std::vector<std::uint8_t> output_a(kOutputVectorBytes);
      std::vector<std::uint8_t> output_b(kOutputVectorBytes);
      std::vector<std::uint8_t> confidence(kOutputScalarBytes);
      [depth_texture getBytes:depth_after.data() bytesPerRow:kWidth
                   fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                  mipmapLevel:0];
      [output_a_texture getBytes:output_a.data() bytesPerRow:kWidth * 4u
                      fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                     mipmapLevel:0];
      [output_b_texture getBytes:output_b.data() bytesPerRow:kWidth * 4u
                      fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                     mipmapLevel:0];
      [confidence_texture getBytes:confidence.data() bytesPerRow:kWidth * 2u
                        fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
                       mipmapLevel:0];
      const std::size_t input_mismatches =
          Mismatches(input, input_after) + Mismatches(depth, depth_after) +
          Mismatches(rcp_rom, rcp_after);
      const std::size_t canary_mismatches =
          CanaryMismatches(input_buffer, kInputBytes) +
          CanaryMismatches(rcp_buffer, kRcpRomBytes);
      Write(output_root / "input-after.bin", input_after);
      Write(output_root / "depth-after.bin", depth_after);
      Write(output_root / "output-a.rg16f", output_a);
      Write(output_root / "output-b.rg16f", output_b);
      Write(output_root / "confidence.r16f", confidence);
      std::vector<std::uint8_t> inputs = input;
      inputs.insert(inputs.end(), depth.begin(), depth.end());
      std::vector<std::uint8_t> outputs = output_a;
      outputs.insert(outputs.end(), output_b.begin(), output_b.end());
      outputs.insert(outputs.end(), confidence.begin(), confidence.end());
      const bool passed = input_mismatches == 0 && canary_mismatches == 0;
      std::ofstream receipt(output_root / "run-receipt.json",
                            std::ios::binary | std::ios::trunc);
      receipt << "{\n"
              << "  \"schema\": \"neuralstudio.fg.mid-l52-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"fixture\": \"" << fixture << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"selected_launch_token\": 52,\n"
              << "  \"module_token\": 6,\n"
              << "  \"module_bytes\": 9064,\n"
              << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
              << "  \"raw_live_parameter_sha256\": \"" << kRawParameterSha << "\",\n"
              << "  \"packed_parameter_bytes\": 168,\n"
              << "  \"grid\": [20, 12, 1],\n"
              << "  \"block\": [16, 16, 1],\n"
              << "  \"input_generator\": \"" << kGenerator << "\",\n"
              << "  \"input_root_sha256\": \"" << Sha256(inputs) << "\",\n"
              << "  \"output_root_sha256\": \"" << Sha256(outputs) << "\",\n"
              << "  \"output_a_sha256\": \"" << Sha256(output_a) << "\",\n"
              << "  \"output_b_sha256\": \"" << Sha256(output_b) << "\",\n"
              << "  \"confidence_sha256\": \"" << Sha256(confidence) << "\",\n"
              << "  \"output_bytes\": " << outputs.size() << ",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"buffer_canary_mismatched_bytes\": " << canary_mismatches << ",\n"
              << "  \"sm120_rcp_rom_sha256\": \"" << kRcpRomSha << "\",\n"
              << "  \"expected_output_fed_to_execution\": false,\n"
              << "  \"archived_intermediates_consumed\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
              << "}\n";
      std::cout << "probe=fg_mid_l52_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "fixture=" << fixture << '\n'
                << "output_root_sha256=" << Sha256(outputs) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "buffer_canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
