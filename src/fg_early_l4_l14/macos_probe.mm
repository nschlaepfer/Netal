#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "contract.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {
namespace contract = fg_early_l4_l14;

std::string HashBytes(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::size_t CountMismatches(const std::vector<std::uint8_t>& left,
                            const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("mismatch inputs differ in size");
  }
  std::size_t count = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    count += left[index] != right[index];
  }
  return count;
}

struct SourceParameters {
  std::uint64_t vector_input;
  std::array<std::uint32_t, 2> vector_origin;
  std::array<std::uint32_t, 2> vector_extent;
  std::array<float, 2> vector_inverse_extent;
  std::uint64_t vector_output;
  std::uint64_t scalar_input;
  std::array<std::uint32_t, 2> scalar_origin;
  std::array<std::uint32_t, 2> scalar_extent;
  std::array<float, 2> scalar_inverse_extent;
  float abi_dead_threshold;
  float score_numerator;
  float abi_dead_scale;
  std::uint8_t mode;
  std::array<std::uint8_t, 3> mode_padding;
  std::uint64_t depth_output;
  std::array<std::uint64_t, 6> abi_dead;
  std::array<std::uint32_t, 2> output_extent;
  std::array<float, 2> vector_output_scale;
};
static_assert(sizeof(SourceParameters) == 160);
static_assert(offsetof(SourceParameters, vector_output) == 32);
static_assert(offsetof(SourceParameters, scalar_input) == 40);
static_assert(offsetof(SourceParameters, mode) == 84);
static_assert(offsetof(SourceParameters, depth_output) == 88);
static_assert(offsetof(SourceParameters, output_extent) == 144);

struct MetalParameters {
  std::uint32_t reset_mode;
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
};
static_assert(sizeof(MetalParameters) == 20);

SourceParameters MakeSourceParameters(bool reset) {
  return SourceParameters{
      0,
      {0, 0},
      {contract::kInputWidth, contract::kInputHeight},
      {std::bit_cast<float>(0x3acccccdu),
       std::bit_cast<float>(0x3b360b61u)},
      0,
      0,
      {0, 0},
      {contract::kInputWidth, contract::kInputHeight},
      {std::bit_cast<float>(0x3acccccdu),
       std::bit_cast<float>(0x3b360b61u)},
      40.0f,
      1.0f,
      600.0f,
      static_cast<std::uint8_t>(reset ? 127u : 0u),
      {0, 0, 0},
      0,
      {0, 0, 0, 0, 0, 0},
      {contract::kOutputWidth, contract::kOutputHeight},
      {std::bit_cast<float>(0x3623d70au),
       std::bit_cast<float>(0x3701742eu)}};
}

std::vector<std::uint8_t> BytesOf(const SourceParameters& value) {
  const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
  return std::vector<std::uint8_t>(first, first + sizeof(value));
}

id<MTLTexture> MakeL4L14Texture(
    id<MTLDevice> device, MTLPixelFormat format, std::uint32_t width,
    std::uint32_t height, MTLTextureUsage usage,
    const std::vector<std::uint8_t>& bytes, std::size_t row_bytes) {
  if (bytes.size() != row_bytes * height) {
    throw std::runtime_error("L4/L14 texture payload size mismatch");
  }
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("L4/L14 texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadL4L14Texture(id<MTLTexture> texture,
                                           std::size_t row_bytes) {
  std::vector<std::uint8_t> bytes(row_bytes * texture.height);
  [texture getBytes:bytes.data()
         bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
         mipmapLevel:0];
  return bytes;
}

std::vector<std::uint8_t> SurfacePayload(
    const std::vector<std::uint8_t>& guarded, std::size_t payload_bytes) {
  return std::vector<std::uint8_t>(guarded.begin(),
                                   guarded.begin() + payload_bytes);
}

std::size_t TailCanaryMismatches(const std::vector<std::uint8_t>& guarded,
                                 std::size_t payload_bytes,
                                 std::uint8_t poison) {
  std::size_t mismatches = 0;
  for (std::size_t index = payload_bytes; index < guarded.size(); ++index) {
    mismatches += guarded[index] != poison;
  }
  return mismatches;
}

std::size_t UntouchedPixels(const std::vector<std::uint8_t>& payload,
                            std::size_t pixel_bytes, std::uint8_t poison) {
  std::size_t untouched = 0;
  for (std::size_t offset = 0; offset < payload.size(); offset += pixel_bytes) {
    bool same = true;
    for (std::size_t byte = 0; byte < pixel_bytes; ++byte) {
      same = same && payload[offset + byte] == poison;
    }
    untouched += same;
  }
  return untouched;
}

std::size_t UniqueWords(const std::vector<std::uint8_t>& bytes) {
  std::set<std::uint32_t> words;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 4u) {
    std::uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    words.insert(word);
  }
  return words.size();
}

void WriteL4L14Receipt(
    const std::filesystem::path& path, const std::string& run_id,
    const std::string& phase, contract::Fixture fixture, std::uint8_t poison,
    const std::string& device_name,
    const std::vector<std::uint8_t>& vector_before,
    const std::vector<std::uint8_t>& vector_after,
    const std::vector<std::uint8_t>& scalar_before,
    const std::vector<std::uint8_t>& scalar_after,
    const std::vector<std::uint8_t>& vector_output,
    const std::vector<std::uint8_t>& depth_output,
    const std::string& fixture_address_free_sha,
    std::size_t input_mismatches,
    std::size_t vector_canary_mismatches,
    std::size_t depth_canary_mismatches,
    std::size_t vector_untouched_pixels,
    std::size_t depth_untouched_pixels) {
  const bool reset = phase == "reset";
  const bool pass = input_mismatches == 0 && vector_canary_mismatches == 0 &&
                    depth_canary_mismatches == 0 &&
                    vector_untouched_pixels == 0 &&
                    depth_untouched_pixels == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L4/L14 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l4-l14-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"phase\": \"" << phase << "\",\n"
         << "  \"fixture\": \"" << contract::FixtureName(fixture) << "\",\n"
         << "  \"selected_launch\": " << (reset ? 4 : 14) << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"metal_function\": \"fg_early_l4_l14\",\n"
         << "  \"source_module_token\": 15,\n"
         << "  \"source_module_bytes\": " << contract::kModuleBytes << ",\n"
         << "  \"source_module_sha256\": \"" << contract::kModuleSha256
         << "\",\n"
         << "  \"source_grid\": [20, 12, 1],\n"
         << "  \"source_block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 160,\n"
         << "  \"frozen_selected_raw_parameter_sha256\": \""
         << (reset ? contract::kResetTraceRawParameterSha256
                   : contract::kGeneratedTraceRawParameterSha256)
         << "\",\n"
         << "  \"frozen_selected_address_free_parameter_sha256\": \""
         << (reset ? contract::kResetAddressFreeParameterSha256
                   : contract::kGeneratedAddressFreeParameterSha256)
         << "\",\n"
         << "  \"fixture_address_free_parameter_sha256\": \""
         << fixture_address_free_sha << "\",\n"
         << "  \"input_generator\": \"" << contract::kInputGenerator
         << "\",\n"
         << "  \"resource_contract\": {\"vector_input\": \"RG16_FLOAT 640x360 point-clamp\", \"scalar_input\": \"R32_FLOAT 640x360 point-border\", \"vector_output\": \"RG32_FLOAT 320x180\", \"depth_output\": \"R32_UINT 320x180\"},\n"
         << "  \"vector_input_pre_sha256\": \"" << HashBytes(vector_before)
         << "\",\n"
         << "  \"vector_input_post_sha256\": \"" << HashBytes(vector_after)
         << "\",\n"
         << "  \"scalar_input_pre_sha256\": \"" << HashBytes(scalar_before)
         << "\",\n"
         << "  \"scalar_input_post_sha256\": \"" << HashBytes(scalar_after)
         << "\",\n"
         << "  \"vector_output_bytes\": " << vector_output.size() << ",\n"
         << "  \"vector_output_sha256\": \"" << HashBytes(vector_output)
         << "\",\n"
         << "  \"depth_output_bytes\": " << depth_output.size() << ",\n"
         << "  \"depth_output_sha256\": \"" << HashBytes(depth_output)
         << "\",\n"
         << "  \"depth_unique_u32_values\": " << UniqueWords(depth_output)
         << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"vector_canary_mismatched_bytes\": "
         << vector_canary_mismatches << ",\n"
         << "  \"depth_canary_mismatched_bytes\": "
         << depth_canary_mismatches << ",\n"
         << "  \"vector_untouched_pixels\": " << vector_untouched_pixels
         << ",\n"
         << "  \"depth_untouched_pixels\": " << depth_untouched_pixels
         << ",\n"
         << "  \"expected_output_fed_to_execution\": false,\n"
         << "  \"archived_intermediates_consumed\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 7) {
      std::cerr << "usage=fg-early-l4-l14-mac <metallib> <output-dir> "
                   "<run-id> <poison-hex> <reset|generated> "
                   "<branch-grid|center-tie>\n";
      return 2;
    }
    try {
      const std::string phase = argv[5];
      if (phase != "reset" && phase != "generated") {
        throw std::runtime_error("unsupported L4/L14 phase");
      }
      const auto fixture = contract::ParseFixture(argv[6]);
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[4], &poison_end, 16);
      if (poison_end == argv[4] || *poison_end != '\0' ||
          parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const std::filesystem::path output_root = argv[2];
      const std::string run_id = argv[3];
      std::filesystem::create_directories(output_root);
      const bool reset = phase == "reset";
      const auto source_parameters = MakeSourceParameters(reset);
      const std::string fixture_address_free_sha =
          HashBytes(BytesOf(source_parameters));
      const char* expected_fixture_sha =
          reset ? contract::kResetFixtureAddressFreeParameterSha256
                : contract::kGeneratedFixtureAddressFreeParameterSha256;
      if (fixture_address_free_sha != expected_fixture_sha) {
        throw std::runtime_error("fixture address-free parameter mismatch");
      }
      const auto vector_before = contract::VectorInput(fixture);
      const auto scalar_before = contract::ScalarInput(fixture);
      const std::size_t vector_guarded_bytes =
          static_cast<std::size_t>(contract::kOutputWidth) *
          contract::kOutputRowsWithCanary * 8u;
      const std::size_t depth_guarded_bytes =
          static_cast<std::size_t>(contract::kOutputWidth) *
          contract::kOutputRowsWithCanary * 4u;
      const std::vector<std::uint8_t> vector_poison(vector_guarded_bytes,
                                                     poison);
      const std::vector<std::uint8_t> depth_poison(depth_guarded_bytes,
                                                    poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_early_l4_l14"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLTexture> vector_input = MakeL4L14Texture(
          device, MTLPixelFormatRG16Float, contract::kInputWidth,
          contract::kInputHeight, MTLTextureUsageShaderRead, vector_before,
          contract::kInputWidth * 4u);
      id<MTLTexture> scalar_input = MakeL4L14Texture(
          device, MTLPixelFormatR32Float, contract::kInputWidth,
          contract::kInputHeight, MTLTextureUsageShaderRead, scalar_before,
          contract::kInputWidth * 4u);
      id<MTLTexture> vector_output = MakeL4L14Texture(
          device, MTLPixelFormatRG32Float, contract::kOutputWidth,
          contract::kOutputRowsWithCanary, MTLTextureUsageShaderWrite,
          vector_poison, contract::kOutputWidth * 8u);
      id<MTLTexture> depth_output = MakeL4L14Texture(
          device, MTLPixelFormatR32Uint, contract::kOutputWidth,
          contract::kOutputRowsWithCanary, MTLTextureUsageShaderWrite,
          depth_poison, contract::kOutputWidth * 4u);
      const MetalParameters parameters{
          reset ? 1u : 0u, contract::kInputWidth, contract::kInputHeight,
          contract::kOutputWidth, contract::kOutputHeight};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:vector_input atIndex:0];
      [encoder setTexture:scalar_input atIndex:1];
      [encoder setTexture:vector_output atIndex:2];
      [encoder setTexture:depth_output atIndex:3];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [encoder dispatchThreads:MTLSizeMake(contract::kOutputWidth,
                                           contract::kOutputHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      const auto vector_after =
          ReadL4L14Texture(vector_input, contract::kInputWidth * 4u);
      const auto scalar_after =
          ReadL4L14Texture(scalar_input, contract::kInputWidth * 4u);
      const auto vector_guarded =
          ReadL4L14Texture(vector_output, contract::kOutputWidth * 8u);
      const auto depth_guarded =
          ReadL4L14Texture(depth_output, contract::kOutputWidth * 4u);
      const auto vector_payload = SurfacePayload(
          vector_guarded, contract::VectorOutputPayloadBytes());
      const auto depth_payload = SurfacePayload(
          depth_guarded, contract::DepthOutputPayloadBytes());
      const std::size_t input_mismatches =
          CountMismatches(vector_before, vector_after) +
          CountMismatches(scalar_before, scalar_after);
      const std::size_t vector_canary_mismatches = TailCanaryMismatches(
          vector_guarded, vector_payload.size(), poison);
      const std::size_t depth_canary_mismatches = TailCanaryMismatches(
          depth_guarded, depth_payload.size(), poison);
      const std::size_t vector_untouched =
          UntouchedPixels(vector_payload, 8u, poison);
      const std::size_t depth_untouched =
          UntouchedPixels(depth_payload, 4u, poison);
      WriteBytes(output_root / "vector-output.bin", vector_payload);
      WriteBytes(output_root / "depth-output.bin", depth_payload);
      WriteL4L14Receipt(
          output_root / "run-receipt.json", run_id, phase, fixture, poison,
          device.name.UTF8String, vector_before, vector_after, scalar_before,
          scalar_after, vector_payload, depth_payload,
          fixture_address_free_sha, input_mismatches,
          vector_canary_mismatches, depth_canary_mismatches,
          vector_untouched, depth_untouched);
      const bool pass = input_mismatches == 0 &&
                        vector_canary_mismatches == 0 &&
                        depth_canary_mismatches == 0 &&
                        vector_untouched == 0 && depth_untouched == 0;
      std::cout << "probe=fg_early_l4_l14_mac\n"
                << "run_id=" << run_id << '\n'
                << "vector_output_sha256=" << HashBytes(vector_payload) << '\n'
                << "depth_output_sha256=" << HashBytes(depth_payload) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
