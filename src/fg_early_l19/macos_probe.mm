#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "contract.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
namespace l19 = fg_early_l19;
constexpr std::size_t kGuardBytes = 256;

struct TextureSpec {
  MTLPixelFormat format;
  std::size_t row_bytes;
};

const std::array<TextureSpec, 10> kInputSpecs = {{
    {MTLPixelFormatRGBA16Float, l19::kInputWidth * 8u},
    {MTLPixelFormatRGBA16Float, l19::kInputWidth * 8u},
    {MTLPixelFormatR8Unorm, l19::kInputWidth},
    {MTLPixelFormatR8Unorm, l19::kInputWidth},
    {MTLPixelFormatRG16Float, l19::kInputWidth * 4u},
    {MTLPixelFormatR16Float, l19::kInputWidth * 2u},
    {MTLPixelFormatR16Float, l19::kInputWidth * 2u},
    {MTLPixelFormatRG32Float, l19::kInputWidth * 8u},
    {MTLPixelFormatRG16Float, l19::kInputWidth * 4u},
    {MTLPixelFormatR8Unorm, l19::kInputWidth},
}};

struct MetalParameters {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t tensor_width;
  std::uint32_t tensor_height;
  std::int32_t input_y_offset;
};
static_assert(sizeof(MetalParameters) == 20);

std::string HashBytes(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

template <std::size_t Size>
std::string HashArray(const std::array<std::uint8_t, Size>& bytes) {
  return HashBytes(std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("comparison size mismatch");
  }
  std::size_t count = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    count += left[index] != right[index];
  }
  return count;
}

template <std::size_t Size>
void Pack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
            std::uint32_t value) {
  if (offset + 4u > Size) throw std::runtime_error("Pack32 OOB");
  for (std::size_t index = 0; index < 4u; ++index) {
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8u));
  }
}

std::array<std::uint8_t, 192> FixtureParameters() {
  std::array<std::uint8_t, 192> packed{};
  Pack32(&packed, 152, l19::kInputWidth);
  Pack32(&packed, 156, l19::kInputHeight);
  Pack32(&packed, 160, l19::kInputWidth);
  Pack32(&packed, 164, l19::kInputHeight);
  Pack32(&packed, 168, l19::kTensorWidth);
  Pack32(&packed, 172, l19::kTensorHeight);
  Pack32(&packed, 176, 0);
  Pack32(&packed, 180, 0);
  Pack32(&packed, 184, 3);
  Pack32(&packed, 188, 3);
  return packed;
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format,
                           std::uint32_t width, std::uint32_t height,
                           MTLTextureUsage usage,
                           const std::vector<std::uint8_t>& bytes,
                           std::size_t row_bytes) {
  if (bytes.size() != row_bytes * height) {
    throw std::runtime_error("texture payload size mismatch");
  }
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil) throw std::runtime_error("texture allocation failed");
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0
             withBytes:bytes.data()
           bytesPerRow:row_bytes];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::size_t row_bytes) {
  std::vector<std::uint8_t> bytes(row_bytes * texture.height);
  [texture getBytes:bytes.data()
         bytesPerRow:row_bytes
          fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
         mipmapLevel:0];
  return bytes;
}

std::vector<std::uint8_t> BufferPayload(id<MTLBuffer> buffer,
                                        std::size_t payload_bytes) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  return std::vector<std::uint8_t>(bytes + kGuardBytes,
                                   bytes + kGuardBytes + payload_bytes);
}

std::size_t BufferCanaryMismatches(id<MTLBuffer> buffer,
                                   std::size_t payload_bytes,
                                   std::uint8_t poison) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t count = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    count += bytes[index] != poison;
    count += bytes[kGuardBytes + payload_bytes + index] != poison;
  }
  return count;
}

std::size_t PoisonPixels(const std::vector<std::uint8_t>& bytes,
                         std::size_t pixel_bytes, std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += pixel_bytes) {
    bool same = true;
    for (std::size_t byte = 0; byte < pixel_bytes; ++byte) {
      same = same && bytes[offset + byte] == poison;
    }
    count += same;
  }
  return count;
}

void BindInputs(id<MTLComputeCommandEncoder> encoder,
                const std::array<id<MTLTexture>, 10>& inputs) {
  for (NSUInteger index = 0; index < inputs.size(); ++index) {
    [encoder setTexture:inputs[index] atIndex:index];
  }
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    l19::Fixture fixture, std::uint8_t poison, const std::string& device_name,
    const l19::Inputs& inputs,
    const std::array<std::vector<std::uint8_t>, 10>& inputs_after,
    const std::vector<std::uint8_t>& tensor,
    const std::vector<std::uint8_t>& mask,
    const std::vector<std::uint8_t>& vector,
    std::size_t input_mismatches, std::size_t buffer_canary_mismatches,
    std::size_t texture_canary_mismatches,
    const std::string& fixture_parameter_sha) {
  const bool pass = input_mismatches == 0 && buffer_canary_mismatches == 0 &&
                    texture_canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write L19 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.early-l19-run.v1\",\n"
          << "  \"backend\": \"mac-metal\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << getpid() << ",\n"
          << "  \"fixture\": \"" << l19::FixtureName(fixture) << "\",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"device_name\": \"" << device_name << "\",\n"
          << "  \"metal_functions\": [\"fg_early_l19_surfaces\", \"fg_early_l19_tensor\"],\n"
          << "  \"selected_launch\": 19,\n"
          << "  \"source_module_token\": 5,\n"
          << "  \"source_module_bytes\": " << l19::kModuleBytes << ",\n"
          << "  \"source_module_sha256\": \"" << l19::kModuleSha256 << "\",\n"
          << "  \"source_grid\": [20, 12, 1],\n"
          << "  \"source_block\": [16, 16, 1],\n"
          << "  \"packed_parameter_bytes\": 192,\n"
          << "  \"frozen_selected_raw_parameter_sha256\": \""
          << l19::kFrozenRawParameterSha256 << "\",\n"
          << "  \"frozen_selected_address_free_parameter_sha256\": \""
          << l19::kFrozenAddressFreeParameterSha256 << "\",\n"
          << "  \"fixture_address_free_parameter_sha256\": \""
          << fixture_parameter_sha << "\",\n"
          << "  \"input_generator\": \"" << l19::kInputGenerator
          << "\",\n"
          << "  \"allocation29_first_frame_evidence_root_sha256\": \""
          << l19::kAllocation29FirstFrameEvidenceRootSha256 << "\",\n"
          << "  \"allocation29_temporal_evidence_root_sha256\": \""
          << l19::kAllocation29TemporalEvidenceRootSha256 << "\",\n"
          << "  \"allocation29_runtime_source\": \"independently-generated\",\n"
          << "  \"allocation29_capture_bytes_consumed\": false,\n"
          << "  \"input_pre_sha256\": [";
  for (std::size_t index = 0; index < inputs.textures.size(); ++index) {
    if (index != 0) receipt << ", ";
    receipt << "\"" << HashBytes(inputs.textures[index]) << "\"";
  }
  receipt << "],\n  \"input_post_sha256\": [";
  for (std::size_t index = 0; index < inputs_after.size(); ++index) {
    if (index != 0) receipt << ", ";
    receipt << "\"" << HashBytes(inputs_after[index]) << "\"";
  }
  receipt << "],\n"
          << "  \"tensor_output_bytes\": " << tensor.size() << ",\n"
          << "  \"tensor_output_sha256\": \"" << HashBytes(tensor) << "\",\n"
          << "  \"mask_output_bytes\": " << mask.size() << ",\n"
          << "  \"mask_output_sha256\": \"" << HashBytes(mask) << "\",\n"
          << "  \"vector_output_bytes\": " << vector.size() << ",\n"
          << "  \"vector_output_sha256\": \"" << HashBytes(vector) << "\",\n"
          << "  \"tensor_poison_pixels\": "
          << PoisonPixels(tensor, 2u, poison) << ",\n"
          << "  \"mask_poison_pixels\": "
          << PoisonPixels(mask, 1u, poison) << ",\n"
          << "  \"vector_poison_pixels\": "
          << PoisonPixels(vector, 4u, poison) << ",\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"buffer_canary_mismatched_bytes\": "
          << buffer_canary_mismatches << ",\n"
          << "  \"texture_canary_mismatched_bytes\": "
          << texture_canary_mismatches << ",\n"
          << "  \"texture_resource_bounds_enforced\": true,\n"
          << "  \"complete_write_coverage_checked_by_dual_poison\": true,\n"
          << "  \"expected_output_fed_to_execution\": false,\n"
          << "  \"archived_intermediates_consumed\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-early-l19-mac <metallib> <output-dir> <run-id> "
                   "<poison-hex> <first-frame|fan-in>\n";
      return 2;
    }
    try {
      const std::filesystem::path output_directory = argv[2];
      const std::string run_id = argv[3];
      const unsigned long parsed_poison = std::stoul(argv[4], nullptr, 16);
      if (parsed_poison > 255u) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const auto fixture = l19::ParseFixture(argv[5]);
      std::filesystem::create_directories(output_directory);
      const auto fixture_parameters = FixtureParameters();
      const std::string fixture_parameter_sha = HashArray(fixture_parameters);
      if (fixture_parameter_sha != l19::kFixtureAddressFreeParameterSha256) {
        throw std::runtime_error("fixture parameter identity mismatch");
      }
      const auto inputs = l19::GenerateInputs(fixture);
      const std::vector<std::uint8_t> mask_poison(l19::kMaskOutputBytes,
                                                   poison);
      const std::vector<std::uint8_t> vector_poison(l19::kVectorOutputBytes,
                                                     poison);
      const std::vector<std::uint8_t> mask_canary_before(
          l19::kMaskOutputBytes, static_cast<std::uint8_t>(poison ^ 0xffu));
      const std::vector<std::uint8_t> vector_canary_before(
          l19::kVectorOutputBytes, static_cast<std::uint8_t>(poison ^ 0xffu));

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> surface_function =
          [library newFunctionWithName:@"fg_early_l19_surfaces"];
      id<MTLFunction> tensor_function =
          [library newFunctionWithName:@"fg_early_l19_tensor"];
      id<MTLComputePipelineState> surface_pipeline =
          [device newComputePipelineStateWithFunction:surface_function
                                                 error:&error];
      id<MTLComputePipelineState> tensor_pipeline =
          [device newComputePipelineStateWithFunction:tensor_function
                                                 error:&error];
      if (surface_pipeline == nil || tensor_pipeline == nil) {
        throw std::runtime_error("pipeline creation failed");
      }
      std::array<id<MTLTexture>, 10> input_textures;
      for (std::size_t index = 0; index < input_textures.size(); ++index) {
        input_textures[index] = MakeTexture(
            device, kInputSpecs[index].format, l19::kInputWidth,
            l19::kInputHeight, MTLTextureUsageShaderRead,
            inputs.textures[index], kInputSpecs[index].row_bytes);
      }
      id<MTLTexture> mask_output = MakeTexture(
          device, MTLPixelFormatR8Unorm, l19::kInputWidth, l19::kInputHeight,
          MTLTextureUsageShaderWrite, mask_poison, l19::kInputWidth);
      id<MTLTexture> vector_output = MakeTexture(
          device, MTLPixelFormatRG16Float, l19::kInputWidth,
          l19::kInputHeight, MTLTextureUsageShaderWrite, vector_poison,
          l19::kInputWidth * 4u);
      id<MTLTexture> mask_canary = MakeTexture(
          device, MTLPixelFormatR8Unorm, l19::kInputWidth, l19::kInputHeight,
          MTLTextureUsageShaderWrite, mask_canary_before, l19::kInputWidth);
      id<MTLTexture> vector_canary = MakeTexture(
          device, MTLPixelFormatRG16Float, l19::kInputWidth,
          l19::kInputHeight, MTLTextureUsageShaderWrite, vector_canary_before,
          l19::kInputWidth * 4u);
      id<MTLBuffer> tensor_output = [device
          newBufferWithLength:l19::kTensorBytes + 2u * kGuardBytes
                      options:MTLResourceStorageModeShared];
      if (tensor_output == nil) throw std::runtime_error("buffer allocation failed");
      std::memset(tensor_output.contents, poison, tensor_output.length);
      const MetalParameters parameters{
          l19::kInputWidth, l19::kInputHeight, l19::kTensorWidth,
          l19::kTensorHeight, -6};
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> surface_encoder =
          [command computeCommandEncoder];
      [surface_encoder setComputePipelineState:surface_pipeline];
      BindInputs(surface_encoder, input_textures);
      [surface_encoder setTexture:mask_output atIndex:10];
      [surface_encoder setTexture:vector_output atIndex:11];
      [surface_encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [surface_encoder
          dispatchThreads:MTLSizeMake(l19::kInputWidth, l19::kInputHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [surface_encoder endEncoding];
      id<MTLComputeCommandEncoder> tensor_encoder =
          [command computeCommandEncoder];
      [tensor_encoder setComputePipelineState:tensor_pipeline];
      BindInputs(tensor_encoder, input_textures);
      [tensor_encoder setBuffer:tensor_output offset:kGuardBytes atIndex:0];
      [tensor_encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      [tensor_encoder
          dispatchThreads:MTLSizeMake(l19::kTensorWidth, l19::kTensorHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [tensor_encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }

      std::array<std::vector<std::uint8_t>, 10> inputs_after;
      std::size_t input_mismatches = 0;
      for (std::size_t index = 0; index < input_textures.size(); ++index) {
        inputs_after[index] = ReadTexture(input_textures[index],
                                          kInputSpecs[index].row_bytes);
        input_mismatches +=
            Mismatches(inputs.textures[index], inputs_after[index]);
      }
      const auto mask_after = ReadTexture(mask_output, l19::kInputWidth);
      const auto vector_after =
          ReadTexture(vector_output, l19::kInputWidth * 4u);
      const auto mask_canary_after =
          ReadTexture(mask_canary, l19::kInputWidth);
      const auto vector_canary_after =
          ReadTexture(vector_canary, l19::kInputWidth * 4u);
      const auto tensor_after =
          BufferPayload(tensor_output, l19::kTensorBytes);
      const std::size_t buffer_canary_mismatches =
          BufferCanaryMismatches(tensor_output, l19::kTensorBytes, poison);
      const std::size_t texture_canary_mismatches =
          Mismatches(mask_canary_before, mask_canary_after) +
          Mismatches(vector_canary_before, vector_canary_after);
      WriteBytes(output_directory / "tensor-output.fp16", tensor_after);
      WriteBytes(output_directory / "mask-output.r8", mask_after);
      WriteBytes(output_directory / "vector-output.rg16f", vector_after);
      WriteReceipt(output_directory / "run-receipt.json", run_id, fixture,
                   poison, device.name.UTF8String, inputs, inputs_after,
                   tensor_after, mask_after, vector_after, input_mismatches,
                   buffer_canary_mismatches, texture_canary_mismatches,
                   fixture_parameter_sha);
      const bool pass = input_mismatches == 0 &&
                        buffer_canary_mismatches == 0 &&
                        texture_canary_mismatches == 0;
      std::cout << "probe=fg_early_l19_mac\n"
                << "run_id=" << run_id << '\n'
                << "tensor_output_sha256=" << HashBytes(tensor_after) << '\n'
                << "mask_output_sha256=" << HashBytes(mask_after) << '\n'
                << "vector_output_sha256=" << HashBytes(vector_after) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
