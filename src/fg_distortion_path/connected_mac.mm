#include "connected_contract.h"

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

namespace connected = fg_distortion_connected;
constexpr std::size_t kRomBytes = 2097152;
constexpr std::size_t kRomGuardBytes = 256;
constexpr char kRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

std::string Hash(const std::vector<std::uint8_t> &bytes) {
  std::array<unsigned char, 32> digest{};
  CC_SHA256(bytes.data(), bytes.size(), digest.data());
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const auto value : digest)
    stream << std::setw(2) << unsigned(value);
  return stream.str();
}

std::vector<std::uint8_t> ReadFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot read " + path.string());
  const auto size = stream.tellg();
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(result.data()), size);
  return result;
}

void Write(const std::filesystem::path &path,
           const std::vector<std::uint8_t> &bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format,
                           std::size_t bytes_per_pixel, MTLTextureUsage usage,
                           const std::vector<std::uint8_t> &bytes) {
  auto *descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:format
                                   width:connected::kWidth
                                  height:connected::kHeight
                               mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (texture == nil || bytes.size() != connected::kPixels * bytes_per_pixel) {
    throw std::runtime_error("texture allocation or size");
  }
  [texture
      replaceRegion:MTLRegionMake2D(0, 0, connected::kWidth, connected::kHeight)
        mipmapLevel:0
          withBytes:bytes.data()
        bytesPerRow:connected::kWidth * bytes_per_pixel];
  return texture;
}

std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                      std::size_t bytes_per_pixel) {
  std::vector<std::uint8_t> result(connected::kPixels * bytes_per_pixel);
  [texture getBytes:result.data()
        bytesPerRow:connected::kWidth * bytes_per_pixel
         fromRegion:MTLRegionMake2D(0, 0, connected::kWidth, connected::kHeight)
        mipmapLevel:0];
  return result;
}

std::size_t Mismatches(const std::vector<std::uint8_t> &left,
                       const std::vector<std::uint8_t> &right) {
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result += left[index] != right[index];
  return result;
}

std::size_t CountPoisonPixels(const std::vector<std::uint8_t> &bytes,
                              std::size_t pixel_bytes, std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += pixel_bytes) {
    bool untouched = true;
    for (std::size_t byte = 0; byte < pixel_bytes; ++byte)
      untouched &= bytes[offset + byte] == poison;
    result += untouched;
  }
  return result;
}

std::vector<std::uint8_t> Coverage(const std::vector<std::uint8_t> &bytes,
                                   std::uint8_t poison) {
  std::vector<std::uint8_t> result(bytes.size());
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    if (bytes[index] == 0)
      result[index] = 1;
    else if (bytes[index] == poison)
      result[index] = 0;
    else
      throw std::runtime_error("scatter output escaped zero/poison domain");
  }
  return result;
}

std::size_t CountOnes(const std::vector<std::uint8_t> &bytes) {
  std::size_t result = 0;
  for (const auto value : bytes)
    result += value != 0;
  return result;
}

void Complete(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error("Metal command failed");
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 6)
    return 2;
  @autoreleasepool {
    try {
      const std::filesystem::path output_root = argv[3];
      std::filesystem::create_directories(output_root);
      const std::string run_id = argv[4];
      const std::uint8_t poison =
          static_cast<std::uint8_t>(std::stoul(argv[5], nullptr, 16));
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Hash(rom) != kRomSha256)
        throw std::runtime_error("RCP ROM identity");

      const auto external_reset = connected::Distortion(0);
      const auto previous_generated = connected::Distortion(1);
      const auto external_generated = connected::Distortion(2);
      const auto flow32 = connected::Flow32();
      const auto flow16 = connected::Flow16();
      const auto mask_reset = connected::Mask(0);
      const auto mask_generated = connected::Mask(1);
      std::vector<std::uint8_t> rgba_poison(connected::kRgba16Bytes, poison);
      std::vector<std::uint8_t> rg_poison(connected::kRg16Bytes, poison);
      std::vector<std::uint8_t> r32_poison(connected::kR32Bytes, poison);
      std::vector<std::uint8_t> r8_poison(connected::kR8Bytes, poison);
      std::vector<std::uint8_t> rgba_canary(connected::kRgba16Bytes,
                                            poison ^ 0xffu);
      std::vector<std::uint8_t> rg_canary(connected::kRg16Bytes,
                                          poison ^ 0xffu);
      std::vector<std::uint8_t> r32_canary(connected::kR32Bytes,
                                           poison ^ 0xffu);
      std::vector<std::uint8_t> r8_canary(connected::kR8Bytes, poison ^ 0xffu);

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
      const auto pipeline = ^id<MTLComputePipelineState>(NSString *name) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function == nil)
          throw std::runtime_error("Metal function missing");
        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function
                                                  error:&pipeline_error];
        if (result == nil)
          throw std::runtime_error("pipeline creation failed");
        return result;
      };
      id<MTLComputePipelineState> pipeline10 =
          pipeline(@"fg_distortion_module10");
      id<MTLComputePipelineState> pipeline9 =
          pipeline(@"fg_distortion_module9");
      id<MTLComputePipelineState> pipeline25 = pipeline(@"fg_early_l8_l18");
      constexpr MTLTextureUsage kReadWrite =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      auto ext_reset = MakeTexture(device, MTLPixelFormatRGBA16Float, 8,
                                   MTLTextureUsageShaderRead, external_reset);
      auto ext_generated =
          MakeTexture(device, MTLPixelFormatRGBA16Float, 8,
                      MTLTextureUsageShaderRead, external_generated);
      auto internal_reset = MakeTexture(device, MTLPixelFormatRGBA16Float, 8,
                                        kReadWrite, rgba_poison);
      auto internal_generated = MakeTexture(device, MTLPixelFormatRGBA16Float,
                                            8, kReadWrite, previous_generated);
      auto flow32_texture = MakeTexture(device, MTLPixelFormatRG32Float, 8,
                                        MTLTextureUsageShaderRead, flow32);
      auto flow16_texture = MakeTexture(device, MTLPixelFormatRG16Float, 4,
                                        MTLTextureUsageShaderRead, flow16);
      auto mask_reset_texture =
          MakeTexture(device, MTLPixelFormatR32Uint, 4, kReadWrite, mask_reset);
      auto mask_generated_texture =
          MakeTexture(device, MTLPixelFormatR32Uint, 4,
                      MTLTextureUsageShaderRead, mask_generated);
      auto flow_a = MakeTexture(device, MTLPixelFormatRG16Float, 4, kReadWrite,
                                rg_poison);
      auto flow_b = MakeTexture(device, MTLPixelFormatRG16Float, 4, kReadWrite,
                                rg_poison);
      auto reset_mask_output =
          MakeTexture(device, MTLPixelFormatR32Uint, 4,
                      MTLTextureUsageShaderWrite, r32_poison);
      auto scatter_reset = MakeTexture(device, MTLPixelFormatR8Unorm, 1,
                                       MTLTextureUsageShaderWrite, r8_poison);
      auto scatter_generated =
          MakeTexture(device, MTLPixelFormatR8Unorm, 1,
                      MTLTextureUsageShaderWrite, r8_poison);
      auto canary_rgba = MakeTexture(device, MTLPixelFormatRGBA16Float, 8,
                                     MTLTextureUsageShaderWrite, rgba_canary);
      auto canary_rg = MakeTexture(device, MTLPixelFormatRG16Float, 4,
                                   MTLTextureUsageShaderWrite, rg_canary);
      auto canary_r32 = MakeTexture(device, MTLPixelFormatR32Uint, 4,
                                    MTLTextureUsageShaderWrite, r32_canary);
      auto canary_r8 = MakeTexture(device, MTLPixelFormatR8Unorm, 1,
                                   MTLTextureUsageShaderWrite, r8_canary);
      id<MTLBuffer> rom_buffer =
          [device newBufferWithLength:kRomBytes + 2 * kRomGuardBytes
                              options:MTLResourceStorageModeShared];
      std::memset(rom_buffer.contents, poison ^ 0xffu,
                  kRomBytes + 2 * kRomGuardBytes);
      std::memcpy(static_cast<std::uint8_t *>(rom_buffer.contents) +
                      kRomGuardBytes,
                  rom.data(), rom.size());
      id<MTLCommandQueue> queue = [device newCommandQueue];

      const auto run10 = ^(id<MTLTexture> input, id<MTLTexture> output) {
        struct {
          std::uint32_t input_origin[2];
          std::uint32_t input_extent[2];
          float inverse_input_extent[2];
          std::uint32_t output_extent[2];
          float output_bias;
          float output_scale;
        } parameters{{0, 0},
                     {connected::kWidth, connected::kHeight},
                     {std::bit_cast<float>(0x3b4ccccdu),
                      std::bit_cast<float>(0x3bb60b61u)},
                     {connected::kWidth, connected::kHeight},
                     0.0f,
                     1.0f};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline10];
        [encoder setTexture:input atIndex:0];
        [encoder setTexture:output atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(connected::kWidth,
                                             connected::kHeight, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Complete(command);
      };
      const auto run9 = ^(id<MTLTexture> a, id<MTLTexture> b,
                          id<MTLTexture> mask_input,
                          id<MTLTexture> mask_output) {
        struct {
          std::uint32_t width, height;
        } parameters{connected::kWidth, connected::kHeight};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline9];
        [encoder setTexture:a atIndex:0];
        [encoder setTexture:b atIndex:1];
        [encoder setTexture:flow32_texture atIndex:2];
        [encoder setTexture:flow16_texture atIndex:3];
        [encoder setTexture:mask_input atIndex:4];
        [encoder setTexture:flow_a atIndex:5];
        [encoder setTexture:flow_b atIndex:6];
        [encoder setTexture:mask_output atIndex:7];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(connected::kWidth,
                                             connected::kHeight, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Complete(command);
      };
      const auto run25 = ^(id<MTLTexture> output) {
        struct {
          std::uint32_t width, height, mode;
        } parameters{connected::kWidth, connected::kHeight, 0};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline25];
        [encoder setTexture:flow_b atIndex:0];
        [encoder setTexture:output atIndex:1];
        [encoder setBuffer:rom_buffer offset:kRomGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(connected::kWidth,
                                             connected::kHeight, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Complete(command);
      };

      run10(ext_reset, internal_reset);
      const auto checkpoint10_reset = ReadTexture(internal_reset, 8);
      run9(internal_reset, internal_generated, mask_reset_texture,
           reset_mask_output);
      const auto checkpoint9_reset_a = ReadTexture(flow_a, 4);
      const auto checkpoint9_reset_b = ReadTexture(flow_b, 4);
      const auto checkpoint9_reset_mask = ReadTexture(reset_mask_output, 4);
      run25(scatter_reset);
      const auto checkpoint25_reset = ReadTexture(scatter_reset, 1);

      run10(ext_generated, internal_generated);
      const auto checkpoint10_generated = ReadTexture(internal_generated, 8);
      run9(internal_generated, internal_reset, mask_generated_texture,
           mask_reset_texture);
      const auto checkpoint9_generated_a = ReadTexture(flow_a, 4);
      const auto checkpoint9_generated_b = ReadTexture(flow_b, 4);
      const auto checkpoint9_generated_mask =
          ReadTexture(mask_reset_texture, 4);
      run25(scatter_generated);
      const auto checkpoint25_generated = ReadTexture(scatter_generated, 1);

      const auto reset_coverage = Coverage(checkpoint25_reset, poison);
      const auto generated_coverage = Coverage(checkpoint25_generated, poison);
      const auto external_reset_after = ReadTexture(ext_reset, 8);
      const auto external_generated_after = ReadTexture(ext_generated, 8);
      const auto flow32_after = ReadTexture(flow32_texture, 8);
      const auto flow16_after = ReadTexture(flow16_texture, 4);
      const auto mask_generated_after = ReadTexture(mask_generated_texture, 4);
      const std::size_t input_mismatches =
          Mismatches(external_reset, external_reset_after) +
          Mismatches(external_generated, external_generated_after) +
          Mismatches(flow32, flow32_after) + Mismatches(flow16, flow16_after) +
          Mismatches(mask_generated, mask_generated_after);
      const std::size_t handoff_mismatches =
          Mismatches(external_reset, checkpoint10_reset) +
          Mismatches(external_generated, checkpoint10_generated);
      std::size_t canary_mismatches =
          Mismatches(rgba_canary, ReadTexture(canary_rgba, 8)) +
          Mismatches(rg_canary, ReadTexture(canary_rg, 4)) +
          Mismatches(r32_canary, ReadTexture(canary_r32, 4)) +
          Mismatches(r8_canary, ReadTexture(canary_r8, 1));
      const auto *rom_bytes =
          static_cast<const std::uint8_t *>(rom_buffer.contents);
      for (std::size_t index = 0; index < kRomGuardBytes; ++index) {
        canary_mismatches += rom_bytes[index] != (poison ^ 0xffu);
        canary_mismatches +=
            rom_bytes[kRomGuardBytes + kRomBytes + index] != (poison ^ 0xffu);
      }
      std::vector<std::uint8_t> rom_after(
          rom_bytes + kRomGuardBytes, rom_bytes + kRomGuardBytes + kRomBytes);
      const std::size_t rom_mismatches = Mismatches(rom, rom_after);

      const std::array<
          std::pair<const char *, const std::vector<std::uint8_t> *>, 10>
          checkpoints = {
              {{"reset-m10.rgba16f", &checkpoint10_reset},
               {"reset-m9-a.rg16f", &checkpoint9_reset_a},
               {"reset-m9-b.rg16f", &checkpoint9_reset_b},
               {"reset-m9-mask.r32u", &checkpoint9_reset_mask},
               {"reset-m25.r8", &checkpoint25_reset},
               {"generated-m10.rgba16f", &checkpoint10_generated},
               {"generated-m9-a.rg16f", &checkpoint9_generated_a},
               {"generated-m9-b.rg16f", &checkpoint9_generated_b},
               {"generated-m9-mask.r32u", &checkpoint9_generated_mask},
               {"generated-m25.r8", &checkpoint25_generated}}};
      for (const auto &[name, bytes] : checkpoints)
        Write(output_root / name, *bytes);
      Write(output_root / "reset-m25.coverage", reset_coverage);
      Write(output_root / "generated-m25.coverage", generated_coverage);
      const bool pass = input_mismatches == 0 && handoff_mismatches == 0 &&
                        canary_mismatches == 0 && rom_mismatches == 0;
      std::ofstream receipt(output_root / "run-receipt.json");
      receipt
          << "{\n\"schema\":\"neuralstudio.fg.distortion-six-launch-run.v1\","
             "\n\"backend\":\"mac-metal\",\n\"run_id\":\""
          << run_id << "\",\n\"process_id\":" << getpid()
          << ",\n\"poison_byte\":" << unsigned(poison)
          << ",\n\"module_sha256\":[\"" << fg_distortion_module10::kModuleSha256
          << "\",\"" << fg_distortion_module9::kModuleSha256
          << "\","
             "\"28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e82"
             "4\"],"
             "\n\"launch_sequence\":[10,9,25,10,9,25],"
             "\n\"selected_launch_tokens\":[\"launch-000005\",\"launch-"
             "000009\",\"launch-000010\",\"launch-000016\",\"launch-000020\","
             "\"launch-000021\"],"
             "\n\"selected_raw_parameter_sha256\":["
             "\"e170b309e44af03949713aa49e319aa27448cc33a002f7a415006d2b3acbfa7"
             "1\","
             "\"b695e0e6d29050462620ad1f54b593dd57b12a616cb58e2367ffef10a2444fe"
             "2\","
             "\"98f06561b037fdfab9742649c3a82e58c39de0038b5df10f638ede5df5aa6b5"
             "5\","
             "\"9c50edeea15f0b497785e9cb6a7e8ca9324992df32f12ce3bba553f55199b4a"
             "5\","
             "\"723e2c4a348298753ace7172a6e1452e22faf4518c10952ab7857f7eaf8406c"
             "3\","
             "\"a765a6a6e229ab59eb65faf1393caf9ccff44d949b23f9af11ad0d957f77ce4"
             "c\"],"
             "\n\"input_generator\":\""
          << connected::kInputGenerator
          << "\",\n\"immutable_input_pre_sha256\":[\"" << Hash(external_reset)
          << "\",\"" << Hash(external_generated) << "\",\"" << Hash(flow32)
          << "\",\"" << Hash(flow16) << "\",\"" << Hash(mask_generated)
          << "\"],\n\"immutable_input_post_sha256\":[\""
          << Hash(external_reset_after) << "\",\""
          << Hash(external_generated_after) << "\",\"" << Hash(flow32_after)
          << "\",\"" << Hash(flow16_after) << "\",\""
          << Hash(mask_generated_after)
          << "\"],\n\"initial_temporal_state_sha256\":[\""
          << Hash(previous_generated) << "\",\"" << Hash(mask_reset)
          << "\"],\n\"checkpoint_sha256\":[";
      for (std::size_t index = 0; index < checkpoints.size(); ++index) {
        if (index)
          receipt << ',';
        receipt << '\"' << Hash(*checkpoints[index].second) << '\"';
      }
      receipt << "],\n\"checkpoint_bytes\":[460800,230400,230400,230400,57600,"
                 "460800,230400,230400,230400,57600],"
                 "\n\"reset_coverage_sha256\":\""
              << Hash(reset_coverage) << "\",\n\"generated_coverage_sha256\":\""
              << Hash(generated_coverage)
              << "\",\n\"reset_written_pixels\":" << CountOnes(reset_coverage)
              << ",\n\"generated_written_pixels\":"
              << CountOnes(generated_coverage)
              << ",\n\"input_mismatched_bytes\":" << input_mismatches
              << ",\n\"handoff_mismatched_bytes\":" << handoff_mismatches
              << ",\n\"rom_mismatched_bytes\":" << rom_mismatches
              << ",\n\"canary_mismatched_bytes\":" << canary_mismatches
              << ",\n\"full_output_poison_pixels\":"
              << (CountPoisonPixels(checkpoint10_reset, 8, poison) +
                  CountPoisonPixels(checkpoint9_reset_a, 4, poison) +
                  CountPoisonPixels(checkpoint9_reset_b, 4, poison) +
                  CountPoisonPixels(checkpoint9_reset_mask, 4, poison) +
                  CountPoisonPixels(checkpoint10_generated, 8, poison) +
                  CountPoisonPixels(checkpoint9_generated_a, 4, poison) +
                  CountPoisonPixels(checkpoint9_generated_b, 4, poison) +
                  CountPoisonPixels(checkpoint9_generated_mask, 4, poison))
              << ",\n\"architecture_rom_sha256\":\"" << kRomSha256
              << "\",\n\"live_device_handoffs\":true,"
                 "\n\"host_roundtrip_used_as_launch_input\":false,"
                 "\n\"declared_region_coverage_accounted\":true,"
                 "\n\"expected_output_fed_to_execution\":false,"
                 "\n\"captured_or_archived_intermediate_consumed\":false,"
                 "\n\"address_fields_recorded\":false,\n\"result\":\""
              << (pass ? "pass" : "fail") << "\"\n}\n";
      return pass ? 0 : 3;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
