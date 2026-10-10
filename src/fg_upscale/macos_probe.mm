#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kGenerator[] = "splitmix64-fg-upscale-signed-spatial-v3";

struct ModuleIdentity {
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
};

constexpr std::array<ModuleIdentity, 4> kModuleIdentities = {{
    {36, 2296,
     "d6962e058c46f6de9c6c5f8c75dafde64a34f094a49f1719af16c505f3b530fd"},
    {40, 2296,
     "070c7f4987251c28bcad097ac8a00e7b1e2644dae4267ae37b676e280478a2b8"},
    {45, 2296,
     "547a05ab275edcb0d413e4b7f89bada85f08035fa0517cd5267960fe695498fc"},
    {49, 2296,
     "2b91bf740602f074acbaa5efba1cd1115c5ee52653a96d087c0d52231de2a74a"},
}};

struct Shape {
  std::uint32_t input_height;
  std::uint32_t input_width;
  std::uint32_t output_height;
  std::uint32_t output_width;
  std::uint32_t channels;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t grid_z;
  std::uint32_t block_x;
  std::uint32_t block_y;
  std::uint32_t block_z;
  std::uint32_t live_launch_token;
  std::uint32_t trace_order;
  std::size_t module_index;
};

constexpr std::array<Shape, 5> kShapes = {{
    {3, 5, 6, 10, 256, 1, 2, 2, 32, 5, 5, 32, 30, 0},
    {6, 10, 12, 20, 128, 1, 3, 2, 16, 8, 8, 36, 34, 1},
    {12, 20, 24, 40, 128, 1, 5, 3, 16, 8, 8, 40, 38, 1},
    {24, 40, 48, 80, 64, 1, 8, 5, 8, 11, 11, 44, 42, 2},
    {48, 80, 96, 160, 32, 1, 10, 6, 4, 16, 16, 48, 46, 3},
}};

struct UpscaleParameters {
  std::uint32_t input_height;
  std::uint32_t input_width;
  std::uint32_t output_height;
  std::uint32_t output_width;
  std::uint32_t physical_output_height;
  std::uint32_t physical_output_width;
  std::uint32_t channels;
};
static_assert(sizeof(UpscaleParameters) == 28);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t InputHalfBits(std::size_t shape_index, std::size_t index,
                            std::uint32_t channels) {
  const std::uint32_t channel =
      static_cast<std::uint32_t>(index % channels);
  const std::uint64_t channel_code = SplitMix64(
      0xA0761D6478BD642Full ^
      (static_cast<std::uint64_t>(shape_index + 1u) *
       0xE7037ED1A0B428DBull) ^
      (static_cast<std::uint64_t>(channel) * 0x8EBC6AF09C88C6E3ull));
  if ((channel_code & 31u) == 0u) return 0x0000u;
  if ((channel_code & 31u) == 1u) return 0x8000u;
  const std::uint64_t mixed = SplitMix64(
      0xD6E8FEB86659FD93ull ^
      (static_cast<std::uint64_t>(shape_index + 1u) *
       0x9E3779B97F4A7C15ull) ^
      (static_cast<std::uint64_t>(index) * 0x94D049BB133111EBull));
  const std::uint16_t sign =
      (mixed >> 63u) != 0u ? 0x8000u : 0x0000u;
  switch (mixed & 15u) {
    case 0:
      return static_cast<std::uint16_t>(sign | 0x3400u);
    case 1:
      return static_cast<std::uint16_t>(sign | 0x3800u);
    case 2:
      return static_cast<std::uint16_t>(sign | 0x3C00u);
    case 3:
      return static_cast<std::uint16_t>(sign | 0x4000u);
    case 4:
      return static_cast<std::uint16_t>(sign | 0x3555u);
    case 5:
      return static_cast<std::uint16_t>(sign | 0x3BFFu);
    case 6:
      return static_cast<std::uint16_t>(sign | 0x3C01u);
    case 7:
      return static_cast<std::uint16_t>(sign | 0x3AAAu);
    default: {
      const std::uint16_t exponent =
          static_cast<std::uint16_t>(13u + ((mixed >> 12u) & 3u));
      return static_cast<std::uint16_t>(
          sign | (exponent << 10u) | (mixed & 0x03FFu));
    }
  }
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::vector<std::uint8_t> GenerateInput(std::size_t shape_index,
                                        const Shape& shape) {
  const std::size_t values = static_cast<std::size_t>(shape.input_height) *
                             shape.input_width * shape.channels;
  std::vector<std::uint8_t> result(values * 2u);
  for (std::size_t index = 0; index < values; ++index) {
    Store16(&result, index * 2u,
            InputHalfBits(shape_index, index, shape.channels));
  }
  return result;
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
  for (const unsigned char byte : digest) {
    text << std::setw(2) << static_cast<unsigned>(byte);
  }
  return text.str();
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>* payload,
                      std::size_t payload_bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:payload_bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
  auto* contents = static_cast<std::uint8_t*>(buffer.contents);
  std::memset(contents, kGuardValue, payload_bytes + 2u * kGuardBytes);
  if (payload != nullptr) {
    if (payload->size() != payload_bytes) {
      throw std::runtime_error("guarded payload size mismatch");
    }
    std::memcpy(contents + kGuardBytes, payload->data(), payload_bytes);
  } else {
    std::memset(contents + kGuardBytes, fill, payload_bytes);
  }
  return buffer;
}

std::vector<std::uint8_t> ReadAll(id<MTLBuffer> buffer) {
  const auto* contents = static_cast<const std::uint8_t*>(buffer.contents);
  return std::vector<std::uint8_t>(contents, contents + buffer.length);
}

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t payload_bytes) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes +
                                       payload_bytes);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t payload_bytes) {
  std::size_t result = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    result += guarded[index] != kGuardValue;
    result += guarded[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return result;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("mismatch comparison size differs");
  }
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] != right[index];
  }
  return result;
}

std::size_t PoisonMatchingHalfValues(const std::vector<std::uint8_t>& output,
                                     std::uint8_t poison) {
  if ((output.size() & 1u) != 0u) {
    throw std::runtime_error("output byte count is not even");
  }
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2u) {
    result += output[offset] == poison && output[offset + 1u] == poison;
  }
  return result;
}

struct LevelResult {
  Shape shape{};
  std::vector<std::uint8_t> input;
  std::vector<std::uint8_t> actual_input;
  std::vector<std::uint8_t> output;
  std::size_t input_mismatches = 0;
  std::size_t canary_mismatches = 0;
  std::size_t poison_matching_half_values = 0;
};

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& device_name,
                  const std::vector<LevelResult>& levels) {
  std::vector<std::uint8_t> output_root;
  std::size_t input_mismatches = 0;
  std::size_t canary_mismatches = 0;
  std::size_t poison_matching_half_values = 0;
  for (const auto& level : levels) {
    output_root.insert(output_root.end(), level.output.begin(),
                       level.output.end());
    input_mismatches += level.input_mismatches;
    canary_mismatches += level.canary_mismatches;
    poison_matching_half_values += level.poison_matching_half_values;
  }
  const bool passed = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.upscale-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"function_name\": \"k_upscale\",\n"
         << "  \"reconstruction_unit\": \"k_upscale-five-launch-family\",\n"
         << "  \"metal_function\": \"fg_upscale\",\n"
         << "  \"input_generator\": \"" << kGenerator << "\",\n"
         << "  \"numerical_domain\": \"signed-finite-fp16-normal-or-signed-zero-spatially-mixed-below-four\",\n"
         << "  \"external_tensor_inputs_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"levels\": [\n";
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const auto& level = levels[index];
    const auto& shape = level.shape;
    const auto& module = kModuleIdentities[shape.module_index];
    stream << "    {\n"
           << "      \"index\": " << index << ",\n"
           << "      \"live_launch_token\": " << shape.live_launch_token
           << ",\n"
           << "      \"trace_order\": " << shape.trace_order << ",\n"
           << "      \"source_module_token\": " << module.token << ",\n"
           << "      \"source_module_bytes\": " << module.bytes << ",\n"
           << "      \"source_module_sha256\": \"" << module.sha256
           << "\",\n"
           << "      \"input_height\": " << shape.input_height << ",\n"
           << "      \"input_width\": " << shape.input_width << ",\n"
           << "      \"output_height\": " << shape.output_height << ",\n"
           << "      \"output_width\": " << shape.output_width << ",\n"
           << "      \"physical_output_height\": " << shape.output_height
           << ",\n"
           << "      \"physical_output_width\": " << shape.output_width
           << ",\n"
           << "      \"channels\": " << shape.channels << ",\n"
           << "      \"live_grid\": [" << shape.grid_x << ", "
           << shape.grid_y << ", " << shape.grid_z << "],\n"
           << "      \"live_block\": [" << shape.block_x << ", "
           << shape.block_y << ", " << shape.block_z << "],\n"
           << "      \"metal_threads\": [" << shape.channels / 2u << ", "
           << shape.output_width << ", " << shape.output_height << "],\n"
           << "      \"input_bytes\": " << level.input.size() << ",\n"
           << "      \"output_bytes\": " << level.output.size() << ",\n"
           << "      \"input_pre_sha256\": \"" << Sha256(level.input)
           << "\",\n"
           << "      \"input_post_sha256\": \""
           << Sha256(level.actual_input) << "\",\n"
           << "      \"output_sha256\": \"" << Sha256(level.output)
           << "\",\n"
           << "      \"input_mismatched_bytes\": "
           << level.input_mismatches << ",\n"
           << "      \"canary_mismatched_bytes\": "
           << level.canary_mismatches << ",\n"
           << "      \"poison_matching_half_values\": "
           << level.poison_matching_half_values << "\n"
           << "    }" << (index + 1u == levels.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"output_root_sha256\": \"" << Sha256(output_root)
         << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_matching_half_values\": "
         << poison_matching_half_values
         << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"single_poison_zero_matching_values\": "
         << (poison_matching_half_values == 0 ? "true" : "false") << ",\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-upscale-mac <metallib> <output-dir> <run-id> "
                   "<poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path output_directory = argv[2];
      const std::string run_id = argv[3];
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[4], &poison_end, 16);
      if (poison_end == argv[4] || *poison_end != '\0' ||
          parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      std::filesystem::create_directories(output_directory);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      NSString* library_path = [NSString stringWithUTF8String:argv[1]];
      id<MTLLibrary> library =
          [device newLibraryWithURL:[NSURL fileURLWithPath:library_path]
                              error:&error];
      if (library == nil) {
        throw std::runtime_error(
            std::string("metallib load failed: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLFunction> function = [library newFunctionWithName:@"fg_upscale"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) {
        throw std::runtime_error(
            std::string("pipeline creation failed: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue creation failed");

      std::vector<LevelResult> levels;
      levels.reserve(kShapes.size());
      for (std::size_t index = 0; index < kShapes.size(); ++index) {
        const Shape& shape = kShapes[index];
        LevelResult level;
        level.shape = shape;
        level.input = GenerateInput(index, shape);
        const std::size_t output_bytes =
            static_cast<std::size_t>(shape.output_height) *
            shape.output_width * shape.channels * 2u;
        id<MTLBuffer> input =
            Guarded(device, &level.input, level.input.size(), 0u);
        id<MTLBuffer> output =
            Guarded(device, nullptr, output_bytes, poison);
        const UpscaleParameters parameters = {
            shape.input_height, shape.input_width, shape.output_height,
            shape.output_width, shape.output_height, shape.output_width,
            shape.channels};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:kGuardBytes atIndex:0];
        [encoder setBuffer:output offset:kGuardBytes atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
        const MTLSize threads = MTLSizeMake(shape.channels / 2u,
                                            shape.output_width,
                                            shape.output_height);
        const MTLSize group = MTLSizeMake(16, 4, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:group];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
          throw std::runtime_error(
              std::string("Metal command failed: ") +
              (command.error != nil
                   ? command.error.localizedDescription.UTF8String
                   : "unknown"));
        }

        const auto input_all = ReadAll(input);
        const auto output_all = ReadAll(output);
        level.actual_input = Payload(input_all, level.input.size());
        level.output = Payload(output_all, output_bytes);
        level.input_mismatches =
            Mismatches(level.input, level.actual_input);
        level.canary_mismatches =
            CanaryMismatches(input_all, level.input.size()) +
            CanaryMismatches(output_all, output_bytes);
        level.poison_matching_half_values =
            PoisonMatchingHalfValues(level.output, poison);
        Write(output_directory /
                  ("input-level-" + std::to_string(index) + "-after.bin"),
              level.actual_input);
        Write(output_directory /
                  ("output-level-" + std::to_string(index) + ".bin"),
              level.output);
        levels.push_back(std::move(level));
      }

      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   device.name.UTF8String, levels);
      std::cout << "result=pass\n";
      return 0;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << "\n";
      return 1;
    }
  }
}
