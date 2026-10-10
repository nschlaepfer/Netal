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

struct Shape {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t live_launch_token;
  std::uint32_t trace_order;
};

constexpr std::array<Shape, 5> kShapes = {{
    {160, 96, 80, 48, 32, 20, 12, 21, 19},
    {80, 48, 40, 24, 64, 10, 6, 23, 21},
    {40, 24, 20, 12, 128, 5, 3, 25, 23},
    {20, 12, 10, 6, 128, 3, 2, 27, 25},
    {10, 6, 5, 3, 256, 2, 1, 29, 27},
}};

struct PoolingParameters {
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
};
static_assert(sizeof(PoolingParameters) == 20);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t FiniteNormal(std::uint64_t value) {
  const std::uint16_t sign =
      static_cast<std::uint16_t>((value >> 48u) & 0x8000u);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(1u + ((value >> 52u) % 30u));
  std::uint16_t bits = static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>(exponent << 10u) |
      static_cast<std::uint16_t>(value & 0x03FFu));
  if (bits == 0x5A5Au || bits == 0xA5A5u) bits = 0x3C00u;
  return bits;
}

std::uint16_t InputHalfBits(std::size_t shape_index, const Shape& shape,
                            std::uint32_t y, std::uint32_t x,
                            std::uint32_t channel) {
  const std::uint64_t cell =
      ((static_cast<std::uint64_t>(y / 2u) * shape.output_width + x / 2u) *
           shape.channels +
       channel);
  const std::uint32_t lane = (y & 1u) * 2u + (x & 1u);
  const std::uint64_t base = SplitMix64(
      0xD1B54A32D192ED03ull ^
      (static_cast<std::uint64_t>(shape_index + 1u) *
       0x9E3779B97F4A7C15ull) ^
      (cell * 0x94D049BB133111EBull));
  switch (base & 31u) {
    case 0:
      return 0x0000u;
    case 1:
      return 0x8000u;
    case 2:
      return (lane & 1u) != 0u ? 0x8000u : 0x0000u;
    case 3:
      return FiniteNormal(base);
    case 4: {
      constexpr std::array<std::uint16_t, 4> values = {
          0xBC00u, 0x0000u, 0x3C00u, 0x4000u};
      return values[lane];
    }
    case 5: {
      constexpr std::array<std::uint16_t, 4> values = {
          0x0001u, 0x8001u, 0x03FFu, 0x83FFu};
      return values[lane];
    }
    case 6: {
      constexpr std::array<std::uint16_t, 4> values = {
          0x0400u, 0x8400u, 0x7BFFu, 0xFBFFu};
      return values[lane];
    }
    default:
      return FiniteNormal(SplitMix64(
          base ^ (static_cast<std::uint64_t>(lane + 1u) *
                  0xBF58476D1CE4E5B9ull)));
  }
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::vector<std::uint8_t> GenerateInput(std::size_t shape_index,
                                        const Shape& shape) {
  const std::size_t values = static_cast<std::size_t>(shape.input_width) *
                             shape.input_height * shape.channels;
  std::vector<std::uint8_t> bytes(values * 2u);
  for (std::uint32_t y = 0; y < shape.input_height; ++y) {
    for (std::uint32_t x = 0; x < shape.input_width; ++x) {
      for (std::uint32_t channel = 0; channel < shape.channels; ++channel) {
        const std::size_t index =
            (static_cast<std::size_t>(y) * shape.input_width + x) *
                shape.channels +
            channel;
        Store16(&bytes, index * 2u,
                InputHalfBits(shape_index, shape, y, x, channel));
      }
    }
  }
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
                                   guarded.begin() + kGuardBytes + payload_bytes);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& guarded,
                             std::size_t payload_bytes) {
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += guarded[index] != kGuardValue;
    mismatches += guarded[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return mismatches;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("mismatch comparison size differs");
  }
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    mismatches += left[index] != right[index];
  }
  return mismatches;
}

std::size_t UntouchedValues(const std::vector<std::uint8_t>& output,
                            std::uint8_t poison) {
  if ((output.size() & 1u) != 0u) {
    throw std::runtime_error("pooling output byte count is not even");
  }
  std::size_t untouched = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2u) {
    untouched += output[offset] == poison && output[offset + 1u] == poison;
  }
  return untouched;
}

struct LevelResult {
  Shape shape{};
  std::vector<std::uint8_t> input;
  std::vector<std::uint8_t> actual_input;
  std::vector<std::uint8_t> output;
  std::size_t input_mismatches = 0;
  std::size_t canary_mismatches = 0;
  std::size_t untouched_values = 0;
};

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& device_name,
                  const std::vector<LevelResult>& levels) {
  std::vector<std::uint8_t> output_root_bytes;
  std::size_t total_input_mismatches = 0;
  std::size_t total_canary_mismatches = 0;
  std::size_t total_untouched_values = 0;
  for (const auto& level : levels) {
    output_root_bytes.insert(output_root_bytes.end(), level.output.begin(),
                             level.output.end());
    total_input_mismatches += level.input_mismatches;
    total_canary_mismatches += level.canary_mismatches;
    total_untouched_values += level.untouched_values;
  }
  const bool passed = total_input_mismatches == 0 &&
                      total_canary_mismatches == 0 &&
                      total_untouched_values == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.pooling-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"reconstruction_unit\": \"k_pooling-five-launch-family\",\n"
         << "  \"metal_function\": \"fg_pooling\",\n"
         << "  \"input_generator\": \"splitmix64-fg-pooling-finite-v1\",\n"
         << "  \"external_tensor_inputs_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"levels\": [\n";
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const auto& level = levels[index];
    const auto& shape = level.shape;
    stream << "    {\n"
           << "      \"index\": " << index << ",\n"
           << "      \"live_launch_token\": " << shape.live_launch_token << ",\n"
           << "      \"trace_order\": " << shape.trace_order << ",\n"
           << "      \"input_width\": " << shape.input_width << ",\n"
           << "      \"input_height\": " << shape.input_height << ",\n"
           << "      \"output_width\": " << shape.output_width << ",\n"
           << "      \"output_height\": " << shape.output_height << ",\n"
           << "      \"channels\": " << shape.channels << ",\n"
           << "      \"grid\": [" << shape.grid_x << ", " << shape.grid_y
           << ", 1],\n"
           << "      \"block\": [4, 4, 16],\n"
           << "      \"dynamic_shared_bytes\": 0,\n"
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
           << "      \"untouched_half_values\": "
           << level.untouched_values << "\n"
           << "    }" << (index + 1u == levels.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"output_root_sha256\": \"" << Sha256(output_root_bytes)
         << "\",\n"
         << "  \"input_mismatched_bytes\": " << total_input_mismatches
         << ",\n"
         << "  \"canary_mismatched_bytes\": " << total_canary_mismatches
         << ",\n"
         << "  \"untouched_half_values\": " << total_untouched_values
         << ",\n"
         << "  \"input_immutability\": "
         << (total_input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"full_write_coverage\": "
         << (total_untouched_values == 0 ? "true" : "false") << ",\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-pooling-mac <metallib> <output-dir> <run-id> <poison-hex>\n";
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
      id<MTLFunction> function = [library newFunctionWithName:@"fg_pooling"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("command queue creation failed");

      std::vector<LevelResult> levels;
      levels.reserve(kShapes.size());
      for (std::size_t index = 0; index < kShapes.size(); ++index) {
        const Shape shape = kShapes[index];
        LevelResult result;
        result.shape = shape;
        result.input = GenerateInput(index, shape);
        const std::size_t output_bytes =
            static_cast<std::size_t>(shape.output_width) *
            shape.output_height * shape.channels * 2u;
        id<MTLBuffer> input = Guarded(device, &result.input,
                                      result.input.size(), 0);
        id<MTLBuffer> output = Guarded(device, nullptr, output_bytes, poison);
        const PoolingParameters parameters = {
            shape.input_width, shape.input_height, shape.output_width,
            shape.output_height, shape.channels};
        id<MTLBuffer> parameter_buffer =
            [device newBufferWithBytes:&parameters
                                length:sizeof(parameters)
                               options:MTLResourceStorageModeShared];
        if (parameter_buffer == nil) {
          throw std::runtime_error("parameter buffer allocation failed");
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:output offset:kGuardBytes atIndex:0];
        [encoder setBuffer:input offset:kGuardBytes atIndex:1];
        [encoder setBuffer:parameter_buffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(shape.grid_x, shape.grid_y, 1)
                   threadsPerThreadgroup:MTLSizeMake(4, 4, 16)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
          throw std::runtime_error(
              std::string("Metal dispatch failed: ") +
              (command.error != nil
                   ? command.error.localizedDescription.UTF8String
                   : "unknown"));
        }

        const auto guarded_input = ReadAll(input);
        const auto guarded_output = ReadAll(output);
        result.actual_input = Payload(guarded_input, result.input.size());
        result.output = Payload(guarded_output, output_bytes);
        result.input_mismatches = Mismatches(result.actual_input, result.input);
        result.canary_mismatches =
            CanaryMismatches(guarded_input, result.input.size()) +
            CanaryMismatches(guarded_output, output_bytes);
        result.untouched_values = UntouchedValues(result.output, poison);
        Write(output_directory /
                  ("level-" + std::to_string(index) + "-input-after.bin"),
              result.actual_input);
        Write(output_directory /
                  ("level-" + std::to_string(index) + "-output.bin"),
              result.output);
        levels.push_back(std::move(result));
      }

      const std::string device_name = device.name.UTF8String;
      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   device_name, levels);
      std::size_t failures = 0;
      for (const auto& level : levels) {
        failures += level.input_mismatches + level.canary_mismatches +
                    level.untouched_values;
      }
      std::cout << "probe=fg_pooling_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "device_name=" << device_name << '\n'
                << "level_count=" << levels.size() << '\n'
                << "result=" << (failures == 0 ? "pass" : "fail") << '\n';
      return failures == 0 ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
