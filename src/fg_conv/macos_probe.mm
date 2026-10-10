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

constexpr std::uint32_t kWidth = 160;
constexpr std::uint32_t kHeight = 96;
constexpr std::uint32_t kInputChannels = 48;
constexpr std::uint32_t kOutputChannels = 32;
constexpr std::size_t kInputBytes =
    std::size_t{kWidth} * kHeight * kInputChannels * 2;
constexpr std::size_t kOutputBytes =
    std::size_t{kWidth} * kHeight * kOutputChannels * 2;
constexpr std::size_t kWeightBytes = 56320;
constexpr std::size_t kDenseWeightBytes = 27648;
constexpr std::size_t kUnusedWeightBytes = 27648;
constexpr std::size_t kBiasOffset = 55296;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;

struct ConvL20Parameters {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t padded_width;
  std::uint32_t padded_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
};
static_assert(sizeof(ConvL20Parameters) == 24);

enum class Scenario { kSpatialMap, kOneHot, kBias };

std::string ScenarioName(Scenario scenario) {
  switch (scenario) {
    case Scenario::kSpatialMap:
      return "spatial-map";
    case Scenario::kOneHot:
      return "one-hot";
    case Scenario::kBias:
      return "bias";
  }
  throw std::runtime_error("unreachable scenario");
}

Scenario ParseScenario(const std::string& name) {
  if (name == "spatial-map") return Scenario::kSpatialMap;
  if (name == "one-hot") return Scenario::kOneHot;
  if (name == "bias") return Scenario::kBias;
  throw std::runtime_error("unknown scenario: " + name);
}

void StoreHalf(std::vector<std::uint8_t>* bytes, std::size_t offset,
               std::uint16_t value) {
  if (offset + 2 > bytes->size()) throw std::runtime_error("half store OOB");
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
}

void SetInput(std::vector<std::uint8_t>* input, std::uint32_t y,
              std::uint32_t x, std::uint32_t channel,
              std::uint16_t value) {
  const std::size_t element =
      (std::size_t{y} * kWidth + x) * kInputChannels + channel;
  StoreHalf(input, element * 2, value);
}

void SetPackedWeight(std::vector<std::uint8_t>* weights,
                     std::uint32_t raw_spatial_tile,
                     std::uint32_t input_channel,
                     std::uint32_t output_channel,
                     std::uint16_t value) {
  const std::uint32_t output_group = output_channel / 8;
  const std::uint32_t n = output_channel % 8;
  const std::uint32_t channel_stage = input_channel / 16;
  const std::uint32_t k = input_channel % 16;
  const std::uint32_t thread_in_group = (k % 8) / 2;
  const std::uint32_t fragment_half = (k >= 8 ? 2 : 0) + (k & 1u);
  const std::uint32_t lane = n * 4 + thread_in_group;
  const std::size_t offset = std::size_t{output_group} * 6912 +
                             std::size_t{raw_spatial_tile} * 768 +
                             std::size_t{channel_stage} * 256 +
                             std::size_t{lane} * 8 +
                             std::size_t{fragment_half} * 2;
  if (offset >= kDenseWeightBytes) {
    throw std::runtime_error("packed weight landed outside active region");
  }
  StoreHalf(weights, offset, value);
}

void SetPackedBias(std::vector<std::uint8_t>* weights,
                   std::uint32_t output_channel, std::uint16_t value) {
  const std::uint32_t group = output_channel / 8;
  const std::uint32_t lane = output_channel % 8;
  for (std::uint32_t repeat = 0; repeat < 16; ++repeat) {
    const std::size_t offset = kBiasOffset + std::size_t{group} * 256 +
                               std::size_t{repeat * 8 + lane} * 2;
    StoreHalf(weights, offset, value);
  }
}

struct Generated {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint8_t> input;
};

Generated Generate(Scenario scenario, std::uint8_t poison) {
  Generated generated{std::vector<std::uint8_t>(kWeightBytes, 0),
                      std::vector<std::uint8_t>(kInputBytes, 0)};
  std::memset(generated.weights.data() + kDenseWeightBytes, poison,
              kUnusedWeightBytes);
  if (scenario == Scenario::kSpatialMap) {
    SetInput(&generated.input, 48, 80, 0, 0x3c00);
    for (std::uint32_t tile = 0; tile < 9; ++tile) {
      SetPackedWeight(&generated.weights, tile, 0, tile, 0x3c00);
    }
  } else if (scenario == Scenario::kOneHot) {
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        for (std::uint32_t channel = 0; channel < kInputChannels; ++channel) {
          const std::uint16_t bits = static_cast<std::uint16_t>(
              0x3c00u + ((x + 3 * y + 5 * channel) & 7u) * 0x80u);
          SetInput(&generated.input, y, x, channel, bits);
        }
      }
    }
    for (std::uint32_t output_channel = 0;
         output_channel < kOutputChannels; ++output_channel) {
      const std::uint32_t tile = output_channel % 9;
      const std::uint32_t input_channel =
          (output_channel * 13 + 7) % kInputChannels;
      const std::uint16_t weight =
          (output_channel & 1u) == 0 ? 0x3c00 : 0xbc00;
      SetPackedWeight(&generated.weights, tile, input_channel,
                      output_channel, weight);
    }
  } else {
    for (std::uint32_t output_channel = 0;
         output_channel < kOutputChannels; ++output_channel) {
      SetPackedBias(&generated.weights, output_channel,
                    (output_channel & 1u) == 0 ? 0x3800 : 0xb800);
    }
  }
  return generated;
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (unsigned char byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write " + path.string());
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>* source,
                      std::size_t payload_bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:payload_bytes + 2 * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  auto* payload = static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes;
  std::memset(payload, fill, payload_bytes);
  if (source != nullptr) std::memcpy(payload, source->data(), payload_bytes);
  return buffer;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer, std::size_t size) {
  const auto* begin = static_cast<const std::uint8_t*>(buffer.contents) +
                      kGuardBytes;
  return std::vector<std::uint8_t>(begin, begin + size);
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += bytes[index] != kGuardValue;
    mismatches += bytes[kGuardBytes + size + index] != kGuardValue;
  }
  return mismatches;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) throw std::runtime_error("size mismatch");
  std::size_t count = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    count += left[index] != right[index];
  }
  return count;
}

std::size_t UntouchedHalfwords(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2) {
    count += output[offset] == poison && output[offset + 1] == poison;
  }
  return count;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, Scenario scenario,
                  std::uint8_t poison, const Generated& generated,
                  const std::vector<std::uint8_t>& actual_weights,
                  const std::vector<std::uint8_t>& actual_input,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t untouched_halfwords,
                  const std::string& device_name) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                    untouched_halfwords == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.conv-l20-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"scenario\": \"" << ScenarioName(scenario) << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"reconstruction_unit\": \"L20/k_conv_fp16_nhwc\",\n"
         << "  \"metal_function\": \"fg_conv_l20\",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_rtx_grid\": [10, 8, 1],\n"
         << "  \"selected_rtx_block\": [32, 4, 1],\n"
         << "  \"metal_dispatch_threads\": [160, 96, 32],\n"
         << "  \"metal_threads_per_threadgroup\": [8, 4, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"abi_scalars\": [160, 96, 162, 98, 160, 96],\n"
         << "  \"input_shape_nhwc\": [1, 96, 160, 48],\n"
         << "  \"output_shape_nhwc\": [1, 96, 160, 32],\n"
         << "  \"packed_weight_bytes\": 56320,\n"
         << "  \"active_weight_bytes\": 27648,\n"
         << "  \"poisoned_gap_offset\": 27648,\n"
         << "  \"poisoned_gap_bytes\": 27648,\n"
         << "  \"bias_offset\": 55296,\n"
         << "  \"generator\": \"fg-conv-l20-synthetic-v1\",\n"
         << "  \"external_data_files_opened\": false,\n"
         << "  \"weight_pre_sha256\": \"" << Sha256(generated.weights)
         << "\",\n"
         << "  \"weight_post_sha256\": \"" << Sha256(actual_weights)
         << "\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(generated.input)
         << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(actual_input)
         << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_halfwords\": " << untouched_halfwords << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

#ifndef FG_CONV_L20_NO_MAIN
int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 6) {
      std::cerr << "usage=fg-conv-l20-mac <metallib> <output-dir> <run-id> "
                   "<poison-hex> <scenario>\n";
      return 2;
    }
    try {
      const std::filesystem::path output_directory = argv[2];
      const std::string run_id = argv[3];
      char* poison_end = nullptr;
      const unsigned long parsed_poison =
          std::strtoul(argv[4], &poison_end, 16);
      if (poison_end == argv[4] || *poison_end != '\0' ||
          parsed_poison > 255) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const Scenario scenario = ParseScenario(argv[5]);
      std::filesystem::create_directories(output_directory);
      const Generated generated = Generate(scenario, poison);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_conv_l20"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");

      id<MTLBuffer> weights =
          Guarded(device, &generated.weights, kWeightBytes, 0);
      id<MTLBuffer> input = Guarded(device, &generated.input, kInputBytes, 0);
      id<MTLBuffer> output = Guarded(device, nullptr, kOutputBytes, poison);
      const ConvL20Parameters parameters{kWidth, kHeight, 162, 98,
                                         kWidth, kHeight};
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
      [encoder setBuffer:input offset:kGuardBytes atIndex:1];
      [encoder setBuffer:output offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, kOutputChannels)
          threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }

      const auto actual_weights = Payload(weights, kWeightBytes);
      const auto actual_input = Payload(input, kInputBytes);
      const auto actual_output = Payload(output, kOutputBytes);
      const std::size_t input_mismatches =
          Mismatches(actual_weights, generated.weights) +
          Mismatches(actual_input, generated.input);
      const std::size_t canary_mismatches =
          CanaryMismatches(weights, kWeightBytes) +
          CanaryMismatches(input, kInputBytes) +
          CanaryMismatches(output, kOutputBytes);
      const std::size_t untouched_halfwords =
          UntouchedHalfwords(actual_output, poison);
      Write(output_directory / "weights-after.bin", actual_weights);
      Write(output_directory / "input-after.bin", actual_input);
      Write(output_directory / "output.bin", actual_output);
      WriteReceipt(output_directory / "run-receipt.json", run_id, scenario,
                   poison, generated, actual_weights, actual_input,
                   actual_output, input_mismatches, canary_mismatches,
                   untouched_halfwords, device.name.UTF8String);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                        untouched_halfwords == 0;
      std::cout << "probe=fg_conv_l20_mac\n"
                << "scenario=" << ScenarioName(scenario) << '\n'
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "output_sha256=" << Sha256(actual_output) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "untouched_halfwords=" << untouched_halfwords << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
#endif
