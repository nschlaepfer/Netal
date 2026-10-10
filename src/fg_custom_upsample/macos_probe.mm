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
#include <vector>
#include <unistd.h>

namespace {

constexpr std::uint32_t kInputHeight = 184;
constexpr std::uint32_t kInputWidth = 320;
constexpr std::uint32_t kOutputHeight = 368;
constexpr std::uint32_t kOutputWidth = 640;
constexpr std::array<std::uint32_t, 3> kChannels = {4, 1, 3};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kSelectedModuleSha256[] =
    "1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6";
constexpr char kAddressFreeParameterSha256[] =
    "80189cac9cec275fc90e1a52f3e814475d063b67ed2fd80b05ae98ee1e9654f6";
constexpr char kRawCapturedParameterSha256[] =
    "2c38d2f149ec90e2e6cac2a843a824485c16421300fafcce34a1acaaeb8a25e4";
constexpr char kGenerator[] =
    "splitmix64-fg-custom-upsample-signed-spatial-v2";

struct CustomUpsamplePlaneParameters {
  std::uint32_t input_height;
  std::uint32_t input_width;
  std::uint32_t output_height;
  std::uint32_t output_width;
  std::uint32_t channels;
};
static_assert(sizeof(CustomUpsamplePlaneParameters) == 20);

std::size_t TensorBytes(std::uint32_t height, std::uint32_t width,
                        std::uint32_t channels) {
  return static_cast<std::size_t>(height) * width * channels * 2u;
}

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

std::uint16_t InputHalfBits(std::size_t plane, std::size_t index,
                            std::uint32_t channels) {
  const std::uint32_t channel = static_cast<std::uint32_t>(index % channels);
  const std::uint64_t mixed = SplitMix64(
      0xD1B54A32D192ED03ull ^
      (static_cast<std::uint64_t>(plane + 1u) * 0x94D049BB133111EBull) ^
      (static_cast<std::uint64_t>(channel) * 0x8EBC6AF09C88C6E3ull) ^
      (static_cast<std::uint64_t>(index) * 0x9E3779B97F4A7C15ull));
  const std::uint16_t sign =
      (mixed >> 63u) != 0u ? 0x8000u : 0x0000u;
  if ((mixed & 63u) == 0u) return sign;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(13u + ((mixed >> 12u) & 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03FFu));
}

std::vector<std::uint8_t> GenerateInput(std::size_t plane) {
  const std::uint32_t channels = kChannels[plane];
  const std::size_t values =
      static_cast<std::size_t>(kInputHeight) * kInputWidth * channels;
  std::vector<std::uint8_t> result(values * 2u);
  for (std::size_t index = 0; index < values; ++index) {
    Store16(&result, index * 2u,
            InputHalfBits(plane, index, channels));
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
  std::size_t result = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2u) {
    result += output[offset] == poison && output[offset + 1u] == poison;
  }
  return result;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::string& device_name,
    const std::array<std::vector<std::uint8_t>, 3>& expected_inputs,
    const std::array<std::vector<std::uint8_t>, 3>& actual_inputs,
    const std::array<std::vector<std::uint8_t>, 3>& outputs,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t poison_matching_half_values) {
  std::vector<std::uint8_t> combined_output;
  for (const auto& output : outputs) {
    combined_output.insert(combined_output.end(), output.begin(), output.end());
  }
  const bool passed = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.custom-upsample-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << static_cast<long long>(getpid()) << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"device\": \"" << device_name << "\",\n"
         << "  \"selected_module_token\": 60,\n"
         << "  \"selected_module_sha256\": \"" << kSelectedModuleSha256
         << "\",\n"
         << "  \"selected_function_name\": \"custom_upsample_hf_kernel\",\n"
         << "  \"selected_launch_token\": 70,\n"
         << "  \"selected_grid\": [40, 23, 3],\n"
         << "  \"selected_block\": [16, 16, 1],\n"
         << "  \"metal_dispatch_count\": 3,\n"
         << "  \"metal_dispatch_strategy\": \"one independent NHWC plane per dispatch\",\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"address_free_parameter_sha256\": \""
         << kAddressFreeParameterSha256 << "\",\n"
         << "  \"raw_captured_parameter_sha256\": \""
         << kRawCapturedParameterSha256 << "\",\n"
         << "  \"input_generator\": \"" << kGenerator << "\",\n"
         << "  \"planes\": [\n";
  for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
    stream << "    {\"plane\": " << plane << ", \"channels\": "
           << kChannels[plane]
           << ", \"input_height\": 184, \"input_width\": 320"
           << ", \"output_height\": 368, \"output_width\": 640"
           << ", \"input_bytes\": " << expected_inputs[plane].size()
           << ", \"output_bytes\": " << outputs[plane].size()
           << ", \"input_pre_sha256\": \""
           << Sha256(expected_inputs[plane])
           << "\", \"input_post_sha256\": \""
           << Sha256(actual_inputs[plane])
           << "\", \"output_sha256\": \"" << Sha256(outputs[plane])
           << "\"}" << (plane + 1u == kChannels.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"combined_output_sha256\": \"" << Sha256(combined_output)
         << "\",\n"
         << "  \"total_output_bytes\": " << combined_output.size() << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_matching_half_values\": "
         << poison_matching_half_values << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"single_poison_full_write_claimed\": false,\n"
         << "  \"independently_generated_inputs\": true,\n"
         << "  \"external_expected_tensors_consumed\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-custom-upsample-mac <metallib> <output-dir> "
                   "<run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path library_path = argv[1];
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

      std::array<std::vector<std::uint8_t>, 3> expected_inputs;
      for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
        expected_inputs[plane] = GenerateInput(plane);
      }

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("no Metal device");
      NSError* error = nil;
      NSURL* url = [NSURL fileURLWithPath:
          [NSString stringWithUTF8String:library_path.string().c_str()]];
      id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
      if (library == nil) {
        throw std::runtime_error(
            "unable to load metallib: " +
            std::string(error.localizedDescription.UTF8String ?: "unknown"));
      }
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_custom_upsample_plane"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) {
        throw std::runtime_error(
            "unable to create Metal pipeline: " +
            std::string(error.localizedDescription.UTF8String ?: "unknown"));
      }

      std::array<id<MTLBuffer>, 3> input_buffers;
      std::array<id<MTLBuffer>, 3> output_buffers;
      for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
        input_buffers[plane] = Guarded(
            device, &expected_inputs[plane], expected_inputs[plane].size(), 0);
        output_buffers[plane] = Guarded(
            device, nullptr,
            TensorBytes(kOutputHeight, kOutputWidth, kChannels[plane]), poison);
      }

      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("Metal command queue failed");
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      if (encoder == nil) throw std::runtime_error("Metal encoder failed");
      [encoder setComputePipelineState:pipeline];
      const MTLSize threads_per_group = MTLSizeMake(16, 16, 1);
      for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
        const CustomUpsamplePlaneParameters parameters = {
            kInputHeight, kInputWidth, kOutputHeight, kOutputWidth,
            kChannels[plane]};
        [encoder setBuffer:input_buffers[plane]
                    offset:kGuardBytes
                   atIndex:0];
        [encoder setBuffer:output_buffers[plane]
                    offset:kGuardBytes
                   atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(kOutputWidth, kOutputHeight,
                                             kChannels[plane])
              threadsPerThreadgroup:threads_per_group];
      }
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        throw std::runtime_error(
            "Metal command failed: " +
            std::string(command_buffer.error.localizedDescription.UTF8String ?:
                            "unknown"));
      }

      std::array<std::vector<std::uint8_t>, 3> actual_inputs;
      std::array<std::vector<std::uint8_t>, 3> actual_outputs;
      std::size_t input_mismatches = 0;
      std::size_t canary_mismatches = 0;
      std::size_t poison_matching_half_values = 0;
      for (std::size_t plane = 0; plane < kChannels.size(); ++plane) {
        const auto guarded_input = ReadAll(input_buffers[plane]);
        const auto guarded_output = ReadAll(output_buffers[plane]);
        actual_inputs[plane] =
            Payload(guarded_input, expected_inputs[plane].size());
        actual_outputs[plane] = Payload(
            guarded_output,
            TensorBytes(kOutputHeight, kOutputWidth, kChannels[plane]));
        input_mismatches +=
            Mismatches(expected_inputs[plane], actual_inputs[plane]);
        canary_mismatches +=
            CanaryMismatches(guarded_input, expected_inputs[plane].size()) +
            CanaryMismatches(guarded_output, actual_outputs[plane].size());
        poison_matching_half_values +=
            PoisonMatchingHalfValues(actual_outputs[plane], poison);
        Write(output_directory /
                  ("input-plane-" + std::to_string(plane) + "-after.bin"),
              actual_inputs[plane]);
        Write(output_directory /
                  ("output-plane-" + std::to_string(plane) + ".bin"),
              actual_outputs[plane]);
      }

      const char* device_utf8 = device.name.UTF8String;
      const std::string device_name =
          device_utf8 != nullptr ? device_utf8 : "unknown";
      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   device_name, expected_inputs, actual_inputs, actual_outputs,
                   input_mismatches, canary_mismatches,
                   poison_matching_half_values);

      std::vector<std::uint8_t> combined_output;
      for (const auto& output : actual_outputs) {
        combined_output.insert(combined_output.end(), output.begin(),
                               output.end());
      }
      const bool passed = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_custom_upsample_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << static_cast<long long>(getpid()) << '\n'
                << "device=" << device_name << '\n'
                << "combined_output_sha256=" << Sha256(combined_output) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "poison_matching_half_values="
                << poison_matching_half_values << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
