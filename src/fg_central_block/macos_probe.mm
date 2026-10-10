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

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 368;
constexpr std::size_t kPixels =
    static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight);
constexpr std::array<std::size_t, 6> kInputBytes = {
    kPixels * 8u, kPixels * 8u, kPixels * 2u,
    kPixels * 8u, kPixels * 2u, kPixels * 6u,
};
constexpr std::size_t kOutputBytes = kPixels * 36u;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kGenerator[] = "splitmix64-fg-central-block-signed-v1";

struct CentralBlockParameters {
  std::uint32_t width;
  std::uint32_t height;
  float factor;
};
static_assert(sizeof(CentralBlockParameters) == 12);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* output, std::size_t offset,
             std::uint16_t value) {
  (*output)[offset] = static_cast<std::uint8_t>(value);
  (*output)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t TensorHalfBits(std::size_t lane, std::size_t index) {
  static constexpr std::array<std::uint16_t, 16> kSpecial = {
      0x0000u, 0x8000u, 0x2C00u, 0xAC00u,
      0x3400u, 0xB400u, 0x3555u, 0xB555u,
      0x3800u, 0xB800u, 0x3A00u, 0xBA00u,
      0x3C00u, 0xBC00u, 0x3E00u, 0xBE00u,
  };
  const std::uint64_t mixed = SplitMix64(
      0xD6E8FEB86659FD93ull ^
      (static_cast<std::uint64_t>(lane + 1u) * 0xA0761D6478BD642Full) ^
      (static_cast<std::uint64_t>(index) * 0xE7037ED1A0B428DBull));
  if ((mixed & 7u) == 0u) return kSpecial[(mixed >> 8u) & 15u];
  const std::uint16_t sign =
      (mixed >> 63u) != 0u ? 0x8000u : 0x0000u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((mixed >> 12u) & 5u));
  return static_cast<std::uint16_t>(
      sign | (exponent << 10u) | ((mixed >> 19u) & 0x03FFu));
}

std::uint16_t MotionHalfBits(std::size_t pixel, std::size_t component) {
  static constexpr std::array<std::uint16_t, 16> kMotion = {
      0x0000u, 0x3400u, 0xB400u, 0x3800u,
      0xB800u, 0x3A00u, 0xBA00u, 0x3D00u,
      0xBD00u, 0x3E00u, 0xBE00u, 0x4080u,
      0xC080u, 0x3555u, 0xB555u, 0xBC00u,
  };
  const std::uint32_t x = static_cast<std::uint32_t>(pixel % kWidth);
  const std::uint32_t y = static_cast<std::uint32_t>(pixel / kWidth);
  if (component == 0u && (x == 0u || x == 1u)) return 0xBA00u;
  if (component == 1u && y == 0u) return 0xB555u;
  if (component == 1u && y == 1u) return 0xBE00u;
  if (component == 2u && x == 0u) return 0xBD00u;
  if (component == 3u && y == 0u) return 0xBA00u;
  if ((component == 0u || component == 2u) && x + 1u == kWidth) {
    return 0x3A00u;
  }
  if ((component == 1u || component == 3u) && y + 1u == kHeight) {
    return 0x3A00u;
  }
  const std::uint64_t mixed = SplitMix64(
      0x8EBC6AF09C88C6E3ull ^
      (static_cast<std::uint64_t>(pixel) * 0x589965CC75374CC3ull) ^
      (static_cast<std::uint64_t>(component + 1u) *
       0x1D8E4E27C47D124Full));
  return kMotion[mixed & 15u];
}

struct Inputs {
  std::array<std::vector<std::uint8_t>, 6> lanes;
};

Inputs GenerateInputs() {
  Inputs result;
  for (std::size_t lane = 0; lane < result.lanes.size(); ++lane) {
    result.lanes[lane].resize(kInputBytes[lane]);
  }
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    for (std::size_t component = 0; component < 4; ++component) {
      Store16(&result.lanes[0], (pixel * 4u + component) * 2u,
              TensorHalfBits(0, pixel * 4u + component));
      Store16(&result.lanes[1], (pixel * 4u + component) * 2u,
              TensorHalfBits(1, pixel * 4u + component));
      Store16(&result.lanes[3], (pixel * 4u + component) * 2u,
              MotionHalfBits(pixel, component));
    }
    Store16(&result.lanes[2], pixel * 2u, TensorHalfBits(2, pixel));
    Store16(&result.lanes[4], pixel * 2u, TensorHalfBits(4, pixel));
    for (std::size_t component = 0; component < 3; ++component) {
      Store16(&result.lanes[5], (pixel * 3u + component) * 2u,
              TensorHalfBits(5 + component, pixel * 3u + component));
    }
  }
  return result;
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (unsigned char byte : digest) {
    text << std::setw(2) << static_cast<unsigned>(byte);
  }
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
                      const std::vector<std::uint8_t>* payload,
                      std::size_t payload_bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:payload_bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
  auto* contents = static_cast<std::uint8_t*>(buffer.contents);
  std::memset(contents, kGuardValue, buffer.length);
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
                                  std::size_t size) {
  return std::vector<std::uint8_t>(guarded.begin() + kGuardBytes,
                                   guarded.begin() + kGuardBytes + size);
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

std::size_t UntouchedPixels(const std::vector<std::uint8_t>& output,
                            std::uint8_t poison) {
  std::size_t result = 0;
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    bool untouched = true;
    for (std::size_t byte = 0; byte < 36u; ++byte) {
      untouched &= output[pixel * 36u + byte] == poison;
    }
    result += untouched;
  }
  return result;
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const Inputs& generated,
                  const std::array<std::vector<std::uint8_t>, 6>& actual_inputs,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t untouched_pixels,
                  const std::string& device_name) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                      untouched_pixels == 0;
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.central-block-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"reconstruction_unit\": \"L71/k_central_block\",\n"
         << "  \"metal_function\": \"fg_central_block\",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"grid\": [40, 23, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"metal_threads\": [640, 368, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"width\": " << kWidth << ",\n"
         << "  \"height\": " << kHeight << ",\n"
         << "  \"factor_f32\": 0.5,\n"
         << "  \"factor_f16_bits\": 14336,\n"
         << "  \"input_generator\": \"" << kGenerator << "\",\n"
         << "  \"input_bytes\": [";
  for (std::size_t lane = 0; lane < kInputBytes.size(); ++lane) {
    stream << kInputBytes[lane] << (lane + 1u == kInputBytes.size() ? "" : ", ");
  }
  stream << "],\n  \"input_pre_sha256\": [";
  for (std::size_t lane = 0; lane < generated.lanes.size(); ++lane) {
    stream << "\"" << Sha256(generated.lanes[lane]) << "\""
           << (lane + 1u == generated.lanes.size() ? "" : ", ");
  }
  stream << "],\n  \"input_post_sha256\": [";
  for (std::size_t lane = 0; lane < actual_inputs.size(); ++lane) {
    stream << "\"" << Sha256(actual_inputs[lane]) << "\""
           << (lane + 1u == actual_inputs.size() ? "" : ", ");
  }
  stream << "],\n"
         << "  \"output_bytes\": " << kOutputBytes << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_pixels\": " << untouched_pixels << ",\n"
         << "  \"input_immutability\": " << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"full_write_coverage\": " << (untouched_pixels == 0 ? "true" : "false") << ",\n"
         << "  \"store_width_contract\": \"nine-u32-per-pixel\",\n"
         << "  \"coordinate_conversion\": \"floor-half-before-signed-clamp\",\n"
         << "  \"implicit_fp_contraction_disabled\": true,\n"
         << "  \"selected_driver_fma_spelled_explicitly\": true,\n"
         << "  \"external_tensor_inputs_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-central-block-mac <metallib> <output-dir> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path library_path = argv[1];
      const std::filesystem::path output_directory = argv[2];
      const std::string run_id = argv[3];
      char* poison_end = nullptr;
      const unsigned long parsed_poison = std::strtoul(argv[4], &poison_end, 16);
      if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
        throw std::runtime_error("invalid poison byte");
      }
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      std::filesystem::create_directories(output_directory);
      const Inputs inputs = GenerateInputs();

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("no Metal device available");
      NSError* error = nil;
      NSURL* url = [NSURL fileURLWithPath:
          [NSString stringWithUTF8String:library_path.string().c_str()]];
      id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
      if (library == nil) {
        throw std::runtime_error("unable to load metallib: " +
                                 std::string(error.localizedDescription.UTF8String));
      }
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_central_block"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) {
        throw std::runtime_error("unable to create pipeline: " +
                                 std::string(error.localizedDescription.UTF8String));
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("unable to create queue");

      NSMutableArray<id<MTLBuffer>>* input_buffers =
          [NSMutableArray arrayWithCapacity:inputs.lanes.size()];
      for (std::size_t lane = 0; lane < inputs.lanes.size(); ++lane) {
        [input_buffers addObject:Guarded(device, &inputs.lanes[lane],
                                         kInputBytes[lane], 0)];
      }
      id<MTLBuffer> output_buffer =
          Guarded(device, nullptr, kOutputBytes, poison);
      const CentralBlockParameters parameters{kWidth, kHeight, 0.5f};

      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (encoder == nil) throw std::runtime_error("unable to create encoder");
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:output_buffer offset:kGuardBytes atIndex:0];
      for (NSUInteger lane = 0; lane < input_buffers.count; ++lane) {
        [encoder setBuffer:input_buffers[lane]
                    offset:kGuardBytes
                   atIndex:lane + 1u];
      }
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:7];
      [encoder dispatchThreads:MTLSizeMake(kWidth, kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal execution failed: " +
                                 std::string(command.error.localizedDescription.UTF8String));
      }

      std::array<std::vector<std::uint8_t>, 6> actual_inputs;
      std::size_t input_mismatches = 0;
      std::size_t canary_mismatches = 0;
      for (std::size_t lane = 0; lane < inputs.lanes.size(); ++lane) {
        const auto guarded = ReadAll(input_buffers[lane]);
        actual_inputs[lane] = Payload(guarded, kInputBytes[lane]);
        input_mismatches += Mismatches(inputs.lanes[lane], actual_inputs[lane]);
        canary_mismatches += CanaryMismatches(guarded, kInputBytes[lane]);
        Write(output_directory / ("input-" + std::to_string(lane) + "-after.bin"),
              actual_inputs[lane]);
      }
      const auto guarded_output = ReadAll(output_buffer);
      const auto output = Payload(guarded_output, kOutputBytes);
      canary_mismatches += CanaryMismatches(guarded_output, kOutputBytes);
      const std::size_t untouched = UntouchedPixels(output, poison);
      Write(output_directory / "output-metal.bin", output);
      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   inputs, actual_inputs, output, input_mismatches,
                   canary_mismatches, untouched,
                   std::string(device.name.UTF8String));
      const bool passed =
          input_mismatches == 0 && canary_mismatches == 0 && untouched == 0;
      std::cout << "result=" << (passed ? "pass" : "fail")
                << " output_sha256=" << Sha256(output) << "\n";
      return passed ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << "\n";
      return 1;
    }
  }
}
