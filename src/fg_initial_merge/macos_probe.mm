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
constexpr std::uint32_t kHeight = 368;
constexpr std::size_t kPixels =
    static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight);
constexpr std::size_t kInputABytes = kPixels * 8;
constexpr std::size_t kInputBBytes = kPixels * 8;
constexpr std::size_t kInputCBytes = kPixels * 2;
constexpr std::size_t kOutputBytes = kPixels * 20;
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;

struct InitialMergeParameters {
  std::uint32_t height;
  std::uint32_t width;
  float interpolation;
};
static_assert(sizeof(InitialMergeParameters) == 12);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void PutU64(std::vector<std::uint8_t>* output, std::size_t offset,
            std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    (*output)[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8u));
  }
}

void PutU16(std::vector<std::uint8_t>* output, std::size_t offset,
            std::uint16_t value) {
  (*output)[offset] = static_cast<std::uint8_t>(value);
  (*output)[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
}

struct Inputs {
  std::vector<std::uint8_t> a;
  std::vector<std::uint8_t> b;
  std::vector<std::uint8_t> c;
};

Inputs MakeInputs() {
  Inputs result{
      std::vector<std::uint8_t>(kInputABytes),
      std::vector<std::uint8_t>(kInputBBytes),
      std::vector<std::uint8_t>(kInputCBytes),
  };
  const std::uint64_t seed_a = 0x243F6A8885A308D3ull;
  const std::uint64_t seed_b = 0x13198A2E03707344ull;
  const std::uint64_t seed_c = 0xA4093822299F31D0ull;
  const std::uint64_t step_b = 0x9E3779B97F4A7C15ull;
  const std::uint64_t step_c = 0xBF58476D1CE4E5B9ull;
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    const std::uint64_t lane = static_cast<std::uint64_t>(pixel);
    PutU64(&result.a, pixel * 8, SplitMix64(seed_a ^ lane));
    PutU64(&result.b, pixel * 8, SplitMix64(seed_b + lane * step_b));
    PutU16(&result.c, pixel * 2,
           static_cast<std::uint16_t>(
               SplitMix64(seed_c ^ (lane * step_c)) >> 19u));
  }
  return result;
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
  if (source != nullptr) {
    if (source->size() != payload_bytes) {
      throw std::runtime_error("Metal input geometry mismatch");
    }
    std::memcpy(payload, source->data(), payload_bytes);
  }
  return buffer;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer,
                                  std::size_t payload_bytes) {
  const auto* begin = static_cast<const std::uint8_t*>(buffer.contents) +
                      kGuardBytes;
  return std::vector<std::uint8_t>(begin, begin + payload_bytes);
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer,
                             std::size_t payload_bytes) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += bytes[index] != kGuardValue;
    mismatches += bytes[kGuardBytes + payload_bytes + index] != kGuardValue;
  }
  return mismatches;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("comparison size mismatch");
  }
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    mismatches += left[index] != right[index];
  }
  return mismatches;
}

std::size_t UntouchedPixels(const std::vector<std::uint8_t>& output,
                            std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    bool untouched = true;
    for (std::size_t byte = 0; byte < 20; ++byte) {
      untouched &= output[pixel * 20 + byte] == poison;
    }
    count += untouched;
  }
  return count;
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const Inputs& inputs,
                  const std::vector<std::uint8_t>& actual_a,
                  const std::vector<std::uint8_t>& actual_b,
                  const std::vector<std::uint8_t>& actual_c,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches, std::size_t canary_mismatches,
                  std::size_t untouched_pixels, const std::string& device_name) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.initial-merge-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"reconstruction_unit\": \"L56/k_initial_merge\",\n"
         << "  \"metal_function\": \"fg_initial_merge\",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"grid\": [40, 23, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"height\": " << kHeight << ",\n"
         << "  \"width\": " << kWidth << ",\n"
         << "  \"interpolation_f32\": 0.5,\n"
         << "  \"interpolation_f16_bits\": 14336,\n"
         << "  \"store_width_contract\": \"five-u32-at-0-4-8-12-16\",\n"
         << "  \"input_generator\": \"splitmix64-fg-initial-merge-v1\",\n"
         << "  \"input_a_pre_sha256\": \"" << Sha256(inputs.a) << "\",\n"
         << "  \"input_a_post_sha256\": \"" << Sha256(actual_a) << "\",\n"
         << "  \"input_b_pre_sha256\": \"" << Sha256(inputs.b) << "\",\n"
         << "  \"input_b_post_sha256\": \"" << Sha256(actual_b) << "\",\n"
         << "  \"input_c_pre_sha256\": \"" << Sha256(inputs.c) << "\",\n"
         << "  \"input_c_post_sha256\": \"" << Sha256(actual_c) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_pixels\": " << untouched_pixels << ",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"full_write_coverage\": "
         << (untouched_pixels == 0 ? "true" : "false") << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"external_input_files_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \""
         << ((input_mismatches == 0 && canary_mismatches == 0 &&
              untouched_pixels == 0)
                 ? "pass"
                 : "fail")
         << "\"\n}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-initial-merge-mac <metallib> <output-dir> "
                   "<run-id> <poison-hex>\n";
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
      std::filesystem::create_directories(output_directory);
      const Inputs inputs = MakeInputs();

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSString* library_path = [NSString stringWithUTF8String:argv[1]];
      NSError* error = nil;
      id<MTLLibrary> library =
          [device newLibraryWithURL:[NSURL fileURLWithPath:library_path]
                              error:&error];
      if (library == nil) {
        throw std::runtime_error(
            std::string("metallib load failed: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_initial_merge"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");

      id<MTLBuffer> input_a = Guarded(device, &inputs.a, kInputABytes, 0);
      id<MTLBuffer> input_b = Guarded(device, &inputs.b, kInputBBytes, 0);
      id<MTLBuffer> input_c = Guarded(device, &inputs.c, kInputCBytes, 0);
      id<MTLBuffer> output =
          Guarded(device, nullptr, kOutputBytes, poison);
      InitialMergeParameters parameters{kHeight, kWidth, 0.5f};
      id<MTLBuffer> parameter_buffer =
          [device newBufferWithBytes:&parameters
                              length:sizeof(parameters)
                             options:MTLResourceStorageModeShared];
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (parameter_buffer == nil || queue == nil) {
        throw std::runtime_error("Metal setup failed");
      }
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:output offset:kGuardBytes atIndex:0];
      [encoder setBuffer:input_a offset:kGuardBytes atIndex:1];
      [encoder setBuffer:input_b offset:kGuardBytes atIndex:2];
      [encoder setBuffer:input_c offset:kGuardBytes atIndex:3];
      [encoder setBuffer:parameter_buffer offset:0 atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(40, 23, 1)
                   threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
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

      const auto actual_a = Payload(input_a, kInputABytes);
      const auto actual_b = Payload(input_b, kInputBBytes);
      const auto actual_c = Payload(input_c, kInputCBytes);
      const auto actual_output = Payload(output, kOutputBytes);
      const std::size_t input_mismatches =
          Mismatches(actual_a, inputs.a) + Mismatches(actual_b, inputs.b) +
          Mismatches(actual_c, inputs.c);
      const std::size_t canary_mismatches =
          CanaryMismatches(input_a, kInputABytes) +
          CanaryMismatches(input_b, kInputBBytes) +
          CanaryMismatches(input_c, kInputCBytes) +
          CanaryMismatches(output, kOutputBytes);
      const std::size_t untouched_pixels =
          UntouchedPixels(actual_output, poison);

      Write(output_directory / "input-a-after.bin", actual_a);
      Write(output_directory / "input-b-after.bin", actual_b);
      Write(output_directory / "input-c-after.bin", actual_c);
      Write(output_directory / "output.bin", actual_output);
      const std::string device_name = device.name.UTF8String;
      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   inputs, actual_a, actual_b, actual_c, actual_output,
                   input_mismatches, canary_mismatches, untouched_pixels,
                   device_name);

      const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                          untouched_pixels == 0;
      std::cout << "probe=fg_initial_merge_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "device_name=" << device_name << '\n'
                << "output_sha256=" << Sha256(actual_output) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "untouched_pixels=" << untouched_pixels << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
