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

constexpr std::uint32_t kWidth = 10;
constexpr std::uint32_t kHeight = 6;
constexpr std::uint32_t kChannels = 256;
constexpr std::size_t kValues =
    static_cast<std::size_t>(kWidth) * kHeight * kChannels;
constexpr std::size_t kTensorBytes = kValues * sizeof(std::uint16_t);
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kSelectedModuleSha256[] =
    "1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26";
constexpr char kLiveParameterSha256[] =
    "0bb4c1c5e582ea1a7386318d294df1511fdc2e81287b7943f23f9a531a1690bc";
constexpr char kNormalizedParameterSha256[] =
    "60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80";
constexpr char kScalarParameterSha256[] =
    "0f2b3cefe270ff83b105bc583212a894fdaf8671acab257be993fcedab912822";

struct ElementwiseAddParameters {
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
  std::uint32_t input_a_width;
  std::uint32_t input_a_height;
  std::uint32_t input_b_width;
  std::uint32_t input_b_height;
};
static_assert(sizeof(ElementwiseAddParameters) == 28);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t InputHalfBits(std::size_t index, bool input_b) {
  constexpr std::array<std::uint16_t, 8> special_a = {
      0x3C00u, 0x3C01u, 0xBC00u, 0xBC01u,
      0x0400u, 0x8400u, 0x3555u, 0xB555u,
  };
  constexpr std::array<std::uint16_t, 8> special_b = {
      0x1000u, 0x1000u, 0x9000u, 0x9000u,
      0x0400u, 0x8400u, 0x3000u, 0xB000u,
  };
  if (index < special_a.size()) {
    return input_b ? special_b[index] : special_a[index];
  }

  constexpr std::uint64_t seed_a = 0x243F6A8885A308D3ull;
  constexpr std::uint64_t seed_b = 0x13198A2E03707344ull;
  constexpr std::uint64_t seed_sign = 0xA4093822299F31D0ull;
  constexpr std::uint64_t step_a = 0x9E3779B97F4A7C15ull;
  constexpr std::uint64_t step_b = 0xBF58476D1CE4E5B9ull;
  constexpr std::uint64_t step_sign = 0x94D049BB133111EBull;
  const std::uint64_t lane = static_cast<std::uint64_t>(index);
  const std::uint64_t value = SplitMix64(
      (input_b ? seed_b : seed_a) + lane * (input_b ? step_b : step_a));
  const std::uint16_t sign = static_cast<std::uint16_t>(
      ((SplitMix64(seed_sign + lane * step_sign) >> 63u) & 1u) << 15u);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(5u + ((value >> 10u) % 19u));
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>(exponent << 10u) |
      static_cast<std::uint16_t>(value & 0x03FFu));
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

struct Inputs {
  std::vector<std::uint8_t> a;
  std::vector<std::uint8_t> b;
};

Inputs GenerateInputs() {
  Inputs inputs{
      std::vector<std::uint8_t>(kTensorBytes),
      std::vector<std::uint8_t>(kTensorBytes),
  };
  for (std::size_t index = 0; index < kValues; ++index) {
    Store16(&inputs.a, index * 2u, InputHalfBits(index, false));
    Store16(&inputs.b, index * 2u, InputHalfBits(index, true));
  }
  return inputs;
}

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const std::streamsize size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size " + path.string());
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
  for (const unsigned char byte : digest) {
    text << std::setw(2) << static_cast<unsigned>(byte);
  }
  return text.str();
}

id<MTLBuffer> Guarded(id<MTLDevice> device,
                      const std::vector<std::uint8_t>* source,
                      std::uint8_t payload_fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:kTensorBytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  auto* payload = static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes;
  std::memset(payload, payload_fill, kTensorBytes);
  if (source != nullptr) {
    if (source->size() != kTensorBytes) {
      throw std::runtime_error("Metal input geometry mismatch");
    }
    std::memcpy(payload, source->data(), source->size());
  }
  return buffer;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer) {
  const auto* begin = static_cast<const std::uint8_t*>(buffer.contents) +
                      kGuardBytes;
  return std::vector<std::uint8_t>(begin, begin + kTensorBytes);
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer) {
  const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    mismatches += bytes[index] != kGuardValue;
    mismatches += bytes[kGuardBytes + kTensorBytes + index] != kGuardValue;
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

std::size_t UntouchedHalfWords(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  const std::uint16_t marker =
      static_cast<std::uint16_t>(poison | (std::uint16_t(poison) << 8u));
  std::size_t untouched = 0;
  for (std::size_t index = 0; index < kValues; ++index) {
    const std::uint16_t value = static_cast<std::uint16_t>(
        output[index * 2u] | (std::uint16_t(output[index * 2u + 1u]) << 8u));
    untouched += value == marker;
  }
  return untouched;
}

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& metallib_sha,
                  const std::string& device_name, const Inputs& inputs,
                  const std::vector<std::uint8_t>& actual_a,
                  const std::vector<std::uint8_t>& actual_b,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t untouched_half_words) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.elementwise-add-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"reconstruction_unit\": \"launch-000033/k_element_wise/module-000037\",\n"
         << "  \"selected_module_sha256\": \""
         << kSelectedModuleSha256 << "\",\n"
         << "  \"captured_raw_parameter_sha256\": \""
         << kLiveParameterSha256 << "\",\n"
         << "  \"address_free_parameter_sha256\": \""
         << kNormalizedParameterSha256 << "\",\n"
         << "  \"scalar_parameter_sha256\": \""
         << kScalarParameterSha256 << "\",\n"
         << "  \"metal_function\": \"fg_elementwise_add_256\",\n"
         << "  \"metallib_sha256\": \"" << metallib_sha << "\",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"parameter_scalars_u32\": [10, 6, 256, 10, 6, 10, 6],\n"
         << "  \"grid\": [3, 2, 1],\n"
         << "  \"block\": [4, 4, 8],\n"
         << "  \"dynamic_shared_bytes\": 0,\n"
         << "  \"shape_nhwc\": [1, 6, 10, 256],\n"
         << "  \"operation\": \"binary16 add.rn, eight channels per vector, NHWC\",\n"
         << "  \"input_generator\": \"splitmix64-fg-elementwise-add-v1\",\n"
         << "  \"input_a_pre_sha256\": \"" << Sha256(inputs.a) << "\",\n"
         << "  \"input_a_post_sha256\": \"" << Sha256(actual_a) << "\",\n"
         << "  \"input_b_pre_sha256\": \"" << Sha256(inputs.b) << "\",\n"
         << "  \"input_b_post_sha256\": \"" << Sha256(actual_b) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"tensor_bytes\": " << kTensorBytes << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"untouched_half_words\": " << untouched_half_words << ",\n"
         << "  \"full_write_coverage\": "
         << (untouched_half_words == 0 ? "true" : "false") << ",\n"
         << "  \"input_immutability\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"external_tensor_input_files_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \""
         << ((input_mismatches == 0 && canary_mismatches == 0 &&
              untouched_half_words == 0)
                 ? "pass"
                 : "fail")
         << "\"\n}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-elementwise-mac <metallib> <output-dir> "
                   "<run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path metallib_path = argv[1];
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

      const Inputs inputs = GenerateInputs();
      const auto metallib_bytes = Read(metallib_path);
      const std::string metallib_sha = Sha256(metallib_bytes);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("no Metal device available");

      NSError* error = nil;
      NSURL* library_url = [NSURL fileURLWithPath:
          [NSString stringWithUTF8String:metallib_path.string().c_str()]];
      id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
      if (library == nil) {
        throw std::runtime_error(
            std::string("unable to load metallib: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_elementwise_add_256"];
      if (function == nil) {
        throw std::runtime_error("Metal function was not found");
      }
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) {
        throw std::runtime_error(
            std::string("unable to create Metal pipeline: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }

      id<MTLBuffer> input_a = Guarded(device, &inputs.a, 0);
      id<MTLBuffer> input_b = Guarded(device, &inputs.b, 0);
      id<MTLBuffer> output = Guarded(device, nullptr, poison);
      const ElementwiseAddParameters parameters = {
          kWidth, kHeight, kChannels, kWidth, kHeight, kWidth, kHeight};

      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input_a offset:kGuardBytes atIndex:0];
      [encoder setBuffer:input_b offset:kGuardBytes atIndex:1];
      [encoder setBuffer:output offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(3, 2, 1)
                   threadsPerThreadgroup:MTLSizeMake(4, 4, 8)];
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

      const auto actual_a = Payload(input_a);
      const auto actual_b = Payload(input_b);
      const auto actual_output = Payload(output);
      const std::size_t input_mismatches =
          Mismatches(actual_a, inputs.a) + Mismatches(actual_b, inputs.b);
      const std::size_t canary_mismatches =
          CanaryMismatches(input_a) + CanaryMismatches(input_b) +
          CanaryMismatches(output);
      const std::size_t untouched_half_words =
          UntouchedHalfWords(actual_output, poison);

      Write(output_directory / "input-a-after.bin", actual_a);
      Write(output_directory / "input-b-after.bin", actual_b);
      Write(output_directory / "output.bin", actual_output);
      const std::string device_name = device.name.UTF8String;
      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   metallib_sha, device_name, inputs, actual_a, actual_b,
                   actual_output, input_mismatches, canary_mismatches,
                   untouched_half_words);

      const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                          untouched_half_words == 0;
      std::cout << "probe=fg_elementwise_add_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "device=" << device_name << '\n'
                << "metallib_sha256=" << metallib_sha << '\n'
                << "input_a_sha256=" << Sha256(actual_a) << '\n'
                << "input_b_sha256=" << Sha256(actual_b) << '\n'
                << "output_sha256=" << Sha256(actual_output) << '\n'
                << "input_mismatched_bytes=" << input_mismatches << '\n'
                << "canary_mismatched_bytes=" << canary_mismatches << '\n'
                << "untouched_half_words=" << untouched_half_words << '\n'
                << "result=" << (passed ? "pass" : "fail") << '\n';
      return passed ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
