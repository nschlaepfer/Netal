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

struct LaunchCase {
  std::uint32_t module_token;
  std::uint32_t launch_token;
  std::uint32_t phase_order;
  std::uint32_t global_order;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t block_z;
  std::size_t module_bytes;
  const char* module_sha256;
  const char* raw_parameter_sha256;
  const char* normalized_parameter_sha256;
  const char* scalar_parameter_sha256;
};

constexpr std::array<LaunchCase, 5> kCases = {{
    {37, 33, 22, 31, 10, 6, 256, 3, 2, 8, 2152,
     "1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26",
     "0bb4c1c5e582ea1a7386318d294df1511fdc2e81287b7943f23f9a531a1690bc",
     "60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80",
     "0f2b3cefe270ff83b105bc583212a894fdaf8671acab257be993fcedab912822"},
    {41, 37, 26, 35, 20, 12, 128, 5, 3, 8, 2152,
     "a717783407a1cea5e3d64b4b23edb451f2b791d8cb7980b374433b9f6ded17e1",
     "6746a095b6748b7257237e90fd20e1a0d8b15a21813fd90f19c3bcd87b3d4296",
     "a143c92c761c72c3b2e384fd13bd94d0b4d6311c3c8ba38de0d45206558f80d2",
     "b76e29be0799d76623d3655509ea507d3f11d2300ac735dfe0a5314c06bc6e12"},
    {41, 41, 30, 39, 40, 24, 128, 10, 6, 8, 2152,
     "a717783407a1cea5e3d64b4b23edb451f2b791d8cb7980b374433b9f6ded17e1",
     "ce20833a64a17bc2a9e38aac82ff4a36753bc837b7268f0b5e4368aab0e7035b",
     "f018d7df61d7d25e61b0648680adfcbfce6ef93d7aec05f8a7172f92b450d8cf",
     "a8c92f29f7e409325b2514e52e884569bde053c5d55b3363c51bfd3f99302aa6"},
    {46, 45, 34, 43, 80, 48, 64, 20, 12, 8, 2152,
     "c733968c9cdca9ba170a9a453d2a831537f467de10e089325aabf8e8ed1cd58f",
     "7c0bcdb4678fa19591fb171585f20c226e23c0097b498caed9e099de67400a45",
     "4e4460b122f091bf427f41d1ffa6822eca5d7cc5344a4c750ed3c57e5a8616b2",
     "39ad4bad3c73d0570f402f9b105b06eb9153ec62da6b3a1a6fa8a4daff3e8eb2"},
    {50, 49, 38, 47, 160, 96, 32, 40, 24, 4, 2144,
     "dc36c503f9fca1cb42944cab551c268b2d738912927f4aab9db4d53ae66766b6",
     "a680b8d6c4aa3ea28e7962297986d8dc0020cc70679dcdd3b8e2cf1f402ce82d",
     "bebf0d883fac729020854143da492c3d22894b49451d9973b27757d000d96bb1",
     "2b746e27a83d6e26c74bb958762f71d52879d76501e7db4bfd80556f58be5193"},
}};

struct ElementwiseFamilyParameters {
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t channels;
  std::uint32_t input_a_width;
  std::uint32_t input_a_height;
  std::uint32_t input_b_width;
  std::uint32_t input_b_height;
  std::uint32_t vector_thread_stride;
};
static_assert(sizeof(ElementwiseFamilyParameters) == 32);

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

std::uint16_t InputHalfBits(std::size_t case_index, std::size_t index,
                            bool input_b) {
  constexpr std::array<std::uint16_t, 12> special_a = {
      0x3C00u, 0x3C01u, 0xBC00u, 0xBC01u, 0x0400u, 0x8400u,
      0x3555u, 0xB555u, 0x3C00u, 0xBC00u, 0x3C01u, 0xBC01u,
  };
  constexpr std::array<std::uint16_t, 12> special_b = {
      0x1000u, 0x1000u, 0x9000u, 0x9000u, 0x0400u, 0x8400u,
      0x3000u, 0xB000u, 0xB800u, 0x3800u, 0xBC00u, 0x3C00u,
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
  constexpr std::uint64_t case_step = 0xD1B54A32D192ED03ull;
  const std::uint64_t lane = static_cast<std::uint64_t>(index);
  const std::uint64_t salt =
      static_cast<std::uint64_t>(case_index + 1u) * case_step;
  const std::uint64_t value = SplitMix64(
      ((input_b ? seed_b : seed_a) ^ salt) +
      lane * (input_b ? step_b : step_a));
  const std::uint16_t sign = static_cast<std::uint16_t>(
      ((SplitMix64((seed_sign ^ salt) + lane * step_sign) >> 63u) & 1u)
       << 15u);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(10u + ((value >> 10u) % 9u));
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

Inputs GenerateInputs(std::size_t case_index, const LaunchCase& launch) {
  const std::size_t values = static_cast<std::size_t>(launch.width) *
                             launch.height * launch.channels;
  Inputs inputs{
      std::vector<std::uint8_t>(values * 2u),
      std::vector<std::uint8_t>(values * 2u),
  };
  for (std::size_t index = 0; index < values; ++index) {
    Store16(&inputs.a, index * 2u,
            InputHalfBits(case_index, index, false));
    Store16(&inputs.b, index * 2u,
            InputHalfBits(case_index, index, true));
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
                      std::size_t payload_bytes, std::uint8_t payload_fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:payload_bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  auto* payload = static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes;
  std::memset(payload, payload_fill, payload_bytes);
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

std::size_t UntouchedHalfWords(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  const std::uint16_t marker =
      static_cast<std::uint16_t>(poison | (std::uint16_t(poison) << 8u));
  std::size_t untouched = 0;
  for (std::size_t offset = 0; offset < output.size(); offset += 2u) {
    const std::uint16_t value = static_cast<std::uint16_t>(
        output[offset] | (std::uint16_t(output[offset + 1u]) << 8u));
    untouched += value == marker;
  }
  return untouched;
}

std::string LaunchStem(std::uint32_t token) {
  std::ostringstream name;
  name << "launch-" << std::setfill('0') << std::setw(6) << token;
  return name.str();
}

struct CaseResult {
  LaunchCase launch{};
  std::vector<std::uint8_t> input_a;
  std::vector<std::uint8_t> input_b;
  std::vector<std::uint8_t> output;
  std::size_t input_mismatches = 0;
  std::size_t canary_mismatches = 0;
  std::size_t untouched_half_words = 0;
};

void WriteReceipt(const std::filesystem::path& path, const std::string& run_id,
                  std::uint8_t poison, const std::string& metallib_sha,
                  const std::string& device_name,
                  const std::vector<CaseResult>& results) {
  std::vector<std::uint8_t> output_root;
  std::size_t total_input_mismatches = 0;
  std::size_t total_canary_mismatches = 0;
  std::size_t total_untouched = 0;
  for (const auto& result : results) {
    output_root.insert(output_root.end(), result.output.begin(),
                       result.output.end());
    total_input_mismatches += result.input_mismatches;
    total_canary_mismatches += result.canary_mismatches;
    total_untouched += result.untouched_half_words;
  }
  const bool passed = total_input_mismatches == 0 &&
                      total_canary_mismatches == 0 && total_untouched == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write run receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.elementwise-family-run.v2\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"poison_byte\": " << static_cast<unsigned>(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"metal_function\": \"fg_elementwise_add_family\",\n"
         << "  \"metallib_sha256\": \"" << metallib_sha << "\",\n"
         << "  \"input_generator\": \"splitmix64-fg-elementwise-family-v2\",\n"
         << "  \"external_tensor_input_files_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"launches\": [\n";
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    const auto& launch = result.launch;
    stream << "    {\n"
           << "      \"index\": " << index << ",\n"
           << "      \"module_token\": " << launch.module_token << ",\n"
           << "      \"module_bytes\": " << launch.module_bytes << ",\n"
           << "      \"module_sha256\": \"" << launch.module_sha256 << "\",\n"
           << "      \"function_name\": \"k_element_wise\",\n"
           << "      \"launch_token\": " << launch.launch_token << ",\n"
           << "      \"evaluate_1_phase_order\": " << launch.phase_order << ",\n"
           << "      \"global_launch_order\": " << launch.global_order << ",\n"
           << "      \"captured_raw_parameter_sha256\": \""
           << launch.raw_parameter_sha256 << "\",\n"
           << "      \"address_free_parameter_sha256\": \""
           << launch.normalized_parameter_sha256 << "\",\n"
           << "      \"scalar_parameter_sha256\": \""
           << launch.scalar_parameter_sha256 << "\",\n"
           << "      \"parameter_scalars_u32\": [" << launch.width << ", "
           << launch.height << ", " << launch.channels << ", "
           << launch.width << ", " << launch.height << ", "
           << launch.width << ", " << launch.height << "],\n"
           << "      \"grid\": [" << launch.grid_x << ", " << launch.grid_y
           << ", 1],\n"
           << "      \"block\": [4, 4, " << launch.block_z << "],\n"
           << "      \"shape_nhwc\": [1, " << launch.height << ", "
           << launch.width << ", " << launch.channels << "],\n"
           << "      \"tensor_bytes\": " << result.output.size() << ",\n"
           << "      \"input_a_pre_sha256\": \"" << Sha256(result.input_a)
           << "\",\n"
           << "      \"input_a_post_sha256\": \"" << Sha256(result.input_a)
           << "\",\n"
           << "      \"input_b_pre_sha256\": \"" << Sha256(result.input_b)
           << "\",\n"
           << "      \"input_b_post_sha256\": \"" << Sha256(result.input_b)
           << "\",\n"
           << "      \"output_sha256\": \"" << Sha256(result.output)
           << "\",\n"
           << "      \"input_mismatched_bytes\": "
           << result.input_mismatches << ",\n"
           << "      \"canary_mismatched_bytes\": "
           << result.canary_mismatches << ",\n"
           << "      \"untouched_half_words\": "
           << result.untouched_half_words << "\n"
           << "    }" << (index + 1u == results.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"output_root_sha256\": \"" << Sha256(output_root) << "\",\n"
         << "  \"input_mismatched_bytes\": " << total_input_mismatches
         << ",\n"
         << "  \"canary_mismatched_bytes\": " << total_canary_mismatches
         << ",\n"
         << "  \"untouched_half_words\": " << total_untouched << ",\n"
         << "  \"input_immutability\": "
         << (total_input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"full_write_coverage\": "
         << (total_untouched == 0 ? "true" : "false") << ",\n"
         << "  \"result\": \"" << (passed ? "pass" : "fail") << "\"\n"
         << "}\n";
  if (!stream) throw std::runtime_error("short run receipt write");
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::cerr << "usage=fg-elementwise-family-mac <metallib> <output-dir> "
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
          [library newFunctionWithName:@"fg_elementwise_add_family"];
      if (function == nil) throw std::runtime_error("Metal function not found");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) {
        throw std::runtime_error(
            std::string("unable to create Metal pipeline: ") +
            (error != nil ? error.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];

      std::vector<CaseResult> results;
      results.reserve(kCases.size());
      for (std::size_t index = 0; index < kCases.size(); ++index) {
        const LaunchCase launch = kCases[index];
        CaseResult result;
        result.launch = launch;
        Inputs inputs = GenerateInputs(index, launch);
        result.input_a = inputs.a;
        result.input_b = inputs.b;
        const std::size_t tensor_bytes = result.input_a.size();
        id<MTLBuffer> input_a =
            Guarded(device, &result.input_a, tensor_bytes, 0);
        id<MTLBuffer> input_b =
            Guarded(device, &result.input_b, tensor_bytes, 0);
        id<MTLBuffer> output = Guarded(device, nullptr, tensor_bytes, poison);
        const ElementwiseFamilyParameters parameters = {
            launch.width, launch.height, launch.channels,
            launch.width, launch.height, launch.width, launch.height,
            launch.block_z,
        };

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_a offset:kGuardBytes atIndex:0];
        [encoder setBuffer:input_b offset:kGuardBytes atIndex:1];
        [encoder setBuffer:output offset:kGuardBytes atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(launch.grid_x,
                                                  launch.grid_y, 1)
                     threadsPerThreadgroup:MTLSizeMake(4, 4, launch.block_z)];
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

        const auto actual_a = Payload(input_a, tensor_bytes);
        const auto actual_b = Payload(input_b, tensor_bytes);
        result.output = Payload(output, tensor_bytes);
        result.input_mismatches = Mismatches(actual_a, result.input_a) +
                                  Mismatches(actual_b, result.input_b);
        result.canary_mismatches =
            CanaryMismatches(input_a, tensor_bytes) +
            CanaryMismatches(input_b, tensor_bytes) +
            CanaryMismatches(output, tensor_bytes);
        result.untouched_half_words =
            UntouchedHalfWords(result.output, poison);

        const std::string stem = LaunchStem(launch.launch_token);
        Write(output_directory / (stem + "-input-a-after.bin"), actual_a);
        Write(output_directory / (stem + "-input-b-after.bin"), actual_b);
        Write(output_directory / (stem + "-output.bin"), result.output);
        results.push_back(std::move(result));
      }

      const std::string device_name = device.name.UTF8String;
      WriteReceipt(output_directory / "run-receipt.json", run_id, poison,
                   metallib_sha, device_name, results);
      std::size_t failures = 0;
      std::vector<std::uint8_t> output_root;
      for (const auto& result : results) {
        failures += result.input_mismatches + result.canary_mismatches +
                    result.untouched_half_words;
        output_root.insert(output_root.end(), result.output.begin(),
                           result.output.end());
      }
      std::cout << "probe=fg_elementwise_family_mac\n"
                << "run_id=" << run_id << '\n'
                << "process_id=" << getpid() << '\n'
                << "device=" << device_name << '\n'
                << "launch_count=" << results.size() << '\n'
                << "output_root_sha256=" << Sha256(output_root) << '\n'
                << "result=" << (failures == 0 ? "pass" : "fail") << '\n';
      return failures == 0 ? 0 : 3;
    } catch (const std::exception& exception) {
      std::cerr << "error=" << exception.what() << '\n';
      return 1;
    }
  }
}
