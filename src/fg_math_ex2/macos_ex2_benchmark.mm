#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

constexpr std::size_t kRecords = 1u << 20u;
constexpr std::size_t kIdentityBytes = kRecords * 16u;
constexpr std::size_t kPayloadBytes = kRecords * 4u;
constexpr std::size_t kUnitBytes = 67'108'868u;
constexpr std::size_t kFractionBytes = 33'554'436u;
constexpr std::size_t kUnitCorrectionBytes = 4'194'305u;
constexpr std::size_t kFractionCorrectionBytes = 2'097'153u;
constexpr std::size_t kGuardBytes = 256u;
constexpr std::uint8_t kGuardValue = 0xc7u;
constexpr std::uint32_t kWarmupIterations = 4u;
constexpr std::uint32_t kMeasuredIterations = 16u;
constexpr std::uint32_t kBlocks = 8u;

struct IdentityRecord {
  std::uint32_t input_bits;
  std::uint32_t direct_bits;
  std::uint32_t normalized_bits;
  std::int32_t binary_scale;
};
static_assert(sizeof(IdentityRecord) == 16);

struct BenchmarkParameters {
  std::uint32_t count;
};

struct Measurement {
  std::string backend;
  std::uint32_t block;
  double gpu_ms_per_dispatch;
  double wall_ms_per_dispatch;
};

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest)
    text << std::setw(2) << static_cast<unsigned int>(byte);
  return text.str();
}

id<MTLBuffer> Guarded(id<MTLDevice> device, const void* source,
                      std::size_t bytes, std::uint8_t fill) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes + 2u * kGuardBytes
                          options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Metal allocation failed");
  std::memset(buffer.contents, kGuardValue, buffer.length);
  auto* payload = static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes;
  if (source) std::memcpy(payload, source, bytes);
  else std::memset(payload, fill, bytes);
  return buffer;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  return {pointer + kGuardBytes, pointer + kGuardBytes + bytes};
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer, std::size_t bytes) {
  const auto* pointer = static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t result = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    result += pointer[index] != kGuardValue;
    result += pointer[kGuardBytes + bytes + index] != kGuardValue;
  }
  return result;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& left,
                       const std::vector<std::uint8_t>& right) {
  if (left.size() != right.size()) return std::max(left.size(), right.size());
  std::size_t result = 0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result += left[index] != right[index];
  return result;
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library,
                                     NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) throw std::runtime_error("Metal function unavailable");
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return pipeline;
}

Measurement Measure(id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> input, id<MTLBuffer> table_a,
                    id<MTLBuffer> table_b, id<MTLBuffer> output,
                    std::uint32_t iterations, std::string backend,
                    std::uint32_t block) {
  const BenchmarkParameters parameters{static_cast<std::uint32_t>(kRecords)};
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:kGuardBytes atIndex:0];
  [encoder setBuffer:table_a offset:0 atIndex:1];
  [encoder setBuffer:table_b offset:0 atIndex:2];
  [encoder setBuffer:output offset:kGuardBytes atIndex:3];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
  for (std::uint32_t index = 0; index < iterations; ++index) {
    [encoder dispatchThreads:MTLSizeMake(kRecords, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  [encoder endEncoding];
  const auto start = std::chrono::steady_clock::now();
  [command commit];
  [command waitUntilCompleted];
  const auto end = std::chrono::steady_clock::now();
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
  const double gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
  const double wall_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return {std::move(backend), block, gpu_ms / iterations,
          wall_ms / iterations};
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2u;
  return values.size() % 2u == 0u
      ? (values[middle - 1u] + values[middle]) * 0.5
      : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage=fg-sm120-ex2-benchmark <metallib> <unit-table> "
                 "<fraction-table> <unit-corrections> "
                 "<fraction-corrections> <identity-records> <output-json> "
                 "<run-id>\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const auto unit = Read(argv[2]);
      const auto fraction = Read(argv[3]);
      const auto unit_corrections = Read(argv[4]);
      const auto fraction_corrections = Read(argv[5]);
      const auto identity = Read(argv[6]);
      const auto output_path = std::filesystem::path(argv[7]);
      if (std::filesystem::exists(output_path))
        throw std::runtime_error("output exists");
      if (unit.size() != kUnitBytes || fraction.size() != kFractionBytes ||
          unit_corrections.size() != kUnitCorrectionBytes ||
          fraction_corrections.size() != kFractionCorrectionBytes ||
          identity.size() != kIdentityBytes)
        throw std::runtime_error("input extent mismatch");
      std::vector<std::uint8_t> inputs(kPayloadBytes);
      std::vector<std::uint8_t> expected(kPayloadBytes);
      for (std::size_t index = 0; index < kRecords; ++index) {
        IdentityRecord record{};
        std::memcpy(&record, identity.data() + index * sizeof(record),
                    sizeof(record));
        std::memcpy(inputs.data() + index * 4u, &record.input_bits, 4u);
        std::memcpy(expected.data() + index * 4u, &record.direct_bits, 4u);
      }

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
      if (!library)
        throw std::runtime_error(error.localizedDescription.UTF8String);
      auto reference_pipeline = Pipeline(
          device, library, @"fg_sm120_ex2_reference_benchmark");
      auto compact_pipeline = Pipeline(
          device, library, @"fg_sm120_ex2_compact_benchmark");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLBuffer> input = Guarded(device, inputs.data(), kPayloadBytes, 0u);
      id<MTLBuffer> reference_output =
          Guarded(device, nullptr, kPayloadBytes, 0xa5u);
      id<MTLBuffer> compact_output =
          Guarded(device, nullptr, kPayloadBytes, 0x5au);
      id<MTLBuffer> unit_buffer = [device newBufferWithBytes:unit.data()
          length:unit.size() options:MTLResourceStorageModeShared];
      id<MTLBuffer> fraction_buffer = [device newBufferWithBytes:fraction.data()
          length:fraction.size() options:MTLResourceStorageModeShared];
      id<MTLBuffer> unit_correction_buffer =
          [device newBufferWithBytes:unit_corrections.data()
                              length:unit_corrections.size()
                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> fraction_correction_buffer =
          [device newBufferWithBytes:fraction_corrections.data()
                              length:fraction_corrections.size()
                             options:MTLResourceStorageModeShared];
      if (!queue || !input || !reference_output || !compact_output ||
          !unit_buffer || !fraction_buffer || !unit_correction_buffer ||
          !fraction_correction_buffer)
        throw std::runtime_error("Metal setup failed");

      Measure(queue, reference_pipeline, input, unit_buffer, fraction_buffer,
              reference_output, kWarmupIterations, "reference", 0u);
      Measure(queue, compact_pipeline, input, unit_correction_buffer,
              fraction_correction_buffer, compact_output, kWarmupIterations,
              "compact", 0u);
      std::vector<Measurement> measurements;
      for (std::uint32_t block = 0; block < kBlocks; ++block) {
        const bool compact_first = (block & 1u) != 0u;
        auto run_reference = [&] {
          measurements.push_back(Measure(
              queue, reference_pipeline, input, unit_buffer, fraction_buffer,
              reference_output, kMeasuredIterations, "reference", block));
        };
        auto run_compact = [&] {
          measurements.push_back(Measure(
              queue, compact_pipeline, input, unit_correction_buffer,
              fraction_correction_buffer, compact_output, kMeasuredIterations,
              "compact", block));
        };
        if (compact_first) {
          run_compact();
          run_reference();
        } else {
          run_reference();
          run_compact();
        }
      }

      const auto actual_reference = Payload(reference_output, kPayloadBytes);
      const auto actual_compact = Payload(compact_output, kPayloadBytes);
      const auto reference_mismatches = Mismatches(expected, actual_reference);
      const auto compact_mismatches = Mismatches(expected, actual_compact);
      const auto cross_mismatches =
          Mismatches(actual_reference, actual_compact);
      const auto input_mismatches = Mismatches(inputs, Payload(input, kPayloadBytes));
      const auto canary_mismatches =
          CanaryMismatches(input, kPayloadBytes) +
          CanaryMismatches(reference_output, kPayloadBytes) +
          CanaryMismatches(compact_output, kPayloadBytes);
      const auto table_mismatches =
          (std::memcmp(unit.data(), unit_buffer.contents, unit.size()) != 0) +
          (std::memcmp(fraction.data(), fraction_buffer.contents,
                       fraction.size()) != 0) +
          (std::memcmp(unit_corrections.data(),
                       unit_correction_buffer.contents,
                       unit_corrections.size()) != 0) +
          (std::memcmp(fraction_corrections.data(),
                       fraction_correction_buffer.contents,
                       fraction_corrections.size()) != 0);
      std::vector<double> reference_gpu, compact_gpu;
      for (const auto& measurement : measurements) {
        (measurement.backend == "reference" ? reference_gpu : compact_gpu)
            .push_back(measurement.gpu_ms_per_dispatch);
      }
      const auto reference_median = Median(reference_gpu);
      const auto compact_median = Median(compact_gpu);
      const bool pass = reference_mismatches == 0u && compact_mismatches == 0u &&
                        cross_mismatches == 0u && input_mismatches == 0u &&
                        canary_mismatches == 0u && table_mismatches == 0u;
      std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
      if (!output) throw std::runtime_error("unable to write output receipt");
      output << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \"neuralstudio.fg.sm120-ex2-microbenchmark-run.v1\",\n"
             << "  \"run_id\": \"" << argv[8] << "\",\n"
             << "  \"process_id\": " << getpid() << ",\n"
             << "  \"device\": \"" << device.name.UTF8String << "\",\n"
             << "  \"records_per_dispatch\": " << kRecords << ",\n"
             << "  \"warmup_iterations\": " << kWarmupIterations << ",\n"
             << "  \"measured_iterations_per_block\": "
             << kMeasuredIterations << ",\n"
             << "  \"block_count_per_backend\": " << kBlocks << ",\n"
             << "  \"reference_table_bytes\": "
             << kUnitBytes + kFractionBytes << ",\n"
             << "  \"compact_correction_bytes\": "
             << kUnitCorrectionBytes + kFractionCorrectionBytes << ",\n"
             << "  \"reference_median_gpu_ms_per_dispatch\": "
             << reference_median << ",\n"
             << "  \"compact_median_gpu_ms_per_dispatch\": "
             << compact_median << ",\n"
             << "  \"median_speedup\": "
             << reference_median / compact_median << ",\n"
             << "  \"reference_output_sha256\": \""
             << Sha256(actual_reference) << "\",\n"
             << "  \"compact_output_sha256\": \""
             << Sha256(actual_compact) << "\",\n"
             << "  \"reference_mismatched_bytes\": "
             << reference_mismatches << ",\n"
             << "  \"compact_mismatched_bytes\": " << compact_mismatches
             << ",\n"
             << "  \"cross_mismatched_bytes\": " << cross_mismatches
             << ",\n"
             << "  \"input_mismatched_bytes\": " << input_mismatches
             << ",\n"
             << "  \"canary_mismatched_bytes\": " << canary_mismatches
             << ",\n"
             << "  \"table_mismatches\": " << table_mismatches << ",\n"
             << "  \"expected_output_bound_to_gpu\": false,\n"
             << "  \"address_fields_recorded\": false,\n"
             << "  \"measurements\": [\n";
      for (std::size_t index = 0; index < measurements.size(); ++index) {
        const auto& measurement = measurements[index];
        output << "    {\"backend\": \"" << measurement.backend
               << "\", \"block\": " << measurement.block
               << ", \"gpu_ms_per_dispatch\": "
               << measurement.gpu_ms_per_dispatch
               << ", \"wall_ms_per_dispatch\": "
               << measurement.wall_ms_per_dispatch << "}"
               << (index + 1u == measurements.size() ? "\n" : ",\n");
      }
      output << "  ],\n"
             << "  \"result\": \"" << (pass ? "pass" : "fail")
             << "\"\n"
             << "}\n";
      std::cout << "reference_median_gpu_ms=" << reference_median << '\n'
                << "compact_median_gpu_ms=" << compact_median << '\n'
                << "median_speedup=" << reference_median / compact_median
                << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
