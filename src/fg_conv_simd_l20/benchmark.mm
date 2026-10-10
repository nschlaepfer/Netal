#define main fg_conv_3x3_probe_unused_main
#include "../fg_conv/macos_3x3_probe.mm"
#undef main

#include <algorithm>
#include <chrono>

namespace {

constexpr std::uint32_t kWarmupIterations = 2u;
constexpr std::uint32_t kMeasuredIterations = 8u;
constexpr std::uint32_t kBlocks = 6u;

struct Timing {
  std::string backend;
  double gpu_ms_per_dispatch;
  double wall_ms_per_dispatch;
};

id<MTLComputePipelineState> BenchPipeline(id<MTLDevice> device,
                                          const char* library_path,
                                          NSString* function_name) {
  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithURL:[NSURL fileURLWithPath:
                                  [NSString stringWithUTF8String:library_path]]
                 error:&error];
  if (library == nil)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil) throw std::runtime_error("benchmark function missing");
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (pipeline == nil)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return pipeline;
}

Timing Measure(id<MTLCommandQueue> queue,
               id<MTLComputePipelineState> pipeline,
               id<MTLBuffer> weights, id<MTLBuffer> input,
               id<MTLBuffer> output, const Conv3x3Parameters& parameters,
               const fg_conv_3x3::Layer& layer,
               std::uint32_t iterations, std::string backend) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weights offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
    [encoder dispatchThreads:MTLSizeMake(layer.width, layer.height,
                                         layer.output_channels)
        threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
  }
  [encoder endEncoding];
  const auto start = std::chrono::steady_clock::now();
  [command commit];
  [command waitUntilCompleted];
  const auto end = std::chrono::steady_clock::now();
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
  return {std::move(backend),
          (command.GPUEndTime - command.GPUStartTime) * 1000.0 / iterations,
          std::chrono::duration<double, std::milli>(end - start).count() /
              iterations};
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2u;
  return values.size() % 2u == 0u
      ? 0.5 * (values[middle - 1u] + values[middle])
      : values[middle];
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-l20-simd-benchmark <scalar-metallib> "
                   "<candidate-metallib> <weights> <output-json> <poison-hex> "
                   "<candidate-function> <layer-id>\n";
      return 2;
    }
    try {
      const auto& layer = fg_conv_3x3::FindLayer(argv[7]);
      const auto weights_before = Read3x3(argv[3]);
      if (weights_before.size() != layer.weight_bytes ||
          Sha256(weights_before) != layer.weight_sha256)
        throw std::runtime_error("live weight identity mismatch");
      char* poison_end = nullptr;
      const auto poison_value = std::strtoul(argv[5], &poison_end, 16);
      if (poison_end == argv[5] || *poison_end != '\0' ||
          poison_value > 255u)
        throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(poison_value);
      const auto input_before = fg_conv_3x3::GenerateInput(layer);
      const auto output_bytes = fg_conv_3x3::OutputBytes(layer);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      auto scalar = BenchPipeline(device, argv[1], @"fg_conv_3x3");
      auto candidate = BenchPipeline(
          device, argv[2], [NSString stringWithUTF8String:argv[6]]);
      id<MTLBuffer> weights =
          Guarded(device, &weights_before, weights_before.size(), 0u);
      id<MTLBuffer> input =
          Guarded(device, &input_before, input_before.size(), 0u);
      id<MTLBuffer> scalar_output =
          Guarded(device, nullptr, output_bytes, poison);
      id<MTLBuffer> candidate_output =
          Guarded(device, nullptr, output_bytes, poison ^ 0xffu);
      const Conv3x3Parameters parameters{
          layer.width, layer.height, layer.input_channels,
          layer.output_channels,
          9u * layer.input_channels * layer.output_channels,
          layer.column_stage_schedule ? 1u : 0u};
      Measure(queue, scalar, weights, input, scalar_output, parameters, layer,
              kWarmupIterations, "scalar");
      Measure(queue, candidate, weights, input, candidate_output, parameters,
              layer,
              kWarmupIterations, "candidate");
      std::vector<Timing> timings;
      for (std::uint32_t block = 0u; block < kBlocks; ++block) {
        const bool candidate_first = (block & 1u) != 0u;
        auto run_scalar = [&] {
          timings.push_back(Measure(queue, scalar, weights, input,
              scalar_output, parameters, layer, kMeasuredIterations,
              "scalar"));
        };
        auto run_candidate = [&] {
          timings.push_back(Measure(queue, candidate, weights, input,
              candidate_output, parameters, layer, kMeasuredIterations,
              "candidate"));
        };
        if (candidate_first) { run_candidate(); run_scalar(); }
        else { run_scalar(); run_candidate(); }
      }
      const auto scalar_bytes = Payload(scalar_output, output_bytes);
      const auto candidate_bytes = Payload(candidate_output, output_bytes);
      const auto weights_after = Payload(weights, weights_before.size());
      const auto input_after = Payload(input, input_before.size());
      const auto mismatched_bytes = Mismatches(scalar_bytes, candidate_bytes);
      const auto immutable_mismatches =
          Mismatches(weights_before, weights_after) +
          Mismatches(input_before, input_after);
      const auto canary_mismatches =
          CanaryMismatches(weights, weights_before.size()) +
          CanaryMismatches(input, input_before.size()) +
          CanaryMismatches(scalar_output, output_bytes) +
          CanaryMismatches(candidate_output, output_bytes);
      std::vector<double> scalar_gpu, candidate_gpu, scalar_wall,
          candidate_wall;
      for (const auto& timing : timings) {
        auto& gpu = timing.backend == "scalar" ? scalar_gpu : candidate_gpu;
        auto& wall = timing.backend == "scalar" ? scalar_wall : candidate_wall;
        gpu.push_back(timing.gpu_ms_per_dispatch);
        wall.push_back(timing.wall_ms_per_dispatch);
      }
      const double scalar_gpu_median = Median(scalar_gpu);
      const double candidate_gpu_median = Median(candidate_gpu);
      std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
      if (!output) throw std::runtime_error("unable to write benchmark receipt");
      output << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \"neuralstudio.fg.conv-3x3-optimization-candidate.v1\",\n"
             << "  \"layer_id\": \"" << layer.id << "\",\n"
             << "  \"device\": \"" << device.name.UTF8String << "\",\n"
             << "  \"scalar_output_sha256\": \"" << Sha256(scalar_bytes)
             << "\",\n"
             << "  \"candidate_function\": \"" << argv[6] << "\",\n"
             << "  \"candidate_output_sha256\": \""
             << Sha256(candidate_bytes)
             << "\",\n"
             << "  \"scalar_candidate_mismatched_bytes\": "
             << mismatched_bytes << ",\n"
             << "  \"immutable_input_mismatched_bytes\": "
             << immutable_mismatches << ",\n"
             << "  \"canary_mismatched_bytes\": " << canary_mismatches
             << ",\n"
             << "  \"warmup_iterations\": " << kWarmupIterations << ",\n"
             << "  \"measured_iterations_per_block\": "
             << kMeasuredIterations << ",\n"
             << "  \"block_count_per_backend\": " << kBlocks << ",\n"
             << "  \"scalar_median_gpu_ms\": " << scalar_gpu_median
             << ",\n"
             << "  \"candidate_median_gpu_ms\": "
             << candidate_gpu_median << ",\n"
             << "  \"candidate_gpu_speedup\": "
             << scalar_gpu_median / candidate_gpu_median << ",\n"
             << "  \"scalar_median_wall_ms\": " << Median(scalar_wall)
             << ",\n"
             << "  \"candidate_median_wall_ms\": "
             << Median(candidate_wall)
             << ",\n"
             << "  \"expected_output_bound_to_gpu\": false,\n"
             << "  \"address_fields_recorded\": false,\n"
             << "  \"correctness_result\": \""
             << (mismatched_bytes == 0u ? "pass" : "fail") << "\",\n"
             << "  \"result\": \"diagnostic\"\n"
             << "}\n";
      std::cout << "scalar_median_gpu_ms=" << scalar_gpu_median << '\n'
                << "candidate_median_gpu_ms=" << candidate_gpu_median << '\n'
                << "candidate_gpu_speedup="
                << scalar_gpu_median / candidate_gpu_median << '\n'
                << "mismatched_bytes=" << mismatched_bytes << '\n'
                << "result=diagnostic\n";
      return immutable_mismatches == 0u && canary_mismatches == 0u ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
