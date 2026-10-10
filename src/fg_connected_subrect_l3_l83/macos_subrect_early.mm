#define main FgSubrectEarlyEmbeddedFixedMain
#include "../fg_connected_l3_l19/macos_connected.mm"
#undef main

#include "contract.h"

#ifdef FG_SUBRECT_FULL_CHAIN
#include "macos_subrect_live_handoff.h"
int FgSubrectTailMain(int argc, const char* argv[]);
#endif

namespace {
namespace subrect = fg_connected_subrect_l3_l83;

struct SubrectM11Parameters {
  std::uint32_t resource_width, resource_height;
  std::uint32_t origin_x, origin_y;
  std::uint32_t active_width, active_height;
  std::uint32_t output_width, output_height;
  std::uint32_t mode;
};

struct SubrectM15Parameters {
  std::uint32_t resource_width, resource_height;
  std::uint32_t origin_x, origin_y;
  std::uint32_t active_width, active_height;
  std::uint32_t output_width, output_height;
  std::array<float, 2> input_inverse, output_inverse;
  std::array<float, 2> lower_coordinate, upper_coordinate, output_scale;
};

struct SubrectM5Parameters {
  std::uint32_t motion_width, motion_height, work_width, work_height;
  std::uint32_t tensor_width, tensor_height;
  std::int32_t input_x_offset, input_y_offset;
};

void ResetPayload(id<MTLBuffer> buffer,
                  const std::vector<std::uint8_t>& bytes) {
  if (buffer.length != bytes.size() + 2u * kGuardBytes)
    throw std::runtime_error("subrect buffer reset size drift");
  std::memcpy(static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes,
              bytes.data(), bytes.size());
}

void EncodeM13Pass(id<MTLCommandQueue> queue,
                   id<MTLComputePipelineState> pipeline,
                   id<MTLTexture> flow, id<MTLTexture> depth,
                   id<MTLTexture> mask, id<MTLBuffer> high,
                   id<MTLBuffer> low, id<MTLBuffer> rom,
                   const L5Parameters& parameters) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:flow atIndex:0];
  [encoder setTexture:depth atIndex:1];
  [encoder setTexture:mask atIndex:2];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:rom offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(parameters.width, parameters.height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
  Wait(command);
}

void EncodeM13Pack(id<MTLCommandQueue> queue,
                   id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> high, id<MTLBuffer> low,
                   id<MTLBuffer> output,
                   const L5Parameters& parameters) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(subrect::kEarlyPixels, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  Wait(command);
}

std::size_t OutsideMismatches(const std::vector<std::uint8_t>& before,
                              const std::vector<std::uint8_t>& after,
                              std::uint32_t bytes_per_pixel,
                              std::uint32_t origin_x,
                              std::uint32_t origin_y) {
  if (before.size() != after.size() ||
      before.size() != std::size_t{subrect::kResourceWidth} *
                           subrect::kResourceHeight * bytes_per_pixel)
    throw std::runtime_error("subrect outside comparison size drift");
  std::size_t mismatches = 0;
  for (std::uint32_t y = 0; y < subrect::kResourceHeight; ++y) {
    for (std::uint32_t x = 0; x < subrect::kResourceWidth; ++x) {
      if (x >= origin_x && x < origin_x + subrect::kWidth &&
          y >= origin_y && y < origin_y + subrect::kHeight)
        continue;
      const auto offset =
          (std::size_t{y} * subrect::kResourceWidth + x) * bytes_per_pixel;
      for (std::uint32_t lane = 0; lane < bytes_per_pixel; ++lane)
        mismatches += before[offset + lane] != after[offset + lane];
    }
  }
  return mismatches;
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 10) {
      std::cerr << "usage=fg-connected-subrect-l3-l83-mac <libs> <model> "
                   "<rom> <sigmoid> <ex2-unit> <ex2-fraction> <output> "
                   "<run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const fs::path libs = argv[1];
      const fs::path output_root = argv[7];
      fs::create_directories(output_root);
      const auto parsed_poison = std::stoul(argv[9], nullptr, 16);
      if (parsed_poison > 255u) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const auto rom = ReadFile(argv[3]);
      if (rom.size() != subrect::kRcpRomBytes ||
          Sha256(rom) != subrect::kRcpRomSha256)
        throw std::runtime_error("RCP ROM identity mismatch");
      const auto external = subrect::GenerateEarlyExternal();

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (device == nil || queue == nil)
        throw std::runtime_error("Metal unavailable");
      auto m11 = Pipeline(device, libs / "subrect_module11.metallib",
                          "fg_subrect_module11");
      auto m15 = Pipeline(device, libs / "subrect_module15.metallib",
                          "fg_subrect_module15");
      auto m13_high = Pipeline(device, libs / "module13.metallib",
                               "fg_early_l5_l15_high");
      auto m13_low = Pipeline(device, libs / "module13.metallib",
                              "fg_early_l5_l15_low");
      auto m13_pack = Pipeline(device, libs / "module13.metallib",
                               "fg_early_l5_l15_pack");
      auto m21 = Pipeline(device, libs / "module21.metallib",
                          "fg_early_l6_l16");
      auto m23 = Pipeline(device, libs / "module23.metallib",
                          "fg_early_l7_l17");
      auto m25 = Pipeline(device, libs / "module25.metallib",
                          "fg_early_l8_l18");
      auto m14 = Pipeline(device, libs / "module14.metallib",
                          "fg_dynamic_module14");
      auto m4 = Pipeline(device, libs / "module4.metallib", "fg_early_l11");
      auto m5_surfaces = Pipeline(
          device, libs / "module5.metallib",
          "fg_connected_subrect_module5_surfaces");
      auto m5_tensor = Pipeline(
          device, libs / "module5.metallib",
          "fg_connected_subrect_module5_tensor");

      id<MTLBuffer> rom_buffer = Buffer(device, rom);
      auto source_reset = Texture(
          device, MTLPixelFormatRGBA8Unorm, subrect::kResourceWidth,
          subrect::kResourceHeight, MTLTextureUsageShaderRead,
          external.source_reset, 4);
      auto source_generated = Texture(
          device, MTLPixelFormatRGBA8Unorm, subrect::kResourceWidth,
          subrect::kResourceHeight, MTLTextureUsageShaderRead,
          external.source_generated, 4);
      const auto history_a_initial = subrect::InitialHistory();
      const std::vector<std::uint8_t> history_b_initial(
          subrect::kHistoryBytes, poison);
      auto history_a = Texture(
          device, MTLPixelFormatRGBA16Float, subrect::kResourceWidth,
          subrect::kResourceHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          history_a_initial, 8);
      auto history_b = Texture(
          device, MTLPixelFormatRGBA16Float, subrect::kResourceWidth,
          subrect::kResourceHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          history_b_initial, 8);
      auto reset_color = Texture(
          device, MTLPixelFormatRGBA16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyColorBytes, poison), 8);
      auto generated_color = Texture(
          device, MTLPixelFormatRGBA16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyColorBytes, poison), 8);
      auto shared_metric = Texture(
          device, MTLPixelFormatR16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyMetricBytes, poison), 2);
      auto vector_reset = Texture(
          device, MTLPixelFormatRG16Float, subrect::kResourceWidth,
          subrect::kResourceHeight, MTLTextureUsageShaderRead,
          external.vector_reset, 4);
      auto vector_generated = Texture(
          device, MTLPixelFormatRG16Float, subrect::kResourceWidth,
          subrect::kResourceHeight, MTLTextureUsageShaderRead,
          external.vector_generated, 4);
      auto scalar_reset = Texture(
          device, MTLPixelFormatR32Float, subrect::kResourceWidth,
          subrect::kResourceHeight, MTLTextureUsageShaderRead,
          external.scalar_reset, 4);
      auto scalar_generated = Texture(
          device, MTLPixelFormatR32Float, subrect::kResourceWidth,
          subrect::kResourceHeight, MTLTextureUsageShaderRead,
          external.scalar_generated, 4);
      auto flow = Texture(
          device, MTLPixelFormatRG32Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyFlowBytes, poison), 8);
      auto reset_depth = Texture(
          device, MTLPixelFormatR32Uint, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyDepthBytes, poison), 4);
      auto generated_depth = Texture(
          device, MTLPixelFormatR32Uint, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          subrect::InitialGeneratedDepth(), 4);
      auto reset_mask = Texture(
          device, MTLPixelFormatR8Unorm, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyMaskBytes, poison), 1);
      auto generated_mask = Texture(
          device, MTLPixelFormatR8Unorm, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          subrect::InitialGeneratedMask(), 1);
      auto working_vector = Texture(
          device, MTLPixelFormatRG16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyVectorBytes, poison), 4);
      auto working_mask = Texture(
          device, MTLPixelFormatR8Unorm, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyMaskBytes, poison), 1);
      auto scatter_reset = Texture(
          device, MTLPixelFormatR8Unorm, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyMaskBytes, 255u), 1);
      auto scatter_generated = Texture(
          device, MTLPixelFormatR8Unorm, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyMaskBytes, 255u), 1);
      auto reset_projection = Texture(
          device, MTLPixelFormatRG16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyVectorBytes, poison), 4);
      auto allocation29 = Texture(
          device, MTLPixelFormatR16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight, MTLTextureUsageShaderRead,
          subrect::InitialAllocation29(), 2);
      auto auxiliary_mask = Texture(
          device, MTLPixelFormatR8Unorm, subrect::kEarlyWidth,
          subrect::kEarlyHeight, MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyMaskBytes, poison), 1);
      auto auxiliary_vector = Texture(
          device, MTLPixelFormatRG16Float, subrect::kEarlyWidth,
          subrect::kEarlyHeight, MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(subrect::kEarlyVectorBytes, poison), 4);
      const std::vector<std::uint8_t> reset_present_initial(
          subrect::kSourceBytes, poison);
      auto reset_present = Texture(
          device, MTLPixelFormatRGBA8Unorm, subrect::kResourceWidth,
          subrect::kResourceHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          reset_present_initial, 4);

      id<MTLBuffer> atomic = Buffer(device, subrect::InitialAtomic());
      std::array<id<MTLBuffer>, 5> pyramid{};
      for (std::size_t index = 0; index < pyramid.size(); ++index)
        pyramid[index] = Buffer(device, std::vector<std::uint8_t>(
            subrect::kEarlyPyramidBytes[index], poison));
      id<MTLBuffer> vector_buffer = Buffer(
          device, std::vector<std::uint8_t>(subrect::kEarlyVectorBytes,
                                            poison));
      id<MTLBuffer> model_input = Buffer(
          device, std::vector<std::uint8_t>(subrect::kModelInputBytes,
                                            0u));

      std::vector<std::pair<std::string, std::string>> checkpoints;
      auto checkpoint = [&](std::uint32_t launch,
                            std::initializer_list<std::vector<std::uint8_t>>
                                parts) {
        std::ostringstream label;
        label << "launch-" << std::setfill('0') << std::setw(6) << launch;
        checkpoints.emplace_back(label.str(), Sha256(Join(parts)));
      };

      auto encode_m11 = [&](id<MTLTexture> source, id<MTLTexture> history,
                            id<MTLTexture> color,
                            id<MTLTexture> next_history) {
        const SubrectM11Parameters parameters{
            subrect::kResourceWidth, subrect::kResourceHeight,
            subrect::kOriginX, subrect::kOriginY,
            subrect::kWidth, subrect::kHeight,
            subrect::kEarlyWidth, subrect::kEarlyHeight, 2};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:m11];
        [encoder setTexture:source atIndex:0];
        [encoder setTexture:history atIndex:1];
        [encoder setTexture:color atIndex:2];
        [encoder setTexture:shared_metric atIndex:3];
        [encoder setTexture:next_history atIndex:4];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:
            MTLSizeMake(subrect::kEarlyWidth, subrect::kEarlyHeight, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        [encoder endEncoding];
        Wait(command);
      };

      auto encode_m15 = [&](id<MTLTexture> vector, id<MTLTexture> scalar,
                            id<MTLTexture> depth) {
        const SubrectM15Parameters parameters{
            subrect::kResourceWidth, subrect::kResourceHeight,
            subrect::kOriginX, subrect::kOriginY,
            subrect::kWidth, subrect::kHeight,
            subrect::kEarlyWidth, subrect::kEarlyHeight,
            {std::bit_cast<float>(0x3b000000u),
             std::bit_cast<float>(0x3b638e39u)},
            {std::bit_cast<float>(0x3b800000u),
             std::bit_cast<float>(0x3be38e39u)},
            {std::bit_cast<float>(0x3a800000u),
             std::bit_cast<float>(0x3ae38e39u)},
            {std::bit_cast<float>(0x3f7fc000u),
             std::bit_cast<float>(0x3f7f8e39u)},
            {std::bit_cast<float>(0x364ccccdu),
             std::bit_cast<float>(0x3721d13au)}};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:m15];
        [encoder setTexture:vector atIndex:0];
        [encoder setTexture:scalar atIndex:1];
        [encoder setTexture:flow atIndex:2];
        [encoder setTexture:depth atIndex:3];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:
            MTLSizeMake(subrect::kEarlyWidth, subrect::kEarlyHeight, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Wait(command);
      };

      std::size_t transient_canary_mismatches = 0;
      auto encode_m13 = [&](id<MTLTexture> depth, id<MTLTexture> mask) {
        const std::vector<std::uint8_t> initial(subrect::kEarlyDepthBytes,
                                                0xffu);
        id<MTLBuffer> high = Buffer(device, initial);
        id<MTLBuffer> low = Buffer(device, initial);
        const L5Parameters parameters{
            subrect::kEarlyWidth, subrect::kEarlyHeight,
            std::bit_cast<float>(0x42b851ecu),
            std::bit_cast<float>(0x4104b5ddu), 1.0f, 0};
        EncodeM13Pass(queue, m13_high, flow, depth, mask, high, low,
                      rom_buffer, parameters);
        EncodeM13Pass(queue, m13_low, flow, depth, mask, high, low,
                      rom_buffer, parameters);
        EncodeM13Pack(queue, m13_pack, high, low, atomic, parameters);
        transient_canary_mismatches += CanaryMismatches(high, initial.size());
        transient_canary_mismatches += CanaryMismatches(low, initial.size());
      };

      auto encode_m21 = [&](id<MTLTexture> opposite_depth,
                            id<MTLTexture> opposite_mask) {
        std::array<std::uint8_t, 224> parameters{};
        Store64(&parameters, 24, atomic.gpuAddress + kGuardBytes);
        for (std::size_t index = 0; index < pyramid.size(); ++index)
          Store64(&parameters, 48 + index * 8,
                  pyramid[index].gpuAddress + kGuardBytes);
        Store32(&parameters, 128, 0x3b000000u);
        Store32(&parameters, 132, 0x3b638e39u);
        Store32(&parameters, 136, subrect::kEarlyWidth);
        Store32(&parameters, 140, subrect::kEarlyHeight);
        Store32(&parameters, 144, 0x42200000u);
        Store32(&parameters, 148, 0x3f800000u);
        Store32(&parameters, 152, 0x44160000u);
        parameters[156] = 0;
        parameters[157] = 1;
        const std::array<float, 16> identity = {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        std::memcpy(parameters.data() + 160, identity.data(), 64);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:m21];
        [encoder setBytes:parameters.data() length:parameters.size()
                     atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        [encoder setTexture:flow atIndex:0];
        [encoder setTexture:opposite_depth atIndex:1];
        [encoder setTexture:opposite_mask atIndex:2];
        [encoder setTexture:working_vector atIndex:3];
        [encoder setTexture:working_mask atIndex:4];
        [encoder useResource:atomic usage:MTLResourceUsageRead];
        for (id<MTLBuffer> item : pyramid)
          [encoder useResource:item usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(16, 9, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        Wait(command);
      };

      auto encode_m23 = [&] {
        BlitTextureToBuffer(queue, working_vector, vector_buffer, 4);
        std::array<std::uint8_t, 72> parameters{};
        Store64(&parameters, 8, pyramid[1].gpuAddress + kGuardBytes);
        Store64(&parameters, 16, pyramid[2].gpuAddress + kGuardBytes);
        Store64(&parameters, 24, pyramid[3].gpuAddress + kGuardBytes);
        Store64(&parameters, 32, pyramid[4].gpuAddress + kGuardBytes);
        Store32(&parameters, 40, subrect::kEarlyWidth);
        Store32(&parameters, 44, subrect::kEarlyHeight);
        Store32(&parameters, 48, 1);
        Store64(&parameters, 56, vector_buffer.gpuAddress + kGuardBytes);
        Store64(&parameters, 64, pyramid[0].gpuAddress + kGuardBytes);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:m23];
        [encoder setBytes:parameters.data() length:parameters.size()
                     atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        for (id<MTLBuffer> item : pyramid)
          [encoder useResource:item usage:MTLResourceUsageRead];
        [encoder useResource:vector_buffer usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(16, 9, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        Wait(command);
        BlitBufferToTexture(queue, vector_buffer, working_vector, 4);
      };

      auto encode_m25 = [&](id<MTLTexture> scatter) {
        const L8Parameters parameters{
            subrect::kEarlyWidth, subrect::kEarlyHeight, 0};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:m25];
        [encoder setTexture:working_vector atIndex:0];
        [encoder setTexture:scatter atIndex:1];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:
            MTLSizeMake(subrect::kEarlyWidth, subrect::kEarlyHeight, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Wait(command);
      };

      encode_m11(source_reset, history_a, reset_color, history_b);
      const auto reset_color_bytes = TextureBytes(reset_color, 8);
      const auto reset_metric_bytes = TextureBytes(shared_metric, 2);
      const auto reset_history_region =
          subrect::HistoryRegion(TextureBytes(history_b, 8), 8);
      Write(output_root / "debug-reset-color.rgba16f", reset_color_bytes);
      Write(output_root / "debug-reset-metric.r16f", reset_metric_bytes);
      Write(output_root / "debug-reset-history.rgba16f", reset_history_region);
      checkpoint(3, {reset_color_bytes, reset_metric_bytes,
                     reset_history_region});
      encode_m15(vector_reset, scalar_reset, reset_depth);
      checkpoint(4, {TextureBytes(flow, 8), TextureBytes(reset_depth, 4)});
      encode_m13(reset_depth, reset_mask);
      checkpoint(5, {Payload(atomic, subrect::kEarlyAtomicBytes),
                     TextureBytes(reset_mask, 1)});
      encode_m21(generated_depth, generated_mask);
      checkpoint(6, {TextureBytes(working_vector, 4),
                     TextureBytes(working_mask, 1),
                     Payload(pyramid[0], subrect::kEarlyPyramidBytes[0]),
                     Payload(pyramid[1], subrect::kEarlyPyramidBytes[1]),
                     Payload(pyramid[2], subrect::kEarlyPyramidBytes[2]),
                     Payload(pyramid[3], subrect::kEarlyPyramidBytes[3]),
                     Payload(pyramid[4], subrect::kEarlyPyramidBytes[4])});
      encode_m23();
      checkpoint(7, {TextureBytes(working_vector, 4)});
      encode_m25(scatter_reset);
      checkpoint(8, {TextureBytes(scatter_reset, 1)});

      const L9Parameters m14_parameters{
          subrect::kEarlyWidth, subrect::kEarlyHeight};
      id<MTLCommandBuffer> m14_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> m14_encoder =
          [m14_command computeCommandEncoder];
      [m14_encoder setComputePipelineState:m14];
      [m14_encoder setTexture:reset_color atIndex:0];
      [m14_encoder setTexture:scatter_reset atIndex:1];
      [m14_encoder setTexture:reset_projection atIndex:2];
      [m14_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
      [m14_encoder setBytes:&m14_parameters length:sizeof(m14_parameters)
                     atIndex:1];
      [m14_encoder dispatchThreads:
          MTLSizeMake(subrect::kEarlyWidth, subrect::kEarlyHeight, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m14_encoder endEncoding];
      Wait(m14_command);
      checkpoint(9, {TextureBytes(reset_projection, 4)});

      const L11Parameters m4_parameters{
          subrect::kOriginX, subrect::kOriginY,
          subrect::kOriginX, subrect::kOriginY,
          subrect::kWidth, subrect::kHeight};
      id<MTLCommandBuffer> m4_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> m4_encoder =
          [m4_command computeCommandEncoder];
      [m4_encoder setComputePipelineState:m4];
      [m4_encoder setTexture:source_reset atIndex:0];
      [m4_encoder setTexture:reset_present atIndex:1];
      [m4_encoder setBytes:&m4_parameters length:sizeof(m4_parameters)
                    atIndex:0];
      [m4_encoder dispatchThreads:MTLSizeMake(subrect::kWidth,
                                              subrect::kHeight, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m4_encoder endEncoding];
      Wait(m4_command);
      checkpoint(10, {subrect::FullRegion(TextureBytes(reset_present, 4), 4)});

      encode_m11(source_generated, history_b, generated_color, history_a);
      const auto generated_color_bytes = TextureBytes(generated_color, 8);
      const auto generated_metric_bytes = TextureBytes(shared_metric, 2);
      const auto generated_history_region =
          subrect::HistoryRegion(TextureBytes(history_a, 8), 8);
      Write(output_root / "debug-generated-color.rgba16f",
            generated_color_bytes);
      Write(output_root / "debug-generated-metric.r16f",
            generated_metric_bytes);
      Write(output_root / "debug-generated-history.rgba16f",
            generated_history_region);
      checkpoint(11, {generated_color_bytes, generated_metric_bytes,
                      generated_history_region});
      encode_m15(vector_generated, scalar_generated, generated_depth);
      checkpoint(12, {TextureBytes(flow, 8),
                      TextureBytes(generated_depth, 4)});
      ResetPayload(atomic, subrect::InitialAtomic());
      encode_m13(generated_depth, generated_mask);
      checkpoint(13, {Payload(atomic, subrect::kEarlyAtomicBytes),
                      TextureBytes(generated_mask, 1)});
      encode_m21(reset_depth, reset_mask);
      checkpoint(14, {TextureBytes(working_vector, 4),
                      TextureBytes(working_mask, 1),
                      Payload(pyramid[0], subrect::kEarlyPyramidBytes[0]),
                      Payload(pyramid[1], subrect::kEarlyPyramidBytes[1]),
                      Payload(pyramid[2], subrect::kEarlyPyramidBytes[2]),
                      Payload(pyramid[3], subrect::kEarlyPyramidBytes[3]),
                      Payload(pyramid[4], subrect::kEarlyPyramidBytes[4])});
      encode_m23();
      checkpoint(15, {TextureBytes(working_vector, 4)});
      encode_m25(scatter_generated);
      checkpoint(16, {TextureBytes(scatter_generated, 1)});

      const std::array<id<MTLTexture>, 10> m5_inputs = {
          reset_color, generated_color, scatter_generated, scatter_reset,
          reset_projection, shared_metric, allocation29, flow,
          working_vector, working_mask};
      const SubrectM5Parameters m5_parameters{
          subrect::kEarlyWidth, subrect::kEarlyHeight,
          subrect::kEarlyWidth, subrect::kEarlyHeight,
          subrect::kTensorWidth, subrect::kTensorHeight, 0, -24};
      id<MTLCommandBuffer> m5_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> m5_surface_encoder =
          [m5_command computeCommandEncoder];
      [m5_surface_encoder setComputePipelineState:m5_surfaces];
      for (NSUInteger index = 0; index < m5_inputs.size(); ++index)
        [m5_surface_encoder setTexture:m5_inputs[index] atIndex:index];
      [m5_surface_encoder setTexture:auxiliary_mask atIndex:10];
      [m5_surface_encoder setTexture:auxiliary_vector atIndex:11];
      [m5_surface_encoder setBytes:&m5_parameters length:sizeof(m5_parameters)
                           atIndex:0];
      [m5_surface_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [m5_surface_encoder dispatchThreads:
          MTLSizeMake(subrect::kEarlyWidth, 120, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m5_surface_encoder endEncoding];
      id<MTLComputeCommandEncoder> m5_tensor_encoder =
          [m5_command computeCommandEncoder];
      [m5_tensor_encoder setComputePipelineState:m5_tensor];
      for (NSUInteger index = 0; index < m5_inputs.size(); ++index)
        [m5_tensor_encoder setTexture:m5_inputs[index] atIndex:index];
      [m5_tensor_encoder setBuffer:model_input offset:kGuardBytes atIndex:0];
      [m5_tensor_encoder setBytes:&m5_parameters length:sizeof(m5_parameters)
                          atIndex:1];
      [m5_tensor_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
      [m5_tensor_encoder dispatchThreads:
          MTLSizeMake(subrect::kTensorWidth, 60, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m5_tensor_encoder endEncoding];
      Wait(m5_command);
      const auto model_input_bytes =
          Payload(model_input, subrect::kModelInputBytes);
      const auto auxiliary_mask_bytes = TextureBytes(auxiliary_mask, 1);
      const auto auxiliary_vector_bytes = TextureBytes(auxiliary_vector, 4);
      checkpoint(17, {model_input_bytes,
                      subrect::Module5ActiveRows(auxiliary_mask_bytes, 1),
                      subrect::Module5ActiveRows(auxiliary_vector_bytes, 4)});

      std::size_t input_mismatches = 0;
      input_mismatches += Mismatches(external.source_reset,
                                     TextureBytes(source_reset, 4));
      input_mismatches += Mismatches(external.source_generated,
                                     TextureBytes(source_generated, 4));
      input_mismatches += Mismatches(external.vector_reset,
                                     TextureBytes(vector_reset, 4));
      input_mismatches += Mismatches(external.vector_generated,
                                     TextureBytes(vector_generated, 4));
      input_mismatches += Mismatches(external.scalar_reset,
                                     TextureBytes(scalar_reset, 4));
      input_mismatches += Mismatches(external.scalar_generated,
                                     TextureBytes(scalar_generated, 4));
      input_mismatches += Mismatches(subrect::InitialAllocation29(),
                                     TextureBytes(allocation29, 2));
      input_mismatches += Mismatches(rom, Payload(rom_buffer, rom.size()));
      const auto history_a_after = TextureBytes(history_a, 8);
      const auto history_b_after = TextureBytes(history_b, 8);
      const auto reset_present_after = TextureBytes(reset_present, 4);
      const std::size_t outside_mismatches =
          OutsideMismatches(history_a_initial, history_a_after, 8, 0, 0) +
          OutsideMismatches(history_b_initial, history_b_after, 8, 0, 0) +
          OutsideMismatches(reset_present_initial, reset_present_after, 4,
                            subrect::kOriginX, subrect::kOriginY);
      std::size_t canary_mismatches = transient_canary_mismatches;
      canary_mismatches += CanaryMismatches(rom_buffer, rom.size());
      canary_mismatches += CanaryMismatches(atomic,
                                            subrect::kEarlyAtomicBytes);
      canary_mismatches += CanaryMismatches(vector_buffer,
                                            subrect::kEarlyVectorBytes);
      canary_mismatches += CanaryMismatches(model_input,
                                            subrect::kModelInputBytes);
      for (std::size_t index = 0; index < pyramid.size(); ++index)
        canary_mismatches += CanaryMismatches(
            pyramid[index], subrect::kEarlyPyramidBytes[index]);

#ifdef FG_SUBRECT_FULL_CHAIN
      FgSubrectSetMacLivePrefixInput(model_input);
      const fs::path tail_output = output_root / "l18-l83";
      std::string output_argument = tail_output.string();
      std::array<const char*, 10> tail_argv = {
          argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
          output_argument.c_str(), argv[8], argv[9]};
      const int tail_status = FgSubrectTailMain(
          static_cast<int>(tail_argv.size()), tail_argv.data());
      FgSubrectSetMacLivePrefixInput(nil);
      if (tail_status != 0)
        throw std::runtime_error("connected subrect L18-L83 tail failed");
      input_mismatches += Mismatches(
          model_input_bytes, Payload(model_input, subrect::kModelInputBytes));
#endif

      std::size_t tensor_padding_nonzero_bytes = 0;
      for (std::uint32_t y = 0; y < subrect::kTensorHeight; ++y) {
        if (y >= 12u && y < 72u) continue;
        const auto begin = std::size_t{y} * subrect::kTensorWidth * 48u * 2u;
        const auto end = begin + std::size_t{subrect::kTensorWidth} * 48u * 2u;
        for (std::size_t index = begin; index < end; ++index)
          tensor_padding_nonzero_bytes += model_input_bytes[index] != 0u;
      }
      const std::size_t auxiliary_untouched_row_mismatches =
          subrect::Module5UntouchedRowMismatches(
              auxiliary_mask_bytes, poison, 1) +
          subrect::Module5UntouchedRowMismatches(
              auxiliary_vector_bytes, poison, 4);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                        outside_mismatches == 0 &&
                        tensor_padding_nonzero_bytes == 0 &&
                        auxiliary_untouched_row_mismatches == 0 &&
                        checkpoints.size() == 15;
      Write(output_root / "allocation37.fp16", model_input_bytes);
      Write(output_root / "allocation33.r8", auxiliary_mask_bytes);
      Write(output_root / "allocation27.rg16f", auxiliary_vector_bytes);
      Write(output_root / "reset-present-region.rgba8",
            subrect::FullRegion(reset_present_after, 4));
      std::ofstream receipt(output_root / "early-run-receipt.json");
      if (!receipt) throw std::runtime_error("receipt open failed");
      receipt << "{\n  \"schema\": "
                 "\"neuralstudio.fg.connected-subrect-l3-l17-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << argv[8] << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"region\": [32, 24, 512, 288],\n"
              << "  \"launch_range\": [3, 17],\n"
              << "  \"launch_count\": 15,\n"
              << "  \"module22_module24_status_executed\": false,\n"
              << "  \"checkpoint_sha256\": {";
      for (std::size_t index = 0; index < checkpoints.size(); ++index)
        receipt << (index ? "," : "") << "\n    \""
                << checkpoints[index].first << "\": \""
                << checkpoints[index].second << "\"";
      receipt << "\n  },\n"
              << "  \"allocation37_bytes\": " << model_input_bytes.size()
              << ",\n  \"allocation37_sha256\": \""
              << Sha256(model_input_bytes) << "\",\n"
              << "  \"auxiliary_mask_sha256\": \""
              << Sha256(auxiliary_mask_bytes) << "\",\n"
              << "  \"auxiliary_vector_sha256\": \""
              << Sha256(auxiliary_vector_bytes) << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches
              << ",\n  \"canary_mismatched_bytes\": " << canary_mismatches
              << ",\n  \"outside_region_mismatched_bytes\": "
              << outside_mismatches << ",\n"
              << "  \"tensor_padding_nonzero_bytes\": "
              << tensor_padding_nonzero_bytes << ",\n"
              << "  \"tensor_active_rows\": 60,\n"
              << "  \"auxiliary_active_rows\": 120,\n"
              << "  \"auxiliary_untouched_row_mismatched_bytes\": "
              << auxiliary_untouched_row_mismatches << ",\n"
              << "  \"raw_parameter_origin\": [32, 24],\n"
              << "  \"module23_mode\": 1,\n"
              << "  \"allocation29_runtime_source\": "
                 "\"generated-half-0.5\",\n"
              << "  \"expected_or_archived_intermediate_ingress\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
#ifdef FG_SUBRECT_FULL_CHAIN
              << "  \"connected_tail_receipt\": "
                 "\"l18-l83/run-receipt.json\",\n"
#endif
              << "  \"result\": \"" << (pass ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "allocation37_sha256=" << Sha256(model_input_bytes)
                << "\ncheckpoint_count=" << checkpoints.size()
                << "\nresult=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
