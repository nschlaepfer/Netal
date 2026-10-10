#define main FgConnectedDynamicEarlyEmbeddedFixedMain
#include "../fg_connected_l3_l19/macos_connected.mm"
#undef main

#include "contract.h"

namespace {
namespace dyn = fg_connected_dynamic_early;

struct Dynamic11Parameters {
  std::uint32_t source_width, source_height, output_width, output_height, mode;
};
struct Dynamic15Parameters {
  std::uint32_t input_width, input_height, output_width, output_height;
  std::array<float, 2> input_inverse, output_inverse;
  std::array<float, 2> lower_coordinate, upper_coordinate, output_scale;
};
struct Dynamic5Parameters {
  std::uint32_t motion_width, motion_height, work_width, work_height;
  std::uint32_t tensor_width, tensor_height;
  std::int32_t input_x_offset, input_y_offset;
};
struct Dynamic25Parameters { std::uint32_t width, height, mode; };
struct Dynamic14Parameters { std::uint32_t width, height; };

void SetPayload(id<MTLBuffer> buffer,
                const std::vector<std::uint8_t>& bytes) {
  if (buffer.length != bytes.size() + 2u * kGuardBytes)
    throw std::runtime_error("buffer reset size drift");
  std::memcpy(static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes,
              bytes.data(), bytes.size());
}

void EncodeModule13Pass(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
    id<MTLTexture> flow, id<MTLTexture> depth, id<MTLTexture> mask,
    id<MTLBuffer> high, id<MTLBuffer> low, id<MTLBuffer> rom,
    const L5Parameters& parameters) {
  auto command = [queue commandBuffer];
  auto encoder = [command computeCommandEncoder];
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

void EncodeModule13Pack(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> high, id<MTLBuffer> low, id<MTLBuffer> output,
    const L5Parameters& parameters) {
  auto command = [queue commandBuffer];
  auto encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:high offset:kGuardBytes atIndex:0];
  [encoder setBuffer:low offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
  [encoder dispatchThreads:
      MTLSizeMake(std::size_t{parameters.width} * parameters.height, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [encoder endEncoding];
  Wait(command);
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-connected-dynamic-early-mac <libs> <rom> "
                   "<output> <run-id> <poison> <resolution> "
                   "<full|hash-only>\n";
      return 2;
    }
    try {
      const fs::path libs = argv[1];
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) !=
          "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750")
        throw std::runtime_error("RCP ROM identity mismatch");
      const fs::path output_root = argv[3];
      fs::create_directories(output_root);
      const std::string run_id = argv[4];
      const auto parsed_poison = std::stoul(argv[5], nullptr, 16);
      if (parsed_poison > 255u) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const auto& resolution = dyn::Find(argv[6]);
      const std::string artifact_mode = argv[7];
      if (artifact_mode != "full" && artifact_mode != "hash-only")
        throw std::runtime_error("invalid artifact mode");
      const bool full_artifacts = artifact_mode == "full";
      const auto& r15 = dyn::Module15(resolution);
      const auto& r13 = dyn::Module13(resolution);
      const auto& r21 = dyn::Module21(resolution);
      const auto& r23 = dyn::Module23(resolution);
      const auto external = dyn::GenerateExternal(resolution);
      const auto pyramid_bytes = dyn::PyramidBytes(resolution);
      const auto extra_bytes = dyn::ExtraPyramidBytes(resolution);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (device == nil || queue == nil)
        throw std::runtime_error("Metal unavailable");
      auto m11 = Pipeline(device, libs / "module11.metallib",
                          "fg_dynamic_module11");
      auto m15 = Pipeline(device, libs / "module15.metallib",
                          "fg_dynamic_module15");
      auto m13_high = Pipeline(device, libs / "module13.metallib",
                               "fg_early_l5_l15_high");
      auto m13_low = Pipeline(device, libs / "module13.metallib",
                              "fg_early_l5_l15_low");
      auto m13_pack = Pipeline(device, libs / "module13.metallib",
                               "fg_early_l5_l15_pack");
      auto m21 = Pipeline(device, libs / "module21.metallib",
                          "fg_early_l6_l16");
      auto m22 = Pipeline(device, libs / "module22.metallib",
                          "fg_resolution_module22");
      auto m23 = Pipeline(device, libs / "module23.metallib",
                          "fg_dynamic_module23_mode0");
      auto m24 = Pipeline(device, libs / "module24.metallib",
          resolution.selected_level == 10 ? "fg_resolution_module24_level10"
                                          : "fg_resolution_module24_level11");
      auto m25 = Pipeline(device, libs / "module25.metallib",
                          "fg_early_l8_l18");
      auto m14 = Pipeline(device, libs / "module14.metallib",
                          "fg_dynamic_module14");
      auto status = Pipeline(device, libs / "status.metallib",
                             "fg_early_status_write");
      auto m4 = Pipeline(device, libs / "module4.metallib", "fg_early_l11");
      auto m5_surfaces = Pipeline(
          device, libs / "module5.metallib",
          "fg_connected_dynamic_module5_surfaces");
      auto m5_tensor = Pipeline(
          device, libs / "module5.metallib",
          "fg_connected_dynamic_module5_tensor");

      id<MTLBuffer> rom_buffer = Buffer(device, rom);
      auto source_reset = Texture(
          device, MTLPixelFormatRGBA8Unorm, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderRead,
          external.source_reset, 4);
      auto source_generated = Texture(
          device, MTLPixelFormatRGBA8Unorm, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderRead,
          external.source_generated, 4);
      auto history_a = Texture(
          device, MTLPixelFormatRGBA16Float, resolution.source_width,
          resolution.source_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          dyn::History(resolution, 0), 8);
      auto history_b = Texture(
          device, MTLPixelFormatRGBA16Float, resolution.source_width,
          resolution.source_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          dyn::History(resolution, poison), 8);
      auto reset_color = Texture(
          device, MTLPixelFormatRGBA16Float, resolution.color_width,
          resolution.color_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(dyn::ColorBytes(resolution), poison), 8);
      auto generated_color = Texture(
          device, MTLPixelFormatRGBA16Float, resolution.color_width,
          resolution.color_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(dyn::ColorBytes(resolution), poison), 8);
      auto shared_metric = Texture(
          device, MTLPixelFormatR16Float, resolution.color_width,
          resolution.color_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(dyn::MetricBytes(resolution), poison), 2);
      auto vector_reset = Texture(
          device, MTLPixelFormatRG16Float, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderRead,
          external.vector_reset, 4);
      auto vector_generated = Texture(
          device, MTLPixelFormatRG16Float, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderRead,
          external.vector_generated, 4);
      auto scalar_reset = Texture(
          device, MTLPixelFormatR32Float, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderRead,
          external.scalar_reset, 4);
      auto scalar_generated = Texture(
          device, MTLPixelFormatR32Float, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderRead,
          external.scalar_generated, 4);
      const auto motion_pixels = dyn::MotionPixels(resolution);
      auto flow = Texture(
          device, MTLPixelFormatRG32Float, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels * 8u, poison), 8);
      auto reset_depth = Texture(
          device, MTLPixelFormatR32Uint, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels * 4u, poison), 4);
      auto generated_depth = Texture(
          device, MTLPixelFormatR32Uint, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          dyn::InitialGeneratedDepth(resolution), 4);
      auto reset_mask = Texture(
          device, MTLPixelFormatR8Unorm, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels, poison), 1);
      auto generated_mask = Texture(
          device, MTLPixelFormatR8Unorm, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          dyn::InitialGeneratedMask(resolution), 1);
      auto working_vector = Texture(
          device, MTLPixelFormatRG16Float, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels * 4u, poison), 4);
      auto working_mask = Texture(
          device, MTLPixelFormatR8Unorm, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels, poison), 1);
      auto scatter_reset = Texture(
          device, MTLPixelFormatR8Unorm, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels, 255u), 1);
      auto scatter_generated = Texture(
          device, MTLPixelFormatR8Unorm, resolution.motion_width,
          resolution.motion_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(motion_pixels, 255u), 1);
      const std::size_t color_pixels =
          std::size_t{resolution.color_width} * resolution.color_height;
      auto reset_projection = Texture(
          device, MTLPixelFormatRG16Float, resolution.color_width,
          resolution.color_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(color_pixels * 4u, poison), 4);
      auto allocation29 = Texture(
          device, MTLPixelFormatR16Float, resolution.color_width,
          resolution.color_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          dyn::InitialAllocation29(resolution), 2);
      auto auxiliary_mask = Texture(
          device, MTLPixelFormatR8Unorm, resolution.color_width,
          resolution.color_height, MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(color_pixels, poison), 1);
      auto auxiliary_vector = Texture(
          device, MTLPixelFormatRG16Float, resolution.color_width,
          resolution.color_height, MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(color_pixels * 4u, poison), 4);
      auto reset_present = Texture(
          device, MTLPixelFormatRGBA8Unorm, resolution.source_width,
          resolution.source_height, MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(
              std::size_t{resolution.source_width} * resolution.source_height *
              4u, poison), 4);
      id<MTLBuffer> atomic = Buffer(device, dyn::InitialAtomic(resolution));
      std::array<id<MTLBuffer>, 5> pyramid;
      for (std::size_t index = 0; index < pyramid.size(); ++index)
        pyramid[index] = Buffer(
            device, std::vector<std::uint8_t>(pyramid_bytes[index], poison));
      std::array<id<MTLBuffer>, 4> extra_pyramid;
      for (std::size_t index = 0; index < extra_pyramid.size(); ++index)
        extra_pyramid[index] = Buffer(
            device, std::vector<std::uint8_t>(extra_bytes[index], poison));
      id<MTLBuffer> model_input = Buffer(
          device, std::vector<std::uint8_t>(
                      dyn::ModelInputBytes(resolution), poison));
      id<MTLBuffer> status_buffer = Buffer(
          device, std::vector<std::uint8_t>(4u, poison));

      std::vector<std::pair<std::string, std::string>> checkpoints;
      auto checkpoint = [&](const char* name,
                            std::initializer_list<std::vector<std::uint8_t>> parts) {
        checkpoints.emplace_back(name, Sha256(Join(parts)));
      };
      auto encode_m11 = [&](id<MTLTexture> source, id<MTLTexture> history,
                            id<MTLTexture> color, id<MTLTexture> next_history) {
        const Dynamic11Parameters parameters{
            resolution.source_width, resolution.source_height,
            resolution.color_width, resolution.color_height,
            resolution.module11_mode};
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m11];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder setTexture:source atIndex:0];
        [encoder setTexture:history atIndex:1];
        [encoder setTexture:color atIndex:2];
        [encoder setTexture:shared_metric atIndex:3];
        [encoder setTexture:next_history atIndex:4];
        [encoder dispatchThreads:
            MTLSizeMake(resolution.color_width, resolution.color_height, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      auto encode_m15 = [&](id<MTLTexture> vector, id<MTLTexture> scalar,
                            id<MTLTexture> depth) {
        const Dynamic15Parameters parameters{
            r15.input_width, r15.input_height,
            r15.output_width, r15.output_height,
            {std::bit_cast<float>(r15.inverse_x_bits),
             std::bit_cast<float>(r15.inverse_y_bits)},
            {std::bit_cast<float>(r15.inverse_x_bits + 0x00800000u),
             std::bit_cast<float>(r15.inverse_y_bits + 0x00800000u)},
            {std::bit_cast<float>(r15.lower_x_bits),
             std::bit_cast<float>(r15.lower_y_bits)},
            {std::bit_cast<float>(r15.upper_x_bits),
             std::bit_cast<float>(r15.upper_y_bits)},
            {std::bit_cast<float>(r15.output_scale_x_bits),
             std::bit_cast<float>(r15.output_scale_y_bits)}};
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m15];
        [encoder setTexture:vector atIndex:0];
        [encoder setTexture:scalar atIndex:1];
        [encoder setTexture:flow atIndex:2];
        [encoder setTexture:depth atIndex:3];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:
            MTLSizeMake(resolution.motion_width, resolution.motion_height, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      std::size_t transient_canary_mismatches = 0;
      auto encode_m13 = [&](id<MTLTexture> depth, id<MTLTexture> mask) {
        const std::vector<std::uint8_t> words(motion_pixels * 4u, 0xffu);
        id<MTLBuffer> high = Buffer(device, words);
        id<MTLBuffer> low = Buffer(device, words);
        const L5Parameters parameters{
            resolution.motion_width, resolution.motion_height,
            std::bit_cast<float>(r13.motion_divisor_bits),
            std::bit_cast<float>(r13.neighbor_divisor_bits), 1.0f, 0u};
        EncodeModule13Pass(queue, m13_high, flow, depth, mask,
                           high, low, rom_buffer, parameters);
        EncodeModule13Pass(queue, m13_low, flow, depth, mask,
                           high, low, rom_buffer, parameters);
        EncodeModule13Pack(queue, m13_pack, high, low, atomic, parameters);
        transient_canary_mismatches += CanaryMismatches(high, words.size());
        transient_canary_mismatches += CanaryMismatches(low, words.size());
      };
      auto encode_m21 = [&](id<MTLTexture> opposite_depth,
                            id<MTLTexture> opposite_mask) {
        std::array<std::uint8_t, 224> parameters{};
        Store64(&parameters, 24, atomic.gpuAddress + kGuardBytes);
        for (std::size_t index = 0; index < pyramid.size(); ++index)
          Store64(&parameters, 48 + index * 8,
                  pyramid[index].gpuAddress + kGuardBytes);
        Store32(&parameters, 128, r21.invx);
        Store32(&parameters, 132, r21.invy);
        Store32(&parameters, 136, resolution.motion_width);
        Store32(&parameters, 140, resolution.motion_height);
        Store32(&parameters, 144, 0x42200000u);
        Store32(&parameters, 148, 0x3f800000u);
        Store32(&parameters, 152, 0x44160000u);
        parameters[156] = 0;
        parameters[157] = static_cast<std::uint8_t>(r21.padding157);
        const std::array<float, 16> identity = {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        std::memcpy(parameters.data() + 160, identity.data(), 64);
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m21];
        [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        [encoder setTexture:flow atIndex:0];
        [encoder setTexture:opposite_depth atIndex:1];
        [encoder setTexture:opposite_mask atIndex:2];
        [encoder setTexture:working_vector atIndex:3];
        [encoder setTexture:working_mask atIndex:4];
        [encoder useResource:atomic usage:MTLResourceUsageRead];
        for (id<MTLBuffer> item : pyramid)
          [encoder useResource:item usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(r21.gx, r21.gy, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      auto encode_m22 = [&] {
        const auto input_width = dyn::CeilDiv(resolution.motion_width, 16u);
        const auto input_height = dyn::CeilDiv(resolution.motion_height, 16u);
        std::array<std::uint8_t, 56> parameters{};
        Store64(&parameters, 0, pyramid[4].gpuAddress + kGuardBytes);
        for (std::size_t index = 0; index < extra_pyramid.size(); ++index)
          Store64(&parameters, 8 + index * 8,
                  extra_pyramid[index].gpuAddress + kGuardBytes);
        Store32(&parameters, 40, resolution.extra_level_count);
        Store32(&parameters, 44, input_width);
        Store32(&parameters, 48, input_height);
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m22];
        [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        [encoder useResource:pyramid[4] usage:MTLResourceUsageRead];
        for (id<MTLBuffer> item : extra_pyramid)
          [encoder useResource:item usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:
            MTLSizeMake(dyn::CeilDiv(input_width, 16u),
                        dyn::CeilDiv(input_height, 16u), 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      std::array<id<MTLBuffer>, 5> selected_levels{};
      if (resolution.selected_level == 10) {
        selected_levels = {pyramid[1], pyramid[2], pyramid[3], pyramid[4],
                           extra_pyramid[0]};
      } else {
        selected_levels = {pyramid[2], pyramid[3], pyramid[4], extra_pyramid[0],
                           extra_pyramid[1]};
      }
      auto encode_m23 = [&] {
        std::array<std::uint8_t, 72> parameters{};
        for (std::size_t index = 0; index < selected_levels.size(); ++index)
          Store64(&parameters, index * 8,
                  selected_levels[index].gpuAddress + kGuardBytes);
        Store32(&parameters, 40, resolution.module23_width);
        Store32(&parameters, 44, resolution.module23_height);
        Store32(&parameters, 48, 0);
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m23];
        [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        for (id<MTLBuffer> item : selected_levels)
          [encoder useResource:item
                         usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(r23.gx, r23.gy, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      auto encode_m24 = [&] {
        std::array<std::uint8_t, 64> parameters{};
        Store64(&parameters, 8, pyramid[0].gpuAddress + kGuardBytes);
        for (std::size_t index = 0; index < 4; ++index)
          Store64(&parameters, 16 + index * 8,
                  pyramid[index + 1].gpuAddress + kGuardBytes);
        Store32(&parameters, 48, resolution.selected_level);
        Store32(&parameters, 52, resolution.motion_width);
        Store32(&parameters, 56, resolution.motion_height);
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m24];
        [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        [encoder setTexture:working_vector atIndex:0];
        [encoder useResource:pyramid[0] usage:MTLResourceUsageRead];
        for (std::size_t index = 0; index < 4; ++index)
          [encoder useResource:pyramid[index + 1]
                         usage:MTLResourceUsageRead];
        [encoder dispatchThreadgroups:
            MTLSizeMake(dyn::CeilDiv(resolution.motion_width, 16u),
                        dyn::CeilDiv(resolution.motion_height, 16u), 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      auto encode_m25 = [&](id<MTLTexture> scatter) {
        const Dynamic25Parameters parameters{
            resolution.motion_width, resolution.motion_height, 0};
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:m25];
        [encoder setTexture:working_vector atIndex:0];
        [encoder setTexture:scatter atIndex:1];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:
            MTLSizeMake(resolution.motion_width, resolution.motion_height, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        Wait(command);
      };
      auto encode_status = [&](std::uint32_t value) {
        const StatusParameters parameters{1, value};
        auto command = [queue commandBuffer];
        auto encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:status];
        [encoder setBuffer:status_buffer offset:kGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        Wait(command);
      };

      encode_m11(source_reset, history_a, reset_color, history_b);
      checkpoint("reset-module11", {TextureBytes(reset_color, 8),
          TextureBytes(shared_metric, 2), TextureBytes(history_b, 8)});
      encode_m15(vector_reset, scalar_reset, reset_depth);
      checkpoint("reset-module15", {TextureBytes(flow, 8),
          TextureBytes(reset_depth, 4)});
      encode_m13(reset_depth, reset_mask);
      checkpoint("reset-module13", {Payload(atomic, dyn::InitialAtomic(resolution).size()),
          TextureBytes(reset_mask, 1)});
      encode_m21(generated_depth, generated_mask);
      checkpoint("reset-module21", {TextureBytes(working_vector, 4),
          TextureBytes(working_mask, 1), Payload(pyramid[0], pyramid_bytes[0]),
          Payload(pyramid[1], pyramid_bytes[1]), Payload(pyramid[2], pyramid_bytes[2]),
          Payload(pyramid[3], pyramid_bytes[3]), Payload(pyramid[4], pyramid_bytes[4])});
      encode_m22();
      if (resolution.extra_level_count == 1)
        checkpoint("reset-module22", {Payload(extra_pyramid[0], extra_bytes[0])});
      else
        checkpoint("reset-module22", {Payload(extra_pyramid[0], extra_bytes[0]),
                                       Payload(extra_pyramid[1], extra_bytes[1])});
      encode_m23();
      checkpoint("reset-module23", {Payload(selected_levels[0],
          resolution.selected_level == 10 ? pyramid_bytes[1] : pyramid_bytes[2])});
      encode_m24();
      checkpoint("reset-module24", {TextureBytes(working_vector, 4)});
      encode_m25(scatter_reset);
      checkpoint("reset-module25", {TextureBytes(scatter_reset, 1)});

      const Dynamic14Parameters m14_parameters{
          resolution.color_width, resolution.color_height};
      auto m14_command = [queue commandBuffer];
      auto m14_encoder = [m14_command computeCommandEncoder];
      [m14_encoder setComputePipelineState:m14];
      [m14_encoder setTexture:reset_color atIndex:0];
      [m14_encoder setTexture:scatter_reset atIndex:1];
      [m14_encoder setTexture:reset_projection atIndex:2];
      [m14_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
      [m14_encoder setBytes:&m14_parameters length:sizeof(m14_parameters)
                     atIndex:1];
      [m14_encoder dispatchThreads:
          MTLSizeMake(resolution.color_width, resolution.color_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m14_encoder endEncoding];
      Wait(m14_command);
      checkpoint("reset-module14", {TextureBytes(reset_projection, 4)});
      encode_status(1);
      checkpoint("reset-module27", {Payload(status_buffer, 4)});
      const L11Parameters m4_parameters{
          0, 0, 0, 0, resolution.source_width, resolution.source_height};
      auto m4_command = [queue commandBuffer];
      auto m4_encoder = [m4_command computeCommandEncoder];
      [m4_encoder setComputePipelineState:m4];
      [m4_encoder setTexture:source_reset atIndex:0];
      [m4_encoder setTexture:reset_present atIndex:1];
      [m4_encoder setBytes:&m4_parameters length:sizeof(m4_parameters) atIndex:0];
      [m4_encoder dispatchThreads:
          MTLSizeMake(resolution.source_width, resolution.source_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m4_encoder endEncoding];
      Wait(m4_command);
      checkpoint("reset-module4", {TextureBytes(reset_present, 4)});

      encode_status(0);
      checkpoint("generated-module27", {Payload(status_buffer, 4)});
      encode_m11(source_generated, history_b, generated_color, history_a);
      checkpoint("generated-module11", {TextureBytes(generated_color, 8),
          TextureBytes(shared_metric, 2), TextureBytes(history_a, 8)});
      encode_m15(vector_generated, scalar_generated, generated_depth);
      checkpoint("generated-module15", {TextureBytes(flow, 8),
          TextureBytes(generated_depth, 4)});
      SetPayload(atomic, dyn::InitialAtomic(resolution));
      encode_m13(generated_depth, generated_mask);
      checkpoint("generated-module13", {Payload(atomic, dyn::InitialAtomic(resolution).size()),
          TextureBytes(generated_mask, 1)});
      encode_m21(reset_depth, reset_mask);
      checkpoint("generated-module21", {TextureBytes(working_vector, 4),
          TextureBytes(working_mask, 1), Payload(pyramid[0], pyramid_bytes[0]),
          Payload(pyramid[1], pyramid_bytes[1]), Payload(pyramid[2], pyramid_bytes[2]),
          Payload(pyramid[3], pyramid_bytes[3]), Payload(pyramid[4], pyramid_bytes[4])});
      encode_m22();
      if (resolution.extra_level_count == 1)
        checkpoint("generated-module22", {Payload(extra_pyramid[0], extra_bytes[0])});
      else
        checkpoint("generated-module22", {Payload(extra_pyramid[0], extra_bytes[0]),
                                           Payload(extra_pyramid[1], extra_bytes[1])});
      encode_m23();
      checkpoint("generated-module23", {Payload(selected_levels[0],
          resolution.selected_level == 10 ? pyramid_bytes[1] : pyramid_bytes[2])});
      encode_m24();
      checkpoint("generated-module24", {TextureBytes(working_vector, 4)});
      encode_m25(scatter_generated);
      checkpoint("generated-module25", {TextureBytes(scatter_generated, 1)});

      const std::array<id<MTLTexture>, 10> m5_inputs = {
          reset_color, generated_color, scatter_generated, scatter_reset,
          reset_projection, shared_metric, allocation29, flow,
          working_vector, working_mask};
      const Dynamic5Parameters m5_parameters{
          resolution.motion_width, resolution.motion_height,
          resolution.color_width, resolution.color_height,
          resolution.tensor_width, resolution.tensor_height,
          0, -static_cast<std::int32_t>(resolution.origin_y * 2u)};
      auto m5_command = [queue commandBuffer];
      auto m5_surface_encoder = [m5_command computeCommandEncoder];
      [m5_surface_encoder setComputePipelineState:m5_surfaces];
      for (NSUInteger index = 0; index < m5_inputs.size(); ++index)
        [m5_surface_encoder setTexture:m5_inputs[index] atIndex:index];
      [m5_surface_encoder setTexture:auxiliary_mask atIndex:10];
      [m5_surface_encoder setTexture:auxiliary_vector atIndex:11];
      [m5_surface_encoder setBytes:&m5_parameters length:sizeof(m5_parameters)
                           atIndex:0];
      [m5_surface_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [m5_surface_encoder dispatchThreads:
          MTLSizeMake(resolution.color_width, resolution.color_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m5_surface_encoder endEncoding];
      auto m5_tensor_encoder = [m5_command computeCommandEncoder];
      [m5_tensor_encoder setComputePipelineState:m5_tensor];
      for (NSUInteger index = 0; index < m5_inputs.size(); ++index)
        [m5_tensor_encoder setTexture:m5_inputs[index] atIndex:index];
      [m5_tensor_encoder setBuffer:model_input offset:kGuardBytes atIndex:0];
      [m5_tensor_encoder setBytes:&m5_parameters length:sizeof(m5_parameters)
                          atIndex:1];
      [m5_tensor_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
      [m5_tensor_encoder dispatchThreads:
          MTLSizeMake(resolution.tensor_width, resolution.tensor_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [m5_tensor_encoder endEncoding];
      Wait(m5_command);
      const auto model_input_bytes =
          Payload(model_input, dyn::ModelInputBytes(resolution));
      const auto auxiliary_mask_bytes = TextureBytes(auxiliary_mask, 1);
      const auto auxiliary_vector_bytes = TextureBytes(auxiliary_vector, 4);
      checkpoint("generated-module5", {model_input_bytes,
          auxiliary_mask_bytes, auxiliary_vector_bytes});

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
      input_mismatches += Mismatches(dyn::InitialAllocation29(resolution),
                                     TextureBytes(allocation29, 2));
      input_mismatches += Mismatches(rom, Payload(rom_buffer, rom.size()));
      std::size_t canary_mismatches = transient_canary_mismatches;
      canary_mismatches += CanaryMismatches(rom_buffer, rom.size());
      canary_mismatches += CanaryMismatches(
          atomic, dyn::InitialAtomic(resolution).size());
      for (std::size_t index = 0; index < pyramid.size(); ++index)
        canary_mismatches += CanaryMismatches(pyramid[index],
                                              pyramid_bytes[index]);
      for (std::size_t index = 0; index < extra_pyramid.size(); ++index)
        canary_mismatches += CanaryMismatches(extra_pyramid[index],
                                              extra_bytes[index]);
      canary_mismatches += CanaryMismatches(
          model_input, dyn::ModelInputBytes(resolution));
      canary_mismatches += CanaryMismatches(status_buffer, 4);
      if (full_artifacts) {
        Write(output_root / "allocation37.fp16", model_input_bytes);
        Write(output_root / "auxiliary-mask.r8", auxiliary_mask_bytes);
        Write(output_root / "auxiliary-vector.rg16f", auxiliary_vector_bytes);
        Write(output_root / "reset-present.rgba8",
              TextureBytes(reset_present, 4));
      }
      const bool pass = input_mismatches == 0 && canary_mismatches == 0 &&
                        checkpoints.size() == 21;
      std::ofstream receipt(output_root / "run-receipt.json");
      if (!receipt) throw std::runtime_error("receipt open failed");
      receipt << "{\n  \"schema\": "
                 "\"neuralstudio.fg.connected-dynamic-early-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << run_id << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"resolution\": \"" << resolution.name << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"selected_launch_count\": 21,\n"
              << "  \"module22_and_module24_executed_each_phase\": true,\n"
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
              << "  \"reset_present_sha256\": \""
              << Sha256(TextureBytes(reset_present, 4)) << "\",\n"
              << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
              << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
              << "  \"artifacts_written\": "
              << (full_artifacts ? "true" : "false") << ",\n"
              << "  \"expected_or_captured_output_ingress\": false,\n"
              << "  \"archived_intermediate_ingress\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n}\n";
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
