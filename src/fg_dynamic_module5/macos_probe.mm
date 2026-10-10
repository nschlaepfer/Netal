#define main FgEarlyL19EmbeddedMacMain
#include "../fg_early_l19/macos_probe.mm"
#undef main

#include "contract.h"

namespace {
namespace dynamic5 = fg_dynamic_module5;

constexpr std::size_t kDynamic5RomBytes = 2097152;
constexpr char kDynamic5RomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

struct Dynamic5MetalParameters {
  std::uint32_t motion_width;
  std::uint32_t motion_height;
  std::uint32_t work_width;
  std::uint32_t work_height;
  std::uint32_t tensor_width;
  std::uint32_t tensor_height;
  std::int32_t input_x_offset;
  std::int32_t input_y_offset;
};
static_assert(sizeof(Dynamic5MetalParameters) == 32);

constexpr std::array<std::size_t, 10> kDynamic5PixelBytes{{
    8, 8, 1, 1, 4, 2, 2, 8, 4, 1}};

std::vector<std::uint8_t> ReadDynamic5File(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read");
  return result;
}

std::string Dynamic5AddressFreeParameterSha(
    const dynamic5::Resolution& resolution) {
  std::array<std::uint8_t, 192> packed{};
  Pack32(&packed, 152, resolution.motion_width);
  Pack32(&packed, 156, resolution.motion_height);
  Pack32(&packed, 160, resolution.work_width);
  Pack32(&packed, 164, resolution.work_height);
  Pack32(&packed, 168, resolution.tensor_width);
  Pack32(&packed, 172, resolution.tensor_height);
  Pack32(&packed, 176, resolution.origin_x);
  Pack32(&packed, 180, resolution.reserved_origin_x);
  Pack32(&packed, 184, resolution.origin_y);
  Pack32(&packed, 188, resolution.reserved_origin_y);
  return HashArray(packed);
}

void WriteDynamic5MacReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const dynamic5::Resolution& resolution, dynamic5::Fixture fixture,
    std::uint8_t poison, const std::string& device_name,
    const dynamic5::Inputs& inputs,
    const std::array<std::vector<std::uint8_t>, 10>& inputs_after,
    const std::vector<std::uint8_t>& tensor,
    const std::vector<std::uint8_t>& mask,
    const std::vector<std::uint8_t>& vector,
    std::size_t input_mismatches, std::size_t buffer_canary_mismatches,
    std::size_t texture_canary_mismatches,
    const std::string& address_free_parameter_sha,
    bool artifacts_written) {
  const bool pass = input_mismatches == 0 &&
                    buffer_canary_mismatches == 0 &&
                    texture_canary_mismatches == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write dynamic module5 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.dynamic-module5-run.v1\",\n"
          << "  \"backend\": \"mac-metal\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << getpid() << ",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"phase_fixture\": \"" << dynamic5::FixtureName(fixture)
          << "\",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"device_name\": \"" << device_name << "\",\n"
          << "  \"module_token\": 5,\n"
          << "  \"module_sha256\": \"" << dynamic5::kModuleSha256 << "\",\n"
          << "  \"metal_functions\": [\"fg_dynamic_module5_surfaces\", "
             "\"fg_dynamic_module5_tensor\"],\n"
          << "  \"grid\": [" << resolution.grid_x << ", "
          << resolution.grid_y << ", 1],\n"
          << "  \"block\": [16, 16, 1],\n"
          << "  \"packed_parameter_bytes\": 192,\n"
          << "  \"motion_extent\": [" << resolution.motion_width << ", "
          << resolution.motion_height << "],\n"
          << "  \"work_extent\": [" << resolution.work_width << ", "
          << resolution.work_height << "],\n"
          << "  \"tensor_extent\": [" << resolution.tensor_width << ", "
          << resolution.tensor_height << ", 48],\n"
          << "  \"logical_origin\": [" << resolution.origin_x << ", "
          << resolution.origin_y << "],\n"
          << "  \"raw_parameter_sha256\": \""
          << resolution.raw_parameter_sha256 << "\",\n"
          << "  \"address_free_parameter_sha256\": \""
          << address_free_parameter_sha << "\",\n"
          << "  \"architecture_rom_sha256\": \""
          << kDynamic5RomSha256 << "\",\n"
          << "  \"normalized_point_fraction_bits\": 21,\n"
          << "  \"vector_surface_half_rounding\": \"toward-zero\",\n"
          << "  \"input_generator\": \"" << dynamic5::kInputGenerator
          << "\",\n"
          << "  \"input_pre_sha256\": [";
  for (std::size_t index = 0; index < inputs.textures.size(); ++index) {
    if (index != 0) receipt << ", ";
    receipt << "\"" << HashBytes(inputs.textures[index]) << "\"";
  }
  receipt << "],\n  \"input_post_sha256\": [";
  for (std::size_t index = 0; index < inputs_after.size(); ++index) {
    if (index != 0) receipt << ", ";
    receipt << "\"" << HashBytes(inputs_after[index]) << "\"";
  }
  receipt << "],\n"
          << "  \"tensor_output_bytes\": " << tensor.size() << ",\n"
          << "  \"tensor_output_sha256\": \"" << HashBytes(tensor) << "\",\n"
          << "  \"mask_output_bytes\": " << mask.size() << ",\n"
          << "  \"mask_output_sha256\": \"" << HashBytes(mask) << "\",\n"
          << "  \"vector_output_bytes\": " << vector.size() << ",\n"
          << "  \"vector_output_sha256\": \"" << HashBytes(vector) << "\",\n"
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"buffer_canary_mismatched_bytes\": "
          << buffer_canary_mismatches << ",\n"
          << "  \"texture_canary_mismatched_bytes\": "
          << texture_canary_mismatches << ",\n"
          << "  \"complete_write_coverage_checked_by_dual_poison\": true,\n"
          << "  \"artifacts_written\": "
          << (artifacts_written ? "true" : "false") << ",\n"
          << "  \"expected_or_captured_output_ingress\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 9) {
      std::cerr << "usage=fg-dynamic-module5-mac <metallib> <rcp-rom> "
                   "<output-dir> <run-id> <poison-hex> <resolution> "
                   "<first-frame|fan-in> <full|hash-only>\n";
      return 2;
    }
    try {
      const auto rom = ReadDynamic5File(argv[2]);
      if (rom.size() != kDynamic5RomBytes ||
          HashBytes(rom) != kDynamic5RomSha256) {
        throw std::runtime_error("RCP ROM identity mismatch");
      }
      const std::filesystem::path output_directory = argv[3];
      const std::string run_id = argv[4];
      const auto parsed_poison = std::stoul(argv[5], nullptr, 16);
      if (parsed_poison > 255u) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed_poison);
      const auto& resolution = dynamic5::FindResolution(argv[6]);
      const auto fixture = dynamic5::ParseFixture(argv[7]);
      const bool full_artifacts = std::string_view(argv[8]) == "full";
      if (!full_artifacts && std::string_view(argv[8]) != "hash-only") {
        throw std::runtime_error("invalid artifact mode");
      }
      std::filesystem::create_directories(output_directory);
      const auto address_free_parameter_sha =
          Dynamic5AddressFreeParameterSha(resolution);
      if (address_free_parameter_sha !=
          resolution.address_free_parameter_sha256) {
        throw std::runtime_error("dynamic module5 address-free ABI mismatch");
      }
      const auto inputs = dynamic5::GenerateInputs(resolution, fixture);
      const auto tensor_bytes = dynamic5::TensorBytes(resolution);
      const auto mask_bytes = dynamic5::MaskBytes(resolution);
      const auto vector_bytes = dynamic5::VectorBytes(resolution);
      const std::vector<std::uint8_t> mask_poison(mask_bytes, poison);
      const std::vector<std::uint8_t> vector_poison(vector_bytes, poison);
      const std::vector<std::uint8_t> mask_canary_before(
          mask_bytes, static_cast<std::uint8_t>(poison ^ 0xffu));
      const std::vector<std::uint8_t> vector_canary_before(
          vector_bytes, static_cast<std::uint8_t>(poison ^ 0xffu));

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> surface_function =
          [library newFunctionWithName:@"fg_dynamic_module5_surfaces"];
      id<MTLFunction> tensor_function =
          [library newFunctionWithName:@"fg_dynamic_module5_tensor"];
      id<MTLComputePipelineState> surface_pipeline =
          [device newComputePipelineStateWithFunction:surface_function
                                                 error:&error];
      id<MTLComputePipelineState> tensor_pipeline =
          [device newComputePipelineStateWithFunction:tensor_function
                                                 error:&error];
      if (surface_pipeline == nil || tensor_pipeline == nil)
        throw std::runtime_error("pipeline creation failed");
      std::array<id<MTLTexture>, 10> input_textures;
      for (std::size_t index = 0; index < input_textures.size(); ++index) {
        input_textures[index] = MakeTexture(
            device, kInputSpecs[index].format, resolution.work_width,
            resolution.work_height, MTLTextureUsageShaderRead,
            inputs.textures[index],
            std::size_t{resolution.work_width} * kDynamic5PixelBytes[index]);
      }
      id<MTLTexture> mask_output = MakeTexture(
          device, MTLPixelFormatR8Unorm, resolution.work_width,
          resolution.work_height, MTLTextureUsageShaderWrite, mask_poison,
          resolution.work_width);
      id<MTLTexture> vector_output = MakeTexture(
          device, MTLPixelFormatRG16Float, resolution.work_width,
          resolution.work_height, MTLTextureUsageShaderWrite, vector_poison,
          std::size_t{resolution.work_width} * 4u);
      id<MTLTexture> mask_canary = MakeTexture(
          device, MTLPixelFormatR8Unorm, resolution.work_width,
          resolution.work_height, MTLTextureUsageShaderWrite,
          mask_canary_before, resolution.work_width);
      id<MTLTexture> vector_canary = MakeTexture(
          device, MTLPixelFormatRG16Float, resolution.work_width,
          resolution.work_height, MTLTextureUsageShaderWrite,
          vector_canary_before, std::size_t{resolution.work_width} * 4u);
      id<MTLBuffer> tensor_output = [device
          newBufferWithLength:tensor_bytes + 2u * kGuardBytes
                      options:MTLResourceStorageModeShared];
      id<MTLBuffer> rom_buffer = [device
          newBufferWithLength:rom.size() + 2u * kGuardBytes
                      options:MTLResourceStorageModeShared];
      if (tensor_output == nil || rom_buffer == nil)
        throw std::runtime_error("buffer allocation failed");
      std::memset(tensor_output.contents, poison, tensor_output.length);
      std::memset(rom_buffer.contents, poison, rom_buffer.length);
      std::memcpy(static_cast<std::uint8_t*>(rom_buffer.contents) + kGuardBytes,
                  rom.data(), rom.size());
      const Dynamic5MetalParameters parameters{
          resolution.motion_width, resolution.motion_height,
          resolution.work_width, resolution.work_height,
          resolution.tensor_width, resolution.tensor_height,
          -static_cast<std::int32_t>(resolution.origin_x * 2u),
          -static_cast<std::int32_t>(resolution.origin_y * 2u)};
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> surface_encoder =
          [command computeCommandEncoder];
      [surface_encoder setComputePipelineState:surface_pipeline];
      BindInputs(surface_encoder, input_textures);
      [surface_encoder setTexture:mask_output atIndex:10];
      [surface_encoder setTexture:vector_output atIndex:11];
      [surface_encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
      [surface_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [surface_encoder
          dispatchThreads:MTLSizeMake(resolution.work_width,
                                      resolution.work_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [surface_encoder endEncoding];
      id<MTLComputeCommandEncoder> tensor_encoder =
          [command computeCommandEncoder];
      [tensor_encoder setComputePipelineState:tensor_pipeline];
      BindInputs(tensor_encoder, input_textures);
      [tensor_encoder setBuffer:tensor_output offset:kGuardBytes atIndex:0];
      [tensor_encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
      [tensor_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
      [tensor_encoder
          dispatchThreads:MTLSizeMake(resolution.tensor_width,
                                      resolution.tensor_height, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [tensor_encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal dynamic module5 command failed");

      std::array<std::vector<std::uint8_t>, 10> inputs_after;
      std::size_t input_mismatches = 0;
      for (std::size_t index = 0; index < input_textures.size(); ++index) {
        inputs_after[index] = ReadTexture(
            input_textures[index],
            std::size_t{resolution.work_width} * kDynamic5PixelBytes[index]);
        input_mismatches +=
            Mismatches(inputs.textures[index], inputs_after[index]);
      }
      const auto mask_after = ReadTexture(mask_output, resolution.work_width);
      const auto vector_after = ReadTexture(
          vector_output, std::size_t{resolution.work_width} * 4u);
      const auto mask_canary_after =
          ReadTexture(mask_canary, resolution.work_width);
      const auto vector_canary_after = ReadTexture(
          vector_canary, std::size_t{resolution.work_width} * 4u);
      const auto tensor_after = BufferPayload(tensor_output, tensor_bytes);
      const auto rom_after = BufferPayload(rom_buffer, rom.size());
      input_mismatches += Mismatches(rom, rom_after);
      const std::size_t buffer_canary_mismatches =
          BufferCanaryMismatches(tensor_output, tensor_bytes, poison) +
          BufferCanaryMismatches(rom_buffer, rom.size(), poison);
      const std::size_t texture_canary_mismatches =
          Mismatches(mask_canary_before, mask_canary_after) +
          Mismatches(vector_canary_before, vector_canary_after);
      if (full_artifacts) {
        WriteBytes(output_directory / "tensor-output.fp16", tensor_after);
        WriteBytes(output_directory / "mask-output.r8", mask_after);
        WriteBytes(output_directory / "vector-output.rg16f", vector_after);
      }
      WriteDynamic5MacReceipt(
          output_directory / "run-receipt.json", run_id, resolution, fixture,
          poison, device.name.UTF8String, inputs, inputs_after, tensor_after,
          mask_after, vector_after, input_mismatches,
          buffer_canary_mismatches, texture_canary_mismatches,
          address_free_parameter_sha, full_artifacts);
      const bool pass = input_mismatches == 0 &&
                        buffer_canary_mismatches == 0 &&
                        texture_canary_mismatches == 0;
      std::cout << "probe=fg_dynamic_module5_mac\n"
                << "run_id=" << run_id << '\n'
                << "tensor_output_sha256=" << HashBytes(tensor_after) << '\n'
                << "mask_output_sha256=" << HashBytes(mask_after) << '\n'
                << "vector_output_sha256=" << HashBytes(vector_after) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
