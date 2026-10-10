#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

#include "contract.h"

#include <algorithm>

namespace {
namespace contract = fg_early_l7_l17;
constexpr std::size_t kRomBytes = 2097152;
constexpr char kRomSha[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short file read");
  return bytes;
}

void Store32(std::array<std::uint8_t, 72>* bytes, std::size_t offset,
             std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

void Store64(std::array<std::uint8_t, 72>* bytes, std::size_t offset,
             std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*bytes)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

std::vector<std::uint8_t> CanonicalOutput(
    const std::vector<std::uint8_t>& output,
    const std::vector<std::uint8_t>& mask, std::uint8_t poison) {
  std::vector<std::uint8_t> canonical(output.size(), 0u);
  for (std::size_t pixel = 0; pixel < contract::kPixelCount; ++pixel) {
    const bool written = (mask[pixel] & 1u) != 0u;
    for (std::size_t lane = 0; lane < 4u; ++lane) {
      const std::size_t index = pixel * 4u + lane;
      if (written) canonical[index] = output[index];
      else if (output[index] != poison)
        throw std::runtime_error("Metal module23 wrote outside selected mask");
    }
  }
  return canonical;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const std::string& phase, contract::Fixture fixture, std::uint8_t poison,
    const std::string& device_name,
    const std::array<std::vector<std::uint8_t>, 4>& levels_before,
    const std::array<std::vector<std::uint8_t>, 4>& levels_after,
    const std::vector<std::uint8_t>& mask_before,
    const std::vector<std::uint8_t>& mask_after,
    const std::vector<std::uint8_t>& output,
    const std::vector<std::uint8_t>& canonical,
    std::size_t input_mismatches, std::size_t canary_mismatches) {
  const std::size_t written = static_cast<std::size_t>(std::count_if(
      mask_before.begin(), mask_before.end(),
      [](std::uint8_t value) { return (value & 1u) != 0u; }));
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write module23 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l7-l17-run.v1\",\n"
         << "  \"backend\": \"mac-metal\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << getpid() << ",\n"
         << "  \"phase\": \"" << phase << "\",\n"
         << "  \"selected_launch\": " << (phase == "reset" ? 7 : 17) << ",\n"
         << "  \"fixture\": \"" << contract::FixtureName(fixture) << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"module_token\": 23,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"metal_function\": \"fg_early_l7_l17\",\n"
         << "  \"selected_grid\": [20, 12, 1],\n"
         << "  \"selected_block\": [256, 1, 1],\n"
         << "  \"packed_parameter_bytes\": 72,\n"
         << "  \"selected_mode\": 1,\n"
         << "  \"dead_offset0_value\": \"null\",\n"
         << "  \"architecture_rom_sha256\": \"" << kRomSha << "\",\n"
         << "  \"surface_half_rounding\": \"toward-zero\",\n"
         << "  \"level_pre_sha256\": [";
  for (std::size_t i = 0; i < 4; ++i)
    stream << (i ? ", \"" : "\"") << Sha256(levels_before[i]) << "\"";
  stream << "],\n  \"level_post_sha256\": [";
  for (std::size_t i = 0; i < 4; ++i)
    stream << (i ? ", \"" : "\"") << Sha256(levels_after[i]) << "\"";
  stream << "],\n"
         << "  \"mask_pre_sha256\": \"" << Sha256(mask_before) << "\",\n"
         << "  \"mask_post_sha256\": \"" << Sha256(mask_after) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"raw_output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"canonical_output_sha256\": \"" << Sha256(canonical) << "\",\n"
         << "  \"written_pixels\": " << written << ",\n"
         << "  \"untouched_pixels\": " << (contract::kPixelCount - written) << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-early-l7-l17-mac <metallib> <rcp-rom> "
                   "<output-dir> <run-id> <poison> <reset|generated> "
                   "<dense|sparse>\n";
      return 2;
    }
    try {
      const auto rom = ReadFile(argv[2]);
      if (rom.size() != kRomBytes || Sha256(rom) != kRomSha)
        throw std::runtime_error("RCP ROM identity mismatch");
      char* end = nullptr;
      const auto parsed = std::strtoul(argv[5], &end, 16);
      if (end == argv[5] || *end != '\0' || parsed > 255u)
        throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed);
      const std::string phase = argv[6];
      if (phase != "reset" && phase != "generated")
        throw std::runtime_error("invalid phase");
      const auto fixture = contract::ParseFixture(argv[7]);
      const std::filesystem::path root = argv[3];
      const std::string run_id = argv[4];
      std::filesystem::create_directories(root);
      const auto levels_before = contract::Levels();
      const auto mask_before = contract::Mask(fixture);

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:
          [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                error:&error];
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_early_l7_l17"];
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (library == nil || function == nil || pipeline == nil)
        throw std::runtime_error("Metal module23 pipeline failed");
      std::array<id<MTLBuffer>, 4> levels;
      for (std::size_t i = 0; i < 4; ++i)
        levels[i] = Guarded(device, &levels_before[i], contract::kLevelBytes[i], 0);
      id<MTLBuffer> mask = Guarded(device, &mask_before, contract::kMaskBytes, 0);
      std::vector<std::uint8_t> output_before(contract::kOutputBytes, poison);
      id<MTLBuffer> output = Guarded(device, &output_before,
                                     contract::kOutputBytes, 0);
      id<MTLBuffer> rom_buffer = Guarded(device, &rom, kRomBytes, 0);
      std::array<std::uint8_t, 72> parameters{};
      for (std::size_t i = 0; i < 4; ++i)
        Store64(&parameters, 8u + i * 8u,
                levels[i].gpuAddress + kGuardBytes);
      Store32(&parameters, 40, contract::kWidth);
      Store32(&parameters, 44, contract::kHeight);
      Store32(&parameters, 48, 1);
      Store64(&parameters, 56, output.gpuAddress + kGuardBytes);
      Store64(&parameters, 64, mask.gpuAddress + kGuardBytes);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
      [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      for (id<MTLBuffer> level : levels)
        [encoder useResource:level usage:MTLResourceUsageRead];
      [encoder useResource:mask usage:MTLResourceUsageRead];
      [encoder useResource:output usage:MTLResourceUsageWrite];
      [encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal module23 command failed");

      std::array<std::vector<std::uint8_t>, 4> levels_after;
      std::size_t input_mismatches = 0;
      std::size_t canary_mismatches = 0;
      for (std::size_t i = 0; i < 4; ++i) {
        levels_after[i] = Payload(levels[i], contract::kLevelBytes[i]);
        input_mismatches += Mismatches(levels_before[i], levels_after[i]);
        canary_mismatches += CanaryMismatches(levels[i], contract::kLevelBytes[i]);
        Write(root / ("level-" + std::to_string(i) + "-after.bin"), levels_after[i]);
      }
      const auto mask_after = Payload(mask, contract::kMaskBytes);
      const auto output_after = Payload(output, contract::kOutputBytes);
      const auto rom_after = Payload(rom_buffer, kRomBytes);
      input_mismatches += Mismatches(mask_before, mask_after) +
                          Mismatches(rom, rom_after);
      canary_mismatches += CanaryMismatches(mask, contract::kMaskBytes) +
                           CanaryMismatches(output, contract::kOutputBytes) +
                           CanaryMismatches(rom_buffer, kRomBytes);
      const auto canonical = CanonicalOutput(output_after, mask_before, poison);
      Write(root / "mask-after.bin", mask_after);
      Write(root / "output.bin", output_after);
      Write(root / "canonical-output.bin", canonical);
      WriteReceipt(root / "run-receipt.json", run_id, phase, fixture, poison,
                   device.name.UTF8String, levels_before, levels_after,
                   mask_before, mask_after, output_after, canonical,
                   input_mismatches, canary_mismatches);
      const bool pass = input_mismatches == 0 && canary_mismatches == 0;
      std::cout << "probe=fg_early_l7_l17_mac\n"
                << "run_id=" << run_id << '\n'
                << "canonical_output_sha256=" << Sha256(canonical) << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
