#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

#include <algorithm>

namespace {
namespace contract = fg_early_l7_l17;

struct OutputSurface {
  CUarray array = nullptr;
  CUsurfObject surface = 0;
  static constexpr std::uint32_t kRows = contract::kHeight + 1u;
  static constexpr std::size_t kRowBytes = contract::kWidth * 4u;

  void Allocate() {
    CUDA_ARRAY_DESCRIPTOR descriptor{};
    descriptor.Width = contract::kWidth;
    descriptor.Height = kRows;
    descriptor.Format = CU_AD_FORMAT_HALF;
    descriptor.NumChannels = 2;
    Check(cuArrayCreate(&array, &descriptor), "cuArrayCreate(output)");
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    Check(cuSurfObjectCreate(&surface, &resource), "cuSurfObjectCreate");
  }

  void Upload(const std::vector<std::uint8_t>& bytes) const {
    if (bytes.size() != kRowBytes * kRows) {
      throw std::runtime_error("output upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = kRowBytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = kRowBytes;
    copy.Height = kRows;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(output upload)");
  }

  std::vector<std::uint8_t> Download() const {
    std::vector<std::uint8_t> bytes(kRowBytes * kRows);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = kRowBytes;
    copy.WidthInBytes = kRowBytes;
    copy.Height = kRows;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(output download)");
    return bytes;
  }

  ~OutputSurface() {
    if (surface != 0) cuSurfObjectDestroy(surface);
    if (array != nullptr) cuArrayDestroy(array);
  }
};

struct Parameters {
  CUdeviceptr dead_mode0;
  CUdeviceptr level0;
  CUdeviceptr level1;
  CUdeviceptr level2;
  CUdeviceptr level3;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t mode;
  std::uint32_t padding52;
  CUsurfObject output;
  CUdeviceptr mask;
};
static_assert(sizeof(Parameters) == 72);

std::vector<std::uint8_t> CanonicalOutput(
    const std::vector<std::uint8_t>& output,
    const std::vector<std::uint8_t>& mask, std::uint8_t poison) {
  std::vector<std::uint8_t> canonical(output.size(), 0u);
  for (std::size_t pixel = 0; pixel < contract::kPixelCount; ++pixel) {
    const bool written = (mask[pixel] & 1u) != 0u;
    for (std::size_t lane = 0; lane < 4u; ++lane) {
      const std::size_t index = pixel * 4u + lane;
      if (written) canonical[index] = output[index];
      else if (output[index] != poison) {
        throw std::runtime_error("module23 wrote outside selected mask");
      }
    }
  }
  return canonical;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const std::string& phase, contract::Fixture fixture, std::uint8_t poison,
    const std::array<std::vector<std::uint8_t>, 4>& levels_before,
    const std::array<std::vector<std::uint8_t>, 4>& levels_after,
    const std::vector<std::uint8_t>& mask_before,
    const std::vector<std::uint8_t>& mask_after,
    const std::vector<std::uint8_t>& output,
    const std::vector<std::uint8_t>& canonical,
    std::size_t input_mismatches, std::size_t canary_mismatches) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write module23 receipt");
  const std::size_t written = static_cast<std::size_t>(std::count_if(
      mask_before.begin(), mask_before.end(),
      [](std::uint8_t value) { return (value & 1u) != 0u; }));
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l7-l17-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"phase\": \"" << phase << "\",\n"
         << "  \"selected_launch\": " << (phase == "reset" ? 7 : 17) << ",\n"
         << "  \"fixture\": \"" << contract::FixtureName(fixture) << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 23,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [256, 1, 1],\n"
         << "  \"packed_parameter_bytes\": 72,\n"
         << "  \"selected_mode\": 1,\n"
         << "  \"dead_offset0_value\": \"null\",\n"
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

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-early-l7-l17-rtx <module> <output-dir> <run-id> "
                 "<poison-hex> <reset|generated> <dense|sparse>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256)
      throw std::runtime_error("module23 identity mismatch");
    char* end = nullptr;
    const auto parsed = std::strtoul(argv[4], &end, 16);
    if (end == argv[4] || *end != '\0' || parsed > 255u)
      throw std::runtime_error("invalid poison");
    const auto poison = static_cast<std::uint8_t>(parsed);
    const std::string phase = argv[5];
    if (phase != "reset" && phase != "generated")
      throw std::runtime_error("invalid phase");
    const auto fixture = contract::ParseFixture(argv[6]);
    const std::filesystem::path root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(root);
    const auto levels_before = contract::Levels();
    const auto mask_before = contract::Mask(fixture);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module23");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get module23 function");
    std::array<GuardedDeviceBuffer, 4> levels;
    for (std::size_t i = 0; i < 4; ++i) {
      levels[i].Allocate(contract::kLevelBytes[i]);
      Check(cuMemcpyHtoD(levels[i].payload(), levels_before[i].data(),
                         levels_before[i].size()), "upload level");
    }
    GuardedDeviceBuffer mask;
    mask.Allocate(contract::kMaskBytes);
    Check(cuMemcpyHtoD(mask.payload(), mask_before.data(), mask_before.size()),
          "upload mask");
    OutputSurface output;
    output.Allocate();
    std::vector<std::uint8_t> output_before(
        OutputSurface::kRows * OutputSurface::kRowBytes, poison);
    output.Upload(output_before);
    Parameters parameters{0,
                          levels[0].payload(), levels[1].payload(),
                          levels[2].payload(), levels[3].payload(),
                          contract::kWidth, contract::kHeight, 1, 0,
                          output.surface, mask.payload()};
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 256, 1, 1, 0, nullptr,
                         nullptr, extra), "launch module23");
    Check(cuCtxSynchronize(), "synchronize module23");
    std::array<std::vector<std::uint8_t>, 4> levels_after;
    std::size_t input_mismatches = 0;
    std::size_t canary_mismatches = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto guarded = levels[i].ReadAll();
      levels_after[i] = Payload(guarded, contract::kLevelBytes[i]);
      input_mismatches += Mismatches(levels_before[i], levels_after[i]);
      canary_mismatches += CanaryMismatches(guarded, contract::kLevelBytes[i]);
      Write(root / ("level-" + std::to_string(i) + "-after.bin"), levels_after[i]);
    }
    const auto mask_guarded = mask.ReadAll();
    const auto mask_after = Payload(mask_guarded, contract::kMaskBytes);
    input_mismatches += Mismatches(mask_before, mask_after);
    canary_mismatches += CanaryMismatches(mask_guarded, contract::kMaskBytes);
    const auto output_guarded = output.Download();
    const std::vector<std::uint8_t> output_after(
        output_guarded.begin(), output_guarded.begin() + contract::kOutputBytes);
    for (std::size_t i = contract::kOutputBytes; i < output_guarded.size(); ++i)
      canary_mismatches += output_guarded[i] != poison;
    const auto canonical =
        CanonicalOutput(output_after, mask_before, poison);
    Write(root / "mask-after.bin", mask_after);
    Write(root / "output.bin", output_after);
    Write(root / "canonical-output.bin", canonical);
    WriteReceipt(root / "run-receipt.json", run_id, phase, fixture, poison,
                 levels_before, levels_after, mask_before, mask_after,
                 output_after, canonical, input_mismatches,
                 canary_mismatches);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_early_l7_l17_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "canonical_output_sha256=" << Sha256(canonical) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
