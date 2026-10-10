#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {
namespace contract = fg_early_l8_l18;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;
  std::uint32_t height = 0;
  std::size_t row_bytes = 0;

  void Allocate(CUarray_format format, unsigned channels,
                std::size_t component_bytes, std::uint32_t rows) {
    height = rows;
    row_bytes = std::size_t{contract::kWidth} * channels * component_bytes;
    CUDA_ARRAY_DESCRIPTOR descriptor{};
    descriptor.Width = contract::kWidth;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    Check(cuArrayCreate(&array, &descriptor), "cuArrayCreate");
  }

  void MakeTexture() {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC descriptor{};
    descriptor.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.filterMode = CU_TR_FILTER_MODE_POINT;
    descriptor.flags = CU_TRSF_NORMALIZED_COORDINATES;
    Check(cuTexObjectCreate(&texture, &resource, &descriptor, nullptr),
          "cuTexObjectCreate");
  }

  void MakeSurface() {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    Check(cuSurfObjectCreate(&surface, &resource), "cuSurfObjectCreate");
  }

  void CopyFromHost(const std::vector<std::uint8_t>& bytes) const {
    if (bytes.size() != row_bytes * height) {
      throw std::runtime_error("array upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = row_bytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = row_bytes;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(HtoA)");
  }

  std::vector<std::uint8_t> CopyToHost() const {
    std::vector<std::uint8_t> bytes(row_bytes * height);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = row_bytes;
    copy.WidthInBytes = row_bytes;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(AtoH)");
    return bytes;
  }

  ~ArrayResource() {
    if (texture != 0) cuTexObjectDestroy(texture);
    if (surface != 0) cuSurfObjectDestroy(surface);
    if (array != nullptr) cuArrayDestroy(array);
  }
};

struct Parameters {
  CUtexObject input;
  CUsurfObject output;
  std::uint32_t mode;
  std::uint32_t source_width;
  std::uint32_t source_height;
  float inverse_width;
  float inverse_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  float unused_threshold;
};
static_assert(sizeof(Parameters) == 48);

std::vector<std::uint8_t> Coverage(const std::vector<std::uint8_t>& output,
                                   std::uint8_t poison) {
  std::vector<std::uint8_t> coverage(output.size());
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (output[index] == 0u) coverage[index] = 1u;
    else if (output[index] == poison) coverage[index] = 0u;
    else throw std::runtime_error("scatter output escaped zero/poison domain");
  }
  return coverage;
}

std::size_t WrittenCount(const std::vector<std::uint8_t>& coverage) {
  std::size_t count = 0;
  for (const auto value : coverage) count += value != 0u;
  return count;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, const std::string& phase,
                  std::uint8_t poison,
                  const std::vector<std::uint8_t>& input_before,
                  const std::vector<std::uint8_t>& input_after,
                  const std::vector<std::uint8_t>& output,
                  const std::vector<std::uint8_t>& coverage,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L8/L18 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l8-l18-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"phase\": \"" << phase << "\",\n"
         << "  \"selected_launch\": " << (phase == "reset" ? 8 : 18) << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 25,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 48,\n"
         << "  \"selected_mode\": 0,\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"coverage_sha256\": \"" << Sha256(coverage) << "\",\n"
         << "  \"written_pixels\": " << WrittenCount(coverage) << ",\n"
         << "  \"untouched_pixels\": "
         << (coverage.size() - WrittenCount(coverage)) << ",\n"
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
  if (argc != 6) {
    std::cerr << "usage=fg-early-l8-l18-rtx <module.bin> <output-dir> "
                 "<run-id> <poison-hex> <reset|generated>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected module25 identity mismatch");
    }
    char* poison_end = nullptr;
    const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' ||
        parsed_poison > 255u) throw std::runtime_error("invalid poison byte");
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::string phase = argv[5];
    if (phase != "reset" && phase != "generated") {
      throw std::runtime_error("invalid L8/L18 phase");
    }
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const auto input_before = contract::Input();
    std::vector<std::uint8_t> output_before(
        contract::kOutputBytes + contract::kWidth, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module25");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get module25 function");
    ArrayResource input;
    ArrayResource output;
    input.Allocate(CU_AD_FORMAT_HALF, 2, 2, contract::kHeight);
    output.Allocate(CU_AD_FORMAT_UNSIGNED_INT8, 1, 1,
                    contract::kHeight + 1u);
    input.CopyFromHost(input_before);
    output.CopyFromHost(output_before);
    input.MakeTexture();
    output.MakeSurface();
    Parameters parameters{input.texture,
                          output.surface,
                          0,
                          contract::kWidth,
                          contract::kHeight,
                          0.00312500004656612873077392578125f,
                          0.0055555556900799274444580078125f,
                          contract::kWidth,
                          contract::kHeight,
                          12.96000003814697265625f};
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 16, 16, 1, 0, nullptr,
                         nullptr, extra), "launch module25");
    Check(cuCtxSynchronize(), "synchronize module25");
    const auto input_after = input.CopyToHost();
    const auto guarded_output = output.CopyToHost();
    const std::vector<std::uint8_t> output_after(
        guarded_output.begin(),
        guarded_output.begin() + contract::kOutputBytes);
    const auto coverage = Coverage(output_after, poison);
    const auto input_mismatches = Mismatches(input_before, input_after);
    std::size_t canary_mismatches = 0;
    for (std::size_t index = contract::kOutputBytes;
         index < guarded_output.size(); ++index) {
      canary_mismatches += guarded_output[index] != poison;
    }
    Write(output_root / "input-after.bin", input_after);
    Write(output_root / "output.bin", output_after);
    Write(output_root / "coverage.bin", coverage);
    WriteReceipt(output_root / "run-receipt.json", run_id, phase, poison,
                 input_before, input_after, output_after, coverage,
                 input_mismatches, canary_mismatches);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_early_l8_l18_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "coverage_sha256=" << Sha256(coverage) << '\n'
              << "written_pixels=" << WrittenCount(coverage) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
