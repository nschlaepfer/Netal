#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {
namespace contract = fg_early_l9;

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
  CUtexObject dead_mask;
  CUsurfObject output;
  std::uint32_t width;
  std::uint32_t height;
};
static_assert(sizeof(Parameters) == 32);

std::vector<std::uint8_t> ActiveRows(
    const std::vector<std::uint8_t>& guarded_output) {
  return std::vector<std::uint8_t>(
      guarded_output.begin(),
      guarded_output.begin() + contract::kOutputBytes);
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& output,
                             std::uint8_t poison) {
  const std::size_t offset = contract::kOutputBytes;
  std::size_t count = 0;
  for (std::size_t index = offset; index < output.size(); ++index) {
    count += output[index] != poison;
  }
  return count;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, std::uint8_t poison,
                  const std::vector<std::uint8_t>& input_before,
                  const std::vector<std::uint8_t>& dead_before,
                  const std::vector<std::uint8_t>& input_after,
                  const std::vector<std::uint8_t>& dead_after,
                  const std::vector<std::uint8_t>& output,
                  std::size_t input_mismatches,
                  std::size_t canary_mismatches,
                  std::size_t poison_halfwords) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L9 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l9-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"selected_launch\": 9,\n"
         << "  \"module_token\": 14,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 32,\n"
         << "  \"dead_parameter_offset\": 8,\n"
         << "  \"input_sha256\": [\"" << Sha256(input_before) << "\", \""
         << Sha256(dead_before) << "\"],\n"
         << "  \"input_post_sha256\": [\"" << Sha256(input_after) << "\", \""
         << Sha256(dead_after) << "\"],\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_collision_halfwords\": " << poison_halfwords << ",\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-early-l9-rtx <module.bin> <output-dir> <run-id> "
                 "<poison-hex>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected L9 module identity mismatch");
    }
    char* poison_end = nullptr;
    const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' ||
        parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const auto input_before = contract::Input();
    const auto dead_before = contract::DeadMask(poison);
    std::vector<std::uint8_t> output_before(
        contract::kOutputBytes + contract::kWidth * 4u, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load L9 module");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get L9 function");

    ArrayResource input;
    ArrayResource dead;
    ArrayResource output;
    input.Allocate(CU_AD_FORMAT_HALF, 4, 2, contract::kHeight);
    dead.Allocate(CU_AD_FORMAT_UNSIGNED_INT8, 1, 1, contract::kHeight);
    output.Allocate(CU_AD_FORMAT_HALF, 2, 2, contract::kHeight + 1u);
    input.CopyFromHost(input_before);
    dead.CopyFromHost(dead_before);
    output.CopyFromHost(output_before);
    input.MakeTexture();
    dead.MakeTexture();
    output.MakeSurface();
    Parameters parameters{input.texture, dead.texture, output.surface,
                          contract::kWidth, contract::kHeight};
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 16, 16, 1, 0, nullptr,
                         nullptr, extra), "launch selected L9");
    Check(cuCtxSynchronize(), "synchronize selected L9");

    const auto input_after = input.CopyToHost();
    const auto dead_after = dead.CopyToHost();
    const auto output_guarded = output.CopyToHost();
    const auto output_after = ActiveRows(output_guarded);
    const auto input_mismatches = Mismatches(input_before, input_after) +
                                  Mismatches(dead_before, dead_after);
    const auto canary_mismatches = CanaryMismatches(output_guarded, poison);
    const auto poison_halfwords = UntouchedHalfwords(output_after, poison);
    Write(output_root / "input-after.bin", input_after);
    Write(output_root / "dead-mask-after.bin", dead_after);
    Write(output_root / "output.bin", output_after);
    WriteReceipt(output_root / "run-receipt.json", run_id, poison,
                 input_before, dead_before, input_after, dead_after,
                 output_after, input_mismatches, canary_mismatches,
                 poison_halfwords);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_early_l9_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(output_after) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
