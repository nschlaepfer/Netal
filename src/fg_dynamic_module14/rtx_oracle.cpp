#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {
namespace contract = fg_dynamic_module14;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_bytes = 0;

  void Allocate(std::uint32_t input_width, std::uint32_t input_height,
                CUarray_format format, unsigned channels,
                std::size_t component_bytes) {
    width = input_width;
    height = input_height;
    row_bytes = std::size_t{width} * channels * component_bytes;
    CUDA_ARRAY_DESCRIPTOR descriptor{};
    descriptor.Width = width;
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

  void Upload(const std::vector<std::uint8_t>& bytes) const {
    if (bytes.size() != row_bytes * height) {
      throw std::runtime_error("typed array upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = row_bytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = row_bytes;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D HtoA");
  }

  std::vector<std::uint8_t> Download() const {
    std::vector<std::uint8_t> bytes(row_bytes * height);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = row_bytes;
    copy.WidthInBytes = row_bytes;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D AtoH");
    return bytes;
  }

  ~ArrayResource() {
    if (texture != 0) cuTexObjectDestroy(texture);
    if (surface != 0) cuSurfObjectDestroy(surface);
    if (array != nullptr) cuArrayDestroy(array);
  }
};

struct PackedParameters {
  CUtexObject input;
  CUtexObject dead_mask;
  CUsurfObject output;
  std::uint32_t width;
  std::uint32_t height;
};
static_assert(sizeof(PackedParameters) == 32);

std::vector<std::uint8_t> ActiveOutput(
    const contract::Resolution& resolution,
    const std::vector<std::uint8_t>& guarded) {
  const std::size_t bytes =
      std::size_t{resolution.width} * resolution.height * 4u;
  return {guarded.begin(), guarded.begin() + bytes};
}

std::size_t GuardMismatches(const contract::Resolution& resolution,
                            const std::vector<std::uint8_t>& guarded,
                            std::uint8_t poison) {
  const std::size_t active =
      std::size_t{resolution.width} * resolution.height * 4u;
  std::size_t mismatches = 0;
  for (std::size_t index = active; index < guarded.size(); ++index) {
    mismatches += guarded[index] != poison;
  }
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-dynamic-module14-rtx <module> <output-dir> "
                 "<run-id> <poison-hex> <resolution> <full|hash-only>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected module14 identity mismatch");
    }
    char* poison_end = nullptr;
    const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const auto& resolution = contract::FindResolution(argv[5]);
    const bool full_artifacts = std::string_view(argv[6]) == "full";
    if (!full_artifacts && std::string_view(argv[6]) != "hash-only") {
      throw std::runtime_error("invalid artifact mode");
    }
    const std::filesystem::path output_root = argv[2];
    std::filesystem::create_directories(output_root);
    const auto input_before = contract::GuardedInput(resolution, poison);
    const auto input_active = contract::InputActive(resolution);
    const auto dead_before = contract::DeadMask(resolution, poison);
    const auto output_before = contract::PoisonedOutput(resolution, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module14");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get module14 function");

    ArrayResource input;
    ArrayResource dead;
    ArrayResource output;
    input.Allocate(resolution.width, resolution.height + 1u,
                   CU_AD_FORMAT_HALF, 4, 2);
    dead.Allocate(resolution.width, resolution.height + 1u,
                  CU_AD_FORMAT_UNSIGNED_INT8, 1, 1);
    output.Allocate(resolution.width, resolution.height + 1u,
                    CU_AD_FORMAT_HALF, 2, 2);
    input.Upload(input_before);
    dead.Upload(dead_before);
    output.Upload(output_before);
    input.MakeTexture();
    dead.MakeTexture();
    output.MakeSurface();

    PackedParameters parameters{input.texture, dead.texture, output.surface,
                                resolution.width, resolution.height};
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, resolution.grid_x, resolution.grid_y, 1,
                         16, 16, 1, 0, nullptr, nullptr, extra),
          "launch selected module14");
    Check(cuCtxSynchronize(), "synchronize selected module14");

    const auto input_after = input.Download();
    const auto dead_after = dead.Download();
    const auto output_guarded = output.Download();
    const auto output_after = ActiveOutput(resolution, output_guarded);
    const auto input_mismatches = Mismatches(input_before, input_after) +
                                  Mismatches(dead_before, dead_after);
    const auto guard_mismatches =
        GuardMismatches(resolution, output_guarded, poison);
    if (full_artifacts) {
      Write(output_root / "input-after.bin", input_after);
      Write(output_root / "dead-mask-after.bin", dead_after);
      Write(output_root / "output.bin", output_after);
    }
    const bool pass = input_mismatches == 0 && guard_mismatches == 0;
    std::ofstream receipt(output_root / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    if (!receipt) throw std::runtime_error("unable to write run receipt");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.dynamic-module14-run.v1\",\n"
            << "  \"backend\": \"rtx-cuda-driver-typed-arrays\",\n"
            << "  \"run_id\": \"" << argv[3] << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"resolution\": \"" << resolution.name << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
            << "  \"function_name\": \"main_kernel\",\n"
            << "  \"extent\": [" << resolution.width << ", "
            << resolution.height << "],\n"
            << "  \"grid\": [" << resolution.grid_x << ", "
            << resolution.grid_y << ", 1],\n"
            << "  \"block\": [16, 16, 1],\n"
            << "  \"packed_parameter_bytes\": 32,\n"
            << "  \"raw_parameter_sha256\": \""
            << resolution.raw_parameter_sha256 << "\",\n"
            << "  \"input_active_sha256\": \"" << Sha256(input_active)
            << "\",\n"
            << "  \"input_resource_sha256\": \"" << Sha256(input_before)
            << "\",\n"
            << "  \"input_post_sha256\": \"" << Sha256(input_after)
            << "\",\n"
            << "  \"dead_resource_sha256\": \"" << Sha256(dead_before)
            << "\",\n"
            << "  \"dead_post_sha256\": \"" << Sha256(dead_after)
            << "\",\n"
            << "  \"output_bytes\": " << output_after.size() << ",\n"
            << "  \"output_sha256\": \"" << Sha256(output_after) << "\",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"output_guard_mismatched_bytes\": " << guard_mismatches
            << ",\n"
            << "  \"artifacts_written\": "
            << (full_artifacts ? "true" : "false") << ",\n"
            << "  \"expected_or_captured_output_ingress\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
            << "}\n";
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
