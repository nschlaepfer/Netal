#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {
namespace contract = fg_mid_l54;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;

  void Allocate() {
    CUDA_ARRAY_DESCRIPTOR descriptor{};
    descriptor.Width = contract::kWidth;
    descriptor.Height = contract::kHeight;
    descriptor.Format = CU_AD_FORMAT_HALF;
    descriptor.NumChannels = 2;
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

  void CopyFromHost(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() != contract::kTextureBytes) {
      throw std::runtime_error("array upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = contract::kWidth * 4u;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = contract::kWidth * 4u;
    copy.Height = contract::kHeight;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(HtoA)");
  }

  std::vector<std::uint8_t> CopyToHost() const {
    std::vector<std::uint8_t> bytes(contract::kTextureBytes);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = contract::kWidth * 4u;
    copy.WidthInBytes = contract::kWidth * 4u;
    copy.Height = contract::kHeight;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(AtoH)");
    return bytes;
  }

  void Free() {
    if (texture != 0) {
      Check(cuTexObjectDestroy(texture), "cuTexObjectDestroy");
      texture = 0;
    }
    if (surface != 0) {
      Check(cuSurfObjectDestroy(surface), "cuSurfObjectDestroy");
      surface = 0;
    }
    if (array != nullptr) {
      Check(cuArrayDestroy(array), "cuArrayDestroy");
      array = nullptr;
    }
  }

  ~ArrayResource() {
    if (texture != 0) cuTexObjectDestroy(texture);
    if (surface != 0) cuSurfObjectDestroy(surface);
    if (array != nullptr) cuArrayDestroy(array);
  }
};

struct Parameters {
  std::uint32_t source_width;
  std::uint32_t source_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  CUtexObject texture_a;
  CUtexObject texture_b;
  CUdeviceptr offset_a;
  CUdeviceptr offset_b;
  CUsurfObject output;
  std::array<std::uint32_t, 8> reserved;
};
static_assert(sizeof(Parameters) == 88);

std::size_t PoisonHalfwords(const std::vector<std::uint8_t>& bytes,
                            std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 2u) {
    count += bytes[offset] == poison && bytes[offset + 1u] == poison;
  }
  return count;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    std::uint8_t poison, const std::vector<std::uint8_t>& texture_a_before,
    const std::vector<std::uint8_t>& texture_b_before,
    const std::vector<std::uint8_t>& offset_a_before,
    const std::vector<std::uint8_t>& offset_b_before,
    const std::vector<std::uint8_t>& texture_a_after,
    const std::vector<std::uint8_t>& texture_b_after,
    const std::vector<std::uint8_t>& offset_a_after,
    const std::vector<std::uint8_t>& offset_b_after,
    const std::vector<std::uint8_t>& output, std::size_t input_mismatches,
    std::size_t canary_mismatches, std::size_t poison_halfwords) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L54 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.mid-l54-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"selected_launch\": 54,\n"
         << "  \"module_token\": 26,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 88,\n"
         << "  \"dimensions\": [320, 180, 320, 180],\n"
         << "  \"resource_contract\": {\"inputs\": [\"RG16F point-border normalized texture\", \"RG16F point-border normalized texture\", \"u64 offset buffer\", \"u64 offset buffer\"], \"output\": \"RG16F surface\"},\n"
         << "  \"texture_pre_sha256\": [\"" << Sha256(texture_a_before)
         << "\", \"" << Sha256(texture_b_before) << "\"],\n"
         << "  \"texture_post_sha256\": [\"" << Sha256(texture_a_after)
         << "\", \"" << Sha256(texture_b_after) << "\"],\n"
         << "  \"offset_pre_sha256\": [\"" << Sha256(offset_a_before)
         << "\", \"" << Sha256(offset_b_before) << "\"],\n"
         << "  \"offset_post_sha256\": [\"" << Sha256(offset_a_after)
         << "\", \"" << Sha256(offset_b_after) << "\"],\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"poison_collision_halfwords\": " << poison_halfwords << ",\n"
         << "  \"per_run_write_coverage_conclusive\": false,\n"
         << "  \"unused_u64_upper_halves_poisoned\": true,\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-mid-l54-rtx <module.bin> <output-dir> <run-id> <poison-hex>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected module identity mismatch");
    }
    char* poison_end = nullptr;
    const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const auto texture_a_before = contract::TextureA();
    const auto texture_b_before = contract::TextureB();
    const auto offset_a_before = contract::OffsetA(poison);
    const auto offset_b_before = contract::OffsetB(poison);
    std::vector<std::uint8_t> output_poison(contract::kOutputBytes, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get function");
    ArrayResource texture_a;
    ArrayResource texture_b;
    ArrayResource output;
    texture_a.Allocate();
    texture_b.Allocate();
    output.Allocate();
    texture_a.CopyFromHost(texture_a_before);
    texture_b.CopyFromHost(texture_b_before);
    output.CopyFromHost(output_poison);
    texture_a.MakeTexture();
    texture_b.MakeTexture();
    output.MakeSurface();
    GuardedDeviceBuffer offset_a;
    GuardedDeviceBuffer offset_b;
    offset_a.Allocate(contract::kOffsetBufferBytes);
    offset_b.Allocate(contract::kOffsetBufferBytes);
    Check(cuMemcpyHtoD(offset_a.payload(), offset_a_before.data(),
                       offset_a_before.size()),
          "copy offset A");
    Check(cuMemcpyHtoD(offset_b.payload(), offset_b_before.data(),
                       offset_b_before.size()),
          "copy offset B");
    Parameters parameters{contract::kWidth,
                          contract::kHeight,
                          contract::kWidth,
                          contract::kHeight,
                          texture_a.texture,
                          texture_b.texture,
                          offset_a.payload(),
                          offset_b.payload(),
                          output.surface,
                          {0, 0, 0, 0, 0, 0, 0, 0}};
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 16, 16, 1, 0, nullptr, nullptr,
                         extra),
          "launch selected L54");
    Check(cuCtxSynchronize(), "synchronize L54");
    const auto texture_a_after = texture_a.CopyToHost();
    const auto texture_b_after = texture_b.CopyToHost();
    const auto output_after = output.CopyToHost();
    const auto offset_a_guarded = offset_a.ReadAll();
    const auto offset_b_guarded = offset_b.ReadAll();
    const auto offset_a_after =
        Payload(offset_a_guarded, contract::kOffsetBufferBytes);
    const auto offset_b_after =
        Payload(offset_b_guarded, contract::kOffsetBufferBytes);
    const auto input_mismatches =
        Mismatches(texture_a_before, texture_a_after) +
        Mismatches(texture_b_before, texture_b_after) +
        Mismatches(offset_a_before, offset_a_after) +
        Mismatches(offset_b_before, offset_b_after);
    const auto canary_mismatches =
        CanaryMismatches(offset_a_guarded, contract::kOffsetBufferBytes) +
        CanaryMismatches(offset_b_guarded, contract::kOffsetBufferBytes);
    Write(output_root / "texture-a-after.bin", texture_a_after);
    Write(output_root / "texture-b-after.bin", texture_b_after);
    Write(output_root / "offset-a-after.bin", offset_a_after);
    Write(output_root / "offset-b-after.bin", offset_b_after);
    Write(output_root / "output.bin", output_after);
    WriteReceipt(output_root / "run-receipt.json", run_id, poison,
                 texture_a_before, texture_b_before, offset_a_before,
                 offset_b_before, texture_a_after, texture_b_after,
                 offset_a_after, offset_b_after, output_after,
                 input_mismatches, canary_mismatches,
                 PoisonHalfwords(output_after, poison));
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_mid_l54_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(output_after) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    offset_a.Free();
    offset_b.Free();
    output.Free();
    texture_b.Free();
    texture_a.Free();
    Check(cuModuleUnload(module), "unload module");
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
