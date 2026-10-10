// Preserved negative control: generic CUDA arrays do not reproduce the
// selected D3D12 RGBA8_UNORM formatted-surface semantics. The accepted RTX
// oracle is rtx_nvapi_oracle.cpp.
#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {
namespace l11 = fg_early_l11;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;

  void Allocate(bool surface_load_store) {
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    descriptor.Width = l11::kTextureWidth;
    descriptor.Height = l11::kTextureHeight;
    descriptor.Depth = 0;
    descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    descriptor.NumChannels = 4;
    descriptor.Flags = surface_load_store ? CUDA_ARRAY3D_SURFACE_LDST : 0u;
    Check(cuArray3DCreate(&array, &descriptor), "cuArray3DCreate(RGBA8)");
  }

  void MakeTexture() {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC descriptor{};
    descriptor.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.filterMode = CU_TR_FILTER_MODE_POINT;
    descriptor.flags = 0;
    Check(cuTexObjectCreate(&texture, &resource, &descriptor, nullptr),
          "cuTexObjectCreate(RGBA8_UNORM)");
  }

  void MakeSurface() {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    Check(cuSurfObjectCreate(&surface, &resource),
          "cuSurfObjectCreate(RGBA8_UNORM)");
  }

  void CopyFromHost(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() != l11::kTextureBytes) {
      throw std::runtime_error("L11 array upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = l11::kTextureWidth * 4u;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = l11::kTextureWidth * 4u;
    copy.Height = l11::kTextureHeight;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(HtoA RGBA8)");
  }

  std::vector<std::uint8_t> CopyToHost() const {
    std::vector<std::uint8_t> bytes(l11::kTextureBytes);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = l11::kTextureWidth * 4u;
    copy.WidthInBytes = l11::kTextureWidth * 4u;
    copy.Height = l11::kTextureHeight;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(AtoH RGBA8)");
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

template <std::size_t Size>
void Pack32(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
            std::uint32_t value) {
  if (offset + sizeof(value) > Size) throw std::runtime_error("Pack32 OOB");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void Pack64(std::array<std::uint8_t, Size>* bytes, std::size_t offset,
            std::uint64_t value) {
  if (offset + sizeof(value) > Size) throw std::runtime_error("Pack64 OOB");
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::size_t OutsideMismatches(const std::vector<std::uint8_t>& output,
                              std::uint8_t poison) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kTextureHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kTextureWidth; ++x) {
      if (l11::InOutputRegion(x, y)) continue;
      const std::size_t offset =
          (std::size_t{y} * l11::kTextureWidth + x) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        count += output[offset + channel] != poison;
      }
    }
  }
  return count;
}

std::size_t RegionMismatches(const std::vector<std::uint8_t>& input,
                             const std::vector<std::uint8_t>& output) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kCopyWidth; ++x) {
      const std::size_t source =
          (std::size_t{y + l11::kInputOriginY} * l11::kTextureWidth +
           x + l11::kInputOriginX) * 4u;
      const std::size_t destination =
          (std::size_t{y + l11::kOutputOriginY} * l11::kTextureWidth +
           x + l11::kOutputOriginX) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        count += input[source + channel] != output[destination + channel];
      }
    }
  }
  return count;
}

std::size_t RegionPoisonPixels(const std::vector<std::uint8_t>& output,
                               std::uint8_t poison) {
  std::size_t count = 0;
  for (std::uint32_t y = 0; y < l11::kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < l11::kCopyWidth; ++x) {
      const std::size_t offset =
          (std::size_t{y + l11::kOutputOriginY} * l11::kTextureWidth +
           x + l11::kOutputOriginX) * 4u;
      count += output[offset] == poison && output[offset + 1u] == poison &&
               output[offset + 2u] == poison &&
               output[offset + 3u] == poison;
    }
  }
  return count;
}

void WriteRunReceipt(const std::filesystem::path& path,
                     const std::string& run_id, std::uint8_t poison,
                     const std::string& device_name,
                     const std::vector<std::uint8_t>& input_before,
                     const std::vector<std::uint8_t>& input_after,
                     const std::vector<std::uint8_t>& output,
                     const std::vector<std::uint8_t>& region,
                     std::size_t input_mismatches,
                     std::size_t region_mismatches,
                     std::size_t outside_mismatches,
                     std::size_t expected_mismatches,
                     std::size_t region_poison_pixels) {
  const bool pass = input_mismatches == 0 && region_mismatches == 0 &&
                    outside_mismatches == 0 && expected_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L11 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l11-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"device_name\": \"" << device_name << "\",\n"
         << "  \"selected_launch\": 11,\n"
         << "  \"module_token\": 4,\n"
         << "  \"module_bytes\": " << l11::kModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << l11::kModuleSha256 << "\",\n"
         << "  \"module_transport\": \"cuModuleLoadData exact selected fatbin\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [40, 23, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 40,\n"
         << "  \"frozen_selected_raw_parameter_sha256\": \""
         << l11::kFrozenRawParameterSha256 << "\",\n"
         << "  \"frozen_selected_address_free_parameter_sha256\": \""
         << l11::kFrozenAddressFreeParameterSha256 << "\",\n"
         << "  \"fixture_address_free_parameter_sha256\": \""
         << l11::kFixtureAddressFreeParameterSha256 << "\",\n"
         << "  \"resource_format\": \"RGBA8_UNORM\",\n"
         << "  \"resource_extent\": [640, 360],\n"
         << "  \"input_origin\": [17, 11],\n"
         << "  \"output_origin\": [29, 19],\n"
         << "  \"copy_extent\": [587, 317],\n"
         << "  \"input_generator\": \"" << l11::kInputGenerator << "\",\n"
         << "  \"input_pre_sha256\": \"" << Sha256(input_before) << "\",\n"
         << "  \"input_post_sha256\": \"" << Sha256(input_after) << "\",\n"
         << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
         << "  \"region_output_sha256\": \"" << Sha256(region) << "\",\n"
         << "  \"output_bytes\": " << output.size() << ",\n"
         << "  \"declared_region_bytes\": " << region.size() << ",\n"
         << "  \"outside_region_pixels\": " << l11::kOutsidePixels << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"declared_region_mismatched_bytes\": " << region_mismatches
         << ",\n"
         << "  \"outside_region_mismatched_bytes\": " << outside_mismatches
         << ",\n"
         << "  \"canary_mismatched_bytes\": " << outside_mismatches << ",\n"
         << "  \"post_execution_expected_mismatched_bytes\": "
         << expected_mismatches << ",\n"
         << "  \"declared_region_poison_pixels\": " << region_poison_pixels
         << ",\n"
         << "  \"input_immutable\": "
         << (input_mismatches == 0 ? "true" : "false") << ",\n"
         << "  \"expected_bytes_fed_to_execution\": false,\n"
         << "  \"archived_intermediates_consumed\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-early-l11-rtx <module.bin> <output-dir> <run-id> "
                 "<poison-hex>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != l11::kModuleBytes ||
        Sha256(module_bytes) != l11::kModuleSha256) {
      throw std::runtime_error("selected L11 module identity mismatch");
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

    const auto input_before = l11::GenerateInput();
    const std::vector<std::uint8_t> output_before(l11::kTextureBytes, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    char device_name[256]{};
    Check(cuDeviceGetName(device_name, sizeof(device_name), device),
          "cuDeviceGetName");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()),
          "cuModuleLoadData(exact selected module4)");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "cuModuleGetFunction(main_kernel)");

    ArrayResource input;
    ArrayResource output;
    input.Allocate(false);
    output.Allocate(true);
    input.CopyFromHost(input_before);
    output.CopyFromHost(output_before);
    input.MakeTexture();
    output.MakeSurface();

    std::array<std::uint8_t, l11::kPackedParameterBytes> packed{};
    Pack64(&packed, 0, input.texture);
    Pack32(&packed, 8, l11::kInputOriginX);
    Pack32(&packed, 12, l11::kInputOriginY);
    Pack64(&packed, 16, output.surface);
    Pack32(&packed, 24, l11::kOutputOriginX);
    Pack32(&packed, 28, l11::kOutputOriginY);
    Pack32(&packed, 32, l11::kCopyWidth);
    Pack32(&packed, 36, l11::kCopyHeight);
    std::size_t packed_size = packed.size();
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        packed.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        &packed_size,
        CU_LAUNCH_PARAM_END,
    };
    Check(cuLaunchKernel(function, 40, 23, 1, 16, 16, 1, 0, nullptr,
                         nullptr, extra),
          "cuLaunchKernel(module4/main_kernel)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto input_after = input.CopyToHost();
    const auto output_after = output.CopyToHost();
    const auto expected = l11::ExpectedOutput(input_before, poison);
    const auto region = l11::ExtractRegion(output_after);
    const std::size_t input_mismatches = Mismatches(input_before, input_after);
    const std::size_t region_mismatches =
        RegionMismatches(input_before, output_after);
    const std::size_t outside_mismatches =
        OutsideMismatches(output_after, poison);
    const std::size_t expected_mismatches = Mismatches(expected, output_after);
    const std::size_t region_poison_pixels =
        RegionPoisonPixels(output_after, poison);

    Write(output_root / "input-after.rgba8", input_after);
    Write(output_root / "output.rgba8", output_after);
    Write(output_root / "region-output.rgba8", region);
    WriteRunReceipt(output_root / "run-receipt.json", run_id, poison,
                    device_name, input_before, input_after, output_after, region,
                    input_mismatches, region_mismatches, outside_mismatches,
                    expected_mismatches, region_poison_pixels);

    const bool pass = input_mismatches == 0 && region_mismatches == 0 &&
                      outside_mismatches == 0 && expected_mismatches == 0;
    std::cout << "probe=fg_early_l11_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "region_output_sha256=" << Sha256(region) << '\n'
              << "output_sha256=" << Sha256(output_after) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "declared_region_mismatched_bytes=" << region_mismatches
              << '\n'
              << "outside_region_mismatched_bytes=" << outside_mismatches
              << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';

    input.Free();
    output.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
