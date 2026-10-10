#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {
namespace contract = fg_mid_l53;

struct TextureResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  std::size_t row_bytes = 0;

  void Allocate(CUarray_format format, unsigned channels,
                std::size_t component_bytes) {
    row_bytes = std::size_t{contract::kWidth} * channels * component_bytes;
    CUDA_ARRAY_DESCRIPTOR descriptor{};
    descriptor.Width = contract::kWidth;
    descriptor.Height = contract::kHeight;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    Check(cuArrayCreate(&array, &descriptor), "cuArrayCreate");
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC texture_descriptor{};
    texture_descriptor.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    texture_descriptor.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    texture_descriptor.filterMode = CU_TR_FILTER_MODE_POINT;
    Check(cuTexObjectCreate(&texture, &resource, &texture_descriptor, nullptr),
          "cuTexObjectCreate");
  }

  void CopyFromHost(const std::vector<std::uint8_t>& bytes) const {
    if (bytes.size() != row_bytes * contract::kHeight) {
      throw std::runtime_error("texture upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = row_bytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = row_bytes;
    copy.Height = contract::kHeight;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(HtoA)");
  }

  std::vector<std::uint8_t> CopyToHost() const {
    std::vector<std::uint8_t> bytes(row_bytes * contract::kHeight);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = row_bytes;
    copy.WidthInBytes = row_bytes;
    copy.Height = contract::kHeight;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(AtoH)");
    return bytes;
  }

  void Free() {
    if (texture != 0) {
      Check(cuTexObjectDestroy(texture), "cuTexObjectDestroy");
      texture = 0;
    }
    if (array != nullptr) {
      Check(cuArrayDestroy(array), "cuArrayDestroy");
      array = nullptr;
    }
  }

  ~TextureResource() {
    if (texture != 0) cuTexObjectDestroy(texture);
    if (array != nullptr) cuArrayDestroy(array);
  }
};

struct Parameters {
  CUtexObject curr_to_prev_flow;
  CUtexObject prev_to_curr_flow;
  CUtexObject curr_depth;
  CUtexObject prev_depth;
  float interpolation_factor;
  std::uint32_t padding36;
  CUdeviceptr curr_to_prev_output;
  CUdeviceptr prev_to_curr_output;
  std::array<std::uint8_t, 56> reserved56;
  std::uint32_t width;
  std::uint32_t height;
  float motion_threshold;
  float depth_threshold;
  float depth_scale;
  float distance_threshold;
  std::uint8_t invert_depth;
  std::array<std::uint8_t, 7> reserved137;
};
static_assert(sizeof(Parameters) == 144);
static_assert(offsetof(Parameters, curr_to_prev_output) == 40);
static_assert(offsetof(Parameters, width) == 112);
static_assert(offsetof(Parameters, invert_depth) == 136);

std::size_t SentinelCount(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() % 8u != 0u) {
    throw std::runtime_error("sentinel scan size mismatch");
  }
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    count += value == contract::kInitialOutput;
  }
  return count;
}

std::size_t PoisonCanaryMismatches(
    const std::vector<std::uint8_t>& guarded, std::size_t payload_bytes,
    std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t index = 0; index < kGuardBytes; ++index) {
    count += guarded[index] != poison;
    count += guarded[kGuardBytes + payload_bytes + index] != poison;
  }
  return count;
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const std::string& scenario, std::uint8_t poison,
    const std::array<std::vector<std::uint8_t>, 4>& inputs_before,
    const std::array<std::vector<std::uint8_t>, 4>& inputs_after,
    const std::vector<std::uint8_t>& output_a,
    const std::vector<std::uint8_t>& output_b,
    std::size_t input_mismatches, std::size_t canary_mismatches,
    std::size_t sentinel_a, std::size_t sentinel_b) {
  const bool pass = input_mismatches == 0 && canary_mismatches == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L53 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.mid-l53-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"scenario\": \"" << scenario << "\",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"selected_launch\": 53,\n"
         << "  \"module_token\": 12,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"raw_live_parameter_sha256\": \""
         << contract::kRawParameterSha256 << "\",\n"
         << "  \"input_generator\": \"" << contract::kInputGenerator
         << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [324, 1, 1],\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"dimensions\": [320, 180],\n"
         << "  \"input_pre_sha256\": [";
  for (std::size_t index = 0; index < inputs_before.size(); ++index) {
    stream << (index == 0 ? "\"" : ", \"")
           << Sha256(inputs_before[index]) << "\"";
  }
  stream << "],\n  \"input_post_sha256\": [";
  for (std::size_t index = 0; index < inputs_after.size(); ++index) {
    stream << (index == 0 ? "\"" : ", \"")
           << Sha256(inputs_after[index]) << "\"";
  }
  stream << "],\n"
         << "  \"output_bytes_each\": " << contract::kOutputBytes << ",\n"
         << "  \"output_sha256\": [\"" << Sha256(output_a) << "\", \""
         << Sha256(output_b) << "\"],\n"
         << "  \"sentinel_u64_count\": [" << sentinel_a << ", "
         << sentinel_b << "],\n"
         << "  \"non_sentinel_u64_count\": ["
         << (contract::kPixelCount - sentinel_a) << ", "
         << (contract::kPixelCount - sentinel_b) << "],\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"canary_mismatched_bytes\": " << canary_mismatches << ",\n"
         << "  \"output_initial_value\": \"signed-int64-max sentinel\",\n"
         << "  \"expected_output_opened\": false,\n"
         << "  \"archived_intermediate_opened\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage=fg-mid-l53-rtx <module.bin> <output-dir> <run-id> "
                 "<poison-hex> <small-flow|pixel-flow>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected L53 module identity mismatch");
    }
    char* poison_end = nullptr;
    const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' ||
        parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::string scenario = argv[5];
    if (scenario != "small-flow" && scenario != "pixel-flow") {
      throw std::runtime_error("invalid L53 scenario");
    }
    const bool pixel_flow = scenario == "pixel-flow";
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const std::array<std::vector<std::uint8_t>, 4> inputs_before = {
        contract::CurrToPrevFlow(), contract::PrevToCurrFlow(pixel_flow),
        contract::CurrDepth(), contract::PrevDepth()};
    const auto initial_output = contract::InitialOutput();

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load L53 module");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get L53 function");

    TextureResource curr_to_prev_flow;
    TextureResource prev_to_curr_flow;
    TextureResource curr_depth;
    TextureResource prev_depth;
    curr_to_prev_flow.Allocate(CU_AD_FORMAT_FLOAT, 2, 4);
    prev_to_curr_flow.Allocate(CU_AD_FORMAT_HALF, 2, 2);
    curr_depth.Allocate(CU_AD_FORMAT_UNSIGNED_INT32, 1, 4);
    prev_depth.Allocate(CU_AD_FORMAT_UNSIGNED_INT32, 1, 4);
    curr_to_prev_flow.CopyFromHost(inputs_before[0]);
    prev_to_curr_flow.CopyFromHost(inputs_before[1]);
    curr_depth.CopyFromHost(inputs_before[2]);
    prev_depth.CopyFromHost(inputs_before[3]);

    GuardedDeviceBuffer output_a;
    GuardedDeviceBuffer output_b;
    output_a.Allocate(contract::kOutputBytes);
    output_b.Allocate(contract::kOutputBytes);
    Check(cuMemsetD8(output_a.base, poison,
                     contract::kOutputBytes + 2u * kGuardBytes),
          "poison output A allocation");
    Check(cuMemsetD8(output_b.base, poison,
                     contract::kOutputBytes + 2u * kGuardBytes),
          "poison output B allocation");
    Check(cuMemcpyHtoD(output_a.payload(), initial_output.data(),
                       initial_output.size()), "initialize output A");
    Check(cuMemcpyHtoD(output_b.payload(), initial_output.data(),
                       initial_output.size()), "initialize output B");

    Parameters parameters{};
    parameters.curr_to_prev_flow = curr_to_prev_flow.texture;
    parameters.prev_to_curr_flow = prev_to_curr_flow.texture;
    parameters.curr_depth = curr_depth.texture;
    parameters.prev_depth = prev_depth.texture;
    parameters.interpolation_factor = 0.5f;
    parameters.curr_to_prev_output = output_a.payload();
    parameters.prev_to_curr_output = output_b.payload();
    parameters.width = contract::kWidth;
    parameters.height = contract::kHeight;
    parameters.motion_threshold = 12.96000003814697265625f;
    parameters.depth_threshold = 40.0f;
    parameters.depth_scale = 1.0f;
    parameters.distance_threshold = 600.0f;
    parameters.invert_depth = 0;
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 324, 1, 1, 0, nullptr,
                         nullptr, extra), "launch selected L53");
    Check(cuCtxSynchronize(), "synchronize selected L53");

    const std::array<std::vector<std::uint8_t>, 4> inputs_after = {
        curr_to_prev_flow.CopyToHost(), prev_to_curr_flow.CopyToHost(),
        curr_depth.CopyToHost(), prev_depth.CopyToHost()};
    const auto guarded_a = output_a.ReadAll();
    const auto guarded_b = output_b.ReadAll();
    const auto actual_a = Payload(guarded_a, contract::kOutputBytes);
    const auto actual_b = Payload(guarded_b, contract::kOutputBytes);
    std::size_t input_mismatches = 0;
    for (std::size_t index = 0; index < inputs_before.size(); ++index) {
      input_mismatches += Mismatches(inputs_before[index], inputs_after[index]);
    }
    const auto canary_mismatches =
        PoisonCanaryMismatches(guarded_a, contract::kOutputBytes, poison) +
        PoisonCanaryMismatches(guarded_b, contract::kOutputBytes, poison);
    const auto sentinel_a = SentinelCount(actual_a);
    const auto sentinel_b = SentinelCount(actual_b);
    for (std::size_t index = 0; index < inputs_after.size(); ++index) {
      Write(output_root / ("input-" + std::to_string(index) + "-after.bin"),
            inputs_after[index]);
    }
    Write(output_root / "output-a.bin", actual_a);
    Write(output_root / "output-b.bin", actual_b);
    WriteReceipt(output_root / "run-receipt.json", run_id, scenario, poison,
                 inputs_before, inputs_after, actual_a, actual_b,
                 input_mismatches, canary_mismatches, sentinel_a, sentinel_b);
    const bool pass = input_mismatches == 0 && canary_mismatches == 0;
    std::cout << "probe=fg_mid_l53_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_a_sha256=" << Sha256(actual_a) << '\n'
              << "output_b_sha256=" << Sha256(actual_b) << '\n'
              << "sentinel_a=" << sentinel_a << '\n'
              << "sentinel_b=" << sentinel_b << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';

    output_a.Free();
    output_b.Free();
    curr_to_prev_flow.Free();
    prev_to_curr_flow.Free();
    curr_depth.Free();
    prev_depth.Free();
    Check(cuModuleUnload(module), "unload L53 module");
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
