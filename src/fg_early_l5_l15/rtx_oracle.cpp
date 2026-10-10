#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include <set>

#include "contract.h"

namespace {
namespace contract = fg_early_l5_l15;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;
  std::size_t row_bytes = 0;
  std::uint32_t rows = 0;

  void Allocate(CUarray_format format, unsigned channels,
                std::size_t component_bytes, std::uint32_t height,
                bool surface_load_store) {
    rows = height;
    row_bytes = std::size_t{contract::kWidth} * channels * component_bytes;
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    descriptor.Width = contract::kWidth;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    descriptor.Flags = surface_load_store ? CUDA_ARRAY3D_SURFACE_LDST : 0u;
    Check(cuArray3DCreate(&array, &descriptor), "cuArray3DCreate");
  }

  void MakeTexture(bool normalized_coordinates, bool read_as_integer) {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC descriptor{};
    descriptor.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.filterMode = CU_TR_FILTER_MODE_POINT;
    descriptor.flags = (normalized_coordinates ? CU_TRSF_NORMALIZED_COORDINATES : 0u) |
                       (read_as_integer ? CU_TRSF_READ_AS_INTEGER : 0u);
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
    if (bytes.size() != row_bytes * rows) {
      throw std::runtime_error("array upload size mismatch");
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = row_bytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = row_bytes;
    copy.Height = rows;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(HtoA)");
  }

  std::vector<std::uint8_t> CopyToHost() const {
    std::vector<std::uint8_t> result(row_bytes * rows);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = result.data();
    copy.dstPitch = row_bytes;
    copy.WidthInBytes = row_bytes;
    copy.Height = rows;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(AtoH)");
    return result;
  }

  ~ArrayResource() {
    if (texture != 0) cuTexObjectDestroy(texture);
    if (surface != 0) cuSurfObjectDestroy(surface);
    if (array != nullptr) cuArrayDestroy(array);
  }
};

struct alignas(16) Parameters {
  CUtexObject flow_texture;
  CUtexObject depth_texture;
  CUdeviceptr atomic_output;
  CUsurfObject mask_surface;
  std::uint32_t dead32;
  std::uint32_t dead36;
  float input_scale_x;
  float input_scale_y;
  std::uint32_t width;
  std::uint32_t height;
  float motion_divisor;
  float neighbor_divisor;
  float dead64;
  float depth_scale;
  float dead72;
  std::uint8_t invert_depth;
  std::uint8_t padding77;
  std::uint16_t padding78;
  std::array<float, 16> transform;
};
static_assert(sizeof(Parameters) == 144);
static_assert(offsetof(Parameters, input_scale_x) == 40);
static_assert(offsetof(Parameters, width) == 48);
static_assert(offsetof(Parameters, transform) == 80);

struct AtomicClassification {
  std::size_t sentinel = 0;
  std::size_t candidate = 0;
  std::size_t negative_high = 0;
  std::set<std::uint32_t> high_words;
  std::set<std::uint32_t> low_words;
};

AtomicClassification ClassifyAtomic(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() != contract::kAtomicBytes) {
    throw std::runtime_error("atomic output size mismatch");
  }
  AtomicClassification result;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    if (value == contract::kAtomicSentinel) {
      ++result.sentinel;
      continue;
    }
    ++result.candidate;
    const auto high = static_cast<std::uint32_t>(value >> 32u);
    const auto low = static_cast<std::uint32_t>(value);
    result.negative_high += (high & 0x80000000u) != 0u;
    result.high_words.insert(high);
    result.low_words.insert(low);
  }
  return result;
}

struct MaskClassification {
  std::size_t zero = 0;
  std::size_t one = 0;
  std::size_t poison = 0;
  std::size_t other = 0;
};

MaskClassification ClassifyMask(const std::vector<std::uint8_t>& bytes,
                                std::uint8_t poison) {
  MaskClassification result;
  for (const auto value : bytes) {
    if (value == 0u) ++result.zero;
    else if (value == 255u) ++result.one;
    else if (value == poison) ++result.poison;
    else ++result.other;
  }
  return result;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::string& run_id, contract::Phase phase,
                  contract::Scenario scenario, std::uint8_t poison,
                  const std::vector<std::uint8_t>& flow_before,
                  const std::vector<std::uint8_t>& depth_before,
                  const std::vector<std::uint8_t>& flow_after,
                  const std::vector<std::uint8_t>& depth_after,
                  const std::vector<std::uint8_t>& atomic_output,
                  const std::vector<std::uint8_t>& mask_output,
                  std::size_t input_mismatches,
                  std::size_t atomic_canary_mismatches,
                  std::size_t mask_guard_mismatches,
                  const AtomicClassification& atomic_class,
                  const MaskClassification& mask_class) {
  const bool tie = scenario == contract::Scenario::kEqualHighTie;
  const bool classification_pass =
      atomic_class.candidate != 0 && atomic_class.negative_high == 0 &&
      atomic_class.low_words.size() > 1u &&
      (!tie || atomic_class.high_words.size() <= 2u) &&
      mask_class.zero != 0 && mask_class.one != 0 &&
      mask_class.poison == 0 && mask_class.other == 0;
  const bool pass = input_mismatches == 0 && atomic_canary_mismatches == 0 &&
                    mask_guard_mismatches == 0 && classification_pass;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L5/L15 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l5-l15-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"phase\": \"" << contract::PhaseName(phase) << "\",\n"
         << "  \"scenario\": \"" << contract::ScenarioName(scenario)
         << "\",\n"
         << "  \"selected_launch\": "
         << (phase == contract::Phase::kReset ? 5 : 15) << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 13,\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 144,\n"
         << "  \"input_generator\": \"" << contract::kInputGenerator
         << "\",\n"
         << "  \"flow_pre_sha256\": \"" << Sha256(flow_before) << "\",\n"
         << "  \"flow_post_sha256\": \"" << Sha256(flow_after) << "\",\n"
         << "  \"depth_pre_sha256\": \"" << Sha256(depth_before) << "\",\n"
         << "  \"depth_post_sha256\": \"" << Sha256(depth_after) << "\",\n"
         << "  \"atomic_output_sha256\": \"" << Sha256(atomic_output)
         << "\",\n"
         << "  \"mask_output_sha256\": \"" << Sha256(mask_output) << "\",\n"
         << "  \"atomic_output_bytes\": " << atomic_output.size() << ",\n"
         << "  \"mask_output_bytes\": " << mask_output.size() << ",\n"
         << "  \"atomic_sentinel_u64\": " << atomic_class.sentinel << ",\n"
         << "  \"atomic_candidate_u64\": " << atomic_class.candidate << ",\n"
         << "  \"atomic_negative_high_words\": " << atomic_class.negative_high
         << ",\n"
         << "  \"atomic_unique_high_words\": " << atomic_class.high_words.size()
         << ",\n"
         << "  \"atomic_unique_low_words\": " << atomic_class.low_words.size()
         << ",\n"
         << "  \"mask_zero_pixels\": " << mask_class.zero << ",\n"
         << "  \"mask_one_pixels\": " << mask_class.one << ",\n"
         << "  \"mask_poison_pixels\": " << mask_class.poison << ",\n"
         << "  \"mask_other_pixels\": " << mask_class.other << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"atomic_canary_mismatched_bytes\": "
         << atomic_canary_mismatches << ",\n"
         << "  \"mask_guard_row_mismatched_bytes\": "
         << mask_guard_mismatches << ",\n"
         << "  \"collision_fixture\": true,\n"
         << "  \"equal_high_tie_fixture\": " << (tie ? "true" : "false")
         << ",\n"
         << "  \"expected_or_archived_ingress\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-early-l5-l15-rtx <module> <output-dir> <run-id> "
                 "<poison-hex> <reset|generated> "
                 "<equal-high-tie|priority-collision>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected module13 identity mismatch");
    }
    char* poison_end = nullptr;
    const auto parsed_poison = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const auto phase = contract::ParsePhase(argv[5]);
    const auto scenario = contract::ParseScenario(argv[6]);
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const auto flow_before = contract::Flow(phase, scenario);
    const auto depth_before = contract::Depth(phase, scenario);
    const auto atomic_before = contract::InitialAtomic();
    std::vector<std::uint8_t> mask_before(contract::kMaskBytes + contract::kWidth,
                                          poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module13");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get module13 function");

    ArrayResource flow;
    ArrayResource depth;
    ArrayResource mask;
    flow.Allocate(CU_AD_FORMAT_FLOAT, 2, 4, contract::kHeight, false);
    depth.Allocate(CU_AD_FORMAT_UNSIGNED_INT32, 1, 4, contract::kHeight, false);
    mask.Allocate(CU_AD_FORMAT_UNSIGNED_INT8, 1, 1, contract::kHeight + 1u,
                  true);
    flow.CopyFromHost(flow_before);
    depth.CopyFromHost(depth_before);
    mask.CopyFromHost(mask_before);
    flow.MakeTexture(true, false);
    depth.MakeTexture(false, true);
    mask.MakeSurface();
    GuardedDeviceBuffer atomic;
    atomic.Allocate(contract::kAtomicBytes);
    Check(cuMemcpyHtoD(atomic.payload(), atomic_before.data(),
                       atomic_before.size()), "upload atomic sentinel");

    const std::array<float, 16> identity = {
        1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1,
    };
    Parameters parameters{flow.texture,
                          depth.texture,
                          atomic.payload(),
                          mask.surface,
                          0,
                          0,
                          1.0f / 640.0f,
                          1.0f / 360.0f,
                          contract::kWidth,
                          contract::kHeight,
                          144.0f,
                          12.96f,
                          40.0f,
                          1.0f,
                          600.0f,
                          0,
                          0,
                          0,
                          identity};
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 16, 16, 1, 0, nullptr,
                         nullptr, extra), "launch module13");
    Check(cuCtxSynchronize(), "synchronize module13");

    const auto flow_after = flow.CopyToHost();
    const auto depth_after = depth.CopyToHost();
    const auto guarded_atomic = atomic.ReadAll();
    const auto atomic_output = Payload(guarded_atomic, contract::kAtomicBytes);
    const auto guarded_mask = mask.CopyToHost();
    const std::vector<std::uint8_t> mask_output(
        guarded_mask.begin(), guarded_mask.begin() + contract::kMaskBytes);
    const std::size_t input_mismatches =
        Mismatches(flow_before, flow_after) +
        Mismatches(depth_before, depth_after);
    const std::size_t atomic_canary_mismatches =
        CanaryMismatches(guarded_atomic, contract::kAtomicBytes);
    std::size_t mask_guard_mismatches = 0;
    for (std::size_t index = contract::kMaskBytes;
         index < guarded_mask.size(); ++index) {
      mask_guard_mismatches += guarded_mask[index] != poison;
    }
    const auto atomic_class = ClassifyAtomic(atomic_output);
    const auto mask_class = ClassifyMask(mask_output, poison);

    Write(output_root / "flow-after.bin", flow_after);
    Write(output_root / "depth-after.bin", depth_after);
    Write(output_root / "atomic-output.bin", atomic_output);
    Write(output_root / "mask-output.bin", mask_output);
    WriteReceipt(output_root / "run-receipt.json", run_id, phase, scenario,
                 poison, flow_before, depth_before, flow_after, depth_after,
                 atomic_output, mask_output, input_mismatches,
                 atomic_canary_mismatches, mask_guard_mismatches,
                 atomic_class, mask_class);
    const bool tie = scenario == contract::Scenario::kEqualHighTie;
    const bool pass = input_mismatches == 0 && atomic_canary_mismatches == 0 &&
                      mask_guard_mismatches == 0 &&
                      atomic_class.candidate != 0 &&
                      atomic_class.negative_high == 0 &&
                      atomic_class.low_words.size() > 1u &&
                      (!tie || atomic_class.high_words.size() <= 2u) &&
                      mask_class.zero != 0 && mask_class.one != 0 &&
                      mask_class.poison == 0 && mask_class.other == 0;
    std::cout << "probe=fg_early_l5_l15_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "atomic_output_sha256=" << Sha256(atomic_output) << '\n'
              << "mask_output_sha256=" << Sha256(mask_output) << '\n'
              << "atomic_sentinel_u64=" << atomic_class.sentinel << '\n'
              << "atomic_unique_high_words=" << atomic_class.high_words.size()
              << '\n'
              << "atomic_unique_low_words=" << atomic_class.low_words.size()
              << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
