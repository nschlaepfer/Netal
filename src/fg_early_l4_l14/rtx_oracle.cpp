#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

#include <set>

namespace {
namespace contract = fg_early_l4_l14;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_bytes = 0;

  void Allocate(std::uint32_t new_width, std::uint32_t new_height,
                CUarray_format format, unsigned channels,
                std::size_t component_bytes) {
    width = new_width;
    height = new_height;
    row_bytes = static_cast<std::size_t>(width) * channels * component_bytes;
    CUDA_ARRAY_DESCRIPTOR descriptor{};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    Check(cuArrayCreate(&array, &descriptor), "cuArrayCreate");
  }

  void MakeTexture(CUaddress_mode address_mode) {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC descriptor{};
    descriptor.addressMode[0] = address_mode;
    descriptor.addressMode[1] = address_mode;
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
  CUtexObject vector_input;
  std::array<std::uint32_t, 2> vector_origin;
  std::array<std::uint32_t, 2> vector_extent;
  std::array<float, 2> vector_inverse_extent;
  CUsurfObject vector_output;
  CUtexObject scalar_input;
  std::array<std::uint32_t, 2> scalar_origin;
  std::array<std::uint32_t, 2> scalar_extent;
  std::array<float, 2> scalar_inverse_extent;
  float abi_dead_threshold;
  float score_numerator;
  float abi_dead_scale;
  std::uint8_t mode;
  std::array<std::uint8_t, 3> mode_padding;
  CUsurfObject depth_output;
  std::array<std::uint64_t, 6> abi_dead;
  std::array<std::uint32_t, 2> output_extent;
  std::array<float, 2> vector_output_scale;
};
static_assert(sizeof(Parameters) == 160);
static_assert(offsetof(Parameters, vector_output) == 32);
static_assert(offsetof(Parameters, scalar_input) == 40);
static_assert(offsetof(Parameters, mode) == 84);
static_assert(offsetof(Parameters, depth_output) == 88);
static_assert(offsetof(Parameters, output_extent) == 144);

Parameters MakeParameters(CUtexObject vector_input, CUsurfObject vector_output,
                          CUtexObject scalar_input, CUsurfObject depth_output,
                          bool reset) {
  return Parameters{
      vector_input,
      {0, 0},
      {contract::kInputWidth, contract::kInputHeight},
      {std::bit_cast<float>(0x3acccccdu),
       std::bit_cast<float>(0x3b360b61u)},
      vector_output,
      scalar_input,
      {0, 0},
      {contract::kInputWidth, contract::kInputHeight},
      {std::bit_cast<float>(0x3acccccdu),
       std::bit_cast<float>(0x3b360b61u)},
      40.0f,
      1.0f,
      600.0f,
      static_cast<std::uint8_t>(reset ? 127u : 0u),
      {0, 0, 0},
      depth_output,
      {0, 0, 0, 0, 0, 0},
      {contract::kOutputWidth, contract::kOutputHeight},
      {std::bit_cast<float>(0x3623d70au),
       std::bit_cast<float>(0x3701742eu)}};
}

std::vector<std::uint8_t> BytesOf(const Parameters& value) {
  const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
  return std::vector<std::uint8_t>(first, first + sizeof(value));
}

std::vector<std::uint8_t> SurfacePayload(
    const std::vector<std::uint8_t>& guarded, std::size_t payload_bytes) {
  return std::vector<std::uint8_t>(guarded.begin(),
                                   guarded.begin() + payload_bytes);
}

std::size_t TailCanaryMismatches(const std::vector<std::uint8_t>& guarded,
                                 std::size_t payload_bytes,
                                 std::uint8_t poison) {
  std::size_t mismatches = 0;
  for (std::size_t index = payload_bytes; index < guarded.size(); ++index) {
    mismatches += guarded[index] != poison;
  }
  return mismatches;
}

std::size_t UntouchedPixels(const std::vector<std::uint8_t>& payload,
                            std::size_t pixel_bytes, std::uint8_t poison) {
  std::size_t untouched = 0;
  for (std::size_t offset = 0; offset < payload.size(); offset += pixel_bytes) {
    bool same = true;
    for (std::size_t byte = 0; byte < pixel_bytes; ++byte) {
      same = same && payload[offset + byte] == poison;
    }
    untouched += same;
  }
  return untouched;
}

std::size_t UniqueWords(const std::vector<std::uint8_t>& bytes) {
  std::set<std::uint32_t> words;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 4u) {
    std::uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    words.insert(word);
  }
  return words.size();
}

void WriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const std::string& phase, contract::Fixture fixture, std::uint8_t poison,
    const std::vector<std::uint8_t>& vector_before,
    const std::vector<std::uint8_t>& vector_after,
    const std::vector<std::uint8_t>& scalar_before,
    const std::vector<std::uint8_t>& scalar_after,
    const std::vector<std::uint8_t>& vector_output,
    const std::vector<std::uint8_t>& depth_output,
    const std::string& fixture_address_free_sha,
    std::size_t input_mismatches,
    std::size_t vector_canary_mismatches,
    std::size_t depth_canary_mismatches,
    std::size_t vector_untouched_pixels,
    std::size_t depth_untouched_pixels) {
  const bool reset = phase == "reset";
  const bool pass = input_mismatches == 0 && vector_canary_mismatches == 0 &&
                    depth_canary_mismatches == 0 &&
                    vector_untouched_pixels == 0 &&
                    depth_untouched_pixels == 0;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write L4/L14 receipt");
  stream << "{\n"
         << "  \"schema\": \"neuralstudio.fg.early-l4-l14-run.v1\",\n"
         << "  \"backend\": \"rtx-cuda-driver\",\n"
         << "  \"run_id\": \"" << run_id << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"phase\": \"" << phase << "\",\n"
         << "  \"fixture\": \"" << contract::FixtureName(fixture) << "\",\n"
         << "  \"selected_launch\": " << (reset ? 4 : 14) << ",\n"
         << "  \"poison_byte\": " << unsigned(poison) << ",\n"
         << "  \"module_token\": 15,\n"
         << "  \"module_bytes\": " << contract::kModuleBytes << ",\n"
         << "  \"module_sha256\": \"" << contract::kModuleSha256 << "\",\n"
         << "  \"function_name\": \"main_kernel\",\n"
         << "  \"grid\": [20, 12, 1],\n"
         << "  \"block\": [16, 16, 1],\n"
         << "  \"packed_parameter_bytes\": 160,\n"
         << "  \"frozen_selected_raw_parameter_sha256\": \""
         << (reset ? contract::kResetTraceRawParameterSha256
                   : contract::kGeneratedTraceRawParameterSha256)
         << "\",\n"
         << "  \"frozen_selected_address_free_parameter_sha256\": \""
         << (reset ? contract::kResetAddressFreeParameterSha256
                   : contract::kGeneratedAddressFreeParameterSha256)
         << "\",\n"
         << "  \"fixture_address_free_parameter_sha256\": \""
         << fixture_address_free_sha << "\",\n"
         << "  \"input_generator\": \"" << contract::kInputGenerator
         << "\",\n"
         << "  \"resource_contract\": {\"vector_input\": \"RG16_FLOAT 640x360 point-clamp\", \"scalar_input\": \"R32_FLOAT 640x360 point-border\", \"vector_output\": \"RG32_FLOAT 320x180\", \"depth_output\": \"R32_UINT 320x180\"},\n"
         << "  \"vector_input_pre_sha256\": \"" << Sha256(vector_before)
         << "\",\n"
         << "  \"vector_input_post_sha256\": \"" << Sha256(vector_after)
         << "\",\n"
         << "  \"scalar_input_pre_sha256\": \"" << Sha256(scalar_before)
         << "\",\n"
         << "  \"scalar_input_post_sha256\": \"" << Sha256(scalar_after)
         << "\",\n"
         << "  \"vector_output_bytes\": " << vector_output.size() << ",\n"
         << "  \"vector_output_sha256\": \"" << Sha256(vector_output)
         << "\",\n"
         << "  \"depth_output_bytes\": " << depth_output.size() << ",\n"
         << "  \"depth_output_sha256\": \"" << Sha256(depth_output)
         << "\",\n"
         << "  \"depth_unique_u32_values\": " << UniqueWords(depth_output)
         << ",\n"
         << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
         << "  \"vector_canary_mismatched_bytes\": "
         << vector_canary_mismatches << ",\n"
         << "  \"depth_canary_mismatched_bytes\": "
         << depth_canary_mismatches << ",\n"
         << "  \"vector_untouched_pixels\": " << vector_untouched_pixels
         << ",\n"
         << "  \"depth_untouched_pixels\": " << depth_untouched_pixels
         << ",\n"
         << "  \"expected_output_fed_to_execution\": false,\n"
         << "  \"archived_intermediates_consumed\": false,\n"
         << "  \"address_fields_recorded\": false,\n"
         << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage=fg-early-l4-l14-rtx <module> <output-dir> <run-id> "
                 "<poison-hex> <reset|generated> <branch-grid|center-tie>\n";
    return 2;
  }
  try {
    const std::string phase = argv[5];
    if (phase != "reset" && phase != "generated") {
      throw std::runtime_error("unsupported L4/L14 phase");
    }
    const auto fixture = contract::ParseFixture(argv[6]);
    char* poison_end = nullptr;
    const unsigned long parsed_poison =
        std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || parsed_poison > 255u) {
      throw std::runtime_error("invalid poison byte");
    }
    const auto poison = static_cast<std::uint8_t>(parsed_poison);
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    std::filesystem::create_directories(output_root);
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256) {
      throw std::runtime_error("selected module15 identity mismatch");
    }
    const auto vector_before = contract::VectorInput(fixture);
    const auto scalar_before = contract::ScalarInput(fixture);
    const std::size_t vector_guarded_bytes =
        static_cast<std::size_t>(contract::kOutputWidth) *
        contract::kOutputRowsWithCanary * 8u;
    const std::size_t depth_guarded_bytes =
        static_cast<std::size_t>(contract::kOutputWidth) *
        contract::kOutputRowsWithCanary * 4u;
    const std::vector<std::uint8_t> vector_poison(vector_guarded_bytes,
                                                   poison);
    const std::vector<std::uint8_t> depth_poison(depth_guarded_bytes, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module15");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get main_kernel");

    ArrayResource vector_input;
    ArrayResource scalar_input;
    ArrayResource vector_output;
    ArrayResource depth_output;
    vector_input.Allocate(contract::kInputWidth, contract::kInputHeight,
                          CU_AD_FORMAT_HALF, 2, 2);
    scalar_input.Allocate(contract::kInputWidth, contract::kInputHeight,
                          CU_AD_FORMAT_FLOAT, 1, 4);
    vector_output.Allocate(contract::kOutputWidth,
                           contract::kOutputRowsWithCanary,
                           CU_AD_FORMAT_FLOAT, 2, 4);
    depth_output.Allocate(contract::kOutputWidth,
                          contract::kOutputRowsWithCanary,
                          CU_AD_FORMAT_UNSIGNED_INT32, 1, 4);
    vector_input.CopyFromHost(vector_before);
    scalar_input.CopyFromHost(scalar_before);
    vector_output.CopyFromHost(vector_poison);
    depth_output.CopyFromHost(depth_poison);
    vector_input.MakeTexture(CU_TR_ADDRESS_MODE_CLAMP);
    scalar_input.MakeTexture(CU_TR_ADDRESS_MODE_BORDER);
    vector_output.MakeSurface();
    depth_output.MakeSurface();

    const bool reset = phase == "reset";
    Parameters parameters = MakeParameters(
        vector_input.texture, vector_output.surface, scalar_input.texture,
        depth_output.surface, reset);
    Parameters address_free = parameters;
    address_free.vector_input = 0;
    address_free.vector_output = 0;
    address_free.scalar_input = 0;
    address_free.depth_output = 0;
    const std::string fixture_address_free_sha = Sha256(BytesOf(address_free));
    const char* expected_fixture_address_free =
        reset ? contract::kResetFixtureAddressFreeParameterSha256
              : contract::kGeneratedFixtureAddressFreeParameterSha256;
    if (fixture_address_free_sha != expected_fixture_address_free) {
      throw std::runtime_error("fixture address-free parameter identity mismatch");
    }
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 20, 12, 1, 16, 16, 1, 0, nullptr, nullptr,
                         extra),
          "launch selected module15");
    Check(cuCtxSynchronize(), "synchronize module15");

    const auto vector_after = vector_input.CopyToHost();
    const auto scalar_after = scalar_input.CopyToHost();
    const auto vector_guarded = vector_output.CopyToHost();
    const auto depth_guarded = depth_output.CopyToHost();
    const auto vector_payload = SurfacePayload(
        vector_guarded, contract::VectorOutputPayloadBytes());
    const auto depth_payload = SurfacePayload(
        depth_guarded, contract::DepthOutputPayloadBytes());
    const std::size_t input_mismatches =
        Mismatches(vector_before, vector_after) +
        Mismatches(scalar_before, scalar_after);
    const std::size_t vector_canary_mismatches = TailCanaryMismatches(
        vector_guarded, vector_payload.size(), poison);
    const std::size_t depth_canary_mismatches = TailCanaryMismatches(
        depth_guarded, depth_payload.size(), poison);
    const std::size_t vector_untouched =
        UntouchedPixels(vector_payload, 8u, poison);
    const std::size_t depth_untouched =
        UntouchedPixels(depth_payload, 4u, poison);
    Write(output_root / "vector-output.bin", vector_payload);
    Write(output_root / "depth-output.bin", depth_payload);
    WriteReceipt(output_root / "run-receipt.json", run_id, phase, fixture,
                 poison, vector_before, vector_after, scalar_before,
                 scalar_after, vector_payload, depth_payload,
                 fixture_address_free_sha, input_mismatches,
                 vector_canary_mismatches,
                 depth_canary_mismatches, vector_untouched, depth_untouched);
    const bool pass = input_mismatches == 0 &&
                      vector_canary_mismatches == 0 &&
                      depth_canary_mismatches == 0 &&
                      vector_untouched == 0 && depth_untouched == 0;
    std::cout << "probe=fg_early_l4_l14_rtx\n"
              << "run_id=" << run_id << '\n'
              << "vector_output_sha256=" << Sha256(vector_payload) << '\n'
              << "depth_output_sha256=" << Sha256(depth_payload) << '\n'
              << "result=" << (pass ? "pass" : "fail") << '\n';
    depth_output.Free();
    vector_output.Free();
    scalar_input.Free();
    vector_input.Free();
    Check(cuModuleUnload(module), "unload module15");
    Check(cuDevicePrimaryCtxRelease(device), "release context");
    return pass ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
