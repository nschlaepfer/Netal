#define main FgDynamicModule15EmbeddedMain
#include "../fg_early_l4_l14/rtx_oracle.cpp"
#undef main

#include "contract.h"

namespace {
namespace dynamic_contract = fg_dynamic_module15;

struct DynamicSourceParameters {
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
static_assert(sizeof(DynamicSourceParameters) == 160);
static_assert(offsetof(DynamicSourceParameters, mode) == 84);
static_assert(offsetof(DynamicSourceParameters, depth_output) == 88);

DynamicSourceParameters DynamicMakeParameters(
    const dynamic_contract::Resolution& resolution, bool reset,
    CUtexObject vector_input, CUsurfObject vector_output,
    CUtexObject scalar_input, CUsurfObject depth_output) {
  return DynamicSourceParameters{
      vector_input,
      {0, 0},
      {resolution.input_width, resolution.input_height},
      {std::bit_cast<float>(resolution.inverse_x_bits),
       std::bit_cast<float>(resolution.inverse_y_bits)},
      vector_output,
      scalar_input,
      {0, 0},
      {resolution.input_width, resolution.input_height},
      {std::bit_cast<float>(resolution.inverse_x_bits),
       std::bit_cast<float>(resolution.inverse_y_bits)},
      40.0f,
      1.0f,
      600.0f,
      0,
      {static_cast<std::uint8_t>(reset ? 127u : 0u), 0, 0},
      depth_output,
      {0, 0, 0, 0, 0, 0},
      {resolution.output_width, resolution.output_height},
      {std::bit_cast<float>(resolution.output_scale_x_bits),
       std::bit_cast<float>(resolution.output_scale_y_bits)}};
}

std::vector<std::uint8_t> DynamicBytes(
    const DynamicSourceParameters& parameters) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&parameters);
  return std::vector<std::uint8_t>(begin, begin + sizeof(parameters));
}

void DynamicWriteReceipt(
    const std::filesystem::path& path, const std::string& run_id,
    const dynamic_contract::Resolution& resolution, bool reset,
    std::uint8_t poison, bool artifacts_written,
    const std::vector<std::uint8_t>& vector_before,
    const std::vector<std::uint8_t>& vector_after,
    const std::vector<std::uint8_t>& scalar_before,
    const std::vector<std::uint8_t>& scalar_after,
    const std::vector<std::uint8_t>& vector_output,
    const std::vector<std::uint8_t>& depth_output,
    const std::string& normalized_parameter_sha,
    std::size_t input_mismatches, std::size_t vector_canary_mismatches,
    std::size_t depth_canary_mismatches, std::size_t vector_untouched,
    std::size_t depth_untouched) {
  const bool pass = input_mismatches == 0 && vector_canary_mismatches == 0 &&
                    depth_canary_mismatches == 0 && vector_untouched == 0 &&
                    depth_untouched == 0;
  std::ofstream receipt(path, std::ios::binary | std::ios::trunc);
  if (!receipt) throw std::runtime_error("unable to write dynamic module15 receipt");
  receipt << "{\n"
          << "  \"schema\": \"neuralstudio.fg.dynamic-module15-run.v1\",\n"
          << "  \"backend\": \"rtx-cuda-driver-typed-arrays\",\n"
          << "  \"run_id\": \"" << run_id << "\",\n"
          << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
          << "  \"resolution\": \"" << resolution.name << "\",\n"
          << "  \"phase\": \"" << (reset ? "reset" : "generated") << "\",\n"
          << "  \"large_trace_launch\": " << (reset ? 4 : 16) << ",\n"
          << "  \"canonical_launch\": " << (reset ? 4 : 14) << ",\n"
          << "  \"poison_byte\": " << unsigned(poison) << ",\n"
          << "  \"module_token\": 15,\n"
          << "  \"module_sha256\": \"" << dynamic_contract::kModuleSha256
          << "\",\n"
          << "  \"packed_parameter_bytes\": 160,\n"
          << "  \"selected_mode_byte_84\": 0,\n"
          << "  \"reset_marker_byte_85\": " << (reset ? 127 : 0) << ",\n"
          << "  \"input_extent\": [" << resolution.input_width << ", "
          << resolution.input_height << "],\n"
          << "  \"output_extent\": [" << resolution.output_width << ", "
          << resolution.output_height << "],\n"
          << "  \"grid\": [" << resolution.grid_x << ", "
          << resolution.grid_y << ", 1],\n"
          << "  \"block\": [16, 16, 1],\n"
          << "  \"trace_raw_parameter_sha256\": \""
          << (reset ? resolution.reset_raw_sha256
                    : resolution.generated_raw_sha256) << "\",\n"
          << "  \"trace_address_free_parameter_sha256\": \""
          << (reset ? resolution.reset_address_free_sha256
                    : resolution.generated_address_free_sha256) << "\",\n"
          << "  \"fixture_address_free_parameter_sha256\": \""
          << normalized_parameter_sha << "\",\n"
          << "  \"input_generator\": \"" << dynamic_contract::kGenerator
          << "\",\n"
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
          << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
          << "  \"vector_canary_mismatched_bytes\": "
          << vector_canary_mismatches << ",\n"
          << "  \"depth_canary_mismatched_bytes\": "
          << depth_canary_mismatches << ",\n"
          << "  \"vector_untouched_pixels\": " << vector_untouched << ",\n"
          << "  \"depth_untouched_pixels\": " << depth_untouched << ",\n"
          << "  \"artifacts_written\": "
          << (artifacts_written ? "true" : "false") << ",\n"
          << "  \"typed_resource_backend\": true,\n"
          << "  \"expected_or_captured_output_ingress\": false,\n"
          << "  \"archived_intermediate_ingress\": false,\n"
          << "  \"address_fields_recorded\": false,\n"
          << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n"
          << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage=fg-dynamic-module15-rtx <module> <output-dir> "
                 "<run-id> <poison> <resolution> <reset|generated> "
                 "<full|hash-only>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != dynamic_contract::kModuleBytes ||
        Sha256(module_bytes) != dynamic_contract::kModuleSha256)
      throw std::runtime_error("selected module15 identity mismatch");
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    char* poison_end = nullptr;
    const unsigned long poison_value = std::strtoul(argv[4], &poison_end, 16);
    if (poison_end == argv[4] || *poison_end != '\0' || poison_value > 255u)
      throw std::runtime_error("invalid poison");
    const auto poison = static_cast<std::uint8_t>(poison_value);
    const auto& resolution = dynamic_contract::ParseResolution(argv[5]);
    const std::string phase = argv[6];
    if (phase != "reset" && phase != "generated")
      throw std::runtime_error("invalid phase");
    const bool reset = phase == "reset";
    const std::string artifact_mode = argv[7];
    if (artifact_mode != "full" && artifact_mode != "hash-only")
      throw std::runtime_error("invalid artifact mode");
    const bool artifacts_written = artifact_mode == "full";
    std::filesystem::create_directories(output_root);
    const auto vector_before = dynamic_contract::VectorInput(resolution, !reset);
    const auto scalar_before = dynamic_contract::ScalarInput(resolution, !reset);
    const std::size_t vector_row_bytes =
        std::size_t{resolution.output_width} * 8u;
    const std::size_t depth_row_bytes =
        std::size_t{resolution.output_width} * 4u;
    const std::vector<std::uint8_t> vector_initial(
        vector_row_bytes * (resolution.output_height + 1u), poison);
    const std::vector<std::uint8_t> depth_initial(
        depth_row_bytes * (resolution.output_height + 1u), poison);

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
    ArrayResource vector_input, scalar_input, vector_output, depth_output;
    vector_input.Allocate(resolution.input_width, resolution.input_height,
                          CU_AD_FORMAT_HALF, 2, 2);
    scalar_input.Allocate(resolution.input_width, resolution.input_height,
                          CU_AD_FORMAT_FLOAT, 1, 4);
    vector_output.Allocate(resolution.output_width,
                           resolution.output_height + 1u,
                           CU_AD_FORMAT_FLOAT, 2, 4);
    depth_output.Allocate(resolution.output_width,
                          resolution.output_height + 1u,
                          CU_AD_FORMAT_UNSIGNED_INT32, 1, 4);
    vector_input.CopyFromHost(vector_before);
    scalar_input.CopyFromHost(scalar_before);
    vector_output.CopyFromHost(vector_initial);
    depth_output.CopyFromHost(depth_initial);
    vector_input.MakeTexture(CU_TR_ADDRESS_MODE_CLAMP);
    scalar_input.MakeTexture(CU_TR_ADDRESS_MODE_BORDER);
    vector_output.MakeSurface();
    depth_output.MakeSurface();
    auto parameters = DynamicMakeParameters(
        resolution, reset, vector_input.texture, vector_output.surface,
        scalar_input.texture, depth_output.surface);
    auto address_free = parameters;
    address_free.vector_input = 0;
    address_free.vector_output = 0;
    address_free.scalar_input = 0;
    address_free.depth_output = 0;
    const std::string normalized_parameter_sha =
        Sha256(DynamicBytes(address_free));
    const char* expected_normalized = reset
        ? resolution.reset_address_free_sha256
        : resolution.generated_address_free_sha256;
    if (normalized_parameter_sha != expected_normalized)
      throw std::runtime_error("address-free parameter mismatch");
    std::size_t parameter_bytes = sizeof(parameters);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &parameters,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &parameter_bytes,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, resolution.grid_x, resolution.grid_y, 1,
                         16, 16, 1, 0, nullptr, nullptr, extra),
          "launch selected module15");
    Check(cuCtxSynchronize(), "synchronize module15");
    const auto vector_after = vector_input.CopyToHost();
    const auto scalar_after = scalar_input.CopyToHost();
    const auto vector_guarded = vector_output.CopyToHost();
    const auto depth_guarded = depth_output.CopyToHost();
    const auto vector_payload = SurfacePayload(
        vector_guarded, dynamic_contract::VectorOutputBytes(resolution));
    const auto depth_payload = SurfacePayload(
        depth_guarded, dynamic_contract::DepthOutputBytes(resolution));
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
    if (artifacts_written) {
      Write(output_root / "vector-output.bin", vector_payload);
      Write(output_root / "depth-output.bin", depth_payload);
    }
    DynamicWriteReceipt(
        output_root / "run-receipt.json", run_id, resolution, reset, poison,
        artifacts_written, vector_before, vector_after, scalar_before,
        scalar_after, vector_payload, depth_payload, normalized_parameter_sha,
        input_mismatches, vector_canary_mismatches,
        depth_canary_mismatches, vector_untouched, depth_untouched);
    const bool pass = input_mismatches == 0 &&
                      vector_canary_mismatches == 0 &&
                      depth_canary_mismatches == 0 &&
                      vector_untouched == 0 && depth_untouched == 0;
    std::cout << "probe=fg_dynamic_module15_rtx\n"
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
