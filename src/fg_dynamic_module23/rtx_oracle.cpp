#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"
using uint = std::uint32_t;
#include "contract.h"
namespace {
namespace c = fg_dynamic_module23;
std::uint8_t ParseDynamicPoison(const char *text) {
  char *end = nullptr;
  const auto value = std::strtoul(text, &end, 16);
  if (end == text || *end != '\0' || value > 255u)
    throw std::runtime_error("poison");
  return static_cast<std::uint8_t>(value);
}
std::vector<std::uint8_t> G(const std::vector<std::uint8_t> &b, std::uint8_t) {
  std::vector<std::uint8_t> x(b.size() + 2 * kGuardBytes, kGuardValue);
  memcpy(x.data() + kGuardBytes, b.data(), b.size());
  return x;
}
template <std::size_t N>
void P32(std::array<std::uint8_t, N> &b, std::size_t o, uint v) {
  memcpy(b.data() + o, &v, 4);
}
template <std::size_t N>
void P64(std::array<std::uint8_t, N> &b, std::size_t o, std::uint64_t v) {
  memcpy(b.data() + o, &v, 8);
}
std::vector<std::uint8_t> Pay(const std::vector<std::uint8_t> &b,
                              std::size_t n) {
  return {b.begin() + kGuardBytes, b.begin() + kGuardBytes + n};
}
} // namespace
int main(int n, char **v) {
  if (n != 8)
    return 2;
  try {
    auto mb = Read(v[1]);
    if (Sha256(mb) != c::kModuleSha)
      throw std::runtime_error("module");
    std::filesystem::path d = v[2];
    std::filesystem::create_directories(d);
    auto z = ParseDynamicPoison(v[4]);
    auto &r = c::Find(v[5]);
    bool gen = std::string(v[6]) == "generated",
         art = std::string(v[7]) == "full";
    auto before = c::Levels(r, gen);
    auto mask0 = c::Mask(r, gen);
    Check(cuInit(0), "init");
    CUdevice dev;
    Check(cuDeviceGet(&dev, 0), "dev");
    CUcontext ctx;
    Check(cuDevicePrimaryCtxRetain(&ctx, dev), "ctx");
    Check(cuCtxSetCurrent(ctx), "set");
    CUmodule mod;
    Check(cuModuleLoadData(&mod, mb.data()), "module");
    CUfunction fn;
    Check(cuModuleGetFunction(&fn, mod, "main_kernel"), "fn");
    std::array<GuardedDeviceBuffer, 4> levels;
    for (int i = 0; i < 4; ++i) {
      levels[i].Allocate(before[i].size());
      auto g = G(before[i], z);
      Check(cuMemcpyHtoD(levels[i].base, g.data(), g.size()), "up");
    }
    GuardedDeviceBuffer mask;
    mask.Allocate(mask0.size());
    auto mg = G(mask0, z);
    Check(cuMemcpyHtoD(mask.base, mg.data(), mg.size()), "mask");
    std::array<std::uint8_t, 72> p{};
    for (int i = 0; i < 4; ++i)
      P64(p, i * 8, levels[i].payload());
    P64(p, 32, mask.payload());
    P32(p, 40, r.w);
    P32(p, 44, r.h);
    P32(p, 48, 0);
    std::size_t sz = p.size();
    void *ex[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, p.data(),
                  CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(fn, r.gx, r.gy, 1, 256, 1, 1, 0, nullptr, nullptr, ex),
          "launch");
    Check(cuCtxSynchronize(), "sync");
    std::array<std::vector<std::uint8_t>, 4> after;
    std::size_t can = 0, immut = 0;
    for (int i = 0; i < 4; ++i) {
      auto x = levels[i].ReadAll();
      after[i] = Pay(x, before[i].size());
      can += CanaryMismatches(x, before[i].size());
      if (i)
        immut += Mismatches(before[i], after[i]);
    }
    auto ma = mask.ReadAll();
    auto mask1 = Pay(ma, mask0.size());
    can += CanaryMismatches(ma, mask0.size());
    immut += Mismatches(mask0, mask1);
    if (art)
      Write(d / "level0-output.bin", after[0]);
    std::ofstream s(d / "run-receipt.json");
    s << "{\n  \"schema\": \"neuralstudio.fg.dynamic-module23-run.v1\",\n  "
         "\"backend\": \"rtx-cuda-driver-raw-buffers\",\n  \"run_id\": \""
      << v[3] << "\",\n  \"process_id\": " << GetCurrentProcessId()
      << ",\n  \"resolution\": \"" << r.name << "\",\n  \"phase\": \""
      << (gen ? "generated" : "reset")
      << "\",\n  \"poison_byte\": " << unsigned(z)
      << ",\n  \"module_sha256\": \"" << c::kModuleSha
      << "\",\n  \"selected_mode\": 0,\n  \"extent\": [" << r.w << ", " << r.h
      << "],\n  \"grid\": [" << r.gx << ", " << r.gy
      << ", 1],\n  \"trace_raw_parameter_sha256\": \"" << r.raw
      << "\",\n  \"level0_pre_sha256\": \"" << Sha256(before[0])
      << "\",\n  \"level0_post_sha256\": \"" << Sha256(after[0])
      << "\",\n  \"level0_bytes\": " << after[0].size()
      << ",\n  \"readonly_input_mismatched_bytes\": " << immut
      << ",\n  \"canary_mismatched_bytes\": " << can
      << ",\n  \"artifacts_written\": " << (art ? "true" : "false")
      << ",\n  \"expected_or_captured_output_ingress\": false,\n  "
         "\"address_fields_recorded\": false,\n  \"result\": \""
      << ((!immut && !can) ? "pass" : "fail") << "\"\n}\n";
    return immut || can ? 3 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
