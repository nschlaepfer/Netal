#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"
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
std::vector<std::uint8_t> RF(const std::filesystem::path &p) {
  std::ifstream s(p, std::ios::binary | std::ios::ate);
  auto n = s.tellg();
  std::vector<std::uint8_t> b((std::size_t)n);
  s.seekg(0);
  s.read((char *)b.data(), n);
  return b;
}
template <std::size_t N>
void P32(std::array<std::uint8_t, N> &b, std::size_t o, uint v) {
  memcpy(b.data() + o, &v, 4);
}
template <std::size_t N>
void P64(std::array<std::uint8_t, N> &b, std::size_t o, std::uint64_t v) {
  memcpy(b.data() + o, &v, 8);
}
} // namespace
int main(int n, const char *v[]) {
  @autoreleasepool {
    if (n != 9)
      return 2;
    try {
      auto rom = RF(v[2]);
      if (Sha256(rom) !=
          "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750")
        throw std::runtime_error("rom");
      auto z = ParseDynamicPoison(v[5]);
      auto &r = c::Find(v[6]);
      bool gen = std::string(v[7]) == "generated",
           art = std::string(v[8]) == "full";
      std::filesystem::path d = v[3];
      std::filesystem::create_directories(d);
      auto before = c::Levels(r, gen);
      auto m0 = c::Mask(r, gen);
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      NSError *er = nil;
      auto lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(v[1])]
                                  error:&er];
      auto fn = [lib newFunctionWithName:@"fg_dynamic_module23_mode0"];
      auto pipe = [dev newComputePipelineStateWithFunction:fn error:&er];
      std::array<id<MTLBuffer>, 4> b;
      for (int i = 0; i < 4; ++i)
        b[i] = Guarded(dev, &before[i], before[i].size(), 0);
      auto mask = Guarded(dev, &m0, m0.size(), 0),
           rb = Guarded(dev, &rom, rom.size(), 0);
      std::array<std::uint8_t, 72> p{};
      for (int i = 0; i < 4; ++i)
        P64(p, i * 8, b[i].gpuAddress + kGuardBytes);
      P64(p, 32, mask.gpuAddress + kGuardBytes);
      P32(p, 40, r.w);
      P32(p, 44, r.h);
      P32(p, 48, 0);
      auto cmd = [[dev newCommandQueue] commandBuffer];
      auto enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipe];
      [enc setBytes:p.data() length:p.size() atIndex:0];
      [enc setBuffer:rb offset:kGuardBytes atIndex:1];
      for (auto x : b)
        [enc useResource:x usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      [enc useResource:mask usage:MTLResourceUsageRead];
      [enc dispatchThreadgroups:MTLSizeMake(r.gx, r.gy, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      std::array<std::vector<std::uint8_t>, 4> a;
      std::size_t can = CanaryMismatches(mask, m0.size()) +
                        CanaryMismatches(rb, rom.size()),
                  immut = Mismatches(m0, Payload(mask, m0.size())) +
                          Mismatches(rom, Payload(rb, rom.size()));
      for (int i = 0; i < 4; ++i) {
        a[i] = Payload(b[i], before[i].size());
        can += CanaryMismatches(b[i], before[i].size());
        if (i)
          immut += Mismatches(before[i], a[i]);
      }
      if (art)
        Write(d / "level0-output.bin", a[0]);
      std::ofstream s(d / "run-receipt.json");
      s << "{\n  \"schema\": \"neuralstudio.fg.dynamic-module23-run.v1\",\n  "
           "\"backend\": \"mac-metal\",\n  \"run_id\": \""
        << v[4] << "\",\n  \"process_id\": " << getpid()
        << ",\n  \"resolution\": \"" << r.name << "\",\n  \"phase\": \""
        << (gen ? "generated" : "reset")
        << "\",\n  \"poison_byte\": " << unsigned(z)
        << ",\n  \"module_sha256\": \"" << c::kModuleSha
        << "\",\n  \"selected_mode\": 0,\n  \"extent\": [" << r.w << ", " << r.h
        << "],\n  \"grid\": [" << r.gx << ", " << r.gy
        << ", 1],\n  \"trace_raw_parameter_sha256\": \"" << r.raw
        << "\",\n  \"level0_pre_sha256\": \"" << Sha256(before[0])
        << "\",\n  \"level0_post_sha256\": \"" << Sha256(a[0])
        << "\",\n  \"level0_bytes\": " << a[0].size()
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
}
