// Harness: split qkv GEMM, accepted kernel vs three-product kernel on a live input; the chain records only the attention output, so this compares
// q/k/v bytewise against the accepted kernel (bit-exact by the chain evidence). usage: sqkv_tp_probe SPLIT_LIB TP_LIB X W RSQRT OUT_DIR [reps]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace fs = std::filesystem;
static std::vector<uint8_t> Read(const fs::path& p) { std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p.string()); const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v; }
struct SplitParams { int32_t shift_x, shift_y; uint32_t flags, tokens; }; struct SpPrepParams { uint32_t steps, cols, step_bytes, reserved; };
int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 7) throw std::runtime_error("usage: sqkv_tp_probe SPLIT_LIB TP_LIB X W RSQRT OUT_DIR [reps]");
  const fs::path out = argv[6]; if (fs::exists(out)) throw std::runtime_error("preserve previous run"); fs::create_directories(out); const int reps = argc > 7 ? std::atoi(argv[7]) : 5;
  NSError* err = nil; id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  auto lib = [&](const char* p) { id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p]] error:&err]; if (!l) throw std::runtime_error(err.localizedDescription.UTF8String); return l; };
  auto pso = [&](id<MTLLibrary> l, const char* n) { id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:n]] error:&err]; if (!p) throw std::runtime_error(std::string(n) + ": " + err.localizedDescription.UTF8String); return p; };
  id<MTLLibrary> slib = lib(argv[1]), tlib = lib(argv[2]); auto X = Read(argv[3]), Wb = Read(argv[4]), RSQ = Read(argv[5]); const size_t QB = 2160u * 16u * 32u;
  auto up = [&](const std::vector<uint8_t>& v) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + 16); std::memcpy(b.contents, v.data(), v.size()); return b; };
  auto zero = [&](size_t n) { id<MTLBuffer> b = [dev newBufferWithLength:n + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, n + 16); return b; };
  id<MTLBuffer> bx = up(X), bw = up(Wb), brsq = up(RSQ); SplitParams prm{0, 0, 0, 2160}; id<MTLBuffer> bprm = [dev newBufferWithLength:sizeof(prm) options:MTLResourceStorageModeShared]; std::memcpy(bprm.contents, &prm, sizeof(prm));
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  id<MTLComputePipelineState> pb = pso(slib, "nr_split_qkv"); id<MTLBuffer> q0 = zero(QB), k0 = zero(QB), v0 = zero(QB); std::vector<double> tb;
  for (int i = 0; i < reps; ++i) { for (id<MTLBuffer> b : {q0, k0, v0}) std::memset(b.contents, 0x5a, QB);
    id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pb]; [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:q0 offset:0 atIndex:2]; [e setBuffer:k0 offset:0 atIndex:3]; [e setBuffer:v0 offset:0 atIndex:4]; [e setBuffer:brsq offset:0 atIndex:5]; [e setBuffer:bprm offset:0 atIndex:6];
    [e dispatchThreads:MTLSizeMake(2160 * 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("baseline failed"); tb.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
  id<MTLComputePipelineState> pp = pso(tlib, "nr_split_tp_prep_generic"), pt = pso(tlib, "nr_split_qkv_tp"); id<MTLBuffer> wl = zero(16u * 32u * 1536u * 2u), wh = zero(16u * 32u * 1536u * 2u), wlh = zero(16u * 32u * 1536u * 2u), wmax = zero(16u * 1536u), q1 = zero(QB), k1 = zero(QB), v1 = zero(QB), bstats = zero(64);
  SpPrepParams ppp{16u, 1536u, 49152u, 0u}; id<MTLBuffer> bppp = [dev newBufferWithLength:sizeof(ppp) options:MTLResourceStorageModeShared]; std::memcpy(bppp.contents, &ppp, sizeof(ppp)); std::vector<double> tp, tprep;
  for (int i = 0; i < reps; ++i) { for (id<MTLBuffer> b : {q1, k1, v1}) std::memset(b.contents, 0x5a, QB); std::memset(bstats.contents, 0, 64);
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pp]; [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:wl offset:0 atIndex:1]; [e setBuffer:wh offset:0 atIndex:2]; [e setBuffer:wlh offset:0 atIndex:3]; [e setBuffer:wmax offset:0 atIndex:4]; [e setBuffer:bppp offset:0 atIndex:5];
      [e dispatchThreads:MTLSizeMake(16 * 1536, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("prep failed"); tprep.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pt]; [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:q1 offset:0 atIndex:2]; [e setBuffer:k1 offset:0 atIndex:3]; [e setBuffer:v1 offset:0 atIndex:4]; [e setBuffer:brsq offset:0 atIndex:5]; [e setBuffer:bprm offset:0 atIndex:6];
      [e setBuffer:wl offset:0 atIndex:7]; [e setBuffer:wh offset:0 atIndex:8]; [e setBuffer:wlh offset:0 atIndex:9]; [e setBuffer:wmax offset:0 atIndex:10]; [e setBuffer:bstats offset:0 atIndex:11];
      [e dispatchThreadgroups:MTLSizeMake(16, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "tp failed"); tp.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); } }
  size_t dq = 0, dk = 0, dv = 0; for (size_t i = 0; i < QB; ++i) { dq += static_cast<const uint8_t*>(q0.contents)[i] != static_cast<const uint8_t*>(q1.contents)[i]; dk += static_cast<const uint8_t*>(k0.contents)[i] != static_cast<const uint8_t*>(k1.contents)[i]; dv += static_cast<const uint8_t*>(v0.contents)[i] != static_cast<const uint8_t*>(v1.contents)[i]; }
  const uint32_t repaired = static_cast<const uint32_t*>(bstats.contents)[0];
  std::ostringstream js; js << "{\n \"stage\": \"split_qkv\",\n \"reps\": " << reps << ",\n \"baseline_gpu_ms_median\": " << median(tb) << ",\n \"tp_gpu_ms_median\": " << median(tp) << ",\n \"tp_prep_gpu_ms_median\": " << median(tprep) << ",\n \"bytes_differing_baseline_vs_tp\": {\"q\": " << dq << ", \"k\": " << dk << ", \"v\": " << dv << "},\n \"bytes_per_output\": " << QB << ",\n \"tp_repaired_element_steps\": " << repaired << "\n}\n";
  std::ofstream(out / "tp-run.json") << js.str(); std::cout << js.str(); return (dq + dk + dv) == 0 ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "sqkv_tp_probe: " << e.what() << "\n"; return 2; } } }
