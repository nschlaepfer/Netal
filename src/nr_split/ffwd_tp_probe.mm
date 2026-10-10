// Harness: split ffwd launch (flags 0 or IN_VIEW) accepted kernel vs three-product kernel; byte compare with the strict reference and each other; coverage; median GPU ms.
// usage: ffwd_tp_probe SPLIT_LIB TP_LIB X W EXPECT OUT_DIR VIEW(0|1) [reps]
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
struct SplitParams { int32_t shift_x, shift_y; uint32_t flags, tokens; };
int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 8) throw std::runtime_error("usage: ffwd_tp_probe SPLIT_LIB TP_LIB X W EXPECT OUT_DIR VIEW [reps]");
  const fs::path out = argv[6]; if (fs::exists(out)) throw std::runtime_error("preserve previous run"); fs::create_directories(out); const bool view = std::atoi(argv[7]) != 0; const int reps = argc > 8 ? std::atoi(argv[8]) : 5;
  NSError* err = nil; id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  auto lib = [&](const char* p) { id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p]] error:&err]; if (!l) throw std::runtime_error(err.localizedDescription.UTF8String); return l; };
  auto pso = [&](id<MTLLibrary> l, const char* n) { id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:n]] error:&err]; if (!p) throw std::runtime_error(std::string(n) + ": " + err.localizedDescription.UTF8String); return p; };
  id<MTLLibrary> slib = lib(argv[1]), tlib = lib(argv[2]); auto X = Read(argv[3]), Wb = Read(argv[4]), EXP = Read(argv[5]); const size_t TIN = 2160u * 512u; if (EXP.size() != TIN) throw std::runtime_error("expected size");
  auto up = [&](const std::vector<uint8_t>& v) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + 16); std::memcpy(b.contents, v.data(), v.size()); return b; };
  auto zero = [&](size_t n) { id<MTLBuffer> b = [dev newBufferWithLength:n + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, n + 16); return b; };
  id<MTLBuffer> bx = up(X), bw = up(Wb); SplitParams prm{0, 0, view ? 8u : 0u, 2160}; id<MTLBuffer> bprm = [dev newBufferWithLength:sizeof(prm) options:MTLResourceStorageModeShared]; std::memcpy(bprm.contents, &prm, sizeof(prm));
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  id<MTLComputePipelineState> pb = pso(slib, "nr_split_ffwd"); id<MTLBuffer> bout0 = zero(TIN), bcov0 = zero(TIN * 4); std::vector<double> tb;
  for (int i = 0; i < reps; ++i) { std::memset(bout0.contents, 0x5a, TIN); std::memset(bcov0.contents, 0, TIN * 4 + 16);
    id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pb]; [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:bout0 offset:0 atIndex:2]; [e setBuffer:bcov0 offset:0 atIndex:3]; [e setBuffer:bprm offset:0 atIndex:4];
    [e dispatchThreads:MTLSizeMake(2160 * 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("baseline failed"); tb.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
  id<MTLComputePipelineState> pp1 = pso(tlib, "nr_split_tp_prep"), pp2 = pso(tlib, "nr_split_ffwd_tp_prep"), pt = pso(tlib, "nr_split_ffwd_tp");
  id<MTLBuffer> w1[3], w2[3], w3[3]; for (int m = 0; m < 3; ++m) { w1[m] = zero(16u * 32u * 512u * 2u); w2[m] = zero(8u * 2u * 32u * 256u * 2u); w3[m] = zero(8u * 8u * 32u * 64u * 2u); }
  id<MTLBuffer> w1max = zero(16u * 512u), w2max = zero(8u * 2u * 256u), w3max = zero(8u * 8u * 64u), bout1 = zero(TIN), bcov1 = zero(TIN * 4), bstats = zero(64); std::vector<double> tp, tprep;
  for (int i = 0; i < reps; ++i) { std::memset(bout1.contents, 0x5a, TIN); std::memset(bcov1.contents, 0, TIN * 4 + 16); std::memset(bstats.contents, 0, 64);
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pp1]; [e setBuffer:bw offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [e setBuffer:w1[m] offset:0 atIndex:1 + m]; [e setBuffer:w1max offset:0 atIndex:4];
      [e dispatchThreads:MTLSizeMake(16 * 512, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; e = [cmd computeCommandEncoder]; [e setComputePipelineState:pp2]; [e setBuffer:bw offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [e setBuffer:w2[m] offset:0 atIndex:1 + m]; [e setBuffer:w2max offset:0 atIndex:4]; for (int m = 0; m < 3; ++m) [e setBuffer:w3[m] offset:0 atIndex:5 + m]; [e setBuffer:w3max offset:0 atIndex:8];
      [e dispatchThreads:MTLSizeMake(8 * 2 * 256 + 8 * 8 * 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("prep failed"); tprep.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pt]; [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:bout1 offset:0 atIndex:2]; [e setBuffer:bcov1 offset:0 atIndex:3]; [e setBuffer:bprm offset:0 atIndex:4];
      for (int m = 0; m < 3; ++m) [e setBuffer:w1[m] offset:0 atIndex:5 + m]; [e setBuffer:w1max offset:0 atIndex:8]; for (int m = 0; m < 3; ++m) [e setBuffer:w2[m] offset:0 atIndex:9 + m]; [e setBuffer:w2max offset:0 atIndex:12]; for (int m = 0; m < 3; ++m) [e setBuffer:w3[m] offset:0 atIndex:13 + m]; [e setBuffer:w3max offset:0 atIndex:16]; [e setBuffer:bstats offset:0 atIndex:17];
      [e dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "tp failed"); tp.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); } }
  const uint8_t* o0 = static_cast<const uint8_t*>(bout0.contents); const uint8_t* o1 = static_cast<const uint8_t*>(bout1.contents); size_t d0 = 0, d1 = 0, d01 = 0; for (size_t i = 0; i < TIN; ++i) { d0 += o0[i] != EXP[i]; d1 += o1[i] != EXP[i]; d01 += o0[i] != o1[i]; }
  auto once = [&](id<MTLBuffer> b) { const uint32_t* c = static_cast<const uint32_t*>(b.contents); size_t n = 0; for (size_t i = 0; i < TIN; ++i) n += c[i] == 1; return n; };
  const uint32_t repaired = static_cast<const uint32_t*>(bstats.contents)[0]; std::ofstream(out / "out_tp.bin", std::ios::binary).write(reinterpret_cast<const char*>(o1), TIN);
  std::ostringstream js; js << "{\n \"stage\": \"ffwd\",\n \"view\": " << (view ? "true" : "false") << ",\n \"reps\": " << reps << ",\n \"baseline_gpu_ms_median\": " << median(tb) << ",\n \"tp_gpu_ms_median\": " << median(tp) << ",\n \"tp_prep_gpu_ms_median\": " << median(tprep) << ",\n \"baseline_bytes_differing_from_reference\": " << d0 << ",\n \"tp_bytes_differing_from_reference\": " << d1 << ",\n \"bytes_differing_baseline_vs_tp\": " << d01 << ",\n \"baseline_coverage_once\": " << once(bcov0) << ",\n \"tp_coverage_once\": " << once(bcov1) << ",\n \"bytes\": " << TIN << ",\n \"tp_repaired_element_steps\": " << repaired << "\n}\n";
  std::ofstream(out / "tp-run.json") << js.str(); std::cout << js.str(); return (d1 == 0 && d0 == 0) ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "ffwd_tp_probe: " << e.what() << "\n"; return 2; } } }
