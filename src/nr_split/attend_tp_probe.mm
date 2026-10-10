// Harness: split attend accepted vs engine kernel. q/k/v are produced live by the accepted nr_split_qkv from X; the attention output is compared with the
// strict reference launch output and the accepted kernel. usage: attend_tp_probe SPLIT_LIB TP_LIB X W RSQRT RCP EXPECT SHIFT_X SHIFT_Y OUT_DIR [reps]
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
  if (argc < 11) throw std::runtime_error("usage: attend_tp_probe SPLIT_LIB TP_LIB X W RSQRT RCP EXPECT SX SY OUT_DIR [reps]");
  const fs::path out = argv[10]; if (fs::exists(out)) throw std::runtime_error("preserve previous run"); fs::create_directories(out); const int reps = argc > 11 ? std::atoi(argv[11]) : 5;
  NSError* err = nil; id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  auto lib = [&](const char* p) { id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p]] error:&err]; if (!l) throw std::runtime_error(err.localizedDescription.UTF8String); return l; };
  auto pso = [&](id<MTLLibrary> l, const char* n) { id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:n]] error:&err]; if (!p) throw std::runtime_error(std::string(n) + ": " + err.localizedDescription.UTF8String); return p; };
  id<MTLLibrary> slib = lib(argv[1]), tlib = lib(argv[2]); auto X = Read(argv[3]), Wb = Read(argv[4]), RSQ = Read(argv[5]), RCP = Read(argv[6]), EXP = Read(argv[7]); const size_t TIN = 2160u * 512u, QB = 2160u * 16u * 32u; if (EXP.size() != TIN) throw std::runtime_error("expected size");
  auto up = [&](const std::vector<uint8_t>& v) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + 16); std::memcpy(b.contents, v.data(), v.size()); return b; };
  auto zero = [&](size_t n) { id<MTLBuffer> b = [dev newBufferWithLength:n + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, n + 16); return b; };
  id<MTLBuffer> bx = up(X), bw = up(Wb), brsq = up(RSQ), brcp = up(RCP), qb = zero(QB), kb = zero(QB), vb = zero(QB); SplitParams prm{std::atoi(argv[8]), std::atoi(argv[9]), 0, 2160}; id<MTLBuffer> bprm = [dev newBufferWithLength:sizeof(prm) options:MTLResourceStorageModeShared]; std::memcpy(bprm.contents, &prm, sizeof(prm));
  { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pso(slib, "nr_split_qkv")]; [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:qb offset:0 atIndex:2]; [e setBuffer:kb offset:0 atIndex:3]; [e setBuffer:vb offset:0 atIndex:4]; [e setBuffer:brsq offset:0 atIndex:5]; [e setBuffer:bprm offset:0 atIndex:6];
    [e dispatchThreads:MTLSizeMake(2160 * 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("qkv failed"); }
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  id<MTLComputePipelineState> pb = pso(slib, "nr_split_attend"); id<MTLBuffer> bout0 = zero(TIN), bcov0 = zero(TIN * 4); std::vector<double> tb;
  for (int i = 0; i < reps; ++i) { std::memset(bout0.contents, 0x5a, TIN); std::memset(bcov0.contents, 0, TIN * 4 + 16);
    id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pb]; [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kb offset:0 atIndex:1]; [e setBuffer:vb offset:0 atIndex:2]; [e setBuffer:bw offset:0 atIndex:3]; [e setBuffer:bout0 offset:0 atIndex:4]; [e setBuffer:bcov0 offset:0 atIndex:5]; [e setBuffer:brcp offset:0 atIndex:6]; [e setBuffer:bprm offset:0 atIndex:7];
    [e dispatchThreads:MTLSizeMake(40 * 16 * 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("baseline failed"); tb.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
  id<MTLComputePipelineState> pp = pso(tlib, "nr_split_attend_tp_prep"), pt = pso(tlib, "nr_split_attend_tp"); const size_t KW = 40u * 16u * 32u * 64u; id<MTLBuffer> k3[3], v3[3]; for (int m = 0; m < 3; ++m) { k3[m] = zero(KW * 2); v3[m] = zero(KW * 2); }
  id<MTLBuffer> kmax = zero(40u * 16u * 64u), vmax = zero(40u * 16u * 2u * 32u), bout1 = zero(TIN), bcov1 = zero(TIN * 4), bstats = zero(64); std::vector<double> tp, tprep;
  for (int i = 0; i < reps; ++i) { std::memset(bout1.contents, 0x5a, TIN); std::memset(bcov1.contents, 0, TIN * 4 + 16); std::memset(bstats.contents, 0, 64);
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pp]; [e setBuffer:kb offset:0 atIndex:0]; [e setBuffer:vb offset:0 atIndex:1]; [e setBuffer:bprm offset:0 atIndex:2]; for (int m = 0; m < 3; ++m) [e setBuffer:k3[m] offset:0 atIndex:3 + m]; [e setBuffer:kmax offset:0 atIndex:6]; for (int m = 0; m < 3; ++m) [e setBuffer:v3[m] offset:0 atIndex:7 + m]; [e setBuffer:vmax offset:0 atIndex:10];
      [e dispatchThreads:MTLSizeMake(40 * 16 * 64 + 40 * 16 * 2 * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("prep failed"); tprep.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pt]; [e setBuffer:qb offset:0 atIndex:0]; [e setBuffer:kb offset:0 atIndex:1]; [e setBuffer:vb offset:0 atIndex:2]; [e setBuffer:bw offset:0 atIndex:3]; [e setBuffer:bout1 offset:0 atIndex:4]; [e setBuffer:bcov1 offset:0 atIndex:5]; [e setBuffer:brcp offset:0 atIndex:6]; [e setBuffer:bprm offset:0 atIndex:7];
      for (int m = 0; m < 3; ++m) [e setBuffer:k3[m] offset:0 atIndex:8 + m]; [e setBuffer:kmax offset:0 atIndex:11]; for (int m = 0; m < 3; ++m) [e setBuffer:v3[m] offset:0 atIndex:12 + m]; [e setBuffer:vmax offset:0 atIndex:15]; [e setBuffer:bstats offset:0 atIndex:16];
      [e dispatchThreadgroups:MTLSizeMake(16, 40 * 4, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "tp failed"); tp.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); } }
  const uint8_t* o0 = static_cast<const uint8_t*>(bout0.contents); const uint8_t* o1 = static_cast<const uint8_t*>(bout1.contents); size_t d0 = 0, d1 = 0, d01 = 0; for (size_t i = 0; i < TIN; ++i) { d0 += o0[i] != EXP[i]; d1 += o1[i] != EXP[i]; d01 += o0[i] != o1[i]; }
  auto once = [&](id<MTLBuffer> b) { const uint32_t* c = static_cast<const uint32_t*>(b.contents); size_t n = 0; for (size_t i = 0; i < TIN; ++i) n += c[i] == 1; return n; };
  const uint32_t repaired = static_cast<const uint32_t*>(bstats.contents)[0];
  std::ostringstream js; js << "{\n \"stage\": \"split_attend\",\n \"shift\": [" << prm.shift_x << ", " << prm.shift_y << "],\n \"reps\": " << reps << ",\n \"baseline_gpu_ms_median\": " << median(tb) << ",\n \"tp_gpu_ms_median\": " << median(tp) << ",\n \"tp_prep_gpu_ms_median\": " << median(tprep) << ",\n \"baseline_bytes_differing_from_reference\": " << d0 << ",\n \"tp_bytes_differing_from_reference\": " << d1 << ",\n \"bytes_differing_baseline_vs_tp\": " << d01 << ",\n \"baseline_coverage_once\": " << once(bcov0) << ",\n \"tp_coverage_once\": " << once(bcov1) << ",\n \"bytes\": " << TIN << ",\n \"tp_repaired_element_steps\": " << repaired << "\n}\n";
  std::ofstream(out / "tp-run.json") << js.str(); std::cout << js.str(); return (d1 == 0 && d0 == 0) ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "attend_tp_probe: " << e.what() << "\n"; return 2; } } }
