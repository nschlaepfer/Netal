// Harness: split conv launch (flags 0) with the accepted kernel (baseline split.metallib) and the three-product prototype; byte compare with the strict
// reference output and with each other; exactly-once coverage; median GPU ms over repetitions; repair counts. Comparisons on the CPU afterwards.
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
  if (argc < 8) throw std::runtime_error("usage: tp_probe SPLIT_LIB TP_LIB X RES W EXPECT OUT_DIR [reps]");
  const fs::path out = argv[7]; if (fs::exists(out)) throw std::runtime_error("preserve previous run"); fs::create_directories(out); const int reps = argc > 8 ? std::atoi(argv[8]) : 5;
  NSError* err = nil; id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  auto lib = [&](const char* p) { id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p]] error:&err]; if (!l) throw std::runtime_error(err.localizedDescription.UTF8String); return l; };
  auto pso = [&](id<MTLLibrary> l, const char* n) { id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:n]] error:&err]; if (!p) throw std::runtime_error(std::string(n) + ": " + err.localizedDescription.UTF8String); return p; };
  id<MTLLibrary> slib = lib(argv[1]), tlib = lib(argv[2]);
  auto X = Read(argv[3]), RES = Read(argv[4]), Wb = Read(argv[5]), EXP = Read(argv[6]); const size_t TIN = 2160u * 512u; if (X.size() != TIN || RES.size() != TIN || EXP.size() != TIN) throw std::runtime_error("operand sizes");
  auto up = [&](const std::vector<uint8_t>& v, size_t extra = 16) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + extra options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + extra); std::memcpy(b.contents, v.data(), v.size()); return b; };
  auto zero = [&](size_t n) { id<MTLBuffer> b = [dev newBufferWithLength:n + 16 options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, n + 16); return b; };
  id<MTLBuffer> bx = up(X), bres = up(RES), bw = up(Wb);
  // layout probe: which (row, col) does each lane's thread_elements hold?
  { std::vector<uint16_t> a(64), ident(64, 0); for (int i = 0; i < 64; ++i) { __fp16 h = __fp16(i); std::memcpy(&a[i], &h, 2); } for (int i = 0; i < 8; ++i) { __fp16 h = __fp16(1); std::memcpy(&ident[i * 8 + i], &h, 2); }
    id<MTLBuffer> ba = [dev newBufferWithLength:128 options:MTLResourceStorageModeShared]; std::memcpy(ba.contents, a.data(), 128); id<MTLBuffer> bb = [dev newBufferWithLength:128 options:MTLResourceStorageModeShared]; std::memcpy(bb.contents, ident.data(), 128); id<MTLBuffer> bo = zero(256);
    id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pso(tlib, "nr_tp_layout_probe")]; [e setBuffer:ba offset:0 atIndex:0]; [e setBuffer:bb offset:0 atIndex:1]; [e setBuffer:bo offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    const float* o = static_cast<const float*>(bo.contents); bool ok = true; std::ostringstream ls;
    for (int l = 0; l < 32; ++l) for (int k = 0; k < 2; ++k) { const int v = int(o[2 * l + k]); const int r = v / 8, c = v % 8; ls << "(" << l << "," << k << ")->(" << r << "," << c << ") "; if (r != (l % 8) / 2 + 4 * (l / 16) || c != 2 * (l % 2) + 4 * ((l / 8) % 2) + k) ok = false; }
    std::ofstream(out / "layout.txt") << ls.str() << "\nassumed_mapping_ok " << (ok ? "true" : "false") << "\n"; std::cout << "layout mapping row=(lane%8)/2+4*(lane/16) col=2*(lane%2)+4*((lane/8)%2)+e: " << (ok ? "OK" : "DIFFERENT (see layout.txt)") << "\n";
    if (!ok) throw std::runtime_error("simdgroup element mapping differs from the kernel's assumption");
  }
  SplitParams prm{0, 0, 0, 2160}; id<MTLBuffer> bprm = [dev newBufferWithLength:sizeof(prm) options:MTLResourceStorageModeShared]; std::memcpy(bprm.contents, &prm, sizeof(prm));
  auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  // ---- baseline ----
  id<MTLComputePipelineState> pb = pso(slib, "nr_split_conv"); id<MTLBuffer> bout0 = zero(TIN), by16 = zero(TIN * 2), bcov0 = zero(TIN * 4); std::vector<double> tb;
  for (int i = 0; i < reps; ++i) { std::memset(bout0.contents, 0x5a, TIN); std::memset(bcov0.contents, 0, TIN * 4 + 16);
    id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pb];
    [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bres offset:0 atIndex:1]; [e setBuffer:bw offset:0 atIndex:2]; [e setBuffer:bout0 offset:0 atIndex:3]; [e setBuffer:by16 offset:0 atIndex:4]; [e setBuffer:bcov0 offset:0 atIndex:5]; [e setBuffer:bprm offset:0 atIndex:6];
    [e dispatchThreads:MTLSizeMake(2160 * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("baseline failed"); tb.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
  // ---- prototype: prep + conv ----
  id<MTLComputePipelineState> pp = pso(tlib, "nr_split_tp_prep"), pt = pso(tlib, "nr_split_conv_tp");
  const size_t WH = 16u * 32u * 512u; id<MTLBuffer> bwl = zero(WH * 2), bwh = zero(WH * 2), bwlh = zero(WH * 2), bmaxcls = zero(16u * 512u), bout1 = zero(TIN), bcov1 = zero(TIN * 4), bstats = zero(64); std::vector<double> tp, tprep;
  for (int i = 0; i < reps; ++i) { std::memset(bout1.contents, 0x5a, TIN); std::memset(bcov1.contents, 0, TIN * 4 + 16); std::memset(bstats.contents, 0, 64);
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pp]; [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bwl offset:0 atIndex:1]; [e setBuffer:bwh offset:0 atIndex:2]; [e setBuffer:bwlh offset:0 atIndex:3]; [e setBuffer:bmaxcls offset:0 atIndex:4];
      [e dispatchThreads:MTLSizeMake(16 * 512, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("prep failed"); tprep.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); }
    { id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pt];
      [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bres offset:0 atIndex:1]; [e setBuffer:bw offset:0 atIndex:2]; [e setBuffer:bout1 offset:0 atIndex:3]; [e setBuffer:by16 offset:0 atIndex:4]; [e setBuffer:bcov1 offset:0 atIndex:5]; [e setBuffer:bprm offset:0 atIndex:6]; [e setBuffer:bwl offset:0 atIndex:7]; [e setBuffer:bwh offset:0 atIndex:8]; [e setBuffer:bwlh offset:0 atIndex:9]; [e setBuffer:bmaxcls offset:0 atIndex:10]; [e setBuffer:bstats offset:0 atIndex:11];
      [e dispatchThreadgroups:MTLSizeMake(512 / 64, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "tp failed"); tp.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0); } }
  const uint8_t* o0 = static_cast<const uint8_t*>(bout0.contents); const uint8_t* o1 = static_cast<const uint8_t*>(bout1.contents); size_t d0 = 0, d1 = 0, d01 = 0; for (size_t i = 0; i < TIN; ++i) { d0 += o0[i] != EXP[i]; d1 += o1[i] != EXP[i]; d01 += o0[i] != o1[i]; }
  auto covok = [&](id<MTLBuffer> b) { const uint32_t* c = static_cast<const uint32_t*>(b.contents); size_t once = 0; for (size_t i = 0; i < TIN; ++i) once += c[i] == 1; return once; };
  const uint32_t repaired = static_cast<const uint32_t*>(bstats.contents)[0];
  std::ofstream(out / "out_baseline.bin", std::ios::binary).write(reinterpret_cast<const char*>(o0), TIN); std::ofstream(out / "out_tp.bin", std::ios::binary).write(reinterpret_cast<const char*>(o1), TIN);
  std::ostringstream js; js << "{\n \"reps\": " << reps << ",\n \"baseline_gpu_ms_median\": " << median(tb) << ",\n \"tp_gpu_ms_median\": " << median(tp) << ",\n \"tp_prep_gpu_ms_median\": " << median(tprep) << ",\n \"baseline_bytes_differing_from_reference\": " << d0 << ",\n \"tp_bytes_differing_from_reference\": " << d1 << ",\n \"bytes_differing_baseline_vs_tp\": " << d01
     << ",\n \"baseline_coverage_once\": " << covok(bcov0) << ",\n \"tp_coverage_once\": " << covok(bcov1) << ",\n \"bytes\": " << TIN << ",\n \"tp_repaired_element_steps\": " << repaired << ",\n \"element_steps\": " << 2160u * 512u * 16u << "\n}\n";
  std::ofstream(out / "tp-run.json") << js.str(); std::cout << js.str(); return (d1 == 0 && d0 == 0) ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "tp_probe: " << e.what() << "\n"; return 2; } } }
