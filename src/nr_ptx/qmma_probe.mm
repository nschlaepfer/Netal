#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 5) { std::cerr << "usage: qmma_probe LIB INPUT_AFTER.bin RTX.bin OUT.bin\n"; return 2; }
  auto rd = [](const char* p) { std::ifstream s(p, std::ios::binary | std::ios::ate); size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read((char*)v.data(), n); return v; };
  auto in = rd(argv[2]), rtx = rd(argv[3]); const size_t cases = in.size() / (32 * 8 * 4);
  NSError* err = nil; id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err]; if (!lib) { std::cerr << err.localizedDescription.UTF8String << "\n"; return 2; }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:(getenv("QMMA_KERNEL") ? getenv("QMMA_KERNEL") : "qmma_probe")]] error:&err]; if (!pso) { std::cerr << err.localizedDescription.UTF8String << "\n"; return 2; }
  id<MTLBuffer> bi = [dev newBufferWithLength:in.size() options:MTLResourceStorageModeShared]; std::memcpy(bi.contents, in.data(), in.size()); id<MTLBuffer> bo = [dev newBufferWithLength:cases * 32 * 8 options:MTLResourceStorageModeShared]; std::memset(bo.contents, 0xee, cases * 32 * 8);
  id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; [e setComputePipelineState:pso]; [e setBuffer:bi offset:0 atIndex:0]; [e setBuffer:bo offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(cases, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) { std::cerr << "command buffer failed\n"; return 2; }
  const uint32_t* o = (const uint32_t*)bo.contents; const uint32_t* r = (const uint32_t*)rtx.data(); size_t mism = 0; int shown = 0;
  for (size_t i = 0; i < cases * 32 * 2; ++i) if (o[i] != r[i]) { ++mism; if (shown++ < 6) std::cout << "case " << i / 64 << " lane " << (i / 2) % 32 << " reg " << i % 2 << " gpu " << std::hex << o[i] << " rtx " << r[i] << std::dec << "\n"; }
  std::ofstream(argv[4], std::ios::binary).write((const char*)o, cases * 32 * 8);
  std::cout << "registers " << cases * 64 << " mismatching " << mism << "\n"; return mism ? 1 : 0;
} }
