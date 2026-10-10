// Host for fast_exp2_dump.metal: writes fast-exp2-unit.u32 (2^24+1 words) and fast-exp2-frac.u32 (2^23+1 words) into a directory.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 3) { std::cerr << "usage: fast_exp2_dump LIB OUT_DIR\n"; return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err]; if (!lib) { std::cerr << err.localizedDescription.UTF8String << "\n"; return 2; }
  id<MTLCommandQueue> q = [dev newCommandQueue];
  struct Job { const char* fn; const char* file; size_t n; } jobs[] = {{"fast_exp2_unit", "fast-exp2-unit.u32", 16777217u}, {"fast_exp2_frac", "fast-exp2-frac.u32", 8388609u}};
  for (const auto& j : jobs) {
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:j.fn]] error:&err]; if (!ps) { std::cerr << err.localizedDescription.UTF8String << "\n"; return 2; }
    id<MTLBuffer> b = [dev newBufferWithLength:j.n * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps]; [enc setBuffer:b offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(j.n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) { std::cerr << "command buffer failed\n"; return 2; }
    std::ofstream(std::string(argv[2]) + "/" + j.file, std::ios::binary).write(static_cast<const char*>(b.contents), std::streamsize(j.n * 4));
    std::cout << j.file << " " << j.n << " words\n";
  }
  return 0; } }
