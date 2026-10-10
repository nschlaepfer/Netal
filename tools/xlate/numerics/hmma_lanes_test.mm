#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int, const char** argv){ @autoreleasepool { id<MTLDevice> d=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLLibrary> L=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[L newFunctionWithName:@"hmma_lanes_test"] error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);return 1;}
  const uint32_t V=1u<<20; id<MTLBuffer> b=[d newBufferWithLength:V*4 options:MTLResourceStorageModeShared]; id<MTLCommandQueue> q=[d newCommandQueue]; unsigned long long bad=0,badold=0,total=0;
  for(uint32_t seed=1; seed<=8; ++seed){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:p]; [en setBuffer:b offset:0 atIndex:0]; [en setBytes:&seed length:4 atIndex:1]; [en dispatchThreads:MTLSizeMake(V*16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const uint32_t* o=(const uint32_t*)b.contents; for(uint32_t i=0;i<V;++i){ bad+=o[i]&1u; badold+=(o[i]>>1)&1u; } total+=V; }
  printf("%llu vectors: fixed lanes helper mismatches vs oracle %llu | OLD lanes helper mismatches %llu\n", total, bad, badold); return bad?3:0; } }
