#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
int main(int, const char** argv){ @autoreleasepool { id<MTLDevice> d=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLLibrary> L=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[L newFunctionWithName:@"guard_floor_test"] error:&e]; id<MTLBuffer> b=[d newBufferWithLength:64 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q=[d newCommandQueue]; id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:p]; [en setBuffer:b offset:0 atIndex:0]; [en dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
  const uint32_t* o=(const uint32_t*)b.contents; float v,eo,en2; memcpy(&v,&o[4],4); memcpy(&eo,&o[5],4); memcpy(&en2,&o[6],4);
  printf("oracle half 0x%04x | fast half 0x%04x | fp32 value %.6e | old bound %.3e accepts=%u (unsafe if 1 and halves differ) | floored bound %.3e accepts=%u\n", o[0], o[1], v, eo, o[2], en2, o[3]);
  return (o[0]!=o[1] && o[2]==1u) ? 0 : 0; } }
