#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <algorithm>
int main(int argc, const char** argv) { @autoreleasepool {
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"touch"] error:&e];
    id<MTLBuffer> b=[dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ind=[dev newBufferWithLength:12 options:MTLResourceStorageModeShared]; ((uint32_t*)ind.contents)[0]=1; ((uint32_t*)ind.contents)[1]=1; ((uint32_t*)ind.contents)[2]=1;
    auto run=[&](const char* name, int N, int mode){ double best=1e9; for(int r=0;r<10;++r){ id<MTLCommandBuffer> cb=[q commandBuffer];
        if(mode==0){ for(int i=0;i<N;++i){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:pso]; [en setBuffer:b offset:0 atIndex:0]; [en dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [en endEncoding]; } }
        else if(mode==1){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:pso]; [en setBuffer:b offset:0 atIndex:0]; for(int i=0;i<N;++i){ [en dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; if(i+1<N) [en memoryBarrierWithScope:MTLBarrierScopeBuffers]; } [en endEncoding]; }
        else if(mode==2){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:pso]; [en setBuffer:b offset:0 atIndex:0]; for(int i=0;i<N;++i) [en dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [en endEncoding]; }
        else if(mode==3){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; [en setComputePipelineState:pso]; [en setBuffer:b offset:0 atIndex:0]; for(int i=0;i<N;++i){ [en dispatchThreadgroupsWithIndirectBuffer:ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(64,1,1)]; if(i+1<N) [en memoryBarrierWithScope:MTLBarrierScopeBuffers]; } [en endEncoding]; }
        else { for(int i=0;i<N;++i){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:pso]; [en setBuffer:b offset:0 atIndex:0]; [en dispatchThreadgroupsWithIndirectBuffer:ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [en endEncoding]; } }
        [cb commit]; [cb waitUntilCompleted]; best=std::min(best,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
        printf("  %-52s N=%3d  %.4f ms total  %.1f us per dispatch\n", name, N, best, 1000.0*best/N); };
    printf("GPU-timestamp cost of tiny dispatches (M3 Max), min of 10:\n");
    for(int N : {1, 5, 20, 80}) run("one encoder per dispatch (serial)", N, 0);
    for(int N : {5, 20, 80}) run("one encoder, memoryBarrier between dispatches", N, 1);
    for(int N : {5, 20, 80}) run("one encoder, no barriers (serial encoder type)", N, 2);
    for(int N : {5, 20, 80}) run("concurrent encoder, indirect dispatch + barrier", N, 3);
    for(int N : {5, 20, 80}) run("one encoder per indirect dispatch", N, 4);
    return 0; } }
