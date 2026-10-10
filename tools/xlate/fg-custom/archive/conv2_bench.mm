// conv2 family: scalar reference against the SIMD rewrite. Synthetic weights - this measures the
// rewrite's correctness against the same source of truth and its speedup on the real shape; the
// live-token validation is a separate step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
struct Params { uint32_t residual; };
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn) {
    NSError* e=nil; id<MTLFunction> F=[L newFunctionWithName:fn];
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e];
    if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p;
}
int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W=80,H=46,C=64;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil;
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
    if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    id<MTLCommandQueue> q=[dev newCommandQueue];
    const size_t nw=8*4608, ni=(size_t)W*H*C, no=ni;
    std::vector<_Float16> wv(nw), iv(ni), rv(no), bv(C+8);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng);
    for(auto&v:rv)v=(_Float16)nd(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    id<MTLBuffer> bw=[dev newBufferWithBytes:wv.data() length:wv.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb=[dev newBufferWithBytes:bv.data() length:bv.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi=[dev newBufferWithBytes:iv.data() length:iv.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> br=[dev newBufferWithBytes:rv.data() length:rv.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo=[dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
    Params P{1u};
    auto run=[&](id<MTLComputePipelineState> pso,bool simd,int reps){
        double best=1e9;
        for(int r=0;r<reps;++r){
            memset(bo.contents,0xA5,no*2);
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
            [en setComputePipelineState:pso];
            [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1];
            [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:br offset:0 atIndex:3];
            [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5];
            if(simd) [en dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            else [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)];
            [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted){printf("cmd failed\n");return -1.0;}
            best=std::min(best,(cb.GPUEndTime-cb.GPUStartTime)*1000.0);
        } return best; };
    std::vector<uint16_t> ref(no),got(no);
    double ts=run(Pipe(dev,L,@"fg_custom_block0_conv2"),false,8);
    memcpy(ref.data(),bo.contents,no*2);
    double tm=run(Pipe(dev,L,@"fg_custom_block0_conv2_simd"),true,8);
    memcpy(got.data(),bo.contents,no*2);
    long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m;
    printf("conv2 scalar %7.3f ms   simd %7.3f ms   %5.2fx   %ld of %zu differing (%.3f%%)\n",
           ts,tm,ts/tm,m,no,100.0*m/no);
    return 0; } }
