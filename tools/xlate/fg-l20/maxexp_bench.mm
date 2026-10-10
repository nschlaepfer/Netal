#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <map>
struct Conv3x3Parameters { uint32_t width, height, input_channels, output_channels, active_weight_half_count, column_stage_schedule; };
int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W=160,H=96,CI=48,CO=32; const Conv3x3Parameters P{W,H,CI,CO,9u*CI*CO,0u};
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&e]; if(!L){printf("%s\n",e.localizedDescription.UTF8String);return 1;}
    auto pipe=[&](NSString* n){ id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[L newFunctionWithName:n] error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; };
    std::vector<uint8_t> wb; { FILE* f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); wb.resize(n); fread(wb.data(),1,n,f); fclose(f); }
    const size_t inh=(size_t)W*H*CI, outh=(size_t)W*H*CO;
    std::vector<_Float16> in(inh); std::mt19937 rng(20260904); std::normal_distribution<float> nd(0.f,1.f); for(auto&v:in) v=(_Float16)nd(rng);
    id<MTLBuffer> bw=[dev newBufferWithBytes:wb.data() length:wb.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi=[dev newBufferWithBytes:in.data() length:inh*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b1=[dev newBufferWithLength:outh options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2=[dev newBufferWithLength:outh options:MTLResourceStorageModeShared];
    id<MTLBuffer> bt=[dev newBufferWithLength:outh options:MTLResourceStorageModeShared];
    memset(b1.contents,0x7f,outh); memset(b2.contents,0x7f,outh); memset(bt.contents,0x7f,outh);
    { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
      [en setComputePipelineState:pipe(@"maxexp_mma")];
      [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bi offset:0 atIndex:1]; [en setBuffer:b1 offset:0 atIndex:2]; [en setBuffer:b2 offset:0 atIndex:3]; [en setBytes:&P length:sizeof(P) atIndex:4];
      [en dispatchThreads:MTLSizeMake(W,H,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding];
      id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder];
      [e2 setComputePipelineState:pipe(@"maxexp_truth")];
      [e2 setBuffer:bw offset:0 atIndex:0]; [e2 setBuffer:bi offset:0 atIndex:1]; [e2 setBuffer:bt offset:0 atIndex:2]; [e2 setBytes:&P length:sizeof(P) atIndex:3];
      [e2 dispatchThreads:MTLSizeMake(W,H,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [e2 endEncoding];
      [cb commit]; [cb waitUntilCompleted]; }
    const char* e1=(const char*)b1.contents; const char* e2v=(const char*)b2.contents; const char* t=(const char*)bt.contents;
    std::map<int,long> d1, d2; long n=0, exact2=0;
    for(size_t i=0;i<outh;++i){ if(t[i]==0x7f||t[i]==-100) continue; ++n; d1[e1[i]-t[i]]++; d2[e2v[i]-t[i]]++; if(e2v[i]==t[i]) ++exact2; }
    printf("%ld elements with a nonzero max term (stage 0)\n", n);
    printf("\nbase-2 sum, leading bit minus true max_exp:\n"); for(auto&[d,c]:d1) printf("   %+3d  %8ld  %6.2f%%\n", d, c, 100.0*c/n);
    printf("\nbase-32 stage-shifted, leading digit minus true max_exp:\n"); for(auto&[d,c]:d2) printf("   %+3d  %8ld  %6.2f%%\n", d, c, 100.0*c/n);
    printf("\nbase-32 method recovers max_exp EXACTLY on %ld of %ld (%.4f%%)%s\n", exact2, n, 100.0*exact2/n, exact2==n?"   <-- EXACT EVERYWHERE":"");
    return 0; } }
