// L20: scalar oracle vs raw SIMD vs split-precision SIMD. The split is done here on the host and
// verified exact for every weight before anything runs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
struct Conv3x3Parameters { uint32_t width, height, input_channels, output_channels,
                           active_weight_half_count, column_stage_schedule; };
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, const char* lib, NSString* fn) {
    NSError* e=nil; id<MTLLibrary> L=[d newLibraryWithURL:[NSURL fileURLWithPath:@(lib)] error:&e];
    if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);exit(1);}
    id<MTLFunction> F=[L newFunctionWithName:fn]; if(!F){printf("fn %s\n",fn.UTF8String);exit(1);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e];
    if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p;
}
int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W=160,H=96,CI=48,CO=32; const Conv3x3Parameters P{W,H,CI,CO,9u*CI*CO,0u};
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
    std::vector<uint8_t> wb; { FILE* f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); wb.resize(n); fread(wb.data(),1,n,f); fclose(f); }
    const size_t nw=wb.size()/2; const uint16_t* w16=(const uint16_t*)wb.data();
    // Split: hi keeps sign, exponent and the top 5 of 10 fraction bits; lo is the exact remainder.
    std::vector<uint16_t> whi(nw), wlo(nw); long inexact=0;
    // Only the active weights are split. The bias region after them (and the zero gap) must stay
    // exact - the kernel reads its bias from this same buffer, and truncating it was a 63% miss.
    const size_t active_end = (size_t)P.active_weight_half_count * 2u;
    for(size_t i=0;i<nw;++i){
        uint16_t b=w16[i];
        if(i >= active_end){ whi[i]=b; wlo[i]=0u; continue; }
        uint16_t hi=b & 0xFFE0u;   // clear low 5 fraction bits
        _Float16 fb,fh; memcpy(&fb,&b,2); memcpy(&fh,&hi,2);
        _Float16 lo=(_Float16)((float)fb-(float)fh);
        if((float)fh+(float)lo!=(float)fb) ++inexact;
        whi[i]=hi; memcpy(&wlo[i],&lo,2);
    }
    printf("weights %zu halves, split inexact for %ld  %s\n", nw, inexact, inexact?"<-- NOT EXACT":"(exact for every weight)");
    const size_t inh=(size_t)W*H*CI, outh=(size_t)W*H*CO;
    std::vector<_Float16> in(inh); std::mt19937 rng(20260904); std::normal_distribution<float> nd(0.f,1.f);
    for(auto&v:in) v=(_Float16)nd(rng);
    id<MTLBuffer> bw=[dev newBufferWithBytes:wb.data() length:wb.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bwh=[dev newBufferWithBytes:whi.data() length:nw*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bwl=[dev newBufferWithBytes:wlo.data() length:nw*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi=[dev newBufferWithBytes:in.data() length:in.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo=[dev newBufferWithLength:outh*2 options:MTLResourceStorageModeShared];
    auto run=[&](id<MTLComputePipelineState> pso, id<MTLBuffer> weights, id<MTLBuffer> lo, int reps){
        double best=1e9;
        for(int r=0;r<reps;++r){ memset(bo.contents,0xA5,outh*2);
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso]; [e setBuffer:weights offset:0 atIndex:0]; [e setBuffer:bi offset:0 atIndex:1];
            [e setBuffer:bo offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
            if(lo) [e setBuffer:lo offset:0 atIndex:4];
            [e dispatchThreads:MTLSizeMake(W,H,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            best=std::min(best,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
        return best; };
    std::vector<uint16_t> ref(outh), got(outh);
    double ts=run(Pipe(dev,argv[2],@"fg_conv_3x3"),bw,nil,10); memcpy(ref.data(),bo.contents,outh*2);
    double tr=run(Pipe(dev,argv[3],@"fg_conv_3x3_simd"),bw,nil,10); memcpy(got.data(),bo.contents,outh*2);
    long mr=0; for(size_t i=0;i<outh;++i) if(got[i]!=ref[i]) ++mr;
    double tp=run(Pipe(dev,argv[4],@"fg_conv_3x3_split"),bwh,bwl,10); memcpy(got.data(),bo.contents,outh*2);
    long mp=0; for(size_t i=0;i<outh;++i) if(got[i]!=ref[i]) ++mp;
    std::vector<uint16_t> tf(outh), sp(outh); memcpy(sp.data(), got.data(), outh*2);
    double tt = run(Pipe(dev,argv[5],@"fg_conv_3x3"),bw,nil,10); memcpy(tf.data(),bo.contents,outh*2);
    long mt=0, sp_vs_tf=0; for(size_t i=0;i<outh;++i){ if(tf[i]!=ref[i]) ++mt; if(sp[i]!=tf[i]) ++sp_vs_tf; }
    double t4 = run(Pipe(dev,argv[6],@"fg_conv_3x3_split4"),bwh,bwl,10); memcpy(got.data(),bo.contents,outh*2);
    long m4=0, s4_vs_tf=0; for(size_t i=0;i<outh;++i){ if(got[i]!=ref[i]) ++m4; if(got[i]!=tf[i]) ++s4_vs_tf; }
    printf("scalar oracle   %7.3f ms\n", ts);
    printf("double-float scalar (exact sum) %7.3f ms  %5.2fx  %ld differing vs oracle\n", tt, ts/tt, mt);
    printf("raw simd        %7.3f ms  %5.2fx  %ld differing (%.4f%%)\n", tr, ts/tr, mr, 100.0*mr/outh);
    printf("split-precision %7.3f ms  %5.2fx  %ld differing (%.4f%%)%s\n", tp, ts/tp, mp, 100.0*mp/outh, mp==0?"  <-- BIT-EXACT":"");
    printf("split4 (a and b) %7.3f ms  %5.2fx  %ld differing vs oracle (%.4f%%)\n", t4, ts/t4, m4, 100.0*m4/outh);
    printf("\n  split-b vs double-float scalar: %ld elements differ   split4 vs double-float: %ld\n", sp_vs_tf, s4_vs_tf);
    printf("  (zero here means the split MMA is computing the exact sum, and the residual vs the oracle is NVIDIA's truncation)\n");
    return 0; } }
