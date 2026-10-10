// conv1 family, design six, end to end against the scalar oracle:
// bounded_wq (pf + packed weight/proxy half4 table on buffer 12, built and verified by build_wq_conv1) -> ordered compaction -> pass 1 (one or two
// exact stages, clz helper) -> pass 2 (full replay over list2, clz helper) -> pool.
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] weights token, argv[4] bias token (optional; synthetic otherwise)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
// ============================================================================================================
// build_wq_conv1: packed weight/proxy table for fg_custom_block0_conv1_bounded_wq (buffer 12). Self-contained: lift
// verbatim. Input: the family's weight buffer w (at least 18432 halves, the range of wi_conv1). Output: a flat half
// array of 9216 half4 = 36864 halves = 73728 bytes, entry index
//   wq_index_conv1(tile, stage, kb, pair, row) = (((tile*2u+stage)*2u+kb)*32u+pair)*8u+row
// holding (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for ic = stage*16u+kb*8u+matrix_row, oc = 2*pair, where
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0   (== the kernel's bound_operand() on the exactly widened half).
// The function verifies the table before returning (exit(2) on failure): along the bounded kernel's own access
// pattern (group.y, g, lane, tile, stage, kb, h) every (.x,.y) is bitwise the half the pf gather
// w[wi_conv1(...)] read, every (.z,.w) equals bound_operand of it (value-wise, non-negative), and every one of the
// 18432 weight slots is gathered exactly once (bijection).
static inline uint32_t wi_conv1(uint32_t tile, uint32_t stage, uint32_t ic, uint32_t oc) { uint32_t g=oc/8u, n=oc%8u, k=ic%16u, lane=n*4u+(k%8u)/2u; uint32_t frag=(k>=8u?2u:0u)+(k&1u); return g*2304u+tile*256u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u); }
static inline uint32_t wq_index_conv1(uint32_t tile, uint32_t stage, uint32_t kb, uint32_t pair, uint32_t row) { return (((tile*2u+stage)*2u+kb)*32u+pair)*8u+row; }
static std::vector<_Float16> build_wq_conv1(const _Float16* w) {
    const uint32_t kEntries = 9216u, kWeights = 18432u;
    std::vector<_Float16> wq((size_t)kEntries*4, (_Float16)0);
    auto q_half = [](_Float16 x) -> _Float16 { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
    auto bound_operand_f = [](float v) -> float { float m = fabsf(v); return m != 0.0f ? std::max(m, 0x1p-14f) : 0.0f; };
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<32; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = stage*16u+kb*8u+row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index_conv1(tile, stage, kb, pair, row)*4];
        e[0] = w[wi_conv1(tile, stage, ic, oc)]; e[1] = w[wi_conv1(tile, stage, ic, oc+1u)]; e[2] = q_half(e[0]); e[3] = q_half(e[1]); }
    std::vector<uint32_t> hits(kWeights, 0); long bad = 0;
    for (uint32_t gy=0; gy<4; ++gy) for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t mr=((lane>>1)&3u)+4u*((lane>>4)&1u), mc=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=gy*16u+mc;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) {
            const _Float16* e = &wq[(size_t)wq_index_conv1(tile, stage, kb, (oc0+8u*g)>>1u, mr)*4]; uint16_t eb[4]; memcpy(eb, e, 8);
            for (uint32_t h=0; h<2; ++h) {
                const uint32_t i = wi_conv1(tile, stage, stage*16u+kb*8u+mr, oc0+8u*g+h); if (i >= kWeights) { ++bad; continue; } ++hits[i];
                uint16_t wb; memcpy(&wb, &w[i], 2); if (eb[h] != wb) ++bad;
                if ((float)e[2+h] != bound_operand_f((float)w[i]) || std::signbit((float)e[2+h])) ++bad; } } }
    if (bad) { printf("wq_conv1 table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once=0; for (uint32_t h: hits) if (h==1u) ++once;
    if (once != (long)hits.size()) { printf("wq_conv1 coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    return wq;
}
// ============================================================================================================
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    if(argc<3){printf("usage: %s <metallib> <formal 0/1> [weights.bin bias.bin]\n",argv[0]);return 1;}
    const uint32_t IW=160,IH=92,CI=32,OW=80,OH=46,CO=64; const size_t ni=(size_t)IW*IH*CI, nm=(size_t)IW*IH*CO, no=(size_t)OW*OH*CO; const int formal=atoi(argv[2]);
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(8*2304+64), iv(ni), bv(CO+8);
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    const char* wsrc="synthetic";
    if(argc>4){ auto load=[&](const char* path, std::vector<_Float16>& dst){ FILE* f=fopen(path,"rb"); if(!f){printf("cannot open %s\n",path);exit(1);}
            fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s is %ld bytes\n",path,n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
        load(argv[3],wv); load(argv[4],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),ni*2);
    std::vector<_Float16> wqv = build_wq_conv1(wv.data()); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);   // 9216 half4
    id<MTLBuffer> bo=mkl(no*2), bint=mkl(nm*2), bm=mkl(nm), brs=mkl(nm), bra=mkl(nm*2), bfa=mkl(nm*2), blist=mkl(nm*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nm*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(nm), bp2=mkl(nm*2), ba2=mkl(nm*2);
    const uint32_t total=(uint32_t)nm, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block0_conv1"), bounded=Pipe(dev,L,@"fg_custom_block0_conv1_bounded_wq",formal),
         pass1=Pipe(dev,L,@"fg_custom_block0_conv1_repair_pass1_wq"), pass2=Pipe(dev,L,@"fg_custom_block0_conv1_listed_wq"), pool=Pipe(dev,L,@"fg_custom_block0_conv1_pool_wq"), repool=Pipe(dev,L,@"fg_custom_block0_conv1_repool_wq"),
         compact=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bint.contents,0xA5,nm*2); memset(bm.contents,0,nm); memset(brs.contents,0xFF,nm); memset(bra.contents,0,nm*2); memset(bfa.contents,0,nm*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    std::vector<uint16_t> ref(no), got(no);
    double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:bo offset:0 atIndex:3];
        [en dispatchThreads:MTLSizeMake(OW,OH,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    memcpy(ref.data(),bo.contents,no*2);
    double tg=1e9; uint32_t c1=0,c2=0;
    for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded];
        [e1 setBuffer:bw offset:0 atIndex:0]; [e1 setBuffer:bb offset:0 atIndex:1]; [e1 setBuffer:bi offset:0 atIndex:2]; [e1 setBuffer:bint offset:0 atIndex:3];
        [e1 setBuffer:bm offset:0 atIndex:4]; [e1 setBuffer:brs offset:0 atIndex:5]; [e1 setBuffer:bra offset:0 atIndex:6]; [e1 setBuffer:bfa offset:0 atIndex:7]; [e1 setBuffer:bs2 offset:0 atIndex:8]; [e1 setBuffer:bp2 offset:0 atIndex:9]; [e1 setBuffer:ba2 offset:0 atIndex:10]; [e1 setBuffer:bo offset:0 atIndex:11]; [e1 setBuffer:bwq offset:0 atIndex:12];
        [e1 dispatchThreadgroups:MTLSizeMake((IW*IH)/8,CO/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((nm+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1];
        [e3 setBuffer:bw offset:0 atIndex:0]; [e3 setBuffer:bb offset:0 atIndex:1]; [e3 setBuffer:bi offset:0 atIndex:2]; [e3 setBuffer:bint offset:0 atIndex:3];
        [e3 setBuffer:blist offset:0 atIndex:4]; [e3 setBuffer:bcnt offset:0 atIndex:5]; [e3 setBuffer:brs offset:0 atIndex:6]; [e3 setBuffer:bra offset:0 atIndex:7];
        [e3 setBuffer:bfa offset:0 atIndex:8]; [e3 setBuffer:bm offset:0 atIndex:9]; [e3 setBuffer:blist2 offset:0 atIndex:10]; [e3 setBuffer:bcnt2 offset:0 atIndex:11]; [e3 setBuffer:bs2 offset:0 atIndex:12]; [e3 setBuffer:bp2 offset:0 atIndex:13]; [e3 setBuffer:ba2 offset:0 atIndex:14];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
        id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&gsz length:4 atIndex:2];
        [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding];
        id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2];
        [e5 setBuffer:bw offset:0 atIndex:0]; [e5 setBuffer:bb offset:0 atIndex:1]; [e5 setBuffer:bi offset:0 atIndex:2]; [e5 setBuffer:bint offset:0 atIndex:3];
        [e5 setBuffer:blist2 offset:0 atIndex:4]; [e5 setBuffer:bcnt2 offset:0 atIndex:5]; [e5 setBuffer:brs offset:0 atIndex:6]; [e5 setBuffer:bra offset:0 atIndex:7];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding];
        id<MTLComputeCommandEncoder> e6=[cb computeCommandEncoder]; [e6 setComputePipelineState:repool]; [e6 setBuffer:bint offset:0 atIndex:0]; [e6 setBuffer:bo offset:0 atIndex:1]; [e6 setBuffer:blist2 offset:0 atIndex:2]; [e6 setBuffer:bcnt2 offset:0 atIndex:3];
        [e6 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e6 endEncoding];
        [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0){c1=*(uint32_t*)bcnt.contents;c2=*(uint32_t*)bcnt2.contents;} }
    memcpy(got.data(),bo.contents,no*2); long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m;
    printf("conv1 wq [%s] %s bound: scalar %7.3f ms | pipeline %7.3f ms  %5.2fx  mismatches %ld of %zu  flagged %u (%.2f%% of %zu)  pass2 %u (%.2f%%)\n",
           wsrc, formal?"formal":"empirical", ts, tg, ts/tg, m, no, c1, 100.0*c1/nm, nm, c2, 100.0*c2/nm);
    return m==0?0:3; } }
