// block1 conv0, design six with the specialised K=8 tail (_pf2) + packed weight/proxy table (_wq, design thirteen): bounded_wq (16 channels/simdgroup,
// two shortcut slots, fused pool) -> ordered compaction -> pass 1 (up to two exact steps) -> pass 2 (full replay over list2) -> re-pool, against the
// scalar oracle. The half4 table (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) is built on the host from the weight buffer by build_wq_block1_conv0 and
// verified against wi() along the bounded kernel's own access pattern (g, lane, tile, slot) before any launch.
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] weights token, argv[4] bias token (optional; synthetic weights otherwise).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
// Plain row-major weight index of the family: (oc*9 + tile)*18 + ic, 16 x 9 x 18 = 2592 halves.
static inline uint32_t wi(uint32_t tile, uint32_t ic, uint32_t oc) { return (oc*9u + tile)*18u + ic; }
// Packed table index, mirroring the kernel's wq_index: (tile, slot, oc-pair, matrix_row) with slot = stage*2 + kb; slot 0/1 = the K=16 stage's kb 0/1
// (k = kb*8 + row), slot 2 = the tail stage's kb=0 (k = 16 + row for row < 2, structurally zero for row >= 2). 9*3*8*8 = 1728 half4 = 13824 bytes;
// for fixed (tile, slot, g) the 32 lanes read pair = (mc + 8g)/2, row = mr: 32 consecutive half4.
static inline uint32_t wq_index(uint32_t tile, uint32_t slot, uint32_t pair, uint32_t row) { return ((tile*3u + slot)*8u + pair)*8u + row; }
static const uint32_t kWqEntries = 9u*3u*8u*8u;
static inline float bound_operand_f(float v) { float m=fabsf(v); return m!=0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static inline _Float16 q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
// Build the packed table from the plain weights (w has at least 2592 halves) and verify it along the kernel's access pattern: bitwise weights,
// proxies value-equal to bound_operand of the widened weight and non-negative, tail rows >= 2 all +0, and every one of the 2592 weight slots gathered
// exactly once (bijection). Returns kWqEntries*4 halves.
static std::vector<_Float16> build_wq_block1_conv0(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries*4, (_Float16)0);
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t slot=0; slot<3; ++slot) for (uint32_t pair=0; pair<8; ++pair) for (uint32_t row=0; row<8; ++row) {
        _Float16* e = &wq[(size_t)wq_index(tile,slot,pair,row)*4];
        if (slot==2u && row>=2u) continue;   // tail rows 2..7: +0 in every component
        const uint32_t ic = slot<2u ? slot*8u + row : 16u + row, oc = pair*2u;
        e[0]=w[wi(tile,ic,oc)]; e[1]=w[wi(tile,ic,oc+1u)]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
    std::vector<uint32_t> hits(16u*9u*18u, 0); long bad=0;
    for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t mr=((lane>>1)&3u)+4u*((lane>>4)&1u), mc=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=mc, oc=oc0+8u*g;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t slot=0; slot<3; ++slot) {
            const _Float16* e=&wq[(size_t)wq_index(tile,slot,(oc0+8u*g)>>1u,mr)*4]; uint16_t eb[4]; memcpy(eb,e,8);
            if (slot==2u && mr>=2u) { if (eb[0]|eb[1]|eb[2]|eb[3]) ++bad; continue; }   // the _pf2 predicate's zeros, bitwise +0
            const uint32_t ic_b = slot<2u ? slot*8u + mr : 16u + mr; const uint32_t i0=wi(tile,ic_b,oc), i1=wi(tile,ic_b,oc+1u); ++hits[i0]; ++hits[i1];
            uint16_t wb[2]; memcpy(wb+0,&w[i0],2); memcpy(wb+1,&w[i1],2);
            if (eb[0]!=wb[0] || eb[1]!=wb[1]) ++bad;
            if ((float)e[2]!=bound_operand_f((float)w[i0]) || (float)e[3]!=bound_operand_f((float)w[i1])) ++bad;
            if (std::signbit((float)e[2]) || std::signbit((float)e[3])) ++bad; } }
    if (bad) { printf("wq table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once=0; for (uint32_t h: hits) if (h==1u) ++once; if (once!=(long)hits.size()) { printf("wq coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    return wq;
}
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    if(argc<3){printf("usage: %s <metallib> <formal 0/1> [weights.bin bias.bin]\n",argv[0]);return 2;}
    const uint32_t IW=640,IH=368,CI=18,OW=320,OH=184,CO=16; const size_t ni=(size_t)IW*IH*CI, nm=(size_t)IW*IH*CO, no=(size_t)OW*OH*CO; const int formal=atoi(argv[2]);
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(16*9*18+64), iv(ni), bv(CO+8);
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    const char* wsrc="synthetic";
    if(argc>4){ auto load=[&](const char* path, std::vector<_Float16>& dst){ FILE* f=fopen(path,"rb"); if(!f){printf("cannot open %s\n",path);exit(1);}
            fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s is %ld bytes\n",path,n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
        load(argv[3],wv); load(argv[4],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),ni*2);
    std::vector<_Float16> wqv=build_wq_block1_conv0(wv.data()); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);
    id<MTLBuffer> bo=mkl(no*2), bint=mkl(nm*2), bm=mkl(nm), brs=mkl(nm), bra=mkl(nm*2), bfa=mkl(nm*2), blist=mkl(nm*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nm*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(nm), bp2=mkl(nm*2), ba2=mkl(nm*2);
    const uint32_t total=(uint32_t)nm, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block1_conv0"), bounded=Pipe(dev,L,@"fg_custom_block1_conv0_bounded_wq",formal),
         pass1=Pipe(dev,L,@"fg_custom_block1_conv0_repair_pass1_wq"), pass2=Pipe(dev,L,@"fg_custom_block1_conv0_listed_wq"), pool=Pipe(dev,L,@"fg_custom_block1_conv0_pool_wq"), repool=Pipe(dev,L,@"fg_custom_block1_conv0_repool_wq"),
         compact=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:bint offset:0 atIndex:3]; };
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bint.contents,0xA5,nm*2); memset(bm.contents,0,nm); memset(brs.contents,0xFF,nm); memset(bra.contents,0,nm*2); memset(bfa.contents,0,nm*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    std::vector<uint16_t> ref(no), got(no);
    double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:bo offset:0 atIndex:3];
        [en dispatchThreads:MTLSizeMake(OW,OH,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    memcpy(ref.data(),bo.contents,no*2);
    double tg=1e9; uint32_t c1=0,c2=0;
    for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1);
        [e1 setBuffer:bm offset:0 atIndex:4]; [e1 setBuffer:brs offset:0 atIndex:5]; [e1 setBuffer:bra offset:0 atIndex:6]; [e1 setBuffer:bfa offset:0 atIndex:7]; [e1 setBuffer:bs2 offset:0 atIndex:8]; [e1 setBuffer:bp2 offset:0 atIndex:9]; [e1 setBuffer:ba2 offset:0 atIndex:10]; [e1 setBuffer:bo offset:0 atIndex:11]; [e1 setBuffer:bwq offset:0 atIndex:12];
        [e1 dispatchThreadgroups:MTLSizeMake((IW*IH)/8,CO/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((nm+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; common(e3);
        [e3 setBuffer:blist offset:0 atIndex:4]; [e3 setBuffer:bcnt offset:0 atIndex:5]; [e3 setBuffer:brs offset:0 atIndex:6]; [e3 setBuffer:bra offset:0 atIndex:7];
        [e3 setBuffer:bfa offset:0 atIndex:8]; [e3 setBuffer:bm offset:0 atIndex:9]; [e3 setBuffer:blist2 offset:0 atIndex:10]; [e3 setBuffer:bcnt2 offset:0 atIndex:11]; [e3 setBuffer:bs2 offset:0 atIndex:12]; [e3 setBuffer:bp2 offset:0 atIndex:13]; [e3 setBuffer:ba2 offset:0 atIndex:14];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
        id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&gsz length:4 atIndex:2];
        [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding];
        id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5);
        [e5 setBuffer:blist2 offset:0 atIndex:4]; [e5 setBuffer:bcnt2 offset:0 atIndex:5]; [e5 setBuffer:brs offset:0 atIndex:6]; [e5 setBuffer:bra offset:0 atIndex:7];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding];
        id<MTLComputeCommandEncoder> e6=[cb computeCommandEncoder]; [e6 setComputePipelineState:repool]; [e6 setBuffer:bint offset:0 atIndex:0]; [e6 setBuffer:bo offset:0 atIndex:1]; [e6 setBuffer:blist2 offset:0 atIndex:2]; [e6 setBuffer:bcnt2 offset:0 atIndex:3];
        [e6 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e6 endEncoding];
        [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0){c1=*(uint32_t*)bcnt.contents;c2=*(uint32_t*)bcnt2.contents;} }
    memcpy(got.data(),bo.contents,no*2); long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m;
    printf("block1_conv0 wq (pf2 + packed weight/proxy table, %u half4 verified) [%s] %s bound: scalar %7.3f ms | pipeline %7.3f ms  %5.2fx  mismatches %ld of %zu  flagged %u (%.2f%% of %zu intermediates)  pass2 %u (%.2f%%)\n",
           kWqEntries, wsrc, formal?"formal":"empirical", ts, tg, ts/tg, m, no, c1, 100.0*c1/nm, nm, c2, 100.0*c2/nm);
    return m==0?0:3; } }
