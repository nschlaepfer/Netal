// block1 conv1, design six + packed weight/proxy table (design thirteen): bounded_wq (16 channels/simdgroup, two shortcut slots) -> ordered compaction ->
// pass 1 (up to two exact stages) -> pass 2 (full replay over list2) -> pool, against the scalar oracle. argv[1] metallib, argv[2] FORMAL 0/1,
// argv[3] weights token, argv[4] bias token (optional; synthetic weights otherwise).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>

// ===================================== build_wq_b1c1 (liftable verbatim) =====================================
// Packed weight/proxy table for block1 conv1's bounded kernel (fg_custom_block1_conv1_bounded_wq).
// Input: the family's weight buffer w, 4608 halves indexed by wi_b1c1(tile, ic, oc) (tile 0..8, ic 0..15, oc 0..31).
// Output: 2304 half4 = 9216 halves (18432 bytes), entry (tile, kb, pair, row) at half offset 4*wq_index_b1c1(...):
//   .x = w[wi(tile, kb*8+row, 2*pair)], .y = w[wi(tile, kb*8+row, 2*pair+1)], .z = q(.x), .w = q(.y),
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0 (half; equals bound_operand(float(x)) after the exact half->float widening).
//   wq_index_b1c1(tile, kb, pair, row) = ((tile*2 + kb)*16 + pair)*8 + row.
static inline uint32_t wi_b1c1(uint32_t tile, uint32_t ic, uint32_t oc) {
    uint32_t g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, frag = (k >= 8u ? 2u : 0u) + (k & 1u);
    return g * 1152u + tile * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u); }
static inline uint32_t wq_index_b1c1(uint32_t tile, uint32_t kb, uint32_t pair, uint32_t row) { return ((tile * 2u + kb) * 16u + pair) * 8u + row; }
static inline _Float16 wq_q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries_b1c1 = 9u * 2u * 16u * 8u;   // half4 entries; the table is 4 * this many halves
static std::vector<_Float16> build_wq_b1c1(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries_b1c1 * 4, (_Float16)0);
    for (uint32_t tile = 0; tile < 9; ++tile) for (uint32_t kb = 0; kb < 2; ++kb) for (uint32_t pair = 0; pair < 16; ++pair) for (uint32_t row = 0; row < 8; ++row) {
        const uint32_t ic = kb * 8u + row, oc = pair * 2u; _Float16* e = &wq[(size_t)wq_index_b1c1(tile, kb, pair, row) * 4];
        e[0] = w[wi_b1c1(tile, ic, oc)]; e[1] = w[wi_b1c1(tile, ic, oc + 1u)]; e[2] = wq_q_half(e[0]); e[3] = wq_q_half(e[1]); }
    return wq; }
// ===================================== end build_wq_b1c1 =====================================================
// Verification along the kernel's own access pattern (group.y, g, lane, tile, kb): (.x,.y) bitwise equal to the halves the
// _pf gather w[wi(tile, kb*8+mr, oc0+8g+h)] reads, (.z,.w) value-equal to bound_operand of them (and +0 signed), and every
// weight slot of the 4608 gathered exactly once.
static inline float wq_bound_operand_f(float v) { float m = fabsf(v); return m != 0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static void verify_wq_b1c1(const _Float16* w, const std::vector<_Float16>& wq) {
    std::vector<uint32_t> hits(4u * 1152u, 0); long bad = 0;
    for (uint32_t gy = 0; gy < 2; ++gy) for (uint32_t g = 0; g < 2; ++g) for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t mr = ((lane >> 1) & 3u) + 4u * ((lane >> 4) & 1u), mc = 2u * (lane & 1u) + 4u * ((lane >> 3) & 1u), oc0 = gy * 16u + mc, oc = oc0 + 8u * g;
        for (uint32_t tile = 0; tile < 9; ++tile) for (uint32_t kb = 0; kb < 2; ++kb) {
            const uint32_t i0 = wi_b1c1(tile, kb * 8u + mr, oc), i1 = wi_b1c1(tile, kb * 8u + mr, oc + 1u); ++hits[i0]; ++hits[i1];
            const _Float16* e = &wq[(size_t)wq_index_b1c1(tile, kb, (oc0 + 8u * g) >> 1u, mr) * 4]; uint16_t eb[4], wb[2]; memcpy(eb, e, 8); memcpy(wb + 0, &w[i0], 2); memcpy(wb + 1, &w[i1], 2);
            if (eb[0] != wb[0] || eb[1] != wb[1]) ++bad;
            if ((float)e[2] != wq_bound_operand_f((float)w[i0]) || (float)e[3] != wq_bound_operand_f((float)w[i1])) ++bad;
            if (std::signbit((float)e[2]) || std::signbit((float)e[3])) ++bad; } }
    if (bad) { printf("wq table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once = 0; for (uint32_t h : hits) if (h == 1u) ++once; if (once != (long)hits.size()) { printf("wq coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    printf("  wq table: %u half4 entries (%zu bytes) verified against wi (bitwise weights, bound proxies, bijective coverage of %zu slots)\n", kWqEntries_b1c1, wq.size() * 2, hits.size()); }
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    if(argc<3){printf("usage: %s <metallib> <formal 0/1> [weights.bin bias.bin]\n",argv[0]);return 2;}
    const uint32_t IW=320,IH=184,CI=16,OW=160,OH=92,CO=32; const size_t ni=(size_t)IW*IH*CI, nm=(size_t)IW*IH*CO, no=(size_t)OW*OH*CO; const int formal=atoi(argv[2]);
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(9*1152+64), iv(ni), bv(CO+8);
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    const char* wsrc="synthetic";
    if(argc>4){ auto load=[&](const char* path, std::vector<_Float16>& dst){ FILE* f=fopen(path,"rb"); if(!f){printf("cannot open %s\n",path);exit(1);}
            fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s is %ld bytes\n",path,n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
        load(argv[3],wv); load(argv[4],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),ni*2);
    std::vector<_Float16> wqv=build_wq_b1c1(wv.data()); verify_wq_b1c1(wv.data(), wqv); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);
    id<MTLBuffer> bo=mkl(no*2), bint=mkl(nm*2), bm=mkl(nm), brs=mkl(nm), bra=mkl(nm*2), bfa=mkl(nm*2), blist=mkl(nm*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nm*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(nm), bp2=mkl(nm*2), ba2=mkl(nm*2);
    const uint32_t total=(uint32_t)nm, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block1_conv1"), bounded=Pipe(dev,L,@"fg_custom_block1_conv1_bounded_wq",formal),
         pass1=Pipe(dev,L,@"fg_custom_block1_conv1_repair_pass1_wq"), pass2=Pipe(dev,L,@"fg_custom_block1_conv1_listed_wq"), pool=Pipe(dev,L,@"fg_custom_block1_conv1_pool_wq"), repool=Pipe(dev,L,@"fg_custom_block1_conv1_repool_wq"),
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
    printf("block1_conv1 wq [%s] %s bound: scalar %7.3f ms | pipeline %7.3f ms  %5.2fx  mismatches %ld of %zu  flagged %u (%.2f%% of %zu intermediates)  pass2 %u (%.2f%%)\n",
           wsrc, formal?"formal":"empirical", ts, tg, ts/tg, m, no, c1, 100.0*c1/nm, nm, c2, 100.0*c2/nm);
    return m==0?0:3; } }
