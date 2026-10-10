// block1 bottom0, design thirteen (_wq) with the producers writing their own indirect-dispatch arguments (_ind):
//   upsample_ind -> bounded_ind -> fg_compact_suspects_ordered_ind (+indirect1) -> pass 1 _ind (+indirect2) -> pass 2 _ind
// The _wq pipeline (ordered compaction + fg_indirect_from_counter, pass 1 _wq, fg_indirect_from_counter, pass 2 _wq) is kept in the same
// binary and timed interleaved (ind, wq, ind, wq, ...), min of 6 each. Every _ind run is checked for exactness against the scalar oracle
// (three planes) and for indirect1[0]==ceil(counter/64), indirect2[0]==ceil(counter2/64), done==compaction groups, done2==indirect1[0];
// then N back-to-back _ind runs (default 200) must all be byte-identical to the first and to the oracle.
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] weights token, argv[4] bias token (optional; synthetic weights otherwise), argv[5] stability runs (default 200)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>

// ===================================== build_wq_b1b0 (liftable verbatim) =====================================
// Packed weight/proxy table for block1 bottom0's bounded kernel (fg_custom_block1_bot0_bounded_wq).
// Input: the family's weight buffer w, 13824 halves indexed by wi_b1b0(t, s, ic, oc) (t 0..8, s 0..1, ic = s*16 + 0..15, oc 0..47).
// Output: 6912 half4 = 27648 halves (55296 bytes), entry (t, s, kb, pair, row) at half offset 4*wq_index_b1b0(...):
//   .x = w[wi(t, s, s*16+kb*8+row, 2*pair)], .y = w[wi(t, s, s*16+kb*8+row, 2*pair+1)], .z = q(.x), .w = q(.y),
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0 (half; equals bound_operand(float(x)) after the exact half->float widening).
//   wq_index_b1b0(t, s, kb, pair, row) = (((t*2 + s)*2 + kb)*24 + pair)*8 + row.
static inline uint32_t wi_b1b0(uint32_t t, uint32_t s, uint32_t ic, uint32_t oc) {
    uint32_t g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
    return g * 2304u + t * 256u + s * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u); }
static inline uint32_t wq_index_b1b0(uint32_t t, uint32_t s, uint32_t kb, uint32_t pair, uint32_t row) { return (((t * 2u + s) * 2u + kb) * 24u + pair) * 8u + row; }
static inline _Float16 wq_q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries_b1b0 = 9u * 2u * 2u * 24u * 8u;   // half4 entries; the table is 4 * this many halves
static std::vector<_Float16> build_wq_b1b0(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries_b1b0 * 4, (_Float16)0);
    for (uint32_t t = 0; t < 9; ++t) for (uint32_t s = 0; s < 2; ++s) for (uint32_t kb = 0; kb < 2; ++kb) for (uint32_t pair = 0; pair < 24; ++pair) for (uint32_t row = 0; row < 8; ++row) {
        const uint32_t ic = s * 16u + kb * 8u + row, oc = pair * 2u; _Float16* e = &wq[(size_t)wq_index_b1b0(t, s, kb, pair, row) * 4];
        e[0] = w[wi_b1b0(t, s, ic, oc)]; e[1] = w[wi_b1b0(t, s, ic, oc + 1u)]; e[2] = wq_q_half(e[0]); e[3] = wq_q_half(e[1]); }
    return wq; }
// ===================================== end build_wq_b1b0 =====================================================
static inline float wq_bound_operand_f(float v) { float m = fabsf(v); return m != 0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static void verify_wq_b1b0(const _Float16* w, const std::vector<_Float16>& wq) {
    std::vector<uint32_t> hits(6u * 2304u, 0); long bad = 0;
    for (uint32_t gy = 0; gy < 3; ++gy) for (uint32_t g = 0; g < 2; ++g) for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t mr = ((lane >> 1) & 3u) + 4u * ((lane >> 4) & 1u), mc = 2u * (lane & 1u) + 4u * ((lane >> 3) & 1u), oc0 = gy * 16u + mc, oc = oc0 + 8u * g;
        for (uint32_t t = 0; t < 9; ++t) for (uint32_t s = 0; s < 2; ++s) for (uint32_t kb = 0; kb < 2; ++kb) {
            const uint32_t ic = s * 16u + kb * 8u + mr, i0 = wi_b1b0(t, s, ic, oc), i1 = wi_b1b0(t, s, ic, oc + 1u); ++hits[i0]; ++hits[i1];
            const _Float16* e = &wq[(size_t)wq_index_b1b0(t, s, kb, (oc0 + 8u * g) >> 1u, mr) * 4]; uint16_t eb[4], wb[2]; memcpy(eb, e, 8); memcpy(wb + 0, &w[i0], 2); memcpy(wb + 1, &w[i1], 2);
            if (eb[0] != wb[0] || eb[1] != wb[1]) ++bad;
            if ((float)e[2] != wq_bound_operand_f((float)w[i0]) || (float)e[3] != wq_bound_operand_f((float)w[i1])) ++bad;
            if (std::signbit((float)e[2]) || std::signbit((float)e[3])) ++bad; } }
    if (bad) { printf("wq table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once = 0; for (uint32_t h : hits) if (h == 1u) ++once; if (once != (long)hits.size()) { printf("wq coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    printf("  wq table: %u half4 entries (%zu bytes) verified against wi (bitwise weights, bound proxies, bijective coverage of %zu slots)\n", kWqEntries_b1b0, wq.size() * 2, hits.size()); }
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    if(argc<3){printf("usage: %s <metallib> <formal 0/1> [weights.bin bias.bin [stability_runs]]\n",argv[0]);return 2;}
    const size_t nin=(size_t)160*92*32, nups=(size_t)320*184*32, nplane=(size_t)320*184*16, nvirt=(size_t)320*184*48; const int formal=atoi(argv[2]); const int stab_runs=argc>5?atoi(argv[5]):200;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(6*2304), iv(nin), bv(48+8);
    std::mt19937 rng(13); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    const char* wsrc="synthetic";
    if(argc>4){ auto load=[&](const char* p, std::vector<_Float16>& d){ FILE* f=fopen(p,"rb"); if(!f){printf("cannot open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>d.size()*2){printf("%s %ld bytes\n",p,n);exit(1);} fread(d.data(),1,n,f); fclose(f); };
        load(argv[3],wv); load(argv[4],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),nin*2), bups=mkl(nups*2);
    std::vector<_Float16> wqv=build_wq_b1b0(wv.data()); verify_wq_b1b0(wv.data(), wqv); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);
    id<MTLBuffer> p0=mkl(nplane*2), p1=mkl(nplane*2), p2=mkl(nplane*2), bm=mkl(nvirt), brs=mkl(nvirt), bra=mkl(nvirt*2), bfa=mkl(nvirt*2), bs2=mkl(nvirt), bp2=mkl(nvirt*2), ba2=mkl(nvirt*2);
    id<MTLBuffer> blist=mkl(nvirt*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nvirt*4), bcnt2=mkl(4), bind2=mkl(12), bdone=mkl(4), bdone2=mkl(4);
    const uint32_t total=(uint32_t)nvirt, gsz=64, per_group2=64; const uint32_t compact_groups=(uint32_t)((nvirt+255)/256);
    auto scalar=Pipe(dev,L,@"fg_custom_block1_bot0");
    auto ups_i=Pipe(dev,L,@"fg_custom_block1_bot0_upsample_ind"), bounded_i=Pipe(dev,L,@"fg_custom_block1_bot0_bounded_ind",formal), pass1_i=Pipe(dev,L,@"fg_custom_block1_bot0_repair_pass1_ind"), pass2_i=Pipe(dev,L,@"fg_custom_block1_bot0_listed_ind"), compact_i=Pipe(dev,L,@"fg_compact_suspects_ordered_ind");
    auto ups_w=Pipe(dev,L,@"fg_custom_block1_bot0_upsample_wq"), bounded_w=Pipe(dev,L,@"fg_custom_block1_bot0_bounded_wq",formal), pass1_w=Pipe(dev,L,@"fg_custom_block1_bot0_repair_pass1_wq"), pass2_w=Pipe(dev,L,@"fg_custom_block1_bot0_listed_wq"), compact_w=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    // Both pipelines' counters, done counters and indirect buffers are reset here; indirect = {0,1,1} so a zero-group pass 2 (no pass 1 groups ran) still has sane y/z.
    auto reset=[&](){ for(auto b:{p0,p1,p2}) memset(b.contents,0xA5,nplane*2); memset(bm.contents,0,nvirt); memset(brs.contents,0xFF,nvirt); memset(bra.contents,0,nvirt*2); memset(bfa.contents,0,nvirt*2); memset(bs2.contents,0xFF,nvirt); memset(bp2.contents,0,nvirt*2); memset(ba2.contents,0,nvirt*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; *(uint32_t*)bdone.contents=0; *(uint32_t*)bdone2.contents=0;
        uint32_t* i1=(uint32_t*)bind.contents; uint32_t* i2=(uint32_t*)bind2.contents; i1[0]=0;i1[1]=1;i1[2]=1; i2[0]=0;i2[1]=1;i2[2]=1; };
    auto planes=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:p0 offset:0 atIndex:3]; [en setBuffer:p1 offset:0 atIndex:4]; [en setBuffer:p2 offset:0 atIndex:5]; };
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bups offset:0 atIndex:2]; planes(en); };
    std::vector<uint16_t> ref(3*nplane), got(3*nplane), first(3*nplane);
    auto grab=[&](std::vector<uint16_t>& v){ memcpy(v.data(),p0.contents,nplane*2); memcpy(v.data()+nplane,p1.contents,nplane*2); memcpy(v.data()+2*nplane,p2.contents,nplane*2); };
    auto enc_ups=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){ id<MTLComputeCommandEncoder> e0=[cb computeCommandEncoder]; [e0 setComputePipelineState:pso]; [e0 setBuffer:bi offset:0 atIndex:0]; [e0 setBuffer:bups offset:0 atIndex:1];
        [e0 dispatchThreads:MTLSizeMake(320,184,32) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [e0 endEncoding]; };
    auto enc_bounded=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:pso]; common(e1);
        [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12]; [e1 setBuffer:bwq offset:0 atIndex:13];
        [e1 dispatchThreadgroups:MTLSizeMake((320*184)/8,48/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; };
    auto bind_pass1=[&](id<MTLComputeCommandEncoder> e3){ common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9];
        [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16]; };
    auto enc_pass2=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pso]; common(e5);
        [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; };
    // _ind: compaction writes indirect1, pass 1 writes indirect2
    auto enc_compact_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_i]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3]; [e2 setBuffer:bdone offset:0 atIndex:4]; [e2 setBuffer:bind offset:0 atIndex:5]; [e2 setBytes:&gsz length:4 atIndex:6];
        [e2 dispatchThreadgroups:MTLSizeMake(compact_groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_i]; bind_pass1(e3); [e3 setBuffer:bdone2 offset:0 atIndex:17]; [e3 setBuffer:bind offset:0 atIndex:18]; [e3 setBuffer:bind2 offset:0 atIndex:19]; [e3 setBytes:&per_group2 length:4 atIndex:20];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto encode_i=[&](id<MTLCommandBuffer> cb){ enc_ups(cb,ups_i); enc_bounded(cb,bounded_i); enc_compact_i(cb); enc_pass1_i(cb); enc_pass2(cb,pass2_i); };
    // _wq reference: compaction + fg_indirect_from_counter in one encoder, pass 1, fg_indirect_from_counter, pass 2 (exactly b1b0_pipe_wq.mm)
    auto enc_compact_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_w]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((nvirt+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_w]; bind_pass1(e3); [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto enc_ind2_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&gsz length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; };
    auto encode_w=[&](id<MTLCommandBuffer> cb){ enc_ups(cb,ups_w); enc_bounded(cb,bounded_w); enc_compact_w(cb); enc_pass1_w(cb); enc_ind2_w(cb); enc_pass2(cb,pass2_w); };
    auto run=[&](void(^enc)(id<MTLCommandBuffer>))->double{ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; if(cb.error){printf("command buffer error: %s\n",cb.error.localizedDescription.UTF8String);exit(2);} return (cb.GPUEndTime-cb.GPUStartTime)*1000.0; };
    long ind_bad=0;
    auto check_ind=[&](const char* tag,int r,bool fused){ const uint32_t c1=*(uint32_t*)bcnt.contents, c2=*(uint32_t*)bcnt2.contents; const uint32_t* i1=(uint32_t*)bind.contents; const uint32_t* i2=(uint32_t*)bind2.contents; const uint32_t d1=*(uint32_t*)bdone.contents, d2=*(uint32_t*)bdone2.contents;
        const uint32_t e1=(c1+gsz-1)/gsz, e2=(c2+per_group2-1)/per_group2;
        bool ok = i1[0]==e1 && i1[1]==1 && i1[2]==1 && i2[0]==e2 && i2[1]==1 && i2[2]==1 && (!fused || (d1==compact_groups && d2==e1)); // _wq never touches the done counters
        if(!ok){ ++ind_bad; printf("  INDIRECT MISMATCH [%s run %d]: counter %u -> indirect1 {%u,%u,%u} expected %u; counter2 %u -> indirect2 {%u,%u,%u} expected %u; done %u/%u done2 %u/%u\n", tag, r, c1, i1[0],i1[1],i1[2], e1, c2, i2[0],i2[1],i2[2], e2, d1, compact_groups, d2, e1); } };
    double ts=1e9; for(int r=0;r<6;++r){ ts=std::min(ts, run(^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; planes(en);
        [en dispatchThreads:MTLSizeMake(320,184,48) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; })); }
    grab(ref);
    // interleaved timing: ind, wq, ind, wq ... min of 6 each; every run is checked for exactness (ind: also indirect args and done counters)
    double ti=1e9, tw=1e9; uint32_t c1=0,c2=0,c1w=0,c2w=0; long mi=0, mw=0;
    for(int r=0;r<6;++r){
        ti=std::min(ti, run(^(id<MTLCommandBuffer> cb){ encode_i(cb); })); check_ind("ind",r,true); c1=*(uint32_t*)bcnt.contents; c2=*(uint32_t*)bcnt2.contents;
        grab(got); for(size_t i=0;i<3*nplane;++i) if(got[i]!=ref[i]) ++mi;
        tw=std::min(tw, run(^(id<MTLCommandBuffer> cb){ encode_w(cb); })); check_ind("wq",r,false); c1w=*(uint32_t*)bcnt.contents; c2w=*(uint32_t*)bcnt2.contents;
        grab(got); for(size_t i=0;i<3*nplane;++i) if(got[i]!=ref[i]) ++mw; }
    // stability: N back-to-back _ind runs, every final output (three planes) byte-identical to the first (and to the oracle), indirect args right every time
    long runs_diff_first=0, runs_diff_ref=0, missed=0; uint32_t c1min=~0u,c1max=0,c2min=~0u,c2max=0;
    for(int r=0;r<stab_runs;++r){ run(^(id<MTLCommandBuffer> cb){ encode_i(cb); }); check_ind("ind-stab",r,true);
        const uint32_t k1=*(uint32_t*)bcnt.contents,k2=*(uint32_t*)bcnt2.contents; c1min=std::min(c1min,k1);c1max=std::max(c1max,k1);c2min=std::min(c2min,k2);c2max=std::max(c2max,k2);
        grab(got);
        if(r==0) first=got;
        else if(memcmp(first.data(),got.data(),3*nplane*2)!=0) ++runs_diff_first;
        if(memcmp(ref.data(),got.data(),3*nplane*2)!=0) ++runs_diff_ref;
        for(size_t i=0;i<3*nplane;++i) if(got[i]==0xA5A5 && ref[i]!=0xA5A5) { ++missed; break; } }
    const long stab_bad=runs_diff_first+runs_diff_ref+missed;
    // per-stage breakdown, separate command buffers, min of 6
    double t[5]={1e9,1e9,1e9,1e9,1e9}; double u[6]={1e9,1e9,1e9,1e9,1e9,1e9};
    for(int r=0;r<6;++r){ auto go=[&](double* slot, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; *slot=std::min(*slot,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
        reset(); go(&t[0],^(id<MTLCommandBuffer> cb){ enc_ups(cb,ups_i); }); go(&t[1],^(id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_i); }); go(&t[2],^(id<MTLCommandBuffer> cb){ enc_compact_i(cb); }); go(&t[3],^(id<MTLCommandBuffer> cb){ enc_pass1_i(cb); }); go(&t[4],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_i); });
        reset(); go(&u[0],^(id<MTLCommandBuffer> cb){ enc_ups(cb,ups_w); }); go(&u[1],^(id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_w); }); go(&u[2],^(id<MTLCommandBuffer> cb){ enc_compact_w(cb); }); go(&u[3],^(id<MTLCommandBuffer> cb){ enc_pass1_w(cb); }); go(&u[4],^(id<MTLCommandBuffer> cb){ enc_ind2_w(cb); }); go(&u[5],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_w); }); }
    printf("block1_bot0 ind [%s] %s bound: scalar %7.3f ms | ind %7.3f ms %5.2fx | wq %7.3f ms %5.2fx | ind/wq %.3f  mismatches ind %ld (over 6 runs) wq %ld (over 6 runs) of %zu (3 planes)  flagged %u (%.2f%% of %zu)  pass2 %u (%.2f%%)  [wq run: flagged %u pass2 %u]\n",
           wsrc, formal?"formal":"empirical", ts, ti, ts/ti, tw, ts/tw, ti/tw, mi, mw, 3*nplane, c1, 100.0*c1/nvirt, nvirt, c2, 100.0*c2/nvirt, c1w, c2w);
    printf("  stability (%d ind runs): runs differing from first %ld, runs differing from oracle %ld, runs with unwritten output %ld; counter range [%u,%u] counter2 range [%u,%u]; indirect/done mismatches %ld\n", stab_runs, runs_diff_first, runs_diff_ref, missed, c1min,c1max,c2min,c2max, ind_bad);
    printf("  breakdown ind (separate command buffers): upsample %.4f | bounded %.4f | compact+ind1 %.4f | pass1+ind2 %.4f | pass2 %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[4],t[0]+t[1]+t[2]+t[3]+t[4]);
    printf("  breakdown wq  (separate command buffers): upsample %.4f | bounded %.4f | compact+ind1 %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f  sum %.4f\n", u[0],u[1],u[2],u[3],u[4],u[5],u[0]+u[1]+u[2]+u[3]+u[4]+u[5]);
    printf("  result: %s\n", (mi==0 && mw==0 && ind_bad==0 && stab_bad==0)?"PASS":"FAIL");
    return (mi==0 && mw==0 && ind_bad==0 && stab_bad==0)?0:3; } }
