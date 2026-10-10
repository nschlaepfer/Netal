// block1 conv1, design thirteen (_wq) with the producers writing their own indirect-dispatch arguments (_ind):
//   bounded_ind -> fg_compact_suspects_ordered_ind (+indirect1) -> pass 1 _ind (+indirect2) -> pass 2 _ind -> repool _ind
// The _wq pipeline (ordered compaction + fg_indirect_from_counter, pass 1 _wq, fg_indirect_from_counter, pass 2 _wq, repool _wq) is kept in
// the same binary and timed interleaved (ind, wq, ind, wq, ...), min of 6 each. Every _ind run is checked for exactness against the scalar
// oracle and for indirect1[0]==ceil(counter/64), indirect2[0]==ceil(counter2/64), done==compaction groups, done2==indirect1[0]; then N
// back-to-back _ind runs (default 200) must all be byte-identical to the first and to the oracle.
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
    if(argc<3){printf("usage: %s <metallib> <formal 0/1> [weights.bin bias.bin [stability_runs]]\n",argv[0]);return 2;}
    const uint32_t IW=320,IH=184,CI=16,OW=160,OH=92,CO=32; const size_t ni=(size_t)IW*IH*CI, nm=(size_t)IW*IH*CO, no=(size_t)OW*OH*CO; const int formal=atoi(argv[2]); const int stab_runs=argc>5?atoi(argv[5]):200;
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
    id<MTLBuffer> bo=mkl(no*2), bint=mkl(nm*2), bm=mkl(nm), brs=mkl(nm), bra=mkl(nm*2), bfa=mkl(nm*2), blist=mkl(nm*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nm*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(nm), bp2=mkl(nm*2), ba2=mkl(nm*2), bdone=mkl(4), bdone2=mkl(4);
    const uint32_t total=(uint32_t)nm, gsz=64, per_group2=64; const uint32_t compact_groups=(uint32_t)((nm+255)/256);
    auto scalar=Pipe(dev,L,@"fg_custom_block1_conv1");
    auto bounded_i=Pipe(dev,L,@"fg_custom_block1_conv1_bounded_ind",formal), pass1_i=Pipe(dev,L,@"fg_custom_block1_conv1_repair_pass1_ind"), pass2_i=Pipe(dev,L,@"fg_custom_block1_conv1_listed_ind"), repool_i=Pipe(dev,L,@"fg_custom_block1_conv1_repool_ind"), compact_i=Pipe(dev,L,@"fg_compact_suspects_ordered_ind");
    auto bounded_w=Pipe(dev,L,@"fg_custom_block1_conv1_bounded_wq",formal), pass1_w=Pipe(dev,L,@"fg_custom_block1_conv1_repair_pass1_wq"), pass2_w=Pipe(dev,L,@"fg_custom_block1_conv1_listed_wq"), repool_w=Pipe(dev,L,@"fg_custom_block1_conv1_repool_wq"), compact_w=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:bint offset:0 atIndex:3]; };
    // Both pipelines' counters, done counters and indirect buffers are reset here; indirect = {0,1,1} so a zero-group pass 2 (no pass 1 groups ran) still has sane y/z.
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bint.contents,0xA5,nm*2); memset(bm.contents,0,nm); memset(brs.contents,0xFF,nm); memset(bra.contents,0,nm*2); memset(bfa.contents,0,nm*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; *(uint32_t*)bdone.contents=0; *(uint32_t*)bdone2.contents=0;
        uint32_t* i1=(uint32_t*)bind.contents; uint32_t* i2=(uint32_t*)bind2.contents; i1[0]=0;i1[1]=1;i1[2]=1; i2[0]=0;i2[1]=1;i2[2]=1; };
    auto enc_bounded=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:pso]; common(e1);
        [e1 setBuffer:bm offset:0 atIndex:4]; [e1 setBuffer:brs offset:0 atIndex:5]; [e1 setBuffer:bra offset:0 atIndex:6]; [e1 setBuffer:bfa offset:0 atIndex:7]; [e1 setBuffer:bs2 offset:0 atIndex:8]; [e1 setBuffer:bp2 offset:0 atIndex:9]; [e1 setBuffer:ba2 offset:0 atIndex:10]; [e1 setBuffer:bo offset:0 atIndex:11]; [e1 setBuffer:bwq offset:0 atIndex:12];
        [e1 dispatchThreadgroups:MTLSizeMake((IW*IH)/8,CO/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; };
    auto bind_pass1=[&](id<MTLComputeCommandEncoder> e3){ common(e3); [e3 setBuffer:blist offset:0 atIndex:4]; [e3 setBuffer:bcnt offset:0 atIndex:5]; [e3 setBuffer:brs offset:0 atIndex:6]; [e3 setBuffer:bra offset:0 atIndex:7];
        [e3 setBuffer:bfa offset:0 atIndex:8]; [e3 setBuffer:bm offset:0 atIndex:9]; [e3 setBuffer:blist2 offset:0 atIndex:10]; [e3 setBuffer:bcnt2 offset:0 atIndex:11]; [e3 setBuffer:bs2 offset:0 atIndex:12]; [e3 setBuffer:bp2 offset:0 atIndex:13]; [e3 setBuffer:ba2 offset:0 atIndex:14]; };
    auto enc_pass2=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pso]; common(e5);
        [e5 setBuffer:blist2 offset:0 atIndex:4]; [e5 setBuffer:bcnt2 offset:0 atIndex:5]; [e5 setBuffer:brs offset:0 atIndex:6]; [e5 setBuffer:bra offset:0 atIndex:7];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; };
    auto enc_repool=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){ id<MTLComputeCommandEncoder> e6=[cb computeCommandEncoder]; [e6 setComputePipelineState:pso]; [e6 setBuffer:bint offset:0 atIndex:0]; [e6 setBuffer:bo offset:0 atIndex:1]; [e6 setBuffer:blist2 offset:0 atIndex:2]; [e6 setBuffer:bcnt2 offset:0 atIndex:3];
        [e6 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e6 endEncoding]; };
    // _ind: compaction writes indirect1, pass 1 writes indirect2
    auto enc_compact_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_i]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3]; [e2 setBuffer:bdone offset:0 atIndex:4]; [e2 setBuffer:bind offset:0 atIndex:5]; [e2 setBytes:&gsz length:4 atIndex:6];
        [e2 dispatchThreadgroups:MTLSizeMake(compact_groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_i]; bind_pass1(e3); [e3 setBuffer:bdone2 offset:0 atIndex:15]; [e3 setBuffer:bind offset:0 atIndex:16]; [e3 setBuffer:bind2 offset:0 atIndex:17]; [e3 setBytes:&per_group2 length:4 atIndex:18];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto encode_i=[&](id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_i); enc_compact_i(cb); enc_pass1_i(cb); enc_pass2(cb,pass2_i); enc_repool(cb,repool_i); };
    // _wq reference: compaction + fg_indirect_from_counter in one encoder, pass 1, fg_indirect_from_counter, pass 2, repool (exactly b1c1_pipe_wq.mm)
    auto enc_compact_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_w]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((nm+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_w]; bind_pass1(e3); [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto enc_ind2_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&gsz length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; };
    auto encode_w=[&](id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_w); enc_compact_w(cb); enc_pass1_w(cb); enc_ind2_w(cb); enc_pass2(cb,pass2_w); enc_repool(cb,repool_w); };
    auto run=[&](void(^enc)(id<MTLCommandBuffer>))->double{ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; if(cb.error){printf("command buffer error: %s\n",cb.error.localizedDescription.UTF8String);exit(2);} return (cb.GPUEndTime-cb.GPUStartTime)*1000.0; };
    // the indirect arguments the producers wrote must equal what fg_indirect_from_counter would have written
    long ind_bad=0;
    auto check_ind=[&](const char* tag,int r,bool fused){ const uint32_t c1=*(uint32_t*)bcnt.contents, c2=*(uint32_t*)bcnt2.contents; const uint32_t* i1=(uint32_t*)bind.contents; const uint32_t* i2=(uint32_t*)bind2.contents; const uint32_t d1=*(uint32_t*)bdone.contents, d2=*(uint32_t*)bdone2.contents;
        const uint32_t e1=(c1+gsz-1)/gsz, e2=(c2+per_group2-1)/per_group2;
        bool ok = i1[0]==e1 && i1[1]==1 && i1[2]==1 && i2[0]==e2 && i2[1]==1 && i2[2]==1 && (!fused || (d1==compact_groups && d2==e1)); // _wq never touches the done counters
        if(!ok){ ++ind_bad; printf("  INDIRECT MISMATCH [%s run %d]: counter %u -> indirect1 {%u,%u,%u} expected %u; counter2 %u -> indirect2 {%u,%u,%u} expected %u; done %u/%u done2 %u/%u\n", tag, r, c1, i1[0],i1[1],i1[2], e1, c2, i2[0],i2[1],i2[2], e2, d1, compact_groups, d2, e1); } };
    std::vector<uint16_t> ref(no), got(no);
    double ts=1e9; for(int r=0;r<6;++r){ ts=std::min(ts, run(^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:bo offset:0 atIndex:3];
        [en dispatchThreads:MTLSizeMake(OW,OH,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; })); }
    memcpy(ref.data(),bo.contents,no*2);
    // interleaved timing: ind, wq, ind, wq ... min of 6 each; every run is checked for exactness (ind: also indirect args and done counters)
    double ti=1e9, tw=1e9; uint32_t c1=0,c2=0,c1w=0,c2w=0; long mi=0, mw=0;
    for(int r=0;r<6;++r){
        ti=std::min(ti, run(^(id<MTLCommandBuffer> cb){ encode_i(cb); })); check_ind("ind",r,true); c1=*(uint32_t*)bcnt.contents; c2=*(uint32_t*)bcnt2.contents;
        memcpy(got.data(),bo.contents,no*2); for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++mi;
        tw=std::min(tw, run(^(id<MTLCommandBuffer> cb){ encode_w(cb); })); check_ind("wq",r,false); c1w=*(uint32_t*)bcnt.contents; c2w=*(uint32_t*)bcnt2.contents;
        memcpy(got.data(),bo.contents,no*2); for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++mw; }
    // stability: N back-to-back _ind runs, every final output byte-identical to the first (and to the oracle), indirect args right every time
    std::vector<uint16_t> first(no); long runs_diff_first=0, runs_diff_ref=0, missed=0; uint32_t c1min=~0u,c1max=0,c2min=~0u,c2max=0;
    for(int r=0;r<stab_runs;++r){ run(^(id<MTLCommandBuffer> cb){ encode_i(cb); }); check_ind("ind-stab",r,true);
        const uint32_t k1=*(uint32_t*)bcnt.contents,k2=*(uint32_t*)bcnt2.contents; c1min=std::min(c1min,k1);c1max=std::max(c1max,k1);c2min=std::min(c2min,k2);c2max=std::max(c2max,k2);
        if(r==0) memcpy(first.data(),bo.contents,no*2);
        else if(memcmp(first.data(),bo.contents,no*2)!=0) ++runs_diff_first;
        if(memcmp(ref.data(),bo.contents,no*2)!=0) ++runs_diff_ref;
        // a missed element would leave the 0xA5A5 fill in place
        const uint16_t* o=(const uint16_t*)bo.contents; for(size_t i=0;i<no;++i) if(o[i]==0xA5A5 && ref[i]!=0xA5A5) { ++missed; break; } }
    const long stab_bad=runs_diff_first+runs_diff_ref+missed;
    // per-stage breakdown, separate command buffers, min of 6
    double t[5]={1e9,1e9,1e9,1e9,1e9}; double u[6]={1e9,1e9,1e9,1e9,1e9,1e9};
    for(int r=0;r<6;++r){ auto go=[&](double* slot, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; *slot=std::min(*slot,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
        reset(); go(&t[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_i); }); go(&t[1],^(id<MTLCommandBuffer> cb){ enc_compact_i(cb); }); go(&t[2],^(id<MTLCommandBuffer> cb){ enc_pass1_i(cb); }); go(&t[3],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_i); }); go(&t[4],^(id<MTLCommandBuffer> cb){ enc_repool(cb,repool_i); });
        reset(); go(&u[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_w); }); go(&u[1],^(id<MTLCommandBuffer> cb){ enc_compact_w(cb); }); go(&u[2],^(id<MTLCommandBuffer> cb){ enc_pass1_w(cb); }); go(&u[3],^(id<MTLCommandBuffer> cb){ enc_ind2_w(cb); }); go(&u[4],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_w); }); go(&u[5],^(id<MTLCommandBuffer> cb){ enc_repool(cb,repool_w); }); }
    printf("block1_conv1 ind [%s] %s bound: scalar %7.3f ms | ind %7.3f ms %5.2fx | wq %7.3f ms %5.2fx | ind/wq %.3f  mismatches ind %ld (over 6 runs) wq %ld (over 6 runs) of %zu  flagged %u (%.2f%% of %zu intermediates)  pass2 %u (%.2f%%)  [wq run: flagged %u pass2 %u]\n",
           wsrc, formal?"formal":"empirical", ts, ti, ts/ti, tw, ts/tw, ti/tw, mi, mw, no, c1, 100.0*c1/nm, nm, c2, 100.0*c2/nm, c1w, c2w);
    printf("  stability (%d ind runs): runs differing from first %ld, runs differing from oracle %ld, runs with unwritten output %ld; counter range [%u,%u] counter2 range [%u,%u]; indirect/done mismatches %ld\n", stab_runs, runs_diff_first, runs_diff_ref, missed, c1min,c1max,c2min,c2max, ind_bad);
    printf("  breakdown ind (separate command buffers): bounded %.4f | compact+ind1 %.4f | pass1+ind2 %.4f | pass2 %.4f | repool %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[4],t[0]+t[1]+t[2]+t[3]+t[4]);
    printf("  breakdown wq  (separate command buffers): bounded %.4f | compact+ind1 %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f | repool %.4f  sum %.4f\n", u[0],u[1],u[2],u[3],u[4],u[5],u[0]+u[1]+u[2]+u[3]+u[4]+u[5]);
    printf("  result: %s\n", (mi==0 && mw==0 && ind_bad==0 && stab_bad==0)?"PASS":"FAIL");
    return (mi==0 && mw==0 && ind_bad==0 && stab_bad==0)?0:3; } }
