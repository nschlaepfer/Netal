// block1 bottom1, design six + packed weight/proxy table, _ind: the producers write their own indirect-dispatch arguments.
//   3x upsample -> bounded_ind -> fg_compact_suspects_ordered_ind (+indirect1) -> pass 1 _ind (+indirect2) -> pass 2 _ind (lane-parallel)
// versus the _wq pipeline (ordered compaction + fg_indirect_from_counter, pass 1 _wq, fg_indirect_from_counter, pass 2 _wq),
// kept in the same binary so the two are timed interleaved (ind, wq, ind, wq, ...; min of 6 each).
// Every ind run is checked: final output vs the scalar oracle, indirect1[0]==ceil(counter/64), indirect2[0]==ceil(counter2/4),
// done==compaction groups, done2==indirect1[0]. Done counters zeroed and indirect buffers preseeded {0,1,1} before each run.
// Stability: N back-to-back ind runs (default 200), every final output byte-identical to the first (and to the oracle).
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] weights token, argv[4] bias token (optional; synthetic otherwise), argv[5] stability runs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
// build_wq_b1b1: as b1b1_pipe_wq.mm (packed weight/proxy table, verified against wi() along the kernel's access pattern).
static inline uint32_t wi(uint32_t t, uint32_t ic, uint32_t oc) { const uint32_t k=ic, lane=oc*4u+(k%8u)/2u, f=(k>=8u?2u:0u)+(k&1u); return t*128u+(f>=2u?64u:0u)+lane*2u+(f&1u); }
static inline uint32_t wq_index(uint32_t t, uint32_t kb, uint32_t pair, uint32_t row) { return ((t*2u+kb)*4u+pair)*8u+row; }
static inline float bound_operand_f(float v) { float m=fabsf(v); return m!=0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static inline _Float16 q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries = 9u*2u*4u*8u;
static const uint32_t kWeights = 9u*128u;
static std::vector<_Float16> build_wq_b1b1(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries*4, (_Float16)0);
    for (uint32_t t=0; t<9; ++t) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<4; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = kb*8u + row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index(t,kb,pair,row)*4];
        e[0]=w[wi(t,ic,oc)]; e[1]=w[wi(t,ic,oc+1u)]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
    std::vector<uint32_t> hits(kWeights, 0); long bad=0;
    for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t mr=((lane>>1)&3u)+4u*((lane>>4)&1u), mc=2u*(lane&1u)+4u*((lane>>3)&1u), oc=mc;
        for (uint32_t t=0; t<9; ++t) for (uint32_t kb=0; kb<2; ++kb) {
            const uint32_t i0=wi(t,kb*8u+mr,oc), i1=wi(t,kb*8u+mr,oc+1u); ++hits[i0]; ++hits[i1];
            const _Float16* e=&wq[(size_t)wq_index(t,kb,oc>>1u,mr)*4]; uint16_t eb[4], wb[2]; memcpy(eb,e,8); memcpy(wb+0,&w[i0],2); memcpy(wb+1,&w[i1],2);
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
    const size_t npix=(size_t)640*368, nin=(size_t)320*184*16, nups=npix*16, nvirt=npix*8;
    const size_t n0=npix*4, n1=npix, n2=npix*3, nout=npix*8;
    const int formal=atoi(argv[2]); const int stab_runs=argc>5?atoi(argv[5]):200;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(kWeights), bv(8+8), sv0(n0), sv1(n1), sv2(n2); std::vector<std::vector<_Float16>> iv(3, std::vector<_Float16>(nin));
    std::mt19937 rng(13); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto& p:iv) for(auto&v:p)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    for(auto&v:sv0)v=(_Float16)na(rng); for(auto&v:sv1)v=(_Float16)na(rng); for(auto&v:sv2)v=(_Float16)na(rng);
    const char* wsrc="synthetic";
    if(argc>4 && argv[3][0]!=0){ auto load=[&](const char* p, std::vector<_Float16>& d){ FILE* f=fopen(p,"rb"); if(!f){printf("cannot open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>d.size()*2){printf("%s %ld bytes\n",p,n);exit(1);} fread(d.data(),1,n,f); fclose(f); };
        load(argv[3],wv); load(argv[4],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2);
    const std::vector<_Float16> wqv=build_wq_b1b1(wv.data()); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);
    id<MTLBuffer> bi[3], bups[3]; for(int k=0;k<3;++k){ bi[k]=mk(iv[k].data(),nin*2); bups[k]=mkl(nups*2); }
    id<MTLBuffer> s0=mk(sv0.data(),n0*2), s1=mk(sv1.data(),n1*2), s2=mk(sv2.data(),n2*2), out=mkl(nout*2);
    id<MTLBuffer> bm=mkl(nvirt), brs=mkl(nvirt), bra=mkl(nvirt*2), bfa=mkl(nvirt*2), bs2=mkl(nvirt), bp2=mkl(nvirt*2), ba2=mkl(nvirt*2), blist=mkl(nvirt*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nvirt*4), bcnt2=mkl(4), bind2=mkl(12), bdone=mkl(4), bdone2=mkl(4);
    const uint32_t total=(uint32_t)nvirt, gsz=64, per_group2=4;   // pass 2: 16 lanes per element, 4 elements per 64-thread group
    const uint32_t compact_groups=(uint32_t)((nvirt+255)/256);
    auto scalar=Pipe(dev,L,@"fg_custom_block1_bot1");
    auto ups_i=Pipe(dev,L,@"fg_custom_block1_bot1_upsample_ind"), bounded_i=Pipe(dev,L,@"fg_custom_block1_bot1_bounded_ind",formal), pass1_i=Pipe(dev,L,@"fg_custom_block1_bot1_repair_pass1_ind"), pass2_i=Pipe(dev,L,@"fg_custom_block1_bot1_listed_ind"), compact_i=Pipe(dev,L,@"fg_compact_suspects_ordered_ind");
    auto ups_w=Pipe(dev,L,@"fg_custom_block1_bot1_upsample_wq"), bounded_w=Pipe(dev,L,@"fg_custom_block1_bot1_bounded_wq",formal), pass1_w=Pipe(dev,L,@"fg_custom_block1_bot1_repair_pass1_wq"), pass2_w=Pipe(dev,L,@"fg_custom_block1_bot1_listed_wq"), compact_w=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    // both pipelines' counters, done counters and indirect buffers are reset here; indirect = {0,1,1} so a zero-group dispatch still has sane y/z
    auto reset=[&](){ memset(out.contents,0xA5,nout*2); memset(bm.contents,0,nvirt); memset(brs.contents,0xFF,nvirt); memset(bra.contents,0,nvirt*2); memset(bfa.contents,0,nvirt*2); memset(bs2.contents,0xFF,nvirt); memset(bp2.contents,0,nvirt*2); memset(ba2.contents,0,nvirt*2);
        *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; *(uint32_t*)bdone.contents=0; *(uint32_t*)bdone2.contents=0;
        uint32_t* i1=(uint32_t*)bind.contents; uint32_t* i2=(uint32_t*)bind2.contents; i1[0]=0;i1[1]=1;i1[2]=1; i2[0]=0;i2[1]=1;i2[2]=1; };
    auto skips=[&](id<MTLComputeCommandEncoder> en, int base){ [en setBuffer:s0 offset:0 atIndex:base]; [en setBuffer:s1 offset:0 atIndex:base+1]; [en setBuffer:s2 offset:0 atIndex:base+2]; [en setBuffer:out offset:0 atIndex:base+3]; };
    auto listed_common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; for(int k=0;k<3;++k) [en setBuffer:bups[k] offset:0 atIndex:2+k]; skips(en,5); };
    auto enc_ups=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> ups){ id<MTLComputeCommandEncoder> e0=[cb computeCommandEncoder]; [e0 setComputePipelineState:ups];
        for(int k=0;k<3;++k){ [e0 setBuffer:bi[k] offset:0 atIndex:0]; [e0 setBuffer:bups[k] offset:0 atIndex:1]; [e0 dispatchThreads:MTLSizeMake(640,368,16) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; } [e0 endEncoding]; };
    auto enc_bounded=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> bounded){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; listed_common(e1);
        [e1 setBuffer:bm offset:0 atIndex:9]; [e1 setBuffer:brs offset:0 atIndex:10]; [e1 setBuffer:bra offset:0 atIndex:11]; [e1 setBuffer:bfa offset:0 atIndex:12]; [e1 setBuffer:bs2 offset:0 atIndex:13]; [e1 setBuffer:bp2 offset:0 atIndex:14]; [e1 setBuffer:ba2 offset:0 atIndex:15]; [e1 setBuffer:bwq offset:0 atIndex:16];
        [e1 dispatchThreadgroups:MTLSizeMake(npix/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; };
    auto bind_pass1=[&](id<MTLComputeCommandEncoder> e3){ listed_common(e3); [e3 setBuffer:blist offset:0 atIndex:9]; [e3 setBuffer:bcnt offset:0 atIndex:10]; [e3 setBuffer:brs offset:0 atIndex:11]; [e3 setBuffer:bra offset:0 atIndex:12];
        [e3 setBuffer:bfa offset:0 atIndex:13]; [e3 setBuffer:bm offset:0 atIndex:14]; [e3 setBuffer:blist2 offset:0 atIndex:15]; [e3 setBuffer:bcnt2 offset:0 atIndex:16]; [e3 setBuffer:bs2 offset:0 atIndex:17]; [e3 setBuffer:bp2 offset:0 atIndex:18]; [e3 setBuffer:ba2 offset:0 atIndex:19]; };
    auto enc_pass2=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pass2){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; listed_common(e5);
        [e5 setBuffer:blist2 offset:0 atIndex:9]; [e5 setBuffer:bcnt2 offset:0 atIndex:10]; [e5 setBuffer:brs offset:0 atIndex:11]; [e5 setBuffer:bra offset:0 atIndex:12];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; };
    // ind: compaction writes indirect1 (done at 4, indirect at 5, group_size at 6); pass 1 writes indirect2 (done2 20, indirect1 21, indirect2 22, per_group2 23)
    auto enc_compact_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_i]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3]; [e2 setBuffer:bdone offset:0 atIndex:4]; [e2 setBuffer:bind offset:0 atIndex:5]; [e2 setBytes:&gsz length:4 atIndex:6];
        [e2 dispatchThreadgroups:MTLSizeMake(compact_groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_i]; bind_pass1(e3); [e3 setBuffer:bdone2 offset:0 atIndex:20]; [e3 setBuffer:bind offset:0 atIndex:21]; [e3 setBuffer:bind2 offset:0 atIndex:22]; [e3 setBytes:&per_group2 length:4 atIndex:23];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto encode_i=[&](id<MTLCommandBuffer> cb){ enc_ups(cb,ups_i); enc_bounded(cb,bounded_i); enc_compact_i(cb); enc_pass1_i(cb); enc_pass2(cb,pass2_i); };
    // wq reference: compaction + fg_indirect_from_counter in one encoder, pass 1 _wq, fg_indirect_from_counter, pass 2 _wq
    auto enc_compact_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_w]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((nvirt+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_w]; bind_pass1(e3); [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto enc_ind2_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&per_group2 length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; };
    auto encode_w=[&](id<MTLCommandBuffer> cb){ enc_ups(cb,ups_w); enc_bounded(cb,bounded_w); enc_compact_w(cb); enc_pass1_w(cb); enc_ind2_w(cb); enc_pass2(cb,pass2_w); };
    auto run=[&](void(^enc)(id<MTLCommandBuffer>))->double{ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; if(cb.error){printf("command buffer error: %s\n",cb.error.localizedDescription.UTF8String);exit(2);} return (cb.GPUEndTime-cb.GPUStartTime)*1000.0; };
    long ind_bad=0;
    auto check_ind=[&](const char* tag,int r,bool fused){ const uint32_t c1=*(uint32_t*)bcnt.contents, c2=*(uint32_t*)bcnt2.contents; const uint32_t* i1=(uint32_t*)bind.contents; const uint32_t* i2=(uint32_t*)bind2.contents; const uint32_t d1=*(uint32_t*)bdone.contents, d2=*(uint32_t*)bdone2.contents;
        const uint32_t e1=(c1+gsz-1)/gsz, e2=(c2+per_group2-1)/per_group2;
        bool ok = i1[0]==e1 && i1[1]==1 && i1[2]==1 && i2[0]==e2 && i2[1]==1 && i2[2]==1 && (!fused || (d1==compact_groups && d2==e1)); // wq never touches the done counters
        if(!ok){ ++ind_bad; printf("  INDIRECT MISMATCH [%s run %d]: counter %u -> indirect1 {%u,%u,%u} expected %u; counter2 %u -> indirect2 {%u,%u,%u} expected %u; done %u/%u done2 %u/%u\n", tag, r, c1, i1[0],i1[1],i1[2], e1, c2, i2[0],i2[1],i2[2], e2, d1, compact_groups, d2, e1); } };
    std::vector<uint16_t> ref(nout), got(nout), gotw(nout), first(nout);
    auto enc_scalar=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1];
        for(int k=0;k<3;++k) [en setBuffer:bi[k] offset:0 atIndex:2+k]; skips(en,5);
        [en dispatchThreads:MTLSizeMake(640,368,8) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; };
    double ts=1e9; for(int r=0;r<6;++r){ ts=std::min(ts, run(^(id<MTLCommandBuffer> cb){ enc_scalar(cb); })); }
    memcpy(ref.data(),out.contents,nout*2);
    // interleaved timing: ind, wq, ind, wq ... min of 6 each; every ind run checked for exactness and indirect args
    double ti=1e9, tw=1e9; uint32_t c1=0,c2=0; long mi=0;
    for(int r=0;r<6;++r){
        ti=std::min(ti, run(^(id<MTLCommandBuffer> cb){ encode_i(cb); })); check_ind("ind",r,true); c1=*(uint32_t*)bcnt.contents; c2=*(uint32_t*)bcnt2.contents;
        memcpy(got.data(),out.contents,nout*2); for(size_t i=0;i<nout;++i) if(got[i]!=ref[i]) ++mi;
        tw=std::min(tw, run(^(id<MTLCommandBuffer> cb){ encode_w(cb); })); check_ind("wq",r,false); }
    memcpy(gotw.data(),out.contents,nout*2); long mw=0; for(size_t i=0;i<nout;++i) if(gotw[i]!=ref[i]) ++mw;
    printf("block1_bot1 ind vs wq [%s, %s] scalar %7.3f ms | ind %7.3f ms %5.2fx | wq %7.3f ms %5.2fx | ind/wq %.4f  mismatches ind %ld (over 6 runs) wq %ld of %zu  flagged %u (%.2f%%)  pass2 %u (%.2f%%)\n",
        wsrc, formal?"formal":"empirical", ts, ti, ts/ti, tw, ts/tw, ti/tw, mi, mw, nout, c1, 100.0*c1/nvirt, c2, 100.0*c2/nvirt);
    // stability: N back-to-back ind runs, every final output byte-identical to the first (and to the oracle), indirect args right every time
    long runs_diff_first=0, runs_diff_ref=0, missed=0; uint32_t c1min=~0u,c1max=0,c2min=~0u,c2max=0;
    for(int r=0;r<stab_runs;++r){ run(^(id<MTLCommandBuffer> cb){ encode_i(cb); }); check_ind("ind-stab",r,true);
        const uint32_t k1=*(uint32_t*)bcnt.contents,k2=*(uint32_t*)bcnt2.contents; c1min=std::min(c1min,k1);c1max=std::max(c1max,k1);c2min=std::min(c2min,k2);c2max=std::max(c2max,k2);
        memcpy(got.data(),out.contents,nout*2);
        if(r==0) first=got; else if(memcmp(first.data(),got.data(),nout*2)!=0) ++runs_diff_first;
        if(memcmp(ref.data(),got.data(),nout*2)!=0) ++runs_diff_ref;
        for(size_t i=0;i<nout;++i) if(got[i]==0xA5A5 && ref[i]!=0xA5A5) { ++missed; break; } }   // a missed element would leave the 0xA5A5 fill in place
    const long stab_bad=runs_diff_first+runs_diff_ref+missed;
    printf("  stability (%d ind runs): runs differing from first %ld, runs differing from oracle %ld, runs with unwritten output %ld; counter range [%u,%u] counter2 range [%u,%u]; indirect mismatches so far %ld\n", stab_runs, runs_diff_first, runs_diff_ref, missed, c1min,c1max,c2min,c2max, ind_bad);
    // per-stage breakdown, separate command buffers, min of 6 (the ups buffers hold the pipeline's upsampled planes)
    double t[4]={1e9,1e9,1e9,1e9}, u[5]={1e9,1e9,1e9,1e9,1e9};
    for(int r=0;r<6;++r){ auto go=[&](double* slot, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; *slot=std::min(*slot,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
        reset(); go(&t[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_i); }); go(&t[1],^(id<MTLCommandBuffer> cb){ enc_compact_i(cb); }); go(&t[2],^(id<MTLCommandBuffer> cb){ enc_pass1_i(cb); }); go(&t[3],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_i); });
        reset(); go(&u[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb,bounded_w); }); go(&u[1],^(id<MTLCommandBuffer> cb){ enc_compact_w(cb); }); go(&u[2],^(id<MTLCommandBuffer> cb){ enc_pass1_w(cb); }); go(&u[3],^(id<MTLCommandBuffer> cb){ enc_ind2_w(cb); }); go(&u[4],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_w); }); }
    printf("  breakdown ind (separate command buffers): bounded %.4f | compact+ind1 %.4f | pass1+ind2 %.4f | pass2 %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[0]+t[1]+t[2]+t[3]);
    printf("  breakdown wq  (separate command buffers): bounded %.4f | compact+ind1 %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f  sum %.4f\n", u[0],u[1],u[2],u[3],u[4],u[0]+u[1]+u[2]+u[3]+u[4]);
    printf("  result: mismatches ind %ld wq %ld; indirect mismatches %ld; stability failures %ld\n", mi, mw, ind_bad, stab_bad);
    return (mi==0 && mw==0 && ind_bad==0 && stab_bad==0)?0:3; } }
