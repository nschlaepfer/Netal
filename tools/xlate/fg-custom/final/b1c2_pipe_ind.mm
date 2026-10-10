// block1 conv2, design six + packed weight/proxy table (bounded_wq), _ind: the producers write their own indirect-dispatch
// arguments. bounded_wq -> fg_compact_suspects_ordered_ind (+indirect1) -> pass 1 _ind (+indirect2) -> pass 2 _ind (lane-parallel)
// versus the _wq pipeline (ordered compaction + fg_indirect_from_counter, pass 1 v6, fg_indirect_from_counter, pass 2 v6),
// kept in the same binary so the two are timed interleaved (ind, wq, ind, wq, ...; min of 6 each) over every live weight pair.
// Every ind run is checked: final output vs the scalar oracle, indirect1[0]==ceil(counter/64), indirect2[0]==ceil(counter2/4),
// done==compaction groups, done2==indirect1[0]. Done counters zeroed and indirect buffers preseeded {0,1,1} before each run.
// Stability (pair 0): N back-to-back ind runs (default 200), every final output byte-identical to the first (and to the oracle).
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] token dir ("" or absent -> synthetic weights), argv[4] stability runs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
struct Params { uint32_t residual; };
// build_wq_b1c2: as b1c2_pipe_wq.mm (packed weight/proxy table, verified against widx along the kernel's access pattern).
static inline uint32_t widx(uint32_t tile, uint32_t stage, uint32_t ic, uint32_t oc) { uint32_t g=oc/8u, n=oc%8u, k=ic%16u; uint32_t lane=n*4u+(k%8u)/2u; uint32_t frag=(k>=8u?2u:0u)+(k&1u); return g*2304u+tile*256u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u); }
static inline uint32_t wq_index(uint32_t tile, uint32_t stage, uint32_t kb, uint32_t pair, uint32_t row) { return (((tile*2u+stage)*2u+kb)*16u+pair)*8u+row; }
static inline float bound_operand_f(float v) { float m=fabsf(v); return m!=0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static inline _Float16 q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries = 9u*2u*2u*16u*8u;
static const uint32_t kWeights = 4u*2304u;
static std::vector<_Float16> build_wq_b1c2(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries*4, (_Float16)0);
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<16; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = stage*16u + kb*8u + row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index(tile,stage,kb,pair,row)*4];
        e[0]=w[widx(tile,stage,ic,oc)]; e[1]=w[widx(tile,stage,ic,oc+1u)]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
    std::vector<uint32_t> hits(kWeights, 0); long bad=0;
    for (uint32_t gy=0; gy<2; ++gy) for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t row=((lane>>1)&3u)+4u*((lane>>4)&1u), col=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=gy*16u+col, oc=oc0+8u*g;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) {
            const uint32_t ic_b=stage*16u+kb*8u+row; const uint32_t i0=widx(tile,stage,ic_b,oc), i1=widx(tile,stage,ic_b,oc+1u); ++hits[i0]; ++hits[i1];
            const _Float16* e=&wq[(size_t)wq_index(tile,stage,kb,(oc0+8u*g)>>1u,row)*4]; uint16_t eb[4], wb[2]; memcpy(eb,e,8); memcpy(wb+0,&w[i0],2); memcpy(wb+1,&w[i1],2);
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
    const uint32_t W=160,H=92,C=32; const size_t no=(size_t)W*H*C; const int formal=atoi(argv[2]);
    const bool live=argc>3 && argv[3][0]!=0; const int stab_runs=argc>4?atoi(argv[4]):200;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(kWeights), iv(no), rv(no), bv(C+8);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:rv)v=(_Float16)nd(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),iv.size()*2), br=mk(rv.data(),rv.size()*2);
    std::vector<_Float16> wqv=build_wq_b1c2(wv.data()); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);
    id<MTLBuffer> bo=mkl(no*2), bm=mkl(no), brs=mkl(no), bra=mkl(no*2), bfa=mkl(no*2), blist=mkl(no*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(no*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(no), bp2=mkl(no*2), ba2=mkl(no*2), bdone=mkl(4), bdone2=mkl(4);
    Params P{1u}; const uint32_t total=(uint32_t)no, gsz=64, per_group2=4;   // pass 2: 16 lanes per element, 4 elements per 64-thread group
    const uint32_t compact_groups=(uint32_t)((no+255)/256);
    auto scalar=Pipe(dev,L,@"fg_custom_block1_conv2"), bounded=Pipe(dev,L,@"fg_custom_block1_conv2_bounded_wq",formal),
         pass1_i=Pipe(dev,L,@"fg_custom_block1_conv2_repair_pass1_ind"), pass2_i=Pipe(dev,L,@"fg_custom_block1_conv2_repair_pass2_ind"), compact_i=Pipe(dev,L,@"fg_compact_suspects_ordered_ind"),
         pass1_w=Pipe(dev,L,@"fg_custom_block1_conv2_repair_pass1_v6"), pass2_w=Pipe(dev,L,@"fg_custom_block1_conv2_repair_pass2_v6"), compact_w=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:br offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; };
    // both pipelines' counters, done counters and indirect buffers are reset here; indirect = {0,1,1} so a zero-group dispatch still has sane y/z
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bm.contents,0,no); memset(brs.contents,0xFF,no); memset(bra.contents,0,no*2); memset(bfa.contents,0,no*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; *(uint32_t*)bdone.contents=0; *(uint32_t*)bdone2.contents=0;
        uint32_t* i1=(uint32_t*)bind.contents; uint32_t* i2=(uint32_t*)bind2.contents; i1[0]=0;i1[1]=1;i1[2]=1; i2[0]=0;i2[1]=1;i2[2]=1; };
    auto enc_bounded=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1); [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12]; [e1 setBuffer:bwq offset:0 atIndex:13];
        [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; };
    auto bind_pass1=[&](id<MTLComputeCommandEncoder> e3){ common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16]; };
    auto enc_pass2=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pass2){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5); [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; };
    // ind: compaction writes indirect1 (done at 4, indirect at 5, group_size at 6); pass 1 writes indirect2 (done2 17, indirect1 18, indirect2 19, per_group2 20)
    auto enc_compact_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_i]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3]; [e2 setBuffer:bdone offset:0 atIndex:4]; [e2 setBuffer:bind offset:0 atIndex:5]; [e2 setBytes:&gsz length:4 atIndex:6];
        [e2 dispatchThreadgroups:MTLSizeMake(compact_groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_i=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_i]; bind_pass1(e3); [e3 setBuffer:bdone2 offset:0 atIndex:17]; [e3 setBuffer:bind offset:0 atIndex:18]; [e3 setBuffer:bind2 offset:0 atIndex:19]; [e3 setBytes:&per_group2 length:4 atIndex:20];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto encode_i=[&](id<MTLCommandBuffer> cb){ enc_bounded(cb); enc_compact_i(cb); enc_pass1_i(cb); enc_pass2(cb,pass2_i); };
    // wq reference: compaction + fg_indirect_from_counter in one encoder, pass 1 v6, fg_indirect_from_counter, pass 2 v6
    auto enc_compact_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_w]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_w]; bind_pass1(e3); [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto enc_ind2_w=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&per_group2 length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; };
    auto encode_w=[&](id<MTLCommandBuffer> cb){ enc_bounded(cb); enc_compact_w(cb); enc_pass1_w(cb); enc_ind2_w(cb); enc_pass2(cb,pass2_w); };
    auto run=[&](void(^enc)(id<MTLCommandBuffer>))->double{ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; if(cb.error){printf("command buffer error: %s\n",cb.error.localizedDescription.UTF8String);exit(2);} return (cb.GPUEndTime-cb.GPUStartTime)*1000.0; };
    long ind_bad=0;
    auto check_ind=[&](const char* tag,int r,bool fused){ const uint32_t c1=*(uint32_t*)bcnt.contents, c2=*(uint32_t*)bcnt2.contents; const uint32_t* i1=(uint32_t*)bind.contents; const uint32_t* i2=(uint32_t*)bind2.contents; const uint32_t d1=*(uint32_t*)bdone.contents, d2=*(uint32_t*)bdone2.contents;
        const uint32_t e1=(c1+gsz-1)/gsz, e2=(c2+per_group2-1)/per_group2;
        bool ok = i1[0]==e1 && i1[1]==1 && i1[2]==1 && i2[0]==e2 && i2[1]==1 && i2[2]==1 && (!fused || (d1==compact_groups && d2==e1)); // wq never touches the done counters
        if(!ok){ ++ind_bad; printf("  INDIRECT MISMATCH [%s run %d]: counter %u -> indirect1 {%u,%u,%u} expected %u; counter2 %u -> indirect2 {%u,%u,%u} expected %u; done %u/%u done2 %u/%u\n", tag, r, c1, i1[0],i1[1],i1[2], e1, c2, i2[0],i2[1],i2[2], e2, d1, compact_groups, d2, e1); } };
    std::vector<std::pair<std::string,std::string>> pairs; if(live){ for(int t=166;t<=194;t+=4){ char a[512],b[512]; snprintf(a,512,"%s/default-token-%06d.bin",argv[3],t); snprintf(b,512,"%s/default-token-%06d.bin",argv[3],t+2); pairs.push_back({a,b}); } } else pairs.push_back({"",""});
    printf("block1 conv2 bounded_wq + repair _ind (pass 1 writes indirect2; compaction writes indirect1; no fg_indirect_from_counter) vs repair v6 (_wq pipeline), %s bound\n", formal?"formal":"empirical");
    double sum_s=0,sum_i=0,sum_w=0; long total_mi=0,total_mw=0, stab_bad=0;
    for(auto& pr: pairs){
        if(!pr.first.empty()){ auto load=[&](const std::string& path, std::vector<_Float16>& dst){ FILE* f=fopen(path.c_str(),"rb"); if(!f){printf("cannot open %s\n",path.c_str());exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s %ld bytes\n",path.c_str(),n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
            load(pr.first,wv); load(pr.second,bv); memcpy(bw.contents,wv.data(),wv.size()*2); memcpy(bb.contents,bv.data(),bv.size()*2); wqv=build_wq_b1c2(wv.data()); memcpy(bwq.contents,wqv.data(),wqv.size()*2); }
        std::vector<uint16_t> ref(no), got(no), gotw(no);
        double ts=1e9; for(int r=0;r<6;++r){ ts=std::min(ts, run(^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; common(en); [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; })); }
        memcpy(ref.data(),bo.contents,no*2);
        // interleaved timing: ind, wq, ind, wq ... min of 6 each; every ind run checked for exactness and indirect args
        double ti=1e9, tw=1e9; uint32_t c1=0,c2=0; long mi=0;
        for(int r=0;r<6;++r){
            ti=std::min(ti, run(^(id<MTLCommandBuffer> cb){ encode_i(cb); })); check_ind("ind",r,true); c1=*(uint32_t*)bcnt.contents; c2=*(uint32_t*)bcnt2.contents;
            memcpy(got.data(),bo.contents,no*2); for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++mi;
            tw=std::min(tw, run(^(id<MTLCommandBuffer> cb){ encode_w(cb); })); check_ind("wq",r,false); }
        memcpy(gotw.data(),bo.contents,no*2); long mw=0; for(size_t i=0;i<no;++i) if(gotw[i]!=ref[i]) ++mw;
        if(&pr==&pairs[0]){
            // stability: N back-to-back ind runs, every final output byte-identical to the first (and to the oracle), indirect args right every time
            std::vector<uint16_t> first(no); long runs_diff_first=0, runs_diff_ref=0, missed=0; uint32_t c1min=~0u,c1max=0,c2min=~0u,c2max=0;
            for(int r=0;r<stab_runs;++r){ run(^(id<MTLCommandBuffer> cb){ encode_i(cb); }); check_ind("ind-stab",r,true);
                const uint32_t k1=*(uint32_t*)bcnt.contents,k2=*(uint32_t*)bcnt2.contents; c1min=std::min(c1min,k1);c1max=std::max(c1max,k1);c2min=std::min(c2min,k2);c2max=std::max(c2max,k2);
                if(r==0) memcpy(first.data(),bo.contents,no*2);
                else if(memcmp(first.data(),bo.contents,no*2)!=0) ++runs_diff_first;
                if(memcmp(ref.data(),bo.contents,no*2)!=0) ++runs_diff_ref;
                const uint16_t* o=(const uint16_t*)bo.contents; for(size_t i=0;i<no;++i) if(o[i]==0xA5A5 && ref[i]!=0xA5A5) { ++missed; break; } }   // a missed element would leave the 0xA5A5 fill in place
            stab_bad=runs_diff_first+runs_diff_ref+missed;
            printf("  stability (%d ind runs, pair 0): runs differing from first %ld, runs differing from oracle %ld, runs with unwritten output %ld; counter range [%u,%u] counter2 range [%u,%u]; indirect mismatches so far %ld\n", stab_runs, runs_diff_first, runs_diff_ref, missed, c1min,c1max,c2min,c2max, ind_bad);
            // per-stage breakdown, separate command buffers, min of 6
            double t[4]={1e9,1e9,1e9,1e9}, u[5]={1e9,1e9,1e9,1e9,1e9};
            for(int r=0;r<6;++r){ auto go=[&](double* slot, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; *slot=std::min(*slot,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
                reset(); go(&t[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb); }); go(&t[1],^(id<MTLCommandBuffer> cb){ enc_compact_i(cb); }); go(&t[2],^(id<MTLCommandBuffer> cb){ enc_pass1_i(cb); }); go(&t[3],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_i); });
                reset(); go(&u[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb); }); go(&u[1],^(id<MTLCommandBuffer> cb){ enc_compact_w(cb); }); go(&u[2],^(id<MTLCommandBuffer> cb){ enc_pass1_w(cb); }); go(&u[3],^(id<MTLCommandBuffer> cb){ enc_ind2_w(cb); }); go(&u[4],^(id<MTLCommandBuffer> cb){ enc_pass2(cb,pass2_w); }); }
            printf("  breakdown ind (separate command buffers): bounded %.4f | compact+ind1 %.4f | pass1+ind2 %.4f | pass2 %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[0]+t[1]+t[2]+t[3]);
            printf("  breakdown wq  (separate command buffers): bounded %.4f | compact+ind1 %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f  sum %.4f\n", u[0],u[1],u[2],u[3],u[4],u[0]+u[1]+u[2]+u[3]+u[4]); }
        total_mi+=mi; total_mw+=mw; sum_s+=ts; sum_i+=ti; sum_w+=tw;
        const char* nm=pr.first.empty()?"synthetic":strrchr(pr.first.c_str(),'/')+1;
        printf("  %-26s scalar %6.3f ms | ind %6.3f ms %5.2fx | wq %6.3f ms %5.2fx | ind/wq %.4f  mismatches ind %ld (over 6 runs) wq %ld  flagged %u (%.2f%%)  pass2 %u (%.2f%%)\n", nm, ts, ti, ts/ti, tw, ts/tw, ti/tw, mi, mw, c1, 100.0*c1/no, c2, 100.0*c2/no);
    }
    printf("  total scalar %.3f ms | ind %.3f ms %.2fx | wq %.3f ms %.2fx | ind/wq %.4f  mismatches ind %ld wq %ld over %zu pairs; indirect mismatches %ld; stability failures %ld\n", sum_s, sum_i, sum_s/sum_i, sum_w, sum_s/sum_w, sum_i/sum_w, total_mi, total_mw, pairs.size(), ind_bad, stab_bad);
    return (total_mi==0 && total_mw==0 && ind_bad==0 && stab_bad==0)?0:3; } }
