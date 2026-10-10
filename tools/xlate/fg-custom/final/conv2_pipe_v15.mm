// conv2 family, design fifteen: the producers write their own indirect-dispatch arguments.
//   bounded_v6 -> ordered compaction (+indirect1) -> pass 1 v8 (+indirect2) -> pass 2 v6
// Three dispatches fewer encoders/two fewer tiny dispatches than v9. The v9 pipeline (compaction, then
// fg_indirect_from_counter twice) is kept in the same binary so the two can be timed interleaved.
// Checks after every run: indirect1[0]==ceil(counter/64), indirect2[0]==ceil(counter2/4); final output vs scalar.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
struct Params { uint32_t residual; };
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    // argv[1] metallib, argv[2] formal 0/1, argv[3] token dir (optional; synthetic weights otherwise), argv[4] stability runs (default 200)
    const uint32_t W=80,H=46,C=64; const size_t no=(size_t)W*H*C; const int formal=atoi(argv[2]); const int stab_runs=argc>4?atoi(argv[4]):200;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(8*4608), iv(no), rv(no), bv(C+8);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:rv)v=(_Float16)nd(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),iv.size()*2), br=mk(rv.data(),rv.size()*2);
    id<MTLBuffer> bo=mkl(no*2), bm=mkl(no), brs=mkl(no), bra=mkl(no*2), bfa=mkl(no*2), blist=mkl(no*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(no*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(no), bp2=mkl(no*2), ba2=mkl(no*2), bdone=mkl(4), bdone2=mkl(4);
    Params P{1u}; const uint32_t total=(uint32_t)no, gsz=64, per_group2=4; const uint32_t compact_groups=(uint32_t)((no+255)/256);
    auto scalar=Pipe(dev,L,@"fg_custom_block0_conv2"), bounded=Pipe(dev,L,@"fg_custom_block0_conv2_bounded_v6",formal),
         pass1=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass1_v8"), pass2=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass2_v6"), compact=Pipe(dev,L,@"fg_compact_suspects_ordered_ind"),
         pass1_v6=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass1_v6"), compact_v9=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:br offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; };
    // Both pipelines' counters, done counters and indirect buffers are reset here; indirect = {0,1,1} so a zero-group pass 2 (no pass 1 groups ran) still has sane y/z.
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bm.contents,0,no); memset(brs.contents,0xFF,no); memset(bra.contents,0,no*2); memset(bfa.contents,0,no*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; *(uint32_t*)bdone.contents=0; *(uint32_t*)bdone2.contents=0;
        uint32_t* i1=(uint32_t*)bind.contents; uint32_t* i2=(uint32_t*)bind2.contents; i1[0]=0;i1[1]=1;i1[2]=1; i2[0]=0;i2[1]=1;i2[2]=1; };
    auto enc_bounded=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1); [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12];
        [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; };
    auto bind_pass1=[&](id<MTLComputeCommandEncoder> e3){ common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16]; };
    auto enc_pass2=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5); [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; };
    // v15: compaction writes indirect1, pass 1 writes indirect2
    auto enc_compact15=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3]; [e2 setBuffer:bdone offset:0 atIndex:4]; [e2 setBuffer:bind offset:0 atIndex:5]; [e2 setBytes:&gsz length:4 atIndex:6];
        [e2 dispatchThreadgroups:MTLSizeMake(compact_groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_15=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; bind_pass1(e3); [e3 setBuffer:bdone2 offset:0 atIndex:17]; [e3 setBuffer:bind offset:0 atIndex:18]; [e3 setBuffer:bind2 offset:0 atIndex:19]; [e3 setBytes:&per_group2 length:4 atIndex:20];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto encode15=[&](id<MTLCommandBuffer> cb){ enc_bounded(cb); enc_compact15(cb); enc_pass1_15(cb); enc_pass2(cb); };
    // v9 reference: compaction + fg_indirect_from_counter in one encoder, pass 1 v6, fg_indirect_from_counter, pass 2
    auto enc_compact9=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact_v9]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; };
    auto enc_pass1_9=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1_v6]; bind_pass1(e3); [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; };
    auto enc_ind2_9=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&per_group2 length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; };
    auto encode9=[&](id<MTLCommandBuffer> cb){ enc_bounded(cb); enc_compact9(cb); enc_pass1_9(cb); enc_ind2_9(cb); enc_pass2(cb); };
    auto run=[&](void(^enc)(id<MTLCommandBuffer>))->double{ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; if(cb.error){printf("command buffer error: %s\n",cb.error.localizedDescription.UTF8String);exit(2);} return (cb.GPUEndTime-cb.GPUStartTime)*1000.0; };
    // the indirect arguments the producers wrote must equal what fg_indirect_from_counter would have written
    long ind_bad=0;
    auto check_ind=[&](const char* tag,int r,bool fused){ const uint32_t c1=*(uint32_t*)bcnt.contents, c2=*(uint32_t*)bcnt2.contents; const uint32_t* i1=(uint32_t*)bind.contents; const uint32_t* i2=(uint32_t*)bind2.contents; const uint32_t d1=*(uint32_t*)bdone.contents, d2=*(uint32_t*)bdone2.contents;
        const uint32_t e1=(c1+gsz-1)/gsz, e2=(c2+per_group2-1)/per_group2;
        bool ok = i1[0]==e1 && i1[1]==1 && i1[2]==1 && i2[0]==e2 && i2[1]==1 && i2[2]==1 && (!fused || (d1==compact_groups && d2==e1)); // v9 never touches the done counters
        if(!ok){ ++ind_bad; printf("  INDIRECT MISMATCH [%s run %d]: counter %u -> indirect1 {%u,%u,%u} expected %u; counter2 %u -> indirect2 {%u,%u,%u} expected %u; done %u/%u done2 %u/%u\n", tag, r, c1, i1[0],i1[1],i1[2], e1, c2, i2[0],i2[1],i2[2], e2, d1, compact_groups, d2, e1); } };
    std::vector<std::pair<std::string,std::string>> pairs; if(argc>3){ for(int t=118;t<=148;t+=4){ char a[512],b[512]; snprintf(a,512,"%s/default-token-%06d.bin",argv[3],t); snprintf(b,512,"%s/default-token-%06d.bin",argv[3],t+2); pairs.push_back({a,b}); } } else pairs.push_back({"",""});
    printf("conv2 design fifteen (producers write indirect args; two fg_indirect_from_counter dispatches removed) vs design nine, %s bound\n", formal?"formal":"empirical");
    double sum_s=0,sum_15=0,sum_9=0; long total_mism=0, total_mism9=0; long stab_bad=0;
    for(auto& pr: pairs){
        if(!pr.first.empty()){ auto load=[&](const std::string& path, std::vector<_Float16>& dst){ FILE* f=fopen(path.c_str(),"rb"); if(!f){printf("cannot open %s\n",path.c_str());exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s %ld bytes\n",path.c_str(),n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
            load(pr.first,wv); load(pr.second,bv); memcpy(bw.contents,wv.data(),wv.size()*2); memcpy(bb.contents,bv.data(),bv.size()*2); }
        std::vector<uint16_t> ref(no), got(no), got9(no);
        double ts=1e9; for(int r=0;r<6;++r){ ts=std::min(ts, run(^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; common(en); [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; })); }
        memcpy(ref.data(),bo.contents,no*2);
        // interleaved timing: v15, v9, v15, v9 ... min of 6 each; every v15 run is checked for exactness and indirect args
        double t15=1e9, t9=1e9; uint32_t c1=0,c2=0; long m15_runs=0;
        for(int r=0;r<6;++r){
            t15=std::min(t15, run(^(id<MTLCommandBuffer> cb){ encode15(cb); })); check_ind("v15",r,true); c1=*(uint32_t*)bcnt.contents; c2=*(uint32_t*)bcnt2.contents;
            memcpy(got.data(),bo.contents,no*2); for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m15_runs;
            t9=std::min(t9, run(^(id<MTLCommandBuffer> cb){ encode9(cb); })); check_ind("v9",r,false); }
        memcpy(got9.data(),bo.contents,no*2); long m9=0; for(size_t i=0;i<no;++i) if(got9[i]!=ref[i]) ++m9;
        if(&pr==&pairs[0]){
            // stability: N back-to-back v15 runs, every final output byte-identical to the first (and to the oracle), indirect args right every time
            std::vector<uint16_t> first(no); long runs_diff_first=0, runs_diff_ref=0, missed=0; uint32_t c1min=~0u,c1max=0,c2min=~0u,c2max=0;
            for(int r=0;r<stab_runs;++r){ run(^(id<MTLCommandBuffer> cb){ encode15(cb); }); check_ind("v15-stab",r,true);
                const uint32_t k1=*(uint32_t*)bcnt.contents,k2=*(uint32_t*)bcnt2.contents; c1min=std::min(c1min,k1);c1max=std::max(c1max,k1);c2min=std::min(c2min,k2);c2max=std::max(c2max,k2);
                if(r==0) memcpy(first.data(),bo.contents,no*2);
                else if(memcmp(first.data(),bo.contents,no*2)!=0) ++runs_diff_first;
                if(memcmp(ref.data(),bo.contents,no*2)!=0) ++runs_diff_ref;
                // a missed element would leave the 0xA5A5 fill in place
                const uint16_t* o=(const uint16_t*)bo.contents; for(size_t i=0;i<no;++i) if(o[i]==0xA5A5 && ref[i]!=0xA5A5) { ++missed; break; } }
            stab_bad=runs_diff_first+runs_diff_ref+missed;
            printf("  stability (%d runs, pair 0): runs differing from first %ld, runs differing from oracle %ld, runs with unwritten output %ld; counter range [%u,%u] counter2 range [%u,%u]; indirect mismatches so far %ld\n", stab_runs, runs_diff_first, runs_diff_ref, missed, c1min,c1max,c2min,c2max, ind_bad);
            // per-stage breakdown, separate command buffers, min of 6 (v15 stages)
            double t[4]={1e9,1e9,1e9,1e9}; double u[5]={1e9,1e9,1e9,1e9,1e9};
            for(int r=0;r<6;++r){ reset();
                auto go=[&](double* slot, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; *slot=std::min(*slot,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
                go(&t[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb); }); go(&t[1],^(id<MTLCommandBuffer> cb){ enc_compact15(cb); }); go(&t[2],^(id<MTLCommandBuffer> cb){ enc_pass1_15(cb); }); go(&t[3],^(id<MTLCommandBuffer> cb){ enc_pass2(cb); });
                reset();
                go(&u[0],^(id<MTLCommandBuffer> cb){ enc_bounded(cb); }); go(&u[1],^(id<MTLCommandBuffer> cb){ enc_compact9(cb); }); go(&u[2],^(id<MTLCommandBuffer> cb){ enc_pass1_9(cb); }); go(&u[3],^(id<MTLCommandBuffer> cb){ enc_ind2_9(cb); }); go(&u[4],^(id<MTLCommandBuffer> cb){ enc_pass2(cb); }); }
            printf("  breakdown v15 (separate command buffers): bounded %.4f | compact+ind %.4f | pass1+ind2 %.4f | pass2 %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[0]+t[1]+t[2]+t[3]);
            printf("  breakdown v9  (separate command buffers): bounded %.4f | compact+ind %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f  sum %.4f\n", u[0],u[1],u[2],u[3],u[4],u[0]+u[1]+u[2]+u[3]+u[4]); }
        long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m; total_mism+=m15_runs; total_mism9+=m9; sum_s+=ts; sum_15+=t15; sum_9+=t9;
        const char* nm=pr.first.empty()?"synthetic":strrchr(pr.first.c_str(),'/')+1;
        printf("  %-26s scalar %6.3f ms | v15 %6.3f ms %5.2fx | v9 %6.3f ms %5.2fx | v15/v9 %.3f  mismatches v15 %ld (over 6 runs) v9 %ld  flagged %u (%.2f%%)  pass2 %u (%.2f%%)\n", nm, ts, t15, ts/t15, t9, ts/t9, t15/t9, m15_runs, m9, c1, 100.0*c1/no, c2, 100.0*c2/no);
    }
    printf("  total scalar %.3f ms | v15 %.3f ms %.2fx | v9 %.3f ms %.2fx | v15/v9 %.3f  mismatches v15 %ld v9 %ld over %zu pairs; indirect mismatches %ld; stability failures %ld\n", sum_s, sum_15, sum_s/sum_15, sum_9, sum_s/sum_9, sum_15/sum_9, total_mism, total_mism9, pairs.size(), ind_bad, stab_bad);
    return (total_mism==0 && ind_bad==0 && stab_bad==0)?0:3; } }
