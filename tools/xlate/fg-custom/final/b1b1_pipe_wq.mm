// block1 bottom1, design six + packed weight/proxy table: 3x upsample -> bounded_wq (ONE fused dispatch over the three
// planes, weights + proxies from a host-built half4 table, skip add inside, short fold, two shortcut slots) -> ordered
// compaction -> pass 1 (two-slot) -> indirect (4 elements / 64 threads) -> pass 2 (lane-parallel listed, 16 lanes per element).
// The table is built by build_wq_b1b1() and verified against wi() along the kernel's own access pattern before any launch.
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] weights token, argv[4] bias token (optional; synthetic otherwise),
// argv[5] "baseline" (optional) runs fg_custom_block1_bot1_bounded_v5 (no table) through the identical encoding.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
// ---------------------------------------------------------------------------------------------------------------
// build_wq_b1b1: packed weight/proxy table for block1 bottom1 (16 -> 8, 9 taps x 1 stage x 2 kb, 4 oc-pairs, 8 rows).
//   wi(t, ic, oc)                     = the kernel family's weight index (custom_block1_bot1_pipeline*.metal)
//   wq_index(t, kb, pair, row)        = ((t*2 + kb)*4 + pair)*8 + row                               (half4 units)
//   entry = (w[wi(t,ic,2p)], w[wi(t,ic,2p+1)], q(.x), q(.y)),  ic = kb*8 + row,  q(x) = x!=0 ? max(|x|,2^-14) : 0
//   size  = 9*2*4*8 = 576 half4 = 2304 halves = 4608 bytes; covers the 9*128 = 1152 weight slots exactly once.
//   For fixed (t, kb) the 32 lanes of a simdgroup (pair = matrix_column/2, row = matrix_row) read 32 consecutive half4.
// ---------------------------------------------------------------------------------------------------------------
static inline uint32_t wi(uint32_t t, uint32_t ic, uint32_t oc) { const uint32_t k=ic, lane=oc*4u+(k%8u)/2u, f=(k>=8u?2u:0u)+(k&1u); return t*128u+(f>=2u?64u:0u)+lane*2u+(f&1u); }
static inline uint32_t wq_index(uint32_t t, uint32_t kb, uint32_t pair, uint32_t row) { return ((t*2u+kb)*4u+pair)*8u+row; }
static inline float bound_operand_f(float v) { float m=fabsf(v); return m!=0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static inline _Float16 q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries = 9u*2u*4u*8u;   // (tap, kb, oc-pair, matrix_row) -> half4
static const uint32_t kWeights = 9u*128u;
static std::vector<_Float16> build_wq_b1b1(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries*4, (_Float16)0);
    for (uint32_t t=0; t<9; ++t) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<4; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = kb*8u + row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index(t,kb,pair,row)*4];
        e[0]=w[wi(t,ic,oc)]; e[1]=w[wi(t,ic,oc+1u)]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
    // verification: kernel-side lookup vs. the v5 load_b gather, bitwise on the weights, value-wise on the proxies; plus every weight slot gathered exactly once (bijection)
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
// ---------------------------------------------------------------------------------------------------------------
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    const size_t npix=(size_t)640*368, nin=(size_t)320*184*16, nups=npix*16, nvirt=npix*8;
    const size_t n0=npix*4, n1=npix, n2=npix*3, nout=npix*8;
    const int formal=atoi(argv[2]); const bool baseline=argc>5 && strcmp(argv[5],"baseline")==0;
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
    id<MTLBuffer> bm=mkl(nvirt), brs=mkl(nvirt), bra=mkl(nvirt*2), bfa=mkl(nvirt*2), bs2=mkl(nvirt), bp2=mkl(nvirt*2), ba2=mkl(nvirt*2), blist=mkl(nvirt*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nvirt*4), bcnt2=mkl(4), bind2=mkl(12);
    const uint32_t total=(uint32_t)nvirt, gsz=64, per_group2=4;   // pass 2: 16 lanes per element, 4 elements per 64-thread group
    auto scalar=Pipe(dev,L,@"fg_custom_block1_bot1"), ups=Pipe(dev,L,@"fg_custom_block1_bot1_upsample_wq"),
         bounded=Pipe(dev,L,baseline?@"fg_custom_block1_bot1_bounded_v5":@"fg_custom_block1_bot1_bounded_wq",formal),
         pass1=Pipe(dev,L,@"fg_custom_block1_bot1_repair_pass1_wq"), pass2=Pipe(dev,L,@"fg_custom_block1_bot1_listed_wq"), compact=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto reset=[&](){ memset(out.contents,0xA5,nout*2); memset(bm.contents,0,nvirt); memset(brs.contents,0xFF,nvirt); memset(bra.contents,0,nvirt*2); memset(bfa.contents,0,nvirt*2); memset(bs2.contents,0xFF,nvirt); memset(bp2.contents,0,nvirt*2); memset(ba2.contents,0,nvirt*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    auto skips=[&](id<MTLComputeCommandEncoder> en, int base){ [en setBuffer:s0 offset:0 atIndex:base]; [en setBuffer:s1 offset:0 atIndex:base+1]; [en setBuffer:s2 offset:0 atIndex:base+2]; [en setBuffer:out offset:0 atIndex:base+3]; };
    // listed-layout buffers 0..8 shared by the fused bounded kernel, pass 1 and pass 2
    auto listed_common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; for(int k=0;k<3;++k) [en setBuffer:bups[k] offset:0 atIndex:2+k]; skips(en,5); };
    auto enc_bounded=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; listed_common(e1);
        [e1 setBuffer:bm offset:0 atIndex:9]; [e1 setBuffer:brs offset:0 atIndex:10]; [e1 setBuffer:bra offset:0 atIndex:11]; [e1 setBuffer:bfa offset:0 atIndex:12]; [e1 setBuffer:bs2 offset:0 atIndex:13]; [e1 setBuffer:bp2 offset:0 atIndex:14]; [e1 setBuffer:ba2 offset:0 atIndex:15];
        if(!baseline) [e1 setBuffer:bwq offset:0 atIndex:16];
        [e1 dispatchThreadgroups:MTLSizeMake(npix/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; };
    std::vector<uint16_t> ref(nout), got(nout);
    double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1];
        for(int k=0;k<3;++k) [en setBuffer:bi[k] offset:0 atIndex:2+k]; skips(en,5);
        [en dispatchThreads:MTLSizeMake(640,368,8) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    memcpy(ref.data(),out.contents,nout*2);
    double tg=1e9; uint32_t c1=0,c2=0;
    for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e0=[cb computeCommandEncoder]; [e0 setComputePipelineState:ups];
        for(int k=0;k<3;++k){ [e0 setBuffer:bi[k] offset:0 atIndex:0]; [e0 setBuffer:bups[k] offset:0 atIndex:1]; [e0 dispatchThreads:MTLSizeMake(640,368,16) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; }
        [e0 endEncoding];
        enc_bounded(cb);
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(((nvirt+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; listed_common(e3);
        [e3 setBuffer:blist offset:0 atIndex:9]; [e3 setBuffer:bcnt offset:0 atIndex:10]; [e3 setBuffer:brs offset:0 atIndex:11]; [e3 setBuffer:bra offset:0 atIndex:12];
        [e3 setBuffer:bfa offset:0 atIndex:13]; [e3 setBuffer:bm offset:0 atIndex:14]; [e3 setBuffer:blist2 offset:0 atIndex:15]; [e3 setBuffer:bcnt2 offset:0 atIndex:16];
        [e3 setBuffer:bs2 offset:0 atIndex:17]; [e3 setBuffer:bp2 offset:0 atIndex:18]; [e3 setBuffer:ba2 offset:0 atIndex:19];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
        id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&per_group2 length:4 atIndex:2];
        [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding];
        id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; listed_common(e5);
        [e5 setBuffer:blist2 offset:0 atIndex:9]; [e5 setBuffer:bcnt2 offset:0 atIndex:10]; [e5 setBuffer:brs offset:0 atIndex:11]; [e5 setBuffer:bra offset:0 atIndex:12];
        [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding];
        [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0){c1=*(uint32_t*)bcnt.contents;c2=*(uint32_t*)bcnt2.contents;} }
    memcpy(got.data(),out.contents,nout*2); long m=0; for(size_t i=0;i<nout;++i) if(got[i]!=ref[i]) ++m;
    // bounded stage alone (separate command buffer, min of 6; the ups buffers still hold the pipeline's upsampled planes)
    double tb=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; enc_bounded(cb); [cb commit]; [cb waitUntilCompleted]; tb=std::min(tb,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    printf("block1_bot1 %s [%s, %s] scalar %7.3f ms | pipeline %7.3f ms  %5.2fx  mismatches %ld of %zu  flagged %u (%.2f%%)  pass2 %u (%.2f%%)  | bounded alone %7.4f ms | table %u half4 = %u B\n", baseline?"BASELINE bounded_v5":"bounded_wq", wsrc, formal?"formal":"empirical", ts, tg, ts/tg, m, nout, c1, 100.0*c1/nvirt, c2, 100.0*c2/nvirt, tb, kWqEntries, kWqEntries*8u);
    return m==0?0:3; } }
