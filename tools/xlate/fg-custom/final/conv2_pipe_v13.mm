// conv2 family, design six + packed weight/proxy table (bounded_v13), end to end against the scalar oracle over every
// live weight pair in a token directory: bounded_v13 -> ordered compaction -> pass 1 (one exact stage) -> pass 2 (full replay).
// The half4 table (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) is built on the host from the weight buffer and verified against
// widx for every entry along the kernel's own access pattern before any launch.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
struct Params { uint32_t residual; };
static inline uint32_t widx(uint32_t tile, uint32_t stage, uint32_t ic, uint32_t oc) { uint32_t g=oc/8u, n=oc%8u, k=ic%16u; uint32_t lane=n*4u+(k%8u)/2u; uint32_t frag=(k>=8u?2u:0u)+(k&1u); return g*4608u+tile*512u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u); }
static inline uint32_t wq_index(uint32_t tile, uint32_t stage, uint32_t kb, uint32_t pair, uint32_t row) { return (((tile*4u+stage)*2u+kb)*32u+pair)*8u+row; }
static inline float bound_operand_f(float v) { float m=fabsf(v); return m!=0.0f ? std::max(m, 0x1p-14f) : 0.0f; }
static inline _Float16 q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries = 9u*4u*2u*32u*8u;   // (tile, stage, kb, oc-pair, matrix_row) -> half4
// Build the packed table and verify every entry against widx along the kernel's access pattern (group.y, g, lane, tile, stage, kb).
static void build_wq(const std::vector<_Float16>& w, std::vector<_Float16>& wq) {
    wq.assign((size_t)kWqEntries*4, (_Float16)0);
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<4; ++stage) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<32; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = stage*16u + kb*8u + row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index(tile,stage,kb,pair,row)*4];
        e[0]=w[widx(tile,stage,ic,oc)]; e[1]=w[widx(tile,stage,ic,oc+1u)]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
    // verification: kernel-side lookup vs. the v6 gather, bitwise on the weights, value-wise on the proxies; plus every weight slot gathered exactly once (bijection)
    std::vector<uint32_t> hits(8u*4608u, 0); long bad=0;
    for (uint32_t gy=0; gy<4; ++gy) for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t row=((lane>>1)&3u)+4u*((lane>>4)&1u), col=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=gy*16u+col, oc=oc0+8u*g;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<4; ++stage) for (uint32_t kb=0; kb<2; ++kb) {
            const uint32_t ic_b=stage*16u+kb*8u+row; const uint32_t i0=widx(tile,stage,ic_b,oc), i1=widx(tile,stage,ic_b,oc+1u); ++hits[i0]; ++hits[i1];
            const _Float16* e=&wq[(size_t)wq_index(tile,stage,kb,(oc0+8u*g)>>1u,row)*4]; uint16_t eb[4], wb[2]; memcpy(eb,e,8); memcpy(wb+0,&w[i0],2); memcpy(wb+1,&w[i1],2);
            if (eb[0]!=wb[0] || eb[1]!=wb[1]) ++bad;
            if ((float)e[2]!=bound_operand_f((float)w[i0]) || (float)e[3]!=bound_operand_f((float)w[i1])) ++bad;
            if (std::signbit((float)e[2]) || std::signbit((float)e[3])) ++bad; } }
    if (bad) { printf("wq table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once=0; for (uint32_t h: hits) if (h==1u) ++once; if (once!=(long)hits.size()) { printf("wq coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
}
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    // argv[1] metallib, argv[2] formal 0/1, argv[3] token dir (optional; synthetic weights otherwise)
    const uint32_t W=80,H=46,C=64; const size_t no=(size_t)W*H*C; const int formal=atoi(argv[2]);
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(8*4608), iv(no), rv(no), bv(C+8);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:rv)v=(_Float16)nd(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),iv.size()*2), br=mk(rv.data(),rv.size()*2);
    std::vector<_Float16> wqv; build_wq(wv, wqv); id<MTLBuffer> bwq=mk(wqv.data(),wqv.size()*2);
    id<MTLBuffer> bo=mkl(no*2), bm=mkl(no), brs=mkl(no), bra=mkl(no*2), bfa=mkl(no*2), blist=mkl(no*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(no*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(no), bp2=mkl(no*2), ba2=mkl(no*2);
    Params P{1u}; const uint32_t total=(uint32_t)no, gsz=64, per_group2=4;
    auto scalar=Pipe(dev,L,@"fg_custom_block0_conv2"), bounded=Pipe(dev,L,@"fg_custom_block0_conv2_bounded_v13",formal), pass1=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass1_v6"),
         pass2=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass2_v6"), compact=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:br offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; };
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bm.contents,0,no); memset(brs.contents,0xFF,no); memset(bra.contents,0,no*2); memset(bfa.contents,0,no*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    std::vector<std::pair<std::string,std::string>> pairs; if(argc>3){ for(int t=118;t<=148;t+=4){ char a[512],b[512]; snprintf(a,512,"%s/default-token-%06d.bin",argv[3],t); snprintf(b,512,"%s/default-token-%06d.bin",argv[3],t+2); pairs.push_back({a,b}); } } else pairs.push_back({"",""});
    printf("conv2 design six + packed weight/proxy table (bounded_v13; %u half4 entries, verified against widx) (2 groups/simdgroup, short fold) (two-slot, clz + vector loads in repair), %s bound\n", kWqEntries, formal?"formal":"empirical");
    double sum_s=0,sum_g=0; long total_mism=0;
    for(auto& pr: pairs){
        if(!pr.first.empty()){ auto load=[&](const std::string& path, std::vector<_Float16>& dst){ FILE* f=fopen(path.c_str(),"rb"); if(!f){printf("cannot open %s\n",path.c_str());exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s %ld bytes\n",path.c_str(),n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
            load(pr.first,wv); load(pr.second,bv); memcpy(bw.contents,wv.data(),wv.size()*2); memcpy(bb.contents,bv.data(),bv.size()*2); build_wq(wv, wqv); memcpy(bwq.contents,wqv.data(),wqv.size()*2); }
        std::vector<uint16_t> ref(no), got(no);
        double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; common(en);
            [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
        memcpy(ref.data(),bo.contents,no*2);
        double tg=1e9; uint32_t c1=0,c2=0;
        for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1); [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12]; [e1 setBuffer:bwq offset:0 atIndex:13];
            [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
            [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
            id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16];
            [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
            id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&per_group2 length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding];
            id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5); [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
            [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding];
            [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0){c1=*(uint32_t*)bcnt.contents;c2=*(uint32_t*)bcnt2.contents;} }
        if(&pr==&pairs[0]){ double t[5]={1e9,1e9,1e9,1e9,1e9}; for(int r=0;r<6;++r){ reset();
            auto go=[&](int k, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; t[k]=std::min(t[k],(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
            go(0,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1); [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12]; [e1 setBuffer:bwq offset:0 atIndex:13];
                [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; });
            go(1,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
                [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; });
            go(2,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16];
                [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; });
            go(3,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&per_group2 length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; });
            go(4,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5); [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
                [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; }); }
            printf("  breakdown (separate command buffers): bounded %.4f | compact+ind %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[4],t[0]+t[1]+t[2]+t[3]+t[4]); }
        memcpy(got.data(),bo.contents,no*2); long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m; total_mism+=m; sum_s+=ts; sum_g+=tg;
        const char* nm=pr.first.empty()?"synthetic":strrchr(pr.first.c_str(),'/')+1;
        printf("  %-26s scalar %6.3f ms | pipeline %6.3f ms  %5.2fx  mismatches %ld  flagged %u (%.2f%%)  pass2 %u (%.2f%%)\n", nm, ts, tg, ts/tg, m, c1, 100.0*c1/no, c2, 100.0*c2/no);
    }
    printf("  total scalar %.3f ms | pipeline %.3f ms  %.2fx  mismatches %ld over %zu launches\n", sum_s, sum_g, sum_s/sum_g, total_mism, pairs.size());
    return total_mism==0?0:3; } }
