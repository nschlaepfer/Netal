// conv2 family, second design, end to end against the scalar oracle over every live weight pair in a
// token directory: bounded_v2 -> ordered compaction -> pass 1 (one exact stage) -> pass 2 (full replay).
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
    id<MTLBuffer> bo=mkl(no*2), bm=mkl(no), brs=mkl(no), bra=mkl(no*2), bfa=mkl(no*2), blist=mkl(no*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(no*4), bcnt2=mkl(4), bind2=mkl(12), bs2=mkl(no), bp2=mkl(no*2), ba2=mkl(no*2);
    Params P{1u}; const uint32_t total=(uint32_t)no, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block0_conv2"), bounded=Pipe(dev,L,@"fg_custom_block0_conv2_bounded_v3",formal), pass1=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass1_v5"),
         pass2=Pipe(dev,L,@"fg_custom_block0_conv2_repair_pass2_v5"), compact=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; [en setBuffer:br offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; };
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bm.contents,0,no); memset(brs.contents,0xFF,no); memset(bra.contents,0,no*2); memset(bfa.contents,0,no*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    std::vector<std::pair<std::string,std::string>> pairs; if(argc>3){ for(int t=118;t<=148;t+=4){ char a[512],b[512]; snprintf(a,512,"%s/default-token-%06d.bin",argv[3],t); snprintf(b,512,"%s/default-token-%06d.bin",argv[3],t+2); pairs.push_back({a,b}); } } else pairs.push_back({"",""});
    printf("conv2 fifth design (two-slot, clz + vector loads in repair), %s bound\n", formal?"formal":"empirical");
    double sum_s=0,sum_g=0; long total_mism=0;
    for(auto& pr: pairs){
        if(!pr.first.empty()){ auto load=[&](const std::string& path, std::vector<_Float16>& dst){ FILE* f=fopen(path.c_str(),"rb"); if(!f){printf("cannot open %s\n",path.c_str());exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s %ld bytes\n",path.c_str(),n);exit(1);} fread(dst.data(),1,n,f); fclose(f); };
            load(pr.first,wv); load(pr.second,bv); memcpy(bw.contents,wv.data(),wv.size()*2); memcpy(bb.contents,bv.data(),bv.size()*2); }
        std::vector<uint16_t> ref(no), got(no);
        double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; common(en);
            [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
        memcpy(ref.data(),bo.contents,no*2);
        double tg=1e9; uint32_t c1=0,c2=0;
        for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1); [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12];
            [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
            [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
            id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16];
            [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
            id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&gsz length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding];
            id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5); [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
            [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding];
            [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0){c1=*(uint32_t*)bcnt.contents;c2=*(uint32_t*)bcnt2.contents;} }
        if(&pr==&pairs[0]){ double t[5]={1e9,1e9,1e9,1e9,1e9}; for(int r=0;r<6;++r){ reset();
            auto go=[&](int k, void(^enc)(id<MTLCommandBuffer>)){ id<MTLCommandBuffer> cb=[q commandBuffer]; enc(cb); [cb commit]; [cb waitUntilCompleted]; t[k]=std::min(t[k],(cb.GPUEndTime-cb.GPUStartTime)*1000.0); };
            go(0,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1); [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8]; [e1 setBuffer:bfa offset:0 atIndex:9]; [e1 setBuffer:bs2 offset:0 atIndex:10]; [e1 setBuffer:bp2 offset:0 atIndex:11]; [e1 setBuffer:ba2 offset:0 atIndex:12];
                [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding]; });
            go(1,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact]; [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
                [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; });
            go(2,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; common(e3); [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13]; [e3 setBuffer:bs2 offset:0 atIndex:14]; [e3 setBuffer:bp2 offset:0 atIndex:15]; [e3 setBuffer:ba2 offset:0 atIndex:16];
                [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding]; });
            go(3,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&gsz length:4 atIndex:2]; [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding]; });
            go(4,^(id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:pass2]; common(e5); [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
                [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e5 endEncoding]; }); }
            printf("  breakdown (separate command buffers): bounded %.4f | compact+ind %.4f | pass1 %.4f | ind2 %.4f | pass2 %.4f  sum %.4f\n", t[0],t[1],t[2],t[3],t[4],t[0]+t[1]+t[2]+t[3]+t[4]); }
        memcpy(got.data(),bo.contents,no*2); long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m; total_mism+=m; sum_s+=ts; sum_g+=tg;
        const char* nm=pr.first.empty()?"synthetic":strrchr(pr.first.c_str(),'/')+1;
        printf("  %-26s scalar %6.3f ms | pipeline %6.3f ms  %5.2fx  mismatches %ld  flagged %u (%.2f%%)  pass2 %u (%.2f%%)\n", nm, ts, tg, ts/tg, m, c1, 100.0*c1/no, c2, 100.0*c2/no);
    }
    printf("  total scalar %.3f ms | pipeline %.3f ms  %.2fx  mismatches %ld over %zu launches\n", sum_s, sum_g, sum_s/sum_g, total_mism, pairs.size());
    return total_mism==0?0:3; } }
