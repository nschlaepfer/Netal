#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn) {
    NSError* e=nil; id<MTLFunction> F=[L newFunctionWithName:fn]; if(!F){printf("missing %s\n",fn.UTF8String);exit(1);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    const size_t npix=(size_t)320*184, nin=(size_t)160*92*32, nups=npix*32, nvirt=npix*8;
    const size_t n0=npix*4, n1=npix, n2=npix*3;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(9*256), bv(8+8); std::vector<std::vector<_Float16>> iv(3, std::vector<_Float16>(nin));
    std::mt19937 rng(13); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto& p:iv) for(auto&v:p)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    const char* wsrc="synthetic";
    if(argc>3){ auto load=[&](const char* p, std::vector<_Float16>& d){ FILE* f=fopen(p,"rb"); if(!f){printf("cannot open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>d.size()*2){printf("%s %ld bytes\n",p,n);exit(1);} fread(d.data(),1,n,f); fclose(f); };
        load(argv[2],wv); load(argv[3],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2);
    id<MTLBuffer> bi[3], bups[3]; for(int k=0;k<3;++k){ bi[k]=mk(iv[k].data(),nin*2); bups[k]=mkl(nups*2); }
    id<MTLBuffer> o0=mkl(n0*2), o1=mkl(n1*2), o2=mkl(n2*2), bm=mkl(nvirt), brs=mkl(nvirt), bra=mkl(nvirt*2), blist=mkl(nvirt*4), bcnt=mkl(4), bind=mkl(12);
    const uint32_t total=(uint32_t)nvirt, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block0_bot1"), ups=Pipe(dev,L,@"fg_custom_block0_bot1_upsample"), bounded=Pipe(dev,L,@"fg_custom_block0_bot1_bounded"),
         listed=Pipe(dev,L,@"fg_custom_block0_bot1_listed"), compact=Pipe(dev,L,@"fg_compact_suspects"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto reset=[&](){ memset(o0.contents,0xA5,n0*2); memset(o1.contents,0xA5,n1*2); memset(o2.contents,0xA5,n2*2); memset(bm.contents,0,nvirt); memset(brs.contents,0xFF,nvirt); memset(bra.contents,0,nvirt*2); *(uint32_t*)bcnt.contents=0; };
    std::vector<uint16_t> ref(n0+n1+n2), got(n0+n1+n2);
    auto grab=[&](std::vector<uint16_t>& v){ memcpy(v.data(),o0.contents,n0*2); memcpy(v.data()+n0,o1.contents,n1*2); memcpy(v.data()+n0+n1,o2.contents,n2*2); };
    double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1];
        for(int k=0;k<3;++k) [en setBuffer:bi[k] offset:0 atIndex:2+k]; [en setBuffer:o0 offset:0 atIndex:5]; [en setBuffer:o1 offset:0 atIndex:6]; [en setBuffer:o2 offset:0 atIndex:7];
        [en dispatchThreads:MTLSizeMake(320,184,8) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    grab(ref);
    double tg=1e9; uint32_t count=0;
    for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e0=[cb computeCommandEncoder]; [e0 setComputePipelineState:ups];
        for(int k=0;k<3;++k){ [e0 setBuffer:bi[k] offset:0 atIndex:0]; [e0 setBuffer:bups[k] offset:0 atIndex:1]; [e0 dispatchThreads:MTLSizeMake(320,184,32) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; }
        [e0 endEncoding];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded];
        [e1 setBuffer:bw offset:0 atIndex:0]; [e1 setBuffer:bb offset:0 atIndex:1]; [e1 setBuffer:o0 offset:0 atIndex:3]; [e1 setBuffer:o1 offset:0 atIndex:4]; [e1 setBuffer:o2 offset:0 atIndex:5];
        [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8];
        for(uint32_t plane=0;plane<3;++plane){ [e1 setBuffer:bups[plane] offset:0 atIndex:2]; [e1 setBytes:&plane length:4 atIndex:9];
            [e1 dispatchThreadgroups:MTLSizeMake(npix/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; }
        [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(nvirt,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:listed];
        [e3 setBuffer:bw offset:0 atIndex:0]; [e3 setBuffer:bb offset:0 atIndex:1]; for(int k=0;k<3;++k) [e3 setBuffer:bups[k] offset:0 atIndex:2+k];
        [e3 setBuffer:o0 offset:0 atIndex:5]; [e3 setBuffer:o1 offset:0 atIndex:6]; [e3 setBuffer:o2 offset:0 atIndex:7];
        [e3 setBuffer:blist offset:0 atIndex:8]; [e3 setBuffer:bcnt offset:0 atIndex:9]; [e3 setBuffer:brs offset:0 atIndex:10]; [e3 setBuffer:bra offset:0 atIndex:11];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
        [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0) count=*(uint32_t*)bcnt.contents; }
    grab(got); long m=0; for(size_t i=0;i<ref.size();++i) if(got[i]!=ref[i]) ++m;
    printf("block0_bot1 [%s] scalar %7.3f ms | 3x upsample + 3-pass guarded %7.3f ms  %5.2fx  %ld mismatching of %zu (3 outputs)  %u flagged (%.2f%%)\n", wsrc, ts, tg, ts/tg, m, ref.size(), count, 100.0*count/nvirt);
    return m==0?0:3; } }
