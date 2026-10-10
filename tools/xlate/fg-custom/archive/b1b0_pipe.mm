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
    const size_t nin=(size_t)160*92*32, nups=(size_t)320*184*32, nplane=(size_t)320*184*16, nvirt=(size_t)320*184*48;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(6*2304), iv(nin), bv(48+8);
    std::mt19937 rng(13); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    const char* wsrc="synthetic";
    if(argc>3){ auto load=[&](const char* p, std::vector<_Float16>& d){ FILE* f=fopen(p,"rb"); if(!f){printf("cannot open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>d.size()*2){printf("%s %ld bytes\n",p,n);exit(1);} fread(d.data(),1,n,f); fclose(f); };
        load(argv[2],wv); load(argv[3],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),nin*2), bups=mkl(nups*2);
    id<MTLBuffer> p0=mkl(nplane*2), p1=mkl(nplane*2), p2=mkl(nplane*2), bm=mkl(nvirt), brs=mkl(nvirt), bra=mkl(nvirt*2), blist=mkl(nvirt*4), bcnt=mkl(4), bind=mkl(12);
    const uint32_t total=(uint32_t)nvirt, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block1_bot0"), ups=Pipe(dev,L,@"fg_custom_block1_bot0_upsample"), bounded=Pipe(dev,L,@"fg_custom_block1_bot0_bounded"),
         listed=Pipe(dev,L,@"fg_custom_block1_bot0_listed"), compact=Pipe(dev,L,@"fg_compact_suspects"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto reset=[&](){ for(auto b:{p0,p1,p2}) memset(b.contents,0xA5,nplane*2); memset(bm.contents,0,nvirt); memset(brs.contents,0xFF,nvirt); memset(bra.contents,0,nvirt*2); *(uint32_t*)bcnt.contents=0; };
    auto planes=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:p0 offset:0 atIndex:3]; [en setBuffer:p1 offset:0 atIndex:4]; [en setBuffer:p2 offset:0 atIndex:5]; };
    std::vector<uint16_t> ref(3*nplane), got(3*nplane);
    auto grab=[&](std::vector<uint16_t>& v){ memcpy(v.data(),p0.contents,nplane*2); memcpy(v.data()+nplane,p1.contents,nplane*2); memcpy(v.data()+2*nplane,p2.contents,nplane*2); };
    double ts=1e9; for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:scalar]; [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2]; planes(en);
        [en dispatchThreads:MTLSizeMake(320,184,48) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    grab(ref);
    double tg=1e9; uint32_t count=0;
    for(int r=0;r<6;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e0=[cb computeCommandEncoder]; [e0 setComputePipelineState:ups]; [e0 setBuffer:bi offset:0 atIndex:0]; [e0 setBuffer:bups offset:0 atIndex:1];
        [e0 dispatchThreads:MTLSizeMake(320,184,32) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [e0 endEncoding];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded];
        [e1 setBuffer:bw offset:0 atIndex:0]; [e1 setBuffer:bb offset:0 atIndex:1]; [e1 setBuffer:bups offset:0 atIndex:2]; planes(e1);
        [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8];
        [e1 dispatchThreadgroups:MTLSizeMake((320*184)/8,48/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(nvirt,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:listed];
        [e3 setBuffer:bw offset:0 atIndex:0]; [e3 setBuffer:bb offset:0 atIndex:1]; [e3 setBuffer:bups offset:0 atIndex:2]; planes(e3);
        [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
        [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); if(r==0) count=*(uint32_t*)bcnt.contents; }
    grab(got); long m=0; for(size_t i=0;i<3*nplane;++i) if(got[i]!=ref[i]) ++m;
    printf("block1_bot0 [%s] scalar %7.3f ms | upsample+guarded %7.3f ms  %5.2fx  %ld mismatching of %zu (3 planes)  %u flagged (%.2f%%)\n", wsrc, ts, tg, ts/tg, m, 3*nplane, count, 100.0*count/nvirt);
    return m==0?0:3; } }
