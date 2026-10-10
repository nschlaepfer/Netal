// block1 bottom1, design six: 3x upsample -> bounded_v4 (ONE fused dispatch over the three planes, skip add inside,
// short fold, two shortcut slots) -> ordered compaction -> pass 1 (two-slot) -> indirect (4 elements / 64 threads) -> pass 2 (lane-parallel listed, 16 lanes per element).
// argv[1] metallib, argv[2] FORMAL 0/1, argv[3] weights token, argv[4] bias token (optional; synthetic otherwise),
// argv[5] "planar" (optional) runs bounded_v4p three times with a plane constant instead of the fused kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int formal=-1) {
    NSError* e=nil; id<MTLFunction> F; if(formal<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool f=formal; [cv setConstantValue:&f type:MTLDataTypeBool atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);} id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    const size_t npix=(size_t)640*368, nin=(size_t)320*184*16, nups=npix*16, nvirt=npix*8;
    const size_t n0=npix*4, n1=npix, n2=npix*3, nout=npix*8;
    const int formal=atoi(argv[2]); const bool planar=argc>5 && strcmp(argv[5],"planar")==0;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(9*128), bv(8+8), sv0(n0), sv1(n1), sv2(n2); std::vector<std::vector<_Float16>> iv(3, std::vector<_Float16>(nin));
    std::mt19937 rng(13); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto& p:iv) for(auto&v:p)v=(_Float16)na(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    for(auto&v:sv0)v=(_Float16)na(rng); for(auto&v:sv1)v=(_Float16)na(rng); for(auto&v:sv2)v=(_Float16)na(rng);
    const char* wsrc="synthetic";
    if(argc>4 && argv[3][0]!=0){ auto load=[&](const char* p, std::vector<_Float16>& d){ FILE* f=fopen(p,"rb"); if(!f){printf("cannot open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>d.size()*2){printf("%s %ld bytes\n",p,n);exit(1);} fread(d.data(),1,n,f); fclose(f); };
        load(argv[3],wv); load(argv[4],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2);
    id<MTLBuffer> bi[3], bups[3]; for(int k=0;k<3;++k){ bi[k]=mk(iv[k].data(),nin*2); bups[k]=mkl(nups*2); }
    id<MTLBuffer> s0=mk(sv0.data(),n0*2), s1=mk(sv1.data(),n1*2), s2=mk(sv2.data(),n2*2), out=mkl(nout*2);
    id<MTLBuffer> bm=mkl(nvirt), brs=mkl(nvirt), bra=mkl(nvirt*2), bfa=mkl(nvirt*2), bs2=mkl(nvirt), bp2=mkl(nvirt*2), ba2=mkl(nvirt*2), blist=mkl(nvirt*4), bcnt=mkl(4), bind=mkl(12), blist2=mkl(nvirt*4), bcnt2=mkl(4), bind2=mkl(12);
    const uint32_t total=(uint32_t)nvirt, gsz=64, per_group2=4;   // pass 2: 16 lanes per element, 4 elements per 64-thread group
    auto scalar=Pipe(dev,L,@"fg_custom_block1_bot1"), ups=Pipe(dev,L,@"fg_custom_block1_bot1_upsample_v4"),
         bounded=Pipe(dev,L,planar?@"fg_custom_block1_bot1_bounded_v4p":@"fg_custom_block1_bot1_bounded_v4",formal),
         pass1=Pipe(dev,L,@"fg_custom_block1_bot1_repair_pass1_v4"), pass2=Pipe(dev,L,@"fg_custom_block1_bot1_listed_v4"), compact=Pipe(dev,L,@"fg_compact_suspects_ordered"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto reset=[&](){ memset(out.contents,0xA5,nout*2); memset(bm.contents,0,nvirt); memset(brs.contents,0xFF,nvirt); memset(bra.contents,0,nvirt*2); memset(bfa.contents,0,nvirt*2); memset(bs2.contents,0xFF,nvirt); memset(bp2.contents,0,nvirt*2); memset(ba2.contents,0,nvirt*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    auto skips=[&](id<MTLComputeCommandEncoder> en, int base){ [en setBuffer:s0 offset:0 atIndex:base]; [en setBuffer:s1 offset:0 atIndex:base+1]; [en setBuffer:s2 offset:0 atIndex:base+2]; [en setBuffer:out offset:0 atIndex:base+3]; };
    // listed-layout buffers 0..8 shared by the fused bounded kernel, pass 1 and pass 2
    auto listed_common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; for(int k=0;k<3;++k) [en setBuffer:bups[k] offset:0 atIndex:2+k]; skips(en,5); };
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
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded];
        if(!planar){ listed_common(e1);
            [e1 setBuffer:bm offset:0 atIndex:9]; [e1 setBuffer:brs offset:0 atIndex:10]; [e1 setBuffer:bra offset:0 atIndex:11]; [e1 setBuffer:bfa offset:0 atIndex:12]; [e1 setBuffer:bs2 offset:0 atIndex:13]; [e1 setBuffer:bp2 offset:0 atIndex:14]; [e1 setBuffer:ba2 offset:0 atIndex:15];
            [e1 dispatchThreadgroups:MTLSizeMake(npix/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; }
        else { [e1 setBuffer:bw offset:0 atIndex:0]; [e1 setBuffer:bb offset:0 atIndex:1]; skips(e1,3);
            [e1 setBuffer:bm offset:0 atIndex:7]; [e1 setBuffer:brs offset:0 atIndex:8]; [e1 setBuffer:bra offset:0 atIndex:9]; [e1 setBuffer:bfa offset:0 atIndex:11]; [e1 setBuffer:bs2 offset:0 atIndex:12]; [e1 setBuffer:bp2 offset:0 atIndex:13]; [e1 setBuffer:ba2 offset:0 atIndex:14];
            for(uint32_t plane=0;plane<3;++plane){ [e1 setBuffer:bups[plane] offset:0 atIndex:2]; [e1 setBytes:&plane length:4 atIndex:10];
                [e1 dispatchThreadgroups:MTLSizeMake(npix/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; } }
        [e1 endEncoding];
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
    printf("block1_bot1 v4%s [%s, %s] scalar %7.3f ms | pipeline %7.3f ms  %5.2fx  mismatches %ld of %zu  flagged %u (%.2f%%)  pass2 %u (%.2f%%)\n", planar?" planar":" fused", wsrc, formal?"formal":"empirical", ts, tg, ts/tg, m, nout, c1, 100.0*c1/nvirt, c2, 100.0*c2/nvirt);
    return m==0?0:3; } }
