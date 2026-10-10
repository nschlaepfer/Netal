// conv2 family, end to end: scalar oracle vs raw SIMD vs the guarded pipeline (bounded fast pass ->
// compaction -> indirect resuming repair). Final output compared byte-for-byte against the oracle.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
struct Params { uint32_t residual; };
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn) {
    NSError* e=nil; id<MTLFunction> F=[L newFunctionWithName:fn]; if(!F){printf("missing %s\n",fn.UTF8String);exit(1);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W=80,H=46,C=64; const size_t no=(size_t)W*H*C;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(8*4608), iv(no), rv(no), bv(C+8);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:rv)v=(_Float16)nd(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    // live weights and bias when given: argv[2]=weights token, argv[3]=bias token
    const char* wsrc = "synthetic";
    if(argc>3){ auto load=[&](const char* path, std::vector<_Float16>& dst){ FILE* f=fopen(path,"rb"); if(!f){printf("cannot open %s\n",path);exit(1);}
            fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); if((size_t)n>dst.size()*2){printf("%s is %ld bytes, buffer %zu\n",path,n,dst.size()*2);exit(1);}
            fread(dst.data(),1,n,f); fclose(f); };
        load(argv[2],wv); load(argv[3],bv); wsrc="LIVE"; }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),iv.size()*2), br=mk(rv.data(),rv.size()*2);
    id<MTLBuffer> bo=[dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bm=[dev newBufferWithLength:no options:MTLResourceStorageModeShared];
    id<MTLBuffer> brs=[dev newBufferWithLength:no options:MTLResourceStorageModeShared];
    id<MTLBuffer> bra=[dev newBufferWithLength:no*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> blist=[dev newBufferWithLength:no*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bcnt=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bind=[dev newBufferWithLength:12 options:MTLResourceStorageModeShared];
    Params P{1u}; const uint32_t total=(uint32_t)no, gsz=64;
    auto scalar=Pipe(dev,L,@"fg_custom_block0_conv2"), bounded=Pipe(dev,L,@"fg_custom_block0_conv2_bounded"),
         listed=Pipe(dev,L,@"fg_custom_block0_conv2_listed"), compact=Pipe(dev,L,@"fg_compact_suspects"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2];
        [en setBuffer:br offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; };
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bm.contents,0,no); memset(brs.contents,0xFF,no); memset(bra.contents,0,no*2); *(uint32_t*)bcnt.contents=0; };
    std::vector<uint16_t> ref(no), got(no);
    double ts=1e9; for(int r=0;r<8;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:scalar]; common(en); [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
        ts=std::min(ts,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
    memcpy(ref.data(),bo.contents,no*2);
    double tg=1e9; uint32_t count=0; double mean_stage=0;
    for(int r=0;r<8;++r){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; [e1 setComputePipelineState:bounded]; common(e1);
        [e1 setBuffer:bm offset:0 atIndex:6]; [e1 setBuffer:brs offset:0 atIndex:7]; [e1 setBuffer:bra offset:0 atIndex:8];
        [e1 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(no,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:listed]; common(e3);
        [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(gsz,1,1)]; [e3 endEncoding];
        [cb commit]; [cb waitUntilCompleted]; tg=std::min(tg,(cb.GPUEndTime-cb.GPUStartTime)*1000.0);
        if(r==0){ count=*(uint32_t*)bcnt.contents; const uint8_t* rs=(const uint8_t*)brs.contents; const uint8_t* mk2=(const uint8_t*)bm.contents; double s=0; long n=0;
            for(size_t i=0;i<no;++i) if(mk2[i]&&rs[i]!=255u){s+=rs[i];++n;} mean_stage=n?s/n:0; } }
    memcpy(got.data(),bo.contents,no*2); long m=0; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++m;
    printf("conv2 [%s weights] scalar %7.3f ms | guarded %7.3f ms  %5.2fx  %ld mismatching  %u flagged (%.2f%%)  first stage %.1f/36\n",
           wsrc, ts, tg, ts/tg, m, count, 100.0*count/no, mean_stage);
    return m==0?0:3; } }
