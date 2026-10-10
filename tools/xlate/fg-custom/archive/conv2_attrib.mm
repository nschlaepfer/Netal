#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
struct Params { uint32_t residual; };
static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, id<MTLLibrary> L, NSString* fn, int mode=-1) {
    NSError* e=nil; id<MTLFunction> F;
    if(mode<0) F=[L newFunctionWithName:fn]; else { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; uint32_t m=mode; [cv setConstantValue:&m type:MTLDataTypeUInt atIndex:0]; F=[L newFunctionWithName:fn constantValues:cv error:&e]; }
    if(!F){printf("missing %s\n",fn.UTF8String);exit(1);}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:F error:&e]; if(!p){printf("pso %s\n",e.localizedDescription.UTF8String);exit(1);} return p; }
int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W=80,H=46,C=64; const size_t no=(size_t)W*H*C;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil; id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLLibrary> L=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e]; if(!L){printf("lib %s\n",e.localizedDescription.UTF8String);return 1;}
    std::vector<_Float16> wv(8*4608), iv(no), rv(no), bv(C+8);
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,0.05f), na(0.f,1.f);
    for(auto&v:wv)v=(_Float16)nd(rng); for(auto&v:iv)v=(_Float16)na(rng); for(auto&v:rv)v=(_Float16)nd(rng); for(auto&v:bv)v=(_Float16)nd(rng);
    if(argc>3){ auto load=[&](const char* path, std::vector<_Float16>& dst){ FILE* f=fopen(path,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); fread(dst.data(),1,n,f); fclose(f); }; load(argv[2],wv); load(argv[3],bv); }
    auto mk=[&](const void* p,size_t n){return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    auto mkl=[&](size_t n){return [dev newBufferWithLength:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> bw=mk(wv.data(),wv.size()*2), bb=mk(bv.data(),bv.size()*2), bi=mk(iv.data(),iv.size()*2), br=mk(rv.data(),rv.size()*2);
    id<MTLBuffer> bo=mkl(no*2), bm=mkl(no), brs=mkl(no), bra=mkl(no*2), blist=mkl(no*4), bcnt=mkl(4), bind=mkl(12), bfa=mkl(no*2), blist2=mkl(no*4), bcnt2=mkl(4), bind2=mkl(12);
    Params P{1u}; const uint32_t total=(uint32_t)no, gsz=64;
    auto common=[&](id<MTLComputeCommandEncoder> en){ [en setBuffer:bw offset:0 atIndex:0]; [en setBuffer:bb offset:0 atIndex:1]; [en setBuffer:bi offset:0 atIndex:2];
        [en setBuffer:br offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4]; [en setBytes:&P length:sizeof(P) atIndex:5]; };
    auto reset=[&](){ memset(bo.contents,0xA5,no*2); memset(bm.contents,0,no); memset(brs.contents,0xFF,no); memset(bra.contents,0,no*2); *(uint32_t*)bcnt.contents=0; *(uint32_t*)bcnt2.contents=0; };
    auto scalar=Pipe(dev,L,@"fg_custom_block0_conv2"), listed=Pipe(dev,L,@"fg_custom_block0_conv2_listed"), compact=Pipe(dev,L,@"fg_compact_suspects"), indk=Pipe(dev,L,@"fg_indirect_from_counter");
    auto raw=Pipe(dev,L,@(argv[4]));
    std::vector<id<MTLComputePipelineState>> modes; for(int m=0;m<10;++m) modes.push_back(Pipe(dev,L,@"fg_conv2_attrib",m));
    const char* names[10]={"0 baseline bounded","1 bit-trick ulp32 (sound)","2 error=0 (no ulp32)","3 no bound MMAs","4 no double-float fold","5 no guard at all","6 tight 3.5/3.5/1.5 bound (sound)","7 formal incl. truncation","8 formal + subnormal fallback","9 formal + exponent proxy"};
    auto shortcut=Pipe(dev,L,@"fg_conv2_listed_shortcut"), pass1=Pipe(dev,L,@"fg_conv2_repair_pass1");
    auto ordered=Pipe(dev,L,@"fg_compact_suspects_ordered");
    printf("PSO maxTotalThreadsPerThreadgroup: scalar %lu raw %lu", (unsigned long)scalar.maxTotalThreadsPerThreadgroup, (unsigned long)raw.maxTotalThreadsPerThreadgroup);
    for(int m=0;m<10;++m) printf(" mode%d %lu", m, (unsigned long)modes[m].maxTotalThreadsPerThreadgroup); printf("\n");
    // fast pass only, per mode, interleaved, min of 8; flags counted from mode's suspect buffer
    double tmin[10]; uint32_t flags[10]; std::vector<uint16_t> base(no), oth(no); long diffvs0[10]={0};
    for(int m=0;m<10;++m) tmin[m]=1e9;
    double traw=1e9, tscalar=1e9;
    for(int r=0;r<8;++r) for(int m=0;m<10;++m){ reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:modes[m]]; common(en); [en setBuffer:bm offset:0 atIndex:6]; [en setBuffer:brs offset:0 atIndex:7]; [en setBuffer:bra offset:0 atIndex:8]; [en setBuffer:bfa offset:0 atIndex:9];
        [en dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
        tmin[m]=std::min(tmin[m],(cb.GPUEndTime-cb.GPUStartTime)*1000.0);
        if(r==0){ uint32_t c=0; const uint8_t* mk2=(const uint8_t*)bm.contents; for(size_t i=0;i<no;++i) c+=(mk2[i]!=0); flags[m]=c;
            if(m==0) memcpy(base.data(),bo.contents,no*2); else { memcpy(oth.data(),bo.contents,no*2); long d=0; for(size_t i=0;i<no;++i) if(oth[i]!=base[i]) ++d; diffvs0[m]=d; } }
        if(m==0){ id<MTLCommandBuffer> c2=[q commandBuffer]; id<MTLComputeCommandEncoder> e2=[c2 computeCommandEncoder]; [e2 setComputePipelineState:raw]; common(e2);
            [e2 dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [e2 endEncoding]; [c2 commit]; [c2 waitUntilCompleted]; traw=std::min(traw,(c2.GPUEndTime-c2.GPUStartTime)*1000.0);
            id<MTLCommandBuffer> c3=[q commandBuffer]; id<MTLComputeCommandEncoder> e3=[c3 computeCommandEncoder]; [e3 setComputePipelineState:scalar]; common(e3);
            [e3 dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [e3 endEncoding]; [c3 commit]; [c3 waitUntilCompleted]; tscalar=std::min(tscalar,(c3.GPUEndTime-c3.GPUStartTime)*1000.0); } }
    printf("scalar %.4f ms   raw SIMD %.4f ms\n", tscalar, traw);
    for(int m=0;m<10;++m) printf("  fast pass %-28s %.4f ms  flags %u (%.2f%%)  fast-output diff vs mode0 %ld\n", names[m], tmin[m], flags[m], 100.0*flags[m]/no, diffvs0[m]);
    // scalar reference for exactness
    std::vector<uint16_t> ref(no); { reset(); id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:scalar]; common(en);
        [en dispatchThreads:MTLSizeMake(W,H,C) threadsPerThreadgroup:MTLSizeMake(8,4,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; memcpy(ref.data(),bo.contents,no*2); }
    // full pipeline matrix: mode x compaction x repair group size; stages timed in separate command buffers; exactness on the final output
    for(int m : {1,7,9}) for(int sc=2;sc<3;++sc) { const int oc=1; const uint32_t g=64u;
        double tb=1e9,tc=1e9,tl=1e9; long mism=0; double mean_stage=0; uint32_t cnt=0; long multi=0; uint32_t second=0;
        for(int r=0;r<8;++r){ reset();
            { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder]; [en setComputePipelineState:modes[m]]; common(en); [en setBuffer:bm offset:0 atIndex:6]; [en setBuffer:brs offset:0 atIndex:7]; [en setBuffer:bra offset:0 atIndex:8]; [en setBuffer:bfa offset:0 atIndex:9];
              [en dispatchThreadgroups:MTLSizeMake((W*H)/8,C/8,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; tb=std::min(tb,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
            { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; [e2 setComputePipelineState:(oc?ordered:compact)];
              [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1]; [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
              [e2 dispatchThreads:MTLSizeMake(((no+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
              [e2 setComputePipelineState:indk]; [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1]; [e2 setBytes:&g length:4 atIndex:2];
              [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding]; [cb commit]; [cb waitUntilCompleted]; tc=std::min(tc,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
            if(sc<2){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:(sc?shortcut:listed)]; common(e3);
              [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11];
              [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(g,1,1)]; [e3 endEncoding]; [cb commit]; [cb waitUntilCompleted]; tl=std::min(tl,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
            else { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder]; [e3 setComputePipelineState:pass1]; common(e3);
              [e3 setBuffer:blist offset:0 atIndex:6]; [e3 setBuffer:bcnt offset:0 atIndex:7]; [e3 setBuffer:brs offset:0 atIndex:8]; [e3 setBuffer:bra offset:0 atIndex:9]; [e3 setBuffer:bfa offset:0 atIndex:10]; [e3 setBuffer:bm offset:0 atIndex:11]; [e3 setBuffer:blist2 offset:0 atIndex:12]; [e3 setBuffer:bcnt2 offset:0 atIndex:13];
              [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(g,1,1)]; [e3 endEncoding];
              id<MTLComputeCommandEncoder> e4=[cb computeCommandEncoder]; [e4 setComputePipelineState:indk]; [e4 setBuffer:bcnt2 offset:0 atIndex:0]; [e4 setBuffer:bind2 offset:0 atIndex:1]; [e4 setBytes:&g length:4 atIndex:2];
              [e4 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e4 endEncoding];
              id<MTLComputeCommandEncoder> e5=[cb computeCommandEncoder]; [e5 setComputePipelineState:listed]; common(e5);
              [e5 setBuffer:blist2 offset:0 atIndex:6]; [e5 setBuffer:bcnt2 offset:0 atIndex:7]; [e5 setBuffer:brs offset:0 atIndex:8]; [e5 setBuffer:bra offset:0 atIndex:9];
              [e5 dispatchThreadgroupsWithIndirectBuffer:bind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(g,1,1)]; [e5 endEncoding]; [cb commit]; [cb waitUntilCompleted]; tl=std::min(tl,(cb.GPUEndTime-cb.GPUStartTime)*1000.0); }
            if(r==0){ const uint16_t* got=(const uint16_t*)bo.contents; for(size_t i=0;i<no;++i) if(got[i]!=ref[i]) ++mism; cnt=*(uint32_t*)bcnt.contents;
                const uint8_t* rs=(const uint8_t*)brs.contents; const uint8_t* mk2=(const uint8_t*)bm.contents; double s=0; long n=0; for(size_t i=0;i<no;++i) if(mk2[i]&&rs[i]!=255u){s+=rs[i];++n;} mean_stage=n?s/n:0; multi=0; for(size_t i=0;i<no;++i) if(mk2[i]>1) ++multi; second=*(uint32_t*)bcnt2.contents; } }
        printf("mode %d  %-8s | bounded %.4f  compact %.4f  repair %.4f  = %.4f ms  (%.2fx scalar)  mismatches %ld  flags %u (%.2f%%)  multi-flag %ld  pass2 %u  mean first stage %.1f/36\n",
               m, sc==0?"full":(sc==1?"shortcut":"two-pass"), tb, tc, tl, tb+tc+tl, tscalar/(tb+tc+tl), mism, cnt, 100.0*cnt/no, multi, second, mean_stage);
    }
    return 0; } }
