// The real two-pass: bounded SIMD writes output + suspect mask, then a masked scalar kernel
// recomputes only the flagged elements. Measures the actual end-to-end and checks the final
// output byte-for-byte against the scalar oracle.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

struct Conv3x3Parameters { uint32_t width, height, input_channels, output_channels,
                           active_weight_half_count, column_stage_schedule; };

static id<MTLComputePipelineState> Pipe(id<MTLDevice> d, const char* lib, NSString* fn) {
    NSError* e = nil;
    id<MTLLibrary> L = [d newLibraryWithURL:[NSURL fileURLWithPath:@(lib)] error:&e];
    if (!L) { printf("lib %s: %s\n", lib, e.localizedDescription.UTF8String); exit(1); }
    id<MTLFunction> F = [L newFunctionWithName:fn];
    if (!F) { printf("fn %s missing\n", fn.UTF8String); exit(1); }
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:F error:&e];
    if (!p) { printf("pso: %s\n", e.localizedDescription.UTF8String); exit(1); }
    return p;
}

int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W = 160, H = 96, CI = 48, CO = 32;
    const Conv3x3Parameters P{W, H, CI, CO, 9u * CI * CO, 0u};
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::vector<uint8_t> wb;
    { FILE* f = fopen(argv[1], "rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
      wb.resize(n); fread(wb.data(),1,n,f); fclose(f); }
    const size_t inh = (size_t)W*H*CI, outh = (size_t)W*H*CO;
    std::vector<_Float16> in(inh);
    std::mt19937 rng(20260904); std::normal_distribution<float> nd(0.f,1.f);
    for (auto& v : in) v = (_Float16)nd(rng);
    id<MTLBuffer> bw=[dev newBufferWithBytes:wb.data() length:wb.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi=[dev newBufferWithBytes:in.data() length:in.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo=[dev newBufferWithLength:outh*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bm=[dev newBufferWithLength:outh options:MTLResourceStorageModeShared];
    id<MTLBuffer> brs=[dev newBufferWithLength:outh options:MTLResourceStorageModeShared];
    id<MTLBuffer> bra=[dev newBufferWithLength:outh*2 options:MTLResourceStorageModeShared];

    auto enc = [&](id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pso, bool mask) {
        [e setComputePipelineState:pso];
        [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bi offset:0 atIndex:1];
        [e setBuffer:bo offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
        if (mask) { [e setBuffer:bm offset:0 atIndex:4];
                    [e setBuffer:brs offset:0 atIndex:5]; [e setBuffer:bra offset:0 atIndex:6]; }
        [e dispatchThreads:MTLSizeMake(W,H,CO) threadsPerThreadgroup:MTLSizeMake(8,4,1)];
    };

    auto scalar = Pipe(dev, argv[2], @"fg_conv_3x3");
    auto bounded = Pipe(dev, argv[3], @"fg_conv_3x3_simd_bounded");
    auto masked  = Pipe(dev, argv[4], @"fg_conv_3x3_masked");
    auto compact = Pipe(dev, argv[5], @"fg_compact_suspects");
    auto indirectk = Pipe(dev, argv[5], @"fg_indirect_from_counter");
    auto listed  = Pipe(dev, argv[5], @"fg_conv_3x3_listed");
    id<MTLBuffer> blist=[dev newBufferWithLength:outh*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bcnt=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bind=[dev newBufferWithLength:12 options:MTLResourceStorageModeShared];

    // reference
    double t_ref = 1e9;
    std::vector<uint16_t> ref(outh);
    for (int r = 0; r < 10; ++r) {
        memset(bo.contents, 0xA5, outh*2);
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        enc(e, scalar, false); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        t_ref = std::min(t_ref, (cb.GPUEndTime-cb.GPUStartTime)*1000.0);
    }
    memcpy(ref.data(), bo.contents, outh*2);

    // real two-pass in one command buffer, two encoders
    double t_two = 1e9; long flagged = 0;
    std::vector<uint16_t> got(outh);
    for (int r = 0; r < 10; ++r) {
        memset(bo.contents, 0xA5, outh*2); memset(bm.contents, 0, outh);
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; enc(e1, bounded, true); [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder]; enc(e2, masked, true); [e2 endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        t_two = std::min(t_two, (cb.GPUEndTime-cb.GPUStartTime)*1000.0);
        if (r == 0) { const uint8_t* m=(const uint8_t*)bm.contents;
                      for (size_t i=0;i<outh;++i) if (m[i]) ++flagged; }
    }
    memcpy(got.data(), bo.contents, outh*2);
    long mism = 0; for (size_t i=0;i<outh;++i) if (got[i]!=ref[i]) ++mism;

    printf("scalar reference   %7.3f ms\n", t_ref);
    printf("real two-pass      %7.3f ms   %5.2fx   %ld mismatching halfwords   %.2f%% flagged\n",
           t_two, t_ref/t_two, mism, 100.0*flagged/outh);
    // compacted three-pass: bounded -> compact -> dense indirect repair
    const uint32_t total = (uint32_t)outh, gsz = 64;
    double t_c = 1e9; uint32_t count = 0;
    for (int r = 0; r < 10; ++r) {
        memset(bo.contents, 0xA5, outh*2); memset(bm.contents, 0, outh);
        memset(brs.contents, 0xFF, outh); memset(bra.contents, 0, outh*2);
        *(uint32_t*)bcnt.contents = 0;
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e1=[cb computeCommandEncoder]; enc(e1, bounded, true); [e1 endEncoding];
        id<MTLComputeCommandEncoder> e2=[cb computeCommandEncoder];
        [e2 setComputePipelineState:compact];
        [e2 setBuffer:bm offset:0 atIndex:0]; [e2 setBuffer:blist offset:0 atIndex:1];
        [e2 setBuffer:bcnt offset:0 atIndex:2]; [e2 setBytes:&total length:4 atIndex:3];
        [e2 dispatchThreads:MTLSizeMake(outh,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e2 setComputePipelineState:indirectk];
        [e2 setBuffer:bcnt offset:0 atIndex:0]; [e2 setBuffer:bind offset:0 atIndex:1];
        [e2 setBytes:&gsz length:4 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e2 endEncoding];
        id<MTLComputeCommandEncoder> e3=[cb computeCommandEncoder];
        [e3 setComputePipelineState:listed];
        [e3 setBuffer:bw offset:0 atIndex:0]; [e3 setBuffer:bi offset:0 atIndex:1];
        [e3 setBuffer:bo offset:0 atIndex:2]; [e3 setBytes:&P length:sizeof(P) atIndex:3];
        [e3 setBuffer:blist offset:0 atIndex:4]; [e3 setBuffer:bcnt offset:0 atIndex:5];
        [e3 setBuffer:brs offset:0 atIndex:6]; [e3 setBuffer:bra offset:0 atIndex:7];
        [e3 dispatchThreadgroupsWithIndirectBuffer:bind indirectBufferOffset:0
                            threadsPerThreadgroup:MTLSizeMake(gsz,1,1)];
        [e3 endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        t_c = std::min(t_c, (cb.GPUEndTime-cb.GPUStartTime)*1000.0);
        if (r == 0) {
            count = *(uint32_t*)bcnt.contents;
            const uint8_t* rs=(const uint8_t*)brs.contents; const uint8_t* mk=(const uint8_t*)bm.contents;
            double sum=0; long n=0;
            for (size_t i=0;i<outh;++i) if (mk[i] && rs[i]!=255u) { sum+=rs[i]; ++n; }
            if (n) printf("   first uncertain stage: mean %.1f of 27  ->  %.0f%% of the element replayed\n",
                          sum/n, 100.0*(27.0-sum/n)/27.0);
        }
    }
    memcpy(got.data(), bo.contents, outh*2);
    long mism_c = 0; for (size_t i=0;i<outh;++i) if (got[i]!=ref[i]) ++mism_c;
    printf("compacted 3-pass   %7.3f ms   %5.2fx   %ld mismatching   %u compacted (%.2f%%)\n",
           t_c, t_ref/t_c, mism_c, count, 100.0*count/outh);
    return (mism == 0 && mism_c == 0) ? 0 : 3;
} }
