#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

int main(int argc, const char** argv) { @autoreleasepool {
    const int TILES = argc > 1 ? atoi(argv[1]) : 16384;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&err];
    if (!lib) { printf("lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    unsigned t = TILES; [fc setConstantValue:&t type:MTLDataTypeUInt atIndex:0];

    std::vector<_Float16> A(TILES*128), B(TILES*128);
    std::mt19937 rng(5);
    std::normal_distribution<float> act(0.f, 1.f), wgt(0.f, 0.02f);
    for (size_t i = 0; i < A.size(); ++i) { A[i] = (_Float16)act(rng); B[i] = (_Float16)wgt(rng); }

    // CPU reference: fp16 products are exact in double, and 16 of them sum exactly there too.
    std::vector<uint16_t> ref(TILES*64);
    for (int t2 = 0; t2 < TILES; ++t2)
      for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
          double s = 0;
          for (int k = 0; k < 16; ++k) s += (double)A[t2*128 + i*16 + k] * (double)B[t2*128 + k*8 + j];
          _Float16 h = (_Float16)s; memcpy(&ref[t2*64 + i*8 + j], &h, 2);
      }

    id<MTLBuffer> ba = [dev newBufferWithBytes:A.data() length:A.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithBytes:B.data() length:B.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc = [dev newBufferWithLength:TILES*64*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    struct Row { const char* name; bool per_tile; };
    Row rows[] = {{"lower_oracle", false}, {"lower_twofloat", false}, {"lower_checked", true}};
    printf("%-16s %10s %10s %12s %10s\n", "lowering", "ms", "vs oracle", "mismatches", "fallback%");
    double base_ms = 0;
    std::vector<uint16_t> oracle_out(TILES*64);
    for (auto& r : rows) {
        id<MTLFunction> fn = [lib newFunctionWithName:@(r.name) constantValues:fc error:&err];
        if (!fn) { printf("%s: %s\n", r.name, err.localizedDescription.UTF8String); continue; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { printf("%s pso: %s\n", r.name, err.localizedDescription.UTF8String); continue; }
        double best = 1e9; long mism = 0, mism_sub = 0; unsigned fb = 0;
        for (int rep = 0; rep < 12; ++rep) {
            memset(bc.contents, 0xA5, TILES*64*2); *(unsigned*)bf.contents = 0;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:ba offset:0 atIndex:0]; [e setBuffer:bb offset:0 atIndex:1];
            [e setBuffer:bc offset:0 atIndex:2]; [e setBuffer:bf offset:0 atIndex:3];
            if (r.per_tile) [e dispatchThreadgroups:MTLSizeMake(TILES,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            else [e dispatchThreads:MTLSizeMake(TILES*64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
            if (rep == 0) {
                const uint16_t* got = (const uint16_t*)bc.contents;
                int shown = 0;
                for (size_t i = 0; i < ref.size(); ++i) if (got[i] != ref[i]) {
                    ++mism;
                    if (((ref[i] >> 10) & 0x1F) == 0 || ((got[i] >> 10) & 0x1F) == 0) ++mism_sub;
                    if (getenv("DEBUG") && shown < 4 && !strcmp(r.name, "lower_twofloat")) {
                        ++shown;
                        int t3 = i / 64, e3 = i % 64, ii = e3 / 8, jj = e3 % 8;
                        long double x = 0; double d = 0;
                        for (int k = 0; k < 16; ++k) {
                            x += (long double)A[t3*128+ii*16+k] * (long double)B[t3*128+k*8+jj];
                            d += (double)A[t3*128+ii*16+k] * (double)B[t3*128+k*8+jj];
                        }
                        _Float16 hg, hr; memcpy(&hg,&got[i],2); memcpy(&hr,&ref[i],2);
                        printf("   [%zu] exact=%.12Lg double=%.12g  gpu=%.9g(0x%04x) ref=%.9g(0x%04x) "
                               "fp32(exact)=%.9g\n", i, x, d, (double)hg, got[i], (double)hr, ref[i],
                               (double)(float)x);
                    }
                }
                fb = *(unsigned*)bf.contents;
                if (!strcmp(r.name, "lower_oracle")) memcpy(oracle_out.data(), got, oracle_out.size()*2);
                else { long d = 0; for (size_t i = 0; i < oracle_out.size(); ++i) if (got[i] != oracle_out[i]) ++d;
                       printf("   vs oracle element-wise: %ld differing of %zu%s\n", d, oracle_out.size(),
                              d == 0 ? "   <-- BIT-IDENTICAL" : ""); }
            }
        }
        if (base_ms == 0) base_ms = best;
        printf("%-16s %10.3f %9.1fx %12ld %9.2f%%   (%ld of them subnormal)\n", r.name, best, base_ms / best, mism,
               100.0 * fb / (TILES * 64.0), mism_sub);
    }
    printf("\n%d tiles = %d K=16 stages, conv-shaped (act N(0,1), weights N(0,0.02))\n",
           TILES, TILES * 64);
    return 0;
} }
