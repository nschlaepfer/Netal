// L20 (160x96, 3x3, 48 -> 32, weight token 63): scalar exact vs raw SIMD vs MMA-bounded SIMD.
// Measures exactness, the guard's flag rate, and - the question that decides soundness - whether
// every real mismatch is flagged.
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
    if (!F) { printf("fn %s missing in %s\n", fn.UTF8String, lib); exit(1); }
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:F error:&e];
    if (!p) { printf("pso: %s\n", e.localizedDescription.UTF8String); exit(1); }
    return p;
}

int main(int argc, const char** argv) { @autoreleasepool {
    const uint32_t W = 160, H = 96, CI = 48, CO = 32;
    const Conv3x3Parameters P{W, H, CI, CO, 9u * CI * CO, 0u};
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];

    std::vector<uint8_t> wbytes;
    { FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
      wbytes.resize(n); fread(wbytes.data(), 1, n, f); fclose(f); }
    printf("weights %zu bytes\n", wbytes.size());

    const size_t in_halves = (size_t)W * H * CI, out_halves = (size_t)W * H * CO;
    std::vector<_Float16> in(in_halves);
    std::mt19937 rng(20260904); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : in) v = (_Float16)nd(rng);

    id<MTLBuffer> bw = [dev newBufferWithBytes:wbytes.data() length:wbytes.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi = [dev newBufferWithBytes:in.data() length:in.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:out_halves*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bm = [dev newBufferWithLength:out_halves options:MTLResourceStorageModeShared];

    auto run = [&](id<MTLComputePipelineState> pso, bool with_mask, int reps) {
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            memset(bo.contents, 0xA5, out_halves*2); memset(bm.contents, 0, out_halves);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bi offset:0 atIndex:1];
            [e setBuffer:bo offset:0 atIndex:2]; [e setBytes:&P length:sizeof(P) atIndex:3];
            if (with_mask) [e setBuffer:bm offset:0 atIndex:4];
            [e dispatchThreads:MTLSizeMake(W, H, CO) threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) { printf("  command failed\n"); return -1.0; }
            best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
        }
        return best;
    };

    std::vector<uint16_t> ref(out_halves), got(out_halves);
    std::vector<uint8_t> mask(out_halves);

    double t_scalar = run(Pipe(dev, argv[2], @"fg_conv_3x3"), false, 10);
    memcpy(ref.data(), bo.contents, out_halves*2);
    printf("scalar   %8.3f ms   (reference)\n", t_scalar);

    double t_simd = run(Pipe(dev, argv[3], @"fg_conv_3x3_simd"), false, 10);
    memcpy(got.data(), bo.contents, out_halves*2);
    long m_simd = 0; for (size_t i = 0; i < out_halves; ++i) if (got[i] != ref[i]) ++m_simd;
    printf("raw simd %8.3f ms   %5.2fx   %ld mismatching halfwords\n", t_simd, t_scalar/t_simd, m_simd);

    double t_b = run(Pipe(dev, argv[4], @"fg_conv_3x3"), false, 10);
    memcpy(got.data(), bo.contents, out_halves*2);
    memcpy(mask.data(), bm.contents, out_halves);
    long m_b = 0, flagged = 0, uncovered = 0;
    for (size_t i = 0; i < out_halves; ++i) {
        if (mask[i]) ++flagged;
        if (got[i] != ref[i]) { ++m_b; if (!mask[i]) ++uncovered; }
    }
    printf("twofloat %8.3f ms   %5.2fx   %ld mismatching, %ld flagged (%.3f%%), %ld UNCOVERED\n",
           t_b, t_scalar/t_b, m_b, flagged, 100.0*flagged/out_halves, uncovered);
    printf("\n%zu output halfwords. Two-pass estimate: %.3f ms fast + %.1f%% of scalar = %.3f ms (%.2fx)\n",
           out_halves, t_b, 100.0*flagged/out_halves,
           t_b + t_scalar * (double)flagged / out_halves,
           t_scalar / (t_b + t_scalar * (double)flagged / out_halves));
    return uncovered == 0 ? 0 : 3;
} }
