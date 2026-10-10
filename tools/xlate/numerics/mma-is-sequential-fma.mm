// Is Apple's simdgroup_multiply_accumulate exact for K=8 half products into an fp32 accumulator?
//
// If it is, FG's K=16 stage is two exact partials, and their single fp32 add can be made exact with
// a two-sum - which turns "18x with a checked fallback" into "18x, bit-exact, no fallback".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cstring>

static const char* kSrc = R"(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// One 8x8 by 8x8 product per threadgroup, fp16 operands into an fp32 accumulator, zero-initialised.
kernel void mma8(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                 device float* C [[buffer(2)]], uint tg [[threadgroup_position_in_grid]]) {
    simdgroup_half8x8 a, b;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_load(a, A + tg * 64, 8);
    simdgroup_load(b, B + tg * 64, 8);
    simdgroup_multiply_accumulate(acc, a, b, acc);
    simdgroup_store(acc, C + tg * 64, 8);
}
)";

int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"mma8"] error:&err];
    if (!pso) { printf("pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    const int TG = 4096;                    // 4096 independent 8x8 products
    std::vector<_Float16> A(TG*64), B(TG*64);
    std::mt19937 rng(11);
    // Values chosen to stress fp32 addition: wide exponent spread inside fp16's range, so partial
    // sums differ in magnitude and any sequential fp32 accumulation must round.
    const char* mode = getenv("DIST") ? getenv("DIST") : "wide";
    std::normal_distribution<float> na(0.0f, 1.0f), nb(0.0f, 0.02f);
    std::uniform_real_distribution<float> mant(-2.0f, 2.0f);
    std::uniform_int_distribution<int> ex(-12, 12);
    for (size_t i = 0; i < A.size(); ++i) {
        if (!strcmp(mode, "conv")) { A[i] = (_Float16)na(rng); B[i] = (_Float16)nb(rng); }
        else if (!strcmp(mode, "unit")) { A[i] = (_Float16)na(rng); B[i] = (_Float16)na(rng); }
        else { A[i] = (_Float16)(mant(rng) * std::ldexp(1.0f, ex(rng)));
               B[i] = (_Float16)(mant(rng) * std::ldexp(1.0f, ex(rng))); }
    }
    printf("distribution: %s\n", mode);
    id<MTLBuffer> ba = [dev newBufferWithBytes:A.data() length:A.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithBytes:B.data() length:B.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc = [dev newBufferWithLength:TG*64*4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:ba offset:0 atIndex:0]; [e setBuffer:bb offset:0 atIndex:1]; [e setBuffer:bc offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(TG,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

    const float* C = (const float*)bc.contents;
    // Exact reference: each product of two fp16 is exact in double; 8 of them sum exactly in double
    // too (fp16 products have <= 22 significand bits, well inside double's 53).
    long mismatch = 0, checked = 0, seq_mismatch = 0, mma_vs_seq = 0;
    double worst_ulp = 0;
    for (int t = 0; t < TG; ++t)
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double exact = 0.0;
            for (int k = 0; k < 8; ++k) exact += (double)A[t*64 + i*8 + k] * (double)B[t*64 + k*8 + j];
            float ref = (float)exact;                       // exact sum, single rounding to fp32
            float got = C[t*64 + i*8 + j];
            ++checked;
            if (got != ref) {
                ++mismatch;
                double u = std::abs((double)got - exact) / std::ldexp(1.0, std::ilogb(ref) - 23);
                if (u > worst_ulp) worst_ulp = u;
            }
            // for comparison: what a naive sequential fp32 accumulation would give
            float seq = 0.0f;
            for (int k = 0; k < 8; ++k) seq = std::fma((float)A[t*64+i*8+k], (float)B[t*64+k*8+j], seq);
            if (seq != ref) ++seq_mismatch;
            if (memcmp(&seq, &got, 4) != 0) ++mma_vs_seq;
        }
    printf("checked %ld outputs (4096 8x8 products, wide exponent spread)\n", checked);
    printf("  simdgroup MMA vs exact-sum-rounded-once : %ld mismatches (%.4f%%), worst %.2f ULP\n",
           mismatch, 100.0*mismatch/checked, worst_ulp);
    printf("  sequential fp32 fma vs the same reference: %ld mismatches (%.4f%%)\n",
           seq_mismatch, 100.0*seq_mismatch/checked);
    printf("  simdgroup MMA vs sequential fp32 fma     : %ld differing bit patterns%s\n",
           mma_vs_seq, mma_vs_seq == 0 ? "  <-- MMA is exactly a sequential fp32 FMA chain" : "");
    return 0;
} }
