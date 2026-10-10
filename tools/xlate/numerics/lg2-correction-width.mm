// How many correction bits does Metal's fast::log2 need to reproduce Blackwell lg2.approx over the
// [1,2) binade? FG's compact EX2 needed two. This is the same question for the primitive SR is
// missing, against ground truth captured on the RTX 5090 today.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <map>
#include <vector>
static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void f(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = as_type<uint>(fast::log2(as_type<float>(0x3F800000u | i)));
})";
int main(int argc, const char** argv) { @autoreleasepool {
    const unsigned N = 1u << 23;
    std::vector<uint32_t> rtx(N);
    FILE* f = fopen(argv[1], "rb"); fread(rtx.data(), 4, N, f); fclose(f);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&e];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"f"] error:&e];
    id<MTLBuffer> out = [dev newBufferWithLength:(size_t)N*4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:pso]; [en setBuffer:out offset:0 atIndex:0];
    [en dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const uint32_t* fast = (const uint32_t*)out.contents;
    std::map<long long,long> hist;
    for (unsigned i = 0; i < N; ++i) hist[(long long)rtx[i] - (long long)fast[i]]++;
    long total = 0; for (auto& kv : hist) total += kv.second;
    printf("all %u mantissas of [1,2): %zu distinct deltas vs Metal fast::log2\n", N, hist.size());
    long cum = 0; int shown = 0;
    for (auto& [d, n] : hist) {
        cum += n;
        if (shown++ < 12) printf("   delta %+4lld  %9ld  %6.3f%%\n", d, n, 100.0*n/total);
    }
    // How much of the binade needs a wide correction, and where does it live?
    printf("\n   |delta| <= B   share of the binade   bits\n");
    for (int bits = 2; bits <= 16; bits += 2) {
        long long B = (1LL << (bits - 1)) - 1; long inside = 0;
        for (auto& [d, n] : hist) if (d >= -B - 1 && d <= B) inside += n;
        printf("   %10lld   %18.4f%%   %2d\n", B, 100.0*inside/total, bits);
    }
    // the wide cases: are they confined to m near 1?
    long wide = 0; unsigned first_wide = N, last_wide = 0;
    for (unsigned i = 0; i < N; ++i) {
        long long d = (long long)rtx[i] - (long long)fast[i];
        if (d > 127 || d < -128) { ++wide; if (i < first_wide) first_wide = i; if (i > last_wide) last_wide = i; }
    }
    printf("\n   entries needing more than 8 bits: %ld (%.4f%%), mantissa index range [%u, %u]\n",
           wide, 100.0*wide/N, first_wide, last_wide);
    printf("   that upper index is m = %.9f, i.e. the region hugging 1.0\n",
           1.0 + (double)last_wide / (double)N);
    long long lo = hist.begin()->first, hi = hist.rbegin()->first;
    printf("   range [%lld, %lld]  -> %d distinct values, %d bits per entry\n", lo, hi,
           (int)hist.size(), (int)ceil(log2((double)hist.size())));
    printf("   table for the binade: 2^23 entries x %d bits = %.2f MiB\n",
           (int)ceil(log2((double)hist.size())),
           (double)N * ceil(log2((double)hist.size())) / 8.0 / 1048576.0);
    return 0;
} }
