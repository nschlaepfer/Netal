// Do SR's recorded RTX transcendental outputs sit inside FG's two-bit correction window?
//
// FG's compact EX2 encodes RTX_bits = fast_bits - code + 1 with a two-bit code, i.e. the RTX result
// is always within {+1, 0, -1, -2} ULP of Metal's fast::exp2. SR independently memoised 29,851 ex2
// and 79,911 lg2 (input, RTX output) pairs from its own captures. If SR's pairs fall inside that
// window, FG's scheme covers SR's domain and the port is validated on SR's own data - two features,
// two capture sets, one instruction.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <map>
#include <vector>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void fast_ex2(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    out[i] = as_type<uint>(fast::exp2(as_type<float>(in[i])));
}
kernel void fast_lg2(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    out[i] = as_type<uint>(fast::log2(as_type<float>(in[i])));
}
)";

static std::vector<std::pair<uint32_t,uint32_t>> load(const char* dir, const char* stem) {
    std::vector<std::pair<uint32_t,uint32_t>> pairs;
    char kp[512], vp[512];
    snprintf(kp, sizeof kp, "%s/%s-input-table.bin", dir, stem);
    snprintf(vp, sizeof vp, "%s/%s-output-table.bin", dir, stem);
    FILE* kf = fopen(kp, "rb"); FILE* vf = fopen(vp, "rb");
    if (!kf || !vf) { printf("missing %s\n", stem); return pairs; }
    std::vector<uint32_t> k(1048576), v(1048576);
    fread(k.data(), 4, k.size(), kf); fread(v.data(), 4, v.size(), vf);
    fclose(kf); fclose(vf);
    for (size_t i = 0; i < k.size(); ++i) if (k[i] != 0) pairs.push_back({k[i], v[i]});
    return pairs;
}

int main(int argc, const char** argv) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    for (const char* stem : {"ex2", "lg2"}) {
        auto pairs = load(argv[1], stem);
        if (pairs.empty()) continue;
        std::vector<uint32_t> keys; keys.reserve(pairs.size());
        for (auto& p : pairs) keys.push_back(p.first);
        id<MTLBuffer> in = [dev newBufferWithBytes:keys.data() length:keys.size()*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:keys.size()*4 options:MTLResourceStorageModeShared];
        char fn[32]; snprintf(fn, sizeof fn, "fast_%s", stem);
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@(fn)] error:&err];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso]; [e setBuffer:in offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake(keys.size(),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];

        const uint32_t* fast = (const uint32_t*)out.contents;
        std::map<long long,long> hist;
        for (size_t i = 0; i < pairs.size(); ++i)
            hist[(long long)pairs[i].second - (long long)fast[i]]++;
        long total = 0, inside = 0;
        for (auto& [d, n] : hist) { total += n; if (d >= -2 && d <= 1) inside += n; }
        printf("\n--- %s: %ld recorded RTX pairs, %zu distinct deltas vs Metal fast::%s ---\n",
               stem, total, hist.size(), stem);
        int shown = 0;
        for (auto& [d, n] : hist) {
            if (shown++ < 10) printf("   RTX-fast = %+4lld   %8ld  %5.2f%%\n", d, n, 100.0*n/total);
        }
        if ((int)hist.size() > 10) printf("   ... %d more\n", (int)hist.size() - 10);
        printf("   inside FG's two-bit window {-2,-1,0,+1}: %ld of %ld  (%.4f%%)\n",
               inside, total, 100.0*inside/total);
    }
    return 0;
} }
