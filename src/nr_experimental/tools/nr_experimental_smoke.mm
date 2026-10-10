// Live-generated-texture smoke harness for the experimental NR runtime (Opus-owned).
//
// It generates its own colour texture on the GPU from a deterministic procedural pattern - NO model bytes,
// NO captured activations, NO reference or expected outputs are read anywhere in this program. It then
// exercises the encode-only path exactly as the game hook will: one caller-owned command buffer, the runtime
// appends work to it, the harness commits and waits (the harness may wait; the game hook may not).
//
// It reports actual GPU time for the NR segment and CPU time spent inside encode(), separately, and checks
// the buffer guard bands after completion. It makes no correctness claim about pixel values.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_experimental_assets.h"
#include "nr_experimental_block0.h"
#include "nr_experimental_encoder.h"
#import <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using nr_exp::Buf;

static id<MTLTexture> make_colour(id<MTLDevice> dev, uint32_t w, uint32_t h) {
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                              width:w height:h mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  d.storageMode = MTLStorageModeShared;
  return [dev newTextureWithDescriptor:d];
}
// Deterministic procedural content, written on the CPU once. Half-float encode of a smooth gradient plus a
// grid, so the input is well-conditioned and reproducible without any captured data.
static void fill_colour(id<MTLTexture> t, uint32_t w, uint32_t h, uint32_t seed, int pattern = 0) {
  std::vector<uint16_t> row(size_t(w) * 4);
  auto half = [](float f) {   // round-to-nearest-even float32 -> float16
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000u; int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (e <= 0) return uint16_t(s);
    if (e >= 31) return uint16_t(s | 0x7bffu);
    uint32_t hm = m >> 13, rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) { ++hm; if (hm == 0x400u) { hm = 0; ++e; if (e >= 31) return uint16_t(s | 0x7bffu); } }
    return uint16_t(s | (uint32_t(e) << 10) | hm);
  };
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const float fx = float(x) / float(w), fy = float(y) / float(h);
      if (pattern == 0) {
        const float grid = ((x / 32 + y / 32 + seed) % 2) ? 0.85f : 0.15f;
        row[size_t(x) * 4 + 0] = half(fx * grid);
        row[size_t(x) * 4 + 1] = half(fy * grid);
        row[size_t(x) * 4 + 2] = half((1.0f - fx) * grid);
        row[size_t(x) * 4 + 3] = half(1.0f);
      } else {
        // Wide dynamic range with sign changes and near-cancellations: the case where an exact fixed-point
        // accumulation and a native fp32 FMA chain are most likely to disagree. Still purely procedural.
        uint32_t v = x * 1973u + y * 9277u + seed * 26699u; v ^= v >> 13; v *= 0x5bd1e995u; v ^= v >> 15;
        auto span = [&](int shift) {
          const int e = int((v >> shift) & 15u) - 8;                 // exponent in [-8, 7]
          const float m = 1.0f + float((v >> (shift + 4)) & 255u) / 256.0f;
          return ((v >> (shift + 12)) & 1u ? -1.0f : 1.0f) * ldexpf(m, e);
        };
        row[size_t(x) * 4 + 0] = half(span(0));
        row[size_t(x) * 4 + 1] = half(span(3));
        row[size_t(x) * 4 + 2] = half(span(6));
        row[size_t(x) * 4 + 3] = half(fx * fy);
      }
    }
    [t replaceRegion:MTLRegionMake2D(0, y, w, 1) mipmapLevel:0 withBytes:row.data() bytesPerRow:w * 8];
  }
}

int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 2) { fprintf(stderr, "usage: nr_experimental_smoke ASSET_ROOT [FRAMES]\n"); return 2; }
  const std::string root = argv[1];
  const int frames = argc > 2 ? atoi(argv[2]) : 3;
  bool segments = false, unpinned = false, hash_outputs = false;
  int pattern = 0;   // 0 = smooth gradient + grid; 1 = wide dynamic range, exercises rounding much harder
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--segments") segments = true;
    else if (a == "--allow-unpinned-libraries") unpinned = true;
    else if (a == "--hash-outputs") hash_outputs = true;
    else if (a.rfind("--pattern=", 0) == 0) pattern = atoi(a.c_str() + 10);
  }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
  id<MTLCommandQueue> q = [dev newCommandQueue];

  auto t0 = std::chrono::steady_clock::now();
  nr_exp::Assets assets; std::string error;
  if (!nr_exp::load_assets(dev, root, assets, error, unpinned)) { fprintf(stderr, "asset load failed: %s\n", error.c_str()); return 3; }
  const double assets_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  t0 = std::chrono::steady_clock::now();
  nr_exp::Block0Resources res;
  if (!nr_exp::allocate_block0(dev, assets, res, error)) { fprintf(stderr, "allocation failed: %s\n", error.c_str()); return 3; }
  const double alloc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  t0 = std::chrono::steady_clock::now();
  nr_exp::EncoderResources enc_res;
  if (!nr_exp::allocate_encoder(dev, assets, res, enc_res, error)) { fprintf(stderr, "encoder allocation failed: %s\n", error.c_str()); return 3; }
  const double enc_alloc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  id<MTLTexture> colour = make_colour(dev, 1920, 1080);
  if (!colour) { fprintf(stderr, "texture creation failed\n"); return 3; }
  fill_colour(colour, 1920, 1080, 1, pattern);

  printf("{\"schema\": \"neuralstudio.nr.experimental-smoke.v1\", \"lane\": \"experimental\", \"exact_parity_claim\": false,\n");
  printf(" \"asset_root\": \"%s\", \"parent_root\": \"%s\", \"unpinned_libraries\": %s,\n",
         root.c_str(), assets.parent_root.c_str(), assets.unpinned_libraries ? "true" : "false");
  if (assets.unpinned_libraries) {
    printf(" \"libraries_as_loaded\": {");
    bool first = true;
    for (const auto& kv : assets.library_sha256) { printf("%s\"%s\": \"%s\"", first ? "" : ", ", kv.first.c_str(), kv.second.substr(0, 16).c_str()); first = false; }
    printf("},\n");
  }
  printf(" \"input_pattern\": %d,\n", pattern);
  printf(" \"prepare_ms\": {\"assets\": %.2f, \"block0_allocation\": %.2f, \"encoder_allocation\": %.2f},\n", assets_ms, alloc_ms, enc_alloc_ms);
  printf(" \"persistent_bytes\": {\"encoder_carry\": %zu, \"encoder_scratch\": %zu},\n", enc_res.carry.total_bytes(), enc_res.scratch.total_bytes());
  int failures = 0;
  if (segments) {
    // Profiling pass: each segment is encoded into its OWN command buffer and timed on its own. The chain is
    // sequential and the buffers persist, so the segments compose into the same work as one buffer; the split
    // exists only to attribute GPU time. This is not how the game path runs.
    struct Seg { const char* name; int kind; uint32_t mask; };
    const Seg segs[] = {{"block0", 0, 0}, {"blocks1-4.C32", 1, 1u << 0}, {"blocks5-8.wide64", 1, 1u << 1},
                        {"blocks9-14.C128", 1, 1u << 2}, {"blocks15-22.C256", 1, 1u << 3}, {"native-exports", 1, 1u << 4}};
    printf(" \"segments\": [\n");
    double total = 0;
    for (size_t i = 0; i < sizeof(segs) / sizeof(segs[0]); ++i) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      nr_exp::Block0Encoded e0{}; nr_exp::EncoderEncoded e1{};
      bool ok = segs[i].kind == 0 ? nr_exp::encode_block0(cb, assets, res, colour, 0, e0, error)
                                  : nr_exp::encode_encoder(cb, assets, res, enc_res, e1, error, segs[i].mask);
      if (!ok) { printf("  {\"segment\": \"%s\", \"error\": \"%s\"},\n", segs[i].name, error.c_str()); ++failures; break; }
      [cb commit]; [cb waitUntilCompleted];
      const double ms = cb.status == MTLCommandBufferStatusCompleted ? (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 : -1.0;
      total += ms > 0 ? ms : 0;
      printf("  {\"segment\": \"%s\", \"dispatches\": %u, \"gpu_ms\": %.2f}%s\n", segs[i].name,
             segs[i].kind == 0 ? e0.dispatches : e1.dispatches, ms, i + 1 < sizeof(segs) / sizeof(segs[0]) ? "," : "");
      fflush(stdout);
    }
    printf(" ],\n \"segment_total_gpu_ms\": %.2f,\n", total);
  }
  printf(" \"frames\": [\n");
  for (int f = 0; f < frames; ++f) {
    id<MTLCommandBuffer> cb = [q commandBuffer];      // stands in for the game's command buffer
    cb.label = @"nr_exp.smoke";
    const auto e0 = std::chrono::steady_clock::now();
    nr_exp::Block0Encoded enc{};
    nr_exp::EncoderEncoded enc22{};
    bool ok = nr_exp::encode_block0(cb, assets, res, colour, uint32_t(f), enc, error);
    if (ok) ok = nr_exp::encode_encoder(cb, assets, res, enc_res, enc22, error);
    const double encode_cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
    if (!ok) { printf("  {\"frame\": %d, \"encoded\": false, \"error\": \"%s\"}%s\n", f, error.c_str(), f + 1 < frames ? "," : ""); ++failures; continue; }
    enc.dispatches += enc22.dispatches; enc.compute_encoders += enc22.compute_encoders;
    const auto c0 = std::chrono::steady_clock::now();
    [cb commit]; [cb waitUntilCompleted];             // the HARNESS may wait; the game hook may not
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
    const bool completed = cb.status == MTLCommandBufferStatusCompleted;
    const double gpu_ms = completed ? (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 : 0.0;
    size_t guards = nr_exp::guard_errors(res.features) + nr_exp::guard_errors(res.skip) + nr_exp::guard_errors(res.encoder) + nr_exp::guard_errors(res.input);
    for (const auto& b : res.scratch) guards += nr_exp::guard_errors(b);
    for (const auto& b : enc_res.carry.slots) guards += nr_exp::guard_errors(b);
    for (const auto& b : enc_res.scratch.slots) guards += nr_exp::guard_errors(b);
    guards += nr_exp::guard_errors(enc_res.entry);
    if (!completed || guards) ++failures;
    std::string out_hash;
    if (hash_outputs && completed) {
      // Harness-only: read the finished buffers back on the CPU to compare configurations. The game path
      // never does this; the runtime's encode() is forbidden from touching buffer contents at all.
      auto hex = [](const unsigned char* d) { std::string o; const char* H = "0123456789abcdef";
                   for (int i = 0; i < 32; ++i) { o += H[d[i] >> 4]; o += H[d[i] & 15]; } return o; };
      CC_SHA256_CTX c; CC_SHA256_Init(&c);
      CC_SHA256_Update(&c, enc_res.entry.data(), CC_LONG(enc_res.entry.bytes));
      for (const auto& sk : enc_res.skips) CC_SHA256_Update(&c, sk.data(), CC_LONG(sk.bytes));
      CC_SHA256_Update(&c, res.skip.data(), CC_LONG(res.skip.bytes));
      unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(d, &c);
      out_hash = hex(d);
    }
    if (!out_hash.empty()) printf("  {\"frame\": %d, \"outputs_sha256\": \"%s\"},\n", f, out_hash.c_str());
    printf("  {\"frame\": %d, \"encoded\": true, \"completed\": %s, \"dispatches\": %u, \"compute_encoders\": %u,"
           " \"encode_cpu_ms\": %.3f, \"submit_wait_ms\": %.3f, \"gpu_ms\": %.3f, \"guard_errors\": %zu%s}%s\n",
           f, completed ? "true" : "false", enc.dispatches, enc.compute_encoders, encode_cpu_ms, wall_ms, gpu_ms, guards,
           completed ? "" : (std::string(", \"error\": \"") + (cb.error ? cb.error.localizedDescription.UTF8String : "unknown") + "\"").c_str(),
           f + 1 < frames ? "," : "");
    fflush(stdout);
  }
  printf(" ],\n \"stage\": \"block 0 and encoder blocks 1-22 with the native skip/entry exports; the body is not ported to the encode-only path yet\",\n");
  printf(" \"entry_bytes\": %zu, \"skip_bytes\": [%zu, %zu, %zu, %zu],\n", enc_res.entry.bytes,
         enc_res.skips.size() > 0 ? enc_res.skips[0].bytes : 0, enc_res.skips.size() > 1 ? enc_res.skips[1].bytes : 0,
         enc_res.skips.size() > 2 ? enc_res.skips[2].bytes : 0, enc_res.skips.size() > 3 ? enc_res.skips[3].bytes : 0);
  printf(" \"failures\": %d}\n", failures);
  return failures ? 1 : 0;
} }
