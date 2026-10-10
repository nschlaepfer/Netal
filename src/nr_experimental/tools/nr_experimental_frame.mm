// End-to-end frame driver for the experimental NR runtime (Opus-owned).
//
// Drives the public API exactly as the game bridge will: prepare once, then per frame create a command buffer,
// let the runtime append the whole model to it, commit it, and call complete() from the completion path. The
// HARNESS owns the buffer and the commit; the runtime never commits or waits. The colour input is generated
// procedurally on the CPU - no captured frames, no reference outputs, no expected values anywhere.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../nr_experimental_api.h"
#define NR_EXPERIMENTAL_NO_VALIDATION 1
#include "nr_body_runtime.h"
#include "nr_experimental_context.h"   // private: the capture buffer lives on Context, not in the public API
#include "nr_experimental_gpu_profile.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <map>
#include <vector>
#include "../runtime/nr_experimental_boundary_diag.h"

using namespace netal_nr_experimental;

static uint16_t to_half(float f) {
  uint32_t x; __builtin_memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000u; int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
  if (e <= 0) return uint16_t(s);
  if (e >= 31) return uint16_t(s | 0x7bffu);
  uint32_t hm = m >> 13, rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) { ++hm; if (hm == 0x400u) { hm = 0; ++e; if (e >= 31) return uint16_t(s | 0x7bffu); } }
  return uint16_t(s | (uint32_t(e) << 10) | hm);
}
static std::string input_sha256(const void* p, size_t n) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(p, CC_LONG(n), d);
  char out[2 * CC_SHA256_DIGEST_LENGTH + 1];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(out + 2 * i, 3, "%02x", d[i]);
  return std::string(out);
}
static void fill(id<MTLTexture> t, uint32_t w, uint32_t h, uint32_t seed) {
  std::vector<uint16_t> row(size_t(w) * 4);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const float fx = float(x) / float(w), fy = float(y) / float(h);
      const float grid = ((x / 32 + y / 32 + seed) % 2) ? 0.85f : 0.15f;
      row[size_t(x) * 4 + 0] = to_half(fx * grid);
      row[size_t(x) * 4 + 1] = to_half(fy * grid);
      row[size_t(x) * 4 + 2] = to_half((1.0f - fx) * grid);
      row[size_t(x) * 4 + 3] = to_half(1.0f);
    }
    [t replaceRegion:MTLRegionMake2D(0, y, w, 1) mipmapLevel:0 withBytes:row.data() bytesPerRow:w * 8];
  }
}

int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 2) { fprintf(stderr, "usage: nr_experimental_frame BUNDLE [FRAMES] [--approximate] [--temporal] (--input RGBA16F | --synthetic) [--dump OUT]\n"); return 2; }
  const std::string bundle = argv[1];
  const int frames = argc > 2 ? atoi(argv[2]) : 1;
  Arithmetic mode = Arithmetic::OriginalKernels;
  bool temporal_chain = false;
  std::string input_path, dump_path;
  bool allow_synthetic = false;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--approximate") mode = Arithmetic::ApproximateNative;
    else if (a == "--temporal") temporal_chain = true;
    // Diagnostic: drive every frame from one captured 1920x1080 RGBA16F model input instead of the
    // synthetic fill, and write the finished output surface out, so two bundles can be compared on
    // identical bytes. Comparison only - the file is an input to the model, never a substitute for a stage.
    else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
    else if (a == "--synthetic") allow_synthetic = true;
    else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
  }
  std::vector<uint8_t> external;
  std::string input_identity;
  // An omitted --input does NOT load a frame from the bundle: fill() generates a synthetic 32x32
  // checkerboard/gradient seeded by the frame index. That is harness test data, not model-representative
  // content, and a correctness or cost number measured on it is evidence about the harness. Requiring the
  // flag is deliberate - omitting it once put a numerical validation and its cost measurement on
  // different data without either run saying so. --synthetic opts in explicitly and labels the run.
  if (input_path.empty() && !allow_synthetic) {
    fprintf(stderr, "refusing to run without --input: the default is a SYNTHETIC harness pattern, not a bundle frame.\n"
                    "Pass --input <live model input>.rgba16f, or --synthetic to opt in explicitly.\n");
    return 2;
  }
  if (!input_path.empty()) {
    FILE* fh = fopen(input_path.c_str(), "rb");
    if (!fh) { fprintf(stderr, "cannot open --input %s\n", input_path.c_str()); return 2; }
    fseek(fh, 0, SEEK_END); const long n = ftell(fh); fseek(fh, 0, SEEK_SET);
    if (n != long(size_t(1920) * 1080 * 8)) { fclose(fh); fprintf(stderr, "--input must be 1920x1080 RGBA16F (%ld bytes, got %ld)\n", long(size_t(1920) * 1080 * 8), n); return 2; }
    external.resize(size_t(n));
    if (fread(external.data(), 1, external.size(), fh) != external.size()) { fclose(fh); fprintf(stderr, "short read on --input\n"); return 2; }
    fclose(fh);
    input_identity = "file:" + input_path + ":sha256=" + input_sha256(external.data(), external.size());
  } else {
    input_identity = "SYNTHETIC-HARNESS-PATTERN:fill(32x32 checkerboard 0.85/0.15 over x/y gradient, seed=frame+1) - NOT a bundle frame, NOT model-representative";
  }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
  id<MTLCommandQueue> q = [dev newCommandQueue];

  Report pr; ContextPtr ctx; PrepareOptions opt; opt.runtime_bundle = bundle; opt.arithmetic = mode;
  if (!prepare(dev, opt, ctx, pr)) { printf("{\"prepared\": false, \"error\": \"%s\"}\n", pr.error.c_str()); return 3; }
  printf("{\"schema\": \"neuralstudio.nr.experimental-frame.v1\", \"lane\": \"experimental\", \"exact_parity_claim\": false,\n");
  printf(" \"prepared\": true, \"prepare_cpu_ms\": %.1f, \"arithmetic\": \"%s\",\n", pr.prepare_cpu_ms, pr.arithmetic_label.c_str());
  printf(" \"model_identity\": \"%s\",\n", pr.model_identity.c_str());
  printf(" \"input_identity\": \"%s\",\n \"frames\": [\n", input_identity.c_str());

  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
  id<MTLTexture> colour = [dev newTextureWithDescriptor:d];
  MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:1920 height:1080 mipmapped:NO];
  md.usage = MTLTextureUsageShaderRead; md.storageMode = MTLStorageModeShared;
  id<MTLTexture> motion = [dev newTextureWithDescriptor:md];   // zero motion: a still camera, a valid temporal case
  int failures = 0;
  for (int f = 0; f < frames; ++f) {
    if (external.empty()) fill(colour, 1920, 1080, uint32_t(f) + 1);
    else [colour replaceRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0 withBytes:external.data() bytesPerRow:1920 * 8];
    id<MTLCommandBuffer> cb = [q commandBuffer];      // the HARNESS owns this, as the game will
    cb.label = @"nr_exp.frame";
    FrameInputs in; in.color = colour; in.stream_id = 1; in.frame_id = uint64_t(f);
    in.reset = (f == 0) || !temporal_chain;   // --temporal: frame 0 resets, the rest are real temporal frames
    if (!in.reset) {
      in.motion = motion; in.motion_units = MotionUnits::PixelsFullRes;
      in.motion_direction = MotionDirection::CurrentToPrevious; in.motion_y_down = true;
    }
    TicketPtr ticket; Report er;
    nr_exp_encode_profile.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const EncodeStatus st = encode(ctx, in, cb, ticket, er);
    const double encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (st == EncodeStatus::RejectedBeforeEncoding) {
      printf("  {\"frame\": %d, \"status\": \"rejected-before-encoding\", \"error\": \"%s\", \"encode_cpu_ms\": %.3f}%s\n",
             f, er.error.c_str(), encode_ms, f + 1 < frames ? "," : "");
      ++failures; continue;
    }
    // The bridge's contract: the output surface must be callable BEFORE the buffer is committed, so that an
    // export can be appended into the same buffer. Check that here, not after completion.
    id<MTLTexture> out_before_commit = output_texture(ticket);
    const bool out_ok = out_before_commit != nil && out_before_commit.width == 1920 && out_before_commit.height == 1080
                        && out_before_commit.pixelFormat == MTLPixelFormatRGBA16Float;
    [cb commit]; [cb waitUntilCompleted];             // the harness may wait here; the game hook may not
    const bool done = cb.status == MTLCommandBufferStatusCompleted;
    const double gpu_ms = done ? (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 : 0.0;
    // Harness-only: hash the finished output surface so two configurations can be compared bit for bit.
    // The runtime never touches it; this read happens after the buffer completed.
    std::string out_sha;
    if (done && out_before_commit) {
      std::vector<uint8_t> px(size_t(1920) * 1080 * 8);
      [out_before_commit getBytes:px.data() bytesPerRow:1920 * 8 fromRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0];
      if (!dump_path.empty()) {
        const std::string path = frames > 1 ? dump_path + "." + std::to_string(f) : dump_path;
        if (FILE* o = fopen(path.c_str(), "wb")) { fwrite(px.data(), 1, px.size(), o); fclose(o); }
        else fprintf(stderr, "cannot write --dump %s\n", path.c_str());
      }
      // Opt-in per-encoder GPU timestamps, aggregated by label. RANKING ONLY: enabling this disables
      // encoder batching (one encoder per dispatch), so the totals here are NOT comparable to a normal
      // run's cost and must never be quoted as a candidate's improvement. It exists to choose which stage
      // to work on next from measured cost instead of from a guess.
      if (nr_exp::gpu_profiling_requested()) {
        std::string perr; nr_exp::GpuProfileStats pstats;
        const std::vector<nr_exp::GpuSpan> spans = nr_exp::gpu_profiler().resolve(perr, pstats);
        std::map<std::string, std::pair<double, int>> byl;
        for (const auto& sp : spans) { byl[sp.label].first += sp.ms; byl[sp.label].second += 1; }
        std::vector<std::pair<std::string, std::pair<double, int>>> v(byl.begin(), byl.end());
        std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.second.first > y.second.first; });
        printf("    per_label_gpu_ms (%s, sum double-counts overlap): ",
               nr_exp::gpu_profiling_per_dispatch()
                 ? "RANKING ONLY - batching DISABLED, so these totals are not comparable to a normal run"
                 : "BATCHED REGIONS - batching preserved; a region may cover many dispatches and is NOT one stage's cost");
        printf("{\"spans_valid\": %zu, \"spans_dropped\": %zu, \"sum_ms\": %.2f, \"union_ms\": %.2f, \"labels\": [",
               pstats.spans_valid, pstats.spans_dropped, pstats.sum_ms, pstats.union_ms);
        for (size_t k = 0; k < v.size() && k < 96; ++k)
          printf("%s{\"label\": \"%s\", \"ms\": %.3f, \"n\": %d}", k ? ", " : "",
                 v[k].first.c_str(), v[k].second.first, v[k].second.second);
        printf("]}\n");
      }
      // Opt-in GROUP capture: the C64 QKV stage output, written and hashed separately so the group can
      // be compared directly rather than inferred from the final frame hash.
      if (const char* cp = std::getenv("NR_EXP_CAPTURE_QKV_C64")) {
        // Read ONLY after a successful completion, and only when the copy actually happened this frame.
        if (cp[0] && ctx->encoder.qkv_capture.metal != nil && ctx->encoder.qkv_capture_bytes &&
            ctx->encoder.qkv_capture_done && cb.status == MTLCommandBufferStatusCompleted) {
          const char* HX = "0123456789abcdef";
          auto sha_of = [&](const uint8_t* b, size_t n) {
            unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(b, CC_LONG(n), d);
            std::string h; for (int i = 0; i < 32; ++i) { h += HX[d[i] >> 4]; h += HX[d[i] & 15]; }
            return h;
          };
          auto write_to = [&](const std::string& path, const uint8_t* b, size_t n) {
            if (FILE* o = fopen(path.c_str(), "wb")) { fwrite(b, 1, n, o); fclose(o); }
            else fprintf(stderr, "cannot write %s\n", path.c_str());
          };
          const auto& E = ctx->encoder;
          const std::string base_path = cp;
          // EXACT-ONCE COVERAGE, on the GPU's own counters: every output of the changed group must have
          // been written exactly once. Reported as counts, not as a pass/fail the reader has to trust.
          const uint32_t* cov = reinterpret_cast<const uint32_t*>(E.qkv_capture_cov.data());
          const uint32_t* pre = reinterpret_cast<const uint32_t*>(E.qkv_capture_cov_pre.data());
          const size_t cov_n = E.qkv_capture_cov_bytes / 4;
          size_t once = 0, zero = 0, many = 0;
          for (size_t i = 0; i < cov_n; ++i) {
            const uint32_t d = cov[i] - pre[i];   // DELTA: the counters are cumulative and never cleared
            once += (d == 1); zero += (d == 0); many += (d > 1);
          }
          // GUARD CANARIES on the capture buffers themselves.
          const size_t g_out = nr_exp::guard_errors(E.qkv_capture),
                       g_in  = nr_exp::guard_errors(E.qkv_capture_in),
                       g_cov = nr_exp::guard_errors(E.qkv_capture_cov) + nr_exp::guard_errors(E.qkv_capture_cov_pre);
          printf("    qkv_c64_group:   %zu bytes sha256 %s\n", E.qkv_capture_bytes,
                 sha_of(E.qkv_capture.data(), E.qkv_capture_bytes).c_str());
          printf("    qkv_c64_operand: %zu bytes sha256 %s\n", E.qkv_capture_in_bytes,
                 sha_of(E.qkv_capture_in.data(), E.qkv_capture_in_bytes).c_str());
          printf("    qkv_c64_coverage: %zu counters exactly_once=%zu never=%zu more_than_once=%zu\n",
                 cov_n, once, zero, many);
          // The PRODUCER's own guard bands, snapshotted after its dispatch. The counts above are the
          // capture destinations and only ever tested the COPY; this line tests what the kernel wrote.
          const uint8_t* pg = E.qkv_capture_guards.data();
          size_t producer_guard_errors = 0;
          for (size_t i = 0; i < 2 * nr_exp::Buf::kGuard; ++i) producer_guard_errors += (pg[i] != 0xc3);
          printf("    qkv_c64_capture_canaries: out=%zu operand=%zu coverage=%zu guard byte errors (these test the COPY)\n",
                 g_out, g_in, g_cov);
          printf("    qkv_c64_producer_guards: %zu of %zu guard bytes wrong on the stage's own output buffer\n",
                 producer_guard_errors, size_t(2 * nr_exp::Buf::kGuard));
          write_to(base_path,              E.qkv_capture.data(),     E.qkv_capture_bytes);
          write_to(base_path + ".operand", E.qkv_capture_in.data(),  E.qkv_capture_in_bytes);
          // BOTH raw coverage files are retained so the delta can be verified independently.
          write_to(base_path + ".coverage.post", E.qkv_capture_cov.data(),     E.qkv_capture_cov_bytes);
          write_to(base_path + ".coverage.pre",  E.qkv_capture_cov_pre.data(), E.qkv_capture_cov_bytes);
          write_to(base_path + ".producer-guards", pg, 2 * nr_exp::Buf::kGuard);
        }
      }
      // Which body kernels actually resolved to the override, recorded during encode. Without this a
      // candidate that never reached its kernel is indistinguishable from one that changed nothing.
      if (ctx->body.override.active) {
        printf("    body_override_resolved:");
        for (const auto& kv : ctx->body.override.resolved)
          printf(" %s=%s", kv.first.c_str(), kv.second.substr(0, 20).c_str());
        printf("\n");
      }
      // BLOCK 0 STAGE-LEVEL DIAGNOSTIC. Block 0's nine Swin-32 stages each write their own persistent
      // scratch buffer y(i) = scratch[4+2i] with a coverage counter c(i) = scratch[5+2i]; the adapter
      // writes raw = scratch[0] (half) and packed = scratch[1] (uchar). After completion those buffers
      // hold the LAST chunk's operands and outputs - the same chunk for every configuration, so two
      // builds can be compared on identical freshly produced operands and the FIRST differing stored
      // partial can be named instead of inferring from the eventual frame. Bounded: a fixed-size prefix
      // per buffer plus the sha256 of the whole, never a full dump of hundreds of megabytes.
      if (const char* bd = std::getenv("NR_EXP_BLOCK0_STAGE_DUMP")) {
        const size_t kPrefix = 1u << 20;   // 1 MiB per buffer
        const char* HX = "0123456789abcdef";
        auto shaof = [&](const uint8_t* b, size_t n) {
          unsigned char dg[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(b, CC_LONG(n), dg);
          std::string h; for (int k = 0; k < 32; ++k) { h += HX[dg[k] >> 4]; h += HX[dg[k] & 15]; }
          return h;
        };
        auto emit = [&](const char* label, const nr_exp::Buf& b) {
          if (!b.metal || !b.bytes) { printf("    b0stage %-12s ABSENT\n", label); return; }
          const uint8_t* p8 = b.data();
          const size_t n = std::min(kPrefix, b.bytes);
          printf("    b0stage %-12s bytes %10zu  sha256 %s  guard_errors %zu  prefix %zu\n",
                 label, b.bytes, shaof(p8, b.bytes).c_str(), nr_exp::guard_errors(b), n);
          const std::string path = std::string(bd) + "/" + label + ".bin";
          if (FILE* o = fopen(path.c_str(), "wb")) { fwrite(p8, 1, n, o); fclose(o); }
          else fprintf(stderr, "cannot write %s\n", path.c_str());
        };
        emit("raw", ctx->block0.scratch[0]);
        emit("packed", ctx->block0.scratch[1]);
        for (size_t k = 0; k < 9; ++k) {
          char ny[16], nc[16];
          snprintf(ny, sizeof(ny), "y%zu", k); snprintf(nc, sizeof(nc), "c%zu", k);
          emit(ny, ctx->block0.scratch[4 + k * 2]);
          emit(nc, ctx->block0.scratch[5 + k * 2]);
        }
        emit("weight", ctx->block0.weight);
      }
      // ENCODER-SIDE PROVENANCE, once per process. body_override_resolved covers the body; without this
      // the encoder half of a combined candidate is pinned only by its manifest, and a manifest is a
      // DECLARATION. resolved_pipeline_source is what the runtime actually bound: for every encode
      // pipeline, "override:<lib>@<sha>" or "pinned:<lib>@<sha>". An unchanged stage can therefore be
      // ASSERTED to have come from the pinned library rather than assumed to.
      if (f == 0) {
        const auto& A = ctx->assets;
        size_t n_override = 0, n_pinned = 0;
        for (const auto& kv : A.resolved_pipeline_source)
          (kv.second.rfind("override:", 0) == 0 ? n_override : n_pinned) += 1;
        printf("    qkv_override_active: %s, manifest_sha256 \"%s\", version \"%s\", root \"%s\"\n",
               A.qkv_override_active ? "true" : "false", A.qkv_override_manifest_sha256.c_str(),
               A.qkv_override_version.c_str(), A.qkv_override_root.c_str());
        printf("    resolved_pipelines: %zu total, %zu from override, %zu from pinned\n",
               A.resolved_pipeline_source.size(), n_override, n_pinned);
        for (const auto& kv : A.resolved_pipeline_source)
          printf("      resolved %s -> %s\n", kv.first.c_str(), kv.second.c_str());
        for (const auto& kv : A.library_sha256)
          printf("      pinned_library %s @ %s\n", kv.first.c_str(), kv.second.c_str());
        for (const auto& kv : A.qkv_override_sha256)
          printf("      override_library %s @ %s\n", kv.first.c_str(), kv.second.c_str());
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
        printf("    encoder_dense_active: %s, manifest_sha256 \"%s\", version \"%s\", root \"%s\", tier_mask 0x%x, prepack_gpu_ms %.3f\n",
               A.encdense_active ? "true" : "false", A.encdense_manifest_sha256.c_str(), A.encdense_version.c_str(),
               A.encdense_root.c_str(), A.encdense_tier_mask, ctx->encoder.encdense_prepack_gpu_ms);
        for (const auto& kv : A.encdense_sha256)
          printf("      encdense_library %s @ %s\n", kv.first.c_str(), kv.second.c_str());
#endif
      }
      unsigned char dg[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(px.data(), CC_LONG(px.size()), dg);
      const char* H = "0123456789abcdef";
      for (int i = 0; i < 16; ++i) { out_sha += H[dg[i] >> 4]; out_sha += H[dg[i] & 15]; }
    }
    Report cr; complete(ticket, cb, cr);
    const bool ok = done && cr.model_encoded && cr.history_advanced && cr.error.empty() && out_ok;
    if (!ok) ++failures;
    printf("  {\"frame\": %d, \"status\": \"%s\", \"completed\": %s, \"model_encoded\": %s, \"history_advanced\": %s,"
           " \"output_sha256\": \"%s\", \"output_before_commit\": %s, \"dispatches\": %u, \"encode_cpu_ms\": %.3f, \"scratch_fill_ms\": %.3f, \"scratch_fill_mib\": %.1f,"
           " \"scratch_fill_calls\": %zu, \"poison_fill_mib\": %.1f, \"nr_gpu_ms\": %.2f, \"error\": \"%s\"}%s\n",
           f, st == EncodeStatus::Encoded ? "encoded" : "encoded-discard-only", done ? "true" : "false",
           cr.model_encoded ? "true" : "false", cr.history_advanced ? "true" : "false",
           out_sha.c_str(), out_ok ? "true" : "false", cr.model_dispatches, encode_ms,
           nr_exp_encode_profile.fill_ms, double(nr_exp_encode_profile.fill_bytes) / 1048576.0,
           nr_exp_encode_profile.fill_calls, double(nr_exp_encode_profile.poison_bytes) / 1048576.0, gpu_ms,
           (st == EncodeStatus::Encoded ? cr.error : er.error + " | completion: " + cr.error).c_str(),
           f + 1 < frames ? "," : "");
    {
      const auto& cb2 = nr_exp::last_chain_bytes();
      if (cb2.total)
        printf("    chain_working_set: total %.2f GiB, wide tier %.2f GiB, coverage counters %.2f GiB"
               " (bound-per-dispatch lower bound, not measured DRAM traffic)\n",
               double(cb2.total) / 1073741824.0, double(cb2.wide) / 1073741824.0, double(cb2.coverage) / 1073741824.0);
    }
    if (nr_exp::boundary_hashes_requested()) {
      // Output-only, post-completion. Compared against the accepted lane's preserved receipt; never fed back in.
      const auto& b = nr_exp::last_boundary_hashes();
      if (b.captured)
        printf("  {\"frame\": %d, \"boundary_hashes\": {\"entry\": \"%s\", \"skip_c256\": \"%s\", \"skip_c128\": \"%s\","
               " \"skip_c64\": \"%s\", \"skip_c32\": \"%s\", \"block0_skip\": \"%s\", \"encoder_buffer\": \"%s\","
               " \"middle_view\": \"%s\", \"decoder_pre_image\": \"%s\", \"decoder_pre_image_source\": \"%s\","
               " \"pool_final\": \"%s\", \"final_surface\": \"%s\"}},\n",
               f, b.entry.c_str(), b.skip_c256.c_str(), b.skip_c128.c_str(), b.skip_c64.c_str(), b.skip_c32.c_str(),
               b.block0_skip.c_str(), b.encoder_buffer.c_str(), b.middle_view.c_str(), b.decoder_pre_image.c_str(),
               b.decoder_pre_image_source.c_str(), b.pool_final.c_str(), b.final_surface.c_str());
    }
    {   // per-encoder GPU spans, resolved only after the buffer completed
      std::string perr; nr_exp::GpuProfileStats pstats; auto spans = nr_exp::gpu_profiler().resolve(perr, pstats);
      if (!spans.empty()) {
        // FULL ORDERED SPAN DUMP (NR_EXP_SPAN_DUMP=<path>). The body creates one encoder per dispatch in every
        // mode, so in regions mode each body span is one real dispatch executed exactly as in a normal frame -
        // no batching change - and its position in this list is its position in the encode order. That lets
        // the ViT, split and decoder internals be sized per dispatch instead of as one "body" region.
        if (const char* sd = std::getenv("NR_EXP_SPAN_DUMP")) {
          if (FILE* o = fopen(sd, "w")) {
            fprintf(o, "index\tlabel\tms\n");
            for (size_t si = 0; si < spans.size(); ++si) fprintf(o, "%zu\t%s\t%.4f\n", si, spans[si].label.c_str(), spans[si].ms);
            fclose(o);
          } else fprintf(stderr, "cannot write NR_EXP_SPAN_DUMP %s\n", sd);
        }
        std::vector<const nr_exp::GpuSpan*> top; for (const auto& sp : spans) top.push_back(&sp);
        std::sort(top.begin(), top.end(), [](const nr_exp::GpuSpan* a, const nr_exp::GpuSpan* b) { return a->ms > b->ms; });
        printf("    gpu_spans: %zu encoders, sum %.1f ms, UNION %.1f ms (overlap %.1f ms), extent %.1f ms\n",
               spans.size(), pstats.sum_ms, pstats.union_ms, pstats.overlap_ms, pstats.span_extent_ms);
        printf("    gpu_profile: mode=%s calibration=%s ticks_to_ms=%.6g expected=%zu returned=%zu valid=%zu dropped=%zu%s\n",
               pstats.batching_disabled_for_profile ? "UNBATCHED (per-encoder sampling forces one encoder per dispatch)" : "batched",
               pstats.calibrated ? "paired-cpu-gpu" : "UNCALIBRATED-assumed-ns",
               pstats.ticks_to_ms, pstats.encoders_expected, pstats.samples_returned, pstats.spans_valid, pstats.spans_dropped,
               nr_exp::gpu_profiler().clamped_from ? "  TRUNCATED" : "");
        printf("    note: shares below are of the whole command (%.1f ms), not of the sampled sum; blit encoders are not sampled\n", gpu_ms);
        for (size_t i = 0; i < top.size() && i < 12; ++i) printf("      %7.2f ms  %s\n", top[i]->ms, top[i]->label.c_str());
        // Grouped totals, so a "largest bottleneck" claim rests on the whole sampled set rather than the top few.
        std::map<std::string, std::pair<double, size_t>> group;
        for (const auto& sp : spans) {
          std::string g = sp.label;
          const size_t sp_pos = g.find(' ');                       // "launch NNN kernel" -> keep the kernel name
          if (g.rfind("launch ", 0) == 0) { const size_t k = g.find(' ', sp_pos + 1); g = k == std::string::npos ? g : g.substr(k + 1); }
          else if (sp_pos != std::string::npos) g = g.substr(0, sp_pos);
          group[g].first += sp.ms; group[g].second += 1;
        }
        std::vector<std::pair<std::string, std::pair<double, size_t>>> gv(group.begin(), group.end());
        std::sort(gv.begin(), gv.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        printf("    gpu_groups (kernel, sampled total):\n");
        // EVERY group, not a top-N. A top-N list hid nr_wide_qkv.c64 entirely, which is exactly the
        // number the C64 pilot needs. (Codex, NR_C64_MATRIX_REVIEW.)
        for (size_t i = 0; i < gv.size(); ++i)
          printf("      %8.2f ms  %4zu enc  %5.1f%% of command  %s\n", gv[i].second.first, gv[i].second.second,
                 gpu_ms > 0 ? 100.0 * gv[i].second.first / gpu_ms : 0.0, gv[i].first.c_str());
      } else if (!perr.empty()) printf("    gpu_spans: %s\n", perr.c_str());
    }
    { printf("    phases:");
      for (int i = 0; i < 8; ++i) if (nr_exp_encode_profile.phase_ms[i] > 0.05)
        printf(" %s=%.2f", NrExpEncodeProfile::phase_name(i), nr_exp_encode_profile.phase_ms[i]);
      printf("\n"); }
    fflush(stdout);
  }
  printf(" ],\n \"failures\": %d}\n", failures);
  return failures ? 1 : 0;
} }
