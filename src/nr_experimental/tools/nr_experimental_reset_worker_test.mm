// Offline safety gate for the bounded reset worker (Opus-owned). No game, no publication.
// Proves: repeated output equality, per-chunk GPU durations, one-chunk-in-flight behaviour, a concurrent
// short-command heartbeat on ANOTHER queue with its worst observed gap, and cancellation stopping between
// chunks with no publishable output.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "../nr_experimental_reset_worker.h"
#include "../runtime/nr_body_runtime.h"   // private: worker CPU accounting
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <string>
#include <vector>
using namespace netal_nr_experimental;

static std::string sha_hex(const void* p, size_t n) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(p, CC_LONG(n), d);
  static const char* H = "0123456789abcdef"; std::string o;
  for (unsigned char c : d) { o.push_back(H[c >> 4]); o.push_back(H[c & 15]); }
  return o;
}

int main(int argc, const char** argv) { @autoreleasepool {
  // ---- CPU regression of the FAILED-submission path, BEFORE any GPU work ----
  // v6 and v7 locked keep->lock twice on this path, which deadlocks a driver thread precisely when a
  // command buffer has failed. A watchdog turns a reintroduced deadlock into a reported failure instead
  // of a hang, because a hang here would look like a slow GPU rather than a defect.
  {
    std::atomic<bool> done_selftest{false};
    std::string why; bool passed = false;
    std::thread wd([&]{
      for (int i = 0; i < 500 && !done_selftest.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (!done_selftest.load()) {
        printf("{\"failure_record_selftest\": false, \"why\": \"did not return within 5s - the failed-submission path is deadlocked\"}\n");
        fflush(stdout); _exit(9);
      }
    });
    passed = reset_worker_selftest_failure_record(why);
    done_selftest.store(true); wd.join();
    printf("{\"failure_record_selftest\": %s, \"why\": \"%s\"}\n", passed ? "true" : "false", why.c_str());
    if (!passed) return 8;
  }

  // --selftest runs ONLY the CPU regression above: no device, no bundle, no GPU work.
  if (argc == 2 && std::string(argv[1]) == "--selftest") return 0;
  if (argc < 3) { fprintf(stderr, "usage: nr_experimental_reset_worker_test BUNDLE INPUT.rgba16f [UNITS_PER_CHUNK] [MAX_ENCODERS] [MAX_DISPATCHES] [MAX_CHUNK_MS] [BODY_EMISSION_BOUND] [original|approx]\n"); return 2; }
  const uint32_t units = argc > 3 ? uint32_t(atoi(argv[3])) : 1u;
  const uint32_t max_enc = argc > 4 ? uint32_t(atoi(argv[4])) : 64u;
  const uint32_t max_disp = argc > 5 ? uint32_t(atoi(argv[5])) : 64u;
  const double max_ms = argc > 6 ? atof(argv[6]) : 0.0;
  const uint32_t emit = argc > 7 ? uint32_t(atoi(argv[7])) : 0u;
  // The qkv hoist is packaged into encoder-approx/, so an A/B that leaves this at the default would run
  // the exact path, where the control and candidate bundles are BYTE-IDENTICAL, and would measure
  // nothing at all. Both sides of the A/B must pass the same value here.
  const bool approx = argc > 8 && std::string(argv[8]) == "approx";
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }

  // caller-owned colour texture, producer already complete (we fill it and wait before handing it over)
  NSData* raw = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
  if (!raw || raw.length != size_t(1920) * 1080 * 8) { fprintf(stderr, "input must be 1920x1080 RGBA16F\n"); return 2; }
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
  id<MTLTexture> colour = [dev newTextureWithDescriptor:d];
  [colour replaceRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0 withBytes:raw.bytes bytesPerRow:1920 * 8];

  ResetOptions o; o.runtime_bundle = argv[1]; o.max_units_per_chunk = units;
  o.arithmetic = approx ? Arithmetic::ApproximateNative : Arithmetic::OriginalKernels;
  o.max_encoders_per_chunk = max_enc; o.max_dispatches_per_chunk = max_disp; o.max_chunk_gpu_ms = max_ms; o.body_emission_encoder_bound = emit;
  ResetWorkerPtr worker; Report pr;
  // TOTAL COST, with preparation SEPARATED from steady state. Preparation is library load, pipeline
  // creation and allocation - it is paid once per worker and is not what an optimization to a kernel
  // changes, so mixing it into a candidate-vs-control comparison would flatter or penalise either side
  // depending on cache state. Both are reported; neither is hidden inside the other.
  const auto t_prep0 = std::chrono::steady_clock::now();
  if (!reset_worker_prepare(dev, o, worker, pr)) { printf("{\"prepared\": false, \"error\": \"%s\"}\n", pr.error.c_str()); return 3; }
  const double prepare_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prep0).count();

  // ---- heartbeat: short commands on ANOTHER queue, measuring the worst gap between completions ----
  id<MTLCommandQueue> hq = [dev newCommandQueue]; hq.label = @"nr_exp.heartbeat";
  std::atomic<bool> beating{true}; std::atomic<double> worst_gap{0}; std::atomic<int> beats{0};
  // A stand-in renderer, not an empty submission: a full-resolution 1920x1080 RGBA16F texture-to-texture
  // copy every beat, so the heartbeat measures contention against real blit traffic (about 16.6 MB moved
  // per beat) rather than a no-op command's latency.
  //
  // WHAT IS NOT ESTABLISHED (Codex, 05:19): these two textures are never written by us, so their contents
  // are undefined and may well be identical. The alternating direction stops the same copy repeating into
  // the same destination; it does NOT establish that the content being copied changes. Read this as a
  // fixed-size copy workload of known bytes, not as a changing-content renderer.
  MTLTextureDescriptor* hd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  hd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; hd.storageMode = MTLStorageModePrivate;
  id<MTLTexture> ha = [dev newTextureWithDescriptor:hd], hb_tex = [dev newTextureWithDescriptor:hd];
  std::thread hb([&]{
    auto last = std::chrono::steady_clock::now();
    uint32_t beat = 0;
    while (beating.load()) {
      id<MTLCommandBuffer> c = [hq commandBuffer];
      id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
      // alternate direction each beat so the same copy does not repeat into the same destination
      id<MTLTexture> src = (beat & 1u) ? ha : hb_tex, dst = (beat & 1u) ? hb_tex : ha;
      [e copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
              sourceSize:MTLSizeMake(1920,1080,1) toTexture:dst destinationSlice:0 destinationLevel:0
       destinationOrigin:MTLOriginMake(0,0,0)];
      [e endEncoding]; ++beat;
      [c commit]; [c waitUntilCompleted];
      const auto now = std::chrono::steady_clock::now();
      const double gap = std::chrono::duration<double, std::milli>(now - last).count();
      last = now; beats.fetch_add(1);
      double w = worst_gap.load(); while (gap > w && !worst_gap.compare_exchange_weak(w, gap)) {}
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });

  ResetJobPtr job; Report br;
  const auto t_steady0 = std::chrono::steady_clock::now();
  const bool started = reset_worker_begin(worker, colour, job, br);
  if (!started) { beating = false; hb.join(); printf("{\"begin\": false, \"error\": \"%s\"}\n", br.error.c_str()); return 3; }
  ResetProgress p;
  for (;;) { p = reset_worker_poll(job); if (p.state != ResetState::Running) break; std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
  const double steady_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_steady0).count();
  beating = false; hb.join();

  auto durs = reset_worker_chunk_durations(job);
  std::string out_sha = "";
  id<MTLTexture> out = reset_worker_output(job);
  if (out) {
    std::vector<uint8_t> px(size_t(1920) * 1080 * 8);
    [out getBytes:px.data() bytesPerRow:1920 * 8 fromRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0];
    out_sha = sha_hex(px.data(), px.size());
  }
  printf("{\"schema\": \"neuralstudio.nr.reset-worker-gate.v1\", \"units_per_chunk\": %u, \"declared_max_encoders\": %u, \"declared_max_dispatches\": %u, \"declared_max_chunk_ms\": %.1f, \"body_emission_bound\": %u,\n", units, max_enc, max_disp, max_ms, emit);
  printf(" \"state\": \"%s\", \"chunks_done\": %u, \"chunks_total\": %u,\n",
         p.state == ResetState::Completed ? "completed" : p.state == ResetState::Failed ? "failed" : p.state == ResetState::Cancelled ? "cancelled" : "running",
         p.chunks_done, p.chunks_total);
  printf(" \"output_publishable\": %s, \"output_sha256\": \"%s\", \"error\": \"%s\",\n",
         p.output_publishable ? "true" : "false", out_sha.c_str(), p.error.c_str());
  printf(" \"worst_submission_gpu_ms\": %.3f, \"total_gpu_ms_all_submissions\": %.1f, \"submissions_recorded\": %zu, \"body_command_buffers\": %u,\n",
         p.worst_chunk_gpu_ms, p.total_gpu_ms, durs.size(), p.body_command_buffers);
  printf(" \"heartbeat_beats\": %d, \"worst_heartbeat_gap_ms\": %.3f,\n", beats.load(), worst_gap.load());
  printf(" \"worker_cpu\": {\"encode_ms\": %.1f, \"encode_calls\": %zu, \"wait_ms\": %.1f, \"wait_calls\": %zu},\n",
         nr_exp_encode_profile.worker_encode_ms, nr_exp_encode_profile.worker_encode_calls,
         nr_exp_encode_profile.worker_wait_ms, nr_exp_encode_profile.worker_wait_calls);
  printf(" \"prepare_wall_ms\": %.3f, \"steady_state_wall_ms\": %.3f, \"encoder_batching\": %s, \"gpu_timestamps\": %s, \"arithmetic\": \"%s\", \"wide_threadgroup\": \"%s\",\n \"weights_identity\": \"%s\",\n",
         prepare_wall_ms, steady_wall_ms,
         (getenv("NR_EXP_ENCODER_BATCH") && getenv("NR_EXP_ENCODER_BATCH")[0] == '0') ? "false" : (getenv("NR_EXP_GPU_TIMESTAMPS") && getenv("NR_EXP_GPU_TIMESTAMPS")[0] == '1' ? "false" : "true"),
         (getenv("NR_EXP_GPU_TIMESTAMPS") && getenv("NR_EXP_GPU_TIMESTAMPS")[0] == '1') ? "true" : "false",
         // The RUNTIME's own label, not the option we asked for. When a per-kernel override is active
         // this names it, so a substituted kernel can never read as the pinned configuration.
         pr.arithmetic_label.c_str(),   // carries the FULL override manifest digest when one is active
         getenv("NR_EXP_WIDE_TG") ? getenv("NR_EXP_WIDE_TG") : "64 (default)",
         pr.weights_identity.c_str());
  // Name the offender and every submission, so a withheld run says WHICH work was slow.
  {
    auto subs = reset_worker_submissions(job);
    const size_t ff = reset_worker_first_failure(job);
    printf(" \"first_failure_index\": %s, \"first_failure_label\": \"%s\", \"first_failure_ms\": %.6f,\n",
           ff == SIZE_MAX ? "null" : std::to_string(ff).c_str(),
           ff == SIZE_MAX ? "" : reset_worker_submission_label(subs[ff]).c_str(),
           ff == SIZE_MAX ? 0.0 : subs[ff].gpu_ms);
    printf(" \"dropped_duration_samples\": %u,\n", reset_worker_dropped_duration_samples(job));
    printf(" \"slowest_submissions\": [");
    std::vector<const ResetSubmission*> byms;
    for (const auto& r : subs) byms.push_back(&r);
    std::sort(byms.begin(), byms.end(), [](const ResetSubmission* a, const ResetSubmission* b){ return a->gpu_ms > b->gpu_ms; });
    for (size_t i = 0; i < byms.size() && i < 5; ++i)
      printf("%s{\"index\": %u, \"ms\": %.3f, \"label\": \"%s\"}", i ? ", " : "",
             byms[i]->index, byms[i]->gpu_ms, reset_worker_submission_label(*byms[i]).c_str());
    printf("],\n");
    // EVERY submission, in submission order, so the labels can be audited for coverage: the launch ranges
    // must tile the body's launch list with no gap and no overlap. A one-off attribution error would show
    // up here as a repeated or skipped launch number, which a top-5 list would hide.
    printf(" \"submissions\": [");
    for (size_t i = 0; i < subs.size(); ++i) {
      const ResetSubmission& r = subs[i];
      printf("%s{\"i\": %u, \"body\": %s, \"unit_lo\": %u, \"unit_hi\": %u, \"launch_lo\": %d, \"launch_hi\": %d, \"ms\": %.3f, \"status\": %d, \"over\": %s, \"label\": \"%s\"}",
             i ? ",\n   " : "", r.index, r.is_body ? "true" : "false", r.unit_lo, r.unit_hi,
             r.launch_lo == 0xffffffffu ? -1 : int(r.launch_lo), r.launch_hi == 0xffffffffu ? -1 : int(r.launch_hi),
             r.gpu_ms, r.status, r.over_budget ? "true" : "false", reset_worker_submission_label(r).c_str());
    }
    printf("],\n");
  }
  printf(" \"chunk_gpu_ms\": [");
  for (size_t i = 0; i < durs.size(); ++i) printf("%s%.3f", i ? ", " : "", durs[i]);
  printf("]}\n");

  // ---- cancellation, on a FRESH worker with MANDATORY assertions ----
  // The previous version began a second job on the SAME worker, which single-use admission refuses, so the
  // whole check silently skipped. It now fails loudly instead of passing by omission.
  int cancel_failures = 0;
  {
    ResetWorkerPtr w2; Report p2;
    if (!reset_worker_prepare(dev, o, w2, p2)) { printf("{\"cancellation\": {\"prepare\": false, \"error\": \"%s\"}}\n", p2.error.c_str()); return 3; }
    ResetJobPtr j2; Report c2;
    if (!reset_worker_begin(w2, colour, j2, c2)) { printf("{\"cancellation\": {\"begin\": false, \"error\": \"%s\"}}\n", c2.error.c_str()); return 3; }
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    reset_worker_cancel(j2);
    ResetProgress q;
    for (;;) { q = reset_worker_poll(j2); if (q.state != ResetState::Running) break; std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    const bool st_ok = q.state == ResetState::Cancelled;
    const bool pub_ok = !q.output_publishable;
    const bool nil_ok = reset_worker_output(j2) == nil;
    const bool part_ok = q.chunks_done > 0 && q.chunks_done < q.chunks_total;
    cancel_failures = (!st_ok) + (!pub_ok) + (!nil_ok) + (!part_ok);
    printf("{\"cancellation\": {\"state_cancelled\": %s, \"not_publishable\": %s, \"output_nil\": %s, \"stopped_partway\": %s, \"chunks_done\": %u, \"assert_failures\": %d}}\n",
           st_ok ? "true" : "false", pub_ok ? "true" : "false", nil_ok ? "true" : "false", part_ok ? "true" : "false", q.chunks_done, cancel_failures);
  }
  // ---- single-use admission must REFUSE a second job on the same worker ----
  {
    ResetJobPtr j3; Report c3;
    const bool refused = !reset_worker_begin(worker, colour, j3, c3);
    printf("{\"single_use_admission\": {\"second_begin_refused\": %s, \"reason\": \"%s\"}}\n", refused ? "true" : "false", c3.error.c_str());
    if (!refused) ++cancel_failures;
  }
  reset_worker_release(job);
  return (p.state == ResetState::Completed && cancel_failures == 0) ? 0 : 1;
} }
