// Warm bounded gate (MEASUREMENT CANDIDATE: build/nr-accept-v1/experiments/WARM-INSTRUMENT-v1.json). No game, no publication.
// One prepared worker runs JOBS consecutive reset frames on its resident context under exactly the bounds of
// nr_experimental_reset_worker_test: the same unit chunking, body emission bound, encoder/dispatch ceilings, per-submission
// stop budget and concurrent heartbeat on another queue. Job 0 is the COLD frame (it pays first touch of storage the body
// allocates lazily); jobs 1.. are WARM frames. One gate record per job, so the harness can report them separately.
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
#ifndef NR_EXPERIMENTAL_WARM_JOBS
  (void)argc; (void)argv;
  fprintf(stderr, "nr_experimental_reset_worker_warm_test was built without NR_EXPERIMENTAL_WARM_JOBS\n");
  return 2;
#else
  if (argc < 3) { fprintf(stderr, "usage: nr_experimental_reset_worker_warm_test BUNDLE INPUT.rgba16f [UNITS_PER_CHUNK] [MAX_ENCODERS] [MAX_DISPATCHES] [MAX_CHUNK_MS] [BODY_EMISSION_BOUND] [original|approx] [JOBS]\n"); return 2; }
  const uint32_t units = argc > 3 ? uint32_t(atoi(argv[3])) : 1u;
  const uint32_t max_enc = argc > 4 ? uint32_t(atoi(argv[4])) : 64u;
  const uint32_t max_disp = argc > 5 ? uint32_t(atoi(argv[5])) : 64u;
  const double max_ms = argc > 6 ? atof(argv[6]) : 0.0;
  const uint32_t emit = argc > 7 ? uint32_t(atoi(argv[7])) : 0u;
  const bool approx = argc > 8 && std::string(argv[8]) == "approx";
  const int jobs = argc > 9 ? atoi(argv[9]) : 4;
  if (jobs < 2 || jobs > 16) { fprintf(stderr, "JOBS must be 2..16 (job 0 is cold, the rest are warm)\n"); return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }

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
  const auto t_prep0 = std::chrono::steady_clock::now();
  if (!reset_worker_prepare(dev, o, worker, pr)) { printf("{\"prepared\": false, \"error\": \"%s\"}\n", pr.error.c_str()); return 3; }
  const double prepare_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prep0).count();

  // The same stand-in renderer as the cold gate: a 1920x1080 RGBA16F copy on ANOTHER queue every beat, for every job.
  id<MTLCommandQueue> hq = [dev newCommandQueue]; hq.label = @"nr_exp.heartbeat";
  std::atomic<bool> beating{true}; std::atomic<double> worst_gap{0}; std::atomic<int> beats{0};
  MTLTextureDescriptor* hd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  hd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; hd.storageMode = MTLStorageModePrivate;
  id<MTLTexture> ha = [dev newTextureWithDescriptor:hd], hb_tex = [dev newTextureWithDescriptor:hd];
  std::thread hb([&]{
    auto last = std::chrono::steady_clock::now();
    uint32_t beat = 0;
    while (beating.load()) {
      id<MTLCommandBuffer> c = [hq commandBuffer];
      id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
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

  int rc = 0;
  for (int k = 0; k < jobs; ++k) {
    nr_exp_encode_profile.reset();
    ResetJobPtr job; Report br;
    const int beats0 = beats.load(); worst_gap.store(0);
    const auto t_steady0 = std::chrono::steady_clock::now();
    if (!reset_worker_begin(worker, colour, job, br)) {
      printf("{\"schema\": \"neuralstudio.nr.reset-worker-gate-warm.v1\", \"job\": %d, \"jobs\": %d, \"state\": \"not-started\", \"error\": \"%s\"}\n", k, jobs, br.error.c_str());
      rc = 3; break;
    }
    ResetProgress p;
    for (;;) { p = reset_worker_poll(job); if (p.state != ResetState::Running) break; std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    const double steady_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_steady0).count();
    auto durs = reset_worker_chunk_durations(job);
    std::string out_sha = "";
    id<MTLTexture> out = reset_worker_output(job);
    if (out) {
      std::vector<uint8_t> px(size_t(1920) * 1080 * 8);
      [out getBytes:px.data() bytesPerRow:1920 * 8 fromRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0];
      out_sha = sha_hex(px.data(), px.size());
    }
    printf("{\"schema\": \"neuralstudio.nr.reset-worker-gate-warm.v1\", \"job\": %d, \"jobs\": %d, \"cold\": %s, \"units_per_chunk\": %u, \"declared_max_encoders\": %u, \"declared_max_dispatches\": %u, \"declared_max_chunk_ms\": %.1f, \"body_emission_bound\": %u,\n",
           k, jobs, k == 0 ? "true" : "false", units, max_enc, max_disp, max_ms, emit);
    printf(" \"state\": \"%s\", \"chunks_done\": %u, \"chunks_total\": %u,\n",
           p.state == ResetState::Completed ? "completed" : p.state == ResetState::Failed ? "failed" : p.state == ResetState::Cancelled ? "cancelled" : "running",
           p.chunks_done, p.chunks_total);
    printf(" \"output_publishable\": %s, \"output_sha256\": \"%s\", \"error\": \"%s\",\n",
           p.output_publishable ? "true" : "false", out_sha.c_str(), p.error.c_str());
    printf(" \"worst_submission_gpu_ms\": %.3f, \"total_gpu_ms_all_submissions\": %.1f, \"submissions_recorded\": %zu, \"body_command_buffers\": %u,\n",
           p.worst_chunk_gpu_ms, p.total_gpu_ms, durs.size(), p.body_command_buffers);
    printf(" \"heartbeat_beats\": %d, \"worst_heartbeat_gap_ms\": %.3f,\n", beats.load() - beats0, worst_gap.load());
    printf(" \"worker_cpu\": {\"encode_ms\": %.1f, \"encode_calls\": %zu, \"wait_ms\": %.1f, \"wait_calls\": %zu},\n",
           nr_exp_encode_profile.worker_encode_ms, nr_exp_encode_profile.worker_encode_calls,
           nr_exp_encode_profile.worker_wait_ms, nr_exp_encode_profile.worker_wait_calls);
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
    {
      const NrExpEncodeProfile& q = nr_exp_encode_profile;
      printf(" \"wall_profile\": {\"fe_submissions\": %zu, \"fe_commit_ms\": %.3f, \"fe_sched_ms\": %.3f, \"fe_handler_ms\": %.3f, \"fe_hop_ms\": %.3f, \"body_rotations\": %zu, \"body_sched_ms\": %.3f, \"body_tail_ms\": %.3f, \"finalize_calls\": %zu, \"finalize_ms\": %.3f, \"complete_ms\": [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]},\n",
             q.wp_fe_submissions, q.wp_fe_commit_ms, q.wp_fe_sched_ms, q.wp_fe_handler_ms, q.wp_fe_hop_ms, q.wp_body_rotations, q.wp_body_sched_ms, q.wp_body_tail_ms,
             q.wp_finalize_calls, q.worker_finalize_ms, q.wp_complete_ms[0], q.wp_complete_ms[1], q.wp_complete_ms[2], q.wp_complete_ms[3], q.wp_complete_ms[4], q.wp_complete_ms[5]);
    }
#endif
    printf(" \"prepare_wall_ms\": %.3f, \"steady_state_wall_ms\": %.3f, \"arithmetic\": \"%s\",\n \"weights_identity\": \"%s\",\n",
           prepare_wall_ms, steady_wall_ms, pr.arithmetic_label.c_str(), pr.weights_identity.c_str());
    {
      auto subs = reset_worker_submissions(job);
      const size_t ff = reset_worker_first_failure(job);
      printf(" \"first_failure_index\": %s, \"first_failure_label\": \"%s\", \"first_failure_ms\": %.6f,\n",
             ff == SIZE_MAX ? "null" : std::to_string(ff).c_str(),
             ff == SIZE_MAX ? "" : reset_worker_submission_label(subs[ff]).c_str(),
             ff == SIZE_MAX ? 0.0 : subs[ff].gpu_ms);
      printf(" \"dropped_duration_samples\": %u,\n", reset_worker_dropped_duration_samples(job));
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
    fflush(stdout);
    if (p.state != ResetState::Completed) { rc = 1; reset_worker_release(job); break; }
    if (k + 1 < jobs) {
      Report rr;
      if (!reset_worker_reopen_admission(worker, job, rr)) {
        printf("{\"schema\": \"neuralstudio.nr.reset-worker-gate-warm.v1\", \"job\": %d, \"jobs\": %d, \"state\": \"reopen-refused\", \"error\": \"%s\"}\n", k + 1, jobs, rr.error.c_str());
        rc = 4; reset_worker_release(job); break;
      }
    }
    reset_worker_release(job);
  }
  beating = false; hb.join();
  return rc;
#endif
} }
