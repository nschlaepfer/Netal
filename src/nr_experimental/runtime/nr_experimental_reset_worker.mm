// Bounded-chunk background reset executor (Opus-owned). Contract: nr_experimental_reset_worker.h.
// It lives in its OWN translation unit precisely because it commits: the encode-only path's file must keep
// its machine-checked property of never committing, never installing a completion handler and never
// creating a queue.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../nr_experimental_reset_worker.h"
#include "nr_experimental_context.h"
#include "nr_experimental_boundary_diag.h"
#include "nr_experimental_encoder_scope_adoption.h"
#include <atomic>
#include <cstdio>
#include <vector>
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
#include <mach/mach_time.h>
#endif

namespace netal_nr_experimental {

// ===================================================================================================
// Bounded-chunk background reset executor. Contract in nr_experimental_reset_worker.h. Implemented in
// this translation unit because Context (which owns every persistent buffer) is defined here.
//
// Unit numbering is FIXED and declared, so "one natural unit per submission" is checkable rather than an
// implementation detail:
//   [0, 68)        block 0, one 512-window chunk each; unit 0 also carries the input blit and the
//                  feature step, because later units read what it writes
//   [68, 90)       encoder blocks 1..22, one block each; block 1 also carries the entry layout
//   [90, 91)       the body (split_a + ViT + split_b + decoder) as ONE unit in this version
// max_dispatches_per_chunk / max_encoders_per_chunk are ENFORCED CEILINGS, never aggregation targets: a
// unit that exceeds them fails the job with that unit named. They can never cause two units to be merged.
struct ResetWorker {
  std::atomic<bool> job_live{false};   // single-job admission: one job per worker at a time
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;      // the worker's OWN queue; never the game's
  ContextPtr ctx;
  ResetOptions options;
};

static constexpr uint32_t kB0Units = 68, kEncBlocks = 22, kBodyUnits = 1;
// Measured: the body emits 391 encoders. Used only to size bookkeeping, never to bound work.
static constexpr uint32_t kBodyEncoderUpperBound = 512;
static constexpr uint32_t kTotalUnits = kB0Units + kEncBlocks + kBodyUnits;

struct ResetJob {
  std::mutex lock;
  ResetWorkerPtr worker;
  id<MTLTexture> color = nil;
  std::atomic<bool> cancel_requested{false};
  ResetState state = ResetState::Idle;
  uint32_t next_unit = 0, chunks_done = 0, chunks_total = 0;
  double last_ms = 0, worst_ms = 0, total_ms = 0;
  std::vector<double> chunk_ms;
  std::vector<ResetSubmission> subs;   // one per submitted buffer, reserved before submission
  size_t first_failure = SIZE_MAX;
  std::string error;
  int error_code = 0;          // set in the driver callback (no allocation); rendered to text by poll()
  double over_budget_ms = 0;
  bool finished_all_units = false;
  uint32_t body_buffers = 0;
  uint32_t dropped_samples = 0;   // must stay 0; non-zero means the reservation was wrong        // extra command buffers the body emitted under bounded emission
  bool finalize_started = false;        // finalize exactly once; also the cancel/finalize race boundary
  dispatch_queue_t host = nullptr;      // owned serial queue: next-chunk CPU encoding never runs inside a driver callback
  NrBodyPending pending{};
  NrBodyResult body_result{};
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  // WALL-DECOMPOSITION diagnostic only: host seconds recorded by the completion handler (no allocation there); the host thread accumulates.
  double wp_commit_t = 0, wp_commit_dur_ms = 0, wp_gpu_start = 0, wp_gpu_end = 0, wp_handler_t = 0;
  bool wp_pending = false;
#endif
};

#ifdef NR_EXPERIMENTAL_WALL_PROFILE
static double wp_host_s() {   // the host clock that GPUStartTime / GPUEndTime are reported in (mach absolute time, seconds)
  static const double k = [] { mach_timebase_info_data_t tb; mach_timebase_info(&tb); return double(tb.numer) / double(tb.denom) * 1e-9; }();
  return double(mach_absolute_time()) * k;
}
// Host-queue side: consume what the completion handler recorded for the previous chunk. The caller holds j.lock.
static void wp_consume_locked(ResetJob& j, double now) {
  nr_exp_encode_profile.wp_fe_commit_ms += j.wp_commit_dur_ms; j.wp_commit_dur_ms = 0;
  if (!j.wp_pending) return;
  nr_exp_encode_profile.wp_fe_sched_ms += (j.wp_gpu_start - j.wp_commit_t) * 1000.0;
  nr_exp_encode_profile.wp_fe_handler_ms += (j.wp_handler_t - j.wp_gpu_end) * 1000.0;
  nr_exp_encode_profile.wp_fe_hop_ms += (now - j.wp_handler_t) * 1000.0;
  nr_exp_encode_profile.wp_fe_submissions += 1;
  j.wp_pending = false;
}
#endif

static uint32_t reset_units_per_chunk(const ResetOptions& o) { return o.max_units_per_chunk ? o.max_units_per_chunk : 1u; }

// Record a FAILED submission. THE CALLER MUST ALREADY HOLD j.lock: j.lock is a plain std::mutex, and a
// second lock_guard here would deadlock on a driver thread at exactly the moment a command buffer had
// failed. That is the defect Codex found at 05:24 in v6/v7. This function exists so the path can be
// exercised on the CPU (reset_worker_selftest_failure_record) instead of only when a real GPU error
// happens - a deadlock here would hang the driver callback and never report anything at all.
static void reset_record_failure_locked(ResetJob& j, bool is_body, uint32_t lo, uint32_t hi,
                                        uint32_t launch_lo, uint32_t launch_hi, int status) {
  ResetSubmission rec; rec.index = uint32_t(j.subs.size());
  rec.is_body = is_body; rec.unit_lo = lo; rec.unit_hi = hi; rec.status = status;
  rec.launch_lo = launch_lo; rec.launch_hi = launch_hi;
  if (j.subs.size() < j.subs.capacity()) j.subs.push_back(rec); else j.dropped_samples += 1;
  if (j.first_failure == SIZE_MAX) j.first_failure = rec.index;
  j.state = ResetState::Failed; j.error_code = 1;
}

bool reset_worker_prepare(id<MTLDevice> device, const ResetOptions& options, ResetWorkerPtr& out, Report& report) {
  out.reset();
  if (!device) { report.error = "no device"; return false; }
  auto w = std::make_shared<ResetWorker>();
  w->device = device; w->options = options;
  w->queue = [device newCommandQueue];
  if (!w->queue) { report.error = "could not create the worker's command queue"; return false; }
  w->queue.label = @"nr_exp.reset_worker";
  PrepareOptions po; po.runtime_bundle = options.runtime_bundle; po.arithmetic = options.arithmetic;
  if (!prepare(device, po, w->ctx, report)) return false;
  out = w;
  return true;
}

// Encode exactly the units [lo, hi) into `cb`. Everything they touch lives in the context, so a later
// chunk resumes from persistent state; only the ENCODING is bounded.
static bool reset_encode_units(const ResetJobPtr& job, id<MTLCommandBuffer> cb, uint32_t lo, uint32_t hi,
                               uint32_t& dispatches, uint32_t& encoders, bool& body_dispatches_unknown, std::string& error) {
  body_dispatches_unknown = false;
  ResetWorker& w = *job->worker; Context& c = *w.ctx;
  dispatches = 0; encoders = 0;
  if (lo < kB0Units) {
    const uint32_t b_hi = hi < kB0Units ? hi : kB0Units;
    nr_exp::Block0Encoded b0{};
    if (!nr_exp::encode_block0(cb, c.assets, c.block0, job->color, 0u, b0, error, nil, nil, lo, b_hi)) return false;
    dispatches += b0.dispatches; encoders += b0.compute_encoders;
  }
  if (hi > kB0Units && lo < kB0Units + kEncBlocks) {
    const uint32_t e_lo = lo > kB0Units ? lo - kB0Units : 0u;
    const uint32_t e_hi_u = hi - kB0Units;
    const uint32_t e_hi = e_hi_u < kEncBlocks ? e_hi_u : kEncBlocks;
    nr_exp::EncoderEncoded e22{};
    if (!nr_exp::encode_encoder(cb, c.assets, c.block0, c.encoder, e22, error, 0xffffffffu, e_lo + 1u, e_hi + 1u)) return false;
    dispatches += e22.dispatches; encoders += e22.compute_encoders;
  }
  if (hi > kB0Units + kEncBlocks) {
    NrBodyFrameInputs bin;
    auto region = [](const nr_exp::Buf& b) {
      NrBodyRegion r; r.buffer = b.metal; r.offset = nr_exp::Buf::kGuard; r.bytes = b.bytes; r.producer = "experimental-encoder"; return r;
    };
    bin.entry = region(c.encoder.entry);
    for (size_t i = 0; i < 4 && i < c.encoder.skips.size(); ++i) bin.skips[i] = region(c.encoder.skips[i]);
    bin.block0_skip = region(c.block0.skip);
    bin.input_color = job->color;
    bin.output_surface = c.output;
    bin.production = true;
    bin.reset = true;                 // reset-only worker
    bin.motion = nil;
    bin.temporal = &c.temporal;
    // Bounded emission: the body commits at launch boundaries and asks us for the next buffer, so the
    // body unit becomes many bounded submissions instead of one multi-second command. Ordering holds
    // because every buffer is committed on THIS queue in order.
    // Scoped to THIS call and restored on every exit, including exceptions: the hook lives on our own
    // context, never process-wide, so no other path - least of all the encode-only API - can inherit it.
    struct BoundaryScope {
      NrBodyContext& b;
      std::function<id<MTLCommandBuffer>(id<MTLCommandBuffer>, uint32_t, uint32_t, uint32_t)> saved_next;
      uint32_t saved_bound, saved_ceiling; id<MTLCommandBuffer> saved_external;
      ~BoundaryScope() {   // restored on EVERY exit, including C++ and ObjC unwinding
        b.nr_chunk_next = saved_next; b.nr_chunk_encoder_bound = saved_bound;
        b.nr_chunk_encoder_ceiling = saved_ceiling; b.external_command = saved_external;
      }
    } boundary_scope{c.body, c.body.nr_chunk_next, c.body.nr_chunk_encoder_bound,
                     c.body.nr_chunk_encoder_ceiling, c.body.external_command};
    // Assigned AFTER the snapshot, so the RAII restore puts back the PRIOR value rather than this buffer.
    c.body.external_command = cb;
    {
      ResetWorker* wp = job->worker.get();
      ResetJob* jp = job.get();
      c.body.nr_chunk_encoder_bound = wp->options.body_emission_encoder_bound;
      c.body.nr_chunk_encoder_ceiling = wp->options.max_encoders_per_chunk;
      c.body.nr_chunk_next = [wp, jp](id<MTLCommandBuffer> just, uint32_t llo, uint32_t lhi, uint32_t lenc) -> id<MTLCommandBuffer> {
        // Runs on the worker's OWN host thread (the body encode for the last unit is dispatched there),
        // never on a game or render thread and never in a game completion callback. We establish the just
        // submitted buffer's GENUINE completion here, so at most one body buffer is ever in flight.
        if (just) {
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
          const double wp_e = wp_host_s();
#endif
          const auto t_w0 = std::chrono::steady_clock::now();
          [just waitUntilCompleted];
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
          nr_exp_encode_profile.worker_wait_ms +=
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_w0).count();
          nr_exp_encode_profile.worker_wait_calls += 1;
#endif
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
          nr_exp_encode_profile.wp_body_sched_ms += (just.GPUStartTime - wp_e) * 1000.0;
          nr_exp_encode_profile.wp_body_tail_ms += (wp_host_s() - just.GPUEndTime) * 1000.0;
          nr_exp_encode_profile.wp_body_rotations += 1;
#endif
          const bool bad = just.status != MTLCommandBufferStatusCompleted;
          const double ms = bad ? 0.0 : (just.GPUEndTime - just.GPUStartTime) * 1000.0;
          std::lock_guard<std::mutex> g(jp->lock);
          ++jp->body_buffers;
          jp->chunk_ms.push_back(ms);                 // EVERY submission is recorded, not just the tail
          jp->total_ms += ms;
          if (ms > jp->worst_ms) jp->worst_ms = ms;
          ResetSubmission rec; rec.index = uint32_t(jp->subs.size()); rec.is_body = true;
          rec.launch_lo = llo; rec.launch_hi = lhi; rec.encoders = lenc; rec.gpu_ms = ms; rec.status = int(just.status);
          const double budget = wp->options.max_chunk_gpu_ms;
          const bool over = budget > 0 && ms > budget;
          rec.over_budget = over;
          if (jp->subs.size() < jp->subs.capacity()) jp->subs.push_back(rec); else jp->dropped_samples += 1;
          if ((bad || over) && jp->first_failure == SIZE_MAX) jp->first_failure = rec.index;
          if (bad) { jp->state = ResetState::Failed; jp->error_code = 1; return nil; }
          if (over) { jp->state = ResetState::Failed; jp->error_code = 2; jp->over_budget_ms = ms; return nil; }
          if (jp->cancel_requested.load()) { jp->state = ResetState::Cancelled; return nil; }
        }
        id<MTLCommandBuffer> nb = [wp->queue commandBuffer];
        nb.label = @"nr_exp.reset_body_chunk";
        return nb;
      };
    }
    const int rc = nr_body_encode(c.body, c.descriptor, bin, nullptr, job->pending);
    if (rc != 0) { error = "body encode returned " + std::to_string(rc); return false; }
    // Real counts, not a placeholder: the body reports its own encoder inventory. It does NOT expose a
    // per-unit dispatch count, so we say so instead of presenting an undercount as an enforced ceiling.
    // With bounded emission the body's encoders are spread across the buffers it committed plus this one,
    // so the ceiling applies per emitted buffer. Report the per-buffer average as the accounted count and
    // keep the dispatch count declared unknown.
    // The ACTUAL maximum encoders in any one buffer the body emitted - not an average, not a total.
    encoders += job->pending.res.nr_max_encoders_per_buffer;
    body_dispatches_unknown = true;
  }
  return true;
}

// Units contained in one chunk, and the per-unit dispatch/encoder counts we will enforce against the
// declared ceilings. Counted by encoding into a throwaway buffer would cost a submission, so instead the
// encode itself reports the counts and the job fails AFTER encoding but BEFORE committing if a ceiling is
// exceeded - nothing is submitted in that case.
static void reset_submit_next(const ResetJobPtr& job);
static void reset_finalize(const ResetJobPtr& job);

static void reset_fail_from_exception(const ResetJobPtr& job, const char* what);
static void reset_finalize_inner(const ResetJobPtr& job) {
  ResetWorker& w = *job->worker; Context& c = *w.ctx;
  int rc = 0;
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  const auto wp_f0 = std::chrono::steady_clock::now();
#endif
  try { rc = nr_body_complete(job->pending, job->body_result); }
  catch (const std::exception& e) { reset_fail_from_exception(job, e.what()); return; }
  catch (...) { reset_fail_from_exception(job, "unknown C++ exception in body completion"); return; }
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  nr_exp_encode_profile.worker_finalize_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wp_f0).count();
  nr_exp_encode_profile.wp_finalize_calls += 1;
#endif
  std::lock_guard<std::mutex> g(job->lock);
  if (rc != 0) { job->state = ResetState::Failed; job->error = "body completion returned " + std::to_string(rc) + ": " + job->body_result.error; return; }
  job->finished_all_units = true;
  job->state = ResetState::Completed;
  c.history_valid = job->body_result.temporal_state_advanced;
}

static void reset_submit_next_inner(const ResetJobPtr& job) {
  uint32_t lo = 0, hi = 0;
  {
    std::lock_guard<std::mutex> g(job->lock);
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
    wp_consume_locked(*job, wp_host_s());
#endif
    if (job->state != ResetState::Running) return;
    if (job->cancel_requested.load()) { job->state = ResetState::Cancelled; nr_body_pending_abandon(job->pending); return; }
    if (job->next_unit >= kTotalUnits) { /* fall through to finalize below */ }
    lo = job->next_unit;
    hi = lo + reset_units_per_chunk(job->worker->options);
    if (hi > kTotalUnits) hi = kTotalUnits;
  }
  if (lo >= kTotalUnits) {
    {   // the cancel-versus-finalize boundary: once finalize starts, cancellation can no longer flip the
        // state, and before it starts a cancel wins. Either way a partial job never exposes output.
      std::lock_guard<std::mutex> g(job->lock);
      if (job->finalize_started) return;
      if (job->cancel_requested.load()) { job->state = ResetState::Cancelled; nr_body_pending_abandon(job->pending); return; }
      job->finalize_started = true;
    }
    reset_finalize(job);
    return;
  }

  ResetWorker& w = *job->worker;
  id<MTLCommandBuffer> cb = [w.queue commandBuffer];
  if (!cb) { std::lock_guard<std::mutex> g(job->lock); job->state = ResetState::Failed; job->error = "no command buffer"; return; }
  cb.label = @"nr_exp.reset_chunk";
  uint32_t dispatches = 0, encoders = 0; std::string error;
  bool ok = false;
  bool body_dispatches_unknown = false;
  {
    // Per-unit encoder ownership: anything left open by a throwing unit is closed here, and both C++ and
    // Objective-C unwinding are caught so no exception escapes into the driver.
    nr_exp::ScopeHolder scopes;
    struct HolderInstall {
      HolderInstall(nr_exp::ScopeHolder* h) { nr_exp::set_current_scope_holder(h); }
      ~HolderInstall() { nr_exp::set_current_scope_holder(nullptr); }
    } holder_install{&scopes};
    const auto t_enc0 = std::chrono::steady_clock::now();
    try { ok = reset_encode_units(job, cb, lo, hi, dispatches, encoders, body_dispatches_unknown, error); }
    catch (const std::exception& e) { ok = false; error = e.what(); }
    catch (id ex) { ok = false; error = std::string("Objective-C exception: ") + [[ex description] UTF8String]; }
    catch (...) { ok = false; error = "unknown exception while encoding a chunk"; }
    if (!scopes.close_all() && ok) { ok = false; error = "an encoder could not be closed after encoding a unit"; }
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
    nr_exp_encode_profile.worker_encode_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc0).count();
    nr_exp_encode_profile.worker_encode_calls += 1;
#endif
  }
  if (!ok) {
    nr_body_pending_abandon(job->pending);   // never leave a half-built pending behind
    std::lock_guard<std::mutex> g(job->lock);
    // A boundary that stopped the run because of cancellation or a budget/status failure has ALREADY set
    // the terminal state and its reason. The encode then unwinds with a generic message; do not let that
    // overwrite the real cause.
    if (job->state == ResetState::Running) { job->state = ResetState::Failed; job->error = error; }
    return;
  }
  // Declared ceilings are enforced here, before any commit: exceeding one is an error naming the unit,
  // never a reason to merge or split units silently.
  const ResetOptions& o = w.options;
  if ((!body_dispatches_unknown && dispatches > o.max_dispatches_per_chunk) || encoders > o.max_encoders_per_chunk) {
    std::lock_guard<std::mutex> g(job->lock);
    job->state = ResetState::Failed;
    nr_body_pending_abandon(job->pending);
    job->error = "unit " + std::to_string(lo) + " needs " + std::to_string(dispatches) + " dispatches / " +
                 std::to_string(encoders) + " encoders, over the declared ceiling of " +
                 std::to_string(o.max_dispatches_per_chunk) + " / " + std::to_string(o.max_encoders_per_chunk) +
                 "; nothing was committed";
    return;
  }
  const bool chunk_has_body = hi > kB0Units + kEncBlocks;
  // Under bounded emission the body commits and rotates buffers mid-encode, so `cb` may already be
  // committed by now. The chunk's completion is the buffer the body ENDED on; the ones it rotated away
  // from were committed earlier on the same queue and therefore complete first.
  id<MTLCommandBuffer> final_cb = (chunk_has_body && job->pending.cmd) ? job->pending.cmd : cb;
  ResetJobPtr keep = job;   // the job (and everything it references) stays alive until this chunk completes
  [final_cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
    // Runs on a driver thread. It must not allocate, must not throw and must not do CPU encoding, so it
    // only records into already-reserved storage and hands the next chunk to our OWN serial host queue.
    @try {
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
      const double wp_entry = wp_host_s();
#endif
      const bool failed = done.status != MTLCommandBufferStatusCompleted;
      const double ms = failed ? 0.0 : (done.GPUEndTime - done.GPUStartTime) * 1000.0;
      bool go_on = false;
      {
        // ONE lock for the whole handler. keep->lock is a plain std::mutex, so a second lock_guard on
        // it inside the failed branch would DEADLOCK on a driver thread - and only when a command
        // buffer had already failed, which is exactly when this record has to be readable. That defect
        // was introduced with the submission record in v6 and shipped in v6 and v7; it is fixed here.
        std::lock_guard<std::mutex> g(keep->lock);
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
        keep->wp_gpu_start = done.GPUStartTime; keep->wp_gpu_end = done.GPUEndTime; keep->wp_handler_t = wp_entry; keep->wp_pending = true;
#endif
        if (failed) {
          reset_record_failure_locked(*keep, chunk_has_body, lo, hi,
                                      chunk_has_body ? keep->pending.res.nr_tail_launch_lo : 0u,
                                      chunk_has_body ? keep->pending.res.nr_tail_launch_hi : 0u,
                                      int(done.status));
        }
        else {
          // Reserved ahead of time so this does not allocate. If the reservation were ever wrong, record
          // that fact rather than silently dropping the sample the accounting exists to report.
          if (keep->chunk_ms.size() < keep->chunk_ms.capacity()) keep->chunk_ms.push_back(ms);
          else keep->dropped_samples += 1;
          ResetSubmission rec; rec.index = uint32_t(keep->subs.size());
          rec.is_body = chunk_has_body; rec.unit_lo = lo; rec.unit_hi = hi;
          if (chunk_has_body) { rec.launch_lo = keep->pending.res.nr_tail_launch_lo; rec.launch_hi = keep->pending.res.nr_tail_launch_hi; rec.encoders = keep->pending.res.nr_tail_encoders; }
          rec.gpu_ms = ms; rec.status = int(done.status);
          const double bud = keep->worker->options.max_chunk_gpu_ms;
          rec.over_budget = bud > 0 && ms > bud;
          if (keep->subs.size() < keep->subs.capacity()) keep->subs.push_back(rec); else keep->dropped_samples += 1;
          if ((failed || rec.over_budget) && keep->first_failure == SIZE_MAX) keep->first_failure = rec.index;
          keep->last_ms = ms; keep->total_ms += ms;
          if (ms > keep->worst_ms) keep->worst_ms = ms;
          ++keep->chunks_done; keep->next_unit = hi;
          const double budget = keep->worker->options.max_chunk_gpu_ms;
          if (budget > 0 && ms > budget) { keep->state = ResetState::Failed; keep->error_code = 2; keep->over_budget_ms = ms; }
          else go_on = keep->state == ResetState::Running;
        }
      }
      if (go_on) { ResetJobPtr again = keep; dispatch_async(keep->host, ^{ reset_submit_next(again); }); }
    } @catch (...) {
      std::lock_guard<std::mutex> g(keep->lock); keep->state = ResetState::Failed; keep->error_code = 3;
    }
  }];
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  { std::lock_guard<std::mutex> g(job->lock); job->wp_commit_t = wp_host_s(); }
  [final_cb commit];
  { const double wp_c1 = wp_host_s(); std::lock_guard<std::mutex> g(job->lock); job->wp_commit_dur_ms += (wp_c1 - job->wp_commit_t) * 1000.0; }
#else
  [final_cb commit];
#endif
  // The body appended to a buffer WE own and just committed, so tell its pending that the commit happened;
  // this only advances the pending's state machine, it does not submit anything.
  if (chunk_has_body) {
    const int crc = nr_body_commit(job->pending);
    if (crc != 0) { std::lock_guard<std::mutex> g(job->lock); job->state = ResetState::Failed; job->error = "body commit returned " + std::to_string(crc); }
  }
}

// Whole-task boundary. Every entry into the worker's state machine - the host-queue task, the encode, the
// commit and the finalize - is wrapped so that neither a C++ nor an Objective-C exception can escape into
// a driver callback or a dispatch queue. A job that trips this ends Failed with a stated reason rather
// than terminating the process, and its pending is abandoned.
static void reset_fail_from_exception(const ResetJobPtr& job, const char* what) {
  if (!job) return;
  nr_body_pending_abandon(job->pending);
  std::lock_guard<std::mutex> g(job->lock);
  if (job->state == ResetState::Running) { job->state = ResetState::Failed; job->error = std::string("unhandled exception: ") + what; }
}

static void reset_finalize(const ResetJobPtr& job) {
  @try { reset_finalize_inner(job); }
  @catch (id ex) { reset_fail_from_exception(job, [[ex description] UTF8String]); return; }
  @catch (...) { reset_fail_from_exception(job, "unknown Objective-C exception in finalize"); return; }
}

static void reset_submit_next(const ResetJobPtr& job) {
  try {
    @try { reset_submit_next_inner(job); }
    @catch (id ex) { reset_fail_from_exception(job, [[ex description] UTF8String]); }
    @catch (...) { reset_fail_from_exception(job, "unknown Objective-C exception in a submission step"); }
  } catch (const std::exception& e) { reset_fail_from_exception(job, e.what()); }
    catch (...) { reset_fail_from_exception(job, "unknown C++ exception in a submission step"); }
}

bool reset_worker_begin(const ResetWorkerPtr& worker, id<MTLTexture> color, ResetJobPtr& job, Report& report) {
  job.reset();
  if (!worker || !worker->ctx) { report.error = "worker not prepared"; return false; }
  // One model frame per job: reset the opt-in capture HERE, not in encode_encoder, which this worker
  // calls once per unit chunk.
  worker->ctx->encoder.qkv_capture_bytes = 0; worker->ctx->encoder.qkv_capture_done = false;
  if (!color) { report.error = "no colour texture"; return false; }
  if (color.width != 1920 || color.height != 1080 || color.pixelFormat != MTLPixelFormatRGBA16Float) {
    report.error = "colour texture must be 1920x1080 RGBA16Float"; return false; }
  if (color.device != worker->device) { report.error = "colour texture belongs to a different device"; return false; }
  bool expected = false;
  if (!worker->job_live.compare_exchange_strong(expected, true)) {
    report.error = "this worker has already been used; it is single-use because its output texture and scratch are context storage a second job would alias - create a fresh worker per reset frame";
    return false;
  }
  auto j = std::make_shared<ResetJob>();
  j->worker = worker; j->color = color;
  j->state = ResetState::Running;
  j->chunks_total = (kTotalUnits + reset_units_per_chunk(worker->options) - 1) / reset_units_per_chunk(worker->options);
  // Sized for the REAL submission count, not the chunk count: the body rotates a command buffer roughly
  // every `body_emission_encoder_bound` encoders, and each rotation is its own submission. Under-reserving
  // here used to make the completion handler's no-allocation guard silently DROP samples once capacity was
  // exceeded, so the accounting could understate the maximum it exists to report.
  {
    const uint32_t bound = worker->options.body_emission_encoder_bound;
    const uint32_t body_est = bound ? (kBodyEncoderUpperBound / bound) + 8u : 1u;
    j->chunk_ms.reserve(j->chunks_total + body_est + 8u);
    j->subs.reserve(j->chunks_total + body_est + 8u);
  }
  j->host = dispatch_queue_create("nr_exp.reset_worker.host", DISPATCH_QUEUE_SERIAL);
  job = j;
  reset_submit_next(j);
  return true;
}

ResetProgress reset_worker_poll(const ResetJobPtr& job) {
  ResetProgress p;
  if (!job) return p;
  std::lock_guard<std::mutex> g(job->lock);
  p.state = job->state; p.chunks_total = job->chunks_total; p.chunks_done = job->chunks_done;
  p.last_chunk_gpu_ms = job->last_ms; p.worst_chunk_gpu_ms = job->worst_ms; p.total_gpu_ms = job->total_ms;
  p.output_publishable = (job->state == ResetState::Completed) && job->finished_all_units;
  // body_buffers counts ROTATIONS; the body also submits the buffer it ends on, so submissions = +1.
  p.body_command_buffers = job->state == ResetState::Idle ? 0u : job->body_buffers + 1u;
  p.error = job->error;
  if (p.error.empty()) switch (job->error_code) {   // rendered here, never allocated in the driver callback
    case 1: p.error = "a chunk did not reach Completed"; break;
    case 2: p.error = "chunk GPU duration " + std::to_string(job->over_budget_ms) + " ms exceeded the declared stop budget"; break;
    case 3: p.error = "an exception escaped inside the completion callback"; break;
    default: break;
  }
  return p;
}

void reset_worker_cancel(const ResetJobPtr& job) { if (job) job->cancel_requested.store(true); }

id<MTLTexture> reset_worker_output(const ResetJobPtr& job) {
  if (!job) return nil;
  std::lock_guard<std::mutex> g(job->lock);
  if (job->state != ResetState::Completed || !job->finished_all_units) return nil;   // never after partial/failed
  return job->worker->ctx->output;
}

void reset_worker_release(ResetJobPtr& job) {
  // Deliberately does NOT reopen admission. A worker is SINGLE-USE for this milestone: its output texture
  // and scratch are context storage that a second job would alias, and releasing the handle says nothing
  // about whether the GPU work has finished. Callers make a fresh worker per reset frame.
  job.reset();
}
#ifdef NR_EXPERIMENTAL_WARM_JOBS
// Candidate WARM-INSTRUMENT-v1 (measurement only; see the header). Admission reopens only for a job of THIS worker that
// completed with every chunk applied: finalize has run, so the body's completion (and its GPU wait) is behind us and no
// chunk of that job is in flight. Failed or cancelled jobs never reopen admission.
bool reset_worker_reopen_admission(const ResetWorkerPtr& worker, const ResetJobPtr& completed_job, Report& report) {
  if (!worker || !completed_job) { report.error = "reopen admission: no worker or no job"; return false; }
  if (completed_job->worker != worker) { report.error = "reopen admission: the job belongs to another worker"; return false; }
  {
    std::lock_guard<std::mutex> g(completed_job->lock);
    if (completed_job->state != ResetState::Completed || !completed_job->finished_all_units) {
      report.error = "reopen admission: the previous job did not complete with every chunk applied"; return false; }
  }
  worker->job_live.store(false);
  return true;
}
#endif

std::vector<ResetSubmission> reset_worker_submissions(const ResetJobPtr& job) {
  if (!job) return {};
  std::lock_guard<std::mutex> g(job->lock);
  return job->subs;
}

size_t reset_worker_first_failure(const ResetJobPtr& job) {
  if (!job) return SIZE_MAX;
  std::lock_guard<std::mutex> g(job->lock);
  return job->first_failure;
}

std::string reset_worker_submission_label(const ResetSubmission& s) {
  char buf[224];
  // unit_hi is EXCLUSIVE (it is the chunk's end), launch_lo/launch_hi are INCLUSIVE (they are the first
  // and last launch actually encoded into the buffer). The label prints inclusive numbers in both cases,
  // so a reader never has to know which convention the field used.
  const uint32_t ulast = s.unit_hi > s.unit_lo ? s.unit_hi - 1u : s.unit_lo;
  if (s.is_body) {
    if (s.launch_lo == 0xffffffffu || s.launch_hi == 0xffffffffu)
      snprintf(buf, sizeof(buf), "body buffer with no launch in it (setup encoders only)");
    else if (s.launch_lo == s.launch_hi)
      snprintf(buf, sizeof(buf), "body launch %u and everything encoded before the next launch head, %u encoders", s.launch_lo, s.encoders);
    else
      snprintf(buf, sizeof(buf), "body launches %u..%u inclusive and everything encoded before the next launch head, %u encoders",
               s.launch_lo, s.launch_hi, s.encoders);
  }
  else if (ulast < 68u)
    snprintf(buf, sizeof(buf), "frontend units %u..%u (block 0 windows)", s.unit_lo, ulast);
  else if (s.unit_lo < 68u)
    snprintf(buf, sizeof(buf), "frontend units %u..%u (block 0 windows and encoder blocks 1..%u)",
             s.unit_lo, ulast, ulast - 67u);
  else
    snprintf(buf, sizeof(buf), "frontend units %u..%u (encoder blocks %u..%u)",
             s.unit_lo, ulast, s.unit_lo - 67u, ulast - 67u);
  return std::string(buf);
}

// CPU-only regression of the FAILED-submission path. Takes the lock exactly the way the driver
// completion handler does, then records through the same helper. If a second lock on the same mutex is
// ever reintroduced, this call never returns and the gate's watchdog reports it - instead of a real GPU
// error silently hanging a driver thread. No device, no queue and no GPU work are involved.
bool reset_worker_selftest_failure_record(std::string& why) {
  ResetJob j;
  j.subs.reserve(4);
  {
    std::lock_guard<std::mutex> g(j.lock);          // the handler's ONE lock
    reset_record_failure_locked(j, true, 90u, 91u, 155u, 156u,
                               int(MTLCommandBufferStatusError));   // 5, not 4 - 4 is Completed (Codex, 05:33)
  }
  std::lock_guard<std::mutex> g(j.lock);            // re-lockable afterwards, so nothing stayed held
  if (j.subs.size() != 1) { why = "the failed submission was not recorded"; return false; }
  if (j.first_failure != 0) { why = "first_failure was not set to the failed submission"; return false; }
  if (j.state != ResetState::Failed || j.error_code != 1) { why = "the job was not marked failed"; return false; }
  if (!j.subs[0].is_body || j.subs[0].launch_hi != 156u || j.subs[0].status != int(MTLCommandBufferStatusError)) { why = "the failed record lost its label or status"; return false; }
  if (j.dropped_samples != 0) { why = "a reserved record was dropped"; return false; }
  return true;
}

uint32_t reset_worker_dropped_duration_samples(const ResetJobPtr& job) {
  if (!job) return 0;
  std::lock_guard<std::mutex> g(job->lock);
  return job->dropped_samples;
}

std::vector<double> reset_worker_chunk_durations(const ResetJobPtr& job) {
  if (!job) return {};
  std::lock_guard<std::mutex> g(job->lock);
  return job->chunk_ms;
}


}  // namespace netal_nr_experimental
