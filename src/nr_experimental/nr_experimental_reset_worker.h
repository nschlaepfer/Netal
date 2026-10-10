#pragma once
// Background-only, bounded-chunk NR reset executor (Opus-owned).
//
// This is a SEPARATE api from the encode-only one in nr_experimental_api.h, which is unchanged and stays
// the path for a caller-owned command buffer. This worker exists for one purpose: run a single reset frame
// that is too long for one command buffer, on our OWN queue, split into bounded submissions, without ever
// blocking the game thread or the game's completion callback.
//
// Why a new API and not a chunked version of encode(): the encode-only contract promises no commit and no
// wait on the caller's buffer. Chunking necessarily commits. Rather than violate that contract silently,
// chunking lives here, where committing is the explicit, documented behaviour.
//
// Safety shape, all enforced by the implementation:
//   * The caller hands over a model-colour texture it owns and whose producer has ALREADY completed. The
//     worker retains it until the job reaches a terminal state.
//   * At most ONE chunk is in flight. Chunk N+1 is enqueued from chunk N's completion handler on our own
//     queue - never from the game thread, and never by waiting on one.
//   * Scratch, intermediate dependencies, original weights, required outputs and every resource lifetime
//     are preserved ACROSS chunks: the chunks are one frame, not independent frames.
//   * Cancellation and error checks happen BETWEEN chunks. An already-submitted chunk is allowed to finish.
//   * Nothing is publishable after a partial, cancelled or failed run: `output()` returns nil unless the
//     job completed with every chunk applied.
//   * Reset-only. No history is consumed, and history advances only on full success.
#include "nr_experimental_api.h"   // shared types only; that header is Codex-owned and unmodified
#include <memory>
#include <string>
#include <vector>

namespace netal_nr_experimental {

struct ResetWorker;  using ResetWorkerPtr = std::shared_ptr<ResetWorker>;
struct ResetJob;     using ResetJobPtr    = std::shared_ptr<ResetJob>;

enum class ResetState { Idle, Running, Completed, Failed, Cancelled };

struct ResetOptions {
  std::string runtime_bundle;
  Arithmetic arithmetic = Arithmetic::OriginalKernels;   // original kernels first
  // Bounded submission. The worker closes a chunk at whichever bound is hit first, and refuses to start
  // if a single indivisible dispatch cannot fit `max_dispatches_per_chunk`, rather than silently
  // submitting an unbounded command.
  // Exactly this many NATURAL UNITS per submission (default 1). Units are never merged to fill a budget.
  uint32_t max_units_per_chunk = 1;
  // ENFORCED CEILINGS, not aggregation targets: if the unit selected for a submission contains more
  // dispatches or encoders than these, the job FAILS and names that unit. They can never cause two units
  // to be combined into one command buffer.
  uint32_t max_dispatches_per_chunk = 64;
  uint32_t max_encoders_per_chunk = 64;
  // Measured stop budget. After each chunk completes, a chunk whose ACTUAL GPU duration exceeded this
  // fails the job between chunks. 0 disables the budget.
  double max_chunk_gpu_ms = 0;
  // How many encoders the BODY may emit into one command buffer before it rotates to a fresh one. This is
  // the emission granularity, deliberately separate from the ceiling above: the ceiling is what we refuse,
  // this is what we aim for. 0 disables rotation (one command for the whole body).
  uint32_t body_emission_encoder_bound = 0;
};

// Per-unit accounting as actually measured. `dispatches_known` is false for body units, because the body
// does not expose a per-unit dispatch counter: the worker reports that honestly and enforces only the
// encoder ceiling there rather than presenting an undercount as an enforced dispatch ceiling.
struct ResetUnitCounts { uint32_t encoders = 0, dispatches = 0; bool dispatches_known = true; };

struct ResetProgress {
  ResetState state = ResetState::Idle;
  uint32_t chunks_total = 0, chunks_done = 0;
  double last_chunk_gpu_ms = 0, worst_chunk_gpu_ms = 0, total_gpu_ms = 0;
  uint32_t body_command_buffers = 0;   // command buffers the body emitted and we waited on individually
  bool output_publishable = false;   // true only when state == Completed and every chunk was applied
  std::string error;                 // non-empty only when state == Failed
};

// Prepare once, off the game thread. Loads pinned assets and builds the worker's own queue.
bool reset_worker_prepare(id<MTLDevice> device, const ResetOptions& options, ResetWorkerPtr& out, Report& report);

// Start ONE reset frame. Returns as soon as the first chunk is enqueued; never waits for the GPU.
// `color` must be 1920x1080 RGBA16Float, owned by the caller, with its producer already completed.
bool reset_worker_begin(const ResetWorkerPtr& worker, id<MTLTexture> color, ResetJobPtr& job, Report& report);

// Safe to call from any thread at any time. Reads a snapshot; never blocks on GPU work.
ResetProgress reset_worker_poll(const ResetJobPtr& job);

// Requests cancellation. Takes effect between chunks; the in-flight chunk still completes and its
// resources are still released normally. After this the job can only reach Cancelled or Failed.
void reset_worker_cancel(const ResetJobPtr& job);

// The finished output surface, or nil unless poll().output_publishable is true.
id<MTLTexture> reset_worker_output(const ResetJobPtr& job);

// Drops the caller's reference. Resources are held until any in-flight chunk completes, so this is safe
// to call while the job is still running.
void reset_worker_release(ResetJobPtr& job);

// Per-chunk GPU durations in submission order, for the offline gate's receipts.
std::vector<double> reset_worker_chunk_durations(const ResetJobPtr& job);

// One record per SUBMITTED command buffer, in submission order, so a run names exactly which work was
// slow or failed instead of leaving it to be inferred from timing. Frontend submissions carry the unit
// range they encoded; body submissions carry the launch range the body put into that buffer.
//
// This is a NEW type reached through a NEW accessor. No existing public struct gains, loses or moves a
// field - v4 shifted ResetProgress and broke a compiled consumer, and that must not recur.
struct ResetSubmission {
  uint32_t index = 0;              // submission order, 0-based
  bool is_body = false;            // false: frontend units; true: a body command buffer
  uint32_t unit_lo = 0, unit_hi = 0;         // frontend only: units [unit_lo, unit_hi), hi EXCLUSIVE
  // Body only: the first and last launch ACTUALLY ENCODED into this buffer, both INCLUSIVE. A boundary
  // runs at a launch HEAD, before any of the incoming launch is encoded, so the buffer being committed
  // ends at the PREVIOUS launch; reporting the incoming one here would name a launch the buffer does not
  // contain. 0xffffffff in either field means the buffer carried no launch at all (setup encoders only).
  uint32_t launch_lo = 0, launch_hi = 0;
  // Encoders actually put into this buffer. Read this together with the launch range: a boundary can
  // only be placed at a LAUNCH HEAD, so a buffer holds its launches PLUS any work encoded after the last
  // launch head and before the next one. The span from launch 57 to launch 100 is the live example - it
  // carries repack 58, ViT blocks 31..38 and repack 99 as well as launch 57, and no emission bound can
  // split it because there is no launch head inside it. A high encoder count against a short launch
  // range is how that shows up. Codex, 05:24.
  uint32_t encoders = 0;
  double gpu_ms = 0;               // GPU duration; 0 when the buffer did not complete
  int status = 0;                  // MTLCommandBufferStatus as reported
  bool over_budget = false;        // this submission is the one that exceeded the stop budget
};

// Every submitted buffer, in order. Safe to call at any time; a snapshot is returned.
std::vector<ResetSubmission> reset_worker_submissions(const ResetJobPtr& job);

// Index into reset_worker_submissions of the FIRST submission that failed or exceeded the budget, or
// SIZE_MAX when none did. This is what names the offender; do not infer it from the largest duration.
size_t reset_worker_first_failure(const ResetJobPtr& job);

// A short human-readable label for one record, e.g. "frontend units 68..69 (encoder block 1)" or
// "body launches 133..137".
std::string reset_worker_submission_label(const ResetSubmission& s);

// Submissions whose duration could not be recorded. MUST be 0; a non-zero value means the durations from
// that run are incomplete and its reported maximum cannot be trusted.
//
// Deliberately a separate accessor, not a ResetProgress member: adding a field mid-struct in v4 shifted
// body_command_buffers, output_publishable and error, and a consumer compiled against the v3 header then
// read `publishable` from the wrong offset. ResetProgress's layout is now frozen and checked by
// nr_experimental_abi_check.
// CPU-only regression of the FAILED-submission recording path, including its locking. Returns false and
// sets `why` on a defect; a reintroduced double-lock would instead never return, which the gate's
// watchdog reports. Needs no device and does no GPU work. Added in v8 after a real double-lock deadlock
// shipped in v6 and v7 on exactly this path.
bool reset_worker_selftest_failure_record(std::string& why);

uint32_t reset_worker_dropped_duration_samples(const ResetJobPtr& job);

#ifdef NR_EXPERIMENTAL_WARM_JOBS
// MEASUREMENT CANDIDATE ONLY (build/nr-accept-v1/experiments/WARM-INSTRUMENT-v1.json). A new function, so no existing struct
// changes layout. Reopens admission on `worker` after `completed_job` - a job of this worker - reached Completed with every
// chunk applied, so an offline gate can time consecutive reset frames on ONE resident context. The caller must have finished
// reading the previous output first: the next job writes the same context storage. Refuses (false + reason) otherwise.
bool reset_worker_reopen_admission(const ResetWorkerPtr& worker, const ResetJobPtr& completed_job, Report& report);
#endif

}  // namespace netal_nr_experimental
