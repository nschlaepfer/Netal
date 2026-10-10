// nr_body_runtime.h — v1 in-memory interface around the verified NR middle (launches 25-132) and decoder/output (133-156) execution core.
// The core never reads activation tensors or expected frames from the filesystem: every live input is a caller-provided Metal buffer/texture with
// validated byte length, offset and pixel format; a missing or mismatched input fails the frame (no recorded-tensor fallback). Model constants
// (libraries, weights, parameter bytes, RTX tables) come from the descriptor at prepare() time and stay resident across frames, as do pipelines.
// Diagnostics (poison, per-launch hashes/coverage, immutability, provenance) are preserved and reported in NrBodyResult; optional diagnostics_dir
// receives the same files the offline harness expects (chain-run.json, out_view_132.bin, tex_out_*.bin). Completion is synchronous per frame.
#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <utility>
struct NrTemporalState;
#include <set>
#include <vector>
struct NrBodyRegion { id<MTLBuffer> buffer = nil; size_t offset = 0; size_t bytes = 0; std::string provenance_sha256; std::string producer; };   // caller-provided live tensor
struct NrBodyFrameInputs {
  NrBodyRegion entry;            // encoder exit (block22-middle view), 1,105,920 bytes
  NrBodyRegion skips[4];         // C256 2,211,840 / C128 4,423,680 / C64 8,847,360 / C32 17,694,720 bytes (pool offsets from the descriptor)
  NrBodyRegion block0_skip;      // 70,778,880 bytes
  id<MTLTexture> input_color = nil;     // RGBA16Float 1920x1080, read-only during the frame
  id<MTLTexture> output_surface = nil;  // optional RGBA16Float 1920x1080 target for the final surface; created internally when nil
  uint8_t poison = 0xa5; bool production = false;
  const char* test_fault = nullptr;   // v3: fault-injection name honoured ONLY by builds compiled with -DNR_BODY_FAULT_TESTS; a production build rejects any non-null value
  // v5 EXPERIMENTAL temporal fields (ignored by v1-v4; the production reset-only route never sets them). See src/nr_temporal/nr_temporal.h.
  struct NrTemporalState* temporal = nullptr;   // shared persistent history state (internal texture); when set, the reference routing is used: 155 stores to the caller output, 156 copies output -> history
  id<MTLTexture> motion = nil;                  // caller RG16F WxH motion texture, read-only for the frame (required when reset == false)
  bool reset = true;                            // false = temporal frame: launch-155 record built by the verified rule, history/motion bound; requires the descriptor's lib_temporal and temporal continuity   // production: final-only decoder checks (per-launch pool hash/coverage scan skipped and listed)
};
struct NrBodyResult {
  int code = 2; std::string error;
  std::string middle_view_sha256, decoder_pre_image_sha256, pool_final_sha256, final_surface_sha256, input_texture_sha256_before, input_texture_sha256_after;
  size_t final_pixels = 0, final_pixels_not_written = 0, final_pixels_written_more_than_once = 0; bool input_texture_immutable = false;
  double middle_gpu_ms = 0, decoder_gpu_ms = 0, wall_ms = 0; std::vector<std::string> checks_skipped;
  // v3 additions: enforced-check accounting. code 3 = one or more computed checks failed (listed in check_failures); 2 = execution error / rejected input.
  size_t checks_enforced = 0, decoder_constant_mutations = 0, out_vit_padding_bytes = 0; std::vector<std::string> check_failures;
  // v3 execution-time provenance: every Metal library ("lib:<path>") and model constant ("file:<path>:<pad>") hashed when first loaded into this context,
  // and every resident constant re-hashed after the frame (constants_unchanged_after_frame). The adapter/harness adds executable and command identity.
  std::vector<std::pair<std::string, std::string>> provenance; bool constants_unchanged_after_frame = false; size_t constants_rehashed = 0;
  // v5 experimental temporal bookkeeping
  std::string temporal_record_sha256, motion_sha256, history_sha256_before, history_sha256_after;
  uint32_t ex2_half_arg_faults = 0, ex2_half_domain_faults = 0;
  size_t history_copy_value_differences = 0;
  bool temporal_state_advanced = false;         // v9: true only when the frame had no check failure and the actual history equals the caller output
  std::string damage;
  uint32_t nr_max_encoders_per_buffer = 0;
  uint32_t nr_tail_launch_lo = 0, nr_tail_launch_hi = 0;
  uint32_t nr_tail_encoders = 0;   // encoders in the buffer the body ended on   // launches carried by the buffer the body ended on   // experimental bounded emission: the ACTUAL maximum encoders emitted into any one command buffer
  uint32_t inventory_command_buffers = 0, inventory_waits = 0, inventory_blit_encoders = 0, inventory_compute_encoders = 0;
  // v12 resident evidence (docs/NR_RESIDENT_ENCODE_CONTRACT.md): every per-launch coverage check ran as a GPU validator with a completion record; every
  // intermediate output was digested on the GPU into a checkpoint table before any slice/buffer reuse; the CPU cross-checks the digest function at the frame boundaries
  uint32_t validators_expected = 0, validators_run = 0, coverage_faults = 0, digests_expected = 0, digests_complete = 0, digest_cpu_crosschecks = 0, digest_cpu_mismatches = 0;
  std::string digest_table_sha256; double frame_gpu_ms = 0;
  std::string decoder_pre_image_source, decoder_pre_image_expected_sha256;   // v13: "gpu-snapshot" (checked: actual restaged pool bytes) | "cpu-reconstruction (+gpu-digest-crosscheck)" (production); expected = CPU reconstruction from the same inputs   // digest table identity (must be identical across processes/poisons); whole-frame GPU time of the single command buffer   // v11i gate-0 inventory (counts only)                           // v11: non-empty when this frame damaged the context/temporal state (reason); recovery = fresh context + fresh state + reset frame   // v8: half values where the ACTUAL history after copy 156 differs from the caller output (must be 0; state advances only then)   // v6: half-domain EX2 table fault counters of the temporal post launch (both must be 0)
};
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
// Experimental lane only: per-frame CPU accounting for the body's encode path.
struct NrExpEncodeProfile { double fill_ms = 0; size_t fill_bytes = 0, fill_calls = 0, poison_bytes = 0, poison_calls = 0;
  // Coarse phase timing of nr_body_encode, in the order the phases run.
  double phase_ms[8] = {0,0,0,0,0,0,0,0};
  static const char* phase_name(int i) { static const char* n[8] = {"prologue", "tables", "entry+split", "repack58", "ViT", "repack99+split100", "decoder", "finish"}; return n[i]; }
  // Worker-level CPU accounting, so wall - GPU can be SPLIT into work and waiting instead of being
  // reported as one lump that looks removable. All measured on the worker's own host thread.
  double worker_encode_ms = 0;    // CPU actually building command buffers
  double worker_wait_ms = 0;      // blocked on waitUntilCompleted for a body buffer
  double worker_finalize_ms = 0;  // completion/finalise work after the last submission
  size_t worker_encode_calls = 0, worker_wait_calls = 0;
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  // WALL-DECOMPOSITION diagnostic (never an accepted lane): where a warm frame's wall time goes besides GPU execution.
  // Host-time intervals in ms, accumulated on the worker's host thread; the completion handler only records timestamps.
  double wp_fe_commit_ms = 0;    // inside -commit of the worker's own chunk buffers
  double wp_fe_sched_ms = 0;     // chunk commit -> GPUStartTime
  double wp_fe_handler_ms = 0;   // GPUEndTime -> completion handler entry
  double wp_fe_hop_ms = 0;       // completion handler entry -> next host-queue task start
  size_t wp_fe_submissions = 0;
  double wp_body_sched_ms = 0;   // rotation callback entry (the body has just committed) -> GPUStartTime of that buffer
  double wp_body_tail_ms = 0;    // GPUEndTime -> waitUntilCompleted returned
  size_t wp_body_rotations = 0, wp_finalize_calls = 0;   // worker_finalize_ms carries the nr_body_complete time
  double wp_complete_ms[6] = {0,0,0,0,0,0};   // segments of the body completion closure, in order (see the marks in nr_experimental_body.mm)
#endif
  void reset() { *this = NrExpEncodeProfile{}; } };
extern NrExpEncodeProfile nr_exp_encode_profile;
#endif
#if defined(NR_EXPERIMENTAL_WALL_PROFILE) && !defined(NR_EXPERIMENTAL_NO_VALIDATION)
#error "NR_EXPERIMENTAL_WALL_PROFILE is a diagnostic of the experimental lane and requires NR_EXPERIMENTAL_NO_VALIDATION"
#endif
// EXPERIMENTAL fail-closed per-(library, function) override for the BODY's libraries, mirroring the
// wide-stage mechanism. Off unless NR_EXP_BODY_OVERRIDE names a directory. A manifest declares, per
// descriptor library path, the sha256 of the replacement library and exactly which functions it may
// replace; every hash is verified and every declared function must exist, before any GPU work.
struct NrBodyOverride {
  bool active = false;
  std::string root, version, manifest_sha256;
  std::map<std::string, id<MTLLibrary>> libraries;        // key: descriptor lib path
  std::map<std::string, std::string> sha256;
  std::map<std::string, std::set<std::string>> functions; // which kernels each may replace
  std::map<std::string, std::string> resolved;            // "<lib>|<fn>" -> "override:<sha>" / "pinned"
#ifdef NR_EXPERIMENTAL_NATIVE_RANGE
  // CANDIDATE ONLY (build/nr-accept-v1/DISPATCH-EXTENSION-PROPOSAL-NATIVE-RANGE.json): a manifest-declared range of decoder
  // launches dispatched as a native block sequence instead of one PTX kernel per launch. Absent from every accepted build.
  struct NativeRange {
    std::string kind;                        // explicit list; first kind: wide256-chained-exact-dense-v1
    unsigned first = 0, last = 0;            // inclusive descriptor launch numbers
    id<MTLLibrary> library = nil, engine = nil;
    std::string library_sha256, engine_sha256;
    std::string resolution;                  // set by the body when the range is encoded ("native") or refused (error)
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
    // DECODER-EXACT-MLP-FUSION-v1 (experiments/DECODER-EXACT-MLP-FUSION-v1.json): declared with a *-chained-exact-dense-v3 kind, normalized to its
    // v2 kind at load; the block's exact expand, activate and first split run as one nrd_expand_activate_split_exact_ws4 dispatch
    bool fused = false;
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
    // DECODER-EXACT-MLP-FUSION-v2 (experiments/DECODER-EXACT-MLP-FUSION-v2.json): declared with a *-chained-exact-dense-v4 kind (fused is also set); the
    // second fusion set replaces five producer/consumer pairs of the block
    bool fused2 = false;
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
    // C32-NATIVE-FUSED-RANGE-v1 (experiments/C32-NATIVE-FUSED-RANGE-v1.json): declared with the wide32-chained-exact-dense-v3 kind, normalized to the v2 kind at
    // load; the block body runs the six gated fused pairs of EXACT-C32-FUSION-GATE-v1 (H, A, D, E, B, C); the upsample entry and the outview tail are unchanged
    bool fused32 = false;
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_POST
    // POST-BLOCK-NATIVE-EXACT-v1 (experiments/POST-BLOCK-NATIVE-EXACT-v1.json): kind post-block-native-exact-v1, launch 155 only, encoded natively on reset
    // frames; on temporal frames the launch keeps the generic lib_temporal PTX dispatch and the range resolves as "temporal-ptx"
    bool post = false;
#endif
  };
  std::vector<NativeRange> native_ranges;
#endif
};
#if defined(NR_EXPERIMENTAL_NATIVE_FUSE) && !(defined(NR_EXPERIMENTAL_NATIVE_RANGE) && defined(NR_EXPERIMENTAL_NO_VALIDATION))
#error "NR_EXPERIMENTAL_NATIVE_FUSE needs NR_EXPERIMENTAL_NATIVE_RANGE and NR_EXPERIMENTAL_NO_VALIDATION (a fused range skips activate's coverage counters)"
#endif
#if defined(NR_EXPERIMENTAL_NATIVE_FUSE2) && !defined(NR_EXPERIMENTAL_NATIVE_FUSE)
#error "NR_EXPERIMENTAL_NATIVE_FUSE2 extends NR_EXPERIMENTAL_NATIVE_FUSE and needs it"
#endif

struct NrBodyContext {
  // Experimental bounded emission. The callback receives the buffer we JUST committed so the owner can
  // establish its genuine completion, check its status and budget, and account it, before handing back the
  // next buffer. Returning nil aborts the encode. nullptr = accepted single-buffer behaviour.
  // (committed buffer, first launch in it, last launch in it) -> next buffer. The launch range lets the
  // owner name exactly which work a submission carried, instead of inferring it from timing.
  std::function<id<MTLCommandBuffer>(id<MTLCommandBuffer>, uint32_t, uint32_t, uint32_t)> nr_chunk_next = nullptr;
  uint32_t nr_chunk_encoder_bound = 0;      // rotate after this many encoders in the current buffer
  uint32_t nr_chunk_encoder_ceiling = 0;    // refuse BEFORE committing if a buffer would exceed this
  id<MTLDevice> dev; id<MTLCommandQueue> queue;
#ifdef NR_EXPERIMENTAL_CALLER_COMMAND
  id<MTLCommandBuffer> external_command = nil;   // experimental lane: the caller-owned buffer this frame is encoded into
#endif
  std::map<std::string, id<MTLLibrary>> libs;
  NrBodyOverride override;   // experimental; inert unless NR_EXP_BODY_OVERRIDE is set
  std::map<std::string, id<MTLComputePipelineState>> pipes;
  // v2 additions (ignored by v1): resident pool/scratch/constant buffers and internal textures keyed by role; constants are hashed once at load (provenance)
  std::map<std::string, id<MTLBuffer>> resident; std::map<std::string, std::string> resident_sha256; std::map<std::string, id<MTLTexture>> resident_tex;
  std::map<std::string, size_t> resident_bytes;   // v3: payload length of each resident constant (for the post-frame re-hash)
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
  // Candidate RESIDENCY-SET-BODY-v1 (build/nr-accept-v1/experiments/RESIDENCY-SET-BODY-v1.json): the resident allocations, kept resident across frames
  // when NR_EXP_RESIDENCY_SET=1; nil otherwise.
  id<MTLResidencySet> residency = nil; bool residency_dirty = false;
#endif
  std::string damaged; bool in_flight = false; uint64_t in_flight_id = 0, frame_serial = 0;
  // v17 (Codex's v16 review): a context is UNIQUE and address-stable - copy and move are deleted (a warmed copy would share Metal resources with
  // independent reservation flags); `alive` is a liveness token copied (weakly) into every pending at encode: after the context is destroyed, any
  // commit/complete/abandon through a pending is refused ("misuse: context destroyed") and never dereferences it. Lifetime rule: the context and the
  // temporal state must outlive every pending that references them (completion, abandonment or destruction); destroying a context with a frame in
  // flight is a misuse by the caller - the pending's later operations are refused and a committed buffer is still waited for on abandonment.
  std::shared_ptr<bool> alive = std::make_shared<bool>(true);
  NrBodyContext() = default; NrBodyContext(const NrBodyContext&) = delete; NrBodyContext& operator=(const NrBodyContext&) = delete; NrBodyContext(NrBodyContext&&) = delete; NrBodyContext& operator=(NrBodyContext&&) = delete;
  ~NrBodyContext() { if (alive) *alive = false; } };   // v16: in_flight_id = frame serial of the encoded/committed frame (0 = none); only the handle owning that serial may commit/complete it   // v15: in_flight = a frame is encoded/committed and not yet completed (one frame per context)   // v11: set by the runtime when a frame fails after GPU work started (reason); every later frame on this context is rejected up front - recreate the context
// v15 public resident API (gate 2): one NrBodyPending per frame; states idle -> encoded -> committed -> completed | failed. Misuse = code 2 with a "misuse:" error.
// Caller contract: every input buffer/texture/temporal state named in NrBodyFrameInputs stays alive and unmodified until nr_body_complete returns.
struct NrBodyPending; void nr_body_pending_abandon(NrBodyPending& p);   // v16: releases/damages an Encoded/Committed handle that is destroyed or overwritten without completion
// Experimental bounded-chunk emission (Opus): when `nr_chunk_next` is set, the body commits its current
// command buffer at a launch boundary - a point where no encoder is open - and continues into a fresh one
// supplied by the caller, once it has emitted at least `nr_chunk_encoder_bound` encoders into the current
// buffer. Ordering is guaranteed because every buffer is committed on the same queue in order, so the
// pending's single-buffer binding still describes the frame's completion. Unset by default: the accepted
// whole-body behaviour is unchanged.

struct NrBodyPending { enum State { Idle = 0, Encoded = 1, Committed = 2, Completed = 3, Failed = 4, Moved = 5 }; int state = Idle; id<MTLCommandBuffer> cmd = nil; std::function<int(NrBodyResult&, double)> complete; NrBodyContext* ctx = nullptr; NrTemporalState* temporal = nullptr; NrBodyResult res; double t_commit = 0; uint64_t frame_id = 0; std::weak_ptr<bool> ctx_alive;   // frame_id: the context's frame serial this handle owns (0 = none); ctx_alive: the context's liveness token (v17)
  // v16 (Codex's ownership review): a pending is UNIQUE - copying is deleted (a copy could complete twice and clear a later frame's reservation); moving transfers ownership and leaves the source in state Moved (commit/complete refused); destroying or overwriting an Encoded handle releases its reservation, destroying or overwriting a Committed handle waits for the buffer and DAMAGES the context and temporal state (its completion checks never ran)
  NrBodyPending() = default; NrBodyPending(const NrBodyPending&) = delete; NrBodyPending& operator=(const NrBodyPending&) = delete;
  NrBodyPending(NrBodyPending&& o) noexcept { *this = std::move(o); }
  NrBodyPending& operator=(NrBodyPending&& o) noexcept { if (this != &o) { nr_body_pending_abandon(*this); state = o.state; cmd = o.cmd; complete = std::move(o.complete); ctx = o.ctx; temporal = o.temporal; res = std::move(o.res); t_commit = o.t_commit; frame_id = o.frame_id; ctx_alive = std::move(o.ctx_alive); o.state = Moved; o.cmd = nil; o.complete = nullptr; o.ctx = nullptr; o.temporal = nullptr; o.frame_id = 0; } return *this; }
  ~NrBodyPending() { nr_body_pending_abandon(*this); } };
int nr_body_encode(NrBodyContext& ctx, NSDictionary* descriptor, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyPending& p);   // validates + encodes into the runtime-owned command buffer; no GPU work submitted
int nr_body_commit(NrBodyPending& p);                                     // submits the buffer (requires Encoded)
int nr_body_complete(NrBodyPending& p, NrBodyResult& res);               // waits, runs every deferred check, commits/damages the temporal state, returns the frame code (requires Committed)
void nr_body_pending_reset(NrBodyPending& p);                             // clears a completed/failed/moved pending for reuse (no-op while in flight)
// descriptor: the chain JSON (libraries, weights/params/table paths, pool layout, decoder launch table); it must not name activation files for the core path
int nr_body_run_frame(NrBodyContext& ctx, NSDictionary* descriptor, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyResult& res);
