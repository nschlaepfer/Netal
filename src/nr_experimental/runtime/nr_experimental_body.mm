// EXPERIMENTAL COPY of the accepted body runtime v17, for the native game lane only.
// Owner: Opus. Derived verbatim from NeuralStudio-nr-fable/src/nr_body/nr_body_runtime_v17.mm; the accepted
// file is unchanged and remains the exact lane's runtime. Two experimental changes, both opt-in and both
// off by default so this file still builds and behaves as v17 when the flags are unset:
//
//   1. NR_EXPERIMENTAL_CALLER_COMMAND - the frame is encoded into a command buffer the CALLER owns
//      (ctx.external_command), instead of one the runtime creates from its own queue. commit and complete
//      become no-ops for the buffer itself: the caller commits it and calls the completion step from its own
//      completion handler. This is what makes the body usable inside a game's encode hook at all.
//   2. NR_EXPERIMENTAL_NO_VALIDATION - the GPU coverage validators, the checkpoint digests, the in-buffer
//      snapshots and the completion-time SHA-256 passes are not encoded and not run. These are research
//      validation, not model work. Weights, kernels, dispatch geometry and launch order are untouched.
//
// This file must never be described as exact-parity evidence.

// v17 (2026-09-06, Codex's v16 review): (1) NrBodyContext is non-copyable and non-movable with a liveness token; every pending holds it weakly and
// commit/complete/abandon refuse ("misuse: context destroyed") instead of dereferencing a destroyed context (a committed buffer is still waited for);
// (2) abandonment damages the context/temporal state ONLY when the handle actually owns the context's in-flight frame (frame_id == in_flight_id != 0) -
// a forged or stale handle cannot damage anything by being destroyed; (3) lifetime rule documented in the header. Receipts/checks unchanged.
// v16 (2026-09-06, Codex's v15 API ownership review, docs/NR_V15_API_OWNERSHIP_REVIEW.md in Codex's checkout): (1) NrBodyPending is unique - copy deleted,
// move transfers ownership (source -> Moved), destructor/overwrite abandons: Encoded -> reservation released, Committed -> wait + context/temporal DAMAGE;
// (2) the context keeps a frame serial: encode assigns p.frame_id = ++frame_serial and in_flight_id; commit and complete are refused ("misuse: stale
// pending handle") unless p.frame_id == ctx.in_flight_id, so no handle - forged state or not - can complete a frame it does not own or clear a later
// frame's reservation; (3) the completion closure rebuilds its skip-region pointers from its OWN copy of the inputs (v15 pointed into the caller's
// NrBodyFrameInputs struct). Receipts/checks unchanged.
// v15 (2026-09-06, gate 2 of docs/NR_RESIDENT_ENCODE_CONTRACT.md): the PUBLIC encode / commit / complete API. nr_body_encode validates and encodes the
// whole frame into the runtime-owned command buffer of an NrBodyPending and stores the completion step as a closure over every encode-time value;
// nr_body_commit submits it; nr_body_complete waits, runs every deferred check, decides the temporal commit/damage and returns the frame code.
// States: idle -> encoded -> committed -> completed | failed. Misuse is refused with code 2 and a "misuse:" error: encode on an encoded/committed
// pending, encode while another frame is in flight on the context, commit when not encoded, complete when not committed, reuse of a failed pending
// without nr_body_pending_reset. A failed command buffer damages the context and the temporal state (as any failure after commit). nr_body_run_frame
// = encode + commit + complete on a local pending (unchanged behaviour and receipts).
// v14 (2026-09-06, Codex 21:10 conventions): (a) the 212 MB pool_post.bin diagnostic dump is written ONLY when the descriptor asks for it
// (`write_pool_post: "true"`), in both modes - checked-mode live recipes keep their snapshots in memory and report hashes; (b) res.wall_ms is assigned
// AFTER the final constant re-hash and the temporal commit (the harness's external wall remains the measured quantity). No check/arithmetic change.
// v13 (2026-09-06, Codex's v12 review c8914c9f…): (1) the decoder pre-image identity is the ACTUAL restaged pool - in checked mode a GPU snapshot of the
// pool taken before launch 133 is SHA-256 hashed at completion (decoder_pre_image_sha256, source "gpu-snapshot") and compared with the CPU
// reconstruction from the same inputs (decoder_pre_image_expected_sha256); production mode keeps the CPU reconstruction (source "cpu-reconstruction")
// plus the GPU digest cross-check and says so; (2) checked mode retains an in-memory snapshot of the pool after EVERY decoder launch (24 x 212 MB,
// resident, never dumped) and hashes them at completion - the 24 per-launch pool SHA-256 checkpoints of v11 checked mode exist again in the batched
// path; (3) the first-failure identity is a full 32-bit element index per validator record plus its observed count (nr_cov_first_obs), not a packed
// key (v12 overflowed above 2^24 elements); (4) the public encode/commit/complete API is still PENDING - nr_body_run_frame remains the only entry.
// v12 (2026-09-06, resident encode/completion contract v2, docs/NR_RESIDENT_ENCODE_CONTRACT.md; gated BODY-ONLY prototype, not a promotion):
// the whole frame is ENCODED into ONE runtime-owned command buffer (entry blit, middle launches, repacks, ViT blocks, decoder restage, decoder
// launches), COMMITTED once and COMPLETED once; every per-launch CPU check of v11 has an in-buffer GPU equivalent that runs BEFORE the checked
// slice or buffer is reused, and every CPU check that needs bytes runs at completion on bytes that are still intact:
//   - coverage: nr_cov_validate per launch slice (exact-once, expected-map for the launch-57 padding contract, at-most-once for decoder pool words)
//     with a sticky first-failure record and one completion record per validator; the record table is POISONED before the frame, the runtime counts
//     the validators it encoded, and completion requires validators_run == expected, every record complete with the expected (launch, elements) and
//     zero faults - a skipped or partial validator is a check failure, not a silent pass;
//   - intermediate checkpoints: nr_digest64 per launch output (and per ViT stage) into a poisoned checkpoint table before any reuse; the digest
//     function is cross-checked on the CPU at every frame boundary whose bytes survive (entry view, middle view, decoder pre-image, launch-155
//     output, history/motion before 155, final pool) - any disagreement is a check failure; the table's identity (digest_table_sha256) must be
//     identical across processes for the same poison (pool-wide entries cover poison-filled regions, so it is poison-class dependent by design; gate 1);
//   - immutability: resident constants re-hashed after completion (as v10/v11), the per-frame temporal record re-hashed, history/motion digested
//     before and after launch 155 on the GPU, the input texture re-hashed at completion;
//   - the driver's padding-token zeroing before repack 58 and the decoder restage (poison fill, middle-view copy, skip patches, control words) are
//     GPU blits/kernels in the same buffer; the decoder pre-image SHA-256 is computed on the CPU from the same inputs and cross-checked against the
//     GPU digest of the restaged pool.
// States: encoded -> committed -> completed | failed. A failure before commit is a rejection (code 2, no damage); after commit the context/temporal
// state is damaged exactly as in v11. The temporal commit stays deferred to the end of a clean frame (v10). Fault-test builds (-DNR_BODY_FAULT_TESTS)
// inject REAL in-buffer faults (blit pokes between a launch and its validator/digest, skipped/partial validators, skipped digests, corrupted restage)
// plus the v9-v11 hooks; a production build rejects any test_fault.
// Unchanged: kernels, dispatch order, dispatch sizes, buffer bindings, poison, arithmetic, the accepted receipt hashes (middle view, decoder
// pre-image, final pool, final surface, input/history/motion hashes, temporal record), nr_body_run_frame's interface and codes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_body_runtime.h"
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
#include "nr_experimental_gpu_profile.h"
#include "nr_experimental_encoder_scope_adoption.h"
#include <chrono>
// Experimental-lane profiling counters, reset by the caller before each encode. Not part of the accepted API.
NrExpEncodeProfile nr_exp_encode_profile;
static std::chrono::steady_clock::time_point nr_exp_phase_t0;
static int nr_exp_phase_cur = -1;
static void nr_exp_phase_mark(int next) {
  const auto now = std::chrono::steady_clock::now();
  if (nr_exp_phase_cur >= 0 && nr_exp_phase_cur < 8)
    nr_exp_encode_profile.phase_ms[nr_exp_phase_cur] += std::chrono::duration<double, std::milli>(now - nr_exp_phase_t0).count();
  nr_exp_phase_cur = next; nr_exp_phase_t0 = now;
}
#endif
#ifdef NR_BODY_SINGLE_WARP
#include "nr_single_warp_guard.h"
#endif
#include "nr_body_digest.h"
#include "nr_temporal/nr_temporal.h"
#import <CommonCrypto/CommonDigest.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
#include <algorithm>
#include <future>
#if !defined(NR_EXPERIMENTAL_NO_VALIDATION) || defined(NR_BODY_FAULT_TESTS)
#error "NR_EXPERIMENTAL_CONCURRENT_RECEIPTS (COMPLETION-RECEIPTS-CONCURRENT-v1) needs NR_EXPERIMENTAL_NO_VALIDATION and cannot be combined with NR_BODY_FAULT_TESTS, whose injections assume the serial completion order"
#endif
#endif
namespace fs = std::filesystem;
static std::vector<uint8_t> Read(const fs::path& p) { std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p.string()); const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v; }
static std::string Sha(const void* d, size_t n) { std::array<unsigned char, 32> h{}; CC_SHA256(d, CC_LONG(n), h.data()); std::ostringstream o; o << std::hex << std::setfill('0'); for (auto b : h) o << std::setw(2) << unsigned(b); return o.str(); }
static std::string Hex64(uint32_t hi, uint32_t lo) { std::ostringstream o; o << std::hex << std::setfill('0') << std::setw(8) << hi << std::setw(8) << lo; return o.str(); }
static std::string JStr(NSDictionary* d, const char* k) { NSString* v = d[[NSString stringWithUTF8String:k]]; return v ? std::string(v.UTF8String) : std::string(); }
static long JInt(NSDictionary* d, const char* k) { NSNumber* v = d[[NSString stringWithUTF8String:k]]; return v ? v.longValue : 0; }
struct SplitParams { int32_t shift_x, shift_y; uint32_t flags, tokens; }; enum : uint32_t { FLAG_RES_VIEW = 1, FLAG_OUT_VIEW = 2, FLAG_EMIT_F16 = 4, FLAG_IN_VIEW = 8 };
struct VitParams { uint32_t tokens, block, stage, reserved; }; struct RepackParams { int width, height; uint32_t in_dwords, out_dwords; };
struct ValidateArgs { uint32_t count, launch_id, launch_slot, mode; }; struct StickyRecord { uint32_t fault_flag, reserved, fault_count, validators_run; }; struct ValidatorRecord { uint32_t launch_id, elements, faults, done, first_elem, first_obs, pad0, pad1; };
struct DigestArgs { uint32_t words, launch_id, slot, reserved; }; struct DigestRecord { uint32_t launch_id, words, lo, hi, done, pad0, pad1, pad2; };
constexpr uint32_t VALIDATOR_DONE = 0xC0DE0001u, DIGEST_DONE = 0xD16E5701u, POISON_WORD = 0xEEEEEEEEu; constexpr size_t MAX_RECORDS = 4096;
constexpr size_t TIN = 2160u * 512u, VIEW = 16u * 69120u, POOLB = 640u * 512u, VITB = 640u * 1024u, T = 640;
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
// Candidate RESIDENCY-SET-BODY-v1 (build/nr-accept-v1/experiments/RESIDENCY-SET-BODY-v1.json): with NR_EXP_RESIDENCY_SET=1 every allocation the context
// keeps across frames joins one MTLResidencySet with requested residency, attached to every body command buffer. Evidence: the first launch of each
// native range paid 22-29 ms per GiB of its idle scratch set outside every encoder (experiments/FIRST-USE-COST-PER-TIER-POSTHOC-v1.json). No kernel,
// layout, batching or arithmetic change.
static bool nr_residency_enabled() { static const bool on = [] { const char* v = std::getenv("NR_EXP_RESIDENCY_SET"); return v != nullptr && std::string(v) == "1"; }(); return on; }
static void nr_residency_add(NrBodyContext& ctx, id<MTLDevice> dev, id<MTLAllocation> a) {
  if (!nr_residency_enabled()) return;
  if (!ctx.residency) { MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new]; d.label = @"nr-body-resident"; d.initialCapacity = 1024; NSError* err = nil;
    ctx.residency = [dev newResidencySetWithDescriptor:d error:&err];
    if (!ctx.residency) throw std::runtime_error(std::string("residency set allocation failed: ") + (err ? err.localizedDescription.UTF8String : "unknown")); }
  [ctx.residency addAllocation:a]; ctx.residency_dirty = true; }
static void nr_residency_remove(NrBodyContext& ctx, id<MTLAllocation> a) { if (ctx.residency && a) { [ctx.residency removeAllocation:a]; ctx.residency_dirty = true; } }
static void nr_residency_attach(NrBodyContext& ctx, id<MTLCommandBuffer> cb) {
  if (!ctx.residency) return;
  if (ctx.residency_dirty) { [ctx.residency commit]; [ctx.residency requestResidency]; ctx.residency_dirty = false; }
  [cb useResidencySet:ctx.residency]; }
#endif
static id<MTLBuffer> resident_buf(NrBodyContext& ctx, id<MTLDevice> dev, const std::string& key, size_t length) { auto it = ctx.resident.find(key); if (it != ctx.resident.end() && it->second.length == length) return it->second; id<MTLBuffer> b = [dev newBufferWithLength:length options:MTLResourceStorageModeShared]; if (!b) throw std::runtime_error("buffer allocation failed: " + key);
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
  if (it != ctx.resident.end()) nr_residency_remove(ctx, it->second);
  nr_residency_add(ctx, dev, b);
#endif
  ctx.resident[key] = b; return b; }
static id<MTLTexture> resident_tex(NrBodyContext& ctx, id<MTLDevice> dev, const std::string& key, MTLTextureDescriptor* d) { auto it = ctx.resident_tex.find(key); if (it != ctx.resident_tex.end() && it->second.width == d.width && it->second.height == d.height && it->second.pixelFormat == d.pixelFormat) return it->second; id<MTLTexture> t = [dev newTextureWithDescriptor:d]; if (!t) throw std::runtime_error("texture allocation failed: " + key);
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
  if (it != ctx.resident_tex.end()) nr_residency_remove(ctx, it->second);
  nr_residency_add(ctx, dev, t);
#endif
  ctx.resident_tex[key] = t; return t; }
static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("live input rejected: ") + what); }
static void damage(NrBodyContext& ctx, NrTemporalState* t, NrBodyResult& res, const std::string& why) { ctx.damaged = why; res.damage = why; if (t) { t->history_sha256 = "damaged:" + why.substr(0, 96); t->last_output_sha256 = "damaged"; } }
static void damage_temporal_only(NrTemporalState* t) { if (t) { t->history_sha256 = "damaged:context destroyed while a committed frame was pending"; t->last_output_sha256 = "damaged"; } }
static int misuse(NrBodyPending& p, const char* what) { p.res.code = 2; p.res.error = std::string("misuse: ") + what; return 2; }
int nr_body_encode(NrBodyContext& ctx, NSDictionary* chain, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyPending& p) { @autoreleasepool {
  if (p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) return misuse(p, "encode on a pending frame that is encoded/committed and not completed");
  if (p.state == NrBodyPending::Failed) return misuse(p, "encode on a failed pending frame: call nr_body_pending_reset first");
  if (ctx.in_flight || ctx.in_flight_id) { p.res = NrBodyResult{}; return misuse(p, "another frame is in flight on this context (encode before the previous complete)"); }
  p.res = NrBodyResult{}; p.cmd = nil; p.complete = nullptr; p.ctx = &ctx; p.temporal = in.temporal; p.frame_id = 0; p.ctx_alive = ctx.alive; NrBodyResult& res = p.res;
  try {
    require(ctx.damaged.empty(), "context damaged by an earlier failed frame: recreate the body context");
    if (in.temporal) require(in.temporal->history_sha256.rfind("damaged:", 0) != 0, "temporal state damaged by an earlier failed frame: recreate the temporal state");
  // v15: check(), pend and mark_damage live in the completion closure (nothing is checked before the buffer completed)
#ifdef NR_BODY_FAULT_TESTS
  const std::string fault = in.test_fault ? in.test_fault : "";   // test builds only: REAL in-buffer faults or post-completion corruptions, pixels untouched unless stated
#else
  if (in.test_fault) throw std::runtime_error("live input rejected: test_fault set on a production build");
  const std::string fault;
#endif
  const bool checked = !in.production;   // checked mode: every intermediate output is also SNAPSHOTTED in-buffer and SHA-256 hashed at completion (byte identity, as v11 diagnostics); production keeps the digests only
  const bool diag = diagnostics_dir && *diagnostics_dir; fs::path out = diag ? fs::path(diagnostics_dir) : fs::path(); if (diag) { if (fs::exists(out)) throw std::runtime_error("preserve previous run"); fs::create_directories(out); }
  NSError* err = nil; if (!chain) throw std::runtime_error("descriptor missing");
  const unsigned poison = in.poison; const fs::path tables = JStr(chain, "tables");
  id<MTLDevice> dev = ctx.dev; id<MTLCommandQueue> queue = ctx.queue;
#ifdef NR_EXPERIMENTAL_CALLER_COMMAND
  (void)queue;   // the caller owns the command buffer; the runtime queue is unused in this lane
#endif
  auto lib = [&](const std::string& p) { auto it = ctx.libs.find(p); if (it != ctx.libs.end()) return it->second; id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]] error:&err]; if (!l) throw std::runtime_error(err.localizedDescription.UTF8String); ctx.libs[p] = l;
    { std::vector<uint8_t> lb = Read(p); ctx.resident_sha256["lib:" + p] = Sha(lb.data(), lb.size()); }   // v3: library identity at load time
    return l; };
  id<MTLLibrary> slib = lib(JStr(chain, "split_lib")), vlib = lib(JStr(chain, "vit_lib"));
  const std::string tplib_path = JStr(chain, "vit_tp_lib"); id<MTLLibrary> tplib = tplib_path.empty() ? nil : lib(tplib_path);   // optional: three-product expand/contract/projection (bit-identical, faster)
  const std::string sptp_path = JStr(chain, "split_tp_lib"); id<MTLLibrary> sptp = sptp_path.empty() ? nil : lib(sptp_path);   // optional: three-product split conv (all flag variants)
  const std::string rlib_path = JStr(chain, "resident_lib"); if (rlib_path.empty()) throw std::runtime_error("descriptor has no resident_lib (v12 requires the validator/digest library nr_body_validate.metal)"); id<MTLLibrary> rlib = lib(rlib_path);
  auto pso = [&](id<MTLLibrary> l, const char* name) { const std::string key = std::to_string(reinterpret_cast<uintptr_t>((__bridge void*)l)) + "|" + name; auto it = ctx.pipes.find(key); if (it != ctx.pipes.end()) return it->second; id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err]; if (!ps) throw std::runtime_error(std::string(name) + ": " + err.localizedDescription.UTF8String); ctx.pipes[key] = ps; return ps; };
  id<MTLComputePipelineState> psoValidate = pso(rlib, "nr_cov_validate"), psoDigest = pso(rlib, "nr_digest64"), psoFirstObs = pso(rlib, "nr_cov_first_obs"), psoPadMask = pso(rlib, "nr_apply_pad_mask");
#ifdef NR_EXPERIMENTAL_BODY_OVERRIDE
  // EXPERIMENTAL body override for the three-product (TP) libraries. The ViT and split launches take their
  // pipelines through pso(tplib / sptp, name), not through a decoder launch record, so the per-launch
  // selection further down never sees them. Selection stays per (descriptor library path, kernel): for every
  // kernel the manifest declares for vit_tp_lib or split_tp_lib, the override pipeline is placed in the
  // pipeline cache under the PINNED library's key, so the unchanged pso(tplib, name) call sites return it and
  // every undeclared kernel still compiles from the pinned library. Each seeded kernel is recorded as
  // resolved and the fail-closed admission below checks every declaration. The mechanism lives entirely in
  // this region, so the copy still reduces to nr_body_runtime_v17.
  if (ctx.override.active) {
    auto seed_tp = [&](id<MTLLibrary> pinned, const std::string& libp) {
      if (pinned == nil || libp.empty()) return;
      // The descriptor library path here is not the manifest's RELATIVE descriptor path, so match on that
      // suffix at a '/' boundary - the same rule as the decoder selection below. The first version of this
      // region matched exactly, seeded nothing, and the fail-closed admission refused every frame.
      std::string rel;
      for (const auto& kv : ctx.override.libraries) {
        const std::string& k = kv.first;
        if (libp == k || (libp.size() > k.size() && libp.compare(libp.size() - k.size(), k.size(), k) == 0 &&
                          libp[libp.size() - k.size() - 1] == '/')) { rel = k; break; }
      }
      if (rel.empty()) return;
      auto ovf = ctx.override.functions.find(rel); auto ovl = ctx.override.libraries.find(rel);
      if (ovf == ctx.override.functions.end() || ovl == ctx.override.libraries.end()) return;
      const std::string tag = "override:" + ctx.override.sha256[rel];
      for (const std::string& fn : ovf->second) {
        const std::string key = std::to_string(reinterpret_cast<uintptr_t>((__bridge void*)pinned)) + "|" + fn;
        const std::string mark = key + "|" + tag;   // seeded once per context; later frames reuse the cached pipeline
        if (ctx.pipes.find(mark) == ctx.pipes.end()) {
          id<MTLFunction> f = [ovl->second newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
          id<MTLComputePipelineState> ps = f ? [dev newComputePipelineStateWithFunction:f error:&err] : nil;
          if (!ps) throw std::runtime_error("body override: cannot build " + rel + "|" + fn);
          ctx.pipes[key] = ps; ctx.pipes[mark] = ps;
        }
        ctx.override.resolved[rel + "|" + fn] = tag;
      }
    };
    seed_tp(tplib, tplib_path); seed_tp(sptp, sptp_path);
  }
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // The validator/digest pipelines are still CREATED (cheap, and keeps the loaded-library provenance
  // identical to the accepted runtime) but nothing encodes them in this lane.
  (void)psoValidate; (void)psoDigest; (void)psoFirstObs;
#endif
  const size_t pool_bytes = size_t(JInt(chain, "pool_bytes")); id<MTLBuffer> pool = resident_buf(ctx, dev, "pool", pool_bytes + 4096); std::memset(pool.contents, poison, pool_bytes); std::memset(static_cast<uint8_t*>(pool.contents) + pool_bytes, 0, 4096);
  require(pool_bytes % 64 == 0, "pool size must be a multiple of 64 bytes (digest chunks)");
  require(in.entry.buffer != nil && in.entry.bytes == VIEW && in.entry.offset + VIEW <= in.entry.buffer.length, "encoder entry buffer/bytes/offset");
  unsigned scratch_seq = 0;   // v2: scratch buffers are resident in the context, keyed by their (deterministic) request order and size, re-filled every frame
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // Profiling of the per-frame CPU cost of the scratch fills, which is what makes encode() expensive.
  auto scratch = [&](size_t n, uint8_t fill) {
    id<MTLBuffer> b = resident_buf(ctx, dev, "scratch:" + std::to_string(scratch_seq++) + ":" + std::to_string(n), n + 16);
    const auto t0 = std::chrono::steady_clock::now();
    std::memset(b.contents, fill, n + 16);
    nr_exp_encode_profile.fill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    nr_exp_encode_profile.fill_bytes += n + 16;
    ++nr_exp_encode_profile.fill_calls;
    if (fill) { nr_exp_encode_profile.poison_bytes += n + 16; ++nr_exp_encode_profile.poison_calls; }
    return b; };
#else
  auto scratch = [&](size_t n, uint8_t fill) { id<MTLBuffer> b = resident_buf(ctx, dev, "scratch:" + std::to_string(scratch_seq++) + ":" + std::to_string(n), n + 16); std::memset(b.contents, fill, n + 16); return b; };
#endif
  auto constant = [&](const fs::path& p, size_t pad = 16) { const std::string key = "file:" + p.string() + ":" + std::to_string(pad); auto it = ctx.resident.find(key); if (it != ctx.resident.end()) return it->second;
    std::vector<uint8_t> v = Read(p); id<MTLBuffer> b = [dev newBufferWithLength:v.size() + pad options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + pad); std::memcpy(b.contents, v.data(), v.size());
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
    nr_residency_add(ctx, dev, b);
#endif
    ctx.resident[key] = b; ctx.resident_sha256[key] = Sha(v.data(), v.size()); ctx.resident_bytes[key] = v.size(); return b; };   // model constants: loaded and hashed once per process
  // v4: every live input is validated here, BEFORE any encoding (the same checks are repeated at their use sites)
  if (NSDictionary* D0 = chain[@"decoder"]) {
    NSArray* patches0 = D0[@"patches"]; require(patches0 && patches0.count == 5, "descriptor patch table (4 skips + Block0 skip)");
    const NrBodyRegion* regs0[5] = {&in.skips[0], &in.skips[1], &in.skips[2], &in.skips[3], &in.block0_skip};
    for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches0[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs0[i];
      require(r.buffer != nil && r.bytes == bytes && r.offset + bytes <= r.buffer.length && off + bytes <= pool_bytes, JStr(P, "file").c_str()); }
    const unsigned tw0 = unsigned(JInt(D0, "tex_w")), th0 = unsigned(JInt(D0, "tex_h"));
    require(in.input_color != nil && in.input_color.pixelFormat == MTLPixelFormatRGBA16Float && in.input_color.width == tw0 && in.input_color.height == th0, "input colour texture (RGBA16F WxH)");
    if (in.output_surface) require(in.output_surface.pixelFormat == MTLPixelFormatRGBA16Float && in.output_surface.width == tw0 && in.output_surface.height == th0, "output surface (RGBA16F WxH)");
    for (NSDictionary* L in D0[@"launches"]) if (JInt(L, "launch") == 155) { id<MTLBuffer> bp = constant(JStr(L, "params"), 64); const size_t ps = bp.length - 64; const uint8_t* pb = static_cast<const uint8_t*>(bp.contents);
      require(ps >= 104, "post block parameter block"); for (size_t i = 88; i < 104; ++i) require(pb[i] == 0, "descriptor launch-155 record must be the accepted reset (frame-0) record: temporal words present");   // the base record is always the accepted frame-0 record; temporal words are constructed per frame, never stored
      if (!in.reset) { require(ps == 184, "launch-155 record must be 184 bytes for temporal construction"); require(!JStr(L, "lib_temporal").empty(), "reset=false requires the descriptor's lib_temporal (texref-mapped post kernel)"); } }
    if (!in.reset) {   // v5: temporal frame requirements (fail closed)
      require(in.temporal != nullptr, "reset=false requires a temporal state"); require(in.temporal->history != nil && in.temporal->width == tw0 && in.temporal->height == th0 && in.temporal->history.pixelFormat == MTLPixelFormatRGBA16Float, "temporal history texture (RGBA16F WxH)");
      require(in.temporal->frames_since_reset >= 1, "reset=false before any frame in this temporal state");
      { bool found = false; for (NSDictionary* L in D0[@"launches"]) if (unsigned(JInt(L, "launch")) == 155u) { found = true; NSDictionary* T = L[@"ex2_half_table"]; require(T != nil, "temporal launch 155 requires the descriptor's ex2_half_table {file, sha256}");
          auto tb = Read(JStr(T, "file")); const std::string got = Sha(tb.data(), tb.size()); require(tb.size() == 524288 && got == JStr(T, "sha256") && got == "f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf", "ex2 half-domain table: size/sha mismatch against the descriptor and the pinned hash"); }
        require(found, "descriptor has no launch 155"); } require(!in.temporal->history_sha256.empty() && in.temporal->history_sha256 == in.temporal->last_output_sha256, "history continuity: history must equal the previous frame's final output of this state");
      require(in.motion != nil && in.motion.pixelFormat == MTLPixelFormatRG16Float && in.motion.width == tw0 && in.motion.height == th0, "motion texture (RG16F WxH) required for reset=false");
      require(in.output_surface != nil, "temporal routing requires a caller output surface");
      { std::vector<uint8_t> hb(size_t(tw0) * th0 * 8); [in.temporal->history getBytes:hb.data() bytesPerRow:tw0 * 8 fromRegion:MTLRegionMake2D(0, 0, tw0, th0) mipmapLevel:0]; require(Sha(hb.data(), hb.size()) == in.temporal->history_sha256, "history texture content does not match the recorded history hash"); } }   // checked BEFORE any encoding
    if (in.temporal) require(in.temporal->history != nil && in.temporal->width == tw0 && in.temporal->height == th0 && in.output_surface != nil, "temporal state present: history texture WxH and caller output surface required");
  }
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(0);
#endif
  // ---- v12 evidence tables: poisoned records, sticky validator state, expectations recorded while encoding ----
  id<MTLBuffer> vrec = scratch(MAX_RECORDS * sizeof(ValidatorRecord), 0), drec = scratch(MAX_RECORDS * sizeof(DigestRecord), 0), sticky = scratch(64, 0), padmask = scratch(VITB, 0);
  { ValidatorRecord* vr = static_cast<ValidatorRecord*>(vrec.contents); for (size_t i = 0; i < MAX_RECORDS; ++i) vr[i] = ValidatorRecord{POISON_WORD, POISON_WORD, 0u, POISON_WORD, 0xffffffffu, POISON_WORD, 0u, 0u};
    DigestRecord* dr = static_cast<DigestRecord*>(drec.contents); for (size_t i = 0; i < MAX_RECORDS; ++i) dr[i] = DigestRecord{POISON_WORD, POISON_WORD, 0u, 0u, POISON_WORD, 0u, 0u, 0u};
    *static_cast<StickyRecord*>(sticky.contents) = StickyRecord{0u, 0u, 0u, 0u}; }
  struct ValExp { uint32_t launch_id, elements; std::string label; }; std::vector<ValExp> vexp;
  struct DigExp { uint32_t launch_id = 0, words = 0; std::string label; bool has_cpu = false; NrDigest64 cpu; id<MTLBuffer> snap = nil; std::string sha; }; std::vector<DigExp> dexp;
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // Nothing is encoded for validators or digests in this lane, and every stub returns slot 0. One inert record
  // in each table keeps that slot in bounds; the verification loops below are skipped, so it is never read as
  // if it were real evidence.
  vexp.push_back({0u, 0u, "experimental lane: validation not encoded"});
  dexp.push_back({});
#endif
  struct LaunchRec { long n; std::string kind; std::vector<std::pair<std::string, uint32_t>> digests; std::vector<uint32_t> validators; std::string extra; }; std::vector<LaunchRec> launches;
#ifdef NR_EXPERIMENTAL_CALLER_COMMAND
  id<MTLCommandBuffer> cmd = ctx.external_command;   // the CALLER owns this buffer; we only append to it
  if (!cmd) throw std::runtime_error("experimental lane: ctx.external_command must be set before nr_body_encode");
  if (cmd.status != MTLCommandBufferStatusNotEnqueued) throw std::runtime_error("experimental lane: the caller command buffer is already enqueued or committed");
  if (cmd.device != ctx.dev) throw std::runtime_error("experimental lane: the caller command buffer belongs to another device");
#else
  id<MTLCommandBuffer> cmd = [queue commandBuffer]; ++res.inventory_command_buffers; if (!cmd) throw std::runtime_error("command buffer allocation failed");
#endif
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
  nr_residency_attach(ctx, cmd);
#endif
  const auto t_all = [NSDate date];
  auto blit = [&]() { id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder]; nr_exp::nr_scope_adopt(b); ++res.inventory_blit_encoders; return b; };
#ifdef NR_EXPERIMENTAL_PROFILE_LABELS
  // DIAGNOSTIC BUILDS ONLY (build/nr-accept-v1/experiments/BODY-TP-COMPOSITION-v1.json): name the body TP encoders by kernel so the regions
  // profiler separates weight preparation from the TP kernels. Without the flag NR_TPL(e, label) is (e): the label is not even evaluated.
#define NR_TPL(enc_expr, label_expr) ([&]() { id<MTLComputeCommandEncoder> e_ = (enc_expr); if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = std::string(label_expr); return e_; }())
#else
#define NR_TPL(enc_expr, label_expr) (enc_expr)
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // Optional per-encoder GPU timestamps (NR_EXP_GPU_TIMESTAMPS=1). Inert otherwise; the encoder is the
  // same object either way and nothing is read during encoding.
  auto compute = [&]() { id<MTLComputeCommandEncoder> e = nr_exp::gpu_profiler().compute(cmd, "body"); nr_exp::nr_scope_adopt(e); ++res.inventory_compute_encoders; return e; };
#else
  auto compute = [&]() { id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; nr_exp::nr_scope_adopt(e); ++res.inventory_compute_encoders; return e; };
#endif
#ifdef NR_EXPERIMENTAL_CHUNKED
  // Bounded emission: at a launch boundary - a point where NO encoder is open - commit the current buffer
  // and continue into a fresh one once enough encoders have gone into this one. Same kernels, same order,
  // same bindings; only the command-buffer split changes. Buffers are committed on one queue in order, so
  // waiting on the last still means the whole frame finished.
  uint32_t nr_enc_at_last_split = 0;
  // nr_cur_launch is the last launch ACTUALLY ENCODED into `cmd`, so it is assigned AFTER the boundary
  // call, never before it. The boundary runs at a launch HEAD: at that moment nothing of the incoming
  // launch has been encoded, so the buffer being committed ends at the PREVIOUS launch. Reporting the
  // incoming launch as the buffer's `hi` would name a launch the buffer does not contain.
  // kNoLaunch marks a buffer that carried no launch at all (setup encoders only).
  static const uint32_t kNoLaunch = 0xffffffffu;
  uint32_t nr_cur_launch = kNoLaunch, nr_launch_at_split = kNoLaunch;
  auto nr_chunk_boundary = [&](uint32_t nr_incoming_launch) {
    if (!ctx.nr_chunk_next || ctx.nr_chunk_encoder_bound == 0) return;
    if (nr_launch_at_split == kNoLaunch) nr_launch_at_split = nr_incoming_launch;
    const uint32_t emitted = res.inventory_compute_encoders + res.inventory_blit_encoders;
    if (emitted - nr_enc_at_last_split < ctx.nr_chunk_encoder_bound) return;
    const uint32_t in_this = emitted - nr_enc_at_last_split;
    // Ceiling is enforced BEFORE the commit, so an over-large buffer is never submitted.
    if (ctx.nr_chunk_encoder_ceiling && in_this > ctx.nr_chunk_encoder_ceiling)
      throw std::runtime_error("bounded emission: a command buffer would carry " + std::to_string(in_this) +
                               " encoders, over the declared per-buffer ceiling of " + std::to_string(ctx.nr_chunk_encoder_ceiling));
    id<MTLCommandBuffer> just = cmd;
    [just commit];
    // Name the work this buffer carried: the launches encoded into it since the previous rotation, ending
    // at the last one encoded - NOT at the incoming launch, which starts the next buffer.
    const uint32_t nr_lo = (nr_cur_launch == kNoLaunch) ? kNoLaunch : nr_launch_at_split;
    cmd = ctx.nr_chunk_next(just, nr_lo, nr_cur_launch, in_this);
    nr_launch_at_split = nr_incoming_launch;
    if (!cmd) throw std::runtime_error("bounded emission: the owner stopped the run after a submitted buffer");
#ifdef NR_EXPERIMENTAL_RESIDENCY_SET
    nr_residency_attach(ctx, cmd);
#endif
    ++res.inventory_command_buffers;
    if (in_this > res.nr_max_encoders_per_buffer) res.nr_max_encoders_per_buffer = in_this;
    nr_enc_at_last_split = emitted;
  };
#endif
  auto fill = [&](id<MTLBuffer> b, size_t off, size_t len, uint8_t v) { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:b range:NSMakeRange(off, len) value:v]; nr_exp::nr_scope_end(bl); };
  // one coverage validator over `elements` counters of `counters` (mode 0 exact once, 1 expected map `expect`, 2 at most once); returns its record slot
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // Experimental lane: coverage validators are research validation, not model work. Nothing is encoded and
  // no record slot is reserved, so the completion step has nothing to check. Signature preserved.
  auto validate = [&](const std::string&, uint32_t, id<MTLBuffer>, size_t, uint32_t, id<MTLBuffer>, bool = true, size_t = 0) -> uint32_t { return 0u; };
#else
  auto validate = [&](const std::string& label, uint32_t launch_id, id<MTLBuffer> counters, size_t elements, uint32_t mode, id<MTLBuffer> expect, bool encode = true, size_t encoded_elements = 0) {
    if (vexp.size() >= MAX_RECORDS) throw std::runtime_error("validator record table full"); const uint32_t slot = uint32_t(vexp.size()); vexp.push_back({launch_id, uint32_t(elements), label}); ++res.validators_expected;
    if (encode) { ValidateArgs a{uint32_t(encoded_elements ? encoded_elements : elements), launch_id, slot, mode}; id<MTLComputeCommandEncoder> e = compute(); [e setComputePipelineState:psoValidate];
      [e setBuffer:counters offset:0 atIndex:0]; [e setBuffer:expect ? expect : padmask offset:0 atIndex:1]; [e setBuffer:sticky offset:0 atIndex:2]; [e setBuffer:vrec offset:0 atIndex:3]; [e setBytes:&a length:sizeof a atIndex:4];
      const size_t threads = std::min<size_t>(a.count, 1u << 18); [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; nr_exp::nr_scope_end(e);
      id<MTLComputeCommandEncoder> e2 = compute(); [e2 setComputePipelineState:psoFirstObs]; [e2 setBuffer:counters offset:0 atIndex:0]; [e2 setBuffer:vrec offset:0 atIndex:1]; [e2 setBytes:&a length:sizeof a atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; nr_exp::nr_scope_end(e2); }   // v13: observed count of the first faulting element while the counters are intact
    return slot; };
  // one checkpoint digest of `bytes` bytes at `off` in `b` (word-sized); optional CPU digest of the same bytes for the boundary cross-check
  // `snapshot` (checked mode only): the same bytes are also blit-copied into a resident snapshot buffer at this point of the frame and SHA-256 hashed at completion
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // Experimental lane: checkpoint digests and their in-buffer snapshots are research validation. Not encoded.
  auto digest = [&](const std::string&, uint32_t, id<MTLBuffer>, size_t, size_t, const uint8_t* = nullptr, bool = true, bool = true) -> uint32_t { return 0u; };
#else
  auto digest = [&](const std::string& label, uint32_t launch_id, id<MTLBuffer> b, size_t off, size_t bytes, const uint8_t* cpu_bytes = nullptr, bool encode = true, bool snapshot = true) {
    if (dexp.size() >= MAX_RECORDS) throw std::runtime_error("digest record table full"); if (bytes % 4 != 0 || off % 4 != 0) throw std::runtime_error("digest region not word aligned: " + label);
    const uint32_t slot = uint32_t(dexp.size()); DigExp d; d.launch_id = launch_id; d.words = uint32_t(bytes / 4); d.label = label; if (cpu_bytes) { d.has_cpu = true; d.cpu = nr_digest64_cpu(cpu_bytes, bytes); }
    if (checked && snapshot) { d.snap = scratch(bytes, 0); id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:b sourceOffset:off toBuffer:d.snap destinationOffset:0 size:bytes]; nr_exp::nr_scope_end(bl); }
    dexp.push_back(d); ++res.digests_expected;
    if (encode) { DigestArgs a{uint32_t(bytes / 4), launch_id, slot, 0u}; id<MTLComputeCommandEncoder> e = compute(); [e setComputePipelineState:psoDigest]; [e setBuffer:b offset:off atIndex:0]; [e setBuffer:drec offset:0 atIndex:1]; [e setBytes:&a length:sizeof a atIndex:2];
      const size_t chunks = (bytes / 4 + 15) / 16, threads = std::min<size_t>(chunks, 1u << 16); [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; nr_exp::nr_scope_end(e); }
    return slot; };
#endif
  auto tex_to_buf = [&](id<MTLTexture> t, id<MTLBuffer> b, size_t bpp) { id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(t.width, t.height, 1) toBuffer:b destinationOffset:0 destinationBytesPerRow:t.width * bpp destinationBytesPerImage:t.width * t.height * bpp]; nr_exp::nr_scope_end(bl); };
#ifdef NR_BODY_FAULT_TESTS
  auto poke = [&](id<MTLBuffer> b, size_t byte, uint8_t v) { fill(b, byte, 1, v); };   // REAL in-buffer fault: one byte of a coverage slice or buffer changed between the launch and its validator/digest
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(1);
#endif
  // ---- entry blit (caller buffer -> pool) + digest of the entry view, cross-checked against the caller bytes ----
#ifdef NR_EXPERIMENTAL_ENTRY_ORDERED_CHECK
  uint32_t entry_src_digest_slot = UINT32_MAX, entry_view_digest_slot = UINT32_MAX;
#endif
  { const size_t off = size_t(JInt(chain, "entry_offset")); id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:in.entry.buffer sourceOffset:in.entry.offset toBuffer:pool destinationOffset:off size:VIEW]; nr_exp::nr_scope_end(bl);
#ifdef NR_EXPERIMENTAL_ENTRY_ORDERED_CHECK
    // ORDERING FIX. The accepted check hashes in.entry.buffer.contents on the CPU AT ENCODE TIME and
    // compares that against a GPU digest of the copy destination. In this lane the entry is PRODUCED by
    // GPU work encoded into this same command buffer and not yet executed, so the CPU read sees
    // pre-production bytes and the check fails identically in every arm - which is exactly what it did.
    // Capture BOTH sides on the GPU at this boundary instead, in encode order, and compare them only
    // after the buffer has completed. No archived data is loaded, the check is not suppressed, and
    // nothing commits or waits inside the caller-owned encode.
    entry_src_digest_slot = digest("entry-source", 24u, in.entry.buffer, in.entry.offset, VIEW, nullptr, true, false);
    launches.push_back({24, "entry", {{"entry_view", digest("entry-view", 24u, pool, off, VIEW, nullptr, true, false)}}, {}, ""});
    entry_view_digest_slot = uint32_t(dexp.size() - 1); }
#else
    const uint8_t* ecpu = in.entry.buffer.contents ? static_cast<const uint8_t*>(in.entry.buffer.contents) + in.entry.offset : nullptr; launches.push_back({24, "entry", {{"entry_view", digest("entry-view", 24u, pool, off, VIEW, ecpu, true, false)}}, {}, ""}); }
#endif
  id<MTLBuffer> rsq = constant(tables / "rtx-rsqrt-half.bin"), rcp = constant(tables / "rtx-rcp-half.bin");
  const size_t covmax = std::max(T * 4096 * 4, TIN * 4); id<MTLBuffer> cov = scratch(covmax, 0), cov2 = scratch(POOLB * 4, 0), y16 = scratch(TIN * 2, 0), qb = scratch(TIN, uint8_t(poison)), kb = scratch(TIN, uint8_t(poison)), vb = scratch(TIN, uint8_t(poison));
  // launch-57 padding contract map (1 = padding byte, must never be written; used as the validator's expected map inverted below and by the pad-mask kernel)
  std::vector<uint8_t> is_pad(VITB, 0), expect57(VITB, 1); { NSArray* pad = chain[@"vit"][@"padding_tokens"]; for (NSNumber* t in pad) { const unsigned tok = t.unsignedIntValue; for (unsigned c = 0; c < 1024; ++c) { const unsigned tile = tok / 16, slot = tok % 16, lane = 4 * (slot % 8) + (c % 16) / 4, reg = (slot % 16) / 8 + 2 * ((c % 32) / 16); const size_t i = tile * 16384 + (c / 32) * 512 + lane * 16 + reg * 4 + c % 4; is_pad[i] = 1; expect57[i] = 0; } } res.out_vit_padding_bytes = size_t(pad.count) * 1024u; }
  std::memcpy(padmask.contents, is_pad.data(), VITB); id<MTLBuffer> expect57b = scratch(VITB, 0); std::memcpy(expect57b.contents, expect57.data(), VITB);
  id<MTLBuffer> spw[3], spmax, spstats; if (sptp) { for (int m = 0; m < 3; ++m) spw[m] = scratch(16u * 32u * 512u * 2u, 0); spmax = scratch(16u * 512u, 0); spstats = scratch(64, 0); }
  id<MTLBuffer> atk[3], atv[3], atkmax, atvmax; if (sptp) { for (int m = 0; m < 3; ++m) { atk[m] = scratch(40u * 16u * 32u * 64u * 2u, 0); atv[m] = scratch(40u * 16u * 2u * 32u * 32u * 2u, 0); } atkmax = scratch(40u * 16u * 64u, 0); atvmax = scratch(40u * 16u * 2u * 32u, 0); }   // split attend operands per (window, head)
  id<MTLBuffer> qw[3], qwmax; if (sptp) { for (int m = 0; m < 3; ++m) qw[m] = scratch(16u * 32u * 1536u * 2u, 0); qwmax = scratch(16u * 1536u, 0); }   // split qkv weights: 16 steps x 1536 columns
  id<MTLBuffer> fw2[3], fw3[3], fw2max, fw3max; if (sptp) { for (int m = 0; m < 3; ++m) { fw2[m] = scratch(8u * 2u * 32u * 256u * 2u, 0); fw3[m] = scratch(8u * 8u * 32u * 64u * 2u, 0); } fw2max = scratch(8u * 2u * 256u, 0); fw3max = scratch(8u * 8u * 64u, 0); }
#ifdef NR_EXPERIMENTAL_TP_STATS
  // DIAGNOSTIC BUILDS ONLY (build/nr-accept-v1/experiments/TP-REPAIR-STATS-v1.json): after every TP dispatch, copy the kernel's repaired
  // element-step count (stats[0]) into a per-dispatch slot and zero the counter; the completion step writes slot/label/count to
  // NR_EXP_TP_STATS_DUMP. No kernel or arithmetic change; the extra blits are an observer effect on timing only.
  id<MTLBuffer> tpsdiag = scratch(4u * 4096u, 0); std::vector<std::string> tps_labels; tps_labels.reserve(4096);
  auto tp_stats_capture = [&](id<MTLBuffer> stats, const std::string& label) {
    if (!stats || tps_labels.size() >= 4096u) return; const size_t slot = tps_labels.size();
    id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:stats sourceOffset:0 toBuffer:tpsdiag destinationOffset:slot * 4u size:4u]; [bl fillBuffer:stats range:NSMakeRange(0, 8) value:0]; nr_exp::nr_scope_end(bl);
    tps_labels.push_back(label); };
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(2);
#endif
  // ---- split launches (encoded; coverage validated and outputs digested in-buffer before the slices are refilled) ----
  auto run_split = [&](NSDictionary* L) {
    const long n = JInt(L, "launch"); const std::string kernel = JStr(L, "kernel"); NSArray* shift = L[@"shift"]; NSDictionary* roles = L[@"roles"]; auto weights = constant(JStr(L, "weights"));
    auto off = [&](const char* role) { NSDictionary* r = roles[[NSString stringWithUTF8String:role]]; if (!r) throw std::runtime_error(std::to_string(n) + ": role " + role); return size_t(JInt(r, "pool_offset")); };
    SplitParams prm{int32_t([shift[0] intValue]), int32_t([shift[1] intValue]), 0, 2160};
    { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:cov range:NSMakeRange(0, covmax + 16) value:0]; [bl fillBuffer:cov2 range:NSMakeRange(0, POOLB * 4 + 16) value:0]; nr_exp::nr_scope_end(bl); }
    std::vector<std::pair<std::string, size_t>> outbytes;
    auto dispatch = [&](const char* name, std::vector<std::pair<id<MTLBuffer>, size_t>> bufs, size_t threads) {
      id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), std::string("body:") + name); [enc setComputePipelineState:pso(slib, name)];
      for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i].first offset:bufs[i].second atIndex:i];
      [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc);
    };
    size_t o_off = 0, o_bytes = 0; std::string o_role;
    if (kernel == "ffwd" || kernel == "ffwd_view") { const bool view = kernel == "ffwd_view"; prm.flags = view ? FLAG_IN_VIEW : 0; o_role = "out"; o_off = off("out"); o_bytes = TIN;
      if (sptp) {
        { id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_tp_prep"); [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:1 + m]; [enc setBuffer:spmax offset:0 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(16u * 512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc); }
        { id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_ffwd_tp_prep"); [enc setComputePipelineState:pso(sptp, "nr_split_ffwd_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw2[m] offset:0 atIndex:1 + m]; [enc setBuffer:fw2max offset:0 atIndex:4]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw3[m] offset:0 atIndex:5 + m]; [enc setBuffer:fw3max offset:0 atIndex:8];
          [enc dispatchThreads:MTLSizeMake(8u * 2u * 256u + 8u * 8u * 64u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc); }
        id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_ffwd_tp"); [enc setComputePipelineState:pso(sptp, "nr_split_ffwd_tp")];
        [enc setBuffer:pool offset:off(view ? "x_view" : "x") atIndex:0]; [enc setBuffer:weights offset:0 atIndex:1]; [enc setBuffer:pool offset:o_off atIndex:2]; [enc setBuffer:cov offset:0 atIndex:3]; [enc setBytes:&prm length:sizeof(prm) atIndex:4];
        for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:5 + m]; [enc setBuffer:spmax offset:0 atIndex:8]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw2[m] offset:0 atIndex:9 + m]; [enc setBuffer:fw2max offset:0 atIndex:12]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw3[m] offset:0 atIndex:13 + m]; [enc setBuffer:fw3max offset:0 atIndex:16]; [enc setBuffer:spstats offset:0 atIndex:17];
        [enc dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(spstats, "launch " + std::to_string(n) + " nr_split_ffwd_tp");
#endif
      } else
      dispatch("nr_split_ffwd", {{pool, off(view ? "x_view" : "x")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 2160u * 8u); }
    else if (kernel == "conv" || kernel == "conv_resview" || kernel == "conv_outview" || kernel == "conv_pool") {
      const bool rv = kernel == "conv_resview", ov = kernel == "conv_outview", pl = kernel == "conv_pool"; prm.flags = (rv ? FLAG_RES_VIEW : 0) | (ov ? FLAG_OUT_VIEW : 0) | (pl ? FLAG_EMIT_F16 : 0);
      o_role = ov ? "out_view" : "out"; o_off = off(o_role.c_str()); o_bytes = ov ? VIEW : TIN;
      if (sptp) {
        { id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_tp_prep"); [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:1 + m]; [enc setBuffer:spmax offset:0 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(16u * 512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc); }
        id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_conv_tp"); [enc setComputePipelineState:pso(sptp, "nr_split_conv_tp")];
        [enc setBuffer:pool offset:off("x") atIndex:0]; [enc setBuffer:pool offset:off(rv ? "res_view" : "res") atIndex:1]; [enc setBuffer:weights offset:0 atIndex:2]; [enc setBuffer:pool offset:o_off atIndex:3]; [enc setBuffer:y16 offset:0 atIndex:4]; [enc setBuffer:cov offset:0 atIndex:5];
        [enc setBytes:&prm length:sizeof(prm) atIndex:6]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:7 + m]; [enc setBuffer:spmax offset:0 atIndex:10]; [enc setBuffer:spstats offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(spstats, "launch " + std::to_string(n) + " nr_split_conv_tp");
#endif
      } else
      dispatch("nr_split_conv", {{pool, off("x")}, {pool, off(rv ? "res_view" : "res")}, {weights, 0}, {pool, o_off}, {y16, 0}, {cov, 0}}, 2160u * 32u);
      if (pl) { dispatch("nr_split_pool", {{y16, 0}, {pool, off("pooled")}, {cov2, 0}}, 640u * 32u); outbytes.push_back({"pooled", off("pooled")}); } }
    else if (kernel == "qkv_attend") { o_role = "out"; o_off = off("out"); o_bytes = TIN;
      if (sptp) {
        struct SpPrepParams { uint32_t steps, cols, step_bytes, reserved; } pp{16u, 1536u, 49152u, 0u};
        { id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_tp_prep_generic"); [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep_generic")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:qw[m] offset:0 atIndex:1 + m]; [enc setBuffer:qwmax offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
          [enc dispatchThreads:MTLSizeMake(16u * 1536u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc); }
        id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_split_qkv_tp"); [enc setComputePipelineState:pso(sptp, "nr_split_qkv_tp")];
        [enc setBuffer:pool offset:off("x") atIndex:0]; [enc setBuffer:weights offset:0 atIndex:1]; [enc setBuffer:qb offset:0 atIndex:2]; [enc setBuffer:kb offset:0 atIndex:3]; [enc setBuffer:vb offset:0 atIndex:4]; [enc setBuffer:rsq offset:0 atIndex:5]; [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        for (int m = 0; m < 3; ++m) [enc setBuffer:qw[m] offset:0 atIndex:7 + m]; [enc setBuffer:qwmax offset:0 atIndex:10]; [enc setBuffer:spstats offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(16, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(spstats, "launch " + std::to_string(n) + " nr_split_qkv_tp");
#endif
        // windowed attention on the engine: per-(window, head) K^T / V operands from the live q/k/v, then the attention tile kernel
        enc = NR_TPL(compute(), "body:nr_split_attend_tp_prep"); [enc setComputePipelineState:pso(sptp, "nr_split_attend_tp_prep")]; [enc setBuffer:kb offset:0 atIndex:0]; [enc setBuffer:vb offset:0 atIndex:1]; [enc setBytes:&prm length:sizeof(prm) atIndex:2];
        for (int m = 0; m < 3; ++m) [enc setBuffer:atk[m] offset:0 atIndex:3 + m]; [enc setBuffer:atkmax offset:0 atIndex:6]; for (int m = 0; m < 3; ++m) [enc setBuffer:atv[m] offset:0 atIndex:7 + m]; [enc setBuffer:atvmax offset:0 atIndex:10];
        [enc dispatchThreads:MTLSizeMake(40u * 16u * 64u + 40u * 16u * 2u * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc);
        enc = NR_TPL(compute(), "body:nr_split_attend_tp"); [enc setComputePipelineState:pso(sptp, "nr_split_attend_tp")];
        [enc setBuffer:qb offset:0 atIndex:0]; [enc setBuffer:kb offset:0 atIndex:1]; [enc setBuffer:vb offset:0 atIndex:2]; [enc setBuffer:weights offset:0 atIndex:3]; [enc setBuffer:pool offset:o_off atIndex:4]; [enc setBuffer:cov offset:0 atIndex:5]; [enc setBuffer:rcp offset:0 atIndex:6]; [enc setBytes:&prm length:sizeof(prm) atIndex:7];
        for (int m = 0; m < 3; ++m) [enc setBuffer:atk[m] offset:0 atIndex:8 + m]; [enc setBuffer:atkmax offset:0 atIndex:11]; for (int m = 0; m < 3; ++m) [enc setBuffer:atv[m] offset:0 atIndex:12 + m]; [enc setBuffer:atvmax offset:0 atIndex:15]; [enc setBuffer:spstats offset:0 atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake(16, 40 * 4, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(spstats, "launch " + std::to_string(n) + " nr_split_attend_tp");
#endif
      } else {
      dispatch("nr_split_qkv", {{pool, off("x")}, {weights, 0}, {qb, 0}, {kb, 0}, {vb, 0}, {rsq, 0}}, 2160u * 16u);
      dispatch("nr_split_attend", {{qb, 0}, {kb, 0}, {vb, 0}, {weights, 0}, {pool, o_off}, {cov, 0}, {rcp, 0}}, 40u * 16u * 64u); } }
    else if (kernel == "final_head") { o_role = "out_vit"; o_off = off("out_vit"); o_bytes = VITB; dispatch("nr_split_final_head", {{pool, off("x_pooled")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 640u * 64u); }
    else if (kernel == "dec_upsample") { o_role = "out"; o_off = off("out"); o_bytes = TIN; dispatch("nr_split_dec_upsample", {{pool, off("x_vit")}, {pool, off("res")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 2160u * 32u); }
    else throw std::runtime_error("unknown kernel " + kernel);
    bool encode_validator = true; size_t validator_elements = 0;
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "middle-coverage" && n == 25) poke(cov, 0, 2);                                   // duplicate: word 0 becomes 2 (was 1)
    if (fault == "coverage-compensating" && n == 25) { poke(cov, 40, 0); poke(cov, 44, 2); }        // word 10 -> 0 and word 11 -> 2: same total, must still fault
    if (fault == "coverage-padding-write" && o_role == "out_vit") { size_t i = 0; while (i < VITB && !is_pad[i]) ++i; poke(cov, 4 * i, 1); }   // a padding byte's counter becomes 1
    if (fault == "validator-skipped" && n == 25) encode_validator = false;                            // expected but never encoded: record stays poisoned, run count short
    if (fault == "validator-partial" && n == 25) validator_elements = o_bytes / 2;                    // ran over half the slice: record.elements != expected
#endif
    LaunchRec lr{n, kernel, {}, {}, ""};
    lr.validators.push_back(validate("launch " + std::to_string(n) + " " + o_role, uint32_t(n), cov, o_bytes, o_role == "out_vit" ? 1u : 0u, o_role == "out_vit" ? expect57b : nil, encode_validator, validator_elements));
    bool encode_digest = true;
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "digest-skipped" && n == 25) encode_digest = false;
#endif
    lr.digests.push_back({o_role, digest("launch " + std::to_string(n) + " " + o_role, uint32_t(n), pool, o_off, o_bytes, nullptr, encode_digest)});
    for (auto& ob : outbytes) { lr.validators.push_back(validate("launch " + std::to_string(n) + " " + ob.first, uint32_t(n), cov2, POOLB, 0u, nil)); lr.digests.push_back({ob.first, digest("launch " + std::to_string(n) + " " + ob.first, uint32_t(n), pool, ob.second, POOLB)}); }
    launches.push_back(lr);
  };
#ifdef NR_EXPERIMENTAL_CHUNKED
  for (NSDictionary* L in chain[@"split_a"]) { const uint32_t nrl = uint32_t(JInt(L, "launch")); nr_chunk_boundary(nrl); nr_cur_launch = nrl; run_split(L); }
#else
  for (NSDictionary* L in chain[@"split_a"]) run_split(L);
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(3);
#endif
  // ---- repack 58 (live input: the final-head output with padding tokens zeroed by the driver's rule, both on the GPU) ----
  NSDictionary* R = chain[@"vit"]; const size_t vit_in_off = size_t(JInt(R, "final_head_offset"));
  id<MTLBuffer> vin = scratch(720896, 0); id<MTLBuffer> tok = scratch(VITB, uint8_t(poison)), rcov = scratch(VITB * 4, 0);
  { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:vin range:NSMakeRange(0, 720896 + 16) value:0]; [bl copyFromBuffer:pool sourceOffset:vit_in_off toBuffer:vin destinationOffset:0 size:VITB]; nr_exp::nr_scope_end(bl);
    const uint32_t nmask = uint32_t(VITB); id<MTLComputeCommandEncoder> e = compute(); [e setComputePipelineState:psoPadMask]; [e setBuffer:vin offset:0 atIndex:0]; [e setBuffer:padmask offset:0 atIndex:1]; [e setBytes:&nmask length:4 atIndex:2]; [e dispatchThreads:MTLSizeMake(VITB, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; nr_exp::nr_scope_end(e);
    launches.push_back({58, "repack_input", {{"repack58_input", digest("repack 58 input (padding zeroed)", 58u, vin, 0, 720896)}}, {}, ""}); }
  auto run_repack = [&](const char* name, long n, id<MTLBuffer> inb, size_t in_bytes, id<MTLBuffer> outb, size_t out_off, size_t out_bytes) {
    RepackParams prm{20, 32, uint32_t(in_bytes / 4), uint32_t(out_bytes / 4)};
    fill(rcov, 0, VITB * 4 + 16, 0);
    id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), std::string("body:") + name); [enc setComputePipelineState:pso(vlib, name)]; [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:out_off atIndex:1]; [enc setBuffer:rcov offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof(prm) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "repack-coverage" && n == 58) poke(rcov, 28, 0);   // dword 7 never written
#endif
    LaunchRec lr{n, name, {}, {}, ""}; lr.validators.push_back(validate(std::string("repack ") + std::to_string(n), uint32_t(n), rcov, out_bytes / 4, 0u, nil)); lr.digests.push_back({name, digest(std::string("repack ") + std::to_string(n), uint32_t(n), outb, out_off, out_bytes)}); launches.push_back(lr);
  };
  run_repack("nr_vit_repack_2d_to_1d", 58, vin, 720896, tok, 0, VITB);
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(4);
#endif
  // ---- ViT blocks 31-38, resident intermediates (validated and digested per block before the next block reuses them) ----
  id<MTLBuffer> expand = scratch(T * 4096, uint8_t(poison)), contract = scratch(T * 1024, uint8_t(poison)), q = scratch(T * 1024, uint8_t(poison)), k = scratch(T * 1024, uint8_t(poison)), v = scratch(T * 1024, uint8_t(poison)), attention = scratch(T * 1024, uint8_t(poison)), projection = scratch(T * 1024, uint8_t(poison));
  id<MTLBuffer> covE = scratch(T * 4096 * 4, 0), covC = scratch(T * 1024 * 4, 0), covQ = scratch(T * 1024 * 4, 0), covK = scratch(T * 1024 * 4, 0), covV = scratch(T * 1024 * 4, 0), covA = scratch(T * 1024 * 4, 0), covP = scratch(T * 1024 * 4, 0);
  id<MTLBuffer> tokens = tok;
  struct TpPrepParams { uint32_t steps, cols, step_bytes, reserved; };
  id<MTLBuffer> tpw[3][3], tpmax[3], tpstats = scratch(64, 0); const uint32_t tp_steps[3] = {32u, 128u, 32u}, tp_cols[3] = {4096u, 1024u, 1024u}, tp_stepb[3] = {131072u, 32768u, 32768u};
  if (tplib) for (int st = 0; st < 3; ++st) { const size_t WH = size_t(tp_steps[st]) * 32u * tp_cols[st]; for (int m = 0; m < 3; ++m) tpw[st][m] = scratch(WH * 2u, 0); tpmax[st] = scratch(size_t(tp_steps[st]) * tp_cols[st], 0); }
  id<MTLBuffer> tqw[3], tqmax; if (tplib) { for (int m = 0; m < 3; ++m) tqw[m] = scratch(32u * 32u * 3072u * 2u, 0); tqmax = scratch(32u * 3072u, 0); }   // qkv weights (blob at +128): 32 steps x 3072 columns
  id<MTLBuffer> tak[3], tav[3], takmax, tavmax; if (tplib) { for (int m = 0; m < 3; ++m) { tak[m] = scratch(32u * 32u * 640u * 2u, 0); tav[m] = scratch(32u * 20u * 32u * 32u * 2u, 0); } takmax = scratch(32u * 640u, 0); tavmax = scratch(32u * 20u * 32u, 0); }   // attention operands per block: K^T and V as l/h/l+h halves + max classes
  for (NSDictionary* B in R[@"blocks"]) {
    const long block = JInt(B, "block"); auto W0 = constant(JStr(B, "w0")), W1 = constant(JStr(B, "w1")), W2 = constant(JStr(B, "w2")), W4 = constant(JStr(B, "w4"));
    VitParams prm{uint32_t(T), uint32_t(block), 0, 0};
    { id<MTLBlitCommandEncoder> bl = blit(); for (id<MTLBuffer> c : {covE, covC, covQ, covK, covV, covA, covP}) [bl fillBuffer:c range:NSMakeRange(0, c.length) value:0]; nr_exp::nr_scope_end(bl); }
    auto dispatch = [&](const char* name, std::vector<id<MTLBuffer>> bufs, size_t threads, uint32_t stage) {
      id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), std::string("body:") + name); [enc setComputePipelineState:pso(vlib, name)];
      for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc);
    };
    if (tplib) {
      auto prep = [&](int st, id<MTLBuffer> W) { TpPrepParams pp{tp_steps[st], tp_cols[st], tp_stepb[st], 0u}; id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_vit_tp_prep"); [enc setComputePipelineState:pso(tplib, "nr_vit_tp_prep")];
        [enc setBuffer:W offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:tpw[st][m] offset:0 atIndex:1 + m]; [enc setBuffer:tpmax[st] offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(size_t(tp_steps[st]) * tp_cols[st], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc); };
      auto tpdispatch = [&](const char* name, int st, std::vector<id<MTLBuffer>> bufs, uint32_t stage) {   // bufs: the accepted kernel's buffers; then params, wl, wh, wlh, maxcls, stats
        id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), std::string("body:") + name); [enc setComputePipelineState:pso(tplib, name)];
        for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; for (int m = 0; m < 3; ++m) [enc setBuffer:tpw[st][m] offset:0 atIndex:bufs.size() + 1 + m]; [enc setBuffer:tpmax[st] offset:0 atIndex:bufs.size() + 4]; [enc setBuffer:tpstats offset:0 atIndex:bufs.size() + 5];
        [enc dispatchThreadgroups:MTLSizeMake(tp_cols[st] / 64, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(tpstats, "block " + std::to_string(block) + " " + name);
#endif
        };
      prep(0, W0); tpdispatch("nr_vit_expand_tp", 0, {tokens, W0, expand, covE}, 1); prep(1, W1); tpdispatch("nr_vit_contract_tp", 1, {expand, tokens, W1, contract, covC}, 2);
      { TpPrepParams pp{32u, 3072u, 98304u, 0u}; id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_vit_tp_prep"); [enc setComputePipelineState:pso(tplib, "nr_vit_tp_prep")];
        [enc setBuffer:W2 offset:128 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:tqw[m] offset:0 atIndex:1 + m]; [enc setBuffer:tqmax offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(32u * 3072u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc);
        enc = NR_TPL(compute(), "body:nr_vit_qkv_tp"); [enc setComputePipelineState:pso(tplib, "nr_vit_qkv_tp")]; std::vector<id<MTLBuffer>> qbv{contract, W2, q, k, v, covQ, covK, covV, rsq};
        for (size_t i = 0; i < qbv.size(); ++i) [enc setBuffer:qbv[i] offset:0 atIndex:i]; prm.stage = 3; [enc setBytes:&prm length:sizeof(prm) atIndex:9]; for (int m = 0; m < 3; ++m) [enc setBuffer:tqw[m] offset:0 atIndex:10 + m]; [enc setBuffer:tqmax offset:0 atIndex:13]; [enc setBuffer:tpstats offset:0 atIndex:14];
        [enc dispatchThreadgroups:MTLSizeMake(32, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(tpstats, "block " + std::to_string(block) + " nr_vit_qkv_tp");
#endif
        }
      { id<MTLComputeCommandEncoder> enc = NR_TPL(compute(), "body:nr_vit_attn_prep"); [enc setComputePipelineState:pso(tplib, "nr_vit_attn_prep")]; [enc setBuffer:k offset:0 atIndex:0]; [enc setBuffer:v offset:0 atIndex:1];
        for (int m = 0; m < 3; ++m) [enc setBuffer:tak[m] offset:0 atIndex:2 + m]; [enc setBuffer:takmax offset:0 atIndex:5]; for (int m = 0; m < 3; ++m) [enc setBuffer:tav[m] offset:0 atIndex:6 + m]; [enc setBuffer:tavmax offset:0 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(32u * 640u + 32u * 20u * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; nr_exp::nr_scope_end(enc);
        enc = NR_TPL(compute(), "body:nr_vit_attention_tp"); [enc setComputePipelineState:pso(tplib, "nr_vit_attention_tp")]; std::vector<id<MTLBuffer>> abf{q, k, v, attention, covA, rcp};
        for (size_t i = 0; i < abf.size(); ++i) [enc setBuffer:abf[i] offset:0 atIndex:i]; prm.stage = 4; [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        for (int m = 0; m < 3; ++m) [enc setBuffer:tak[m] offset:0 atIndex:7 + m]; [enc setBuffer:takmax offset:0 atIndex:10]; for (int m = 0; m < 3; ++m) [enc setBuffer:tav[m] offset:0 atIndex:11 + m]; [enc setBuffer:tavmax offset:0 atIndex:14]; [enc setBuffer:tpstats offset:0 atIndex:15];
        [enc dispatchThreadgroups:MTLSizeMake(32, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; nr_exp::nr_scope_end(enc);
#ifdef NR_EXPERIMENTAL_TP_STATS
        tp_stats_capture(tpstats, "block " + std::to_string(block) + " nr_vit_attention_tp");
#endif
        }
      prep(2, W4); tpdispatch("nr_vit_projection_tp", 2, {attention, contract, W4, projection, covP}, 5);
    } else {
    dispatch("nr_vit_expand", {tokens, W0, expand, covE}, T * 256, 1); dispatch("nr_vit_contract", {expand, tokens, W1, contract, covC}, T * 64, 2);
    dispatch("nr_vit_qkv", {contract, W2, q, k, v, covQ, covK, covV, rsq}, T * 32, 3); dispatch("nr_vit_attention", {q, k, v, attention, covA, rcp}, T * 32, 4); dispatch("nr_vit_projection", {attention, contract, W4, projection, covP}, T * 64, 5);
    }
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "vit-coverage" && block == 31) poke(covE, 12, 0);   // expand byte 3 never written
#endif
    LaunchRec lr{block, "vit_block", {}, {}, ""};
    for (auto& pr : std::vector<std::pair<const char*, std::pair<id<MTLBuffer>, id<MTLBuffer>>>>{{"expand", {covE, expand}}, {"contract", {covC, contract}}, {"q", {covQ, q}}, {"k", {covK, k}}, {"v", {covV, v}}, {"attention", {covA, attention}}, {"projection", {covP, projection}}}) {
      const size_t bytes = std::string(pr.first) == "expand" ? T * 4096 : T * 1024; const std::string label = "block " + std::to_string(block) + " " + pr.first;
      lr.validators.push_back(validate(label, uint32_t(block), pr.second.first, bytes, 0u, nil)); lr.digests.push_back({pr.first, digest(label, uint32_t(block), pr.second.second, 0, bytes)}); }
    launches.push_back(lr);
    std::swap(tokens, projection); if (tokens == tok) {} // next block reads this block's projection; the freed buffer becomes the next projection target
    if (projection == tok) { projection = scratch(T * 1024, uint8_t(poison)); }
  }
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(5);
#endif
  // ---- repack 99 -> pool (launch 100 input), then split launches 100-132 ----
  run_repack("nr_vit_repack_1d_to_2d", 99, tokens, VITB, pool, size_t(JInt(R, "repack99_out_offset")), VITB);
#ifdef NR_EXPERIMENTAL_CHUNKED
  for (NSDictionary* L in chain[@"split_b"]) { const uint32_t nrl = uint32_t(JInt(L, "launch")); nr_chunk_boundary(nrl); nr_cur_launch = nrl; run_split(L); }
#else
  for (NSDictionary* L in chain[@"split_b"]) run_split(L);
#endif
  // ---- middle view: kept in a runtime buffer (CPU SHA-256 at completion = the accepted middle_view_sha256) and digested in place ----
  const size_t fo = size_t(JInt(chain, "final_offset")); id<MTLBuffer> midkeep = scratch(VIEW, 0);
  { id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:pool sourceOffset:fo toBuffer:midkeep destinationOffset:0 size:VIEW]; nr_exp::nr_scope_end(bl); launches.push_back({132, "middle_view", {{"middle_view", digest("middle view (launch 132 output)", 132u, pool, fo, VIEW, nullptr, true, false)}}, {}, ""}); }
  const uint32_t mid_digest_slot = uint32_t(dexp.size() - 1);
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(6);
#endif
  // ---- decoder/output section (launches 133-156) on the same resident pool, restaged on the GPU ----
  NSDictionary* D = chain[@"decoder"]; if (!D) throw std::runtime_error("v12 requires the decoder section (resident frame = middle + decoder)");
  const bool final_only = in.production; const fs::path dtables = JStr(D, "tables");
  const unsigned tw = unsigned(JInt(D, "tex_w")), th = unsigned(JInt(D, "tex_h")); require(in.input_color != nil && in.input_color.pixelFormat == MTLPixelFormatRGBA16Float && in.input_color.width == tw && in.input_color.height == th, "input colour texture (RGBA16F WxH)");
  const size_t texb = size_t(tw) * th * 8, motb = size_t(tw) * th * 4;
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // The accepted runtime copies the caller's colour texture to the CPU here and SHA-256s all 16.6 MB, so
  // that completion can prove the input was immutable across the frame. Two reasons that cannot happen on
  // the game seam. It costs about 14 ms of the encode call, on the render thread; and more importantly the
  // producer wrote that texture EARLIER IN THE SAME UNCOMMITTED COMMAND BUFFER, so a CPU read here does not
  // observe the frame's pixels at all - it is exactly the current-frame readback the encode-only contract
  // forbids. The immutability evidence is dropped, and checks_skipped records that it was.
  std::vector<uint8_t> texin;
  const std::string texin_sha;
  res.checks_skipped.push_back("experimental lane: input-texture immutability (no CPU readback of a current-frame texture on the encode path)");
#else
  std::vector<uint8_t> texin(texb); [in.input_color getBytes:texin.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
  const std::string texin_sha = Sha(texin.data(), texin.size()); res.input_texture_sha256_before = texin_sha;
#endif
  std::vector<uint8_t> hist_before, mot_before;
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // Two more CPU readbacks the game seam cannot afford, and one of them cannot even be correct here.
  //  * the MOTION texture is the caller's, and on this seam the game may write it in the same uncommitted
  //    command buffer, so reading it on the CPU during encode is the same forbidden current-frame read as
  //    the colour texture was. It also costs 8.3 MB of copy plus a SHA-256 every temporal frame.
  //  * the HISTORY texture is ours and was written by the PREVIOUS frame's committed buffer, so reading it
  //    is legal - but it costs 16.6 MB of copy plus a SHA-256 only to re-prove a hash we recorded
  //    ourselves. Continuity is still enforced: a temporal frame is admitted only when the runtime's
  //    history is valid, and the body still refuses one whose recorded hashes do not line up.
  if (!in.reset) { res.history_sha256_before = in.temporal->history_sha256; }
  if (!in.reset) res.checks_skipped.push_back("experimental lane: motion/history byte re-hash on the encode path");
#else
  if (!in.reset) { mot_before.resize(motb); [in.motion getBytes:mot_before.data() bytesPerRow:tw * 4 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; res.motion_sha256 = Sha(mot_before.data(), mot_before.size());
    hist_before.resize(texb); [in.temporal->history getBytes:hist_before.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; require(Sha(hist_before.data(), hist_before.size()) == in.temporal->history_sha256, "history texture content does not match the recorded history hash"); res.history_sha256_before = in.temporal->history_sha256; }
#endif
  // restage: poison the whole pool, put this run's middle view back, apply the live-producer patches, set the control words - all in-buffer
  NSArray* patches = D[@"patches"]; require(patches && patches.count == 5, "descriptor patch table (4 skips + Block0 skip)"); const NrBodyRegion* regs[5] = {&in.skips[0], &in.skips[1], &in.skips[2], &in.skips[3], &in.block0_skip};
  NSArray* ctrl = D[@"control"]; const size_t c0 = [ctrl[0] unsignedLongValue], c1 = [ctrl[1] unsignedLongValue];
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // The only place in this runtime where a throw can happen with an ENCODER OPEN: the patch-region
  // validation sits inside the restage blit. On the game seam the caller's command buffer carries the
  // game's own rendering and must stay committable, so an encoder left open would break the frame for
  // the game, not just for us. The check is hoisted ahead of the encoder; it is the same check on the
  // same values, only earlier, so nothing about the restage changes.
  for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs[i];
    require(r.buffer != nil && r.bytes == bytes && r.offset + bytes <= r.buffer.length && off + bytes <= pool_bytes, JStr(P, "file").c_str()); }
#endif
  { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:pool range:NSMakeRange(0, pool_bytes) value:uint8_t(poison)]; [bl copyFromBuffer:midkeep sourceOffset:0 toBuffer:pool destinationOffset:fo size:VIEW];
    for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs[i];
      require(r.buffer != nil && r.bytes == bytes && r.offset + bytes <= r.buffer.length && off + bytes <= pool_bytes, JStr(P, "file").c_str()); [bl copyFromBuffer:r.buffer sourceOffset:r.offset toBuffer:pool destinationOffset:off size:bytes]; }
    [bl fillBuffer:pool range:NSMakeRange(c0, c1 - c0) value:0xff]; nr_exp::nr_scope_end(bl); }
#ifdef NR_BODY_FAULT_TESTS
  if (fault == "restage-corrupt") poke(pool, pool_bytes - 1, uint8_t(poison ^ 0xffu));   // one poison byte of the restaged pool differs from the CPU-constructed pre-image (pixels unaffected)
#endif
  launches.push_back({133, "decoder_pre_image", {{"decoder_pre_image", digest("decoder pre-image (restaged pool)", 133u, pool, 0, pool_bytes, nullptr, true, true)}}, {}, ""}); const uint32_t pre_digest_slot = uint32_t(dexp.size() - 1);
  auto opt = [&](const char* name) { fs::path f = dtables / name; if (fs::exists(f)) return constant(f); return scratch(16, 0); };
  id<MTLBuffer> bpool = pool, brsq = opt("rtx-rsqrt-half.bin"), brcp = opt("rtx-rcp-half.bin"), brom = opt("sm120-rcp-correction-rom.bin"), bex2u = opt("unit-corrections.bin"), bex2f = opt("fraction-corrections.bin");
  id<MTLBuffer> bcov = scratch(pool_bytes + 4096 - 16, 0); id<MTLBuffer> btexcov = scratch(size_t(tw) * th * 4, 0);
  id<MTLBuffer> bdbg = scratch(64 * 32 * 4, 0);
  id<MTLBuffer> tex155buf = scratch(texb, 0), histpre = scratch(texb, 0), histpost = scratch(texb, 0), motpre = scratch(motb, 0), motpost = scratch(motb, 0);   // texture checkpoints (blit copies for the digest kernel)
  MTLTextureDescriptor* dIn = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; dIn.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; dIn.storageMode = MTLStorageModeShared;
  if (in.output_surface) require(in.output_surface.pixelFormat == MTLPixelFormatRGBA16Float && in.output_surface.width == tw && in.output_surface.height == th, "output surface (RGBA16F WxH)");
  id<MTLTexture> tex[3] = {in.input_color, resident_tex(ctx, dev, "tex:1", dIn), in.output_surface ? in.output_surface : resident_tex(ctx, dev, "tex:2", dIn)};
  std::vector<uint8_t> tpoison(texb, uint8_t(poison)); for (int i = 1; i < 3; ++i) [tex[i] replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:tpoison.data() bytesPerRow:tw * 8];   // before commit: the previous frame's GPU work is complete
  struct DecLaunch { unsigned n; std::string kernel, texmode, p_sha, w_key; id<MTLBuffer> bprm, bw, bfault; size_t params_size, weights_size; uint32_t pool_digest_slot = 0xffffffffu, pool_validator_slot = 0xffffffffu, tex_validator_slot = 0xffffffffu; bool temporal_launch = false, temporal_copy = false; }; std::vector<DecLaunch> decl;
  uint32_t histpre_slot = 0xffffffffu, histpost_slot = 0xffffffffu, motpre_slot = 0xffffffffu, motpost_slot = 0xffffffffu, tex155_slot = 0xffffffffu;
#ifdef NR_EXPERIMENTAL_NATIVE_POST
  // diagnostic mode only (NR_EXP_BODY_DIAG_DIR): the native post block's resident intermediates and a blit copy of the output texture taken right after the
  // head, written at completion next to tex_out_155.bin; nothing is encoded or allocated outside diagnostic mode
  id<MTLBuffer> np_xin = nil, np_y16 = nil, np_y8 = nil, np_texafter = nil, np_texend1 = nil, np_texend2 = nil;
#endif
  for (NSDictionary* L in D[@"launches"]) {
#ifdef NR_EXPERIMENTAL_CHUNKED
    const uint32_t nrl = uint32_t(JInt(L, "launch"));
    nr_chunk_boundary(nrl);
    nr_cur_launch = nrl;
#endif
    DecLaunch dl; dl.n = unsigned(JInt(L, "launch")); const unsigned n = dl.n; dl.temporal_launch = (n == 155u && !in.reset); dl.temporal_copy = (n == 156u && in.temporal != nullptr);
    if (dl.temporal_copy) require(!JStr(L, "lib_temporal").empty(), "a temporal state requires the descriptor's lib_temporal for launch 156 (sign-preserving copy)");
    const std::string libp = (dl.temporal_launch || dl.temporal_copy) ? JStr(L, "lib_temporal") : JStr(L, "lib"); dl.kernel = JStr(L, "kernel"); dl.bw = constant(JStr(L, "weights"), 4096); dl.bprm = constant(JStr(L, "params"), 64); dl.params_size = dl.bprm.length - 64; dl.weights_size = dl.bw.length - 4096; dl.w_key = "file:" + JStr(L, "weights") + ":4096";
    id<MTLBuffer> bex2h = nil;
    if (dl.temporal_launch) {   // v6: audited half-domain EX2 table, pinned and bound only for the temporal post launch
      NSDictionary* Tt = L[@"ex2_half_table"]; require(Tt != nil, "temporal launch 155 requires the descriptor's ex2_half_table {file, sha256}");
      const std::string tpath = JStr(Tt, "file"), tsha = JStr(Tt, "sha256"); auto tb = Read(tpath); const std::string got = Sha(tb.data(), tb.size());
      require(tb.size() == 524288 && got == tsha && got == "f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf", "ex2 half-domain table: size/sha mismatch against the descriptor and the pinned hash");
      bex2h = constant(tpath, 16); res.provenance.emplace_back("ex2_half_table:" + tpath, got); dl.bfault = scratch(64, 0);
#ifdef NR_BODY_FAULT_TESTS
      if (fault == "ex2-table-row-corrupt") { uint32_t* row = static_cast<uint32_t*>(bex2h.contents) + 2u * 0xbb80u; row[0] ^= 1u; }   // real GPU lookup fault: the row of half 0xbb80 no longer matches its argument
#endif
    }
    if (dl.temporal_launch) {   // v5: per-frame constructed record (the resident base record stays untouched and keeps its immutability check)
      id<MTLBuffer> base = dl.bprm; dl.bprm = scratch(dl.params_size, 0); const uint8_t* b0 = static_cast<const uint8_t*>(base.contents); uint8_t* outp = static_cast<uint8_t*>(dl.bprm.contents);
      require(nr_temporal_build_post_record(b0, dl.params_size, tw, th, 1u, 6u, outp), "temporal record construction"); res.temporal_record_sha256 = Sha(outp, dl.params_size);
      const size_t allowed[] = {88, 89, 90, 91, 96, 97, 98, 99, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163};
      for (size_t i = 0; i < dl.params_size; ++i) if (outp[i] != b0[i]) { bool ok = false; for (size_t a : allowed) ok |= (a == i); require(ok, "temporal record deviates from the accepted base outside the marker/dimension words"); } }
    const uint8_t* params = static_cast<const uint8_t*>(dl.bprm.contents);
    NSArray* g = L[@"grid"]; NSArray* b = L[@"block"]; dl.texmode = JStr(L, "textures");
#ifdef NR_EXPERIMENTAL_BODY_OVERRIDE
    // EXPERIMENTAL body override, selected per (descriptor library path, kernel). Fail-closed: the
    // manifest, every library hash and every declared function were verified during preparation, so
    // the only decision here is whether THIS kernel was declared for THIS library. Anything not
    // declared keeps the pinned library, and the resolution is recorded so it can be read off a
    // receipt rather than inferred.
    // The whole selection lives inside this flagged region, and the #else branch below is the
    // ACCEPTED line byte-for-byte, so the copy still reduces to nr_body_runtime_v17.
    id<MTLLibrary> dlib = lib(libp);
    {
      // libp here is the ABSOLUTE library path; a manifest declares the RELATIVE descriptor path, so
      // match on that suffix. Getting this wrong made the override a silent no-op: every kernel fell
      // back to the pinned library while the run still looked like a candidate.
      std::string matched;
      if (ctx.override.active) {
        for (const auto& kv : ctx.override.libraries) {
          const std::string& rel = kv.first;
          if (libp == rel || (libp.size() > rel.size() &&
                              libp.compare(libp.size() - rel.size(), rel.size(), rel) == 0 &&
                              libp[libp.size() - rel.size() - 1] == '/')) { matched = rel; break; }
        }
      }
      auto ovf = matched.empty() ? ctx.override.functions.end() : ctx.override.functions.find(matched);
      const bool declared = !matched.empty() && ovf != ctx.override.functions.end() &&
                            ovf->second.count(dl.kernel) != 0;
      if (declared) {
        dlib = ctx.override.libraries[matched];
        ctx.override.resolved[matched + "|" + dl.kernel] = "override:" + ctx.override.sha256[matched];
      } else if (ctx.override.active) {
        ctx.override.resolved[(matched.empty() ? libp : matched) + "|" + dl.kernel] = "pinned";
      }
    }
    id<MTLComputePipelineState> dps = pso(dlib, dl.kernel.c_str());
#else
    id<MTLComputePipelineState> dps = pso(lib(libp), dl.kernel.c_str());
#endif
#ifdef NR_BODY_SINGLE_WARP
    // OPT-IN CANDIDATE ONLY (docs/NR_LAUNCH155_CENSUS.md, Codex reviews 2026-09-08 05:11 and 06:17). Undefined in every accepted
    // build, so the default executable is byte-identical. When defined, the guard runs HERE - after the ACTUAL pipeline for this
    // launch exists and the ACTUAL dispatch shape is known, but BEFORE any blit or compute encoder for it is opened, so a refusal
    // throws with no encoder left open and nothing committed. It records the width/shape/library it saw and FAILS CLOSED.
    if (n == 155u) {
      NrSingleWarpFacts swf{(unsigned long)dps.threadExecutionWidth, (unsigned long)dps.maxTotalThreadsPerThreadgroup,
                            (unsigned long)[b[0] unsignedIntValue], (unsigned long)[b[1] unsignedIntValue], (unsigned long)[b[2] unsignedIntValue]};
      std::string swwhy; const bool swok = nr_single_warp_facts_ok(swf, swwhy);
      res.provenance.push_back({"single-warp-guard:launch155",
        std::string("{\"ok\": ") + (swok ? "true" : "false") + ", \"why\": \"" + swwhy + "\", \"threadExecutionWidth\": " + std::to_string(swf.thread_execution_width) +
        ", \"maxTotalThreadsPerThreadgroup\": " + std::to_string(swf.max_total_threads) + ", \"threadgroup\": [" + std::to_string(swf.tg_w) + ", " + std::to_string(swf.tg_h) + ", " + std::to_string(swf.tg_d) +
        "], \"library\": \"" + libp + "\", \"kernel\": \"" + dl.kernel + "\"}"});
      require(swok, ("single-warp launch-155 guard refused the dispatch: " + swwhy).c_str());
    }
#endif
    if (n == 155u && in.reset) { require(dl.params_size >= 104, "post block parameter block"); for (size_t i = 88; i < 104; ++i) require(params[i] == 0, "temporal inputs present (launch 155 params +88/+96): unsupported path"); }   // see docs/NR_INPUT_CONTRACT.md
    dl.p_sha = Sha(dl.bprm.contents, dl.params_size);   // immutability of the (possibly per-frame) record: re-hashed at completion
    if (dl.temporal_launch) { tex_to_buf(in.temporal->history, histpre, 8); histpre_slot = digest("history before 155", 155u, histpre, 0, texb, hist_before.data(), true, false); tex_to_buf(in.motion, motpre, 4); motpre_slot = digest("motion before 155", 155u, motpre, 0, motb, mot_before.data(), true, false); }
#ifdef NR_EXPERIMENTAL_NATIVE_RANGE
    // CANDIDATE ONLY (build/nr-accept-v1/DISPATCH-EXTENSION-PROPOSAL-NATIVE-RANGE.json). A manifest-declared launch range is
    // dispatched as the exact dense C256 block sequence instead of the PTX kernel. Admission reads THIS launch's pinned params
    // and fails the frame on any disagreement - there is no silent fallback. Pool equivalence: the launch's output tensor is
    // exported into its own pool range and its control words are zero-filled, exactly the bytes the PTX launch writes.
    if (NrBodyOverride::NativeRange* nrr = [&]() -> NrBodyOverride::NativeRange* {
          if (!ctx.override.active) return nullptr;
          for (auto& r : ctx.override.native_ranges) if (n >= r.first && n <= r.last) {
#ifdef NR_EXPERIMENTAL_NATIVE_POST
            if (r.post && dl.temporal_launch) { r.resolution = "temporal-ptx"; return nullptr; }   // temporal frames keep the generic lib_temporal PTX dispatch, untouched
#endif
            return &r; }
          return nullptr; }()) {
      struct NrdPrepack { uint32_t kind, count; }; struct NrdTileShape { uint32_t tiles; }; struct NrdSeed { uint32_t rows, kind; };
      struct NrdLinear { uint32_t rows, out_width, in_stride, chunks, kind, seeded; }; struct WideWindowMap { uint32_t width, height, sgw, sgh, dgw, dgh; int32_t sx, sy, dx, dy; };
      struct EncoderExportShape { uint32_t width, height, channels, planar; }; struct NrdImportShape { uint32_t width, height, channels, grid_width, grid_height; };
      auto u64 = [&](size_t o) { uint64_t v = 0; std::memcpy(&v, params + o, 8); return v; };
      auto i32 = [&](size_t o) { int32_t v = 0; std::memcpy(&v, params + o, 4); return v; };
#ifdef NR_EXPERIMENTAL_NATIVE_POST
      if (nrr->post) {
        // POST-BLOCK-NATIVE-EXACT-v1 (experiments/POST-BLOCK-NATIVE-EXACT-v1.json, FUSED-POST-REVISION-1.RESULT: byte-exact on the recorded launch): launch 155 on a
        // reset frame = entry (154 tensor x S1, Block0 x S2 as half2 fma, satfinite e4m3; the unquantised half kept for the contract prior) -> the C32 chained block
        // body over the 1920x1152 pixel grid (window origin -4,-4; the gated fused pairs H/A/D/E/B of EXACT-C32-FUSION-GATE-v1; nr_post_runner mode candf) -> the
        // extra fp16 32->16 linear on the T5 tile -> the reset head (one colour tap, the exact fma sequence, clamp, RZ half store). No pool bytes are written.
        // Admission reads THIS launch's pinned record (LAYOUT-TRACE.json) and fails the frame on any disagreement - no silent fallback.
        const uint64_t NR_PTX_POOL_BASE = 0x100000000ull;
        const uint32_t W = 32, H = 128, IW = 1920, IH = 1152;
        const size_t T154 = size_t(2) * (IH / 2) * (IW / 2) * 16, TB0 = size_t(IW / 4) * (IH / 4) * 16 * W, LAYER_OFF = 512, LAYER = 21808, WBLOCK = 20672;
        require(dl.kernel == "ptx_post_block_swin_1h_32", "native post: launch kernel is not ptx_post_block_swin_1h_32");
        require(dl.params_size == 184, "native post: launch-155 record must be 184 bytes");
        require(dl.texmode == "post", "native post: launch textures are not the post set");
        require(dl.weights_size >= LAYER_OFF + LAYER, "native post: weight record shorter than the layer record");
        const uint64_t a0 = u64(0), a8 = u64(8);
        require(a0 >= NR_PTX_POOL_BASE && a0 - NR_PTX_POOL_BASE + T154 <= pool_bytes && a8 >= NR_PTX_POOL_BASE && a8 - NR_PTX_POOL_BASE + TB0 <= pool_bytes, "native post: tensor pointers outside the pool");
        require(i32(32) == int32_t(IH) && i32(36) == int32_t(IW), "native post: pixel grid is not 1920x1152");
        const int32_t ox = i32(40), oy = i32(44);
        require((ox == 0 || ox == -4) && (oy == 0 || oy == -4), "native post: window origin");
        const uint32_t gw = uint32_t((int32_t(IW) - ox + 7) / 8), gh = uint32_t((int32_t(IH) - oy + 7) / 8);
        require([g[0] unsignedIntValue] == gw && [g[1] unsignedIntValue] == gh && [g[2] unsignedIntValue] == 1u, "native post: descriptor grid disagrees with the params origin");
        auto f32 = [&](size_t o) { float v = 0; std::memcpy(&v, params + o, 4); return v; };
        const int32_t bw = i32(172), bh = i32(176);
        require(bw > 0 && bh > 0 && uint32_t(bw) <= tw && uint32_t(bh) <= th, "native post: colour bounds outside the output texture");
        const size_t in154 = size_t(a0 - NR_PTX_POOL_BASE), inb0 = size_t(a8 - NR_PTX_POOL_BASE);
        const size_t rows_all = size_t(gw) * gh * 64; const uint32_t rows_all32 = uint32_t(rows_all);
        // chunking by window rows: the standalone gate used 29 window rows per chunk (5 chunks); 15 halves the resident scratch (about 0.9 GB) at 10 chunks -
        // dispatch geometry only, every kernel is per row / per window and every chunk boundary is a window row
        const uint32_t CH_ROWS = 15, nchunks = (gh + CH_ROWS - 1) / CH_ROWS; const size_t max_rows = size_t(CH_ROWS) * gw * 64;
        auto rb = [&](const std::string& key, size_t nbytes) { return resident_buf(ctx, dev, "native-range:post:" + key, nbytes + 16); };
        id<MTLBuffer> xin = rb("xin", rows_all * W), ywall = rb("yw", rows_all * W * 2), y8all = rb("y8", rows_all * W * 2), y16 = rb("y16", rows_all * 16 * 2);
        id<MTLBuffer> XL0 = rb("XL0", max_rows * W * 2), XH0 = rb("XH0", max_rows * W * 2), XL1 = rb("XL1", max_rows * H * 2), XH1 = rb("XH1", max_rows * H * 2);
        id<MTLBuffer> S2 = rb("S2", max_rows * W * 2), y2 = rb("y2", max_rows * W * 2), XL2 = rb("XL2", max_rows * W * 2), XH2 = rb("XH2", max_rows * W * 2), y3 = rb("y3", max_rows * 96 * 2);
        id<MTLBuffer> QL = rb("QL", max_rows * 64 * 2), QH = rb("QH", max_rows * 64 * 2), y5 = rb("y5", max_rows * 64 * 2), PL = rb("PL", max_rows * 64 * 2), PH = rb("PH", max_rows * 64 * 2);
        id<MTLBuffer> VL = rb("VL", (max_rows / 64) * 2048 * 2), VH = rb("VH", (max_rows / 64) * 2048 * 2), XL7 = rb("XL7", max_rows * W * 2), XH7 = rb("XH7", max_rows * W * 2), S8 = rb("S8", max_rows * W * 2);
        id<MTLBuffer> Sz = rb("Sz", 16);   // never read: the unseeded stages take the half(0) branch
        struct StageDef32 { uint32_t kind, out_width, in_stride, chunks, seeded; size_t count; };
        const StageDef32 defs[4] = { {0, 128, 32, 1, 0, size_t(128) * 32}, {1, 32, 128, 4, 1, size_t(4) * 32 * 32}, {2, 96, 32, 1, 0, size_t(96) * 32}, {3, 32, 32, 1, 1, size_t(32) * 32} };
        std::array<id<MTLBuffer>, 4> WL, WHb, TC;
        for (int s = 0; s < 4; ++s) { const std::string k = std::to_string(s); WL[s] = rb("WL:" + k, defs[s].count * 2); WHb[s] = rb("WH:" + k, defs[s].count * 2); TC[s] = rb("TC:" + k, (defs[s].count / 32 / 8) * 4); }
        // the layer record (weights +512, 21808 B) in the chained layout the C32 engine offsets address: blk[0:8272) = layer[0:8272), blk[8288:20672) = layer[8400:20784)
        // (LAYOUT-TRACE: the two scale blocks S1/S2 sit at layer 8272..8400 and are read by the entry from the layer itself)
        id<MTLBuffer> wblock = rb("blk", WBLOCK);
        { id<MTLBlitCommandEncoder> bl = blit();
          [bl fillBuffer:wblock range:NSMakeRange(0, WBLOCK) value:0];
          [bl copyFromBuffer:dl.bw sourceOffset:LAYER_OFF toBuffer:wblock destinationOffset:0 size:8272];
          [bl copyFromBuffer:dl.bw sourceOffset:LAYER_OFF + 8400 toBuffer:wblock destinationOffset:8288 size:WBLOCK - 8288];
          [bl fillBuffer:btexcov range:NSMakeRange(0, size_t(tw) * th * 4 + 16) value:0];
          nr_exp::nr_scope_end(bl); }
        struct NrpEntry { uint32_t gw, gh; int32_t ox, oy; uint32_t iw, ih, tw, th; };
        struct NrpHead { uint32_t gw, gh; int32_t ox, oy; uint32_t out_w, out_h; float scale; uint32_t clamp_flag; float uvA[2], uvB[2], uvC[2]; uint32_t bw, bh; float fbw, fbh; uint32_t has_colour; uint32_t pad_; };
        id<MTLTexture> tex_out = in.temporal ? tex[2] : tex[1];   // the post product's texture, exactly as the generic dispatch binds texture 1 (a temporal state may be present on a reset frame)
        NrpEntry ep{gw, gh, ox, oy, IW, IH, IW / 2, IH / 2};
        NrpHead hp{gw, gh, ox, oy, tw, th, f32(48), uint32_t(i32(52)), {f32(72), f32(76)}, {f32(64), f32(68)}, {f32(80), f32(84)}, uint32_t(bw), uint32_t(bh), float(bw), float(bh), 1u, 0u};
        id<MTLComputeCommandEncoder> e = compute();
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
        if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " native:" + nrr->kind;
#endif
        auto t64 = [&](size_t count) { [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; };
        auto tg32 = [&](size_t gx, size_t gy) { [e dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; };
        auto bind = [&](std::initializer_list<std::pair<id<MTLBuffer>, NSUInteger>> bs) { for (const auto& bi : bs) [e setBuffer:bi.first offset:0 atIndex:bi.second]; };
        id<MTLLibrary> DL = nrr->library;
        auto p32 = [&](const char* fn) { id<MTLComputePipelineState> p = pso(DL, fn); require(p.threadExecutionWidth == 32, "native post: a fused 32-lane kernel needs 32-thread SIMD groups"); return p; };
        [e setComputePipelineState:pso(DL, "nrp_entry")]; [e setBuffer:pool offset:in154 atIndex:0]; [e setBuffer:pool offset:inb0 atIndex:1]; [e setBuffer:dl.bw offset:LAYER_OFF atIndex:2]; bind({{xin, 3}, {ywall, 5}}); [e setBytes:&ep length:sizeof ep atIndex:4]; t64(rows_all * 16);
        for (int s = 0; s < 4; ++s) {
          NrdPrepack pp{defs[s].kind, uint32_t(defs[s].count)};
          [e setComputePipelineState:pso(DL, "nrd32_prepack_exact")]; bind({{wblock, 0}, {WL[s], 1}, {WHb[s], 2}}); [e setBytes:&pp length:sizeof pp atIndex:3]; t64(defs[s].count);
          NrdTileShape ts{uint32_t((defs[s].count / 32 / 8) * 4)};
          [e setComputePipelineState:pso(DL, "nrd_tile_class")]; bind({{WHb[s], 0}, {TC[s], 1}}); [e setBytes:&ts length:sizeof ts atIndex:2]; t64(ts.tiles);
        }
        for (uint32_t c = 0; c < nchunks; ++c) {
          const uint32_t wr0 = c * CH_ROWS, wr1 = std::min(gh, wr0 + CH_ROWS); const size_t rows = size_t(wr1 - wr0) * gw * 64, roff = size_t(wr0) * gw * 64;
          const uint32_t rows32 = uint32_t(rows), windows = uint32_t(rows / 64); const size_t xoff = roff * W;
          auto shape = [&](NSUInteger idx) { [e setBytes:&rows32 length:4 atIndex:idx]; };
          auto linear = [&](int s, id<MTLBuffer> XL, id<MTLBuffer> XH, id<MTLBuffer> Sb, id<MTLBuffer> o, NSUInteger ooff) { const StageDef32& d = defs[s]; NrdLinear lp{rows32, d.out_width, d.in_stride, d.chunks, 2u, d.seeded};
            [e setComputePipelineState:pso(DL, "nrd_linear_exact_ws4")]; bind({{XL, 0}, {XH, 1}, {WL[s], 2}, {WHb[s], 3}, {TC[s], 4}, {Sb, 5}}); [e setBuffer:o offset:ooff atIndex:6]; [e setBytes:&lp length:sizeof lp atIndex:7]; tg32(d.out_width / 8, rows / 32); };
          { const uint32_t cnt = uint32_t(rows * W); [e setComputePipelineState:pso(DL, "nrd_split_bytes")]; [e setBuffer:xin offset:xoff atIndex:0]; bind({{XL0, 1}, {XH0, 2}}); [e setBytes:&cnt length:4 atIndex:3]; t64(rows * W); }
          [e setComputePipelineState:p32("nrd32_expand_activate_split_exact_ws4")]; bind({{XL0, 0}, {XH0, 1}, {WL[0], 2}, {WHb[0], 3}, {TC[0], 4}, {XL1, 5}, {XH1, 6}}); shape(7); tg32(H / 8, rows / 32);   // H: expand + activate + split
          { NrdSeed sp{rows32, 2u}; [e setComputePipelineState:pso(DL, "nrd32_seed")]; [e setBuffer:xin offset:xoff atIndex:0]; [e setBuffer:ywall offset:roff * W * 2 atIndex:1]; bind({{wblock, 2}, {S2, 3}}); [e setBytes:&sp length:sizeof sp atIndex:4]; t64(rows * W); }   // contract prior from the unquantised entry half
          { const StageDef32& d = defs[1]; NrdLinear lp{rows32, d.out_width, d.in_stride, d.chunks, 2u, d.seeded};   // A: contract + quant split, half output kept for the seeds
            [e setComputePipelineState:p32("nrd_linear_quant_split_keep_exact_ws4")]; bind({{XL1, 0}, {XH1, 1}, {WL[1], 2}, {WHb[1], 3}, {TC[1], 4}, {S2, 5}, {y2, 6}, {XL2, 7}, {XH2, 8}}); [e setBytes:&lp length:sizeof lp atIndex:9]; tg32(W / 8, rows / 32); }
          linear(2, XL2, XH2, Sz, y3, 0);
          [e setComputePipelineState:pso(DL, "nrd32_normalize_split")]; bind({{y3, 0}, {wblock, 1}, {QL, 2}, {QH, 3}}); shape(4); t64(rows * 2);   // D
          [e setComputePipelineState:pso(DL, "nrd32_scores_exact_ws4")]; bind({{QL, 0}, {QH, 1}, {wblock, 2}, {y5, 3}}); shape(4); tg32(8, size_t(windows) * 2);
          [e setComputePipelineState:pso(DL, "nrd_probabilities_perm_split")]; bind({{y5, 0}, {PL, 1}, {PH, 2}}); shape(3); t64(rows);   // E
          { const uint32_t cnt = uint32_t(size_t(windows) * 2048); [e setComputePipelineState:pso(DL, "nrd32_perm_values")]; bind({{y3, 0}, {VL, 1}, {VH, 2}}); [e setBytes:&cnt length:4 atIndex:3]; t64(cnt); }
          [e setComputePipelineState:p32("nrd32_values_quant_split_exact_ws4")]; bind({{PL, 0}, {PH, 1}, {VL, 2}, {VH, 3}, {XL7, 4}, {XH7, 5}}); shape(6); tg32(W / 8, size_t(windows) * 2);   // B
          { NrdSeed sp{rows32, 1u}; [e setComputePipelineState:pso(DL, "nrd32_seed")]; [e setBuffer:xin offset:xoff atIndex:0]; bind({{y2, 1}, {wblock, 2}, {S8, 3}}); [e setBytes:&sp length:sizeof sp atIndex:4]; t64(rows * W); }
          linear(3, XL7, XH7, S8, y8all, roff * W * 2);
        }
        [e setComputePipelineState:pso(DL, "nrp_head_linear")]; bind({{y8all, 0}, {y16, 2}}); [e setBuffer:dl.bw offset:LAYER_OFF atIndex:1]; [e setBytes:&rows_all32 length:4 atIndex:3]; t64(rows_all * 16);
        [e setComputePipelineState:pso(DL, "nrp_head")]; bind({{y16, 0}, {brom, 1}, {btexcov, 2}}); [e setBytes:&hp length:sizeof hp atIndex:3]; [e setTexture:tex[0] atIndex:0]; [e setTexture:tex_out atIndex:1];
        [e dispatchThreads:MTLSizeMake(tw, th, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        nr_exp::nr_scope_end(e);
        if (diag) { np_xin = xin; np_y16 = y16; np_y8 = y8all; np_texafter = rb("diag-texafter", texb); tex_to_buf(tex_out, np_texafter, 8); }
        nrr->resolution = "native";
        LaunchRec lr{long(n), "native:" + nrr->kind, {}, {}, ""};
        launches.push_back(lr); decl.push_back(dl);
        continue;
      }
#endif
      if (nrr->kind == "wide32-chained-exact-dense-v2") {
        // The 1h C32 tier (build/nr-accept-v1/experiments/NR-TIER-1H-C32-v1*.json, NR-RANGE-133-154-v1.json): the swin32 WALK layout, not the wide
        // formulas; 1h params are 96 B (tokens @24, origin @32, previous control words @40, own control words @56, skip @80). Every QMMA site is an
        // exact dense C32 kernel; activate/normalize/probabilities/pack/rewindow/export come from the frozen prefix.metallib swin32 engine. The
        // upsample launch's contract prior multiplies the UNQUANTIZED pre-stage half (151 PTX lines 1514/2114/3608): admitted at origin 0,0 only.
        const uint64_t NR_PTX_POOL_BASE = 0x100000000ull;
        const uint32_t W = 32, H = 128, IW = 960, IH = 576, CGW = IW / 8, CGH = IH / 8;
        const size_t TENSOR = size_t(IW / 4) * (IH / 4) * 16 * W, SPLIT = size_t(IW / 2) * (IH / 2) * 64;
        const bool outview = dl.kernel == "ptx_swin_1h_32_1_outview_wait", upsample = dl.kernel == "ptx_swin_1h_32_1_upsample_tilesync";
        require(dl.kernel == "ptx_swin_1h_32_1_chained" || outview || upsample, "native range (C32): launch kernel is not ptx_swin_1h_32_1_chained, _outview_wait or _upsample_tilesync");
        require(dl.params_size >= 88, "native range (C32): launch params shorter than 88 bytes");
        if (upsample) require(n == nrr->first, "native range (C32): an upsample_tilesync launch must be the first launch of its range");
        const uint64_t a0 = u64(0), a8 = u64(8), a56 = u64(56), a80 = upsample ? u64(80) : 0;
        require(a0 >= NR_PTX_POOL_BASE && a0 - NR_PTX_POOL_BASE + (upsample ? SPLIT : TENSOR) <= pool_bytes && a8 >= NR_PTX_POOL_BASE && a8 - NR_PTX_POOL_BASE + TENSOR <= pool_bytes, "native range (C32): tensor pointers outside the pool");
        if (upsample) require(a80 >= NR_PTX_POOL_BASE && a80 - NR_PTX_POOL_BASE + TENSOR <= pool_bytes, "native range (C32): skip tensor pointer outside the pool");
        require(i32(24) == int32_t(IH) && i32(28) == int32_t(IW), "native range (C32): token dimensions are not 576x960");
        const int32_t ox = i32(32), oy = i32(36);
        require((ox == 0 || ox == -4) && (oy == 0 || oy == -4), "native range (C32): window origin");
        if (upsample) require(ox == 0 && oy == 0, "native range (C32): the upsample launch is admitted at window origin 0,0 only");
        const uint32_t gw = uint32_t((int32_t(IW) - ox + 7) / 8), gh = uint32_t((int32_t(IH) - oy + 7) / 8);
        require([g[0] unsignedIntValue] == gw && [g[1] unsignedIntValue] == gh && [g[2] unsignedIntValue] == 1u, "native range (C32): descriptor grid disagrees with the params origin");
        const size_t in_off = size_t(a0 - NR_PTX_POOL_BASE), out_off = size_t(a8 - NR_PTX_POOL_BASE);
        if (outview) require(a56 == 0, "native range (C32): an outview_wait launch must not carry control words");
        else require(a56 >= NR_PTX_POOL_BASE && size_t(a56 - NR_PTX_POOL_BASE) >= c0 && size_t(a56 - NR_PTX_POOL_BASE) + size_t(gw) * gh * 4 <= c1, "native range (C32): control words outside the control region");
        const size_t sync_off = outview ? 0 : size_t(a56 - NR_PTX_POOL_BASE);
        require(dl.weights_size >= (upsample ? size_t(22768) : size_t(20656)), "native range (C32): weight record shorter than the walk layout");
        static thread_local unsigned nr32_prev_n = 0; static thread_local size_t nr32_prev_out = 0;
        if (n == nrr->first) nrr->resolution.clear();
        else require(nr32_prev_n + 1 == n && nr32_prev_out == in_off, "native range (C32): launches are not chained");
        nr32_prev_n = n; nr32_prev_out = out_off;
        const size_t rows = size_t(gw) * gh * 64, max_rows = size_t(CGW + 1) * (CGH + 1) * 64, canon_rows = size_t(CGW) * CGH * 64;
        const uint32_t windows = uint32_t(rows / 64), rows32 = uint32_t(rows);
        auto rb = [&](const std::string& key, size_t nbytes) { return resident_buf(ctx, dev, "native-range:c32:" + key, nbytes + 16); };
        id<MTLBuffer> canon = rb("canon", canon_rows * W), xin = rb("xin", max_rows * W), rwc_in = rb("rwc-in", max_rows * W * 4), rwc_out = rb("rwc-out", canon_rows * W * 4), expc = rb("export-counts", TENSOR * 4);
        id<MTLBuffer> y0 = rb("y0", max_rows * H * 2), y1 = rb("y1", max_rows * H), cv1 = rb("c1", max_rows * H * 4), y2 = rb("y2", max_rows * W * 2), y3 = rb("y3", max_rows * 96 * 2);
        id<MTLBuffer> y4 = rb("y4", max_rows * 64), cv4 = rb("c4", max_rows * 64 * 4), y5 = rb("y5", max_rows * 64 * 2), y6 = rb("y6", max_rows * 64), cv6 = rb("c6", max_rows * 64 * 4);
        id<MTLBuffer> y7 = rb("y7", max_rows * W * 2), y8 = rb("y8", max_rows * W * 2), y9 = rb("y9", max_rows * W), cv9 = rb("c9", max_rows * W * 4);
        id<MTLBuffer> XL0 = rb("XL0", max_rows * W * 2), XH0 = rb("XH0", max_rows * W * 2), XL1 = rb("XL1", max_rows * H * 2), XH1 = rb("XH1", max_rows * H * 2), XL2 = rb("XL2", max_rows * W * 2), XH2 = rb("XH2", max_rows * W * 2);
        id<MTLBuffer> QL = rb("QL", max_rows * 64 * 2), QH = rb("QH", max_rows * 64 * 2), PL = rb("PL", max_rows * 64 * 2), PH = rb("PH", max_rows * 64 * 2), VL = rb("VL", (max_rows / 64) * 2048 * 2), VH = rb("VH", (max_rows / 64) * 2048 * 2);
        id<MTLBuffer> XL7 = rb("XL7", max_rows * W * 2), XH7 = rb("XH7", max_rows * W * 2), S2 = rb("S2", max_rows * W * 2), S8 = rb("S8", max_rows * W * 2), Sz = rb("Sz", max_rows * H * 2), canonV = rb("canonV", canon_rows * W * 2);
        struct StageDef32 { uint32_t kind, out_width, in_stride, chunks, seeded; size_t count; };
        const StageDef32 defs[4] = { {0, 128, 32, 1, 0, size_t(128) * 32}, {1, 32, 128, 4, 1, size_t(4) * 32 * 32}, {2, 96, 32, 1, 0, size_t(96) * 32}, {3, 32, 32, 1, 1, size_t(32) * 32} };
        std::array<id<MTLBuffer>, 4> WL, WHb, TC;
        for (int s = 0; s < 4; ++s) { const std::string k = std::to_string(n) + ":" + std::to_string(s); WL[s] = rb("WL:" + k, defs[s].count * 2); WHb[s] = rb("WH:" + k, defs[s].count * 2); TC[s] = rb("TC:" + k, (defs[s].count / 32 / 8) * 4); }
        id<MTLBuffer> wblock = dl.bw;
        if (upsample) {   // byte copies only: blk[0:8192] = w[0:8192], contract scales blk[8208:8272] = w[10256:10320], blk[8288:20656] = w[10400:22768]
          wblock = rb("blk:" + std::to_string(n), 20672);
          id<MTLBlitCommandEncoder> bl = blit();
          [bl fillBuffer:wblock range:NSMakeRange(0, 20672) value:0];
          [bl copyFromBuffer:dl.bw sourceOffset:0 toBuffer:wblock destinationOffset:0 size:8192];
          [bl copyFromBuffer:dl.bw sourceOffset:10256 toBuffer:wblock destinationOffset:8208 size:64];
          [bl copyFromBuffer:dl.bw sourceOffset:10400 toBuffer:wblock destinationOffset:8288 size:12368];
          nr_exp::nr_scope_end(bl);
        }
        id<MTLComputeCommandEncoder> e = compute();
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
        if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " native:" + nrr->kind + (nrr->fused32 ? "+fused32" : "");
#else
        if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " native:" + nrr->kind;
#endif
#endif
        auto t64 = [&](size_t count) { [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; };
        auto tg32 = [&](size_t gx, size_t gy) { [e dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; };
        auto bind = [&](std::initializer_list<std::pair<id<MTLBuffer>, NSUInteger>> bs) { for (const auto& bi : bs) [e setBuffer:bi.first offset:0 atIndex:bi.second]; };
        auto shape = [&](NSUInteger idx) { [e setBytes:&rows32 length:4 atIndex:idx]; };
        id<MTLLibrary> DL = nrr->library, EL = nrr->engine;
        if (upsample) {   // pre-block stage -> canon + the unquantized half (standalone evidence: build/nr-native-tier-v1/s133/s151/run-D1)
          struct NrdUpImport { uint32_t rows, width, height, channels; }; struct NrdUpCombine { uint32_t grid_width, grid_height, coarse_width; };
          const size_t up_rows = ((size_t(IW / 2) * (IH / 2) + 31) / 32) * 32, up_count = size_t(2) * 32 * 32;
          id<MTLBuffer> uskip = rb("up-skip", canon_rows * W), uXL = rb("up-XL", up_rows * 64 * 2), uXH = rb("up-XH", up_rows * 64 * 2), uWL = rb("up-WL", up_count * 2), uWH = rb("up-WH", up_count * 2);
          id<MTLBuffer> uTC = rb("up-TC", (up_count / 32 / 8) * 4), uS = rb("up-S", up_rows * W * 2), uU = rb("up-U", up_rows * W * 2);
          NrdImportShape is{IW, IH, W, CGW, CGH};
          [e setComputePipelineState:pso(DL, "nrd_import_tensor")]; [e setBuffer:pool offset:size_t(a80 - NR_PTX_POOL_BASE) atIndex:0]; [e setBuffer:uskip offset:0 atIndex:1]; [e setBytes:&is length:sizeof is atIndex:2]; t64(canon_rows * W);
          NrdUpImport ui{uint32_t(up_rows), IW / 2, IH / 2, 64};
          [e setComputePipelineState:pso(DL, "nrd_up_import")]; [e setBuffer:pool offset:in_off atIndex:0]; bind({{uXL, 1}, {uXH, 2}}); [e setBytes:&ui length:sizeof ui atIndex:3]; t64(up_rows * 64);
          NrdPrepack upp{4, uint32_t(up_count)};
          [e setComputePipelineState:pso(DL, "nrd32_prepack_exact")]; [e setBuffer:dl.bw offset:0 atIndex:0]; bind({{uWL, 1}, {uWH, 2}}); [e setBytes:&upp length:sizeof upp atIndex:3]; t64(up_count);
          NrdTileShape uts{uint32_t((up_count / 32 / 8) * 4)};
          [e setComputePipelineState:pso(DL, "nrd_tile_class")]; bind({{uWH, 0}, {uTC, 1}}); [e setBytes:&uts length:sizeof uts atIndex:2]; t64(uts.tiles);
          NrdLinear ul{uint32_t(up_rows), W, 64, 2, 2, 0};
          [e setComputePipelineState:pso(DL, "nrd_linear_exact_ws4")]; bind({{uXL, 0}, {uXH, 1}, {uWL, 2}, {uWH, 3}, {uTC, 4}, {uS, 5}, {uU, 6}}); [e setBytes:&ul length:sizeof ul atIndex:7]; tg32(W / 8, up_rows / 32);
          NrdUpCombine uc{CGW, CGH, IW / 2};
          [e setComputePipelineState:pso(DL, "nrd32_up_combine_v")]; bind({{uskip, 0}, {uU, 1}, {canon, 3}, {canonV, 4}}); [e setBuffer:dl.bw offset:0 atIndex:2]; [e setBytes:&uc length:sizeof uc atIndex:5]; t64(canon_rows * W);
        } else if (n == nrr->first) {
          NrdImportShape is{IW, IH, W, CGW, CGH};
          [e setComputePipelineState:pso(DL, "nrd_import_tensor")]; [e setBuffer:pool offset:in_off atIndex:0]; [e setBuffer:canon offset:0 atIndex:1]; [e setBytes:&is length:sizeof is atIndex:2]; t64(canon_rows * W);
        }
        { WideWindowMap to{IW, IH, CGW, CGH, gw, gh, 0, 0, ox, oy}; [e setComputePipelineState:pso(EL, "nr_swin32_rewindow")]; bind({{canon, 0}, {xin, 1}, {rwc_in, 2}}); [e setBytes:&to length:sizeof to atIndex:3]; t64(rows * W); }
        for (int s = 0; s < 4; ++s) {
          NrdPrepack pp{defs[s].kind, uint32_t(defs[s].count)};
          [e setComputePipelineState:pso(DL, "nrd32_prepack_exact")]; [e setBuffer:wblock offset:0 atIndex:0]; bind({{WL[s], 1}, {WHb[s], 2}}); [e setBytes:&pp length:sizeof pp atIndex:3]; t64(defs[s].count);
          NrdTileShape ts{uint32_t((defs[s].count / 32 / 8) * 4)};
          [e setComputePipelineState:pso(DL, "nrd_tile_class")]; bind({{WHb[s], 0}, {TC[s], 1}}); [e setBytes:&ts length:sizeof ts atIndex:2]; t64(ts.tiles);
        }
        auto split = [&](const char* fn, id<MTLBuffer> src, id<MTLBuffer> L_, id<MTLBuffer> H_, size_t count) { uint32_t cnt = uint32_t(count);
          [e setComputePipelineState:pso(DL, fn)]; bind({{src, 0}, {L_, 1}, {H_, 2}}); [e setBytes:&cnt length:4 atIndex:3]; t64(count); };
        auto linear = [&](int s, id<MTLBuffer> XL, id<MTLBuffer> XH, id<MTLBuffer> S, id<MTLBuffer> out) { const StageDef32& d = defs[s];
          NrdLinear lp{rows32, d.out_width, d.in_stride, d.chunks, 2u, d.seeded};
          [e setComputePipelineState:pso(DL, "nrd_linear_exact_ws4")]; bind({{XL, 0}, {XH, 1}, {WL[s], 2}, {WHb[s], 3}, {TC[s], 4}, {S, 5}, {out, 6}}); [e setBytes:&lp length:sizeof lp atIndex:7]; tg32(d.out_width / 8, rows / 32); };
        auto seed = [&](uint32_t kind, id<MTLBuffer> partial, id<MTLBuffer> S) { NrdSeed sp{rows32, kind};
          [e setComputePipelineState:pso(DL, "nrd32_seed")]; bind({{xin, 0}, {partial, 1}, {S, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; [e setBytes:&sp length:sizeof sp atIndex:4]; t64(rows * W); };
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
        if (nrr->fused32) {   // the six gated fused pairs (EXACT-C32-FUSION-GATE-v1, MASK 63; bindings = the gate's fused_H/A/D/E/B/C): every producer/consumer pair one dispatch
          auto p32 = [&](const char* fn) { id<MTLComputePipelineState> p = pso(DL, fn); require(p.threadExecutionWidth == 32, "native range (C32): a fused 32-lane kernel needs 32-thread SIMD groups"); return p; };
          split("nrd_split_bytes", xin, XL0, XH0, rows * W);
          [e setComputePipelineState:p32("nrd32_expand_activate_split_exact_ws4")]; bind({{XL0, 0}, {XH0, 1}, {WL[0], 2}, {WHb[0], 3}, {TC[0], 4}, {XL1, 5}, {XH1, 6}}); shape(7); tg32(H / 8, rows / 32);   // H
          if (upsample) seed(2u, canonV, S2); else seed(0u, y2, S2);
          { NrdLinear lp{rows32, 32, 128, 4, 2u, 1};   // A: contract + quant split, its half output kept for the project seed
            [e setComputePipelineState:p32("nrd_linear_quant_split_keep_exact_ws4")]; bind({{XL1, 0}, {XH1, 1}, {WL[1], 2}, {WHb[1], 3}, {TC[1], 4}, {S2, 5}, {y2, 6}, {XL2, 7}, {XH2, 8}}); [e setBytes:&lp length:sizeof lp atIndex:9]; tg32(W / 8, rows / 32); }
          linear(2, XL2, XH2, Sz, y3);
          [e setComputePipelineState:pso(DL, "nrd32_normalize_split")]; bind({{y3, 0}, {wblock, 1}, {QL, 2}, {QH, 3}}); shape(4); t64(rows * 2);   // D
          [e setComputePipelineState:pso(DL, "nrd32_scores_exact_ws4")]; bind({{QL, 0}, {QH, 1}, {y5, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; shape(4); tg32(8, size_t(windows) * 2);
          [e setComputePipelineState:pso(DL, "nrd_probabilities_perm_split")]; bind({{y5, 0}, {PL, 1}, {PH, 2}}); shape(3); t64(rows);   // E
          split("nrd32_perm_values", y3, VL, VH, size_t(windows) * 2048);
          [e setComputePipelineState:p32("nrd32_values_quant_split_exact_ws4")]; bind({{PL, 0}, {PH, 1}, {VL, 2}, {VH, 3}, {XL7, 4}, {XH7, 5}}); shape(6); tg32(W / 8, size_t(windows) * 2);   // B
          seed(1u, y2, S8);
          { NrdLinear lp{rows32, 32, 32, 1, 2u, 1};   // C: project + nr_swin32_pack
            [e setComputePipelineState:p32("nrd_linear_pack_exact_ws4")]; bind({{XL7, 0}, {XH7, 1}, {WL[3], 2}, {WHb[3], 3}, {TC[3], 4}, {S8, 5}, {y9, 6}}); [e setBytes:&lp length:sizeof lp atIndex:7]; tg32(W / 8, rows / 32); }
        } else {
#endif
        split("nrd_split_bytes", xin, XL0, XH0, rows * W);            linear(0, XL0, XH0, Sz, y0);
        [e setComputePipelineState:pso(EL, "nr_swin32_activate")]; bind({{y0, 0}, {y1, 1}, {cv1, 2}}); shape(3); t64(rows * H);
        split("nrd_split_bytes", y1, XL1, XH1, rows * H);
        if (upsample) seed(2u, canonV, S2); else seed(0u, y2, S2);
        linear(1, XL1, XH1, S2, y2);
        split("nrd_split_quant_perm", y2, XL2, XH2, rows * W);         linear(2, XL2, XH2, Sz, y3);
        [e setComputePipelineState:pso(EL, "nr_swin32_normalize")]; bind({{y3, 0}, {y4, 2}, {cv4, 3}}); [e setBuffer:wblock offset:0 atIndex:1]; shape(4); t64(rows * 2);
        split("nrd_split_bytes", y4, QL, QH, rows * 64);
        [e setComputePipelineState:pso(DL, "nrd32_scores_exact_ws4")]; bind({{QL, 0}, {QH, 1}, {y5, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; shape(4); tg32(8, size_t(windows) * 2);
        [e setComputePipelineState:pso(EL, "nr_swin32_probabilities")]; bind({{y5, 0}, {y6, 1}, {cv6, 2}}); shape(3); t64(rows);
        split("nrd_perm_probs", y6, PL, PH, rows * 64);
        split("nrd32_perm_values", y3, VL, VH, size_t(windows) * 2048);
        [e setComputePipelineState:pso(DL, "nrd32_values_exact_ws4")]; bind({{PL, 0}, {PH, 1}, {VL, 2}, {VH, 3}, {y7, 4}}); shape(5); tg32(4, size_t(windows) * 2);
        split("nrd_split_quant_perm", y7, XL7, XH7, rows * W);
        seed(1u, y2, S8);
        linear(3, XL7, XH7, S8, y8);
        [e setComputePipelineState:pso(EL, "nr_swin32_pack")]; bind({{y8, 0}, {y9, 1}, {cv9, 2}}); shape(3); t64(rows * W);
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
        }
#endif
        { WideWindowMap back{IW, IH, gw, gh, CGW, CGH, ox, oy, 0, 0}; [e setComputePipelineState:pso(EL, "nr_swin32_rewindow")]; bind({{y9, 0}, {canon, 1}, {rwc_out, 2}}); [e setBytes:&back length:sizeof back atIndex:3]; t64(canon_rows * W); }
        { EncoderExportShape xs{IW, IH, W, outview ? 1u : 0u}; [e setComputePipelineState:pso(EL, "nr_encoder_native_export")]; [e setBuffer:canon offset:0 atIndex:0]; [e setBuffer:pool offset:out_off atIndex:1]; [e setBuffer:expc offset:0 atIndex:2]; [e setBytes:&xs length:sizeof xs atIndex:3]; t64(size_t(IW) * IH * W); }
        nr_exp::nr_scope_end(e);
        if (!outview) { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:pool range:NSMakeRange(sync_off, size_t(gw) * gh * 4) value:0]; nr_exp::nr_scope_end(bl); }
        if (n == nrr->last) nrr->resolution = "native";
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
        LaunchRec lr{long(n), "native:" + nrr->kind + (nrr->fused32 ? "+fused32" : ""), {}, {}, ""};
#else
        LaunchRec lr{long(n), "native:" + nrr->kind, {}, {}, ""};
#endif
        launches.push_back(lr); decl.push_back(dl);
        continue;
      }
      const uint64_t NR_PTX_POOL_BASE = 0x100000000ull;   // the PTX runtime's fake pool base (nr_ptx_rt.metal NR_POOL_BASE)
      // Tier from the declaration kind (build/nr-accept-v1/experiments/NR-RANGE-133-146-v1.json): wide256 = the 8h C256 tier (120x72 tokens, 15x9
      // canonical windows), wide128 = the 4h C128 tier (240x144, 30x18). Every constant below is derived from W; for W = 256 they equal the values
      // the 133-140 range was verified with (tensor 2211840; engine record 689232 with RES 360464 / QKV 360992 / ARES 688704; upsample record
      // tiles 360448, RES 491520, skip scales 492032, part 2 at 492544).
      // wide64 = the 2h C64 tier (480x288, 60x36; experiments/NR-TIER-2H-v1.json).
      const bool c128 = nrr->kind == "wide128-chained-exact-dense-v2", c64 = nrr->kind == "wide64-chained-exact-dense-v2";
      const uint32_t W = c64 ? 64 : c128 ? 128 : 256, H = 4 * W, HEADS = W / 32, IW = 30720 / W, IH = 18432 / W, CGW = IW / 8, CGH = IH / 8;   // 120x72 / 240x144 / 480x288
      const size_t TENSOR = size_t(IW / 4) * (IH / 4) * 16 * W;
      const std::string tier = c64 ? "ptx_swin_2h_64_2_" : c128 ? "ptx_swin_4h_128_4_" : "ptx_swin_8h_256_8_";
      const size_t MIX = 4 * size_t(W) * W + 128 * size_t(W), RES = MIX + size_t(W) * W + 16, QKV = RES + 2 * W + 16;
      const size_t ARES = ((QKV + 3 * size_t(W) * W + 8192 * size_t(HEADS) + 4 * HEADS + 15) / 16) * 16 + size_t(W) * W, CEND = ARES + 2 * W + 16;
      const size_t UPT = MIX + size_t(W) * W, URES = UPT + 2 * size_t(W) * W, USKS = URES + 2 * W, UP2 = USKS + 2 * W;
      (void)USKS;
      require(dl.params_size >= 72, "native range: launch params shorter than 72 bytes");
      // outview_wait is the same C256 block; it writes its output tensor in the PLANAR export layout and no control words
      // (recorded RTX deltas; build/nr-accept-v1/experiments/S140-OUTVIEW-AS-CHAINED-v1.RESULT.json follow-up layout scan).
      const bool outview = dl.kernel == tier + "outview_wait";
      // upsample_tilesync (launch 133): the recovered pre-block stage feeds the same block (build/nr-accept-v1/experiments/S133-UPSAMPLE-RECOVERY-v2*.json):
      // exact 16-chunk upsample QMMA of the coarse split tensor, nearest 2x, exact skip*scale + U, satfinite e4m3; the block reads the launch's
      // weight record through a byte-copied chained layout. Only a kind-v2 declaration whose range starts at the launch admits it.
      const bool upsample = dl.kernel == tier + "upsample_tilesync";
      require(dl.kernel == tier + "chained" || outview || upsample, "native range: launch kernel is not the declared tier's chained, outview_wait or upsample_tilesync kernel");
      if (upsample) require((nrr->kind == "wide256-chained-exact-dense-v2" || c128 || c64) && n == nrr->first, "native range: an upsample_tilesync launch needs a kind-v2 range that starts at it");
      const size_t SPLIT = size_t(IW / 2) * (IH / 2) * 2 * W;   // the coarse planar-in-16s input an upsample launch reads at +0 (8h: 60x36x512, 4h: launch 140's 120x72x256)
      const uint64_t a0 = u64(0), a8 = u64(8), a24 = upsample ? u64(24) : 0, a64 = u64(64);
      require(a0 >= NR_PTX_POOL_BASE && a0 - NR_PTX_POOL_BASE + (upsample ? SPLIT : TENSOR) <= pool_bytes && a8 >= NR_PTX_POOL_BASE && a8 - NR_PTX_POOL_BASE + TENSOR <= pool_bytes, "native range: tensor pointers outside the pool");
      if (upsample) require(a24 >= NR_PTX_POOL_BASE && a24 - NR_PTX_POOL_BASE + TENSOR <= pool_bytes, "native range: skip tensor pointer outside the pool");
      require(i32(32) == int32_t(IH) && i32(36) == int32_t(IW), "native range: token dimensions disagree with the declared tier");
      const int32_t ox = i32(40), oy = i32(44);
      require((ox == 0 || ox == -4) && (oy == 0 || oy == -4), "native range: window origin");
      // the pre-block stage writes canonical rows (origin 0,0) and the rewindow maps them to the launch grid, so an upsample launch admits the same origins
      // (133 has 0,0; 141's own origin is the one its standalone check used: build/nr-native-tier-v1/s133/s141/run-C1)
      const uint32_t gw = uint32_t((int32_t(IW) - ox + 7) / 8), gh = uint32_t((int32_t(IH) - oy + 7) / 8);
      require([g[0] unsignedIntValue] == gw && [g[1] unsignedIntValue] == gh && [g[2] unsignedIntValue] == 1u, "native range: descriptor grid disagrees with the params origin");
      const size_t in_off = size_t(a0 - NR_PTX_POOL_BASE), out_off = size_t(a8 - NR_PTX_POOL_BASE);
      if (outview) require(a64 == 0, "native range: an outview_wait launch must not carry control words");
      else require(a64 >= NR_PTX_POOL_BASE && size_t(a64 - NR_PTX_POOL_BASE) >= c0 && size_t(a64 - NR_PTX_POOL_BASE) + size_t(gw) * gh * 4 <= c1, "native range: sync words outside the control region");
      const size_t sync_off = outview ? 0 : size_t(a64 - NR_PTX_POOL_BASE);
      require(dl.weights_size >= CEND, "native range: weight record shorter than the engine layout");
      if (upsample) require(dl.weights_size >= UP2 + (ARES + 2 * W - QKV), "native range: upsample_tilesync weight record shorter than its recovered layout");
      static thread_local unsigned nr_prev_n = 0; static thread_local size_t nr_prev_out = 0;
      if (n == nrr->first) nrr->resolution.clear();
      else require(nr_prev_n + 1 == n && nr_prev_out == in_off, "native range: launches are not chained");
      nr_prev_n = n; nr_prev_out = out_off;
      const size_t rows = size_t(gw) * gh * 64, max_rows = size_t(CGW + 1) * (CGH + 1) * 64, canon_rows = size_t(CGW) * CGH * 64;
      const uint32_t windows = uint32_t(rows / 64), rows32 = uint32_t(rows);
      auto rb = [&](const std::string& key, size_t nbytes) { return resident_buf(ctx, dev, "native-range:w" + std::to_string(W) + ":" + key, nbytes + 16); };
      id<MTLBuffer> canon = rb("canon", canon_rows * W), xin = rb("xin", max_rows * W), rwc_in = rb("rwc-in", max_rows * W * 4), rwc_out = rb("rwc-out", canon_rows * W * 4), expc = rb("export-counts", TENSOR * 4);
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
      // DECODER-EXACT-MLP-FUSION-v1 (experiments/DECODER-EXACT-MLP-FUSION-v1.json): a fused range never writes the expanded half, the activation bytes
      // or activate's coverage, so those scratch buffers are not allocated
      const bool fused_head = nrr->fused;
      id<MTLBuffer> y0 = fused_head ? nil : rb("y0", max_rows * H * 2), y1 = fused_head ? nil : rb("y1", max_rows * H), c1b = fused_head ? nil : rb("c1", max_rows * H * 4);
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
      // DECODER-EXACT-MLP-FUSION-v2 (experiments/DECODER-EXACT-MLP-FUSION-v2.json): a fused2 range also never writes the mid, values and project halves,
      // the normalize and probabilities bytes, or the coverage of normalize, probabilities and pack, so those buffers are not allocated
      const bool fused2 = nrr->fused2;
      id<MTLBuffer> y2 = fused2 ? nil : rb("y2", max_rows * W * 2), y3 = rb("y3", max_rows * W * 2), y4 = rb("y4", max_rows * W * 3 * 2);
#else
      id<MTLBuffer> y2 = rb("y2", max_rows * W * 2), y3 = rb("y3", max_rows * W * 2), y4 = rb("y4", max_rows * W * 3 * 2);
#endif
#else
      id<MTLBuffer> y0 = rb("y0", max_rows * H * 2), y1 = rb("y1", max_rows * H), c1b = rb("c1", max_rows * H * 4), y2 = rb("y2", max_rows * W * 2), y3 = rb("y3", max_rows * W * 2), y4 = rb("y4", max_rows * W * 3 * 2);
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
      id<MTLBuffer> y5 = fused2 ? nil : rb("y5", max_rows * W * 2), c5b = fused2 ? nil : rb("c5", max_rows * W * 2 * 4), y6 = rb("y6", max_rows * HEADS * 64 * 2), y7 = fused2 ? nil : rb("y7", max_rows * HEADS * 64), c7b = fused2 ? nil : rb("c7", max_rows * HEADS * 64 * 4);
      id<MTLBuffer> y8 = fused2 ? nil : rb("y8", max_rows * W * 2), y9 = fused2 ? nil : rb("y9", max_rows * W * 2), y10 = rb("y10", max_rows * W), c10b = fused2 ? nil : rb("c10", max_rows * W * 4);
#else
      id<MTLBuffer> y5 = rb("y5", max_rows * W * 2), c5b = rb("c5", max_rows * W * 2 * 4), y6 = rb("y6", max_rows * HEADS * 64 * 2), y7 = rb("y7", max_rows * HEADS * 64), c7b = rb("c7", max_rows * HEADS * 64 * 4);
      id<MTLBuffer> y8 = rb("y8", max_rows * W * 2), y9 = rb("y9", max_rows * W * 2), y10 = rb("y10", max_rows * W), c10b = rb("c10", max_rows * W * 4);
#endif
      id<MTLBuffer> XL0 = rb("XL0", max_rows * W * 2), XH0 = rb("XH0", max_rows * W * 2), XL1 = rb("XL1", max_rows * H * 2), XH1 = rb("XH1", max_rows * H * 2);
      id<MTLBuffer> XL2 = rb("XL2", max_rows * W * 2), XH2 = rb("XH2", max_rows * W * 2), XL3 = rb("XL3", max_rows * W * 2), XH3 = rb("XH3", max_rows * W * 2), XL8 = rb("XL8", max_rows * W * 2), XH8 = rb("XH8", max_rows * W * 2);
      id<MTLBuffer> S3 = rb("S3", max_rows * W * 2), S9 = rb("S9", max_rows * W * 2), QL = rb("QL", max_rows * HEADS * 64 * 2), QH = rb("QH", max_rows * HEADS * 64 * 2), PL = rb("PL", max_rows * HEADS * 64 * 2), PH = rb("PH", max_rows * HEADS * 64 * 2);
      id<MTLBuffer> VL = rb("VL", (max_rows / 64) * HEADS * 2048 * 2), VH = rb("VH", (max_rows / 64) * HEADS * 2048 * 2);
      struct StageDef { uint32_t kind, out_width, in_stride, chunks, seeded; size_t count; };
      const StageDef defs[5] = { {0, H, W, HEADS, 0, size_t(HEADS) * H * 32}, {1, W, H, 4, 0, size_t(HEADS) * 4 * 32 * 32}, {2, W, W, HEADS, 1, size_t(HEADS) * W * 32},
                                 {3, 3 * W, W, HEADS, 0, size_t(HEADS) * 3 * W * 32}, {4, W, W, HEADS, 1, size_t(HEADS) * W * 32} };
      std::array<id<MTLBuffer>, 5> WL, WHb, TC;
      for (int s = 0; s < 5; ++s) { const std::string k = std::to_string(n) + ":" + std::to_string(s); WL[s] = rb("WL:" + k, defs[s].count * 2); WHb[s] = rb("WH:" + k, defs[s].count * 2); TC[s] = rb("TC:" + k, (defs[s].count / 32 / 8) * 4); }
      id<MTLBuffer> wblock = dl.bw;   // the block's weight record in the chained layout the engine offsets address
      if (upsample) {   // byte copies only: blk[0:UPT] = w[0:UPT], RES blk[RES:RES+2W] = w[URES:URES+2W], blk[QKV:ARES+2W] = w[UP2:UP2+(ARES+2W-QKV)]
        wblock = rb("blk:" + std::to_string(n), CEND);
        id<MTLBlitCommandEncoder> bl = blit();
        [bl copyFromBuffer:dl.bw sourceOffset:0 toBuffer:wblock destinationOffset:0 size:UPT];
        [bl copyFromBuffer:dl.bw sourceOffset:URES toBuffer:wblock destinationOffset:RES size:2 * W];
        [bl copyFromBuffer:dl.bw sourceOffset:UP2 toBuffer:wblock destinationOffset:QKV size:ARES + 2 * W - QKV];
        nr_exp::nr_scope_end(bl);
      }
      id<MTLComputeCommandEncoder> e = compute();
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
      if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " native:" + nrr->kind + (fused2 ? "+fused-mlp-head+fused-block2" : fused_head ? "+fused-mlp-head" : "");
#else
      if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " native:" + nrr->kind + (fused_head ? "+fused-mlp-head" : "");
#endif
#else
      if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " native:" + nrr->kind;
#endif
#endif
      auto t64 = [&](size_t count) { [e dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; };
      auto tg32 = [&](size_t gx, size_t gy) { [e dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; };
      auto bind = [&](std::initializer_list<std::pair<id<MTLBuffer>, NSUInteger>> bs) { for (const auto& bi : bs) [e setBuffer:bi.first offset:0 atIndex:bi.second]; };
      id<MTLLibrary> DL = nrr->library, EL = nrr->engine;
      if (n == nrr->first && !upsample) {
        NrdImportShape is{IW, IH, W, CGW, CGH};
        [e setComputePipelineState:pso(DL, "nrd_import_tensor")]; [e setBuffer:pool offset:in_off atIndex:0]; [e setBuffer:canon offset:0 atIndex:1]; [e setBytes:&is length:sizeof is atIndex:2]; t64(canon_rows * W);
      }
      if (upsample) {   // pre-block stage -> canon (standalone evidence: 133 build/nr-native-tier-v1/s133/run-v2-D1, 141 s133/s141/run-D1; 0 bytes differ)
        struct NrdUpImport { uint32_t rows, width, height, channels; }; struct NrdUpCombine { uint32_t grid_width, grid_height, coarse_width; };
        const size_t up_rows = ((size_t(IW / 2) * (IH / 2) + 31) / 32) * 32, up_count = size_t(2 * W / 32) * W * 32;
        id<MTLBuffer> uskip = rb("up-skip", canon_rows * W), uXL = rb("up-XL", up_rows * 2 * W * 2), uXH = rb("up-XH", up_rows * 2 * W * 2), uWL = rb("up-WL", up_count * 2), uWH = rb("up-WH", up_count * 2);
        id<MTLBuffer> uTC = rb("up-TC", (up_count / 32 / 8) * 4), uS = rb("up-S", up_rows * W * 2), uU = rb("up-U", up_rows * W * 2);
        NrdImportShape is{IW, IH, W, CGW, CGH};
        [e setComputePipelineState:pso(DL, "nrd_import_tensor")]; [e setBuffer:pool offset:size_t(a24 - NR_PTX_POOL_BASE) atIndex:0]; [e setBuffer:uskip offset:0 atIndex:1]; [e setBytes:&is length:sizeof is atIndex:2]; t64(canon_rows * W);
        NrdUpImport ui{uint32_t(up_rows), IW / 2, IH / 2, 2 * W};
        [e setComputePipelineState:pso(DL, "nrd_up_import")]; [e setBuffer:pool offset:in_off atIndex:0]; bind({{uXL, 1}, {uXH, 2}}); [e setBytes:&ui length:sizeof ui atIndex:3]; t64(up_rows * 2 * W);
        NrdPrepack upp{5, uint32_t(up_count)};
        [e setComputePipelineState:pso(DL, "nrd_prepack_exact")]; [e setBuffer:dl.bw offset:0 atIndex:0]; bind({{uWL, 1}, {uWH, 2}}); [e setBytes:&upp length:sizeof upp atIndex:3]; t64(up_count);
        NrdTileShape uts{uint32_t((up_count / 32 / 8) * 4)};
        [e setComputePipelineState:pso(DL, "nrd_tile_class")]; bind({{uWH, 0}, {uTC, 1}}); [e setBytes:&uts length:sizeof uts atIndex:2]; t64(uts.tiles);
        NrdLinear ul{uint32_t(up_rows), W, 2 * W, 2 * W / 32, 5, 0};
        [e setComputePipelineState:pso(DL, "nrd_linear_exact_ws4")]; bind({{uXL, 0}, {uXH, 1}, {uWL, 2}, {uWH, 3}, {uTC, 4}, {uS, 5}, {uU, 6}}); [e setBytes:&ul length:sizeof ul atIndex:7]; tg32(W / 8, up_rows / 32);
        NrdUpCombine uc{CGW, CGH, IW / 2};
        [e setComputePipelineState:pso(DL, "nrd_up_combine")]; bind({{uskip, 0}, {uU, 1}, {canon, 3}}); [e setBuffer:dl.bw offset:0 atIndex:2]; [e setBytes:&uc length:sizeof uc atIndex:4]; t64(canon_rows * W);
      }
      { WideWindowMap to{IW, IH, CGW, CGH, gw, gh, 0, 0, ox, oy}; [e setComputePipelineState:pso(EL, "nr_wide_rewindow")]; bind({{canon, 0}, {xin, 1}, {rwc_in, 2}}); [e setBytes:&to length:sizeof to atIndex:3]; t64(rows * W); }
      for (int s = 0; s < 5; ++s) {   // split the launch's own verified weight buffer once per frame; tile classes on the GPU
        NrdPrepack pp{defs[s].kind, uint32_t(defs[s].count)};
        [e setComputePipelineState:pso(DL, "nrd_prepack_exact")]; [e setBuffer:wblock offset:0 atIndex:0]; bind({{WL[s], 1}, {WHb[s], 2}}); [e setBytes:&pp length:sizeof pp atIndex:3]; t64(defs[s].count);
        NrdTileShape ts{uint32_t((defs[s].count / 32 / 8) * 4)};
        [e setComputePipelineState:pso(DL, "nrd_tile_class")]; bind({{WHb[s], 0}, {TC[s], 1}}); [e setBytes:&ts length:sizeof ts atIndex:2]; t64(ts.tiles);
      }
      auto split = [&](const char* fn, id<MTLBuffer> src, id<MTLBuffer> L_, id<MTLBuffer> H_, size_t count) { uint32_t c32 = uint32_t(count);
        [e setComputePipelineState:pso(DL, fn)]; bind({{src, 0}, {L_, 1}, {H_, 2}}); [e setBytes:&c32 length:4 atIndex:3]; t64(count); };
      auto linear = [&](int s, id<MTLBuffer> XL, id<MTLBuffer> XH, id<MTLBuffer> S, id<MTLBuffer> out) { const StageDef& d = defs[s];
        if (d.seeded) { NrdSeed sp{rows32, d.kind == 2 ? 0u : 1u}; [e setComputePipelineState:pso(DL, "nrd_seed")]; bind({{xin, 0}, {y3, 1}, {S, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; [e setBytes:&sp length:sizeof sp atIndex:4]; t64(rows * W); }
        NrdLinear lp{rows32, d.out_width, d.in_stride, d.chunks, d.kind, d.seeded};
        [e setComputePipelineState:pso(DL, "nrd_linear_exact_ws4")]; bind({{XL, 0}, {XH, 1}, {WL[s], 2}, {WHb[s], 3}, {TC[s], 4}, {S, 5}, {out, 6}}); [e setBytes:&lp length:sizeof lp atIndex:7]; tg32(d.out_width / 8, rows / 32); };
      auto engine_rows = [&](const char* fn, std::initializer_list<std::pair<id<MTLBuffer>, NSUInteger>> bs, NSUInteger shape_index, uint32_t shape, size_t threads) {
        [e setComputePipelineState:pso(EL, fn)]; bind(bs); [e setBytes:&shape length:4 atIndex:shape_index]; t64(threads); };
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
      if (fused2) {   // the fused head (v1) and the second fusion set: every producer/consumer pair runs as one dispatch (experiments/EXACT-BLOCK-FUSION-GATE-v1.RESULT.json)
        auto p32 = [&](const char* fn) { id<MTLComputePipelineState> p = pso(DL, fn); require(p.threadExecutionWidth == 32, "native range: a fused 32-lane kernel needs 32-thread SIMD groups"); return p; };
        split("nrd_split_bytes", xin, XL0, XH0, rows * W);
        [e setComputePipelineState:p32("nrd_expand_activate_split_exact_ws4")]; bind({{XL0, 0}, {XH0, 1}, {WL[0], 2}, {WHb[0], 3}, {TC[0], 4}, {XL1, 5}, {XH1, 6}}); [e setBytes:&rows32 length:4 atIndex:7]; tg32(H / 8, rows / 32);
        { const StageDef& d = defs[1]; NrdLinear lp{rows32, d.out_width, d.in_stride, d.chunks, d.kind, d.seeded};   // (A) mid + nrd_split_quant_perm
          [e setComputePipelineState:p32("nrd_linear_quant_split_exact_ws4")]; bind({{XL1, 0}, {XH1, 1}, {WL[1], 2}, {WHb[1], 3}, {TC[1], 4}, {S3, 5}, {XL2, 6}, {XH2, 7}}); [e setBytes:&lp length:sizeof lp atIndex:8]; tg32(d.out_width / 8, rows / 32); }
        linear(2, XL2, XH2, S3, y3);
        split("nrd_split_quant_perm", y3, XL3, XH3, rows * W);        linear(3, XL3, XH3, S3, y4);
        [e setComputePipelineState:pso(DL, "nrd_normalize_split")]; bind({{y4, 0}, {QL, 2}, {QH, 3}}); [e setBuffer:wblock offset:0 atIndex:1]; [e setBytes:&rows32 length:4 atIndex:4]; t64(rows * HEADS * 2);   // (D)
        [e setComputePipelineState:pso(DL, "nrd_scores_exact_ws4")]; bind({{QL, 0}, {QH, 1}, {y6, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; [e setBytes:&rows32 length:4 atIndex:4]; tg32(HEADS * 8, size_t(windows) * 2);
        { const uint32_t prow = rows32 * HEADS; [e setComputePipelineState:pso(DL, "nrd_probabilities_perm_split")]; bind({{y6, 0}, {PL, 1}, {PH, 2}}); [e setBytes:&prow length:4 atIndex:3]; t64(rows * HEADS); }   // (E)
        split("nrd_perm_values", y4, VL, VH, size_t(windows) * HEADS * 2048);
        [e setComputePipelineState:p32("nrd_values_quant_split_exact_ws4")]; bind({{PL, 0}, {PH, 1}, {VL, 2}, {VH, 3}, {XL8, 4}, {XH8, 5}}); [e setBytes:&rows32 length:4 atIndex:6]; tg32(W / 8, size_t(windows) * 2);   // (B)
        { NrdSeed sp{rows32, 1u}; [e setComputePipelineState:pso(DL, "nrd_seed")]; bind({{xin, 0}, {y3, 1}, {S9, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; [e setBytes:&sp length:sizeof sp atIndex:4]; t64(rows * W); }
        { const StageDef& d = defs[4]; NrdLinear lp{rows32, d.out_width, d.in_stride, d.chunks, d.kind, d.seeded};   // (C) project + nr_wide_pack
          [e setComputePipelineState:p32("nrd_linear_pack_exact_ws4")]; bind({{XL8, 0}, {XH8, 1}, {WL[4], 2}, {WHb[4], 3}, {TC[4], 4}, {S9, 5}, {y10, 6}}); [e setBytes:&lp length:sizeof lp atIndex:7]; tg32(d.out_width / 8, rows / 32); }
      } else {
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
      if (fused_head) {   // ONE dispatch: exact expand (stage 0 accumulators and RNE), activate and split, writing mid's operands (experiments/EXACT-MLP-FUSION-GATE-v1.RESULT.json)
        split("nrd_split_bytes", xin, XL0, XH0, rows * W);
        id<MTLComputePipelineState> fz = pso(DL, "nrd_expand_activate_split_exact_ws4");
        require(fz.threadExecutionWidth == 32, "native range: the fused MLP head's lane mapping needs 32-thread SIMD groups");
        [e setComputePipelineState:fz]; bind({{XL0, 0}, {XH0, 1}, {WL[0], 2}, {WHb[0], 3}, {TC[0], 4}, {XL1, 5}, {XH1, 6}}); [e setBytes:&rows32 length:4 atIndex:7]; tg32(H / 8, rows / 32);
        linear(1, XL1, XH1, S3, y2);
      } else {
#endif
      split("nrd_split_bytes", xin, XL0, XH0, rows * W);            linear(0, XL0, XH0, S3, y0);
      engine_rows("nr_wide_activate", {{y0, 0}, {y1, 1}, {c1b, 2}}, 3, rows32, rows * H);
      split("nrd_split_bytes", y1, XL1, XH1, rows * H);             linear(1, XL1, XH1, S3, y2);
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
      }
#endif
      split("nrd_split_quant_perm", y2, XL2, XH2, rows * W);        linear(2, XL2, XH2, S3, y3);
      split("nrd_split_quant_perm", y3, XL3, XH3, rows * W);        linear(3, XL3, XH3, S3, y4);
      [e setComputePipelineState:pso(EL, "nr_wide_normalize")]; bind({{y4, 0}, {y5, 2}, {c5b, 3}}); [e setBuffer:wblock offset:0 atIndex:1]; [e setBytes:&rows32 length:4 atIndex:4]; t64(rows * HEADS * 2);
      split("nrd_split_bytes", y5, QL, QH, rows * HEADS * 64);
      [e setComputePipelineState:pso(DL, "nrd_scores_exact_ws4")]; bind({{QL, 0}, {QH, 1}, {y6, 3}}); [e setBuffer:wblock offset:0 atIndex:2]; [e setBytes:&rows32 length:4 atIndex:4]; tg32(HEADS * 8, size_t(windows) * 2);
      engine_rows("nr_swin32_probabilities", {{y6, 0}, {y7, 1}, {c7b, 2}}, 3, rows32 * HEADS, rows * HEADS);
      split("nrd_perm_probs", y7, PL, PH, rows * HEADS * 64);
      split("nrd_perm_values", y4, VL, VH, size_t(windows) * HEADS * 2048);
      [e setComputePipelineState:pso(DL, "nrd_values_exact_ws4")]; bind({{PL, 0}, {PH, 1}, {VL, 2}, {VH, 3}, {y8, 4}}); [e setBytes:&rows32 length:4 atIndex:5]; tg32(W / 8, size_t(windows) * 2);
      split("nrd_split_quant_perm", y8, XL8, XH8, rows * W);        linear(4, XL8, XH8, S9, y9);
      engine_rows("nr_wide_pack", {{y9, 0}, {y10, 1}, {c10b, 2}}, 3, rows32, rows * W);
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
      }
#endif
      { WideWindowMap back{IW, IH, gw, gh, CGW, CGH, ox, oy, 0, 0}; [e setComputePipelineState:pso(EL, "nr_wide_rewindow")]; bind({{y10, 0}, {canon, 1}, {rwc_out, 2}}); [e setBytes:&back length:sizeof back atIndex:3]; t64(canon_rows * W); }
      { EncoderExportShape xs{IW, IH, W, outview ? 1u : 0u};[e setComputePipelineState:pso(EL, "nr_encoder_native_export")]; [e setBuffer:canon offset:0 atIndex:0]; [e setBuffer:pool offset:out_off atIndex:1]; [e setBuffer:expc offset:0 atIndex:2]; [e setBytes:&xs length:sizeof xs atIndex:3]; t64(size_t(IW) * IH * W); }
      nr_exp::nr_scope_end(e);
      if (!outview) { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:pool range:NSMakeRange(sync_off, size_t(gw) * gh * 4) value:0]; nr_exp::nr_scope_end(bl); }
      if (n == nrr->last) nrr->resolution = "native";
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
      LaunchRec lr{long(n), "native:" + nrr->kind + (fused2 ? "+fused-mlp-head+fused-block2" : fused_head ? "+fused-mlp-head" : ""), {}, {}, ""};
#else
      LaunchRec lr{long(n), "native:" + nrr->kind + (fused_head ? "+fused-mlp-head" : ""), {}, {}, ""};
#endif
#else
      LaunchRec lr{long(n), "native:" + nrr->kind, {}, {}, ""};
#endif
      launches.push_back(lr); decl.push_back(dl);
      continue;
    }
#endif
    { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:bcov range:NSMakeRange(0, pool_bytes + 4096) value:0]; if (!dl.texmode.empty()) [bl fillBuffer:btexcov range:NSMakeRange(0, size_t(tw) * th * 4 + 16) value:0]; nr_exp::nr_scope_end(bl); }
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
    id<MTLComputeCommandEncoder> enc = compute();
    if (!nr_exp::gpu_profiler().labels.empty()) nr_exp::gpu_profiler().labels.back() = "launch " + std::to_string(n) + " " + dl.kernel;
    [enc setComputePipelineState:dps];
#else
    id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:dps];
#endif
    [enc setBuffer:bpool offset:0 atIndex:0]; [enc setBuffer:dl.bw offset:0 atIndex:1]; [enc setBuffer:dl.bprm offset:0 atIndex:2]; [enc setBuffer:brsq offset:0 atIndex:3]; [enc setBuffer:brcp offset:0 atIndex:4]; [enc setBuffer:bdbg offset:0 atIndex:5]; [enc setBuffer:bcov offset:0 atIndex:6];
    [enc setBuffer:brom offset:0 atIndex:7]; [enc setBuffer:bex2u offset:0 atIndex:8]; [enc setBuffer:bex2f offset:0 atIndex:9]; [enc setBuffer:btexcov offset:0 atIndex:10];
    if (dl.temporal_launch) { [enc setBuffer:bex2h offset:0 atIndex:11]; [enc setBuffer:dl.bfault offset:0 atIndex:12]; }
    if (dl.texmode == "post") { [enc setTexture:tex[0] atIndex:0]; [enc setTexture:in.temporal ? tex[2] : tex[1] atIndex:1]; if (dl.temporal_launch) { [enc setTexture:in.motion atIndex:2]; [enc setTexture:in.temporal->history atIndex:3]; } }
    else if (dl.texmode == "copy") { if (in.temporal) { [enc setTexture:tex[2] atIndex:0]; [enc setTexture:in.temporal->history atIndex:1]; } else { [enc setTexture:tex[1] atIndex:0]; [enc setTexture:tex[2] atIndex:1]; } }
    [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntValue], [g[1] unsignedIntValue], [g[2] unsignedIntValue]) threadsPerThreadgroup:MTLSizeMake([b[0] unsignedIntValue], [b[1] unsignedIntValue], [b[2] unsignedIntValue])]; nr_exp::nr_scope_end(enc);
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "decoder-pool-multi" && n == 134u && !final_only) poke(bcov, 400, 2);   // pool word 100 written twice (pixels unaffected)
#endif
    LaunchRec lr{long(n), "ptx:" + dl.kernel, {}, {}, ""};
    if (!final_only) { dl.pool_validator_slot = validate("launch " + std::to_string(n) + " pool words at most once", n, bcov, pool_bytes / 4, 2u, nil); lr.validators.push_back(dl.pool_validator_slot); dl.pool_digest_slot = digest("launch " + std::to_string(n) + " pool", n, pool, 0, pool_bytes, nullptr, true, true); lr.digests.push_back({"pool", dl.pool_digest_slot}); }
    if (dl.texmode == "post") { id<MTLTexture> to = in.temporal ? tex[2] : tex[1]; tex_to_buf(to, tex155buf, 8); tex155_slot = digest("launch 155 output texture", 155u, tex155buf, 0, texb, nullptr, true, false); lr.digests.push_back({"texture_out", tex155_slot});
      if (dl.temporal_launch) { tex_to_buf(in.temporal->history, histpost, 8); histpost_slot = digest("history after 155", 155u, histpost, 0, texb, nullptr, true, false); tex_to_buf(in.motion, motpost, 4); motpost_slot = digest("motion after 155", 155u, motpost, 0, motb, nullptr, true, false); lr.digests.push_back({"history_after", histpost_slot}); lr.digests.push_back({"motion_after", motpost_slot}); } }
    if (dl.texmode == "copy") { dl.tex_validator_slot = validate("launch 156 final surface pixels exactly once", 156u, btexcov, size_t(tw) * th, 0u, nil); lr.validators.push_back(dl.tex_validator_slot); }
    launches.push_back(lr); decl.push_back(dl);
  }
#ifdef NR_EXPERIMENTAL_NATIVE_RANGE
  // CANDIDATE ONLY: every declared native range must have been encoded natively, first to last, in THIS frame. A declaration
  // that never matched its launches is the silent no-op a candidate cannot tolerate - raise before commit.
  if (ctx.override.active)
    for (const auto& r : ctx.override.native_ranges)
      if (r.resolution != "native"
#ifdef NR_EXPERIMENTAL_NATIVE_POST
          && !(r.post && !in.reset && r.resolution == "temporal-ptx")   // the post kind keeps the generic lib_temporal dispatch on temporal frames by design
#endif
          )
        throw std::runtime_error("native range " + r.kind + " " + std::to_string(r.first) + "-" + std::to_string(r.last) + " was declared but not encoded natively in this frame");
#endif
#ifdef NR_EXPERIMENTAL_BODY_OVERRIDE
  // Fail-closed body override admission: every (library, kernel) the manifest DECLARES must have been
  // resolved to that override library during this encode. A declaration that never matched a launch is
  // the silent no-op this lane cannot tolerate - the run would look like a candidate while executing the
  // pinned kernels. Raise before commit; do not fall back.
  if (ctx.override.active) {
    for (const auto& kv : ctx.override.functions) {
      for (const std::string& fn : kv.second) {
        const std::string key = kv.first + "|" + fn;
        auto it = ctx.override.resolved.find(key);
        if (it == ctx.override.resolved.end() || it->second.compare(0, 9, "override:") != 0)
          throw std::runtime_error("body override declared " + key + " but it resolved to '" +
                                   (it == ctx.override.resolved.end() ? std::string("<never encoded>") : it->second) +
                                   "'; the override did not take effect");
      }
    }
  }
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_POST
  if (diag && np_y16) { np_texend1 = resident_buf(ctx, dev, "native-range:post:diag-texend1", texb + 16); tex_to_buf(tex[1], np_texend1, 8); np_texend2 = resident_buf(ctx, dev, "native-range:post:diag-texend2", texb + 16); tex_to_buf(tex[2], np_texend2, 8); }
#endif
  launches.push_back({156, "pool_final", {{"pool_final", digest("final pool", 156u, pool, 0, pool_bytes, nullptr, true, false)}}, {}, ""}); const uint32_t final_pool_slot = uint32_t(dexp.size() - 1);
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  nr_exp_phase_mark(7);
#endif
  // ---- ENCODED: the buffer is complete; the completion step closes over every encode-time value and runs after nr_body_commit + wait ----
#ifdef NR_EXPERIMENTAL_CHUNKED
  { const uint32_t emitted = res.inventory_compute_encoders + res.inventory_blit_encoders;
    const uint32_t tail = emitted - nr_enc_at_last_split;
    if (ctx.nr_chunk_encoder_ceiling && tail > ctx.nr_chunk_encoder_ceiling)
      throw std::runtime_error("bounded emission: the final command buffer would carry " + std::to_string(tail) +
                               " encoders, over the declared per-buffer ceiling of " + std::to_string(ctx.nr_chunk_encoder_ceiling));
    if (tail > res.nr_max_encoders_per_buffer) res.nr_max_encoders_per_buffer = tail;
    res.nr_tail_launch_lo = nr_launch_at_split; res.nr_tail_launch_hi = nr_cur_launch; res.nr_tail_encoders = tail; }
#endif
  p.cmd = cmd; p.frame_id = ++ctx.frame_serial; ctx.in_flight_id = p.frame_id; ctx.in_flight = true; p.state = NrBodyPending::Encoded;   // v16: this handle owns frame serial frame_id
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  const bool cr_on = [] { const char* v = std::getenv("NR_EXP_CONCURRENT_RECEIPTS"); return v && std::strcmp(v, "1") == 0; }();   // COMPLETION-RECEIPTS-CONCURRENT-v1 opt-in, fixed per frame at encode
#endif
  p.complete = [=, &ctx](NrBodyResult& res, double commit_wall_ms) mutable -> int {   // mutable: the captured expectation tables receive the completion hashes
    std::vector<std::string>& fails = res.check_failures; auto check = [&](bool ok, const std::string& what) { ++res.checks_enforced; if (!ok) fails.push_back(what); };
    struct { bool seen = false, ok = false; std::string hist, out; } pend;   // v10: deferred temporal commit, applied only at the end of a clean frame
    auto mark_damage = [&](const std::string& why) { damage(ctx, in.temporal, res, why); };
    const NrBodyRegion* regs[5] = {&in.skips[0], &in.skips[1], &in.skips[2], &in.skips[3], &in.block0_skip};   // v16: pointers into THIS closure's copy of the inputs, not the caller's struct
    try {
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
      auto wp_t = std::chrono::steady_clock::now();
      auto wp_mark = [&](int k) { const auto wp_n = std::chrono::steady_clock::now(); nr_exp_encode_profile.wp_complete_ms[k] += std::chrono::duration<double, std::milli>(wp_n - wp_t).count(); wp_t = wp_n; };
#endif
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
      // COMPLETION-RECEIPTS-CONCURRENT-v1 (opt-in NR_EXP_CONCURRENT_RECEIPTS=1): the independent, read-only receipt workloads of this closure
      // start here on their own threads - decoder pre-image construction + SHA-256, final pool SHA-256, texture readbacks, hashes and value
      // compare, resident constant re-hash. The sequential code below consumes each value exactly where the serial path computes it, so every
      // receipt value, check, counter and failure message is unchanged. The GPU has completed; the tasks read frame-final bytes through
      // pointers captured here and write only their own result slots. The futures are declared after every result they fill, so on an early
      // exception their destructors join before those results go away.
      struct CrTex { std::vector<uint8_t> post, o, hist_actual; std::string post_sha, final_sha, hsha; size_t hdiff = 0; };
      std::string cr_pre_sha, cr_pool_sha; CrTex cr_tex; std::vector<std::string> cr_ckey, cr_csha; std::vector<const uint8_t*> cr_cptr; std::vector<size_t> cr_cn;
      bool cr_all_cpu = true; for (unsigned i = 0; i < 5; ++i) if (!regs[i]->buffer.contents) cr_all_cpu = false;
      std::future<void> cr_f_pre, cr_f_pool, cr_f_tex; std::vector<std::future<void>> cr_f_const;
      if (cr_on) {
        const uint8_t* cr_mid = static_cast<const uint8_t*>(midkeep.contents); const uint8_t* cr_poolp = static_cast<const uint8_t*>(pool.contents);
        size_t cr_off[5] = {}, cr_bytes[5] = {}; const uint8_t* cr_src[5] = {};
        for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; cr_off[i] = size_t(JInt(P, "offset")); cr_bytes[i] = size_t(JInt(P, "bytes")); if (regs[i]->buffer.contents) cr_src[i] = static_cast<const uint8_t*>(regs[i]->buffer.contents) + regs[i]->offset; }
        if (cr_all_cpu) cr_f_pre = std::async(std::launch::async, [&cr_pre_sha, cr_mid, cr_off, cr_bytes, cr_src, nb = size_t(pool_bytes), pz = uint8_t(poison), at = size_t(fo), b0 = size_t(c0), b1 = size_t(c1)]() {
          std::vector<uint8_t> img(nb, pz); std::memcpy(img.data() + at, cr_mid, VIEW);
          for (unsigned i = 0; i < 5; ++i) std::memcpy(img.data() + cr_off[i], cr_src[i], cr_bytes[i]);
          std::memset(img.data() + b0, 0xff, b1 - b0);
          cr_pre_sha = Sha(img.data(), img.size()); });
        cr_f_pool = std::async(std::launch::async, [&cr_pool_sha, cr_poolp, nb = size_t(pool_bytes)]() { cr_pool_sha = Sha(cr_poolp, nb); });
        bool has_post = false, has_copy = false; for (auto& dl : decl) { if (dl.texmode == "post") has_post = true; if (dl.texmode == "copy") has_copy = true; }
        if (has_post || has_copy) {
          id<MTLTexture> post_src = in.temporal ? tex[2] : tex[1], copy_src = in.temporal ? in.temporal->history : tex[2], copy_out = tex[2];
          cr_f_tex = std::async(std::launch::async, [&cr_tex, has_post, has_copy, post_src, copy_src, copy_out, temporal = in.temporal != nullptr, w = tw, h = th, nb = size_t(texb)]() { @autoreleasepool {
            if (has_post) { cr_tex.post.resize(nb); [post_src getBytes:cr_tex.post.data() bytesPerRow:w * 8 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0]; cr_tex.post_sha = Sha(cr_tex.post.data(), cr_tex.post.size()); }
            if (has_copy) {
              cr_tex.o.resize(nb); [copy_src getBytes:cr_tex.o.data() bytesPerRow:w * 8 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
              if (temporal) { cr_tex.hist_actual = cr_tex.o; std::vector<uint8_t> ob(nb); [copy_out getBytes:ob.data() bytesPerRow:w * 8 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0]; cr_tex.o.swap(ob); }
              cr_tex.final_sha = Sha(cr_tex.o.data(), cr_tex.o.size());
              if (temporal) { cr_tex.hsha = Sha(cr_tex.hist_actual.data(), cr_tex.hist_actual.size()); size_t hd = 0; for (size_t i = 0; i + 1 < cr_tex.o.size(); i += 2) hd += (cr_tex.o[i] != cr_tex.hist_actual[i] || cr_tex.o[i + 1] != cr_tex.hist_actual[i + 1]); cr_tex.hdiff = hd; } } } });
        }
        for (auto& kv : ctx.resident_sha256) if (kv.first.rfind("file:", 0) == 0) { auto bi = ctx.resident.find(kv.first); auto ni = ctx.resident_bytes.find(kv.first);
          if (bi != ctx.resident.end() && ni != ctx.resident_bytes.end()) { cr_ckey.push_back(kv.first); cr_cptr.push_back(static_cast<const uint8_t*>(bi->second.contents)); cr_cn.push_back(ni->second); } }
        cr_csha.assign(cr_ckey.size(), std::string());
        const size_t cr_k = std::min<size_t>(4, cr_ckey.size());   // size-balanced greedy partition; each task writes only its own slots
        std::vector<std::vector<size_t>> cr_part(cr_k); std::vector<size_t> cr_load(cr_k, 0), cr_order(cr_ckey.size());
        for (size_t i = 0; i < cr_order.size(); ++i) cr_order[i] = i;
        std::sort(cr_order.begin(), cr_order.end(), [&](size_t a, size_t b) { return cr_cn[a] > cr_cn[b]; });
        for (size_t i : cr_order) { const size_t k = size_t(std::min_element(cr_load.begin(), cr_load.end()) - cr_load.begin()); cr_part[k].push_back(i); cr_load[k] += cr_cn[i]; }
        for (size_t k = 0; k < cr_k; ++k) cr_f_const.push_back(std::async(std::launch::async, [&cr_csha, &cr_cptr, &cr_cn, idx = std::move(cr_part[k])]() { for (size_t i : idx) cr_csha[i] = Sha(cr_cptr[i], cr_cn[i]); }));
      }
#endif
#ifdef NR_EXPERIMENTAL_TP_STATS
      if (const char* tps_path = std::getenv("NR_EXP_TP_STATS_DUMP")) { std::ofstream tpo(tps_path); tpo << "slot\tlabel\trepaired\n"; const uint32_t* tpv = static_cast<const uint32_t*>(tpsdiag.contents); for (size_t i = 0; i < tps_labels.size(); ++i) tpo << i << "\t" << tps_labels[i] << "\t" << tpv[i] << "\n"; }
#endif
  // ---- COMPLETION: every deferred check on bytes that are still intact; nothing here changes GPU state ----
  const StickyRecord* st = static_cast<const StickyRecord*>(sticky.contents); const ValidatorRecord* vr = static_cast<const ValidatorRecord*>(vrec.contents); const DigestRecord* dr = static_cast<const DigestRecord*>(drec.contents);
  res.validators_run = st->validators_run; res.coverage_faults = st->fault_count;
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // No validators were encoded in this lane, so none are expected. The counters are reported as zero
  // rather than as a passing check, and checks_skipped below says so explicitly.
  res.validators_expected = 0; res.validators_run = 0; res.digests_expected = 0; res.digests_complete = 0;
  res.checks_skipped.push_back("experimental lane: GPU coverage validators, checkpoint digests and in-buffer snapshots are not encoded");
#else
  check(st->validators_run == vexp.size(), "coverage validators: " + std::to_string(st->validators_run) + " ran, " + std::to_string(vexp.size()) + " expected");
#endif
#ifndef NR_EXPERIMENTAL_NO_VALIDATION
  for (size_t i = 0; i < vexp.size(); ++i) { const ValidatorRecord& r = vr[i]; const bool complete = r.done == VALIDATOR_DONE && r.launch_id == vexp[i].launch_id && r.elements == vexp[i].elements;
    check(complete, vexp[i].label + ": validator record incomplete or mismatched (done=" + Hex64(0, r.done).substr(8) + ", launch=" + std::to_string(r.launch_id) + ", elements=" + std::to_string(r.elements) + " expected " + std::to_string(vexp[i].elements) + ")");
    if (complete) check(r.faults == 0, vexp[i].label + ": " + std::to_string(r.faults) + " coverage faults (exactly-once / padding / at-most-once contract)" + (r.faults ? "; first failing element " + std::to_string(r.first_elem) + " observed " + std::to_string(r.first_obs) + " writes" : std::string())); }
  if (st->fault_flag) { std::string first = "none"; for (size_t i = 0; i < vexp.size(); ++i) if (vr[i].done == VALIDATOR_DONE && vr[i].faults) { first = vexp[i].label + " element " + std::to_string(vr[i].first_elem) + " observed " + std::to_string(vr[i].first_obs); break; } check(false, "coverage first failure: " + first + " (" + std::to_string(st->fault_count) + " faulting elements in the frame)"); }
  std::ostringstream dtab; for (size_t i = 0; i < dexp.size(); ++i) { const DigestRecord& r = dr[i]; const bool complete = r.done == DIGEST_DONE && r.launch_id == dexp[i].launch_id && r.words == dexp[i].words; if (complete) ++res.digests_complete;
    check(complete, dexp[i].label + ": checkpoint digest record incomplete or mismatched"); dtab << i << " " << dexp[i].label << " " << r.words << " " << Hex64(r.hi, r.lo) << "\n";
    if (dexp[i].snap) { dexp[i].sha = Sha(dexp[i].snap.contents, size_t(dexp[i].words) * 4); ++res.digest_cpu_crosschecks; const NrDigest64 c = nr_digest64_cpu(static_cast<const uint8_t*>(dexp[i].snap.contents), size_t(dexp[i].words) * 4); const bool same = complete && c.lo == r.lo && c.hi == r.hi; if (!same) ++res.digest_cpu_mismatches; check(same, dexp[i].label + ": GPU digest " + Hex64(r.hi, r.lo) + " != CPU digest of the in-buffer snapshot " + Hex64(c.hi, c.lo)); }
    if (dexp[i].has_cpu) { ++res.digest_cpu_crosschecks; const bool same = complete && r.lo == dexp[i].cpu.lo && r.hi == dexp[i].cpu.hi; if (!same) ++res.digest_cpu_mismatches; check(same, dexp[i].label + ": GPU digest " + Hex64(r.hi, r.lo) + " != CPU digest " + Hex64(dexp[i].cpu.hi, dexp[i].cpu.lo)); } }
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // No digests were encoded, so there is nothing to compare against and nothing may be reported as compared.
  // The digest table is empty and its hash is therefore the hash of nothing, not of a checkpoint set.
  std::ostringstream dtab;
  (void)vr; (void)dr; (void)st;
  auto digest_at = [&](uint32_t) { return NrDigest64{0, 0}; };
  auto crosscheck = [&](uint32_t, const uint8_t*, size_t, const std::string&) { };
#else
  auto digest_at = [&](uint32_t slot) { return NrDigest64{dr[slot].lo, dr[slot].hi}; };
  auto crosscheck = [&](uint32_t slot, const uint8_t* p, size_t bytes, const std::string& what) { ++res.digest_cpu_crosschecks; const NrDigest64 c = nr_digest64_cpu(p, bytes), g = digest_at(slot); if (!(c == g)) ++res.digest_cpu_mismatches; check(c == g, what + ": GPU digest " + Hex64(g.hi, g.lo) + " != CPU digest " + Hex64(c.hi, c.lo)); };
#endif
#ifdef NR_EXPERIMENTAL_ENTRY_ORDERED_CHECK
  // ORDERED ENTRY CHECK, evaluated only now that the command buffer has completed successfully: the two
  // GPU digests captured at the entry-copy boundary - the SOURCE the caller handed us and the DESTINATION
  // the copy produced in the pool - must agree. Both were computed by GPU work encoded in order, so this
  // compares what actually executed rather than what the CPU could see before it ran.
  // `checked` gates it: the production lane encodes no digests, so there would be nothing to compare
  // and a comparison of two absent values would pass vacuously.
  if (checked && entry_src_digest_slot != UINT32_MAX && entry_view_digest_slot != UINT32_MAX) {
    const bool both_done = dr[entry_src_digest_slot].done == DIGEST_DONE && dr[entry_view_digest_slot].done == DIGEST_DONE;
    check(both_done, "entry-view: a boundary digest did not complete, so the copy cannot be verified");
    if (both_done) {
      const NrDigest64 src = digest_at(entry_src_digest_slot), dst = digest_at(entry_view_digest_slot);
      ++res.digest_cpu_crosschecks;
      check(src.lo == dst.lo && src.hi == dst.hi,
            "entry-view: the pool copy " + Hex64(dst.hi, dst.lo).substr(0, 16) + " differs from the entry "
            "source " + Hex64(src.hi, src.lo).substr(0, 16) + " captured at the same boundary");
    }
  }
#endif
  // middle view (accepted receipt hash) and its digest cross-check
  res.middle_view_sha256 = Sha(midkeep.contents, VIEW); crosscheck(mid_digest_slot, static_cast<const uint8_t*>(midkeep.contents), VIEW, "middle view"); if (diag) std::ofstream(out / "out_view_132.bin", std::ios::binary).write(static_cast<const char*>(midkeep.contents), VIEW);
  // decoder pre-image: constructed on the CPU from the same inputs (poison, middle view, caller skip regions, control words); SHA-256 is the accepted receipt hash, the digest proves the GPU restage produced the same bytes
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  if (cr_on) {   // COMPLETION-RECEIPTS-CONCURRENT-v1: the serial block below with the construction + SHA-256 taken from cr_f_pre (in this lane the image is only hashed)
    const bool all_cpu = cr_all_cpu;
    std::string expected_sha; if (all_cpu) { cr_f_pre.get(); expected_sha = cr_pre_sha; crosscheck(pre_digest_slot, nullptr, pool_bytes, "decoder pre-image (expected reconstruction vs GPU digest)"); } else res.checks_skipped.push_back("decoder pre-image CPU construction (a skip region is not CPU-visible)");
    res.decoder_pre_image_expected_sha256 = expected_sha;
    if (dexp[pre_digest_slot].snap) {
      const std::string actual = Sha(dexp[pre_digest_slot].snap.contents, pool_bytes); res.decoder_pre_image_sha256 = actual; res.decoder_pre_image_source = "gpu-snapshot";
      if (all_cpu) check(actual == expected_sha, "decoder pre-image: actual restaged pool (GPU snapshot) " + actual.substr(0, 16) + " != expected reconstruction " + expected_sha.substr(0, 16)); }
    else { res.decoder_pre_image_sha256 = all_cpu ? expected_sha : "digest:" + Hex64(dr[pre_digest_slot].hi, dr[pre_digest_slot].lo); res.decoder_pre_image_source = all_cpu ? "cpu-reconstruction (+gpu-digest-crosscheck)" : "gpu-digest-only"; } }
  else
#endif
  { std::vector<uint8_t> img(pool_bytes, uint8_t(poison)); std::memcpy(img.data() + fo, midkeep.contents, VIEW); bool all_cpu = true;
    for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs[i]; if (r.buffer.contents) std::memcpy(img.data() + off, static_cast<const uint8_t*>(r.buffer.contents) + r.offset, bytes); else all_cpu = false; }
    std::memset(img.data() + c0, 0xff, c1 - c0);
    std::string expected_sha; if (all_cpu) { expected_sha = Sha(img.data(), img.size()); crosscheck(pre_digest_slot, img.data(), img.size(), "decoder pre-image (expected reconstruction vs GPU digest)"); } else res.checks_skipped.push_back("decoder pre-image CPU construction (a skip region is not CPU-visible)");
    res.decoder_pre_image_expected_sha256 = expected_sha;
    if (dexp[pre_digest_slot].snap) {   // v13 checked mode: the ACTUAL restaged pool bytes (GPU snapshot before launch 133) are the identity; the reconstruction is the expectation
      const std::string actual = Sha(dexp[pre_digest_slot].snap.contents, pool_bytes); res.decoder_pre_image_sha256 = actual; res.decoder_pre_image_source = "gpu-snapshot";
      if (all_cpu) check(actual == expected_sha, "decoder pre-image: actual restaged pool (GPU snapshot) " + actual.substr(0, 16) + " != expected reconstruction " + expected_sha.substr(0, 16)); }
    else { res.decoder_pre_image_sha256 = all_cpu ? expected_sha : "digest:" + Hex64(dr[pre_digest_slot].hi, dr[pre_digest_slot].lo); res.decoder_pre_image_source = all_cpu ? "cpu-reconstruction (+gpu-digest-crosscheck)" : "gpu-digest-only"; } }
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  wp_mark(0);   // entry checks + middle view SHA-256 + decoder pre-image (CPU construction + SHA-256)
#endif
  // decoder launches: constants (final re-hash below), constructed record, EX2 faults, textures
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  std::vector<uint8_t> texout(cr_on ? 0 : texb), texback(cr_on ? 0 : texb); std::string tex155_out_sha;
#else
  std::vector<uint8_t> texout(texb), texback(texb); std::string tex155_out_sha;
#endif
  for (auto& dl : decl) {
    const bool p_ok = Sha(dl.bprm.contents, dl.params_size) == dl.p_sha; check(p_ok, "launch " + std::to_string(dl.n) + ": params buffer changed during the frame"); if (!p_ok) ++res.decoder_constant_mutations;
    if (dl.temporal_launch) {   // v6: the half-domain EX2 lookups must all have matched exactly on finite halves
      const uint32_t* fl = static_cast<const uint32_t*>(dl.bfault.contents); res.ex2_half_arg_faults = fl[0]; res.ex2_half_domain_faults = fl[1];
#ifdef NR_BODY_FAULT_TESTS
      if (fault == "ex2-table-fault") res.ex2_half_arg_faults = 1;
#endif
      check(res.ex2_half_arg_faults == 0, "ex2 half-domain table: " + std::to_string(res.ex2_half_arg_faults) + " argument mismatches"); check(res.ex2_half_domain_faults == 0, "ex2 half-domain table: " + std::to_string(res.ex2_half_domain_faults) + " nonfinite half operands");
      check(digest_at(histpre_slot) == digest_at(histpost_slot), "history texture changed during launch 155 (GPU digests before/after differ)"); check(digest_at(motpre_slot) == digest_at(motpost_slot), "motion texture changed during launch 155 (GPU digests before/after differ)"); }
    if (dl.texmode == "post") {   // the post product is still intact at completion (156 only reads it): its bytes cross-check the in-buffer digest taken right after 155
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
      if (cr_on) { if (cr_f_tex.valid()) cr_f_tex.get(); tex155_out_sha = cr_tex.post_sha; crosscheck(tex155_slot, cr_tex.post.data(), texb, "launch 155 output texture");
        if (diag) std::ofstream(out / "tex_out_155.bin", std::ios::binary).write(reinterpret_cast<const char*>(cr_tex.post.data()), std::streamsize(cr_tex.post.size())); } else {
#endif
      id<MTLTexture> to = in.temporal ? tex[2] : tex[1]; [to getBytes:texout.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; tex155_out_sha = Sha(texout.data(), texout.size()); crosscheck(tex155_slot, texout.data(), texb, "launch 155 output texture");
      if (diag) std::ofstream(out / "tex_out_155.bin", std::ios::binary).write(reinterpret_cast<const char*>(texout.data()), std::streamsize(texout.size()));
#ifdef NR_EXPERIMENTAL_NATIVE_POST
      if (diag && np_y16) { auto wr = [&](const char* name, id<MTLBuffer> bb) { std::ofstream(out / name, std::ios::binary).write(static_cast<const char*>(bb.contents), std::streamsize(bb.length - 16)); };
        wr("native_post_xin.bin", np_xin); wr("native_post_y16.bin", np_y16); wr("native_post_y8.bin", np_y8); wr("native_post_tex_after_head.bin", np_texafter);
        if (np_texend1) { wr("native_post_tex1_at_end.bin", np_texend1); wr("native_post_tex2_at_end.bin", np_texend2); } }
#endif
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
      }
#endif
      }
    if (dl.texmode == "copy") {
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
      if (cr_on) {   // COMPLETION-RECEIPTS-CONCURRENT-v1: the serial branch below with the readbacks, hashes and value compare taken from cr_f_tex
        const uint32_t* tcc = static_cast<const uint32_t*>(btexcov.contents); size_t c1n = 0, cmn = 0; for (size_t i = 0; i < size_t(tw) * th; ++i) { if (tcc[i] == 1) ++c1n; else if (tcc[i] > 1) ++cmn; }
        if (cr_f_tex.valid()) cr_f_tex.get();
        if (diag) std::ofstream(out / "tex_out_156.bin", std::ios::binary).write(reinterpret_cast<const char*>((in.temporal ? cr_tex.hist_actual : cr_tex.o).data()), std::streamsize(texb));
        res.final_surface_sha256 = cr_tex.final_sha; res.final_pixels = size_t(tw) * th; res.final_pixels_not_written = size_t(tw) * th - c1n - cmn; res.final_pixels_written_more_than_once = cmn;
        check(res.final_pixels_not_written == 0 && res.final_pixels_written_more_than_once == 0, "final surface not written exactly once (not_written=" + std::to_string(res.final_pixels_not_written) + ", multi=" + std::to_string(res.final_pixels_written_more_than_once) + ")");
        if (in.temporal) {
          const std::string osha_c = cr_tex.final_sha, hsha_c = cr_tex.hsha; const size_t hdiff_c = cr_tex.hdiff;
          res.history_sha256_after = hsha_c; res.history_copy_value_differences = hdiff_c;
          check(hdiff_c == 0 && osha_c == hsha_c, "history after the copy differs from the caller output (" + std::to_string(hdiff_c) + " half values; history " + hsha_c.substr(0, 12) + ", output " + osha_c.substr(0, 12) + "); temporal state NOT advanced");
          pend.seen = true; pend.ok = (hdiff_c == 0 && osha_c == hsha_c); pend.hist = hsha_c; pend.out = osha_c; }
      } else {
#endif
      const uint32_t* tc = static_cast<const uint32_t*>(btexcov.contents); size_t t1 = 0, tm = 0; for (size_t i = 0; i < size_t(tw) * th; ++i) { if (tc[i] == 1) ++t1; else if (tc[i] > 1) ++tm; }   // the last coverage slice is still intact: CPU scan for the receipt fields
#ifdef NR_BODY_FAULT_TESTS
      if (fault == "history-texel-corrupt" && in.temporal) { uint16_t junk[4] = {0x3c00, 0, 0, 0x3c00}; [in.temporal->history replaceRegion:MTLRegionMake2D(5, 7, 1, 1) mipmapLevel:0 withBytes:junk bytesPerRow:8]; }   // real history fault: one texel differs from the caller output
#endif
      std::vector<uint8_t> o(texb); id<MTLTexture> to = in.temporal ? in.temporal->history : tex[2]; [to getBytes:o.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
      std::vector<uint8_t> hist_actual;   // v8: under a temporal state, o = the ACTUAL history texture after the copy (kept), the caller output is read separately
      if (in.temporal) { hist_actual = o; std::vector<uint8_t> ob(texb); [tex[2] getBytes:ob.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; o.swap(ob); }
      if (diag) std::ofstream(out / "tex_out_156.bin", std::ios::binary).write(reinterpret_cast<const char*>((in.temporal ? hist_actual : o).data()), std::streamsize(texb));
      res.final_surface_sha256 = Sha(o.data(), o.size()); res.final_pixels = size_t(tw) * th; res.final_pixels_not_written = size_t(tw) * th - t1 - tm; res.final_pixels_written_more_than_once = tm;
      check(res.final_pixels_not_written == 0 && res.final_pixels_written_more_than_once == 0, "final surface not written exactly once (not_written=" + std::to_string(res.final_pixels_not_written) + ", multi=" + std::to_string(res.final_pixels_written_more_than_once) + ")");
      if (in.temporal) {   // v8: independent hashes of the actual history texture and of the caller output, compared BEFORE the state advances
        const std::string osha = Sha(o.data(), o.size()), hsha = Sha(hist_actual.data(), hist_actual.size()); size_t hdiff = 0; for (size_t i = 0; i + 1 < o.size(); i += 2) hdiff += (o[i] != hist_actual[i] || o[i + 1] != hist_actual[i + 1]);
        res.history_sha256_after = hsha; res.history_copy_value_differences = hdiff;
#ifdef NR_BODY_FAULT_TESTS
        if (fault == "history-copy-mismatch") { hdiff = 1; }
#endif
        check(hdiff == 0 && osha == hsha, "history after the copy differs from the caller output (" + std::to_string(hdiff) + " half values; history " + hsha.substr(0, 12) + ", output " + osha.substr(0, 12) + "); temporal state NOT advanced");
        pend.seen = true; pend.ok = (hdiff == 0 && osha == hsha); pend.hist = hsha; pend.out = osha; }
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
      }
#endif
      }   // v10: commit deferred to the end of the frame
  }
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  wp_mark(1);   // decoder launch receipts: params re-hash, EX2 counters, launch 155/156 texture readbacks, hashes and compare
#endif
#ifdef NR_EXPERIMENTAL_NO_VALIDATION
  // The "before" hash was never taken (see the encode path), so there is nothing to compare against.
  // Reading and hashing another 16.6 MB here would cost about 14 ms in the caller's completion handler
  // and prove nothing. input_texture_immutable stays false and is listed in checks_skipped.
  (void)texback; res.input_texture_immutable = false;
#else
  [tex[0] getBytes:texback.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; res.input_texture_sha256_after = Sha(texback.data(), texback.size()); res.input_texture_immutable = res.input_texture_sha256_after == texin_sha; check(res.input_texture_immutable, "input colour texture changed during the frame");
#endif
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  if (cr_on) { cr_f_pool.get(); res.pool_final_sha256 = cr_pool_sha; crosscheck(final_pool_slot, nullptr, pool_bytes, "final pool"); } else {
#endif
  res.pool_final_sha256 = Sha(pool.contents, pool_bytes); crosscheck(final_pool_slot, static_cast<const uint8_t*>(pool.contents), pool_bytes, "final pool");
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  }
#endif
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  wp_mark(2);   // input texture (absent in this lane) + final pool SHA-256
#endif
  if (final_only) res.checks_skipped = {"decoder per-launch pool digest", "decoder per-launch pool coverage validator", "intermediate SHA-256 snapshots (checked mode only; GPU digests + boundary SHA-256 in production)", "decoder pre-image actual GPU snapshot SHA-256 (production reports the CPU reconstruction with the GPU digest cross-check)"};
  if (diag && JStr(D, "write_pool_post") == "true") std::ofstream(out / "pool_post.bin", std::ios::binary).write(static_cast<const char*>(pool.contents), std::streamsize(pool_bytes));   // v14: explicit request only
  res.digest_table_sha256 = Sha(dtab.str().data(), dtab.str().size());
  const double wall_total = -[t_all timeIntervalSinceNow] * 1000.0; res.middle_gpu_ms = 0; res.decoder_gpu_ms = 0;   // v14: res.wall_ms is set after the final re-hash/commit below   // one command buffer: per-section GPU time is not separable; frame_gpu_ms carries the whole frame
  // ---- diagnostics JSON (launch list with validator/digest slots, the digest table, evidence summary) ----
  if (diag) { std::ostringstream js; js << "{\n \"schema\": \"neuralstudio.nr.body-runtime.v17-resident-experimental\",\n \"mode\": \"" << (checked ? "checked (in-buffer snapshots, SHA-256 per intermediate)" : "production (digests per intermediate, SHA-256 at the frame boundaries)") << "\",\n \"pid\": " << getpid() << ",\n \"poison\": " << poison << ",\n \"command_buffers\": " << res.inventory_command_buffers << ",\n \"frame_gpu_ms\": " << res.frame_gpu_ms << ",\n \"commit_to_completion_wall_ms\": " << commit_wall_ms << ",\n \"launches\": [\n"; bool first = true;
    for (auto& l : launches) { js << (first ? "" : ",\n") << "  {\"launch\": " << l.n << ", \"kind\": \"" << l.kind << "\", \"digests\": {"; bool f2 = true; for (auto& d : l.digests) { const DigestRecord& r = dr[d.second]; js << (f2 ? "" : ", ") << "\"" << d.first << "\": \"" << Hex64(r.hi, r.lo) << "\""; if (dexp[d.second].snap) js << ", \"" << d.first << "_sha256\": \"" << dexp[d.second].sha << "\""; f2 = false; } js << "}, \"validators\": ["; f2 = true; for (auto s : l.validators) { const ValidatorRecord& r = vr[s]; js << (f2 ? "" : ", ") << "{\"slot\": " << s << ", \"elements\": " << r.elements << ", \"faults\": " << r.faults << ", \"done\": " << (r.done == VALIDATOR_DONE ? "true" : "false") << "}"; f2 = false; } js << "]}"; first = false; }
    js << "\n ],\n \"digest_table\": [\n"; for (size_t i = 0; i < dexp.size(); ++i) { const DigestRecord& r = dr[i]; js << (i ? ",\n" : "") << "  {\"slot\": " << i << ", \"label\": \"" << dexp[i].label << "\", \"launch\": " << dexp[i].launch_id << ", \"words\": " << r.words << ", \"digest\": \"" << Hex64(r.hi, r.lo) << "\", \"done\": " << (r.done == DIGEST_DONE ? "true" : "false") << (dexp[i].has_cpu ? ", \"cpu\": \"" + Hex64(dexp[i].cpu.hi, dexp[i].cpu.lo) + "\"" : std::string()) << (dexp[i].snap ? ", \"sha256\": \"" + dexp[i].sha + "\"" : std::string()) << "}"; }
    js << "\n ],\n \"digest_table_sha256\": \"" << res.digest_table_sha256 << "\",\n \"validators\": {\"expected\": " << vexp.size() << ", \"run\": " << st->validators_run << ", \"faults\": " << st->fault_count << "},\n \"middle_view_sha256\": \"" << res.middle_view_sha256 << "\",\n \"decoder_pre_image_sha256\": \"" << res.decoder_pre_image_sha256 << "\",\n \"decoder_pre_image_source\": \"" << res.decoder_pre_image_source << "\",\n \"decoder_pre_image_expected_sha256\": \"" << res.decoder_pre_image_expected_sha256 << "\",\n \"pool_final_sha256\": \"" << res.pool_final_sha256 << "\",\n \"final_surface_sha256\": \"" << res.final_surface_sha256 << "\",\n \"tex155_output_sha256\": \"" << tex155_out_sha << "\",\n \"wall_ms_total\": " << wall_total << "\n}\n";
    std::ofstream(out / "chain-run.json") << js.str(); }
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  if (cr_on) std::cout << "body receipts: concurrent (COMPLETION-RECEIPTS-CONCURRENT-v1)\n";
#endif
  std::cout << "frame gpu_ms " << res.frame_gpu_ms << " wall_ms " << wall_total << " validators " << st->validators_run << "/" << vexp.size() << " digests " << res.digests_complete << "/" << dexp.size() << "\n";
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  wp_mark(3);   // digest table + frame report
#endif
  // v3 provenance: report every library/constant identity recorded at load and re-hash every resident constant now (post-frame)
#ifdef NR_BODY_FAULT_TESTS
  if (fault == "constant-late-mutation") for (auto& kv : ctx.resident) if (kv.first.rfind("file:", 0) == 0) { static_cast<uint8_t*>(kv.second.contents)[0] ^= 1u; break; }   // after the last use of every constant: only the final re-hash can see it
  if (fault == "decoder-weights") for (auto& dl : decl) if (dl.n == 133u) { static_cast<uint8_t*>(dl.bw.contents)[5] ^= 0x01; break; }   // launch-133 weights flipped after completion: pixels unaffected, the resident re-hash must trip
#endif
  res.constants_unchanged_after_frame = true;
#ifdef NR_EXPERIMENTAL_CONCURRENT_RECEIPTS
  if (cr_on) {   // COMPLETION-RECEIPTS-CONCURRENT-v1: the same loop; each re-hash was computed by the partition tasks for the same entry list, in the same order
    for (auto& f : cr_f_const) f.get();
    size_t ci = 0;
    for (auto& kv : ctx.resident_sha256) { res.provenance.push_back(kv); if (kv.first.rfind("file:", 0) == 0) { auto bi = ctx.resident.find(kv.first); auto ni = ctx.resident_bytes.find(kv.first); if (bi != ctx.resident.end() && ni != ctx.resident_bytes.end()) {
      if (ci >= cr_ckey.size() || cr_ckey[ci] != kv.first) throw std::runtime_error("concurrent receipts: the resident constant list changed during completion");
      ++res.constants_rehashed; if (cr_csha[ci++] != kv.second) { res.constants_unchanged_after_frame = false; ++res.decoder_constant_mutations; fails.push_back("model constant changed since load: " + kv.first); } } } }
    if (ci != cr_ckey.size()) throw std::runtime_error("concurrent receipts: the resident constant list changed during completion"); }
  else
#endif
  for (auto& kv : ctx.resident_sha256) { res.provenance.push_back(kv); if (kv.first.rfind("file:", 0) == 0) { auto bi = ctx.resident.find(kv.first); auto ni = ctx.resident_bytes.find(kv.first); if (bi != ctx.resident.end() && ni != ctx.resident_bytes.end()) { ++res.constants_rehashed; if (Sha(bi->second.contents, ni->second) != kv.second) { res.constants_unchanged_after_frame = false; ++res.decoder_constant_mutations; fails.push_back("model constant changed since load: " + kv.first); } } } }
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  wp_mark(4);   // resident constant re-hash
#endif
  if (in.temporal && pend.seen) {   // v10: the ONLY place the temporal metadata changes: after every check of this frame, and only if none failed
    res.temporal_state_advanced = fails.empty() && pend.ok;
    if (res.temporal_state_advanced) { in.temporal->history_sha256 = pend.hist; in.temporal->last_output_sha256 = pend.out; in.temporal->frames_since_reset = in.reset ? 1u : in.temporal->frames_since_reset + 1u; } }
  res.wall_ms = -[t_all timeIntervalSinceNow] * 1000.0;   // v14: includes completion checks, the constant re-hash and the temporal commit decision
#ifdef NR_EXPERIMENTAL_WALL_PROFILE
  wp_mark(5);   // temporal commit decision
#endif
  if (!fails.empty()) { std::ostringstream e; e << fails.size() << " computed check(s) failed: "; for (size_t i = 0; i < fails.size() && i < 6; ++i) e << (i ? "; " : "") << fails[i]; if (fails.size() > 6) e << "; ..."; res.code = 3; res.error = e.str(); mark_damage("code 3: " + fails[0]); return 3; }
  res.code = 0; res.error.clear(); return 0;
    } catch (const std::exception& e) { res.code = 2; res.error = e.what(); mark_damage(std::string("exception after commit: ") + e.what()); return 2; }
  };
  return 0;
  } catch (const std::exception& e) { res.code = 2; res.error = e.what(); p.state = NrBodyPending::Failed; return 2; } } }
int nr_body_commit(NrBodyPending& p) {
  if (p.state == NrBodyPending::Moved) return misuse(p, "commit on a moved-from pending handle");
  { auto a = p.ctx_alive.lock(); if (p.state == NrBodyPending::Encoded && (!a || !*a)) { p.state = NrBodyPending::Failed; p.cmd = nil; p.complete = nullptr; return misuse(p, "context destroyed while the frame was pending"); } }
  if (p.state != NrBodyPending::Encoded) return misuse(p, p.state == NrBodyPending::Committed ? "commit on an already committed frame" : "commit on a frame that is not encoded");
  if (!p.ctx || p.ctx->in_flight_id != p.frame_id || p.frame_id == 0) return misuse(p, "stale pending handle: this handle does not own the context's in-flight frame");
#ifdef NR_EXPERIMENTAL_CALLER_COMMAND
  // The CALLER owns this buffer and commits it; we only advance the state machine so that complete() is
  // still gated on a committed frame. We must never commit a buffer we do not own.
  p.t_commit = [[NSDate date] timeIntervalSince1970]; p.state = NrBodyPending::Committed; return 0; }
#else
  p.t_commit = [[NSDate date] timeIntervalSince1970]; [p.cmd commit]; p.state = NrBodyPending::Committed; return 0; }
#endif
int nr_body_complete(NrBodyPending& p, NrBodyResult& res) { @autoreleasepool {
  if (p.state == NrBodyPending::Moved) { const int rc = misuse(p, "complete on a moved-from pending handle"); res = p.res; return rc; }
  { auto a = p.ctx_alive.lock(); if ((p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) && (!a || !*a)) { if (p.cmd && p.state == NrBodyPending::Committed) [p.cmd waitUntilCompleted]; if (p.state == NrBodyPending::Committed && p.temporal) damage_temporal_only(p.temporal); p.state = NrBodyPending::Failed; p.cmd = nil; p.complete = nullptr; p.frame_id = 0; const int rc = misuse(p, "context destroyed while the frame was pending (temporal state damaged if the buffer had been committed)"); res = p.res; return rc; } }
  if (p.state != NrBodyPending::Committed) { const int rc = misuse(p, p.state == NrBodyPending::Encoded ? "complete before commit" : "complete on a frame that is not committed (idle, completed or failed)"); res = p.res; return rc; }
  if (!p.ctx || p.ctx->in_flight_id != p.frame_id || p.frame_id == 0 || !p.cmd || !p.complete) { const int rc = misuse(p, "stale pending handle: this handle does not own the context's in-flight frame"); res = p.res; return rc; }
#ifdef NR_EXPERIMENTAL_CALLER_COMMAND
  // Called from the CALLER's completion handler: the buffer has already finished, so there is nothing to
  // wait for and waiting here would deadlock inside the callback.
  if (p.cmd.status != MTLCommandBufferStatusCompleted && p.cmd.status != MTLCommandBufferStatusError) return misuse(p, "experimental lane: complete() must be called from the caller command buffer's completion handler");
  const double commit_wall_ms = ([[NSDate date] timeIntervalSince1970] - p.t_commit) * 1000.0; int rc;
#else
  ++p.res.inventory_waits; [p.cmd waitUntilCompleted]; const double commit_wall_ms = ([[NSDate date] timeIntervalSince1970] - p.t_commit) * 1000.0; int rc;
#endif
  if (p.cmd.status != MTLCommandBufferStatusCompleted) { p.res.code = 2; p.res.error = p.cmd.error ? p.cmd.error.localizedDescription.UTF8String : "command buffer failed"; damage(*p.ctx, p.temporal, p.res, "command buffer failed: " + p.res.error); rc = 2; }
  else { p.res.frame_gpu_ms = (p.cmd.GPUEndTime - p.cmd.GPUStartTime) * 1000.0; rc = p.complete(p.res, commit_wall_ms); }
  p.ctx->in_flight = false; p.ctx->in_flight_id = 0; p.state = rc == 0 ? NrBodyPending::Completed : NrBodyPending::Failed; p.complete = nullptr; p.cmd = nil; p.frame_id = 0; res = p.res; return rc; } }
void nr_body_pending_reset(NrBodyPending& p) { if (p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) return; p.state = NrBodyPending::Idle; p.cmd = nil; p.complete = nullptr; p.ctx = nullptr; p.temporal = nullptr; p.res = NrBodyResult{}; p.t_commit = 0; p.frame_id = 0; }   // an in-flight pending cannot be reset
void nr_body_pending_abandon(NrBodyPending& p) {   // v16/v17: destructor / move-assignment target; bounded abandonment semantics, ownership-guarded
  auto a = p.ctx_alive.lock(); const bool ctx_ok = a && *a && p.ctx; const bool owner = ctx_ok && p.frame_id != 0 && p.ctx->in_flight_id == p.frame_id;   // only the handle that owns the context's in-flight frame can release or damage it
  if (p.state == NrBodyPending::Encoded) { if (owner) { p.ctx->in_flight = false; p.ctx->in_flight_id = 0; } }   // never submitted: release the reservation, no damage
  else if (p.state == NrBodyPending::Committed) { if (p.cmd) [p.cmd waitUntilCompleted];
    if (owner) { damage(*p.ctx, p.temporal, p.res, "frame abandoned after commit without completion (checks never ran)"); p.ctx->in_flight = false; p.ctx->in_flight_id = 0; }
    else if (!ctx_ok && p.frame_id != 0 && p.cmd) damage_temporal_only(p.temporal); }   // the context is gone: only the temporal state can still be marked
  p.state = (p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) ? NrBodyPending::Failed : p.state; p.cmd = nil; p.complete = nullptr; p.frame_id = 0; p.ctx_alive.reset(); }
int nr_body_run_frame(NrBodyContext& ctx, NSDictionary* chain, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyResult& res) {
  NrBodyPending p; int rc = nr_body_encode(ctx, chain, in, diagnostics_dir, p); if (rc) { res = p.res; return rc; } rc = nr_body_commit(p); if (rc) { res = p.res; return rc; } return nr_body_complete(p, res); }
