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
static id<MTLBuffer> resident_buf(NrBodyContext& ctx, id<MTLDevice> dev, const std::string& key, size_t length) { auto it = ctx.resident.find(key); if (it != ctx.resident.end() && it->second.length == length) return it->second; id<MTLBuffer> b = [dev newBufferWithLength:length options:MTLResourceStorageModeShared]; if (!b) throw std::runtime_error("buffer allocation failed: " + key); ctx.resident[key] = b; return b; }
static id<MTLTexture> resident_tex(NrBodyContext& ctx, id<MTLDevice> dev, const std::string& key, MTLTextureDescriptor* d) { auto it = ctx.resident_tex.find(key); if (it != ctx.resident_tex.end() && it->second.width == d.width && it->second.height == d.height && it->second.pixelFormat == d.pixelFormat) return it->second; id<MTLTexture> t = [dev newTextureWithDescriptor:d]; if (!t) throw std::runtime_error("texture allocation failed: " + key); ctx.resident_tex[key] = t; return t; }
static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("live input rejected: ") + what); }
static void damage(NrBodyContext& ctx, NrTemporalState* t, NrBodyResult& res, const std::string& why) { ctx.damaged = why; res.damage = why; if (t) { t->history_sha256 = "damaged:" + why.substr(0, 96); t->last_output_sha256 = "damaged"; } }
static int misuse(NrBodyPending& p, const char* what) { p.res.code = 2; p.res.error = std::string("misuse: ") + what; return 2; }
int nr_body_encode(NrBodyContext& ctx, NSDictionary* chain, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyPending& p) { @autoreleasepool {
  if (p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) return misuse(p, "encode on a pending frame that is encoded/committed and not completed");
  if (p.state == NrBodyPending::Failed) return misuse(p, "encode on a failed pending frame: call nr_body_pending_reset first");
  if (ctx.in_flight || ctx.in_flight_id) { p.res = NrBodyResult{}; return misuse(p, "another frame is in flight on this context (encode before the previous complete)"); }
  p.res = NrBodyResult{}; p.cmd = nil; p.complete = nullptr; p.ctx = &ctx; p.temporal = in.temporal; p.frame_id = 0; NrBodyResult& res = p.res;
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
  auto lib = [&](const std::string& p) { auto it = ctx.libs.find(p); if (it != ctx.libs.end()) return it->second; id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]] error:&err]; if (!l) throw std::runtime_error(err.localizedDescription.UTF8String); ctx.libs[p] = l;
    { std::vector<uint8_t> lb = Read(p); ctx.resident_sha256["lib:" + p] = Sha(lb.data(), lb.size()); }   // v3: library identity at load time
    return l; };
  id<MTLLibrary> slib = lib(JStr(chain, "split_lib")), vlib = lib(JStr(chain, "vit_lib"));
  const std::string tplib_path = JStr(chain, "vit_tp_lib"); id<MTLLibrary> tplib = tplib_path.empty() ? nil : lib(tplib_path);   // optional: three-product expand/contract/projection (bit-identical, faster)
  const std::string sptp_path = JStr(chain, "split_tp_lib"); id<MTLLibrary> sptp = sptp_path.empty() ? nil : lib(sptp_path);   // optional: three-product split conv (all flag variants)
  const std::string rlib_path = JStr(chain, "resident_lib"); if (rlib_path.empty()) throw std::runtime_error("descriptor has no resident_lib (v12 requires the validator/digest library nr_body_validate.metal)"); id<MTLLibrary> rlib = lib(rlib_path);
  auto pso = [&](id<MTLLibrary> l, const char* name) { const std::string key = std::to_string(reinterpret_cast<uintptr_t>((__bridge void*)l)) + "|" + name; auto it = ctx.pipes.find(key); if (it != ctx.pipes.end()) return it->second; id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err]; if (!ps) throw std::runtime_error(std::string(name) + ": " + err.localizedDescription.UTF8String); ctx.pipes[key] = ps; return ps; };
  id<MTLComputePipelineState> psoValidate = pso(rlib, "nr_cov_validate"), psoDigest = pso(rlib, "nr_digest64"), psoFirstObs = pso(rlib, "nr_cov_first_obs"), psoPadMask = pso(rlib, "nr_apply_pad_mask");
  const size_t pool_bytes = size_t(JInt(chain, "pool_bytes")); id<MTLBuffer> pool = resident_buf(ctx, dev, "pool", pool_bytes + 4096); std::memset(pool.contents, poison, pool_bytes); std::memset(static_cast<uint8_t*>(pool.contents) + pool_bytes, 0, 4096);
  require(pool_bytes % 64 == 0, "pool size must be a multiple of 64 bytes (digest chunks)");
  require(in.entry.buffer != nil && in.entry.bytes == VIEW && in.entry.offset + VIEW <= in.entry.buffer.length, "encoder entry buffer/bytes/offset");
  unsigned scratch_seq = 0;   // v2: scratch buffers are resident in the context, keyed by their (deterministic) request order and size, re-filled every frame
  auto scratch = [&](size_t n, uint8_t fill) { id<MTLBuffer> b = resident_buf(ctx, dev, "scratch:" + std::to_string(scratch_seq++) + ":" + std::to_string(n), n + 16); std::memset(b.contents, fill, n + 16); return b; };
  auto constant = [&](const fs::path& p, size_t pad = 16) { const std::string key = "file:" + p.string() + ":" + std::to_string(pad); auto it = ctx.resident.find(key); if (it != ctx.resident.end()) return it->second;
    std::vector<uint8_t> v = Read(p); id<MTLBuffer> b = [dev newBufferWithLength:v.size() + pad options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + pad); std::memcpy(b.contents, v.data(), v.size()); ctx.resident[key] = b; ctx.resident_sha256[key] = Sha(v.data(), v.size()); ctx.resident_bytes[key] = v.size(); return b; };   // model constants: loaded and hashed once per process
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
  // ---- v12 evidence tables: poisoned records, sticky validator state, expectations recorded while encoding ----
  id<MTLBuffer> vrec = scratch(MAX_RECORDS * sizeof(ValidatorRecord), 0), drec = scratch(MAX_RECORDS * sizeof(DigestRecord), 0), sticky = scratch(64, 0), padmask = scratch(VITB, 0);
  { ValidatorRecord* vr = static_cast<ValidatorRecord*>(vrec.contents); for (size_t i = 0; i < MAX_RECORDS; ++i) vr[i] = ValidatorRecord{POISON_WORD, POISON_WORD, 0u, POISON_WORD, 0xffffffffu, POISON_WORD, 0u, 0u};
    DigestRecord* dr = static_cast<DigestRecord*>(drec.contents); for (size_t i = 0; i < MAX_RECORDS; ++i) dr[i] = DigestRecord{POISON_WORD, POISON_WORD, 0u, 0u, POISON_WORD, 0u, 0u, 0u};
    *static_cast<StickyRecord*>(sticky.contents) = StickyRecord{0u, 0u, 0u, 0u}; }
  struct ValExp { uint32_t launch_id, elements; std::string label; }; std::vector<ValExp> vexp;
  struct DigExp { uint32_t launch_id = 0, words = 0; std::string label; bool has_cpu = false; NrDigest64 cpu; id<MTLBuffer> snap = nil; std::string sha; }; std::vector<DigExp> dexp;
  struct LaunchRec { long n; std::string kind; std::vector<std::pair<std::string, uint32_t>> digests; std::vector<uint32_t> validators; std::string extra; }; std::vector<LaunchRec> launches;
  id<MTLCommandBuffer> cmd = [queue commandBuffer]; ++res.inventory_command_buffers; if (!cmd) throw std::runtime_error("command buffer allocation failed");
  const auto t_all = [NSDate date];
  auto blit = [&]() { id<MTLBlitCommandEncoder> b = [cmd blitCommandEncoder]; ++res.inventory_blit_encoders; return b; };
  auto compute = [&]() { id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder]; ++res.inventory_compute_encoders; return e; };
  auto fill = [&](id<MTLBuffer> b, size_t off, size_t len, uint8_t v) { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:b range:NSMakeRange(off, len) value:v]; [bl endEncoding]; };
  // one coverage validator over `elements` counters of `counters` (mode 0 exact once, 1 expected map `expect`, 2 at most once); returns its record slot
  auto validate = [&](const std::string& label, uint32_t launch_id, id<MTLBuffer> counters, size_t elements, uint32_t mode, id<MTLBuffer> expect, bool encode = true, size_t encoded_elements = 0) {
    if (vexp.size() >= MAX_RECORDS) throw std::runtime_error("validator record table full"); const uint32_t slot = uint32_t(vexp.size()); vexp.push_back({launch_id, uint32_t(elements), label}); ++res.validators_expected;
    if (encode) { ValidateArgs a{uint32_t(encoded_elements ? encoded_elements : elements), launch_id, slot, mode}; id<MTLComputeCommandEncoder> e = compute(); [e setComputePipelineState:psoValidate];
      [e setBuffer:counters offset:0 atIndex:0]; [e setBuffer:expect ? expect : padmask offset:0 atIndex:1]; [e setBuffer:sticky offset:0 atIndex:2]; [e setBuffer:vrec offset:0 atIndex:3]; [e setBytes:&a length:sizeof a atIndex:4];
      const size_t threads = std::min<size_t>(a.count, 1u << 18); [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding];
      id<MTLComputeCommandEncoder> e2 = compute(); [e2 setComputePipelineState:psoFirstObs]; [e2 setBuffer:counters offset:0 atIndex:0]; [e2 setBuffer:vrec offset:0 atIndex:1]; [e2 setBytes:&a length:sizeof a atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e2 endEncoding]; }   // v13: observed count of the first faulting element while the counters are intact
    return slot; };
  // one checkpoint digest of `bytes` bytes at `off` in `b` (word-sized); optional CPU digest of the same bytes for the boundary cross-check
  // `snapshot` (checked mode only): the same bytes are also blit-copied into a resident snapshot buffer at this point of the frame and SHA-256 hashed at completion
  auto digest = [&](const std::string& label, uint32_t launch_id, id<MTLBuffer> b, size_t off, size_t bytes, const uint8_t* cpu_bytes = nullptr, bool encode = true, bool snapshot = true) {
    if (dexp.size() >= MAX_RECORDS) throw std::runtime_error("digest record table full"); if (bytes % 4 != 0 || off % 4 != 0) throw std::runtime_error("digest region not word aligned: " + label);
    const uint32_t slot = uint32_t(dexp.size()); DigExp d; d.launch_id = launch_id; d.words = uint32_t(bytes / 4); d.label = label; if (cpu_bytes) { d.has_cpu = true; d.cpu = nr_digest64_cpu(cpu_bytes, bytes); }
    if (checked && snapshot) { d.snap = scratch(bytes, 0); id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:b sourceOffset:off toBuffer:d.snap destinationOffset:0 size:bytes]; [bl endEncoding]; }
    dexp.push_back(d); ++res.digests_expected;
    if (encode) { DigestArgs a{uint32_t(bytes / 4), launch_id, slot, 0u}; id<MTLComputeCommandEncoder> e = compute(); [e setComputePipelineState:psoDigest]; [e setBuffer:b offset:off atIndex:0]; [e setBuffer:drec offset:0 atIndex:1]; [e setBytes:&a length:sizeof a atIndex:2];
      const size_t chunks = (bytes / 4 + 15) / 16, threads = std::min<size_t>(chunks, 1u << 16); [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding]; }
    return slot; };
  auto tex_to_buf = [&](id<MTLTexture> t, id<MTLBuffer> b, size_t bpp) { id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(t.width, t.height, 1) toBuffer:b destinationOffset:0 destinationBytesPerRow:t.width * bpp destinationBytesPerImage:t.width * t.height * bpp]; [bl endEncoding]; };
#ifdef NR_BODY_FAULT_TESTS
  auto poke = [&](id<MTLBuffer> b, size_t byte, uint8_t v) { fill(b, byte, 1, v); };   // REAL in-buffer fault: one byte of a coverage slice or buffer changed between the launch and its validator/digest
#endif
  // ---- entry blit (caller buffer -> pool) + digest of the entry view, cross-checked against the caller bytes ----
  { const size_t off = size_t(JInt(chain, "entry_offset")); id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:in.entry.buffer sourceOffset:in.entry.offset toBuffer:pool destinationOffset:off size:VIEW]; [bl endEncoding];
    const uint8_t* ecpu = in.entry.buffer.contents ? static_cast<const uint8_t*>(in.entry.buffer.contents) + in.entry.offset : nullptr; launches.push_back({24, "entry", {{"entry_view", digest("entry-view", 24u, pool, off, VIEW, ecpu, true, false)}}, {}, ""}); }
  id<MTLBuffer> rsq = constant(tables / "rtx-rsqrt-half.bin"), rcp = constant(tables / "rtx-rcp-half.bin");
  const size_t covmax = std::max(T * 4096 * 4, TIN * 4); id<MTLBuffer> cov = scratch(covmax, 0), cov2 = scratch(POOLB * 4, 0), y16 = scratch(TIN * 2, 0), qb = scratch(TIN, uint8_t(poison)), kb = scratch(TIN, uint8_t(poison)), vb = scratch(TIN, uint8_t(poison));
  // launch-57 padding contract map (1 = padding byte, must never be written; used as the validator's expected map inverted below and by the pad-mask kernel)
  std::vector<uint8_t> is_pad(VITB, 0), expect57(VITB, 1); { NSArray* pad = chain[@"vit"][@"padding_tokens"]; for (NSNumber* t in pad) { const unsigned tok = t.unsignedIntValue; for (unsigned c = 0; c < 1024; ++c) { const unsigned tile = tok / 16, slot = tok % 16, lane = 4 * (slot % 8) + (c % 16) / 4, reg = (slot % 16) / 8 + 2 * ((c % 32) / 16); const size_t i = tile * 16384 + (c / 32) * 512 + lane * 16 + reg * 4 + c % 4; is_pad[i] = 1; expect57[i] = 0; } } res.out_vit_padding_bytes = size_t(pad.count) * 1024u; }
  std::memcpy(padmask.contents, is_pad.data(), VITB); id<MTLBuffer> expect57b = scratch(VITB, 0); std::memcpy(expect57b.contents, expect57.data(), VITB);
  id<MTLBuffer> spw[3], spmax, spstats; if (sptp) { for (int m = 0; m < 3; ++m) spw[m] = scratch(16u * 32u * 512u * 2u, 0); spmax = scratch(16u * 512u, 0); spstats = scratch(64, 0); }
  id<MTLBuffer> atk[3], atv[3], atkmax, atvmax; if (sptp) { for (int m = 0; m < 3; ++m) { atk[m] = scratch(40u * 16u * 32u * 64u * 2u, 0); atv[m] = scratch(40u * 16u * 2u * 32u * 32u * 2u, 0); } atkmax = scratch(40u * 16u * 64u, 0); atvmax = scratch(40u * 16u * 2u * 32u, 0); }   // split attend operands per (window, head)
  id<MTLBuffer> qw[3], qwmax; if (sptp) { for (int m = 0; m < 3; ++m) qw[m] = scratch(16u * 32u * 1536u * 2u, 0); qwmax = scratch(16u * 1536u, 0); }   // split qkv weights: 16 steps x 1536 columns
  id<MTLBuffer> fw2[3], fw3[3], fw2max, fw3max; if (sptp) { for (int m = 0; m < 3; ++m) { fw2[m] = scratch(8u * 2u * 32u * 256u * 2u, 0); fw3[m] = scratch(8u * 8u * 32u * 64u * 2u, 0); } fw2max = scratch(8u * 2u * 256u, 0); fw3max = scratch(8u * 8u * 64u, 0); }
  // ---- split launches (encoded; coverage validated and outputs digested in-buffer before the slices are refilled) ----
  auto run_split = [&](NSDictionary* L) {
    const long n = JInt(L, "launch"); const std::string kernel = JStr(L, "kernel"); NSArray* shift = L[@"shift"]; NSDictionary* roles = L[@"roles"]; auto weights = constant(JStr(L, "weights"));
    auto off = [&](const char* role) { NSDictionary* r = roles[[NSString stringWithUTF8String:role]]; if (!r) throw std::runtime_error(std::to_string(n) + ": role " + role); return size_t(JInt(r, "pool_offset")); };
    SplitParams prm{int32_t([shift[0] intValue]), int32_t([shift[1] intValue]), 0, 2160};
    { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:cov range:NSMakeRange(0, covmax + 16) value:0]; [bl fillBuffer:cov2 range:NSMakeRange(0, POOLB * 4 + 16) value:0]; [bl endEncoding]; }
    std::vector<std::pair<std::string, size_t>> outbytes;
    auto dispatch = [&](const char* name, std::vector<std::pair<id<MTLBuffer>, size_t>> bufs, size_t threads) {
      id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(slib, name)];
      for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i].first offset:bufs[i].second atIndex:i];
      [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
    };
    size_t o_off = 0, o_bytes = 0; std::string o_role;
    if (kernel == "ffwd" || kernel == "ffwd_view") { const bool view = kernel == "ffwd_view"; prm.flags = view ? FLAG_IN_VIEW : 0; o_role = "out"; o_off = off("out"); o_bytes = TIN;
      if (sptp) {
        { id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:1 + m]; [enc setBuffer:spmax offset:0 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(16u * 512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_ffwd_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw2[m] offset:0 atIndex:1 + m]; [enc setBuffer:fw2max offset:0 atIndex:4]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw3[m] offset:0 atIndex:5 + m]; [enc setBuffer:fw3max offset:0 atIndex:8];
          [enc dispatchThreads:MTLSizeMake(8u * 2u * 256u + 8u * 8u * 64u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_ffwd_tp")];
        [enc setBuffer:pool offset:off(view ? "x_view" : "x") atIndex:0]; [enc setBuffer:weights offset:0 atIndex:1]; [enc setBuffer:pool offset:o_off atIndex:2]; [enc setBuffer:cov offset:0 atIndex:3]; [enc setBytes:&prm length:sizeof(prm) atIndex:4];
        for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:5 + m]; [enc setBuffer:spmax offset:0 atIndex:8]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw2[m] offset:0 atIndex:9 + m]; [enc setBuffer:fw2max offset:0 atIndex:12]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw3[m] offset:0 atIndex:13 + m]; [enc setBuffer:fw3max offset:0 atIndex:16]; [enc setBuffer:spstats offset:0 atIndex:17];
        [enc dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
      } else
      dispatch("nr_split_ffwd", {{pool, off(view ? "x_view" : "x")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 2160u * 8u); }
    else if (kernel == "conv" || kernel == "conv_resview" || kernel == "conv_outview" || kernel == "conv_pool") {
      const bool rv = kernel == "conv_resview", ov = kernel == "conv_outview", pl = kernel == "conv_pool"; prm.flags = (rv ? FLAG_RES_VIEW : 0) | (ov ? FLAG_OUT_VIEW : 0) | (pl ? FLAG_EMIT_F16 : 0);
      o_role = ov ? "out_view" : "out"; o_off = off(o_role.c_str()); o_bytes = ov ? VIEW : TIN;
      if (sptp) {
        { id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:1 + m]; [enc setBuffer:spmax offset:0 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(16u * 512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_conv_tp")];
        [enc setBuffer:pool offset:off("x") atIndex:0]; [enc setBuffer:pool offset:off(rv ? "res_view" : "res") atIndex:1]; [enc setBuffer:weights offset:0 atIndex:2]; [enc setBuffer:pool offset:o_off atIndex:3]; [enc setBuffer:y16 offset:0 atIndex:4]; [enc setBuffer:cov offset:0 atIndex:5];
        [enc setBytes:&prm length:sizeof(prm) atIndex:6]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:7 + m]; [enc setBuffer:spmax offset:0 atIndex:10]; [enc setBuffer:spstats offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
      } else
      dispatch("nr_split_conv", {{pool, off("x")}, {pool, off(rv ? "res_view" : "res")}, {weights, 0}, {pool, o_off}, {y16, 0}, {cov, 0}}, 2160u * 32u);
      if (pl) { dispatch("nr_split_pool", {{y16, 0}, {pool, off("pooled")}, {cov2, 0}}, 640u * 32u); outbytes.push_back({"pooled", off("pooled")}); } }
    else if (kernel == "qkv_attend") { o_role = "out"; o_off = off("out"); o_bytes = TIN;
      if (sptp) {
        struct SpPrepParams { uint32_t steps, cols, step_bytes, reserved; } pp{16u, 1536u, 49152u, 0u};
        { id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep_generic")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:qw[m] offset:0 atIndex:1 + m]; [enc setBuffer:qwmax offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
          [enc dispatchThreads:MTLSizeMake(16u * 1536u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_qkv_tp")];
        [enc setBuffer:pool offset:off("x") atIndex:0]; [enc setBuffer:weights offset:0 atIndex:1]; [enc setBuffer:qb offset:0 atIndex:2]; [enc setBuffer:kb offset:0 atIndex:3]; [enc setBuffer:vb offset:0 atIndex:4]; [enc setBuffer:rsq offset:0 atIndex:5]; [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        for (int m = 0; m < 3; ++m) [enc setBuffer:qw[m] offset:0 atIndex:7 + m]; [enc setBuffer:qwmax offset:0 atIndex:10]; [enc setBuffer:spstats offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(16, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
        // windowed attention on the engine: per-(window, head) K^T / V operands from the live q/k/v, then the attention tile kernel
        enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_attend_tp_prep")]; [enc setBuffer:kb offset:0 atIndex:0]; [enc setBuffer:vb offset:0 atIndex:1]; [enc setBytes:&prm length:sizeof(prm) atIndex:2];
        for (int m = 0; m < 3; ++m) [enc setBuffer:atk[m] offset:0 atIndex:3 + m]; [enc setBuffer:atkmax offset:0 atIndex:6]; for (int m = 0; m < 3; ++m) [enc setBuffer:atv[m] offset:0 atIndex:7 + m]; [enc setBuffer:atvmax offset:0 atIndex:10];
        [enc dispatchThreads:MTLSizeMake(40u * 16u * 64u + 40u * 16u * 2u * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
        enc = compute(); [enc setComputePipelineState:pso(sptp, "nr_split_attend_tp")];
        [enc setBuffer:qb offset:0 atIndex:0]; [enc setBuffer:kb offset:0 atIndex:1]; [enc setBuffer:vb offset:0 atIndex:2]; [enc setBuffer:weights offset:0 atIndex:3]; [enc setBuffer:pool offset:o_off atIndex:4]; [enc setBuffer:cov offset:0 atIndex:5]; [enc setBuffer:rcp offset:0 atIndex:6]; [enc setBytes:&prm length:sizeof(prm) atIndex:7];
        for (int m = 0; m < 3; ++m) [enc setBuffer:atk[m] offset:0 atIndex:8 + m]; [enc setBuffer:atkmax offset:0 atIndex:11]; for (int m = 0; m < 3; ++m) [enc setBuffer:atv[m] offset:0 atIndex:12 + m]; [enc setBuffer:atvmax offset:0 atIndex:15]; [enc setBuffer:spstats offset:0 atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake(16, 40 * 4, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
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
  for (NSDictionary* L in chain[@"split_a"]) run_split(L);
  // ---- repack 58 (live input: the final-head output with padding tokens zeroed by the driver's rule, both on the GPU) ----
  NSDictionary* R = chain[@"vit"]; const size_t vit_in_off = size_t(JInt(R, "final_head_offset"));
  id<MTLBuffer> vin = scratch(720896, 0); id<MTLBuffer> tok = scratch(VITB, uint8_t(poison)), rcov = scratch(VITB * 4, 0);
  { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:vin range:NSMakeRange(0, 720896 + 16) value:0]; [bl copyFromBuffer:pool sourceOffset:vit_in_off toBuffer:vin destinationOffset:0 size:VITB]; [bl endEncoding];
    const uint32_t nmask = uint32_t(VITB); id<MTLComputeCommandEncoder> e = compute(); [e setComputePipelineState:psoPadMask]; [e setBuffer:vin offset:0 atIndex:0]; [e setBuffer:padmask offset:0 atIndex:1]; [e setBytes:&nmask length:4 atIndex:2]; [e dispatchThreads:MTLSizeMake(VITB, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding];
    launches.push_back({58, "repack_input", {{"repack58_input", digest("repack 58 input (padding zeroed)", 58u, vin, 0, 720896)}}, {}, ""}); }
  auto run_repack = [&](const char* name, long n, id<MTLBuffer> inb, size_t in_bytes, id<MTLBuffer> outb, size_t out_off, size_t out_bytes) {
    RepackParams prm{20, 32, uint32_t(in_bytes / 4), uint32_t(out_bytes / 4)};
    fill(rcov, 0, VITB * 4 + 16, 0);
    id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(vlib, name)]; [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:out_off atIndex:1]; [enc setBuffer:rcov offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof(prm) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "repack-coverage" && n == 58) poke(rcov, 28, 0);   // dword 7 never written
#endif
    LaunchRec lr{n, name, {}, {}, ""}; lr.validators.push_back(validate(std::string("repack ") + std::to_string(n), uint32_t(n), rcov, out_bytes / 4, 0u, nil)); lr.digests.push_back({name, digest(std::string("repack ") + std::to_string(n), uint32_t(n), outb, out_off, out_bytes)}); launches.push_back(lr);
  };
  run_repack("nr_vit_repack_2d_to_1d", 58, vin, 720896, tok, 0, VITB);
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
    { id<MTLBlitCommandEncoder> bl = blit(); for (id<MTLBuffer> c : {covE, covC, covQ, covK, covV, covA, covP}) [bl fillBuffer:c range:NSMakeRange(0, c.length) value:0]; [bl endEncoding]; }
    auto dispatch = [&](const char* name, std::vector<id<MTLBuffer>> bufs, size_t threads, uint32_t stage) {
      id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(vlib, name)];
      for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
    };
    if (tplib) {
      auto prep = [&](int st, id<MTLBuffer> W) { TpPrepParams pp{tp_steps[st], tp_cols[st], tp_stepb[st], 0u}; id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(tplib, "nr_vit_tp_prep")];
        [enc setBuffer:W offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:tpw[st][m] offset:0 atIndex:1 + m]; [enc setBuffer:tpmax[st] offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(size_t(tp_steps[st]) * tp_cols[st], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; };
      auto tpdispatch = [&](const char* name, int st, std::vector<id<MTLBuffer>> bufs, uint32_t stage) {   // bufs: the accepted kernel's buffers; then params, wl, wh, wlh, maxcls, stats
        id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(tplib, name)];
        for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; for (int m = 0; m < 3; ++m) [enc setBuffer:tpw[st][m] offset:0 atIndex:bufs.size() + 1 + m]; [enc setBuffer:tpmax[st] offset:0 atIndex:bufs.size() + 4]; [enc setBuffer:tpstats offset:0 atIndex:bufs.size() + 5];
        [enc dispatchThreadgroups:MTLSizeMake(tp_cols[st] / 64, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; };
      prep(0, W0); tpdispatch("nr_vit_expand_tp", 0, {tokens, W0, expand, covE}, 1); prep(1, W1); tpdispatch("nr_vit_contract_tp", 1, {expand, tokens, W1, contract, covC}, 2);
      { TpPrepParams pp{32u, 3072u, 98304u, 0u}; id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(tplib, "nr_vit_tp_prep")];
        [enc setBuffer:W2 offset:128 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:tqw[m] offset:0 atIndex:1 + m]; [enc setBuffer:tqmax offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(32u * 3072u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
        enc = compute(); [enc setComputePipelineState:pso(tplib, "nr_vit_qkv_tp")]; std::vector<id<MTLBuffer>> qbv{contract, W2, q, k, v, covQ, covK, covV, rsq};
        for (size_t i = 0; i < qbv.size(); ++i) [enc setBuffer:qbv[i] offset:0 atIndex:i]; prm.stage = 3; [enc setBytes:&prm length:sizeof(prm) atIndex:9]; for (int m = 0; m < 3; ++m) [enc setBuffer:tqw[m] offset:0 atIndex:10 + m]; [enc setBuffer:tqmax offset:0 atIndex:13]; [enc setBuffer:tpstats offset:0 atIndex:14];
        [enc dispatchThreadgroups:MTLSizeMake(32, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; }
      { id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:pso(tplib, "nr_vit_attn_prep")]; [enc setBuffer:k offset:0 atIndex:0]; [enc setBuffer:v offset:0 atIndex:1];
        for (int m = 0; m < 3; ++m) [enc setBuffer:tak[m] offset:0 atIndex:2 + m]; [enc setBuffer:takmax offset:0 atIndex:5]; for (int m = 0; m < 3; ++m) [enc setBuffer:tav[m] offset:0 atIndex:6 + m]; [enc setBuffer:tavmax offset:0 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(32u * 640u + 32u * 20u * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
        enc = compute(); [enc setComputePipelineState:pso(tplib, "nr_vit_attention_tp")]; std::vector<id<MTLBuffer>> abf{q, k, v, attention, covA, rcp};
        for (size_t i = 0; i < abf.size(); ++i) [enc setBuffer:abf[i] offset:0 atIndex:i]; prm.stage = 4; [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        for (int m = 0; m < 3; ++m) [enc setBuffer:tak[m] offset:0 atIndex:7 + m]; [enc setBuffer:takmax offset:0 atIndex:10]; for (int m = 0; m < 3; ++m) [enc setBuffer:tav[m] offset:0 atIndex:11 + m]; [enc setBuffer:tavmax offset:0 atIndex:14]; [enc setBuffer:tpstats offset:0 atIndex:15];
        [enc dispatchThreadgroups:MTLSizeMake(32, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; }
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
  // ---- repack 99 -> pool (launch 100 input), then split launches 100-132 ----
  run_repack("nr_vit_repack_1d_to_2d", 99, tokens, VITB, pool, size_t(JInt(R, "repack99_out_offset")), VITB);
  for (NSDictionary* L in chain[@"split_b"]) run_split(L);
  // ---- middle view: kept in a runtime buffer (CPU SHA-256 at completion = the accepted middle_view_sha256) and digested in place ----
  const size_t fo = size_t(JInt(chain, "final_offset")); id<MTLBuffer> midkeep = scratch(VIEW, 0);
  { id<MTLBlitCommandEncoder> bl = blit(); [bl copyFromBuffer:pool sourceOffset:fo toBuffer:midkeep destinationOffset:0 size:VIEW]; [bl endEncoding]; launches.push_back({132, "middle_view", {{"middle_view", digest("middle view (launch 132 output)", 132u, pool, fo, VIEW, nullptr, true, false)}}, {}, ""}); }
  const uint32_t mid_digest_slot = uint32_t(dexp.size() - 1);
  // ---- decoder/output section (launches 133-156) on the same resident pool, restaged on the GPU ----
  NSDictionary* D = chain[@"decoder"]; if (!D) throw std::runtime_error("v12 requires the decoder section (resident frame = middle + decoder)");
  const bool final_only = in.production; const fs::path dtables = JStr(D, "tables");
  const unsigned tw = unsigned(JInt(D, "tex_w")), th = unsigned(JInt(D, "tex_h")); require(in.input_color != nil && in.input_color.pixelFormat == MTLPixelFormatRGBA16Float && in.input_color.width == tw && in.input_color.height == th, "input colour texture (RGBA16F WxH)");
  const size_t texb = size_t(tw) * th * 8, motb = size_t(tw) * th * 4;
  std::vector<uint8_t> texin(texb); [in.input_color getBytes:texin.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
  const std::string texin_sha = Sha(texin.data(), texin.size()); res.input_texture_sha256_before = texin_sha;
  std::vector<uint8_t> hist_before, mot_before;
  if (!in.reset) { mot_before.resize(motb); [in.motion getBytes:mot_before.data() bytesPerRow:tw * 4 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; res.motion_sha256 = Sha(mot_before.data(), mot_before.size());
    hist_before.resize(texb); [in.temporal->history getBytes:hist_before.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; require(Sha(hist_before.data(), hist_before.size()) == in.temporal->history_sha256, "history texture content does not match the recorded history hash"); res.history_sha256_before = in.temporal->history_sha256; }
  // restage: poison the whole pool, put this run's middle view back, apply the live-producer patches, set the control words - all in-buffer
  NSArray* patches = D[@"patches"]; require(patches && patches.count == 5, "descriptor patch table (4 skips + Block0 skip)"); const NrBodyRegion* regs[5] = {&in.skips[0], &in.skips[1], &in.skips[2], &in.skips[3], &in.block0_skip};
  NSArray* ctrl = D[@"control"]; const size_t c0 = [ctrl[0] unsignedLongValue], c1 = [ctrl[1] unsignedLongValue];
  { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:pool range:NSMakeRange(0, pool_bytes) value:uint8_t(poison)]; [bl copyFromBuffer:midkeep sourceOffset:0 toBuffer:pool destinationOffset:fo size:VIEW];
    for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs[i];
      require(r.buffer != nil && r.bytes == bytes && r.offset + bytes <= r.buffer.length && off + bytes <= pool_bytes, JStr(P, "file").c_str()); [bl copyFromBuffer:r.buffer sourceOffset:r.offset toBuffer:pool destinationOffset:off size:bytes]; }
    [bl fillBuffer:pool range:NSMakeRange(c0, c1 - c0) value:0xff]; [bl endEncoding]; }
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
  for (NSDictionary* L in D[@"launches"]) {
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
    id<MTLComputePipelineState> dps = pso(lib(libp), dl.kernel.c_str());
    if (n == 155u && in.reset) { require(dl.params_size >= 104, "post block parameter block"); for (size_t i = 88; i < 104; ++i) require(params[i] == 0, "temporal inputs present (launch 155 params +88/+96): unsupported path"); }   // see docs/NR_INPUT_CONTRACT.md
    dl.p_sha = Sha(dl.bprm.contents, dl.params_size);   // immutability of the (possibly per-frame) record: re-hashed at completion
    if (dl.temporal_launch) { tex_to_buf(in.temporal->history, histpre, 8); histpre_slot = digest("history before 155", 155u, histpre, 0, texb, hist_before.data(), true, false); tex_to_buf(in.motion, motpre, 4); motpre_slot = digest("motion before 155", 155u, motpre, 0, motb, mot_before.data(), true, false); }
    { id<MTLBlitCommandEncoder> bl = blit(); [bl fillBuffer:bcov range:NSMakeRange(0, pool_bytes + 4096) value:0]; if (!dl.texmode.empty()) [bl fillBuffer:btexcov range:NSMakeRange(0, size_t(tw) * th * 4 + 16) value:0]; [bl endEncoding]; }
    id<MTLComputeCommandEncoder> enc = compute(); [enc setComputePipelineState:dps];
    [enc setBuffer:bpool offset:0 atIndex:0]; [enc setBuffer:dl.bw offset:0 atIndex:1]; [enc setBuffer:dl.bprm offset:0 atIndex:2]; [enc setBuffer:brsq offset:0 atIndex:3]; [enc setBuffer:brcp offset:0 atIndex:4]; [enc setBuffer:bdbg offset:0 atIndex:5]; [enc setBuffer:bcov offset:0 atIndex:6];
    [enc setBuffer:brom offset:0 atIndex:7]; [enc setBuffer:bex2u offset:0 atIndex:8]; [enc setBuffer:bex2f offset:0 atIndex:9]; [enc setBuffer:btexcov offset:0 atIndex:10];
    if (dl.temporal_launch) { [enc setBuffer:bex2h offset:0 atIndex:11]; [enc setBuffer:dl.bfault offset:0 atIndex:12]; }
    if (dl.texmode == "post") { [enc setTexture:tex[0] atIndex:0]; [enc setTexture:in.temporal ? tex[2] : tex[1] atIndex:1]; if (dl.temporal_launch) { [enc setTexture:in.motion atIndex:2]; [enc setTexture:in.temporal->history atIndex:3]; } }
    else if (dl.texmode == "copy") { if (in.temporal) { [enc setTexture:tex[2] atIndex:0]; [enc setTexture:in.temporal->history atIndex:1]; } else { [enc setTexture:tex[1] atIndex:0]; [enc setTexture:tex[2] atIndex:1]; } }
    [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntValue], [g[1] unsignedIntValue], [g[2] unsignedIntValue]) threadsPerThreadgroup:MTLSizeMake([b[0] unsignedIntValue], [b[1] unsignedIntValue], [b[2] unsignedIntValue])]; [enc endEncoding];
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
  launches.push_back({156, "pool_final", {{"pool_final", digest("final pool", 156u, pool, 0, pool_bytes, nullptr, true, false)}}, {}, ""}); const uint32_t final_pool_slot = uint32_t(dexp.size() - 1);
  // ---- ENCODED: the buffer is complete; the completion step closes over every encode-time value and runs after nr_body_commit + wait ----
  p.cmd = cmd; p.frame_id = ++ctx.frame_serial; ctx.in_flight_id = p.frame_id; ctx.in_flight = true; p.state = NrBodyPending::Encoded;   // v16: this handle owns frame serial frame_id
  p.complete = [=, &ctx](NrBodyResult& res, double commit_wall_ms) mutable -> int {   // mutable: the captured expectation tables receive the completion hashes
    std::vector<std::string>& fails = res.check_failures; auto check = [&](bool ok, const std::string& what) { ++res.checks_enforced; if (!ok) fails.push_back(what); };
    struct { bool seen = false, ok = false; std::string hist, out; } pend;   // v10: deferred temporal commit, applied only at the end of a clean frame
    auto mark_damage = [&](const std::string& why) { damage(ctx, in.temporal, res, why); };
    const NrBodyRegion* regs[5] = {&in.skips[0], &in.skips[1], &in.skips[2], &in.skips[3], &in.block0_skip};   // v16: pointers into THIS closure's copy of the inputs, not the caller's struct
    try {
  // ---- COMPLETION: every deferred check on bytes that are still intact; nothing here changes GPU state ----
  const StickyRecord* st = static_cast<const StickyRecord*>(sticky.contents); const ValidatorRecord* vr = static_cast<const ValidatorRecord*>(vrec.contents); const DigestRecord* dr = static_cast<const DigestRecord*>(drec.contents);
  res.validators_run = st->validators_run; res.coverage_faults = st->fault_count;
  check(st->validators_run == vexp.size(), "coverage validators: " + std::to_string(st->validators_run) + " ran, " + std::to_string(vexp.size()) + " expected");
  for (size_t i = 0; i < vexp.size(); ++i) { const ValidatorRecord& r = vr[i]; const bool complete = r.done == VALIDATOR_DONE && r.launch_id == vexp[i].launch_id && r.elements == vexp[i].elements;
    check(complete, vexp[i].label + ": validator record incomplete or mismatched (done=" + Hex64(0, r.done).substr(8) + ", launch=" + std::to_string(r.launch_id) + ", elements=" + std::to_string(r.elements) + " expected " + std::to_string(vexp[i].elements) + ")");
    if (complete) check(r.faults == 0, vexp[i].label + ": " + std::to_string(r.faults) + " coverage faults (exactly-once / padding / at-most-once contract)" + (r.faults ? "; first failing element " + std::to_string(r.first_elem) + " observed " + std::to_string(r.first_obs) + " writes" : std::string())); }
  if (st->fault_flag) { std::string first = "none"; for (size_t i = 0; i < vexp.size(); ++i) if (vr[i].done == VALIDATOR_DONE && vr[i].faults) { first = vexp[i].label + " element " + std::to_string(vr[i].first_elem) + " observed " + std::to_string(vr[i].first_obs); break; } check(false, "coverage first failure: " + first + " (" + std::to_string(st->fault_count) + " faulting elements in the frame)"); }
  std::ostringstream dtab; for (size_t i = 0; i < dexp.size(); ++i) { const DigestRecord& r = dr[i]; const bool complete = r.done == DIGEST_DONE && r.launch_id == dexp[i].launch_id && r.words == dexp[i].words; if (complete) ++res.digests_complete;
    check(complete, dexp[i].label + ": checkpoint digest record incomplete or mismatched"); dtab << i << " " << dexp[i].label << " " << r.words << " " << Hex64(r.hi, r.lo) << "\n";
    if (dexp[i].snap) { dexp[i].sha = Sha(dexp[i].snap.contents, size_t(dexp[i].words) * 4); ++res.digest_cpu_crosschecks; const NrDigest64 c = nr_digest64_cpu(static_cast<const uint8_t*>(dexp[i].snap.contents), size_t(dexp[i].words) * 4); const bool same = complete && c.lo == r.lo && c.hi == r.hi; if (!same) ++res.digest_cpu_mismatches; check(same, dexp[i].label + ": GPU digest " + Hex64(r.hi, r.lo) + " != CPU digest of the in-buffer snapshot " + Hex64(c.hi, c.lo)); }
    if (dexp[i].has_cpu) { ++res.digest_cpu_crosschecks; const bool same = complete && r.lo == dexp[i].cpu.lo && r.hi == dexp[i].cpu.hi; if (!same) ++res.digest_cpu_mismatches; check(same, dexp[i].label + ": GPU digest " + Hex64(r.hi, r.lo) + " != CPU digest " + Hex64(dexp[i].cpu.hi, dexp[i].cpu.lo)); } }
  auto digest_at = [&](uint32_t slot) { return NrDigest64{dr[slot].lo, dr[slot].hi}; };
  auto crosscheck = [&](uint32_t slot, const uint8_t* p, size_t bytes, const std::string& what) { ++res.digest_cpu_crosschecks; const NrDigest64 c = nr_digest64_cpu(p, bytes), g = digest_at(slot); if (!(c == g)) ++res.digest_cpu_mismatches; check(c == g, what + ": GPU digest " + Hex64(g.hi, g.lo) + " != CPU digest " + Hex64(c.hi, c.lo)); };
  // middle view (accepted receipt hash) and its digest cross-check
  res.middle_view_sha256 = Sha(midkeep.contents, VIEW); crosscheck(mid_digest_slot, static_cast<const uint8_t*>(midkeep.contents), VIEW, "middle view"); if (diag) std::ofstream(out / "out_view_132.bin", std::ios::binary).write(static_cast<const char*>(midkeep.contents), VIEW);
  // decoder pre-image: constructed on the CPU from the same inputs (poison, middle view, caller skip regions, control words); SHA-256 is the accepted receipt hash, the digest proves the GPU restage produced the same bytes
  { std::vector<uint8_t> img(pool_bytes, uint8_t(poison)); std::memcpy(img.data() + fo, midkeep.contents, VIEW); bool all_cpu = true;
    for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs[i]; if (r.buffer.contents) std::memcpy(img.data() + off, static_cast<const uint8_t*>(r.buffer.contents) + r.offset, bytes); else all_cpu = false; }
    std::memset(img.data() + c0, 0xff, c1 - c0);
    std::string expected_sha; if (all_cpu) { expected_sha = Sha(img.data(), img.size()); crosscheck(pre_digest_slot, img.data(), img.size(), "decoder pre-image (expected reconstruction vs GPU digest)"); } else res.checks_skipped.push_back("decoder pre-image CPU construction (a skip region is not CPU-visible)");
    res.decoder_pre_image_expected_sha256 = expected_sha;
    if (dexp[pre_digest_slot].snap) {   // v13 checked mode: the ACTUAL restaged pool bytes (GPU snapshot before launch 133) are the identity; the reconstruction is the expectation
      const std::string actual = Sha(dexp[pre_digest_slot].snap.contents, pool_bytes); res.decoder_pre_image_sha256 = actual; res.decoder_pre_image_source = "gpu-snapshot";
      if (all_cpu) check(actual == expected_sha, "decoder pre-image: actual restaged pool (GPU snapshot) " + actual.substr(0, 16) + " != expected reconstruction " + expected_sha.substr(0, 16)); }
    else { res.decoder_pre_image_sha256 = all_cpu ? expected_sha : "digest:" + Hex64(dr[pre_digest_slot].hi, dr[pre_digest_slot].lo); res.decoder_pre_image_source = all_cpu ? "cpu-reconstruction (+gpu-digest-crosscheck)" : "gpu-digest-only"; } }
  // decoder launches: constants (final re-hash below), constructed record, EX2 faults, textures
  std::vector<uint8_t> texout(texb), texback(texb); std::string tex155_out_sha;
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
      id<MTLTexture> to = in.temporal ? tex[2] : tex[1]; [to getBytes:texout.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; tex155_out_sha = Sha(texout.data(), texout.size()); crosscheck(tex155_slot, texout.data(), texb, "launch 155 output texture");
      if (diag) std::ofstream(out / "tex_out_155.bin", std::ios::binary).write(reinterpret_cast<const char*>(texout.data()), std::streamsize(texout.size())); }
    if (dl.texmode == "copy") {
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
        pend.seen = true; pend.ok = (hdiff == 0 && osha == hsha); pend.hist = hsha; pend.out = osha; } }   // v10: commit deferred to the end of the frame
  }
  [tex[0] getBytes:texback.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; res.input_texture_sha256_after = Sha(texback.data(), texback.size()); res.input_texture_immutable = res.input_texture_sha256_after == texin_sha; check(res.input_texture_immutable, "input colour texture changed during the frame");
  res.pool_final_sha256 = Sha(pool.contents, pool_bytes); crosscheck(final_pool_slot, static_cast<const uint8_t*>(pool.contents), pool_bytes, "final pool");
  if (final_only) res.checks_skipped = {"decoder per-launch pool digest", "decoder per-launch pool coverage validator", "intermediate SHA-256 snapshots (checked mode only; GPU digests + boundary SHA-256 in production)", "decoder pre-image actual GPU snapshot SHA-256 (production reports the CPU reconstruction with the GPU digest cross-check)"};
  if (diag && JStr(D, "write_pool_post") == "true") std::ofstream(out / "pool_post.bin", std::ios::binary).write(static_cast<const char*>(pool.contents), std::streamsize(pool_bytes));   // v14: explicit request only
  res.digest_table_sha256 = Sha(dtab.str().data(), dtab.str().size());
  const double wall_total = -[t_all timeIntervalSinceNow] * 1000.0; res.middle_gpu_ms = 0; res.decoder_gpu_ms = 0;   // v14: res.wall_ms is set after the final re-hash/commit below   // one command buffer: per-section GPU time is not separable; frame_gpu_ms carries the whole frame
  // ---- diagnostics JSON (launch list with validator/digest slots, the digest table, evidence summary) ----
  if (diag) { std::ostringstream js; js << "{\n \"schema\": \"neuralstudio.nr.body-runtime.v16-resident-experimental\",\n \"mode\": \"" << (checked ? "checked (in-buffer snapshots, SHA-256 per intermediate)" : "production (digests per intermediate, SHA-256 at the frame boundaries)") << "\",\n \"pid\": " << getpid() << ",\n \"poison\": " << poison << ",\n \"command_buffers\": " << res.inventory_command_buffers << ",\n \"frame_gpu_ms\": " << res.frame_gpu_ms << ",\n \"commit_to_completion_wall_ms\": " << commit_wall_ms << ",\n \"launches\": [\n"; bool first = true;
    for (auto& l : launches) { js << (first ? "" : ",\n") << "  {\"launch\": " << l.n << ", \"kind\": \"" << l.kind << "\", \"digests\": {"; bool f2 = true; for (auto& d : l.digests) { const DigestRecord& r = dr[d.second]; js << (f2 ? "" : ", ") << "\"" << d.first << "\": \"" << Hex64(r.hi, r.lo) << "\""; if (dexp[d.second].snap) js << ", \"" << d.first << "_sha256\": \"" << dexp[d.second].sha << "\""; f2 = false; } js << "}, \"validators\": ["; f2 = true; for (auto s : l.validators) { const ValidatorRecord& r = vr[s]; js << (f2 ? "" : ", ") << "{\"slot\": " << s << ", \"elements\": " << r.elements << ", \"faults\": " << r.faults << ", \"done\": " << (r.done == VALIDATOR_DONE ? "true" : "false") << "}"; f2 = false; } js << "]}"; first = false; }
    js << "\n ],\n \"digest_table\": [\n"; for (size_t i = 0; i < dexp.size(); ++i) { const DigestRecord& r = dr[i]; js << (i ? ",\n" : "") << "  {\"slot\": " << i << ", \"label\": \"" << dexp[i].label << "\", \"launch\": " << dexp[i].launch_id << ", \"words\": " << r.words << ", \"digest\": \"" << Hex64(r.hi, r.lo) << "\", \"done\": " << (r.done == DIGEST_DONE ? "true" : "false") << (dexp[i].has_cpu ? ", \"cpu\": \"" + Hex64(dexp[i].cpu.hi, dexp[i].cpu.lo) + "\"" : std::string()) << (dexp[i].snap ? ", \"sha256\": \"" + dexp[i].sha + "\"" : std::string()) << "}"; }
    js << "\n ],\n \"digest_table_sha256\": \"" << res.digest_table_sha256 << "\",\n \"validators\": {\"expected\": " << vexp.size() << ", \"run\": " << st->validators_run << ", \"faults\": " << st->fault_count << "},\n \"middle_view_sha256\": \"" << res.middle_view_sha256 << "\",\n \"decoder_pre_image_sha256\": \"" << res.decoder_pre_image_sha256 << "\",\n \"decoder_pre_image_source\": \"" << res.decoder_pre_image_source << "\",\n \"decoder_pre_image_expected_sha256\": \"" << res.decoder_pre_image_expected_sha256 << "\",\n \"pool_final_sha256\": \"" << res.pool_final_sha256 << "\",\n \"final_surface_sha256\": \"" << res.final_surface_sha256 << "\",\n \"tex155_output_sha256\": \"" << tex155_out_sha << "\",\n \"wall_ms_total\": " << wall_total << "\n}\n";
    std::ofstream(out / "chain-run.json") << js.str(); }
  std::cout << "frame gpu_ms " << res.frame_gpu_ms << " wall_ms " << wall_total << " validators " << st->validators_run << "/" << vexp.size() << " digests " << res.digests_complete << "/" << dexp.size() << "\n";
  // v3 provenance: report every library/constant identity recorded at load and re-hash every resident constant now (post-frame)
#ifdef NR_BODY_FAULT_TESTS
  if (fault == "constant-late-mutation") for (auto& kv : ctx.resident) if (kv.first.rfind("file:", 0) == 0) { static_cast<uint8_t*>(kv.second.contents)[0] ^= 1u; break; }   // after the last use of every constant: only the final re-hash can see it
  if (fault == "decoder-weights") for (auto& dl : decl) if (dl.n == 133u) { static_cast<uint8_t*>(dl.bw.contents)[5] ^= 0x01; break; }   // launch-133 weights flipped after completion: pixels unaffected, the resident re-hash must trip
#endif
  res.constants_unchanged_after_frame = true;
  for (auto& kv : ctx.resident_sha256) { res.provenance.push_back(kv); if (kv.first.rfind("file:", 0) == 0) { auto bi = ctx.resident.find(kv.first); auto ni = ctx.resident_bytes.find(kv.first); if (bi != ctx.resident.end() && ni != ctx.resident_bytes.end()) { ++res.constants_rehashed; if (Sha(bi->second.contents, ni->second) != kv.second) { res.constants_unchanged_after_frame = false; ++res.decoder_constant_mutations; fails.push_back("model constant changed since load: " + kv.first); } } } }
  if (in.temporal && pend.seen) {   // v10: the ONLY place the temporal metadata changes: after every check of this frame, and only if none failed
    res.temporal_state_advanced = fails.empty() && pend.ok;
    if (res.temporal_state_advanced) { in.temporal->history_sha256 = pend.hist; in.temporal->last_output_sha256 = pend.out; in.temporal->frames_since_reset = in.reset ? 1u : in.temporal->frames_since_reset + 1u; } }
  res.wall_ms = -[t_all timeIntervalSinceNow] * 1000.0;   // v14: includes completion checks, the constant re-hash and the temporal commit decision
  if (!fails.empty()) { std::ostringstream e; e << fails.size() << " computed check(s) failed: "; for (size_t i = 0; i < fails.size() && i < 6; ++i) e << (i ? "; " : "") << fails[i]; if (fails.size() > 6) e << "; ..."; res.code = 3; res.error = e.str(); mark_damage("code 3: " + fails[0]); return 3; }
  res.code = 0; res.error.clear(); return 0;
    } catch (const std::exception& e) { res.code = 2; res.error = e.what(); mark_damage(std::string("exception after commit: ") + e.what()); return 2; }
  };
  return 0;
  } catch (const std::exception& e) { res.code = 2; res.error = e.what(); p.state = NrBodyPending::Failed; return 2; } } }
int nr_body_commit(NrBodyPending& p) {
  if (p.state == NrBodyPending::Moved) return misuse(p, "commit on a moved-from pending handle");
  if (p.state != NrBodyPending::Encoded) return misuse(p, p.state == NrBodyPending::Committed ? "commit on an already committed frame" : "commit on a frame that is not encoded");
  if (!p.ctx || p.ctx->in_flight_id != p.frame_id || p.frame_id == 0) return misuse(p, "stale pending handle: this handle does not own the context's in-flight frame");
  p.t_commit = [[NSDate date] timeIntervalSince1970]; [p.cmd commit]; p.state = NrBodyPending::Committed; return 0; }
int nr_body_complete(NrBodyPending& p, NrBodyResult& res) { @autoreleasepool {
  if (p.state == NrBodyPending::Moved) { const int rc = misuse(p, "complete on a moved-from pending handle"); res = p.res; return rc; }
  if (p.state != NrBodyPending::Committed) { const int rc = misuse(p, p.state == NrBodyPending::Encoded ? "complete before commit" : "complete on a frame that is not committed (idle, completed or failed)"); res = p.res; return rc; }
  if (!p.ctx || p.ctx->in_flight_id != p.frame_id || p.frame_id == 0 || !p.cmd || !p.complete) { const int rc = misuse(p, "stale pending handle: this handle does not own the context's in-flight frame"); res = p.res; return rc; }
  ++p.res.inventory_waits; [p.cmd waitUntilCompleted]; const double commit_wall_ms = ([[NSDate date] timeIntervalSince1970] - p.t_commit) * 1000.0; int rc;
  if (p.cmd.status != MTLCommandBufferStatusCompleted) { p.res.code = 2; p.res.error = p.cmd.error ? p.cmd.error.localizedDescription.UTF8String : "command buffer failed"; damage(*p.ctx, p.temporal, p.res, "command buffer failed: " + p.res.error); rc = 2; }
  else { p.res.frame_gpu_ms = (p.cmd.GPUEndTime - p.cmd.GPUStartTime) * 1000.0; rc = p.complete(p.res, commit_wall_ms); }
  p.ctx->in_flight = false; p.ctx->in_flight_id = 0; p.state = rc == 0 ? NrBodyPending::Completed : NrBodyPending::Failed; p.complete = nullptr; p.cmd = nil; p.frame_id = 0; res = p.res; return rc; } }
void nr_body_pending_reset(NrBodyPending& p) { if (p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) return; p.state = NrBodyPending::Idle; p.cmd = nil; p.complete = nullptr; p.ctx = nullptr; p.temporal = nullptr; p.res = NrBodyResult{}; p.t_commit = 0; p.frame_id = 0; }   // an in-flight pending cannot be reset
void nr_body_pending_abandon(NrBodyPending& p) {   // v16: destructor / move-assignment target; bounded abandonment semantics
  if (p.state == NrBodyPending::Encoded) { if (p.ctx && p.ctx->in_flight_id == p.frame_id) { p.ctx->in_flight = false; p.ctx->in_flight_id = 0; } }   // never submitted: release the reservation, no damage
  else if (p.state == NrBodyPending::Committed) { if (p.cmd) [p.cmd waitUntilCompleted]; if (p.ctx) { damage(*p.ctx, p.temporal, p.res, "frame abandoned after commit without completion (checks never ran)"); if (p.ctx->in_flight_id == p.frame_id) { p.ctx->in_flight = false; p.ctx->in_flight_id = 0; } } }
  p.state = (p.state == NrBodyPending::Encoded || p.state == NrBodyPending::Committed) ? NrBodyPending::Failed : p.state; p.cmd = nil; p.complete = nullptr; p.frame_id = 0; }
int nr_body_run_frame(NrBodyContext& ctx, NSDictionary* chain, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyResult& res) {
  NrBodyPending p; int rc = nr_body_encode(ctx, chain, in, diagnostics_dir, p); if (rc) { res = p.res; return rc; } rc = nr_body_commit(p); if (rc) { res = p.res; return rc; } return nr_body_complete(p, res); }
