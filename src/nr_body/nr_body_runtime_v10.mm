// v10 (2026-09-06, Codex's late-failure finding): the temporal metadata commit is DEFERRED to the very end of the frame, after every per-launch check and the
// final resident-constant re-hash; it happens only when the failure list is empty at that point. Fault-test builds add constant-late-mutation: one resident
// constant byte is flipped after its last use, so only the final re-hash can catch it - the frame must return code 3 without advancing the state.
// v9 (2026-09-06, Codex's failure-state concern): the temporal state advances ONLY when no check has failed so far in this frame AND the actual history equals
// the caller output; res.temporal_state_advanced records it. Fault-test builds add two REAL GPU-path faults: ex2-table-row-corrupt (one table row's argument
// word is corrupted in the bound buffer after the hash check, so the kernel's lookup itself counts a fault) and history-texel-corrupt (one history texel is
// overwritten after copy 156 and before the readback, so the independent byte comparison itself fails).
// v8 (2026-09-06, after Codex's 13:54 audit): the copy-156 branch keeps the ACTUAL history readback and the caller output as separate byte vectors and compares
// their independently computed hashes BEFORE advancing the temporal state (mismatch = check failure, state not advanced); v6/v7 swapped the vectors first and
// compared the caller output with itself. Under a temporal state launch 156 uses the descriptor's lib_temporal (copy built with the exact-texel read so signed
// zeros survive), reset-only production keeps lib.
// v7 (2026-09-06): as v6, but the half-domain EX2 table is validated in the UP-FRONT block (before any GPU work; negative control temporal-ex2-table-wrong-sha rejects there).
// v6 (2026-09-06): temporal launch 155 binds the audited half-domain EX2 table (descriptor launch field ex2_half_table {file, sha256}; size 524288 and the pinned sha
// are verified at load, provenance recorded) at buffer 11 and a zeroed fault counter at buffer 12; after the launch fault[0] (argument mismatch) and fault[1]
// (nonfinite half) must be zero (code 3). Reset frames are unchanged (the production kernels declare neither buffer).
// nr_body_runtime_v1.mm — execution core (see nr_body_runtime.h). Derived from the verified fused probe; kernels, dispatch order, poison,
// coverage and hashing are unchanged; only the input side changed: live inputs arrive as caller buffers/textures, never as files.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_body_runtime.h"
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
static std::string JStr(NSDictionary* d, const char* k) { NSString* v = d[[NSString stringWithUTF8String:k]]; return v ? std::string(v.UTF8String) : std::string(); }
static long JInt(NSDictionary* d, const char* k) { NSNumber* v = d[[NSString stringWithUTF8String:k]]; return v ? v.longValue : 0; }
struct SplitParams { int32_t shift_x, shift_y; uint32_t flags, tokens; }; enum : uint32_t { FLAG_RES_VIEW = 1, FLAG_OUT_VIEW = 2, FLAG_EMIT_F16 = 4, FLAG_IN_VIEW = 8 };
struct VitParams { uint32_t tokens, block, stage, reserved; }; struct RepackParams { int width, height; uint32_t in_dwords, out_dwords; };
constexpr size_t TIN = 2160u * 512u, VIEW = 16u * 69120u, POOLB = 640u * 512u, VITB = 640u * 1024u, T = 640;
// Persistent context for serve mode: device, queue, libraries and pipelines survive across frames (the per-process fixed cost is paid once).
static id<MTLBuffer> resident_buf(NrBodyContext& ctx, id<MTLDevice> dev, const std::string& key, size_t length) { auto it = ctx.resident.find(key); if (it != ctx.resident.end() && it->second.length == length) return it->second; id<MTLBuffer> b = [dev newBufferWithLength:length options:MTLResourceStorageModeShared]; if (!b) throw std::runtime_error("buffer allocation failed: " + key); ctx.resident[key] = b; return b; }
static id<MTLTexture> resident_tex(NrBodyContext& ctx, id<MTLDevice> dev, const std::string& key, MTLTextureDescriptor* d) { auto it = ctx.resident_tex.find(key); if (it != ctx.resident_tex.end() && it->second.width == d.width && it->second.height == d.height && it->second.pixelFormat == d.pixelFormat) return it->second; id<MTLTexture> t = [dev newTextureWithDescriptor:d]; if (!t) throw std::runtime_error("texture allocation failed: " + key); ctx.resident_tex[key] = t; return t; }
static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(std::string("live input rejected: ") + what); }
int nr_body_run_frame(NrBodyContext& ctx, NSDictionary* chain, const NrBodyFrameInputs& in, const char* diagnostics_dir, NrBodyResult& res) { @autoreleasepool { res = NrBodyResult{}; try {
  // v3: every computed check is ENFORCED here (coverage of every middle/repack/ViT output, decoder constant immutability, final-surface write coverage,
  // input immutability) regardless of diagnostics; diagnostics only decide what is stored. Failures collect into res.check_failures and make the frame
  // return code 3 (checks failed) - distinct from 2 (execution error / rejected input).
  std::vector<std::string>& fails = res.check_failures; auto check = [&](bool ok, const std::string& what) { ++res.checks_enforced; if (!ok) fails.push_back(what); };
  struct { bool seen = false, ok = false; std::string hist, out; } pend;   // v10: deferred temporal commit (filled by the copy-156 branch, applied only at the end of a clean frame)
#ifdef NR_BODY_FAULT_TESTS
  const std::string fault = in.test_fault ? in.test_fault : "";   // test builds only: deliberately corrupt ONE computed check after its kernel ran, pixels untouched
#else
  if (in.test_fault) throw std::runtime_error("live input rejected: test_fault set on a production build");
  const std::string fault;
#endif
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
  auto pso = [&](id<MTLLibrary> l, const char* name) { const std::string key = std::to_string(reinterpret_cast<uintptr_t>((__bridge void*)l)) + "|" + name; auto it = ctx.pipes.find(key); if (it != ctx.pipes.end()) return it->second; id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err]; if (!ps) throw std::runtime_error(std::string(name) + ": " + err.localizedDescription.UTF8String); ctx.pipes[key] = ps; return ps; };
  const size_t pool_bytes = size_t(JInt(chain, "pool_bytes")); id<MTLBuffer> pool = resident_buf(ctx, dev, "pool", pool_bytes + 4096); std::memset(pool.contents, poison, pool_bytes); std::memset(static_cast<uint8_t*>(pool.contents) + pool_bytes, 0, 4096);
  require(in.entry.buffer != nil && in.entry.bytes == VIEW && in.entry.offset + VIEW <= in.entry.buffer.length, "encoder entry buffer/bytes/offset");
  unsigned scratch_seq = 0;   // v2: scratch buffers are resident in the context, keyed by their (deterministic) request order and size, re-filled every frame
  auto scratch = [&](size_t n, uint8_t fill) { id<MTLBuffer> b = resident_buf(ctx, dev, "scratch:" + std::to_string(scratch_seq++) + ":" + std::to_string(n), n + 16); std::memset(b.contents, fill, n + 16); return b; };
  auto constant = [&](const fs::path& p, size_t pad = 16) { const std::string key = "file:" + p.string() + ":" + std::to_string(pad); auto it = ctx.resident.find(key); if (it != ctx.resident.end()) return it->second;
    std::vector<uint8_t> v = Read(p); id<MTLBuffer> b = [dev newBufferWithLength:v.size() + pad options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + pad); std::memcpy(b.contents, v.data(), v.size()); ctx.resident[key] = b; ctx.resident_sha256[key] = Sha(v.data(), v.size()); ctx.resident_bytes[key] = v.size(); return b; };   // model constants: loaded and hashed once per process
  // v4: every live input is validated here, BEFORE the entry blit and before any compute dispatch (the same checks are repeated at their use sites)
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
      { std::vector<uint8_t> hb(size_t(tw0) * th0 * 8); [in.temporal->history getBytes:hb.data() bytesPerRow:tw0 * 8 fromRegion:MTLRegionMake2D(0, 0, tw0, th0) mipmapLevel:0]; require(Sha(hb.data(), hb.size()) == in.temporal->history_sha256, "history texture content does not match the recorded history hash"); } }   // checked BEFORE any dispatch
    if (in.temporal) require(in.temporal->history != nil && in.temporal->width == tw0 && in.temporal->height == th0 && in.output_surface != nil, "temporal state present: history texture WxH and caller output surface required");
  }
  { const size_t off = size_t(JInt(chain, "entry_offset")); id<MTLCommandBuffer> c = [queue commandBuffer]; id<MTLBlitCommandEncoder> bl = [c blitCommandEncoder]; [bl copyFromBuffer:in.entry.buffer sourceOffset:in.entry.offset toBuffer:pool destinationOffset:off size:VIEW]; [bl endEncoding]; [c commit]; [c waitUntilCompleted]; if (c.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("entry copy failed"); }
  id<MTLBuffer> rsq = constant(tables / "rtx-rsqrt-half.bin"), rcp = constant(tables / "rtx-rcp-half.bin");
  const size_t covmax = std::max(T * 4096 * 4, TIN * 4); id<MTLBuffer> cov = scratch(covmax, 0), cov2 = scratch(POOLB * 4, 0), y16 = scratch(TIN * 2, 0), qb = scratch(TIN, uint8_t(poison)), kb = scratch(TIN, uint8_t(poison)), vb = scratch(TIN, uint8_t(poison));
  std::ostringstream js; js << "{\n \"schema\": \"neuralstudio.nr.body-runtime.v5-temporal-experimental\",\n \"pid\": " << getpid() << ",\n \"poison\": " << poison << ",\n \"launches\": [\n"; bool first = true; double gpu_total = 0; const auto t_all = [NSDate date];
  auto cov_summary = [&](id<MTLBuffer> c, size_t bytes, size_t& once, size_t& multi, size_t& zero) { const uint32_t* p = static_cast<const uint32_t*>(c.contents); once = multi = zero = 0; for (size_t i = 0; i < bytes; ++i) { if (p[i] == 1) ++once; else if (p[i] > 1) ++multi; else ++zero; } };
  auto record = [&](long n, const std::string& kind, id<MTLCommandBuffer> cmd, const std::vector<std::pair<std::string, std::string>>& outs, const std::string& covline, double wall_ms) {
    const double gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0; gpu_total += gpu_ms;
    js << (first ? "" : ",\n") << "  {\"launch\": " << n << ", \"kind\": \"" << kind << "\", \"gpu_ms\": " << gpu_ms << ", \"wall_ms\": " << wall_ms << ", \"outputs\": {"; bool f2 = true;
    for (auto& o : outs) { js << (f2 ? "" : ", ") << "\"" << o.first << "\": \"" << o.second << "\""; f2 = false; }
    js << "}" << covline << "}"; first = false; std::cout << "launch " << n << " " << kind << " gpu_ms " << gpu_ms << "\n";
  };
  auto pool_at = [&](size_t off) { return static_cast<uint8_t*>(pool.contents) + off; };
  id<MTLBuffer> spw[3], spmax, spstats; if (sptp) { for (int m = 0; m < 3; ++m) spw[m] = scratch(16u * 32u * 512u * 2u, 0); spmax = scratch(16u * 512u, 0); spstats = scratch(64, 0); }
  id<MTLBuffer> atk[3], atv[3], atkmax, atvmax; if (sptp) { for (int m = 0; m < 3; ++m) { atk[m] = scratch(40u * 16u * 32u * 64u * 2u, 0); atv[m] = scratch(40u * 16u * 2u * 32u * 32u * 2u, 0); } atkmax = scratch(40u * 16u * 64u, 0); atvmax = scratch(40u * 16u * 2u * 32u, 0); }   // split attend operands per (window, head)
  id<MTLBuffer> qw[3], qwmax; if (sptp) { for (int m = 0; m < 3; ++m) qw[m] = scratch(16u * 32u * 1536u * 2u, 0); qwmax = scratch(16u * 1536u, 0); }   // split qkv weights: 16 steps x 1536 columns
  id<MTLBuffer> fw2[3], fw3[3], fw2max, fw3max; if (sptp) { for (int m = 0; m < 3; ++m) { fw2[m] = scratch(8u * 2u * 32u * 256u * 2u, 0); fw3[m] = scratch(8u * 8u * 32u * 64u * 2u, 0); } fw2max = scratch(8u * 2u * 256u, 0); fw3max = scratch(8u * 8u * 64u, 0); }
  // ---- split launches ----
  auto run_split = [&](NSDictionary* L) {
    const long n = JInt(L, "launch"); const std::string kernel = JStr(L, "kernel"); NSArray* shift = L[@"shift"]; NSDictionary* roles = L[@"roles"]; auto weights = constant(JStr(L, "weights"));
    auto off = [&](const char* role) { NSDictionary* r = roles[[NSString stringWithUTF8String:role]]; if (!r) throw std::runtime_error(std::to_string(n) + ": role " + role); return size_t(JInt(r, "pool_offset")); };
    SplitParams prm{int32_t([shift[0] intValue]), int32_t([shift[1] intValue]), 0, 2160};
    id<MTLCommandBuffer> cmd = [queue commandBuffer]; id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder]; [blit fillBuffer:cov range:NSMakeRange(0, covmax + 16) value:0]; [blit fillBuffer:cov2 range:NSMakeRange(0, POOLB * 4 + 16) value:0]; [blit endEncoding];
    std::vector<std::pair<std::string, size_t>> outs;   // role -> (offset, bytes) recorded after
    std::vector<std::pair<std::string, size_t>> outbytes;
    auto dispatch = [&](const char* name, std::vector<std::pair<id<MTLBuffer>, size_t>> bufs, size_t threads) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(slib, name)];
      for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i].first offset:bufs[i].second atIndex:i];
      [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
    };
    size_t o_off = 0, o_bytes = 0; std::string o_role;
    if (kernel == "ffwd" || kernel == "ffwd_view") { const bool view = kernel == "ffwd_view"; prm.flags = view ? FLAG_IN_VIEW : 0; o_role = "out"; o_off = off("out"); o_bytes = TIN;
      if (sptp) {
        { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:1 + m]; [enc setBuffer:spmax offset:0 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(16u * 512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_ffwd_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw2[m] offset:0 atIndex:1 + m]; [enc setBuffer:fw2max offset:0 atIndex:4]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw3[m] offset:0 atIndex:5 + m]; [enc setBuffer:fw3max offset:0 atIndex:8];
          [enc dispatchThreads:MTLSizeMake(8u * 2u * 256u + 8u * 8u * 64u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_ffwd_tp")];
        [enc setBuffer:pool offset:off(view ? "x_view" : "x") atIndex:0]; [enc setBuffer:weights offset:0 atIndex:1]; [enc setBuffer:pool offset:o_off atIndex:2]; [enc setBuffer:cov offset:0 atIndex:3]; [enc setBytes:&prm length:sizeof(prm) atIndex:4];
        for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:5 + m]; [enc setBuffer:spmax offset:0 atIndex:8]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw2[m] offset:0 atIndex:9 + m]; [enc setBuffer:fw2max offset:0 atIndex:12]; for (int m = 0; m < 3; ++m) [enc setBuffer:fw3[m] offset:0 atIndex:13 + m]; [enc setBuffer:fw3max offset:0 atIndex:16]; [enc setBuffer:spstats offset:0 atIndex:17];
        [enc dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
      } else
      dispatch("nr_split_ffwd", {{pool, off(view ? "x_view" : "x")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 2160u * 8u); }
    else if (kernel == "conv" || kernel == "conv_resview" || kernel == "conv_outview" || kernel == "conv_pool") {
      const bool rv = kernel == "conv_resview", ov = kernel == "conv_outview", pl = kernel == "conv_pool"; prm.flags = (rv ? FLAG_RES_VIEW : 0) | (ov ? FLAG_OUT_VIEW : 0) | (pl ? FLAG_EMIT_F16 : 0);
      o_role = ov ? "out_view" : "out"; o_off = off(o_role.c_str()); o_bytes = ov ? VIEW : TIN;
      if (sptp) {
        { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:1 + m]; [enc setBuffer:spmax offset:0 atIndex:4];
          [enc dispatchThreads:MTLSizeMake(16u * 512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_conv_tp")];
        [enc setBuffer:pool offset:off("x") atIndex:0]; [enc setBuffer:pool offset:off(rv ? "res_view" : "res") atIndex:1]; [enc setBuffer:weights offset:0 atIndex:2]; [enc setBuffer:pool offset:o_off atIndex:3]; [enc setBuffer:y16 offset:0 atIndex:4]; [enc setBuffer:cov offset:0 atIndex:5];
        [enc setBytes:&prm length:sizeof(prm) atIndex:6]; for (int m = 0; m < 3; ++m) [enc setBuffer:spw[m] offset:0 atIndex:7 + m]; [enc setBuffer:spmax offset:0 atIndex:10]; [enc setBuffer:spstats offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(8, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
      } else
      dispatch("nr_split_conv", {{pool, off("x")}, {pool, off(rv ? "res_view" : "res")}, {weights, 0}, {pool, o_off}, {y16, 0}, {cov, 0}}, 2160u * 32u);
      if (pl) { dispatch("nr_split_pool", {{y16, 0}, {pool, off("pooled")}, {cov2, 0}}, 640u * 32u); outbytes.push_back({"pooled", off("pooled")}); } }
    else if (kernel == "qkv_attend") { o_role = "out"; o_off = off("out"); o_bytes = TIN;
      if (sptp) {
        struct SpPrepParams { uint32_t steps, cols, step_bytes, reserved; } pp{16u, 1536u, 49152u, 0u};
        { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_tp_prep_generic")]; [enc setBuffer:weights offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:qw[m] offset:0 atIndex:1 + m]; [enc setBuffer:qwmax offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
          [enc dispatchThreads:MTLSizeMake(16u * 1536u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_qkv_tp")];
        [enc setBuffer:pool offset:off("x") atIndex:0]; [enc setBuffer:weights offset:0 atIndex:1]; [enc setBuffer:qb offset:0 atIndex:2]; [enc setBuffer:kb offset:0 atIndex:3]; [enc setBuffer:vb offset:0 atIndex:4]; [enc setBuffer:rsq offset:0 atIndex:5]; [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        for (int m = 0; m < 3; ++m) [enc setBuffer:qw[m] offset:0 atIndex:7 + m]; [enc setBuffer:qwmax offset:0 atIndex:10]; [enc setBuffer:spstats offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(16, 2160 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
        // windowed attention on the engine: per-(window, head) K^T / V operands from the live q/k/v, then the attention tile kernel
        enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_attend_tp_prep")]; [enc setBuffer:kb offset:0 atIndex:0]; [enc setBuffer:vb offset:0 atIndex:1]; [enc setBytes:&prm length:sizeof(prm) atIndex:2];
        for (int m = 0; m < 3; ++m) [enc setBuffer:atk[m] offset:0 atIndex:3 + m]; [enc setBuffer:atkmax offset:0 atIndex:6]; for (int m = 0; m < 3; ++m) [enc setBuffer:atv[m] offset:0 atIndex:7 + m]; [enc setBuffer:atvmax offset:0 atIndex:10];
        [enc dispatchThreads:MTLSizeMake(40u * 16u * 64u + 40u * 16u * 2u * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
        enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(sptp, "nr_split_attend_tp")];
        [enc setBuffer:qb offset:0 atIndex:0]; [enc setBuffer:kb offset:0 atIndex:1]; [enc setBuffer:vb offset:0 atIndex:2]; [enc setBuffer:weights offset:0 atIndex:3]; [enc setBuffer:pool offset:o_off atIndex:4]; [enc setBuffer:cov offset:0 atIndex:5]; [enc setBuffer:rcp offset:0 atIndex:6]; [enc setBytes:&prm length:sizeof(prm) atIndex:7];
        for (int m = 0; m < 3; ++m) [enc setBuffer:atk[m] offset:0 atIndex:8 + m]; [enc setBuffer:atkmax offset:0 atIndex:11]; for (int m = 0; m < 3; ++m) [enc setBuffer:atv[m] offset:0 atIndex:12 + m]; [enc setBuffer:atvmax offset:0 atIndex:15]; [enc setBuffer:spstats offset:0 atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake(16, 40 * 4, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
      } else {
      dispatch("nr_split_qkv", {{pool, off("x")}, {weights, 0}, {qb, 0}, {kb, 0}, {vb, 0}, {rsq, 0}}, 2160u * 16u);
      dispatch("nr_split_attend", {{qb, 0}, {kb, 0}, {vb, 0}, {weights, 0}, {pool, o_off}, {cov, 0}, {rcp, 0}}, 40u * 16u * 64u); } }
    else if (kernel == "final_head") { o_role = "out_vit"; o_off = off("out_vit"); o_bytes = VITB; dispatch("nr_split_final_head", {{pool, off("x_pooled")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 640u * 64u); }
    else if (kernel == "dec_upsample") { o_role = "out"; o_off = off("out"); o_bytes = TIN; dispatch("nr_split_dec_upsample", {{pool, off("x_vit")}, {pool, off("res")}, {weights, 0}, {pool, o_off}, {cov, 0}}, 2160u * 32u); }
    else throw std::runtime_error("unknown kernel " + kernel);
    const auto t0 = [NSDate date]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("command buffer failed at launch " + std::to_string(n));
    const double wall_ms = -[t0 timeIntervalSinceNow] * 1000.0;
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "middle-coverage" && n == 25) static_cast<uint32_t*>(cov.contents)[0] = 2u;
#endif
    size_t once, multi, zero; cov_summary(cov, o_bytes, once, multi, zero); std::ostringstream cl; cl << ", \"coverage\": {\"" << o_role << "\": [" << once << ", " << multi << ", " << zero << "]";
    check(multi == 0, "launch " + std::to_string(n) + " " + o_role + ": " + std::to_string(multi) + " bytes written more than once");
    if (o_role == "out_vit") {   // launch 57 contract: exactly the padding tokens' bytes are never written (the driver zeroes them before repack 58); everything else exactly once
      NSArray* pad = chain[@"vit"][@"padding_tokens"]; std::vector<uint8_t> is_pad(o_bytes, 0);
      for (NSNumber* t in pad) { const unsigned tok = t.unsignedIntValue; for (unsigned c = 0; c < 1024; ++c) { const unsigned tile = tok / 16, slot = tok % 16, lane = 4 * (slot % 8) + (c % 16) / 4, reg = (slot % 16) / 8 + 2 * ((c % 32) / 16); is_pad[tile * 16384 + (c / 32) * 512 + lane * 16 + reg * 4 + c % 4] = 1; } }
      const uint32_t* cp = static_cast<const uint32_t*>(cov.contents); size_t bad = 0; for (size_t i = 0; i < o_bytes; ++i) if ((cp[i] == 0) != (is_pad[i] == 1)) ++bad;
      res.out_vit_padding_bytes = size_t(pad.count) * 1024u; check(bad == 0 && zero == res.out_vit_padding_bytes, "launch " + std::to_string(n) + " out_vit: " + std::to_string(bad) + " bytes violate the padding-token write contract (zero=" + std::to_string(zero) + ", padding=" + std::to_string(size_t(pad.count) * 1024u) + ")");
    } else check(zero == 0, "launch " + std::to_string(n) + " " + o_role + ": " + std::to_string(zero) + " bytes never written");
    std::vector<std::pair<std::string, std::string>> hashes{{o_role, Sha(pool_at(o_off), o_bytes)}};
    for (auto& ob : outbytes) { size_t o1, m1, z1; cov_summary(cov2, POOLB, o1, m1, z1); cl << ", \"" << ob.first << "\": [" << o1 << ", " << m1 << ", " << z1 << "]"; hashes.push_back({ob.first, Sha(pool_at(ob.second), POOLB)});
      check(m1 == 0 && z1 == 0, "launch " + std::to_string(n) + " " + ob.first + ": coverage once=" + std::to_string(o1) + " multi=" + std::to_string(m1) + " zero=" + std::to_string(z1)); }
    cl << "}"; record(n, kernel, cmd, hashes, cl.str(), wall_ms);
  };
  for (NSDictionary* L in chain[@"split_a"]) run_split(L);
  // ---- repack 58 (single direction, live input: the final-head output with padding tokens zeroed by the driver's rule) ----
  NSDictionary* R = chain[@"vit"]; const size_t vit_in_off = size_t(JInt(R, "final_head_offset"));
  std::vector<uint8_t> vit_in(720896, 0); std::memcpy(vit_in.data(), pool_at(vit_in_off), VITB);
  { NSArray* pad = R[@"padding_tokens"]; for (NSNumber* t in pad) { const unsigned tok = t.unsignedIntValue; for (unsigned c = 0; c < 1024; ++c) { const unsigned tile = tok / 16, slot = tok % 16, lane = 4 * (slot % 8) + (c % 16) / 4, reg = (slot % 16) / 8 + 2 * ((c % 32) / 16); vit_in[tile * 16384 + (c / 32) * 512 + lane * 16 + reg * 4 + c % 4] = 0; } } }
  id<MTLBuffer> vin = scratch(vit_in.size(), 0); std::memcpy(vin.contents, vit_in.data(), vit_in.size()); id<MTLBuffer> tok = scratch(VITB, uint8_t(poison)), rcov = scratch(VITB * 4, 0);
  auto run_repack = [&](const char* name, long n, id<MTLBuffer> in, size_t in_bytes, id<MTLBuffer> outb, size_t out_off, size_t out_bytes) {
    RepackParams prm{20, 32, uint32_t(in_bytes / 4), uint32_t(out_bytes / 4)}; id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder]; [blit fillBuffer:rcov range:NSMakeRange(0, VITB * 4 + 16) value:0]; [blit endEncoding];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(vlib, name)]; [enc setBuffer:in offset:0 atIndex:0]; [enc setBuffer:outb offset:out_off atIndex:1]; [enc setBuffer:rcov offset:0 atIndex:2]; [enc setBytes:&prm length:sizeof(prm) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding];
    const auto t0 = [NSDate date]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("repack failed");
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "repack-coverage" && n == 58) static_cast<uint32_t*>(rcov.contents)[7] = 0u;
#endif
    const uint32_t* c = static_cast<const uint32_t*>(rcov.contents); size_t once = 0, multi = 0, zero = 0; for (size_t i = 0; i < out_bytes / 4; ++i) { if (c[i] == 1) ++once; else if (c[i] > 1) ++multi; else ++zero; }
    std::ostringstream cl; cl << ", \"coverage_dwords\": [" << once << ", " << multi << ", " << zero << "]";
    check(multi == 0 && zero == 0, std::string("repack ") + std::to_string(n) + ": coverage dwords once=" + std::to_string(once) + " multi=" + std::to_string(multi) + " zero=" + std::to_string(zero));
    record(n, name, cmd, {{name, Sha(static_cast<uint8_t*>(outb.contents) + out_off, out_bytes)}}, cl.str(), -[t0 timeIntervalSinceNow] * 1000.0);
  };
  run_repack("nr_vit_repack_2d_to_1d", 58, vin, 720896, tok, 0, VITB);
  // ---- ViT blocks 31-38, resident intermediates ----
  id<MTLBuffer> expand = scratch(T * 4096, uint8_t(poison)), contract = scratch(T * 1024, uint8_t(poison)), q = scratch(T * 1024, uint8_t(poison)), k = scratch(T * 1024, uint8_t(poison)), v = scratch(T * 1024, uint8_t(poison)), attention = scratch(T * 1024, uint8_t(poison)), projection = scratch(T * 1024, uint8_t(poison));
  id<MTLBuffer> covE = scratch(T * 4096 * 4, 0), covC = scratch(T * 1024 * 4, 0), covQ = scratch(T * 1024 * 4, 0), covK = scratch(T * 1024 * 4, 0), covV = scratch(T * 1024 * 4, 0), covA = scratch(T * 1024 * 4, 0), covP = scratch(T * 1024 * 4, 0);
  id<MTLBuffer> tokens = tok;
  // three-product weight preparation buffers (l/h/l+h halves and per-(step, column) max class), reused per block
  struct TpPrepParams { uint32_t steps, cols, step_bytes, reserved; };
  id<MTLBuffer> tpw[3][3], tpmax[3], tpstats = scratch(64, 0); const uint32_t tp_steps[3] = {32u, 128u, 32u}, tp_cols[3] = {4096u, 1024u, 1024u}, tp_stepb[3] = {131072u, 32768u, 32768u};
  if (tplib) for (int st = 0; st < 3; ++st) { const size_t WH = size_t(tp_steps[st]) * 32u * tp_cols[st]; for (int m = 0; m < 3; ++m) tpw[st][m] = scratch(WH * 2u, 0); tpmax[st] = scratch(size_t(tp_steps[st]) * tp_cols[st], 0); }
  id<MTLBuffer> tqw[3], tqmax; if (tplib) { for (int m = 0; m < 3; ++m) tqw[m] = scratch(32u * 32u * 3072u * 2u, 0); tqmax = scratch(32u * 3072u, 0); }   // qkv weights (blob at +128): 32 steps x 3072 columns
  id<MTLBuffer> tak[3], tav[3], takmax, tavmax; if (tplib) { for (int m = 0; m < 3; ++m) { tak[m] = scratch(32u * 32u * 640u * 2u, 0); tav[m] = scratch(32u * 20u * 32u * 32u * 2u, 0); } takmax = scratch(32u * 640u, 0); tavmax = scratch(32u * 20u * 32u, 0); }   // attention operands per block: K^T and V as l/h/l+h halves + max classes
  for (NSDictionary* B in R[@"blocks"]) {
    const long block = JInt(B, "block"); auto W0 = constant(JStr(B, "w0")), W1 = constant(JStr(B, "w1")), W2 = constant(JStr(B, "w2")), W4 = constant(JStr(B, "w4"));
    VitParams prm{uint32_t(T), uint32_t(block), 0, 0}; id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder]; for (id<MTLBuffer> c : {covE, covC, covQ, covK, covV, covA, covP}) [blit fillBuffer:c range:NSMakeRange(0, c.length) value:0]; [blit endEncoding];
    auto dispatch = [&](const char* name, std::vector<id<MTLBuffer>> bufs, size_t threads, uint32_t stage) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(vlib, name)];
      for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
      prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
    };
    if (tplib) {
      auto prep = [&](int st, id<MTLBuffer> W) { TpPrepParams pp{tp_steps[st], tp_cols[st], tp_stepb[st], 0u}; id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(tplib, "nr_vit_tp_prep")];
        [enc setBuffer:W offset:0 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:tpw[st][m] offset:0 atIndex:1 + m]; [enc setBuffer:tpmax[st] offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(size_t(tp_steps[st]) * tp_cols[st], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; };
      auto tpdispatch = [&](const char* name, int st, std::vector<id<MTLBuffer>> bufs, uint32_t stage) {   // bufs: the accepted kernel's buffers; then params, wl, wh, wlh, maxcls, stats
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(tplib, name)];
        for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()]; for (int m = 0; m < 3; ++m) [enc setBuffer:tpw[st][m] offset:0 atIndex:bufs.size() + 1 + m]; [enc setBuffer:tpmax[st] offset:0 atIndex:bufs.size() + 4]; [enc setBuffer:tpstats offset:0 atIndex:bufs.size() + 5];
        [enc dispatchThreadgroups:MTLSizeMake(tp_cols[st] / 64, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; };
      prep(0, W0); tpdispatch("nr_vit_expand_tp", 0, {tokens, W0, expand, covE}, 1); prep(1, W1); tpdispatch("nr_vit_contract_tp", 1, {expand, tokens, W1, contract, covC}, 2);
      { TpPrepParams pp{32u, 3072u, 98304u, 0u}; id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(tplib, "nr_vit_tp_prep")];
        [enc setBuffer:W2 offset:128 atIndex:0]; for (int m = 0; m < 3; ++m) [enc setBuffer:tqw[m] offset:0 atIndex:1 + m]; [enc setBuffer:tqmax offset:0 atIndex:4]; [enc setBytes:&pp length:sizeof(pp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(32u * 3072u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
        enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(tplib, "nr_vit_qkv_tp")]; std::vector<id<MTLBuffer>> qb{contract, W2, q, k, v, covQ, covK, covV, rsq};
        for (size_t i = 0; i < qb.size(); ++i) [enc setBuffer:qb[i] offset:0 atIndex:i]; prm.stage = 3; [enc setBytes:&prm length:sizeof(prm) atIndex:9]; for (int m = 0; m < 3; ++m) [enc setBuffer:tqw[m] offset:0 atIndex:10 + m]; [enc setBuffer:tqmax offset:0 atIndex:13]; [enc setBuffer:tpstats offset:0 atIndex:14];
        [enc dispatchThreadgroups:MTLSizeMake(32, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; }
      { id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(tplib, "nr_vit_attn_prep")]; [enc setBuffer:k offset:0 atIndex:0]; [enc setBuffer:v offset:0 atIndex:1];
        for (int m = 0; m < 3; ++m) [enc setBuffer:tak[m] offset:0 atIndex:2 + m]; [enc setBuffer:takmax offset:0 atIndex:5]; for (int m = 0; m < 3; ++m) [enc setBuffer:tav[m] offset:0 atIndex:6 + m]; [enc setBuffer:tavmax offset:0 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(32u * 640u + 32u * 20u * 32u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
        enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(tplib, "nr_vit_attention_tp")]; std::vector<id<MTLBuffer>> abf{q, k, v, attention, covA, rcp};
        for (size_t i = 0; i < abf.size(); ++i) [enc setBuffer:abf[i] offset:0 atIndex:i]; prm.stage = 4; [enc setBytes:&prm length:sizeof(prm) atIndex:6];
        for (int m = 0; m < 3; ++m) [enc setBuffer:tak[m] offset:0 atIndex:7 + m]; [enc setBuffer:takmax offset:0 atIndex:10]; for (int m = 0; m < 3; ++m) [enc setBuffer:tav[m] offset:0 atIndex:11 + m]; [enc setBuffer:tavmax offset:0 atIndex:14]; [enc setBuffer:tpstats offset:0 atIndex:15];
        [enc dispatchThreadgroups:MTLSizeMake(32, T / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; }
      prep(2, W4); tpdispatch("nr_vit_projection_tp", 2, {attention, contract, W4, projection, covP}, 5);
    } else {
    dispatch("nr_vit_expand", {tokens, W0, expand, covE}, T * 256, 1); dispatch("nr_vit_contract", {expand, tokens, W1, contract, covC}, T * 64, 2);
    dispatch("nr_vit_qkv", {contract, W2, q, k, v, covQ, covK, covV, rsq}, T * 32, 3); dispatch("nr_vit_attention", {q, k, v, attention, covA, rcp}, T * 32, 4); dispatch("nr_vit_projection", {attention, contract, W4, projection, covP}, T * 64, 5);
    }
    const auto t0 = [NSDate date]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("vit block failed");
    std::ostringstream cl; cl << ", \"coverage\": {"; bool f3 = true;
#ifdef NR_BODY_FAULT_TESTS
    if (fault == "vit-coverage" && block == 31) static_cast<uint32_t*>(covE.contents)[3] = 0u;
#endif
    for (auto& pr : std::vector<std::pair<const char*, std::pair<id<MTLBuffer>, size_t>>>{{"expand", {covE, T * 4096}}, {"contract", {covC, T * 1024}}, {"q", {covQ, T * 1024}}, {"k", {covK, T * 1024}}, {"v", {covV, T * 1024}}, {"attention", {covA, T * 1024}}, {"projection", {covP, T * 1024}}}) { size_t o1, m1, z1; cov_summary(pr.second.first, pr.second.second, o1, m1, z1); check(m1 == 0 && z1 == 0, "block " + std::to_string(block) + " " + pr.first + ": coverage once=" + std::to_string(o1) + " multi=" + std::to_string(m1) + " zero=" + std::to_string(z1)); cl << (f3 ? "" : ", ") << "\"" << pr.first << "\": [" << o1 << ", " << m1 << ", " << z1 << "]"; f3 = false; }
    cl << "}";
    record(block, "vit_block", cmd, {{"expand", Sha(expand.contents, T * 4096)}, {"contract", Sha(contract.contents, T * 1024)}, {"q", Sha(q.contents, T * 1024)}, {"k", Sha(k.contents, T * 1024)}, {"v", Sha(v.contents, T * 1024)}, {"attention", Sha(attention.contents, T * 1024)}, {"projection", Sha(projection.contents, T * 1024)}}, cl.str(), -[t0 timeIntervalSinceNow] * 1000.0);
    std::swap(tokens, projection); if (tokens == tok) {} // next block reads this block's projection; the freed buffer becomes the next projection target
    if (projection == tok) { projection = scratch(T * 1024, uint8_t(poison)); }
  }
  // ---- repack 99 -> pool (launch 100 input), then split launches 100-132 ----
  run_repack("nr_vit_repack_1d_to_2d", 99, tokens, VITB, pool, size_t(JInt(R, "repack99_out_offset")), VITB);
  for (NSDictionary* L in chain[@"split_b"]) run_split(L);
  { const size_t fo = size_t(JInt(chain, "final_offset")); res.middle_view_sha256 = Sha(pool_at(fo), VIEW); if (diag) std::ofstream(out / "out_view_132.bin", std::ios::binary).write(reinterpret_cast<const char*>(pool_at(fo)), VIEW); }
  const double middle_wall_ms = -[t_all timeIntervalSinceNow] * 1000.0; const double middle_gpu_ms = gpu_total; std::string tail_fields;
  // ---- optional fused decoder/output section (launches 133-156, transliterated PTX kernels) on the same resident pool ----
  if (NSDictionary* D = chain[@"decoder"]) {
    const auto t_dec = [NSDate date];
    // restage to the strict decoder pre-image: keep this run's middle view, poison everything else, apply the live-producer patches, control words
    const size_t fo = size_t(JInt(chain, "final_offset")); std::vector<uint8_t> mid(VIEW); std::memcpy(mid.data(), pool_at(fo), VIEW);
    std::memset(pool.contents, poison, pool_bytes); std::memcpy(pool_at(fo), mid.data(), VIEW);
    { NSArray* patches = D[@"patches"]; require(patches && patches.count == 5, "descriptor patch table (4 skips + Block0 skip)");
      const NrBodyRegion* regs[5] = {&in.skips[0], &in.skips[1], &in.skips[2], &in.skips[3], &in.block0_skip}; id<MTLCommandBuffer> c = [queue commandBuffer]; id<MTLBlitCommandEncoder> bl = [c blitCommandEncoder];
      for (unsigned i = 0; i < 5; ++i) { NSDictionary* P = patches[i]; const size_t off = size_t(JInt(P, "offset")), bytes = size_t(JInt(P, "bytes")); const NrBodyRegion& r = *regs[i];
        require(r.buffer != nil && r.bytes == bytes && r.offset + bytes <= r.buffer.length && off + bytes <= pool_bytes, JStr(P, "file").c_str());
        [bl copyFromBuffer:r.buffer sourceOffset:r.offset toBuffer:pool destinationOffset:off size:bytes]; }
      [bl endEncoding]; [c commit]; [c waitUntilCompleted]; if (c.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("skip copy failed"); }
    { NSArray* c = D[@"control"]; const size_t c0 = [c[0] unsignedLongValue], c1 = [c[1] unsignedLongValue]; std::memset(pool_at(c0), 0xff, c1 - c0); }
    const std::string pre_sha = Sha(pool.contents, pool_bytes);
    const bool final_only = in.production; const fs::path dtables = JStr(D, "tables");
    const unsigned tw = unsigned(JInt(D, "tex_w")), th = unsigned(JInt(D, "tex_h")); require(in.input_color != nil && in.input_color.pixelFormat == MTLPixelFormatRGBA16Float && in.input_color.width == tw && in.input_color.height == th, "input colour texture (RGBA16F WxH)");
    std::vector<uint8_t> texin(size_t(tw) * th * 8); [in.input_color getBytes:texin.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
    auto opt = [&](const char* name) { fs::path f = dtables / name; if (fs::exists(f)) return constant(f); return scratch(16, 0); };
    id<MTLBuffer> bpool = pool, brsq = opt("rtx-rsqrt-half.bin"), brcp = opt("rtx-rcp-half.bin"), brom = opt("sm120-rcp-correction-rom.bin"), bex2u = opt("unit-corrections.bin"), bex2f = opt("fraction-corrections.bin");
    id<MTLBuffer> bcov = scratch(pool_bytes + 4096 - 16, 0); id<MTLBuffer> btexcov = scratch(size_t(tw) * th * 4, 0);
    id<MTLBuffer> bdbg = scratch(64 * 32 * 4, 0);
    MTLTextureDescriptor* dIn = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; dIn.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; dIn.storageMode = MTLStorageModeShared;
    if (in.output_surface) require(in.output_surface.pixelFormat == MTLPixelFormatRGBA16Float && in.output_surface.width == tw && in.output_surface.height == th, "output surface (RGBA16F WxH)");
    id<MTLTexture> tex[3] = {in.input_color, resident_tex(ctx, dev, "tex:1", dIn), in.output_surface ? in.output_surface : resident_tex(ctx, dev, "tex:2", dIn)};
    std::vector<uint8_t> tpoison(size_t(tw) * th * 8, uint8_t(poison)); for (int i = 1; i < 3; ++i) [tex[i] replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:tpoison.data() bytesPerRow:tw * 8];
    const std::string texin_sha = Sha(texin.data(), texin.size()); res.input_texture_sha256_before = texin_sha;
    if (!in.reset) { std::vector<uint8_t> mb(size_t(tw) * th * 4); [in.motion getBytes:mb.data() bytesPerRow:tw * 4 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; res.motion_sha256 = Sha(mb.data(), mb.size());
      std::vector<uint8_t> hb(size_t(tw) * th * 8); [in.temporal->history getBytes:hb.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; require(Sha(hb.data(), hb.size()) == in.temporal->history_sha256, "history texture content does not match the recorded history hash"); res.history_sha256_before = in.temporal->history_sha256; }
        for (NSDictionary* L in D[@"launches"]) {
      const unsigned n = unsigned(JInt(L, "launch")); const bool temporal_launch = (n == 155u && !in.reset); const bool temporal_copy = (n == 156u && in.temporal != nullptr);
      if (temporal_copy) require(!JStr(L, "lib_temporal").empty(), "a temporal state requires the descriptor's lib_temporal for launch 156 (sign-preserving copy)");
      const std::string libp = (temporal_launch || temporal_copy) ? JStr(L, "lib_temporal") : JStr(L, "lib"), kernel = JStr(L, "kernel"); id<MTLBuffer> bw = constant(JStr(L, "weights"), 4096); id<MTLBuffer> bprm = constant(JStr(L, "params"), 64); const size_t params_size = bprm.length - 64, weights_size = bw.length - 4096;
      id<MTLBuffer> bex2h = nil, bfault = nil;
      if (temporal_launch) {   // v6: audited half-domain EX2 table, pinned and bound only for the temporal post launch
        NSDictionary* T = L[@"ex2_half_table"]; require(T != nil, "temporal launch 155 requires the descriptor's ex2_half_table {file, sha256}");
        const std::string tpath = JStr(T, "file"), tsha = JStr(T, "sha256"); auto tb = Read(tpath); const std::string got = Sha(tb.data(), tb.size());
        require(tb.size() == 524288 && got == tsha && got == "f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf", "ex2 half-domain table: size/sha mismatch against the descriptor and the pinned hash");
        bex2h = constant(tpath, 16); res.provenance.emplace_back("ex2_half_table:" + tpath, got); bfault = scratch(64, 0); std::memset(bfault.contents, 0, 64);
#ifdef NR_BODY_FAULT_TESTS
        if (fault == "ex2-table-row-corrupt") { uint32_t* row = static_cast<uint32_t*>(bex2h.contents) + 2u * 0xbb80u; row[0] ^= 1u; }   // real GPU lookup fault: the row of half 0xbb80 no longer matches its argument
#endif
        }
      if (temporal_launch) {   // v5: per-frame constructed record (the resident base record stays untouched and keeps its immutability check)
        id<MTLBuffer> base = bprm; bprm = scratch(params_size, 0); const uint8_t* b0 = static_cast<const uint8_t*>(base.contents); uint8_t* out = static_cast<uint8_t*>(bprm.contents);
        require(nr_temporal_build_post_record(b0, params_size, tw, th, 1u, 6u, out), "temporal record construction"); res.temporal_record_sha256 = Sha(out, params_size);
        size_t changed = 0; for (size_t i = 0; i < params_size; ++i) changed += (out[i] != b0[i]); const size_t allowed[] = {88, 89, 90, 91, 96, 97, 98, 99, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163};
        for (size_t i = 0; i < params_size; ++i) if (out[i] != b0[i]) { bool ok = false; for (size_t a : allowed) ok |= (a == i); require(ok, "temporal record deviates from the accepted base outside the marker/dimension words"); } (void)changed; }
      const uint8_t* params = static_cast<const uint8_t*>(bprm.contents);
      NSArray* g = L[@"grid"]; NSArray* b = L[@"block"]; const std::string texmode = JStr(L, "textures");
      id<MTLComputePipelineState> dps = pso(lib(libp), kernel.c_str());
      if (n == 155u && in.reset) { require(params_size >= 104, "post block parameter block"); for (size_t i = 88; i < 104; ++i) require(params[i] == 0, "temporal inputs present (launch 155 params +88/+96): unsupported path"); }   // see docs/NR_INPUT_CONTRACT.md
      const std::string w_sha = Sha(bw.contents, weights_size), p_sha = Sha(bprm.contents, params_size);   // immutability of the resident constants across the launch
      const auto t0 = [NSDate date]; id<MTLCommandBuffer> cmd = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder]; [blit fillBuffer:bcov range:NSMakeRange(0, pool_bytes + 4096) value:0]; if (!texmode.empty()) [blit fillBuffer:btexcov range:NSMakeRange(0, size_t(tw) * th * 4 + 16) value:0]; [blit endEncoding];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:dps];
      [enc setBuffer:bpool offset:0 atIndex:0]; [enc setBuffer:bw offset:0 atIndex:1]; [enc setBuffer:bprm offset:0 atIndex:2]; [enc setBuffer:brsq offset:0 atIndex:3]; [enc setBuffer:brcp offset:0 atIndex:4]; [enc setBuffer:bdbg offset:0 atIndex:5]; [enc setBuffer:bcov offset:0 atIndex:6];
      [enc setBuffer:brom offset:0 atIndex:7]; [enc setBuffer:bex2u offset:0 atIndex:8]; [enc setBuffer:bex2f offset:0 atIndex:9]; [enc setBuffer:btexcov offset:0 atIndex:10];
      if (temporal_launch) { [enc setBuffer:bex2h offset:0 atIndex:11]; [enc setBuffer:bfault offset:0 atIndex:12]; }
      if (texmode == "post") { [enc setTexture:tex[0] atIndex:0]; [enc setTexture:in.temporal ? tex[2] : tex[1] atIndex:1]; if (temporal_launch) { [enc setTexture:in.motion atIndex:2]; [enc setTexture:in.temporal->history atIndex:3]; } }
      else if (texmode == "copy") { if (in.temporal) { [enc setTexture:tex[2] atIndex:0]; [enc setTexture:in.temporal->history atIndex:1]; } else { [enc setTexture:tex[1] atIndex:0]; [enc setTexture:tex[2] atIndex:1]; } }
      [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntValue], [g[1] unsignedIntValue], [g[2] unsignedIntValue]) threadsPerThreadgroup:MTLSizeMake([b[0] unsignedIntValue], [b[1] unsignedIntValue], [b[2] unsignedIntValue])]; [enc endEncoding];
      [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "command buffer failed");
      const double gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0, wall_ms = -[t0 timeIntervalSinceNow] * 1000.0; gpu_total += gpu_ms;
#ifdef NR_BODY_FAULT_TESTS
      if (fault == "decoder-pool-multi" && n == 134u && !final_only) static_cast<uint32_t*>(bcov.contents)[100] = 2u;   // before the scan: pixels unaffected, coverage must trip
#endif
      const uint32_t* cv = static_cast<const uint32_t*>(bcov.contents); size_t once = 0, multi = 0; if (!final_only) for (size_t i = 0; i < pool_bytes / 4; ++i) { if (cv[i] == 1) ++once; else if (cv[i] > 1) ++multi; }
      if (!final_only) check(multi == 0, "launch " + std::to_string(n) + ": " + std::to_string(multi) + " pool words written more than once");
      std::string tex_line;
      if (!texmode.empty()) {
        const uint32_t* tc = static_cast<const uint32_t*>(btexcov.contents); size_t t1 = 0, tm = 0; for (size_t i = 0; i < size_t(tw) * th; ++i) { if (tc[i] == 1) ++t1; else if (tc[i] > 1) ++tm; }
#ifdef NR_BODY_FAULT_TESTS
        if (fault == "history-texel-corrupt" && texmode == "copy" && in.temporal) { uint16_t junk[4] = {0x3c00, 0, 0, 0x3c00}; [in.temporal->history replaceRegion:MTLRegionMake2D(5, 7, 1, 1) mipmapLevel:0 withBytes:junk bytesPerRow:8]; }   // real history fault: one texel differs from the caller output
#endif
        std::vector<uint8_t> o(size_t(tw) * th * 8); id<MTLTexture> to = in.temporal ? (texmode == "post" ? tex[2] : in.temporal->history) : tex[texmode == "post" ? 1 : 2]; [to getBytes:o.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
        if (diag) std::ofstream(out / ("tex_out_" + std::to_string(n) + ".bin"), std::ios::binary).write(reinterpret_cast<const char*>(o.data()), std::streamsize(o.size()));
        std::vector<uint8_t> hist_actual;   // v8: under a temporal state, o = the ACTUAL history texture after the copy (kept), the caller output is read separately
        if (texmode == "copy") { if (in.temporal) { hist_actual = o; std::vector<uint8_t> ob(size_t(tw) * th * 8); [tex[2] getBytes:ob.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; o.swap(ob); } res.final_surface_sha256 = Sha(o.data(), o.size()); res.final_pixels = size_t(tw) * th; res.final_pixels_not_written = size_t(tw) * th - t1 - tm; check(res.final_pixels_not_written == 0 && res.final_pixels_written_more_than_once == 0, "final surface not written exactly once (not_written=" + std::to_string(res.final_pixels_not_written) + ", multi=" + std::to_string(res.final_pixels_written_more_than_once) + ")"); res.final_pixels_written_more_than_once = tm; }
        std::vector<uint8_t> back(size_t(tw) * th * 8); [(in.temporal ? (texmode == "post" ? tex[0] : tex[2]) : tex[texmode == "post" ? 0 : 1]) getBytes:back.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
        if (temporal_launch) {   // v6: the half-domain EX2 lookups must all have matched exactly on finite halves
          const uint32_t* fl = static_cast<const uint32_t*>(bfault.contents); res.ex2_half_arg_faults = fl[0]; res.ex2_half_domain_faults = fl[1];
#ifdef NR_BODY_FAULT_TESTS
          if (fault == "ex2-table-fault") res.ex2_half_arg_faults = 1;
#endif
          check(res.ex2_half_arg_faults == 0, "ex2 half-domain table: " + std::to_string(res.ex2_half_arg_faults) + " argument mismatches"); check(res.ex2_half_domain_faults == 0, "ex2 half-domain table: " + std::to_string(res.ex2_half_domain_faults) + " nonfinite half operands"); }
        if (in.temporal && texmode == "post") {   // under temporal routing the post product IS the caller output surface; history/motion must be unchanged by 155
          if (temporal_launch) { std::vector<uint8_t> hb(size_t(tw) * th * 8); [in.temporal->history getBytes:hb.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; check(Sha(hb.data(), hb.size()) == in.temporal->history_sha256, "history texture changed during launch 155");
            std::vector<uint8_t> mb(size_t(tw) * th * 4); [in.motion getBytes:mb.data() bytesPerRow:tw * 4 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; check(Sha(mb.data(), mb.size()) == res.motion_sha256, "motion texture changed during launch 155"); } }
        if (in.temporal && texmode == "copy") {   // v8: independent hashes of the actual history texture and of the caller output, compared BEFORE the state advances
          const std::string osha = Sha(o.data(), o.size()), hsha = Sha(hist_actual.data(), hist_actual.size()); size_t hdiff = 0; for (size_t i = 0; i + 1 < o.size(); i += 2) hdiff += (o[i] != hist_actual[i] || o[i + 1] != hist_actual[i + 1]);
          res.history_sha256_after = hsha; res.history_copy_value_differences = hdiff;
#ifdef NR_BODY_FAULT_TESTS
          if (fault == "history-copy-mismatch") { hdiff = 1; }
#endif
          check(hdiff == 0 && osha == hsha, "history after the copy differs from the caller output (" + std::to_string(hdiff) + " half values; history " + hsha.substr(0, 12) + ", output " + osha.substr(0, 12) + "); temporal state NOT advanced");
          pend.seen = true; pend.ok = (hdiff == 0 && osha == hsha); pend.hist = hsha; pend.out = osha; }   // v10: commit deferred to the end of the frame
        if (texmode == "post") { res.input_texture_sha256_after = Sha(back.data(), back.size()); res.input_texture_immutable = res.input_texture_sha256_after == texin_sha; check(res.input_texture_immutable, "input colour texture changed during the post block"); }
        std::ostringstream ts; ts << ", \"texture\": {\"pixels_once\": " << t1 << ", \"pixels_multi\": " << tm << ", \"pixels\": " << size_t(tw) * th << ", \"output_sha256\": \"" << Sha(o.data(), o.size()) << "\", \"input_after_sha256\": \"" << Sha(back.data(), back.size()) << "\"}"; tex_line = ts.str();
      }
      const std::string pool_sha = final_only ? std::string("skipped") : Sha(pool.contents, pool_bytes);
#ifdef NR_BODY_FAULT_TESTS
      if (fault == "decoder-weights" && n == 133u) static_cast<uint8_t*>(bw.contents)[5] ^= 0x01;   // after the launch: pixels unaffected, immutability must trip
#endif
      const bool w_ok = Sha(bw.contents, weights_size) == w_sha, p_ok = Sha(bprm.contents, params_size) == p_sha;
      check(w_ok, "launch " + std::to_string(n) + ": weights buffer changed during the launch"); check(p_ok, "launch " + std::to_string(n) + ": params buffer changed during the launch");
      if (w_ok && p_ok) {} else ++res.decoder_constant_mutations;
      js << (first ? "" : ",\n") << "  {\"launch\": " << n << ", \"kind\": \"ptx\", \"kernel\": \"" << kernel << "\", \"gpu_ms\": " << gpu_ms << ", \"wall_ms\": " << wall_ms << ", \"pool_post_sha256\": \"" << pool_sha << "\", \"words_written_once\": " << (final_only ? std::string("null") : std::to_string(once)) << ", \"words_written_more_than_once\": " << (final_only ? std::string("null") : std::to_string(multi))
         << ", \"weights_immutable\": " << (w_ok ? "true" : "false") << ", \"params_immutable\": " << (p_ok ? "true" : "false") << ", \"weights_sha256\": \"" << w_sha << "\"" << tex_line << "}"; first = false;
      std::cout << "launch " << n << " ptx gpu_ms " << gpu_ms << " wall_ms " << wall_ms << "\n";
    }
    std::ostringstream tf; tf << ",\n \"decoder\": {\"checks\": \"" << (final_only ? "final" : "per-launch") << "\", \"pool_pre_sha256\": \"" << pre_sha << "\", \"pool_final_sha256\": \"" << Sha(pool.contents, pool_bytes) << "\", \"texture_in_sha256\": \"" << texin_sha << "\", \"gpu_ms\": " << (gpu_total - middle_gpu_ms) << ", \"wall_ms\": " << (-[t_dec timeIntervalSinceNow] * 1000.0) << "}";
    // production (final-only checks): the final pool hash above is the evidence; the 212 MB pool dump is skipped unless the chain asks for it
    res.decoder_pre_image_sha256 = pre_sha; res.pool_final_sha256 = Sha(pool.contents, pool_bytes); res.decoder_gpu_ms = gpu_total - middle_gpu_ms; if (final_only) { res.checks_skipped = {"decoder per-launch pool hash", "decoder per-launch pool coverage scan"}; }
    if (diag && (!final_only || JStr(D, "write_pool_post") == "true")) std::ofstream(out / "pool_post.bin", std::ios::binary).write(static_cast<const char*>(pool.contents), std::streamsize(pool_bytes));
    tail_fields = tf.str();
  }
  const double wall_total = -[t_all timeIntervalSinceNow] * 1000.0;
  js << "\n ],\n \"middle_gpu_ms\": " << middle_gpu_ms << ",\n \"middle_wall_ms\": " << middle_wall_ms << tail_fields << ",\n \"gpu_ms_total\": " << gpu_total << ",\n \"wall_ms_total\": " << wall_total << "\n}\n";
  if (diag) std::ofstream(out / "chain-run.json") << js.str();
  res.middle_gpu_ms = middle_gpu_ms; res.wall_ms = wall_total;
  // v3 provenance: report every library/constant identity recorded at load and re-hash every resident constant now (post-frame)
#ifdef NR_BODY_FAULT_TESTS
  if (fault == "constant-late-mutation") for (auto& kv : ctx.resident) if (kv.first.rfind("file:", 0) == 0) { static_cast<uint8_t*>(kv.second.contents)[0] ^= 1u; break; }   // after the last use of every constant: only the final re-hash can see it
#endif
  res.constants_unchanged_after_frame = true;
  for (auto& kv : ctx.resident_sha256) { res.provenance.push_back(kv); if (kv.first.rfind("file:", 0) == 0) { auto bi = ctx.resident.find(kv.first); auto ni = ctx.resident_bytes.find(kv.first); if (bi != ctx.resident.end() && ni != ctx.resident_bytes.end()) { ++res.constants_rehashed; if (Sha(bi->second.contents, ni->second) != kv.second) { res.constants_unchanged_after_frame = false; fails.push_back("model constant changed since load: " + kv.first); } } } }
  if (in.temporal && pend.seen) {   // v10: the ONLY place the temporal metadata changes: after every check of this frame, and only if none failed
    res.temporal_state_advanced = fails.empty() && pend.ok;
    if (res.temporal_state_advanced) { in.temporal->history_sha256 = pend.hist; in.temporal->last_output_sha256 = pend.out; in.temporal->frames_since_reset = in.reset ? 1u : in.temporal->frames_since_reset + 1u; } }
  if (!fails.empty()) { std::ostringstream e; e << fails.size() << " computed check(s) failed: "; for (size_t i = 0; i < fails.size() && i < 6; ++i) e << (i ? "; " : "") << fails[i]; if (fails.size() > 6) e << "; ..."; res.code = 3; res.error = e.str(); return 3; }
  res.code = 0; res.error.clear(); return 0;
} catch (const std::exception& e) { res.code = 2; res.error = e.what(); return 2; } } }

