#include "nr_experimental_encoder.h"
#import <Foundation/Foundation.h>
#include "nr_experimental_gpu_profile.h"
#include <cstdlib>
#include "nr_experimental_encoder_scope_adoption.h"
#include <array>
#include <cstring>

#if defined(NR_EXPERIMENTAL_DENSE_FUSE) && !(defined(NR_EXPERIMENTAL_ENCODER_DENSE) && defined(NR_EXPERIMENTAL_NO_VALIDATION))
#error "NR_EXPERIMENTAL_DENSE_FUSE needs NR_EXPERIMENTAL_ENCODER_DENSE and NR_EXPERIMENTAL_NO_VALIDATION: fused stages skip activate's coverage counters"
#endif

namespace nr_exp {

struct Grid { uint32_t width, height; int32_t x, y; };
struct WindowMap { uint32_t image_width, image_height, from_w, from_h, to_w, to_h; int32_t from_x, from_y, to_x, to_y; };

// Verbatim from the accepted encoder: grids per tier, and the stage tables for the C32 and wide paths.
static const std::array<Grid, 4> kGrids32   = {Grid{120, 72, 0, 0}, Grid{121, 73, -4, -4}, Grid{121, 72, -4, 0}, Grid{120, 73, 0, -4}};
static const std::array<Grid, 4> kGrids64   = {Grid{60, 36, 0, 0}, Grid{61, 37, -4, -4}, Grid{61, 36, -4, 0}, Grid{60, 37, 0, -4}};
static const std::array<Grid, 6> kGrids128  = {Grid{30, 18, 0, 0}, Grid{31, 19, -4, -4}, Grid{31, 18, -4, 0}, Grid{30, 19, 0, -4}, Grid{30, 18, 0, 0}, Grid{31, 19, -4, -4}};
static const std::array<Grid, 8> kGrids256  = {Grid{15, 9, 0, 0}, Grid{16, 10, -4, -4}, Grid{16, 9, -4, 0}, Grid{15, 10, 0, -4}, Grid{15, 9, 0, 0}, Grid{16, 10, -4, -4}, Grid{16, 9, -4, 0}, Grid{15, 10, 0, -4}};
static const std::array<unsigned, 10> kC32Elements = {128, 128, 128, 96, 64, 64, 64, 64, 32, 32};
static const std::array<unsigned, 10> kC32Bytes    = {2, 1, 2, 2, 1, 2, 1, 2, 2, 1};
static const std::array<unsigned, 10> kC32Dispatch = {128, 128, 32, 96, 2, 64, 1, 32, 32, 32};
static const char* kC32Fn[13] = {"nr_swin32_expand", "nr_swin32_activate", "nr_swin32_contract", "nr_swin32_qkv",
                                 "nr_swin32_normalize", "nr_swin32_scores", "nr_swin32_probabilities", "nr_swin32_values",
                                 "nr_swin32_project", "nr_swin32_pack", "nr_swin32_rewindow",
                                 "nr_swin32_downsample_pool", "nr_swin32_downsample_project"};
static const char* kWideFn[11] = {"nr_wide_expand", "nr_wide_activate", "nr_wide_mid", "nr_wide_mix", "nr_wide_qkv",
                                  "nr_wide_normalize", "nr_wide_scores", "nr_swin32_probabilities", "nr_wide_values",
                                  "nr_wide_project", "nr_wide_pack"};
// Two Bufs alias when they are the same allocation or share one MTLBuffer. The chain's allocator hands
// out a distinct MTLBuffer per Buf, so this is identity in practice; it is written out so that a future
// suballocating allocator makes this check fail rather than silently become a tautology.
// Which wide STAGE to capture: 0 expand, 3 mix, 4 qkv, 9 project - the four a candidate can change.
// Default 4 (qkv), so every capture recorded before this keeps its meaning.
static uint32_t qkv_capture_stage() {
  static const uint32_t st = [] {
    const char* v = std::getenv("NR_EXP_CAPTURE_QKV_STAGE");
    const unsigned long n = v ? strtoul(v, nullptr, 10) : 4ul;
    // Stage 2 is nr_wide_mid, added so a candidate that replaces it can have its OWN output captured
    // rather than inferred from a downstream stage or from the final surface (which this baseline
    // shows is insensitive to these substitutions).
    // Any stage index the shape arrays cover. Both kC32Elements/kC32Bytes (10 entries) and the wide
    // elements/bytes (11) are indexed by this, so 0..9 is always in range for either block, and every
    // engine operation can have its OWN output captured rather than inferred from a downstream stage.
    return (n <= 9ul) ? uint32_t(n) : 4u;
  }();
  return st;
}
static uint32_t qkv_capture_tier() {
  static const uint32_t t = [] {
    const char* v = std::getenv("NR_EXP_CAPTURE_QKV_TIER");
    const unsigned long n = v ? strtoul(v, nullptr, 10) : 64ul;
    // 32 selects the C32 block (block32), whose stages live in a DIFFERENT library from the wide
    // stages and have their own contracts; 64/128/256 select the wide tiers as before.
    return (n == 32ul || n == 64ul || n == 128ul || n == 256ul) ? uint32_t(n) : 64u;
  }();
  return t;
}
static bool buffers_overlap(const Buf* a, const Buf* b) {
  if (!a || !b) return false;
  if (a == b) return true;
  return a->metal != nil && a->metal == b->metal;
}
static const char* tier_library(uint32_t channels) {
  return channels == 128 ? "wide128.metallib" : channels == 256 ? "wide256.metallib" : "prefix.metallib";
}
static size_t weight_bytes_for(uint32_t block) {
  if (block <= 4)  return block == 4 ? 22720 : 20672;
  if (block <= 8)  return block == 8 ? 69936 : 61760;
  if (block <= 14) return block == 14 ? 229936 : 197184;
  return block == 22 ? 820288 : 689232;
}

Buf* Pool::get(size_t bytes) {
  const size_t idx = next++;
  if (mode == Mode::Measure) {
    if (idx >= sizes.size()) sizes.resize(idx + 1, 0);
    if (bytes > sizes[idx]) sizes[idx] = bytes;
    probes.push_back(Buf{});                       // pointer-stable placeholder; carries the logical size only
    probes.back().bytes = bytes;
    return &probes.back();
  }
  if (idx >= slots.size()) { error = "pool exhausted: the encode walk asked for slot " + std::to_string(idx) + " but prepare() reserved " + std::to_string(slots.size()); return nullptr; }
  Buf* b = &slots[idx];
  if (b->bytes < bytes) { error = "pool slot too small: reserved " + std::to_string(b->bytes) + ", encode wants " + std::to_string(bytes); return nullptr; }
  return b;
}
// Host mirror of the kernel's nr_weight_index. Kept adjacent to the prepack so the two cannot drift apart
// silently; the exhaustive read-back in prepare() is what actually proves they agree.
static inline uint32_t nr_weight_index_host(uint32_t k, uint32_t n) {
  return (n / 16u) * 512u + (n % 8u) * 64u + ((k % 16u) / 4u) * 16u +
         ((n % 16u) / 8u) * 8u + k % 4u + (k / 16u) * 4u;
}

bool Pool::allocate() {
  slots.clear(); slots.reserve(sizes.size());
  for (size_t s : sizes) {
    Buf b = make_buffer(device, s, false);
    if (!b.metal) { error = "pool allocation failed for " + std::to_string(s) + " bytes"; return false; }
    slots.push_back(b);
  }
  probes.clear(); mode = Mode::Use; rewind();
  return true;
}
size_t Pool::total_bytes() const { size_t t = 0; for (const auto& b : slots) t += b.bytes + 2 * Buf::kGuard; return t; }

namespace {
// One walk of the whole encoder chain. With `cb == nil` it only reserves pool slots (prepare); with a real
// command buffer it also encodes. Keeping one function for both guarantees the two walks cannot diverge.
struct Walk {
  id<MTLCommandBuffer> cb = nil;      // nil for the measuring/alignment walk
  id<MTLCommandBuffer> real_cb = nil; // the caller's buffer; cb is set from it only for selected tiers
  uint32_t tier_mask = 0xffffffffu;   // profiling only: which tiers actually encode (0 C32, 1 wide64, 2 C128, 3 C256, 4 exports)
  Assets* assets = nullptr;
  EncoderResources* res = nullptr;
  Pool* carry = nullptr;      // survives across blocks
  Pool* scratch = nullptr;    // rewound at the start of every block
  std::string error;
  uint32_t dispatches = 0, encoders = 0;
  static constexpr NSUInteger G = Buf::kGuard;

  // Bounded-chunk support for the background reset worker: the walk always does its full CPU bookkeeping
  // (which Buf is which, pool offsets, tier setup) but only ENCODES blocks inside [block_lo, block_hi).
  // The non-encoding path is the same one the measuring walk already uses, so resuming re-derives identical
  // state without re-encoding anything.
  uint32_t cur_block = 0, block_lo = 0, block_hi = 0xffffffffu;
  bool encoding() const { return cb != nil && cur_block >= block_lo && cur_block < block_hi; }
  id<MTLComputePipelineState> ps(const char* lib, const char* fn) {
    if (!encoding()) return nil;
    std::string e; auto p = assets->pipeline(lib, fn, e);
    if (!p) error = "pipeline unavailable: " + e;
    return p;
  }
  // Blocks 1-22 issued one dispatch per compute encoder: 288 encoders for 288 dispatches, and the
  // per-encoder cost dominated the phase. Metal serialises dispatches inside one encoder with implicit
  // barriers (MTLDispatchTypeSerial, the default), so consecutive dependent dispatches can share an
  // encoder with identical ordering and identical results. Per-encoder GPU timestamps need one encoder
  // each, so batching is off while profiling and can be forced off with NR_EXP_ENCODER_BATCH=0.
  id<MTLComputeCommandEncoder> open_enc = nil;
  static bool batching() {
    static const bool on = [] {
      const char* v = std::getenv("NR_EXP_ENCODER_BATCH");
      if (v && v[0] == '0') return false;
      // Only PER-DISPATCH profiling needs one encoder per dispatch. Regions mode samples the encoders
      // batching already creates, so batching stays on and the frame executes as it normally would.
      return !nr_exp::gpu_profiling_per_dispatch();
    }();
    return on;
  }
  id<MTLComputeCommandEncoder> begin(const char* label) {
    if (batching() && open_enc != nil) return open_enc;
    id<MTLComputeCommandEncoder> enc = nr_exp::gpu_profiler().compute(cb, label); nr_exp::nr_scope_adopt(enc);
    if (!enc) { error = "compute encoder unavailable"; return nil; }
    enc.label = [NSString stringWithUTF8String:label];
    ++encoders;
    if (batching()) open_enc = enc;
    return enc;
  }
  // Threadgroup width for the encoder chain. Default 64, the accepted geometry; NR_EXP_WIDE_TG sweeps it.
  //
  // An older comment here asserted the width was "a free dispatch parameter" because the wide kernels
  // report staticThreadgroupMemory 0. That was an observation about the ACCEPTED kernels at one point in
  // time, not a property of the dispatch, and it is false for any substituted kernel that holds
  // threadgroup state - the nr_wide_qkv hoist is exactly such a kernel. It is not treated as a measured
  // fact any more: the admission check below decides what is allowed, and any substitution pins 64.
  static size_t chain_threadgroup() {
    static const size_t w = [] {
      const char* v = std::getenv("NR_EXP_WIDE_TG");
      if (!v) return size_t(64);
      const long n = strtol(v, nullptr, 10);
      if (n < 32 || n > 1024 || (n % 32) != 0) return size_t(64);   // must be a whole number of simdgroups
      return size_t(n);
    }();
    return w;
  }
  unsigned long long bound_bytes = 0, bound_bytes_wide = 0, bound_bytes_coverage = 0;
  void account(unsigned long long data, unsigned long long coverage, bool wide) {
    bound_bytes += data + coverage; bound_bytes_coverage += coverage;
    if (wide) bound_bytes_wide += data + coverage;
  }
  void dispatch(id<MTLComputeCommandEncoder> enc, size_t threads) {
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(chain_threadgroup(), 1, 1)];
    if (!batching()) nr_exp::nr_scope_end(enc);
    ++dispatches;
  }
  void flush() { if (open_enc != nil) { nr_exp::nr_scope_end(open_enc); open_enc = nil; } }

#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  // ---- ENCODER-DENSE-WS8 (build/nr-accept-v1/experiments/ENCODER-DENSE-WS8-TIER-GATE-v1.json) ----
  struct NrdCountHost { uint32_t count; };
  struct NrdSeedHost { uint32_t rows, kind; };
  struct NrdLinearHost { uint32_t rows, out_width, in_stride, chunks, kind, seeded; };
  bool dense_tier(uint32_t channels) const {
    const uint32_t bit = channels == 64 ? 2u : channels == 128 ? 4u : channels == 256 ? 8u : 0u;
    return assets->encdense_active && (assets->encdense_tier_mask & bit) != 0;
  }
  static const char* dense_library(uint32_t channels) {
    return channels == 64 ? "enc_dense64.metallib" : channels == 128 ? "enc_dense128.metallib" : "enc_dense256.metallib";
  }
  // Wide stage i (0 expand, 2 mid, 3 mix, 4 qkv, 9 project) on the dense kernels, appended to the batched encoder:
  // operand preparation into the shared operand buffer of its width (decode, or quantize + dense-permute), the residual
  // seed when the stage is seeded, then the weight-shared x8 linear kernel over (out_width/8, rows/64) groups of 32.
  // The inputs are the engine stage's own: the same operand, the same original bytes (mix) and ffn (project) for the
  // seeds, and the same output buffer.
  bool dense_stage(id<MTLComputeCommandEncoder> enc, size_t i, Buf* x, Buf* w, const std::array<Buf*, 11>& y,
                   Buf* XW, Buf* XH, Buf* SD, uint32_t rows, uint32_t channels) {
    const uint32_t hidden = 4u * channels, heads = channels / 32u;
    const uint32_t kind = i == 0 ? 0u : i == 2 ? 1u : i == 3 ? 2u : i == 4 ? 3u : 4u;
    const uint32_t out_width = kind == 0 ? hidden : kind == 3 ? 3u * channels : channels;
    const uint32_t in_stride = kind == 1 ? hidden : channels;
    const uint32_t seeded = (kind == 2 || kind == 4) ? 1u : 0u;
    Buf* src = kind == 0 ? x : kind == 1 ? y[1] : kind == 2 ? y[2] : kind == 3 ? y[3] : y[8];
    Buf* XA = kind == 1 ? XH : XW;
    const size_t bi = size_t(cur_block) - 1;
    if (bi >= res->encdense_tables.size() || res->encdense_tables[bi].size() != 5 || res->encdense_tables[bi][kind].metal == nil) {
      error = "encoder dense: block " + std::to_string(cur_block) + " has no prepacked weight tables"; return false; }
    const char* dl = dense_library(channels);
    auto pprep = ps(dl, kind <= 1 ? "nrd_decode" : "nrd_quant_perm"); if (!pprep) return false;
    auto plin = ps(dl, "nrd_linear_ws8"); if (!plin) return false;
    const NrdCountHost count{rows * in_stride};
    [enc setComputePipelineState:pprep];
    [enc setBuffer:src->metal offset:G atIndex:0]; [enc setBuffer:XA->metal offset:G atIndex:1];
    [enc setBytes:&count length:sizeof count atIndex:2];
    dispatch(enc, size_t(count.count));
    if (seeded) {
      auto pseed = ps(dl, "nrd_seed"); if (!pseed) return false;
      const NrdSeedHost sp{rows, kind == 2 ? 0u : 1u};
      [enc setComputePipelineState:pseed];
      [enc setBuffer:x->metal offset:G atIndex:0]; [enc setBuffer:y[3]->metal offset:G atIndex:1]; [enc setBuffer:w->metal offset:G atIndex:2];
      [enc setBuffer:SD->metal offset:G atIndex:3]; [enc setBytes:&sp length:sizeof sp atIndex:4];
      dispatch(enc, size_t(rows) * channels);
    }
    const NrdLinearHost lp{rows, out_width, in_stride, kind == 1 ? 4u : heads, kind, seeded};
    [enc setComputePipelineState:plin];
    [enc setBuffer:XA->metal offset:G atIndex:0]; [enc setBuffer:res->encdense_tables[bi][kind].metal offset:G atIndex:1];
    [enc setBuffer:SD->metal offset:G atIndex:2]; [enc setBuffer:y[i]->metal offset:G atIndex:3];
    [enc setBytes:&lp length:sizeof lp atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(out_width / 8, rows / 64, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ++dispatches;
    return true;
  }
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
  // ---- DENSE-MLP-FUSION-v1 (experiments/WIDE-MLP-FUSION-GATE-v1.json): wide stage 0 as expand's operand decode plus ONE
  // nrd_expand_activate_ws8 dispatch that writes the activation bytes nr_wide_activate would write into y[1]. y[0] is not written and
  // activate does not run; the arithmetic is identical by construction and was bit-identical on every gated launch.
  bool dense_fused_stage(id<MTLComputeCommandEncoder> enc, Buf* x, const std::array<Buf*, 11>& y, Buf* XW, uint32_t rows, uint32_t channels) {
    const size_t bi = size_t(cur_block) - 1;
    if (bi >= res->encdense_tables.size() || res->encdense_tables[bi].size() != 5 || res->encdense_tables[bi][0].metal == nil) {
      error = "encoder dense (fused): block " + std::to_string(cur_block) + " has no prepacked expand table"; return false; }
    const char* dl = dense_library(channels);
    auto pdec = ps(dl, "nrd_decode"); if (!pdec) return false;
    auto pfz = ps(dl, "nrd_expand_activate_ws8"); if (!pfz) return false;
    const NrdCountHost count{rows * channels};
    [enc setComputePipelineState:pdec];
    [enc setBuffer:x->metal offset:G atIndex:0]; [enc setBuffer:XW->metal offset:G atIndex:1];
    [enc setBytes:&count length:sizeof count atIndex:2];
    dispatch(enc, size_t(count.count));
    [enc setComputePipelineState:pfz];
    [enc setBuffer:XW->metal offset:G atIndex:0]; [enc setBuffer:res->encdense_tables[bi][0].metal offset:G atIndex:1];
    [enc setBuffer:y[1]->metal offset:G atIndex:2]; [enc setBytes:&rows length:sizeof rows atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((4u * channels) / 8, rows / 64, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ++dispatches;
    return true;
  }
#endif
#endif

  Buf* rewindow(const char* lib, Buf* src, Grid from, Grid to, uint32_t channels, uint32_t width, uint32_t height) {
    const size_t size = size_t(to.width) * to.height * 64 * channels;
    Buf* out = carry->get(size); if (!out) { error = carry->error; return nullptr; }
    Buf* counts = scratch->get(size * 4); if (!counts) { error = scratch->error; return nullptr; }
    if (!encoding()) return out;
    if (src->bytes != size_t(from.width) * from.height * 64 * channels) { error = "rewindow source extent"; return nullptr; }
    auto p = ps(lib, channels == 32 ? "nr_swin32_rewindow" : "nr_wide_rewindow"); if (!p) return nullptr;
    WindowMap shape{width, height, from.width, from.height, to.width, to.height, from.x, from.y, to.x, to.y};
    auto enc = begin("nr_exp.rewindow"); if (!enc) return nullptr;
    [enc setComputePipelineState:p];
    [enc setBuffer:src->metal offset:G atIndex:0]; [enc setBuffer:out->metal offset:G atIndex:1];
    [enc setBuffer:counts->metal offset:G atIndex:2]; [enc setBytes:&shape length:sizeof(shape) atIndex:3];
    dispatch(enc, size);
    return out;
  }

#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  // ---- ENCODER-DENSE32-HYBRID (tier "32"; build/nr-accept-v1/experiments/ENCODER-DENSE32-HYBRID-GATE-v1.json) ----
  // C32 stage i (0 expand, 3 qkv) on the gated dense kernels, appended to the batched encoder: operand decode (expand: the block
  // input bytes) or quantize + dense-permute (qkv: the contract's chunk-3 partial) into the shared rows*32 float buffer, then the
  // weight-shared x8 kernel with the prepacked table, writing the engine stage's own output buffer.
  struct Nrd32LinearHost { uint32_t rows, out_width, in_stride, chunks, windowed, seed_mode, direct, partial, wstride, woff; };
  bool dense32_on() const { return assets->encdense_active && (assets->encdense_tier_mask & 1u) != 0; }
  bool dense32_stage(id<MTLComputeCommandEncoder> enc, size_t i, Buf* input, const std::array<Buf*, 10>& y, Buf* X32, uint32_t rows) {
    const size_t bi = size_t(cur_block) - 1, table = i == 0 ? 0 : 1;
    if (bi >= res->encdense32_tables.size() || res->encdense32_tables[bi].size() != 2 || res->encdense32_tables[bi][table].metal == nil) {
      error = "encoder dense (32): block " + std::to_string(cur_block) + " has no prepacked tables"; return false; }
    auto pprep = ps("enc_dense32.metallib", i == 0 ? "nrd32w_decode" : "nrd32w_quant_perm"); if (!pprep) return false;
    auto plin = ps("enc_dense32.metallib", "nrd32w_linear_ws8"); if (!plin) return false;
    const uint32_t count = rows * 32u;
    [enc setComputePipelineState:pprep];
    if (i == 0) [enc setBuffer:input->metal offset:G atIndex:0];
    else        [enc setBuffer:y[2]->metal offset:G + size_t(3) * rows * 32 * 2 atIndex:0];   // the chunk-3 contract partial
    [enc setBuffer:X32->metal offset:G atIndex:1]; [enc setBytes:&count length:sizeof count atIndex:2];
    dispatch(enc, size_t(count));
    const Nrd32LinearHost lp{rows, i == 0 ? 128u : 96u, 32u, 1u, 0u, 0u, 1u, 0u, 32u, 0u};
    [enc setComputePipelineState:plin];
    [enc setBuffer:X32->metal offset:G atIndex:0]; [enc setBuffer:res->encdense32_tables[bi][table].metal offset:G atIndex:1];
    [enc setBuffer:X32->metal offset:G atIndex:2];   // seed slot: never read (seed_mode 0), bound so every declared buffer is set
    [enc setBuffer:y[i]->metal offset:G atIndex:3]; [enc setBytes:&lp length:sizeof lp atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(lp.out_width / 8, rows / 64, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ++dispatches;
    return true;
  }
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
  // ---- DENSE-MLP-FUSION-v1 (experiments/C32-MLP-FUSION-GATE-v1.json): walk C32 stage 0 as expand's operand decode plus ONE
  // nrd32w_expand_activate_ws8 dispatch that writes nr_swin32_activate's bytes into y[1]; y[0] is not written and activate does not run.
  bool dense32_fused_stage(id<MTLComputeCommandEncoder> enc, Buf* input, const std::array<Buf*, 10>& y, Buf* X32, uint32_t rows) {
    const size_t bi = size_t(cur_block) - 1;
    if (bi >= res->encdense32_tables.size() || res->encdense32_tables[bi].size() != 2 || res->encdense32_tables[bi][0].metal == nil) {
      error = "encoder dense (32, fused): block " + std::to_string(cur_block) + " has no prepacked expand table"; return false; }
    auto pdec = ps("enc_dense32.metallib", "nrd32w_decode"); if (!pdec) return false;
    auto pfz = ps("enc_dense32.metallib", "nrd32w_expand_activate_ws8"); if (!pfz) return false;
    const uint32_t count = rows * 32u;
    [enc setComputePipelineState:pdec];
    [enc setBuffer:input->metal offset:G atIndex:0]; [enc setBuffer:X32->metal offset:G atIndex:1]; [enc setBytes:&count length:sizeof count atIndex:2];
    dispatch(enc, size_t(count));
    [enc setComputePipelineState:pfz];
    [enc setBuffer:X32->metal offset:G atIndex:0]; [enc setBuffer:res->encdense32_tables[bi][0].metal offset:G atIndex:1];
    [enc setBuffer:y[1]->metal offset:G atIndex:2]; [enc setBytes:&rows length:sizeof rows atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(128u / 8, rows / 64, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ++dispatches;
    return true;
  }
#endif
#endif

  // C32 block: ten stages, output y[9], optional projection y[8].
  Buf* block32(Buf* input, Buf* weight, uint32_t rows, Buf** projection, size_t weight_block_index = SIZE_MAX) {
    scratch->rewind();
    std::array<Buf*, 10> y{}, c{};
    for (size_t i = 0; i < 10; ++i) {
      const size_t n = size_t(rows) * kC32Elements[i] * kC32Bytes[i];
      const bool carried = (i == 9) || (i == 8 && projection);   // the block output, and a projection a downsample consumes
      y[i] = carried ? carry->get(n) : scratch->get(n);
      c[i] = scratch->get(size_t(rows) * kC32Elements[i] * 4);
      if (!y[i] || !c[i]) { error = carried ? carry->error : scratch->error; if (error.empty()) error = scratch->error; return nullptr; }
    }
    if (projection) *projection = y[8];
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
    // Hybrid dense expand/qkv: one float operand buffer of rows*32, reserved by the measuring walk too, so the pools agree.
    const bool dense32 = dense32_on();
    Buf* X32 = nullptr;
    if (dense32) { X32 = scratch->get(size_t(rows) * 32 * 4); if (!X32) { error = scratch->error; return nullptr; } }
#endif
    // C32 capture extents, recorded from the REAL extent here and before the measuring walk returns,
    // exactly as the wide block does. kC32Elements/kC32Bytes are this block's OWN contract, not the
    // wide array - the two blocks share stage NAMES, not shapes.
    if (qkv_capture_tier() == 32u && res->qkv_capture_needed == 0) {
      const uint32_t cs = qkv_capture_stage() < 10u ? qkv_capture_stage() : 0u;
      res->qkv_capture_needed    = size_t(rows) * kC32Elements[cs] * kC32Bytes[cs];
      res->qkv_capture_in_bytes  = size_t(rows) * kC32Elements[cs ? cs - 1u : 0u] * kC32Bytes[cs ? cs - 1u : 0u];
      res->qkv_capture_cov_bytes = size_t(rows) * kC32Elements[cs] * 4;
    }
    if (!encoding()) return y[9];
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
    // The hybrid stages append two dispatches to ONE batched encoder and were gated only at the accepted 64-thread geometry.
    if (dense32 && (chain_threadgroup() != 64 || !batching() || (rows % 64u) != 0)) {
      error = "encoder dense (32): the hybrid C32 stages are admitted only at the accepted 64-thread geometry, with encoder batching on "
              "and rows a multiple of 64; refusing before any GPU work"; return nullptr; }
#endif
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
    const bool fuse32 = dense32 && (assets->encdense_fuse_mask & 1u) != 0;
    if (fuse32 && qkv_capture_tier() == 32u && (qkv_capture_stage() < 10u ? qkv_capture_stage() : 0u) <= 1u &&
        (res->qkv_capture.metal != nil || res->qkv_capture_cov_pre.metal != nil)) {
      error = "encoder dense (32, fused): expand+activate are fused, so stage 0 writes no output and activate does not run; capture those stages "
              "with the unfused package. Refusing before any GPU work."; return nullptr; }
#endif
    for (size_t i = 0; i < 10; ++i) {
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
      if (fuse32 && i == 1) continue;   // nr_swin32_activate: its bytes were written by the fused stage 0 dispatch
#endif
      auto p = ps("prefix.metallib", kC32Fn[i]); if (!p) return nullptr;
      const uint32_t cs32 = qkv_capture_stage() < 10u ? qkv_capture_stage() : 0u;
      if (qkv_capture_tier() == 32u && uint32_t(i) == cs32 && res->qkv_capture_cov_pre.metal != nil && !res->qkv_capture_done) {
        const size_t want_cov = size_t(rows) * kC32Elements[cs32] * 4;
        if (want_cov > res->qkv_capture_cov_pre.bytes) { error = "c32 capture: coverage extent does not fit the pre buffer"; return nullptr; }
        flush();
        id<MTLBlitCommandEncoder> pb = [cb blitCommandEncoder];
        if (!pb) { error = "c32 capture: pre-coverage blit unavailable"; return nullptr; }
        nr_exp::nr_scope_adopt(pb);
        [pb copyFromBuffer:c[cs32]->metal sourceOffset:G toBuffer:res->qkv_capture_cov_pre.metal destinationOffset:G size:want_cov];
        if (const char* pv = std::getenv("NR_EXP_CAPTURE_POISON")) {   // PRODUCER poison on the stage's own output
          // The extent is the PRODUCER SLOT's own data region, not the formula. block32 rewinds its scratch
          // and is called four times with different grids, so a slot's data region is not always this call's
          // rows*elements*bytes. Poisoning by the formula overran the slot and destroyed the trailing guard -
          // which is exactly what "256 of 512 wrong in every arm, trailing band reading 0x01" was reporting:
          // my own poison, not a guard violation.
          const size_t poison_bytes = std::min(size_t(rows) * kC32Elements[cs32] * kC32Bytes[cs32], y[cs32]->bytes);
          const uint8_t poison = uint8_t(strtoul(pv, nullptr, 0));
          [pb fillBuffer:y[cs32]->metal range:NSMakeRange(G, poison_bytes) value:poison];
        }
        nr_exp::nr_scope_end(pb);
      }
      auto enc = begin("nr_exp.block32"); if (!enc) return nullptr;
      [enc setComputePipelineState:p];
      auto bind = [&](Buf* b, NSUInteger j) { [enc setBuffer:b->metal offset:G atIndex:j]; };
      auto shape = [&](NSUInteger j) { [enc setBytes:&rows length:4 atIndex:j]; };
      // Stage 0 reads the PREPACKED weights when that layout is active and this build's native expand is the
      // selected kernel. Every other stage keeps the original buffer, so the derived layout is scoped to the one
      // kernel that was written to consume it.
      Buf* w0 = weight;
      if (res->prepack_active && weight_block_index < res->prepacked_weights.size() &&
          res->prepacked_weights[weight_block_index].metal != nil) w0 = &res->prepacked_weights[weight_block_index];
      if (i == 0) { bind(input, 0); bind(w0, 1); bind(y[0], 2); bind(c[0], 3); shape(4); }
      if (i == 1) { bind(y[0], 0); bind(y[1], 1); bind(c[1], 2); shape(3); }
      if (i == 2) { bind(input, 0); bind(y[1], 1); bind(weight, 2); bind(y[2], 3); bind(c[2], 4); shape(5); }
      if (i == 3) { bind(y[2], 0); bind(weight, 1); bind(y[3], 2); bind(c[3], 3); shape(4); }
      if (i == 4) { bind(y[3], 0); bind(weight, 1); bind(y[4], 2); bind(c[4], 3); shape(4); }
      if (i == 5) { bind(y[4], 0); bind(weight, 1); bind(y[5], 2); bind(c[5], 3); shape(4); }
      if (i == 6) { bind(y[5], 0); bind(y[6], 1); bind(c[6], 2); shape(3); }
      if (i == 7) { bind(y[6], 0); bind(y[3], 1); bind(y[7], 2); bind(c[7], 3); shape(4); }
      if (i == 8) { bind(y[7], 0); bind(y[2], 1); bind(weight, 2); bind(y[8], 3); bind(c[8], 4); shape(5); }
      if (i == 9) { bind(y[8], 0); bind(y[9], 1); bind(c[9], 2); shape(3); }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
      if (fuse32 && i == 0) {
        if (!dense32_fused_stage(enc, input, y, X32, rows)) return nullptr;
      } else
#endif
      if (dense32 && (i == 0 || i == 3)) {
        if (!dense32_stage(enc, i, input, y, X32, rows)) return nullptr;
      } else
#endif
      dispatch(enc, size_t(rows) * kC32Dispatch[i]);
      if (qkv_capture_tier() == 32u && uint32_t(i) == cs32 && res->qkv_capture.metal != nil && !res->qkv_capture_done) {
        const size_t want     = std::min(size_t(rows) * kC32Elements[cs32] * kC32Bytes[cs32], y[cs32]->bytes);
        const size_t want_in  = size_t(rows) * kC32Elements[cs32 ? cs32 - 1u : 0u] * kC32Bytes[cs32 ? cs32 - 1u : 0u];
        const size_t want_cov = size_t(rows) * kC32Elements[cs32] * 4;
        if (want == 0 || want > res->qkv_capture.bytes || want_in > res->qkv_capture_in.bytes || want_cov > res->qkv_capture_cov.bytes) {
          error = "c32 capture: extent does not fit; refusing a partial capture"; return nullptr; }
        flush();
        id<MTLBlitCommandEncoder> ob = [cb blitCommandEncoder];
        if (!ob) { error = "c32 capture: output blit unavailable"; return nullptr; }
        nr_exp::nr_scope_adopt(ob);
        [ob copyFromBuffer:y[cs32]->metal sourceOffset:G toBuffer:res->qkv_capture.metal destinationOffset:G size:want];
        [ob copyFromBuffer:(cs32 ? y[cs32 - 1u] : input)->metal sourceOffset:G toBuffer:res->qkv_capture_in.metal destinationOffset:G size:want_in];
        [ob copyFromBuffer:c[cs32]->metal sourceOffset:G toBuffer:res->qkv_capture_cov.metal destinationOffset:G size:want_cov];
        // The PRODUCER buffer's OWN guard bands, snapshotted after its dispatch and before any reuse.
        // The trailing band sits at G + y[]->bytes, NOT at G + want: the pool allocates each slot at the
        // measured maximum, so using the requested extent reads inside the slot and reports every byte
        // wrong. That exact mistake was made once already in this campaign.
        // The DESTINATION offsets are data() offsets - G and G + kGuard - because the reader takes
        // qkv_capture_guards.data(), which is contents + kGuard. Writing at raw 0 and kGuard put the two
        // bands one slot low, so the reader saw one real band plus untouched creation poison and reported
        // 256 of 512 wrong in every arm. The SOURCE offsets are raw: the producer's own bands are at 0 and
        // at G + its slot size.
        [ob copyFromBuffer:y[cs32]->metal sourceOffset:0 toBuffer:res->qkv_capture_guards.metal destinationOffset:G size:Buf::kGuard];
        [ob copyFromBuffer:y[cs32]->metal sourceOffset:G + y[cs32]->bytes toBuffer:res->qkv_capture_guards.metal
                    destinationOffset:G + Buf::kGuard size:Buf::kGuard];
        nr_exp::nr_scope_end(ob);
        res->qkv_capture_done = true;
        res->qkv_capture_bytes = want;   // in/cov extents were already recorded when the capture was sized
      }
    }
    return y[9];
  }

  // Wide block (64/128/256 channels): eleven stages, output y[10], optional projection y[9].
  Buf* block_wide(Buf* x, Buf* w, uint32_t rows, uint32_t channels, Buf** projection) {
    const uint32_t heads = channels / 32, hidden = channels * 4;
    const std::array<unsigned, 11> elements = {hidden, hidden, channels, channels, channels * 3, channels * 2, heads * 64, heads * 64, channels, channels, channels};
    const std::array<unsigned, 11> bytes    = {2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1};
    const std::array<unsigned, 11> disp     = {hidden, hidden, channels, channels, channels * 3, heads * 2, heads * 64, heads, channels, channels, channels};
    scratch->rewind();
    std::array<Buf*, 11> y{}, c{};
    for (size_t i = 0; i < 11; ++i) {
      const size_t n = size_t(rows) * elements[i] * bytes[i];
      const bool carried = (i == 10) || (i == 9 && projection);
      y[i] = carried ? carry->get(n) : scratch->get(n);
      c[i] = scratch->get(size_t(rows) * elements[i] * 4);
      if (!y[i] || !c[i]) { error = carried ? carry->error : scratch->error; if (error.empty()) error = scratch->error; return nullptr; }
    }
    if (projection) *projection = y[9];
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
    // Dense stages: one float operand buffer per width and one half seed buffer, reused stage to stage inside this
    // block (the dispatches of one encoder run in order). Reserved by the measuring walk too, so the pools agree.
    const bool dense = dense_tier(channels);
    Buf* XW = nullptr; Buf* XH = nullptr; Buf* SD = nullptr;
    if (dense) {
      XW = scratch->get(size_t(rows) * channels * 4); XH = scratch->get(size_t(rows) * hidden * 4); SD = scratch->get(size_t(rows) * channels * 2);
      if (!XW || !XH || !SD) { error = scratch->error; return nullptr; }
    }
#endif
    // The extent the opt-in C64 capture needs, recorded from the REAL extent here rather than guessed.
    // It must be before the encoding() early return, because the measuring walk - the walk that runs
    // before allocation - takes that return and never reaches the stage loop below.
    if (channels == qkv_capture_tier() && res->qkv_capture_needed == 0) {
      res->qkv_capture_needed = size_t(rows) * elements[qkv_capture_stage()] * bytes[qkv_capture_stage()];
      res->qkv_capture_in_bytes = size_t(rows) * elements[qkv_capture_stage() ? qkv_capture_stage()-1 : 0] * bytes[qkv_capture_stage() ? qkv_capture_stage()-1 : 0];
      res->qkv_capture_cov_bytes = size_t(rows) * elements[qkv_capture_stage()] * 4;
    }
    if (!encoding()) return y[10];
    if (x->bytes != size_t(rows) * channels || !rows || rows % 64) { error = "wide extent mismatch"; return nullptr; }
    // ---- HOST FAIL-CLOSED ADMISSION for this wide block ----
    // ACCURATE SCOPE (Codex review): this refuses THIS block's dispatches before any of them is
    // encoded, and prepare() already refuses an unsupported width before a frame starts. It does NOT
    // mean the caller's command buffer is empty - entry layout, rewindowing and earlier blocks can
    // already have encoded work into it. Returning nullptr here propagates to the encode failure path,
    // where w.flush() closes the batched encoder and the caller's DISCARD-ONLY contract applies: NR
    // output is discarded, the caller's buffer stays valid for submission, and nothing here shortens
    // the lifetime of anything the caller owns.
    // A substituted (unpinned) kernel library may hold threadgroup state that the ACCEPTED kernels do
    // not - the qkv x-quantisation hoist is the live example: it stages a row into threadgroup memory,
    // so a threadgroup must not straddle a row, which needs (3*channels) % width == 0. The accepted
    // kernels tolerate any width, so this was previously left to a comment.
    //
    // Refusing inside the kernel is NOT good enough. Research validation is removed in this lane, so a
    // kernel that returns early leaves y unwritten, and poisoned or stale values can become finite
    // later; an offline hash test would catch it but a live frame would not fail closed. So the check
    // belongs here, on the host, with an explicit error and nothing encoded. (Codex, 07:36.)
    {
      const size_t tg = chain_threadgroup();
      const size_t lanes = size_t(channels) * 3;
      if (tg == 0 || tg > 1024 || (tg % 32) != 0) {
        error = "wide dispatch admission: threadgroup width " + std::to_string(tg) +
                " is not a whole number of 32-lane simdgroups in 1..1024"; return nullptr;
      }
      if ((lanes % tg) != 0) {
        error = "wide dispatch admission: threadgroup width " + std::to_string(tg) + " does not divide 3*channels = " +
                std::to_string(lanes) + ", so a threadgroup would straddle a row; a substituted kernel that stages a row "
                "in threadgroup memory would read the wrong row. Refusing before any GPU work.";
        return nullptr;
      }
      // (2) EITHER form of substitution pins the width. unpinned_libraries alone was not enough: it is
      // FALSE when every base library matches its pin and only the per-kernel override is active, which
      // is exactly the configuration that was benchmarked. A substituted kernel may hold threadgroup
      // state, so 64 - the accepted geometry, and the only width this candidate's contract admits - is
      // the only width allowed with either.
      if ((assets->unpinned_libraries || assets->qkv_override_active) && tg != 64) {
        error = std::string("wide dispatch admission: a substituted kernel is loaded (") +
                (assets->unpinned_libraries ? "unpinned libraries" : "per-kernel nr_wide_qkv override") +
                "), so the threadgroup width is fixed at the accepted geometry of 64; refusing the requested width " +
                std::to_string(tg) + ". Do not combine a width sweep with a substituted kernel.";
        return nullptr;
      }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
      // The dense stages append several dispatches to ONE batched encoder, so per-dispatch profiling (one encoder per
      // dispatch) cannot carry them, and they were proven only at the accepted 64-thread geometry of the other stages.
      // A tier whose override declares the mma8x8-rows8-v1 mapping dispatches past the dense branch, so it is refused too.
      if (dense) {
        const auto gd = assets->qkv_override_dispatch.find(tier_library(channels));
        if (tg != 64 || !batching() || (gd != assets->qkv_override_dispatch.end() && gd->second == "mma8x8-rows8-v1")) {
          error = "encoder dense: admitted only at the accepted 64-thread geometry, with encoder batching on and no "
                  "mma8x8-rows8-v1 declaration for this tier; refusing before any GPU work"; return nullptr; }
      }
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
      if (dense && (assets->encdense_fuse_mask & (channels == 64 ? 2u : channels == 128 ? 4u : 8u)) != 0 && channels == qkv_capture_tier() &&
          qkv_capture_stage() <= 1u && (res->qkv_capture.metal != nil || res->qkv_capture_cov_pre.metal != nil)) {
        error = "encoder dense (fused): expand+activate are fused at this tier, so stage 0 writes no output and activate does not run; capture "
                "those stages with the unfused package. Refusing before any GPU work."; return nullptr; }
#endif
#endif
      // x and y must not alias. The hoist reads the whole row into threadgroup memory BEFORE any thread
      // writes y, where the accepted kernel interleaved the two, so an overlap would change results.
      // Stage 4 is x = y[3] -> y = y[4]; check every stage's in/out pair that this walk binds.
      const std::array<std::pair<Buf*, Buf*>, 4> io = {{ {x, y[0]}, {y[3], y[4]}, {y[7], y[8]}, {y[8], y[9]} }};
      for (const auto& pr : io) {
        if (!pr.first || !pr.second) continue;
        if (pr.first == pr.second || buffers_overlap(pr.first, pr.second)) {
          error = "wide dispatch admission: an input and its output alias, which a substituted kernel that stages a row "
                  "before writing would compute differently. Refusing before any GPU work."; return nullptr;
        }
      }
    }
    const char* lib = tier_library(channels);
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
    const bool fusew = dense && (assets->encdense_fuse_mask & (channels == 64 ? 2u : channels == 128 ? 4u : 8u)) != 0;
#endif
    for (size_t i = 0; i < 11; ++i) {
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
      if (fusew && i == 1) continue;   // nr_wide_activate: its bytes were written by the fused stage 0 dispatch
#endif
      auto p = ps(lib, kWideFn[i]); if (!p) return nullptr;
      // ATTRIBUTION: label each wide encoder with its TIER. The profiler groups by label, so with a
      // bare "nr_wide_qkv" the three tiers were summed into one number and the per-tier cost could
      // only be guessed at by ablation - which Codex correctly rejected, because removing a kernel's
      // arithmetic changes what every downstream stage reads, so the difference is not that kernel's
      // cost. Labelling the tiers measures them directly instead.
      // Coverage BEFORE stage 4 runs. It must be copied before begin() opens stage 4's encoder: the
      // counters are cumulative across scratch reuse and are never cleared, so an absolute count says
      // nothing and only (after - before) shows exact-once. Copying it after begin() would flush the
      // very encoder the stage is about to dispatch into.
      if (uint32_t(i) == qkv_capture_stage() && channels == qkv_capture_tier() && res->qkv_capture_cov_pre.metal != nil && !res->qkv_capture_done) {
        const size_t want_cov = size_t(rows) * elements[qkv_capture_stage()] * 4;
        if (want_cov > res->qkv_capture_cov_pre.bytes) {
          error = "qkv capture: coverage extent does not fit the pre buffer"; return nullptr; }
        flush();
        id<MTLBlitCommandEncoder> pb = [cb blitCommandEncoder];
        if (!pb) { error = "qkv capture: pre-coverage blit unavailable"; return nullptr; }
        nr_exp::nr_scope_adopt(pb);
        [pb copyFromBuffer:c[qkv_capture_stage()]->metal sourceOffset:G toBuffer:res->qkv_capture_cov_pre.metal destinationOffset:G size:want_cov];
        // PRODUCER poison: fill the stage's own output range - NOT the capture destination, and never
        // the guard bands - so anything the kernel fails to write stays poison. With two different
        // poison values, an identical captured group across both runs is what shows every value came
        // from the kernel. The input operand is not touched.
        if (const char* pv = std::getenv("NR_EXP_CAPTURE_POISON")) {
          const uint8_t poison = uint8_t(strtoul(pv, nullptr, 0));
          [pb fillBuffer:y[qkv_capture_stage()]->metal range:NSMakeRange(G, size_t(rows) * elements[qkv_capture_stage()] * bytes[qkv_capture_stage()]) value:poison];
        }
        nr_exp::nr_scope_end(pb);
      }
      char wlabel[64];
      snprintf(wlabel, sizeof(wlabel), "%s.c%u", kWideFn[i], channels);
      auto enc = begin(wlabel); if (!enc) return nullptr;
      [enc setComputePipelineState:p];
      auto bind = [&](Buf* b, NSUInteger j) { [enc setBuffer:b->metal offset:G atIndex:j]; };
      uint32_t shape_value = rows * (i == 7 ? heads : 1);
      auto shape = [&](NSUInteger j) { [enc setBytes:&shape_value length:4 atIndex:j]; };
      if (i == 0)  { bind(x, 0); bind(w, 1); bind(y[0], 2); bind(c[0], 3); shape(4); }
      if (i == 1)  { bind(y[0], 0); bind(y[1], 1); bind(c[1], 2); shape(3); }
      if (i == 2)  { bind(y[1], 0); bind(w, 1); bind(y[2], 2); bind(c[2], 3); shape(4); }
      if (i == 3)  { bind(x, 0); bind(y[2], 1); bind(w, 2); bind(y[3], 3); bind(c[3], 4); shape(5); }
      if (i == 4)  { bind(y[3], 0); bind(w, 1); bind(y[4], 2); bind(c[4], 3); shape(4); }
      if (i == 5)  { bind(y[4], 0); bind(w, 1); bind(y[5], 2); bind(c[5], 3); shape(4); }
      if (i == 6)  { bind(y[5], 0); bind(w, 1); bind(y[6], 2); bind(c[6], 3); shape(4); }
      if (i == 7)  { bind(y[6], 0); bind(y[7], 1); bind(c[7], 2); shape(3); }
      if (i == 8)  { bind(y[7], 0); bind(y[4], 1); bind(y[8], 2); bind(c[8], 3); shape(4); }
      if (i == 9)  { bind(y[8], 0); bind(y[3], 1); bind(w, 2); bind(y[9], 3); bind(c[9], 4); shape(5); }
      if (i == 10) { bind(y[9], 0); bind(y[10], 1); bind(c[10], 2); shape(3); }
      account(static_cast<unsigned long long>(rows) * elements[i] * bytes[i],
              static_cast<unsigned long long>(rows) * elements[i] * 4ull, true);
      // The THREAD COUNT IS NEVER CHANGED. Codex's review found that the accepted dispatch already has
      // exactly as many 64-thread groups as there are 8x8 output tiles: rows*3*channels/64 groups and
      // (rows/8)*(3*channels/8) tiles are the same number. A candidate that maps group g to tile g
      // therefore needs no new geometry at all - my earlier version changed the thread count for no
      // reason. What is checked here is that the identity actually holds for this extent, and that a
      // tier declaring the tile mapping gets it verified rather than assumed.
      // Geometry check for ANY overridden wide stage, not just stage 4. Each stage has its own output
      // width disp[i], so the "one 8x8 tile per 64-thread group" identity must be verified per stage.
      // Keyed on the PER-LIBRARY selection, not the union. qkv_override_functions is documented in the
      // header as "union, for labels and back-compat only", and using it here made a stage inherit
      // another library's declared geometry: declaring nr_swin32_probabilities (out_width 1) for
      // pre.metallib made the tile8x8 check fire for it against prefix.metallib's declaration and refuse
      // the frame, so the experiment could not run at all. qkv_override_by_library is what selection
      // actually uses.
      const auto byl_geom = assets->qkv_override_by_library.find(lib);
      if (assets->qkv_override_active && byl_geom != assets->qkv_override_by_library.end() &&
          byl_geom->second.count(kWideFn[i])) {
        auto g = assets->qkv_override_dispatch.find(lib);
        if (g != assets->qkv_override_dispatch.end() && g->second == "mma8x8-rows8-v1" &&
            std::string(kWideFn[i]) == kQkvOverrideFunction) {
          // The PRESERVED alternative: two 8x8 tiles per 64-thread group, one per SIMD group. It is the
          // only case that changes the thread count, and it is admitted only because a manifest names
          // it, so both preserved mappings can be compared inside ONE runtime.
          if ((rows % 8u) != 0 || ((3u * channels) % 8u) != 0) {
            error = "qkv override: mma8x8-rows8-v1 needs rows and 3*channels to be multiples of 8"; return nullptr;
          }
          dispatch(enc, (size_t(rows) / 8) * ((size_t(channels) * 3) / 8) * 32);
          goto wide_dispatched;
        }
        if (g != assets->qkv_override_dispatch.end() && g->second == "tile8x8-in-accepted-dispatch-v1") {
          const size_t groups = (size_t(rows) * disp[i]) / chain_threadgroup();
          const size_t tiles  = (size_t(rows) / 8) * (size_t(disp[i]) / 8);
          if ((rows % 8u) != 0 || (disp[i] % 8u) != 0 ||
              ((size_t(rows) * disp[i]) % chain_threadgroup()) != 0 || groups != tiles) {
            error = std::string("qkv override: the tile8x8-in-accepted-dispatch-v1 mapping requires one 8x8 tile "
                    "per threadgroup, but stage ") + kWideFn[i] + " gives " + std::to_string(groups) +
                    " groups and " + std::to_string(tiles) + " tiles (rows=" + std::to_string(rows) +
                    " out_width=" + std::to_string(disp[i]) + "); refusing before any GPU work"; return nullptr;
          }
        }
      }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
      if (fusew && i == 0) {
        if (!dense_fused_stage(enc, x, y, XW, rows, channels)) return nullptr;
      } else
#endif
      if (dense && (i == 0 || i == 2 || i == 3 || i == 4 || i == 9)) {
        if (!dense_stage(enc, i, x, w, y, XW, XH, SD, rows, channels)) return nullptr;
      } else
#endif
      dispatch(enc, size_t(rows) * disp[i]);
    wide_dispatched:
      // DIAGNOSTIC capture of the FIRST C64 QKV stage output, AFTER its dispatch is encoded. flush()
      // closes the batched compute encoder first, so the blit never lands inside an open encoder and
      // the dispatch above is never encoded into an encoder this block already ended - which is
      // precisely the ordering defect in the preserved WIP counterexample.
      if (uint32_t(i) == qkv_capture_stage() && channels == qkv_capture_tier() && res->qkv_capture.metal != nil && !res->qkv_capture_done) {
        const size_t want = size_t(rows) * elements[qkv_capture_stage()] * bytes[qkv_capture_stage()];
        if (want == 0 || want > res->qkv_capture.bytes) {
          error = "qkv capture: stage extent " + std::to_string(want) + " does not fit the capture buffer (" +
                  std::to_string(res->qkv_capture.bytes) + "); refusing a partial capture"; return nullptr;
        }
        const size_t want_in  = size_t(rows) * elements[qkv_capture_stage() ? qkv_capture_stage()-1 : 0] * bytes[qkv_capture_stage() ? qkv_capture_stage()-1 : 0];
        const size_t want_cov = size_t(rows) * elements[qkv_capture_stage()] * 4;           // its per-output write counters
        if (want_in > res->qkv_capture_in.bytes || want_cov > res->qkv_capture_cov.bytes) {
          error = "qkv capture: operand or coverage extent does not fit; refusing a partial capture"; return nullptr;
        }
        flush();
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        if (!bl) { error = "qkv capture: blit encoder unavailable"; return nullptr; }
        nr_exp::nr_scope_adopt(bl);
        [bl copyFromBuffer:y[qkv_capture_stage()]->metal sourceOffset:G toBuffer:res->qkv_capture.metal destinationOffset:G size:want];
        [bl copyFromBuffer:y[qkv_capture_stage() ? qkv_capture_stage()-1 : 0]->metal sourceOffset:G toBuffer:res->qkv_capture_in.metal destinationOffset:G size:want_in];
        [bl copyFromBuffer:c[qkv_capture_stage()]->metal sourceOffset:G toBuffer:res->qkv_capture_cov.metal destinationOffset:G size:want_cov];
        // the producer buffer's own guard bands, before the pool hands this memory to a later block
        [bl copyFromBuffer:y[qkv_capture_stage()]->metal sourceOffset:0 toBuffer:res->qkv_capture_guards.metal destinationOffset:G size:Buf::kGuard];
        // The trailing guard sits at G + the SLOT's size, not G + want: Pool allocates each slot at the
        // measured MAXIMUM across the blocks that share it, so for C64 the slot is larger than this
        // stage's extent and G + want lands inside the data region. Reading there reported 256 wrong
        // "guard" bytes that were simply unwritten interior slot memory.
        [bl copyFromBuffer:y[qkv_capture_stage()]->metal sourceOffset:G + y[qkv_capture_stage()]->bytes toBuffer:res->qkv_capture_guards.metal
              destinationOffset:G + Buf::kGuard size:Buf::kGuard];
        nr_exp::nr_scope_end(bl);
        res->qkv_capture_bytes = want; res->qkv_capture_in_bytes = want_in;
        res->qkv_capture_cov_bytes = want_cov; res->qkv_capture_done = true;
      }
    }
    return y[10];
  }

  // C32 downsample after block 4 (fixed 480x288 destination).
  Buf* downsample32(Buf* projected, Buf* weight) {
    constexpr size_t rows = 480 * 288;
    scratch->rewind();
    Buf* pooled = scratch->get(rows * 32); Buf* result = carry->get(rows * 64);
    Buf* pc = scratch->get(rows * 32 * 4); Buf* rc = scratch->get(rows * 64 * 4);
    if (!pooled || !result || !pc || !rc) { error = scratch->error + carry->error; return nullptr; }
    if (!encoding()) return result;
    for (unsigned i = 0; i < 2; ++i) {
      auto p = ps("prefix.metallib", kC32Fn[11 + i]); if (!p) return nullptr;
      auto enc = begin("nr_exp.block4.downsample"); if (!enc) return nullptr;
      [enc setComputePipelineState:p];
      auto bind = [&](Buf* b, NSUInteger j) { [enc setBuffer:b->metal offset:G atIndex:j]; };
      uint32_t r = uint32_t(rows);
      if (i == 0) { bind(projected, 0); bind(pooled, 1); bind(pc, 2); }
      else { bind(pooled, 0); bind(weight, 1); bind(result, 2); bind(rc, 3); [enc setBytes:&r length:4 atIndex:4]; }
      dispatch(enc, rows * (i ? 64 : 32));
    }
    return result;
  }

  // Wide downsample after blocks 8, 14 and 22.
  Buf* downsample_wide(Buf* source, Buf* weight, Grid grid, uint32_t channels, uint32_t width, uint32_t height) {
    const uint32_t dw = (width / 2 + 7) / 8, dh = (height / 2 + 7) / 8;
    const uint32_t rows = dw * dh * 64;
    scratch->rewind();
    Buf* pooled = scratch->get(size_t(rows) * channels);
    Buf* result = carry->get(size_t(rows) * channels * 2);
    Buf* pc = scratch->get(size_t(rows) * channels * 4);
    Buf* rc = scratch->get(size_t(rows) * channels * 2 * 4);
    if (!pooled || !result || !pc || !rc) { error = scratch->error + carry->error; return nullptr; }
    if (!encoding()) return result;
    if (source->bytes != size_t(grid.width) * grid.height * 64 * channels * 2) { error = "wide downsample extent"; return nullptr; }
    WindowMap shape{width, height, grid.width, grid.height, dw, dh, grid.x, grid.y, 0, 0};
    const char* lib = tier_library(channels);
    for (unsigned i = 0; i < 2; ++i) {
      auto p = ps(lib, i ? "nr_wide_down_project" : "nr_wide_pool"); if (!p) return nullptr;
      auto enc = begin("nr_exp.downsample"); if (!enc) return nullptr;
      [enc setComputePipelineState:p];
      auto bind = [&](Buf* b, NSUInteger j) { [enc setBuffer:b->metal offset:G atIndex:j]; };
      if (i == 0) { bind(source, 0); bind(pooled, 1); bind(pc, 2); [enc setBytes:&shape length:sizeof(shape) atIndex:3]; }
      else { bind(pooled, 0); bind(weight, 1); bind(result, 2); bind(rc, 3); [enc setBytes:&rows length:4 atIndex:4]; }
      dispatch(enc, size_t(rows) * channels * (i ? 2 : 1));
    }
    return result;
  }

  Buf* native_export(Buf* source, uint32_t width, uint32_t height, uint32_t channels, bool planar) {
    const size_t bytes = size_t(width) * height * channels;
    scratch->rewind();
    Buf* result = carry->get(bytes); Buf* counts = scratch->get(bytes * 4);
    if (!result || !counts) { error = carry->error + scratch->error; return nullptr; }
    if (!encoding()) return result;
    const size_t rows = size_t((width + 7) / 8) * ((height + 7) / 8) * 64;
    if (source->bytes != rows * channels || width % 4 || height % 4 || channels % 32) { error = "native export extent"; return nullptr; }
    auto p = ps("wide256.metallib", "nr_encoder_native_export"); if (!p) return nullptr;
    const uint32_t shape[4] = {width, height, channels, planar ? 1u : 0u};
    auto enc = begin("nr_exp.native_export"); if (!enc) return nullptr;
    [enc setComputePipelineState:p];
    [enc setBuffer:source->metal offset:G atIndex:0]; [enc setBuffer:result->metal offset:G atIndex:1];
    [enc setBuffer:counts->metal offset:G atIndex:2]; [enc setBytes:shape length:sizeof(shape) atIndex:3];
    dispatch(enc, bytes);
    return result;
  }

  void tier(uint32_t t) { cb = (real_cb && (tier_mask & (1u << t))) ? real_cb : nil; }

  bool run(Buf* block0_encoder) {
    carry->rewind(); scratch->rewind();
    Buf* input = block0_encoder;
    Buf* skip4 = nullptr; Buf* skip8 = nullptr; Buf* skip14 = nullptr; Buf* skip22 = nullptr;
    Buf* downsample = nullptr;
    // ---- blocks 1-4, C32 on the 960x576 grid ----
    tier(0);
    for (uint32_t i = 0; i < 4; ++i) {
      cur_block = i + 1;
      Buf* w = &res->weights[i];
      Buf* prepared = i == 0 ? input : rewindow("prefix.metallib", input, kGrids32[0], kGrids32[i], 32, 960, 576);
      if (!prepared) return false;
      Buf* projection = nullptr;
      Buf* result = block32(prepared, w, kGrids32[i].width * kGrids32[i].height * 64, i == 3 ? &projection : nullptr, size_t(i));
      if (!result) return false;
      if (i == 3) { downsample = downsample32(projection, w); if (!downsample) return false; }
      input = i == 0 ? result : rewindow("prefix.metallib", result, kGrids32[i], kGrids32[0], 32, 960, 576);
      if (!input) return false;
      if (i == 3) skip4 = input;
    }
    // ---- blocks 5-8, 64 channels on 480x288 ----
    tier(1);
    Buf* wide = downsample; Buf* enc128 = nullptr;
    for (uint32_t b = 5; b <= 8; ++b) {
      cur_block = b;
      Buf* w = &res->weights[b - 1]; Grid g = kGrids64[b - 5];
      Buf* prepared = b == 5 ? wide : rewindow("prefix.metallib", wide, kGrids64[0], g, 64, 480, 288);
      if (!prepared) return false;
      Buf* projection = nullptr;
      Buf* result = block_wide(prepared, w, g.width * g.height * 64, 64, b == 8 ? &projection : nullptr);
      if (!result) return false;
      if (b == 8) { enc128 = downsample_wide(projection, w, g, 64, 480, 288); if (!enc128) return false; }
      wide = b == 5 ? result : rewindow("prefix.metallib", result, g, kGrids64[0], 64, 480, 288);
      if (!wide) return false;
      if (b == 8) skip8 = wide;
    }
    // ---- blocks 9-14, 128 channels on 240x144 ----
    tier(2);
    Buf* enc256 = nullptr;
    for (uint32_t b = 9; b <= 14; ++b) {
      cur_block = b;
      Buf* w = &res->weights[b - 1]; Grid g = kGrids128[b - 9];
      Buf* prepared = b == 9 ? enc128 : rewindow("wide128.metallib", enc128, kGrids128[0], g, 128, 240, 144);
      if (!prepared) return false;
      Buf* projection = nullptr;
      Buf* result = block_wide(prepared, w, g.width * g.height * 64, 128, b == 14 ? &projection : nullptr);
      if (!result) return false;
      if (b == 14) { enc256 = downsample_wide(projection, w, g, 128, 240, 144); if (!enc256) return false; }
      enc128 = b == 9 ? result : rewindow("wide128.metallib", result, g, kGrids128[0], 128, 240, 144);
      if (!enc128) return false;
      if (b == 14) skip14 = enc128;
    }
    // ---- blocks 15-22, 256 channels on 120x72 ----
    tier(3);
    Buf* middle = nullptr;
    for (uint32_t b = 15; b <= 22; ++b) {
      cur_block = b;
      Buf* w = &res->weights[b - 1]; Grid g = kGrids256[b - 15];
      Buf* prepared = b == 15 ? enc256 : rewindow("wide256.metallib", enc256, kGrids256[0], g, 256, 120, 72);
      if (!prepared) return false;
      Buf* projection = nullptr;
      Buf* result = block_wide(prepared, w, g.width * g.height * 64, 256, b == 22 ? &projection : nullptr);
      if (!result) return false;
      if (b == 22) { middle = downsample_wide(projection, w, g, 256, 120, 72); if (!middle) return false; }
      enc256 = b == 15 ? result : rewindow("wide256.metallib", result, g, kGrids256[0], 256, 120, 72);
      if (!enc256) return false;
      if (b == 22) skip22 = enc256;
    }
    // ---- native exports: four skips in body order (C256, C128, C64, C32) and the entry view ----
    tier(4);
    const std::array<Buf*, 4> src = {skip4, skip8, skip14, skip22};
    const std::array<uint32_t, 4> w4 = {960, 480, 240, 120}, h4 = {576, 288, 144, 72}, c4 = {32, 64, 128, 256};
    res->skips.assign(4, Buf{});
    for (size_t i = 0; i < 4; ++i) {
      Buf* e = native_export(src[i], w4[i], h4[i], c4[i], false);
      if (!e) return false;
      res->skips[3 - i] = *e;
    }
    Buf* entry = native_export(middle, 60, 36, 512, true);
    if (!entry) return false;
    res->entry = *entry;
    return true;
  }
};
} // namespace

size_t wide_threadgroup_width() { return Walk::chain_threadgroup(); }

bool allocate_encoder(id<MTLDevice> device, Assets& assets, const Block0Resources& b0, EncoderResources& res, std::string& error) {
  @autoreleasepool {
    if (!b0.ready) { error = "block 0 resources required first"; return false; }
    // Every pinned block weight, verified by load_assets() and uploaded once here.
    res.weights.assign(22, Buf{});
    res.prepack_active = [] { const char* v = std::getenv("NR_EXP_PREPACK_WEIGHTS"); return v && v[0] == '1'; }();
    if (res.prepack_active) res.prepacked_weights.assign(22, Buf{});
    for (uint32_t b = 1; b <= 22; ++b) {
      const std::string key = "weights/block" + std::to_string(b) + ".bin";
      const auto& bytes = assets.constant(key);
      if (bytes.size() != weight_bytes_for(b)) { error = "weight extent for block " + std::to_string(b); return false; }
      res.weights[b - 1] = make_buffer(device, bytes.size(), false);
      if (!res.weights[b - 1].metal) { error = "weight buffer allocation for block " + std::to_string(b); return false; }
      std::memcpy(res.weights[b - 1].data(), bytes.data(), bytes.size());
      // PREPACK, opt-in and verified exhaustively here rather than trusted. nr_weight_index is a BIJECTION
      // from (k<32, n<32) onto 0..1023, so the linear layout prepacked[block*1024 + k*32 + n] =
      // original[block*1024 + nr_weight_index(k,n)] is loss-free and reversible. Every byte it writes is
      // checked back against the original; a single mismatch fails preparation.
      if (res.prepack_active) {
        Buf& pw = res.prepacked_weights[b - 1];
        pw = make_buffer(device, bytes.size(), false);
        if (!pw.metal) { error = "prepacked weight allocation for block " + std::to_string(b); return false; }
        std::memcpy(pw.data(), bytes.data(), bytes.size());          // regions we do not permute stay byte-equal
        const size_t blocks = bytes.size() / 1024;
        for (size_t blk = 0; blk < blocks; ++blk) {
          const uint8_t* src = bytes.data() + blk * 1024;
          uint8_t* dst = pw.data() + blk * 1024;
          for (uint32_t k = 0; k < 32; ++k)
            for (uint32_t n = 0; n < 32; ++n)
              dst[k * 32 + n] = src[nr_weight_index_host(k, n)];
        }
        for (size_t blk = 0; blk < blocks; ++blk) {                  // exhaustive read-back proof
          const uint8_t* src = bytes.data() + blk * 1024;
          const uint8_t* dst = pw.data() + blk * 1024;
          for (uint32_t k = 0; k < 32; ++k)
            for (uint32_t n = 0; n < 32; ++n)
              if (dst[k * 32 + n] != src[nr_weight_index_host(k, n)]) {
                error = "prepacked weights: derived layout does not reproduce the original bytes at block " +
                        std::to_string(blk) + " k=" + std::to_string(k) + " n=" + std::to_string(n);
                return false;
              }
          res.prepack_verified_bytes += 1024;
        }
      }
    }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
    // ENCODER-DENSE-WS8: the five weight tables of every block on a selected wide tier, prepacked ONCE here by the dense
    // library's own nrd_prepack on a transient queue. Preparation cost, recorded separately; never on the encode path.
    res.encdense_tables.assign(22, std::vector<Buf>{});
    res.encdense_prepack_gpu_ms = 0;
    if (assets.encdense_active) {
      id<MTLCommandQueue> pq = [device newCommandQueue];
      id<MTLCommandBuffer> pcb = pq ? [pq commandBuffer] : nil;
      id<MTLComputeCommandEncoder> pe = pcb ? [pcb computeCommandEncoder] : nil;
      if (!pe) { error = "encoder dense: prepack command encoder unavailable"; return false; }
      pe.label = @"nr_exp.encdense.prepack";
      struct PrepackHost { uint32_t kind, count; };
      for (uint32_t b = 5; b <= 22; ++b) {
        const uint32_t ch = b <= 8 ? 64u : b <= 14 ? 128u : 256u;
        if (!(assets.encdense_tier_mask & (ch == 64 ? 2u : ch == 128 ? 4u : 8u))) continue;
        const uint32_t hidden = 4u * ch, heads = ch / 32u;
        const std::array<uint32_t, 5> counts = {heads * hidden * 32u, heads * 4u * 32u * 32u, heads * ch * 32u, heads * 3u * ch * 32u, heads * ch * 32u};
        const std::string lib = ch == 64 ? "enc_dense64.metallib" : ch == 128 ? "enc_dense128.metallib" : "enc_dense256.metallib";
        std::string perr;
        id<MTLComputePipelineState> pp = assets.pipeline(lib, "nrd_prepack", perr);
        if (!pp) { error = "encoder dense: " + perr; return false; }
        auto& tables = res.encdense_tables[b - 1];
        for (uint32_t kind = 0; kind < 5; ++kind) {
          Buf t = make_buffer(device, size_t(counts[kind]) * 4, false);
          if (!t.metal) { error = "encoder dense: weight table allocation for block " + std::to_string(b); return false; }
          const PrepackHost p{kind, counts[kind]};
          [pe setComputePipelineState:pp];
          [pe setBuffer:res.weights[b - 1].metal offset:Buf::kGuard atIndex:0]; [pe setBuffer:t.metal offset:Buf::kGuard atIndex:1];
          [pe setBytes:&p length:sizeof p atIndex:2];
          [pe dispatchThreads:MTLSizeMake(counts[kind], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
          tables.push_back(t);
        }
      }
      if (assets.encdense_tier_mask & 1u) {   // ENCODER-DENSE32-HYBRID (tier "32"): the expand and qkv tables of C32 walk blocks 1-4
        res.encdense32_tables.assign(4, std::vector<Buf>{});
        std::string perr32;
        id<MTLComputePipelineState> pp32 = assets.pipeline("enc_dense32.metallib", "nrd32w_prepack", perr32);
        if (!pp32) { error = "encoder dense (32): " + perr32; return false; }
        const std::array<std::pair<uint32_t, uint32_t>, 2> kinds32 = {{ {0u, 128u * 32u}, {2u, 96u * 32u} }};   // kind 0 expand, kind 2 qkv
        for (uint32_t b = 1; b <= 4; ++b) {
          auto& tables32 = res.encdense32_tables[b - 1];
          for (const auto& kc : kinds32) {
            Buf t = make_buffer(device, size_t(kc.second) * 4, false);
            if (!t.metal) { error = "encoder dense (32): weight table allocation for block " + std::to_string(b); return false; }
            const PrepackHost p{kc.first, kc.second};
            [pe setComputePipelineState:pp32];
            [pe setBuffer:res.weights[b - 1].metal offset:Buf::kGuard atIndex:0]; [pe setBuffer:t.metal offset:Buf::kGuard atIndex:1];
            [pe setBytes:&p length:sizeof p atIndex:2];
            [pe dispatchThreads:MTLSizeMake(kc.second, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            tables32.push_back(t);
          }
        }
      }
      [pe endEncoding]; [pcb commit]; [pcb waitUntilCompleted];
      if (pcb.status != MTLCommandBufferStatusCompleted) { error = "encoder dense: prepack command buffer did not complete"; return false; }
      res.encdense_prepack_gpu_ms = (pcb.GPUEndTime - pcb.GPUStartTime) * 1000.0;
    }
#endif
    // Measuring walk: the identical call sequence the encode walk takes, so the pools cannot disagree with it.
    res.carry.device = device; res.scratch.device = device;
    res.carry.mode = Pool::Mode::Measure; res.scratch.mode = Pool::Mode::Measure;
    res.carry.sizes.clear(); res.scratch.sizes.clear();
    Walk w; w.assets = &assets; w.res = &res; w.carry = &res.carry; w.scratch = &res.scratch; w.tier_mask = 0;
    // The accepted encoder relays block 0's 17,694,720-byte export through nr_pre_entry_layout before
    // blocks 1-22 read it. This port fed b0.encoder straight in, which is why its C32 export was the
    // first boundary to diverge from the accepted lane while block 0 itself was byte-exact.
    res.layout = make_buffer(device, b0.encoder.bytes, false);
    if (!res.layout.metal) { error = "entry layout buffer allocation failed"; return false; }
    Buf block0_out = res.layout;
    if (!w.run(&block0_out)) { error = w.error.empty() ? (res.carry.error + res.scratch.error) : w.error; return false; }
    if (!res.carry.allocate()) { error = res.carry.error; return false; }
    if (!res.scratch.allocate()) { error = res.scratch.error; return false; }
    // The measuring walk filled res.skips/res.entry with placeholders; redo the walk in Use mode so they
    // point at the real allocations. Nothing is encoded because the command buffer is still nil.
    if (!w.run(&block0_out)) { error = w.error.empty() ? (res.carry.error + res.scratch.error) : w.error; return false; }
    // Opt-in diagnostic capture buffer for the first C64 QKV stage output. Allocated only when asked
    // for; when the variable is absent nothing is allocated and the capture path never runs.
    if (const char* c = std::getenv("NR_EXP_CAPTURE_QKV_C64")) {
      if (c[0]) {
        const size_t bytes = res.qkv_capture_needed;                 // the real extent, from the measuring walk
        if (bytes == 0) { error = "qkv capture requested but no C64 wide stage was measured"; return false; }
        // Poison the capture destinations, so an incomplete copy shows as poison rather than as
        // plausible-looking stale bytes. NR_EXP_CAPTURE_POISON selects the pattern; two different
        // values across two runs show the result does not depend on residual memory.
        uint8_t poison = 0xA5;
        if (const char* pv = std::getenv("NR_EXP_CAPTURE_POISON"))
          poison = uint8_t(strtoul(pv, nullptr, 0));
        res.qkv_capture     = make_buffer(device, bytes, true, poison);
        res.qkv_capture_in  = make_buffer(device, res.qkv_capture_in_bytes, true, poison);
        res.qkv_capture_cov = make_buffer(device, res.qkv_capture_cov_bytes, true, poison);
        res.qkv_capture_cov_pre = make_buffer(device, res.qkv_capture_cov_bytes, true, poison);
        res.qkv_capture_guards  = make_buffer(device, 2 * Buf::kGuard, true, poison);
        if (!res.qkv_capture.metal || !res.qkv_capture_in.metal || !res.qkv_capture_cov.metal ||
            !res.qkv_capture_cov_pre.metal || !res.qkv_capture_guards.metal) {
          error = "could not allocate the QKV capture buffers"; return false; }
      }
    }
    res.ready = true;
    return true;
  }
}

bool encode_encoder(id<MTLCommandBuffer> cb, Assets& assets, Block0Resources& b0, EncoderResources& res,
                    EncoderEncoded& out, std::string& error, uint32_t tier_mask,
                    uint32_t block_lo, uint32_t block_hi) {
  if (!cb) { error = "no caller command buffer"; return false; }
  if (!res.ready) { error = "encoder resources not allocated"; return false; }
  @autoreleasepool {
    res.carry.error.clear(); res.scratch.error.clear();
    // Entry layout: block 0's export -> the layout blocks 1-22 consume. Encode-only, on the caller's buffer.
    std::string perr;
    id<MTLComputePipelineState> layout_ps = assets.pipeline("entry.metallib", "nr_pre_entry_layout", perr);
    if (!layout_ps) { error = "entry layout pipeline unavailable: " + perr; return false; }
    uint32_t layout_dispatches = 0, layout_encoders = 0;
    // The entry layout feeds block 1, so it belongs to whichever chunk encodes block 1 - not only to a
    // whole-chain call. Gating it on block_lo == 0 meant a chunked caller never ran it.
    if (block_lo <= 1) {
      id<MTLComputeCommandEncoder> enc = nr_exp::gpu_profiler().compute(cb, "encoder.block"); nr_exp::nr_scope_adopt(enc);
      if (!enc) { error = "compute encoder unavailable"; return false; }
      enc.label = @"nr_exp.encoder.entry_layout";
      [enc setComputePipelineState:layout_ps];
      [enc setBuffer:b0.encoder.metal offset:Buf::kGuard atIndex:0];
      [enc setBuffer:res.layout.metal offset:Buf::kGuard atIndex:1];
      [enc setBuffer:b0.encoder_counts.metal offset:Buf::kGuard atIndex:2];
      [enc dispatchThreads:MTLSizeMake(b0.encoder.bytes, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      nr_exp::nr_scope_end(enc);
      ++layout_dispatches; ++layout_encoders;
    }
    Walk w; w.real_cb = cb; w.tier_mask = tier_mask; w.assets = &assets; w.res = &res; w.carry = &res.carry; w.scratch = &res.scratch;
    w.block_lo = block_lo; w.block_hi = block_hi;
    const bool ok = w.run(&res.layout);
    w.flush();                       // the batched encoder must be closed before the caller adds anything else
    if (!ok) { error = w.error.empty() ? (res.carry.error + res.scratch.error) : w.error; return false; }
    out.dispatches = w.dispatches + layout_dispatches; out.compute_encoders = w.encoders + layout_encoders;
    out.bound_bytes = w.bound_bytes; out.bound_bytes_wide = w.bound_bytes_wide; out.bound_bytes_coverage = w.bound_bytes_coverage;
    return true;
  }
}
} // namespace nr_exp
