#include "nr_experimental_block0.h"
#import <Foundation/Foundation.h>
#include "nr_experimental_gpu_profile.h"
#include "nr_experimental_encoder_scope_adoption.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace nr_exp {

// Geometry and dispatch constants, copied verbatim from the accepted encoder's generated block 0.
struct PreShape { uint32_t width, height, padded_height, frame, phase_math; };
struct PreChunk { uint32_t rows, base_window; };
static constexpr uint32_t kWidth = 1920, kHeight = 1080, kPaddedHeight = 1152;
static constexpr uint32_t kWindows = 240 * 144, kChunk = 512;
static constexpr size_t kInputBytes = size_t(kWidth) * kHeight * 8;      // RGBA16F
static constexpr size_t kFeatureBytes = 70778880, kFeatureCountBytes = 141557760;
static constexpr size_t kSkipBytes = 70778880, kSkipCountBytes = 283115520;
static constexpr size_t kEncoderBytes = 17694720, kEncoderCountBytes = 70778880;
static constexpr size_t kDeltaBytes = 67108864, kEscapeBytes = 2780432;
static const std::array<unsigned, 9> kElements = {128, 128, 128, 96, 64, 64, 64, 64, 32};
static const std::array<unsigned, 9> kSizes    = {2, 1, 2, 2, 1, 2, 1, 2, 2};
static const std::array<unsigned, 9> kDispatch = {128, 128, 32, 96, 2, 64, 1, 32, 32};
static const char* kStageFn[9] = {"nr_swin32_expand", "nr_swin32_activate", "nr_swin32_contract", "nr_swin32_qkv",
                                  "nr_swin32_normalize", "nr_swin32_scores", "nr_swin32_probabilities",
                                  "nr_swin32_values", "nr_swin32_project"};

Buf make_buffer(id<MTLDevice> device, size_t bytes, bool poison, uint8_t poison_value) {
  Buf b; b.bytes = bytes;
  b.metal = [device newBufferWithLength:bytes + 2 * Buf::kGuard options:MTLResourceStorageModeShared];
  if (!b.metal) return b;
  std::memset(b.metal.contents, 0xc3, Buf::kGuard);
  std::memset(b.data() + bytes, 0xc3, Buf::kGuard);
  if (poison) std::memset(b.data(), poison_value, bytes);
  return b;
}
size_t guard_errors(const Buf& b) {
  if (!b.metal) return 1;
  const uint8_t* p = static_cast<const uint8_t*>(b.metal.contents);
  size_t e = 0;
  for (size_t i = 0; i < Buf::kGuard; ++i) e += (p[i] != 0xc3) + (p[Buf::kGuard + b.bytes + i] != 0xc3);
  return e;
}

bool allocate_block0(id<MTLDevice> device, const Assets& assets, Block0Resources& r, std::string& error) {
  @autoreleasepool {
    if (!assets.loaded) { error = "assets not loaded"; return false; }
    const auto& w = assets.constant("block0.weights");
    const auto& d = assets.constant("instruction-deltas.u8");
    const auto& e = assets.constant("instruction-escapes.u32");
    if (w.size() != 21696) { error = "block0.weights size"; return false; }
    if (d.size() != kDeltaBytes || e.size() != kEscapeBytes) { error = "math-correction table size"; return false; }
    r.input          = make_buffer(device, kInputBytes, false);
    r.weight         = make_buffer(device, w.size(), false);
    r.features       = make_buffer(device, kFeatureBytes, false);
    r.skip           = make_buffer(device, kSkipBytes, false);
    r.encoder        = make_buffer(device, kEncoderBytes, false);
    r.deltas         = make_buffer(device, kDeltaBytes, false);
    r.escapes        = make_buffer(device, kEscapeBytes, false);
    r.feature_counts = make_buffer(device, kFeatureCountBytes, false);
    r.temporal_features = make_buffer(device, kFeatureBytes, false);
    r.temporal_coverage = make_buffer(device, kFeatureBytes * 2, false);
    {   // the audited reciprocal ROM the temporal feature kernel reads, uploaded once
      const auto& rom = assets.constant("temporal-rcp-rom.bin");
      r.rcp_rom = make_buffer(device, rom.size(), false);
      if (!r.rcp_rom.metal) { error = "rcp rom allocation failed"; return false; }
      std::memcpy(r.rcp_rom.data(), rom.data(), rom.size());
    }
    r.skip_counts    = make_buffer(device, kSkipCountBytes, false);
    r.encoder_counts = make_buffer(device, kEncoderCountBytes, false);
    // Chunk scratch, sized for the largest chunk exactly as the accepted encoder sizes it.
    const uint32_t rows = kChunk * 64;
    size_t slot = 0;
    auto scratch = [&](size_t bytes) { r.scratch[slot++] = make_buffer(device, bytes, false); };
    scratch(size_t(rows) * 64); scratch(size_t(rows) * 32); scratch(size_t(rows) * 128); scratch(size_t(rows) * 128);
    for (size_t i = 0; i < 9; ++i) { scratch(size_t(rows) * kElements[i] * kSizes[i]); scratch(size_t(rows) * kElements[i] * 4); }
    if (slot != r.scratch.size()) { error = "scratch slot inventory"; return false; }
    for (const Buf* b : {&r.input, &r.weight, &r.features, &r.skip, &r.encoder, &r.deltas, &r.escapes,
                         &r.feature_counts, &r.skip_counts, &r.encoder_counts})
      if (!b->metal) { error = "buffer allocation failed"; return false; }
    for (const auto& b : r.scratch) if (!b.metal) { error = "scratch allocation failed"; return false; }
    // Immutable constants are uploaded ONCE here, not per frame.
    std::memcpy(r.weight.data(), w.data(), w.size());
    std::memcpy(r.deltas.data(), d.data(), d.size());
    std::memcpy(r.escapes.data(), e.data(), e.size());
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
    // ENCODER-DENSE32-HYBRID (tier "b0"): Block 0's expand and qkv tables, prepacked ONCE here from the uploaded Block 0 weights by
    // the dense library's own kernel on a transient queue (preparation, never the encode path).
    r.encdense32_active = false;
    if (assets.encdense_active && (assets.encdense_tier_mask & 0x20u)) {
      const auto pp = assets.pipelines.find("enc_dense32b0.metallib:nrd32w_prepack");
      if (pp == assets.pipelines.end() || pp->second == nil) { error = "encoder dense (b0): the prepack pipeline was not created"; return false; }
      r.encdense32_x = make_buffer(device, size_t(rows) * 32 * 4, false);
      r.encdense32_tables[0] = make_buffer(device, size_t(128) * 32 * 4, false);
      r.encdense32_tables[1] = make_buffer(device, size_t(96) * 32 * 4, false);
      if (!r.encdense32_x.metal || !r.encdense32_tables[0].metal || !r.encdense32_tables[1].metal) { error = "encoder dense (b0): buffer allocation failed"; return false; }
      id<MTLCommandQueue> pq = [device newCommandQueue];
      id<MTLCommandBuffer> pcb = pq ? [pq commandBuffer] : nil;
      id<MTLComputeCommandEncoder> pe = pcb ? [pcb computeCommandEncoder] : nil;
      if (!pe) { error = "encoder dense (b0): prepack command encoder unavailable"; return false; }
      pe.label = @"nr_exp.encdense32.block0.prepack";
      struct PrepackHost { uint32_t kind, count; };
      const PrepackHost kinds[2] = {{0u, 128u * 32u}, {2u, 96u * 32u}};   // nrd32w_prepack kind 0 expand, kind 2 qkv
      for (int k = 0; k < 2; ++k) {
        [pe setComputePipelineState:pp->second];
        [pe setBuffer:r.weight.metal offset:Buf::kGuard atIndex:0];
        [pe setBuffer:r.encdense32_tables[k].metal offset:Buf::kGuard atIndex:1];
        [pe setBytes:&kinds[k] length:sizeof kinds[k] atIndex:2];
        [pe dispatchThreads:MTLSizeMake(kinds[k].count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      [pe endEncoding]; [pcb commit]; [pcb waitUntilCompleted];
      if (pcb.status != MTLCommandBufferStatusCompleted) { error = "encoder dense (b0): prepack command buffer did not complete"; return false; }
      r.encdense32_active = true;
    }
#endif
    r.ready = true;
    return true;
  }
}

bool encode_block0(id<MTLCommandBuffer> cb, Assets& assets, Block0Resources& r,
                   id<MTLTexture> color, uint32_t frame_index, Block0Encoded& out, std::string& error,
                   id<MTLTexture> history, id<MTLTexture> motion,
                   uint32_t unit_lo, uint32_t unit_hi) {
  // ---- validate everything BEFORE encoding anything; a rejection must leave the caller's buffer untouched ----
  if (!cb) { error = "no caller command buffer"; return false; }
  if (!r.ready) { error = "block 0 resources not allocated"; return false; }
  if (!color) { error = "no colour texture"; return false; }
  if (color.width != kWidth || color.height != kHeight) {
    error = "colour texture is " + std::to_string(color.width) + "x" + std::to_string(color.height) + ", model input is 1920x1080"; return false; }
  if (color.pixelFormat != MTLPixelFormatRGBA16Float) { error = "colour texture is not RGBA16Float"; return false; }
  if (color.textureType != MTLTextureType2D) { error = "colour texture is not 2D"; return false; }
  if (!(color.usage & MTLTextureUsageShaderRead)) { error = "colour texture lacks ShaderRead usage"; return false; }
  std::string e;
  id<MTLComputePipelineState> features_ps = assets.pipeline("pre.metallib", "nr_pre_features", e);
  id<MTLComputePipelineState> adapter_ps  = assets.pipeline("pre.metallib", "nr_pre_adapter", e);
  id<MTLComputePipelineState> export_ps   = assets.pipeline("pre.metallib", "nr_pre_export", e);
  std::array<id<MTLComputePipelineState>, 9> stage_ps{};
  for (size_t i = 0; i < 9; ++i) stage_ps[i] = assets.pipeline("pre.metallib", kStageFn[i], e);
  if (!features_ps || !adapter_ps || !export_ps) { error = "block 0 pipeline unavailable: " + e; return false; }
  for (auto p : stage_ps) if (!p) { error = "block 0 stage pipeline unavailable: " + e; return false; }
  const bool temporal = history != nil || motion != nil;
  id<MTLComputePipelineState> temporal_ps = nil;
  if (temporal) {
    if (!history || !motion) { error = "a temporal frame needs both a history and a motion texture"; return false; }
    if (history.width != kWidth || history.height != kHeight || history.pixelFormat != MTLPixelFormatRGBA16Float) { error = "history texture is not 1920x1080 RGBA16Float"; return false; }
    if (motion.width != kWidth || motion.height != kHeight || motion.pixelFormat != MTLPixelFormatRG16Float) { error = "motion texture is not 1920x1080 RG16Float"; return false; }
    temporal_ps = assets.pipeline("temporal-hardware.metallib", "nr_pre_temporal_features", e);
    if (!temporal_ps) { error = "temporal feature pipeline unavailable: " + e; return false; }
  }

  @autoreleasepool {
    const NSUInteger G = Buf::kGuard;
    uint32_t& dispatches = out.dispatches; uint32_t& encoders = out.compute_encoders;
    dispatches = 0; encoders = 0;
    const bool prologue = (unit_lo == 0);
    // ---- input: blit the caller's texture into the model input buffer, on the caller's buffer ----
    if (prologue) {
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; nr_exp::nr_scope_adopt(bl);
      if (!bl) { error = "blit encoder unavailable"; return false; }
      bl.label = @"nr_exp.block0.input";
      [bl copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                 toBuffer:r.input.metal destinationOffset:G
        destinationBytesPerRow:kWidth * 8 destinationBytesPerImage:kInputBytes];
      nr_exp::nr_scope_end(bl);
      ++encoders;
    }
    // ---- features ----
    PreShape shape{kWidth, kHeight, kPaddedHeight, frame_index, 1};
    uint32_t escape_count = uint32_t(kEscapeBytes / 8);
    if (prologue) {
      id<MTLComputeCommandEncoder> enc = nr_exp::gpu_profiler().compute(cb, "block0.features"); nr_exp::nr_scope_adopt(enc);
      if (!enc) { error = "compute encoder unavailable"; return false; }
      [enc setComputePipelineState:features_ps];
      [enc setBuffer:r.input.metal offset:G atIndex:0];
      [enc setBuffer:r.features.metal offset:G atIndex:1];
      [enc setBuffer:r.feature_counts.metal offset:G atIndex:2];
      [enc setBytes:&shape length:sizeof(shape) atIndex:3];
      [enc setBuffer:r.deltas.metal offset:G atIndex:4];
      [enc setBuffer:r.escapes.metal offset:G atIndex:5];
      [enc setBytes:&escape_count length:4 atIndex:6];
      [enc dispatchThreads:MTLSizeMake(size_t(kWidth) * kPaddedHeight, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      nr_exp::nr_scope_end(enc);
      ++dispatches; ++encoders;
    }
    // ---- temporal features: the accepted block-0 step, history and motion sampled ON THE GPU ----
    Buf& adapter_features = temporal ? r.temporal_features : r.features;
    if (temporal && prologue) {
      id<MTLComputeCommandEncoder> t = nr_exp::gpu_profiler().compute(cb, "block0.temporal"); nr_exp::nr_scope_adopt(t);
      if (!t) { error = "compute encoder unavailable"; return false; }
      [t setComputePipelineState:temporal_ps];
      uint32_t tshape[4] = {kWidth, kHeight, kPaddedHeight, 0};
      [t setBuffer:r.features.metal offset:G atIndex:0];
      [t setBuffer:r.temporal_features.metal offset:G atIndex:1];
      [t setBuffer:r.temporal_coverage.metal offset:G atIndex:2];
      [t setBuffer:r.rcp_rom.metal offset:G atIndex:3];
      [t setBytes:tshape length:sizeof(tshape) atIndex:4];
      [t setTexture:history atIndex:0];
      [t setTexture:motion atIndex:1];
      [t dispatchThreads:MTLSizeMake(kWidth, kPaddedHeight, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      nr_exp::nr_scope_end(t);
      ++dispatches; ++encoders;
    }
    // ---- the windowed pass: same chunking, same scratch reuse, no waits between chunks ----
    uint32_t unit = 0;
    for (uint32_t base = 0; base < kWindows; base += kChunk, ++unit) {
      if (unit < unit_lo || unit >= unit_hi) continue;   // bounded chunk: state lives in persistent buffers
      PreChunk s{std::min(kChunk, kWindows - base) * 64, base};
      const uint32_t rows = s.rows;
      Buf& raw = r.scratch[0]; Buf& packed = r.scratch[1]; Buf& rc = r.scratch[2]; Buf& pc = r.scratch[3];
      auto y = [&](size_t i) -> Buf& { return r.scratch[4 + i * 2]; };
      auto c = [&](size_t i) -> Buf& { return r.scratch[5 + i * 2]; };
      {
        id<MTLComputeCommandEncoder> a = nr_exp::gpu_profiler().compute(cb, "block0.adapter"); nr_exp::nr_scope_adopt(a);
        if (!a) { error = "compute encoder unavailable"; return false; }
        [a setComputePipelineState:adapter_ps];
        const Buf* ab[6] = {&adapter_features, &r.weight, &raw, &packed, &rc, &pc};
        for (NSUInteger i = 0; i < 6; ++i) [a setBuffer:ab[i]->metal offset:G atIndex:i];
        [a setBytes:&s length:sizeof(s) atIndex:6];
        // Test-only: throw WHILE this encoder is open, which is the case the old global tracker could not
        // handle. Off unless NR_EXP_FAIL_WITH_ENCODER_OPEN=1.
        { static const bool inject = [] { const char* v = std::getenv("NR_EXP_FAIL_WITH_ENCODER_OPEN"); return v && v[0] == '1'; }();
          if (inject && base == 0) throw std::runtime_error("injected failure with a compute encoder open (NR_EXP_FAIL_WITH_ENCODER_OPEN)"); }
        [a dispatchThreads:MTLSizeMake(size_t(rows) * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        nr_exp::nr_scope_end(a);
        ++dispatches; ++encoders;
      }
      for (size_t i = 0; i < 9; ++i) {
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
        // DENSE-MLP-FUSION-v1: with the Block 0 tier fused, stage 0 writes nr_swin32_activate's bytes into y(1), so activate gets no encoder
        if (r.encdense32_active && (assets.encdense_fuse_mask & 0x20u) != 0 && i == 1) continue;
#endif
        id<MTLComputeCommandEncoder> enc = nr_exp::gpu_profiler().compute(cb, kStageFn[i]); nr_exp::nr_scope_adopt(enc);
        if (!enc) { error = "compute encoder unavailable"; return false; }
        [enc setComputePipelineState:stage_ps[i]];
        auto bind = [&](Buf& b, NSUInteger j) { [enc setBuffer:b.metal offset:G atIndex:j]; };
        auto shape_at = [&](NSUInteger j) { [enc setBytes:&s.rows length:4 atIndex:j]; };
        if (i == 0) { bind(packed, 0); bind(r.weight, 1); bind(y(0), 2); bind(c(0), 3); shape_at(4); }
        if (i == 1) { bind(y(0), 0); bind(y(1), 1); bind(c(1), 2); shape_at(3); }
        if (i == 2) { bind(raw, 0); bind(y(1), 1); bind(r.weight, 2); bind(y(2), 3); bind(c(2), 4); shape_at(5); }
        if (i == 3) { bind(y(2), 0); bind(r.weight, 1); bind(y(3), 2); bind(c(3), 3); shape_at(4); }
        if (i == 4) { bind(y(3), 0); bind(r.weight, 1); bind(y(4), 2); bind(c(4), 3); shape_at(4); }
        if (i == 5) { bind(y(4), 0); bind(r.weight, 1); bind(y(5), 2); bind(c(5), 3); shape_at(4); }
        if (i == 6) { bind(y(5), 0); bind(y(6), 1); bind(c(6), 2); shape_at(3); }
        if (i == 7) { bind(y(6), 0); bind(y(3), 1); bind(y(7), 2); bind(c(7), 3); shape_at(4); }
        if (i == 8) { bind(y(7), 0); bind(y(2), 1); bind(r.weight, 2); bind(y(8), 3); bind(c(8), 4); shape_at(5); }
        // DECLARED DISPATCH KIND, opt-in per (library, function). Without a manifest declaring
        // c32-tile32-v1 for pre.metallib AND naming this stage, the accepted geometry below is used
        // byte for byte, exactly as before. With it, the stage issues one 32x32 output tile per
        // 64-thread group - (rows/32)*(out_width/32) groups, one sixteenth of the accepted count -
        // because a null kernel that returns immediately costs 42.51 ms of the real kernel's 50.52 ms
        // on this stage: 84% of it is threadgroup issue. Refuses before any GPU work if the identity
        // does not hold; every other accepted dispatch kind is untouched.
        bool tile32 = false;
        {
          auto g = assets.qkv_override_dispatch.find("pre.metallib");
          auto byl = assets.qkv_override_by_library.find("pre.metallib");
          if (assets.qkv_override_active && g != assets.qkv_override_dispatch.end() &&
              g->second == "c32-tile32-v1" && byl != assets.qkv_override_by_library.end() &&
              byl->second.count(kStageFn[i])) {
            // Only nr_swin32_expand has a 32x32 implementation. Any other stage under this declaration
            // would be an 8x8 kernel handed one sixteenth of the groups it needs, so refuse rather than
            // silently fall back - a silent fallback would compute a wrong frame.
            if (std::string(kStageFn[i]) != "nr_swin32_expand") {
              error = std::string("c32-tile32-v1 is declared for pre.metallib but stage ") + kStageFn[i] +
                      " has no 32x32 implementation in this build; refusing before any GPU work";
              return false;
            }
            if ((rows % 32u) != 0 || (kDispatch[i] % 32u) != 0) {
              error = std::string("c32-tile32-v1 needs rows and out_width to be multiples of 32, but stage ") +
                      kStageFn[i] + " has rows=" + std::to_string(rows) +
                      " out_width=" + std::to_string(kDispatch[i]) + "; refusing before any GPU work";
              return false;
            }
            tile32 = true;
          }
        }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
        // ENCODER-DENSE32-HYBRID (tier "b0"): expand and qkv on the dense kernels, inside this stage's own encoder. Refused together
        // with the c32-tile32-v1 declaration, which would dispatch a different geometry for the same stage.
        if (r.encdense32_active && (i == 0 || i == 3)) {
          if (tile32) { error = "encoder dense (b0): refusing to combine the Block 0 hybrid with the c32-tile32-v1 dispatch declaration"; return false; }
          if ((rows % 64u) != 0) { error = "encoder dense (b0): unit rows are not a multiple of 64"; return false; }
          id<MTLComputePipelineState> pprep = assets.pipeline("enc_dense32b0.metallib", i == 0 ? "nrd32w_decode" : "nrd32w_quant_perm", e);
          id<MTLComputePipelineState> plin = pprep ? assets.pipeline("enc_dense32b0.metallib", "nrd32w_linear_ws8", e) : nil;
          if (!pprep || !plin) { error = "encoder dense (b0): pipeline unavailable: " + e; return false; }
          struct Nrd32LinearHost { uint32_t rows, out_width, in_stride, chunks, windowed, seed_mode, direct, partial, wstride, woff; };
          const uint32_t count = rows * 32u;
          [enc setComputePipelineState:pprep];
          if (i == 0) [enc setBuffer:packed.metal offset:G atIndex:0];                                  // the adapter's packed bytes
          else        [enc setBuffer:y(2).metal offset:G + size_t(3) * rows * 32 * 2 atIndex:0];        // the chunk-3 contract partial
          [enc setBuffer:r.encdense32_x.metal offset:G atIndex:1]; [enc setBytes:&count length:sizeof count atIndex:2];
          [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
          if (i == 0 && (assets.encdense_fuse_mask & 0x20u) != 0) {   // expand+activate in ONE dispatch writing the activation bytes into y(1)
            id<MTLComputePipelineState> pfz = assets.pipeline("enc_dense32b0.metallib", "nrd32w_expand_activate_ws8", e);
            if (!pfz) { error = "encoder dense (b0, fused): pipeline unavailable: " + e; return false; }
            [enc setComputePipelineState:pfz];
            [enc setBuffer:r.encdense32_x.metal offset:G atIndex:0]; [enc setBuffer:r.encdense32_tables[0].metal offset:G atIndex:1];
            [enc setBuffer:y(1).metal offset:G atIndex:2]; [enc setBytes:&rows length:sizeof rows atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(128u / 8, rows / 64, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
          } else {
#endif
          const Nrd32LinearHost lp{rows, i == 0 ? 128u : 96u, 32u, 1u, 0u, 0u, 1u, 0u, 32u, 0u};
          [enc setComputePipelineState:plin];
          [enc setBuffer:r.encdense32_x.metal offset:G atIndex:0]; [enc setBuffer:r.encdense32_tables[i == 0 ? 0 : 1].metal offset:G atIndex:1];
          [enc setBuffer:r.encdense32_x.metal offset:G atIndex:2];   // seed slot: never read (seed_mode 0)
          [enc setBuffer:y(i).metal offset:G atIndex:3]; [enc setBytes:&lp length:sizeof lp atIndex:4];
          [enc dispatchThreadgroups:MTLSizeMake(lp.out_width / 8, rows / 64, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
          }
#endif
          ++dispatches;
        } else
#endif
        if (tile32) {
          const size_t groups = (size_t(rows) / 32u) * (size_t(kDispatch[i]) / 32u);
          [enc dispatchThreads:MTLSizeMake(groups * 64u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else
        [enc dispatchThreads:MTLSizeMake(size_t(rows) * kDispatch[i], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        nr_exp::nr_scope_end(enc);
        ++dispatches; ++encoders;
      }
      {
        id<MTLComputeCommandEncoder> x = nr_exp::gpu_profiler().compute(cb, "block0.export"); nr_exp::nr_scope_adopt(x);
        if (!x) { error = "compute encoder unavailable"; return false; }
        [x setComputePipelineState:export_ps];
        const Buf* xb[5] = {&y(8), &r.skip, &r.encoder, &r.skip_counts, &r.encoder_counts};
        for (NSUInteger i = 0; i < 5; ++i) [x setBuffer:xb[i]->metal offset:G atIndex:i];
        [x setBytes:&s length:sizeof(s) atIndex:5];
        [x dispatchThreads:MTLSizeMake(size_t(rows) * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        nr_exp::nr_scope_end(x);
        ++dispatches; ++encoders;
      }
    }
    return true;
  }
}
} // namespace nr_exp
