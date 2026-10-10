#pragma once
// Encode-only block 0 (features + the 34,560-window Swin-32 pass + export), appended to a CALLER-OWNED
// command buffer. Same pinned weights, same kernels, same dispatch geometry as the accepted encoder.
//
// What this drops relative to the accepted encoder, and why it is safe to drop here (experimental lane):
//   * the per-chunk [commit]/[waitUntilCompleted] - 68 round trips per frame. Within one command buffer,
//     separate compute encoders run in submission order and Metal hazard-tracks the shared scratch, so the
//     chunk-to-chunk reuse that the waits used to serialise stays correctly ordered.
//   * the CPU coverage scans and the SHA-256 of activations - research validation, not model work.
//   * the per-frame poison memsets of ~950 MB of buffers, and all diagnostic file/stdout output.
// The coverage OUTPUT buffers still exist and are still bound, because the pinned kernels write to them;
// they are allocated once and never cleared or read. Guard bands are kept and are checkable off the
// encode path. No weight, constant, kernel or dispatch size is changed.
#import <Metal/Metal.h>
#include <array>
#include <string>
#include "nr_experimental_assets.h"

namespace nr_exp {

struct Buf {
  id<MTLBuffer> metal = nil;
  size_t bytes = 0;
  static constexpr size_t kGuard = 256;
  uint8_t* data() const { return static_cast<uint8_t*>(metal.contents) + kGuard; }
};
// Allocates bytes + two guard bands, writes the guard pattern once. Poison is applied only when asked
// (tests and the smoke harness), never on the per-frame encode path.
Buf make_buffer(id<MTLDevice> device, size_t bytes, bool poison, uint8_t poison_value = 0);
size_t guard_errors(const Buf& b);   // CPU-side, off the encode path only

// Everything block 0 needs, allocated once in prepare() and reused every frame.
struct Block0Resources {
  Buf input, weight, features, skip, encoder, deltas, escapes;
  Buf feature_counts, skip_counts, encoder_counts;      // pinned kernels write these; never cleared or read here
  Buf temporal_features, temporal_coverage, rcp_rom;   // the accepted block-0 temporal feature step
  std::array<Buf, 22> scratch{};                        // the accepted chunk scratch set, allocated once
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  // ENCODER-DENSE32-HYBRID (tier "b0"): Block 0's dense expand and qkv. The two float weight tables (expand 128x32, qkv 96x32) are
  // prepacked ONCE in prepare() from the Block 0 weights by enc_dense32b0's own nrd32w_prepack; one float operand buffer sized for
  // the largest unit (rows*32) is reused by both stages of every unit.
  std::array<Buf, 2> encdense32_tables{};
  Buf encdense32_x{};
  bool encdense32_active = false;
#endif
  bool ready = false;
};
bool allocate_block0(id<MTLDevice> device, const Assets& assets, Block0Resources& out, std::string& error);

struct Block0Encoded { uint32_t dispatches = 0, compute_encoders = 0; };

// Appends the whole of block 0 to `cb`. `color` must be a 1920x1080 RGBA16F texture written earlier in the
// SAME command buffer. Nothing is committed, waited on, or read back. Returns false with a reason on any
// validation failure, in which case NOTHING has been encoded.
// `history` and `motion` are non-nil only for a temporal frame. When they are, the accepted block-0 temporal
// feature step (`nr_pre_temporal_features`, 1920x1152 threads, history and motion sampled on the GPU) runs
// between the feature pass and the windowed pass, exactly as the accepted temporal encoder does - and its
// output, not the base features, feeds the window adapter. No CPU read of either texture happens anywhere.
bool encode_block0(id<MTLCommandBuffer> cb, Assets& assets, Block0Resources& res,
                   id<MTLTexture> color, uint32_t frame_index, Block0Encoded& out, std::string& error,
                   id<MTLTexture> history = nil, id<MTLTexture> motion = nil,
                   uint32_t unit_lo = 0, uint32_t unit_hi = 0xffffffffu);

} // namespace nr_exp
