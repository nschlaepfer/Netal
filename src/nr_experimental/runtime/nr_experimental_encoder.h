#pragma once
// Encode-only encoder blocks 1-22 (C32 -> wide64 -> C128 -> C256) plus the four native skip exports and the
// middle export, appended to a CALLER-OWNED command buffer. Same pinned weights, kernels and dispatch
// geometry as the accepted encoder; the per-block commit/wait, the coverage scans, the SHA-256 of every
// intermediate and the per-block reallocation are removed. No weight or dispatch size is changed.
#import <Metal/Metal.h>
#include <deque>
#include <string>
#include <vector>
#include "nr_experimental_assets.h"
#include "nr_experimental_block0.h"

namespace nr_exp {

// Persistent buffers for the whole encoder chain. Allocation happens once, in prepare(), by walking the exact
// same call sequence the encode path walks, so the two cannot disagree.
//
// Two pools, because holding every block's intermediates at once would be about 15 GB. `carry` holds what
// chains between blocks (block outputs, rewindow results, projections a downsample consumes, the skips and
// the exports). `scratch` holds the per-block stage buffers and every coverage buffer, and is rewound at the
// start of each block: within one command buffer, successive compute encoders run in order and Metal
// hazard-tracks these buffers, so reuse across blocks is correctly serialised, exactly as in block 0.
struct Pool {
  enum class Mode { Measure, Use };
  Mode mode = Mode::Measure;
  std::vector<size_t> sizes;     // measured maximum per slot index
  std::vector<Buf> slots;        // allocated to those maxima
  std::deque<Buf> probes;        // pointer-stable placeholders used during the measuring walk
  size_t next = 0;
  id<MTLDevice> device = nil;
  std::string error;
  Buf* get(size_t bytes);        // nullptr with `error` set on any failure
  void rewind() { next = 0; }
  bool allocate();               // Measure -> Use: allocates every slot at its measured maximum
  size_t total_bytes() const;
};

struct EncoderResources {
  Pool carry, scratch;
  std::vector<Buf> weights;        // blocks 1..22, uploaded once
  Buf layout;                      // nr_pre_entry_layout output: block 0's encoder export relaid for blocks 1-22
  Buf entry;                       // block22-middle native export: the body's entry view
  std::vector<Buf> skips;          // native skip exports for blocks 4, 8, 14, 22 in body order (C256..C32)
  // Opt-in DIAGNOSTIC capture of one wide QKV stage output, so a candidate's GROUP output can be
  // compared directly instead of being inferred from the final frame hash. Off unless
  // NR_EXP_CAPTURE_QKV_C64 is set; nothing is allocated, encoded or read back otherwise, and the
  // capture is never an input to anything.
  Buf qkv_capture;        // the C64 QKV stage OUTPUT slice
  Buf qkv_capture_in;     // the x operand slice that stage actually consumed
  Buf qkv_capture_cov;     // that stage's per-output write counters AFTER its dispatch
  Buf qkv_capture_cov_pre; // and BEFORE it - the counters are cumulative across buffer reuse and are
                           // never cleared, so only the DELTA can show exact-once
  // Opt-in derived weight layout (NR_EXP_PREPACK_WEIGHTS=1). One prepacked copy per block, built ONCE in
  // prepare() from the original bytes and reused every frame. Layout preparation only: the same e4m3 bytes in
  // a different order, no decode, no requantisation, and the originals are untouched.
  std::vector<Buf> prepacked_weights;
  bool prepack_active = false;
  size_t prepack_verified_bytes = 0;   // bytes proved equal to the original under the permutation
  Buf qkv_capture_guards;  // the PRODUCER buffer's own guard bands, snapshotted after its dispatch and
                           // before the pool reuses it. Guards on the capture destinations only ever
                           // tested the copy; these test what the kernel actually wrote. (Codex.)
  size_t qkv_capture_bytes = 0;    // bytes actually copied THIS frame; reset per MODEL FRAME
  size_t qkv_capture_in_bytes = 0, qkv_capture_cov_bytes = 0;
  size_t qkv_capture_needed = 0;   // extent recorded by the measuring walk, used to size the buffer
  bool qkv_capture_done = false;   // reset per frame, so a multi-frame run does not keep frame 0
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  // ENCODER-DENSE-WS8: per block (index block-1), the five float weight tables of a selected wide tier (expand, mid,
  // mix, qkv, project), prepacked ONCE in prepare() by the dense library's own nrd_prepack and reused every frame.
  std::vector<std::vector<Buf>> encdense_tables;
  // ENCODER-DENSE32-HYBRID (tier "32"): per C32 walk block (index block-1, blocks 1-4), the [expand, qkv] float tables.
  std::vector<std::vector<Buf>> encdense32_tables;
  double encdense_prepack_gpu_ms = 0;
#endif
  bool ready = false;
};

// The wide-chain threadgroup width actually in force (NR_EXP_WIDE_TG, default 64). Exposed so that an
// unsupported candidate configuration can be refused during PREPARATION, before any frame is encoded,
// rather than only at the encode-time admission check.
size_t wide_threadgroup_width();

bool allocate_encoder(id<MTLDevice> device, Assets& assets, const Block0Resources& b0, EncoderResources& out, std::string& error);

struct EncoderEncoded {
  uint32_t dispatches = 0, compute_encoders = 0;
  // Bytes of buffer storage bound per dispatch, summed. A lower bound on traffic: it counts each bound
  // buffer once per dispatch and cannot see cache reuse, so treat it as "resident working set touched",
  // not as measured DRAM traffic.
  unsigned long long bound_bytes = 0, bound_bytes_wide = 0, bound_bytes_coverage = 0;
};

// Appends blocks 1-22 and the exports. `b0.encoder` (block 0's 17,694,720-byte output) is the input.
// `tier_mask` is a PROFILING aid only: bit 0 C32 blocks 1-4, bit 1 wide-64 blocks 5-8, bit 2 C128 blocks 9-14,
// bit 3 C256 blocks 15-22, bit 4 the native exports. The default encodes the whole chain; a partial mask
// produces an incomplete result and must never be used for a frame that is published.
// `block_lo`/`block_hi` bound which of blocks 1-22 are ENCODED; the walk still derives all state, so a
// later call with a later range resumes exactly. Defaults encode the whole chain, unchanged.
bool encode_encoder(id<MTLCommandBuffer> cb, Assets& assets, Block0Resources& b0, EncoderResources& res,
                    EncoderEncoded& out, std::string& error, uint32_t tier_mask = 0xffffffffu,
                    uint32_t block_lo = 0, uint32_t block_hi = 0xffffffffu);

} // namespace nr_exp
