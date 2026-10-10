// Output-only post-completion boundary hashes, for localizing where this runtime first diverges from the
// accepted lane's preserved reference on the same input.
//
// Everything here is read AFTER the caller's command buffer has reached a terminal state and the body's
// completion step has run. Nothing is read during encoding, no current-frame GPU value is consulted before
// completion, and none of it feeds execution: these are diagnostics, compared against a receipt, never
// injected into a stage. Off unless NR_EXP_BOUNDARY_HASHES=1.
#pragma once
#include <string>

namespace nr_exp {

struct BoundaryHashes {
  bool captured = false;
  // The accepted receipt's six encoder regions are nested prefixes of one pool region at offset 256, so
  // these are prefixes of this runtime's block-0 skip buffer at the same five lengths, plus the full region.
  std::string entry, skip_c256, skip_c128, skip_c64, skip_c32, block0_skip;
  // This runtime's separate encoder-tier buffer, which is NOT one of those prefixes; kept for reference.
  std::string encoder_buffer;
  // Body boundaries, taken from the body's completion result. Same four the accepted receipt records.
  std::string middle_view, decoder_pre_image, pool_final, final_surface;
  std::string decoder_pre_image_source;
};

// Encoder-chain working set touched per frame (see EncoderEncoded for what this does and does not count).
struct ChainBytes { unsigned long long total = 0, wide = 0, coverage = 0; };
const ChainBytes& last_chain_bytes();
void set_last_chain_bytes(const ChainBytes&);

// Bounded-chunk range for the body, consulted only inside NR_EXPERIMENTAL_CHUNKED regions. Passing it
// this way rather than as a new parameter keeps nr_body_encode's signature identical to the accepted v17
// runtime, so the "differs only in flagged regions" reduction still holds.
struct BodyChunk { uint32_t lo = 0, hi = 0xffffffffu; };
const BodyChunk& body_chunk();
void set_body_chunk(const BodyChunk&);

bool boundary_hashes_requested();
const BoundaryHashes& last_boundary_hashes();
void set_last_boundary_hashes(const BoundaryHashes&);

}  // namespace nr_exp
