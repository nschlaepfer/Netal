// nr_temporal.h — versioned groundwork (v1, experimental; the production reset-only route does not include this header) for the shared
// temporal state consumed by BOTH endpoints: Codex's encoder runtime (Block0 reads history at its +8) and Fable's body runtime (post 155 reads
// history at +88; copy 156 writes history). Contract from the audited RTX reference (docs/NR_TEMPORAL_REFERENCE_READY.md, root b2f44873…):
//   - history is an INTERNAL persistent RGBA16F 1920x1080 array, distinct from the caller output; 155 stores to the caller output;
//     156 copies caller output -> history; history after frame N == output N; history is read through a normalized/clamp/LINEAR view.
//   - motion is a caller RG16F 1920x1080 texture read through a normalized/clamp/LINEAR view (zero in the tested reference).
//   - confidence is a model constant (half 0.73974609375, transfer 150) already inside the staged launch-155 weight blob: no caller input.
//   - history in a live sequence comes ONLY from that sequence's prior native output (never a downloaded expected frame).
#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <string>
struct NrTexSpec { bool normalized = true; bool clamp_to_edge = true; bool linear = true; };   // the recorded sampler modes of history view 1 and motion 6
struct NrTemporalState {
  id<MTLTexture> history = nil;            // internal, persistent, RGBA16F WxH; written only by the copy after launch 155 completes
  uint32_t width = 0, height = 0;
  uint32_t frames_since_reset = 0;         // 0 before any frame; the next frame is a reset frame iff the caller says so (reset=true), never inferred
  std::string history_sha256;              // hash recorded right after each frame's copy (== that frame's final output hash by the reference contract)
  std::string last_output_sha256;          // the body's final-surface hash of the last frame; a temporal frame requires history_sha256 == last_output_sha256
  NrTexSpec history_view, motion_view;     // both default to normalized/clamp/linear as recorded
};
// Create the internal history texture (RGBA16F, shader read + write, shared storage) filled with the poison byte; frames_since_reset = 0.
inline bool nr_temporal_state_create(NrTemporalState& s, id<MTLDevice> dev, uint32_t w, uint32_t h, uint8_t poison) {
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:w height:h mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared; s.history = [dev newTextureWithDescriptor:d];
  if (!s.history) return false; s.width = w; s.height = h; s.frames_since_reset = 0; s.history_sha256.clear(); s.last_output_sha256.clear();
  std::string pz(size_t(w) * h * 8, char(poison)); [s.history replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:pz.data() bytesPerRow:w * 8]; return true;
}
// Launch-155 record construction rule, verified on CPU against the captured reset[1,0] second call (build/nr_temporal/record-construct-v1):
// start from the FULL accepted frame-0 record of the descriptor class; write only the two binding markers and the eight dimension/reciprocal
// words; pointers stay the runtime's own replay-local encodings. Returns false if the base record is not 184 bytes.
inline bool nr_temporal_build_post_record(const uint8_t* base184, size_t n, uint32_t w, uint32_t h, uint32_t history_marker, uint32_t motion_marker, uint8_t* out184) {
  if (n != 184 || !base184 || !out184) return false; for (size_t i = 0; i < 184; ++i) out184[i] = base184[i];
  auto put32 = [&](size_t off, uint32_t v) { for (int i = 0; i < 4; ++i) out184[off + i] = uint8_t(v >> (8 * i)); };
  auto putf = [&](size_t off, float f) { uint32_t v; __builtin_memcpy(&v, &f, 4); put32(off, v); };
  put32(88, history_marker); put32(96, motion_marker);
  putf(124, float(w)); putf(128, float(h)); putf(132, 1.0f / float(w)); putf(136, 1.0f / float(h));
  putf(148, float(w)); putf(152, float(h)); putf(156, 1.0f / float(w)); putf(160, 1.0f / float(h)); return true;
}
