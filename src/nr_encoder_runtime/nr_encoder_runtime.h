#pragma once
#import <Metal/Metal.h>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

struct NrEncoderState;
struct NrEncoderContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::shared_ptr<NrEncoderState> state;
};
struct NrEncoderRegion {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0, bytes = 0;
  std::string sha256;
};
struct NrEncoderFrame {
  id<MTLTexture> color = nil;
  // If supplied, must already be committed; we wait for successful completion.
  // Otherwise the caller guarantees producer completion before this call.
  id<MTLCommandBuffer> producer_completion = nil;
  bool reset = true;
  uint32_t frame_index = 0;
};
struct NrEncoderResult {
  int code = 2;
  std::string error, input_sha256, feature_sha256, bundle_root;
  NrEncoderRegion entry, block0_skip;
  std::array<NrEncoderRegion,4> skips; // C256, C128, C64, C32: body order
  uint64_t coverage_values_checked = 0, guarded_outputs_checked = 0;
  bool input_immutable = false, gpu_complete = false;
};
// Immutable generated asset hashes bind this API to the accepted parent bundle.
// Preparation reads model constants only. No activation/expected tensor path.
int nr_encoder_prepare(NrEncoderContext&, const char* asset_root, std::string& error);
// Synchronous, fail-closed reset0-frame 1920x1080 RGBA16F contract. The returned
// regions retain their Metal allocations; callers keep the result alive until
// the downstream consumer completes. No tensor filesystem handoff is required.
// Optional diagnostics are output-only and must name a fresh directory.
// Initial extraction keeps legacy scratch allocations and checks; not optimized.
int nr_encoder_run_frame(NrEncoderContext&, const NrEncoderFrame&, uint8_t poison,
                         NrEncoderResult&, const char* diagnostics_dir = nullptr);
