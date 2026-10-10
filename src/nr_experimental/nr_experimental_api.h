#pragma once
// v1 contract: actual NR on caller-owned Metal3 commands, experimental only.
// Ownership ACK: Codex interface/bridge, Opus runtime. No exact-parity implication.
#import <Metal/Metal.h>
#include <cstdint>
#include <memory>
#include <string>

namespace netal_nr_experimental {
enum class Arithmetic { OriginalKernels, ApproximateNative };
enum class MotionUnits { Unspecified, PixelsFullRes, NormalizedDeviceCoords, TexelsRenderRes };
enum class MotionDirection { Unspecified, CurrentToPrevious };
// DiscardOnly discards NR OUTPUT, NEVER the caller's command buffer. That buffer
// already contains the game's rendering and must remain valid for submission.
enum class EncodeStatus { RejectedBeforeEncoding, Encoded, EncodedDiscardOnly };
struct Context;
struct Ticket;
using ContextPtr = std::shared_ptr<Context>;
using TicketPtr = std::shared_ptr<Ticket>;

struct PrepareOptions {
  std::string runtime_bundle;
  uint32_t width = 1920, height = 1080;
  Arithmetic arithmetic = Arithmetic::OriginalKernels;
};
struct FrameInputs {
  id<MTLTexture> color = nil; // Full-size RGBA16F, producer earlier in SAME command buffer.
  id<MTLTexture> motion = nil; // Full-size RG16F; required for temporal.
  // Canonical model mapping: history_uv = current_uv + motion / full_size.
  // Bridge owns resampling/unit conversion. Initial runtime accepts only
  // PixelsFullRes, CurrentToPrevious, y-down, scales1; it rejects other units.
  // Reset frames may omit motion and leave its convention unspecified.
  MotionUnits motion_units = MotionUnits::Unspecified;
  MotionDirection motion_direction = MotionDirection::Unspecified;
  float motion_scale_x = 1, motion_scale_y = 1;
  bool motion_y_down = true;
  bool reset = true;
  uint64_t stream_id = 0, frame_id = 0;
};
struct Report {
  std::string error, model_identity, weights_identity, arithmetic_label;
  bool model_encoded = false, history_advanced = false, experimental = true;
  uint32_t model_dispatches = 0;
  uint64_t stream_id = 0, frame_id = 0;
  // Set only when actually measured for NR, never copied from whole game CB time.
  bool nr_gpu_time_valid = false;
  double nr_gpu_ms = 0;
  bool encode_cpu_time_valid = false, prepare_cpu_time_valid = false;
  double encode_cpu_ms = 0, prepare_cpu_ms = 0;
};

// CPU preparation may load immutable weights/libraries; do it outside the game hook.
// No new backend, weight changes or research-capture dependencies.
bool prepare(id<MTLDevice>, const PrepareOptions&, ContextPtr&, Report&);

// Append actual model work, never commit or wait. No CPU reads of current-frame
// GPU values, archived activation ingress or reference comparison in this path.
// At most one admitted frame per context initially. Busy/unsupported rejects.
// A non-null ticket MUST be retained/finalized even after EncodedDiscardOnly,
// because already-encoded scratch work cannot be rolled back.
// Close every encoder opened by the runtime on success AND exception paths.
// No return path may require the game to abandon its command buffer. Keep NR
// scratch/resources alive through genuine command completion even on failure.
EncodeStatus encode(const ContextPtr&, const FrameInputs&,
                    id<MTLCommandBuffer> caller_command, TicketPtr&, Report&);

// Only valid for an Encoded ticket; the bridge owns export to the game texture.
// The texture is not CPU-ready until the command buffer completes.
id<MTLTexture> output_texture(const TicketPtr&);

// Call exactly once from the SAME command buffer's completion callback. Never
// wait here. Verify identity/status and advance or invalidate OWN NR history.
// Ticket/context and all used resources remain alive through this callback.
// Reject wrong-buffer and ALL nonterminal calls without consuming the ticket or
// releasing its in-flight reservation, including EncodedDiscardOnly tickets.
// Abandonment of an unsubmitted buffer is not expressed through this function.
void complete(const TicketPtr&, id<MTLCommandBuffer> completed_command, Report&);

// Mark continuity lost (fallback, missed frame, resize, reset, error, mode change).
// Do not mutate in-flight GPU resources; force a reset at next admission.
void invalidate(const ContextPtr&, const char* reason);
} // namespace netal_nr_experimental
