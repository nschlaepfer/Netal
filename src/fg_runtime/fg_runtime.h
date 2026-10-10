#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NSFGSession NSFGSession;

typedef enum NSFGStatus {
  NSFG_STATUS_OK = 0,
  NSFG_STATUS_INVALID_ARGUMENT = 1,
  NSFG_STATUS_UNSUPPORTED_PROFILE = 2,
  NSFG_STATUS_NEEDS_RESET = 3,
  NSFG_STATUS_OUT_OF_ORDER_FRAME = 4,
  NSFG_STATUS_BUSY = 5,
  NSFG_STATUS_DEVICE_ERROR = 6,
  NSFG_STATUS_INTERNAL_ERROR = 7,
} NSFGStatus;

typedef enum NSFGSessionPhase {
  NSFG_SESSION_UNINITIALIZED = 0,
  NSFG_SESSION_NEEDS_RESET = 1,
  NSFG_SESSION_READY = 2,
  NSFG_SESSION_EVALUATING = 3,
  NSFG_SESSION_FAULTED = 4,
} NSFGSessionPhase;

typedef enum NSFGEx2Mode {
  NSFG_EX2_REFERENCE = 0,
  NSFG_EX2_COMPACT = 1,
} NSFGEx2Mode;

typedef enum NSFGAuxiliaryMode {
  NSFG_AUXILIARY_INTERNAL_SMALL_FLOW = 0,
  NSFG_AUXILIARY_LIVE_EARLY_RESOURCES = 1,
} NSFGAuxiliaryMode;

typedef enum NSFGSpanDomain {
  NSFG_SPAN_EARLY = 0,
  NSFG_SPAN_TAIL = 1,
  NSFG_SPAN_OUTPUT_COPY = 2,
} NSFGSpanDomain;

typedef struct NSFGCommandSpan {
  NSFGSpanDomain domain;
  uint32_t ordinal;
  double cpu_commit_wait_ms;
  double gpu_ms;
} NSFGCommandSpan;

// All Metal objects are unretained caller references on entry. The session
// retains the device and queue for its lifetime and retains the most recently
// accepted current backbuffer only until the next evaluation/reset.
typedef struct NSFGCreateInfo {
  void* metal_device;
  void* metal_command_queue;
  const char* metallib_directory;
  const char* model_directory;
  const char* rcp_rom_path;
  const char* sigmoid_lut_path;
  const char* ex2_unit_table_path;
  const char* ex2_fraction_table_path;
  uint32_t width;
  uint32_t height;
  NSFGEx2Mode ex2_mode;
  NSFGAuxiliaryMode auxiliary_mode;
  uint8_t poison_byte;
  bool enable_audit;
  bool enable_profiling;
  // Metal 4 mode: metal_command_queue is ignored; frames are encoded into the
  // MTL4CommandBuffer passed to NSFGSessionEvaluateMetal4 (audit and profiling
  // are unavailable, and reset frames skip the CPU-side poison of the state,
  // which the poison-invariance receipts show does not reach the output).
  bool metal4;
} NSFGCreateInfo;

typedef struct NSFGFrameInfo {
  void* current_backbuffer_rgba8;
  void* previous_backbuffer_rgba8;
  void* motion_rg16f;
  void* depth_r32f;
  void* generated_target_rgba8;
  uint64_t frame_id;
  bool reset;
} NSFGFrameInfo;

typedef struct NSFGFrameResult {
  bool generated_output_valid;
  uint32_t selected_launch_count;
  uint64_t frame_id;
  uint64_t resource_allocations_total;
  uint64_t resource_allocations_during_evaluate;
} NSFGFrameResult;

typedef struct NSFGSessionStats {
  NSFGSessionPhase phase;
  uint64_t reset_count;
  uint64_t generated_frame_count;
  uint64_t last_accepted_frame_id;
  uint64_t resource_allocations_total;
  uint64_t resource_allocations_after_warmup;
  uint64_t input_mismatched_bytes;
  uint64_t canary_mismatched_bytes;
} NSFGSessionStats;

// On failure, *out_session is null and the caller should present its current
// backbuffer. No generated target is touched.
NSFGStatus NSFGSessionCreate(const NSFGCreateInfo* info,
                             NSFGSession** out_session);

// Reset frames warm the temporal state and intentionally produce no generated
// frame. Non-reset frames require a monotonically adjacent frame_id and a
// previous_backbuffer matching the last accepted current-backbuffer object.
// Any non-OK result leaves generated_target_rgba8 untouched.
NSFGStatus NSFGSessionEvaluate(NSFGSession* session,
                               const NSFGFrameInfo* frame,
                               NSFGFrameResult* result);

// Encode the frame into the caller's command buffer: an id<MTL4CommandBuffer>
// the caller has begun and will end and commit, or an id<MTLCommandBuffer> the
// caller will commit (Metal 3 games). Same contract as NSFGSessionEvaluate
// otherwise. Requires a session created with metal4=true.
NSFGStatus NSFGSessionEvaluateMetal4(NSFGSession* session,
                                     const NSFGFrameInfo* frame,
                                     void* mtl4_command_buffer,
                                     NSFGFrameResult* result);

// Resources of the last generated frame that a resolution-decoupled compositor needs (unretained
// Metal objects owned by the session, valid until the next evaluate): the network's per-pixel
// output (640x368 x 8 halves: flow to previous, flow to current, blend logit, ...), launch 53's
// candidate flow buffers (320x180 ulong, half2 pairs in normalised units, 0x7fffffff7fffffff =
// invalid), launch 54's RG16F blend weights (320x180) and the sigmoid LUT (65536 floats as bits,
// indexed by the logit's half bits).
typedef struct NSFGTailResources {
  void* motion_half8_buffer;
  uint64_t motion_offset;
  void* candidate_a_buffer;
  void* candidate_b_buffer;
  uint64_t candidate_offset;
  void* blend_rg16f_texture;
  void* sigmoid_lut_buffer;
  uint64_t sigmoid_offset;
} NSFGTailResources;
NSFGStatus NSFGSessionCopyLastTailResources(const NSFGSession* session,
                                            NSFGTailResources* resources);

// Discards temporal state without evaluating a frame. The next evaluation
// must carry reset=true.
NSFGStatus NSFGSessionInvalidate(NSFGSession* session);

NSFGStatus NSFGSessionGetStats(const NSFGSession* session,
                               NSFGSessionStats* stats);

// Audit-only checkpoint access. Reset evaluations expose 9 hashes and
// generated evaluations expose 76 in selected-launch order. Non-audit
// sessions expose zero. `sha256` must have room for 65 bytes.
size_t NSFGSessionLastCheckpointCount(const NSFGSession* session);
NSFGStatus NSFGSessionCopyLastCheckpoint(const NSFGSession* session,
                                         size_t index, char* sha256,
                                         size_t capacity);

// Diagnostic-only command-buffer spans. Enabling this instrumentation creates
// a timing-perturbed run which cannot replace an untimed correctness receipt.
size_t NSFGSessionLastCommandSpanCount(const NSFGSession* session);
NSFGStatus NSFGSessionCopyLastCommandSpan(const NSFGSession* session,
                                          size_t index,
                                          NSFGCommandSpan* span);

// Returns a session-owned, null-terminated diagnostic valid until the next API
// call on this session. It is never a substitute for NSFGStatus handling.
const char* NSFGSessionLastError(const NSFGSession* session);

void NSFGSessionDestroy(NSFGSession* session);

const char* NSFGStatusName(NSFGStatus status);

#ifdef __cplusplus
}  // extern "C"
#endif
