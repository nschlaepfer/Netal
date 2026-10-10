#pragma once
// Per-encoder GPU timestamp profiling for the experimental lane (Opus-owned).
//
// Apple M3 Max exposes the `timestamp` counter set with STAGE-boundary sampling, so every compute encoder can
// carry a start and end GPU timestamp at no cost to correctness. That gives per-launch GPU attribution in one
// run, instead of splitting the chain across command buffers and hoping the pieces compose.
//
// It is OFF unless NR_EXP_GPU_TIMESTAMPS=1 is set in the environment, so the shipping encode path is unchanged.
// Timestamps are resolved by the harness AFTER the command buffer completes; nothing is read during encoding.
#import <Metal/Metal.h>
#include <string>
#include <vector>

namespace nr_exp {

struct GpuSpan { std::string label; double ms = 0; uint64_t start = 0, end = 0; };   // ms is calibrated; start/end are RAW GPU ticks

// What a resolve actually saw, so a profile is never mistaken for a complete or exclusive accounting.
struct GpuProfileStats {
  size_t encoders_expected = 0;      // encoders we attached sampling to
  size_t samples_returned = 0;       // timestamps the counter buffer gave back
  size_t spans_valid = 0, spans_dropped = 0;   // dropped = error value or non-monotonic
  double ticks_to_ms = 0;            // calibrated GPU-tick -> millisecond factor (1e-6 only if the timebase is ns)
  bool calibrated = false;           // true when a paired CPU/GPU sample gave the factor
  double union_ms = 0, sum_ms = 0;   // union of the intervals vs the naive sum: their gap IS the overlap
  double overlap_ms = 0, span_extent_ms = 0;   // extent = last end - first start
  bool batching_disabled_for_profile = false;  // per-encoder sampling needs one encoder per dispatch
};

struct GpuProfiler {
  id<MTLDevice> device = nil;
  id<MTLCounterSampleBuffer> samples = nil;
  std::vector<std::string> labels;      // one per encoder, in encode order
  size_t capacity = 0, clamped_from = 0;   // clamped_from != 0 means the profile covers only the first `capacity` encoders
  bool enabled = false;

  bool prepare(id<MTLDevice> dev, size_t max_encoders, std::string& error);
  void begin_frame() { labels.clear(); }
  // Returns a compute encoder with sampling attached when enabled, or a plain one when not.
  id<MTLComputeCommandEncoder> compute(id<MTLCommandBuffer> cb, const char* label);
  // Call after the command buffer has completed. Returns one span per encoder, in encode order.
  std::vector<GpuSpan> resolve(std::string& error) const;
  std::vector<GpuSpan> resolve(std::string& error, GpuProfileStats& stats) const;
  // Paired CPU/GPU sample (Apple's documented conversion). Returns false when the device cannot provide it.
  bool calibrate(double& ticks_to_ms) const;
};

// Process-wide instance used by the experimental body; the harness owns its lifetime through prepare().
GpuProfiler& gpu_profiler();

// Every compute encoder this runtime opens is registered here and cleared when it is ended, so an exception
// path can close the one that is still open. The caller's command buffer must always be left committable: it
// carries the game's own rendering, and the game will commit it whatever happens to our work.
void encoder_opened(id<MTLCommandEncoder> e);
void encoder_ended();
void close_open_encoder();
bool gpu_profiling_requested();      // either mode
bool gpu_profiling_regions();        // NR_EXP_GPU_TIMESTAMPS=regions - sampling WITHOUT disabling batching
bool gpu_profiling_per_dispatch();   // NR_EXP_GPU_TIMESTAMPS=1 - one encoder per dispatch, changes execution

} // namespace nr_exp
