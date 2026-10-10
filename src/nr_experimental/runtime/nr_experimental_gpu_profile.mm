#include "nr_experimental_gpu_profile.h"
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include "nr_experimental_encoder_scope_adoption.h"
#import <Foundation/Foundation.h>
#include <cstdlib>

namespace nr_exp {

GpuProfiler& gpu_profiler() { static GpuProfiler p; return p; }

// The process-global open-encoder tracker is retired: it could not see body exceptions converted into
// return codes, untracked blits, stale ended encoders or Objective-C unwinding. Ownership now lives in a
// per-invocation ScopeHolder built on Codex's EncoderScope helper.
void encoder_opened(id<MTLCommandEncoder> e) { nr_scope_adopt(e); }
void encoder_ended() { }   // retained for source compatibility; closing goes through the owning scope
void close_open_encoder() { if (ScopeHolder* h = current_scope_holder()) h->close_all(); }
// Two profiling modes, deliberately distinguished.
//
//   NR_EXP_GPU_TIMESTAMPS=1        PER-DISPATCH. Forces one encoder per dispatch in the encoder Walk, which
//                                  CHANGES what is executed. Its totals are a RANKING and are not comparable
//                                  to a normal run - the observer's effect is large and is quantified in
//                                  build/nr-accept-v1/OBSERVER-EFFECT.json.
//   NR_EXP_GPU_TIMESTAMPS=regions  BATCHED REGIONS. Sampling is attached to exactly the encoders the runtime
//                                  would have created anyway; batching is NOT disabled. The computation, its
//                                  inputs and the encoder structure are preserved, so the spans are the real
//                                  regions the frame pays for. A region may cover many dispatches, and it is
//                                  reported as a region - never as an individual stage's cost.
//
// Block 0 opens one encoder per stage in BOTH modes (it never batched), so its spans mean the same thing in
// either; the difference is confined to the encoder Walk for blocks 1-22.
static const char* profiling_env() { const char* v = std::getenv("NR_EXP_GPU_TIMESTAMPS"); return v ? v : ""; }
bool gpu_profiling_regions() {
  static const bool on = [] { const char* v = profiling_env(); return v[0] == 'r'; }();
  return on;
}
bool gpu_profiling_per_dispatch() {
  static const bool on = [] { const char* v = profiling_env(); return v[0] == '1'; }();
  return on;
}
bool gpu_profiling_requested() { return gpu_profiling_per_dispatch() || gpu_profiling_regions(); }

bool GpuProfiler::prepare(id<MTLDevice> dev, size_t max_encoders, std::string& error) {
  device = dev; capacity = 0; samples = nil; enabled = false;
  if (!gpu_profiling_requested()) return true;                 // not requested: stay inert
  if (![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { error = "stage-boundary counter sampling unavailable"; return false; }
  id<MTLCounterSet> ts = nil;
  for (id<MTLCounterSet> cs in dev.counterSets) if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = cs;
  if (!ts) { error = "no timestamp counter set"; return false; }
  // Metal caps a counter sample buffer at 32,768 bytes, i.e. 4,096 timestamps, i.e. 2,048 encoders. Clamp
  // rather than fail, and record the clamp so a truncated profile is never mistaken for a complete one.
  const size_t kMaxSamples = 4096;
  size_t sample_count = max_encoders * 2;
  if (sample_count > kMaxSamples) { sample_count = kMaxSamples; clamped_from = max_encoders; }
  MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
  d.counterSet = ts; d.storageMode = MTLStorageModeShared; d.sampleCount = sample_count;
  NSError* err = nil;
  samples = [dev newCounterSampleBufferWithDescriptor:d error:&err];
  if (!samples) { error = std::string("counter sample buffer: ") + (err ? err.localizedDescription.UTF8String : "unknown"); return false; }
  capacity = sample_count / 2; enabled = true;
  return true;
}

id<MTLComputeCommandEncoder> GpuProfiler::compute(id<MTLCommandBuffer> cb, const char* label) {
  if (!enabled || labels.size() >= capacity) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (e && label) e.label = [NSString stringWithUTF8String:label];
    return e;
  }
  const NSUInteger i = labels.size();
  MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
  MTLComputePassSampleBufferAttachmentDescriptor* a = pd.sampleBufferAttachments[0];
  a.sampleBuffer = samples;
  a.startOfEncoderSampleIndex = i * 2;
  a.endOfEncoderSampleIndex = i * 2 + 1;
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDescriptor:pd];
  if (e) { if (label) e.label = [NSString stringWithUTF8String:label]; labels.push_back(label ? label : "unlabelled"); }
  return e;
}

bool GpuProfiler::calibrate(double& ticks_to_ms) const {
  if (!device) return false;
  // Apple's documented pairing: two samples bracket the same instant on both clocks, so the ratio of the
  // GPU-tick delta to the CPU-nanosecond delta converts ticks to time without assuming a nanosecond timebase.
  MTLTimestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
  [device sampleTimestamps:&c0 gpuTimestamp:&g0];
  usleep(2000);   // a real interval between the two pairings; this runs once, after completion
  [device sampleTimestamps:&c1 gpuTimestamp:&g1];
  if (g1 <= g0 || c1 <= c0) return false;
  const double cpu_ns = double(c1 - c0), gpu_ticks = double(g1 - g0);
  ticks_to_ms = (cpu_ns / gpu_ticks) / 1e6;   // CPU timestamps are nanoseconds
  return ticks_to_ms > 0 && std::isfinite(ticks_to_ms);
}

std::vector<GpuSpan> GpuProfiler::resolve(std::string& error) const {
  GpuProfileStats ignored; return resolve(error, ignored);
}

std::vector<GpuSpan> GpuProfiler::resolve(std::string& error, GpuProfileStats& stats) const {
  std::vector<GpuSpan> out;
  stats = GpuProfileStats{};
  stats.batching_disabled_for_profile = gpu_profiling_per_dispatch();   // false in regions mode: batching is preserved
  if (!enabled || labels.empty()) return out;
  stats.encoders_expected = labels.size();
  double f = 1e-6;                              // fallback: assume a nanosecond timebase, and say so
  stats.calibrated = calibrate(f);
  if (!stats.calibrated) f = 1e-6;
  stats.ticks_to_ms = f;
  NSData* d = [samples resolveCounterRange:NSMakeRange(0, labels.size() * 2)];
  if (!d) { error = "counter resolve returned nothing"; return out; }
  const MTLCounterResultTimestamp* t = static_cast<const MTLCounterResultTimestamp*>(d.bytes);
  const size_t n = d.length / sizeof(MTLCounterResultTimestamp);
  stats.samples_returned = n;
  for (size_t i = 0; i + 1 < n && i / 2 < labels.size(); i += 2) {
    const uint64_t a = t[i].timestamp, b = t[i + 1].timestamp;
    if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) { ++stats.spans_dropped; continue; }
    out.push_back({labels[i / 2], double(b - a) * f, a, b});
    ++stats.spans_valid;
  }
  // Union of the raw intervals: encoders can overlap on the GPU, so the naive sum over-counts. The union is
  // the honest occupancy, and (sum - union) is exactly how much overlap the sum double-counted.
  std::vector<std::pair<uint64_t, uint64_t>> iv;
  iv.reserve(out.size());
  for (const auto& s : out) { iv.push_back({s.start, s.end}); stats.sum_ms += s.ms; }
  std::sort(iv.begin(), iv.end());
  uint64_t union_ticks = 0, cur_a = 0, cur_b = 0; bool open = false;
  for (const auto& p : iv) {
    if (!open) { cur_a = p.first; cur_b = p.second; open = true; continue; }
    if (p.first > cur_b) { union_ticks += cur_b - cur_a; cur_a = p.first; cur_b = p.second; }
    else if (p.second > cur_b) cur_b = p.second;
  }
  if (open) union_ticks += cur_b - cur_a;
  stats.union_ms = double(union_ticks) * f;
  stats.overlap_ms = stats.sum_ms - stats.union_ms;
  if (!iv.empty()) stats.span_extent_ms = double(iv.back().second - iv.front().first) * f;
  return out;
}
} // namespace nr_exp

namespace nr_exp {
// Per-invocation scope holder. Thread-local rather than global: two encode invocations on different threads
// must never share encoder ownership. Replaces the process-global open-encoder tracker.
static thread_local ScopeHolder* g_scope_holder = nullptr;
ScopeHolder* current_scope_holder() { return g_scope_holder; }
void set_current_scope_holder(ScopeHolder* h) { g_scope_holder = h; }
}
