#import "span_recorder.h"

#include <chrono>
#include <stdexcept>

#include <atomic>

namespace {
thread_local bool g_enabled = false;
thread_local bool g_defer = false;
thread_local std::vector<NSFGCommandSpan> g_spans;
id<MTLCommandBuffer> g_last_deferred = nil;
std::atomic<int> g_deferred_failures{0};
}

void NSFGSpanSetDeferWaits(bool defer) { g_defer = defer; }

void NSFGSpanFlush(void) {
  id<MTLCommandBuffer> last = g_last_deferred;
  g_last_deferred = nil;
  if (last != nil) [last waitUntilCompleted];
  if (g_deferred_failures.exchange(0) != 0)
    throw std::runtime_error("a deferred FG command buffer did not complete");
}

void NSFGSpanBegin(bool enabled) {
  g_enabled = enabled;
  g_spans.clear();
}

void NSFGWaitAndRecord(id<MTLCommandBuffer> command, NSFGSpanDomain domain) {
  if (g_defer && domain != NSFG_SPAN_OUTPUT_COPY) {
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (completed.status != MTLCommandBufferStatusCompleted) ++g_deferred_failures;
    }];
    [command commit];
    g_last_deferred = command;
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  [command commit];
  [command waitUntilCompleted];
  if (g_defer) {
    g_last_deferred = nil;   // the output copy completes after every earlier buffer on this queue
    if (g_deferred_failures.exchange(0) != 0)
      throw std::runtime_error("a deferred FG command buffer did not complete");
  }
  const auto end = std::chrono::steady_clock::now();
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
  if (!g_enabled) return;
  const double gpu_ms = command.GPUEndTime > command.GPUStartTime
      ? (command.GPUEndTime - command.GPUStartTime) * 1000.0
      : 0.0;
  g_spans.push_back({
      domain,
      static_cast<std::uint32_t>(g_spans.size()),
      std::chrono::duration<double, std::milli>(end - start).count(),
      gpu_ms,
  });
}

const std::vector<NSFGCommandSpan>& NSFGRecordedSpans(void) {
  return g_spans;
}
