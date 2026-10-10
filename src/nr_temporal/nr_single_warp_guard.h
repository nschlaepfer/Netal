// nr_single_warp_guard.h — HOST-SIDE dispatch guard for the opt-in single-warp launch-155 candidate.
//
// Codex review 2026-09-08 (05:04): "query the actual pipeline's threadExecutionWidth, require 32, and verify the
// actual dispatched threadgroup is exactly 32x1x1 - not merely a manifest assertion; reject a mismatch before dispatch".
// Codex review 2026-09-08 (05:11): the guard must be CALLED by the actual candidate body immediately before the
// launch-155 dispatch and must fail closed; a standalone binary that is never linked proves nothing.
//
// The predicate below is pure (no Metal types) so it can be unit-tested on the CPU for every rejection case.
// nr_single_warp_guard_ok() in nr_single_warp_guard.mm adapts a real MTLComputePipelineState to it.
#pragma once
#include <string>
struct NrSingleWarpFacts {           // everything the guard is allowed to look at, all read from the ACTUAL objects
  unsigned long thread_execution_width = 0;      // pso.threadExecutionWidth
  unsigned long max_total_threads = 0;           // pso.maxTotalThreadsPerThreadgroup
  unsigned long tg_w = 0, tg_h = 0, tg_d = 0;    // the threadgroup this dispatch will actually use
};
// Fails closed: returns false for every shape that is not exactly 32x1x1 on a 32-wide pipeline, including 0, 31, 33 and 64.
inline bool nr_single_warp_facts_ok(const NrSingleWarpFacts& f, std::string& why) {
  if (f.thread_execution_width != 32) { why = "pipeline threadExecutionWidth " + std::to_string(f.thread_execution_width) + " != 32"; return false; }
  if (f.tg_w != 32 || f.tg_h != 1 || f.tg_d != 1) {
    why = "threadgroup is " + std::to_string(f.tg_w) + "x" + std::to_string(f.tg_h) + "x" + std::to_string(f.tg_d) + ", not 32x1x1"; return false; }
  if (f.max_total_threads < 32) { why = "maxTotalThreadsPerThreadgroup " + std::to_string(f.max_total_threads) + " < 32"; return false; }
  why = "ok"; return true;
}
