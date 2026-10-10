#pragma once
// Shared lifecycle helper, Codex-owned. Runtime adoption is Opus-owned.
// This manages encoders, not command-buffer submission or resource/ticket lifetime.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstddef>

namespace netal_nr_experimental {
struct FrameEncoderState {
  // Per encode INVOCATION. Never global, and never shared between concurrent calls.
  std::size_t started = 0;
  bool cleanup_failed = false;
  FrameEncoderState() = default;
  FrameEncoderState(const FrameEncoderState&) = delete;
  FrameEncoderState& operator=(const FrameEncoderState&) = delete;
};

class EncoderScope final {
  FrameEncoderState& state_;
  __strong id<MTLCommandEncoder> encoder_;
public:
  // Construct IMMEDIATELY after obtaining an encoder, before commands, pipeline
  // lookups, validation or other throwing operations. Conservative counting starts
  // now, not after endEncoding: a later failure must retain a discard-only ticket.
  EncoderScope(FrameEncoderState& state, id<MTLCommandEncoder> encoder) noexcept
      : state_(state), encoder_(encoder) { if (encoder_) ++state_.started; }
  EncoderScope(const EncoderScope&) = delete;
  EncoderScope& operator=(const EncoderScope&) = delete;
  // A factory may return ownership after safely performing its bookkeeping.
  // Moving transfers the one close obligation; it does not count a new encoder.
  EncoderScope(EncoderScope&& other) noexcept : state_(other.state_), encoder_(other.encoder_) { other.encoder_ = nil; }
  EncoderScope& operator=(EncoderScope&&) = delete;
  ~EncoderScope() noexcept { close(); }

  bool valid() const noexcept { return encoder_ != nil; }
  id<MTLCommandEncoder> get() const noexcept { return encoder_; }
  // Replace raw [encoder endEncoding] with scope.close(), or use a lexical scope.
  // Never end the same encoder independently; close() is idempotent only for calls
  // through THIS scope. It does not inspect another component's encoder state.
  bool close() noexcept {
    if (!encoder_) return !state_.cleanup_failed;
    id<MTLCommandEncoder> encoder = encoder_;
    encoder_ = nil;  // exactly one attempted end, including a throwing Metal call
    try {
      @try { [encoder endEncoding]; }
      @catch (id exception) { (void)exception; state_.cleanup_failed = true; }
    } catch (...) { state_.cleanup_failed = true; }
    return !state_.cleanup_failed;
  }
};
} // namespace netal_nr_experimental

// Adoption obligations:
// - Use ARC/ObjC++ exception unwinding and run the companion audit with the actual
//   production compiler flags. Every encoder creation needs one immediate scope.
//   Creator factories must acquire ownership BEFORE any throwing profiler/cache
//   bookkeeping, not only after returning a raw encoder to their caller.
// - Keep FrameEncoderState alive outside any stage-local catch/return so callers
//   can conservatively classify partial work, including errors returned normally.
// - Catch/report both C++ and Objective-C stage exceptions at the runtime boundary;
//   a scope closes encoders but does not itself publish/retain a ticket.
// - Retain all referenced GPU resources through genuine command completion. Never
//   commit, wait, abandon the caller buffer, or free pending scratch in this helper.
// - A cleanup_failed flag is NOT a claim that the buffer is healthy; report that
//   underlying Metal/API failure and disable the affected context. Device/driver
//   failures cannot be turned into successful rendering by a scope guard.
