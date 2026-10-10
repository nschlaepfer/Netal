#pragma once
// CPU-only admission protocol for one delayed, frozen NR diagnostic.
// This does not execute the model or establish that pixels reached the screen.
#include <cstdint>
#include <mutex>

namespace netal_nr_adapter {
class DelayedOnceState {
public:
  enum class Phase { Idle, Capturing, Computing, Ready, PreparingPublication, Publishing, Recovering, Done, Failed, WaitingForCompute };
  struct Snapshot {
    Phase phase = Phase::Idle;
    uint64_t source_frame = 0, publication_frame = 0, stream = 0;
    uint32_t width = 0, height = 0;
    bool model_computed = false, export_completed = false;
    bool publication_may_be_encoded = false, publication_error = false;
    bool drawable_presented = false, original_rendering_recovered = false;
  };
private:
  mutable std::mutex mutex_;
  Snapshot state_;
public:
  Snapshot snapshot() const { std::lock_guard<std::mutex> lock(mutex_); return state_; }
  bool begin_capture(uint64_t frame, uint64_t stream, uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Idle || !stream || width != 1920 || height != 1200) return false;
    state_.source_frame = frame; state_.stream = stream;
    state_.width = width; state_.height = height; state_.phase = Phase::Capturing;
    return true;
  }
  bool capture_completed(bool success, bool wait_for_compute = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Capturing) return false;
    state_.phase = success ? (wait_for_compute ? Phase::WaitingForCompute : Phase::Computing) : Phase::Failed;
    return success;
  }
  // Used only after the owned input conversion has completed. A manual release
  // cannot revive cancellation, re-arm a job, or skip the initial capture.
  bool release_compute(bool released) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!released || state_.phase != Phase::WaitingForCompute) return false;
    state_.phase = Phase::Computing;
    return true;
  }
  bool compute_completed(bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Computing) return false;
    state_.model_computed = state_.model_computed || success;
    state_.phase = success ? Phase::Ready : Phase::Failed;
    return success;
  }
  // Record actual NN completion even if later color/export validation rejects
  // the result. Computation, export eligibility and presentation are distinct.
  bool note_model_computed() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Computing) return false;
    state_.model_computed = true;
    return true;
  }
  // Other scaler streams are ignored, never mistaken for the selected stream.
  // A geometry change in the selected stream permanently cancels the diagnostic.
  bool observe_geometry(uint64_t stream, uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase == Phase::Idle || stream != state_.stream) return true;
    if (width == state_.width && height == state_.height) return true;
    if (!state_.publication_may_be_encoded && state_.phase != Phase::Done) state_.phase = Phase::Failed;
    return false;
  }
  bool begin_publication(uint64_t frame, uint64_t stream, uint32_t width, uint32_t height, bool released = true) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Ready || stream != state_.stream || frame <= state_.source_frame) return false;
    if (width != state_.width || height != state_.height) { state_.phase = Phase::Failed; return false; }
    if (!released) return false;
    state_.publication_frame = frame; state_.phase = Phase::PreparingPublication;
    return true;
  }
  // Irreversible command-encoding boundary. Cancellation before this prevents
  // the copy; cancellation afterward cannot retract GPU commands already added.
  bool commit_publication() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::PreparingPublication) return false;
    state_.publication_may_be_encoded = true; state_.phase = Phase::Publishing;
    return true;
  }
  bool publication_completed(bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Publishing) return false;
    success = success && !state_.publication_error;
    state_.export_completed = success;
    state_.phase = success ? Phase::Recovering : Phase::Failed;
    return success;
  }
  // Presentation may precede or follow command completion. This flag records an
  // association only: external screenshot/video evidence is still required.
  bool note_presented(uint64_t frame, uint64_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_.publication_may_be_encoded ||
        (state_.phase != Phase::Publishing && state_.phase != Phase::Recovering && state_.phase != Phase::Done && state_.phase != Phase::Failed) ||
        frame != state_.publication_frame || stream != state_.stream) return false;
    state_.drawable_presented = true;
    return true;
  }
  bool original_completed(uint64_t frame, uint64_t stream, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((state_.phase != Phase::Recovering && !(state_.phase == Phase::Failed && state_.publication_may_be_encoded)) ||
        state_.original_rendering_recovered || stream != state_.stream || frame <= state_.publication_frame) return false;
    state_.original_rendering_recovered = success;
    if (state_.phase != Phase::Failed) state_.phase = success ? Phase::Done : Phase::Failed;
    return success;
  }
  void fail() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase == Phase::Publishing) state_.publication_error = true;
    else if (state_.phase != Phase::Done) state_.phase = Phase::Failed;
  }
};
} // namespace netal_nr_adapter
