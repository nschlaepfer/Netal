/*
 * NeuralStudio runtime - pipeline state machine
 * SPDX-License-Identifier: MIT
 *
 * Owns the five product states and, more importantly, the rule that makes them
 * trustworthy: Active is a claim about observed evaluations and fresh output,
 * never about a library having loaded or a call having returned success.
 *
 * This file is deliberately free of GPU, adapter, and IPC dependencies so the
 * transition rules can be unit tested exhaustively without hardware.
 */

#ifndef NEURALSTUDIO_RUNTIME_PIPELINE_STATE_H
#define NEURALSTUDIO_RUNTIME_PIPELINE_STATE_H

#include <cstdint>
#include <string>

namespace neuralstudio {

enum class PipelineState {
    Standby,    /* session created, not yet evaluating */
    Active,     /* successful evaluations observed and output is fresh */
    Degraded,   /* evaluating, but a stage was bypassed or a target is missed */
    Bypassed,   /* source passes through unmodified */
    Failed      /* cannot run; reason recorded */
};

const char* ToString(PipelineState state);

/*
 * Everything the state machine is allowed to reason about. If a fact is not
 * here, it cannot influence the reported state - which is what prevents
 * optimistic states from creeping back in.
 */
struct Evidence {
    std::uint64_t successful_evaluations = 0;
    std::uint64_t consecutive_failures = 0;
    std::uint64_t dropped_frames = 0;
    bool output_fresh = false;
    bool used_fallback_path = false;
    double last_evaluation_seconds = 0.0;
    double last_inference_ms = 0.0;
};

/* One observation from the frame loop. */
struct Observation {
    double now_seconds = 0.0;
    bool evaluated = false;
    bool output_fresh = false;
    bool used_fallback_path = false;
    double inference_ms = 0.0;
    bool stage_bypassed = false;
    bool device_lost = false;
    bool adapter_host_lost = false;
    bool frame_dropped = false;
};

struct Thresholds {
    /* Consecutive failures tolerated before the pipeline stops trying. */
    std::uint64_t failure_tolerance = 3;

    /* How long output may go stale while running before the state degrades. */
    double stale_output_seconds = 1.0;

    /* Frame-time budget. Exceeding it degrades rather than fails, because the
       user may deliberately accept a slow, high-quality pipeline. */
    double target_frame_ms = 0.0;  /* 0 disables the check */

    /* Whether the source can still be presented when the model stops. When
       false, a failure is terminal instead of a bypass. */
    bool source_can_pass_through = true;
};

struct Transition {
    bool changed = false;
    PipelineState from = PipelineState::Standby;
    PipelineState to = PipelineState::Standby;
    std::string reason;
};

/*
 * Activation gate.
 *
 * Temporal models must not start mid-flight. VALIDATION.md records what happens
 * when activation races the host application's own frames: the backend takes a
 * fallback path and the submission becomes unsafe. The gate requires a run of
 * consecutive frames at a stable resolution before the model is allowed to
 * start.
 */
class ActivationGate {
public:
    ActivationGate() = default;
    explicit ActivationGate(std::uint32_t required_stable_frames)
        : required_stable_frames_(required_stable_frames) {}

    void ObserveResolution(std::uint32_t width, std::uint32_t height);
    void Invalidate();

    bool IsReady() const { return stable_frames_ >= required_stable_frames_; }
    std::uint32_t stable_frames() const { return stable_frames_; }

private:
    std::uint32_t required_stable_frames_ = 4;
    std::uint32_t stable_frames_ = 0;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
};

class PipelineStateMachine {
public:
    PipelineStateMachine() = default;
    explicit PipelineStateMachine(Thresholds thresholds) : thresholds_(thresholds) {}

    PipelineState state() const { return state_; }
    const Evidence& evidence() const { return evidence_; }

    /* Feed one frame's outcome. Returns the transition, if any. */
    Transition Observe(const Observation& observation);

    /* Explicit user or policy actions. */
    Transition RequestBypass(bool bypass, double now_seconds);
    Transition Fail(const std::string& reason);

    /* Clears evaluation evidence without changing thresholds. Called on reset
       so a stale Active claim cannot survive a history reset. */
    void ClearEvidence();

private:
    Transition MoveTo(PipelineState next, std::string reason);

    Thresholds thresholds_;
    PipelineState state_ = PipelineState::Standby;
    Evidence evidence_;
    bool bypass_requested_ = false;
};

}  // namespace neuralstudio

#endif  /* NEURALSTUDIO_RUNTIME_PIPELINE_STATE_H */
