/*
 * NeuralStudio runtime - pipeline state machine
 * SPDX-License-Identifier: MIT
 */

#include "pipeline_state.h"

namespace neuralstudio {

const char* ToString(PipelineState state) {
    switch (state) {
    case PipelineState::Standby:  return "Standby";
    case PipelineState::Active:   return "Active";
    case PipelineState::Degraded: return "Degraded";
    case PipelineState::Bypassed: return "Bypassed";
    case PipelineState::Failed:   return "Failed";
    }
    return "Unknown";
}

void ActivationGate::ObserveResolution(std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0) {
        Invalidate();
        return;
    }
    if (width != width_ || height != height_) {
        width_ = width;
        height_ = height;
        stable_frames_ = 1;
        return;
    }
    if (stable_frames_ < required_stable_frames_) {
        ++stable_frames_;
    }
}

void ActivationGate::Invalidate() {
    stable_frames_ = 0;
    width_ = 0;
    height_ = 0;
}

Transition PipelineStateMachine::MoveTo(PipelineState next, std::string reason) {
    Transition transition;
    transition.from = state_;
    transition.to = next;
    transition.reason = std::move(reason);
    transition.changed = (next != state_);
    state_ = next;
    return transition;
}

void PipelineStateMachine::ClearEvidence() {
    evidence_.successful_evaluations = 0;
    evidence_.consecutive_failures = 0;
    evidence_.output_fresh = false;
    evidence_.used_fallback_path = false;
    evidence_.last_evaluation_seconds = 0.0;
    evidence_.last_inference_ms = 0.0;
}

Transition PipelineStateMachine::RequestBypass(bool bypass, double now_seconds) {
    (void)now_seconds;
    bypass_requested_ = bypass;
    if (bypass) {
        return MoveTo(PipelineState::Bypassed, "Bypass requested");
    }
    if (state_ == PipelineState::Bypassed) {
        /* Resuming never returns straight to Active. The pipeline must earn it
           again with fresh evidence, because history was discarded. */
        ClearEvidence();
        return MoveTo(PipelineState::Standby, "Bypass released; awaiting evaluation evidence");
    }
    return MoveTo(state_, "No change");
}

Transition PipelineStateMachine::Fail(const std::string& reason) {
    return MoveTo(PipelineState::Failed, reason);
}

Transition PipelineStateMachine::Observe(const Observation& observation) {
    /* Terminal conditions first. A lost device cannot be bypassed around. */
    if (observation.device_lost) {
        evidence_.output_fresh = false;
        return MoveTo(PipelineState::Failed, "Device lost");
    }

    if (observation.adapter_host_lost) {
        evidence_.output_fresh = false;
        evidence_.consecutive_failures++;
        if (thresholds_.source_can_pass_through) {
            return MoveTo(PipelineState::Bypassed,
                          "Adapter host crashed; source passed through unmodified");
        }
        return MoveTo(PipelineState::Failed, "Adapter host crashed and the source cannot pass through");
    }

    if (bypass_requested_) {
        return MoveTo(PipelineState::Bypassed, "Bypass requested");
    }

    if (observation.frame_dropped) {
        evidence_.dropped_frames++;
    }

    if (observation.evaluated && observation.output_fresh) {
        evidence_.successful_evaluations++;
        evidence_.consecutive_failures = 0;
        evidence_.output_fresh = true;
        evidence_.used_fallback_path = observation.used_fallback_path;
        evidence_.last_evaluation_seconds = observation.now_seconds;
        evidence_.last_inference_ms = observation.inference_ms;

        /* A fallback path means the backend produced output through a route the
           host did not intend. That is exactly the condition that made the
           earlier research crash, so it is never reported as fully Active. */
        if (observation.used_fallback_path) {
            return MoveTo(PipelineState::Degraded,
                          "Adapter reported a fallback path");
        }
        if (observation.stage_bypassed) {
            return MoveTo(PipelineState::Degraded, "A pipeline stage was bypassed");
        }
        if (thresholds_.target_frame_ms > 0.0 &&
            observation.inference_ms > thresholds_.target_frame_ms) {
            return MoveTo(PipelineState::Degraded, "Frame-time budget exceeded");
        }
        return MoveTo(PipelineState::Active, "Successful evaluation with fresh output");
    }

    /* Not evaluated, or evaluated without producing fresh output. Both are
       failures from the host's point of view; an adapter that returns success
       without fresh output is still not delivering frames. */
    evidence_.consecutive_failures++;
    evidence_.output_fresh = false;

    if (evidence_.consecutive_failures >= thresholds_.failure_tolerance) {
        if (thresholds_.source_can_pass_through) {
            return MoveTo(PipelineState::Bypassed,
                          "Evaluation failed repeatedly; source passed through unmodified");
        }
        return MoveTo(PipelineState::Failed, "Evaluation failed repeatedly");
    }

    /* Under tolerance. If output has gone stale while the pipeline claims to be
       running, degrade rather than continuing to advertise Active. */
    if (state_ == PipelineState::Active || state_ == PipelineState::Degraded) {
        const double age = observation.now_seconds - evidence_.last_evaluation_seconds;
        if (thresholds_.stale_output_seconds > 0.0 && age > thresholds_.stale_output_seconds) {
            return MoveTo(PipelineState::Degraded, "Output is stale");
        }
        return MoveTo(PipelineState::Degraded, "Evaluation did not produce fresh output");
    }

    return MoveTo(state_, "Awaiting evaluation evidence");
}

}  // namespace neuralstudio
