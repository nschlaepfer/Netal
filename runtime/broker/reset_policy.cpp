/*
 * NeuralStudio runtime - temporal reset policy
 * SPDX-License-Identifier: MIT
 */

#include "reset_policy.h"

namespace neuralstudio {

const char* ToString(ResetReason reason) {
    switch (reason) {
    case ResetReason::None:             return "None";
    case ResetReason::SessionStart:     return "SessionStart";
    case ResetReason::SourceChange:     return "SourceChange";
    case ResetReason::Seek:             return "Seek";
    case ResetReason::Loop:             return "Loop";
    case ResetReason::ResolutionChange: return "ResolutionChange";
    case ResetReason::FormatChange:     return "FormatChange";
    case ResetReason::ColorSpaceChange: return "ColorSpaceChange";
    case ResetReason::SceneCut:         return "SceneCut";
    case ResetReason::ResumeFromBypass: return "ResumeFromBypass";
    case ResetReason::AdapterRequested: return "AdapterRequested";
    }
    return "Unknown";
}

void ResetPolicy::Invalidate() {
    previous_ = FrameContext{};
}

ResetDecision ResetPolicy::Make(ResetReason reason, std::string detail) {
    ResetDecision decision;
    decision.reason = reason;
    decision.detail = std::move(detail);

    if (reason == ResetReason::None) {
        return decision;
    }

    ++reset_count_;

    /* A resolution or format change always requires recreation regardless of
       the manifest's declared semantics: the model's allocated resources no
       longer match the frame. Soft semantics only cover history. */
    const bool structural = (reason == ResetReason::ResolutionChange ||
                             reason == ResetReason::FormatChange ||
                             reason == ResetReason::SourceChange);

    decision.requires_recreate =
        structural || config_.semantics == ResetSemantics::Recreate;
    return decision;
}

ResetDecision ResetPolicy::Evaluate(const FrameContext& current, const ResetSignals& signals) {
    /* A stateless model has no history to discard. Reporting a reset anyway
       would make reset counts meaningless when comparing packages. */
    if (!config_.stateful) {
        previous_ = current;
        return ResetDecision{};
    }

    const bool had_previous = previous_.valid;

    /* Structural changes are checked before explicit signals, because a seek
       that also changes resolution must recreate, not soft-reset. */

    if (!had_previous) {
        previous_ = current;
        if (Enabled(kTriggerSessionStart)) {
            return Make(ResetReason::SessionStart, "First frame of the session");
        }
        return ResetDecision{};
    }

    if (current.source_id != previous_.source_id) {
        previous_ = current;
        if (Enabled(kTriggerSourceChange)) {
            return Make(ResetReason::SourceChange, "Source changed");
        }
        return ResetDecision{};
    }

    if (current.width != previous_.width || current.height != previous_.height) {
        const std::string detail =
            "Resolution changed from " + std::to_string(previous_.width) + "x" +
            std::to_string(previous_.height) + " to " + std::to_string(current.width) +
            "x" + std::to_string(current.height);
        previous_ = current;
        if (Enabled(kTriggerResolutionChange)) {
            return Make(ResetReason::ResolutionChange, detail);
        }
        return ResetDecision{};
    }

    if (current.color_format != previous_.color_format) {
        previous_ = current;
        if (Enabled(kTriggerFormatChange)) {
            return Make(ResetReason::FormatChange, "Color format changed");
        }
        return ResetDecision{};
    }

    if (current.color_space != previous_.color_space) {
        previous_ = current;
        if (Enabled(kTriggerColorSpaceChange)) {
            return Make(ResetReason::ColorSpaceChange, "Color space changed");
        }
        return ResetDecision{};
    }

    /* Explicit signals. */

    if (signals.adapter_requested && Enabled(kTriggerAdapterRequested)) {
        previous_ = current;
        return Make(ResetReason::AdapterRequested, "Adapter requested a reset");
    }

    if (signals.resume_from_bypass && Enabled(kTriggerResumeFromBypass)) {
        previous_ = current;
        return Make(ResetReason::ResumeFromBypass, "Resumed after a bypass");
    }

    if (signals.loop && Enabled(kTriggerLoop)) {
        previous_ = current;
        return Make(ResetReason::Loop, "Source looped");
    }

    if (signals.seek && Enabled(kTriggerSeek)) {
        previous_ = current;
        return Make(ResetReason::Seek, "Seek reported by the source");
    }

    /* Detected seek: presentation time moved backward, or jumped forward by
       more than the configured tolerance. */
    if (Enabled(kTriggerSeek) && current.presentation_seconds > 0.0 &&
        previous_.presentation_seconds > 0.0) {
        const double delta = current.presentation_seconds - previous_.presentation_seconds;
        if (delta < 0.0) {
            previous_ = current;
            return Make(ResetReason::Seek, "Presentation time moved backward");
        }
        if (config_.seek_jump_seconds > 0.0 && delta > config_.seek_jump_seconds) {
            previous_ = current;
            return Make(ResetReason::Seek, "Presentation time jumped forward by " +
                                               std::to_string(delta) + "s");
        }
    }

    if (signals.session_start && Enabled(kTriggerSessionStart)) {
        previous_ = current;
        return Make(ResetReason::SessionStart, "Session restarted");
    }

    if (signals.scene_cut && Enabled(kTriggerSceneCut)) {
        previous_ = current;
        return Make(ResetReason::SceneCut, "Scene discontinuity detected");
    }

    previous_ = current;
    return ResetDecision{};
}

}  // namespace neuralstudio
