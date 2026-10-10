/*
 * NeuralStudio runtime - temporal reset policy
 * SPDX-License-Identifier: MIT
 *
 * Decides when a temporal model's history must be discarded. Getting this wrong
 * is the difference between a working video pipeline and one that smears across
 * every cut, and the failure is subtle enough to ship unnoticed - so the policy
 * is isolated here and tested directly rather than being scattered through the
 * frame loop.
 *
 * The package manifest declares which triggers apply and whether a reset is
 * soft (a flag on the next evaluate) or requires session recreation. The policy
 * consumes those declarations; it does not invent them.
 */

#ifndef NEURALSTUDIO_RUNTIME_RESET_POLICY_H
#define NEURALSTUDIO_RUNTIME_RESET_POLICY_H

#include <cstdint>
#include <string>

namespace neuralstudio {

enum class ResetReason {
    None,
    SessionStart,
    SourceChange,
    Seek,
    Loop,
    ResolutionChange,
    FormatChange,
    ColorSpaceChange,
    SceneCut,
    ResumeFromBypass,
    AdapterRequested
};

const char* ToString(ResetReason reason);

enum class ResetSemantics {
    Soft,      /* a reset flag on the next evaluate is sufficient */
    Recreate   /* the session must be destroyed and rebuilt */
};

/* Trigger set, as a bitmask so a manifest's declared triggers map directly. */
enum ResetTrigger : std::uint32_t {
    kTriggerSessionStart     = 1u << 0,
    kTriggerSourceChange     = 1u << 1,
    kTriggerSeek             = 1u << 2,
    kTriggerLoop             = 1u << 3,
    kTriggerResolutionChange = 1u << 4,
    kTriggerFormatChange     = 1u << 5,
    kTriggerColorSpaceChange = 1u << 6,
    kTriggerSceneCut         = 1u << 7,
    kTriggerResumeFromBypass = 1u << 8,
    kTriggerAdapterRequested = 1u << 9,
    kTriggerAll              = 0x3FFu
};

/* What the frame loop observed about this frame's source. */
struct FrameContext {
    std::uint64_t source_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t color_format = 0;
    std::uint32_t color_space = 0;
    double presentation_seconds = 0.0;
    bool valid = false;
};

/* Signals the frame loop and guide layer raise explicitly. */
struct ResetSignals {
    bool session_start = false;
    bool seek = false;
    bool loop = false;
    bool scene_cut = false;
    bool resume_from_bypass = false;
    bool adapter_requested = false;
};

struct ResetDecision {
    ResetReason reason = ResetReason::None;
    bool requires_recreate = false;
    std::string detail;

    bool needed() const { return reason != ResetReason::None; }
};

struct ResetPolicyConfig {
    bool stateful = true;
    ResetSemantics semantics = ResetSemantics::Soft;
    std::uint32_t triggers = kTriggerAll;

    /* Backward presentation time, or a jump larger than this, is treated as a
       seek even when the source did not announce one. Container timestamps are
       not always trustworthy, so the policy detects rather than assumes. */
    double seek_jump_seconds = 1.0;
};

class ResetPolicy {
public:
    ResetPolicy() = default;
    explicit ResetPolicy(ResetPolicyConfig config) : config_(config) {}

    const ResetPolicyConfig& config() const { return config_; }

    /*
     * Evaluate one frame. Only the highest-priority reason is reported, because
     * a reset clears all history regardless of how many conditions triggered it
     * and reporting a list would imply otherwise.
     */
    ResetDecision Evaluate(const FrameContext& current, const ResetSignals& signals);

    /* Discards remembered context so the next frame is treated as a start. */
    void Invalidate();

    std::uint64_t reset_count() const { return reset_count_; }

private:
    bool Enabled(ResetTrigger trigger) const { return (config_.triggers & trigger) != 0u; }
    ResetDecision Make(ResetReason reason, std::string detail);

    ResetPolicyConfig config_;
    FrameContext previous_;
    std::uint64_t reset_count_ = 0;
};

}  // namespace neuralstudio

#endif  /* NEURALSTUDIO_RUNTIME_RESET_POLICY_H */
