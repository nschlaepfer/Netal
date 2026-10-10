/*
 * NeuralStudio runtime - broker core tests
 * SPDX-License-Identifier: MIT
 *
 * No test framework dependency on purpose: this must build and run with only
 * the MSVC toolchain the repository already requires.
 *
 * These cases encode the rules the product's honesty depends on. The most
 * important ones are the negative cases - the pipeline must refuse to report
 * Active when it has not actually produced fresh output.
 */

#include "pipeline_state.h"
#include "reset_policy.h"
#include "test_support.h"

#include <cstdio>
#include <string>

void TestCompatibility();
void TestManifest();

namespace {

using nstest::Check;
using nstest::Group;
using namespace neuralstudio;

Observation GoodFrame(double now) {
    Observation observation;
    observation.now_seconds = now;
    observation.evaluated = true;
    observation.output_fresh = true;
    observation.inference_ms = 4.0;
    return observation;
}

FrameContext Frame(std::uint32_t width, std::uint32_t height, double pts,
                   std::uint64_t source_id = 1, std::uint32_t format = 1,
                   std::uint32_t color_space = 1) {
    FrameContext context;
    context.valid = true;
    context.source_id = source_id;
    context.width = width;
    context.height = height;
    context.color_format = format;
    context.color_space = color_space;
    context.presentation_seconds = pts;
    return context;
}

void TestStateMachine() {
    Group("pipeline state machine");

    {
        PipelineStateMachine machine;
        Check(machine.state() == PipelineState::Standby, "starts in Standby");

        machine.Observe(GoodFrame(1.0));
        Check(machine.state() == PipelineState::Active,
              "reaches Active on a successful evaluation with fresh output");
        Check(machine.evidence().successful_evaluations == 1,
              "records the evaluation as evidence");
    }

    {
        /* The single most important negative case. */
        PipelineStateMachine machine;
        Observation observation = GoodFrame(1.0);
        observation.output_fresh = false;
        machine.Observe(observation);
        Check(machine.state() != PipelineState::Active,
              "never reports Active when output is not fresh",
              ToString(machine.state()));
    }

    {
        PipelineStateMachine machine;
        Observation observation = GoodFrame(1.0);
        observation.used_fallback_path = true;
        machine.Observe(observation);
        Check(machine.state() == PipelineState::Degraded,
              "a reported fallback path degrades instead of activating");
    }

    {
        PipelineStateMachine machine;
        Observation observation = GoodFrame(1.0);
        observation.stage_bypassed = true;
        machine.Observe(observation);
        Check(machine.state() == PipelineState::Degraded,
              "a bypassed stage degrades");
    }

    {
        Thresholds thresholds;
        thresholds.target_frame_ms = 8.0;
        PipelineStateMachine machine(thresholds);
        Observation observation = GoodFrame(1.0);
        observation.inference_ms = 20.0;
        machine.Observe(observation);
        Check(machine.state() == PipelineState::Degraded,
              "exceeding the frame budget degrades but does not fail");
    }

    {
        Thresholds thresholds;
        thresholds.failure_tolerance = 3;
        thresholds.source_can_pass_through = true;
        PipelineStateMachine machine(thresholds);

        Observation bad;
        bad.now_seconds = 1.0;
        bad.evaluated = false;

        machine.Observe(bad);
        machine.Observe(bad);
        Check(machine.state() != PipelineState::Bypassed,
              "tolerates failures below the threshold");
        machine.Observe(bad);
        Check(machine.state() == PipelineState::Bypassed,
              "bypasses to the source after repeated failures");
    }

    {
        Thresholds thresholds;
        thresholds.failure_tolerance = 2;
        thresholds.source_can_pass_through = false;
        PipelineStateMachine machine(thresholds);

        Observation bad;
        bad.evaluated = false;
        machine.Observe(bad);
        machine.Observe(bad);
        Check(machine.state() == PipelineState::Failed,
              "fails when repeated failures cannot be bypassed");
    }

    {
        PipelineStateMachine machine;
        Observation observation;
        observation.device_lost = true;
        machine.Observe(observation);
        Check(machine.state() == PipelineState::Failed,
              "device loss is terminal");
    }

    {
        PipelineStateMachine machine;
        Observation observation;
        observation.adapter_host_lost = true;
        machine.Observe(observation);
        Check(machine.state() == PipelineState::Bypassed,
              "a crashed adapter host bypasses instead of killing the pipeline");
    }

    {
        PipelineStateMachine machine;
        machine.Observe(GoodFrame(1.0));
        machine.RequestBypass(true, 2.0);
        Check(machine.state() == PipelineState::Bypassed, "explicit bypass is honored");

        machine.RequestBypass(false, 3.0);
        Check(machine.state() == PipelineState::Standby,
              "releasing a bypass returns to Standby, not straight to Active");
        Check(machine.evidence().successful_evaluations == 0,
              "releasing a bypass clears stale evaluation evidence");
    }
}

void TestActivationGate() {
    Group("activation gate");

    ActivationGate gate(4);
    Check(!gate.IsReady(), "not ready before any frames");

    gate.ObserveResolution(3840, 2160);
    gate.ObserveResolution(3840, 2160);
    gate.ObserveResolution(3840, 2160);
    Check(!gate.IsReady(), "not ready before the required stable-frame run");

    gate.ObserveResolution(3840, 2160);
    Check(gate.IsReady(), "ready after a stable run at one resolution");

    gate.ObserveResolution(1920, 1080);
    Check(!gate.IsReady(), "a resolution change restarts the stable run");

    gate.ObserveResolution(0, 0);
    Check(!gate.IsReady(), "an invalid resolution invalidates the gate");
}

void TestResetPolicy() {
    Group("reset policy");

    {
        ResetPolicyConfig config;
        config.stateful = false;
        ResetPolicy policy(config);
        Check(!policy.Evaluate(Frame(1920, 1080, 0.0), {}).needed(),
              "a stateless package never resets");
    }

    {
        ResetPolicy policy;
        const ResetDecision first = policy.Evaluate(Frame(1920, 1080, 0.0), {});
        Check(first.reason == ResetReason::SessionStart,
              "the first frame of a session is a reset");
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 0.0), {});
        const ResetDecision decision = policy.Evaluate(Frame(3840, 2160, 0.033), {});
        Check(decision.reason == ResetReason::ResolutionChange,
              "a resolution change resets");
        Check(decision.requires_recreate,
              "a resolution change requires session recreation even under soft semantics");
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 0.0, 1, 1), {});
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 0.033, 1, 2), {});
        Check(decision.reason == ResetReason::FormatChange, "a format change resets");
        Check(decision.requires_recreate, "a format change requires recreation");
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 0.0, 1), {});
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 0.033, 2), {});
        Check(decision.reason == ResetReason::SourceChange, "a source change resets");
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 10.0), {});
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 5.0), {});
        Check(decision.reason == ResetReason::Seek,
              "presentation time moving backward is detected as a seek");
    }

    {
        ResetPolicyConfig config;
        config.seek_jump_seconds = 1.0;
        ResetPolicy policy(config);
        policy.Evaluate(Frame(1920, 1080, 10.0), {});
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 30.0), {});
        Check(decision.reason == ResetReason::Seek,
              "a large forward jump is detected as a seek");
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 10.0), {});
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 10.033), {});
        Check(!decision.needed(),
              "an ordinary frame step does not reset",
              ToString(decision.reason));
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 0.0), {});
        ResetSignals signals;
        signals.scene_cut = true;
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 0.033), signals);
        Check(decision.reason == ResetReason::SceneCut, "a scene cut resets");
        Check(!decision.requires_recreate,
              "a scene cut is a soft reset under soft semantics");
    }

    {
        ResetPolicyConfig config;
        config.semantics = ResetSemantics::Recreate;
        ResetPolicy policy(config);
        policy.Evaluate(Frame(1920, 1080, 0.0), {});
        ResetSignals signals;
        signals.scene_cut = true;
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 0.033), signals);
        Check(decision.requires_recreate,
              "recreate semantics force recreation for every reset");
    }

    {
        /* A package that does not declare the scene-cut trigger must not be
           reset on scene cuts, even though the guide layer detected one. */
        ResetPolicyConfig config;
        config.triggers = kTriggerAll & ~static_cast<std::uint32_t>(kTriggerSceneCut);
        ResetPolicy policy(config);
        policy.Evaluate(Frame(1920, 1080, 0.0), {});
        ResetSignals signals;
        signals.scene_cut = true;
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 0.033), signals);
        Check(!decision.needed(),
              "a trigger the manifest does not declare is not applied");
    }

    {
        ResetPolicy policy;
        policy.Evaluate(Frame(1920, 1080, 0.0), {});
        ResetSignals signals;
        signals.resume_from_bypass = true;
        const ResetDecision decision = policy.Evaluate(Frame(1920, 1080, 0.033), signals);
        Check(decision.reason == ResetReason::ResumeFromBypass,
              "resuming from a bypass resets history");
    }
}

void TestTenThousandFrameSoak() {
    Group("10,000-frame broker soak");
    PipelineStateMachine state;
    ResetPolicy reset;
    for (std::uint64_t frame = 0; frame < 10000; ++frame) {
        const double now = static_cast<double>(frame) / 60.0;
        const ResetDecision decision =
            reset.Evaluate(Frame(1920, 1080, now), {});
        Observation observation;
        observation.now_seconds = now;
        observation.evaluated = true;
        observation.output_fresh = true;
        observation.inference_ms = 4.0;
        (void)state.Observe(observation);
        if ((frame == 0) != decision.needed()) {
            Check(false, "10,000 ordinary frames produce only the session-start reset");
            return;
        }
    }
    Check(state.state() == PipelineState::Active &&
              state.evidence().successful_evaluations == 10000,
          "10,000 fresh evaluations preserve exact Active evidence");
    Check(reset.reset_count() == 1,
          "10,000 ordinary frames produce only the session-start reset");
}

}  // namespace

int main() {
    std::printf("NeuralStudio broker core tests\n");

    TestStateMachine();
    TestActivationGate();
    TestResetPolicy();
    TestTenThousandFrameSoak();
    TestCompatibility();
    TestManifest();

    std::printf("\n%d checks, %d failure(s)\n", nstest::TotalChecks(), nstest::TotalFailures());
    return nstest::TotalFailures() == 0 ? 0 : 1;
}
