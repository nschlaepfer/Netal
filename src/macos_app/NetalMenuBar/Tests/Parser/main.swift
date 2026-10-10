// Headless checks for StatusParser, run by scripts/test-macos-netal-menubar.sh.
// Deliberately not XCTest: this builds with plain swiftc alongside the app sources.
import Foundation

var failures = 0

func check(_ condition: Bool, _ what: String, file: StaticString = #file, line: UInt = #line) {
    if condition {
        print("  ok    \(what)")
    } else {
        print("  FAIL  \(what)   (\(file):\(line))")
        failures += 1
    }
}

func equal<T: Equatable>(_ got: T, _ want: T, _ what: String, line: UInt = #line) {
    check(got == want, "\(what): got \(got), want \(want)", line: line)
}

// MARK: - key=value extraction

print("key=value extraction")

// A prefix of another key must not match: `ms=` inside `mean_ms=`, `frame=` inside `worst_frame=`.
let pacing = "event=sr_pacing frame=120 window=118 mean_ms=22.50 fps=44.4 min_ms=1.95 "
           + "max_ms=387.14 spikes_over_25ms=6 worst_ms_session=387.14 rss_mb=5787"
equal(StatusParser.value(of: "event", in: pacing), "sr_pacing", "event")
equal(StatusParser.number(of: "frame", in: pacing), 120, "frame")
equal(StatusParser.double(of: "fps", in: pacing), 44.4, "fps")
equal(StatusParser.double(of: "mean_ms", in: pacing), 22.50, "mean_ms")
equal(StatusParser.double(of: "min_ms", in: pacing), 1.95, "min_ms")
equal(StatusParser.double(of: "max_ms", in: pacing), 387.14, "max_ms")
equal(StatusParser.number(of: "spikes_over_25ms", in: pacing), 6, "spikes")
equal(StatusParser.double(of: "rss_mb", in: pacing), 5787, "rss_mb")
// `ms` is a suffix of three keys here and a key of none: it must not resolve.
equal(StatusParser.value(of: "ms", in: pacing), nil, "bare `ms` does not match a suffix")
equal(StatusParser.value(of: "absent", in: pacing), nil, "absent key")

// detail= is free text with spaces and always last on its line.
let fallback = "event=sr_fallback api=metal4 reason=contract_or_encode frame=38904 "
             + "detail=temporal frame requires an accepted reset frame"
equal(StatusParser.value(of: "reason", in: fallback), "contract_or_encode", "reason before detail")
equal(StatusParser.value(of: "detail", in: fallback),
      "temporal frame requires an accepted reset frame", "detail takes the remainder")
equal(StatusParser.number(of: "frame", in: fallback), 38904, "frame alongside detail")

// MARK: - session fold

print("\nsession fold")

var p = StatusParser()
p.consume(line: "event=encode_hook class=_MFXTemporalScalingEffectV4 original=0x7ffb1491302a")
equal(p.status.state, .noRuntime, "hooks alone are not a state change")
p.consume(line: "event=bridge_ready process=re9.exe descriptor=MTLFXTemporalScalerDescriptor encode_hooks=12")
equal(p.status.state, .injected, "bridge_ready -> injected")
p.consume(line: "event=sr_arm_ready bundle=/x one_frame=false sustained=true")
equal(p.status.state, .armed, "sr_arm_ready -> armed")
p.consume(line: "event=sr_profile_selected output=1728x972 render=1152x648")
equal(p.status.outputResolution, "1728x972", "output resolution")
equal(p.status.renderResolution, "1152x648", "render resolution")
p.consume(line: "event=sr_exact_scheduled mode=sustained api=metal4 frame=120 reset=0 game_reset=0 "
              + "jitter=0.1,0.2 fallbacks=0 game_resets=1")
equal(p.status.state, .active, "scheduled -> active")
equal(p.status.framesScheduled, 120, "frames scheduled")
equal(p.status.gameResets, 1, "game resets")
p.consume(line: pacing)
equal(p.status.pacing?.fps, 44.4, "pacing captured")

// MARK: - fallbacks must not be double counted

print("\nfallback accounting")

var f = StatusParser()
f.consume(line: "event=sr_fallback api=metal4 reason=contract_or_encode frame=100 detail=x")
f.consume(line: "event=sr_fallback api=metal4 reason=contract_or_encode frame=200 detail=x")
equal(f.status.fallbacks, 2, "counted from lines when nothing authoritative yet")
// sr_exact_scheduled carries the runtime's own total; it must win rather than add.
f.consume(line: "event=sr_exact_scheduled mode=sustained api=metal4 frame=240 reset=0 game_reset=0 "
              + "jitter=0,0 fallbacks=7 game_resets=0")
equal(f.status.fallbacks, 7, "authoritative total replaces the line count")
// The runtime stops logging individual sr_fallback lines after the first 64, so a later line must
// not push the total past what the runtime itself reports.
f.consume(line: "event=sr_fallback api=metal4 reason=contract_or_encode frame=300 detail=x")
equal(f.status.fallbacks, 7, "a later line does not exceed the authoritative total")
f.consume(line: "event=sr_exact_scheduled mode=sustained api=metal4 frame=360 reset=0 game_reset=0 "
              + "jitter=0,0 fallbacks=8 game_resets=0")
equal(f.status.fallbacks, 8, "the runtime's next total is adopted")

// MARK: - failure and recovery

print("\nfailure and recovery")

var d = StatusParser()
d.consume(line: "event=sr_arm_ready bundle=/x one_frame=false sustained=true")
d.consume(line: "event=sr_disarmed reason=gpu_error frame=42 status=4")
if case .failed(let why) = d.status.state {
    check(why.contains("gpu_error"), "disarm records the reason: \(why)")
} else {
    check(false, "disarm -> failed")
}
d.consume(line: "event=sr_exact_scheduled mode=sustained frame=300 reset=0 game_reset=0 jitter=0,0 fallbacks=0")
equal(d.status.state, .active, "serving frames again clears the failure")
equal(d.status.lastFailure, nil, "failure text cleared on recovery")

// MARK: - pause and resume

print("\npause and resume")

var q = StatusParser()
q.consume(line: "event=sr_exact_scheduled mode=sustained frame=120 reset=0 game_reset=0 jitter=0,0 fallbacks=0")
q.consume(line: "event=sr_paused marker=/x/re9-sr-pause")
equal(q.status.state, .paused, "paused")
q.consume(line: "event=sr_resumed marker=/x/re9-sr-pause")
equal(q.status.state, .active, "resumed with frames already served -> active")

// MARK: - reset across sessions

print("\nreset")

var r = StatusParser()
r.consume(line: "event=sr_exact_scheduled mode=sustained frame=999 reset=0 game_reset=0 jitter=0,0 fallbacks=3")
r.reset()
equal(r.status.state, .noRuntime, "reset clears state")
equal(r.status.framesScheduled, 0, "reset clears frames")
equal(r.status.fallbacks, 0, "reset clears fallbacks")

// MARK: - malformed input must not crash or corrupt

print("\nmalformed input")

var m = StatusParser()
m.consume(line: "")
m.consume(line: "no equals here at all")
m.consume(line: "event=")
m.consume(line: "event=sr_pacing frame=notanumber fps=alsonot")
m.consume(line: "=leading equals")
m.consume(line: "event=sr_pacing fps=")
check(true, "malformed lines did not crash")
equal(m.status.pacing?.fps, 0, "unparsable numbers fall back to zero rather than stale values")

// MARK: - artifact marks

print("\nartifact marks")

var k = StatusParser()
equal(k.status.artifactMarks, 0, "none to begin with")
k.consume(line: "event=sr_artifact_marked mark=1 frame=1234 note=menubar")
equal(k.status.artifactMarks, 1, "mark counted")
equal(k.status.lastMarkFrame, 1234, "frame recorded for correlation")
k.consume(line: "event=sr_artifact_marked mark=2 frame=5678 note=-")
equal(k.status.artifactMarks, 2, "second mark")
equal(k.status.lastMarkFrame, 5678, "frame updated")
k.reset()
equal(k.status.artifactMarks, 0, "marks cleared with the session")

// MARK: - frame generation observed at the interpolator seam

print("\nframe generation")

var fg = StatusParser()
equal(fg.status.frameGeneration, nil, "nothing until the game creates an interpolator")
fg.consume(line: "event=fg_interpolator_create api=metal4 count=1 device=0x1 target=0x2 "
    + "target_class=_M4FXFrameInterpolatorEffect input=1728x1080 output=1728x1080 color_format=115 "
    + "depth_format=252 motion_format=65 ui_format=115 output_format=115 scaler=0x3 scaler_class=_M4FXTemporalScalingEffectV4")
equal(fg.status.frameGeneration?.api, "metal4", "api")
equal(fg.status.frameGeneration?.interpolatorClass, "_M4FXFrameInterpolatorEffect", "class")
equal(fg.status.frameGeneration?.outputResolution, "1728x1080", "output")
equal(fg.status.frameGeneration?.hooked, false, "not hooked yet")
fg.consume(line: "event=fg_encode_hook_install class=_M4FXFrameInterpolatorEffect installed=1")
equal(fg.status.frameGeneration?.hooked, true, "hook installed")
fg.consume(line: "event=fg_encode count=3 fi=0x2 fi_class=_M4FXFrameInterpolatorEffect")
equal(fg.status.frameGeneration?.encodes, 3, "encode count")
fg.consume(line: "event=fg_pacing count=120 window=60 mean_ms=16.70 source_fps=59.9 min_ms=15.1 max_ms=19.2")
equal(fg.status.frameGeneration?.encodes, 120, "pacing advances the count")
equal(fg.status.frameGeneration?.sourceFps, 59.9, "source fps")
equal(fg.status.frameGeneration?.meanMs, 16.70, "mean interval")
equal(fg.status.frameGeneration?.pacing?.fps, 59.9, "fg pacing as a Pacing sample")
equal(fg.status.effectivePacing?.meanMs, nil, "not effective until FG is serving")
equal(fg.status.state, .noRuntime, "FG observation says nothing about the SR runtime state")
equal(fg.status.frameGeneration?.isServing, false, "observed is not serving")
fg.consume(line: "event=fg_arm_ready bundle=/x/re9-fg-runtime-bundle attempts=1 profile=640x360")
equal(fg.status.frameGeneration?.armed, true, "armed")
equal(fg.status.frameGeneration?.lowering, nil, "legacy arming leaves arithmetic unknown")
fg.consume(line: "event=fg_arm_ready lowering=hardware")
equal(fg.status.frameGeneration?.lowering, .accelerated, "hardware arithmetic recorded")
fg.consume(line: "event=fg_arm_ready lowering=exact")
equal(fg.status.frameGeneration?.lowering, .reference, "legacy exact label is reference selection")
fg.consume(line: "event=fg_arm_ready lowering=other")
equal(fg.status.frameGeneration?.lowering, nil, "unknown arithmetic is not guessed")
fg.consume(line: "event=fg_arm_ready lowering=hardware")
fg.consume(line: "event=fg_arm_ready")
equal(fg.status.frameGeneration?.lowering, nil, "missing mode clears a previous mode")
fg.consume(line: "event=fg_served api=metal4 frame=120 generated=118 resets=2 game_reset=0 fallbacks=1 render=1002x564 output=1728x972")
equal(fg.status.frameGeneration?.served, 120, "served count")
equal(fg.status.frameGeneration?.fallbacks, 1, "fallbacks from the served line")
equal(fg.status.frameGeneration?.isServing, true, "serving")
equal(fg.status.state, .active, "serving FG makes the session active even without SR frames")
equal(fg.status.effectivePacing?.fps, 59.9, "with SR idle the FG pacing drives the stats")
fg.consume(line: "event=sr_exact_scheduled mode=sustained frame=10 reset=0 game_reset=0 jitter=0,0 fallbacks=0")
fg.consume(line: "event=sr_pacing frame=10 window=10 mean_ms=14.0 fps=71.4 min_ms=12 max_ms=16 spikes_over_25ms=0 worst_ms_session=16 rss_mb=1")
equal(fg.status.effectivePacing?.fps, 71.4, "SR pacing wins once SR serves frames")
fg.consume(line: "event=fg_fallback frame=121 fallbacks=2 detail=interpolator geometry unsupported")
equal(fg.status.frameGeneration?.fallbacks, 2, "fallback counted")
equal(fg.status.frameGeneration?.lastFailure, "interpolator geometry unsupported", "fallback detail kept")
fg.consume(line: "event=fg_disarmed reason=consecutive_fallbacks count=240")
equal(fg.status.frameGeneration?.armed, false, "disarmed")
equal(fg.status.frameGeneration?.isServing, false, "historical frames cannot keep disarmed FG serving")
fg.consume(line: "event=fg_paused control=/x/netal-control")
equal(fg.status.frameGeneration?.paused, true, "paused by the control file")
equal(fg.status.frameGeneration?.isServing, false, "paused FG is not serving")
fg.consume(line: "event=fg_resumed control=/x/netal-control")
equal(fg.status.frameGeneration?.paused, false, "resumed")
fg.consume(line: "event=fg_paused control=/x/netal-control")
fg.consume(line: "event=fg_served api=metal4 frame=200 generated=198 resets=2 game_reset=0 fallbacks=2 render=1002x564 output=1728x972")
equal(fg.status.frameGeneration?.paused, false, "a served frame means it is not paused")
equal(fg.status.frameGeneration?.isServing, true, "new served frame restores current serving state")
fg.consume(line: "event=fg_served frame=201 generated=0")
equal(fg.status.frameGeneration?.isServing, false, "handled source frames with zero generated frames are not network serving")
fg.consume(line: "event=fg_served frame=202")
equal(fg.status.frameGeneration?.generated, nil, "missing generated count remains unknown")
equal(fg.status.frameGeneration?.isServing, false, "source-frame count alone cannot prove frame generation")
fg.consume(line: "event=fg_arm_ready lowering=hardware")
equal(fg.status.frameGeneration?.generated, nil, "new arming clears previous generated evidence")
fg.reset()
equal(fg.status.frameGeneration, nil, "cleared with the session")

var fgZero = StatusParser()
fgZero.consume(line: "event=fg_arm_ready lowering=hardware")
equal(fgZero.status.state, .armed, "FG-only initialization is armed, not active")
fgZero.consume(line: "event=fg_served frame=1 generated=0")
equal(fgZero.status.state, .armed, "reset-only source handling cannot activate FG")
fgZero.consume(line: "event=fg_served frame=2 generated=1")
equal(fgZero.status.state, .active, "reported generation activates FG-only session")
fgZero.consume(line: "event=fg_paused")
equal(fgZero.status.state, .armed, "paused FG-only session loses active state")

// MARK: - attachment identity

print("\nattachment identity")

var att = StatusParser()
att.consume(line: "event=bridge_ready process=re9.exe pid=21881 descriptor=MTLFXTemporalScalerDescriptor encode_hooks=7")
equal(att.status.processName, "re9.exe", "process from bridge_ready")
equal(att.status.processID, 21881, "pid from bridge_ready")
var legacy = StatusParser()
legacy.consume(line: "event=bridge_ready process=re9.exe descriptor=MTLFXTemporalScalerDescriptor encode_hooks=7")
equal(legacy.status.processID, nil, "older bridges log no pid")

// MARK: - the pause marker path must match the runtime's own derivation

print("\npause marker path")

// The runtime sets g_sr_pause_marker to <dirname of log>/re9-sr-pause. The app must derive the
// identical path or the toggle silently does nothing.
let log = URL(fileURLWithPath: "/a/b/build/re9_crossover/re9-live-metalfx.log")
let marker = log.deletingLastPathComponent().appendingPathComponent("re9-sr-pause")
equal(marker.path, "/a/b/build/re9_crossover/re9-sr-pause", "pause marker sits beside the log")
let mark = log.deletingLastPathComponent().appendingPathComponent("re9-sr-mark")
equal(mark.path, "/a/b/build/re9_crossover/re9-sr-mark", "mark marker sits beside the log")


// MARK: - flashes and SR cost

print("\nflash and SR-cost reporting")

var flashParser = StatusParser()
flashParser.consume(line: "event=sr_exact_scheduled mode=sustained frame=120 fallbacks=0")
flashParser.consume(line: "event=sr_exact_completed mode=sustained frame=120 status=4 "
                        + "command_gpu_ms=12.4 flashes=0 error=none")
equal(flashParser.status.flashes, 0, "a clean session reports zero rather than nothing")
check(!flashParser.status.isUnhealthy, "zero flashes is healthy")

flashParser.consume(line: "event=sr_flash frame=1440 fill=511 hot=0 nonfinite=0 of=512 "
                        + "gpu_ms=41.20 total=1 detail=poison fill reached the output")
equal(flashParser.status.flashes, 1, "a flash event counts")
check(flashParser.status.isUnhealthy, "any flash is unhealthy, even with no fallbacks")

// The completion line carries the authoritative total, so it wins over counted events.
flashParser.consume(line: "event=sr_exact_completed mode=sustained frame=1560 status=4 flashes=3 error=none")
equal(flashParser.status.flashes, 3, "the runtime's own total wins")

// gpu_ms rides on the pacing line so it shares a sample with frame time.
var gpuParser = StatusParser()
gpuParser.consume(line: "event=sr_pacing frame=60 window=60 mean_ms=17.9 fps=55.8 min_ms=7.0 "
                      + "max_ms=29.0 spikes_over_25ms=4 worst_ms_session=51.2 rss_mb=3517 gpu_ms=8.85")
equal(gpuParser.status.pacing?.gpuMs, 8.85, "gpu_ms parsed")
equal(gpuParser.status.pacing?.rssMB, 3517, "rss_mb still parsed beside it")

// An older runtime does not emit it, and that must read as absent rather than as zero.
var oldParser = StatusParser()
oldParser.consume(line: "event=sr_pacing frame=60 window=60 mean_ms=17.9 fps=55.8 min_ms=7.0 "
                      + "max_ms=29.0 spikes_over_25ms=4 worst_ms_session=51.2 rss_mb=3517")
check(oldParser.status.pacing?.gpuMs == nil, "absent gpu_ms is nil, not zero")

// MARK: - history

print("\nmetrics history")

var history = MetricsHistory()
check(history.isEmpty, "starts empty")
for frame in stride(from: UInt64(60), through: 600, by: 60) {
    var p = Pacing(); p.frame = frame; p.meanMs = 16; p.minMs = 14; p.maxMs = 20
    history.append(p)
}
equal(history.samples.count, 10, "one sample per pacing line")

// The monitor re-reads the tail after a truncation; a repeated frame would draw a step that
// never happened.
var repeated = Pacing(); repeated.frame = 600; repeated.meanMs = 99
history.append(repeated)
equal(history.samples.count, 10, "a repeated frame is ignored")

// One stall must not flatten the axis for everything else.
var stall = Pacing(); stall.frame = 660; stall.meanMs = 16; stall.minMs = 14; stall.maxMs = 1639
history.append(stall)
check(history.frameTimeCeiling < 100,
      "a 1.6 s stall does not set the ceiling (got \(history.frameTimeCeiling))")
check(history.frameTimeCeiling >= 20, "the 60 Hz line always fits")

for frame in stride(from: UInt64(720), through: 20_000, by: 60) {
    var p = Pacing(); p.frame = frame; p.maxMs = 20
    history.append(p)
}
equal(history.samples.count, MetricsHistory.capacity, "the window is bounded")

history.reset()
check(history.isEmpty, "reset clears it")

print(failures == 0 ? "\nall checks passed" : "\n\(failures) FAILURES")
exit(failures == 0 ? 0 : 1)
