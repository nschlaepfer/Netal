// The denied-permission integration fixture injects only a false preflight result.
// It never asks the OS for capture permission, discovers windows, or starts capture.
import AppKit
if CommandLine.arguments == [CommandLine.arguments[0], "--test-mode-validation"] {
    var checks = 0
    func check(_ value: Bool) {
        precondition(value, "Capture mode validation failed at check \(checks + 1)")
        checks += 1
    }
    // Each observed combination must identify exactly one mode. In particular,
    // a source-only 1x presentation is neither SR nor FG evidence.
    for mode in CaptureValidationMode.allCases {
        for generated: UInt64 in [0, 12] {
            for upscaled: UInt64 in [0, 24] {
                let expected = (mode.requestsFG == (generated > 0))
                    && (mode.requestsSR == (upscaled > 0))
                check(mode.validates(captured: 12, presented: 24,
                                     generated: generated, upscaled: upscaled) == expected)
            }
        }
        check(!mode.validates(captured: 0, presented: 24, generated: 12, upscaled: 24))
        check(!mode.validates(captured: 12, presented: 0, generated: 0, upscaled: 0))
        check(!mode.validates(captured: 12, presented: 24, generated: 25, upscaled: 24))
        check(!mode.validates(captured: 12, presented: 24, generated: 12, upscaled: 25))
    }
    var queue = CapturePresentationQueue<String>()
    let tick = 1.0 / 120.0
    queue.enqueue(source: "source A", midpoint: "mid A", sourceDelay: tick)
    check(queue.take(at: 0) == "mid A")
    queue.enqueue(source: "source B", midpoint: "mid B", sourceDelay: tick)
    check(queue.take(at: tick) == "source A") // a started pair must finish
    queue.enqueue(source: "source C", midpoint: "mid C", sourceDelay: tick)
    check(queue.take(at: tick * 2) == "mid C") // coalesce only unstarted pairs
    check(queue.take(at: tick * 3) == "source C")
    check(queue.take(at: tick * 4) == nil) // no repeated textures/FPS inflation
    queue.enqueue(source: "slow source", midpoint: "slow mid", sourceDelay: tick * 2)
    check(queue.take(at: 1) == "slow mid")
    check(queue.take(at: 1 + tick) == nil)
    check(queue.take(at: 1 + tick * 2) == "slow source")
    queue.enqueue(source: "source only", midpoint: nil, sourceDelay: tick)
    check(queue.take(at: 2) == "source only")
    check(queue.take(at: 2 + tick) == nil)
    queue.enqueue(source: "stale source", midpoint: "stale mid", sourceDelay: tick)
    check(queue.take(at: 3) == "stale mid")
    queue.enqueue(source: "pending source", midpoint: "pending mid", sourceDelay: tick)
    queue.reset()
    check(queue.take(at: 3 + tick) == nil) // resize/hide/stop clears both slots
    check(CaptureWindowLevel.output(above: 0) == NSWindow.Level.floating.rawValue)
    check(CaptureWindowLevel.output(above: 21) == 22) // CrossOver gameplay window
    check(CaptureWindowLevel.output(above: NSWindow.Level.popUpMenu.rawValue - 2) == NSWindow.Level.popUpMenu.rawValue - 1)
    check(CaptureWindowLevel.output(above: NSWindow.Level.popUpMenu.rawValue - 1) == nil)
    check(CaptureWindowLevel.output(above: NSWindow.Level.screenSaver.rawValue) == nil)
    check(CaptureWindowLevel.output(above: Int.max) == nil)
    print("capture mode, pacing and window-level validation: PASS (\(checks) checks)")
    exit(0)
}
MainActor.assumeIsolated {
    if !CaptureDiagnostics.runIfRequested(permissionCheck: { false }) { exit(64) }
}
