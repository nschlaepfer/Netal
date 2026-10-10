// Opt-in, bounded receipts for one explicitly named game window. No screen pixels,
// audio, other window titles, or application paths are saved by this harness.
import AppKit
import CoreGraphics
import Foundation

/// A baseline is distinct from SR-only, FG-only and their combination. Requested modes are
/// accepted only when actual presentation callbacks carry the matching processing evidence.
enum CaptureValidationMode: String, CaseIterable {
    case baseline, sr, fg, combined
    var requestsSR: Bool { self == .sr || self == .combined }
    var requestsFG: Bool { self == .fg || self == .combined }
    func validates(captured: UInt64, presented: UInt64, generated: UInt64, upscaled: UInt64) -> Bool {
        guard captured > 0, presented > 0, generated <= presented, upscaled <= presented else { return false }
        return (requestsFG ? generated > 0 : generated == 0)
            && (requestsSR ? upscaled > 0 : upscaled == 0)
    }
}

@MainActor
enum CaptureDiagnostics {
    /// Call before normal SwiftUI scenes. Diagnostic modes own a small AppKit run loop
    /// and terminate after writing their receipt; ordinary launches simply return false.
    static func runIfRequested(permissionCheck: () -> Bool = { CGPreflightScreenCaptureAccess() }) -> Bool {
        let args = Array(CommandLine.arguments.dropFirst())
        if args == ["--capture-permission-status"] {
            emit(["granted": permissionCheck()])
            exit(0)
        }
        guard args.contains(where: { $0.hasPrefix("--capture-") }) else { return false }
        do {
            let granted = permissionCheck()
            let options = try Options(args, permissionGranted: granted)
            guard granted else {
                finish(options, status: "permission-denied", code: 77,
                       error: "Screen Recording permission is not already granted. No permission was requested.")
            }
            NSApplication.shared.setActivationPolicy(.accessory)
            try validateSelectedWindow(options, requireForeground: options.waitForeground == 0)
            DispatchQueue.main.async {
                Task { @MainActor in await run(options) }
            }
            NSApplication.shared.run()
            return true
        } catch {
            emit(["status": "invalid-request", "error": error.localizedDescription])
            exit(64)
        }
    }

    private struct Options {
        let windowID: UInt32
        let processID: pid_t
        let seconds: Double
        let outputURL: URL
        let mode: CaptureValidationMode
        let waitForeground: Double
        let inspectOutput: Bool
        var frameGeneration: Bool { mode.requestsFG }

        init(_ args: [String], permissionGranted: Bool) throws {
            var values: [String: String] = [:]
            var fg = false
            var inspectOutput = false
            var i = 0
            while i < args.count {
                let key = args[i]
                if key == "--inspect-output" {
                    guard !inspectOutput else { throw Failure("Duplicate --inspect-output flag.") }
                    inspectOutput = true; i += 1; continue
                }
                if key == "--fg" {
                    guard !fg else { throw Failure("Duplicate --fg flag.") }
                    fg = true; i += 1; continue
                }
                guard ["--capture-smoke-window", "--capture-smoke-bundle", "--pid", "--seconds", "--output", "--mode", "--wait-foreground"].contains(key),
                      i + 1 < args.count, values[key] == nil else {
                    throw Failure("Usage: --capture-smoke-window WINDOW_ID --pid PID --seconds 1…30 --output NEW_JSON [--mode baseline|sr|fg|combined] [--wait-foreground 0…120] [--inspect-output]")
                }
                values[key] = args[i + 1]; i += 2
            }
            guard let seconds = values["--seconds"].flatMap(Double.init), seconds.isFinite,
                  (1...30).contains(seconds), let output = values["--output"], !output.isEmpty else {
                throw Failure("Provide a valid game window ID, its process ID, 1…30 seconds, and a new JSON output file.")
            }
            let mode: CaptureValidationMode
            if let text = values["--mode"] {
                guard !fg, let parsed = CaptureValidationMode(rawValue: text) else {
                    throw Failure("Choose one mode: baseline, sr, fg or combined. Do not combine --mode with the legacy --fg flag.")
                }
                mode = parsed
            } else { mode = fg ? .fg : .baseline }
            let wait = values["--wait-foreground"].flatMap(Double.init) ?? 0
            guard wait.isFinite, (0...120).contains(wait), values["--wait-foreground"] == nil || Double(values["--wait-foreground"]!) != nil else {
                throw Failure("Foreground wait must be between 0 and 120 seconds.")
            }
            let url = URL(fileURLWithPath: output).standardizedFileURL
            guard !FileManager.default.fileExists(atPath: url.path) else {
                throw Failure("The receipt file already exists. Choose a new output file.")
            }
            var directory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.deletingLastPathComponent().path, isDirectory: &directory),
                  directory.boolValue else { throw Failure("The receipt's parent directory must already exist.") }
            let windowID: UInt32
            let processID: pid_t
            if let bundle = values["--capture-smoke-bundle"], !bundle.isEmpty {
                guard values["--capture-smoke-window"] == nil, values["--pid"] == nil else {
                    throw Failure("Choose either an explicit window ID and PID or an exact app bundle identifier.")
                }
                if permissionGranted {
                    let apps = NSWorkspace.shared.runningApplications.filter { $0.bundleIdentifier == bundle && $0.processIdentifier != getpid() }
                    guard apps.count == 1, let app = apps.first else {
                        throw Failure("The requested app must have exactly one running process. No app was launched.")
                    }
                    let descriptions = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
                    let matching = descriptions.filter {
                        guard ($0[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == app.processIdentifier,
                              ($0[kCGWindowLayer as String] as? NSNumber)?.intValue == 0,
                              let dictionary = $0[kCGWindowBounds as String] as? [String: Any],
                              let rect = CGRect(dictionaryRepresentation: dictionary as CFDictionary) else { return false }
                        return rect.width >= 64 && rect.height >= 64
                    }
                    guard matching.count == 1, let id = (matching.first?[kCGWindowNumber as String] as? NSNumber)?.uint32Value else {
                        throw Failure("The requested app must have exactly one visible normal window. Use an explicit window ID when it has more than one.")
                    }
                    windowID = id; processID = app.processIdentifier
                } else {
                    // Permission denial is tested before discovering any windows.
                    windowID = 0; processID = 0
                }
            } else {
                guard let id = values["--capture-smoke-window"].flatMap(UInt32.init), id > 0,
                      let pid = values["--pid"].flatMap(Int32.init), pid > 0, pid != getpid() else {
                    throw Failure("Provide a valid game window ID and PID, or --capture-smoke-bundle with an exact app identifier.")
                }
                windowID = id; processID = pid
            }
            self.windowID = windowID; self.processID = processID; self.seconds = seconds
            outputURL = url; self.mode = mode; waitForeground = wait
            self.inspectOutput = inspectOutput
        }
    }

    private static func validateSelectedWindow(_ options: Options, requireForeground: Bool = true) throws {
        guard !requireForeground || NSWorkspace.shared.frontmostApplication?.processIdentifier == options.processID else {
            throw Failure("The specified game must already be the foreground app. No app was activated.")
        }
        let descriptions = CGWindowListCopyWindowInfo(.optionIncludingWindow, options.windowID) as? [[String: Any]] ?? []
        let matches: ([String: Any]) -> Bool = {
            ($0[kCGWindowNumber as String] as? NSNumber)?.uint32Value == options.windowID
        }
        let selected = descriptions.first(where: matches)
            ?? (CGWindowListCopyWindowInfo(.optionAll, kCGNullWindowID) as? [[String: Any]])?.first(where: matches)
        guard let selected else {
            throw Failure("The selected window is unavailable. Wait for the game's display change to finish, then refresh its window ID.")
        }
        guard (selected[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == options.processID else {
            throw Failure("The selected window no longer belongs to the requested game process. No capture started.")
        }
        guard !requireForeground || (selected[kCGWindowIsOnscreen as String] as? Bool) == true else {
            throw Failure("The selected game window is hidden or minimized. No capture started.")
        }
    }

    private static func run(_ options: Options) async {
        let engine = GameCaptureEngine.shared
        if options.waitForeground > 0 {
            let deadline = ProcessInfo.processInfo.systemUptime + options.waitForeground
            while NSWorkspace.shared.frontmostApplication?.processIdentifier != options.processID,
                  ProcessInfo.processInfo.systemUptime < deadline {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
        do { try validateSelectedWindow(options) }
        catch { finish(options, status: "foreground-unavailable", code: 2, error: error.localizedDescription) }
        engine.presentationMode = options.mode.requestsSR ? .fillDisplay : .window
        engine.start(windowID: options.windowID, scale: 1, frameGeneration: options.frameGeneration,
                     superResolution: options.mode.requestsSR,
                     expectedProcessID: options.processID, activateGame: false, allowPermissionRequest: false,
                     allowOutputInspection: options.inspectOutput)
        let preparationDeadline = ProcessInfo.processInfo.systemUptime + 10
        while engine.state == .preparing && ProcessInfo.processInfo.systemUptime < preparationDeadline {
            if NSWorkspace.shared.frontmostApplication?.processIdentifier != options.processID {
                engine.stop()
                finish(options, status: "foreground-changed", code: 2,
                       error: "The game lost foreground status during preparation.", engine: engine)
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        guard engine.isRunning else {
            let message = engine.errorMessage ?? "Capture did not become ready within 10 seconds."
            engine.stop()
            finish(options, status: "start-failed", code: 2, error: message, engine: engine)
        }
        fputs("Netal capture started: \(options.mode.rawValue)\n", stderr)
        fflush(stderr)
        let start = ProcessInfo.processInfo.systemUptime
        var ended: String?
        while ProcessInfo.processInfo.systemUptime - start < options.seconds {
            if NSWorkspace.shared.frontmostApplication?.processIdentifier != options.processID {
                ended = "The game lost foreground status; diagnostics stopped without activating it."
                break
            }
            if !engine.isRunning { ended = engine.errorMessage ?? "Capture stopped before the requested duration."; break }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        engine.stop()
        let stopDeadline = ProcessInfo.processInfo.systemUptime + 2
        while engine.state == .stopping && ProcessInfo.processInfo.systemUptime < stopDeadline {
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        if let ended { finish(options, status: "interrupted", code: 2, error: ended, engine: engine, elapsed: elapsed) }
        let passed = options.mode.validates(captured: engine.capturedFrameCount,
            presented: engine.presentedFrameCount, generated: engine.generatedFrameCount,
            upscaled: engine.spatiallyUpscaledFrameCount)
        finish(options, status: passed ? "passed" : "requested-mode-unverified", code: passed ? 0 : 2,
               error: passed ? nil : "The requested SR/FG mode was not observed in presented frames. SR requires a genuinely smaller source window; FG requires generated frame presentations.",
               engine: engine, elapsed: elapsed)
    }

    private static func finish(_ options: Options, status: String, code: Int32, error: String?,
                               engine: GameCaptureEngine? = nil, elapsed: Double? = nil) -> Never {
        var receipt: [String: Any] = ["schema": "netal.window-capture-smoke.v1", "status": status,
            "window_id": options.windowID, "process_id": options.processID,
            "requested_seconds": options.seconds, "frame_generation_requested": options.frameGeneration,
            "mode": options.mode.rawValue, "super_resolution_requested": options.mode.requestsSR,
            "foreground_wait_seconds": options.waitForeground,
            "output_inspection_requested": options.inspectOutput,
            "foreground_is_game_at_finish": NSWorkspace.shared.frontmostApplication?.processIdentifier == options.processID,
            "foreground_is_diagnostic_at_finish": NSWorkspace.shared.frontmostApplication?.processIdentifier == getpid(),
            "permission_requested": false, "audio_captured": false, "screen_pixels_saved": false,
            "other_apps_activated": false]
        if let error { receipt["error"] = error }
        if let elapsed { receipt["elapsed_seconds"] = elapsed }
        if let engine {
            receipt["captured_frames"] = engine.capturedFrameCount
            receipt["presented_frames"] = engine.presentedFrameCount
            receipt["generated_presented_frames"] = engine.generatedFrameCount
            receipt["spatially_upscaled_presented_frames"] = engine.spatiallyUpscaledFrameCount
            receipt["dropped_frames"] = engine.droppedFrameCount
            receipt["source_resolution"] = engine.sourceResolution
            receipt["output_resolution"] = engine.outputResolution
            receipt["source_window_level"] = engine.sourceWindowLevel
            receipt["output_window_level"] = engine.outputWindowLevel
            receipt["requested_display_hz"] = engine.requestedDisplayRefreshRate
            receipt["last_processing_gpu_ms"] = engine.processingMilliseconds
            let ticks = engine.displayTickSamples
            receipt["display_tick_sample_count"] = ticks.count
            receipt["presentation_uses_transactions"] = engine.presentationUsesTransactions
            receipt["submitted_presentations"] = engine.submittedPresentationCount
            if ticks.count > 1, let first = ticks.first, let last = ticks.last, last > first {
                receipt["display_callback_hz"] = Double(ticks.count - 1) / (last - first)
                let gaps = zip(ticks.dropFirst(), ticks).map { ($0 - $1) * 1_000 }.sorted()
                receipt["display_tick_interval_ms"] = ["min": gaps.first!, "median": gaps[gaps.count / 2], "max": gaps.last!]
            }
            let samples = engine.presentationSamples.sorted { $0.time < $1.time }
            receipt["presentations"] = samples.map { ["time_seconds": $0.time, "generated": $0.generated, "spatially_upscaled": $0.spatiallyUpscaled] as [String: Any] }
            receipt["presentation_sample_limit"] = 4096
            receipt["presentation_samples_truncated"] = engine.presentedFrameCount > UInt64(samples.count)
            if samples.count > 1 {
                let intervals = zip(samples.dropFirst(), samples).map { ($0.time - $1.time) * 1000 }.sorted()
                receipt["presentation_interval_ms"] = ["min": intervals.first!, "median": intervals[intervals.count / 2],
                    "p95": intervals[min(intervals.count - 1, Int(Double(intervals.count) * 0.95))], "max": intervals.last!]
                receipt["nonpositive_presentation_intervals"] = intervals.filter { $0 <= 0 }.count
            }
            receipt["limitation"] = "Counters and presentation timestamps establish observed delivery only, not game FPS, image quality, input latency, or universal compatibility."
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: options.outputURL, options: .withoutOverwriting)
            emit(receipt)
            exit(code)
        } catch {
            emit(["status": "receipt-write-failed", "error": error.localizedDescription])
            exit(74)
        }
    }

    private static func emit(_ value: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) { print(text); fflush(stdout) }
    }

    private struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
