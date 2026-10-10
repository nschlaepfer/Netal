import Foundation

/// One `sr_pacing` sample: the runtime emits these every 60 frames - about a second.
struct Pacing: Equatable {
    var frame: UInt64 = 0
    var fps: Double = 0
    var meanMs: Double = 0
    var minMs: Double = 0
    var maxMs: Double = 0
    var spikes: UInt64 = 0          // cumulative frames over 25 ms this session
    var worstMs: Double = 0
    var gpuMs: Double?              // SR pass GPU time for the most recent frame
    /// The machine's own thermal state, which is the difference between "Netal got slower" and
    /// "everything got slower". Absent in older runtime builds.
    var thermal: String?
    var rssMB: Double?              // absent in older runtime builds
}

/// Frame generation as seen at the MetalFX frame-interpolator seam (`fg_*` events).
///
/// Creation and hook observations are separate from reported generated frames.
/// Arithmetic selection is recorded separately from numerical validation.
struct FrameGeneration: Equatable {
    var api: String?                 // metal3 | metal4
    var interpolatorClass: String?
    var inputResolution: String?     // WxH the interpolator was created for
    var outputResolution: String?
    var hooked = false               // the observe-only encode hook is installed
    var encodes: UInt64 = 0          // one per real source frame
    var sourceFps: Double?           // from fg_pacing
    var meanMs: Double?              // source frame interval
    /// The same numbers as a Pacing sample, so the Status stats and graphs can run on frame
    /// generation alone (Super Resolution idle: DLAA, or no profile for the size).
    var pacing: Pacing?
    /// Netal's own network is generating the frames (fg_served); until then the row is observed only.
    var armed = false
    enum Lowering: String {
        case reference = "exact"
        case accelerated = "hardware"
        var label: String { self == .reference ? "Reference" : "Accelerated" }
    }
    var lowering: Lowering?
    /// The control file asked the bridge to hand frames back to MetalFX (fg=0).
    var paused = false
    var served: UInt64 = 0          // source frames handled by the adapter
    var generated: UInt64?         // missing in old/incomplete records
    var fallbacks: UInt64 = 0
    var lastFailure: String?
    var isServing: Bool { armed && !paused && (generated ?? 0) > 0 && lastFailure == nil }
}

/// What the runtime is doing, in the order a session moves through it.
enum RuntimeState: Equatable {
    case noRuntime                  // no log, or nothing has happened in it
    case injected                   // bridge is in the process, session not armed
    case armed                      // session loaded, no frames served yet
    case active                     // frames are being served by Netal
    case paused                     // passing through to MetalFX (pause marker)
    case failed(String)             // disarmed or failed to arm; reason attached

    var title: String {
        switch self {
        case .noRuntime: return "Not running"
        case .injected:  return "Injected"
        case .armed:     return "Armed"
        case .active:    return "Active"
        case .paused:    return "Paused"
        case .failed:    return "Error"
        }
    }

    /// SF Symbol for the menu-bar item. Distinct shapes, not just colours, so the state is legible
    /// in a monochrome menu bar and to anyone who cannot rely on hue.
    var symbol: String {
        switch self {
        case .noRuntime: return "circle.dotted"
        case .injected:  return "circle"
        case .armed:     return "circle.lefthalf.filled"
        case .active:    return "circle.fill"
        case .paused:    return "pause.circle"
        case .failed:    return "exclamationmark.triangle.fill"
        }
    }
}

/// Everything the UI needs, derived only from the runtime log and the pause marker.
struct RuntimeStatus: Equatable {
    var state: RuntimeState = .noRuntime
    var pacing: Pacing?
    var framesScheduled: UInt64 = 0
    var gameResets: UInt64 = 0

    /// The runtime logs individual `sr_fallback` lines only for the first 64, but every
    /// `sr_exact_scheduled` carries its own running total. Prefer that total when we have one, so a
    /// late line cannot push the count past what the runtime itself reports.
    var fallbacksReported: UInt64?
    var fallbackLines: UInt64 = 0
    var fallbacks: UInt64 { fallbacksReported ?? fallbackLines }
    var outputResolution: String?
    var renderResolution: String?
    var lastFailure: String?
    var logPresent = false
    /// Artifact marks the runtime has recorded this session (`event=sr_artifact_marked`).
    var artifactMarks: UInt64 = 0

    /// Frames the runtime detected as blown - the white flash, self-reported. The running
    /// total rides on every periodic completion line, so a session that saw none says so.
    var flashes: UInt64 = 0
    /// Batched-pipeline coverage for the selected profile. A layer without one runs the transpiled
    /// per-window kernel: correct, roughly twice the GPU time. A short set therefore looks like a
    /// healthy session that is simply slow, which is exactly how it went unnoticed - so the panel
    /// says it outright.
    /// Process the bridge is inside, from `bridge_ready`. Used when the log path carries no
    /// container to resolve an app bundle from.
    var processName: String?
    /// The bridge's pid from `bridge_ready`, so a running game can be told apart from a stale log.
    var processID: Int?
    var pipelineLayersPresent: Int?
    var pipelineLayersTotal: Int?
    var pipelineLayersMissing: String?
    var pipelineIncomplete: Bool {
        guard let present = pipelineLayersPresent, let total = pipelineLayersTotal, total > 0 else { return false }
        return present < total
    }
    /// The frame the most recent mark landed on, for correlating against a dump or a pacing window.
    var lastMarkFrame: UInt64?

    /// How long the runtime has been arming, and why, from `sr_arm_waiting`.
    ///
    /// Arming compiles pipelines, and at a resolution with no batched set it compiles the
    /// transpiled kernel for every layer instead - which ran to 91 seconds in Resident Evil 4 at
    /// 3456x1944. Throughout that the panel said "waiting for the game to render" while the game
    /// rendered six thousand frames. The runtime was saying exactly what it was doing all along;
    /// nothing was listening.
    var armingSeconds: Double?
    var armingDetail: String?

    /// Present once the game has created a MetalFX frame interpolator this session.
    var frameGeneration: FrameGeneration?

    /// The pacing to show: Super Resolution's when it is producing frames, otherwise frame
    /// generation's (its source-frame interval). Both describe the game's rendered frames.
    var effectivePacing: Pacing? {
        if let pacing, framesScheduled > 0 { return pacing }
        if let fg = frameGeneration, fg.isServing, let fgPacing = fg.pacing { return fgPacing }
        return pacing
    }
    /// Super Resolution is producing frames (as opposed to armed with nothing to upscale).
    var superResolutionServing: Bool { framesScheduled > 0 }

    /// True when the runtime reported a hard failure, or is shedding frames to MetalFX.
    var isUnhealthy: Bool {
        if case .failed = state { return true }
        return fallbacks > 0 || flashes > 0
    }
}

/// Parses the runtime's `key=value` log lines into a `RuntimeStatus`.
///
/// The log has no timestamps, so this deliberately says nothing about liveness — the monitor
/// decides that from file growth. Feed lines in order; the parser is a fold over the session.
struct StatusParser {
    private(set) var status = RuntimeStatus()

    /// Cleared when a new session truncates the log, so stale state never leaks across runs.
    mutating func reset() { status = RuntimeStatus() }

    mutating func consume(line: String) {
        guard let event = Self.value(of: "event", in: line) else { return }
        switch event {
        case "bridge_ready":
            if case .noRuntime = status.state { status.state = .injected }
            // The only identity a non-sandboxed runtime carries. A CrossOver or bare-process game
            // has no container in its log path, so this is the difference between naming the game
            // and showing nothing at all.
            status.processName = Self.value(of: "process", in: line)
            status.processID = Self.number(of: "pid", in: line).map(Int.init)

        case "sr_arm_ready":
            if status.state != .active { status.state = .armed }
            status.armingSeconds = nil
            status.armingDetail = nil

        case "sr_arm_waiting":
            status.armingSeconds = Self.double(of: "elapsed_s", in: line)
            status.armingDetail = Self.value(of: "detail", in: line)

        case "scaler_create":
            // The first proof the game is drawing through MetalFX, and it arrives seconds in -
            // long before the session can report a thing. It is also the only place the output
            // geometry appears when arming never finishes, which is when knowing it matters most.
            status.outputResolution = Self.value(of: "output", in: line)
            status.renderResolution = Self.value(of: "render", in: line)

        case "sr_arm_failed", "bridge_failed", "sr_init_thread_failed":
            let detail = Self.value(of: "detail", in: line) ?? Self.value(of: "reason", in: line)
            status.lastFailure = detail ?? event
            status.state = .failed(status.lastFailure!)

        case "sr_disarmed":
            let reason = Self.value(of: "reason", in: line) ?? "unknown"
            status.lastFailure = "disarmed: \(reason)"
            status.state = .failed(status.lastFailure!)

        case "sr_pipeline_layers":
            status.pipelineLayersPresent = Self.number(of: "present", in: line).map(Int.init)
            status.pipelineLayersTotal = Self.number(of: "total", in: line).map(Int.init)
            let missing = Self.value(of: "missing", in: line)
            status.pipelineLayersMissing = (missing == "-" || missing?.isEmpty == true) ? nil : missing
        case "sr_profile_selected":
            status.outputResolution = Self.value(of: "output", in: line)
            status.renderResolution = Self.value(of: "render", in: line)

        case "sr_profile_unavailable":
            let detail = Self.value(of: "detail", in: line) ?? "no profile for this resolution"
            status.lastFailure = detail
            status.state = .failed(detail)

        case "sr_exact_scheduled":
            // Serving frames again clears a previous failure: the runtime recovered.
            if case .failed = status.state { status.lastFailure = nil }
            if status.state != .paused { status.state = .active }
            if let frame = Self.number(of: "frame", in: line) { status.framesScheduled = frame }
            if let f = Self.number(of: "fallbacks", in: line) { status.fallbacksReported = f }
            if let r = Self.number(of: "game_resets", in: line) { status.gameResets = r }

        case "sr_flash":
            // The event itself is per-occurrence; the authoritative total arrives on the
            // completion line below, so only take it if we have nothing better yet.
            if let n = Self.number(of: "total", in: line) { status.flashes = n }
        
        case "sr_exact_completed":
            if let n = Self.number(of: "flashes", in: line) { status.flashes = n }
        
        case "sr_fallback":
            status.fallbackLines += 1

        case "sr_artifact_marked":
            if let n = Self.number(of: "mark", in: line) { status.artifactMarks = n }
            if let f = Self.number(of: "frame", in: line) { status.lastMarkFrame = f }

        case "sr_paused":
            status.state = .paused

        case "sr_resumed":
            status.state = status.framesScheduled > 0 ? .active : .armed

        case "fg_interpolator_create":
            var fg = status.frameGeneration ?? FrameGeneration()
            fg.api = Self.value(of: "api", in: line)
            fg.interpolatorClass = Self.value(of: "target_class", in: line)
            fg.inputResolution = Self.value(of: "input", in: line)
            fg.outputResolution = Self.value(of: "output", in: line)
            status.frameGeneration = fg

        case "fg_encode_hook_install":
            var fg = status.frameGeneration ?? FrameGeneration()
            fg.hooked = Self.value(of: "installed", in: line) == "1"
            status.frameGeneration = fg

        case "fg_encode":
            var fg = status.frameGeneration ?? FrameGeneration()
            if let n = Self.number(of: "count", in: line) { fg.encodes = max(fg.encodes, n) }
            status.frameGeneration = fg

        case "fg_arm_ready":
            var fg = status.frameGeneration ?? FrameGeneration()
            fg.armed = true
            fg.served = 0
            fg.generated = nil
            switch status.state {
            case .noRuntime, .injected: status.state = .armed
            case .active where !status.superResolutionServing: status.state = .armed
            default: break
            }
            fg.lowering = Self.value(of: "lowering", in: line).flatMap(FrameGeneration.Lowering.init(rawValue:))
            status.frameGeneration = fg

        case "fg_paused", "fg_resumed":
            var fg = status.frameGeneration ?? FrameGeneration()
            fg.paused = event == "fg_paused"
            if fg.paused && !status.superResolutionServing, case .active = status.state { status.state = .armed }
            status.frameGeneration = fg

        case "fg_served":
            var fg = status.frameGeneration ?? FrameGeneration()
            fg.armed = true
            fg.paused = false
            if let n = Self.number(of: "frame", in: line) { fg.served = max(fg.served, n) }
            fg.generated = Self.number(of: "generated", in: line)
            if let f = Self.number(of: "fallbacks", in: line) { fg.fallbacks = f }
            fg.lastFailure = nil
            if fg.isServing {
                switch status.state {
                case .noRuntime, .injected, .armed: status.state = .active
                case .active, .paused, .failed: break
                }
            } else if !status.superResolutionServing, case .active = status.state {
                status.state = .armed
            }
            status.frameGeneration = fg

        case "fg_fallback", "fg_disarmed", "fg_arm_failed":
            var fg = status.frameGeneration ?? FrameGeneration()
            if let f = Self.number(of: "fallbacks", in: line) { fg.fallbacks = f }
            fg.lastFailure = Self.value(of: "detail", in: line) ?? Self.value(of: "reason", in: line) ?? event
            if event == "fg_disarmed" { fg.armed = false }
            if !status.superResolutionServing, case .active = status.state { status.state = .armed }
            status.frameGeneration = fg

        case "fg_pacing":
            var fg = status.frameGeneration ?? FrameGeneration()
            if let n = Self.number(of: "count", in: line) { fg.encodes = max(fg.encodes, n) }
            fg.sourceFps = Self.double(of: "source_fps", in: line)
            fg.meanMs = Self.double(of: "mean_ms", in: line)
            var p = Pacing()
            p.frame = Self.number(of: "count", in: line) ?? 0
            p.fps = fg.sourceFps ?? 0
            p.meanMs = fg.meanMs ?? 0
            p.minMs = Self.double(of: "min_ms", in: line) ?? 0
            p.maxMs = Self.double(of: "max_ms", in: line) ?? 0
            fg.pacing = p
            status.frameGeneration = fg

        case "sr_pacing":
            var p = Pacing()
            p.frame = Self.number(of: "frame", in: line) ?? 0
            p.fps = Self.double(of: "fps", in: line) ?? 0
            p.meanMs = Self.double(of: "mean_ms", in: line) ?? 0
            p.minMs = Self.double(of: "min_ms", in: line) ?? 0
            p.maxMs = Self.double(of: "max_ms", in: line) ?? 0
            p.spikes = Self.number(of: "spikes_over_25ms", in: line) ?? 0
            p.worstMs = Self.double(of: "worst_ms_session", in: line) ?? 0
            p.rssMB = Self.double(of: "rss_mb", in: line)
            p.gpuMs = Self.double(of: "gpu_ms", in: line)
            p.thermal = Self.value(of: "thermal", in: line)
            status.pacing = p

        default:
            break
        }
        status.logPresent = true
    }

    // MARK: - key=value extraction

    /// Reads `key=` up to the next space. Values in this log never contain spaces except for
    /// free-text `detail=`, which is always last on its line — so that case takes the remainder.
    static func value(of key: String, in line: String) -> String? {
        var searchStart = line.startIndex
        while let range = line.range(of: key + "=", range: searchStart..<line.endIndex) {
            // Must be at the start of the line or preceded by a space, so `ms=` does not match
            // `mean_ms=` and `frame=` does not match `worst_frame=`.
            let atFieldStart = range.lowerBound == line.startIndex
                || line[line.index(before: range.lowerBound)] == " "
            if atFieldStart {
                let rest = line[range.upperBound...]
                if key == "detail" { return String(rest) }
                if let space = rest.firstIndex(of: " ") { return String(rest[..<space]) }
                return String(rest)
            }
            searchStart = range.upperBound
        }
        return nil
    }

    static func number(of key: String, in line: String) -> UInt64? {
        guard let raw = value(of: key, in: line) else { return nil }
        return UInt64(raw)
    }

    static func double(of key: String, in line: String) -> Double? {
        guard let raw = value(of: key, in: line) else { return nil }
        return Double(raw)
    }
}
