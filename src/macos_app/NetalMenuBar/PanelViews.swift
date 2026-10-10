import SwiftUI
import AppKit

// MARK: - Status

struct StatusView: View {
    @StateObject private var profileCompletion = ProfileCompletion()
    @ObservedObject var monitor: RuntimeMonitor
    /// Progress or failure from a relaunch the user asked for.
    @State private var relaunchNote: String?
    @State private var isLaunching = false
    @State private var restartName = ""
    @State private var restartAction: (() -> Void)?
    @State private var confirmRestart = false
    /// So the model warning can put you where the fix is, rather than naming a tab and leaving
    /// you to find it.
    @Binding var tab: PanelTab

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let failure = monitor.status.lastFailure, case .failed = monitor.displayState {
                Notice(text: failure, tint: .red, symbol: "exclamationmark.triangle.fill")
            }
            if let error = monitor.toggleError {
                Notice(text: error, tint: .red, symbol: "exclamationmark.triangle.fill")
            }
            // A game running without the bridge in it is the single most confusing state this app
            // can be in: everything looks fine and nothing happens. Say what is actually true.
            if let stranded = monitor.unattached.first, !monitor.status.logPresent || !monitor.isLive {
                VStack(alignment: .leading, spacing: 6) {
                    Notice(text: "\(stranded.name) is running without Netal.",
                           tint: .orange, symbol: "bolt.slash.fill")
                    // The notice carries its own fix. A game already running can never be attached
                    // to - DYLD_INSERT_LIBRARIES applies at exec - so the only cure is to start it
                    // again through the launcher, and making the user go and find a shell for that
                    // is how a fixable state becomes a permanent one.
                    if GameLauncher.isAvailable {
                        Button {
                            restartName = stranded.name
                            restartAction = {
                            relaunchNote = "Quitting \(stranded.name)…"
                            isLaunching = true
                            GameLauncher.relaunch(stranded) { outcome in
                                isLaunching = false
                                switch outcome {
                                case .launched:               relaunchNote = "Launched through Netal."
                                case .launcherMissing(let p): relaunchNote = "Launcher not found at \(p)"
                                case .failed(let why):        relaunchNote = why
                                }
                            }
                            }
                            confirmRestart = true
                        } label: {
                            Label("Relaunch with Netal", systemImage: "bolt.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                        .disabled(isLaunching)
                    }
                    if let relaunchNote {
                        Text(relaunchNote).font(.caption2).foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        Text("It has to be started through Netal — a running game cannot be attached to.")
                            .font(.caption2).foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            // Games Netal has never launched were invisible: every other discovery path starts from
            // a log Netal itself wrote. This one asks whether a running app links MetalFX, which is
            // answerable from the binary on disk without privilege. Capability, not use - hence
            // "can use", never "is using".
            if let capable = monitor.metalFXUnattached.first {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        if let icon = capable.icon {
                            Image(nsImage: icon).resizable().frame(width: 22, height: 22)
                                .clipShape(RoundedRectangle(cornerRadius: 5))
                        }
                        VStack(alignment: .leading, spacing: 1) {
                            Text("\(capable.name) references MetalFX").font(.callout)
                            Text(monitor.metalFXUnattached.count > 1
                                 ? "and \(monitor.metalFXUnattached.count - 1) more running"
                                 : "Compatibility is not yet verified.")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                        Spacer(minLength: 0)
                    }
                    if GameLauncher.isAvailable {
                        Button {
                            restartName = capable.name
                            restartAction = {
                            let identity = GameIdentity(bundleID: capable.bundleID, name: capable.name,
                                                        icon: capable.icon, appURL: capable.appURL)
                            isLaunching = true
                            GameLauncher.relaunch(identity) { outcome in
                                isLaunching = false
                                switch outcome {
                                case .launched:               relaunchNote = "Restarting \(capable.name) through Netal."
                                case .launcherMissing(let p): relaunchNote = "Launcher not found at \(p)"
                                case .failed(let why):        relaunchNote = why
                                }
                            }
                            }
                            confirmRestart = true
                        } label: {
                            Label("Restart with Netal", systemImage: "bolt.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                        .disabled(isLaunching)
                    }
                    if let relaunchNote {
                        Text(relaunchNote).font(.caption2).foregroundStyle(.tertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            // A Windows game in a bottle is invisible to the container scan, so without this the
            // panel sat idle while a game ran without Netal and said nothing. It cannot be attached
            // to while running - DYLD_INSERT_LIBRARIES applies at exec, and games are
            // hardened-runtime signed - so the offer is to start it again, not to attach.
            if let bottled = monitor.crossoverUnattached {
                VStack(alignment: .leading, spacing: 6) {
                    Notice(text: "\(bottled.name) is running without Netal.",
                           tint: .orange, symbol: "bolt.horizontal.circle.fill")
                    if GameLauncher.isCrossOverAvailable {
                        Button {
                            restartName = bottled.name
                            restartAction = {
                            isLaunching = true
                            relaunchNote = "Starting \(bottled.name) with Netal…"
                            GameLauncher.relaunchCrossOver(bottle: bottled.bottle, appID: bottled.appID,
                                                          executable: bottled.executable,
                                                          frameGeneration: monitor.fgModel.isReady) { outcome in
                                isLaunching = false
                                switch outcome {
                                case .launched:               relaunchNote = "Launched through Netal."
                                case .launcherMissing(let p): relaunchNote = "Launcher not found at \(p)"
                                case .failed(let why):        relaunchNote = why
                                }
                            }
                            }
                            confirmRestart = true
                        } label: {
                            Label("Restart with Netal", systemImage: "bolt.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                        .disabled(isLaunching)
                    }
                    Text(relaunchNote ?? "A running game cannot be attached to — it has to start with Netal.")
                        .font(.caption2).foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if monitor.model.state == .noBundle {
                Button {
                    withAnimation(.snappy(duration: 0.24)) { tab = .settings }
                } label: {
                    Notice(text: "Set up Netal — choose a prepared runtime folder.", tint: .orange,
                           symbol: "shippingbox", chevron: true)
                }
                .buttonStyle(.plain)
            }

            // A missing model blocks everything, and without this the user would only find out
            // from a failed arm after launching a game. Surfaced here because Status is where
            // people look first; the detail and the fix live in Settings.
            if !monitor.model.isReady, monitor.model.state != .noBundle {
                Button {
                    withAnimation(.snappy(duration: 0.24)) { tab = .dlss }
                } label: {
                    Notice(text: monitor.model.summary, tint: .orange,
                           symbol: "exclamationmark.circle.fill", chevron: true)
                }
                .buttonStyle(.plain)
            }

            // A profile missing layers is a ~2x slowdown that looks like a healthy session: the
            // fallback kernels are correct, just slower, so nothing fails and the only symptom is a
            // frame rate the user cannot explain. Say it outright, and offer the fix.
            if monitor.status.pipelineIncomplete,
               let present = monitor.status.pipelineLayersPresent,
               let total = monitor.status.pipelineLayersTotal {
                VStack(alignment: .leading, spacing: 6) {
                    Notice(text: "Profile incomplete: \(present) of \(total) layers. "
                                 + "The rest run slower fallback kernels.",
                           tint: .orange, symbol: "exclamationmark.triangle.fill")

                    switch profileCompletion.phase {
                    case .idle:
                        if let resolution = monitor.status.outputResolution, profileCompletion.isAvailable {
                            Button("Generate the missing layers") {
                                profileCompletion.complete(resolution: resolution)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }
                    case .running(let step):
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text(step).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                        }
                    case .done(let message):
                        Text(message).font(.caption2).foregroundStyle(.green)
                    case .failed(let reason):
                        Text(reason).font(.caption2).foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            if monitor.isLive, let p = monitor.status.effectivePacing {
                // The three numbers worth a glance, large enough to read without hunting.
                HStack(spacing: 0) {
                    // With frame generation on, the runtime's rate is the game's rendered frames; the
                    // presented rate is double that and is the game's to report, so say which this is.
                    Stat(value: String(format: "%.0f", p.fps), unit: "fps",
                         label: "Source rate")
                    Divider().frame(height: 26)
                    Stat(value: String(format: "%.1f", p.meanMs), unit: "ms", label: "Frame time")
                    Divider().frame(height: 26)
                    // The Metal 4 path reports gpu_ms=0.00 because MTL4CommandBuffer has no
                    // completion handler. Rendering that as "0.0 ms" claims the pass is free, which
                    // is the opposite of true - it is unmeasured.
                    if let fg = monitor.status.frameGeneration, fg.isServing, !fg.paused {
                        // Netal is generating every other presented frame; the count is the honest
                        // number (the presented rate is the game's to report, not ours).
                        Stat(value: fg.generated.map(Display.compact) ?? "—", unit: "frames", label: "Generated")
                    } else {
                        Stat(value: p.gpuMs.flatMap { $0 > 0 ? String(format: "%.1f", $0) : nil } ?? "—",
                             unit: "ms", label: "SR cost")
                    }
                }
            } else {
                IdleState(monitor: monitor)
            }

            VStack(spacing: 5) {
                // Output and render used to be rows here as well as in the header. The header
                // carries them; repeating them was most of why this view scrolled.
                if monitor.status.framesScheduled > 0 {
                    DetailRow("Frames", monitor.status.framesScheduled.formatted())
                }
                // Health counters appear only when non-zero, except flashes, which are worth
                // stating as a zero: a clean run is the result we are looking for.
                if monitor.status.framesScheduled > 0 {
                    DetailRow("Flashes", monitor.status.flashes.formatted(),
                              tint: monitor.status.flashes > 0 ? .orange : .secondary)
                }
                if monitor.status.fallbacks > 0 {
                    DetailRow("Fallbacks", monitor.status.fallbacks.formatted(), tint: .orange)
                }
            }

            if monitor.isLive, monitor.status.effectivePacing != nil {
                Divider()
                MarkControl(monitor: monitor)
            }
            Divider()

            // What is on, in one glance, with the switches one tap away. The controls themselves
            // live on the DLSS tab beside the feature they govern; a lone "Enabled" switch here used
            // to mean Super Resolution without saying so, and had no answer for Frame Generation.
            FeatureSummary(monitor: monitor, tab: $tab)
        }
        .confirmationDialog("Save your game before restarting", isPresented: $confirmRestart, titleVisibility: .visible) {
            Button("Restart \(restartName)", role: .destructive) {
                restartAction?()
                restartAction = nil
            }
            Button("Cancel", role: .cancel) { restartAction = nil }
        } message: {
            Text("Netal will ask \(restartName) to quit and start again. Unsaved progress may be lost. Save and leave any multiplayer session before continuing. Launching with Netal does not establish SR or FG compatibility.")
        }
    }
}

/// One line per live feature: name, state, and a chevron to the DLSS tab where its switch is.
struct FeatureSummary: View {
    @ObservedObject var monitor: RuntimeMonitor
    @Binding var tab: PanelTab

    var body: some View {
        Button {
            withAnimation(.snappy(duration: 0.24)) { tab = .dlss }
        } label: {
            HStack(spacing: 10) {
                pill("Super Resolution", on: srOn, state: srState)
                pill("Frame Generation", on: fgOn, state: fgState)
                Spacer(minLength: 2)
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Feature switches are on the DLSS tab")
    }

    private var srOn: Bool { !monitor.isPauseRequested && monitor.logFileExists && monitor.status.superResolutionServing && monitor.isLive }
    private var srState: String {
        if !monitor.isLive { return "No live session" }
        if monitor.isPauseRequested { return "Off" }
        if monitor.status.superResolutionServing && monitor.isLive { return "On" }
        return "Ready"
    }
    private var fgOn: Bool { monitor.isLive && (monitor.status.frameGeneration?.isServing ?? false) && !(monitor.status.frameGeneration?.paused ?? false) }
    private var fgState: String {
        guard monitor.isLive else { return "No live session" }
        guard let fg = monitor.status.frameGeneration else { return "Off" }
        if fg.paused { return "Off" }
        if fg.isServing { return "On" }
        return fg.armed ? "Ready" : "MetalFX"
    }

    private func pill(_ name: String, on: Bool, state: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            HStack(spacing: 4) {
                Circle().fill(on ? Color.green : Color.secondary.opacity(0.35)).frame(width: 6, height: 6)
                Text(state).font(.caption.weight(.medium)).foregroundStyle(on ? Color.primary : Color.secondary)
                    .contentTransition(.opacity)
            }
        }
    }
}

/// Catching a rare artifact needs a "that just happened" signal: a fixed dump window cannot
/// anticipate one. Pressing this stamps the frame the runtime is on into the log.
struct MarkControl: View {
    @ObservedObject var monitor: RuntimeMonitor

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { monitor.markArtifact() } label: {
                Label("Mark artifact", systemImage: "flag.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(!monitor.logFileExists)
            .keyboardShortcut("m", modifiers: [.command])

            Text(caption)
                .font(.caption2).foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
                .contentTransition(.opacity)
        }
    }

    private var caption: String {
        if let feedback = monitor.markFeedback { return "\(feedback) · waiting for the runtime" }
        if monitor.marksRecorded > 0 {
            if let frame = monitor.status.lastMarkFrame {
                return "\(monitor.marksRecorded) marked · last at frame \(frame)"
            }
            return "\(monitor.marksRecorded) marked this session"
        }
        return "Press when you see one."
    }
}

// MARK: - Graphs

struct GraphsView: View {
    @ObservedObject var monitor: RuntimeMonitor

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if monitor.history.isEmpty {
                Text("Graphs fill once the runtime reports frames.")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 120, alignment: .center)
                    .multilineTextAlignment(.center)
            } else {
                let samples = monitor.history.samples
                ChartCard(title: "Frame time",
                          value: String(format: "%.1f ms", samples.last?.meanMs ?? 0),
                          caption: "band = spread per window · dashes = 60 Hz") {
                    FrameTimeChart(samples: samples, ceiling: monitor.history.frameTimeCeiling)
                }
                let gpu = samples.compactMap(\.gpuMs).filter { $0 > 0 }
                ChartCard(title: "SR cost",
                          value: samples.last?.gpuMs.flatMap { $0 > 0 ? String(format: "%.1f ms", $0) : nil } ?? "—",
                          caption: gpu.isEmpty ? "this runtime build does not report it yet" : nil,
                          height: 34) {
                    // An empty series draws nothing, which reads as a bug rather than as absence.
                    if gpu.isEmpty {
                        RoundedRectangle(cornerRadius: 4).fill(.quaternary.opacity(0.5))
                    } else {
                        Sparkline(values: gpu, ceiling: max(4, (gpu.max() ?? 4) * 1.2), tint: .purple)
                    }
                }
                ChartCard(title: "Memory",
                          value: samples.last?.rssMB.map { String(format: "%.1f GB", $0 / 1024) } ?? "—",
                          height: 34) {
                    Sparkline(values: samples.compactMap(\.rssMB),
                              ceiling: monitor.history.memoryCeiling, tint: .teal)
                }
                if let p = monitor.status.effectivePacing, p.spikes > 0 {
                    Text("\(p.spikes.formatted()) frames over 25 ms · worst \(String(format: "%.0f", p.worstMs)) ms")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                // The difference between "Netal got slower" and "the machine got slower". Shown
                // only when it is not nominal, because a healthy machine needs no explanation.
                if let thermal = monitor.status.effectivePacing?.thermal, thermal != "nominal" {
                    Notice(text: "The machine is thermally \(thermal) — frame times will suffer "
                               + "regardless of Netal.",
                           tint: thermal == "fair" ? .yellow : .orange, symbol: "thermometer.high")
                }
            }
        }
    }
}

// MARK: - DLSS

/// What DLSS can do, and what Netal has been given to do it with.
///
/// Split out from Settings once the model section landed: Settings had become a long page mixing
/// two different questions - what this runtime *is*, and how you would like it to behave. This tab
/// answers the first, and it is where a new user is sent when they have not supplied a model, so it
/// needs to be somewhere you can be pointed at rather than a section to scroll to.
struct DLSSView: View {
    @ObservedObject var monitor: RuntimeMonitor

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            FeatureRoster(monitor: monitor)
            Divider()
            ModelSection(monitor: monitor)
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @ObservedObject var monitor: RuntimeMonitor
    @State private var hudEnabled = MetalHUD.isEnabled
    @State private var hudRefused = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            RuntimeSetupView(monitor: monitor)
            Divider()

            VStack(alignment: .leading, spacing: 4) {
                SettingRow("Metal performance HUD") {
                    Toggle("", isOn: $hudEnabled)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .onChange(of: hudEnabled) { _, want in
                            hudRefused = !MetalHUD.setEnabled(want)
                            if hudRefused { hudEnabled = MetalHUD.isEnabled }
                        }
                }
                Text(hudRefused
                     ? "The system refused the change — a managed preference may own it."
                     : "Apple's overlay, in every Metal app. Netal turns it off when it quits.")
                    .font(.caption2)
                    .foregroundStyle(hudRefused ? Color.red : Color.tertiaryLabel)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                Text("Runtime log").font(.caption).foregroundStyle(.secondary)
                Text(monitor.logURL.path)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1).truncationMode(.head)
                    .help(monitor.logURL.path)
                    .textSelection(.enabled)
                if let game = monitor.game {
                    Text(game.isRunning ? "\(game.name) is running" : "\(game.name) is not running")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                Text("Follows the game launched through Netal.")
                    .font(.caption2).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Divider()

            HStack {
                Button("Reveal Log") {
                    NSWorkspace.shared.selectFile(monitor.logURL.path,
                                                  inFileViewerRootedAtPath:
                                                    monitor.logURL.deletingLastPathComponent().path)
                }
                .disabled(!monitor.logFileExists)
                Spacer()
                Button("Quit Netal") { NSApplication.shared.terminate(nil) }
            }
            .buttonStyle(.link).font(.callout)
        }
        .onAppear { hudEnabled = MetalHUD.isEnabled }
    }
}

/// What DLSS can do, and where Netal is against it.
///
/// Sits at the top of Settings rather than in a tab of its own: it will become a control surface
/// when Ray Reconstruction lands, so it belongs with the controls - but it changes about once a
/// year, which does not earn a permanent slot in the tab bar or space on the Status view that live
/// metrics want.
///
/// Super Resolution's row shows the runtime's actual state, not a ticked box that cannot be
/// unticked. Switching it off is not a feature choice - it is the runtime not running - and that
/// switch already exists on the Status view, so the caption points there instead of leaving a dead
/// control to be poked at.
struct FeatureRoster: View {
    @ObservedObject var monitor: RuntimeMonitor
    @State private var showSROptions = false
    private var features: [DLSSFeature] {
        // Preserve the feature order in the user-approved running-app screenshot.
        let order = ["sr": 0, "rr": 1, "fg": 2, "nr": 3]
        return DLSSFeature.all(chip: AppleChip.current, frameGeneration: monitor.status.frameGeneration)
            .sorted { order[$0.id, default: 4] < order[$1.id, default: 4] }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("DLSS features").font(.caption).foregroundStyle(.secondary)

            ForEach(features) { feature in
                HStack(alignment: .center, spacing: 8) {
                    Image(systemName: symbol(for: feature))
                        .font(.system(size: 11))
                        .foregroundStyle(tint(for: feature))
                        .frame(width: 13)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(feature.name)
                            .font(.caption)
                            .foregroundStyle(isBuilt(feature) ? Color.primary : Color.secondary)
                        Text(detail(for: feature))
                            .font(.caption2).foregroundStyle(detailTint(for: feature))
                            .lineLimit(1).truncationMode(.tail)
                            .contentTransition(.opacity)
                    }
                    Spacer(minLength: 6)
                    control(for: feature)
                }
                .help(help(for: feature))
            }

            // Super Resolution's lowering controls, folded away: they change how the network is
            // lowered and cost a shader rebuild, so they are not something to brush past on the way
            // to a switch.
            if monitor.logFileExists {
                DisclosureGroup(isExpanded: $showSROptions) {
                    VStack(alignment: .leading, spacing: 8) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Arithmetic mode").font(.caption)
                            Picker("", selection: Binding(
                                get: { monitor.lowering },
                                set: { monitor.setLowering($0) }
                            )) {
                                ForEach(RuntimeMonitor.Lowering.allCases) { Text($0.label).tag($0) }
                            }
                            .labelsHidden()
                            .pickerStyle(.segmented)
                            .controlSize(.small)
                            .frame(maxWidth: .infinity)
                        }
                        SettingRow("Batched pipelines") {
                            Toggle("", isOn: Binding(
                                get: { monitor.batchedPipelines },
                                set: { monitor.setBatchedPipelines($0) }
                            ))
                            .labelsHidden().toggleStyle(.switch).controlSize(.small)
                        }
                        ControlNote(monitor: monitor)
                    }
                    .padding(.top, 6)
                } label: {
                    Text("Super Resolution options").font(.caption).foregroundStyle(.secondary)
                }
                .padding(.leading, 21)
            } else {
                Text("Start a game through Netal to enable the switches.")
                    .font(.caption2).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - State

    private var srPaused: Bool { monitor.isPauseRequested }
    /// Super Resolution's own activity, not the session's: with frame generation serving, the
    /// session is active while SR may have nothing to upscale (DLAA, no profile for the size).
    private var srRunning: Bool { monitor.status.superResolutionServing && monitor.isLive && !monitor.isPauseRequested }
    private var fg: FrameGeneration? { monitor.status.frameGeneration }
    private var fgOn: Bool { monitor.frameGenerationEnabled && !(fg?.paused ?? false) }
    private var fgServing: Bool { monitor.isLive && (fg?.isServing ?? false) && !(fg?.paused ?? false) }
    /// The FG switch does something only once the bridge armed the session (`--fg`).
    private var fgArmed: Bool { fg?.armed ?? false }

    private func isBuilt(_ feature: DLSSFeature) -> Bool {
        switch feature.availability {
        case .live, .observed, .available: return true
        case .planned, .experimental: return false
        }
    }

    @ViewBuilder
    private func control(for feature: DLSSFeature) -> some View {
        switch feature.id {
        case "sr":
            Toggle("", isOn: Binding(get: { monitor.isLive && !monitor.isPauseRequested }, set: { monitor.setEnabled($0) }))
                .labelsHidden().toggleStyle(.switch).controlSize(.small)
                .disabled(!monitor.logFileExists || !monitor.isLive)
                .accessibilityLabel("Super Resolution")
        case "fg":
            Toggle("", isOn: Binding(get: { monitor.isLive && fgOn }, set: { monitor.setFrameGenerationEnabled($0) }))
                .labelsHidden().toggleStyle(.switch).controlSize(.small)
                .disabled(!monitor.logFileExists || !monitor.isLive || !fgArmed)
                .accessibilityLabel("Frame Generation")
        default:
            Text(feature.id == "nr" ? "Experimental" : "Planned")
                .font(.caption2.weight(.medium)).foregroundStyle(Color.tertiaryLabel)
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(Capsule().fill(.quaternary.opacity(0.5)))
        }
    }

    private func symbol(for feature: DLSSFeature) -> String {
        if !monitor.isLive && (feature.id == "sr" || feature.id == "fg") { return "circle" }
        switch feature.id {
        case "sr":
            if !monitor.logFileExists { return "circle" }
            return srPaused ? "pause.circle.fill" : (srRunning ? "checkmark.circle.fill" : "circle.lefthalf.filled")
        case "fg":
            guard let fg else { return "circle" }
            if fg.paused || !monitor.frameGenerationEnabled { return "pause.circle.fill" }
            if fg.isServing { return "checkmark.circle.fill" }
            return fg.armed ? "circle.lefthalf.filled" : "eye.circle"
        default:
            if case .experimental = feature.availability { return "flask" }
            return "circle.dashed"
        }
    }

    private func tint(for feature: DLSSFeature) -> Color {
        if !monitor.isLive && (feature.id == "sr" || feature.id == "fg") { return .secondary }
        switch feature.id {
        case "sr":
            if !monitor.logFileExists || srPaused { return .secondary }
            return srRunning ? .green : .secondary
        case "fg":
            guard let fg else { return .tertiaryLabel }
            if fg.paused || !monitor.frameGenerationEnabled { return .secondary }
            if fg.isServing { return .green }
            return fg.armed ? .secondary : .orange
        default:
            return .tertiaryLabel
        }
    }

    /// The second line says the state first, then what the feature does or is doing.
    private func detail(for feature: DLSSFeature) -> String {
        if !monitor.isLive && (feature.id == "sr" || feature.id == "fg") { return "No live session" }
        switch feature.id {
        case "sr":
            if !monitor.logFileExists { return "Upscales the rendered frame" }
            if srPaused { return "Off · MetalFX is upscaling" }
            if srRunning {
                if let r = monitor.status.renderResolution, let o = monitor.status.outputResolution {
                    return r == o ? "Active · DLAA at \(Display.res(o))" : "Active · \(Display.res(r)) → \(Display.res(o))"
                }
                return "Active · upscaling"
            }
            return monitor.status.outputResolution == nil ? "Ready · the game is not upscaling" : "Ready · waiting for frames"
        case "fg":
            guard let fg else { return monitor.logFileExists ? "Off · not armed for this session" : "Interpolates between frames" }
            if fg.paused || !monitor.frameGenerationEnabled { return "Off · MetalFX is interpolating" }
            return DLSSFeature.frameGenerationDetail(fg)
        default:
            return feature.detail
        }
    }

    private func detailTint(for feature: DLSSFeature) -> Color {
        if feature.id == "fg", let fg, !fg.armed, !fg.paused, monitor.frameGenerationEnabled, monitor.logFileExists {
            return .orange   // observed only: MetalFX is doing the work, not Netal
        }
        return .tertiaryLabel
    }

    private func help(for feature: DLSSFeature) -> String {
        switch feature.availability {
        case .live:
            if feature.id == "fg" {
                return DLSSFeature.frameGenerationHelp(fg)
            }
            return "Super Resolution — Netal upscales through the game's MetalFX temporal scaler. "
                 + "Switching it off hands frames back to MetalFX without restarting the game."
        case .available: return "Frame Generation needs a prepared FG bundle and a compatible game. The current native profile is 640×360 internally."
        case .observed(let note): return "\(feature.name) — \(note) " + DLSSFeature.frameGenerationHelp(fg)
        case .planned:                  return "\(feature.name) — designed, not built yet"
        case .experimental(let note): return "\(feature.name) — \(note)"
        }
    }
}

/// Supported SR model identities and FG model/runtime integrity are verified separately.
/// The SR importer requires its local helpers; FG accepts an assembled local bundle.
struct ModelSection: View {
    @ObservedObject var monitor: RuntimeMonitor
    @StateObject private var ingest = ModelIngest()

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Models").font(.caption).foregroundStyle(.secondary)

            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Image(systemName: symbol)
                    .font(.system(size: 11)).foregroundStyle(tint).frame(width: 13)
                VStack(alignment: .leading, spacing: 0) {
                    Text(title).font(.caption)
                        .foregroundStyle(monitor.model.isReady ? Color.primary : Color.secondary)
                        .lineLimit(1).truncationMode(.middle)
                    Text(detail).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                }
                Spacer(minLength: 6)
                Text(badge)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(monitor.model.isReady ? tint : Color.tertiaryLabel)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(.quaternary.opacity(monitor.model.isReady ? 0 : 0.5)))
            }
            .help([monitor.model.name, monitor.model.summary].compactMap { $0 }.joined(separator: "\n"))

            // Verify supported model/math identities and the complete recorded inventory before
            // presenting a green state or requesting FG during launch.
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Image(systemName: monitor.fgModel.isReady ? "checkmark.seal.fill" : "circle.dashed")
                    .font(.system(size: 11))
                    .foregroundStyle(monitor.fgModel.isReady ? Color.green : Color.tertiaryLabel)
                    .frame(width: 13)
                VStack(alignment: .leading, spacing: 0) {
                    Text("Frame generation model").font(.caption)
                        .foregroundStyle(monitor.fgModel.isReady ? Color.primary : Color.secondary)
                    Text(monitor.fgModel.summary)
                        .font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                }
                Spacer(minLength: 6)
                Text(monitor.fgModel.badge)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(monitor.fgModel.isReady ? Color.green : Color.tertiaryLabel)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(.quaternary.opacity(monitor.fgModel.isReady ? 0 : 0.5)))
            }
            .help(monitor.fgModel.bundleURL?.path ?? "The FG runtime bundle was not found beside the log or at the build default.")

            // SR import reports the installer's model-byte validation result. It does not run
            // an RTX comparison or certify gameplay.
            VStack(alignment: .leading, spacing: 5) {
                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    Image(systemName: ingestSymbol)
                        .font(.system(size: 11)).foregroundStyle(ingestTint).frame(width: 13)
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Add from nvngx_dlss.dll").font(.caption).foregroundStyle(.secondary)
                        Text(ingestDetail).font(.caption2).foregroundStyle(.tertiary)
                            .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 6)
                    if ingest.isAvailable {
                        Button(ingest.isRunning ? "Working…" : "Choose…") { ingest.choose() }
                            .buttonStyle(.bordered).controlSize(.small)
                            .disabled(ingest.isRunning)
                    } else {
                        Text("Unavailable")
                            .font(.caption2.weight(.medium)).foregroundStyle(Color.tertiaryLabel)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Capsule().fill(.quaternary.opacity(0.5)))
                    }
                }
            }

            if let bundle = monitor.model.bundleURL {
                Button("Reveal Model Folder") {
                    let folder = bundle.appendingPathComponent("model")
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folder.path)
                }
                .buttonStyle(.link).font(.caption)
            }
        }
    }

    private var symbol: String {
        switch monitor.model.state {
        case .verified: return "checkmark.seal.fill"
        case .modified: return "exclamationmark.triangle.fill"
        case .missing:  return "xmark.circle"
        case .unverified: return "exclamationmark.shield"
        case .noBundle: return "questionmark.circle"
        }
    }

    private var tint: Color {
        switch monitor.model.state {
        case .verified: return .green
        case .modified: return .orange
        default:        return .secondary
        }
    }

    private var title: String {
        monitor.model.isReady && monitor.model.name == ModelInspector.supportedName
            ? "DLSS 4 · Preset K" : "Super resolution model"
    }

    private var detail: String {
        switch monitor.model.state {
        case .verified:
            let mb = Double(monitor.model.weightsBytes ?? 0) / 1_048_576
            return String(format: "Supported model verified · %.1f MB", mb)
        case .modified(let files):
            return files.joined(separator: ", ") + " does not match"
        case .missing:  return "Supply weights.bin and constants.bin"
        case .unverified: return "SR model could not be verified"
        case .noBundle: return "Choose a prepared runtime folder in Settings"
        }
    }

    private var ingestSymbol: String {
        switch ingest.phase {
        case .running:   return "arrow.triangle.2.circlepath"
        case .installed: return "checkmark.seal.fill"
        case .refused:   return "exclamationmark.triangle.fill"
        case .idle:      return "circle.dashed"
        }
    }

    private var ingestTint: Color {
        switch ingest.phase {
        case .installed: return .green
        case .refused:   return .orange
        default:         return .secondary
        }
    }

    /// Says what is happening, or - at rest - the one thing a new user has to understand.
    private var ingestDetail: String {
        switch ingest.phase {
        case .running(let stage):  return stage
        case .installed(let note): return note
        case .refused(let why):    return why
        case .idle:
            return ingest.isAvailable ? "Weights are not bundled — supply your own"
                                      : (ingest.unavailabilityReason ?? "Complete runtime setup first")
        }
    }

    private var badge: String {
        switch monitor.model.state {
        case .verified: return "Verified"
        case .modified: return "Modified"
        case .missing:  return "Missing"
        case .unverified: return "Unverified"
        case .noBundle: return "Unknown"
        }
    }
}

// MARK: - About

/// The character and its ground shadow at one instant.
///
/// Split out from AboutView so a pose can be rendered directly. The animation is the interesting
/// part and a still cannot show it, so the render harness draws the whole sequence as a filmstrip
/// instead - which is the only way to review timing-driven work in the same loop as the code.
struct NetalSprite: View {
    var lift: CGFloat
    var squash: CGSize
    var tucked: Bool
    var height: CGFloat = NetalSprite.standardHeight
    /// What the eyes should be saying. Idle by default, so a still render is not lit.
    var vitals: NetalVitals = NetalVitals()
    @State private var pulse = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Sized to sit in the panel rather than dominate it: the About view is a caption with a mark
    /// above it, not a poster.
    static let standardHeight: CGFloat = 74
    /// Every other measurement here was tuned against a 100pt character, so they scale with it and
    /// the proportions hold at any size.
    private static let referenceHeight: CGFloat = 100
    static var apex: CGFloat { 18 * standardHeight / referenceHeight }
    /// Room for the jump and the shadow, so neither clips.
    static var containerHeight: CGFloat { standardHeight + apex + 12 }

    private var unit: CGFloat { height / Self.referenceHeight }
    /// 0 on the ground, 1 at the apex. The shadow reads this every frame rather than being keyed
    /// to the hop's phases, so it tracks the real height instead of stepping between two states.
    private var altitude: CGFloat { min(1, max(0, -lift / (18 * unit))) }

    var body: some View {
        ZStack(alignment: .bottom) {
            Ellipse()
                .fill(.black.opacity(0.26 - 0.14 * altitude))
                .frame(width: (94 + 18 * altitude) * unit, height: (13 - 2 * altitude) * unit)
                .blur(radius: (5 + 6.5 * altitude) * unit)
                .offset(y: 7 * unit)

            if let frame = tucked ? NetalArt.hop : NetalArt.rest {
                Image(nsImage: frame)
                    .resizable().scaledToFit()
                    .frame(height: height)
                    // Inside the image's own frame, so the glow lands on the eyes at any size and
                    // rides the squash and the lift with them rather than sliding around.
                    .overlay { eyeGlow }
                    .scaleEffect(x: squash.width, y: squash.height, anchor: .bottom)
                    .offset(y: lift)
            } else if let mark = NetalArt.mark {
                Image(nsImage: mark).resizable().scaledToFit().frame(height: height)
            }
        }
    }

    /// Two soft lights at the measured eye positions, added over the artwork.
    ///
    /// `plusLighter` rather than a plain overlay because an eye is emissive: light should add to
    /// what is painted there, not replace it. At zero intensity this draws nothing at all, so an
    /// idle Netal looks exactly as it did before any of this existed.
    @ViewBuilder private var eyeGlow: some View {
        if vitals.intensity > 0, !NetalArt.eyes.isEmpty {
            GeometryReader { geometry in
                // scaledToFit letterboxes the image inside whatever frame it is given, so the
                // overlay's geometry is NOT the drawn image: positioning against it puts the glow
                // off the eyes by however much slack there is. Work out the fitted rect instead.
                let canvas = (tucked ? NetalArt.hop : NetalArt.rest)?.size
                    ?? CGSize(width: 1, height: 1)
                let fit = min(geometry.size.width / canvas.width, geometry.size.height / canvas.height)
                let drawn = CGSize(width: canvas.width * fit, height: canvas.height * fit)
                let originX = (geometry.size.width - drawn.width) / 2
                let originY = (geometry.size.height - drawn.height) / 2
                ZStack {
                    ForEach(Array(NetalArt.eyes.enumerated()), id: \.offset) { _, eye in
                        let side = drawn.width
                        // Wide, and much wider than the eye. The eyes are painted near-white, so
                        // light added on top of them only saturates to white and no colour reads.
                        // The colour has to land on the grey body around them, which means a halo
                        // rather than a highlight.
                        let diameter = side * eye.r * 10
                        Circle()
                            .fill(RadialGradient(
                                // Bright at the eye and falling away quickly: a slow falloff
                                // reads as a smear across the body rather than as light leaving
                                // two points.
                                stops: [.init(color: tint.opacity(0.90), location: 0),
                                        .init(color: tint.opacity(0.45), location: 0.22),
                                        .init(color: tint.opacity(0.10), location: 0.55),
                                        .init(color: tint.opacity(0), location: 1)],
                                center: .center, startRadius: 0, endRadius: diameter / 2))
                            .frame(width: diameter, height: diameter)
                            .position(x: originX + drawn.width * eye.x,
                                      y: originY + drawn.height * eye.y)
                    }
                }
                .blendMode(.plusLighter)
                .opacity(vitals.intensity * (reduceMotion ? 1 : (pulse ? 1.0 : 0.72)))
                .animation(reduceMotion ? nil : vitals.period.map { .easeInOut(duration: $0 / 2).repeatForever(autoreverses: true) }, value: pulse)
            }
            .allowsHitTesting(false)
            .onAppear { pulse = !reduceMotion && !CommandLine.arguments.contains("--render") && vitals.period != nil }
            .onChange(of: vitals.period) { _, period in pulse = !reduceMotion && period != nil }
            .onChange(of: reduceMotion) { _, reduced in pulse = !reduced && vitals.period != nil }
        }
    }

    private var tint: Color { Color(vitals) }

    /// The poses the hop passes through, in order, for the filmstrip.
    static let storyboard: [(String, CGFloat, CGSize, Bool)] = [
        ("rest",         0,    CGSize(width: 1.00, height: 1.00), false),
        ("anticipate",   0,    CGSize(width: 1.09, height: 0.90), false),
        ("take-off",    -apex, CGSize(width: 0.93, height: 1.11), true),
        ("settle",      -apex * 0.83, CGSize(width: 1.00, height: 1.00), true),
        ("float",       -apex * 0.97, CGSize(width: 1.00, height: 1.00), true),
        ("fall",        -apex * 0.22, CGSize(width: 0.96, height: 1.06), true),
        ("impact",       0,    CGSize(width: 1.13, height: 0.87), false),
        ("recover",      0,    CGSize(width: 1.00, height: 1.00), false),
    ]
}

struct AboutView: View {
    /// Live state for the eyes. Idle by default so the render harness is not lit.
    var vitals: NetalVitals = NetalVitals()
    /// Height above the ground line, in points. Negative is up.
    @State private var lift: CGFloat = 0
    /// Non-uniform scale, anchored at the feet: squash and stretch.
    @State private var squash = CGSize(width: 1, height: 1)
    /// Which of the two poses is showing.
    @State private var tucked = false
    @State private var motion: Task<Void, Never>?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private static let apex: CGFloat = NetalSprite.apex

    /// A hop, not a cross-dissolve and not a slide.
    ///
    /// The first version swapped the pose instantly and sprang the offset, and it read as a
    /// slideshow: two still images alternating, because nothing about the motion explained why the
    /// character changed shape. What sells a hop is the part either side of the jump.
    ///
    ///   anticipation  it squashes toward the ground first, feet still planted. A jump with no
    ///                 wind-up reads as teleporting.
    ///   take-off      the pose swaps HERE, during the fastest upward motion, where the eye cannot
    ///                 resolve it - the feet tuck because it is leaving the ground, and the body
    ///                 stretches along the direction of travel.
    ///   settle        scale returns to neutral just past the apex, slightly under-damped, and then
    ///                 the float begins so it is hovering rather than frozen mid-air.
    ///   fall          on exit it accelerates down, still tucked, stretched again.
    ///   landing       the pose swaps back at contact, under an impact squash that absorbs it, then
    ///                 springs out.
    ///
    /// The shadow is not keyed to any of that: it is computed from `lift` every frame, so it tracks
    /// the real height continuously instead of stepping between two states.
    init(lifted: Bool = false, vitals: NetalVitals = NetalVitals()) {
        self.vitals = vitals
        _lift = State(initialValue: lifted ? -NetalSprite.apex * 0.83 : 0)
        _tucked = State(initialValue: lifted)
    }

    var body: some View {
        VStack(spacing: 12) {
            NetalSprite(lift: lift, squash: squash, tucked: tucked, vitals: vitals)
            .frame(height: NetalSprite.containerHeight, alignment: .bottom)
            .contentShape(Rectangle())
            .accessibilityLabel("Netal, the blob")
            .onHover { inside in
                motion?.cancel()
                guard !reduceMotion else { return }
                motion = Task { @MainActor in inside ? await takeOff() : await land() }
            }

            VStack(spacing: 2) {
                Text("Netal").font(.title3.weight(.semibold))
                Text("Super resolution and frame generation on Apple silicon.")
                    .font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text("Source preview · NR experimental")
                .font(.caption2).foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)

            Divider().padding(.horizontal, 40).padding(.top, 2)

            VStack(spacing: 1) {
                Text("Built by Nicolas Schlaepfer")
                    .font(.caption2).foregroundStyle(.secondary)
                Link("Source and issue tracker", destination: URL(string: "https://github.com/nschlaepfer/Netal")!)
                    .font(.caption2)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 2)
        .onDisappear { motion?.cancel() }
        .onChange(of: reduceMotion) { _, reduced in
            if reduced { motion?.cancel(); lift = 0; squash = CGSize(width: 1, height: 1); tucked = false }
        }
    }

    @MainActor private func takeOff() async {
        withAnimation(.easeOut(duration: 0.09)) {
            squash = CGSize(width: 1.09, height: 0.90)          // wind up, feet planted
        }
        guard await pause(0.09) else { return }

        tucked = true                                            // swap under the launch
        withAnimation(.timingCurve(0.15, 0.85, 0.3, 1, duration: 0.24)) {
            lift = -Self.apex
            squash = CGSize(width: 0.93, height: 1.11)           // stretch along travel
        }
        guard await pause(0.20) else { return }

        withAnimation(.spring(response: 0.40, dampingFraction: 0.52)) {
            squash = CGSize(width: 1, height: 1)
            lift = -Self.apex * 0.83
        }
        guard await pause(0.34) else { return }

        // Floating, not parked: a slow bob so the held state still has life in it.
        withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) {
            lift = -Self.apex * 0.97
        }
    }

    @MainActor private func land() async {
        withAnimation(.easeIn(duration: 0.17)) {
            lift = 0
            squash = CGSize(width: 0.96, height: 1.06)           // stretch into the fall
        }
        guard await pause(0.165) else { return }

        tucked = false                                           // feet back out, at contact
        withAnimation(.easeOut(duration: 0.07)) {
            squash = CGSize(width: 1.13, height: 0.87)           // impact absorbs it
        }
        guard await pause(0.07) else { return }

        withAnimation(.spring(response: 0.38, dampingFraction: 0.48)) {
            squash = CGSize(width: 1, height: 1)
        }
    }

    /// Sleep, reporting whether the sequence still owns the animation. A hover that reverses
    /// mid-hop cancels this task, and the replacement sets every value itself, so an interrupted
    /// sequence simply stops rather than fighting the new one.
    private func pause(_ seconds: Double) async -> Bool {
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        return !Task.isCancelled
    }
}

// MARK: - Small pieces

/// Small presentation helpers. `1280x720` is a filename; `1280×720` is a resolution.
enum Display {
    static func res(_ value: String) -> String { value.replacingOccurrences(of: "x", with: "×") }
    /// Counts that keep growing for the life of a session: 999 → "999", 12,345 → "12.3k", 1,234,567 → "1.2M".
    static func compact(_ value: UInt64) -> String {
        if value < 10_000 { return value.formatted() }
        if value < 1_000_000 { return String(format: "%.1fk", Double(value) / 1_000) }
        return String(format: "%.1fM", Double(value) / 1_000_000)
    }
}

struct Stat: View {
    let value: String, unit: String, label: String
    var body: some View {
        VStack(spacing: 1) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.title3.monospacedDigit().weight(.medium))
                    .contentTransition(.numericText())
                Text(unit).font(.caption2).foregroundStyle(.secondary)
            }
            Text(label).font(.caption2).foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct DetailRow: View {
    let label: String, value: String
    var tint: Color = .secondary
    init(_ label: String, _ value: String, tint: Color = .secondary) {
        self.label = label; self.value = value; self.tint = tint
    }
    var body: some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospacedDigit())
                .foregroundStyle(tint == .secondary ? Color.primary : tint)
                .contentTransition(.numericText())
        }
    }
}

struct Notice: View {
    let text: String, tint: Color, symbol: String
    /// Shown when the notice is a button. Without it a tappable row looks like a label.
    var chevron = false

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: symbol).foregroundStyle(tint).font(.caption)
            Text(text).font(.caption).foregroundStyle(tint)
                .fixedSize(horizontal: false, vertical: true)
            if chevron {
                Spacer(minLength: 2)
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(tint.opacity(0.7))
            }
        }
        .contentShape(Rectangle())
    }
}

extension Color {
    static var tertiaryLabel: Color { Color(nsColor: .tertiaryLabelColor) }
}


/// The single caption under the controls: whichever fact matters most in the current state - a
/// warning while a slow path is selected, otherwise what the next change will cost.
struct ControlNote: View {
    @ObservedObject var monitor: RuntimeMonitor

    private var note: (String, Color) {
        if !monitor.logFileExists {
            return ("Start a game through Netal to enable these.", .secondary)
        }
        if monitor.lowering == .reference {
            return ("Reference mode disables acceleration and may be too slow for play.", .orange)
        }
        if !monitor.batchedPipelines {
            return ("Unbatched fast kernels. Slower; kept for comparison.", .orange)
        }
        if monitor.isPauseRequested { return ("MetalFX is handling frames.", .secondary) }
        if monitor.status.effectivePacing == nil { return ("Applies to the next game you start.", .secondary) }
        return ("Fast mode can differ numerically from the reference. Changes rebuild the session.", .secondary)
    }

    var body: some View {
        let (text, tint) = note
        Text(text)
            .font(.caption2)
            .foregroundStyle(tint == .secondary ? Color.tertiaryLabel : tint)
            .fixedSize(horizontal: false, vertical: true)
            .animation(.snappy(duration: 0.2), value: text)
    }
}


/// Label left, control hard against the right edge - the settings-pane convention. Controls that
/// hug their labels sit at as many different x positions as there are labels, which is what made
/// three switches read as three unrelated widgets instead of one group.
struct SettingRow<Content: View>: View {
    let label: String
    @ViewBuilder var content: Content
    init(_ label: String, @ViewBuilder content: () -> Content) {
        self.label = label; self.content = content()
    }
    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.callout)
            Spacer(minLength: 8)
            content
        }
    }
}


/// What the panel shows before any frame has arrived.
///
/// This used to be the sentence "No frames yet." alone in the space three statistics had vacated,
/// above a prominent button for marking an artifact that cannot exist yet. An empty state is a state
/// worth designing: say what is true, say what to do next, and do not offer actions that cannot run.
struct IdleState: View {
    @ObservedObject var monitor: RuntimeMonitor

    private var message: (String, String) {
        switch monitor.displayState {
        case .noRuntime:
            return ("gamecontroller", "Choose a game below to get started.")
        case .injected:
            // Arming is the long pole, and the runtime reports its progress the whole way. Saying
            // "waiting for the game to render" through a minute of it was wrong in both halves:
            // the game is rendering, and it is not what we are waiting for.
            if let seconds = monitor.status.armingSeconds {
                let at = monitor.status.outputResolution.map { " for \($0)" } ?? ""
                return ("gearshape.2",
                        "Preparing pipelines\(at) — \(Int(seconds))s. MetalFX is drawing meanwhile.")
            }
            if let output = monitor.status.outputResolution {
                return ("hourglass", "Attached — the game is rendering at \(output).")
            }
            return ("hourglass", "Attached — waiting for the game to render.")
        case .armed:
            return ("bolt.badge.clock", "Armed. Frames will appear once the game starts drawing.")
        case .paused:
            return ("pause.circle", "Paused — MetalFX is handling frames.")
        case .failed:
            return ("exclamationmark.triangle", monitor.status.lastFailure ?? "The runtime stopped.")
        case .active:
            return ("waveform", "Running.")
        }
    }

    var body: some View {
        let (symbol, text) = message
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.title3)
                .foregroundStyle(.tertiary)
                .frame(width: 26)
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 2)
    }
}
