import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// The library is useful before a runtime is installed. Discovery and compatibility inspection
/// only read local app metadata; recording permission and game launches always need a button press.
@MainActor
struct GamesView: View {
    @ObservedObject var monitor: RuntimeMonitor
    @ObservedObject private var capture = GameCaptureEngine.shared
    @ObservedObject private var library = GameLibraryPresentationState.shared
    @AppStorage("netal.gameLibrary.selectedApp") private var selectedGameID = ""
    @State private var search = ""
    @State private var choosingGame = true
    @State private var sourceFilter: SourceFilter = .all
    @State private var selectedCrossOverExecutable = ""
    @State private var applicationRevision = 0
    @State private var crossOverProcessIDs: Set<pid_t> = []
    @State private var crossOverVirtualDesktop = false
    @State private var libraryError: String?
    @State private var compatibility: NativeGameCompatibility?
    @State private var isInspecting = false
    @AppStorage("netal.gameLibrary.enhancementPath") private var mode: Mode = .window
    @State private var windowID: UInt32?
    @AppStorage("netal.gameLibrary.windowSuperResolution") private var windowSR = true
    @AppStorage("netal.gameLibrary.windowInterpolation") private var interpolation = false
    @State private var nativeSR = true
    @State private var nativeFG = false
    @State private var confirmLaunch = false
    @State private var isLaunching = false
    @State private var launchNote: String?
    @State private var launchFailed = false

    /// The render harness supplies a preloaded library. It never enumerates windows or requests
    /// capture permission, and it does not overwrite the user's persisted library selection.
    init(monitor: RuntimeMonitor, initialGames: [GameLibrary.Candidate]? = nil,
         initialSelection: String? = nil) {
        self.monitor = monitor
        _choosingGame = State(initialValue: initialSelection == nil)
        previewSelection = initialSelection
        _library = ObservedObject(wrappedValue: initialGames.map { GameLibraryPresentationState(games: $0) }
                                  ?? GameLibraryPresentationState.shared)
    }

    private let previewSelection: String?
    private enum SourceFilter: String, CaseIterable, Identifiable {
        case all = "All", mac = "Mac", crossOver = "CrossOver"
        var id: String { rawValue }
    }

    private enum Mode: String, CaseIterable, Identifiable {
        case window = "Game window", native = "Native models"
        var id: String { rawValue }
        var title: String { self == .window ? "Window" : "Integrated" }
    }

    private var selectedGame: GameLibrary.Candidate? {
        library.games.first { $0.id == (previewSelection ?? selectedGameID) } ?? library.games.first
    }

    private var filteredGames: [GameLibrary.Candidate] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return library.games.filter { game in
            let sourceMatches = sourceFilter == .all
                || (sourceFilter == .crossOver ? game.crossOver != nil : game.crossOver == nil)
            let text = "\(game.name) \(game.sourceDetail)"
            return sourceMatches && (query.isEmpty || text.localizedStandardContains(query))
        }
    }

    /// App path plus process identity supports games with no bundle identifier without exposing
    /// unrelated windows. A real bundle identifier is a fallback for unusual app launchers.
    private var gameWindows: [CaptureGameWindow] {
        guard let game = selectedGame else { return [] }
        let processIDs = runningProcessIDs(for: game)
        return capture.windows.filter {
            processIDs.contains($0.processID)
                || (!game.bundleID.isEmpty && !game.bundleID.hasPrefix("path:")
                    && $0.bundleIdentifier == game.bundleID)
        }
    }

    private var visibleRowCapacity: Int { max(2, min(6, Int(((NSScreen.main?.visibleFrame.height ?? 800) - 310) / 52))) }

    private var sessionBusy: Bool { capture.isRunning || capture.isBusy || isLaunching }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if choosingGame || selectedGame == nil {
                libraryHeader
                libraryPicker
            } else if let game = selectedGame {
                selectedGameHeader(game)
                Picker("Enhancement path", selection: $mode) {
                    ForEach(Mode.allCases) { mode in Text(mode.title).tag(mode) }
                }
                .pickerStyle(.segmented).labelsHidden().padding(.horizontal, 2)
                .disabled(sessionBusy)
                .help("Window processing captures the game image. Integrated models require a supported game interface and prepared runtime.")
                VStack(alignment: .leading, spacing: 12) {
                    if mode == .window { windowControls(game) }
                    else if let target = game.crossOver { crossOverControls(game, target: target) }
                    else { nativeControls(game) }
                    if let error = libraryError {
                        message(error, symbol: "exclamationmark.triangle", tint: .orange)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .id(game.id + mode.rawValue)
            }
        }
        .task {
            guard !CommandLine.arguments.contains("--render") else { return }
            if !library.didLoad { await refreshLibrary() }
            else { inspectSelection() }
        }
        .onReceive(NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didLaunchApplicationNotification)) { _ in refreshGameProcesses() }
        .onReceive(NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didTerminateApplicationNotification)) { _ in refreshGameProcesses() }
        .onReceive(NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didActivateApplicationNotification)) { _ in refreshGameProcesses() }
        .onChange(of: selectedGameID) { _, _ in inspectSelection() }
        .onChange(of: capture.windows) { _, _ in refreshGameProcesses() }
        .alert("Start \(selectedGame?.name ?? "game") with Netal?", isPresented: $confirmLaunch) {
            Button("Cancel", role: .cancel) { }
            Button(selectedGame?.crossOver != nil || selectedGame.map(isGameRunning) == true ? "Save complete — start" : "Launch game") {
                launchSelectedGame()
            }
        } message: {
            Text(selectedGame?.crossOver != nil
                 ? "Save and quit the game first. Netal will start the selected game in its CrossOver bottle with SR and the requested FG setting. Compatibility and frame delivery remain unverified until the game reports them."
                 : "Save your progress first. If the game is running, Netal will ask it to quit before restarting it. Integrated enhancement requires supported game interfaces; launch alone does not confirm SR or FG activity.")
        }
    }

    private var libraryHeader: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Your games").font(.headline)
                if choosingGame || selectedGame == nil {
                    Text("Mac, Steam and CrossOver")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 8)
            Button {
                Task { await refreshLibrary() }
            } label: {
                if library.isLoading { ProgressView().controlSize(.mini) }
                else { Image(systemName: "arrow.clockwise") }
            }
            .oraculumGlassButton()
            .accessibilityLabel("Refresh game library")
            .help("Scan installed games again")
            .disabled(library.isLoading || sessionBusy)
            Button("Add game…", action: chooseGame)
                .oraculumGlassButton()
                .disabled(sessionBusy)
        }
        .controlSize(.small)
    }

    private var libraryPicker: some View {
        VStack(spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Find a game", text: $search)
                    .textFieldStyle(.plain)
                    .accessibilityLabel("Search installed games")
                if !search.isEmpty {
                    Button { search = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                        .accessibilityLabel("Clear game search")
                }
            }
            .padding(9)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 9))
            Picker("Game source", selection: $sourceFilter) {
                ForEach(SourceFilter.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden()
            .disabled(sessionBusy)

            if library.isLoading && library.games.isEmpty {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Finding installed games…").font(.callout).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 76)
            } else if filteredGames.isEmpty {
                VStack(spacing: 5) {
                    Image(systemName: search.isEmpty ? "gamecontroller" : "magnifyingglass")
                        .font(.title2).foregroundStyle(.secondary)
                    Text(!search.isEmpty ? "No matching games" : sourceFilter == .crossOver ? "No CrossOver games found" : "Add your first game")
                        .font(.callout.weight(.medium))
                    Text(!search.isEmpty ? "Try another name or clear your search." : sourceFilter == .crossOver ? "Install a game in a CrossOver bottle, then refresh the library." : "Choose a game’s .app file to get started.")
                        .font(.caption).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity).padding(.vertical, 14)
            } else {
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(filteredGames) { game in gameRow(game) }
                    }
                }
                .scrollIndicators(.visible)
                .frame(height: CGFloat(min(filteredGames.count, visibleRowCapacity)) * 52)
                .id(sourceFilter.rawValue + search)
                .disabled(sessionBusy)
                HStack {
                    Text("\(filteredGames.count) \(filteredGames.count == 1 ? "game" : "games")")
                    Spacer()
                    if filteredGames.count > visibleRowCapacity { Text("Scroll for more") }
                }
                .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func gameRow(_ game: GameLibrary.Candidate) -> some View {
        Button {
            selectedGameID = game.id
            choosingGame = false
        } label: {
            HStack(spacing: 9) {
                gameIcon(game, size: 32)
                VStack(alignment: .leading, spacing: 3) {
                    Text(game.name).font(.callout.weight(.medium)).lineLimit(1)
                    Text(game.sourceDetail).font(.caption2).foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                Image(systemName: game.id == selectedGameID ? "checkmark" : "chevron.right")
                    .font(.caption.weight(.medium)).foregroundStyle(.secondary)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(game.id == selectedGameID ? Color.primary.opacity(0.06) : .clear, in: RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(game.name), \(game.source.rawValue)")
        .accessibilityAddTraits(game.id == selectedGame?.id ? .isSelected : [])
    }

    private func selectedGameHeader(_ game: GameLibrary.Candidate) -> some View {
        HStack(alignment: .top, spacing: 10) {
            gameIcon(game, size: 38)
            VStack(alignment: .leading, spacing: 4) {
                Text(game.name).font(.headline).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 6) {
                    badge(game.sourceDetail)
                    if isGameRunning(game) { badge("Running") }
                }
            }
            Spacer(minLength: 0)
            Button("Change game") { choosingGame = true }
                .oraculumGlassButton()
                .controlSize(.small).disabled(sessionBusy)
        }
        .padding(.top, 2)
    }

    private func windowControls(_ game: GameLibrary.Candidate) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if capture.isRunning || capture.isBusy {
                captureSession
            } else {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Window enhancement").font(.caption).foregroundStyle(.secondary)
                        Text("MetalFX SR · Optical-flow FG").font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer()
                    badge("Experimental")
                }
                if capture.permissionGranted && !gameWindows.isEmpty {
                    HStack {
                        Text("Window").font(.callout)
                        Picker("Game window", selection: $windowID) {
                            Text("Choose a window").tag(nil as UInt32?)
                            ForEach(gameWindows) { window in
                                Text(window.title.isEmpty ? window.applicationName : window.title).tag(Optional(window.id))
                            }
                        }
                        .labelsHidden()
                        Button { capture.refreshWindows() } label: { Image(systemName: "arrow.clockwise") }
                            .controlSize(.small).help("Refresh game windows")
                    }
                }
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Super Resolution").font(.callout)
                        Text("MetalFX spatial").font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 12)
                    Toggle("Super Resolution", isOn: $windowSR).labelsHidden()
                        .toggleStyle(.switch).controlSize(.small)
                }
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Frame Generation").font(.callout)
                        Text("Experimental · optical flow").font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 12)
                    Toggle("Frame Generation", isOn: $interpolation).labelsHidden()
                        .toggleStyle(.switch).controlSize(.small)
                        .disabled(!capture.frameGenerationAvailable)
                }
                if interpolation {
                    Text("May distort moving edges and add latency. A higher displayed frame rate is not guaranteed.")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Picker("Output", selection: $capture.presentationMode) {
                    Text("Match game window").tag(CapturePresentationMode.window)
                    Text("Fill display · controller").tag(CapturePresentationMode.fillDisplay)
                }
                .font(.callout)
                if capture.presentationMode == .fillDisplay {
                    Text(windowSR
                         ? "Use a smaller game window and a controller. Mouse menus keep the original window coordinates."
                         : "Standard scaling; SR is off. Use a controller—mouse menus keep the original window coordinates.")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                } else if windowSR {
                    Text("Choose Fill display with a smaller game window to use SR.")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Divider()
                Text(nextWindowStep(game)).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button {
                    if !isGameRunning(game) { openNormally(game) }
                    else if !capture.permissionGranted || gameWindows.isEmpty { capture.refreshWindows() }
                    else if let windowID, let window = gameWindows.first(where: { $0.id == windowID }) {
                        capture.start(windowID: windowID, scale: 1,
                                      frameGeneration: interpolation && capture.frameGenerationAvailable,
                                      superResolution: windowSR,
                                      expectedProcessID: window.processID)
                    }
                } label: {
                    HStack(spacing: 7) {
                        if isLaunching || capture.isRefreshing { ProgressView().controlSize(.small) }
                        else { Image(systemName: !isGameRunning(game) ? "arrow.up.forward.app" : "play.fill") }
                        Text(nextWindowAction(game))
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLaunching || capture.isRefreshing || (isGameRunning(game) && capture.permissionGranted && !gameWindows.isEmpty && windowID == nil))
                if let target = game.crossOver, CrossOverLaunch.virtualDesktopExecutable(for: target) != nil {
                    DisclosureGroup("Launch options") {
                        VStack(alignment: .leading, spacing: 6) {
                            Toggle("Use a virtual desktop", isOn: Binding(
                                get: { crossOverVirtualDesktop },
                                set: { enabled in
                                    crossOverVirtualDesktop = enabled
                                    UserDefaults.standard.set(enabled, forKey: CrossOverLaunch.virtualDesktopPreferenceKey(for: target))
                                }))
                                .toggleStyle(.switch).controlSize(.small)
                            Text("Opens Unity at 1280 × 720. Try this if it crashes before the title screen. Applies to the next launch; keep Steam signed in.")
                                .font(.caption2).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }.padding(.top, 6)
                    }.font(.caption)
                }
                DisclosureGroup("How it works") {
                    Text("SR and FG can run separately or together. Window mode uses MetalFX spatial scaling and optical-flow interpolation, not DLSS. DLSS 4.5 is not available in Netal yet. Capture alone does not reduce the game’s rendering workload.")
                        .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .font(.caption)
            }
            if let error = capture.errorMessage { message(error, symbol: "exclamationmark.triangle", tint: .orange) }
            if let launchNote { message(launchNote, symbol: launchFailed ? "exclamationmark.triangle" : "info.circle", tint: launchFailed ? .orange : .secondary) }
        }
    }

    private func nextWindowAction(_ game: GameLibrary.Candidate) -> String {
        if isLaunching { return "Opening game…" }
        if capture.isRefreshing { return "Finding windows…" }
        if !isGameRunning(game) { return "Open game" }
        if !capture.permissionGranted { return "Allow screen recording" }
        if gameWindows.isEmpty { return "Find game window" }
        return windowID == nil ? "Choose a window above" : "Start enhancement"
    }

    private func nextWindowStep(_ game: GameLibrary.Candidate) -> String {
        if !isGameRunning(game) { return "Open \(game.name) to get started." }
        if !capture.permissionGranted { return "Allow Screen Recording to capture your chosen game window. Audio is not captured." }
        if gameWindows.isEmpty { return "Keep the game window visible, then find its window." }
        return "Ready to process your selected window on this Mac."
    }

    private func crossOverControls(_ game: GameLibrary.Candidate, target: CrossOverLibrary.Target) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Opens in the \(target.bottle) bottle.").font(.caption).foregroundStyle(.secondary)
            Button { openNormally(game) } label: {
                Label(isLaunching ? "Opening…" : "Open in CrossOver", systemImage: "arrow.up.forward.app")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).disabled(sessionBusy)
            DisclosureGroup("Netal model integration") {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Experimental SR integration. Launch success does not establish compatibility or frame delivery.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    if target.steamAppID != nil && !target.executables.isEmpty {
                        Picker("Game process", selection: $selectedCrossOverExecutable) {
                            Text("Choose executable").tag("")
                            ForEach(target.executables, id: \.self) { Text($0).tag($0) }
                        }
                        Toggle("Frame Generation", isOn: $nativeFG)
                            .toggleStyle(.switch).controlSize(.small).disabled(!monitor.fgModel.isReady || sessionBusy)
                        Text(monitor.model.isReady ? "SR model verified" : monitor.model.summary)
                            .font(.caption2).foregroundStyle(.secondary)
                        Button("Restart with Netal…") { confirmLaunch = true }
                            .buttonStyle(.bordered)
                            .disabled(!canLaunchCrossOver)
                    } else {
                        Text("Open this shortcut in CrossOver. Netal’s integrated launcher currently requires a Steam title and a known game executable.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                }.padding(.top, 6)
            }.font(.caption)
            if let launchNote { message(launchNote, symbol: launchFailed ? "exclamationmark.triangle" : "info.circle", tint: launchFailed ? .orange : .secondary) }
        }
    }

    private var canLaunchCrossOver: Bool {
        !sessionBusy && selectedGame?.crossOver?.steamAppID != nil
            && selectedGame?.crossOver?.executables.contains(selectedCrossOverExecutable) == true
            && monitor.model.isReady && (!nativeFG || monitor.fgModel.isReady) && GameLauncher.isCrossOverAvailable
    }

    private var captureSession: some View {
        VStack(alignment: .leading, spacing: 10) {
            message(capture.statusMessage,
                    symbol: capture.isRunning ? "rectangle.inset.filled" : "hourglass",
                    tint: capture.isRunning ? OraculumTheme.accent : .secondary)
            if capture.capturedFrameCount > 0 {
                HStack {
                    Text("Pixels").foregroundStyle(.secondary)
                    Spacer()
                    Text("\(capture.sourceResolution) → \(capture.outputResolution)")
                        .monospacedDigit()
                }
                .font(.caption)
                HStack {
                    captureMetric(capture.capturedFPS, title: "Captured fps")
                    Spacer()
                    captureMetric(capture.presentedFPS, title: "Presented fps")
                    Spacer()
                    VStack(alignment: .leading, spacing: 2) {
                        Text(capture.generatedFrameCount.formatted()).font(.callout.monospacedDigit().weight(.semibold))
                        Text("Generated frames").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Text("Capture and presentation rates are measured by Netal; they are not the game’s rendering rate.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Text("Processing cost").foregroundStyle(.secondary)
                    Spacer()
                    Text(capture.processingMilliseconds > 0
                         ? String(format: "%.1f ms", capture.processingMilliseconds) : "—")
                        .monospacedDigit()
                }
                .font(.caption)
                if capture.droppedFrameCount > 0 {
                    HStack {
                        Text("Dropped capture frames").foregroundStyle(.secondary)
                        Spacer()
                        Text(capture.droppedFrameCount.formatted()).monospacedDigit()
                    }
                    .font(.caption)
                }
                Text("Processing cost excludes capture, display and input latency.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let note = capture.interpolationStatus {
                    Text(note).font(.caption2).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Button(role: .destructive) { capture.stop() } label: {
                Label("Stop window enhancement", systemImage: "stop.fill")
                    .frame(maxWidth: .infinity)
            }
            .oraculumGlassButton()
        }
    }

    private func nativeControls(_ game: GameLibrary.Candidate) -> some View {
        OraculumCard {
            VStack(alignment: .leading, spacing: 12) {
                Text("Models inside the game").font(.callout.weight(.semibold))
                Text("For compatible MetalFX games. Netal must start the game, and the game must supply the inputs each model needs.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if isInspecting {
                    HStack(spacing: 7) {
                        ProgressView().controlSize(.small)
                        Text("Checking this app…").font(.caption).foregroundStyle(.secondary)
                    }
                } else if let compatibility {
                    message(compatibility.summary,
                            symbol: compatibility.canLaunchModels ? "checkmark.circle" : "info.circle",
                            tint: compatibility.canLaunchModels ? .secondary : .orange)
                    Text(compatibility.detail).font(.caption2).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Divider()
                nativeFeature("Super Resolution (SR)", ready: monitor.model.isReady,
                              value: $nativeSR, summary: monitor.model.summary)
                nativeFeature("Frame Generation (FG)", ready: monitor.fgModel.isReady,
                              value: $nativeFG,
                              summary: monitor.fgModel.isReady ? "Model verified · game integration still required" : "A verified FG runtime is required")
                if !GameLauncher.isAvailable {
                    Text("Choose a prepared native runtime in Settings to enable model launches.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Button {
                    confirmLaunch = true
                } label: {
                    HStack(spacing: 7) {
                        if isLaunching { ProgressView().controlSize(.small) }
                        else { Image(systemName: "play.fill") }
                        Text(isLaunching ? "Launching…" : isGameRunning(game) ? "Restart with native models…" : "Launch with native models…")
                    }
                    .frame(maxWidth: .infinity)
                }
                .oraculumGlassButton(prominent: true)
                .disabled(!canLaunchNative)
                if let launchNote {
                    message(launchNote, symbol: launchFailed ? "exclamationmark.triangle" : "info.circle",
                            tint: launchFailed ? .orange : .secondary)
                }
            }
        }
    }

    private var canLaunchNative: Bool {
        !sessionBusy && !isInspecting && compatibility?.canLaunchModels == true && GameLauncher.isAvailable
            && ((nativeSR && monitor.model.isReady) || (nativeFG && monitor.fgModel.isReady))
    }

    private func nativeFeature(_ title: String, ready: Bool, value: Binding<Bool>, summary: String) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.callout)
                Text(summary).font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Toggle(title, isOn: Binding(get: { value.wrappedValue && ready }, set: { value.wrappedValue = $0 }))
                .labelsHidden().toggleStyle(.switch).controlSize(.small)
                .disabled(!ready || sessionBusy || compatibility?.canLaunchModels != true)
        }
    }

    private func gameIcon(_ game: GameLibrary.Candidate, size: CGFloat) -> some View {
        Group {
            if let icon = game.icon { Image(nsImage: icon).resizable() }
            else {
                Image(systemName: "gamecontroller.fill").resizable().scaledToFit()
                    .padding(7).foregroundStyle(.secondary)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.2))
        .accessibilityHidden(true)
    }

    private func badge(_ text: String) -> some View {
        Text(text).font(.system(size: 10, weight: .medium))
            .foregroundStyle(.secondary).padding(.horizontal, 6).padding(.vertical, 3)
            .background(.quaternary.opacity(0.6), in: Capsule())
    }

    private func message(_ text: String, symbol: String, tint: Color) -> some View {
        Label {
            Text(text).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: symbol)
        }
        .font(.caption).foregroundStyle(tint)
    }

    private func captureMetric(_ value: Double, title: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value > 0 ? String(format: "%.0f", value) : "—")
                .font(.callout.monospacedDigit().weight(.semibold))
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func refreshLibrary() async {
        await library.refresh()
        if !library.games.contains(where: { $0.id == selectedGameID }) {
            selectedGameID = library.games.first?.id ?? ""
        }
        else { inspectSelection() }
    }

    private func runningProcessIDs(for game: GameLibrary.Candidate) -> Set<pid_t> {
        if game.crossOver != nil { return game.id == selectedGame?.id ? crossOverProcessIDs : [] }
        let selectedPath = game.appURL.standardizedFileURL.resolvingSymlinksInPath().path
        return Set(NSWorkspace.shared.runningApplications.compactMap { app -> pid_t? in
            if app.bundleURL?.standardizedFileURL.resolvingSymlinksInPath().path == selectedPath {
                return app.processIdentifier
            }
            if let executable = app.executableURL?.standardizedFileURL.resolvingSymlinksInPath().path,
               executable.hasPrefix(selectedPath + "/Contents/") { return app.processIdentifier }
            return nil
        })
    }

    private func refreshGameProcesses() {
        applicationRevision &+= 1
        crossOverProcessIDs = []
        if let target = selectedGame?.crossOver, !CommandLine.arguments.contains("--render") {
            let candidates = Set(NSWorkspace.shared.runningApplications.map(\.processIdentifier))
                .union(capture.windows.map(\.processID))
            crossOverProcessIDs = Set(candidates.filter {
                CrossOverProcessIdentity.matches(processID: $0, target: target)
            })
        }
        reconcileWindowSelection()
    }

    private func isGameRunning(_ game: GameLibrary.Candidate) -> Bool {
        _ = applicationRevision
        return !runningProcessIDs(for: game).isEmpty || (!game.bundleID.hasPrefix("path:") && game.identity.isRunning)
    }

    private func inspectSelection() {
        windowID = nil
        compatibility = nil
        launchNote = nil
        launchFailed = false
        refreshGameProcesses()
        guard let game = selectedGame else { isInspecting = false; return }
        if let target = game.crossOver {
            isInspecting = false
            selectedCrossOverExecutable = target.executables.count == 1 ? target.executables[0] : ""
            crossOverVirtualDesktop = UserDefaults.standard.bool(forKey: CrossOverLaunch.virtualDesktopPreferenceKey(for: target))
            reconcileWindowSelection()
            return
        }
        isInspecting = true
        NativeGameCompatibility.inspect(game.appURL) { result in
            Task { @MainActor in
                guard selectedGame?.id == game.id else { return }
                compatibility = result
                isInspecting = false
            }
        }
        reconcileWindowSelection()
    }

    private func reconcileWindowSelection() {
        if let windowID, gameWindows.contains(where: { $0.id == windowID }) { return }
        windowID = gameWindows.count == 1 ? gameWindows.first?.id : nil
    }

    private func chooseGame() {
        let panel = NSOpenPanel()
        panel.title = "Add a game"
        panel.message = "Choose the game’s application. Steam games are usually inside a steamapps/common folder."
        panel.prompt = "Add game"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.applicationBundle]
        panel.directoryURL = URL(fileURLWithPath: "/Applications")
        NSApp.activate(ignoringOtherApps: true)
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                guard let game = GameLibrary.add(url) else {
                    libraryError = "This application could not be added. Choose a game’s .app file."
                    return
                }
                libraryError = nil
                if !library.games.contains(where: { $0.id == game.id }) {
                    library.games.append(game)
                    library.games.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
                }
                search = ""
                selectedGameID = game.id
                choosingGame = false
            }
        }
    }

    private func launchSelectedGame() {
        guard let game = selectedGame else { return }
        if let target = game.crossOver, let appID = target.steamAppID {
            guard canLaunchCrossOver else { return }
            isLaunching = true; launchFailed = false; launchNote = "Preparing \(game.name) in \(target.bottle)…"
            GameLauncher.relaunchCrossOver(bottle: target.bottle, appID: appID,
                executable: selectedCrossOverExecutable, frameGeneration: nativeFG && monitor.fgModel.isReady) { outcome in
                    finishLaunch(outcome)
                }
            return
        }
        guard canLaunchNative else { return }
        isLaunching = true
        launchFailed = false
        launchNote = isGameRunning(game) ? "Asking \(game.name) to quit safely…" : "Starting \(game.name)…"
        let options = GameLauncher.Options(superResolution: nativeSR && monitor.model.isReady,
                                           frameGeneration: nativeFG && monitor.fgModel.isReady,
                                           observeOnly: false, steamAppID: game.steamAppID)
        GameLauncher.relaunch(game.identity, options: options) { outcome in
            finishLaunch(outcome)
        }
    }

    private func finishLaunch(_ outcome: GameLauncher.Outcome) {
        isLaunching = false
        switch outcome {
        case .launched: launchNote = "Launch requested. Check Status for measured SR and FG activity."
        case .launcherMissing: launchFailed = true; launchNote = "Choose a prepared runtime in Settings to use the integrated launcher."
        case .failed(let reason): launchFailed = true; launchNote = reason
        }
    }

    /// Opening from the window path never asks the game to quit and never injects a runtime.
    private func openNormally(_ game: GameLibrary.Candidate) {
        launchFailed = false
        launchNote = nil
        if let target = game.crossOver {
            isLaunching = true
            CrossOverLaunch.open(target, virtualDesktop: crossOverVirtualDesktop) { error in
                isLaunching = false; launchFailed = error != nil
                launchNote = error ?? (crossOverVirtualDesktop
                    ? "Sent to CrossOver in a 1280 × 720 desktop. Sign in to Ubisoft if requested."
                    : "Sent to CrossOver · \(target.bottle)")
            }
            return
        }
        if let appID = game.steamAppID,
           !appID.isEmpty, appID.allSatisfy(\.isNumber),
           let url = URL(string: "steam://run/\(appID)") {
            if NSWorkspace.shared.open(url) {
                launchNote = "Sent to Steam. When the game window appears, refresh windows."
            } else {
                launchFailed = true
                launchNote = "Steam could not open this game. Open Steam, start the game, then refresh windows."
            }
            return
        }
        isLaunching = true
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        NSWorkspace.shared.openApplication(at: game.appURL, configuration: configuration) { _, error in
            Task { @MainActor in
                isLaunching = false
                launchFailed = error != nil
                launchNote = error.map { "Could not open \(game.name): \($0.localizedDescription)" }
                    ?? "Opened \(game.name). When its window appears, refresh windows."
            }
        }
    }
}

/// Keep the discovered library alive while the user moves between menu tabs. App locations are
/// persisted by GameLibrary; windows are deliberately refreshed from the current running game.
@MainActor
private final class GameLibraryPresentationState: ObservableObject {
    static let shared = GameLibraryPresentationState()
    @Published var games: [GameLibrary.Candidate] = []
    @Published private(set) var isLoading = false
    private(set) var didLoad = false

    init(games: [GameLibrary.Candidate]? = nil) {
        if let games {
            self.games = games
            didLoad = true
        }
    }

    func refresh() async {
        guard !isLoading else { return }
        isLoading = true
        games = await Task.detached(priority: .userInitiated) { GameLibrary.discover() }.value
        isLoading = false
        didLoad = true
    }
}
