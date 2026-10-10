import SwiftUI
import AppKit

@main
struct NetalMenuBarApp: App {
    /// Notification work happens after launch, not in init - see NetalLaunchDelegate.
    @NSApplicationDelegateAdaptor(NetalLaunchDelegate.self) private var delegate

    init() {
        // Offscreen render mode exits before any UI exists; see RenderHarness. This one is safe in
        // init because it draws without a running application and then leaves.
        if RenderHarness.runIfRequested() { exit(0) }
        if CaptureDiagnostics.runIfRequested() { exit(0) }
    }

    @StateObject private var monitor: RuntimeMonitor = {
        let monitor = RuntimeMonitor(logURL: NetalPaths.logURL(), logURLProvider: NetalPaths.logURL)
        // The CrossOver runtime is not sandboxed into a container, so the container scan cannot see
        // it; without this a Windows game is invisible to the app however healthy its session is.
        monitor.extraLogPaths = [URL(fileURLWithPath: NetalBuildDefaults.crossoverLogPath)]
        monitor.modelBundleFallbacks = [URL(fileURLWithPath: NetalBuildDefaults.srBundlePath)]
        monitor.fgBundleFallbacks = [URL(fileURLWithPath: NetalBuildDefaults.fgBundlePath)]
        monitor.onNotifications = { notifications in
            for notification in notifications { NetalNotifier.post(notification) }
        }
        return monitor
    }()

    @StateObject private var capture = GameCaptureEngine.shared

    var body: some Scene {
        MenuBarExtra {
            StatusPanel(monitor: monitor)
        } label: {
            MenuBarLabel(state: monitor.displayState, fps: monitor.status.effectivePacing?.fps,
                         vitals: NetalVitals.from(state: monitor.displayState,
                                                  pacing: monitor.status.effectivePacing),
                         captureActive: capture.isRunning)
        }
        .menuBarExtraStyle(.window)
    }
}

/// Where the runtime writes its log. Baked at build time from the repository this was built in,
/// overridable so the app can follow a runtime somewhere else.
enum NetalPaths {
    static func logURL() -> URL {
        if let override = UserDefaults.standard.string(forKey: "NetalLogPath"), !override.isEmpty {
            return URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
        }
        if let env = ProcessInfo.processInfo.environment["NEURALSTUDIO_RE9_METALFX_LOG"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        return URL(fileURLWithPath: NetalBuildDefaults.logPath)
    }
}

/// The bundled artwork, loaded once.
enum NetalArt {
    static let mark: NSImage? = Bundle.main.image(forResource: "NetalMark")
    /// The two hop frames, registered to a shared canvas by tools/netal-sprite.swift: same width,
    /// same ground line, so swapping one for the other reads as the character moving rather than
    /// as two pictures.
    static let rest: NSImage? = Bundle.main.image(forResource: "NetalRest")
    static let hop: NSImage? = Bundle.main.image(forResource: "NetalHop")

    /// Where the eyes sit in those frames, as fractions of the canvas, measured by
    /// tools/netal-sprite.swift when the artwork was cut. Baked rather than found at runtime: the
    /// app should not have to know how the sprite was cropped, and a measurement that runs once is
    /// one that cannot drift.
    struct Eye: Decodable { let x: Double, y: Double, r: Double }
    static let eyes: [Eye] = load("NetalEyes")
    /// The same, for the menu-bar template — a different source image, so different positions.
    static let templateEyes: [Eye] = load("NetalTemplateEyes")

    /// The menu-bar icon with the eye light burned into it.
    ///
    /// This is not how the About orb does it, and the difference is forced. `MenuBarExtra` builds
    /// its status item by pulling an image out of the label view: an `Image` survives, and anything
    /// layered around it - shapes, gradients, a plain `Circle` - is dropped, not merely tinted. So
    /// the light cannot sit behind the icon in SwiftUI; it has to be part of the image.
    ///
    /// The eyes are punched out of the artwork, so painting the glow first and the body over it
    /// leaves colour showing through exactly the eye holes, with no second mask to keep in sync.
    static func menuBarImage(hue: Double, saturation: Double, intensity: Double,
                             dark: Bool) -> NSImage? {
        guard let base = template else { return nil }
        // Nothing to show: hand back the plain template, which macOS adapts by itself. That
        // adaptation is better than anything computed here, so it is worth keeping for the state
        // the icon spends most of its life in.
        guard intensity > 0, !templateEyes.isEmpty else { return base }

        let size = base.size
        let glow = NSColor(hue: hue, saturation: saturation, brightness: 1, alpha: 1)
        let body = NSImage(size: size)
        body.lockFocus()
        base.draw(in: NSRect(origin: .zero, size: size))
        // A template image carries only a shape; the colour has to be supplied. `sourceAtop` paints
        // inside the existing alpha, so the eye holes stay holes.
        (dark ? NSColor.white : NSColor.black).set()
        NSRect(origin: .zero, size: size).fill(using: .sourceAtop)
        body.unlockFocus()

        let out = NSImage(size: size)
        out.lockFocus()
        for eye in templateEyes {
            let hole = size.width * CGFloat(eye.r)
            let radius = hole * 3
            // Flipped: the measurements are in image coordinates, AppKit draws bottom-up.
            let centre = NSPoint(x: size.width * CGFloat(eye.x),
                                 y: size.height * (1 - CGFloat(eye.y)))
            let gradient = NSGradient(colorsAndLocations:
                (glow.withAlphaComponent(intensity), 0),
                (glow.withAlphaComponent(intensity), hole / radius),
                (glow.withAlphaComponent(0), 1))
            gradient?.draw(fromCenter: centre, radius: 0, toCenter: centre, radius: radius,
                           options: [])
        }
        body.draw(in: NSRect(origin: .zero, size: size))
        // A little of the light back over the face, so the eyes read as lit rather than as two
        // coloured dots. Additive, and faint enough to survive being wrong on an odd wallpaper.
        for eye in templateEyes {
            let radius = size.width * CGFloat(eye.r) * 3
            let centre = NSPoint(x: size.width * CGFloat(eye.x),
                                 y: size.height * (1 - CGFloat(eye.y)))
            let gradient = NSGradient(colorsAndLocations:
                (glow.withAlphaComponent(0.32 * intensity), 0),
                (glow.withAlphaComponent(0), 1))
            gradient?.draw(fromCenter: centre, radius: 0, toCenter: centre, radius: radius,
                           options: [])
        }
        out.unlockFocus()
        // Colour only survives on a non-template image; macOS would flatten a template one to a
        // single menu-bar-coloured silhouette.
        out.isTemplate = false
        return out
    }

    private static func load(_ name: String) -> [Eye] {
        guard let url = Bundle.main.url(forResource: name, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([String: [Eye]].self, from: data)
        else { return [] }
        return decoded["eyes"] ?? []
    }
    /// Template so macOS renders it monochrome and it inverts with the menu bar, like every other
    /// status item. Colour in the menu bar would be the one thing that looks out of place.
    static let template: NSImage? = {
        guard let image = Bundle.main.image(forResource: "NetalTemplate") else { return nil }
        image.isTemplate = true
        image.size = NSSize(width: 17, height: 17)
        return image
    }()
}

// MARK: - Menu bar item

extension Color {
    /// The one place the health colour is decided. Both the About orb and the menu-bar icon read
    /// it, so a change to the scale moves them together.
    init(_ vitals: NetalVitals) {
        guard vitals.condition != .idle else { self = .clear; return }
        self = Color(hue: vitals.tintHue, saturation: vitals.tintSaturation, brightness: 1.0)
    }
}

struct MenuBarLabel: View {
    let state: RuntimeState
    let fps: Double?
    /// Health, so the eyes in the menu bar carry it as well as the ones on the About page. This is
    /// the icon a person actually looks at while playing.
    var vitals: NetalVitals = NetalVitals()
    var captureActive = false

    /// Whether the menu bar is currently drawing light-on-dark. Only consulted while a session is
    /// active - the rest of the time the icon stays a template and macOS decides for itself.
    static var menuBarIsDark: Bool {
        NSApp.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
    }

    var body: some View {
        HStack(spacing: 4) {
            if let icon = NetalArt.menuBarImage(hue: vitals.tintHue,
                                                saturation: vitals.tintSaturation,
                                                intensity: vitals.intensity,
                                                dark: MenuBarLabel.menuBarIsDark) {
                Image(nsImage: icon)
            } else {
                Image(systemName: state.symbol)
            }
            // A badge only when something needs attention. Serving frames is the normal case and
            // gets no ornament - the frame rate is the signal.
            if captureActive {
                Image(systemName: "record.circle").imageScale(.small)
            } else if let badge = state.badgeSymbol {
                Image(systemName: badge).imageScale(.small)
            } else if case .active = state, let fps, fps > 0 {
                Text("\(Int(fps.rounded()))").font(.system(size: 11).monospacedDigit())
                    .help("Game source frames per second; display rate is not measured")
            }
        }
        .accessibilityLabel(captureActive ? "Netal, window mode running" : "Netal, \(state.title)")
    }
}

extension MenuBarLabel {
    /// Same scale as the About view's glow - literally the same, so the two cannot disagree.
    var eyeTint: Color { Color(vitals) }
}

private extension RuntimeState {
    /// Nil while everything is fine, so the menu bar stays quiet.
    var badgeSymbol: String? {
        switch self {
        case .failed:   return "exclamationmark.triangle.fill"
        case .paused:   return "pause.fill"
        case .noRuntime, .injected, .armed: return nil
        case .active:   return nil
        }
    }
}

// MARK: - Panel

enum PanelTab: String, CaseIterable, Identifiable {
    case status = "Status", graphs = "Graphs", dlss = "DLSS", settings = "Settings", about = "About"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .status:   return "waveform"
        case .graphs:   return "chart.xyaxis.line"
        case .dlss:     return "cube.transparent"
        case .settings: return "gearshape"
        case .about:    return "info.circle"
        }
    }
}

struct StatusPanel: View {
    @ObservedObject var monitor: RuntimeMonitor
    @ObservedObject private var capture = GameCaptureEngine.shared
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var tab: PanelTab
    @State private var showGames: Bool
    private let initialGames: [GameLibrary.Candidate]?
    private let initialSelectedGame: String?
    
    /// The harness renders one tab per image, so the starting tab is injectable. Defaults to the
    /// one a person wants first.
    init(monitor: RuntimeMonitor, initialTab: PanelTab = .status, initialGames: [GameLibrary.Candidate]? = nil,
         initialGamesVisible: Bool = false, initialSelectedGame: String? = nil) {
        self.monitor = monitor
        self.initialGames = initialGames
        self.initialSelectedGame = initialSelectedGame
        _tab = State(initialValue: initialTab)
        _showGames = State(initialValue: initialGamesVisible)
    }

    private var state: RuntimeState { monitor.displayState }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, 14)
                .padding(.top, 12)
                .padding(.bottom, 10)

            PanelTabBar(selection: $tab)
                .padding(.horizontal, 14)

            if capture.isRunning || capture.isBusy || capture.errorMessage != nil {
                captureControls.padding(.horizontal, 14).padding(.top, 10)
            }

            Divider().padding(.top, 10)

            Group {
                switch tab {
                case .status:
                    if showGames {
                        VStack(alignment: .leading, spacing: 10) {
                            Button { showGames = false } label: {
                                Label("Back to Status", systemImage: "chevron.left")
                            }
                            .buttonStyle(.link).font(.caption)
                            GamesView(monitor: monitor, initialGames: initialGames, initialSelection: initialSelectedGame)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    } else {
                        VStack(alignment: .leading, spacing: 12) {
                            StatusView(monitor: monitor, tab: $tab)
                            Button("Choose game…") { showGames = true }
                                .buttonStyle(.bordered).controlSize(.small)
                        }
                    }
                case .graphs:   GraphsView(monitor: monitor)
                case .dlss:     DLSSView(monitor: monitor)
                case .settings: SettingsView(monitor: monitor)
                case .about:    AboutView(vitals: NetalVitals.from(state: state,
                                                                pacing: monitor.status.effectivePacing))
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(width: tab == .status && showGames ? 360 : 300)
        .onChange(of: tab) { _, selection in
            if selection != .status { showGames = false }
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.22), value: tab)
        .animation(reduceMotion ? nil : .smooth(duration: 0.3), value: state)
    }

    private var captureControls: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(capture.errorMessage != nil ? "Window mode needs attention" : "Window mode")
                    .font(.caption.weight(.medium))
                Text(capture.errorMessage ?? capture.statusMessage)
                    .font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 2)
            if capture.isRunning || capture.isBusy {
                Button("Stop") { capture.stop() }
                    .oraculumGlassButton().controlSize(.small)
                    .keyboardShortcut("n", modifiers: [.control, .option, .command])
                    .help("Stop window mode (Control–Option–Command–N)")
            } else {
                Button("Review") { tab = .status; showGames = true }.oraculumGlassButton().controlSize(.small)
            }
        }
        .padding(.vertical, 2)
    }

    private var header: some View {
        HStack(spacing: 9) {
            StatusDot(state: state)
            VStack(alignment: .leading, spacing: 1) {
                Text("Netal").font(.headline)
                Text(subtitle)
                    .font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.tail)
            }
            Spacer(minLength: 6)
            // The game this runtime is attached to, in the corner: which game is being upscaled is
            // the first thing to know, and an icon says it faster than a bundle identifier.
            if let game = monitor.game {
                if let icon = game.icon {
                    Image(nsImage: icon)
                        .resizable().frame(width: 26, height: 26)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .opacity(game.isRunning ? 1 : 0.4)
                        .help(game.isRunning ? game.name : "\(game.name) — not running")
                        .transition(.scale.combined(with: .opacity))
                } else {
                    Text(game.name).font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var subtitle: String {
        if capture.isRunning { return "Window mode · capture and presentation" }
        switch state {
        case .active:
            // Render == output is DLAA - the network run for anti-aliasing, not for pixels. Calling
            // that "upscaling to 1280x720" is simply wrong, and it is what the panel used to say.
            let output = monitor.status.outputResolution
            let render = monitor.status.renderResolution
            let fgServing = monitor.status.frameGeneration.map { $0.isServing && !$0.paused } ?? false
            // Frame generation alone: the game is not upscaling (DLAA, or no profile), and calling
            // that "Upscaling" was wrong.
            if fgServing && !monitor.status.superResolutionServing {
                return "Frame generation" + (monitor.status.frameGeneration?.outputResolution.map { " · \(Display.res($0))" } ?? "")
            }
            let fgSuffix = fgServing ? " · FG" : ""
            if let output, let render, output == render { return "DLAA · \(Display.res(output))" + fgSuffix }
            if let output, let render { return "\(Display.res(render)) → \(Display.res(output))" + fgSuffix }
            if let output { return "Upscaling to \(Display.res(output))" + fgSuffix }
            return "Upscaling" + fgSuffix
        case .paused:
            return monitor.isPausePending ? "Pausing…" : "MetalFX is handling frames"
        case .armed:    return "Armed · waiting for frames"
        case .injected:
            let served = monitor.status.framesScheduled > 0 || (monitor.status.frameGeneration?.served ?? 0) > 0
            return served ? "Idle · game not rendering" : "Attached"
        case .failed:   return monitor.status.lastFailure ?? "Error"
        case .noRuntime: return "No runtime detected"
        }
    }
}

/// A soft filled dot rather than a glyph: at this size a shape reads as state faster than an icon,
/// and the pulse makes "serving frames" legible without a number.
struct StatusDot: View {
    let state: RuntimeState
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false

    var body: some View {
        ZStack {
            Circle().fill(tint.opacity(0.18)).frame(width: 26, height: 26)
            Circle().fill(tint).frame(width: 9, height: 9)
                .scaleEffect(pulse ? 1.18 : 1)
                .opacity(pulse ? 0.75 : 1)
        }
        .onAppear { if isActive { start() } }
        .onChange(of: isActive) { _, now in now ? start() : stop() }
        .onChange(of: reduceMotion) { _, reduced in reduced ? stop() : (isActive ? start() : stop()) }
    }

    private var isActive: Bool { if case .active = state { return true }; return false }
    private func start() {
        guard !reduceMotion, !CommandLine.arguments.contains("--render") else { return }
        withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) { pulse = true }
    }
    private func stop() { withAnimation(.default) { pulse = false } }

    private var tint: Color {
        switch state {
        case .active: return .green
        case .armed:  return .yellow
        case .paused: return .secondary
        case .failed: return .red
        default:      return .secondary
        }
    }
}


/// The panel's view switcher.
///
/// A custom bar rather than a segmented `Picker`, for two reasons. It reads better at this size -
/// an icon-only segmented control is a row of boxes, where a sliding pill shows which view you are
/// in and animates the move. And it is pure SwiftUI, so the offscreen render harness draws it;
/// AppKit-backed controls come out as placeholder glyphs, which would leave the design unreviewable
/// exactly where navigation lives.
struct PanelTabBar: View {
    @Binding var selection: PanelTab
    @Namespace private var pill

    var body: some View {
        HStack(spacing: 2) {
            ForEach(PanelTab.allCases) { tab in
                Button {
                    withAnimation(.snappy(duration: 0.24)) { selection = tab }
                } label: {
                    Image(systemName: tab.symbol)
                        .font(.system(size: 12, weight: .medium))
                        .frame(maxWidth: .infinity, minHeight: 21)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(selection == tab ? Color.primary : Color.secondary)
                .background {
                    if selection == tab {
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(.background.opacity(0.9))
                            .shadow(color: .black.opacity(0.12), radius: 1, y: 0.5)
                            .matchedGeometryEffect(id: "selection", in: pill)
                    }
                }
                .help(tab.rawValue)
            }
        }
        .padding(2)
        .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
            .fill(.quaternary.opacity(0.55)))
    }
}


/// Does the things that require a launched application.
///
/// `UNUserNotificationCenter` refuses an app that has not finished launching - it answers
/// "Notifications are not allowed for this application", which reads exactly like a permissions
/// problem and is not one. Asking from `App.init()` is too early: NSApplication has not registered
/// the process yet. It cost a signing certificate and a trip through /Applications to work that
/// out, neither of which was the cause.
final class NetalLaunchDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // The self-test needs a launched app for the same reason the real request does, so it runs
        // here rather than in init.
        if NetalNotifier.runSelfTestIfRequested() { return }
        NetalNotifier.requestAuthorization()
        // Also on launch, not only on quit: if Netal was force-quit or crashed while the HUD was on,
        // the quit hook never ran and the overlay would still be sitting in every game. Clearing it
        // here is what makes "restart Netal and it is gone" true rather than usually true.
        MetalHUD.relinquish()
    }

    func applicationWillTerminate(_ notification: Notification) {
        GameCaptureEngine.shared.stop()
        MetalHUD.relinquish()
    }
}
