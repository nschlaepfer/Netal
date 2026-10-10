import SwiftUI
import AppKit

/// Draws the real panel in both appearances without opening the menu bar or starting capture.
/// Usage: Netal --render <directory> [log path]
/// No fixture activity is presented as a live session: optional log input is read by the monitor,
/// and the preview files contain only the state that input establishes.
@MainActor
enum RenderHarness {
    static func runIfRequested() -> Bool {
        let arguments = CommandLine.arguments
        guard let flag = arguments.firstIndex(of: "--render"), flag + 1 < arguments.count else {
            return false
        }
        let directory = URL(fileURLWithPath: arguments[flag + 1])
        do { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }
        catch { print("result=fail reason=\(error.localizedDescription)"); exit(1) }

        let logPath = arguments.count > flag + 2 ? arguments[flag + 2] : NetalPaths.logURL().path
        let monitor = RuntimeMonitor(logURL: URL(fileURLWithPath: logPath), startImmediately: false)
        monitor.modelBundleFallbacks = [URL(fileURLWithPath: NetalBuildDefaults.srBundlePath)]
        monitor.fgBundleFallbacks = [URL(fileURLWithPath: NetalBuildDefaults.fgBundlePath)]
        monitor.poll()
        // Synchronous metadata discovery is only used in this offscreen command. The real library
        // loads asynchronously; injecting its snapshot avoids capturing an arbitrary loading frame.
        let games = GameLibrary.discover()
        var failures = 0
        print("preview=layout_and_native_material_fallback live_glass_refraction=not_captured")

        for appearance in ["light", "dark"] {
            let dark = appearance == "dark"
            NSApplication.shared.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
            for tab in PanelTab.allCases {
                let view = StatusPanel(monitor: monitor, initialTab: tab, initialGames: games)
                    .environment(\.colorScheme, dark ? .dark : .light)
                    .background(dark ? Color(white: 0.12) : Color(white: 0.96))
                guard let png = snapshot(view, dark: dark) else { failures += 1; continue }
                let name = "panel-\(tab.rawValue.lowercased())-\(appearance).png"
                do {
                    try png.write(to: directory.appendingPathComponent(name))
                    print("rendered=\(name)")
                } catch { failures += 1; print("failed=\(name)") }
            }

            // The library is a child of Status, not an extra item in the approved five-tab bar.
            let gamesView = StatusPanel(monitor: monitor, initialGames: games, initialGamesVisible: true)
                .environment(\.colorScheme, dark ? .dark : .light)
                .background(dark ? Color(white: 0.12) : Color(white: 0.96))
            if let png = snapshot(gamesView, dark: dark) {
                let name = "panel-games-\(appearance).png"
                do {
                    try png.write(to: directory.appendingPathComponent(name))
                    print("rendered=\(name)")
                } catch { failures += 1; print("failed=\(name)") }
            } else { failures += 1 }

            // Both detail routes are part of picker QA; no game is launched by the harness.
            for (name, game) in [("mac", games.first { $0.crossOver == nil }),
                                 ("crossover", games.first { $0.crossOver != nil })] {
                guard let game else { continue }
                let selected = StatusPanel(monitor: monitor, initialGames: games,
                    initialGamesVisible: true, initialSelectedGame: game.id)
                    .environment(\.colorScheme, dark ? .dark : .light)
                    .background(dark ? Color(white: 0.12) : Color(white: 0.96))
                if let png = snapshot(selected, dark: dark) {
                    do { try png.write(to: directory.appendingPathComponent("panel-game-\(name)-\(appearance).png")) }
                    catch { failures += 1 }
                } else { failures += 1 }
            }

            // The original blob template and status shapes at actual menu-bar size.
            let samples: [(String, RuntimeState, Bool)] = [
                ("Idle", .noRuntime, false), ("Ready", .armed, false),
                ("Active", .active, false), ("Paused", .paused, false),
                ("Attention", .failed("Preview"), false), ("Window", .noRuntime, true)
            ]
            let strip = HStack(spacing: 22) {
                ForEach(samples.indices, id: \.self) { index in
                    let sample = samples[index]
                    VStack(spacing: 8) {
                        MenuBarLabel(state: sample.1, fps: nil, captureActive: sample.2)
                        Text(sample.0).font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(18)
            .environment(\.colorScheme, dark ? .dark : .light)
            .background(dark ? Color(white: 0.12) : Color(white: 0.96))
            if let png = snapshot(strip, dark: dark) {
                do {
                    try png.write(to: directory.appendingPathComponent("menubar-\(appearance).png"))
                    print("rendered=menubar-\(appearance).png")
                } catch { failures += 1 }
            } else { failures += 1 }
        }
        NSApplication.shared.appearance = nil
        guard failures == 0 else { print("result=fail count=\(failures)"); exit(1) }
        print("result=pass")
        return true
    }

    private final class KeyWindow: NSWindow {
        override var isKeyWindow: Bool { true }
        override var canBecomeKey: Bool { true }
    }

    /// AppKit-backed controls need a hosting view, not ImageRenderer, to appear correctly.
    private static func snapshot<V: View>(_ view: V, dark: Bool) -> Data? {
        let hosting = NSHostingView(rootView: view)
        hosting.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        let size = hosting.fittingSize
        hosting.frame = NSRect(origin: .zero, size: NSSize(width: max(size.width, 1),
                                                        height: max(size.height, 1)))
        let window = KeyWindow(contentRect: hosting.frame, styleMask: .borderless, backing: .buffered, defer: false)
        window.appearance = hosting.appearance
        window.isReleasedWhenClosed = false
        window.contentView = hosting
        window.setFrameOrigin(NSPoint(x: -10_000, y: -10_000))
        window.orderFront(nil)
        defer { window.orderOut(nil) }
        // Let intrinsic-height preferences settle before caching a detail scroll view.
        for _ in 0..<5 {
            hosting.layoutSubtreeIfNeeded()
            RunLoop.main.run(until: Date().addingTimeInterval(0.01))
        }
        hosting.layoutSubtreeIfNeeded()
        hosting.displayIfNeeded()
        guard let rep = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else { return nil }
        hosting.cacheDisplay(in: hosting.bounds, to: rep)
        return rep.representation(using: .png, properties: [:])
    }
}
