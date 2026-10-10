import AppKit

/// Which game the runtime is attached to, and what it looks like.
///
/// The runtime is sandboxed into the game's container, so the log path itself carries the answer:
/// `~/Library/Containers/<bundle id>/Data/netal/netal-live.log`. That is more reliable than parsing
/// the process name out of the log, because it is true before the first line is ever written.
struct GameIdentity: Equatable {
    var bundleID: String
    var name: String
    var icon: NSImage?
    /// Where the app lives, so it can be relaunched through Netal. Nil when the bundle is gone.
    var appURL: URL?

    static func == (a: GameIdentity, b: GameIdentity) -> Bool { a.bundleID == b.bundleID }

    static var nativeLogRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Netal/NativeGames")
    }

    static func nativeBundleID(from log: URL, root: URL = nativeLogRoot) -> String? {
        let directory = log.deletingLastPathComponent()
        guard log.lastPathComponent == "netal-live.log",
              directory.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL
                == root.resolvingSymlinksInPath().standardizedFileURL,
              directory.resolvingSymlinksInPath().deletingLastPathComponent().standardizedFileURL
                == root.resolvingSymlinksInPath().standardizedFileURL,
              directory.lastPathComponent.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$",
                                                 options: .regularExpression) != nil else { return nil }
        return directory.lastPathComponent
    }

    /// What to show when the log path carries no container: a Windows game under CrossOver, or any
    /// bare process. The bridge logs `process=`, which is the only identity available - so name the
    /// game from it rather than leaving the header blank, which is what it did before and which
    /// reads as "Netal has lost the game" rather than "this runtime is not sandboxed".
    static func fromProcessName(_ process: String, logURL: URL) -> GameIdentity? {
        let trimmed = process.hasSuffix(".exe") ? String(process.dropLast(4)) : process
        guard !trimmed.isEmpty else { return nil }
        // A Windows executable has no bundle to take an icon from, and the host that is actually
        // running it is CrossOver - so borrow its icon rather than inventing one or showing none.
        let host = process.hasSuffix(".exe")
            ? NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.codeweavers.CrossOver")
            : nil
        var icon: NSImage?
        if let host {
            icon = NSWorkspace.shared.icon(forFile: host.path)
            icon?.size = NSSize(width: 64, height: 64)
        }
        return GameIdentity(bundleID: "process:" + process, name: trimmed, icon: icon, appURL: nil)
    }

    /// Native sessions live either in a game's sandbox or Netal's own NativeGames directory.
    /// CrossOver process sessions remain identified by their logged process name.
    static func fromContainerLog(_ url: URL) -> GameIdentity? {
        let parts = url.pathComponents
        let bundleID: String
        if let native = nativeBundleID(from: url) {
            bundleID = native
        } else if let containers = parts.firstIndex(of: "Containers"), containers + 1 < parts.count {
            bundleID = parts[containers + 1]
            guard bundleID.contains(".") else { return nil }
        } else {
            return nil
        }

        let workspace = NSWorkspace.shared
        guard let appURL = workspace.urlForApplication(withBundleIdentifier: bundleID) else {
            // The container exists but the app is gone: still worth naming, just not picturing.
            return GameIdentity(bundleID: bundleID, name: bundleID, icon: nil, appURL: nil)
        }
        let icon = workspace.icon(forFile: appURL.path)
        icon.size = NSSize(width: 64, height: 64)
        let name = (try? appURL.resourceValues(forKeys: [.localizedNameKey]).localizedName)
            ?? appURL.deletingPathExtension().lastPathComponent
        return GameIdentity(bundleID: bundleID,
                            name: name.replacingOccurrences(of: ".app", with: ""),
                            icon: icon, appURL: appURL)
    }

    /// True while that game is actually running, so the panel can distinguish "this is the game we
    /// are configured for" from "this game is on screen right now".
    var isRunning: Bool {
        if let appURL {
            let canonical = appURL.resolvingSymlinksInPath().standardizedFileURL
            return NSWorkspace.shared.runningApplications.contains {
                $0.bundleURL?.resolvingSymlinksInPath().standardizedFileURL == canonical && !$0.isTerminated
            }
        }
        return !NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).isEmpty
    }
}
