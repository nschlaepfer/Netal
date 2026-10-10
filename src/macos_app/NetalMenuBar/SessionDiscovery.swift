import Foundation
import AppKit

/// Finds the runtime session that is actually live, rather than trusting a setting.
///
/// The log path used to come from `NetalLogPath`, which the launcher writes when it starts a game.
/// That makes reattaching a configuration problem: open the app on its own, or open it after a
/// different game was launched, and it faithfully watches a file nothing is writing and reports
/// idle. It was right about the file and wrong about the session.
///
/// Every Netal runtime writes to `~/Library/Containers/<bundle id>/Data/netal/netal-live.log`,
/// because it is sandboxed into the game's container - so the live session can simply be found. The
/// setting stays as a preference and a starting point; discovery overrides it only when it finds a
/// log that is demonstrably being written and the configured one is not.
enum SessionDiscovery {
    static func nativeLogs(in root: URL = GameIdentity.nativeLogRoot) -> [URL] {
        guard let entries = try? FileManager.default.contentsOfDirectory(at: root,
                    includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: .skipsHiddenFiles) else { return [] }
        return entries.compactMap { entry in
            guard let values = try? entry.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]),
                  values.isDirectory == true, values.isSymbolicLink != true else { return nil }
            let log = entry.appendingPathComponent("netal-live.log")
            guard GameIdentity.nativeBundleID(from: log, root: root) != nil,
                  FileManager.default.fileExists(atPath: log.path) else { return nil }
            return log
        }
    }
    /// A log counts as live if it grew this recently. Long enough to survive a loading screen or a
    /// pause between pacing lines, short enough that a finished session stops winning.
    static let liveWindow: TimeInterval = 12
    /// How long the configured log must be quiet before discovery is allowed to move. Prevents
    /// flapping between two logs that are both being written.
    static let patience: TimeInterval = 6

    struct Candidate {
        var log: URL
        var modified: Date
        var bundleID: String?
        var appIsRunning: Bool

        func isLive(now: Date) -> Bool { now.timeIntervalSince(modified) < liveWindow }
    }

    /// Every runtime log on this machine, plus whatever the caller already knows about.
    ///
    /// The extra path covers the CrossOver and bare-process runtimes, which are not sandboxed into
    /// a container and so are not discoverable by the glob.
    static func candidates(including extras: [URL] = [], now: Date = Date()) -> [Candidate] {
        var found: [URL] = []
        let containers = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Containers")
        if let entries = try? FileManager.default.contentsOfDirectory(at: containers,
                                                                     includingPropertiesForKeys: nil) {
            for entry in entries {
                let log = entry.appendingPathComponent("Data/netal/netal-live.log")
                if FileManager.default.fileExists(atPath: log.path) { found.append(log) }
            }
        }
        found.append(contentsOf: nativeLogs())
        found.append(contentsOf: RuntimeSession.logs())
        for extra in extras where FileManager.default.fileExists(atPath: extra.path) &&
           !found.contains(where: { $0.standardizedFileURL == extra.standardizedFileURL }) {
            found.append(extra)
        }

        return found.compactMap { log in
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: log.path),
                  let modified = attributes[.modificationDate] as? Date else { return nil }
            let identity = GameIdentity.fromContainerLog(log)
            return Candidate(log: log, modified: modified, bundleID: identity?.bundleID,
                             appIsRunning: identity?.isRunning ?? RuntimeSession.isRunning(log))
        }
    }

    /// Games that Netal knows how to attach to, are running right now, and are NOT being upscaled.
    ///
    /// This is the case that looked like a bug and is not one. A game launched normally has no
    /// bridge in it - `DYLD_INSERT_LIBRARIES` is applied at exec and cannot be added to a process
    /// that is already running - so there is no session to find, and the panel correctly showed
    /// nothing. Correct, and useless: what the user needs to be told is that the game has to be
    /// started *through* Netal.
    ///
    /// "Knows how to attach to" means a container Netal has staged into before, which is precise:
    /// it names games this machine has actually run under Netal rather than guessing from a
    /// process list.
    static func attachableButUnattached(now: Date = Date()) -> [GameIdentity] {
        let containers = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Containers")
        let entries = (try? FileManager.default.contentsOfDirectory(at: containers,
                                                                        includingPropertiesForKeys: nil)) ?? []
        var out: [GameIdentity] = []
        for entry in entries {
            let staged = entry.appendingPathComponent("Data/netal/re9-sr-runtime-bundle/manifest.json")
            guard FileManager.default.fileExists(atPath: staged.path) else { continue }
            let log = entry.appendingPathComponent("Data/netal/netal-live.log")
            // Being written means it *is* attached; nothing to say.
            if let attributes = try? FileManager.default.attributesOfItem(atPath: log.path),
               let modified = attributes[.modificationDate] as? Date,
               now.timeIntervalSince(modified) < liveWindow { continue }
            guard let identity = GameIdentity.fromContainerLog(log), identity.isRunning else { continue }
            out.append(identity)
        }
        for log in nativeLogs() {
            if let attributes = try? FileManager.default.attributesOfItem(atPath: log.path),
               let modified = attributes[.modificationDate] as? Date,
               now.timeIntervalSince(modified) < liveWindow { continue }
            guard let identity = GameIdentity.fromContainerLog(log), identity.isRunning,
                  !out.contains(where: { $0.bundleID == identity.bundleID }) else { continue }
            out.append(identity)
        }
        return out
    }

    /// The log to follow, given the one currently being followed.
    ///
    /// Split from the filesystem scan so the decision can be tested without a real home directory -
    /// this is the part with the rules in it, and rules that cannot be exercised are rules that
    /// quietly rot.
    ///
    /// Returns `current` unless something else is clearly better, so a steady session is never
    /// interrupted. "Better" means the current log is not being written and another one is, with a
    /// running owner preferred over a merely recent file: a game that quit half a second ago should
    /// not out-rank one that is playing.
    static func choose(current: URL, from all: [Candidate], now: Date = Date()) -> URL {
        guard !all.isEmpty else { return current }
        let currentEntry = all.first { $0.log.standardizedFileURL == current.standardizedFileURL }

        if let currentEntry, currentEntry.isLive(now: now) { return current }
        // Give the current session a moment before looking elsewhere: a pacing line about once a
        // second means a brief gap is normal, and switching on one is how flapping starts.
        if let currentEntry, now.timeIntervalSince(currentEntry.modified) < patience { return current }

        let live = all.filter { $0.isLive(now: now) }
        let best = live.sorted { a, b in
            if a.appIsRunning != b.appIsRunning { return a.appIsRunning }
            return a.modified > b.modified
        }.first
        if let best, best.log.standardizedFileURL != current.standardizedFileURL { return best.log }

        // Nothing is live. If the configured log does not exist at all, fall back to the most recent
        // one that does, so a fresh install shows something rather than nothing.
        if currentEntry == nil,
           let recent = all.sorted(by: { $0.modified > $1.modified }).first {
            return recent.log
        }
        return current
    }

    static func resolve(current: URL, extras: [URL] = [], now: Date = Date()) -> URL {
        choose(current: current, from: candidates(including: extras, now: now), now: now)
    }
}
