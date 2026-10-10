import Foundation
import Combine

/// Follows the runtime log and the pause marker, and publishes a `RuntimeStatus`.
///
/// The log is append-only and truncated at the start of each session, and a separate process can
/// read it mid-write, so this reads incrementally from a byte offset, detects truncation or
/// replacement, and buffers any partial trailing line rather than parsing half a record.
@MainActor
final class RuntimeMonitor: ObservableObject {
    @Published private(set) var status = RuntimeStatus()
    @Published private(set) var isLive = false          // log grew recently
    @Published private(set) var isPauseRequested = false
    @Published private(set) var toggleError: String?
    /// The log file exists, whether or not anything has been parsed from it yet. The toggle depends
    /// on this rather than on parsed content: an empty log still means a runtime directory to write
    /// the marker into.
    @Published private(set) var logFileExists = false
    /// Transient confirmation for the Mark button, cleared once the runtime acknowledges the mark.
    @Published private(set) var markFeedback: String?
    /// Marks the runtime has recorded this session.
    @Published private(set) var marksRecorded: UInt64 = 0

    /// Logs that the container glob cannot find, injected rather than read from the build-generated
    /// defaults so this file stays compilable on its own for the tests. The CrossOver runtime is the
    /// reason it exists: it is not sandboxed into a container and writes a differently named log, so
    /// a Windows game running under it is invisible unless the app is told where to look.
    /// Running apps that link MetalFX and are not going through Netal. Capability, not use: the
    /// wording in the panel says "can use" for that reason.
    @Published private(set) var metalFXUnattached: [MetalFXApps.Candidate] = []

    /// A Windows game running in a bottle with no Netal session. Nil whenever a session is live, so
    /// the panel only raises it when there is genuinely nothing attached.
    @Published private(set) var crossoverUnattached: CrossOverGames.Candidate?
    private var crossoverPollCounter = 0

    var extraLogPaths: [URL] = []

    /// The log being followed. A var rather than a let because the runtime moves between games:
    /// each launch points the setting at that game's container, and the monitor should follow
    /// without the app being restarted.
    /// Rolling samples for the graphs. Published so the panel redraws as they arrive.
    @Published private(set) var history = MetricsHistory()
    /// The game the log belongs to, resolved from the container path. Recomputed only when that
    /// path changes: it hits the workspace, and an icon does not change mid-session.
    @Published private(set) var game: GameIdentity?
    /// Whether the DLSS model is present and matches its manifest. Checked from the filesystem
    /// rather than from the log, so the answer exists before a game has ever been launched -
    /// which is exactly when the user needs to be told they have not supplied one.
    @Published private(set) var model = ModelStatus()
    /// Supported FG model/math identities and the recorded runtime inventory must verify.
    @Published private(set) var fgModel = FGBundleStatus()
    /// Where the bundles live when they are not beside the log (the CrossOver runtime keeps them
    /// in the repository's work directory). Injected from the build defaults, like the extra logs.
    var modelBundleFallbacks: [URL] = []
    var fgBundleFallbacks: [URL] = []
    private var fgInspector = FGBundleInspector()
    /// Games running right now that Netal could upscale but is not, because they were not launched
    /// through it. Surfaced because the alternative - showing idle and saying nothing - is exactly
    /// what makes this look like a bug.
    @Published private(set) var unattached: [GameIdentity] = []
    private var inspector = ModelInspector()
    private(set) var logURL: URL
    /// Re-resolves the log path. Nil in tests, which pin a single file on purpose.
    private let logURLProvider: (() -> URL)?
    /// What the setting said at the previous poll, so a *change* to it can be told apart from it
    /// merely still holding an old value.
    private var lastConfigured: URL?

    /// Decides which state changes are worth a system notification. The rules live in
    /// NotificationPolicy so they can be tested; delivery is injected, so a monitor under test
    /// never touches UserNotifications.
    private var policy = NotificationPolicy()
    private var previousState: RuntimeState = .noRuntime
    var onNotifications: (([NetalNotification]) -> Void)?
    var pauseMarkerURL: URL { logURL.deletingLastPathComponent().appendingPathComponent("re9-sr-pause") }
    /// Written when the person playing sees an artifact. The runtime consumes it on its next
    /// encode check and stamps the current frame into the log.
    var markMarkerURL: URL { logURL.deletingLastPathComponent().appendingPathComponent("re9-sr-mark") }

    /// Without a timestamp in the log, "live" means it grew within this window.
    private let liveWindow: TimeInterval = 5

    private var parser = StatusParser()
    private var offset: UInt64 = 0
    private var identity: (dev: dev_t, ino: ino_t)?
    /// First bytes of the file. A new session truncates the log in place, keeping the inode, so
    /// `size < offset` only catches it when the new session is still shorter than the old one at
    /// poll time. Re-checking the head catches the rest: the first line carries an ASLR'd address,
    /// so it differs between runs.
    private var headFingerprint: Data?
    private var partialLine = Data()
    private var lastGrowth = Date.distantPast
    private var timer: Timer?

    /// Guards against an unbounded partial line if the log is ever not newline-terminated.
    private static let maxPartialLine = 64 * 1024
    /// Bytes compared to detect a restarted session.
    private static let headBytes = 512
    /// Most to read in one poll, so opening against a large existing log cannot stall the UI. The
    /// remainder is picked up by the next poll.
    private static let maxReadPerPoll = 1 << 20

    init(logURL: URL, startImmediately: Bool = true, logURLProvider: (() -> URL)? = nil) {
        self.logURL = logURL
        self.logURLProvider = logURLProvider
        self.game = GameIdentity.fromContainerLog(logURL)
        // Start following at construction: the menu-bar icon must reflect state whether or not the
        // menu has ever been opened.
        if startImmediately { start() }
    }

    func start(interval: TimeInterval = 1.0) {
        stop()
        poll()
        let timer = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.poll() }
        }
        // .common keeps it firing while a menu is open, which is exactly when it is being read.
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - Pause marker

    /// Netal on/off without relaunching the game: while the marker exists the runtime passes every
    /// encode through to MetalFX. Resuming forces a history reset on the runtime side.
    func setEnabled(_ enabled: Bool) {
        do {
            if enabled {
                if FileManager.default.fileExists(atPath: pauseMarkerURL.path) {
                    try FileManager.default.removeItem(at: pauseMarkerURL)
                }
            } else {
                let directory = pauseMarkerURL.deletingLastPathComponent()
                guard FileManager.default.fileExists(atPath: directory.path) else {
                    // Writing the marker into a directory the runtime is not watching would look
                    // like success and do nothing, so refuse instead.
                    throw CocoaError(.fileNoSuchFile)
                }
                try Data().write(to: pauseMarkerURL, options: .atomic)
            }
            toggleError = nil
        } catch {
            toggleError = enabled ? "Could not resume: \(error.localizedDescription)"
                                  : "Could not pause: \(error.localizedDescription)"
        }
        refreshPauseMarker()
    }

    /// Mark the moment an artifact was seen. White flashes are rare and unpredictable, so a fixed
    /// dump window cannot catch one; this records the frame the runtime was on when the button was
    /// pressed. The runtime deletes the marker once it has recorded it, so each press counts once.
    func markArtifact(note: String = "menubar") {
        guard logFileExists else {
            toggleError = "No runtime is running, so there is no frame to mark."
            return
        }
        do {
            try Data(note.utf8).write(to: markMarkerURL, options: .atomic)
            markFeedback = "Marked frame \(status.framesScheduled)"
            toggleError = nil
        } catch {
            markFeedback = nil
            toggleError = "Could not mark: \(error.localizedDescription)"
        }
    }

    // MARK: - Live control

    /// The control file the runtime polls, beside the pause marker. Key=value so the runtime can
    /// parse it without a JSON dependency in the injected code.
    var controlURL: URL { logURL.deletingLastPathComponent().appendingPathComponent("netal-control") }

    /// How the network is lowered. `.hardware` is what ships; `.reference` is the scalar path the
    /// receipts are checked against, kept reachable because it is the only way to see what the
    /// hardware lowering is being compared to - and worth a warning, because it runs at a fraction
    /// of the speed.
    enum Lowering: String, CaseIterable, Identifiable {
        case hardware = "_hw7"
        case reference = ""
        var id: String { rawValue }
        var label: String { self == .hardware ? "Fast" : "Reference" }
        var detail: String {
            self == .hardware
                ? "Accelerated arithmetic. Output can differ from the reference."
                : "Slower translated arithmetic for comparison; acceleration is disabled."
        }
    }

    @Published private(set) var lowering: Lowering = .hardware
    @Published private(set) var batchedPipelines = true
    /// Frame generation on/off. Off hands every interpolator encode back to MetalFX; the bridge
    /// logs fg_paused/fg_resumed and resets the network's history when it comes back.
    @Published private(set) var frameGenerationEnabled = true

    func setLowering(_ value: Lowering) {
        lowering = value
        batchedPipelines = value == .hardware
        writeControl()
    }
    func setBatchedPipelines(_ value: Bool) {
        batchedPipelines = value
        if value { lowering = .hardware }
        writeControl()
    }
    func setFrameGenerationEnabled(_ value: Bool) { frameGenerationEnabled = value; writeControl() }

    /// Rewrites the whole file rather than patching a line: it is three keys, and a partial write
    /// that the runtime reads mid-update would apply half a change.
    private func writeControl() {
        let body = "suffix=\(lowering.rawValue)\npipeline=\(batchedPipelines ? 1 : 0)\nfg=\(frameGenerationEnabled ? 1 : 0)\n"
        do {
            let directory = controlURL.deletingLastPathComponent()
            guard FileManager.default.fileExists(atPath: directory.path) else {
                toggleError = "No runtime directory yet — start a game first."
                return
            }
            try body.write(to: controlURL, atomically: true, encoding: .utf8)
            toggleError = nil
        } catch {
            toggleError = "Could not write control file: \(error.localizedDescription)"
        }
    }

    /// Adopt whatever is already on disk, so the UI reflects a runtime someone else configured
    /// rather than snapping it back to this app's defaults.
    private func refreshControl() {
        guard let text = try? String(contentsOf: controlURL, encoding: .utf8) else { return }
        var suffix = lowering.rawValue
        var pipeline = batchedPipelines
        var fg = frameGenerationEnabled
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { continue }
            if parts[0] == "suffix" { suffix = String(parts[1]) }
            if parts[0] == "pipeline" { pipeline = parts[1] != "0" }
            if parts[0] == "fg" { fg = parts[1] != "0" }
        }
        if let parsed = Lowering(rawValue: suffix) { lowering = parsed }
        if lowering == .reference { pipeline = false }  // same canonical rule as the bridge
        if pipeline != batchedPipelines { batchedPipelines = pipeline }
        if fg != frameGenerationEnabled { frameGenerationEnabled = fg }
    }

    private func refreshPauseMarker() {
        refreshControl()
        let requested = FileManager.default.fileExists(atPath: pauseMarkerURL.path)
        if requested != isPauseRequested { isPauseRequested = requested }
    }

    // MARK: - Log following

    /// Internal rather than private so tests can drive it deterministically instead of waiting
    /// on the timer.
    /// Hand any newsworthy transition to the delivery hook, if one is set.
    private func announce() {
        guard let onNotifications else { previousState = status.state; return }
        policy.reset(session: logURL.path)
        let due = policy.evaluate(previous: previousState, status: status,
                                  game: game?.name, modelReady: model.isReady)
        previousState = status.state
        if !due.isEmpty { onNotifications(due) }
    }

    /// Switch to a different session. Treated exactly like a restarted one, because that is what
    /// it is: everything parsed so far belongs to a different runtime.
    private func follow(_ url: URL) {
        guard url != logURL else { return }
        logURL = url
        game = GameIdentity.fromContainerLog(url)
        history.reset()
        parser.reset()
        previousState = .noRuntime
        offset = 0
        partialLine.removeAll(keepingCapacity: true)
        identity = nil
        headFingerprint = nil
        lastGrowth = .distantPast
        isLive = false
        status = parser.status
    }

    func poll() {
        // Decide which session to follow, before reading anything.
        //
        // Two sources, and the order matters. The setting wins when it *changes*, because that means
        // a launcher just started a game and is telling us where to look. Otherwise discovery wins,
        // because a setting written at some past launch is not evidence about what is running now -
        // which is exactly how the app ended up watching a dead log while a game was playing.
        if let logURLProvider {
            let configured = logURLProvider()
            if configured != lastConfigured {
                lastConfigured = configured
                follow(configured)
            } else {
                follow(SessionDiscovery.resolve(current: logURL,
                                                extras: [configured] + extraLogPaths))
            }
        }
        // A non-sandboxed runtime has no container in its log path, so fromContainerLog gives
        // nothing and the header sat blank. The process name only arrives once the log has been
        // parsed - after follow() set `game` - so resolve it here instead.
        if game == nil, let process = status.processName {
            game = GameIdentity.fromProcessName(process, logURL: logURL)
        }
        refreshPauseMarker()
        // Reading process command lines means spawning `ps`, so this runs about once every ten
        // seconds rather than every poll, and only while nothing is attached. A bottled game that
        // has been running for an hour can wait a few seconds to be noticed.
        crossoverPollCounter += 1
        if isLive {
            if crossoverUnattached != nil { crossoverUnattached = nil }
            if !metalFXUnattached.isEmpty { metalFXUnattached = [] }
        } else if crossoverPollCounter % 10 == 0 {
            // A quiet log is not a missing bridge: a game sitting in a menu encodes nothing, and
            // this used to offer to restart a game that was attached all along. Attachment is
            // decided by identity - the pid the bridge logged, or failing that the executable it
            // named - never by whether frames happen to be flowing right now.
            var found = CrossOverGames.running()
            if let candidate = found, status.logPresent, Self.isAttached(candidate, to: status) { found = nil }
            if found != crossoverUnattached { crossoverUnattached = found }
            // Anything already attached is not worth offering, and neither is the game this monitor
            // is following - it is only unattached because it has stopped rendering.
            var attached = Set(unattached.map(\.bundleID))
            if let game { attached.insert(game.bundleID) }
            let capable = MetalFXApps.runningUnattached(attachedBundleIDs: attached)
            if capable != metalFXUnattached { metalFXUnattached = capable }
        }
        let running = SessionDiscovery.attachableButUnattached()
        if running.map(\.bundleID) != unattached.map(\.bundleID) { unattached = running }
        let sessionBundles = RuntimeSession.bundlePaths(for: logURL)
        let currentModel = inspector.status(forLog: logURL, fallbacks: (sessionBundles.map { [$0.sr] } ?? []) + modelBundleFallbacks)
        let currentFG = fgInspector.status(forLog: logURL, fallbacks: (sessionBundles.map { [$0.fg] } ?? []) + fgBundleFallbacks)
        if currentFG != fgModel { fgModel = currentFG }
        if currentModel != model { model = currentModel }

        guard let attributes = try? FileManager.default.attributesOfItem(atPath: logURL.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value else {
            // No log yet: a session has not started. Keep any state we already parsed rather than
            // flapping, but stop calling it live.
            if isLive { isLive = false }
            if logFileExists { logFileExists = false }
            if !status.logPresent { status = RuntimeStatus() }
            return
        }
        if !logFileExists { logFileExists = true }

        let dev = (attributes[.systemNumber] as? NSNumber)?.int32Value ?? 0
        let ino = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        let current = (dev: dev_t(dev), ino: ino_t(ino))

        // A new session truncates the log in place (same inode) or replaces it (different inode).
        // Either way everything parsed so far belongs to a dead session.
        let head = readHead()
        let replaced = identity.map { $0 != current } ?? false
        let restarted = headFingerprint.map { head != nil && $0 != head! } ?? false
        if replaced || restarted || size < offset {
            parser.reset()
            history.reset()
            offset = 0
            partialLine.removeAll(keepingCapacity: true)
            status = parser.status
        }
        identity = current
        if let head { headFingerprint = head }

        if size > offset {
            readNewBytes(upTo: size)
            // Opening an old log is not fresh game activity. Use the file's actual
            // write time, including when catching up across several bounded reads.
            lastGrowth = min(Date(), attributes[.modificationDate] as? Date ?? .distantPast)
        }
        let live = Date().timeIntervalSince(lastGrowth) < liveWindow
        if live != isLive { isLive = live }
    }

    private func readHead() -> Data? {
        guard let handle = try? FileHandle(forReadingFrom: logURL) else { return nil }
        defer { try? handle.close() }
        return try? handle.read(upToCount: Self.headBytes)
    }

    private func readNewBytes(upTo size: UInt64) {
        guard let handle = try? FileHandle(forReadingFrom: logURL) else { return }
        defer { try? handle.close() }
        do {
            try handle.seek(toOffset: offset)
            let wanted = min(size - offset, UInt64(Self.maxReadPerPoll))
            guard let chunk = try handle.read(upToCount: Int(wanted)), !chunk.isEmpty else { return }
            offset += UInt64(chunk.count)

            var buffer = partialLine
            buffer.append(chunk)
            partialLine.removeAll(keepingCapacity: true)

            var lineStart = buffer.startIndex
            while let newline = buffer[lineStart...].firstIndex(of: UInt8(ascii: "\n")) {
                let lineData = buffer[lineStart..<newline]
                if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                    let previousPacing = parser.status.effectivePacing
                    parser.consume(line: line)
                    // Append here rather than once per poll: several samples can arrive between polls, and
                    // the graph should show all of them rather than only the newest.
                    if let pacing = parser.status.effectivePacing, pacing != previousPacing {
                        history.append(pacing)
                    }
                }
                lineStart = buffer.index(after: newline)
            }
            // Whatever follows the last newline is a record still being written.
            let remainder = buffer[lineStart...]
            partialLine = remainder.count > Self.maxPartialLine ? Data() : Data(remainder)

            if parser.status != status { status = parser.status }
            announce()
            if status.artifactMarks != marksRecorded {
                marksRecorded = status.artifactMarks
                markFeedback = nil        // the runtime acknowledged it; stop showing "Marked …"
            }
        } catch {
            // A read that fails mid-session is not fatal; the next poll re-reads from the same
            // offset. Resetting here would silently discard a whole session's state.
        }
    }

    // MARK: - Derived UI state

    /// The state to show. `isLive` refines the parsed state: a log that has stopped growing means
    /// the game exited, which the log itself cannot say because it carries no timestamps.
    /// The marker is written immediately but the runtime only checks it every 30 encodes, so
    /// there is a real window where we have asked for a pause that has not taken effect. Say so
    /// rather than claiming it is already paused.
    var isPausePending: Bool {
        isPauseRequested && status.state != .paused && isLive
    }

    /// The running bottled game is the one this log came from when the pids match; older bridges
    /// logged no pid, so the executable name breaks the tie then.
    static func isAttached(_ candidate: CrossOverGames.Candidate, to status: RuntimeStatus) -> Bool {
        if let pid = status.processID, let running = candidate.pid { return pid == running }
        if let process = status.processName { return process.caseInsensitiveCompare(candidate.executable) == .orderedSame }
        return false
    }

    var displayState: RuntimeState {
        if isPauseRequested { return .paused }
        if case .failed = status.state { return status.state }
        if !isLive {
            switch status.state {
            case .active, .armed: return .injected   // was running, no longer producing frames
            default: return status.state
            }
        }
        return status.state
    }
}
