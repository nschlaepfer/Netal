// Integration checks for RuntimeMonitor's file following: truncation, replacement, partial lines.
// Driven by explicit poll() calls rather than the timer, so the checks are deterministic.
import Foundation
import CryptoKit

@MainActor
func run() -> Int {
    var failures = 0
    func check(_ ok: Bool, _ what: String, line: UInt = #line) {
        print(ok ? "  ok    \(what)" : "  FAIL  \(what)   (line \(line))")
        if !ok { failures += 1 }
    }

    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("netal-monitor-\(getpid())")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let log = dir.appendingPathComponent("re9-live-metalfx.log")

    let monitor = RuntimeMonitor(logURL: log, startImmediately: false)

    print("no log yet")
    monitor.poll()
    check(monitor.status.state == .noRuntime, "state is noRuntime")
    check(!monitor.logFileExists, "log reported absent")

    print("\nopening an old session")
    let oldLog = dir.appendingPathComponent("old-session.log")
    try? "event=bridge_ready process=old.exe\nevent=sr_arm_ready bundle=/x\n"
        .write(to: oldLog, atomically: true, encoding: .utf8)
    try? FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-3600)],
                                          ofItemAtPath: oldLog.path)
    let historical = RuntimeMonitor(logURL: oldLog, startImmediately: false)
    historical.poll()
    check(historical.status.state == .armed, "old session remains inspectable")
    check(!historical.isLive && historical.displayState == .injected,
          "reading an old log does not report a live ready session")
    if let handle = try? FileHandle(forWritingTo: oldLog) {
        try? handle.seekToEnd()
        try? handle.write(contentsOf: Data("event=sr_arm_ready bundle=/x\n".utf8))
        try? handle.close()
    }
    historical.poll()
    check(historical.isLive, "a fresh runtime write establishes activity")

    print("\nsession A")
    try? ("event=encode_hook class=A original=0xAAAA\n"
        + "event=bridge_ready process=a.exe\n"
        + "event=sr_arm_ready bundle=/x\n"
        + "event=sr_profile_selected output=800x600 render=400x300\n"
        + "event=sr_exact_scheduled mode=sustained frame=120 reset=0 game_reset=0 jitter=0,0 fallbacks=3\n")
        .write(to: log, atomically: false, encoding: .utf8)
    monitor.poll()
    check(monitor.logFileExists, "log reported present")
    check(monitor.status.state == .active, "state is active")
    check(monitor.status.framesScheduled == 120, "frames from session A")
    check(monitor.status.fallbacks == 3, "fallbacks from session A")
    check(monitor.status.outputResolution == "800x600", "resolution from session A")

    // The hard case: a new session truncates in place and immediately writes MORE than the old
    // file held, so `size < offset` never fires and the inode never changes. Only the head
    // fingerprint catches it. This must truncate the way the launcher's `: > "$log"` does —
    // String.write replaces the file and would be caught by the inode check instead, proving
    // nothing.
    print("\nsession B truncates in place and is longer")
    var b = "event=encode_hook class=B original=0xBBBB\nevent=bridge_ready process=b.exe\n"
    b += "event=sr_profile_selected output=1920x1080 render=1280x720\n"
    b += "event=sr_arm_ready bundle=/x\n"
    for f in stride(from: 120, through: 960, by: 120) {
        b += "event=sr_exact_scheduled mode=sustained frame=\(f) reset=0 game_reset=0 jitter=0,0 fallbacks=0\n"
    }
    let inodeBefore = (try? FileManager.default.attributesOfItem(atPath: log.path))?[.systemFileNumber] as? NSNumber
    if let handle = try? FileHandle(forWritingTo: log) {
        try? handle.truncate(atOffset: 0)
        try? handle.write(contentsOf: Data(b.utf8))
        try? handle.close()
    }
    let inodeAfter = (try? FileManager.default.attributesOfItem(atPath: log.path))?[.systemFileNumber] as? NSNumber
    check(inodeBefore == inodeAfter, "precondition: truncated in place, inode unchanged")
    monitor.poll()
    check(monitor.status.framesScheduled == 960, "frames are session B's, not carried over")
    check(monitor.status.fallbacks == 0, "session A's fallbacks did not leak into B")
    // The discriminator: this line sits early in B, before session A's byte offset. Resuming from
    // that stale offset would skip it and leave session A's resolution in place.
    check(monitor.status.outputResolution == "1920x1080",
          "session B was re-read from the start, not resumed mid-file")

    print("\npartial trailing line")
    let partial = "event=sr_pacing frame=1080 window=120 mean_ms=16.0 fps=62.5 min_ms=15.0"
    if let handle = try? FileHandle(forWritingTo: log) {
        try? handle.seekToEnd(); try? handle.write(contentsOf: Data(partial.utf8)); try? handle.close()
    }
    monitor.poll()
    check(monitor.status.pacing?.frame != 1080, "an unterminated line is not parsed yet")
    // Completing the line makes it parseable on the next poll.
    if let handle = try? FileHandle(forWritingTo: log) {
        try? handle.seekToEnd()
        try? handle.write(contentsOf: Data(" max_ms=17.0 spikes_over_25ms=0 worst_ms_session=0 rss_mb=100\n".utf8))
        try? handle.close()
    }
    monitor.poll()
    check(monitor.status.pacing?.frame == 1080, "the completed line is parsed")
    check(monitor.status.pacing?.fps == 62.5, "and its fields are intact across the split")

    print("\npause marker round trip")
    monitor.setEnabled(false)
    check(FileManager.default.fileExists(atPath: monitor.pauseMarkerURL.path), "marker written")
    check(monitor.isPauseRequested, "pause reflected")
    check(monitor.displayState == .paused, "display state is paused")
    monitor.setEnabled(true)
    check(!FileManager.default.fileExists(atPath: monitor.pauseMarkerURL.path), "marker removed")
    check(!monitor.isPauseRequested, "resume reflected")
    check(monitor.toggleError == nil, "no toggle error")

    print("\nartifact mark round trip")
    // Refuse to mark when nothing is running: writing a marker no runtime will read would look
    // like success and do nothing.
    let saved = monitor.logFileExists
    check(saved, "precondition: a log exists")
    monitor.markArtifact(note: "test")
    check(FileManager.default.fileExists(atPath: monitor.markMarkerURL.path), "marker written")
    check(monitor.markFeedback != nil, "transient confirmation shown")
    check(monitor.toggleError == nil, "no error")
    // The runtime consumes the marker and logs it; the confirmation must then give way to the
    // recorded count, so the UI never claims a mark the runtime did not take.
    try? FileManager.default.removeItem(at: monitor.markMarkerURL)
    if let handle = try? FileHandle(forWritingTo: log) {
        try? handle.seekToEnd()
        try? handle.write(contentsOf: Data("event=sr_artifact_marked mark=1 frame=4242 note=test\n".utf8))
        try? handle.close()
    }
    monitor.poll()
    check(monitor.marksRecorded == 1, "runtime acknowledgement observed")
    check(monitor.status.lastMarkFrame == 4242, "frame captured for correlation")
    check(monitor.markFeedback == nil, "transient confirmation cleared once acknowledged")

    print("\nlog deleted mid-session")
    try? FileManager.default.removeItem(at: log)
    monitor.poll()
    check(!monitor.logFileExists, "log reported absent again")
    check(monitor.status.framesScheduled == 960, "last known state retained rather than flapping")
    // With no runtime, marking must refuse rather than drop a file nothing will read.
    monitor.markArtifact(note: "test")
    check(monitor.toggleError != nil, "marking with no runtime is refused, with a reason")
    check(!FileManager.default.fileExists(atPath: monitor.markMarkerURL.path), "no orphan marker left")

    // MARK: - Arithmetic mode controls
    // Selection is one coherent mode, not a reference suffix with accelerated pipelines still on.
    monitor.setLowering(.reference)
    var control = (try? String(contentsOf: monitor.controlURL, encoding: .utf8)) ?? ""
    check(monitor.lowering == .reference && !monitor.batchedPipelines, "Reference disables batched acceleration")
    check(control.contains("suffix=\npipeline=0"), "Reference writes the matching runtime controls")
    monitor.setLowering(.hardware)
    control = (try? String(contentsOf: monitor.controlURL, encoding: .utf8)) ?? ""
    check(monitor.batchedPipelines && control.contains("suffix=_hw7\npipeline=1"), "Fast restores the accelerated configuration")
    monitor.setLowering(.reference)
    monitor.setBatchedPipelines(true)
    check(monitor.lowering == .hardware, "explicit batching selects Fast instead of mislabeling Reference")
    try? "suffix=\npipeline=1\nfg=1\n".write(to: monitor.controlURL, atomically: true, encoding: .utf8)
    monitor.poll()
    check(monitor.lowering == .reference && !monitor.batchedPipelines, "contradictory external controls follow the bridge's Reference rule")

    // MARK: - Portable runtime session discovery
    let sessions = dir.appendingPathComponent("sessions")
    let session = sessions.appendingPathComponent("12345-100-200")
    try? FileManager.default.createDirectory(at: session, withIntermediateDirectories: true)
    let portableLog = session.appendingPathComponent("re9-live-metalfx.log")
    try? "event=bridge_ready\n".write(to: portableLog, atomically: true, encoding: .utf8)
    let config = session.appendingPathComponent("launch.conf")
    let configText = "netal.runtime.launch.v1\npid=12345\nuid=\(getuid())\nstart_sec=100\nstart_usec=200\nsr_bundle=/models/SR with spaces\nfg_bundle=/models/FG=bundle\n"
    try? configText.write(to: config, atomically: true, encoding: .utf8)
    try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: config.path)
    check(RuntimeSession.logs(in: sessions).map { $0.resolvingSymlinksInPath().path } == [portableLog.resolvingSymlinksInPath().path], "discovers per-process application-support logs")
    check(RuntimeSession.bundlePaths(for: portableLog, sessionRoot: sessions)?.sr.path == "/models/SR with spaces", "session config supplies model discovery paths")
    check(RuntimeSession.identity("12345--100-200") == nil, "malformed session identity rejected")
    check(RuntimeSession.identity("012345-100-200") == nil, "noncanonical session identity rejected")
    check(!RuntimeSession.isRunning(portableLog), "stale process lifetime is not reported running")
    try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: config.path)
    check(RuntimeSession.bundlePaths(for: portableLog, sessionRoot: sessions) == nil, "public-readable session config rejected")
    try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: config.path)
    try? (configText + "sr_bundle=/other\n").write(to: config, atomically: true, encoding: .utf8)
    try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: config.path)
    check(RuntimeSession.bundlePaths(for: portableLog, sessionRoot: sessions) == nil, "ambiguous duplicate session field rejected")

    // MARK: - Model verification
    //
    // "Verified" is a claim the app makes on screen, so it is checked rather than assumed: the two
    // model files are hashed against the manifest that shipped beside them, and present-but-wrong
    // has to be distinguishable from present and from absent.

    print("\nmodel verification")

    var inspector = ModelInspector(expectedModel: [
        "model/weights.bin": .init(bytes: 4096, sha256: SHA256.hash(data: Data(repeating: 0x11, count: 4096)).map { String(format: "%02x", $0) }.joined()),
        "model/constants.bin": .init(bytes: 2048, sha256: SHA256.hash(data: Data(repeating: 0x22, count: 2048)).map { String(format: "%02x", $0) }.joined()),
    ])
    let bundle = dir.appendingPathComponent("re9-sr-runtime-bundle")
    let modelDir = bundle.appendingPathComponent("model")

    check(inspector.status(forLog: log).state == .noBundle, "no bundle reads as unknown, not missing")

    try? FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    let weights = Data(repeating: 0x11, count: 4096)
    let constants = Data(repeating: 0x22, count: 2048)
    try? weights.write(to: modelDir.appendingPathComponent("weights.bin"))
    try? constants.write(to: modelDir.appendingPathComponent("constants.bin"))

    func sha(_ d: Data) -> String {
        SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
    }
    func writeManifest(weightsHash: String, constantsHash: String) {
        let manifest: [String: Any] = [
            "model": "DLSS_SR_Feature1_Quality_K",
            "schema": "neuralstudio.re9.exact_sr_immutable_runtime_bundle.v1",
            "files": [
                ["path": "model/weights.bin", "bytes": "4096", "sha256": weightsHash],
                ["path": "model/constants.bin", "bytes": "2048", "sha256": constantsHash],
            ],
        ]
        let data = try! JSONSerialization.data(withJSONObject: manifest)
        try? data.write(to: bundle.appendingPathComponent("manifest.json"))
    }

    writeManifest(weightsHash: sha(weights), constantsHash: sha(constants))
    var status = inspector.status(forLog: log)
    check(status.state == .verified, "matching hashes verify")
    check(status.name == "DLSS_SR_Feature1_Quality_K", "the model names itself")
    check(status.weightsBytes == 4096, "size reported")

    // The manifest truncates some digests, so a short expected value must still compare.
    writeManifest(weightsHash: String(sha(weights).prefix(40)),
                  constantsHash: String(sha(constants).prefix(40)))
    check(inspector.status(forLog: log).state == .verified, "a truncated manifest digest still verifies")

    // Manifest-only changes must invalidate verification even when both model files are untouched.
    writeManifest(weightsHash: "", constantsHash: sha(constants))
    check(inspector.status(forLog: log).state == .unverified, "empty digest cannot verify")
    writeManifest(weightsHash: String(sha(weights).prefix(1)), constantsHash: sha(constants))
    check(inspector.status(forLog: log).state == .unverified, "one-character digest cannot verify")
    try? Data("{}".utf8).write(to: bundle.appendingPathComponent("manifest.json"))
    check(inspector.status(forLog: log).state == .unverified, "missing manifest entries cannot verify")
    try? Data("broken-json".utf8).write(to: bundle.appendingPathComponent("manifest.json"))
    check(inspector.status(forLog: log).state == .unverified, "malformed manifest cannot verify")
    writeManifest(weightsHash: String(repeating: "0", count: 64), constantsHash: sha(constants))
    check(!inspector.status(forLog: log).isReady, "manifest-only hash change is rechecked")
    writeManifest(weightsHash: sha(weights), constantsHash: sha(constants))
    check(inspector.status(forLog: log).state == .verified, "restoring the manifest restores verification")

    // A model that is present but not the one the manifest describes is NOT the same as a model
    // that is absent, and the difference is the whole point of hashing it.
    try? Data(repeating: 0x33, count: 4096).write(to: modelDir.appendingPathComponent("weights.bin"))
    status = inspector.status(forLog: log)
    if case .modified(let files) = status.state {
        check(files == ["model/weights.bin"], "the wrong file is named: \(files)")
    } else {
        check(false, "a changed model reads as modified, got \(status.state)")
    }
    check(!status.isReady, "modified is not ready")

    // The app's supported identity must be independent of the selected bundle.
    writeManifest(weightsHash: sha(Data(repeating: 0x33, count: 4096)), constantsHash: sha(constants))
    check(inspector.status(forLog: log).state == .unverified, "rewritten SR manifest cannot authorize changed model bytes")
    try! weights.write(to: modelDir.appendingPathComponent("weights.bin"))
    writeManifest(weightsHash: sha(weights), constantsHash: sha(constants))
    check(inspector.status(forLog: log).isReady, "restored supported SR fixture verifies")
    var productionInspector = ModelInspector()
    check(!productionInspector.status(forLog: log).isReady, "production SR identities reject a self-consistent arbitrary fixture")
    let weightsURL = modelDir.appendingPathComponent("weights.bin")
    let originalDate = try! FileManager.default.attributesOfItem(atPath: weightsURL.path)[.modificationDate] as! Date
    try! Data(repeating: 0x44, count: 4096).write(to: weightsURL)
    try! FileManager.default.setAttributes([.modificationDate: originalDate], ofItemAtPath: weightsURL.path)
    check(inspector.status(forLog: log).state == .modified(["model/weights.bin"]), "same-size SR edit with restored mtime invalidates cache")
    try! weights.write(to: weightsURL)
    let manifestURL = bundle.appendingPathComponent("manifest.json")
    func rewriteManifest(_ edit: (inout [String: Any]) -> Void) {
        writeManifest(weightsHash: sha(weights), constantsHash: sha(constants))
        var value = try! JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        edit(&value)
        try! JSONSerialization.data(withJSONObject: value).write(to: manifestURL)
    }
    rewriteManifest { value in
        var rows = value["files"] as! [[String: Any]]; rows.append(rows[0]); value["files"] = rows
    }
    check(inspector.status(forLog: log).state == .unverified, "duplicate SR model row rejected")
    rewriteManifest { $0["model"] = "unsupported" }
    check(!inspector.status(forLog: log).isReady, "unsupported SR model name rejected")
    rewriteManifest { $0["schema"] = "unknown" }
    check(!inspector.status(forLog: log).isReady, "unknown SR manifest schema rejected")
    for invalidBytes: Any in [true, 4095, 4096.5, "04096", "4096x"] {
        rewriteManifest { value in
            var rows = value["files"] as! [[String: Any]]; rows[0]["bytes"] = invalidBytes; value["files"] = rows
        }
        check(!inspector.status(forLog: log).isReady, "invalid SR declared byte count rejected: \(invalidBytes)")
    }
    rewriteManifest { $0["schema"] = "netal.sr.runtime.v2" }
    check(inspector.status(forLog: log).isReady, "v2 model rows verify supported full hashes")
    rewriteManifest { value in
        value["schema"] = "netal.sr.runtime.v2"
        var rows = value["files"] as! [[String: Any]]; rows[0]["sha256"] = String(sha(weights).prefix(40)); value["files"] = rows
    }
    check(!inspector.status(forLog: log).isReady, "v2 cannot use a legacy truncated hash")
    writeManifest(weightsHash: sha(weights), constantsHash: sha(constants))
    let linkedWeights = dir.appendingPathComponent("original-weights")
    try! weights.write(to: linkedWeights)
    try! FileManager.default.removeItem(at: weightsURL)
    try! FileManager.default.createSymbolicLink(at: weightsURL, withDestinationURL: linkedWeights)
    check(!inspector.status(forLog: log).isReady, "matching SR model behind symlink rejected")
    try! FileManager.default.removeItem(at: weightsURL); try! weights.write(to: weightsURL)
    let linkedManifest = dir.appendingPathComponent("original-manifest")
    try! FileManager.default.copyItem(at: manifestURL, to: linkedManifest)
    try! FileManager.default.removeItem(at: manifestURL)
    try! FileManager.default.createSymbolicLink(at: manifestURL, withDestinationURL: linkedManifest)
    check(!inspector.status(forLog: log).isReady, "linked SR manifest rejected")
    try! FileManager.default.removeItem(at: manifestURL); try! FileManager.default.copyItem(at: linkedManifest, to: manifestURL)
    let movedModel = dir.appendingPathComponent("original-model")
    try! FileManager.default.moveItem(at: modelDir, to: movedModel)
    try! FileManager.default.createSymbolicLink(at: modelDir, withDestinationURL: movedModel)
    check(!inspector.status(forLog: log).isReady, "linked SR model directory rejected")
    try! FileManager.default.removeItem(at: modelDir); try! FileManager.default.moveItem(at: movedModel, to: modelDir)
    let linkedBundle = dir.appendingPathComponent("linked-sr-bundle")
    try! FileManager.default.createSymbolicLink(at: linkedBundle, withDestinationURL: bundle)
    check(!inspector.status(forLog: dir.appendingPathComponent("elsewhere/log"), fallbacks: [linkedBundle]).isReady,
          "linked SR bundle root rejected")
    check(inspector.status(forLog: dir.appendingPathComponent("elsewhere/log"), fallbacks: [bundle]).isReady,
          "valid SR fallback verifies")
    try! Data(repeating: 0x11, count: 2048).write(to: weightsURL)
    check(inspector.status(forLog: log).state == .modified(["model/weights.bin"]), "truncated SR bytes rejected")
    try! weights.write(to: weightsURL)

    try? FileManager.default.removeItem(at: modelDir.appendingPathComponent("weights.bin"))
    try? FileManager.default.removeItem(at: modelDir.appendingPathComponent("constants.bin"))
    check(inspector.status(forLog: log).state == .missing, "no model files reads as missing")

    // MARK: - Session discovery
    //
    // The app used to follow whatever NetalLogPath said, which the launcher writes at launch time.
    // Reopen the app while a game is running and it would faithfully watch a dead log and report
    // idle - right about the file, wrong about the session. These are the rules that fix that.

    print("\nsession discovery")

    let now = Date()
    func candidate(_ name: String, ageSeconds: TimeInterval, running: Bool) -> SessionDiscovery.Candidate {
        SessionDiscovery.Candidate(log: URL(fileURLWithPath: "/tmp/\(name)/netal-live.log"),
                                   modified: now.addingTimeInterval(-ageSeconds),
                                   bundleID: name, appIsRunning: running)
    }
    func url(_ name: String) -> URL { URL(fileURLWithPath: "/tmp/\(name)/netal-live.log") }

    let playing = candidate("playing", ageSeconds: 1, running: true)
    let quiet   = candidate("quiet",   ageSeconds: 900, running: false)
    let alsoLive = candidate("alsolive", ageSeconds: 2, running: false)

    check(SessionDiscovery.choose(current: url("playing"), from: [playing, quiet], now: now) == url("playing"),
          "a live session is never interrupted")
    check(SessionDiscovery.choose(current: url("quiet"), from: [playing, quiet], now: now) == url("playing"),
          "a dead log is abandoned for the one being written")
    check(SessionDiscovery.choose(current: url("quiet"), from: [quiet], now: now) == url("quiet"),
          "with nothing live, it stays put rather than flapping")

    // Brief gaps are normal - a pacing line lands about once a second, and a loading screen is
    // longer than that. Switching on one would flap.
    let pausedBriefly = candidate("paused", ageSeconds: 3, running: true)
    check(SessionDiscovery.choose(current: url("paused"), from: [pausedBriefly, alsoLive], now: now) == url("paused"),
          "a brief gap does not hand the session away")

    // A game that quit a moment ago must not out-rank one that is playing.
    let justQuit = candidate("justquit", ageSeconds: 1, running: false)
    let stale = candidate("stale", ageSeconds: 60, running: false)
    check(SessionDiscovery.choose(current: url("stale"), from: [justQuit, playing, stale], now: now) == url("playing"),
          "a running owner beats a merely recent file")

    // Opening the app with no configured log at all still finds something.
    check(SessionDiscovery.choose(current: url("nonexistent"), from: [quiet, stale], now: now) == url("stale"),
          "an unknown configured log falls back to the most recent real one")
    check(SessionDiscovery.choose(current: url("nonexistent"), from: [], now: now) == url("nonexistent"),
          "with no candidates at all it changes nothing")

    // MARK: - Notification policy
    //
    // The hard part is not what to say, it is what not to say: a notification that fires while
    // nothing changed is noise, and noise gets the whole app muted.

    print("\nnotification policy")

    var policy = NotificationPolicy()
    policy.reset(session: "session-a")

    var active = RuntimeStatus()
    active.state = .active
    active.outputResolution = "3840x2160"

    var first = policy.evaluate(previous: .armed, status: active, game: "Death Stranding", modelReady: true)
    check(first.count == 1 && first[0].kind == .engaged, "engaging says so once")
    check(first.first?.body.contains("3840x2160") == true, "and says where")

    // The whole point: a run that keeps running keeps quiet. Accumulated rather than asserted per
    // poll, so fifty silent polls are one result line and not fifty.
    var spoke = 0
    for _ in 0..<50 where !policy.evaluate(previous: .active, status: active,
                                           game: "Death Stranding", modelReady: true).isEmpty {
        spoke += 1
    }
    check(spoke == 0, "a steady session stays silent across 50 polls (spoke \(spoke) times)")

    // Health counters climb; the notification does not repeat.
    var degrading = active
    degrading.fallbacksReported = 1
    let fb = policy.evaluate(previous: .active, status: degrading, game: "Death Stranding", modelReady: true)
    check(fb.count == 1 && fb[0].kind == .fallback, "the first fallback is reported")
    degrading.fallbacksReported = 900
    check(policy.evaluate(previous: .active, status: degrading, game: "Death Stranding", modelReady: true).isEmpty,
          "900 more fallbacks stay silent")

    var flashing = degrading
    flashing.flashes = 3
    let fl = policy.evaluate(previous: .active, status: flashing, game: "Death Stranding", modelReady: true)
    check(fl.count == 1 && fl[0].kind == .flash, "a blown frame is reported once")

    // Pausing is something the user just did on purpose.
    var paused = RuntimeStatus(); paused.state = .paused
    check(policy.evaluate(previous: .active, status: paused, game: "Death Stranding", modelReady: true).isEmpty,
          "a deliberate pause is not news")

    // Losing a session is worth one line.
    var idle = RuntimeStatus(); idle.state = .armed
    let stopped = policy.evaluate(previous: .active, status: idle, game: "Death Stranding", modelReady: true)
    check(stopped.contains { $0.kind == .stopped }, "losing a running session is reported")

    // Attached with no model is a dead end the user has to act on.
    var noModel = NotificationPolicy(); noModel.reset(session: "session-b")
    let missing = noModel.evaluate(previous: .noRuntime, status: idle, game: "Lies of P", modelReady: false)
    check(missing.contains { $0.kind == .modelMissing }, "no model is reported")
    check(noModel.evaluate(previous: .armed, status: idle, game: "Lies of P", modelReady: false).isEmpty,
          "and not repeated")

    // A different game is a different session, and starts the conversation over.
    policy.reset(session: "session-c")
    let again = policy.evaluate(previous: .armed, status: active, game: "Lies of P", modelReady: true)
    check(again.contains { $0.kind == .engaged }, "a new session may announce again")
    check(again.first?.body.contains("3840x2160") == true, "with its own detail")

    // MARK: - Eye vitals
    //
    // The thresholds are the ones the rest of the product already uses - 16.7 ms is the 60 Hz line
    // on the graph, 25 ms is the spike threshold the runtime counts - so there is nothing new to
    // learn from the colour. These checks keep it that way.

    print("\nnetal vitals")

    func pacing(_ ms: Double) -> Pacing { var p = Pacing(); p.meanMs = ms; p.fps = 1000 / ms; return p }

    check(NetalVitals.from(state: .noRuntime, pacing: nil).condition == .idle, "no runtime is dark")
    check(NetalVitals.from(state: .paused, pacing: pacing(10)).condition == .idle,
          "a deliberate pause is dark, not green")
    check(NetalVitals.from(state: .active, pacing: pacing(12)).condition == .good, "12 ms is good")
    check(NetalVitals.from(state: .active, pacing: pacing(20)).condition == .fair,
          "20 ms is fair — past 60 Hz, short of the spike threshold")
    check(NetalVitals.from(state: .active, pacing: pacing(40)).condition == .poor, "40 ms is poor")
    check(NetalVitals.from(state: .failed("x"), pacing: pacing(10)).condition == .failed,
          "a failure shows as failed whatever the frame time")
    check(NetalVitals.from(state: .failed("x"), pacing: nil).period == nil,
          "an error does not pulse — a pulsing error reads as activity")

    // The pulse is a readout, not decoration: slower frames must give a slower pulse, and it must
    // stay inside a range the eye can follow.
    let fast = NetalVitals.from(state: .active, pacing: pacing(16.7)).period!
    let slow = NetalVitals.from(state: .active, pacing: pacing(40)).period!
    check(slow > fast, "slower frames pulse slower (\(fast) -> \(slow))")
    check(NetalVitals.from(state: .active, pacing: pacing(2)).period! >= 0.9, "a fast game does not strobe")
    check(NetalVitals.from(state: .active, pacing: pacing(500)).period! <= 3.0, "a stall does not stop it dead")
    check(NetalVitals.from(state: .active, pacing: nil).intensity > 0,
          "active with no pacing yet still lights up")

    print("\nattachment identity")
    do {
        var s = RuntimeStatus(); s.logPresent = true; s.processName = "re9.exe"; s.processID = 4242
        let same = CrossOverGames.Candidate(bottle: "Steam", appID: "1", name: "RE9", executable: "re9.exe", pid: 4242)
        let other = CrossOverGames.Candidate(bottle: "Steam", appID: "1", name: "RE9", executable: "re9.exe", pid: 99)
        check(RuntimeMonitor.isAttached(same, to: s), "same pid is attached even with a quiet log")
        check(!RuntimeMonitor.isAttached(other, to: s), "a different pid is a new, unattached process")
        var legacy = RuntimeStatus(); legacy.logPresent = true; legacy.processName = "re9.exe"
        check(RuntimeMonitor.isAttached(other, to: legacy), "without a logged pid the executable name decides")
        var none = RuntimeStatus(); none.logPresent = true
        check(!RuntimeMonitor.isAttached(other, to: none), "a log that never reported ready attaches nothing")
    }

    print(failures == 0 ? "\nall checks passed" : "\n\(failures) FAILURES")
    return failures
}

exit(Int32(MainActor.assumeIsolated { run() }))
