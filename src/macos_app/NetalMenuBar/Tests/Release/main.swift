import Foundation

@MainActor
func run() -> Int {
    var failures = 0
    func check(_ passed: Bool, _ description: String) {
        print("\(passed ? "ok" : "FAIL") \(description)")
        if !passed { failures += 1 }
    }
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try! FileManager.default.createDirectory(at: temporary.appendingPathComponent("scripts"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let copiedApp = temporary.appendingPathComponent("Applications/Netal.app")
    let fallback = NetalBuildDefaults.resolve(environment: [:], selected: nil, app: copiedApp, home: temporary)
    check(fallback.path == temporary.appendingPathComponent("Library/Application Support/Netal/Runtime").path,
          "relocated app uses user setup, not a build-machine path")
    check(NetalBuildDefaults.resolve(environment: ["NETAL_RUNTIME_ROOT": "/configured"], selected: "/selected",
                                    app: copiedApp, home: temporary).path == "/configured", "environment overrides selection")
    let arguments = GameLauncher.crossOverArguments(bottle: "Test Bottle", appID: "123", executable: "Game.exe", frameGeneration: true)
    check(arguments.contains("Game.exe") && arguments.contains("Test Bottle") && arguments.contains("--fg"),
          "launch preserves game, bottle and FG")
    check(!GameLauncher.crossOverArguments(bottle: "Steam", appID: "123", executable: "Game.exe", frameGeneration: false).contains("--fg"),
          "SR-only launch does not arm FG")
    let nativeApp = URL(fileURLWithPath: "/Games/Space Game.app")
    let srOnly = GameLauncher.nativeArguments(app: nativeApp,
        options: .init(superResolution: true, frameGeneration: false, observeOnly: false, steamAppID: "892970"), checkOnly: true)
    check(srOnly.first == nativeApp.path && srOnly.contains("--no-fg") && !srOnly.contains("--no-sr")
          && srOnly.contains("--check") && srOnly.suffix(3).contains("892970"),
          "native SR-only preflight preserves spaces and Steam app identity")
    let fgOnly = GameLauncher.nativeArguments(app: nativeApp,
        options: .init(superResolution: false, frameGeneration: true))
    check(fgOnly.contains("--no-sr") && !fgOnly.contains("--no-fg"), "native FG-only choice is independent")
    check(GameLauncher.nativeArguments(app: nativeApp, options: .init(observeOnly: true)).contains("--observe"),
          "native observation is explicit")
    let m3 = AppleChip(name: "Apple M3 Max", generation: 3, tier: .max)
    if let nr = DLSSFeature.all(chip: m3).first(where: { $0.id == "nr" }), case .experimental = nr.availability {
        check(true, "NR is experimental on M3; no invented M5 gate")
    } else { check(false, "NR experimental state") }

    var fgMode = FrameGeneration()
    fgMode.armed = true; fgMode.served = 1; fgMode.generated = 1; fgMode.lowering = .accelerated
    check(DLSSFeature.frameGenerationDetail(fgMode).contains("Accelerated"), "FG detail names accelerated arithmetic")
    check(DLSSFeature.frameGenerationHelp(fgMode).contains("can differ"), "accelerated tooltip does not promise exact output")
    fgMode.lowering = .reference
    check(DLSSFeature.frameGenerationDetail(fgMode).contains("Reference"), "reference selection is distinct from accelerated")
    fgMode.lowering = nil
    check(DLSSFeature.frameGenerationHelp(fgMode).contains("not recorded"), "missing arithmetic remains unknown")
    fgMode.served = 0; fgMode.generated = 0
    check(DLSSFeature.frameGenerationDetail(fgMode).contains("waiting"), "initialization does not imply interpolation is running")

    let script = temporary.appendingPathComponent("scripts/verbose.sh")
    try! "for i in {1..20000}; do print 'a verbose runtime line'; done\nprint 'result=fail final failure'; exit 7\n".write(to: script, atomically: true, encoding: .utf8)
    let finished = DispatchSemaphore(value: 0)
    let lock = NSLock()
    var result: RuntimeProcess.Result?
    RuntimeProcess.run(script: script.path) { output in
        lock.lock(); result = output; lock.unlock(); finished.signal()
    }
    check(finished.wait(timeout: .now() + 15) == .success, "verbose child does not block on a full pipe")
    lock.lock(); let captured = result; lock.unlock()
    check(captured?.status == 7 && captured?.lastLine == "result=fail final failure", "final failure and exit status survive output draining")
    check((captured?.output.utf8.count ?? Int.max) <= 131_072, "captured output is bounded")
    check(!RuntimeProcess.Result(status: 0, output: "unverified").passed, "exit zero alone is not verification")
    check(RuntimeProcess.Result(status: 0, output: "result=pass complete\n").passed, "explicit success is accepted")
    return failures
}
exit(Int32(MainActor.assumeIsolated { run() }))
