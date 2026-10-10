import Foundation
import AppKit

/// Starts a game with the native bridge. Window Capture does not need this restart workflow.
@MainActor
enum GameLauncher {
    private static var launching = false
    enum Outcome: Equatable {
        case launched
        case launcherMissing(String)
        case failed(String)
    }

    struct Options: Equatable {
        var superResolution = true
        var frameGeneration = true
        var observeOnly = false
        var steamAppID: String? = nil
    }

    static func nativeArguments(app: URL, options: Options = Options(), checkOnly: Bool = false) -> [String] {
        var arguments = [app.path]
        if options.observeOnly { arguments.append("--observe") }
        if !options.superResolution { arguments.append("--no-sr") }
        if !options.frameGeneration { arguments.append("--no-fg") }
        if let appID = options.steamAppID { arguments += ["--steam-app-id", appID] }
        if checkOnly { arguments.append("--check") }
        return arguments
    }

    static var isAvailable: Bool {
        FileManager.default.isExecutableFile(atPath: NetalBuildDefaults.launcherPath)
    }

    static var isCrossOverAvailable: Bool {
        FileManager.default.isExecutableFile(atPath: NetalBuildDefaults.crossoverLauncherPath)
    }

    /// Keep the selected bottle, executable and FG choice together all the way to the injector.
    static func crossOverArguments(bottle: String, appID: String, executable: String,
                                   frameGeneration: Bool) -> [String] {
        var arguments = ["--bottle", bottle, "--appid", appID, "--exe", executable,
                         "--sustained", "--mode", "hw", "--relaunch"]
        if frameGeneration { arguments.append("--fg") }
        return arguments
    }

    static func relaunchCrossOver(bottle: String, appID: String, executable: String,
                                  frameGeneration: Bool, completion: @escaping (Outcome) -> Void) {
        guard isCrossOverAvailable else {
            completion(.launcherMissing(NetalBuildDefaults.crossoverLauncherPath)); return
        }
        guard !launching else { completion(.failed("A game launch is already in progress.")); return }
        launching = true
        RuntimeProcess.run(script: NetalBuildDefaults.crossoverLauncherPath,
                           arguments: crossOverArguments(bottle: bottle, appID: appID,
                                                         executable: executable, frameGeneration: frameGeneration)) { result in
            Task { @MainActor in
                launching = false
                completion(result.status == 0 ? .launched : .failed(result.lastLine))
            }
        }
    }

    /// The caller confirms a restart. Validate all requested runtime parts before asking the game
    /// to quit, then wait for its actual exit so a save dialog cannot race the new launch.
    static func relaunch(_ game: GameIdentity, options: Options = Options(),
                         completion: @escaping (Outcome) -> Void) {
        guard isAvailable else {
            completion(.launcherMissing(NetalBuildDefaults.launcherPath)); return
        }
        guard let app = game.appURL else {
            completion(.failed("Cannot find \(game.name) on disk")); return
        }
        guard options.observeOnly || options.superResolution || options.frameGeneration else {
            completion(.failed("Choose Super Resolution, Frame Generation, or an observation run.")); return
        }

        guard !launching else { completion(.failed("A game launch is already in progress.")); return }
        launching = true

        RuntimeProcess.run(script: NetalBuildDefaults.launcherPath,
                           arguments: nativeArguments(app: app, options: options, checkOnly: true)) { preflight in
            Task { @MainActor in
                guard preflight.passed else {
                    launching = false
                    completion(.failed(preflight.lastLine)); return
                }
                let canonical = app.resolvingSymlinksInPath().standardizedFileURL
                let running = NSWorkspace.shared.runningApplications.filter {
                    $0.bundleURL?.resolvingSymlinksInPath().standardizedFileURL == canonical && !$0.isTerminated
                }
                for application in running {
                    guard application.terminate() else {
                        launching = false
                        completion(.failed("\(game.name) could not quit. Save and quit the game, then try again.")); return
                    }
                }
                let deadline = Date().addingTimeInterval(30)
                while running.contains(where: { !$0.isTerminated }) && Date() < deadline {
                    try? await Task.sleep(nanoseconds: 250_000_000)
                }
                guard running.allSatisfy(\.isTerminated) else {
                    launching = false
                    completion(.failed("\(game.name) is still open. Finish saving and quit it, then try again.")); return
                }
                RuntimeProcess.run(script: NetalBuildDefaults.launcherPath,
                                   arguments: nativeArguments(app: app, options: options)) { result in
                    Task { @MainActor in
                        launching = false
                        completion(result.status == 0 ? .launched : .failed(result.lastLine))
                    }
                }
            }
        }
    }
}
