import Foundation

/// Resolve a developer runtime at launch rather than embedding the builder's home directory.
/// A copied app starts in setup mode until a runtime folder is selected. This is deliberately
/// not a claim that the research runtime is already a self-contained downloadable product.
enum NetalBuildDefaults {
    static var runtimeRoot: URL {
        resolve(environment: ProcessInfo.processInfo.environment,
                selected: UserDefaults.standard.string(forKey: "NetalRuntimeRoot"),
                app: Bundle.main.bundleURL,
                home: FileManager.default.homeDirectoryForCurrentUser)
    }

    static func resolve(environment: [String: String], selected: String?, app: URL, home: URL) -> URL {
        if let path = environment["NETAL_RUNTIME_ROOT"], !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        if let path = selected, !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        // Source build: <checkout>/build/netal-menubar/Netal.app.
        let checkout = app.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: checkout.appendingPathComponent("scripts/netal-native.sh").path) {
            return checkout
        }
        return home.appendingPathComponent("Library/Application Support/Netal/Runtime")
    }

    private static func path(_ relative: String) -> String { runtimeRoot.appendingPathComponent(relative).path }
    static var logPath: String { path("build/re9_crossover/re9-live-metalfx.log") }
    static var crossoverLogPath: String { logPath }
    static var launcherPath: String { path("scripts/netal-native.sh") }
    static var crossoverLauncherPath: String { path("scripts/crossover-exact-sr.sh") }
    static var ingestPath: String { path("scripts/netal-dlss-ingest.sh") }
    static var completeProfilePath: String { path("scripts/netal-complete-profile.sh") }
    static var srBundlePath: String { path("work/re9-sr-runtime-bundle") }
    static var fgBundlePath: String { path("work/re9-fg-runtime-bundle") }
}
