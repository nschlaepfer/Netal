import Foundation

/// Static eligibility only. A candidate is supported only after the runtime serves game frames.
struct NativeGameCompatibility: Decodable, Equatable {
    var canObserve: Bool
    var canLaunchModels: Bool
    var summary: String
    var detail: String
    var sandboxed: Bool
    var hasMetalFXEvidence: Bool
    var bundleID: String
    var executable: String
    var architectures: [String]

    private static func unavailable(_ detail: String) -> Self {
        Self(canObserve: false, canLaunchModels: false, summary: "Cannot check native compatibility",
             detail: detail, sandboxed: false, hasMetalFXEvidence: false,
             bundleID: "", executable: "", architectures: [])
    }

    /// Runs signing and bounded bundle-code inspection off the UI thread. Completion is on main.
    static func inspect(_ appURL: URL, completion: @escaping (Self) -> Void) {
        let script = NetalBuildDefaults.runtimeRoot.appendingPathComponent("scripts/netal-native-preflight.sh")
        guard FileManager.default.isReadableFile(atPath: script.path) else {
            DispatchQueue.main.async {
                completion(.unavailable("Choose a prepared Netal runtime folder in Settings."))
            }
            return
        }
        RuntimeProcess.run(script: script.path, arguments: [appURL.path, "--json"]) { result in
            let decoded = try? JSONDecoder().decode(Self.self, from: Data(result.output.utf8))
            let inspection = decoded ?? .unavailable(result.lastLine)
            DispatchQueue.main.async { completion(inspection) }
        }
    }
}
