import Foundation
import AppKit

/// Install a DLSS model from a DLL the user picks.
///
/// The work is done by `scripts/netal-dlss-ingest.sh`, deliberately: it is the same path a person
/// runs from a terminal, it is testable without a GUI, and it owns the part that matters — a model
/// is installed only if both model files match the supported reference hashes. Duplicating any of that here would mean two versions of the rule.
///
/// So this is a thin front end. Its whole job is to pick a file, report progress honestly, and
/// surface the script's own verdict rather than inventing one.
@MainActor
final class ModelIngest: ObservableObject {
    enum Phase: Equatable {
        case idle
        case running(String)
        case installed(String)
        case refused(String)
    }

    @Published private(set) var phase: Phase = .idle

    var unavailabilityReason: String? {
        let root = NetalBuildDefaults.runtimeRoot
        let manager = FileManager.default
        guard manager.isExecutableFile(atPath: NetalBuildDefaults.ingestPath) else {
            return "Choose a runtime folder in Settings."
        }
        guard manager.fileExists(atPath: root.appendingPathComponent("work/re9-sr-runtime-bundle/manifest.json").path) else {
            return "Set up the runtime before importing a model."
        }
        let captureDirs = ["build/model-capture", "vendor/dlss-capture"]
        let helpers = ["dlss-sr-cuda-capture.exe", "nvcuda.dll", "nvapi64.dll", "nvngx.dll", "nvcuda-forward-trace.dll"]
        guard captureDirs.contains(where: { directory in
            helpers.allSatisfy { manager.fileExists(atPath: root.appendingPathComponent(directory).appendingPathComponent($0).path) }
        }) else { return "The local model import helpers are missing." }
        return nil
    }
    var isAvailable: Bool { unavailabilityReason == nil }
    var isRunning: Bool { if case .running = phase { return true }; return false }

    /// Ask for a DLL, then install from it.
    func choose() {
        let panel = NSOpenPanel()
        panel.title = "Choose nvngx_dlss.dll"
        panel.message = "Netal ships no NVIDIA weights. Pick a DLSS runtime to take a model from — "
                      + "it stays on this machine."
        panel.allowedContentTypes = []
        panel.allowsOtherFileTypes = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        // Bringing the app forward first: the panel opens behind the menu-bar popover otherwise,
        // which looks like nothing happened.
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        run(url)
    }

    private func run(_ dll: URL) {
        phase = .running("Running the DLL's own loader…")

        RuntimeProcess.run(script: NetalBuildDefaults.ingestPath, arguments: [dll.path], progress: { line in
            let stage: String?
            if line.contains("extracting") { stage = "Reading the selected model…" }
            else if line.contains("verifying") { stage = "Checking the model…" }
            else { stage = nil }
            if let stage { Task { @MainActor in
                if self.isRunning { self.phase = .running(stage) }
            } }
        }, completion: { result in
            Task { @MainActor in
                self.phase = result.passed
                    ? .installed("Supported model bytes verified and installed.")
                    : .refused(result.lastLine.replacingOccurrences(of: "result=fail", with: "")
                        .trimmingCharacters(in: .whitespaces))
            }
        })
    }
}
