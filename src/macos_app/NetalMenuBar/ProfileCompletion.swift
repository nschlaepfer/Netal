import Foundation

/// Complete a half-generated pipeline profile for one output resolution.
///
/// A resolution with no shipped profile gets a synthesized pipeline set, and any layer without a
/// batched pipeline silently falls back to the transpiled per-window kernel — correct, and roughly
/// twice the GPU time. 1728x972 shipped with 6 of 11 layers and cost Resident Evil Requiem about
/// 17 ms a frame, with nothing anywhere saying so, because a fallback is a legitimate code path.
///
/// The work is `scripts/netal-complete-profile.sh` for the same reason ingest is a script: it is the
/// path a person runs from a terminal, it is testable without a GUI, and it owns the rule that a
/// short set is a failure rather than a warning. This is a thin front end that reports the script's
/// verdict instead of inventing one.
@MainActor
final class ProfileCompletion: ObservableObject {
    enum Phase: Equatable {
        case idle
        case running(String)
        case done(String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    var isAvailable: Bool {
        FileManager.default.isExecutableFile(atPath: NetalBuildDefaults.completeProfilePath)
    }
    var isRunning: Bool { if case .running = phase { return true }; return false }

    /// Generating runs the session once per layer and compiles pipelines, so it is tens of seconds,
    /// not instant. The phases are narrated rather than hidden behind a spinner.
    func complete(resolution: String) {
        guard !isRunning else { return }
        phase = .running("Generating pipelines for \(resolution)…")

        RuntimeProcess.run(script: NetalBuildDefaults.completeProfilePath, arguments: [resolution],
                           completion: { result in
            Task { @MainActor in
                self.phase = result.passed
                    ? .done("Profile complete — restart the game to use it.")
                    : .failed(String(result.lastLine.prefix(200)))
            }
        })
    }

    func dismiss() { phase = .idle }
}
