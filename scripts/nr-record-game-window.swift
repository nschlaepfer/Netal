// Window-only visual evidence, not an NR backend. No audio or other app content.
import Foundation
import AppKit
import ScreenCaptureKit
import CoreMedia
import AVFoundation
import Darwin

final class RecordingWitness: NSObject, SCRecordingOutputDelegate, SCStreamDelegate, SCStreamOutput {
    private let lock = NSLock()
    private var started = false
    private var finished = false
    private var errorText: String?
    private var frames: [[String: Any]] = []
    func recordingOutputDidStartRecording(_ output: SCRecordingOutput) {
        lock.lock(); started = true; lock.unlock()
        print("RECORDING_STARTED"); fflush(stdout)
    }
    func recordingOutputDidFinishRecording(_ output: SCRecordingOutput) {
        lock.lock(); finished = true; lock.unlock()
        print("RECORDING_FINISHED"); fflush(stdout)
    }
    func recordingOutput(_ output: SCRecordingOutput, didFailWithError error: Error) {
        lock.lock(); errorText = error.localizedDescription; lock.unlock()
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        lock.lock(); errorText = error.localizedDescription; lock.unlock()
    }
    func stream(_ stream: SCStream, didOutputSampleBuffer sample: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sample.isValid else { return }
        let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]]
        let a = attachments?.first ?? [:]
        let status = a[.status] as? Int ?? -1
        var record: [String: Any] = ["pts_seconds": CMTimeGetSeconds(sample.presentationTimeStamp),
            "callback_uptime": ProcessInfo.processInfo.systemUptime, "status": status]
        if let display = a[.displayTime] as? UInt64 { record["display_time_ticks"] = display }
        lock.lock(); frames.append(record); lock.unlock()
    }
    func snapshot() -> (Bool, Bool, String?) {
        lock.lock(); defer { lock.unlock() }; return (started, finished, errorText)
    }
    func samples() -> [[String: Any]] {
        lock.lock(); defer { lock.unlock() }; return frames
    }
}

@main struct WindowRecording {
    @MainActor static var retained: RecordingWitness?
    @MainActor static func main() async {
        do {
            // Initialize the WindowServer connection without creating a window,
            // Dock item or stealing focus from the game.
            NSApplication.shared.setActivationPolicy(.prohibited)
            let args = CommandLine.arguments
            guard args.count == 4, let pid = Int32(args[1]), pid > 0,
                  let seconds = Double(args[3]), seconds >= 2, seconds <= 120 else {
                throw NSError(domain: "NRWindowCapture", code: 1, userInfo: [NSLocalizedDescriptionKey:
                    "usage: nr-record-game-window GAME_PID NEW_MOVIE.mp4 SECONDS(2..120)"])
            }
            let url = URL(fileURLWithPath: args[2])
            let metaURL = url.appendingPathExtension("json")
            let readyURL = url.appendingPathExtension("ready")
            for p in [url, metaURL, readyURL] where FileManager.default.fileExists(atPath: p.path) {
                throw NSError(domain: "NRWindowCapture", code: 2, userInfo: [NSLocalizedDescriptionKey: "refusing existing output: \(p.path)"])
            }
            let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
            for w in content.windows where w.owningApplication?.processID == pid ||
                w.owningApplication?.bundleIdentifier == "com.neowiz.game.lop" {
                print("GAME_WINDOW_CANDIDATE pid=\(w.owningApplication?.processID ?? 0) bundle=\(w.owningApplication?.bundleIdentifier ?? "") id=\(w.windowID) frame=\(w.frame)")
            }
            fflush(stdout)
            let windows = content.windows.filter {
                $0.owningApplication?.processID == pid &&
                $0.owningApplication?.bundleIdentifier == "com.neowiz.game.lop" &&
                $0.frame.width > 100 && $0.frame.height > 100
            }.sorted { $0.frame.width * $0.frame.height > $1.frame.width * $1.frame.height }
            guard let window = windows.first else {
                throw NSError(domain: "NRWindowCapture", code: 3, userInfo: [NSLocalizedDescriptionKey: "no visible Lies of P window for that PID"])
            }
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let config = SCStreamConfiguration()
            config.width = max(2, Int(filter.contentRect.width * CGFloat(filter.pointPixelScale)) / 2 * 2)
            config.height = max(2, Int(filter.contentRect.height * CGFloat(filter.pointPixelScale)) / 2 * 2)
            config.minimumFrameInterval = CMTime(value: 1, timescale: 120)
            config.queueDepth = 8
            config.showsCursor = false
            config.capturesAudio = false
            config.captureMicrophone = false
            config.ignoreShadowsSingleWindow = true
            let witness = RecordingWitness(); retained = witness
            let stream = SCStream(filter: filter, configuration: config, delegate: witness)
            try stream.addStreamOutput(witness, type: .screen, sampleHandlerQueue: DispatchQueue(label: "nr.window.capture.samples"))
            let settings = SCRecordingOutputConfiguration()
            settings.outputURL = url; settings.videoCodecType = .h264; settings.outputFileType = .mp4
            let recording = SCRecordingOutput(configuration: settings, delegate: witness)
            try stream.addRecordingOutput(recording)
            let wall = Date().timeIntervalSince1970
            let uptime = ProcessInfo.processInfo.systemUptime
            try await stream.startCapture()
            let startDeadline = Date().addingTimeInterval(10)
            while !witness.snapshot().0 && witness.snapshot().2 == nil && Date() < startDeadline {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            guard witness.snapshot().0, witness.snapshot().2 == nil else {
                throw NSError(domain: "NRWindowCapture", code: 4, userInfo: [NSLocalizedDescriptionKey: witness.snapshot().2 ?? "recording did not start"])
            }
            try Data("window-only recording ready; pid=\(pid)\n".utf8).write(to: readyURL, options: .withoutOverwriting)
            print("WINDOW_ONLY pid=\(pid) window=\(window.windowID) size=\(config.width)x\(config.height) no_audio=1 output=\(url.path)"); fflush(stdout)
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            try await stream.stopCapture()
            let endDeadline = Date().addingTimeInterval(10)
            while !witness.snapshot().1 && witness.snapshot().2 == nil && Date() < endDeadline {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            let state = witness.snapshot()
            let metadata: [String: Any] = ["schema": "netal.nr.game-window-recording.v1", "pid": pid,
                "window_id": window.windowID, "window_title": window.title ?? "", "width": config.width,
                "height": config.height, "requested_max_fps": 120, "audio_captured": false,
                "filter": "only the selected Lies of P window", "start_wall_unix": wall,
                "start_uptime": uptime, "finished": state.1, "error": state.2 ?? "", "samples": witness.samples(),
                "note": "Window compositor recording. Correlate actual recorded frames with game present events; this alone does not identify NR pixels or establish exact parity."]
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
                .write(to: metaURL, options: .withoutOverwriting)
            guard state.1, state.2 == nil else { throw NSError(domain: "NRWindowCapture", code: 5) }
            retained = nil
        } catch {
            fputs("recording failed: \(error)\n", stderr); exit(1)
        }
    }
}
