import Foundation

/// Drain output while the child runs. Waiting before reading can deadlock a verbose launcher;
/// combining readabilityHandler with a second reader can lose the script's final failure line.
enum RuntimeProcess {
    struct Result {
        let status: Int32
        let output: String
        var lastLine: String {
            output.split(separator: "\n").last.map(String.init) ?? "Runtime tool exited with status \(status)."
        }
        var passed: Bool {
            status == 0 && output.split(separator: "\n").contains { $0.trimmingCharacters(in: .whitespaces).hasPrefix("result=pass") }
        }
    }

    static func run(script: String, arguments: [String] = [],
                    progress: @escaping (String) -> Void = { _ in },
                    completion: @escaping (Result) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/zsh")
            process.arguments = [script] + arguments
            process.currentDirectoryURL = URL(fileURLWithPath: script).deletingLastPathComponent().deletingLastPathComponent()
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            do { try process.run() } catch {
                completion(Result(status: -1, output: error.localizedDescription)); return
            }
            var tail = Data()
            var pending = Data()
            while true {
                let data = pipe.fileHandleForReading.availableData
                if data.isEmpty { break }
                tail.append(data)
                if tail.count > 131_072 { tail.removeFirst(tail.count - 131_072) }
                pending.append(data)
                while let newline = pending.firstIndex(of: 10) {
                    progress(String(decoding: pending[..<newline], as: UTF8.self))
                    pending.removeSubrange(...newline)
                }
                if pending.count > 16_384 { pending.removeFirst(pending.count - 16_384) }
            }
            if !pending.isEmpty { progress(String(decoding: pending, as: UTF8.self)) }
            process.waitUntilExit()
            completion(Result(status: process.terminationStatus, output: String(decoding: tail, as: UTF8.self)))
        }
    }
}
