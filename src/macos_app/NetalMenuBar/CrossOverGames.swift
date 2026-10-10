import AppKit
import Darwin

/// Windows games running under CrossOver, which the container scan cannot see.
///
/// `SessionDiscovery` finds native games by looking for a runtime log inside
/// `~/Library/Containers/<bundle id>/`. A CrossOver game is not sandboxed into a container and its
/// executable is not an app bundle, so it is invisible to that scan: the app would sit idle while a
/// Windows game ran without Netal, and say nothing.
///
/// There is no way to attach to one that is already running - `DYLD_INSERT_LIBRARIES` applies at
/// `exec`, and every shipping game is signed with the hardened runtime, which blocks the
/// inject-into-a-live-process route. So the useful thing is to notice, name it, and offer to start
/// it again with the bridge.
enum CrossOverGames {
    struct Candidate: Equatable {
        var bottle: String
        var appID: String
        var name: String
        var executable: String
        var pid: Int?
    }

    /// Command lines of running processes.
    ///
    /// `proc_pidpath` was the obvious route and it does not work here: it returns the real Unix
    /// binary, which for a Windows game is a wine loader inside CrossOver.app with no trace of the
    /// game in it. The Windows path - `C:\...\steamapps\common\<game>\game.exe` - exists only in
    /// the process arguments, so this asks `ps` for them. Callers throttle it; it is not something
    /// to run every poll.
    private static func commandLines() -> [(pid: Int?, command: String)] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-Ao", "pid=,command="]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (String(data: data, encoding: .utf8)?.split(separator: "\n") ?? []).map { line in
            let trimmed = line.drop(while: { $0 == " " })
            let parts = trimmed.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
            guard parts.count == 2, let pid = Int(parts[0]) else { return (nil, String(trimmed)) }
            return (pid, String(parts[1]))
        }
    }

    /// A Windows game currently running in a bottle, if there is one.
    ///
    /// Steam's own helpers live in the same bottle and are not games, so they are excluded by name
    /// rather than by guessing: a false positive here would offer to relaunch Steam itself.
    static func running() -> Candidate? {
        let ignored: Set<String> = ["steam.exe", "steamwebhelper.exe", "steamservice.exe",
                                    "steamerrorreporter.exe", "gameoverlayui.exe", "explorer.exe",
                                    "services.exe", "winedevice.exe", "plugplay.exe", "rpcss.exe",
                                    "conhost.exe", "start.exe", "wineboot.exe"]
        for (pid, path) in commandLines() {
            // A Windows command line, backslash separated, with arguments after the executable.
            guard let exeRange = path.range(of: ".exe", options: [.caseInsensitive]) else { continue }
            let windowsPath = String(path[..<exeRange.upperBound])
            guard windowsPath.lowercased().contains("\\steamapps\\common\\") else { continue }
            let executable = windowsPath.split(separator: "\\").last.map(String.init) ?? windowsPath
            guard !ignored.contains(executable.lowercased()) else { continue }
            // The command line does not name the bottle, so ask which bottles exist and match the
            // install directory inside each. One bottle is the overwhelmingly common case.
            guard let installDir = steamInstallDirectory(inWindowsPath: windowsPath) else { continue }
            guard let bottle = bottleContaining(installDirectory: installDir) else { continue }
            guard let (appID, name) = steamApp(bottle: bottle, installDirectory: installDir) else { continue }
            return Candidate(bottle: bottle, appID: appID, name: name, executable: executable, pid: pid)
        }
        return nil
    }

    private static func steamInstallDirectory(inWindowsPath path: String) -> String? {
        let parts = path.split(separator: "\\").map(String.init)
        guard let index = parts.firstIndex(where: { $0.caseInsensitiveCompare("common") == .orderedSame }),
              index + 1 < parts.count else { return nil }
        return parts[index + 1]
    }

    private static func bottleContaining(installDirectory: String) -> String? {
        let bottles = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/CrossOver/Bottles")
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: bottles.path) else { return nil }
        for name in names where steamApp(bottle: name, installDirectory: installDirectory) != nil {
            return name
        }
        return nil
    }

    /// Resolve the Steam app id and display name from the bottle's own manifests, so the relaunch
    /// uses the id Steam expects rather than one inferred from a folder name.
    private static func steamApp(bottle: String, installDirectory: String) -> (String, String)? {
        let steamApps = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/CrossOver/Bottles")
            .appendingPathComponent(bottle)
            .appendingPathComponent("drive_c/Program Files (x86)/Steam/steamapps")
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: steamApps.path) else { return nil }
        for entry in entries where entry.hasPrefix("appmanifest_") && entry.hasSuffix(".acf") {
            guard let text = try? String(contentsOf: steamApps.appendingPathComponent(entry), encoding: .utf8),
                  text.contains("\"\(installDirectory)\"") else { continue }
            let appID = entry.replacingOccurrences(of: "appmanifest_", with: "")
                             .replacingOccurrences(of: ".acf", with: "")
            let name = value(of: "name", in: text) ?? installDirectory
            return (appID, name)
        }
        return nil
    }

    private static func value(of key: String, in acf: String) -> String? {
        for line in acf.split(separator: "\n") where line.contains("\"\(key)\"") {
            let quoted = line.split(separator: "\"").map(String.init).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            if quoted.count >= 2, quoted[0] == key { return quoted[1] }
        }
        return nil
    }
}
