import Foundation
import Darwin

enum RuntimeSession {
    static var root: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Netal/Sessions")
    }

    static func logs(in directory: URL = root) -> [URL] {
        let manager = FileManager.default
        guard let children = try? manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil,
                                                               options: .skipsHiddenFiles) else { return [] }
        return children.compactMap { child in
            guard identity(child.lastPathComponent) != nil,
                  let attributes = try? manager.attributesOfItem(atPath: child.path),
                  attributes[.type] as? FileAttributeType == .typeDirectory else { return nil }
            let log = child.appendingPathComponent("re9-live-metalfx.log")
            guard let file = try? manager.attributesOfItem(atPath: log.path),
                  file[.type] as? FileAttributeType == .typeRegular else { return nil }
            return log
        }
    }

    static func identity(_ name: String) -> (pid: Int32, seconds: UInt64, microseconds: UInt64)? {
        let parts = name.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count == 3, let pid = Int32(parts[0]), pid > 0,
              let seconds = UInt64(parts[1]), let microseconds = UInt64(parts[2]), microseconds < 1_000_000 else { return nil }
        guard name == "\(pid)-\(seconds)-\(microseconds)" else { return nil }
        return (pid, seconds, microseconds)
    }

    static func isRunning(_ log: URL) -> Bool {
        guard let saved = identity(log.deletingLastPathComponent().lastPathComponent) else { return false }
        var info = proc_bsdinfo()
        let size = MemoryLayout<proc_bsdinfo>.stride
        guard proc_pidinfo(saved.pid, PROC_PIDTBSDINFO, 0, &info, Int32(size)) == Int32(size) else { return false }
        return info.pbi_uid == getuid() && info.pbi_start_tvsec == saved.seconds && info.pbi_start_tvusec == saved.microseconds
    }

    /// Only the app-owned session namespace is allowed to supply model folder discovery hints.
    static func bundlePaths(for log: URL, sessionRoot: URL = root) -> (sr: URL, fg: URL)? {
        let directory = log.deletingLastPathComponent()
        guard directory.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL.path == sessionRoot.resolvingSymlinksInPath().standardizedFileURL.path,
              let saved = identity(directory.lastPathComponent),
              let dirInfo = try? FileManager.default.attributesOfItem(atPath: directory.path),
              dirInfo[.type] as? FileAttributeType == .typeDirectory else { return nil }
        let path = directory.appendingPathComponent("launch.conf")
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path.path),
              attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.ownerAccountID] as? NSNumber)?.uint32Value == getuid(),
              let permissions = (attributes[.posixPermissions] as? NSNumber)?.uint16Value,
              permissions & 0o077 == 0,
              let size = (attributes[.size] as? NSNumber)?.intValue, size <= 65_536,
              let text = try? String(contentsOf: path, encoding: .utf8), !text.contains("\0") else { return nil }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.first == "netal.runtime.launch.v1" else { return nil }
        var values: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            let pair = line.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2, values[String(pair[0])] == nil, pair[1].utf8.count <= 4096 else { return nil }
            values[String(pair[0])] = String(pair[1])
        }
        guard values["pid"] == String(saved.pid), values["uid"] == String(getuid()),
              values["start_sec"] == String(saved.seconds), values["start_usec"] == String(saved.microseconds),
              let sr = values["sr_bundle"], sr.hasPrefix("/"),
              let fg = values["fg_bundle"], fg.hasPrefix("/") else { return nil }
        return (URL(fileURLWithPath: sr), URL(fileURLWithPath: fg))
    }
}
