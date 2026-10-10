import AppKit
import Foundation

/// Running apps that could use MetalFX but are not going through Netal.
///
/// Everything else the app knows about is retrospective: a native game is discovered by the runtime
/// log Netal itself wrote, which only exists after Netal launched it, and a bottled game by its
/// command line. Neither can see a game that has never been started through Netal — so the app sat
/// blind next to exactly the games a user would most want it to notice.
///
/// This asks a different question: which running apps *link* MetalFX? That is answerable without any
/// privilege and without touching the processes, because the answer is in the executable on disk.
///
/// Two honest limits, both of which shape how this is presented:
///
///   - linking is capability, not use. An app that links MetalFX may never create a scaler, so the
///     wording is "can use", never "is using". The bridge log remains the only evidence of actual
///     scaling, and only for a process Netal is inside.
///   - a Windows game under CrossOver is invisible here: the executable on disk is a wine loader
///     with no MetalFX in it. `CrossOverGames` covers that case separately.
enum MetalFXApps {
    struct Candidate: Equatable, Identifiable {
        var bundleID: String
        var name: String
        var icon: NSImage?
        var appURL: URL
        var id: String { bundleID }
        static func == (a: Candidate, b: Candidate) -> Bool { a.bundleID == b.bundleID }
    }

    /// Answers are cached per executable path and modification date. A game binary is hundreds of
    /// megabytes and does not change while it runs; re-reading it on every poll would be wasteful
    /// for an answer that cannot have changed.
    private static var cache: [String: (Date, Bool)] = [:]

    /// Static MetalFX evidence in each Mach-O slice, including a slice beyond the first 4 MiB.
    /// Dynamic loading can still escape inspection, so a negative result is unknown capability.
    static func linksMetalFX(_ executable: URL) -> Bool {
        let path = executable.path
        let modified = (try? FileManager.default.attributesOfItem(atPath: path)[.modificationDate] as? Date) ?? nil
        if let (stamp, answer) = cache[path], stamp == modified { return answer }

        var answer = false
        if let handle = try? FileHandle(forReadingFrom: executable) {
            defer { try? handle.close() }
            if let header = try? handle.read(upToCount: 1032) {
                for slice in slices(header: header, fileSize: (try? handle.seekToEnd()) ?? 0) {
                    try? handle.seek(toOffset: slice.offset)
                    if let head = try? handle.read(upToCount: Int(min(slice.length, 4 << 20))),
                       hasEvidence(head) { answer = true; break }
                }
            }
        }
        if let modified { cache[path] = (modified, answer) }
        return answer
    }

    private static func hasEvidence(_ data: Data) -> Bool {
        ["MetalFX.framework", "newTemporalScalerWithDevice:", "newFrameInterpolatorWithDevice:"]
            .contains { data.range(of: Data($0.utf8)) != nil }
    }

    static func slices(header: Data, fileSize: UInt64) -> [(offset: UInt64, length: UInt64)] {
        let bytes = [UInt8](header)
        guard bytes.count >= 4 else { return [] }
        let magic = Array(bytes.prefix(4))
        let thin: [[UInt8]] = [[0xcf, 0xfa, 0xed, 0xfe], [0xce, 0xfa, 0xed, 0xfe],
                               [0xfe, 0xed, 0xfa, 0xcf], [0xfe, 0xed, 0xfa, 0xce]]
        if thin.contains(magic) { return [(0, fileSize)] }
        let big = magic == [0xca, 0xfe, 0xba, 0xbe] || magic == [0xca, 0xfe, 0xba, 0xbf]
        let little = magic == [0xbe, 0xba, 0xfe, 0xca] || magic == [0xbf, 0xba, 0xfe, 0xca]
        guard (big || little), bytes.count >= 8 else { return [] }
        func integer(_ position: Int, _ size: Int) -> UInt64 {
            let range = bytes[position..<(position + size)]
            return (big ? Array(range) : Array(range.reversed())).reduce(0) { ($0 << 8) | UInt64($1) }
        }
        let count = integer(4, 4)
        let wide = magic == [0xca, 0xfe, 0xba, 0xbf] || magic == [0xbf, 0xba, 0xfe, 0xca]
        let rowSize = wide ? 32 : 20
        guard count <= 32, bytes.count >= 8 + Int(count) * rowSize else { return [] }
        return (0..<Int(count)).compactMap { index in
            let position = 8 + index * rowSize + 8
            let size = wide ? 8 : 4
            let offset = integer(position, size), length = integer(position + size, size)
            guard length >= 28, offset <= fileSize, length <= fileSize - offset else { return nil }
            return (offset, length)
        }
    }

    /// Unity and other engines can put MetalFX in a bundled library instead of the entry point.
    static func hasMetalFXEvidence(in appURL: URL) -> Bool {
        guard let executable = executableURL(for: appURL) else { return false }
        if linksMetalFX(executable) { return true }
        let manager = FileManager.default
        var inspected = 0, visited = 0
        for directory in ["Frameworks", "PlugIns", "MacOS"] {
            let root = appURL.appendingPathComponent("Contents/\(directory)")
            guard let entries = manager.enumerator(at: root,
                                                   includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
                                                   options: [.skipsHiddenFiles]) else { continue }
            while let entry = entries.nextObject() as? URL {
                visited += 1
                if visited > 4096 { return false }
                if entry.pathComponents.count - root.pathComponents.count > 7 {
                    entries.skipDescendants(); continue
                }
                guard let values = try? entry.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]),
                      values.isSymbolicLink != true else { entries.skipDescendants(); continue }
                guard values.isRegularFile == true, entry != executable,
                      let handle = try? FileHandle(forReadingFrom: entry) else { continue }
                let header = try? handle.read(upToCount: 1032)
                let size = (try? handle.seekToEnd()) ?? 0
                try? handle.close()
                guard let header, !slices(header: header, fileSize: size).isEmpty else { continue }
                inspected += 1
                if linksMetalFX(entry) { return true }
                if inspected >= 96 { return false }
            }
        }
        return false
    }

    /// Running apps that link MetalFX, minus anything already attached and minus Netal itself.
    ///
    /// `NSWorkspace` rather than a process scan: it gives bundle identity, a display name and an
    /// icon directly, and `.regular` filters out the agents and helpers that would otherwise
    /// dominate the list.
    static func runningUnattached(attachedBundleIDs: Set<String> = []) -> [Candidate] {
        let own = Bundle.main.bundleIdentifier
        var found: [Candidate] = []
        for app in NSWorkspace.shared.runningApplications {
            guard app.activationPolicy == .regular,
                  let bundleID = app.bundleIdentifier,
                  bundleID != own,
                  !attachedBundleIDs.contains(bundleID),
                  let appURL = app.bundleURL,
                  hasMetalFXEvidence(in: appURL)
            else { continue }
            let icon = NSWorkspace.shared.icon(forFile: appURL.path)
            icon.size = NSSize(width: 64, height: 64)
            let name = app.localizedName ?? appURL.deletingPathExtension().lastPathComponent
            found.append(Candidate(bundleID: bundleID, name: name, icon: icon, appURL: appURL))
        }
        return found
    }

    private static func executableURL(for appURL: URL) -> URL? {
        guard let bundle = Bundle(url: appURL), let executable = bundle.executableURL else { return nil }
        return FileManager.default.isReadableFile(atPath: executable.path) ? executable : nil
    }
}
