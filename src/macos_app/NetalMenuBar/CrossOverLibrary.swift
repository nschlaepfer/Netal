import AppKit
import Darwin

/// Installed titles, not just running Wine processes. Reads bottle-local Steam manifests and
/// registered CrossOver app wrappers; never evaluates commands from shortcut metadata.
struct CrossOverTarget: Equatable {
    var bottleURL: URL
    var installationURL: URL
    var steamAppID: String?
    var steamExecutable: String?
    var executables: [String]
    var wrapperURL: URL?
    var bottle: String { bottleURL.lastPathComponent }
    var id: String {
        let key = steamAppID.map { "steam:\($0)" } ?? "app:\(wrapperURL?.path ?? installationURL.path)"
        return "crossover:\(bottleURL.standardizedFileURL.resolvingSymlinksInPath().path):\(key)"
    }
}

/// A window belongs to a game only when its owning process identifies both the
/// selected bottle and a known executable inside that game's installation.
/// Launcher arguments, display names and Steam IDs alone are not capture identity.
enum CrossOverProcessIdentity {
    struct Identity {
        let executable: String
        let winePrefix: String
    }

    static func matches(processID: pid_t, target: CrossOverTarget) -> Bool {
        guard processID > 0, let identity = read(processID) else { return false }
        return matches(identity, target: target)
    }

    static func matches(_ identity: Identity, target: CrossOverTarget) -> Bool {
        let prefix = URL(fileURLWithPath: identity.winePrefix).standardizedFileURL.resolvingSymlinksInPath()
        guard identity.winePrefix.hasPrefix("/"),
              prefix == target.bottleURL.standardizedFileURL.resolvingSymlinksInPath(),
              let executable = CrossOverLibrary.windowsURL(identity.executable, bottle: prefix) else { return false }
        let actual = executable.standardizedFileURL.resolvingSymlinksInPath().path
        return target.executables.contains { relative in
            let expected = target.installationURL.appendingPathComponent(relative)
                .standardizedFileURL.resolvingSymlinksInPath().path
            return actual.caseInsensitiveCompare(expected) == .orderedSame
        }
    }

    private static func read(_ processID: pid_t) -> Identity? {
        var mib = [CTL_KERN, KERN_PROCARGS2, processID]
        // macOS process arguments are bounded; failure is an unmatched process.
        var bytes = [UInt8](repeating: 0, count: 1_048_576)
        var size = bytes.count
        let result = bytes.withUnsafeMutableBytes { buffer in
            sysctl(&mib, u_int(mib.count), buffer.baseAddress, &size, nil, 0)
        }
        guard result == 0, size >= MemoryLayout<Int32>.size, size <= bytes.count else { return nil }
        return decode(Array(bytes.prefix(size)))
    }

    // The only retained fields are argv[0] and WINEPREFIX. Other arguments and
    // environment values are never exposed to the UI, diagnostics or storage.
    static func decode(_ bytes: [UInt8]) -> Identity? {
        guard bytes.count >= 4 else { return nil }
        let argc = bytes.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }
        guard argc > 0, argc < 16_384 else { return nil }
        var offset = 4
        func nextString() -> String? {
            guard offset < bytes.count, let end = bytes[offset...].firstIndex(of: 0),
                  let value = String(bytes: bytes[offset..<end], encoding: .utf8) else { return nil }
            offset = end + 1
            return value
        }
        guard nextString() != nil else { return nil } // kernel executable path
        while offset < bytes.count && bytes[offset] == 0 { offset += 1 }
        var executable: String?
        for index in 0..<argc {
            guard let argument = nextString() else { return nil }
            if index == 0 { executable = argument }
        }
        guard let executable, !executable.isEmpty else { return nil }
        var prefix: String?
        while offset < bytes.count {
            guard let entry = nextString() else { return nil }
            if entry.hasPrefix("WINEPREFIX=") {
                guard prefix == nil else { return nil }
                prefix = String(entry.dropFirst("WINEPREFIX=".count))
            }
        }
        guard let prefix, !prefix.isEmpty else { return nil }
        return Identity(executable: executable, winePrefix: prefix)
    }
}

enum CrossOverLibrary {
    typealias Target = CrossOverTarget

    static var defaultBottlesRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/CrossOver/Bottles")
    }
    static var defaultApplicationsRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/CrossOver")
    }
    static var crossOverURL: URL {
        if let root = ProcessInfo.processInfo.environment["NETAL_CROSSOVER_ROOT"], !root.isEmpty {
            return URL(fileURLWithPath: root)
        }
        return URL(fileURLWithPath: "/Applications/CrossOver.app/Contents/SharedSupport/CrossOver")
    }

    static func discover(bottlesRoot: URL, applicationsRoot: URL, loadIcons: Bool) -> [GameLibrary.Candidate] {
        let fm = FileManager.default
        let bottles = ((try? fm.contentsOfDirectory(at: bottlesRoot, includingPropertiesForKeys: [.isDirectoryKey],
                                                    options: [.skipsHiddenFiles])) ?? [])
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        let wrappers = wrapperApps(in: applicationsRoot)
        var results: [String: GameLibrary.Candidate] = [:]
        for bottle in bottles.sorted(by: { $0.path < $1.path }) {
            let icons = menuIcons(in: bottle)
            let bottleWrappers = wrappers.filter { info($0)?["CXHelperAppBottleName"] as? String == bottle.lastPathComponent }
            let steamLocations = ["C:\\Program Files (x86)\\Steam", "C:\\Program Files\\Steam"]
            for steamPath in steamLocations {
                guard let steam = windowsURL(steamPath, bottle: bottle), fm.fileExists(atPath: steam.appendingPathComponent("steam.exe").path) else { continue }
                var libraries = [steam]
                for relative in ["steamapps/libraryfolders.vdf", "config/libraryfolders.vdf"] {
                    if let object = readVDF(steam.appendingPathComponent(relative))?["libraryfolders"]?.object {
                        for (key, value) in object.entries where isID(key) {
                            if let path = value.string ?? value.object?["path"]?.string,
                               let library = windowsURL(path, bottle: bottle) { libraries.append(library) }
                        }
                    }
                }
                var seen: Set<String> = []
                for library in libraries where seen.insert(library.resolvingSymlinksInPath().path).inserted {
                    let steamApps = library.appendingPathComponent("steamapps")
                    for manifest in (try? fm.contentsOfDirectory(at: steamApps, includingPropertiesForKeys: nil)) ?? [] {
                        let filename = manifest.lastPathComponent
                        guard filename.hasPrefix("appmanifest_"), filename.hasSuffix(".acf") else { continue }
                        let appID = String(filename.dropFirst(12).dropLast(4))
                        guard isID(appID), appID != "228980", let state = readVDF(manifest)?["AppState"]?.object,
                              state["appid"]?.string == appID, let name = state["name"]?.string, !excludedTitle(name),
                              let directory = state["installdir"]?.string, safeComponent(directory) else { continue }
                        if let flags = state["StateFlags"]?.string.flatMap(Int.init), flags & 4 == 0 { continue }
                        let installation = steamApps.appendingPathComponent("common").appendingPathComponent(directory)
                        guard (try? installation.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { continue }
                        let wrapper = bottleWrappers.first { wrapperName($0).caseInsensitiveCompare(name) == .orderedSame }
                        let target = Target(bottleURL: bottle, installationURL: installation, steamAppID: appID,
                                            steamExecutable: steamPath + "\\steam.exe", executables: gameExecutables(in: installation), wrapperURL: wrapper)
                        let icon = loadIcons ? (icons[name.lowercased()].flatMap { NSImage(contentsOf: $0) }
                                                ?? wrapper.map { NSWorkspace.shared.icon(forFile: $0.path) }) : nil
                        results[target.id] = GameLibrary.Candidate(bundleID: target.id, name: name, appURL: installation,
                            icon: icon, source: .crossOver, steamAppID: appID, crossOver: target)
                    }
                }
            }
            for wrapper in bottleWrappers {
                let name = wrapperName(wrapper)
                guard !excludedTitle(name), !results.values.contains(where: {
                    $0.crossOver?.bottleURL == bottle && $0.name.caseInsensitiveCompare(name) == .orderedSame
                }) else { continue }
                // A stale Steam URL shortcut must not resurrect an absent/uninstalled Steam game.
                if let path = info(wrapper)?["CrossOverHelperMenuPath"] as? String, path.lowercased().hasSuffix(".url") { continue }
                let target = Target(bottleURL: bottle, installationURL: wrapper, steamAppID: nil,
                                    steamExecutable: nil, executables: [], wrapperURL: wrapper)
                results[target.id] = GameLibrary.Candidate(bundleID: target.id, name: name, appURL: wrapper,
                    icon: loadIcons ? NSWorkspace.shared.icon(forFile: wrapper.path) : nil,
                    source: .crossOver, steamAppID: nil, crossOver: target)
            }
        }
        return Array(results.values)
    }

    /// Windows drive mappings are bottle-specific, including external Steam libraries.
    static func windowsURL(_ path: String, bottle: URL) -> URL? {
        let normalized = path.replacingOccurrences(of: "\\", with: "/")
        let chars = Array(normalized)
        guard chars.count >= 3, chars[1] == ":", chars[2] == "/",
              chars[0].isASCII, chars[0].isLetter else { return nil }
        let parts = normalized.dropFirst(3).split(separator: "/").map(String.init)
        guard parts.allSatisfy({ $0 != "." && $0 != ".." && !$0.contains("\0") }) else { return nil }
        let drive = String(chars[0]).lowercased()
        let mapping = bottle.appendingPathComponent("dosdevices/\(drive):")
        let base: URL
        if FileManager.default.fileExists(atPath: mapping.path) { base = mapping.resolvingSymlinksInPath() }
        else if drive == "c" { base = bottle.appendingPathComponent("drive_c") }
        else { return nil }
        return parts.reduce(base) { $0.appendingPathComponent($1) }.standardizedFileURL
    }

    private static func isID(_ text: String) -> Bool { !text.isEmpty && text.utf8.allSatisfy { (48...57).contains($0) } }
    private static func safeComponent(_ text: String) -> Bool {
        !text.isEmpty && text != "." && text != ".." && !text.contains("/") && !text.contains("\\") && !text.contains("\0")
    }
    private static func excludedTitle(_ title: String) -> Bool {
        let name = title.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return name.isEmpty || ["steam", "steam support center", "ubisoft connect", "rockstar games launcher",
            "epic games launcher", "gog galaxy", "battle.net"].contains(name)
            || name.contains("redistributable") || name.contains("dedicated server")
            || name.hasPrefix("uninstall ") || name.hasSuffix(" sdk")
    }
    private static func readVDF(_ path: URL) -> SteamVDF.Object? {
        guard let size = try? path.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 2_097_152,
              let text = try? String(contentsOf: path, encoding: .utf8) else { return nil }
        return SteamVDF.parse(text)
    }
    private static func info(_ app: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: app.appendingPathComponent("Contents/Info.plist")) else { return nil }
        return try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any]
    }
    private static func wrapperName(_ app: URL) -> String {
        (info(app)?["CFBundleName"] as? String) ?? app.deletingPathExtension().lastPathComponent
    }
    private static func wrapperApps(in root: URL) -> [URL] {
        var found: [URL] = []
        func walk(_ directory: URL, depth: Int) {
            guard depth <= 4 else { return }
            for path in (try? FileManager.default.contentsOfDirectory(at: directory,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])) ?? [] {
                let values = try? path.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                guard values?.isSymbolicLink != true else { continue }
                if path.pathExtension == "app" {
                    if let executable = info(path)?["CFBundleExecutable"] as? String, safeComponent(executable),
                       FileManager.default.isExecutableFile(atPath: path.appendingPathComponent("Contents/MacOS/\(executable)").path) {
                        found.append(path)
                    }
                } else if values?.isDirectory == true { walk(path, depth: depth + 1) }
            }
        }
        walk(root, depth: 0)
        return found
    }
    private static func menuIcons(in bottle: URL) -> [String: URL] {
        let file = bottle.appendingPathComponent("desktopdata/cxmenu/cxmenu_macosx.plist")
        guard let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 4_194_304,
              let data = try? Data(contentsOf: file),
              let object = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else { return [:] }
        var result: [String: URL] = [:]
        let root = bottle.resolvingSymlinksInPath().path + "/"
        func visit(_ object: [String: Any], depth: Int) {
            guard depth < 16 else { return }
            for (name, value) in object {
                guard let entry = value as? [String: Any] else { continue }
                if let path = entry["Icon"] as? String {
                    let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
                    if url.path.hasPrefix(root), let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                       size <= 4_194_304 { result[name.lowercased()] = url }
                }
                if let children = entry["Children"] as? [String: Any] { visit(children, depth: depth + 1) }
            }
        }
        visit(object, depth: 0)
        return result
    }
    private static func gameExecutables(in root: URL) -> [String] {
        var found: Set<String> = []; var visited = 0
        let skipped: Set<String> = ["support", "_commonredist", "redist", "redistributables", "installers", ".git"]
        func walk(_ directory: URL, depth: Int) {
            guard depth <= 4, visited < 20_000 else { return }
            for url in (try? FileManager.default.contentsOfDirectory(at: directory,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])) ?? [] {
                visited += 1; if visited > 20_000 { break }
                let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                guard values?.isSymbolicLink != true else { continue }
                let name = url.lastPathComponent.lowercased()
                if values?.isDirectory == true {
                    if !skipped.contains(name) { walk(url, depth: depth + 1) }
                } else if url.pathExtension.lowercased() == "exe",
                          !["crash", "report", "install", "unins", "redist", "dxsetup"].contains(where: { name.contains($0) }),
                          !["steam.exe", "steamservice.exe", "launcher.exe", "launcherpatcher.exe"].contains(name) {
                    found.insert(url.lastPathComponent)
                }
            }
        }
        walk(root, depth: 0)
        return found.sorted { $0.localizedStandardCompare($1) == .orderedAscending }
    }
}

/// Launch plans contain distinct arguments, never a shell command assembled from metadata.
@MainActor
enum CrossOverLaunch {
    private static var activeProcess: Process?
    struct Plan: Equatable {
        let program: String
        let arguments: [String]
    }

    nonisolated static func virtualDesktopPreferenceKey(for target: CrossOverLibrary.Target) -> String {
        "netal.gameLibrary.virtualDesktop." + target.id
    }

    /// This launch profile has been exercised with Steam Unity. Other games keep
    /// their normal Steam launcher until their desktop/launcher requirements are known.
    nonisolated static func virtualDesktopExecutable(for target: CrossOverLibrary.Target) -> URL? {
        guard target.steamAppID == "289650",
              let name = target.executables.first(where: { $0.caseInsensitiveCompare("ACU.exe") == .orderedSame }) else { return nil }
        let directory = target.installationURL.standardizedFileURL.resolvingSymlinksInPath()
        let executable = directory.appendingPathComponent(name).resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard executable.deletingLastPathComponent() == directory,
              FileManager.default.fileExists(atPath: executable.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else { return nil }
        return executable
    }

    nonisolated static func plan(for target: CrossOverLibrary.Target, virtualDesktop: Bool = false) -> Plan? {
        guard let id = target.steamAppID, !id.isEmpty, id.utf8.allSatisfy({ (48...57).contains($0) }),
              let steam = target.steamExecutable else { return nil }
        let prefix = ["--bottle", target.bottle, "--no-update", "--no-wait"]
        if virtualDesktop {
            guard let executable = virtualDesktopExecutable(for: target) else { return nil }
            // CrossOver converts existing native file paths to Windows paths. Keep
            // paths as separate arguments, including spaces and apostrophes.
            return Plan(program: "wine", arguments: prefix + ["--env", "SteamAppId=\(id)",
                "--workdir", target.installationURL.path, "explorer",
                "/desktop=NetalUnity,1280x720", executable.path, "-uplay_steam_mode"])
        }
        return Plan(program: "cxstart", arguments: prefix + [steam, "-applaunch", id])
    }

    nonisolated static func arguments(for target: CrossOverLibrary.Target) -> [String]? {
        plan(for: target)?.arguments
    }
    static func open(_ target: CrossOverLibrary.Target, virtualDesktop: Bool = false,
                     completion: @escaping (String?) -> Void) {
        guard activeProcess == nil else { completion("A CrossOver launch is already in progress."); return }
        if let plan = plan(for: target, virtualDesktop: virtualDesktop) {
            let executable = CrossOverLibrary.crossOverURL.appendingPathComponent("bin/\(plan.program)")
            guard FileManager.default.isExecutableFile(atPath: executable.path) else {
                completion("CrossOver could not be found. Open CrossOver and launch this game from its bottle."); return
            }
            let process = Process(); process.executableURL = executable; process.arguments = plan.arguments
            process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
            process.terminationHandler = { process in
                DispatchQueue.main.async { activeProcess = nil; completion(process.terminationStatus == 0 ? nil : "CrossOver could not launch the game (\(process.terminationStatus)). Open its bottle in CrossOver and try again.") }
            }
            activeProcess = process
            do { try process.run() } catch { activeProcess = nil; completion(error.localizedDescription) }
        } else if virtualDesktop {
            completion("Unity's executable is no longer available. Refresh the game library before using its virtual desktop.")
        } else if let wrapper = target.wrapperURL {
            NSWorkspace.shared.openApplication(at: wrapper, configuration: NSWorkspace.OpenConfiguration()) { _, error in
                DispatchQueue.main.async { completion(error?.localizedDescription) }
            }
        } else { completion("No launch shortcut is available for this title.") }
    }
}
