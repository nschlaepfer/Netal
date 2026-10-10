import AppKit

/// Installed games, independent of whether a game has already produced a Netal log.
/// Discovery only reads app bundles and Steam manifests; it never starts or changes a game.
enum GameLibrary {
    private static let addedApplicationsKey = "netal.gameLibrary.addedApplications"

    enum Source: String, Equatable {
        case steam = "Steam"
        case crossOver = "CrossOver"
        case macAppStore = "Mac App Store"
        case mac = "Mac"
    }

    struct Candidate: Identifiable, Equatable {
        /// A bundle identifier alone is not unique: the same game can live in several libraries.
        var id: String { crossOver?.id ?? appURL.standardizedFileURL.resolvingSymlinksInPath().path }
        var bundleID: String
        var name: String
        var appURL: URL
        var icon: NSImage?
        var source: Source
        var steamAppID: String?
        var crossOver: CrossOverTarget? = nil
        var sourceDetail: String { crossOver.map { "CrossOver · \($0.bottle)" } ?? source.rawValue }

        var identity: GameIdentity {
            GameIdentity(bundleID: bundleID, name: name, icon: icon, appURL: appURL)
        }

        static func == (lhs: Candidate, rhs: Candidate) -> Bool {
            lhs.id == rhs.id && lhs.bundleID == rhs.bundleID && lhs.name == rhs.name
                && lhs.source == rhs.source && lhs.steamAppID == rhs.steamAppID && lhs.crossOver == rhs.crossOver
        }
    }

    /// Run away from the UI thread: external Steam libraries can be on slower disks.
    static func discover() -> [Candidate] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return discover(applicationRoots: [URL(fileURLWithPath: "/Applications"),
                                           home.appendingPathComponent("Applications")],
                        steamRoot: home.appendingPathComponent("Library/Application Support/Steam"),
                        crossOverBottlesRoot: CrossOverLibrary.defaultBottlesRoot,
                        crossOverApplicationsRoot: CrossOverLibrary.defaultApplicationsRoot,
                        userAddedApps: (UserDefaults.standard.stringArray(forKey: addedApplicationsKey) ?? [])
                            .map { URL(fileURLWithPath: $0) })
    }

    /// Explicitly chosen apps need not declare a Games category in their metadata.
    static func candidate(for appURL: URL) -> Candidate? {
        if appInfo(appURL)?["CXHelperAppBottleName"] != nil {
            let path = appURL.resolvingSymlinksInPath().standardizedFileURL
            return CrossOverLibrary.discover(bottlesRoot: CrossOverLibrary.defaultBottlesRoot,
                applicationsRoot: appURL.deletingLastPathComponent(), loadIcons: true).first {
                    $0.crossOver?.wrapperURL?.resolvingSymlinksInPath().standardizedFileURL.path == path.path
                }
        }
        return candidate(for: appURL, loadIcons: true)
    }

    /// Retain only the app's location. The app is revalidated on every library refresh.
    static func add(_ appURL: URL) -> Candidate? {
        guard let game = candidate(for: appURL) else { return nil }
        var paths = UserDefaults.standard.stringArray(forKey: addedApplicationsKey) ?? []
        if !paths.contains(game.id) {
            paths.append(game.id)
            UserDefaults.standard.set(paths, forKey: addedApplicationsKey)
        }
        return game
    }

    /// Root and icon injection keep fixture checks isolated from installed games and the desktop.
    static func discover(applicationRoots: [URL], steamRoot: URL,
                         crossOverBottlesRoot: URL? = nil, crossOverApplicationsRoot: URL? = nil, userAddedApps: [URL] = [],
                         loadIcons: Bool = true) -> [Candidate] {
        var games: [String: Candidate] = [:]
        for app in userAddedApps {
            if let game = candidate(for: app, loadIcons: loadIcons) { games[game.id] = game }
        }
        for root in applicationRoots {
            for app in applications(in: root, maximumDepth: 3) {
                guard let info = appInfo(app), info["CXHelperAppBottleName"] == nil,
                      info["CFBundleIdentifier"] as? String != "com.epicgames.EpicGamesLauncher", let category = info["LSApplicationCategoryType"] as? String,
                      isGameCategory(category), let game = candidate(for: app, loadIcons: loadIcons) else {
                    continue
                }
                games[game.id] = game
            }
        }

        for library in steamLibraries(root: steamRoot) {
            let steamApps = library.appendingPathComponent("steamapps")
            let manifests = (try? FileManager.default.contentsOfDirectory(at: steamApps,
                includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
            for manifest in manifests.sorted(by: { $0.path < $1.path }) {
                let filename = manifest.lastPathComponent
                guard filename.hasPrefix("appmanifest_"), filename.hasSuffix(".acf") else { continue }
                let fileID = String(filename.dropFirst("appmanifest_".count).dropLast(".acf".count))
                guard isAppID(fileID), let document = readVDF(manifest),
                      let state = document["AppState"]?.object,
                      let appID = state["appid"]?.string, appID == fileID,
                      let directory = state["installdir"]?.string,
                      isInstallDirectory(directory) else { continue }
                let installation = steamApps.appendingPathComponent("common")
                    .appendingPathComponent(directory, isDirectory: true)
                let installedApps = applications(in: installation, maximumDepth: 4)
                    .compactMap { candidate(for: $0, loadIcons: loadIcons) }
                for var game in installedApps {
                    game.source = .steam
                    game.steamAppID = appID
                    if installedApps.count == 1, let name = state["name"]?.string,
                       !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        game.name = name
                    }
                    games[game.id] = game
                }
            }
        }
        if let bottles = crossOverBottlesRoot, let applications = crossOverApplicationsRoot {
            for game in CrossOverLibrary.discover(bottlesRoot: bottles, applicationsRoot: applications, loadIcons: loadIcons) {
                games[game.id] = game
            }
        }
        return games.values.sorted {
            let order = $0.name.localizedStandardCompare($1.name)
            return order == .orderedSame ? $0.id < $1.id : order == .orderedAscending
        }
    }

    static func candidate(for appURL: URL, loadIcons: Bool) -> Candidate? {
        let app = appURL.standardizedFileURL.resolvingSymlinksInPath()
        guard app.pathExtension.lowercased() == "app", let info = appInfo(app),
              let executable = info["CFBundleExecutable"] as? String,
              !executable.isEmpty, !executable.contains("/"), executable != ".", executable != "..",
              FileManager.default.isExecutableFile(atPath: app.appendingPathComponent("Contents/MacOS")
                .appendingPathComponent(executable).path) else { return nil }
        // Some native Steam games (including Hades) omit a bundle identifier. Keep them visible
        // using a path identity; consumers launch or capture them by appURL rather than bundle ID.
        let declaredID = (info["CFBundleIdentifier"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let bundleID = declaredID.flatMap { $0.isEmpty ? nil : $0 } ?? "path:\(app.path)"
        let name = [info["CFBundleDisplayName"] as? String, info["CFBundleName"] as? String]
            .compactMap { $0 }.first { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            ?? app.deletingPathExtension().lastPathComponent
        let fromStore = FileManager.default.fileExists(atPath:
            app.appendingPathComponent("Contents/_MASReceipt/receipt").path)
        let icon = loadIcons ? NSWorkspace.shared.icon(forFile: app.path) : nil
        icon?.size = NSSize(width: 64, height: 64)
        return Candidate(bundleID: bundleID, name: name, appURL: app, icon: icon,
                         source: fromStore ? .macAppStore : .mac, steamAppID: nil)
    }

    private static func appInfo(_ app: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: app.appendingPathComponent("Contents/Info.plist")),
              let info = try? PropertyListSerialization.propertyList(from: data, format: nil)
                as? [String: Any] else { return nil }
        return info
    }

    private static func isGameCategory(_ category: String) -> Bool {
        let normalized = category.lowercased()
        return normalized == "public.app-category.games"
            || (normalized.hasPrefix("public.app-category.") && normalized.hasSuffix("-games"))
    }

    private static func isAppID(_ id: String) -> Bool {
        !id.isEmpty && id.utf8.allSatisfy { (48...57).contains($0) }
    }

    private static func isInstallDirectory(_ directory: String) -> Bool {
        !directory.isEmpty && directory != "." && directory != ".."
            && !directory.contains("/") && !directory.contains("\\") && !directory.contains("\0")
    }

    /// Never enter app bundles, other packages, hidden folders, or symlinked directory trees.
    /// Bounds prevent accidentally crawling a complete disk if a library is misconfigured.
    private static func applications(in root: URL, maximumDepth: Int) -> [URL] {
        if root.pathExtension.lowercased() == "app" { return [root] }
        var found: [URL] = []
        func visit(_ directory: URL, depth: Int) {
            guard depth <= maximumDepth,
                  let entries = try? FileManager.default.contentsOfDirectory(at: directory,
                    includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey, .isPackageKey],
                    options: [.skipsHiddenFiles]) else { return }
            for entry in entries.sorted(by: { $0.path < $1.path }) {
                if entry.pathExtension.lowercased() == "app" {
                    found.append(entry)
                } else if let values = try? entry.resourceValues(forKeys:
                    [.isDirectoryKey, .isSymbolicLinkKey, .isPackageKey]),
                    values.isDirectory == true, values.isSymbolicLink != true, values.isPackage != true {
                    visit(entry, depth: depth + 1)
                }
            }
        }
        visit(root, depth: 0)
        return found
    }

    private static func steamLibraries(root: URL) -> [URL] {
        var libraries = [root]
        // Current Steam uses steamapps; older clients used config.
        for location in ["steamapps/libraryfolders.vdf", "config/libraryfolders.vdf"] {
            guard let folders = readVDF(root.appendingPathComponent(location))?["libraryfolders"]?.object else {
                continue
            }
            for (key, value) in folders.entries where isAppID(key) {
                guard let path = value.string ?? value.object?["path"]?.string,
                      path.hasPrefix("/") else { continue }
                libraries.append(URL(fileURLWithPath: path, isDirectory: true))
            }
        }
        var seen: Set<String> = []
        return libraries.map { $0.standardizedFileURL.resolvingSymlinksInPath() }
            .filter { seen.insert($0.path).inserted }.sorted { $0.path < $1.path }
    }

    private static func readVDF(_ url: URL) -> SteamVDF.Object? {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size <= 2 * 1024 * 1024,
              let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        return SteamVDF.parse(text)
    }
}

/// Valve KeyValues lexer/parser. Quoted paths can contain braces, escaped quotes and slashes;
/// comments and nested `apps` objects must not be mistaken for a library's `path`.
enum SteamVDF {
    indirect enum Value {
        case string(String)
        case object(Object)
        var string: String? { if case .string(let value) = self { return value }; return nil }
        var object: Object? { if case .object(let value) = self { return value }; return nil }
    }

    final class Object {
        let entries: [(String, Value)]
        init(entries: [(String, Value)]) { self.entries = entries }
        subscript(_ key: String) -> Value? {
            entries.last { $0.0.caseInsensitiveCompare(key) == .orderedSame }?.1
        }
    }

    static func parse(_ text: String) -> Object? {
        var parser = Parser(characters: Array(text))
        do { return try parser.object(nested: false, depth: 0) } catch { return nil }
    }

    private enum ParseError: Error { case invalid }
    private enum Token { case text(String), open, close }
    private struct Parser {
        var characters: [Character]
        var position = 0

        mutating func object(nested: Bool, depth: Int) throws -> Object {
            guard depth <= 64 else { throw ParseError.invalid }
            var entries: [(String, Value)] = []
            while let token = try next() {
                if case .close = token {
                    guard nested else { throw ParseError.invalid }
                    return Object(entries: entries)
                }
                guard case .text(let key) = token, let value = try next() else { throw ParseError.invalid }
                switch value {
                case .text(let string): entries.append((key, .string(string)))
                case .open: entries.append((key, .object(try object(nested: true, depth: depth + 1))))
                case .close: throw ParseError.invalid
                }
            }
            guard !nested else { throw ParseError.invalid }
            return Object(entries: entries)
        }

        mutating func next() throws -> Token? {
            while position < characters.count {
                if characters[position].isWhitespace || characters[position] == "\u{feff}" {
                    position += 1
                } else if matches("/", "/") {
                    while position < characters.count && characters[position] != "\n" { position += 1 }
                } else if matches("/", "*") {
                    position += 2
                    while position < characters.count && !matches("*", "/") { position += 1 }
                    guard position < characters.count else { throw ParseError.invalid }
                    position += 2
                } else { break }
            }
            guard position < characters.count else { return nil }
            let first = characters[position]
            position += 1
            if first == "{" { return .open }
            if first == "}" { return .close }
            var value = ""
            if first == "\"" {
                while position < characters.count {
                    let character = characters[position]
                    position += 1
                    if character == "\"" { return .text(value) }
                    if character == "\\" {
                        guard position < characters.count else { throw ParseError.invalid }
                        let escaped = characters[position]
                        position += 1
                        switch escaped {
                        case "\\", "\"": value.append(escaped)
                        case "n": value.append("\n")
                        case "r": value.append("\r")
                        case "t": value.append("\t")
                        default: value.append("\\"); value.append(escaped)
                        }
                    } else { value.append(character) }
                }
                throw ParseError.invalid
            }
            value.append(first)
            while position < characters.count {
                let character = characters[position]
                if character.isWhitespace || character == "{" || character == "}"
                    || matches("/", "/") || matches("/", "*") { break }
                guard character != "\"" else { throw ParseError.invalid }
                value.append(character)
                position += 1
            }
            return .text(value)
        }

        private func matches(_ first: Character, _ second: Character) -> Bool {
            position + 1 < characters.count && characters[position] == first && characters[position + 1] == second
        }
    }
}
