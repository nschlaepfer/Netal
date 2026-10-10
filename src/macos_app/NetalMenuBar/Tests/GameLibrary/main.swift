import Foundation

var failures = 0
var checks = 0
func check(_ passed: Bool, _ description: String) {
    checks += 1
    print("\(passed ? "ok" : "FAIL") \(description)")
    if !passed { failures += 1 }
}

let files = FileManager.default
let temporary = files.temporaryDirectory.appendingPathComponent("netal-library-\(UUID().uuidString)")
try files.createDirectory(at: temporary, withIntermediateDirectories: true)
defer { try? files.removeItem(at: temporary) }

func write(_ text: String, to url: URL) throws {
    try files.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try text.write(to: url, atomically: true, encoding: .utf8)
}

@discardableResult
func makeApp(_ url: URL, name: String, bundleID: String? = "test.game",
             category: String? = nil, store: Bool = false, executable: Bool = true) throws -> URL {
    var info: [String: Any] = ["CFBundleName": name, "CFBundleExecutable": "fixture-game"]
    if let bundleID { info["CFBundleIdentifier"] = bundleID }
    if let category { info["LSApplicationCategoryType"] = category }
    let contents = url.appendingPathComponent("Contents")
    try files.createDirectory(at: contents, withIntermediateDirectories: true)
    try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
        .write(to: contents.appendingPathComponent("Info.plist"))
    if executable {
        let binary = contents.appendingPathComponent("MacOS/fixture-game")
        try write("#!/bin/sh\nexit 93\n", to: binary)
        try files.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
    }
    if store { try write("fixture receipt", to: contents.appendingPathComponent("_MASReceipt/receipt")) }
    return url
}

func quote(_ value: String) -> String {
    "\"" + value.replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"") + "\""
}

func manifest(_ library: URL, id: String, name: String, directory: String) throws {
    try write("\"AppState\" { \"appid\" \"\(id)\" \"name\" \(quote(name)) \"installdir\" \(quote(directory)) }",
              to: library.appendingPathComponent("steamapps/appmanifest_\(id).acf"))
}

let applications = temporary.appendingPathComponent("Applications")
let homeApplications = temporary.appendingPathComponent("HomeApplications")
let steam = temporary.appendingPathComponent("Steam")
let external = temporary.appendingPathComponent("External \"Games\" \\ Library")
let legacy = temporary.appendingPathComponent("Legacy Games")
let macGame = try makeApp(applications.appendingPathComponent("A Native.app"), name: "A Native",
                         category: "public.app-category.games")
try makeApp(applications.appendingPathComponent("Games/B Store.app"), name: "B Store",
            category: "public.app-category.action-games", store: true)
try makeApp(homeApplications.appendingPathComponent("C Home.app"), name: "C Home",
            category: "public.app-category.role-playing-games")
try makeApp(applications.appendingPathComponent("Epic Games Launcher.app"), name: "Epic Games Launcher",
            bundleID: "com.epicgames.EpicGamesLauncher", category: "public.app-category.games")
let editor = try makeApp(applications.appendingPathComponent("Editor.app"), name: "Editor",
                         category: "public.app-category.developer-tools")
try makeApp(applications.appendingPathComponent("Broken.app"), name: "Broken",
            category: "public.app-category.games", executable: false)
try makeApp(macGame.appendingPathComponent("Contents/Helpers/Nested.app"), name: "Nested",
            category: "public.app-category.games")
let valheim = try makeApp(steam.appendingPathComponent("steamapps/common/Valheim/Valheim.app"), name: "valheim",
                         bundleID: "com.irongate.valheim")
try manifest(steam, id: "892970", name: "Valheim", directory: "Valheim")
let hades = try makeApp(external.appendingPathComponent("steamapps/common/Hades/Game.macOS.app"), name: "Hades",
                       bundleID: nil)
try manifest(external, id: "1145360", name: "Hades", directory: "Hades")
try makeApp(legacy.appendingPathComponent("steamapps/common/Legacy/Legacy.app"), name: "Legacy")
try manifest(legacy, id: "100", name: "Legacy", directory: "Legacy")
try manifest(steam, id: "200", name: "Windows only", directory: "Windows")
try write("fixture PE binary", to: steam.appendingPathComponent("steamapps/common/Windows/game.exe"))
try manifest(steam, id: "201", name: "Uninstalled", directory: "Not Installed")
try manifest(steam, id: "202", name: "Traversal", directory: "../../../Applications")
try write("\"AppState\" { \"appid\" \"999\" \"installdir\" \"Valheim\" }",
          to: steam.appendingPathComponent("steamapps/appmanifest_203.acf"))
try write("\"AppState\" { \"appid\" \"204\" \"installdir\" \"Valheim\"",
          to: steam.appendingPathComponent("steamapps/appmanifest_204.acf"))

let libraryFile = """
// Steam library folder fixture. A comment can contain "path" "not a library".
"libraryfolders"
{
    "0" { "path" \(quote(steam.path)) "apps" { "892970" "12345" } }
    "1" { "path" \(quote(external.path)) "label" "braces { in } text" "apps" { "1145360" "1" } }
    /* legacy format remains valid */
    "2" \(quote(legacy.path))
    "3" { "path" \(quote(external.path)) }
    "4" { "path" "relative/library" }
    "not-a-library" { "path" "/" }
}
"""
try write(libraryFile, to: steam.appendingPathComponent("steamapps/libraryfolders.vdf"))

let roots = [applications, homeApplications, temporary.appendingPathComponent("Absent")]
let games = GameLibrary.discover(applicationRoots: roots, steamRoot: steam, loadIcons: false)
check(games.count == 6, "finds categorized Mac apps and native Steam apps only")
check(games.map(\.name) == ["A Native", "B Store", "C Home", "Hades", "Legacy", "Valheim"],
      "games use display names and stable alphabetical ordering")
check(games.first { $0.name == "B Store" }?.source == .macAppStore, "receipt identifies Mac App Store source")
check(games.first { $0.name == "A Native" }?.source == .mac, "ordinary native game has Mac source")
check(games.first { $0.name == "Valheim" }?.steamAppID == "892970", "Valheim resolves app ID from its manifest")
check(games.first { $0.name == "Valheim" }?.appURL.path == valheim.path, "Valheim app URL is preserved")
check(games.first { $0.name == "Hades" }?.source == .steam, "external Steam library is discovered")
check(games.first { $0.name == "Hades" }?.bundleID == "path:\(hades.path)",
      "Steam game with no bundle ID retains a stable path identity")
check(games.first { $0.name == "Legacy" } != nil, "legacy library folder strings are supported")
check(games.allSatisfy { $0.icon == nil }, "fixture discovery does not fetch desktop icons")
check(!games.contains { $0.name == "Nested" }, "helper apps inside app bundles stay hidden")
check(!games.contains { $0.name == "Epic Games Launcher" }, "launcher Games category does not make it an installed game")
check(!games.contains { $0.name == "Editor" }, "uncategorized non-games are not automatically offered")
check(!games.contains { $0.name == "Broken" }, "incomplete apps with no executable stay hidden")
check(GameLibrary.candidate(for: editor, loadIcons: false)?.name == "Editor", "explicit app picker can add uncategorized apps")
check(GameLibrary.candidate(for: temporary, loadIcons: false) == nil, "arbitrary folders are not launch candidates")
let manuallyAdded = GameLibrary.discover(applicationRoots: roots, steamRoot: steam,
                                        userAddedApps: [editor, macGame], loadIcons: false)
check(manuallyAdded.count == 7 && manuallyAdded.contains { $0.name == "Editor" },
      "saved manually selected apps merge with discovery without duplicates")
let alias = temporary.appendingPathComponent("Alias.app")
try files.createSymbolicLink(at: alias, withDestinationURL: macGame)
check(GameLibrary.candidate(for: alias, loadIcons: false)?.id == GameLibrary.candidate(for: macGame, loadIcons: false)?.id,
      "symbolic app aliases share canonical path identity")
check(games.first { $0.name == "Valheim" }?.identity.bundleID == "com.irongate.valheim",
      "candidate identity retains native bundle ID")
check(games.first { $0.name == "Valheim" }?.identity.appURL?.path == valheim.path,
      "candidate identity retains launch URL")

let parsed = SteamVDF.parse(libraryFile)
check(parsed?["LIBRARYFOLDERS"]?.object?["1"]?.object?["path"]?.string == external.path,
      "VDF fields are case insensitive and escaped library paths round-trip")
check(parsed?["libraryfolders"]?.object?["1"]?.object?["label"]?.string == "braces { in } text",
      "quoted braces are text, not structure")
check(SteamVDF.parse("root { name \"Game // title\" }")?["root"]?.object?["name"]?.string == "Game // title",
      "comment markers inside strings remain text")
check(SteamVDF.parse("\u{feff} root { key value } // tail")?["root"]?.object?["key"]?.string == "value",
      "BOM, bare tokens and trailing comments are accepted")
check(SteamVDF.parse("root { key \"unterminated }") == nil, "unterminated quoted value rejects the document")
check(SteamVDF.parse("root { key value") == nil, "missing closing brace rejects the document")
check(SteamVDF.parse("root { key }") == nil, "missing value rejects the document")
check(SteamVDF.parse("root {} }") == nil, "extra closing brace rejects the document")
check(SteamVDF.parse("/* unterminated") == nil, "unterminated block comment rejects the document")
check(SteamVDF.parse(String(repeating: "key {", count: 66) + String(repeating: "}", count: 66)) == nil,
      "unreasonable nesting is bounded")

try files.removeItem(at: steam.appendingPathComponent("steamapps/libraryfolders.vdf"))
try write("\"LibraryFolders\" { \"0\" \(quote(external.path)) }",
          to: steam.appendingPathComponent("config/libraryfolders.vdf"))
let olderClient = GameLibrary.discover(applicationRoots: [], steamRoot: steam, loadIcons: false)
check(olderClient.map(\.name) == ["Hades", "Valheim"], "older Steam config location is supported")
check(GameLibrary.discover(applicationRoots: [], steamRoot: temporary.appendingPathComponent("NoSteam"),
                          loadIcons: false).isEmpty, "missing Steam installation returns an empty library")
print("\(checks) discovery checks; \(failures) failures")
exit(Int32(failures))
