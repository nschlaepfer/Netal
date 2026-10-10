import Foundation

let fm = FileManager.default
let root = fm.temporaryDirectory.appendingPathComponent("netal-crossover-library-\(UUID().uuidString)")
try fm.createDirectory(at: root, withIntermediateDirectories: true)
defer { try? fm.removeItem(at: root) }
var count = 0, failures = 0
func check(_ pass: Bool, _ name: String) {
    count += 1; if !pass { failures += 1 }; print("\(pass ? "ok" : "FAIL") \(name)")
}
func write(_ text: String, _ file: URL) throws {
    try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
    try text.write(to: file, atomically: true, encoding: .utf8)
}
func steam(_ bottle: URL) throws -> URL {
    let path = bottle.appendingPathComponent("drive_c/Program Files (x86)/Steam")
    try write("fixture PE bytes; never run", path.appendingPathComponent("steam.exe")); return path
}
func manifest(_ steam: URL, _ id: String, _ name: String, _ directory: String, flags: Int = 4, makeGame: Bool = true) throws {
    try write("\"AppState\" { \"appid\" \"\(id)\" \"name\" \"\(name)\" \"installdir\" \"\(directory)\" \"StateFlags\" \"\(flags)\" }", steam.appendingPathComponent("steamapps/appmanifest_\(id).acf"))
    if makeGame { try write("fixture", steam.appendingPathComponent("steamapps/common/\(directory)/game.exe")) }
}
@discardableResult
func wrapper(_ apps: URL, _ name: String, _ bottle: String, shortcut: String = "game.lnk") throws -> URL {
    let url = apps.appendingPathComponent("\(name).app")
    let data: [String: Any] = ["CFBundleName": name, "CFBundleExecutable": "Menu Helper",
        "CXHelperAppBottleName": bottle, "CrossOverHelperMenuPath": shortcut]
    let exe = url.appendingPathComponent("Contents/MacOS/Menu Helper")
    try write("#!/bin/sh\nexit 91", exe)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: exe.path)
    try PropertyListSerialization.data(fromPropertyList: data, format: .xml, options: 0)
        .write(to: url.appendingPathComponent("Contents/Info.plist"))
    return url
}
let bottles = root.appendingPathComponent("Bottles"), apps = root.appendingPathComponent("Apps")
let a = bottles.appendingPathComponent("Steam ' Games"), b = bottles.appendingPathComponent("Second Bottle")
let sa = try steam(a), sb = try steam(b)
try manifest(sa, "100", "Shared Game", "Shared Game")
try manifest(sb, "100", "Shared Game", "Shared Game")
try manifest(sa, "228980", "Steamworks Common Redistributables", "Steamworks Shared")
try manifest(sa, "200", "Uninstalled", "Uninstalled", flags: 0)
try manifest(sa, "201", "Missing", "Missing", makeGame: false)
try manifest(sa, "202", "Traversal", "../Outside", makeGame: false)
try write("\"AppState\" { \"appid\" \"999\" \"name\" \"Mismatch\" \"installdir\" \"Shared Game\" }", sa.appendingPathComponent("steamapps/appmanifest_203.acf"))
try write("fixture", sa.appendingPathComponent("steamapps/common/Shared Game/CrashReport.exe"))
try write("fixture", sa.appendingPathComponent("steamapps/common/Shared Game/UbisoftConnectInstaller.exe"))
try write("fixture", sa.appendingPathComponent("steamapps/common/Shared Game/Support/tool.exe"))
let sharedWrapper = try wrapper(apps, "Shared Game", a.lastPathComponent, shortcut: "Shared Game.url")
try wrapper(apps, "Local RPG", a.lastPathComponent)
try wrapper(apps, "Steam", a.lastPathComponent)
try wrapper(apps, "Rockstar Games Launcher", a.lastPathComponent)
try wrapper(apps, "Missing Steam Shortcut", a.lastPathComponent, shortcut: "Missing.url")
try wrapper(apps, "Absent Bottle", "Nowhere")
let external = root.appendingPathComponent("External")
let externalLibrary = external.appendingPathComponent("Steam Library")
try manifest(externalLibrary, "300", "External Game", "External Game")
try fm.createDirectory(at: a.appendingPathComponent("dosdevices"), withIntermediateDirectories: true)
try fm.createSymbolicLink(at: a.appendingPathComponent("dosdevices/y:"), withDestinationURL: external)
try write("\"libraryfolders\" { \"0\" { \"path\" \"C:\\\\Program Files (x86)\\\\Steam\" } \"1\" { \"path\" \"Y:\\\\Steam Library\" } \"2\" { \"path\" \"Y:\\\\Steam Library\" } }", sa.appendingPathComponent("steamapps/libraryfolders.vdf"))
let games = CrossOverLibrary.discover(bottlesRoot: bottles, applicationsRoot: apps, loadIcons: false)
check(games.count == 4, "installed Steam titles and registered non-Steam game discovered")
check(games.filter { $0.name == "Shared Game" }.count == 2, "same app ID in separate bottles remains separate")
check(Set(games.map(\.id)).count == games.count, "bottle and title identity are unique")
check(games.allSatisfy { $0.source == .crossOver && $0.icon == nil }, "CrossOver source and fixture icon isolation")
check(games.contains { $0.name == "External Game" }, "mapped external Windows Steam library discovered")
check(!games.contains { ["Uninstalled", "Missing", "Traversal", "Mismatch"].contains($0.name) }, "uninstalled missing and malformed entries excluded")
check(!games.contains { $0.name.contains("Redistributables") }, "Steam shared runtimes are not games")
check(!games.contains { ["Steam", "Rockstar Games Launcher"].contains($0.name) }, "launchers excluded")
check(!games.contains { $0.name == "Missing Steam Shortcut" }, "stale URL shortcut cannot recreate missing game")
check(!games.contains { $0.name == "Absent Bottle" }, "shortcut requires an existing bottle")
let shared = games.first { $0.name == "Shared Game" && $0.crossOver?.bottle == a.lastPathComponent }!
check(shared.crossOver?.wrapperURL?.resolvingSymlinksInPath().path == sharedWrapper.resolvingSymlinksInPath().path, "registered shortcut augments manifest rather than duplicating title")
check(shared.crossOver?.executables == ["game.exe"], "crash reports installers and support tools excluded from process choices")
check(shared.sourceDetail == "CrossOver · Steam ' Games", "bottle name shown with source")
let arguments = CrossOverLaunch.arguments(for: shared.crossOver!)!
check(arguments == ["--bottle", "Steam ' Games", "--no-update", "--no-wait", "C:\\Program Files (x86)\\Steam\\steam.exe", "-applaunch", "100"], "launch preserves bottle and Windows path as separate arguments")
var bad = shared.crossOver!; bad.steamAppID = "100; touch /tmp/never"
check(CrossOverLaunch.arguments(for: bad) == nil, "launch rejects nonnumeric Steam ID")
check(CrossOverLaunch.arguments(for: games.first { $0.name == "Local RPG" }!.crossOver!) == nil, "non-Steam shortcut never becomes a native Steam URI")
check(CrossOverLibrary.windowsURL("Y:\\Steam Library", bottle: a)?.resolvingSymlinksInPath().path == externalLibrary.resolvingSymlinksInPath().path, "drive mapping resolves quoted/spaced library paths")
check(CrossOverLibrary.windowsURL("Q:\\Missing", bottle: a) == nil, "unmapped drive rejected")
check(CrossOverLibrary.windowsURL("Y:\\..\\Outside", bottle: a) == nil, "parent traversal rejected")
check(CrossOverLibrary.windowsURL("/tmp/absolute", bottle: a) == nil, "Unix path is not accepted as a Windows drive")
check(CrossOverLibrary.windowsURL("\\\\server\\share", bottle: a) == nil, "UNC network path is not scanned")
let merged = GameLibrary.discover(applicationRoots: [], steamRoot: root.appendingPathComponent("Absent Native Steam"),
    crossOverBottlesRoot: bottles, crossOverApplicationsRoot: apps, loadIcons: false)
check(merged.count == 4, "CrossOver catalog merges into the main library")
check(merged.map(\.name) == ["External Game", "Local RPG", "Shared Game", "Shared Game"], "combined library order is stable")
check(CrossOverLibrary.discover(bottlesRoot: root.appendingPathComponent("Absent"), applicationsRoot: apps, loadIcons: false).isEmpty, "absent CrossOver is an empty catalog")

let gamePath = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Shared Game\\game.exe"
let target = shared.crossOver!
func identity(_ executable: String = gamePath, prefix: URL = a) -> CrossOverProcessIdentity.Identity {
    .init(executable: executable, winePrefix: prefix.path)
}
check(CrossOverProcessIdentity.matches(identity(), target: target), "exact game and bottle match")
check(!CrossOverProcessIdentity.matches(identity(prefix: b), target: target), "same game in another bottle cannot supply a window")
check(!CrossOverProcessIdentity.matches(identity("C:\\Program Files (x86)\\Steam\\steam.exe"), target: target), "Steam launcher cannot supply the game's window")
check(!CrossOverProcessIdentity.matches(identity("game.exe"), target: target), "ambiguous executable basename is rejected")
check(!CrossOverProcessIdentity.matches(identity(gamePath + ".other"), target: target), "executable suffix cannot impersonate the game")
check(!CrossOverProcessIdentity.matches(identity(gamePath.replacingOccurrences(of: "Shared Game", with: "Other Game")), target: target), "same executable name in a different installation is rejected")
check(!CrossOverProcessIdentity.matches(identity(gamePath.replacingOccurrences(of: "game.exe", with: "CrashReport.exe")), target: target), "crash reporter cannot supply the game's window")
check(CrossOverProcessIdentity.matches(identity(gamePath.uppercased()), target: target), "Windows executable matching is case insensitive")
check(!CrossOverProcessIdentity.matches(processID: -1, target: target), "invalid PID is rejected")
check(!CrossOverProcessIdentity.matches(processID: getpid(), target: target), "the test's native process cannot be mistaken for a game")
func processBytes(arguments: [String], environment: [String]) -> [UInt8] {
    var count = Int32(arguments.count)
    var bytes = withUnsafeBytes(of: &count) { Array($0) }
    bytes += Array("/wine/loader\0\0".utf8)
    for string in arguments + environment { bytes += Array(string.utf8) + [0] }
    return bytes
}
let parsed = CrossOverProcessIdentity.decode(processBytes(arguments: [gamePath, "-uplay_steam_mode", ""],
    environment: ["UNRELATED=value", "WINEPREFIX=\(a.path)", ""]))
check(parsed?.executable == gamePath && parsed?.winePrefix == a.path, "macOS process arguments retain only executable and bottle identity")
check(CrossOverProcessIdentity.decode(processBytes(arguments: [gamePath], environment: [])) == nil, "missing bottle identity is rejected")
check(CrossOverProcessIdentity.decode(processBytes(arguments: [gamePath], environment: ["WINEPREFIX=\(a.path)", "WINEPREFIX=\(b.path)"])) == nil, "duplicate bottle identity is rejected")
check(CrossOverProcessIdentity.decode([1, 0, 0]) == nil, "truncated process header is rejected")
check(CrossOverProcessIdentity.decode(Array(processBytes(arguments: [gamePath], environment: ["WINEPREFIX=\(a.path)"]).dropLast())) == nil, "unterminated process metadata is rejected")

var unity = target
unity.steamAppID = "289650"
unity.installationURL = sa.appendingPathComponent("steamapps/common/Assassin's Creed Unity")
unity.executables = ["ACU.exe"]
let unityExecutable = unity.installationURL.appendingPathComponent("ACU.exe")
try write("fixture; never execute", unityExecutable)
let desktop = CrossOverLaunch.plan(for: unity, virtualDesktop: true)
check(desktop?.program == "wine", "virtual desktop uses the Wine launch interface")
check(desktop?.arguments == ["--bottle", a.lastPathComponent, "--no-update", "--no-wait",
    "--env", "SteamAppId=289650", "--workdir", unity.installationURL.path,
    "explorer", "/desktop=NetalUnity,1280x720", unityExecutable.path, "-uplay_steam_mode"],
    "Unity desktop launch preserves bottle, working directory, app ID and Ubisoft startup arguments")
check(CrossOverLaunch.plan(for: unity)?.program == "cxstart", "normal Steam launch remains the default")
check(CrossOverLaunch.plan(for: target, virtualDesktop: true) == nil, "Unity profile cannot launch an unrelated Steam game")
var missingUnity = unity; missingUnity.executables = []
check(CrossOverLaunch.plan(for: missingUnity, virtualDesktop: true) == nil, "Unity desktop requires a discovered game executable")
var otherBottleUnity = unity; otherBottleUnity.bottleURL = b
check(CrossOverLaunch.virtualDesktopPreferenceKey(for: unity) != CrossOverLaunch.virtualDesktopPreferenceKey(for: otherBottleUnity), "desktop preference is scoped to this game and bottle")
try fm.removeItem(at: unityExecutable)
check(CrossOverLaunch.plan(for: unity, virtualDesktop: true) == nil, "removed Unity executable cannot be launched through a stale profile")
let externalExecutable = root.appendingPathComponent("Outside/ACU.exe")
try write("fixture; never execute", externalExecutable)
try fm.createSymbolicLink(at: unityExecutable, withDestinationURL: externalExecutable)
check(CrossOverLaunch.plan(for: unity, virtualDesktop: true) == nil, "desktop profile rejects an executable link outside the game installation")
print("\(count) CrossOver checks; \(failures) failures")
exit(Int32(failures))
