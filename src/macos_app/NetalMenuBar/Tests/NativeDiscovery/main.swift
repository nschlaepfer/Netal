import AppKit
import Foundation

var failures = 0
func check(_ condition: Bool, _ title: String) {
    print("\(condition ? "ok" : "FAIL") \(title)")
    if !condition { failures += 1 }
}
let manager = FileManager.default
let root = manager.temporaryDirectory.appendingPathComponent("netal-native-discovery-\(UUID().uuidString)")
try manager.createDirectory(at: root, withIntermediateDirectories: true)
defer { try? manager.removeItem(at: root) }

func be32(_ value: UInt32) -> [UInt8] {
    [UInt8((value >> 24) & 255), UInt8((value >> 16) & 255), UInt8((value >> 8) & 255), UInt8(value & 255)]
}
func thin(_ evidence: Bool) -> Data {
    var data = Data([0xcf, 0xfa, 0xed, 0xfe])
    data.append(Data(repeating: 0, count: 28))
    if evidence { data.append(Data("MetalFX.framework".utf8)) }
    return data
}
let fat = root.appendingPathComponent("fat")
let farOffset: UInt32 = 5 << 20
var header = Data(be32(0xcafebabe) + be32(2))
header.append(Data(be32(0x1000007) + be32(3) + be32(4096) + be32(UInt32(thin(false).count)) + be32(12)))
header.append(Data(be32(0x100000c) + be32(0) + be32(farOffset) + be32(UInt32(thin(true).count)) + be32(12)))
try header.write(to: fat)
let handle = try FileHandle(forWritingTo: fat)
try handle.seek(toOffset: 4096); try handle.write(contentsOf: thin(false))
try handle.seek(toOffset: UInt64(farOffset)); try handle.write(contentsOf: thin(true)); try handle.close()
check(MetalFXApps.linksMetalFX(fat), "MetalFX evidence found in a later universal-binary slice")
check(MetalFXApps.slices(header: Data(be32(0xcafebabe) + be32(0xffffffff)), fileSize: 100).isEmpty,
      "malformed slice counts are rejected")
check(MetalFXApps.slices(header: header, fileSize: 64).isEmpty, "out-of-file slices are rejected")

let app = root.appendingPathComponent("Engine game.app")
let contents = app.appendingPathComponent("Contents")
try manager.createDirectory(at: contents.appendingPathComponent("MacOS"), withIntermediateDirectories: true)
try manager.createDirectory(at: contents.appendingPathComponent("Frameworks"), withIntermediateDirectories: true)
let plist: [String: Any] = ["CFBundleIdentifier": "com.example.engine", "CFBundleExecutable": "Game", "CFBundlePackageType": "APPL"]
try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0).write(to: contents.appendingPathComponent("Info.plist"))
try thin(false).write(to: contents.appendingPathComponent("MacOS/Game"))
try manager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: contents.appendingPathComponent("MacOS/Game").path)
try thin(true).write(to: contents.appendingPathComponent("Frameworks/UnityPlayer.dylib"))
check(MetalFXApps.hasMetalFXEvidence(in: app), "engine framework evidence reaches running-app discovery")

let sessions = root.appendingPathComponent("NativeGames")
let session = sessions.appendingPathComponent("com.example.engine")
try manager.createDirectory(at: session, withIntermediateDirectories: true)
let log = session.appendingPathComponent("netal-live.log")
try Data().write(to: log)
check(GameIdentity.nativeBundleID(from: log, root: sessions) == "com.example.engine", "unsandboxed session identifies its bundle")
check(GameIdentity.nativeBundleID(from: session.appendingPathComponent("other.log"), root: sessions) == nil,
      "unrelated native files are not sessions")
check(SessionDiscovery.nativeLogs(in: sessions).map { $0.resolvingSymlinksInPath().path }
      == [log.resolvingSymlinksInPath().path], "unsandboxed session log is discoverable")
try manager.createSymbolicLink(at: sessions.appendingPathComponent("com.example.alias"), withDestinationURL: session)
check(SessionDiscovery.nativeLogs(in: sessions).count == 1, "symlinked session directories do not duplicate discovery")
check(GameIdentity.nativeBundleID(from: root.appendingPathComponent("elsewhere/com.example.game/netal-live.log"), root: sessions) == nil,
      "unowned paths cannot become native sessions")
exit(Int32(failures))
