import Foundation
import CryptoKit

let manager = FileManager.default
let contractData = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
if CommandLine.arguments.count == 3 {
    // Optional read-only integration check against locally supplied real runtime bytes.
    var inspector = FGBundleInspector(contractData: contractData)
    let result = inspector.status(forLog: URL(fileURLWithPath: "/unused/netal.log"),
                                  fallbacks: [URL(fileURLWithPath: CommandLine.arguments[2])])
    print("status=\(result.badge) detail=\(result.summary)")
    exit(result.isReady ? 0 : 1)
}
var failures = 0
func check(_ passed: Bool, _ text: String) {
    print("\(passed ? "ok" : "FAIL") \(text)")
    if !passed { failures += 1 }
}
func identity(_ data: Data) -> [String: Any] {
    ["bytes": data.count, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
}
let root = manager.temporaryDirectory.appendingPathComponent("Netal FG \(UUID().uuidString)")
try manager.createDirectory(at: root, withIntermediateDirectories: true)
defer { try? manager.removeItem(at: root) }
let log = root.appendingPathComponent("runtime.log")
let bundle = FGBundleInspector.bundleURL(forLog: log)
let model = bundle.appendingPathComponent("model")
try manager.createDirectory(at: model, withIntermediateDirectories: true)
var contract = try JSONSerialization.jsonObject(with: contractData) as! [String: Any]
var pinnedModel: [String: Any] = [:], pinnedMath: [String: Any] = [:]
var payload: [String: Data] = [:]
for (index, name) in (contract["model"] as! [String: Any]).keys.sorted().enumerated() {
    let data = Data(repeating: UInt8(index), count: 16)
    pinnedModel[name] = identity(data); payload["model/" + name] = data
}
for name in (contract["math"] as! [String: Any]).keys {
    let data = Data(repeating: 0x71, count: 16)
    pinnedMath[name] = identity(data); payload[name] = data
}
for name in contract["required_engine_files"] as! [String] { payload[name] = Data(repeating: 0x31, count: 16) }
contract["model"] = pinnedModel; contract["math"] = pinnedMath
let fixtureContract = try JSONSerialization.data(withJSONObject: contract)
var files = payload.mapValues(identity)
let manifest = bundle.appendingPathComponent("manifest.json")
func writeManifest() throws {
    try JSONSerialization.data(withJSONObject: ["schema": "netal.fg.runtime-bundle.v1", "profile": "640x360", "files": files])
        .write(to: manifest)
}
for (name, data) in payload {
    let path = bundle.appendingPathComponent(name)
    try manager.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: path)
}
var inspector = FGBundleInspector(contractData: fixtureContract)
check(inspector.status(forLog: log).state == .noBundle, "no manifest cannot enable FG")
try writeManifest()
check(inspector.status(forLog: log).isReady, "supported model/math and complete engine inventory verify")
check(inspector.status(forLog: log).isReady, "unchanged poll preserves verification")
var absentContract = FGBundleInspector(contractData: nil)
check(!absentContract.status(forLog: log).isReady, "missing app identity resource fails closed")
var malformedContract = FGBundleInspector(contractData: Data("{}".utf8))
check(!malformedContract.status(forLog: log).isReady, "malformed app identity resource fails closed")
let tokenName = "model/default-token-000063.bin"
let token = bundle.appendingPathComponent(tokenName)
let originalDate = try manager.attributesOfItem(atPath: token.path)[.modificationDate] as! Date
let changed = Data(repeating: 0x94, count: 16)
try changed.write(to: token)
try manager.setAttributes([.modificationDate: originalDate], ofItemAtPath: token.path)
check(inspector.status(forLog: log).state == .modified([tokenName]), "same-size mutation with restored mtime invalidates cache")
files[tokenName] = identity(changed)
try writeManifest()
check(inspector.status(forLog: log).state == .unverified, "rewritten manifest cannot approve unsupported model bytes")
files[tokenName] = identity(payload[tokenName]!)
try payload[tokenName]!.write(to: token); try writeManifest()
check(inspector.status(forLog: log).isReady, "restored files verify again")
try manager.removeItem(at: token)
check(inspector.status(forLog: log).state == .missing([tokenName]), "missing token is reported")
let target = root.appendingPathComponent("linked-buffer")
try payload[tokenName]!.write(to: target)
try manager.createSymbolicLink(at: token, withDestinationURL: target)
check(!inspector.status(forLog: log).isReady, "matching token behind symlink rejected")
try manager.removeItem(at: token); try payload[tokenName]!.write(to: token)
let extra = model.appendingPathComponent("extra.bin")
try changed.write(to: extra)
check(!inspector.status(forLog: log).isReady, "unlisted extra model buffer rejected")
files["model/extra.bin"] = identity(changed); try writeManifest()
check(!inspector.status(forLog: log).isReady, "listed extra model buffer still rejected")
files.removeValue(forKey: "model/extra.bin"); try manager.removeItem(at: extra); try writeManifest()
let mathName = "sigmoid.bin"
try changed.write(to: bundle.appendingPathComponent(mathName))
check(inspector.status(forLog: log).state == .modified([mathName]), "modified arithmetic table rejected")
try payload[mathName]!.write(to: bundle.appendingPathComponent(mathName))
let shaderName = (contract["required_engine_files"] as! [String]).first { $0.hasPrefix("libs/") }!
let shader = bundle.appendingPathComponent(shaderName)
try changed.write(to: shader)
check(inspector.status(forLog: log).state == .modified([shaderName]), "modified shader invalidates FG readiness")
try manager.removeItem(at: shader)
files.removeValue(forKey: shaderName); try writeManifest()
check(inspector.status(forLog: log).state == .missing([shaderName]), "required shader cannot be removed by editing manifest")
try payload[shaderName]!.write(to: shader); files[shaderName] = identity(payload[shaderName]!); try writeManifest()
files["../outside"] = identity(changed); try writeManifest()
check(!inspector.status(forLog: log).isReady, "unsafe manifest path rejected")
files.removeValue(forKey: "../outside"); try writeManifest()
files[tokenName] = ["bytes": 16, "sha256": ""]
try writeManifest()
check(!inspector.status(forLog: log).isReady, "empty hash cannot verify")
files[tokenName] = identity(payload[tokenName]!); try writeManifest()
let cleanManifest = try Data(contentsOf: manifest)
let manifestDate = try manager.attributesOfItem(atPath: manifest.path)[.modificationDate] as! Date
var changedManifest = String(data: cleanManifest, encoding: .utf8)!
changedManifest = changedManifest.replacingOccurrences(of: "640x360", with: "800x600")
try Data(changedManifest.utf8).write(to: manifest)
try manager.setAttributes([.modificationDate: manifestDate], ofItemAtPath: manifest.path)
check(!inspector.status(forLog: log).isReady, "same-length manifest edit with restored mtime invalidates cache")
try cleanManifest.write(to: manifest)
let alternateLog = root.appendingPathComponent("elsewhere/log")
check(inspector.status(forLog: alternateLog, fallbacks: [bundle]).isReady, "configured fallback with spaces verifies")
let bundleLink = root.appendingPathComponent("linked-bundle")
try manager.createSymbolicLink(at: bundleLink, withDestinationURL: bundle)
check(!inspector.status(forLog: alternateLog, fallbacks: [bundleLink]).isReady, "linked bundle root rejected")
check(inspector.status(forLog: log).isReady, "returning to real bundle verifies")
try manager.moveItem(at: model, to: root.appendingPathComponent("saved-model"))
check(inspector.status(forLog: log).state == .missing(["model"]), "missing model directory reported as missing")
exit(Int32(failures))
