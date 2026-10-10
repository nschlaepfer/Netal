import Foundation
import CryptoKit
import CoreFoundation

/// Supported SR model-byte verification, separate from engine and gameplay validation.
/// Runtime manifests describe files; they cannot authorize another model identity.
struct ModelStatus: Equatable {
    enum State: Equatable {
        case noBundle                 // the runtime bundle itself is not there
        case missing                  // bundle present, model files absent
        case modified([String])       // present, but not the supported model bytes
        case unverified               // present, but no complete usable manifest hashes
        case verified                 // supported full model identities and matching metadata
    }

    var state: State = .noBundle
    /// The model's own identity, from the manifest - e.g. "DLSS_SR_Feature1_Quality_K".
    var name: String?
    var weightsBytes: Int?
    var constantsBytes: Int?
    var bundleURL: URL?

    var isReady: Bool { state == .verified }

    /// One line for the UI, saying what is true rather than what is wanted.
    var summary: String {
        switch state {
        case .noBundle:  return "No runtime bundle found"
        case .missing:   return "No model — Netal cannot upscale"
        case .unverified: return "SR model could not be verified"
        case .modified:  return "Model does not match the supported SR model"
        case .verified:  return "Supported SR model verified"
        }
    }
}

/// Checks the full supported model hashes even when a legacy manifest stores prefixes.
struct ModelInspector {
    struct Identity {
        let bytes: Int
        let sha256: String
    }
    static let supportedName = "DLSS_SR_Feature1_Quality_K"
    static let supportedModel: [String: Identity] = [
        "model/weights.bin": Identity(bytes: 3545824, sha256: "ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1"),
        "model/constants.bin": Identity(bytes: 524288, sha256: "305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773"),
    ]
    private let expectedModel: [String: Identity]
    private var cached: [RuntimeFileSignature]?
    private var result = ModelStatus()

    // Alternate identities are for headless fixtures. Production uses the pinned default;
    // there is no runtime-folder, preference, or environment override for supported hashes.
    init(expectedModel: [String: Identity] = Self.supportedModel) {
        self.expectedModel = expectedModel
    }

    static func bundleURL(forLog log: URL) -> URL {
        log.deletingLastPathComponent().appendingPathComponent("re9-sr-runtime-bundle")
    }

    mutating func status(forLog log: URL, fallbacks: [URL] = []) -> ModelStatus {
        guard let bundle = ([Self.bundleURL(forLog: log)] + fallbacks).first(where: {
            RuntimeFileSignature.read($0.appendingPathComponent("manifest.json")) != nil
        }) else {
            cached = nil; result = ModelStatus(); return result
        }
        let paths = ["model/weights.bin", "model/constants.bin"]
        var status = ModelStatus(state: .unverified, bundleURL: bundle)
        let model = bundle.appendingPathComponent("model")
        let manifest = bundle.appendingPathComponent("manifest.json")
        guard let rootInfo = RuntimeFileSignature.read(bundle), rootInfo.directory,
              let manifestInfo = RuntimeFileSignature.read(manifest), manifestInfo.regular else {
            cached = nil; result = status; return result
        }
        guard let modelInfo = RuntimeFileSignature.read(model) else {
            cached = nil; status.state = .missing; result = status; return result
        }
        guard modelInfo.directory else { cached = nil; result = status; return result }
        let entries = paths.compactMap { RuntimeFileSignature.read(bundle.appendingPathComponent($0)) }
        guard entries.count == paths.count else {
            cached = nil; status.state = .missing; result = status; return result
        }
        guard entries.allSatisfy(\.regular) else { cached = nil; result = status; return result }
        let fingerprint = [rootInfo, modelInfo, manifestInfo] + entries
        if cached == fingerprint { return result }
        cached = nil
        status.weightsBytes = Int(entries[0].size)
        status.constantsBytes = Int(entries[1].size)
        guard let data = try? Data(contentsOf: manifest),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            result = status; return result
        }
        status.name = json["model"] as? String
        guard status.name == Self.supportedName,
              let schema = json["schema"] as? String,
              ["netal.sr.runtime.v2", "neuralstudio.re9.exact_sr_immutable_runtime_bundle.v1"].contains(schema),
              let files = json["files"] as? [[String: Any]],
              Set(expectedModel.keys) == Set(paths) else { result = status; return result }
        var wrong: [String] = []
        for path in paths {
            let rows = files.filter { ($0["path"] as? String) == path }
            guard rows.count == 1, let expected = expectedModel[path],
                  expected.bytes > 0, expected.sha256.count == 64,
                  let declaredHash = (rows[0]["sha256"] as? String)?.lowercased(),
                  (40...64).contains(declaredHash.count),
                  declaredHash.allSatisfy({ "0123456789abcdef".contains($0) }),
                  expected.sha256.hasPrefix(declaredHash),
                  schema != "netal.sr.runtime.v2" || declaredHash.count == 64 else {
                result = status; return result
            }
            // Older manifests may encode byte lengths as decimal strings. Reject booleans,
            // fractions and noncanonical lengths instead of silently coercing them to Int.
            let declaredBytes: String?
            if let text = rows[0]["bytes"] as? String {
                declaredBytes = text
            } else if let number = rows[0]["bytes"] as? NSNumber,
                      CFGetTypeID(number) != CFBooleanGetTypeID() {
                declaredBytes = number.stringValue
            } else { declaredBytes = nil }
            guard declaredBytes == String(expected.bytes) else { result = status; return result }
            guard let contents = try? Data(contentsOf: bundle.appendingPathComponent(path)),
                  contents.count == expected.bytes else { wrong.append(path); continue }
            let digest = SHA256.hash(data: contents).map { String(format: "%02x", $0) }.joined()
            if digest != expected.sha256 { wrong.append(path) }
        }
        guard fingerprint.allSatisfy({ RuntimeFileSignature.read(URL(fileURLWithPath: $0.path)) == $0 }) else {
            result = status; return result
        }
        status.state = wrong.isEmpty ? .verified : .modified(wrong)
        cached = fingerprint; result = status; return result
    }
}
