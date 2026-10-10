import Foundation
import CryptoKit
import Darwin

/// Verifies supported model/math identities and the selected bundle's file inventory.
/// Hash verification is separate from gameplay readiness or trust in supplied executable code.
struct FGBundleStatus: Equatable {
    enum State: Equatable {
        case noBundle
        case missing([String])
        case modified([String])
        case unverified
        case verified(tokens: Int, bytes: Int)
    }
    var state: State = .noBundle
    var bundleURL: URL?
    var isReady: Bool { if case .verified = state { return true }; return false }
    var badge: String {
        switch state {
        case .noBundle, .missing: return "Missing"
        case .modified: return "Modified"
        case .unverified: return "Unverified"
        case .verified: return "Verified"
        }
    }
    var summary: String {
        switch state {
        case .noBundle: return "Choose an FG runtime bundle"
        case .missing: return "Some frame generation files are missing"
        case .modified: return "Frame generation files do not match"
        case .unverified: return "Frame generation files could not be verified"
        case .verified: return "Model and runtime files verified"
        }
    }
}

struct FGBundleInspector {
    private struct Identity: Decodable, Equatable {
        let bytes: Int
        let sha256: String
        var valid: Bool {
            bytes > 0 && sha256.count == 64 && sha256.allSatisfy { "0123456789abcdef".contains($0) }
        }
    }
    private struct Contract: Decodable {
        let schema: String
        let profile: String
        let model: [String: Identity]
        let math: [String: Identity]
        let required_engine_files: [String]
    }
    private struct Manifest: Decodable {
        let schema: String
        let profile: String
        let files: [String: Identity]
    }
    private let contract: Contract?
    private var cached: [RuntimeFileSignature]?
    private var result = FGBundleStatus()

    static func bundledContract() -> Data? {
        guard let url = Bundle.main.url(forResource: "FGRuntimeInputs", withExtension: "json") else { return nil }
        return try? Data(contentsOf: url)
    }

    init(contractData: Data? = FGBundleInspector.bundledContract()) {
        if let data = contractData, let value = try? JSONDecoder().decode(Contract.self, from: data),
           value.schema == "netal.fg.runtime-inputs.v1", value.profile == "640x360",
           Set(value.model.keys) == Set((stride(from: 63, through: 95, by: 2).map { $0 }
                + stride(from: 106, through: 204, by: 2).map { $0 }).map { String(format: "default-token-%06d.bin", $0) }),
           Set(value.math.keys) == Set(["rom.bin", "sigmoid.bin", "ex2-unit.bin", "ex2-fraction.bin"]),
           value.model.values.allSatisfy(\.valid), value.math.values.allSatisfy(\.valid),
           !value.required_engine_files.isEmpty,
           Set(value.required_engine_files).count == value.required_engine_files.count,
           value.required_engine_files.allSatisfy({ Self.safePath($0) && ($0.hasPrefix("libs/") || $0.hasPrefix("lib/") || $0.hasPrefix("include/")) }) {
            contract = value
        } else {
            contract = nil
        }
    }

    static func bundleURL(forLog log: URL) -> URL {
        log.deletingLastPathComponent().appendingPathComponent("re9-fg-runtime-bundle")
    }

    private static func safePath(_ path: String) -> Bool {
        !path.isEmpty && path.utf8.allSatisfy { (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 47, 95].contains($0) }
            && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    mutating func status(forLog log: URL, fallbacks: [URL] = []) -> FGBundleStatus {
        let manager = FileManager.default
        guard let bundle = ([Self.bundleURL(forLog: log)] + fallbacks).first(where: {
            RuntimeFileSignature.read($0.appendingPathComponent("manifest.json")) != nil
        }) else {
            cached = nil
            result = FGBundleStatus()
            return result
        }
        func status(_ state: FGBundleStatus.State) -> FGBundleStatus {
            FGBundleStatus(state: state, bundleURL: bundle)
        }
        guard let model = RuntimeFileSignature.read(bundle.appendingPathComponent("model")) else {
            cached = nil; result = status(.missing(["model"])); return result
        }
        // Foundation enumeration expands /var to /private/var, while its URL
        // resolver may shorten it again. POSIX realpath supplies the same prefix.
        guard let resolved = realpath(bundle.path, nil) else {
            cached = nil; result = status(.unverified); return result
        }
        let enumerationRoot = URL(fileURLWithPath: String(cString: resolved))
        free(resolved)
        var traversalFailed = false
        guard let root = RuntimeFileSignature.read(bundle), root.directory, model.directory,
              let manifestSignature = RuntimeFileSignature.read(bundle.appendingPathComponent("manifest.json")), manifestSignature.regular,
              let enumerator = manager.enumerator(at: enumerationRoot, includingPropertiesForKeys: nil, errorHandler: { _, _ in
                  traversalFailed = true; return false
              }) else {
            cached = nil; result = status(.unverified); return result
        }
        var signatures = [root]
        var files: [String: RuntimeFileSignature] = [:]
        for case let url as URL in enumerator {
            guard let item = RuntimeFileSignature.read(url), item.directory || item.regular else {
                cached = nil; result = status(.unverified); return result
            }
            signatures.append(item)
            if item.regular { files[String(url.path.dropFirst(enumerationRoot.path.count + 1))] = item }
        }
        guard !traversalFailed else {
            cached = nil; result = status(.unverified); return result
        }
        signatures.sort { $0.path < $1.path }
        if cached == signatures { return result }
        cached = nil
        guard let contract,
              let data = try? Data(contentsOf: bundle.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
              manifest.schema == "netal.fg.runtime-bundle.v1", manifest.profile == contract.profile,
              manifest.files.keys.allSatisfy(Self.safePath), manifest.files.values.allSatisfy(\.valid),
              manifest.files["manifest.json"] == nil else {
            result = status(.unverified); return result
        }
        let pinned = Dictionary(uniqueKeysWithValues: contract.model.map { ("model/" + $0.key, $0.value) }).merging(contract.math) { a, _ in a }
        let required = Set(pinned.keys).union(contract.required_engine_files)
        let missing = required.subtracting(files.keys).union(Set(manifest.files.keys).subtracting(files.keys))
        if !missing.isEmpty {
            result = status(.missing(missing.sorted())); return result
        }
        guard required.isSubset(of: Set(manifest.files.keys)),
              Set(files.keys) == Set(manifest.files.keys).union(["manifest.json"]),
              Set(manifest.files.keys.filter { $0.hasPrefix("model/") }) == Set(contract.model.keys.map { "model/" + $0 }),
              pinned.allSatisfy({ manifest.files[$0.key] == $0.value }) else {
            result = status(.unverified); return result
        }
        var wrong: [String] = []
        for (path, expected) in manifest.files.sorted(by: { $0.key < $1.key }) {
            let url = bundle.appendingPathComponent(path)
            guard files[path]?.size == Int64(expected.bytes),
                  let contents = try? Data(contentsOf: url), contents.count == expected.bytes else {
                wrong.append(path); continue
            }
            let actual = SHA256.hash(data: contents).map { String(format: "%02x", $0) }.joined()
            if actual != expected.sha256 { wrong.append(path) }
        }
        // Do not cache a verification assembled while files were changing.
        guard signatures.allSatisfy({ RuntimeFileSignature.read(URL(fileURLWithPath: $0.path)) == $0 }) else {
            result = status(.unverified); return result
        }
        result = wrong.isEmpty ? status(.verified(tokens: contract.model.count, bytes: contract.model.values.reduce(0) { $0 + $1.bytes })) : status(.modified(wrong))
        cached = signatures
        return result
    }
}
