import Foundation
import CoreML

// Per-operation compute-device assignment, straight from Core ML's own planner.
//
// This is the honest way to ask "does it stay on the ANE": not a stopwatch, not a power counter,
// but the plan the runtime will actually execute. Public API, no private ANE classes, no sudo.
func describe(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "other"
    }
}

@main
struct Probe {
    static func main() async {
        for path in CommandLine.arguments.dropFirst() {
            let url = URL(fileURLWithPath: path)
            do {
                let compiled = try await MLModel.compileModel(at: url)
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine
                let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: config)
                let cost = plan
                let structure = try await MLModelStructure.load(contentsOf: compiled)
                guard case let .program(program) = structure,
                      let main = program.functions["main"] else {
                    print("\(url.lastPathComponent): not an ML program"); continue
                }
                var tally: [String: Int] = [:]
                var offAne: [String] = []
                for op in main.block.operations {
                    guard let usage = plan.deviceUsage(for: op) else { continue }
                    let dev = describe(usage.preferred)
                    tally[dev, default: 0] += 1
                    if dev != "ANE" {
                        let supported = usage.supported.map(describe).joined(separator: "/")
                        offAne.append("      \(op.operatorName) -> \(dev)  (can run on: \(supported))")
                    }
                }
                let total = tally.values.reduce(0, +)
                let ane = tally["ANE"] ?? 0
                print("\n\(url.lastPathComponent): \(ane)/\(total) operations on the ANE")
                for (k, v) in tally.sorted(by: { $0.value > $1.value }) { print("   \(k): \(v)") }
                if !offAne.isEmpty {
                    print("   not on ANE:")
                    for line in offAne.prefix(12) { print(line) }
                    if offAne.count > 12 { print("      … \(offAne.count - 12) more") }
                }
            } catch {
                print("\(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
    }
}
