import Foundation
import CoreML

func describe(_ d: MLComputeDevice) -> String {
    switch d { case .cpu: return "CPU"; case .gpu: return "GPU"; case .neuralEngine: return "ANE"
    @unknown default: return "?" }
}

func input(for model: MLModel) throws -> MLFeatureProvider {
    var feats: [String: MLFeatureValue] = [:]
    for (name, desc) in model.modelDescription.inputDescriptionsByName {
        guard let c = desc.multiArrayConstraint else { continue }
        // Match the model's declared dtype; a float32 buffer handed to a float16 input is a crash,
        // not an error.
        let a = try MLMultiArray(shape: c.shape, dataType: c.dataType)
        for i in 0..<a.count { a[i] = NSNumber(value: Float(i % 17) * 0.01) }
        feats[name] = MLFeatureValue(multiArray: a)
    }
    return try MLDictionaryFeatureProvider(dictionary: feats)
}

// median of `iters` predictions after warmup; the whole prediction, because a per-layer ANE
// dispatch that is fast but unaffordable to reach is not a win.
func time(_ model: MLModel, _ iters: Int) throws -> (Double, Double) {
    let x = try input(for: model)
    for _ in 0..<10 { _ = try model.prediction(from: x) }
    var ms: [Double] = []
    for _ in 0..<iters {
        let t = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: x)
        ms.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
    }
    ms.sort()
    return (ms[ms.count / 2], ms[0])
}

@main
struct Bench {
    static func main() async {
        func pad(_ t: String, _ n: Int) -> String { t.count >= n ? t : t + String(repeating: " ", count: n - t.count) }
        print(pad("model", 34) + pad("ANE med/min ms", 18) + pad("GPU med/min ms", 18) + "placement")
        for path in CommandLine.arguments.dropFirst() {
            let url = URL(fileURLWithPath: path)
            let name = url.deletingPathExtension().lastPathComponent
            do {
                let compiled = try await MLModel.compileModel(at: url)
                var row: [String] = []
                var place = ""
                for units in [MLComputeUnits.cpuAndNeuralEngine, .cpuAndGPU] {
                    let cfg = MLModelConfiguration(); cfg.computeUnits = units
                    if units == .cpuAndNeuralEngine {
                        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: cfg)
                        if case let .program(p) = plan.modelStructure, let main = p.functions["main"] {
                            var ane = 0, tot = 0
                            for op in main.block.operations {
                                guard let u = plan.deviceUsage(for: op) else { continue }
                                tot += 1; if describe(u.preferred) == "ANE" { ane += 1 }
                            }
                            place = "\(ane)/\(tot) ANE"
                        }
                    }
                    let m = try MLModel(contentsOf: compiled, configuration: cfg)
                    let (med, min) = try time(m, 60)
                    row.append(String(format: "%6.3f/%6.3f", med, min))
                }
                print(pad(name, 34) + pad(row[0], 18) + pad(row[1], 18) + place)
            } catch { print("\(name): \(error.localizedDescription)") }
        }
    }
}
