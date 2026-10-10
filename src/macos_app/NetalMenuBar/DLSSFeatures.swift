import Foundation

/// The Apple silicon chip this Mac has, parsed from the brand string ("Apple M3 Max").
///
/// Used for one thing: saying whether a feature that needs newer hardware will ever arrive on *this*
/// machine. "Requires M5 Max or newer" is a specification; "requires M5 Max or newer — this Mac is
/// M3 Max" is an answer.
struct AppleChip {
    let name: String
    let generation: Int
    let tier: Tier

    enum Tier: Int, Comparable {
        case base = 0, pro = 1, max = 2, ultra = 3
        static func < (a: Tier, b: Tier) -> Bool { a.rawValue < b.rawValue }
    }

    static let current: AppleChip? = {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        let brand = String(cString: buffer)

        // "Apple M3 Max" -> generation 3, tier max. Anything that does not match this shape is left
        // as nil rather than guessed at; an Intel Mac has no business answering this question.
        let parts = brand.split(separator: " ").map(String.init)
        guard parts.first == "Apple", parts.count >= 2, parts[1].hasPrefix("M"),
              let generation = Int(parts[1].dropFirst()) else { return nil }
        let tier: Tier
        switch parts.count > 2 ? parts[2].lowercased() : "" {
        case "pro":   tier = .pro
        case "max":   tier = .max
        case "ultra": tier = .ultra
        default:      tier = .base
        }
        return AppleChip(name: brand, generation: generation, tier: tier)
    }()

}

/// What DLSS can do, and what Netal does today.
///
/// Availability, arithmetic selection and observed frame delivery are distinct states.
struct DLSSFeature: Identifiable {
    enum Availability: Equatable {
        case available                  // implemented, no active session yet
        case live                       // built, and its state comes from the runtime
        case observed(String)           // running in the game through MetalFX; Netal watches the seam
        case planned                    // designed, not built
        case experimental(String)       // research path; not enabled by the release launcher
    }

    let id: String
    let name: String
    let detail: String
    let availability: Availability
    var isObserved: Bool { if case .observed = availability { return true }; return false }

    static func all(chip: AppleChip?, frameGeneration: FrameGeneration? = nil) -> [DLSSFeature] {
        [
            DLSSFeature(id: "sr", name: "Super Resolution",
                        detail: "Upscales the rendered frame",
                        availability: .live),
            // Serving is a runtime observation. A present bundle alone cannot establish
            // that Netal generated a frame or that the game's output uses it.
            DLSSFeature(id: "fg", name: "Frame Generation",
                        detail: frameGeneration.map(Self.frameGenerationDetail) ?? "Interpolates between frames",
                        availability: frameGeneration.map { fg in
                            fg.isServing
                            ? .live
                            : .observed(fg.paused ? "Netal FG is paused; the game uses its fallback path."
                                      : "Netal is not currently reporting generated frames.")
                        } ?? .available),
            DLSSFeature(id: "nr", name: "Neural Rendering",
                        detail: "Native / MLX · off by default",
                        availability: .experimental("Original model weights with native or MLX execution. "
                            + "Runs have been recorded on M3 Max; continuous game integration and moving-frame "
                            + "quality are not release-validated. NR stays off in normal launches.")),
            DLSSFeature(id: "rr", name: "Ray Reconstruction",
                        detail: "Denoising for ray tracing",
                        availability: .planned),
        ]
    }

    /// One line: what is interpolating, at what size, how fast the source frames arrive.
    static func frameGenerationDetail(_ fg: FrameGeneration) -> String {
        if fg.paused { return "Paused · MetalFX fallback" }
        var parts = [fg.isServing ? "Netal" : (fg.armed ? "Armed · waiting" : "MetalFX fallback")]
        if let lowering = fg.lowering { parts.append(lowering.label) }
        if let fps = fg.sourceFps, fps > 0 { parts.append(String(format: "%.0f source fps", fps)) }
        return parts.joined(separator: " · ")
    }

    static func frameGenerationHelp(_ fg: FrameGeneration?) -> String {
        let mode: String
        switch fg?.lowering {
        case .accelerated: mode = "Accelerated arithmetic is selected; output can differ from the reference path."
        case .reference: mode = "Reference arithmetic is selected; this is not a guarantee of pixel parity in every game."
        case nil: mode = "The arithmetic mode was not recorded for this session."
        }
        return "Netal generates intermediate frames using a 640×360 internal network profile. "
            + mode + " Switching FG off selects the game's MetalFX fallback path."
    }

}
