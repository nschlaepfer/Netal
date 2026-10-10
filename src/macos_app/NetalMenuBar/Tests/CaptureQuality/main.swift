import Foundation
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Metal

// Ground truth is drawn analytically at the requested time/resolution. It is not
// produced by the interpolation shader or by upscaling one of the input frames.
enum Scene: String, CaseIterable {
    case diagonal, fence, occlusion, hud
    case courtyard, orbit, fastTurn = "fast-turn", sceneCut = "scene-cut"
    static let analytical: [Scene] = [.diagonal, .fence, .occlusion, .hud]
    var title: String {
        switch self {
        case .diagonal: return "Thin diagonal · camera pan"
        case .fence: return "Straight fence · repeated detail"
        case .occlusion: return "Moving foreground · occlusion"
        case .hud: return "Fixed HUD · moving background"
        case .courtyard: return "Courtyard · walking and parallax"
        case .orbit: return "Courtyard · camera orbit"
        case .fastTurn: return "Fast camera turn · dropped captures"
        case .sceneCut: return "Courtyard · hard scene change"
        }
    }
    var simulationProfile: Int? {
        switch self {
        case .courtyard: return 0
        case .orbit: return 1
        case .fastTurn: return 2
        case .sceneCut: return 3
        default: return nil
        }
    }
    func sampleTime(_ frame: Int) -> Double {
        // Missing two source instants every ninth capture increases motion and
        // leaves real gaps on the exported movie's source timeline.
        Double(frame + (self == .fastTurn ? (frame / 9) * 2 : 0))
    }
    func crossesCut(_ frame: Int) -> Bool { self == .sceneCut && frame == 12 }
    func color(x: Double, y: Double, time: Double) -> SIMD3<Double> {
        let world = x + time * 7
        switch self {
        case .diagonal:
            let center = 550 + 0.31 * (y - 360) + time * 7
            let coverage = min(1, max(0, 1.4 - abs(x - center)))
            return SIMD3(repeating: 0.08 + coverage * 0.86)
        case .fence:
            let distance = abs((world + 18).truncatingRemainder(dividingBy: 36) - 18)
            let vertical = min(1, max(0, 2.0 - distance))
            let horizontal = abs(y - 220) < 2 || abs(y - 500) < 2
            return SIMD3(repeating: horizontal ? 0.9 : 0.12 + vertical * 0.75)
        case .occlusion, .hud:
            let cx = Int(floor(world / 8)), cy = Int(floor(y / 8))
            let pattern = Double((cx * 67 + cy * 43 + cx * cy * 19) % 160) / 255 + 0.08
            var color = SIMD3(pattern * 0.75, pattern * 0.9, pattern)
            if self == .occlusion {
                let left = 400 + time * 19
                if x >= left && x < left + 170 && y >= 120 && y < 580 {
                    let shade = 0.4 + 0.15 * sin((x - left) / 7) * sin(y / 9)
                    color = SIMD3(shade, shade * 0.3, shade * 0.2)
                }
            } else if ((40..<320).contains(x) && (40..<43).contains(y))
                        || ((40..<43).contains(x) && (40..<130).contains(y))
                        || ((76..<79).contains(x) && (60..<100).contains(y))
                        || ((96..<99).contains(x) && (60..<100).contains(y))
                        || ((76..<99).contains(x) && (78..<81).contains(y)) {
                color = SIMD3(repeating: 1)
            }
            return color
        default: preconditionFailure("3D fixtures must use the independent simulation renderer")
        }
    }
}

enum Mode: String, CaseIterable {
    case sr, fg, combined
    var generated: Bool { self != .sr }
    var scale: Int { self == .fg ? 1 : 2 }
}

struct Failure: Error, CustomStringConvertible { let description: String }
let width = 1280, height = 720

struct Options {
    var frames = 6
    var scenes = Scene.analytical
    var modes = Mode.allCases
    var saveEvery = 1
    var video = false
    var fixtureHash = ""
    var backend = "netal"
    init(_ args: [String]) throws {
        var i = 0, seen = Set<String>()
        while i < args.count {
            let key = args[i]
            guard seen.insert(key).inserted else { throw Failure(description: "Duplicate option: \(key)") }
            if key == "--video" { video = true; i += 1; continue }
            guard i + 1 < args.count else { throw Failure(description: "Missing value for \(key)") }
            let value = args[i + 1]
            switch key {
            case "--frames":
                guard let count = Int(value), (3...120).contains(count) else { throw Failure(description: "Frames must be 3…120") }
                frames = count
            case "--save-every":
                guard let count = Int(value), (1...120).contains(count) else { throw Failure(description: "Save interval must be 1…120") }
                saveEvery = count
            case "--scenes":
                let names = value.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                scenes = names.compactMap(Scene.init(rawValue:))
                guard scenes.count == names.count, !scenes.isEmpty, Set(names).count == names.count else { throw Failure(description: "Unknown or duplicate scene") }
            case "--modes":
                let names = value.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                modes = names.compactMap(Mode.init(rawValue:))
                guard modes.count == names.count, !modes.isEmpty, Set(names).count == names.count else { throw Failure(description: "Unknown or duplicate mode") }
            case "--fixture-hash": fixtureHash = value
            case "--backend":
                guard ["netal", "apple-video"].contains(value) else { throw Failure(description: "Choose netal or apple-video backend") }
                backend = value
            default: throw Failure(description: "Unknown option: \(key)")
            }
            i += 2
        }
        guard !scenes.contains(.sceneCut) || frames >= 14 else { throw Failure(description: "Scene-cut runs require at least 14 source frames") }
    }
}

func render(_ scene: Scene, time: Double, scale: Int, simulation: GameSimulation) throws -> [UInt8] {
    if let profile = scene.simulationProfile {
        return try simulation.render(profile: profile, time: time, width: width * scale, height: height * scale)
    }
    return draw(scene, time: time, scale: scale)
}

func draw(_ scene: Scene, time: Double, scale: Int) -> [UInt8] {
    let w = width * scale, h = height * scale
    var pixels = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let rgb = scene.color(x: (Double(x) + 0.5) / Double(scale),
                                  y: (Double(y) + 0.5) / Double(scale), time: time)
            let i = (y * w + x) * 4
            for c in 0..<3 { pixels[i + c] = UInt8((min(1, max(0, rgb[2 - c])) * 255).rounded()) }
        }
    }
    return pixels
}

func buffer(_ pixels: [UInt8]) throws -> CVPixelBuffer {
    var result: CVPixelBuffer?
    let attributes: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any]]
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                             attributes as CFDictionary, &result) == kCVReturnSuccess,
          let result else { throw Failure(description: "Could not allocate fixture pixels") }
    guard CVPixelBufferLockBaseAddress(result, []) == kCVReturnSuccess,
          let destination = CVPixelBufferGetBaseAddress(result) else {
        throw Failure(description: "Could not write fixture pixels")
    }
    defer { CVPixelBufferUnlockBaseAddress(result, []) }
    pixels.withUnsafeBytes { source in
        for y in 0..<height {
            destination.advanced(by: y * CVPixelBufferGetBytesPerRow(result)).copyMemory(
                from: source.baseAddress!.advanced(by: y * width * 4), byteCount: width * 4)
        }
    }
    return result
}

func read(_ texture: MTLTexture, device: MTLDevice) throws -> [UInt8] {
    let row = ((texture.width * 4 + 255) / 256) * 256
    guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
          let staging = device.makeBuffer(length: row * texture.height, options: .storageModeShared),
          let blit = command.makeBlitCommandEncoder() else { throw Failure(description: "Could not read GPU output") }
    blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(x: 0, y: 0, z: 0),
              sourceSize: .init(width: texture.width, height: texture.height, depth: 1), to: staging,
              destinationOffset: 0, destinationBytesPerRow: row, destinationBytesPerImage: row * texture.height)
    blit.endEncoding(); command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else { throw Failure(description: "GPU readback failed") }
    var result = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
    result.withUnsafeMutableBytes { destination in
        for y in 0..<texture.height {
            destination.baseAddress!.advanced(by: y * texture.width * 4).copyMemory(
                from: staging.contents().advanced(by: y * row), byteCount: texture.width * 4)
        }
    }
    return result
}

func savePNG(_ pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
          let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                              bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: [.byteOrder32Little, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)],
                              provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw Failure(description: "Could not create PNG") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw Failure(description: "Could not save PNG") }
}

func measure(_ actual: [UInt8], _ ideal: [UInt8], w: Int, h: Int, scene: Scene, time: Double, scale: Int,
             previous: (actual: [UInt8], ideal: [UInt8])?) -> [String: Any] {
    var total = 0.0, edgeError = 0.0, edges = 0
    var changedError = 0.0
    // Include edges in either image, so invented edges count as errors too.
    for y in 1..<(h - 1) {
        for x in 1..<(w - 1) {
            let i = (y * w + x) * 4
            var error = 0.0, gradient = 0
            for c in 0..<3 {
                error += Double(abs(Int(actual[i + c]) - Int(ideal[i + c]))) / (255 * 3)
                for offset in [-4, 4, -w * 4, w * 4] {
                    gradient = max(gradient, abs(Int(ideal[i + c]) - Int(ideal[i + offset + c])))
                    gradient = max(gradient, abs(Int(actual[i + c]) - Int(actual[i + offset + c])))
                }
            }
            total += error
            if gradient >= 24 {
                edgeError += error; edges += 1
                if let previous {
                    for c in 0..<3 {
                        let residual = Int(actual[i+c]) - Int(ideal[i+c])
                        let old = Int(previous.actual[i+c]) - Int(previous.ideal[i+c])
                        changedError += Double(abs(residual-old)) / (255 * 3)
                    }
                }
            }
        }
    }
    let edgeMAE = edges > 0 ? edgeError / Double(edges) : 0
    var result: [String: Any] = ["mae": total / Double((w - 2) * (h - 2)),
        "edge_mae": edgeMAE, "edge_pixels": edges, "edge_gate_passed": edges > 0 && edgeMAE <= 0.06]
    if previous != nil, edges > 0 { result["edge_error_change_mae"] = changedError / Double(edges) }
    if scene == .diagonal {
        var errors = [Double](), missing = 0
        for y in (20 * scale)..<((height - 20) * scale) {
            let expected = (550 + 0.31 * ((Double(y) + 0.5) / Double(scale) - 360) + time * 7) * Double(scale)
            var weight = 0.0, moment = 0.0
            for x in max(0, Int(expected) - 16 * scale)..<min(w, Int(expected) + 16 * scale) {
                let v = max(0, Double(actual[(y * w + x) * 4]) / 255 - 0.4)
                weight += v; moment += (Double(x) + 0.5) * v
            }
            if weight < 0.1 { missing += 1 }
            else { errors.append(abs(moment / weight - expected)) }
        }
        errors.sort()
        let p95 = errors.isEmpty ? Double(w) : errors[min(errors.count - 1, Int(Double(errors.count) * 0.95))]
        let fraction = Double(missing) / Double((height - 40) * scale)
        result["line_error_p95_px"] = p95
        result["missing_line_fraction"] = fraction
        result["line_gate_passed"] = p95 <= 1 && fraction <= 0.01
    }
    return result
}

let viewer = #"""
<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Netal · Offline quality lab</title><style>
body{margin:0;background:#151618;color:#f2f2f3;font:15px -apple-system,BlinkMacSystemFont,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:26px;margin:0 0 8px}p{color:#aaa;line-height:1.5}button,select,input{font:inherit}button,select{padding:8px 12px;border-radius:8px;border:1px solid #555;background:#292b2f;color:inherit}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0;min-width:0}img{width:100%;background:#000;image-rendering:auto}figcaption{padding:8px 0;color:#aaa}.metrics{font:14px ui-monospace,monospace;white-space:pre-wrap;color:#bbb}.warn{color:#ffc57b}input{width:220px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style><main><h1>Netal · Offline quality lab</h1><p>Synthetic scenes processed by Netal’s real Metal engine. No game or screen recording. The reference is drawn independently at the exact target time and resolution.</p>
<p id="provider"></p><nav><select id="scene" aria-label="Scene"></select><select id="mode" aria-label="Processing mode"><option value="sr">SR only</option><option value="fg" selected>FG only</option><option value="combined">SR + FG</option></select><button id="play">Play slowly</button><input id="frame" type="range" min="0" max="4" value="0" aria-label="Frame"><span id="position"></span></nav>
<p id="movie-label" hidden>Motion replay: Output under test on the left, independent reference on the right. Preview is resized and compressed.</p><video id="movie" controls playsinline loop style="width:100%;max-height:650px" hidden></video><div class="grid"><figure><img id="actual" alt="Netal output"><figcaption>Output under test</figcaption></figure><figure><img id="ideal" alt="Independent reference"><figcaption>Independent reference</figcaption></figure></div>
<p id="status" class="warn"></p><p class="metrics" id="metrics"></p><p>Playback is deliberately slowed for inspection. GPU processing cost is not game FPS or end-to-end latency. Fixture gates are engineering targets; passing them does not certify gameplay quality.</p></main>
<script src="report.js"></script><script>
const r=window.NETAL_REPORT,s=document.querySelector('#scene'),m=document.querySelector('#mode'),f=document.querySelector('#frame');let timer;
document.querySelector("#provider").textContent=r.backend==="apple-video"?"Research comparator: Apple VideoToolbox FG + MetalFX spatial SR":"Production comparator: Netal block-motion FG + MetalFX spatial SR";
for(const scene of r.scenes){const o=document.createElement('option');o.value=scene.id;o.textContent=scene.title;s.append(o)}
function show(){const run=r.runs.find(x=>x.scene===s.value&&x.mode===m.value),frames=run.frames.filter(x=>x.actual),n=Number(f.value);f.max=frames.length-1;const row=frames[Math.min(n,frames.length-1)];const movie=document.querySelector("#movie");movie.hidden=!run.video;document.querySelector("#movie-label").hidden=!run.video;if(run.video&&movie.getAttribute("src")!==run.video)movie.src=run.video;document.querySelector('#actual').src=row.actual;document.querySelector('#ideal').src=row.ideal;document.querySelector('#position').textContent=`Frame ${row.frame}`;document.querySelector('#status').textContent=row.passed?'This frame meets the fixture targets.':'This frame fails a fixture target — quality work remains.';document.querySelector('#metrics').textContent=JSON.stringify({edge_error:row.edge_mae,line_error_p95_px:row.line_error_p95_px,missing_line_fraction:row.missing_line_fraction,gpu_processing_ms:row.gpu_ms,processing_wall_ms:row.processing_wall_ms,edge_error_change:row.edge_error_change_mae},null,2)}
for(const o of [...m.options])if(!r.runs.some(x=>x.mode===o.value))o.remove();s.onchange=m.onchange=()=>{f.value=0;show()};f.oninput=show;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play slowly'}else{timer=setInterval(()=>{f.value=(Number(f.value)+1)%(Number(f.max)+1);show()},350);document.querySelector('#play').textContent='Pause'}};show();
</script></html>
"""#

do {
    guard CommandLine.arguments.count >= 3, let device = MTLCreateSystemDefaultDevice() else {
        throw Failure(description: "Usage: capture-quality NEW_OUTPUT_DIRECTORY ENGINE_SHA256 [options]; Metal is required")
    }
    let options = try Options(Array(CommandLine.arguments.dropFirst(3)))
    let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    guard !FileManager.default.fileExists(atPath: output.path) else { throw Failure(description: "Output already exists") }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    var runs = [[String: Any]](), failures = 0
    let processor: any QualityFrameProcessor
    if options.backend == "apple-video" {
        if #available(macOS 26.0, *) { processor = try AppleVideoComparator(device: device) }
        else { throw Failure(description: "The Apple comparison backend requires macOS 26 or newer") }
    } else { processor = try CaptureFrameProcessor(device: device) }
    let simulation = try GameSimulation(device: device)
    for scene in options.scenes {
        let sources = try (0..<options.frames).map { try render(scene, time: scene.sampleTime($0), scale: 1, simulation: simulation) }
        for mode in options.modes {
            processor.reset()
            var frames = [[String: Any]]()
            var previous: (actual: [UInt8], ideal: [UInt8])?
            let w = width * mode.scale, h = height * mode.scale
            let movieName = "\(scene.rawValue)-\(mode.rawValue).mp4"
            let movie = options.video ? try ReplayVideo(url: output.appendingPathComponent(movieName)) : nil
            for index in 0..<options.frames {
                try autoreleasepool {
                    let input = try buffer(sources[index])
                    let processingStart = ProcessInfo.processInfo.systemUptime
                    let processed = try processor.process(pixelBuffer: input,
                        outputWidth: w, outputHeight: h, frameGeneration: mode.generated)
                    let processingWall = (ProcessInfo.processInfo.systemUptime - processingStart) * 1000
                    let currentTime = scene.sampleTime(index)
                    let currentReference = (movie != nil || !mode.generated || scene.crossesCut(index))
                        ? try render(scene, time: currentTime, scale: mode.scale, simulation: simulation) : nil
                    if index == 0 {
                        if let movie, let currentReference {
                            try movie.append(actual: read(processed.source, device: device), ideal: currentReference,
                                imageWidth: w, imageHeight: h, time: 0)
                        }
                        return
                    }
                    let midTime = (scene.sampleTime(index-1) + currentTime) / 2
                    let expectsGenerated = mode.generated && !scene.crossesCut(index)
                    let time = expectsGenerated ? midTime : currentTime
                    let ideal = expectsGenerated
                        ? try render(scene, time: time, scale: mode.scale, simulation: simulation) : currentReference!
                    let observedGenerated = mode.generated && processed.midpoint != nil
                    let texture = mode.generated ? processed.midpoint ?? processed.source : processed.source
                    let actual = try read(texture, device: device)
                    // A discontinuity has no meaningful image-space temporal residual.
                    if scene.crossesCut(index) { previous = nil }
                    var row = measure(actual, ideal, w: w, h: h, scene: scene, time: time, scale: mode.scale, previous: previous)
                    previous = (actual, ideal)
                    let passed = (row["edge_gate_passed"] as? Bool == true)
                        && (row["line_gate_passed"] as? Bool ?? true) && observedGenerated == expectsGenerated
                    if !passed { failures += 1 }
                    if index == 1 || index == options.frames-1 || index % options.saveEvery == 0 {
                        let prefix = "\(scene.rawValue)-\(mode.rawValue)-\(index)"
                        try savePNG(actual, width: w, height: h, to: output.appendingPathComponent(prefix + "-actual.png"))
                        try savePNG(ideal, width: w, height: h, to: output.appendingPathComponent(prefix + "-ideal.png"))
                        row["actual"] = prefix + "-actual.png"; row["ideal"] = prefix + "-ideal.png"
                    }
                    if let movie, let currentReference {
                        if observedGenerated {
                            let midpointReference = expectsGenerated ? ideal
                                : try render(scene, time: midTime, scale: mode.scale, simulation: simulation)
                            try movie.append(actual: actual, ideal: midpointReference, imageWidth: w, imageHeight: h, time: midTime / 30)
                        }
                        try movie.append(actual: read(processed.source, device: device), ideal: currentReference,
                            imageWidth: w, imageHeight: h, time: currentTime / 30)
                    }
                    row["frame"] = index; row["target_time"] = time; row["passed"] = passed
                    row["generated"] = observedGenerated; row["expected_generated"] = expectsGenerated
                    row["scene_cut"] = scene.crossesCut(index)
                    row["upscaled"] = processed.spatiallyUpscaled
                    if let gpu = processed.gpuMilliseconds { row["gpu_ms"] = gpu }
                    row["processing_wall_ms"] = processingWall
                    if let note = processed.frameGenerationNote { row["processing_note"] = note }
                    frames.append(row)
                }
            }
            if let movie { try movie.finish(at: (scene.sampleTime(options.frames-1)+1) / 30) }
            var run: [String: Any] = ["scene": scene.rawValue, "mode": mode.rawValue, "frames": frames,
                "input_size": [width, height], "output_size": [w, h]]
            if movie != nil { run["video"] = movieName }
            runs.append(run)
            print("\(scene.rawValue) / \(mode.rawValue): \(frames.filter { $0["passed"] as? Bool == true }.count)/\(frames.count) frame gates pass")
            fflush(stdout)
        }
    }
    let report: [String: Any] = ["schema": "netal.offline-window-quality.v2", "device": device.name,
        "engine_sha256": CommandLine.arguments[2], "fixture_sha256": options.fixtureHash,
        "source_frames_per_scene": options.frames, "nominal_source_fps": 30,
        "backend": options.backend, "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        "save_every": options.saveEvery, "video_preview_size": [1600, 450],
        "game_used": false, "screen_capture_used": false, "dlss_model_executed": false,
        "failed_frame_gates": failures, "status": failures == 0 ? "fixture-gates-passed" : "quality-work-required",
        "limits": ["edge_mae": 0.06, "line_error_p95_px": 1.0, "missing_line_fraction": 0.01],
        "scenes": options.scenes.map { ["id": $0.rawValue, "title": $0.title] }, "runs": runs]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: output.appendingPathComponent("report.json"))
    try ("window.NETAL_REPORT = " + String(decoding: data, as: UTF8.self) + ";").write(
        to: output.appendingPathComponent("report.js"), atomically: true, encoding: .utf8)
    try viewer.write(to: output.appendingPathComponent("index.html"), atomically: true, encoding: .utf8)
    print("Failed frame gates: \(failures). Viewer: \(output.appendingPathComponent("index.html").path)")
    exit(failures == 0 ? 0 : 2)
} catch {
    fputs("Offline quality harness failed: \(error)\n", stderr)
    exit(1)
}
