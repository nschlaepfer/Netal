// Produce the transparent artwork the menu-bar app ships.
//
// The brand PNGs are opaque: the pebble sits on a soft white vignette. Dropped onto the app's
// translucent material that background reads as a grey card floating on the glass, which is exactly
// what the material is trying not to look like. So the subject is lifted out once, here, and the
// result is bundled - rather than keyed at runtime, or approximated with a blend mode that only
// works in one appearance.
//
// Vision's foreground-instance mask is the same subject lift Preview's "Remove Background" uses, so
// the edge quality is the system's rather than a luminance threshold's. A threshold would eat the
// pebble's lit top, which is nearly as bright as the background behind it.
//
//   netal-cutout <source.png> <cutout.png> <template.png>
//
// The template is the menu-bar image: macOS renders those monochrome from alpha alone, so it is the
// subject's silhouette with the two eyes punched back out, which is what keeps the shape readable
// at 18 points.
import AppKit
import CoreImage
import Vision

func die(_ message: String) -> Never { FileHandle.standardError.write(Data((message + "\n").utf8)); exit(1) }

let arguments = CommandLine.arguments
guard arguments.count == 4 else { die("usage: netal-cutout <source.png> <cutout.png> <template.png>") }

guard let source = CIImage(contentsOf: URL(fileURLWithPath: arguments[1])) else { die("cannot read \(arguments[1])") }
let context = CIContext()

// 1. Lift the subject.
let request = VNGenerateForegroundInstanceMaskRequest()
let handler = VNImageRequestHandler(ciImage: source)
try? handler.perform([request])
guard let observation = request.results?.first else { die("no subject found") }
let maskBuffer = try! observation.generateScaledMaskForImage(forInstances: observation.allInstances,
                                                             from: handler)
let mask = CIImage(cvPixelBuffer: maskBuffer)

// 2. Apply it as alpha, over nothing.
let blend = CIFilter(name: "CIBlendWithMask")!
blend.setValue(source, forKey: kCIInputImageKey)
blend.setValue(CIImage.empty(), forKey: kCIInputBackgroundImageKey)
blend.setValue(mask, forKey: kCIInputMaskImageKey)
guard let cutout = blend.outputImage?.cropped(to: source.extent) else { die("mask blend failed") }

func writePNG(_ image: CIImage, to path: String) {
    guard let data = context.pngRepresentation(of: image,
                                               format: .RGBA8,
                                               colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
    else { die("cannot encode \(path)") }
    try! data.write(to: URL(fileURLWithPath: path))
}
writePNG(cutout, to: arguments[2])

// 3. The menu-bar template: the silhouette in black, with the eyes cut back out.
//
// Only alpha survives in a template image, so colour is irrelevant and shape is everything.
//
// The eyes are NOT simply the brightest pixels. Measuring the lifted subject, 3% of it sits above
// luminance 234 - far more than two small dots - because a broad specular sheen runs across the
// pebble's top, and an absolute threshold punches that out too, leaving a hole where the highlight
// is. What separates the eyes is that they are *locally* bright: small and high-frequency, where
// the sheen is large and low-frequency. So the discriminator is luminance minus its own local mean,
// which cancels anything broad and keeps anything small.
let width = Int(source.extent.width), height = Int(source.extent.height)
var pixels = [UInt8](repeating: 0, count: width * height * 4)
let bitmap = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                       bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
bitmap.draw(context.createCGImage(cutout, from: cutout.extent)!,
            in: CGRect(x: 0, y: 0, width: width, height: height))

var luminance = [Float](repeating: 0, count: width * height)
var alpha = [Float](repeating: 0, count: width * height)
for i in 0..<(width * height) {
    let a = Float(pixels[i * 4 + 3]) / 255
    alpha[i] = a
    guard a > 0.01 else { continue }
    // Un-premultiply, so an edge pixel is not read as dark.
    let r = Float(pixels[i * 4]) / 255 / a, g = Float(pixels[i * 4 + 1]) / 255 / a,
        b = Float(pixels[i * 4 + 2]) / 255 / a
    luminance[i] = min(1, 0.30 * r + 0.59 * g + 0.11 * b)
}

/// Separable box blur, alpha-weighted so the background does not drag the mean down at the rim.
func localMean(_ value: [Float], radius: Int) -> [Float] {
    var horizontal = [Float](repeating: 0, count: width * height)
    var output = [Float](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            var sum: Float = 0, weight: Float = 0
            for k in max(0, x - radius)...min(width - 1, x + radius) {
                let i = y * width + k
                sum += value[i] * alpha[i]; weight += alpha[i]
            }
            horizontal[y * width + x] = weight > 0 ? sum / weight : 0
        }
    }
    for y in 0..<height {
        for x in 0..<width {
            var sum: Float = 0, count: Float = 0
            for k in max(0, y - radius)...min(height - 1, y + radius) {
                sum += horizontal[k * width + x]; count += 1
            }
            output[y * width + x] = sum / count
        }
    }
    return output
}

// Radius is a fraction of the image so the tool works at any source size; ~3% of the width is
// comfortably larger than an eye and comfortably smaller than the sheen.
let mean = localMean(luminance, radius: max(4, width * 3 / 100))
for i in 0..<(width * height) {
    let excess = luminance[i] - mean[i]
    // Fade rather than clip, so the eye edge is antialiased instead of stair-stepped.
    let eye = min(1, max(0, (excess - 0.06) / 0.10))
    let a = UInt8(max(0, min(255, alpha[i] * (1 - eye) * 255)))
    pixels[i * 4] = 0; pixels[i * 4 + 1] = 0; pixels[i * 4 + 2] = 0   // black, premultiplied
    pixels[i * 4 + 3] = a
}
// Where the eyes are, as fractions of the template, so the menu-bar icon can light them.
//
// The template has them punched out, so colour placed behind it shows through exactly there - no
// second mask needed, and the shape stays whatever the artwork says. Measured from the same eye
// mask the punch used, one pass later, because this tool is the only thing that knows where they
// went.
do {
    var members: [Int] = []
    for i in 0..<(width * height) {
        let excess = luminance[i] - mean[i]
        if alpha[i] > 0.5 && excess > 0.11 { members.append(i) }
    }
    var seen = Set<Int>()
    var blobs: [[Int]] = []
    for start in members where !seen.contains(start) {
        var blob: [Int] = []; var work = [start]; seen.insert(start)
        let candidates = Set(members)
        while let i = work.popLast() {
            blob.append(i)
            let x = i % width, y = i / width
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            where nx >= 0 && nx < width && ny >= 0 && ny < height {
                let j = ny * width + nx
                if candidates.contains(j) && !seen.contains(j) { seen.insert(j); work.append(j) }
            }
        }
        blobs.append(blob)
    }
    // Not simply the two largest blobs. The specular sheen survives the threshold as a blob of
    // its own, and taking the top two picked it: the result was one point above the other rather
    // than side by side, with radii differing by half. Eyes are a *pair* - comparable in size, at
    // comparable height, separated horizontally - so choose the pair that looks like one.
    blobs.sort { $0.count > $1.count }
    func centre(_ blob: [Int]) -> (Double, Double, Double) {
        let cx = Double(blob.map { $0 % width }.reduce(0, +)) / Double(blob.count)
        let cy = Double(blob.map { $0 / width }.reduce(0, +)) / Double(blob.count)
        return (cx / Double(width), cy / Double(height),
                (Double(blob.count) / Double.pi).squareRoot() / Double(width))
    }
    let considered = Array(blobs.prefix(6)).filter { $0.count >= 8 }
    var best: [(Double, Double, Double)]? = nil
    var bestScore = Double.greatestFiniteMagnitude
    for i in considered.indices {
        for j in considered.indices where j > i {
            let a = centre(considered[i]), b = centre(considered[j])
            let sizeRatio = max(a.2, b.2) / max(min(a.2, b.2), 1e-6)
            let dy = abs(a.1 - b.1), dx = abs(a.0 - b.0)
            // Side by side, not stacked; similar size; far enough apart to be two features.
            guard sizeRatio < 1.6, dy < 0.10, dx > 0.06 else { continue }
            // Prefer the most similar, most level pair.
            let score = (sizeRatio - 1) + dy * 4
            if score < bestScore { bestScore = score; best = [a, b].sorted { $0.0 < $1.0 } }
        }
    }
    if let eyes = best {
        let json = "{\n  \"eyes\": [\n" + eyes.map {
            String(format: "    { \"x\": %.5f, \"y\": %.5f, \"r\": %.5f }", $0.0, $0.1, $0.2)
        }.joined(separator: ",\n") + "\n  ]\n}\n"
        let out = URL(fileURLWithPath: arguments[3]).deletingLastPathComponent()
            .appendingPathComponent("NetalTemplateEyes.json")
        try? json.write(to: out, atomically: true, encoding: .utf8)
        print(String(format: "template eyes=(%.3f,%.3f) (%.3f,%.3f) r=%.4f",
                     eyes[0].0, eyes[0].1, eyes[1].0, eyes[1].1, eyes[0].2))
    } else {
        print("template eyes=not-found (\(blobs.count) blobs) — the menu bar icon will not light")
    }
}

let templateImage = bitmap.makeImage()!
let templateData = NSBitmapImageRep(cgImage: templateImage).representation(using: .png, properties: [:])!
try! templateData.write(to: URL(fileURLWithPath: arguments[3]))

print("cutout=\(arguments[2])")
print("template=\(arguments[3])")
print("result=pass")
