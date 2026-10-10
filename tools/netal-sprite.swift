// Cut two poses out of the brand banners and register them as one sprite pair.
//
// The About view animates the orb with two frames: landed, and tucked-and-lifted. They come from
// two different banner renders, at different sizes and positions in their frames, so swapping the
// raw images would read as two pictures rather than as one character moving. Registration is the
// whole job:
//
//   - lift each subject with Vision, which drops the baked-in ground shadow along with the
//     background. The shadow is redrawn in SwiftUI instead, because a shadow that cannot respond to
//     the lift is what makes this kind of animation look pasted on.
//   - scale both so their silhouettes are the same WIDTH. Width, not height: the character is
//     wider than it is tall in both poses, and its width barely changes between them, while its
//     height changes a lot - that height change is the tuck, and it should survive.
//   - centre both horizontally, and align them on a common ground line, so the only thing the
//     animation has to add is the lift itself.
//
//   netal-sprite <landed.png> <lifted.png> <out-rest.png> <out-hop.png>
import AppKit
import CoreImage
import Vision

func die(_ m: String) -> Never { FileHandle.standardError.write(Data((m + "\n").utf8)); exit(1) }
let arguments = CommandLine.arguments
guard arguments.count >= 5 else {
    die("usage: netal-sprite <landed> <lifted> <out-rest> <out-hop> [scale] [--overlay <sheet.png>]")
}
let context = CIContext()

/// Fill any region of the mask that the background cannot reach from the image border.
func fillInteriorHoles(_ mask: CIImage, extent: CGRect) -> CIImage {
    let w = Int(extent.width), h = Int(extent.height)
    guard w > 0, h > 0, let cg = context.createCGImage(mask, from: extent) else { return mask }
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    let bitmap = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                           bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    bitmap.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

    var isBackground = [Bool](repeating: false, count: w * h)
    for i in 0..<(w * h) { isBackground[i] = pixels[i * 4] < 128 }

    var exterior = [Bool](repeating: false, count: w * h)
    var stack: [Int] = []
    for x in 0..<w {
        for y in [0, h - 1] where isBackground[y * w + x] { stack.append(y * w + x) }
    }
    for y in 0..<h {
        for x in [0, w - 1] where isBackground[y * w + x] { stack.append(y * w + x) }
    }
    while let i = stack.popLast() {
        if exterior[i] || !isBackground[i] { continue }
        exterior[i] = true
        let x = i % w, y = i / w
        if x > 0 { stack.append(i - 1) }
        if x < w - 1 { stack.append(i + 1) }
        if y > 0 { stack.append(i - w) }
        if y < h - 1 { stack.append(i + w) }
    }

    var holes: [(Int, Int)] = []
    for i in 0..<(w * h) {
        if isBackground[i] && !exterior[i] { holes.append((i % w, i / w)) }
        let value: UInt8 = exterior[i] ? 0 : 255
        pixels[i * 4] = value; pixels[i * 4 + 1] = value
        pixels[i * 4 + 2] = value; pixels[i * 4 + 3] = 255
    }
    holePixels = holes
    if !holes.isEmpty { print("  filled \(holes.count) interior mask pixels (the eyes)") }
    return CIImage(cgImage: bitmap.makeImage()!)
}

/// Set by fillInteriorHoles: the pixels it filled. Kept only for the log line.
var holePixels: [(Int, Int)] = []

/// Push interior colour outward into the soft edge.
///
/// A feathered alpha is only half of a clean cut-out. The edge pixels in the *source* are already a
/// blend of the subject and the white backdrop - that is what an antialiased render is - so however
/// good the alpha is, those pixels carry the backdrop with them and show as a pale halo. The steep
/// remap that used to be here hid that by making the edge nearly binary, at the cost of the
/// silhouette looking rough.
///
/// So the colour is decontaminated instead: a partly transparent pixel takes an alpha-weighted
/// average of its neighbours, which are mostly interior and therefore mostly the subject's own
/// colour. Fully opaque pixels are left exactly as rendered. That allows a genuinely soft edge
/// with no halo.
func extendEdgeColour(_ image: CIImage, radius: Int) -> CIImage {
    let w = Int(image.extent.width), h = Int(image.extent.height)
    guard w > 2, h > 2, let cg = context.createCGImage(image, from: image.extent) else { return image }
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    let bitmap = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                           bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    bitmap.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

    let count = w * h
    var alpha = [Float](repeating: 0, count: count)
    var straight = [[Float]](repeating: [Float](repeating: 0, count: count), count: 3)
    for i in 0..<count {
        let a = Float(pixels[i * 4 + 3]) / 255
        alpha[i] = a
        guard a > 0.004 else { continue }
        for c in 0..<3 { straight[c][i] = Float(pixels[i * 4 + c]) / 255 / a }
    }

    // Separable and alpha-weighted: interior pixels carry the weight, transparent ones carry none,
    // so the average at an edge is dominated by the subject rather than by what is behind it.
    func weightedBlur(_ value: [Float]) -> [Float] {
        var horizontal = [Float](repeating: 0, count: count)
        var weightH = [Float](repeating: 0, count: count)
        for y in 0..<h {
            for x in 0..<w {
                var sum: Float = 0, weight: Float = 0
                for k in max(0, x - radius)...min(w - 1, x + radius) {
                    let i = y * w + k
                    sum += value[i] * alpha[i]; weight += alpha[i]
                }
                horizontal[y * w + x] = sum; weightH[y * w + x] = weight
            }
        }
        var output = [Float](repeating: 0, count: count)
        for y in 0..<h {
            for x in 0..<w {
                var sum: Float = 0, weight: Float = 0
                for k in max(0, y - radius)...min(h - 1, y + radius) {
                    sum += horizontal[k * w + x]; weight += weightH[k * w + x]
                }
                output[y * w + x] = weight > 0 ? sum / weight : 0
            }
        }
        return output
    }
    let extended = (0..<3).map { weightedBlur(straight[$0]) }

    for i in 0..<count {
        let a = alpha[i]
        guard a > 0.004 else {
            pixels[i * 4] = 0; pixels[i * 4 + 1] = 0; pixels[i * 4 + 2] = 0
            continue
        }
        let blend = min(1, max(0, (0.98 - a) / 0.98))
        for c in 0..<3 {
            let value = straight[c][i] * (1 - blend) + extended[c][i] * blend
            pixels[i * 4 + c] = UInt8(max(0, min(255, value * a * 255)))
        }
    }
    return CIImage(cgImage: bitmap.makeImage()!)
}


func subject(_ path: String) -> CIImage {
    guard let source = CIImage(contentsOf: URL(fileURLWithPath: path)) else { die("cannot read \(path)") }
    let request = VNGenerateForegroundInstanceMaskRequest()
    let handler = VNImageRequestHandler(ciImage: source)
    try? handler.perform([request])
    guard let observation = request.results?.first,
          let buffer = try? observation.generateScaledMaskForImage(forInstances: observation.allInstances,
                                                                   from: handler)
    else { die("no subject in \(path)") }
    // Vision generates its mask at a lower resolution and scales it up, so its alpha arrives with
    // visible stair-steps. Left alone those show as a serrated silhouette. Blurring the mask
    // feathers the edge; the gamma afterwards pulls the midpoint back so the feather does not also
    // inflate the silhouette by a pixel all the way round.
    // Vision punches holes in the mask where the eyes are. They are emissive and nearly as bright
    // as the white backdrop, so the segmenter calls them background - which shows up as the eyes
    // being transparent rather than white, with the panel behind them. Eroding the mask made it
    // worse, because eroding a mask widens its holes.
    //
    // Filled by flooding the background inward from the border: whatever the flood cannot reach is
    // interior, whatever its brightness. That fills a hole of any size and, unlike a morphological
    // close, does not touch the outer silhouette - the gaps between the feet in the landed pose
    // are concavities open to the border, so they survive exactly as they are.
    let raw = fillInteriorHoles(CIImage(cvPixelBuffer: buffer), extent: source.extent)

    // Vision's mask is generated well below the image's resolution and scaled up, so its contour
    // arrives with stair-steps of a couple of pixels. This is what has to be softened, and the
    // amount is set against the source width so the tool behaves the same on any render.
    let feather = CIFilter(name: "CIGaussianBlur")!
    feather.setValue(raw, forKey: kCIInputImageKey)
    feather.setValue(max(2.0, source.extent.width / 320), forKey: kCIInputRadiusKey)
    // Then trim the tail. A blur leaves a long run of very low alpha, and Vision's mask has faint
    // spill of its own around a soft subject; left in, that reads as haze around the silhouette and
    // inflates the bounding box measured below, so the sprite gets padded with fog.
    //
    // The remap used to be steep (x1.6), which cut the tail but also crushed the feather back to a
    // near-binary edge - undoing the blur and leaving the silhouette looking rough. It is gentle
    // now: 0.10 and below go to nothing, 0.90 and above go to solid, and the several pixels in
    // between survive as a real gradient.
    let tighten = CIFilter(name: "CIColorMatrix")!
    tighten.setValue(feather.outputImage!.cropped(to: raw.extent), forKey: kCIInputImageKey)
    tighten.setValue(CIVector(x: 1.25, y: 0, z: 0, w: 0), forKey: "inputRVector")
    tighten.setValue(CIVector(x: 0, y: 1.25, z: 0, w: 0), forKey: "inputGVector")
    tighten.setValue(CIVector(x: 0, y: 0, z: 1.25, w: 0), forKey: "inputBVector")
    tighten.setValue(CIVector(x: -0.125, y: -0.125, z: -0.125, w: 0), forKey: "inputBiasVector")
    let clamped = CIFilter(name: "CIColorClamp")!
    clamped.setValue(tighten.outputImage!, forKey: kCIInputImageKey)

    let blend = CIFilter(name: "CIBlendWithMask")!
    blend.setValue(source, forKey: kCIInputImageKey)
    blend.setValue(CIImage.empty(), forKey: kCIInputBackgroundImageKey)
    blend.setValue(clamped.outputImage!, forKey: kCIInputMaskImageKey)
    guard let masked = blend.outputImage?.cropped(to: source.extent) else { die("mask failed") }
    let cut = extendEdgeColour(masked, radius: max(2, Int(source.extent.width) / 300))
    // Trim to the opaque bounds so both poses are measured from their own silhouette rather than
    // from wherever they happened to sit in a 1200x320 banner.
    let bounds = context.createCGImage(cut, from: cut.extent).map { image -> CGRect in
        var minX = image.width, minY = image.height, maxX = 0, maxY = 0
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let bitmap = CGContext(data: &pixels, width: image.width, height: image.height,
                               bitsPerComponent: 8, bytesPerRow: image.width * 4,
                               space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        bitmap.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        for y in 0..<image.height {
            for x in 0..<image.width where pixels[(y * image.width + x) * 4 + 3] > 16 {
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        // CoreImage's origin is bottom-left; the bitmap above is top-down.
        return CGRect(x: CGFloat(minX), y: CGFloat(image.height - 1 - maxY),
                      width: CGFloat(maxX - minX + 1), height: CGFloat(maxY - minY + 1))
    } ?? cut.extent
    return cut.cropped(to: bounds).transformed(by: .init(translationX: -bounds.minX, y: -bounds.minY))
}


let rest = subject(arguments[1])       // landed
let hop  = subject(arguments[2])       // lifted

// Scale is a parameter, not a measurement, and that is deliberate.
//
// The two poses are separate renders at different camera distances, so there is no "true" relative
// size recoverable from the pixels - the artist chose it, and only an eye can judge it. Automated
// landmark detection was tried and abandoned: the eyes are the only distinctive feature, they are
// contaminated by the specular sheen, and the detector is not scale-invariant, so measuring the
// same subject on two differently sized canvases disagreed by 22% and put the head 60 px out.
//
// So the pair is registered the way an animator would: horizontally on the silhouette's centre,
// which is stable because the character is near-symmetric, vertically on a shared ground line, and
// with the relative scale set by looking at an overlay. --overlay writes that overlay.
// 1.10 was chosen from the overlay sheet, and it is a compromise rather than a fit. The eyes line
// up best around 1.15-1.20; the bodies match best around 1.05-1.10; no single number does both,
// because the two renders differ in camera distance AND angle. Body match wins: a character that
// changes size mid-hop reads as a different picture, while a few pixels of eye shift inside a
// quarter-second jump is not perceptible.
let scale = CGFloat(arguments.count > 5 ? Double(arguments[5]) ?? 1.10 : 1.10)
let hopImage = hop.transformed(by: .init(scaleX: scale, y: scale))

let canvasWidth = max(rest.extent.width, hopImage.extent.width).rounded(.up)
let canvasHeight = max(rest.extent.height, hopImage.extent.height).rounded(.up)

/// Where the eyes are, in the finished sprite, as fractions of the canvas.
///
/// Measured here rather than guessed in the app, because the app must not have to know how the
/// artwork was cropped. Both frames share one canvas, so a single measurement pass gives
/// coordinates that stay valid whatever size the sprite is drawn at.
///
/// An eye is not simply bright - the specular sheen across the pebble's top is brighter still - it
/// is *locally* bright: small and high-frequency where the sheen is large and low-frequency. So the
/// discriminator is luminance minus its own local mean, and then the two largest connected blobs,
/// because averaging every pixel over the threshold lets a little sheen drag the centre around.
func measureEyes(_ image: CIImage) -> [(CGPoint, CGFloat)] {
    let w = Int(image.extent.width), h = Int(image.extent.height)
    guard w > 8, h > 8, let cg = context.createCGImage(image, from: image.extent) else { return [] }
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    let bitmap = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                           bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    bitmap.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

    let count = w * h
    var luminance = [Float](repeating: 0, count: count), alpha = [Float](repeating: 0, count: count)
    for i in 0..<count {
        let a = Float(pixels[i * 4 + 3]) / 255
        alpha[i] = a
        guard a > 0.01 else { continue }
        luminance[i] = min(1, 0.30 * Float(pixels[i * 4]) / 255 / a
                            + 0.59 * Float(pixels[i * 4 + 1]) / 255 / a
                            + 0.11 * Float(pixels[i * 4 + 2]) / 255 / a)
    }
    let radius = max(4, w * 3 / 100)
    var horizontal = [Float](repeating: 0, count: count), mean = [Float](repeating: 0, count: count)
    for y in 0..<h { for x in 0..<w {
        var sum: Float = 0, weight: Float = 0
        for k in max(0, x - radius)...min(w - 1, x + radius) {
            let i = y * w + k; sum += luminance[i] * alpha[i]; weight += alpha[i]
        }
        horizontal[y * w + x] = weight > 0 ? sum / weight : 0
    } }
    for y in 0..<h { for x in 0..<w {
        var sum: Float = 0, n: Float = 0
        for k in max(0, y - radius)...min(h - 1, y + radius) { sum += horizontal[k * w + x]; n += 1 }
        mean[y * w + x] = sum / n
    } }

    var isEye = [Bool](repeating: false, count: count)
    for i in 0..<count where alpha[i] > 0.5 && luminance[i] - mean[i] > 0.10 { isEye[i] = true }

    var label = [Int](repeating: -1, count: count)
    var blobs: [[(Int, Int)]] = []
    for start in 0..<count where isEye[start] && label[start] < 0 {
        var blob: [(Int, Int)] = []; var stack = [start]; label[start] = blobs.count
        while let i = stack.popLast() {
            blob.append((i % w, i / w))
            let x = i % w, y = i / w
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            where nx >= 0 && nx < w && ny >= 0 && ny < h {
                let j = ny * w + nx
                if isEye[j] && label[j] < 0 { label[j] = blobs.count; stack.append(j) }
            }
        }
        blobs.append(blob)
    }
    blobs.sort { $0.count > $1.count }
    guard blobs.count >= 2, blobs[1].count >= 8,
          Double(blobs[0].count) / Double(blobs[1].count) < 3 else { return [] }

    return blobs.prefix(2).map { blob in
        let cx = Double(blob.map(\.0).reduce(0, +)) / Double(blob.count)
        let cy = Double(blob.map(\.1).reduce(0, +)) / Double(blob.count)
        // A blob's area gives its radius more stably than its bounding box, which one stray pixel
        // can stretch.
        let r = (Double(blob.count) / Double.pi).squareRoot()
        // Top-down, matching the bitmap this was measured in and SwiftUI's .position, which is
        // what consumes it. (The bbox pass above converts the other way for CoreImage, which is
        // bottom-left — the two coordinate systems live side by side in this file.)
        return (CGPoint(x: cx / Double(w), y: cy / Double(h)), CGFloat(r / Double(w)))
    }.sorted { $0.0.x < $1.0.x }
}

func place(_ image: CIImage, into path: String) {
    // Centred horizontally, sitting on the canvas floor.
    let dx = ((canvasWidth - image.extent.width) / 2).rounded()
    let positioned = image.transformed(by: .init(translationX: dx, y: 0))
    let canvas = CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight)
    // Composite onto a transparent canvas rather than cropping to it: cropped(to:) intersects, it
    // does not extend, so a shorter pose would keep its own height and the two PNGs would scale
    // differently under scaledToFit.
    let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: canvas)
    guard let data = context.pngRepresentation(of: positioned.composited(over: transparent),
                                               format: .RGBA8,
                                               colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
    else { die("cannot encode \(path)") }
    try! data.write(to: URL(fileURLWithPath: path))
    print(String(format: "wrote=%@ silhouette=%.0fx%.0f", path, image.extent.width, image.extent.height))
}
place(rest, into: arguments[3])
place(hopImage, into: arguments[4])
print(String(format: "canvas=%.0fx%.0f scale=%.3f", canvasWidth, canvasHeight, scale))

// Eye positions, for the app to glow. Normalised to the canvas and measured from the resting pose,
// which both frames share, so one set of coordinates serves both.
let restPlaced = rest.transformed(by: .init(translationX: ((canvasWidth - rest.extent.width) / 2).rounded(), y: 0))
    .composited(over: CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0))
        .cropped(to: CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight)))
let eyes = measureEyes(restPlaced)
if eyes.count == 2 {
    let json = "{\n  \"eyes\": [\n" + eyes.map {
        String(format: "    { \"x\": %.5f, \"y\": %.5f, \"r\": %.5f }", $0.0.x, $0.0.y, $0.1)
    }.joined(separator: ",\n") + "\n  ]\n}\n"
    let out = URL(fileURLWithPath: arguments[3]).deletingLastPathComponent()
        .appendingPathComponent("NetalEyes.json")
    try? json.write(to: out, atomically: true, encoding: .utf8)
    print(String(format: "eyes=(%.3f,%.3f) (%.3f,%.3f) r=%.4f", eyes[0].0.x, eyes[0].0.y,
                 eyes[1].0.x, eyes[1].0.y, eyes[0].1))
} else {
    print("eyes=not-found (\(eyes.count) blobs) — the glow will fall back to no overlay")
}

// A contact sheet of candidate scales, each with the two poses superimposed, so the right one can
// be picked by looking rather than argued about.
if let overlayIndex = arguments.firstIndex(of: "--overlay"), overlayIndex + 1 < arguments.count {
    let candidates: [CGFloat] = [1.00, 1.05, 1.10, 1.15, 1.20]
    var sheet = CIImage.empty()
    let cellWidth = rest.extent.width + 40
    for (column, candidate) in candidates.enumerated() {
        let scaled = hop.transformed(by: .init(scaleX: candidate, y: candidate))
        let restDX = CGFloat(column) * cellWidth + 20
        let hopDX = restDX + (rest.extent.width - scaled.extent.width) / 2
        // Landed solid, lifted at half strength on top: where the heads disagree is obvious.
        var cell = rest.transformed(by: .init(translationX: restDX, y: 20))
        let faded = CIFilter(name: "CIColorMatrix")!
        faded.setValue(scaled.transformed(by: .init(translationX: hopDX, y: 20)), forKey: kCIInputImageKey)
        faded.setValue(CIVector(x: 0, y: 0, z: 0, w: 0.55), forKey: "inputAVector")
        cell = faded.outputImage!.composited(over: cell)
        sheet = cell.composited(over: sheet)
    }
    let bounds = CGRect(x: 0, y: 0, width: cellWidth * CGFloat(candidates.count),
                        height: rest.extent.height + 40)
    let ground = CIImage(color: CIColor(red: 1, green: 0.35, blue: 0.3)).cropped(
        to: CGRect(x: 0, y: 19, width: bounds.width, height: 1))
    let backdrop = CIImage(color: CIColor(red: 0.30, green: 0.31, blue: 0.34)).cropped(to: bounds)
    let composed = sheet.composited(over: ground.composited(over: backdrop))
    if let data = context.pngRepresentation(of: composed.cropped(to: bounds), format: .RGBA8,
                                            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) {
        try? data.write(to: URL(fileURLWithPath: arguments[overlayIndex + 1]))
        print("overlay=\(arguments[overlayIndex + 1]) scales=\(candidates)")
    }
}
print("result=pass")
