import Foundation
import CoreVideo
import Metal
import MetalFX

// Standalone GPU integration test. No game capture or screen-recording permission needed.
// xcrun swiftc -target arm64-apple-macosx14.0 -O -framework Metal -framework MetalFX \
//   -framework CoreVideo CaptureFrameProcessor.swift Tests/Capture/main.swift \
//   -o /tmp/netal-capture-tests && /tmp/netal-capture-tests

struct TestError: Error, CustomStringConvertible {
    let description: String
}

var failures = 0
func check(_ condition: Bool, _ message: String) {
    print("\(condition ? "ok" : "FAIL") \(message)")
    if !condition { failures += 1 }
}

let width = 160
let height = 96
let rectangleY = 32
let rectangleSize = 32

// The rectangle's texture travels with it. Two-pixel cells survive downsampling
// while a deterministic, nonrepeating pattern gives block matching a unique minimum.
func scene(rectangleX: Int, rectangleY: Int = rectangleY,
           width: Int = width, height: Int = height,
           rectangleSize: Int = rectangleSize, cellSize: Int = 2) -> [UInt8] {
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let localX = x - rectangleX
            let localY = y - rectangleY
            let inside = (0..<rectangleSize).contains(localX) && (0..<rectangleSize).contains(localY)
            let cellX = localX / cellSize
            let cellY = localY / cellSize
            let pattern = inside ? (cellX * 67 + cellY * 43 + cellX * cellY * 19) % 120 : 0
            let value = inside ? UInt8(128 + pattern) : UInt8(16)
            let offset = (y * width + x) * 4
            pixels[offset] = value
            pixels[offset + 1] = value
            pixels[offset + 2] = value
            pixels[offset + 3] = 255
        }
    }
    return pixels
}

func pixelBuffer(_ pixels: [UInt8], width: Int = width, height: Int = height) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes: [CFString: Any] = [
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any]
    ]
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                   kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
    guard status == kCVReturnSuccess, let buffer else {
        throw TestError(description: "CVPixelBuffer allocation failed: \(status)")
    }
    let lockStatus = CVPixelBufferLockBaseAddress(buffer, [])
    guard lockStatus == kCVReturnSuccess, let destination = CVPixelBufferGetBaseAddress(buffer) else {
        throw TestError(description: "CVPixelBuffer lock failed: \(lockStatus)")
    }
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    pixels.withUnsafeBytes { source in
        for y in 0..<height {
            destination.advanced(by: y * rowBytes).copyMemory(
                from: source.baseAddress!.advanced(by: y * width * 4), byteCount: width * 4)
        }
    }
    return buffer
}

func readback(_ texture: MTLTexture, device: MTLDevice) throws -> [UInt8] {
    let supported: [MTLPixelFormat] = [.bgra8Unorm, .bgra8Unorm_srgb, .rgba8Unorm, .rgba8Unorm_srgb]
    guard supported.contains(texture.pixelFormat) else {
        throw TestError(description: "Unsupported readback format: \(texture.pixelFormat.rawValue)")
    }
    let rowBytes = ((texture.width * 4 + 255) / 256) * 256
    guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
          let buffer = device.makeBuffer(length: rowBytes * texture.height, options: .storageModeShared),
          let blit = command.makeBlitCommandEncoder() else {
        throw TestError(description: "Could not create texture readback resources")
    }
    blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
              to: buffer, destinationOffset: 0, destinationBytesPerRow: rowBytes,
              destinationBytesPerImage: rowBytes * texture.height)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw TestError(description: "Texture readback failed: \(String(describing: command.error))")
    }
    var pixels = [UInt8](repeating: 0, count: texture.width * texture.height * 4)
    pixels.withUnsafeMutableBytes { destination in
        for y in 0..<texture.height {
            destination.baseAddress!.advanced(by: y * texture.width * 4).copyMemory(
                from: buffer.contents().advanced(by: y * rowBytes), byteCount: texture.width * 4)
        }
    }
    return pixels
}

func meanError(_ a: [UInt8], _ b: [UInt8], region: (Range<Int>, Range<Int>)? = nil,
               inputWidth: Int = width) -> Double {
    let bounds = region ?? (0..<width, 0..<height)
    var error = 0.0
    var count = 0
    for y in bounds.1 {
        for x in bounds.0 {
            for channel in 0..<3 {
                let index = (y * inputWidth + x) * 4 + channel
                error += abs(Double(a[index]) - Double(b[index])) / 255.0
                count += 1
            }
        }
    }
    return error / Double(count)
}

func meanLuminance(_ pixels: [UInt8]) -> Double {
    var sum = 0.0
    for index in stride(from: 0, to: pixels.count, by: 4) { sum += Double(pixels[index]) }
    return sum / Double(pixels.count / 4) / 255.0
}

func checkTranslation(_ name: String, from start: (Int, Int), to end: (Int, Int),
                      processor: CaptureFrameProcessor, device: MTLDevice) throws {
    processor.reset()
    let previous = scene(rectangleX: start.0, rectangleY: start.1)
    let current = scene(rectangleX: end.0, rectangleY: end.1)
    let expected = ((start.0 + end.0) / 2, (start.1 + end.1) / 2)
    let ideal = scene(rectangleX: expected.0, rectangleY: expected.1)
    _ = try processor.process(pixelBuffer: try pixelBuffer(previous), outputWidth: width,
                              outputHeight: height, frameGeneration: true)
    let output = try processor.process(pixelBuffer: try pixelBuffer(current), outputWidth: width,
                                       outputHeight: height, frameGeneration: true)
    check(output.midpoint != nil, "\(name) produces a midpoint")
    guard let midpoint = output.midpoint else { return }
    let actual = try readback(midpoint, device: device)
    let average = zip(previous, current).map { UInt8((Int($0) + Int($1)) / 2) }
    let region = ((min(start.0, end.0) - 4)..<(max(start.0, end.0) + rectangleSize + 4),
                  (min(start.1, end.1) - 4)..<(max(start.1, end.1) + rectangleSize + 4))
    let error = meanError(actual, ideal, region: region)
    let averageError = meanError(average, ideal, region: region)
    let horizontal = start.0 != end.0
    let expectedPosition = horizontal ? expected.0 : expected.1
    let candidates = (expectedPosition - 8)...(expectedPosition + 8)
    let bestPosition = candidates.min {
        let a = scene(rectangleX: horizontal ? $0 : expected.0,
                      rectangleY: horizontal ? expected.1 : $0)
        let b = scene(rectangleX: horizontal ? $1 : expected.0,
                      rectangleY: horizontal ? expected.1 : $1)
        return meanError(actual, a, region: region) < meanError(actual, b, region: region)
    }!
    print(String(format: "%@ ideal_MAE=%.6f average_MAE=%.6f best_axis_position=%d expected=%d",
                 name, error, averageError, bestPosition, expectedPosition))
    check(abs(bestPosition - expectedPosition) <= 1, "\(name) localizes motion at the halfway position")
    check(error < averageError * 0.95, "\(name) improves on endpoint averaging")
}

func checkStationaryHUD(processor: CaptureFrameProcessor, device: MTLDevice) throws {
    let w = 640, h = 360
    func hud(_ x: Int, _ y: Int) -> Bool {
        (40..<220).contains(x) && (40..<43).contains(y)
            || (40..<43).contains(x) && (40..<100).contains(y)
    }
    func pan(_ offset: Int) -> [UInt8] {
        var image = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let cx = (x + offset) / 8, cy = y / 8
                let v: UInt8 = hud(x, y) ? 255 : UInt8(24 + (cx * 67 + cy * 43 + cx * cy * 19) % 160)
                for c in 0..<3 { image[(y * w + x) * 4 + c] = v }
            }
        }
        return image
    }
    processor.reset()
    _ = try processor.process(pixelBuffer: try pixelBuffer(pan(0), width: w, height: h),
                              outputWidth: w, outputHeight: h, frameGeneration: true)
    let result = try processor.process(pixelBuffer: try pixelBuffer(pan(8), width: w, height: h),
                                      outputWidth: w, outputHeight: h, frameGeneration: true)
    check(result.midpoint != nil, "camera pan with a fixed HUD produces a midpoint")
    guard let midpoint = result.midpoint else { return }
    let pixels = try readback(midpoint, device: device)
    var error = 0.0, count = 0.0
    for y in 0..<h {
        for x in 0..<w where hud(x, y) {
            error += (255.0 - Double(pixels[(y * w + x) * 4])) / 255.0
            count += 1
        }
    }
    print(String(format: "stationary_HUD_MAE=%.6f", error / count))
    check(error / count < 0.015, "stationary HUD strokes remain intact during camera motion")
}

func benchmark(inputWidth: Int, inputHeight: Int, processor: CaptureFrameProcessor) throws {
    processor.reset()
    var timings: [Double] = []
    let cellSize = inputHeight / 180
    for frame in 0..<8 {
        let pixels = scene(rectangleX: inputWidth / 3 + frame * cellSize * 2,
                           rectangleY: inputHeight / 3, width: inputWidth, height: inputHeight,
                           rectangleSize: cellSize * 32, cellSize: cellSize)
        let output = try processor.process(
            pixelBuffer: try pixelBuffer(pixels, width: inputWidth, height: inputHeight),
            outputWidth: 1920, outputHeight: 1080, frameGeneration: true)
        if frame >= 2, let timing = output.gpuMilliseconds { timings.append(timing) }
    }
    check(timings.count == 6 && timings.allSatisfy { $0.isFinite && $0 >= 0 },
          "\(inputWidth)×\(inputHeight) benchmark records six warm GPU samples")
    if timings.count == 6 {
        let sorted = timings.sorted()
        let median = (sorted[2] + sorted[3]) / 2
        print(String(format: "benchmark input=%dx%d output=1920x1080 FG=true warm_samples=%d GPU_ms_median=%.3f min=%.3f max=%.3f",
                     inputWidth, inputHeight, timings.count, median, sorted.first!, sorted.last!))
    }
}

func run() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw TestError(description: "This test requires a real Metal GPU")
    }
    print("device=\(device.name)")
    let processor = try CaptureFrameProcessor(device: device)
    let previousPixels = scene(rectangleX: 32)
    let currentPixels = scene(rectangleX: 40)
    let idealPixels = scene(rectangleX: 36)
    let previousBuffer = try pixelBuffer(previousPixels)
    let currentBuffer = try pixelBuffer(currentPixels)

    let first = try processor.process(pixelBuffer: previousBuffer, outputWidth: width,
                                      outputHeight: height, frameGeneration: true)
    check(first.midpoint == nil, "the first frame has no invented history")
    check(first.source.width == width && first.source.height == height, "native-size source output dimensions")
    check(!first.spatiallyUpscaled, "a native-size copy is not recorded as SR")
    let firstPixels = try readback(first.source, device: device)
    check(meanError(firstPixels, previousPixels) <= 1.0 / 255.0, "native-size source preserves captured pixels")

    let second = try processor.process(pixelBuffer: currentBuffer, outputWidth: width,
                                       outputHeight: height, frameGeneration: true)
    check(second.midpoint != nil, "the next frame produces a motion-interpolated midpoint")
    check(try readback(first.source, device: device) == firstPixels,
          "processing another frame does not overwrite retained source output")
    let secondPixels = try readback(second.source, device: device)
    check(meanError(secondPixels, currentPixels) <= 1.0 / 255.0, "second source contains the current capture")
    if let midpoint = second.midpoint {
        check(midpoint.width == width && midpoint.height == height, "midpoint dimensions match presentation")
        let midpointPixels = try readback(midpoint, device: device)
        let averagePixels = zip(previousPixels, currentPixels).map { UInt8((Int($0) + Int($1)) / 2) }
        let region = (28..<80, 28..<68)
        let interpolationError = meanError(midpointPixels, idealPixels, region: region)
        let averageError = meanError(averagePixels, idealPixels, region: region)
        let differenceFromAverage = meanError(midpointPixels, averagePixels, region: region)
        let bestPosition = (28...44).min {
            meanError(midpointPixels, scene(rectangleX: $0), region: region)
                < meanError(midpointPixels, scene(rectangleX: $1), region: region)
        }!
        print(String(format: "motion ideal_MAE=%.6f average_MAE=%.6f difference_from_average=%.6f best_rectangle_x=%d",
                     interpolationError, averageError, differenceFromAverage, bestPosition))
        check(abs(bestPosition - 36) <= 1, "midpoint localizes the translated rectangle at x36")
        check(interpolationError < averageError * 0.95,
              "motion interpolation improves moving-texture accuracy over endpoint averaging")
        check(differenceFromAverage > 1.0 / 255.0, "midpoint is not a simple endpoint crossfade")

        _ = try processor.process(pixelBuffer: try pixelBuffer(scene(rectangleX: 48)),
                                  outputWidth: width, outputHeight: height, frameGeneration: true)
        check(try readback(midpoint, device: device) == midpointPixels,
              "processing another frame does not overwrite retained midpoint output")
        check(try readback(second.source, device: device) == secondPixels,
              "retained second source remains stable after history advances")
    }
    if let gpuMilliseconds = second.gpuMilliseconds {
        print(String(format: "native_process_GPU_ms=%.3f", gpuMilliseconds))
        check(gpuMilliseconds.isFinite && gpuMilliseconds >= 0, "GPU timing is finite and nonnegative")
    }

    processor.reset()
    let afterReset = try processor.process(pixelBuffer: currentBuffer, outputWidth: width,
                                           outputHeight: height, frameGeneration: true)
    check(afterReset.midpoint == nil, "reset discards interpolation history")
    let disabled = try processor.process(pixelBuffer: previousBuffer, outputWidth: width,
                                         outputHeight: height, frameGeneration: false)
    check(disabled.midpoint == nil, "disabled frame generation produces source only")

    try checkTranslation("negative-X", from: (56, 32), to: (48, 32), processor: processor, device: device)
    try checkTranslation("positive-Y", from: (48, 24), to: (48, 32), processor: processor, device: device)
    try checkTranslation("negative-Y", from: (48, 40), to: (48, 32), processor: processor, device: device)
    try checkStationaryHUD(processor: processor, device: device)

    processor.reset()
    _ = try processor.process(pixelBuffer: currentBuffer, outputWidth: width,
                              outputHeight: height, frameGeneration: true)
    let staticFrame = try processor.process(pixelBuffer: currentBuffer, outputWidth: width,
                                           outputHeight: height, frameGeneration: true)
    check(staticFrame.midpoint != nil, "a stationary scene still produces a midpoint")
    if let midpoint = staticFrame.midpoint {
        let staticError = meanError(try readback(midpoint, device: device), currentPixels)
        print(String(format: "static_scene_MAE=%.6f", staticError))
        check(staticError <= 1.0 / 255.0, "a stationary scene remains stable without shimmer")
    }

    let oddWidth = 163, oddHeight = 99
    let oddPrevious = scene(rectangleX: 128, rectangleY: 63, width: oddWidth, height: oddHeight)
    let oddCurrent = scene(rectangleX: 128, rectangleY: 67, width: oddWidth, height: oddHeight)
    let oddIdeal = scene(rectangleX: 128, rectangleY: 65, width: oddWidth, height: oddHeight)
    let resized = try processor.process(
        pixelBuffer: try pixelBuffer(oddPrevious, width: oddWidth, height: oddHeight),
        outputWidth: oddWidth, outputHeight: oddHeight, frameGeneration: true)
    check(resized.midpoint == nil, "capture resize resets incompatible interpolation history")
    check(try readback(resized.source, device: device) == oddPrevious,
          "odd-sized capture preserves pixels through the last row and column")
    let oddSecond = try processor.process(
        pixelBuffer: try pixelBuffer(oddCurrent, width: oddWidth, height: oddHeight),
        outputWidth: oddWidth, outputHeight: oddHeight, frameGeneration: true)
    check(oddSecond.midpoint != nil, "odd-sized capture produces a midpoint")
    if let midpoint = oddSecond.midpoint {
        let actual = try readback(midpoint, device: device)
        let average = zip(oddPrevious, oddCurrent).map { UInt8((Int($0) + Int($1)) / 2) }
        let region = (124..<oddWidth, 59..<oddHeight)
        let oddError = meanError(actual, oddIdeal, region: region, inputWidth: oddWidth)
        let averageError = meanError(average, oddIdeal, region: region, inputWidth: oddWidth)
        print(String(format: "odd_edge_motion ideal_MAE=%.6f average_MAE=%.6f", oddError, averageError))
        check(oddError < averageError * 0.95, "partial flow blocks at frame edges improve moving-texture accuracy")
    }

    processor.reset()
    var black = [UInt8](repeating: 0, count: width * height * 4)
    for index in stride(from: 3, to: black.count, by: 4) { black[index] = 255 }
    _ = try processor.process(pixelBuffer: try pixelBuffer(black), outputWidth: width,
                              outputHeight: height, frameGeneration: true)
    let white = [UInt8](repeating: 255, count: width * height * 4)
    let cut = try processor.process(pixelBuffer: try pixelBuffer(white), outputWidth: width,
                                   outputHeight: height, frameGeneration: true)
    check(cut.midpoint == nil, "a black-to-white scene cut suppresses an invalid midpoint")
    check(try readback(cut.source, device: device) == white, "scene-cut suppression still delivers the current frame")

    processor.reset()
    let upscaled = try processor.process(pixelBuffer: currentBuffer, outputWidth: width * 2,
                                         outputHeight: height * 2, frameGeneration: false)
    check(upscaled.source.width == width * 2 && upscaled.source.height == height * 2,
          "spatial upscaling produces the requested output dimensions")
    check(upscaled.inputWidth == width && upscaled.inputHeight == height && upscaled.spatiallyUpscaled,
          "SR evidence retains the actual captured dimensions")
    check(upscaled.midpoint == nil, "spatial upscaling alone does not generate a midpoint")
    let upscaledPixels = try readback(upscaled.source, device: device)
    let nativeLuminance = meanLuminance(currentPixels)
    let scaledLuminance = meanLuminance(upscaledPixels)
    print(String(format: "upscale mean_luminance_native=%.6f mean_luminance_output=%.6f",
                 nativeLuminance, scaledLuminance))
    check(abs(nativeLuminance - scaledLuminance) < 0.035, "upscaled output preserves scene brightness")

    processor.reset()
    _ = try processor.process(pixelBuffer: previousBuffer, outputWidth: width * 2,
                              outputHeight: height * 2, frameGeneration: true)
    let combined = try processor.process(pixelBuffer: currentBuffer, outputWidth: width * 2,
                                         outputHeight: height * 2, frameGeneration: true)
    check(combined.source.width == width * 2 && combined.source.height == height * 2,
          "combined SR and FG retains the requested source output dimensions")
    check(combined.midpoint?.width == width * 2 && combined.midpoint?.height == height * 2,
          "combined SR and FG produces a midpoint at the requested output dimensions")

    processor.reset()
    _ = try processor.process(pixelBuffer: previousBuffer, outputWidth: width,
                              outputHeight: height, frameGeneration: true)
    for (targetWidth, targetHeight) in [(width - 1, height), (width, height - 1), (0, height)] {
        do {
            _ = try processor.process(pixelBuffer: currentBuffer, outputWidth: targetWidth,
                                      outputHeight: targetHeight, frameGeneration: false)
            check(false, "invalid output \(targetWidth)×\(targetHeight) must throw")
        } catch {
            check(true, "invalid output \(targetWidth)×\(targetHeight) rejected")
        }
        let recovered = try processor.process(pixelBuffer: currentBuffer, outputWidth: width,
                                              outputHeight: height, frameGeneration: true)
        check(recovered.midpoint == nil, "failed processing clears stale interpolation history")
    }
    try benchmark(inputWidth: 1920, inputHeight: 1080, processor: processor)
    try benchmark(inputWidth: 1280, inputHeight: 720, processor: processor)
}

do {
    try run()
} catch {
    check(false, "GPU test failed: \(error)")
}
print("capture GPU tests: \(failures == 0 ? "PASS" : "FAIL") failures=\(failures)")
exit(Int32(min(failures, 255)))
