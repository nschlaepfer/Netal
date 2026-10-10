import Foundation
import AVFoundation
import CoreGraphics
import CoreVideo

/// A compact, lossy inspection preview. Full-resolution saved PNGs and JSON are
/// authoritative for pixel measurements. The movie itself is not an FPS receipt.
final class ReplayVideo {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adapter: AVAssetWriterInputPixelBufferAdaptor
    private var lastTime = -Double.infinity
    private let width = 1600, height = 450

    init(url: URL) throws {
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 10_000_000]])
        input.expectsMediaDataInRealTime = false
        adapter = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
            sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferCGImageCompatibilityKey as String: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey as String: true])
        guard writer.canAdd(input) else { throw Failure(description: "Movie encoder unavailable") }
        writer.add(input)
        guard writer.startWriting() else { throw Failure(description: "Movie could not start: \(writer.error?.localizedDescription ?? "unknown")") }
        writer.startSession(atSourceTime: .zero)
    }

    func append(actual: [UInt8], ideal: [UInt8], imageWidth: Int, imageHeight: Int, time: Double) throws {
        guard time > lastTime, let pool = adapter.pixelBufferPool else { throw Failure(description: "Invalid movie timeline or pixel pool") }
        var pixelBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer) == kCVReturnSuccess,
              let pixelBuffer else { throw Failure(description: "Movie pixel allocation failed") }
        guard CVPixelBufferLockBaseAddress(pixelBuffer, []) == kCVReturnSuccess else { throw Failure(description: "Movie pixel lock failed") }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        let bitmap = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmap) else { throw Failure(description: "Movie drawing failed") }
        context.interpolationQuality = .high
        // A raw bitmap context and these CGImages already share the BGRA row order.
        for (index, pixels) in [actual, ideal].enumerated() {
            guard let provider = CGDataProvider(data: Data(pixels) as CFData),
                  let image = CGImage(width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                    bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: bitmap),
                    provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
            else { throw Failure(description: "Movie input image failed") }
            context.draw(image, in: CGRect(x: index * width / 2, y: 0, width: width / 2, height: height))
        }
        let deadline = Date().addingTimeInterval(10)
        while !input.isReadyForMoreMediaData && writer.status == .writing && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.001)
        }
        guard input.isReadyForMoreMediaData,
              adapter.append(pixelBuffer, withPresentationTime: CMTime(seconds: time, preferredTimescale: 60_000)) else {
            throw Failure(description: "Movie append failed: \(writer.error?.localizedDescription ?? "encoder timeout")")
        }
        lastTime = time
    }

    func finish(at seconds: Double) throws {
        writer.endSession(atSourceTime: CMTime(seconds: seconds, preferredTimescale: 60_000))
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        guard done.wait(timeout: .now() + 20) == .success, writer.status == .completed else {
            throw Failure(description: "Movie finalization failed: \(writer.error?.localizedDescription ?? "timeout")")
        }
    }
}
