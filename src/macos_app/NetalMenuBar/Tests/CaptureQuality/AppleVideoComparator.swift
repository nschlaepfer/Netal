import Foundation
import CoreVideo
import CoreMedia
import Metal
import VideoToolbox

protocol QualityFrameProcessor {
    func reset()
    func process(pixelBuffer: CVPixelBuffer, outputWidth: Int, outputHeight: Int,
                 frameGeneration: Bool) throws -> CaptureProcessedFrames
}
extension CaptureFrameProcessor: QualityFrameProcessor {}

/// Research-only adapter written against Apple's public API contracts. This is
/// not installed in Window mode and is not DLSS. FG runs at the original input
/// size; requested spatial scaling uses Netal's existing MetalFX spatial pass
/// on source and midpoint separately. It is not Apple's joint 2x VT mode.
@available(macOS 26.0, *)
final class AppleVideoComparator: QualityFrameProcessor {
    private let spatial: CaptureFrameProcessor
    private let transfer: VTPixelTransferSession
    private var processor: VTFrameProcessor?
    private var configuration: VTLowLatencyFrameInterpolationConfiguration?
    private var previous: VTFrameProcessorFrame?
    private var ordinal: Int64 = 0

    init(device: MTLDevice) throws {
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported else {
            throw Failure(description: "Apple low-latency frame interpolation is unsupported on this device")
        }
        spatial = try CaptureFrameProcessor(device: device)
        var session: VTPixelTransferSession?
        guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr,
              let session else { throw Failure(description: "Pixel conversion unavailable") }
        transfer = session
    }

    deinit { processor?.endSession(); VTPixelTransferSessionInvalidate(transfer) }

    func reset() {
        processor?.endSession(); processor = nil; configuration = nil
        previous = nil; ordinal = 0; spatial.reset()
    }

    func process(pixelBuffer: CVPixelBuffer, outputWidth: Int, outputHeight: Int,
                 frameGeneration: Bool) throws -> CaptureProcessedFrames {
        let source = try spatial.process(pixelBuffer: pixelBuffer, outputWidth: outputWidth,
                                         outputHeight: outputHeight, frameGeneration: false)
        guard frameGeneration else { previous = nil; return source }
        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        if configuration?.frameWidth != w || configuration?.frameHeight != h {
            reset()
            if #available(macOS 27.0, *) {
                if let dimension = VTLowLatencyFrameInterpolationConfiguration.maximumDimension(forSpatialScaleFactor: 1),
                   max(w, h) > dimension { throw Failure(description: "Input exceeds Apple's temporal interpolation dimension limit") }
                if let pixels = VTLowLatencyFrameInterpolationConfiguration.maximumPixelCount(forSpatialScaleFactor: 1),
                   w * h > pixels { throw Failure(description: "Input exceeds Apple's temporal interpolation pixel-count limit") }
            }
            guard let config = VTLowLatencyFrameInterpolationConfiguration(frameWidth: w, frameHeight: h,
                numberOfInterpolatedFrames: 1) else { throw Failure(description: "Apple rejected these interpolation dimensions") }
            let vt = VTFrameProcessor()
            try vt.startSession(configuration: config)
            processor = vt; configuration = config
        }
        guard let processor, let configuration else { throw Failure(description: "Apple session missing") }
        let currentBuffer = try Self.allocate(configuration.sourcePixelBufferAttributes)
        guard VTPixelTransferSessionTransferImage(transfer, from: pixelBuffer, to: currentBuffer) == noErr else {
            throw Failure(description: "Source conversion to Apple's supported format failed")
        }
        let time = CMTime(value: ordinal, timescale: 30)
        ordinal += 1
        guard let current = VTFrameProcessorFrame(buffer: currentBuffer, presentationTimeStamp: time) else {
            throw Failure(description: "Apple requires an IOSurface-backed source")
        }
        guard let previous else { self.previous = current; return source }
        guard CVPixelBufferGetWidth(previous.buffer) == w, CVPixelBufferGetHeight(previous.buffer) == h else {
            throw Failure(description: "Apple interpolation history dimensions changed")
        }
        let output = try Self.allocate(configuration.destinationPixelBufferAttributes)
        let midpointTime = CMTimeMultiplyByFloat64(CMTimeAdd(previous.presentationTimeStamp, time), multiplier: 0.5)
        guard let destination = VTFrameProcessorFrame(buffer: output, presentationTimeStamp: midpointTime),
              let parameters = VTLowLatencyFrameInterpolationParameters(sourceFrame: current, previousFrame: previous,
                interpolationPhase: [0.5], destinationFrames: [destination]) else {
            throw Failure(description: "Apple interpolation parameters were rejected")
        }
        let completion = Completion()
        processor.process(parameters: parameters) { _, error in completion.finish(error) }
        guard completion.done.wait(timeout: .now() + 30) == .success else {
            throw Failure(description: "Apple interpolation did not finish within 30 seconds")
        }
        if let error = completion.error { throw error }
        let rgba = try Self.allocate([kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        guard VTPixelTransferSessionTransferImage(transfer, from: output, to: rgba) == noErr else {
            throw Failure(description: "Apple output conversion to BGRA failed")
        }
        // The spatial processor makes immutable Metal-owned output, so neither
        // frame refers to a recycled CVPixelBuffer when the next pair is processed.
        let midpoint = try spatial.process(pixelBuffer: rgba, outputWidth: outputWidth,
                                           outputHeight: outputHeight, frameGeneration: false)
        self.previous = current
        return CaptureProcessedFrames(source: source.source, inputWidth: w, inputHeight: h,
            midpoint: midpoint.source, gpuMilliseconds: nil,
            frameGenerationNote: "Apple VideoToolbox comparator; no scene-cut guard or game integration")
    }

    private static func allocate(_ attributes: [String: Any]) throws -> CVPixelBuffer {
        let extra: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true,
                                     kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any]]
        var resolved: CFDictionary?
        guard CVPixelBufferCreateResolvedAttributesDictionary(kCFAllocatorDefault,
            [attributes as CFDictionary, extra as CFDictionary] as CFArray, &resolved) == kCVReturnSuccess,
              let resolved else { throw Failure(description: "Could not resolve Apple buffer requirements") }
        var chosen = resolved as! [String: Any]
        // Resolved format constraints may be an array even for a single format.
        let value = chosen[kCVPixelBufferPixelFormatTypeKey as String]
        let format = (value as? NSNumber)?.uint32Value ?? (value as? [NSNumber])?.first?.uint32Value
        guard let w = chosen[kCVPixelBufferWidthKey as String] as? Int,
              let h = chosen[kCVPixelBufferHeightKey as String] as? Int, let format else {
            throw Failure(description: "Incomplete Apple pixel-buffer requirements")
        }
        chosen[kCVPixelBufferPixelFormatTypeKey as String] = format
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, format, chosen as CFDictionary, &buffer) == kCVReturnSuccess,
              let buffer else { throw Failure(description: "Apple pixel-buffer allocation failed") }
        return buffer
    }

    private final class Completion: @unchecked Sendable {
        let done = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var storedError: Error?
        var error: Error? { lock.lock(); defer { lock.unlock() }; return storedError }
        func finish(_ error: Error?) { lock.lock(); storedError = error; lock.unlock(); done.signal() }
    }
}
