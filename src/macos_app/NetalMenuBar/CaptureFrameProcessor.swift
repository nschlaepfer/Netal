import Foundation
import CoreVideo
import Metal
import MetalFX

struct CaptureProcessedFrames {
    /// A complete captured frame, optionally spatially upscaled with MetalFX.
    /// The texture owns its pixels and remains valid after the capture buffer is released.
    let source: MTLTexture
    let inputWidth: Int
    let inputHeight: Int
    var spatiallyUpscaled: Bool { source.width > inputWidth || source.height > inputHeight }
    /// A motion-compensated frame halfway between the previous and current captures.
    /// Nil during warmup, after a resize/reset, or when a scene cut makes interpolation unsafe.
    let midpoint: MTLTexture?
    let gpuMilliseconds: Double?
    let frameGenerationNote: String?
}

/// The public-API capture path has color frames, but no game motion vectors or depth.
/// SR therefore uses MetalFX *spatial* scaling; experimental FG estimates bidirectional
/// block motion from the captured pictures. It is not the game's native temporal SR/FG.
///
/// Call from the capture worker, never from the main thread. One command buffer completes
/// before this method returns, so ScreenCaptureKit can recycle its IOSurface safely.
final class CaptureFrameProcessor: @unchecked Sendable {
    enum Failure: LocalizedError {
        case unavailable(String)
        var errorDescription: String? {
            switch self { case .unavailable(let message): return message }
        }
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let textureCache: CVMetalTextureCache
    private let downsample: MTLComputePipelineState
    private let motion: MTLComputePipelineState
    private let interpolate: MTLComputePipelineState
    private let summarize: MTLComputePipelineState
    private let lock = NSLock()
    private var scaler: MTLFXSpatialScaler?
    private var previous: MTLTexture?
    private var previousLow: MTLTexture?
    private var previousInputSize = SIMD2<Int>(0, 0)

    init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw Failure.unavailable("This Mac could not create a Metal processing queue.")
        }
        self.queue = queue
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else {
            throw Failure.unavailable("This Mac could not connect screen capture to Metal.")
        }
        textureCache = cache
        let library = try device.makeLibrary(source: Self.shaders, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw Failure.unavailable("The capture processing shader is missing: \(name).")
            }
            return try device.makeComputePipelineState(function: function)
        }
        downsample = try pipeline("netal_capture_luma")
        motion = try pipeline("netal_capture_motion")
        interpolate = try pipeline("netal_capture_midpoint")
        summarize = try pipeline("netal_capture_motion_summary")
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        previous = nil
        previousLow = nil
        previousInputSize = .zero
        scaler = nil
        CVMetalTextureCacheFlush(textureCache, 0)
    }

    func process(pixelBuffer: CVPixelBuffer, outputWidth: Int, outputHeight: Int,
                 frameGeneration: Bool) throws -> CaptureProcessedFrames {
        lock.lock()
        var completed = false
        defer {
            // Any failed frame breaks temporal continuity. Do not interpolate across
            // an allocation/format/encoder failure when the stream resumes.
            if !completed {
                previous = nil
                previousLow = nil
                previousInputSize = .zero
                scaler?.colorTexture = nil
                scaler?.outputTexture = nil
            }
            lock.unlock()
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw Failure.unavailable("Capture requires SDR BGRA frames. HDR capture is not supported yet.")
        }
        guard width > 0, height > 0, outputWidth >= width, outputHeight >= height,
              outputWidth <= 8192, outputHeight <= 8192, outputWidth * outputHeight <= 33_554_432 else {
            throw Failure.unavailable("Choose an output at least as large as the captured game, up to 8K. Netal will not reduce the game's captured resolution.")
        }
        let inputSize = SIMD2(width, height)
        if inputSize != previousInputSize || previous?.width != outputWidth || previous?.height != outputHeight {
            previous = nil
            previousLow = nil
        }

        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
                                                               pixelBuffer, nil, .bgra8Unorm,
                                                               width, height, 0, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture,
              let captured = CVMetalTextureGetTexture(cvTexture) else {
            throw Failure.unavailable("Screen capture did not provide a Metal-compatible frame (\(status)).")
        }
        guard let commands = queue.makeCommandBuffer() else {
            throw Failure.unavailable("Metal could not prepare this frame.")
        }
        commands.label = "Netal captured-frame processing"

        let needsScaling = width != outputWidth || height != outputHeight
        if needsScaling {
            guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else {
                throw Failure.unavailable("MetalFX spatial upscaling is not supported on this Mac.")
            }
            if scaler?.inputWidth != width || scaler?.inputHeight != height ||
                scaler?.outputWidth != outputWidth || scaler?.outputHeight != outputHeight {
                let descriptor = MTLFXSpatialScalerDescriptor()
                descriptor.inputWidth = width
                descriptor.inputHeight = height
                descriptor.outputWidth = outputWidth
                descriptor.outputHeight = outputHeight
                descriptor.colorTextureFormat = .bgra8Unorm
                descriptor.outputTextureFormat = .bgra8Unorm
                descriptor.colorProcessingMode = .perceptual
                scaler = descriptor.makeSpatialScaler(device: device)
            }
            guard scaler != nil else {
                throw Failure.unavailable("MetalFX could not create an upscaler for this resolution.")
            }
        }

        // A private copy is required: holding a CVMetalTexture does not give the capture
        // stream permission to overwrite it while it is being displayed/interpolated.
        let inputUsage: MTLTextureUsage = needsScaling ? (scaler!.colorTextureUsage.union(.shaderRead)) : .shaderRead
        let input = try texture(width: width, height: height, format: .bgra8Unorm,
                                usage: inputUsage, label: "Netal owned capture")
        guard let copy = commands.makeBlitCommandEncoder() else {
            throw Failure.unavailable("Metal could not retain the captured frame.")
        }
        copy.copy(from: captured, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(x: 0, y: 0, z: 0),
                  sourceSize: .init(width: width, height: height, depth: 1),
                  to: input, destinationSlice: 0, destinationLevel: 0, destinationOrigin: .init(x: 0, y: 0, z: 0))
        copy.endEncoding()
        let output: MTLTexture
        if needsScaling, let scaler {
            output = try texture(width: outputWidth, height: outputHeight, format: .bgra8Unorm,
                                 usage: scaler.outputTextureUsage.union(.shaderRead), label: "Netal MetalFX output")
            scaler.inputContentWidth = width
            scaler.inputContentHeight = height
            scaler.colorTexture = input
            scaler.outputTexture = output
            scaler.encode(commandBuffer: commands)
        } else {
            // 1× is an exact copy. Do not shrink an already full-resolution capture just
            // to claim an SR result, and do not add an unnecessary sharpening pass.
            output = input
        }

        var midpoint: MTLTexture?
        var low: MTLTexture?
        var summary: MTLBuffer?
        var note: String?
        if frameGeneration {
            let scale = min(1.0, min(320.0 / Double(outputWidth), 180.0 / Double(outputHeight)))
            let lowWidth = max(1, Int((Double(outputWidth) * scale).rounded()))
            let lowHeight = max(1, Int((Double(outputHeight) * scale).rounded()))
            let currentLow = try texture(width: lowWidth, height: lowHeight, format: .r16Float,
                                         usage: [.shaderRead, .shaderWrite], label: "Netal motion luma")
            low = currentLow
            try encode(downsample, commands: commands, textures: [output, currentLow],
                       width: lowWidth, height: lowHeight)
            if let previous, let previousLow,
               previousLow.width == lowWidth, previousLow.height == lowHeight {
                let flowWidth = (lowWidth + 7) / 8, flowHeight = (lowHeight + 7) / 8
                let forward = try texture(width: flowWidth, height: flowHeight, format: .rgba16Float,
                                          usage: [.shaderRead, .shaderWrite], label: "Netal forward motion")
                let backward = try texture(width: flowWidth, height: flowHeight, format: .rgba16Float,
                                           usage: [.shaderRead, .shaderWrite], label: "Netal backward motion")
                try encode(motion, commands: commands, textures: [previousLow, currentLow, forward],
                           width: flowWidth, height: flowHeight)
                try encode(motion, commands: commands, textures: [currentLow, previousLow, backward],
                           width: flowWidth, height: flowHeight)
                let generated = try texture(width: outputWidth, height: outputHeight, format: .bgra8Unorm,
                                            usage: [.shaderRead, .shaderWrite], label: "Netal motion midpoint")
                guard let encoder = commands.makeComputeCommandEncoder(),
                      let stats = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
                    throw Failure.unavailable("Metal could not allocate frame-generation state.")
                }
                encoder.setComputePipelineState(interpolate)
                for (index, texture) in [previous, output, forward, backward, generated].enumerated() {
                    encoder.setTexture(texture, index: index)
                }
                var lowSize = SIMD2<Float>(Float(lowWidth), Float(lowHeight))
                encoder.setBytes(&lowSize, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
                dispatch(encoder, pipeline: interpolate, width: outputWidth, height: outputHeight)
                encoder.endEncoding()
                guard let statistics = commands.makeComputeCommandEncoder() else {
                    throw Failure.unavailable("Metal could not verify the generated frame.")
                }
                statistics.setComputePipelineState(summarize)
                statistics.setTexture(forward, index: 0)
                statistics.setBuffer(stats, offset: 0, index: 0)
                statistics.dispatchThreads(.init(width: 1, height: 1, depth: 1),
                                           threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
                statistics.endEncoding()
                summary = stats
                midpoint = generated
            } else {
                note = "Waiting for the next captured frame."
            }
        }

        commands.commit()
        // Explicitly keep the CV wrapper alive until all reads from its IOSurface finish.
        withExtendedLifetime(cvTexture) { commands.waitUntilCompleted() }
        guard commands.status == .completed else {
            previous = nil
            previousLow = nil
            throw Failure.unavailable(commands.error?.localizedDescription ?? "Metal could not finish processing this frame.")
        }
        if let summary, summary.contents().load(as: Float.self) > 0.16 {
            midpoint = nil
            note = "Scene changed; frame generation is warming up again."
        }
        previous = frameGeneration ? output : nil
        previousLow = frameGeneration ? low : nil
        previousInputSize = inputSize
        // Scaler resource properties retain textures even after the command buffer retires.
        scaler?.colorTexture = nil
        scaler?.outputTexture = nil
        let elapsed = commands.gpuEndTime - commands.gpuStartTime
        completed = true
        return CaptureProcessedFrames(source: output, inputWidth: width, inputHeight: height, midpoint: midpoint,
                                      gpuMilliseconds: elapsed > 0 ? elapsed * 1_000 : nil,
                                      frameGenerationNote: note)
    }

    private func texture(width: Int, height: Int, format: MTLPixelFormat,
                         usage: MTLTextureUsage, label: String) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width,
                                                                 height: height, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = usage
        guard let result = device.makeTexture(descriptor: descriptor) else {
            throw Failure.unavailable("There is not enough GPU memory for this capture resolution.")
        }
        result.label = label
        return result
    }

    private func encode(_ pipeline: MTLComputePipelineState, commands: MTLCommandBuffer,
                        textures: [MTLTexture], width: Int, height: Int) throws {
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw Failure.unavailable("Metal could not encode capture processing.")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, texture) in textures.enumerated() { encoder.setTexture(texture, index: index) }
        dispatch(encoder, pipeline: pipeline, width: width, height: height)
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState,
                          width: Int, height: Int) {
        let w = min(pipeline.threadExecutionWidth, 16)
        let h = min(8, max(1, pipeline.maxTotalThreadsPerThreadgroup / w))
        encoder.dispatchThreads(.init(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: .init(width: w, height: h, depth: 1))
    }

    private static let shaders = """
    #include <metal_stdlib>
    using namespace metal;
    constexpr sampler linearSample(coord::normalized, address::clamp_to_edge, filter::linear);

    kernel void netal_capture_luma(texture2d<float, access::sample> source [[texture(0)]],
                                   texture2d<float, access::write> target [[texture(1)]],
                                   uint2 xy [[thread_position_in_grid]]) {
        if (xy.x >= target.get_width() || xy.y >= target.get_height()) return;
        float2 uv = (float2(xy) + 0.5f) / float2(target.get_width(), target.get_height());
        // Four taps suppress aliasing before the bounded motion search.
        float2 step = 0.25f / float2(target.get_width(), target.get_height());
        float3 rgb = (source.sample(linearSample, uv + float2(-step.x, -step.y)).rgb
                    + source.sample(linearSample, uv + float2( step.x, -step.y)).rgb
                    + source.sample(linearSample, uv + float2(-step.x,  step.y)).rgb
                    + source.sample(linearSample, uv + float2( step.x,  step.y)).rgb) * 0.25f;
        target.write(float4(dot(rgb, float3(0.2126f, 0.7152f, 0.0722f))), xy);
    }

    kernel void netal_capture_motion(texture2d<float, access::read> a [[texture(0)]],
                                     texture2d<float, access::read> b [[texture(1)]],
                                     texture2d<float, access::write> flow [[texture(2)]],
                                     uint2 tile [[thread_position_in_grid]]) {
        if (tile.x >= flow.get_width() || tile.y >= flow.get_height()) return;
        int2 limit = int2(a.get_width(), a.get_height()) - 1;
        int2 origin = int2(tile) * 8;
        float best = 1e9f, residual = 1.0f;
        int2 vector = int2(0);
        // A bounded, genuine motion search, not frame duplication or an endpoint dissolve.
        // The zero-vector penalty resolves ambiguous flat regions without invented motion.
        for (int dy = -12; dy <= 12; ++dy) {
            for (int dx = -12; dx <= 12; ++dx) {
                float sad = 0;
                for (int y = 0; y < 8; y += 2) {
                    for (int x = 0; x < 8; x += 2) {
                        int2 pa = clamp(origin + int2(x, y), int2(0), limit);
                        int2 pb = clamp(pa + int2(dx, dy), int2(0), limit);
                        sad += abs(a.read(uint2(pa)).r - b.read(uint2(pb)).r);
                    }
                }
                sad *= (1.0f / 16.0f);
                float cost = sad + 0.00015f * float(abs(dx) + abs(dy));
                if (cost < best) { best = cost; residual = sad; vector = int2(dx, dy); }
            }
        }
        flow.write(float4(float2(vector), clamp(1.0f - residual * 6.0f, 0.0f, 1.0f), residual), tile);
    }

    // Motion textures have one sample per 8x8 low-resolution block. Account for the
    // partially filled last block; normalized texture coordinates alone shift the grid.
    float4 motionAt(texture2d<float, access::sample> flow, float2 uv, float2 lowSize) {
        float2 gridUV = (uv * lowSize) / (8.0f * float2(flow.get_width(), flow.get_height()));
        return flow.sample(linearSample, gridUV);
    }

    kernel void netal_capture_midpoint(texture2d<float, access::sample> previous [[texture(0)]],
                                       texture2d<float, access::sample> current [[texture(1)]],
                                       texture2d<float, access::sample> forward [[texture(2)]],
                                       texture2d<float, access::sample> backward [[texture(3)]],
                                       texture2d<float, access::write> target [[texture(4)]],
                                       constant float2 &lowSize [[buffer(0)]],
                                       uint2 xy [[thread_position_in_grid]]) {
        if (xy.x >= target.get_width() || xy.y >= target.get_height()) return;
        float2 uv = (float2(xy) + 0.5f) / float2(target.get_width(), target.get_height());
        // Thin stationary edges (including HUD strokes) can disappear in the low-
        // resolution flow field. Preserve an unchanged high-contrast stroke at its
        // actual location instead of warping it with the background's block motion.
        float4 unwarped = current.sample(linearSample, uv);
        float3 delta = abs(previous.sample(linearSample, uv).rgb - unwarped.rgb);
        if (max(delta.r, max(delta.g, delta.b)) < 0.008f) {
            float2 texel = 1.0f / float2(target.get_width(), target.get_height());
            bool stable[4];
            float contrast = 0;
            const float2 offsets[4] = {float2(-1, 0), float2(1, 0), float2(0, -1), float2(0, 1)};
            for (int n = 0; n < 4; ++n) {
                float2 p = uv + offsets[n] * texel;
                float3 before = previous.sample(linearSample, p).rgb;
                float3 after = current.sample(linearSample, p).rgb;
                float3 difference = abs(before - after);
                stable[n] = max(difference.r, max(difference.g, difference.b)) < 0.008f;
                contrast = max(contrast, length(after - unwarped.rgb));
            }
            if (contrast > 0.08f && ((stable[0] && stable[1]) || (stable[2] && stable[3]))) {
                target.write(float4(unwarped.rgb, 1), xy);
                return;
            }
        }
        float2 pa = uv, pb = uv;
        // Inverse warp to the halfway instant from both endpoints. Two refinements
        // follow the field toward the source coordinate instead of shifting in-place.
        for (int n = 0; n < 3; ++n) {
            pa = uv - 0.5f * motionAt(forward, pa, lowSize).xy / lowSize;
            pb = uv - 0.5f * motionAt(backward, pb, lowSize).xy / lowSize;
        }
        float4 fa = motionAt(forward, pa, lowSize);
        float4 fb = motionAt(backward, pb, lowSize);
        float4 a = previous.sample(linearSample, pa);
        float4 b = current.sample(linearSample, pb);
        // A disagreement marks an occlusion: favor the better supported warped sample.
        // Where both directions agree, combine their motion-aligned samples.
        float disagreement = length(fa.xy + fb.xy);
        float4 color;
        if (disagreement > 2.0f && abs(fa.z - fb.z) > 0.08f) color = fa.z > fb.z ? a : b;
        else {
            float wa = max(0.01f, fa.z), wb = max(0.01f, fb.z);
            color = (a * wa + b * wb) / (wa + wb);
        }
        color.a = 1;
        target.write(color, xy);
    }

    kernel void netal_capture_motion_summary(texture2d<float, access::read> flow [[texture(0)]],
                                             device float *meanResidual [[buffer(0)]],
                                             uint index [[thread_position_in_grid]]) {
        if (index != 0) return;
        float total = 0;
        for (uint y = 0; y < flow.get_height(); ++y)
            for (uint x = 0; x < flow.get_width(); ++x) total += flow.read(uint2(x, y)).w;
        meanResidual[0] = total / float(flow.get_width() * flow.get_height());
    }
    """
}
