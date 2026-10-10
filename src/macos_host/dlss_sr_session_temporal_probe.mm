#import "dlss_sr_session.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>

@interface NeuralStudioDLSSSRSession (TemporalDiagnostics)
- (nullable NSData*)neuralStudioDiagnosticTemporalState:(NSUInteger)arrayIndex;
- (nullable NSData*)neuralStudioDiagnosticArena6AfterStage:(NSUInteger)stageIndex;
- (nullable NSData*)neuralStudioDiagnosticArena:(NSUInteger)arena
                                     afterStage:(NSUInteger)stageIndex;
- (nullable NSData*)neuralStudioDiagnosticMufuTrace:(NSUInteger)operation;
- (BOOL)neuralStudioDiagnosticLoopStage:(NSUInteger)stageIndex repeats:(NSUInteger)repeats toCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end

namespace {

constexpr NSUInteger kCanaryBytes = 256;
constexpr NSUInteger kDefaultFrameCount = 4;

// Same 18-phase jitter cycle as the RTX capture harness (TemporalJitter).
constexpr std::array<std::array<float, 2>, 18> kJitter = {{
    {-0.3125f, -0.351851851f}, {0.1875f, -0.0185185075f},
    {-0.0625f, 0.314814806f}, {0.4375f, -0.240740746f},
    {-0.46875f, 0.092592597f}, {0.03125f, 0.42592591f},
    {-0.21875f, -0.42592591f}, {0.0f, -0.166666657f},
    {-0.25f, 0.166666687f}, {0.25f, -0.388888896f},
    {-0.375f, -0.0555555522f}, {0.125f, 0.277777791f},
    {-0.125f, -0.277777791f}, {0.375f, 0.055555582f},
    {-0.4375f, 0.388888896f}, {0.0625f, -0.462962955f},
    {-0.1875f, -0.129629612f}, {0.3125f, 0.203703701f},
}};

NSUInteger FrameCount() {
  const char* value = std::getenv("NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT");
  if (value == nullptr || *value == '\0') return kDefaultFrameCount;
  const long parsed = std::strtol(value, nullptr, 10);
  return parsed >= 1 && parsed <= 64 ? static_cast<NSUInteger>(parsed) : kDefaultFrameCount;
}

// Frames that carry Reset=1; frame 0 always resets. Mirrors the harness's
// NEURALSTUDIO_SR_RESET_FRAME for mid-sequence reset oracles.
bool FrameResets(NSUInteger frame) {
  if (frame == 0) return true;
  const char* value = std::getenv("NEURALSTUDIO_SR_TEMPORAL_RESET_FRAMES");
  if (value == nullptr) return false;
  std::string list(value);
  std::size_t start = 0;
  while (start <= list.size()) {
    std::size_t end = list.find(',', start);
    if (end == std::string::npos) end = list.size();
    if (end > start && std::strtoul(list.substr(start, end - start).c_str(), nullptr, 10) == frame) {
      return true;
    }
    start = end + 1;
  }
  return false;
}

NSData* Load(NSURL* root, NSString* relative) {
  NSError* error = nil;
  NSData* data = [NSData dataWithContentsOfURL:[root URLByAppendingPathComponent:relative]
                                      options:NSDataReadingMappedIfSafe
                                        error:&error];
  if (data == nil) {
    std::cerr << "load_error=" << error.localizedDescription.UTF8String << '\n';
  }
  return data;
}

id<MTLBuffer> Buffer(id<MTLDevice> device, NSData* data) {
  if (data == nil) return nil;
  return [device newBufferWithBytes:data.bytes
                             length:data.length
                            options:MTLResourceStorageModeShared];
}

std::string Sha256(const void* bytes, NSUInteger length) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
  CC_SHA256(bytes, static_cast<CC_LONG>(length), digest);
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (NSUInteger index = 0; index < CC_SHA256_DIGEST_LENGTH; ++index) {
    result[index * 2] = digits[digest[index] >> 4];
    result[index * 2 + 1] = digits[digest[index] & 15];
  }
  return result;
}

bool CanaryIntact(const std::uint8_t* bytes, NSUInteger begin, NSUInteger end) {
  for (NSUInteger index = begin; index < end; ++index) {
    if (bytes[index] != 0xa5) return false;
  }
  return true;
}

std::uint8_t OutputPoison() {
  const char* value = std::getenv("NEURALSTUDIO_SR_OUTPUT_POISON");
  if (value == nullptr || *value == '\0') return 0x5a;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 0);
  if (end == value || *end != '\0' || parsed > 255) return 0xff;
  return static_cast<std::uint8_t>(parsed);
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 4) {
      std::cerr << "usage=dlss-sr-session-temporal-probe <runtime-bundle> "
                   "<temporal-capture-root> <receipt>\n";
      return 2;
    }
    const std::uint8_t poison = OutputPoison();
    if (poison == 0xff) {
      std::cerr << "error=invalid_output_poison\n";
      return 2;
    }
    NSURL* bundle = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    NSURL* capture = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];
    NSString* receipt_path = [NSString stringWithUTF8String:argv[3]];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return 3;
    NSError* error = nil;
    NeuralStudioDLSSSRSession* session =
        [[NeuralStudioDLSSSRSession alloc] initWithDevice:device
                                         runtimeBundleURL:bundle
                                                    error:&error];
    const NSUInteger kOutputBytes = session == nil ? 0 : session.outputWidth * session.outputHeight * 4 * 2;
    const float motion_scale_x = session == nil ? 1280.0f : static_cast<float>(session.inputWidth);
    const float motion_scale_y = session == nil ? 720.0f : static_cast<float>(session.inputHeight);
    if (session == nil) {
      std::cerr << "session_error="
                << (error == nil ? "unknown" : error.localizedDescription.UTF8String)
                << '\n';
      return 3;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    // NEURALSTUDIO_SR_GPU_TRACE=<path.gputrace>: capture frame 1 for Xcode's Metal
    // debugger / shader profiler (requires MTL_CAPTURE_ENABLED=1 in the environment).
    const char* gpu_trace = std::getenv("NEURALSTUDIO_SR_GPU_TRACE");
    MTLCaptureManager* capture_manager = gpu_trace != nullptr ? [MTLCaptureManager sharedCaptureManager] : nil;
    // NEURALSTUDIO_SR_METAL4=1: run the same frames through Metal 4 command
    // buffers (bit-identity check of the Metal 4 encode path).
    const bool metal4 = std::getenv("NEURALSTUDIO_SR_METAL4") != nullptr &&
                        [device supportsFamily:MTLGPUFamilyMetal4];
    id<MTL4CommandQueue> queue4 = metal4 ? [device newMTL4CommandQueue] : nil;
    id<MTL4CommandAllocator> allocator4 = metal4 ? [device newCommandAllocator] : nil;
    id<MTLSharedEvent> done4 = metal4 ? [device newSharedEvent] : nil;
    MTLResidencySetDescriptor* residency_descriptor = [[MTLResidencySetDescriptor alloc] init];
    residency_descriptor.initialCapacity = 64;
    id<MTLResidencySet> residency4 = metal4 ? [device newResidencySetWithDescriptor:residency_descriptor error:nil] : nil;
    std::cout << "metal4=" << (metal4 ? 1 : 0) << '\n';
    if (queue == nil) return 3;

    NSMutableString* receipt = [NSMutableString string];
    [receipt appendString:@"probe=dlss_sr_persistent_temporal_oracle\n"];
    [receipt appendFormat:@"process_id=%d\n", getpid()];
    [receipt appendFormat:@"metal_device_name=%@\n", device.name];
    [receipt appendFormat:@"runtime_bundle_root_sha256=%@\n", session.runtimeBundleRoot];
    [receipt appendFormat:@"output_poison=0x%02x\n", poison];
    [receipt appendString:@"feature_sessions=1\n"];
    [receipt appendString:@"command_queues=1\n"];
    const NSUInteger kFrameCount = FrameCount();
    [receipt appendFormat:@"frames=%lu\n", static_cast<unsigned long>(kFrameCount)];
    [receipt appendFormat:@"logical_network_stages=%lu\n",
                          static_cast<unsigned long>(kFrameCount * 14)];
    [receipt appendString:@"physical_compute_dispatches=unmeasured\n"];
    for (const char* key : {"NEURALSTUDIO_SR_PWIN_SUFFIX", "NEURALSTUDIO_SR_PWIN_PIPELINE",
                            "NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY", "NEURALSTUDIO_SR_DEPTH_OUTPUT_LIBRARY",
                            "NEURALSTUDIO_SR_EXPOSURE_LIBRARY"}) {
      const char* value = std::getenv(key);
      [receipt appendFormat:@"requested_%s=%s\n", key, value == nullptr ? "<unset>" : value];
    }
    [receipt appendString:@"expected_opened_after_each_gpu_completion=true\n"];
    const char* state_oracle_path =
        std::getenv("NEURALSTUDIO_SR_TEMPORAL_STATE_ORACLE");
    NSURL* state_oracle = state_oracle_path == nullptr
        ? nil
        : [NSURL fileURLWithPath:[NSString stringWithUTF8String:state_oracle_path]];
    const bool stage_diagnostics =
        std::getenv("NEURALSTUDIO_SR_TEMPORAL_STAGE_DIAGNOSTICS") != nullptr;
    const char* stage_dump_path =
        std::getenv("NEURALSTUDIO_SR_TEMPORAL_STAGE_DUMP_DIR");
    const char* scalar_debug_path =
        std::getenv("NEURALSTUDIO_SR_TEMPORAL_SCALAR_DEBUG_PATH");

    // NEURALSTUDIO_SR_LAYER_LOOP="<stage index>,<repeats>,<rounds>": after the frames,
    // re-encode one PWin stage repeatedly (GPU counter attribution per layer).
    const char* layer_loop = std::getenv("NEURALSTUDIO_SR_LAYER_LOOP");
    bool pass = true;
    bool execution_safe = true;
    NSUInteger oracle_frames = 0;
    for (NSUInteger frame = 0; frame < kFrameCount; ++frame) {
      NSString* prefix = [NSString stringWithFormat:@"frame-%03lu",
                                                   static_cast<unsigned long>(frame)];
      NSData* color_data = Load(capture,
          [prefix stringByAppendingString:@"-input-color.rgba16le"]);
      NSData* depth_data = Load(capture,
          [prefix stringByAppendingString:@"-input-depth.r32le"]);
      NSData* motion_data = Load(capture,
          [prefix stringByAppendingString:@"-input-motion.rg16le"]);
      NSData* exposure_data = Load(capture,
          [prefix stringByAppendingString:@"-input-exposure.r32le"]);
      id<MTLBuffer> color = Buffer(device, color_data);
      id<MTLBuffer> depth = Buffer(device, depth_data);
      id<MTLBuffer> motion = Buffer(device, motion_data);
      id<MTLBuffer> exposure = Buffer(device, exposure_data);
      id<MTLBuffer> output =
          [device newBufferWithLength:kOutputBytes + 2 * kCanaryBytes
                              options:MTLResourceStorageModeShared];
      if (color == nil || depth == nil || motion == nil || exposure == nil ||
          output == nil) {
        std::cerr << "error=frame_buffer_allocation frame=" << frame << '\n';
        return 4;
      }
      std::memset(output.contents, 0xa5, output.length);
      std::memset(static_cast<std::uint8_t*>(output.contents) + kCanaryBytes,
                  poison, kOutputBytes);
      // Live captures (bridge --dump) carry per-frame parameters; RTX captures use the
      // jitter table and the reset schedule.
      float frame_jitter_x = kJitter[frame % kJitter.size()][0];
      float frame_jitter_y = kJitter[frame % kJitter.size()][1];
      float frame_scale_x = motion_scale_x, frame_scale_y = motion_scale_y;
      BOOL frame_reset = FrameResets(frame);
      BOOL frame_depth_reversed = YES;
      if (NSData* params_data = Load(capture, [prefix stringByAppendingString:@"-params.json"])) {
        NSDictionary* params = [NSJSONSerialization JSONObjectWithData:params_data options:0 error:nil];
        if ([params isKindOfClass:NSDictionary.class]) {
          frame_jitter_x = [params[@"jitter_x"] floatValue];
          frame_jitter_y = [params[@"jitter_y"] floatValue];
          frame_scale_x = [params[@"motion_scale_x"] floatValue];
          frame_scale_y = [params[@"motion_scale_y"] floatValue];
          frame_reset = frame == 0 ? YES : [params[@"reset"] boolValue];
          frame_depth_reversed = [params[@"depth_reversed"] boolValue];
        }
      }
      [receipt appendFormat:@"frame_%lu_input_color_sha256=%s\n", static_cast<unsigned long>(frame), Sha256(color_data.bytes, color_data.length).c_str()];
      [receipt appendFormat:@"frame_%lu_input_depth_sha256=%s\n", static_cast<unsigned long>(frame), Sha256(depth_data.bytes, depth_data.length).c_str()];
      [receipt appendFormat:@"frame_%lu_input_motion_sha256=%s\n", static_cast<unsigned long>(frame), Sha256(motion_data.bytes, motion_data.length).c_str()];
      [receipt appendFormat:@"frame_%lu_input_exposure_sha256=%s\n", static_cast<unsigned long>(frame), Sha256(exposure_data.bytes, exposure_data.length).c_str()];
      [receipt appendFormat:@"frame_%lu_jitter=%.9g,%.9g\n", static_cast<unsigned long>(frame), frame_jitter_x, frame_jitter_y];
      [receipt appendFormat:@"frame_%lu_motion_scale=%.9g,%.9g\n", static_cast<unsigned long>(frame), frame_scale_x, frame_scale_y];
      [receipt appendFormat:@"frame_%lu_depth_reversed=%s\n", static_cast<unsigned long>(frame), frame_depth_reversed ? "true" : "false"];
      if (capture_manager != nil && frame == 1 && [capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        MTLCaptureDescriptor* capture = [[MTLCaptureDescriptor alloc] init];
        capture.captureObject = queue;
        capture.destination = MTLCaptureDestinationGPUTraceDocument;
        capture.outputURL = [NSURL fileURLWithPath:@(gpu_trace)];
        NSError* capture_error = nil;
        if (![capture_manager startCaptureWithDescriptor:capture error:&capture_error]) {
          std::cerr << "gpu_trace_error=" << (capture_error == nil ? "unknown" : capture_error.localizedDescription.UTF8String) << '\n';
        }
      }
      id<MTLCommandBuffer> command = metal4 ? nil : [queue commandBuffer];
      id<MTL4CommandBuffer> command4 = nil;
      if (metal4) {
        command4 = [device newCommandBuffer];
        [command4 beginCommandBufferWithAllocator:allocator4];
        for (id<MTLBuffer> resident in @[color, depth, motion, exposure, output]) [residency4 addAllocation:resident];
        [residency4 commit];
        [command4 useResidencySet:residency4];
      }
      error = nil;
      const BOOL encoded = metal4
          ? [session encodeFrameToMTL4CommandBuffer:command4
                                                   colorBuffer:color
                                                   colorOffset:0
                                                  motionBuffer:motion
                                                  motionOffset:0
                                                   depthBuffer:depth
                                                   depthOffset:0
                                                exposureBuffer:exposure
                                                exposureOffset:0
                                                  outputBuffer:output
                                                  outputOffset:kCanaryBytes
                                                 jitterOffsetX:frame_jitter_x
                                                 jitterOffsetY:frame_jitter_y
                                            motionVectorScaleX:frame_scale_x
                                            motionVectorScaleY:frame_scale_y
                                                         reset:frame_reset
                                                 depthReversed:frame_depth_reversed
                                                         error:&error]
          : [session encodeFrameToCommandBuffer:command
                                                   colorBuffer:color
                                                   colorOffset:0
                                                  motionBuffer:motion
                                                  motionOffset:0
                                                   depthBuffer:depth
                                                   depthOffset:0
                                                exposureBuffer:exposure
                                                exposureOffset:0
                                                  outputBuffer:output
                                                  outputOffset:kCanaryBytes
                                                 jitterOffsetX:frame_jitter_x
                                                 jitterOffsetY:frame_jitter_y
                                            motionVectorScaleX:frame_scale_x
                                            motionVectorScaleY:frame_scale_y
                                                         reset:frame_reset
                                                 depthReversed:frame_depth_reversed
                                                         error:&error];
      if (!encoded) {
        std::cerr << "encode_error frame=" << frame << " detail="
                  << (error == nil ? "unknown" : error.localizedDescription.UTF8String)
                  << '\n';
        return 5;
      }
      if (metal4) {
        [command4 endCommandBuffer];
        id<MTL4CommandBuffer> list[1] = {command4};
        [queue4 commit:list count:1];
        [queue4 signalEvent:done4 value:frame + 1];
        const auto wait_start = std::chrono::steady_clock::now();
        const BOOL signaled = [done4 waitUntilSignaledValue:frame + 1 timeoutMS:20000];
        std::cout << "metal4_frame=" << frame << " wait_ms="
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count()
                  << " signaled=" << (signaled ? 1 : 0) << " event_value=" << done4.signaledValue << '\n';
        if (!signaled) {
          std::cerr << "command_error frame=" << frame << " detail=metal4_timeout\n";
          return 6;
        }
        [allocator4 reset];
      } else {
        [command commit];
        [command waitUntilCompleted];
      }
      if (capture_manager != nil && frame == 1 && capture_manager.isCapturing) {
        [capture_manager stopCapture];
        std::cout << "gpu_trace=" << gpu_trace << '\n';
      }
      if (!metal4 && command.status != MTLCommandBufferStatusCompleted) {
        std::cerr << "command_error frame=" << frame << " detail="
                  << (command.error == nil ? "unknown"
                                           : command.error.localizedDescription.UTF8String)
                  << '\n';
        return 6;
      }

      // Oracle bytes are deliberately unavailable until the GPU frame finishes.
      NSData* expected = Load(capture,
          [prefix stringByAppendingString:@"-output.rgba16le"]);
      if (expected == nil) {
        // A successful encode without reference output cannot establish RTX parity.
        pass = false;
        const auto* actual = static_cast<const std::uint8_t*>(output.contents) + kCanaryBytes;
        const auto* guarded = static_cast<const std::uint8_t*>(output.contents);
        const bool canaries = CanaryIntact(guarded, 0, kCanaryBytes) &&
                              CanaryIntact(guarded, kCanaryBytes + kOutputBytes, output.length);
        execution_safe &= canaries;
        [receipt appendFormat:@"frame_%lu_output_sha256=%s\n", static_cast<unsigned long>(frame), Sha256(actual, kOutputBytes).c_str()];
        [receipt appendFormat:@"frame_%lu_canaries_intact=%s\n", static_cast<unsigned long>(frame), canaries ? "true" : "false"];
        [receipt appendFormat:@"frame_%lu_exact=unverified\n", static_cast<unsigned long>(frame)];
        // Live capture without an RTX oracle: dump only (output and, when stage
        // diagnostics are on, the per-stage arena snapshots).
        const char* output_dump = std::getenv("NEURALSTUDIO_SR_TEMPORAL_OUTPUT_DUMP_DIR");
        if (output_dump != nullptr && *output_dump != '\0') {
          const auto* actual = static_cast<const std::uint8_t*>(output.contents) + kCanaryBytes;
          NSData* image = [NSData dataWithBytes:actual length:kOutputBytes];
          [image writeToFile:[NSString stringWithFormat:@"%s/%@-output.rgba16le", output_dump, prefix] atomically:NO];
        }
        if (stage_dump_path != nullptr && *stage_dump_path != '\0') {
          for (NSUInteger stage = 0; stage < 12; ++stage) {
            NSData* state = [session neuralStudioDiagnosticArena6AfterStage:stage];
            if (state != nil) [state writeToFile:[NSString stringWithFormat:@"%s/%@-stage-%02lu.bin", stage_dump_path, prefix, (unsigned long)stage] atomically:NO];
            for (NSUInteger arena = 7; arena <= 8; ++arena) {
              NSData* side = [session neuralStudioDiagnosticArena:arena afterStage:stage];
              if (side != nil) [side writeToFile:[NSString stringWithFormat:@"%s/%@-stage-%02lu-arena%lu.bin", stage_dump_path, prefix, (unsigned long)stage, (unsigned long)arena] atomically:NO];
            }
          }
        }
        [receipt appendFormat:@"frame_%lu_oracle=none\n", static_cast<unsigned long>(frame)];
        continue;
      }
      if (expected.length != kOutputBytes) return 7;
      ++oracle_frames;
      const auto* actual =
          static_cast<const std::uint8_t*>(output.contents) + kCanaryBytes;
      const auto* oracle = static_cast<const std::uint8_t*>(expected.bytes);
      NSUInteger mismatches = 0;
      for (NSUInteger index = 0; index < kOutputBytes; ++index) {
        mismatches += actual[index] != oracle[index];
      }
      const auto* guarded = static_cast<const std::uint8_t*>(output.contents);
      const bool canaries =
          CanaryIntact(guarded, 0, kCanaryBytes) &&
          CanaryIntact(guarded, kCanaryBytes + kOutputBytes, output.length);
      execution_safe &= canaries;
      const std::string actual_sha = Sha256(actual, kOutputBytes);
      const char* output_dump = std::getenv("NEURALSTUDIO_SR_TEMPORAL_OUTPUT_DUMP_DIR");
      if (output_dump != nullptr && *output_dump != '\0') {
        NSData* image = [NSData dataWithBytes:actual length:kOutputBytes];
        NSString* path = [NSString stringWithFormat:@"%s/%@-output.rgba16le",
                                                    output_dump, prefix];
        if (![image writeToFile:path atomically:NO]) return 15;
        for (NSUInteger array_index = 1; array_index <= 6; ++array_index) {
          NSData* state = [session neuralStudioDiagnosticTemporalState:array_index];
          if (state == nil) continue;
          NSString* array_path = [NSString stringWithFormat:@"%s/%@-array-%04lu.bin",
              output_dump, prefix, static_cast<unsigned long>(array_index)];
          if (![state writeToFile:array_path atomically:NO]) return 15;
        }
      }
      const std::string expected_sha = Sha256(expected.bytes, expected.length);
      [receipt appendFormat:@"frame_%lu_expected_sha256=%s\n", static_cast<unsigned long>(frame), expected_sha.c_str()];
      const bool frame_pass = mismatches == 0 && canaries &&
                              actual_sha == expected_sha &&
                              session.encodedFrameCount == frame + 1;
      pass &= frame_pass;
      [receipt appendFormat:@"frame_%lu_reset=%s\n",
                            static_cast<unsigned long>(frame), FrameResets(frame) ? "true" : "false"];
      [receipt appendFormat:@"frame_%lu_mismatched_bytes=%lu\n",
                            static_cast<unsigned long>(frame),
                            static_cast<unsigned long>(mismatches)];
      [receipt appendFormat:@"frame_%lu_canaries_intact=%s\n",
                            static_cast<unsigned long>(frame), canaries ? "true" : "false"];
      [receipt appendFormat:@"frame_%lu_output_sha256=%s\n",
                            static_cast<unsigned long>(frame), actual_sha.c_str()];
      [receipt appendFormat:@"frame_%lu_exact=%s\n",
                            static_cast<unsigned long>(frame), frame_pass ? "true" : "false"];
      [receipt appendFormat:@"frame_%lu_gpu_ms=%.3f\n", static_cast<unsigned long>(frame),
                            (metal4 ? 0.0 : (command.GPUEndTime - command.GPUStartTime) * 1000.0)];
      if (NSData* timings = [session neuralStudioDiagnosticMufuTrace:7]) {
        const auto* stamps = static_cast<const std::uint64_t*>(timings.bytes);
        const NSUInteger pairs = timings.length / 16;
        std::uint64_t total = 0;
        for (NSUInteger index = 0; index < pairs; ++index) total += stamps[index * 2 + 1] - stamps[index * 2];
        const double frame_ms = (metal4 ? 0.0 : (command.GPUEndTime - command.GPUStartTime) * 1000.0);
        NSMutableString* row = [NSMutableString string];
        for (NSUInteger index = 0; index < pairs; ++index) {
          const std::uint64_t ticks = stamps[index * 2 + 1] - stamps[index * 2];
          [row appendFormat:@"%s%.2f", index == 0 ? "" : ",",
                            total == 0 ? 0.0 : frame_ms * static_cast<double>(ticks) / static_cast<double>(total)];
        }
        [receipt appendFormat:@"frame_%lu_dispatch_ms=%@\n", static_cast<unsigned long>(frame), row];
        // Counter sampling changes the result, so a receipt taken under it must never be mistaken
        // for a clean one.
        //
        // Measured on the M3 Max: with NEURALSTUDIO_SR_DISPATCH_TIMING_SAMPLES high enough to reach
        // encoder 95 of frame 0 - the 3.4 ms hiluma output pass - frames 1 onward change, while
        // frame 0's own output does not. Sampling that encoder perturbs state frame 0 leaves behind
        // as history without altering what it displays. It is deterministic, not a race: two timed
        // runs agree with each other exactly, and disagree with the untimed reference.
        //
        // The attachment itself is correct (both start and end indices set, capacity checked), so
        // this is a property of computeCommandEncoderWithDescriptor: on this driver rather than a
        // slip here. Until that is understood, the receipt says so out loud.
        if (frame == 0) {
          [receipt appendString:@"dispatch_timing=1\n"];
          [receipt appendString:@"output_hashes_comparable=0  # counter sampling perturbs history "
                                 "from the output pass onward; use these timings for shares, never "
                                 "for hashes\n"];
        }
      }
      if (NSData* stats = [session neuralStudioDiagnosticMufuTrace:8]) {
        const auto* counters = static_cast<const std::uint32_t*>(stats.bytes);
        for (NSUInteger stage = 0; stage < stats.length / 256; ++stage) {
          NSMutableString* row = [NSMutableString string];
          for (NSUInteger index = 0; index < 8; ++index) {
            [row appendFormat:@"%s%u", index == 0 ? "" : ",", counters[stage * 64 + index]];
          }
          [receipt appendFormat:@"frame_%lu_pwin_stats_%lu=%@\n", static_cast<unsigned long>(frame),
                                static_cast<unsigned long>(stage), row];
        }
      }
      const NSUInteger oracle_launch = 14 * (frame + 1);
      NSString* oracle_probe = [NSString stringWithFormat:@"array-launch-%04lu-array-0001.bin",
                                                          static_cast<unsigned long>(oracle_launch)];
      if (state_oracle != nil && Load(state_oracle, oracle_probe) != nil) {
        const NSUInteger launch = 14 * (frame + 1);
        for (NSUInteger array_index = 1; array_index <= 6; ++array_index) {
          NSData* actual_state =
              [session neuralStudioDiagnosticTemporalState:array_index];
          NSString* name = [NSString stringWithFormat:
              @"array-launch-%04lu-array-%04lu.bin",
              static_cast<unsigned long>(launch),
              static_cast<unsigned long>(array_index)];
          NSData* expected_state = Load(state_oracle, name);
          NSUInteger state_mismatches = 0;
          if (actual_state.length != expected_state.length) {
            state_mismatches = MAX(actual_state.length, expected_state.length);
          } else {
            const auto* actual_state_bytes =
                static_cast<const std::uint8_t*>(actual_state.bytes);
            const auto* expected_state_bytes =
                static_cast<const std::uint8_t*>(expected_state.bytes);
            for (NSUInteger index = 0; index < actual_state.length; ++index) {
              state_mismatches +=
                  actual_state_bytes[index] != expected_state_bytes[index];
            }
          }
          [receipt appendFormat:@"frame_%lu_array_%lu_mismatched_bytes=%lu\n",
                                static_cast<unsigned long>(frame),
                                static_cast<unsigned long>(array_index),
                                static_cast<unsigned long>(state_mismatches)];
        }
      }
      if (stage_diagnostics) {
        for (NSUInteger stage = 0; stage < 12; ++stage) {
          NSData* state = [session neuralStudioDiagnosticArena6AfterStage:stage];
          if (state == nil) return 10;
          const std::string state_sha = Sha256(state.bytes, state.length);
          [receipt appendFormat:@"frame_%lu_stage_%lu_arena6_sha256=%s\n",
                                static_cast<unsigned long>(frame),
                                static_cast<unsigned long>(stage),
                                state_sha.c_str()];
          if (stage_dump_path != nullptr && *stage_dump_path != '\0') {
            NSString* path = [NSString stringWithFormat:@"%s/frame-%03lu-stage-%02lu.bin",
                stage_dump_path, static_cast<unsigned long>(frame),
                static_cast<unsigned long>(stage)];
            if (![state writeToFile:path atomically:NO]) return 11;
          }
          for (NSUInteger arena = 7; arena <= 8; ++arena) {
            NSData* side = [session neuralStudioDiagnosticArena:arena afterStage:stage];
            if (side == nil) return 10;
            [receipt appendFormat:@"frame_%lu_stage_%lu_arena%lu_sha256=%s\n",
                                  static_cast<unsigned long>(frame),
                                  static_cast<unsigned long>(stage),
                                  static_cast<unsigned long>(arena),
                                  Sha256(side.bytes, side.length).c_str()];
            if (stage_dump_path != nullptr && *stage_dump_path != '\0') {
              NSString* path = [NSString stringWithFormat:@"%s/frame-%03lu-stage-%02lu-arena%lu.bin",
                  stage_dump_path, static_cast<unsigned long>(frame),
                  static_cast<unsigned long>(stage), static_cast<unsigned long>(arena)];
              if (![side writeToFile:path atomically:NO]) return 11;
            }
          }
        }
      }
      if (frame == 1 && scalar_debug_path != nullptr &&
          *scalar_debug_path != '\0') {
        NSData* trace = [session neuralStudioDiagnosticMufuTrace:3];
        if (trace == nil ||
            ![trace writeToFile:[NSString stringWithUTF8String:scalar_debug_path]
                     atomically:NO]) return 13;
        [receipt appendFormat:@"scalar_debug_sha256=%s\n",
                              Sha256(trace.bytes, trace.length).c_str()];
      }
      const char* trace_dump = std::getenv("NEURALSTUDIO_SR_TEMPORAL_TRACE_DUMP_PATH");
      const char* trace_frame_value = std::getenv("NEURALSTUDIO_SR_TEMPORAL_TRACE_DUMP_FRAME");
      const NSUInteger trace_frame = trace_frame_value == nullptr ? 1
          : static_cast<NSUInteger>(std::strtoul(trace_frame_value, nullptr, 10));
      if (frame == trace_frame && trace_dump != nullptr && *trace_dump != '\0') {
        NSData* trace = [session neuralStudioDiagnosticMufuTrace:6];
        if (trace == nil ||
            ![trace writeToFile:[NSString stringWithUTF8String:trace_dump] atomically:NO]) return 16;
        [receipt appendFormat:@"trace_sha256=%s\n", Sha256(trace.bytes, trace.length).c_str()];
      }
      const char* frame1_miss_path =
          std::getenv("NEURALSTUDIO_SR_TEMPORAL_MUFU_MISS_DIR");
      if (frame == 1 && frame1_miss_path != nullptr && *frame1_miss_path != '\0') {
        for (NSUInteger operation = 4; operation <= 5; ++operation) {
          NSData* misses = [session neuralStudioDiagnosticMufuTrace:operation];
          if (misses == nil) return 14;
          const auto* values = static_cast<const std::uint32_t*>(misses.bytes);
          NSUInteger count = 0;
          for (NSUInteger index = 0; index < misses.length / 4; ++index) {
            count += values[index] != 0;
          }
          [receipt appendFormat:@"frames_0_1_mufu_%s_miss_count=%lu\n",
                                operation == 4 ? "lg2" : "ex2",
                                static_cast<unsigned long>(count)];
        }
      }
    }
    [receipt appendFormat:@"encoded_frame_count=%lu\n",
                          static_cast<unsigned long>(session.encodedFrameCount)];
    const char* mufu_trace_path =
        std::getenv("NEURALSTUDIO_SR_TEMPORAL_MUFU_TRACE_DIR");
    if (mufu_trace_path != nullptr && *mufu_trace_path != '\0') {
      for (NSUInteger operation = 1; operation <= 2; ++operation) {
        NSData* trace = [session neuralStudioDiagnosticMufuTrace:operation];
        if (trace == nil) return 12;
        NSString* name = operation == 1 ? @"lg2-input-table.bin"
                                        : @"ex2-input-table.bin";
        NSString* path = [NSString stringWithFormat:@"%s/%@",
                                                    mufu_trace_path, name];
        if (![trace writeToFile:path atomically:NO]) return 12;
        [receipt appendFormat:@"mufu_%@_input_table_sha256=%s\n", name,
                              Sha256(trace.bytes, trace.length).c_str()];
      }
    }
    const char* mufu_miss_path =
        std::getenv("NEURALSTUDIO_SR_TEMPORAL_MUFU_MISS_DIR");
    if (mufu_miss_path != nullptr && *mufu_miss_path != '\0') {
      for (NSUInteger operation = 4; operation <= 5; ++operation) {
        NSData* misses = [session neuralStudioDiagnosticMufuTrace:operation];
        if (misses == nil) return 14;
        const auto* values = static_cast<const std::uint32_t*>(misses.bytes);
        NSUInteger count = 0;
        for (NSUInteger index = 0; index < misses.length / 4; ++index) {
          count += values[index] != 0;
        }
        NSString* name = operation == 4 ? @"lg2-miss-table.bin"
                                        : @"ex2-miss-table.bin";
        NSString* path = [NSString stringWithFormat:@"%s/%@",
                                                    mufu_miss_path, name];
        if (![misses writeToFile:path atomically:NO]) return 14;
        [receipt appendFormat:@"mufu_%@_count=%lu\n", name,
                              static_cast<unsigned long>(count)];
      }
    }
    if (layer_loop != nullptr && *layer_loop != '\0') {
    unsigned stage_index = 0, repeats = 1, rounds = 1;
    std::sscanf(layer_loop, "%u,%u,%u", &stage_index, &repeats, &rounds);
    for (unsigned round = 0; round < rounds; ++round) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (![session neuralStudioDiagnosticLoopStage:stage_index repeats:repeats toCommandBuffer:command]) break;
      [command commit];
      [command waitUntilCompleted];
      std::cout << "layer_loop stage=" << stage_index << " repeats=" << repeats << " round=" << round
                << " gpu_ms=" << (command.GPUEndTime - command.GPUStartTime) * 1000.0 << '\n';
    }
  }
  [receipt appendFormat:@"oracle_frames=%lu\n", static_cast<unsigned long>(oracle_frames)];
  [receipt appendFormat:@"output_canaries_all_intact=%s\n", execution_safe ? "true" : "false"];
  [receipt appendFormat:@"persistent_temporal_rtx_parity=%s\n",
                          pass && oracle_frames == kFrameCount ? "true" : "false"];
    [receipt appendString:@"in_game_sustained_inference_proven=false\n"];
    [receipt appendString:@"visual_quality_validated=false\n"];
    [receipt appendString:@"realtime_performance_validated=false\n"];
    [receipt appendFormat:@"result=%s\n", !execution_safe ? "fail" : (oracle_frames != kFrameCount ? "unverified" : (pass ? "pass" : "fail"))];
    if (![receipt writeToFile:receipt_path
                   atomically:YES
                     encoding:NSUTF8StringEncoding
                        error:&error]) {
      return 8;
    }
    std::cout << receipt.UTF8String;
    return pass ? 0 : 9;
  }
}
