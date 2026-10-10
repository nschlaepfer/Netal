// Isolated native Netal NR bridge. Reuses Netal's temporal-scaler interception seam.
// No FG hooks, SR replacement, game-buffer commit/wait, or accepted-default changes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>
#include "../nr_experimental/nr_experimental_api.h"
#include "../nr_experimental/nr_encoder_scope.h"
#include "netal_nr_aspect.h"
#include "netal_nr_frame_capture.h"
#include "netal_nr_motion_pair_capture.h"

#ifndef NETAL_NR_OBSERVE_ONLY
#define NETAL_NR_OBSERVE_ONLY 0
#endif
#ifndef NETAL_NR_DELAYED_ONCE
#define NETAL_NR_DELAYED_ONCE 0
#endif
#if NETAL_NR_DELAYED_ONCE && !NETAL_NR_OBSERVE_ONLY
#include "../nr_experimental/nr_experimental_reset_worker.h"
#include "netal_nr_delayed_once_state.h"
#endif
namespace {
using EncodeFn = void (*)(id, SEL, id<MTLCommandBuffer>);
using CreateFn = id (*)(id, SEL, id<MTLDevice>);
struct Hook { Class type; IMP original; };
std::mutex hook_mutex, log_mutex;
std::vector<Hook> hooks;
IMP original_create = nullptr;
FILE* logfile = nullptr;
std::atomic<uint64_t> creates{0}, calls{0};
char stream_key, stream_metadata_key;
bool once_mode = false;
bool delayed_once_mode = false;
bool motion_pair_mode = false;
std::unique_ptr<netal_nr_adapter::MotionPairCapture> motion_pair_capture;
bool delayed_phase_gates = false; // Optional manual compute/publication release; delayed diagnostic only.
std::string capture_root, arm_file;
enum class ColorMode { Raw, BoundedReinhardSrgb };
ColorMode color_mode = ColorMode::Raw;
bool ParseColorMode(const char* text, ColorMode& mode) {
  if (!text || !*text || std::strcmp(text,"raw")==0) { mode=ColorMode::Raw; return true; }
  if (std::strcmp(text,"bounded-reinhard-srgb")==0) { mode=ColorMode::BoundedReinhardSrgb; return true; }
  return false;
}
const char* ColorModeName(ColorMode mode) { return mode==ColorMode::Raw ? "raw" : "bounded-reinhard-srgb"; }

// Optional macOS27 metadata: observe absence explicitly instead of assuming a
// flag was false or invoking selectors an older implementation cannot handle.
long long OptionalUInt(id object, const char* name) {
  SEL sel=sel_registerName(name);
  if (![object respondsToSelector:sel]) return -1;
  const NSUInteger value=reinterpret_cast<NSUInteger (*)(id,SEL)>(objc_msgSend)(object,sel);
  // Never let an oversized unsigned offset alias the -1 "unsupported" sentinel.
  const auto maximum=std::numeric_limits<long long>::max();
  return value>static_cast<NSUInteger>(maximum) ? maximum : (long long)value;
}
int OptionalBool(id object, const char* name) {
  SEL sel=sel_registerName(name);
  if (![object respondsToSelector:sel]) return -1;
  return reinterpret_cast<BOOL (*)(id,SEL)>(objc_msgSend)(object,sel) ? 1 : 0;
}
bool WholeFrameProfile(id scaler, NSUInteger width, NSUInteger height) {
  return OptionalUInt(scaler,"outputOffsetX") <= 0 && OptionalUInt(scaler,"outputOffsetY") <= 0 &&
    OptionalUInt(scaler,"outputWidth") == (long long)width && OptionalUInt(scaler,"outputHeight") == (long long)height;
}

void Log(const char* format, ...) {
  std::lock_guard<std::mutex> lock(log_mutex);
  FILE* out = logfile ? logfile : stderr;
  std::fprintf(out, "netal_nr experimental=1 time=%.6f ", CFAbsoluteTimeGetCurrent());
  va_list args; va_start(args, format); std::vfprintf(out, format, args); va_end(args);
  std::fputc('\n', out); std::fflush(out);
}
#if NETAL_NR_DELAYED_ONCE && !NETAL_NR_OBSERVE_ONLY
void NRDelayedPresented(uint64_t command);
#endif
#include "netal_nr_present_observer.h"
bool TemporalClass(Class cls) {
  const SEL required[] = {@selector(colorTexture), @selector(outputTexture), @selector(motionTexture),
    @selector(depthTexture), @selector(exposureTexture), @selector(inputContentWidth), @selector(inputContentHeight),
    @selector(jitterOffsetX), @selector(jitterOffsetY), @selector(motionVectorScaleX), @selector(motionVectorScaleY),
    @selector(preExposure), @selector(reset), @selector(fence), @selector(outputWidth), @selector(outputHeight)};
  for (SEL s : required) if (!class_getInstanceMethod(cls, s)) return false;
  // Frame interpolators can share a private base with temporal scalers.
  if (class_getInstanceMethod(cls, NSSelectorFromString(@"previousColorTexture")) ||
      class_getInstanceMethod(cls, NSSelectorFromString(@"interpolatedTexture"))) return false;
  return class_getInstanceMethod(cls, @selector(encodeToCommandBuffer:)) != nullptr;
}
IMP Original(Class cls) {
  std::lock_guard<std::mutex> lock(hook_mutex);
  for (Class c = cls; c; c = class_getSuperclass(c))
    for (const auto& h : hooks) if (h.type == c) return h.original;
  return nullptr;
}
void Observe(id<MTLFXTemporalScaler> s, id<MTLCommandBuffer> cb, uint64_t frame, uint64_t stream) {
  if(!capture_root.empty() && !motion_pair_mode){
    @try{try{InstallNRPresentObserver(cb);}catch(...){Log("event=present_observer_unavailable");}}
    @catch(NSException* exception){(void)exception;Log("event=present_observer_unavailable");}
  }
  if (frame > 12 && frame % 120 != 0) return;
  const bool m3 = [(id)cb respondsToSelector:@selector(addCompletedHandler:)] &&
                  [(id)cb respondsToSelector:@selector(commandQueue)];
  Log("event=live_inputs frame=%llu stream=%llu metal3=%d content=%lux%lu color=%lux%lu color_format=%lu "
      "output=%lux%lu output_format=%lu motion=%lux%lu motion_format=%lu depth_format=%lu exposure_format=%lu "
      "motion_scale=%.9g,%.9g jitter=%.9g,%.9g pre_exposure=%.9g reset=%d fence=%d output_usage=%lu output_hazard=%lu output_storage=%lu",
      (unsigned long long)frame, (unsigned long long)stream, m3,
      (unsigned long)s.inputContentWidth, (unsigned long)s.inputContentHeight,
      (unsigned long)s.colorTexture.width, (unsigned long)s.colorTexture.height, (unsigned long)s.colorTexture.pixelFormat,
      (unsigned long)s.outputTexture.width, (unsigned long)s.outputTexture.height, (unsigned long)s.outputTexture.pixelFormat,
      (unsigned long)s.motionTexture.width, (unsigned long)s.motionTexture.height, (unsigned long)s.motionTexture.pixelFormat,
      (unsigned long)s.depthTexture.pixelFormat, (unsigned long)s.exposureTexture.pixelFormat,
      s.motionVectorScaleX, s.motionVectorScaleY, s.jitterOffsetX, s.jitterOffsetY, s.preExposure, s.reset, s.fence != nil,
      (unsigned long)s.outputTexture.usage, (unsigned long)s.outputTexture.hazardTrackingMode, (unsigned long)s.outputTexture.storageMode);
  NSDictionary* metadata=objc_getAssociatedObject(s,&stream_metadata_key);
  Log("event=live_layout frame=%llu stream=%llu output_extent=%lldx%lld color_offset=%lld,%lld motion_offset=%lld,%lld output_offset=%lld,%lld "
      "motion_at_output_resolution=%d motion_includes_jitter=%d caller_retains_resources=%d whole_frame_1080p=%d unknown_metadata_is_minus_one=1",
      (unsigned long long)frame,(unsigned long long)stream,
      OptionalUInt(s,"outputWidth"),OptionalUInt(s,"outputHeight"),
      OptionalUInt(s,"colorContentOffsetX"),OptionalUInt(s,"colorContentOffsetY"),
      OptionalUInt(s,"motionContentOffsetX"),OptionalUInt(s,"motionContentOffsetY"),
      OptionalUInt(s,"outputOffsetX"),OptionalUInt(s,"outputOffsetY"),
      metadata ? [metadata[@"output_motion"] intValue] : -1,
      metadata ? [metadata[@"jittered_motion"] intValue] : -1,OptionalBool(cb,"retainedReferences"),WholeFrameProfile(s,1920,1080));
  if (m3) [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
    Log("event=game_command_completed frame=%llu stream=%llu success=%d whole_command_gpu_ms=%.6f observation_only_record=1",
      (unsigned long long)frame, (unsigned long long)stream, done.status == MTLCommandBufferStatusCompleted,
      (done.GPUEndTime - done.GPUStartTime) * 1000.0);
  }];
}

#if !NETAL_NR_OBSERVE_ONLY
namespace nr = netal_nr_experimental;
std::mutex state_mutex;
nr::ContextPtr context;
#if NETAL_NR_DELAYED_ONCE
nr::ResetWorkerPtr reset_worker;
nr::Report reset_prepare_report;
#endif
id<MTLComputePipelineState> import_rg11 = nil, scan_finite = nil, publish_finite = nil;
id<MTLComputePipelineState> fit_input = nil, fit_publish = nil;
id<MTLComputePipelineState> color_forward = nil, color_inverse = nil;
enum class Ready { Cold, Preparing, Ready, Failed };
Ready ready = Ready::Cold;
std::atomic<bool> attempted{false};
// Completion telemetry is separate from scheduling; also used by the standalone
// adapter harness, which must not mistake encode success for a published frame.
std::atomic<uint64_t> nr_completed{0}, nr_published{0};
std::atomic<uint32_t> nr_rejection_bits{0};
std::atomic<uint64_t> captures_finished{0}, captures_succeeded{0};
uint64_t selected_stream = 0;

struct Lease {
  nr::ContextPtr context;
  nr::TicketPtr ticket;
  id<MTLTexture> staged_color = nil, game_output = nil, game_color = nil;
  id<MTLTexture> bounded_color = nil;
  id<MTLTexture> source_snapshot = nil, publication = nil, nr_output = nil;
  id<MTLBuffer> import_buffer = nil, export_buffer = nil, rejected = nil;
  id<MTLTexture> motion = nil, depth = nil, exposure = nil;
  id<MTLFence> fence = nil;
  uint64_t frame = 0, stream = 0;
  bool export_encoded = false;
  netal_nr_adapter::AspectLayout layout;
  nr::FrameEncoderState encoders;
  std::shared_ptr<netal_nr_adapter::FrameCapture> capture;
};
// Never let a backend exception escape the game's hook/completion callback.
void Invalidate(const nr::ContextPtr& ctx, const char* reason) noexcept {
  try { nr::invalidate(ctx, reason); }
  catch (...) { Log("event=nr_invalidate_failed reason=%s", reason); }
}
void StartPrepare(id<MTLDevice> device) {
  std::lock_guard<std::mutex> lock(state_mutex);
  if (ready != Ready::Cold) return;
  const char* root = std::getenv("NETAL_NR_RUNTIME_BUNDLE");
  if (!root || !*root) { ready = Ready::Failed; Log("event=nr_unavailable reason=runtime_bundle_missing"); return; }
  nr::PrepareOptions options; options.runtime_bundle = root;
  const char* math = std::getenv("NETAL_NR_ARITHMETIC");
  if (math && std::strcmp(math, "approximate") == 0) options.arithmetic = nr::Arithmetic::ApproximateNative;
  ready = Ready::Preparing;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    @autoreleasepool {
      nr::ContextPtr result; nr::Report report; bool ok = false;
#if NETAL_NR_DELAYED_ONCE
      nr::ResetWorkerPtr worker_result;
#endif
      id<MTLComputePipelineState> ip = nil, sp = nil, pp = nil, fp = nil, op = nil, cp = nil, dp = nil;
      try {
        const char* codec = std::getenv("NETAL_NR_FORMAT_LIBRARY");
        NSError* error = nil;
        id<MTLLibrary> lib = codec ? [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:codec]] error:&error] : nil;
        if (lib) {
          id<MTLFunction> inf = [lib newFunctionWithName:@"neuralstudio_sr_import_rg11b10f"];
          id<MTLFunction> scan = [lib newFunctionWithName:@"netal_nr_scan_finite"];
          id<MTLFunction> publish = [lib newFunctionWithName:@"netal_nr_publish_finite"];
          id<MTLFunction> fit = [lib newFunctionWithName:@"netal_nr_fit_input"];
          id<MTLFunction> unfit = [lib newFunctionWithName:@"netal_nr_fit_publish"];
          if (inf) ip = [device newComputePipelineStateWithFunction:inf error:&error];
          if (scan) sp = [device newComputePipelineStateWithFunction:scan error:&error];
          if (publish) pp = [device newComputePipelineStateWithFunction:publish error:&error];
          if (fit) fp = [device newComputePipelineStateWithFunction:fit error:&error];
          if (unfit) op = [device newComputePipelineStateWithFunction:unfit error:&error];
          if (color_mode==ColorMode::BoundedReinhardSrgb) {
            auto cf=[lib newFunctionWithName:@"netal_nr_color_to_bounded"];
            auto cd=[lib newFunctionWithName:@"netal_nr_color_from_bounded"];
            if(cf)cp=[device newComputePipelineStateWithFunction:cf error:&error];
            if(cd)dp=[device newComputePipelineStateWithFunction:cd error:&error];
          }
        }
        if (!ip || !sp || !pp || !fp || !op || (color_mode==ColorMode::BoundedReinhardSrgb && (!cp || !dp))) report.error = "Netal format/safety/aspect/color library unavailable";
        else {
#if NETAL_NR_DELAYED_ONCE
          if (delayed_once_mode) {
            nr::ResetOptions reset_options;
            reset_options.runtime_bundle=options.runtime_bundle;
            reset_options.arithmetic=nr::Arithmetic::OriginalKernels;
            // Candidate settings for the OFFLINE delayed-adapter gate. The
            // separate live launcher still requires reviewed evidence for the
            // exact source/archive. These are not OS watchdog thresholds.
            reset_options.max_units_per_chunk=1;
            reset_options.max_dispatches_per_chunk=512;
            reset_options.max_encoders_per_chunk=512;
            reset_options.body_emission_encoder_bound=4;
            reset_options.max_chunk_gpu_ms=500;
            Log("event=nr_delayed_worker_options units=1 encoder_ceiling=512 dispatch_ceiling=512 body_dispatch_count_known=0 body_emission_target=4 chunk_stop_budget_ms=500 watchdogs_unchanged=1 requires_offline_review=1");
            ok=nr::reset_worker_prepare(device,reset_options,worker_result,report);
          } else
#endif
          ok = nr::prepare(device, options, result, report);
        }
      }
      catch (const std::exception& e) { report.error = e.what(); }
      catch (...) { report.error = "unknown prepare exception"; }
      bool have_runtime=bool(result);
#if NETAL_NR_DELAYED_ONCE
      if(delayed_once_mode)have_runtime=bool(worker_result);
#endif
      { std::lock_guard<std::mutex> l(state_mutex); context = result; import_rg11 = ip; scan_finite = sp; publish_finite = pp; fit_input = fp; fit_publish = op; color_forward=cp; color_inverse=dp; ready = ok && have_runtime ? Ready::Ready : Ready::Failed;
#if NETAL_NR_DELAYED_ONCE
        reset_worker=worker_result;reset_prepare_report=report;
#endif
      }
      Log("event=nr_prepare success=%d model=%s weights=%s arithmetic=%s detail=%s", ok && have_runtime,
          report.model_identity.c_str(), report.weights_identity.c_str(), report.arithmetic_label.c_str(), report.error.c_str());
    }
  });
}
void AppendDiagnosticOnce(id<MTLFXTemporalScaler> s, id<MTLCommandBuffer> cb, uint64_t frame, uint64_t stream) {
  if (!once_mode || attempted.load()) return;
  if (![(id)cb respondsToSelector:@selector(addCompletedHandler:)] ||
      ![(id)cb respondsToSelector:@selector(commandQueue)] || cb.status >= MTLCommandBufferStatusCommitted) return;
  id<MTLTexture> output = s.outputTexture;
  // Initial color-adapted game profile is the observed Lies of P RGB-only output.
  // Do not invent alpha semantics for other game output formats.
  const bool bounded=color_mode==ColorMode::BoundedReinhardSrgb;
  if (bounded && output.pixelFormat!=MTLPixelFormatRG11B10Float) return;
  // Initial profile is a whole output frame, not a subrect inside an allocation.
  // Older APIs without output offsets had implicit zero origins; unknown modern
  // motion metadata is NOT similarly treated as validated temporal input.
  if (!output || !netal_nr_adapter::SupportedGameExtent(uint32_t(output.width),uint32_t(output.height)) ||
      !WholeFrameProfile(s,output.width,output.height)) return;
  netal_nr_adapter::AspectLayout layout;
  if (!netal_nr_adapter::MakeAspectLayout(uint32_t(output.width),uint32_t(output.height),layout)) return;
  const bool aspect=!layout.identity();
  const NSUInteger gw=output.width,gh=output.height;
  if (
      (output.pixelFormat != MTLPixelFormatRGBA16Float && output.pixelFormat != MTLPixelFormatRG11B10Float) ||
      output.sampleCount != 1 || output.textureType != MTLTextureType2D || output.framebufferOnly ||
      output.storageMode == MTLStorageModeMemoryless || !stream || !s.inputContentWidth || !s.inputContentHeight ||
      s.inputContentWidth >= output.width || s.inputContentHeight >= output.height) return;
  if (output.device != cb.device || (output.hazardTrackingMode == MTLHazardTrackingModeUntracked && s.fence == nil)) return;
  StartPrepare(cb.device);
  // A diagnostic may warm up during loading but wait for an explicit arm file
  // before selecting its gameplay stream. File CONTENTS are never interpreted.
  if (!arm_file.empty() && access(arm_file.c_str(),F_OK)!=0) return;
  nr::ContextPtr ctx;
  { std::lock_guard<std::mutex> lock(state_mutex);
    if (ready != Ready::Ready || (selected_stream && selected_stream != stream)) return;
    selected_stream = stream; ctx = context;
  }
  bool expected = false; if (!attempted.compare_exchange_strong(expected, true)) return;
  auto lease = std::make_shared<Lease>(); lease->context = ctx; lease->frame = frame; lease->stream = stream;
  lease->layout=layout;
  if (!capture_root.empty()) {
    lease->capture=netal_nr_adapter::MakeFrameCapture(cb.device,capture_root.c_str(),layout,output.pixelFormat,frame,stream);
    if (!lease->capture) { Invalidate(ctx,"capture allocation failed");Log("event=nr_fallback reason=capture_allocation");return; }
    lease->capture->command_id=NRObjectID(cb);lease->capture->game_texture_id=NRObjectID(output);
    lease->capture->color_mode=ColorModeName(color_mode);
    lease->capture->color_reconstruction=bounded ? "inverse-reinhard-srgb" : "identity";
    MarkNRCommand(cb,frame,stream);
  }
  lease->game_output = output; lease->game_color = s.colorTexture; lease->motion = s.motionTexture;
  lease->depth = s.depthTexture; lease->exposure = s.exposureTexture; lease->fence = s.fence;
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  d.hazardTrackingMode = MTLHazardTrackingModeTracked;
  lease->staged_color = [cb.device newTextureWithDescriptor:d];
  if(bounded)lease->bounded_color=[cb.device newTextureWithDescriptor:d];
  const bool packed = output.pixelFormat == MTLPixelFormatRG11B10Float;
  d.pixelFormat = output.pixelFormat; d.width=gw; d.height=gh;
  lease->publication = [cb.device newTextureWithDescriptor:d];
  const auto opts = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
  lease->export_buffer = [cb.device newBufferWithLength:1920 * 1080 * 8 options:opts];
  lease->rejected = [cb.device newBufferWithLength:sizeof(uint32_t)
    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
  if (packed || aspect) {
    lease->source_snapshot = [cb.device newTextureWithDescriptor:d];
  }
  if (packed && !aspect) {
    lease->import_buffer = [cb.device newBufferWithLength:1920 * 1080 * 8 options:opts];
  }
  if (!lease->staged_color || (bounded && !lease->bounded_color) || !lease->publication || !lease->export_buffer || !lease->rejected ||
      ((packed || aspect) && !lease->source_snapshot) || (packed && !aspect && !lease->import_buffer)) {
    Invalidate(ctx, "input allocation failed"); Log("event=nr_fallback reason=allocation"); return;
  }
  // Install retention before appending anything. Even failed/partial scratch work survives completion.
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
    nr::Report report;
    try { if (lease->ticket) nr::complete(lease->ticket, done, report); }
    catch (const std::exception& e) { report.error = e.what(); Invalidate(lease->context, "completion exception"); }
    catch (...) { report.error = "unknown completion exception"; Invalidate(lease->context, "completion exception"); }
    // Read only AFTER completion. No current-frame CPU decision or blocking wait.
    const bool valid = done.status == MTLCommandBufferStatusCompleted;
    const uint32_t reject_bits = valid && lease->export_encoded ? *(const uint32_t*)lease->rejected.contents : 0;
    const bool rejected = reject_bits != 0;
    nr_rejection_bits.fetch_or(reject_bits);
    const bool published = valid && lease->export_encoded && !rejected;
    if (!published) Invalidate(lease->context, "frame not successfully published; history unusable");
    Log("event=nr_command_completed frame=%llu stream=%llu success=%d export_encoded=%d rejection_bits=%u published=%d history_advanced=%d history_usable=%d "
        "whole_command_gpu_ms=%.6f nr_gpu_time_valid=%d nr_gpu_ms=%.6f detail=%s",
        (unsigned long long)lease->frame, (unsigned long long)lease->stream,
        valid, lease->export_encoded, reject_bits, published, report.history_advanced, published && report.history_advanced,
        (done.GPUEndTime-done.GPUStartTime)*1000.0, report.nr_gpu_time_valid, report.nr_gpu_ms, report.error.c_str());
    if (published) nr_published.fetch_add(1);
    if (lease->capture) {
      auto capture=lease->capture;
      const double ms=(done.GPUEndTime-done.GPUStartTime)*1000.0;
      dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY,0), ^{
        @autoreleasepool {
          NSString* error=nil; bool ok=false;
          @try { try { ok=netal_nr_adapter::WriteFrameCapture(capture,valid,published,ms,&error); }
            catch (...) { error=@"C++ capture writer exception"; } }
          @catch (NSException* exception) { error=exception.reason; }
          Log("event=nr_capture_finished frame=%llu stream=%llu success=%d detail=%s",
            (unsigned long long)capture->frame,(unsigned long long)capture->stream,ok,error?error.UTF8String:"");
          if(ok)captures_succeeded.fetch_add(1);
          captures_finished.fetch_add(1);
        }
      });
    }
    nr_completed.fetch_add(1);
  }];
  id<MTLBlitCommandEncoder> import = [cb blitCommandEncoder];
  nr::EncoderScope import_scope(lease->encoders,import);
  if (!import) { Invalidate(ctx, "import encoder unavailable"); return; }
  if (lease->fence) [import waitForFence:lease->fence];
  [import copyFromTexture:output sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
              sourceSize:MTLSizeMake(gw,gh,1) toTexture:((packed || aspect) ? lease->source_snapshot : lease->staged_color) destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [import copyFromTexture:output sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
              sourceSize:MTLSizeMake(gw,gh,1) toTexture:lease->publication destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [import fillBuffer:lease->rejected range:NSMakeRange(0,sizeof(uint32_t)) value:0];
  netal_nr_adapter::CopyCapture(import,lease->capture,0,output);
  if (!import_scope.close()) { Invalidate(ctx,"import close failed"); return; }
  const uint32_t dimensions[2] = {1920,1080};
  if (aspect) {
    id<MTLComputeCommandEncoder> c=[cb computeCommandEncoder];
    nr::EncoderScope scope(lease->encoders,c);
    if (!c) { Invalidate(ctx,"aspect import encoder unavailable"); return; }
    [c setComputePipelineState:fit_input]; [c setTexture:lease->source_snapshot atIndex:0];
    [c setTexture:lease->staged_color atIndex:1]; [c setBytes:&layout length:sizeof layout atIndex:0];
    [c dispatchThreads:MTLSizeMake(1920,1080,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    if (!scope.close()) { Invalidate(ctx,"aspect import close failed"); return; }
  } else if (packed) {
    id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
    nr::EncoderScope scope(lease->encoders,c);
    if (!c) { Invalidate(ctx,"format import encoder unavailable"); return; }
    [c setComputePipelineState:import_rg11]; [c setTexture:lease->source_snapshot atIndex:0];
    [c setBuffer:lease->import_buffer offset:0 atIndex:0]; [c setBytes:dimensions length:sizeof dimensions atIndex:1];
    [c dispatchThreads:MTLSizeMake(1920,1080,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    if (!scope.close()) { Invalidate(ctx,"format import close failed"); return; }
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    nr::EncoderScope blit_scope(lease->encoders,b);
    if (!b) { Invalidate(ctx,"format import blit unavailable"); return; }
    [b copyFromBuffer:lease->import_buffer sourceOffset:0 sourceBytesPerRow:1920*8 sourceBytesPerImage:1920*1080*8
           sourceSize:MTLSizeMake(1920,1080,1) toTexture:lease->staged_color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    if (!blit_scope.close()) { Invalidate(ctx,"format import blit close failed"); return; }
  }
  id<MTLTexture> model_color=lease->staged_color;
  if(bounded) {
    auto c=[cb computeCommandEncoder];nr::EncoderScope scope(lease->encoders,c);
    if(!c){Invalidate(ctx,"color input encoder unavailable");return;}
    [c setComputePipelineState:color_forward];[c setTexture:lease->staged_color atIndex:0];
    [c setTexture:lease->bounded_color atIndex:1];[c setBuffer:lease->rejected offset:0 atIndex:0];
    [c dispatchThreads:MTLSizeMake(1920,1080,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];
    if(!scope.close()){Invalidate(ctx,"color input close failed");return;}
    model_color=lease->bounded_color;
  }
  if (lease->capture) {
    auto b=[cb blitCommandEncoder];nr::EncoderScope scope(lease->encoders,b);
    if (!b) {Invalidate(ctx,"model-input capture encoder unavailable");return;}
    netal_nr_adapter::CopyCapture(b,lease->capture,1,model_color);
    if (!scope.close()) {Invalidate(ctx,"model-input capture close failed");return;}
  }
  nr::FrameInputs in; in.color = model_color; in.reset = true; in.frame_id = frame; in.stream_id = stream;
  nr::Report report; nr::EncodeStatus status = nr::EncodeStatus::RejectedBeforeEncoding;
  try { status = nr::encode(ctx, in, cb, lease->ticket, report); }
  catch (const std::exception& e) { report.error = e.what(); }
  catch (...) { report.error = "unknown encode exception"; }
  if (lease->capture) { lease->capture->model_identity=report.model_identity;lease->capture->arithmetic=report.arithmetic_label; }
  if (status != nr::EncodeStatus::Encoded || !lease->ticket || !report.model_encoded || !report.model_dispatches) {
    Invalidate(ctx, "model not encoded; original MetalFX retained");
    Log("event=nr_fallback reason=backend status=%d detail=%s", (int)status, report.error.c_str()); return;
  }
  id<MTLTexture> result = nil;
  try { result = nr::output_texture(lease->ticket); }
  catch (...) { Invalidate(ctx, "output accessor exception"); Log("event=nr_fallback reason=output_accessor"); return; }
  if (!result || result == output || result.device != cb.device || result.pixelFormat != MTLPixelFormatRGBA16Float ||
      result.width != 1920 || result.height != 1080 || result.sampleCount != 1 ||
      result.textureType != MTLTextureType2D || result.framebufferOnly || result.storageMode == MTLStorageModeMemoryless ||
      result.hazardTrackingMode == MTLHazardTrackingModeUntracked) {
    Invalidate(ctx, "output contract failed; original MetalFX retained"); Log("event=nr_fallback reason=output_contract"); return;
  }
  lease->nr_output = result;
  {
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    nr::EncoderScope blit_scope(lease->encoders,b);
    if (!b) { Invalidate(ctx,"format output blit unavailable"); return; }
    [b copyFromTexture:result sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1920,1080,1)
              toBuffer:lease->export_buffer destinationOffset:0 destinationBytesPerRow:1920*8 destinationBytesPerImage:1920*1080*8];
    netal_nr_adapter::CopyCapture(b,lease->capture,2,result);
    if (!blit_scope.close()) { Invalidate(ctx,"output read close failed"); return; }
    if(bounded) {
      auto c=[cb computeCommandEncoder];nr::EncoderScope scope(lease->encoders,c);
      if(!c){Invalidate(ctx,"color output encoder unavailable");return;}
      [c setComputePipelineState:color_inverse];[c setBuffer:lease->export_buffer offset:0 atIndex:0];
      [c setBuffer:lease->rejected offset:0 atIndex:1];[c setBytes:dimensions length:sizeof dimensions atIndex:2];
      [c dispatchThreads:MTLSizeMake(1920,1080,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];
      if(!scope.close()){Invalidate(ctx,"color output close failed");return;}
    }
    // Two separate encoders: publication must see the completed whole-frame scan.
    const uint32_t publication_dimensions[4] = {1920,1080,uint32_t(packed),0};
    const id<MTLComputePipelineState> passes[] = {scan_finite, aspect ? fit_publish : publish_finite};
    for (NSUInteger pass = 0; pass < 2; ++pass) {
      id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
      nr::EncoderScope scope(lease->encoders,c);
      if (!c) { Invalidate(ctx,"publication safety encoder unavailable"); return; }
      [c setComputePipelineState:passes[pass]]; [c setBuffer:lease->export_buffer offset:0 atIndex:0];
      [c setBuffer:lease->rejected offset:0 atIndex:1];
      if (pass==1 && aspect) [c setBytes:&layout length:sizeof layout atIndex:2];
      else [c setBytes:publication_dimensions length:sizeof publication_dimensions atIndex:2];
      if (pass == 1) [c setTexture:lease->publication atIndex:0];
      [c dispatchThreads:MTLSizeMake(pass==1 ? gw : 1920,pass==1 ? gh : 1080,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
      if (!scope.close()) { Invalidate(ctx,"publication pass close failed"); return; }
    }
  }
  id<MTLBlitCommandEncoder> export_blit = [cb blitCommandEncoder];
  nr::EncoderScope export_scope(lease->encoders,export_blit);
  if (!export_blit) { Invalidate(ctx, "export encoder unavailable"); return; }
  [export_blit copyFromTexture:lease->publication sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                   sourceSize:MTLSizeMake(gw,gh,1) toTexture:output destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  netal_nr_adapter::CopyCapture(export_blit,lease->capture,3,output);
  if (lease->fence) [export_blit updateFence:lease->fence];
  if (!export_scope.close()) { Invalidate(ctx,"publication blit close failed"); return; }
  lease->export_encoded = true;
  Log("event=nr_color_mapping profile=%s inverse_on_export_copy=%d inverse_does_not_modify_model_history=1 experimental=1",
      ColorModeName(color_mode),bounded);
  Log("event=nr_aspect_mapping game=%lux%lu model=1920x1080 content=%u,%u,%u,%u aspect_adapted=%d game_content_cropped=0 exact_parity_claim=0",
      (unsigned long)gw,(unsigned long)gh,layout.x,layout.y,layout.width,layout.height,aspect);
  Log("event=nr_model_scheduled frame=%llu stream=%llu dispatches=%u mode=reset_only_once arithmetic=%s "
      "encode_cpu_time_valid=%d encode_cpu_ms=%.6f presentation_not_yet_verified=1",
      (unsigned long long)frame, (unsigned long long)stream, report.model_dispatches, report.arithmetic_label.c_str(), report.encode_cpu_time_valid, report.encode_cpu_ms);
}
#if NETAL_NR_DELAYED_ONCE
#include "netal_nr_delayed_once_bridge.h"
#endif
#endif

void DirectEncode(id object, SEL selector, id<MTLCommandBuffer> cb) {
  @autoreleasepool {
    IMP original = Original(object_getClass(object));
    if (!original) { Log("event=hook_error reason=missing_original"); return; }
    uint64_t pair_frame=0;
    if(motion_pair_mode && motion_pair_capture && TemporalClass(object_getClass(object))) {
      pair_frame=++calls;
      // Read-only input snapshot BEFORE the scaler can recycle inputs. Fence
      // ordering and completion leases live in the bounded capture helper.
      @try { try {
        id<MTLFXTemporalScaler> s=object;
        NSDictionary* metadata=objc_getAssociatedObject(s,&stream_metadata_key);
        netal_nr_adapter::MotionPairInput input;
        input.frame=pair_frame;input.stream=[objc_getAssociatedObject(s,&stream_key) unsignedLongLongValue];
        input.command=NRObjectID(cb);input.color=s.colorTexture;input.motion=s.motionTexture;input.fence=s.fence;
        input.metadata=@{
          @"input_width":@(s.inputContentWidth),@"input_height":@(s.inputContentHeight),
          @"output_width":@(OptionalUInt(s,"outputWidth")),@"output_height":@(OptionalUInt(s,"outputHeight")),
          @"color_offset_x":@(OptionalUInt(s,"colorContentOffsetX")),@"color_offset_y":@(OptionalUInt(s,"colorContentOffsetY")),
          @"motion_offset_x":@(OptionalUInt(s,"motionContentOffsetX")),@"motion_offset_y":@(OptionalUInt(s,"motionContentOffsetY")),
          @"output_offset_x":@(OptionalUInt(s,"outputOffsetX")),@"output_offset_y":@(OptionalUInt(s,"outputOffsetY")),
          @"motion_at_output_resolution":metadata[@"output_motion"]?:@(-1),
          @"motion_includes_jitter":metadata[@"jittered_motion"]?:@(-1),
          @"motion_scale_x":@(s.motionVectorScaleX),@"motion_scale_y":@(s.motionVectorScaleY),
          @"jitter_x":@(s.jitterOffsetX),@"jitter_y":@(s.jitterOffsetY),
          @"pre_exposure":@(s.preExposure),@"reset":@(s.reset)};
        if(motion_pair_capture->encode(cb,input))Log("event=motion_pair_copy_encoded frame=%llu stream=%llu actual_nr=0 completion_pending=1",
          (unsigned long long)input.frame,(unsigned long long)input.stream);
      } catch(...) {motion_pair_capture->stop();Log("event=motion_pair_stopped reason=cpp_exception");} }
      @catch(NSException* e){(void)e;motion_pair_capture->stop();Log("event=motion_pair_stopped reason=objc_exception");}
    }
    // Original scaler is ALWAYS upstream/fallback. Never substitute MetalFX for NR claims.
    reinterpret_cast<EncodeFn>(original)(object, selector, cb);
    if (!TemporalClass(object_getClass(object))) return;
    const uint64_t frame = pair_frame ? pair_frame : ++calls;
    NSNumber* ident = objc_getAssociatedObject(object, &stream_key); const uint64_t stream = ident.unsignedLongLongValue;
    Observe((id<MTLFXTemporalScaler>)object, cb, frame, stream);
#if !NETAL_NR_OBSERVE_ONLY
    try {
#if NETAL_NR_DELAYED_ONCE
      if(delayed_once_mode)AppendDelayedOnce((id<MTLFXTemporalScaler>)object,cb,frame,stream);
      else
#endif
      AppendDiagnosticOnce((id<MTLFXTemporalScaler>)object, cb, frame, stream);
    }
    catch (...) {
      // Disarm this one-shot experiment. Previously registered completion leases
      // still own any resources already referenced by the command buffer.
      attempted.store(true);
#if NETAL_NR_DELAYED_ONCE
      if(delayed_once_mode)if(auto job=DelayedJob())DelayedFail(job,"game-side bridge exception; no delayed publication");
#endif
      Log("event=nr_fallback reason=unexpected_bridge_exception");
    }
#endif
  }
}
bool InstallEncode(Class cls) {
  if (!TemporalClass(cls)) return false;
  SEL sel = @selector(encodeToCommandBuffer:); Method m = class_getInstanceMethod(cls, sel);
  std::lock_guard<std::mutex> lock(hook_mutex);
  for (const auto& h : hooks) if (h.type == cls) return true;
  IMP original = method_getImplementation(m);
  if (original == reinterpret_cast<IMP>(&DirectEncode)) {
    original = nullptr;
    for (Class p = class_getSuperclass(cls); p && !original; p = class_getSuperclass(p))
      for (const auto& h : hooks) if (h.type == p) { original = h.original; break; }
    if (!original) return false;
  }
  if (!class_addMethod(cls, sel, reinterpret_cast<IMP>(&DirectEncode), method_getTypeEncoding(m)))
    method_setImplementation(class_getInstanceMethod(cls, sel), reinterpret_cast<IMP>(&DirectEncode));
  hooks.push_back({cls, original}); Log("event=temporal_hook class=%s", class_getName(cls)); return true;
}
id NewScaler(id, SEL, id<MTLDevice>) NS_RETURNS_RETAINED;
id NewScaler(id descriptor, SEL sel, id<MTLDevice> device) {
  id target = reinterpret_cast<CreateFn>(original_create)(descriptor, sel, device);
  if (!target) return target;
  const uint64_t stream = ++creates; objc_setAssociatedObject(target, &stream_key, @(stream), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  objc_setAssociatedObject(target,&stream_metadata_key,@{
    @"output_motion":@(OptionalBool(descriptor,"isOutputResolutionMotionVectorsEnabled")),
    @"jittered_motion":@(OptionalBool(descriptor,"isJitteredMotionVectorsEnabled"))
  },OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  const bool hooked = InstallEncode(object_getClass(target));
  MTLFXTemporalScalerDescriptor* d = (MTLFXTemporalScalerDescriptor*)descriptor;
  Log("event=scaler_created stream=%llu class=%s hook=%d input=%lux%lu output=%lux%lu color_format=%lu output_format=%lu",
    (unsigned long long)stream, class_getName(object_getClass(target)), hooked,
    (unsigned long)d.inputWidth, (unsigned long)d.inputHeight, (unsigned long)d.outputWidth, (unsigned long)d.outputHeight,
    (unsigned long)d.colorTextureFormat, (unsigned long)d.outputTextureFormat);
  return target;
}
__attribute__((constructor)) void Initialize() {
  @autoreleasepool {
    const char* enabled = std::getenv("NETAL_NR_EXPERIMENTAL");
    if (!enabled || std::strcmp(enabled,"1") || ![NSBundle.mainBundle.bundleIdentifier isEqualToString:@"com.neowiz.game.lop"]) return;
    const char* path = std::getenv("NETAL_NR_EXPERIMENTAL_LOG"); if (path && *path) logfile = std::fopen(path,"a");
    const char* mode = std::getenv("NETAL_NR_MODE"); once_mode = mode && std::strcmp(mode,"once")==0;
    delayed_once_mode=mode && std::strcmp(mode,"delayed-once")==0;
    motion_pair_mode=mode && std::strcmp(mode,"motion-pair")==0;
#if !NETAL_NR_OBSERVE_ONLY
    if(motion_pair_mode){Log("event=bridge_declined reason=motion_pair_requires_model_free_observer");return;}
#endif
    const char* gates=std::getenv("NETAL_NR_DELAYED_PHASE_GATES");
    if(gates && std::strcmp(gates,"0")!=0 && std::strcmp(gates,"1")!=0){Log("event=bridge_declined reason=invalid_delayed_phase_gates");return;}
    delayed_phase_gates=gates && std::strcmp(gates,"1")==0;
    if(delayed_phase_gates && !delayed_once_mode){Log("event=bridge_declined reason=phase_gates_require_delayed_mode");return;}
#if !NETAL_NR_DELAYED_ONCE
    if(delayed_once_mode){Log("event=bridge_declined reason=delayed_worker_not_linked");return;}
#endif
    if(!ParseColorMode(std::getenv("NETAL_NR_COLOR_MODE"),color_mode)) {
      Log("event=bridge_declined reason=unknown_color_profile");return;
    }
    if(delayed_once_mode && color_mode!=ColorMode::BoundedReinhardSrgb){Log("event=bridge_declined reason=delayed_requires_bounded_color");return;}
    const char* captures=std::getenv("NETAL_NR_CAPTURE_DIR"); if(captures && (once_mode || delayed_once_mode || motion_pair_mode)) capture_root=captures;
    const char* arm=std::getenv("NETAL_NR_ARM_FILE"); if(arm && (once_mode || delayed_once_mode || motion_pair_mode)) arm_file=arm;
    if(motion_pair_mode){
      if(capture_root.empty() || capture_root[0]!='/' || arm_file.empty() || arm_file[0]!='/'){
        Log("event=bridge_declined reason=motion_pair_requires_absolute_capture_and_arm_paths");return;}
      motion_pair_capture=std::make_unique<netal_nr_adapter::MotionPairCapture>(capture_root,arm_file);
    }
    if(delayed_once_mode && (capture_root.empty() || arm_file.empty())){Log("event=bridge_declined reason=delayed_requires_capture_and_manual_arm");return;}
    Method m = class_getInstanceMethod([MTLFXTemporalScalerDescriptor class], @selector(newTemporalScalerWithDevice:));
    if (!m) { Log("event=bridge_declined reason=temporal_factory_missing"); return; }
    original_create = method_setImplementation(m,reinterpret_cast<IMP>(&NewScaler));
    Log("event=bridge_ready target=com.neowiz.game.lop api=metal3 model_linked=%d mode=%s color_profile=%s accepted_default_changed=0",
      !NETAL_NR_OBSERVE_ONLY, motion_pair_mode ? "motion_pair_capture_only" : (NETAL_NR_OBSERVE_ONLY ? "observe_only" : (delayed_once_mode ? "delayed_reset_only_once" : (once_mode ? "reset_only_once" : "observe_only"))),ColorModeName(color_mode));
  }
}
} // namespace
