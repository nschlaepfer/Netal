#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>

#import "dlss_sr_texture_adapter.h"
#import "dlss_fg_session.h"
#include "netal_sr_control_policy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mach/mach.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <vector>
#include <unistd.h>
#include <string>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreGraphics/CoreGraphics.h>

namespace {

std::atomic<unsigned long long> g_create_count{0};
std::atomic<unsigned long long> g_encode_count{0};
IMP g_original_new_scaler = nullptr;
// Metal 4 MetalFX (GPTK 4 / D3DMetal 4): scalers created with
// newTemporalScalerWithDevice:compiler: encode into MTL4 command buffers,
// which the Metal 3 session cannot encode into yet; they are logged and passed
// through untouched.
IMP g_original_new_scaler_m4 = nullptr;
IMP g_original_new_spatial_m4 = nullptr;
std::vector<void*> g_metal4_scalers;
std::mutex g_metal4_mutex;
unsigned long g_metal4_passthrough_logs = 0;
bool IsMetal4Scaler(id object) {
  std::lock_guard<std::mutex> lock(g_metal4_mutex);
  for (void* p : g_metal4_scalers) if (p == (__bridge void*)object) return true;
  return false;
}
FILE* g_log = nullptr;

struct EncodeHook {
  Class scaler_class;
  IMP original;
};

std::mutex g_hook_mutex;
std::vector<EncodeHook> g_encode_hooks;
std::mutex g_sr_mutex;
__strong NeuralStudioDLSSSRSession* g_sr_session = nil;
__strong id<MTLDevice> g_sr_device = nil;
__strong NSURL* g_sr_bundle_url = nil;
unsigned long g_sr_profile_failures = 0;
// After this many failed session builds the bridge gives up for the life of the process, rather
// than recompiling shaders on every scaler query for a profile that is never going to load.
constexpr unsigned long kProfileFailureCap = 8;
void Log(const char* format, ...);
bool RespondsToFrameInterpolatorInputs(Class effect_class);
// Live diagnostics (env / injector markers): jitter and motion-vector sign
// switches for A/B tests of the MetalFX vs DLSS conventions, a live
// input/output dump for offline replay, and presentation logging / forced vsync.
float g_sr_jitter_sign = 1.0f;
float g_sr_mv_sign = 1.0f;
std::string g_sr_dump_dir;
unsigned long long g_sr_dump_start = 0;
unsigned long long g_sr_flash_frames = 0;
unsigned long long g_sr_dump_count = 0;
unsigned long long g_sr_dump_written = 0;
bool g_force_vsync = false;
unsigned long g_layer_logs = 0;
// Staging for the Metal 4 commit hook; the hook itself is defined further down, next to the
// Metal 4 scaler plumbing. Declared here because the environment is read before that point.
enum class CommitHookStage { off, observe, feedback };
CommitHookStage g_commit_stage = CommitHookStage::off;

void ReadLiveDiagnosticsEnvironment() {
  const char* js = std::getenv("NEURALSTUDIO_RE9_JITTER_SIGN");
  if (js != nullptr && std::strcmp(js, "-1") == 0) g_sr_jitter_sign = -1.0f;
  const char* ms = std::getenv("NEURALSTUDIO_RE9_MV_SIGN");
  if (ms != nullptr && std::strcmp(ms, "-1") == 0) g_sr_mv_sign = -1.0f;
  const char* dump = std::getenv("NEURALSTUDIO_RE9_SR_DUMP");  // "dir,start,count"
  if (dump != nullptr && *dump != '\0') {
    std::string spec(dump);
    const size_t c1 = spec.find(','), c2 = spec.find(',', c1 == std::string::npos ? spec.size() : c1 + 1);
    if (c1 != std::string::npos && c2 != std::string::npos) {
      g_sr_dump_dir = spec.substr(0, c1);
      g_sr_dump_start = std::strtoull(spec.substr(c1 + 1, c2 - c1 - 1).c_str(), nullptr, 10);
      g_sr_dump_count = std::strtoull(spec.substr(c2 + 1).c_str(), nullptr, 10);
    }
  }
  const char* vsync = std::getenv("NEURALSTUDIO_RE9_FORCE_VSYNC");
  g_force_vsync = vsync != nullptr && std::strcmp(vsync, "1") == 0;
  if (const char* stage = std::getenv("NEURALSTUDIO_SR_METAL4_COMMIT_HOOK")) {
    if (std::strcmp(stage, "observe") == 0) g_commit_stage = CommitHookStage::observe;
    else if (std::strcmp(stage, "feedback") == 0) g_commit_stage = CommitHookStage::feedback;
  }
  Log("event=live_diagnostics jitter_sign=%g mv_sign=%g dump=%s start=%llu count=%llu force_vsync=%d",
      g_sr_jitter_sign, g_sr_mv_sign, g_sr_dump_dir.empty() ? "-" : g_sr_dump_dir.c_str(),
      g_sr_dump_start, g_sr_dump_count, g_force_vsync ? 1 : 0);
}

bool WriteBuffer(const std::string& path, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger length) {
  if (buffer == nil || buffer.contents == nullptr || offset + length > buffer.length) return false;
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) return false;
  const size_t written = std::fwrite(static_cast<const std::uint8_t*>(buffer.contents) + offset, 1, length, file);
  std::fclose(file);
  return written == length;
}

// CAMetalLayer presentation diagnostics: log the layer state the first times
// a drawable is requested and optionally force display sync (vsync).
IMP g_original_next_drawable = nullptr;
id NextDrawableHook(id self, SEL _cmd) {
  CAMetalLayer* layer = (CAMetalLayer*)self;
  const BOOL original_sync = layer.displaySyncEnabled;
  if (g_force_vsync && !layer.displaySyncEnabled) layer.displaySyncEnabled = YES;
  if (g_layer_logs < 8) {
    ++g_layer_logs;
    // Display refresh: nominal rate of the main display's current mode (0 on
    // variable-refresh panels that do not report one) and the display size.
    double refresh = 0.0; size_t display_w = 0, display_h = 0;
    CGDirectDisplayID display = CGMainDisplayID();
    if (CGDisplayModeRef mode = CGDisplayCopyDisplayMode(display)) {
      refresh = CGDisplayModeGetRefreshRate(mode);
      display_w = CGDisplayModeGetPixelWidth(mode);
      display_h = CGDisplayModeGetPixelHeight(mode);
      CGDisplayModeRelease(mode);
    }
    Log("event=metal_layer layer=%p displaySyncEnabled=%d maximumDrawableCount=%lu presentsWithTransaction=%d "
        "framebufferOnly=%d pixelFormat=%lu drawableSize=%.0fx%.0f forced=%d display_refresh_hz=%.2f display_pixels=%zux%zu original_displaySyncEnabled=%d",
        (__bridge void*)layer, layer.displaySyncEnabled ? 1 : 0, (unsigned long)layer.maximumDrawableCount,
        layer.presentsWithTransaction ? 1 : 0, layer.framebufferOnly ? 1 : 0, (unsigned long)layer.pixelFormat,
        layer.drawableSize.width, layer.drawableSize.height, g_force_vsync ? 1 : 0, refresh, display_w, display_h, original_sync ? 1 : 0);
  }
  return ((id (*)(id, SEL))g_original_next_drawable)(self, _cmd);
}

void InstallLayerHook() {
  Method method = class_getInstanceMethod([CAMetalLayer class], @selector(nextDrawable));
  if (method == nullptr) { Log("event=metal_layer_hook_failed"); return; }
  g_original_next_drawable = method_setImplementation(method, reinterpret_cast<IMP>(&NextDrawableHook));
  Log("event=metal_layer_hook installed=1");
}
__strong NeuralStudioDLSSTextureAdapter* g_sr_adapter = nil;
// Session creation compiles ~15 pipelines; in the game process that can take minutes on a cold
// shader cache, so it never runs on the loader or render thread: a serial background queue
// builds the session and swaps it in (release fence, then the armed flag).
std::atomic<bool> g_sr_init_pending{false};
// libdispatch worker threads are not reliably scheduled inside the Wine process (a dispatch_async
// block never ran in a live launch), so background work uses a detached pthread.
bool RunDetached(void (^block)(void)) {
  bool started = true;
  void (^copy)(void) = [block copy];
  pthread_t thread;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_setstacksize(&attributes, 16u << 20);
  pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  void* context = (void*)CFBridgingRetain(copy);
  if (pthread_create(&thread, &attributes, [](void* raw) -> void* {
        @autoreleasepool {
          void (^run)(void) = (void (^)(void))CFBridgingRelease(raw);
          run();
        }
        return nullptr;
      }, context) != 0) {
    CFBridgingRelease(context);
    Log("event=runtime_thread_failed");
    started = false;
  }
  pthread_attr_destroy(&attributes);
  return started;
}
bool g_sr_armed = false;
bool g_sr_encoded = false;
std::atomic<unsigned int> g_sr_fallback_logs{0};
std::atomic<unsigned int> g_sr_passthrough_logs{0};

void Log(const char* format, ...) {
  if (g_log == nullptr) return;
  flockfile(g_log);
  va_list arguments;
  va_start(arguments, format);
  std::vfprintf(g_log, format, arguments);
  va_end(arguments);
  std::fputc('\n', g_log);
  std::fflush(g_log);
  funlockfile(g_log);
}

// The bridge is injected into one chosen process, but Wine loads it into helpers too, so it
// double-checks that this really is the game. NEURALSTUDIO_TARGET_EXE carries whichever executable
// the launch targeted (--exe); without it this falls back to RE9 so older callers are unchanged.
bool IsTargetProcess() {
  NSString* target = @"re9.exe";
  if (const char* env = std::getenv("NEURALSTUDIO_TARGET_EXE")) {
    NSString* value = [@(env) stringByTrimmingCharactersInSet:
                           [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (value.length > 0) target = value;
  }
  target = target.lowercaseString;
  NSString* stem = target.stringByDeletingPathExtension;
  NSString* process = NSProcessInfo.processInfo.processName.lowercaseString;
  if ([process isEqualToString:target] || [process isEqualToString:stem]) return true;
  for (NSString* argument in NSProcessInfo.processInfo.arguments) {
    NSString* value = argument.lowercaseString;
    if ([value isEqualToString:target] ||
        [value hasSuffix:[@"\\" stringByAppendingString:target]] ||
        [value hasSuffix:[@"/" stringByAppendingString:target]]) {
      return true;
    }
  }
  return false;
}

void LogEncode(id<MTLFXTemporalScaler> scaler,
               id<MTLCommandBuffer> command_buffer) {
  const unsigned long long count = ++g_encode_count;
  id<MTLTexture> color = scaler.colorTexture;
  id<MTLTexture> depth = scaler.depthTexture;
  id<MTLTexture> motion = scaler.motionTexture;
  id<MTLTexture> exposure = scaler.exposureTexture;
  id<MTLTexture> output = scaler.outputTexture;
  if (count <= 8 || count % 600 == 0) {
    Log("event=encode count=%llu scaler=%p scaler_class=%s command=%p "
        "color=%p color_size=%lux%lu color_format=%lu "
        "color_storage=%lu color_usage=%lu depth=%p depth_size=%lux%lu "
        "depth_format=%lu depth_storage=%lu depth_usage=%lu motion=%p "
        "motion_size=%lux%lu motion_format=%lu motion_storage=%lu "
        "motion_usage=%lu exposure=%p exposure_size=%lux%lu "
        "exposure_format=%lu exposure_storage=%lu exposure_usage=%lu "
        "output=%p output_size=%lux%lu "
        "output_format=%lu output_storage=%lu output_usage=%lu "
        "content=%lux%lu jitter=%.9g,%.9g motion_scale=%.9g,%.9g "
        "pre_exposure=%.9g reset=%d depth_reversed=%d",
        count, (__bridge void*)scaler,
        class_getName([(NSObject*)scaler class]),
        (__bridge void*)command_buffer, (__bridge void*)color,
        static_cast<unsigned long>(color.width),
        static_cast<unsigned long>(color.height),
        static_cast<unsigned long>(color.pixelFormat),
        static_cast<unsigned long>(color.storageMode),
        static_cast<unsigned long>(color.usage), (__bridge void*)depth,
        static_cast<unsigned long>(depth.width),
        static_cast<unsigned long>(depth.height),
        static_cast<unsigned long>(depth.pixelFormat),
        static_cast<unsigned long>(depth.storageMode),
        static_cast<unsigned long>(depth.usage), (__bridge void*)motion,
        static_cast<unsigned long>(motion.width),
        static_cast<unsigned long>(motion.height),
        static_cast<unsigned long>(motion.pixelFormat),
        static_cast<unsigned long>(motion.storageMode),
        static_cast<unsigned long>(motion.usage), (__bridge void*)exposure,
        static_cast<unsigned long>(exposure.width),
        static_cast<unsigned long>(exposure.height),
        static_cast<unsigned long>(exposure.pixelFormat),
        static_cast<unsigned long>(exposure.storageMode),
        static_cast<unsigned long>(exposure.usage), (__bridge void*)output,
        static_cast<unsigned long>(output.width),
        static_cast<unsigned long>(output.height),
        static_cast<unsigned long>(output.pixelFormat),
        static_cast<unsigned long>(output.storageMode),
        static_cast<unsigned long>(output.usage),
        static_cast<unsigned long>(scaler.inputContentWidth),
        static_cast<unsigned long>(scaler.inputContentHeight),
        scaler.jitterOffsetX, scaler.jitterOffsetY,
        scaler.motionVectorScaleX, scaler.motionVectorScaleY,
        scaler.preExposure, scaler.reset, scaler.depthReversed);
  }
}

// Sustained mode (NEURALSTUDIO_RE9_SR_SUSTAINED=1): every hooked MetalFX
// encode runs the exact 14-launch graph; any frame the contract or encode
// rejects falls back to MetalFX for that frame and forces reset=1 on the next
// exact frame, because the session's temporal history missed a frame.
bool g_sr_sustained = false;
bool g_sr_force_reset = true;
unsigned long long g_sr_exact_frames = 0;
unsigned long long g_sr_fallback_frames = 0;
unsigned long long g_sr_game_resets = 0;  // frames where the game itself requested reset (camera cuts)
double g_sr_gpu_seconds_total = 0.0;
double g_sr_gpu_seconds_max = 0.0;
// Most recent frame's GPU time, so the pacing line can carry it: the menu bar plots frame time
// and SR cost on the same axis, and two series sampled at different points do not line up.
double g_sr_gpu_seconds_last = 0.0;

// Resolution profiles: the session is (re)created for the scaler's output size
// (profile) and input content size (render resolution). The session reads
// NEURALSTUDIO_SR_PROFILE / NEURALSTUDIO_SR_RENDER at creation; a missing
// profile in the bundle keeps MetalFX for that scaler.
// A game alternating between an upscaling scaler and a 1:1 one must not rebuild the session every
// frame. While an upscale has been asked for within this many scaler queries, 1:1 stays with
// MetalFX; past it the game is taken to be in DLAA and the 1:1 scaler is served. A few seconds at
// any frame rate.
constexpr unsigned long long kUpscalePreferenceQueries = 600;
unsigned long long g_scaler_queries = 0;
unsigned long long g_last_upscale_query = 0;

bool EnsureSessionForScaler(id<MTLFXTemporalScaler> scaler) {
  id<MTLTexture> output = scaler.outputTexture;
  const NSUInteger output_width = output.width;
  const NSUInteger output_height = output.height;
  const NSUInteger render_width = scaler.inputContentWidth;
  const NSUInteger render_height = scaler.inputContentHeight;
  if (output == nil || output_width == 0 || render_width == 0) {
    static unsigned long long no_geometry_refusals = 0;
    if (++no_geometry_refusals <= 4) {
      Log("event=sr_passthrough reason=no_geometry output_texture=%s render=%lux%lu output=%lux%lu",
          output == nil ? "nil" : "present",
          static_cast<unsigned long>(render_width), static_cast<unsigned long>(render_height),
          static_cast<unsigned long>(output_width), static_cast<unsigned long>(output_height));
    }
    return false;
  }
  // A game may keep several scalers alive and alternate between them frame to frame. Lies of P
  // runs one at 1728x1080 -> 1920x1200 and another at 1920x1200 -> 1920x1200, and switching
  // between them rebuilt the session every time - each rebuild recompiling shaders for minutes, so
  // it never stabilised and served no frames at all.
  //
  // A 1:1 scaler is degenerate for *upscaling* - there is nothing to reconstruct - but not for the
  // network, and that is what DLAA is: the same reconstruction at native resolution, for
  // anti-aliasing rather than for pixels. Refusing every 1:1 scaler meant a game set to DLAA
  // silently got nothing, with the runtime healthy and the log saying only "passthrough".
  //
  // The refusal still applies while a game is *also* upscaling, which is the case the comment above
  // describes: an active upscaling scaler keeps identity ones on MetalFX so the session holds one
  // geometry. Only when nothing has asked to upscale for a while is 1:1 served - a game genuinely in
  // DLAA rather than alternating.
  ++g_scaler_queries;
  const bool downscale = render_width > output_width || render_height > output_height;
  const bool identity = !downscale && render_width == output_width && render_height == output_height;
  if (!downscale && !identity) g_last_upscale_query = g_scaler_queries;
  if (downscale) {
    static unsigned long long downscale_refusals = 0;
    if (++downscale_refusals <= 4) {
      Log("event=sr_passthrough reason=downscale render=%lux%lu output=%lux%lu",
          static_cast<unsigned long>(render_width), static_cast<unsigned long>(render_height),
          static_cast<unsigned long>(output_width), static_cast<unsigned long>(output_height));
    }
    return false;
  }
  if (identity && g_last_upscale_query != 0 &&
      g_scaler_queries - g_last_upscale_query < kUpscalePreferenceQueries) {
    static unsigned long long identity_passthroughs = 0;
    if (++identity_passthroughs <= 4) {
      Log("event=sr_passthrough reason=identity_scale_upscaler_active render=%lux%lu output=%lux%lu"
          " queries_since_upscale=%llu",
          static_cast<unsigned long>(render_width), static_cast<unsigned long>(render_height),
          static_cast<unsigned long>(output_width), static_cast<unsigned long>(output_height),
          g_scaler_queries - g_last_upscale_query);
    }
    return false;
  }
  if (identity) {
    static bool announced = false;
    if (!announced) {
      announced = true;
      Log("event=sr_dlaa render=%lux%lu output=%lux%lu (identity scale served: the network runs at"
          " native resolution for anti-aliasing)",
          static_cast<unsigned long>(render_width), static_cast<unsigned long>(render_height),
          static_cast<unsigned long>(output_width), static_cast<unsigned long>(output_height));
    }
  }
  // A background build owns the pending configuration. Do not read partially published
  // session pointers or rewrite its geometry environment while it is compiling.
  if (g_sr_init_pending.load()) return false;
  if (g_sr_session != nil && g_sr_adapter != nil && !g_sr_session.isConsumed &&
      g_sr_session.outputWidth == output_width && g_sr_session.outputHeight == output_height &&
      g_sr_session.inputWidth == render_width && g_sr_session.inputHeight == render_height) {
    return true;
  }
  if (g_sr_device == nil || g_sr_bundle_url == nil) {
    // Nothing to build a session from: the bundle never resolved, or arming ran before the device
    // was known. Every frame passes through, so say so once rather than looking like a healthy run.
    static bool announced = false;
    if (!announced) {
      announced = true;
      Log("event=sr_passthrough reason=not_armed device=%s bundle=%s",
          g_sr_device == nil ? "nil" : "present", g_sr_bundle_url == nil ? "nil" : "present");
    }
    return false;
  }
  char profile[32];
  char render[32];
  std::snprintf(profile, sizeof(profile), "%lux%lu", static_cast<unsigned long>(output_width),
                static_cast<unsigned long>(output_height));
  std::snprintf(render, sizeof(render), "%lux%lu", static_cast<unsigned long>(render_width),
                static_cast<unsigned long>(render_height));
  setenv("NEURALSTUDIO_SR_PROFILE", profile, 1);
  setenv("NEURALSTUDIO_SR_RENDER", render, 1);
  // Hardware (Apple-native) mode: the non-exact Hiluma input kernel keeps the
  // RTX deviation band (81.8 dB at frame 3) and saves ~0.6 ms; the output
  // kernel stays exact (the legacy pair drifts to 64.7 dB).
  const char* suffix = std::getenv("NEURALSTUDIO_SR_PWIN_SUFFIX");
  // every hardware lowering family (_hw, _hw7 = split kernels, ...) shares the Apple-native input kernel
  if (suffix != nullptr && std::strncmp(suffix, "_hw", 3) == 0 &&
      std::getenv("NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY") == nullptr) {
    setenv("NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY", "hiluma_input_depthinv", 1);
  }
  if (suffix != nullptr && *suffix != '\0') setenv("NEURALSTUDIO_SR_ALLOW_ANY_JITTER", "1", 1);
  // batched per-token pipelines (bundle profiles/<WxH>/pipeline/<entry>.json) replace the transpiled
  // enc0/enc1/dec0 kernels: bit-identical to the _hw lowering on every captured profile, ~1.5 ms faster
  // at 1080p; NEURALSTUDIO_SR_PWIN_PIPELINE=0 in the environment keeps the transpiled kernels
  if (suffix != nullptr && std::strncmp(suffix, "_hw", 3) == 0 &&
      std::getenv("NEURALSTUDIO_SR_PWIN_PIPELINE") == nullptr) {
    setenv("NEURALSTUDIO_SR_PWIN_PIPELINE", "1", 1);
  }
  if (g_sr_profile_failures >= kProfileFailureCap) {
    // Terminal: after this many failed session builds the bridge stops trying for the rest of the
    // process. Every previous attempt logged sr_profile_unavailable, but nothing said the retries
    // had stopped, so the log went quiet and a dead runtime read as a healthy one.
    static bool announced = false;
    if (!announced) {
      announced = true;
      Log("event=sr_disabled reason=profile_failures failures=%llu output=%s render=%s"
          " (no further session builds will be attempted; MetalFX serves every frame from here)",
          static_cast<unsigned long long>(g_sr_profile_failures), profile, render);
    }
    return false;
  }
  g_sr_init_pending = true;
  std::string profile_s(profile), render_s(render);
  Log("event=sr_profile_pending output=%s render=%s", profile, render);
  if (!RunDetached(^{
    NSError* session_error = nil;
    NeuralStudioDLSSSRSession* session = [[NeuralStudioDLSSSRSession alloc]
        initWithDevice:g_sr_device runtimeBundleURL:g_sr_bundle_url error:&session_error];
    NeuralStudioDLSSTextureAdapter* adapter =
        session == nil ? nil
                       : [[NeuralStudioDLSSTextureAdapter alloc] initWithDevice:g_sr_device
                                                                       session:session
                                                                         error:&session_error];
    // Publish the session/adapter pair under the same mutex as the SR render hook.
    // Control changes are deferred while pending; FG polling never mutates SR state.
    std::lock_guard<std::mutex> publish_lock(g_sr_mutex);
    if (adapter == nil) {
      ++g_sr_profile_failures;
      Log("event=sr_profile_unavailable output=%s render=%s detail=%s", profile_s.c_str(), render_s.c_str(),
          session_error == nil ? "unknown" : session_error.localizedDescription.UTF8String);
    } else {
      g_sr_session = session;
      g_sr_adapter = adapter;
      // Batched-pipeline coverage, into the log the app follows. The session reports this on
      // stderr, which nothing reads: a profile missing layers is a ~2x slowdown that looks
      // like a healthy session, so it has to be visible where someone is actually looking.
      // total==0 means the batched pipelines are switched off entirely, which is a choice rather
      // than a half-generated profile; logging it would only be noise the app has to filter.
      if (session.pipelineLayersTotal > 0)
      Log("event=sr_pipeline_layers present=%lu total=%lu missing=%s",
          static_cast<unsigned long>(session.pipelineLayersPresent),
          static_cast<unsigned long>(session.pipelineLayersTotal),
          session.pipelineLayersMissing.length > 0 ? session.pipelineLayersMissing.UTF8String : "-");

      std::atomic_thread_fence(std::memory_order_release);
      g_sr_force_reset = true;
      Log("event=sr_profile_selected output=%s render=%s", profile_s.c_str(), render_s.c_str());
    }
    g_sr_init_pending = false;
  })) {
    g_sr_init_pending = false;
    ++g_sr_profile_failures;
    Log("event=sr_profile_unavailable reason=thread_start_failed");
  }
  return false;
}

unsigned long long g_sr_metal4_frames = 0;
// Runtime pause switch: while <log dir>/re9-sr-pause exists every hooked encode
// passes through to the original MetalFX encode (instant A/B without a
// relaunch); resuming forces a history reset. Checked every 30 encodes.
std::string g_sr_pause_marker;
bool g_sr_paused = false;
unsigned long long g_sr_pause_checks = 0;
// "That just happened" marker. White flashes are rare and unpredictable, so a fixed dump window
// cannot catch one; this lets whoever is playing stamp the current frame into the log the moment
// they see an artifact. Checked on the same 30-encode cadence as the pause marker, and consumed
// (deleted) so each press records exactly once.
std::string g_sr_mark_marker;
unsigned long long g_sr_marks = 0;
void CheckArtifactMark(unsigned long long frame) {
  if (g_sr_mark_marker.empty()) return;
  if (access(g_sr_mark_marker.c_str(), F_OK) != 0) return;
  char note[256] = {0};
  FILE* file = std::fopen(g_sr_mark_marker.c_str(), "r");
  if (file != nullptr) {
    if (std::fgets(note, sizeof note - 1, file) != nullptr) {
      for (char* c = note; *c != '\0'; ++c) { if (*c == '\n' || *c == '\r') { *c = '\0'; break; } }
    }
    std::fclose(file);
  }
  std::remove(g_sr_mark_marker.c_str());
  ++g_sr_marks;
  Log("event=sr_artifact_marked mark=%llu frame=%llu note=%s",
      g_sr_marks, frame, note[0] != '\0' ? note : "-");
}

// Live control channel. The pause marker proved the shape works - a file the runtime polls, so the
// app never has to talk to a process it did not start - so the rest of the knobs use the same
// mechanism instead of inventing a second one.
//
// `netal-control` sits beside the pause marker and holds key=value lines. Two of these change how
// the network is lowered, which is fixed when the session is built, so applying them drops the
// session; the next scaler query rebuilds it with the new environment. That costs a shader
// recompile, which is why the file is polled rather than watched: a user dragging a switch should
// not rebuild once per intermediate value.
std::string g_sr_control_path;
std::string g_sr_control_suffix;
int g_sr_control_pipeline = 1;
// Frame generation on/off from the same control file (`fg=0` hands the interpolator back to
// MetalFX for every frame; `fg=1` or no key serves). Read by the FG hook through FgControlEnabled().
std::atomic<bool> g_fg_control_enabled{true};
bool g_sr_control_seeded = false;
netal::ControlPollCadence g_control_poll_cadence;
std::mutex g_sr_control_mutex;

void InvalidateSessionForControlChange(const char* what) {
  Log("event=sr_control_applied change=%s (session dropped; rebuilding on next scaler)", what);
  g_sr_session = nil;
  g_sr_adapter = nil;
}

// apply_sr=true is called only while the SR hook owns g_sr_mutex. FG callers can
// refresh their atomic enable flag without touching SR objects or its build environment.
void PollControlFile(bool apply_sr) {
  std::lock_guard<std::mutex> control_lock(g_sr_control_mutex);
  if (g_sr_control_path.empty()) return;
  if (!g_control_poll_cadence.Due(apply_sr)) return;
  FILE* file = std::fopen(g_sr_control_path.c_str(), "r");
  if (file == nullptr) return;
  if (!g_sr_control_seeded) {
    // Seed from the environment the session was built with. Without this the first read of a file
    // that appears after launch looks like "nothing changed" and applies silently, so the switch
    // does nothing until the session happens to rebuild for another reason.
    const char* env_suffix = std::getenv("NEURALSTUDIO_SR_PWIN_SUFFIX");
    const char* env_pipeline = std::getenv("NEURALSTUDIO_SR_PWIN_PIPELINE");
    g_sr_control_suffix = env_suffix != nullptr ? env_suffix : "";
    g_sr_control_pipeline = (env_pipeline != nullptr && env_pipeline[0] == '0') ? 0 : 1;
    g_sr_control_seeded = true;
  }
  std::string suffix = g_sr_control_suffix;
  int pipeline = g_sr_control_pipeline;
  bool fg_enabled = true;
  char line[256];
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    std::string text(line);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
      text.pop_back();
    }
    const size_t split = text.find('=');
    if (split == std::string::npos) continue;
    const std::string key = text.substr(0, split);
    const std::string value = text.substr(split + 1);
    if (key == "suffix") suffix = value;            // "" = exact reference lowering, _hw7 = hardware
    else if (key == "pipeline") pipeline = value == "0" ? 0 : 1;
    else if (key == "fg") fg_enabled = value != "0";
  }
  std::fclose(file);
  if (fg_enabled != g_fg_control_enabled.load()) {
    g_fg_control_enabled.store(fg_enabled);
    Log("event=%s control=%s", fg_enabled ? "fg_resumed" : "fg_paused", g_sr_control_path.c_str());
  }

  if (!netal::MayApplySRControl(apply_sr, g_sr_init_pending.load())) return;
  const char* current_input = std::getenv("NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY");
  const auto mode = netal::PlanSRControl(suffix, pipeline != 0,
                                        current_input == nullptr ? "" : current_input);
  pipeline = mode.pipeline ? 1 : 0;
  if (mode.clear_hardware_input) {
    unsetenv("NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY");
    Log("event=sr_control_boundary mode=reference previous=hiluma_input_depthinv selection=bundle_default");
    InvalidateSessionForControlChange("reference_boundary");
  }

  if (suffix != g_sr_control_suffix) {
    if (suffix.empty()) unsetenv("NEURALSTUDIO_SR_PWIN_SUFFIX");
    else setenv("NEURALSTUDIO_SR_PWIN_SUFFIX", suffix.c_str(), 1);
    Log("event=sr_control_suffix from=%s to=%s", g_sr_control_suffix.c_str(), suffix.c_str());
    g_sr_control_suffix = suffix;
    InvalidateSessionForControlChange("suffix");
  }
  if (pipeline != g_sr_control_pipeline) {
    setenv("NEURALSTUDIO_SR_PWIN_PIPELINE", pipeline ? "1" : "0", 1);
    Log("event=sr_control_pipeline from=%d to=%d", g_sr_control_pipeline, pipeline);
    g_sr_control_pipeline = pipeline;
    InvalidateSessionForControlChange("pipeline");
  }
}

bool SrPaused() {
  if (g_sr_pause_marker.empty()) return false;
  if ((g_sr_pause_checks++ % 30) == 0) {
    const bool paused = access(g_sr_pause_marker.c_str(), F_OK) == 0;
    if (paused != g_sr_paused) {
      g_sr_paused = paused;
      if (!paused) g_sr_force_reset = true;
      Log("event=%s marker=%s", paused ? "sr_paused" : "sr_resumed", g_sr_pause_marker.c_str());
    }
  }
  return g_sr_paused;
}

// Frame pacing, measured on the CPU as the interval between successive SR encodes. On the Metal 4
// path this is the only speed signal there is: MTL4CommandBuffer has no completion handler, so
// command_gpu_ms cannot be measured. mean_ms is the game's frame time over the window and max_ms
// catches the spikes that show red on the Metal HUD.
struct FramePacing {
  std::chrono::steady_clock::time_point last{};
  double sum_ms = 0.0, min_ms = 1e9, max_ms = 0.0, worst_ms = 0.0;
  unsigned long long count = 0, over_25ms = 0;
  bool started = false;
};
FramePacing g_pacing;
// One pacing sample per 60 frames - about a second at playable rates, which is the cadence the
// menu bar polls at. 120 left the graph with a point every two seconds and visible stepping.
constexpr unsigned long long kPacingWindow = 60;

void NotePacing(unsigned long long frame) {
  const auto now = std::chrono::steady_clock::now();
  if (g_pacing.started) {
    const double dt = std::chrono::duration<double, std::milli>(now - g_pacing.last).count();
    if (dt > 0.0 && dt < 2000.0) {
      g_pacing.sum_ms += dt;
      ++g_pacing.count;
      if (dt < g_pacing.min_ms) g_pacing.min_ms = dt;
      if (dt > g_pacing.max_ms) g_pacing.max_ms = dt;
      if (dt > g_pacing.worst_ms) g_pacing.worst_ms = dt;
      if (dt > 25.0) ++g_pacing.over_25ms;
    }
  }
  g_pacing.last = now;
  g_pacing.started = true;
  if (frame % kPacingWindow == 0 && g_pacing.count > 0) {
    // Resident size of the game process alongside the pacing, so a frame-time decay can be told
    // apart from a leak: rss climbing with fps falling is ours, flat rss is contention elsewhere.
    double rss_mb = 0.0;
    mach_task_basic_info info{};
    mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &info_count) == KERN_SUCCESS) {
      rss_mb = static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
    // Thermal state, because the frame-time decay needs one measurement to settle it.
    //
    // A long session degrades badly - the SR pass goes 8 to 37 ms - and command_gpu_ms alone cannot
    // say why: it is GPU wall-clock for our command buffer, so it inflates identically whether our
    // code got slower or the GPU got slower underneath it. The shape of the curve argues for the
    // latter - flat for 31,000 frames, then a break, with min_ms unmoved and RSS plateaued while
    // the degradation continued - and this records the one variable that would confirm it. Needs no
    // privileges, unlike powermetrics.
    const char* thermal = "nominal";
    switch (NSProcessInfo.processInfo.thermalState) {
      case NSProcessInfoThermalStateNominal:  thermal = "nominal"; break;
      case NSProcessInfoThermalStateFair:     thermal = "fair"; break;
      case NSProcessInfoThermalStateSerious:  thermal = "serious"; break;
      case NSProcessInfoThermalStateCritical: thermal = "critical"; break;
    }

    Log("event=sr_pacing frame=%llu window=%llu mean_ms=%.2f fps=%.1f min_ms=%.2f max_ms=%.2f "
        "spikes_over_25ms=%llu worst_ms_session=%.2f rss_mb=%.0f gpu_ms=%.2f thermal=%s",
        frame, g_pacing.count, g_pacing.sum_ms / static_cast<double>(g_pacing.count),
        1000.0 * static_cast<double>(g_pacing.count) / g_pacing.sum_ms,
        g_pacing.min_ms, g_pacing.max_ms, g_pacing.over_25ms, g_pacing.worst_ms, rss_mb,
            g_sr_gpu_seconds_last * 1000.0, thermal);
    g_pacing.sum_ms = 0.0; g_pacing.count = 0; g_pacing.min_ms = 1e9; g_pacing.max_ms = 0.0;
  }
}

// Watch the finished frame for the signature of a blown-out output, so a rare flash reports
// itself instead of depending on someone seeing it.
//
// Cheap enough to leave on: the output buffer is Shared, so the CPU reads it directly with no GPU
// work and no readback, and only a few hundred pixels are sampled. It runs in the completion
// handler, after the GPU has finished with that frame.
//
// Counted straight from the raw half bit patterns, so no conversion is needed:
//   fill      exactly 0x5a5a - the poison the removed per-frame memset used to leave behind,
//             which is half 203.25. Non-zero here means that race is back.
//   hot       magnitude >= 0x4c00 (16.0). HDR legitimately exceeds 1.0, but a whole frame above
//             16 is not a scene, it is a blown buffer.
//   nonfinite exponent all ones - Inf or NaN, which would point at the maths rather than a race.
struct FlashSample { unsigned fill = 0, hot = 0, nonfinite = 0, taken = 0; };

FlashSample SampleOutputForFlash(id<MTLBuffer> output) {
  FlashSample sample;
  if (output == nil || output.contents == nullptr || output.length < 2) return sample;
  const std::uint16_t* halfs = static_cast<const std::uint16_t*>(output.contents);
  const std::size_t count = output.length / 2;
  // A flash is whole-frame, so a sparse sweep catches it and costs nothing.
  const std::size_t want = 512;
  const std::size_t stride = count > want ? count / want : 1;
  for (std::size_t i = 0; i < count; i += stride) {
    const std::uint16_t bits = halfs[i];
    ++sample.taken;
    if (bits == 0x5a5au) ++sample.fill;
    const std::uint16_t magnitude = bits & 0x7fffu;
    if ((magnitude & 0x7c00u) == 0x7c00u) ++sample.nonfinite;
    else if (magnitude >= 0x4c00u) ++sample.hot;
  }
  return sample;
}

bool TryEncodeSustainedSRMetal4(id scaler_object, id command_object) {
  if (!g_sr_armed) return false;
  id<MTLFXTemporalScaler> scaler = (id<MTLFXTemporalScaler>)scaler_object;  // MTLFXTemporalScalerBase properties
  if (!EnsureSessionForScaler(scaler)) return false;
  const bool reset = scaler.reset || g_sr_force_reset;
  if (scaler.reset) ++g_sr_game_resets;
  NSError* error = nil;
  // The Metal 4 scaler class also carries a fence (selectors fence/setFence:).
  g_sr_adapter.pendingFence = [scaler_object respondsToSelector:@selector(fence)] ? [scaler_object fence] : nil;
  const BOOL encoded = [g_sr_adapter
      encodeTemporalFrameToMTL4CommandBuffer:(id<MTL4CommandBuffer>)command_object
                                colorTexture:scaler.colorTexture
                               motionTexture:scaler.motionTexture
                                depthTexture:scaler.depthTexture
                             exposureTexture:scaler.exposureTexture
                               outputTexture:scaler.outputTexture
                           inputContentWidth:scaler.inputContentWidth
                          inputContentHeight:scaler.inputContentHeight
                               jitterOffsetX:scaler.jitterOffsetX * g_sr_jitter_sign
                               jitterOffsetY:scaler.jitterOffsetY * g_sr_jitter_sign
                          motionVectorScaleX:scaler.motionVectorScaleX * g_sr_mv_sign
                          motionVectorScaleY:scaler.motionVectorScaleY * g_sr_mv_sign
                                 preExposure:scaler.preExposure
                                       reset:reset
                               depthReversed:scaler.depthReversed
                                       error:&error];
  if (!encoded) {
    ++g_sr_fallback_frames;
    g_sr_force_reset = true;
    if (g_sr_fallback_frames <= 64) {
      Log("event=sr_fallback api=metal4 reason=contract_or_encode frame=%llu detail=%s",
          g_sr_exact_frames + g_sr_fallback_frames,
          error == nil ? "unknown" : error.localizedDescription.UTF8String);
    }
    return false;
  }
  g_sr_force_reset = false;
  const unsigned long long frame = ++g_sr_exact_frames;
  ++g_sr_metal4_frames;
  NotePacing(frame);
  CheckArtifactMark(frame);
  if (frame <= 8 || frame % 120 == 0) {
    Log("event=sr_exact_scheduled mode=sustained api=metal4 frame=%llu reset=%d game_reset=%d jitter=%g,%g "
        "fallbacks=%llu game_resets=%llu",
        frame, reset ? 1 : 0, scaler.reset ? 1 : 0, scaler.jitterOffsetX, scaler.jitterOffsetY,
        g_sr_fallback_frames, g_sr_game_resets);
  }
  return true;
}

bool TryEncodeSustainedSR(id<MTLFXTemporalScaler> scaler,
                          id<MTLCommandBuffer> command_buffer) {
  if (!g_sr_armed) return false;
  if (!EnsureSessionForScaler(scaler)) return false;
  const bool reset = scaler.reset || g_sr_force_reset;
  if (scaler.reset) ++g_sr_game_resets;
  NSError* error = nil;
  g_sr_adapter.pendingFence = scaler.fence;
  static unsigned long fence_logs = 0;
  if (fence_logs < 3) {
    ++fence_logs;
    Log("event=sr_sync fence=%p color_hazard=%lu output_hazard=%lu motion_hazard=%lu depth_hazard=%lu",
        (__bridge void*)scaler.fence, (unsigned long)scaler.colorTexture.hazardTrackingMode,
        (unsigned long)scaler.outputTexture.hazardTrackingMode, (unsigned long)scaler.motionTexture.hazardTrackingMode,
        (unsigned long)scaler.depthTexture.hazardTrackingMode);
  }
  const BOOL encoded = [g_sr_adapter
      encodeTemporalFrameToCommandBuffer:command_buffer
                            colorTexture:scaler.colorTexture
                           motionTexture:scaler.motionTexture
                            depthTexture:scaler.depthTexture
                         exposureTexture:scaler.exposureTexture
                           outputTexture:scaler.outputTexture
                       inputContentWidth:scaler.inputContentWidth
                      inputContentHeight:scaler.inputContentHeight
                           jitterOffsetX:scaler.jitterOffsetX * g_sr_jitter_sign
                           jitterOffsetY:scaler.jitterOffsetY * g_sr_jitter_sign
                      motionVectorScaleX:scaler.motionVectorScaleX * g_sr_mv_sign
                      motionVectorScaleY:scaler.motionVectorScaleY * g_sr_mv_sign
                             preExposure:scaler.preExposure
                                   reset:reset
                           depthReversed:scaler.depthReversed
                                   error:&error];
  if (!encoded) {
    ++g_sr_fallback_frames;
    g_sr_force_reset = true;
    if (g_sr_fallback_frames <= 64) {
      Log("event=sr_fallback reason=contract_or_encode frame=%llu detail=%s",
          g_sr_exact_frames + g_sr_fallback_frames,
          error == nil ? "unknown" : error.localizedDescription.UTF8String);
    }
    return false;
  }
  g_sr_force_reset = false;
  const unsigned long long frame = ++g_sr_exact_frames;
  NotePacing(frame);
  CheckArtifactMark(frame);
  // Keep the game's per-frame textures alive until our (long) command buffer
  // has executed; the game recycles transient exposure/output textures.
  id<MTLTexture> retained_color = scaler.colorTexture;
  id<MTLTexture> retained_motion = scaler.motionTexture;
  id<MTLTexture> retained_depth = scaler.depthTexture;
  id<MTLTexture> retained_exposure = scaler.exposureTexture;
  id<MTLTexture> retained_output = scaler.outputTexture;
  if (frame <= 8 || frame % 120 == 0) {
    Log("event=sr_exact_scheduled mode=sustained frame=%llu reset=%d game_reset=%d "
        "jitter=%g,%g fallbacks=%llu",
        frame, reset ? 1 : 0, scaler.reset ? 1 : 0, scaler.jitterOffsetX,
        scaler.jitterOffsetY, g_sr_fallback_frames);
  }
  const bool dump_this_frame = !g_sr_dump_dir.empty() && frame >= g_sr_dump_start &&
                               frame < g_sr_dump_start + g_sr_dump_count;
  const float dump_jitter_x = scaler.jitterOffsetX * g_sr_jitter_sign;
  const float dump_jitter_y = scaler.jitterOffsetY * g_sr_jitter_sign;
  const float dump_scale_x = scaler.motionVectorScaleX * g_sr_mv_sign;
  const float dump_scale_y = scaler.motionVectorScaleY * g_sr_mv_sign;
  const float dump_pre_exposure = scaler.preExposure;
  const bool dump_depth_reversed = scaler.depthReversed;
  const NSUInteger dump_render_w = scaler.inputContentWidth, dump_render_h = scaler.inputContentHeight;
  const NSUInteger dump_output_w = scaler.outputTexture.width, dump_output_h = scaler.outputTexture.height;
  NeuralStudioDLSSTextureAdapter* dump_adapter = g_sr_adapter;
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    (void)retained_color; (void)retained_motion; (void)retained_depth;
    (void)retained_exposure; (void)retained_output;
    const double gpu_seconds = completed.GPUEndTime - completed.GPUStartTime;
    if (dump_this_frame && completed.status == MTLCommandBufferStatusCompleted) {
      // Same layout as an RTX capture directory (inputs at render resolution,
      // output at profile resolution) plus a per-frame params file.
      const unsigned long long index = frame - g_sr_dump_start;
      char prefix[64];
      std::snprintf(prefix, sizeof(prefix), "/frame-%03llu", index);
      const std::string base = g_sr_dump_dir + prefix;
      const bool ok = WriteBuffer(base + "-input-color.rgba16le", dump_adapter.diagnosticColorBuffer, 0, dump_render_w * dump_render_h * 8) &&
                      WriteBuffer(base + "-input-motion.rg16le", dump_adapter.diagnosticMotionBuffer, 0, dump_render_w * dump_render_h * 4) &&
                      WriteBuffer(base + "-input-depth.r32le", dump_adapter.diagnosticDepthBuffer, 0, dump_render_w * dump_render_h * 4) &&
                      WriteBuffer(base + "-input-exposure.r32le", dump_adapter.diagnosticExposureBuffer, 0, 4) &&
                      WriteBuffer(base + "-live-output.rgba16le", dump_adapter.diagnosticModelOutputBuffer, 0, dump_output_w * dump_output_h * 8);
      FILE* params = std::fopen((base + "-params.json").c_str(), "w");
      if (params != nullptr) {
        std::fprintf(params, "{\"frame\": %llu, \"jitter_x\": %.9g, \"jitter_y\": %.9g, \"motion_scale_x\": %.9g, \"motion_scale_y\": %.9g, "
                     "\"pre_exposure\": %.9g, \"reset\": %d, \"depth_reversed\": %d, \"render\": [%lu, %lu], \"output\": [%lu, %lu]}\n",
                     frame, dump_jitter_x, dump_jitter_y, dump_scale_x, dump_scale_y, dump_pre_exposure, reset ? 1 : 0,
                     dump_depth_reversed ? 1 : 0, (unsigned long)dump_render_w, (unsigned long)dump_render_h,
                     (unsigned long)dump_output_w, (unsigned long)dump_output_h);
        std::fclose(params);
      }
      ++g_sr_dump_written;
      Log("event=sr_dump frame=%llu index=%llu ok=%d dir=%s", frame, index, ok ? 1 : 0, g_sr_dump_dir.c_str());
    }
    std::lock_guard<std::mutex> completion_lock(g_sr_mutex);
    g_sr_gpu_seconds_last = gpu_seconds;
    g_sr_gpu_seconds_total += gpu_seconds;
    if (gpu_seconds > g_sr_gpu_seconds_max) g_sr_gpu_seconds_max = gpu_seconds;
    if (completed.status != MTLCommandBufferStatusCompleted && g_sr_armed) {
      // Never let an exact-path GPU error take the game down: disarm and let
      // MetalFX carry every following frame.
      g_sr_armed = false;
      Log("event=sr_disarmed reason=gpu_error frame=%llu status=%lu", frame,
          static_cast<unsigned long>(completed.status));
    }
    // Frame-ring backpressure: a rising wait count means the game submits SR frames faster than
    // they complete, which is the condition under which the ring used to be rewritten under an
    // executing frame. Logged the moment it changes, not on the 120-frame cadence, because the
    // artifact it used to cause was rare.
    static std::uint64_t last_waits = 0, last_timeouts = 0;
    std::uint64_t waits = 0, timeouts = 0; int max_in_flight = 0;
    if (g_sr_session != nil) {
      [g_sr_session neuralStudioRingStatsWaits:&waits timeouts:&timeouts maxInFlight:&max_in_flight];
    }
    // A flash is whole-frame, so require most of the sweep to agree before calling it one.
    const FlashSample flash = SampleOutputForFlash([g_sr_adapter diagnosticModelOutputBuffer]);
    const bool flashed = flash.taken > 0 &&
                         (flash.fill * 2 > flash.taken || flash.hot * 2 > flash.taken ||
                          flash.nonfinite > 0);
    if (flashed) {
      ++g_sr_flash_frames;
      Log("event=sr_flash frame=%llu fill=%u hot=%u nonfinite=%u of=%u gpu_ms=%.2f total=%llu detail=%s",
          frame, flash.fill, flash.hot, flash.nonfinite, flash.taken, gpu_seconds * 1000.0,
          g_sr_flash_frames,
          flash.fill * 2 > flash.taken
              ? "poison fill reached the output - the removed memset race is back"
              : (flash.nonfinite > 0 ? "non-finite output - maths, not a race"
                                     : "output blown above 16.0"));
    }
    const bool ring_changed = waits != last_waits || timeouts != last_timeouts;
    if (ring_changed) { last_waits = waits; last_timeouts = timeouts; }
    if (frame <= 8 || frame % 120 == 0 || ring_changed ||
        completed.status != MTLCommandBufferStatusCompleted) {
      Log("event=sr_exact_completed mode=sustained frame=%llu status=%lu "
          "command_gpu_ms=%.3f mean_command_gpu_ms=%.3f max_command_gpu_ms=%.3f game_resets=%llu "
          "ring_waits=%llu ring_timeouts=%llu max_in_flight=%d flashes=%llu error=%s",
          frame, static_cast<unsigned long>(completed.status), gpu_seconds * 1000.0,
          g_sr_gpu_seconds_total / static_cast<double>(frame) * 1000.0,
          g_sr_gpu_seconds_max * 1000.0, g_sr_game_resets, waits, timeouts, max_in_flight,
          g_sr_flash_frames,
          completed.error == nil ? "none" : completed.error.localizedDescription.UTF8String);
    }
  }];
  return true;
}

bool TryEncodeExactSR(id<MTLFXTemporalScaler> scaler,
                      id<MTLCommandBuffer> command_buffer) {
  std::lock_guard<std::mutex> lock(g_sr_mutex);
  // Metal 4 is decided by the command buffer the game hands us, not by how the
  // scaler was created (scalers made before injection never hit the creation
  // hook). A Metal 3 command buffer never conforms to MTL4CommandBuffer.
  PollControlFile(true);
  if (SrPaused()) return false;  // original MetalFX encode runs
  const bool metal4_command = [command_buffer conformsToProtocol:@protocol(MTL4CommandBuffer)] ||
                              IsMetal4Scaler(scaler);
  if (metal4_command) {
    if (g_metal4_passthrough_logs < 4) {
      ++g_metal4_passthrough_logs;
      Log("event=encode_metal4 scaler=%p scaler_class=%s command_class=%s sustained=%d", (__bridge void*)scaler,
          class_getName([(NSObject*)scaler class]), class_getName([(NSObject*)command_buffer class]),
          g_sr_sustained ? 1 : 0);
    }
    if (!g_sr_sustained) return false;
    @try {
      return TryEncodeSustainedSRMetal4(scaler, command_buffer);
    } @catch (NSException* exception) {
      g_sr_armed = false;
      Log("event=sr_disarmed reason=exception api=metal4 name=%s detail=%s", exception.name.UTF8String,
          exception.reason == nil ? "-" : exception.reason.UTF8String);
      return false;  // original MetalFX encode runs
    }
  }
  if (g_sr_sustained) {
    @try {
      return TryEncodeSustainedSR(scaler, command_buffer);
    } @catch (NSException* exception) {
      g_sr_armed = false;
      Log("event=sr_disarmed reason=exception api=metal3 name=%s detail=%s", exception.name.UTF8String,
          exception.reason == nil ? "-" : exception.reason.UTF8String);
      return false;
    }
  }
  if (g_sr_encoded) {
    if (++g_sr_passthrough_logs <= 8) {
      Log("event=sr_passthrough reason=one_frame_complete");
    }
    return false;
  }
  if (!g_sr_armed) return false;
  if (g_sr_adapter == nil) {
    if (++g_sr_fallback_logs <= 64) {
      Log("event=sr_fallback reason=session_unavailable");
    }
    return false;
  }
  NSError* error = nil;
  BOOL encoded = [g_sr_adapter
      encodeResetFrameToCommandBuffer:command_buffer
                          colorTexture:scaler.colorTexture
                         motionTexture:scaler.motionTexture
                          depthTexture:scaler.depthTexture
                       exposureTexture:scaler.exposureTexture
                         outputTexture:scaler.outputTexture
                     inputContentWidth:scaler.inputContentWidth
                    inputContentHeight:scaler.inputContentHeight
                         jitterOffsetX:scaler.jitterOffsetX
                         jitterOffsetY:scaler.jitterOffsetY
                    motionVectorScaleX:scaler.motionVectorScaleX
                    motionVectorScaleY:scaler.motionVectorScaleY
                           preExposure:scaler.preExposure
                                 reset:scaler.reset
                         depthReversed:scaler.depthReversed
                                 error:&error];
  if (!encoded) {
    if (++g_sr_fallback_logs <= 64) {
      Log("event=sr_fallback reason=contract_or_encode detail=%s",
          error == nil ? "unknown" : error.localizedDescription.UTF8String);
    }
    if (g_sr_session.isConsumed) g_sr_armed = false;
    return false;
  }
  g_sr_encoded = true;
  g_sr_armed = false;
  Log("event=sr_exact_scheduled bundle=%s input=1280x720 output=1920x1080 "
      "dispatches=14 fallback_used=false",
      g_sr_session.runtimeBundleRoot.UTF8String);
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    Log("event=sr_exact_completed status=%lu error=%s",
        static_cast<unsigned long>(completed.status),
        completed.error == nil ? "none"
                               : completed.error.localizedDescription.UTF8String);
  }];
  return true;
}

}  // namespace

@interface NeuralStudioTemporalScalerProxy : NSProxy {
 @private
  id<MTLFXTemporalScaler> _target;
}
+ (instancetype)proxyWithTarget:(id<MTLFXTemporalScaler>)target;
@end

@implementation NeuralStudioTemporalScalerProxy

+ (instancetype)proxyWithTarget:(id<MTLFXTemporalScaler>)target {
  NeuralStudioTemporalScalerProxy* proxy =
      [NeuralStudioTemporalScalerProxy alloc];
  proxy->_target = target;
  return proxy;
}

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
  LogEncode(_target, commandBuffer);
  [_target encodeToCommandBuffer:commandBuffer];
}

- (NSMethodSignature*)methodSignatureForSelector:(SEL)selector {
  return [(NSObject*)_target methodSignatureForSelector:selector];
}

- (void)forwardInvocation:(NSInvocation*)invocation {
  [invocation invokeWithTarget:_target];
}

- (BOOL)respondsToSelector:(SEL)selector {
  return selector == @selector(encodeToCommandBuffer:) ||
         [(NSObject*)_target respondsToSelector:selector];
}

- (BOOL)conformsToProtocol:(Protocol*)protocol {
  return protocol == @protocol(MTLFXTemporalScaler) ||
         [(NSObject*)_target conformsToProtocol:protocol];
}

- (Class)class {
  return [(NSObject*)_target class];
}

- (BOOL)isKindOfClass:(Class)classObject {
  return [(NSObject*)_target isKindOfClass:classObject];
}

- (NSUInteger)hash {
  return [(NSObject*)_target hash];
}

- (BOOL)isEqual:(id)object {
  return [(NSObject*)_target isEqual:object];
}

- (NSString*)description {
  return [(NSObject*)_target description];
}

@end

namespace {

using NewScaler = id (*)(id, SEL, id<MTLDevice>);
using Encode = void (*)(id, SEL, id<MTLCommandBuffer>);

IMP FindOriginalEncode(Class scaler_class) {
  std::lock_guard<std::mutex> lock(g_hook_mutex);
  for (Class current = scaler_class; current != Nil;
       current = class_getSuperclass(current)) {
    for (const EncodeHook& hook : g_encode_hooks) {
      if (hook.scaler_class == current) return hook.original;
    }
  }
  return nullptr;
}

// Every property the encode path reads off a scaler.
//
// The class-name filter below is a net, not an identification: "MFX" matches MetalFX's spatial
// scaler and its frame interpolator as surely as its temporal one, and those are different objects
// with different properties. Hooking one of them and then reading `depthTexture` off it is an
// unrecognized selector, which in Objective-C is not a nil return but an abort - and the abort
// happens on the game's render thread, so the game dies. Resident Evil 4 ran 6000 temporal encodes
// and was killed by the first frame it put through _MFXSpatialScalingEffectEFFECT_NAME_V1.
//
// So the real question is not what a class is called but whether it answers what we intend to ask.
// This is that question, asked once per class at install time. A scaler that cannot answer is not
// one we can replace anyway: no depth, no motion, no jitter means nothing the network needs.
bool RespondsToTemporalEncodeInputs(Class scaler_class) {
  static const SEL kInputs[] = {
      @selector(colorTexture),      @selector(depthTexture),
      @selector(motionTexture),     @selector(exposureTexture),
      @selector(outputTexture),     @selector(inputContentWidth),
      @selector(inputContentHeight), @selector(jitterOffsetX),
      @selector(jitterOffsetY),     @selector(motionVectorScaleX),
      @selector(motionVectorScaleY), @selector(preExposure),
      @selector(reset),             @selector(isDepthReversed),
  };
  for (SEL input : kInputs) {
    if (!class_respondsToSelector(scaler_class, input) &&
        class_getInstanceMethod(scaler_class, input) == nullptr) {
      return false;
    }
  }
  return true;
}

void DirectEncode(id scaler, SEL selector,
                  id<MTLCommandBuffer> command_buffer) {
  // This is the per-frame boundary, and it is called on the game's render thread, which never
  // drains an autorelease pool of its own. Without this pool every frame's temporaries - the three
  // mutableCopy argument blobs, the compute encoders, the per-frame resource arrays - accumulate
  // for the life of the process. That showed up as a monotonic frame-time decay in the live log
  // (48 -> 34 fps over 3500 frames, with the spike rate climbing 6 -> 89 per 120-frame window).
  @autoreleasepool {
    IMP original = FindOriginalEncode([(NSObject*)scaler class]);
    if (original == nullptr) {
      Log("event=encode_failed reason=original_missing scaler=%p class=%s",
          (__bridge void*)scaler, class_getName([(NSObject*)scaler class]));
      return;
    }
    // The install gate should make this impossible. It is here anyway because being wrong costs
    // the user their game: a subclass registered after the scan inherits this IMP, and the only
    // thing worse than not replacing a scaler is aborting the process that owns it. Forward and
    // let MetalFX do its job.
    if (!RespondsToTemporalEncodeInputs([(NSObject*)scaler class])) {
      reinterpret_cast<Encode>(original)(scaler, selector, command_buffer);
      return;
    }
    LogEncode((id<MTLFXTemporalScaler>)scaler, command_buffer);
    if (TryEncodeExactSR((id<MTLFXTemporalScaler>)scaler, command_buffer)) return;
    reinterpret_cast<Encode>(original)(scaler, selector, command_buffer);
  }
}

bool InstallEncodeHook(Class scaler_class) {
  if (scaler_class == Nil) return false;
  const char* name = class_getName(scaler_class);
  if (name == nullptr || std::strstr(name, "NeuralStudio") != nullptr ||
      (std::strstr(name, "MTLFX") == nullptr &&
       std::strstr(name, "MetalFX") == nullptr &&
       std::strstr(name, "MFX") == nullptr &&
       std::strstr(name, "M4FX") == nullptr &&
       std::strstr(name, "TemporalScal") == nullptr)) {
    return false;
  }
  // Named like a MetalFX effect, but not a temporal scaler: a spatial scaler, a frame
  // interpolator, something a future MetalFX adds. Leave it completely alone - an unhooked class
  // cannot be broken by us - and say so, because "the game uses MetalFX spatial" is the answer to
  // "why did Netal never engage?" and the log is where that answer has to live.
  if (!RespondsToTemporalEncodeInputs(scaler_class)) {
    Log("event=encode_hook_declined class=%s reason=not_a_temporal_scaler", name);
    return false;
  }
  // MetalFX's frame interpolator classes answer the temporal questions too (shared private base),
  // and the class scan used to install this SR hook on them. An interpolator encode through the SR
  // path would read the SR contract off an object that has none; it belongs to the FG hook.
  if (RespondsToFrameInterpolatorInputs(scaler_class)) {
    Log("event=encode_hook_declined class=%s reason=frame_interpolator", name);
    return false;
  }
  SEL selector = @selector(encodeToCommandBuffer:);
  Method method = class_getInstanceMethod(scaler_class, selector);
  if (method == nullptr) return false;

  std::lock_guard<std::mutex> lock(g_hook_mutex);
  for (const EncodeHook& hook : g_encode_hooks) {
    if (hook.scaler_class == scaler_class) return true;
  }
  IMP original = method_getImplementation(method);
  const char* types = method_getTypeEncoding(method);
  if (!class_addMethod(scaler_class, selector,
                       reinterpret_cast<IMP>(&DirectEncode), types)) {
    Method own_method = class_getInstanceMethod(scaler_class, selector);
    original = method_setImplementation(
        own_method, reinterpret_cast<IMP>(&DirectEncode));
  }
  g_encode_hooks.push_back({scaler_class, original});
  Log("event=encode_hook class=%s original=%p", name,
      reinterpret_cast<void*>(original));
  return true;
}

unsigned int InstallLoadedEncodeHooks() {
  int class_count = objc_getClassList(nullptr, 0);
  if (class_count <= 0) return 0;
  Class __unsafe_unretained* classes =
      static_cast<Class __unsafe_unretained*>(
          std::calloc(static_cast<size_t>(class_count), sizeof(Class)));
  if (classes == nullptr) return 0;
  class_count = objc_getClassList(classes, class_count);
  unsigned int installed = 0;
  for (int index = 0; index < class_count; ++index) {
    if (InstallEncodeHook(classes[index])) ++installed;
  }
  std::free(classes);
  return installed;
}

id NewTemporalScaler(id descriptor, SEL selector,
                     id<MTLDevice> device) NS_RETURNS_RETAINED;

id NewTemporalScaler(id descriptor, SEL selector, id<MTLDevice> device) {
  if (g_original_new_scaler == nullptr) return nil;
  id<MTLFXTemporalScaler> target =
      reinterpret_cast<NewScaler>(g_original_new_scaler)(descriptor, selector,
                                                          device);
  if (target == nil) {
    Log("event=scaler_create_failed device=%p", (__bridge void*)device);
    return nil;
  }
  const unsigned long long count = ++g_create_count;
  // The descriptor states the upscaling contract directly: render size, output size and the four
  // texture formats. Games recreate the scaler whenever graphics settings change, so this is where
  // the current configuration is legible - encode logging is throttled and cannot be relied on for
  // it.
  MTLFXTemporalScalerDescriptor* d = (MTLFXTemporalScalerDescriptor*)descriptor;
  if ([d isKindOfClass:[MTLFXTemporalScalerDescriptor class]]) {
    const NSUInteger iw = d.inputWidth, ih = d.inputHeight;
    const NSUInteger ow = d.outputWidth, oh = d.outputHeight;
    Log("event=scaler_create count=%llu device=%p target=%p target_class=%s "
        "render=%lux%lu output=%lux%lu scale=%.3f color_format=%lu depth_format=%lu "
        "motion_format=%lu output_format=%lu auto_exposure=%d",
        count, (__bridge void*)device, (__bridge void*)target, class_getName([(NSObject*)target class]),
        (unsigned long)iw, (unsigned long)ih, (unsigned long)ow, (unsigned long)oh,
        ih == 0 ? 0.0 : static_cast<double>(oh) / static_cast<double>(ih),
        (unsigned long)d.colorTextureFormat, (unsigned long)d.depthTextureFormat,
        (unsigned long)d.motionTextureFormat, (unsigned long)d.outputTextureFormat,
        d.isAutoExposureEnabled ? 1 : 0);
  } else {
    Log("event=scaler_create count=%llu device=%p target=%p target_class=%s",
        count, (__bridge void*)device, (__bridge void*)target,
        class_getName([(NSObject*)target class]));
  }
  if (InstallEncodeHook([(NSObject*)target class])) return target;
  return [NeuralStudioTemporalScalerProxy proxyWithTarget:target];
}

// ---- Metal 4 commit hook ---------------------------------------------------------------------
//
// MTL4CommandBuffer has no addCompletedHandler:, carries no reference to its queue, and the game
// owns the commit, so under GPTK 4 the runtime never learns that a frame finished. Four things
// depend on that callback and are therefore missing: GPU time per frame, command-buffer error
// reporting, releasing the frame ring, and any per-frame readback (which is why frame dumps are
// Metal 3 only).
//
// MTL4CommitOptions/addFeedbackHandler: supplies exactly what is absent - MTL4CommitFeedback
// carries `error`, `GPUStartTime` and `GPUEndTime` - but the only place to attach one is the
// queue's commit. So hook the two commit selectors on the concrete queue class, discovered at
// runtime by creating our own queue and taking its class. The name (AGXG15XFamilyCommandQueue_
// mtlnext on an M3) is device-specific and must never be hardcoded.
//
// Staged on purpose: this is the first hook that sits in the game's own submission path rather
// than on a MetalFX effect.
//   observe   record which commit variant the game calls, and whether it reuses one
//             MTL4CommitOptions instance. Registers nothing; no behaviour change at all.
//   feedback  additionally attach a feedback handler.
// NEURALSTUDIO_SR_METAL4_COMMIT_HOOK selects the stage; unset leaves the hook uninstalled.

IMP g_original_commit = nullptr;          // commit:count:
IMP g_original_commit_options = nullptr;  // commit:count:options:
std::atomic<unsigned long long> g_commit_plain{0};
std::atomic<unsigned long long> g_commit_with_options{0};
// Distinct MTL4CommitOptions instances seen. The header warns the class is not thread-safe, and a
// game that reuses one instance would accumulate a handler per commit if we attached blindly - so
// stage 1 measures this before stage 2 relies on it.
std::mutex g_commit_options_mutex;
std::vector<void*> g_commit_options_seen;

using CommitPlain = void (*)(id, SEL, const void*, NSUInteger);
using CommitWithOptions = void (*)(id, SEL, const void*, NSUInteger, id);

// Throttled like the rest of the log: the first few, then every 600th (~10 s at 60 Hz).
bool ShouldLogCommit(unsigned long long n) { return n <= 4 || n % 600 == 0; }

void CommitHook(id queue, SEL selector, const void* buffers, NSUInteger count) {
  const unsigned long long n = g_commit_plain.fetch_add(1, std::memory_order_relaxed) + 1;
  if (ShouldLogCommit(n)) {
    Log("event=metal4_commit variant=plain n=%llu buffers=%lu", n, static_cast<unsigned long>(count));
  }
  if (g_original_commit != nullptr) {
    reinterpret_cast<CommitPlain>(g_original_commit)(queue, selector, buffers, count);
  }
}

void CommitOptionsHook(id queue, SEL selector, const void* buffers, NSUInteger count, id options) {
  const unsigned long long n = g_commit_with_options.fetch_add(1, std::memory_order_relaxed) + 1;
  bool first_sighting = false;
  size_t distinct = 0;
  {
    std::lock_guard<std::mutex> lock(g_commit_options_mutex);
    void* key = (__bridge void*)options;
    if (std::find(g_commit_options_seen.begin(), g_commit_options_seen.end(), key) ==
        g_commit_options_seen.end()) {
      first_sighting = true;
      if (g_commit_options_seen.size() < 64) g_commit_options_seen.push_back(key);
    }
    distinct = g_commit_options_seen.size();
  }
  if (ShouldLogCommit(n) || first_sighting) {
    Log("event=metal4_commit variant=options n=%llu buffers=%lu options=%p distinct_options=%zu new=%d",
        n, static_cast<unsigned long>(count), (__bridge void*)options, distinct, first_sighting ? 1 : 0);
  }
  if (g_original_commit_options != nullptr) {
    reinterpret_cast<CommitWithOptions>(g_original_commit_options)(queue, selector, buffers, count, options);
  }
}

void InstallMetal4CommitHook(id<MTLDevice> device) {
  if (g_commit_stage == CommitHookStage::off || device == nil) return;
  static std::once_flag once;
  std::call_once(once, [device] {
    if (![device respondsToSelector:@selector(newMTL4CommandQueue)]) {
      Log("event=metal4_commit_hook_failed reason=no_mtl4_queue_api");
      return;
    }
    // Our own queue exists only to name the class the game's queue will also be. It is released
    // straight away; hooking is per-class, not per-instance.
    id<MTL4CommandQueue> probe = [device newMTL4CommandQueue];
    if (probe == nil) {
      Log("event=metal4_commit_hook_failed reason=queue_creation_returned_nil");
      return;
    }
    Class queue_class = object_getClass(probe);
    if (queue_class == Nil) {
      Log("event=metal4_commit_hook_failed reason=no_class");
      return;
    }
    unsigned installed = 0;
    struct { SEL selector; IMP replacement; IMP* original; const char* name; } targets[] = {
      {@selector(commit:count:), reinterpret_cast<IMP>(&CommitHook), &g_original_commit, "commit:count:"},
      {@selector(commit:count:options:), reinterpret_cast<IMP>(&CommitOptionsHook),
       &g_original_commit_options, "commit:count:options:"},
    };
    for (auto& target : targets) {
      Method method = class_getInstanceMethod(queue_class, target.selector);
      if (method == nullptr) {
        Log("event=metal4_commit_hook_missing selector=%s class=%s", target.name, class_getName(queue_class));
        continue;
      }
      *target.original = method_setImplementation(method, target.replacement);
      ++installed;
    }
    Log("event=metal4_commit_hook class=%s stage=%s installed=%u",
        class_getName(queue_class),
        g_commit_stage == CommitHookStage::observe ? "observe" : "feedback", installed);
  });
}

typedef id (*NewScalerM4)(id, SEL, id<MTLDevice>, id);
id NewTemporalScalerM4(id descriptor, SEL selector, id<MTLDevice> device, id compiler) NS_RETURNS_RETAINED;
id NewTemporalScalerM4(id descriptor, SEL selector, id<MTLDevice> device, id compiler) {
  if (g_original_new_scaler_m4 == nullptr) return nil;
  id target = reinterpret_cast<NewScalerM4>(g_original_new_scaler_m4)(descriptor, selector, device, compiler);
  InstallMetal4CommitHook(device);
  Log("event=scaler_create_metal4 device=%p compiler=%p target=%p target_class=%s", (__bridge void*)device,
      (__bridge void*)compiler, (__bridge void*)target, target == nil ? "nil" : class_getName([(NSObject*)target class]));
  if (target != nil) {
    { std::lock_guard<std::mutex> lock(g_metal4_mutex); g_metal4_scalers.push_back((__bridge void*)target); }
    // Log the class's own selectors so the Metal 4 encode entry point is known.
    unsigned int method_count = 0;
    Method* methods = class_copyMethodList([(NSObject*)target class], &method_count);
    NSMutableString* names = [NSMutableString string];
    for (unsigned int i = 0; i < method_count && names.length < 1500; ++i) {
      const char* name = sel_getName(method_getName(methods[i]));
      if (std::strstr(name, "ncode") != nullptr || std::strstr(name, "fence") != nullptr || std::strstr(name, "Fence") != nullptr) {
        [names appendFormat:@"%s ", name];
      }
    }
    free(methods);
    Log("event=metal4_scaler_selectors class=%s count=%u encode_like=%s", class_getName([(NSObject*)target class]),
        method_count, names.UTF8String);
    const bool hooked = InstallEncodeHook([(NSObject*)target class]);  // observe encodes (passed through)
    Log("event=metal4_encode_hook class=%s installed=%d", class_getName([(NSObject*)target class]), hooked ? 1 : 0);
  }
  return target;
}
id NewSpatialScalerM4(id descriptor, SEL selector, id<MTLDevice> device, id compiler) NS_RETURNS_RETAINED;
id NewSpatialScalerM4(id descriptor, SEL selector, id<MTLDevice> device, id compiler) {
  if (g_original_new_spatial_m4 == nullptr) return nil;
  id target = reinterpret_cast<NewScalerM4>(g_original_new_spatial_m4)(descriptor, selector, device, compiler);
  Log("event=spatial_scaler_create_metal4 target=%p target_class=%s", (__bridge void*)target,
      target == nil ? "nil" : class_getName([(NSObject*)target class]));
  return target;
}
void InstallMetal4CreationHooks() {
  Class temporal = NSClassFromString(@"MTLFXTemporalScalerDescriptor");
  Class spatial = NSClassFromString(@"MTLFXSpatialScalerDescriptor");
  Method m = temporal == Nil ? nullptr : class_getInstanceMethod(temporal, NSSelectorFromString(@"newTemporalScalerWithDevice:compiler:"));
  if (m != nullptr) g_original_new_scaler_m4 = method_setImplementation(m, reinterpret_cast<IMP>(&NewTemporalScalerM4));
  Method ms = spatial == Nil ? nullptr : class_getInstanceMethod(spatial, NSSelectorFromString(@"newSpatialScalerWithDevice:compiler:"));
  if (ms != nullptr) g_original_new_spatial_m4 = method_setImplementation(ms, reinterpret_cast<IMP>(&NewSpatialScalerM4));
  Log("event=metal4_creation_hooks temporal=%d spatial=%d", m != nullptr ? 1 : 0, ms != nullptr ? 1 : 0);
}


// ---------------------------------------------------------------------------------------------
// Frame generation: observe-only hook on MetalFX frame interpolators.
//
// GPTK 4's D3DMetal maps DLSS-G (the DLSSG.* parameter block) onto a MetalFX frame interpolator,
// created through MTLFXFrameInterpolatorDescriptor (Metal 3: newFrameInterpolatorWithDevice:,
// Metal 4: newFrameInterpolatorWithDevice:compiler:). This hook owns that seam separately from the
// SR temporal-scaler hook: its own original IMPs, its own registry and mutex, nothing shared with
// the SR mutex or ring. It records the real contract (class, selector, texture sizes/formats,
// timing). The original interpolator provides fallback when Netal does not serve a frame.
// Netal uses a 640x360 internal network profile and a separate game-resolution adapter;
// the default game arithmetic is accelerated, with reference mode available for validation.
std::mutex g_fg_mutex;
std::vector<EncodeHook> g_fg_hooks;
IMP g_original_new_interpolator = nullptr;      // newFrameInterpolatorWithDevice:
IMP g_original_new_interpolator_m4 = nullptr;   // newFrameInterpolatorWithDevice:compiler:
std::atomic<unsigned long long> g_fg_create_count{0};
std::atomic<unsigned long long> g_fg_encode_count{0};
double g_fg_window_start_s = 0.0;
double g_fg_window_min_ms = 1e9, g_fg_window_max_ms = 0.0;
double g_fg_last_encode_s = 0.0;

// Serving state. Armed from a background thread (pipeline compilation and the model load stay off
// the render thread); served frames replace the MetalFX interpolation, every failure falls open to it.
std::atomic<bool> g_fg_armed{false};
__strong NeuralStudioDLSSFGSession* g_fg_session = nil;
std::atomic<unsigned long long> g_fg_served{0};
std::atomic<unsigned long long> g_fg_fallbacks{0};
std::atomic<unsigned long long> g_fg_game_resets{0};
bool g_fg_force_reset = true;
constexpr unsigned long long kFGFallbackDisarm = 240;   // consecutive failures before giving up
unsigned long long g_fg_consecutive_failures = 0;

// Returns true when Netal generated this frame; false leaves the interpolator to MetalFX.
void PollControlFile(bool apply_sr);
bool TryServeFG(id fi_object, id command_object) {
  if (!g_fg_armed.load(std::memory_order_acquire)) return false;
  PollControlFile(false);
  if (!g_fg_control_enabled.load()) { g_fg_force_reset = true; return false; }
  const bool metal4 = [command_object conformsToProtocol:@protocol(MTL4CommandBuffer)];
  if (!metal4 && ![command_object conformsToProtocol:@protocol(MTLCommandBuffer)]) return false;
  id<MTLFXFrameInterpolatorBase> fi = (id<MTLFXFrameInterpolatorBase>)fi_object;
  const bool game_reset = fi.shouldResetHistory;
  if (game_reset) ++g_fg_game_resets;
  id<MTLFence> fence = [fi_object respondsToSelector:@selector(fence)] ? [fi_object fence] : nil;
  NSError* error = nil;
  BOOL served = NO;
  @try {
    [g_fg_session setDumpUITexture:fi.uiTexture];
    if (!metal4) {
      served = [g_fg_session encodeToCommandBuffer:(id<MTLCommandBuffer>)command_object
                                       colorTexture:fi.colorTexture depthTexture:fi.depthTexture motionTexture:fi.motionTexture
                                      outputTexture:fi.outputTexture contentWidth:fi.outputWidth contentHeight:fi.outputHeight
                                        renderWidth:fi.inputWidth renderHeight:fi.inputHeight
                                 motionVectorScaleX:fi.motionVectorScaleX motionVectorScaleY:fi.motionVectorScaleY
                                              fence:fence reset:game_reset || g_fg_force_reset error:&error];
    } else
    served = [g_fg_session encodeToMTL4CommandBuffer:(id<MTL4CommandBuffer>)command_object
                                         colorTexture:fi.colorTexture
                                         depthTexture:fi.depthTexture
                                        motionTexture:fi.motionTexture
                                        outputTexture:fi.outputTexture
                                         contentWidth:fi.outputWidth
                                        contentHeight:fi.outputHeight
                                          renderWidth:fi.inputWidth
                                         renderHeight:fi.inputHeight
                                   motionVectorScaleX:fi.motionVectorScaleX
                                   motionVectorScaleY:fi.motionVectorScaleY
                                                fence:fence
                                                reset:game_reset || g_fg_force_reset
                                                error:&error];
  } @catch (NSException* exception) {
    served = NO;
    Log("event=fg_exception reason=%s", exception.reason.UTF8String);
  }
  if (!served) {
    const unsigned long long fallbacks = ++g_fg_fallbacks;
    g_fg_force_reset = true;
    [g_fg_session invalidate];
    if (fallbacks <= 64 || fallbacks % 600 == 0)
      Log("event=fg_fallback frame=%llu fallbacks=%llu detail=%s", g_fg_served.load() + fallbacks, fallbacks,
          error == nil ? "unknown" : error.localizedDescription.UTF8String);
    if (++g_fg_consecutive_failures >= kFGFallbackDisarm) {
      g_fg_armed.store(false);
      Log("event=fg_disarmed reason=consecutive_fallbacks count=%llu", g_fg_consecutive_failures);
    }
    return false;
  }
  g_fg_consecutive_failures = 0;
  g_fg_force_reset = false;
  const unsigned long long frame = ++g_fg_served;
  if (frame <= 8 || frame % 120 == 0)
    Log("event=fg_served api=%s frame=%llu generated=%llu resets=%llu game_reset=%d fallbacks=%llu "
        "render=%lux%lu output=%lux%lu fullres=%d fullres_frames=%llu fullres_skipped=%llu",
        metal4 ? "metal4" : "metal3", frame, g_fg_session.generatedFrames, g_fg_session.resetFrames, game_reset ? 1 : 0, g_fg_fallbacks.load(),
        (unsigned long)fi.inputWidth, (unsigned long)fi.inputHeight, (unsigned long)fi.outputWidth,
        (unsigned long)fi.outputHeight, g_fg_session.fullResolutionEnabled ? 1 : 0,
        g_fg_session.fullResolutionFrames, g_fg_session.fullResolutionSkipped);
  return true;
}

void ArmFG() {
  const char* arm = std::getenv("NEURALSTUDIO_RE9_FG_ARM");
  const char* bundle_path = std::getenv("NEURALSTUDIO_RE9_FG_BUNDLE");
  if (arm == nullptr || std::strcmp(arm, "1") != 0 || bundle_path == nullptr || *bundle_path == '\0') return;
  std::string bundle_s(bundle_path);
  Log("event=fg_arm_begin bundle=%s", bundle_path);
  RunDetached(^{
    unsigned attempt = 0;
    for (;;) {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError* error = nil;
      NeuralStudioDLSSFGSession* session = [[NeuralStudioDLSSFGSession alloc]
          initWithDevice:device runtimeBundleURL:[NSURL fileURLWithPath:@(bundle_s.c_str())] error:&error];
      if (session != nil) {
        g_fg_session = session;
        std::atomic_thread_fence(std::memory_order_release);
        g_fg_armed.store(true, std::memory_order_release);
        Log("event=fg_arm_ready bundle=%s attempts=%u profile=640x360 lowering=%s fullres=%d", bundle_s.c_str(), attempt + 1,
            session.exactLowering ? "exact" : "hardware", session.fullResolutionEnabled ? 1 : 0);
        return;
      }
      if (attempt == 0 || attempt % 10 == 0)
        Log("event=fg_arm_failed attempt=%u retrying=1 detail=%s", attempt + 1,
            error == nil ? "unknown" : error.localizedDescription.UTF8String);
      ++attempt;
      sleep(attempt < 4 ? (2u << attempt) : 15u);
    }
  });
}

// A class is a frame interpolator when it answers the interpolator's own questions - the previous
// colour, the UI layer and the frame delta - not when its name contains "Interpolator".
bool RespondsToFrameInterpolatorInputs(Class effect_class) {
  static const SEL kInputs[] = {
      @selector(colorTexture),     @selector(prevColorTexture), @selector(depthTexture),
      @selector(motionTexture),    @selector(uiTexture),        @selector(outputTexture),
      @selector(deltaTime),        @selector(motionVectorScaleX), @selector(motionVectorScaleY),
      @selector(shouldResetHistory), @selector(inputWidth),     @selector(outputWidth),
  };
  for (SEL input : kInputs) {
    if (!class_respondsToSelector(effect_class, input) &&
        class_getInstanceMethod(effect_class, input) == nullptr) {
      return false;
    }
  }
  return true;
}

IMP FindOriginalFGEncode(Class effect_class) {
  std::lock_guard<std::mutex> lock(g_fg_mutex);
  for (Class current = effect_class; current != Nil; current = class_getSuperclass(current)) {
    for (const EncodeHook& hook : g_fg_hooks) {
      if (hook.scaler_class == current) return hook.original;
    }
  }
  return nullptr;
}

double FGNowSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void LogFGEncode(id<MTLFXFrameInterpolatorBase> fi, id command_buffer, unsigned long long count) {
  auto tex = [](id<MTLTexture> t, char* out, size_t n) {
    if (t == nil) { std::snprintf(out, n, "nil"); return; }
    std::snprintf(out, n, "%p:%lux%lu:f%lu:s%lu:u%lu", (__bridge void*)t,
                  static_cast<unsigned long>(t.width), static_cast<unsigned long>(t.height),
                  static_cast<unsigned long>(t.pixelFormat), static_cast<unsigned long>(t.storageMode),
                  static_cast<unsigned long>(t.usage));
  };
  char color[96], prev[96], depth[96], motion[96], ui[96], output[96], distortion[96];
  tex(fi.colorTexture, color, sizeof color);
  tex(fi.prevColorTexture, prev, sizeof prev);
  tex(fi.depthTexture, depth, sizeof depth);
  tex(fi.motionTexture, motion, sizeof motion);
  tex(fi.uiTexture, ui, sizeof ui);
  tex(fi.outputTexture, output, sizeof output);
  id<MTLTexture> dist = nil;
  unsigned long content_w = 0, content_h = 0;
  if ([(NSObject*)fi respondsToSelector:@selector(distortionTexture)]) dist = [(NSObject*)fi valueForKey:@"distortionTexture"];
  if ([(NSObject*)fi respondsToSelector:@selector(contentWidth)]) {
    content_w = [[(NSObject*)fi valueForKey:@"contentWidth"] unsignedLongValue];
    content_h = [[(NSObject*)fi valueForKey:@"contentHeight"] unsignedLongValue];
  }
  tex(dist, distortion, sizeof distortion);
  Log("event=fg_encode count=%llu fi=%p fi_class=%s command=%p command_class=%s "
      "input=%lux%lu output_decl=%lux%lu content=%lux%lu color=%s prev_color=%s depth=%s motion=%s ui=%s "
      "output=%s distortion=%s delta_time=%.6g motion_scale=%.9g,%.9g jitter=%.9g,%.9g "
      "near=%.9g far=%.9g fov=%.9g aspect=%.9g ui_composited=%d reset=%d",
      count, (__bridge void*)fi, class_getName([(NSObject*)fi class]), (__bridge void*)command_buffer,
      command_buffer == nil ? "nil" : class_getName([(NSObject*)command_buffer class]),
      static_cast<unsigned long>(fi.inputWidth), static_cast<unsigned long>(fi.inputHeight),
      static_cast<unsigned long>(fi.outputWidth), static_cast<unsigned long>(fi.outputHeight),
      content_w, content_h, color, prev, depth, motion, ui, output, distortion,
      static_cast<double>(fi.deltaTime), static_cast<double>(fi.motionVectorScaleX),
      static_cast<double>(fi.motionVectorScaleY), static_cast<double>(fi.jitterOffsetX),
      static_cast<double>(fi.jitterOffsetY), static_cast<double>(fi.nearPlane),
      static_cast<double>(fi.farPlane), static_cast<double>(fi.fieldOfView),
      static_cast<double>(fi.aspectRatio), fi.isUITextureComposited ? 1 : 0,
      fi.shouldResetHistory ? 1 : 0);
}

// Same IMP for Metal 3 (id<MTLCommandBuffer>) and Metal 4 (id<MTL4CommandBuffer>) encodes: the
// selector name is encodeToCommandBuffer: on both classes and the argument is an object either way.
void FGDirectEncode(id fi, SEL selector, id command_buffer) {
  @autoreleasepool {
    IMP original = FindOriginalFGEncode([(NSObject*)fi class]);
    if (original == nullptr) {
      Log("event=fg_encode_failed reason=original_missing fi=%p class=%s", (__bridge void*)fi,
          class_getName([(NSObject*)fi class]));
      return;
    }
    const unsigned long long count = ++g_fg_encode_count;
    const double now = FGNowSeconds();
    if (count <= 8 || count % 600 == 0) {
      @try {
        LogFGEncode((id<MTLFXFrameInterpolatorBase>)fi, command_buffer, count);
      } @catch (NSException* exception) {
        Log("event=fg_encode_log_failed count=%llu reason=%s", count, exception.reason.UTF8String);
      }
    }
    if (g_fg_last_encode_s != 0.0) {
      const double gap_ms = (now - g_fg_last_encode_s) * 1000.0;
      g_fg_window_min_ms = std::min(g_fg_window_min_ms, gap_ms);
      g_fg_window_max_ms = std::max(g_fg_window_max_ms, gap_ms);
    }
    g_fg_last_encode_s = now;
    if (g_fg_window_start_s == 0.0) g_fg_window_start_s = now;
    if (count % 60 == 0) {
      const double window_s = now - g_fg_window_start_s;
      // One encode per real source frame; presented frames are twice that when the game shows every
      // generated frame (pinned model: 2x only).
      Log("event=fg_pacing count=%llu window=60 mean_ms=%.2f source_fps=%.1f min_ms=%.2f max_ms=%.2f",
          count, window_s * 1000.0 / 60.0, window_s > 0 ? 60.0 / window_s : 0.0,
          g_fg_window_min_ms, g_fg_window_max_ms);
      g_fg_window_start_s = now;
      g_fg_window_min_ms = 1e9;
      g_fg_window_max_ms = 0.0;
    }
    if (TryServeFG(fi, command_buffer)) return;
    reinterpret_cast<void (*)(id, SEL, id)>(original)(fi, selector, command_buffer);
  }
}

bool InstallFGEncodeHook(Class effect_class) {
  if (effect_class == Nil) return false;
  const char* name = class_getName(effect_class);
  if (name == nullptr || std::strstr(name, "NeuralStudio") != nullptr) return false;
  if (!RespondsToFrameInterpolatorInputs(effect_class)) {
    Log("event=fg_encode_hook_declined class=%s reason=not_a_frame_interpolator", name);
    return false;
  }
  SEL selector = @selector(encodeToCommandBuffer:);
  Method method = class_getInstanceMethod(effect_class, selector);
  if (method == nullptr) {
    Log("event=fg_encode_hook_declined class=%s reason=no_encode_selector", name);
    return false;
  }
  std::lock_guard<std::mutex> lock(g_fg_mutex);
  for (const EncodeHook& hook : g_fg_hooks) {
    if (hook.scaler_class == effect_class) return true;
  }
  IMP original = method_getImplementation(method);
  const char* types = method_getTypeEncoding(method);
  if (!class_addMethod(effect_class, selector, reinterpret_cast<IMP>(&FGDirectEncode), types)) {
    Method own_method = class_getInstanceMethod(effect_class, selector);
    original = method_setImplementation(own_method, reinterpret_cast<IMP>(&FGDirectEncode));
  }
  g_fg_hooks.push_back({effect_class, original});
  Log("event=fg_encode_hook class=%s original=%p types=%s", name, reinterpret_cast<void*>(original),
      types == nullptr ? "-" : types);
  return true;
}

void LogFGCreate(const char* api, id descriptor, id<MTLDevice> device, id target, unsigned long long count) {
  MTLFXFrameInterpolatorDescriptor* d = (MTLFXFrameInterpolatorDescriptor*)descriptor;
  if ([d isKindOfClass:[MTLFXFrameInterpolatorDescriptor class]]) {
    id scaler = d.scaler;
    Log("event=fg_interpolator_create api=%s count=%llu device=%p target=%p target_class=%s "
        "input=%lux%lu output=%lux%lu color_format=%lu depth_format=%lu motion_format=%lu "
        "ui_format=%lu output_format=%lu scaler=%p scaler_class=%s",
        api, count, (__bridge void*)device, (__bridge void*)target,
        target == nil ? "nil" : class_getName([(NSObject*)target class]),
        (unsigned long)d.inputWidth, (unsigned long)d.inputHeight, (unsigned long)d.outputWidth,
        (unsigned long)d.outputHeight, (unsigned long)d.colorTextureFormat,
        (unsigned long)d.depthTextureFormat, (unsigned long)d.motionTextureFormat,
        (unsigned long)d.uiTextureFormat, (unsigned long)d.outputTextureFormat, (__bridge void*)scaler,
        scaler == nil ? "nil" : class_getName([(NSObject*)scaler class]));
  } else {
    Log("event=fg_interpolator_create api=%s count=%llu device=%p target=%p target_class=%s "
        "descriptor_class=%s", api, count, (__bridge void*)device, (__bridge void*)target,
        target == nil ? "nil" : class_getName([(NSObject*)target class]),
        descriptor == nil ? "nil" : class_getName([(NSObject*)descriptor class]));
  }
  if (target == nil) return;
  // The class's own selectors, so the encode entry point and any fence/reset methods are known.
  unsigned int method_count = 0;
  Method* methods = class_copyMethodList([(NSObject*)target class], &method_count);
  NSMutableString* names = [NSMutableString string];
  for (unsigned int i = 0; i < method_count && names.length < 1500; ++i) {
    [names appendFormat:@"%s ", sel_getName(method_getName(methods[i]))];
  }
  free(methods);
  Log("event=fg_interpolator_selectors class=%s count=%u names=%s",
      class_getName([(NSObject*)target class]), method_count, names.UTF8String);
  const bool hooked = InstallFGEncodeHook([(NSObject*)target class]);
  Log("event=fg_encode_hook_install class=%s installed=%d", class_getName([(NSObject*)target class]),
      hooked ? 1 : 0);
}

typedef id (*NewInterpolator)(id, SEL, id<MTLDevice>);
typedef id (*NewInterpolatorM4)(id, SEL, id<MTLDevice>, id);
id NewFrameInterpolator(id descriptor, SEL selector, id<MTLDevice> device) NS_RETURNS_RETAINED;
id NewFrameInterpolator(id descriptor, SEL selector, id<MTLDevice> device) {
  if (g_original_new_interpolator == nullptr) return nil;
  id target = reinterpret_cast<NewInterpolator>(g_original_new_interpolator)(descriptor, selector, device);
  LogFGCreate("metal3", descriptor, device, target, ++g_fg_create_count);
  return target;
}
id NewFrameInterpolatorM4(id descriptor, SEL selector, id<MTLDevice> device, id compiler) NS_RETURNS_RETAINED;
id NewFrameInterpolatorM4(id descriptor, SEL selector, id<MTLDevice> device, id compiler) {
  if (g_original_new_interpolator_m4 == nullptr) return nil;
  id target = reinterpret_cast<NewInterpolatorM4>(g_original_new_interpolator_m4)(descriptor, selector, device, compiler);
  InstallMetal4CommitHook(device);
  LogFGCreate("metal4", descriptor, device, target, ++g_fg_create_count);
  return target;
}

void InstallFrameInterpolatorCreationHooks() {
  Class descriptor = NSClassFromString(@"MTLFXFrameInterpolatorDescriptor");
  Method m3 = descriptor == Nil ? nullptr
      : class_getInstanceMethod(descriptor, NSSelectorFromString(@"newFrameInterpolatorWithDevice:"));
  if (m3 != nullptr) g_original_new_interpolator = method_setImplementation(m3, reinterpret_cast<IMP>(&NewFrameInterpolator));
  Method m4 = descriptor == Nil ? nullptr
      : class_getInstanceMethod(descriptor, NSSelectorFromString(@"newFrameInterpolatorWithDevice:compiler:"));
  if (m4 != nullptr) g_original_new_interpolator_m4 = method_setImplementation(m4, reinterpret_cast<IMP>(&NewFrameInterpolatorM4));
  // Interpolators created before injection (the game reached gameplay inside the settle window)
  // never pass through the creation hook, so hook every loaded interpolator class now; instances
  // that already exist are served from their next encode.
  unsigned int hooked = 0;
  int class_count = objc_getClassList(nullptr, 0);
  if (class_count > 0) {
    Class __unsafe_unretained* classes = static_cast<Class __unsafe_unretained*>(
        std::calloc(static_cast<size_t>(class_count), sizeof(Class)));
    if (classes != nullptr) {
      class_count = objc_getClassList(classes, class_count);
      for (int index = 0; index < class_count; ++index) {
        const char* name = class_getName(classes[index]);
        if (name == nullptr || (std::strstr(name, "FX") == nullptr) || std::strstr(name, "Descriptor") != nullptr) continue;
        if (!RespondsToFrameInterpolatorInputs(classes[index])) continue;
        if (InstallFGEncodeHook(classes[index])) ++hooked;
      }
      std::free(classes);
    }
  }
  Log("event=fg_creation_hooks metal3=%d metal4=%d loaded_interpolator_classes_hooked=%u",
      m3 != nullptr ? 1 : 0, m4 != nullptr ? 1 : 0, hooked);
}

__attribute__((constructor)) void InitializeBridge() {
  @autoreleasepool {
    if (std::getenv("NEURALSTUDIO_RE9_METALFX_NOOP") == nullptr ||
        !IsTargetProcess()) {
      return;
    }
    const char* path = std::getenv("NEURALSTUDIO_RE9_METALFX_LOG");
    if (path != nullptr && *path != '\0') {
      std::string directory(path);
      const size_t slash = directory.rfind('/');
      const std::string log_dir = slash == std::string::npos ? std::string(".") : directory.substr(0, slash);
      g_sr_pause_marker = log_dir + "/re9-sr-pause";
      g_sr_control_path = log_dir + "/netal-control";
      g_sr_mark_marker = log_dir + "/re9-sr-mark";
    }
    if (path == nullptr || *path == '\0') return;
    g_log = std::fopen(path, "a");
    if (g_log == nullptr) return;
    Class descriptor = NSClassFromString(@"MTLFXTemporalScalerDescriptor");
    Method method = class_getInstanceMethod(
        descriptor, @selector(newTemporalScalerWithDevice:));
    if (descriptor == Nil || method == nullptr) {
      Log("event=bridge_failed reason=metalfx_descriptor_method_missing");
      return;
    }
    g_original_new_scaler = method_getImplementation(method);
    method_setImplementation(method, reinterpret_cast<IMP>(&NewTemporalScaler));
    unsigned int encode_hooks = InstallLoadedEncodeHooks();
    InstallMetal4CreationHooks();
    InstallFrameInterpolatorCreationHooks();
    ArmFG();
    ReadLiveDiagnosticsEnvironment();
    InstallLayerHook();
    const char* arm = std::getenv("NEURALSTUDIO_RE9_SR_ARM");
    const char* bundle_path = std::getenv("NEURALSTUDIO_RE9_SR_BUNDLE");
    if (arm != nullptr && std::strcmp(arm, "1") == 0 &&
        bundle_path != nullptr && *bundle_path != '\0') {
      Log("event=sr_arm_begin bundle=%s suffix=%s", bundle_path,
          std::getenv("NEURALSTUDIO_SR_PWIN_SUFFIX") == nullptr ? "-" : std::getenv("NEURALSTUDIO_SR_PWIN_SUFFIX"));
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSURL* bundle = [NSURL fileURLWithPath:
          [NSString stringWithUTF8String:bundle_path]];
      g_sr_device = device;
      g_sr_bundle_url = bundle;
      const char* sustained_env = std::getenv("NEURALSTUDIO_RE9_SR_SUSTAINED");
      const bool sustained = sustained_env != nullptr && std::strcmp(sustained_env, "1") == 0;
      std::string bundle_s(bundle_path);
      g_sr_init_pending = true;
      // Arming compiles the session's pipelines, and that can queue behind a game compiling its
      // own: Death Stranding held the Metal compiler for minutes at startup, during which the log
      // showed sr_arm_begin and then nothing at all. A silent gap is indistinguishable from a hang,
      // so say so periodically rather than leaving the reader to sample the process to find out.
      RunDetached(^{
        const auto arm_start = std::chrono::steady_clock::now();
        unsigned waits = 0;
        while (g_sr_init_pending) {
          usleep(500 * 1000);
          if (!g_sr_init_pending) break;
          if (++waits % 30 == 0) {  // every 15s
            Log("event=sr_arm_waiting elapsed_s=%.0f detail=compiling pipelines, likely queued "
                "behind the game's own shader compilation",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - arm_start).count());
          }
        }
      });
      if (!RunDetached(^{
        // Keep trying, rather than failing for the life of the process.
        //
        // Arming can fail for reasons that stop being true a moment later: the model has not been
        // supplied yet, a bundle is still being staged, the Metal compiler is wedged behind a
        // game's own shader build. Giving up on the first attempt means the only cure is quitting
        // the game - so a model dropped in while the game runs is picked up here instead.
        //
        // Backs off 2, 4, 8, 15 seconds and holds, and logs only on a change of state plus an
        // occasional heartbeat, so a genuinely broken bundle does not fill the log.
        unsigned attempt = 0;
        for (;;) {
          {
            std::lock_guard<std::mutex> state_lock(g_sr_mutex);
            g_sr_init_pending = true;
          }
          NSError* session_error = nil;
          NeuralStudioDLSSSRSession* session = [[NeuralStudioDLSSSRSession alloc]
              initWithDevice:g_sr_device runtimeBundleURL:g_sr_bundle_url error:&session_error];
          NeuralStudioDLSSTextureAdapter* adapter =
              session == nil ? nil
                             : [[NeuralStudioDLSSTextureAdapter alloc] initWithDevice:g_sr_device
                                                                             session:session
                                                                               error:&session_error];
          if (adapter != nil) {
            std::lock_guard<std::mutex> publish_lock(g_sr_mutex);
            g_sr_session = session;
            g_sr_adapter = adapter;
            // Batched-pipeline coverage, into the log the app follows. The session reports this on
            // stderr, which nothing reads: a profile missing layers is a ~2x slowdown that looks
            // like a healthy session, so it has to be visible where someone is actually looking.
            // total==0 means the batched pipelines are switched off entirely, which is a choice rather
      // than a half-generated profile; logging it would only be noise the app has to filter.
      if (session.pipelineLayersTotal > 0)
      Log("event=sr_pipeline_layers present=%lu total=%lu missing=%s",
                static_cast<unsigned long>(session.pipelineLayersPresent),
                static_cast<unsigned long>(session.pipelineLayersTotal),
                session.pipelineLayersMissing.length > 0 ? session.pipelineLayersMissing.UTF8String : "-");

            g_sr_sustained = sustained;
            std::atomic_thread_fence(std::memory_order_release);
            g_sr_armed = true;
            Log("event=sr_arm_ready bundle=%s one_frame=%s sustained=%s attempts=%u", bundle_s.c_str(),
                sustained ? "false" : "true", sustained ? "true" : "false", attempt + 1);
            g_sr_init_pending = false;
            break;
          } else {
            // First failure is reported in full; after that only every tenth, so a bundle that
            // will never load does not drown the log while the retry stands by.
            if (attempt == 0 || attempt % 10 == 0) {
              Log("event=sr_arm_failed attempt=%u retrying=1 detail=%s", attempt + 1,
                  session_error == nil ? "unknown"
                                       : session_error.localizedDescription.UTF8String);
            }
            // The watchdog above reports a slow *compile*; once arming has actually failed that
            // message would be wrong, and the retry logs carry the story from here.
            {
              std::lock_guard<std::mutex> state_lock(g_sr_mutex);
              g_sr_init_pending = false;
            }
          }
          ++attempt;
          const useconds_t backoff = attempt == 1 ? 2 : attempt == 2 ? 4 : attempt == 3 ? 8 : 15;
          usleep(backoff * 1000 * 1000);
        }
      })) {
        g_sr_init_pending = false;
        Log("event=sr_arm_failed retrying=0 detail=thread_start_failed");
      }
    }
    Log("event=bridge_ready process=%s pid=%d descriptor=%s encode_hooks=%u",
        NSProcessInfo.processInfo.processName.UTF8String, static_cast<int>(getpid()),
        class_getName(descriptor), encode_hooks);
  }
}

}  // namespace
