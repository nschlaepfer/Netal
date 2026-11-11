// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) Netal Contributors.
//
// Opt-in continuous in-process DLSS 5 Neural Rendering bridge for Apple Silicon.
// Supports Metal 3 MTLFXTemporalScaler, optional MTLFXSpatialScaler, and optional
// CAMetalLayer + Metal 4 MTL4CommandQueue signalDrawable: presentation seams,
// in both inline synchronous and non-blocking 60 Hz async delta-reuse modes.

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include "../nr_fast/nr_fast_runtime.h"
#include "netal_nr_continuous_state.h"
#include "netal_nr_optical_flow.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace netal_nr_adapter;
using EncodeM3Fn = void (*)(id, SEL, id<MTLCommandBuffer>);
using CreateTemporalFn = id (*)(id, SEL, id<MTLDevice>);
using CreateSpatialFn = id (*)(id, SEL, id<MTLDevice>);
using SetDeviceFn = void (*)(id, SEL, id<MTLDevice>);
using SignalDrawableFn = void (*)(id, SEL, id<MTLDrawable>);

enum HookBits : uint32_t {
  kHookTemporalScaler = 1u << 0,
  kHookSpatialScaler = 1u << 1,
  kHookLayerSetDevice = 1u << 2,
  kHookMtl4SignalDrawable = 1u << 3,
};

enum class SeamKind : uint32_t {
  Temporal = 0,
  Spatial = 1,
  Drawable = 2,
};

enum class PrepStatus : uint32_t {
  Cold = 0,
  Preparing = 1,
  Ready = 2,
  Failed = 3,
};

const char* SeamKindName(SeamKind s) {
  switch (s) {
    case SeamKind::Temporal: return "temporal";
    case SeamKind::Spatial: return "spatial";
    case SeamKind::Drawable: return "drawable";
  }
  return "unknown";
}

netal_nr_fast::ControlProfile ToFastProfile(NrProfile p) {
  switch (p) {
    case NrProfile::Standard: return netal_nr_fast::ControlProfile::Standard;
    case NrProfile::Natural: return netal_nr_fast::ControlProfile::Natural;
    case NrProfile::Cinematic: return netal_nr_fast::ControlProfile::Cinematic;
    case NrProfile::Neutral: return netal_nr_fast::ControlProfile::Neutral;
  }
  return netal_nr_fast::ControlProfile::Standard;
}

void BridgeSelfAnchor() {}

std::string FindBridgeDirectory() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&BridgeSelfAnchor), &info) != 0 &&
      info.dli_fname != nullptr && info.dli_fname[0] != '\0') {
    std::string full(info.dli_fname);
    size_t slash = full.find_last_of('/');
    if (slash != std::string::npos) {
      return full.substr(0, slash);
    }
  }
  return {};
}

bool FileExistsRegular(const std::string& path) {
  if (path.empty()) return false;
  struct stat st {};
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Constant-initialized atomics/PODs safe before any C++ dynamic initialization.
constinit std::atomic<uint32_t> g_installed_hooks{0};
constinit char g_stream_id_assoc_key = 0;
constinit char g_seam_kind_assoc_key = 0;

struct ClassEncodeHook {
  Class cls;
  IMP original;
};

// Heap-allocated singleton avoids Mach-O __attribute__((constructor)) vs C++ global
// init-order fiasco AND __cxa_finalize destruction races against async Metal callbacks.
struct BridgeRuntime {
  ContinuousAdapterConfig cfg{};
  ContinuousAdapterState state{};

  std::mutex log_mutex;
  FILE* log_file = nullptr;
  std::string pause_marker_path;
  bool srgb_transfer_override = false;
  bool sync_prepare = false;

  std::atomic<uint64_t> next_stream_id{0};
  std::atomic<uint64_t> total_frames{0};
  std::atomic<uint64_t> fallback_logs{0};
  std::atomic<double> last_worker_gpu_ms{0.0};

  std::mutex hook_mutex;
  std::vector<ClassEncodeHook> encode_hooks;
  IMP orig_new_temporal = nullptr;
  IMP orig_new_spatial = nullptr;
  SetDeviceFn orig_layer_set_device = nullptr;
  SignalDrawableFn orig_q4_signal_drawable = nullptr;

  std::mutex prep_mutex;
  std::atomic<PrepStatus> prep_status{PrepStatus::Cold};
  id<MTLDevice> device = nil;
  std::unique_ptr<netal_nr_fast::Context> nr_ctx;
  std::unique_ptr<OpticalFlowSynthesizer> flow_synth;
  std::mutex nr_encode_mutex;

  std::mutex blit_mutex;
  id<MTLLibrary> blit_library = nil;
  id<MTLFunction> blit_vs = nil;
  id<MTLFunction> blit_fs = nil;
  id<MTLFunction> resize_fs = nil;
  id<MTLFunction> compose_baked_fs = nil;
  std::map<NSUInteger, id<MTLRenderPipelineState>> blit_psos;
  std::map<NSUInteger, id<MTLRenderPipelineState>> resize_psos;
  std::map<NSUInteger, id<MTLRenderPipelineState>> compose_baked_psos;

  std::mutex stage_mutex;
  id<MTLTexture> inline_src_copy_tex = nil;
  id<MTLTexture> inline_composed_tex = nil;

  std::mutex async_res_mutex;
  id<MTLCommandQueue> worker_queue = nil;
  id<MTLSharedEvent> worker_event = nil;
  std::atomic<uint64_t> worker_event_val{0};
  id<MTLTexture> worker_color_tex[ContinuousAdapterState::kNumSlots] = {nil, nil};
  id<MTLTexture> worker_motion_tex[ContinuousAdapterState::kNumSlots] = {nil, nil};
  id<MTLTexture> baked_delta_slot_tex[ContinuousAdapterState::kNumSlots] = {nil, nil};
  uint32_t delta_slot_w[ContinuousAdapterState::kNumSlots] = {0, 0};
  uint32_t delta_slot_h[ContinuousAdapterState::kNumSlots] = {0, 0};

  std::mutex drawable_mutex;
  id<MTLCommandQueue> drawable_queue = nil;
  id<MTLSharedEvent> drawable_sync_ev = nil;
  uint64_t drawable_ev_val = 0;
  id<MTLTexture> drawable_copy_tex = nil;

  std::string capture_pair_dir;
  std::vector<uint64_t> capture_frames;
  dispatch_queue_t capture_write_q = nil;
};

BridgeRuntime& RT() {
  static BridgeRuntime* rt = new BridgeRuntime();
  return *rt;
}

void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Log(const char* fmt, ...) {
  auto& rt = RT();
  std::lock_guard<std::mutex> lock(rt.log_mutex);
  FILE* out = rt.log_file ? rt.log_file : stderr;
  std::fprintf(out, "netal_nr_fast time=%.6f pid=%d ", CFAbsoluteTimeGetCurrent(), static_cast<int>(getpid()));
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(out, fmt, ap);
  va_end(ap);
  std::fputc('\n', out);
  std::fflush(out);
}

bool RunDetached(void (^block)(void)) {
  void (^copy)(void) = [block copy];
  pthread_t thread;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_setstacksize(&attributes, 16u << 20);
  pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  void* ctx = (void*)CFBridgingRetain(copy);
  int rc = pthread_create(
      &thread,
      &attributes,
      [](void* raw) -> void* {
        @autoreleasepool {
          void (^run)(void) = (void (^)(void))CFBridgingRelease(raw);
          run();
        }
        return nullptr;
      },
      ctx);
  pthread_attr_destroy(&attributes);
  if (rc != 0) {
    CFBridgingRelease(ctx);
    return false;
  }
  return true;
}

bool IsTargetProcess() {
  const char* env = std::getenv("NEURALSTUDIO_TARGET_EXE");
  if (!env || *env == '\0') {
    return true;
  }
  NSCharacterSet* ws = [NSCharacterSet whitespaceAndNewlineCharacterSet];
  NSMutableArray<NSString*>* targets = [NSMutableArray array];
  for (NSString* raw in [@(env) componentsSeparatedByString:@","]) {
    NSString* token = [raw stringByTrimmingCharactersInSet:ws].lowercaseString;
    if (token.length > 0) [targets addObject:token];
  }
  if (targets.count == 0) {
    return true;
  }
  NSString* process = NSProcessInfo.processInfo.processName.lowercaseString;
  NSArray<NSString*>* arguments = NSProcessInfo.processInfo.arguments;
  for (NSString* target in targets) {
    NSString* stem = target.stringByDeletingPathExtension;
    if ([process isEqualToString:target] || (stem.length > 0 && [process isEqualToString:stem])) {
      return true;
    }
    NSString* win_suffix = [@"\\" stringByAppendingString:target];
    NSString* posix_suffix = [@"/" stringByAppendingString:target];
    for (NSString* argument in arguments) {
      NSString* value = argument.lowercaseString;
      if ([value hasPrefix:@"-"]) continue;
      if ([value isEqualToString:target] ||
          [value hasSuffix:win_suffix] ||
          [value hasSuffix:posix_suffix]) {
        return true;
      }
    }
  }
  return false;
}

bool IsSRGBPixelFormat(MTLPixelFormat pf) {
  return pf == MTLPixelFormatRGBA8Unorm_sRGB || pf == MTLPixelFormatBGRA8Unorm_sRGB;
}

bool CanShaderWriteDirectly(id<MTLTexture> tex) {
  if (!tex) return false;
  if ((tex.usage & MTLTextureUsageShaderWrite) == 0) return false;
  if (IsSRGBPixelFormat(tex.pixelFormat)) return false;
  switch (tex.pixelFormat) {
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatRGBA32Float:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRG11B10Float:
      return true;
    default:
      return false;
  }
}

id<MTLTexture> EnsurePrivateTex(
    id<MTLDevice> dev,
    id<MTLTexture> existing,
    MTLPixelFormat pf,
    uint32_t w,
    uint32_t h,
    MTLTextureUsage usage,
    NSString* label) {
  if (existing && existing.pixelFormat == pf &&
      existing.width == w && existing.height == h &&
      (existing.usage & usage) == usage) {
    return existing;
  }
  MTLTextureDescriptor* td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = usage;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
  if (tex && label) tex.label = label;
  return tex;
}

void EnsureBridgeRenderFunctionsLocked(id<MTLDevice> dev) {
  auto& rt = RT();
  if (rt.blit_vs && rt.blit_fs && rt.resize_fs && rt.compose_baked_fs) return;
  if (!rt.blit_library) {
    NSString* fallback_src =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "inline float l2s(float c) { float x=clamp(c,0.0f,1.0f); return x<=0.0031308f?(x*12.92f):(1.055f*precise::pow(x,1.0f/2.4f)-0.055f); }\n"
         "inline float s2l(float c) { float x=clamp(c,0.0f,1.0f); return x<=0.04045f?(x/12.92f):precise::pow((x+0.055f)/1.055f,2.4f); }\n"
         "struct VOut { float4 pos [[position]]; };\n"
         "vertex VOut nr_bridge_blit_vs(uint vid [[vertex_id]]) {\n"
         "  VOut o; float2 uv = float2((vid << 1u) & 2u, vid & 2u);\n"
         "  o.pos = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);\n"
         "  return o;\n"
         "}\n"
         "fragment float4 nr_bridge_blit_fs(VOut in [[stage_in]], texture2d<float, access::read> t [[texture(0)]]) {\n"
         "  return float4(t.read(uint2(in.pos.xy)).rgb, 1.0f);\n"
         "}\n"
         "fragment float4 nr_bridge_resize_fs(VOut in [[stage_in]], texture2d<float, access::sample> t [[texture(0)]], constant float2& inv [[buffer(0)]]) {\n"
         "  constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);\n"
         "  return float4(t.sample(s, in.pos.xy * inv).rgb, 1.0f);\n"
         "}\n"
         "fragment float4 nr_bridge_compose_baked_fs(VOut in [[stage_in]], texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::sample> dtex [[texture(1)]], constant uint4& dims [[buffer(0)]]) {\n"
         "  constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);\n"
         "  float3 c = src.read(uint2(in.pos.xy)).rgb;\n"
         "  if (dims.z != 0u) c = float3(l2s(c.x), l2s(c.y), l2s(c.z));\n"
         "  float3 d = dtex.sample(s, in.pos.xy / float2(float(dims.x), float(dims.y))).rgb;\n"
         "  float3 o = clamp(c + d, 0.0f, 1.0f);\n"
         "  if (dims.z != 0u) o = float3(s2l(o.x), s2l(o.y), s2l(o.z));\n"
         "  return float4(o, 1.0f);\n"
         "}\n";
    NSError* cerr = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    rt.blit_library = [dev newLibraryWithSource:fallback_src options:opts error:&cerr];
  }
  if (rt.blit_library) {
    if (!rt.blit_vs) rt.blit_vs = [rt.blit_library newFunctionWithName:@"nr_bridge_blit_vs"];
    if (!rt.blit_fs) rt.blit_fs = [rt.blit_library newFunctionWithName:@"nr_bridge_blit_fs"];
    if (!rt.resize_fs) rt.resize_fs = [rt.blit_library newFunctionWithName:@"nr_bridge_resize_fs"];
    if (!rt.compose_baked_fs) rt.compose_baked_fs = [rt.blit_library newFunctionWithName:@"nr_bridge_compose_baked_fs"];
  }
}

id<MTLRenderPipelineState> GetOrBuildRenderHelperPSO(
    id<MTLDevice> dev,
    MTLPixelFormat pf,
    int kind) {
  auto& rt = RT();
  std::lock_guard<std::mutex> lock(rt.blit_mutex);
  auto& cache = (kind == 0) ? rt.blit_psos : (kind == 1 ? rt.resize_psos : rt.compose_baked_psos);
  auto it = cache.find(static_cast<NSUInteger>(pf));
  if (it != cache.end()) return it->second;

  EnsureBridgeRenderFunctionsLocked(dev);
  id<MTLFunction> fs = (kind == 0) ? rt.blit_fs : (kind == 1 ? rt.resize_fs : rt.compose_baked_fs);
  if (!rt.blit_vs || !fs) return nil;

  MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
  rpd.label = (kind == 0) ? @"netal.nr.fast.bridge.blit"
                          : (kind == 1 ? @"netal.nr.fast.bridge.resize" : @"netal.nr.fast.bridge.compose_baked");
  rpd.vertexFunction = rt.blit_vs;
  rpd.fragmentFunction = fs;
  rpd.colorAttachments[0].pixelFormat = pf;
  NSError* perr = nil;
  id<MTLRenderPipelineState> pso = [dev newRenderPipelineStateWithDescriptor:rpd error:&perr];
  if (pso) {
    cache[static_cast<NSUInteger>(pf)] = pso;
  } else {
    Log("event=blit_pso_failed kind=%d format=%lu error=%s",
        kind,
        static_cast<unsigned long>(pf),
        perr ? perr.localizedDescription.UTF8String : "unknown");
  }
  return pso;
}

id<MTLRenderPipelineState> GetOrBuildBlitPSO(id<MTLDevice> dev, MTLPixelFormat pf) {
  return GetOrBuildRenderHelperPSO(dev, pf, 0);
}

bool EncodeRenderResize(
    id<MTLDevice> dev,
    id<MTLCommandBuffer> cb,
    id<MTLTexture> src_tex,
    id<MTLTexture> dst_tex) {
  if (!src_tex || !dst_tex) return false;
  if (src_tex.width == dst_tex.width && src_tex.height == dst_tex.height &&
      src_tex.pixelFormat == dst_tex.pixelFormat) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return false;
    [blit copyFromTexture:src_tex
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(dst_tex.width, dst_tex.height, 1)
                toTexture:dst_tex
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return true;
  }
  if ((dst_tex.usage & MTLTextureUsageRenderTarget) == 0) return false;
  id<MTLRenderPipelineState> pso = GetOrBuildRenderHelperPSO(dev, dst_tex.pixelFormat, 1);
  if (!pso) return false;
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = dst_tex;
  pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:pass];
  if (!renc) return false;
  renc.label = @"Netal NR Async Input Downsample";
  const float inv_dst[2] = {
      1.0f / static_cast<float>(std::max<NSUInteger>(1, dst_tex.width)),
      1.0f / static_cast<float>(std::max<NSUInteger>(1, dst_tex.height)),
  };
  [renc setRenderPipelineState:pso];
  [renc setFragmentTexture:src_tex atIndex:0];
  [renc setFragmentBytes:inv_dst length:sizeof(inv_dst) atIndex:0];
  [renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [renc endEncoding];
  return true;
}

bool EncodeRenderComposeBakedToOutput(
    id<MTLDevice> dev,
    id<MTLCommandBuffer> cb,
    id<MTLTexture> src_tex,
    id<MTLTexture> baked_delta_tex,
    bool srgb_transfer,
    id<MTLTexture> dst_tex) {
  if (!src_tex || !baked_delta_tex || !dst_tex || (dst_tex.usage & MTLTextureUsageRenderTarget) == 0) {
    return false;
  }
  id<MTLRenderPipelineState> pso = GetOrBuildRenderHelperPSO(dev, dst_tex.pixelFormat, 2);
  if (!pso) return false;
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = dst_tex;
  pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:pass];
  if (!renc) return false;
  renc.label = @"Netal NR Single-Pass Baked Delta Compose";
  const uint32_t dims[4] = {
      static_cast<uint32_t>(dst_tex.width),
      static_cast<uint32_t>(dst_tex.height),
      srgb_transfer ? 1u : 0u,
      0u,
  };
  [renc setRenderPipelineState:pso];
  [renc setFragmentTexture:src_tex atIndex:0];
  [renc setFragmentTexture:baked_delta_tex atIndex:1];
  [renc setFragmentBytes:dims length:sizeof(dims) atIndex:0];
  [renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [renc endEncoding];
  return true;
}

bool EncodeRenderBlitToOutput(
    id<MTLDevice> dev,
    id<MTLCommandBuffer> cb,
    id<MTLTexture> src_rgba16f,
    id<MTLTexture> dst_tex) {
  if ((dst_tex.usage & MTLTextureUsageRenderTarget) != 0) {
    id<MTLRenderPipelineState> pso = GetOrBuildBlitPSO(dev, dst_tex.pixelFormat);
    if (!pso) return false;
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = dst_tex;
    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:pass];
    if (!renc) return false;
    renc.label = @"Netal NR RenderTarget Writeback";
    [renc setRenderPipelineState:pso];
    [renc setFragmentTexture:src_rgba16f atIndex:0];
    [renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [renc endEncoding];
    return true;
  }
  if (src_rgba16f.pixelFormat == dst_tex.pixelFormat) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return false;
    [blit copyFromTexture:src_rgba16f
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(dst_tex.width, dst_tex.height, 1)
                toTexture:dst_tex
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return true;
  }
  return false;
}

void DoPrepareOnThread(id<MTLDevice> device) {
  auto& rt = RT();
  const auto t0 = std::chrono::steady_clock::now();
  const std::string bridge_dir = FindBridgeDirectory();

  const char* raw_metallib = std::getenv("NETAL_NR_METALLIB");
  std::string metallib_path =
      (raw_metallib && *raw_metallib)
          ? std::string(raw_metallib)
          : (bridge_dir.empty() ? std::string{} : bridge_dir + "/nr_fast_kernels.metallib");

  const char* raw_flow_lib = std::getenv("NETAL_NR_FLOW_METALLIB");
  std::string flow_metallib_path =
      (raw_flow_lib && *raw_flow_lib)
          ? std::string(raw_flow_lib)
          : (bridge_dir.empty() ? std::string{} : bridge_dir + "/netal_nr_optical_flow.metallib");

  const char* raw_model = std::getenv("NETAL_NR_MODEL");
  const char* raw_synth = std::getenv("NETAL_NR_SYNTHETIC");
  std::string model_path = raw_model ? std::string(raw_model) : std::string{};
  std::string synth_spec = raw_synth ? std::string(raw_synth) : std::string{};
  if (synth_spec.empty() && model_path.compare(0, 10, "synthetic:") == 0) {
    synth_spec = model_path.substr(10);
  }

  std::string err;
  std::unique_ptr<netal_nr_fast::Context> built_ctx;
  if (!synth_spec.empty()) {
    netal_nr_fast::TopologyKind kind = netal_nr_fast::TopologyKind::Small31;
    if (synth_spec == "ht" || synth_spec == "ht71") {
      kind = netal_nr_fast::TopologyKind::HT71;
    } else if (synth_spec != "small" && synth_spec != "small31") {
      err = "invalid NETAL_NR_SYNTHETIC='" + synth_spec + "' (expected small|ht)";
    }
    if (err.empty()) {
      netal_nr_fast::SyntheticOptions sopt{};
      sopt.metallib_path = metallib_path;
      built_ctx = netal_nr_fast::Context::PrepareSynthetic(device, kind, sopt, &err);
    }
  } else if (!model_path.empty()) {
    netal_nr_fast::LoadOptions lopt{};
    lopt.metallib_path = metallib_path;
    built_ctx = netal_nr_fast::Context::LoadFromPath(device, model_path, lopt, &err);
  } else {
    err = "neither NETAL_NR_MODEL nor NETAL_NR_SYNTHETIC was provided";
  }

  if (!built_ctx) {
    Log("event=nr_prepare status=failed error=\"%s\"", err.c_str());
    rt.prep_status.store(PrepStatus::Failed, std::memory_order_release);
    return;
  }

  if (FileExistsRegular(metallib_path)) {
    std::lock_guard<std::mutex> blit_lock(rt.blit_mutex);
    NSError* lerr = nil;
    NSURL* lurl = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path.c_str()]];
    id<MTLLibrary> lib = [device newLibraryWithURL:lurl error:&lerr];
    if (lib) {
      id<MTLFunction> vs = [lib newFunctionWithName:@"nr_bridge_blit_vs"];
      id<MTLFunction> fs = [lib newFunctionWithName:@"nr_bridge_blit_fs"];
      id<MTLFunction> rfs = [lib newFunctionWithName:@"nr_bridge_resize_fs"];
      id<MTLFunction> cfs = [lib newFunctionWithName:@"nr_bridge_compose_baked_fs"];
      if (vs && fs) {
        rt.blit_library = lib;
        rt.blit_vs = vs;
        rt.blit_fs = fs;
        rt.resize_fs = rfs;
        rt.compose_baked_fs = cfs;
      }
    }
  }

  auto flow = std::make_unique<OpticalFlowSynthesizer>();
  std::string flow_err;
  bool flow_ok = false;
  if (FileExistsRegular(flow_metallib_path)) {
    flow_ok = flow->init_from_metallib(
        device, [NSString stringWithUTF8String:flow_metallib_path.c_str()], FlowConfig{}, &flow_err);
  } else {
    flow_err = "flow metallib not found at '" + flow_metallib_path + "'";
  }

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  {
    std::lock_guard<std::mutex> lock(rt.prep_mutex);
    rt.device = device;
    rt.nr_ctx = std::move(built_ctx);
    if (flow_ok) {
      rt.flow_synth = std::move(flow);
    }
    rt.prep_status.store(PrepStatus::Ready, std::memory_order_release);
  }

  Log("event=nr_prepare status=ok topology=%s blocks=%d synthetic=%d weight_format=%s flow_ready=%d elapsed_ms=%.2f",
      rt.nr_ctx->topology().name.c_str(),
      rt.nr_ctx->topology().total_blocks,
      rt.nr_ctx->is_synthetic() ? 1 : 0,
      rt.nr_ctx->weight_format().c_str(),
      flow_ok ? 1 : 0,
      elapsed_ms);
}

bool EnsurePrepared(id<MTLDevice> device) {
  auto& rt = RT();
  PrepStatus cur = rt.prep_status.load(std::memory_order_acquire);
  if (cur == PrepStatus::Ready) return true;
  if (cur == PrepStatus::Failed || cur == PrepStatus::Preparing || !device) return false;

  bool start_now = false;
  {
    std::lock_guard<std::mutex> lock(rt.prep_mutex);
    if (rt.prep_status.load(std::memory_order_relaxed) == PrepStatus::Cold) {
      rt.prep_status.store(PrepStatus::Preparing, std::memory_order_release);
      start_now = true;
    }
  }
  if (!start_now) {
    return rt.prep_status.load(std::memory_order_acquire) == PrepStatus::Ready;
  }

  if (rt.sync_prepare) {
    DoPrepareOnThread(device);
    return rt.prep_status.load(std::memory_order_acquire) == PrepStatus::Ready;
  }

  if (!RunDetached(^{
        DoPrepareOnThread(device);
      })) {
    Log("event=nr_prepare status=failed error=\"detached_thread_start_failed\"");
    rt.prep_status.store(PrepStatus::Failed, std::memory_order_release);
  }
  return false;
}

bool ShouldCapturePairFrame(uint64_t frame_id) {
  auto& rt = RT();
  if (rt.capture_pair_dir.empty() || rt.capture_frames.empty()) return false;
  return std::find(rt.capture_frames.begin(), rt.capture_frames.end(), frame_id) != rt.capture_frames.end();
}

void WriteDiagnosticPNG(
    id<MTLBuffer> buf,
    uint32_t w,
    uint32_t h,
    size_t row_bytes,
    MTLPixelFormat pf,
    const std::string& out_path) {
  const uint8_t* raw = static_cast<const uint8_t*>(buf.contents);
  if (!raw) return;

  std::vector<uint8_t> converted_rgba8;
  const void* bitmap_ptr = raw;
  size_t bitmap_row_bytes = row_bytes;
  CGBitmapInfo bi = static_cast<CGBitmapInfo>(
      static_cast<uint32_t>(kCGImageAlphaNoneSkipLast) | static_cast<uint32_t>(kCGBitmapByteOrder32Big));

  if (pf == MTLPixelFormatRGBA16Float) {
    converted_rgba8.resize(static_cast<size_t>(w) * h * 4u, 255u);
    for (uint32_t y = 0; y < h; ++y) {
      const uint16_t* row16 = reinterpret_cast<const uint16_t*>(raw + static_cast<size_t>(y) * row_bytes);
      for (uint32_t x = 0; x < w; ++x) {
        size_t dst_idx = (static_cast<size_t>(y) * w + x) * 4u;
        for (size_t c = 0; c < 3; ++c) {
          _Float16 hv = 0;
          std::memcpy(&hv, &row16[x * 4u + c], 2);
          float fv = std::clamp(static_cast<float>(hv), 0.0f, 1.0f);
          converted_rgba8[dst_idx + c] = static_cast<uint8_t>(std::lround(fv * 255.0f));
        }
      }
    }
    bitmap_ptr = converted_rgba8.data();
    bitmap_row_bytes = static_cast<size_t>(w) * 4u;
  } else if (pf == MTLPixelFormatBGRA8Unorm || pf == MTLPixelFormatBGRA8Unorm_sRGB) {
    bi = static_cast<CGBitmapInfo>(
        static_cast<uint32_t>(kCGImageAlphaNoneSkipFirst) | static_cast<uint32_t>(kCGBitmapByteOrder32Little));
  }

  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGContextRef ctx = CGBitmapContextCreate(const_cast<void*>(bitmap_ptr), w, h, 8, bitmap_row_bytes, cs, bi);
  CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : nullptr;
  std::string tmp_path = out_path + ".tmp";
  NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:tmp_path.c_str()]];
  CGImageDestinationRef dst =
      url ? CGImageDestinationCreateWithURL((__bridge CFURLRef)url, (__bridge CFStringRef)UTTypePNG.identifier, 1, nullptr)
          : nullptr;

  bool ok = false;
  if (ctx && img && dst) {
    CGImageDestinationAddImage(dst, img, nullptr);
    ok = CGImageDestinationFinalize(dst);
  }
  if (dst) CFRelease(dst);
  if (img) CGImageRelease(img);
  if (ctx) CGContextRelease(ctx);
  if (cs) CGColorSpaceRelease(cs);

  if (ok) {
    rename(tmp_path.c_str(), out_path.c_str());
    Log("event=nr_capture_saved path=%s size=%ux%u", out_path.c_str(), w, h);
  } else {
    unlink(tmp_path.c_str());
  }
}

void LogFallbackRateLimited(const char* reason, SeamKind seam, uint64_t frame_id) {
  uint64_t cnt = ++RT().fallback_logs;
  if (cnt <= 8 || (cnt % 300) == 0) {
    Log("event=nr_fallback seam=%s frame=%llu reason=%s count=%llu",
        SeamKindName(seam),
        static_cast<unsigned long long>(frame_id),
        reason,
        static_cast<unsigned long long>(cnt));
  }
}

netal_nr_fast::FrameParams BuildFastParams(
    uint32_t out_w,
    uint32_t out_h,
    uint32_t net_w,
    uint32_t net_h,
    uint64_t frame_id,
    bool reset,
    float mv_scale_x,
    float mv_scale_y,
    bool srgb_transfer,
    bool depth_inverted) {
  const auto& cfg = RT().cfg;
  netal_nr_fast::FrameParams fp{};
  fp.output_width = out_w;
  fp.output_height = out_h;
  fp.network_width = net_w;
  fp.network_height = net_h;
  fp.geometry_mode = netal_nr_fast::GeometryMode::ScaledNetwork;
  fp.profile = ToFastProfile(cfg.profile);
  fp.detail_strength = cfg.detail;
  fp.colour_strength = cfg.colour;
  fp.intensity = cfg.intensity;
  fp.motion_scale_x = mv_scale_x;
  fp.motion_scale_y = mv_scale_y;
  fp.frame_index = static_cast<uint32_t>(frame_id & 0xFFFFFFFFu);
  fp.reset = reset;
  fp.depth_inverted = depth_inverted;
  fp.use_closest_depth = true;
  fp.srgb_transfer = srgb_transfer;
  return fp;
}

bool ProcessFrameOnCommandBuffer(
    id<MTLDevice> dev,
    id<MTLCommandBuffer> game_cb,
    SeamKind seam,
    uint64_t frame_id,
    uint64_t stream_id,
    id<MTLTexture> raw_src_color,
    id<MTLTexture> native_motion,
    float native_mv_scale_x,
    float native_mv_scale_y,
    id<MTLTexture> native_depth,
    bool depth_inverted,
    id<MTLTexture> dst_output,
    bool game_reset) {
  if (!dev || !game_cb || !raw_src_color || !dst_output) return false;
  auto& rt = RT();

  if ((frame_id % 30u) == 1u && !rt.pause_marker_path.empty()) {
    rt.state.poll_pause_file(rt.pause_marker_path);
  }

  if (!EnsurePrepared(dev)) {
    LogFallbackRateLimited("preparing", seam, frame_id);
    return false;
  }

  const uint32_t out_w = static_cast<uint32_t>(dst_output.width);
  const uint32_t out_h = static_cast<uint32_t>(dst_output.height);
  const bool has_native_motion = (native_motion != nil);

  const auto d = rt.state.begin_frame(frame_id, stream_id, out_w, out_h, has_native_motion, game_reset);
  if (d.action == ContinuousAdapterState::Action::Disabled ||
      d.action == ContinuousAdapterState::Action::Paused ||
      d.action == ContinuousAdapterState::Action::InvalidGeometry ||
      d.action == ContinuousAdapterState::Action::OtherStream ||
      d.action == ContinuousAdapterState::Action::OneShotExhausted) {
    LogFallbackRateLimited("state_gate", seam, frame_id);
    return false;
  }

  id<MTLTexture> src_color = raw_src_color;
  if (src_color == dst_output || (src_color.usage & MTLTextureUsageShaderRead) == 0) {
    std::lock_guard<std::mutex> slock(rt.stage_mutex);
    rt.inline_src_copy_tex = EnsurePrivateTex(
        dev,
        rt.inline_src_copy_tex,
        raw_src_color.pixelFormat,
        static_cast<uint32_t>(raw_src_color.width),
        static_cast<uint32_t>(raw_src_color.height),
        MTLTextureUsageShaderRead,
        @"netal.nr.fast.src_stage");
    if (!rt.inline_src_copy_tex) {
      if (d.admitted) rt.state.complete_slot(d.slot, d.generation, frame_id, false);
      return false;
    }
    id<MTLBlitCommandEncoder> blit = [game_cb blitCommandEncoder];
    [blit copyFromTexture:raw_src_color
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(raw_src_color.width, raw_src_color.height, 1)
                toTexture:rt.inline_src_copy_tex
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    src_color = rt.inline_src_copy_tex;
  }

  const bool effective_srgb =
      rt.srgb_transfer_override || IsSRGBPixelFormat(src_color.pixelFormat) || IsSRGBPixelFormat(dst_output.pixelFormat);

  float mv_scale_x = 1.0f;
  float mv_scale_y = 1.0f;
  if (has_native_motion) {
    const float mw = static_cast<float>(std::max<NSUInteger>(1, native_motion.width));
    const float mh = static_cast<float>(std::max<NSUInteger>(1, native_motion.height));
    mv_scale_x = (native_mv_scale_x != 0.0f) ? (native_mv_scale_x / mw) : (1.0f / mw);
    mv_scale_y = (native_mv_scale_y != 0.0f) ? (native_mv_scale_y / mh) : (1.0f / mh);
  }

  // ---- INLINE EXECUTION MODE ----
  if (d.action == ContinuousAdapterState::Action::AdmitInline) {
    const bool direct_shader_write = CanShaderWriteDirectly(dst_output);
    id<MTLTexture> nr_target = dst_output;
    if (!direct_shader_write) {
      std::lock_guard<std::mutex> slock(rt.stage_mutex);
      rt.inline_composed_tex = EnsurePrivateTex(
          dev,
          rt.inline_composed_tex,
          MTLPixelFormatRGBA16Float,
          out_w,
          out_h,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          @"netal.nr.fast.inline_composed");
      nr_target = rt.inline_composed_tex;
    }
    if (!nr_target) {
      rt.state.complete_slot(d.slot, d.generation, frame_id, false);
      LogFallbackRateLimited("alloc_failed", seam, frame_id);
      return false;
    }

    bool encode_ok = false;
    std::string err;
    {
      std::lock_guard<std::mutex> nrlock(rt.nr_encode_mutex);
      id<MTLTexture> motion_tex = native_motion;
      if (d.synthesize_motion && rt.flow_synth && rt.flow_synth->ready()) {
        FlowOutput fout{};
        if (rt.flow_synth->encode(game_cb, src_color, d.reset, fout, &err)) {
          motion_tex = fout.flow_texture;
          mv_scale_x = 1.0f / static_cast<float>(std::max<NSUInteger>(1, motion_tex.width));
          mv_scale_y = 1.0f / static_cast<float>(std::max<NSUInteger>(1, motion_tex.height));
        }
      }

      netal_nr_fast::FrameInputs fi{};
      fi.color_texture = src_color;
      fi.motion_texture = motion_tex;
      fi.depth_texture = native_depth;
      fi.output_texture = nr_target;
      fi.params = BuildFastParams(
          out_w, out_h, d.network_width, d.network_height,
          frame_id, d.reset, mv_scale_x, mv_scale_y, effective_srgb, depth_inverted);

      netal_nr_fast::FrameTicket ticket{};
      netal_nr_fast::Report rep{};
      encode_ok = rt.nr_ctx->encode(fi, game_cb, ticket, rep, &err);
    }

    if (!encode_ok) {
      rt.state.complete_slot(d.slot, d.generation, frame_id, false);
      Log("event=nr_fallback seam=%s frame=%llu reason=encode_failed detail=\"%s\"",
          SeamKindName(seam), static_cast<unsigned long long>(frame_id), err.c_str());
      return false;
    }

    if (!direct_shader_write) {
      if (!EncodeRenderBlitToOutput(dev, game_cb, nr_target, dst_output)) {
        rt.state.complete_slot(d.slot, d.generation, frame_id, false);
        LogFallbackRateLimited("blit_failed", seam, frame_id);
        return false;
      }
    }

    const uint32_t slot = d.slot;
    const uint64_t gen = d.generation;
    const uint32_t nw = d.network_width;
    const uint32_t nh = d.network_height;
    const bool did_reset = d.reset;
    const bool did_synth = d.synthesize_motion;
    [game_cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
      auto& cbrt = RT();
      const bool ok = (done.status == MTLCommandBufferStatusCompleted);
      const double gpu_ms =
          (done.GPUEndTime > done.GPUStartTime && done.GPUStartTime > 0.0)
              ? (done.GPUEndTime - done.GPUStartTime) * 1000.0
              : 0.0;
      cbrt.state.complete_slot(slot, gen, frame_id, ok);
      if (ok && cbrt.state.note_published(stream_id, gen)) {
        auto snap = cbrt.state.snapshot();
        const uint64_t pub = snap.telemetry.published;
        if (pub <= 8 || (pub % 120u) == 0u) {
          Log("event=nr_frame_published frame=%llu published=%llu stream=%llu seam=%s mode=inline "
              "net=%ux%u out=%ux%u reset=%d synth_motion=%d gpu_ms=%.3f",
              static_cast<unsigned long long>(frame_id),
              static_cast<unsigned long long>(pub),
              static_cast<unsigned long long>(stream_id),
              SeamKindName(seam),
              nw, nh, out_w, out_h,
              did_reset ? 1 : 0,
              did_synth ? 1 : 0,
              gpu_ms);
        }
        if (pub == 1 || (pub % 60u) == 0u) {
          Log("event=nr_pacing mode=inline admitted=%llu completed=%llu published=%llu skipped_busy=%llu "
              "synth_motion=%llu resets=%llu display_gpu_ms=%.3f worker_gpu_ms=0.000",
              static_cast<unsigned long long>(snap.telemetry.admitted),
              static_cast<unsigned long long>(snap.telemetry.completed),
              static_cast<unsigned long long>(snap.telemetry.published),
              static_cast<unsigned long long>(snap.telemetry.skipped_busy),
              static_cast<unsigned long long>(snap.telemetry.synthesized_motion_frames),
              static_cast<unsigned long long>(snap.telemetry.resets),
              gpu_ms);
        }
      }
    }];
    return true;
  }

  // ---- ASYNC WORKER DISPATCH (AdmitAsync) ----
  if (d.action == ContinuousAdapterState::Action::AdmitAsync) {
    bool async_dispatched = false;
    id<MTLCommandQueue> wq = nil;
    id<MTLSharedEvent> wev = nil;
    id<MTLTexture> w_col = nil;
    id<MTLTexture> w_mv = nil;
    id<MTLTexture> w_baked = nil;

    {
      std::lock_guard<std::mutex> alock(rt.async_res_mutex);
      if (!rt.worker_queue) rt.worker_queue = [dev newCommandQueue];
      if (!rt.worker_event) rt.worker_event = [dev newSharedEvent];
      wq = rt.worker_queue;
      wev = rt.worker_event;

      // Stage worker input directly at (network_width, network_height) so both input
      // staging and OpticalFlowSynthesizer execute at network scale instead of full 3.5K/4K.
      const MTLPixelFormat stage_pf =
          (src_color.pixelFormat == MTLPixelFormatRGBA16Float) ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA8Unorm;
      rt.worker_color_tex[d.slot] = EnsurePrivateTex(
          dev,
          rt.worker_color_tex[d.slot],
          stage_pf,
          d.network_width,
          d.network_height,
          MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget,
          @"netal.nr.fast.worker_color");
      w_col = rt.worker_color_tex[d.slot];

      if (has_native_motion) {
        rt.worker_motion_tex[d.slot] = EnsurePrivateTex(
            dev,
            rt.worker_motion_tex[d.slot],
            native_motion.pixelFormat,
            static_cast<uint32_t>(native_motion.width),
            static_cast<uint32_t>(native_motion.height),
            MTLTextureUsageShaderRead,
            @"netal.nr.fast.worker_motion");
        w_mv = rt.worker_motion_tex[d.slot];
      }

      rt.baked_delta_slot_tex[d.slot] = EnsurePrivateTex(
          dev,
          rt.baked_delta_slot_tex[d.slot],
          MTLPixelFormatRGBA16Float,
          d.network_width,
          d.network_height,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          @"netal.nr.fast.baked_delta");
      w_baked = rt.baked_delta_slot_tex[d.slot];
      rt.delta_slot_w[d.slot] = d.network_width;
      rt.delta_slot_h[d.slot] = d.network_height;
    }

    if (wq && wev && w_col && w_baked && (!has_native_motion || w_mv)) {
      if (EncodeRenderResize(dev, game_cb, src_color, w_col)) {
        if (has_native_motion && w_mv) {
          id<MTLBlitCommandEncoder> blit = [game_cb blitCommandEncoder];
          if (blit) {
            [blit copyFromTexture:native_motion
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(native_motion.width, native_motion.height, 1)
                        toTexture:w_mv
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
          }
        }

        const uint64_t sig_val = ++rt.worker_event_val;
        [game_cb encodeSignalEvent:wev value:sig_val];

        id<MTLCommandBuffer> worker_cb = [wq commandBuffer];
        if (worker_cb) {
          worker_cb.label = @"Netal NR Async Worker";
          [worker_cb encodeWaitForEvent:wev value:sig_val];

          std::string werr;
          bool w_ok = false;
          {
            std::lock_guard<std::mutex> nrlock(rt.nr_encode_mutex);
            id<MTLTexture> motion_for_worker = w_mv;
            float w_mv_sx = mv_scale_x;
            float w_mv_sy = mv_scale_y;
            if (d.synthesize_motion && rt.flow_synth && rt.flow_synth->ready()) {
              FlowOutput fout{};
              if (rt.flow_synth->encode(worker_cb, w_col, d.reset, fout, &werr)) {
                motion_for_worker = fout.flow_texture;
                w_mv_sx = 1.0f / static_cast<float>(std::max<NSUInteger>(1, motion_for_worker.width));
                w_mv_sy = 1.0f / static_cast<float>(std::max<NSUInteger>(1, motion_for_worker.height));
              }
            }

            netal_nr_fast::FrameInputs w_fi{};
            w_fi.color_texture = w_col;
            w_fi.motion_texture = motion_for_worker;
            w_fi.output_texture = nil;
            w_fi.export_baked_delta_texture = w_baked;
            w_fi.params = BuildFastParams(
                out_w, out_h, d.network_width, d.network_height,
                frame_id, d.reset, w_mv_sx, w_mv_sy, effective_srgb, depth_inverted);

            netal_nr_fast::FrameTicket w_ticket{};
            netal_nr_fast::Report w_rep{};
            w_ok = rt.nr_ctx->encode(w_fi, worker_cb, w_ticket, w_rep, &werr);
          }

          if (w_ok) {
            const uint32_t w_slot = d.slot;
            const uint64_t w_gen = d.generation;
            [worker_cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
              auto& cbrt = RT();
              const bool ok = (done.status == MTLCommandBufferStatusCompleted);
              if (ok && done.GPUEndTime > done.GPUStartTime && done.GPUStartTime > 0.0) {
                cbrt.last_worker_gpu_ms.store((done.GPUEndTime - done.GPUStartTime) * 1000.0, std::memory_order_relaxed);
              }
              cbrt.state.complete_slot(w_slot, w_gen, frame_id, ok);
            }];
            [worker_cb commit];
            async_dispatched = true;
          }
        }
      }
    }

    if (!async_dispatched) {
      rt.state.complete_slot(d.slot, d.generation, frame_id, false);
    }
  }

  // ---- SINGLE-PASS BAKED DELTA COMPOSITION ON LIVE DISPLAY FRAME (~0.25 ms) ----
  if (d.has_reusable_delta) {
    id<MTLTexture> r_baked = nil;
    uint32_t r_w = 0, r_h = 0;
    {
      std::lock_guard<std::mutex> alock(rt.async_res_mutex);
      r_baked = rt.baked_delta_slot_tex[d.reusable_slot];
      r_w = rt.delta_slot_w[d.reusable_slot];
      r_h = rt.delta_slot_h[d.reusable_slot];
    }

    if (r_baked && r_w > 0 && r_h > 0) {
      bool composed_ok = false;
      if ((dst_output.usage & MTLTextureUsageRenderTarget) != 0) {
        composed_ok = EncodeRenderComposeBakedToOutput(
            dev, game_cb, src_color, r_baked, effective_srgb, dst_output);
      } else if (CanShaderWriteDirectly(dst_output)) {
        netal_nr_fast::FrameParams cparams = BuildFastParams(
            out_w, out_h, r_w, r_h, frame_id, false, 1.0f, 1.0f, effective_srgb, depth_inverted);
        std::string cerr;
        composed_ok = rt.nr_ctx->encode_compose_baked_delta(
            src_color, r_baked, cparams, dst_output, game_cb, &cerr);
      }

      if (composed_ok) {
        const uint64_t gen = d.generation;
        const uint64_t r_frame = d.reusable_frame;
        [game_cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
              auto& cbrt = RT();
              const bool ok = (done.status == MTLCommandBufferStatusCompleted);
              const double disp_ms =
                  (done.GPUEndTime > done.GPUStartTime && done.GPUStartTime > 0.0)
                      ? (done.GPUEndTime - done.GPUStartTime) * 1000.0
                      : 0.0;
              if (ok && cbrt.state.note_published(stream_id, gen)) {
                auto snap = cbrt.state.snapshot();
                const uint64_t pub = snap.telemetry.published;
                const double w_ms = cbrt.last_worker_gpu_ms.load(std::memory_order_relaxed);
                if (pub <= 8 || (pub % 120u) == 0u) {
                  Log("event=nr_frame_published frame=%llu delta_frame=%llu published=%llu stream=%llu "
                      "seam=%s mode=async_compose net=%ux%u out=%ux%u gpu_ms=%.3f worker_gpu_ms=%.3f",
                      static_cast<unsigned long long>(frame_id),
                      static_cast<unsigned long long>(r_frame),
                      static_cast<unsigned long long>(pub),
                      static_cast<unsigned long long>(stream_id),
                      SeamKindName(seam),
                      r_w, r_h, out_w, out_h,
                      disp_ms,
                      w_ms);
                }
                if (pub == 1 || (pub % 60u) == 0u) {
                  Log("event=nr_pacing mode=async admitted=%llu completed=%llu published=%llu skipped_busy=%llu "
                      "synth_motion=%llu resets=%llu display_gpu_ms=%.3f worker_gpu_ms=%.3f",
                      static_cast<unsigned long long>(snap.telemetry.admitted),
                      static_cast<unsigned long long>(snap.telemetry.completed),
                      static_cast<unsigned long long>(snap.telemetry.published),
                      static_cast<unsigned long long>(snap.telemetry.skipped_busy),
                      static_cast<unsigned long long>(snap.telemetry.synthesized_motion_frames),
                      static_cast<unsigned long long>(snap.telemetry.resets),
                      disp_ms,
                      w_ms);
                }
              }
            }];
        return true;
      }
    }
  }

  LogFallbackRateLimited("async_warming_or_busy", seam, frame_id);
  return false;
}

// ============================================================================
// Seam A & B: Metal 3 MTLFXTemporalScaler & MTLFXSpatialScaler Hooks
// ============================================================================

IMP FindOriginalEncode(Class cls) {
  auto& rt = RT();
  std::lock_guard<std::mutex> lock(rt.hook_mutex);
  for (Class c = cls; c != Nil; c = class_getSuperclass(c)) {
    for (const auto& entry : rt.encode_hooks) {
      if (entry.cls == c) return entry.original;
    }
  }
  return nullptr;
}

bool IsMetal3CommandBuffer(id cb) {
  return cb != nil &&
         [cb respondsToSelector:@selector(computeCommandEncoderWithDispatchType:)] &&
         [cb respondsToSelector:@selector(blitCommandEncoder)] &&
         [cb respondsToSelector:@selector(addCompletedHandler:)];
}

uint64_t GetOrAssignStreamId(id obj) {
  if (!obj) return 1;
  NSNumber* existing = objc_getAssociatedObject(obj, &g_stream_id_assoc_key);
  if (existing) return existing.unsignedLongLongValue;
  uint64_t sid = ++RT().next_stream_id;
  objc_setAssociatedObject(obj, &g_stream_id_assoc_key, @(sid), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return sid;
}

void HookScalerEncodeToCommandBuffer(id self, SEL _cmd, id<MTLCommandBuffer> cb) {
  IMP orig = FindOriginalEncode(object_getClass(self));
  if (!IsMetal3CommandBuffer(cb)) {
    if (orig) reinterpret_cast<EncodeM3Fn>(orig)(self, _cmd, cb);
    return;
  }

  const uint64_t frame_id = ++RT().total_frames;
  const uint64_t stream_id = GetOrAssignStreamId(self);
  NSNumber* kind_num = objc_getAssociatedObject(self, &g_seam_kind_assoc_key);
  const SeamKind seam = (kind_num && kind_num.unsignedIntValue == static_cast<uint32_t>(SeamKind::Spatial))
                            ? SeamKind::Spatial
                            : SeamKind::Temporal;

  id<MTLDevice> dev = cb.device;
  bool handled = false;

  if (seam == SeamKind::Spatial) {
    id<MTLFXSpatialScaler> sp = (id<MTLFXSpatialScaler>)self;
    handled = ProcessFrameOnCommandBuffer(
        dev, cb, SeamKind::Spatial, frame_id, stream_id,
        sp.colorTexture, nil, 1.0f, 1.0f, nil, false, sp.outputTexture, false);
  } else {
    id<MTLFXTemporalScaler> ts = (id<MTLFXTemporalScaler>)self;
    handled = ProcessFrameOnCommandBuffer(
        dev, cb, SeamKind::Temporal, frame_id, stream_id,
        ts.colorTexture,
        ts.motionTexture,
        ts.motionVectorScaleX,
        ts.motionVectorScaleY,
        ts.depthTexture,
        ts.depthReversed,
        ts.outputTexture,
        ts.reset);
  }

  if (!handled && orig) {
    reinterpret_cast<EncodeM3Fn>(orig)(self, _cmd, cb);
  }
}

void EnsureScalerClassHooked(Class cls) {
  if (!cls) return;
  auto& rt = RT();
  std::lock_guard<std::mutex> lock(rt.hook_mutex);
  for (const auto& entry : rt.encode_hooks) {
    if (entry.cls == cls) return;
  }
  Method m = class_getInstanceMethod(cls, @selector(encodeToCommandBuffer:));
  if (!m) return;
  IMP prev = method_getImplementation(m);
  IMP old = class_replaceMethod(
      cls, @selector(encodeToCommandBuffer:), reinterpret_cast<IMP>(&HookScalerEncodeToCommandBuffer), method_getTypeEncoding(m));
  rt.encode_hooks.push_back({cls, old ? old : prev});
}

id HookNewTemporalScalerWithDevice(id self, SEL _cmd, id<MTLDevice> device) {
  id scaler = reinterpret_cast<CreateTemporalFn>(RT().orig_new_temporal)(self, _cmd, device);
  if (scaler) {
    GetOrAssignStreamId(scaler);
    objc_setAssociatedObject(
        scaler, &g_seam_kind_assoc_key, @(static_cast<uint32_t>(SeamKind::Temporal)), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    EnsureScalerClassHooked(object_getClass(scaler));
    EnsurePrepared(device);
  }
  return scaler;
}

id HookNewSpatialScalerWithDevice(id self, SEL _cmd, id<MTLDevice> device) {
  id scaler = reinterpret_cast<CreateSpatialFn>(RT().orig_new_spatial)(self, _cmd, device);
  if (scaler) {
    GetOrAssignStreamId(scaler);
    objc_setAssociatedObject(
        scaler, &g_seam_kind_assoc_key, @(static_cast<uint32_t>(SeamKind::Spatial)), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    EnsureScalerClassHooked(object_getClass(scaler));
    EnsurePrepared(device);
  }
  return scaler;
}

// ============================================================================
// Seam C: CAMetalLayer setDevice: + Metal 4 MTL4CommandQueue signalDrawable:
// ============================================================================

void HookMtl4SignalDrawable(id self, SEL _cmd, id<MTLDrawable> drawable) {
  auto& rt = RT();
  const uint64_t frame_id = ++rt.total_frames;

  if (drawable != nil && [(id)drawable respondsToSelector:@selector(texture)]) {
    id<MTLTexture> drawable_tex = [(id<CAMetalDrawable>)drawable texture];
    id<MTL4CommandQueue> q4 = (id<MTL4CommandQueue>)self;
    id<MTLDevice> dev = [q4 respondsToSelector:@selector(device)] ? q4.device : nil;

    if (drawable_tex != nil && dev != nil) {
      uint64_t stream_id = 1;
      if ([(id)drawable respondsToSelector:@selector(layer)]) {
        CAMetalLayer* layer = [(id<CAMetalDrawable>)drawable layer];
        if (layer) stream_id = GetOrAssignStreamId(layer);
      }

      if (EnsurePrepared(dev)) {
        std::lock_guard<std::mutex> dlock(rt.drawable_mutex);
        if (!rt.drawable_queue) rt.drawable_queue = [dev newCommandQueue];
        if (!rt.drawable_sync_ev) rt.drawable_sync_ev = [dev newSharedEvent];

        const uint32_t w = static_cast<uint32_t>(drawable_tex.width);
        const uint32_t h = static_cast<uint32_t>(drawable_tex.height);
        rt.drawable_copy_tex = EnsurePrivateTex(
            dev, rt.drawable_copy_tex, drawable_tex.pixelFormat, w, h,
            MTLTextureUsageShaderRead, @"netal.nr.fast.drawable_copy");

        if (rt.drawable_queue && rt.drawable_sync_ev && rt.drawable_copy_tex) {
          const uint64_t v1 = ++rt.drawable_ev_val;
          [q4 signalEvent:rt.drawable_sync_ev value:v1];

          id<MTLCommandBuffer> cb = [rt.drawable_queue commandBuffer];
          if (cb) {
            cb.label = @"Netal NR Drawable Seam";
            [cb encodeWaitForEvent:rt.drawable_sync_ev value:v1];

            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:drawable_tex
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(w, h, 1)
                        toTexture:rt.drawable_copy_tex
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];

            const bool want_capture = ShouldCapturePairFrame(frame_id);
            const size_t bpp = (drawable_tex.pixelFormat == MTLPixelFormatRGBA16Float) ? 8u : 4u;
            const size_t row_bytes = ((static_cast<size_t>(w) * bpp + 255u) & ~size_t(255u));
            id<MTLBuffer> raw_cap_buf = nil;
            id<MTLBuffer> nr_cap_buf = nil;

            if (want_capture &&
                (drawable_tex.pixelFormat == MTLPixelFormatBGRA8Unorm ||
                 drawable_tex.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB ||
                 drawable_tex.pixelFormat == MTLPixelFormatRGBA8Unorm ||
                 drawable_tex.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ||
                 drawable_tex.pixelFormat == MTLPixelFormatRGBA16Float)) {
              raw_cap_buf = [dev newBufferWithLength:row_bytes * h options:MTLResourceStorageModeShared];
              nr_cap_buf = [dev newBufferWithLength:row_bytes * h options:MTLResourceStorageModeShared];
              if (raw_cap_buf) {
                id<MTLBlitCommandEncoder> cblit = [cb blitCommandEncoder];
                [cblit copyFromTexture:rt.drawable_copy_tex
                           sourceSlice:0
                           sourceLevel:0
                          sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(w, h, 1)
                              toBuffer:raw_cap_buf
                     destinationOffset:0
                destinationBytesPerRow:row_bytes
              destinationBytesPerImage:row_bytes * h];
                [cblit endEncoding];
              }
            }

            const bool nr_applied = ProcessFrameOnCommandBuffer(
                dev, cb, SeamKind::Drawable, frame_id, stream_id,
                rt.drawable_copy_tex, nil, 1.0f, 1.0f, nil, false, drawable_tex, false);

            if (raw_cap_buf && nr_cap_buf && nr_applied && rt.capture_write_q) {
              id<MTLBlitCommandEncoder> cblit2 = [cb blitCommandEncoder];
              [cblit2 copyFromTexture:drawable_tex
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(w, h, 1)
                             toBuffer:nr_cap_buf
                    destinationOffset:0
               destinationBytesPerRow:row_bytes
             destinationBytesPerImage:row_bytes * h];
              [cblit2 endEncoding];

              const MTLPixelFormat pf = drawable_tex.pixelFormat;
              const std::string dir = rt.capture_pair_dir;
              dispatch_queue_t wq = rt.capture_write_q;
              [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                if (done.status != MTLCommandBufferStatusCompleted) return;
                dispatch_async(wq, ^{
                  @autoreleasepool {
                    char rname[64], nname[64];
                    std::snprintf(rname, sizeof(rname), "/raw_%05llu.png", static_cast<unsigned long long>(frame_id));
                    std::snprintf(nname, sizeof(nname), "/nr_%05llu.png", static_cast<unsigned long long>(frame_id));
                    WriteDiagnosticPNG(raw_cap_buf, w, h, row_bytes, pf, dir + rname);
                    WriteDiagnosticPNG(nr_cap_buf, w, h, row_bytes, pf, dir + nname);
                  }
                });
              }];
            }

            const uint64_t v2 = ++rt.drawable_ev_val;
            [cb encodeSignalEvent:rt.drawable_sync_ev value:v2];
            [cb commit];
            [q4 waitForEvent:rt.drawable_sync_ev value:v2];
          }
        }
      } else {
        LogFallbackRateLimited("preparing", SeamKind::Drawable, frame_id);
      }
    }
  }

  if (rt.orig_q4_signal_drawable) {
    rt.orig_q4_signal_drawable(self, _cmd, drawable);
  }
}

void InstallMetal4QueueHookOnce(id<MTLDevice> device) {
  static dispatch_once_t once_token;
  dispatch_once(&once_token, ^{
    if (!device) return;
    if (@available(macOS 26.0, *)) {
      if (![device respondsToSelector:@selector(newMTL4CommandQueueWithDescriptor:error:)]) return;
      NSError* err = nil;
      MTL4CommandQueueDescriptor* desc = [MTL4CommandQueueDescriptor new];
      id<MTL4CommandQueue> probe_q4 = [device newMTL4CommandQueueWithDescriptor:desc error:&err];
      if (!probe_q4) {
        Log("event=mtl4_hook_skipped reason=probe_queue_nil detail=\"%s\"",
            err ? err.localizedDescription.UTF8String : "none");
        return;
      }
      Class q4_cls = object_getClass(probe_q4);
      Method m = class_getInstanceMethod(q4_cls, @selector(signalDrawable:));
      if (!m) {
        Log("event=mtl4_hook_skipped reason=selector_missing class=%s", class_getName(q4_cls));
        return;
      }
      IMP prev = method_getImplementation(m);
      IMP old = class_replaceMethod(
          q4_cls,
          @selector(signalDrawable:),
          reinterpret_cast<IMP>(&HookMtl4SignalDrawable),
          method_getTypeEncoding(m));
      RT().orig_q4_signal_drawable = reinterpret_cast<SignalDrawableFn>(old ? old : prev);
      g_installed_hooks.fetch_or(kHookMtl4SignalDrawable, std::memory_order_release);
      Log("event=mtl4_hook_installed class=%s device=\"%s\"",
          class_getName(q4_cls), device.name.UTF8String);
    }
  });
}

void HookLayerSetDevice(id self, SEL _cmd, id<MTLDevice> device) {
  auto& rt = RT();
  if (rt.orig_layer_set_device) {
    rt.orig_layer_set_device(self, _cmd, device);
  }
  if ([self isKindOfClass:[CAMetalLayer class]]) {
    CAMetalLayer* ml = (CAMetalLayer*)self;
    ml.framebufferOnly = NO;
    GetOrAssignStreamId(ml);
    Log("event=layer_configured layer=%p framebufferOnly=0 device=\"%s\"",
        (__bridge void*)ml, device ? device.name.UTF8String : "nil");
  }
  if (device) {
    InstallMetal4QueueHookOnce(device);
    EnsurePrepared(device);
  }
}

void ParseCaptureDiagnosticsFromEnv() {
  const char* cap_dir = std::getenv("NETAL_NR_CAPTURE_PAIR_DIR");
  const char* cap_frames = std::getenv("NETAL_NR_CAPTURE_FRAMES");
  if (!cap_dir || *cap_dir == '\0' || !cap_frames || *cap_frames == '\0') {
    return;
  }
  auto& rt = RT();
  rt.capture_pair_dir = cap_dir;
  const char* p = cap_frames;
  while (*p && rt.capture_frames.size() < 256u) {
    while (*p == ' ' || *p == '\t' || *p == ',') ++p;
    if (!*p) break;
    char* end = nullptr;
    unsigned long long v = std::strtoull(p, &end, 10);
    if (end == p) break;
    rt.capture_frames.push_back(static_cast<uint64_t>(v));
    p = (*end == ',') ? (end + 1) : end;
  }
  if (!rt.capture_frames.empty()) {
    [[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:rt.capture_pair_dir.c_str()]
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nullptr];
    rt.capture_write_q = dispatch_queue_create("netal.nr.fast.capture_pair", DISPATCH_QUEUE_SERIAL);
  }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) uint32_t NetalNRFastBridgeInstalledHookMask() {
  return g_installed_hooks.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) uint64_t NetalNRFastBridgePublishedCount() {
  if (g_installed_hooks.load(std::memory_order_acquire) == 0u) {
    return 0;
  }
  return RT().state.snapshot().telemetry.published;
}

extern "C" __attribute__((visibility("default"))) uint64_t NetalNRFastBridgeCompletedCount() {
  if (g_installed_hooks.load(std::memory_order_acquire) == 0u) {
    return 0;
  }
  return RT().state.snapshot().telemetry.completed;
}

__attribute__((constructor))
static void NetalNRFastBridgeInit() {
  @autoreleasepool {
    ContinuousAdapterConfig cfg{};
    const bool valid = ContinuousAdapterConfig::parse_from_env(cfg);
    if (!valid) {
      if (const char* log_env = std::getenv("NETAL_NR_LOG"); log_env && *log_env) {
        RT().log_file = std::fopen(log_env, "a");
      }
      Log("event=config_invalid error=\"%s\"", cfg.error.c_str());
      return;
    }

    // Strict opt-in: default Off exits immediately without allocating state, opening files, or installing hooks.
    if (cfg.mode == AdapterMode::Off) {
      return;
    }

    if (!IsTargetProcess()) {
      return;
    }

    auto& rt = RT();
    rt.cfg = cfg;
    rt.state.configure(cfg);

    if (const char* log_env = std::getenv("NETAL_NR_LOG"); log_env && *log_env) {
      rt.log_file = std::fopen(log_env, "a");
      std::string lp(log_env);
      size_t slash = lp.find_last_of('/');
      if (slash != std::string::npos) {
        rt.pause_marker_path = lp.substr(0, slash) + "/netal-nr-pause";
      }
    }
    if (const char* pause_env = std::getenv("NETAL_NR_PAUSE_FILE"); pause_env && *pause_env) {
      rt.pause_marker_path = pause_env;
    }
    if (const char* srgb_env = std::getenv("NETAL_NR_SRGB"); srgb_env && std::strcmp(srgb_env, "1") == 0) {
      rt.srgb_transfer_override = true;
    }
    if (const char* sync_env = std::getenv("NETAL_NR_SYNC_PREPARE"); sync_env && std::strcmp(sync_env, "1") == 0) {
      rt.sync_prepare = true;
    }

    ParseCaptureDiagnosticsFromEnv();

    // Seam A: Metal 3 MTLFXTemporalScalerDescriptor
    if (Method m_temp = class_getInstanceMethod(
            [MTLFXTemporalScalerDescriptor class], @selector(newTemporalScalerWithDevice:))) {
      IMP prev = method_getImplementation(m_temp);
      IMP old = class_replaceMethod(
          [MTLFXTemporalScalerDescriptor class],
          @selector(newTemporalScalerWithDevice:),
          reinterpret_cast<IMP>(&HookNewTemporalScalerWithDevice),
          method_getTypeEncoding(m_temp));
      rt.orig_new_temporal = old ? old : prev;
      g_installed_hooks.fetch_or(kHookTemporalScaler, std::memory_order_release);
    }

    // Seam B: Optional Metal 3 MTLFXSpatialScalerDescriptor
    if (cfg.hook_spatial) {
      if (Method m_spat = class_getInstanceMethod(
              [MTLFXSpatialScalerDescriptor class], @selector(newSpatialScalerWithDevice:))) {
        IMP prev = method_getImplementation(m_spat);
        IMP old = class_replaceMethod(
            [MTLFXSpatialScalerDescriptor class],
            @selector(newSpatialScalerWithDevice:),
            reinterpret_cast<IMP>(&HookNewSpatialScalerWithDevice),
            method_getTypeEncoding(m_spat));
        rt.orig_new_spatial = old ? old : prev;
        g_installed_hooks.fetch_or(kHookSpatialScaler, std::memory_order_release);
      }
    }

    // Seam C: Optional CAMetalLayer + Metal 4 MTL4CommandQueue signalDrawable:
    if (cfg.hook_drawable) {
      if (Method m_layer = class_getInstanceMethod([CAMetalLayer class], @selector(setDevice:))) {
        IMP prev = method_getImplementation(m_layer);
        IMP old = class_replaceMethod(
            [CAMetalLayer class],
            @selector(setDevice:),
            reinterpret_cast<IMP>(&HookLayerSetDevice),
            method_getTypeEncoding(m_layer));
        rt.orig_layer_set_device = reinterpret_cast<SetDeviceFn>(old ? old : prev);
        g_installed_hooks.fetch_or(kHookLayerSetDevice, std::memory_order_release);
      }
    }

    Log("event=bridge_ready mode=%s scale=%.3f profile=%s detail=%.2f colour=%.2f intensity=%.2f "
        "async=%d hook_spatial=%d hook_drawable=%d hooks_mask=0x%x",
        AdapterModeName(cfg.mode),
        cfg.scale,
        NrProfileName(cfg.profile),
        cfg.detail,
        cfg.colour,
        cfg.intensity,
        cfg.async_worker ? 1 : 0,
        cfg.hook_spatial ? 1 : 0,
        cfg.hook_drawable ? 1 : 0,
        g_installed_hooks.load(std::memory_order_relaxed));
  }
}

__attribute__((destructor))
static void NetalNRFastBridgeFini() {
  if (g_installed_hooks.load(std::memory_order_acquire) == 0u) {
    return;
  }
  auto& rt = RT();
  if (rt.capture_write_q) {
    dispatch_sync(rt.capture_write_q, ^{});
  }
  auto snap = rt.state.snapshot();
  Log("event=bridge_shutdown admitted=%llu completed=%llu published=%llu skipped_busy=%llu",
      static_cast<unsigned long long>(snap.telemetry.admitted),
      static_cast<unsigned long long>(snap.telemetry.completed),
      static_cast<unsigned long long>(snap.telemetry.published),
      static_cast<unsigned long long>(snap.telemetry.skipped_busy));
  std::lock_guard<std::mutex> lock(rt.log_mutex);
  if (rt.log_file) {
    std::fflush(rt.log_file);
  }
}
