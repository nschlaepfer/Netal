#import "dlss_fg_session.h"
#import "fg_runtime.h"
#include "netal_fg_lowering.h"

#import <CommonCrypto/CommonDigest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr NSUInteger kWidth = 640, kHeight = 360;   // the runtime's pinned profile
constexpr NSUInteger kSlots = 4;                    // frames that may be in flight

void SetError(NSError** error, NSInteger code, NSString* message) {
  if (error != nullptr)
    *error = [NSError errorWithDomain:@"NeuralStudioDLSSFG" code:code
                             userInfo:@{NSLocalizedDescriptionKey: message}];
}

struct ImportParams {
  uint32_t content_w, content_h;   // interpolator output/content rect in the game textures
  uint32_t render_w, render_h;     // motion/depth texture content size
  float motion_scale_x, motion_scale_y;   // texel value -> pixels at (render_w, render_h)
  uint32_t out_w, out_h;           // output texture size for the export pass
};
struct FullresParams {
  uint32_t content_w, content_h, net_w, net_h, net_padded_h, cand_w, cand_h, out_w, out_h, pad0, pad1, pad2;
};
constexpr NSUInteger kParamsStride = 512;   // per slot: ImportParams at 0, FullresParams at 256

// Compiled at runtime: a handful of resamplers, no dependence on a Metal toolchain at build time.
const char* kAdapterSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct ImportParams { uint content_w, content_h, render_w, render_h; float msx, msy; uint out_w, out_h; };
constexpr sampler kLinear(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler kNearest(coord::normalized, filter::nearest, address::clamp_to_edge);
// Game colour (any float-readable format, content rect at the origin) -> RGBA8 640x360, 2x2 box of
// bilinear taps so a 2.7x reduction does not alias badly.
kernel void fg_import_color(texture2d<float> color [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                            constant ImportParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= 640u || gid.y >= 360u) return;
  float2 content = float2(p.content_w, p.content_h);
  float2 tex = float2(color.get_width(), color.get_height());
  float2 step = content / float2(640.0f, 360.0f);
  if (p.content_w == 640u && p.content_h == 360u) {   // native geometry: the frame itself, untouched
    out.write(float4(color.read(gid).rgb, 1.0f), gid);
    return;
  }
  float2 base = (float2(gid) + 0.5f) * step;
  float4 acc = 0.0f;
  for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
    float2 pix = base + (float2(i, j) - 0.5f) * step * 0.5f;
    acc += color.sample(kLinear, pix / tex);
  }
  out.write(float4(saturate(acc.rgb * 0.25f), 1.0f), gid);
}
// Motion texels are in units the interpolator scales by (msx, msy) to render pixels; the runtime
// wants pixels of its own 640x360 frame, so scale by 640/render_w and 360/render_h on top.
kernel void fg_import_motion(texture2d<float> motion [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                             constant ImportParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= 640u || gid.y >= 360u) return;
  float2 render = float2(p.render_w, p.render_h);
  float2 tex = float2(motion.get_width(), motion.get_height());
  float2 uv = (float2(gid) + 0.5f) / float2(640.0f, 360.0f) * render / tex;
  float2 v = motion.sample(kLinear, uv).xy * float2(p.msx, p.msy);       // render pixels
  v *= float2(640.0f, 360.0f) / render;                                    // 640x360 pixels
  out.write(float4(v, 0.0f, 0.0f), gid);
}
kernel void fg_import_depth(depth2d<float> depth [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                            constant ImportParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= 640u || gid.y >= 360u) return;
  float2 render = float2(p.render_w, p.render_h);
  float2 tex = float2(depth.get_width(), depth.get_height());
  float2 uv = (float2(gid) + 0.5f) / float2(640.0f, 360.0f) * render / tex;
  out.write(float4(depth.sample(kNearest, uv), 0.0f, 0.0f, 0.0f), gid);
}
// Generated RGBA8 640x360 -> the interpolator's output over the content rect; pixels outside the
// content rect (the game allocates taller textures than it renders) pass the current colour through.
kernel void fg_export(texture2d<float> generated [[texture(0)]], texture2d<float> color [[texture(1)]],
                      texture2d<float, access::write> out [[texture(2)]],
                      constant ImportParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.out_w || gid.y >= p.out_h) return;
  if (gid.x < p.content_w && gid.y < p.content_h) {
    float2 uv = (float2(gid) + 0.5f) / float2(p.content_w, p.content_h);
    float4 g = generated.sample(kLinear, uv);
    out.write(float4(g.rgb, 1.0f), gid);
  } else {
    out.write(color.read(gid), gid);
  }
}
kernel void fg_passthrough(texture2d<float> color [[texture(0)]], texture2d<float, access::write> out [[texture(1)]],
                           constant ImportParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.out_w || gid.y >= p.out_h) return;
  out.write(color.read(gid), gid);
}

// ---- Resolution-decoupled compositor: launches 55 and 84 of the DLSS-G graph, evaluated at the
// game's resolution. Both address everything by normalised coordinates: the coarse candidate flows
// (320x180, half2 pairs in UV units), the blend weights (320x180) and the network's fine flows and
// logit (640x360, in 640x360 pixels). The two real frames are sampled at full resolution, so the
// generated frame keeps their detail; only the network's small outputs are interpolated.
struct FullresParams { uint content_w, content_h, net_w, net_h, net_padded_h, cand_w, cand_h, out_w, out_h, pad0, pad1, pad2; };
// Launch 55's point sample: coordinates snapped to 1/4096, floor to a texel, zero outside.
inline float4 l55_point_sample(texture2d<float> source, float2 coordinate, uint2 content) {
  float2 fixed12 = floor(coordinate * 4096.0f) * (1.0f / 4096.0f);
  int2 location = int2(floor(fixed12 * float2(content)));
  if (location.x < 0 || location.y < 0 || location.x >= int(content.x) || location.y >= int(content.y)) return float4(0.0f);
  return source.read(uint2(location));
}
inline float2 decode_candidate(ulong packed) {
  return float2(float(as_type<half>(ushort((packed >> 16) & 0xffff))), float(as_type<half>(ushort(packed & 0xffff))));
}
kernel void fg_fullres_candidates(texture2d<float> prev [[texture(0)]], texture2d<float> cur [[texture(1)]],
                                  texture2d<float> blend [[texture(2)]],
                                  texture2d<float, access::write> surf_a [[texture(3)]],
                                  texture2d<float, access::write> surf_b [[texture(4)]],
                                  constant FullresParams& p [[buffer(0)]],
                                  device const ulong* cand_a [[buffer(1)]], device const ulong* cand_b [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.content_w || gid.y >= p.content_h) return;
  uint2 content = uint2(p.content_w, p.content_h);
  float2 uv = (float2(gid) + 0.5f) / float2(content);
  float cx = fma(uv.x, float(p.cand_w), -0.5f) - as_type<float>(0x34000000u);
  float cy = fma(uv.y, float(p.cand_h), -0.5f) - as_type<float>(0x34000000u);
  int ix = clamp(int(rint(cx)), 0, int(p.cand_w) - 1), iy = clamp(int(rint(cy)), 0, int(p.cand_h) - 1);
  ulong packed_a = cand_a[uint(iy) * p.cand_w + uint(ix)], packed_b = cand_b[uint(iy) * p.cand_w + uint(ix)];
  bool invalid_a = packed_a == 0x7fffffff7ffffffful, invalid_b = packed_b == 0x7fffffff7ffffffful;
  float2 offset_a = invalid_a ? float2(0.0f) : decode_candidate(packed_a);
  float2 offset_b = invalid_b ? float2(0.0f) : decode_candidate(packed_b);
  float3 base_a = prev.read(gid).xyz, base_b = cur.read(gid).xyz;
  float3 moved_a = l55_point_sample(prev, uv + offset_b, content).xyz;
  float3 moved_b = l55_point_sample(cur, uv + offset_a, content).xyz;
  float3 resolved_a = invalid_b ? moved_b : moved_a;
  float3 resolved_b = invalid_a ? moved_a : moved_b;
  // Blend weights live on the 320x180 grid; the same 1/4096-snapped floor lookup as launch 55.
  float2 w = l55_point_sample(blend, uv, uint2(p.cand_w, p.cand_h)).xy;
  surf_a.write(float4(mix(base_a, resolved_a, w.x), 1.0f), gid);
  surf_b.write(float4(mix(base_b, resolved_b, w.y), 1.0f), gid);
}
inline float3 bilinear_rgb(texture2d<float> tex, float2 position, uint2 content) {
  int2 base = int2(floor(position));
  float2 f = position - float2(base);
  int x0 = clamp(base.x, 0, int(content.x) - 1), x1 = clamp(base.x + 1, 0, int(content.x) - 1);
  int y0 = clamp(base.y, 0, int(content.y) - 1), y1 = clamp(base.y + 1, 0, int(content.y) - 1);
  float3 c00 = tex.read(uint2(x0, y0)).xyz, c10 = tex.read(uint2(x1, y0)).xyz;
  float3 c01 = tex.read(uint2(x0, y1)).xyz, c11 = tex.read(uint2(x1, y1)).xyz;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}
kernel void fg_fullres_blend(texture2d<float> surf_a [[texture(0)]], texture2d<float> surf_b [[texture(1)]],
                             texture2d<float> color [[texture(2)]], texture2d<float, access::write> out [[texture(3)]],
                             constant FullresParams& p [[buffer(0)]],
                             device const half* motion [[buffer(1)]], device const uint* sigmoid_lut [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.out_w || gid.y >= p.out_h) return;
  if (gid.x >= p.content_w || gid.y >= p.content_h) { out.write(color.read(gid), gid); return; }
  uint2 content = uint2(p.content_w, p.content_h);
  float2 net = float2(p.net_w, p.net_h);
  // Bilinear over the network grid, pixel centres aligned; the 8-half record per network pixel is
  // flow-to-a (2), flow-to-b (2), blend logit (1), unused (3).
  float2 q = (float2(gid) + 0.5f) * net / float2(content) - 0.5f;
  int2 q0 = int2(floor(q)); float2 f = q - float2(q0);
  int x0 = clamp(q0.x, 0, int(p.net_w) - 1), x1 = clamp(q0.x + 1, 0, int(p.net_w) - 1);
  int y0 = clamp(q0.y, 0, int(p.net_h) - 1), y1 = clamp(q0.y + 1, 0, int(p.net_h) - 1);
  uint i00 = (uint(y0) * p.net_w + uint(x0)) * 8u, i10 = (uint(y0) * p.net_w + uint(x1)) * 8u;
  uint i01 = (uint(y1) * p.net_w + uint(x0)) * 8u, i11 = (uint(y1) * p.net_w + uint(x1)) * 8u;
  float4 w4 = float4((1 - f.x) * (1 - f.y), f.x * (1 - f.y), (1 - f.x) * f.y, f.x * f.y);
  float2 fa = w4.x * float2(motion[i00], motion[i00 + 1]) + w4.y * float2(motion[i10], motion[i10 + 1])
            + w4.z * float2(motion[i01], motion[i01 + 1]) + w4.w * float2(motion[i11], motion[i11 + 1]);
  float2 fb = w4.x * float2(motion[i00 + 2], motion[i00 + 3]) + w4.y * float2(motion[i10 + 2], motion[i10 + 3])
            + w4.z * float2(motion[i01 + 2], motion[i01 + 3]) + w4.w * float2(motion[i11 + 2], motion[i11 + 3]);
  // The blend is the sigmoid of the logit (the graph's LUT, indexed by the half's bits); interpolate
  // the four taps' blends rather than the logits so a sharp weight edge stays sharp.
  float blend = w4.x * as_type<float>(sigmoid_lut[as_type<ushort>(motion[i00 + 4])])
              + w4.y * as_type<float>(sigmoid_lut[as_type<ushort>(motion[i10 + 4])])
              + w4.z * as_type<float>(sigmoid_lut[as_type<ushort>(motion[i01 + 4])])
              + w4.w * as_type<float>(sigmoid_lut[as_type<ushort>(motion[i11 + 4])]);
  float2 scale = float2(content) / net;   // network pixels -> game pixels
  float3 a = bilinear_rgb(surf_a, float2(gid) + fa * scale, content);
  float3 b = bilinear_rgb(surf_b, float2(gid) + fb * scale, content);
  out.write(float4(saturate(mix(b, a, blend)), 1.0f), gid);
}
)METAL";

std::string Sha256File(NSString* path) {
  NSData* data = [NSData dataWithContentsOfFile:path];
  if (data == nil) return "";
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), digest.data());
  char out[65];
  for (size_t i = 0; i < digest.size(); ++i) std::snprintf(out + 2 * i, 3, "%02x", digest[i]);
  return std::string(out, 64);
}
}  // namespace

@implementation NeuralStudioDLSSFGSession {
  id<MTLDevice> _device;
  NSFGSession* _session;
  std::string _libs, _model, _rom, _sigmoid, _unit, _fraction;   // paths kept alive for the runtime
  id<MTLComputePipelineState> _importColor, _importMotion, _importDepth, _export, _passthrough;
  id<MTLComputePipelineState> _fullresCandidates, _fullresBlend;
  // Full-resolution compositor state: the game's colour retained per slot (previous real frame),
  // and the two candidate surfaces at content size. Recreated when the game's geometry changes.
  bool _fullres;
  bool _exact;
  std::array<id<MTLTexture>, kSlots> _colorCopy;
  id<MTLTexture> _lastColorCopy;
  id<MTLTexture> _surfA, _surfB;
  NSUInteger _fullresW, _fullresH; MTLPixelFormat _fullresColorFormat;
  uint64_t _fullresFrames, _fullresSkipped;
  std::array<id<MTLTexture>, kSlots> _color, _motion, _depth, _output;
  id<MTLBuffer> _params;                       // kSlots x 256 bytes
  std::array<id<MTL4ArgumentTable>, kSlots * 8> _tables;
  id<MTLResidencySet> _residency;
  uint64_t _frameIndex;                        // slot selection
  uint64_t _runtimeFrameId;                    // monotonic ids the runtime requires
  id<MTLTexture> _lastCurrent;                 // lineage: previous must be the last accepted current
  id<MTLTexture> _lastCurrentForDump;
  BOOL _needsReset;
  uint64_t _generated, _resets;
  NSUInteger _lastDispatches;
  // NEURALSTUDIO_FG_DUMP_DIR: frames 60..63 of the runtime's inputs/outputs are blitted into shared
  // textures and written to disk at frame 72 (after those command buffers have long completed).
  std::string _dumpDir;
  std::array<id<MTLTexture>, 4> _dumpColor, _dumpPrev, _dumpMotion, _dumpDepth, _dumpOut;
  std::array<id<MTLTexture>, 4> _dumpGameColor, _dumpGameOut, _dumpGameUI;
  id<MTLTexture> _dumpUISource;   // the interpolator's UI texture for the current frame (dump only)
  bool _dumpWritten;
}

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device runtimeBundleURL:(NSURL*)bundle error:(NSError**)error {
  self = [super init];
  if (!self) return nil;
  // The default game path uses accelerated arithmetic. Preserve explicit per-stage
  // overrides, but never report reference mode while inherited hardware flags remain.
  const bool requestReference = std::getenv("NEURALSTUDIO_FG_EXACT") != nullptr &&
      std::getenv("NEURALSTUDIO_FG_EXACT")[0] == '1';
  const auto lowering = NetalResolveFGLowering(requestReference,
      std::getenv("NSFG_PREFIX_MODE"), std::getenv("NSFG_BLOCK_MODE"));
  if (!lowering.valid) {
    SetError(error, 6, @"Reference FG requested with hardware arithmetic overrides; remove the NSFG hardware overrides and restart the game");
    return nil;
  }
  if (!requestReference &&
      (setenv("NSFG_BLOCK_MODE", "hw", 0) != 0 || setenv("NSFG_PREFIX_MODE", "hw", 0) != 0)) {
    SetError(error, 6, @"Unable to configure FG arithmetic");
    return nil;
  }
  _exact = lowering.reference();

  _device = device;
  _needsReset = YES;
  if (const char* dump = std::getenv("NEURALSTUDIO_FG_DUMP_DIR")) _dumpDir = dump;
  NSString* root = bundle.path;
  NSString* manifestPath = [root stringByAppendingPathComponent:@"manifest.json"];
  NSData* manifestData = [NSData dataWithContentsOfFile:manifestPath];
  if (manifestData == nil) { SetError(error, 1, [NSString stringWithFormat:@"FG bundle manifest missing: %@", manifestPath]); return nil; }
  NSError* json_error = nil;
  NSDictionary* manifest = [NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&json_error];
  NSDictionary* files = manifest[@"files"];
  if (![files isKindOfClass:[NSDictionary class]]) { SetError(error, 1, @"FG bundle manifest has no files"); return nil; }
  for (NSString* relative in files) {
    NSString* expected = files[relative][@"sha256"];
    if ([relative hasPrefix:@"lib/"] || [relative hasPrefix:@"include/"]) continue;   // host-side, not loaded here
    const auto actual = Sha256File([root stringByAppendingPathComponent:relative]);
    if (actual != expected.UTF8String) {
      SetError(error, 2, [NSString stringWithFormat:@"FG bundle file does not match its manifest: %@", relative]);
      return nil;
    }
  }
  _libs = [root stringByAppendingPathComponent:@"libs"].UTF8String;
  _model = [root stringByAppendingPathComponent:@"model"].UTF8String;
  _rom = [root stringByAppendingPathComponent:@"rom.bin"].UTF8String;
  _sigmoid = [root stringByAppendingPathComponent:@"sigmoid.bin"].UTF8String;
  _unit = [root stringByAppendingPathComponent:@"ex2-unit.bin"].UTF8String;
  _fraction = [root stringByAppendingPathComponent:@"ex2-fraction.bin"].UTF8String;

  // Adapter pipelines.
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  options.mathMode = MTLMathModeSafe;
  NSError* compile_error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:@(kAdapterSource) options:options error:&compile_error];
  if (library == nil) {
    SetError(error, 3, [NSString stringWithFormat:@"FG adapter kernels failed to compile: %@", compile_error.localizedDescription]);
    return nil;
  }
  auto pso = [&](NSString* name) -> id<MTLComputePipelineState> {
    NSError* e = nil;
    id<MTLFunction> f = [library newFunctionWithName:name];
    return f == nil ? nil : [device newComputePipelineStateWithFunction:f error:&e];
  };
  _importColor = pso(@"fg_import_color");
  _importMotion = pso(@"fg_import_motion");
  _importDepth = pso(@"fg_import_depth");
  _export = pso(@"fg_export");
  _passthrough = pso(@"fg_passthrough");
  _fullresCandidates = pso(@"fg_fullres_candidates");
  _fullresBlend = pso(@"fg_fullres_blend");
  if (!_fullresCandidates || !_fullresBlend) { SetError(error, 3, @"FG full-resolution compositor pipelines failed"); return nil; }
  // NEURALSTUDIO_FG_FULLRES=0 keeps the 640x360 upsample (the A/B baseline).
  _fullres = std::getenv("NEURALSTUDIO_FG_FULLRES") == nullptr || std::getenv("NEURALSTUDIO_FG_FULLRES")[0] != '0';
  if (!_importColor || !_importMotion || !_importDepth || !_export || !_passthrough) {
    SetError(error, 3, @"FG adapter pipeline state failed");
    return nil;
  }

  // Runtime input/output ring at the pinned profile.
  auto texture = [&](MTLPixelFormat format, MTLTextureUsage usage) -> id<MTLTexture> {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:kWidth height:kHeight mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = usage;
    return [device newTextureWithDescriptor:d];
  };
  for (NSUInteger s = 0; s < kSlots; ++s) {
    _color[s] = texture(MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    _motion[s] = texture(MTLPixelFormatRG16Float, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    _depth[s] = texture(MTLPixelFormatR32Float, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    _output[s] = texture(MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    if (!_color[s] || !_motion[s] || !_depth[s] || !_output[s]) { SetError(error, 4, @"FG ring allocation failed"); return nil; }
  }
  _params = [device newBufferWithLength:kSlots * kParamsStride options:MTLResourceStorageModeShared];
  MTL4ArgumentTableDescriptor* td = [[MTL4ArgumentTableDescriptor alloc] init];
  td.maxBufferBindCount = 3;
  td.maxTextureBindCount = 5;
  td.initializeBindings = YES;
  for (auto& table : _tables) {
    NSError* e = nil;
    table = [device newArgumentTableWithDescriptor:td error:&e];
    if (table == nil) { SetError(error, 4, @"FG argument table allocation failed"); return nil; }
  }
  MTLResidencySetDescriptor* rd = [[MTLResidencySetDescriptor alloc] init];
  rd.initialCapacity = 32;
  NSError* re = nil;
  _residency = [device newResidencySetWithDescriptor:rd error:&re];
  if (_residency == nil) { SetError(error, 4, @"FG residency set failed"); return nil; }
  for (NSUInteger s = 0; s < kSlots; ++s) {
    [_residency addAllocation:_color[s]];
    [_residency addAllocation:_motion[s]];
    [_residency addAllocation:_depth[s]];
    [_residency addAllocation:_output[s]];
  }
  [_residency addAllocation:_params];
  [_residency commit];

  // The runtime session, Metal 4 mode. Unlike the SR session the FG runtime allows one per process.
  NSFGCreateInfo create{};
  create.metal_device = (__bridge void*)device;
  create.metal_command_queue = nullptr;
  create.metallib_directory = _libs.c_str();
  create.model_directory = _model.c_str();
  create.rcp_rom_path = _rom.c_str();
  create.sigmoid_lut_path = _sigmoid.c_str();
  create.ex2_unit_table_path = _unit.c_str();
  create.ex2_fraction_table_path = _fraction.c_str();
  create.width = kWidth;
  create.height = kHeight;
  create.ex2_mode = NSFG_EX2_COMPACT;
  create.auxiliary_mode = NSFG_AUXILIARY_LIVE_EARLY_RESOURCES;
  create.poison_byte = 0xa5;
  create.metal4 = true;
  const NSFGStatus status = NSFGSessionCreate(&create, &_session);
  if (status != NSFG_STATUS_OK || _session == nullptr) {
    SetError(error, 5, [NSString stringWithFormat:@"FG runtime session: %s", NSFGStatusName(status)]);
    return nil;
  }
  return self;
}

- (void)dealloc {
  if (_session != nullptr) NSFGSessionDestroy(_session);
}

- (void)invalidate { _needsReset = YES; _lastColorCopy = nil; }

// Metal 3 seam (native MetalFX titles): the same adapter with plain compute encoders; the runtime
// encodes through the shim into the same command buffer with real encoders, and Metal's hazard
// tracking orders everything.
- (BOOL)encodeToCommandBuffer:(id<MTLCommandBuffer>)command
                 colorTexture:(id<MTLTexture>)color
                 depthTexture:(id<MTLTexture>)depth
                motionTexture:(id<MTLTexture>)motion
                outputTexture:(id<MTLTexture>)output
                 contentWidth:(NSUInteger)contentWidth
                contentHeight:(NSUInteger)contentHeight
                  renderWidth:(NSUInteger)renderWidth
                 renderHeight:(NSUInteger)renderHeight
           motionVectorScaleX:(float)motionScaleX
           motionVectorScaleY:(float)motionScaleY
                        fence:(id<MTLFence>)fence
                        reset:(BOOL)reset
                        error:(NSError**)error {
  if (command == nil || color == nil || depth == nil || motion == nil || output == nil) { SetError(error, 10, @"interpolator inputs missing"); return NO; }
  if (contentWidth == 0 || contentHeight == 0 || contentWidth > color.width || contentHeight > color.height ||
      contentWidth > output.width || contentHeight > output.height || renderWidth == 0 || renderHeight == 0 ||
      renderWidth > motion.width || renderHeight > motion.height || renderWidth > depth.width || renderHeight > depth.height) {
    SetError(error, 11, @"interpolator geometry unsupported"); return NO;
  }
  if (!std::isfinite(motionScaleX) || !std::isfinite(motionScaleY) || motionScaleX == 0.0f || motionScaleY == 0.0f) { SetError(error, 12, @"motion vector scale unusable"); return NO; }
  if (!(output.usage & MTLTextureUsageShaderWrite)) { SetError(error, 14, @"output texture is not shader-writable"); return NO; }
  const bool do_reset = reset || _needsReset || _lastCurrent == nil;
  const NSUInteger slot = static_cast<NSUInteger>(_frameIndex++ % kSlots);
  id<MTLTexture> current = _color[slot];
  auto* params = reinterpret_cast<ImportParams*>(static_cast<uint8_t*>(_params.contents) + slot * kParamsStride);
  *params = {static_cast<uint32_t>(contentWidth), static_cast<uint32_t>(contentHeight), static_cast<uint32_t>(renderWidth),
             static_cast<uint32_t>(renderHeight), motionScaleX, motionScaleY, static_cast<uint32_t>(output.width), static_cast<uint32_t>(output.height)};
  auto* fparams = reinterpret_cast<FullresParams*>(static_cast<uint8_t*>(_params.contents) + slot * kParamsStride + 256);
  *fparams = {static_cast<uint32_t>(contentWidth), static_cast<uint32_t>(contentHeight), kWidth, kHeight, 368u, 320u, 180u,
              static_cast<uint32_t>(output.width), static_cast<uint32_t>(output.height), 0u, 0u, 0u};
  const bool fullres = _fullres && [self ensureFullresResources:color contentWidth:contentWidth contentHeight:contentHeight];
  const MTLSize group = MTLSizeMake(16, 16, 1), grid640 = MTLSizeMake(kWidth, kHeight, 1);
  auto run = [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, id<MTLTexture> t0, id<MTLTexture> t1,
                 id<MTLTexture> t2, MTLSize grid, id<MTLTexture> t3 = nil, id<MTLTexture> t4 = nil, bool full = false,
                 id<MTLBuffer> b1 = nil, NSUInteger o1 = 0, id<MTLBuffer> b2 = nil, NSUInteger o2 = 0) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:_params offset:slot * kParamsStride + (full ? 256 : 0) atIndex:0];
    if (b1 != nil) [enc setBuffer:b1 offset:o1 atIndex:1];
    if (b2 != nil) [enc setBuffer:b2 offset:o2 atIndex:2];
    [enc setTexture:t0 atIndex:0];
    if (t1 != nil) [enc setTexture:t1 atIndex:1];
    if (t2 != nil) [enc setTexture:t2 atIndex:2];
    if (t3 != nil) [enc setTexture:t3 atIndex:3];
    if (t4 != nil) [enc setTexture:t4 atIndex:4];
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
  };
  id<MTLComputeCommandEncoder> import = [command computeCommandEncoder];
  if (import == nil) { SetError(error, 20, @"compute encoder unavailable"); return NO; }
  if (fence != nil) [import waitForFence:fence];
  run(import, _importColor, color, current, nil, grid640);
  run(import, _importMotion, motion, _motion[slot], nil, grid640);
  run(import, _importDepth, depth, _depth[slot], nil, grid640);
  [import endEncoding];
  if (fullres) {
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:color toTexture:_colorCopy[slot]];
    [blit endEncoding];
  }
  NSFGFrameInfo frame{};
  frame.current_backbuffer_rgba8 = (__bridge void*)current;
  frame.previous_backbuffer_rgba8 = (__bridge void*)(do_reset ? current : _lastCurrent);
  frame.motion_rg16f = (__bridge void*)_motion[slot];
  frame.depth_r32f = (__bridge void*)_depth[slot];
  frame.generated_target_rgba8 = do_reset ? nullptr : (__bridge void*)_output[slot];
  frame.reset = do_reset;
  if (do_reset) { _runtimeFrameId = 0; NSFGSessionInvalidate(_session); } else { ++_runtimeFrameId; }
  frame.frame_id = _runtimeFrameId;
  NSFGFrameResult result{};
  const NSFGStatus status = NSFGSessionEvaluateMetal4(_session, &frame, (__bridge void*)command, &result);
  if (status != NSFG_STATUS_OK) {
    _needsReset = YES; _lastCurrent = nil;
    SetError(error, 30, [NSString stringWithFormat:@"FG runtime %s: %s", NSFGStatusName(status), NSFGSessionLastError(_session)]);
    return NO;
  }
  _lastCurrent = current; _needsReset = NO;
  id<MTLComputeCommandEncoder> exporter = [command computeCommandEncoder];
  if (exporter == nil) { _needsReset = YES; SetError(error, 20, @"compute encoder unavailable"); return NO; }
  const MTLSize grid_out = MTLSizeMake(output.width, output.height, 1);
  NSFGTailResources tail{};
  const bool fullres_now = fullres && !do_reset && _lastColorCopy != nil && NSFGSessionCopyLastTailResources(_session, &tail) == NSFG_STATUS_OK;
  if (do_reset) {
    run(exporter, _passthrough, color, output, nil, grid_out);
  } else if (fullres_now) {
    run(exporter, _fullresCandidates, _lastColorCopy, _colorCopy[slot], (__bridge id<MTLTexture>)tail.blend_rg16f_texture,
        MTLSizeMake(contentWidth, contentHeight, 1), _surfA, _surfB, true,
        (__bridge id<MTLBuffer>)tail.candidate_a_buffer, tail.candidate_offset, (__bridge id<MTLBuffer>)tail.candidate_b_buffer, tail.candidate_offset);
    run(exporter, _fullresBlend, _surfA, _surfB, color, grid_out, output, nil, true,
        (__bridge id<MTLBuffer>)tail.motion_half8_buffer, tail.motion_offset, (__bridge id<MTLBuffer>)tail.sigmoid_lut_buffer, tail.sigmoid_offset);
    ++_fullresFrames;
  } else {
    if (fullres) ++_fullresSkipped;
    run(exporter, _export, _output[slot], color, output, grid_out);
  }
  if (fence != nil) [exporter updateFence:fence];
  [exporter endEncoding];
  if (fullres) _lastColorCopy = _colorCopy[slot];
  if (do_reset) ++_resets; else ++_generated;
  _lastDispatches = result.selected_launch_count;
  return YES;
}
- (uint64_t)fullResolutionFrames { return _fullresFrames; }
- (uint64_t)fullResolutionSkipped { return _fullresSkipped; }
- (BOOL)fullResolutionEnabled { return _fullres; }
- (BOOL)exactLowering { return _exact; }

/// Colour copies at the game's format/size and RGBA16F candidate surfaces at content size; rebuilt
/// when the game changes geometry. Returns NO when they cannot be made (the caller then upsamples).
- (BOOL)ensureFullresResources:(id<MTLTexture>)color contentWidth:(NSUInteger)w contentHeight:(NSUInteger)h {
  if (_surfA != nil && _fullresW == w && _fullresH == h && _fullresColorFormat == color.pixelFormat &&
      _colorCopy[0].width == color.width && _colorCopy[0].height == color.height) return YES;
  MTLTextureDescriptor* cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:color.pixelFormat width:color.width height:color.height mipmapped:NO];
  cd.storageMode = MTLStorageModePrivate; cd.usage = MTLTextureUsageShaderRead;
  MTLTextureDescriptor* sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:w height:h mipmapped:NO];
  sd.storageMode = MTLStorageModePrivate; sd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  std::array<id<MTLTexture>, kSlots> copies;
  for (auto& c : copies) { c = [_device newTextureWithDescriptor:cd]; if (c == nil) return NO; }
  id<MTLTexture> a = [_device newTextureWithDescriptor:sd], b = [_device newTextureWithDescriptor:sd];
  if (a == nil || b == nil) return NO;
  for (NSUInteger s = 0; s < kSlots; ++s) { if (_colorCopy[s] != nil) [_residency removeAllocation:_colorCopy[s]]; }
  if (_surfA != nil) [_residency removeAllocation:_surfA];
  if (_surfB != nil) [_residency removeAllocation:_surfB];
  _colorCopy = copies; _surfA = a; _surfB = b; _lastColorCopy = nil;   // geometry changed: no valid previous
  _fullresW = w; _fullresH = h; _fullresColorFormat = color.pixelFormat;
  for (auto& c : _colorCopy) [_residency addAllocation:c];
  [_residency addAllocation:_surfA]; [_residency addAllocation:_surfB];
  [_residency commit];
  return YES;
}
- (uint64_t)generatedFrames { return _generated; }
- (uint64_t)resetFrames { return _resets; }
- (NSUInteger)runtimeDispatchesLastFrame { return _lastDispatches; }

- (BOOL)encodeToMTL4CommandBuffer:(id<MTL4CommandBuffer>)command
                     colorTexture:(id<MTLTexture>)color
                     depthTexture:(id<MTLTexture>)depth
                    motionTexture:(id<MTLTexture>)motion
                    outputTexture:(id<MTLTexture>)output
                     contentWidth:(NSUInteger)contentWidth
                    contentHeight:(NSUInteger)contentHeight
                      renderWidth:(NSUInteger)renderWidth
                     renderHeight:(NSUInteger)renderHeight
               motionVectorScaleX:(float)motionScaleX
               motionVectorScaleY:(float)motionScaleY
                            fence:(id<MTLFence>)fence
                            reset:(BOOL)reset
                            error:(NSError**)error {
  if (command == nil || color == nil || depth == nil || motion == nil || output == nil) {
    SetError(error, 10, @"interpolator inputs missing");
    return NO;
  }
  if (contentWidth == 0 || contentHeight == 0 || contentWidth > color.width || contentHeight > color.height ||
      contentWidth > output.width || contentHeight > output.height || renderWidth == 0 || renderHeight == 0 ||
      renderWidth > motion.width || renderHeight > motion.height || renderWidth > depth.width || renderHeight > depth.height) {
    SetError(error, 11, [NSString stringWithFormat:@"interpolator geometry unsupported: content %lux%lu render %lux%lu color %lux%lu output %lux%lu",
                         (unsigned long)contentWidth, (unsigned long)contentHeight, (unsigned long)renderWidth, (unsigned long)renderHeight,
                         (unsigned long)color.width, (unsigned long)color.height, (unsigned long)output.width, (unsigned long)output.height]);
    return NO;
  }
  if (!std::isfinite(motionScaleX) || !std::isfinite(motionScaleY) || motionScaleX == 0.0f || motionScaleY == 0.0f) {
    SetError(error, 12, @"motion vector scale unusable");
    return NO;
  }
  const bool depth_ok = depth.pixelFormat == MTLPixelFormatDepth32Float || depth.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ||
                        depth.pixelFormat == MTLPixelFormatDepth16Unorm || depth.pixelFormat == MTLPixelFormatR32Float;
  if (!depth_ok) { SetError(error, 13, @"depth format unsupported"); return NO; }
  if (!(output.usage & MTLTextureUsageShaderWrite)) { SetError(error, 14, @"output texture is not shader-writable"); return NO; }

  const bool do_reset = reset || _needsReset || _lastCurrent == nil;
  const NSUInteger slot = static_cast<NSUInteger>(_frameIndex++ % kSlots);
  id<MTLTexture> current = _color[slot];
  if (!do_reset && current == _lastCurrent) {   // cannot happen with kSlots > 1, guard anyway
    SetError(error, 15, @"ring collision");
    return NO;
  }
  auto* params = reinterpret_cast<ImportParams*>(static_cast<uint8_t*>(_params.contents) + slot * kParamsStride);
  params->content_w = static_cast<uint32_t>(contentWidth);
  params->content_h = static_cast<uint32_t>(contentHeight);
  params->render_w = static_cast<uint32_t>(renderWidth);
  params->render_h = static_cast<uint32_t>(renderHeight);
  params->motion_scale_x = motionScaleX;
  params->motion_scale_y = motionScaleY;
  params->out_w = static_cast<uint32_t>(output.width);
  params->out_h = static_cast<uint32_t>(output.height);
  const MTLGPUAddress params_address = _params.gpuAddress + slot * kParamsStride;
  auto* fparams = reinterpret_cast<FullresParams*>(static_cast<uint8_t*>(_params.contents) + slot * kParamsStride + 256);
  *fparams = {static_cast<uint32_t>(contentWidth), static_cast<uint32_t>(contentHeight), kWidth, kHeight, 368u, 320u, 180u,
              static_cast<uint32_t>(output.width), static_cast<uint32_t>(output.height), 0u, 0u, 0u};
  const MTLGPUAddress fparams_address = params_address + 256;
  // Full-resolution compositor resources, sized to the game's geometry.
  const bool fullres = _fullres && [self ensureFullresResources:color contentWidth:contentWidth contentHeight:contentHeight];

  [command useResidencySet:_residency];
  // Import.
  id<MTL4ComputeCommandEncoder> import = [command computeCommandEncoder];
  if (import == nil) { SetError(error, 20, @"Metal 4 compute encoder unavailable"); return NO; }
  [import barrierAfterQueueStages:MTLStageAll beforeStages:(MTLStageDispatch | MTLStageBlit) visibilityOptions:MTL4VisibilityOptionDevice];
  if (fence != nil) [import waitForFence:fence beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)];
  const MTLSize group = MTLSizeMake(16, 16, 1);
  const MTLSize grid640 = MTLSizeMake(kWidth, kHeight, 1);
  auto run = [&](id<MTL4ComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, NSUInteger table_index,
                 id<MTLTexture> t0, id<MTLTexture> t1, id<MTLTexture> t2, MTLSize grid,
                 id<MTLTexture> t3 = nil, id<MTLTexture> t4 = nil, MTLGPUAddress params_at = 0,
                 MTLGPUAddress buffer1 = 0, MTLGPUAddress buffer2 = 0) {
    id<MTL4ArgumentTable> table = _tables[slot * 8 + table_index];
    [table setAddress:(params_at != 0 ? params_at : params_address) atIndex:0];
    if (buffer1 != 0) [table setAddress:buffer1 atIndex:1];
    if (buffer2 != 0) [table setAddress:buffer2 atIndex:2];
    [table setTexture:t0.gpuResourceID atIndex:0];
    if (t1 != nil) [table setTexture:t1.gpuResourceID atIndex:1];
    if (t2 != nil) [table setTexture:t2.gpuResourceID atIndex:2];
    if (t3 != nil) [table setTexture:t3.gpuResourceID atIndex:3];
    if (t4 != nil) [table setTexture:t4.gpuResourceID atIndex:4];
    [enc setComputePipelineState:pso];
    [enc setArgumentTable:table];
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)
                 visibilityOptions:MTL4VisibilityOptionDevice];
  };
  run(import, _importColor, 0, color, current, nil, grid640);
  run(import, _importMotion, 1, motion, _motion[slot], nil, grid640);
  run(import, _importDepth, 2, depth, _depth[slot], nil, grid640);
  if (fullres) {
    // The interpolator gives no previous-colour texture, so the game's colour is retained per slot.
    [import copyFromTexture:color toTexture:_colorCopy[slot]];
    [import barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)
                    visibilityOptions:MTL4VisibilityOptionDevice];
  }
  [import endEncoding];

  // Runtime.
  NSFGFrameInfo frame{};
  frame.current_backbuffer_rgba8 = (__bridge void*)current;
  frame.previous_backbuffer_rgba8 = (__bridge void*)(do_reset ? current : _lastCurrent);
  frame.motion_rg16f = (__bridge void*)_motion[slot];
  frame.depth_r32f = (__bridge void*)_depth[slot];
  frame.generated_target_rgba8 = do_reset ? nullptr : (__bridge void*)_output[slot];
  frame.reset = do_reset;
  if (do_reset) {
    _runtimeFrameId = 0;
    NSFGSessionInvalidate(_session);
  } else {
    ++_runtimeFrameId;
  }
  frame.frame_id = _runtimeFrameId;
  NSFGFrameResult result{};
  const NSFGStatus status = NSFGSessionEvaluateMetal4(_session, &frame, (__bridge void*)command, &result);
  if (status != NSFG_STATUS_OK) {
    _needsReset = YES;
    _lastCurrent = nil;
    SetError(error, 30, [NSString stringWithFormat:@"FG runtime %s: %s", NSFGStatusName(status), NSFGSessionLastError(_session)]);
    return NO;
  }
  _lastCurrent = current;
  _needsReset = NO;

  // Export: the generated frame into the interpolator's output, or the real frame on a reset.
  id<MTL4ComputeCommandEncoder> exporter = [command computeCommandEncoder];
  if (exporter == nil) { _needsReset = YES; SetError(error, 20, @"Metal 4 compute encoder unavailable"); return NO; }
  [exporter barrierAfterQueueStages:MTLStageAll beforeStages:(MTLStageDispatch | MTLStageBlit) visibilityOptions:MTL4VisibilityOptionDevice];
  const MTLSize grid_out = MTLSizeMake(output.width, output.height, 1);
  NSFGTailResources tail{};
  const bool fullres_now = fullres && !do_reset && _lastColorCopy != nil &&
                           NSFGSessionCopyLastTailResources(_session, &tail) == NSFG_STATUS_OK;
  if (do_reset) {
    run(exporter, _passthrough, 3, color, output, nil, grid_out);
  } else if (fullres_now) {
    // Launch 55 at content size: candidate-warped surfaces from the previous and current frames.
    id<MTLBuffer> cand_a = (__bridge id<MTLBuffer>)tail.candidate_a_buffer;
    id<MTLBuffer> cand_b = (__bridge id<MTLBuffer>)tail.candidate_b_buffer;
    id<MTLBuffer> motion8 = (__bridge id<MTLBuffer>)tail.motion_half8_buffer;
    id<MTLBuffer> sigmoid = (__bridge id<MTLBuffer>)tail.sigmoid_lut_buffer;
    id<MTLTexture> blend = (__bridge id<MTLTexture>)tail.blend_rg16f_texture;
    run(exporter, _fullresCandidates, 4, _lastColorCopy, _colorCopy[slot], blend, MTLSizeMake(contentWidth, contentHeight, 1),
        _surfA, _surfB, fparams_address, cand_a.gpuAddress + tail.candidate_offset, cand_b.gpuAddress + tail.candidate_offset);
    // Launch 84 at output size: fine flows and the sigmoid blend over the two surfaces.
    run(exporter, _fullresBlend, 5, _surfA, _surfB, color, grid_out, output, nil, fparams_address,
        motion8.gpuAddress + tail.motion_offset, sigmoid.gpuAddress + tail.sigmoid_offset);
    ++_fullresFrames;
  } else {
    if (fullres) ++_fullresSkipped;
    run(exporter, _export, 3, _output[slot], color, output, grid_out);
  }
  if (fullres) _lastColorCopy = _colorCopy[slot];
  if (fence != nil) [exporter updateFence:fence afterEncoderStages:MTLStageDispatch];
  [exporter endEncoding];
  if (do_reset) ++_resets; else ++_generated;
  _lastDispatches = result.selected_launch_count;
  if (!_dumpDir.empty()) [self dumpFrame:_generated + _resets slot:slot command:command color:color output:output prev:do_reset ? current : _lastCurrentForDump];
  _lastCurrentForDump = current;
  return YES;
}

- (id<MTLTexture>)sharedCopyOf:(id<MTLTexture>)source {
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat width:source.width height:source.height mipmapped:NO];
  d.storageMode = MTLStorageModeShared;
  d.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> t = [_device newTextureWithDescriptor:d];
  if (t != nil) { [_residency addAllocation:t]; }
  return t;
}

- (void)setDumpUITexture:(id<MTLTexture>)ui { _dumpUISource = ui; }

- (void)dumpFrame:(uint64_t)frame slot:(NSUInteger)slot command:(id<MTL4CommandBuffer>)command color:(id<MTLTexture>)color output:(id<MTLTexture>)output prev:(id<MTLTexture>)prev {
  if (frame >= 60 && frame < 64) {
    const NSUInteger i = static_cast<NSUInteger>(frame - 60);
    _dumpColor[i] = [self sharedCopyOf:_color[slot]];
    _dumpPrev[i] = [self sharedCopyOf:prev];
    _dumpMotion[i] = [self sharedCopyOf:_motion[slot]];
    _dumpDepth[i] = [self sharedCopyOf:_depth[slot]];
    _dumpOut[i] = [self sharedCopyOf:_output[slot]];
    _dumpGameColor[i] = [self sharedCopyOf:color];
    _dumpGameOut[i] = [self sharedCopyOf:output];
    if (_dumpUISource != nil) _dumpGameUI[i] = [self sharedCopyOf:_dumpUISource];
    [_residency commit];
    id<MTL4ComputeCommandEncoder> enc = [command computeCommandEncoder];
    [enc barrierAfterQueueStages:MTLStageAll beforeStages:(MTLStageDispatch | MTLStageBlit) visibilityOptions:MTL4VisibilityOptionDevice];
    [enc copyFromTexture:_color[slot] toTexture:_dumpColor[i]];
    [enc copyFromTexture:prev toTexture:_dumpPrev[i]];
    [enc copyFromTexture:_motion[slot] toTexture:_dumpMotion[i]];
    [enc copyFromTexture:_depth[slot] toTexture:_dumpDepth[i]];
    [enc copyFromTexture:_output[slot] toTexture:_dumpOut[i]];
    [enc copyFromTexture:color toTexture:_dumpGameColor[i]];
    [enc copyFromTexture:output toTexture:_dumpGameOut[i]];
    if (_dumpGameUI[i] != nil) [enc copyFromTexture:_dumpUISource toTexture:_dumpGameUI[i]];
    [enc endEncoding];
  }
  if (frame == 72 && !_dumpWritten) {
    _dumpWritten = true;
    auto write = [&](id<MTLTexture> t, const char* name, NSUInteger i, NSUInteger bpp) {
      if (t == nil) return;
      std::vector<uint8_t> bytes(t.width * t.height * bpp);
      [t getBytes:bytes.data() bytesPerRow:t.width * bpp fromRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0];
      char path[1024];
      std::snprintf(path, sizeof path, "%s/f%lu-%s-%lux%lu-f%lu.bin", _dumpDir.c_str(), (unsigned long)(60 + i), name,
                    (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.pixelFormat);
      FILE* f = std::fopen(path, "wb");
      if (f) { std::fwrite(bytes.data(), 1, bytes.size(), f); std::fclose(f); }
    };
    for (NSUInteger i = 0; i < 4; ++i) {
      write(_dumpColor[i], "color", i, 4); write(_dumpPrev[i], "prev", i, 4); write(_dumpMotion[i], "motion", i, 4);
      write(_dumpDepth[i], "depth", i, 4); write(_dumpOut[i], "out", i, 4); write(_dumpGameColor[i], "gamecolor", i, 4);
      write(_dumpGameOut[i], "gameout", i, 4); write(_dumpGameUI[i], "gameui", i, 4);
    }
  }
}

@end
