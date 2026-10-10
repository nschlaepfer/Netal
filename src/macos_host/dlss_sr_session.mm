#import "dlss_sr_session.h"
#include "netal_sr_boundary_geometry.h"
#include "netal_sr_runtime_manifest.h"

#import <CommonCrypto/CommonDigest.h>

#include <map>
#include <string>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

NSErrorDomain const NeuralStudioDLSSSRSessionErrorDomain =
    @"NeuralStudioDLSSSRSessionError";

namespace {

constexpr NSUInteger kInputWidth = 1280;
constexpr NSUInteger kInputHeight = 720;
constexpr NSUInteger kOutputWidth = 1920;
constexpr NSUInteger kOutputHeight = 1080;
constexpr NSUInteger kExposureBytes = 4;
constexpr NSUInteger kCanary = 256;
constexpr NSUInteger kArena6Bytes = 11059200;
constexpr NSUInteger kArena7Bytes = 13271040;
constexpr NSUInteger kArena8Bytes = 8890560;
constexpr NSUInteger kWeightsBytes = 3545824;
constexpr NSUInteger kConstantsBytes = 524288;

NSData* Load(NSURL* root, NSString* relative, NSError** error);
void SetError(NSError** error, NSInteger code, NSString* description);
bool BundleHas(NSURL* root, NSString* relative);
struct Geometry;
void PatchRenderFields(NSMutableData* arguments, bool output_kernel, const Geometry& g);
bool SynthesizeGeometry(NSURL* root, Geometry* geometry, NSUInteger ow, NSUInteger oh, NSError** error);

// Resolution profile. Legacy bundles (no profile requested) keep the
// 1280x720 -> 1920x1080 geometry at the bundle root; profiles live under
// profiles/<WxH>/ (scripts/build-sr-profile.py) and carry the PWin extents,
// arena sizes and internal arrays of one output resolution plus the Hiluma
// templates captured at a reference render resolution.
struct ArrayGeometry {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t format = 16;
  std::uint32_t channels = 1;
};

struct Geometry {
  NSUInteger input_width = kInputWidth;
  NSUInteger input_height = kInputHeight;
  NSUInteger output_width = kOutputWidth;
  NSUInteger output_height = kOutputHeight;
  NSUInteger reference_width = kInputWidth;   // render size the Hiluma templates were captured at
  NSUInteger reference_height = kInputHeight;
  NSUInteger arena6_bytes = kArena6Bytes;
  NSUInteger arena7_bytes = kArena7Bytes;
  NSUInteger arena8_bytes = kArena8Bytes;
  ArrayGeometry arrays[6] = {{3840, 2176, 16, 1}, {3840, 2176, 16, 1}, {1920, 1088, 16, 4},
                             {1920, 1088, 16, 4}, {480, 288, 16, 4}, {1, 1, 16, 1}};
  MTLSize hiluma_input_grid = MTLSizeMake(120, 18, 1);
  MTLSize hiluma_output_grid = MTLSizeMake(120, 68, 1);
  NSString* prefix = @"";           // "" (legacy root) or "profiles/WxH/"
  bool profile = false;
  bool synthesized = false;
  NSMutableDictionary<NSString*, NSData*>* overrides = nil;  // synthesized artifacts by profile-relative path
  NSUInteger color_bytes() const { return input_width * input_height * 4 * 2; }
  NSUInteger motion_bytes() const { return input_width * input_height * 2 * 2; }
  NSUInteger depth_bytes() const { return input_width * input_height * 4; }
  NSUInteger output_bytes() const { return output_width * output_height * 4 * 2; }
};

bool ParseSize(const char* text, NSUInteger* width, NSUInteger* height) {
  if (text == nullptr || *text == '\0') return false;
  unsigned long w = 0, h = 0;
  if (std::sscanf(text, "%lux%lu", &w, &h) != 2 || w == 0 || h == 0) return false;
  *width = w;
  *height = h;
  return true;
}

std::uint16_t HalfBits(float value) {
  __fp16 half = static_cast<__fp16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

bool LoadGeometry(NSURL* root, Geometry* geometry, NSError** error) {
  const char* profile = std::getenv("NEURALSTUDIO_SR_PROFILE");
  NSUInteger profile_width = 0, profile_height = 0;
  if (!ParseSize(profile, &profile_width, &profile_height)) return true;  // legacy geometry
  NSString* prefix = [NSString stringWithFormat:@"profiles/%lux%lu/", static_cast<unsigned long>(profile_width),
                                                static_cast<unsigned long>(profile_height)];
  const char* force_synthesis = std::getenv("NEURALSTUDIO_SR_PROFILE_SYNTHESIZE");
  const bool synthesize = (force_synthesis != nullptr && *force_synthesis == '1') ||
                          !BundleHas(root, [prefix stringByAppendingString:@"manifest.json"]);
  if (synthesize) {
    if (!SynthesizeGeometry(root, geometry, profile_width, profile_height, error)) return false;
    NSUInteger render_width = 0, render_height = 0;
    geometry->input_width = geometry->reference_width;
    geometry->input_height = geometry->reference_height;
    if (ParseSize(std::getenv("NEURALSTUDIO_SR_RENDER"), &render_width, &render_height)) {
      geometry->input_width = render_width;
      geometry->input_height = render_height;
    }
    return true;
  }
  NSData* manifest_data = Load(root, [prefix stringByAppendingString:@"manifest.json"], error);
  if (manifest_data == nil) return false;
  NSDictionary* manifest = [NSJSONSerialization JSONObjectWithData:manifest_data options:0 error:error];
  if (![manifest isKindOfClass:NSDictionary.class] ||
      ![manifest[@"schema"] isEqual:@"neuralstudio.re9.sr_profile.v1"]) {
    SetError(error, 40, @"profile manifest invalid");
    return false;
  }
  NSArray* output = manifest[@"output"];
  NSArray* reference = manifest[@"render_reference"];
  NSDictionary* arenas = manifest[@"arenas"];
  NSArray* arrays = manifest[@"arrays"];
  if (output.count != 2 || reference.count != 2 || arenas == nil || arrays.count < 6) {
    SetError(error, 40, @"profile manifest incomplete");
    return false;
  }
  geometry->output_width = [output[0] unsignedIntegerValue];
  geometry->output_height = [output[1] unsignedIntegerValue];
  geometry->reference_width = [reference[0] unsignedIntegerValue];
  geometry->reference_height = [reference[1] unsignedIntegerValue];
  geometry->input_width = geometry->reference_width;
  geometry->input_height = geometry->reference_height;
  NSUInteger render_width = 0, render_height = 0;
  if (ParseSize(std::getenv("NEURALSTUDIO_SR_RENDER"), &render_width, &render_height)) {
    geometry->input_width = render_width;
    geometry->input_height = render_height;
  }
  geometry->arena6_bytes = [arenas[@"6"] unsignedIntegerValue];
  geometry->arena7_bytes = [arenas[@"7"] unsignedIntegerValue];
  geometry->arena8_bytes = [arenas[@"8"] unsignedIntegerValue];
  for (NSDictionary* array in arrays) {
    const NSUInteger id = [array[@"id"] unsignedIntegerValue];
    if (id < 1 || id > 6) continue;
    geometry->arrays[id - 1] = {[array[@"width"] unsignedIntValue], [array[@"height"] unsignedIntValue],
                                [array[@"format"] unsignedIntValue], [array[@"channels"] unsignedIntValue]};
  }
  if (geometry->output_width != profile_width || geometry->output_height != profile_height ||
      geometry->arena6_bytes == 0 || geometry->arena7_bytes == 0 || geometry->arena8_bytes == 0) {
    SetError(error, 40, @"profile geometry mismatch");
    return false;
  }
  NSArray* input_grid = manifest[@"hiluma_input"][@"grid"];
  NSArray* output_grid = manifest[@"hiluma_output"][@"grid"];
  if (input_grid.count != 3 || output_grid.count != 3) {
    SetError(error, 40, @"profile Hiluma grids missing");
    return false;
  }
  geometry->hiluma_input_grid = MTLSizeMake([input_grid[0] unsignedIntegerValue], [input_grid[1] unsignedIntegerValue],
                                            [input_grid[2] unsignedIntegerValue]);
  geometry->hiluma_output_grid = MTLSizeMake([output_grid[0] unsignedIntegerValue], [output_grid[1] unsignedIntegerValue],
                                             [output_grid[2] unsignedIntegerValue]);
  geometry->prefix = prefix;
  geometry->profile = true;
  return true;
}

NSData* LoadProfile(NSURL* root, const Geometry& geometry, NSString* relative, NSError** error) {
  NSData* override = geometry.overrides[relative];
  if (override != nil) return override;
  return Load(root, [geometry.prefix stringByAppendingString:relative], error);
}

NSUInteger RoundUp(NSUInteger value, NSUInteger multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

float Reciprocal(NSUInteger value) { return 1.0f / static_cast<float>(value); }

// Synthesize the profile for an output size that has no captured profile,
// from the captured 1080p profile, with the closed-form rules verified on
// seven RTX captures (scripts/verify-sr-profile-formulas.py,
// scripts/synthesize-sr-profile.py): extents = max(256, round_up(out/4, 32))
// per axis with a halving pyramid; arenas and tensor offsets from level areas;
// internal arrays; Hiluma grids and output-resolution words.
bool SynthesizeGeometry(NSURL* root, Geometry* geometry, NSUInteger ow, NSUInteger oh, NSError** error) {
  NSString* base = @"profiles/1920x1080/";
  NSData* manifest_data = Load(root, [base stringByAppendingString:@"manifest.json"], error);
  if (manifest_data == nil) return false;
  NSDictionary* manifest = [NSJSONSerialization JSONObjectWithData:manifest_data options:0 error:error];
  NSArray* stages = manifest[@"stages"];
  if (![manifest isKindOfClass:NSDictionary.class] || stages.count != 11) {
    SetError(error, 41, @"base profile for synthesis invalid");
    return false;
  }
  NSUInteger extent_w = MAX(NSUInteger(256), RoundUp((ow + 3) / 4, 32));
  NSUInteger extent_h = MAX(NSUInteger(256), RoundUp((oh + 3) / 4, 32));
  NSUInteger level_w[6], level_h[6], area[6];
  for (int level = 0; level < 6; ++level) {
    level_w[level] = level == 0 ? extent_w : (level_w[level - 1] + 1) / 2;
    level_h[level] = level == 0 ? extent_h : (level_h[level - 1] + 1) / 2;
    area[level] = level_w[level] * level_h[level];
  }
  static const int kLevel[11] = {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0};
  // Tensor offsets per (stage, argument offset, allocation): level-area multiples.
  struct Rule { int stage; std::uint32_t argument; std::uint32_t allocation; int level; NSUInteger factor; };
  static const Rule kRules[] = {
      {0, 32, 8, 3, 20}, {0, 48, 8, 3, 20}, {1, 32, 6, 0, 8}, {2, 32, 7, 1, 12}, {3, 32, 6, 2, 16},
      {4, 32, 7, 3, 20}, {6, 16, 7, 3, 20}, {7, 16, 6, 2, 16}, {8, 16, 7, 1, 12}, {9, 16, 6, 0, 8},
      {10, 16, 8, 3, 20}, {10, 40, 7, 0, 32}};
  NSMutableDictionary<NSString*, NSData*>* overrides = [NSMutableDictionary dictionary];
  for (NSUInteger index = 0; index < 11; ++index) {
    NSDictionary* stage = stages[index];
    NSString* directory = stage[@"directory"];
    NSMutableData* config = [Load(root, [base stringByAppendingFormat:@"%@/replay-config.bin", directory], error) mutableCopy];
    NSMutableData* arguments = [Load(root, [base stringByAppendingFormat:@"%@/arguments-template.bin", directory], error) mutableCopy];
    if (config.length != 296 || arguments.length < 64) {
      SetError(error, 41, @"base profile stage artifacts invalid");
      return false;
    }
    auto* config_bytes = static_cast<std::uint8_t*>(config.mutableBytes);
    auto* argument_bytes = static_cast<std::uint8_t*>(arguments.mutableBytes);
    const std::uint32_t extent[2] = {static_cast<std::uint32_t>(level_w[kLevel[index]]),
                                     static_cast<std::uint32_t>(level_h[kLevel[index]])};
    std::memcpy(argument_bytes, extent, 8);
    std::uint32_t shift[2];
    std::memcpy(shift, argument_bytes + 56, 8);
    std::uint32_t grid[3];
    std::memcpy(grid, config_bytes + 140, 12);
    grid[0] = (extent[0] + shift[0] + 7) / 8;
    grid[1] = (extent[1] + shift[1] + 7) / 8;
    std::memcpy(config_bytes + 140, grid, 12);
    std::uint32_t count = 0;
    std::memcpy(&count, config_bytes + 164, 4);
    for (std::uint32_t r = 0; r < count && r < 8; ++r) {
      std::uint32_t argument = 0, allocation = 0;
      std::memcpy(&argument, config_bytes + 168 + r * 16, 4);
      std::memcpy(&allocation, config_bytes + 172 + r * 16, 4);
      if (allocation == 9) continue;  // weights: constant across profiles
      std::uint64_t offset = 0;
      for (const Rule& rule : kRules) {
        if (rule.stage == static_cast<int>(index) && rule.argument == argument && rule.allocation == allocation) {
          offset = static_cast<std::uint64_t>(area[rule.level]) * rule.factor;
        }
      }
      std::memcpy(config_bytes + 176 + r * 16, &offset, 8);
    }
    overrides[[directory stringByAppendingString:@"/replay-config.bin"]] = config;
    overrides[[directory stringByAppendingString:@"/arguments-template.bin"]] = arguments;
  }
  NSMutableData* input = [Load(root, [base stringByAppendingString:@"abi/hiluma-input-reset-template.bin"], error) mutableCopy];
  NSMutableData* output = [Load(root, [base stringByAppendingString:@"abi/hiluma-output-reset-template.bin"], error) mutableCopy];
  NSData* exposure = Load(root, [base stringByAppendingString:@"abi/exposure-reset-template.bin"], error);
  if (input.length != 248 || output.length != 328 || exposure.length != 16) {
    SetError(error, 41, @"base profile Hiluma templates invalid");
    return false;
  }
  auto* in = static_cast<std::uint8_t*>(input.mutableBytes);
  auto* out = static_cast<std::uint8_t*>(output.mutableBytes);
  auto put_u = [](std::uint8_t* b, NSUInteger o, NSUInteger v) { std::uint32_t w = static_cast<std::uint32_t>(v); std::memcpy(b + o, &w, 4); };
  auto put_f = [](std::uint8_t* b, NSUInteger o, float v) { std::memcpy(b + o, &v, 4); };
  put_u(in, 64, extent_w);
  put_f(in, 120, Reciprocal(ow)); put_f(in, 124, Reciprocal(RoundUp(oh, 16))); put_f(in, 128, Reciprocal(4 * extent_w));
  put_f(in, 132, Reciprocal(4 * extent_h)); put_f(in, 136, Reciprocal(ow)); put_f(in, 140, Reciprocal(oh));
  put_u(out, 80, extent_w); put_u(out, 84, extent_h); put_u(out, 136, ow); put_u(out, 140, oh);
  put_f(out, 144, Reciprocal(ow)); put_f(out, 148, Reciprocal(RoundUp(oh, 16))); put_f(out, 152, Reciprocal(ow)); put_f(out, 156, Reciprocal(oh));
  overrides[@"abi/hiluma-input-reset-template.bin"] = input;
  overrides[@"abi/hiluma-output-reset-template.bin"] = output;
  overrides[@"abi/exposure-reset-template.bin"] = exposure;
  geometry->output_width = ow;
  geometry->output_height = oh;
  geometry->arena6_bytes = area[0] * 80;
  geometry->arena7_bytes = area[0] * 96;
  geometry->arena8_bytes = area[0] * 64 + area[3] * 20;
  geometry->arrays[0] = {static_cast<std::uint32_t>(2 * ow), static_cast<std::uint32_t>(2 * RoundUp(oh, 16)), 16, 1};
  geometry->arrays[1] = geometry->arrays[0];
  geometry->arrays[2] = {static_cast<std::uint32_t>(ow), static_cast<std::uint32_t>(RoundUp(oh, 16)), 16, 4};
  geometry->arrays[3] = geometry->arrays[2];
  geometry->arrays[4] = {static_cast<std::uint32_t>(extent_w), static_cast<std::uint32_t>(extent_h), 16, 4};
  geometry->arrays[5] = {1, 1, 16, 1};
  geometry->hiluma_input_grid = MTLSizeMake(extent_w / 4, extent_h / 16, 1);
  geometry->hiluma_output_grid = MTLSizeMake((ow + 15) / 16, (oh + 15) / 16, 1);
  // Hiluma templates come from the 1280x720 capture; the render words are
  // re-derived at encode time (reference != render).
  geometry->reference_width = 1280;
  geometry->reference_height = 720;
  geometry->input_width = geometry->reference_width;
  geometry->input_height = geometry->reference_height;
  // The output size changed even when render size did not. Re-derive render/output ratios
  // and both per-axis limits now; encode-time patching handles a later render-size override.
  PatchRenderFields(input, false, *geometry);
  PatchRenderFields(output, true, *geometry);
  geometry->overrides = overrides;
  geometry->prefix = base;
  geometry->profile = true;
  geometry->synthesized = true;
  // NEURALSTUDIO_SR_EXPORT_SYNTH_DIR=<dir>: write the synthesized profile artefacts to disk so the batched
  // pipelines can be generated for this resolution offline (tools/pwin_pipeline reads the PWin arguments and
  // replay configs). The session stays the single source of truth for the synthesis rules.
  if (const char* export_dir = std::getenv("NEURALSTUDIO_SR_EXPORT_SYNTH_DIR")) {
    NSFileManager* fm = [NSFileManager defaultManager];
    for (NSString* relative in overrides) {
      NSString* path = [[NSString stringWithUTF8String:export_dir] stringByAppendingPathComponent:relative];
      [fm createDirectoryAtPath:[path stringByDeletingLastPathComponent]
     withIntermediateDirectories:YES attributes:nil error:nil];
      [overrides[relative] writeToFile:path atomically:NO];
    }
    std::fprintf(stderr, "neuralstudio: exported %lu synthesized artefacts for %lux%lu to %s\n",
                 static_cast<unsigned long>(overrides.count), static_cast<unsigned long>(ow),
                 static_cast<unsigned long>(oh), export_dir);
  }
  return true;
}

// Re-derive the render-resolution words of a Hiluma template for the session's
// render size (validated against RTX captures at 960x540, 1280x720 and
// 1706x960 -> 1080p, see checkpoint 2026-09-01 profile_generalisation).
void PatchRenderFields(NSMutableData* arguments, bool output_kernel, const Geometry& g) {
  auto* bytes = static_cast<std::uint8_t*>(arguments.mutableBytes);
  const auto ratios = netal::RenderRatios(static_cast<std::uint32_t>(g.input_width), static_cast<std::uint32_t>(g.input_height),
                                           static_cast<std::uint32_t>(g.output_width), static_cast<std::uint32_t>(g.output_height));
  const float x = ratios.x;
  const float y = ratios.y;
  const float x_half = x / 2.0f;
  const float y_half = y / 2.0f;
  const std::uint32_t w1 = static_cast<std::uint32_t>(g.input_width - 1);
  const std::uint32_t h1 = static_cast<std::uint32_t>(g.input_height - 1);
  auto put_f = [&](NSUInteger offset, float value) { std::memcpy(bytes + offset, &value, 4); };
  auto put_u = [&](NSUInteger offset, std::uint32_t value) { std::memcpy(bytes + offset, &value, 4); };
  if (!output_kernel) {
    put_f(8, 1.0f / x); put_f(12, 1.0f / y);
    put_f(40, x); put_f(44, y); put_f(48, x); put_f(52, y);
    for (NSUInteger offset : {80u, 96u, 112u}) put_u(offset, w1);
    for (NSUInteger offset : {84u, 100u, 116u}) put_u(offset, h1);
  } else {
    put_f(0, 1.0f / x); put_f(4, 1.0f / y);
    put_f(40, x); put_f(44, y); put_f(48, x_half); put_f(52, y_half); put_f(56, x); put_f(60, y);
    for (NSUInteger offset : {104u, 120u, 216u}) put_u(offset, w1);
    for (NSUInteger offset : {108u, 124u, 220u}) put_u(offset, h1);
    const std::uint16_t lo = HalfBits(8.0f);
    const auto limits = netal::BoundaryLimits(static_cast<std::uint32_t>(g.input_width), static_cast<std::uint32_t>(g.input_height),
                                              static_cast<std::uint32_t>(g.output_width), static_cast<std::uint32_t>(g.output_height));
    const std::uint16_t hi_x = HalfBits(limits.x);
    const std::uint16_t hi_y = HalfBits(limits.y);
    std::memcpy(bytes + 188, &lo, 2);
    std::memcpy(bytes + 190, &hi_x, 2);
    std::memcpy(bytes + 192, &hi_y, 2);
  }
}
constexpr NSUInteger kHalfLutBytes = 65536 * 2;
constexpr NSUInteger kRsqrtRomBytes = 1u << 22u;
constexpr NSUInteger kRcpRomBytes = 1u << 21u;
constexpr NSUInteger kMufuTraceBytes = (1u << 20u) * sizeof(std::uint32_t);
constexpr NSUInteger kTextureWeightTableBytes = 257u * 257u * 4u * sizeof(std::uint32_t);
constexpr const char* kBundleRoot =
    "c6e2a9d9f6da012092e9957a35d47ebc7645b5af9d6669acbb625919bbb4f39b";
constexpr const char* kImplementationRoot =
    "04cbe24e1bbf6326fbd02401af6f1f18e2c56060efcece7954048472af2908db";

#pragma pack(push, 1)
struct Relocation {
  std::uint32_t argument_offset;
  std::uint32_t allocation_id;
  std::uint64_t allocation_offset;
};

struct ReplayConfig {
  char magic[8];
  std::uint32_t version;
  char entry[64];
  char debug_symbol[64];
  std::uint32_t grid[3];
  std::uint32_t block[3];
  std::uint32_t relocation_count;
  Relocation relocations[8];
};
#pragma pack(pop)
static_assert(sizeof(ReplayConfig) == 296);

struct ResourceDescriptor {
  std::uint64_t data;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t format;
  std::uint32_t channels;
};
static_assert(sizeof(ResourceDescriptor) == 24);

struct GuardedBuffer {
  __strong id<MTLBuffer> buffer = nil;
  NSUInteger bytes = 0;
};

struct DescribedBuffer {
  GuardedBuffer storage;
  __strong id<MTLBuffer> descriptor = nil;
};

// Per-frame PWin window-shift schedule (u32 pair at argument offsets 56/60).
// Derived from the RTX 32-frame temporal capture
// work/rtx-dlss-sr-temporal-32frame-20260901-v1 by
// scripts/derive-pwin-shift-schedule.py: period 8 in the frame index since
// reset; enc0-2 and dec0-2 stay at the template value (4,4).
constexpr NSUInteger kPwinShiftPeriod = 8;
struct PwinShift { std::uint32_t x; std::uint32_t y; };
constexpr PwinShift kPwinShiftA[kPwinShiftPeriod] = {  // enc3, dec3
    {4, 4}, {0, 4}, {4, 4}, {0, 4}, {0, 0}, {4, 0}, {0, 0}, {4, 0}};
constexpr PwinShift kPwinShiftB[kPwinShiftPeriod] = {  // enc4, dec4
    {4, 0}, {6, 0}, {4, 4}, {6, 4}, {2, 6}, {0, 6}, {2, 2}, {0, 2}};
constexpr PwinShift kPwinShiftC[kPwinShiftPeriod] = {  // dec5
    {0, 2}, {5, 6}, {4, 0}, {1, 4}, {7, 5}, {2, 1}, {3, 7}, {6, 3}};

const PwinShift* PwinShiftTable(int launch_position) {
  switch (launch_position) {
    case 5: case 9: return kPwinShiftA;
    case 6: case 8: return kPwinShiftB;
    case 7: return kPwinShiftC;
    default: return nullptr;
  }
}

struct Stage {
  ReplayConfig config{};
  __strong NSData* argument_template = nil;
  __strong id<MTLBuffer> arguments = nil;
  // arguments_by_phase[frame_index % kPwinShiftPeriod]; all entries alias
  // `arguments` for stages whose shift does not vary.
  std::vector<__strong id<MTLBuffer>> arguments_by_phase;
  // Launch grid per phase: ceil((width + shift_x) / 8) x ceil((height + shift_y) / 8),
  // verified against every launch of the RTX 32-frame capture.
  std::vector<MTLSize> grid_by_phase;
  __strong id<MTLComputePipelineState> pipeline = nil;
  int launch_position = 0;
  bool exact_math = false;
  NSUInteger scratch_bytes_per_threadgroup = 0;  // dispatch sidecar: device exchange for split lowerings
  // dispatch sidecar "windows_per_threadgroup": persistent threadgroups loop over K windows;
  // the dispatch grid becomes ceil(gx*gy/K) x 1 x 1 and buffer 17 carries {gx, gy, K, 0}.
  NSUInteger windows_per_threadgroup = 1;
  std::vector<MTLSize> dispatch_by_phase;
  std::vector<__strong id<MTLBuffer>> window_grid_by_phase;
  // Batched pipeline replacement (bundle pipeline/<entry>.json, enabled by
  // NEURALSTUDIO_SR_PWIN_PIPELINE=1): a list of dispatches run instead of the monolithic kernel.
  struct PipelineDispatch {
    __strong id<MTLComputePipelineState> pipeline = nil;
    MTLSize grid{1, 1, 1}, block{1, 1, 1};
    std::vector<std::pair<NSUInteger, std::pair<__strong id<MTLBuffer>, NSUInteger>>> bindings;  // index -> (buffer, offset)
    __strong id<MTLBuffer> constants = nil;  // "bytes" argument, static
    NSUInteger constants_index = 0;
  };
  // pipeline_dispatches_by_phase[frame_index % kPwinShiftPeriod]: the five deep layers shift their window
  // grid every frame, so their batched pipelines are generated once per phase
  // (bundle pipeline/<entry>.phase<N>.json); layers without a shift have all entries aliasing phase 0.
  std::vector<std::vector<PipelineDispatch>> pipeline_dispatches_by_phase;
};

// Optional per-dispatch GPU timestamps (NEURALSTUDIO_SR_DISPATCH_TIMING=1):
// every compute encoder samples the timestamp counter at its start and end.
struct DispatchTiming {
  __strong id<MTLCounterSampleBuffer> samples = nil;
  NSUInteger next_index = 0;
  NSUInteger capacity = 0;
};
DispatchTiming* g_dispatch_timing = nullptr;

// Metal 4 support: per-frame rings of argument tables and small descriptor
// buffers (deep enough for the frames a game keeps in flight) and one residency
// set holding every session buffer.
// kRingTables must exceed the dispatches in one frame. The batched 1080p profile encodes 97 (94
// layer dispatches plus input, output and the exposure copy), so 64 wrapped every frame: dispatches
// 65..97 rewrote tables 0..32 while 1..33 were still queued.
//
// That has been harmless only because a device-scope barrier follows every dispatch, so exactly one
// is ever in flight and a table is free long before it is reused. It is not a property to keep
// relying on: thinning those barriers - the obvious next optimisation - would let dispatches overlap
// and turn the wrap into a live race, and the receipts would catch it only intermittently. Sized
// here with headroom, and the overflow is reported rather than silently wrapped.
constexpr NSUInteger kRingSlots = 6, kRingBuffers = 10, kRingTables = 160, kRingBytes = 4096;

struct SessionState {
  __strong id<MTLDevice> device = nil;
  Geometry geometry;
  __strong NSURL* bundle_url = nil;
  __strong NSString* runtime_root = nil;
  std::mutex mutex;
  std::atomic_bool consumed{false};
  bool temporal_state_initialized = false;
  __strong id<MTLResidencySet> residency = nil;
  std::vector<__strong id<MTL4ArgumentTable>> argument_tables;  // kRingSlots x kRingTables
  std::vector<__strong id<MTLBuffer>> frame_ring;                // kRingSlots x kRingBuffers x kRingBytes
  std::uint64_t frames_started = 0;
  NSUInteger ring_slot = 0, ring_next = 0, table_next = 0;
  NSUInteger pipeline_layers_total = 0, pipeline_layers_present = 0;
  std::string pipeline_layers_missing;
  // The frame ring holds CPU-written (Shared) argument and descriptor buffers. Metal's hazard
  // tracking covers GPU-GPU access, not a CPU memcpy into a buffer an in-flight command buffer
  // is still reading, so slot N may not be rewritten until frame N - kRingSlots has completed.
  dispatch_semaphore_t ring_slots = nil;
  std::atomic<std::uint64_t> ring_waits{0}, ring_timeouts{0};
  std::atomic<int> frames_in_flight{0};
  int frames_in_flight_max = 0;
  std::uint64_t encoded_frame_count = 0;
  float previous_jitter_x = 0.0f;

  GuardedBuffer arena6;
  GuardedBuffer arena7;
  GuardedBuffer arena8;
  GuardedBuffer weights;
  GuardedBuffer constants;
  DescribedBuffer internal3;
  DescribedBuffer internal1;
  DescribedBuffer internal5;
  DescribedBuffer internal4;
  DescribedBuffer internal2;
  DescribedBuffer internal_exposure;

  __strong id<MTLBuffer> ex2_lut = nil;
  __strong id<MTLBuffer> half_rcp_lut = nil;
  __strong id<MTLBuffer> rsqrt_rom = nil;
  __strong id<MTLBuffer> rcp_rom = nil;
  __strong id<MTLBuffer> mufu_lg2_trace = nil;
  __strong id<MTLBuffer> mufu_ex2_trace = nil;
  __strong id<MTLBuffer> mufu_lg2_keys = nil;
  __strong id<MTLBuffer> mufu_lg2_values = nil;
  __strong id<MTLBuffer> mufu_ex2_keys = nil;
  __strong id<MTLBuffer> mufu_ex2_values = nil;
  __strong id<MTLBuffer> mufu_lg2_misses = nil;
  __strong id<MTLBuffer> mufu_ex2_misses = nil;
  __strong id<MTLBuffer> texture_weight_table = nil;
  __strong id<MTLBuffer> temporal_scalar_debug = nil;
  __strong id<MTLBuffer> temporal_trace = nil;  // f32 register trace (buffer 12)
  std::vector<id<MTLBuffer>> pwin_stats;  // NEURALSTUDIO_SR_PWIN_STATS=1: per-stage counters (buffer 13)
  __strong id<MTLBuffer> pwin_scratch = nil;  // per-threadgroup MMA staging for hardware lowerings (buffer 14)
  std::vector<__strong id<MTLBuffer>> pipeline_weights;   // pipeline/<file>.bin weight blobs (one per pipeline stage)
  __strong id<MTLBuffer> pipeline_scratch = nil;           // shared intermediate tensors of the batched pipelines
  __strong id<MTLBuffer> weights_f32_a = nil;  // weights as float, 512-byte tiles reordered to row-major A tiles (buffer 15)
  __strong id<MTLBuffer> weights_f32_b = nil;  // same, B-tile order (buffer 16)
  DispatchTiming dispatch_timing;
  __strong id<MTLComputePipelineState> input_pipeline = nil;
  __strong id<MTLComputePipelineState> output_pipeline = nil;
  __strong id<MTLComputePipelineState> depth_inverted_input_pipeline = nil;
  __strong id<MTLComputePipelineState> depth_inverted_output_pipeline = nil;
  __strong id<MTLComputePipelineState> exposure_pipeline = nil;
  __strong NSData* input_template = nil;
  __strong NSData* output_template = nil;
  __strong NSData* exposure_template = nil;
  std::vector<Stage> pwin_stages;
  std::vector<__strong id<MTLBuffer>> retained_frame_buffers;
  std::vector<__strong id<MTLBuffer>> arena6_stage_snapshots;
  std::vector<__strong id<MTLBuffer>> arena7_stage_snapshots;
  std::vector<__strong id<MTLBuffer>> arena8_stage_snapshots;
};

void SetError(NSError** error, NSInteger code, NSString* description) {
  if (error == nullptr) return;
  *error = [NSError errorWithDomain:NeuralStudioDLSSSRSessionErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

NSString* Sha256(NSData* data) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
  CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), digest);
  NSMutableString* result = [NSMutableString stringWithCapacity:64];
  for (unsigned char byte : digest) [result appendFormat:@"%02x", byte];
  return result;
}

NSData* Load(NSURL* root, NSString* relative, NSError** error) {
  NSURL* url = [root URLByAppendingPathComponent:relative];
  NSData* data = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:error];
  return data;
}

bool BundleHas(NSURL* root, NSString* relative) {
  NSURL* url = [root URLByAppendingPathComponent:relative];
  return [NSFileManager.defaultManager fileExistsAtPath:url.path];
}

// Exact overlay: the RTX-verified kernel variants are shipped next to the
// immutable bundle files as <name>_exact.metallib (built by
// scripts/build-re9-sr-exact-overlay.py). They are the defaults when present;
// environment overrides still win for candidate work.
NSString* ExactLibrary(NSURL* root, NSString* legacy, NSString* exact, const char* override_env) {
  const char* override_value = override_env == nullptr ? nullptr : std::getenv(override_env);
  if (override_value != nullptr && *override_value != '\0') {
    return [NSString stringWithUTF8String:override_value];
  }
  NSString* relative = [NSString stringWithFormat:@"metallib/%@.metallib", exact];
  return BundleHas(root, relative) ? exact : legacy;
}

bool ValidateRuntimeBundle(NSURL* root, NSString** validated_root, NSError** error) {
  NSData* manifest_data = Load(root, @"manifest.json", error);
  if (manifest_data == nil) return false;
  NSDictionary* manifest =
      [NSJSONSerialization JSONObjectWithData:manifest_data options:0 error:error];
  if ([manifest isKindOfClass:NSDictionary.class] && [manifest[@"schema"] isEqual:@"netal.sr.runtime.v2"]) {
    NSString* reason=nil;
    if (!NetalValidateSRManifestV2(root,manifest,&reason)) { SetError(error,1,reason); return false; }
    *validated_root=manifest[@"asset_root_sha256"];
    return true;
  }
  if (![manifest isKindOfClass:NSDictionary.class] ||
      ![manifest[@"schema"] isEqual:@"neuralstudio.re9.exact_sr_immutable_runtime_bundle.v1"] ||
      ![manifest[@"bundle_root_sha256"] isEqual:@(kBundleRoot)] ||
      ![manifest[@"implementation_root_sha256"] isEqual:@(kImplementationRoot)]) {
    SetError(error, 1, @"runtime bundle manifest identity mismatch");
    return false;
  }
  NSDictionary* claims = manifest[@"claims"];
  if (![claims isKindOfClass:NSDictionary.class] ||
      ![claims[@"bundle_identity_proven"] boolValue] ||
      [claims[@"live_texture_contract_proven"] boolValue] ||
      [claims[@"in_game_model_execution_proven"] boolValue]) {
    SetError(error, 2, @"runtime bundle claim boundary mismatch");
    return false;
  }
  NSArray* files = manifest[@"files"];
  if (![files isKindOfClass:NSArray.class] || files.count!=48) { SetError(error,3,@"legacy file inventory invalid");return false; }
  for(id row in files)if(![row isKindOfClass:NSDictionary.class] || ![row[@"path"] isKindOfClass:NSString.class]) { SetError(error,3,@"legacy file row invalid");return false; }
  NSMutableDictionary* identity=[manifest mutableCopy];
  identity[@"files"]=[files sortedArrayUsingComparator:^NSComparisonResult(NSDictionary* a,NSDictionary* b) { return [a[@"path"] compare:b[@"path"]]; }];
  [identity removeObjectForKey:@"bundle_root_sha256"];
  [identity removeObjectForKey:@"ingest"]; // local installation receipt, not the historical core identity
  NSData* canonical=[NSJSONSerialization dataWithJSONObject:identity options:(NSJSONWritingSortedKeys|NSJSONWritingWithoutEscapingSlashes) error:error];
  if (canonical==nil || ![Sha256(canonical) isEqual:@(kBundleRoot)]) {
    SetError(error,3,@"legacy manifest contents do not match its pinned root");return false;
  }
  if (![files isKindOfClass:NSArray.class] || files.count != 48) {
    SetError(error, 3, @"runtime bundle file ledger mismatch");
    return false;
  }
  for (NSDictionary* row in files) {
    NSString* relative = row[@"path"];
    if (![relative isKindOfClass:NSString.class] || relative.length == 0 ||
        [relative hasPrefix:@"/"] || [relative containsString:@".."] ||
        [relative.lowercaseString containsString:@"expected"] ||
        [relative.lowercaseString containsString:@"capture"]) {
      SetError(error, 4, @"unsafe runtime bundle path");
      return false;
    }
    NSData* data = Load(root, relative, error);
    if (data == nil || data.length != [row[@"bytes"] unsignedLongLongValue] ||
        ![Sha256(data) isEqual:row[@"sha256"]]) {
      SetError(error, 5, [@"runtime artifact mismatch: " stringByAppendingString:relative]);
      return false;
    }
    NSArray* offsets = row[@"sanitized_pointer_offsets"];
    if ([offsets isKindOfClass:NSArray.class]) {
      const auto* bytes = static_cast<const std::uint8_t*>(data.bytes);
      for (NSNumber* value in offsets) {
        const NSUInteger offset = value.unsignedIntegerValue;
        if (offset + 8 > data.length) {
          SetError(error, 6, @"sanitized pointer mask outside template");
          return false;
        }
        for (NSUInteger index = 0; index < 8; ++index) {
          if (bytes[offset + index] != 0) {
            SetError(error, 7, @"captured pointer survived runtime sanitization");
            return false;
          }
        }
      }
    }
  }
  *validated_root=@(kBundleRoot);
  return true;
}

GuardedBuffer MakeGuarded(id<MTLDevice> device, NSUInteger bytes,
                          std::uint8_t fill, const void* initial = nullptr) {
  GuardedBuffer result;
  result.bytes = bytes;
  result.buffer = [device newBufferWithLength:bytes + 2 * kCanary
                                      options:MTLResourceStorageModeShared];
  if (result.buffer == nil) return result;
  std::memset(result.buffer.contents, 0xa5, result.buffer.length);
  std::memset(static_cast<std::uint8_t*>(result.buffer.contents) + kCanary, fill, bytes);
  if (initial != nullptr) {
    std::memcpy(static_cast<std::uint8_t*>(result.buffer.contents) + kCanary,
                initial, bytes);
  }
  return result;
}

std::uint64_t Address(const GuardedBuffer& buffer, NSUInteger offset = 0) {
  return buffer.buffer.gpuAddress + kCanary + offset;
}

void Fill(GuardedBuffer& buffer, std::uint8_t value) {
  std::memset(static_cast<std::uint8_t*>(buffer.buffer.contents) + kCanary,
              value, buffer.bytes);
}

// Diagnostic-only: how many times to encode the output dispatch. One unless asked, so nothing
// outside a measurement run sees any difference.
NSUInteger OutputRepeat() {
  static const NSUInteger value = [] {
    const char* raw = std::getenv("NEURALSTUDIO_SR_OUTPUT_REPEAT");
    if (raw == nullptr || *raw == '\0') return NSUInteger(1);
    long parsed = std::strtol(raw, nullptr, 10);
    return parsed >= 1 && parsed <= 256 ? NSUInteger(parsed) : NSUInteger(1);
  }();
  return value;
}

id<MTLBuffer> Descriptor(id<MTLDevice> device, std::uint64_t address,
                         std::uint32_t width, std::uint32_t height,
                         std::uint32_t format, std::uint32_t channels) {
  ResourceDescriptor descriptor = {address, width, height, format, channels};
  return [device newBufferWithBytes:&descriptor
                             length:sizeof(descriptor)
                            options:MTLResourceStorageModeShared];
}

DescribedBuffer MakeDescribed(id<MTLDevice> device, std::uint32_t width,
                              std::uint32_t height, std::uint32_t format,
                              std::uint32_t channels, std::uint8_t fill) {
  DescribedBuffer result;
  result.storage = MakeGuarded(
      device, static_cast<NSUInteger>(width) * height * channels * (format / 8), fill);
  if (result.storage.buffer != nil) {
    result.descriptor = Descriptor(device, Address(result.storage), width, height,
                                   format, channels);
  }
  return result;
}

void Patch64(NSMutableData* data, NSUInteger offset, std::uint64_t value) {
  std::memcpy(static_cast<std::uint8_t*>(data.mutableBytes) + offset, &value, 8);
}

id<MTLBuffer> BufferFromData(id<MTLDevice> device, NSData* data) {
  return [device newBufferWithBytes:data.bytes
                             length:data.length
                            options:MTLResourceStorageModeShared];
}

// Per-frame small buffers come from the ring when it exists (resident for
// Metal 4, no per-frame allocation); otherwise a fresh shared buffer.
id<MTLBuffer> FrameScratch(SessionState* state, const void* bytes, NSUInteger length) {
  if (state->frame_ring.empty() || length > kRingBytes || state->ring_next >= kRingBuffers) {
    return [state->device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> buffer = state->frame_ring[state->ring_slot * kRingBuffers + state->ring_next++];
  std::memcpy(buffer.contents, bytes, length);
  return buffer;
}

id<MTLBuffer> FrameDescriptor(SessionState* state, std::uint64_t address, std::uint32_t width,
                              std::uint32_t height, std::uint32_t format, std::uint32_t channels) {
  ResourceDescriptor descriptor = {address, width, height, format, channels};
  return FrameScratch(state, &descriptor, sizeof(descriptor));
}

void BuildMetal4Support(SessionState* state) {
  id<MTLDevice> device = state->device;
  if (![device supportsFamily:MTLGPUFamilyMetal4]) return;
  MTL4ArgumentTableDescriptor* table_descriptor = [[MTL4ArgumentTableDescriptor alloc] init];
  table_descriptor.maxBufferBindCount = 17;
  table_descriptor.maxTextureBindCount = 0;
  for (NSUInteger index = 0; index < kRingSlots * kRingTables; ++index) {
    NSError* table_error = nil;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&table_error];
    if (table == nil) return;
    state->argument_tables.push_back(table);
  }
  state->ring_slots = dispatch_semaphore_create(static_cast<long>(kRingSlots));
  for (NSUInteger index = 0; index < kRingSlots * kRingBuffers; ++index) {
    state->frame_ring.push_back([device newBufferWithLength:kRingBytes options:MTLResourceStorageModeShared]);
  }
  MTLResidencySetDescriptor* residency_descriptor = [[MTLResidencySetDescriptor alloc] init];
  residency_descriptor.initialCapacity = 512;
  NSError* residency_error = nil;
  id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&residency_error];
  if (residency == nil) return;
  auto add = [&](id<MTLBuffer> buffer) { if (buffer != nil) [residency addAllocation:buffer]; };
  add(state->arena6.buffer); add(state->arena7.buffer); add(state->arena8.buffer);
  add(state->weights.buffer); add(state->constants.buffer);
  add(state->ex2_lut); add(state->half_rcp_lut); add(state->rsqrt_rom); add(state->rcp_rom);
  add(state->mufu_lg2_trace); add(state->mufu_ex2_trace); add(state->mufu_lg2_keys); add(state->mufu_lg2_values);
  add(state->mufu_ex2_keys); add(state->mufu_ex2_values); add(state->mufu_lg2_misses); add(state->mufu_ex2_misses);
  add(state->texture_weight_table); add(state->temporal_scalar_debug); add(state->temporal_trace);
  add(state->pwin_scratch); add(state->weights_f32_a); add(state->weights_f32_b);
  for (const Stage& stage : state->pwin_stages) {
    for (id<MTLBuffer> buffer : stage.window_grid_by_phase) {
      if (buffer != nil) add(buffer);
    }
    for (const std::vector<Stage::PipelineDispatch>& phase : stage.pipeline_dispatches_by_phase)
      for (const Stage::PipelineDispatch& d : phase) add(d.constants);
  }
  for (id<MTLBuffer> w : state->pipeline_weights) add(w);
  add(state->pipeline_scratch);
  for (id<MTLBuffer> buffer : state->pwin_stats) add(buffer);
  for (const Stage& stage : state->pwin_stages) {
    add(stage.arguments);
    for (id<MTLBuffer> buffer : stage.arguments_by_phase) add(buffer);
  }
  for (const DescribedBuffer* described : {&state->internal1, &state->internal2, &state->internal3, &state->internal4,
                                           &state->internal5, &state->internal_exposure}) {
    add(described->storage.buffer); add(described->descriptor);
  }
  for (id<MTLBuffer> buffer : state->frame_ring) add(buffer);
  for (id<MTLBuffer> buffer : state->arena6_stage_snapshots) add(buffer);
  for (id<MTLBuffer> buffer : state->arena7_stage_snapshots) add(buffer);
  for (id<MTLBuffer> buffer : state->arena8_stage_snapshots) add(buffer);
  [residency commit];
  state->residency = residency;
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device, NSURL* root,
                                     NSString* library_name, NSString* entry,
                                     NSError** error) {
  NSURL* url = [[root URLByAppendingPathComponent:@"metallib"]
      URLByAppendingPathComponent:[library_name stringByAppendingString:@".metallib"]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:error];
  id<MTLFunction> function = [library newFunctionWithName:entry];
  if (library == nil || function == nil) {
    SetError(error, 8,
             [NSString stringWithFormat:@"Metal function missing: %@/%@",
                                        library_name, entry]);
    return nil;
  }
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:error];
  if (pipeline == nil && (error == nullptr || *error == nil)) {
    SetError(error, 24,
             [NSString stringWithFormat:@"Metal pipeline creation failed: %@/%@",
                                        library_name, entry]);
  }
  return pipeline;
}

ReplayConfig Enc0Config() {
  ReplayConfig config{};
  std::memcpy(config.magic, "NSPWIN1", 7);
  config.version = 1;
  std::strcpy(config.entry, "dltss_pwin_enc0_layer");
  config.grid[0] = 61;
  config.grid[1] = 37;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.relocation_count = 5;
  config.relocations[0] = {8, 6, 0};
  config.relocations[1] = {24, 7, 0};
  config.relocations[2] = {32, 8, 43200};
  config.relocations[3] = {48, 8, 43200};
  config.relocations[4] = {64, 9, 0};
  return config;
}

// One dispatch goes either to a Metal 3 command buffer (own encoder per
// dispatch) or to an open Metal 4 compute encoder (argument table from the
// frame ring, encoder barrier after the dispatch).
struct FrameSink {
  id<MTLCommandBuffer> metal3 = nil;
  // Reused across dispatches when NEURALSTUDIO_SR_METAL3_SHARED_ENCODER is set; see Metal3Sharing().
  id<MTLComputeCommandEncoder> metal3_shared = nil;
  id<MTL4ComputeCommandEncoder> metal4 = nil;
  id<MTL4CommandBuffer> metal4_command = nil;
  bool metal4_split = false;  // one compute encoder per dispatch (queue-stage barriers)
  SessionState* state = nullptr;
};

// NEURALSTUDIO_SR_METAL3_SHARED_ENCODER: 0 off (default), 1 one encoder serial, 2 one encoder
// concurrent.
//
// The Metal 3 path opens a fresh MTLComputeCommandEncoder for each of the ~97 dispatches in a frame
// and ends it immediately. Every encoder boundary is a full drain, so no two dispatches in the frame
// ever overlap. The existing evidence that this is free compares Metal 3 against Metal 4 - but the
// Metal 4 path puts a device barrier after every dispatch, so both are fully serialised and finding
// them equal says nothing about what overlap would be worth.
//
// Mode 1 keeps ordering exactly (serial dispatch inside one encoder) and only removes the boundaries
// - safe, and shippable if it wins. Mode 2 drops ordering entirely with no barriers: it is NOT
// correct and exists only to price the ceiling. Read mode 2's receipts as a hazard report, not a
// pass.
int Metal3Sharing() {
  static const int mode = [] {
    const char* raw = std::getenv("NEURALSTUDIO_SR_METAL3_SHARED_ENCODER");
    if (raw == nullptr || *raw == '\0') return 0;
    return *raw == '2' ? 2 : (*raw == '0' ? 0 : 1);
  }();
  return mode;
}

void EndMetal3Encoder(FrameSink& sink) {
  if (sink.metal3_shared == nil) return;
  [sink.metal3_shared endEncoding];
  sink.metal3_shared = nil;
}

void Encode(FrameSink& sink,
            id<MTLComputePipelineState> pipeline, id<MTLBuffer> arguments,
            MTLSize grid, MTLSize block,
            const std::vector<std::pair<id<MTLResource>, MTLResourceUsage>>& resources,
            const std::vector<std::pair<id<MTLBuffer>, NSUInteger>>& extras = {}) {
  if (sink.metal4 != nil) {
    SessionState* state = sink.state;
    if (sink.metal4_split && state->table_next > 0) {
      [sink.metal4 endEncoding];
      sink.metal4 = [sink.metal4_command computeCommandEncoder];
      [sink.metal4 barrierAfterQueueStages:MTLStageAll
                              beforeStages:MTLStageDispatch
                         visibilityOptions:MTL4VisibilityOptionDevice];
    }
    if (state->table_next == kRingTables) {
      // Loud rather than silent: a wrapped table is only safe while dispatches cannot overlap.
      std::fprintf(stderr, "neuralstudio: argument tables exhausted (%lu) - dispatches in this frame "
                           "exceed capacity and tables are being reused within the frame\n",
                   static_cast<unsigned long>(kRingTables));
    }
    const NSUInteger index = state->ring_slot * kRingTables + (state->table_next++ % kRingTables);
    id<MTL4ArgumentTable> table = state->argument_tables[index];
    [table setAddress:arguments.gpuAddress atIndex:0];
    for (const auto& binding : extras) {
      if (binding.first != nil) [table setAddress:binding.first.gpuAddress atIndex:binding.second];
    }
    [sink.metal4 setComputePipelineState:pipeline];
    [sink.metal4 setArgumentTable:table];
    [sink.metal4 dispatchThreadgroups:grid threadsPerThreadgroup:block];
    [sink.metal4 barrierAfterEncoderStages:MTLStageDispatch
                        beforeEncoderStages:MTLStageDispatch
                          visibilityOptions:MTL4VisibilityOptionDevice];
    return;
  }
  id<MTLCommandBuffer> command = sink.metal3;
  id<MTLComputeCommandEncoder> encoder = nil;
  // Diagnostic for issue 5: take the descriptor path with NO counter attachment, which
  // separates "using a descriptor changes the result" from "sampling changes the result".
  static const bool bare_descriptor = [] {
    const char* value = std::getenv("NEURALSTUDIO_SR_ENCODER_DESCRIPTOR");
    return value != nullptr && *value == '1';
  }();
  if (bare_descriptor) {
    encoder = [command computeCommandEncoderWithDescriptor:
                  [MTLComputePassDescriptor computePassDescriptor]];
  } else
  if (g_dispatch_timing != nullptr && g_dispatch_timing->samples != nil &&
      g_dispatch_timing->next_index + 2 <= g_dispatch_timing->capacity) {
    MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
    MTLComputePassSampleBufferAttachmentDescriptor* attachment = pass.sampleBufferAttachments[0];
    attachment.sampleBuffer = g_dispatch_timing->samples;
    attachment.startOfEncoderSampleIndex = g_dispatch_timing->next_index;
    attachment.endOfEncoderSampleIndex = g_dispatch_timing->next_index + 1;
    g_dispatch_timing->next_index += 2;
    encoder = [command computeCommandEncoderWithDescriptor:pass];
  } else if (Metal3Sharing() != 0) {
    if (sink.metal3_shared == nil) {
      sink.metal3_shared = [command computeCommandEncoderWithDispatchType:
          Metal3Sharing() == 2 ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    }
    encoder = sink.metal3_shared;
  } else {
    encoder = [command computeCommandEncoder];
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:arguments offset:0 atIndex:0];
  for (const auto& binding : extras) {
    if (binding.first != nil) {
      [encoder setBuffer:binding.first offset:0 atIndex:binding.second];
    }
  }
  for (const auto& resource : resources) {
    if (resource.first != nil) {
      [encoder useResource:resource.first usage:resource.second];
    }
  }
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:block];
  if (encoder != sink.metal3_shared) [encoder endEncoding];
}

// One dispatch of a batched pipeline: explicit (buffer, offset) bindings plus a static constants buffer.
void EncodeDispatch(FrameSink& sink, const Stage::PipelineDispatch& d,
                    const std::vector<std::pair<id<MTLResource>, MTLResourceUsage>>& resources) {
  if (sink.metal4 != nil) {
    SessionState* state = sink.state;
    if (sink.metal4_split && state->table_next > 0) {
      [sink.metal4 endEncoding];
      sink.metal4 = [sink.metal4_command computeCommandEncoder];
      [sink.metal4 barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch
                         visibilityOptions:MTL4VisibilityOptionDevice];
    }
    if (state->table_next == kRingTables) {
      // Loud rather than silent: a wrapped table is only safe while dispatches cannot overlap.
      std::fprintf(stderr, "neuralstudio: argument tables exhausted (%lu) - dispatches in this frame "
                           "exceed capacity and tables are being reused within the frame\n",
                   static_cast<unsigned long>(kRingTables));
    }
    const NSUInteger index = state->ring_slot * kRingTables + (state->table_next++ % kRingTables);
    id<MTL4ArgumentTable> table = state->argument_tables[index];
    for (const auto& b : d.bindings) {
      if (b.second.first != nil) [table setAddress:b.second.first.gpuAddress + b.second.second atIndex:b.first];
    }
    if (d.constants != nil) [table setAddress:d.constants.gpuAddress atIndex:d.constants_index];
    [sink.metal4 setComputePipelineState:d.pipeline];
    [sink.metal4 setArgumentTable:table];
    [sink.metal4 dispatchThreadgroups:d.grid threadsPerThreadgroup:d.block];
    [sink.metal4 barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                          visibilityOptions:MTL4VisibilityOptionDevice];
    return;
  }
  id<MTLComputeCommandEncoder> encoder = nil;
  // Diagnostic for issue 5: take the descriptor path with NO counter attachment, which
  // separates "using a descriptor changes the result" from "sampling changes the result".
  static const bool bare_descriptor = [] {
    const char* value = std::getenv("NEURALSTUDIO_SR_ENCODER_DESCRIPTOR");
    return value != nullptr && *value == '1';
  }();
  if (bare_descriptor) {
    encoder = [sink.metal3 computeCommandEncoderWithDescriptor:
                  [MTLComputePassDescriptor computePassDescriptor]];
  } else
  if (g_dispatch_timing != nullptr && g_dispatch_timing->samples != nil &&
      g_dispatch_timing->next_index + 2 <= g_dispatch_timing->capacity) {
    MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
    MTLComputePassSampleBufferAttachmentDescriptor* attachment = pass.sampleBufferAttachments[0];
    attachment.sampleBuffer = g_dispatch_timing->samples;
    attachment.startOfEncoderSampleIndex = g_dispatch_timing->next_index;
    attachment.endOfEncoderSampleIndex = g_dispatch_timing->next_index + 1;
    g_dispatch_timing->next_index += 2;
    encoder = [sink.metal3 computeCommandEncoderWithDescriptor:pass];
  } else if (Metal3Sharing() != 0) {
    // Diagnostics that attach a sample buffer need their own encoder, so sharing yields to them
    // above; here there is nothing to yield to.
    if (sink.metal3_shared == nil) {
      sink.metal3_shared = [sink.metal3 computeCommandEncoderWithDispatchType:
          Metal3Sharing() == 2 ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    }
    encoder = sink.metal3_shared;
  } else {
    encoder = [sink.metal3 computeCommandEncoder];
  }
  [encoder setComputePipelineState:d.pipeline];
  for (const auto& b : d.bindings) {
    if (b.second.first != nil) [encoder setBuffer:b.second.first offset:b.second.second atIndex:b.first];
  }
  if (d.constants != nil) [encoder setBuffer:d.constants offset:0 atIndex:d.constants_index];
  for (const auto& resource : resources) {
    if (resource.first != nil) [encoder useResource:resource.first usage:resource.second];
  }
  [encoder dispatchThreadgroups:d.grid threadsPerThreadgroup:d.block];
  if (encoder != sink.metal3_shared) [encoder endEncoding];
}

void EncodeArena6Snapshot(FrameSink& sink,
                          const GuardedBuffer& arena,
                          id<MTLBuffer> snapshot) {
  if (snapshot == nil) return;
  if (sink.metal4 != nil) {
    [sink.metal4 barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageBlit
                          visibilityOptions:MTL4VisibilityOptionDevice];
    [sink.metal4 copyFromBuffer:arena.buffer sourceOffset:kCanary toBuffer:snapshot destinationOffset:0
                           size:MIN(arena.bytes, snapshot.length)];
    [sink.metal4 barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageDispatch
                          visibilityOptions:MTL4VisibilityOptionDevice];
    return;
  }
  id<MTLCommandBuffer> command = sink.metal3;
  EndMetal3Encoder(sink);   // a command buffer cannot hold two open encoders
  id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
  [encoder copyFromBuffer:arena.buffer
             sourceOffset:kCanary
                 toBuffer:snapshot
        destinationOffset:0
                     size:arena.bytes];
  [encoder endEncoding];
}

bool Fits(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes) {
  return buffer != nil && offset <= buffer.length && bytes <= buffer.length - offset;
}

}  // namespace

BOOL NeuralStudioDLSSSRIsProvenLiveJitter(float jitter_x, float jitter_y) {
  static constexpr std::array<std::array<float, 2>, 18> kJitterCycle = {{
      {-0.3125f, -0.351851851f},
      {0.1875f, -0.0185185075f},
      {-0.0625f, 0.314814806f},
      {0.4375f, -0.240740746f},
      {-0.46875f, 0.092592597f},
      {0.03125f, 0.42592591f},
      {-0.21875f, -0.42592591f},
      {0.0f, -0.166666657f},
      {-0.25f, 0.166666687f},
      {0.25f, -0.388888896f},
      {-0.375f, -0.0555555522f},
      {0.125f, 0.277777791f},
      {-0.125f, -0.277777791f},
      {0.375f, 0.055555582f},
      {-0.4375f, 0.388888896f},
      {0.0625f, -0.462962955f},
      {-0.1875f, -0.129629612f},
      {0.3125f, 0.203703701f},
  }};
  for (const auto& value : kJitterCycle) {
    if (jitter_x == value[0] && jitter_y == value[1]) return YES;
  }
  return NO;
}

@interface NeuralStudioDLSSSRSession ()
- (BOOL)encodeFrameWithSink:(FrameSink&)sink
                colorBuffer:(id<MTLBuffer>)color
                colorOffset:(NSUInteger)color_offset
               motionBuffer:(id<MTLBuffer>)motion
               motionOffset:(NSUInteger)motion_offset
                depthBuffer:(id<MTLBuffer>)depth
                depthOffset:(NSUInteger)depth_offset
             exposureBuffer:(id<MTLBuffer>)exposure
             exposureOffset:(NSUInteger)exposure_offset
               outputBuffer:(id<MTLBuffer>)output
               outputOffset:(NSUInteger)output_offset
              jitterOffsetX:(float)jitter_x
              jitterOffsetY:(float)jitter_y
         motionVectorScaleX:(float)motion_scale_x
         motionVectorScaleY:(float)motion_scale_y
                      reset:(BOOL)reset
              depthReversed:(BOOL)depth_reversed
                      error:(NSError**)error;
@end

@implementation NeuralStudioDLSSSRSession {
  std::unique_ptr<SessionState> _state;
}

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                       runtimeBundleURL:(NSURL*)runtimeBundleURL
                                  error:(NSError**)error {
  self = [super init];
  if (self == nil) return nil;
  if (device == nil) {
    SetError(error, 22, @"Metal device unavailable");
    return nil;
  }
  if (runtimeBundleURL == nil) {
    SetError(error, 23, @"runtime bundle URL unavailable");
    return nil;
  }
  NSString* validated_root=nil;
  if (!ValidateRuntimeBundle(runtimeBundleURL, &validated_root, error)) {
    return nil;
  }
  _state = std::make_unique<SessionState>();
  _state->device = device;
  _state->bundle_url = runtimeBundleURL;
  _state->runtime_root = validated_root;
  if (!LoadGeometry(runtimeBundleURL, &_state->geometry, error)) return nil;
  const Geometry& geometry = _state->geometry;

  NSData* weights = Load(runtimeBundleURL, @"model/weights.bin", error);
  NSData* constants = Load(runtimeBundleURL, @"model/constants.bin", error);
  NSData* ex2 = Load(runtimeBundleURL, @"math/ex2-f16x2-lut.bin", error);
  NSData* half_rcp = Load(runtimeBundleURL, @"math/rcp-f16-to-f16-lut.bin", error);
  NSData* rsqrt = Load(runtimeBundleURL, @"math/rsq-correction-rom.bin", error);
  NSData* rcp = Load(runtimeBundleURL, @"math/rcp-correction-rom.bin", error);
  if (weights.length != kWeightsBytes || constants.length != kConstantsBytes ||
      ex2.length != kHalfLutBytes || half_rcp.length != kHalfLutBytes ||
      rsqrt.length != kRsqrtRomBytes || rcp.length != kRcpRomBytes) {
    SetError(error, 9, @"runtime model or math resource size mismatch");
    return nil;
  }

  _state->arena6 = MakeGuarded(device, geometry.arena6_bytes, 0x5a);
  _state->arena7 = MakeGuarded(device, geometry.arena7_bytes, 0x5a);
  _state->arena8 = MakeGuarded(device, geometry.arena8_bytes, 0x5a);
  _state->weights = MakeGuarded(device, kWeightsBytes, 0, weights.bytes);
  {
    // Float copies of the weights with the 512-byte tiles listed in
    // model/weight-tiles.json reordered from SM120 fragment order (lane L word w
    // at L*16 + 4w) to row-major: A role row (L>>2)+8(w&1), k 2(L&3)+8(w>>1)+e;
    // B role k 2(L&3)+8(w&1)+e, column (L>>2)+8(w>>1). Hardware lowerings
    // simdgroup_load those operands directly (buffers 15/16).
    const NSUInteger halves = kWeightsBytes / 2;
    std::vector<float> tile_a(halves + 256, 0.0f), tile_b(halves + 256, 0.0f);
    const auto* source = static_cast<const std::uint16_t*>(weights.bytes);
    for (NSUInteger index = 0; index < halves; ++index) {
      __fp16 h; std::memcpy(&h, source + index, 2);
      tile_a[index] = tile_b[index] = static_cast<float>(h);
    }
    NSData* tiles_data = Load(runtimeBundleURL, @"model/weight-tiles.json", nil);
    NSDictionary* tiles = tiles_data == nil ? nil : [NSJSONSerialization JSONObjectWithData:tiles_data options:0 error:nil];
    auto reorder = [&](NSArray* offsets, std::vector<float>& out, bool role_a) {
      for (NSNumber* offset in offsets) {
        const NSUInteger base = [offset unsignedIntegerValue] / 2;  // element index of the tile
        if (base + 256 > halves) continue;
        for (NSUInteger lane = 0; lane < 32; ++lane) {
          for (NSUInteger word = 0; word < 4; ++word) {
            for (NSUInteger e = 0; e < 2; ++e) {
              __fp16 h; std::memcpy(&h, source + base + lane * 8 + word * 2 + e, 2);
              const float value = static_cast<float>(h);
              const NSUInteger row = role_a ? (lane >> 2) + 8 * (word & 1) : 2 * (lane & 3) + 8 * (word & 1) + e;
              const NSUInteger col = role_a ? 2 * (lane & 3) + 8 * (word >> 1) + e : (lane >> 2) + 8 * (word >> 1);
              out[base + row * 16 + col] = value;
            }
          }
        }
      }
    };
    if ([tiles isKindOfClass:NSDictionary.class]) {
      reorder(tiles[@"a"], tile_a, true);
      reorder(tiles[@"b"], tile_b, false);
    }
    _state->weights_f32_a = [device newBufferWithBytes:tile_a.data() length:tile_a.size() * 4 options:MTLResourceStorageModeShared];
    _state->weights_f32_b = [device newBufferWithBytes:tile_b.data() length:tile_b.size() * 4 options:MTLResourceStorageModeShared];
  }
  _state->constants = MakeGuarded(device, kConstantsBytes, 0, constants.bytes);
  auto described = [&](int array_id, std::uint8_t fill) {
    const ArrayGeometry& a = geometry.arrays[array_id - 1];
    return MakeDescribed(device, a.width, a.height, a.format, a.channels, fill);
  };
  _state->internal3 = described(3, 0);
  _state->internal1 = described(1, 0);
  _state->internal5 = described(5, 0);
  _state->internal4 = described(4, 0x5a);
  _state->internal2 = described(2, 0x5a);
  _state->internal_exposure = described(6, 0x5a);
  _state->ex2_lut = BufferFromData(device, ex2);
  _state->half_rcp_lut = BufferFromData(device, half_rcp);
  _state->rsqrt_rom = BufferFromData(device, rsqrt);
  _state->rcp_rom = BufferFromData(device, rcp);
  _state->input_template = LoadProfile(runtimeBundleURL, geometry, @"abi/hiluma-input-reset-template.bin", error);
  _state->output_template = LoadProfile(runtimeBundleURL, geometry, @"abi/hiluma-output-reset-template.bin", error);
  _state->exposure_template = LoadProfile(runtimeBundleURL, geometry, @"abi/exposure-reset-template.bin", error);
  if (_state->input_template.length != 248 || _state->output_template.length != 328 ||
      _state->exposure_template.length != 16) {
    SetError(error, 10, @"sanitized boundary ABI template mismatch");
    return nil;
  }

  _state->input_pipeline = Pipeline(
      device, runtimeBundleURL, @"hiluma_input_poison_a",
      @"hiluma_engine_input_depthreg_mvlo_hdr_v2_rel", error);
  _state->output_pipeline =
      Pipeline(device, runtimeBundleURL, @"hiluma_output_exact_math_safe",
               @"hiluma_engine_output_depthreg_mvlo_hdr_max_v2_rel", error);
  _state->exposure_pipeline =
      Pipeline(device, runtimeBundleURL,
               ExactLibrary(runtimeBundleURL, @"cuda_copy_exposure_kernel", @"netal_copy_exposure",
                            "NEURALSTUDIO_SR_EXPOSURE_LIBRARY"),
               @"cuda_copy_exposure_kernel", error);
  _state->depth_inverted_input_pipeline = Pipeline(
      device, runtimeBundleURL,
      ExactLibrary(runtimeBundleURL, @"hiluma_input_depthinv", @"hiluma_input_depthinv_exact",
                   "NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY"),
      @"hiluma_engine_input_depthinv_mvlo_hdr_v2_rel", error);
  _state->depth_inverted_output_pipeline = Pipeline(
      device, runtimeBundleURL,
      ExactLibrary(runtimeBundleURL, @"hiluma_output_depthinv", @"hiluma_output_depthinv_exact",
                   "NEURALSTUDIO_SR_DEPTH_OUTPUT_LIBRARY"),
      @"hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel", error);
  if (_state->input_pipeline == nil || _state->output_pipeline == nil ||
      _state->exposure_pipeline == nil ||
      _state->depth_inverted_input_pipeline == nil ||
      _state->depth_inverted_output_pipeline == nil) {
    return nil;
  }

  _state->pwin_stages.reserve(11);
  Stage enc0;
  enc0.config = Enc0Config();
  if (geometry.profile) {
    NSData* enc0_config = LoadProfile(runtimeBundleURL, geometry, @"pwin/enc0/replay-config.bin", error);
    if (enc0_config.length != sizeof(enc0.config)) {
      SetError(error, 11, @"profile enc0 replay config size mismatch");
      return nil;
    }
    std::memcpy(&enc0.config, enc0_config.bytes, sizeof(enc0.config));
  }
  enc0.argument_template =
      geometry.profile ? LoadProfile(runtimeBundleURL, geometry, @"pwin/enc0/arguments-template.bin", error)
                       : Load(runtimeBundleURL, @"pwin/enc0-arguments-template.bin", error);
  enc0.launch_position = 2;
  _state->pwin_stages.push_back(std::move(enc0));
  for (int position = 2; position <= 11; ++position) {
    NSString* directory = [NSString stringWithFormat:@"pwin/pwin-%02d", position];
    NSData* config_data = LoadProfile(runtimeBundleURL, geometry,
        [directory stringByAppendingPathComponent:@"replay-config.bin"], error);
    Stage stage;
    if (config_data.length != sizeof(stage.config)) {
      SetError(error, 11, @"PWin replay config size mismatch");
      return nil;
    }
    std::memcpy(&stage.config, config_data.bytes, sizeof(stage.config));
    stage.argument_template = LoadProfile(runtimeBundleURL, geometry,
        [directory stringByAppendingPathComponent:@"arguments-template.bin"], error);
    stage.exact_math = position >= 3 && position <= 10;
    stage.launch_position = position + 1;
    _state->pwin_stages.push_back(std::move(stage));
  }

  std::array<GuardedBuffer*, 4> allocations = {
      &_state->arena6, &_state->arena7, &_state->arena8, &_state->weights};
  for (Stage& stage : _state->pwin_stages) {
    if (std::memcmp(stage.config.magic, "NSPWIN1", 7) != 0 ||
        stage.config.version != 1 || stage.config.relocation_count > 8 ||
        stage.argument_template.length != 176 ||
        strnlen(stage.config.entry, sizeof(stage.config.entry)) == sizeof(stage.config.entry)) {
      SetError(error, 12, @"PWin stage contract invalid");
      return nil;
    }
    NSMutableData* arguments = [stage.argument_template mutableCopy];
    for (std::uint32_t index = 0; index < stage.config.relocation_count; ++index) {
      const Relocation& relocation = stage.config.relocations[index];
      if (relocation.allocation_id < 6 || relocation.allocation_id > 9 ||
          relocation.argument_offset + 8 > arguments.length) {
        SetError(error, 13, @"PWin relocation invalid");
        return nil;
      }
      GuardedBuffer& allocation = *allocations[relocation.allocation_id - 6];
      if (relocation.allocation_offset > allocation.bytes) {
        SetError(error, 14, @"PWin relocation outside allocation");
        return nil;
      }
      Patch64(arguments, relocation.argument_offset,
              Address(allocation, static_cast<NSUInteger>(relocation.allocation_offset)));
    }
    stage.arguments = BufferFromData(device, arguments);
    const PwinShift* shifts = PwinShiftTable(stage.launch_position);
    stage.arguments_by_phase.assign(kPwinShiftPeriod, stage.arguments);
    stage.grid_by_phase.assign(
        kPwinShiftPeriod,
        MTLSizeMake(stage.config.grid[0], stage.config.grid[1], stage.config.grid[2]));
    if (shifts != nullptr) {
      std::uint32_t extent[2] = {0, 0};
      std::memcpy(extent, arguments.bytes, 8);
      for (NSUInteger phase = 0; phase < kPwinShiftPeriod; ++phase) {
        stage.grid_by_phase[phase] = MTLSizeMake(
            (extent[0] + shifts[phase].x + 7) / 8,
            (extent[1] + shifts[phase].y + 7) / 8, stage.config.grid[2]);
      }
      if (stage.grid_by_phase[0].width != stage.config.grid[0] ||
          stage.grid_by_phase[0].height != stage.config.grid[1]) {
        SetError(error, 15, @"PWin grid formula disagrees with replay config");
        return nil;
      }
      std::uint32_t template_shift[2] = {0, 0};
      std::memcpy(template_shift,
                  static_cast<const std::uint8_t*>(arguments.bytes) + 56, 8);
      if (template_shift[0] != shifts[0].x || template_shift[1] != shifts[0].y) {
        SetError(error, 15, @"PWin shift template disagrees with schedule");
        return nil;
      }
      for (NSUInteger phase = 0; phase < kPwinShiftPeriod; ++phase) {
        NSMutableData* phased = [arguments mutableCopy];
        const std::uint32_t shift[2] = {shifts[phase].x, shifts[phase].y};
        std::memcpy(static_cast<std::uint8_t*>(phased.mutableBytes) + 56, shift, 8);
        stage.arguments_by_phase[phase] = BufferFromData(device, phased);
        if (stage.arguments_by_phase[phase] == nil) {
          SetError(error, 15, @"PWin phased argument allocation failed");
          return nil;
        }
      }
    }
    NSString* entry = [NSString stringWithUTF8String:stage.config.entry];
    // Per-entry library override for exactness candidates:
    // NEURALSTUDIO_SR_PWIN_LIBRARY_<entry>=<metallib name>.
    const std::string override_name = std::string("NEURALSTUDIO_SR_PWIN_LIBRARY_") + stage.config.entry;
    NSString* library = ExactLibrary(runtimeBundleURL, entry,
                                     [entry stringByAppendingString:@"_exact"],
                                     override_name.c_str());
    // Bundle-wide PWin lowering selection (live modes): NEURALSTUDIO_SR_PWIN_SUFFIX=_hw
    // picks <entry>_hw.metallib when the bundle ships it; per-entry overrides win.
    const char* pwin_suffix = std::getenv("NEURALSTUDIO_SR_PWIN_SUFFIX");
    if (std::getenv(override_name.c_str()) == nullptr && pwin_suffix != nullptr && *pwin_suffix != '\0') {
      NSString* suffixed = [entry stringByAppendingString:[NSString stringWithUTF8String:pwin_suffix]];
      if (BundleHas(runtimeBundleURL, [NSString stringWithFormat:@"metallib/%@.metallib", suffixed])) {
        library = suffixed;
      }
    }
    stage.pipeline = Pipeline(device, runtimeBundleURL, library, entry, error);
    // Optional dispatch sidecar next to the library (metallib/<library>.json):
    // split lowerings run several SIMD groups per original warp and declare
    // their threadgroup size as "threads_per_threadgroup": [x, y, z].
    {
      NSURL* sidecar = [[runtimeBundleURL URLByAppendingPathComponent:@"metallib"]
          URLByAppendingPathComponent:[library stringByAppendingString:@".json"]];
      NSData* data = [NSData dataWithContentsOfURL:sidecar];
      if (data != nil) {
        NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        NSArray* block = [json isKindOfClass:[NSDictionary class]] ? json[@"threads_per_threadgroup"] : nil;
        NSNumber* scratch = [json isKindOfClass:[NSDictionary class]] ? json[@"scratch_bytes_per_threadgroup"] : nil;
        if ([scratch isKindOfClass:[NSNumber class]]) {
          stage.scratch_bytes_per_threadgroup = [scratch unsignedIntegerValue];
        }
        NSNumber* windows = [json isKindOfClass:[NSDictionary class]] ? json[@"windows_per_threadgroup"] : nil;
        if ([windows isKindOfClass:[NSNumber class]] && [windows unsignedIntegerValue] > 1) {
          stage.windows_per_threadgroup = [windows unsignedIntegerValue];
        }
        if ([block isKindOfClass:[NSArray class]] && block.count == 3) {
          for (NSUInteger axis = 0; axis < 3; ++axis) {
            stage.config.block[axis] = static_cast<std::uint32_t>([block[axis] unsignedIntegerValue]);
          }
          std::fprintf(stderr, "neuralstudio: pwin_dispatch_sidecar entry=%s library=%s block=%u,%u,%u\n",
                       stage.config.entry, library.UTF8String, stage.config.block[0],
                       stage.config.block[1], stage.config.block[2]);
        }
      }
    }
    const std::uint64_t threads = static_cast<std::uint64_t>(stage.config.block[0]) *
                                  stage.config.block[1] * stage.config.block[2];
    if (stage.arguments == nil || stage.pipeline == nil ||
        threads > stage.pipeline.maxTotalThreadsPerThreadgroup) {
      SetError(error, 15, @"PWin pipeline setup failed");
      return nil;
    }
    stage.dispatch_by_phase = stage.grid_by_phase;
    stage.window_grid_by_phase.assign(stage.grid_by_phase.size(), nil);
    if (stage.windows_per_threadgroup > 1) {
      for (NSUInteger phase = 0; phase < stage.grid_by_phase.size(); ++phase) {
        const MTLSize grid = stage.grid_by_phase[phase];
        const NSUInteger windows = grid.width * grid.height * grid.depth;
        const NSUInteger k = stage.windows_per_threadgroup;
        stage.dispatch_by_phase[phase] = MTLSizeMake((windows + k - 1) / k, 1, 1);
        const std::uint32_t info[4] = {static_cast<std::uint32_t>(grid.width), static_cast<std::uint32_t>(grid.height),
                                       static_cast<std::uint32_t>(k), 0};
        stage.window_grid_by_phase[phase] = [device newBufferWithBytes:info length:sizeof(info)
                                                               options:MTLResourceStorageModeShared];
        if (stage.window_grid_by_phase[phase] == nil) {
          SetError(error, 15, @"PWin window grid allocation failed");
          return nil;
        }
      }
    }
  }

  // Batched pipelines (bundle pipeline/<entry>.json; schema of tools/pwin_pipeline/harness.mm)
  const char* pipeline_env = std::getenv("NEURALSTUDIO_SR_PWIN_PIPELINE");
  // Where the batched pipelines live. A synthesized geometry (a resolution with no profiles/<WxH>/ directory)
  // reuses the 1920x1080 prefix with patched PWin templates, so the profile's own pipelines would bake the
  // wrong token grid; those resolutions get their own directory instead, generated offline from the exported
  // synthesis (NEURALSTUDIO_SR_EXPORT_SYNTH_DIR plus tools/pwin_pipeline/build-profile-pipelines.py --synth).
  // Without one, the transpiled kernels run: correct, just slower.
  NSString* pipeline_dir =
      geometry.synthesized
          ? [NSString stringWithFormat:@"pipeline-synth/%lux%lu/", static_cast<unsigned long>(geometry.output_width),
                                       static_cast<unsigned long>(geometry.output_height)]
          : [geometry.prefix stringByAppendingString:@"pipeline/"];
  if (pipeline_env != nullptr && *pipeline_env != '\0' && std::strcmp(pipeline_env, "0") != 0) {
    NSUInteger scratch_needed = 0;
    // One compute pipeline state per (library, function, function constants). The window-shift phases of a
    // layer mostly differ in the launch grid and a byte parameter, so without this the same specialization
    // is compiled up to eight times and arming takes minutes inside the game process.
    std::map<std::string, __strong id<MTLComputePipelineState>> pipeline_cache;
    // A layer with no batched pipeline silently runs the transpiled per-window kernel instead:
    // correct, and roughly twice the GPU time. A half-generated profile therefore reads as a healthy
    // run - that is how 1728x972 shipped with 6 of 11 layers and cost RE9 ~17 ms a frame with
    // nothing in the log. Count what was found and say so below.
    std::vector<std::string> missing_entries;
    for (Stage& stage : _state->pwin_stages) {
      NSString* entry = [NSString stringWithUTF8String:stage.config.entry];
      // per-profile pipelines (the dispatch grids and token strides depend on the profile geometry):
      // profiles/<WxH>/pipeline/<entry>.json, or pipeline/<entry>.json for the legacy root profile
      NSString* relative = [pipeline_dir stringByAppendingFormat:@"%@.json", entry];
      if (!BundleHas(runtimeBundleURL, relative)) { missing_entries.push_back(stage.config.entry); continue; }
      NSData* data = [NSData dataWithContentsOfURL:[runtimeBundleURL URLByAppendingPathComponent:relative]];
      NSDictionary* json = data == nil ? nil : [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
      if (![json isKindOfClass:[NSDictionary class]]) continue;
      NSString* library_name = json[@"library"];
      id<MTLLibrary> library = [device newLibraryWithURL:[[runtimeBundleURL URLByAppendingPathComponent:@"metallib"]
          URLByAppendingPathComponent:[library_name stringByAppendingString:@".metallib"]] error:error];
      if (library == nil) { SetError(error, 40, @"pipeline library missing"); return nil; }
      scratch_needed = MAX(scratch_needed, NSUInteger([json[@"scratch_bytes"] unsignedLongLongValue]));
      std::map<std::string, std::pair<id<MTLBuffer>, NSUInteger>> named;   // name -> (buffer, base offset)
      named["arena6"] = {_state->arena6.buffer, kCanary};
      named["arena7"] = {_state->arena7.buffer, kCanary};
      named["arena8"] = {_state->arena8.buffer, kCanary};
      named["weights"] = {_state->weights.buffer, kCanary};
      named["params"] = {stage.arguments, 0};
      for (NSDictionary* fj in json[@"files"]) {
        NSData* blob = Load(runtimeBundleURL, fj[@"path"], error);
        if (blob == nil) { SetError(error, 40, @"pipeline weight file missing"); return nil; }
        id<MTLBuffer> buffer = BufferFromData(device, blob);
        _state->pipeline_weights.push_back(buffer);
        named[[fj[@"name"] UTF8String]] = {buffer, 0};
      }
      // one dispatch list per window-shift phase: phase 0 is <entry>.json, the others <entry>.phase<N>.json
      // (absent for layers whose window grid does not shift, which then reuse phase 0)
      bool parse_failed = false;
      auto parse = [&](NSDictionary* doc) -> std::vector<Stage::PipelineDispatch> {
      std::vector<Stage::PipelineDispatch> dispatches;
      for (NSDictionary* dj in doc[@"dispatches"]) {
        Stage::PipelineDispatch d;
        NSDictionary* consts = dj[@"constants"];
        std::string key = std::string([library_name UTF8String]) + "|" + [dj[@"function"] UTF8String];
        if ([consts isKindOfClass:[NSDictionary class]] && consts.count > 0) {
          for (NSString* ck in [consts.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
            key += "|" + std::string([ck UTF8String]) + "=" + std::to_string([consts[ck] unsignedIntValue]);
          }
        }
        auto cached = pipeline_cache.find(key);
        if (cached != pipeline_cache.end()) {
          d.pipeline = cached->second;
        } else {
          id<MTLFunction> function = nil;
          if ([consts isKindOfClass:[NSDictionary class]] && consts.count > 0) {
            MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
            for (NSString* ck in consts) {
              std::uint32_t v = [consts[ck] unsignedIntValue];
              [values setConstantValue:&v type:MTLDataTypeUInt atIndex:[ck integerValue]];
            }
            function = [library newFunctionWithName:dj[@"function"] constantValues:values error:error];
          } else {
            function = [library newFunctionWithName:dj[@"function"]];
          }
          if (function == nil) { SetError(error, 40, @"pipeline function missing"); parse_failed = true; return {}; }
          d.pipeline = [device newComputePipelineStateWithFunction:function error:error];
          if (d.pipeline == nil) { SetError(error, 40, @"pipeline creation failed"); parse_failed = true; return {}; }
          pipeline_cache[key] = d.pipeline;
        }
        NSArray* g = dj[@"grid"]; NSArray* b = dj[@"block"];
        d.grid = MTLSizeMake([g[0] unsignedIntegerValue], [g[1] unsignedIntegerValue], [g[2] unsignedIntegerValue]);
        d.block = MTLSizeMake([b[0] unsignedIntegerValue], [b[1] unsignedIntegerValue], [b[2] unsignedIntegerValue]);
        for (NSDictionary* bj in dj[@"buffers"]) {
          const std::string name = [bj[@"name"] UTF8String];
          if (name == "scratch") {
            d.bindings.push_back({[bj[@"index"] unsignedIntegerValue], {nil, [bj[@"offset"] unsignedIntegerValue]}});  // resolved below
            continue;
          }
          if (!named.count(name)) { SetError(error, 40, @"pipeline buffer name unknown"); parse_failed = true; return {}; }
          d.bindings.push_back({[bj[@"index"] unsignedIntegerValue],
                                {named[name].first, named[name].second + [bj[@"offset"] unsignedIntegerValue]}});
        }
        for (NSDictionary* cj in dj[@"bytes"]) {
          std::vector<std::uint32_t> words;
          for (NSNumber* n in cj[@"u32"]) words.push_back([n unsignedIntValue]);
          d.constants = [device newBufferWithBytes:words.data() length:words.size() * 4 options:MTLResourceStorageModeShared];
          d.constants_index = [cj[@"index"] unsignedIntegerValue];
        }
        dispatches.push_back(std::move(d));
      }
      return dispatches;
      };
      std::vector<Stage::PipelineDispatch> phase0 = parse(json);
      if (parse_failed) return nil;
      stage.pipeline_dispatches_by_phase.assign(kPwinShiftPeriod, phase0);
      NSUInteger phased = 0;
      for (NSUInteger phase = 1; phase < kPwinShiftPeriod; ++phase) {
        NSString* pr = [pipeline_dir stringByAppendingFormat:@"%@.phase%lu.json", entry,
                        static_cast<unsigned long>(phase)];
        if (!BundleHas(runtimeBundleURL, pr)) continue;
        NSData* pd = [NSData dataWithContentsOfURL:[runtimeBundleURL URLByAppendingPathComponent:pr]];
        NSDictionary* pj = pd == nil ? nil : [NSJSONSerialization JSONObjectWithData:pd options:0 error:nil];
        if (![pj isKindOfClass:[NSDictionary class]]) { SetError(error, 40, @"pipeline phase json invalid"); return nil; }
        scratch_needed = MAX(scratch_needed, NSUInteger([pj[@"scratch_bytes"] unsignedLongLongValue]));
        stage.pipeline_dispatches_by_phase[phase] = parse(pj);
        if (parse_failed) return nil;
        ++phased;
      }
      std::fprintf(stderr, "neuralstudio: pwin_pipeline %s entry=%s dispatches=%zu phases=%lu compiled=%zu\n",
                   [pipeline_dir UTF8String],
                   stage.config.entry, phase0.size(), static_cast<unsigned long>(phased + 1),
                   pipeline_cache.size());
    }
    _state->pipeline_layers_total = _state->pwin_stages.size();
    _state->pipeline_layers_present = _state->pwin_stages.size() - missing_entries.size();
    {
      std::string joined;
      for (const std::string& name : missing_entries) { if (!joined.empty()) joined += ","; joined += name; }
      _state->pipeline_layers_missing = joined;
    }
    if (missing_entries.empty()) {
      std::fprintf(stderr, "neuralstudio: pwin_pipeline_set %s complete layers=%zu/%zu\n",
                   [pipeline_dir UTF8String], _state->pwin_stages.size(), _state->pwin_stages.size());
    } else {
      std::string list;
      for (const std::string& name : missing_entries) { if (!list.empty()) list += ","; list += name; }
      std::fprintf(stderr,
                   "neuralstudio: pwin_pipeline_set %s INCOMPLETE layers=%zu/%zu missing=%s"
                   " (those layers run the transpiled per-window kernel: correct, ~2x the GPU time;"
                   " regenerate with tools/pwin_pipeline/build-profile-pipelines.py)\n",
                   [pipeline_dir UTF8String], _state->pwin_stages.size() - missing_entries.size(),
                   _state->pwin_stages.size(), list.c_str());
    }
    if (scratch_needed > 0) {
      _state->pipeline_scratch = [device newBufferWithLength:scratch_needed options:MTLResourceStorageModePrivate];
      if (_state->pipeline_scratch == nil) { SetError(error, 40, @"pipeline scratch allocation failed"); return nil; }
      for (Stage& stage : _state->pwin_stages) {
        for (std::vector<Stage::PipelineDispatch>& phase : stage.pipeline_dispatches_by_phase) {
          for (Stage::PipelineDispatch& d : phase) {
            for (auto& b : d.bindings) if (b.second.first == nil) b.second.first = _state->pipeline_scratch;
          }
        }
      }
    }
  }

  if (_state->arena6.buffer == nil || _state->arena7.buffer == nil ||
      _state->arena8.buffer == nil || _state->weights.buffer == nil ||
      _state->constants.buffer == nil || _state->internal3.descriptor == nil ||
      _state->internal1.descriptor == nil || _state->internal5.descriptor == nil ||
      _state->internal4.descriptor == nil || _state->internal2.descriptor == nil ||
      _state->internal_exposure.descriptor == nil || _state->ex2_lut == nil ||
      _state->half_rcp_lut == nil || _state->rsqrt_rom == nil || _state->rcp_rom == nil) {
    SetError(error, 16, @"SR session resource allocation failed");
    return nil;
  }
  const char* diagnostics =
      std::getenv("NEURALSTUDIO_SR_TEMPORAL_STAGE_DIAGNOSTICS");
  if (diagnostics != nullptr && std::strcmp(diagnostics, "1") == 0) {
    _state->arena6_stage_snapshots.reserve(12);
    for (NSUInteger index = 0; index < 12; ++index) {
      id<MTLBuffer> snapshot =
          [device newBufferWithLength:geometry.arena6_bytes
                              options:MTLResourceStorageModeShared];
      if (snapshot == nil) {
        SetError(error, 27, @"temporal diagnostic allocation failed");
        return nil;
      }
      _state->arena6_stage_snapshots.push_back(snapshot);
    }
    _state->arena7_stage_snapshots.reserve(12);
    _state->arena8_stage_snapshots.reserve(12);
    for (NSUInteger index = 0; index < 12; ++index) {
      id<MTLBuffer> snapshot7 =
          [device newBufferWithLength:geometry.arena7_bytes
                              options:MTLResourceStorageModeShared];
      id<MTLBuffer> snapshot8 =
          [device newBufferWithLength:geometry.arena8_bytes
                              options:MTLResourceStorageModeShared];
      if (snapshot7 == nil || snapshot8 == nil) {
        SetError(error, 27, @"temporal diagnostic allocation failed");
        return nil;
      }
      _state->arena7_stage_snapshots.push_back(snapshot7);
      _state->arena8_stage_snapshots.push_back(snapshot8);
    }
  }
  const char* mufu_trace = std::getenv("NEURALSTUDIO_SR_TEMPORAL_MUFU_TRACE");
  if (mufu_trace != nullptr && std::strcmp(mufu_trace, "1") == 0) {
    _state->mufu_lg2_trace = [device newBufferWithLength:kMufuTraceBytes
                                                 options:MTLResourceStorageModeShared];
    _state->mufu_ex2_trace = [device newBufferWithLength:kMufuTraceBytes
                                                 options:MTLResourceStorageModeShared];
    if (_state->mufu_lg2_trace == nil || _state->mufu_ex2_trace == nil) {
      SetError(error, 29, @"temporal MUFU trace allocation failed");
      return nil;
    }
    std::memset(_state->mufu_lg2_trace.contents, 0, kMufuTraceBytes);
    std::memset(_state->mufu_ex2_trace.contents, 0, kMufuTraceBytes);
  }
  const char* mufu_contract_path =
      std::getenv("NEURALSTUDIO_SR_TEMPORAL_MUFU_CONTRACT_DIR");
  NSURL* default_contract = [runtimeBundleURL URLByAppendingPathComponent:@"math/sm120-mufu"];
  if ((mufu_contract_path == nullptr || *mufu_contract_path == '\0') &&
      BundleHas(runtimeBundleURL, @"math/sm120-mufu/lg2-input-table.bin")) {
    mufu_contract_path = default_contract.path.UTF8String;
  }
  if (mufu_contract_path != nullptr && *mufu_contract_path != '\0') {
    NSURL* contract = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:mufu_contract_path]];
    NSData* lg2_keys = Load(contract, @"lg2-input-table.bin", error);
    NSData* lg2_values = Load(contract, @"lg2-output-table.bin", error);
    NSData* ex2_keys = Load(contract, @"ex2-input-table.bin", error);
    NSData* ex2_values = Load(contract, @"ex2-output-table.bin", error);
    if (lg2_keys.length != kMufuTraceBytes ||
        lg2_values.length != kMufuTraceBytes ||
        ex2_keys.length != kMufuTraceBytes ||
        ex2_values.length != kMufuTraceBytes) {
      SetError(error, 30, @"temporal MUFU contract size mismatch");
      return nil;
    }
    _state->mufu_lg2_keys = BufferFromData(device, lg2_keys);
    _state->mufu_lg2_values = BufferFromData(device, lg2_values);
    _state->mufu_ex2_keys = BufferFromData(device, ex2_keys);
    _state->mufu_ex2_values = BufferFromData(device, ex2_values);
    _state->mufu_lg2_misses = [device newBufferWithLength:kMufuTraceBytes
                                                  options:MTLResourceStorageModeShared];
    _state->mufu_ex2_misses = [device newBufferWithLength:kMufuTraceBytes
                                                  options:MTLResourceStorageModeShared];
    if (_state->mufu_lg2_keys == nil || _state->mufu_lg2_values == nil ||
        _state->mufu_ex2_keys == nil || _state->mufu_ex2_values == nil ||
        _state->mufu_lg2_misses == nil || _state->mufu_ex2_misses == nil) {
      SetError(error, 31, @"temporal MUFU contract allocation failed");
      return nil;
    }
    std::memset(_state->mufu_lg2_misses.contents, 0, kMufuTraceBytes);
    std::memset(_state->mufu_ex2_misses.contents, 0, kMufuTraceBytes);
  }
  const char* texture_weights_path =
      std::getenv("NEURALSTUDIO_SR_TEXTURE_WEIGHT_TABLE");
  NSURL* default_weights = [runtimeBundleURL URLByAppendingPathComponent:@"math/sm120-texture-weights.bin"];
  if ((texture_weights_path == nullptr || *texture_weights_path == '\0') &&
      BundleHas(runtimeBundleURL, @"math/sm120-texture-weights.bin")) {
    texture_weights_path = default_weights.path.UTF8String;
  }
  if (texture_weights_path != nullptr && *texture_weights_path != '\0') {
    NSData* file = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:texture_weights_path]];
    if (file.length != kTextureWeightTableBytes + 20 ||
        std::memcmp(file.bytes, "NSTEXW1", 7) != 0) {
      SetError(error, 30, @"SM120 texture weight contract mismatch");
      return nil;
    }
    NSData* payload = [file subdataWithRange:
        NSMakeRange(20, kTextureWeightTableBytes)];
    _state->texture_weight_table = BufferFromData(device, payload);
    if (_state->texture_weight_table == nil) {
      SetError(error, 31, @"SM120 texture weight allocation failed");
      return nil;
    }
  }
  const char* dispatch_timing = std::getenv("NEURALSTUDIO_SR_DISPATCH_TIMING");
  if (dispatch_timing != nullptr && std::strcmp(dispatch_timing, "1") == 0) {
    id<MTLCounterSet> timestamp_set = nil;
    for (id<MTLCounterSet> counter_set in device.counterSets) {
      if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamp_set = counter_set;
    }
    if (timestamp_set != nil) {
      MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
      descriptor.counterSet = timestamp_set;
      descriptor.storageMode = MTLStorageModeShared;
      NSUInteger sample_count = 32;
      if (const char* cap = std::getenv("NEURALSTUDIO_SR_DISPATCH_TIMING_SAMPLES")) {
        const long parsed = std::strtol(cap, nullptr, 10);
        if (parsed > 0 && parsed <= 4096) sample_count = static_cast<NSUInteger>(parsed);
      }
      descriptor.sampleCount = sample_count;
      NSError* sample_error = nil;
      _state->dispatch_timing.samples =
          [device newCounterSampleBufferWithDescriptor:descriptor error:&sample_error];
      _state->dispatch_timing.capacity = _state->dispatch_timing.samples == nil ? 0 : sample_count;
    }
  }
  const char* trace_bytes = std::getenv("NEURALSTUDIO_SR_TEMPORAL_TRACE_BYTES");
  if (trace_bytes != nullptr && *trace_bytes != '\0') {
    const NSUInteger bytes = static_cast<NSUInteger>(std::strtoull(trace_bytes, nullptr, 10));
    _state->temporal_trace = [device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (_state->temporal_trace == nil) {
      SetError(error, 32, @"temporal trace allocation failed");
      return nil;
    }
    auto* words = static_cast<std::uint32_t*>(_state->temporal_trace.contents);
    for (NSUInteger index = 0; index < bytes / 4; ++index) words[index] = 0x7fc0deadu;
  }
  {
    NSUInteger largest_grid = 0;
    NSUInteger largest_scratch = 0;
    for (const Stage& stage : _state->pwin_stages) {
      for (const MTLSize& grid : stage.grid_by_phase) {
        const NSUInteger warps = MAX(NSUInteger(1), NSUInteger(stage.config.block[0]) * stage.config.block[1] * stage.config.block[2] / 32);
        largest_grid = MAX(largest_grid, grid.width * grid.height * grid.depth * warps);
        largest_scratch = MAX(largest_scratch, grid.width * grid.height * grid.depth * stage.scratch_bytes_per_threadgroup);
      }
    }
    _state->pwin_scratch = [device newBufferWithLength:MAX(MAX(largest_grid, NSUInteger(1)) * 3072, largest_scratch)
                                                options:MTLResourceStorageModePrivate];
    if (_state->pwin_scratch == nil) {
      SetError(error, 32, @"pwin scratch allocation failed");
      return nil;
    }
  }
  const char* pwin_stats = std::getenv("NEURALSTUDIO_SR_PWIN_STATS");
  if (pwin_stats != nullptr && *pwin_stats == '1') {
    for (NSUInteger index = 0; index < _state->pwin_stages.size(); ++index) {
      id<MTLBuffer> counters = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
      if (counters == nil) {
        SetError(error, 32, @"pwin stats allocation failed");
        return nil;
      }
      std::memset(counters.contents, 0, 256);
      _state->pwin_stats.push_back(counters);
    }
  }
  const char* scalar_debug =
      std::getenv("NEURALSTUDIO_SR_TEMPORAL_SCALAR_DEBUG");
  const char* pair_debug =
      std::getenv("NEURALSTUDIO_SR_TEMPORAL_PAIR_DEBUG");
  if ((scalar_debug != nullptr && std::strcmp(scalar_debug, "1") == 0) ||
      (pair_debug != nullptr && std::strcmp(pair_debug, "1") == 0)) {
    const NSUInteger debug_bytes =
        (pair_debug != nullptr && std::strcmp(pair_debug, "1") == 0)
            ? (120u * 18u * 16u * 16u * 8u)
            : 1024u;
    _state->temporal_scalar_debug =
        [device newBufferWithLength:debug_bytes
                            options:MTLResourceStorageModeShared];
    if (_state->temporal_scalar_debug == nil) {
      SetError(error, 32, @"temporal scalar debug allocation failed");
      return nil;
    }
    std::memset(_state->temporal_scalar_debug.contents, 0, debug_bytes);
  }
  BuildMetal4Support(_state.get());
  return self;
}

- (NSUInteger)inputWidth { return _state->geometry.input_width; }
- (NSUInteger)inputHeight { return _state->geometry.input_height; }
- (NSUInteger)outputWidth { return _state->geometry.output_width; }
- (NSUInteger)outputHeight { return _state->geometry.output_height; }
- (NSString*)runtimeBundleRoot {
  return _state->runtime_root;
}
- (NSURL*)runtimeBundleURL { return _state->bundle_url; }
- (BOOL)isConsumed { return _state != nullptr && _state->consumed.load(); }
- (NSUInteger)pipelineLayersTotal { return _state == nullptr ? 0 : _state->pipeline_layers_total; }
- (NSUInteger)pipelineLayersPresent { return _state == nullptr ? 0 : _state->pipeline_layers_present; }
- (NSString*)pipelineLayersMissing {
  if (_state == nullptr || _state->pipeline_layers_missing.empty()) return @"";
  return [NSString stringWithUTF8String:_state->pipeline_layers_missing.c_str()];
}

- (NSUInteger)encodedFrameCount {
  return _state == nullptr ? 0 : static_cast<NSUInteger>(_state->encoded_frame_count);
}

- (NSData*)neuralStudioDiagnosticTemporalState:(NSUInteger)array_index {
  if (_state == nullptr) return nil;
  std::lock_guard<std::mutex> lock(_state->mutex);
  GuardedBuffer* buffer = nullptr;
  switch (array_index) {
    case 1: buffer = &_state->internal1.storage; break;
    case 2: buffer = &_state->internal2.storage; break;
    case 3: buffer = &_state->internal3.storage; break;
    case 4: buffer = &_state->internal4.storage; break;
    case 5: buffer = &_state->internal5.storage; break;
    case 6: buffer = &_state->internal_exposure.storage; break;
    default: return nil;
  }
  return [NSData dataWithBytes:static_cast<const std::uint8_t*>(buffer->buffer.contents) +
                               kCanary
                         length:buffer->bytes];
}

- (NSData*)neuralStudioDiagnosticArena6AfterStage:(NSUInteger)stage_index {
  if (_state == nullptr) return nil;
  std::lock_guard<std::mutex> lock(_state->mutex);
  if (stage_index >= _state->arena6_stage_snapshots.size()) return nil;
  id<MTLBuffer> snapshot = _state->arena6_stage_snapshots[stage_index];
  return [NSData dataWithBytes:snapshot.contents length:snapshot.length];
}

- (NSData*)neuralStudioDiagnosticArena:(NSUInteger)arena
                            afterStage:(NSUInteger)stage_index {
  if (_state == nullptr) return nil;
  std::lock_guard<std::mutex> lock(_state->mutex);
  const std::vector<__strong id<MTLBuffer>>* snapshots = nullptr;
  if (arena == 6) snapshots = &_state->arena6_stage_snapshots;
  else if (arena == 7) snapshots = &_state->arena7_stage_snapshots;
  else if (arena == 8) snapshots = &_state->arena8_stage_snapshots;
  if (snapshots == nullptr || stage_index >= snapshots->size()) return nil;
  id<MTLBuffer> snapshot = (*snapshots)[stage_index];
  return [NSData dataWithBytes:snapshot.contents length:snapshot.length];
}

// Study helper: re-encode one PWin stage `repeats` times into a command buffer
// (inputs are whatever the arenas hold; only the GPU counters matter).
- (BOOL)neuralStudioDiagnosticLoopStage:(NSUInteger)stage_index
                                repeats:(NSUInteger)repeats
                        toCommandBuffer:(id<MTLCommandBuffer>)command {
  if (_state == nullptr || stage_index >= _state->pwin_stages.size()) return NO;
  std::lock_guard<std::mutex> lock(_state->mutex);
  const Stage& stage = _state->pwin_stages[stage_index];
  FrameSink sink; sink.metal3 = command; sink.state = _state.get();
  std::vector<std::pair<id<MTLBuffer>, NSUInteger>> extras;
  if (stage.exact_math) extras = {{_state->ex2_lut, 1}, {_state->half_rcp_lut, 2}};
  extras.push_back({_state->pwin_scratch, 14});
  extras.push_back({_state->weights_f32_a, 15});
  extras.push_back({_state->weights_f32_b, 16});
  if (stage.window_grid_by_phase[0] != nil) extras.push_back({stage.window_grid_by_phase[0], 17});
  for (NSUInteger repeat = 0; repeat < repeats; ++repeat) {
    Encode(sink, stage.pipeline, stage.arguments_by_phase[0], stage.dispatch_by_phase[0],
           MTLSizeMake(stage.config.block[0], stage.config.block[1], stage.config.block[2]),
           {{_state->arena6.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
            {_state->arena7.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
            {_state->arena8.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
            {_state->weights.buffer, MTLResourceUsageRead},
            {_state->ex2_lut, MTLResourceUsageRead},
            {_state->half_rcp_lut, MTLResourceUsageRead}},
           extras);
  }
  return YES;
}

- (NSData*)neuralStudioDiagnosticMufuTrace:(NSUInteger)operation {
  if (_state == nullptr) return nil;
  std::lock_guard<std::mutex> lock(_state->mutex);
  id<MTLBuffer> buffer = nil;
  if (operation == 1) buffer = _state->mufu_lg2_trace;
  else if (operation == 2) buffer = _state->mufu_ex2_trace;
  else if (operation == 3) buffer = _state->temporal_scalar_debug;
  else if (operation == 4) buffer = _state->mufu_lg2_misses;
  else if (operation == 5) buffer = _state->mufu_ex2_misses;
  else if (operation == 6) buffer = _state->temporal_trace;
  else if (operation == 8) {
    if (_state->pwin_stats.empty()) return nil;
    NSMutableData* all = [NSMutableData data];
    for (id<MTLBuffer> counters : _state->pwin_stats) [all appendBytes:counters.contents length:256];
    return all;
  }
  else if (operation == 7) {
    // Resolved per-dispatch timestamps: pairs (start, end) in GPU timestamp units.
    if (_state->dispatch_timing.samples == nil) return nil;
    return [_state->dispatch_timing.samples
        resolveCounterRange:NSMakeRange(0, _state->dispatch_timing.next_index)];
  }
  if (buffer == nil) return nil;
  return [NSData dataWithBytes:buffer.contents length:buffer.length];
}

- (BOOL)encodeResetFrameToCommandBuffer:(id<MTLCommandBuffer>)command
                             colorBuffer:(id<MTLBuffer>)color
                             colorOffset:(NSUInteger)color_offset
                            motionBuffer:(id<MTLBuffer>)motion
                            motionOffset:(NSUInteger)motion_offset
                             depthBuffer:(id<MTLBuffer>)depth
                             depthOffset:(NSUInteger)depth_offset
                          exposureBuffer:(id<MTLBuffer>)exposure
                          exposureOffset:(NSUInteger)exposure_offset
                            outputBuffer:(id<MTLBuffer>)output
                            outputOffset:(NSUInteger)output_offset
                                   error:(NSError**)error {
  return [self encodeResetFrameToCommandBuffer:command
                                   colorBuffer:color
                                   colorOffset:color_offset
                                  motionBuffer:motion
                                  motionOffset:motion_offset
                                   depthBuffer:depth
                                   depthOffset:depth_offset
                                exposureBuffer:exposure
                                exposureOffset:exposure_offset
                                  outputBuffer:output
                                  outputOffset:output_offset
                                 jitterOffsetX:0.0f
                                 jitterOffsetY:0.0f
                            motionVectorScaleX:1.0f
                            motionVectorScaleY:1.0f
                                 depthReversed:NO
                                         error:error];
}

- (BOOL)encodeResetFrameToCommandBuffer:(id<MTLCommandBuffer>)command
                             colorBuffer:(id<MTLBuffer>)color
                             colorOffset:(NSUInteger)color_offset
                            motionBuffer:(id<MTLBuffer>)motion
                            motionOffset:(NSUInteger)motion_offset
                             depthBuffer:(id<MTLBuffer>)depth
                             depthOffset:(NSUInteger)depth_offset
                          exposureBuffer:(id<MTLBuffer>)exposure
                          exposureOffset:(NSUInteger)exposure_offset
                            outputBuffer:(id<MTLBuffer>)output
                            outputOffset:(NSUInteger)output_offset
                           jitterOffsetX:(float)jitter_x
                           jitterOffsetY:(float)jitter_y
                      motionVectorScaleX:(float)motion_scale_x
                      motionVectorScaleY:(float)motion_scale_y
                           depthReversed:(BOOL)depth_reversed
                                   error:(NSError**)error {
  if (_state == nullptr) {
    SetError(error, 17, @"SR session is not initialized");
    return NO;
  }
  bool expected = false;
  if (!_state->consumed.compare_exchange_strong(expected, true)) {
    SetError(error, 18, @"one-frame SR session already consumed");
    return NO;
  }
  const BOOL encoded = [self encodeFrameToCommandBuffer:command
                                             colorBuffer:color
                                             colorOffset:color_offset
                                            motionBuffer:motion
                                            motionOffset:motion_offset
                                             depthBuffer:depth
                                             depthOffset:depth_offset
                                          exposureBuffer:exposure
                                          exposureOffset:exposure_offset
                                            outputBuffer:output
                                            outputOffset:output_offset
                                           jitterOffsetX:jitter_x
                                           jitterOffsetY:jitter_y
                                      motionVectorScaleX:motion_scale_x
                                      motionVectorScaleY:motion_scale_y
                                                   reset:YES
                                           depthReversed:depth_reversed
                                                   error:error];
  if (!encoded) _state->consumed.store(false);
  return encoded;
}

- (BOOL)encodeFrameWithSink:(FrameSink&)sink
                        colorBuffer:(id<MTLBuffer>)color
                        colorOffset:(NSUInteger)color_offset
                       motionBuffer:(id<MTLBuffer>)motion
                       motionOffset:(NSUInteger)motion_offset
                        depthBuffer:(id<MTLBuffer>)depth
                        depthOffset:(NSUInteger)depth_offset
                     exposureBuffer:(id<MTLBuffer>)exposure
                     exposureOffset:(NSUInteger)exposure_offset
                       outputBuffer:(id<MTLBuffer>)output
                       outputOffset:(NSUInteger)output_offset
                      jitterOffsetX:(float)jitter_x
                      jitterOffsetY:(float)jitter_y
                 motionVectorScaleX:(float)motion_scale_x
                 motionVectorScaleY:(float)motion_scale_y
                              reset:(BOOL)reset
                      depthReversed:(BOOL)depth_reversed
                              error:(NSError**)error {
  if (_state == nullptr) {
    SetError(error, 17, @"SR session is not initialized");
    return NO;
  }
  std::lock_guard<std::mutex> lock(_state->mutex);
  id<MTLCommandBuffer> command = sink.metal3;
  // Claim this frame's ring slot: wait until the frame kRingSlots back has finished with it, so the
  // memcpy below cannot land under a command buffer that is still reading those arguments. Under
  // normal pacing at most one or two frames are in flight and this never blocks.
  // Held until the GPU is done with the slot; if this frame fails to encode, the destructor gives
  // the token straight back rather than leaking it.
  struct RingRelease {
    SessionState* state = nullptr;
    void release() {
      if (state == nullptr) return;
      state->frames_in_flight.fetch_sub(1, std::memory_order_relaxed);
      dispatch_semaphore_signal(state->ring_slots);
      state = nullptr;
    }
    ~RingRelease() { release(); }
  } ring_release;
  if (_state->ring_slots != nil && command != nil) {
    if (dispatch_semaphore_wait(_state->ring_slots, DISPATCH_TIME_NOW) != 0) {
      _state->ring_waits.fetch_add(1, std::memory_order_relaxed);
      if (dispatch_semaphore_wait(_state->ring_slots,
                                  dispatch_time(DISPATCH_TIME_NOW, 250 * NSEC_PER_MSEC)) != 0) {
        // The queue is wedged rather than merely deep; fall back for this frame instead of
        // overwriting an executing frame's arguments.
        _state->ring_timeouts.fetch_add(1, std::memory_order_relaxed);
        SetError(error, 32, @"frame ring stalled");
        return NO;
      }
    }
    ring_release.state = _state.get();
    const int flight = _state->frames_in_flight.fetch_add(1, std::memory_order_relaxed) + 1;
    if (flight > _state->frames_in_flight_max) _state->frames_in_flight_max = flight;
  }
  // Per-frame ring slot (descriptors, argument buffers, Metal 4 argument tables)
  // before the first FrameDescriptor/FrameScratch of this frame.
  _state->ring_slot = static_cast<NSUInteger>(_state->frames_started++ % kRingSlots);
  _state->ring_next = 0;
  _state->table_next = 0;
  if ((command == nil && sink.metal4 == nil) || (command != nil && command.device != _state->device) ||
      !Fits(color, color_offset, _state->geometry.color_bytes()) ||
      !Fits(motion, motion_offset, _state->geometry.motion_bytes()) ||
      !Fits(depth, depth_offset, _state->geometry.depth_bytes()) ||
      !Fits(exposure, exposure_offset, kExposureBytes) ||
      !Fits(output, output_offset, _state->geometry.output_bytes())) {
    SetError(error, 19, @"frame buffer contract mismatch");
    return NO;
  }
  const bool baseline_profile = !depth_reversed && jitter_x == 0.0f &&
                                jitter_y == 0.0f && motion_scale_x == 1.0f &&
                                motion_scale_y == 1.0f;
  const bool jitter_profile = !depth_reversed && jitter_x == 0.125f &&
                              jitter_y == 0.277777791f && motion_scale_x == 1.0f &&
                              motion_scale_y == 1.0f;
  // Proven motion-vector scale: the render size (every RTX profile capture
  // used NEURALSTUDIO_SR_MV_SCALE = render size).
  const float proven_scale_x = static_cast<float>(_state->geometry.input_width);
  const float proven_scale_y = static_cast<float>(_state->geometry.input_height);
  const bool motion_profile = !depth_reversed && jitter_x == 0.0f &&
                              jitter_y == 0.0f && motion_scale_x == proven_scale_x &&
                              motion_scale_y == proven_scale_y;
  const bool depth_profile = depth_reversed && jitter_x == 0.0f &&
                             jitter_y == 0.0f && motion_scale_x == 1.0f &&
                             motion_scale_y == 1.0f;
  // Non-exact (hardware) live paths may run with any finite sub-pixel jitter
  // and either motion-vector sign (NEURALSTUDIO_SR_ALLOW_ANY_JITTER=1).
  const char* any_jitter = std::getenv("NEURALSTUDIO_SR_ALLOW_ANY_JITTER");
  // A native game reports its own motion-vector convention: Lies of P uses 1,1 (motion already in
  // pixels) where every RTX capture used the render size. The arithmetic downstream does not care -
  // the input words are derived as motion_scale / render_ratio - so this is a policy about what has
  // been verified, not a numerical requirement.
  //
  // NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE=1 accepts any finite non-zero scale and says plainly
  // that the result is unverified: there is no RTX oracle for a native title, so nothing can prove
  // the output correct, only plausible. Default off, so RE9 keeps its guarantees.
  const char* any_scale = std::getenv("NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE");
  const bool scale_relaxed = any_scale != nullptr && *any_scale == '1' &&
                             std::isfinite(motion_scale_x) && std::isfinite(motion_scale_y) &&
                             motion_scale_x != 0.0f && motion_scale_y != 0.0f;
  const bool scale_ok = scale_relaxed || (std::fabs(motion_scale_x) == proven_scale_x &&
                                          std::fabs(motion_scale_y) == proven_scale_y);
  const bool relaxed = any_jitter != nullptr && *any_jitter == '1' && depth_reversed &&
                       std::isfinite(jitter_x) && std::isfinite(jitter_y) &&
                       std::fabs(jitter_x) <= 0.5f && std::fabs(jitter_y) <= 0.5f && scale_ok;
  const bool combined_profile =
      relaxed || (depth_reversed && scale_ok &&
                  NeuralStudioDLSSSRIsProvenLiveJitter(jitter_x, jitter_y));
  const bool temporal_profile = combined_profile;
  if ((reset && !baseline_profile && !jitter_profile && !motion_profile &&
       !depth_profile && !combined_profile) ||
      (!reset && !temporal_profile)) {
    SetError(error, 25, @"dynamic frame profile is not RTX-proven");
    return NO;
  }

  if (!reset && !_state->temporal_state_initialized) {
    SetError(error, 26, @"temporal frame requires an accepted reset frame");
    return NO;
  }
  if (reset) {
    // These fills are the same kind of CPU write as the one removed below, and they are not
    // optional - they initialise the temporal state. So drain first: take the remaining ring
    // tokens, which are only returned once a frame's command buffer completes, and the arenas
    // are ours alone. A reset is a discontinuity the game already pays for, so the wait costs
    // nothing in steady state. Metal 4 holds no token and has no completion handler to return
    // one, so draining there would deadlock - it is skipped, and left as a known gap.
    int drained = 0;
    if (_state->ring_slots != nil && command != nil) {
      while (drained < static_cast<int>(kRingSlots) - 1 &&
             dispatch_semaphore_wait(_state->ring_slots,
                                     dispatch_time(DISPATCH_TIME_NOW, 250 * NSEC_PER_MSEC)) == 0) {
        ++drained;
      }
    }
    Fill(_state->arena6, 0x5a);
    Fill(_state->arena7, 0x5a);
    Fill(_state->arena8, 0x5a);
    Fill(_state->internal3.storage, 0);
    Fill(_state->internal1.storage, 0);
    Fill(_state->internal5.storage, 0);
    Fill(_state->internal4.storage, 0x5a);
    Fill(_state->internal2.storage, 0x5a);
    Fill(_state->internal_exposure.storage, 0x5a);
    for (int i = 0; i < drained; ++i) dispatch_semaphore_signal(_state->ring_slots);
  }
  // The output buffer is deliberately NOT cleared here. It used to be memset to 0x5a every
  // frame, which is a CPU write into a Shared buffer that an in-flight frame's dispatches are
  // still writing and its export kernel is still reading - Metal's hazard tracking does not
  // cover CPU writes, and up to three of our command buffers are in flight at once. When the
  // timing lined up, the export copied the fill instead of the frame: 0x5a5a is half 203.25,
  // which lands on screen as blown-out white. That is the white flash, and a partially
  // overwritten buffer is the accompanying artefact.
  //
  // Nothing depended on the fill: re-running the four-frame RTX temporal receipts with the
  // poison byte changed to 0xa5 produced byte-identical output, so every output pixel is
  // written by the pipeline and none of them ever came from the fill.

  const std::uint32_t iw = static_cast<std::uint32_t>(_state->geometry.input_width);
  const std::uint32_t ih = static_cast<std::uint32_t>(_state->geometry.input_height);
  const std::uint32_t ow = static_cast<std::uint32_t>(_state->geometry.output_width);
  const std::uint32_t oh = static_cast<std::uint32_t>(_state->geometry.output_height);
  id<MTLBuffer> color_descriptor = FrameDescriptor(_state.get(), color.gpuAddress + color_offset, iw, ih, 16, 4);
  id<MTLBuffer> motion_descriptor = FrameDescriptor(_state.get(), motion.gpuAddress + motion_offset, iw, ih, 16, 2);
  id<MTLBuffer> depth_descriptor = FrameDescriptor(_state.get(), depth.gpuAddress + depth_offset, iw, ih, 32, 1);
  id<MTLBuffer> exposure_descriptor = FrameDescriptor(_state.get(), exposure.gpuAddress + exposure_offset, 1, 1, 32, 1);
  id<MTLBuffer> output_descriptor = FrameDescriptor(_state.get(), output.gpuAddress + output_offset, ow, oh, 16, 4);
  if (color_descriptor == nil || motion_descriptor == nil || depth_descriptor == nil ||
      exposure_descriptor == nil || output_descriptor == nil) {
    SetError(error, 20, @"live resource descriptor allocation failed");
    return NO;
  }

  const std::uint64_t frame_index = reset ? 0 : _state->encoded_frame_count;
  const bool even_frame = (frame_index & 1u) == 0;
  DescribedBuffer& prior_color_history =
      even_frame ? _state->internal3 : _state->internal4;
  DescribedBuffer& next_color_history =
      even_frame ? _state->internal4 : _state->internal3;
  DescribedBuffer& prior_scalar_history =
      even_frame ? _state->internal1 : _state->internal2;
  DescribedBuffer& next_scalar_history =
      even_frame ? _state->internal2 : _state->internal1;
  NSMutableData* input_arguments = [_state->input_template mutableCopy];
  const Geometry& geometry = _state->geometry;
  const bool rerender = geometry.input_width != geometry.reference_width ||
                        geometry.input_height != geometry.reference_height;
  if (rerender) PatchRenderFields(input_arguments, false, geometry);
  std::uint32_t input_flags = 0;
  std::memcpy(&input_flags, input_arguments.bytes, sizeof(input_flags));
  input_flags = (input_flags & ~1u) | (depth_reversed ? 1u : 0u);
  std::memcpy(input_arguments.mutableBytes, &input_flags, sizeof(input_flags));
  // DLSS derives the motion scale words as scale / ratio (ratio words at +40/+44,
  // render * float32(1/output)); this reproduces the captured values exactly at
  // 960x540, 1280x720 and 1706x960 -> 1080p.
  float input_ratio_x = 0.0f;
  float input_ratio_y = 0.0f;
  std::memcpy(&input_ratio_x,
              static_cast<const std::uint8_t*>(input_arguments.bytes) + 40, 4);
  std::memcpy(&input_ratio_y,
              static_cast<const std::uint8_t*>(input_arguments.bytes) + 44, 4);
  float input_scale_x = motion_scale_x / input_ratio_x;
  float input_scale_y = motion_scale_y / input_ratio_y;
  std::memcpy(static_cast<std::uint8_t*>(input_arguments.mutableBytes) + 8,
              &input_scale_x, 4);
  std::memcpy(static_cast<std::uint8_t*>(input_arguments.mutableBytes) + 12,
              &input_scale_y, 4);
  std::memcpy(static_cast<std::uint8_t*>(input_arguments.mutableBytes) + 24,
              &jitter_x, 4);
  std::memcpy(static_cast<std::uint8_t*>(input_arguments.mutableBytes) + 28,
              &jitter_y, 4);
  const std::uint32_t reset_flag = reset ? 1u : 0u;
  std::memcpy(static_cast<std::uint8_t*>(input_arguments.mutableBytes) + 32,
              &reset_flag, 4);
  Patch64(input_arguments, 168, color_descriptor.gpuAddress);
  Patch64(input_arguments, 176, prior_color_history.descriptor.gpuAddress);
  Patch64(input_arguments, 184, motion_descriptor.gpuAddress);
  Patch64(input_arguments, 192, depth_descriptor.gpuAddress);
  Patch64(input_arguments, 200, exposure_descriptor.gpuAddress);
  Patch64(input_arguments, 208,
          reset ? exposure_descriptor.gpuAddress
                : _state->internal_exposure.descriptor.gpuAddress);
  Patch64(input_arguments, 216, exposure_descriptor.gpuAddress);
  Patch64(input_arguments, 224, _state->internal5.descriptor.gpuAddress);
  Patch64(input_arguments, 240, Address(_state->arena6));
  id<MTLBuffer> input_descriptor = FrameScratch(_state.get(), input_arguments.bytes, input_arguments.length);

  NSMutableData* output_arguments = [_state->output_template mutableCopy];
  if (rerender) PatchRenderFields(output_arguments, true, geometry);
  float output_ratio_x = 0.0f;
  float output_ratio_y = 0.0f;
  std::memcpy(&output_ratio_x,
              static_cast<const std::uint8_t*>(output_arguments.bytes) + 40, 4);
  std::memcpy(&output_ratio_y,
              static_cast<const std::uint8_t*>(output_arguments.bytes) + 44, 4);
  float output_scale_x = motion_scale_x / output_ratio_x;
  float output_scale_y = motion_scale_y / output_ratio_y;
  const float jitter_bias_x = -0.5f + jitter_x;
  const float jitter_bias_y = -0.5f + jitter_y;
  const float previous_jitter_bias_x =
      reset ? -0.5f : -0.5f + _state->previous_jitter_x;
  std::memcpy(output_arguments.mutableBytes, &output_scale_x, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 4,
              &output_scale_y, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 16,
              &jitter_bias_x, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 20,
              &jitter_bias_y, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 24,
              &previous_jitter_bias_x, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 28,
              &previous_jitter_bias_x, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 92,
              &jitter_y, 4);
  std::memcpy(static_cast<std::uint8_t*>(output_arguments.mutableBytes) + 32,
              &reset_flag, 4);
  Patch64(output_arguments, 200, color_descriptor.gpuAddress);
  Patch64(output_arguments, 224, prior_color_history.descriptor.gpuAddress);
  Patch64(output_arguments, 232, prior_scalar_history.descriptor.gpuAddress);
  Patch64(output_arguments, 240, motion_descriptor.gpuAddress);
  Patch64(output_arguments, 248, depth_descriptor.gpuAddress);
  Patch64(output_arguments, 256, exposure_descriptor.gpuAddress);
  Patch64(output_arguments, 264,
          reset ? exposure_descriptor.gpuAddress
                : _state->internal_exposure.descriptor.gpuAddress);
  Patch64(output_arguments, 272, Address(_state->arena6));
  Patch64(output_arguments, 280, Address(_state->constants, 327680));
  Patch64(output_arguments, 288, next_color_history.descriptor.gpuAddress);
  Patch64(output_arguments, 296, next_scalar_history.descriptor.gpuAddress);
  Patch64(output_arguments, 304, _state->internal5.descriptor.gpuAddress);
  Patch64(output_arguments, 312, output_descriptor.gpuAddress);
  id<MTLBuffer> hiluma_output_descriptor = FrameScratch(_state.get(), output_arguments.bytes, output_arguments.length);
  if (const char* args_dump = std::getenv("NEURALSTUDIO_SR_TEMPORAL_ARGS_DUMP_DIR")) {
    if (*args_dump != '\0') {
      NSString* base = [NSString stringWithUTF8String:args_dump];
      [input_arguments writeToFile:[base stringByAppendingFormat:@"/frame-%03llu-input-args.bin",
                                    static_cast<unsigned long long>(frame_index)] atomically:NO];
      [output_arguments writeToFile:[base stringByAppendingFormat:@"/frame-%03llu-output-args.bin",
                                     static_cast<unsigned long long>(frame_index)] atomically:NO];
    }
  }

  NSMutableData* exposure_arguments = [_state->exposure_template mutableCopy];
  Patch64(exposure_arguments, 0, _state->internal_exposure.descriptor.gpuAddress);
  Patch64(exposure_arguments, 8, exposure_descriptor.gpuAddress);
  id<MTLBuffer> exposure_copy_descriptor =
      FrameScratch(_state.get(), exposure_arguments.bytes, exposure_arguments.length);
  if (input_descriptor == nil || hiluma_output_descriptor == nil ||
      exposure_copy_descriptor == nil) {
    SetError(error, 21, @"boundary argument allocation failed");
    return NO;
  }

  _state->retained_frame_buffers = {
      color_descriptor, motion_descriptor, depth_descriptor, exposure_descriptor,
      output_descriptor, input_descriptor, hiluma_output_descriptor,
      exposure_copy_descriptor, color, motion, depth, exposure, output};

  id<MTLComputePipelineState> input_pipeline = depth_reversed
      ? _state->depth_inverted_input_pipeline
      : _state->input_pipeline;
  id<MTLComputePipelineState> output_pipeline = depth_reversed
      ? _state->depth_inverted_output_pipeline
      : _state->output_pipeline;
  std::vector<std::pair<id<MTLBuffer>, NSUInteger>> input_extras = {
      {_state->rcp_rom, 4}};
  if (_state->mufu_lg2_trace != nil && _state->mufu_ex2_trace != nil) {
    input_extras.push_back({_state->mufu_lg2_trace, 5});
    input_extras.push_back({_state->mufu_ex2_trace, 6});
  }
  if (_state->mufu_lg2_keys != nil) {
    input_extras.push_back({_state->mufu_lg2_keys, 5});
    input_extras.push_back({_state->mufu_lg2_values, 6});
    input_extras.push_back({_state->mufu_ex2_keys, 7});
    input_extras.push_back({_state->mufu_ex2_values, 8});
    input_extras.push_back({_state->mufu_lg2_misses, 9});
    input_extras.push_back({_state->mufu_ex2_misses, 10});
  }
  if (_state->texture_weight_table != nil) {
    input_extras.push_back({_state->texture_weight_table, 11});
  }
  if (_state->temporal_scalar_debug != nil) {
    input_extras.push_back({_state->temporal_scalar_debug, 13});
  }
  _state->dispatch_timing.next_index = 0;
  g_dispatch_timing = &_state->dispatch_timing;
  Encode(sink, input_pipeline, input_descriptor,
         _state->geometry.hiluma_input_grid, MTLSizeMake(16, 16, 1),
         {{_state->arena6.buffer, MTLResourceUsageWrite},
          {color, MTLResourceUsageRead},
          {prior_color_history.storage.buffer, MTLResourceUsageRead},
          {motion, MTLResourceUsageRead},
          {depth, MTLResourceUsageRead},
          {exposure, MTLResourceUsageRead},
          {_state->internal5.storage.buffer, MTLResourceUsageRead},
          {color_descriptor, MTLResourceUsageRead},
          {prior_color_history.descriptor, MTLResourceUsageRead},
          {motion_descriptor, MTLResourceUsageRead},
          {depth_descriptor, MTLResourceUsageRead},
          {exposure_descriptor, MTLResourceUsageRead},
          {_state->internal5.descriptor, MTLResourceUsageRead},
          {_state->rcp_rom, MTLResourceUsageRead},
          {_state->mufu_lg2_trace, MTLResourceUsageRead | MTLResourceUsageWrite},
          {_state->mufu_ex2_trace, MTLResourceUsageRead | MTLResourceUsageWrite},
          {_state->mufu_lg2_keys, MTLResourceUsageRead},
          {_state->mufu_lg2_values, MTLResourceUsageRead},
          {_state->mufu_ex2_keys, MTLResourceUsageRead},
          {_state->mufu_ex2_values, MTLResourceUsageRead},
          {_state->mufu_lg2_misses, MTLResourceUsageWrite},
          {_state->mufu_ex2_misses, MTLResourceUsageWrite},
          {_state->texture_weight_table, MTLResourceUsageRead},
          {_state->temporal_scalar_debug, MTLResourceUsageWrite}},
         input_extras);
  if (!_state->arena6_stage_snapshots.empty()) {
    EncodeArena6Snapshot(sink, _state->arena6,
                         _state->arena6_stage_snapshots[0]);
    EncodeArena6Snapshot(sink, _state->arena7,
                         _state->arena7_stage_snapshots[0]);
    EncodeArena6Snapshot(sink, _state->arena8,
                         _state->arena8_stage_snapshots[0]);
  }

  NSUInteger pwin_index = 0;
  for (const Stage& stage : _state->pwin_stages) {
    std::vector<std::pair<id<MTLBuffer>, NSUInteger>> extras;
    if (stage.exact_math) extras = {{_state->ex2_lut, 1}, {_state->half_rcp_lut, 2}};
    if (_state->temporal_trace != nil) extras.push_back({_state->temporal_trace, 12});
    if (!_state->pwin_stats.empty()) extras.push_back({_state->pwin_stats[pwin_index], 13});
    extras.push_back({_state->pwin_scratch, 14});
    extras.push_back({_state->weights_f32_a, 15});
    extras.push_back({_state->weights_f32_b, 16});
    if (stage.window_grid_by_phase[frame_index % kPwinShiftPeriod] != nil) {
      extras.push_back({stage.window_grid_by_phase[frame_index % kPwinShiftPeriod], 17});
    }
    const std::vector<Stage::PipelineDispatch>* phase_dispatches =
        stage.pipeline_dispatches_by_phase.empty()
            ? nullptr
            : &stage.pipeline_dispatches_by_phase[frame_index % kPwinShiftPeriod];
    if (phase_dispatches != nullptr && !phase_dispatches->empty()) {
      const std::vector<std::pair<id<MTLResource>, MTLResourceUsage>> pipeline_resources = {
          {_state->arena6.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
          {_state->arena7.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
          {_state->arena8.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
          {_state->weights.buffer, MTLResourceUsageRead},
          {_state->pipeline_scratch, MTLResourceUsageRead | MTLResourceUsageWrite}};
      std::vector<std::pair<id<MTLResource>, MTLResourceUsage>> resources = pipeline_resources;
      for (id<MTLBuffer> w : _state->pipeline_weights) resources.push_back({w, MTLResourceUsageRead});
      for (const Stage::PipelineDispatch& d : *phase_dispatches) EncodeDispatch(sink, d, resources);
      if (!_state->arena6_stage_snapshots.empty()) {
        EncodeArena6Snapshot(sink, _state->arena6, _state->arena6_stage_snapshots[pwin_index + 1]);
        EncodeArena6Snapshot(sink, _state->arena7, _state->arena7_stage_snapshots[pwin_index + 1]);
        EncodeArena6Snapshot(sink, _state->arena8, _state->arena8_stage_snapshots[pwin_index + 1]);
      }
      ++pwin_index;
      continue;
    }
    Encode(sink, stage.pipeline,
           stage.arguments_by_phase[frame_index % kPwinShiftPeriod],
           stage.dispatch_by_phase[frame_index % kPwinShiftPeriod],
           MTLSizeMake(stage.config.block[0], stage.config.block[1], stage.config.block[2]),
           {{_state->arena6.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
            {_state->arena7.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
            {_state->arena8.buffer, MTLResourceUsageRead | MTLResourceUsageWrite},
            {_state->weights.buffer, MTLResourceUsageRead},
            {_state->ex2_lut, MTLResourceUsageRead},
            {_state->half_rcp_lut, MTLResourceUsageRead},
            {_state->temporal_trace, MTLResourceUsageWrite}},
           extras);
    if (!_state->arena6_stage_snapshots.empty()) {
      EncodeArena6Snapshot(sink, _state->arena6,
                           _state->arena6_stage_snapshots[pwin_index + 1]);
      EncodeArena6Snapshot(sink, _state->arena7,
                           _state->arena7_stage_snapshots[pwin_index + 1]);
      EncodeArena6Snapshot(sink, _state->arena8,
                           _state->arena8_stage_snapshots[pwin_index + 1]);
    }
    ++pwin_index;
  }

  std::vector<std::pair<id<MTLBuffer>, NSUInteger>> output_extras = {
      {_state->ex2_lut, 1}, {_state->rsqrt_rom, 2}, {_state->rcp_rom, 4}};
  if (_state->mufu_lg2_keys != nil) {
    output_extras.push_back({_state->mufu_lg2_keys, 5});
    output_extras.push_back({_state->mufu_lg2_values, 6});
    output_extras.push_back({_state->mufu_ex2_keys, 7});
    output_extras.push_back({_state->mufu_ex2_values, 8});
    output_extras.push_back({_state->mufu_lg2_misses, 9});
    output_extras.push_back({_state->mufu_ex2_misses, 10});
  }
  if (_state->texture_weight_table != nil) {
    output_extras.push_back({_state->texture_weight_table, 11});
  }
  if (_state->temporal_trace != nil) {
    output_extras.push_back({_state->temporal_trace, 12});
  }
  // NEURALSTUDIO_SR_OUTPUT_REPEAT=<n>: encode the output dispatch n times instead of once.
  //
  // A four-frame capture yields four GPU timings per process, and run-to-run spread on a machine
  // doing anything else at all is over a millisecond - wider than every effect worth chasing in
  // this kernel, which is why three "measurements" of the same change came back 6.43, 7.29 and 7.73.
  // Repeating the dispatch multiplies the signal while leaving the noise where it is: at n=32 a
  // 0.05 ms effect becomes a 1.6 ms one.
  //
  // It is sound only because this dispatch is idempotent - it reads prior_* and arena6, writes
  // next_*, internal5 and output, and reads nothing it writes. The receipt gate checks that claim
  // rather than trusting it: if repeating changed the output, the four hashes would move.
  for (NSUInteger repeat = 0; repeat < OutputRepeat(); ++repeat)
  Encode(sink, output_pipeline, hiluma_output_descriptor,
         _state->geometry.hiluma_output_grid, MTLSizeMake(16, 16, 1),
         {{_state->arena6.buffer, MTLResourceUsageRead},
          {_state->constants.buffer, MTLResourceUsageRead},
          {color, MTLResourceUsageRead},
          {prior_color_history.storage.buffer, MTLResourceUsageRead},
          {prior_scalar_history.storage.buffer, MTLResourceUsageRead},
          {motion, MTLResourceUsageRead},
          {depth, MTLResourceUsageRead},
          {exposure, MTLResourceUsageRead},
          {next_color_history.storage.buffer, MTLResourceUsageWrite},
          {next_scalar_history.storage.buffer, MTLResourceUsageWrite},
          {_state->internal5.storage.buffer, MTLResourceUsageWrite},
          {output, MTLResourceUsageWrite},
          {color_descriptor, MTLResourceUsageRead},
          {prior_color_history.descriptor, MTLResourceUsageRead},
          {prior_scalar_history.descriptor, MTLResourceUsageRead},
          {motion_descriptor, MTLResourceUsageRead},
          {depth_descriptor, MTLResourceUsageRead},
          {exposure_descriptor, MTLResourceUsageRead},
          {next_color_history.descriptor, MTLResourceUsageRead},
          {next_scalar_history.descriptor, MTLResourceUsageRead},
          {_state->internal5.descriptor, MTLResourceUsageRead},
          {output_descriptor, MTLResourceUsageRead},
          {_state->ex2_lut, MTLResourceUsageRead},
          {_state->rsqrt_rom, MTLResourceUsageRead},
          {_state->rcp_rom, MTLResourceUsageRead},
          {_state->mufu_lg2_keys, MTLResourceUsageRead},
          {_state->mufu_lg2_values, MTLResourceUsageRead},
          {_state->mufu_ex2_keys, MTLResourceUsageRead},
          {_state->mufu_ex2_values, MTLResourceUsageRead},
          {_state->mufu_lg2_misses, MTLResourceUsageWrite},
          {_state->mufu_ex2_misses, MTLResourceUsageWrite},
          {_state->texture_weight_table, MTLResourceUsageRead},
          {_state->temporal_trace, MTLResourceUsageWrite}},
         output_extras);

  Encode(sink, _state->exposure_pipeline, exposure_copy_descriptor,
         MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1),
         {{_state->internal_exposure.storage.buffer, MTLResourceUsageWrite},
          {_state->internal_exposure.descriptor, MTLResourceUsageRead},
          {exposure, MTLResourceUsageRead},
          {exposure_descriptor, MTLResourceUsageRead}});

  EndMetal3Encoder(sink);
  g_dispatch_timing = nullptr;
  // Hosts such as D3DMetal may create command buffers with unretained
  // references; keep every per-frame resource alive until the GPU is done.
  NSArray* per_frame_resources = @[
      color_descriptor, motion_descriptor, depth_descriptor, exposure_descriptor,
      output_descriptor, input_descriptor, hiluma_output_descriptor,
      exposure_copy_descriptor, color, motion, depth, exposure, output];
  SessionState* ring_owner = ring_release.state;
  ring_release.state = nullptr;   // the GPU owns the slot from here; the handler below returns it
  if (command != nil) [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    (void)completed;
    (void)per_frame_resources;
    if (ring_owner != nullptr) {
      ring_owner->frames_in_flight.fetch_sub(1, std::memory_order_relaxed);
      dispatch_semaphore_signal(ring_owner->ring_slots);
    }
  }];
  _state->temporal_state_initialized = true;
  _state->encoded_frame_count = frame_index + 1;
  _state->previous_jitter_x = jitter_x;
  return YES;
}

- (void)neuralStudioRingStatsWaits:(uint64_t*)waits
                          timeouts:(uint64_t*)timeouts
                       maxInFlight:(int*)maxInFlight {
  if (_state == nullptr) return;
  if (waits != nullptr) *waits = _state->ring_waits.load(std::memory_order_relaxed);
  if (timeouts != nullptr) *timeouts = _state->ring_timeouts.load(std::memory_order_relaxed);
  if (maxInFlight != nullptr) *maxInFlight = _state->frames_in_flight_max;
}

- (BOOL)encodeFrameToCommandBuffer:(id<MTLCommandBuffer>)command
                        colorBuffer:(id<MTLBuffer>)color
                        colorOffset:(NSUInteger)color_offset
                       motionBuffer:(id<MTLBuffer>)motion
                       motionOffset:(NSUInteger)motion_offset
                        depthBuffer:(id<MTLBuffer>)depth
                        depthOffset:(NSUInteger)depth_offset
                     exposureBuffer:(id<MTLBuffer>)exposure
                     exposureOffset:(NSUInteger)exposure_offset
                       outputBuffer:(id<MTLBuffer>)output
                       outputOffset:(NSUInteger)output_offset
                      jitterOffsetX:(float)jitter_x
                      jitterOffsetY:(float)jitter_y
                 motionVectorScaleX:(float)motion_scale_x
                 motionVectorScaleY:(float)motion_scale_y
                              reset:(BOOL)reset
                      depthReversed:(BOOL)depth_reversed
                              error:(NSError**)error {
  FrameSink sink;
  sink.metal3 = command;
  sink.state = _state.get();
  return [self encodeFrameWithSink:sink colorBuffer:color colorOffset:color_offset motionBuffer:motion
                      motionOffset:motion_offset depthBuffer:depth depthOffset:depth_offset
                    exposureBuffer:exposure exposureOffset:exposure_offset outputBuffer:output
                      outputOffset:output_offset jitterOffsetX:jitter_x jitterOffsetY:jitter_y
                motionVectorScaleX:motion_scale_x motionVectorScaleY:motion_scale_y reset:reset
                     depthReversed:depth_reversed error:error];
}

// Metal 4: the caller's command buffer must already carry residency sets for
// the frame buffers it passes; the session adds its own set, opens one compute
// encoder, waits for all prior queue work, encodes the frame, and closes it.
- (BOOL)encodeFrameToMTL4CommandBuffer:(id<MTL4CommandBuffer>)command
                            colorBuffer:(id<MTLBuffer>)color
                            colorOffset:(NSUInteger)color_offset
                           motionBuffer:(id<MTLBuffer>)motion
                           motionOffset:(NSUInteger)motion_offset
                            depthBuffer:(id<MTLBuffer>)depth
                            depthOffset:(NSUInteger)depth_offset
                         exposureBuffer:(id<MTLBuffer>)exposure
                         exposureOffset:(NSUInteger)exposure_offset
                           outputBuffer:(id<MTLBuffer>)output
                           outputOffset:(NSUInteger)output_offset
                          jitterOffsetX:(float)jitter_x
                          jitterOffsetY:(float)jitter_y
                     motionVectorScaleX:(float)motion_scale_x
                     motionVectorScaleY:(float)motion_scale_y
                                  reset:(BOOL)reset
                          depthReversed:(BOOL)depth_reversed
                                  error:(NSError**)error {
  if (_state == nullptr || _state->residency == nil || command == nil) {
    SetError(error, 33, @"Metal 4 encode unavailable (no residency set or command buffer)");
    return NO;
  }
  [command useResidencySet:_state->residency];
  id<MTL4ComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (encoder == nil) {
    SetError(error, 33, @"Metal 4 compute encoder unavailable");
    return NO;
  }
  [encoder barrierAfterQueueStages:MTLStageAll
                      beforeStages:MTLStageDispatch
                 visibilityOptions:MTL4VisibilityOptionDevice];
  FrameSink sink;
  sink.metal4 = encoder;
  sink.metal4_command = command;
  sink.metal4_split = std::getenv("NEURALSTUDIO_SR_METAL4_SPLIT") != nullptr;
  sink.state = _state.get();
  const BOOL encoded = [self encodeFrameWithSink:sink colorBuffer:color colorOffset:color_offset motionBuffer:motion
                                    motionOffset:motion_offset depthBuffer:depth depthOffset:depth_offset
                                  exposureBuffer:exposure exposureOffset:exposure_offset outputBuffer:output
                                    outputOffset:output_offset jitterOffsetX:jitter_x jitterOffsetY:jitter_y
                              motionVectorScaleX:motion_scale_x motionVectorScaleY:motion_scale_y reset:reset
                                   depthReversed:depth_reversed error:error];
  [sink.metal4 endEncoding];
  return encoded;
}

@end
