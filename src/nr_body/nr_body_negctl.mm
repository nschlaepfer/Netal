// nr_body_negctl.mm — fail-closed negative controls that reach the RUNTIME's own checks (not the file adapter): each case builds valid-looking
// caller inputs from the descriptor's sizes with one deliberate defect and expects nr_body_run_frame to reject it with code 2, the named reason,
// no output, and no GPU dispatch (wall bound). No fixture data is read: buffers are filled with a constant. Usage: nr_body_negctl CHAIN.json
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_body_runtime.h"
#include "nr_temporal/nr_temporal.h"
#include <chrono>
#include <functional>
#include <vector>
#include <cstring>
#include <iostream>
#include <string>
static std::string JStr(NSDictionary* d, const char* k) { NSString* v = d[[NSString stringWithUTF8String:k]]; return v ? std::string(v.UTF8String) : std::string(); }
static long JInt(NSDictionary* d, const char* k) { NSNumber* v = d[[NSString stringWithUTF8String:k]]; return v ? v.longValue : 0; }
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 2) { std::cerr << "usage: nr_body_negctl CHAIN.json\n"; return 2; }
  NSError* err = nil; NSDictionary* chain = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:&err]; if (!chain) { std::cerr << "chain json\n"; return 2; }
  NSDictionary* D = chain[@"decoder"]; if (!D) { std::cerr << "descriptor has no decoder section\n"; return 2; }
  NrBodyContext ctx; ctx.dev = MTLCreateSystemDefaultDevice(); ctx.queue = [ctx.dev newCommandQueue];
  auto buf = [&](size_t n) { id<MTLBuffer> b = [ctx.dev newBufferWithLength:n options:MTLResourceStorageModeShared]; std::memset(b.contents, 0x11, n); return b; };
  auto tex = [&](MTLPixelFormat f, unsigned w, unsigned h) { MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO]; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared; return [ctx.dev newTextureWithDescriptor:d]; };
  const unsigned tw = unsigned(JInt(D, "tex_w")), th = unsigned(JInt(D, "tex_h")); NSArray* patches = D[@"patches"];
  auto valid = [&]() { NrBodyFrameInputs in; in.production = true; in.entry = {buf(16u * 69120u), 0, 16u * 69120u, "", "negctl"};
    for (unsigned i = 0; i < 5; ++i) { const size_t bytes = size_t(JInt(patches[i], "bytes")); NrBodyRegion r{buf(bytes), 0, bytes, "", "negctl"}; if (i < 4) in.skips[i] = r; else in.block0_skip = r; }
    in.input_color = tex(MTLPixelFormatRGBA16Float, tw, th); return in; };
  // temporal fixtures (no GPU work: rejections happen in the up-front validation)
  NrTemporalState tstate_fresh, tstate_ok, tstate_broken; nr_temporal_state_create(tstate_fresh, ctx.dev, tw, th, 0xa5); nr_temporal_state_create(tstate_ok, ctx.dev, tw, th, 0xa5); nr_temporal_state_create(tstate_broken, ctx.dev, tw, th, 0xa5);
  NrTemporalState tstate_dmg; nr_temporal_state_create(tstate_dmg, ctx.dev, tw, th, 0xa5); tstate_dmg.frames_since_reset = 1; tstate_dmg.history_sha256 = "damaged:test"; tstate_dmg.last_output_sha256 = "damaged";
  tstate_ok.frames_since_reset = 1; tstate_ok.history_sha256 = "h"; tstate_ok.last_output_sha256 = "h"; tstate_broken.frames_since_reset = 1; tstate_broken.history_sha256 = "h"; tstate_broken.last_output_sha256 = "other";
  id<MTLTexture> tex_rg = tex(MTLPixelFormatRG16Float, tw, th); bool use_plain_descriptor = false, use_bad_table_descriptor = false;
  NSMutableDictionary* chain_plain = [chain mutableCopy]; { NSMutableDictionary* Dp = [D mutableCopy]; NSMutableArray* Ls = [NSMutableArray array]; for (NSDictionary* L in D[@"launches"]) { NSMutableDictionary* M = [L mutableCopy]; [M removeObjectForKey:@"lib_temporal"]; [Ls addObject:M]; } Dp[@"launches"] = Ls; chain_plain[@"decoder"] = Dp; }
  NSMutableDictionary* chain_t = [chain mutableCopy]; { NSMutableDictionary* Dt = [D mutableCopy]; NSMutableArray* Ls = [NSMutableArray array]; for (NSDictionary* L in D[@"launches"]) { NSMutableDictionary* M = [L mutableCopy]; if ([M[@"launch"] intValue] == 155 && !M[@"lib_temporal"]) M[@"lib_temporal"] = M[@"lib"]; [Ls addObject:M]; } Dt[@"launches"] = Ls; chain_t[@"decoder"] = Dt; }
  NSMutableDictionary* chain_badtbl = [chain mutableCopy]; { NSMutableDictionary* Db = [D mutableCopy]; NSMutableArray* Ls = [NSMutableArray array]; for (NSDictionary* L in D[@"launches"]) { NSMutableDictionary* M = [L mutableCopy]; if ([M[@"launch"] intValue] == 155) { if (!M[@"lib_temporal"]) M[@"lib_temporal"] = M[@"lib"]; NSMutableDictionary* T = M[@"ex2_half_table"] ? [M[@"ex2_half_table"] mutableCopy] : [NSMutableDictionary dictionary]; T[@"sha256"] = @"0000000000000000000000000000000000000000000000000000000000000000"; if (!T[@"file"]) T[@"file"] = @"/dev/null"; M[@"ex2_half_table"] = T; } [Ls addObject:M]; } Db[@"launches"] = Ls; chain_badtbl[@"decoder"] = Db; }
  struct Case { std::string name; std::string expect; std::function<void(NrBodyFrameInputs&)> defect; };
  std::vector<Case> cases = {
    {"tex-wrong-size", "input colour texture", [&](NrBodyFrameInputs& in) { in.input_color = tex(MTLPixelFormatRGBA16Float, tw / 2, th); }},
    {"tex-wrong-format", "input colour texture", [&](NrBodyFrameInputs& in) { in.input_color = tex(MTLPixelFormatRGBA32Float, tw, th); }},
    {"tex-missing", "input colour texture", [&](NrBodyFrameInputs& in) { in.input_color = nil; }},
    {"output-wrong-size", "output surface", [&](NrBodyFrameInputs& in) { in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th - 1); }},
    {"skip2-short", JStr(patches[2], "file"), [&](NrBodyFrameInputs& in) { in.skips[2].bytes -= 1; }},
    {"skip0-offset-overrun", JStr(patches[0], "file"), [&](NrBodyFrameInputs& in) { in.skips[0].offset = 4096; }},
    {"block0-missing", JStr(patches[4], "file"), [&](NrBodyFrameInputs& in) { in.block0_skip.buffer = nil; }},
    {"entry-short", "encoder entry", [&](NrBodyFrameInputs& in) { in.entry.bytes = 16u * 69120u - 512u; }},
    // v5 experimental temporal path: every reset=false request must be rejected before any GPU work unless the full contract holds
    {"temporal-no-state", "requires a temporal state", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = nullptr; }},
    {"temporal-before-any-frame", "before any frame", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_fresh; in.motion = tex_rg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"temporal-continuity-broken", "history continuity", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_broken; in.motion = tex_rg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"temporal-motion-missing", "motion texture", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_ok; in.motion = nil; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"temporal-motion-wrong-format", "motion texture", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_ok; in.motion = tex(MTLPixelFormatRGBA16Float, tw, th); in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"temporal-no-output-surface", "caller output surface", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_ok; in.motion = tex_rg; in.output_surface = nil; }},
    {"temporal-descriptor-without-lib", "lib_temporal", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_ok; in.motion = tex_rg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); use_plain_descriptor = true; }},
    {"temporal-stale-history-content", "history texture content", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_ok; in.motion = tex_rg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"context-damaged-reset-frame", "context damaged", [&](NrBodyFrameInputs& in) { (void)in; ctx.damaged = "negctl"; }},
    {"temporal-state-damaged-reset-frame", "temporal state damaged", [&](NrBodyFrameInputs& in) { in.reset = true; in.temporal = &tstate_dmg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"temporal-state-damaged-nonreset-frame", "temporal state damaged", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_dmg; in.motion = tex_rg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); }},
    {"temporal-ex2-table-wrong-sha", "ex2 half-domain table", [&](NrBodyFrameInputs& in) { in.reset = false; in.temporal = &tstate_ok; in.motion = tex_rg; in.output_surface = tex(MTLPixelFormatRGBA16Float, tw, th); use_bad_table_descriptor = true; }},
  };
  int failures = 0;
  for (auto& c : cases) { use_plain_descriptor = false; use_bad_table_descriptor = false; ctx.damaged.clear(); NrBodyFrameInputs in = valid(); c.defect(in); NrBodyResult res; const auto t0 = std::chrono::steady_clock::now();
    const int rc = nr_body_run_frame(ctx, use_plain_descriptor ? chain_plain : (use_bad_table_descriptor ? chain_badtbl : chain_t), in, nullptr, res); const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = rc == 2 && res.error.find("live input rejected") != std::string::npos && res.error.find(c.expect) != std::string::npos && res.final_surface_sha256.empty() && res.middle_view_sha256.empty() && ms < 2000.0;
    if (!ok) ++failures; std::cout << (ok ? "PASS " : "FAIL ") << c.name << ": rc " << rc << " wall " << ms << " ms :: " << res.error << "\n"; }
  std::cout << (failures ? "negative controls FAILED " : "negative controls passed ") << cases.size() - failures << "/" << cases.size() << "\n"; return failures ? 1 : 0;
} }
