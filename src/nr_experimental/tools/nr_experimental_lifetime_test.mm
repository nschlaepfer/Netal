// Lifetime, admission and reset tests for the experimental NR runtime (Opus-owned).
//
// These EXECUTE the API rather than reading the source. Two groups:
//   * "nodevice" needs no Metal device and no GPU at all - it is always safe to run, including while another
//     agent holds the GPU reservation;
//   * "device" creates a device, textures and pipelines. It allocates and compiles, and encodes into a command
//     buffer that is never committed, so it still submits NO GPU work; it is opt-in with --with-device so the
//     distinction stays visible.
// Nothing here commits or waits, and no reference or model output is ever compared.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../nr_experimental_api.h"
#include <cstdio>
#include <string>

using namespace netal_nr_experimental;
static int total = 0, failed = 0;
static void ok(const char* name, bool cond, const std::string& detail = "") {
  ++total; if (!cond) ++failed;
  printf("%s %s%s%s\n", cond ? "PASS" : "FAIL", name, detail.empty() ? "" : " :: ", detail.c_str());
}
static bool has(const std::string& hay, const char* needle) { return hay.find(needle) != std::string::npos; }

int main(int argc, const char** argv) { @autoreleasepool {
  const bool with_device = argc > 1 && std::string(argv[1]) == "--with-device";
  const std::string bundle = argc > 2 ? argv[2] : "";

  // ---------- group 1: no device, no GPU ----------
  { Report r; ContextPtr c; PrepareOptions o; o.runtime_bundle = "x";
    ok("prepare-rejects-null-device", !prepare(nil, o, c, r) && has(r.error, "no Metal device"), r.error); }
  { Report r; ContextPtr c; PrepareOptions o; o.runtime_bundle = "x"; o.width = 1280; o.height = 720;
    ok("prepare-rejects-other-resolutions", !prepare(nil, o, c, r), "pinned to 1920x1080"); }
  { Report r; ContextPtr c; PrepareOptions o;
    ok("prepare-requires-a-bundle", !prepare(nil, o, c, r), "runtime_bundle must be given"); }
  { TicketPtr t; ok("output-texture-of-a-null-ticket-is-nil", output_texture(t) == nil); }
  { Report r; TicketPtr t; complete(t, nil, r); ok("complete-of-a-null-ticket-reports", has(r.error, "no ticket"), r.error); }
  { invalidate(nullptr, "no context"); ok("invalidate-of-a-null-context-is-safe", true); }
  { Report r; ContextPtr c; TicketPtr t; FrameInputs in;
    ok("encode-without-a-context-is-rejected",
       encode(c, in, nil, t, r) == EncodeStatus::RejectedBeforeEncoding && has(r.error, "not prepared") && !t, r.error); }
  { Report r; ContextPtr c; TicketPtr t; FrameInputs in;
    encode(c, in, nil, t, r);
    ok("a-rejected-frame-reports-cpu-time", r.encode_cpu_time_valid, "encode_cpu_ms is always reported"); }
  { Report r; ContextPtr c; TicketPtr t; FrameInputs in;
    encode(c, in, nil, t, r);
    ok("a-rejected-frame-never-claims-nr", !r.model_encoded && r.model_dispatches == 0 && r.experimental); }

  if (!with_device) { printf("%d/%d passed (no-device group only; pass --with-device BUNDLE for the rest)\n", total - failed, total); return failed ? 1 : 0; }

  // ---------- group 2: device present; allocates and encodes, but commits nothing ----------
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { printf("no Metal device available\n"); return 2; }
  id<MTLCommandQueue> q = [dev newCommandQueue];
  Report r; ContextPtr ctx; PrepareOptions o; o.runtime_bundle = bundle;
  if (!prepare(dev, o, ctx, r)) { printf("FAIL prepare :: %s\n", r.error.c_str()); return 1; }
  ok("prepare-succeeds-on-the-pinned-bundle", true, "prepare_cpu_ms " + std::to_string(r.prepare_cpu_ms));
  ok("prepare-reports-identities", !r.model_identity.empty() && !r.weights_identity.empty(), r.model_identity);
  ok("prepare-labels-the-lane", has(r.arithmetic_label, "experimental") && r.experimental, r.arithmetic_label);

  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
  id<MTLTexture> colour = [dev newTextureWithDescriptor:d];
  MTLTextureDescriptor* dm = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:1920 height:1080 mipmapped:NO];
  dm.usage = MTLTextureUsageShaderRead; dm.storageMode = MTLStorageModeShared;
  id<MTLTexture> motion = [dev newTextureWithDescriptor:dm];

  { FrameInputs in; in.color = colour; in.reset = true; TicketPtr t; Report rr;
    ok("encode-rejects-a-null-command-buffer", encode(ctx, in, nil, t, rr) == EncodeStatus::RejectedBeforeEncoding && has(rr.error, "no caller command buffer"), rr.error); }
  { FrameInputs in; in.color = colour; in.reset = false; TicketPtr t; Report rr; id<MTLCommandBuffer> cb = [q commandBuffer];
    ok("temporal-without-history-is-rejected", encode(ctx, in, cb, t, rr) == EncodeStatus::RejectedBeforeEncoding && has(rr.error, "history is invalid"), rr.error); }
  { FrameInputs in; in.color = colour; in.reset = false; in.motion = motion; TicketPtr t; Report rr; id<MTLCommandBuffer> cb = [q commandBuffer];
    invalidate(ctx, "test"); encode(ctx, in, cb, t, rr);
    ok("temporal-still-needs-valid-history-even-with-motion", has(rr.error, "history is invalid"), rr.error); }
  { FrameInputs in; in.color = nil; in.reset = true; TicketPtr t; Report rr; id<MTLCommandBuffer> cb = [q commandBuffer];
    ok("encode-rejects-a-missing-colour-texture", encode(ctx, in, cb, t, rr) == EncodeStatus::RejectedBeforeEncoding && has(rr.error, "colour"), rr.error); }
  { FrameInputs in; in.color = motion; in.reset = true; TicketPtr t; Report rr; id<MTLCommandBuffer> cb = [q commandBuffer];
    ok("encode-rejects-a-wrong-format-colour-texture", encode(ctx, in, cb, t, rr) == EncodeStatus::RejectedBeforeEncoding && has(rr.error, "RGBA16Float"), rr.error); }
  { FrameInputs in; in.color = colour; in.reset = true; TicketPtr t; Report rr;
    id<MTLCommandBuffer> cb = [q commandBuffer]; [cb commit];
    ok("encode-rejects-an-already-committed-buffer", encode(ctx, in, cb, t, rr) == EncodeStatus::RejectedBeforeEncoding && has(rr.error, "already enqueued or committed"), rr.error);
    [cb waitUntilCompleted]; }
  printf("%d/%d passed\n", total - failed, total);
  return failed ? 1 : 0;
} }
