// Codex-specified failure test: a failure AFTER NR scratch work is already on the caller's command buffer.
//
// The buffer carries the game's own rendering, so the contract is: our failure must not stop the game's work
// from completing, must not publish an NR output, must not release anything early, and must leave the buffer
// committable. Game markers are written before and after the NR encode and checked after completion.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../nr_experimental_api.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace netal_nr_experimental;
static int fails = 0;
static void ok(const char* n, bool c, const std::string& d = "") {
  if (!c) ++fails;
  printf("%s %s%s%s\n", c ? "PASS" : "FAIL", n, d.empty() ? "" : " :: ", d.c_str());
}
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 2) { fprintf(stderr, "usage: nr_experimental_failure_test BUNDLE [case]\n"); return 2; }
  const std::string kase = argc > 2 ? argv[2] : "after-block0";
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  Report pr; ContextPtr ctx; PrepareOptions o; o.runtime_bundle = argv[1];
  if (!prepare(dev, o, ctx, pr)) { printf("FAIL prepare :: %s\n", pr.error.c_str()); return 3; }

  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
  id<MTLTexture> colour = [dev newTextureWithDescriptor:d];
  // Stand-ins for the game's own rendering, on the same command buffer as NR.
  id<MTLBuffer> marker = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  std::memset(marker.contents, 0, 16);

  id<MTLCommandBuffer> cb = [q commandBuffer]; cb.label = @"game+nr";
  { id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder]; [b fillBuffer:marker range:NSMakeRange(0, 8) value:0xA1]; [b endEncoding]; }

  FrameInputs in; in.color = colour; in.reset = true; in.stream_id = 7; in.frame_id = 100;
  TicketPtr t; Report er;
  const EncodeStatus st = encode(ctx, in, cb, t, er);
  ok(("failure-is-discard-only[" + kase + "]").c_str(), st == EncodeStatus::EncodedDiscardOnly, er.error.substr(0, 120));
  ok("discard-only-still-returns-a-ticket", t != nullptr);
  ok("no-output-is-published", output_texture(t) == nil, "output_texture is nil for a discard-only ticket");
  ok("caller-is-not-told-to-abandon-the-buffer",
     er.error.find("do NOT publish") != std::string::npos && er.error.find("DISCARD THE COMMAND BUFFER") == std::string::npos,
     "the instruction is to commit as normal and not publish");

  // The game keeps writing to the same buffer after our failure; this must work.
  bool second_marker_ok = true;
  @try { id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder]; [b fillBuffer:marker range:NSMakeRange(8, 8) value:0xB2]; [b endEncoding]; }
  @catch (NSException* e) { second_marker_ok = false; }
  ok("game-can-still-encode-after-our-failure", second_marker_ok, "no encoder was left open on the caller's buffer");

  { Report cr; complete(t, cb, cr); ok("completion-before-terminal-is-rejected",
      cr.error.find("terminal state") != std::string::npos, cr.error.substr(0, 90)); }

  [cb commit]; [cb waitUntilCompleted];
  ok("game-command-buffer-completes", cb.status == MTLCommandBufferStatusCompleted,
     cb.error ? cb.error.localizedDescription.UTF8String : "completed");
  const uint8_t* m = static_cast<const uint8_t*>(marker.contents);
  ok("game-work-before-nr-executed", m[0] == 0xA1 && m[7] == 0xA1);
  ok("game-work-after-nr-executed", m[8] == 0xB2 && m[15] == 0xB2);

  Report cr; complete(t, cb, cr);
  ok("discard-only-finalizes-after-completion", cr.error.find("discard-only") != std::string::npos, cr.error.substr(0, 90));
  ok("history-not-advanced", !cr.history_advanced);
  { Report cr2; complete(t, cb, cr2); ok("ticket-finalizes-exactly-once", cr2.error.find("already finalized") != std::string::npos, cr2.error); }

  // The context must be usable again after a reset, and must refuse a temporal frame first.
  FrameInputs t2; t2.color = colour; t2.reset = false; t2.stream_id = 7; t2.frame_id = 101;
  id<MTLCommandBuffer> cb2 = [q commandBuffer];
  TicketPtr tk2; Report r2;
  ok("temporal-refused-after-an-invalidated-frame",
     encode(ctx, t2, cb2, tk2, r2) == EncodeStatus::RejectedBeforeEncoding, r2.error.substr(0, 90));
  printf("%s\n", fails ? "FAILURES" : "ALL PASS");
  return fails ? 1 : 0;
} }
