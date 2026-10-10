// nr_single_warp_guard.mm — HOST-SIDE dispatch guard for the opt-in single-warp launch-155 candidate
// (Codex review 2026-09-08: "query the actual pipeline's threadExecutionWidth, require 32, and verify the actual
// dispatched threadgroup is exactly 32x1x1 - not merely a manifest assertion; reject a mismatch before dispatch").
//
// It creates a device and a compute pipeline and REPORTS; it dispatches nothing and takes no command buffer.
// Any harness that binds a --single-warp library must call nr_single_warp_guard_ok() before encoding, and refuse
// the frame if it returns false. Usage: nr_single_warp_guard LIB.metallib KERNEL [W H D] -> prints a JSON verdict.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <string>
#include "nr_single_warp_guard.h"
bool nr_single_warp_guard_ok(id<MTLComputePipelineState> pso, MTLSize threadsPerThreadgroup, std::string& why) {
  if (!pso) { why = "no pipeline"; return false; }
  NrSingleWarpFacts f{(unsigned long)pso.threadExecutionWidth, (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                      (unsigned long)threadsPerThreadgroup.width, (unsigned long)threadsPerThreadgroup.height, (unsigned long)threadsPerThreadgroup.depth};
  return nr_single_warp_facts_ok(f, why);
}
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 3) { fprintf(stderr, "usage: nr_single_warp_guard LIB.metallib KERNEL [W H D]\n"); return 2; }
  const NSUInteger w = argc > 3 ? atoi(argv[3]) : 32, h = argc > 4 ? atoi(argv[4]) : 1, d = argc > 5 ? atoi(argv[5]) : 1;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
  if (!lib) { printf("{\"ok\": false, \"why\": \"library: %s\"}\n", err.localizedDescription.UTF8String); return 3; }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:argv[2]]] error:&err];
  if (!pso) { printf("{\"ok\": false, \"why\": \"pipeline: %s\"}\n", err.localizedDescription.UTF8String); return 3; }
  std::string why; const bool ok = nr_single_warp_guard_ok(pso, MTLSizeMake(w, h, d), why);
  printf("{\"ok\": %s, \"why\": \"%s\", \"threadExecutionWidth\": %lu, \"maxTotalThreadsPerThreadgroup\": %lu, \"threadgroup\": [%lu, %lu, %lu], \"dispatched\": false}\n",
         ok ? "true" : "false", why.c_str(), (unsigned long)pso.threadExecutionWidth, (unsigned long)pso.maxTotalThreadsPerThreadgroup, (unsigned long)w, (unsigned long)h, (unsigned long)d);
  return ok ? 0 : 1;
} }
