// Measured pipeline limits for the launches under consideration (Opus-owned).
//
// Codex's correction: ns/thread is amortised throughput, not per-thread latency, and does NOT rule out
// structural overhead. Before proposing any specialisation this reports what the hardware actually says about
// each pipeline: static threadgroup memory, the maximum threads per threadgroup the compiler will allow, and
// the execution width. Those three decide occupancy. Nothing here is inferred.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 2) { fprintf(stderr, "usage: nr_experimental_pipeline_limits BUNDLE [launch ...]\n"); return 2; }
  const std::string bundle = argv[1];
  std::vector<long> want;
  for (int i = 2; i < argc; ++i) want.push_back(atol(argv[i]));
  if (want.empty()) want = {133, 147, 155};
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSData* d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(bundle + "/body/chain.json").c_str()]];
  if (!d) { fprintf(stderr, "cannot read the descriptor\n"); return 3; }
  NSDictionary* chain = [NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
  NSString* base = [NSString stringWithUTF8String:(bundle + "/body/").c_str()];
  auto resolve = [&](NSString* p) { return [p hasPrefix:@"/"] ? p : [base stringByAppendingString:p]; };

  printf("{\"schema\": \"neuralstudio.nr.pipeline-limits.v1\", \"device\": \"%s\",\n", dev.name.UTF8String);
  printf(" \"maxThreadgroupMemory\": %lu,\n \"pipelines\": [\n", (unsigned long)dev.maxThreadgroupMemoryLength);
  bool first = true;
  // Encoder tier kernels: the wide path is the largest measured phase, so its threadgroup-size freedom
  // decides whether the dispatch geometry can be tuned without touching the accepted kernels.
  {
    const char* wide[11] = {"nr_wide_expand", "nr_wide_activate", "nr_wide_mid", "nr_wide_mix", "nr_wide_qkv",
                            "nr_wide_normalize", "nr_wide_scores", "nr_wide_probabilities", "nr_wide_values",
                            "nr_wide_project", "nr_wide_residual"};
    for (NSString* libname : @[@"prefix.metallib", @"wide128.metallib", @"wide256.metallib"]) {
      NSString* full = [NSString stringWithFormat:@"%s/encoder/%@", argv[1], libname];
      NSError* e = nil;
      id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:full] error:&e];
      if (!lib) continue;
      for (const char* kn : wide) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kn]];
        if (!fn) continue;
        NSError* pe = nil;
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!ps) continue;
        printf("%s  {\"library\": \"%s\", \"kernel\": \"%s\", \"staticThreadgroupMemory\": %lu,"
               " \"maxTotalThreadsPerThreadgroup\": %lu, \"threadExecutionWidth\": %lu}",
               first ? "" : ",\n", libname.UTF8String, kn, (unsigned long)ps.staticThreadgroupMemoryLength,
               (unsigned long)ps.maxTotalThreadsPerThreadgroup, (unsigned long)ps.threadExecutionWidth);
        first = false;
      }
    }
  }
  std::map<std::string, id<MTLLibrary>> libs;
  /* first declared above */
  for (NSDictionary* L in chain[@"decoder"][@"launches"]) {
    const long n = [L[@"launch"] longValue];
    bool wanted = false; for (long w : want) wanted |= (w == n);
    if (!wanted) continue;
    for (NSString* key : @[@"lib", @"lib_temporal"]) {
      NSString* lp = L[key];
      if (![lp isKindOfClass:[NSString class]] || !lp.length) continue;
      NSString* full = resolve(lp);
      const std::string k = full.UTF8String;
      if (!libs.count(k)) { NSError* e = nil; libs[k] = [dev newLibraryWithURL:[NSURL fileURLWithPath:full] error:&e]; }
      id<MTLLibrary> lib = libs[k];
      if (!lib) continue;
      NSString* kern = L[@"kernel"];
      id<MTLFunction> fn = [lib newFunctionWithName:kern];
      if (!fn) continue;
      NSError* e = nil;
      id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
      if (!ps) continue;
      NSArray* b = L[@"block"]; NSArray* g = L[@"grid"];
      const unsigned long bw = [b[0] unsignedLongValue] * [b[1] unsignedLongValue] * [b[2] unsignedLongValue];
      const unsigned long threads = bw * [g[0] unsignedLongValue] * [g[1] unsignedLongValue] * [g[2] unsignedLongValue];
      printf("%s  {\"launch\": %ld, \"which\": \"%s\", \"kernel\": \"%s\","
             " \"staticThreadgroupMemory\": %lu, \"maxTotalThreadsPerThreadgroup\": %lu, \"threadExecutionWidth\": %lu,"
             " \"dispatchBlock\": %lu, \"threads\": %lu}",
             first ? "" : ",\n", n, key.UTF8String, kern.UTF8String,
             (unsigned long)ps.staticThreadgroupMemoryLength, (unsigned long)ps.maxTotalThreadsPerThreadgroup,
             (unsigned long)ps.threadExecutionWidth, bw, threads);
      first = false;
    }
  }
  printf("\n ]}\n");
  return 0;
} }
