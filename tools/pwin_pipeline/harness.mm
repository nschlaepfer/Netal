// Standalone PWin stage harness: runs one layer (the reference metallib or a
// replacement pipeline of dispatches) on captured arena snapshots and compares
// the arenas with the snapshot taken after that stage.
//
//   harness --bundle <runtime bundle> --stage-in <dir> --frame 000 --stage 1
//           --params pwin/enc0-arguments-template.bin --config <replay-config.bin | enc0>
//           [--reference <metallib name>]      (single dispatch of the transpiled kernel)
//           [--pipeline <json>]                 (list of dispatches, see below)
//           [--loops N]                         (timing loop after the check)
//
// Pipeline JSON: {"library": "<path.metallib>", "scratch_bytes": N,
//   "dispatches": [{"function": "name", "grid": [x,y,z], "block": [x,y,z],
//                   "buffers": [{"index": i, "name": "arena6|arena7|arena8|weights|params|scratch", "offset": o}],
//                   "bytes": [{"index": i, "u32": [..]}]}]}
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#pragma pack(push, 1)
struct Relocation { uint32_t param_offset, buffer_id; uint64_t byte_offset; };
struct ReplayConfig {
  char magic[8]; uint32_t version; char entry[64]; char debug_symbol[64]; uint32_t grid[3]; uint32_t block[3]; uint32_t relocation_count;
  Relocation relocations[8];
};
#pragma pack(pop)
static_assert(sizeof(ReplayConfig) == 296, "replay config layout");

static NSData* ReadFile(NSString* path) {
  NSData* d = [NSData dataWithContentsOfFile:path];
  if (d == nil) { fprintf(stderr, "cannot read %s\n", path.UTF8String); exit(2); }
  return d;
}

int main(int argc, char** argv) {
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
  NSString* bundle = @(opt["--bundle"].c_str());
  NSString* stage_in = @(opt["--stage-in"].c_str());
  std::string frame = opt.count("--frame") ? opt["--frame"] : "000";
  int stage = opt.count("--stage") ? atoi(opt["--stage"].c_str()) : 1;
  int loops = opt.count("--loops") ? atoi(opt["--loops"].c_str()) : 0;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];

  auto snapshot = [&](int s, const char* arena) {
    NSString* name = arena == nullptr ? [NSString stringWithFormat:@"frame-%s-stage-%02d.bin", frame.c_str(), s]
                                      : [NSString stringWithFormat:@"frame-%s-stage-%02d-%s.bin", frame.c_str(), s, arena];
    return ReadFile([stage_in stringByAppendingPathComponent:name]);
  };
  NSData* in6 = snapshot(stage - 1, nullptr);
  NSData* in7 = snapshot(stage - 1, "arena7");
  NSData* in8 = snapshot(stage - 1, "arena8");
  NSData* ex6 = snapshot(stage, nullptr);
  NSData* ex7 = snapshot(stage, "arena7");
  NSData* ex8 = snapshot(stage, "arena8");
  NSData* weights = ReadFile([bundle stringByAppendingPathComponent:@"model/weights.bin"]);
  NSData* params = ReadFile([bundle stringByAppendingPathComponent:@(opt["--params"].c_str())]);

  auto make = [&](NSData* d) {
    id<MTLBuffer> b = [device newBufferWithBytes:d.bytes length:d.length options:MTLResourceStorageModeShared];
    return b;
  };
  std::map<std::string, id<MTLBuffer>> buffers;
  buffers["arena6"] = make(in6);
  buffers["arena7"] = make(in7);
  buffers["arena8"] = make(in8);
  buffers["weights"] = make(weights);
  const std::map<std::string, uint32_t> ids = {{"arena6", 6}, {"arena7", 7}, {"arena8", 8}, {"weights", 9}};

  // params with relocations resolved to GPU addresses (same as the session)
  ReplayConfig config{};
  if (opt.count("--config")) {
    NSData* c = ReadFile([bundle stringByAppendingPathComponent:@(opt["--config"].c_str())]);
    if (c.length != sizeof(config)) { fprintf(stderr, "config size %lu != %lu\n", (unsigned long)c.length, sizeof(config)); return 2; }
    memcpy(&config, c.bytes, sizeof(config));
  } else {  // enc0 (session's Enc0Config)
    strcpy(config.entry, "dltss_pwin_enc0_layer");
    config.grid[0] = 61; config.grid[1] = 37; config.grid[2] = 1; config.block[0] = 32; config.block[1] = 1; config.block[2] = 1;
    config.relocation_count = 5;
    Relocation r[5] = {{8, 6, 0}, {24, 7, 0}, {32, 8, 43200}, {48, 8, 43200}, {64, 9, 0}};
    memcpy(config.relocations, r, sizeof(r));
  }
  std::vector<uint8_t> args(static_cast<const uint8_t*>(params.bytes), static_cast<const uint8_t*>(params.bytes) + params.length);
  for (uint32_t i = 0; i < config.relocation_count; ++i) {
    const Relocation& rel = config.relocations[i];
    id<MTLBuffer> target = nil;
    for (const auto& kv : ids) if (kv.second == rel.buffer_id) target = buffers[kv.first];
    uint64_t address = target.gpuAddress + rel.byte_offset;
    memcpy(args.data() + rel.param_offset, &address, 8);
  }
  buffers["params"] = [device newBufferWithBytes:args.data() length:args.size() options:MTLResourceStorageModeShared];

  struct Dispatch { id<MTLComputePipelineState> pso; MTLSize grid, block; std::vector<std::pair<int, std::pair<id<MTLBuffer>, NSUInteger>>> bufs;
                    std::vector<std::pair<int, std::vector<uint32_t>>> bytes; };
  std::vector<Dispatch> dispatches;
  NSError* error = nil;
  if (opt.count("--reference")) {
    NSString* lib_path = [bundle stringByAppendingPathComponent:[NSString stringWithFormat:@"metallib/%s.metallib", opt["--reference"].c_str()]];
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:lib_path] error:&error];
    if (!lib) { fprintf(stderr, "library error %s\n", error.localizedDescription.UTF8String); return 2; }
    Dispatch d;
    d.pso = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@(config.entry)] error:&error];
    if (!d.pso) { fprintf(stderr, "pipeline error %s\n", error.localizedDescription.UTF8String); return 2; }
    NSData* sidecar = [NSData dataWithContentsOfFile:[lib_path.stringByDeletingPathExtension stringByAppendingPathExtension:@"json"]];
    uint32_t block[3] = {config.block[0], config.block[1], config.block[2]};
    if (sidecar) {
      NSDictionary* json = [NSJSONSerialization JSONObjectWithData:sidecar options:0 error:nil];
      NSArray* b = json[@"threads_per_threadgroup"];
      if ([b isKindOfClass:[NSArray class]] && b.count == 3) for (int a = 0; a < 3; ++a) block[a] = [b[a] unsignedIntValue];
    }
    d.grid = MTLSizeMake(config.grid[0], config.grid[1], config.grid[2]);
    d.block = MTLSizeMake(block[0], block[1], block[2]);
    d.bufs = {{0, {buffers["params"], 0}}};
    dispatches.push_back(d);
  }
  if (opt.count("--pipeline")) {
    NSData* jd = ReadFile(@(opt["--pipeline"].c_str()));
    NSDictionary* json = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&error];
    if (!json) { fprintf(stderr, "pipeline json error %s\n", error.localizedDescription.UTF8String); return 2; }
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:json[@"library"]] error:&error];
    if (!lib) { fprintf(stderr, "library error %s\n", error.localizedDescription.UTF8String); return 2; }
    NSUInteger scratch_bytes = [json[@"scratch_bytes"] unsignedIntegerValue];
    buffers["scratch"] = [device newBufferWithLength:MAX(scratch_bytes, NSUInteger(256)) options:MTLResourceStorageModeShared];
    for (NSDictionary* fj in json[@"files"]) {  // extra file-backed buffers: {"name": ..., "path": ...}
      buffers[[fj[@"name"] UTF8String]] = make(ReadFile(fj[@"path"]));
    }
    for (NSDictionary* dj in json[@"dispatches"]) {
      Dispatch d;
      MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
      NSDictionary* consts = dj[@"constants"];
      if ([consts isKindOfClass:[NSDictionary class]] && consts.count > 0) {
        MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
        for (NSString* key in consts) {
          uint32_t v = [consts[key] unsignedIntValue];
          [values setConstantValue:&v type:MTLDataTypeUInt atIndex:[key integerValue]];
        }
        id<MTLFunction> fn = [lib newFunctionWithName:dj[@"function"] constantValues:values error:&error];
        if (fn == nil) { fprintf(stderr, "function constants error for %s: %s\n", [dj[@"function"] UTF8String], error.localizedDescription.UTF8String); return 2; }
        pd.computeFunction = fn;
      } else {
        id<MTLFunction> fn = [lib newFunctionWithName:dj[@"function"]];
        if (fn == nil) { fprintf(stderr, "function %s not found\n", [dj[@"function"] UTF8String]); return 2; }
        pd.computeFunction = fn;
      }
      pd.label = dj[@"function"];
      d.pso = [device newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&error];
      if (!d.pso) { fprintf(stderr, "pipeline error %s: %s\n", [dj[@"function"] UTF8String], error.localizedDescription.UTF8String); return 2; }
      NSArray* g = dj[@"grid"]; NSArray* b = dj[@"block"];
      d.grid = MTLSizeMake([g[0] unsignedIntegerValue], [g[1] unsignedIntegerValue], [g[2] unsignedIntegerValue]);
      d.block = MTLSizeMake([b[0] unsignedIntegerValue], [b[1] unsignedIntegerValue], [b[2] unsignedIntegerValue]);
      for (NSDictionary* bj in dj[@"buffers"]) {
        std::string name = [bj[@"name"] UTF8String];
        if (!buffers.count(name)) { fprintf(stderr, "unknown buffer %s\n", name.c_str()); return 2; }
        d.bufs.push_back({[bj[@"index"] intValue], {buffers[name], [bj[@"offset"] unsignedIntegerValue]}});
      }
      for (NSDictionary* cj in dj[@"bytes"]) {
        std::vector<uint32_t> v; for (NSNumber* n in cj[@"u32"]) v.push_back([n unsignedIntValue]);
        d.bytes.push_back({[cj[@"index"] intValue], v});
      }
      dispatches.push_back(d);
    }
  }
  if (dispatches.empty()) { fprintf(stderr, "nothing to run\n"); return 2; }

  // --one-encoder: all dispatches in one MTLDispatchTypeSerial encoder. Serial encoders run each
  // dispatch to completion before the next starts, so the chain stays ordered without the
  // per-dispatch encoder boundary.
  const bool one_encoder = opt.count("--one-encoder") != 0;
  auto run = [&](int repeats) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> shared = nil;
    if (one_encoder) {
      shared = [cb computeCommandEncoder];
      for (const auto& kv : buffers) [shared useResource:kv.second usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    for (int r = 0; r < repeats; ++r) {
      for (const Dispatch& d : dispatches) {
        id<MTLComputeCommandEncoder> enc = shared;
        if (enc == nil) {
          enc = [cb computeCommandEncoder];
          for (const auto& kv : buffers) [enc useResource:kv.second usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        [enc setComputePipelineState:d.pso];
        for (const auto& b : d.bufs) [enc setBuffer:b.second.first offset:b.second.second atIndex:b.first];
        for (const auto& c : d.bytes) [enc setBytes:c.second.data() length:c.second.size() * 4 atIndex:c.first];
        [enc dispatchThreadgroups:d.grid threadsPerThreadgroup:d.block];
        if (shared == nil) [enc endEncoding];
      }
    }
    if (shared != nil) [shared endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
  };
  double first = run(1);

  auto compare = [&](const char* name, id<MTLBuffer> got, NSData* in, NSData* expected) {
    const uint8_t* g = static_cast<const uint8_t*>(got.contents);
    const uint8_t* e = static_cast<const uint8_t*>(expected.bytes);
    const uint8_t* i = static_cast<const uint8_t*>(in.bytes);
    NSUInteger n = expected.length;
    NSUInteger written_expected = 0, mismatched = 0, first_bad = 0, last_bad = 0, untouched_changed = 0;
    for (NSUInteger k = 0; k < n; ++k) {
      const bool changed = e[k] != i[k];
      if (changed) ++written_expected;
      if (g[k] != e[k]) {
        if (!changed) ++untouched_changed;
        if (mismatched == 0) first_bad = k;
        last_bad = k; ++mismatched;
      }
    }
    // max fp16 difference over 2-byte words
    double max_diff = 0; NSUInteger words_bad = 0;
    for (NSUInteger k = 0; k + 1 < n; k += 2) {
      uint16_t a = g[k] | (g[k + 1] << 8), b = e[k] | (e[k + 1] << 8);
      if (a != b) {
        ++words_bad;
        auto h2f = [](uint16_t h) { uint32_t s = (h >> 15) & 1, ex = (h >> 10) & 31, m = h & 1023; double v;
          if (ex == 0) v = std::ldexp(double(m), -24); else if (ex == 31) v = m ? NAN : INFINITY; else v = std::ldexp(double(m + 1024), int(ex) - 25);
          return s ? -v : v; };
        double dd = std::fabs(h2f(a) - h2f(b)); if (dd > max_diff) max_diff = dd;
      }
    }
    printf("%s: expected_written_bytes=%lu mismatched_bytes=%lu (range %lu..%lu) words_bad=%lu max_fp16_diff=%g outside_written=%lu\n",
           name, (unsigned long)written_expected, (unsigned long)mismatched, (unsigned long)first_bad, (unsigned long)last_bad,
           (unsigned long)words_bad, max_diff, (unsigned long)untouched_changed);
    return mismatched;
  };
  NSUInteger bad = compare("arena6", buffers["arena6"], in6, ex6) + compare("arena7", buffers["arena7"], in7, ex7) + compare("arena8", buffers["arena8"], in8, ex8);
  printf("first_run_ms=%.3f result=%s\n", first, bad == 0 ? "bit-identical" : "MISMATCH");
  if (opt.count("--dump-arenas")) {
    for (const char* name : {"arena6", "arena7", "arena8"}) {
      NSData* d = [NSData dataWithBytes:buffers[name].contents length:buffers[name].length];
      [d writeToFile:[NSString stringWithFormat:@"%s-%s.bin", opt["--dump-arenas"].c_str(), name] atomically:NO];
    }
  }
  if (opt.count("--dump-scratch") && buffers.count("scratch")) {
    NSData* d = [NSData dataWithBytes:buffers["scratch"].contents length:buffers["scratch"].length];
    [d writeToFile:@(opt["--dump-scratch"].c_str()) atomically:NO];
    printf("scratch dumped to %s (%lu bytes)\n", opt["--dump-scratch"].c_str(), (unsigned long)d.length);
  }
  if (loops > 0) {
    double best = 1e9;
    for (int r = 0; r < 3; ++r) best = std::min(best, run(loops));
    printf("loop_ms_per_call=%.4f (loops=%d)\n", best / loops, loops);
    if (opt.count("--profile")) {
      // each dispatch alone, repeated (inputs stale but timing representative)
      for (size_t di = 0; di < dispatches.size(); ++di) {
        double b = 1e9;
        for (int r = 0; r < 3; ++r) {
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          for (int k = 0; k < loops; ++k) {
            const Dispatch& d = dispatches[di];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:d.pso];
            for (const auto& kv : buffers) [enc useResource:kv.second usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            for (const auto& bb : d.bufs) [enc setBuffer:bb.second.first offset:bb.second.second atIndex:bb.first];
            for (const auto& c : d.bytes) [enc setBytes:c.second.data() length:c.second.size() * 4 atIndex:c.first];
            [enc dispatchThreadgroups:d.grid threadsPerThreadgroup:d.block];
            [enc endEncoding];
          }
          [cb commit]; [cb waitUntilCompleted];
          b = std::min(b, (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / loops);
        }
        printf("dispatch %2zu %-20s %.4f ms\n", di, dispatches[di].pso.label.UTF8String ? dispatches[di].pso.label.UTF8String : "", b);
      }
    }
  }
  return bad == 0 ? 0 : 1;
}
