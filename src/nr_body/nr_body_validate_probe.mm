// nr_body_validate_probe.mm — synthetic tests of the resident-body GPU evidence kernels (no model data). Validator: exact once passes; one missing
// element, one duplicate, a COMPENSATING missing+duplicate pair (same total), and a padding write must each fault with the right first-failure
// identity; at-most-once accepts {0,1} and faults on 2; a skipped validator must be detectable from validators_run and the (poisoned) completion
// records; a partial validator from record.elements. Digest: the GPU digest of random data must equal the CPU mirror for two different thread
// counts (scheduling independence) and must change when one word changes. Usage: nr_body_validate_probe LIB OUT.json
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "nr_body_digest.h"
struct Args { uint32_t count, launch_id, launch_slot, mode; };
struct Sticky { uint32_t fault_flag, reserved, fault_count, validators_run; };
struct Rec { uint32_t launch_id, elements, faults, done, first_elem, first_obs, pad0, pad1; };
struct DArgs { uint32_t words, launch_id, slot, reserved; };
struct DRec { uint32_t launch_id, words, lo, hi, done, pad0, pad1, pad2; };
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc != 3) { fprintf(stderr, "usage: nr_body_validate_probe LIB OUT.json\n"); return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil; id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err]; if (!lib) { fprintf(stderr, "lib\n"); return 2; }
  id<MTLComputePipelineState> pv = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_cov_validate"] error:&err], pd = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_digest64"] error:&err], po = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_cov_first_obs"] error:&err]; if (!pv || !pd || !po) { fprintf(stderr, "pso\n"); return 2; }
  id<MTLCommandQueue> q = [dev newCommandQueue]; const uint32_t N = 1u << 20; std::vector<std::string> out; bool all_ok = true;
  // encode `nvalidators` validators over the same counters but only `nencoded` of them are actually encoded (skipped-validator test); `partial` shrinks the count of the first one
  auto run = [&](const char* name, std::vector<uint32_t> counters, std::vector<uint8_t> expect, uint32_t mode, uint32_t nvalidators, uint32_t nencoded, bool partial, bool expect_fault, uint32_t expect_first_elem, uint32_t expect_first_obs) {
    id<MTLBuffer> bc = [dev newBufferWithBytes:counters.data() length:counters.size() * 4 options:MTLResourceStorageModeShared]; std::vector<uint8_t> one(1, 1); id<MTLBuffer> be = [dev newBufferWithBytes:expect.empty() ? one.data() : expect.data() length:expect.empty() ? 1 : expect.size() options:MTLResourceStorageModeShared];
    Sticky s0{0, 0, 0, 0}; id<MTLBuffer> bs = [dev newBufferWithBytes:&s0 length:sizeof s0 options:MTLResourceStorageModeShared]; std::vector<Rec> recs(4, Rec{0xEEEEEEEEu, 0xEEEEEEEEu, 0u, 0xEEEEEEEEu, 0xffffffffu, 0xEEEEEEEEu, 0u, 0u}); id<MTLBuffer> br = [dev newBufferWithBytes:recs.data() length:recs.size() * sizeof(Rec) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    for (uint32_t v = 0; v < nencoded; ++v) { Args a{uint32_t(counters.size()) / ((partial && v == 0) ? 2u : 1u), 100u + v, v, mode}; id<MTLBuffer> ba = [dev newBufferWithBytes:&a length:sizeof a options:MTLResourceStorageModeShared];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; [e setComputePipelineState:pv]; [e setBuffer:bc offset:0 atIndex:0]; [e setBuffer:be offset:0 atIndex:1]; [e setBuffer:bs offset:0 atIndex:2]; [e setBuffer:br offset:0 atIndex:3]; [e setBuffer:ba offset:0 atIndex:4];
      [e dispatchThreads:MTLSizeMake(4096, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding];
      id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder]; [e2 setComputePipelineState:po]; [e2 setBuffer:bc offset:0 atIndex:0]; [e2 setBuffer:br offset:0 atIndex:1]; [e2 setBuffer:ba offset:0 atIndex:2]; [e2 dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e2 endEncoding]; }
    [cb commit]; [cb waitUntilCompleted];
    const Sticky* s = static_cast<const Sticky*>(bs.contents); const Rec* r = static_cast<const Rec*>(br.contents);
    const bool faulted = s->fault_flag != 0; uint32_t fe = 0xffffffffu, fo = 0xffffffffu; for (uint32_t v = 0; v < nvalidators; ++v) if (r[v].done == 0xC0DE0001u && r[v].faults) { fe = r[v].first_elem; fo = r[v].first_obs; break; }   // first faulting validator's first element (full 32-bit index) and its observed count
    // the CPU completion contract: run count == expected AND every expected record complete with the expected identity/size AND zero faults
    bool records_ok = s->validators_run == nvalidators; uint32_t rec_faults = 0;
    for (uint32_t v = 0; v < nvalidators; ++v) { records_ok = records_ok && r[v].done == 0xC0DE0001u && r[v].launch_id == 100u + v && r[v].elements == counters.size(); rec_faults += (r[v].done == 0xC0DE0001u) ? r[v].faults : 0u; }
    const bool contract_pass = records_ok && !faulted; const bool expect_pass = !expect_fault && nencoded == nvalidators && !partial;
    bool ok = contract_pass == expect_pass; if (expect_fault) ok = ok && faulted && fe == expect_first_elem && fo == expect_first_obs && s->fault_count > 0 && rec_faults == s->fault_count;
    char buf[640]; snprintf(buf, sizeof buf, "{\"case\": \"%s\", \"faulted\": %s, \"fault_count\": %u, \"record_faults\": %u, \"first_element\": %u, \"first_observed\": %u, \"validators_run\": %u, \"validators_expected\": %u, \"records_ok\": %s, \"contract_pass\": %s, \"ok\": %s}", name, faulted ? "true" : "false", s->fault_count, rec_faults, fe, fo, s->validators_run, nvalidators, records_ok ? "true" : "false", contract_pass ? "true" : "false", ok ? "true" : "false");
    printf("%s\n", buf); out.push_back(buf); all_ok = all_ok && ok; };
  std::vector<uint32_t> once(N, 1); run("exact-once", once, {}, 0, 1, 1, false, false, 0, 0);
  auto m = once; m[777] = 0; run("one-missing", m, {}, 0, 1, 1, false, true, 777, 0);
  auto d = once; d[4242] = 2; run("one-duplicate", d, {}, 0, 1, 1, false, true, 4242, 2);
  auto comp = once; comp[10] = 0; comp[11] = 2; run("compensating-missing-plus-duplicate", comp, {}, 0, 1, 1, false, true, 10, 0);
  std::vector<uint8_t> ex(N, 1); for (uint32_t i = 5000; i < 6000; ++i) ex[i] = 0; auto pad = once; for (uint32_t i = 5000; i < 6000; ++i) pad[i] = 0; run("padding-respected", pad, ex, 1, 1, 1, false, false, 0, 0);
  auto padw = pad; padw[5500] = 1; run("padding-written", padw, ex, 1, 1, 1, false, true, 5500, 1);
  run("at-most-once-accepts-zero-and-one", pad, {}, 2, 1, 1, false, false, 0, 0);
  auto dup2 = pad; dup2[9] = 2; run("at-most-once-faults-on-two", dup2, {}, 2, 1, 1, false, true, 9, 2);
  run("three-validators-all-run", once, {}, 0, 3, 3, false, false, 0, 0);
  run("two-of-three-validators-skipped", once, {}, 0, 3, 2, false, false, 0, 0);   // must NOT pass the contract (records_ok false: poisoned third record, run count 2)
  run("partial-validator-half-count", once, {}, 0, 1, 1, true, false, 0, 0);       // must NOT pass the contract (record.elements != expected)
  // full-range first-failure identity (Codex finding 3): indices above 2^24 and the final pool word (53,129,984 words = the 212,519,936-byte pool)
  { const uint32_t P = 53129984u; std::vector<uint32_t> big(P, 1); big[7] = 2; big[16777223u] = 2; run("first-failure-7-and-16777223-count2", big, {}, 0, 1, 1, false, true, 7, 2);
    std::vector<uint32_t> hi(P, 1); hi[16777223u] = 2; run("first-failure-only-16777223", hi, {}, 0, 1, 1, false, true, 16777223u, 2);
    std::vector<uint32_t> last(P, 1); last[P - 1] = 0; run("first-failure-final-pool-word-missing", last, {}, 0, 1, 1, false, true, P - 1, 0);
    std::vector<uint32_t> lastdup(P, 0); lastdup[P - 1] = 2; run("at-most-once-final-pool-word-duplicate", lastdup, {}, 2, 1, 1, false, true, P - 1, 2); }
  // digest: GPU vs CPU mirror, two thread counts, and sensitivity to one word
  { std::mt19937 rng(20260906u); std::vector<uint32_t> data(N); for (auto& w : data) w = rng(); id<MTLBuffer> bd = [dev newBufferWithBytes:data.data() length:N * 4 options:MTLResourceStorageModeShared];
    auto gpu = [&](uint32_t threads) { std::vector<DRec> rec(1, DRec{0xEEEEEEEEu, 0xEEEEEEEEu, 0, 0, 0xEEEEEEEEu, 0, 0, 0}); id<MTLBuffer> br = [dev newBufferWithBytes:rec.data() length:sizeof(DRec) options:MTLResourceStorageModeShared]; DArgs a{N, 7u, 0u, 0u}; id<MTLBuffer> ba = [dev newBufferWithBytes:&a length:sizeof a options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; [e setComputePipelineState:pd]; [e setBuffer:bd offset:0 atIndex:0]; [e setBuffer:br offset:0 atIndex:1]; [e setBuffer:ba offset:0 atIndex:2]; [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; return *static_cast<const DRec*>(br.contents); };
    const NrDigest64 cpu = nr_digest64_cpu(reinterpret_cast<const uint8_t*>(data.data()), N * 4); const DRec g1 = gpu(65536), g2 = gpu(1024);
    const bool ok1 = g1.done == 0xD16E5701u && g1.launch_id == 7u && g1.words == N && g1.lo == cpu.lo && g1.hi == cpu.hi, ok2 = g2.lo == cpu.lo && g2.hi == cpu.hi && g2.done == 0xD16E5701u;
    data[123456] ^= 1u; std::memcpy(bd.contents, data.data(), N * 4); const DRec g3 = gpu(65536); const NrDigest64 cpu3 = nr_digest64_cpu(reinterpret_cast<const uint8_t*>(data.data()), N * 4); const bool ok3 = (g3.lo != g1.lo || g3.hi != g1.hi) && g3.lo == cpu3.lo && g3.hi == cpu3.hi;
    char b[512]; snprintf(b, sizeof b, "{\"case\": \"digest64-gpu-vs-cpu\", \"cpu\": \"%08x%08x\", \"gpu_65536_threads\": \"%08x%08x\", \"gpu_1024_threads\": \"%08x%08x\", \"one_word_changed_gpu\": \"%08x%08x\", \"matches\": %s, \"scheduling_independent\": %s, \"sensitive\": %s, \"ok\": %s}", cpu.hi, cpu.lo, g1.hi, g1.lo, g2.hi, g2.lo, g3.hi, g3.lo, ok1 ? "true" : "false", ok2 ? "true" : "false", ok3 ? "true" : "false", (ok1 && ok2 && ok3) ? "true" : "false");
    printf("%s\n", b); out.push_back(b); all_ok = all_ok && ok1 && ok2 && ok3; }
  std::ofstream f(argv[2]); f << "{\"schema\": \"neuralstudio.nr.cov-validator-probe.v3\", \"cases\": ["; for (size_t i = 0; i < out.size(); ++i) f << (i ? ",\n" : "\n") << out[i]; f << "\n], \"all_ok\": " << (all_ok ? "true" : "false") << "}\n";
  printf("all_ok %s\n", all_ok ? "true" : "false"); return all_ok ? 0 : 1;
} }
