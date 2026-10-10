// Guarded exact pipelines for the custom block (launches 57-83) inside the runtime tail. Numerics and kernel
// selection are those verified in src/fg_connected_custom_l56_l83/macos_connected_pipelines.mm (28/28 launch
// checkpoints byte-identical to the RTX 5090). Pipelines carry FORMAL = true (the proven bound). Scratch,
// packed weight tables and pipeline states are created once per process and reused; the per-frame counter and
// indirect-argument reset runs on the GPU inside the block's command buffer, so no CPU work per frame.
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
namespace nsfg_guarded {
struct GB { id<MTLBuffer> buf; NSUInteger off; };
struct Scratch { id<MTLBuffer> flag, rs, ra, fa, s2, p2, a2, list, cnt, ind, list2, cnt2, ind2, done, done2; std::size_t n; };
// NSFG_BLOCK_TIMING=1: every dispatch of the block gets a start/end GPU timestamp at its encoder's
// stage boundaries (one encoder per dispatch already), reported per kernel after the frame completes.
// Diagnostic only; needs NSFG_SYNC_WAITS=1 so the read-back happens after the GPU has finished.
struct Timing {
  bool enabled = false;
  id<MTLCounterSampleBuffer> buffer = nil;
  std::vector<std::string> names;
  NSUInteger used = 0;
  std::unordered_map<void*, std::string> pso_names;
  MTLTimestamp cpu0 = 0, gpu0 = 0;
  void Init(id<MTLDevice> d) {
    enabled = std::getenv("NSFG_BLOCK_TIMING") != nullptr;
    if (!enabled) return;
    id<MTLCounterSet> timestamps = nil;
    for (id<MTLCounterSet> set in d.counterSets) if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
    if (timestamps == nil || ![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { enabled = false; std::fprintf(stderr, "block timing: no stage-boundary timestamps\n"); return; }
    MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
    desc.counterSet = timestamps; desc.sampleCount = 512; desc.storageMode = MTLStorageModeShared;
    NSError* error = nil; buffer = [d newCounterSampleBufferWithDescriptor:desc error:&error];
    if (buffer == nil) { enabled = false; return; }
    [d sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
  }
  void Report(id<MTLDevice> d) {
    if (!enabled || used == 0) return;
    MTLTimestamp cpu1 = 0, gpu1 = 0; [d sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    const double ns_per_tick = (gpu1 > gpu0) ? double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1.0;
    NSData* data = [buffer resolveCounterRange:NSMakeRange(0, used)];
    if (data == nil) return;
    const auto* stamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
    std::unordered_map<std::string, double> total; double sum = 0.0;
    for (NSUInteger i = 0; i + 1 < used; i += 2) {
      const double ms = double(stamps[i + 1].timestamp - stamps[i].timestamp) * ns_per_tick * 1e-6;
      total[names[i / 2]] += ms; sum += ms;
    }
    std::vector<std::pair<std::string, double>> rows(total.begin(), total.end());
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::fprintf(stderr, "block_timing total_ms=%.3f dispatches=%lu", sum, (unsigned long)(used / 2));
    for (auto& r : rows) std::fprintf(stderr, " %s=%.3f", r.first.c_str(), r.second);
    std::fprintf(stderr, "\n");
  }
};
struct Enc {
  id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> e = nil; Timing* timing = nullptr;
  id<MTLComputeCommandEncoder> begin(id<MTLComputePipelineState> p) {
    if (timing != nullptr && timing->enabled && timing->used + 2 <= 512) {
      MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
      d.sampleBufferAttachments[0].sampleBuffer = timing->buffer;
      d.sampleBufferAttachments[0].startOfEncoderSampleIndex = timing->used;
      d.sampleBufferAttachments[0].endOfEncoderSampleIndex = timing->used + 1;
      timing->used += 2;
      auto it = timing->pso_names.find((__bridge void*)p);
      timing->names.push_back(it == timing->pso_names.end() ? "?" : it->second);
      e = [cmd computeCommandEncoderWithDescriptor:d];
    } else {
      e = [cmd computeCommandEncoder];
    }
    [e setComputePipelineState:p]; return e;
  }
  void bind(std::initializer_list<GB> bs, NSUInteger start = 0) { NSUInteger i = start; for (auto& b : bs) [e setBuffer:b.buf offset:b.off atIndex:i++]; }
  void end() { [e endEncoding]; e = nil; }
};
inline id<MTLBuffer> Plain(id<MTLDevice> d, std::size_t bytes) {
  id<MTLBuffer> b = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (b == nil) throw std::runtime_error("guarded block scratch allocation");
  return b;
}
inline Scratch MakeScratch(id<MTLDevice> d, std::size_t n) {
  return Scratch{Plain(d, n), Plain(d, n), Plain(d, n * 2), Plain(d, n * 2), Plain(d, n), Plain(d, n * 2), Plain(d, n * 2),
                 Plain(d, n * 4), Plain(d, 4), Plain(d, 12), Plain(d, n * 4), Plain(d, 4), Plain(d, 12), Plain(d, 4), Plain(d, 4), n};
}
// NSFG_BLOCK_MODE=hw: the guarded fast passes run with HW=true (no bound matrices, flags, repair or
// re-pool), the hardware lowering for the game path; anything else is the exact, receipted block.
inline bool BlockHW() { static const bool hw = std::getenv("NSFG_BLOCK_MODE") != nullptr && std::strcmp(std::getenv("NSFG_BLOCK_MODE"), "hw") == 0; return hw; }
// NSFG_FUSED_UPSAMPLE (optional candidate, off by default): the bottom launches named in the value ("68", "69",
// "82", "83", comma separated; "1" or "all" for the four) fetch each bilinear operand inside the convolution
// (custom_*_pipeline_fu.metal, the MLX-DLSS fused-output-head idea) instead of dispatching the upsample
// pre-kernel and reading the materialised tensor. Same connected formula per operand; tools/xlate/fg-custom/RESULTS.md.
inline bool BlockFusedUpsample(const char* launch) {
  const char* v = std::getenv("NSFG_FUSED_UPSAMPLE");
  if (v == nullptr || *v == 0) return false;
  if (std::strcmp(v, "1") == 0 || std::strcmp(v, "all") == 0) return true;
  return std::strstr(v, launch) != nullptr;
}
inline id<MTLComputePipelineState> PSO(id<MTLDevice> d, const std::filesystem::path& lib, const char* name, bool formal_constant) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  const std::string key = lib.string() + "#" + name + (formal_constant && BlockHW() ? "#hw" : "");
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  NSError* error = nil;
  id<MTLLibrary> library = [d newLibraryWithURL:[NSURL fileURLWithPath:@(lib.string().c_str())] error:&error];
  if (library == nil) throw std::runtime_error("guarded block metallib: " + lib.string());
  id<MTLFunction> function = nil;
  if (formal_constant) {
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    bool formal = true, hw = BlockHW();
    [values setConstantValue:&formal type:MTLDataTypeBool atIndex:0];
    [values setConstantValue:&hw type:MTLDataTypeBool atIndex:1];
    function = [library newFunctionWithName:@(name) constantValues:values error:&error];
  } else {
    function = [library newFunctionWithName:@(name)];
  }
  if (function == nil) throw std::runtime_error(std::string("guarded block function ") + name);
  id<MTLComputePipelineState> pso = [d newComputePipelineStateWithFunction:function error:&error];
  if (pso == nil) throw std::runtime_error(std::string("guarded block pipeline ") + name);
  cache.emplace(key, pso);
  return pso;
}
// ---- packed weight/proxy tables (half4 per lane per stage), one per launch, built once from the model buffers ----
inline _Float16 QHalf(_Float16 x) {
  _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x;
  if (m == (_Float16)0) return (_Float16)0;
  return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m;
}
using Table = std::vector<_Float16>;
using WI = std::function<unsigned(unsigned, unsigned, unsigned, unsigned)>;   // (tile, stage, ic, oc)
// generic (tile, stage, kb, pair, row) layout with `stages` stages of K=16 and `pairs` channel pairs
inline Table BuildTable(const _Float16* w, unsigned stages, unsigned pairs, const WI& wi) {
  Table t((std::size_t)9 * stages * 2 * pairs * 8 * 4, (_Float16)0);
  for (unsigned tile = 0; tile < 9; ++tile) for (unsigned s = 0; s < stages; ++s) for (unsigned kb = 0; kb < 2; ++kb)
    for (unsigned pair = 0; pair < pairs; ++pair) for (unsigned row = 0; row < 8; ++row) {
      const unsigned ic = s * 16 + kb * 8 + row, oc = pair * 2;
      _Float16* e = &t[(std::size_t)((((tile * stages + s) * 2 + kb) * pairs + pair) * 8 + row) * 4];
      e[0] = w[wi(tile, s, ic, oc)]; e[1] = w[wi(tile, s, ic, oc + 1)]; e[2] = QHalf(e[0]); e[3] = QHalf(e[1]);
    }
  return t;
}
inline unsigned FragIndex(unsigned base_g, unsigned tile_stride, unsigned stage_stride, unsigned tile, unsigned stage, unsigned ic, unsigned oc) {
  const unsigned g = oc / 8, n = oc % 8, k = ic % 16, lane = n * 4 + (k % 8) / 2, frag = (k >= 8 ? 2 : 0) + (k & 1);
  return g * base_g + tile * tile_stride + stage * stage_stride + (frag >= 2 ? 64 : 0) + lane * 2 + (frag & 1);
}
inline const _Float16* Weights(id<MTLBuffer> guarded) { return reinterpret_cast<const _Float16*>(static_cast<const std::uint8_t*>(guarded.contents) + kGuardBytes); }

struct Block {
  id<MTLDevice> device; std::filesystem::path libs;
  id<MTLComputePipelineState> compact, compactInd, indk;
  id<MTLComputePipelineState> preB, preP1, preP2, prePool, c0B, c0P1, c0P2, c0Pool, c1B, c1P1, c1P2, c1Pool;
  id<MTLComputePipelineState> c2B, c2P1, c2P2, b0Up, b0B, b0P1, b0P2, b0b1Up, b0b1B, b0b1P1, b0b1P2, up3;
  id<MTLComputePipelineState> b1c0B, b1c0P1, b1c0P2, b1c0Pool, b1c1B, b1c1P1, b1c1P2, b1c1Pool, b1c2B, b1c2P1, b1c2P2;
  id<MTLComputePipelineState> b1b0Up, b1b0B, b1b0P1, b1b0P2, b1b1Up, b1b1B, b1b1P1, b1b1P2;
  id<MTLComputePipelineState> b0FuB = nil, b0FuP1 = nil, b0FuP2 = nil, b0b1FuB = nil, b0b1FuP1 = nil, b0b1FuP2 = nil;
  id<MTLComputePipelineState> b1b0FuB = nil, b1b0FuP1 = nil, b1b0FuP2 = nil, b1b1FuB = nil, b1b1FuP1 = nil, b1b1FuP2 = nil;
  bool fu68 = false, fu69 = false, fu82 = false, fu83 = false;   // NSFG_FUSED_UPSAMPLE, read once in Init
  id<MTLBuffer> inter57, inter58, inter59, inter72, inter73, ups68, ups82, ups69a, ups69b, ups69c, ups83a, ups83b, ups83c, preseed;
  Scratch sPre, sC0, sC1, sB0, sB0b1, sB1c0, sB1c1, sB1b0, sB1b1; std::array<Scratch, 8> sC2, sB1c2;
  id<MTLBuffer> wqPre, wqC0, wqC1, wqB1c0, wqB1c1, wqBot0, wqB1b0, wqB0b1, wqB1b1; std::array<id<MTLBuffer>, 8> wqC2, wqB1c2;
  bool ready = false;

  void Init(id<MTLDevice> d, const std::filesystem::path& l, const std::array<id<MTLBuffer>, 25>& w) {
    if (ready) return;
    device = d; libs = l;
    timing.Init(d);
    auto P = [&](const char* lib, const char* name, bool formal = false) { auto p = PSO(d, libs / lib, name, formal); timing.pso_names[(__bridge void*)p] = name; return p; };
    compact = P("compact_ordered.metallib", "fg_compact_suspects_ordered"); compactInd = P("compact_ordered_ind.metallib", "fg_compact_suspects_ordered_ind");
    indk = P("compact_suspects.metallib", "fg_indirect_from_counter");
    preB = P("p_conv_pre.metallib", "fg_custom_block0_conv_pre_bounded_wq", true); preP1 = P("p_conv_pre.metallib", "fg_custom_block0_conv_pre_repair_pass1_wq"); preP2 = P("p_conv_pre.metallib", "fg_custom_block0_conv_pre_listed_wq"); prePool = P("p_conv_pre.metallib", "fg_custom_block0_conv_pre_repool_wq");
    c0B = P("p_conv0.metallib", "fg_custom_block0_conv0_c8_bounded_wq", true); c0P1 = P("p_conv0.metallib", "fg_custom_block0_conv0_c8_repair_pass1_wq"); c0P2 = P("p_conv0.metallib", "fg_custom_block0_conv0_c8_listed_wq"); c0Pool = P("p_conv0.metallib", "fg_custom_block0_conv0_c8_repool_wq");
    c1B = P("p_conv1.metallib", "fg_custom_block0_conv1_bounded_wq", true); c1P1 = P("p_conv1.metallib", "fg_custom_block0_conv1_repair_pass1_wq"); c1P2 = P("p_conv1.metallib", "fg_custom_block0_conv1_listed_wq"); c1Pool = P("p_conv1.metallib", "fg_custom_block0_conv1_repool_wq");
    c2B = P("p_conv2b.metallib", "fg_custom_block0_conv2_bounded_v13", true); c2P1 = P("p_conv2r8.metallib", "fg_custom_block0_conv2_repair_pass1_v8"); c2P2 = P("p_conv2r.metallib", "fg_custom_block0_conv2_repair_pass2_v6");
    b0Up = P("connected_upsamples.metallib", "fg_connected_block0_bot0_upsample"); b0B = P("p_bot0.metallib", "fg_custom_block0_bot0_bounded_wq", true); b0P1 = P("p_bot0.metallib", "fg_custom_block0_bot0_repair_pass1_wq"); b0P2 = P("p_bot0.metallib", "fg_custom_block0_bot0_listed_wq");
    b0b1Up = P("connected_upsamples.metallib", "fg_connected_block0_bot1_upsample"); b0b1B = P("p_b0bot1.metallib", "fg_custom_block0_bot1_bounded_wq", true); b0b1P1 = P("p_b0bot1.metallib", "fg_custom_block0_bot1_repair_pass1_wq"); b0b1P2 = P("p_b0bot1.metallib", "fg_custom_block0_bot1_listed_wq");
    up3 = P("upsample3.metallib", "fg_custom_upsample_planes3");
    b1c0B = P("p_b1conv0.metallib", "fg_custom_block1_conv0_bounded_wq", true); b1c0P1 = P("p_b1conv0.metallib", "fg_custom_block1_conv0_repair_pass1_wq"); b1c0P2 = P("p_b1conv0.metallib", "fg_custom_block1_conv0_listed_wq"); b1c0Pool = P("p_b1conv0.metallib", "fg_custom_block1_conv0_repool_wq");
    b1c1B = P("p_b1conv1.metallib", "fg_custom_block1_conv1_bounded_wq", true); b1c1P1 = P("p_b1conv1.metallib", "fg_custom_block1_conv1_repair_pass1_wq"); b1c1P2 = P("p_b1conv1.metallib", "fg_custom_block1_conv1_listed_wq"); b1c1Pool = P("p_b1conv1.metallib", "fg_custom_block1_conv1_repool_wq");
    b1c2B = P("p_b1conv2b.metallib", "fg_custom_block1_conv2_bounded_wq", true); b1c2P1 = P("p_b1conv2r.metallib", "fg_custom_block1_conv2_repair_pass1_ind"); b1c2P2 = P("p_b1conv2r.metallib", "fg_custom_block1_conv2_repair_pass2_ind");
    b1b0Up = P("connected_upsamples.metallib", "fg_connected_block1_bot0_upsample"); b1b0B = P("p_b1bot0.metallib", "fg_custom_block1_bot0_bounded_wq", true); b1b0P1 = P("p_b1bot0.metallib", "fg_custom_block1_bot0_repair_pass1_wq"); b1b0P2 = P("p_b1bot0.metallib", "fg_custom_block1_bot0_listed_wq");
    b1b1Up = P("connected_upsamples.metallib", "fg_connected_block1_bot1_upsample"); b1b1B = P("p_b1bot1.metallib", "fg_custom_block1_bot1_bounded_wq", true); b1b1P1 = P("p_b1bot1.metallib", "fg_custom_block1_bot1_repair_pass1_wq"); b1b1P2 = P("p_b1bot1.metallib", "fg_custom_block1_bot1_listed_wq");
    inter57 = Plain(d, 640 * 368 * 32 * 2); inter58 = Plain(d, 320 * 184 * 32 * 2); inter59 = Plain(d, 160 * 92 * 64 * 2); inter72 = Plain(d, 640 * 368 * 16 * 2); inter73 = Plain(d, 320 * 184 * 32 * 2);
    fu68 = BlockFusedUpsample("68"); fu69 = BlockFusedUpsample("69"); fu82 = BlockFusedUpsample("82"); fu83 = BlockFusedUpsample("83");
    if (fu68) { b0FuB = P("p_bot0_fu.metallib", "fg_custom_block0_bot0_bounded_fu", true); b0FuP1 = P("p_bot0_fu.metallib", "fg_custom_block0_bot0_repair_pass1_fu"); b0FuP2 = P("p_bot0_fu.metallib", "fg_custom_block0_bot0_listed_fu"); }
    else ups68 = Plain(d, 160 * 92 * 64 * 2);
    if (fu69) { b0b1FuB = P("p_b0bot1_fu.metallib", "fg_custom_block0_bot1_bounded_fu", true); b0b1FuP1 = P("p_b0bot1_fu.metallib", "fg_custom_block0_bot1_repair_pass1_fu"); b0b1FuP2 = P("p_b0bot1_fu.metallib", "fg_custom_block0_bot1_listed_fu"); }
    else { ups69a = Plain(d, 320 * 184 * 32 * 2); ups69b = Plain(d, 320 * 184 * 32 * 2); ups69c = Plain(d, 320 * 184 * 32 * 2); }
    if (fu82) { b1b0FuB = P("p_b1bot0_fu.metallib", "fg_custom_block1_bot0_bounded_fu", true); b1b0FuP1 = P("p_b1bot0_fu.metallib", "fg_custom_block1_bot0_repair_pass1_fu"); b1b0FuP2 = P("p_b1bot0_fu.metallib", "fg_custom_block1_bot0_listed_fu"); }
    else ups82 = Plain(d, 320 * 184 * 32 * 2);
    if (fu83) { b1b1FuB = P("p_b1bot1_fu.metallib", "fg_custom_block1_bot1_bounded_fu", true); b1b1FuP1 = P("p_b1bot1_fu.metallib", "fg_custom_block1_bot1_repair_pass1_fu"); b1b1FuP2 = P("p_b1bot1_fu.metallib", "fg_custom_block1_bot1_listed_fu"); }
    else { ups83a = Plain(d, 640 * 368 * 16 * 2); ups83b = Plain(d, 640 * 368 * 16 * 2); ups83c = Plain(d, 640 * 368 * 16 * 2); }
    if (fu68 || fu69 || fu82 || fu83)
      std::fprintf(stderr, "guarded block: fused bilinear fetch on launch%s%s%s%s\n", fu68 ? " 68" : "", fu69 ? " 69" : "", fu82 ? " 82" : "", fu83 ? " 83" : "");
    const std::uint32_t seed[3] = {0u, 1u, 1u}; preseed = [d newBufferWithBytes:seed length:12 options:MTLResourceStorageModeShared];
    sPre = MakeScratch(d, 640 * 368 * 32); sC0 = MakeScratch(d, 320 * 184 * 32); sC1 = MakeScratch(d, 160 * 92 * 64); sB0 = MakeScratch(d, 160 * 92 * 96); sB0b1 = MakeScratch(d, 320 * 184 * 8);
    sB1c0 = MakeScratch(d, 640 * 368 * 16); sB1c1 = MakeScratch(d, 320 * 184 * 32); sB1b0 = MakeScratch(d, 320 * 184 * 48); sB1b1 = MakeScratch(d, 640 * 368 * 8);
    for (int i = 0; i < 8; ++i) { sC2[i] = MakeScratch(d, 80 * 46 * 64); sB1c2[i] = MakeScratch(d, 160 * 92 * 32); }
    auto Upload = [&](const Table& t) { id<MTLBuffer> b = [d newBufferWithBytes:t.data() length:t.size() * 2 options:MTLResourceStorageModeShared]; if (b == nil) throw std::runtime_error("guarded block table"); return b; };
    // per-family weight index functions (the shipped kernels' wi()), table shapes as in the family harnesses
    wqPre = Upload(BuildTable(Weights(w[0]), 1, 16, [](unsigned t, unsigned, unsigned ic, unsigned oc) { return FragIndex(1152, 128, 0, t, 0, ic, oc); }));
    wqC0 = Upload(BuildTable(Weights(w[1]), 2, 16, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { return FragIndex(2304, 256, 128, t, s, ic, oc); }));
    wqC1 = Upload(BuildTable(Weights(w[2]), 2, 32, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { return FragIndex(2304, 256, 128, t, s, ic, oc); }));   // 32 in: group stride 2304, tile stride 256
    for (int i = 0; i < 8; ++i) wqC2[i] = Upload(BuildTable(Weights(w[3 + i]), 4, 32, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { return FragIndex(4608, 512, 128, t, s, ic, oc); }));
    wqBot0 = Upload(BuildTable(Weights(w[11]), 4, 48, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { return FragIndex(4608, 512, 128, t, s, ic, oc); }));
    wqB0b1 = Upload(BuildTable(Weights(w[12]), 2, 4, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { const unsigned k = ic % 16, lane = oc * 4 + (k % 8) / 2, f = (k >= 8 ? 2 : 0) + (k & 1); return t * 256 + s * 128 + (f >= 2 ? 64 : 0) + lane * 2 + (f & 1); }));
    { // block1 conv0: plain row-major (oc*9+tile)*18 + k; slots 0/1 = K=16 stage, slot 2 = the K=8 tail (k = 16 + row for row < 2)
      const _Float16* ww = Weights(w[13]); Table t((std::size_t)9 * 3 * 8 * 8 * 4, (_Float16)0);
      for (unsigned tile = 0; tile < 9; ++tile) for (unsigned slot = 0; slot < 3; ++slot) for (unsigned pair = 0; pair < 8; ++pair) for (unsigned row = 0; row < 8; ++row) {
        const int k = slot < 2 ? int(slot * 8 + row) : (row < 2 ? int(16 + row) : -1); if (k < 0) continue; const unsigned oc = pair * 2;
        _Float16* e = &t[(std::size_t)(((tile * 3 + slot) * 8 + pair) * 8 + row) * 4]; e[0] = ww[(oc * 9 + tile) * 18 + k]; e[1] = ww[((oc + 1) * 9 + tile) * 18 + k]; e[2] = QHalf(e[0]); e[3] = QHalf(e[1]); }
      wqB1c0 = Upload(t); }
    wqB1c1 = Upload(BuildTable(Weights(w[14]), 1, 16, [](unsigned t, unsigned, unsigned ic, unsigned oc) { return FragIndex(1152, 128, 0, t, 0, ic, oc); }));
    for (int i = 0; i < 8; ++i) wqB1c2[i] = Upload(BuildTable(Weights(w[15 + i]), 2, 16, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { return FragIndex(2304, 256, 128, t, s, ic, oc); }));
    wqB1b0 = Upload(BuildTable(Weights(w[23]), 2, 24, [](unsigned t, unsigned s, unsigned ic, unsigned oc) { return FragIndex(2304, 256, 128, t, s, ic, oc); }));
    wqB1b1 = Upload(BuildTable(Weights(w[24]), 1, 4, [](unsigned t, unsigned, unsigned ic, unsigned oc) { const unsigned k = ic, lane = oc * 4 + (k % 8) / 2, f = (k >= 8 ? 2 : 0) + (k & 1); return t * 128 + (f >= 2 ? 64 : 0) + lane * 2 + (f & 1); }));
    ready = true;
  }

  static constexpr std::uint32_t kGsz = 64;
  void ResetOne(id<MTLBlitCommandEncoder> blit, const Scratch& s) {
    [blit fillBuffer:s.cnt range:NSMakeRange(0, 4) value:0]; [blit fillBuffer:s.cnt2 range:NSMakeRange(0, 4) value:0];
    [blit fillBuffer:s.done range:NSMakeRange(0, 4) value:0]; [blit fillBuffer:s.done2 range:NSMakeRange(0, 4) value:0];
    [blit copyFromBuffer:preseed sourceOffset:0 toBuffer:s.ind destinationOffset:0 size:12]; [blit copyFromBuffer:preseed sourceOffset:0 toBuffer:s.ind2 destinationOffset:0 size:12];
  }
  void Repair(Enc& en, Scratch& s, id<MTLComputePipelineState> P1, id<MTLComputePipelineState> P2, std::initializer_list<GB> fixed, NSUInteger listIdx, NSUInteger p1extra, std::uint32_t pass2PerGroup) {
    const std::uint32_t total = (std::uint32_t)s.n;
    auto e = en.begin(compact); [e setBuffer:s.flag offset:0 atIndex:0]; [e setBuffer:s.list offset:0 atIndex:1]; [e setBuffer:s.cnt offset:0 atIndex:2]; [e setBytes:&total length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake(((s.n + 255) / 256) * 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setComputePipelineState:indk]; [e setBuffer:s.cnt offset:0 atIndex:0]; [e setBuffer:s.ind offset:0 atIndex:1]; [e setBytes:&kGsz length:4 atIndex:2]; [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; en.end();
    e = en.begin(P1); en.bind(fixed); en.bind({{s.list, 0}, {s.cnt, 0}, {s.rs, 0}, {s.ra, 0}}, listIdx); en.bind({{s.fa, 0}, {s.flag, 0}, {s.list2, 0}, {s.cnt2, 0}, {s.s2, 0}, {s.p2, 0}, {s.a2, 0}}, p1extra);
    [e dispatchThreadgroupsWithIndirectBuffer:s.ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz, 1, 1)]; en.end();
    e = en.begin(indk); [e setBuffer:s.cnt2 offset:0 atIndex:0]; [e setBuffer:s.ind2 offset:0 atIndex:1]; [e setBytes:&pass2PerGroup length:4 atIndex:2]; [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; en.end();
    e = en.begin(P2); en.bind(fixed); en.bind({{s.list2, 0}, {s.cnt2, 0}, {s.rs, 0}, {s.ra, 0}}, listIdx);
    [e dispatchThreadgroupsWithIndirectBuffer:s.ind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz, 1, 1)]; en.end();
  }
  void RepairInd(Enc& en, Scratch& s, id<MTLComputePipelineState> P1, id<MTLComputePipelineState> P2, std::initializer_list<GB> fixed, NSUInteger listIdx, NSUInteger p1extra, NSUInteger p1ind, std::uint32_t pass2PerGroup) {
    const std::uint32_t total = (std::uint32_t)s.n;
    auto e = en.begin(compactInd); [e setBuffer:s.flag offset:0 atIndex:0]; [e setBuffer:s.list offset:0 atIndex:1]; [e setBuffer:s.cnt offset:0 atIndex:2]; [e setBytes:&total length:4 atIndex:3]; [e setBuffer:s.done offset:0 atIndex:4]; [e setBuffer:s.ind offset:0 atIndex:5]; [e setBytes:&kGsz length:4 atIndex:6];
    [e dispatchThreads:MTLSizeMake(((s.n + 255) / 256) * 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; en.end();
    e = en.begin(P1); en.bind(fixed); en.bind({{s.list, 0}, {s.cnt, 0}, {s.rs, 0}, {s.ra, 0}}, listIdx); en.bind({{s.fa, 0}, {s.flag, 0}, {s.list2, 0}, {s.cnt2, 0}, {s.s2, 0}, {s.p2, 0}, {s.a2, 0}}, p1extra);
    [e setBuffer:s.done2 offset:0 atIndex:p1ind]; [e setBuffer:s.ind offset:0 atIndex:p1ind + 1]; [e setBuffer:s.ind2 offset:0 atIndex:p1ind + 2]; [e setBytes:&pass2PerGroup length:4 atIndex:p1ind + 3];
    [e dispatchThreadgroupsWithIndirectBuffer:s.ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz, 1, 1)]; en.end();
    e = en.begin(P2); en.bind(fixed); en.bind({{s.list2, 0}, {s.cnt2, 0}, {s.rs, 0}, {s.ra, 0}}, listIdx);
    [e dispatchThreadgroupsWithIndirectBuffer:s.ind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz, 1, 1)]; en.end();
  }
  void Pooled(Enc& en, Scratch& s, id<MTLComputePipelineState> Bk, id<MTLComputePipelineState> P1, id<MTLComputePipelineState> P2, id<MTLComputePipelineState> Pool, GB w, GB bias, GB in, id<MTLBuffer> inter, GB out, std::uint32_t IW, std::uint32_t IH, std::uint32_t CO, id<MTLBuffer> wq) {
    auto e = en.begin(Bk); en.bind({w, bias, in, {inter, 0}, {s.flag, 0}, {s.rs, 0}, {s.ra, 0}, {s.fa, 0}, {s.s2, 0}, {s.p2, 0}, {s.a2, 0}, out, {wq, 0}});
    [e dispatchThreadgroups:MTLSizeMake((IW * IH) / 8, CO / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; en.end();
    if (BlockHW()) return;   // the fast pass pooled every element itself; nothing to repair or re-pool
    Repair(en, s, P1, P2, {w, bias, in, {inter, 0}}, 4, 8, kGsz);
    e = en.begin(Pool); en.bind({{inter, 0}, out, {s.list2, 0}, {s.cnt2, 0}}); [e dispatchThreadgroupsWithIndirectBuffer:s.ind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz, 1, 1)]; en.end();
  }
  void Resid(Enc& en, Scratch& s, id<MTLComputePipelineState> Bk, id<MTLComputePipelineState> P1, id<MTLComputePipelineState> P2, GB w, GB bias, GB in, GB res, GB out, GB params, std::uint32_t W, std::uint32_t H, std::uint32_t CO, id<MTLBuffer> wq) {
    auto e = en.begin(Bk); en.bind({w, bias, in, res, out, params, {s.flag, 0}, {s.rs, 0}, {s.ra, 0}, {s.fa, 0}, {s.s2, 0}, {s.p2, 0}, {s.a2, 0}, {wq, 0}});
    [e dispatchThreadgroups:MTLSizeMake((W * H) / 8, CO / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; en.end();
    if (BlockHW()) return;
    RepairInd(en, s, P1, P2, {w, bias, in, res, out, params}, 6, 10, 17, 4);
  }
  Timing timing;
  void ReportTimings(id<MTLDevice> d) { timing.Report(d); }
  void Dispatch(Enc& en, id<MTLComputePipelineState> p, std::initializer_list<GB> bs, MTLSize threads, MTLSize group) { auto e = en.begin(p); en.bind(bs); [e dispatchThreads:threads threadsPerThreadgroup:group]; en.end(); }

  // The block: same buffer roles and order as the scalar dispatch list it replaces (launches 57-83).
  void Encode(id<MTLCommandBuffer> cmd, NSUInteger G, const std::array<id<MTLBuffer>, 25>& w, const std::array<id<MTLBuffer>, 25>& b,
              id<MTLBuffer> l56, id<MTLBuffer> l57, id<MTLBuffer> l58, const std::array<id<MTLBuffer>, 8>& b0in, const std::array<id<MTLBuffer>, 8>& b0res, const std::array<id<MTLBuffer>, 8>& b0out, id<MTLBuffer> l67,
              id<MTLBuffer> l68a, id<MTLBuffer> l68b, id<MTLBuffer> l68c, id<MTLBuffer> l69a, id<MTLBuffer> l69b, id<MTLBuffer> l69c, id<MTLBuffer> l70a, id<MTLBuffer> l70b, id<MTLBuffer> l70c,
              id<MTLBuffer> l71, id<MTLBuffer> l72, const std::array<id<MTLBuffer>, 8>& b1in, const std::array<id<MTLBuffer>, 8>& b1res, const std::array<id<MTLBuffer>, 8>& b1out, id<MTLBuffer> l81,
              id<MTLBuffer> l82a, id<MTLBuffer> l82b, id<MTLBuffer> l82c, id<MTLBuffer> l83, id<MTLBuffer> residual_on, id<MTLBuffer> residual_off, id<MTLComputePipelineState> central, id<MTLBuffer> linear_a, id<MTLBuffer> linear_b, id<MTLBuffer> linear_mask, id<MTLBuffer> central_params) {
    if (!BlockHW()) { id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      for (Scratch* s : {&sPre, &sC0, &sC1, &sB0, &sB0b1, &sB1c0, &sB1c1, &sB1b0, &sB1b1}) ResetOne(blit, *s);
      for (int i = 0; i < 8; ++i) { ResetOne(blit, sC2[i]); ResetOne(blit, sB1c2[i]); }
      [blit endEncoding]; }
    Enc en{cmd}; en.timing = &timing; timing.used = 0; timing.names.clear(); const NSUInteger Z = 0;
    Pooled(en, sPre, preB, preP1, preP2, prePool, {w[0], G}, {b[0], G}, {l56, G}, inter57, {l57, G}, 640, 368, 32, wqPre);
    Pooled(en, sC0, c0B, c0P1, c0P2, c0Pool, {w[1], G}, {b[1], G}, {l57, G}, inter58, {l58, G}, 320, 184, 32, wqC0);
    Pooled(en, sC1, c1B, c1P1, c1P2, c1Pool, {w[2], G}, {b[2], G}, {l58, G}, inter59, {b0in[0], G}, 160, 92, 64, wqC1);
    for (int i = 0; i < 8; ++i) Resid(en, sC2[i], c2B, c2P1, c2P2, {w[3 + i], G}, {b[3 + i], G}, {b0in[i], G}, {b0res[i], G}, {b0out[i], G}, {i & 1 ? residual_on : residual_off, Z}, 80, 46, 64, wqC2[i]);
    // Bottom families: the upsample pre-kernel materialises the 2x operand tensor, or (fuNN, NSFG_FUSED_UPSAMPLE)
    // the _fu kernels read the low-resolution source and compute each operand at the point of use.
    const GB in68 = fu68 ? GB{l67, G} : GB{ups68, 0};
    if (!fu68) Dispatch(en, b0Up, {{l67, G}, {ups68, 0}}, MTLSizeMake(160, 92, 1), MTLSizeMake(16, 8, 1));
    { auto e = en.begin(fu68 ? b0FuB : b0B); en.bind({{w[11], G}, {b[11], G}, in68, {l68a, G}, {l68b, G}, {l68c, G}, {sB0.flag, 0}, {sB0.rs, 0}, {sB0.ra, 0}, {sB0.fa, 0}, {sB0.s2, 0}, {sB0.p2, 0}, {sB0.a2, 0}, {wqBot0, 0}});
      [e dispatchThreadgroups:MTLSizeMake((160 * 92) / 8, 96 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; en.end();
      if (!BlockHW()) Repair(en, sB0, fu68 ? b0FuP1 : b0P1, fu68 ? b0FuP2 : b0P2, {{w[11], G}, {b[11], G}, in68, {l68a, G}, {l68b, G}, {l68c, G}}, 6, 10, kGsz); }
    const GB in69a = fu69 ? GB{l68a, G} : GB{ups69a, 0}, in69b = fu69 ? GB{l68b, G} : GB{ups69b, 0}, in69c = fu69 ? GB{l68c, G} : GB{ups69c, 0};
    if (!fu69) { Dispatch(en, b0b1Up, {{l68a, G}, {ups69a, 0}}, MTLSizeMake(320, 184, 1), MTLSizeMake(16, 8, 1)); Dispatch(en, b0b1Up, {{l68b, G}, {ups69b, 0}}, MTLSizeMake(320, 184, 1), MTLSizeMake(16, 8, 1)); Dispatch(en, b0b1Up, {{l68c, G}, {ups69c, 0}}, MTLSizeMake(320, 184, 1), MTLSizeMake(16, 8, 1)); }
    { auto e = en.begin(fu69 ? b0b1FuB : b0b1B); en.bind({{w[12], G}, {b[12], G}, in69a, in69b, in69c, {l69a, G}, {l69b, G}, {l69c, G}, {sB0b1.flag, 0}, {sB0b1.rs, 0}, {sB0b1.ra, 0}, {sB0b1.fa, 0}, {sB0b1.s2, 0}, {sB0b1.p2, 0}, {sB0b1.a2, 0}, {wqB0b1, 0}});
      [e dispatchThreadgroups:MTLSizeMake((320 * 184) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; en.end();
      if (!BlockHW()) Repair(en, sB0b1, fu69 ? b0b1FuP1 : b0b1P1, fu69 ? b0b1FuP2 : b0b1P2, {{w[12], G}, {b[12], G}, in69a, in69b, in69c, {l69a, G}, {l69b, G}, {l69c, G}}, 8, 12, 4); }
    Dispatch(en, up3, {{l69a, G}, {l69b, G}, {l69c, G}, {l70a, G}, {l70b, G}, {l70c, G}}, MTLSizeMake(640, 368, 3), MTLSizeMake(16, 8, 1));
    Dispatch(en, central, {{l71, G}, {linear_a, G}, {linear_b, G}, {linear_mask, G}, {l70a, G}, {l70b, G}, {l70c, G}, {central_params, Z}}, MTLSizeMake(640, 368, 1), MTLSizeMake(16, 16, 1));
    Pooled(en, sB1c0, b1c0B, b1c0P1, b1c0P2, b1c0Pool, {w[13], G}, {b[13], G}, {l71, G}, inter72, {l72, G}, 640, 368, 16, wqB1c0);
    Pooled(en, sB1c1, b1c1B, b1c1P1, b1c1P2, b1c1Pool, {w[14], G}, {b[14], G}, {l72, G}, inter73, {b1in[0], G}, 320, 184, 32, wqB1c1);
    for (int i = 0; i < 8; ++i) Resid(en, sB1c2[i], b1c2B, b1c2P1, b1c2P2, {w[15 + i], G}, {b[15 + i], G}, {b1in[i], G}, {b1res[i], G}, {b1out[i], G}, {i & 1 ? residual_on : residual_off, Z}, 160, 92, 32, wqB1c2[i]);
    const GB in82 = fu82 ? GB{l81, G} : GB{ups82, 0};
    if (!fu82) Dispatch(en, b1b0Up, {{l81, G}, {ups82, 0}}, MTLSizeMake(320, 184, 1), MTLSizeMake(16, 8, 1));
    { auto e = en.begin(fu82 ? b1b0FuB : b1b0B); en.bind({{w[23], G}, {b[23], G}, in82, {l82a, G}, {l82b, G}, {l82c, G}, {sB1b0.flag, 0}, {sB1b0.rs, 0}, {sB1b0.ra, 0}, {sB1b0.fa, 0}, {sB1b0.s2, 0}, {sB1b0.p2, 0}, {sB1b0.a2, 0}, {wqB1b0, 0}});
      [e dispatchThreadgroups:MTLSizeMake((320 * 184) / 8, 48 / 16, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; en.end();
      if (!BlockHW()) Repair(en, sB1b0, fu82 ? b1b0FuP1 : b1b0P1, fu82 ? b1b0FuP2 : b1b0P2, {{w[23], G}, {b[23], G}, in82, {l82a, G}, {l82b, G}, {l82c, G}}, 6, 10, kGsz); }
    const GB in83a = fu83 ? GB{l82a, G} : GB{ups83a, 0}, in83b = fu83 ? GB{l82b, G} : GB{ups83b, 0}, in83c = fu83 ? GB{l82c, G} : GB{ups83c, 0};
    if (!fu83) { Dispatch(en, b1b1Up, {{l82a, G}, {ups83a, 0}}, MTLSizeMake(640, 368, 1), MTLSizeMake(16, 8, 1)); Dispatch(en, b1b1Up, {{l82b, G}, {ups83b, 0}}, MTLSizeMake(640, 368, 1), MTLSizeMake(16, 8, 1)); Dispatch(en, b1b1Up, {{l82c, G}, {ups83c, 0}}, MTLSizeMake(640, 368, 1), MTLSizeMake(16, 8, 1)); }
    { auto e = en.begin(fu83 ? b1b1FuB : b1b1B); en.bind({{w[24], G}, {b[24], G}, in83a, in83b, in83c, {l70a, G}, {l70b, G}, {l70c, G}, {l83, G}, {sB1b1.flag, 0}, {sB1b1.rs, 0}, {sB1b1.ra, 0}, {sB1b1.fa, 0}, {sB1b1.s2, 0}, {sB1b1.p2, 0}, {sB1b1.a2, 0}, {wqB1b1, 0}});
      [e dispatchThreadgroups:MTLSizeMake((640 * 368) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; en.end();
      if (!BlockHW()) Repair(en, sB1b1, fu83 ? b1b1FuP1 : b1b1P1, fu83 ? b1b1FuP2 : b1b1P2, {{w[24], G}, {b[24], G}, in83a, in83b, in83c, {l70a, G}, {l70b, G}, {l70c, G}, {l83, G}}, 9, 13, 4); }
  }
};
inline Block& TheBlock() { static auto* block = new Block(); return *block; }
}  // namespace nsfg_guarded
