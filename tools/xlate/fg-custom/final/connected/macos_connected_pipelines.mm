// The connected custom block (launches 56-83) with the guarded fast-path pipelines in place of the scalar
// convolutions, run end to end inside ONE command buffer, timed, and verified launch by launch against the
// checkpoint hashes that matched the RTX 5090 (evidence/specs/fg-connected-custom-l56-l83-v1/verification.json).
// Bottom-branch upsamples use the connected formula (connected_upsamples.metal). FORMAL bound throughout.
#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"
#include "contract_full.h"
#include <initializer_list>
#include <regex>
#include <map>
#include <functional>
namespace {
namespace c = fg_connected_custom;
struct InitialParams { std::uint32_t height,width; float interpolation; };
struct ResidualParams { std::uint32_t residual; };
struct UpsampleParams { std::uint32_t input_height,input_width,output_height,output_width,channels; };
struct CentralParams { std::uint32_t width,height; float factor; };
struct B { id<MTLBuffer> buf; NSUInteger off; };
std::vector<std::uint8_t> ReadB(const std::filesystem::path&p){std::ifstream s(p,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error("open "+p.string());auto n=s.tellg();std::vector<std::uint8_t>b(static_cast<std::size_t>(n));s.seekg(0);s.read((char*)b.data(),n);return b;}
std::string TokenName(int token){std::ostringstream s;s<<"default-token-"<<std::setfill('0')<<std::setw(6)<<token<<".bin";return s.str();}
struct Libs {
  std::vector<id<MTLLibrary>> libs; id<MTLDevice> dev;
  bool formalValue=true;
  id<MTLComputePipelineState> get(const char* name, bool formal=false) {
    NSError* e=nil; for (id<MTLLibrary> l : libs) {
      id<MTLFunction> f;
      if (formal) { MTLFunctionConstantValues* cv=[MTLFunctionConstantValues new]; bool t=formalValue; [cv setConstantValue:&t type:MTLDataTypeBool atIndex:0]; f=[l newFunctionWithName:@(name) constantValues:cv error:&e]; }
      else f=[l newFunctionWithName:@(name)];
      if (!f) continue; id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:f error:&e]; if(!p) throw std::runtime_error(std::string("pso ")+name+": "+e.localizedDescription.UTF8String); return p; }
    throw std::runtime_error(std::string("function ")+name+" not found"); } };
// per guarded launch scratch: N virtual elements
struct Scratch { id<MTLBuffer> flag,rs,ra,fa,s2,p2,a2,list,cnt,ind,list2,cnt2,ind2,done,done2; std::size_t n; };
Scratch MakeScratch(id<MTLDevice> d, std::size_t n){ auto mk=[&](std::size_t bytes){ return [d newBufferWithLength:bytes options:MTLResourceStorageModeShared]; };
  return Scratch{mk(n),mk(n),mk(n*2),mk(n*2),mk(n),mk(n*2),mk(n*2),mk(n*4),mk(4),mk(12),mk(n*4),mk(4),mk(12),mk(4),mk(4),n}; }
void Zero(Scratch& s){ *(std::uint32_t*)s.cnt.contents=0; *(std::uint32_t*)s.cnt2.contents=0; *(std::uint32_t*)s.done.contents=0; *(std::uint32_t*)s.done2.contents=0;
  std::uint32_t* i1=(std::uint32_t*)s.ind.contents; std::uint32_t* i2=(std::uint32_t*)s.ind2.contents; i1[0]=0; i1[1]=1; i1[2]=1; i2[0]=0; i2[1]=1; i2[2]=1; }
struct Enc { id<MTLCommandBuffer> cmd; id<MTLComputeCommandEncoder> e=nil;
  id<MTLComputeCommandEncoder> begin(id<MTLComputePipelineState> p){ e=[cmd computeCommandEncoder]; [e setComputePipelineState:p]; return e; }
  void bind(std::initializer_list<B> bs, NSUInteger start=0){ NSUInteger i=start; for(auto& b: bs) [e setBuffer:b.buf offset:b.off atIndex:i++]; }
  void end(){ [e endEncoding]; e=nil; } };
const std::uint32_t kGsz=64;
id<MTLComputePipelineState> gCompactInd;   // fg_compact_suspects_ordered_ind
void RepairInd(Enc& en, Scratch& s, id<MTLComputePipelineState> pass1, id<MTLComputePipelineState> pass2, std::initializer_list<B> fixed, NSUInteger listIdx, NSUInteger p1extra, NSUInteger p1ind, std::uint32_t pass2PerGroup) {
  const std::uint32_t total=(std::uint32_t)s.n;
  auto e=en.begin(gCompactInd); [e setBuffer:s.flag offset:0 atIndex:0]; [e setBuffer:s.list offset:0 atIndex:1]; [e setBuffer:s.cnt offset:0 atIndex:2]; [e setBytes:&total length:4 atIndex:3]; [e setBuffer:s.done offset:0 atIndex:4]; [e setBuffer:s.ind offset:0 atIndex:5]; [e setBytes:&kGsz length:4 atIndex:6];
  [e dispatchThreads:MTLSizeMake(((s.n+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; en.end();
  e=en.begin(pass1); en.bind(fixed); en.bind({{s.list,0},{s.cnt,0},{s.rs,0},{s.ra,0}}, listIdx); en.bind({{s.fa,0},{s.flag,0},{s.list2,0},{s.cnt2,0},{s.s2,0},{s.p2,0},{s.a2,0}}, p1extra);
  [e setBuffer:s.done2 offset:0 atIndex:p1ind]; [e setBuffer:s.ind offset:0 atIndex:p1ind+1]; [e setBuffer:s.ind2 offset:0 atIndex:p1ind+2]; [e setBytes:&pass2PerGroup length:4 atIndex:p1ind+3];
  [e dispatchThreadgroupsWithIndirectBuffer:s.ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz,1,1)]; en.end();
  e=en.begin(pass2); en.bind(fixed); en.bind({{s.list2,0},{s.cnt2,0},{s.rs,0},{s.ra,0}}, listIdx);
  [e dispatchThreadgroupsWithIndirectBuffer:s.ind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz,1,1)]; en.end(); }
// compaction -> pass1 -> indirect -> pass2, with the family's fixed bindings supplied by the caller
void Repair(Enc& en, Scratch& s, id<MTLComputePipelineState> compact, id<MTLComputePipelineState> indk, id<MTLComputePipelineState> pass1, id<MTLComputePipelineState> pass2,
            std::initializer_list<B> fixed, NSUInteger listIdx /* list at this index, then counter, rs, ra */, NSUInteger p1extra /* pass1: fa, flag, list2, cnt2, s2, p2, a2 start */, std::uint32_t pass2PerGroup=64) {
  const std::uint32_t total=(std::uint32_t)s.n;
  auto e=en.begin(compact); [e setBuffer:s.flag offset:0 atIndex:0]; [e setBuffer:s.list offset:0 atIndex:1]; [e setBuffer:s.cnt offset:0 atIndex:2]; [e setBytes:&total length:4 atIndex:3];
  [e dispatchThreads:MTLSizeMake(((s.n+255)/256)*256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e setComputePipelineState:indk]; [e setBuffer:s.cnt offset:0 atIndex:0]; [e setBuffer:s.ind offset:0 atIndex:1]; [e setBytes:&kGsz length:4 atIndex:2]; [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; en.end();
  e=en.begin(pass1); en.bind(fixed); en.bind({{s.list,0},{s.cnt,0},{s.rs,0},{s.ra,0}}, listIdx); en.bind({{s.fa,0},{s.flag,0},{s.list2,0},{s.cnt2,0},{s.s2,0},{s.p2,0},{s.a2,0}}, p1extra);
  [e dispatchThreadgroupsWithIndirectBuffer:s.ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz,1,1)]; en.end();
  e=en.begin(indk); [e setBuffer:s.cnt2 offset:0 atIndex:0]; [e setBuffer:s.ind2 offset:0 atIndex:1]; [e setBytes:&pass2PerGroup length:4 atIndex:2]; [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; en.end();
  e=en.begin(pass2); en.bind(fixed); en.bind({{s.list,0},{s.cnt,0},{s.rs,0},{s.ra,0}}, listIdx); // pass 2 reads list2/cnt2
  [e setBuffer:s.list2 offset:0 atIndex:listIdx]; [e setBuffer:s.cnt2 offset:0 atIndex:listIdx+1];
  [e dispatchThreadgroupsWithIndirectBuffer:s.ind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz,1,1)]; en.end(); }
}

// ---- packed weight/proxy table builders lifted from the family harnesses (design thirteen) ----
// ============================================================================================================
// build_wq_conv_pre: packed weight/proxy table for fg_custom_block0_conv_pre_bounded_wq (buffer 12). Self-contained: lift
// verbatim. Input: the family's weight buffer w (at least 4608 halves, the range of wi_conv_pre). Output: a flat half
// array of 2304 half4 = 9216 halves = 18432 bytes, entry index
//   wq_index_conv_pre(tile, kb, pair, row) = ((tile*2u+kb)*16u+pair)*8u+row
// holding (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for ic = kb*8u+matrix_row, oc = 2*pair, where
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0   (== the kernel's bound_operand() on the exactly widened half).
// The function verifies the table before returning (exit(2) on failure): along the bounded kernel's own access
// pattern (group.y, g, lane, tile, kb, h) every (.x,.y) is bitwise the half the pf gather
// w[wi_conv_pre(...)] read, every (.z,.w) equals bound_operand of it (value-wise, non-negative), and every one of the
// 4608 weight slots is gathered exactly once (bijection).
static inline uint32_t wi_conv_pre(uint32_t tile, uint32_t ic, uint32_t oc) { uint32_t g=oc/8u, n=oc%8u, k=ic%16u, lane=n*4u+(k%8u)/2u; uint32_t frag=(k>=8u?2u:0u)+(k&1u); return g*1152u+tile*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u); }
static inline uint32_t wq_index_conv_pre(uint32_t tile, uint32_t kb, uint32_t pair, uint32_t row) { return ((tile*2u+kb)*16u+pair)*8u+row; }
static std::vector<_Float16> build_wq_conv_pre(const _Float16* w) {
    const uint32_t kEntries = 2304u, kWeights = 4608u;
    std::vector<_Float16> wq((size_t)kEntries*4, (_Float16)0);
    auto q_half = [](_Float16 x) -> _Float16 { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
    auto bound_operand_f = [](float v) -> float { float m = fabsf(v); return m != 0.0f ? std::max(m, 0x1p-14f) : 0.0f; };
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<16; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = kb*8u+row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index_conv_pre(tile, kb, pair, row)*4];
        e[0] = w[wi_conv_pre(tile, ic, oc)]; e[1] = w[wi_conv_pre(tile, ic, oc+1u)]; e[2] = q_half(e[0]); e[3] = q_half(e[1]); }
    std::vector<uint32_t> hits(kWeights, 0); long bad = 0;
    for (uint32_t gy=0; gy<2; ++gy) for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t mr=((lane>>1)&3u)+4u*((lane>>4)&1u), mc=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=gy*16u+mc;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t kb=0; kb<2; ++kb) {
            const _Float16* e = &wq[(size_t)wq_index_conv_pre(tile, kb, (oc0+8u*g)>>1u, mr)*4]; uint16_t eb[4]; memcpy(eb, e, 8);
            for (uint32_t h=0; h<2; ++h) {
                const uint32_t i = wi_conv_pre(tile, kb*8u+mr, oc0+8u*g+h); if (i >= kWeights) { ++bad; continue; } ++hits[i];
                uint16_t wb; memcpy(&wb, &w[i], 2); if (eb[h] != wb) ++bad;
                if ((float)e[2+h] != bound_operand_f((float)w[i]) || std::signbit((float)e[2+h])) ++bad; } } }
    if (bad) { printf("wq_conv_pre table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once=0; for (uint32_t h: hits) if (h==1u) ++once;
    if (once != (long)hits.size()) { printf("wq_conv_pre coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    return wq;
}

// ============================================================================================================
// build_wq_conv0_c8: packed weight/proxy table for fg_custom_block0_conv0_c8_bounded_wq (buffer 12). Self-contained: lift
// verbatim. Input: the family's weight buffer w (at least 9216 halves, the range of wi_conv0_c8). Output: a flat half
// array of 4608 half4 = 18432 halves = 36864 bytes, entry index
//   wq_index_conv0_c8(tile, stage, kb, pair, row) = (((tile*2u+stage)*2u+kb)*16u+pair)*8u+row
// holding (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for ic = stage*16u+kb*8u+matrix_row, oc = 2*pair, where
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0   (== the kernel's bound_operand() on the exactly widened half).
// The function verifies the table before returning (exit(2) on failure): along the bounded kernel's own access
// pattern (group.y, g, lane, tile, stage, kb, h) every (.x,.y) is bitwise the half the pf gather
// w[wi_conv0_c8(...)] read, every (.z,.w) equals bound_operand of it (value-wise, non-negative), and every one of the
// 9216 weight slots is gathered exactly once (bijection).
static inline uint32_t wi_conv0_c8(uint32_t tile, uint32_t stage, uint32_t ic, uint32_t oc) { uint32_t g=oc/8u, n=oc%8u, k=ic%16u, lane=n*4u+(k%8u)/2u; uint32_t frag=(k>=8u?2u:0u)+(k&1u); return g*2304u+tile*256u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u); }
static inline uint32_t wq_index_conv0_c8(uint32_t tile, uint32_t stage, uint32_t kb, uint32_t pair, uint32_t row) { return (((tile*2u+stage)*2u+kb)*16u+pair)*8u+row; }
static std::vector<_Float16> build_wq_conv0_c8(const _Float16* w) {
    const uint32_t kEntries = 4608u, kWeights = 9216u;
    std::vector<_Float16> wq((size_t)kEntries*4, (_Float16)0);
    auto q_half = [](_Float16 x) -> _Float16 { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
    auto bound_operand_f = [](float v) -> float { float m = fabsf(v); return m != 0.0f ? std::max(m, 0x1p-14f) : 0.0f; };
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<16; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = stage*16u+kb*8u+row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index_conv0_c8(tile, stage, kb, pair, row)*4];
        e[0] = w[wi_conv0_c8(tile, stage, ic, oc)]; e[1] = w[wi_conv0_c8(tile, stage, ic, oc+1u)]; e[2] = q_half(e[0]); e[3] = q_half(e[1]); }
    std::vector<uint32_t> hits(kWeights, 0); long bad = 0;
    for (uint32_t gy=0; gy<2; ++gy) for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t mr=((lane>>1)&3u)+4u*((lane>>4)&1u), mc=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=gy*16u+mc;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) {
            const _Float16* e = &wq[(size_t)wq_index_conv0_c8(tile, stage, kb, (oc0+8u*g)>>1u, mr)*4]; uint16_t eb[4]; memcpy(eb, e, 8);
            for (uint32_t h=0; h<2; ++h) {
                const uint32_t i = wi_conv0_c8(tile, stage, stage*16u+kb*8u+mr, oc0+8u*g+h); if (i >= kWeights) { ++bad; continue; } ++hits[i];
                uint16_t wb; memcpy(&wb, &w[i], 2); if (eb[h] != wb) ++bad;
                if ((float)e[2+h] != bound_operand_f((float)w[i]) || std::signbit((float)e[2+h])) ++bad; } } }
    if (bad) { printf("wq_conv0_c8 table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once=0; for (uint32_t h: hits) if (h==1u) ++once;
    if (once != (long)hits.size()) { printf("wq_conv0_c8 coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    return wq;
}

// ============================================================================================================
// build_wq_conv1: packed weight/proxy table for fg_custom_block0_conv1_bounded_wq (buffer 12). Self-contained: lift
// verbatim. Input: the family's weight buffer w (at least 18432 halves, the range of wi_conv1). Output: a flat half
// array of 9216 half4 = 36864 halves = 73728 bytes, entry index
//   wq_index_conv1(tile, stage, kb, pair, row) = (((tile*2u+stage)*2u+kb)*32u+pair)*8u+row
// holding (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for ic = stage*16u+kb*8u+matrix_row, oc = 2*pair, where
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0   (== the kernel's bound_operand() on the exactly widened half).
// The function verifies the table before returning (exit(2) on failure): along the bounded kernel's own access
// pattern (group.y, g, lane, tile, stage, kb, h) every (.x,.y) is bitwise the half the pf gather
// w[wi_conv1(...)] read, every (.z,.w) equals bound_operand of it (value-wise, non-negative), and every one of the
// 18432 weight slots is gathered exactly once (bijection).
static inline uint32_t wi_conv1(uint32_t tile, uint32_t stage, uint32_t ic, uint32_t oc) { uint32_t g=oc/8u, n=oc%8u, k=ic%16u, lane=n*4u+(k%8u)/2u; uint32_t frag=(k>=8u?2u:0u)+(k&1u); return g*2304u+tile*256u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u); }
static inline uint32_t wq_index_conv1(uint32_t tile, uint32_t stage, uint32_t kb, uint32_t pair, uint32_t row) { return (((tile*2u+stage)*2u+kb)*32u+pair)*8u+row; }
static std::vector<_Float16> build_wq_conv1(const _Float16* w) {
    const uint32_t kEntries = 9216u, kWeights = 18432u;
    std::vector<_Float16> wq((size_t)kEntries*4, (_Float16)0);
    auto q_half = [](_Float16 x) -> _Float16 { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
    auto bound_operand_f = [](float v) -> float { float m = fabsf(v); return m != 0.0f ? std::max(m, 0x1p-14f) : 0.0f; };
    for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) for (uint32_t pair=0; pair<32; ++pair) for (uint32_t row=0; row<8; ++row) {
        const uint32_t ic = stage*16u+kb*8u+row, oc = pair*2u; _Float16* e = &wq[(size_t)wq_index_conv1(tile, stage, kb, pair, row)*4];
        e[0] = w[wi_conv1(tile, stage, ic, oc)]; e[1] = w[wi_conv1(tile, stage, ic, oc+1u)]; e[2] = q_half(e[0]); e[3] = q_half(e[1]); }
    std::vector<uint32_t> hits(kWeights, 0); long bad = 0;
    for (uint32_t gy=0; gy<4; ++gy) for (uint32_t g=0; g<2; ++g) for (uint32_t lane=0; lane<32; ++lane) {
        const uint32_t mr=((lane>>1)&3u)+4u*((lane>>4)&1u), mc=2u*(lane&1u)+4u*((lane>>3)&1u), oc0=gy*16u+mc;
        for (uint32_t tile=0; tile<9; ++tile) for (uint32_t stage=0; stage<2; ++stage) for (uint32_t kb=0; kb<2; ++kb) {
            const _Float16* e = &wq[(size_t)wq_index_conv1(tile, stage, kb, (oc0+8u*g)>>1u, mr)*4]; uint16_t eb[4]; memcpy(eb, e, 8);
            for (uint32_t h=0; h<2; ++h) {
                const uint32_t i = wi_conv1(tile, stage, stage*16u+kb*8u+mr, oc0+8u*g+h); if (i >= kWeights) { ++bad; continue; } ++hits[i];
                uint16_t wb; memcpy(&wb, &w[i], 2); if (eb[h] != wb) ++bad;
                if ((float)e[2+h] != bound_operand_f((float)w[i]) || std::signbit((float)e[2+h])) ++bad; } } }
    if (bad) { printf("wq_conv1 table verification FAILED: %ld mismatching entries\n", bad); exit(2); }
    long once=0; for (uint32_t h: hits) if (h==1u) ++once;
    if (once != (long)hits.size()) { printf("wq_conv1 coverage FAILED: %ld of %zu weight slots hit exactly once\n", once, hits.size()); exit(2); }
    return wq;
}


// ===================================== build_wq_b1c1 (liftable verbatim) =====================================
// Packed weight/proxy table for block1 conv1's bounded kernel (fg_custom_block1_conv1_bounded_wq).
// Input: the family's weight buffer w, 4608 halves indexed by wi_b1c1(tile, ic, oc) (tile 0..8, ic 0..15, oc 0..31).
// Output: 2304 half4 = 9216 halves (18432 bytes), entry (tile, kb, pair, row) at half offset 4*wq_index_b1c1(...):
//   .x = w[wi(tile, kb*8+row, 2*pair)], .y = w[wi(tile, kb*8+row, 2*pair+1)], .z = q(.x), .w = q(.y),
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0 (half; equals bound_operand(float(x)) after the exact half->float widening).
//   wq_index_b1c1(tile, kb, pair, row) = ((tile*2 + kb)*16 + pair)*8 + row.
static inline uint32_t wi_b1c1(uint32_t tile, uint32_t ic, uint32_t oc) {
    uint32_t g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, frag = (k >= 8u ? 2u : 0u) + (k & 1u);
    return g * 1152u + tile * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u); }
static inline uint32_t wq_index_b1c1(uint32_t tile, uint32_t kb, uint32_t pair, uint32_t row) { return ((tile * 2u + kb) * 16u + pair) * 8u + row; }
static inline _Float16 wq_q_half(_Float16 x) { _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; }
static const uint32_t kWqEntries_b1c1 = 9u * 2u * 16u * 8u;   // half4 entries; the table is 4 * this many halves
static std::vector<_Float16> build_wq_b1c1(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries_b1c1 * 4, (_Float16)0);
    for (uint32_t tile = 0; tile < 9; ++tile) for (uint32_t kb = 0; kb < 2; ++kb) for (uint32_t pair = 0; pair < 16; ++pair) for (uint32_t row = 0; row < 8; ++row) {
        const uint32_t ic = kb * 8u + row, oc = pair * 2u; _Float16* e = &wq[(size_t)wq_index_b1c1(tile, kb, pair, row) * 4];
        e[0] = w[wi_b1c1(tile, ic, oc)]; e[1] = w[wi_b1c1(tile, ic, oc + 1u)]; e[2] = wq_q_half(e[0]); e[3] = wq_q_half(e[1]); }
    return wq; }
// ===================================== build_wq_bot0 (liftable verbatim) =====================================
// Packed weight/proxy table for block0 bottom0's bounded kernel (fg_custom_block0_bot0_bounded_wq).
// Input: the family's weight buffer w, 55296 halves indexed by wi_bot0(t, s, ic, oc) (t 0..8, s 0..3, ic = s*16 + 0..15, oc 0..95).
// Output: 27648 half4 = 110592 halves (221184 bytes), entry (t, s, kb, pair, row) at half offset 4*wq_index_bot0(...):
//   .x = w[wi(t, s, s*16+kb*8+row, 2*pair)], .y = w[wi(t, s, s*16+kb*8+row, 2*pair+1)], .z = q(.x), .w = q(.y),
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0 (half; equals bound_operand(float(x)) after the exact half->float widening).
//   wq_index_bot0(t, s, kb, pair, row) = (((t*4 + s)*2 + kb)*48 + pair)*8 + row.
static inline uint32_t wi_bot0(uint32_t t, uint32_t s, uint32_t ic, uint32_t oc) {
    uint32_t g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
    return g * 4608u + t * 512u + s * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u); }
static inline uint32_t wq_index_bot0(uint32_t t, uint32_t s, uint32_t kb, uint32_t pair, uint32_t row) { return (((t * 4u + s) * 2u + kb) * 48u + pair) * 8u + row; }

static const uint32_t kWqEntries_bot0 = 9u * 4u * 2u * 48u * 8u;   // half4 entries; the table is 4 * this many halves
static std::vector<_Float16> build_wq_bot0(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries_bot0 * 4, (_Float16)0);
    for (uint32_t t = 0; t < 9; ++t) for (uint32_t s = 0; s < 4; ++s) for (uint32_t kb = 0; kb < 2; ++kb) for (uint32_t pair = 0; pair < 48; ++pair) for (uint32_t row = 0; row < 8; ++row) {
        const uint32_t ic = s * 16u + kb * 8u + row, oc = pair * 2u; _Float16* e = &wq[(size_t)wq_index_bot0(t, s, kb, pair, row) * 4];
        e[0] = w[wi_bot0(t, s, ic, oc)]; e[1] = w[wi_bot0(t, s, ic, oc + 1u)]; e[2] = wq_q_half(e[0]); e[3] = wq_q_half(e[1]); }
    return wq; }
// ===================================== build_wq_b1b0 (liftable verbatim) =====================================
// Packed weight/proxy table for block1 bottom0's bounded kernel (fg_custom_block1_bot0_bounded_wq).
// Input: the family's weight buffer w, 13824 halves indexed by wi_b1b0(t, s, ic, oc) (t 0..8, s 0..1, ic = s*16 + 0..15, oc 0..47).
// Output: 6912 half4 = 27648 halves (55296 bytes), entry (t, s, kb, pair, row) at half offset 4*wq_index_b1b0(...):
//   .x = w[wi(t, s, s*16+kb*8+row, 2*pair)], .y = w[wi(t, s, s*16+kb*8+row, 2*pair+1)], .z = q(.x), .w = q(.y),
//   q(x) = x != 0 ? max(|x|, 2^-14) : 0 (half; equals bound_operand(float(x)) after the exact half->float widening).
//   wq_index_b1b0(t, s, kb, pair, row) = (((t*2 + s)*2 + kb)*24 + pair)*8 + row.
static inline uint32_t wi_b1b0(uint32_t t, uint32_t s, uint32_t ic, uint32_t oc) {
    uint32_t g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
    return g * 2304u + t * 256u + s * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u); }
static inline uint32_t wq_index_b1b0(uint32_t t, uint32_t s, uint32_t kb, uint32_t pair, uint32_t row) { return (((t * 2u + s) * 2u + kb) * 24u + pair) * 8u + row; }

static const uint32_t kWqEntries_b1b0 = 9u * 2u * 2u * 24u * 8u;   // half4 entries; the table is 4 * this many halves
static std::vector<_Float16> build_wq_b1b0(const _Float16* w) {
    std::vector<_Float16> wq((size_t)kWqEntries_b1b0 * 4, (_Float16)0);
    for (uint32_t t = 0; t < 9; ++t) for (uint32_t s = 0; s < 2; ++s) for (uint32_t kb = 0; kb < 2; ++kb) for (uint32_t pair = 0; pair < 24; ++pair) for (uint32_t row = 0; row < 8; ++row) {
        const uint32_t ic = s * 16u + kb * 8u + row, oc = pair * 2u; _Float16* e = &wq[(size_t)wq_index_b1b0(t, s, kb, pair, row) * 4];
        e[0] = w[wi_b1b0(t, s, ic, oc)]; e[1] = w[wi_b1b0(t, s, ic, oc + 1u)]; e[2] = wq_q_half(e[0]); e[3] = wq_q_half(e[1]); }
    return wq; }

static std::vector<_Float16> build_wq_b0b1(const _Float16* w) {
  // block0 bottom1: one group of 8; wi = t*256 + s*128 + (f>=2?64:0) + (oc*4+(k%8)/2)*2 + (f&1); table (t, s, kb, pair, row), 1152 half4
  auto q=[](_Float16 x){ _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
  auto wi=[](unsigned t,unsigned s,unsigned ic,unsigned oc){ unsigned k=ic%16,lane=oc*4+(k%8)/2,f=(k>=8?2:0)+(k&1); return t*256+s*128+(f>=2?64:0)+lane*2+(f&1); };
  std::vector<_Float16> tb((size_t)9*2*2*4*8*4,(_Float16)0);
  for(unsigned t=0;t<9;++t) for(unsigned s=0;s<2;++s) for(unsigned kb=0;kb<2;++kb) for(unsigned pair=0;pair<4;++pair) for(unsigned row=0;row<8;++row){
    const unsigned ic=s*16+kb*8+row, oc=pair*2; _Float16* e=&tb[(size_t)((((t*2+s)*2+kb)*4+pair)*8+row)*4];
    e[0]=w[wi(t,s,ic,oc)]; e[1]=w[wi(t,s,ic,oc+1)]; e[2]=q(e[0]); e[3]=q(e[1]); }
  return tb;
}
static std::vector<_Float16> build_wq_b1b1(const _Float16* w) {
  // block1 bottom1: one group of 8, one stage per tap; wi = t*128 + (f>=2?64:0) + (oc*4+(k%8)/2)*2 + (f&1); table (t, kb, pair, row), 576 half4
  auto q=[](_Float16 x){ _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
  auto wi=[](unsigned t,unsigned ic,unsigned oc){ unsigned k=ic,lane=oc*4+(k%8)/2,f=(k>=8?2:0)+(k&1); return t*128+(f>=2?64:0)+lane*2+(f&1); };
  std::vector<_Float16> tb((size_t)9*2*4*8*4,(_Float16)0);
  for(unsigned t=0;t<9;++t) for(unsigned kb=0;kb<2;++kb) for(unsigned pair=0;pair<4;++pair) for(unsigned row=0;row<8;++row){
    const unsigned ic=kb*8+row, oc=pair*2; _Float16* e=&tb[(size_t)(((t*2+kb)*4+pair)*8+row)*4];
    e[0]=w[wi(t,ic,oc)]; e[1]=w[wi(t,ic,oc+1)]; e[2]=q(e[0]); e[3]=q(e[1]); }
  return tb;
}
static std::vector<_Float16> build_wq_b1c2(const _Float16* w) {
  // block1 conv2: wi = g*2304 + tile*256 + stage*128 + (frag>=2?64:0) + lane*2 + (frag&1); table (tile, stage, kb, pair, row) -> half4
  auto q=[](_Float16 x){ _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
  auto wi=[](unsigned tile,unsigned stage,unsigned ic,unsigned oc){ unsigned g=oc/8,n=oc%8,k=ic%16,lane=n*4+(k%8)/2,frag=(k>=8?2:0)+(k&1); return g*2304+tile*256+stage*128+(frag>=2?64:0)+lane*2+(frag&1); };
  std::vector<_Float16> t((size_t)9*2*2*16*8*4,(_Float16)0);
  for(unsigned tile=0;tile<9;++tile) for(unsigned stage=0;stage<2;++stage) for(unsigned kb=0;kb<2;++kb) for(unsigned pair=0;pair<16;++pair) for(unsigned row=0;row<8;++row){
    const unsigned ic=stage*16+kb*8+row, oc=pair*2; _Float16* e=&t[(size_t)((((tile*2+stage)*2+kb)*16+pair)*8+row)*4];
    e[0]=w[wi(tile,stage,ic,oc)]; e[1]=w[wi(tile,stage,ic,oc+1)]; e[2]=q(e[0]); e[3]=q(e[1]); }
  return t;
}

int main(int argc,const char*v[]){@autoreleasepool{ if(argc<4){std::cerr<<"usage: host <libs dir> <model dir> <verification.json> [reps]\n";return 2;} try{
std::filesystem::path libs=v[1],model=v[2],verif=v[3]; int reps=argc>4?atoi(v[4]):5; const bool formal=argc>5?atoi(v[5])!=0:true; const std::uint8_t poison=0xa5;
id<MTLDevice>d=MTLCreateSystemDefaultDevice(); Libs L{{},d}; L.formalValue=formal; std::cout<<(formal?"FORMAL bound":"EMPIRICAL bound")<<"\n"; for(auto& ent: std::filesystem::directory_iterator(libs)) if(ent.path().extension()==".metallib"){ NSError*e=nil; id<MTLLibrary> l=[d newLibraryWithURL:[NSURL fileURLWithPath:@(ent.path().string().c_str())] error:&e]; if(!l) throw std::runtime_error(e.localizedDescription.UTF8String); L.libs.push_back(l); }
// scalar (reference) pipelines, as the original probe
auto initial=L.get("fg_initial_merge"),pre=L.get("fg_custom_block0_conv_pre"),conv0=L.get("fg_custom_block0_conv0_c8"),conv1=L.get("fg_custom_block0_conv1"),conv2=L.get("fg_custom_block0_conv2"),bot0=L.get("fg_connected_block0_bot0"),bot1=L.get("fg_connected_block0_bot1"),upsample=L.get("fg_custom_upsample_plane"),upsample3=L.get("fg_custom_upsample_planes3"),central=L.get("fg_central_block"),b1conv0=L.get("fg_custom_block1_conv0"),b1conv1=L.get("fg_custom_block1_conv1"),b1conv2=L.get("fg_custom_block1_conv2"),b1bot0=L.get("fg_connected_block1_bot0"),b1bot1=L.get("fg_connected_block1_bot1");
// guarded pipelines
auto compact=L.get("fg_compact_suspects_ordered"),indk=L.get("fg_indirect_from_counter");
auto preB=L.get("fg_custom_block0_conv_pre_bounded_wq",true),preP1=L.get("fg_custom_block0_conv_pre_repair_pass1_wq"),preP2=L.get("fg_custom_block0_conv_pre_listed_wq"),prePool=L.get("fg_custom_block0_conv_pre_repool_wq");
auto c0B=L.get("fg_custom_block0_conv0_c8_bounded_wq",true),c0P1=L.get("fg_custom_block0_conv0_c8_repair_pass1_wq"),c0P2=L.get("fg_custom_block0_conv0_c8_listed_wq"),c0Pool=L.get("fg_custom_block0_conv0_c8_repool_wq");
auto c1B=L.get("fg_custom_block0_conv1_bounded_wq",true),c1P1=L.get("fg_custom_block0_conv1_repair_pass1_wq"),c1P2=L.get("fg_custom_block0_conv1_listed_wq"),c1Pool=L.get("fg_custom_block0_conv1_repool_wq");
auto c2B=L.get("fg_custom_block0_conv2_bounded_v13",true),c2P1=L.get("fg_custom_block0_conv2_repair_pass1_v8"),c2P2=L.get("fg_custom_block0_conv2_repair_pass2_v6"); gCompactInd=L.get("fg_compact_suspects_ordered_ind");
auto b0Up=L.get("fg_connected_block0_bot0_upsample"),b0B=L.get("fg_custom_block0_bot0_bounded_wq",true),b0P1=L.get("fg_custom_block0_bot0_repair_pass1_wq"),b0P2=L.get("fg_custom_block0_bot0_listed_wq");
auto b1c0B=L.get("fg_custom_block1_conv0_bounded_wq",true),b1c0P1=L.get("fg_custom_block1_conv0_repair_pass1_wq"),b1c0P2=L.get("fg_custom_block1_conv0_listed_wq"),b1c0Pool=L.get("fg_custom_block1_conv0_repool_wq");
auto b1c1B=L.get("fg_custom_block1_conv1_bounded_wq",true),b1c1P1=L.get("fg_custom_block1_conv1_repair_pass1_wq"),b1c1P2=L.get("fg_custom_block1_conv1_listed_wq"),b1c1Pool=L.get("fg_custom_block1_conv1_repool_wq");
auto b1c2B=L.get("fg_custom_block1_conv2_bounded_wq",true),b1c2P1=L.get("fg_custom_block1_conv2_repair_pass1_ind"),b1c2P2=L.get("fg_custom_block1_conv2_repair_pass2_ind");
auto b1b0Up=L.get("fg_connected_block1_bot0_upsample"),b1b0B=L.get("fg_custom_block1_bot0_bounded_wq",true),b1b0P1=L.get("fg_custom_block1_bot0_repair_pass1_wq"),b1b0P2=L.get("fg_custom_block1_bot0_listed_wq");
auto b0b1Up=L.get("fg_connected_block0_bot1_upsample"),b0b1B=L.get("fg_custom_block0_bot1_bounded_wq",true),b0b1P1=L.get("fg_custom_block0_bot1_repair_pass1_wq"),b0b1P2=L.get("fg_custom_block0_bot1_listed_wq");
auto b1b1Up=L.get("fg_connected_block1_bot1_upsample"),b1b1B=L.get("fg_custom_block1_bot1_bounded_wq",true),b1b1P1=L.get("fg_custom_block1_bot1_repair_pass1_wq"),b1b1P2=L.get("fg_custom_block1_bot1_listed_wq");
// model
std::array<std::vector<std::uint8_t>,25>w0,b0;std::array<id<MTLBuffer>,25>wb,bb;for(std::size_t i=0;i<c::kFullModels.size();++i){auto&m=c::kFullModels[i];w0[i]=ReadB(model/TokenName(m.weight_token));b0[i]=ReadB(model/TokenName(m.bias_token));if(w0[i].size()!=m.weight_bytes||Sha256(w0[i])!=m.weight_sha||b0[i].size()!=m.bias_bytes||Sha256(b0[i])!=m.bias_sha)throw std::runtime_error("model identity");wb[i]=Guarded(d,&w0[i],m.weight_bytes,0);bb[i]=Guarded(d,&b0[i],m.bias_bytes,0);}
auto q_half=[](_Float16 x){ _Float16 m = x < (_Float16)0 ? (_Float16)(-x) : x; if (m == (_Float16)0) return (_Float16)0; return m < (_Float16)0x1p-14f ? (_Float16)0x1p-14f : m; };
auto widx_c2=[](unsigned tile,unsigned stage,unsigned ic,unsigned oc){ unsigned g=oc/8,n=oc%8,k=ic%16,lane=n*4+(k%8)/2,frag=(k>=8?2:0)+(k&1); return g*4608+tile*512+stage*128+(frag>=2?64:0)+lane*2+(frag&1); };
std::array<id<MTLBuffer>,8> wqC2; for(int i=0;i<8;++i){ const _Float16* wsrc=(const _Float16*)((const std::uint8_t*)wb[3+i].contents+kGuardBytes); std::vector<_Float16> t((size_t)9*4*2*32*8*4,(_Float16)0);
  for(unsigned tile=0;tile<9;++tile) for(unsigned stage=0;stage<4;++stage) for(unsigned kb=0;kb<2;++kb) for(unsigned pair=0;pair<32;++pair) for(unsigned row=0;row<8;++row){ const unsigned ic=stage*16+kb*8+row, oc=pair*2; _Float16* e=&t[(size_t)((((tile*4+stage)*2+kb)*32+pair)*8+row)*4]; e[0]=wsrc[widx_c2(tile,stage,ic,oc)]; e[1]=wsrc[widx_c2(tile,stage,ic,oc+1)]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
  wqC2[i]=[d newBufferWithBytes:t.data() length:t.size()*2 options:MTLResourceStorageModeShared]; }
id<MTLBuffer> wqB1c0; { const _Float16* wsrc=(const _Float16*)((const std::uint8_t*)wb[13].contents+kGuardBytes); std::vector<_Float16> t((size_t)9*3*8*8*4,(_Float16)0);
  for(unsigned tile=0;tile<9;++tile) for(unsigned slot=0;slot<3;++slot) for(unsigned pair=0;pair<8;++pair) for(unsigned row=0;row<8;++row){ _Float16* e=&t[(size_t)(((tile*3+slot)*8+pair)*8+row)*4]; const unsigned oc=pair*2;
    int k = slot<2 ? int(slot*8+row) : (row<2 ? int(16+row) : -1); if(k<0) continue; e[0]=wsrc[(oc*9+tile)*18+k]; e[1]=wsrc[((oc+1)*9+tile)*18+k]; e[2]=q_half(e[0]); e[3]=q_half(e[1]); }
  wqB1c0=[d newBufferWithBytes:t.data() length:t.size()*2 options:MTLResourceStorageModeShared]; }
auto mkwq=[&](std::vector<_Float16> t){ return [d newBufferWithBytes:t.data() length:t.size()*2 options:MTLResourceStorageModeShared]; };
auto wsrc=[&](int i){ return (const _Float16*)((const std::uint8_t*)wb[i].contents+kGuardBytes); };
id<MTLBuffer> wqPre=mkwq(build_wq_conv_pre(wsrc(0))), wqC0=mkwq(build_wq_conv0_c8(wsrc(1))), wqC1=mkwq(build_wq_conv1(wsrc(2)));
id<MTLBuffer> wqB1c1=mkwq(build_wq_b1c1(wsrc(14))), wqBot0=mkwq(build_wq_bot0(wsrc(11))), wqB1b0=mkwq(build_wq_b1b0(wsrc(23)));
id<MTLBuffer> wqB0b1=mkwq(build_wq_b0b1(wsrc(12))), wqB1b1=mkwq(build_wq_b1b1(wsrc(24))); std::array<id<MTLBuffer>,8> wqB1c2; for(int i=0;i<8;++i) wqB1c2[i]=mkwq(build_wq_b1c2(wsrc(15+i)));
auto ext0=c::Generate(); id<MTLBuffer>a=Guarded(d,&ext0.a,c::kInputABytes,0),b=Guarded(d,&ext0.b,c::kInputBBytes,0),scalar=Guarded(d,&ext0.c,c::kInputCBytes,0);
auto GB=[&](std::size_t bytes){ return Guarded(d,nullptr,bytes,poison); };
id<MTLBuffer>l56=GB(c::kL56Bytes),l57=GB(c::kL57Bytes),l58=GB(c::kL58Bytes),l59=GB(c::kLow64Bytes),l60=GB(c::kLow64Bytes),l61=GB(c::kLow64Bytes),l62=GB(c::kLow64Bytes),l63=GB(c::kLow64Bytes),l64=GB(c::kLow64Bytes),l65=GB(c::kLow64Bytes),l66=GB(c::kLow64Bytes),l67=GB(c::kLow64Bytes);
id<MTLBuffer>l68a=GB(c::kL68PlaneBytes),l68b=GB(c::kL68PlaneBytes),l68c=GB(c::kL68PlaneBytes),l69a=GB(c::kL69Out0Bytes),l69b=GB(c::kL69Out1Bytes),l69c=GB(c::kL69Out2Bytes),l70a=GB(c::kL70Out0Bytes),l70b=GB(c::kL70Out1Bytes),l70c=GB(c::kL70Out2Bytes),l71=GB(c::kL71Bytes);
id<MTLBuffer>l72=GB(c::kL72Bytes),l73=GB(c::kBlock1Bytes),l74=GB(c::kBlock1Bytes),l75=GB(c::kBlock1Bytes),l76=GB(c::kBlock1Bytes),l77=GB(c::kBlock1Bytes),l78=GB(c::kBlock1Bytes),l79=GB(c::kBlock1Bytes),l80=GB(c::kBlock1Bytes),l81=GB(c::kBlock1Bytes),l82a=GB(c::kL82PlaneBytes),l82b=GB(c::kL82PlaneBytes),l82c=GB(c::kL82PlaneBytes),l83=GB(c::kL83Bytes);
InitialParams ip{368,640,0.5f};ResidualParams off{0},on{1};UpsampleParams up0{184,320,368,640,4},up1{184,320,368,640,1},up2{184,320,368,640,3};CentralParams cp{640,368,0.5f};
auto PB=[&](const void* p,size_t n){ return [d newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]; };
id<MTLBuffer>ipb=PB(&ip,sizeof ip),offb=PB(&off,sizeof off),onb=PB(&on,sizeof on),up0b=PB(&up0,sizeof up0),up1b=PB(&up1,sizeof up1),up2b=PB(&up2,sizeof up2),cpb=PB(&cp,sizeof cp);
const NSUInteger G=kGuardBytes, Z=0;
// pipeline scratch: pre-pool intermediates, upsampled tensors, per-launch side buffers
auto plain=[&](std::size_t bytes){ return [d newBufferWithLength:bytes options:MTLResourceStorageModeShared]; };
id<MTLBuffer> inter57=plain(640*368*32*2), inter58=plain(320*184*32*2), inter59=plain(160*92*64*2), ups68=plain(160*92*64*2), inter72=plain(640*368*16*2), inter73=plain(320*184*32*2);
id<MTLBuffer> ups82=plain(320*184*32*2), ups69a=plain(320*184*32*2), ups69b=plain(320*184*32*2), ups69c=plain(320*184*32*2), ups83a=plain(640*368*16*2), ups83b=plain(640*368*16*2), ups83c=plain(640*368*16*2);
Scratch sPre=MakeScratch(d,640*368*32), sC0=MakeScratch(d,320*184*32), sC1=MakeScratch(d,160*92*64), sB0=MakeScratch(d,160*92*96), sB1c0=MakeScratch(d,640*368*16), sB1c1=MakeScratch(d,320*184*32), sB1b0=MakeScratch(d,320*184*48), sB0b1=MakeScratch(d,320*184*8), sB1b1=MakeScratch(d,640*368*8);
std::array<Scratch,8> sC2, sB1c2; for(int i=0;i<8;++i){ sC2[i]=MakeScratch(d,80*46*64); sB1c2[i]=MakeScratch(d,160*92*32); }
std::array<id<MTLBuffer>,8>ci={l59,l60,l61,l62,l63,l64,l65,l66},cr={l59,l59,l61,l61,l63,l63,l65,l65},co={l60,l61,l62,l63,l64,l65,l66,l67};
std::array<id<MTLBuffer>,8>b1ci={l73,l74,l75,l76,l77,l78,l79,l80},b1cr={l73,l73,l75,l75,l77,l77,l79,l79},b1co={l74,l75,l76,l77,l78,l79,l80,l81};
id<MTLCommandQueue> q=[d newCommandQueue];
auto Dispatch=[&](Enc& en,id<MTLComputePipelineState>p,std::initializer_list<B>bs,MTLSize threads,MTLSize group){ auto e=en.begin(p); en.bind(bs); [e dispatchThreads:threads threadsPerThreadgroup:group]; en.end(); };
auto ZeroAll=[&](){ for(Scratch* s: {&sPre,&sC0,&sC1,&sB0,&sB1c0,&sB1c1,&sB1b0,&sB0b1,&sB1b1}) Zero(*s); for(int i=0;i<8;++i){ Zero(sC2[i]); Zero(sB1c2[i]); } };
// pooled family: bounded(w,b,in,inter,+7 side) -> repair -> pool(inter,out)
auto Pooled=[&](Enc& en, Scratch& s, id<MTLComputePipelineState> Bk, id<MTLComputePipelineState> P1, id<MTLComputePipelineState> P2, id<MTLComputePipelineState> Pool, id<MTLBuffer> w, id<MTLBuffer> bias, B in, id<MTLBuffer> inter, B out, std::uint32_t IW, std::uint32_t IH, std::uint32_t CO, MTLSize outGrid, id<MTLBuffer> wq=nil, NSUInteger wqIdx=12){
  auto e=en.begin(Bk); en.bind({{w,G},{bias,G},in,{inter,0},{s.flag,0},{s.rs,0},{s.ra,0},{s.fa,0},{s.s2,0},{s.p2,0},{s.a2,0},out}); if(wq) [e setBuffer:wq offset:0 atIndex:wqIdx];
  [e dispatchThreadgroups:MTLSizeMake((IW*IH)/8,CO/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; en.end();
  Repair(en,s,compact,indk,P1,P2,{{w,G},{bias,G},in,{inter,0}},4,8);
  e=en.begin(Pool); en.bind({{inter,0},out,{s.list2,0},{s.cnt2,0}}); [e dispatchThreadgroupsWithIndirectBuffer:s.ind2 indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(kGsz,1,1)]; en.end(); (void)outGrid; };
// residual conv family (conv2 kinds): bounded(w,b,in,res,out,params,+7) -> repair
auto Resid=[&](Enc& en, Scratch& s, id<MTLComputePipelineState> Bk, id<MTLComputePipelineState> P1, id<MTLComputePipelineState> P2, id<MTLBuffer> w, id<MTLBuffer> bias, id<MTLBuffer> in, id<MTLBuffer> res, id<MTLBuffer> out, id<MTLBuffer> params, std::uint32_t W, std::uint32_t H, std::uint32_t CO, id<MTLBuffer> wq=nil, bool ind=false){
  auto e=en.begin(Bk); en.bind({{w,G},{bias,G},{in,G},{res,G},{out,G},{params,Z},{s.flag,0},{s.rs,0},{s.ra,0},{s.fa,0},{s.s2,0},{s.p2,0},{s.a2,0}}); if(wq) [e setBuffer:wq offset:0 atIndex:13];
  [e dispatchThreadgroups:MTLSizeMake((W*H)/8,CO/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; en.end();
  if(ind) RepairInd(en,s,P1,P2,{{w,G},{bias,G},{in,G},{res,G},{out,G},{params,Z}},6,10,17,4);
  else Repair(en,s,compact,indk,P1,P2,{{w,G},{bias,G},{in,G},{res,G},{out,G},{params,Z}},6,10,4); };
auto StepsScalar=[&](){ std::vector<std::function<void(id<MTLCommandBuffer>)>> S;
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,initial,{{l56,G},{a,G},{b,G},{scalar,G},{ipb,Z}},MTLSizeMake(640,368,1),MTLSizeMake(16,16,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,pre,{{wb[0],G},{bb[0],G},{l56,G},{l57,G}},MTLSizeMake(320,184,32),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,conv0,{{wb[1],G},{bb[1],G},{l57,G},{l58,G}},MTLSizeMake(160,92,32),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,conv1,{{wb[2],G},{bb[2],G},{l58,G},{l59,G}},MTLSizeMake(80,46,64),MTLSizeMake(8,4,1)); });
  for(int i=0;i<8;++i) S.push_back([&,i](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,conv2,{{wb[3+i],G},{bb[3+i],G},{ci[i],G},{cr[i],G},{co[i],G},{i&1?onb:offb,Z}},MTLSizeMake(80,46,64),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,bot0,{{wb[11],G},{bb[11],G},{l67,G},{l68a,G},{l68b,G},{l68c,G}},MTLSizeMake(160,92,96),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,bot1,{{wb[12],G},{bb[12],G},{l68a,G},{l68b,G},{l68c,G},{l69a,G},{l69b,G},{l69c,G}},MTLSizeMake(320,184,8),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,upsample,{{l69a,G},{l70a,G},{up0b,Z}},MTLSizeMake(640,368,4),MTLSizeMake(8,4,1));Dispatch(en,upsample,{{l69b,G},{l70b,G},{up1b,Z}},MTLSizeMake(640,368,1),MTLSizeMake(8,4,1));Dispatch(en,upsample,{{l69c,G},{l70c,G},{up2b,Z}},MTLSizeMake(640,368,3),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,central,{{l71,G},{a,G},{b,G},{scalar,G},{l70a,G},{l70b,G},{l70c,G},{cpb,Z}},MTLSizeMake(640,368,1),MTLSizeMake(16,16,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1conv0,{{wb[13],G},{bb[13],G},{l71,G},{l72,G}},MTLSizeMake(320,184,16),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1conv1,{{wb[14],G},{bb[14],G},{l72,G},{l73,G}},MTLSizeMake(160,92,32),MTLSizeMake(8,4,1)); });
  for(int i=0;i<8;++i) S.push_back([&,i](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1conv2,{{wb[15+i],G},{bb[15+i],G},{b1ci[i],G},{b1cr[i],G},{b1co[i],G},{i&1?onb:offb,Z}},MTLSizeMake(160,92,32),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1bot0,{{wb[23],G},{bb[23],G},{l81,G},{l82a,G},{l82b,G},{l82c,G}},MTLSizeMake(320,184,48),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1bot1,{{wb[24],G},{bb[24],G},{l82a,G},{l82b,G},{l82c,G},{l70a,G},{l70b,G},{l70c,G},{l83,G}},MTLSizeMake(640,368,8),MTLSizeMake(8,4,1)); });
  return S; };
auto EncodeScalar=[&](id<MTLCommandBuffer> cmd){ for(auto& st: StepsScalar()) st(cmd); };
auto StepsPipelines=[&](){ std::vector<std::function<void(id<MTLCommandBuffer>)>> S;
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,initial,{{l56,G},{a,G},{b,G},{scalar,G},{ipb,Z}},MTLSizeMake(640,368,1),MTLSizeMake(16,16,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Pooled(en,sPre,preB,preP1,preP2,prePool,wb[0],bb[0],{l56,G},inter57,{l57,G},640,368,32,MTLSizeMake(320,184,32),wqPre,12); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Pooled(en,sC0,c0B,c0P1,c0P2,c0Pool,wb[1],bb[1],{l57,G},inter58,{l58,G},320,184,32,MTLSizeMake(160,92,32),wqC0,12); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Pooled(en,sC1,c1B,c1P1,c1P2,c1Pool,wb[2],bb[2],{l58,G},inter59,{l59,G},160,92,64,MTLSizeMake(80,46,64),wqC1,12); });
  for(int i=0;i<8;++i) S.push_back([&,i](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Resid(en,sC2[i],c2B,c2P1,c2P2,wb[3+i],bb[3+i],ci[i],cr[i],co[i],i&1?onb:offb,80,46,64,wqC2[i],true); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b0Up,{{l67,G},{ups68,0}},MTLSizeMake(160,92,64),MTLSizeMake(8,4,1)); { auto e=en.begin(b0B); en.bind({{wb[11],G},{bb[11],G},{ups68,0},{l68a,G},{l68b,G},{l68c,G},{sB0.flag,0},{sB0.rs,0},{sB0.ra,0},{sB0.fa,0},{sB0.s2,0},{sB0.p2,0},{sB0.a2,0},{wqBot0,0}}); [e dispatchThreadgroups:MTLSizeMake((160*92)/8,96/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; en.end(); Repair(en,sB0,compact,indk,b0P1,b0P2,{{wb[11],G},{bb[11],G},{ups68,0},{l68a,G},{l68b,G},{l68c,G}},6,10); } });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b0b1Up,{{l68a,G},{ups69a,0}},MTLSizeMake(320,184,32),MTLSizeMake(8,4,1)); Dispatch(en,b0b1Up,{{l68b,G},{ups69b,0}},MTLSizeMake(320,184,32),MTLSizeMake(8,4,1)); Dispatch(en,b0b1Up,{{l68c,G},{ups69c,0}},MTLSizeMake(320,184,32),MTLSizeMake(8,4,1)); { auto e=en.begin(b0b1B); en.bind({{wb[12],G},{bb[12],G},{ups69a,0},{ups69b,0},{ups69c,0},{l69a,G},{l69b,G},{l69c,G},{sB0b1.flag,0},{sB0b1.rs,0},{sB0b1.ra,0},{sB0b1.fa,0},{sB0b1.s2,0},{sB0b1.p2,0},{sB0b1.a2,0},{wqB0b1,0}}); [e dispatchThreadgroups:MTLSizeMake((320*184)/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; en.end(); Repair(en,sB0b1,compact,indk,b0b1P1,b0b1P2,{{wb[12],G},{bb[12],G},{ups69a,0},{ups69b,0},{ups69c,0},{l69a,G},{l69b,G},{l69c,G}},8,12,4); } });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,upsample3,{{l69a,G},{l69b,G},{l69c,G},{l70a,G},{l70b,G},{l70c,G}},MTLSizeMake(640,368,8),MTLSizeMake(8,4,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,central,{{l71,G},{a,G},{b,G},{scalar,G},{l70a,G},{l70b,G},{l70c,G},{cpb,Z}},MTLSizeMake(640,368,1),MTLSizeMake(16,16,1)); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Pooled(en,sB1c0,b1c0B,b1c0P1,b1c0P2,b1c0Pool,wb[13],bb[13],{l71,G},inter72,{l72,G},640,368,16,MTLSizeMake(320,184,16),wqB1c0,12); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Pooled(en,sB1c1,b1c1B,b1c1P1,b1c1P2,b1c1Pool,wb[14],bb[14],{l72,G},inter73,{l73,G},320,184,32,MTLSizeMake(160,92,32),wqB1c1,12); });
  for(int i=0;i<8;++i) S.push_back([&,i](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Resid(en,sB1c2[i],b1c2B,b1c2P1,b1c2P2,wb[15+i],bb[15+i],b1ci[i],b1cr[i],b1co[i],i&1?onb:offb,160,92,32,wqB1c2[i],true); });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1b0Up,{{l81,G},{ups82,0}},MTLSizeMake(320,184,32),MTLSizeMake(8,4,1)); { auto e=en.begin(b1b0B); en.bind({{wb[23],G},{bb[23],G},{ups82,0},{l82a,G},{l82b,G},{l82c,G},{sB1b0.flag,0},{sB1b0.rs,0},{sB1b0.ra,0},{sB1b0.fa,0},{sB1b0.s2,0},{sB1b0.p2,0},{sB1b0.a2,0},{wqB1b0,0}}); [e dispatchThreadgroups:MTLSizeMake((320*184)/8,48/16,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; en.end(); Repair(en,sB1b0,compact,indk,b1b0P1,b1b0P2,{{wb[23],G},{bb[23],G},{ups82,0},{l82a,G},{l82b,G},{l82c,G}},6,10); } });
  S.push_back([&](id<MTLCommandBuffer> cmd){ Enc en{cmd}; Dispatch(en,b1b1Up,{{l82a,G},{ups83a,0}},MTLSizeMake(640,368,16),MTLSizeMake(8,4,1)); Dispatch(en,b1b1Up,{{l82b,G},{ups83b,0}},MTLSizeMake(640,368,16),MTLSizeMake(8,4,1)); Dispatch(en,b1b1Up,{{l82c,G},{ups83c,0}},MTLSizeMake(640,368,16),MTLSizeMake(8,4,1)); { auto e=en.begin(b1b1B); en.bind({{wb[24],G},{bb[24],G},{ups83a,0},{ups83b,0},{ups83c,0},{l70a,G},{l70b,G},{l70c,G},{l83,G},{sB1b1.flag,0},{sB1b1.rs,0},{sB1b1.ra,0},{sB1b1.fa,0},{sB1b1.s2,0},{sB1b1.p2,0},{sB1b1.a2,0},{wqB1b1,0}}); [e dispatchThreadgroups:MTLSizeMake((640*368)/8,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; en.end(); Repair(en,sB1b1,compact,indk,b1b1P1,b1b1P2,{{wb[24],G},{bb[24],G},{ups83a,0},{ups83b,0},{ups83c,0},{l70a,G},{l70b,G},{l70c,G},{l83,G}},9,13,4); } });
  return S; };
auto EncodePipelines=[&](id<MTLCommandBuffer> cmd){ for(auto& st: StepsPipelines()) st(cmd); };
auto Checkpoints=[&](){ std::array<std::string,28> ch; auto Cat=[&](std::initializer_list<std::pair<id<MTLBuffer>,std::size_t>> parts){ std::vector<std::uint8_t> r; for(auto& p: parts){ auto x=Payload(p.first,p.second); r.insert(r.end(),x.begin(),x.end()); } return Sha256(r); };
  id<MTLBuffer> low[]={l59,l60,l61,l62,l63,l64,l65,l66,l67}; ch[0]=Sha256(Payload(l56,c::kL56Bytes)); ch[1]=Sha256(Payload(l57,c::kL57Bytes)); ch[2]=Sha256(Payload(l58,c::kL58Bytes)); for(int i=0;i<9;++i) ch[3+i]=Sha256(Payload(low[i],c::kLow64Bytes));
  ch[12]=Cat({{l68a,c::kL68PlaneBytes},{l68b,c::kL68PlaneBytes},{l68c,c::kL68PlaneBytes}}); ch[13]=Cat({{l69a,c::kL69Out0Bytes},{l69b,c::kL69Out1Bytes},{l69c,c::kL69Out2Bytes}}); ch[14]=Cat({{l70a,c::kL70Out0Bytes},{l70b,c::kL70Out1Bytes},{l70c,c::kL70Out2Bytes}}); ch[15]=Sha256(Payload(l71,c::kL71Bytes)); ch[16]=Sha256(Payload(l72,c::kL72Bytes));
  id<MTLBuffer> blk[]={l73,l74,l75,l76,l77,l78,l79,l80,l81}; for(int i=0;i<9;++i) ch[17+i]=Sha256(Payload(blk[i],c::kBlock1Bytes)); ch[26]=Cat({{l82a,c::kL82PlaneBytes},{l82b,c::kL82PlaneBytes},{l82c,c::kL82PlaneBytes}}); ch[27]=Sha256(Payload(l83,c::kL83Bytes)); return ch; };
// expected hashes (RTX-matched) from verification.json
std::map<int,std::string> expected; { std::ifstream f(verif); std::stringstream ss; ss<<f.rdbuf(); std::string j=ss.str(); std::regex re("\"launch-0000(\\d\\d)\": \"([0-9a-f]{64})\""); for(std::sregex_iterator it(j.begin(),j.end(),re),end; it!=end; ++it) expected[std::stoi((*it)[1])]=(*it)[2]; }
if(expected.size()!=28) throw std::runtime_error("verification.json: expected 28 checkpoint hashes");
auto Run=[&](auto&& encode, const char* label){ double best=1e9; for(int r=0;r<reps;++r){ ZeroAll(); id<MTLCommandBuffer> cmd=[q commandBuffer]; encode(cmd); [cmd commit]; [cmd waitUntilCompleted]; if(cmd.status==MTLCommandBufferStatusError) throw std::runtime_error(cmd.error.localizedDescription.UTF8String); best=std::min(best,(cmd.GPUEndTime-cmd.GPUStartTime)*1000.0); }
  auto ch=Checkpoints(); int bad=0; for(int i=0;i<28;++i) if(ch[i]!=expected[56+i]){ ++bad; std::cout<<"  "<<label<<": launch "<<56+i<<" differs from the RTX-matched checkpoint\n"; }
  std::cout<<label<<": block GPU time "<<std::fixed<<std::setprecision(3)<<best<<" ms (min of "<<reps<<"), checkpoints matching RTX "<<28-bad<<"/28, final output "<<ch[27].substr(0,16)<<"...\n"; return bad; };
int badS=Run(EncodeScalar,"scalar   "); int badP=Run(EncodePipelines,"pipelines");
{ auto SS=StepsScalar(), SP=StepsPipelines(); if(SS.size()!=SP.size()) throw std::runtime_error("step count mismatch"); std::vector<double> ts(SS.size(),1e9), tp(SP.size(),1e9);
  for(int r=0;r<reps;++r){ ZeroAll(); for(std::size_t k=0;k<SS.size();++k){ id<MTLCommandBuffer> cmd=[q commandBuffer]; SS[k](cmd); [cmd commit]; [cmd waitUntilCompleted]; ts[k]=std::min(ts[k],(cmd.GPUEndTime-cmd.GPUStartTime)*1000.0); }
    ZeroAll(); for(std::size_t k=0;k<SP.size();++k){ id<MTLCommandBuffer> cmd=[q commandBuffer]; SP[k](cmd); [cmd commit]; [cmd waitUntilCompleted]; tp[k]=std::min(tp[k],(cmd.GPUEndTime-cmd.GPUStartTime)*1000.0); } }
  const char* names[]={"initial_merge","convPre","conv0","conv1","conv2 0","conv2 1","conv2 2","conv2 3","conv2 4","conv2 5","conv2 6","conv2 7","bottom0","bottom1","upsample x3","central","b1 conv0","b1 conv1","b1 conv2 0","b1 conv2 1","b1 conv2 2","b1 conv2 3","b1 conv2 4","b1 conv2 5","b1 conv2 6","b1 conv2 7","b1 bottom0","b1 bottom1"};
  std::cout<<"per-launch (own command buffer, min of "<<reps<<"; each carries the ~7 us command-buffer floor):\n"; double sumS=0,sumP=0;
  for(std::size_t k=0;k<SS.size();++k){ sumS+=ts[k]; sumP+=tp[k]; std::cout<<"  "<<std::left<<std::setw(14)<<(k<28?names[k]:"?")<<std::right<<" scalar "<<std::setw(7)<<ts[k]<<"  pipeline "<<std::setw(7)<<tp[k]<<"  "<<std::setw(5)<<std::setprecision(2)<<ts[k]/tp[k]<<"x"<<std::setprecision(3)<<"\n"; }
  std::cout<<"  sums: scalar "<<sumS<<"  pipeline "<<sumP<<"\n"; }
// flag statistics from the last pipeline run
auto pct=[&](Scratch& s){ std::uint32_t c1=*(std::uint32_t*)s.cnt.contents, c2=*(std::uint32_t*)s.cnt2.contents; std::ostringstream o; o<<std::setprecision(2)<<std::fixed<<100.0*c1/s.n<<"%/"<<100.0*c2/s.n<<"%"; return o.str(); };
std::cout<<"flagged/pass2: convPre "<<pct(sPre)<<" conv0 "<<pct(sC0)<<" conv1 "<<pct(sC1)<<" conv2[0] "<<pct(sC2[0])<<" bot0 "<<pct(sB0)<<" b1conv0 "<<pct(sB1c0)<<" b1conv1 "<<pct(sB1c1)<<" b1conv2[0] "<<pct(sB1c2[0])<<" b1bot0 "<<pct(sB1b0)<<" bot1 "<<pct(sB0b1)<<" b1bot1 "<<pct(sB1b1)<<"\n";
return (badS||badP)?3:0; }catch(const std::exception&e){std::cerr<<"error="<<e.what()<<'\n';return 1;}}}
