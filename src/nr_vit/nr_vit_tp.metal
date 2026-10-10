// Three-product exact QMMA GEMM engine for the ViT-1D stages (expand 1024->4096, contract 4096->1024, projection 1024->1024): the same
// structure as the split-conv prototype (src/nr_split/nr_split_tp.metal). One SIMD group per 8 tokens x 64 columns; per rounded K-step of 32:
// stage the 8 x 32 A bytes as l/h/(l+h) halves + exponent-class masks, 12 simdgroup half->float MMAs per 8x8 sub-tile against prepared weight
// halves, int64 reconstruction, max_exp by mask intersection (operand-exponent-sum convention), +-33G certificate, compacted per-element repair
// with the accepted rule (nr_qmma_step). Every K-step rounds to half exactly as the scalar kernels do.
#include "nr_vit_common.metal"
#include <metal_simdgroup_matrix>
struct TpPrepParams { uint steps; uint cols; uint step_bytes; uint reserved; };
inline int tp_int9(uchar b) { const int e = int((b >> 3u) & 15u), m = int(b & 7u); const int sig = (e == 0) ? m : 8 + m; const int q = (e == 0) ? -9 : e - 10; const int v = sig << uint(q + 9); return (b & 0x80u) ? -v : v; }
inline int tp_class(uchar b) { const int e = int((b >> 3u) & 15u), m = int(b & 7u); if (e == 0) { if (m == 0) return -1; return (31 - int(clz(uint(m)))); } return e + 2; }   // floor(log2|x|) + 9 in [0, 17], -1 for zero
kernel void nr_vit_tp_prep(device const uchar* w [[buffer(0)]], device half* wl [[buffer(1)]], device half* wh [[buffer(2)]], device half* wlh [[buffer(3)]], device uchar* maxcls [[buffer(4)]], constant TpPrepParams& pp [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  const uint s = gid / pp.cols, n = gid % pp.cols; if (s >= pp.steps) return; int mx = -1;
  for (uint k = 0u; k < 32u; ++k) {
    const uchar b = w[s * pp.step_bytes + (n / 64u) * 2048u + nr_weight_index(k, n % 64u)]; const int m = tp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
    const uint o = (s * 32u + k) * pp.cols + n; wl[o] = half(l); wh[o] = half(h); wlh[o] = half(l + h); mx = max(mx, tp_class(b));
  }
  maxcls[s * pp.cols + n] = uchar(mx + 1);          // 0 = column has no nonzero weight in this step; else floor(log2|w|) + 10
}
constant uint TP_ROWS = 16u;                       // tokens per SIMD group; each lane owns exactly 32 of the 16 x 64 outputs
struct TpShared { half al[TP_ROWS * 32u], ah[TP_ROWS * 32u], alh[TP_ROWS * 32u]; uchar ab[TP_ROWS * 32u]; uint maxA[TP_ROWS]; };   // shared operands only
inline uint tp_row_of(uint lane) { return (lane % 8u) / 2u + 4u * (lane / 16u); }              // measured M3 Max simdgroup_float8x8 element mapping
inline uint tp_col_of(uint lane, uint e) { return 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u) + e; }
template <uint SUBS> inline uint tp_elem(uint rb, uint sub, uint e) { return (rb * SUBS + sub) * 2u + e; }   // lane-local element index 0..4*SUBS-1
// One-pack certificate (validated on CPU: scripts/nr-qmma-cheapcert-check.py). S in units 2^-24, G the rule's grid in the same units (0 = every
// term aligned). Rounding to half is monotone, so RNE_half is constant on [S-33G, S+33G] iff no half midpoint lies inside: in-binade midpoints
// sit at r = ulp/2 (r = |S| mod ulp), the lower binade's last midpoint sits ulp/4 below the binade edge; the upper binade's first midpoint is
// already excluded by the in-binade test. Subnormal half region (|S| < 2^10 units): certified only when G = 0.
inline bool nr_tp_certified(long S, long G) {
  if (G == 0l) return true;
  const ulong mag = ulong(S < 0l ? -S : S); if (mag == 0ul) return false;
  const uint hi = uint(mag >> 32u); const uint bl = hi != 0u ? 64u - clz(hi) : 32u - clz(uint(mag));     // bit length of |S|
  if (bl <= 10u) return false;
  const long ulp = 1l << (bl - 11u), half_ulp = ulp >> 1u; const long r = long(mag & ulong(ulp - 1l)); const long d = r - half_ulp;
  const long edge_dist = long(mag) - (1l << (bl - 1u)); const long g33 = 33l * G;
  return (g33 < (d < 0l ? -d : d)) && (g33 < edge_dist + (ulp >> 2u));
}
inline long tp_c_units(ushort cb, thread int& cexp) {
  const int ce = int((cb >> 10u) & 31u), cf = int(cb & 1023u); int sig, q; if (ce != 0) { sig = 1024 + cf; q = ce - 25; } else { sig = cf; q = -24; }
  if (sig == 0) { cexp = -1000; return 0l; } cexp = q + (31 - int(clz(uint(sig)))); long v = long(sig) << uint(q + 24); return (cb & 0x8000u) ? -v : v;
}
// A-byte accessors (Metal has no lambdas): the e4m3 byte of token row r, k-slot k of step s
struct TpAIdentity { device const uchar* p; uint tb; uint s; uchar operator()(uint r, uint k) const { return p[nr_a_index(tb + r, 32u * s + k, 1024u)]; } };
template <uint ACH> struct TpADLayout { device const uchar* p; uint tb; uint s; uchar operator()(uint r, uint k) const { return p[nr_d_index(tb + r, 32u * s + nr_slot_channel(k), ACH)]; } };
struct TpWBlob { device const uchar* w; uint step_bytes; uchar operator()(uint s, uint n, uint j) const { return w[s * step_bytes + (n / 64u) * 2048u + nr_weight_index(j, n % 64u)]; } };   // e4m3 weight byte (step s, column n, k-slot j)
// One rounded K-step for the TP_ROWS x 64 tile with lane-local ownership: acc[32] are this lane's accumulators (half), repaired in place.
template <uint SUBS = 8u, typename AByte, typename BByte>
inline void tp_step(uint s, thread const AByte& abyte, thread const BByte& bbyte, uint cols, device const half* wl, device const half* wh, device const half* wlh, device const uchar* maxcls,
                    thread half* acc, threadgroup TpShared& sh, uint nb, uint lane, thread uint& repaired, bool stage_a = true) {
  // stage A: lane -> row lane/2, k half (lane%2)*16; row max class by a pair shuffle (no atomics). stage_a = false reuses the operand staged in sh.
  if (stage_a) { const uint r = lane / 2u, k0 = (lane % 2u) * 16u; uint mx = 0u;
    for (uint j = 0u; j < 16u; ++j) { const uint k = k0 + j, i = r * 32u + k; const uchar b = abyte(r, k); const int m = tp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
      sh.al[i] = half(l); sh.ah[i] = half(h); sh.alh[i] = half(l + h); sh.ab[i] = b; mx = max(mx, uint(tp_class(b) + 1)); }
    mx = max(mx, simd_shuffle_xor(mx, 1u)); if ((lane % 2u) == 0u) sh.maxA[r] = mx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong rmask = 0ul;                                   // one bit per lane-local element (up to 64 for SUBS <= 16)
  for (uint sub = 0u; sub < SUBS; ++sub) {
    const uint n0 = nb + sub * 8u; simdgroup_half8x8 BL[4], BH[4], BLH[4];
    for (uint kk = 0u; kk < 4u; ++kk) { const uint boff = (s * 32u + kk * 8u) * cols + n0; simdgroup_load(BL[kk], wl + boff, cols); simdgroup_load(BH[kk], wh + boff, cols); simdgroup_load(BLH[kk], wlh + boff, cols); }
    for (uint rb = 0u; rb < TP_ROWS / 8u; ++rb) {
      simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), P1 = P0, P2 = P0; simdgroup_half8x8 A;
      for (uint kk = 0u; kk < 4u; ++kk) { const uint ao = rb * 256u + kk * 8u;
        simdgroup_load(A, sh.al + ao, 32u); simdgroup_multiply_accumulate(P0, A, BL[kk], P0);
        simdgroup_load(A, sh.ah + ao, 32u); simdgroup_multiply_accumulate(P1, A, BH[kk], P1);
        simdgroup_load(A, sh.alh + ao, 32u); simdgroup_multiply_accumulate(P2, A, BLH[kk], P2); }
      for (uint e = 0u; e < 2u; ++e) {
        const uint r = rb * 8u + tp_row_of(lane), c = tp_col_of(lane, e), n = n0 + c, ei = tp_elem<SUBS>(rb, sub, e);
        // narrow exact arithmetic: partials < 2^23 and the cross term < 2^25 fit int32; widen only for the shifts and the prior
        const int p0 = int(P0.thread_elements()[e]), p1 = int(P1.thread_elements()[e]), p2 = int(P2.thread_elements()[e]); const int cross = p2 - p0 - p1;
        const long recon = long(p0) + (long(cross) << 9u) + (long(p1) << 18u);                  // sum(a*b) * 2^18
        const ushort cb = as_type<ushort>(acc[ei]); int cexp; const long cval = tp_c_units(cb, cexp); const long S = (recon << 6u) + cval;
        const uint ma1 = sh.maxA[r], mb1 = uint(maxcls[s * cols + n]); int maxexp = (ma1 != 0u && mb1 != 0u) ? int(ma1 + mb1) - 20 : -1000; maxexp = max(maxexp, cexp);
        if (maxexp == -1000) acc[ei] = as_type<half>(ushort(0u));
        else { const int gexp = maxexp - 25; const long G = (gexp + 24 >= 0) ? (1l << uint(gexp + 24)) : 0l;
          if (nr_tp_certified(S, G)) acc[ei] = nr_hw_fixed_to_half_rne(S, -24); else rmask |= 1ul << ei; }
      }
    }
  }
  // lane-local repair: this lane's uncertain elements, by the accepted rule, from the staged A bytes and the e4m3 weights
  repaired += popcount(rmask);
  while (rmask != 0ul) { const uint ei = uint(ctz(rmask)); rmask &= rmask - 1ul; const uint rb = ei / (2u * SUBS), sub = (ei % (2u * SUBS)) / 2u, e = ei % 2u;
    const uint r = rb * 8u + tp_row_of(lane), n = nb + sub * 8u + tp_col_of(lane, e); half a[32], b[32];
    for (uint j = 0u; j < 32u; ++j) { a[j] = nr_e4m3(sh.ab[r * 32u + j]); b[j] = nr_e4m3(bbyte(s, n, j)); }
    acc[ei] = nr_qmma_step(a, b, acc[ei]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // the next step restages sh.al/ah/alh/ab
}
// ---- expand: tokens (A identity, 1024) -> 4096 columns, 32 steps, SiLU, D layout ----
kernel void nr_vit_expand_tp(device const uchar* tokens [[buffer(0)]], device const uchar* weights [[buffer(1)]], device uchar* out [[buffer(2)]], device atomic_uint* coverage [[buffer(3)]], constant NrParams& p [[buffer(4)]],
                             device const half* wl [[buffer(5)]], device const half* wh [[buffer(6)]], device const half* wlh [[buffer(7)]], device const uchar* maxcls [[buffer(8)]], device atomic_uint* stats [[buffer(9)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup TpShared sh; const uint tb = tg.y * TP_ROWS, nb = tg.x * 64u; if (tb >= p.tokens) return;
  half acc[32]; for (uint i = 0u; i < 32u; ++i) acc[i] = half(0.0h); uint repaired = 0u;
  { TpWBlob bb{weights, 131072u}; for (uint s = 0u; s < 32u; ++s) { TpAIdentity ab{tokens, tb, s}; tp_step(s, ab, bb, 4096u, wl, wh, wlh, maxcls, acc, sh, nb, lane, repaired); } }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint r = tb + rb * 8u + tp_row_of(lane), c = nb + sub * 8u + tp_col_of(lane, e); const half h = acc[tp_elem<8u>(rb, sub, e)];
    const half tt = min(max(h, nr_hbits(0xc400u)), nr_hbits(0x4400u)); const half u = fma(nr_hbits(0xab28u), abs(tt), nr_hbits(0x3728u)); const half sv = fma(tt, u, nr_hbits(0x3b28u));
    const uint idx = nr_d_index(r, c, 4096u); out[idx] = nr_to_e4m3(h * sv); atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed); }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}
// ---- contract / projection: A in the D layout (channels 4096 / 1024), 1024 columns, K quarters chained in half with the residual folded into quarter 0 ----
template <uint ACH, uint QSTEPS, uint STEP_BYTES, uint SCALE_OFF>
inline void tp_residual_gemm(device const uchar* A, device const uchar* residual, device const uchar* weights, device uchar* out, device atomic_uint* coverage, uint tokens,
                             device const half* wl, device const half* wh, device const half* wlh, device const uchar* maxcls, device atomic_uint* stats, uint2 tg, uint lane, threadgroup TpShared& sh) {
  const uint tb = tg.y * TP_ROWS, nb = tg.x * 64u; if (tb >= tokens) return; device const half* scale = reinterpret_cast<device const half*>(weights + SCALE_OFF); uint repaired = 0u;
  half acc[32], total[32];
  for (uint z = 0u; z < 4u; ++z) {
    for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
      const uint r = tb + rb * 8u + tp_row_of(lane), n = nb + sub * 8u + tp_col_of(lane, e);
      acc[tp_elem<8u>(rb, sub, e)] = (z == 0u) ? half(nr_e4m3(residual[nr_a_index(r, nr_cfd(n), 1024u)]) * scale[n]) : half(0.0h); }
    { TpWBlob bb{weights, STEP_BYTES}; for (uint s = QSTEPS * z; s < QSTEPS * z + QSTEPS; ++s) { TpADLayout<ACH> ab{A, tb, s}; tp_step(s, ab, bb, 1024u, wl, wh, wlh, maxcls, acc, sh, nb, lane, repaired); } }
    for (uint i = 0u; i < 32u; ++i) total[i] = (z == 0u) ? acc[i] : half(total[i] + acc[i]);
  }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint r = tb + rb * 8u + tp_row_of(lane), n = nb + sub * 8u + tp_col_of(lane, e), idx = nr_a_index(r, nr_cfd(n), 1024u);
    out[idx] = nr_to_e4m3(total[tp_elem<8u>(rb, sub, e)]); atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed); }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}
kernel void nr_vit_contract_tp(device const uchar* expanded [[buffer(0)]], device const uchar* residual [[buffer(1)]], device const uchar* weights [[buffer(2)]], device uchar* out [[buffer(3)]], device atomic_uint* coverage [[buffer(4)]], constant NrParams& p [[buffer(5)]],
                               device const half* wl [[buffer(6)]], device const half* wh [[buffer(7)]], device const half* wlh [[buffer(8)]], device const uchar* maxcls [[buffer(9)]], device atomic_uint* stats [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup TpShared sh; tp_residual_gemm<4096u, 32u, 32768u, 4194304u>(expanded, residual, weights, out, coverage, p.tokens, wl, wh, wlh, maxcls, stats, tg, lane, sh);
}
kernel void nr_vit_projection_tp(device const uchar* attended [[buffer(0)]], device const uchar* residual [[buffer(1)]], device const uchar* weights [[buffer(2)]], device uchar* out [[buffer(3)]], device atomic_uint* coverage [[buffer(4)]], constant NrParams& p [[buffer(5)]],
                                 device const half* wl [[buffer(6)]], device const half* wh [[buffer(7)]], device const half* wlh [[buffer(8)]], device const uchar* maxcls [[buffer(9)]], device atomic_uint* stats [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup TpShared sh; tp_residual_gemm<1024u, 8u, 32768u, 1048576u>(attended, residual, weights, out, coverage, p.tokens, wl, wh, wlh, maxcls, stats, tg, lane, sh);
}

// ---- qkv: tokens (A identity, 1024) -> 3072 columns as 32 heads x 96 (q 0-31, k 32-63, v 64-95); K in two halves of 16 steps summed in half;
//      per-head QK normalisation epilogue (sum-of-squares tree with the fused first level, RTX rsqrt table). Tile: 16 tokens x one head (96 columns).
inline half tp_sumsq32(threadgroup const half* t) {
  half P[8];
  for (uint c = 0u; c < 8u; ++c) { const half a = fma(t[8u + c], t[8u + c], half(t[24u + c] * t[24u + c])); const half b = fma(t[c], t[c], half(t[16u + c] * t[16u + c])); P[c] = a + b; }
  half R1[8], R2[8]; for (uint c = 0u; c < 8u; ++c) R1[c] = P[c] + P[c ^ 4u]; for (uint c = 0u; c < 8u; ++c) R2[c] = R1[c] + R1[c ^ 2u];
  return max(half(R2[0] + R2[1]), nr_hbits(0x0410u));
}
kernel void nr_vit_qkv_tp(device const uchar* tokens [[buffer(0)]], device const uchar* weights [[buffer(1)]], device uchar* q_out [[buffer(2)]], device uchar* k_out [[buffer(3)]], device uchar* v_out [[buffer(4)]],
                          device atomic_uint* coverage_q [[buffer(5)]], device atomic_uint* coverage_k [[buffer(6)]], device atomic_uint* coverage_v [[buffer(7)]], device const ushort* rsqrt_table [[buffer(8)]], constant NrParams& p [[buffer(9)]],
                          device const half* wl [[buffer(10)]], device const half* wh [[buffer(11)]], device const half* wlh [[buffer(12)]], device const uchar* maxcls [[buffer(13)]], device atomic_uint* stats [[buffer(14)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup TpShared sh; threadgroup half tv[TP_ROWS * 96u];
  const uint h = tg.x, tb = tg.y * TP_ROWS, nb = 96u * h; if (tb >= p.tokens) return; device const uchar* blob = weights + 128u; uint repaired = 0u;
  half acc[48], p0[48];
  for (uint z = 0u; z < 2u; ++z) {
    for (uint i = 0u; i < 48u; ++i) acc[i] = half(0.0h);
    { TpWBlob bb{blob, 98304u}; for (uint s = 16u * z; s < 16u * z + 16u; ++s) { TpAIdentity ab{tokens, tb, s}; tp_step<12u>(s, ab, bb, 3072u, wl, wh, wlh, maxcls, acc, sh, nb, lane, repaired); } }
    for (uint i = 0u; i < 48u; ++i) { if (z == 0u) p0[i] = acc[i]; else acc[i] = half(p0[i] + acc[i]); }
  }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 12u; ++sub) for (uint e = 0u; e < 2u; ++e) { const uint r = rb * 8u + tp_row_of(lane), c = sub * 8u + tp_col_of(lane, e); tv[r * 96u + c] = acc[tp_elem<12u>(rb, sub, e)]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // epilogue: lanes 0-15 handle q of token lane; lanes 16-31 handle k and v of token lane-16
  const half head_scale = half(reinterpret_cast<device const float*>(weights)[h]); const half c32 = nr_hbits(0x45a8u);
  const uint r = lane % 16u, t = tb + r; threadgroup const half* row = tv + r * 96u;
  if (lane < 16u) { const half inv_q = nr_table_lookup(rsqrt_table, tp_sumsq32(row));
    for (uint d = 0u; d < 32u; ++d) { const uint qi = nr_d_index(t, 32u * h + d, 1024u); q_out[qi] = nr_to_e4m3(half(half(half(row[d] * inv_q) * c32) * head_scale)); atomic_fetch_add_explicit(&coverage_q[qi], 1u, memory_order_relaxed); } }
  else { const half inv_k = nr_table_lookup(rsqrt_table, tp_sumsq32(row + 32));
    for (uint d = 0u; d < 32u; ++d) { const uint ki = nr_k_index(t, d, h), vi = nr_v_index(t, d, h); k_out[ki] = nr_to_e4m3(half(row[32u + d] * inv_k)); v_out[vi] = nr_to_e4m3(row[64u + d]);
      atomic_fetch_add_explicit(&coverage_k[ki], 1u, memory_order_relaxed); atomic_fetch_add_explicit(&coverage_v[vi], 1u, memory_order_relaxed); } }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}

// ---- attention (full 640 keys per head): scores = one QMMA step (K = 32 dims, C = 0) of q against every key; e = bit trick; probabilities e4m3;
//      row sum per 64-key block as exact half trees (nr_tree8_exact order), blocks accumulated from 0; inv = RTX rcp table; PV = 20 chained QMMA
//      steps over keys; out = e4m3(half(PV) * inv). Both GEMMs are exact sums over identical term sets, so the natural dim / key order replaces the
//      slot permutation of the scalar kernel. Tile: 16 queries x one head. Operands prepared per launch from k/v: K^T [h][32 dims][640 keys],
//      V [h][20 key steps][32 keys][32 dims] as l/h/(l+h) halves plus max classes.
kernel void nr_vit_attn_prep(device const uchar* k_buf [[buffer(0)]], device const uchar* v_buf [[buffer(1)]], device half* kl [[buffer(2)]], device half* kh [[buffer(3)]], device half* klh [[buffer(4)]], device uchar* kmax [[buffer(5)]],
                             device half* vl [[buffer(6)]], device half* vh [[buffer(7)]], device half* vlh [[buffer(8)]], device uchar* vmax [[buffer(9)]], uint gid [[thread_position_in_grid]]) {
  // gid < 32*640: (h, key) -> K^T column; gid in [20480, 20480 + 32*20*32): (h, step, dim) -> V column
  if (gid < 32u * 640u) { const uint h = gid / 640u, key = gid % 640u; int mx = -1;
    for (uint d = 0u; d < 32u; ++d) { const uchar b = k_buf[nr_k_index(key, d, h)]; const int m = tp_int9(b); const int hh = (m + 256) >> 9; const int l = m - 512 * hh; const uint o = (h * 32u + d) * 640u + key; kl[o] = half(l); kh[o] = half(hh); klh[o] = half(l + hh); mx = max(mx, tp_class(b)); }
    kmax[h * 640u + key] = uchar(mx + 1); return; }
  const uint q2 = gid - 32u * 640u; if (q2 >= 32u * 20u * 32u) return; const uint h = q2 / 640u, s = (q2 / 32u) % 20u, d = q2 % 32u; int mx = -1;
  for (uint k = 0u; k < 32u; ++k) { const uchar b = v_buf[nr_v_index(32u * s + k, d, h)]; const int m = tp_int9(b); const int hh = (m + 256) >> 9; const int l = m - 512 * hh; const uint o = ((h * 20u + s) * 32u + k) * 32u + d; vl[o] = half(l); vh[o] = half(hh); vlh[o] = half(l + hh); mx = max(mx, tp_class(b)); }
  vmax[(h * 20u + s) * 32u + d] = uchar(mx + 1);
}
struct TpAQ { device const uchar* q; uint tb; uint h; uchar operator()(uint r, uint d) const { return q[nr_d_index(tb + r, 32u * h + d, 1024u)]; } };
struct TpBK { device const uchar* k; uint h; uchar operator()(uint, uint key, uint d) const { return k[nr_k_index(key, d, h)]; } };
struct TpAProb { threadgroup const uchar* prob; uint base; uchar operator()(uint r, uint j) const { return prob[r * 64u + base + j]; } };
struct TpBV { device const uchar* v; uint h; uchar operator()(uint s, uint d, uint j) const { return v[nr_v_index(32u * s + j, d, h)]; } };
inline half tp_tree8(threadgroup const half* e, uint b0) { const half s01 = e[b0] + e[b0 + 8u], s23 = e[b0 + 16u] + e[b0 + 24u], s45 = e[b0 + 32u] + e[b0 + 40u], s67 = e[b0 + 48u] + e[b0 + 56u]; const half s0123 = s01 + s23; const half s012345 = s0123 + s45; return s012345 + s67; }
kernel void nr_vit_attention_tp(device const uchar* q_buf [[buffer(0)]], device const uchar* k_buf [[buffer(1)]], device const uchar* v_buf [[buffer(2)]], device uchar* out [[buffer(3)]], device atomic_uint* coverage [[buffer(4)]], device const ushort* rcp_table [[buffer(5)]], constant NrParams& p [[buffer(6)]],
                                device const half* kl [[buffer(7)]], device const half* kh [[buffer(8)]], device const half* klh [[buffer(9)]], device const uchar* kmax [[buffer(10)]],
                                device const half* vl [[buffer(11)]], device const half* vh [[buffer(12)]], device const half* vlh [[buffer(13)]], device const uchar* vmax [[buffer(14)]], device atomic_uint* stats [[buffer(15)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup TpShared sh; threadgroup half eT[TP_ROWS * 64u]; threadgroup uchar probT[TP_ROWS * 64u]; threadgroup half totalT[TP_ROWS];   // (a second staged struct for q measured slower: tp-attempt-v9)
  const uint h = tg.x, tb = tg.y * TP_ROWS; if (tb >= p.tokens) return; uint repaired = 0u;
  device const half* hkl = kl + h * 32u * 640u; device const half* hkh = kh + h * 32u * 640u; device const half* hklh = klh + h * 32u * 640u; device const uchar* hkmax = kmax + h * 640u;
  device const half* hvl = vl + h * 20u * 32u * 32u; device const half* hvh = vh + h * 20u * 32u * 32u; device const half* hvlh = vlh + h * 20u * 32u * 32u; device const uchar* hvmax = vmax + h * 20u * 32u;
  if (lane < TP_ROWS) totalT[lane] = nr_hbits(0u);
  half accd[16]; for (uint i = 0u; i < 16u; ++i) accd[i] = half(0.0h);
  TpAQ aq{q_buf, tb, h}; TpBK bk{k_buf, h}; TpBV bv{v_buf, h};
  for (uint blk = 0u; blk < 10u; ++blk) {
    // scores for keys 64blk..64blk+63: one K = 32 step, C = 0, columns = keys (cols stride 640)
    half sc[32]; for (uint i = 0u; i < 32u; ++i) sc[i] = half(0.0h);
    tp_step<8u>(0u, aq, bk, 640u, hkl, hkh, hklh, hkmax, sc, sh, 64u * blk, lane, repaired);
    for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
      const uint r = rb * 8u + tp_row_of(lane), c = sub * 8u + tp_col_of(lane, e); const half score = sc[tp_elem<8u>(rb, sub, e)];
      half vv = fma(score, nr_hbits(0x2dbbu), nr_hbits(0x3ed6u)); vv = min(max(vv, nr_hbits(0x3dc2u)), nr_hbits(0x3fe9u));
      const half ee = as_type<half>(ushort((uint(as_type<ushort>(vv)) << 4u) + 0x4000u)); eT[r * 64u + c] = ee; probT[r * 64u + c] = nr_to_e4m3(ee); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < TP_ROWS) { threadgroup const half* e = eT + lane * 64u; half slo = nr_hbits(0u), shi = nr_hbits(0u);
      for (uint qd = 0u; qd < 4u; ++qd) { const half plo = tp_tree8(e, 2u * qd), phi = tp_tree8(e, 2u * qd + 1u); slo = (qd == 0u) ? plo : half(slo + plo); shi = (qd == 0u) ? phi : half(shi + phi); }
      totalT[lane] = half(totalT[lane] + half(slo + shi)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // PV over this block's 64 keys: two chained steps (global key steps 2blk, 2blk+1), columns = 32 dims (cols stride 32), 4 sub-tiles
    for (uint st = 0u; st < 2u; ++st) { TpAProb ap{probT, 32u * st}; tp_step<4u>(2u * blk + st, ap, bv, 32u, hvl, hvh, hvlh, hvmax, accd, sh, 0u, lane, repaired); }
  }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 4u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint r = rb * 8u + tp_row_of(lane), d = sub * 8u + tp_col_of(lane, e); const half inv = nr_table_lookup(rcp_table, max(totalT[r], nr_hbits(0x0410u)));
    const uint idx = nr_d_index(tb + r, 32u * h + d, 1024u); out[idx] = nr_to_e4m3(half(accd[tp_elem<4u>(rb, sub, e)] * inv)); atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed); }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}
