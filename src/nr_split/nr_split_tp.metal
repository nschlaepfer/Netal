// Three-product exact QMMA split conv (v5 engine structure, see src/nr_vit/nr_vit_tp.metal): [tokens x 512] x [512 x 512], K = 16 rounded steps,
// residual init acc = half(res[cfd(n)] * scale[n]); flags as nr_split_conv (RES_VIEW / OUT_VIEW / EMIT_F16). One SIMD group per 16 tokens x 64 columns;
// each lane owns 32 outputs in registers; l/h/(l+h) halves of A staged in threadgroup memory, weights prepared once per launch (nr_split_tp_prep);
// max_exp bounded by row max class + column max class; one-pack certificate; lane-local repair with the accepted rule (nr_qmma_step).
#include "nr_split_common.metal"
#include <metal_simdgroup_matrix>
constant uint SP_ROWS = 16u;
struct SpShared { half al[SP_ROWS * 32u], ah[SP_ROWS * 32u], alh[SP_ROWS * 32u]; uchar ab[SP_ROWS * 32u]; uint maxA[SP_ROWS]; };
inline int sp_int9(uchar b) { const int e = int((b >> 3u) & 15u), m = int(b & 7u); const int sig = (e == 0) ? m : 8 + m; const int q = (e == 0) ? -9 : e - 10; const int v = sig << uint(q + 9); return (b & 0x80u) ? -v : v; }
inline int sp_class(uchar b) { const int e = int((b >> 3u) & 15u), m = int(b & 7u); if (e == 0) { if (m == 0) return -1; return (31 - int(clz(uint(m)))); } return e + 2; }   // floor(log2|x|) + 9, -1 for zero
kernel void nr_split_tp_prep(device const uchar* w [[buffer(0)]], device half* wl [[buffer(1)]], device half* wh [[buffer(2)]], device half* wlh [[buffer(3)]], device uchar* maxcls [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const uint s = gid / 512u, n = gid % 512u; if (s >= 16u) return; int mx = -1;
  for (uint k = 0u; k < 32u; ++k) { const uchar b = w[s * 16384u + (n / 64u) * 2048u + nr_weight_index(k, n % 64u)]; const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
    const uint o = (s * 32u + k) * 512u + n; wl[o] = half(l); wh[o] = half(h); wlh[o] = half(l + h); mx = max(mx, sp_class(b)); }
  maxcls[s * 512u + n] = uchar(mx + 1);
}
inline uint sp_row_of(uint lane) { return (lane % 8u) / 2u + 4u * (lane / 16u); }
inline uint sp_col_of(uint lane, uint e) { return 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u) + e; }
inline uint sp_elem(uint rb, uint sub, uint e) { return (rb * 8u + sub) * 2u + e; }
inline bool sp_certified(long S, long G) {
  if (G == 0l) return true; const ulong mag = ulong(S < 0l ? -S : S); if (mag == 0ul) return false;
  const uint hi = uint(mag >> 32u); const uint bl = hi != 0u ? 64u - clz(hi) : 32u - clz(uint(mag)); if (bl <= 10u) return false;
  const long ulp = 1l << (bl - 11u), half_ulp = ulp >> 1u; const long r = long(mag & ulong(ulp - 1l)); const long d = r - half_ulp; const long edge_dist = long(mag) - (1l << (bl - 1u)); const long g33 = 33l * G;
  return (g33 < (d < 0l ? -d : d)) && (g33 < edge_dist + (ulp >> 2u));
}
inline long sp_c_units(ushort cb, thread int& cexp) {
  const int ce = int((cb >> 10u) & 31u), cf = int(cb & 1023u); int sig, q; if (ce != 0) { sig = 1024 + cf; q = ce - 25; } else { sig = cf; q = -24; }
  if (sig == 0) { cexp = -1000; return 0l; } cexp = q + (31 - int(clz(uint(sig)))); long v = long(sig) << uint(q + 24); return (cb & 0x8000u) ? -v : v;
}
kernel void nr_split_conv_tp(device const uchar* x [[buffer(0)]], device const uchar* res [[buffer(1)]], device const uchar* w [[buffer(2)]], device uchar* out [[buffer(3)]], device half* y16 [[buffer(4)]], device atomic_uint* coverage [[buffer(5)]], constant NrSplitParams& p [[buffer(6)]],
                             device const half* wl [[buffer(7)]], device const half* wh [[buffer(8)]], device const half* wlh [[buffer(9)]], device const uchar* maxcls [[buffer(10)]], device atomic_uint* stats [[buffer(11)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup SpShared sh; const uint tb = tg.y * SP_ROWS, nb = tg.x * 64u; if (tb >= p.tokens) return;
  const bool res_view = (p.flags & NR_FLAG_RES_VIEW) != 0u, out_view = (p.flags & NR_FLAG_OUT_VIEW) != 0u, emit = (p.flags & NR_FLAG_EMIT_F16) != 0u;
  device const half* scale = reinterpret_cast<device const half*>(w + 262144u);
  half acc[32]; uint repaired = 0u;
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) { const uint t = tb + rb * 8u + sp_row_of(lane), n = nb + sub * 8u + sp_col_of(lane, e); acc[sp_elem(rb, sub, e)] = half(nr_load_token(res, t, nr_cfd(n), res_view) * scale[n]); }
  for (uint s = 0u; s < 16u; ++s) {
    { const uint r = lane / 2u, k0 = (lane % 2u) * 16u; uint mx = 0u;
      for (uint j = 0u; j < 16u; ++j) { const uint k = k0 + j, i = r * 32u + k; const uchar b = x[nr_tin(tb + r, 32u * s + k)]; const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
        sh.al[i] = half(l); sh.ah[i] = half(h); sh.alh[i] = half(l + h); sh.ab[i] = b; mx = max(mx, uint(sp_class(b) + 1)); }
      mx = max(mx, simd_shuffle_xor(mx, 1u)); if ((lane % 2u) == 0u) sh.maxA[r] = mx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rmask = 0u;
    for (uint sub = 0u; sub < 8u; ++sub) {
      const uint n0 = nb + sub * 8u; simdgroup_half8x8 BL[4], BH[4], BLH[4];
      for (uint kk = 0u; kk < 4u; ++kk) { const uint boff = (s * 32u + kk * 8u) * 512u + n0; simdgroup_load(BL[kk], wl + boff, 512u); simdgroup_load(BH[kk], wh + boff, 512u); simdgroup_load(BLH[kk], wlh + boff, 512u); }
      for (uint rb = 0u; rb < 2u; ++rb) {
        simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), P1 = P0, P2 = P0; simdgroup_half8x8 A;
        for (uint kk = 0u; kk < 4u; ++kk) { const uint ao = rb * 256u + kk * 8u;
          simdgroup_load(A, sh.al + ao, 32u); simdgroup_multiply_accumulate(P0, A, BL[kk], P0); simdgroup_load(A, sh.ah + ao, 32u); simdgroup_multiply_accumulate(P1, A, BH[kk], P1); simdgroup_load(A, sh.alh + ao, 32u); simdgroup_multiply_accumulate(P2, A, BLH[kk], P2); }
        for (uint e = 0u; e < 2u; ++e) {
          const uint r = rb * 8u + sp_row_of(lane), c = sp_col_of(lane, e), n = n0 + c, ei = sp_elem(rb, sub, e);
          const int p0 = int(P0.thread_elements()[e]), p1 = int(P1.thread_elements()[e]), p2 = int(P2.thread_elements()[e]); const int cross = p2 - p0 - p1;
          const long recon = long(p0) + (long(cross) << 9u) + (long(p1) << 18u);
          const ushort cb = as_type<ushort>(acc[ei]); int cexp; const long cval = sp_c_units(cb, cexp); const long S = (recon << 6u) + cval;
          const uint ma1 = sh.maxA[r], mb1 = uint(maxcls[s * 512u + n]); int maxexp = (ma1 != 0u && mb1 != 0u) ? int(ma1 + mb1) - 20 : -1000; maxexp = max(maxexp, cexp);
          if (maxexp == -1000) acc[ei] = as_type<half>(ushort(0u));
          else { const int gexp = maxexp - 25; const long G = (gexp + 24 >= 0) ? (1l << uint(gexp + 24)) : 0l; if (sp_certified(S, G)) acc[ei] = nr_hw_fixed_to_half_rne(S, -24); else rmask |= 1u << ei; }
        }
      }
    }
    repaired += popcount(rmask);
    while (rmask != 0u) { const uint ei = ctz(rmask); rmask &= rmask - 1u; const uint rb = ei / 16u, sub = (ei % 16u) / 2u, e = ei % 2u; const uint r = rb * 8u + sp_row_of(lane), n = nb + sub * 8u + sp_col_of(lane, e); half a[32], b[32];
      for (uint j = 0u; j < 32u; ++j) { a[j] = nr_e4m3(sh.ab[r * 32u + j]); b[j] = nr_wload(w, s, 16384u, n, j); }
      acc[ei] = nr_qmma_step(a, b, acc[ei]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint t = tb + rb * 8u + sp_row_of(lane), n = nb + sub * 8u + sp_col_of(lane, e), ch = nr_cfd(n); uint rr = 0u, cc = 0u; nr_token_rc(t, rr, cc);
    const uint idx = out_view ? nr_view_index(rr, cc, ch) : nr_tin(t, ch); const half v = acc[sp_elem(rb, sub, e)];
    out[idx] = nr_to_e4m3(v); atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed); if (emit) y16[t * 512u + n] = v; }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}
// layout probe (harness self-check of the simdgroup element mapping): C = A(index matrix) * I; each lane stores its two thread_elements
kernel void nr_tp_layout_probe(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device float* outp [[buffer(2)]], uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_half8x8 A, B; simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_load(A, a, 8); simdgroup_load(B, b, 8); simdgroup_multiply_accumulate(C, A, B, C);
  outp[2u * lane] = C.thread_elements()[0]; outp[2u * lane + 1u] = C.thread_elements()[1];
}

// ============================================================================================================================
// Split ffwd (two-layer MLP per 64-hidden group g): layer 1 [tokens x 512] x [512 x 64g..64g+63] (16 rounded steps) -> h1 (e4m3-rounded);
// layer 2 per 32-output chunk s3: K = 64 as two rounded steps over h1 (slot order nr_slot_channel) -> h2, cubic SiLU, e4m3-round;
// layer 3: acc[64] += chunk(32) x w3 (8 rounded steps). Same tile engine: 16 tokens x 64 columns per SIMD group, lane-local ownership.
// Generic tile step over an A-byte source (device or threadgroup) with prepared weights [steps][32][cols].
// ============================================================================================================================
struct SpSrcX { device const uchar* x; uint tb; uint s; bool view; uchar operator()(uint r, uint k) const { const uint t = tb + r; if (!view) return x[nr_tin(t, 32u * s + k)]; uint rr, cc; nr_token_rc(t, rr, cc); return x[nr_view_index(rr, cc, 32u * s + k)]; } };
struct SpSrcTG { threadgroup const uchar* h; uint base; uchar operator()(uint r, uint k) const { return h[r * 64u + base + nr_slot_channel(k)]; } };   // K-slot order over a staged 16 x 64 e4m3 tile
template <typename AByte, typename BByte>
inline void sp_tile_step(uint s, thread const AByte& abyte, thread const BByte& bbyte, uint cols, device const half* wl, device const half* wh, device const half* wlh, device const uchar* maxcls,
                         thread half* acc, threadgroup SpShared& sh, uint nb, uint lane, thread uint& repaired) {
  { const uint r = lane / 2u, k0 = (lane % 2u) * 16u; uint mx = 0u;
    for (uint j = 0u; j < 16u; ++j) { const uint k = k0 + j, i = r * 32u + k; const uchar b = abyte(r, k); const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
      sh.al[i] = half(l); sh.ah[i] = half(h); sh.alh[i] = half(l + h); sh.ab[i] = b; mx = max(mx, uint(sp_class(b) + 1)); }
    mx = max(mx, simd_shuffle_xor(mx, 1u)); if ((lane % 2u) == 0u) sh.maxA[r] = mx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint rmask = 0u;
  for (uint sub = 0u; sub < 8u; ++sub) {
    const uint n0 = nb + sub * 8u; simdgroup_half8x8 BL[4], BH[4], BLH[4];
    for (uint kk = 0u; kk < 4u; ++kk) { const uint boff = (s * 32u + kk * 8u) * cols + n0; simdgroup_load(BL[kk], wl + boff, cols); simdgroup_load(BH[kk], wh + boff, cols); simdgroup_load(BLH[kk], wlh + boff, cols); }
    for (uint rb = 0u; rb < 2u; ++rb) {
      simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), P1 = P0, P2 = P0; simdgroup_half8x8 A;
      for (uint kk = 0u; kk < 4u; ++kk) { const uint ao = rb * 256u + kk * 8u;
        simdgroup_load(A, sh.al + ao, 32u); simdgroup_multiply_accumulate(P0, A, BL[kk], P0); simdgroup_load(A, sh.ah + ao, 32u); simdgroup_multiply_accumulate(P1, A, BH[kk], P1); simdgroup_load(A, sh.alh + ao, 32u); simdgroup_multiply_accumulate(P2, A, BLH[kk], P2); }
      for (uint e = 0u; e < 2u; ++e) {
        const uint r = rb * 8u + sp_row_of(lane), c = sp_col_of(lane, e), n = n0 + c, ei = sp_elem(rb, sub, e);
        const int p0 = int(P0.thread_elements()[e]), p1 = int(P1.thread_elements()[e]), p2 = int(P2.thread_elements()[e]); const int cross = p2 - p0 - p1;
        const long recon = long(p0) + (long(cross) << 9u) + (long(p1) << 18u);
        const ushort cb = as_type<ushort>(acc[ei]); int cexp; const long cval = sp_c_units(cb, cexp); const long S = (recon << 6u) + cval;
        const uint ma1 = sh.maxA[r], mb1 = uint(maxcls[s * cols + n]); int maxexp = (ma1 != 0u && mb1 != 0u) ? int(ma1 + mb1) - 20 : -1000; maxexp = max(maxexp, cexp);
        if (maxexp == -1000) acc[ei] = as_type<half>(ushort(0u));
        else { const int gexp = maxexp - 25; const long G = (gexp + 24 >= 0) ? (1l << uint(gexp + 24)) : 0l; if (sp_certified(S, G)) acc[ei] = nr_hw_fixed_to_half_rne(S, -24); else rmask |= 1u << ei; }
      }
    }
  }
  repaired += popcount(rmask);
  while (rmask != 0u) { const uint ei = ctz(rmask); rmask &= rmask - 1u; const uint rb = ei / 16u, sub = (ei % 16u) / 2u, e = ei % 2u; const uint r = rb * 8u + sp_row_of(lane), n = nb + sub * 8u + sp_col_of(lane, e); half a[32], b[32];
    for (uint j = 0u; j < 32u; ++j) { a[j] = nr_e4m3(sh.ab[r * 32u + j]); b[j] = nr_e4m3(bbyte(s, n, j)); }
    acc[ei] = nr_qmma_step(a, b, acc[ei]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// weight byte accessors for the repair path (same indexing as nr_split_ffwd)
struct SpW1 { device const uchar* w; uchar operator()(uint s, uint n, uint j) const { return w[s * 16384u + (n / 64u) * 2048u + nr_weight_index(j, n % 64u)]; } };
struct SpW2 { device const uchar* w2; uchar operator()(uint s, uint n2, uint j) const { return w2[(s ? 8192u : 0u) + (n2 / 32u) * 1024u + nr_weight_index(j, n2 % 32u)]; } };
struct SpW3 { device const uchar* w3; uchar operator()(uint s3, uint i, uint j) const { return w3[s3 * 2048u + nr_weight_index(j, i)]; } };
// prep: layer 1 uses nr_split_tp_prep (16 steps x 512 columns). Layer 2: per group g, [2 steps][32][256]; layer 3: per group g, [8 steps][32][64].
kernel void nr_split_ffwd_tp_prep(device const uchar* w [[buffer(0)]], device half* w2l [[buffer(1)]], device half* w2h [[buffer(2)]], device half* w2lh [[buffer(3)]], device uchar* w2max [[buffer(4)]],
                                  device half* w3l [[buffer(5)]], device half* w3h [[buffer(6)]], device half* w3lh [[buffer(7)]], device uchar* w3max [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  // gid < 8*2*256 : layer-2 entry (g, s, n2); gid in [4096, 4096 + 8*8*64): layer-3 entry (g, s3, i)
  if (gid < 8u * 2u * 256u) { const uint g = gid / 512u, s = (gid / 256u) % 2u, n2 = gid % 256u; device const uchar* w2 = w + 262144u + 16384u * g; SpW2 acc{w2}; int mx = -1;
    for (uint k = 0u; k < 32u; ++k) { const uchar b = acc(s, n2, k); const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h; const uint o = ((g * 2u + s) * 32u + k) * 256u + n2; w2l[o] = half(l); w2h[o] = half(h); w2lh[o] = half(l + h); mx = max(mx, sp_class(b)); }
    w2max[(g * 2u + s) * 256u + n2] = uchar(mx + 1); return; }
  const uint q = gid - 8u * 2u * 256u; if (q >= 8u * 8u * 64u) return; const uint g = q / 512u, s3 = (q / 64u) % 8u, i = q % 64u; device const uchar* w3 = w + 393216u + 16384u * g; SpW3 acc{w3}; int mx = -1;
  for (uint k = 0u; k < 32u; ++k) { const uchar b = acc(s3, i, k); const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h; const uint o = ((g * 8u + s3) * 32u + k) * 64u + i; w3l[o] = half(l); w3h[o] = half(h); w3lh[o] = half(l + h); mx = max(mx, sp_class(b)); }
  w3max[(g * 8u + s3) * 64u + i] = uchar(mx + 1);
}
kernel void nr_split_ffwd_tp(device const uchar* x [[buffer(0)]], device const uchar* w [[buffer(1)]], device uchar* out [[buffer(2)]], device atomic_uint* coverage [[buffer(3)]], constant NrSplitParams& p [[buffer(4)]],
                             device const half* w1l [[buffer(5)]], device const half* w1h [[buffer(6)]], device const half* w1lh [[buffer(7)]], device const uchar* w1max [[buffer(8)]],
                             device const half* w2l [[buffer(9)]], device const half* w2h [[buffer(10)]], device const half* w2lh [[buffer(11)]], device const uchar* w2max [[buffer(12)]],
                             device const half* w3l [[buffer(13)]], device const half* w3h [[buffer(14)]], device const half* w3lh [[buffer(15)]], device const uchar* w3max [[buffer(16)]], device atomic_uint* stats [[buffer(17)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup SpShared sh; threadgroup uchar h1b[16u * 64u], h2b[16u * 64u];
  const uint g = tg.x, tb = tg.y * SP_ROWS; if (tb >= p.tokens) return; const bool view = (p.flags & NR_FLAG_IN_VIEW) != 0u; uint repaired = 0u;
  // ---- layer 1: 16 tokens x 64 hidden (columns 64g..64g+63 of the 512-wide prepared w1) ----
  half acc[32]; for (uint i = 0u; i < 32u; ++i) acc[i] = half(0.0h);
  { SpW1 bb{w}; for (uint s = 0u; s < 16u; ++s) { SpSrcX ab{x, tb, s, view}; sp_tile_step(s, ab, bb, 512u, w1l, w1h, w1lh, w1max, acc, sh, 64u * g, lane, repaired); } }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) { const uint r = rb * 8u + sp_row_of(lane), c = sub * 8u + sp_col_of(lane, e); h1b[r * 64u + c] = nr_to_e4m3(acc[sp_elem(rb, sub, e)]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // ---- layers 2 and 3 in pairs of 32-output chunks: h2 tile 16 x 64 (chunks 2q, 2q+1; K = 64 as two rounded steps), SiLU + e4m3 round, then two layer-3 steps ----
  half acc3[32]; for (uint i = 0u; i < 32u; ++i) acc3[i] = half(0.0h);
  device const half* g2l = w2l + (g * 2u) * 32u * 256u; device const half* g2h = w2h + (g * 2u) * 32u * 256u; device const half* g2lh = w2lh + (g * 2u) * 32u * 256u; device const uchar* g2max = w2max + (g * 2u) * 256u;
  device const half* g3l = w3l + (g * 8u) * 32u * 64u; device const half* g3h = w3h + (g * 8u) * 32u * 64u; device const half* g3lh = w3lh + (g * 8u) * 32u * 64u; device const uchar* g3max = w3max + (g * 8u) * 64u;
  SpW2 bb2{w + 262144u + 16384u * g}; SpW3 bb3{w + 393216u + 16384u * g};
  for (uint q = 0u; q < 4u; ++q) {
    half acc2[32]; for (uint i = 0u; i < 32u; ++i) acc2[i] = half(0.0h);
    for (uint s = 0u; s < 2u; ++s) { SpSrcTG ab{h1b, 32u * s}; sp_tile_step(s, ab, bb2, 256u, g2l, g2h, g2lh, g2max, acc2, sh, 64u * q, lane, repaired); }
    for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) { const uint r = rb * 8u + sp_row_of(lane), c = sub * 8u + sp_col_of(lane, e); h2b[r * 64u + c] = nr_to_e4m3(nr_cubic_silu(acc2[sp_elem(rb, sub, e)])); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_ = 0u; half_ < 2u; ++half_) { const uint s3 = 2u * q + half_; SpSrcTG ab{h2b, 32u * half_}; sp_tile_step(s3, ab, bb3, 64u, g3l, g3h, g3lh, g3max, acc3, sh, 0u, lane, repaired); }
  }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint t = tb + rb * 8u + sp_row_of(lane), i = sub * 8u + sp_col_of(lane, e), idx = nr_tin(t, nr_cfd(64u * g + i)); out[idx] = nr_to_e4m3(acc3[sp_elem(rb, sub, e)]); atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed); }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}

// ============================================================================================================================
// Split qkv GEMM (cc_split_swin_16h_qkv first phase): [tokens x 512] x [512 x 1536] (16 heads x 96: q 0-31, k 32-63, v 64-95), 16 rounded
// steps, per-head normalisation (split sum-of-squares order, RTX rsqrt table, q = half(half(acc*inv_q)*head_scale)), compact output
// (t*16+h)*32+d. Tile: 16 tokens x one head (96 columns = 12 sub-tiles, 48 lane-local elements, 64-bit repair mask).
// ============================================================================================================================
struct SpPrepParams { uint steps; uint cols; uint step_bytes; uint reserved; };
kernel void nr_split_tp_prep_generic(device const uchar* w [[buffer(0)]], device half* wl [[buffer(1)]], device half* wh [[buffer(2)]], device half* wlh [[buffer(3)]], device uchar* maxcls [[buffer(4)]], constant SpPrepParams& pp [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  const uint s = gid / pp.cols, n = gid % pp.cols; if (s >= pp.steps) return; int mx = -1;
  for (uint k = 0u; k < 32u; ++k) { const uchar b = w[s * pp.step_bytes + (n / 64u) * 2048u + nr_weight_index(k, n % 64u)]; const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
    const uint o = (s * 32u + k) * pp.cols + n; wl[o] = half(l); wh[o] = half(h); wlh[o] = half(l + h); mx = max(mx, sp_class(b)); }
  maxcls[s * pp.cols + n] = uchar(mx + 1);
}
template <uint SUBS> inline uint sp_elem_w(uint rb, uint sub, uint e) { return (rb * SUBS + sub) * 2u + e; }
template <uint SUBS, typename AByte, typename BByte>
inline void sp_tile_step_w(uint s, thread const AByte& abyte, thread const BByte& bbyte, uint cols, device const half* wl, device const half* wh, device const half* wlh, device const uchar* maxcls,
                           thread half* acc, threadgroup SpShared& sh, uint nb, uint lane, thread uint& repaired) {
  { const uint r = lane / 2u, k0 = (lane % 2u) * 16u; uint mx = 0u;
    for (uint j = 0u; j < 16u; ++j) { const uint k = k0 + j, i = r * 32u + k; const uchar b = abyte(r, k); const int m = sp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h;
      sh.al[i] = half(l); sh.ah[i] = half(h); sh.alh[i] = half(l + h); sh.ab[i] = b; mx = max(mx, uint(sp_class(b) + 1)); }
    mx = max(mx, simd_shuffle_xor(mx, 1u)); if ((lane % 2u) == 0u) sh.maxA[r] = mx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong rmask = 0ul;
  for (uint sub = 0u; sub < SUBS; ++sub) {
    const uint n0 = nb + sub * 8u; simdgroup_half8x8 BL[4], BH[4], BLH[4];
    for (uint kk = 0u; kk < 4u; ++kk) { const uint boff = (s * 32u + kk * 8u) * cols + n0; simdgroup_load(BL[kk], wl + boff, cols); simdgroup_load(BH[kk], wh + boff, cols); simdgroup_load(BLH[kk], wlh + boff, cols); }
    for (uint rb = 0u; rb < 2u; ++rb) {
      simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), P1 = P0, P2 = P0; simdgroup_half8x8 A;
      for (uint kk = 0u; kk < 4u; ++kk) { const uint ao = rb * 256u + kk * 8u;
        simdgroup_load(A, sh.al + ao, 32u); simdgroup_multiply_accumulate(P0, A, BL[kk], P0); simdgroup_load(A, sh.ah + ao, 32u); simdgroup_multiply_accumulate(P1, A, BH[kk], P1); simdgroup_load(A, sh.alh + ao, 32u); simdgroup_multiply_accumulate(P2, A, BLH[kk], P2); }
      for (uint e = 0u; e < 2u; ++e) {
        const uint r = rb * 8u + sp_row_of(lane), c = sp_col_of(lane, e), n = n0 + c, ei = sp_elem_w<SUBS>(rb, sub, e);
        const int p0 = int(P0.thread_elements()[e]), p1 = int(P1.thread_elements()[e]), p2 = int(P2.thread_elements()[e]); const int cross = p2 - p0 - p1;
        const long recon = long(p0) + (long(cross) << 9u) + (long(p1) << 18u);
        const ushort cb = as_type<ushort>(acc[ei]); int cexp; const long cval = sp_c_units(cb, cexp); const long S = (recon << 6u) + cval;
        const uint ma1 = sh.maxA[r], mb1 = uint(maxcls[s * cols + n]); int maxexp = (ma1 != 0u && mb1 != 0u) ? int(ma1 + mb1) - 20 : -1000; maxexp = max(maxexp, cexp);
        if (maxexp == -1000) acc[ei] = as_type<half>(ushort(0u));
        else { const int gexp = maxexp - 25; const long G = (gexp + 24 >= 0) ? (1l << uint(gexp + 24)) : 0l; if (sp_certified(S, G)) acc[ei] = nr_hw_fixed_to_half_rne(S, -24); else rmask |= 1ul << ei; }
      }
    }
  }
  repaired += popcount(rmask);
  while (rmask != 0ul) { const uint ei = uint(ctz(rmask)); rmask &= rmask - 1ul; const uint rb = ei / (2u * SUBS), sub = (ei % (2u * SUBS)) / 2u, e = ei % 2u; const uint r = rb * 8u + sp_row_of(lane), n = nb + sub * 8u + sp_col_of(lane, e); half a[32], b[32];
    for (uint j = 0u; j < 32u; ++j) { a[j] = nr_e4m3(sh.ab[r * 32u + j]); b[j] = nr_e4m3(bbyte(s, n, j)); }
    acc[ei] = nr_qmma_step(a, b, acc[ei]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
struct SpWQ { device const uchar* w; uchar operator()(uint s, uint n, uint j) const { return w[s * 49152u + (n / 64u) * 2048u + nr_weight_index(j, n % 64u)]; } };
inline half sp_split_sumsq32(threadgroup const half* t) {
  half P[8];
  for (uint c = 0u; c < 8u; ++c) { const half a = fma(t[c], t[c], half(t[16u + c] * t[16u + c])); const half b = fma(t[8u + c], t[8u + c], half(t[24u + c] * t[24u + c])); P[c] = b + a; }
  half R1[8], R2[8]; for (uint c = 0u; c < 8u; ++c) R1[c] = P[c] + P[c ^ 4u]; for (uint c = 0u; c < 8u; ++c) R2[c] = R1[c] + R1[c ^ 2u];
  return max(half(R2[0] + R2[1]), nr_hbits(0x0410u));
}
kernel void nr_split_qkv_tp(device const uchar* x [[buffer(0)]], device const uchar* w [[buffer(1)]], device uchar* q_out [[buffer(2)]], device uchar* k_out [[buffer(3)]], device uchar* v_out [[buffer(4)]], device const ushort* rsqrt_table [[buffer(5)]], constant NrSplitParams& p [[buffer(6)]],
                            device const half* wl [[buffer(7)]], device const half* wh [[buffer(8)]], device const half* wlh [[buffer(9)]], device const uchar* maxcls [[buffer(10)]], device atomic_uint* stats [[buffer(11)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup SpShared sh; threadgroup half tv[SP_ROWS * 96u];
  const uint h = tg.x, tb = tg.y * SP_ROWS, nb = 96u * h; if (tb >= p.tokens) return; uint repaired = 0u;
  half acc[48]; for (uint i = 0u; i < 48u; ++i) acc[i] = half(0.0h);
  { SpWQ bb{w}; for (uint s = 0u; s < 16u; ++s) { SpSrcX ab{x, tb, s, false}; sp_tile_step_w<12u>(s, ab, bb, 1536u, wl, wh, wlh, maxcls, acc, sh, nb, lane, repaired); } }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 12u; ++sub) for (uint e = 0u; e < 2u; ++e) { const uint r = rb * 8u + sp_row_of(lane), c = sub * 8u + sp_col_of(lane, e); tv[r * 96u + c] = acc[sp_elem_w<12u>(rb, sub, e)]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const half hs = half(reinterpret_cast<device const float*>(w + 917504u)[h]); const uint r = lane % 16u, t = tb + r; threadgroup const half* row = tv + r * 96u; const uint base = (t * 16u + h) * 32u;
  if (lane < 16u) { const half inv_q = nr_table_lookup(rsqrt_table, sp_split_sumsq32(row)); for (uint d = 0u; d < 32u; ++d) q_out[base + d] = nr_to_e4m3(half(half(row[d] * inv_q) * hs)); }
  else { const half inv_k = nr_table_lookup(rsqrt_table, sp_split_sumsq32(row + 32)); for (uint d = 0u; d < 32u; ++d) { k_out[base + d] = nr_to_e4m3(half(row[32u + d] * inv_k)); v_out[base + d] = nr_to_e4m3(row[64u + d]); } }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}

// ============================================================================================================================
// Split windowed attention (cc_split_swin_16h_qkv second phase) on the engine. Per (window, head, 16-query quarter): scores = one K = 32 step of
// the 16 queries against the window's 64 keys with the per-(query, key) bias as C (out-of-grid keys have a zero K^T column, so score = bias exactly,
// as in the scalar kernel); e = bit trick; row sum as the exact half trees; inv = RTX rcp; prob = e4m3(half(e * inv)); PV = two chained steps over the
// window keys against the window's V (zero rows for out-of-grid keys); out = e4m3(acc) at channel 32h + cfd(d) for in-grid queries. K^T and V per
// (window, head) are prepared from the compact q/k/v buffers; both GEMMs are exact sums over identical term sets, so natural dim/key order is used.
// ============================================================================================================================
inline int sp_win_token(uint win, uint slot, int sx, int sy) {
  const uint wx = win % 8u, wy = win / 8u; const int ox = int(8u * wx) + sx, oy = int(8u * wy) + sy; const uint mt = slot / 16u, g = slot % 16u;
  const int tx = (ox >> 2) + int(mt % 2u), ty = (oy >> 2) + int(mt / 2u);
  return (tx < 0 || ty < 0 || tx >= int(NR_SPLIT_TILES_X) || ty >= int(NR_SPLIT_TILES_Y)) ? -1 : int((uint(ty) * NR_SPLIT_TILES_X + uint(tx)) * 16u + g);
}
kernel void nr_split_attend_tp_prep(device const uchar* k_buf [[buffer(0)]], device const uchar* v_buf [[buffer(1)]], constant NrSplitParams& p [[buffer(2)]],
                                    device half* kl [[buffer(3)]], device half* kh [[buffer(4)]], device half* klh [[buffer(5)]], device uchar* kmax [[buffer(6)]],
                                    device half* vl [[buffer(7)]], device half* vh [[buffer(8)]], device half* vlh [[buffer(9)]], device uchar* vmax [[buffer(10)]], uint gid [[thread_position_in_grid]]) {
  // gid < 40*16*64: (win, h, key slot) -> K^T column [win][h][32 d][64 keys]; gid in [40960, 40960 + 40*16*2*32): (win, h, step, dim) -> V column [win][h][2][32 keys][32 d]
  if (gid < 40u * 16u * 64u) { const uint win = gid / 1024u, h = (gid / 64u) % 16u, ks = gid % 64u; const int kt = sp_win_token(win, ks, p.shift_x, p.shift_y); int mx = -1;
    for (uint d = 0u; d < 32u; ++d) { const uchar b = kt < 0 ? uchar(0) : k_buf[(uint(kt) * 16u + h) * 32u + d]; const int m = sp_int9(b); const int hh = (m + 256) >> 9; const int l = m - 512 * hh; const uint o = ((win * 16u + h) * 32u + d) * 64u + ks; kl[o] = half(l); kh[o] = half(hh); klh[o] = half(l + hh); mx = max(mx, sp_class(b)); }
    kmax[(win * 16u + h) * 64u + ks] = uchar(mx + 1); return; }
  const uint q2 = gid - 40u * 16u * 64u; if (q2 >= 40u * 16u * 2u * 32u) return; const uint win = q2 / 1024u, h = (q2 / 64u) % 16u, s = (q2 / 32u) % 2u, d = q2 % 32u; int mx = -1;
  for (uint k = 0u; k < 32u; ++k) { const int kt = sp_win_token(win, 32u * s + k, p.shift_x, p.shift_y); const uchar b = kt < 0 ? uchar(0) : v_buf[(uint(kt) * 16u + h) * 32u + d]; const int m = sp_int9(b); const int hh = (m + 256) >> 9; const int l = m - 512 * hh; const uint o = (((win * 16u + h) * 2u + s) * 32u + k) * 32u + d; vl[o] = half(l); vh[o] = half(hh); vlh[o] = half(l + hh); mx = max(mx, sp_class(b)); }
  vmax[((win * 16u + h) * 2u + s) * 32u + d] = uchar(mx + 1);
}
struct SpAQ { device const uchar* q; uint win; uint qq; uint h; int sx; int sy; uchar operator()(uint r, uint d) const { const int qt = sp_win_token(win, 16u * qq + r, sx, sy); return qt < 0 ? uchar(0) : q[(uint(qt) * 16u + h) * 32u + d]; } };
struct SpBK { device const uchar* k; uint win; uint h; int sx; int sy; uchar operator()(uint, uint ks, uint d) const { const int kt = sp_win_token(win, ks, sx, sy); return kt < 0 ? uchar(0) : k[(uint(kt) * 16u + h) * 32u + d]; } };
struct SpAProb { threadgroup const uchar* prob; uint base; uchar operator()(uint r, uint j) const { return prob[r * 64u + base + j]; } };
struct SpBV { device const uchar* v; uint win; uint h; int sx; int sy; uchar operator()(uint s, uint d, uint j) const { const int kt = sp_win_token(win, 32u * s + j, sx, sy); return kt < 0 ? uchar(0) : v[(uint(kt) * 16u + h) * 32u + d]; } };
inline half sp_tree8(threadgroup const half* e, uint b0) { const half s01 = e[b0] + e[b0 + 8u], s23 = e[b0 + 16u] + e[b0 + 24u], s45 = e[b0 + 32u] + e[b0 + 40u], s67 = e[b0 + 48u] + e[b0 + 56u]; const half s0123 = s01 + s23; const half s012345 = s0123 + s45; return s012345 + s67; }
kernel void nr_split_attend_tp(device const uchar* q_buf [[buffer(0)]], device const uchar* k_buf [[buffer(1)]], device const uchar* v_buf [[buffer(2)]], device const uchar* w [[buffer(3)]], device uchar* out [[buffer(4)]], device atomic_uint* coverage [[buffer(5)]], device const ushort* rcp_table [[buffer(6)]], constant NrSplitParams& p [[buffer(7)]],
                               device const half* kl [[buffer(8)]], device const half* kh [[buffer(9)]], device const half* klh [[buffer(10)]], device const uchar* kmax [[buffer(11)]],
                               device const half* vl [[buffer(12)]], device const half* vh [[buffer(13)]], device const half* vlh [[buffer(14)]], device const uchar* vmax [[buffer(15)]], device atomic_uint* stats [[buffer(16)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup SpShared sh; threadgroup half eT[SP_ROWS * 64u]; threadgroup uchar probT[SP_ROWS * 64u]; threadgroup half invT[SP_ROWS];
  const uint h = tg.x, win = tg.y / 4u, qq = tg.y % 4u; if (win >= 40u) return; uint repaired = 0u;
  const uint wh = win * 16u + h; device const half* bias = reinterpret_cast<device const half*>(w + 786432u + 8192u * h);
  device const half* hkl = kl + wh * 32u * 64u; device const half* hkh = kh + wh * 32u * 64u; device const half* hklh = klh + wh * 32u * 64u; device const uchar* hkmax = kmax + wh * 64u;
  device const half* hvl = vl + wh * 2u * 32u * 32u; device const half* hvh = vh + wh * 2u * 32u * 32u; device const half* hvlh = vlh + wh * 2u * 32u * 32u; device const uchar* hvmax = vmax + wh * 2u * 32u;
  // scores: C = bias(qs, k) per element
  half sc[32];
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint qs = 16u * qq + rb * 8u + sp_row_of(lane), k = sub * 8u + sp_col_of(lane, e); const uint chunk = 4u * (qs / 16u) + (k / 8u) / 2u, ln = 4u * (qs % 8u) + (k % 8u) / 2u, hf = ((k / 8u) % 2u) * 4u + ((qs % 16u) / 8u) * 2u + k % 2u;
    sc[sp_elem_w<8u>(rb, sub, e)] = bias[chunk * 256u + ln * 8u + hf]; }
  { SpAQ aq{q_buf, win, qq, h, p.shift_x, p.shift_y}; SpBK bk{k_buf, win, h, p.shift_x, p.shift_y}; sp_tile_step_w<8u>(0u, aq, bk, 64u, hkl, hkh, hklh, hkmax, sc, sh, 0u, lane, repaired); }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 8u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint r = rb * 8u + sp_row_of(lane), c = sub * 8u + sp_col_of(lane, e); half vv = fma(sc[sp_elem_w<8u>(rb, sub, e)], nr_hbits(0x29c0u), nr_hbits(0x3d34u)); vv = min(max(vv, nr_hbits(0x3c20u)), nr_hbits(0x3e47u));
    eT[r * 64u + c] = as_type<half>(ushort(((uint(as_type<ushort>(vv)) << 5u) + 0x8000u) & 0xffffu)); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < SP_ROWS) { threadgroup const half* e = eT + lane * 64u; half slo = half(0.0h), shi = half(0.0h);
    for (uint m = 0u; m < 4u; ++m) { const half plo = sp_tree8(e, 2u * m), phi = sp_tree8(e, 2u * m + 1u); slo = (m == 0u) ? plo : half(slo + plo); shi = (m == 0u) ? phi : half(shi + phi); }
    const half inv = nr_table_lookup(rcp_table, max(half(slo + shi), nr_hbits(0x0410u))); invT[lane] = inv;
    for (uint k = 0u; k < 64u; ++k) probT[lane * 64u + k] = nr_to_e4m3(half(e[k] * inv)); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  half accd[16]; for (uint i = 0u; i < 16u; ++i) accd[i] = half(0.0h);
  { SpBV bv{v_buf, win, h, p.shift_x, p.shift_y}; for (uint st = 0u; st < 2u; ++st) { SpAProb ap{probT, 32u * st}; sp_tile_step_w<4u>(st, ap, bv, 32u, hvl, hvh, hvlh, hvmax, accd, sh, 0u, lane, repaired); } }
  for (uint rb = 0u; rb < 2u; ++rb) for (uint sub = 0u; sub < 4u; ++sub) for (uint e = 0u; e < 2u; ++e) {
    const uint r = rb * 8u + sp_row_of(lane), d = sub * 8u + sp_col_of(lane, e); const int qt = sp_win_token(win, 16u * qq + r, p.shift_x, p.shift_y); if (qt < 0) continue;
    const uint idx = nr_tin(uint(qt), 32u * h + nr_cfd(d)); out[idx] = nr_to_e4m3(accd[sp_elem_w<4u>(rb, sub, e)]); atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed); }
  const uint tot = simd_sum(repaired); if (lane == 0u) atomic_fetch_add_explicit(&stats[0], tot, memory_order_relaxed);
}
