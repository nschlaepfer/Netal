#include "nr_split_common.metal"

// ---------------------------------------------------------------------------------------------
// Split FFN (cc_split_swin_16h_ffwd_*): X(512) -> H1 = X W1 (512 columns, 16 steps) ; per 64-column
// group g: H2_g = e4m3(H1_g) W2_g (256, 2 steps) -> a = e4m3(cubic_silu(H2_g)) -> OUT_g = a W3_g
// (64, 8 steps). Blob: W1 [16][8 x 2048] ; W2_g at 262144 + 16384 g as [2 steps][8 x 1024 (32 cols)] ;
// W3_g at 393216 + 16384 g as [8 steps][1 x 2048]. Output column n lands on channel nr_cfd(n).
// One thread per (token, group). H2 is produced 32 columns at a time (one OUT k-step each).
// ---------------------------------------------------------------------------------------------
kernel void nr_split_ffwd(device const uchar* x [[buffer(0)]],
                          device const uchar* w [[buffer(1)]],
                          device uchar* out [[buffer(2)]],
                          device atomic_uint* coverage [[buffer(3)]],
                          constant NrSplitParams& p [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 8u, g = gid % 8u;
  if (t >= p.tokens) return;
  const bool view = (p.flags & NR_FLAG_IN_VIEW) != 0u;
  half a[32], b[32];
  half h1[64];
  for (uint i = 0u; i < 64u; ++i) h1[i] = half(0.0h);
  for (uint s = 0u; s < 16u; ++s) {
    for (uint j = 0u; j < 32u; ++j) a[j] = nr_load_token(x, t, 32u * s + j, view);
    for (uint i = 0u; i < 64u; ++i) {
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_wload(w, s, 16384u, 64u * g + i, j);
      h1[i] = nr_qmma_step(a, b, h1[i]);
    }
  }
  for (uint i = 0u; i < 64u; ++i) h1[i] = nr_round_e4m3(h1[i]);
  device const uchar* w2 = w + 262144u + 16384u * g;
  device const uchar* w3 = w + 393216u + 16384u * g;
  half acc[64];
  for (uint i = 0u; i < 64u; ++i) acc[i] = half(0.0h);
  half a1[32], a2[32];
  for (uint j = 0u; j < 32u; ++j) { a1[j] = h1[nr_slot_channel(j)]; a2[j] = h1[32u + nr_slot_channel(j)]; }
  for (uint s3 = 0u; s3 < 8u; ++s3) {
    half h2[32];
    for (uint i = 0u; i < 32u; ++i) {
      const uint n2 = 32u * s3 + i;
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w2[(n2 / 32u) * 1024u + nr_weight_index(j, n2 % 32u)]);
      half v = nr_qmma_step(a1, b, half(0.0h));
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w2[8192u + (n2 / 32u) * 1024u + nr_weight_index(j, n2 % 32u)]);
      h2[i] = nr_qmma_step(a2, b, v);
    }
    for (uint j = 0u; j < 32u; ++j) a[j] = nr_round_e4m3(nr_cubic_silu(h2[nr_slot_channel(j)]));
    for (uint i = 0u; i < 64u; ++i) {
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w3[s3 * 2048u + nr_weight_index(j, i)]);
      acc[i] = nr_qmma_step(a, b, acc[i]);
    }
  }
  for (uint i = 0u; i < 64u; ++i) {
    const uint idx = nr_tin(t, nr_cfd(64u * g + i));
    out[idx] = nr_to_e4m3(acc[i]);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// Residual-folded 1x1 conv 512 -> 512 (ffwd_proj, proj, proj_pool, proj_outview):
// Y[n] = e4m3(chain_16(W[:, n] . X, C0 = half(res[cfd(n)] * scale[n]))) stored at channel cfd(n).
// Blob: [16 steps][8 x 2048] then 512 half scales. Flags select the residual source (tin | view), the
// output target (tin | view) and an optional pre-rounding half copy Y16[token*512 + n] for the pool.
// One thread per (token, 16 columns).
// ---------------------------------------------------------------------------------------------
kernel void nr_split_conv(device const uchar* x [[buffer(0)]],
                          device const uchar* res [[buffer(1)]],
                          device const uchar* w [[buffer(2)]],
                          device uchar* out [[buffer(3)]],
                          device half* y16 [[buffer(4)]],
                          device atomic_uint* coverage [[buffer(5)]],
                          constant NrSplitParams& p [[buffer(6)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 32u, g = gid % 32u;
  if (t >= p.tokens) return;
  const bool res_view = (p.flags & NR_FLAG_RES_VIEW) != 0u, out_view = (p.flags & NR_FLAG_OUT_VIEW) != 0u, emit = (p.flags & NR_FLAG_EMIT_F16) != 0u;
  device const half* scale = reinterpret_cast<device const half*>(w + 262144u);
  half acc[16], a[32], b[32];
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i;
    acc[i] = half(nr_load_token(res, t, nr_cfd(n), res_view) * scale[n]);
  }
  for (uint s = 0u; s < 16u; ++s) {
    for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(x[nr_tin(t, 32u * s + j)]);
    for (uint i = 0u; i < 16u; ++i) {
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_wload(w, s, 16384u, 16u * g + i, j);
      acc[i] = nr_qmma_step(a, b, acc[i]);
    }
  }
  uint r = 0u, c = 0u; nr_token_rc(t, r, c);
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i, ch = nr_cfd(n);
    const uint idx = out_view ? nr_view_index(r, c, ch) : nr_tin(t, ch);
    out[idx] = nr_to_e4m3(acc[i]);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
    if (emit) y16[t * 512u + n] = acc[i];
  }
}

// ---------------------------------------------------------------------------------------------
// 2x2 average pool of the proj_pool conv (pre-rounding halves): pooled(R, C) =
// e4m3(half(half(half(a + b) + half(c + d)) * 0.25)) with a = (2R, 2C), b = (2R, 2C+1), c = (2R+1, 2C),
// d = (2R+1, 2C+1); tokens outside the 60x36 grid contribute zero. Every token of the padded 32x20
// pooled grid is written (the recorded launch writes them all). One thread per (pooled token, 16 cols).
// ---------------------------------------------------------------------------------------------
kernel void nr_split_pool(device const half* y16 [[buffer(0)]],
                          device uchar* out [[buffer(1)]],
                          device atomic_uint* coverage [[buffer(2)]],
                          constant NrSplitParams& p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint pt = gid / 32u, g = gid % 32u;
  if (pt >= NR_POOL_TOKENS) return;
  uint R = 0u, C = 0u; nr_pool_token_rc(pt, R, C);
  const bool inside = (2u * R + 1u < NR_SPLIT_H) && (2u * C + 1u < NR_SPLIT_W);
  const uint ta = inside ? nr_token_of(2u * R, 2u * C) : 0u, tb = inside ? nr_token_of(2u * R, 2u * C + 1u) : 0u;
  const uint tc = inside ? nr_token_of(2u * R + 1u, 2u * C) : 0u, td = inside ? nr_token_of(2u * R + 1u, 2u * C + 1u) : 0u;
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i;
    half v = half(0.0h);
    if (inside) {
      const half s = half(half(y16[ta * 512u + n] + y16[tb * 512u + n]) + half(y16[tc * 512u + n] + y16[td * 512u + n]));
      v = half(s * nr_hbits(0x3400u));
    }
    const uint idx = nr_tin(pt, nr_cfd(n));
    out[idx] = nr_to_e4m3(v);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// QKV with RMS normalisation of q and k over the 32 head dims (cc_split_swin_16h_qkv, first phase).
// Blob: [16 steps][24 groups x 2048] (head h: columns 96h + [q 0-31 | k 32-63 | v 64-95]); relative
// position bias at 786432 (16 x 8192 bytes); f32 head scales at 917504. q = e4m3(half(half(t*inv_q) * hs)),
// k = e4m3(half(t*inv_k)), v = e4m3(t); the sum-of-squares tree is the ViT one (nr_sumsq32).
// Scratch layout: [token][head][32] bytes for q, k and v. One thread per (token, head).
// ---------------------------------------------------------------------------------------------
inline half nr_split_sumsq32(thread const half* t) {
  half P[8];
  for (uint c = 0u; c < 8u; ++c) {
    const half a = fma(t[c], t[c], half(t[16u + c] * t[16u + c]));            // fused first square: (nt0, nt2)
    const half b = fma(t[8u + c], t[8u + c], half(t[24u + c] * t[24u + c]));  // (nt1, nt3)
    P[c] = b + a;
  }
  half R1[8], R2[8];
  for (uint c = 0u; c < 8u; ++c) R1[c] = P[c] + P[c ^ 4u];
  for (uint c = 0u; c < 8u; ++c) R2[c] = R1[c] + R1[c ^ 2u];
  return max(half(R2[0] + R2[1]), nr_hbits(0x0410u));
}
kernel void nr_split_qkv(device const uchar* x [[buffer(0)]],
                         device const uchar* w [[buffer(1)]],
                         device uchar* q_out [[buffer(2)]],
                         device uchar* k_out [[buffer(3)]],
                         device uchar* v_out [[buffer(4)]],
                         device const ushort* rsqrt_table [[buffer(5)]],
                         constant NrSplitParams& p [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 16u, h = gid % 16u;
  if (t >= p.tokens) return;
  half acc[96], a[32], b[32];
  for (uint i = 0u; i < 96u; ++i) acc[i] = half(0.0h);
  for (uint s = 0u; s < 16u; ++s) {
    for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(x[nr_tin(t, 32u * s + j)]);
    for (uint i = 0u; i < 96u; ++i) {
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_wload(w, s, 49152u, 96u * h + i, j);
      acc[i] = nr_qmma_step(a, b, acc[i]);
    }
  }
  const half inv_q = nr_table_lookup(rsqrt_table, nr_split_sumsq32(acc));
  const half inv_k = nr_table_lookup(rsqrt_table, nr_split_sumsq32(acc + 32));
  const half hs = half(reinterpret_cast<device const float*>(w + 917504u)[h]);
  const uint base = (t * 16u + h) * 32u;
  for (uint d = 0u; d < 32u; ++d) {
    q_out[base + d] = nr_to_e4m3(half(half(acc[d] * inv_q) * hs));
    k_out[base + d] = nr_to_e4m3(half(acc[32u + d] * inv_k));
    v_out[base + d] = nr_to_e4m3(acc[64u + d]);
  }
}

// ---------------------------------------------------------------------------------------------
// Windowed attention (cc_split_swin_16h_qkv, second phase). Window (wx, wy) with shift (sx, sy) covers
// tokens rows 8wy+sy.., cols 8wx+sx.. as four 4x4 tiles in m-tile order [(ty0,tx0),(ty0,tx1),(ty1,tx0),
// (ty1,tx1)]; tiles outside the grid hold zero q/k/v (their keys still take e(bias) in the row sum,
// their queries are not stored).
//   score  = fp32 chain over the 32 dims in slot order starting from the half bias, rounded to half
//   v      = clamp(fma(score, 0x29c0, 0x3d34), 0x3c20, 0x3e47) ; e = as_half((bits(v) << 5) + 0x8000)
//   rowsum = per lane m = (k%8)/2 and parity k%2: tree8 over the 8 key n-tiles, ((L0+L1)+L2)+L3, lo+hi
//   inv    = RTX rcp table of max(rowsum, 0x0410) ; P = e4m3(half(e * inv))
//   out    = e4m3(chain over two 32-key steps in slot order of P . v) at channel 32h + cfd(d)
// Bias table per head (8192 bytes): chunk = 4*(q/16) + (k/8)/2, lane = 4*(q%8) + (k%8)/2,
// half = ((k/8)%2)*4 + ((q%16)/8)*2 + k%2.  One thread per (window, head, query slot).
// ---------------------------------------------------------------------------------------------
kernel void nr_split_attend(device const uchar* q_buf [[buffer(0)]],
                            device const uchar* k_buf [[buffer(1)]],
                            device const uchar* v_buf [[buffer(2)]],
                            device const uchar* w [[buffer(3)]],
                            device uchar* out [[buffer(4)]],
                            device atomic_uint* coverage [[buffer(5)]],
                            device const ushort* rcp_table [[buffer(6)]],
                            constant NrSplitParams& p [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint qs = gid % 64u, h = (gid / 64u) % 16u, win = gid / 1024u, wx = win % 8u, wy = win / 8u;
  if (wy >= 5u) return;
  const int ox = int(8u * wx) + p.shift_x, oy = int(8u * wy) + p.shift_y;
  // window tokens: slot i = 16*mt + g ; mt = 2*ty + tx
  int tok[64];
  for (uint i = 0u; i < 64u; ++i) {
    const uint mt = i / 16u, g = i % 16u;
    const int tx = (ox >> 2) + int(mt % 2u), ty = (oy >> 2) + int(mt / 2u);
    tok[i] = (tx < 0 || ty < 0 || tx >= int(NR_SPLIT_TILES_X) || ty >= int(NR_SPLIT_TILES_Y)) ? -1 : int((uint(ty) * NR_SPLIT_TILES_X + uint(tx)) * 16u + g);
  }
  if (tok[qs] < 0) return;                                  // out-of-bounds query: nothing stored
  const uint qt = uint(tok[qs]);
  device const half* bias = reinterpret_cast<device const half*>(w + 786432u + 8192u * h);
  half qa[32];
  for (uint j = 0u; j < 32u; ++j) qa[j] = nr_e4m3(q_buf[(qt * 16u + h) * 32u + nr_slot_channel(j)]);
  half e[64], kb[32];
  for (uint k = 0u; k < 64u; ++k) {
    const uint chunk = 4u * (qs / 16u) + (k / 8u) / 2u, lane = 4u * (qs % 8u) + (k % 8u) / 2u, hf = ((k / 8u) % 2u) * 4u + ((qs % 16u) / 8u) * 2u + k % 2u;
    const half c0 = bias[chunk * 256u + lane * 8u + hf];
    half score;
    if (tok[k] < 0) score = c0;
    else {
      const uint kt = uint(tok[k]);
      for (uint j = 0u; j < 32u; ++j) kb[j] = nr_e4m3(k_buf[(kt * 16u + h) * 32u + nr_slot_channel(j)]);
      score = nr_qmma_step(qa, kb, c0);
    }
    half vv = fma(score, nr_hbits(0x29c0u), nr_hbits(0x3d34u));
    vv = min(max(vv, nr_hbits(0x3c20u)), nr_hbits(0x3e47u));
    e[k] = as_type<half>(ushort(((uint(as_type<ushort>(vv)) << 5u) + 0x8000u) & 0xffffu));
  }
  half qsum_lo = half(0.0h), qsum_hi = half(0.0h);
  for (uint m = 0u; m < 4u; ++m) {
    const uint b0 = 2u * m;
    const half plo = nr_tree8_exact(e[b0], e[b0 + 8u], e[b0 + 16u], e[b0 + 24u], e[b0 + 32u], e[b0 + 40u], e[b0 + 48u], e[b0 + 56u]);
    const half phi = nr_tree8_exact(e[b0 + 1u], e[b0 + 9u], e[b0 + 17u], e[b0 + 25u], e[b0 + 33u], e[b0 + 41u], e[b0 + 49u], e[b0 + 57u]);
    qsum_lo = (m == 0u) ? plo : half(qsum_lo + plo);
    qsum_hi = (m == 0u) ? phi : half(qsum_hi + phi);
  }
  const half inv = nr_table_lookup(rcp_table, max(half(qsum_lo + qsum_hi), nr_hbits(0x0410u)));
  half prob[64];
  for (uint k = 0u; k < 64u; ++k) prob[k] = nr_round_e4m3(half(e[k] * inv));
  half pa[32], vb[32], accd[32];
  for (uint d = 0u; d < 32u; ++d) accd[d] = half(0.0h);
  for (uint s = 0u; s < 2u; ++s) {
    for (uint j = 0u; j < 32u; ++j) pa[j] = prob[32u * s + nr_slot_channel(j)];
    for (uint d = 0u; d < 32u; ++d) {
      for (uint j = 0u; j < 32u; ++j) {
        const int kt = tok[32u * s + nr_slot_channel(j)];
        vb[j] = (kt < 0) ? half(0.0h) : nr_e4m3(v_buf[(uint(kt) * 16u + h) * 32u + d]);
      }
      accd[d] = nr_qmma_step(pa, vb, accd[d]);
    }
  }
  for (uint d = 0u; d < 32u; ++d) {
    const uint idx = nr_tin(qt, 32u * h + nr_cfd(d));
    out[idx] = nr_to_e4m3(accd[d]);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// Final head (cc_split_swin_16h_final_head_512_wait): pooled tokens 512 -> 1024, plain 16-step chain
// from zero, output in the ViT 2D tin layout (16384-byte tiles, 8 per row) at channel cfd(n); only
// the 30 x 18 valid pooled tokens are written. Blob: [16 steps][16 x 2048] (+16 unused bytes).
// One thread per (pooled token, 16 columns).
// ---------------------------------------------------------------------------------------------
kernel void nr_split_final_head(device const uchar* pooled [[buffer(0)]],
                                device const uchar* w [[buffer(1)]],
                                device uchar* out [[buffer(2)]],
                                device atomic_uint* coverage [[buffer(3)]],
                                constant NrSplitParams& p [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
  const uint pt = gid / 64u, g = gid % 64u;
  if (pt >= NR_POOL_TOKENS) return;
  uint R = 0u, C = 0u; nr_pool_token_rc(pt, R, C);
  if (R >= NR_POOL_H || C >= NR_POOL_W) return;
  half acc[16], a[32], b[32];
  for (uint i = 0u; i < 16u; ++i) acc[i] = half(0.0h);
  for (uint s = 0u; s < 16u; ++s) {
    for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(pooled[nr_tin(pt, 32u * s + j)]);
    for (uint i = 0u; i < 16u; ++i) {
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_wload(w, s, 32768u, 16u * g + i, j);
      acc[i] = nr_qmma_step(a, b, acc[i]);
    }
  }
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i, idx = nr_a_index(pt, nr_cfd(n), 1024u);
    out[idx] = nr_to_e4m3(acc[i]);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// Decoder input upsample (cc_dec_input_upsample_1024_512_tilesync): nearest 2x upsample of the ViT
// output (1024 channels, 16384-byte tiles, 8 per row), conv 1024 -> 512 in four K quarters of 8
// steps each rounded to half and summed ((p0 + p1) + p2) + p3, then out = e4m3(fma(res[cfd(n)],
// scale[n], total)) at channel cfd(n). Blob: [32 steps][8 x 2048] then 512 half scales.
// One thread per (token, 16 columns).
// ---------------------------------------------------------------------------------------------
kernel void nr_split_dec_upsample(device const uchar* vit_out [[buffer(0)]],
                                  device const uchar* res [[buffer(1)]],
                                  device const uchar* w [[buffer(2)]],
                                  device uchar* out [[buffer(3)]],
                                  device atomic_uint* coverage [[buffer(4)]],
                                  constant NrSplitParams& p [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 32u, g = gid % 32u;
  if (t >= p.tokens) return;
  uint r = 0u, c = 0u; nr_token_rc(t, r, c);
  const uint lo = nr_pool_token_of(r / 2u, c / 2u);
  device const half* scale = reinterpret_cast<device const half*>(w + 524288u);
  half total[16], a[32], b[32];
  for (uint z = 0u; z < 4u; ++z) {
    half part[16];
    for (uint i = 0u; i < 16u; ++i) part[i] = half(0.0h);
    for (uint s = 8u * z; s < 8u * z + 8u; ++s) {
      for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(vit_out[nr_a_index(lo, 32u * s + j, 1024u)]);
      for (uint i = 0u; i < 16u; ++i) {
        for (uint j = 0u; j < 32u; ++j) b[j] = nr_wload(w, s, 16384u, 16u * g + i, j);
        part[i] = nr_qmma_step(a, b, part[i]);
      }
    }
    for (uint i = 0u; i < 16u; ++i) total[i] = (z == 0u) ? part[i] : half(total[i] + part[i]);
  }
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i, idx = nr_tin(t, nr_cfd(n));
    const half rv = nr_e4m3(res[idx]);
    out[idx] = nr_to_e4m3(fma(rv, scale[n], total[i]));
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}
