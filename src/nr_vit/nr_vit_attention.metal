#include "nr_vit_common.metal"

// ---------------------------------------------------------------------------------------------
// QKV projection with per-head QK normalisation. One thread per (token, head): 96 columns
// (q 0-31, k 32-63, v 64-95) of the [32 steps][48 groups][2048] weight blob, K split in two halves
// summed in half (t = p0 + p1), sum-of-squares tree with the ptxas mul+add fusion on the first level,
// RTX rsqrt table, q = ((t*inv) * h(sqrt 32)) * h(head_scale), k = t*inv, v = t.
// ---------------------------------------------------------------------------------------------
inline half nr_sumsq32(thread const half* t) {   // t: the head's 32 values in column order
  half P[8];
  for (uint c = 0u; c < 8u; ++c) {
    const half a = fma(t[8u + c], t[8u + c], half(t[24u + c] * t[24u + c]));   // fused first square
    const half b = fma(t[c], t[c], half(t[16u + c] * t[16u + c]));
    P[c] = a + b;
  }
  half R1[8], R2[8];
  for (uint c = 0u; c < 8u; ++c) R1[c] = P[c] + P[c ^ 4u];
  for (uint c = 0u; c < 8u; ++c) R2[c] = R1[c] + R1[c ^ 2u];
  return max(half(R2[0] + R2[1]), nr_hbits(0x0410u));
}

kernel void nr_vit_qkv(device const uchar* tokens [[buffer(0)]],
                       device const uchar* weights [[buffer(1)]],       // 128-byte f32 head scales, then the blob
                       device uchar* q_out [[buffer(2)]],
                       device uchar* k_out [[buffer(3)]],
                       device uchar* v_out [[buffer(4)]],
                       device atomic_uint* coverage_q [[buffer(5)]],
                       device atomic_uint* coverage_k [[buffer(6)]],
                       device atomic_uint* coverage_v [[buffer(7)]],
                       device const ushort* rsqrt_table [[buffer(8)]],
                       constant NrParams& p [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 32u, h = gid % 32u;
  if (t >= p.tokens) return;
  device const uchar* blob = weights + 128u;
  const half head_scale = half(reinterpret_cast<device const float*>(weights)[h]);
  half tv[96];
  half a[32], b[32];
  half p0[96];
  for (uint z = 0u; z < 2u; ++z) {
    half part[96];
    for (uint i = 0u; i < 96u; ++i) part[i] = half(0.0h);
    for (uint s = 16u * z; s < 16u * z + 16u; ++s) {
      for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(tokens[nr_a_index(t, 32u * s + j, 1024u)]);
      for (uint i = 0u; i < 96u; ++i) {
        const uint n = 96u * h + i;
        device const uchar* w = blob + s * 98304u + (n / 64u) * 2048u;
        for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w[nr_weight_index(j, n % 64u)]);
        part[i] = nr_qmma_step(a, b, part[i]);
      }
    }
    for (uint i = 0u; i < 96u; ++i) { if (z == 0u) p0[i] = part[i]; else tv[i] = half(p0[i] + part[i]); }
  }
  const half inv_q = nr_table_lookup(rsqrt_table, nr_sumsq32(tv));
  const half inv_k = nr_table_lookup(rsqrt_table, nr_sumsq32(tv + 32));
  const half c32 = nr_hbits(0x45a8u);      // h(sqrt.approx.f32(32)) = 5.65625
  for (uint d = 0u; d < 32u; ++d) {
    const uint qi = nr_d_index(t, 32u * h + d, 1024u), ki = nr_k_index(t, d, h), vi = nr_v_index(t, d, h);
    q_out[qi] = nr_to_e4m3(half(half(half(tv[d] * inv_q) * c32) * head_scale));
    k_out[ki] = nr_to_e4m3(half(tv[32u + d] * inv_k));
    v_out[vi] = nr_to_e4m3(tv[64u + d]);
    atomic_fetch_add_explicit(&coverage_q[qi], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&coverage_k[ki], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&coverage_v[vi], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// Full 640-key attention per head. One thread per (query, head).
//   score  = QMMA step over the 32 head dims in k-slot order (C = 0)
//   e      = bit trick: v = clamp(fma(score, 0.0895, 1.7094), 1.4395, 1.9775); e = as_half((bits(v) << 4) + 0x4000)
//   prob   = e4m3(e)      (A operand of the P·V steps)
//   rowsum = per 64-key block: quad partials tree8 over the 8 key n-tiles for even/odd key slots,
//            ((P0 + P1) + P2) + P3, then even + odd; blocks accumulated in order from 0
//   inv    = RTX rcp table of max(rowsum, 6.2e-5)
//   out    = e4m3(half(PV) * inv), PV = 20 chained QMMA steps over keys in slot order
// ---------------------------------------------------------------------------------------------
inline half nr_tree8_exact(half x0, half x1, half x2, half x3, half x4, half x5, half x6, half x7) {
  const half s01 = x0 + x1, s23 = x2 + x3, s45 = x4 + x5, s67 = x6 + x7;
  const half s0123 = s01 + s23;
  const half s012345 = s0123 + s45;
  return s012345 + s67;
}

kernel void nr_vit_attention(device const uchar* q_buf [[buffer(0)]],
                             device const uchar* k_buf [[buffer(1)]],
                             device const uchar* v_buf [[buffer(2)]],
                             device uchar* out [[buffer(3)]],
                             device atomic_uint* coverage [[buffer(4)]],
                             device const ushort* rcp_table [[buffer(5)]],
                             constant NrParams& p [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
  const uint qrow = gid / 32u, h = gid % 32u;
  if (qrow >= p.tokens) return;
  const uint keys = p.tokens;               // 640: ten blocks of 64, no padding
  half qa[32];
  for (uint j = 0u; j < 32u; ++j) qa[j] = nr_e4m3(q_buf[nr_d_index(qrow, 32u * h + nr_slot_channel(j), 1024u)]);
  // scores -> probabilities per 64-key block, row sum accumulated block by block
  uchar prob[640];
  half kb[32], e[64];
  half total = nr_hbits(0u);
  for (uint blk = 0u; blk < keys / 64u; ++blk) {
    for (uint i = 0u; i < 64u; ++i) {
      const uint key = 64u * blk + i;
      for (uint j = 0u; j < 32u; ++j) kb[j] = nr_e4m3(k_buf[nr_k_index(key, nr_slot_channel(j), h)]);
      const half score = nr_qmma_step(qa, kb, nr_hbits(0u));
      half vv = fma(score, nr_hbits(0x2dbbu), nr_hbits(0x3ed6u));                       // 0.08953857421875 * s + 1.708984375
      vv = min(max(vv, nr_hbits(0x3dc2u)), nr_hbits(0x3fe9u));                          // clamp [1.439453125, 1.9775390625]
      e[i] = as_type<half>(ushort((uint(as_type<ushort>(vv)) << 4u) + 0x4000u));
      prob[key] = nr_to_e4m3(e[i]);
    }
    half slo = nr_hbits(0u), shi = nr_hbits(0u);
    for (uint qd = 0u; qd < 4u; ++qd) {
      const uint b0 = 2u * qd;
      const half plo = nr_tree8_exact(e[b0], e[b0 + 8u], e[b0 + 16u], e[b0 + 24u], e[b0 + 32u], e[b0 + 40u], e[b0 + 48u], e[b0 + 56u]);
      const half phi = nr_tree8_exact(e[b0 + 1u], e[b0 + 9u], e[b0 + 17u], e[b0 + 25u], e[b0 + 33u], e[b0 + 41u], e[b0 + 49u], e[b0 + 57u]);
      slo = (qd == 0u) ? plo : half(slo + plo);
      shi = (qd == 0u) ? phi : half(shi + phi);
    }
    total = half(total + half(slo + shi));
  }
  const half inv = nr_table_lookup(rcp_table, max(total, nr_hbits(0x0410u)));
  half pa[32], vb[32];
  half accd[32];
  for (uint d = 0u; d < 32u; ++d) accd[d] = half(0.0h);
  for (uint s = 0u; s < keys / 32u; ++s) {
    for (uint j = 0u; j < 32u; ++j) pa[j] = nr_e4m3(prob[32u * s + nr_slot_channel(j)]);
    for (uint d = 0u; d < 32u; ++d) {
      for (uint j = 0u; j < 32u; ++j) vb[j] = nr_e4m3(v_buf[nr_v_index(32u * s + nr_slot_channel(j), d, h)]);
      accd[d] = nr_qmma_step(pa, vb, accd[d]);
    }
  }
  for (uint d = 0u; d < 32u; ++d) {
    const uint idx = nr_d_index(qrow, 32u * h + d, 1024u);
    out[idx] = nr_to_e4m3(half(accd[d] * inv));
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}
