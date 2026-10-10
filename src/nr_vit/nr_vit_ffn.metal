#include "nr_vit_common.metal"

// ---------------------------------------------------------------------------------------------
// FFN expand: 1024 -> 4096, cubic-SiLU activation, E4M3 out in the D-fragment layout.
// One thread per (token, 16 output channels). K = 32 steps; per step the token's 32 channels are
// read in k-slot order (identity for the A-identity token layout) and the 16 weight columns' slots.
// ---------------------------------------------------------------------------------------------
kernel void nr_vit_expand(device const uchar* tokens [[buffer(0)]],
                          device const uchar* weights [[buffer(1)]],
                          device uchar* out [[buffer(2)]],
                          device atomic_uint* coverage [[buffer(3)]],
                          constant NrParams& p [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 256u, g = gid % 256u;
  if (t >= p.tokens) return;
  half acc[16];
  for (uint i = 0u; i < 16u; ++i) acc[i] = half(0.0h);
  half a[32], b[32];
  for (uint s = 0u; s < 32u; ++s) {
    for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(tokens[nr_a_index(t, 32u * s + j, 1024u)]);
    for (uint i = 0u; i < 16u; ++i) {
      const uint c = 16u * g + i;
      device const uchar* w = weights + s * 131072u + (c / 64u) * 2048u;
      for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w[nr_weight_index(j, c % 64u)]);
      acc[i] = nr_qmma_step(a, b, acc[i]);
    }
  }
  for (uint i = 0u; i < 16u; ++i) {
    const uint c = 16u * g + i;
    const half h = acc[i];
    const half tt = min(max(h, nr_hbits(0xc400u)), nr_hbits(0x4400u));            // clamp to [-4, 4]
    const half u = fma(nr_hbits(0xab28u), abs(tt), nr_hbits(0x3728u));              // -0.055908203125 * |t| + 0.447265625
    const half sv = fma(tt, u, nr_hbits(0x3b28u));                                  // t * u + 0.89453125
    const uint idx = nr_d_index(t, c, 4096u);
    out[idx] = nr_to_e4m3(h * sv);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// FFN contract: 4096 -> 1024 with the residual folded into the first K quarter's C operand
// (C0 = half(res[cfd(n)] * scale[n])), four K quarters chained in half: ((p0 + p1) + p2) + p3,
// output E4M3 at token channel cfd(n) in the A-identity layout.
// One thread per (token, 16 columns).
// ---------------------------------------------------------------------------------------------
kernel void nr_vit_contract(device const uchar* expanded [[buffer(0)]],
                            device const uchar* residual [[buffer(1)]],
                            device const uchar* weights [[buffer(2)]],
                            device uchar* out [[buffer(3)]],
                            device atomic_uint* coverage [[buffer(4)]],
                            constant NrParams& p [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 64u, g = gid % 64u;
  if (t >= p.tokens) return;
  device const half* scale = reinterpret_cast<device const half*>(weights + 4194304u);
  half total[16];
  half a[32], b[32];
  for (uint z = 0u; z < 4u; ++z) {
    half part[16];
    for (uint i = 0u; i < 16u; ++i) {
      const uint n = 16u * g + i;
      part[i] = (z == 0u) ? half(nr_e4m3(residual[nr_a_index(t, nr_cfd(n), 1024u)]) * scale[n]) : half(0.0h);
    }
    for (uint s = 32u * z; s < 32u * z + 32u; ++s) {
      // the expanded tensor is in the D layout: slot j of step s holds channel 32*s + nr_slot_channel(j)
      for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(expanded[nr_d_index(t, 32u * s + nr_slot_channel(j), 4096u)]);
      for (uint i = 0u; i < 16u; ++i) {
        const uint n = 16u * g + i;
        device const uchar* w = weights + s * 32768u + (n / 64u) * 2048u;
        for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w[nr_weight_index(j, n % 64u)]);
        part[i] = nr_qmma_step(a, b, part[i]);
      }
    }
    for (uint i = 0u; i < 16u; ++i) total[i] = (z == 0u) ? part[i] : half(total[i] + part[i]);
  }
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i, idx = nr_a_index(t, nr_cfd(n), 1024u);
    out[idx] = nr_to_e4m3(total[i]);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// Attention output projection: 1024 -> 1024, residual = the contract output, four K quarters of 8
// steps, same chaining and swizzles as the contract. Input A is the attention output (D layout).
// ---------------------------------------------------------------------------------------------
kernel void nr_vit_projection(device const uchar* attended [[buffer(0)]],
                              device const uchar* residual [[buffer(1)]],
                              device const uchar* weights [[buffer(2)]],
                              device uchar* out [[buffer(3)]],
                              device atomic_uint* coverage [[buffer(4)]],
                              constant NrParams& p [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
  const uint t = gid / 64u, g = gid % 64u;
  if (t >= p.tokens) return;
  device const half* scale = reinterpret_cast<device const half*>(weights + 1048576u);
  half total[16];
  half a[32], b[32];
  for (uint z = 0u; z < 4u; ++z) {
    half part[16];
    for (uint i = 0u; i < 16u; ++i) {
      const uint n = 16u * g + i;
      part[i] = (z == 0u) ? half(nr_e4m3(residual[nr_a_index(t, nr_cfd(n), 1024u)]) * scale[n]) : half(0.0h);
    }
    for (uint s = 8u * z; s < 8u * z + 8u; ++s) {
      for (uint j = 0u; j < 32u; ++j) a[j] = nr_e4m3(attended[nr_d_index(t, 32u * s + nr_slot_channel(j), 1024u)]);
      for (uint i = 0u; i < 16u; ++i) {
        const uint n = 16u * g + i;
        device const uchar* w = weights + s * 32768u + (n / 64u) * 2048u;
        for (uint j = 0u; j < 32u; ++j) b[j] = nr_e4m3(w[nr_weight_index(j, n % 64u)]);
        part[i] = nr_qmma_step(a, b, part[i]);
      }
    }
    for (uint i = 0u; i < 16u; ++i) total[i] = (z == 0u) ? part[i] : half(total[i] + part[i]);
  }
  for (uint i = 0u; i < 16u; ++i) {
    const uint n = 16u * g + i, idx = nr_a_index(t, nr_cfd(n), 1024u);
    out[idx] = nr_to_e4m3(total[i]);
    atomic_fetch_add_explicit(&coverage[idx], 1u, memory_order_relaxed);
  }
}
