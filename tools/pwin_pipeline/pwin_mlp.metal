// Fused MLP for the PWin pipeline (enc0 chains 59-125), per 16-token tile, hidden never leaves
// threadgroup memory:
//   a  = fp16( u * s )                                (per-channel scale, 32 ch)
//   for group g in 0..3:  h_g = act( fp16( be_g + a . We_g ) )        (16 x 32 hidden of the group; expand
//         split quirk: hidden tile 0 of group 0 rounds after the first 16 K for window blocks 0..2)
//         z = fp16( C + h_g . Wc_g ),  C = fp16(bc + u) for g = 0, else the previous z (fp16 handoff)
//   out = z
// fp32 accumulation in hw3 order; act = GELU approximation of the transpiled kernel.
#include <metal_stdlib>
using namespace metal;

struct MlpParams {
  uint tokens, width, shift_y;
  uint u_stride, out_stride;
  uint pad0, pad1, pad2;
};

static inline half mlp_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

static inline void load_ab(thread simdgroup_float8x8 (&m)[2][2], threadgroup const half* src, uint stride) {
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i)
  #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
    simdgroup_half8x8 hm;
    simdgroup_load(hm, src + i * 8u * stride + j * 8u, stride);
    m[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
    m[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
  }
}
static inline void load_b_device(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i)
  #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
    simdgroup_half8x8 hm;
    simdgroup_load(hm, src + i * 8u * stride + j * 8u, stride);
    m[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
    m[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
  }
}
static inline void mma16(thread simdgroup_float8x8 (&acc)[2][2], thread simdgroup_float8x8 (&a)[2][2], thread simdgroup_float8x8 (&b)[2][2]) {
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i)
  #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
    simdgroup_multiply_accumulate(acc[i][j], a[i][0], b[0][j], acc[i][j]);
    simdgroup_multiply_accumulate(acc[i][j], a[i][1], b[1][j], acc[i][j]);
  }
}

kernel void pwin_mlp_tile(device const half* U [[buffer(0)]],        // [tokens x 32] input u (residual source)
                          device const half* S [[buffer(1)]],        // 32 scales
                          device const half* We [[buffer(2)]],       // [32 x 128] expand weights (row-major)
                          device const half* Be [[buffer(3)]],       // 128 expand biases
                          device const half* Wc [[buffer(4)]],       // [128 x 32] contract weights (row-major)
                          device const half* Bc [[buffer(5)]],       // 32 contract biases (group 0 residual bias)
                          device half* Out [[buffer(6)]],
                          constant MlpParams& p [[buffer(7)]],
                          uint simd_index [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint simdgroups [[simdgroups_per_threadgroup]]) {
  threadgroup half tiles[4 * (16 * 32 + 16 * 32 + 16 * 32)];   // per SIMD group: a (16x32), h (16x32), z (16x32)
  threadgroup float stage[4 * 256];
  threadgroup half* a = tiles + simd_index * 1536u;
  threadgroup half* h = a + 512u;
  threadgroup half* zt = h + 512u;
  threadgroup float* mine = stage + simd_index * 256u;
  const uint row0 = (tg.x * simdgroups + simd_index) * 16u;
  if (row0 >= p.tokens) return;
  // a = fp16(u * s); z = fp16(bc + u)
  for (uint e = lane; e < 512u; e += 32u) {
    const uint r = e >> 5u, c = e & 31u;
    const uint row = min(row0 + r, p.tokens - 1u);
    const half uv = U[row * p.u_stride + c];
    a[e] = uv * S[c];
    zt[e] = half(Bc[c] + uv);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 A[2][2][2];   // a as two K tiles
  load_ab(A[0], a, 32u);
  load_ab(A[1], a + 16u, 32u);
  #pragma clang loop unroll(full)
  for (uint g = 0u; g < 4u; ++g) {
    // expand: h[:, 0:32] = act(fp16(be + a . We[:, 32g:32g+32]))
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      const uint hid = 32u * g + 16u * n;
      simdgroup_float8x8 acc[2][2];
      // bias broadcast
      for (uint e = lane; e < 256u; e += 32u) mine[e] = float(Be[hid + (e & 15u)]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 2u; ++k) {
        simdgroup_float8x8 B[2][2];
        load_b_device(B, We + (k * 16u) * 128u + hid, 128u);
        mma16(acc, A[k], B);
        if (g == 0u && n == 0u && k == 0u) {   // split quirk (hidden tile 0): fp16 handoff for window blocks 0..2
          #pragma clang loop unroll(full)
          for (uint i = 0u; i < 2u; ++i)
          #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint e = lane; e < 256u; e += 32u) {
            const uint row = min(row0 + (e >> 4u), p.tokens - 1u);
            const uint block = (((row / p.width) + p.shift_y) & 7u) >> 1u;
            if (block != 3u) mine[e] = float(half(mine[e]));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint i = 0u; i < 2u; ++i)
          #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 256u; e += 32u) h[(e >> 4u) * 32u + 16u * n + (e & 15u)] = mlp_gelu(half(mine[e]));
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // contract: z = fp16(z + h . Wc[32g:32g+32, :])
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      simdgroup_float8x8 acc[2][2];
      for (uint e = lane; e < 256u; e += 32u) mine[e] = float(zt[(e >> 4u) * 32u + 16u * n + (e & 15u)]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint k = 0u; k < 2u; ++k) {
        simdgroup_float8x8 Ah[2][2], B[2][2];
        load_ab(Ah, h + k * 16u, 32u);
        load_b_device(B, Wc + (32u * g + 16u * k) * 32u + 16u * n, 32u);
        mma16(acc, Ah, B);
      }
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 256u; e += 32u) zt[(e >> 4u) * 32u + 16u * n + (e & 15u)] = half(mine[e]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const bool full = row0 + 16u <= p.tokens;
  for (uint e = lane; e < 512u; e += 32u) {
    const uint r = e >> 5u, c = e & 31u;
    if (full || row0 + r < p.tokens) Out[(row0 + r) * p.out_stride + c] = zt[e];
  }
}
