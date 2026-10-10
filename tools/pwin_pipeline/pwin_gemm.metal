// Batched per-token GEMM for the PWin pipeline: Y[t, n] = fp16( C[t, n] + sum_k X[t, k] * W[k, n] )
// with fp32 accumulation in the same 8x8-block order as the hw3 lowering of the transpiled
// kernels (per 16-wide K step: A[i][0]*B[0][j] then A[i][1]*B[1][j]), so the output is
// bit-identical to the `_hw` layers for the same operands.
//
// Operands are row-major fp16 in device memory and are loaded straight into simdgroup
// matrices (no staging): X is [tokens x K] with row stride `x_stride` halves, W is [K x N]
// with row stride N, C is an fp32 [16 x N] tile per N-tile (bias broadcast over rows; the
// accumulator input of the transpiled chains) and Y is [tokens x N] with stride `y_stride`.
// Each SIMD group owns 16 tokens; a threadgroup has `simdgroups` SIMD groups.
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
  uint tokens;      // number of rows
  uint k;           // reduction length (multiple of 16)
  uint n;           // output columns (multiple of 16)
  uint x_stride;    // halves between consecutive rows of X
  uint y_stride;    // halves between consecutive rows of Y
  uint c_stride;    // c_mode 0: floats between rows of the C tiles ([16 x n] fp32); c_mode 1: halves between rows of Cin
  uint epilogue;    // 0: none, 1: fp16 relu, 2: fp16 GELU approximation (t=clamp(h,-2,2); h*fma(t, fnma(a,|t|,b), 0.5))
  uint c_mode;      // 0: C = fp32 tile (bias broadcast); 1: C[t,n] = fp16(Cin[t,n] + Cbias[n]) (Cbias may be all zero)
  uint split_k16;   // transpiled split-chain quirk for N tile 0: the accumulator is rounded to fp16 after the first 16 K values;
                    //    1: every row (window-position variants are chosen by the consumer), 2: only tokens whose own window block ((y+shift_y)&7)>>1 != 3
  uint width;       // token grid width (for the quirk's window row)
  uint shift_y;     // window shift (4 for enc0)
  uint w_stride;    // halves between consecutive rows of W (0: = n)
  uint pad1, pad2, pad3, pad4;
};

static inline half gelu_approx(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

kernel void pwin_gemm_tokens(device const half* X [[buffer(0)]],
                             device const half* W [[buffer(1)]],
                             device const float* C [[buffer(2)]],
                             device half* Y [[buffer(3)]],
                             constant GemmParams& p [[buffer(4)]],
                             device const half* Cin [[buffer(5)]],     // c_mode 1: per-row fp16 accumulator input
                             device const half* Cbias [[buffer(6)]],   // c_mode 1: per-column fp16 bias added to Cin
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint simdgroups [[simdgroups_per_threadgroup]]) {
  threadgroup float stage[8 * 256];  // one 16x16 fp32 tile per SIMD group for the fp16 conversion
  threadgroup float* mine = stage + simd_index * 256u;
  const uint row0 = (tg.x * simdgroups + simd_index) * 16u;
  if (row0 >= p.tokens) return;
  const bool full = row0 + 16u <= p.tokens;
  const uint ws = p.w_stride == 0u ? p.n : p.w_stride;
  device const half* x = X + row0 * p.x_stride;
  for (uint n0 = 0u; n0 < p.n; n0 += 16u) {
    simdgroup_float8x8 acc[2][2];
    if (p.c_mode == 0u) {
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j)
        simdgroup_load(acc[i][j], C + n0 + j * 8u, p.c_stride, ulong2(0, i * 8u));
    } else {
      // per-row accumulator input: fp16(Cin + bias) converted to fp32 through the stage tile
      for (uint e = lane; e < 256u; e += 32u) {
        const uint r = e >> 4u, c = e & 15u;
        const uint row = min(row0 + r, p.tokens - 1u);
        mine[e] = float(half(Cin[row * p.c_stride + n0 + c] + Cbias[n0 + c]));
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k0 = 0u; k0 < p.k; k0 += 16u) {
      simdgroup_half8x8 ah[2][2], bh[2][2];
      simdgroup_float8x8 a[2][2], b[2][2];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_load(ah[i][j], x + k0 + j * 8u, p.x_stride, ulong2(0, i * 8u));
        simdgroup_load(bh[i][j], W + (k0 + i * 8u) * ws + n0 + j * 8u, ws);
        a[i][j].thread_elements()[0] = float(ah[i][j].thread_elements()[0]);
        a[i][j].thread_elements()[1] = float(ah[i][j].thread_elements()[1]);
        b[i][j].thread_elements()[0] = float(bh[i][j].thread_elements()[0]);
        b[i][j].thread_elements()[1] = float(bh[i][j].thread_elements()[1]);
      }
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], a[i][0], b[0][j], acc[i][j]);
        simdgroup_multiply_accumulate(acc[i][j], a[i][1], b[1][j], acc[i][j]);
      }
      if (p.split_k16 != 0u && k0 == 0u && n0 == 0u && p.k > 16u) {
        // fp16 handoff between the two K tiles for window blocks 0..2 (row-dependent)
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 256u; e += 32u) {
          const uint row = min(row0 + (e >> 4u), p.tokens - 1u);
          const uint y = row / p.width;
          const uint block = ((y + p.shift_y) & 7u) >> 1u;
          if (p.split_k16 == 1u || block != 3u) mine[e] = float(half(mine[e]));   // 1: every row, 2: window blocks 0..2
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
      for (uint j = 0u; j < 2u; ++j)
      simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // 256 values, 8 per lane: rows lane>>1, columns (lane&1)*8 .. +8
    const uint r = lane >> 1u, c0 = (lane & 1u) * 8u;
    if (full || row0 + r < p.tokens) {
      device half* y = Y + (row0 + r) * p.y_stride + n0 + c0;
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 8u; ++c) {
        half v = half(mine[r * 16u + c0 + c]);
        if (p.epilogue == 1u) v = max(v, half(0.0h));
        else if (p.epilogue == 2u) v = gelu_approx(v);
        y[c] = v;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
