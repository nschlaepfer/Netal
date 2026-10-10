// Window mixing + projection + residual for the PWin pipeline (enc0 chains 19-34 / 43-58):
//   for one 8x8 window (shifted, reflect-padded):
//     Z    = gather of the per-token tensor Zin[token, 0:32]   (64 window tokens x 32 features)
//     M    = fp16( W3 (64x64 token mixing) . Z )                (64 x 32), K order = token block 0..3
//     out  = fp16( R (residual, per token) + M . W4 )           (64 x 32)
//   stores out for the real (in-image) tokens only.
// fp32 accumulation in hw3 order (per 16-K step: A[i][0]*B[0][j] then A[i][1]*B[1][j]).
// One threadgroup per window, 4 SIMD groups, SIMD group r owns window rows 2r, 2r+1 (16 tokens).
#include <metal_stdlib>
using namespace metal;

struct MixParams {
  uint width, height;      // token grid
  int shift_x, shift_y;    // window shift (subtracted from tx*8)
  uint z_stride;           // halves per token row of Zin
  uint r_stride;           // halves per token row of the residual tensor
  uint o_stride;           // halves per token row of the output tensor
  uint r_mode;             // 0: C = R (raw fp16); 1: C = fp16(max(R, 0) + Rbias[n])
  uint alt_mode;           // 1: window tokens in block 3 (rows 6,7) take channels 0..15 of Z from Zalt (the fused-K variant)
  uint alt_stride;         // halves per token row of Zalt
  uint border_only;        // 1: skip interior windows (handled by pwin_mix4_interior)
  uint pad3;
};

static inline int reflect(int v, int n) {
  // transpiled formula: min((n << 1) - |v| - 2, |v|)
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}

kernel void pwin_mix_window(device const half* Zin [[buffer(0)]],
                            device const half* W3 [[buffer(1)]],     // [64 x 64] row-major (m, k)
                            device const half* W4 [[buffer(2)]],     // [32 x 32] row-major (k, n)
                            device const half* R [[buffer(3)]],      // residual source per token [tokens x 32]
                            device half* Out [[buffer(4)]],
                            constant MixParams& p [[buffer(5)]],
                            device const half* Rbias [[buffer(6)]],  // residual = fp16(max(R, 0) + Rbias[n]) (32 halves)
                            device const half* Zalt [[buffer(7)]],   // alt_mode: fused-K variant of Zin channels 0..15
                            uint simd_index [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Z[64 * 32];          // window tokens x features (row-major)
  threadgroup half M[64 * 32];          // mixed tokens x features
  threadgroup float stage[4 * 256];     // per SIMD group fp32 tile for fp16 rounding
  threadgroup float* mine = stage + simd_index * 256u;
  const int x0 = int(tg.x) * 8 - p.shift_x;
  const int y0 = int(tg.y) * 8 - p.shift_y;
  if (p.border_only != 0u && x0 >= 0 && y0 >= 0 && x0 + 8 <= int(p.width) && y0 + 8 <= int(p.height)) return;
  // gather: 64 tokens x 32 halves = 2048 halves, 128 threads -> 16 halves each (one token, half a row)
  {
    const uint t = lid >> 1u;               // token 0..63
    const uint h = (lid & 1u) * 16u;        // feature half
    const int x = reflect(x0 + int(t & 7u), int(p.width));
    const int y = reflect(y0 + int(t >> 3u), int(p.height));
    const bool alt = p.alt_mode != 0u && h == 0u && ((t >> 3u) >> 1u) == 3u;
    device const half* src = alt ? (Zalt + (uint(y) * p.width + uint(x)) * p.alt_stride)
                                 : (Zin + (uint(y) * p.width + uint(x)) * p.z_stride + h);
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 16u; ++c) Z[t * 32u + h + c] = src[c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = simd_index;              // m-tile (rows 16r..16r+15 of the window)
  // M[r] = W3[r rows, :] . Z   (K = 64 in 4 steps)
  #pragma clang loop unroll(full)
  for (uint n0 = 0u; n0 < 32u; n0 += 16u) {
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    #pragma clang loop unroll(full)
    for (uint k0 = 0u; k0 < 64u; k0 += 16u) {
      simdgroup_half8x8 ah[2][2], bh[2][2];
      simdgroup_float8x8 a[2][2], b[2][2];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_load(ah[i][j], W3 + (r * 16u + i * 8u) * 64u + k0 + j * 8u, 64u);
        simdgroup_load(bh[i][j], Z + (k0 + i * 8u) * 32u + n0 + j * 8u, 32u);
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
    }
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 256u; e += 32u) M[(r * 16u + (e >> 4u)) * 32u + n0 + (e & 15u)] = half(mine[e]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  // out[r] = R + M[r] . W4   (K = 32 in 2 steps); residual rows are the window tokens (reflected coords)
  #pragma clang loop unroll(full)
  for (uint n0 = 0u; n0 < 32u; n0 += 16u) {
    simdgroup_float8x8 acc[2][2];
    // residual tile: gather 16 rows x 16 cols fp32 into the stage, then load as C
    for (uint e = lane; e < 256u; e += 32u) {
      const uint row = e >> 4u, col = e & 15u;
      const uint t = r * 16u + row;
      const int x = reflect(x0 + int(t & 7u), int(p.width));
      const int y = reflect(y0 + int(t >> 3u), int(p.height));
      const half rv = R[(uint(y) * p.width + uint(x)) * p.r_stride + n0 + col];
      mine[e] = p.r_mode == 1u ? float(half(max(rv, half(0.0h)) + Rbias[n0 + col])) : (p.r_mode == 2u ? float(half(rv + Rbias[n0 + col])) : float(rv));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint k0 = 0u; k0 < 32u; k0 += 16u) {
      simdgroup_half8x8 ah[2][2], bh[2][2];
      simdgroup_float8x8 a[2][2], b[2][2];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_load(ah[i][j], M + (r * 16u + i * 8u) * 32u + k0 + j * 8u, 32u);
        simdgroup_load(bh[i][j], W4 + (k0 + i * 8u) * 32u + n0 + j * 8u, 32u);
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
    }
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 256u; e += 32u) {
      const uint row = e >> 4u, col = e & 15u;
      const uint t = r * 16u + row;
      const int x = x0 + int(t & 7u), y = y0 + int(t >> 3u);
      if (x >= 0 && y >= 0 && x < int(p.width) && y < int(p.height))
        Out[(uint(y) * p.width + uint(x)) * p.o_stride + n0 + col] = half(mine[e]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
