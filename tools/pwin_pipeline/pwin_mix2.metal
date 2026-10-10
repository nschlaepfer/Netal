// Window mixing + projection + residual, version 2: one threadgroup per 8x8 window with 8 SIMD groups;
// SIMD group g owns output tile (r = g >> 1, n = g & 1) of both GEMMs. Same numerics as pwin_mix_window.
//   Z    = gather of Zin (64 window tokens x 32 features, reflect-padded; alt variant for block-3 rows)
//   M    = fp16( W3 (64x64) . Z )                      K order = token block 0..3
//   out  = fp16( C + M . W4 ),  C = R (raw) or fp16(max(R,0) + Rbias)
#include <metal_stdlib>
using namespace metal;

struct Mix2Params {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, r_stride, o_stride, r_mode;
  uint alt_mode, alt_stride, debug, pad3;
};

static inline int mix2_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}

static inline void mix2_load(thread simdgroup_float8x8 (&m)[2][2], threadgroup const half* src, uint stride) {
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
static inline void mix2_load_dev(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
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

kernel void pwin_mix2_window(device const half* Zin [[buffer(0)]],
                             device const half* W3 [[buffer(1)]],
                             device const half* W4 [[buffer(2)]],
                             device const half* R [[buffer(3)]],
                             device half* Out [[buffer(4)]],
                             constant Mix2Params& p [[buffer(5)]],
                             device const half* Rbias [[buffer(6)]],
                             device const half* Zalt [[buffer(7)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Z[64 * 32];
  threadgroup half M[64 * 32];
  threadgroup float stage[8 * 256];
  threadgroup float* mine = stage + simd_index * 256u;
  const int x0 = int(tg.x) * 8 - p.shift_x;
  const int y0 = int(tg.y) * 8 - p.shift_y;
  const uint r = simd_index >> 1u, n0 = (simd_index & 1u) * 16u;
  // constant weights for this group's tiles, loaded before the gather so their latency overlaps it
  simdgroup_float8x8 A3[4][2][2], B4[2][2][2];
#pragma clang loop unroll(full)
  for (uint k = 0u; k < 4u; ++k) mix2_load_dev(A3[k], W3 + (r * 16u) * 64u + k * 16u, 64u);
#pragma clang loop unroll(full)
  for (uint k = 0u; k < 2u; ++k) mix2_load_dev(B4[k], W4 + k * 16u * 32u + n0, 32u);
  // gather: 256 threads, one thread per (token, 8-feature chunk): 64 x 4 chunks
  {
    const uint t = lid >> 2u;
    const uint h = (lid & 3u) * 8u;
    const int x = mix2_reflect(x0 + int(t & 7u), int(p.width));
    const int y = mix2_reflect(y0 + int(t >> 3u), int(p.height));
    const bool alt = p.alt_mode != 0u && h < 16u && ((t >> 3u) >> 1u) == 3u;
    device const half* src = alt ? (Zalt + (uint(y) * p.width + uint(x)) * p.alt_stride + h)
                                 : (Zin + (uint(y) * p.width + uint(x)) * p.z_stride + h);
    const half4 v0 = p.debug == 1u ? half4(half(lid)) : *reinterpret_cast<device const half4*>(src);
    const half4 v1 = p.debug == 1u ? half4(half(lid)) : *reinterpret_cast<device const half4*>(src + 4);
    *reinterpret_cast<threadgroup half4*>(Z + t * 32u + h) = v0;
    *reinterpret_cast<threadgroup half4*>(Z + t * 32u + h + 4u) = v1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // M tile (r, n): W3[16r.., :] . Z[:, n0..]
  {
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
      if (p.debug == 2u) break;
      simdgroup_float8x8 b[2][2];
      mix2_load(b, Z + k * 16u * 32u + n0, 32u);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], A3[k][i][0], b[0][j], acc[i][j]);
        simdgroup_multiply_accumulate(acc[i][j], A3[k][i][1], b[1][j], acc[i][j]);
      }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 256u; e += 32u) M[(r * 16u + (e >> 4u)) * 32u + n0 + (e & 15u)] = half(mine[e]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // out tile (r, n): C + M[r] . W4[:, n0..]
  {
    simdgroup_float8x8 acc[2][2];
    for (uint e = lane; e < 256u; e += 32u) {
      const uint row = e >> 4u, col = e & 15u;
      const uint t = r * 16u + row;
      const int x = mix2_reflect(x0 + int(t & 7u), int(p.width));
      const int y = mix2_reflect(y0 + int(t >> 3u), int(p.height));
      const half rv = p.debug == 3u ? half(e) : R[(uint(y) * p.width + uint(x)) * p.r_stride + n0 + col];
      mine[e] = p.r_mode == 1u ? float(half(max(rv, half(0.0h)) + Rbias[n0 + col])) : float(rv);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 2u; ++k) {
      simdgroup_float8x8 a[2][2];
      mix2_load(a, M + (r * 16u) * 32u + k * 16u, 32u);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], a[i][0], B4[k][0][j], acc[i][j]);
        simdgroup_multiply_accumulate(acc[i][j], a[i][1], B4[k][1][j], acc[i][j]);
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
  }
}
