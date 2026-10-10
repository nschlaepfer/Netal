// Window mixing + projection + residual on the padded grid (no reflection, no alt variant):
// one threadgroup per window, 4 SIMD groups (group r = window rows 2r, 2r+1).
//   Z = Zin rows of the 64 window positions (padded index), M = fp16(W3 . Z), out = fp16(C + M . W4),
//   C = R (r_mode 0) | fp16(max(R,0) + Rbias) (1) | fp16(R + Rbias) (2); stores for in-image positions only.
#include <metal_stdlib>
using namespace metal;

struct MixPParams {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, r_stride, o_stride, r_mode;
  uint out_padded;         // 1: Out is a padded-grid tensor (store all 64 positions); 0: image layout, clipped
  uint pad1, pad2, pad3;
};

kernel void pwin_mixp_window(device const half* Zin [[buffer(0)]],
                             device const half* W3 [[buffer(1)]],
                             device const half* W4 [[buffer(2)]],
                             device const half* R [[buffer(3)]],
                             device half* Out [[buffer(4)]],
                             constant MixPParams& p [[buffer(5)]],
                             device const half* Rbias [[buffer(6)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Z[64 * 32];
  threadgroup half M[64 * 32];
  threadgroup float stage[4 * 256];
  threadgroup float* mine = stage + simd_index * 256u;
  const uint Wp = p.width + 8u;
  const int x0 = int(tg.x) * 8 - p.shift_x;      // padded coordinates are x0 + 4 .. (always >= 0)
  const int y0 = int(tg.y) * 8 - p.shift_y;
  {
    const uint t = lid >> 1u;
    const uint h = (lid & 1u) * 16u;
    const uint tp = uint(y0 + 4 + int(t >> 3u)) * Wp + uint(x0 + 4 + int(t & 7u));
    device const half* src = Zin + tp * p.z_stride + h;
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 16u; c += 4u) *reinterpret_cast<threadgroup half4*>(Z + t * 32u + h + c) = *reinterpret_cast<device const half4*>(src + c);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = simd_index;
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
  #pragma clang loop unroll(full)
  for (uint n0 = 0u; n0 < 32u; n0 += 16u) {
    simdgroup_float8x8 acc[2][2];
    for (uint e = lane; e < 256u; e += 32u) {
      const uint row = e >> 4u, col = e & 15u;
      const uint t = r * 16u + row;
      const uint tp = uint(y0 + 4 + int(t >> 3u)) * Wp + uint(x0 + 4 + int(t & 7u));
      const half rv = R[tp * p.r_stride + n0 + col];
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
      if (p.out_padded != 0u) {
        Out[(uint(y + 4) * Wp + uint(x + 4)) * p.o_stride + n0 + col] = half(mine[e]);
      } else if (x >= 0 && y >= 0 && x < int(p.width) && y < int(p.height)) {
        Out[(uint(y) * p.width + uint(x)) * p.o_stride + n0 + col] = half(mine[e]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
