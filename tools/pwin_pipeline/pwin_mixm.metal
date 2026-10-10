// Token mixing only (enc1): m1[t, c] = sum_k Msel(c)[t, k] * Z[k, c] over the 64 tokens k of the window,
// Msel = Ma for channels 0..31 and Mb for channels 32..63 (two different 64x64 matrices, row-major
// [out token][source token]). One threadgroup per window (8x8 tokens, shifted grid), 4 SIMD groups
// (group r = output tokens 16r..16r+15), Z gathered with reflection into threadgroup memory
// (64 tokens x 64 channels), A tiles streamed from device memory. fp32 accumulation over all 64 source
// tokens in raster order, one fp16 rounding, stores clipped to in-image tokens.
#include <metal_stdlib>
using namespace metal;

struct MixmParams {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, o_stride, border_only, pad1;   // border_only: skip windows fully inside the image (pwin_mixm2_interior)
};

static inline int mixm_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}
static inline void mixm_load_dev(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
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
static inline void mixm_load_tg(thread simdgroup_float8x8 (&m)[2][2], threadgroup const half* src, uint stride) {
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

kernel void pwin_mixm_window(device const half* Zin [[buffer(0)]],
                             device const half* Ma [[buffer(1)]],
                             device const half* Mb [[buffer(2)]],
                             device half* Out [[buffer(3)]],
                             constant MixmParams& p [[buffer(4)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Z[64 * 64];
  const int x0 = int(tg.x) * 8 - p.shift_x;
  const int y0 = int(tg.y) * 8 - p.shift_y;
  if (p.border_only != 0u && x0 >= 0 && y0 >= 0 && x0 + 8 <= int(p.width) && y0 + 8 <= int(p.height)) return;
  {
    const uint t = lid >> 1u;
    const uint h = (lid & 1u) * 32u;
    const int x = mixm_reflect(x0 + int(t & 7u), int(p.width));
    const int y = mixm_reflect(y0 + int(t >> 3u), int(p.height));
    device const half* src = Zin + (uint(y) * p.width + uint(x)) * p.z_stride + h;
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 32u; c += 4u)
      *reinterpret_cast<threadgroup half4*>(Z + t * 64u + h + c) = *reinterpret_cast<device const half4*>(src + c);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = simd_index;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  simdgroup_float8x8 M[4][2][2];
  #pragma clang loop unroll(full)
  for (uint n = 0u; n < 4u; ++n)
  #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) for (uint j = 0u; j < 2u; ++j) M[n][i][j] = simdgroup_float8x8(0.0f);
  #pragma clang loop unroll(full)
  for (uint k0 = 0u; k0 < 64u; k0 += 16u) {
    simdgroup_float8x8 a[2][2][2];
    mixm_load_dev(a[0], Ma + (r * 16u) * 64u + k0, 64u);
    mixm_load_dev(a[1], Mb + (r * 16u) * 64u + k0, 64u);
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 4u; ++n) {
      simdgroup_float8x8 b[2][2];
      mixm_load_tg(b, Z + k0 * 64u + n * 16u, 64u);
      const uint sel = n >> 1u;
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(M[n][i][j], a[sel][i][0], b[0][j], M[n][i][j]);
        simdgroup_multiply_accumulate(M[n][i][j], a[sel][i][1], b[1][j], M[n][i][j]);
      }
    }
  }
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    const int y = y0 + int(2u * r + i);
    if (y < 0 || y >= int(p.height)) continue;
    const bool row_full = x0 >= 0 && x0 + 8 <= int(p.width);
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 4u; ++n)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
      simdgroup_half8x8 hm;
      hm.thread_elements()[0] = half(M[n][i][j].thread_elements()[0]);
      hm.thread_elements()[1] = half(M[n][i][j].thread_elements()[1]);
      if (row_full) {
        simdgroup_store(hm, Out + (uint(y) * p.width + uint(x0)) * p.o_stride + n * 16u + j * 8u, p.o_stride);
      } else {
        const int x = x0 + int(er);
        if (x >= 0 && x < int(p.width)) {
          device half* op = Out + (uint(y) * p.width + uint(x)) * p.o_stride + n * 16u + j * 8u + ec;
          op[0] = hm.thread_elements()[0];
          op[1] = hm.thread_elements()[1];
        }
      }
    }
  }
}
