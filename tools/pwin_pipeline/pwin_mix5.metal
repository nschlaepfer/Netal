// Window mixing + projection + residual, version 5: one threadgroup per window, 4 SIMD groups
// (group r = window rows 2r, 2r+1), W3/W4 copied into threadgroup memory once per threadgroup,
// Z gathered into threadgroup memory, M kept in registers (rounded to fp16 in place) as the A
// operand of the projection, stage-free epilogue. Threadgroup memory: 4 KB Z + 10 KB weights.
// Same numerics as pwin_mix_window.
#include <metal_stdlib>
using namespace metal;

struct Mix5Params {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, r_stride, o_stride, r_mode;
  uint alt_mode, alt_stride, pad2, pad3;
};

static inline int mix5_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}
static inline void mix5_load_tg(thread simdgroup_float8x8 (&m)[2][2], threadgroup const half* src, uint stride) {
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

kernel void pwin_mix5_window(device const half* Zin [[buffer(0)]],
                             device const half* W3 [[buffer(1)]],
                             device const half* W4 [[buffer(2)]],
                             device const half* R [[buffer(3)]],
                             device half* Out [[buffer(4)]],
                             constant Mix5Params& p [[buffer(5)]],
                             device const half* Rbias [[buffer(6)]],
                             device const half* Zalt [[buffer(7)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Z[64 * 32];
  threadgroup half W3t[64 * 64];
  threadgroup half W4t[32 * 32];
  const int x0 = int(tg.x) * 8 - p.shift_x;
  const int y0 = int(tg.y) * 8 - p.shift_y;
  // weights: 4096 + 1024 halves, 128 threads -> 40 half4 stores each
  for (uint e = lid; e < 1024u; e += 128u) *reinterpret_cast<threadgroup half4*>(W3t + e * 4u) = *reinterpret_cast<device const half4*>(W3 + e * 4u);
  for (uint e = lid; e < 256u; e += 128u) *reinterpret_cast<threadgroup half4*>(W4t + e * 4u) = *reinterpret_cast<device const half4*>(W4 + e * 4u);
  // gather Z: 64 tokens x 32 halves, 128 threads -> 16 halves each
  {
    const uint t = lid >> 1u;
    const uint h = (lid & 1u) * 16u;
    const int x = mix5_reflect(x0 + int(t & 7u), int(p.width));
    const int y = mix5_reflect(y0 + int(t >> 3u), int(p.height));
    const bool alt = p.alt_mode != 0u && h == 0u && ((t >> 3u) >> 1u) == 3u;
    device const half* src = alt ? (Zalt + (uint(y) * p.width + uint(x)) * p.alt_stride)
                                 : (Zin + (uint(y) * p.width + uint(x)) * p.z_stride + h);
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 16u; c += 4u)
      *reinterpret_cast<threadgroup half4*>(Z + t * 32u + h + c) = *reinterpret_cast<device const half4*>(src + c);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = simd_index;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  // M[r] (both n tiles) = W3[r] . Z
  simdgroup_float8x8 M[2][2][2];
  #pragma clang loop unroll(full)
  for (uint n = 0u; n < 2u; ++n)
  #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) for (uint j = 0u; j < 2u; ++j) M[n][i][j] = simdgroup_float8x8(0.0f);
  #pragma clang loop unroll(full)
  for (uint k0 = 0u; k0 < 64u; k0 += 16u) {
    simdgroup_float8x8 a[2][2];
    mix5_load_tg(a, W3t + (r * 16u) * 64u + k0, 64u);
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      simdgroup_float8x8 b[2][2];
      mix5_load_tg(b, Z + k0 * 32u + n * 16u, 32u);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(M[n][i][j], a[i][0], b[0][j], M[n][i][j]);
        simdgroup_multiply_accumulate(M[n][i][j], a[i][1], b[1][j], M[n][i][j]);
      }
    }
  }
  #pragma clang loop unroll(full)
  for (uint n = 0u; n < 2u; ++n)
  #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) for (uint j = 0u; j < 2u; ++j) {
    M[n][i][j].thread_elements()[0] = float(half(M[n][i][j].thread_elements()[0]));
    M[n][i][j].thread_elements()[1] = float(half(M[n][i][j].thread_elements()[1]));
  }
  // out[r] = C + M[r] . W4
  #pragma clang loop unroll(full)
  for (uint n = 0u; n < 2u; ++n) {
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
      const uint t = r * 16u + i * 8u + er;
      const int x = mix5_reflect(x0 + int(t & 7u), int(p.width));
      const int y = mix5_reflect(y0 + int(t >> 3u), int(p.height));
      const half2 rv = *reinterpret_cast<device const half2*>(R + (uint(y) * p.width + uint(x)) * p.r_stride + n * 16u + j * 8u + ec);
      if (p.r_mode == 1u) {
        const half2 bv = *reinterpret_cast<device const half2*>(Rbias + n * 16u + j * 8u + ec);
        acc[i][j].thread_elements()[0] = float(half(max(rv.x, half(0.0h)) + bv.x));
        acc[i][j].thread_elements()[1] = float(half(max(rv.y, half(0.0h)) + bv.y));
      } else {
        acc[i][j].thread_elements()[0] = float(rv.x);
        acc[i][j].thread_elements()[1] = float(rv.y);
      }
    }
    #pragma clang loop unroll(full)
    for (uint kk = 0u; kk < 2u; ++kk) {
      simdgroup_float8x8 b[2][2];
      mix5_load_tg(b, W4t + (kk * 16u) * 32u + n * 16u, 32u);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], M[kk][i][0], b[0][j], acc[i][j]);
        simdgroup_multiply_accumulate(acc[i][j], M[kk][i][1], b[1][j], acc[i][j]);
      }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
      const int y = y0 + int(2u * r + i);
      if (y < 0 || y >= int(p.height)) continue;
      const bool row_full = x0 >= 0 && x0 + 8 <= int(p.width);
      #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        simdgroup_half8x8 hm;
        hm.thread_elements()[0] = half(acc[i][j].thread_elements()[0]);
        hm.thread_elements()[1] = half(acc[i][j].thread_elements()[1]);
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
}
