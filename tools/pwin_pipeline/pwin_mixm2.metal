// enc1 token mixing, interior windows: persistent SIMD groups, no threadgroup memory. Group task
// c = (row block r, channel half s): A = rows 16r..16r+15 of M1_s (4 K tiles) kept in registers across
// the windows the group visits; B = h1 rows of the window (8 contiguous tokens per window row when the
// window is inside the image) loaded straight from the token tensor. Same numerics and layout as
// pwin_mixm_window (fp32 accumulation over the 64 source tokens in raster order, one fp16 rounding).
// Border windows (any token outside the image) are skipped here and left to pwin_mixm_window.
#include <metal_stdlib>
using namespace metal;

constant uint FC_ZS [[function_constant(4)]];   // optional compile-time z_stride / o_stride (runtime strides in simdgroup_load are slow)
constant uint FC_OS [[function_constant(5)]];
constant uint FC_DBG [[function_constant(6)]];   // timing diagnostics only: 1 = no B loads (B := A tiles), 2 = no MMAs (acc := B)

struct Mixm2Params {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, o_stride, windows_x, windows_y;
};

static inline int mixm2_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}
static inline void mixm2_load_dev(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
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

template <bool INTERIOR>
static inline void mixm2_window(device const half* Zin, thread simdgroup_float8x8 (&A)[4][2][2], device half* Out, constant Mixm2Params& p, uint r, uint s, int x0, int y0, uint zs, uint os, uint er, uint ec) {
  const bool interior = INTERIOR;
  simdgroup_float8x8 acc[2][2][2];   // [n][i][j]
#pragma clang loop unroll(full)
  for (uint n = 0u; n < 2u; ++n)
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) acc[n][i][j] = simdgroup_float8x8(0.0f);
#pragma clang loop unroll(full)
  for (uint k = 0u; k < 4u; ++k) {
#pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      simdgroup_float8x8 b[2][2];
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i) {
        const int yy = y0 + int(2u * k + i);
        device const half* base = Zin + (uint(yy) * p.width + uint(x0)) * zs + s * 32u + n * 16u;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          if (is_function_constant_defined(FC_DBG) && FC_DBG == 1u) { b[i][j] = A[k][i][j]; continue; }
          if (interior) {
            simdgroup_half8x8 hm;
            simdgroup_load(hm, base + j * 8u, zs);
            b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          } else {
            // border window: B element (row er = source token column er of window row yy, cols ec, ec+1)
            const int xr = mixm2_reflect(x0 + int(er), int(p.width));
            const int yr = mixm2_reflect(yy, int(p.height));
            const half2 v = *reinterpret_cast<device const half2*>(Zin + (uint(yr) * p.width + uint(xr)) * zs + s * 32u + n * 16u + j * 8u + ec);
            b[i][j].thread_elements()[0] = float(v.x);
            b[i][j].thread_elements()[1] = float(v.y);
          }
        }
      }
      if (is_function_constant_defined(FC_DBG) && FC_DBG == 2u) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          acc[n][i][j].thread_elements()[0] += b[i][j].thread_elements()[0];
          acc[n][i][j].thread_elements()[1] += b[i][j].thread_elements()[1];
        }
        continue;
      }
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(acc[n][i][j], A[k][i][0], b[0][j], acc[n][i][j]);
        simdgroup_multiply_accumulate(acc[n][i][j], A[k][i][1], b[1][j], acc[n][i][j]);
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    const int yy = y0 + int(2u * r + i);
    if (yy < 0 || yy >= int(p.height)) continue;
    const int xx = x0 + int(er);
    const bool xok = xx >= 0 && xx < int(p.width);
#pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n)
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
      simdgroup_half8x8 hm;
      hm.thread_elements()[0] = half(acc[n][i][j].thread_elements()[0]);
      hm.thread_elements()[1] = half(acc[n][i][j].thread_elements()[1]);
      if (interior) {
        simdgroup_store(hm, Out + (uint(yy) * p.width + uint(x0)) * os + s * 32u + n * 16u + j * 8u, os);
      } else if (xok) {
        *reinterpret_cast<device half2*>(Out + (uint(yy) * p.width + uint(xx)) * os + s * 32u + n * 16u + j * 8u + ec) = half2(hm.thread_elements()[0], hm.thread_elements()[1]);
      }
    }
  }
}

kernel void pwin_mixm2_interior(device const half* Zin [[buffer(0)]],
                                device const half* Ma [[buffer(1)]],
                                device const half* Mb [[buffer(2)]],
                                device half* Out [[buffer(3)]],
                                constant Mixm2Params& p [[buffer(4)]],
                                uint simd_index [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                uint simdgroups [[simdgroups_per_threadgroup]],
                                uint3 tgs [[threadgroups_per_grid]]) {
  const uint group_linear = tg.x * simdgroups + simd_index;
  const uint groups_total = tgs.x * simdgroups;
  const uint task = group_linear & 7u;                 // r = task >> 1, s = task & 1
  const uint r = task >> 1u, s = task & 1u;
  const uint stride_windows = groups_total >> 3u;
  const uint first_window = group_linear >> 3u;
  const uint n_windows = p.windows_x * p.windows_y;
  const uint zs = is_function_constant_defined(FC_ZS) ? FC_ZS : p.z_stride;
  const uint os = is_function_constant_defined(FC_OS) ? FC_OS : p.o_stride;
  device const half* M = s == 0u ? Ma : Mb;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  simdgroup_float8x8 A[4][2][2];
#pragma clang loop unroll(full)
  for (uint k = 0u; k < 4u; ++k) mixm2_load_dev(A[k], M + (r * 16u) * 64u + k * 16u, 64u);
  for (uint w = first_window; w < n_windows; w += stride_windows) {
    const uint wy = w / p.windows_x, wx = w - wy * p.windows_x;
    const int x0 = int(wx) * 8 - p.shift_x;
    const int y0 = int(wy) * 8 - p.shift_y;
    const bool interior = x0 >= 0 && y0 >= 0 && x0 + 8 <= int(p.width) && y0 + 8 <= int(p.height);
    if (!interior && p.windows_y == 0u) continue;   // windows_y == 0: interior-only mode (border windows by pwin_mixm_window)
    if (interior) mixm2_window<true>(Zin, A, Out, p, r, s, x0, y0, zs, os, er, ec);
    else mixm2_window<false>(Zin, A, Out, p, r, s, x0, y0, zs, os, er, ec);
  }
}
