// Window mixing + projection + residual on the padded grid, persistent version: no threadgroup memory,
// no barriers. SIMD group task r (window rows 2r, 2r+1 of the output) loops over windows; every window's
// 64 positions exist in the padded tensors, so Z sub-blocks load straight from the token tensor (8
// contiguous tokens per window row), M = fp16(W3 . Z) stays in registers as the A operand of the
// projection, C is built per element, and out = fp16(C + M . W4) is stored per element (padded layout or
// clipped image layout). Same numerics as pwin_mixp_window.
#include <metal_stdlib>
using namespace metal;

constant uint FC_PZS [[function_constant(11)]];   // Z row stride (halves)
constant uint FC_PRS [[function_constant(12)]];   // R row stride
constant uint FC_POS [[function_constant(13)]];   // Out row stride

struct MixP2Params {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, r_stride, o_stride, r_mode;
  uint out_padded, windows_x, windows_y, pad3;
};

static inline void mixp2_load_dev(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
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

kernel void pwin_mixp2_persistent(device const half* Zin [[buffer(0)]],
                                  device const half* W3 [[buffer(1)]],
                                  device const half* W4 [[buffer(2)]],
                                  device const half* R [[buffer(3)]],
                                  device half* Out [[buffer(4)]],
                                  constant MixP2Params& p [[buffer(5)]],
                                  device const half* Rbias [[buffer(6)]],
                                  uint simd_index [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  uint simdgroups [[simdgroups_per_threadgroup]],
                                  uint3 tgs [[threadgroups_per_grid]]) {
  const uint group_linear = tg.x * simdgroups + simd_index;
  const uint groups_total = tgs.x * simdgroups;
  const uint r = group_linear & 3u;
  const uint stride_windows = groups_total >> 2u;
  const uint first_window = group_linear >> 2u;
  const uint n_windows = p.windows_x * p.windows_y;
  const uint Wp = p.width + 8u;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  for (uint w = first_window; w < n_windows; w += stride_windows) {
    const uint wy = w / p.windows_x, wx = w - wy * p.windows_x;
    const int x0 = int(wx) * 8 - p.shift_x;
    const int y0 = int(wy) * 8 - p.shift_y;
    const uint px0 = uint(x0 + 4), py0 = uint(y0 + 4);      // padded origin (always >= 0)
    // M[r] = fp16(W3[16r..16r+15, :] . Z): both n tiles, K = 64 source tokens in raster order
    simdgroup_float8x8 M[2][2][2];   // [n][i][j]
#pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n)
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) M[n][i][j] = simdgroup_float8x8(0.0f);
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
      simdgroup_float8x8 a[2][2];
      mixp2_load_dev(a, W3 + (r * 16u) * 64u + k * 16u, 64u);
#pragma clang loop unroll(full)
      for (uint n = 0u; n < 2u; ++n) {
        simdgroup_float8x8 b[2][2];
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i) {
          // B rows 16k + 8i .. +7 = window row 2k + i, 8 contiguous padded tokens
          device const half* base = Zin + ((py0 + 2u * k + i) * Wp + px0) * FC_PZS + n * 16u;
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_half8x8 hm;
            simdgroup_load(hm, base + j * 8u, FC_PZS);
            b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          }
        }
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
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          M[n][i][j].thread_elements()[0] = float(half(M[n][i][j].thread_elements()[0]));
          M[n][i][j].thread_elements()[1] = float(half(M[n][i][j].thread_elements()[1]));
        }
    // out[r] = fp16(C + M[r] . W4), K = 32 (k tile kk uses M[n = kk])
#pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      simdgroup_float8x8 acc[2][2];
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i) {
        const uint t = r * 16u + i * 8u + er;
        const uint tp = (py0 + (t >> 3u)) * Wp + px0 + (t & 7u);
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          const uint col = n * 16u + j * 8u + ec;
          const half2 rv = *reinterpret_cast<device const half2*>(R + tp * FC_PRS + col);
          if (p.r_mode == 0u) {
            acc[i][j].thread_elements()[0] = float(rv.x);
            acc[i][j].thread_elements()[1] = float(rv.y);
          } else {
            const half2 bv = *reinterpret_cast<device const half2*>(Rbias + col);
            if (p.r_mode == 1u) {
              acc[i][j].thread_elements()[0] = float(half(max(rv.x, half(0.0h)) + bv.x));
              acc[i][j].thread_elements()[1] = float(half(max(rv.y, half(0.0h)) + bv.y));
            } else {
              acc[i][j].thread_elements()[0] = float(half(rv.x + bv.x));
              acc[i][j].thread_elements()[1] = float(half(rv.y + bv.y));
            }
          }
        }
      }
#pragma clang loop unroll(full)
      for (uint kk = 0u; kk < 2u; ++kk) {
        simdgroup_float8x8 b[2][2];
        mixp2_load_dev(b, W4 + (kk * 16u) * 32u + n * 16u, 32u);
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
        const uint t = r * 16u + i * 8u + er;
        const int x = x0 + int(t & 7u), y = y0 + int(t >> 3u);
        const uint tp = (py0 + (t >> 3u)) * Wp + px0 + (t & 7u);
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          const uint col = n * 16u + j * 8u + ec;
          const half2 v = half2(half(acc[i][j].thread_elements()[0]), half(acc[i][j].thread_elements()[1]));
          if (p.out_padded != 0u) {
            *reinterpret_cast<device half2*>(Out + tp * FC_POS + col) = v;
          } else if (x >= 0 && y >= 0 && x < int(p.width) && y < int(p.height)) {
            *reinterpret_cast<device half2*>(Out + (uint(y) * p.width + uint(x)) * FC_POS + col) = v;
          }
        }
      }
    }
  }
}
