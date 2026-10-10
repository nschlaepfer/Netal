// Window mixing + projection + residual, version 4 (interior windows): no threadgroup memory, no
// barriers. Each SIMD group handles tasks (window, r) for a fixed r = simd group's row block and keeps
// W3 rows 16r..16r+15 (4 K tiles) in registers across its windows. For an interior window the 8
// tokens of a window row are contiguous in the per-token tensor, so Z sub-blocks (8 tokens x 8
// features) load directly with simdgroup_load (stride = token row stride). Block-3 rows of Z take
// channels 0..15 from the fused-K variant Zalt. M (fp32 accumulators) is rounded to fp16 in place and
// used directly as the A operand of the projection. Same numerics as pwin_mix_window.
// Border windows (any window row outside the image) are skipped here and left to the gather kernel.
#include <metal_stdlib>
using namespace metal;

struct Mix4Params {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, r_stride, o_stride, r_mode;
  uint alt_mode, alt_stride, windows_x, windows_y;
};

static inline void mix4_load_dev(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
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
static inline void mix4_round_half(thread simdgroup_float8x8& m) {
  m.thread_elements()[0] = float(half(m.thread_elements()[0]));
  m.thread_elements()[1] = float(half(m.thread_elements()[1]));
}

kernel void pwin_mix4_interior(device const half* Zin [[buffer(0)]],
                               device const half* W3 [[buffer(1)]],
                               device const half* W4 [[buffer(2)]],
                               device const half* R [[buffer(3)]],
                               device half* Out [[buffer(4)]],
                               constant Mix4Params& p [[buffer(5)]],
                               device const half* Rbias [[buffer(6)]],
                               device const half* Zalt [[buffer(7)]],
                               uint simd_index [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]],
                               uint simdgroups [[simdgroups_per_threadgroup]],
                               uint3 tgs [[threadgroups_per_grid]]) {
  const uint r = simd_index & 3u;                         // row block of W3 / window rows 2r, 2r+1
  const uint group_linear = tg.x * simdgroups + simd_index;
  const uint groups_total = tgs.x * simdgroups;
  const uint stride_windows = groups_total / 4u;           // groups sharing the same r
  const uint first_window = (group_linear / 4u);
  const uint n_windows = p.windows_x * p.windows_y;
  // W3 rows in registers: 4 K tiles x [2][2]
  simdgroup_float8x8 A3[4][2][2];
#pragma clang loop unroll(full)
  for (uint k = 0u; k < 4u; ++k) mix4_load_dev(A3[k], W3 + (r * 16u) * 64u + k * 16u, 64u);
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  for (uint w = first_window; w < n_windows; w += stride_windows) {
    const uint wy = w / p.windows_x, wx = w - wy * p.windows_x;
    const int x0 = int(wx) * 8 - p.shift_x;
    const int y0 = int(wy) * 8 - p.shift_y;
    if (x0 < 0 || y0 < 0 || x0 + 8 > int(p.width) || y0 + 8 > int(p.height)) continue;   // border: gather kernel
    // M[r] = W3[r] . Z (both n tiles): 8 accumulators
    simdgroup_float8x8 M[2][2][2];   // [n][i][j]
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n)
    #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i) for (uint j = 0u; j < 2u; ++j) M[n][i][j] = simdgroup_float8x8(0.0f);
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
      // B tile (k, n): rows = 16 tokens of block k = window rows 2k, 2k+1 (8 contiguous tokens each)
      #pragma clang loop unroll(full)
      for (uint n = 0u; n < 2u; ++n) {
        simdgroup_float8x8 b[2][2];
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i) {
          const uint y = uint(y0) + 2u * k + i;
          const bool alt = p.alt_mode != 0u && n == 0u && k == 3u;
          device const half* base = alt ? (Zalt + (y * p.width + uint(x0)) * p.alt_stride)
                                        : (Zin + (y * p.width + uint(x0)) * p.z_stride + n * 16u);
          const uint stride = alt ? p.alt_stride : p.z_stride;
          #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_half8x8 hm;
            simdgroup_load(hm, base + j * 8u, stride);
            b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          }
        }
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
          simdgroup_multiply_accumulate(M[n][i][j], A3[k][i][0], b[0][j], M[n][i][j]);
          simdgroup_multiply_accumulate(M[n][i][j], A3[k][i][1], b[1][j], M[n][i][j]);
        }
      }
    }
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n)
    #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i) for (uint j = 0u; j < 2u; ++j) mix4_round_half(M[n][i][j]);
    // out[r] = C + M[r] . W4  (K = 32: k-tile kk uses M[n = kk])
    #pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      simdgroup_float8x8 acc[2][2];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        const uint y = uint(y0) + 2u * r + i;
        device const half* rp = R + (y * p.width + uint(x0) + er) * p.r_stride + n * 16u + j * 8u + ec;
        const half2 rv = *reinterpret_cast<device const half2*>(rp);
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
        mix4_load_dev(b, W4 + (kk * 16u) * 32u + n * 16u, 32u);
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
        const uint y = uint(y0) + 2u * r + i;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          hm.thread_elements()[0] = half(acc[i][j].thread_elements()[0]);
          hm.thread_elements()[1] = half(acc[i][j].thread_elements()[1]);
          simdgroup_store(hm, Out + (y * p.width + uint(x0)) * p.o_stride + n * 16u + j * 8u, p.o_stride);
        }
      }
    }
  }
}
