// Window mixing + projection + residual, version 3: no fp32 staging. One threadgroup per window,
// 8 SIMD groups, group g owns output tile (r = g >> 1, n = g & 1). Accumulators are converted to
// fp16 per element in registers (same RNE rounding as the staged path), the residual accumulator
// is initialised per element, and tiles are stored with simdgroup_store. Threadgroup memory: 8 KB.
// Element ownership of an 8x8 simdgroup matrix on this GPU (probed): lane l holds row
// ((l>>1)&3) + 4*((l>>4)&1), columns 2*(l&1) + 4*((l>>3)&1) + {0,1}.
#include <metal_stdlib>
using namespace metal;

struct Mix3Params {
  uint width, height;
  int shift_x, shift_y;
  uint z_stride, r_stride, o_stride, r_mode;
  uint alt_mode, alt_stride, debug, pad3;
};

static inline int mix3_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}
static inline void mix3_load_tg(thread simdgroup_float8x8 (&m)[2][2], threadgroup const half* src, uint stride) {
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
static inline void mix3_load_dev(thread simdgroup_float8x8 (&m)[2][2], device const half* src, uint stride) {
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

kernel void pwin_mix3_window(device const half* Zin [[buffer(0)]],
                             device const half* W3 [[buffer(1)]],
                             device const half* W4 [[buffer(2)]],
                             device const half* R [[buffer(3)]],
                             device half* Out [[buffer(4)]],
                             constant Mix3Params& p [[buffer(5)]],
                             device const half* Rbias [[buffer(6)]],
                             device const half* Zalt [[buffer(7)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Z[64 * 32];
  threadgroup half M[64 * 32];
  const int x0 = int(tg.x) * 8 - p.shift_x;
  const int y0 = int(tg.y) * 8 - p.shift_y;
  {
    const uint t = lid >> 2u;
    const uint h = (lid & 3u) * 8u;
    const int x = mix3_reflect(x0 + int(t & 7u), int(p.width));
    const int y = mix3_reflect(y0 + int(t >> 3u), int(p.height));
    const bool alt = p.alt_mode != 0u && h < 16u && ((t >> 3u) >> 1u) == 3u;
    device const half* src = alt ? (Zalt + (uint(y) * p.width + uint(x)) * p.alt_stride + h)
                                 : (Zin + (uint(y) * p.width + uint(x)) * p.z_stride + h);
    *reinterpret_cast<threadgroup half4*>(Z + t * 32u + h) = *reinterpret_cast<device const half4*>(src);
    *reinterpret_cast<threadgroup half4*>(Z + t * 32u + h + 4u) = *reinterpret_cast<device const half4*>(src + 4);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = simd_index >> 1u, n0 = (simd_index & 1u) * 16u;
  // element coordinates of this lane inside an 8x8 matrix
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  // M tile (r, n)
  {
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    #pragma clang loop unroll(full)
    for (uint k0 = 0u; k0 < 64u; k0 += 16u) {
      simdgroup_float8x8 a[2][2], b[2][2];
      mix3_load_dev(a, W3 + (r * 16u) * 64u + k0, 64u);
      mix3_load_tg(b, Z + k0 * 32u + n0, 32u);
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
      for (uint j = 0u; j < 2u; ++j) {
      simdgroup_half8x8 hm;
      hm.thread_elements()[0] = half(acc[i][j].thread_elements()[0]);
      hm.thread_elements()[1] = half(acc[i][j].thread_elements()[1]);
      simdgroup_store(hm, M + (r * 16u + i * 8u) * 32u + n0 + j * 8u, 32u);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // out tile (r, n): C + M[r] . W4[:, n0..]
  {
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
      const uint t = r * 16u + i * 8u + er;
      const int x = mix3_reflect(x0 + int(t & 7u), int(p.width));
      const int y = mix3_reflect(y0 + int(t >> 3u), int(p.height));
      device const half* rp = R + (uint(y) * p.width + uint(x)) * p.r_stride + n0 + j * 8u + ec;
      const half2 rv = *reinterpret_cast<device const half2*>(rp);
      if (p.r_mode == 1u) {
        const half2 bv = *reinterpret_cast<device const half2*>(Rbias + n0 + j * 8u + ec);
        acc[i][j].thread_elements()[0] = float(half(max(rv.x, half(0.0h)) + bv.x));
        acc[i][j].thread_elements()[1] = float(half(max(rv.y, half(0.0h)) + bv.y));
      } else {
        acc[i][j].thread_elements()[0] = float(rv.x);
        acc[i][j].thread_elements()[1] = float(rv.y);
      }
    }
    #pragma clang loop unroll(full)
    for (uint k0 = 0u; k0 < 32u; k0 += 16u) {
      simdgroup_float8x8 a[2][2], b[2][2];
      mix3_load_tg(a, M + (r * 16u) * 32u + k0, 32u);
      mix3_load_dev(b, W4 + k0 * 32u + n0, 32u);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], a[i][0], b[0][j], acc[i][j]);
        simdgroup_multiply_accumulate(acc[i][j], a[i][1], b[1][j], acc[i][j]);
      }
    }
    // store: rows of this tile are window tokens 16r + i*8 + row; each 8-row block is one window row
    // (x = x0 + col, y = y0 + (2r + i)), stored only when inside the image
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
          simdgroup_store(hm, Out + (uint(y) * p.width + uint(x0)) * p.o_stride + n0 + j * 8u, p.o_stride);
        } else {
          const int x = x0 + int(er);
          if (x >= 0 && x < int(p.width)) {
            device half* op = Out + (uint(y) * p.width + uint(x)) * p.o_stride + n0 + j * 8u + ec;
            op[0] = hm.thread_elements()[0];
            op[1] = hm.thread_elements()[1];
          }
        }
      }
    }
  }
}
