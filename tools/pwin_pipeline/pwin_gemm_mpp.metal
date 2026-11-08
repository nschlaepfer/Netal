// Opt-in, approximate PWin GEMM candidate using public Metal Performance Primitives.
// Each 128-thread group streams 64x16 and 16x32 half tiles (3 KiB total), so deep
// model shapes never allocate K*N threadgroup storage. grid.y partitions N in 32s.
// FP16 handoff/epilogue points follow gemm3; MPP's FP32 reduction order differs.
// This is not the reference-exact path and does not implement a DLSS 4.5 model.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifdef PWIN_MPP_SHARED_SHAPES
// The concatenated PWin library has already declared these two indices.
#define MPP_K FC_K
#define MPP_N FC_N
#else
constant uint MPP_K [[function_constant(0)]];
constant uint MPP_N [[function_constant(1)]];
#endif

// ABI shared with mpp_backend.py. The last word is reserved, and must be zero.
struct GemmMppParams {
  uint tokens, k, n, x_stride, y_stride, c_stride, epilogue, c_mode;
  uint split_k16, width, shift_y, w_stride, store_cols, r_stride, row_filter, reserved;
};

static inline half mpp_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

using mpp_tg_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

#if defined(PWIN_MPP_DIRECT) && PWIN_MPP_DIRECT
// Experimental direct-device MAC path. Full tiles use static extents; partial
// tiles retain bounded dynamic extents. No software staging/barriers are needed.
// Public API guidance: Apple MPP Programming Guide sections 2.2 and 2.3.5.
template <bool Full>
static inline auto mpp_direct_a(device const half* X, constant GemmMppParams& p,
                                uint row0, uint k0) {
  if constexpr (Full)
    return tensor<device half, extents<int32_t, 16, 64>, tensor_inline>(
        const_cast<device half*>(X) + row0 * p.x_stride + k0, extents<int32_t, 16, 64>{},
        array<int32_t, 2>{1, int32_t(p.x_stride)});
  else
    return tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device half*>(X) + row0 * p.x_stride + k0,
        dextents<int32_t, 2>{16, int32_t(min(64u, p.tokens - row0))},
        array<int32_t, 2>{1, int32_t(p.x_stride)});
}
template <bool Full>
static inline auto mpp_direct_b(device const half* W, uint ws, uint col0, uint k0) {
  if constexpr (Full)
    return tensor<device half, extents<int32_t, 32, 16>, tensor_inline>(
        const_cast<device half*>(W) + k0 * ws + col0, extents<int32_t, 32, 16>{},
        array<int32_t, 2>{1, int32_t(ws)});
  else
    return tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        const_cast<device half*>(W) + k0 * ws + col0,
        dextents<int32_t, 2>{int32_t(min(32u, MPP_N - col0)), 16},
        array<int32_t, 2>{1, int32_t(ws)});
}
template <bool Full>
static inline void mpp_direct_block(device const half* X, device const half* W,
                                    device const float* C, device half* Y,
                                    constant GemmMppParams& p,
                                    device const half* Cin, device const half* Cbias,
                                    uint row0, uint col0, uint ws, uint limit) {
  constexpr auto desc = matmul2d_descriptor(64, 32, 16, false, false, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto tA = mpp_direct_a<Full>(X, p, row0, 0u);
  auto tB = mpp_direct_b<Full>(W, ws, col0, 0u);
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 16; ++i) {
    if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
    cT[i] = 0.0f;
    const auto idx = cT.get_multidimensional_index(i);
    const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
    if (rr >= p.tokens || cc >= MPP_N) continue;
    // C is the original repeating 16-row FP32 MMA bias tile, not a token tensor.
    cT[i] = p.c_mode == 0u ? C[(rr & 15u) * p.c_stride + cc]
                           : float(half(Cin[rr * p.c_stride + cc] + Cbias[cc]));
  }
  const uint ktiles = MPP_K / 16u;
  const uint smode = p.split_k16 & 15u;
  const uint srounds = (p.split_k16 >> 4) != 0u ? (p.split_k16 >> 4) : ktiles - 1u;
  for (uint kt = 0u; kt < ktiles; ++kt) {
    auto a = mpp_direct_a<Full>(X, p, row0, kt * 16u);
    auto b = mpp_direct_b<Full>(W, ws, col0, kt * 16u);
    op.run(a, b, cT);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
      const auto idx = cT.get_multidimensional_index(i);
      const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
      const uint blk = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
      const bool round_here =
          ((smode == 1u || smode == 2u) && kt == 0u && cc < 16u && ktiles > 1u) ||
          (smode == 3u && kt == 1u) ||
          (smode == 4u && (kt & 1u) == 1u && kt + 1u < ktiles) ||
          ((smode == 5u || smode == 6u) && cc < 16u && kt < srounds);
      if (round_here && !(smode == 2u && blk == 3u) && !(smode == 5u && (blk & 1u)))
        cT[i] = float(half(cT[i]));
    }
  }
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 16; ++i) {
    if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
    const auto idx = cT.get_multidimensional_index(i);
    const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
    if (rr >= p.tokens || cc >= limit) continue;
    const uint blk = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
    if (p.row_filter != 0u && (blk & 1u)) continue;
    half h = half(cT[i]);
    if (p.epilogue == 1u) h = max(h, half(0));
    else if (p.epilogue == 2u) h = mpp_gelu(h);
    else if (p.epilogue == 3u) h = h + Cin[rr * p.r_stride + cc];
    Y[rr * p.y_stride + cc] = h;
  }
}
kernel void pwin_gemm_mpp(device const half* X [[buffer(0)]],
                          device const half* W [[buffer(1)]],
                          device const float* C [[buffer(2)]],
                          device half* Y [[buffer(3)]],
                          constant GemmMppParams& p [[buffer(4)]],
                          device const half* Cin [[buffer(5)]],
                          device const half* Cbias [[buffer(6)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint3 threads [[threads_per_threadgroup]]) {
  // Host validation rejects unsupported shapes and parameter values. These
  // uniform guards also fail closed if a hand-written descriptor violates ABI.
  const uint limit = p.store_cols == 0u ? MPP_N : p.store_cols;
  const uint ws = p.w_stride == 0u ? MPP_N : p.w_stride;
  const uint row0 = tg.x * 64u, col0 = tg.y * 32u;
  if (threads.x != 128u || threads.y != 1u || threads.z != 1u ||
      p.k != MPP_K || p.n != MPP_N || MPP_K == 0u || MPP_K > 640u ||
      MPP_N == 0u || MPP_N > 768u || (MPP_K & 15u) || (MPP_N & 15u) ||
      p.width == 0u || p.x_stride < MPP_K || ws < MPP_N ||
      limit > MPP_N || p.y_stride < limit || p.epilogue > 3u ||
      p.c_mode > 1u || p.row_filter > 1u || (p.split_k16 & 15u) > 6u ||
      p.reserved != 0u || row0 >= p.tokens || col0 >= limit) return;

  if (row0 + 64u <= p.tokens && col0 + 32u <= MPP_N)
    mpp_direct_block<true>(X, W, C, Y, p, Cin, Cbias, row0, col0, ws, limit);
  else
    mpp_direct_block<false>(X, W, C, Y, p, Cin, Cbias, row0, col0, ws, limit);
}
#else
kernel void pwin_gemm_mpp(device const half* X [[buffer(0)]],
                          device const half* W [[buffer(1)]],
                          device const float* C [[buffer(2)]],
                          device half* Y [[buffer(3)]],
                          constant GemmMppParams& p [[buffer(4)]],
                          device const half* Cin [[buffer(5)]],
                          device const half* Cbias [[buffer(6)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint3 threads [[threads_per_threadgroup]]) {
  // Host validation rejects unsupported shapes and parameter values. These
  // uniform guards also fail closed if a hand-written descriptor violates ABI.
  const uint limit = p.store_cols == 0u ? MPP_N : p.store_cols;
  const uint ws = p.w_stride == 0u ? MPP_N : p.w_stride;
  const uint row0 = tg.x * 64u, col0 = tg.y * 32u;
  if (threads.x != 128u || threads.y != 1u || threads.z != 1u ||
      p.k != MPP_K || p.n != MPP_N || MPP_K == 0u || MPP_K > 640u ||
      MPP_N == 0u || MPP_N > 768u || (MPP_K & 15u) || (MPP_N & 15u) ||
      p.width == 0u || p.x_stride < MPP_K || ws < MPP_N ||
      limit > MPP_N || p.y_stride < limit || p.epilogue > 3u ||
      p.c_mode > 1u || p.row_filter > 1u || (p.split_k16 & 15u) > 6u ||
      p.reserved != 0u || row0 >= p.tokens || col0 >= limit) return;

  threadgroup half Xs[64 * 16];
  threadgroup half Ws[16 * 32];
  constexpr auto desc = matmul2d_descriptor(64, 32, 16, false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto tA = mpp_tg_tensor(Xs, dextents<int32_t, 2>(16, 64), array<int32_t, 2>{1, 16});
  auto tB = mpp_tg_tensor(Ws, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 32});
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  float acc[16];
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 16; ++i) {
    acc[i] = 0.0f;
    if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
    const auto idx = cT.get_multidimensional_index(i);
    const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
    if (rr >= p.tokens || cc >= MPP_N) continue;
    // C is the original repeating 16-row FP32 MMA bias tile, not a token tensor.
    acc[i] = p.c_mode == 0u ? C[(rr & 15u) * p.c_stride + cc]
                           : float(half(Cin[rr * p.c_stride + cc] + Cbias[cc]));
  }
  const uint ktiles = MPP_K / 16u;
  const uint smode = p.split_k16 & 15u;
  const uint srounds = (p.split_k16 >> 4) != 0u ? (p.split_k16 >> 4) : ktiles - 1u;
  for (uint kt = 0u; kt < ktiles; ++kt) {
    for (uint e = lid; e < 64u * 16u; e += 128u) {
      const uint rr = row0 + e / 16u;
      Xs[e] = rr < p.tokens ? X[rr * p.x_stride + kt * 16u + e % 16u] : half(0);
    }
    for (uint e = lid; e < 16u * 32u; e += 128u) {
      const uint cc = col0 + e % 32u;
      Ws[e] = cc < MPP_N ? W[(kt * 16u + e / 32u) * ws + cc] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // run overwrites cT; add each FP32 tile explicitly. The reduction inside
    // MPP differs from the SIMD reference and requires end-to-end validation.
    op.run(tA, tB, cT);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
      const auto idx = cT.get_multidimensional_index(i);
      const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
      acc[i] += cT[i];
      const uint blk = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
      const bool round_here =
          ((smode == 1u || smode == 2u) && kt == 0u && cc < 16u && ktiles > 1u) ||
          (smode == 3u && kt == 1u) ||
          (smode == 4u && (kt & 1u) == 1u && kt + 1u < ktiles) ||
          ((smode == 5u || smode == 6u) && cc < 16u && kt < srounds);
      if (round_here && !(smode == 2u && blk == 3u) && !(smode == 5u && (blk & 1u)))
        acc[i] = float(half(acc[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 16; ++i) {
    if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
    const auto idx = cT.get_multidimensional_index(i);
    const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
    if (rr >= p.tokens || cc >= limit) continue;
    const uint blk = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
    if (p.row_filter != 0u && (blk & 1u)) continue;
    half h = half(acc[i]);
    if (p.epilogue == 1u) h = max(h, half(0));
    else if (p.epilogue == 2u) h = mpp_gelu(h);
    else if (p.epilogue == 3u) h = h + Cin[rr * p.r_stride + cc];
    Y[rr * p.y_stride + cc] = h;
  }
}
#endif
