// Opt-in approximate full-K FP16 TensorOps GEMM. Original public API-only code.
// Same entry/16-u32 ABI as pwin_gemm_mpp. One TensorOps operation computes the
// complete dot product in FP32; original intermediate K16 FP16 handoffs are
// deliberately omitted, and the original real-unit bias is added afterward.
// Existing staged/direct/Reference/Batched implementations remain separate.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct GemmFastHalfParams {
  uint tokens, k, n, x_stride, y_stride, c_stride, epilogue, c_mode;
  uint original_split_ignored, width, shift_y, w_stride, store_cols, r_stride, row_filter, reserved;
};

static inline half fast_half_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

template <typename A, typename B>
static inline void fast_half_block(const thread A& a, const thread B& b,
                                   device const float* C, device half* Y,
                                   constant GemmFastHalfParams& p,
                                   device const half* Cin, device const half* Cbias,
                                   uint row0, uint col0, uint limit) {
  constexpr auto desc = matmul2d_descriptor(64, 32, int(dynamic_extent), false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto product = op.template get_destination_cooperative_tensor<A, B, float>();
  op.run(a, b, product);
#pragma clang loop unroll(full)
  for (uint16_t i = 0u; i < product.get_capacity(); ++i) {
    if (!product.is_valid_element(i)) continue;
    const auto index = product.get_multidimensional_index(i);
    const uint rr = row0 + uint(index[1]), cc = col0 + uint(index[0]);
    if (rr >= p.tokens || cc >= limit) continue;
    const uint block = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
    if (p.row_filter != 0u && (block & 1u)) continue;
    const float bias = p.c_mode == 0u ? C[(rr & 15u) * p.c_stride + cc]
                                    : float(half(Cin[rr * p.c_stride + cc] + Cbias[cc]));
    half h = half(product[i] + bias);
    if (p.epilogue == 1u) h = max(h, half(0));
    else if (p.epilogue == 2u) h = fast_half_gelu(h);
    else if (p.epilogue == 3u) h = h + Cin[rr * p.r_stride + cc];
    Y[rr * p.y_stride + cc] = h;
  }
}

// FC_K/N are declared by the preceding original public GEMM sources.
kernel void pwin_gemm_mpp(device const half* X [[buffer(0)]],
                           device const half* W [[buffer(1)]],
                           device const float* C [[buffer(2)]],
                           device half* Y [[buffer(3)]],
                           constant GemmFastHalfParams& p [[buffer(4)]],
                           device const half* Cin [[buffer(5)]],
                           device const half* Cbias [[buffer(6)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint3 threads [[threads_per_threadgroup]]) {
  const uint row0 = tg.x * 64u, col0 = tg.y * 32u;
  const uint ws = p.w_stride == 0u ? FC_N : p.w_stride;
  const uint limit = p.store_cols == 0u ? FC_N : p.store_cols;
  if (threads.x != 128u || threads.y != 1u || threads.z != 1u || tg.z ||
      p.k != FC_K || p.n != FC_N || FC_K == 0u || FC_K > 640u ||
      FC_N == 0u || FC_N > 768u || (FC_K & 15u) || (FC_N & 15u) ||
      p.width == 0u || p.x_stride < FC_K || ws < FC_N || limit > FC_N || p.y_stride < limit ||
      p.epilogue > 3u || p.c_mode > 1u || p.row_filter > 1u || p.reserved != 0u ||
      (p.c_mode == 1u && p.c_stride < FC_N) ||
      (p.c_mode == 0u && p.c_stride != 0u && p.c_stride < FC_N) ||
      (p.epilogue == 3u && p.r_stride < limit) || row0 >= p.tokens || col0 >= limit) return;
  using input = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
  // SDK input tensor element types are unqualified half; these views only read.
  auto full_a = input(const_cast<device half*>(X), dextents<int32_t, 2>{int32_t(FC_K), int32_t(p.tokens)},
                       array<int32_t, 2>{1, int32_t(p.x_stride)});
  auto full_b = input(const_cast<device half*>(W), dextents<int32_t, 2>{int32_t(FC_N), int32_t(FC_K)},
                       array<int32_t, 2>{1, int32_t(ws)});
  if (row0 + 64u <= p.tokens && col0 + 32u <= FC_N) {
    auto a = full_a.slice<dynamic_extent, 64>(0, row0);
    auto b = full_b.slice<32, dynamic_extent>(col0, 0);
    fast_half_block(a, b, C, Y, p, Cin, Cbias, row0, col0, limit);
  } else {
    // Dynamic extents let TensorOps bound partial token/N tiles without padding.
    auto a = full_a.slice(0, row0);
    auto b = full_b.slice(col0, 0);
    fast_half_block(a, b, C, Y, p, Cin, Cbias, row0, col0, limit);
  }
}
