// Experimental native FP8 E4M3FN PWin GEMM. Original public API-only code.
// A is GPU-quantized per token row; W is privately packed per output column in
// transposed N,K order. FP32 dequantization precedes unscaled bias/epilogues.
// This candidate deliberately omits original intermediate K16 FP16 handoffs.
// It is approximate, not Reference/Batched arithmetic or DLSS 4.5.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct QuantizeRowsFp8Params {
  uint tokens, k, x_stride, a_stride_bytes, k_padded, rows_padded, reserved0, reserved1;
};

kernel void pwin_quantize_rows_e4m3(device const half* X [[buffer(0)]],
                                   device uchar* Aq [[buffer(1)]],
                                   device float* AScale [[buffer(2)]],
                                   constant QuantizeRowsFp8Params& p [[buffer(3)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint3 tg [[threadgroup_position_in_grid]],
                                   uint3 threads [[threads_per_threadgroup]]) {
  const uint row = tg.x * 4u + simd;
  if (threads.x != 128u || threads.y != 1u || threads.z != 1u || tg.y || tg.z ||
      p.tokens == 0u || p.k == 0u || p.k > 640u || p.x_stride < p.k ||
      p.k_padded < p.k || (p.k_padded & 31u) || p.a_stride_bytes < p.k_padded ||
      (p.a_stride_bytes & 127u) || p.rows_padded < p.tokens || (p.rows_padded & 63u) ||
      p.reserved0 || p.reserved1 || row >= p.rows_padded) return;
  float amax = 0.0f;
  uint bad = 0u;
  if (row < p.tokens) {
    for (uint k = lane; k < p.k; k += 32u) {
      const float v = float(X[row * p.x_stride + k]);
      amax = max(amax, abs(v));
      bad |= !isfinite(v);
    }
  }
  amax = simd_max(amax);
  bad = simd_max(bad);
  const float scale = bad != 0u ? NAN : (amax > 0.0f ? amax / 448.0f : 1.0f);
  if (lane == 0u) AScale[row] = scale;
  for (uint k = lane * 4u; k < p.a_stride_bytes; k += 128u) {
    float4 v(0.0f);
    if (row < p.tokens) {
      for (uint i = 0u; i < 4u; ++i)
        if (k + i < p.k) v[i] = float(X[row * p.x_stride + k + i]) / scale;
    }
    auto packed = pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even,
                       saturation_mode::saturate>(v);
    auto bytes = packed.as_storage_type();
    for (uint i = 0u; i < 4u; ++i) Aq[row * p.a_stride_bytes + k + i] = bytes[i];
  }
}

struct GemmFp8Params {
  uint tokens, k, n, a_stride_bytes, y_stride, c_stride, epilogue, c_mode;
  uint original_split_ignored, width, shift_y, w_stride_bytes, store_cols, r_stride, row_filter, k_padded;
};

static inline half fp8_pwin_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

// FC_K/N come from the original public GEMM sources in the combined library.
kernel void pwin_gemm_fp8_e4m3(device uchar* Aq [[buffer(0)]],
                               device uchar* Wq [[buffer(1)]],
                               device const float* C [[buffer(2)]],
                               device half* Y [[buffer(3)]],
                               constant GemmFp8Params& p [[buffer(4)]],
                               device const half* Cin [[buffer(5)]],
                               device const half* Cbias [[buffer(6)]],
                               device const float* AScale [[buffer(7)]],
                               device const float* WScale [[buffer(8)]],
                               uint3 tg [[threadgroup_position_in_grid]],
                               uint3 threads [[threads_per_threadgroup]]) {
  const uint row0 = tg.x * 64u, col0 = tg.y * 32u;
  const uint limit = p.store_cols == 0u ? FC_N : p.store_cols;
  if (threads.x != 128u || threads.y != 1u || threads.z != 1u || tg.z ||
      p.k != FC_K || p.n != FC_N || FC_K == 0u || FC_K > 640u ||
      FC_N == 0u || FC_N > 768u || p.k_padded < FC_K || (p.k_padded & 31u) ||
      p.a_stride_bytes < p.k_padded || (p.a_stride_bytes & 127u) ||
      p.w_stride_bytes < p.k_padded || (p.w_stride_bytes & 127u) ||
      p.width == 0u || limit > FC_N || p.y_stride < limit || p.epilogue > 3u ||
      p.c_mode > 1u || p.row_filter > 1u || row0 >= p.tokens || col0 >= limit) return;
  // Rows/columns are explicitly padded by the quantizer/private pack. Full
  // outer tiles are valid, including tokens/N tails; only valid outputs store.
  using input = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
  const uint rows_padded = ((p.tokens + 63u) / 64u) * 64u;
  const uint cols_padded = ((FC_N + 31u) / 32u) * 32u;
  auto full_a = input(Aq, dextents<int32_t, 2>{int32_t(p.k_padded), int32_t(rows_padded)},
                        array<int32_t, 2>{1, int32_t(p.a_stride_bytes)});
  auto full_b = input(Wq, dextents<int32_t, 2>{int32_t(p.k_padded), int32_t(cols_padded)},
                        array<int32_t, 2>{1, int32_t(p.w_stride_bytes)});
  auto a = full_a.slice<dynamic_extent, 64>(0, row0);
  auto b = full_b.slice<dynamic_extent, 32>(0, col0);
  constexpr auto desc = matmul2d_descriptor(64, 32, int(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto product = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  op.run(a, b, product);
#pragma clang loop unroll(full)
  for (uint16_t i = 0u; i < product.get_capacity(); ++i) {
    if (!product.is_valid_element(i)) continue;
    const auto idx = product.get_multidimensional_index(i);
    const uint rr = row0 + uint(idx[1]), cc = col0 + uint(idx[0]);
    if (rr >= p.tokens || cc >= limit) continue;
    const uint blk = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
    if (p.row_filter != 0u && (blk & 1u)) continue;
    const float bias = p.c_mode == 0u ? C[(rr & 15u) * p.c_stride + cc]
                                    : float(half(Cin[rr * p.c_stride + cc] + Cbias[cc]));
    const float value = (product[i] * AScale[rr]) * WScale[cc] + bias;
    half h = half(value);
    if (p.epilogue == 1u) h = max(h, half(0));
    else if (p.epilogue == 2u) h = fp8_pwin_gelu(h);
    else if (p.epilogue == 3u) h = h + Cin[rr * p.r_stride + cc];
    Y[rr * p.y_stride + cc] = h;
  }
}
