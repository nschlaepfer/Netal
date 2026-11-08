// Experimental FP16 TensorOps fused MLP for the six public PWin MLP2 sites.
// This is mpp-half, not FP8. Weights and the 8-u32/buffer/function-constant ABI
// remain unchanged. A group owns 64 tokens; dispatch ceil(tokens/64), 128 threads.
// Scaled A (8 KiB) and one hidden group (4 KiB) stay on chip. TensorOps' FP32
// reduction order may differ from the SIMD implementation; this is approximate.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// FC_K/N/XS/CS/YS are declared by the preceding public GEMM sources.
constant uint FC_HD [[function_constant(14)]];
constant uint FC_WCS [[function_constant(15)]];

struct Mlp2MppParams {
  uint tokens, exp_split, width, shift_y, x_scale, exp_rounds, pad2, pad3;
};

static inline half mpp_mlp_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

using mlp_a_tensor = tensor<threadgroup half, extents<int32_t, 16, 64>, tensor_inline>;
using mlp_h_tensor = tensor<threadgroup half, extents<int32_t, 32, 64>, tensor_inline>;
using mlp_we_tensor = tensor<device half, extents<int32_t, 32, 16>, tensor_inline>;
using mlp_wc_tensor = tensor<device half, extents<int32_t, 32, 32>, tensor_inline>;

kernel void pwin_mlp2_tokens(device const half* X [[buffer(0)]],
                             device const half* We [[buffer(1)]],
                             device const float* Ce [[buffer(2)]],
                             device const half* Wc [[buffer(3)]],
                             device const half* Cin [[buffer(4)]],
                             device const half* Cbias [[buffer(5)]],
                             device half* Y [[buffer(6)]],
                             constant Mlp2MppParams& p [[buffer(7)]],
                             device const half* Xs [[buffer(8)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint3 threads [[threads_per_threadgroup]]) {
  const uint row0 = tg.x * 64u;
  const uint wcs = (is_function_constant_defined(FC_WCS) && FC_WCS != 0u) ? FC_WCS : FC_N;
  const bool known_shape = (FC_K == 32u && FC_HD == 128u && FC_N == 32u) ||
                           (FC_K == 64u && FC_HD == 256u && FC_N == 64u);
  if (!known_shape || threads.x != 128u || threads.y != 1u || threads.z != 1u ||
      tg.y != 0u || tg.z != 0u || p.width == 0u || p.x_scale > 1u ||
      (p.exp_split != 0u && p.exp_split != 2u && p.exp_split != 5u && p.exp_split != 6u) ||
      p.exp_rounds >= FC_K / 16u || p.pad2 != 0u || p.pad3 != 0u ||
      FC_XS < FC_K || FC_CS < FC_N || FC_YS < FC_N || wcs < FC_N || row0 >= p.tokens) return;

  threadgroup half A[64 * 64];
  threadgroup half H[64 * 32];
  for (uint e = lid; e < 64u * FC_K; e += 128u) {
    const uint rr = row0 + e / FC_K, kk = e % FC_K;
    half v = rr < p.tokens ? X[rr * FC_XS + kk] : half(0);
    if (p.x_scale != 0u) v = v * Xs[kk];
    A[e] = v;  // explicit FP16 scale result, before any matrix product
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto expand_desc = matmul2d_descriptor(64, 32, 16, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto contract_desc = matmul2d_descriptor(64, 32, 32, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<expand_desc, execution_simdgroups<4>> expand_op;
  matmul2d<contract_desc, execution_simdgroups<4>> contract_op;
  auto a0 = mlp_a_tensor(A, extents<int32_t, 16, 64>{}, array<int32_t, 2>{1, int32_t(FC_K)});
  auto we0 = mlp_we_tensor(const_cast<device half*>(We), extents<int32_t, 32, 16>{},
                           array<int32_t, 2>{1, int32_t(FC_HD)});
  auto ht = mlp_h_tensor(H, extents<int32_t, 32, 64>{}, array<int32_t, 2>{1, 32});
  auto wc0 = mlp_wc_tensor(const_cast<device half*>(Wc), extents<int32_t, 32, 32>{},
                           array<int32_t, 2>{1, int32_t(wcs)});
  // TensorOps input element types are unqualified half; these input views are
  // only read. No device weight/input buffer is modified by the kernel.
  auto expanded = expand_op.template get_destination_cooperative_tensor<decltype(a0), decltype(we0), float>();
  auto out0 = contract_op.template get_destination_cooperative_tensor<decltype(ht), decltype(wc0), float>();
  auto out1 = contract_op.template get_destination_cooperative_tensor<decltype(ht), decltype(wc0), float>();
  for (uint n = 0u; n < FC_N / 32u; ++n) {
    thread auto& out = n == 0u ? out0 : out1;
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= out.get_capacity() || !out.is_valid_element(i)) continue;
      const auto idx = out.get_multidimensional_index(i);
      const uint rr = row0 + uint(idx[1]), cc = n * 32u + uint(idx[0]);
      out[i] = rr < p.tokens ? float(half(Cin[rr * FC_CS + cc] + Cbias[cc])) : 0.0f;
    }
  }

  const uint ktiles = FC_K / 16u;
  const uint rounds = p.exp_rounds != 0u ? p.exp_rounds : ktiles - 1u;
  for (uint g0 = 0u; g0 < FC_HD; g0 += 32u) {
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= expanded.get_capacity() || !expanded.is_valid_element(i)) continue;
      const auto idx = expanded.get_multidimensional_index(i);
      expanded[i] = Ce[(uint(idx[1]) & 15u) * FC_HD + g0 + uint(idx[0])];
    }
    for (uint kt = 0u; kt < ktiles; ++kt) {
      auto at = mlp_a_tensor(A + kt * 16u, extents<int32_t, 16, 64>{},
                              array<int32_t, 2>{1, int32_t(FC_K)});
      auto wet = mlp_we_tensor(const_cast<device half*>(We) + kt * 16u * FC_HD + g0,
                                extents<int32_t, 32, 16>{}, array<int32_t, 2>{1, int32_t(FC_HD)});
      expand_op.run(at, wet, expanded);
#pragma clang loop unroll(full)
      for (uint16_t i = 0; i < 16; ++i) {
        if (i >= expanded.get_capacity() || !expanded.is_valid_element(i)) continue;
        const auto idx = expanded.get_multidimensional_index(i);
        const uint cc = g0 + uint(idx[0]);
        const uint rr = min(row0 + uint(idx[1]), p.tokens - 1u);
        const uint blk = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
        const bool split2 = p.exp_split == 2u && kt == 0u && cc < 16u && ktiles > 1u;
        const bool split56 = (p.exp_split == 5u || p.exp_split == 6u) && cc < 16u && kt < rounds;
        if ((split2 && blk != 3u) || (split56 && (p.exp_split == 6u || (blk & 1u) == 0u)))
          expanded[i] = float(half(expanded[i]));
      }
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= expanded.get_capacity() || !expanded.is_valid_element(i)) continue;
      const auto idx = expanded.get_multidimensional_index(i);
      H[uint(idx[1]) * 32u + uint(idx[0])] = mpp_mlp_gelu(half(expanded[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = 0u; n < FC_N / 32u; ++n) {
    thread auto& out = n == 0u ? out0 : out1;
      auto wct = mlp_wc_tensor(const_cast<device half*>(Wc) + g0 * wcs + n * 32u,
                                extents<int32_t, 32, 32>{}, array<int32_t, 2>{1, int32_t(wcs)});
      contract_op.run(ht, wct, out);
      if (g0 + 32u < FC_HD) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < 16; ++i) {
          if (i >= out.get_capacity() || !out.is_valid_element(i)) continue;
          out[i] = float(half(out[i]));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint n = 0u; n < FC_N / 32u; ++n) {
    thread auto& out = n == 0u ? out0 : out1;
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= out.get_capacity() || !out.is_valid_element(i)) continue;
      const auto idx = out.get_multidimensional_index(i);
      const uint rr = row0 + uint(idx[1]), cc = n * 32u + uint(idx[0]);
      if (rr < p.tokens) Y[rr * FC_YS + cc] = half(out[i]);
    }
  }
}
