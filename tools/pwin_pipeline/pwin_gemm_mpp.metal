// Batched per-token GEMM on Metal Performance Primitives: one threadgroup (4 SIMD groups) computes
// 64 tokens x N. The X tile (64 x K) and the weights (K x N) are staged once into threadgroup
// memory as inline tensors (the configuration that reached 12.6 TFLOPS in the microbenchmark);
// matmul2d accumulates in a cooperative fp32 tensor; epilogues run per element with the element's
// (n, token) coordinates. Rounding points match the hw3-order kernels (fp32 accumulate, one fp16
// rounding, split-K fp16 handoff on N tile 0 as two K=16 runs, relu / GELU), but the fp32 summation
// order inside a run is MPP's, so results can differ from `_hw` by rounding.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint FC_K [[function_constant(0)]];
constant uint FC_N [[function_constant(1)]];

struct GemmMppParams {
  uint tokens, k, n, x_stride, y_stride, c_stride, epilogue, c_mode;
  uint split_k16, width, shift_y, w_stride, pad1, pad2, pad3, pad4;
};

static inline half mpp_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

using tg_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

// one 64 x 32 output block with a K chunk size of KK. matmul2d::run OVERWRITES the cooperative tensor
// (measured), so chunks are accumulated manually in fp32 and C is added at the end.
template <int KK>
static inline void block_kk(threadgroup half* Xs, threadgroup half* Ws, device const float* C, device half* Y,
                            constant GemmMppParams& p, device const half* Cin, device const half* Cbias,
                            int row0, int col0, bool split_round) {
  constexpr auto desc = matmul2d_descriptor(64, 32, KK, false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto tA0 = tg_tensor(Xs, dextents<int32_t, 2>(KK, 64), array<int32_t, 2>{1, int(FC_K)});
  auto tB0 = tg_tensor(Ws + col0, dextents<int32_t, 2>(32, KK), array<int32_t, 2>{1, int(FC_N)});
  auto cT = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
  float acc[16];
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 16; ++i) acc[i] = 0.0f;
  for (int k0 = 0; k0 < int(FC_K); k0 += KK) {
    auto tA = tg_tensor(Xs + k0, dextents<int32_t, 2>(KK, 64), array<int32_t, 2>{1, int(FC_K)});
    auto tB = tg_tensor(Ws + k0 * int(FC_N) + col0, dextents<int32_t, 2>(32, KK), array<int32_t, 2>{1, int(FC_N)});
    op.run(tA, tB, cT);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) {
      if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
      acc[i] += cT[i];
    }
    if (split_round && k0 == 0) {
#pragma clang loop unroll(full)
      for (uint16_t i = 0; i < 16; ++i) {
        if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
        const auto idx = cT.get_multidimensional_index(i);
        if (col0 + int(idx[0]) >= 16) continue;
        const int row = min(row0 + int(idx[1]), int(p.tokens) - 1);
        const uint block = ((uint(row) / p.width + p.shift_y) & 7u) >> 1u;
        // the transpiled kernel rounds (C + partial sum) to fp16 here; C is added below, so include it now
        const int c = col0 + int(idx[0]);
        const float cval = p.c_mode == 0u ? C[c] : float(half(Cin[row * int(p.c_stride) + c] + Cbias[c]));
        if (p.split_k16 == 1u || block != 3u) acc[i] = float(half(acc[i] + cval)) - cval;
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < 16; ++i) {
    if (i >= cT.get_capacity() || !cT.is_valid_element(i)) continue;
    const auto idx = cT.get_multidimensional_index(i);
    const int c = col0 + int(idx[0]);
    const int rr = row0 + int(idx[1]);
    if (rr >= int(p.tokens)) continue;
    const float cval = p.c_mode == 0u ? C[c] : float(half(Cin[rr * int(p.c_stride) + c] + Cbias[c]));
    half h = half(acc[i] + cval);
    if (p.epilogue == 1u) h = max(h, half(0.0h));
    else if (p.epilogue == 2u) h = mpp_gelu(h);
    Y[rr * int(p.y_stride) + c] = h;
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
                          uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half Xs[64 * 128];
  threadgroup half Ws[8192];
  const int row0 = int(tg.x) * 64;
  if (row0 >= int(p.tokens)) return;
  const uint ws = p.w_stride == 0u ? FC_N : p.w_stride;
  for (uint e = lid; e < FC_K * FC_N; e += 128u) {
    const uint kk = e / FC_N, nn = e - kk * FC_N;
    Ws[e] = W[kk * ws + nn];
  }
  for (uint e = lid; e < 64u * FC_K; e += 128u) {
    const uint r = e / FC_K, kk = e - r * FC_K;
    const uint row = min(uint(row0) + r, p.tokens - 1u);
    Xs[e] = X[row * p.x_stride + kk];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int col0 = 0; col0 < int(FC_N); col0 += 32) {
    const bool split = p.split_k16 != 0u && col0 == 0 && FC_K > 16u;
    if (FC_K == 16u || split) block_kk<16>(Xs, Ws, C, Y, p, Cin, Cbias, row0, col0, split);
    else block_kk<32>(Xs, Ws, C, Y, p, Cin, Cbias, row0, col0, false);
  }
}
