// Fused MLP per 16-token block: hidden = act(fp16(be + X . We)) (16-wide hidden tiles, GELU in registers,
// never stored) feeding the contraction out = fp16(C + act(hidden) . Wc) with the fp16 handoff after every
// group of 32 hidden units and C = fp16(Cin + Cbias) per element. Numerics identical to the separate
// gemm3 dispatches (expand with split mode exp_split, contract with split mode 4).
//   exp_split: 0 none; 2 = hidden tile 0 rounded after K tile 0 for window blocks 0..2 (enc0/dec0);
//              5 = hidden tile 0 rounded after every K tile for window blocks 0 and 2 (enc1's 4-way split)
// Function constants: FC_K (0) input channels, FC_N (1) output channels, FC_XS (2) X stride, FC_CS (3) Cin
// stride, FC_YS (7) Y stride, FC_HD (14) hidden units (We row stride; Wc has FC_N per row).
#include <metal_stdlib>
using namespace metal;

constant uint FC_HD [[function_constant(14)]];
constant uint FC_WCS [[function_constant(15)]];   // optional Wc row stride (0 / undefined = FC_N): contraction over a column slice

struct Mlp2Params {
  uint tokens, exp_split, width, shift_y, x_scale, exp_rounds, pad2, pad3;
  // x_scale: 1 = A = fp16(X * Xs[col]) on load
  // exp_rounds: with exp_split 5, round hidden tile 0 after this many leading K tiles (0 = all but the last,
  // which is the n-way split of enc1/enc2; enc3 and dec3 split 1+1+1+1+2 over six tiles, so they pass 4)
};

static inline half mlp2_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

kernel void pwin_mlp2_tokens(device const half* X [[buffer(0)]],
                             device const half* We [[buffer(1)]],
                             device const float* Ce [[buffer(2)]],
                             device const half* Wc [[buffer(3)]],
                             device const half* Cin [[buffer(4)]],
                             device const half* Cbias [[buffer(5)]],
                             device half* Y [[buffer(6)]],
                             constant Mlp2Params& p [[buffer(7)]],
                             device const half* Xs [[buffer(8)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint simdgroups [[simdgroups_per_threadgroup]]) {
  const uint wcs = (is_function_constant_defined(FC_WCS) && FC_WCS != 0u) ? FC_WCS : FC_N;
  const uint row0 = (tg.x * simdgroups + simd_index) * 16u;
  if (row0 >= p.tokens) return;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  device const half* x = X + row0 * FC_XS;
  const uint ktiles = FC_K / 16u;
  uint rowv[2]; bool rok[2]; uint blk[2];
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    rowv[i] = row0 + 8u * i + er;
    rok[i] = rowv[i] < p.tokens;
    const uint rr = min(rowv[i], p.tokens - 1u);
    blk[i] = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
  }
  // input tiles resident when K <= 32 (enc0/dec0); streamed per hidden tile otherwise (enc1)
  simdgroup_float8x8 A[2][2][2];
  if (FC_K <= 32u) {
#pragma clang loop unroll(full)
    for (uint kt = 0u; kt < 2u; ++kt) {
      if (kt >= ktiles) break;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          simdgroup_load(hm, x + kt * 16u + j * 8u, FC_XS, ulong2(0, i * 8u));
          if (p.x_scale != 0u) {
            const half2 sv = *reinterpret_cast<device const half2*>(Xs + kt * 16u + j * 8u + ec);
            hm.thread_elements()[0] = hm.thread_elements()[0] * sv.x;
            hm.thread_elements()[1] = hm.thread_elements()[1] * sv.y;
          }
          A[kt][i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          A[kt][i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
        }
    }
  }
  // contraction accumulators: C = fp16(Cin + Cbias)
  simdgroup_float8x8 out[4][2][2];   // [n tile][i][j], FC_N <= 64
#pragma clang loop unroll(full)
  for (uint n = 0u; n < 4u; ++n) {
    if (n * 16u >= FC_N) break;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
      const uint rr = min(rowv[i], p.tokens - 1u);
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        const half2 cv = *reinterpret_cast<device const half2*>(Cin + rr * FC_CS + n * 16u + 8u * j + ec);
        const half2 cb = *reinterpret_cast<device const half2*>(Cbias + n * 16u + 8u * j + ec);
        out[n][i][j].thread_elements()[0] = float(cv.x + cb.x);
        out[n][i][j].thread_elements()[1] = float(cv.y + cb.y);
      }
    }
  }
  // hidden groups of 32 = two 16-wide hidden tiles sharing the A loads; the contraction is rounded after every group
  for (uint g0 = 0u; g0 < FC_HD; g0 += 32u) {
    simdgroup_float8x8 acc[2][2][2];   // [tile][i][j]
#pragma clang loop unroll(full)
    for (uint t = 0u; t < 2u; ++t)
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[t][i][j], Ce + g0 + 16u * t + j * 8u, FC_HD, ulong2(0, i * 8u));
#pragma clang loop unroll(full)
    for (uint kt = 0u; kt < ktiles; ++kt) {
      simdgroup_float8x8 a[2][2];
      if (FC_K <= 32u) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) a[i][j] = A[kt & 1u][i][j];
      } else {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_half8x8 hm;
            simdgroup_load(hm, x + kt * 16u + j * 8u, FC_XS, ulong2(0, i * 8u));
            if (p.x_scale != 0u) {
              const half2 sv = *reinterpret_cast<device const half2*>(Xs + kt * 16u + j * 8u + ec);
              hm.thread_elements()[0] = hm.thread_elements()[0] * sv.x;
              hm.thread_elements()[1] = hm.thread_elements()[1] * sv.y;
            }
            a[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            a[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          }
      }
#pragma clang loop unroll(full)
      for (uint t = 0u; t < 2u; ++t) {
        const uint h0 = g0 + 16u * t;
        simdgroup_float8x8 b[2][2];
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_half8x8 hm;
            simdgroup_load(hm, We + (kt * 16u + i * 8u) * FC_HD + h0 + j * 8u, FC_HD);
            b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          }
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_multiply_accumulate(acc[t][i][j], a[i][0], b[0][j], acc[t][i][j]);
            simdgroup_multiply_accumulate(acc[t][i][j], a[i][1], b[1][j], acc[t][i][j]);
          }
        const bool split2 = p.exp_split == 2u && kt == 0u && h0 == 0u && ktiles > 1u;
        const uint rounds = p.exp_rounds != 0u ? p.exp_rounds : ktiles - 1u;
        // 5: only each warp's first token block (blocks 0 and 2, the two-warp layers)
        // 6: every token block (the four-warp layers, where a warp owns a single block)
        const bool split5 = (p.exp_split == 5u || p.exp_split == 6u) && h0 == 0u && kt < rounds;
        if (split2 || split5) {
#pragma clang loop unroll(full)
          for (uint i = 0u; i < 2u; ++i) {
            if (split2 && blk[i] == 3u) continue;
            if (split5 && p.exp_split == 5u && (blk[i] & 1u) != 0u) continue;
#pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
              acc[t][i][j].thread_elements()[0] = float(half(acc[t][i][j].thread_elements()[0]));
              acc[t][i][j].thread_elements()[1] = float(half(acc[t][i][j].thread_elements()[1]));
            }
          }
        }
      }
    }
    // activation in registers; tile t becomes the A operand of contraction K tile (g0/16 + t)
#pragma clang loop unroll(full)
    for (uint t = 0u; t < 2u; ++t) {
      const uint h0 = g0 + 16u * t;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          acc[t][i][j].thread_elements()[0] = float(mlp2_gelu(half(acc[t][i][j].thread_elements()[0])));
          acc[t][i][j].thread_elements()[1] = float(mlp2_gelu(half(acc[t][i][j].thread_elements()[1])));
        }
#pragma clang loop unroll(full)
      for (uint n = 0u; n < 4u; ++n) {
        if (n * 16u >= FC_N) break;
        simdgroup_float8x8 b[2][2];
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_half8x8 hm;
            simdgroup_load(hm, Wc + (h0 + i * 8u) * wcs + n * 16u + j * 8u, wcs);
            b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          }
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            simdgroup_multiply_accumulate(out[n][i][j], acc[t][i][0], b[0][j], out[n][i][j]);
            simdgroup_multiply_accumulate(out[n][i][j], acc[t][i][1], b[1][j], out[n][i][j]);
          }
      }
    }
    // fp16 handoff after every group of 32 hidden units (the store rounds the last one)
    if (g0 + 32u < FC_HD) {
#pragma clang loop unroll(full)
      for (uint n = 0u; n < 4u; ++n) {
        if (n * 16u >= FC_N) break;
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            out[n][i][j].thread_elements()[0] = float(half(out[n][i][j].thread_elements()[0]));
            out[n][i][j].thread_elements()[1] = float(half(out[n][i][j].thread_elements()[1]));
          }
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint n = 0u; n < 4u; ++n) {
    if (n * 16u >= FC_N) break;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
      if (!rok[i]) continue;
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        const uint col = n * 16u + 8u * j + ec;
        *reinterpret_cast<device half2*>(Y + rowv[i] * FC_YS + col) =
            half2(half(out[n][i][j].thread_elements()[0]), half(out[n][i][j].thread_elements()[1]));
      }
    }
  }
}
