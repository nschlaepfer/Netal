#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// conv2 family, guarded fast pass, second design. Changes from custom_conv2_bounded.metal, each measured
// on live weights (tools/xlate/fg-custom/RESULTS.md, "second design"):
//   - ulp32 by exponent bits instead of frexp/exp2 (identical for normals; products of two halves are
//     never fp32 subnormal, and a zero bound means zero error, so it is exact everywhere reachable);
//   - a formally sound bound (FORMAL=1, default) covering the fp32 FMA chain (7 roundings of a K=8 chain
//     seeded at zero - the first FMA is exact), NVIDIA's per-term truncation (17 terms x < 1 grid unit,
//     grid 2^(max_exp-25), 2^max_exp <= M := max(bnd0, bnd1, |prior|)), and the double-float combination
//     plus interval endpoints (1.5 ulp32 of the value). The bound matrices carry an exponent proxy -
//     nonzero operands clamped up to 2^-14 - because the oracle gives fp16 subnormals exponent -14
//     whatever their magnitude. FORMAL=0 is the previous 4/4/2 bound: 12% faster, exact on everything
//     measured, not a proof;
//   - per element, besides the first flagged stage and the accumulator going into it, the half the fast
//     pass produced at that stage and the number of flagged stages, so the repair can settle 90% of the
//     flagged elements with one exact stage (custom_conv2_repair.metal).

struct Params { uint residual; };
constant bool FORMAL [[function_constant(0)]];

inline uint widx(uint tile, uint stage, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u;
  uint lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 4608u + tile * 512u + stage * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}

kernel void fg_custom_block0_conv2_bounded_v16(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* input [[buffer(2)]],
    device const half* residual [[buffer(3)]], device half* out [[buffer(4)]], constant Params& p [[buffer(5)]],
    device uchar* flag_count [[buffer(6)]], device uchar* resume_stage [[buffer(7)]], device half* resume_acc [[buffer(8)]],
    device half* fast_after [[buffer(9)]],
    device uchar* stage2 [[buffer(10)]], device half* prior2 [[buffer(11)]], device half* after2 [[buffer(12)]],
    device const float2* wstats [[buffer(13)]],   // [tile][stage][kb][oc]: (unused, max exponent of proxied |w| over the block's 8 k; -1000 if all zero)
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 element = matrix_element(lane);
  const uint matrix_row = element.x, matrix_column = element.y;
  const uint pixel = group.x * 8u + matrix_row, oc0 = group.y * 16u + matrix_column;
  if (pixel >= 80u * 46u || oc0 >= 64u) return;
  const uint px = pixel % 80u, py = pixel / 80u;

  simdgroup_float8x8 acc[2];
  uchar flags[2][2] = {{0u, 0u}, {0u, 0u}}, first_stage[2][2] = {{255u, 255u}, {255u, 255u}}, second_stage[2][2] = {{255u, 255u}, {255u, 255u}};
  half resume_val[2][2], after_val[2][2], prior2_val[2][2], after2_val[2][2];
#pragma clang loop unroll(full)
  for (uint g = 0u; g < 2u; ++g) {
    const uint oc = oc0 + 8u * g;
    acc[g].thread_elements()[0] = float(bias[oc]); acc[g].thread_elements()[1] = float(bias[oc + 1u]);
    resume_val[g][0] = resume_val[g][1] = after_val[g][0] = after_val[g][1] = prior2_val[g][0] = prior2_val[g][1] = after2_val[g][0] = after2_val[g][1] = half(0.0h);
  }

  for (uint tile = 0u; tile < 9u; ++tile) {
    const int ix = int(px) + int(tile % 3u) - 1, iy = int(py) + int(tile / 3u) - 1;
    const bool ok = ix >= 0 && ix < 80 && iy >= 0 && iy < 46;
    const uint base = ok ? (uint(iy) * 80u + uint(ix)) * 64u : 0u;
    for (uint stage = 0u; stage < 4u; ++stage) {
      simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb) {
        const uint ic_a = stage * 16u + kb * 8u + matrix_column, ic_b = stage * 16u + kb * 8u + matrix_row;
        const float a0 = ok ? float(input[base + ic_a]) : 0.0f, a1 = ok ? float(input[base + ic_a + 1u]) : 0.0f;
        a[kb].thread_elements()[0] = a0; a[kb].thread_elements()[1] = a1;
        qa[kb].thread_elements()[0] = FORMAL ? bound_operand(a0) : fabs(a0); qa[kb].thread_elements()[1] = FORMAL ? bound_operand(a1) : fabs(a1);
#pragma clang loop unroll(full)
        for (uint g = 0u; g < 2u; ++g) {
          const uint oc = oc0 + 8u * g;
          const float b0 = float(w[widx(tile, stage, ic_b, oc)]), b1 = float(w[widx(tile, stage, ic_b, oc + 1u)]);
          b[g][kb].thread_elements()[0] = b0; b[g][kb].thread_elements()[1] = b1;
          qb[g][kb].thread_elements()[0] = FORMAL ? bound_operand(b0) : fabs(b0); qb[g][kb].thread_elements()[1] = FORMAL ? bound_operand(b1) : fabs(b1);
        }
      }
      // exponent estimate for the truncation grid: max exponent of a over this row's 8 k per K=8 block
      float aexp[2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb) {
        float m = max(qa[kb].thread_elements()[0], qa[kb].thread_elements()[1]);
        m = max(m, simd_shuffle_xor(m, 1u)); m = max(m, simd_shuffle_xor(m, 8u));
        aexp[kb] = m != 0.0f ? float(int((as_type<uint>(m) >> 23u) & 0xFFu) - 127) : -1000.0f;
      }
      const uint stage_index = tile * 4u + stage;
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        const float prior0 = acc[g].thread_elements()[0], prior1 = acc[g].thread_elements()[1];
        const uint sbase = ((tile * 4u + stage) * 2u) * 64u + oc0 + 8u * g;   // oc0 already carries matrix_column (review fix)
        const float eb00 = wstats[sbase].y, eb01 = wstats[sbase + 1u].y, eb10 = wstats[sbase + 64u].y, eb11 = wstats[sbase + 65u].y;
        simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), bnd0 = z, bnd1 = z, part0 = z, part1 = z;
        simdgroup_multiply_accumulate(bnd0, qa[0], qb[g][0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[g][1], bnd1);
        simdgroup_multiply_accumulate(part0, a[0], b[g][0], part0); simdgroup_multiply_accumulate(part1, a[1], b[g][1], part1);
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const float prior = h == 0u ? prior0 : prior1;
          const float t0 = part0.thread_elements()[h], t1 = part1.thread_elements()[h];
          // two exact two_sums, errors added once: prior + t0 + t1 = s2 + (e0 + e1) up to one rounding of e0 + e1
          const float s1 = prior + t0, bp1 = s1 - prior, e0 = (prior - (s1 - bp1)) + (t0 - bp1);
          const float s2 = s1 + t1, bp2 = s2 - s1, e1 = (s1 - (s2 - bp2)) + (t1 - bp2);
          const float value = s2 + (e0 + e1);
          float error;
          if (FORMAL) {
            const float c0 = bnd0.thread_elements()[h] * (1.0f + 0x1p-20f), c1 = bnd1.thread_elements()[h] * (1.0f + 0x1p-20f);
            // truncation grid from exponents: max_exp <= max(-21, e(prior), eA0 + eB0, eA1 + eB1); 17 terms x < 1 grid unit
            const float pe = prior != 0.0f ? float(int((as_type<uint>(fabs(prior)) >> 23u) & 0xFFu) - 127) : -1000.0f;
            const float ehat = max(max(-21.0f, pe), max(aexp[0] + (h == 0u ? eb00 : eb01), aexp[1] + (h == 0u ? eb10 : eb11)));
            const float grid = as_type<float>(uint(int(ehat) - 25 + 127) << 23u);
            error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 17.0f * grid + 1.5625f * ulp32_bits(value);
          } else {
            error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
          }
          const half rounded = half(value);
          if (as_type<ushort>(half(value - error)) != as_type<ushort>(half(value + error))) {
            if (flags[g][h] == 0u) { first_stage[g][h] = uchar(stage_index); resume_val[g][h] = half(prior); after_val[g][h] = rounded; }
            else if (flags[g][h] == 1u) { second_stage[g][h] = uchar(stage_index); prior2_val[g][h] = half(prior); after2_val[g][h] = rounded; }
            flags[g][h] = uchar(min(uint(flags[g][h]) + 1u, 255u));
          }
          acc[g].thread_elements()[h] = float(rounded);
        }
      }
    }
  }
  const half slope = half(0.01f);
#pragma clang loop unroll(full)
  for (uint g = 0u; g < 2u; ++g)
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    half v = half(acc[g].thread_elements()[h]);
    v = min(max(max(v, half(v * slope)), half(-6.0f)), half(6.0f));
    const uint index = (py * 80u + px) * 64u + oc0 + 8u * g + h;
    const half r = p.residual != 0u ? residual[index] : half(0.0f);
    out[index] = half(v + r);
    flag_count[index] = flags[g][h]; resume_stage[index] = first_stage[g][h]; resume_acc[index] = resume_val[g][h]; fast_after[index] = after_val[g][h]; stage2[index] = second_stage[g][h]; prior2[index] = prior2_val[g][h]; after2[index] = after2_val[g][h];
  }
}
