#include <metal_stdlib>
// (oracle header not needed by the fast pass; the repair kernels carry the exact helpers)
using namespace metal;

// block1 conv2 (module 64, launches 74-81): 160x92, 32 -> 32, two K=16 stages per tap. Design six
// (custom_block1_conv2_bounded_v6.metal) plus one micro-optimisation, the packed weight/proxy table of
// conv2 design thirteen (custom_conv2_bounded_v13.metal):
//   - the host builds once, from the same weight buffer, a half4 table indexed (tile, stage, kb, oc-pair,
//     matrix_row) holding (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) with q(x) = x != 0 ? max(|x|, 2^-14) : 0
//     (== bound_operand in fp32 after the exact half->float widening), so a lane loads one half4 per (kb, g)
//     per stage (4 loads) instead of 8 gathered halves through widx plus 8 bound_operand()s.
//     Layout: for fixed (tile, stage, kb, g) the 32 lanes of a simdgroup read 32 consecutive half4 (256 B):
//     pair = group.y*8 + 4g + matrix_column/2 covers 4 consecutive pairs, matrix_row the 8 rows inside each.
//     Table size 9*2*2*16*8 = 4608 half4 = 36864 B (every one of the 4*2304 weights appears exactly once).
// Everything else (a loads, proxy on a, MMAs, fold, bound with the 2^-21 floor, checkpoints, epilogue) is v6.

struct Params { uint residual; };
constant bool FORMAL [[function_constant(0)]];
// HW: hardware lowering - no bound matrices, no flags, no repair; the fast pass alone (PSNR-gated, not exact).
constant bool HW [[function_constant(1)]];

inline uint widx(uint tile, uint stage, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u;
  uint lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 2304u + tile * 256u + stage * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}
inline uint wq_index(uint tile, uint stage, uint kb, uint pair, uint row) {
  return (((tile * 2u + stage) * 2u + kb) * 16u + pair) * 8u + row;
}
inline ushort zero_norm(half h) { const ushort b = as_type<ushort>(h); return b == 0x8000u ? ushort(0) : b; }
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}

kernel void fg_custom_block1_conv2_bounded_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* input [[buffer(2)]],
    device const half* residual [[buffer(3)]], device half* out [[buffer(4)]], constant Params& p [[buffer(5)]],
    device uchar* flag_count [[buffer(6)]], device uchar* resume_stage [[buffer(7)]], device half* resume_acc [[buffer(8)]],
    device half* fast_after [[buffer(9)]],
    device uchar* stage2 [[buffer(10)]], device half* prior2 [[buffer(11)]], device half* after2 [[buffer(12)]],
    device const half4* wq [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 element = matrix_element(lane);
  const uint matrix_row = element.x, matrix_column = element.y;
  const uint pixel = group.x * 8u + matrix_row, oc0 = group.y * 16u + matrix_column;
  if (pixel >= 160u * 92u || oc0 >= 32u) return;
  const uint px = pixel % 160u, py = pixel / 160u;

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
    const bool ok = ix >= 0 && ix < 160 && iy >= 0 && iy < 92;
    const uint base = ok ? (uint(iy) * 160u + uint(ix)) * 32u : 0u;
    for (uint stage = 0u; stage < 2u; ++stage) {
      simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb) {
        const uint ic_a = stage * 16u + kb * 8u + matrix_column;
        const float a0 = ok ? float(input[base + ic_a]) : 0.0f, a1 = ok ? float(input[base + ic_a + 1u]) : 0.0f;
        a[kb].thread_elements()[0] = a0; a[kb].thread_elements()[1] = a1;
        qa[kb].thread_elements()[0] = FORMAL ? bound_operand(a0) : fabs(a0); qa[kb].thread_elements()[1] = FORMAL ? bound_operand(a1) : fabs(a1);
#pragma clang loop unroll(full)
        for (uint g = 0u; g < 2u; ++g) {
          const half4 t = wq[wq_index(tile, stage, kb, (oc0 + 8u * g) >> 1u, matrix_row)];
          const float b0 = float(t.x), b1 = float(t.y);
          b[g][kb].thread_elements()[0] = b0; b[g][kb].thread_elements()[1] = b1;
          qb[g][kb].thread_elements()[0] = FORMAL ? float(t.z) : fabs(b0); qb[g][kb].thread_elements()[1] = FORMAL ? float(t.w) : fabs(b1);
        }
      }
      const uint stage_index = tile * 2u + stage;
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        if (HW) {   // hardware lowering: accumulate straight into the tile, round the stage to half
          simdgroup_multiply_accumulate(acc[g], a[0], b[g][0], acc[g]); simdgroup_multiply_accumulate(acc[g], a[1], b[g][1], acc[g]);
          acc[g].thread_elements()[0] = float(half(acc[g].thread_elements()[0])); acc[g].thread_elements()[1] = float(half(acc[g].thread_elements()[1]));
          continue;
        }
        const float prior0 = acc[g].thread_elements()[0], prior1 = acc[g].thread_elements()[1];
        simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), bnd0 = z, bnd1 = z, part0 = z, part1 = z;
        if (!HW) { simdgroup_multiply_accumulate(bnd0, qa[0], qb[g][0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[g][1], bnd1); }
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
            const float M = max(0x1p-21f, max(max(c0, c1), fabs(prior)));   // the oracle's exponent floor: max_exp >= -21 even when every term is smaller
            error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.0f * ulp32_bits(M)   /* n*delta: at most n nonzero products lose a grid unit */ + 1.5625f * ulp32_bits(value);
          } else {
            error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
          }
          half rounded = half(value); if (as_type<ushort>(rounded) == 0x8000u) rounded = half(0.0h);   // rounds-to-zero is +0 (RTX)
          if (!HW && zero_norm(half(value - error)) != zero_norm(half(value + error))) {
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
    const uint index = (py * 160u + px) * 32u + oc0 + 8u * g + h;
    const half r = p.residual != 0u ? residual[index] : half(0.0f);
    out[index] = half(v + r);
    if (!HW) { flag_count[index] = flags[g][h]; resume_stage[index] = first_stage[g][h]; resume_acc[index] = resume_val[g][h]; fast_after[index] = after_val[g][h]; stage2[index] = second_stage[g][h]; prior2[index] = prior2_val[g][h]; after2[index] = after2_val[g][h]; }
  }
}
