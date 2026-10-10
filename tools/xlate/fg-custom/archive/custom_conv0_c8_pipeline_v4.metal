#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_lanes_fast.metalh"
using namespace metal;

// block0 conv0-c8 (module 55, launch 58), design seven: design six (the v3 file) with pass 2 lane-parallel, 16 lanes per element
// (custom_conv2_repair_v6.metal pass 2 / hmma_exact_lanes_fast.metalh); every other kernel is the v3 kernel verbatim, renamed.
// Original v3 note: custom_conv0_c8_pipeline_v2.metal plus the deltas
// measured on conv2 (custom_conv2_bounded_v6.metal / custom_conv2_repair_v5.metal):
//   - two 8-channel output groups per simdgroup (16 output channels) sharing the activation matrices;
//     grid y is channels / 16;
//   - a second shortcut slot per element: the stage, prior and fast half of the second flagged stage;
//   - the shorter exact fold (two two_sums, errors added once) with the value slack 1.5 -> 1.5625
//     (FORMAL) and 2.0 -> 2.0625 (empirical); the rest of both bounds unchanged;
//   - pass 1 settles elements with one or two flagged stages before the full replay; both repair kernels
//     use the clz helper (hmma_exact_fast.metalh, same arithmetic as the oracle header) and load the 16
//     activation halves of a stage with four ushort4 loads (32 halves per pixel, 16 per stage: the stage
//     start is 32-byte aligned). The pool kernel is the v2 pool verbatim, renamed so this file is
//     self-contained.

constant bool FORMAL [[function_constant(0)]];

inline uint wi(uint tile, uint stage, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 2304u + tile * 256u + stage * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline half activate(half a) { const half slope = half(0.01f); return min(max(max(a, half(a * slope)), half(-6.0f)), half(6.0f)); }

// --- fast pass: SIMD conv at 320x184 with the guard, 16 output channels per simdgroup ------------------
kernel void fg_custom_block0_conv0_c8_bounded_v4(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device uchar* flag_count [[buffer(4)]],
    device uchar* resume_stage [[buffer(5)]], device half* resume_acc [[buffer(6)]],
    device half* fast_after [[buffer(7)]],
    device uchar* stage2 [[buffer(8)]], device half* prior2 [[buffer(9)]], device half* after2 [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc0 = group.y * 16u + mc;
  if (pixel >= 320u * 184u || oc0 >= 32u) return;
  const uint px = pixel % 320u, py = pixel / 320u;
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
    const bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
    const uint base = ok ? (uint(iy) * 320u + uint(ix)) * 32u : 0u;
    for (uint stage = 0u; stage < 2u; ++stage) {
      simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const float av = ok ? float(in[base + stage * 16u + kb * 8u + mc + h]) : 0.0f;
          a[kb].thread_elements()[h] = av;
          qa[kb].thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
#pragma clang loop unroll(full)
          for (uint g = 0u; g < 2u; ++g) {
            const float bv = float(w[wi(tile, stage, stage * 16u + kb * 8u + mr, oc0 + 8u * g + h)]);
            b[g][kb].thread_elements()[h] = bv;
            qb[g][kb].thread_elements()[h] = FORMAL ? bound_operand(bv) : fabs(bv);
          }
        }
      const uint stage_index = tile * 2u + stage;
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        const float prior0 = acc[g].thread_elements()[0], prior1 = acc[g].thread_elements()[1];
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
            const float M = max(max(c0, c1), fabs(prior));
            error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.25f * ulp32_bits(M) + 1.5625f * ulp32_bits(value);
          } else {
            error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0625f * ulp32_bits(value);
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
#pragma clang loop unroll(full)
  for (uint g = 0u; g < 2u; ++g)
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint index = (py * 320u + px) * 32u + oc0 + 8u * g + h;
    inter[index] = activate(half(acc[g].thread_elements()[h]));
    flag_count[index] = flags[g][h]; resume_stage[index] = first_stage[g][h]; resume_acc[index] = resume_val[g][h]; fast_after[index] = after_val[g][h];
    stage2[index] = second_stage[g][h]; prior2[index] = prior2_val[g][h]; after2[index] = after2_val[g][h];
  }
}

// --- one exact stage (step = tile * 2 + stage), shared by pass 1 and the replay -----------------------
inline half conv0_c8_stage(device const half* w, device const half* in, uint x, uint y, uint oc, uint step, half a) {
  const uint t = step / 2u, stage = step % 2u;
  int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
  bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
  uint base = ok ? (uint(iy) * 320u + uint(ix)) * 32u : 0u;
  ushort av[16], bv[16];
  if (ok) {
    device const ushort4* src = reinterpret_cast<device const ushort4*>(in + base + stage * 16u);   // 32-byte aligned
    const ushort4 v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
    av[0] = v0.x; av[1] = v0.y; av[2] = v0.z; av[3] = v0.w; av[4] = v1.x; av[5] = v1.y; av[6] = v1.z; av[7] = v1.w;
    av[8] = v2.x; av[9] = v2.y; av[10] = v2.z; av[11] = v2.w; av[12] = v3.x; av[13] = v3.y; av[14] = v3.z; av[15] = v3.w;
  } else {
    for (uint k = 0u; k < 16u; ++k) av[k] = ushort(0);
  }
  for (uint k = 0u; k < 16u; ++k) bv[k] = as_type<ushort>(w[wi(t, stage, stage * 16u + k, oc)]);
  return fgf_hmma_f16_bits_16(av, bv, a);
}

// --- pass 1: first flagged stage exactly; disagree -> resume at start+1, list2. One flag and agree ->
//     done. Otherwise the second flagged stage from the fast pass's prior there (certified exact, since
//     slot 0 agreed and no stage in between was flagged); two flags and agree -> done; else resume at
//     second_stage+1 with that exact value and go to list2 ---------------------------------------------
kernel void fg_custom_block0_conv0_c8_repair_pass1_v4(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device uchar* resume_stage [[buffer(6)]], device half* resume_acc [[buffer(7)]],
    device const half* fast_after [[buffer(8)]], device const uchar* flag_count [[buffer(9)]],
    device uint* list2 [[buffer(10)]], device atomic_uint* counter2 [[buffer(11)]],
    device const uchar* stage2 [[buffer(12)]], device const half* prior2 [[buffer(13)]], device const half* after2 [[buffer(14)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 320u, y = pixel / 320u;
  const uint start = uint(resume_stage[linear]);
  const half a = conv0_c8_stage(w, in, x, y, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  const uint n = flag_count[linear];
  if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
    resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
    list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear; return;
  }
  if (n == 1u) return;
  const uint s2 = uint(stage2[linear]);
  const half a2 = conv0_c8_stage(w, in, x, y, oc, s2, prior2[linear]);
  if (n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear])) return;
  resume_stage[linear] = uchar(s2 + 1u); resume_acc[linear] = a2;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- one exact stage, lane-parallel: lane k holds operand k of the sixteen (same operand the scalar stage
//     above uses for index k); the result is uniform across the 16 lanes ------------------------------------
inline half conv0_c8_stage_lanes(device const half* w, device const half* in, uint x, uint y, uint oc, uint step, half a, uint k) {
  const uint t = step / 2u, stage = step % 2u;
  int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
  bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
  uint base = ok ? (uint(iy) * 320u + uint(ix)) * 32u : 0u;
  const uint ic = stage * 16u + k;
  const ushort av = ok ? as_type<ushort>(in[base + ic]) : ushort(0);
  const ushort bv = as_type<ushort>(w[wi(t, stage, ic, oc)]);
  return fgf_hmma_f16_lanes16(av, bv, a, k);
}

// --- pass 2: the listed replay over list2, resuming, 16 lanes per element (conv2 repair_v6 pass 2) ----
kernel void fg_custom_block0_conv0_c8_listed_v4(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device const uchar* resume_stage [[buffer(6)]], device const half* resume_acc [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const uint elem = tid / 16u, k = tid % 16u;
  if (elem >= counter[0]) return;
  const uint linear = list[elem], oc = linear % 32u, pixel = linear / 32u, x = pixel % 320u, y = pixel / 320u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 18u; ++step) a = conv0_c8_stage_lanes(w, in, x, y, oc, step, a, k);
  if (k != 0u) return;
  inter[linear] = activate(a);
}

// --- pool: the scalar epilogue's arithmetic, verbatim (v2 pool, renamed) -----------------------------
kernel void fg_custom_block0_conv0_c8_pool_v4(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                           uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 160u || g.y >= 92u || g.z >= 32u) return;
  const uint x = 2u * g.x, y = 2u * g.y, oc = g.z;
  const half a = inter[(y * 320u + x) * 32u + oc], bb = inter[(y * 320u + x + 1u) * 32u + oc];
  const half c = inter[((y + 1u) * 320u + x) * 32u + oc], d = inter[((y + 1u) * 320u + x + 1u) * 32u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[(g.y * 160u + g.x) * 32u + oc] = half(p * half(0.25f));
}
