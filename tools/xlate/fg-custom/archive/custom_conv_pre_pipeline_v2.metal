#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// block0 convPre (module 54, launch 57), second design. Same pipeline as custom_conv_pre_pipeline.metal
// (3x3 conv 10 -> 32 at 640x368, ten real channels zero-padded to K=16, half intermediate, separate 2x2
// pool), with the six changes measured on conv2 (custom_conv2_bounded_v2.metal / custom_conv2_repair.metal):
// exponent-bit ulp32, the formal bound (FORMAL=1) with 2^-14 operand clamps, per element the number of
// flagged stages and the half the fast pass produced at the first flagged stage, and a one-stage pass 1
// that settles most flagged elements before the full replay (the unchanged listed kernel) runs over list2.
// Every stage here is a K=16 stage split into two K=8 partials seeded at zero, so the bound is the
// reference's; the zero pad only makes the second partial's tail terms zero.

constant bool FORMAL [[function_constant(0)]];

inline uint wi(uint tile, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 1152u + tile * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline half activate(half a) { const half slope = half(0.01f); return min(max(max(a, half(a * slope)), half(-6.0f)), half(6.0f)); }

// --- fast pass: SIMD conv at 640x368 with the guard; writes intermediate + flag count + resume state ---
kernel void fg_custom_block0_conv_pre_bounded_v2(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device uchar* flag_count [[buffer(4)]],
    device uchar* resume_stage [[buffer(5)]], device half* resume_acc [[buffer(6)]],
    device half* fast_after [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = group.y * 8u + mc;
  if (pixel >= 640u * 368u || oc >= 32u) return;
  const uint px = pixel % 640u, py = pixel / 640u;
  simdgroup_float8x8 acc;
  acc.thread_elements()[0] = float(bias[oc]); acc.thread_elements()[1] = float(bias[oc + 1u]);
  uchar flags[2] = {0u, 0u}, first_stage[2] = {255u, 255u};
  half resume_val[2] = {half(0.0h), half(0.0h)}, after_val[2] = {half(0.0h), half(0.0h)};
  for (uint tile = 0u; tile < 9u; ++tile) {
    const int ix = int(px) + int(tile % 3u) - 1, iy = int(py) + int(tile / 3u) - 1;
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 10u : 0u;   // ten channels per pixel
    simdgroup_float8x8 a[2], b[2], qa[2], qb[2];
#pragma clang loop unroll(full)
    for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint ic = kb * 8u + mc + h;
        const float av = (ok && ic < 10u) ? float(in[base + ic]) : 0.0f;   // channels 10..15 are the zero pad
        const float bv = float(w[wi(tile, kb * 8u + mr, oc + h)]);
        a[kb].thread_elements()[h] = av; b[kb].thread_elements()[h] = bv;
        if (FORMAL) { qa[kb].thread_elements()[h] = bound_operand(av); qb[kb].thread_elements()[h] = bound_operand(bv); }
        else { qa[kb].thread_elements()[h] = fabs(av); qb[kb].thread_elements()[h] = fabs(bv); }
      }
    const float prior0 = acc.thread_elements()[0], prior1 = acc.thread_elements()[1];
    simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), bnd0 = z, bnd1 = z, part0 = z, part1 = z;
    simdgroup_multiply_accumulate(bnd0, qa[0], qb[0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[1], bnd1);
    simdgroup_multiply_accumulate(part0, a[0], b[0], part0); simdgroup_multiply_accumulate(part1, a[1], b[1], part1);
    const uint stage_index = tile;
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const float prior = h == 0u ? prior0 : prior1;
      float hi = prior, lo = 0.0f;
      const float terms[2] = {part0.thread_elements()[h], part1.thread_elements()[h]};
#pragma clang loop unroll(full)
      for (uint t = 0u; t < 2u; ++t) {
        float s = hi + terms[t], bp = s - hi; float e = (hi - (s - bp)) + (terms[t] - bp); e += lo;
        const float s2 = s + e, bp2 = s2 - s; lo = (s - (s2 - bp2)) + (e - bp2); hi = s2;
      }
      const float value = hi + lo;
      float error;
      if (FORMAL) {
        const float b0 = bnd0.thread_elements()[h] * (1.0f + 0x1p-20f), b1 = bnd1.thread_elements()[h] * (1.0f + 0x1p-20f);
        const float M = max(max(b0, b1), fabs(prior));
        error = 3.5f * (ulp32_bits(b0) + ulp32_bits(b1)) + 4.25f * ulp32_bits(M) + 1.5f * ulp32_bits(value);
      } else {
        error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0f * ulp32_bits(value);
      }
      const half rounded = half(value);
      if (as_type<ushort>(half(value - error)) != as_type<ushort>(half(value + error))) {
        if (flags[h] == 0u) { first_stage[h] = uchar(stage_index); resume_val[h] = half(prior); after_val[h] = rounded; }
        flags[h] = uchar(min(uint(flags[h]) + 1u, 255u));
      }
      acc.thread_elements()[h] = float(rounded);
    }
  }
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint index = (py * 640u + px) * 32u + oc + h;
    inter[index] = activate(half(acc.thread_elements()[h]));
    flag_count[index] = flags[h]; resume_stage[index] = first_stage[h]; resume_acc[index] = resume_val[h]; fast_after[index] = after_val[h];
  }
}

// --- one exact stage, shared by pass 1 and the replay -----------------------------------------------
inline half conv_pre_stage(device const half* w, device const half* in, uint x, uint y, uint oc, uint t, half a) {
  int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
  bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
  uint base = ok ? (uint(iy) * 640u + uint(ix)) * 10u : 0u;
  ushort av[16], bv[16];
  for (uint k = 0u; k < 16u; ++k) { av[k] = (ok && k < 10u) ? as_type<ushort>(in[base + k]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, k, oc)]); }
  return fg_hmma_f16_bits_16(av, bv, a);
}

// --- pass 1: the first flagged stage only; agree with the fast pass and no later flag -> the fast
//     intermediate stands (nothing written); otherwise advance the resume state and go to list2 --------
kernel void fg_custom_block0_conv_pre_repair_pass1(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device uchar* resume_stage [[buffer(6)]], device half* resume_acc [[buffer(7)]],
    device const half* fast_after [[buffer(8)]], device const uchar* flag_count [[buffer(9)]],
    device uint* list2 [[buffer(10)]], device atomic_uint* counter2 [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  const half a = conv_pre_stage(w, in, x, y, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  if (flag_count[linear] == 1u && as_type<ushort>(a) == as_type<ushort>(fast_after[linear])) return;
  resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- pass 2: the shipped scalar conv, over the compacted list, resuming (unchanged) -------------------
kernel void fg_custom_block0_conv_pre_listed(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device const uchar* resume_stage [[buffer(6)]], device const half* resume_acc [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint t = start; t < 9u; ++t) {
    int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
    bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    uint base = ok ? (uint(iy) * 640u + uint(ix)) * 10u : 0u;
    ushort av[16], bv[16];
    for (uint k = 0u; k < 16u; ++k) { av[k] = (ok && k < 10u) ? as_type<ushort>(in[base + k]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, k, oc)]); }
    a = fg_hmma_f16_bits_16(av, bv, a);
  }
  inter[linear] = activate(a);
}

// --- pool: the scalar epilogue's arithmetic, verbatim (unchanged) ------------------------------------
kernel void fg_custom_block0_conv_pre_pool(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                        uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 32u) return;
  const uint x = 2u * g.x, y = 2u * g.y, oc = g.z;
  const half a = inter[(y * 640u + x) * 32u + oc], bb = inter[(y * 640u + x + 1u) * 32u + oc];
  const half c = inter[((y + 1u) * 640u + x) * 32u + oc], d = inter[((y + 1u) * 640u + x + 1u) * 32u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[(g.y * 320u + g.x) * 32u + oc] = half(p * half(0.25f));
}
