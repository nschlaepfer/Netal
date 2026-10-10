#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// block1 bottom1 (module 66, launch 83), second design port of custom_block1_bot1_pipeline.metal.
// Three latent planes at 640x368, one K=16 stage per tap (9 stages), skip-add epilogue inside the
// kernel. The bounded kernel runs once per plane (constant `plane`) and writes only the channels that
// plane owns, fast_after included. Pass 1 settles one exact stage; pass 2 is the original listed kernel.

inline half up(device const half* in, int x, int y, uint c) {
  if (x < 0 || x >= 640 || y < 0 || y >= 368) return half(0);
  const float fx = (float(x) - 0.5f) * 0.5f, fy = (float(y) - 0.5f) * 0.5f;
  const uint x0 = fx < 0 ? 0u : min(uint(fx), 319u), x1 = fx < 0 ? 0u : min(uint(fx + 1.0f), 319u);
  const uint y0 = fy < 0 ? 0u : min(uint(fy), 183u), y1 = fy < 0 ? 0u : min(uint(fy + 1.0f), 183u);
  const half wx1 = half(fx - float(x0)), wx0 = half(1.0f - (fx - float(x0)));
  const half wy1 = half(fy - float(y0)), wy0 = half(1.0f - (fy - float(y0)));
  const half a = in[(y0 * 320u + x0) * 16u + c], b = in[(y0 * 320u + x1) * 16u + c];
  const half d = in[(y1 * 320u + x0) * 16u + c], e = in[(y1 * 320u + x1) * 16u + c];
  half top = half(half(wx0 * a) + half(wx1 * b)); top = half(wy0 * top);
  half bot = half(half(wx0 * d) + half(wx1 * e)); bot = half(wy1 * bot);
  return half(top + bot);
}
inline uint wi(uint t, uint ic, uint oc) {
  const uint k = ic, lane = oc * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
  return t * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
constant bool FORMAL [[function_constant(0)]];
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline uint plane_of(uint oc) { return oc < 4u ? 0u : (oc == 4u ? 1u : 2u); }
inline void finish(device const half* s0, device const half* s1, device const half* s2, device half* out, uint px, uint py, uint oc, half a) {
  const uint p = py * 640u + px;
  half skip;
  if (oc < 4u) skip = s0[p * 4u + oc]; else if (oc == 4u) skip = s1[p]; else skip = s2[p * 3u + oc - 5u];
  if (oc < 4u) { const half scaled = half(half(2.0f) * skip); a = half(a + scaled); }
  else { a = min(max(a, half(-6.0f)), half(6.0f)); const half scaled = half(half(1.0f) * skip); a = half(a + scaled); }
  out[p * 8u + oc] = a;
}

kernel void fg_custom_block1_bot1_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]],
                                           uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 640u || g.y >= 368u || g.z >= 16u) return;
  ups[(g.y * 640u + g.x) * 16u + g.z] = up(in, int(g.x), int(g.y), g.z);
}

// --- fast pass, second design (one dispatch per plane) --------------------------------------------
kernel void fg_custom_block1_bot1_bounded_v2(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device const half* s0 [[buffer(3)]], device const half* s1 [[buffer(4)]], device const half* s2 [[buffer(5)]], device half* out [[buffer(6)]],
    device uchar* flag_count [[buffer(7)]], device uchar* resume_stage [[buffer(8)]], device half* resume_acc [[buffer(9)]],
    constant uint& plane [[buffer(10)]], device half* fast_after [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = mc;
  if (pixel >= 640u * 368u) return;
  const uint px = pixel % 640u, py = pixel / 640u;
  simdgroup_float8x8 acc; acc.thread_elements()[0] = float(bias[oc]); acc.thread_elements()[1] = float(bias[oc + 1u]);
  uchar flags[2] = {0u, 0u}, first_stage[2] = {255u, 255u};
  half resume_val[2] = {half(0.0h), half(0.0h)}, after_val[2] = {half(0.0h), half(0.0h)};
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 16u : 0u;
    simdgroup_float8x8 a[2], b[2], qa[2], qb[2];
#pragma clang loop unroll(full)
    for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const float av = ok ? float(ups[base + kb * 8u + mc + h]) : 0.0f;
        const float bv = float(w[wi(t, kb * 8u + mr, oc + h)]);
        a[kb].thread_elements()[h] = av; b[kb].thread_elements()[h] = bv;
        if (FORMAL) { qa[kb].thread_elements()[h] = bound_operand(av); qb[kb].thread_elements()[h] = bound_operand(bv); }
        else { qa[kb].thread_elements()[h] = fabs(av); qb[kb].thread_elements()[h] = fabs(bv); }
      }
    const float prior0 = acc.thread_elements()[0], prior1 = acc.thread_elements()[1];
    simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), bnd0 = z, bnd1 = z, part0 = z, part1 = z;
    simdgroup_multiply_accumulate(bnd0, qa[0], qb[0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[1], bnd1);
    simdgroup_multiply_accumulate(part0, a[0], b[0], part0); simdgroup_multiply_accumulate(part1, a[1], b[1], part1);
    const uint stage_index = t;
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const float prior = h == 0u ? prior0 : prior1;
      float hi = prior, lo = 0.0f;
      const float terms[2] = {part0.thread_elements()[h], part1.thread_elements()[h]};
#pragma clang loop unroll(full)
      for (uint q = 0u; q < 2u; ++q) {
        float sm = hi + terms[q], bp = sm - hi; float e = (hi - (sm - bp)) + (terms[q] - bp); e += lo;
        const float s2 = sm + e, bp2 = s2 - sm; lo = (sm - (s2 - bp2)) + (e - bp2); hi = s2;
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
    const uint c = oc + h;
    if (plane_of(c) != plane) continue;
    finish(s0, s1, s2, out, px, py, c, half(acc.thread_elements()[h]));
    const uint linear = (py * 640u + px) * 8u + c;
    flag_count[linear] = flags[h]; resume_stage[linear] = first_stage[h]; resume_acc[linear] = resume_val[h]; fast_after[linear] = after_val[h];
  }
}

// --- repair pass 1: one exact stage (the first flagged tap) and a shortcut --------------------------
inline half b1b1_stage(device const half* w, device const half* ups, uint px, uint py, uint oc, uint t, half a) {
  const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
  const bool ok = x >= 0 && x < 640 && y >= 0 && y < 368;
  const uint base = ok ? (uint(y) * 640u + uint(x)) * 16u : 0u;
  ushort av[16], bv[16];
  for (uint k = 0u; k < 16u; ++k) { av[k] = ok ? as_type<ushort>(ups[base + k]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, k, oc)]); }
  return fg_hmma_f16_bits_16(av, bv, a);
}
kernel void fg_custom_block1_bot1_repair_pass1(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]],
    device const half* ups0 [[buffer(2)]], device const half* ups1 [[buffer(3)]], device const half* ups2 [[buffer(4)]],
    device const half* s0 [[buffer(5)]], device const half* s1 [[buffer(6)]], device const half* s2 [[buffer(7)]], device half* out [[buffer(8)]],
    device const uint* list [[buffer(9)]], device const uint* counter [[buffer(10)]],
    device uchar* resume_stage [[buffer(11)]], device half* resume_acc [[buffer(12)]],
    device const half* fast_after [[buffer(13)]], device const uchar* flag_count [[buffer(14)]],
    device uint* list2 [[buffer(15)]], device atomic_uint* counter2 [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 8u, pixel = linear / 8u, px = pixel % 640u, py = pixel / 640u;
  device const half* ups = oc < 4u ? ups0 : (oc == 4u ? ups1 : ups2);
  const uint start = uint(resume_stage[linear]);
  const half a = b1b1_stage(w, ups, px, py, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  if (flag_count[linear] == 1u && as_type<ushort>(a) == as_type<ushort>(fast_after[linear])) return;
  resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- repair pass 2: the original listed kernel, unchanged ------------------------------------------
kernel void fg_custom_block1_bot1_listed(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]],
    device const half* ups0 [[buffer(2)]], device const half* ups1 [[buffer(3)]], device const half* ups2 [[buffer(4)]],
    device const half* s0 [[buffer(5)]], device const half* s1 [[buffer(6)]], device const half* s2 [[buffer(7)]], device half* out [[buffer(8)]],
    device const uint* list [[buffer(9)]], device const uint* counter [[buffer(10)]],
    device const uchar* resume_stage [[buffer(11)]], device const half* resume_acc [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 8u, pixel = linear / 8u, px = pixel % 640u, py = pixel / 640u;
  device const half* ups = oc < 4u ? ups0 : (oc == 4u ? ups1 : ups2);
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint t = start; t < 9u; ++t) {
    const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
    const bool ok = x >= 0 && x < 640 && y >= 0 && y < 368;
    const uint base = ok ? (uint(y) * 640u + uint(x)) * 16u : 0u;
    ushort av[16], bv[16];
    for (uint k = 0u; k < 16u; ++k) { av[k] = ok ? as_type<ushort>(ups[base + k]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, k, oc)]); }
    a = fg_hmma_f16_bits_16(av, bv, a);
  }
  finish(s0, s1, s2, out, px, py, oc, a);
}
