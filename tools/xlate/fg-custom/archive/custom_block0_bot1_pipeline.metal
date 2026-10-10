#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// block0 bottom1 (module 59, launch 69): three input planes, each bilinearly upsampled 160x92 ->
// 320x184 at 32 channels; a 3x3 conv to 8 output channels in two K=16 stages, where WHICH plane
// feeds the conv depends on the output channel (0-3 -> plane 0, 4 -> plane 1, 5-7 -> plane 2).
// That breaks the MMA's shared-A structure - A is the pixel x channel operand and cannot differ by
// output column - so the fast pass runs once per plane, computes all eight channels from it, and
// keeps only the channels that plane owns. Three passes over an 8-channel kernel is cheap.

inline half up(device const half* in, int x, int y, uint c) {
  if (x < 0 || x >= 320 || y < 0 || y >= 184) return half(0);
  const float sx = (float(x) - 0.5f) * 0.5f, sy = (float(y) - 0.5f) * 0.5f;
  const uint x0 = sx < 0.0f ? 0u : min(uint(sx), uint(159)), x1 = sx < 0.0f ? 0u : min(uint(sx + 1.0f), uint(159));
  const uint y0 = sy < 0.0f ? 0u : min(uint(sy), uint(91)), y1 = sy < 0.0f ? 0u : min(uint(sy + 1.0f), uint(91));
  const half wx1 = half(sx - float(x0)), wx0 = half(1.0f - (sx - float(x0)));
  const half wy1 = half(sy - float(y0)), wy0 = half(1.0f - (sy - float(y0)));
  const half a = in[(y0 * 160u + x0) * 32u + c], b = in[(y0 * 160u + x1) * 32u + c];
  const half c2 = in[(y1 * 160u + x0) * 32u + c], d = in[(y1 * 160u + x1) * 32u + c];
  half top = half(half(wx0 * a) + half(wx1 * b)); top = half(wy0 * top);
  half bottom = half(half(wx0 * c2) + half(wx1 * d)); bottom = half(wy1 * bottom);
  return half(top + bottom);
}
inline uint wi(uint t, uint s, uint ic, uint oc) {
  const uint k = ic % 16u, lane = oc * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
  return t * 256u + s * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline float ulp32_of(float v) { int e = 0; (void)frexp(fabs(v), e); return exp2(float(e - 24)); }
inline uint plane_of(uint oc) { return oc < 4u ? 0u : (oc == 4u ? 1u : 2u); }
inline void scatter(device half* o0, device half* o1, device half* o2, uint px, uint py, uint oc, half a) {
  if (oc >= 4u) a = min(max(a, half(-6.0f)), half(6.0f));
  if (oc < 4u) o0[(py * 320u + px) * 4u + oc] = a;
  else if (oc == 4u) o1[py * 320u + px] = a;
  else o2[(py * 320u + px) * 3u + oc - 5u] = a;
}

kernel void fg_custom_block0_bot1_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]],
                                           uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 32u) return;
  ups[(g.y * 320u + g.x) * 32u + g.z] = up(in, int(g.x), int(g.y), g.z);
}

kernel void fg_custom_block0_bot1_bounded(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device uchar* suspect [[buffer(6)]], device uchar* resume_stage [[buffer(7)]], device half* resume_acc [[buffer(8)]],
    constant uint& plane [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = mc;    // one output group of eight
  if (pixel >= 320u * 184u) return;
  const uint px = pixel % 320u, py = pixel / 320u;
  simdgroup_float8x8 acc; acc.thread_elements()[0] = float(bias[oc]); acc.thread_elements()[1] = float(bias[oc + 1u]);
  bool suspect_el[2] = {false, false}; uchar first_stage[2] = {255u, 255u}; half resume_val[2] = {half(0.0h), half(0.0h)};
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    const bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
    const uint base = ok ? (uint(iy) * 320u + uint(ix)) * 32u : 0u;
    for (uint s = 0u; s < 2u; ++s) {
      simdgroup_float8x8 a[2], b[2], qa[2], qb[2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const float av = ok ? float(ups[base + s * 16u + kb * 8u + mc + h]) : 0.0f;
          const float bv = float(w[wi(t, s, s * 16u + kb * 8u + mr, oc + h)]);
          a[kb].thread_elements()[h] = av; qa[kb].thread_elements()[h] = fabs(av);
          b[kb].thread_elements()[h] = bv; qb[kb].thread_elements()[h] = fabs(bv);
        }
      const float prior0 = acc.thread_elements()[0], prior1 = acc.thread_elements()[1];
      simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), bnd0 = z, bnd1 = z, part0 = z, part1 = z;
      simdgroup_multiply_accumulate(bnd0, qa[0], qb[0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[1], bnd1);
      simdgroup_multiply_accumulate(part0, a[0], b[0], part0); simdgroup_multiply_accumulate(part1, a[1], b[1], part1);
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        float hi = h == 0u ? prior0 : prior1, lo = 0.0f;
        const float terms[2] = {part0.thread_elements()[h], part1.thread_elements()[h]};
#pragma clang loop unroll(full)
        for (uint q = 0u; q < 2u; ++q) {
          float sm = hi + terms[q], bp = sm - hi; float e = (hi - (sm - bp)) + (terms[q] - bp); e += lo;
          const float s2 = sm + e, bp2 = s2 - sm; lo = (sm - (s2 - bp2)) + (e - bp2); hi = s2;
        }
        const float value = hi + lo;
        const float error = 4.0f * ulp32_of(bnd0.thread_elements()[h]) + 4.0f * ulp32_of(bnd1.thread_elements()[h]) + 2.0f * ulp32_of(value);
        if (as_type<ushort>(half(value - error)) != as_type<ushort>(half(value + error))) {
          if (!suspect_el[h]) { first_stage[h] = uchar(t * 2u + s); resume_val[h] = half(h == 0u ? prior0 : prior1); }
          suspect_el[h] = true;
        }
        acc.thread_elements()[h] = float(half(value));
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint c = oc + h;
    if (plane_of(c) != plane) continue;      // this pass owns only its plane's channels
    scatter(o0, o1, o2, px, py, c, half(acc.thread_elements()[h]));
    const uint linear = (py * 320u + px) * 8u + c;
    suspect[linear] = suspect_el[h] ? 1u : 0u; resume_stage[linear] = first_stage[h]; resume_acc[linear] = resume_val[h];
  }
}

kernel void fg_custom_block0_bot1_listed(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]],
    device const half* ups0 [[buffer(2)]], device const half* ups1 [[buffer(3)]], device const half* ups2 [[buffer(4)]],
    device half* o0 [[buffer(5)]], device half* o1 [[buffer(6)]], device half* o2 [[buffer(7)]],
    device const uint* list [[buffer(8)]], device const uint* counter [[buffer(9)]],
    device const uchar* resume_stage [[buffer(10)]], device const half* resume_acc [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 8u, pixel = linear / 8u, px = pixel % 320u, py = pixel / 320u;
  device const half* ups = oc < 4u ? ups0 : (oc == 4u ? ups1 : ups2);
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 18u; ++step) {
    const uint t = step / 2u, s = step % 2u;
    const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
    const bool ok = x >= 0 && x < 320 && y >= 0 && y < 184;
    const uint base = ok ? (uint(y) * 320u + uint(x)) * 32u : 0u;
    ushort av[16], bv[16];
    for (uint k = 0u; k < 16u; ++k) { const uint ic = s * 16u + k; av[k] = ok ? as_type<ushort>(ups[base + ic]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, s, ic, oc)]); }
    a = fg_hmma_f16_bits_16(av, bv, a);
  }
  scatter(o0, o1, o2, px, py, oc, a);
}
