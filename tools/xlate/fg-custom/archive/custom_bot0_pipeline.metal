#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// block0 bottom0 (module 58, launch 68): bilinear upsample 80x46 -> 160x92 of a 64-channel tensor,
// then a 3x3 conv 64 -> 96 in four K=16 stages, leaky, clamp on branches 1 and 2 only, scattered to
// three 32-channel planes. The upsample is a fixed half-arithmetic expression, so a separate kernel
// materialises it verbatim and the conv reads it as an ordinary input.

inline half up(device const half* in, int x, int y, uint c) {
  if (x < 0 || x >= 160 || y < 0 || y >= 92) return half(0);
  float fx = (float(x) - 0.5f) * 0.5f, fy = (float(y) - 0.5f) * 0.5f;
  uint x0 = fx < 0.0f ? 0u : min(uint(fx), 79u), x1 = fx < 0.0f ? 0u : min(uint(fx + 1.0f), 79u);
  uint y0 = fy < 0.0f ? 0u : min(uint(fy), 45u), y1 = fy < 0.0f ? 0u : min(uint(fy + 1.0f), 45u);
  half wx1 = half(fx - float(x0)), wx0 = half(1.0f - (fx - float(x0)));
  half wy1 = half(fy - float(y0)), wy0 = half(1.0f - (fy - float(y0)));
  half a = in[(y0 * 80u + x0) * 64u + c], b = in[(y0 * 80u + x1) * 64u + c];
  half d = in[(y1 * 80u + x0) * 64u + c], e = in[(y1 * 80u + x1) * 64u + c];
  half top = half(half(wx0 * a) + half(wx1 * b)); top = half(wy0 * top);
  half bot = half(half(wx0 * d) + half(wx1 * e)); bot = half(wy1 * bot);
  return half(top + bot);
}
inline uint wi(uint t, uint s, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 4608u + t * 512u + s * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline float ulp32_of(float v) { int e = 0; (void)frexp(fabs(v), e); return exp2(float(e - 24)); }

// --- upsample: the scalar kernel's up() at every (x, y, c) it can be asked for ----------------------
kernel void fg_custom_block0_bot0_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]],
                                           uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 160u || g.y >= 92u || g.z >= 64u) return;
  ups[(g.y * 160u + g.x) * 64u + g.z] = up(in, int(g.x), int(g.y), g.z);
}

inline void scatter(device half* o0, device half* o1, device half* o2, uint px, uint py, uint oc, half a) {
  const half q = half(0.01f); a = max(a, half(a * q));
  const uint branch = oc / 32u, lc = oc % 32u;
  if (branch != 0u) a = min(max(a, half(-6.0f)), half(6.0f));
  const uint i = (py * 160u + px) * 32u + lc;
  if (branch == 0u) o0[i] = a; else if (branch == 1u) o1[i] = a; else o2[i] = a;
}

// --- fast pass -----------------------------------------------------------------------------------
kernel void fg_custom_block0_bot0_bounded(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device uchar* suspect [[buffer(6)]], device uchar* resume_stage [[buffer(7)]], device half* resume_acc [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = group.y * 8u + mc;
  if (pixel >= 160u * 92u || oc >= 96u) return;
  const uint px = pixel % 160u, py = pixel / 160u;
  simdgroup_float8x8 acc; acc.thread_elements()[0] = float(bias[oc]); acc.thread_elements()[1] = float(bias[oc + 1u]);
  bool suspect_el[2] = {false, false}; uchar first_stage[2] = {255u, 255u}; half resume_val[2] = {half(0.0h), half(0.0h)};
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    // up() returns 0 outside [0,160)x[0,92); the materialised tensor has no border, so guard here.
    const bool ok = ix >= 0 && ix < 160 && iy >= 0 && iy < 92;
    const uint base = ok ? (uint(iy) * 160u + uint(ix)) * 64u : 0u;
    for (uint s = 0u; s < 4u; ++s) {
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
          if (!suspect_el[h]) { first_stage[h] = uchar(t * 4u + s); resume_val[h] = half(h == 0u ? prior0 : prior1); }
          suspect_el[h] = true;
        }
        acc.thread_elements()[h] = float(half(value));
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    scatter(o0, o1, o2, px, py, oc + h, half(acc.thread_elements()[h]));
    const uint linear = (py * 160u + px) * 96u + oc + h;
    suspect[linear] = suspect_el[h] ? 1u : 0u; resume_stage[linear] = first_stage[h]; resume_acc[linear] = resume_val[h];
  }
}

// --- repair --------------------------------------------------------------------------------------
kernel void fg_custom_block0_bot0_listed(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device const uint* list [[buffer(6)]], device const uint* counter [[buffer(7)]],
    device const uchar* resume_stage [[buffer(8)]], device const half* resume_acc [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 96u, pixel = linear / 96u, px = pixel % 160u, py = pixel / 160u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 36u; ++step) {
    const uint t = step / 4u, s = step % 4u;
    const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
    const bool ok = x >= 0 && x < 160 && y >= 0 && y < 92;
    const uint base = ok ? (uint(y) * 160u + uint(x)) * 64u : 0u;
    ushort av[16], bv[16];
    for (uint k = 0u; k < 16u; ++k) { const uint ic = s * 16u + k; av[k] = ok ? as_type<ushort>(ups[base + ic]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, s, ic, oc)]); }
    a = fg_hmma_f16_bits_16(av, bv, a);
  }
  scatter(o0, o1, o2, px, py, oc, a);
}
