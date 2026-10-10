#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// block1 conv0 (module 62, launch 72): a 3x3 conv 18 -> 16 at 640x368 - a K=16 stage plus a K=8 tail for channels 16-17, each rounded to half separately -, fused with a 2x2 average
// pool to 320x184. Weights are plain row-major, (oc*9+tile)*18 + k. The scalar kernel computes four full convs per output and pools them with half
// rounding at each add. Here the conv runs once at full resolution into a half intermediate - which
// is lossless, since cv() returns half - through the same guard pipeline as conv2, and a separate
// pool kernel repeats the scalar epilogue's arithmetic exactly.

inline uint wi(uint tile, uint ic, uint oc) { return (oc * 9u + tile) * 18u + ic; }
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline float ulp32_of(float v) { int e = 0; (void)frexp(fabs(v), e); return exp2(float(e - 24)); }
inline half activate(half a) { const half slope = half(0.01f); return min(max(max(a, half(a * slope)), half(-6.0f)), half(6.0f)); }

// --- fast pass: SIMD conv at 320x184 with the prior-free guard; writes intermediate + flags ---------
kernel void fg_custom_block1_conv0_bounded(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device uchar* suspect [[buffer(4)]],
    device uchar* resume_stage [[buffer(5)]], device half* resume_acc [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = group.y * 8u + mc;
  if (pixel >= 640u * 368u || oc >= 16u) return;
  const uint px = pixel % 640u, py = pixel / 640u;
  simdgroup_float8x8 acc;
  acc.thread_elements()[0] = float(bias[oc]); acc.thread_elements()[1] = float(bias[oc + 1u]);
  bool suspect_el[2] = {false, false}; uchar first_stage[2] = {255u, 255u}; half resume_val[2] = {half(0.0h), half(0.0h)};
  for (uint tile = 0u; tile < 9u; ++tile) {
    const int ix = int(px) + int(tile % 3u) - 1, iy = int(py) + int(tile / 3u) - 1;
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 18u : 0u;   // eighteen channels per pixel
   for (uint stage = 0u; stage < 2u; ++stage) {
    // stage 0: channels 0..15 as two K=8 MMAs. stage 1: the tail - channels 16,17 land in k-block 0
    // positions 0,1 with everything else zero, so it is one K=8 MMA and the second half is identically 0.
    simdgroup_float8x8 a[2], b[2], qa[2], qb[2];
#pragma clang loop unroll(full)
    for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint k_a = kb * 8u + mc + h, k_b = kb * 8u + mr;
        const bool live_a = stage == 0u ? true : (kb == 0u && k_a < 2u);
        const bool live_b = stage == 0u ? true : (kb == 0u && k_b < 2u);
        const uint ic_a = stage == 0u ? k_a : 16u + k_a, ic_b = stage == 0u ? k_b : 16u + k_b;
        const float av = (ok && live_a) ? float(in[base + ic_a]) : 0.0f;
        const float bv = live_b ? float(w[wi(tile, ic_b, oc + h)]) : 0.0f;
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
      for (uint t = 0u; t < 2u; ++t) {
        float s = hi + terms[t], bp = s - hi; float e = (hi - (s - bp)) + (terms[t] - bp); e += lo;
        const float s2 = s + e, bp2 = s2 - s; lo = (s - (s2 - bp2)) + (e - bp2); hi = s2;
      }
      const float value = hi + lo;
      const float error = 4.0f * ulp32_of(bnd0.thread_elements()[h]) + 4.0f * ulp32_of(bnd1.thread_elements()[h]) + 2.0f * ulp32_of(value);
      if (as_type<ushort>(half(value - error)) != as_type<ushort>(half(value + error))) {
        if (!suspect_el[h]) { first_stage[h] = uchar(tile * 2u + stage); resume_val[h] = half(h == 0u ? prior0 : prior1); }
        suspect_el[h] = true;
      }
      acc.thread_elements()[h] = float(half(value));
    }
   }
  }
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint index = (py * 640u + px) * 16u + oc + h;
    inter[index] = activate(half(acc.thread_elements()[h]));
    suspect[index] = suspect_el[h] ? 1u : 0u; resume_stage[index] = first_stage[h]; resume_acc[index] = resume_val[h];
  }
}

// --- repair: the shipped scalar conv, over the compacted list, resuming ----------------------------
kernel void fg_custom_block1_conv0_listed(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device const uchar* resume_stage [[buffer(6)]], device const half* resume_acc [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 16u, pixel = linear / 16u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 18u; ++step) {
    const uint t = step / 2u, stage = step % 2u;
    int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
    bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    uint base = ok ? (uint(iy) * 640u + uint(ix)) * 18u : 0u;
    ushort av[16], bv[16];
    if (stage == 0u) {
      for (uint k = 0u; k < 16u; ++k) { av[k] = ok ? as_type<ushort>(in[base + k]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, k, oc)]); }
    } else {
      for (uint k = 0u; k < 16u; ++k) { av[k] = ushort(0); bv[k] = ushort(0); }
      av[0] = ok ? as_type<ushort>(in[base + 16u]) : ushort(0); av[1] = ok ? as_type<ushort>(in[base + 17u]) : ushort(0);
      bv[0] = as_type<ushort>(w[wi(t, 16u, oc)]); bv[1] = as_type<ushort>(w[wi(t, 17u, oc)]);
    }
    a = fg_hmma_f16_bits_16(av, bv, a);
  }
  inter[linear] = activate(a);
}

// --- pool: the scalar epilogue's arithmetic, verbatim -----------------------------------------------
kernel void fg_custom_block1_conv0_pool(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                        uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 16u) return;
  const uint x = 2u * g.x, y = 2u * g.y, oc = g.z;
  const half a = inter[(y * 640u + x) * 16u + oc], bb = inter[(y * 640u + x + 1u) * 16u + oc];
  const half c = inter[((y + 1u) * 640u + x) * 16u + oc], d = inter[((y + 1u) * 640u + x + 1u) * 16u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[(g.y * 320u + g.x) * 16u + oc] = half(p * half(0.25f));
}
