#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
using namespace metal;

// block1 bottom1 (module 66, launch 83), design six port of custom_block1_bot1_pipeline_v2.metal.
// Three latent planes at 640x368, one K=16 stage per tap (9 stages), skip-add epilogue in finish().
// Deltas over _v2:
//   - the three per-plane dispatches are fused into one: the weight matrices b/qb are loaded once per
//     tap, the activation matrices a/qa once per plane (all three upsampled inputs bound), the MMAs and
//     the guarded fold run per plane into acc[3] with state [plane][h], and the epilogue writes each
//     channel from the plane that owns it (0-3 -> plane 0, 4 -> plane 1, 5-7 -> plane 2), the rule the
//     _v2 kernel applied per pass. A per-plane variant with the same numerics (bounded_v3p, constant
//     `plane`) is kept for the fused-vs-three-pass A/B;
//   - a second shortcut slot (second_stage / prior2 / after2) recorded when the second flag occurs;
//   - the shorter exact fold (two two_sums, errors added once) with value slack 1.5625 / 2.0625;
//   - repair pass 1 with the two-slot logic, pass 2 the listed kernel; both on hmma_exact_fast.metalh
//     (clz) with four ushort4 loads per tap (ups rows are 16 halves = 32 B, so every tap starts 32 B aligned).
// Upsample and finish are unchanged.

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

kernel void fg_custom_block1_bot1_upsample_v3(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]],
                                              uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 640u || g.y >= 368u || g.z >= 16u) return;
  ups[(g.y * 640u + g.x) * 16u + g.z] = up(in, int(g.x), int(g.y), g.z);
}

// One guarded stage for one plane: MMAs on the shared weights, short exact fold, two shortcut slots.
struct Slots { uchar flags[2], first_stage[2], second_stage[2]; half resume_val[2], after_val[2], prior2_val[2], after2_val[2]; };
inline void slots_init(thread Slots& s) {
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) { s.flags[h] = 0u; s.first_stage[h] = 255u; s.second_stage[h] = 255u; s.resume_val[h] = s.after_val[h] = s.prior2_val[h] = s.after2_val[h] = half(0.0h); }
}
inline void guarded_stage(thread simdgroup_float8x8& acc, thread Slots& st, uint stage_index,
                          thread const simdgroup_float8x8* a, thread const simdgroup_float8x8* qa,
                          thread const simdgroup_float8x8* b, thread const simdgroup_float8x8* qb) {
  const float prior0 = acc.thread_elements()[0], prior1 = acc.thread_elements()[1];
  simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), bnd0 = z, bnd1 = z, part0 = z, part1 = z;
  simdgroup_multiply_accumulate(bnd0, qa[0], qb[0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[1], bnd1);
  simdgroup_multiply_accumulate(part0, a[0], b[0], part0); simdgroup_multiply_accumulate(part1, a[1], b[1], part1);
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
      error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.25f * ulp32_bits(M) + 1.5625f * ulp32_bits(value);
    } else {
      error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
    }
    const half rounded = half(value);
    if (as_type<ushort>(half(value - error)) != as_type<ushort>(half(value + error))) {
      if (st.flags[h] == 0u) { st.first_stage[h] = uchar(stage_index); st.resume_val[h] = half(prior); st.after_val[h] = rounded; }
      else if (st.flags[h] == 1u) { st.second_stage[h] = uchar(stage_index); st.prior2_val[h] = half(prior); st.after2_val[h] = rounded; }
      st.flags[h] = uchar(min(uint(st.flags[h]) + 1u, 255u));
    }
    acc.thread_elements()[h] = float(rounded);
  }
}
inline void load_a(thread simdgroup_float8x8* a, thread simdgroup_float8x8* qa, device const half* ups, uint base, bool ok, uint mc) {
#pragma clang loop unroll(full)
  for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const float av = ok ? float(ups[base + kb * 8u + mc + h]) : 0.0f;
      a[kb].thread_elements()[h] = av; qa[kb].thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
    }
}
inline void load_b(thread simdgroup_float8x8* b, thread simdgroup_float8x8* qb, device const half* w, uint t, uint mr, uint oc) {
#pragma clang loop unroll(full)
  for (uint kb = 0u; kb < 2u; ++kb)
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const float bv = float(w[wi(t, kb * 8u + mr, oc + h)]);
      b[kb].thread_elements()[h] = bv; qb[kb].thread_elements()[h] = FORMAL ? bound_operand(bv) : fabs(bv);
    }
}
inline void emit(device uchar* flag_count, device uchar* resume_stage, device half* resume_acc, device half* fast_after,
                 device uchar* stage2, device half* prior2, device half* after2, uint linear, thread const Slots& st, uint h) {
  flag_count[linear] = st.flags[h]; resume_stage[linear] = st.first_stage[h]; resume_acc[linear] = st.resume_val[h]; fast_after[linear] = st.after_val[h];
  stage2[linear] = st.second_stage[h]; prior2[linear] = st.prior2_val[h]; after2[linear] = st.after2_val[h];
}

// --- fast pass, design six, fused: one dispatch, three planes, weights loaded once per tap -----------
// Buffer layout follows the listed kernels (w, bias, ups0-2, s0-2, out) so the harness shares one binder.
kernel void fg_custom_block1_bot1_bounded_v3(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]],
    device const half* ups0 [[buffer(2)]], device const half* ups1 [[buffer(3)]], device const half* ups2 [[buffer(4)]],
    device const half* s0 [[buffer(5)]], device const half* s1 [[buffer(6)]], device const half* s2 [[buffer(7)]], device half* out [[buffer(8)]],
    device uchar* flag_count [[buffer(9)]], device uchar* resume_stage [[buffer(10)]], device half* resume_acc [[buffer(11)]],
    device half* fast_after [[buffer(12)]],
    device uchar* stage2 [[buffer(13)]], device half* prior2 [[buffer(14)]], device half* after2 [[buffer(15)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = mc;
  if (pixel >= 640u * 368u) return;
  const uint px = pixel % 640u, py = pixel / 640u;
  simdgroup_float8x8 acc[3]; Slots st[3];
#pragma clang loop unroll(full)
  for (uint p = 0u; p < 3u; ++p) { acc[p].thread_elements()[0] = float(bias[oc]); acc[p].thread_elements()[1] = float(bias[oc + 1u]); slots_init(st[p]); }
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 16u : 0u;
    simdgroup_float8x8 b[2], qb[2];
    load_b(b, qb, w, t, mr, oc);
#pragma clang loop unroll(full)
    for (uint p = 0u; p < 3u; ++p) {
      device const half* ups = p == 0u ? ups0 : (p == 1u ? ups1 : ups2);
      simdgroup_float8x8 a[2], qa[2];
      load_a(a, qa, ups, base, ok, mc);
      guarded_stage(acc[p], st[p], t, a, qa, b, qb);
    }
  }
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint c = oc + h, owner = plane_of(c), linear = (py * 640u + px) * 8u + c;
#pragma clang loop unroll(full)
    for (uint p = 0u; p < 3u; ++p) {
      if (owner != p) continue;      // each channel comes from the plane that owns it
      finish(s0, s1, s2, out, px, py, c, half(acc[p].thread_elements()[h]));
      emit(flag_count, resume_stage, resume_acc, fast_after, stage2, prior2, after2, linear, st[p], h);
    }
  }
}

// --- fast pass, design six numerics, one dispatch per plane (for the fused-vs-three-pass A/B) --------
// Layout as bounded_v2 (ups at 2, plane at 10, fast_after at 11) plus stage2 / prior2 / after2 at 12-14.
kernel void fg_custom_block1_bot1_bounded_v3p(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device const half* s0 [[buffer(3)]], device const half* s1 [[buffer(4)]], device const half* s2 [[buffer(5)]], device half* out [[buffer(6)]],
    device uchar* flag_count [[buffer(7)]], device uchar* resume_stage [[buffer(8)]], device half* resume_acc [[buffer(9)]],
    constant uint& plane [[buffer(10)]], device half* fast_after [[buffer(11)]],
    device uchar* stage2 [[buffer(12)]], device half* prior2 [[buffer(13)]], device half* after2 [[buffer(14)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc = mc;
  if (pixel >= 640u * 368u) return;
  const uint px = pixel % 640u, py = pixel / 640u;
  simdgroup_float8x8 acc; acc.thread_elements()[0] = float(bias[oc]); acc.thread_elements()[1] = float(bias[oc + 1u]);
  Slots st; slots_init(st);
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 16u : 0u;
    simdgroup_float8x8 a[2], qa[2], b[2], qb[2];
    load_b(b, qb, w, t, mr, oc); load_a(a, qa, ups, base, ok, mc);
    guarded_stage(acc, st, t, a, qa, b, qb);
  }
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint c = oc + h;
    if (plane_of(c) != plane) continue;
    finish(s0, s1, s2, out, px, py, c, half(acc.thread_elements()[h]));
    emit(flag_count, resume_stage, resume_acc, fast_after, stage2, prior2, after2, (py * 640u + px) * 8u + c, st, h);
  }
}

// --- one exact tap (clz helper, vector activation loads) ---------------------------------------------
inline half b1b1_stage(device const half* w, device const half* ups, uint px, uint py, uint oc, uint t, half a) {
  const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
  const bool ok = x >= 0 && x < 640 && y >= 0 && y < 368;
  const uint base = ok ? (uint(y) * 640u + uint(x)) * 16u : 0u;
  ushort av[16], bv[16];
  if (ok) {
    // ups rows are 16 halves (32 B); a tap covers the whole row, so the start is 32 B aligned.
    device const ushort4* v = reinterpret_cast<device const ushort4*>(ups + base);
    const ushort4 v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
    av[0] = v0.x; av[1] = v0.y; av[2] = v0.z; av[3] = v0.w; av[4] = v1.x; av[5] = v1.y; av[6] = v1.z; av[7] = v1.w;
    av[8] = v2.x; av[9] = v2.y; av[10] = v2.z; av[11] = v2.w; av[12] = v3.x; av[13] = v3.y; av[14] = v3.z; av[15] = v3.w;
  } else {
    for (uint k = 0u; k < 16u; ++k) av[k] = ushort(0);
  }
  for (uint k = 0u; k < 16u; ++k) bv[k] = as_type<ushort>(w[wi(t, k, oc)]);
  return fgf_hmma_f16_bits_16(av, bv, a);
}

// --- repair pass 1: two-slot shortcut -------------------------------------------------------------
kernel void fg_custom_block1_bot1_repair_pass1_v3(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]],
    device const half* ups0 [[buffer(2)]], device const half* ups1 [[buffer(3)]], device const half* ups2 [[buffer(4)]],
    device const half* s0 [[buffer(5)]], device const half* s1 [[buffer(6)]], device const half* s2 [[buffer(7)]], device half* out [[buffer(8)]],
    device const uint* list [[buffer(9)]], device const uint* counter [[buffer(10)]],
    device uchar* resume_stage [[buffer(11)]], device half* resume_acc [[buffer(12)]],
    device const half* fast_after [[buffer(13)]], device const uchar* flag_count [[buffer(14)]],
    device uint* list2 [[buffer(15)]], device atomic_uint* counter2 [[buffer(16)]],
    device const uchar* stage2 [[buffer(17)]], device const half* prior2 [[buffer(18)]], device const half* after2 [[buffer(19)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 8u, pixel = linear / 8u, px = pixel % 640u, py = pixel / 640u;
  device const half* ups = oc < 4u ? ups0 : (oc == 4u ? ups1 : ups2);
  const uint start = uint(resume_stage[linear]);
  const half a = b1b1_stage(w, ups, px, py, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  const uint n = uint(flag_count[linear]);
  if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
    resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
    list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear; return;
  }
  if (n == 1u) return;
  // slot 0 agreed, so every tap up to the second flagged one is certified and the fast pass's prior there is exact
  const uint t2 = uint(stage2[linear]);
  const half a2 = b1b1_stage(w, ups, px, py, oc, t2, prior2[linear]);
  if (n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear])) return;
  resume_stage[linear] = uchar(t2 + 1u); resume_acc[linear] = a2;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- repair pass 2: the listed kernel on the clz helper ---------------------------------------------
kernel void fg_custom_block1_bot1_listed_v3(
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
  for (uint t = start; t < 9u; ++t) a = b1b1_stage(w, ups, px, py, oc, t, a);
  finish(s0, s1, s2, out, px, py, oc, a);
}
