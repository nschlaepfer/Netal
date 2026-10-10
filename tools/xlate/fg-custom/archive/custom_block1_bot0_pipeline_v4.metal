#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_lanes_fast.metalh"
using namespace metal;

// _v4: the ONLY change vs _v3 is repair pass 2. `fg_custom_block1_bot0_listed_v4` runs sixteen lanes per
// element (tid/16 = element, tid%16 = k): each lane loads its own a_k / b_k for the stage and the
// reduction is the xor-shuffle form in hmma_exact_lanes_fast.metalh (same rule, same rounding, same
// bits as fgf_hmma_f16_bits_16). Lane 0 runs the epilogue. Every other kernel is the _v3 kernel verbatim
// under a _v4 name. The harness passes per_group2 = 4 so a 64-thread group holds four elements.

// block1 bottom0 (module 65, launch 82), design six port of custom_block1_bot0_pipeline_v2.metal.
// Deltas over _v2, copied structurally from custom_conv2_bounded_v6.metal / custom_conv2_repair_v5.metal:
//   - two 8-channel output groups per simdgroup (16 channels = exactly one output plane), sharing the
//     activation matrices a/qa loaded once per stage; grid y = 48/16 = 3;
//   - a second shortcut slot (second_stage / prior2 / after2) recorded when the second flag occurs;
//   - the shorter exact fold (two two_sums, errors added once) with value slack 1.5625 / 2.0625;
//   - repair pass 1 with the two-slot logic, pass 2 the listed kernel; both on hmma_exact_fast.metalh
//     (clz) with four ushort4 loads per stage (ups rows are 32 halves = 64 bytes, stage starts 32 B).
// Upsample and scatter are unchanged.

inline half up(device const half* in, int x, int y, uint c) {
  if (x < 0 || x >= 320 || y < 0 || y >= 184) return half(0);
  float fx = (float(x) - 0.5f) * 0.5f, fy = (float(y) - 0.5f) * 0.5f;
  uint x0 = fx < 0.0f ? 0u : min(uint(fx), 159u), x1 = fx < 0.0f ? 0u : min(uint(fx + 1.0f), 159u);
  uint y0 = fy < 0.0f ? 0u : min(uint(fy), 91u), y1 = fy < 0.0f ? 0u : min(uint(fy + 1.0f), 91u);
  half wx1 = half(fx - float(x0)), wx0 = half(1.0f - (fx - float(x0)));
  half wy1 = half(fy - float(y0)), wy0 = half(1.0f - (fy - float(y0)));
  half a = in[(y0 * 160u + x0) * 32u + c], b = in[(y0 * 160u + x1) * 32u + c];
  half d = in[(y1 * 160u + x0) * 32u + c], e = in[(y1 * 160u + x1) * 32u + c];
  half top = half(half(wx0 * a) + half(wx1 * b)); top = half(wy0 * top);
  half bot = half(half(wx0 * d) + half(wx1 * e)); bot = half(wy1 * bot);
  return half(top + bot);
}
inline uint wi(uint t, uint s, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u, f = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 2304u + t * 256u + s * 128u + (f >= 2u ? 64u : 0u) + lane * 2u + (f & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
constant bool FORMAL [[function_constant(0)]];
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }

// --- upsample: the scalar kernel's up() at every (x, y, c) it can be asked for ----------------------
kernel void fg_custom_block1_bot0_upsample_v4(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]],
                                              uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 32u) return;
  ups[(g.y * 320u + g.x) * 32u + g.z] = up(in, int(g.x), int(g.y), g.z);
}

inline void scatter(device half* o0, device half* o1, device half* o2, uint px, uint py, uint oc, half a) {
  const half q = half(0.01f); a = max(half(a * q), a);
  const uint branch = oc / 16u, lc = oc % 16u;
  const uint i = (py * 320u + px) * 16u + lc;
  if (branch == 0u) o0[i] = a; else if (branch == 1u) o1[i] = a; else o2[i] = a;
}

// --- fast pass, design six: two output groups per simdgroup, shared activations, two shortcut slots ---
kernel void fg_custom_block1_bot0_bounded_v4(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device uchar* flag_count [[buffer(6)]], device uchar* resume_stage [[buffer(7)]], device half* resume_acc [[buffer(8)]],
    device half* fast_after [[buffer(9)]],
    device uchar* stage2 [[buffer(10)]], device half* prior2 [[buffer(11)]], device half* after2 [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc0 = group.y * 16u + mc;
  if (pixel >= 320u * 184u || oc0 >= 48u) return;
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
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    // up() returns 0 outside [0,160)x[0,92); the materialised tensor has no border, so guard here.
    const bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
    const uint base = ok ? (uint(iy) * 320u + uint(ix)) * 32u : 0u;
    for (uint s = 0u; s < 2u; ++s) {
      simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb) {
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const float av = ok ? float(ups[base + s * 16u + kb * 8u + mc + h]) : 0.0f;
          a[kb].thread_elements()[h] = av;
          qa[kb].thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
        }
#pragma clang loop unroll(full)
        for (uint g = 0u; g < 2u; ++g) {
          const uint oc = oc0 + 8u * g;
#pragma clang loop unroll(full)
          for (uint h = 0u; h < 2u; ++h) {
            const float bv = float(w[wi(t, s, s * 16u + kb * 8u + mr, oc + h)]);
            b[g][kb].thread_elements()[h] = bv;
            qb[g][kb].thread_elements()[h] = FORMAL ? bound_operand(bv) : fabs(bv);
          }
        }
      }
      const uint stage_index = t * 2u + s;
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
    const uint oc = oc0 + 8u * g + h;
    scatter(o0, o1, o2, px, py, oc, half(acc[g].thread_elements()[h]));
    const uint linear = (py * 320u + px) * 48u + oc;
    flag_count[linear] = flags[g][h]; resume_stage[linear] = first_stage[g][h]; resume_acc[linear] = resume_val[g][h]; fast_after[linear] = after_val[g][h];
    stage2[linear] = second_stage[g][h]; prior2[linear] = prior2_val[g][h]; after2[linear] = after2_val[g][h];
  }
}

// --- one exact stage (clz helper, vector activation loads) -------------------------------------------
inline half b1b0_stage(device const half* w, device const half* ups, uint px, uint py, uint oc, uint step, half a) {
  const uint t = step / 2u, s = step % 2u;
  const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
  const bool ok = x >= 0 && x < 320 && y >= 0 && y < 184;
  const uint base = ok ? (uint(y) * 320u + uint(x)) * 32u : 0u;
  ushort av[16], bv[16];
  if (ok) {
    // ups rows are 32 halves (64 B); stage s starts at half offset base + 16 s, i.e. 32 B aligned.
    device const ushort4* v = reinterpret_cast<device const ushort4*>(ups + base + s * 16u);
    const ushort4 v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
    av[0] = v0.x; av[1] = v0.y; av[2] = v0.z; av[3] = v0.w; av[4] = v1.x; av[5] = v1.y; av[6] = v1.z; av[7] = v1.w;
    av[8] = v2.x; av[9] = v2.y; av[10] = v2.z; av[11] = v2.w; av[12] = v3.x; av[13] = v3.y; av[14] = v3.z; av[15] = v3.w;
  } else {
    for (uint k = 0u; k < 16u; ++k) av[k] = ushort(0);
  }
  for (uint k = 0u; k < 16u; ++k) bv[k] = as_type<ushort>(w[wi(t, s, s * 16u + k, oc)]);
  return fgf_hmma_f16_bits_16(av, bv, a);
}

// --- repair pass 1: two-slot shortcut -------------------------------------------------------------
kernel void fg_custom_block1_bot0_repair_pass1_v4(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device const uint* list [[buffer(6)]], device const uint* counter [[buffer(7)]],
    device uchar* resume_stage [[buffer(8)]], device half* resume_acc [[buffer(9)]],
    device const half* fast_after [[buffer(10)]], device const uchar* flag_count [[buffer(11)]],
    device uint* list2 [[buffer(12)]], device atomic_uint* counter2 [[buffer(13)]],
    device const uchar* stage2 [[buffer(14)]], device const half* prior2 [[buffer(15)]], device const half* after2 [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 48u, pixel = linear / 48u, px = pixel % 320u, py = pixel / 320u;
  const uint start = uint(resume_stage[linear]);
  const half a = b1b0_stage(w, ups, px, py, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  const uint n = uint(flag_count[linear]);
  if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
    resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
    list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear; return;
  }
  if (n == 1u) return;
  // slot 0 agreed, so every stage up to the second flagged one is certified and the fast pass's prior there is exact
  const uint s2 = uint(stage2[linear]);
  const half a2 = b1b0_stage(w, ups, px, py, oc, s2, prior2[linear]);
  if (n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear])) return;
  resume_stage[linear] = uchar(s2 + 1u); resume_acc[linear] = a2;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- repair pass 2: lane-parallel full replay, sixteen lanes per element ----------------------------
inline half b1b0_stage_lanes(device const half* w, device const half* ups, uint px, uint py, uint oc, uint step, half a, uint k) {
  const uint t = step / 2u, s = step % 2u;
  const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
  const bool ok = x >= 0 && x < 320 && y >= 0 && y < 184;
  const uint base = ok ? (uint(y) * 320u + uint(x)) * 32u : 0u;
  const uint ic = s * 16u + k;
  const ushort av = ok ? as_type<ushort>(ups[base + ic]) : ushort(0);
  const ushort bv = as_type<ushort>(w[wi(t, s, ic, oc)]);
  return fgf_hmma_f16_lanes16(av, bv, a, k);
}
kernel void fg_custom_block1_bot0_listed_v4(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device const uint* list [[buffer(6)]], device const uint* counter [[buffer(7)]],
    device const uchar* resume_stage [[buffer(8)]], device const half* resume_acc [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  const uint elem = tid / 16u, k = tid % 16u;
  if (elem >= counter[0]) return;
  const uint linear = list[elem], oc = linear % 48u, pixel = linear / 48u, px = pixel % 320u, py = pixel / 320u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 18u; ++step) a = b1b0_stage_lanes(w, ups, px, py, oc, step, a, k);
  if (k != 0u) return;
  scatter(o0, o1, o2, px, py, oc, a);
}
