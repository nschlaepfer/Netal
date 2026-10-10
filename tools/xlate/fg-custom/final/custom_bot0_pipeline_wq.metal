#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_u32.metalh"
using namespace metal;

// block0 bottom0 (module 58, launch 68), design six. Port of custom_conv2_bounded_v6.metal /
// custom_conv2_repair_v5.metal onto custom_bot0_pipeline_v2.metal. Deltas over _v2:
//   - one simdgroup computes 16 output channels (two 8-channel groups g = 0, 1) sharing the activation
//     matrices loaded once; grid y is 96 / 16 = 6. A 16-channel pair never straddles a 32-channel plane, and
//     the scatter stays per channel;
//   - a second shortcut slot per element (second flagged step, the prior into it, the fast half there);
//   - the shorter exact fold with the value slack 1.5 -> 1.5625 (formal), 2.0 -> 2.0625 (empirical);
//   - pass 1 with the two-slot logic; pass 1 and pass 2 on hmma_exact_fast.metalh (clz) and, since a stage's
//     16 activations start at base + s*16 in a 64-channel row (32-byte aligned), four ushort4 loads.
// The upsample pre-kernel is unchanged (copied under a _v3 name so this file stands alone in a metallib).
//
// _wq (design thirteen, from custom_conv2_bounded_v13.metal): the bounded kernel reads its weights and their
// bound proxies from a host-built half4 table instead of gathering 2 groups x 2 kb x 2 h halves through wi()
// and applying bound_operand() to each. Table entry (t, s, kb, oc-pair, matrix_row) =
// (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) with q(x) = x != 0 ? max(|x|, 2^-14) : 0 (== bound_operand after the
// exact half->float widening); for fixed (t, s, kb, g, group.y) the 32 lanes read 32 consecutive half4.
// wq_index(t, s, kb, pair, row) = (((t*4 + s)*2 + kb)*48 + pair)*8 + row; 9*4*2*48*8 = 27648 half4 (221184 bytes).
// Nothing else changes; pass 1 / pass 2 still read w through wi(). Every kernel is copied under a _wq name.

constant bool FORMAL [[function_constant(0)]];

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
inline uint wq_index(uint t, uint s, uint kb, uint pair, uint row) { return (((t * 4u + s) * 2u + kb) * 48u + pair) * 8u + row; }
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline ushort zero_norm(half h) { const ushort b = as_type<ushort>(h); return b == 0x8000u ? ushort(0) : b; }
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }

// --- upsample: the scalar kernel's up() at every (x, y, c) it can be asked for (unchanged) -----------------
kernel void fg_custom_block0_bot0_upsample_wq(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]],
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

// --- fast pass: 16 output channels per simdgroup ---------------------------------------------------------------
kernel void fg_custom_block0_bot0_bounded_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device uchar* flag_count [[buffer(6)]], device uchar* resume_stage [[buffer(7)]], device half* resume_acc [[buffer(8)]],
    device half* fast_after [[buffer(9)]],
    device uchar* stage2 [[buffer(10)]], device half* prior2 [[buffer(11)]], device half* after2 [[buffer(12)]],
    device const half4* wq [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint pixel = group.x * 8u + mr, oc0 = group.y * 16u + mc;
  if (pixel >= 160u * 92u || oc0 >= 96u) return;
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
  for (uint t = 0u; t < 9u; ++t) {
    const int ix = int(px) + int(t % 3u) - 1, iy = int(py) + int(t / 3u) - 1;
    // up() returns 0 outside [0,160)x[0,92); the materialised tensor has no border, so guard here.
    const bool ok = ix >= 0 && ix < 160 && iy >= 0 && iy < 92;
    const uint base = ok ? (uint(iy) * 160u + uint(ix)) * 64u : 0u;
    for (uint s = 0u; s < 4u; ++s) {
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
          // one half4 per (kb, g): (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for oc = oc0 + 8 g, ic = s*16 + kb*8 + mr
          const half4 tq = wq[wq_index(t, s, kb, (oc0 + 8u * g) >> 1u, mr)];
          const float b0 = float(tq.x), b1 = float(tq.y);
          b[g][kb].thread_elements()[0] = b0; b[g][kb].thread_elements()[1] = b1;
          qb[g][kb].thread_elements()[0] = FORMAL ? float(tq.z) : fabs(b0); qb[g][kb].thread_elements()[1] = FORMAL ? float(tq.w) : fabs(b1);
        }
      }
      const uint stage_index = t * 4u + s;
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
            const float M = max(0x1p-21f, max(max(c0, c1), fabs(prior)));   // the oracle's exponent floor: max_exp >= -21 even when every term is smaller
            error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.0f * ulp32_bits(M)   /* n*delta: at most n nonzero products lose a grid unit */ + 1.5625f * ulp32_bits(value);
          } else {
            error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
          }
          half rounded = half(value); if (as_type<ushort>(rounded) == 0x8000u) rounded = half(0.0h);   // rounds-to-zero is +0 (RTX)
          if (zero_norm(half(value - error)) != zero_norm(half(value + error))) {
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
    const uint linear = (py * 160u + px) * 96u + oc;
    flag_count[linear] = flags[g][h]; resume_stage[linear] = first_stage[g][h]; resume_acc[linear] = resume_val[g][h]; fast_after[linear] = after_val[g][h];
    stage2[linear] = second_stage[g][h]; prior2[linear] = prior2_val[g][h]; after2[linear] = after2_val[g][h];
  }
}

// --- one exact step (tile step/4, stage step%4) from prior a, the listed kernel's arithmetic on the clz helper.
// A stage's 16 activations start at base + s*16 in a 64-half row: a multiple of 16 halves (32 bytes), so four
// ushort4 loads are aligned.
inline half bot0_stage(device const half* w, device const half* ups, uint px, uint py, uint oc, uint step, half a) {
  const uint t = step / 4u, s = step % 4u;
  const int x = int(px) + int(t % 3u) - 1, y = int(py) + int(t / 3u) - 1;
  const bool ok = x >= 0 && x < 160 && y >= 0 && y < 92;
  const uint base = ok ? (uint(y) * 160u + uint(x)) * 64u : 0u;
  ushort av[16], bv[16];
  if (ok) {
    device const half* row = ups + base + s * 16u;
    const ushort4 v0 = *reinterpret_cast<device const ushort4*>(row), v1 = *reinterpret_cast<device const ushort4*>(row + 4u),
                  v2 = *reinterpret_cast<device const ushort4*>(row + 8u), v3 = *reinterpret_cast<device const ushort4*>(row + 12u);
    av[0] = v0.x; av[1] = v0.y; av[2] = v0.z; av[3] = v0.w; av[4] = v1.x; av[5] = v1.y; av[6] = v1.z; av[7] = v1.w;
    av[8] = v2.x; av[9] = v2.y; av[10] = v2.z; av[11] = v2.w; av[12] = v3.x; av[13] = v3.y; av[14] = v3.z; av[15] = v3.w;
  } else for (uint k = 0u; k < 16u; ++k) av[k] = ushort(0);
  for (uint k = 0u; k < 16u; ++k) bv[k] = as_type<ushort>(w[wi(t, s, s * 16u + k, oc)]);
  return fgu_hmma_f16_bits_16(av, bv, a);
}

// --- pass 1: two shortcut slots; writes nothing when the fast output stands (already scattered) -------------
kernel void fg_custom_block0_bot0_repair_pass1_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device const uint* list [[buffer(6)]], device const uint* counter [[buffer(7)]],
    device uchar* resume_stage [[buffer(8)]], device half* resume_acc [[buffer(9)]],
    device const half* fast_after [[buffer(10)]], device const uchar* flag_count [[buffer(11)]],
    device uint* list2 [[buffer(12)]], device atomic_uint* counter2 [[buffer(13)]],
    device const uchar* stage2 [[buffer(14)]], device const half* prior2 [[buffer(15)]], device const half* after2 [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 96u, pixel = linear / 96u, px = pixel % 160u, py = pixel / 160u;
  const uint start = uint(resume_stage[linear]);
  half a = bot0_stage(w, ups, px, py, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  const uint n = flag_count[linear];
  if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
    resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
    list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear; return;
  }
  if (n == 1u) return;
  // slot 0 agreed, so every step up to the second flagged one is certified and the fast pass's prior there is exact
  const uint s2 = uint(stage2[linear]);
  half a2 = bot0_stage(w, ups, px, py, oc, s2, prior2[linear]);
  if (n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear])) return;
  resume_stage[linear] = uchar(s2 + 1u); resume_acc[linear] = a2;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- pass 2: full replay over list2, resuming, on the clz helper; scatters ----------------------------------
kernel void fg_custom_block0_bot0_listed_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* ups [[buffer(2)]],
    device half* o0 [[buffer(3)]], device half* o1 [[buffer(4)]], device half* o2 [[buffer(5)]],
    device const uint* list [[buffer(6)]], device const uint* counter [[buffer(7)]],
    device const uchar* resume_stage [[buffer(8)]], device const half* resume_acc [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 96u, pixel = linear / 96u, px = pixel % 160u, py = pixel / 160u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 36u; ++step) a = bot0_stage(w, ups, px, py, oc, step, a);
  scatter(o0, o1, o2, px, py, oc, a);
}
