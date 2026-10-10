#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_u32.metalh"
using namespace metal;

// block0 convPre (module 54, launch 57), design six. custom_conv_pre_pipeline_v2.metal plus the deltas
// measured on conv2 (custom_conv2_bounded_v6.metal / custom_conv2_repair_v5.metal):
//   - two 8-channel output groups per simdgroup (16 output channels) sharing the activation matrices;
//     grid y is channels / 16;
//   - a second shortcut slot per element: the stage, prior and fast half of the second flagged stage;
//   - the shorter exact fold (two two_sums, errors added once) with the value slack 1.5 -> 1.5625
//     (FORMAL) and 2.0 -> 2.0625 (empirical); the rest of both bounds unchanged;
//   - pass 1 settles elements with one or two flagged stages before the full replay; both repair kernels
//     use the clz helper (hmma_exact_fast.metalh, same arithmetic as the oracle header).
// convPre stores ten halves per input pixel (20 bytes), so a stage start is only 4-byte aligned on odd
// pixels: the repair keeps scalar activation loads here (the ushort4 loads of the conv2 reference need
// 8-byte alignment). The pool kernel is the v2 pool verbatim, renamed so this file is self-contained.

constant bool FORMAL [[function_constant(0)]];
// HW: hardware lowering - no bound matrices, no flags, no repair; the fast pass alone (PSNR-gated, not exact).
constant bool HW [[function_constant(1)]];

// _wq: custom_conv_pre_pipeline_pf.metal plus the packed weight/proxy table (design thirteen, measured on conv2 as
// custom_conv2_bounded_v13.metal). The host builds once, from the same weight buffer, a half4 table indexed by
// (tile, kb, oc-pair, matrix_row) holding (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])), q(x) = x != 0 ? max(|x|, 2^-14) : 0
// (== bound_operand after the exact half->float widening), with ic = kb*8 + matrix_row and oc = 2*pair. For fixed
// (tile, kb, g) the 32 lanes of a simdgroup read 32 consecutive half4 (256 bytes): pair = gy*8 + 4g + mc/2 spans four
// consecutive values, times eight rows. The bounded kernel loads one half4 per (kb, g) per tile instead of gathering
// 2 x 2 x 2 halves through wi() and applying bound_operand() to each; everything else is the pf kernel verbatim.
// convPre has one K=16 stage per tap, so the table is 9 tiles x 2 kb x 16 pairs x 8 rows = 2304 half4 (18432 bytes).
// Buffer 12 of the bounded kernel; the repair kernels keep reading w through wi().
inline uint wq_index(uint tile, uint kb, uint pair, uint row) {
  return ((tile * 2u + kb) * 16u + pair) * 8u + row;
}

inline uint wi(uint tile, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 1152u + tile * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline ushort zero_norm(half h) { const ushort b = as_type<ushort>(h); return b == 0x8000u ? ushort(0) : b; }
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline half activate(half a) { const half slope = half(0.01f); return min(max(max(a, half(a * slope)), half(-6.0f)), half(6.0f)); }

// --- fast pass: SIMD conv at 640x368 with the guard, 16 output channels per simdgroup ------------------
kernel void fg_custom_block0_conv_pre_bounded_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device uchar* flag_count [[buffer(4)]],
    device uchar* resume_stage [[buffer(5)]], device half* resume_acc [[buffer(6)]],
    device half* fast_after [[buffer(7)]],
    device uchar* stage2 [[buffer(8)]], device half* prior2 [[buffer(9)]], device half* after2 [[buffer(10)]],
    device half* out [[buffer(11)]], device const half4* wq [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  // the simdgroup's eight pixels form a 4x2 block, so every 2x2 pool window is complete inside it
  const uint bx = group.x % 160u, by = group.x / 160u, oc0 = group.y * 16u + mc;
  if (by >= 184u || oc0 >= 32u) return;
  const uint px = bx * 4u + (mr & 3u), py = by * 2u + (mr >> 2u);
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
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 10u : 0u;   // ten channels per pixel
    simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
    for (uint kb = 0u; kb < 2u; ++kb) {
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const uint ic = kb * 8u + mc + h;
        const float av = (ok && ic < 10u) ? float(in[base + ic]) : 0.0f;   // channels 10..15 are the zero pad
        a[kb].thread_elements()[h] = av;
        qa[kb].thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
      }
      // packed weight/proxy table: one half4 per (kb, g) = (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for ic = kb*8 + mr
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        const half4 t = wq[wq_index(tile, kb, (oc0 + 8u * g) >> 1u, mr)];
        const float b0 = float(t.x), b1 = float(t.y);
        b[g][kb].thread_elements()[0] = b0; b[g][kb].thread_elements()[1] = b1;
        qb[g][kb].thread_elements()[0] = FORMAL ? float(t.z) : fabs(b0); qb[g][kb].thread_elements()[1] = FORMAL ? float(t.w) : fabs(b1);
      }
    }
    const uint stage_index = tile;
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
          error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 2.5f * ulp32_bits(M)   /* n*delta: at most n nonzero products lose a grid unit */ + 1.5625f * ulp32_bits(value);
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
#pragma clang loop unroll(full)
  for (uint g = 0u; g < 2u; ++g)
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint index = (py * 640u + px) * 32u + oc0 + 8u * g + h;
    const half v = activate(half(acc[g].thread_elements()[h]));
    inter[index] = v;
    if (!HW) { flag_count[index] = flags[g][h]; resume_stage[index] = first_stage[g][h]; resume_acc[index] = resume_val[g][h]; fast_after[index] = after_val[g][h]; }
    stage2[index] = second_stage[g][h]; prior2[index] = prior2_val[g][h]; after2[index] = after2_val[g][h];
    // 2x2 pool in the oracle's order: (x,y) + (x+1,y), + (x,y+1), + (x+1,y+1), * 0.25. Matrix row bit0 is
    // lane bit1 and row bit2 is lane bit4, so the window's other three values sit in lanes ^2, ^16, ^18.
    const ushort vb = as_type<ushort>(v);
    const half bb = as_type<half>(simd_shuffle_xor(vb, 2u)), c = as_type<half>(simd_shuffle_xor(vb, 16u)), d = as_type<half>(simd_shuffle_xor(vb, 18u));
    if ((mr & 1u) == 0u && (mr >> 2u) == 0u) {
      half p = half(v + bb); p = half(p + c); p = half(p + d);
      out[((py >> 1u) * 320u + (px >> 1u)) * 32u + oc0 + 8u * g + h] = half(p * half(0.25f));
    }
  }
}

// --- one exact stage, shared by pass 1 and the replay (scalar loads: 20-byte pixel stride) ------------
inline half conv_pre_stage(device const half* w, device const half* in, uint x, uint y, uint oc, uint t, half a) {
  int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
  bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
  uint base = ok ? (uint(iy) * 640u + uint(ix)) * 10u : 0u;
  ushort av[16], bv[16];
  for (uint k = 0u; k < 16u; ++k) { av[k] = (ok && k < 10u) ? as_type<ushort>(in[base + k]) : ushort(0); bv[k] = as_type<ushort>(w[wi(t, k, oc)]); }
  return fgu_hmma_f16_bits_16(av, bv, a);
}

// --- pass 1: first flagged stage exactly; disagree -> resume at start+1, list2. One flag and agree ->
//     done. Otherwise the second flagged stage from the fast pass's prior there (certified exact, since
//     slot 0 agreed and no stage in between was flagged); two flags and agree -> done; else resume at
//     second_stage+1 with that exact value and go to list2 ---------------------------------------------
kernel void fg_custom_block0_conv_pre_repair_pass1_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device uchar* resume_stage [[buffer(6)]], device half* resume_acc [[buffer(7)]],
    device const half* fast_after [[buffer(8)]], device const uchar* flag_count [[buffer(9)]],
    device uint* list2 [[buffer(10)]], device atomic_uint* counter2 [[buffer(11)]],
    device const uchar* stage2 [[buffer(12)]], device const half* prior2 [[buffer(13)]], device const half* after2 [[buffer(14)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  const half a = conv_pre_stage(w, in, x, y, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  const uint n = flag_count[linear];
  if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
    resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
    list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear; return;
  }
  if (n == 1u) return;
  const uint s2 = uint(stage2[linear]);
  const half a2 = conv_pre_stage(w, in, x, y, oc, s2, prior2[linear]);
  if (n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear])) return;
  resume_stage[linear] = uchar(s2 + 1u); resume_acc[linear] = a2;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- pass 2: the listed replay over list2, resuming; the v2 listed kernel on the clz helper -----------
kernel void fg_custom_block0_conv_pre_listed_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device const uchar* resume_stage [[buffer(6)]], device const half* resume_acc [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint t = start; t < 9u; ++t) a = conv_pre_stage(w, in, x, y, oc, t, a);
  inter[linear] = activate(a);
}

// --- pool: the scalar epilogue's arithmetic, verbatim (v2 pool, renamed) -----------------------------
kernel void fg_custom_block0_conv_pre_pool_wq(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                           uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 32u) return;
  const uint x = 2u * g.x, y = 2u * g.y, oc = g.z;
  const half a = inter[(y * 640u + x) * 32u + oc], bb = inter[(y * 640u + x + 1u) * 32u + oc];
  const half c = inter[((y + 1u) * 640u + x) * 32u + oc], d = inter[((y + 1u) * 640u + x + 1u) * 32u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[(g.y * 320u + g.x) * 32u + oc] = half(p * half(0.25f));
}

// --- pass 3: pass 2 rewrote inter[] for the elements in list2; re-pool just their windows -----------------
kernel void fg_custom_block0_conv_pre_repool_wq(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                                device const uint* list2 [[buffer(2)]], device const uint* counter2 [[buffer(3)]],
                                                uint tid [[thread_position_in_grid]]) {
  if (tid >= counter2[0]) return;
  const uint linear = list2[tid], oc = linear % 32u, pixel = linear / 32u, x = (pixel % 640u) & ~1u, y = (pixel / 640u) & ~1u;
  const half a = inter[(y * 640u + x) * 32u + oc], bb = inter[(y * 640u + x + 1u) * 32u + oc];
  const half c = inter[((y + 1u) * 640u + x) * 32u + oc], d = inter[((y + 1u) * 640u + x + 1u) * 32u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[((y >> 1u) * 320u + (x >> 1u)) * 32u + oc] = half(p * half(0.25f));
}
