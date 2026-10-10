#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_u32.metalh"
using namespace metal;

// block1 conv0 (module 62, launch 72), design six with the specialised K=8 tail (custom_block1_conv0_pipeline_pf2.metal)
// plus one micro-optimisation ("packed weight/proxy table", design thirteen, after custom_conv2_bounded_v13.metal):
//   - the bounded kernel's weight loads come from a host-built half4 table wq holding, per entry,
//     (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) with q(x) = x != 0 ? max(|x|, 2^-14) : 0 (== bound_operand of the
//     exactly widened half, in fp32), indexed (tile, slot, oc-pair, matrix_row) with slot = stage*2 + kb, i.e.
//     slot 0/1 = the K=16 stage's kb = 0/1, slot 2 = the tail stage's kb = 0 (the tail has no kb = 1 loads):
//       wq_index(tile, slot, pair, row) = ((tile*3 + slot)*8 + pair)*8 + row      9*3*8*8 = 1728 half4 = 13824 bytes
//     For fixed (tile, slot, g) the 32 lanes read pair = (mc + 8g)/2, row = mr: 32 consecutive half4 (256 bytes).
//     A lane loads one half4 per (kb, g) per stage (stage 0: 4 loads, tail: 2 loads) instead of 2 gathered halves
//     plus 2 bound_operand()s per (kb, g, h). Tail entries for rows >= 2 (k >= 2) are +0 in all four components,
//     exactly the (0.0f, bound_operand(0.0f) = fabs(0.0f) = +0) the _pf2 predicated load produced, so the tail's
//     `live_b = mr < 2` predicate is gone with it.
// Everything else (a loads, proxy on a, MMAs, fold_guard, error formulas, stage indices, checkpoints, fused pool,
// pass 1, pass 2, pool and re-pool) is copied verbatim from _pf2 under _wq names. The plain weight buffer w stays
// bound at buffer(0) for the repair kernels (and for signature parity); the bounded kernel no longer reads it.

constant bool FORMAL [[function_constant(0)]];

inline uint wi(uint tile, uint ic, uint oc) { return (oc * 9u + tile) * 18u + ic; }
inline uint wq_index(uint tile, uint slot, uint pair, uint row) { return ((tile * 3u + slot) * 8u + pair) * 8u + row; }
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline ushort zero_norm(half h) { const ushort b = as_type<ushort>(h); return b == 0x8000u ? ushort(0) : b; }
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline half activate(half a) { const half slope = half(0.01f); return min(max(max(a, half(a * slope)), half(-6.0f)), half(6.0f)); }

// The fold and the guard for one accumulator element; returns the rounded value that becomes the new prior.
// t1/b1 are the kb=1 partial and bound; the tail stage passes literal zeros and `tail = true` (a literal at
// every call site, so the branch folds after inlining).
inline float fold_guard(bool tail, float prior, float t0, float t1, float b0, float b1, uint stage_index,
                        thread uchar& flag, thread uchar& first, thread uchar& second,
                        thread half& resume, thread half& after, thread half& prior2, thread half& after2) {
  // two exact two_sums, errors added once: prior + t0 + t1 = s2 + (e0 + e1) up to one rounding of e0 + e1
  const float s1 = prior + t0, bp1 = s1 - prior, e0 = (prior - (s1 - bp1)) + (t0 - bp1);
  const float s2 = s1 + t1, bp2 = s2 - s1, e1 = (s1 - (s2 - bp2)) + (t1 - bp2);
  const float value = s2 + (e0 + e1);
  float error;
  if (FORMAL) {
    const float c0 = b0 * (1.0f + 0x1p-20f);
    if (tail) {
      const float M = max(0x1p-21f, max(c0, fabs(prior)));   // the oracle's exponent floor: max_exp >= -21 even when every term is smaller
      error = 3.5f * ulp32_bits(c0) + 0.5f * ulp32_bits(M) + 1.5625f * ulp32_bits(value);   /* tail: two live products -> 2*delta */
    } else {
      const float c1 = b1 * (1.0f + 0x1p-20f);
      const float M = max(0x1p-21f, max(max(c0, c1), fabs(prior)));
      error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.0f * ulp32_bits(M) + 1.5625f * ulp32_bits(value);   /* n*delta, n = 16 */
    }
  } else {
    if (tail) error = 4.0f * ulp32_bits(b0) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
    else error = 4.0f * ulp32_bits(b0) + 4.0f * ulp32_bits(b1) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
  }
  half rounded = half(value); if (as_type<ushort>(rounded) == 0x8000u) rounded = half(0.0h);   // rounds-to-zero is +0 (RTX)
  if (zero_norm(half(value - error)) != zero_norm(half(value + error))) {
    if (flag == 0u) { first = uchar(stage_index); resume = half(prior); after = rounded; }
    else if (flag == 1u) { second = uchar(stage_index); prior2 = half(prior); after2 = rounded; }
    flag = uchar(min(uint(flag) + 1u, 255u));
  }
  return float(rounded);
}

// --- fast pass: SIMD conv at 640x368 with the guard, 16 output channels per simdgroup -----------------------
kernel void fg_custom_block1_conv0_bounded_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device uchar* flag_count [[buffer(4)]],
    device uchar* resume_stage [[buffer(5)]], device half* resume_acc [[buffer(6)]], device half* fast_after [[buffer(7)]],
    device uchar* stage2 [[buffer(8)]], device half* prior2 [[buffer(9)]], device half* after2 [[buffer(10)]],
    device half* out [[buffer(11)]], device const half4* wq [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  // 4x2 pixel block per simdgroup so every 2x2 pool window is complete inside it
  const uint bx = group.x % 160u, by = group.x / 160u, oc0 = group.y * 16u + mc;
  if (by >= 184u || oc0 >= 16u) return;
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
  const simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint tile = 0u; tile < 9u; ++tile) {
    const int ix = int(px) + int(tile % 3u) - 1, iy = int(py) + int(tile / 3u) - 1;
    const bool ok = ix >= 0 && ix < 640 && iy >= 0 && iy < 368;
    const uint base = ok ? (uint(iy) * 640u + uint(ix)) * 18u : 0u;   // eighteen channels per pixel
    // ---- stage 0: channels 0..15 as two K=8 MMAs (unchanged) ----
    {
      simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb) {
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          const uint ic_a = kb * 8u + mc + h;
          const float av = ok ? float(in[base + ic_a]) : 0.0f;
          a[kb].thread_elements()[h] = av;
          qa[kb].thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
        }
#pragma clang loop unroll(full)
        for (uint g = 0u; g < 2u; ++g) {
          // one half4 per (kb, g): (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for oc = oc0 + 8g (even), k = kb*8 + mr
          const half4 t = wq[wq_index(tile, kb, (oc0 + 8u * g) >> 1u, mr)];
          const float b0 = float(t.x), b1 = float(t.y);
          b[g][kb].thread_elements()[0] = b0; b[g][kb].thread_elements()[1] = b1;
          qb[g][kb].thread_elements()[0] = FORMAL ? float(t.z) : fabs(b0); qb[g][kb].thread_elements()[1] = FORMAL ? float(t.w) : fabs(b1);
        }
      }
      const uint stage_index = tile * 2u;
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        const float prior0 = acc[g].thread_elements()[0], prior1 = acc[g].thread_elements()[1];
        simdgroup_float8x8 bnd0 = z, bnd1 = z, part0 = z, part1 = z;
        simdgroup_multiply_accumulate(bnd0, qa[0], qb[g][0], bnd0); simdgroup_multiply_accumulate(bnd1, qa[1], qb[g][1], bnd1);
        simdgroup_multiply_accumulate(part0, a[0], b[g][0], part0); simdgroup_multiply_accumulate(part1, a[1], b[g][1], part1);
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h)
          acc[g].thread_elements()[h] = fold_guard(false, h == 0u ? prior0 : prior1, part0.thread_elements()[h], part1.thread_elements()[h],
                     bnd0.thread_elements()[h], bnd1.thread_elements()[h], stage_index,
                     flags[g][h], first_stage[g][h], second_stage[g][h],
                     resume_val[g][h], after_val[g][h], prior2_val[g][h], after2_val[g][h]);
      }
    }
    // ---- stage 1, the tail: channels 16, 17 in k positions 0, 1 of k-block 0; one K=8 MMA per operand pair ----
    {
      simdgroup_float8x8 a, qa, b[2], qb[2];
      const bool live_a = mc == 0u;   // k_a = mc + h < 2 exactly when mc == 0 (h is 0 or 1)
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const float av = (ok && live_a) ? float(in[base + 16u + h]) : 0.0f;
        a.thread_elements()[h] = av;
        qa.thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
      }
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        // slot 2: rows 0, 1 hold channels 16, 17 (weight and proxy); rows 2..7 are +0 in every component, as the
        // _pf2 `live_b = mr < 2` predicate produced
        const half4 t = wq[wq_index(tile, 2u, (oc0 + 8u * g) >> 1u, mr)];
        const float b0 = float(t.x), b1 = float(t.y);
        b[g].thread_elements()[0] = b0; b[g].thread_elements()[1] = b1;
        qb[g].thread_elements()[0] = FORMAL ? float(t.z) : fabs(b0); qb[g].thread_elements()[1] = FORMAL ? float(t.w) : fabs(b1);
      }
      const uint stage_index = tile * 2u + 1u;
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        const float prior0 = acc[g].thread_elements()[0], prior1 = acc[g].thread_elements()[1];
        simdgroup_float8x8 bnd0 = z, part0 = z;
        simdgroup_multiply_accumulate(bnd0, qa, qb[g], bnd0);
        simdgroup_multiply_accumulate(part0, a, b[g], part0);
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h)
          acc[g].thread_elements()[h] = fold_guard(true, h == 0u ? prior0 : prior1, part0.thread_elements()[h], 0.0f,
                     bnd0.thread_elements()[h], 0.0f, stage_index,
                     flags[g][h], first_stage[g][h], second_stage[g][h],
                     resume_val[g][h], after_val[g][h], prior2_val[g][h], after2_val[g][h]);
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint g = 0u; g < 2u; ++g)
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint index = (py * 640u + px) * 16u + oc0 + 8u * g + h;
    const half v = activate(half(acc[g].thread_elements()[h]));
    inter[index] = v;
    // 2x2 pool in the oracle's order; the window's other values sit in lanes ^2, ^16, ^18
    const ushort vb = as_type<ushort>(v);
    const half pb = as_type<half>(simd_shuffle_xor(vb, 2u)), pc = as_type<half>(simd_shuffle_xor(vb, 16u)), pd = as_type<half>(simd_shuffle_xor(vb, 18u));
    if ((mr & 1u) == 0u && (mr >> 2u) == 0u) {
      half p = half(v + pb); p = half(p + pc); p = half(p + pd);
      out[((py >> 1u) * 320u + (px >> 1u)) * 16u + oc0 + 8u * g + h] = half(p * half(0.25f));
    }
    flag_count[index] = flags[g][h]; resume_stage[index] = first_stage[g][h]; resume_acc[index] = resume_val[g][h]; fast_after[index] = after_val[g][h];
    stage2[index] = second_stage[g][h]; prior2[index] = prior2_val[g][h]; after2[index] = after2_val[g][h];
  }
}

// --- one exact step (tile step/2, stage step%2) from prior a, the listed kernel's arithmetic on the clz helper.
// Scalar activation loads: an 18-half row is 36 bytes, so a stage start is only 4-byte aligned on odd pixels.
inline half b1c0_stage(device const half* w, device const half* in, uint x, uint y, uint oc, uint step, half a) {
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
  return fgu_hmma_f16_bits_16(av, bv, a);
}

// --- pass 1: two shortcut slots ----------------------------------------------------------------------------
kernel void fg_custom_block1_conv0_repair_pass1_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device uchar* resume_stage [[buffer(6)]], device half* resume_acc [[buffer(7)]],
    device const half* fast_after [[buffer(8)]], device const uchar* flag_count [[buffer(9)]],
    device uint* list2 [[buffer(10)]], device atomic_uint* counter2 [[buffer(11)]],
    device const uchar* stage2 [[buffer(12)]], device const half* prior2 [[buffer(13)]], device const half* after2 [[buffer(14)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 16u, pixel = linear / 16u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  half a = b1c0_stage(w, in, x, y, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
  const uint n = flag_count[linear];
  if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
    resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
    list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear; return;
  }
  if (n == 1u) return;
  // slot 0 agreed, so every step up to the second flagged one is certified and the fast pass's prior there is exact
  const uint s2 = uint(stage2[linear]);
  half a2 = b1c0_stage(w, in, x, y, oc, s2, prior2[linear]);
  if (n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear])) return;
  resume_stage[linear] = uchar(s2 + 1u); resume_acc[linear] = a2;
  list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
}

// --- pass 2: the listed kernel over list2, resuming, on the clz helper ---------------------------------------
kernel void fg_custom_block1_conv0_listed_wq(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device const uchar* resume_stage [[buffer(6)]], device const half* resume_acc [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 16u, pixel = linear / 16u, x = pixel % 640u, y = pixel / 640u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint step = start; step < 18u; ++step) a = b1c0_stage(w, in, x, y, oc, step, a);
  inter[linear] = activate(a);
}

// --- pool: the scalar epilogue's arithmetic, verbatim (unchanged from _pf) ---------------------------------
kernel void fg_custom_block1_conv0_pool_wq(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                            uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 16u) return;
  const uint x = 2u * g.x, y = 2u * g.y, oc = g.z;
  const half a = inter[(y * 640u + x) * 16u + oc], bb = inter[(y * 640u + x + 1u) * 16u + oc];
  const half c = inter[((y + 1u) * 640u + x) * 16u + oc], d = inter[((y + 1u) * 640u + x + 1u) * 16u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[(g.y * 320u + g.x) * 16u + oc] = half(p * half(0.25f));
}

kernel void fg_custom_block1_conv0_repool_wq(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
    device const uint* list2 [[buffer(2)]], device const uint* counter2 [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= counter2[0]) return;
  const uint linear = list2[tid], oc = linear % 16u, pixel = linear / 16u, x = (pixel % 640u) & ~1u, y = (pixel / 640u) & ~1u;
  const half a = inter[(y * 640u + x) * 16u + oc], bb = inter[(y * 640u + x + 1u) * 16u + oc];
  const half c = inter[((y + 1u) * 640u + x) * 16u + oc], d = inter[((y + 1u) * 640u + x + 1u) * 16u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[((y >> 1u) * 320u + (x >> 1u)) * 16u + oc] = half(p * half(0.25f));
}
