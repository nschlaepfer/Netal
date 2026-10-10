#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_u32.metalh"
using namespace metal;

// block1 conv1 (module 63, launch 73), design six. Port of custom_conv2_bounded_v6.metal /
// custom_conv2_repair_v5.metal onto custom_block1_conv1_pipeline_v2.metal. Deltas over _v2:
//   - one simdgroup computes 16 output channels (two 8-channel groups g = 0, 1) sharing the activation
//     matrices loaded once; grid y is channels / 16;
//   - a second shortcut slot per element: the second flagged stage, the prior going into it and the half
//     the fast pass produced there (three new buffers: uchar, half, half);
//   - the shorter exact fold (two two_sums, errors added once) with the value slack 1.5 -> 1.5625 (formal)
//     and 2.0 -> 2.0625 (empirical); the rest of both bound formulas is unchanged;
//   - pass 1 with the two-slot logic; pass 1 and pass 2 on hmma_exact_fast.metalh (clz) and, since a stage's
//     16 activations are one 32-byte aligned row here (16 channels per pixel), four ushort4 loads.
// The pool epilogue is unchanged (copied under a _pf name so this file stands alone in a metallib).
//
// _wq (design thirteen, from custom_conv2_bounded_v13.metal): the bounded kernel reads its weights and their
// bound proxies from a host-built half4 table instead of gathering 2 groups x 2 kb x 2 h halves through wi()
// and applying bound_operand() to each. Table entry (tile, kb, oc-pair, matrix_row) =
// (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) with q(x) = x != 0 ? max(|x|, 2^-14) : 0 (== bound_operand after the
// exact half->float widening); for fixed (tile, kb, g, group.y) the 32 lanes read 32 consecutive half4.
// wq_index(tile, kb, pair, row) = ((tile*2 + kb)*16 + pair)*8 + row; 9*2*16*8 = 2304 half4 (18432 bytes).
// Nothing else changes; pass 1 / pass 2 / pool still read w through wi(). Every kernel is copied under a _wq name.

//
// _ind: every kernel of the _wq file under an _ind name, with repair pass 1 also writing pass 2's indirect-dispatch
// arguments (port of custom_conv2_repair_v8.metal). Pass-1 arithmetic is unchanged; the early returns become
// predication so every thread reaches the threadgroup barrier. Four new buffers at the next free indices of pass 1:
// done2 (atomic uint, zeroed per run), indirect1 (read: [0] = the group count this dispatch was launched with, written
// by fg_compact_suspects_ordered_ind), indirect2 (write: {ceil(counter2/per_group2), 1, 1}), per_group2 (constant).
// After the group's work: threadgroup_barrier(mem_device) + device-scope seq_cst fence, one thread bumps done2, and the
// group observing done2 == indirect1[0]-1 fences again and writes indirect2. Needs -std=metal3.2.

constant bool FORMAL [[function_constant(0)]];

inline uint wi(uint tile, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u, lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 1152u + tile * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}
inline uint wq_index(uint tile, uint kb, uint pair, uint row) { return ((tile * 2u + kb) * 16u + pair) * 8u + row; }
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
inline half activate(half a) { const half slope = half(0.01f); return min(max(max(a, half(a * slope)), half(-6.0f)), half(6.0f)); }

// --- fast pass: SIMD conv at 320x184 with the guard, 16 output channels per simdgroup -----------------------
kernel void fg_custom_block1_conv1_bounded_ind(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device uchar* flag_count [[buffer(4)]],
    device uchar* resume_stage [[buffer(5)]], device half* resume_acc [[buffer(6)]], device half* fast_after [[buffer(7)]],
    device uchar* stage2 [[buffer(8)]], device half* prior2 [[buffer(9)]], device half* after2 [[buffer(10)]],
    device half* out [[buffer(11)]], device const half4* wq [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  // 4x2 pixel block per simdgroup so every 2x2 pool window is complete inside it
  const uint bx = group.x % 80u, by = group.x / 80u, oc0 = group.y * 16u + mc;
  if (by >= 92u || oc0 >= 32u) return;
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
    const bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
    const uint base = ok ? (uint(iy) * 320u + uint(ix)) * 16u : 0u;
    simdgroup_float8x8 a[2], qa[2], b[2][2], qb[2][2];
#pragma clang loop unroll(full)
    for (uint kb = 0u; kb < 2u; ++kb) {
#pragma clang loop unroll(full)
      for (uint h = 0u; h < 2u; ++h) {
        const float av = ok ? float(in[base + kb * 8u + mc + h]) : 0.0f;
        a[kb].thread_elements()[h] = av;
        qa[kb].thread_elements()[h] = FORMAL ? bound_operand(av) : fabs(av);
      }
#pragma clang loop unroll(full)
      for (uint g = 0u; g < 2u; ++g) {
        // one half4 per (kb, g): (w[oc], w[oc+1], q(w[oc]), q(w[oc+1])) for oc = oc0 + 8 g, ic = kb*8 + mr
        const half4 t = wq[wq_index(tile, kb, (oc0 + 8u * g) >> 1u, mr)];
        const float b0 = float(t.x), b1 = float(t.y);
        b[g][kb].thread_elements()[0] = b0; b[g][kb].thread_elements()[1] = b1;
        qb[g][kb].thread_elements()[0] = FORMAL ? float(t.z) : fabs(b0); qb[g][kb].thread_elements()[1] = FORMAL ? float(t.w) : fabs(b1);
      }
    }
    const uint stage_index = tile;
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
          error = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.25f * ulp32_bits(M) + 1.5625f * ulp32_bits(value);
        } else {
          error = 4.0f * ulp32_bits(bnd0.thread_elements()[h]) + 4.0f * ulp32_bits(bnd1.thread_elements()[h]) + 2.0625f * ulp32_bits(value) + 4.25f * 0x1p-44f;
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
#pragma clang loop unroll(full)
  for (uint g = 0u; g < 2u; ++g)
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    const uint index = (py * 320u + px) * 32u + oc0 + 8u * g + h;
    const half v = activate(half(acc[g].thread_elements()[h]));
    inter[index] = v;
    // 2x2 pool in the oracle's order; the window's other values sit in lanes ^2, ^16, ^18
    const ushort vb = as_type<ushort>(v);
    const half pb = as_type<half>(simd_shuffle_xor(vb, 2u)), pc = as_type<half>(simd_shuffle_xor(vb, 16u)), pd = as_type<half>(simd_shuffle_xor(vb, 18u));
    if ((mr & 1u) == 0u && (mr >> 2u) == 0u) {
      half p = half(v + pb); p = half(p + pc); p = half(p + pd);
      out[((py >> 1u) * 160u + (px >> 1u)) * 32u + oc0 + 8u * g + h] = half(p * half(0.25f));
    }
    flag_count[index] = flags[g][h]; resume_stage[index] = first_stage[g][h]; resume_acc[index] = resume_val[g][h]; fast_after[index] = after_val[g][h];
    stage2[index] = second_stage[g][h]; prior2[index] = prior2_val[g][h]; after2[index] = after2_val[g][h];
  }
}

// --- one exact stage (tile t) from prior a, the listed kernel's arithmetic on the clz helper ---------------
// A stage's 16 activations are the pixel's whole 16-channel row: base is a multiple of 16 halves (32 bytes),
// so four ushort4 loads are aligned.
inline half b1c1_stage(device const half* w, device const half* in, uint x, uint y, uint oc, uint t, half a) {
  int ix = int(x) + int(t % 3u) - 1, iy = int(y) + int(t / 3u) - 1;
  bool ok = ix >= 0 && ix < 320 && iy >= 0 && iy < 184;
  uint base = ok ? (uint(iy) * 320u + uint(ix)) * 16u : 0u;
  ushort av[16], bv[16];
  if (ok) {
    const ushort4 v0 = *reinterpret_cast<device const ushort4*>(in + base), v1 = *reinterpret_cast<device const ushort4*>(in + base + 4u),
                  v2 = *reinterpret_cast<device const ushort4*>(in + base + 8u), v3 = *reinterpret_cast<device const ushort4*>(in + base + 12u);
    av[0] = v0.x; av[1] = v0.y; av[2] = v0.z; av[3] = v0.w; av[4] = v1.x; av[5] = v1.y; av[6] = v1.z; av[7] = v1.w;
    av[8] = v2.x; av[9] = v2.y; av[10] = v2.z; av[11] = v2.w; av[12] = v3.x; av[13] = v3.y; av[14] = v3.z; av[15] = v3.w;
  } else for (uint k = 0u; k < 16u; ++k) av[k] = ushort(0);
  for (uint k = 0u; k < 16u; ++k) bv[k] = as_type<ushort>(w[wi(t, k, oc)]);
  return fgu_hmma_f16_bits_16(av, bv, a);
}

// --- pass 1: two shortcut slots; _ind also writes pass 2's indirect arguments (buffers 15..18) ----------------
kernel void fg_custom_block1_conv1_repair_pass1_ind(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device uchar* resume_stage [[buffer(6)]], device half* resume_acc [[buffer(7)]],
    device const half* fast_after [[buffer(8)]], device const uchar* flag_count [[buffer(9)]],
    device uint* list2 [[buffer(10)]], device atomic_uint* counter2 [[buffer(11)]],
    device const uchar* stage2 [[buffer(12)]], device const half* prior2 [[buffer(13)]], device const half* after2 [[buffer(14)]],
    device atomic_uint* done2 [[buffer(15)]], device const uint* indirect1 [[buffer(16)]], device uint* indirect2 [[buffer(17)]], constant uint& per_group2 [[buffer(18)]],
    uint tid [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
  if (tid < counter[0]) {
    const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 320u, y = pixel / 320u;
    const uint start = uint(resume_stage[linear]);
    half a = b1c1_stage(w, in, x, y, oc, start, start == 0u ? bias[oc] : resume_acc[linear]);
    const uint n = flag_count[linear];
    if (as_type<ushort>(a) != as_type<ushort>(fast_after[linear])) {
      resume_stage[linear] = uchar(start + 1u); resume_acc[linear] = a;
      list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
    } else if (n != 1u) {
      // slot 0 agreed, so every stage up to the second flagged one is certified and the fast pass's prior there is exact
      const uint s2 = uint(stage2[linear]);
      half a2 = b1c1_stage(w, in, x, y, oc, s2, prior2[linear]);
      if (!(n == 2u && as_type<ushort>(a2) == as_type<ushort>(after2[linear]))) {
        resume_stage[linear] = uchar(s2 + 1u); resume_acc[linear] = a2;
        list2[atomic_fetch_add_explicit(counter2, 1u, memory_order_relaxed)] = linear;
      }
    }
  }
  // release: this group's counter2 adds (and list2 writes) are ordered before its done add
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (lid == 0u) {
    const uint prev = atomic_fetch_add_explicit(done2, 1u, memory_order_relaxed);
    if (prev == indirect1[0] - 1u) {
      // acquire: every other group's done add, hence its counter2 adds, are ordered before this read
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      const uint c = atomic_load_explicit(counter2, memory_order_relaxed);
      indirect2[0] = (c + per_group2 - 1u) / per_group2; indirect2[1] = 1u; indirect2[2] = 1u;
    }
  }
}

// --- pass 2: the listed kernel over list2, resuming, on the clz helper ---------------------------------------
kernel void fg_custom_block1_conv1_listed_ind(
    device const half* w [[buffer(0)]], device const half* bias [[buffer(1)]], device const half* in [[buffer(2)]],
    device half* inter [[buffer(3)]], device const uint* list [[buffer(4)]], device const uint* counter [[buffer(5)]],
    device const uchar* resume_stage [[buffer(6)]], device const half* resume_acc [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= counter[0]) return;
  const uint linear = list[tid], oc = linear % 32u, pixel = linear / 32u, x = pixel % 320u, y = pixel / 320u;
  const uint start = uint(resume_stage[linear]);
  half a = start == 0u ? bias[oc] : resume_acc[linear];
  for (uint t = start; t < 9u; ++t) a = b1c1_stage(w, in, x, y, oc, t, a);
  inter[linear] = activate(a);
}

// --- pool: the scalar epilogue's arithmetic, verbatim (unchanged from _v2) ---------------------------------
kernel void fg_custom_block1_conv1_pool_ind(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
                                           uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 160u || g.y >= 92u || g.z >= 32u) return;
  const uint x = 2u * g.x, y = 2u * g.y, oc = g.z;
  const half a = inter[(y * 320u + x) * 32u + oc], bb = inter[(y * 320u + x + 1u) * 32u + oc];
  const half c = inter[((y + 1u) * 320u + x) * 32u + oc], d = inter[((y + 1u) * 320u + x + 1u) * 32u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[(g.y * 160u + g.x) * 32u + oc] = half(p * half(0.25f));
}

kernel void fg_custom_block1_conv1_repool_ind(device const half* inter [[buffer(0)]], device half* out [[buffer(1)]],
    device const uint* list2 [[buffer(2)]], device const uint* counter2 [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= counter2[0]) return;
  const uint linear = list2[tid], oc = linear % 32u, pixel = linear / 32u, x = (pixel % 320u) & ~1u, y = (pixel / 320u) & ~1u;
  const half a = inter[(y * 320u + x) * 32u + oc], bb = inter[(y * 320u + x + 1u) * 32u + oc];
  const half c = inter[((y + 1u) * 320u + x) * 32u + oc], d = inter[((y + 1u) * 320u + x + 1u) * 32u + oc];
  half p = half(a + bb); p = half(p + c); p = half(p + d);
  out[((y >> 1u) * 160u + (x >> 1u)) * 32u + oc] = half(p * half(0.25f));
}
