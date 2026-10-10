// Windowed attention core (enc4/dec1 family): per (window, head, query block b of 16 tokens):
//   S(b, b') = Bpos_h(b, b') + Q(b) . K(b')^T    (fp32 over the 32 head channels, one fp16 rounding)
//   softmax per query row: t = clamp(S, +-21.640625); sq = t*t; pp = fma(sq, K1, 1); qq = fma(t, pp, K3); e = exp2(qq)
//     tot = tree64(e) in the transpiled order; rcp = half(1.0f / float(tot)); P = e * rcp          (all fp16 ops)
//   O(b, n) = sum_b' P(b, b') . V(b', n) with the layer's fp16 handoff table (bit per key block b' after which the
//     accumulator is rounded; entry index 2b + n)
// Q, K, V come from the per-token QKV tensor ([tokens][heads * 96]: head h columns 96h.. = Q | K | V, 32 each),
// gathered per element with reflection (out-of-image window tokens are keys/values of in-image queries); O is stored
// per in-image query token at columns 32h.. of the O tensor.
#include <metal_stdlib>
using namespace metal;

struct AttnParams {
  uint width, height;
  int shift_x, shift_y;
  uint qkv_stride, o_stride, heads, windows_x;
  uint split_table[8];      // bit b' set: round O(b, n) to fp16 after key block b' (index 2b + n)
  uint padded;              // 1: QKV/O live on the padded grid (the windows tile it exactly; no reflection, no clipping)
  uint wp;                  // padded row length
  uint pad2, pad3;
};

static inline int attn_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}

kernel void pwin_attn_window(device const half* QKV [[buffer(0)]],
                             device const half* Bpos [[buffer(1)]],    // [heads][64][64] fp16
                             device half* Out [[buffer(2)]],
                             constant AttnParams& p [[buffer(3)]],
                             uint simd_index [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint simdgroups [[simdgroups_per_threadgroup]]) {
  // task = (window, head, query block)
  const uint task = tg.x * simdgroups + simd_index;
  const uint b = task & 3u;
  const uint h = (task >> 2u) % p.heads;
  const uint w = (task >> 2u) / p.heads;
  const uint wy = w / p.windows_x, wx = w - wy * p.windows_x;
  const int x0 = int(wx) * 8 - p.shift_x;
  const int y0 = int(wy) * 8 - p.shift_y;
  if (y0 >= int(p.height) || x0 >= int(p.width)) return;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  // token t of the window -> row address in QKV (reflected)
  auto row_of = [&](uint t) -> device const half* {
    if (p.padded != 0u) {
      const uint tp = (wy * 8u + (t >> 3u)) * p.wp + wx * 8u + (t & 7u);
      return QKV + tp * p.qkv_stride + h * 96u;
    }
    const int x = attn_reflect(x0 + int(t & 7u), int(p.width));
    const int y = attn_reflect(y0 + int(t >> 3u), int(p.height));
    return QKV + (uint(y) * p.width + uint(x)) * p.qkv_stride + h * 96u;
  };
  // Q(b, k) tiles: rows = query tokens 16b + 8i + er, cols = head channels 16k + 8j + ec
  simdgroup_float8x8 Qm[2][2][2];   // [k][i][j]
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    device const half* qrow = row_of(16u * b + 8u * i + er);
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 2u; ++k)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        const half2 v = *reinterpret_cast<device const half2*>(qrow + 16u * k + 8u * j + ec);
        Qm[k][i][j].thread_elements()[0] = float(v.x);
        Qm[k][i][j].thread_elements()[1] = float(v.y);
      }
  }
  // scores for the 4 key blocks: S[b'][i][j] (rows = queries, cols = keys 16b' + 8j + ec)
  simdgroup_float8x8 S[4][2][2];
#pragma clang loop unroll(full)
  for (uint bp = 0u; bp < 4u; ++bp) {
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        simdgroup_half8x8 hm;
        simdgroup_load(hm, Bpos + (h * 64u + 16u * b + 8u * i) * 64u + 16u * bp + 8u * j, 64u);
        S[bp][i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
        S[bp][i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
      }
    // K(b')^T tiles: B[k' = channel 16k + 8i + er][n' = key token 16b' + 8j + ec + e]
#pragma clang loop unroll(full)
    for (uint k = 0u; k < 2u; ++k) {
      simdgroup_float8x8 kt[2][2];
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          device const half* r0 = row_of(16u * bp + 8u * j + ec) + 32u + 16u * k + 8u * i + er;
          device const half* r1 = row_of(16u * bp + 8u * j + ec + 1u) + 32u + 16u * k + 8u * i + er;
          kt[i][j].thread_elements()[0] = float(*r0);
          kt[i][j].thread_elements()[1] = float(*r1);
        }
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_multiply_accumulate(S[bp][i][j], Qm[k][i][0], kt[0][j], S[bp][i][j]);
          simdgroup_multiply_accumulate(S[bp][i][j], Qm[k][i][1], kt[1][j], S[bp][i][j]);
        }
    }
  }
  // softmax per query row, in registers: e values replace S; per lane the row's keys are 16b' + 8j + ec + e
  const half K1 = as_type<half>(ushort(0x91d5)), K3 = as_type<half>(ushort(0xc50d)), CL = as_type<half>(ushort(0x4d69));
  half e[4][2][2][2];   // [b'][i][j][element]
#pragma clang loop unroll(full)
  for (uint bp = 0u; bp < 4u; ++bp)
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j)
#pragma clang loop unroll(full)
        for (uint el = 0u; el < 2u; ++el) {
          const half s = half(S[bp][i][j].thread_elements()[el]);
          const half t = max(min(s, CL), -CL);
          const half sq = t * t;
          const half pp = fma(sq, K1, half(1.0h));
          const half qq = fma(t, pp, K3);
          e[bp][i][j][el] = exp2(qq);
        }
  // row sums: P1 = (v(0,0) + v(1,0)) + (v(2,0) + v(3,0)); P2 = same with j = 1; Qs = P1 + P2 (per parity el);
  // then across the 4 lanes of the row (lane bits 0 and 3): (Q0 + Q1) + (Q2 + Q3); tot = el0 + el1
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    half Qs[2];
#pragma clang loop unroll(full)
    for (uint el = 0u; el < 2u; ++el) {
      const half P1 = (e[0][i][0][el] + e[1][i][0][el]) + (e[2][i][0][el] + e[3][i][0][el]);
      const half P2 = (e[0][i][1][el] + e[1][i][1][el]) + (e[2][i][1][el] + e[3][i][1][el]);
      Qs[el] = P1 + P2;
    }
    half Sx[2];
#pragma clang loop unroll(full)
    for (uint el = 0u; el < 2u; ++el) {
      const half a = Qs[el] + simd_shuffle_xor(Qs[el], 1u);
      Sx[el] = a + simd_shuffle_xor(a, 8u);
    }
    const half tot = Sx[0] + Sx[1];
    const half rcp = half(1.0f / float(tot));
#pragma clang loop unroll(full)
    for (uint bp = 0u; bp < 4u; ++bp)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j)
#pragma clang loop unroll(full)
        for (uint el = 0u; el < 2u; ++el) S[bp][i][j].thread_elements()[el] = float(e[bp][i][j][el] * rcp);
  }
  // O(b, n) = sum_b' P(b, b') . V(b', n), rounding after the key blocks flagged in the split table
  simdgroup_float8x8 O[2][2][2];   // [n][i][j]
#pragma clang loop unroll(full)
  for (uint n = 0u; n < 2u; ++n)
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) O[n][i][j] = simdgroup_float8x8(0.0f);
#pragma clang loop unroll(full)
  for (uint bp = 0u; bp < 4u; ++bp) {
    // V(b', n) tiles: B[k' = key token 16b' + 8i + er][n' = channel 16n + 8j + ec + e]
#pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n) {
      simdgroup_float8x8 vt[2][2];
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i) {
        device const half* vrow = row_of(16u * bp + 8u * i + er) + 64u + 16u * n;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          const half2 v = *reinterpret_cast<device const half2*>(vrow + 8u * j + ec);
          vt[i][j].thread_elements()[0] = float(v.x);
          vt[i][j].thread_elements()[1] = float(v.y);
        }
      }
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_multiply_accumulate(O[n][i][j], S[bp][i][0], vt[0][j], O[n][i][j]);
          simdgroup_multiply_accumulate(O[n][i][j], S[bp][i][1], vt[1][j], O[n][i][j]);
        }
      if ((p.split_table[2u * b + n] >> bp) & 1u) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            O[n][i][j].thread_elements()[0] = float(half(O[n][i][j].thread_elements()[0]));
            O[n][i][j].thread_elements()[1] = float(half(O[n][i][j].thread_elements()[1]));
          }
      }
    }
  }
  // store O for in-image query tokens: rows 16b + 8i + er -> token (x0 + (t&7), y0 + (t>>3)), columns 32h + 16n + 8j + ec
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    const uint t = 16u * b + 8u * i + er;
    const int x = x0 + int(t & 7u), y = y0 + int(t >> 3u);
    device half* orow;
    if (p.padded != 0u) {
      orow = Out + ((wy * 8u + (t >> 3u)) * p.wp + wx * 8u + (t & 7u)) * p.o_stride + 32u * h;
    } else {
      if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height)) continue;
      orow = Out + (uint(y) * p.width + uint(x)) * p.o_stride + 32u * h;
    }
#pragma clang loop unroll(full)
    for (uint n = 0u; n < 2u; ++n)
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j)
        *reinterpret_cast<device half2*>(orow + 16u * n + 8u * j + ec) =
            half2(half(O[n][i][j].thread_elements()[0]), half(O[n][i][j].thread_elements()[1]));
  }
}
