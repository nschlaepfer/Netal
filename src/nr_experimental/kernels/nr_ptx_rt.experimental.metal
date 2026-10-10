// Runtime for PTX-transliterated NR kernels (scripts/ptx_warp_to_msl.py): one Metal thread per PTX lane,
// one SIMD group per warp, threadgroup memory as shared memory. Every helper mirrors the semantics of
// scripts/ptx_warp_emu.py, which reproduces the recorded RTX launches byte for byte.
// Compile with -fno-fast-math -ffp-contract=off.
#pragma once
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---- E4M3 codec (identical to nr_vit_common.metal) ----
inline half nr_e4m3(uchar value) {
  uint exponent = (uint(value) >> 3u) & 15u, fraction = uint(value) & 7u;
  ushort sign = ushort(uint(value & 128u) << 8u);
  if (exponent == 0u) return as_type<half>(ushort(sign | as_type<ushort>(half(fraction) * half(0.001953125f))));
  if (exponent == 15u && fraction == 7u) return as_type<half>(ushort(sign | 0x7e00u));
  return as_type<half>(ushort(sign | ((exponent + 8u) << 10u) | (fraction << 7u)));
}
inline uint nr_rne_shift(uint value, uint shift) {
  if (shift == 0u) return value;
  uint q = value >> shift, r = value & ((1u << shift) - 1u), tie = 1u << (shift - 1u);
  return q + uint(r > tie || (r == tie && (q & 1u)));
}
inline uchar nr_to_e4m3(half value) {
  uint bits = as_type<ushort>(value), sign = (bits >> 8u) & 128u;
  uint exponent = (bits >> 10u) & 31u, fraction = bits & 1023u;
  if (exponent == 31u) return uchar(sign | (fraction ? 127u : 126u));
  if ((bits & 0x7fffu) >= 0x5f00u) return uchar(sign | 126u);
  if (exponent < 9u) {
    uint mantissa = exponent ? fraction + 1024u : fraction;
    return uchar(sign | nr_rne_shift(mantissa, exponent ? 16u - exponent : 15u));
  }
  uint mantissa = nr_rne_shift(1024u + fraction, 7u);
  uint e = exponent - 8u;
  if (mantissa == 16u) { mantissa = 8u; ++e; }
  return uchar(sign | min(126u, (e << 3u) + mantissa - 8u));
}
// f16x2 <-> e4m3x2 (packed pairs; the e4m3 pair lives in the low 16 bits of a b32 / a b16 register)
inline uint nr_cvt_e4m3x2_f16x2(uint f16x2) {
  return uint(nr_to_e4m3(as_type<half>(ushort(f16x2 & 0xffffu)))) | (uint(nr_to_e4m3(as_type<half>(ushort(f16x2 >> 16u)))) << 8u);
}
inline uint nr_cvt_f16x2_e4m3x2(uint e4m3x2) {
  return uint(as_type<ushort>(nr_e4m3(uchar(e4m3x2 & 0xffu)))) | (uint(as_type<ushort>(nr_e4m3(uchar((e4m3x2 >> 8u) & 0xffu)))) << 16u);
}
// ---- packed half arithmetic on b32 containers ----
inline half2 nr_h2(uint b) { return as_type<half2>(b); }
inline uint nr_b32(half2 v) { return as_type<uint>(v); }
inline float nr_f32(uint b) { return as_type<float>(b); }
inline uint nr_b32f(float v) { return as_type<uint>(v); }
inline half nr_h(ushort b) { return as_type<half>(b); }
inline ushort nr_b16(half v) { return as_type<ushort>(v); }
// ---- RTX rsqrt/rcp.approx on half-valued f32 (tables: one entry per positive finite half) ----
inline float nr_rsqrt_approx(device const ushort* table, float x) {
  half hb = half(x);
  if (float(hb) == x && x > 0.0f && as_type<ushort>(hb) <= 0x7bffu) return float(as_type<half>(table[uint(as_type<ushort>(hb)) - 1u]));
  return 1.0f / sqrt(x);      // fallback (unused by the recorded kernels)
}
inline float nr_rcp_approx(device const ushort* table, float x) {
  half hb = half(x);
  if (float(hb) == x && x > 0.0f && as_type<ushort>(hb) <= 0x7bffu) return float(as_type<half>(table[uint(as_type<ushort>(hb)) - 1u]));
  return 1.0f / x;
}
// ---- warp primitives ----
inline uint nr_shfl_idx(uint v, uint idx) { return simd_shuffle(v, ushort(idx & 31u)); }
inline uint nr_shfl_bfly(uint v, uint m) { return simd_shuffle_xor(v, ushort(m & 31u)); }
inline uint nr_prmt(uint a, uint b, uint sel) {
  uint out = 0u;
  for (uint i = 0u; i < 4u; ++i) {
    uint s = (sel >> (4u * i)) & 15u; uint byte = ((s & 7u) < 4u) ? ((a >> (8u * (s & 7u))) & 0xffu) : ((b >> (8u * ((s & 7u) - 4u))) & 0xffu);
    if (s & 8u) byte = (byte & 0x80u) ? 0xffu : 0u;
    out |= byte << (8u * i);
  }
  return out;
}
// movmatrix.sync.trans.aligned.m8n8.b16: lane l holds row l/4, cols 2(l%4), 2(l%4)+1 of an 8x8 b16 matrix; returns the transpose in the same layout
inline uint nr_movmatrix(uint v, uint lane) {
  const uint g = lane / 4u, t = lane % 4u;
  // T[g][2t] = M[2t][g] held by lane 4*(2t) + g/2, half g%2 ; T[g][2t+1] = M[2t+1][g] held by lane 4*(2t+1) + g/2
  uint lo_src = simd_shuffle(v, ushort(8u * t + g / 2u)), hi_src = simd_shuffle(v, ushort(8u * t + 4u + g / 2u));
  uint lo = (g & 1u) ? (lo_src >> 16u) : (lo_src & 0xffffu), hi = (g & 1u) ? (hi_src >> 16u) : (hi_src & 0xffffu);
  return lo | (hi << 16u);
}
// ---- mma.sync.m16n8k32.row.col.f16.e4m3.e4m3.f16 ----
// Lane l: g = l/4, t = l%4. A: a0 = row g slots 4t..4t+3, a1 = row g+8, a2 = row g slots 16+4t.., a3 = row g+8 slots 16+4t..
// B: b0 = col g slots 4t.., b1 = col g slots 16+4t.. ; C/D: c0 = (row g, cols 2t, 2t+1), c1 = (row g+8, ...).
// QMMA rule: exact products, fp32 FMA chain in slot order starting from the half C, one rounding to half.
inline ulong nr_round_shift_rne64(ulong value, uint shift);
inline half nr_fixed_to_half_rne(long value, int scale);
// e4m3 byte -> integer significand and exponent: value = sig * 2^exp (normal: sig = 8+m, exp = e-10; subnormal: sig = m, exp = -9)
inline void nr_e4m3_fields(uchar b, thread int& sig, thread int& ex) {
  const int e = int((b >> 3u) & 15u), m = int(b & 7u);
  if (e == 0) { sig = m; ex = -9; } else { sig = 8 + m; ex = e - 10; }
}
#ifndef NR_CLZ_BITLEN
#define NR_CLZ_BITLEN 1
#endif
#if NR_CLZ_BITLEN
inline int nr_bitlen(int v) { return v > 0 ? 32 - int(clz(uint(v))) : 0; }
inline uint nr_bitlen64_minus1(ulong m) { const uint hi = uint(m >> 32u); return hi != 0u ? 63u - clz(hi) : 31u - clz(uint(m)); }   // floor(log2 m), m > 0
#else
inline int nr_bitlen(int v) { int n = 0; while (v > 0) { v >>= 1; ++n; } return n; }
inline uint nr_bitlen64_minus1(ulong m) { uint leading = 0u; for (ulong scan = m; scan > 1ul; scan >>= 1u) ++leading; return leading; }
#endif
// QMMA m16n8k32 e4m3 -> f16, the rule that matches every RTX register of the 64-case discriminator and launch 151/152's
// operands (Codex audit 2026-09-05, FG HMMA-style): max_exp = max over nonzero products of floor(log2|a|)+floor(log2|b|)
// and floor(log2|C|); every term's magnitude truncated to units 2^(max_exp-25); exact signed sum; one RNE to half.
#ifndef NR_QMMA_U32
#define NR_QMMA_U32 0
#endif
#ifndef NR_QMMA_EXACTSUM
#define NR_QMMA_EXACTSUM 0
#endif
#ifndef NR_QMMA_QGUARD
#define NR_QMMA_QGUARD 0
#endif
#ifndef NR_QMMA_GUARDED
#define NR_QMMA_GUARDED 0
#endif
#if NR_QMMA_QGUARD
// Guard without an error bound. Every term is an integer multiple of 2^qmin (products: sig < 2^8 at 2^(qa+qb); C: sig < 2^11 at 2^cq). If
// max_exp - qmin <= 17 the sum of the 33 magnitudes is below 33 * 2^18 * 2^qmin < 2^24 * 2^qmin, so every partial sum of the fp32 FMA chain is an
// exactly representable integer multiple of 2^qmin: the chain is exact in any order, no term lies below the 2^(max_exp-25) grid, and the fixed-point
// rule reduces to RNE_half(exact sum) = half(chain). Subnormal or overflowing results go to the exact loop. Otherwise the exact loop runs.
inline half nr_qmma_element_trunc(thread const uchar* arow, thread const uchar* bcol, half prior);
inline half nr_qmma_element(thread const uchar* arow, thread const uchar* bcol, half prior) {
  int maxexp = -100000, minq = 100000;
  for (uint k = 0u; k < 32u; ++k) {
    const uchar ab = arow[k], bb = bcol[k]; const int ea = int((ab >> 3u) & 15u), eb = int((bb >> 3u) & 15u), ma = int(ab & 7u), mb = int(bb & 7u);
    const int sa = (ea == 0) ? ma : 8 + ma, sb = (eb == 0) ? mb : 8 + mb; if (sa == 0 || sb == 0) continue;
    const int qa = (ea == 0) ? -9 : ea - 10, qb = (eb == 0) ? -9 : eb - 10;
    maxexp = max(maxexp, (qa + (31 - int(clz(uint(sa))))) + (qb + (31 - int(clz(uint(sb)))))); minq = min(minq, qa + qb);
  }
  const ushort cb = as_type<ushort>(prior); const int cf = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  if (cf != 0) { maxexp = max(maxexp, cf - 15); minq = min(minq, cf - 25); } else if (cfr != 0) { maxexp = max(maxexp, -24 + (31 - int(clz(uint(cfr))))); minq = min(minq, -24); }
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  if (maxexp - minq <= 17) {
    float acc = float(prior);
    for (uint k = 0u; k < 32u; ++k) acc = fma(float(nr_e4m3(arow[k])), float(nr_e4m3(bcol[k])), acc);      // exact: integer multiples of 2^qmin below 2^24
    const float a = fabs(acc);
    if (a >= 6.103515625e-05f && a < 65504.0f) return half(acc);                                         // normal, non-overflowing: RNE == the rule's rounding
  }
  return nr_qmma_element_trunc(arow, bcol, prior);
}
inline half nr_qmma_element_trunc(thread const uchar* arow, thread const uchar* bcol, half prior) {
#elif NR_QMMA_GUARDED
// Guarded fast path, provably identical to the fixed-point rule: (1) the fp32 FMA chain over the 32 exact products and C is computed together
// with T = sum of |terms| (fp32); its distance to the exact sum is below 34 * 2^-24 * T; (2) from one cheap exponent pass we know max_exp and
// the smallest term scale; if no term lies below the 2^(max_exp-25) grid, the fixed-point result is RNE_half(exact sum); (3) if the fp32 sum is
// farther than the error bound from every half rounding midpoint (and not subnormal/overflowing), RNE_half(fp32 sum) == RNE_half(exact sum).
// Otherwise fall back to the exact integer loop. Any deviation from the exact library is a bug in the bound, so the A/B must stay bit-identical.
inline half nr_qmma_element_trunc(thread const uchar* arow, thread const uchar* bcol, half prior);
inline half nr_qmma_element(thread const uchar* arow, thread const uchar* bcol, half prior) {
  float acc = float(prior), tsum = fabs(float(prior)); int maxexp = -100000, minq = 100000;
  for (uint k = 0u; k < 32u; ++k) {
    const uchar ab = arow[k], bb = bcol[k]; const int ea = int((ab >> 3u) & 15u), eb = int((bb >> 3u) & 15u), ma = int(ab & 7u), mb = int(bb & 7u);
    const int sa = (ea == 0) ? ma : 8 + ma, sb = (eb == 0) ? mb : 8 + mb; if (sa == 0 || sb == 0) continue;
    const int qa = (ea == 0) ? -9 : ea - 10, qb = (eb == 0) ? -9 : eb - 10;
    const float pa = float(nr_e4m3(ab)), pb = float(nr_e4m3(bb)); const float prod = pa * pb;        // exact: 4-bit x 4-bit significands
    acc = fma(pa, pb, acc); tsum += fabs(prod);
    maxexp = max(maxexp, (qa + (31 - int(clz(uint(sa))))) + (qb + (31 - int(clz(uint(sb)))))); minq = min(minq, qa + qb);
  }
  const ushort cb = as_type<ushort>(prior); const int cf = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  if (cf != 0) { maxexp = max(maxexp, cf - 15); minq = min(minq, cf - 25); } else if (cfr != 0) { maxexp = max(maxexp, -24 + (31 - int(clz(uint(cfr))))); minq = min(minq, -24); }
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  if (minq >= maxexp - 25 && tsum < 32768.0f) {
    const float bound = tsum * 2.03e-6f;                                                              // 34 * 2^-24 * T
    const half h = half(acc); const float hf = float(h); const uint hb = as_type<ushort>(h) & 0x7fffu;
    if (hb >= 0x0400u && hb < 0x7bffu) {                                                              // normal half, not overflowing
      const float ulp = as_type<float>((uint(hb >> 10u) + 112u - 10u) << 23u);                       // 2^(e-10): the half ulp at |h|
      const float dist = 0.5f * ulp - fabs(acc - hf);                                                 // distance of the fp32 sum to the nearest rounding midpoint
      if (dist > bound && acc != 0.0f) return h;
    }
  }
  return nr_qmma_element_trunc(arow, bcol, prior);
}
inline half nr_qmma_element_trunc(thread const uchar* arow, thread const uchar* bcol, half prior) {
#elif NR_QMMA_EXACTSUM
// Same rule, one pass: every term is an exact integer in units 2^-24 (e4m3 products: sig < 2^8 at 2^(qa+qb) with qa+qb >= -18; half C: sig < 2^11 at
// 2^cq with cq >= -24), so the exact signed sum fits in 64 bits. When no term lies below the 2^(max_exp-25) grid (checked from the same pass), the
// fixed-point rule's truncations are no-ops and its result is RNE(exact sum); otherwise fall back to the truncating loop (rare: ~0.1% of elements).
inline half nr_qmma_element_trunc(thread const uchar* arow, thread const uchar* bcol, half prior);
inline half nr_qmma_element(thread const uchar* arow, thread const uchar* bcol, half prior) {
  long total = 0l; int maxexp = -100000, minq = 100000;
  for (uint k = 0u; k < 32u; ++k) {
    const uchar ab = arow[k], bb = bcol[k]; const int ea = int((ab >> 3u) & 15u), eb = int((bb >> 3u) & 15u), ma = int(ab & 7u), mb = int(bb & 7u);
    const int sa = (ea == 0) ? ma : 8 + ma, sb = (eb == 0) ? mb : 8 + mb; const int ps = sa * sb;
    if (ps == 0) continue;
    const int q = ((ea == 0) ? -9 : ea - 10) + ((eb == 0) ? -9 : eb - 10);                        // product = ps * 2^q
    const int e = q + (31 - int(clz(uint(ps))));                                                    // floor(log2 product) (operand-exponent sum equals this only when both sigs are normal; use the rule's definition below)
    const int ea_f = (sa > 0 ? ((ea == 0) ? -9 : ea - 10) + (31 - int(clz(uint(sa)))) : 0), eb_f = ((eb == 0) ? -9 : eb - 10) + (31 - int(clz(uint(sb))));
    maxexp = max(maxexp, ea_f + eb_f); minq = min(minq, q); (void)e;
    const long mag = long(ps) << uint(q + 24);
    total += (((ab ^ bb) & 0x80u) != 0u) ? -mag : mag;
  }
  const ushort cb = as_type<ushort>(prior); const int cf = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  int csig = 0, cq = 0;
  if (cf != 0) { csig = 1024 + cfr; cq = cf - 25; } else if (cfr != 0) { csig = cfr; cq = -24; }
  if (csig != 0) { maxexp = max(maxexp, cq + (31 - int(clz(uint(csig))))); minq = min(minq, cq); const long mag = long(csig) << uint(cq + 24); total += (cb & 0x8000u) != 0u ? -mag : mag; }
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  if (minq >= maxexp - 25) return nr_fixed_to_half_rne(total, -24);          // no term below the grid: fixed-point result == RNE(exact sum)
  return nr_qmma_element_trunc(arow, bcol, prior);
}
inline half nr_qmma_element_trunc(thread const uchar* arow, thread const uchar* bcol, half prior) {
#else
inline half nr_qmma_element(thread const uchar* arow, thread const uchar* bcol, half prior) {
#endif
  int psig[32], pexp[32], pq[32]; bool pneg[32]; int maxexp = -100000;
  for (uint k = 0u; k < 32u; ++k) {
    int sa, ea, sb, eb; nr_e4m3_fields(arow[k], sa, ea); nr_e4m3_fields(bcol[k], sb, eb);
    psig[k] = sa * sb; pq[k] = ea + eb; pneg[k] = ((arow[k] ^ bcol[k]) & 0x80u) != 0u;
    pexp[k] = (sa > 0 && sb > 0) ? (ea + nr_bitlen(sa) - 1) + (eb + nr_bitlen(sb) - 1) : -100000;
    if (psig[k] != 0) maxexp = max(maxexp, pexp[k]);
  }
  const ushort cb = as_type<ushort>(prior); const int cf = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  int csig = 0, cq = 0;
  if (cf != 0) { csig = 1024 + cfr; cq = cf - 25; } else if (cfr != 0) { csig = cfr; cq = -24; }
  if (csig != 0) maxexp = max(maxexp, cq + nr_bitlen(csig) - 1);
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25;
#if NR_QMMA_U32
  // every aligned term is < 2^26 (a term at the maximum exponent is sig << (26 - bitlen(sig))), 33 of them fit an unsigned 32-bit sum:
  // accumulate positive and negative magnitudes in 32-bit and combine once, avoiding emulated 64-bit arithmetic per term
  uint pos = 0u, neg = 0u;
  for (uint k = 0u; k < 32u; ++k) {
    if (psig[k] == 0) continue;
    const int sh = pq[k] - scale; const uint mag = (sh >= 0) ? (uint(psig[k]) << uint(sh)) : (sh <= -31 ? 0u : (uint(psig[k]) >> uint(-sh)));
    if (pneg[k]) neg += mag; else pos += mag;
  }
  if (csig != 0) {
    const int sh = cq - scale; const uint mag = (sh >= 0) ? (uint(csig) << uint(sh)) : (sh <= -31 ? 0u : (uint(csig) >> uint(-sh)));
    if ((cb & 0x8000u) != 0u) neg += mag; else pos += mag;
  }
  return nr_fixed_to_half_rne(long(pos) - long(neg), scale);
#else
  long total = 0l;
  for (uint k = 0u; k < 32u; ++k) {
    if (psig[k] == 0) continue;
    const int sh = pq[k] - scale; long mag = long(psig[k]);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += pneg[k] ? -mag : mag;
  }
  if (csig != 0) {
    const int sh = cq - scale; long mag = long(csig);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (cb & 0x8000u) != 0u ? -mag : mag;
  }
  return nr_fixed_to_half_rne(total, scale);
#endif
}
#ifndef NR_QMMA_FAST
#define NR_QMMA_FAST 0
#endif
#ifndef NR_QMMA_PACKED
#define NR_QMMA_PACKED 0
#endif
#ifndef NR_QMMA_FSCALE
#define NR_QMMA_FSCALE 1
#endif
#if NR_QMMA_FSCALE
// Candidate 11 (corrected 2026-09-05 after Codex's field-alignment audit; CPU check scripts/nr-qmma-rule-check.py: 0/4096 RTX registers
// differ, the product-exponent form differs in 2): the same rule in float-scaled form. e4m3 x e4m3 products are exact in fp32
// (<= 8 significant bits). max_exp = max over nonzero products of floor(log2|a|) + floor(log2|b|) (the fp32 exponent fields of the
// operands, NOT of the product: a significand carry would move the grid) and floor(log2|C|). Per-term truncation to units 2^(max_exp-25)
// is trunc(p * 2^(25-max_exp)): the scaled magnitude is < 2^27 and any fp32 value >= 2^24 with <= 8 significant bits is already an
// integer, so trunc is exact; 32 products + C sum to at most 3,841,949,696 < 2^32 for finite FP8 operands with a half C (Codex bound),
// so unsigned 32-bit pos/neg magnitude sums are exact. Not valid for unrestricted half x half operands. One RNE pack as before.
inline float nr_e4m3_to_f32(uint b) {
  const uint e = (b >> 3u) & 15u, m = b & 7u, sgn = (b & 0x80u) << 24u;
  const float normal = as_type<float>(sgn | ((e + 120u) << 23u) | (m << 20u));
  const float sub = as_type<float>(sgn | as_type<uint>(float(m) * 0.001953125f));          // m * 2^-9, exact
  return (e == 0u) ? sub : normal;
}
inline half nr_qmma_element_fscale(thread const uchar* arow, thread const uchar* bcol, half prior) {
  float p[32]; int maxexp = -300;
  for (uint k = 0u; k < 32u; ++k) {
    const float fa = nr_e4m3_to_f32(uint(arow[k])), fb = nr_e4m3_to_f32(uint(bcol[k])); p[k] = fa * fb;
    const int ex = int((as_type<uint>(fa) >> 23u) & 255u) + int((as_type<uint>(fb) >> 23u) & 255u) - 254;
    if (p[k] != 0.0f) maxexp = max(maxexp, ex);
  }
  const float cf = float(prior);
  if (cf != 0.0f) maxexp = max(maxexp, int((as_type<uint>(cf) >> 23u) & 255u) - 127);
  if (maxexp == -300) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25; const float fs = as_type<float>(uint(127 + 25 - maxexp) << 23u);   // 2^(25 - max_exp)
  uint pos = 0u, neg = 0u;
  for (uint k = 0u; k < 32u; ++k) { const float t = trunc(p[k] * fs); const uint m = uint(fabs(t)); if (t < 0.0f) neg += m; else pos += m; }
  { const float t = trunc(cf * fs); const uint m = uint(fabs(t)); if (t < 0.0f) neg += m; else pos += m; }
  return nr_fixed_to_half_rne(long(pos) - long(neg), scale);
}
#endif
// Candidate 10: the same rule over packed 32-bit operand words (a8[i] holds e4m3 bytes k = 4i..4i+3), fully unrolled so every
// index is a constant and no per-product state lives in thread arrays; the second pass re-decodes instead of reading psig/pq/pexp.
inline half nr_qmma_element_packed(thread const uint* a8, thread const uint* b8, half prior) {
  int maxexp = -100000;
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) {
    const uint wa = a8[i], wb = b8[i];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const uint ab = (wa >> (8u * j)) & 0xffu, bb = (wb >> (8u * j)) & 0xffu;
      const int ea = int((ab >> 3u) & 15u), eb = int((bb >> 3u) & 15u);
      const int sa = (ea == 0) ? int(ab & 7u) : 8 + int(ab & 7u), sb = (eb == 0) ? int(bb & 7u) : 8 + int(bb & 7u);
      const int qa = (ea == 0) ? -9 : ea - 10, qb = (eb == 0) ? -9 : eb - 10;
      if (sa != 0 && sb != 0) maxexp = max(maxexp, qa + qb + (62 - int(clz(uint(sa))) - int(clz(uint(sb)))));
    }
  }
  const ushort cb = as_type<ushort>(prior); const int cf = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  int csig = 0, cq = 0;
  if (cf != 0) { csig = 1024 + cfr; cq = cf - 25; } else if (cfr != 0) { csig = cfr; cq = -24; }
  if (csig != 0) maxexp = max(maxexp, cq + 31 - int(clz(uint(csig))));
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25;
  long total = 0l;
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) {
    const uint wa = a8[i], wb = b8[i];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
      const uint ab = (wa >> (8u * j)) & 0xffu, bb = (wb >> (8u * j)) & 0xffu;
      const int ea = int((ab >> 3u) & 15u), eb = int((bb >> 3u) & 15u);
      const int sa = (ea == 0) ? int(ab & 7u) : 8 + int(ab & 7u), sb = (eb == 0) ? int(bb & 7u) : 8 + int(bb & 7u);
      const int psig = sa * sb; if (psig == 0) continue;
      const int qa = (ea == 0) ? -9 : ea - 10, qb = (eb == 0) ? -9 : eb - 10;
      const int sh = qa + qb - scale; long mag = long(psig);
      mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
      total += (((ab ^ bb) & 0x80u) != 0u) ? -mag : mag;
    }
  }
  if (csig != 0) {
    const int sh = cq - scale; long mag = long(csig);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (cb & 0x8000u) != 0u ? -mag : mag;
  }
  return nr_fixed_to_half_rne(total, scale);
}
// Same rule as nr_qmma_element, restructured: each operand byte is decoded once per fragment (not once per output element) into
// (signed significand, value exponent floor(log2|v|), scale q), bit lengths via clz. Must stay bit-identical to nr_qmma_element.
struct NrE4M3Dec { int sig; int q; int e; };      // |value| = sig * 2^q, e = floor(log2 |value|) (sig > 0), sign in the byte
inline NrE4M3Dec nr_e4m3_decode(uchar b) {
  NrE4M3Dec d; const int ee = int((b >> 3u) & 15u), m = int(b & 7u);
  if (ee == 0) { d.sig = m; d.q = -9; } else { d.sig = 8 + m; d.q = ee - 10; }
  d.e = d.sig > 0 ? d.q + (31 - int(clz(uint(d.sig)))) : -100000; return d;
}
inline half nr_qmma_element_fast(thread const NrE4M3Dec* a, thread const uchar* ab, thread const NrE4M3Dec* b, thread const uchar* bb, half prior) {
  int maxexp = -100000;
  for (uint k = 0u; k < 32u; ++k) if (a[k].sig != 0 && b[k].sig != 0) maxexp = max(maxexp, a[k].e + b[k].e);
  const ushort cb = as_type<ushort>(prior); const int cf = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  int csig = 0, cq = 0;
  if (cf != 0) { csig = 1024 + cfr; cq = cf - 25; } else if (cfr != 0) { csig = cfr; cq = -24; }
  if (csig != 0) maxexp = max(maxexp, cq + (31 - int(clz(uint(csig)))));
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25; long total = 0l;
  for (uint k = 0u; k < 32u; ++k) {
    const int ps = a[k].sig * b[k].sig; if (ps == 0) continue;
    const int sh = a[k].q + b[k].q - scale; long mag = long(ps);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (((ab[k] ^ bb[k]) & 0x80u) != 0u) ? -mag : mag;
  }
  if (csig != 0) {
    const int sh = cq - scale; long mag = long(csig);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (cb & 0x8000u) != 0u ? -mag : mag;
  }
  return nr_fixed_to_half_rne(total, scale);
}
inline void nr_mma_e4m3(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, uint c0, uint c1, thread uint& d0, thread uint& d1, uint lane) {
  const uint g = lane / 4u, t = lane % 4u;
#if NR_QMMA_PACKED
  uint pa0[8], pa1[8], pb0[8], pb1[8];
#pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q) {
    const ushort src_a = ushort(4u * g + q), src_b0 = ushort(4u * (2u * t) + q), src_b1 = ushort(4u * (2u * t + 1u) + q);
    pa0[q] = simd_shuffle(a0, src_a); pa0[4u + q] = simd_shuffle(a2, src_a); pa1[q] = simd_shuffle(a1, src_a); pa1[4u + q] = simd_shuffle(a3, src_a);
    pb0[q] = simd_shuffle(b0, src_b0); pb0[4u + q] = simd_shuffle(b1, src_b0); pb1[q] = simd_shuffle(b0, src_b1); pb1[4u + q] = simd_shuffle(b1, src_b1);
  }
  const half2 pc0 = as_type<half2>(c0), pc1 = as_type<half2>(c1);
  d0 = as_type<uint>(half2(nr_qmma_element_packed(pa0, pb0, pc0.x), nr_qmma_element_packed(pa0, pb1, pc0.y)));
  d1 = as_type<uint>(half2(nr_qmma_element_packed(pa1, pb0, pc1.x), nr_qmma_element_packed(pa1, pb1, pc1.y)));
  return;
#endif
  uchar arow0[32], arow1[32], bcol0[32], bcol1[32];
  for (uint q = 0u; q < 4u; ++q) {
    const ushort src_a = ushort(4u * g + q);                 // lanes holding rows g and g+8
    const uint ra0 = simd_shuffle(a0, src_a), ra1 = simd_shuffle(a1, src_a), ra2 = simd_shuffle(a2, src_a), ra3 = simd_shuffle(a3, src_a);
    const ushort src_b0 = ushort(4u * (2u * t) + q), src_b1 = ushort(4u * (2u * t + 1u) + q);   // lanes holding cols 2t, 2t+1
    const uint rb00 = simd_shuffle(b0, src_b0), rb01 = simd_shuffle(b1, src_b0), rb10 = simd_shuffle(b0, src_b1), rb11 = simd_shuffle(b1, src_b1);
    for (uint j = 0u; j < 4u; ++j) {
      arow0[4u * q + j] = uchar((ra0 >> (8u * j)) & 0xffu); arow0[16u + 4u * q + j] = uchar((ra2 >> (8u * j)) & 0xffu);
      arow1[4u * q + j] = uchar((ra1 >> (8u * j)) & 0xffu); arow1[16u + 4u * q + j] = uchar((ra3 >> (8u * j)) & 0xffu);
      bcol0[4u * q + j] = uchar((rb00 >> (8u * j)) & 0xffu); bcol0[16u + 4u * q + j] = uchar((rb01 >> (8u * j)) & 0xffu);
      bcol1[4u * q + j] = uchar((rb10 >> (8u * j)) & 0xffu); bcol1[16u + 4u * q + j] = uchar((rb11 >> (8u * j)) & 0xffu);
    }
  }
  const half2 c0h = as_type<half2>(c0), c1h = as_type<half2>(c1);
#if NR_EXPERIMENTAL_APPROX_FP32
  // EXPERIMENTAL LANE ONLY (Opus). Same weights, same graph, same dispatch geometry: only the accumulation
  // rule changes, from the exact fixed-point emulation of the RTX QMMA rule to a native fp32 fma chain. This
  // block is a REORDER of the NR_QMMA_MEASURE_FP32 path that already exists further down; it is placed ahead of
  // the FSCALE return because FSCALE must stay DEFINED for the guarded and threadgroup paths that call
  // nr_qmma_element_fscale directly. Removing this one region restores the accepted source exactly.
  {
    float m00 = float(c0h.x), m01 = float(c0h.y), m10 = float(c1h.x), m11 = float(c1h.y);
    for (uint k = 0u; k < 32u; ++k) {
      const float a0f = float(nr_e4m3(arow0[k])), a1f = float(nr_e4m3(arow1[k])), b0f = float(nr_e4m3(bcol0[k])), b1f = float(nr_e4m3(bcol1[k]));
      m00 = fma(a0f, b0f, m00); m01 = fma(a0f, b1f, m01); m10 = fma(a1f, b0f, m10); m11 = fma(a1f, b1f, m11);
    }
    d0 = as_type<uint>(half2(half(m00), half(m01))); d1 = as_type<uint>(half2(half(m10), half(m11)));
    return;
  }
#endif
#if NR_QMMA_FSCALE
  d0 = as_type<uint>(half2(nr_qmma_element_fscale(arow0, bcol0, c0h.x), nr_qmma_element_fscale(arow0, bcol1, c0h.y)));
  d1 = as_type<uint>(half2(nr_qmma_element_fscale(arow1, bcol0, c1h.x), nr_qmma_element_fscale(arow1, bcol1, c1h.y)));
  return;
#endif
#if NR_QMMA_MEASURE_FP32
  // MEASUREMENT ONLY, NOT EXACT: fp32 fma chain instead of the RTX rule, to bound the rule's share of kernel time.
  float m00 = float(c0h.x), m01 = float(c0h.y), m10 = float(c1h.x), m11 = float(c1h.y);
  for (uint k = 0u; k < 32u; ++k) { const float a0f = float(nr_e4m3(arow0[k])), a1f = float(nr_e4m3(arow1[k])), b0f = float(nr_e4m3(bcol0[k])), b1f = float(nr_e4m3(bcol1[k]));
    m00 = fma(a0f, b0f, m00); m01 = fma(a0f, b1f, m01); m10 = fma(a1f, b0f, m10); m11 = fma(a1f, b1f, m11); }
  d0 = as_type<uint>(half2(half(m00), half(m01))); d1 = as_type<uint>(half2(half(m10), half(m11))); return;
#endif
#if NR_QMMA_FAST
  NrE4M3Dec da0[32], da1[32], db0[32], db1[32];
  for (uint k = 0u; k < 32u; ++k) { da0[k] = nr_e4m3_decode(arow0[k]); da1[k] = nr_e4m3_decode(arow1[k]); db0[k] = nr_e4m3_decode(bcol0[k]); db1[k] = nr_e4m3_decode(bcol1[k]); }
  d0 = as_type<uint>(half2(nr_qmma_element_fast(da0, arow0, db0, bcol0, c0h.x), nr_qmma_element_fast(da0, arow0, db1, bcol1, c0h.y)));
  d1 = as_type<uint>(half2(nr_qmma_element_fast(da1, arow1, db0, bcol0, c1h.x), nr_qmma_element_fast(da1, arow1, db1, bcol1, c1h.y)));
#else
  d0 = as_type<uint>(half2(nr_qmma_element(arow0, bcol0, c0h.x), nr_qmma_element(arow0, bcol1, c0h.y)));
  d1 = as_type<uint>(half2(nr_qmma_element(arow1, bcol0, c1h.x), nr_qmma_element(arow1, bcol1, c1h.y)));
#endif
}
#ifndef NR_QMMA_TP
#define NR_QMMA_TP 1
#endif
// Three-product exact QMMA instruction (candidate 14): the warp's m16n8k32 e4m3 fragments are staged as l/h/(l+h) halves (x = 2^-9 (l + 512 h),
// balanced l in [-256, 256), |h| <= 448, |l+h| <= 448), multiplied with 24 simdgroup half->float MMAs (P0 = LA LB, P1 = HA HB, P2 = (LA+HA)(LB+HB);
// every partial sum <= 32*448^2 < 2^23 so the fp32 accumulation is exact), reconstructed exactly in int64, then rounded by the RTX rule:
// max_exp = max operand-exponent sum (and C exponent), G = 2^(max_exp-25); the truncated sum lies within +-33 G of the exact sum; if RNE_half agrees
// at both ends the element is certified, otherwise it is repaired with nr_qmma_element_fscale (the accepted rule) from the staged bytes. Results are
// exchanged through the scratch back into the CUDA D-fragment layout. Requires one warp per threadgroup (as all transliterated kernels use).
struct NrTpScratch {
  half al[512], ah[512];                 // A 16 x 32 (row stride 32): l and h parts
  half bl[256], bh[256];                 // B 32 x 8 (row stride 8)
  uchar ab[512], bb[256];                // raw e4m3 bytes (classes and repair are derived from these)
  half c[128], d[128];                   // C and D, row-major 16 x 8
  atomic_uint repair_mask[4];            // bit per element (128)
  atomic_uint maxA[16], maxB[8];         // row / column max exponent class + 1 (0 = all zero): sound upper bound on the operand-exponent sum
  bool enabled;                          // set in the kernel prologue: this warp owns a scratch
};                                       // ~4.5 KB per warp
// One-pack certificate (validated on CPU: scripts/nr-qmma-cheapcert-check.py). S in units 2^-24, G the rule's grid in the same units (0 = every
// term aligned). Rounding to half is monotone, so RNE_half is constant on [S-33G, S+33G] iff no half midpoint lies inside: in-binade midpoints
// sit at r = ulp/2 (r = |S| mod ulp), the lower binade's last midpoint sits ulp/4 below the binade edge; the upper binade's first midpoint is
// already excluded by the in-binade test. Subnormal half region (|S| < 2^10 units): certified only when G = 0.
inline bool nr_tp_certified(long S, long G) {
  if (G == 0l) return true;
  const ulong mag = ulong(S < 0l ? -S : S); if (mag == 0ul) return false;
  const uint hi = uint(mag >> 32u); const uint bl = hi != 0u ? 64u - clz(hi) : 32u - clz(uint(mag));     // bit length of |S|
  if (bl <= 10u) return false;
  const long ulp = 1l << (bl - 11u), half_ulp = ulp >> 1u; const long r = long(mag & ulong(ulp - 1l)); const long d = r - half_ulp;
  const long edge_dist = long(mag) - (1l << (bl - 1u)); const long g33 = 33l * G;
  return (g33 < (d < 0l ? -d : d)) && (g33 < edge_dist + (ulp >> 2u));
}
inline int nr_tp_int9(uint b) { const int e = int((b >> 3u) & 15u), m = int(b & 7u); const int sig = (e == 0) ? m : 8 + m; const int q = (e == 0) ? -9 : e - 10; const int v = sig << uint(q + 9); return (b & 0x80u) ? -v : v; }
inline int nr_tp_class(uint b) { const int e = int((b >> 3u) & 15u), m = int(b & 7u); if (e == 0) return m == 0 ? 0 : (31 - int(clz(uint(m)))) + 1; return e + 3; }   // 0 = zero, else floor(log2|x|) + 10
inline void nr_tp_stage_a(threadgroup NrTpScratch& s, uint row, uint k0, uint word) {
  uint mx = 0u;
  for (uint j = 0u; j < 4u; ++j) { const uint b = (word >> (8u * j)) & 0xffu; const int m = nr_tp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h; const uint i = row * 32u + k0 + j; s.al[i] = half(l); s.ah[i] = half(h); s.ab[i] = uchar(b); mx = max(mx, uint(nr_tp_class(b))); }
  if (mx != 0u) atomic_fetch_max_explicit(&s.maxA[row], mx, memory_order_relaxed);
}
inline void nr_tp_stage_b(threadgroup NrTpScratch& s, uint k0, uint col, uint word) {
  uint mx = 0u;
  for (uint j = 0u; j < 4u; ++j) { const uint b = (word >> (8u * j)) & 0xffu; const int m = nr_tp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h; const uint i = (k0 + j) * 8u + col; s.bl[i] = half(l); s.bh[i] = half(h); s.bb[i] = uchar(b); mx = max(mx, uint(nr_tp_class(b))); }
  if (mx != 0u) atomic_fetch_max_explicit(&s.maxB[col], mx, memory_order_relaxed);
}
// Register-only three-product QMMA instruction (candidate 18) for multi-warp kernels (no threadgroup scratch): the warp's fragments are gathered
// into simdgroup matrices by shuffles (per-lane source lane, uniform register), reconstructed and certified per element exactly as the scratch
// path, results returned to the CUDA D layout by shuffles; repair gathers the needed row/column bytes by shuffles (warp-uniform control flow).
// Measured M3 Max simdgroup 8x8 element mapping: lane L holds (row (L%8)/2 + 4(L/16), col 2(L%2) + 4((L/8)%2) + e), e = 0,1.
inline uint nr_tp_src_lane_a(uint row, uint k) { return 4u * (row % 8u) + (k % 16u) / 4u; }     // CUDA A fragment: lane (g=row%8, t=(k%16)/4)
inline uint nr_tp_src_lane_b(uint n, uint k) { return 4u * n + (k % 16u) / 4u; }               // CUDA B fragment: lane (g=n, t=(k%16)/4)
inline void nr_mma_e4m3_reg(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, uint c0, uint c1, thread uint& d0, thread uint& d1, uint lane) {
  const uint g = lane / 4u, t = lane % 4u; const uint mrow = (lane % 8u) / 2u + 4u * (lane / 16u), mcol0 = 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u);
  // row / column max classes of the CUDA-held bytes, reduced over the quad that holds a row (lanes 4g..4g+3) resp. a column (same quad, column g)
  uint mr0 = 0u, mr1 = 0u, mc = 0u;
  for (uint j = 0u; j < 4u; ++j) { mr0 = max(mr0, max(uint(nr_tp_class((a0 >> (8u * j)) & 0xffu)), uint(nr_tp_class((a2 >> (8u * j)) & 0xffu)))); mr1 = max(mr1, max(uint(nr_tp_class((a1 >> (8u * j)) & 0xffu)), uint(nr_tp_class((a3 >> (8u * j)) & 0xffu)))); mc = max(mc, max(uint(nr_tp_class((b0 >> (8u * j)) & 0xffu)), uint(nr_tp_class((b1 >> (8u * j)) & 0xffu)))); }
  mr0 = max(mr0, simd_shuffle_xor(mr0, 1u)); mr0 = max(mr0, simd_shuffle_xor(mr0, 2u)); mr1 = max(mr1, simd_shuffle_xor(mr1, 1u)); mr1 = max(mr1, simd_shuffle_xor(mr1, 2u)); mc = max(mc, simd_shuffle_xor(mc, 1u)); mc = max(mc, simd_shuffle_xor(mc, 2u));
  // this lane's MMA-layout elements need the row max of rows mrow / mrow+8 (held by quad 4*(row%8)) and the column max of columns mcol0, mcol0+1 (quad 4*col)
  const uint maxA0 = simd_shuffle(mr0, ushort(4u * (mrow % 8u))), maxA1 = simd_shuffle(mr1, ushort(4u * (mrow % 8u)));
  const uint maxB0 = simd_shuffle(mc, ushort(4u * mcol0)), maxB1 = simd_shuffle(mc, ushort(4u * (mcol0 + 1u)));
  // C at this lane's MMA positions: C(row, col) is held by lane (row%8, col/2) in c0 (row<8) / c1 (row>=8), half col%2
  const uint cw00 = simd_shuffle(c0, ushort(4u * (mrow % 8u) + mcol0 / 2u)), cw10 = simd_shuffle(c1, ushort(4u * (mrow % 8u) + mcol0 / 2u));
  const half2 ch0 = as_type<half2>(cw00), ch1 = as_type<half2>(cw10);   // .x -> col mcol0 (even), .y -> col mcol0+1
  half res[4]; uint rmask = 0u;
  for (uint rb = 0u; rb < 2u; ++rb) {
    simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), P1 = P0, P2 = P0;
    for (uint kk = 0u; kk < 4u; ++kk) {
      simdgroup_half8x8 AL, AH, ALH, BL, BH, BLH;
      // A block (rows rb*8.., k kk*8..): element (mrow + 8rb, kk*8 + mcol0 + e); register uniform: a0/a1 for k<16, a2/a3 for k>=16; rows >=8 use a1/a3
      const uint areg = (kk < 2u) ? (rb == 0u ? a0 : a1) : (rb == 0u ? a2 : a3);
      for (uint e = 0u; e < 2u; ++e) { const uint k = kk * 8u + mcol0 + e; const uint w = simd_shuffle(areg, ushort(nr_tp_src_lane_a(mrow, k))); const uint b = (w >> (8u * (k % 4u))) & 0xffu;
        const int m = nr_tp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h; AL.thread_elements()[e] = half(l); AH.thread_elements()[e] = half(h); ALH.thread_elements()[e] = half(l + h); }
      // B block (k kk*8.., cols): element (k = kk*8 + mrow, n = mcol0 + e); register uniform: b0 for k<16, b1 for k>=16
      const uint breg = (kk < 2u) ? b0 : b1;
      for (uint e = 0u; e < 2u; ++e) { const uint k = kk * 8u + mrow, n = mcol0 + e; const uint w = simd_shuffle(breg, ushort(nr_tp_src_lane_b(n, k))); const uint b = (w >> (8u * (k % 4u))) & 0xffu;
        const int m = nr_tp_int9(b); const int h = (m + 256) >> 9; const int l = m - 512 * h; BL.thread_elements()[e] = half(l); BH.thread_elements()[e] = half(h); BLH.thread_elements()[e] = half(l + h); }
      simdgroup_multiply_accumulate(P0, AL, BL, P0); simdgroup_multiply_accumulate(P1, AH, BH, P1); simdgroup_multiply_accumulate(P2, ALH, BLH, P2);
    }
    for (uint e = 0u; e < 2u; ++e) {
      const uint ei = rb * 2u + e; const int p0 = int(P0.thread_elements()[e]), p1 = int(P1.thread_elements()[e]), cross = int(P2.thread_elements()[e]) - p0 - p1;
      const long recon = long(p0) + (long(cross) << 9u) + (long(p1) << 18u);
      const half ch = (rb == 0u) ? (e == 0u ? ch0.x : ch0.y) : (e == 0u ? ch1.x : ch1.y); const ushort cb = as_type<ushort>(ch); const int ce = int((cb >> 10u) & 31u), cf = int(cb & 1023u);
      int csig, cq; if (ce != 0) { csig = 1024 + cf; cq = ce - 25; } else { csig = cf; cq = -24; }
      long cval = 0l; int cexp = -1000; if (csig != 0) { cexp = cq + (31 - int(clz(uint(csig)))); cval = long(csig) << uint(cq + 24); if (cb & 0x8000u) cval = -cval; }
      const long S = (recon << 6u) + cval; const uint ma1 = rb == 0u ? maxA0 : maxA1, mb1 = e == 0u ? maxB0 : maxB1;
      int maxexp = (ma1 != 0u && mb1 != 0u) ? int(ma1 + mb1) - 20 : -1000; maxexp = max(maxexp, cexp);
      if (maxexp == -1000) res[ei] = as_type<half>(ushort(0u));
      else { const int gexp = maxexp - 25; const long G = (gexp + 24 >= 0) ? (1l << uint(gexp + 24)) : 0l; if (nr_tp_certified(S, G)) res[ei] = nr_fixed_to_half_rne(S, -24); else rmask |= 1u << ei; }
    }
  }
  // repair (warp-uniform control flow): for each element slot any lane needs, gather that lane's row of A and column of B by shuffles and apply the accepted rule
  for (uint ei = 0u; ei < 4u; ++ei) {
    if (!simd_any((rmask >> ei) & 1u)) continue;
    const uint rb = ei / 2u, e = ei % 2u; const uint row = mrow + 8u * rb, col = mcol0 + e; uchar arow[32], bcol[32];
    const uint ra = rb == 0u ? a0 : a1, rc2 = rb == 0u ? a2 : a3;
    for (uint q = 0u; q < 4u; ++q) { const ushort src = ushort(4u * (row % 8u) + q); const uint wa = simd_shuffle(ra, src), wc = simd_shuffle(rc2, src); const ushort srcb = ushort(4u * col + q); const uint wb0 = simd_shuffle(b0, srcb), wb1 = simd_shuffle(b1, srcb);
      for (uint j = 0u; j < 4u; ++j) { arow[4u * q + j] = uchar((wa >> (8u * j)) & 0xffu); arow[16u + 4u * q + j] = uchar((wc >> (8u * j)) & 0xffu); bcol[4u * q + j] = uchar((wb0 >> (8u * j)) & 0xffu); bcol[16u + 4u * q + j] = uchar((wb1 >> (8u * j)) & 0xffu); } }
    if ((rmask >> ei) & 1u) res[ei] = nr_qmma_element_fscale(arow, bcol, (rb == 0u) ? (e == 0u ? ch0.x : ch0.y) : (e == 0u ? ch1.x : ch1.y));
  }
  // back to the CUDA D layout: this lane needs (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1); (row, col) lives at lane 16(row/4 % 2) + 8(col/4) + 2(row%4) + (col/2)%2, element rb = row/8, e = col%2
  const uint lo0 = 16u * ((g / 4u) % 2u) + 8u * ((2u * t) / 4u) + 2u * (g % 4u) + ((2u * t) / 2u) % 2u, lo1 = 16u * ((g / 4u) % 2u) + 8u * ((2u * t + 1u) / 4u) + 2u * (g % 4u) + ((2u * t + 1u) / 2u) % 2u;
  const half v00 = as_type<half>(simd_shuffle(as_type<ushort>(res[0]), ushort(lo0))), v01 = as_type<half>(simd_shuffle(as_type<ushort>(res[1]), ushort(lo1)));
  const half v10 = as_type<half>(simd_shuffle(as_type<ushort>(res[2]), ushort(lo0))), v11 = as_type<half>(simd_shuffle(as_type<ushort>(res[3]), ushort(lo1)));
  d0 = as_type<uint>(half2(v00, v01)); d1 = as_type<uint>(half2(v10, v11));
}
#ifndef NR_QMMA_TP_REG
#define NR_QMMA_TP_REG 0   // multi-warp kernels (no scratch): 1 = register-only three-product path, 0 = accepted scalar rule
#endif
#ifndef NR_QMMA_TP_REG_ALL
#define NR_QMMA_TP_REG_ALL 0   // 1 = every kernel (single-warp scratch kernels included) uses the register-only three-product path; candidate for launch 155 (attempt-v19)
#endif
inline void nr_mma_e4m3(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, uint c0, uint c1, thread uint& d0, thread uint& d1, uint lane, threadgroup NrTpScratch& s) {
#if !NR_QMMA_TP
  nr_mma_e4m3(a0, a1, a2, a3, b0, b1, c0, c1, d0, d1, lane); (void)s; return;
#else
#if NR_QMMA_TP_REG_ALL
  nr_mma_e4m3_reg(a0, a1, a2, a3, b0, b1, c0, c1, d0, d1, lane); (void)s; return;
#endif
  if (!s.enabled) {
#if NR_QMMA_TP_REG
    nr_mma_e4m3_reg(a0, a1, a2, a3, b0, b1, c0, c1, d0, d1, lane);
#else
    nr_mma_e4m3(a0, a1, a2, a3, b0, b1, c0, c1, d0, d1, lane);
#endif
    return; }
  const uint g = lane / 4u, t = lane % 4u;
  if (lane < 4u) atomic_store_explicit(&s.repair_mask[lane], 0u, memory_order_relaxed);
  if (lane < 16u) atomic_store_explicit(&s.maxA[lane], 0u, memory_order_relaxed); if (lane < 8u) atomic_store_explicit(&s.maxB[lane], 0u, memory_order_relaxed);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  nr_tp_stage_a(s, g, 4u * t, a0); nr_tp_stage_a(s, g + 8u, 4u * t, a1); nr_tp_stage_a(s, g, 16u + 4u * t, a2); nr_tp_stage_a(s, g + 8u, 16u + 4u * t, a3);
  nr_tp_stage_b(s, 4u * t, g, b0); nr_tp_stage_b(s, 16u + 4u * t, g, b1);
  { const half2 ch0 = as_type<half2>(c0), ch1 = as_type<half2>(c1); s.c[g * 8u + 2u * t] = ch0.x; s.c[g * 8u + 2u * t + 1u] = ch0.y; s.c[(g + 8u) * 8u + 2u * t] = ch1.x; s.c[(g + 8u) * 8u + 2u * t + 1u] = ch1.y; }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 A, B;
  for (uint rb = 0u; rb < 2u; ++rb) {
    // four exact products with only l and h staged: LL, HH, LH, HL (each partial sum <= 32*448^2 < 2^23)
    simdgroup_float8x8 PLL = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), PHH = PLL, PLH = PLL, PHL = PLL;
    for (uint kk = 0u; kk < 4u; ++kk) {
      const uint ao = rb * 256u + kk * 8u, bo = kk * 64u; simdgroup_half8x8 AL, AH, BL, BH;
      simdgroup_load(AL, s.al + ao, 32u); simdgroup_load(AH, s.ah + ao, 32u); simdgroup_load(BL, s.bl + bo, 8u); simdgroup_load(BH, s.bh + bo, 8u);
      simdgroup_multiply_accumulate(PLL, AL, BL, PLL); simdgroup_multiply_accumulate(PHH, AH, BH, PHH); simdgroup_multiply_accumulate(PLH, AL, BH, PLH); simdgroup_multiply_accumulate(PHL, AH, BL, PHL);
    }
    for (uint e = 0u; e < 2u; ++e) {
      const uint row = rb * 8u + (lane % 8u) / 2u + 4u * (lane / 16u), col = 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u) + e;   // measured M3 Max element mapping
      const long pll = long(PLL.thread_elements()[e]), phh = long(PHH.thread_elements()[e]), plh = long(PLH.thread_elements()[e]), phl = long(PHL.thread_elements()[e]);
      const long recon = pll + 512l * (plh + phl) + 262144l * phh;                       // sum(a*b) * 2^18, exact in int64
      const ushort cb = as_type<ushort>(s.c[row * 8u + col]); const int ce = int((cb >> 10u) & 31u), cf = int(cb & 1023u);
      int csig, cq; if (ce != 0) { csig = 1024 + cf; cq = ce - 25; } else { csig = cf; cq = -24; }
      long cval = 0l; int cexp = -1000; if (csig != 0) { cexp = cq + (31 - int(clz(uint(csig)))); cval = long(csig) << uint(cq + 24); if (cb & 0x8000u) cval = -cval; }
      const long S = (recon << 6u) + cval;                                                // units 2^-24; |S| < 2^47
      // sound upper bound on the max operand-exponent sum (row max class + column max class; classes are floor(log2)+10); a larger bound only
      // widens the certificate interval (CPU-validated: same coverage on model data)
      const uint ma1 = atomic_load_explicit(&s.maxA[row], memory_order_relaxed), mb1 = atomic_load_explicit(&s.maxB[col], memory_order_relaxed);
      int maxexp = (ma1 != 0u && mb1 != 0u) ? int(ma1 + mb1) - 20 : -1000; maxexp = max(maxexp, cexp);
      half result; bool certified = true;
      if (maxexp == -1000) result = as_type<half>(ushort(0u));                            // no nonzero term: canonical +0 (the rule's zero)
      else { const int gexp = maxexp - 25;
        // grid finer than 2^-24 (gexp + 24 < 0): every term is a multiple of 2^-24 (products of 2^-18 units, C of 2^-24 units) hence aligned: remainder 0
        const long G = (gexp + 24 >= 0) ? (1l << uint(gexp + 24)) : 0l;
        if (nr_tp_certified(S, G)) result = nr_fixed_to_half_rne(S, -24); else certified = false; }
      const uint idx = row * 8u + col;
      if (certified) s.d[idx] = result; else atomic_fetch_or_explicit(&s.repair_mask[idx / 32u], 1u << (idx % 32u), memory_order_relaxed);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  { // compacted repair: lane i takes the i-th set bit (works for 0..128 uncertified elements)
    const uint m0 = atomic_load_explicit(&s.repair_mask[0], memory_order_relaxed), m1 = atomic_load_explicit(&s.repair_mask[1], memory_order_relaxed), m2 = atomic_load_explicit(&s.repair_mask[2], memory_order_relaxed), m3 = atomic_load_explicit(&s.repair_mask[3], memory_order_relaxed);
    const uint total = popcount(m0) + popcount(m1) + popcount(m2) + popcount(m3);
    for (uint i = lane; i < total; i += 32u) {
      uint rank = i, w = 0u, mw = m0;
      if (rank >= popcount(m0)) { rank -= popcount(m0); w = 1u; mw = m1; if (rank >= popcount(m1)) { rank -= popcount(m1); w = 2u; mw = m2; if (rank >= popcount(m2)) { rank -= popcount(m2); w = 3u; mw = m3; } } }
      uint bit = 0u; for (uint r = 0u; ; ++bit) { if ((mw >> bit) & 1u) { if (r == rank) break; ++r; } }
      const uint idx = w * 32u + bit; const uint row = idx / 8u, col = idx % 8u; uchar arow[32], bcol[32];
      for (uint k = 0u; k < 32u; ++k) { arow[k] = s.ab[row * 32u + k]; bcol[k] = s.bb[k * 8u + col]; }
      s.d[idx] = nr_qmma_element_fscale(arow, bcol, s.c[idx]);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  d0 = as_type<uint>(half2(s.d[g * 8u + 2u * t], s.d[g * 8u + 2u * t + 1u])); d1 = as_type<uint>(half2(s.d[(g + 8u) * 8u + 2u * t], s.d[(g + 8u) * 8u + 2u * t + 1u]));
#endif
}
// ---- global address resolution: fake bases as in the emulator ----
constant ulong NR_POOL_BASE = 0x100000000ul;
constant ulong NR_WBASE = 0x200000000ul;
inline device uchar* nr_gptr(device uchar* pool, device uchar* wbuf, ulong a) { return (a >= NR_WBASE) ? (wbuf + (a - NR_WBASE)) : (pool + (a - NR_POOL_BASE)); }
// write coverage: one counter per 4-byte word of the pool (writes to the weight range are not counted)
inline void nr_cover(device atomic_uint* cov, ulong a, uint bytes) {
  if (a >= NR_WBASE) return;
  const ulong w0 = (a - NR_POOL_BASE) / 4ul, w1 = (a - NR_POOL_BASE + bytes - 1ul) / 4ul;
  for (ulong w = w0; w <= w1; ++w) atomic_fetch_add_explicit(&cov[w], 1u, memory_order_relaxed);
}
inline ulong nr_ld_param64(constant uchar* prm, uint off) { ulong v = 0ul; for (uint i = 0u; i < 8u; ++i) v |= ulong(prm[off + i]) << (8u * i); return v; }
inline uint nr_ld_param32(constant uchar* prm, uint off) { uint v = 0u; for (uint i = 0u; i < 4u; ++i) v |= uint(prm[off + i]) << (8u * i); return v; }
inline uint nr_active_lanes() { return popcount((uint)(simd_vote::vote_t)simd_active_threads_mask()); }
// ---- texture sampling / surface store (post block). tex.2d.v4.f32.f32 with normalized float coords; the sampler mode is
// a hypothesis validated only against the recorded output image (NR_TEX_FILTER: 0 nearest, 1 linear).
#ifndef NR_QMMA_C_FIRST
#define NR_QMMA_C_FIRST 0
#endif
#ifndef NR_EX2_POS_UNIT
#define NR_EX2_POS_UNIT 1
#endif
#ifndef NR_SUST_RNE
#define NR_SUST_RNE 0
#endif
#ifndef NR_TEX_FILTER
#define NR_TEX_FILTER 1
#endif
// EXPERIMENTAL (temporal builds only): NR_TEX_IN_MODE 3 routes the tex_in reads through the fitted RTX rule too (exact texels at centres, sign of zero kept);
// the accepted production kernels keep the Metal hardware sampler (NR_TEX_IN_MODE 0).
#ifndef NR_TEX_IN_MODE
#define NR_TEX_IN_MODE 0
#endif
#ifndef NR_TEX_MODE
#define NR_TEX_MODE 0
#endif
#if NR_TEX_MODE == 3 || NR_TEX_IN_MODE == 3
#include "nr_tex_sample.metal"
#endif
inline float4 nr_tex2d(texture2d<float, access::sample> t, ulong texref, float x, float y) {
  (void)texref;
#if NR_TEX_IN_MODE == 3
  return nr_tex2d_bilinear_rtx(t, x, y);
#else
#if NR_TEX_FILTER
  constexpr sampler smp(coord::normalized, filter::linear, address::clamp_to_edge);
#else
  constexpr sampler smp(coord::normalized, filter::nearest, address::clamp_to_edge);
#endif
  return t.sample(smp, float2(x, y));
#endif
}
// EXPERIMENTAL temporal taps (history/motion, emitted only by --texref-map builds): NR_TEX_MODE 0 = Metal hardware linear sampler (recorded modes:
// normalized, clamp-to-edge, linear); 1 = software bilinear with the fraction truncated to 1/256; 2 = rounded to 1/256 (src/nr_temporal/nr_tex_sample.metal);
// 3 = the RTX rule fitted to Codex's micro-probe (nr_tex2d_bilinear_rtx in src/nr_temporal/nr_tex_sample.metal; compile with -I src/nr_temporal).
#ifndef NR_TEX_MODE
#define NR_TEX_MODE 0
#endif
inline float4 nr_tex2d_mapped(texture2d<float, access::sample> t, ulong texref, float x, float y) {
  (void)texref;
#if NR_TEX_MODE == 0
  constexpr sampler smp(coord::normalized, filter::linear, address::clamp_to_edge); return t.sample(smp, float2(x, y));
#elif NR_TEX_MODE == 3
  return nr_tex2d_bilinear_rtx(t, x, y);
#else
  const float tx = x * float(t.get_width()) - 0.5f, ty = y * float(t.get_height()) - 0.5f; const float fx0 = floor(tx), fy0 = floor(ty); float fx = tx - fx0, fy = ty - fy0;
#if NR_TEX_MODE == 2
  fx = rint(fx * 256.0f) / 256.0f; fy = rint(fy * 256.0f) / 256.0f;
#else
  fx = floor(fx * 256.0f) / 256.0f; fy = floor(fy * 256.0f) / 256.0f;
#endif
  const int Wd = int(t.get_width()), Hd = int(t.get_height()); const int x0 = clamp(int(fx0), 0, Wd - 1), x1 = clamp(int(fx0) + 1, 0, Wd - 1), y0 = clamp(int(fy0), 0, Hd - 1), y1 = clamp(int(fy0) + 1, 0, Hd - 1);
  const float4 a = t.read(uint2(x0, y0)), b = t.read(uint2(x1, y0)), d = t.read(uint2(x0, y1)), e = t.read(uint2(x1, y1));
  const float4 top = a + (b - a) * fx, bot = d + (e - d) * fx; return top + (bot - top) * fy;
#endif
}
// EXPERIMENTAL (temporal builds, --ex2-half-table): ex2.approx.ftz.f32 of a half promoted to fp32 and scaled by 0xbfb8aa3b takes Codex's audited
// half-domain table (65536 x {exact scaled argument bits, RTX result bits}, indexed by the ORIGINAL half bits; sha pinned by the runtime). The lookup
// verifies the exact fp32 argument; a mismatch or a nonfinite half counts a fault (fault[0] argument, fault[1] domain) and yields NaN, so the frame fails
// closed in the runtime's fault check and in any reference comparison. No approximate inversion, no per-value special case.
inline float nr_ex2_half_table(device const uint2* tbl, device atomic_uint* fault, ushort h, float x) {
  if ((h & 0x7c00u) == 0x7c00u) { atomic_fetch_add_explicit(fault + 1, 1u, memory_order_relaxed); return as_type<float>(0x7fc00000u); }
  const uint2 e = tbl[h];
  if (e.x != as_type<uint>(x)) { atomic_fetch_add_explicit(fault, 1u, memory_order_relaxed); return as_type<float>(0x7fc00000u); }
  return as_type<float>(e.y);
}
// Formatted surface store of f32 data to a half-float surface truncates (round toward zero): the FG module24 RG16F store
// was exact only with RZ (Mac-fg-research src/fg_resolution_preprocess_l22_l24). Ported bit-exact conversion.
inline ushort nr_f32_to_f16_rz_bits(float value) {
  const uint bits = as_type<uint>(value); const ushort sign = ushort((bits >> 16u) & 0x8000u); const uint exponent = (bits >> 23u) & 0xffu; const uint mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
  const int half_exponent = int(exponent) - 127 + 15;
  if (half_exponent >= 31) return ushort(sign | 0x7bffu);                    // RZ: overflow truncates to the largest finite half
  if (half_exponent <= 0) { if (half_exponent < -10) return sign; const uint significand = mantissa | 0x800000u; return ushort(sign | ushort(significand >> uint(14 - half_exponent))); }
  return ushort(sign | ushort(uint(half_exponent) << 10u) | ushort(mantissa >> 13u));
}
inline void nr_sust2d(texture2d<half, access::write> t, device atomic_uint* texcov, ulong surfref, uint x, uint y, float4 v) {
  (void)surfref;
  if (x < t.get_width() && y < t.get_height()) {
    atomic_fetch_add_explicit(&texcov[y * t.get_width() + x], 1u, memory_order_relaxed);      // per-pixel write count (exactly-once evidence)
#if NR_SUST_RNE
    t.write(half4(v), uint2(x, y));
#else
    t.write(half4(as_type<half>(nr_f32_to_f16_rz_bits(v.x)), as_type<half>(nr_f32_to_f16_rz_bits(v.y)), as_type<half>(nr_f32_to_f16_rz_bits(v.z)), as_type<half>(nr_f32_to_f16_rz_bits(v.w))), uint2(x, y));
#endif
  }
}
// ---- sm_120 MUFU models for general f32 inputs (post block), from the FG evidence (Mac-fg-research):
// rcp.approx.ftz.f32 = correctly rounded reciprocal corrected by a 2-bit ROM indexed by the input mantissa (2 MiB, SHA 7cf4f8ce...);
// div.approx.ftz.f32 = a * rcp(b) (960/960 on the FG l85 domain); ex2.approx.ftz.f32 = Metal fast::exp2 corrected by 2-bit streams
// (unit table: -i*2^-24, i in [0, 2^24]; fraction table: -2 + i*2^-23, i in [0, 2^23]; exact for finite x <= 0 in FG).
// For x > 0 the same reduction is applied (2^x = 2^(n+2) * ex2(x - n - 2), unit table for 0 < x < 1): a HYPOTHESIS tested only
// against the recorded output image.
inline float nr_rne_rcp_f32(float value) {
  const float candidate = 1.0f / value; const uint bits = as_type<uint>(candidate);
  const float below = as_type<float>(bits - 1u), above = as_type<float>(bits + 1u);
  const float cr = fabs(fma(-value, candidate, 1.0f)), br = fabs(fma(-value, below, 1.0f)), ar = fabs(fma(-value, above, 1.0f));
  float best = candidate, best_r = cr;
  if (br < best_r || (br == best_r && ((bits - 1u) & 1u) == 0u)) { best = below; best_r = br; }
  if (ar < best_r || (ar == best_r && ((bits + 1u) & 1u) == 0u)) { best = above; }
  return best;
}
#ifndef NR_RCP_EXACT
#define NR_RCP_EXACT 0   // experimental: 1 = correctly rounded 1/x instead of the sm_120 ROM model (temporal diagnostics only)
#endif
#ifndef NR_EX2_EXACT
#define NR_EX2_EXACT 0   // experimental: 1 = precise::exp2 instead of the corrected fast::exp2 model
#endif
inline float nr_rcp_rom(float value, device const uchar* rom) {
#if NR_RCP_EXACT
  (void)rom; return 1.0f / value;
#endif
  const uint input_bits = as_type<uint>(value); uint base_bits = as_type<uint>(nr_rne_rcp_f32(value));
  const uint ie = (input_bits >> 23u) & 0xffu, oe = (base_bits >> 23u) & 0xffu;
  if ((input_bits >> 31u) != 0u || ie == 0u || ie == 0xffu || oe == 0u || oe == 0xffu) return as_type<float>(base_bits);
  const uint mantissa = input_bits & 0x7fffffu; const uint code = (uint(rom[mantissa >> 2u]) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u) base_bits -= 1u; else if (code == 2u) base_bits += 1u;
  return as_type<float>(base_bits);
}
#ifndef NR_RCP_ROM_ONLY
#define NR_RCP_ROM_ONLY 0   // 1 (experimental temporal builds): never take the half-rounded RTX table for rcp.approx.f32 whose result is consumed in fp32 (post block r37/r38 etc.)
#endif
inline float nr_rcp_approx2(device const ushort* table, device const uchar* rom, float x) {
#ifndef NR_RCP_DIM_ULP_BIAS
#define NR_RCP_DIM_ULP_BIAS 0   // diagnostic: add this many fp32 ulps to the result of nr_rcp_approx2 (the dimension reciprocals r37/r38 of the post block)
#endif
#if NR_RCP_ROM_ONLY
#ifndef NR_RCP_H_ULP_BIAS
#define NR_RCP_H_ULP_BIAS 0   // diagnostic: bias applied only when x == 1080 (the height reciprocal r38)
#endif
  (void)table; { const float r = nr_rcp_rom(x, rom); const int bias = NR_RCP_DIM_ULP_BIAS + ((x == 1080.0f) ? NR_RCP_H_ULP_BIAS : 0); return as_type<float>(uint(int(as_type<uint>(r)) + bias)); }
#endif
  half hb = half(x);
  if (float(hb) == x && x > 0.0f && as_type<ushort>(hb) <= 0x7bffu) return float(as_type<half>(table[uint(as_type<ushort>(hb)) - 1u]));   // RTX-captured half domain (verified in the Swin kernels)
  return nr_rcp_rom(x, rom);
}
#ifndef NR_DIV_MODE
#define NR_DIV_MODE 0   // 0 = a * rcp_rom(b) (accepted FG model); diagnostics: 1 = IEEE a/b; 2 = one Newton-refined quotient q + fma(fma(-b,q,a), rcp, 0)
#endif
inline float nr_div_approx(float a, float b, device const uchar* rom) {
#if NR_DIV_MODE == 1
  (void)rom; return a / b;
#elif NR_DIV_MODE == 2
  const float r = nr_rcp_rom(b, rom); const float q = a * r; const float e = fma(-b, q, a); return fma(e, r, q);
#else
  return a * nr_rcp_rom(b, rom);
#endif
}
inline uint nr_ex2_corrected_bits(float input, device const uchar* packed, uint index) {
  const uint fast_bits = as_type<uint>(fast::exp2(input)); const int delta = int((uint(packed[index >> 2u]) >> ((index & 3u) * 2u)) & 3u) - 1;
  return uint(int(fast_bits) - delta);
}
inline float nr_ex2_sm120(float value, device const uchar* unit, device const uchar* frac) {
#if NR_EX2_EXACT
  (void)unit; (void)frac; return precise::exp2(value);
#endif
  const uint bits = as_type<uint>(value); const uint exponent = (bits >> 23u) & 0xffu;
  if (value == 0.0f || exponent <= 103u) return 1.0f;
  if (value < 0.0f) {
    if (exponent <= 126u) { const uint index = min(uint((-value) * 16777216.0f), 16777216u); const float input = -float(index) * as_type<float>(uint(0x33800000u)); return as_type<float>(nr_ex2_corrected_bits(input, unit, index)); }
    if (value < -126.0f) return 0.0f;
  } else {
    if (value >= 128.0f) return INFINITY;
#if NR_EX2_POS_UNIT
    if (exponent <= 126u) { const uint index = 16777216u - min(uint(value * 16777216.0f), 16777216u); const float input = -float(index) * as_type<float>(uint(0x33800000u)); return 2.0f * as_type<float>(nr_ex2_corrected_bits(input, unit, index)); }
#endif
  }
  const float integral = floor(value); const int binary_scale = int(integral) + 2; const float normalized = value - float(binary_scale);
  const uint index = min(uint((normalized + 2.0f) * 8388608.0f), 8388608u); const float input = fma(float(index), as_type<float>(uint(0x34000000u)), -2.0f);
  const uint base_bits = nr_ex2_corrected_bits(input, frac, index); const int scaled_bits = int(base_bits) + binary_scale * 0x00800000;
  return as_type<float>(uint(scaled_bits));
}
// ---- mma.sync.m16n8k16.row.col.f16.f16.f16.f16: the HMMA.16816.F16 rule (two guard bits, truncation onto the
// 2^(max_exp-25) grid, exact integer sum, one RNE), ported from Mac-fg-research src/fg_conv/hmma_exact.metalh.
inline ulong nr_round_shift_rne64(ulong value, uint shift) {
  if (shift == 0u) return value; if (shift >= 64u) return 0ul;
  const ulong q = value >> shift, r = value & ((1ul << shift) - 1ul), h = 1ul << (shift - 1u);
  return q + ulong(r > h || (r == h && (q & 1ul) != 0ul));
}
inline half nr_fixed_to_half_rne(long value, int scale) {
  if (value == 0l) return as_type<half>(ushort(0u));
  const bool negative = value < 0l; const ulong magnitude = negative ? ulong(-value) : ulong(value);
  const uint leading = nr_bitlen64_minus1(magnitude);
  int exponent = int(leading) + scale; ulong significand = 0ul; uint result = 0u;
  if (exponent >= -14) {
    const int shift = int(leading) - 10;
    significand = shift > 0 ? nr_round_shift_rne64(magnitude, uint(shift)) : magnitude << uint(-shift);
    if (significand >= 2048ul) { significand >>= 1u; ++exponent; }
    result = (exponent > 15) ? 0x7c00u : ((uint(exponent + 15) << 10u) | (uint(significand) - 1024u));
  } else {
    const int shift = -(scale + 24);
    significand = shift > 0 ? nr_round_shift_rne64(magnitude, uint(shift)) : magnitude << uint(-shift);
    result = significand >= 1024ul ? 0x0400u : uint(significand);
  }
  if (negative) result |= 0x8000u;
  return as_type<half>(ushort(result));
}
inline half nr_hmma16_element(thread const ushort* a_bits, thread const ushort* b_bits, half prior) {
  uint psig[16]; int pexp[16]; bool pneg[16]; int maxexp = -21;
  for (uint i = 0u; i < 16u; ++i) {
    const uint af = (uint(a_bits[i]) >> 10u) & 31u, bf = (uint(b_bits[i]) >> 10u) & 31u;
    uint am = uint(a_bits[i]) & 1023u, bm = uint(b_bits[i]) & 1023u; if (af != 0u) am += 1024u; if (bf != 0u) bm += 1024u;
    psig[i] = am * bm * 8u; pexp[i] = (af == 0u ? -14 : int(af) - 15) + (bf == 0u ? -14 : int(bf) - 15); pneg[i] = ((a_bits[i] ^ b_bits[i]) & 0x8000u) != 0u;
    if (psig[i] != 0u) maxexp = max(maxexp, pexp[i]);
  }
  const ushort cb = as_type<ushort>(prior); const uint cf = (uint(cb) >> 10u) & 31u, cfr = uint(cb) & 1023u; uint csig = 0u; int cexp = -1024;
  if (cf != 0u) { csig = (1024u + cfr) << 13u; cexp = int(cf) - 15; }
  else if (cfr != 0u) { const uint lead = nr_bitlen64_minus1(ulong(cfr)); csig = cfr << (23u - lead); cexp = int(lead) - 24; }
  if (csig != 0u) maxexp = max(maxexp, cexp);
  long total = 0l;
  for (uint i = 0u; i < 16u; ++i) { const int sh = maxexp - pexp[i]; const uint al = sh > 31 ? 0u : (psig[i] << 2u) >> uint(sh); total += pneg[i] ? -long(al) : long(al); }
  if (csig != 0u) { const int sh = maxexp - cexp; const uint al = sh > 31 ? 0u : (csig << 2u) >> uint(sh); total += (cb & 0x8000u) != 0u ? -long(al) : long(al); }
  return nr_fixed_to_half_rne(total, maxexp - 25);
}
// fragments (m16n8k16 f16): a0 = row g, k 2t,2t+1 ; a1 = row g+8, k 2t.. ; a2 = row g, k 8+2t.. ; a3 = row g+8, k 8+2t.. ; b0 = k 2t.., col g ; b1 = k 8+2t.., col g
inline void nr_mma_f16k16(uint a0, uint a1, uint a2, uint a3, uint b0, uint b1, uint c0, uint c1, thread uint& d0, thread uint& d1, uint lane) {
  const uint g = lane / 4u, t = lane % 4u;
  ushort arow0[16], arow1[16], bcol0[16], bcol1[16];
  for (uint q = 0u; q < 4u; ++q) {
    const ushort sa = ushort(4u * g + q);
    const uint ra0 = simd_shuffle(a0, sa), ra1 = simd_shuffle(a1, sa), ra2 = simd_shuffle(a2, sa), ra3 = simd_shuffle(a3, sa);
    const ushort sb0 = ushort(4u * (2u * t) + q), sb1 = ushort(4u * (2u * t + 1u) + q);
    const uint rb00 = simd_shuffle(b0, sb0), rb01 = simd_shuffle(b1, sb0), rb10 = simd_shuffle(b0, sb1), rb11 = simd_shuffle(b1, sb1);
    arow0[2u * q] = ushort(ra0 & 0xffffu); arow0[2u * q + 1u] = ushort(ra0 >> 16u); arow0[8u + 2u * q] = ushort(ra2 & 0xffffu); arow0[8u + 2u * q + 1u] = ushort(ra2 >> 16u);
    arow1[2u * q] = ushort(ra1 & 0xffffu); arow1[2u * q + 1u] = ushort(ra1 >> 16u); arow1[8u + 2u * q] = ushort(ra3 & 0xffffu); arow1[8u + 2u * q + 1u] = ushort(ra3 >> 16u);
    bcol0[2u * q] = ushort(rb00 & 0xffffu); bcol0[2u * q + 1u] = ushort(rb00 >> 16u); bcol0[8u + 2u * q] = ushort(rb01 & 0xffffu); bcol0[8u + 2u * q + 1u] = ushort(rb01 >> 16u);
    bcol1[2u * q] = ushort(rb10 & 0xffffu); bcol1[2u * q + 1u] = ushort(rb10 >> 16u); bcol1[8u + 2u * q] = ushort(rb11 & 0xffffu); bcol1[8u + 2u * q + 1u] = ushort(rb11 >> 16u);
  }
  half2 c0h = as_type<half2>(c0), c1h = as_type<half2>(c1);
  d0 = as_type<uint>(half2(nr_hmma16_element(arow0, bcol0, c0h.x), nr_hmma16_element(arow0, bcol1, c0h.y)));
  d1 = as_type<uint>(half2(nr_hmma16_element(arow1, bcol0, c1h.x), nr_hmma16_element(arow1, bcol1, c1h.y)));
}
