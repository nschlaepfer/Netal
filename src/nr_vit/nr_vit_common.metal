// NR ViT-1D family on Metal: shared numerics and layouts.
//
// Every formula here was pinned against the RTX 5090 FP8 pool oracle through the PTX warp emulator
// (scripts/ptx_warp_emu.py) and the numpy reference (scripts/nr_vit_ref.py):
//   - QMMA rule: E4M3 products are exact; one m16n8k32 step accumulates 32 products onto the f16 C
//     operand as an fp32 FMA chain in k-slot order and rounds once to half.
//   - ptxas contracts `mul.f16x2` feeding the FIRST operand of an `add.f16x2` into an f16 fma.
//   - rsqrt.approx.ftz.f32 / rcp.approx.ftz.f32 on half-valued inputs are taken from RTX-captured
//     tables (one entry per positive finite half).
// Compile with -fno-fast-math -ffp-contract=off. Half arithmetic must stay in half precision.
#pragma once
#include <metal_stdlib>
using namespace metal;
inline half nr_hbits(ushort b) { return as_type<half>(b); }

// ---- E4M3 codec (Codex's proven helpers, src/nr_swin/swin32_mlp.metal) ----
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

// ---- layouts ----
// Weight blob: [k-step of 32][N/64 groups of 2048 bytes][weight_index(kk, n%64)].
inline uint nr_weight_index(uint k, uint n) {
  return (n / 16u) * 512u + (n % 8u) * 64u + ((k % 16u) / 4u) * 16u + ((n % 16u) / 8u) * 8u + k % 4u + (k / 16u) * 4u;
}
// Token buffer (A-fragment identity): 16-token tiles, 512-byte steps of 32 channels, lane record of 16 bytes.
inline uint nr_a_index(uint t, uint c, uint channels) {
  uint lane = 4u * (t % 8u) + ((c % 32u) % 16u) / 4u, reg = ((t % 16u) / 8u) + 2u * ((c % 32u) / 16u);
  return (t / 16u) * (16u * channels) + (c / 32u) * 512u + lane * 16u + reg * 4u + c % 4u;
}
// D-fragment layout (expand output, q, attention output): 64-channel blocks of 1024 bytes,
// [c%64/32][lane = 4*(t%8) + (c%8)/2][(c%32)/16][(t%16)/8][(c%16)/8][c%2].
inline uint nr_d_index(uint t, uint c, uint channels) {
  return (t / 16u) * (16u * channels) + (c / 64u) * 1024u + ((c % 64u) / 32u) * 512u + (4u * (t % 8u) + (c % 8u) / 2u) * 16u +
         ((c % 32u) / 16u) * 8u + ((t % 16u) / 8u) * 4u + ((c % 16u) / 8u) * 2u + (c % 2u);
}
// A k-slot j of a 32-step holds channel nr_slot_channel(j) when a D-layout tensor is read as an A operand.
inline uint nr_slot_channel(uint j) { return 2u * ((j % 16u) / 4u) + (j % 2u) + 8u * ((j % 4u) / 2u) + 16u * (j / 16u); }
// MMA column n of a 1024-wide GEMM lands on token channel nr_cfd(n) (the contract/projection swizzle).
inline uint nr_cfd(uint n) { return (n & 1u) + ((n & 6u) << 1u) + ((n & 8u) >> 2u) + (n & 16u) + (n & ~31u); }
// K buffer record: [(t%16)/8][(d%32)/16][(d%16)/8][d%2] with lane = 4*(t%8) + (d%8)/2, per (16-key tile, head).
inline uint nr_k_index(uint t, uint d, uint h) {
  uint lane = 4u * (t % 8u) + (d % 8u) / 2u;
  return (t / 16u) * 16384u + h * 512u + lane * 16u + ((t % 16u) / 8u) * 8u + ((d % 32u) / 16u) * 4u + ((d % 16u) / 8u) * 2u + (d % 2u);
}
// V buffer: 32-key blocks of 32768 bytes, head 1024, dim/16 512, lane = 4*(d%8) + (t%8)/2, [(d%16)/8][(t%32)/16][(t%16)/8][t%2].
inline uint nr_v_index(uint t, uint d, uint h) {
  uint lane = 4u * (d % 8u) + (t % 8u) / 2u;
  return (t / 32u) * 32768u + h * 1024u + (d / 16u) * 512u + lane * 16u + ((d % 16u) / 8u) * 8u + ((t % 32u) / 16u) * 4u + ((t % 16u) / 8u) * 2u + (t % 2u);
}

// ---- exact primitives ----
// One QMMA step (mma.sync m16n8k32 e4m3 -> f16), the rule that matches every RTX register of the 64-case discriminator and the
// launch 151/152 operands (2026-09-05): max_exp = max over nonzero products of floor(log2|a|) + floor(log2|b|) and floor(log2|C|);
// every term's magnitude is truncated to units 2^(max_exp - 25); exact signed integer sum; one RNE to half. Operands arrive as
// halves holding exact e4m3 values (all nonzero e4m3 values are normal halves). NR_QMMA_FP32_CHAIN=1 restores the old fp32 chain.
inline ulong nr_hw_round_shift_rne64(ulong value, uint shift) {
  if (shift == 0u) return value; if (shift >= 64u) return 0ul;
  const ulong q = value >> shift, r = value & ((1ul << shift) - 1ul), h = 1ul << (shift - 1u);
  return q + ulong(r > h || (r == h && (q & 1ul) != 0ul));
}
#ifndef NR_CLZ_BITLEN
#define NR_CLZ_BITLEN 1
#endif
#if NR_CLZ_BITLEN
inline int nr_vit_bitlen(int v) { return v > 0 ? 32 - int(clz(uint(v))) : 0; }
inline uint nr_vit_bitlen64_minus1(ulong m) { const uint hi = uint(m >> 32u); return hi != 0u ? 63u - clz(hi) : 31u - clz(uint(m)); }
#else
inline int nr_vit_bitlen(int v) { int n = 0; for (int x = v; x > 0; x >>= 1) ++n; return n; }
inline uint nr_vit_bitlen64_minus1(ulong m) { uint leading = 0u; for (ulong scan = m; scan > 1ul; scan >>= 1u) ++leading; return leading; }
#endif
inline half nr_hw_fixed_to_half_rne(long value, int scale) {
  if (value == 0l) return as_type<half>(ushort(0u));
  const bool negative = value < 0l; const ulong magnitude = negative ? ulong(-value) : ulong(value);
  const uint leading = nr_vit_bitlen64_minus1(magnitude);
  int exponent = int(leading) + scale; ulong significand = 0ul; uint result = 0u;
  if (exponent >= -14) {
    const int shift = int(leading) - 10;
    significand = shift > 0 ? nr_hw_round_shift_rne64(magnitude, uint(shift)) : magnitude << uint(-shift);
    if (significand >= 2048ul) { significand >>= 1u; ++exponent; }
    result = (exponent > 15) ? 0x7c00u : ((uint(exponent + 15) << 10u) | (uint(significand) - 1024u));
  } else {
    const int shift = -(scale + 24);
    significand = shift > 0 ? nr_hw_round_shift_rne64(magnitude, uint(shift)) : magnitude << uint(-shift);
    result = significand >= 1024ul ? 0x0400u : uint(significand);
  }
  if (negative) result |= 0x8000u;
  return as_type<half>(ushort(result));
}
#ifndef NR_QMMA_NOARRAY
#define NR_QMMA_NOARRAY 0
#endif
#ifndef NR_QMMA_FSCALE
#define NR_QMMA_FSCALE 1
#endif
inline half nr_qmma_step(thread const half* a, thread const half* b, half c) {
#if NR_QMMA_FP32_CHAIN
  float acc = float(c);
  for (uint j = 0u; j < 32u; ++j) acc = fma(float(a[j]), float(b[j]), acc);
  return half(acc);
#elif NR_QMMA_FSCALE
  // Candidate 11, corrected (see nr_ptx_rt.metal): float-scaled form of the same rule with max_exp from the OPERAND exponent sum.
  // Requires e4m3-valued half operands (every caller converts e4m3 bytes with nr_e4m3): products then have <= 8 significant bits and
  // the u32 magnitude sums stay below 2^32. Not valid for unrestricted half x half.
  float p[32]; int maxexp = -300;
  for (uint j = 0u; j < 32u; ++j) {
    const float fa = float(a[j]), fb = float(b[j]); p[j] = fa * fb;
    const int ex = int((as_type<uint>(fa) >> 23u) & 255u) + int((as_type<uint>(fb) >> 23u) & 255u) - 254;
    if (p[j] != 0.0f) maxexp = max(maxexp, ex);
  }
  const float cf = float(c);
  if (cf != 0.0f) maxexp = max(maxexp, int((as_type<uint>(cf) >> 23u) & 255u) - 127);
  if (maxexp == -300) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25; const float fs = as_type<float>(uint(127 + 25 - maxexp) << 23u);
  uint pos = 0u, neg = 0u;
  for (uint j = 0u; j < 32u; ++j) { const float t = trunc(p[j] * fs); const uint m = uint(fabs(t)); if (t < 0.0f) neg += m; else pos += m; }
  { const float t = trunc(cf * fs); const uint m = uint(fabs(t)); if (t < 0.0f) neg += m; else pos += m; }
  return nr_hw_fixed_to_half_rne(long(pos) - long(neg), scale);
#elif NR_QMMA_NOARRAY
  // Candidate 10: same rule, no per-product thread arrays; the second pass re-decodes the (cheap) half fields.
  int maxexp = -100000;
#pragma clang loop unroll(full)
  for (uint j = 0u; j < 32u; ++j) {
    const ushort ab = as_type<ushort>(a[j]), bb = as_type<ushort>(b[j]);
    const int ea = int((ab >> 10u) & 31u), eb = int((bb >> 10u) & 31u);
    const int sa = (ea == 0) ? int(ab & 1023u) : 1024 + int(ab & 1023u), sb = (eb == 0) ? int(bb & 1023u) : 1024 + int(bb & 1023u);
    const int qa = (ea == 0) ? -24 : ea - 25, qb = (eb == 0) ? -24 : eb - 25;
    if (sa != 0 && sb != 0) maxexp = max(maxexp, qa + qb + (62 - int(clz(uint(sa))) - int(clz(uint(sb)))));
  }
  const ushort cb = as_type<ushort>(c); const int ce = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  int csig = 0, cq = 0;
  if (ce != 0) { csig = 1024 + cfr; cq = ce - 25; } else if (cfr != 0) { csig = cfr; cq = -24; }
  if (csig != 0) maxexp = max(maxexp, cq + 31 - int(clz(uint(csig))));
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25; long total = 0l;
#pragma clang loop unroll(full)
  for (uint j = 0u; j < 32u; ++j) {
    const ushort ab = as_type<ushort>(a[j]), bb = as_type<ushort>(b[j]);
    const int ea = int((ab >> 10u) & 31u), eb = int((bb >> 10u) & 31u);
    const int sa = (ea == 0) ? int(ab & 1023u) : 1024 + int(ab & 1023u), sb = (eb == 0) ? int(bb & 1023u) : 1024 + int(bb & 1023u);
    const int psig = sa * sb; if (psig == 0) continue;
    const int qa = (ea == 0) ? -24 : ea - 25, qb = (eb == 0) ? -24 : eb - 25;
    const int sh = qa + qb - scale; long mag = long(psig);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (((ab ^ bb) & 0x8000u) != 0u) ? -mag : mag;
  }
  if (csig != 0) {
    const int sh = cq - scale; long mag = long(csig);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (cb & 0x8000u) != 0u ? -mag : mag;
  }
  return nr_hw_fixed_to_half_rne(total, scale);
#else
  int psig[32], pq[32], pexp[32]; bool pneg[32]; int maxexp = -100000;
  for (uint j = 0u; j < 32u; ++j) {
    const ushort ab = as_type<ushort>(a[j]), bb = as_type<ushort>(b[j]);
    const int ea = int((ab >> 10u) & 31u), eb = int((bb >> 10u) & 31u);
    const int sa = (ea == 0) ? int(ab & 1023u) : 1024 + int(ab & 1023u), sb = (eb == 0) ? int(bb & 1023u) : 1024 + int(bb & 1023u);
    const int qa = (ea == 0) ? -24 : ea - 25, qb = (eb == 0) ? -24 : eb - 25;           // value = sig * 2^q
    psig[j] = sa * sb; pq[j] = qa + qb; pneg[j] = ((ab ^ bb) & 0x8000u) != 0u;
    const int la = nr_vit_bitlen(sa), lb = nr_vit_bitlen(sb);
    pexp[j] = (sa > 0 && sb > 0) ? (qa + la - 1) + (qb + lb - 1) : -100000;
    if (psig[j] != 0) maxexp = max(maxexp, pexp[j]);
  }
  const ushort cb = as_type<ushort>(c); const int ce = int((cb >> 10u) & 31u), cfr = int(cb & 1023u);
  int csig = 0, cq = 0;
  if (ce != 0) { csig = 1024 + cfr; cq = ce - 25; } else if (cfr != 0) { csig = cfr; cq = -24; }
  if (csig != 0) { const int lc = nr_vit_bitlen(csig); maxexp = max(maxexp, cq + lc - 1); }
  if (maxexp == -100000) return as_type<half>(ushort(0u));
  const int scale = maxexp - 25; long total = 0l;
  for (uint j = 0u; j < 32u; ++j) {
    if (psig[j] == 0) continue;
    const int sh = pq[j] - scale; long mag = long(psig[j]);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += pneg[j] ? -mag : mag;
  }
  if (csig != 0) {
    const int sh = cq - scale; long mag = long(csig);
    mag = (sh >= 0) ? (mag << uint(sh)) : (sh <= -63 ? 0l : (mag >> uint(-sh)));
    total += (cb & 0x8000u) != 0u ? -mag : mag;
  }
  return nr_hw_fixed_to_half_rne(total, scale);
#endif
}
inline half nr_table_lookup(device const ushort* table, half x) {          // RTX rsqrt/rcp for a positive finite half
  uint bits = as_type<ushort>(x);
  return as_type<half>(table[clamp(bits, 1u, 0x7bffu) - 1u]);
}
struct NrParams { uint tokens; uint block_index; uint stage; uint reserved; };
