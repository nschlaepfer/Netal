#include <metal_stdlib>
#include "hmma_exact.metalh"
#include "hmma_exact_lanes_fast.metalh"
using namespace metal;
inline half fgf_hmma_f16_lanes16_OLD(ushort a_bits, ushort b_bits, half prior, uint lane) {
  const uint a_field = (uint(a_bits) >> 10u) & 31u, b_field = (uint(b_bits) >> 10u) & 31u;
  uint a_mantissa = uint(a_bits) & 1023u, b_mantissa = uint(b_bits) & 1023u;
  if (a_field != 0u) a_mantissa += 1024u;
  if (b_field != 0u) b_mantissa += 1024u;
  const int a_exponent = a_field == 0u ? -14 : int(a_field) - 15;
  const int b_exponent = b_field == 0u ? -14 : int(b_field) - 15;
  const uint sig = a_mantissa * b_mantissa * 8u;
  const int exponent = a_exponent + b_exponent;
  const bool negative = ((a_bits ^ b_bits) & 0x8000u) != 0u;

  int max_exp = sig != 0u ? exponent : -21;
  max_exp = max(max_exp, simd_shuffle_xor(max_exp, 1u));
  max_exp = max(max_exp, simd_shuffle_xor(max_exp, 2u));
  max_exp = max(max_exp, simd_shuffle_xor(max_exp, 4u));
  max_exp = max(max_exp, simd_shuffle_xor(max_exp, 8u));

  const ushort c_bits = as_type<ushort>(prior);
  const uint c_field = (uint(c_bits) >> 10u) & 31u, c_fraction = uint(c_bits) & 1023u;
  uint c_significand = 0u; int c_exponent = -1024;
  if (c_field != 0u) { c_significand = (1024u + c_fraction) << 13u; c_exponent = int(c_field) - 15; }
  else if (c_fraction != 0u) { const uint leading = 31u - clz(c_fraction); c_significand = c_fraction << (23u - leading); c_exponent = int(leading) - 24; }
  if (c_significand != 0u) max_exp = max(max_exp, c_exponent);

  const int shift = max_exp - exponent;
  const uint aligned = shift > 31 ? 0u : (sig << 2u) >> uint(shift);
  int term = negative ? -int(aligned) : int(aligned);
  term += simd_shuffle_xor(term, 1u);
  term += simd_shuffle_xor(term, 2u);
  term += simd_shuffle_xor(term, 4u);
  term += simd_shuffle_xor(term, 8u);
  long total = long(term);
  if (c_significand != 0u) {
    const int cshift = max_exp - c_exponent;
    const uint caligned = cshift > 31 ? 0u : (c_significand << 2u) >> uint(cshift);
    total += (c_bits & 0x8000u) != 0u ? -long(caligned) : long(caligned);
  }
  return fgf_fixed_to_half_rne(total, max_exp - 25);
}

// Lane-parallel helper vs the scalar oracle: 16 lanes share one random vector (seeded by the vector index),
// each lane supplies its product's operands; lane 0 also runs the scalar oracle. Mixes normals, subnormals,
// zeros, tiny magnitudes (to exercise the -21 exponent floor) and signs.
inline uint hash(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
inline ushort pick(uint r, uint mode){
  ushort v;
  switch (mode) {
    case 0: v = ushort(r & 0xFFFFu); break;                                  // any pattern
    case 1: v = ushort(r & 0x83FFu); break;                                  // subnormals and zeros
    case 2: v = ushort((r & 0x8000u) | (0x0400u + (r & 0x0FFFu))); break;    // tiny normals: exponents -14..-11
    case 3: v = ushort(0x7bffu ^ (r & 0x8000u)); break;                      // max magnitude
    default: v = ushort((r & 0x8000u) | ((0x0400u + (r & 0x3FFFu)) & 0x7BFFu));
  }
  if ((v & 0x7C00u) == 0x7C00u) v &= 0x7BFFu; return v; }
kernel void hmma_lanes_test(device uint* out [[buffer(0)]], constant uint& seed [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  const uint vec = tid / 16u, k = tid % 16u; uint r = hash(vec * 2654435761u + seed);
  const uint mode = (vec >> 6) % 6u;
  ushort av[16], bv[16];
  for (uint i = 0u; i < 16u; ++i) {
    if (mode == 5u) { r = hash(r); av[i] = ushort((r & 0x8000u) | ((r >> 4) & 0x0FFFu) | (((r >> 16) & 1u) ? 0x0400u : 0x0000u)); r = hash(r); bv[i] = ushort((r & 0x8000u) | ((r >> 4) & 0x0FFFu) | (((r >> 16) & 1u) ? 0x0400u : 0x0000u)); if ((av[i] & 0x7FFFu) == 0u) av[i] |= 1u; if ((bv[i] & 0x7FFFu) == 0u) bv[i] |= 1u; }
    else { r = hash(r); av[i] = pick(r, (mode + (r >> 20)) % 5u); r = hash(r); bv[i] = pick(r, (mode + (r >> 21)) % 5u); } }
  r = hash(r); const ushort p = (mode == 5u) ? ((r & 4u) ? ushort(0) : ushort(r & 0x83FFu)) : ((mode == 2u && (r & 4u)) ? ushort(0) : pick(r, r % 5u));   // zero priors often in the tiny mode
  const half prior = as_type<half>(p);
  const half lanes = fgf_hmma_f16_lanes16(av[k], bv[k], prior, k);
  const half lanes_old = fgf_hmma_f16_lanes16_OLD(av[k], bv[k], prior, k);
  if (k == 0u) { const half o = fg_hmma_f16_bits_16(av, bv, prior); out[vec] = (as_type<ushort>(o) != as_type<ushort>(lanes) ? 1u : 0u) | (as_type<ushort>(o) != as_type<ushort>(lanes_old) ? 2u : 0u); }
}
