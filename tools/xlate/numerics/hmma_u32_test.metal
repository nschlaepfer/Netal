#include <metal_stdlib>
#include "hmma_exact.metalh"
#include "hmma_exact_u32.metalh"
using namespace metal;
// Compares the u32 helper with the oracle on pseudo-random operand vectors mixing normals, subnormals, zeros,
// large magnitudes and signs; also the worst-case magnitude vector (all 0x7bff). out[tid] = 1 on mismatch.
inline uint hash(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
inline ushort pick(uint r, uint mode){
  switch (mode) {
    case 0: return ushort(r & 0xFFFFu);                       // any bit pattern (may include inf/nan patterns: excluded below)
    case 1: return ushort((r & 0x83FFu));                     // subnormals and zeros
    case 2: return ushort(0x3800u + (r & 0x0FFFu) - 0x800u);  // around 1
    case 3: return ushort(0x7bffu ^ (r & 0x8000u));           // max magnitude
    default: return ushort((r & 0x8000u) | ((0x0400u + (r & 0x3FFFu)) & 0x7BFFu));
  }
}
kernel void hmma_u32_test(device uint* out [[buffer(0)]], constant uint& seed [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  ushort av[16], bv[16]; uint r = hash(tid * 2654435761u + seed);
  const uint mode = (tid >> 8) % 5u;
  for (uint i = 0u; i < 16u; ++i) { r = hash(r); ushort a = pick(r, (mode + (r >> 20)) % 5u); r = hash(r); ushort b = pick(r, (mode + (r >> 21)) % 5u);
    if ((a & 0x7C00u) == 0x7C00u) a &= 0x7BFFu; if ((b & 0x7C00u) == 0x7C00u) b &= 0x7BFFu; av[i] = a; bv[i] = b; }
  r = hash(r); ushort p = pick(r, r % 5u); if ((p & 0x7C00u) == 0x7C00u) p &= 0x7BFFu;
  const half o = fg_hmma_f16_bits_16(av, bv, as_type<half>(p)), u = fgu_hmma_f16_bits_16(av, bv, as_type<half>(p));
  out[tid] = as_type<ushort>(o) != as_type<ushort>(u) ? 1u : 0u;
}
