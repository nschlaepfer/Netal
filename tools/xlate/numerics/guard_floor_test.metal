#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;
// Regression for the exponent-floor hole: prior 0, one product 2^-25, fifteen products 2^-48. The oracle's
// grid is 2^(-21-25) = 2^-46, the fifteen small products truncate away, the tie 2^-25 rounds to even (0x0000);
// the fp32 sum 2^-25 + 15*2^-48 rounds to 0x0001. Reports the oracle result, the fp32 value, and both bounds.
inline float ulp32_bits(float v) { return as_type<float>(as_type<uint>(v) & 0x7F800000u) * 0x1p-23f; }
inline float bound_operand(float v) { const float m = fabs(v); return m != 0.0f ? max(m, 0x1p-14f) : 0.0f; }
kernel void guard_floor_test(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0u) return;
  ushort av[16], bv[16]; av[0] = as_type<ushort>(half(0x1p-13h)); bv[0] = as_type<ushort>(half(0x1p-12h));
  for (uint k = 1u; k < 16u; ++k) { av[k] = as_type<ushort>(half(0x1p-24h)); bv[k] = as_type<ushort>(half(0x1p-24h)); }
  const half oracle = fg_hmma_f16_bits_16(av, bv, half(0.0h));
  float part0 = 0.0f, part1 = 0.0f, bnd0 = 0.0f, bnd1 = 0.0f;
  for (uint k = 0u; k < 8u; ++k) { const float a = float(as_type<half>(av[k])), b = float(as_type<half>(bv[k])); part0 = fma(a, b, part0); bnd0 = fma(bound_operand(a), bound_operand(b), bnd0); }
  for (uint k = 8u; k < 16u; ++k) { const float a = float(as_type<half>(av[k])), b = float(as_type<half>(bv[k])); part1 = fma(a, b, part1); bnd1 = fma(bound_operand(a), bound_operand(b), bnd1); }
  const float value = (0.0f + part0) + part1;
  const float c0 = bnd0 * (1.0f + 0x1p-20f), c1 = bnd1 * (1.0f + 0x1p-20f);
  const float M_old = max(max(c0, c1), 0.0f), M_new = max(0x1p-21f, M_old);
  const float e_old = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.25f * ulp32_bits(M_old) + 1.5625f * ulp32_bits(value);
  const float e_new = 3.5f * (ulp32_bits(c0) + ulp32_bits(c1)) + 4.25f * ulp32_bits(M_new) + 1.5625f * ulp32_bits(value);
  out[0] = as_type<ushort>(oracle); out[1] = as_type<ushort>(half(value));
  out[2] = as_type<ushort>(half(value - e_old)) == as_type<ushort>(half(value + e_old)) ? 1u : 0u;   // 1 = accepted (unsafe)
  out[3] = as_type<ushort>(half(value - e_new)) == as_type<ushort>(half(value + e_new)) ? 1u : 0u;   // 0 = flagged (repair)
  out[4] = as_type<uint>(value); out[5] = as_type<uint>(e_old); out[6] = as_type<uint>(e_new);
}
