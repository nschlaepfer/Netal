#include <metal_stdlib>

using namespace metal;

constant uint kDenseCount = (2u << 21u) + 1u;
constant uint kRandomCount = 1u << 20u;

struct RandomRecord {
  uint input_bits;
  uint ex2_bits;
  ushort low_half_rz;
  ushort high_half_rz;
};

inline ulong splitmix64(ulong value) {
  value += 0x9E3779B97F4A7C15ul;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ul;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBul;
  return value ^ (value >> 31u);
}

inline float selected_sm120_rcp(float value,
                                device const uchar* correction_rom) {
  const uint input_bits = as_type<uint>(value);
  uint base_bits = as_type<uint>(1.0f / value);
  const uint mantissa = input_bits & 0x7fffffu;
  const uchar packed = correction_rom[mantissa >> 2u];
  const uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u) base_bits -= 1u;
  else if (code == 2u) base_bits += 1u;
  return as_type<float>(base_bits);
}

inline ushort float_to_half_rz_bits(float value) {
  const uint bits = as_type<uint>(value);
  const ushort sign = ushort((bits >> 16u) & 0x8000u);
  const uint exponent = (bits >> 23u) & 0xffu;
  const uint mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu)
    return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
  const int half_exponent = int(exponent) - 127 + 15;
  if (half_exponent >= 31) return ushort(sign | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return sign;
    return ushort(sign | ushort((mantissa | 0x800000u) >>
                                uint(14 - half_exponent)));
  }
  return ushort(sign | ushort(uint(half_exponent) << 10u) |
                ushort(mantissa >> 13u));
}

kernel void fg_mid_l52_dense_ex2(
    device uint* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= kDenseCount) return;
  const float input = -float(index) * as_type<float>(uint(0x35000000u));
  output[index] = as_type<uint>(fast::exp2(input));
}

kernel void fg_mid_l52_random_ex2(
    device RandomRecord* output [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= kRandomCount) return;
  const ulong mixed = splitmix64(0xD6E8FEB86659FD93ul ^ ulong(index));
  uint input_bits;
  if (index == 0u) input_bits = 0u;
  else if (index == 1u) input_bits = 0x80000000u;
  else {
    const uint exponent = 110u + uint((mixed >> 48u) % 21u);
    input_bits = 0x80000000u | (exponent << 23u) |
                 uint(mixed & 0x7ffffful);
  }
  const float exponential = fast::exp2(as_type<float>(input_bits));
  const float reciprocal = selected_sm120_rcp(exponential + 1.0f,
                                               correction_rom);
  RandomRecord record;
  record.input_bits = input_bits;
  record.ex2_bits = as_type<uint>(exponential);
  record.low_half_rz = float_to_half_rz_bits(exponential * reciprocal);
  record.high_half_rz = float_to_half_rz_bits(reciprocal);
  output[index] = record;
}
