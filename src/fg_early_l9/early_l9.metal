#include <metal_stdlib>

using namespace metal;

struct EarlyL9Parameters {
  uint width;
  uint height;
};

inline float selected_sm120_rcp(float value,
                                device const uchar* correction_rom) {
  const uint input_bits = as_type<uint>(value);
  uint base_bits = as_type<uint>(1.0f / value);
  const uint mantissa = input_bits & 0x7fffffu;
  const uchar packed = correction_rom[mantissa >> 2u];
  const uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u) {
    base_bits -= 1u;
  } else if (code == 2u) {
    base_bits += 1u;
  }
  return as_type<float>(base_bits);
}

inline ushort float_to_half_rz_bits(float value) {
  const uint bits = as_type<uint>(value);
  const ushort sign = ushort((bits >> 16u) & 0x8000u);
  const uint exponent = (bits >> 23u) & 0xffu;
  const uint mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
  }
  const int half_exponent = int(exponent) - 127 + 15;
  if (half_exponent >= 31) return ushort(sign | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return sign;
    const uint significand = mantissa | 0x800000u;
    const uint shift = uint(14 - half_exponent);
    return ushort(sign | ushort(significand >> shift));
  }
  return ushort(sign | ushort(uint(half_exponent) << 10u) |
                ushort(mantissa >> 13u));
}

kernel void fg_early_l9(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    device const uchar* correction_rom [[buffer(0)]],
    constant EarlyL9Parameters& parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.width || position.y >= parameters.height) {
    return;
  }
  const float4 value = float4(input.read(position));
  const float rg = value.x + value.y;
  const float rgb = value.z + rg;
  const float average = saturate(
      rgb * selected_sm120_rcp(as_type<float>(uint(0x40400000u)),
                               correction_rom));
  const half first = as_type<half>(float_to_half_rz_bits(average));
  output.write(half4(first, 0.0h, 0.0h, 0.0h), position);
}
