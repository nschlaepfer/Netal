#include <metal_stdlib>
#include "../fg_math_ex2/sm120_ex2.metalh"

using namespace metal;

struct MidL54Parameters {
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

inline float selected_div(float numerator, float denominator,
                          device const uchar* correction_rom) {
  return numerator * selected_sm120_rcp(denominator, correction_rom);
}

inline float2 integer_sample(texture2d<half, access::sample> input,
                             int x, int y, uint width, uint height) {
  if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) {
    return float2(0.0f);
  }
  return float2(input.read(uint2(uint(x), uint(y))).xy);
}

inline float2 normalized_point_sample(
    texture2d<half, access::sample> input, float x, float y,
    uint width, uint height) {
  if (!(x >= 0.0f && x < 1.0f && y >= 0.0f && y < 1.0f)) {
    return float2(0.0f);
  }
  const uint fixed_x = uint(x * 4096.0f);
  const uint fixed_y = uint(y * 4096.0f);
  const int ix = int((fixed_x * width) >> 12u);
  const int iy = int((fixed_y * height) >> 12u);
  return integer_sample(input, ix, iy, width, height);
}

inline float confidence(float shifted, float reference,
                        device const uchar* correction_rom,
                        device const uint* ex2_unit_table,
                        device const uint* ex2_fraction_table) {
  const float exponent = abs(shifted - reference) *
                         as_type<float>(uint(0xBFB8AA3Bu));
  const float attenuation = neuralstudio_fg_math::sm120_ex2_nonpositive(
      exponent, ex2_unit_table, ex2_fraction_table);
  const float numerator = shifted > reference ? 1.0f : attenuation;
  return selected_div(numerator, attenuation + 1.0f, correction_rom);
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

kernel void fg_mid_l54(
    texture2d<half, access::sample> texture_a [[texture(0)]],
    texture2d<half, access::sample> texture_b [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    device const ulong* offset_a [[buffer(0)]],
    device const ulong* offset_b [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant MidL54Parameters& parameters [[buffer(3)]],
    device const uint* ex2_unit_table [[buffer(4)]],
    device const uint* ex2_fraction_table [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.width || position.y >= parameters.height) {
    return;
  }
  const uint index = position.y * parameters.width + position.x;
  const float normalized_x = clamp(
      selected_div(float(position.x) + 0.5f, float(parameters.width),
                   correction_rom),
      0.0f, 1.0f);
  const float normalized_y = clamp(
      selected_div(float(position.y) + 0.5f, float(parameters.height),
                   correction_rom),
      0.0f, 1.0f);
  const float pixel_x = fma(normalized_x, float(parameters.width), -0.5f) +
                        as_type<float>(uint(0xB4000000u));
  const float pixel_y = fma(normalized_y, float(parameters.height), -0.5f) +
                        as_type<float>(uint(0xB4000000u));
  const int reference_x = int(rint(pixel_x));
  const int reference_y = int(rint(pixel_y));
  const float2 reference_a = integer_sample(
      texture_a, reference_x, reference_y, parameters.width,
      parameters.height);
  const float2 reference_b = integer_sample(
      texture_b, reference_x, reference_y, parameters.width,
      parameters.height);

  const ulong packed_a = offset_a[index];
  const ulong packed_b = offset_b[index];
  const float2 delta_a = packed_a == 0x7fffffff7ffffffful
      ? float2(0.0f)
      : float2(float(as_type<half>(ushort(packed_a & 0xffffu))),
               float(as_type<half>(ushort((packed_a >> 16u) & 0xffffu))));
  const float2 delta_b = packed_b == 0x7fffffff7ffffffful
      ? float2(0.0f)
      : float2(float(as_type<half>(ushort(packed_b & 0xffffu))),
               float(as_type<half>(ushort((packed_b >> 16u) & 0xffffu))));
  const float2 shifted_a = normalized_point_sample(
      texture_a, normalized_x + delta_b.y, normalized_y + delta_b.x,
      parameters.width, parameters.height);
  const float2 shifted_b = normalized_point_sample(
      texture_b, normalized_x + delta_a.y, normalized_y + delta_a.x,
      parameters.width, parameters.height);
  output.write(
      half4(half2(
                as_type<half>(float_to_half_rz_bits(
                    confidence(shifted_a.x, reference_a.y, correction_rom,
                               ex2_unit_table, ex2_fraction_table))),
                as_type<half>(float_to_half_rz_bits(
                    confidence(shifted_b.x, reference_b.y, correction_rom,
                               ex2_unit_table, ex2_fraction_table)))),
            half2(0.0h)),
      position);
}
