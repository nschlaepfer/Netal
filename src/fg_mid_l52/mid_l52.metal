#include <metal_stdlib>

using namespace metal;

struct MidL52Parameters {
  uint width;
  uint height;
  uint input_width;
  uint input_height;
  uint input_offset_x;
  uint input_offset_y;
  uint crop_top;
  uint crop_bottom;
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
  if (exponent == 0xffu)
    return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
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

inline half half_rz(float value) {
  return as_type<half>(float_to_half_rz_bits(value));
}

kernel void fg_mid_l52(
    device const half* input [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    constant MidL52Parameters& parameters [[buffer(2)]],
    texture2d<float, access::read> depth [[texture(0)]],
    texture2d<half, access::write> output_a [[texture(1)]],
    texture2d<half, access::write> output_b [[texture(2)]],
    texture2d<half, access::write> confidence [[texture(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.width || position.y >= parameters.height)
    return;
  const uint available_width =
      parameters.input_width - parameters.input_offset_x;
  const uint available_height = parameters.input_height - parameters.crop_top -
                                parameters.crop_bottom;
  const uint input_x = parameters.input_offset_x +
                       min(position.x >> 1u, available_width - 1u);
  const uint input_y = parameters.input_offset_y + parameters.crop_top +
                       min(position.y >> 1u, available_height - 1u);
  const uint subpixel = (position.x & 1u) | ((position.y & 1u) << 1u);
  const uint input_base =
      (input_y * parameters.input_width + input_x) * 16u + subpixel;
  float a0 = float(input[input_base]);
  const float a1 = float(input[input_base + 4u]);
  float b0 = float(input[input_base + 8u]);
  const float b1 = float(input[input_base + 12u]);
  const float depth_value = depth.read(position).x;
  const float replacement = -1000.0f;
  a0 = fma(replacement - a0, depth_value, a0);
  b0 = fma(replacement - b0, depth_value, b0);
  const float exponent_input =
      abs(b0 - b1) * as_type<float>(uint(0xBFB8AA3Bu));
  const float attenuation = fast::exp2(exponent_input);
  const float numerator = b0 > b1 ? 1.0f : attenuation;
  const float confidence_value =
      numerator * selected_sm120_rcp(attenuation + 1.0f, correction_rom);
  output_a.write(half4(half_rz(a0), half_rz(a1), 0.0h, 0.0h), position);
  output_b.write(half4(half_rz(b0), half_rz(b1), 0.0h, 0.0h), position);
  confidence.write(half4(half_rz(confidence_value), 0.0h, 0.0h, 0.0h),
                   position);
}
