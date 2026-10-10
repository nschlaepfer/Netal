#include <metal_stdlib>

using namespace metal;

struct EarlyL3L13Parameters {
  uint width;
  uint height;
  uint half_width;
  uint half_height;
};

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
    return ushort(sign | ushort(significand >> uint(14 - half_exponent)));
  }
  return ushort(sign | ushort(uint(half_exponent) << 10u) |
                ushort(mantissa >> 13u));
}

inline half half_rz(float value) {
  return as_type<half>(float_to_half_rz_bits(value));
}

inline half4 half4_rz(float3 value) {
  return half4(half_rz(value.x), half_rz(value.y), half_rz(value.z),
               half(0.0h));
}

inline float selected_luminance(float3 value) {
  const float green = value.y * as_type<float>(uint(0x3f1645a2u));
  const float red_green = fma(
      value.x, as_type<float>(uint(0x3e991687u)), green);
  return saturate(fma(value.z, as_type<float>(uint(0x3de978d5u)),
                      red_green));
}

kernel void fg_early_l3_l13_mode2(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<half, access::read> history [[texture(1)]],
    texture2d<half, access::write> half_color [[texture(2)]],
    texture2d<half, access::write> half_metric [[texture(3)]],
    texture2d<half, access::write> next_history [[texture(4)]],
    constant EarlyL3L13Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.half_width ||
      position.y >= parameters.half_height) {
    return;
  }
  const uint2 p00 = position * 2u;
  const uint2 p10 = p00 + uint2(1u, 0u);
  const uint2 p01 = p00 + uint2(0u, 1u);
  const uint2 p11 = p00 + uint2(1u, 1u);
  const float3 c00 = source.read(p00).xyz;
  const float3 c10 = source.read(p10).xyz;
  const float3 c01 = source.read(p01).xyz;
  const float3 c11 = source.read(p11).xyz;
  next_history.write(half4_rz(c00), p00);
  next_history.write(half4_rz(c10), p10);
  next_history.write(half4_rz(c01), p01);
  next_history.write(half4_rz(c11), p11);

  const float3 h00 = float3(history.read(p00).xyz);
  const float3 h10 = float3(history.read(p10).xyz);
  const float3 h01 = float3(history.read(p01).xyz);
  const float3 h11 = float3(history.read(p11).xyz);
  const float d00 = abs(selected_luminance(c00) - selected_luminance(h00));
  const float d10 = abs(selected_luminance(c10) - selected_luminance(h10));
  const float d01 = abs(selected_luminance(c01) - selected_luminance(h01));
  const float d11 = abs(selected_luminance(c11) - selected_luminance(h11));
  const float3 row0 = c00 + c10;
  const float3 row1 = c01 + c11;
  const float3 average = (row0 + row1) * 0.25f;
  const float metric = min(min(d00, d10), min(d01, d11));
  half_color.write(half4_rz(average), position);
  half_metric.write(half4(half_rz(metric), half(0.0h), half(0.0h),
                          half(0.0h)), position);
}
