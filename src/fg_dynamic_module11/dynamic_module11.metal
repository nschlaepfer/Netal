#include <metal_stdlib>

using namespace metal;

struct DynamicModule11Parameters {
  uint source_width;
  uint source_height;
  uint output_width;
  uint output_height;
  uint mode;
};

inline ushort dynamic_float_to_half_rz_bits(float value) {
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

inline half dynamic_half_rz(float value) {
  return as_type<half>(dynamic_float_to_half_rz_bits(value));
}

inline half4 dynamic_half4_rz(float3 value) {
  return half4(dynamic_half_rz(value.x), dynamic_half_rz(value.y),
               dynamic_half_rz(value.z), half(0.0h));
}

inline float dynamic_luminance(float3 value) {
  const float green = value.y * as_type<float>(uint(0x3f1645a2u));
  const float red_green =
      fma(value.x, as_type<float>(uint(0x3e991687u)), green);
  return saturate(fma(value.z, as_type<float>(uint(0x3de978d5u)),
                      red_green));
}

inline float dynamic_metric(float3 source, float3 history) {
  return abs(dynamic_luminance(source) - dynamic_luminance(history));
}

kernel void fg_dynamic_module11(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<half, access::read> history [[texture(1)]],
    texture2d<half, access::write> output_color [[texture(2)]],
    texture2d<half, access::write> output_metric [[texture(3)]],
    texture2d<half, access::write> next_history [[texture(4)]],
    constant DynamicModule11Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height)
    return;

  if (parameters.mode == 2u) {
    const uint2 p00 = position * 2u;
    const uint2 p10 = p00 + uint2(1u, 0u);
    const uint2 p01 = p00 + uint2(0u, 1u);
    const uint2 p11 = p00 + uint2(1u, 1u);
    const float3 c00 = source.read(p00).xyz;
    const float3 c10 = source.read(p10).xyz;
    const float3 c01 = source.read(p01).xyz;
    const float3 c11 = source.read(p11).xyz;
    next_history.write(dynamic_half4_rz(c00), p00);
    next_history.write(dynamic_half4_rz(c10), p10);
    next_history.write(dynamic_half4_rz(c01), p01);
    next_history.write(dynamic_half4_rz(c11), p11);
    const float m00 = dynamic_metric(c00, float3(history.read(p00).xyz));
    const float m10 = dynamic_metric(c10, float3(history.read(p10).xyz));
    const float m01 = dynamic_metric(c01, float3(history.read(p01).xyz));
    const float m11 = dynamic_metric(c11, float3(history.read(p11).xyz));
    const float3 row0 = c00 + c10;
    const float3 row1 = c01 + c11;
    const float3 average = (row0 + row1) * 0.25f;
    const float metric = min(min(m00, m10), min(m01, m11));
    output_color.write(dynamic_half4_rz(average), position);
    output_metric.write(
        half4(dynamic_half_rz(metric), half(0.0h), half(0.0h), half(0.0h)),
        position);
    return;
  }

  if (parameters.mode != 4u) return;
  const uint2 base = position * 4u;
  float3 color[4][4];
  float metric[4][4];
  for (uint y = 0; y < 4u; ++y) {
    for (uint x = 0; x < 4u; ++x) {
      const uint2 location = base + uint2(x, y);
      color[y][x] = source.read(location).xyz;
      next_history.write(dynamic_half4_rz(color[y][x]), location);
      metric[y][x] = dynamic_metric(
          color[y][x], float3(history.read(location).xyz));
    }
  }
  const float3 q00 = (color[0][0] + color[0][1]) +
                     (color[1][0] + color[1][1]);
  const float3 q01 = (color[0][2] + color[0][3]) +
                     (color[1][2] + color[1][3]);
  const float3 q10 = (color[2][0] + color[2][1]) +
                     (color[3][0] + color[3][1]);
  const float3 q11 = (color[2][2] + color[2][3]) +
                     (color[3][2] + color[3][3]);
  const float3 top = q00 + q01;
  const float3 bottom = q10 + q11;
  const float3 average = (top + bottom) * 0.0625f;
  const float m00 = min(min(metric[0][0], metric[0][1]),
                        min(metric[1][0], metric[1][1]));
  const float m01 = min(min(metric[0][2], metric[0][3]),
                        min(metric[1][2], metric[1][3]));
  const float m10 = min(min(metric[2][0], metric[2][1]),
                        min(metric[3][0], metric[3][1]));
  const float m11 = min(min(metric[2][2], metric[2][3]),
                        min(metric[3][2], metric[3][3]));
  const float metric_average = ((m00 + m01) + (m10 + m11)) * 0.25f;
  output_color.write(dynamic_half4_rz(average), position);
  output_metric.write(
      half4(dynamic_half_rz(metric_average), half(0.0h), half(0.0h),
            half(0.0h)),
      position);
}
