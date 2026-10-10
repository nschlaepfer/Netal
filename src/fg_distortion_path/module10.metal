#include <metal_stdlib>

using namespace metal;

struct Module10Parameters {
  uint2 input_origin;
  uint2 input_extent;
  float2 inverse_input_extent;
  uint2 output_extent;
  float output_bias;
  float output_scale;
};

inline int2 module10_point_coordinate(float2 coordinate, uint2 extent) {
  if (!(coordinate.x >= 0.0f && coordinate.x < 1.0f &&
        coordinate.y >= 0.0f && coordinate.y < 1.0f)) {
    return int2(-1);
  }
  const uint2 fixed12 = uint2(coordinate * 4096.0f);
  return int2((fixed12 * extent) >> 12u);
}

inline float4 module10_read(texture2d<half, access::read> input,
                            int2 position, uint2 extent) {
  if (any(position < 0) || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) {
    return 0.0f;
  }
  return float4(input.read(uint2(position)));
}

inline ushort module10_float_to_half_rz(float value) {
  const uint bits = as_type<uint>(value);
  const ushort sign = ushort((bits >> 16u) & 0x8000u);
  const uint exponent = (bits >> 23u) & 0xffu;
  const uint mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu)
    return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
  const int half_exponent = int(exponent) - 112;
  if (half_exponent >= 31) return ushort(sign | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return sign;
    return ushort(sign | ((mantissa | 0x800000u) >> uint(14 - half_exponent)));
  }
  return ushort(sign | (uint(half_exponent) << 10u) | (mantissa >> 13u));
}

kernel void fg_distortion_module10(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant Module10Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (any(position >= parameters.output_extent)) return;

  const float2 normalized =
      (float2(position) + 0.5f) *
      float2(as_type<float>(0x3b4ccccdu), as_type<float>(0x3bb60b61u));
  const float2 source = fma(normalized, float2(parameters.input_extent),
                            float2(parameters.input_origin));
  float2 coordinate = source * parameters.inverse_input_extent;
  // Exact directed endpoint products for the selected live parameters:
  // mul.rp(0.5, inverse) and mul.rm(extent-0.5, inverse).
  coordinate = clamp(
      coordinate,
      float2(as_type<float>(0x3acccccdu), as_type<float>(0x3b360b61u)),
      float2(as_type<float>(0x3f7f9999u), as_type<float>(0x3f7f49f5u)));
  const float4 sampled = module10_read(
      input, module10_point_coordinate(coordinate, parameters.input_extent),
      parameters.input_extent);
  const float4 transformed =
      fma(parameters.output_scale, sampled, parameters.output_bias);
  output.write(
      half4(as_type<half>(module10_float_to_half_rz(transformed.x)),
            as_type<half>(module10_float_to_half_rz(transformed.y)),
            as_type<half>(module10_float_to_half_rz(transformed.z)),
            as_type<half>(module10_float_to_half_rz(transformed.w))),
      position);
}
