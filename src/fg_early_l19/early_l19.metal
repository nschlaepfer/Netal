#include <metal_stdlib>

using namespace metal;

struct L19Parameters {
  uint input_width;
  uint input_height;
  uint tensor_width;
  uint tensor_height;
  int input_y_offset;
};

struct L19Features {
  float value[10];
  float mask_output;
  float2 vector_output;
};

inline int2 normalized_point_coordinate(float2 coordinate, uint2 extent) {
  if (!(coordinate.x >= 0.0f && coordinate.x < 1.0f &&
        coordinate.y >= 0.0f && coordinate.y < 1.0f)) {
    return int2(-1);
  }
  const uint2 fixed12 = uint2(coordinate * 4096.0f);
  return int2((fixed12 * extent) >> 12u);
}

inline float4 read_half4(texture2d<half, access::read> input,
                         int2 position, uint2 extent) {
  if (position.x < 0 || position.y < 0 || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) {
    return float4(0.0f);
  }
  return float4(input.read(uint2(position)));
}

inline float2 read_half2(texture2d<half, access::read> input,
                         int2 position, uint2 extent) {
  return read_half4(input, position, extent).xy;
}

inline float read_half1(texture2d<half, access::read> input,
                        int2 position, uint2 extent) {
  return read_half4(input, position, extent).x;
}

inline float4 sample_half4(texture2d<half, access::read> input,
                           float2 coordinate, uint2 extent) {
  return read_half4(input,
                    normalized_point_coordinate(coordinate, extent), extent);
}

inline float2 sample_half2(texture2d<half, access::read> input,
                           float2 coordinate, uint2 extent) {
  return sample_half4(input, coordinate, extent).xy;
}

inline float2 read_float2(texture2d<float, access::read> input,
                          int2 position, uint2 extent) {
  if (position.x < 0 || position.y < 0 || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) {
    return float2(0.0f);
  }
  return input.read(uint2(position)).xy;
}

inline float2 sample_float2(texture2d<float, access::read> input,
                            float2 coordinate, uint2 extent) {
  return read_float2(input,
                     normalized_point_coordinate(coordinate, extent), extent);
}

inline float read_unorm(texture2d<float, access::read> input,
                        int2 position, uint2 extent) {
  if (position.x < 0 || position.y < 0 || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) {
    return 0.0f;
  }
  return input.read(uint2(position)).x;
}

inline float sample_unorm(texture2d<float, access::read> input,
                          float2 coordinate, uint2 extent) {
  return read_unorm(input,
                    normalized_point_coordinate(coordinate, extent), extent);
}

inline float luma(float4 color) {
  const float green = color.y * as_type<float>(0x3F1645A2u);
  const float red_green = fma(
      color.x, as_type<float>(0x3E991687u), green);
  return clamp(fma(color.z, as_type<float>(0x3DE978D5u), red_green),
               0.0f, 1.0f);
}

inline float mean_absolute_rgb(float4 left, float4 right) {
  const float3 difference = abs(left.xyz - right.xyz);
  const float first_two = difference.x + difference.y;
  const float total = difference.z + first_two;
  return clamp(total * as_type<float>(0x3EAAAAABu), 0.0f, 1.0f);
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

inline bool negative_half_y_retains_center(int y) {
  switch (y) {
    case 58:
    case 97:
    case 103:
    case 110:
    case 116:
    case 123:
    case 129:
    case 142:
    case 148:
    case 155:
    case 161:
    case 168:
    case 174:
      return true;
    default:
      return false;
  }
}

inline bool positive_half_y_advances(int y) {
  switch (y) {
    case 57:
    case 70:
    case 83:
    case 102:
    case 115:
    case 147:
    case 160:
      return true;
    default:
      return false;
  }
}

inline L19Features compute_features(
    texture2d<half, access::read> color_reset,
    texture2d<half, access::read> color_generated,
    texture2d<float, access::read> mask_generated,
    texture2d<float, access::read> mask_reset,
    texture2d<half, access::read> projection,
    texture2d<half, access::read> scalar_shared,
    texture2d<half, access::read> scalar_persistent,
    texture2d<float, access::read> vector_current,
    texture2d<half, access::read> vector_repaired,
    texture2d<float, access::read> mask_working,
    int2 position, uint2 extent) {
  L19Features result{};
  if (position.x < 0 || position.y < 0 || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) {
    return result;
  }
  const float2 inverse_extent = float2(
      as_type<float>(0x3B4CCCCDu), as_type<float>(0x3BB60B61u));
  const float2 normalized =
      (float2(position) + 0.5f) * inverse_extent;
  const float working_mask = sample_unorm(mask_working, normalized, extent);
  const float reset_mask = sample_unorm(mask_reset, normalized, extent);
  const float generated_mask = sample_unorm(mask_generated, normalized, extent);
  result.mask_output = clamp(
      (working_mask * reset_mask) * generated_mask, 0.0f, 1.0f);

  const float4 reset_base = read_half4(color_reset, position, extent);
  const float4 generated_base = read_half4(color_generated, position, extent);
  const float2 repaired = sample_half2(vector_repaired, normalized, extent);
  const float2 repaired_normalized = repaired * inverse_extent;
  const float2 current = sample_float2(vector_current, normalized, extent);
  const float4 reset_shifted =
      sample_half4(color_reset, normalized + current, extent);
  int2 generated_coordinate = normalized_point_coordinate(
      normalized + repaired_normalized, extent);
  if (repaired.y == -0.5f && negative_half_y_retains_center(position.y)) {
    generated_coordinate.y = position.y;
  } else if (repaired.y == 0.5f &&
             positive_half_y_advances(position.y)) {
    generated_coordinate.y = position.y + 1;
  }
  const float4 generated_shifted =
      read_half4(color_generated, generated_coordinate, extent);
  const float generated_difference =
      mean_absolute_rgb(generated_shifted, reset_base);
  const float reset_difference =
      mean_absolute_rgb(reset_shifted, generated_base);
  result.vector_output = float2(reset_difference, generated_difference);

  const float persistent = read_half1(scalar_persistent, position, extent);
  result.value[0] = read_half1(scalar_shared, position, extent);
  result.value[1] = luma(reset_base);
  result.value[2] = generated_difference;
  result.value[3] = read_half2(projection, position, extent).x;
  result.value[4] = working_mask;
  result.value[5] = luma(generated_base);
  result.value[6] = reset_difference;
  result.value[7] = generated_mask;
  // cvt.rn.f16x2 packs the second source into the low halfword on SM120.
  result.value[8] = persistent;
  result.value[9] = 1.0f - persistent;
  return result;
}

kernel void fg_early_l19_surfaces(
    texture2d<half, access::read> color_reset [[texture(0)]],
    texture2d<half, access::read> color_generated [[texture(1)]],
    texture2d<float, access::read> mask_generated [[texture(2)]],
    texture2d<float, access::read> mask_reset [[texture(3)]],
    texture2d<half, access::read> projection [[texture(4)]],
    texture2d<half, access::read> scalar_shared [[texture(5)]],
    texture2d<half, access::read> scalar_persistent [[texture(6)]],
    texture2d<float, access::read> vector_current [[texture(7)]],
    texture2d<half, access::read> vector_repaired [[texture(8)]],
    texture2d<float, access::read> mask_working [[texture(9)]],
    texture2d<float, access::write> mask_output [[texture(10)]],
    texture2d<half, access::write> vector_output [[texture(11)]],
    constant L19Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.input_width ||
      position.y >= parameters.input_height) {
    return;
  }
  const uint2 extent = uint2(parameters.input_width, parameters.input_height);
  const L19Features features = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(position), extent);
  mask_output.write(float4(features.mask_output, 0.0f, 0.0f, 0.0f),
                    position);
  vector_output.write(
      half4(as_type<half>(float_to_half_rz_bits(features.vector_output.x)),
            as_type<half>(float_to_half_rz_bits(features.vector_output.y)),
            0.0h, 0.0h),
      position);
}

kernel void fg_early_l19_tensor(
    texture2d<half, access::read> color_reset [[texture(0)]],
    texture2d<half, access::read> color_generated [[texture(1)]],
    texture2d<float, access::read> mask_generated [[texture(2)]],
    texture2d<float, access::read> mask_reset [[texture(3)]],
    texture2d<half, access::read> projection [[texture(4)]],
    texture2d<half, access::read> scalar_shared [[texture(5)]],
    texture2d<half, access::read> scalar_persistent [[texture(6)]],
    texture2d<float, access::read> vector_current [[texture(7)]],
    texture2d<half, access::read> vector_repaired [[texture(8)]],
    texture2d<float, access::read> mask_working [[texture(9)]],
    device ushort* tensor_output [[buffer(0)]],
    constant L19Parameters& parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.tensor_width ||
      position.y >= parameters.tensor_height) {
    return;
  }
  const uint2 extent = uint2(parameters.input_width, parameters.input_height);
  const int base_x = int(position.x * 2u);
  const int base_y = int(position.y * 2u) + parameters.input_y_offset;
  L19Features subpixels[4];
  subpixels[0] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x, base_y), extent);
  subpixels[1] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x + 1, base_y), extent);
  subpixels[2] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x, base_y + 1), extent);
  subpixels[3] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x + 1, base_y + 1), extent);
  const uint base =
      (position.y * parameters.tensor_width + position.x) * 48u;
  for (uint feature = 0; feature < 10u; ++feature) {
    for (uint subpixel = 0; subpixel < 4u; ++subpixel) {
      tensor_output[base + feature * 4u + subpixel] =
          as_type<ushort>(half(subpixels[subpixel].value[feature]));
    }
  }
  for (uint channel = 40u; channel < 48u; ++channel) {
    tensor_output[base + channel] = 0u;
  }
}
