#include <metal_stdlib>

using namespace metal;

#ifndef FG_DYNAMIC_MODULE5_POINT_BITS
#define FG_DYNAMIC_MODULE5_POINT_BITS 21
#endif

struct DynamicModule5Parameters {
  uint motion_width;
  uint motion_height;
  uint work_width;
  uint work_height;
  uint tensor_width;
  uint tensor_height;
  int input_x_offset;
  int input_y_offset;
};

struct DynamicModule5Features {
  float value[10];
  float mask_output;
  float2 vector_output;
};

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

inline float selected_div(float numerator, float denominator,
                          device const uchar* correction_rom) {
  return numerator * selected_sm120_rcp(denominator, correction_rom);
}

inline int2 point_coordinate_21(float2 coordinate, uint2 extent) {
  if (!(coordinate.x >= 0.0f && coordinate.x < 1.0f &&
        coordinate.y >= 0.0f && coordinate.y < 1.0f)) return int2(-1);
  const uint scale = 1u << FG_DYNAMIC_MODULE5_POINT_BITS;
  const uint2 fixed = uint2(coordinate * float(scale));
  return int2((fixed * extent) >> FG_DYNAMIC_MODULE5_POINT_BITS);
}

inline float4 read_half4(texture2d<half, access::read> input,
                         int2 position, uint2 extent) {
  if (any(position < 0) || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) return 0.0f;
  return float4(input.read(uint2(position)));
}

inline float4 sample_half4(texture2d<half, access::read> input,
                           float2 coordinate, uint2 extent) {
  return read_half4(input, point_coordinate_21(coordinate, extent), extent);
}

inline float2 read_float2(texture2d<float, access::read> input,
                          int2 position, uint2 extent) {
  if (any(position < 0) || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) return 0.0f;
  return input.read(uint2(position)).xy;
}

inline float2 sample_float2(texture2d<float, access::read> input,
                            float2 coordinate, uint2 extent) {
  return read_float2(input, point_coordinate_21(coordinate, extent), extent);
}

inline float read_unorm(texture2d<float, access::read> input,
                        int2 position, uint2 extent) {
  if (any(position < 0) || position.x >= int(extent.x) ||
      position.y >= int(extent.y)) return 0.0f;
  return input.read(uint2(position)).x;
}

inline float sample_unorm(texture2d<float, access::read> input,
                          float2 coordinate, uint2 extent) {
  return read_unorm(input, point_coordinate_21(coordinate, extent), extent);
}

inline float selected_luma(float4 color) {
  const float green = color.y * as_type<float>(0x3F1645A2u);
  const float red_green = fma(color.x, as_type<float>(0x3E991687u), green);
  return clamp(fma(color.z, as_type<float>(0x3DE978D5u), red_green),
               0.0f, 1.0f);
}

inline float selected_mean_absolute_rgb(
    float4 left, float4 right, device const uchar* correction_rom) {
  const float3 difference = abs(left.xyz - right.xyz);
  const float first_two = difference.x + difference.y;
  const float total = difference.z + first_two;
  return clamp(selected_div(total, 3.0f, correction_rom), 0.0f, 1.0f);
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

inline DynamicModule5Features compute_features(
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
    int2 position, constant DynamicModule5Parameters& parameters,
    device const uchar* correction_rom) {
  DynamicModule5Features result{};
  const uint2 work_extent = uint2(parameters.work_width,
                                  parameters.work_height);
  if (any(position < 0) || position.x >= int(work_extent.x) ||
      position.y >= int(work_extent.y)) return result;
  const float2 normalized = float2(
      selected_div(float(position.x) + 0.5f, float(parameters.work_width),
                   correction_rom),
      selected_div(float(position.y) + 0.5f, float(parameters.work_height),
                   correction_rom));
  const float working_mask = sample_unorm(mask_working, normalized, work_extent);
  const float reset_mask = sample_unorm(mask_reset, normalized, work_extent);
  const float generated_mask =
      sample_unorm(mask_generated, normalized, work_extent);
  result.mask_output = clamp(
      (working_mask * reset_mask) * generated_mask, 0.0f, 1.0f);

  const float4 reset_base = read_half4(color_reset, position, work_extent);
  const float4 generated_base =
      read_half4(color_generated, position, work_extent);
  const float2 repaired = sample_half4(
      vector_repaired, normalized, work_extent).xy;
  const float2 repaired_normalized = float2(
      selected_div(repaired.x, float(parameters.motion_width), correction_rom),
      selected_div(repaired.y, float(parameters.motion_height), correction_rom));
  const float2 current = sample_float2(
      vector_current, normalized, work_extent);
  const float4 reset_shifted = sample_half4(
      color_reset, normalized + current, work_extent);
  const float4 generated_shifted = sample_half4(
      color_generated, normalized + repaired_normalized, work_extent);
  const float generated_difference = selected_mean_absolute_rgb(
      generated_shifted, reset_base, correction_rom);
  const float reset_difference = selected_mean_absolute_rgb(
      reset_shifted, generated_base, correction_rom);
  result.vector_output = float2(reset_difference, generated_difference);

  const float persistent =
      read_half4(scalar_persistent, position, work_extent).x;
  result.value[0] = read_half4(scalar_shared, position, work_extent).x;
  result.value[1] = selected_luma(reset_base);
  result.value[2] = generated_difference;
  result.value[3] = read_half4(projection, position, work_extent).x;
  result.value[4] = working_mask;
  result.value[5] = selected_luma(generated_base);
  result.value[6] = reset_difference;
  result.value[7] = generated_mask;
  result.value[8] = persistent;
  result.value[9] = 1.0f - persistent;
  return result;
}

kernel void fg_dynamic_module5_surfaces(
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
    constant DynamicModule5Parameters& parameters [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.work_width ||
      position.y >= parameters.work_height) return;
  const auto features = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(position), parameters, correction_rom);
  mask_output.write(float4(features.mask_output, 0.0f, 0.0f, 0.0f), position);
  vector_output.write(
      half4(as_type<half>(float_to_half_rz_bits(features.vector_output.x)),
            as_type<half>(float_to_half_rz_bits(features.vector_output.y)),
            0.0h, 0.0h), position);
}

kernel void fg_dynamic_module5_tensor(
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
    constant DynamicModule5Parameters& parameters [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.tensor_width ||
      position.y >= parameters.tensor_height) return;
  const int base_x = int(position.x * 2u) + parameters.input_x_offset;
  const int base_y = int(position.y * 2u) + parameters.input_y_offset;
  DynamicModule5Features subpixels[4];
  subpixels[0] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x, base_y), parameters, correction_rom);
  subpixels[1] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x + 1, base_y), parameters, correction_rom);
  subpixels[2] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x, base_y + 1), parameters, correction_rom);
  subpixels[3] = compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x + 1, base_y + 1), parameters, correction_rom);
  const uint base =
      (position.y * parameters.tensor_width + position.x) * 48u;
  for (uint feature = 0; feature < 10u; ++feature) {
    for (uint subpixel = 0; subpixel < 4u; ++subpixel) {
      tensor_output[base + feature * 4u + subpixel] =
          as_type<ushort>(half(subpixels[subpixel].value[feature]));
    }
  }
  for (uint channel = 40u; channel < 48u; ++channel)
    tensor_output[base + channel] = 0u;
}
