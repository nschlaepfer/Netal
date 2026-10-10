#include <metal_stdlib>

using namespace metal;

struct SubrectL84Parameters {
  uint width;
  uint height;
  uint padded_height;
  uint divisor;
  uint output_origin_x;
  uint output_origin_y;
  uint resource_width;
  uint resource_height;
};

struct SubrectL84ReductionParameters {
  uint input_width;
  uint input_height;
  uint output_width;
  uint output_height;
};

inline float subrect_selected_sm120_rcp(
    float value, device const uchar* correction_rom) {
  const uint input_bits = as_type<uint>(value);
  uint base_bits = as_type<uint>(1.0f / value);
  const uint mantissa = input_bits & 0x7fffffu;
  const uchar packed = correction_rom[mantissa >> 2u];
  const uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u) base_bits -= 1u;
  else if (code == 2u) base_bits += 1u;
  return as_type<float>(base_bits);
}

inline bool subrect_mask_need(device const uchar* mask, int x, int y,
                              uint width, uint height) {
  if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) return true;
  return mask[uint(y) * width + uint(x)] != 0u;
}

inline bool subrect_fully_covered(device const uchar* mask, int x, int y,
                                  uint width, uint height) {
  const bool m00 = subrect_mask_need(mask, x - 2, y - 2, width, height);
  const bool m10 = subrect_mask_need(mask, x - 1, y - 2, width, height);
  const bool m20 = subrect_mask_need(mask, x, y - 2, width, height);
  const bool m30 = subrect_mask_need(mask, x + 1, y - 2, width, height);
  const bool m40 = subrect_mask_need(mask, x + 2, y - 2, width, height);
  const bool m01 = subrect_mask_need(mask, x - 2, y - 1, width, height);
  const bool m11 = subrect_mask_need(mask, x - 1, y - 1, width, height);
  const bool m21 = subrect_mask_need(mask, x, y - 1, width, height);
  const bool m31 = subrect_mask_need(mask, x + 1, y - 1, width, height);
  const bool m41 = subrect_mask_need(mask, x + 2, y - 1, width, height);
  const bool m02 = subrect_mask_need(mask, x - 2, y, width, height);
  const bool m12 = subrect_mask_need(mask, x - 1, y, width, height);
  const bool m22 = subrect_mask_need(mask, x, y, width, height);
  const bool m32 = subrect_mask_need(mask, x + 1, y, width, height);
  const bool m42 = subrect_mask_need(mask, x + 2, y, width, height);
  const bool m03 = subrect_mask_need(mask, x - 2, y + 1, width, height);
  const bool m13 = subrect_mask_need(mask, x - 1, y + 1, width, height);
  const bool m23 = subrect_mask_need(mask, x, y + 1, width, height);
  const bool m33 = subrect_mask_need(mask, x + 1, y + 1, width, height);
  const bool m43 = subrect_mask_need(mask, x + 2, y + 1, width, height);
  const bool m04 = subrect_mask_need(mask, x - 2, y + 2, width, height);
  const bool m14 = subrect_mask_need(mask, x - 1, y + 2, width, height);
  const bool m24 = subrect_mask_need(mask, x, y + 2, width, height);
  const bool m34 = subrect_mask_need(mask, x + 1, y + 2, width, height);
  const bool m44 = subrect_mask_need(mask, x + 2, y + 2, width, height);
  const bool top_left = m00 || m10 || m20 || m01 || m11 || m21 || m02 ||
                        m12 || m22 || m03 || m13 || m23;
  const bool top_middle =
      m10 || m20 || m30 || m11 || m21 || m31 || m12 || m22 || m32;
  const bool top_right = m20 || m30 || m40 || m21 || m31 || m41 || m22 ||
                         m32 || m42 || m23 || m33 || m43;
  const bool middle_left =
      m01 || m11 || m21 || m02 || m12 || m22 || m03 || m13 || m23;
  const bool middle =
      m11 || m21 || m31 || m12 || m22 || m32 || m13 || m23 || m33;
  const bool middle_right =
      m21 || m31 || m41 || m22 || m32 || m42 || m23 || m33 || m43;
  const bool bottom_left = m02 || m12 || m22 || m03 || m13 || m23 || m04 ||
                           m14 || m24;
  const bool bottom_middle =
      m12 || m22 || m32 || m13 || m23 || m33 || m14 || m24 || m34;
  const bool bottom_right = m22 || m32 || m42 || m23 || m33 || m43 || m24 ||
                            m34 || m44;
  return top_left && top_middle && top_right && middle_left && middle &&
         middle_right && bottom_left && bottom_middle && bottom_right;
}

inline float3 subrect_sample_bilinear(
    device const half4* color, float2 position, uint width, uint height) {
  const int2 base = int2(floor(position));
  const float2 fraction = position - float2(base);
  const int x0 = clamp(base.x, 0, int(width) - 1);
  const int x1 = clamp(base.x + 1, 0, int(width) - 1);
  const int y0 = clamp(base.y, 0, int(height) - 1);
  const int y1 = clamp(base.y + 1, 0, int(height) - 1);
  const float3 c00 = float3(color[uint(y0) * width + uint(x0)].xyz);
  const float3 c10 = float3(color[uint(y0) * width + uint(x1)].xyz);
  const float3 c01 = float3(color[uint(y1) * width + uint(x0)].xyz);
  const float3 c11 = float3(color[uint(y1) * width + uint(x1)].xyz);
  const float one_minus_x = 1.0f - fraction.x;
  const float one_minus_y = 1.0f - fraction.y;
  const float3 row0 = fma(float3(one_minus_x * one_minus_y), c00,
                          float3(fraction.x * one_minus_y) * c10);
  const float3 with_row1_left =
      fma(float3(fraction.y * one_minus_x), c01, row0);
  return fma(float3(fraction.x * fraction.y), c11, with_row1_left);
}

inline float subrect_selected_unorm8(float value) {
  const uint fixed12 = uint(saturate(value) * 4096.0f);
  const uint encoded = min(255u, (fixed12 * 255u + 2047u) >> 12u);
  return float(encoded) * (1.0f / 255.0f);
}

kernel void fg_subrect_postprocess_l84_pixels(
    device const half4* color_a [[buffer(0)]],
    device const half4* color_b [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const half* motion [[buffer(3)]],
    device const uint* sigmoid_lut [[buffer(4)]],
    device float4* pixel_scratch [[buffer(5)]],
    constant SubrectL84Parameters& parameters [[buffer(6)]],
    texture2d<float, access::write> output_color [[texture(0)]],
    texture2d<float, access::write> output_mask [[texture(1)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.width || position.y >= parameters.padded_height)
    return;
  const uint index = position.y * parameters.width + position.x;
  if (position.y >= parameters.height) {
    pixel_scratch[index] = float4(0.0f);
    return;
  }
  const uint motion_base = index * 8u;
  const float scale = float(parameters.divisor);
  const float2 source_a =
      float2(position) + float2(float(motion[motion_base]),
                                float(motion[motion_base + 1u])) * scale;
  const float2 source_b =
      float2(position) + float2(float(motion[motion_base + 2u]),
                                float(motion[motion_base + 3u])) * scale;
  const ushort logit_bits = as_type<ushort>(motion[motion_base + 4u]);
  const float blend = as_type<float>(sigmoid_lut[uint(logit_bits)]);
  bool covered = subrect_fully_covered(
      mask, int(position.x), int(position.y), parameters.width,
      parameters.height);
  if (position.y + 1u == parameters.height &&
      (position.x == 0u || position.x + 1u == parameters.width))
    covered = false;
  float3 result = float3(0.0f);
  if (!covered) {
    const float3 a = subrect_sample_bilinear(
        color_a, source_a, parameters.width, parameters.height);
    const float3 b = subrect_sample_bilinear(
        color_b, source_b, parameters.width, parameters.height);
    result = saturate(fma(float3(blend), a,
                          float3(1.0f - blend) * b));
  }
  pixel_scratch[index] = float4(result, covered ? 0.0f : 1.0f);
  output_mask.write(float4(covered ? 1.0f : 0.0f), position);
  const uint2 destination =
      position + uint2(parameters.output_origin_x, parameters.output_origin_y);
  if (destination.x < parameters.resource_width &&
      destination.y < parameters.resource_height) {
    output_color.write(
        float4(subrect_selected_unorm8(result.x),
               subrect_selected_unorm8(result.y),
               subrect_selected_unorm8(result.z), 1.0f), destination);
  }
}

inline float4 subrect_load_pixel(device const float4* pixels, uint2 p,
                                 uint width) {
  return pixels[p.y * width + p.x];
}

inline float4 subrect_sum2(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = subrect_load_pixel(pixels, p, width) +
                      subrect_load_pixel(pixels, p + uint2(1, 0), width);
  const float4 row1 = subrect_load_pixel(pixels, p + uint2(0, 1), width) +
                      subrect_load_pixel(pixels, p + uint2(1, 1), width);
  return row0 + row1;
}

inline float4 subrect_sum4(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = subrect_sum2(pixels, p, width) +
                      subrect_sum2(pixels, p + uint2(2, 0), width);
  const float4 row1 = subrect_sum2(pixels, p + uint2(0, 2), width) +
                      subrect_sum2(pixels, p + uint2(2, 2), width);
  return row0 + row1;
}

inline float4 subrect_sum8(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = subrect_sum4(pixels, p, width) +
                      subrect_sum4(pixels, p + uint2(4, 0), width);
  const float4 row1 = subrect_sum4(pixels, p + uint2(0, 4), width) +
                      subrect_sum4(pixels, p + uint2(4, 4), width);
  return row0 + row1;
}

inline float4 subrect_sum16(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = subrect_sum8(pixels, p, width) +
                      subrect_sum8(pixels, p + uint2(8, 0), width);
  const float4 row1 = subrect_sum8(pixels, p + uint2(0, 8), width) +
                      subrect_sum8(pixels, p + uint2(8, 8), width);
  return row0 + row1;
}

inline half4 subrect_normalize_sum(
    float4 sum, float count, device const uchar* correction_rom) {
  const float denominator =
      max(sum.w, as_type<float>(uint(0x2b8cbcccu)));
  const float reciprocal = subrect_selected_sm120_rcp(
      denominator, correction_rom);
  const float count_reciprocal =
      subrect_selected_sm120_rcp(count, correction_rom);
  return half4(half3(sum.xyz * reciprocal), half(sum.w * count_reciprocal));
}

kernel void fg_subrect_postprocess_l84_level0(
    device const float4* pixels [[buffer(0)]],
    device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant SubrectL84ReductionParameters& parameters [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) return;
  output[position.y * parameters.output_width + position.x] =
      subrect_normalize_sum(
          subrect_sum2(pixels, position * 2u, parameters.input_width),
          4.0f, correction_rom);
}

kernel void fg_subrect_postprocess_l84_level1(
    device const float4* pixels [[buffer(0)]], device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant SubrectL84ReductionParameters& parameters [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) return;
  output[position.y * parameters.output_width + position.x] =
      subrect_normalize_sum(
          subrect_sum4(pixels, position * 4u, parameters.input_width),
          16.0f, correction_rom);
}

kernel void fg_subrect_postprocess_l84_level2(
    device const float4* pixels [[buffer(0)]], device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant SubrectL84ReductionParameters& parameters [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) return;
  output[position.y * parameters.output_width + position.x] =
      subrect_normalize_sum(
          subrect_sum8(pixels, position * 8u, parameters.input_width),
          64.0f, correction_rom);
}

kernel void fg_subrect_postprocess_l84_level3(
    device const float4* pixels [[buffer(0)]], device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant SubrectL84ReductionParameters& parameters [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) return;
  output[position.y * parameters.output_width + position.x] =
      subrect_normalize_sum(
          subrect_sum16(pixels, position * 16u, parameters.input_width),
          256.0f, correction_rom);
}
