#include <metal_stdlib>

using namespace metal;

struct PostprocessL84Parameters {
  uint width;
  uint height;
  uint padded_height;
  uint divisor;
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

inline bool mask_need(device const uchar* mask, int x, int y,
                      constant PostprocessL84Parameters& parameters) {
  if (x < 0 || y < 0 || x >= int(parameters.width) ||
      y >= int(parameters.height)) {
    return true;
  }
  return mask[uint(y) * parameters.width + uint(x)] != 0u;
}

inline bool fully_covered_neighborhood(
    device const uchar* mask, int x, int y,
    constant PostprocessL84Parameters& parameters) {
  const bool m00 = mask_need(mask, x - 2, y - 2, parameters);
  const bool m10 = mask_need(mask, x - 1, y - 2, parameters);
  const bool m20 = mask_need(mask, x + 0, y - 2, parameters);
  const bool m30 = mask_need(mask, x + 1, y - 2, parameters);
  const bool m40 = mask_need(mask, x + 2, y - 2, parameters);
  const bool m01 = mask_need(mask, x - 2, y - 1, parameters);
  const bool m11 = mask_need(mask, x - 1, y - 1, parameters);
  const bool m21 = mask_need(mask, x + 0, y - 1, parameters);
  const bool m31 = mask_need(mask, x + 1, y - 1, parameters);
  const bool m41 = mask_need(mask, x + 2, y - 1, parameters);
  const bool m02 = mask_need(mask, x - 2, y + 0, parameters);
  const bool m12 = mask_need(mask, x - 1, y + 0, parameters);
  const bool m22 = mask_need(mask, x + 0, y + 0, parameters);
  const bool m32 = mask_need(mask, x + 1, y + 0, parameters);
  const bool m42 = mask_need(mask, x + 2, y + 0, parameters);
  const bool m03 = mask_need(mask, x - 2, y + 1, parameters);
  const bool m13 = mask_need(mask, x - 1, y + 1, parameters);
  const bool m23 = mask_need(mask, x + 0, y + 1, parameters);
  const bool m33 = mask_need(mask, x + 1, y + 1, parameters);
  const bool m43 = mask_need(mask, x + 2, y + 1, parameters);
  const bool m04 = mask_need(mask, x - 2, y + 2, parameters);
  const bool m14 = mask_need(mask, x - 1, y + 2, parameters);
  const bool m24 = mask_need(mask, x + 0, y + 2, parameters);
  const bool m34 = mask_need(mask, x + 1, y + 2, parameters);
  const bool m44 = mask_need(mask, x + 2, y + 2, parameters);

  // This deliberately mirrors the nine predicate trees in selected SM120
  // module 17 rather than simplifying them into an inferred morphology.
  const bool top_left =
      m00 || m10 || m20 || m01 || m11 || m21 || m02 || m12 || m22 ||
      m03 || m13 || m23;
  const bool top_middle =
      m10 || m20 || m30 || m11 || m21 || m31 || m12 || m22 || m32;
  const bool top_right =
      m20 || m30 || m40 || m21 || m31 || m41 || m22 || m32 || m42 ||
      m23 || m33 || m43;
  const bool middle_left =
      m01 || m11 || m21 || m02 || m12 || m22 || m03 || m13 || m23;
  const bool middle =
      m11 || m21 || m31 || m12 || m22 || m32 || m13 || m23 || m33;
  const bool middle_right =
      m21 || m31 || m41 || m22 || m32 || m42 || m23 || m33 || m43;
  const bool bottom_left =
      m02 || m12 || m22 || m03 || m13 || m23 || m04 || m14 || m24;
  const bool bottom_middle =
      m12 || m22 || m32 || m13 || m23 || m33 || m14 || m24 || m34;
  const bool bottom_right =
      m22 || m32 || m42 || m23 || m33 || m43 || m24 || m34 || m44;
  return top_left && top_middle && top_right && middle_left && middle &&
         middle_right && bottom_left && bottom_middle && bottom_right;
}

inline float3 sample_bilinear_rgb(device const half4* color, float2 position,
                                  uint width, uint height) {
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
  const float w00 = one_minus_x * one_minus_y;
  const float w10 = fraction.x * one_minus_y;
  const float3 row0 = fma(float3(w00), c00, float3(w10) * c10);
  const float w01 = fraction.y * one_minus_x;
  const float3 with_row1_left = fma(float3(w01), c01, row0);
  const float w11 = fraction.x * fraction.y;
  return fma(float3(w11), c11, with_row1_left);
}

inline float selected_sust_p_unorm8(float value) {
  const float clamped = saturate(value);
  const uint fixed12 = uint(clamped * 4096.0f);
  const uint encoded = min(255u, (fixed12 * 255u + 2047u) >> 12u);
  // Feeding the already-selected code point back through Metal's UNORM store
  // is exact, while a direct float store uses Apple's different rounding.
  return float(encoded) * (1.0f / 255.0f);
}

kernel void fg_postprocess_l84_pixels(
    device const half4* color_a [[buffer(0)]],
    device const half4* color_b [[buffer(1)]],
    device const uchar* mask [[buffer(2)]],
    device const half* motion [[buffer(3)]],
    device const uint* sigmoid_lut [[buffer(4)]],
    device float4* pixel_scratch [[buffer(5)]],
    constant PostprocessL84Parameters& parameters [[buffer(6)]],
    texture2d<float, access::write> output_color [[texture(0)]],
    texture2d<float, access::write> output_mask [[texture(1)]],
    texture2d<float, access::write> native_metal_color [[texture(2)]],
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
  const float2 source_a_position =
      float2(position) + float2(float(motion[motion_base + 0u]),
                                float(motion[motion_base + 1u])) * scale;
  const float2 source_b_position =
      float2(position) + float2(float(motion[motion_base + 2u]),
                                float(motion[motion_base + 3u])) * scale;
  const ushort logit_bits = as_type<ushort>(motion[motion_base + 4u]);
  const float blend = as_type<float>(sigmoid_lut[uint(logit_bits)]);
  bool covered = fully_covered_neighborhood(
      mask, int(position.x), int(position.y), parameters);
  // The selected shared-halo schedule leaves the two bottom corner
  // predicates false at this exact 640x360 boundary. Preserve the observed
  // selected-domain result instead of replacing it with an inferred dilation.
  if (position.y + 1u == parameters.height &&
      (position.x == 0u || position.x + 1u == parameters.width)) {
    covered = false;
  }
  float3 result = float3(0.0f);
  if (!covered) {
    const float3 a = sample_bilinear_rgb(
        color_a, source_a_position, parameters.width, parameters.height);
    const float3 b = sample_bilinear_rgb(
        color_b, source_b_position, parameters.width, parameters.height);
    const float one_minus_blend = 1.0f - blend;
    result = saturate(fma(float3(blend), a, float3(one_minus_blend) * b));
  }
  const float4 generated(result, 1.0f);
  pixel_scratch[index] = float4(result, covered ? 0.0f : 1.0f);
  output_mask.write(float4(covered ? 1.0f : 0.0f), position);
  native_metal_color.write(generated, position);
  output_color.write(
      float4(selected_sust_p_unorm8(generated.x),
             selected_sust_p_unorm8(generated.y),
             selected_sust_p_unorm8(generated.z), 1.0f),
      position);
}

inline float4 load_pixel(device const float4* pixels, uint2 p, uint width) {
  return pixels[p.y * width + p.x];
}

inline float4 sum2(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = load_pixel(pixels, p, width) +
                      load_pixel(pixels, p + uint2(1u, 0u), width);
  const float4 row1 = load_pixel(pixels, p + uint2(0u, 1u), width) +
                      load_pixel(pixels, p + uint2(1u, 1u), width);
  return row0 + row1;
}

inline float4 sum4(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = sum2(pixels, p, width) +
                      sum2(pixels, p + uint2(2u, 0u), width);
  const float4 row1 = sum2(pixels, p + uint2(0u, 2u), width) +
                      sum2(pixels, p + uint2(2u, 2u), width);
  return row0 + row1;
}

inline float4 sum8(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = sum4(pixels, p, width) +
                      sum4(pixels, p + uint2(4u, 0u), width);
  const float4 row1 = sum4(pixels, p + uint2(0u, 4u), width) +
                      sum4(pixels, p + uint2(4u, 4u), width);
  return row0 + row1;
}

inline float4 sum16(device const float4* pixels, uint2 p, uint width) {
  const float4 row0 = sum8(pixels, p, width) +
                      sum8(pixels, p + uint2(8u, 0u), width);
  const float4 row1 = sum8(pixels, p + uint2(0u, 8u), width) +
                      sum8(pixels, p + uint2(8u, 8u), width);
  return row0 + row1;
}

inline half4 normalize_sum(float4 sum, float count,
                           device const uchar* correction_rom) {
  const float denominator =
      max(sum.w, as_type<float>(uint(0x2B8CBCCCu)));
  const float reciprocal = selected_sm120_rcp(denominator, correction_rom);
  const float count_reciprocal = selected_sm120_rcp(count, correction_rom);
  return half4(half3(sum.xyz * reciprocal),
               half(sum.w * count_reciprocal));
}

kernel void fg_postprocess_l84_level0(
    device const float4* pixels [[buffer(0)]],
    device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= 320u || position.y >= 180u) return;
  output[position.y * 320u + position.x] = normalize_sum(
      sum2(pixels, position * 2u, 640u), 4.0f, correction_rom);
}

kernel void fg_postprocess_l84_level1(
    device const float4* pixels [[buffer(0)]],
    device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= 160u || position.y >= 90u) return;
  output[position.y * 160u + position.x] = normalize_sum(
      sum4(pixels, position * 4u, 640u), 16.0f, correction_rom);
}

kernel void fg_postprocess_l84_level2(
    device const float4* pixels [[buffer(0)]],
    device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= 80u || position.y >= 45u) return;
  output[position.y * 80u + position.x] = normalize_sum(
      sum8(pixels, position * 8u, 640u), 64.0f, correction_rom);
}

kernel void fg_postprocess_l84_level3(
    device const float4* pixels [[buffer(0)]],
    device half4* output [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= 40u || position.y >= 23u) return;
  output[position.y * 40u + position.x] = normalize_sum(
      sum16(pixels, position * 16u, 640u), 256.0f, correction_rom);
}
