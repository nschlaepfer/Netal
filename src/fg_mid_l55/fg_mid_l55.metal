#include <metal_stdlib>

using namespace metal;

struct L55Parameters {
  uint physical_width;
  uint physical_height;
  uint accumulation_mode;
  uint candidate_width;
  uint candidate_height;
  uint secondary_width;
  uint secondary_height;
  uint active_width;
  uint active_height;
  uint optional_ui_enabled;
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

inline float selected_div_approx(float numerator, float denominator,
                                 device const uchar* correction_rom) {
  return numerator * selected_sm120_rcp(denominator, correction_rom);
}

inline float half_bits_to_float(ushort bits) {
  return float(as_type<half>(bits));
}

inline float2 decode_candidate(ulong packed) {
  return float2(half_bits_to_float(ushort((packed >> 16u) & 0xffffu)),
                half_bits_to_float(ushort(packed & 0xffffu)));
}

inline float4 selected_point_sample(
    texture2d<float, access::sample> source, float2 coordinate) {
  const float2 dimensions(float(source.get_width()), float(source.get_height()));
  const float2 fixed12 = floor(coordinate * 4096.0f) * (1.0f / 4096.0f);
  const int2 location = int2(floor(fixed12 * dimensions));
  if (location.x < 0 || location.y < 0 ||
      location.x >= int(source.get_width()) ||
      location.y >= int(source.get_height())) {
    return float4(0.0f);
  }
  return source.read(uint2(location));
}

kernel void fg_mid_l55_exact(
    device const ulong* candidate_a [[buffer(0)]],
    device const ulong* candidate_b [[buffer(1)]],
    device half4* linear_a [[buffer(2)]],
    device half4* linear_b [[buffer(3)]],
    device half* linear_mask [[buffer(4)]],
    device const uchar* correction_rom [[buffer(5)]],
    constant L55Parameters& p [[buffer(6)]],
    texture2d<float, access::sample> source_a [[texture(0)]],
    texture2d<float, access::sample> source_b [[texture(1)]],
    texture2d<float, access::sample> motion [[texture(2)]],
    texture2d<float, access::sample> blend [[texture(3)]],
    texture2d<float, access::write> surface_a [[texture(4)]],
    texture2d<float, access::write> surface_b [[texture(5)]],
    texture2d<float, access::write> surface_mask [[texture(6)]],
    uint2 position [[thread_position_in_grid]]) {
  if (p.physical_width != 640u || p.physical_height != 368u ||
      p.accumulation_mode != 1u || p.candidate_width != 320u ||
      p.candidate_height != 180u || p.secondary_width != 320u ||
      p.secondary_height != 180u || p.active_width != 640u ||
      p.active_height != 360u || p.optional_ui_enabled != 0u ||
      position.x >= p.physical_width || position.y >= p.physical_height) {
    return;
  }

  float4 output_a = float4(0.0f);
  float4 output_b = float4(0.0f);
  bool invalid_mask = false;
  if (position.x < p.active_width && position.y < p.active_height) {
    const float u = clamp(selected_div_approx(float(position.x) + 0.5f,
                                               float(p.active_width),
                                               correction_rom), 0.0f, 1.0f);
    const float v = clamp(selected_div_approx(float(position.y) + 0.5f,
                                               float(p.active_height),
                                               correction_rom), 0.0f, 1.0f);
    const float candidate_x =
        fma(u, float(p.candidate_width), -0.5f) -
        as_type<float>(uint(0x34000000u));
    const float candidate_y =
        fma(v, float(p.candidate_height), -0.5f) -
        as_type<float>(uint(0x34000000u));
    const int ix = int(rint(candidate_x));
    const int iy = int(rint(candidate_y));
    const uint candidate_index =
        uint(iy) * p.candidate_width + uint(ix);
    const ulong packed_a = candidate_a[candidate_index];
    const ulong packed_b = candidate_b[candidate_index];
    const bool invalid_a = packed_a == 0x7fffffff7ffffffful;
    const bool invalid_b = packed_b == 0x7fffffff7ffffffful;
    const float2 offset_a = invalid_a ? float2(0.0f)
                                      : decode_candidate(packed_a);
    const float2 offset_b = invalid_b ? float2(0.0f)
                                      : decode_candidate(packed_b);
    const float2 uv(u, v);

    const float4 base_a = source_a.read(position);
    const float4 moved_a = selected_point_sample(source_a, uv + offset_b);
    const float4 base_b = source_b.read(position);
    const float4 moved_b = selected_point_sample(source_b, uv + offset_a);
    const float3 resolved_a =
        fma(float(invalid_b), moved_b.xyz - moved_a.xyz, moved_a.xyz);
    const float3 resolved_b =
        fma(float(invalid_a), moved_a.xyz - moved_b.xyz, moved_b.xyz);
    const float2 weights = selected_point_sample(blend, uv).xy;
    invalid_mask = invalid_a && invalid_b &&
                   (weights.x >= 0.5f || weights.y >= 0.5f);
    const float2 motion_base = selected_point_sample(motion, uv).xy;
    const float2 motion_a = selected_point_sample(motion, uv + offset_b).xy;
    const float2 motion_b = selected_point_sample(motion, uv + offset_a).xy;

    output_a = float4(
        fma(weights.x, resolved_a.x - base_a.x, base_a.x),
        fma(weights.x, resolved_a.y - base_a.y, base_a.y),
        fma(weights.x, resolved_a.z - base_a.z, base_a.z),
        fma(weights.x, motion_a.x - motion_base.x, motion_base.x));
    output_b = float4(
        fma(weights.y, resolved_b.x - base_b.x, base_b.x),
        fma(weights.y, resolved_b.y - base_b.y, base_b.y),
        fma(weights.y, resolved_b.z - base_b.z, base_b.z),
        fma(weights.y, motion_b.y - motion_base.y, motion_base.y));
    surface_a.write(output_a, position);
    surface_b.write(output_b, position);
    surface_mask.write(float4(invalid_mask ? 1.0f : 0.0f), position);
  }

  const uint linear_index = position.y * p.physical_width + position.x;
  linear_a[linear_index] = half4(
      selected_div_approx(output_a.x, 1.0f, correction_rom),
      selected_div_approx(output_a.y, 1.0f, correction_rom),
      selected_div_approx(output_a.z, 1.0f, correction_rom),
      selected_div_approx(output_a.w, 1.0f, correction_rom));
  linear_b[linear_index] = half4(
      selected_div_approx(output_b.x, 1.0f, correction_rom),
      selected_div_approx(output_b.y, 1.0f, correction_rom),
      selected_div_approx(output_b.z, 1.0f, correction_rom),
      selected_div_approx(output_b.w, 1.0f, correction_rom));
  linear_mask[linear_index] = half(invalid_mask ? 1 : 0);
}
