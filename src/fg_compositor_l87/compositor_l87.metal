#include <metal_stdlib>

using namespace metal;

struct CompositorL87Parameters {
  uint level;
  uint width;
  uint height;
  uint reserved;
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

inline float4 load_selected_level(device const half4* level0,
                                  uint2 position,
                                  constant CompositorL87Parameters& p) {
  const uint level_width = (p.width + 1u) >> 1u;
  const uint level_height = (p.height + 1u) >> 1u;
  if (position.x >= level_width || position.y >= level_height) {
    return float4(0.0f);
  }
  return float4(level0[position.y * level_width + position.x]);
}

inline float4 premultiply(float4 value) {
  return float4(value.xyz * value.w, value.w);
}

inline float4 selected_reconstruction(device const half4* level0,
                                      device const uchar* correction_rom,
                                      uint2 output_position,
                                      constant CompositorL87Parameters& p) {
  const uint2 base = output_position >> 1u;
  const int x_delta = (output_position.x & 1u) != 0u ? 1 : -1;
  const int y_delta = (output_position.y & 1u) != 0u ? 1 : -1;
  // The selected level is 10. The PTX clips its neighbor to [0, 2^10-1]
  // before the real 320x180 bounds test. This intentionally duplicates the
  // low edge while treating the high halo as zero.
  const int coordinate_limit = (1 << int(p.level)) - 1;
  const uint neighbor_x = uint(clamp(int(base.x) + x_delta, 0,
                                     coordinate_limit));
  const uint neighbor_y = uint(clamp(int(base.y) + y_delta, 0,
                                     coordinate_limit));
  const float4 v00 = premultiply(load_selected_level(level0, base, p));
  const float4 v10 =
      premultiply(load_selected_level(level0, uint2(neighbor_x, base.y), p));
  const float4 v01 =
      premultiply(load_selected_level(level0, uint2(base.x, neighbor_y), p));
  const float4 v11 = premultiply(
      load_selected_level(level0, uint2(neighbor_x, neighbor_y), p));

  float4 accumulated = fma(v00, float4(0.5625f), float4(0.0f));
  accumulated = fma(v10, float4(0.1875f), accumulated);
  accumulated = fma(v01, float4(0.1875f), accumulated);
  accumulated = fma(v11, float4(0.0625f), accumulated);
  const float denominator =
      clamp(accumulated.w, as_type<float>(uint(0x2b8cbcccu)), 1.0f);
  return float4(
      selected_div_approx(accumulated.x, denominator, correction_rom),
      selected_div_approx(accumulated.y, denominator, correction_rom),
      selected_div_approx(accumulated.z, denominator, correction_rom), 1.0f);
}

inline float selected_unorm8(float value) {
  // The selected SUST.P path first truncates UNORM input to 12 fractional
  // bits, then scales to 8 bits with nearest/tie-down rounding. The 2047 bias
  // is therefore deliberate (2048 would round its sole half tie upward).
  const uint fixed12 =
      uint(floor(clamp(value, 0.0f, 1.0f) * 4096.0f));
  const uint encoded = min(255u, (fixed12 * 255u + 2047u) >> 12u);
  return float(encoded) * (1.0f / 255.0f);
}

inline uint selected_pack_candidate(float4 value) {
  return pack_float_to_unorm4x8(value);
}

kernel void fg_compositor_l87_exact(
    device const half4* level0 [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    constant CompositorL87Parameters& parameters [[buffer(2)]],
    texture2d<float, access::read> mask [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 output_position [[thread_position_in_grid]]) {
  if (parameters.level != 10u || parameters.reserved != 0u ||
      output_position.x >= parameters.width ||
      output_position.y >= parameters.height) {
    return;
  }
  if (!(mask.read(output_position).x > 0.0f)) {
    return;
  }
  const float4 reconstructed = selected_reconstruction(
      level0, correction_rom, output_position, parameters);
  output.write(float4(selected_unorm8(reconstructed.x),
                      selected_unorm8(reconstructed.y),
                      selected_unorm8(reconstructed.z), 1.0f),
               output_position);
}

kernel void fg_compositor_l87_float_diagnostic(
    device const half4* level0 [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    constant CompositorL87Parameters& parameters [[buffer(2)]],
    texture2d<float, access::read> mask [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 output_position [[thread_position_in_grid]]) {
  if (parameters.level != 10u || parameters.reserved != 0u ||
      output_position.x >= parameters.width ||
      output_position.y >= parameters.height ||
      !(mask.read(output_position).x > 0.0f)) {
    return;
  }
  output.write(selected_reconstruction(level0, correction_rom,
                                       output_position, parameters),
               output_position);
}

kernel void fg_compositor_l87_packed_candidate(
    device const half4* level0 [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    constant CompositorL87Parameters& parameters [[buffer(2)]],
    device uint* packed_output [[buffer(3)]],
    texture2d<float, access::read> mask [[texture(0)]],
    uint2 output_position [[thread_position_in_grid]]) {
  if (parameters.level != 10u || parameters.reserved != 0u ||
      output_position.x >= parameters.width ||
      output_position.y >= parameters.height ||
      !(mask.read(output_position).x > 0.0f)) {
    return;
  }
  packed_output[output_position.y * parameters.width + output_position.x] =
      selected_pack_candidate(selected_reconstruction(
          level0, correction_rom, output_position, parameters));
}

// Retained as a numerical counterexample. It is never the accepted backend.
kernel void fg_compositor_l87_strict_division(
    device const half4* level0 [[buffer(0)]],
    constant CompositorL87Parameters& parameters [[buffer(1)]],
    texture2d<float, access::read> mask [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 output_position [[thread_position_in_grid]]) {
  if (parameters.level != 10u || parameters.reserved != 0u ||
      output_position.x >= parameters.width ||
      output_position.y >= parameters.height ||
      !(mask.read(output_position).x > 0.0f)) {
    return;
  }
  const uint2 base = output_position >> 1u;
  const int x_delta = (output_position.x & 1u) != 0u ? 1 : -1;
  const int y_delta = (output_position.y & 1u) != 0u ? 1 : -1;
  const int limit = (1 << int(parameters.level)) - 1;
  const uint nx = uint(clamp(int(base.x) + x_delta, 0, limit));
  const uint ny = uint(clamp(int(base.y) + y_delta, 0, limit));
  float4 sum = fma(premultiply(load_selected_level(level0, base, parameters)),
                   float4(0.5625f), float4(0.0f));
  sum = fma(premultiply(load_selected_level(level0, uint2(nx, base.y),
                                            parameters)),
            float4(0.1875f), sum);
  sum = fma(premultiply(load_selected_level(level0, uint2(base.x, ny),
                                            parameters)),
            float4(0.1875f), sum);
  sum = fma(premultiply(load_selected_level(level0, uint2(nx, ny), parameters)),
            float4(0.0625f), sum);
  const float denominator =
      clamp(sum.w, as_type<float>(uint(0x2b8cbcccu)), 1.0f);
  output.write(float4(sum.xyz / denominator, 1.0f), output_position);
}
