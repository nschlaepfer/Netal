#include <metal_stdlib>

using namespace metal;

struct EarlyScatterParameters {
  uint width;
  uint height;
  uint mode;
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

inline float2 selected_point_sample(
    texture2d<half, access::read> input, float2 coordinate,
    uint2 dimensions) {
  if (!(coordinate.x >= 0.0f && coordinate.x < 1.0f &&
        coordinate.y >= 0.0f && coordinate.y < 1.0f)) {
    return float2(0.0f);
  }
  const float2 fixed12 = floor(coordinate * 4096.0f) * (1.0f / 4096.0f);
  const int2 location = int2(floor(fixed12 * float2(dimensions)));
  if (any(location < int2(0)) || any(location >= int2(dimensions))) {
    return float2(0.0f);
  }
  return float2(input.read(uint2(location)).xy);
}

inline void scatter_zero(texture2d<half, access::write> output,
                         int2 location, uint2 dimensions) {
  if (any(location < int2(0)) || any(location >= int2(dimensions))) return;
  output.write(half4(0.0h), uint2(location));
}

kernel void fg_early_l8_l18(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    device const uchar* correction_rom [[buffer(0)]],
    constant EarlyScatterParameters& parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint2 dimensions(parameters.width, parameters.height);
  if (any(position >= dimensions)) return;
  const float2 center = float2(position) + 0.5f;
  const float2 uv = float2(
      center.x * selected_sm120_rcp(float(parameters.width), correction_rom),
      center.y * selected_sm120_rcp(float(parameters.height), correction_rom));
  float2 flow = selected_point_sample(input, saturate(uv), dimensions);
  if (parameters.mode == 0u) {
    flow.x = flow.x * selected_sm120_rcp(float(parameters.width),
                                         correction_rom);
    flow.y = flow.y * selected_sm120_rcp(float(parameters.height),
                                         correction_rom);
  }
  flow *= float2(dimensions);
  const int2 base = int2(position) + int2(flow);
  const int2 step(flow.x > 0.0f ? 1 : -1,
                  flow.y > 0.0f ? 1 : -1);
  scatter_zero(output, base, dimensions);
  scatter_zero(output, base + int2(0, step.y), dimensions);
  scatter_zero(output, base + int2(step.x, 0), dimensions);
  scatter_zero(output, base + step, dimensions);
}
