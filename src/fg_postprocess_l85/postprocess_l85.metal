#include <metal_stdlib>

using namespace metal;

struct PostprocessL85Parameters {
  uint width;
  uint height;
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

inline float4 reduce_tile(device const half4* input,
                          constant PostprocessL85Parameters& parameters,
                          uint2 output_position) {
  float4 values[4] = {float4(0.0f), float4(0.0f), float4(0.0f),
                      float4(0.0f)};
  for (uint lane = 0; lane < 4u; ++lane) {
    const uint2 input_position =
        output_position * 2u + uint2(lane & 1u, lane >> 1u);
    if (input_position.x < parameters.width &&
        input_position.y < parameters.height) {
      values[lane] =
          float4(input[input_position.y * parameters.width + input_position.x]);
    }
  }
  const float3 product0 = values[0].xyz * values[0].w;
  const float3 product1 = values[1].xyz * values[1].w;
  const float3 product2 = values[2].xyz * values[2].w;
  const float3 product3 = values[3].xyz * values[3].w;
  const float3 numerator = (product0 + product1) + (product2 + product3);
  const float alpha =
      (values[0].w + values[1].w) + (values[2].w + values[3].w);
  return float4(numerator, alpha);
}

kernel void fg_postprocess_l85_exact(
    device const half4* input [[buffer(0)]],
    device half4* level0 [[buffer(1)]],
    device half4* level1 [[buffer(2)]],
    device half4* level2 [[buffer(3)]],
    device half4* level3 [[buffer(4)]],
    device const uchar* correction_rom [[buffer(5)]],
    constant PostprocessL85Parameters& parameters [[buffer(6)]],
    uint2 output_position [[thread_position_in_grid]]) {
  (void)level1;
  (void)level2;
  (void)level3;
  const uint output_width = (parameters.width + 1u) / 2u;
  const uint output_height = (parameters.height + 1u) / 2u;
  if (output_position.x >= output_width || output_position.y >= output_height) {
    return;
  }
  const float4 sums = reduce_tile(input, parameters, output_position);
  const float denominator =
      max(sums.w, as_type<float>(uint(0x2B8CBCCCu)));
  const float4 result(
      selected_div_approx(sums.x, denominator, correction_rom),
      selected_div_approx(sums.y, denominator, correction_rom),
      selected_div_approx(sums.z, denominator, correction_rom),
      selected_div_approx(sums.w, 4.0f, correction_rom));
  level0[output_position.y * output_width + output_position.x] = half4(result);
}

// Deliberate negative control. The selected PTX uses div.approx.ftz.f32;
// ordinary precise Metal division changes accepted half outputs on the frozen
// input domain and must never replace fg_postprocess_l85_exact.
kernel void fg_postprocess_l85_strict_division(
    device const half4* input [[buffer(0)]],
    device half4* level0 [[buffer(1)]],
    constant PostprocessL85Parameters& parameters [[buffer(2)]],
    uint2 output_position [[thread_position_in_grid]]) {
  const uint output_width = (parameters.width + 1u) / 2u;
  const uint output_height = (parameters.height + 1u) / 2u;
  if (output_position.x >= output_width || output_position.y >= output_height) {
    return;
  }
  const float4 sums = reduce_tile(input, parameters, output_position);
  const float denominator =
      max(sums.w, as_type<float>(uint(0x2B8CBCCCu)));
  const float4 result(sums.x / denominator, sums.y / denominator,
                      sums.z / denominator, sums.w / 4.0f);
  level0[output_position.y * output_width + output_position.x] = half4(result);
}
