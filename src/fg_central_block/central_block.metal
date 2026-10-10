#include <metal_stdlib>

using namespace metal;

struct CentralBlockParameters {
  uint width;
  uint height;
  float factor;
};

inline half4 selected_bilinear(
    device const half4* source,
    uint width,
    uint height,
    half coordinate_x,
    half coordinate_y) {
  // cvt.rmi.s32.f16 is floor, including for negative coordinates.  Convert
  // the already-rounded half coordinate exactly to float, then floor before
  // applying the signed clamp used by the selected PTX.
  const int raw_x = int(floor(float(coordinate_x)));
  const int raw_y = int(floor(float(coordinate_y)));
  const half fraction_x = coordinate_x - half(raw_x);
  const half fraction_y = coordinate_y - half(raw_y);
  const int maximum_x = int(width) - 1;
  const int maximum_y = int(height) - 1;
  const uint x0 = uint(clamp(raw_x, 0, maximum_x));
  const uint x1 = uint(clamp(raw_x + 1, 0, maximum_x));
  const uint y0 = uint(clamp(raw_y, 0, maximum_y));
  const uint y1 = uint(clamp(raw_y + 1, 0, maximum_y));

  const half4 sample00 = source[y0 * width + x0];
  const half4 sample10 = source[y0 * width + x1];
  const half4 sample01 = source[y1 * width + x0];
  const half4 sample11 = source[y1 * width + x1];
  const half one = half(1.0f);
  const half one_minus_x = one - fraction_x;
  const half one_minus_y = one - fraction_y;

  // The SM120 JIT contracts the raw PTX's accumulation sequence in one
  // specific direction: round w10*s10 first, then fuse w00*s00 + that term,
  // followed by fused w01 and w11 accumulations.  Spell those contractions
  // explicitly while leaving every weight operation separately rounded.
  const half weight00 = one_minus_x * one_minus_y;
  const half weight10 = fraction_x * one_minus_y;
  const half4 term10 = half4(weight10) * sample10;
  const half4 sum0 = fma(half4(weight00), sample00, term10);
  const half weight01 = one_minus_x * fraction_y;
  const half4 sum1 = fma(half4(weight01), sample01, sum0);
  const half weight11 = fraction_x * fraction_y;
  return fma(half4(weight11), sample11, sum1);
}

// Dependency-closed semantic form of launch 71.  NVIDIA's selected
// k_central_block stages the same nine packed u32 values per pixel through a
// 9,216-byte threadgroup array solely to coalesce global stores; the sibling
// k_central_block_original entry writes this semantic form directly.
kernel void fg_central_block(
    device uint* output_words [[buffer(0)]],
    device const half4* source_a [[buffer(1)]],
    device const half4* source_b [[buffer(2)]],
    device const half* scalar_a [[buffer(3)]],
    device const half4* motion [[buffer(4)]],
    device const half* scalar_b [[buffer(5)]],
    device const half* triple [[buffer(6)]],
    constant CentralBlockParameters& parameters [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.width || position.y >= parameters.height) {
    return;
  }

  const uint pixel = position.y * parameters.width + position.x;
  const half4 flow = motion[pixel];
  const half two = half(2.0f);
  const half coordinate_a_x = fma(flow.x, two, half(position.x));
  const half coordinate_a_y = fma(flow.y, two, half(position.y));
  const half coordinate_b_x = fma(flow.z, two, half(position.x));
  const half coordinate_b_y = fma(flow.w, two, half(position.y));

  const half4 sampled_a = selected_bilinear(
      source_a, parameters.width, parameters.height,
      coordinate_a_x, coordinate_a_y);
  const half4 sampled_b = selected_bilinear(
      source_b, parameters.width, parameters.height,
      coordinate_b_x, coordinate_b_y);

  const uint output = pixel * 9u;
  output_words[output + 0u] = as_type<uint>(sampled_a.xy);
  output_words[output + 1u] = as_type<uint>(sampled_a.zw);
  output_words[output + 2u] = as_type<uint>(sampled_b.xy);
  output_words[output + 3u] = as_type<uint>(sampled_b.zw);
  output_words[output + 4u] = as_type<uint>(flow.xy);
  output_words[output + 5u] = as_type<uint>(flow.zw);
  output_words[output + 6u] =
      as_type<uint>(half2(scalar_b[pixel], scalar_a[pixel]));
  output_words[output + 7u] =
      as_type<uint>(half2(triple[pixel * 3u], triple[pixel * 3u + 1u]));
  output_words[output + 8u] =
      as_type<uint>(half2(triple[pixel * 3u + 2u], half(parameters.factor)));
}
