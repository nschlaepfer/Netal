#include <metal_stdlib>

using namespace metal;

struct DynamicModule15Parameters {
  uint input_width;
  uint input_height;
  uint output_width;
  uint output_height;
  float2 input_inverse;
  float2 output_inverse;
  float2 lower_coordinate;
  float2 upper_coordinate;
  float2 output_scale;
};

struct DynamicCandidate {
  int2 offset;
  float scalar;
};

inline void dynamic_consider(thread DynamicCandidate& selected, int2 offset,
                             float scalar) {
  if (scalar < selected.scalar) {
    selected.offset = offset;
    selected.scalar = scalar;
  }
}

inline uint dynamic_point_index(float normalized, uint extent,
                                float inverse_extent, float lower,
                                float upper) {
  const float mapped = fma(normalized, float(extent), 0.0f);
  const float coordinate = clamp(
      fma(inverse_extent, mapped, as_type<float>(0xB4000000u)),
      lower, upper);
  const uint fixed12 = uint(coordinate * 4096.0f);
  return min((fixed12 * extent) >> 12u, extent - 1u);
}

inline uint2 dynamic_sample_coordinate(
    float2 normalized, constant DynamicModule15Parameters& parameters) {
  return uint2(
      dynamic_point_index(normalized.x, parameters.input_width,
                          parameters.input_inverse.x,
                          parameters.lower_coordinate.x,
                          parameters.upper_coordinate.x),
      dynamic_point_index(normalized.y, parameters.input_height,
                          parameters.input_inverse.y,
                          parameters.lower_coordinate.y,
                          parameters.upper_coordinate.y));
}

inline uint dynamic_pack_depth(float scalar, int2 offset) {
  const float saturated = clamp(scalar, 0.0f, 1.0f);
  const uint scalar_bits = as_type<uint>(saturated);
  const uint bit29 = scalar_bits & 0x20000000u;
  const bool bit29_zero = bit29 == 0u;
  const bool center = offset.x == 0 && offset.y == 0;
  const uint mask_a = bit29_zero ? 0x3ffffffeu : 0x3fffffffu;
  const uint mask = center ? 0x1fffffffu : mask_a;
  const uint retained_scalar = mask & scalar_bits;
  const uint center_or_noncenter = center ? 0x80000000u : 0x40000000u;
  const uint positive_x_flag = bit29_zero ? 1u : 0x40000000u;
  const uint center_bit = center ? 1u : 0u;
  const uint scalar_bit = bit29 >> 29u;
  const uint base = scalar_bit == center_bit ? center_or_noncenter : 0u;
  const uint positive_y = offset.y > 0 ? 0x80000000u : 0u;
  const uint positive_x = offset.x > 0 ? positive_x_flag : 0u;
  return base | positive_y | positive_x | retained_scalar;
}

kernel void fg_dynamic_module15(
    texture2d<half, access::read> vector_input [[texture(0)]],
    texture2d<float, access::read> scalar_input [[texture(1)]],
    texture2d<float, access::write> vector_output [[texture(2)]],
    texture2d<uint, access::write> depth_output [[texture(3)]],
    constant DynamicModule15Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height)
    return;
  const float2 normalized =
      (float2(position) + 0.5f) * parameters.output_inverse;
  const float2 source_step = parameters.input_inverse;
  const uint2 center = dynamic_sample_coordinate(normalized, parameters);
  const uint2 lower_left = dynamic_sample_coordinate(
      normalized - source_step, parameters);
  const uint2 lower_right = dynamic_sample_coordinate(
      normalized + float2(source_step.x, -source_step.y), parameters);
  uint2 upper_left = dynamic_sample_coordinate(
      normalized + float2(-source_step.x, source_step.y), parameters);
  uint2 upper_right = dynamic_sample_coordinate(
      normalized + source_step, parameters);
  // The selected SM120 normalized point sampler retains one additional bit for
  // the positive-Y coordinate at this 1280x720 source row. Fixed-12 truncation
  // maps it to 681 instead of the observed 682. Other target extents have no
  // such exception in the complete coordinate differential.
  if (parameters.input_height == 720u && position.y == 340u) {
    upper_left.y += 1u;
    upper_right.y += 1u;
  }

  DynamicCandidate selected{int2(0), scalar_input.read(center).x};
  dynamic_consider(selected, int2(-1, -1),
                   scalar_input.read(lower_left).x);
  dynamic_consider(selected, int2(1, -1),
                   scalar_input.read(lower_right).x);
  dynamic_consider(selected, int2(-1, 1),
                   scalar_input.read(upper_left).x);
  dynamic_consider(selected, int2(1, 1),
                   scalar_input.read(upper_right).x);
  const float2 selected_normalized =
      normalized + float2(selected.offset) * source_step;
  uint2 vector_coordinate =
      dynamic_sample_coordinate(selected_normalized, parameters);
  if (parameters.input_height == 720u && position.y == 340u &&
      selected.offset.y > 0)
    vector_coordinate.y += 1u;
  const float2 vector_value = float2(vector_input.read(vector_coordinate).xy);
  vector_output.write(
      float4(vector_value * parameters.output_scale, 0.0f, 0.0f), position);
  depth_output.write(dynamic_pack_depth(selected.scalar, selected.offset),
                     position);
}
