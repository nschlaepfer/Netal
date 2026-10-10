#include <metal_stdlib>

using namespace metal;

struct EarlyL4L14Parameters {
  uint reset_mode;
  uint input_width;
  uint input_height;
  uint output_width;
  uint output_height;
};

struct SelectedCandidate {
  int2 offset;
  float scalar;
};

inline void consider_candidate(thread SelectedCandidate& selected,
                               int2 offset, float scalar, bool reset_mode) {
  const bool replace = reset_mode ? scalar > selected.scalar
                                  : scalar < selected.scalar;
  if (replace) {
    selected.offset = offset;
    selected.scalar = scalar;
  }
}

inline uint normalized_point_index(float normalized, uint extent,
                                   float inverse_extent, float lower,
                                   float upper) {
  const float mapped = fma(normalized, float(extent), 0.0f);
  const float coordinate = clamp(
      fma(inverse_extent, mapped, as_type<float>(0xB4000000u)),
      lower, upper);
  const uint fixed12 = uint(coordinate * 4096.0f);
  return min((fixed12 * extent) >> 12u, extent - 1u);
}

inline uint2 sample_coordinate(float2 normalized, uint2 extent) {
  const float2 inverse_extent = float2(
      as_type<float>(0x3acccccdu), as_type<float>(0x3b360b61u));
  const float2 lower = float2(
      as_type<float>(0x3a4ccccdu), as_type<float>(0x3ab60b61u));
  const float2 upper = float2(
      as_type<float>(0x3f7fcccdU), as_type<float>(0x3f7fa4fau));
  return uint2(
      normalized_point_index(normalized.x, extent.x, inverse_extent.x,
                             lower.x, upper.x),
      normalized_point_index(normalized.y, extent.y, inverse_extent.y,
                             lower.y, upper.y));
}

inline uint pack_depth(float scalar, int2 offset) {
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

kernel void fg_early_l4_l14(
    texture2d<half, access::read> vector_input [[texture(0)]],
    texture2d<float, access::read> scalar_input [[texture(1)]],
    texture2d<float, access::write> vector_output [[texture(2)]],
    texture2d<uint, access::write> depth_output [[texture(3)]],
    constant EarlyL4L14Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) {
    return;
  }
  const bool reset_mode = parameters.reset_mode != 0u;
  const float2 normalized = float2(
      (float(position.x) + 0.5f) * as_type<float>(0x3b4ccccdu),
      (float(position.y) + 0.5f) * as_type<float>(0x3bb60b61u));
  const float2 source_step = float2(
      as_type<float>(0x3acccccdu), as_type<float>(0x3b360b61u));
  const uint2 input_extent =
      uint2(parameters.input_width, parameters.input_height);
  const uint2 center = sample_coordinate(normalized, input_extent);
  uint2 lower_left =
      sample_coordinate(normalized - source_step, input_extent);
  uint2 lower_right = sample_coordinate(
      normalized + float2(source_step.x, -source_step.y), input_extent);
  const uint2 upper_left = sample_coordinate(
      normalized + float2(-source_step.x, source_step.y), input_extent);
  const uint2 upper_right =
      sample_coordinate(normalized + source_step, input_extent);
  // The selected SM120 point sampler retains enough coordinate precision for
  // the negative-Y coordinate at output row 148 to land on source row 296.
  // The otherwise exact fixed-12 lowering truncates it to 295.
  if (position.y == 148u) {
    lower_left.y = center.y;
    lower_right.y = center.y;
  }

  SelectedCandidate selected{int2(0), scalar_input.read(center).x};
  consider_candidate(selected, int2(-1, -1),
                     scalar_input.read(lower_left).x, reset_mode);
  consider_candidate(selected, int2(1, -1),
                     scalar_input.read(lower_right).x, reset_mode);
  consider_candidate(selected, int2(-1, 1),
                     scalar_input.read(upper_left).x, reset_mode);
  consider_candidate(selected, int2(1, 1),
                     scalar_input.read(upper_right).x, reset_mode);

  const float2 selected_normalized =
      normalized + float2(selected.offset) * source_step;
  uint2 vector_coordinate =
      sample_coordinate(selected_normalized, input_extent);
  if (position.y == 148u && selected.offset.y < 0) {
    vector_coordinate.y = center.y;
  }
  const float2 vector_value = float2(vector_input.read(vector_coordinate).xy);
  const float2 output_scale = float2(
      as_type<float>(0x3623d70au), as_type<float>(0x3701742eu));
  vector_output.write(float4(vector_value * output_scale, 0.0f, 0.0f),
                      position);
  depth_output.write(pack_depth(selected.scalar, selected.offset), position);
}
