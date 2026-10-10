#include <metal_stdlib>

using namespace metal;

struct CustomUpsamplePlaneParameters {
  uint input_height;
  uint input_width;
  uint output_height;
  uint output_width;
  uint channels;
};

// Dependency-closed reconstruction of evaluate_1 launch 70 / module 60.
//
// The selected CUDA kernel uses ctaid.z to choose one of three independent
// NHWC half planes.  Metal dispatches this kernel once per plane instead.  The
// The PTX spells nine separate half operations, but direct selected-module
// execution proves that the RTX driver contracts the left product of both
// horizontal adds and of the final vertical add.  The right products remain
// separately rounded.  Explicit fma calls reproduce that observed lowering;
// the checked-in build disables any additional implicit contraction.
kernel void fg_custom_upsample_plane(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant CustomUpsamplePlaneParameters& parameters [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  const uint output_x = position.x;
  const uint output_y = position.y;
  const uint channel = position.z;
  if (output_x >= parameters.output_width ||
      output_y >= parameters.output_height ||
      channel >= parameters.channels) {
    return;
  }

  // Exact selected coordinate contract: (output - 0.5) * 0.5.  All live
  // dimensions are a strict 2x upscale, so no approximate division appears.
  const float source_x = (float(output_x) - 0.5f) * 0.5f;
  const float source_y = (float(output_y) - 0.5f) * 0.5f;
  const uint x0 = source_x < 0.0f
                      ? 0u
                      : min(uint(source_x), parameters.input_width - 1u);
  const uint x1 = source_x < 0.0f
                      ? 0u
                      : min(uint(source_x + 1.0f),
                            parameters.input_width - 1u);
  const uint y0 = source_y < 0.0f
                      ? 0u
                      : min(uint(source_y), parameters.input_height - 1u);
  const uint y1 = source_y < 0.0f
                      ? 0u
                      : min(uint(source_y + 1.0f),
                            parameters.input_height - 1u);

  const half weight_y1 = half(source_y - float(y0));
  const half weight_x1 = half(source_x - float(x0));
  const half weight_y0 = half((float(y0) - source_y) + 1.0f);
  const half weight_x0 = half((float(x0) - source_x) + 1.0f);

  const uint input_row_values = parameters.input_width * parameters.channels;
  const uint top_left =
      y0 * input_row_values + x0 * parameters.channels + channel;
  const uint top_right =
      y0 * input_row_values + x1 * parameters.channels + channel;
  const uint bottom_left =
      y1 * input_row_values + x0 * parameters.channels + channel;
  const uint bottom_right =
      y1 * input_row_values + x1 * parameters.channels + channel;

  const half top_right_weighted = weight_x1 * input[top_right];
  const half top_sum =
      fma(weight_x0, input[top_left], top_right_weighted);
  const half bottom_right_weighted = weight_x1 * input[bottom_right];
  const half bottom_sum =
      fma(weight_x0, input[bottom_left], bottom_right_weighted);
  const half bottom_weighted = weight_y1 * bottom_sum;
  const half result = fma(weight_y0, top_sum, bottom_weighted);

  const uint output_index =
      (output_y * parameters.output_width + output_x) * parameters.channels +
      channel;
  output[output_index] = result;
}
