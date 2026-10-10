#include <metal_stdlib>
using namespace metal;
// The block's launch 70 as one dispatch: z in [0, 8) selects plane 0 channels 0-3, plane 1 channel 0, plane 2
// channels 0-2 (the three NHWC planes of 4, 1 and 3 channels). Per-element arithmetic is verbatim
// fg_custom_upsample_plane, so the output is bit-identical; only the dispatch count changes (3 -> 1).
kernel void fg_custom_upsample_planes3(
    device const half* in0 [[buffer(0)]], device const half* in1 [[buffer(1)]], device const half* in2 [[buffer(2)]],
    device half* out0 [[buffer(3)]], device half* out1 [[buffer(4)]], device half* out2 [[buffer(5)]],
    uint3 position [[thread_position_in_grid]]) {
  const uint output_x = position.x, output_y = position.y, z = position.z;
  if (output_x >= 640u || output_y >= 368u || z >= 8u) return;
  const uint plane = z < 4u ? 0u : (z == 4u ? 1u : 2u), channel = z < 4u ? z : (z == 4u ? 0u : z - 5u), channels = z < 4u ? 4u : (z == 4u ? 1u : 3u);
  device const half* input = plane == 0u ? in0 : (plane == 1u ? in1 : in2);
  device half* output = plane == 0u ? out0 : (plane == 1u ? out1 : out2);
  const uint input_height = 184u, input_width = 320u, output_width = 640u;
  const float source_x = (float(output_x) - 0.5f) * 0.5f;
  const float source_y = (float(output_y) - 0.5f) * 0.5f;
  const uint x0 = source_x < 0.0f ? 0u : min(uint(source_x), input_width - 1u);
  const uint x1 = source_x < 0.0f ? 0u : min(uint(source_x + 1.0f), input_width - 1u);
  const uint y0 = source_y < 0.0f ? 0u : min(uint(source_y), input_height - 1u);
  const uint y1 = source_y < 0.0f ? 0u : min(uint(source_y + 1.0f), input_height - 1u);
  const half weight_y1 = half(source_y - float(y0));
  const half weight_x1 = half(source_x - float(x0));
  const half weight_y0 = half((float(y0) - source_y) + 1.0f);
  const half weight_x0 = half((float(x0) - source_x) + 1.0f);
  const uint input_row_values = input_width * channels;
  const uint top_left = y0 * input_row_values + x0 * channels + channel;
  const uint top_right = y0 * input_row_values + x1 * channels + channel;
  const uint bottom_left = y1 * input_row_values + x0 * channels + channel;
  const uint bottom_right = y1 * input_row_values + x1 * channels + channel;
  const half top_right_weighted = weight_x1 * input[top_right];
  const half top_sum = fma(weight_x0, input[top_left], top_right_weighted);
  const half bottom_right_weighted = weight_x1 * input[bottom_right];
  const half bottom_sum = fma(weight_x0, input[bottom_left], bottom_right_weighted);
  const half bottom_weighted = weight_y1 * bottom_sum;
  const half result = fma(weight_y0, top_sum, bottom_weighted);
  output[(output_y * output_width + output_x) * channels + channel] = result;
}
