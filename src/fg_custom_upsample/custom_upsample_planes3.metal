#include <metal_stdlib>
using namespace metal;
// Launch 70: bilinear 2x upsample of the three block0 bottom planes (4, 1 and 3 channels) from 320x184
// to 640x368. One thread per (output pixel, plane); the plane's channels are done in one thread so the
// stores are contiguous. Per-channel arithmetic and operand order are unchanged from the per-channel
// kernel, so every result is bit-identical.
kernel void fg_custom_upsample_planes3(
    device const half* in0 [[buffer(0)]], device const half* in1 [[buffer(1)]], device const half* in2 [[buffer(2)]],
    device half* out0 [[buffer(3)]], device half* out1 [[buffer(4)]], device half* out2 [[buffer(5)]],
    uint3 position [[thread_position_in_grid]]) {
  const uint output_x = position.x, output_y = position.y, plane = position.z;
  if (output_x >= 640u || output_y >= 368u || plane >= 3u) return;
  const uint channels = plane == 0u ? 4u : (plane == 1u ? 1u : 3u);
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
  const uint top_left = y0 * input_row_values + x0 * channels;
  const uint top_right = y0 * input_row_values + x1 * channels;
  const uint bottom_left = y1 * input_row_values + x0 * channels;
  const uint bottom_right = y1 * input_row_values + x1 * channels;
  const uint out_base = (output_y * output_width + output_x) * channels;
  for (uint channel = 0u; channel < channels; ++channel) {
    const half top_right_weighted = weight_x1 * input[top_right + channel];
    const half top_sum = fma(weight_x0, input[top_left + channel], top_right_weighted);
    const half bottom_right_weighted = weight_x1 * input[bottom_right + channel];
    const half bottom_sum = fma(weight_x0, input[bottom_left + channel], bottom_right_weighted);
    const half bottom_weighted = weight_y1 * bottom_sum;
    output[out_base + channel] = fma(weight_y0, top_sum, bottom_weighted);
  }
}
