#include <metal_stdlib>
#include "hmma_exact_clz.metalh"
using namespace metal;

struct Conv3x3Parameters {
  uint width;
  uint height;
  uint input_channels;
  uint output_channels;
  uint active_weight_half_count;
  uint column_stage_schedule;
};

constant uint kClzSpatialOrder[9] = {0u, 3u, 6u, 1u, 4u, 7u, 2u, 5u, 8u};

inline uint clz_weight_index(uint tile, uint input_channel,
                             uint output_channel, uint input_channels) {
  const uint output_group = output_channel / 8u;
  const uint n = output_channel % 8u;
  const uint channel_stage = input_channel / 16u;
  const uint k = input_channel % 16u;
  const uint thread_in_group = (k % 8u) / 2u;
  const uint fragment_half = (k >= 8u ? 2u : 0u) + (k & 1u);
  const uint lane = n * 4u + thread_in_group;
  return output_group * (9u * input_channels * 8u) +
         tile * (input_channels * 8u) + channel_stage * 128u +
         lane * 4u + fragment_half;
}

kernel void fg_conv_3x3(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Conv3x3Parameters& parameters [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= parameters.width || gid.y >= parameters.height ||
      gid.z >= parameters.output_channels) return;
  const uint output_channel = gid.z;
  const uint output_group = output_channel / 8u;
  const uint output_in_group = output_channel % 8u;
  const uint bias_half_offset = parameters.active_weight_half_count * 2u;
  half accumulator = packed_weights[
      bias_half_offset + output_group * 128u + output_in_group];
  const uint channel_stages = parameters.input_channels / 16u;
  const uint schedule_steps = 9u * channel_stages;
  for (uint schedule_step = 0u; schedule_step < schedule_steps;
       ++schedule_step) {
      uint channel_stage = 0u;
      uint tile = 0u;
      if (parameters.column_stage_schedule == 0u) {
        channel_stage = schedule_step / 9u;
        tile = kClzSpatialOrder[schedule_step % 9u];
      } else {
        const uint column_block = schedule_step / (channel_stages * 3u);
        const uint within_column = schedule_step % (channel_stages * 3u);
        channel_stage = within_column / 3u;
        const uint row = within_column % 3u;
        tile = column_block + row * 3u;
      }
      const int input_y = int(gid.y) + int(tile / 3u) - 1;
      const int input_x = int(gid.x) + int(tile % 3u) - 1;
      const bool in_bounds = input_y >= 0 &&
                             input_y < int(parameters.height) &&
                             input_x >= 0 &&
                             input_x < int(parameters.width);
      const uint input_base = in_bounds
          ? (uint(input_y) * parameters.width + uint(input_x)) *
                parameters.input_channels : 0u;
      ushort a_values[16];
      ushort b_values[16];
      for (uint inner = 0u; inner < 16u; ++inner) {
        const uint input_channel = channel_stage * 16u + inner;
        a_values[inner] = in_bounds
            ? as_type<ushort>(input[input_base + input_channel]) : ushort(0u);
        b_values[inner] = as_type<ushort>(packed_weights[
            clz_weight_index(tile, input_channel, output_channel,
                             parameters.input_channels)]);
      }
      accumulator = fg_hmma_f16_bits_16_clz(a_values, b_values, accumulator);
  }
  const half slope = half(0.01f);
  accumulator = max(accumulator, half(accumulator * slope));
  output[(gid.y * parameters.width + gid.x) * parameters.output_channels +
         output_channel] = accumulator;
}
