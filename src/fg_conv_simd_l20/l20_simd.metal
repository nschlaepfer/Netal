#include <metal_stdlib>

using namespace metal;

struct Conv3x3Parameters {
  uint width;
  uint height;
  uint input_channels;
  uint output_channels;
  uint active_weight_half_count;
  uint column_stage_schedule;
};

constant uint kSpatialOrder[9] = {0u, 3u, 6u, 1u, 4u, 7u, 2u, 5u, 8u};

inline uint packed_weight_index(uint tile, uint input_channel,
                                uint output_channel,
                                uint input_channels) {
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

inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u),
               2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}

// First numerical candidate: one SIMD group owns a 16-pixel x 8-channel
// output tile. The host intentionally uses the accepted 8x4x1 dispatch; z
// slices 0 and 1 cover the two 16-pixel halves of that tile, while the other
// six z slices return without writing.
kernel void fg_conv_3x3_simd(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Conv3x3Parameters& parameters [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint output_group = group.z / 8u;
  const uint pixel_slice = group.z & 7u;
  if (pixel_slice >= 2u || output_group * 8u >= parameters.output_channels ||
      parameters.column_stage_schedule != 0u) {
    return;
  }

  const uint2 element = matrix_element(lane);
  const uint matrix_row = element.x;
  const uint matrix_column = element.y;
  const uint x0 = group.x * 8u;
  const uint y0 = group.y * 4u + pixel_slice * 2u;
  const uint bias_base = parameters.active_weight_half_count * 2u +
                         output_group * 128u;
  simdgroup_float8x8 accumulator[2];
#pragma clang loop unroll(full)
  for (uint row_block = 0u; row_block < 2u; ++row_block) {
    accumulator[row_block].thread_elements()[0] = float(
        packed_weights[bias_base + matrix_column]);
    accumulator[row_block].thread_elements()[1] = float(
        packed_weights[bias_base + matrix_column + 1u]);
  }

  const uint channel_stages = parameters.input_channels / 16u;
  for (uint channel_stage = 0u; channel_stage < channel_stages;
       ++channel_stage) {
    for (uint spatial_step = 0u; spatial_step < 9u; ++spatial_step) {
      const uint tile = kSpatialOrder[spatial_step];
      simdgroup_float8x8 a[2][2];
      simdgroup_float8x8 b[2];
#pragma clang loop unroll(full)
      for (uint row_block = 0u; row_block < 2u; ++row_block) {
        const uint local_pixel = row_block * 8u + matrix_row;
        const uint output_x = x0 + (local_pixel & 7u);
        const uint output_y = y0 + (local_pixel >> 3u);
        const int input_x = int(output_x) + int(tile % 3u) - 1;
        const int input_y = int(output_y) + int(tile / 3u) - 1;
        const bool in_bounds = output_x < parameters.width &&
                               output_y < parameters.height &&
                               input_x >= 0 && input_x < int(parameters.width) &&
                               input_y >= 0 && input_y < int(parameters.height);
        const uint input_base = in_bounds
            ? (uint(input_y) * parameters.width + uint(input_x)) *
                  parameters.input_channels
            : 0u;
#pragma clang loop unroll(full)
        for (uint k_block = 0u; k_block < 2u; ++k_block) {
          const uint input_channel = channel_stage * 16u + k_block * 8u +
                                     matrix_column;
          a[row_block][k_block].thread_elements()[0] = in_bounds
              ? float(input[input_base + input_channel]) : 0.0f;
          a[row_block][k_block].thread_elements()[1] = in_bounds
              ? float(input[input_base + input_channel + 1u]) : 0.0f;
        }
      }
#pragma clang loop unroll(full)
      for (uint k_block = 0u; k_block < 2u; ++k_block) {
        const uint input_channel = channel_stage * 16u + k_block * 8u +
                                   matrix_row;
        const uint output_channel = output_group * 8u + matrix_column;
        b[k_block].thread_elements()[0] = float(packed_weights[
            packed_weight_index(tile, input_channel, output_channel,
                                parameters.input_channels)]);
        b[k_block].thread_elements()[1] = float(packed_weights[
            packed_weight_index(tile, input_channel, output_channel + 1u,
                                parameters.input_channels)]);
      }
#pragma clang loop unroll(full)
      for (uint row_block = 0u; row_block < 2u; ++row_block) {
        simdgroup_multiply_accumulate(accumulator[row_block],
                                      a[row_block][0], b[0],
                                      accumulator[row_block]);
        simdgroup_multiply_accumulate(accumulator[row_block],
                                      a[row_block][1], b[1],
                                      accumulator[row_block]);
        accumulator[row_block].thread_elements()[0] = float(half(
            accumulator[row_block].thread_elements()[0]));
        accumulator[row_block].thread_elements()[1] = float(half(
            accumulator[row_block].thread_elements()[1]));
      }
    }
  }

#pragma clang loop unroll(full)
  for (uint row_block = 0u; row_block < 2u; ++row_block) {
    const uint local_pixel = row_block * 8u + matrix_row;
    const uint output_x = x0 + (local_pixel & 7u);
    const uint output_y = y0 + (local_pixel >> 3u);
    if (output_x >= parameters.width || output_y >= parameters.height) continue;
    const uint output_base =
        (output_y * parameters.width + output_x) * parameters.output_channels +
        output_group * 8u + matrix_column;
    half value0 = half(accumulator[row_block].thread_elements()[0]);
    half value1 = half(accumulator[row_block].thread_elements()[1]);
    const half slope = half(0.01f);
    value0 = max(value0, half(value0 * slope));
    value1 = max(value1, half(value1 * slope));
    output[output_base] = value0;
    output[output_base + 1u] = value1;
  }
}
