#include <metal_stdlib>
#include "../fg_conv/hmma_exact.metalh"

using namespace metal;

struct Conv3x3Parameters {
  uint width;
  uint height;
  uint input_channels;
  uint output_channels;
  uint active_weight_half_count;
  uint column_stage_schedule;
};

constant uint kMultiSpatialOrder[9] = {0u, 3u, 6u, 1u, 4u, 7u, 2u, 5u, 8u};

inline uint multi_weight_index(uint tile, uint input_channel,
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

inline float product(thread const ushort* a_values,
                     thread const ushort* b_values, uint index) {
  return float(as_type<half>(a_values[index])) *
         float(as_type<half>(b_values[index]));
}

inline half multi_order_hmma(thread const ushort* a_values,
                             thread const ushort* b_values,
                             half prior) {
  float forward = float(prior);
  float reverse = float(prior);
  float even_odd = float(prior);
  float outside_in = float(prior);
#pragma clang loop unroll(full)
  for (uint index = 0u; index < 16u; ++index) {
    forward = fma(float(as_type<half>(a_values[index])),
                  float(as_type<half>(b_values[index])), forward);
    const uint reverse_index = 15u - index;
    reverse = fma(float(as_type<half>(a_values[reverse_index])),
                  float(as_type<half>(b_values[reverse_index])), reverse);
    const uint even_odd_index = index < 8u ? index * 2u
                                           : (index - 8u) * 2u + 1u;
    even_odd = fma(float(as_type<half>(a_values[even_odd_index])),
                   float(as_type<half>(b_values[even_odd_index])), even_odd);
    const uint outside_index = (index & 1u) == 0u
        ? index / 2u : 15u - index / 2u;
    outside_in = fma(float(as_type<half>(a_values[outside_index])),
                     float(as_type<half>(b_values[outside_index])), outside_in);
  }
  float pair[8];
#pragma clang loop unroll(full)
  for (uint index = 0u; index < 8u; ++index)
    pair[index] = product(a_values, b_values, index * 2u) +
                  product(a_values, b_values, index * 2u + 1u);
  float quad[4];
#pragma clang loop unroll(full)
  for (uint index = 0u; index < 4u; ++index)
    quad[index] = pair[index * 2u] + pair[index * 2u + 1u];
  const float tree = float(prior) +
      ((quad[0] + quad[1]) + (quad[2] + quad[3]));
  const ushort forward_bits = as_type<ushort>(half(forward));
  const ushort reverse_bits = as_type<ushort>(half(reverse));
  const ushort even_odd_bits = as_type<ushort>(half(even_odd));
  const ushort outside_bits = as_type<ushort>(half(outside_in));
  const ushort tree_bits = as_type<ushort>(half(tree));
  if (forward_bits == reverse_bits && forward_bits == even_odd_bits &&
      forward_bits == outside_bits && forward_bits == tree_bits) {
    return as_type<half>(forward_bits);
  }
  return fg_hmma_f16_bits_16(a_values, b_values, prior);
}

kernel void fg_conv_3x3(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Conv3x3Parameters& parameters [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= parameters.width || gid.y >= parameters.height ||
      gid.z >= parameters.output_channels ||
      parameters.column_stage_schedule != 0u) {
    return;
  }
  const uint output_channel = gid.z;
  const uint output_group = output_channel / 8u;
  const uint output_in_group = output_channel % 8u;
  const uint bias_half_offset = parameters.active_weight_half_count * 2u;
  half accumulator = packed_weights[
      bias_half_offset + output_group * 128u + output_in_group];
  const uint channel_stages = parameters.input_channels / 16u;
  for (uint channel_stage = 0u; channel_stage < channel_stages;
       ++channel_stage) {
    for (uint spatial_step = 0u; spatial_step < 9u; ++spatial_step) {
      const uint tile = kMultiSpatialOrder[spatial_step];
      const int input_y = int(gid.y) + int(tile / 3u) - 1;
      const int input_x = int(gid.x) + int(tile % 3u) - 1;
      const bool in_bounds = input_y >= 0 &&
                             input_y < int(parameters.height) &&
                             input_x >= 0 &&
                             input_x < int(parameters.width);
      const uint input_base = in_bounds
          ? (uint(input_y) * parameters.width + uint(input_x)) *
                parameters.input_channels
          : 0u;
      ushort a_values[16];
      ushort b_values[16];
#pragma clang loop unroll(full)
      for (uint inner = 0u; inner < 16u; ++inner) {
        const uint input_channel = channel_stage * 16u + inner;
        a_values[inner] = in_bounds
            ? as_type<ushort>(input[input_base + input_channel])
            : ushort(0u);
        b_values[inner] = as_type<ushort>(packed_weights[
            multi_weight_index(tile, input_channel, output_channel,
                               parameters.input_channels)]);
      }
      accumulator = multi_order_hmma(a_values, b_values, accumulator);
    }
  }
  const half slope = half(0.01f);
  accumulator = max(accumulator, half(accumulator * slope));
  output[(gid.y * parameters.width + gid.x) * parameters.output_channels +
         output_channel] = accumulator;
}
