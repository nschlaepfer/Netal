#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

struct ConvL51Parameters {
  uint width;
  uint height;
  uint output_width;
  uint output_height;
};

inline uint l51_packed_weight_half_index(uint input_channel,
                                         uint output_channel) {
  const uint output_group = output_channel / 8u;
  const uint n = output_channel % 8u;
  const uint channel_stage = input_channel / 16u;
  const uint k = input_channel % 16u;
  const uint thread_in_group = (k % 8u) / 2u;
  const uint fragment_half = (k >= 8u ? 2u : 0u) + (k & 1u);
  const uint lane = n * 4u + thread_in_group;
  return (output_group * 512u + channel_stage * 256u +
          lane * 8u + fragment_half * 2u) / 2u;
}

kernel void fg_conv_l51(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ConvL51Parameters& parameters [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= parameters.output_width ||
      gid.y >= parameters.output_height || gid.z >= 16u) {
    return;
  }
  const uint pixel = gid.y * parameters.width + gid.x;
  const uint input_base = pixel * 32u;
  const uint output_channel = gid.z;
  const uint output_group = output_channel / 8u;
  const uint output_in_group = output_channel % 8u;
  half accumulator =
      packed_weights[1024u + output_group * 128u + output_in_group];
  for (uint channel_stage = 0u; channel_stage < 2u; ++channel_stage) {
    ushort a_values[16];
    ushort b_values[16];
    for (uint inner = 0u; inner < 16u; ++inner) {
      const uint input_channel = channel_stage * 16u + inner;
      a_values[inner] = as_type<ushort>(input[input_base + input_channel]);
      b_values[inner] = as_type<ushort>(packed_weights[
          l51_packed_weight_half_index(input_channel, output_channel)]);
    }
    accumulator = fg_hmma_f16_bits_16(a_values, b_values, accumulator);
  }
  output[pixel * 16u + output_channel] = accumulator;
}
