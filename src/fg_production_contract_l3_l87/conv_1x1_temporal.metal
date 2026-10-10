#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

struct Conv1x1Parameters {
  uint width;
  uint height;
  uint input_channels;
  uint output_channels;
  uint active_weight_half_count;
  uint apply_leaky_relu;
};

inline uint packed_1x1_weight_half_index(uint input_channel,
                                         uint output_channel,
                                         uint input_channels) {
  const uint output_group = output_channel / 8u;
  const uint n = output_channel % 8u;
  const uint channel_stage = input_channel / 16u;
  const uint k = input_channel % 16u;
  const uint thread_in_group = (k % 8u) / 2u;
  const uint fragment_half = (k >= 8u ? 2u : 0u) + (k & 1u);
  const uint lane = n * 4u + thread_in_group;
  const uint group_stride_half = input_channels * 8u;
  return output_group * group_stride_half + channel_stage * 128u +
         lane * 4u + fragment_half;
}

kernel void fg_conv_1x1(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Conv1x1Parameters& parameters [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= parameters.width || gid.y >= parameters.height ||
      gid.z >= parameters.output_channels) {
    return;
  }
  const uint pixel = gid.y * parameters.width + gid.x;
  const uint input_base = pixel * parameters.input_channels;
  const uint output_channel = gid.z;
  const uint output_group = output_channel / 8u;
  const uint output_in_group = output_channel % 8u;
  const uint bias_half_offset = parameters.active_weight_half_count * 2u;
  half accumulator = packed_weights[
      bias_half_offset + output_group * 128u + output_in_group];
  const uint channel_stages = parameters.input_channels / 16u;
  for (uint channel_stage = 0u; channel_stage < channel_stages;
       ++channel_stage) {
    ushort a_values[16];
    ushort b_values[16];
    for (uint inner = 0u; inner < 16u; ++inner) {
      const uint input_channel = channel_stage * 16u + inner;
      a_values[inner] = as_type<ushort>(input[input_base + input_channel]);
      b_values[inner] = as_type<ushort>(packed_weights[
          packed_1x1_weight_half_index(input_channel, output_channel,
                                       parameters.input_channels)]);
    }
    accumulator = fg_hmma_f16_bits_16(a_values, b_values, accumulator);
    // Blackwell HMMA canonicalizes an exact nonzero negative total that rounds
    // to zero as +0.  The shared scalar emulator otherwise preserves -0.
    if (as_type<ushort>(accumulator) == 0x8000u)
      accumulator = as_type<half>(ushort(0u));
  }
  if (parameters.apply_leaky_relu != 0u) {
    const half slope = half(0.01f);
    accumulator = max(accumulator, half(accumulator * slope));
  }
  output[pixel * parameters.output_channels + output_channel] = accumulator;
}

