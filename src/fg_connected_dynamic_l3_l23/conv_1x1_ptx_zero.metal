#include <metal_stdlib>
#include "../fg_conv/hmma_exact.metalh"
using namespace metal;

struct ConnectedConv1x1Parameters {
  uint width, height, input_channels, output_channels;
  uint active_weight_half_count, apply_leaky_relu;
};

inline uint connected_weight_index(uint input_channel, uint output_channel,
                                   uint input_channels) {
  const uint output_group = output_channel / 8u;
  const uint n = output_channel % 8u;
  const uint channel_stage = input_channel / 16u;
  const uint k = input_channel % 16u;
  const uint thread_in_group = (k % 8u) / 2u;
  const uint fragment_half = (k >= 8u ? 2u : 0u) + (k & 1u);
  const uint lane = n * 4u + thread_in_group;
  return output_group * input_channels * 8u + channel_stage * 128u +
         lane * 4u + fragment_half;
}

inline half ptx_leaky_relu(half value) {
  const half product = half(value * half(0.01f));
  half result = max(value, product);
  const ushort input_bits = as_type<ushort>(value);
  // SM120 max.f16 canonicalizes a zero result to positive zero, including
  // max(-0,-0). A negative subnormal whose product underflows to -0 must keep
  // the -0 chosen as the larger operand, so the input (not result) is tested.
  if ((input_bits & 0x7fffu) == 0u)
    result = as_type<half>(ushort(0u));
  return result;
}

kernel void fg_connected_dynamic_conv_1x1(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ConnectedConv1x1Parameters& parameters [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= parameters.width || gid.y >= parameters.height ||
      gid.z >= parameters.output_channels) return;
  const uint pixel = gid.y * parameters.width + gid.x;
  const uint input_base = pixel * parameters.input_channels;
  const uint output_channel = gid.z;
  const uint output_group = output_channel / 8u;
  const uint output_in_group = output_channel % 8u;
  const uint bias_half_offset = parameters.active_weight_half_count * 2u;
  half accumulator = packed_weights[
      bias_half_offset + output_group * 128u + output_in_group];
  for (uint channel_stage = 0u;
       channel_stage < parameters.input_channels / 16u; ++channel_stage) {
    ushort a_values[16];
    ushort b_values[16];
    for (uint inner = 0u; inner < 16u; ++inner) {
      const uint input_channel = channel_stage * 16u + inner;
      a_values[inner] = as_type<ushort>(input[input_base + input_channel]);
      b_values[inner] = as_type<ushort>(packed_weights[
          connected_weight_index(input_channel, output_channel,
                                 parameters.input_channels)]);
    }
    accumulator = fg_hmma_f16_bits_16(a_values, b_values, accumulator);
  }
  if (parameters.apply_leaky_relu != 0u)
    accumulator = ptx_leaky_relu(accumulator);
  output[pixel * parameters.output_channels + output_channel] = accumulator;
}
