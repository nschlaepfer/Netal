#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

inline uint conv_pre_weight_half_index(uint tile, uint input_channel,
                                       uint output_channel) {
  const uint output_group = output_channel / 8u;
  const uint n = output_channel % 8u;
  const uint thread_in_group = (input_channel % 8u) / 2u;
  const uint fragment_half = (input_channel >= 8u ? 2u : 0u) +
                             (input_channel & 1u);
  const uint lane = n * 4u + thread_in_group;
  const uint plane = fragment_half >= 2u ? 64u : 0u;
  return output_group * 1152u + tile * 128u + plane + lane * 2u +
         (fragment_half & 1u);
}

inline half conv_pre_value(device const half* weights,
                           device const half* bias,
                           device const half* input,
                           uint x, uint y, uint output_channel) {
  half accumulator = bias[output_channel];
  for (uint tile = 0u; tile < 9u; ++tile) {
    const int input_x = int(x) + int(tile % 3u) - 1;
    const int input_y = int(y) + int(tile / 3u) - 1;
    const bool in_bounds = input_x >= 0 && input_x < 640 &&
                           input_y >= 0 && input_y < 368;
    const uint input_base = in_bounds
        ? (uint(input_y) * 640u + uint(input_x)) * 10u
        : 0u;
    ushort a_values[16];
    ushort b_values[16];
    for (uint input_channel = 0u; input_channel < 16u; ++input_channel) {
      a_values[input_channel] = in_bounds && input_channel < 10u
          ? as_type<ushort>(input[input_base + input_channel])
          : ushort(0u);
      b_values[input_channel] = as_type<ushort>(weights[
          conv_pre_weight_half_index(tile, input_channel, output_channel)]);
    }
    accumulator = fg_hmma_f16_bits_16(a_values, b_values, accumulator);
  }
  const half slope = half(0.01f);
  const half leaky = max(accumulator, half(accumulator * slope));
  return min(max(leaky, half(-6.0f)), half(6.0f));
}

kernel void fg_custom_block0_conv_pre(
    device const half* weights [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= 320u || gid.y >= 184u || gid.z >= 32u) return;
  const uint x = gid.x * 2u;
  const uint y = gid.y * 2u;
  const uint output_channel = gid.z;
  const half v00 = conv_pre_value(weights, bias, input, x, y,
                                  output_channel);
  const half v01 = conv_pre_value(weights, bias, input, x + 1u, y,
                                  output_channel);
  const half v10 = conv_pre_value(weights, bias, input, x, y + 1u,
                                  output_channel);
  const half v11 = conv_pre_value(weights, bias, input, x + 1u, y + 1u,
                                  output_channel);
  half pooled = half(v00 + v01);
  pooled = half(pooled + v10);
  pooled = half(pooled + v11);
  pooled = half(pooled * half(0.25f));
  output[(gid.y * 320u + gid.x) * 32u + output_channel] = pooled;
}
