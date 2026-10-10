#include <metal_stdlib>

#include "hmma_exact.metalh"

using namespace metal;

inline half fg_block0_bot1_up(device const half *input, int x, int y,
                              uint channel) {
  if (x < 0 || x >= 320 || y < 0 || y >= 184) return half(0);
  const float source_x = (float(x) - 0.5f) * 0.5f;
  const float source_y = (float(y) - 0.5f) * 0.5f;
  const uint x0 =
      source_x < 0.0f ? 0u : min(uint(source_x), uint(159));
  const uint x1 =
      source_x < 0.0f ? 0u : min(uint(source_x + 1.0f), uint(159));
  const uint y0 =
      source_y < 0.0f ? 0u : min(uint(source_y), uint(91));
  const uint y1 =
      source_y < 0.0f ? 0u : min(uint(source_y + 1.0f), uint(91));
  const half wx1 = half(source_x - float(x0));
  const half wx0 = half(1.0f - (source_x - float(x0)));
  const half wy1 = half(source_y - float(y0));
  const half wy0 = half(1.0f - (source_y - float(y0)));
  const half a = input[(y0 * 160u + x0) * 32u + channel];
  const half b = input[(y0 * 160u + x1) * 32u + channel];
  const half c = input[(y1 * 160u + x0) * 32u + channel];
  const half d = input[(y1 * 160u + x1) * 32u + channel];
  half top = half(half(wx0 * a) + half(wx1 * b));
  top = half(wy0 * top);
  half bottom = half(half(wx0 * c) + half(wx1 * d));
  bottom = half(wy1 * bottom);
  return half(top + bottom);
}

inline uint fg_block0_bot1_weight_index(uint spatial_tile,
                                        uint channel_stage,
                                        uint input_channel,
                                        uint output_channel) {
  const uint k = input_channel % 16u;
  const uint lane = output_channel * 4u + (k % 8u) / 2u;
  const uint fragment_half = (k >= 8u ? 2u : 0u) + (k & 1u);
  return spatial_tile * 256u + channel_stage * 128u +
         (fragment_half >= 2u ? 64u : 0u) + lane * 2u +
         (fragment_half & 1u);
}

kernel void fg_custom_block0_bot1(
    device const half *weights [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const half *input0 [[buffer(2)]],
    device const half *input1 [[buffer(3)]],
    device const half *input2 [[buffer(4)]],
    device half *output0 [[buffer(5)]],
    device half *output1 [[buffer(6)]],
    device half *output2 [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= 320 || gid.y >= 184 || gid.z >= 8) return;
  const uint output_channel = gid.z;
  device const half *input =
      output_channel < 4 ? input0 : (output_channel == 4 ? input1 : input2);
  half accumulator = bias[output_channel];
  for (uint spatial_tile = 0; spatial_tile < 9; ++spatial_tile) {
    const int x = int(gid.x) + int(spatial_tile % 3u) - 1;
    const int y = int(gid.y) + int(spatial_tile / 3u) - 1;
    for (uint channel_stage = 0; channel_stage < 2; ++channel_stage) {
      ushort a[16];
      ushort b[16];
      for (uint k = 0; k < 16; ++k) {
        const uint input_channel = channel_stage * 16u + k;
        a[k] = as_type<ushort>(
            fg_block0_bot1_up(input, x, y, input_channel));
        b[k] = as_type<ushort>(weights[fg_block0_bot1_weight_index(
            spatial_tile, channel_stage, input_channel, output_channel)]);
      }
      accumulator = fg_hmma_f16_bits_16(a, b, accumulator);
    }
  }
  if (output_channel >= 4) {
    accumulator = min(max(accumulator, half(-6.0f)), half(6.0f));
  }
  if (output_channel < 4) {
    output0[(gid.y * 320u + gid.x) * 4u + output_channel] = accumulator;
  } else if (output_channel == 4) {
    output1[gid.y * 320u + gid.x] = accumulator;
  } else {
    output2[(gid.y * 320u + gid.x) * 3u + output_channel - 5u] =
        accumulator;
  }
}
