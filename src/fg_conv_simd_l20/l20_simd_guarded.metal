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

constant uint kGuardedSpatialOrder[9] = {
    0u, 3u, 6u, 1u, 4u, 7u, 2u, 5u, 8u};

inline uint guarded_weight_index(uint tile, uint input_channel,
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

inline uint2 guarded_matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u),
               2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}

inline half guarded_stage_result(
    float approximate, half prior, device const half* packed_weights,
    device const half* input, bool in_bounds, uint input_base, uint tile,
    uint channel_stage, uint output_channel, uint input_channels) {
  ushort a_values[16];
  ushort b_values[16];
  int maximum_exponent = -21;
#pragma clang loop unroll(full)
  for (uint inner = 0u; inner < 16u; ++inner) {
    const uint input_channel = channel_stage * 16u + inner;
    const ushort a_bits = in_bounds
        ? as_type<ushort>(input[input_base + input_channel]) : ushort(0u);
    const ushort b_bits = as_type<ushort>(packed_weights[
        guarded_weight_index(tile, input_channel, output_channel,
                             input_channels)]);
    a_values[inner] = a_bits;
    b_values[inner] = b_bits;
    const uint a_field = (uint(a_bits) >> 10u) & 31u;
    const uint b_field = (uint(b_bits) >> 10u) & 31u;
    const uint a_mantissa = (uint(a_bits) & 1023u) +
                            (a_field != 0u ? 1024u : 0u);
    const uint b_mantissa = (uint(b_bits) & 1023u) +
                            (b_field != 0u ? 1024u : 0u);
    if (a_mantissa != 0u && b_mantissa != 0u) {
      const int a_exponent = a_field == 0u ? -14 : int(a_field) - 15;
      const int b_exponent = b_field == 0u ? -14 : int(b_field) - 15;
      maximum_exponent = max(maximum_exponent, a_exponent + b_exponent);
    }
  }
  const ushort prior_bits = as_type<ushort>(prior);
  const uint prior_field = (uint(prior_bits) >> 10u) & 31u;
  const uint prior_fraction = uint(prior_bits) & 1023u;
  if (prior_field != 0u) {
    maximum_exponent = max(maximum_exponent, int(prior_field) - 15);
  } else if (prior_fraction != 0u) {
    maximum_exponent = max(
        maximum_exponent, int(31u - clz(prior_fraction)) - 24);
  }

  const half rounded = half(approximate);
  const float center = float(rounded);
  const float lower = float(nextafter(rounded, half(-INFINITY)));
  const float upper = float(nextafter(rounded, half(INFINITY)));
  const float lower_midpoint = (lower + center) * 0.5f;
  const float upper_midpoint = (center + upper) * 0.5f;
  const float distance = min(abs(approximate - lower_midpoint),
                             abs(upper_midpoint - approximate));
  // 256 units of the exact helper's common 2^(max_exp-25) grid covers
  // per-product truncation plus conservative FP32 accumulation error.
  const float error_bound = ldexp(1.0f, maximum_exponent - 17);
  if (isfinite(approximate) && distance > error_bound) return rounded;
  return fg_hmma_f16_bits_16_clz(a_values, b_values, prior);
}

kernel void fg_conv_3x3_simd_guarded(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant Conv3x3Parameters& parameters [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint output_group = group.z / 8u;
  const uint pixel_slice = group.z & 7u;
  if (pixel_slice >= 2u || output_group * 8u >= parameters.output_channels ||
      parameters.column_stage_schedule != 0u) return;

  const uint2 element = guarded_matrix_element(lane);
  const uint matrix_row = element.x;
  const uint matrix_column = element.y;
  const uint x0 = group.x * 8u;
  const uint y0 = group.y * 4u + pixel_slice * 2u;
  const uint bias_base = parameters.active_weight_half_count * 2u +
                         output_group * 128u;
  simdgroup_float8x8 accumulator[2];
#pragma clang loop unroll(full)
  for (uint row_block = 0u; row_block < 2u; ++row_block) {
    accumulator[row_block].thread_elements()[0] =
        float(packed_weights[bias_base + matrix_column]);
    accumulator[row_block].thread_elements()[1] =
        float(packed_weights[bias_base + matrix_column + 1u]);
  }

  const uint channel_stages = parameters.input_channels / 16u;
  for (uint channel_stage = 0u; channel_stage < channel_stages;
       ++channel_stage) {
    for (uint spatial_step = 0u; spatial_step < 9u; ++spatial_step) {
      const uint tile = kGuardedSpatialOrder[spatial_step];
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
                  parameters.input_channels : 0u;
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
            guarded_weight_index(tile, input_channel, output_channel,
                                 parameters.input_channels)]);
        b[k_block].thread_elements()[1] = float(packed_weights[
            guarded_weight_index(tile, input_channel, output_channel + 1u,
                                 parameters.input_channels)]);
      }
#pragma clang loop unroll(full)
      for (uint row_block = 0u; row_block < 2u; ++row_block) {
        const half prior0 = half(accumulator[row_block].thread_elements()[0]);
        const half prior1 = half(accumulator[row_block].thread_elements()[1]);
        simdgroup_multiply_accumulate(accumulator[row_block],
                                      a[row_block][0], b[0],
                                      accumulator[row_block]);
        simdgroup_multiply_accumulate(accumulator[row_block],
                                      a[row_block][1], b[1],
                                      accumulator[row_block]);
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
                  parameters.input_channels : 0u;
        accumulator[row_block].thread_elements()[0] = float(
            guarded_stage_result(
                accumulator[row_block].thread_elements()[0], prior0,
                packed_weights, input, in_bounds, input_base, tile,
                channel_stage, output_group * 8u + matrix_column,
                parameters.input_channels));
        accumulator[row_block].thread_elements()[1] = float(
            guarded_stage_result(
                accumulator[row_block].thread_elements()[1], prior1,
                packed_weights, input, in_bounds, input_base, tile,
                channel_stage, output_group * 8u + matrix_column + 1u,
                parameters.input_channels));
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
