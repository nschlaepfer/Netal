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

inline float ulp32_of(float v) {
  int e = 0; (void)frexp(fabs(v), e); return exp2(float(e - 24));
}
// 2^(exponent(v) - 25): one unit of the HMMA alignment grid when v's exponent bounds the maximum
// term exponent. The grid sits 25 below the leading bit of the largest aligned term.
inline float grid_unit_below(float v) {
  int e = 0; (void)frexp(fabs(v), e); return exp2(float(e - 1 - 25));
}

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
kernel void fg_conv_3x3_split_guarded(
    device const half* packed_weights [[buffer(0)]],
    device const half* weights_lo [[buffer(4)]],
    device uchar* suspect [[buffer(5)]],
    device uchar* resume_stage [[buffer(6)]],
    device half* resume_acc [[buffer(7)]],
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
  bool suspect_element[2][2] = {{false, false}, {false, false}};
  uchar first_stage[2][2] = {{255u, 255u}, {255u, 255u}};
  half resume_value[2][2] = {{half(0.0h), half(0.0h)}, {half(0.0h), half(0.0h)}};
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
      const uint stage_index = channel_stage * 9u + spatial_step;
      simdgroup_float8x8 a[2][2];
      simdgroup_float8x8 b[2];
      simdgroup_float8x8 blo[2];
      simdgroup_float8x8 qa[2][2];
      simdgroup_float8x8 qb[2];
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
        const uint wi0 = packed_weight_index(tile, input_channel, output_channel,
                                             parameters.input_channels);
        const uint wi1 = packed_weight_index(tile, input_channel, output_channel + 1u,
                                             parameters.input_channels);
        b[k_block].thread_elements()[0] = float(packed_weights[wi0]);
        b[k_block].thread_elements()[1] = float(packed_weights[wi1]);
        blo[k_block].thread_elements()[0] = float(weights_lo[wi0]);
        blo[k_block].thread_elements()[1] = float(weights_lo[wi1]);
        // |b| = |b_hi| + |b_lo|: the remainder carries b's sign, so the magnitudes add exactly.
        qb[k_block].thread_elements()[0] = fabs(b[k_block].thread_elements()[0]) + fabs(blo[k_block].thread_elements()[0]);
        qb[k_block].thread_elements()[1] = fabs(b[k_block].thread_elements()[1]) + fabs(blo[k_block].thread_elements()[1]);
#pragma clang loop unroll(full)
        for (uint rb = 0u; rb < 2u; ++rb) {
          qa[rb][k_block].thread_elements()[0] = fabs(a[rb][k_block].thread_elements()[0]);
          qa[rb][k_block].thread_elements()[1] = fabs(a[rb][k_block].thread_elements()[1]);
        }
      }
#pragma clang loop unroll(full)
      for (uint row_block = 0u; row_block < 2u; ++row_block) {
        // hi and lo partial sums, each seeded at zero so neither carries the prior's width.
        const float prior0 = accumulator[row_block].thread_elements()[0];
        const float prior1 = accumulator[row_block].thread_elements()[1];
        // Sum of magnitudes over the 16 products, plus |prior|: its exponent bounds the maximum
        // term exponent, which is what the truncation grid hangs off.
        simdgroup_float8x8 bound;
        bound.thread_elements()[0] = fabs(prior0);
        bound.thread_elements()[1] = fabs(prior1);
        simdgroup_multiply_accumulate(bound, qa[row_block][0], qb[0], bound);
        simdgroup_multiply_accumulate(bound, qa[row_block][1], qb[1], bound);
        simdgroup_float8x8 shi = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 slo = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_multiply_accumulate(shi, a[row_block][0], b[0], shi);
        simdgroup_multiply_accumulate(shi, a[row_block][1], b[1], shi);
        simdgroup_multiply_accumulate(slo, a[row_block][0], blo[0], slo);
        simdgroup_multiply_accumulate(slo, a[row_block][1], blo[1], slo);
#pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
          // prior + shi + slo, exactly, in a double-float pair, then one rounding to half.
          float hi = accumulator[row_block].thread_elements()[h], lo = 0.0f;
          const float terms[2] = {shi.thread_elements()[h], slo.thread_elements()[h]};
#pragma clang loop unroll(full)
          for (uint t = 0u; t < 2u; ++t) {
            float s = hi + terms[t], bp = s - hi;
            float e = (hi - (s - bp)) + (terms[t] - bp);
            e += lo;
            const float s2 = s + e, bp2 = s2 - s;
            lo = (s - (s2 - bp2)) + (e - bp2);
            hi = s2;
          }
          const float value = hi + lo;
          // S is exact (three routes agree to 62 of 491,520). NVIDIA's result differs from round(S)
          // only when S lies within the total truncation loss of an fp16 midpoint. Each of the 17
          // terms loses under one grid unit; the grid is 2^(max_exp - 25); and exp(bound) is an
          // upper bound on max_exp. Plus the double-float combination's own two roundings.
          const float error = 17.0f * grid_unit_below(bound.thread_elements()[h]) + 2.0f * ulp32_of(value);
          const half low = half(value - error), high = half(value + error);
          if (as_type<ushort>(low) != as_type<ushort>(high)) {
            if (!suspect_element[row_block][h]) {
              first_stage[row_block][h] = uchar(stage_index);
              resume_value[row_block][h] = half(h == 0u ? prior0 : prior1);
            }
            suspect_element[row_block][h] = true;
          }
          accumulator[row_block].thread_elements()[h] = float(half(value));
        }
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
    suspect[output_base] = suspect_element[row_block][0] ? 1u : 0u;
    suspect[output_base + 1u] = suspect_element[row_block][1] ? 1u : 0u;
    resume_stage[output_base] = first_stage[row_block][0];
    resume_stage[output_base + 1u] = first_stage[row_block][1];
    resume_acc[output_base] = resume_value[row_block][0];
    resume_acc[output_base + 1u] = resume_value[row_block][1];
  }
}
