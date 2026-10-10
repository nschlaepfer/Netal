#include <metal_stdlib>

using namespace metal;

struct ElementwiseAddParameters {
  uint output_width;
  uint output_height;
  uint channels;
  uint input_a_width;
  uint input_a_height;
  uint input_b_width;
  uint input_b_height;
};

// Independent Metal reconstruction of the live-selected launch-000033
// k_element_wise variant. The selected module processes eight FP16 channels
// per z-thread, with eight z-threads covering the 32 channel vectors.
kernel void fg_elementwise_add_256(
    device const half2* input_a [[buffer(0)]],
    device const half2* input_b [[buffer(1)]],
    device half2* output [[buffer(2)]],
    constant ElementwiseAddParameters& parameters [[buffer(3)]],
    uint3 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) {
    return;
  }

  const uint vectors_per_position = parameters.channels / 8u;
  const uint output_position =
      position.y * parameters.output_width + position.x;
  const uint input_a_position =
      (position.y % parameters.input_a_height) * parameters.input_a_width +
      (position.x % parameters.input_a_width);
  const uint input_b_position =
      (position.y % parameters.input_b_height) * parameters.input_b_width +
      (position.x % parameters.input_b_width);

  for (uint vector_index = position.z; vector_index < vectors_per_position;
       vector_index += 8u) {
    const uint output_half =
        output_position * parameters.channels + vector_index * 8u;
    const uint input_a_half =
        input_a_position * parameters.channels + vector_index * 8u;
    const uint input_b_half =
        input_b_position * parameters.channels + vector_index * 8u;
    const uint output_pair = output_half / 2u;
    const uint input_a_pair = input_a_half / 2u;
    const uint input_b_pair = input_b_half / 2u;

    output[output_pair + 0u] =
        input_a[input_a_pair + 0u] + input_b[input_b_pair + 0u];
    output[output_pair + 1u] =
        input_a[input_a_pair + 1u] + input_b[input_b_pair + 1u];
    output[output_pair + 2u] =
        input_a[input_a_pair + 2u] + input_b[input_b_pair + 2u];
    output[output_pair + 3u] =
        input_a[input_a_pair + 3u] + input_b[input_b_pair + 3u];
  }
}
