#include <metal_stdlib>

using namespace metal;

struct ElementwiseFamilyParameters {
  uint output_width;
  uint output_height;
  uint channels;
  uint input_a_width;
  uint input_a_height;
  uint input_b_width;
  uint input_b_height;
  uint vector_thread_stride;
};

// Shared semantic lowering for the four selected k_element_wise modules.
// Each invocation retains its live grid and block geometry. The only module
// specialization is the number of eight-half vectors per spatial position.
kernel void fg_elementwise_add_family(
    device const half2* input_a [[buffer(0)]],
    device const half2* input_b [[buffer(1)]],
    device half2* output [[buffer(2)]],
    constant ElementwiseFamilyParameters& parameters [[buffer(3)]],
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
       vector_index += parameters.vector_thread_stride) {
    const uint output_pair =
        (output_position * parameters.channels + vector_index * 8u) / 2u;
    const uint input_a_pair =
        (input_a_position * parameters.channels + vector_index * 8u) / 2u;
    const uint input_b_pair =
        (input_b_position * parameters.channels + vector_index * 8u) / 2u;

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
