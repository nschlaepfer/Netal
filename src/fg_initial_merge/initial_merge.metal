#include <metal_stdlib>

using namespace metal;

struct InitialMergeParameters {
  uint height;
  uint width;
  float interpolation;
};

// Width-faithful lowering of live-selected L56 / k_initial_merge.
//
// The CUDA output stride is twenty bytes.  Keep the five explicit u32 stores:
// using ulong output stores here would overlap adjacent records at this stride.
kernel void fg_initial_merge(
    device uint* output_words [[buffer(0)]],
    device const uint* input_a_words [[buffer(1)]],
    device const uint* input_b_words [[buffer(2)]],
    device const ushort* input_c [[buffer(3)]],
    constant InitialMergeParameters& parameters [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.width || position.y >= parameters.height) {
    return;
  }

  const uint index = position.y * parameters.width + position.x;
  const uint input_word = index * 2u;
  const uint output_word = index * 5u;
  const half interpolation_half = half(parameters.interpolation);
  const ushort interpolation_bits = as_type<ushort>(interpolation_half);

  output_words[output_word + 0u] = input_a_words[input_word + 0u];
  output_words[output_word + 1u] = input_a_words[input_word + 1u];
  output_words[output_word + 2u] = input_b_words[input_word + 0u];
  output_words[output_word + 3u] = input_b_words[input_word + 1u];
  output_words[output_word + 4u] =
      uint(input_c[index]) | (uint(interpolation_bits) << 16u);
}
