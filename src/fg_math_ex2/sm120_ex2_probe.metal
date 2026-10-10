#include <metal_stdlib>
#include "sm120_ex2.metalh"

using namespace metal;

struct ProbeParameters {
  uint count;
};

kernel void fg_sm120_ex2_probe(
    device const uint* input_bits [[buffer(0)]],
    device const uint* unit_table [[buffer(1)]],
    device const uint* fraction_table [[buffer(2)]],
    device uint* output_bits [[buffer(3)]],
    constant ProbeParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= parameters.count) return;
  const float input = as_type<float>(input_bits[index]);
  output_bits[index] = as_type<uint>(
      neuralstudio_fg_math::sm120_ex2_nonpositive(
          input, unit_table, fraction_table));
}

