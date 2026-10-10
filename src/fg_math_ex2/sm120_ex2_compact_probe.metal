#include <metal_stdlib>
#include "sm120_ex2_compact.metalh"

using namespace metal;

struct CompactProbeParameters {
  uint count;
};

kernel void fg_sm120_ex2_compact_probe(
    device const uint* input_bits [[buffer(0)]],
    device const uchar* unit_corrections [[buffer(1)]],
    device const uchar* fraction_corrections [[buffer(2)]],
    device uint* output_bits [[buffer(3)]],
    constant CompactProbeParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= parameters.count) return;
  const float input = as_type<float>(input_bits[index]);
  output_bits[index] = as_type<uint>(
      neuralstudio_fg_math::sm120_ex2_nonpositive_compact(
          input, unit_corrections, fraction_corrections));
}
