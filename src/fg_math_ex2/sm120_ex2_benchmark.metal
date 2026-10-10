#include <metal_stdlib>
#include "sm120_ex2.metalh"
#include "sm120_ex2_compact.metalh"

using namespace metal;

struct BenchmarkParameters {
  uint count;
};

kernel void fg_sm120_ex2_reference_benchmark(
    device const uint* input_bits [[buffer(0)]],
    device const uint* unit_table [[buffer(1)]],
    device const uint* fraction_table [[buffer(2)]],
    device uint* output_bits [[buffer(3)]],
    constant BenchmarkParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= parameters.count) return;
  output_bits[index] = as_type<uint>(
      neuralstudio_fg_math::sm120_ex2_nonpositive(
          as_type<float>(input_bits[index]), unit_table, fraction_table));
}

kernel void fg_sm120_ex2_compact_benchmark(
    device const uint* input_bits [[buffer(0)]],
    device const uchar* unit_corrections [[buffer(1)]],
    device const uchar* fraction_corrections [[buffer(2)]],
    device uint* output_bits [[buffer(3)]],
    constant BenchmarkParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= parameters.count) return;
  output_bits[index] = as_type<uint>(
      neuralstudio_fg_math::sm120_ex2_nonpositive_compact(
          as_type<float>(input_bits[index]), unit_corrections,
          fraction_corrections));
}
