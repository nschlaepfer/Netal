#include <metal_stdlib>

using namespace metal;

constant uint kUnitCount = (1u << 24u) + 1u;
constant uint kFractionCount = (1u << 23u) + 1u;

kernel void fg_ex2_fast_unit_table(
    device uint* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= kUnitCount) return;
  const float input = -float(index) * as_type<float>(uint(0x33800000u));
  output[index] = as_type<uint>(fast::exp2(input));
}

kernel void fg_ex2_fast_fraction_table(
    device uint* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= kFractionCount) return;
  const float input = fma(float(index),
                          as_type<float>(uint(0x34000000u)), -2.0f);
  output[index] = as_type<uint>(fast::exp2(input));
}

