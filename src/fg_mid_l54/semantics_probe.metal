#include <metal_stdlib>

using namespace metal;

constant uint kDenseEx2Count = (2u << 21u) + 1u;

kernel void fg_mid_l54_dense_ex2(
    device uint* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= kDenseEx2Count) return;
  const float input = -float(index) * as_type<float>(uint(0x35000000u));
  output[index] = as_type<uint>(fast::exp2(input));
}
