#include <metal_stdlib>

using namespace metal;

struct StatusWriteParameters {
  uint count;
  uint value;
};

kernel void fg_early_status_write(
    device uint* output [[buffer(0)]],
    constant StatusWriteParameters& parameters [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < parameters.count) output[index] = parameters.value;
}
