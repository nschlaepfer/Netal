#include <metal_stdlib>

using namespace metal;

struct EarlyL11Parameters {
  uint2 input_origin;
  uint2 output_origin;
  uint2 copy_extent;
};

kernel void fg_early_l11(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant EarlyL11Parameters& parameters [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.copy_extent.x ||
      position.y >= parameters.copy_extent.y) {
    return;
  }
  const uint2 source = position + parameters.input_origin;
  const uint2 destination = position + parameters.output_origin;
  output.write(input.read(source), destination);
}
