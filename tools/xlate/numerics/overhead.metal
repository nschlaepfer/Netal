#include <metal_stdlib>
using namespace metal;
kernel void touch(device uint* b [[buffer(0)]], uint gid [[thread_position_in_grid]]) { if (gid == 0u) b[0] += 1u; }
