// Dump Metal's fast::exp2 bit patterns on the two sm_120 ex2 correction grids (the FG correction streams are deltas relative to
// this function), so a CPU emulator can reproduce ex2.approx.ftz.f32 exactly: RTX_bits = fast_bits - delta.
#include <metal_stdlib>
using namespace metal;
kernel void fast_exp2_unit(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i > 16777216u) return; const float input = -float(i) * as_type<float>(uint(0x33800000u)); out[i] = as_type<uint>(fast::exp2(input));
}
kernel void fast_exp2_frac(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i > 8388608u) return; const float input = fma(float(i), as_type<float>(uint(0x34000000u)), -2.0f); out[i] = as_type<uint>(fast::exp2(input));
}
