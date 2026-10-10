#include <metal_stdlib>
using namespace metal;
// Hardware lowering of the prefix 1x1 convolution (launches 22/30/31/34/35/38/39/42/43/46/47/50/51).
// See conv_3x3_hw.metal for the layout argument; here a stage's weights sit at
// output_group * (input_channels * 8) + stage * 128 + n * 16.
struct Conv1x1Parameters { uint width, height, input_channels, output_channels, active_weight_half_count, apply_leaky_relu; };

inline float dot16_permuted(half4 a0, half4 a1, half4 a2, half4 a3, half4 b0, half4 b1, half4 b2, half4 b3, float sum) {
  // b position p holds weight k(p); a is in k order. k(p) = (p & 2 ? 8 : 0) + 2 * (p >> 2) + (p & 1).
  const half a[16] = {a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3], a2[0], a2[1], a2[2], a2[3], a3[0], a3[1], a3[2], a3[3]};
  const half b[16] = {b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2], b1[3], b2[0], b2[1], b2[2], b2[3], b3[0], b3[1], b3[2], b3[3]};
#pragma clang loop unroll(full)
  for (uint p = 0u; p < 16u; ++p) {
    const uint k = ((p & 2u) != 0u ? 8u : 0u) + 2u * (p >> 2u) + (p & 1u);
    sum = fma(float(a[k]), float(b[p]), sum);
  }
  return sum;
}
inline float stage16(device const half* a, device const half* b, bool a_ok, float sum) {
  device const half4* A = reinterpret_cast<device const half4*>(a);
  device const half4* B = reinterpret_cast<device const half4*>(b);
  const half4 z = half4(0.0h);
  return dot16_permuted(a_ok ? A[0] : z, a_ok ? A[1] : z, a_ok ? A[2] : z, a_ok ? A[3] : z, B[0], B[1], B[2], B[3], sum);
}

kernel void fg_conv_1x1(device const half* packed_weights [[buffer(0)]], device const half* input [[buffer(1)]],
                        device half* output [[buffer(2)]], constant Conv1x1Parameters& p [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]]) {
  const uint oc = gid.x, px = gid.y, py = gid.z;
  if (oc >= p.output_channels || px >= p.width || py >= p.height) return;
  const uint output_group = oc / 8u, n = oc % 8u;
  const uint bias_half_offset = p.active_weight_half_count * 2u;
  half acc = packed_weights[bias_half_offset + output_group * 128u + n];
  const uint stages = p.input_channels / 16u;
  device const half* w = packed_weights + output_group * (p.input_channels * 8u) + n * 16u;
  device const half* in = input + (py * p.width + px) * p.input_channels;
  for (uint s = 0u; s < stages; ++s) {
    acc = half(stage16(in + s * 16u, w + s * 128u, true, float(acc)));
  }
  if (p.apply_leaky_relu != 0u) acc = max(acc, half(acc * half(0.01f)));
  output[(py * p.width + px) * p.output_channels + oc] = acc;
}
