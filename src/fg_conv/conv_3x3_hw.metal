#include <metal_stdlib>
using namespace metal;
// Hardware lowering of the prefix 3x3 convolution (launches 20/24/26/28). Same parameters, buffers,
// bias/leaky-relu epilogue and output layout as fg_conv_3x3; the arithmetic is an fp32 FMA chain per
// K=16 stage rounded to half (the exact kernel reproduces NVIDIA's HMMA truncation bit for bit and is
// the oracle this is PSNR-gated against).
//
// The packed weight layout stores each (output channel, tile, stage) as 16 contiguous halves in the
// order p = ((k % 8) / 2) * 4 + (k >= 8 ? 2 : 0) + (k & 1), so the stage's weights are two half8 loads
// and the input taps are permuted once in registers to match (half4 loads; Metal 3.1 has no half8). Threads are launched output-channel
// fastest inside a threadgroup: the 8 lanes of one pixel read consecutive weight blocks and share
// the input, so both streams coalesce.
struct Conv3x3Parameters { uint width, height, input_channels, output_channels, active_weight_half_count, column_stage_schedule; };

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

kernel void fg_conv_3x3(device const half* packed_weights [[buffer(0)]], device const half* input [[buffer(1)]],
                        device half* output [[buffer(2)]], constant Conv3x3Parameters& p [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]]) {
  // Grid is launched (output_channels, width, height): x = output channel, y = pixel x, z = pixel y.
  const uint oc = gid.x, px = gid.y, py = gid.z;
  if (oc >= p.output_channels || px >= p.width || py >= p.height) return;
  const uint output_group = oc / 8u, n = oc % 8u;
  const uint bias_half_offset = p.active_weight_half_count * 2u;
  half acc = packed_weights[bias_half_offset + output_group * 128u + n];
  const uint stages = p.input_channels / 16u;
  const uint group_stride = 9u * p.input_channels * 8u, spatial_stride = p.input_channels * 8u;
  device const half* wgroup = packed_weights + output_group * group_stride + n * 16u;
  for (uint tile = 0u; tile < 9u; ++tile) {
    const int iy = int(py) + int(tile / 3u) - 1, ix = int(px) + int(tile % 3u) - 1;
    const bool ok = iy >= 0 && iy < int(p.height) && ix >= 0 && ix < int(p.width);
    device const half* wtile = wgroup + tile * spatial_stride;
    device const half* in = input + (ok ? (uint(iy) * p.width + uint(ix)) * p.input_channels : 0u);
    for (uint s = 0u; s < stages; ++s) {
      acc = half(stage16(in + s * 16u, wtile + s * 128u, ok, float(acc)));
    }
  }
  acc = max(acc, half(acc * half(0.01f)));
  output[(py * p.width + px) * p.output_channels + oc] = acc;
}
