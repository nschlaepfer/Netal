#include <metal_stdlib>
using namespace metal;
// Bilinear 2x upsample pre-kernels with the CONNECTED block's formula (src/fg_connected_custom_l56_l83/
// connected_*.metal), which is what matched the RTX 5090 byte for byte across launches 56-83: horizontal
// blend as one half fma per row, vertical blend as one half fma. The custom_* scalar kernels use a
// two-rounding form; the two differ in the last bit, so a bottom pipeline wired into the block must
// materialise this form. Compile with -fno-fast-math -ffp-contract=off so wx1*b stays a rounded multiply.
//
// One thread per output pixel, all channels as half4 (componentwise IEEE, so each channel's result is
// the scalar formula's bit for bit). The previous one-thread-per-(pixel, channel) form wrote 16-halves
// apart across a threadgroup and ran the three 640x368x16 planes at ~30 GB/s; this form streams.
#define CONNECTED_UP4(NAME, OW, OH, SW, SH, C)                                                                 \
kernel void NAME(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]], uint2 g [[thread_position_in_grid]]) { \
  if (g.x >= uint(OW) || g.y >= uint(OH)) return;                                                             \
  const float fx = (float(g.x) - 0.5f) * 0.5f, fy = (float(g.y) - 0.5f) * 0.5f;                               \
  const uint x0 = fx < 0 ? 0u : min(uint(fx), uint(SW - 1)), x1 = fx < 0 ? 0u : min(uint(fx + 1.0f), uint(SW - 1)); \
  const uint y0 = fy < 0 ? 0u : min(uint(fy), uint(SH - 1)), y1 = fy < 0 ? 0u : min(uint(fy + 1.0f), uint(SH - 1)); \
  const half wx1 = half(fx - float(x0)), wx0 = half((float(x0) - fx) + 1.0f);                                 \
  const half wy1 = half(fy - float(y0)), wy0 = half((float(y0) - fy) + 1.0f);                                 \
  device const half4* A = reinterpret_cast<device const half4*>(in + (y0 * SW + x0) * C);                     \
  device const half4* B = reinterpret_cast<device const half4*>(in + (y0 * SW + x1) * C);                     \
  device const half4* D = reinterpret_cast<device const half4*>(in + (y1 * SW + x0) * C);                     \
  device const half4* E = reinterpret_cast<device const half4*>(in + (y1 * SW + x1) * C);                     \
  device half4* O = reinterpret_cast<device half4*>(ups + (g.y * OW + g.x) * C);                              \
  _Pragma("clang loop unroll(full)")                                                                          \
  for (uint q = 0u; q < uint(C) / 4u; ++q) {                                                                   \
    const half4 a = A[q], b = B[q], d = D[q], e = E[q];                                                        \
    const half4 rb = half4(wx1) * b; const half4 top = fma(half4(wx0), a, rb);                                 \
    const half4 re = half4(wx1) * e; const half4 bot = fma(half4(wx0), d, re);                                 \
    const half4 lower = half4(wy1) * bot; O[q] = fma(half4(wy0), top, lower);                                  \
  }                                                                                                           \
}
CONNECTED_UP4(fg_connected_block0_bot0_upsample, 160, 92, 80, 46, 64u)
CONNECTED_UP4(fg_connected_block0_bot1_upsample, 320, 184, 160, 92, 32u)
CONNECTED_UP4(fg_connected_block1_bot0_upsample, 320, 184, 160, 92, 32u)
CONNECTED_UP4(fg_connected_block1_bot1_upsample, 640, 368, 320, 184, 16u)
