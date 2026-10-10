#include <metal_stdlib>
using namespace metal;
// Bilinear 2x upsample pre-kernels with the CONNECTED block's formula (src/fg_connected_custom_l56_l83/
// connected_*.metal), which is what matched the RTX 5090 byte for byte across launches 56-83: horizontal
// blend as one half fma per row, vertical blend as one half fma. The custom_* scalar kernels use a
// two-rounding form; the two differ in the last bit, so a bottom pipeline wired into the block must
// materialise this form. Compile with -fno-fast-math -ffp-contract=off so wx1*b stays a rounded multiply.
#define CONNECTED_UP(NAME, OW, OH, SW, SH, C)                                                                  \
inline half NAME(device const half* in, int x, int y, uint c) {                                               \
  if (x < 0 || x >= OW || y < 0 || y >= OH) return half(0);                                                   \
  float fx = (float(x) - 0.5f) * 0.5f, fy = (float(y) - 0.5f) * 0.5f;                                         \
  uint x0 = fx < 0 ? 0u : min(uint(fx), uint(SW - 1)), x1 = fx < 0 ? 0u : min(uint(fx + 1.0f), uint(SW - 1)); \
  uint y0 = fy < 0 ? 0u : min(uint(fy), uint(SH - 1)), y1 = fy < 0 ? 0u : min(uint(fy + 1.0f), uint(SH - 1)); \
  half wx1 = half(fx - float(x0)), wx0 = half((float(x0) - fx) + 1.0f);                                       \
  half wy1 = half(fy - float(y0)), wy0 = half((float(y0) - fy) + 1.0f);                                       \
  half a = in[(y0 * SW + x0) * C + c], b = in[(y0 * SW + x1) * C + c];                                        \
  half d = in[(y1 * SW + x0) * C + c], e = in[(y1 * SW + x1) * C + c];                                        \
  half rb = wx1 * b; half top = fma(wx0, a, rb);                                                              \
  half re = wx1 * e; half bot = fma(wx0, d, re);                                                              \
  half lower = wy1 * bot; return fma(wy0, top, lower);                                                        \
}
CONNECTED_UP(up_b0_bot0, 160, 92, 80, 46, 64u)
CONNECTED_UP(up_b0_bot1, 320, 184, 160, 92, 32u)
CONNECTED_UP(up_b1_bot0, 320, 184, 160, 92, 32u)
CONNECTED_UP(up_b1_bot1, 640, 368, 320, 184, 16u)
kernel void fg_connected_block0_bot0_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 160u || g.y >= 92u || g.z >= 64u) return; ups[(g.y * 160u + g.x) * 64u + g.z] = up_b0_bot0(in, int(g.x), int(g.y), g.z); }
kernel void fg_connected_block0_bot1_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 32u) return; ups[(g.y * 320u + g.x) * 32u + g.z] = up_b0_bot1(in, int(g.x), int(g.y), g.z); }
kernel void fg_connected_block1_bot0_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 320u || g.y >= 184u || g.z >= 32u) return; ups[(g.y * 320u + g.x) * 32u + g.z] = up_b1_bot0(in, int(g.x), int(g.y), g.z); }
kernel void fg_connected_block1_bot1_upsample(device const half* in [[buffer(0)]], device half* ups [[buffer(1)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 640u || g.y >= 368u || g.z >= 16u) return; ups[(g.y * 640u + g.x) * 16u + g.z] = up_b1_bot1(in, int(g.x), int(g.y), g.z); }
