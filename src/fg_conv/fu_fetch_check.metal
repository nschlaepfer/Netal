#include <metal_stdlib>
#include "fu_upsample.metalh"
using namespace metal;

// Operand-level check for the fused bilinear fetch: evaluate fu_fetch1 / fu_fetch2 / fu_fetch4 at every element of
// an upsampled grid from its low-resolution source, so the host can compare the bytes with the tensor the
// materialising pre-kernel (connected_upsamples.metal) wrote from the same source. The fused kernels' in-grid
// operands are exactly these; out-of-grid taps select zero in both designs by the same unchanged `ok` test.
struct FuCheckDims { uint OW, OH, SW, SH, C; };

kernel void fg_fu_fetch_check1(device const half* src [[buffer(0)]], device half* out [[buffer(1)]],
                               constant FuCheckDims& d [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= d.OW || g.y >= d.OH) return;
  const FuTap T = fu_tap(true, int(g.x), int(g.y), d.SW, d.SH, d.C);
  const uint base = (g.y * d.OW + g.x) * d.C;
  for (uint c = 0u; c < d.C; ++c) out[base + c] = fu_fetch1(src, T, c);
}

kernel void fg_fu_fetch_check2(device const half* src [[buffer(0)]], device half* out [[buffer(1)]],
                               constant FuCheckDims& d [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= d.OW || g.y >= d.OH) return;
  const FuTap T = fu_tap(true, int(g.x), int(g.y), d.SW, d.SH, d.C);
  const uint base = (g.y * d.OW + g.x) * d.C;
  for (uint c = 0u; c < d.C; c += 2u) { const half2 v = fu_fetch2(src, T, c); out[base + c] = v.x; out[base + c + 1u] = v.y; }
}

kernel void fg_fu_fetch_check4(device const half* src [[buffer(0)]], device half* out [[buffer(1)]],
                               constant FuCheckDims& d [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= d.OW || g.y >= d.OH) return;
  const FuTap T = fu_tap(true, int(g.x), int(g.y), d.SW, d.SH, d.C);
  const uint base = (g.y * d.OW + g.x) * d.C;
  for (uint c = 0u; c < d.C; c += 4u) {
    const half4 v = fu_fetch4(src, T, c);
    out[base + c] = v.x; out[base + c + 1u] = v.y; out[base + c + 2u] = v.z; out[base + c + 3u] = v.w;
  }
}
