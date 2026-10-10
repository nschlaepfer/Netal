#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;

// SIMD rewrite of the block0/block1 conv2 family - 16 of the 25 custom launches, and the largest
// single share of the 74.8 ms custom block.
//
// The scalar kernel reads 36 stages x 32 halfwords per output element, about 543 MB for one launch,
// which puts it past the M3 Max's bandwidth roof before arithmetic is considered. A simdgroup shares
// its operands across 32 lanes, so the same work reads a fraction of that. This is the lever; the
// matrix unit is incidental.
//
// 80x46 spatial, 64 in, 64 out, 9 tiles x 4 channel stages in tile-major order, leaky 0.01, clamp
// to +/-6, optional residual.

struct Params { uint residual; };

inline uint widx(uint tile, uint stage, uint ic, uint oc) {
  uint g = oc / 8u, n = oc % 8u, k = ic % 16u;
  uint lane = n * 4u + (k % 8u) / 2u;
  uint frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * 4608u + tile * 512u + stage * 128u + (frag >= 2u ? 64u : 0u) + lane * 2u + (frag & 1u);
}

inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u),
               2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}

kernel void fg_custom_block0_conv2_simd(
    device const half* w [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device const half* residual [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant Params& p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint2 element = matrix_element(lane);
  const uint matrix_row = element.x, matrix_column = element.y;
  const uint pixel = group.x * 8u + matrix_row;
  const uint oc = group.y * 8u + matrix_column;
  if (pixel >= 80u * 46u || oc >= 64u) return;
  const uint px = pixel % 80u, py = pixel / 80u;

  simdgroup_float8x8 acc;
  acc.thread_elements()[0] = float(bias[oc]);
  acc.thread_elements()[1] = float(bias[oc + 1u]);

  for (uint tile = 0u; tile < 9u; ++tile) {
    const int ix = int(px) + int(tile % 3u) - 1;
    const int iy = int(py) + int(tile / 3u) - 1;
    const bool ok = ix >= 0 && ix < 80 && iy >= 0 && iy < 46;
    const uint base = ok ? (uint(iy) * 80u + uint(ix)) * 64u : 0u;
    for (uint stage = 0u; stage < 4u; ++stage) {
      simdgroup_float8x8 a[2], b[2];
#pragma clang loop unroll(full)
      for (uint kb = 0u; kb < 2u; ++kb) {
        const uint ic_a = stage * 16u + kb * 8u + matrix_column;
        a[kb].thread_elements()[0] = ok ? float(input[base + ic_a]) : 0.0f;
        a[kb].thread_elements()[1] = ok ? float(input[base + ic_a + 1u]) : 0.0f;
        const uint ic_b = stage * 16u + kb * 8u + matrix_row;
        b[kb].thread_elements()[0] = float(w[widx(tile, stage, ic_b, oc)]);
        b[kb].thread_elements()[1] = float(w[widx(tile, stage, ic_b, oc + 1u)]);
      }
      simdgroup_multiply_accumulate(acc, a[0], b[0], acc);
      simdgroup_multiply_accumulate(acc, a[1], b[1], acc);
      // NVIDIA's f16 accumulate rounds back to half after every K=16 stage.
      acc.thread_elements()[0] = float(half(acc.thread_elements()[0]));
      acc.thread_elements()[1] = float(half(acc.thread_elements()[1]));
    }
  }

  const half slope = half(0.01f);
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) {
    half v = half(acc.thread_elements()[h]);
    v = min(max(max(v, half(v * slope)), half(-6.0f)), half(6.0f));
    const uint index = (py * 80u + px) * 64u + oc + h;
    if (oc + h < 64u) {
      const half r = p.residual != 0u ? residual[index] : half(0.0f);
      out[index] = half(v + r);
    }
  }
}
