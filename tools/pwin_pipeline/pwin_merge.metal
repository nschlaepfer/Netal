// 2x2 patch merge gather for the PWin output projection: for every super-token (sy, sx) of the
// half-resolution grid, A[st, :] = concat(X[2sy, 2sx], X[2sy, 2sx+1], X[2sy+1, 2sx], X[2sy+1, 2sx+1])
// (32 channels each, K = 128). The projection itself is the generic per-token GEMM.
#include <metal_stdlib>
using namespace metal;

struct MergeParams {
  uint width, height;   // full-resolution token grid (even)
  uint x_stride;        // halves per token row of X
  uint a_stride;        // halves per super-token row of A (128)
};

kernel void pwin_merge2x2(device const half* X [[buffer(0)]],
                          device half* A [[buffer(1)]],
                          constant MergeParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint sw = p.width / 2u, sh = p.height / 2u;
  const uint st = gid >> 2u;            // super-token
  const uint q = gid & 3u;              // sub-token (dy, dx) = (q >> 1, q & 1)
  if (st >= sw * sh) return;
  const uint sy = st / sw, sx = st - sy * sw;
  const uint y = 2u * sy + (q >> 1u), x = 2u * sx + (q & 1u);
  device const half* src = X + (y * p.width + x) * p.x_stride;
  device half* dst = A + st * p.a_stride + q * 32u;
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 32u; c += 2u) {
    *reinterpret_cast<device half2*>(dst + c) = *reinterpret_cast<device const half2*>(src + c);
  }
}

// Same gather with a channel count (enc1: 64 channels per sub-token, K = 256).
struct MergeCParams {
  uint width, height, x_stride, a_stride, channels, pad0;
};

kernel void pwin_merge2x2c(device const half* X [[buffer(0)]],
                           device half* A [[buffer(1)]],
                           constant MergeCParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  const uint sw = p.width / 2u, sh = p.height / 2u;
  const uint st = gid >> 2u, q = gid & 3u;
  if (st >= sw * sh) return;
  const uint sy = st / sw, sx = st - sy * sw;
  const uint y = 2u * sy + (q >> 1u), x = 2u * sx + (q & 1u);
  device const half* src = X + (y * p.width + x) * p.x_stride;
  device half* dst = A + st * p.a_stride + q * p.channels;
  for (uint c = 0u; c < p.channels; c += 4u) *reinterpret_cast<device half4*>(dst + c) = *reinterpret_cast<device const half4*>(src + c);
}
