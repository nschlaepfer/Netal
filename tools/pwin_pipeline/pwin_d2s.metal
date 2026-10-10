// Depth-to-space plus skip add for the decoder input (dec0 P0 -> x):
//   x[t, c] = fp16( H0[st, 32q + c] + skip[t, c] ),  t = (y, x) full-res token, st = (y>>1, x>>1) half-res token,
//   q = 2*(y&1) + (x&1)  (channels 32q..32q+31 of the 128-channel half-res projection belong to sub-pixel (dy, dx))
#include <metal_stdlib>
using namespace metal;

struct D2sParams {
  uint width, height;    // full-res token grid
  uint h_stride;         // halves per half-res token row of H0 (128)
  uint s_stride;         // halves per token row of skip (32)
  uint o_stride;         // halves per token row of the output (32)
  uint pad0, pad1, pad2;
};

kernel void pwin_d2s_add(device const half* H0 [[buffer(0)]],
                         device const half* Skip [[buffer(1)]],
                         device half* Out [[buffer(2)]],
                         constant D2sParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint t = gid >> 2u;               // token
  const uint part = gid & 3u;             // 8-channel chunk
  if (t >= p.width * p.height) return;
  const uint y = t / p.width, x = t - y * p.width;
  const uint st = (y >> 1u) * (p.width >> 1u) + (x >> 1u);
  const uint q = 2u * (y & 1u) + (x & 1u);
  device const half* h = H0 + st * p.h_stride + 32u * q + part * 8u;
  device const half* s = Skip + t * p.s_stride + part * 8u;
  device half* o = Out + t * p.o_stride + part * 8u;
  const half4 h0 = *reinterpret_cast<device const half4*>(h), h1 = *reinterpret_cast<device const half4*>(h + 4);
  const half4 s0 = *reinterpret_cast<device const half4*>(s), s1 = *reinterpret_cast<device const half4*>(s + 4);
  *reinterpret_cast<device half4*>(o) = h0 + s0;
  *reinterpret_cast<device half4*>(o + 4) = h1 + s1;
}
