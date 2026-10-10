// Padded-grid helpers for the PWin pipeline. Every per-token phase runs on the padded grid
// P = (H + 8) x (W + 8) (window shift 4 on each side): padded index t = (y + 4) * (W + 8) + (x + 4).
// Padded positions carry the values the transpiled kernel computed for them (reflected inputs,
// their own window-block split rule), so the window mixing gathers without reflection.
#include <metal_stdlib>
using namespace metal;

struct PadParams {
  uint width, height;      // image token grid
  uint in_stride;          // halves per input token row (arena)
  uint out_stride;         // halves per padded-grid row
  uint channels;           // halves copied per token (multiple of 8)
  uint h_stride;           // d2s: halves per half-res token row (128)
  uint s_stride;           // d2s: halves per skip row (32)
  uint pad0;
};

static inline int pad_reflect(int v, int n) {
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}

// enc0 input: X_pad[t, 0:C] = In[reflect(y), reflect(x), 0:C]
kernel void pwin_pad_gather(device const half* In [[buffer(0)]],
                            device half* Out [[buffer(1)]],
                            constant PadParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint Wp = p.width + 8u, Hp = p.height + 8u;
  const uint chunks = p.channels / 8u;
  const uint t = gid / chunks, part = gid - t * chunks;
  if (t >= Wp * Hp) return;
  const int y = int(t / Wp) - 4, x = int(t - (t / Wp) * Wp) - 4;
  const int ry = pad_reflect(y, int(p.height)), rx = pad_reflect(x, int(p.width));
  device const half* src = In + (uint(ry) * p.width + uint(rx)) * p.in_stride + part * 8u;
  device half* dst = Out + t * p.out_stride + part * 8u;
  *reinterpret_cast<device half4*>(dst) = *reinterpret_cast<device const half4*>(src);
  *reinterpret_cast<device half4*>(dst + 4) = *reinterpret_cast<device const half4*>(src + 4);
}

// dec0 input: x_pad[t, c] = fp16( H0[reflect_half(floor(y/2)), reflect_half(floor(x/2))][32 q + c] + skip[reflect(y), reflect(x)][c] ),
// q = 2 (y & 1) + (x & 1) computed on the padded (signed) coordinate
kernel void pwin_pad_d2s_add(device const half* H0 [[buffer(0)]],
                             device const half* Skip [[buffer(1)]],
                             device half* Out [[buffer(2)]],
                             constant PadParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
  const uint Wp = p.width + 8u, Hp = p.height + 8u;
  const uint t = gid >> 2u, part = gid & 3u;
  if (t >= Wp * Hp) return;
  const int y = int(t / Wp) - 4, x = int(t - (t / Wp) * Wp) - 4;
  const int hy = pad_reflect(y >> 1, int(p.height) >> 1), hx = pad_reflect(x >> 1, int(p.width) >> 1);   // arithmetic shift = floor
  const uint q = 2u * uint(y & 1) + uint(x & 1);
  const int ry = pad_reflect(y, int(p.height)), rx = pad_reflect(x, int(p.width));
  device const half* h = H0 + (uint(hy) * (p.width >> 1u) + uint(hx)) * p.h_stride + 32u * q + part * 8u;
  device const half* s = Skip + (uint(ry) * p.width + uint(rx)) * p.s_stride + part * 8u;
  device half* o = Out + t * p.out_stride + part * 8u;
  const half4 h0 = *reinterpret_cast<device const half4*>(h), h1 = *reinterpret_cast<device const half4*>(h + 4);
  const half4 s0 = *reinterpret_cast<device const half4*>(s), s1 = *reinterpret_cast<device const half4*>(s + 4);
  *reinterpret_cast<device half4*>(o) = h0 + s0;
  *reinterpret_cast<device half4*>(o + 4) = h1 + s1;
}

// copy real tokens from a padded-grid tensor into an image-layout tensor (clip), C halves per token
kernel void pwin_pad_store(device const half* In [[buffer(0)]],
                           device half* Out [[buffer(1)]],
                           constant PadParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  const uint chunks = p.channels / 8u;
  const uint t = gid / chunks, part = gid - t * chunks;
  if (t >= p.width * p.height) return;
  const uint y = t / p.width, x = t - y * p.width;
  const uint tp = (y + 4u) * (p.width + 8u) + x + 4u;
  device const half* src = In + tp * p.in_stride + part * 8u;
  device half* dst = Out + t * p.out_stride + part * 8u;
  *reinterpret_cast<device half4*>(dst) = *reinterpret_cast<device const half4*>(src);
  *reinterpret_cast<device half4*>(dst + 4) = *reinterpret_cast<device const half4*>(src + 4);
}

// 2x2 merge from the padded grid: A[st, 32 q + c] = X_pad[(2 sy + dy, 2 sx + dx)][c]
kernel void pwin_pad_merge2x2(device const half* X [[buffer(0)]],
                              device half* A [[buffer(1)]],
                              constant PadParams& p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  const uint sw = p.width / 2u, sh = p.height / 2u;
  const uint st = gid >> 2u, q = gid & 3u;
  if (st >= sw * sh) return;
  const uint sy = st / sw, sx = st - sy * sw;
  const uint y = 2u * sy + (q >> 1u), x = 2u * sx + (q & 1u);
  const uint tp = (y + 4u) * (p.width + 8u) + x + 4u;
  device const half* src = X + tp * p.in_stride;
  device half* dst = A + st * p.out_stride + q * 32u;
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 32u; c += 4u) *reinterpret_cast<device half4*>(dst + c) = *reinterpret_cast<device const half4*>(src + c);
}
