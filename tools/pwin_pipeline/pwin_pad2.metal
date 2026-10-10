// General padded-grid helpers (padded grid = the windows of a layer tiled exactly: Wp = 8*gx, Hp = 8*gy,
// padded index p = (y + pady) * Wp + (x + padx) for the signed image coordinate (x, y)).
// Used by the decoder layers whose per-token inputs depend on the padded position itself (the half-resolution
// parent is floor(y/2) of the padded coordinate, so a reflected token is NOT equal to its in-image source).
#include <metal_stdlib>
using namespace metal;

struct Pad2Params {
  uint width, height;        // image token grid
  uint wp, hp;               // padded grid
  int padx, pady;            // padded origin: padded (0,0) is image (-padx, -pady)
  uint channels;             // halves per token copied / added
  uint in_stride;            // halves per row of the parent tensor (P0: 512)
  uint skip_stride;          // halves per row of the skip tensor
  uint out_stride;           // halves per row of the padded output
  uint pad0, pad1;
};

static inline int pad2_reflect(int v, int n) {   // reflN(i) = min(|i|, 2n - 2 - |i|)
  int a = v < 0 ? -v : v;
  int b = (n << 1) - a - 2;
  return b < a ? b : a;
}
static inline int pad2_reflect_half(int v, int n_even) {   // half-res: min(|i|, (H & ~1) - 2 - |i|)
  int a = v < 0 ? -v : v;
  int b = n_even - 2 - a;
  return b < a ? b : a;
}

// x_pad[p][c] = fp16( P0[reflH2(y>>1), reflW2(x>>1)][channels * q + c] + skip[reflH(y), reflW(x)][c] ),
// q = 2 (y & 1) + (x & 1) on the signed padded coordinate (arithmetic shift = floor).
kernel void pwin_padg_d2s_skip(device const half* P0 [[buffer(0)]],
                               device const half* Skip [[buffer(1)]],
                               device half* Out [[buffer(2)]],
                               constant Pad2Params& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  const uint chunks = p.channels / 8u;
  const uint t = gid / chunks, part = gid - t * chunks;
  if (t >= p.wp * p.hp) return;
  const uint py = t / p.wp, px = t - py * p.wp;
  const int y = int(py) - p.pady, x = int(px) - p.padx;
  const int Y = y >> 1, X = x >> 1;                   // floor
  const uint q = uint(2 * (y & 1) + (x & 1));
  const int hy = pad2_reflect_half(Y, int(p.height & ~1u));
  const int hx = pad2_reflect_half(X, int(p.width & ~1u));
  const int sy = pad2_reflect(y, int(p.height)), sx = pad2_reflect(x, int(p.width));
  device const half* a = P0 + (uint(hy) * (p.width >> 1) + uint(hx)) * p.in_stride + q * p.channels + part * 8u;
  device const half* b = Skip + (uint(sy) * p.width + uint(sx)) * p.skip_stride + part * 8u;
  device half* o = Out + t * p.out_stride + part * 8u;
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 8u; c += 4u) {
    const half4 av = *reinterpret_cast<device const half4*>(a + c);
    const half4 bv = *reinterpret_cast<device const half4*>(b + c);
    *reinterpret_cast<device half4*>(o + c) = av + bv;
  }
}

// copy the in-image tokens of a padded tensor into an image-layout tensor
kernel void pwin_padg_store(device const half* In [[buffer(0)]],
                            device half* Out [[buffer(1)]],
                            constant Pad2Params& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint chunks = p.channels / 8u;
  const uint t = gid / chunks, part = gid - t * chunks;
  if (t >= p.width * p.height) return;
  const uint y = t / p.width, x = t - y * p.width;
  const uint tp = (y + uint(p.pady)) * p.wp + x + uint(p.padx);
  device const half* a = In + tp * p.in_stride + part * 8u;
  device half* o = Out + t * p.out_stride + part * 8u;
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 8u; c += 4u)
    *reinterpret_cast<device half4*>(o + c) = *reinterpret_cast<device const half4*>(a + c);
}
