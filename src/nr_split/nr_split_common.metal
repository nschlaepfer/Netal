// NR split-Swin 16-head family on Metal: shared geometry and layouts.
//
// Every formula was pinned against the RTX 5090 FP8 pool oracle through the PTX warp emulator
// (scripts/ptx_warp_emu.py) and the numpy reference (scripts/nr_split_ref.py). The numerics helpers
// (E4M3 codec, QMMA step, weight layout, A-identity token layout, cfd swizzle, slot permutation, RTX
// rsqrt/rcp tables) are shared with the ViT-1D family (nr_vit_common.metal).
// Compile with -fno-fast-math -ffp-contract=off.
#pragma once
#include "nr_vit_common.metal"

// Token grid: 60 wide x 36 tall, 4x4 spatial tiles (15 per row, 9 rows), row-major slots inside a tile,
// 8192 bytes per tile (16 tokens x 512 channels, A-identity layout). token = tile*16 + slot.
constant uint NR_SPLIT_W = 60u, NR_SPLIT_H = 36u, NR_SPLIT_TILES_X = 15u, NR_SPLIT_TILES_Y = 9u, NR_SPLIT_TOKENS = 2160u;
// Pooled / ViT grid: 30 x 18 valid tokens inside a 32 x 20 padded grid, 8 tiles per row, 5 tile rows.
constant uint NR_POOL_W = 30u, NR_POOL_H = 18u, NR_POOL_TILES_X = 8u, NR_POOL_TOKENS = 640u;

struct NrSplitParams { int shift_x; int shift_y; uint flags; uint tokens; };
constant uint NR_FLAG_RES_VIEW = 1u;     // residual read from the space-to-depth view instead of the tin layout
constant uint NR_FLAG_OUT_VIEW = 2u;     // output written to the view
constant uint NR_FLAG_EMIT_F16 = 4u;     // also write the pre-rounding half result (for the 2x2 pool)
constant uint NR_FLAG_IN_VIEW = 8u;      // ffwd input read from the view

inline uint nr_tin(uint token, uint c) { return nr_a_index(token, c, 512u); }
inline void nr_token_rc(uint token, thread uint& r, thread uint& c) {
  const uint tile = token / 16u, slot = token % 16u;
  r = 4u * (tile / NR_SPLIT_TILES_X) + slot / 4u; c = 4u * (tile % NR_SPLIT_TILES_X) + slot % 4u;
}
inline uint nr_token_of(uint r, uint c) { return ((r / 4u) * NR_SPLIT_TILES_X + c / 4u) * 16u + 4u * (r % 4u) + c % 4u; }
// 2x2 space-to-depth view: [cg = ch/32][R = r/2 + 18*((ch%32)/16)][C = c/2 + 30*(r%2)][ch%16 + 16*(c%2)]
inline uint nr_view_index(uint r, uint c, uint ch) {
  return (ch / 32u) * 69120u + (r / 2u + 18u * ((ch % 32u) / 16u)) * 1920u + (c / 2u + 30u * (r % 2u)) * 32u + (ch % 16u) + 16u * (c % 2u);
}
inline half nr_load_token(device const uchar* buf, uint token, uint ch, bool view) {
  if (!view) return nr_e4m3(buf[nr_tin(token, ch)]);
  uint r, c; nr_token_rc(token, r, c); return nr_e4m3(buf[nr_view_index(r, c, ch)]);
}
// pooled-grid token: 8 tiles per row of the 32-wide padded grid, row-major slots; 8192 bytes per tile (512 channels)
inline uint nr_pool_token_of(uint R, uint C) { return ((R / 4u) * NR_POOL_TILES_X + C / 4u) * 16u + 4u * (R % 4u) + C % 4u; }
inline void nr_pool_token_rc(uint token, thread uint& R, thread uint& C) {
  const uint tile = token / 16u, slot = token % 16u;
  R = 4u * (tile / NR_POOL_TILES_X) + slot / 4u; C = 4u * (tile % NR_POOL_TILES_X) + slot % 4u;
}
// weights: a dense [K][N] E4M3 matrix stored as [k-step][N/64 groups of 2048][weight_index]; group of 32 cols = 1024 bytes
inline half nr_wload(device const uchar* blob, uint step, uint step_bytes, uint n, uint j) {
  return nr_e4m3(blob[step * step_bytes + (n / 64u) * 2048u + nr_weight_index(j, n % 64u)]);
}
inline half nr_round_e4m3(half x) { return nr_e4m3(nr_to_e4m3(x)); }
inline half nr_cubic_silu(half h) {
  const half tt = min(max(h, nr_hbits(0xc400u)), nr_hbits(0x4400u));
  const half u = fma(nr_hbits(0xab28u), abs(tt), nr_hbits(0x3728u));
  const half sv = fma(tt, u, nr_hbits(0x3b28u));
  return half(h * sv);
}
inline half nr_tree8_exact(half x0, half x1, half x2, half x3, half x4, half x5, half x6, half x7) {
  const half s01 = x0 + x1, s23 = x2 + x3, s45 = x4 + x5, s67 = x6 + x7;
  const half s0123 = s01 + s23;
  const half s012345 = s0123 + s45;
  return s012345 + s67;
}
