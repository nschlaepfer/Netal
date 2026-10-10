// nr_tex_sample.metal — ONE reusable fractional-sampling primitive for both endpoints (Block0 history read at pre +8, post 155 history/motion
// taps). Status (2026-09-07): nonfinite rule added from Codex's RTX edge oracle (CPU 4312/4312; Metal branch NOT yet validated on the GPU). Status (2026-09-06 14:26 UTC): nr_tex2d_bilinear_rtx below is FITTED to Codex's RTX step-texture micro-probe (4 x 1212 values) AND finite-half capture (4 x 5524 values), 0 unequal on CPU and GPU (build/nr_temporal/tex-rtx-fit-v2;
// build/nr_temporal/tex-rtx-fit-v1); the q8 variants are retained as the rejected hypotheses. The audited zero-motion reference puts every history tap on a texel centre (weights (0,1,0,0), see
// scripts/test-nr-tex-sample.py), so it validates coordinate conversion, clamping and centre fetches but NOT the fractional weights. The
// weight-quantisation rule (NR_TEX_Q8_MODE) must be pinned against a nonzero-motion RTX capture before either endpoint relies on it.
//   - NVIDIA texture units compute the bilinear weights from the fractional texel coordinate at 8-bit precision (256 sub-texel steps);
//     whether the fraction is truncated (mode 0) or rounded (mode 1) to that grid, and the exact blend order, are the open parameters.
//   - Normalized coordinates, clamp-to-edge addressing, linear filter: as recorded for history view 1 and motion view 6.
#pragma once
#include <metal_stdlib>
using namespace metal;
#ifndef NR_TEX_Q8_MODE
#define NR_TEX_Q8_MODE 0   // 0 = truncate the fraction to 1/256 steps, 1 = round to nearest 1/256 step; both are hypotheses
#endif
struct NrTexCoord { int x0, y0, x1, y1; float fx, fy; };
// normalized uv -> texel space (u*W - 0.5), clamp-to-edge integer neighbours, quantised fractions
inline NrTexCoord nr_tex_coord_q8(float u, float v, uint w, uint h) {
  const float tx = u * float(w) - 0.5f, ty = v * float(h) - 0.5f;
  const float fx0 = floor(tx), fy0 = floor(ty); float fx = tx - fx0, fy = ty - fy0;
#if NR_TEX_Q8_MODE
  fx = rint(fx * 256.0f) / 256.0f; fy = rint(fy * 256.0f) / 256.0f;
#else
  fx = floor(fx * 256.0f) / 256.0f; fy = floor(fy * 256.0f) / 256.0f;
#endif
  const int W = int(w), H = int(h); NrTexCoord c;
  c.x0 = clamp(int(fx0), 0, W - 1); c.x1 = clamp(int(fx0) + 1, 0, W - 1); c.y0 = clamp(int(fy0), 0, H - 1); c.y1 = clamp(int(fy0) + 1, 0, H - 1); c.fx = fx; c.fy = fy; return c;
}
// fixed blend order: rows first (x lerp), then y, in the lerp form a + (b - a) * f so that equal neighbours (clamped edges) reproduce the texel exactly
// (the two-weight form a*(1-f) + b*f does not); texels read exactly (no sampler involved)
template <typename T> inline float4 nr_tex2d_bilinear_q8(texture2d<T, access::read> t, float u, float v) {
  const NrTexCoord c = nr_tex_coord_q8(u, v, t.get_width(), t.get_height());
  const float4 a = float4(t.read(uint2(c.x0, c.y0))), b = float4(t.read(uint2(c.x1, c.y0))), d = float4(t.read(uint2(c.x0, c.y1))), e = float4(t.read(uint2(c.x1, c.y1)));
  const float4 top = a + (b - a) * c.fx, bot = d + (e - d) * c.fx; return top + (bot - top) * c.fy;
}
// the Metal hardware sampler path with the recorded modes, for A/B against the software primitive
template <typename T> inline float4 nr_tex2d_bilinear_hw(texture2d<T, access::sample> t, float u, float v) {
  constexpr sampler smp(coord::normalized, filter::linear, address::clamp_to_edge); return float4(t.sample(smp, float2(u, v)));
}

// ---- fitted RTX rule (2026-09-06; Codex's step-texture micro-probe build/nr_tex_micro, 4 runs x 1212 values, 0 unequal with this rule) ----
// coordinate: the normalised coordinate is truncated to 21 fractional bits; texel coordinate = U*dim/2^21 - 0.5 in exact integer arithmetic;
// the 8-bit fraction is rounded half up and may be 256; clamp-to-edge on the two texel indices. weights: wd = (fx*fy + 128) >> 8 (the only
// rounding), wb = fx - wd, wc = fy - wd, wa = 256 - fx - fy + wd (sum 256). Accumulation: wa*a, fma(wb, b), fma(wc, c), fma(wd, d), * 1/256 -
// this order/precision is a HYPOTHESIS the 0/1 probe texture cannot see; it is mirrored bit for bit by the emulator (tex_bilinear_rtx).
struct NrTexRtxCoord { int i, f; };
inline NrTexRtxCoord nr_tex_coord_rtx(float u, int dim) {
  u = isnan(u) ? 0.0f : clamp(u, -1.0f, 2.0f);                        // outside [0,1) both neighbours clamp to the same edge texel; NaN as 0 (hypothesis)
  const long U = long(floor(u * 2097152.0f));                         // u * 2^21 is exact (power-of-two scale), floor exact
  const long T = U * long(dim) - 1048576L;                             // (u_q * dim - 0.5) in units of 2^-21
  const long i = T >> 21;                                              // floor (arithmetic shift)
  const long frac = T - (i << 21);
  NrTexRtxCoord c; c.i = int(i); c.f = int((frac * 256L + 1048576L) >> 21); return c;
}
template <typename T, access A> inline float4 nr_tex2d_bilinear_rtx(texture2d<T, A> t, float u, float v) {
  const int W = int(t.get_width()), H = int(t.get_height()); const NrTexRtxCoord cx = nr_tex_coord_rtx(u, W), cy = nr_tex_coord_rtx(v, H);
  const int x0 = clamp(cx.i, 0, W - 1), x1 = clamp(cx.i + 1, 0, W - 1), y0 = clamp(cy.i, 0, H - 1), y1 = clamp(cy.i + 1, 0, H - 1);
  const int wd = (cx.f * cy.f + 128) >> 8, wb = cx.f - wd, wc = cy.f - wd, wa = 256 - cx.f - cy.f + wd;
  const float4 a = float4(t.read(uint2(x0, y0))), b = float4(t.read(uint2(x1, y0))), c = float4(t.read(uint2(x0, y1))), d = float4(t.read(uint2(x1, y1)));
  // accumulation fitted to Codex's finite-half capture (5,524/5,524; build/nr_temporal/tex-rtx-fit-v2): nonzero-weight texels aligned to the exponent of the
  // largest one on a 15-significant-bit grid and truncated toward zero, exact integer weighted sum, result rounded to half with ties away from zero.
  // Zero-weight taps do not contribute; an all-negative-zero set returns -0 (history copy 156); cancellation to exactly zero returns +0 (hypothesis).
  float4 r;
  for (int ch = 0; ch < 4; ++ch) {
    const float t[4] = {a[ch], b[ch], c[ch], d[ch]}; const int w[4] = {wa, wb, wc, wd};
    // exceptional operands (Codex's RTX sampler edge oracle, 2026-09-07 00:01 UTC, 22 cases x 49 queries, 4 processes identical; CPU rule 4312/4312):
    // over the NONZERO-weight taps only - any NaN -> canonical NaN 0x7fffe000; +inf and -inf together -> the same NaN; a lone infinity -> that infinity.
    // Zero-weight taps never contaminate the result. The finite path below is unchanged. (Metal validation of this branch pending: see journal.)
    bool anynan = false, pinf = false, ninf = false;
    for (int i = 0; i < 4; ++i) if (w[i]) { anynan = anynan || isnan(t[i]); pinf = pinf || (t[i] == INFINITY); ninf = ninf || (t[i] == -INFINITY); }
    if (anynan || (pinf && ninf)) { r[ch] = as_type<float>(0x7fffe000u); continue; }
    if (pinf) { r[ch] = INFINITY; continue; } if (ninf) { r[ch] = -INFINITY; continue; }
    float mx = 0.0f; bool allneg0 = true, any = false;
    for (int i = 0; i < 4; ++i) if (w[i]) { any = true; mx = max(mx, abs(t[i])); allneg0 = allneg0 && (t[i] == 0.0f && signbit(t[i])); }
    if (mx == 0.0f) { r[ch] = (any && allneg0) ? -0.0f : 0.0f; continue; }
    const int E = int(floor(log2(mx))); const float up = ldexp(1.0f, 14 - E);          // t * 2^(14-E) is exact (half value, |.| < 2^15)
    long acc = 0; for (int i = 0; i < 4; ++i) if (w[i]) acc += long(w[i]) * long(trunc(t[i] * up));   // |acc| < 2^26, exact
    const bool neg = acc < 0; long m = neg ? -acc : acc; int sh = 0; { long v = m; while (v >= (1L << 11)) { v >>= 1; ++sh; } }   // 11 significant bits
    int ge = (E - 22) + sh;                                                              // exponent of the half grid for m * 2^(E-22)
    if (ge < -24) { sh += (-24 - ge); ge = -24; }                                        // half subnormal grid
    long rr = sh > 0 ? ((m + (1L << (sh - 1))) >> sh) : m;                              // ties away from zero
    const float val = ldexp(float(rr), ge); r[ch] = neg ? -val : val;
  }
  return r;
}
