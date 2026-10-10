// Per-token elementwise passes of the PWin pipeline (one thread per token, 32 channels).
//   mode 0 (RMS-style norm of the transpiled layers, bit-exact op order):
//     x = relu(in) (fp16); q = x*x (fp16);
//     per channel-pair lane l (l = 0..3) and parity h: s[l][h] = (q[2l+h] + q[16+2l+h]) + (q[8+2l+h] + q[24+2l+h])
//     t[l] = s[l] + s[l^1]; u[l] = t[l] + t[l^2]; total = u[0][0] + u[0][1]   (all fp16 adds)
//     r = half( precise_rsqrt( float(total) ) );  out[c] = (gamma[c] * r) * x[c]   (fp16 multiplies)
//   mode 1 (per-channel scale): out[c] = fp16(in[c] * gamma[c])
#include <metal_stdlib>
using namespace metal;

struct NormParams {
  uint tokens;
  uint in_stride;    // halves per token row of the input
  uint out_stride;   // halves per token row of the output
  uint mode;
};

kernel void pwin_norm32(device const half* In [[buffer(0)]],
                        device half* Out [[buffer(1)]],
                        device const half* Gamma [[buffer(2)]],
                        constant NormParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.tokens) return;
  device const half* in = In + gid * p.in_stride;
  device half* out = Out + gid * p.out_stride;
  half x[32];
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 32u; c += 2u) {
    const half2 v = *reinterpret_cast<device const half2*>(in + c);
    x[c] = v.x; x[c + 1u] = v.y;
  }
  if (p.mode == 1u) {
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 32u; c += 2u) {
      const half2 g = *reinterpret_cast<device const half2*>(Gamma + c);
      *reinterpret_cast<device half2*>(out + c) = half2(x[c] * g.x, x[c + 1u] * g.y);
    }
    return;
  }
  half q[32];
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 32u; ++c) {
    if (p.mode == 0u) x[c] = max(x[c], half(0.0h));   // mode 2: same norm without the relu (decoder input)
    q[c] = x[c] * x[c];
  }
  half s[4][2], t[4][2], u[4][2];
  #pragma clang loop unroll(full)
  for (uint l = 0u; l < 4u; ++l)
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h)
      s[l][h] = (q[2u * l + h] + q[16u + 2u * l + h]) + (q[8u + 2u * l + h] + q[24u + 2u * l + h]);
  #pragma clang loop unroll(full)
  for (uint l = 0u; l < 4u; ++l)
  #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) t[l][h] = s[l][h] + s[l ^ 1u][h];
  #pragma clang loop unroll(full)
  for (uint l = 0u; l < 4u; ++l)
  #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) u[l][h] = t[l][h] + t[l ^ 2u][h];
  const half total = u[0][0] + u[0][1];
  const half r = half(precise::rsqrt(float(total)));
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 32u; c += 2u) {
    const half2 g = *reinterpret_cast<device const half2*>(Gamma + c);
    const half a = (g.x * r) * x[c];
    const half b = (g.y * r) * x[c + 1u];
    *reinterpret_cast<device half2*>(out + c) = half2(a, b);
  }
}

// 64-channel variant (enc1): mode 0 = RMS-style norm of the raw input, exact op order of the transpiled
// kernel (work/native/enc1/phase-report.md section 1; no relu, no eps, no 1/64):
//   sq[c] = x*x; c = 2q+p (q = 0..3, p = 0..1):
//   t3 = (sq[c] + sq[c+16]) + (sq[c+32] + sq[c+48]);  t6 = (sq[c+8] + sq[c+24]) + (sq[c+40] + sq[c+56]);  P[q][p] = t3 + t6
//   S[p] = (P[0][p] + P[1][p]) + (P[2][p] + P[3][p]);  tot = S[0] + S[1]   (all fp16 adds)
//   r = half(precise_rsqrt(float(tot)));  out[c] = x[c] * (r * gamma[c])
// mode 1 = per-channel scale out[c] = fp16(in[c] * gamma[c]).
kernel void pwin_norm64(device const half* In [[buffer(0)]],
                        device half* Out [[buffer(1)]],
                        device const half* Gamma [[buffer(2)]],
                        constant NormParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.tokens) return;
  device const half* in = In + gid * p.in_stride;
  device half* out = Out + gid * p.out_stride;
  half x[64];
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 64u; c += 4u) {
    const half4 v = *reinterpret_cast<device const half4*>(in + c);
    x[c] = v.x; x[c + 1u] = v.y; x[c + 2u] = v.z; x[c + 3u] = v.w;
  }
  if (p.mode == 1u) {
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 64u; c += 4u) {
      const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
      *reinterpret_cast<device half4*>(out + c) = half4(x[c] * g.x, x[c + 1u] * g.y, x[c + 2u] * g.z, x[c + 3u] * g.w);
    }
    return;
  }
  half sq[64];
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 64u; ++c) sq[c] = x[c] * x[c];
  half P[4][2];
  #pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q)
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint c = 2u * q + h;
      const half t3 = (sq[c] + sq[c + 16u]) + (sq[c + 32u] + sq[c + 48u]);
      const half t6 = (sq[c + 8u] + sq[c + 24u]) + (sq[c + 40u] + sq[c + 56u]);
      P[q][h] = t3 + t6;
    }
  half S[2];
  #pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) S[h] = (P[0][h] + P[1][h]) + (P[2][h] + P[3][h]);
  const half total = S[0] + S[1];
  const half r = half(precise::rsqrt(float(total)));
  #pragma clang loop unroll(full)
  for (uint c = 0u; c < 64u; c += 4u) {
    const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
    *reinterpret_cast<device half4*>(out + c) = half4(x[c] * (r * g.x), x[c + 1u] * (r * g.y), x[c + 2u] * (r * g.z), x[c + 3u] * (r * g.w));
  }
}

// 128-channel variant (enc4): RMS-style norm of the raw input, exact op order of the transpiled kernel
// (work/native/enc4/phase-report.md section 1): sq = x*x; c' = 2q+p (q = 0..3, p = 0..1);
//   e(g,g') = sq[8g + c'] + sq[8g' + c'] for (g,g') = (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15)
//   a = e02 + e46; b = e13 + e57; cc = e810 + e1214; d = e911 + e1315; even = a + cc; odd = b + d; Q[q][p] = even + odd
//   S[p] = (Q[0][p] + Q[1][p]) + (Q[2][p] + Q[3][p]); tot = S[0] + S[1]; r = half(precise_rsqrt(float(tot)))
//   out[c] = x[c] * (r * gamma[c])           (all fp16 ops, one rounding each)
kernel void pwin_norm128(device const half* In [[buffer(0)]],
                         device half* Out [[buffer(1)]],
                         device const half* Gamma [[buffer(2)]],
                         constant NormParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= p.tokens) return;
  device const half* in = In + gid * p.in_stride;
  device half* out = Out + gid * p.out_stride;
  half x[128];
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 128u; c += 4u) {
    const half4 v = *reinterpret_cast<device const half4*>(in + c);
    x[c] = v.x; x[c + 1u] = v.y; x[c + 2u] = v.z; x[c + 3u] = v.w;
  }
  if (p.mode == 1u) {                       // per-channel scale: out[c] = fp16(in[c] * gamma[c])
#pragma clang loop unroll(full)
    for (uint c = 0u; c < 128u; c += 4u) {
      const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
      *reinterpret_cast<device half4*>(out + c) = half4(x[c] * g.x, x[c + 1u] * g.y, x[c + 2u] * g.z, x[c + 3u] * g.w);
    }
    return;
  }
  half sq[128];
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 128u; ++c) sq[c] = x[c] * x[c];
  half Q[4][2];
#pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q)
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint c = 2u * q + h;
      const half e02 = sq[c] + sq[16u + c], e13 = sq[8u + c] + sq[24u + c];
      const half e46 = sq[32u + c] + sq[48u + c], e57 = sq[40u + c] + sq[56u + c];
      const half e810 = sq[64u + c] + sq[80u + c], e911 = sq[72u + c] + sq[88u + c];
      const half e1214 = sq[96u + c] + sq[112u + c], e1315 = sq[104u + c] + sq[120u + c];
      const half a = e02 + e46, b = e13 + e57, cc = e810 + e1214, d = e911 + e1315;
      const half even = a + cc, odd = b + d;
      Q[q][h] = even + odd;
    }
  half S[2];
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) S[h] = (Q[0][h] + Q[1][h]) + (Q[2][h] + Q[3][h]);
  const half total = S[0] + S[1];
  const half r = half(precise::rsqrt(float(total)));
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 128u; c += 4u) {
    const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
    *reinterpret_cast<device half4*>(out + c) = half4(x[c] * (r * g.x), x[c + 1u] * (r * g.y), x[c + 2u] * (r * g.z), x[c + 3u] * (r * g.w));
  }
}

// 160-channel variant (dec5): the 8-word tree of the 128/64-channel norms plus the last word pair
// (work/native/dec5/phase-report.md section 1): with w_g = sq[c' + 8g], c' = 2q + p:
//   E = ((w0 + w2) + (w4 + w6)) + ((w8 + w10) + (w12 + w14)) + (w16 + w18)   [outer: f16(f16(8-word tree) + f16(w16 + w18))]
//   O = the same over the odd words; L[q][p] = E + O; S[p] = (L[0][p] + L[1][p]) + (L[2][p] + L[3][p]); tot = S[0] + S[1]
kernel void pwin_norm160(device const half* In [[buffer(0)]],
                         device half* Out [[buffer(1)]],
                         device const half* Gamma [[buffer(2)]],
                         constant NormParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= p.tokens) return;
  device const half* in = In + gid * p.in_stride;
  device half* out = Out + gid * p.out_stride;
  half x[160];
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 160u; c += 4u) {
    const half4 v = *reinterpret_cast<device const half4*>(in + c);
    x[c] = v.x; x[c + 1u] = v.y; x[c + 2u] = v.z; x[c + 3u] = v.w;
  }
  if (p.mode == 1u) {                       // per-channel scale: out[c] = fp16(in[c] * gamma[c])
#pragma clang loop unroll(full)
    for (uint c = 0u; c < 160u; c += 4u) {
      const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
      *reinterpret_cast<device half4*>(out + c) = half4(x[c] * g.x, x[c + 1u] * g.y, x[c + 2u] * g.z, x[c + 3u] * g.w);
    }
    return;
  }
  half sq[160];
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 160u; ++c) sq[c] = x[c] * x[c];
  half L[4][2];
#pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q)
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      half t[2];
#pragma clang loop unroll(full)
      for (uint par = 0u; par < 2u; ++par) {      // even words (par 0) at c', odd words (par 1) at c' + 8
        const uint c = 2u * q + h + 8u * par;
        const half a = (sq[c] + sq[c + 16u]) + (sq[c + 32u] + sq[c + 48u]);
        const half b = (sq[c + 64u] + sq[c + 80u]) + (sq[c + 96u] + sq[c + 112u]);
        t[par] = (a + b) + (sq[c + 128u] + sq[c + 144u]);
      }
      L[q][h] = t[0] + t[1];
    }
  half S[2];
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) S[h] = (L[0][h] + L[1][h]) + (L[2][h] + L[3][h]);
  const half total = S[0] + S[1];
  const half r = half(precise::rsqrt(float(total)));
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 160u; c += 4u) {
    const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
    *reinterpret_cast<device half4*>(out + c) = half4(x[c] * (r * g.x), x[c + 1u] * (r * g.y), x[c + 2u] * (r * g.z), x[c + 3u] * (r * g.w));
  }
}

// 96-channel variant (enc3, dec3): the same fp16 tree over twelve 8-wide words, in triples rather than
// pairs (work/native/enc3/phase-report.md section 1; no relu, no eps, no 1/96):
//   e(g,g') = sq[8g + c'] + sq[8g' + c']  for (0,2),(1,3),(4,6),(5,7),(8,10),(9,11);  c' = 2q + p
//   even = (e02 + e46) + e810;  odd = (e13 + e57) + e911;  Q[q][p] = even + odd
//   S[p] = (Q[0][p] + Q[1][p]) + (Q[2][p] + Q[3][p]);  tot = S[0] + S[1]
kernel void pwin_norm96(device const half* In [[buffer(0)]],
                        device half* Out [[buffer(1)]],
                        device const half* Gamma [[buffer(2)]],
                        constant NormParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.tokens) return;
  device const half* in = In + gid * p.in_stride;
  device half* out = Out + gid * p.out_stride;
  half x[96];
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 96u; c += 4u) {
    const half4 v = *reinterpret_cast<device const half4*>(in + c);
    x[c] = v.x; x[c + 1u] = v.y; x[c + 2u] = v.z; x[c + 3u] = v.w;
  }
  if (p.mode == 1u) {                       // per-channel scale: out[c] = fp16(in[c] * gamma[c])
#pragma clang loop unroll(full)
    for (uint c = 0u; c < 96u; c += 4u) {
      const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
      *reinterpret_cast<device half4*>(out + c) = half4(x[c] * g.x, x[c + 1u] * g.y, x[c + 2u] * g.z, x[c + 3u] * g.w);
    }
    return;
  }
  half sq[96];
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 96u; ++c) sq[c] = x[c] * x[c];
  half Q[4][2];
#pragma clang loop unroll(full)
  for (uint q = 0u; q < 4u; ++q)
#pragma clang loop unroll(full)
    for (uint h = 0u; h < 2u; ++h) {
      const uint c = 2u * q + h;
      const half e02 = sq[c] + sq[c + 16u], e13 = sq[c + 8u] + sq[c + 24u];
      const half e46 = sq[c + 32u] + sq[c + 48u], e57 = sq[c + 40u] + sq[c + 56u];
      const half e810 = sq[c + 64u] + sq[c + 80u], e911 = sq[c + 72u] + sq[c + 88u];
      const half even = (e02 + e46) + e810;
      const half odd = (e13 + e57) + e911;
      Q[q][h] = even + odd;
    }
  half S[2];
#pragma clang loop unroll(full)
  for (uint h = 0u; h < 2u; ++h) S[h] = (Q[0][h] + Q[1][h]) + (Q[2][h] + Q[3][h]);
  const half total = S[0] + S[1];
  const half r = half(precise::rsqrt(float(total)));
#pragma clang loop unroll(full)
  for (uint c = 0u; c < 96u; c += 4u) {
    const half4 g = *reinterpret_cast<device const half4*>(Gamma + c);
    *reinterpret_cast<device half4*>(out + c) = half4(x[c] * (r * g.x), x[c + 1u] * (r * g.y), x[c + 2u] * (r * g.z), x[c + 3u] * (r * g.w));
  }
}
