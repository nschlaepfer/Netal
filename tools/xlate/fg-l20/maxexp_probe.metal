#include <metal_stdlib>
using namespace metal;

// Can a simdgroup recover NVIDIA's max_exp - the maximum over 17 term exponents - using only MMAs?
//
// max over k of (e_a[r,k] + e_b[k,c]) is a max-plus product, which the MMA does not compute. But
// sum over k of 2^(e_a + e_b) is an ordinary product of 2^e_a and 2^e_b, and its leading bit lies
// in [max_exp, max_exp + 4] because at most 17 terms share the top. In base 32 the runner-ups cannot
// carry - 17 * 32^(m-1) < 32^m - so floor(log32) is exact, provided the exponent range fits fp32.
// Shifting every exponent against a stage-wide maximum (one simd_max) keeps it in range.
//
// Probe for stage 0 of L20. Writes the MMA-derived max_exp per element; a scalar kernel writes the
// truth; the host counts disagreements.

struct Conv3x3Parameters { uint width, height, input_channels, output_channels,
                           active_weight_half_count, column_stage_schedule; };

inline uint packed_weight_index(uint tile, uint ic, uint oc, uint input_channels) {
  const uint g = oc / 8u, n = oc % 8u, cs = ic / 16u, k = ic % 16u;
  const uint tig = (k % 8u) / 2u, frag = (k >= 8u ? 2u : 0u) + (k & 1u);
  return g * (9u * input_channels * 8u) + tile * (input_channels * 8u) + cs * 128u + (n * 4u + tig) * 4u + frag;
}
inline uint2 matrix_element(uint lane) {
  return uint2(((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u), 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u));
}
// fp16 exponent as NVIDIA's rule counts it: biased field minus 15, subnormals as -14, zero excluded.
inline int e16(half v) {
  const ushort b = as_type<ushort>(v); const uint f = (b >> 10u) & 31u;
  if (f == 0u) return (b & 1023u) == 0u ? -1000 : -14;
  return int(f) - 15;
}

kernel void maxexp_mma(device const half* w [[buffer(0)]], device const half* input [[buffer(1)]],
                       device char* out_e1 [[buffer(2)]], device char* out_e2 [[buffer(3)]],
                       constant Conv3x3Parameters& P [[buffer(4)]],
                       uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  const uint og = group.z / 8u, slice = group.z & 7u;
  if (slice >= 2u || og * 8u >= P.output_channels) return;
  const uint2 el = matrix_element(lane); const uint mr = el.x, mc = el.y;
  const uint x0 = group.x * 8u, y0 = group.y * 4u + slice * 2u;
  const uint tile = 0u, cs = 0u;

  // exponents of this lane's operands, and the stage-wide maxima
  int ea[2][2][2]; int eb[2][2]; int max_ea = -1000, max_eb = -1000;
  for (uint rb = 0u; rb < 2u; ++rb) {
    const uint lp = rb * 8u + mr, ox = x0 + (lp & 7u), oy = y0 + (lp >> 3u);
    const int ix = int(ox) + int(tile % 3u) - 1, iy = int(oy) + int(tile / 3u) - 1;
    const bool ok = ox < P.width && oy < P.height && ix >= 0 && ix < int(P.width) && iy >= 0 && iy < int(P.height);
    const uint base = ok ? (uint(iy) * P.width + uint(ix)) * P.input_channels : 0u;
    for (uint kb = 0u; kb < 2u; ++kb) for (uint h = 0u; h < 2u; ++h) {
      const half v = ok ? input[base + cs * 16u + kb * 8u + mc + h] : half(0.0h);
      ea[rb][kb][h] = e16(v); max_ea = max(max_ea, ea[rb][kb][h]);
    }
  }
  for (uint kb = 0u; kb < 2u; ++kb) for (uint h = 0u; h < 2u; ++h) {
    const half v = w[packed_weight_index(tile, cs * 16u + kb * 8u + mr, og * 8u + mc + h, P.input_channels)];
    eb[kb][h] = e16(v); max_eb = max(max_eb, eb[kb][h]);
  }
  max_ea = simd_max(max_ea); max_eb = simd_max(max_eb);

  for (uint rb = 0u; rb < 2u; ++rb) {
    simdgroup_float8x8 a1[2], b1[2], a2[2], b2[2];
    for (uint kb = 0u; kb < 2u; ++kb) for (uint h = 0u; h < 2u; ++h) {
      // base 2, unshifted: leading bit within [max_exp, max_exp+4]
      a1[kb].thread_elements()[h] = ea[rb][kb][h] < -900 ? 0.0f : exp2(float(ea[rb][kb][h]));
      b1[kb].thread_elements()[h] = eb[kb][h] < -900 ? 0.0f : exp2(float(eb[kb][h]));
      // base 32, shifted against the stage maxima and clamped so nothing underflows: exact leading digit
      const int sa = ea[rb][kb][h] < -900 ? -1000 : max(ea[rb][kb][h] - max_ea, -12);
      const int sb = eb[kb][h] < -900 ? -1000 : max(eb[kb][h] - max_eb, -12);
      a2[kb].thread_elements()[h] = sa < -900 ? 0.0f : exp2(5.0f * float(sa));
      b2[kb].thread_elements()[h] = sb < -900 ? 0.0f : exp2(5.0f * float(sb));
    }
    simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 s2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_multiply_accumulate(s1, a1[0], b1[0], s1); simdgroup_multiply_accumulate(s1, a1[1], b1[1], s1);
    simdgroup_multiply_accumulate(s2, a2[0], b2[0], s2); simdgroup_multiply_accumulate(s2, a2[1], b2[1], s2);
    for (uint h = 0u; h < 2u; ++h) {
      const uint lp = rb * 8u + mr, ox = x0 + (lp & 7u), oy = y0 + (lp >> 3u);
      if (ox >= P.width || oy >= P.height) continue;
      const uint idx = (oy * P.width + ox) * P.output_channels + og * 8u + mc + h;
      int e = 0;
      const float v1 = s1.thread_elements()[h]; (void)frexp(v1, e);
      out_e1[idx] = char(v1 == 0.0f ? -100 : e - 1);
      const float v2 = s2.thread_elements()[h]; (void)frexp(v2, e);
      // leading digit of s2 in base 32: floor(log2(v2)/5), then undo the shift
      out_e2[idx] = char(v2 == 0.0f ? -100 : (int(floor(float(e - 1) / 5.0f)) + max_ea + max_eb));
    }
  }
}

kernel void maxexp_truth(device const half* w [[buffer(0)]], device const half* input [[buffer(1)]],
                         device char* out [[buffer(2)]], constant Conv3x3Parameters& P [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= P.width || gid.y >= P.height || gid.z >= P.output_channels) return;
  const uint tile = 0u, cs = 0u;
  const int ix = int(gid.x) - 1, iy = int(gid.y) - 1;
  const bool ok = ix >= 0 && ix < int(P.width) && iy >= 0 && iy < int(P.height);
  const uint base = ok ? (uint(iy) * P.width + uint(ix)) * P.input_channels : 0u;
  int m = -1000;
  for (uint k = 0u; k < 16u; ++k) {
    const half a = ok ? input[base + cs * 16u + k] : half(0.0h);
    const half b = w[packed_weight_index(tile, cs * 16u + k, gid.z, P.input_channels)];
    const int ea = e16(a), eb = e16(b);
    if (ea > -900 && eb > -900) m = max(m, ea + eb);   // products only; the prior is tested separately
  }
  out[(gid.y * P.width + gid.x) * P.output_channels + gid.z] = char(m < -900 ? -100 : m);
}
