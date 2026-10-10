#include <metal_stdlib>
using namespace metal;

inline float selected_sm120_rcp(float value,
                                device const uchar* correction_rom) {
  const uint input_bits = as_type<uint>(value);
  uint base_bits = as_type<uint>(1.0f / value);
  const uint mantissa = input_bits & 0x7fffffu;
  const uchar packed = correction_rom[mantissa >> 2u];
  const uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u)
    base_bits -= 1u;
  else if (code == 2u)
    base_bits += 1u;
  return as_type<float>(base_bits);
}

struct P {
  uint width, height;
};
inline int2 pc(float2 u, uint2 e) {
  if (!(u.x >= 0 && u.x < 1 && u.y >= 0 && u.y < 1))
    return int2(-1);
  uint2 f = uint2(u * 4096.0f);
  return int2((f * e) >> 12u);
}
inline int2 pc21(float2 u, uint2 e) {
  if (!(u.x >= 0 && u.x < 1 && u.y >= 0 && u.y < 1))
    return int2(-1);
  uint2 f = uint2(u * 2097152.0f);
  return int2((f * e) >> 21u);
}
inline float4 rh4(texture2d<half, access::read> t, int2 p, uint2 e) {
  if (any(p < 0) || p.x >= int(e.x) || p.y >= int(e.y))
    return 0;
  return float4(t.read(uint2(p)));
}
inline float4 sh4(texture2d<half, access::read> t, float2 u, uint2 e) {
  return rh4(t, pc(u, e), e);
}
inline float2 sf2(texture2d<float, access::read> t, float2 u, uint2 e) {
  int2 p = pc(u, e);
  if (any(p < 0) || p.x >= int(e.x) || p.y >= int(e.y))
    return 0;
  return t.read(uint2(p)).xy;
}
inline uint su(texture2d<uint, access::read> t, float2 u, uint2 e) {
  int2 p = pc(u, e);
  if (any(p < 0) || p.x >= int(e.x) || p.y >= int(e.y))
    return 0;
  return t.read(uint2(p)).x;
}
inline ushort rz(float v) {
  uint b = as_type<uint>(v), s = (b >> 16) & 0x8000, e = (b >> 23) & 255,
       m = b & 0x7fffff;
  if (e == 255)
    return ushort(s | (m ? 0x7e00 : 0x7c00));
  int he = int(e) - 112;
  if (he >= 31)
    return ushort(s | 0x7c00);
  if (he <= 0) {
    if (he < -10)
      return ushort(s);
    return ushort(s | ((m | 0x800000) >> uint(14 - he)));
  }
  return ushort(s | (uint(he) << 10) | (m >> 13));
}
kernel void
fg_connected_distortion_module9(texture2d<half, access::read> a [[texture(0)]],
                      texture2d<half, access::read> b [[texture(1)]],
                      texture2d<float, access::read> flow32 [[texture(2)]],
                      texture2d<half, access::read> flow16 [[texture(3)]],
                      texture2d<uint, access::read> mask [[texture(4)]],
                      texture2d<half, access::write> oa [[texture(5)]],
                      texture2d<half, access::write> ob [[texture(6)]],
                      texture2d<uint, access::write> om [[texture(7)]],
                      constant P &p [[buffer(0)]],
                      device const uchar* correction_rom [[buffer(1)]],
                      uint2 q [[thread_position_in_grid]]) {
  if (q.x >= p.width || q.y >= p.height)
    return;
  uint2 e = uint2(p.width, p.height);
  float2 n = (float2(q) + .5f) *
             float2(as_type<float>(0x3b4ccccd), as_type<float>(0x3bb60b61));
  float4 av = sh4(a, n, e), bv = sh4(b, n, e);
  float2 c = n + av.xy;
  om.write(su(mask, c, e), q);
  float2 fv = sf2(flow32, c, e);
  float2 c2 = c + fv;
  float4 bs = sh4(b, c2, e);
  float2 va = (c2 + bs.zw) - n;
  float2 d;
  d.x = n.x + bv.x;
  d.y = n.y + bv.y;
  // The module-9 coordinate probe isolates this RG16F lookup: for its
  // independently generated nonzero fixtures, the selected SM120 lookup is
  // exactly a 21-bit normalized coordinate grid. The generic fixed12 rule is
  // one texel early in 29,400 pixels of the standalone fixture.
  const float2 sampled_flow = rh4(flow16, pc21(d, e), e).xy;
  float2 d2;
  // The selected module SASS contracts the PTX div.approx + add pair into
  // MUFU.RCP + FFMA. Preserve that architecture boundary explicitly.
  d2.x = fma(sampled_flow.x,
             selected_sm120_rcp(float(p.width), correction_rom), d.x);
  d2.y = fma(sampled_flow.y,
             selected_sm120_rcp(float(p.height), correction_rom), d.y);
  // The connected two-phase probe independently proves the same 21-bit grid
  // for this second path-B lookup. Its fixed12 mismatch sets exactly equal the
  // rejected output-B mismatch sets in both phases.
  float4 as = rh4(a, pc21(d2, e), e);
  float2 advanced;
  advanced.x = d2.x + as.z;
  advanced.y = d2.y + as.w;
  float2 vb;
  vb.x = advanced.x - n.x;
  vb.y = advanced.y - n.y;
  oa.write(half4(as_type<half>(rz(va.x)), as_type<half>(rz(va.y)), 0, 0), q);
  float2 px;
  px.x = vb.x * float(p.width);
  px.y = vb.y * float(p.height);
  ob.write(half4(as_type<half>(rz(px.x)), as_type<half>(rz(px.y)), 0, 0), q);
}
