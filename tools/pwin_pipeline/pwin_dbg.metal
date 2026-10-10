// Diagnostics: map every fp16 bit pattern through the transcendentals used by the softmax so they can be
// compared against the interpreter's definitions (ex2.approx.f16x2 -> f16(exp2(f64 x)); rcp.approx.ftz.f32).
#include <metal_stdlib>
using namespace metal;

kernel void pwin_dbg_math(device half* Out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= 65536u) return;
  const half h = as_type<half>(ushort(gid));
  Out[gid] = exp2(h);
  Out[65536u + gid] = half(precise::exp2(float(h)));
  Out[131072u + gid] = half(1.0f / float(h));
  Out[196608u + gid] = half(precise::divide(1.0f, float(h)));
  Out[262144u + gid] = half(precise::rsqrt(float(h)));
  Out[327680u + gid] = half(precise::divide(1.0f, precise::sqrt(float(h))));
  // packed half2 exp2 (what the transpiled kernels call) versus the scalar form
  const half2 h2 = half2(h, h);
  Out[393216u + gid] = exp2(h2).x;
  // the full softmax element chain, packed vs scalar, with the score s = h
  const half CL = as_type<half>(ushort(0x4d69)), K1 = as_type<half>(ushort(0x91d5)), K3 = as_type<half>(ushort(0xc50d));
  const half2 t2 = max(min(half2(h, h), half2(CL)), half2(-CL));
  const half2 e2 = exp2(fma(t2, fma(t2 * t2, half2(K1), half2(1.0h)), half2(K3)));
  Out[458752u + gid] = e2.x;
  const half t = max(min(h, CL), -CL);
  Out[524288u + gid] = exp2(fma(t, fma(t * t, K1, half(1.0h)), K3));
}
