// nr_tex_probe.metal — runs nr_tex2d_bilinear_rtx (and the q8 variants) on a list of normalised (u, v) queries over an RGBA16F texture; one
// thread per query; exact-write coverage counter. Diagnostic for scripts/test-nr-tex-rtx-probe-metal.py against Codex's RTX micro-probe.
#include <metal_stdlib>
using namespace metal;
#include "nr_tex_sample.metal"
kernel void nr_tex_probe_rtx(texture2d<half, access::read> tex [[texture(0)]], device const float2* q [[buffer(0)]], device float4* out [[buffer(1)]], device atomic_uint* cov [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return; out[i] = nr_tex2d_bilinear_rtx(tex, q[i].x, q[i].y); atomic_fetch_add_explicit(cov + i, 1u, memory_order_relaxed);
}
