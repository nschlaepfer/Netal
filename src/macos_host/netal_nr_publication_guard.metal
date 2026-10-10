// Experimental NR adapter safety, not model arithmetic or an exactness check.
// Separate encoders with tracked buffers establish scan -> publication ordering.
#include <metal_stdlib>
using namespace metal;

kernel void netal_nr_scan_finite(device const half4* candidate [[buffer(0)]],
                                device atomic_uint* rejected [[buffer(1)]],
                                constant uint4& size [[buffer(2)]],
                                uint2 p [[thread_position_in_grid]]) {
  if (p.x >= size.x || p.y >= size.y) return;
  // Inspect IEEE half bits rather than fast-math isfinite assumptions.
  ushort4 bits = as_type<ushort4>(candidate[p.y * size.x + p.x]);
  if (any((bits & ushort4(0x7c00)) == ushort4(0x7c00)))
    atomic_fetch_or_explicit(rejected, 1u, memory_order_relaxed);
  // size.z selects RG11B10F publication. A finite half can still overflow the
  // narrower packed mantissa on conversion. Conservatively retain the original
  // frame above each channel's largest finite packed value (R/G65024,B64512).
  if (size.z && any((bits.xyz & ushort3(0x8000)) == ushort3(0) &&
                   bits.xyz > ushort3(0x7bf0,0x7bf0,0x7be0)))
    atomic_fetch_or_explicit(rejected, 2u, memory_order_relaxed);
}

kernel void netal_nr_publish_finite(device const half4* candidate [[buffer(0)]],
                                   device atomic_uint* rejected [[buffer(1)]],
                                   constant uint4& size [[buffer(2)]],
                                   texture2d<half, access::write> destination [[texture(0)]],
                                   uint2 p [[thread_position_in_grid]]) {
  if (p.x >= size.x || p.y >= size.y) return;
  // Destination was blitted from the original game texture, preserving packed
  // bits exactly. An invalid candidate leaves the WHOLE destination unchanged.
  if (atomic_load_explicit(rejected, memory_order_relaxed) != 0) return;
  destination.write(candidate[p.y * size.x + p.x], p);
}
