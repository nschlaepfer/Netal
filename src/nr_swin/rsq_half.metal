#include <metal_stdlib>
using namespace metal;
kernel void nr_rsq_half(device ushort* output [[buffer(0)]],
                        device atomic_uint* coverage [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if(gid>=0x7bffu)return;
  half value=as_type<half>(ushort(gid+1u));
  output[gid]=as_type<ushort>(half(precise::rsqrt(float(value))));
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}
