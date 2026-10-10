#include <metal_stdlib>
using namespace metal;
// Pure bijection from Block0's native C16 planar port to the encoder's existing
// canonical 8x8 windows. No arithmetic, padding, fixtures, or value-dependent rule.
kernel void nr_pre_entry_layout(device const uchar* input [[buffer(0)]],device uchar* output [[buffer(1)]],
                                 device atomic_uint* counts [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=17694720u)return;
  uint row=gid/32u,c=gid%32u,window=row/64u,token=row%64u,tile=token/16u,p=token%16u;
  uint x=(window%120u)*8u+(tile/2u)*4u+p%4u,y=(window/120u)*8u+(tile%2u)*4u+p/4u;
  uint source=((c/16u)*960u*576u+y*960u+x)*16u+c%16u;
  output[gid]=input[source];atomic_fetch_add_explicit(&counts[gid],1u,memory_order_relaxed);
}
