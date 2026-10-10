#include "swin32_mlp.metal"

// The source is the last contraction partial produced in this command buffer.
// Preserve the E4M3 storage boundary before the Q/K/V projections.
kernel void nr_swin32_qkv(device const half* partial [[buffer(0)]],
                         device const uchar* weights [[buffer(1)]],
                         device half* output [[buffer(2)]],
                         device atomic_uint* coverage [[buffer(3)]],
                         constant MLPShape& shape [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*96u)return;
  uint row=gid/96u,n=gid%96u;
  NrQmmaAccumulator sum=nr_qmma_begin(half(0));
  for(uint k=0u;k<32u;++k) {
    half source=partial[(3u*shape.rows+row)*32u+nr_dense_from_channel(k)];
    half quantized=nr_e4m3(nr_to_e4m3(source));
    nr_qmma_add(sum,k,quantized,nr_e4m3(weights[8288u+nr_weight_index(k,n)]));
  }
  output[gid]=nr_qmma_finish(sum);
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}
