#include "swin32_chain.metal"

// Selected Block 4 has a 120x73 grid at origin (0,-4). Keep the
// attention projection in half precision until after the spatial reduction.
inline uint nr_block4_projection_row(uint x,uint y) {
  uint yy=y+4u,window=(yy/8u)*120u+x/8u;
  uint tile=((x%8u)/4u)*2u+(yy%8u)/4u;
  return window*64u+tile*16u+(yy%4u)*4u+x%4u;
}

kernel void nr_swin32_downsample_pool(device const half* projected [[buffer(0)]],
                                     device uchar* pooled [[buffer(1)]],
                                     device atomic_uint* coverage [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if(gid>=480u*288u*32u)return;
  uint row=gid/32u,k=gid%32u,window=row/64u,token=row%64u,tile=token/16u,p=token%16u;
  uint x=(window%60u)*8u+(tile/2u)*4u+p%4u;
  uint y=(window/60u)*8u+(tile%2u)*4u+p/4u;
  uint n=nr_dense_from_channel(k);
  half a=projected[nr_block4_projection_row(x*2u,y*2u)*32u+n];
  half b=projected[nr_block4_projection_row(x*2u+1u,y*2u)*32u+n];
  half c=projected[nr_block4_projection_row(x*2u,y*2u+1u)*32u+n];
  half d=projected[nr_block4_projection_row(x*2u+1u,y*2u+1u)*32u+n];
#ifdef NR_DS_QUANTIZED_SOURCE
  // Deliberately incorrect negative control; never enabled in acceptance builds.
  a=nr_e4m3(nr_to_e4m3(a));b=nr_e4m3(nr_to_e4m3(b));
  c=nr_e4m3(nr_to_e4m3(c));d=nr_e4m3(nr_to_e4m3(d));
#endif
  half top=half(float(a)+float(b)),bottom=half(float(c)+float(d));
  half sum=half(float(top)+float(bottom));
  pooled[gid]=nr_to_e4m3(half(float(sum)*0.25f));
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}

kernel void nr_swin32_downsample_project(device const uchar* pooled [[buffer(0)]],
                                        device const uchar* weights [[buffer(1)]],
                                        device uchar* output [[buffer(2)]],
                                        device atomic_uint* coverage [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
  if(gid>=480u*288u*64u)return;
  uint row=gid/64u,c=gid%64u,n=(c/32u)*32u+nr_dense_from_channel(c%32u);
  NrQmmaAccumulator sum=nr_qmma_begin(half(0));
  for(uint k=0u;k<32u;++k)
    nr_qmma_add(sum,k,nr_e4m3(pooled[row*32u+k]),nr_e4m3(weights[20656u+(n/32u)*1024u+nr_weight_index(k,n%32u)]));
  output[gid]=nr_to_e4m3(nr_qmma_finish(sum));
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}
