#include "swin32_attention.metal"

kernel void nr_swin32_probabilities(device const half* scores [[buffer(0)]],
                                   device uchar* probabilities [[buffer(1)]],
                                   device atomic_uint* coverage [[buffer(2)]],
                                   constant MLPShape& shape [[buffer(3)]],
                                   uint row [[thread_position_in_grid]]) {
  if(row>=shape.rows)return;
  half e[64];
  for(uint k=0u;k<64u;++k) {
    half value=half(fma(float(scores[row*64u+k]),0.044921875f,1.30078125f));
    value=clamp(value,half(1.03125f),half(1.5693359375f));
    e[k]=as_type<half>(ushort((uint(as_type<ushort>(value))<<5u)+0x8000u));
  }
  half sums[8];
  for(uint p=0u;p<8u;++p) {
    half a=half(float(half(float(e[p])+float(e[p+8u])))+
                 float(half(float(e[p+32u])+float(e[p+40u]))));
    a=half(float(a)+float(half(float(e[p+16u])+float(e[p+24u]))));
    sums[p]=half(float(a)+float(half(float(e[p+48u])+float(e[p+56u]))));
  }
  half lo=half(float(half(float(half(float(sums[0])+float(sums[2])))+float(sums[4])))+float(sums[6]));
  half hi=half(float(half(float(half(float(sums[1])+float(sums[3])))+float(sums[5])))+float(sums[7]));
  half denominator=max(half(float(lo)+float(hi)),half(6.198883056640625e-05f));
  half inverse=half(1.0f/float(denominator));
  for(uint k=0u;k<64u;++k) {
    uint index=row*64u+k;
    probabilities[index]=nr_to_e4m3(half(float(e[k])*float(inverse)));
    atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);
  }
}

kernel void nr_swin32_values(device const uchar* probabilities [[buffer(0)]],
                            device const half* qkv [[buffer(1)]],
                            device half* partial [[buffer(2)]],
                            device atomic_uint* coverage [[buffer(3)]],
                            constant MLPShape& shape [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*32u)return;
  uint row=gid/32u,n=gid%32u,window=(row/64u)*64u;
  half sum=half(0.0f);
  for(uint chunk=0u;chunk<2u;++chunk) {
    NrQmmaAccumulator value=nr_qmma_begin(sum);
    for(uint k=0u;k<32u;++k) {
      uint key=chunk*16u+(k>=16u?32u:0u)+nr_dense_from_channel(k%16u);
      half v=nr_e4m3(nr_to_e4m3(qkv[(window+key)*96u+64u+n]));
      nr_qmma_add(value,k,nr_e4m3(probabilities[row*64u+key]),v);
    }
    sum=nr_qmma_finish(value);
    uint index=(chunk*shape.rows+row)*32u+n;
    partial[index]=sum;
    atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);
  }
}

kernel void nr_swin32_project(device const half* value_partial [[buffer(0)]],
                             device const half* mlp_partial [[buffer(1)]],
                             device const uchar* weights [[buffer(2)]],
                             device half* projected [[buffer(3)]],
                             device atomic_uint* coverage [[buffer(4)]],
                             constant MLPShape& shape [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*32u)return;
  uint row=gid/32u,n=gid%32u;
  device const half* scales=reinterpret_cast<device const half*>(weights+20592u);
  half prior=half(float(mlp_partial[(3u*shape.rows+row)*32u+n])*float(scales[n]));
  NrQmmaAccumulator value=nr_qmma_begin(prior);
  for(uint k=0u;k<32u;++k) {
    half input=value_partial[(shape.rows+row)*32u+nr_dense_from_channel(k)];
    nr_qmma_add(value,k,nr_e4m3(nr_to_e4m3(input)),nr_e4m3(weights[19568u+nr_weight_index(k,n)]));
  }
  projected[gid]=nr_qmma_finish(value);
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}

kernel void nr_swin32_pack(device const half* projected [[buffer(0)]],
                          device uchar* output [[buffer(1)]],
                          device atomic_uint* coverage [[buffer(2)]],
                          constant MLPShape& shape [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*32u)return;
  uint row=gid/32u,k=gid%32u;
  output[gid]=nr_to_e4m3(projected[row*32u+nr_dense_from_channel(k)]);
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}
