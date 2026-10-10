#include "swin32_qkv.metal"

// Retained only for named negative controls. FP32 FMA followed by half
// conversion is not equivalent to NVIDIA's single-rounded half FMA.
inline half nr_norm32_float32_counterfactual(device const half* value) {
  half lanes[8];
  for(uint p=0u;p<8u;++p) {
    half a=half(fma(float(value[8u+p]),float(value[8u+p]),
                    float(half(float(value[24u+p])*float(value[24u+p])))));
    half b=half(fma(float(value[p]),float(value[p]),
                    float(half(float(value[16u+p])*float(value[16u+p])))));
    lanes[p]=half(float(a)+float(b));
  }
  half low=half(float(half(float(lanes[0])+float(lanes[4])))+
                float(half(float(lanes[2])+float(lanes[6]))));
  half high=half(float(half(float(lanes[1])+float(lanes[5])))+
                 float(half(float(lanes[3])+float(lanes[7]))));
  return max(half(float(low)+float(high)),half(6.198883056640625e-05f));
}

inline half nr_norm32(device const half* value) {
  half lanes[8];
  for(uint p=0u;p<8u;++p) {
    // Frame12/Block4 witness: a=0xc4c1, c=0x17f1. Single half FMA gives
    // 0x4da7; FP32 FMA then half incorrectly double-rounds to 0x4da6.
    half a=fma(value[8u+p],value[8u+p],half(float(value[24u+p])*float(value[24u+p])));
    half b=fma(value[p],value[p],half(float(value[16u+p])*float(value[16u+p])));
    lanes[p]=half(float(a)+float(b));
  }
  half low=half(float(half(float(lanes[0])+float(lanes[4])))+
                float(half(float(lanes[2])+float(lanes[6]))));
  half high=half(float(half(float(lanes[1])+float(lanes[5])))+
                 float(half(float(lanes[3])+float(lanes[7]))));
  return max(half(float(low)+float(high)),half(6.198883056640625e-05f));
}

kernel void nr_swin32_normalize(device const half* qkv [[buffer(0)]],
                               device const uchar* weights [[buffer(1)]],
                               device uchar* normalized [[buffer(2)]],
                               device atomic_uint* coverage [[buffer(3)]],
                               constant MLPShape& shape [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*2u)return;
  uint row=gid/2u,component=gid%2u;
  device const half* values=qkv+row*96u+component*32u;
  half inverse=half(precise::rsqrt(float(nr_norm32(values))));
  half scale=half(*reinterpret_cast<device const float*>(weights+19552u));
  for(uint k=0u;k<32u;++k) {
    half value=half(float(values[nr_dense_from_channel(k)])*float(inverse));
    if(component==0u)value=half(float(value)*float(scale));
    uint index=row*64u+component*32u+k;
    normalized[index]=nr_to_e4m3(value);
    atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);
  }
}

inline uint nr_bias_offset(uint query,uint key) {
  uint qt=query/16u,kt=key/16u;
  uint query_tile=((qt&1u)<<1u)+(qt>>1u);
  uint key_tile=((kt&1u)<<1u)+(kt>>1u);
  uint lane=(query%8u)*4u+(key%8u)/2u;
  uint fragment_index=(key%16u)/8u;
  uint component=(query%16u)/8u*2u+key%2u;
  return 11360u+query_tile*2048u+key_tile*512u+lane*16u+fragment_index*8u+component*2u;
}

kernel void nr_swin32_scores(device const uchar* normalized [[buffer(0)]],
                            device const uchar* weights [[buffer(1)]],
                            device half* scores [[buffer(2)]],
                            device atomic_uint* coverage [[buffer(3)]],
                            constant MLPShape& shape [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*64u)return;
  uint row=gid/64u,key=gid%64u,query=row%64u,window=row/64u;
  NrQmmaAccumulator sum=nr_qmma_begin(*reinterpret_cast<device const half*>(weights+nr_bias_offset(query,key)));
  for(uint k=0u;k<32u;++k)
    nr_qmma_add(sum,k,nr_e4m3(normalized[row*64u+k]),nr_e4m3(normalized[(window*64u+key)*64u+32u+k]));
  scores[gid]=nr_qmma_finish(sum);
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}
