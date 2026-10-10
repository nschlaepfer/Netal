#include <metal_stdlib>
using namespace metal;
#include "qmma_fixed.metalh"

struct MLPShape { uint rows; };

inline half nr_e4m3(uchar value) {
  uint exponent=(uint(value)>>3u)&15u, fraction=uint(value)&7u;
  ushort sign=ushort(uint(value&128u)<<8u);
  if(exponent==0u) return as_type<half>(ushort(sign | as_type<ushort>(half(fraction)*half(0.001953125f))));
  if(exponent==15u && fraction==7u) return as_type<half>(ushort(sign|0x7e00u));
  return as_type<half>(ushort(sign|((exponent+8u)<<10u)|(fraction<<7u)));
}

inline uint nr_rne_shift(uint value,uint shift) {
  if(shift==0u)return value;
  uint q=value>>shift, r=value&((1u<<shift)-1u), tie=1u<<(shift-1u);
  return q+uint(r>tie||(r==tie&&(q&1u)));
}

inline uchar nr_to_e4m3(half value) {
  uint bits=as_type<ushort>(value), sign=(bits>>8u)&128u;
  uint exponent=(bits>>10u)&31u, fraction=bits&1023u;
  if(exponent==31u)return uchar(sign|(fraction?127u:126u));
  if((bits&0x7fffu)>=0x5f00u)return uchar(sign|126u);
  if(exponent<9u) {
    uint mantissa=exponent?fraction+1024u:fraction;
    return uchar(sign|nr_rne_shift(mantissa,exponent?16u-exponent:15u));
  }
  uint mantissa=nr_rne_shift(1024u+fraction,7u);
  uint e=exponent-8u;
  if(mantissa==16u){mantissa=8u;++e;}
  return uchar(sign|min(126u,(e<<3u)+mantissa-8u));
}

inline uint nr_weight_index(uint k,uint n) {
  return (n/16u)*512u+(n%8u)*64u+((k%16u)/4u)*16u+
         ((n%16u)/8u)*8u+k%4u+(k/16u)*4u;
}

inline uint nr_dense_from_channel(uint k) {
  return (k&1u)+((k&12u)>>1u)+((k&2u)<<2u)+(k&16u);
}

inline uint nr_channel_from_dense(uint n) {
  return (n&1u)+((n&6u)<<1u)+((n&8u)>>2u)+(n&16u);
}

kernel void nr_swin32_expand(device const uchar* input [[buffer(0)]],
                            device const uchar* weights [[buffer(1)]],
                            device half* expanded [[buffer(2)]],
                            device atomic_uint* coverage [[buffer(3)]],
                            constant MLPShape& shape [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*128u)return;
  uint row=gid/128u, column=gid%128u;
  uint base=(column/32u)*1024u, n=column%32u;
  NrQmmaAccumulator sum=nr_qmma_begin(half(0));
  for(uint k=0u;k<32u;++k)
    nr_qmma_add(sum,k,nr_e4m3(input[row*32u+k]),nr_e4m3(weights[base+nr_weight_index(k,n)]));
  expanded[gid]=nr_qmma_finish(sum);
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}

kernel void nr_swin32_activate(device const half* expanded [[buffer(0)]],
                              device uchar* activation [[buffer(1)]],
                              device atomic_uint* coverage [[buffer(2)]],
                              constant MLPShape& shape [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*128u)return;
  uint row=gid/128u, column=gid%128u;
  uint dense=(column/32u)*32u+nr_dense_from_channel(column%32u);
  half value=expanded[row*128u+dense];
  half clipped=clamp(value,half(-4.0f),half(4.0f));
  half t0=half(fma(abs(float(clipped)),-float(as_type<half>(ushort(0x2b28u))),0.447265625f));
  half t1=half(fma(float(clipped),float(t0),0.89453125f));
  activation[gid]=nr_to_e4m3(half(float(value)*float(t1)));
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}

kernel void nr_swin32_contract(device const uchar* input [[buffer(0)]],
                              device const uchar* activation [[buffer(1)]],
                              device const uchar* weights [[buffer(2)]],
                              device half* partial [[buffer(3)]],
                              device atomic_uint* coverage [[buffer(4)]],
                              constant MLPShape& shape [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*32u)return;
  uint row=gid/32u,n=gid%32u;
  device const half* scales=reinterpret_cast<device const half*>(weights+8208u);
  half sum=half(float(nr_e4m3(input[row*32u+nr_channel_from_dense(n)]))*float(scales[n]));
  for(uint chunk=0u;chunk<4u;++chunk) {
    NrQmmaAccumulator value=nr_qmma_begin(sum);
    for(uint k=0u;k<32u;++k)
      nr_qmma_add(value,k,nr_e4m3(activation[row*128u+chunk*32u+k]),nr_e4m3(weights[4096u+chunk*1024u+nr_weight_index(k,n)]));
    sum=nr_qmma_finish(value);
    uint index=(chunk*shape.rows+row)*32u+n;
    partial[index]=sum;
    atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);
  }
}
