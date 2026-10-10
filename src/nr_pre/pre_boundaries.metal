// Included after the versioned ordinary C32 core and FG HMMA numerical rule.
struct PreChunk {uint rows,base_window;};
inline uint nr_pre_row(uint x,uint y) {
  return ((y/8u)*240u+x/8u)*64u+((x%8u)/4u)*32u+((y%8u)/4u)*16u+(y%4u)*4u+x%4u;
}
kernel void nr_pre_adapter(device const half* features [[buffer(0)]],device const uchar* weights [[buffer(1)]],
                           device half* raw [[buffer(2)]],device uchar* packed [[buffer(3)]],
                           device atomic_uint* raw_counts [[buffer(4)]],device atomic_uint* packed_counts [[buffer(5)]],
                           constant PreChunk& shape [[buffer(6)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=shape.rows*32u)return;
  uint row=gid/32u,n=gid%32u,token=row%64u,tile=token/16u,p=token%16u;
  uint x=(tile/2u)*4u+p%4u,y=(tile%2u)*4u+p/4u;
  uint feature_base=(shape.base_window+row/64u)*1024u+(y*8u+x)*8u;
  ushort a[16],b[16];
  for(uint k=0;k<16u;++k){
    a[k]=as_type<ushort>(features[feature_base+(k/8u)*512u+k%8u]);
    uint wi=(n/16u)*512u+(n%8u)*64u+((k%8u)/2u)*16u+((n%16u)/8u)*8u+(k/8u)*4u+(k%2u)*2u;
    b[k]=*reinterpret_cast<device const ushort*>(weights+8208u+wi);
  }
  half value=fg_hmma_f16_bits_16(a,b,half(0));raw[gid]=value;
  uint index=row*32u+nr_channel_from_dense(n);packed[index]=nr_to_e4m3(value);
  atomic_fetch_add_explicit(&raw_counts[gid],1u,memory_order_relaxed);
  atomic_fetch_add_explicit(&packed_counts[index],1u,memory_order_relaxed);
}
kernel void nr_pre_export(device const half* projection [[buffer(0)]],device uchar* skip [[buffer(1)]],
                          device uchar* encoder [[buffer(2)]],device atomic_uint* skip_counts [[buffer(3)]],
                          device atomic_uint* encoder_counts [[buffer(4)]],constant PreChunk& s [[buffer(5)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*32u)return;
  uint row=gid/32u,k=gid%32u,window=s.base_window+row/64u,token=row%64u,tile=token/16u,p=token%16u;
  uint x=(window%240u)*8u+(tile/2u)*4u+p%4u,y=(window/240u)*8u+(tile%2u)*4u+p/4u;
  uint cell=(y/4u)*480u+x/4u,pixel=(y%4u)*4u+x%4u;
  uint index=cell*512u+(pixel%8u)*64u+((k%16u)/4u)*16u+(pixel/8u)*4u+(k/16u)*8u+k%4u;
  uint n=nr_dense_from_channel(k);skip[index]=nr_to_e4m3(projection[row*32u+n]);
  atomic_fetch_add_explicit(&skip_counts[index],1u,memory_order_relaxed);
  if((x%2u)==0u&&(y%2u)==0u){
    uint base=s.base_window*64u;
    half a=projection[(nr_pre_row(x,y)-base)*32u+n],b=projection[(nr_pre_row(x+1u,y)-base)*32u+n];
    half c=projection[(nr_pre_row(x,y+1u)-base)*32u+n],d=projection[(nr_pre_row(x+1u,y+1u)-base)*32u+n];
    half sum=half(half(a+b)+half(c+d));
    uint out=((k/16u)*960u*576u+(y/2u)*960u+x/2u)*16u+k%16u;
    encoder[out]=nr_to_e4m3(half(sum*half(0.25f)));
    atomic_fetch_add_explicit(&encoder_counts[out],1u,memory_order_relaxed);
  }
}
