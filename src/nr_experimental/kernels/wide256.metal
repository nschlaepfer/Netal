#include "swin32_block.metal"

#ifndef NR_WIDE_CHANNELS
#define NR_WIDE_CHANNELS 64
#endif
constant uint W=NR_WIDE_CHANNELS,H=4u*W,HEADS=W/32u;
constant uint MID=4u*W*W,MIX=MID+128u*W,RES=MIX+W*W+16u;
constant uint QKV=RES+2u*W+16u,BIAS=QKV+3u*W*W;
constant uint SCALE=BIAS+8192u*HEADS,PROJ=((SCALE+4u*HEADS+15u)/16u)*16u,ARES=PROJ+W*W;
inline uint wide_dense(uint k){return (k/32u)*32u+nr_dense_from_channel(k%32u);}
inline uint wide_channel(uint n){return (n/32u)*32u+nr_channel_from_dense(n%32u);}

inline half wide_norm32(device const half* value) {
#ifdef NR_WIDE_FLOAT32_NORM
  // Preserved counterfactual: float32-then-half can double-round a half FMA.
  return nr_norm32_float32_counterfactual(value);
#else
  half lanes[8];
  for(uint p=0;p<8;++p) {
#ifdef NR_WIDE_UNFUSED_NORM
    half a=half(float(half(float(value[8u+p])*float(value[8u+p])))+
                float(half(float(value[24u+p])*float(value[24u+p]))));
    half b=half(float(half(float(value[p])*float(value[p])))+
                float(half(float(value[16u+p])*float(value[16u+p]))));
#else
    // Block11 counterexample: fma(h(-6.375), h(-6.375), h(bits=0x0004))
    // must round to 0x5115, not float32-then-half's 0x5114.
    half a=fma(value[8u+p],value[8u+p],half(float(value[24u+p])*float(value[24u+p])));
    half b=fma(value[p],value[p],half(float(value[16u+p])*float(value[16u+p])));
#endif
    lanes[p]=half(float(a)+float(b));
  }
  half lo=half(float(half(float(lanes[0])+float(lanes[4])))+float(half(float(lanes[2])+float(lanes[6]))));
  half hi=half(float(half(float(lanes[1])+float(lanes[5])))+float(half(float(lanes[3])+float(lanes[7]))));
  return max(half(float(lo)+float(hi)),half(6.198883056640625e-05f));
#endif
}

// The selected C64 PTX has two 128-wide activation groups, each reduced
// independently to 32 channels, then a quantized C->C mixing projection.
kernel void nr_wide_expand(device const uchar* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
                           device half* y [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                           constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*H)return;uint row=gid/H,n=gid%H;half acc=half(0);
  for(uint chunk=0;chunk<HEADS;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k){uint offset=(n/128u)*128u*W+chunk*4096u+nr_weight_index(k,n%128u);
      nr_qmma_add(value,k,nr_e4m3(x[row*W+chunk*32u+k]),nr_e4m3(w[offset]));}
    acc=nr_qmma_finish(value);
  }y[gid]=acc;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_activate(device const half* x [[buffer(0)]],device uchar* y [[buffer(1)]],
                             device atomic_uint* count [[buffer(2)]],constant MLPShape& s [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*H)return;uint row=gid/H,n=gid%H;half value=x[row*H+wide_dense(n)];
  half clip=clamp(value,half(-4),half(4));
  half t=half(fma(abs(float(clip)),-float(as_type<half>(ushort(0x2b28))),0.447265625f));
  t=half(fma(float(clip),float(t),0.89453125f));y[gid]=nr_to_e4m3(half(float(value)*float(t)));
  atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_mid(device const uchar* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
                        device half* y [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                        constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W)return;uint row=gid/W,n=gid%W;half acc=half(0);
  for(uint chunk=0;chunk<4;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k)nr_qmma_add(value,k,nr_e4m3(x[row*H+(n/32u)*128u+chunk*32u+k]),
      nr_e4m3(w[MID+(n/32u)*4096u+chunk*1024u+nr_weight_index(k,n%32u)]));
    acc=nr_qmma_finish(value);
  }y[gid]=acc;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_mix(device const uchar* original [[buffer(0)]],device const half* x [[buffer(1)]],
                        device const uchar* w [[buffer(2)]],device half* y [[buffer(3)]],device atomic_uint* count [[buffer(4)]],
                        constant MLPShape& s [[buffer(5)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W)return;uint row=gid/W,n=gid%W;
  half acc=half(float(nr_e4m3(original[row*W+wide_channel(n)]))*float(reinterpret_cast<device const half*>(w+RES)[n]));
  for(uint chunk=0;chunk<HEADS;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k)nr_qmma_add(value,k,nr_e4m3(nr_to_e4m3(x[row*W+chunk*32u+nr_dense_from_channel(k)])),
      nr_e4m3(w[MIX+chunk*W*32u+nr_weight_index(k,n)]));
    acc=nr_qmma_finish(value);
  }y[gid]=acc;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_qkv(device const half* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
                        device half* y [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                        constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W*3u)return;uint row=gid/(3u*W),n=gid%(3u*W);half acc=half(0);
  for(uint chunk=0;chunk<HEADS;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k)nr_qmma_add(value,k,nr_e4m3(nr_to_e4m3(x[row*W+chunk*32u+nr_dense_from_channel(k)])),
      nr_e4m3(w[QKV+chunk*3u*W*32u+nr_weight_index(k,n)]));
    acc=nr_qmma_finish(value);
  }y[gid]=acc;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_normalize(device const half* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
                              device uchar* y [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                              constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*HEADS*2u)return;uint row=gid/(HEADS*2u),head=gid/2u%HEADS,component=gid%2u;
  device const half* values=x+row*W*3u+head*96u+component*32u;
  half inv=half(precise::rsqrt(float(wide_norm32(values))));half scale=half(reinterpret_cast<device const float*>(w+SCALE)[head]);
  for(uint k=0;k<32;++k){half value=half(float(values[nr_dense_from_channel(k)])*float(inv));if(component==0)value=half(float(value)*float(scale));
    uint index=(row*HEADS+head)*64u+component*32u+k;y[index]=nr_to_e4m3(value);atomic_fetch_add_explicit(&count[index],1u,memory_order_relaxed);}
}
kernel void nr_wide_scores(device const uchar* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
                           device half* y [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                           constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*HEADS*64u)return;uint row=gid/(HEADS*64u),head=gid/64u%HEADS,key=gid%64u;
  uint offset=BIAS+head*8192u+nr_bias_offset(row%64u,key)-11360u;
  NrQmmaAccumulator value=nr_qmma_begin(*reinterpret_cast<device const half*>(w+offset));
  for(uint k=0;k<32;++k)nr_qmma_add(value,k,nr_e4m3(x[(row*HEADS+head)*64u+k]),
    nr_e4m3(x[(((row/64u)*64u+key)*HEADS+head)*64u+32u+k]));
  y[gid]=nr_qmma_finish(value);atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_values(device const uchar* probs [[buffer(0)]],device const half* qkv [[buffer(1)]],
                           device half* y [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                           constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W)return;uint row=gid/W,n=gid%W,head=n/32u;half acc=half(0);
  for(uint chunk=0;chunk<2;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k){uint key=chunk*16u+(k>=16u?32u:0u)+nr_dense_from_channel(k%16u);
      half v=nr_e4m3(nr_to_e4m3(qkv[((row/64u)*64u+key)*W*3u+head*96u+64u+n%32u]));
      nr_qmma_add(value,k,nr_e4m3(probs[(row*HEADS+head)*64u+key]),v);}
    acc=nr_qmma_finish(value);
  }y[gid]=acc;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_project(device const half* x [[buffer(0)]],device const half* ffn [[buffer(1)]],
                            device const uchar* w [[buffer(2)]],device half* y [[buffer(3)]],device atomic_uint* count [[buffer(4)]],
                            constant MLPShape& s [[buffer(5)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W)return;uint row=gid/W,n=gid%W;
  // Unlike C32, this selected wider kernel reloads its FFN residual from
  // E4M3 shared storage before the coefficient multiply.
  half residual=nr_e4m3(nr_to_e4m3(ffn[gid]));
  half acc=half(float(residual)*float(reinterpret_cast<device const half*>(w+ARES)[n]));
  for(uint chunk=0;chunk<HEADS;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k)nr_qmma_add(value,k,nr_e4m3(nr_to_e4m3(x[row*W+chunk*32u+nr_dense_from_channel(k)])),
      nr_e4m3(w[PROJ+chunk*W*32u+nr_weight_index(k,n)]));
    acc=nr_qmma_finish(value);
  }y[gid]=acc;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_pack(device const half* x [[buffer(0)]],device uchar* y [[buffer(1)]],
                         device atomic_uint* count [[buffer(2)]],constant MLPShape& s [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W)return;uint row=gid/W,n=gid%W;y[gid]=nr_to_e4m3(x[row*W+wide_dense(n)]);
  atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}

// Appended after swin_wide.metal; W and the packed-weight helpers are shared.
struct WideWindowMap {
  uint width,height,source_grid_width,source_grid_height,destination_grid_width,destination_grid_height;
  int source_x,source_y,destination_x,destination_y;
};
inline uint wide_image_row(uint x,uint y,uint grid_width,int origin_x,int origin_y) {
  uint xx=uint(int(x)-origin_x),yy=uint(int(y)-origin_y);
  return ((yy/8u)*grid_width+xx/8u)*64u+((xx%8u)/4u)*32u+((yy%8u)/4u)*16u+(yy%4u)*4u+xx%4u;
}
kernel void nr_wide_rewindow(device const uchar* source [[buffer(0)]],device uchar* destination [[buffer(1)]],
                             device atomic_uint* count [[buffer(2)]],constant WideWindowMap& s [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.destination_grid_width*s.destination_grid_height*64u*W)return;
  uint row=gid/W,c=gid%W,window=row/64u,token=row%64u,tile=token/16u,p=token%16u;
  int x=int(window%s.destination_grid_width)*8+s.destination_x+int(tile/2u)*4+int(p%4u);
  int y=int(window/s.destination_grid_width)*8+s.destination_y+int(tile%2u)*4+int(p/4u);
  uchar value=0;
  if(x>=0&&y>=0&&x<int(s.width)&&y<int(s.height))value=source[wide_image_row(uint(x),uint(y),s.source_grid_width,s.source_x,s.source_y)*W+c];
  destination[gid]=value;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_pool(device const half* source [[buffer(0)]],device uchar* pooled [[buffer(1)]],
                         device atomic_uint* count [[buffer(2)]],constant WideWindowMap& s [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.destination_grid_width*s.destination_grid_height*64u*W)return;
  uint row=gid/W,k=gid%W,window=row/64u,token=row%64u,tile=token/16u,p=token%16u;
  uint x=(window%s.destination_grid_width)*8u+(tile/2u)*4u+p%4u,y=(window/s.destination_grid_width)*8u+(tile%2u)*4u+p/4u;
  if(x>=s.width/2u||y>=s.height/2u) {
    // Canonical comparison windows pad the non-8-aligned final encoder image.
    // The native consumer export below excludes these out-of-image pixels.
    pooled[gid]=0;atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);return;
  }
  uint n=wide_dense(k);half a=source[wide_image_row(x*2u,y*2u,s.source_grid_width,s.source_x,s.source_y)*W+n];
  half b=source[wide_image_row(x*2u+1u,y*2u,s.source_grid_width,s.source_x,s.source_y)*W+n];
  half c=source[wide_image_row(x*2u,y*2u+1u,s.source_grid_width,s.source_x,s.source_y)*W+n];
  half d=source[wide_image_row(x*2u+1u,y*2u+1u,s.source_grid_width,s.source_x,s.source_y)*W+n];
  half sum=half(float(half(float(a)+float(b)))+float(half(float(c)+float(d))));
  pooled[gid]=nr_to_e4m3(half(float(sum)*0.25f));atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}
kernel void nr_wide_down_project(device const uchar* pooled [[buffer(0)]],device const uchar* weight [[buffer(1)]],
                                 device uchar* result [[buffer(2)]],device atomic_uint* count [[buffer(3)]],
                                 constant MLPShape& s [[buffer(4)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.rows*W*2u)return;uint row=gid/(W*2u),n=wide_dense(gid%(W*2u));half acc=half(0);
  for(uint chunk=0;chunk<HEADS;++chunk){NrQmmaAccumulator value=nr_qmma_begin(acc);
    for(uint k=0;k<32;++k)nr_qmma_add(value,k,nr_e4m3(pooled[row*W+chunk*32u+k]),
      nr_e4m3(weight[ARES+W*2u+chunk*W*64u+nr_weight_index(k,n)]));
    acc=nr_qmma_finish(value);
  }result[gid]=nr_to_e4m3(acc);atomic_fetch_add_explicit(&count[gid],1u,memory_order_relaxed);
}

struct EncoderExportShape { uint width,height,channels,planar; };
kernel void nr_encoder_native_export(device const uchar* source [[buffer(0)]],device uchar* destination [[buffer(1)]],
                                      device atomic_uint* counts [[buffer(2)]],constant EncoderExportShape& s [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=s.width*s.height*s.channels)return;
  uint pixel=gid/s.channels,c=gid%s.channels,x=pixel%s.width,y=pixel/s.width;
  uint row=wide_image_row(x,y,(s.width+7u)/8u,0,0),index;
  if(s.planar)index=((c/16u)*s.width*s.height+pixel)*16u+c%16u;
  else {
    uint cell=(y/4u)*(s.width/4u)+x/4u,p=(y%4u)*4u+x%4u;
    index=cell*16u*s.channels+(c/32u)*512u+(p%8u)*64u+((c%16u)/4u)*16u+(p/8u)*4u+((c%32u)/16u)*8u+c%4u;
  }
  destination[index]=source[row*s.channels+c];
  atomic_fetch_add_explicit(&counts[index],1u,memory_order_relaxed);
}
