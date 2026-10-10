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
