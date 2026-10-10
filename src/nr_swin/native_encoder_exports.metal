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
