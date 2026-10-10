#include "swin32_block.metal"

struct WindowMap {
  uint width,height,source_grid_width,source_grid_height,destination_grid_width,destination_grid_height;
  int source_x,source_y,destination_x,destination_y;
};

kernel void nr_swin32_rewindow(device const uchar* source [[buffer(0)]],
                              device uchar* destination [[buffer(1)]],
                              device atomic_uint* coverage [[buffer(2)]],
                              constant WindowMap& shape [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  uint rows=shape.destination_grid_width*shape.destination_grid_height*64u;
  if(gid>=rows*32u)return;
  uint row=gid/32u,channel=gid%32u,window=row/64u,token=row%64u,tile=token/16u,p=token%16u;
  int x=int(window%shape.destination_grid_width)*8+shape.destination_x+int(tile/2u)*4+int(p%4u);
  int y=int(window/shape.destination_grid_width)*8+shape.destination_y+int(tile%2u)*4+int(p/4u);
  uchar value=0u;
  if(x>=0&&y>=0&&x<int(shape.width)&&y<int(shape.height)) {
    uint xx=uint(x-shape.source_x),yy=uint(y-shape.source_y);
    uint source_window=(yy/8u)*shape.source_grid_width+xx/8u;
    uint source_tile=((xx%8u)/4u)*2u+(yy%8u)/4u;
    uint source_token=source_tile*16u+(yy%4u)*4u+xx%4u;
    value=source[(source_window*64u+source_token)*32u+channel];
  }
  destination[gid]=value;
  atomic_fetch_add_explicit(&coverage[gid],1u,memory_order_relaxed);
}
