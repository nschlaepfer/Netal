#include <metal_stdlib>
#include "hmma_exact.metalh"
using namespace metal;
struct Params{uint residual;};
inline uint widx(uint tile,uint stage,uint ic,uint oc){uint g=oc/8u,n=oc%8u,k=ic%16u,lane=n*4u+(k%8u)/2u,frag=(k>=8u?2u:0u)+(k&1u);return g*4608u+tile*512u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u);}
kernel void fg_custom_block0_conv2_listed(device const half*w[[buffer(0)]],device const half*b[[buffer(1)]],device const half*input[[buffer(2)]],device const half*residual[[buffer(3)]],device half*out[[buffer(4)]],constant Params&p[[buffer(5)]],device const uint*list[[buffer(6)]],device const uint*counter[[buffer(7)]],device const uchar*resume_stage[[buffer(8)]],device const half*resume_acc[[buffer(9)]],uint tid[[thread_position_in_grid]]){
  if(tid>=counter[0])return;
  const uint linear=list[tid];
  const uint3 gid=uint3((linear/64u)%80u,(linear/64u)/80u,linear%64u);
  uint oc=gid.z;
  // resume from the first uncertain stage with the accumulator as it stood going in
  const uint start=uint(resume_stage[linear]);
  half a=start==0u?b[oc]:resume_acc[linear];
  const uint start_tile=start/4u, start_stage=start%4u;
  for(uint tile=start_tile;tile<9;++tile){int ix=int(gid.x)+int(tile%3u)-1,iy=int(gid.y)+int(tile/3u)-1;bool ok=ix>=0&&ix<80&&iy>=0&&iy<46;uint base=ok?(uint(iy)*80u+uint(ix))*64u:0u;for(uint stage=(tile==start_tile?start_stage:0u);stage<4;++stage){ushort av[16],bv[16];for(uint k=0;k<16;++k){uint ic=stage*16u+k;av[k]=ok?as_type<ushort>(input[base+ic]):ushort(0);bv[k]=as_type<ushort>(w[widx(tile,stage,ic,oc)]);}a=fg_hmma_f16_bits_16(av,bv,a);}}half slope=half(0.01f);a=min(max(max(a,half(a*slope)),half(-6.0f)),half(6.0f));uint index=(gid.y*80u+gid.x)*64u+oc;half residual_value=p.residual!=0u?residual[index]:half(0.0f);a=half(a+residual_value);out[index]=a;}
