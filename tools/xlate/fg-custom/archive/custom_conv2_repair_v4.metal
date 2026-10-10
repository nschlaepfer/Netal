#include <metal_stdlib>
#include "hmma_exact_lanes.metalh"
using namespace metal;
// conv2 repair, lane-parallel: 16 lanes per flagged element (4 elements per 64-thread group).
// Pass 1: first flagged stage exactly; if it agrees with the fast pass and it was the only flag, done; if a
// second stage was flagged, its fast-pass prior is exact (the stages between are certified), so check it
// too. Anything else advances its resume state and goes to the uniform pass 2.
struct Params{uint residual;};
inline uint widx(uint tile,uint stage,uint ic,uint oc){uint g=oc/8u,n=oc%8u,k=ic%16u,lane=n*4u+(k%8u)/2u,frag=(k>=8u?2u:0u)+(k&1u);return g*4608u+tile*512u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u);}
inline half conv2_stage_lanes(device const half*w,device const half*input,uint x,uint y,uint oc,uint step,half a,uint k){
  const uint tile=step/4u,stage=step%4u;int ix=int(x)+int(tile%3u)-1,iy=int(y)+int(tile/3u)-1;bool ok=ix>=0&&ix<80&&iy>=0&&iy<46;uint base=ok?(uint(iy)*80u+uint(ix))*64u:0u;
  const uint ic=stage*16u+k;const ushort av=ok?as_type<ushort>(input[base+ic]):ushort(0);const ushort bv=as_type<ushort>(w[widx(tile,stage,ic,oc)]);
  return fg_hmma_f16_lanes16(av,bv,a,k);}
kernel void fg_custom_block0_conv2_repair_pass1_v4(device const half*w[[buffer(0)]],device const half*b[[buffer(1)]],device const half*input[[buffer(2)]],device const half*residual[[buffer(3)]],device half*out[[buffer(4)]],constant Params&p[[buffer(5)]],
    device const uint*list[[buffer(6)]],device const uint*counter[[buffer(7)]],device uchar*resume_stage[[buffer(8)]],device half*resume_acc[[buffer(9)]],device const half*fast_after[[buffer(10)]],device const uchar*flag_count[[buffer(11)]],
    device uint*list2[[buffer(12)]],device atomic_uint*counter2[[buffer(13)]],device const uchar*stage2[[buffer(14)]],device const half*prior2[[buffer(15)]],device const half*after2[[buffer(16)]],uint tid[[thread_position_in_grid]]){
  const uint elem=tid/16u,k=tid%16u; if(elem>=counter[0])return;
  const uint linear=list[elem],oc=linear%64u,x=(linear/64u)%80u,y=(linear/64u)/80u;
  const uint start=uint(resume_stage[linear]);
  half a=conv2_stage_lanes(w,input,x,y,oc,start,start==0u?b[oc]:resume_acc[linear],k);
  const uint n=flag_count[linear];
  if(as_type<ushort>(a)!=as_type<ushort>(fast_after[linear])){ if(k==0u){resume_stage[linear]=uchar(start+1u);resume_acc[linear]=a;list2[atomic_fetch_add_explicit(counter2,1u,memory_order_relaxed)]=linear;} return;}
  if(n==1u)return;
  const uint s2=uint(stage2[linear]);
  half a2=conv2_stage_lanes(w,input,x,y,oc,s2,prior2[linear],k);
  if(n==2u&&as_type<ushort>(a2)==as_type<ushort>(after2[linear]))return;
  if(k==0u){resume_stage[linear]=uchar(s2+1u);resume_acc[linear]=a2;list2[atomic_fetch_add_explicit(counter2,1u,memory_order_relaxed)]=linear;}}
kernel void fg_custom_block0_conv2_repair_pass2_v4(device const half*w[[buffer(0)]],device const half*b[[buffer(1)]],device const half*input[[buffer(2)]],device const half*residual[[buffer(3)]],device half*out[[buffer(4)]],constant Params&p[[buffer(5)]],
    device const uint*list[[buffer(6)]],device const uint*counter[[buffer(7)]],device const uchar*resume_stage[[buffer(8)]],device const half*resume_acc[[buffer(9)]],uint tid[[thread_position_in_grid]]){
  const uint elem=tid/16u,k=tid%16u; if(elem>=counter[0])return;
  const uint linear=list[elem],oc=linear%64u,x=(linear/64u)%80u,y=(linear/64u)/80u;
  const uint start=uint(resume_stage[linear]);
  half a=start==0u?b[oc]:resume_acc[linear];
  for(uint step=start;step<36u;++step)a=conv2_stage_lanes(w,input,x,y,oc,step,a,k);
  if(k!=0u)return;
  half slope=half(0.01f);a=min(max(max(a,half(a*slope)),half(-6.0f)),half(6.0f));uint index=(y*80u+x)*64u+oc;half rv=p.residual!=0u?residual[index]:half(0.0f);out[index]=half(a+rv);}
