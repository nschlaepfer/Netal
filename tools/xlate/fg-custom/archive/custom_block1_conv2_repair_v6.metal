#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_u32.metalh"
#include "hmma_exact_lanes_fast.metalh"
using namespace metal;
// Two-pass repair for the conv2 family. Pass 1 runs over the compacted list of flagged elements and
// computes only the first flagged stage exactly; every unflagged stage is certified (given the same prior,
// the fast pass's half equals NVIDIA's), so if that one stage agrees with what the fast pass produced and
// no later stage was flagged, the fast output already written is correct. The rest - about one in ten -
// advance their resume state in place and go to a second list, which the full replay (the original listed
// kernel) consumes as a uniform dispatch. Splitting the two keeps every SIMD group's lanes at the same
// trip count; with one dispatch the 10% long lanes made 97% of the groups run the full replay.
struct Params{uint residual;};
inline uint widx(uint tile,uint stage,uint ic,uint oc){uint g=oc/8u,n=oc%8u,k=ic%16u,lane=n*4u+(k%8u)/2u,frag=(k>=8u?2u:0u)+(k&1u);return g*2304u+tile*256u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u);}
inline half conv2_stage(device const half*w,device const half*input,uint x,uint y,uint oc,uint step,half a){
  const uint tile=step/2u,stage=step%2u;int ix=int(x)+int(tile%3u)-1,iy=int(y)+int(tile/3u)-1;bool ok=ix>=0&&ix<160&&iy>=0&&iy<92;uint base=ok?(uint(iy)*160u+uint(ix))*32u:0u;
  ushort av[16],bv[16];
  if(ok){ const ushort4 v0=*reinterpret_cast<device const ushort4*>(input+base+stage*16u), v1=*reinterpret_cast<device const ushort4*>(input+base+stage*16u+4u), v2=*reinterpret_cast<device const ushort4*>(input+base+stage*16u+8u), v3=*reinterpret_cast<device const ushort4*>(input+base+stage*16u+12u);
    av[0]=v0.x;av[1]=v0.y;av[2]=v0.z;av[3]=v0.w;av[4]=v1.x;av[5]=v1.y;av[6]=v1.z;av[7]=v1.w;av[8]=v2.x;av[9]=v2.y;av[10]=v2.z;av[11]=v2.w;av[12]=v3.x;av[13]=v3.y;av[14]=v3.z;av[15]=v3.w; }
  else for(uint k=0;k<16;++k)av[k]=ushort(0);
  for(uint k=0;k<16;++k)bv[k]=as_type<ushort>(w[widx(tile,stage,stage*16u+k,oc)]);
  return fgu_hmma_f16_bits_16(av,bv,a);}
kernel void fg_custom_block1_conv2_repair_pass1_v6(device const half*w[[buffer(0)]],device const half*b[[buffer(1)]],device const half*input[[buffer(2)]],device const half*residual[[buffer(3)]],device half*out[[buffer(4)]],constant Params&p[[buffer(5)]],
    device const uint*list[[buffer(6)]],device const uint*counter[[buffer(7)]],device uchar*resume_stage[[buffer(8)]],device half*resume_acc[[buffer(9)]],device const half*fast_after[[buffer(10)]],device const uchar*flag_count[[buffer(11)]],
    device uint*list2[[buffer(12)]],device atomic_uint*counter2[[buffer(13)]],device const uchar*stage2[[buffer(14)]],device const half*prior2[[buffer(15)]],device const half*after2[[buffer(16)]],uint tid[[thread_position_in_grid]]){
  if(tid>=counter[0])return;
  const uint linear=list[tid],oc=linear%32u,x=(linear/32u)%160u,y=(linear/32u)/160u;
  const uint start=uint(resume_stage[linear]);
  half a=conv2_stage(w,input,x,y,oc,start,start==0u?b[oc]:resume_acc[linear]);
  const uint n=flag_count[linear];
  if(as_type<ushort>(a)!=as_type<ushort>(fast_after[linear])){resume_stage[linear]=uchar(start+1u);resume_acc[linear]=a;list2[atomic_fetch_add_explicit(counter2,1u,memory_order_relaxed)]=linear;return;}
  if(n==1u)return;
  // slot 0 agreed, so every stage up to the second flagged one is certified and the fast pass's prior there is exact
  const uint s2=uint(stage2[linear]);
  half a2=conv2_stage(w,input,x,y,oc,s2,prior2[linear]);
  if(n==2u&&as_type<ushort>(a2)==as_type<ushort>(after2[linear]))return;
  resume_stage[linear]=uchar(s2+1u);resume_acc[linear]=a2;list2[atomic_fetch_add_explicit(counter2,1u,memory_order_relaxed)]=linear;}
inline half conv2_stage_lanes(device const half*w,device const half*input,uint x,uint y,uint oc,uint step,half a,uint k){
  const uint tile=step/2u,stage=step%2u;int ix=int(x)+int(tile%3u)-1,iy=int(y)+int(tile/3u)-1;bool ok=ix>=0&&ix<160&&iy>=0&&iy<92;uint base=ok?(uint(iy)*160u+uint(ix))*32u:0u;
  const uint ic=stage*16u+k;const ushort av=ok?as_type<ushort>(input[base+ic]):ushort(0);const ushort bv=as_type<ushort>(w[widx(tile,stage,ic,oc)]);
  return fgf_hmma_f16_lanes16(av,bv,a,k);}
kernel void fg_custom_block1_conv2_repair_pass2_v6(device const half*w[[buffer(0)]],device const half*b[[buffer(1)]],device const half*input[[buffer(2)]],device const half*residual[[buffer(3)]],device half*out[[buffer(4)]],constant Params&p[[buffer(5)]],
    device const uint*list[[buffer(6)]],device const uint*counter[[buffer(7)]],device const uchar*resume_stage[[buffer(8)]],device const half*resume_acc[[buffer(9)]],uint tid[[thread_position_in_grid]]){
  const uint elem=tid/16u,k=tid%16u; if(elem>=counter[0])return;
  const uint linear=list[elem],oc=linear%32u,x=(linear/32u)%160u,y=(linear/32u)/160u;
  const uint start=uint(resume_stage[linear]);
  half a=start==0u?b[oc]:resume_acc[linear];
  for(uint step=start;step<18u;++step)a=conv2_stage_lanes(w,input,x,y,oc,step,a,k);
  if(k!=0u)return;
  half slope=half(0.01f);a=min(max(max(a,half(a*slope)),half(-6.0f)),half(6.0f));uint index=(y*160u+x)*32u+oc;half rv=p.residual!=0u?residual[index]:half(0.0f);out[index]=half(a+rv);}
