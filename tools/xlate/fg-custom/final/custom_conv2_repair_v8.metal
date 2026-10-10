#include <metal_stdlib>
#include "hmma_exact_fast.metalh"
#include "hmma_exact_u32.metalh"
using namespace metal;
// Pass 1 of the conv2 two-pass repair (identical arithmetic to fg_custom_block0_conv2_repair_pass1_v6) that
// also writes the indirect-dispatch arguments for pass 2, removing the fg_indirect_from_counter dispatch
// between them. Every 64-thread group, after its threads' appends to list2/counter2, bumps a "groups done"
// counter; the group that sees it reach indirect1[0]-1 (the group count this dispatch was launched with) is
// last and writes indirect2 = {ceil(counter2/per_group2), 1, 1}. Ordering as in fg_compact_suspects_ordered_ind:
// threadgroup_barrier(mem_device) + device-scope seq_cst fence on both sides of the done add.
// Early returns are replaced by predication so every thread reaches the barrier. Needs -std=metal3.2.
struct Params{uint residual;};
inline uint widx(uint tile,uint stage,uint ic,uint oc){uint g=oc/8u,n=oc%8u,k=ic%16u,lane=n*4u+(k%8u)/2u,frag=(k>=8u?2u:0u)+(k&1u);return g*4608u+tile*512u+stage*128u+(frag>=2u?64u:0u)+lane*2u+(frag&1u);}
inline half conv2_stage(device const half*w,device const half*input,uint x,uint y,uint oc,uint step,half a){
  const uint tile=step/4u,stage=step%4u;int ix=int(x)+int(tile%3u)-1,iy=int(y)+int(tile/3u)-1;bool ok=ix>=0&&ix<80&&iy>=0&&iy<46;uint base=ok?(uint(iy)*80u+uint(ix))*64u:0u;
  ushort av[16],bv[16];
  if(ok){ const ushort4 v0=*reinterpret_cast<device const ushort4*>(input+base+stage*16u), v1=*reinterpret_cast<device const ushort4*>(input+base+stage*16u+4u), v2=*reinterpret_cast<device const ushort4*>(input+base+stage*16u+8u), v3=*reinterpret_cast<device const ushort4*>(input+base+stage*16u+12u);
    av[0]=v0.x;av[1]=v0.y;av[2]=v0.z;av[3]=v0.w;av[4]=v1.x;av[5]=v1.y;av[6]=v1.z;av[7]=v1.w;av[8]=v2.x;av[9]=v2.y;av[10]=v2.z;av[11]=v2.w;av[12]=v3.x;av[13]=v3.y;av[14]=v3.z;av[15]=v3.w; }
  else for(uint k=0;k<16;++k)av[k]=ushort(0);
  for(uint k=0;k<16;++k)bv[k]=as_type<ushort>(w[widx(tile,stage,stage*16u+k,oc)]);
  return fgu_hmma_f16_bits_16(av,bv,a);}
kernel void fg_custom_block0_conv2_repair_pass1_v8(device const half*w[[buffer(0)]],device const half*b[[buffer(1)]],device const half*input[[buffer(2)]],device const half*residual[[buffer(3)]],device half*out[[buffer(4)]],constant Params&p[[buffer(5)]],
    device const uint*list[[buffer(6)]],device const uint*counter[[buffer(7)]],device uchar*resume_stage[[buffer(8)]],device half*resume_acc[[buffer(9)]],device const half*fast_after[[buffer(10)]],device const uchar*flag_count[[buffer(11)]],
    device uint*list2[[buffer(12)]],device atomic_uint*counter2[[buffer(13)]],device const uchar*stage2[[buffer(14)]],device const half*prior2[[buffer(15)]],device const half*after2[[buffer(16)]],
    device atomic_uint*done2[[buffer(17)]],device const uint*indirect1[[buffer(18)]],device uint*indirect2[[buffer(19)]],constant uint&per_group2[[buffer(20)]],
    uint tid[[thread_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){
  if(tid<counter[0]){
    const uint linear=list[tid],oc=linear%64u,x=(linear/64u)%80u,y=(linear/64u)/80u;
    const uint start=uint(resume_stage[linear]);
    half a=conv2_stage(w,input,x,y,oc,start,start==0u?b[oc]:resume_acc[linear]);
    const uint n=flag_count[linear];
    if(as_type<ushort>(a)!=as_type<ushort>(fast_after[linear])){resume_stage[linear]=uchar(start+1u);resume_acc[linear]=a;list2[atomic_fetch_add_explicit(counter2,1u,memory_order_relaxed)]=linear;}
    else if(n!=1u){
      // slot 0 agreed, so every stage up to the second flagged one is certified and the fast pass's prior there is exact
      const uint s2=uint(stage2[linear]);
      half a2=conv2_stage(w,input,x,y,oc,s2,prior2[linear]);
      if(!(n==2u&&as_type<ushort>(a2)==as_type<ushort>(after2[linear]))){resume_stage[linear]=uchar(s2+1u);resume_acc[linear]=a2;list2[atomic_fetch_add_explicit(counter2,1u,memory_order_relaxed)]=linear;}
    }
  }
  // release: this group's counter2 adds (and list2 writes) are ordered before its done add
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if(lid==0u){
    const uint prev=atomic_fetch_add_explicit(done2,1u,memory_order_relaxed);
    if(prev==indirect1[0]-1u){
      // acquire: every other group's done add, hence its counter2 adds, are ordered before this read
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      const uint c=atomic_load_explicit(counter2,memory_order_relaxed);
      indirect2[0]=(c+per_group2-1u)/per_group2; indirect2[1]=1u; indirect2[2]=1u;
    }
  }
}
