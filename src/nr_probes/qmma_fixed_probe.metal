#include "../nr_swin/swin32_mlp.metal"
kernel void nr_qmma_fixed_probe(device const uint* fragments [[buffer(0)]],device uint* output [[buffer(1)]],
                                 device atomic_uint* counts [[buffer(2)]],constant uint& cases [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
  if(gid>=cases*64u)return;
  uint testcase=gid/64u,lane=(gid%64u)/2u,reg=gid%2u,row=lane/4u+reg*8u,first_col=(lane%4u)*2u;
  device const uint* input=fragments+testcase*256u;uint result=0;
  for(uint component=0;component<2u;++component){uint col=first_col+component;
    uint c=input[((row%8u)*4u+col/2u)*8u+6u+row/8u];half prior=as_type<half>(ushort(c>>((col%2u)*16u)));
    NrQmmaAccumulator q=nr_qmma_begin(prior);
    for(uint k=0;k<32u;++k){uint alane=(row%8u)*4u+(k%16u)/4u,ar=row/8u+(k/16u)*2u;
      uint blane=col*4u+(k%16u)/4u,br=4u+k/16u;
      uchar a=uchar(input[alane*8u+ar]>>((k%4u)*8u)),b=uchar(input[blane*8u+br]>>((k%4u)*8u));
      nr_qmma_add(q,k,nr_e4m3(a),nr_e4m3(b));
    }
    result|=uint(as_type<ushort>(nr_qmma_finish(q)))<<(component*16u);
  }
  output[gid]=result;atomic_fetch_add_explicit(&counts[gid],1u,memory_order_relaxed);
}
