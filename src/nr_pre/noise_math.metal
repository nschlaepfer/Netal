#include <metal_stdlib>
using namespace metal;
// Model-independent baselines for lossless instruction-delta construction.
inline float nr_noise_phase_rz(float u){
  float a=u*as_type<float>(0x40c90fdbu),b=as_type<float>(0x3e22f983u),v=a*b;
  return fma(a,b,-v)<0.0f?as_type<float>(as_type<uint>(v)-1u):v;
}
kernel void nr_noise_math_baseline(device uint* output [[buffer(0)]],device atomic_uint* coverage [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
  if(i>=16777216u)return;
  float u=float(i+1u)*as_type<float>(0x33800000u),phase=nr_noise_phase_rz(u);
  float values[5]={precise::log2(u),precise::sinpi(2.0f*phase),precise::cospi(2.0f*phase),
    precise::sqrt(as_type<float>(0x3f800000u+(i&0x7fffffu))),precise::sqrt(as_type<float>(0x40000000u+(i&0x7fffffu)))};
  for(uint k=0;k<5;++k){uint index=i*5u+k;output[index]=as_type<uint>(values[k]);atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);}
}
