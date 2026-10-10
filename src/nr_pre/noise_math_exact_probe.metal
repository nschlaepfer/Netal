#include "noise_math_exact.metalh"
kernel void nr_noise_math_exact_probe(device uint* output [[buffer(0)]],device atomic_uint* coverage [[buffer(1)]],
                                       device const uchar* deltas [[buffer(2)]],device const uint2* escapes [[buffer(3)]],
                                       constant uint& escape_count [[buffer(4)]],uint i [[thread_position_in_grid]]) {
  if(i>=16777216u)return;
  float u=float(i+1u)*as_type<float>(0x33800000u);
  float lg=nr_noise_exact_lg2_unit24(u,deltas,escapes,escape_count);
  float natural=lg*as_type<float>(0x3f317218u),radius_arg=natural*-2.0f;
  float values[6]={lg,nr_noise_exact_trig_unit24(u,false,deltas,escapes,escape_count),
    nr_noise_exact_trig_unit24(u,true,deltas,escapes,escape_count),
    nr_noise_exact_sqrt_positive(as_type<float>(0x3f800000u+(i&0x7fffffu)),deltas,escapes,escape_count),
    nr_noise_exact_sqrt_positive(as_type<float>(0x40000000u+(i&0x7fffffu)),deltas,escapes,escape_count),
    nr_noise_exact_sqrt_positive(radius_arg,deltas,escapes,escape_count)};
  for(uint k=0;k<6;++k){uint index=i*6u+k;output[index]=as_type<uint>(values[k]);atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);}
}
