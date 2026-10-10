#include <metal_stdlib>
using namespace metal;
#if NR_PRE_EXACT_MATH
#include "noise_math_exact.metalh"
#endif

// Diagnostic candidate for the pinned color-only, frame-zero feature builder.
// Precise Metal SFUs and direct center pixel reads are hypotheses, not RTX parity.
struct PreShape { uint width,height,padded_height,frame,phase_math; };
inline uint nr_pre_hash(uint state) {
  uint word=((state>>((state>>28u)+4u))^state)*277803737u;
  return (word>>22u)^word;
}
inline float nr_pre_uniform(uint state) {
  uint word=((state>>((state>>28u)+4u))^state)*277803737u;
  return float(((word>>30u)^(word>>8u))+1u)*as_type<float>(0x33800000u);
}
inline float nr_pre_mul_rz_positive(float a,float b) {
  float rounded=a*b;
  float error=fma(a,b,-rounded);
  return error<0.0f?as_type<float>(as_type<uint>(rounded)-1u):rounded;
}
kernel void nr_pre_features(device const half4* image [[buffer(0)]],
                            device half* features [[buffer(1)]],
                            device atomic_uint* coverage [[buffer(2)]],
                            constant PreShape& shape [[buffer(3)]],
#if NR_PRE_EXACT_MATH
                            device const uchar* deltas [[buffer(4)]],device const uint2* escapes [[buffer(5)]],
                            constant uint& escape_count [[buffer(6)]],
#endif
                            uint pixel [[thread_position_in_grid]]) {
  if(pixel>=shape.width*shape.padded_height)return;
  uint x=pixel%shape.width,y=pixel/shape.width;
  uint seed=x*uint(-1918454973)^y*uint(-669632447)^shape.frame*uint(-1640531527)^608135816u;
  uint hash=nr_pre_hash(seed);
  float u0=nr_pre_uniform(hash*747796405u+uint(-1403630843));
  float u1=nr_pre_uniform(hash*uint(-93469191)+1192405134u);
  float u2=nr_pre_uniform(hash*uint(-895109107)+568162667u);
  float u3=nr_pre_uniform(hash*uint(-2094846927)+878960812u);
#if NR_PRE_EXACT_MATH
  float l0=nr_noise_exact_lg2_unit24(u0,deltas,escapes,escape_count)*as_type<float>(0x3f317218u);
  float l2=nr_noise_exact_lg2_unit24(u2,deltas,escapes,escape_count)*as_type<float>(0x3f317218u);
  float radius0=nr_noise_exact_sqrt_positive(l0*-2.0f,deltas,escapes,escape_count);
  float radius2=nr_noise_exact_sqrt_positive(l2*-2.0f,deltas,escapes,escape_count);
  float c1=nr_noise_exact_trig_unit24(u1,true,deltas,escapes,escape_count);
  float s1=nr_noise_exact_trig_unit24(u1,false,deltas,escapes,escape_count);
  float c3=nr_noise_exact_trig_unit24(u3,true,deltas,escapes,escape_count);
#else
  float l0=precise::log2(u0)*as_type<float>(0x3f317218u);
  float l2=precise::log2(u2)*as_type<float>(0x3f317218u);
  float radius0=precise::sqrt(l0*-2.0f),radius2=precise::sqrt(l2*-2.0f);
  float c1,s1,c3;
  if(shape.phase_math){
    float phase1=u1,phase3=u3;
    if(shape.phase_math==2u){
      phase1=nr_pre_mul_rz_positive(u1*as_type<float>(0x40c90fdbu),as_type<float>(0x3e22f983u));
      phase3=nr_pre_mul_rz_positive(u3*as_type<float>(0x40c90fdbu),as_type<float>(0x3e22f983u));
    }
    c1=precise::cospi(2.0f*phase1);s1=precise::sinpi(2.0f*phase1);c3=precise::cospi(2.0f*phase3);
  }
  else{float a1=u1*as_type<float>(0x40c90fdbu),a3=u3*as_type<float>(0x40c90fdbu);c1=precise::cos(a1);s1=precise::sin(a1);c3=precise::cos(a3);}
#endif
  uint source_y=y<shape.height?y:2u*shape.height-y-2u;
  half4 color=image[source_y*shape.width+x];
  half r=half(half(color.r-half(0.5f))*half(0.125f));
  half g=half(half(color.g-half(0.5f))*half(0.125f));
  half b=half(half(color.b-half(0.5f))*half(0.125f));
  half f[16]={half(radius0*c1),half(radius0*s1),half(radius2*c3),half(1),r,g,b,r,g,b,half(0),half(1),half(1),half(-1),half(-1),half(0)};
  // Original shared layout: first eight channels of 64 pixels, then last eight.
  uint tile=(y/8u)*(shape.width/8u)+x/8u,slot=(y%8u)*8u+x%8u;
  for(uint k=0;k<16u;++k){uint index=tile*1024u+(k/8u)*512u+slot*8u+k%8u;
    features[index]=f[k];atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);}
}
