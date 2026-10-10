// EXPERIMENTAL: pinned full-size, no-depth/no-mask temporal feature composition.
// Base features must be freshly computed with the current noise-frame index.
// Only history RGB channels7..9 differ beyond that base in the captured class.
// Sampler implementation is imported, versioned and owned by Fable. No parity
// claim until this tensor and the complete Block0/encoder outputs match RTX.
#include <metal_stdlib>
using namespace metal;
#include "nr_pre_temporal_support.h"
#include "nr_tex_sample.metal"
#ifndef NR_PRE_HISTORY_Q8
#define NR_PRE_HISTORY_Q8 0
#endif
#ifndef NR_PRE_HISTORY_RTX
#define NR_PRE_HISTORY_RTX 0
#endif
#if NR_PRE_HISTORY_Q8 || NR_PRE_HISTORY_RTX
#define NR_PRE_TEX_ACCESS access::read
#else
#define NR_PRE_TEX_ACCESS access::sample
#endif
struct NrPreTemporalShape { uint width,height,padded_height,reserved; };
inline float pt_ftz(float x){uint b=as_type<uint>(x);return (b&0x7f800000u)==0u?as_type<float>(b&0x80000000u):x;}
inline float pt_add(float a,float b){return pt_ftz(pt_ftz(a)+pt_ftz(b));}
inline float pt_sub(float a,float b){return pt_ftz(pt_ftz(a)-pt_ftz(b));}
inline float pt_mul(float a,float b){return pt_ftz(pt_ftz(a)*pt_ftz(b));}
inline float pt_fma(float a,float b,float c){return pt_ftz(fma(pt_ftz(a),pt_ftz(b),pt_ftz(c)));}
inline float pt_div(float a,float b,device const uchar* rom){return pt_mul(a,nr_rcp_rom(b,rom));}
inline float4 pt_sample(texture2d<float,NR_PRE_TEX_ACCESS> tex,float2 uv){
#if NR_PRE_HISTORY_RTX
  return nr_tex2d_bilinear_rtx(tex,uv.x,uv.y);
#elif NR_PRE_HISTORY_Q8
  return nr_tex2d_bilinear_q8(tex,uv.x,uv.y);
#else
  return nr_tex2d_bilinear_hw(tex,uv.x,uv.y);
#endif
}
struct PtAxis {float w0,w3,sum,t0,t12,t2;};
inline PtAxis pt_axis(float normalized,float dimension,device const uchar* rom){
  // Original SM120 SASS pre10d0/10e0 and1120/1150 contracts these products.
  float c=pt_add(floor(pt_fma(dimension,normalized,-0.5f)),0.5f);
  float f=min(max(pt_fma(dimension,normalized,-c),0.0f),1.0f),f2=pt_mul(f,f),f3=pt_mul(f,f2);
  float w0=pt_fma(pt_add(f,f3),-0.5f,f2);
  float w1=pt_add(pt_fma(f3,1.5f,-pt_mul(f2,2.5f)),1.0f);
  float w3=pt_mul(pt_sub(f3,f2),0.5f);
  float w2=pt_sub(pt_sub(pt_sub(1.0f,w0),w1),w3),sum=pt_add(w1,w2);
  float hi=pt_add(dimension,-0.5f);
  return {w0,w3,sum,min(max(pt_add(c,-1.0f),0.5f),hi),
    min(max(pt_fma(w2,nr_rcp_rom(sum,rom),c),0.5f),hi),min(max(pt_add(c,2.0f),0.5f),hi)};
}
// Preserve the pre-block's multiply -> FMA -> multiply coordinate transform;
// algebraically cancelling dimensions would remove NVIDIA rounding points.
inline float pt_history_uv(float position,float dimension,float invdim,device const uchar* rom){
  return pt_mul(invdim,pt_fma(dimension,pt_mul(nr_rcp_rom(dimension,rom),position),0.0f));
}
kernel void nr_pre_temporal_features(device const half* base [[buffer(0)]],device half* result [[buffer(1)]],
    device atomic_uint* coverage [[buffer(2)]],device const uchar* rom [[buffer(3)]],
    constant NrPreTemporalShape& shape [[buffer(4)]],texture2d<float,NR_PRE_TEX_ACCESS> history [[texture(0)]],
    texture2d<float,NR_PRE_TEX_ACCESS> motion [[texture(1)]],uint2 p [[thread_position_in_grid]]){
  if(shape.width!=1920u||shape.height!=1080u||shape.padded_height!=1152u||shape.reserved!=0u)return;
  if(p.x>=shape.width||p.y>=shape.padded_height)return;
  const float W=float(shape.width),H=float(shape.height),iw=1.0f/W,ih=1.0f/H;
  uint sy=p.y<shape.height?p.y:2u*shape.height-p.y-2u;
  float u=pt_div(pt_add(float(p.x),0.5f),W,rom),v=pt_div(pt_add(float(sy),0.5f),H,rom);
  float2 muv(pt_mul(pt_fma(u,W,0.0f),iw),pt_mul(pt_fma(v,H,0.0f),ih));
  float4 mv=pt_sample(motion,muv);
  PtAxis x=pt_axis(pt_fma(mv.x,iw,u),W,rom);
  PtAxis y=pt_axis(pt_fma(mv.y,ih,v),H,rom);
  float x0=pt_history_uv(x.t0,W,iw,rom),x12=pt_history_uv(x.t12,W,iw,rom),x2=pt_history_uv(x.t2,W,iw,rom);
  float y0=pt_history_uv(y.t0,H,ih,rom),y12=pt_history_uv(y.t12,H,ih,rom),y2=pt_history_uv(y.t2,H,ih,rom);
  float4 a=pt_sample(history,float2(x0,y12)),b=pt_sample(history,float2(x12,y0));
  float4 c=pt_sample(history,float2(x12,y12)),d=pt_sample(history,float2(x12,y2)),e=pt_sample(history,float2(x2,y12));
  float wa=pt_mul(x.w0,y.sum),wb=pt_mul(y.w0,x.sum),wc=pt_mul(x.sum,y.sum),wd=pt_mul(y.w3,x.sum),we=pt_mul(x.w3,y.sum);
  float inv=nr_rcp_rom(pt_add(pt_add(pt_add(pt_add(wb,wa),wc),wd),we),rom);half h[3];
  for(uint k=0;k<3;++k){float z=pt_mul(wb,b[k]);z=pt_fma(wa,a[k],z);z=pt_fma(wc,c[k],z);z=pt_fma(wd,d[k],z);z=pt_fma(we,e[k],z);
    half raw=half(pt_mul(inv,z));h[k]=half(half(raw-half(0.5f))*half(0.125f));}
  uint tile=(p.y/8u)*(shape.width/8u)+p.x/8u,slot=(p.y%8u)*8u+p.x%8u;
  for(uint k=0;k<16u;++k){uint index=tile*1024u+(k/8u)*512u+slot*8u+k%8u;
    result[index]=(k>=7u&&k<=9u)?h[k-7u]:base[index];atomic_fetch_add_explicit(&coverage[index],1u,memory_order_relaxed);}
}
