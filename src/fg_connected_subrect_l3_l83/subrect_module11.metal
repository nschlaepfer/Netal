#include <metal_stdlib>

using namespace metal;

struct SubrectModule11Parameters {
  uint resource_width;
  uint resource_height;
  uint origin_x;
  uint origin_y;
  uint active_width;
  uint active_height;
  uint output_width;
  uint output_height;
  uint mode;
};

inline ushort subrect_m11_half_rz_bits(float value) {
  const uint bits=as_type<uint>(value);
  const ushort sign=ushort((bits>>16u)&0x8000u);
  const uint exponent=(bits>>23u)&0xffu,mantissa=bits&0x7fffffu;
  if(exponent==0xffu)return ushort(sign|(mantissa?0x7e00u:0x7c00u));
  const int half_exponent=int(exponent)-112;
  if(half_exponent>=31)return ushort(sign|0x7c00u);
  if(half_exponent<=0){
    if(half_exponent < -10)return sign;
    return ushort(sign|ushort((mantissa|0x800000u)>>uint(14-half_exponent)));
  }
  return ushort(sign|ushort(uint(half_exponent)<<10u)|ushort(mantissa>>13u));
}

inline half subrect_m11_half_rz(float value){
  return as_type<half>(subrect_m11_half_rz_bits(value));
}

inline half4 subrect_m11_half4_rz(float3 value){
  return half4(subrect_m11_half_rz(value.x),subrect_m11_half_rz(value.y),
               subrect_m11_half_rz(value.z),half(0.0h));
}

inline float subrect_m11_luma(float3 value){
  const float green=value.y*as_type<float>(uint(0x3f1645a2u));
  const float red_green=fma(value.x,as_type<float>(uint(0x3e991687u)),green);
  return saturate(fma(value.z,as_type<float>(uint(0x3de978d5u)),red_green));
}

kernel void fg_subrect_module11(
    texture2d<float,access::read> source [[texture(0)]],
    texture2d<half,access::read> history [[texture(1)]],
    texture2d<half,access::write> output_color [[texture(2)]],
    texture2d<half,access::write> output_metric [[texture(3)]],
    texture2d<half,access::write> next_history [[texture(4)]],
    constant SubrectModule11Parameters& p [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]) {
  if(position.x>=p.output_width||position.y>=p.output_height||p.mode!=2u)
    return;
  const uint2 local=position*2u;
  if(local.x+1u>=p.active_width||local.y+1u>=p.active_height)return;
  const uint2 base=local+uint2(p.origin_x,p.origin_y);
  if(base.x+1u>=p.resource_width||base.y+1u>=p.resource_height)return;
  const uint2 p00=base,p10=base+uint2(1,0),p01=base+uint2(0,1),
              p11=base+uint2(1,1);
  const float4 c00=source.read(p00),c10=source.read(p10),
               c01=source.read(p01),c11=source.read(p11);
  const uint2 l00=local,l10=local+uint2(1,0),l01=local+uint2(0,1),
              l11=local+uint2(1,1);
  next_history.write(subrect_m11_half4_rz(c00.xyz),l00);
  next_history.write(subrect_m11_half4_rz(c10.xyz),l10);
  next_history.write(subrect_m11_half4_rz(c01.xyz),l01);
  next_history.write(subrect_m11_half4_rz(c11.xyz),l11);
  const float m00=abs(subrect_m11_luma(c00.xyz)-subrect_m11_luma(
      float3(history.read(l00).xyz)));
  const float m10=abs(subrect_m11_luma(c10.xyz)-subrect_m11_luma(
      float3(history.read(l10).xyz)));
  const float m01=abs(subrect_m11_luma(c01.xyz)-subrect_m11_luma(
      float3(history.read(l01).xyz)));
  const float m11=abs(subrect_m11_luma(c11.xyz)-subrect_m11_luma(
      float3(history.read(l11).xyz)));
  const float3 average=((c00.xyz+c10.xyz)+(c01.xyz+c11.xyz))*0.25f;
  const float metric=min(min(m00,m10),min(m01,m11));
  output_color.write(subrect_m11_half4_rz(average),position);
  output_metric.write(half4(subrect_m11_half_rz(metric),0,0,0),position);
}
