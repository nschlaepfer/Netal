#include <metal_stdlib>

using namespace metal;

struct SubrectModule15Parameters {
  uint resource_width;
  uint resource_height;
  uint origin_x;
  uint origin_y;
  uint active_width;
  uint active_height;
  uint output_width;
  uint output_height;
  float2 input_inverse;
  float2 output_inverse;
  float2 lower_coordinate;
  float2 upper_coordinate;
  float2 output_scale;
};

struct SubrectModule15Candidate { int2 offset; float scalar; };

inline void subrect_m15_consider(
    thread SubrectModule15Candidate& selected,int2 offset,float scalar){
  if(scalar<selected.scalar){selected.offset=offset;selected.scalar=scalar;}
}

inline uint subrect_m15_point(float normalized,uint extent,float inverse,
                              float lower,float upper){
  const float mapped=fma(normalized,float(extent),0.0f);
  const float coordinate=clamp(
      fma(inverse,mapped,as_type<float>(0xb4000000u)),lower,upper);
  return min((uint(coordinate*4096.0f)*extent)>>12u,extent-1u);
}

inline uint2 subrect_m15_local(
    float2 normalized,constant SubrectModule15Parameters& p){
  return uint2(subrect_m15_point(normalized.x,p.active_width,
                                 p.input_inverse.x,p.lower_coordinate.x,
                                 p.upper_coordinate.x),
               subrect_m15_point(normalized.y,p.active_height,
                                 p.input_inverse.y,p.lower_coordinate.y,
                                 p.upper_coordinate.y));
}

inline uint2 subrect_m15_resource(
    float2 normalized,constant SubrectModule15Parameters& p){
  return subrect_m15_local(normalized,p)+uint2(p.origin_x,p.origin_y);
}

inline uint subrect_m15_pack_depth(float scalar,int2 offset){
  const uint bits=as_type<uint>(clamp(scalar,0.0f,1.0f));
  const uint bit29=bits&0x20000000u;
  const bool bit29_zero=bit29==0u,center=all(offset==int2(0));
  const uint retained=(center?0x1fffffffu:
                       (bit29_zero?0x3ffffffeu:0x3fffffffu))&bits;
  const uint center_or_noncenter=center?0x80000000u:0x40000000u;
  const uint positive_x_flag=bit29_zero?1u:0x40000000u;
  const uint base=((bit29>>29u)==uint(center))?center_or_noncenter:0u;
  return base|(offset.y>0?0x80000000u:0u)|
         (offset.x>0?positive_x_flag:0u)|retained;
}

kernel void fg_subrect_module15(
    texture2d<half,access::read> vector_input [[texture(0)]],
    texture2d<float,access::read> scalar_input [[texture(1)]],
    texture2d<float,access::write> vector_output [[texture(2)]],
    texture2d<uint,access::write> depth_output [[texture(3)]],
    constant SubrectModule15Parameters& p [[buffer(0)]],
    uint2 position [[thread_position_in_grid]]){
  if(position.x>=p.output_width||position.y>=p.output_height)return;
  const float2 normalized=(float2(position)+0.5f)*p.output_inverse;
  const float2 step=p.input_inverse;
  const uint2 center=subrect_m15_resource(normalized,p);
  const uint2 lower_left=subrect_m15_resource(normalized-step,p);
  const uint2 lower_right=subrect_m15_resource(
      normalized+float2(step.x,-step.y),p);
  const uint2 upper_left=subrect_m15_resource(
      normalized+float2(-step.x,step.y),p);
  const uint2 upper_right=subrect_m15_resource(normalized+step,p);
  SubrectModule15Candidate selected{int2(0),scalar_input.read(center).x};
  subrect_m15_consider(selected,int2(-1,-1),scalar_input.read(lower_left).x);
  subrect_m15_consider(selected,int2(1,-1),scalar_input.read(lower_right).x);
  subrect_m15_consider(selected,int2(-1,1),scalar_input.read(upper_left).x);
  subrect_m15_consider(selected,int2(1,1),scalar_input.read(upper_right).x);
  const uint2 vector_coordinate=subrect_m15_resource(
      normalized+float2(selected.offset)*step,p);
  const float2 vector_value=float2(vector_input.read(vector_coordinate).xy);
  vector_output.write(float4(vector_value*p.output_scale,0,0),position);
  depth_output.write(subrect_m15_pack_depth(selected.scalar,selected.offset),
                     position);
}
