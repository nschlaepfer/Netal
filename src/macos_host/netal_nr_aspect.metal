#include <metal_stdlib>
using namespace metal;
struct AspectLayout { uint4 extent; uint4 content; };

// Full game image -> aspect-preserving model content rectangle. Padding is
// explicit opaque black; no game pixels are cropped or stretched.
kernel void netal_nr_fit_input(texture2d<half,access::sample> source [[texture(0)]],
                               texture2d<half,access::write> model [[texture(1)]],
                               constant AspectLayout& a [[buffer(0)]],
                               uint2 p [[thread_position_in_grid]]) {
  if(any(p>=a.extent.zw))return;
  half4 value=half4(0,0,0,1);
  if(all(p>=a.content.xy) && all(p-a.content.xy<a.content.zw)) {
    constexpr sampler linear_edge(coord::normalized,address::clamp_to_edge,filter::linear);
    float2 uv=(float2(p-a.content.xy)+0.5f)/float2(a.content.zw);
    value=source.sample(linear_edge,uv);
  }
  model.write(value,p);
}

// The publication target was copied bitwise from the original game output.
// Any rejected model pixel leaves that WHOLE original target unchanged.
kernel void netal_nr_fit_publish(device const half4* model [[buffer(0)]],
                                 device atomic_uint* rejected [[buffer(1)]],
                                 constant AspectLayout& a [[buffer(2)]],
                                 texture2d<half,access::write> target [[texture(0)]],
                                 uint2 p [[thread_position_in_grid]]) {
  if(any(p>=a.extent.xy) || atomic_load_explicit(rejected,memory_order_relaxed))return;
  float2 pos=float2(a.content.xy)+(float2(p)+0.5f)*float2(a.content.zw)/float2(a.extent.xy)-0.5f;
  // Clamp to content texel centers, not to the model border, so interpolation
  // never mixes the synthetic padding into the edges of the displayed image.
  pos=clamp(pos,float2(a.content.xy),float2(a.content.xy+a.content.zw-1u));
  uint2 lo=uint2(floor(pos)),hi=min(lo+1u,a.content.xy+a.content.zw-1u);
  float2 f=pos-float2(lo); uint stride=a.extent.z;
  float4 v=mix(mix(float4(model[lo.y*stride+lo.x]),float4(model[lo.y*stride+hi.x]),f.x),
               mix(float4(model[hi.y*stride+lo.x]),float4(model[hi.y*stride+hi.x]),f.x),f.y);
  target.write(half4(v),p);
}
