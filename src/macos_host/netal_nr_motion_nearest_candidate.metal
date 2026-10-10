#include <metal_stdlib>
using namespace metal;
// Experimental nearest-sample motion conversion, NOT a selected live policy.
// Host must first validate metadata/convention. Any rejection invalidates the
// WHOLE converted frame; zeros written on failure are not valid temporal input.
struct MotionParams { uint4 extent; uint4 content; uint2 input; float2 scale; };
kernel void netal_nr_motion_nearest_candidate(
    texture2d<half,access::read> source [[texture(0)]],
    texture2d<half,access::write> output [[texture(1)]],
    device atomic_uint* rejected [[buffer(0)]],
    constant MotionParams& a [[buffer(1)]],
    device atomic_uint* coverage [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  const uint2 actual=uint2(output.get_width(),output.get_height());
  if(any(p>=actual))return;
  half2 value=half2(0);
  const bool metadata_ok=all(a.extent.zw==actual) && all(actual==uint2(1920,1080)) &&
    all(a.input>0u) && all(a.input<=16384u) &&
    all(a.input==uint2(source.get_width(),source.get_height())) &&
    all(a.content.zw>0u) && all(a.content.xy<=actual) &&
    all(a.content.zw<=actual-min(a.content.xy,actual)) &&
    all(isfinite(a.scale)) && all(a.scale!=0.0f);
  if(!metadata_ok)atomic_fetch_or_explicit(rejected,4u,memory_order_relaxed);
  else if(all(p>=a.content.xy) && all(p-a.content.xy<a.content.zw)) {
    // Exact rational pixel-center nearest mapping. This avoids float error at
    // exact sample boundaries. The admitted bounds keep the uint products safe.
    const uint2 q=min(((2u*(p-a.content.xy)+1u)*a.input)/(2u*a.content.zw),a.input-1u);
    const float2 raw=float2(source.read(q).xy);
    if(!all(isfinite(raw)))atomic_fetch_or_explicit(rejected,1u,memory_order_relaxed);
    else {
      const float2 v=raw*a.scale;
      if(!all(isfinite(v)) || any(abs(v)>65504.0f))atomic_fetch_or_explicit(rejected,2u,memory_order_relaxed);
      else value=half2(v);
    }
  }
  output.write(half4(value,0,0),p);
  atomic_fetch_add_explicit(&coverage[p.y*actual.x+p.x],1u,memory_order_relaxed);
}
