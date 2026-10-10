#include <metal_stdlib>
using namespace metal;

// Explicit producer-supplied UI, premultiplied in the same numerical LDR color
// space as RGBA8 FG output. This does not infer/remove a HUD from a screenshot.
// The caller supplies distinct, equally sized input/output textures and waits
// for both producers before encoding. Scene interpolation never sees this UI.
kernel void fg_composite_hud(texture2d<float, access::read> scene [[texture(0)]],
    texture2d<float, access::read> premultipliedHUD [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    uint2 xy [[thread_position_in_grid]]) {
    if (xy.x >= output.get_width() || xy.y >= output.get_height()) return;
    float4 ui = premultipliedHUD.read(xy);
    float3 background = scene.read(xy).rgb;
    output.write(float4(ui.rgb + background * (1.0f-ui.a), 1.0f), xy);
}
