// Netal texture staging against the host adapter's public resource ABI.
// Metal performs format conversion; no model, recorded frame or shader binary is embedded.
#include <metal_stdlib>
using namespace metal;

kernel void neuralstudio_sr_import_rgb9e5(texture2d<float, access::read> source [[texture(0)]],
                                         device half4* destination [[buffer(0)]],
                                         constant uint2& size [[buffer(1)]],
                                         uint2 position [[thread_position_in_grid]]) {
    if (any(position >= size)) return;
    destination[position.y * size.x + position.x] = half4(source.read(position));
}
kernel void neuralstudio_sr_import_depth32_stencil8(depth2d<float, access::read> source [[texture(0)]],
                                                    device float* destination [[buffer(0)]],
                                                    constant uint2& size [[buffer(1)]],
                                                    uint2 position [[thread_position_in_grid]]) {
    if (any(position >= size)) return;
    destination[position.y * size.x + position.x] = source.read(position);
}
kernel void neuralstudio_sr_import_exposure_r16(texture2d<float, access::read> source [[texture(0)]],
                                               device float* destination [[buffer(0)]],
                                               uint2 position [[thread_position_in_grid]]) {
    if (any(position != uint2(0))) return;
    destination[0] = source.read(uint2(0)).x;
}
kernel void neuralstudio_sr_export_rgb9e5(device const half4* source [[buffer(0)]],
                                         texture2d<float, access::write> destination [[texture(0)]],
                                         constant uint2& size [[buffer(1)]],
                                         uint2 position [[thread_position_in_grid]]) {
    if (any(position >= size)) return;
    destination.write(float4(source[position.y * size.x + position.x]), position);
}
