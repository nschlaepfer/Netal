// SPDX-License-Identifier: Apache-2.0
// Exposure conversion behavior adapted from CuMetal's generated helper contract.
// See LICENSES/CuMetal-Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
// Netal exposure-state copy, authored against the host's 24-byte resource descriptor ABI.
// The supported contract is one float32 source channel -> one half destination channel.
// Conversion intentionally preserves the existing port's truncation behavior (including
// its infinity-on-overflow convention), rather than assuming Metal's default half rounding.
#include <metal_stdlib>
using namespace metal;

struct NetalExposureResource {
    ulong address;
    uint width;
    uint height;
    uint format;
    uint channels;
};

static ushort netal_exposure_half(float value) {
    const uint bits = as_type<uint>(value);
    const uint sign = (bits >> 16) & 0x8000;
    const uint exp = (bits >> 23) & 255;
    const uint frac = bits & 0x7fffff;
    // The legacy center-sampling path canonicalizes nonfinite inputs and signed zero.
    if (exp == 255) return ushort(0x7e00);
    if (exp == 0) return ushort(0);  // The legacy sampler flushes float32 subnormals too.
    if (exp > 142) return ushort(sign | 0x7c00);
    if (exp < 103) return ushort(sign);
    if (exp < 113) return ushort(sign | ((frac | 0x800000) >> (126 - exp)));
    return ushort(sign | ((exp - 112) << 10) | (frac >> 13));
}

kernel void cuda_copy_exposure_kernel(constant ulong2& handles [[buffer(0)]],
                                      uint lane [[thread_position_in_grid]]) {
    if (lane != 0) return;
    device const NetalExposureResource* destination =
        reinterpret_cast<device const NetalExposureResource*>(handles.x);
    device const NetalExposureResource* source =
        reinterpret_cast<device const NetalExposureResource*>(handles.y);
    if (source->width != 1 || source->height != 1 || source->channels != 1 || source->format != 32 ||
        destination->width != 1 || destination->height != 1 || destination->channels != 1 || destination->format != 16)
        return;
    const float value = *reinterpret_cast<device const float*>(source->address);
    *reinterpret_cast<device ushort*>(destination->address) = netal_exposure_half(value);
}
