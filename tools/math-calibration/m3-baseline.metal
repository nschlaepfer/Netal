#include <metal_stdlib>
using namespace metal;
// Synthetic positive-normal FP32 domain; no model or captured frame input.
kernel void netal_rcp_baseline(device uint* output [[buffer(0)]], device atomic_uint* counts [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= 16777216u) return;
    uint bits = ((127u + (gid >> 23u)) << 23u) | (gid & 0x7fffffu);
    output[gid] = as_type<uint>(1.0f / as_type<float>(bits));
    atomic_fetch_add_explicit(&counts[gid], 1u, memory_order_relaxed);
}
kernel void netal_rsq_baseline(device uint* output [[buffer(0)]], device atomic_uint* counts [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= 16777216u) return;
    uint bits = ((128u + (gid >> 23u)) << 23u) | (gid & 0x7fffffu);
    output[gid] = as_type<uint>(precise::rsqrt(as_type<float>(bits)));
    atomic_fetch_add_explicit(&counts[gid], 1u, memory_order_relaxed);
}
