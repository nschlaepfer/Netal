#include <metal_stdlib>
using namespace metal;

// Codex-owned, experimental color boundary. Not part of the pinned NR model.
// Enabled only by an explicit experimental color profile; raw remains default.
// The shared rejection word preserves the original game image on ANY failure.
constant uint kInvalidSourceColor = 0x10u;
constant uint kNonInvertibleModelColor = 0x20u;

// Value-keyed repairs for FP32 evaluation at FP16 rounding boundaries. These
// constants come only from the color formulas, independently cross-checked at
// 80/160 decimal digits by verify-nr-color-rounding-cells.py. They are NOT frame,
// pixel, model-output or temporal-history data. The whole finite input domain is
// checked on the target GPU; preserve the unrepaired audit as a counterexample.
inline half color_forward_round(half source, half predicted) {
  switch (as_type<ushort>(source)) {
    case 0x0de8: return as_type<half>(ushort(0x1cc5));
    case 0x302f: return as_type<half>(ushort(0x35fd));
    case 0x3e97: return as_type<half>(ushort(0x3a7d));
    case 0x44e1: return as_type<half>(ushort(0x3b5f));
    case 0x4975: return as_type<half>(ushort(0x3bb3));
    case 0x532c: return as_type<half>(ushort(0x3bf1));
    case 0x60af: return as_type<half>(ushort(0x3bff));
    default: return predicted;
  }
}
inline half color_inverse_round(half source, half predicted) {
  switch (as_type<ushort>(source)) {
    case 0x3ad5: return as_type<half>(ushort(0x40a7));
    case 0x3bfd: return as_type<half>(ushort(0x5cae));
    default: return predicted;
  }
}

kernel void netal_nr_color_to_bounded(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<half, access::write> bounded [[texture(1)]],
    device atomic_uint* rejected [[buffer(0)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= source.get_width() || p.y >= source.get_height()) return;
  if (source.get_width() != bounded.get_width() || source.get_height() != bounded.get_height()) {
    atomic_fetch_or_explicit(rejected, kInvalidSourceColor, memory_order_relaxed);
    return;
  }
  float4 v = source.read(p);
  if (!all(isfinite(v)) || any(v.rgb < 0.0f)) {
    atomic_fetch_or_explicit(rejected, kInvalidSourceColor, memory_order_relaxed);
    bounded.write(half4(0.0h, 0.0h, 0.0h, 1.0h), p);
    return;
  }
  float3 linear = v.rgb / (1.0f + v.rgb);
  float3 encoded = select(12.92f * linear,
      1.055f * pow(linear, float3(1.0f / 2.4f)) - 0.055f,
      linear > 0.0031308f);
  // Round explicitly to FP16 before the texture store, rather than relying on
  // float-to-format conversion by the texture write path.
  half3 rounded = half3(encoded);
  rounded.x = color_forward_round(half(v.x), rounded.x);
  rounded.y = color_forward_round(half(v.y), rounded.y);
  rounded.z = color_forward_round(half(v.z), rounded.z);
  bounded.write(half4(rounded, half(v.a)), p);
}

// Invert the diagnostic transfer without discarding the model's color changes.
// Quantization makes this lossy. q == 1 has NO finite inverse: reject the WHOLE
// publication rather than silently clipping an unbounded reconstruction.
// The raw NR output/history texture is untouched; this edits only an export copy.
kernel void netal_nr_color_from_bounded(
    device half4* export_copy [[buffer(0)]],
    device atomic_uint* rejected [[buffer(1)]],
    constant uint2& shape [[buffer(2)]],
    uint2 p [[thread_position_in_grid]]) {
  if (any(p >= shape)) return;
  uint index = p.y * shape.x + p.x;
  float4 v = float4(export_copy[index]);
  if (!all(isfinite(v)) || any(v.rgb < 0.0f) || any(v.rgb >= 1.0f)) {
    atomic_fetch_or_explicit(rejected, kNonInvertibleModelColor, memory_order_relaxed);
    export_copy[index] = half4(0.0h, 0.0h, 0.0h, 1.0h);
    return;
  }
  float3 linear = select(v.rgb / 12.92f,
      pow((v.rgb + 0.055f) / 1.055f, float3(2.4f)), v.rgb > 0.04045f);
  float3 restored = linear / (1.0f - linear);
  if (!all(isfinite(restored)) || any(restored > 65504.0f)) {
    atomic_fetch_or_explicit(rejected, kNonInvertibleModelColor, memory_order_relaxed);
    export_copy[index] = half4(0.0h, 0.0h, 0.0h, 1.0h);
    return;
  }
  half3 rounded = half3(restored);
  rounded.x = color_inverse_round(half(v.x), rounded.x);
  rounded.y = color_inverse_round(half(v.y), rounded.y);
  rounded.z = color_inverse_round(half(v.z), rounded.z);
  export_copy[index] = half4(rounded, half(v.a));
}
