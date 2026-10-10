#!/usr/bin/env python3
"""Restore selected module21 texture/surface semantics to CuMetal SSA MSL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


INPUT_SHA = "78ec43778093177bf769cc945a892d1d93b203a60cf809b310755ae7efc01a56"

OLD_MATH = """static inline float cm_rcp_approx_ftz_f32(float value) {
    return 1.0f / value;
}
static inline float cm_div_approx_ftz_f32(float numerator, float denominator) {
    return numerator / denominator;
}"""

NEW_MATH = """static inline float cm_rcp_approx_ftz_f32(
    float value, device const uchar* correction_rom) {
    uint input_bits = as_type<uint>(value);
    uint base_bits = as_type<uint>(1.0f / value);
    uint mantissa = input_bits & 0x7fffffu;
    uchar packed = correction_rom[mantissa >> 2u];
    uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
    if (code == 1u) base_bits -= 1u;
    else if (code == 2u) base_bits += 1u;
    return as_type<float>(base_bits);
}
static inline float cm_div_approx_ftz_f32(
    float numerator, float denominator,
    device const uchar* correction_rom) {
    return numerator * cm_rcp_approx_ftz_f32(denominator, correction_rom);
}
static inline ushort cm_float_to_half_rz_bits(float value) {
    uint bits = as_type<uint>(value);
    ushort sign = ushort((bits >> 16u) & 0x8000u);
    uint exponent = (bits >> 23u) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return ushort(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
    int half_exponent = int(exponent) - 127 + 15;
    if (half_exponent >= 31) return ushort(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        uint significand = mantissa | 0x800000u;
        return ushort(sign | ushort(significand >> uint(14 - half_exponent)));
    }
    return ushort(sign | ushort(uint(half_exponent) << 10u) |
                  ushort(mantissa >> 13u));
}
static inline float4 cm_sample_normalized(
    texture2d<float, access::read> input, float2 coordinate,
    uint2 dimensions) {
    if (!(coordinate.x >= 0.0f && coordinate.x < 1.0f &&
          coordinate.y >= 0.0f && coordinate.y < 1.0f))
        return float4(0.0f);
    float2 fixed12 = floor(coordinate * 4096.0f) * (1.0f / 4096.0f);
    int2 location = clamp(int2(floor(fixed12 * float2(dimensions))),
                          int2(0), int2(dimensions) - 1);
    return input.read(uint2(location));
}"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"module21 {label} anchor drift: {source.count(old)}")
    return source.replace(old, new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    if sha(source_bytes) != INPUT_SHA:
        raise SystemExit("CuMetal module21 base MSL identity mismatch")
    source = source_bytes.decode()
    source = replace_once(source, OLD_MATH, NEW_MATH, "math helper")
    signature = """kernel void main_kernel(
    constant uchar* main_kernel_param_0_224_ [[buffer(0)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    replacement = """kernel void fg_early_l6_l16(
    constant uchar* main_kernel_param_0_224_ [[buffer(0)]],
    device const uchar* cm_rcp_rom [[buffer(1)]],
    texture2d<float, access::read> cm_flow [[texture(0)]],
    texture2d<uint, access::read> cm_depth [[texture(1)]],
    texture2d<float, access::read> cm_input_mask [[texture(2)]],
    texture2d<half, access::write> cm_vector_output [[texture(3)]],
    texture2d<half, access::write> cm_mask_output [[texture(4)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    source = replace_once(source, signature, replacement, "signature")
    source = replace_once(
        source, "        v214 = v213 * 3277;",
        "        v214 = ushort((uint(v213) * 3277u) >> 16u);",
        "mul.hi.u16")
    source = replace_once(
        source, "    *reinterpret_cast<threadgroup ushort*>(v205) = v203;",
        "    *reinterpret_cast<threadgroup uchar*>(v205) = uchar(v203);",
        "first shared u8 store")
    source = replace_once(
        source, "        *reinterpret_cast<threadgroup ushort*>(v942) = v401;",
        "        *reinterpret_cast<threadgroup uchar*>(v942) = uchar(v401);",
        "second shared u8 store")
    source = replace_once(
        source, "        *reinterpret_cast<device ushort*>(v576) = v571;",
        "        *reinterpret_cast<device uchar*>(v576) = uchar(v571);",
        "global u8 store")
    source, div_count = re.subn(
        r"cm_div_approx_ftz_f32\((v\d+), (v\d+)\)",
        r"cm_div_approx_ftz_f32(\1, \2, cm_rcp_rom)", source)
    source, rcp_count = re.subn(
        r"cm_rcp_approx_ftz_f32\((v\d+)\)",
        r"cm_rcp_approx_ftz_f32(\1, cm_rcp_rom)", source)
    if div_count != 24 or rcp_count != 2:
        raise SystemExit(f"module21 approximate-site drift: div={div_count} rcp={rcp_count}")

    source = replace_once(source, """        v61 = 0;
        v62 = 0;
        v63 = 0;
        v64 = 0;""", """        uint4 cm_depth_a = cm_depth.read(uint2(v15, v19));
        v61 = cm_depth_a.x;
        v62 = cm_depth_a.y;
        v63 = cm_depth_a.z;
        v64 = cm_depth_a.w;""", "first depth texture")
    source = replace_once(source, """        v157 = 0x0p+0f;
        v158 = 0x0p+0f;
        v159 = 0x0p+0f;
        v160 = 0x0p+0f;""", """        float4 cm_flow_a = cm_sample_normalized(
            cm_flow, float2(v154, v155), uint2(v148, v149));
        v157 = cm_flow_a.x;
        v158 = cm_flow_a.y;
        v159 = cm_flow_a.z;
        v160 = cm_flow_a.w;""", "first flow texture")
    source = replace_once(source, """        v163 = 0x0p+0f;
        v164 = 0x0p+0f;
        v165 = 0x0p+0f;
        v166 = 0x0p+0f;""", """        float4 cm_mask_a =
            cm_input_mask.read(uint2(v15, v19));
        v163 = cm_mask_a.x;
        v164 = cm_mask_a.y;
        v165 = cm_mask_a.z;
        v166 = cm_mask_a.w;""", "first mask texture")
    source = replace_once(source, """            v259 = 0;
            v260 = 0;
            v261 = 0;
            v262 = 0;""", """            uint4 cm_depth_b = cm_depth.read(uint2(v219, v220));
            v259 = cm_depth_b.x;
            v260 = cm_depth_b.y;
            v261 = cm_depth_b.z;
            v262 = cm_depth_b.w;""", "second depth texture")
    source = replace_once(source, """            v355 = 0x0p+0f;
            v356 = 0x0p+0f;
            v357 = 0x0p+0f;
            v358 = 0x0p+0f;""", """            float4 cm_flow_b = cm_sample_normalized(
                cm_flow, float2(v352, v353), uint2(v346, v347));
            v355 = cm_flow_b.x;
            v356 = cm_flow_b.y;
            v357 = cm_flow_b.z;
            v358 = cm_flow_b.w;""", "second flow texture")
    source = replace_once(source, """            v361 = 0x0p+0f;
            v362 = 0x0p+0f;
            v363 = 0x0p+0f;
            v364 = 0x0p+0f;""", """            float4 cm_mask_b =
                cm_input_mask.read(uint2(v219, v220));
            v361 = cm_mask_b.x;
            v362 = cm_mask_b.y;
            v363 = cm_mask_b.z;
            v364 = cm_mask_b.w;""", "second mask texture")
    surface_anchor = """        v568 = v567 ? 0x0p+0f : 0x1p+0f;
        v569 = v549 ? 1 : 0;"""
    surface_restore = """        v568 = v567 ? 0x0p+0f : 0x1p+0f;
        cm_vector_output.write(
            half4(as_type<half>(cm_float_to_half_rz_bits(v553)),
                  as_type<half>(cm_float_to_half_rz_bits(v556)), 0.0h, 0.0h),
            uint2(v409, v410));
        cm_mask_output.write(half4(half(v568)), uint2(v409, v410));
        v569 = v549 ? 1 : 0;"""
    source = replace_once(source, surface_anchor, surface_restore,
                          "surface store")
    prefix = """// Exact selected module21 L6/L16 CuMetal SSA specialization.
// Six typed texture operations and two typed surface stores are restored from
// the pinned PTX. SM120 approximate division uses the accepted RCP ROM, and
// RG16F formatted stores use explicit round-toward-zero conversion.

"""
    output = (prefix + source).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    receipt = {
        "schema": "neuralstudio.fg.early-module21-cumetal-lowering.v1",
        "input_msl_sha256": INPUT_SHA,
        "output_msl_sha256": sha(output),
        "division_sites_rewritten": div_count,
        "reciprocal_sites_rewritten": rcp_count,
        "depth_texture_sites_restored": 2,
        "flow_texture_sites_restored": 2,
        "mask_texture_sites_restored": 2,
        "typed_surface_sites_restored": 2,
        "mul_hi_u16_sites_repaired": 1,
        "shared_u8_store_width_sites_repaired": 2,
        "global_u8_store_width_sites_repaired": 1,
        "vector_surface_rounding": "binary16 round-toward-zero",
        "expected_or_captured_output_embedded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = sha(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")).encode())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"output_msl_sha256={receipt['output_msl_sha256']}")
    print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
