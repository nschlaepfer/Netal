#!/usr/bin/env python3
"""Restore the selected module22/module24 semantics to CuMetal SSA MSL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


INPUTS = {
    "module22": "580f25e8841f298b485a768b4b6d493131901f1317dec87114afc0ca56647618",
    "module24-level10": "58311f06170dd464a83624aa6f1bc4f09369ddb813a4909013094f76af8f7c14",
    "module24-level11": "740c5f58e05f7a8f4214581fb90e60f2bfe895a87493e4bfcd311b06fb8e772b",
}

OLD_MATH = """static inline float cm_rcp_approx_ftz_f32(float value) {
    return 1.0f / value;
}
static inline float cm_div_approx_ftz_f32(float numerator, float denominator) {
    return numerator / denominator;
}"""

NEW_MATH = """static inline float cm_rne_rcp_f32(float value) {
    float candidate = 1.0f / value;
    uint bits = as_type<uint>(candidate);
    float below = as_type<float>(bits - 1u);
    float above = as_type<float>(bits + 1u);
    float candidate_residual = fabs(fma(-value, candidate, 1.0f));
    float below_residual = fabs(fma(-value, below, 1.0f));
    float above_residual = fabs(fma(-value, above, 1.0f));
    float best = candidate;
    float best_residual = candidate_residual;
    if (below_residual < best_residual ||
        (below_residual == best_residual && ((bits - 1u) & 1u) == 0u)) {
        best = below;
        best_residual = below_residual;
    }
    if (above_residual < best_residual ||
        (above_residual == best_residual && ((bits + 1u) & 1u) == 0u)) {
        best = above;
    }
    return best;
}
static inline float cm_rcp_approx_ftz_f32(
    float value, device const uchar* correction_rom) {
    uint input_bits = as_type<uint>(value);
    uint base_bits = as_type<uint>(cm_rne_rcp_f32(value));
    uint input_exponent = (input_bits >> 23u) & 0xffu;
    uint output_exponent = (base_bits >> 23u) & 0xffu;
    if ((input_bits >> 31u) != 0u || input_exponent == 0u ||
        input_exponent == 0xffu || output_exponent == 0u ||
        output_exponent == 0xffu)
        return as_type<float>(base_bits);
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
}"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label} anchor drift: {count}")
    return source.replace(old, new)


def lower_module22(source_bytes: bytes) -> tuple[bytes, dict[str, object]]:
    if sha(source_bytes) != INPUTS["module22"]:
        raise SystemExit("module22 base MSL identity mismatch")
    source = replace_once(source_bytes.decode(), OLD_MATH, NEW_MATH,
                          "module22 math")
    signature = """kernel void main_kernel(
    constant uchar* main_kernel_param_0_56_ [[buffer(0)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    replacement = """kernel void fg_resolution_module22(
    constant uchar* main_kernel_param_0_56_ [[buffer(0)]],
    device const uchar* cm_rcp_rom [[buffer(1)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    source = replace_once(source, signature, replacement,
                          "module22 signature")
    source, division_count = re.subn(
        r"cm_div_approx_ftz_f32\((v\d+), (v\d+)\)",
        r"cm_div_approx_ftz_f32(\1, \2, cm_rcp_rom)", source)
    if division_count != 12:
        raise SystemExit(f"module22 division-site drift: {division_count}")
    prefix = """// Exact selected module22 resolution-pyramid lowering.
// The SSA schedule comes from the pinned SM120 module. Approximate division
// is recovered with the accepted SM120 reciprocal correction ROM. Buffer
// pointer order and binary16 round-to-nearest-even stores remain unchanged.

"""
    output = ((prefix + source).rstrip() + "\n").encode()
    return output, {
        "input_msl_sha256": INPUTS["module22"],
        "output_msl_sha256": sha(output),
        "division_sites_rewritten": division_count,
        "binary16_store_rounding": "round-to-nearest-even",
        "restored_surface_sites": 0,
    }


def lower_module24(source_bytes: bytes, level: int) -> tuple[bytes, dict[str, object]]:
    key = f"module24-level{level}"
    if sha(source_bytes) != INPUTS[key]:
        raise SystemExit(f"module24 level{level} base MSL identity mismatch")
    source = replace_once(source_bytes.decode(), OLD_MATH, NEW_MATH,
                          f"module24 level{level} math")
    signature = """kernel void main_kernel(
    constant uchar* main_kernel_param_0_64_ [[buffer(0)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    replacement = f"""kernel void fg_resolution_module24_level{level}(
    constant uchar* main_kernel_param_0_64_ [[buffer(0)]],
    device const uchar* cm_rcp_rom [[buffer(1)]],
    texture2d<half, access::write> cm_output [[texture(0)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    source = replace_once(source, signature, replacement,
                          f"module24 level{level} signature")
    source, division_count = re.subn(
        r"cm_div_approx_ftz_f32\((v\d+), (v\d+)\)",
        r"cm_div_approx_ftz_f32(\1, \2, cm_rcp_rom)", source)
    if division_count != 14:
        raise SystemExit(
            f"module24 level{level} division-site drift: {division_count}")
    source = replace_once(
        source,
        """    uint v74;
    uint v75;
    device float* v564;
    device float* v565;
    device float* v566;""",
        """    threadgroup uchar* v74;
    threadgroup uchar* v75;
    threadgroup float* v564;
    threadgroup float* v565;
    threadgroup float* v566;""",
        f"module24 level{level} shared-pointer declarations")
    source = replace_once(
        source,
        """        v74 = v71 ? v73 : v72;
        v75 = v22 * 12 + v74;
        v564 = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(v75) + 0);
        *reinterpret_cast<device float*>(v564) = v537;
        v565 = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(v75) + 4);
        *reinterpret_cast<device float*>(v565) = v538;
        v566 = reinterpret_cast<device float*>(reinterpret_cast<device uchar*>(v75) + 8);
        *reinterpret_cast<device float*>(v566) = v539;""",
        """        v74 = v71 ? v73 : v72;
        v75 = reinterpret_cast<threadgroup uchar*>(v74 + v22 * 12u);
        v564 = reinterpret_cast<threadgroup float*>(v75 + 0);
        *v564 = v537;
        v565 = reinterpret_cast<threadgroup float*>(v75 + 4);
        *v565 = v538;
        v566 = reinterpret_cast<threadgroup float*>(v75 + 8);
        *v566 = v539;""",
        f"module24 level{level} shared-pointer use")
    tail = """    if (v536) {
        return;
    } else {
        return;
    }
}"""
    restored = """    if (v536) {
        return;
    } else {
        cm_output.write(
            half4(as_type<half>(cm_float_to_half_rz_bits(v527)),
                  as_type<half>(cm_float_to_half_rz_bits(v528)), 0.0h, 0.0h),
            uint2(v450, v452));
        return;
    }
}"""
    source = replace_once(source, tail, restored,
                          f"module24 level{level} surface store")
    pointer = "ABI offset16" if level == 10 else "ABI offset24"
    prefix = f"""// Exact selected module24 level-{level} resolution lowering.
// CuMetal's SSA schedule is specialized to selected level {level} ({pointer}).
// Approximate division uses the accepted SM120 reciprocal correction ROM and
// the unsupported RG16F surface store is restored with round-toward-zero.

"""
    output = ((prefix + source).rstrip() + "\n").encode()
    return output, {
        "input_msl_sha256": INPUTS[key],
        "output_msl_sha256": sha(output),
        "division_sites_rewritten": division_count,
        "selected_level": level,
        "selected_pointer": pointer,
        "restored_surface_sites": 1,
        "shared_address_space_repairs": 1,
        "surface_format": "RG16Float",
        "surface_rounding": "binary16 round-toward-zero",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module22", type=Path, required=True)
    parser.add_argument("--module24-level10", type=Path, required=True)
    parser.add_argument("--module24-level11", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    outputs: dict[str, dict[str, object]] = {}
    work = [
        ("module22.metal", lower_module22(args.module22.read_bytes())),
        ("module24_level10.metal",
         lower_module24(args.module24_level10.read_bytes(), 10)),
        ("module24_level11.metal",
         lower_module24(args.module24_level11.read_bytes(), 11)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (data, item) in work:
        (args.output_dir / name).write_bytes(data)
        outputs[name] = item
    receipt: dict[str, object] = {
        "schema": "neuralstudio.fg.resolution-l22-l24-cumetal-lowering.v1",
        "modules": outputs,
        "sm120_rcp_correction_rom_required": True,
        "expected_or_captured_output_embedded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = sha(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")).encode())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
