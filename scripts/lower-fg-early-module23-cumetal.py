#!/usr/bin/env python3
"""Specialize CuMetal's module23 SSA body to selected mode 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


INPUT_SHA = "d278535348df59fd462d8bb12ec5f90185fcf3e2b14f921a26c46b73f7d7f8ba"

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
}"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    if sha(source_bytes) != INPUT_SHA:
        raise SystemExit("CuMetal module23 input identity mismatch")
    source = source_bytes.decode()
    if source.count(OLD_MATH) != 1:
        raise SystemExit("CuMetal module23 math helper drift")
    source = source.replace(OLD_MATH, NEW_MATH)
    signature = """kernel void main_kernel(
    constant uchar* main_kernel_param_0_72_ [[buffer(0)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    replacement = """kernel void fg_early_l7_l17(
    constant uchar* main_kernel_param_0_72_ [[buffer(0)]],
    device const uchar* cm_rcp_rom [[buffer(1)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],"""
    if source.count(signature) != 1:
        raise SystemExit("CuMetal module23 signature drift")
    source = source.replace(signature, replacement)
    source, division_count = re.subn(
        r"cm_div_approx_ftz_f32\((v\d+), (v\d+)\)",
        r"cm_div_approx_ftz_f32(\1, \2, cm_rcp_rom)",
        source,
    )
    if division_count != 54:
        raise SystemExit(f"module23 division-site drift: {division_count}")
    store_anchor = "        v1234 = v1224;\n"
    store = """        v1234 = v1224;
        device half* cm_selected_output =
            reinterpret_cast<device half*>(v2);
        ulong cm_selected_index =
            (ulong(v1154) * ulong(v8) + ulong(v1153)) * 2ul;
        cm_selected_output[cm_selected_index] =
            as_type<half>(cm_float_to_half_rz_bits(v1224));
        cm_selected_output[cm_selected_index + 1ul] =
            as_type<half>(cm_float_to_half_rz_bits(v1225));
"""
    if source.count(store_anchor) != 1:
        raise SystemExit("module23 selected surface-store anchor drift")
    source = source.replace(store_anchor, store)
    prefix = """// Exact selected-mode lowering of canonical FG L7/L17 module23.
// The SSA schedule is CuMetal-derived from the pinned SM120 PTX. The accepted
// SM120 RCP ROM replaces approximate division, the constant bfind input was
// independently proven to yield 2, and the one unsupported RG16F SUST.P is
// restored as an explicit round-toward-zero half2 store. Mode 0 remains
// rejected by the host; its dead null input is retained for auditability.

"""
    output = (prefix + source).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    receipt = {
        "schema": "neuralstudio.fg.early-module23-cumetal-lowering.v1",
        "input_msl_sha256": INPUT_SHA,
        "output_msl_sha256": sha(output),
        "division_sites_rewritten": division_count,
        "folded_bfind_input": 536870912,
        "folded_bfind_output": 2,
        "restored_surface_store": "RG16F float2 to half2 round-toward-zero",
        "selected_mode": 1,
        "mode0_supported": False,
        "expected_output_bytes_embedded": False,
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
