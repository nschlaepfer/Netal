#!/usr/bin/env python3
"""Specialize the accepted CuMetal module19 body for selected subrect mode 1."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(before, after)


def generate(source: str) -> str:
    source = replace_once(
        source,
        """    if (descriptor->format == 16u) {
        return float(*reinterpret_cast<device const half*>(data + element * 2ul));
    }
    return *reinterpret_cast<device const float*>(data + element * 4ul);
""",
        """    if (descriptor->format == 8u) {
        return float(data[element]) * (1.0f / 255.0f);
    }
    if (descriptor->format == 16u) {
        return float(*reinterpret_cast<device const half*>(data + element * 2ul));
    }
    return *reinterpret_cast<device const float*>(data + element * 4ul);
""",
        "R8 UNORM texture read",
    )
    source = replace_once(
        source,
        """        if (descriptor->format == 16u) {
            *reinterpret_cast<device ushort*>(data +
                (base + ulong(component)) * 2ul) =
                    cm_float_to_half_rz_bits(values[component]);
        } else {
            *reinterpret_cast<device float*>(data +
                (base + ulong(component)) * 4ul) = values[component];
        }
""",
        """        if (descriptor->format == 8u) {
            const float clamped = saturate(values[component]);
            const uint fixed12 = uint(clamped * 4096.0f);
            data[base + ulong(component)] = uchar(min(
                255u, (fixed12 * 255u + 2047u) >> 12u));
        } else if (descriptor->format == 16u) {
            *reinterpret_cast<device ushort*>(data +
                (base + ulong(component)) * 2ul) =
                    cm_float_to_half_rz_bits(values[component]);
        } else {
            *reinterpret_cast<device float*>(data +
                (base + ulong(component)) * 4ul) = values[component];
        }
""",
        "R8 UNORM surface write",
    )
    source = replace_once(
        source,
        """kernel void fg_postprocess_l86_mode0(
    device uchar* level0 [[buffer(0)]],
    device const uchar* level1 [[buffer(1)]],
    device const uchar* level2 [[buffer(2)]],
    device const uchar* level3 [[buffer(3)]],
    device const uchar* level4 [[buffer(4)]],
    device const uchar* correction_rom [[buffer(5)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],
    uint3 cm_threadgroup_position [[threadgroup_position_in_grid]],
    uint cm_lane_id [[thread_index_in_simdgroup]]
) {
""",
        """static inline uint cm_parameter_u32(constant uchar* parameters,
                                    uint offset) {
    return *reinterpret_cast<constant uint*>(parameters + offset);
}

static inline ulong cm_parameter_u64(constant uchar* parameters,
                                     uint offset) {
    return *reinterpret_cast<constant ulong*>(parameters + offset);
}

kernel void fg_postprocess_l86_mode1_subrect(
    constant uchar* parameters [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    uint3 cm_thread_position [[thread_position_in_threadgroup]],
    uint3 cm_threadgroup_position [[threadgroup_position_in_grid]],
    uint cm_lane_id [[thread_index_in_simdgroup]]
) {
""",
        "kernel ABI",
    )
    source = replace_once(
        source,
        """    v1 = 0u;
    v2 = 0u;
    v3 = 0ul;
    v4 = 0ul;
    v5 = 0u;
    v6 = const_cast<device uchar*>(level3);
    v7 = const_cast<device uchar*>(level2);
    v8 = const_cast<device uchar*>(level1);
    v9 = level0;
    v10 = 320u;
    v11 = 180u;
    v12 = const_cast<device uchar*>(level4);
""",
        """    v1 = cm_parameter_u32(parameters, 72u);
    v2 = cm_parameter_u32(parameters, 76u);
    v3 = cm_parameter_u64(parameters, 64u);
    v4 = cm_parameter_u64(parameters, 56u);
    v5 = cm_parameter_u32(parameters, 48u);
    v6 = reinterpret_cast<device uchar*>(cm_parameter_u64(parameters, 24u));
    v7 = reinterpret_cast<device uchar*>(cm_parameter_u64(parameters, 16u));
    v8 = reinterpret_cast<device uchar*>(cm_parameter_u64(parameters, 8u));
    v9 = reinterpret_cast<device uchar*>(cm_parameter_u64(parameters, 0u));
    v10 = cm_parameter_u32(parameters, 40u);
    v11 = cm_parameter_u32(parameters, 44u);
    v12 = reinterpret_cast<device uchar*>(cm_parameter_u64(parameters, 32u));
""",
        "selected parameter decode",
    )
    header = """// GENERATED FILE: exact selected module19 mode-1 specialization.
// Source is the accepted CuMetal SSA body; generation fails closed on drift.
// R8 typed texture/surface behavior is reconstructed in the descriptor helpers.

"""
    return header + source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"preserve existing generated source: {args.output}")
    result = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f"generated={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
