#!/usr/bin/env python3
"""Generate exact-shape 512x288 custom-convolution Metal sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SOURCES: dict[str, dict[int, int]] = {
    "src/fg_conv/custom_conv_pre.metal":
        {640: 512, 368: 288, 320: 256, 184: 144},
    "src/fg_conv/custom_conv0_c8.metal":
        {320: 256, 184: 144, 160: 128, 92: 72},
    "src/fg_conv/custom_conv1.metal":
        {160: 128, 92: 72, 80: 64, 46: 36},
    "src/fg_conv/custom_conv2.metal": {80: 64, 46: 36},
    "src/fg_connected_custom_l56_l83/connected_block0_bot0.metal":
        {160: 128, 92: 72, 80: 64, 79: 63, 45: 35},
    "src/fg_production_contract_l3_l87/connected_block0_bot1_production.metal":
        {320: 256, 184: 144, 160: 128, 159: 127, 91: 71},
    "src/fg_conv/custom_block1_conv0.metal":
        {640: 512, 368: 288, 320: 256, 184: 144},
    "src/fg_conv/custom_block1_conv1.metal":
        {320: 256, 184: 144, 160: 128, 92: 72},
    "src/fg_conv/custom_block1_conv2.metal": {160: 128, 92: 72},
    "src/fg_connected_custom_l56_l83/connected_block1_bot0.metal":
        {320: 256, 184: 144, 160: 128, 159: 127, 91: 71},
    "src/fg_connected_custom_l56_l83/connected_block1_bot1.metal":
        {640: 512, 368: 288, 320: 256, 319: 255, 183: 143},
}


def rewrite(source: str, replacements: dict[int, int], label: str) -> str:
    total = 0
    for before, after in replacements.items():
        pattern = re.compile(rf"(?<![0-9A-Za-z_]){before}(u?)(?![0-9A-Za-z_])")
        source, count = pattern.subn(
            lambda match: f"{after}{match.group(1)}", source
        )
        if count == 0:
            raise RuntimeError(f"{label}: spatial token {before} not found")
        total += count
    if total < 4:
        raise RuntimeError(f"{label}: implausibly small rewrite count {total}")
    return (
        "// GENERATED 512x288 selected-shape specialization; do not edit.\n"
        + source
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"preserve existing output directory: {args.output}")
    args.output.mkdir(parents=True)
    for relative, replacements in SOURCES.items():
        source = args.repository / relative
        if not source.is_file():
            raise RuntimeError(f"source missing: {relative}")
        target = args.output / source.name
        target.write_text(rewrite(source.read_text(), replacements, relative))
        print(f"generated={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
