#!/usr/bin/env python3
"""Verify the independent SM120 semantics probes used by FG launch 54."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Any


RIG_SCHEMA = "neuralstudio.fg.mid-l54-semantics-rig.v1"
RCP_ROM_SHA256 = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
RTX_SOURCE_SHA256 = "fcc3ddf8662c475e24623de0f2cad30cf306d096b85e50a975d74808b4dc4b9f"
RTX_HARNESS_SHA256 = "a1ae161376d2d6ad64e5a93bfb33d9b01b0f7933d0d33a56a02b3b43fcb356cb"
MAC_SOURCE_SHA256 = "77ac6cd68d5df86d94f1c7aa0e201f0dc6f5f7ede1c5998c40590459ebc62ffc"
MAC_HOST_SHA256 = "b1942833497499860c7713bbd82b8543b45f4283ba5af92798e0a5e9320cba76"
MAC_HARNESS_SHA256 = "c20d6164bbb8640fdd47bee715d288b5310da66d0680d29dc8f52a564a8583fe"
POINT_SHA256 = "e187a9aa644622698b0175e0485b02ecb0913cdc299c195f7c3c62c116958683"
PAIR_SHA256 = "496a4dfa79184edc49ae97a2492ca2db9c902d45b3faf5d82dde732077aad3a9"
DENSE_RTX_SHA256 = "04e80f5f88555cb75b07e6f40bb9eb502d3faa7485c1c686d8b72adeb5fa7bb1"
DENSE_MAC_SHA256 = "47c25084a256afdef6c53fc19269fe08b6ca2d3bebdf1381278ed6d321f58aa2"
WIDTH = 320
HEIGHT = 180
OFFSET_BITS = (
    0x0000, 0x8000, 0x1400, 0x9400, 0x1800,
    0x9800, 0x1C00, 0x9C00, 0x2000,
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def half_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def selected_rcp(value: float, rom: bytes) -> float:
    input_bits = f32_bits(value)
    output_bits = f32_bits(f32(1.0 / value))
    mantissa = input_bits & 0x7FFFFF
    code = (rom[mantissa >> 2] >> ((mantissa & 3) * 2)) & 3
    require(code != 3, "RCP ROM contains unsupported sampler denominator")
    if code == 1:
        output_bits -= 1
    elif code == 2:
        output_bits += 1
    return bits_f32(output_bits)


def normalized_coordinate(index: int, extent: int, delta: float,
                          rom: bytes) -> float:
    numerator = f32(f32(float(index)) + f32(0.5))
    quotient = f32(numerator * selected_rcp(f32(float(extent)), rom))
    return f32(quotient + f32(delta))


def selected_index(index: int, extent: int, delta: float,
                   rom: bytes) -> int:
    coordinate = normalized_coordinate(index, extent, delta, rom)
    if not (coordinate >= 0.0 and coordinate < 1.0):
        return -1
    fixed = int(coordinate * 4096.0)
    result = (fixed * extent) >> 12
    return result if 0 <= result < extent else -1


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-root", type=Path, required=True)
    parser.add_argument("--mac-dense", type=Path, required=True)
    parser.add_argument("--rcp-rom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    try:
        rig = json.loads((args.rtx_root / "rig-receipt.json").read_text())
        require(rig.get("schema") == RIG_SCHEMA and rig.get("result") == "pass",
                "semantics rig receipt identity drift")
        require(rig.get("source_sha256") == RTX_SOURCE_SHA256,
                "semantics RTX source drift")
        require(rig.get("process_count") == 2 and
                len(set(rig.get("process_ids", []))) == 2 and
                rig.get("repeat_equal") is True,
                "semantics fresh-process repeat gate failed")
        require(rig.get("expected_frame_bytes_read") is False and
                rig.get("archived_intermediates_read") is False and
                rig.get("address_fields_recorded") is False,
                "semantics forbidden-ingress/address gate failed")
        require(sha256_file(repository / "src/fg_mid_l54/rtx_semantics_probe.cu") ==
                RTX_SOURCE_SHA256, "local semantics RTX source drift")
        require(sha256_file(repository / "scripts/probe-fg-mid-l54-semantics.ps1") ==
                RTX_HARNESS_SHA256, "local semantics RTX harness drift")
        require(sha256_file(repository / "src/fg_mid_l54/semantics_probe.metal") ==
                MAC_SOURCE_SHA256, "local semantics Metal source drift")
        require(sha256_file(repository / "src/fg_mid_l54/macos_semantics_probe.mm") ==
                MAC_HOST_SHA256, "local semantics Mac host drift")
        require(sha256_file(repository / "scripts/probe-macos-fg-mid-l54-semantics.sh") ==
                MAC_HARNESS_SHA256, "local semantics Mac harness drift")

        rom = args.rcp_rom.read_bytes()
        require(len(rom) == 2097152 and sha256_bytes(rom) == RCP_ROM_SHA256,
                "semantics RCP ROM identity drift")
        for name, expected_hash in (
            ("point-selected-u32.bin", POINT_SHA256),
            ("pair-semantics.bin", PAIR_SHA256),
            ("dense-ex2-minus2-to0-u32.bin", DENSE_RTX_SHA256),
        ):
            first = args.rtx_root / "process-a" / name
            second = args.rtx_root / "process-b" / name
            require(sha256_file(first) == expected_hash,
                    f"{name}: RTX output identity drift")
            require(first.read_bytes() == second.read_bytes(),
                    f"{name}: fresh-process repeat mismatch")

        point_bytes = (args.rtx_root / "process-a/point-selected-u32.bin").read_bytes()
        require(len(point_bytes) == WIDTH * HEIGHT * 9 * 9 * 4,
                "point-sampler record count drift")
        points = array("I")
        points.frombytes(point_bytes)
        offsets = tuple(half_value(bits) for bits in OFFSET_BITS)
        x_indices = [[selected_index(x, WIDTH, delta, rom)
                      for delta in offsets] for x in range(WIDTH)]
        y_indices = [[selected_index(y, HEIGHT, delta, rom)
                      for delta in offsets] for y in range(HEIGHT)]
        point_mismatches = 0
        cursor = 0
        for y in range(HEIGHT):
            for x in range(WIDTH):
                for dy in range(9):
                    selected_y = y_indices[y][dy]
                    for dx in range(9):
                        selected_x = x_indices[x][dx]
                        expected = (
                            0 if selected_x < 0 or selected_y < 0
                            else 1 + selected_y * WIDTH + selected_x
                        )
                        point_mismatches += points[cursor] != expected
                        cursor += 1
        require(point_mismatches == 0,
                f"fixed12 normalized point rule mismatch: {point_mismatches}")

        dense_rtx_bytes = (
            args.rtx_root / "process-a/dense-ex2-minus2-to0-u32.bin"
        ).read_bytes()
        dense_mac_bytes = args.mac_dense.read_bytes()
        require(sha256_bytes(dense_rtx_bytes) == DENSE_RTX_SHA256,
                "dense RTX EX2 identity drift")
        require(sha256_bytes(dense_mac_bytes) == DENSE_MAC_SHA256,
                "dense Metal EX2 identity drift")
        require(len(dense_rtx_bytes) == len(dense_mac_bytes) == 16777220,
                "dense EX2 byte count drift")
        dense_rtx = array("I")
        dense_rtx.frombytes(dense_rtx_bytes)
        dense_mac = array("I")
        dense_mac.frombytes(dense_mac_bytes)
        deltas = Counter(int(left) - int(right)
                         for left, right in zip(dense_rtx, dense_mac))
        expected_deltas = {-2: 403306, -1: 3312594, 0: 478405}
        require(dict(deltas) == expected_deltas,
                f"dense EX2 delta census drift: {dict(deltas)}")

        result: dict[str, Any] = {
            "schema": "neuralstudio.fg.mid-l54-semantics-verification.v1",
            "rtx_job_id": rig.get("job_id"),
            "rtx_rig_receipt_sha256": sha256_file(
                args.rtx_root / "rig-receipt.json"),
            "fresh_rtx_process_count": 2,
            "point_sampler": {
                "records": len(points),
                "output_sha256": POINT_SHA256,
                "normalized_coordinate_fraction_bits": 12,
                "fixed12_rule_mismatches": point_mismatches,
                "rule": "floor(trunc(coord*4096)*extent/4096)",
            },
            "dense_ex2": {
                "records": len(dense_rtx),
                "range": [-2.0, 0.0],
                "step": "2^-21",
                "rtx_sha256": DENSE_RTX_SHA256,
                "mac_sha256": DENSE_MAC_SHA256,
                "rtx_minus_mac_bit_delta_counts": {
                    str(key): value for key, value in sorted(deltas.items())
                },
                "used_as_runtime_lookup": False,
            },
            "pair_semantics_sha256": PAIR_SHA256,
            "expected_frame_bytes_read": False,
            "archived_intermediates_read": False,
            "address_fields_recorded": False,
            "claims": {
                "sampler_rule_exact_for_probed_contract_domain": True,
                "general_arbitrary_f32_ex2_recovered": False,
                "complete_frame": False,
            },
            "result": "pass",
        }
        result["receipt_root_sha256"] = canonical_hash(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
        print(f"receipt_root_sha256={result['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
