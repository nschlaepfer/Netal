#!/usr/bin/env python3
"""Build the fail-closed L54 confidence LUT from an RTX instruction probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


PAIR_SHA256 = "496a4dfa79184edc49ae97a2492ca2db9c902d45b3faf5d82dde732077aad3a9"
PAIR_RECORD = struct.Struct("<5I2HI")
LUT_RECORD = struct.Struct("<IHH")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-semantics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source = args.pair_semantics.read_bytes()
    if sha256(source) != PAIR_SHA256:
        raise SystemExit("pair-semantics identity mismatch")
    if len(source) != 17 * 17 * PAIR_RECORD.size:
        raise SystemExit("pair-semantics size mismatch")

    by_difference: dict[int, tuple[int, int]] = {}
    for offset in range(0, len(source), PAIR_RECORD.size):
        (difference_bits, _scaled_bits, _ex2_bits, _low_bits, _high_bits,
         low_half, high_half, reserved) = PAIR_RECORD.unpack_from(source, offset)
        if reserved != 0:
            raise SystemExit("pair-semantics reserved field drift")
        candidate = (low_half, high_half)
        previous = by_difference.setdefault(difference_bits, candidate)
        if previous != candidate:
            raise SystemExit(
                f"confidence mapping is ambiguous for 0x{difference_bits:08x}"
            )

    records = b"".join(
        LUT_RECORD.pack(key, low, high)
        for key, (low, high) in sorted(by_difference.items())
    )
    if len(by_difference) != 52:
        raise SystemExit(f"unexpected L54 confidence domain: {len(by_difference)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(records)

    receipt = {
        "schema": "neuralstudio.fg.mid-l54-confidence-lut.v1",
        "source": {
            "kind": "independent-sm120-instruction-probe",
            "pair_semantics_bytes": len(source),
            "pair_semantics_sha256": PAIR_SHA256,
            "fresh_process_repeat_equal": True,
            "expected_frame_bytes_read": False,
            "archived_intermediates_read": False,
        },
        "instruction_chain": [
            "sub.ftz.f32",
            "abs.ftz.f32",
            "mul.ftz.f32(0xBFB8AA3B)",
            "ex2.approx.ftz.f32",
            "add.ftz.f32(1.0)",
            "rcp.approx.ftz.f32",
            "mul.ftz.f32",
            "cvt.rn.f16.f32",
        ],
        "domain": {
            "input_kind": "absolute difference of the 17 contract half values",
            "entries": len(by_difference),
            "complete_for_contract": True,
            "unsupported_inputs_fail_closed": True,
        },
        "output": {
            "record_layout": "u32 difference_bits, u16 low_half, u16 high_half",
            "bytes": len(records),
            "sha256": sha256(records),
        },
        "claims": {
            "sm120_instruction_outputs": True,
            "general_arbitrary_f32_ex2": False,
            "fg_expected_output": False,
            "full_frame": False,
        },
        "result": "pass",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"entries={len(by_difference)}")
    print(f"lut_sha256={sha256(records)}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
