#!/usr/bin/env python3
"""Extend generated connected L3-L21 hosts with live L22-L23 tails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(source: str, backend: str) -> str:
    include = f'#include "{backend}_tail.inc"'
    replacement = (include +
                   f'\n#include "fg_connected_dynamic_l3_l23/{backend}_tail.inc"')
    if source.count(include) != 1:
        raise RuntimeError(f"{backend}: base tail marker drift")
    pass_marker = "l20_l21_pass && checkpoints.size() == 23"
    if source.count(pass_marker) != 1:
        raise RuntimeError(f"{backend}: pass marker drift")
    if source.count("neuralstudio.fg.connected-dynamic-l3-l21-run.v1") != 1:
        raise RuntimeError(f"{backend}: schema marker drift")
    if source.count('selected_launch_count\\\": 23') != 1:
        raise RuntimeError(f"{backend}: launch count marker drift")
    result = source.replace(include, replacement)
    result = result.replace(
        pass_marker,
        "l20_l21_pass && l22_l23_pass && checkpoints.size() == 25")
    result = result.replace(
        "neuralstudio.fg.connected-dynamic-l3-l21-run.v1",
        "neuralstudio.fg.connected-dynamic-l3-l23-run.v1")
    result = result.replace('selected_launch_count\\\": 23',
                            'selected_launch_count\\\": 25')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base_receipt = json.loads((base / "generator-receipt.json").read_text())
    if (base_receipt.get("receipt_root_sha256") !=
            "3963039331e816c0ee4b5730fc13ee2ce3f2433cbef51e1a85bf9ae246fedb2d"):
        raise RuntimeError("base generator identity mismatch")
    generated = {}
    for backend, filename in (("mac", "generated_macos.mm"),
                              ("rtx", "generated_rtx.cpp")):
        source_path = base / filename
        raw = source_path.read_bytes()
        result = transform(raw.decode(), backend).encode()
        destination = output / filename
        destination.write_bytes(result)
        generated[backend] = {
            "base_generated_sha256": digest(raw),
            "generated_sha256": digest(result),
            "injected_tail": f"{backend}_l22_l23_tail.inc",
        }
    receipt = {
        "schema": "neuralstudio.fg.connected-dynamic-l3-l23-generator.v1",
        "base_generator_receipt_root_sha256":
            base_receipt["receipt_root_sha256"],
        "generated": generated,
        "schema_replacements": 2,
        "launch_count_replacements": 2,
        "pass_gate_replacements": 2,
        "live_tail_injections": 2,
        "expected_or_archived_intermediate_embedded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = digest(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    (output / "generator-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"generator_receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
