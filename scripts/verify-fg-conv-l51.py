#!/usr/bin/env python3
"""Verify live-weight RTX/Metal parity for selected FG convolution L51."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any


MODULE_SHA256 = "27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2"
WEIGHT_SHA256 = "477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e"
MANIFEST_SHA256 = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
OUTPUT_BYTES = 160 * 96 * 16 * 2
RUN_SCHEMA = "neuralstudio.fg.conv-l51-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.conv-l51-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.conv-l51-verification.v1"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key != "address_fields_recorded":
                    require(
                        not any(
                            part in key.lower()
                            for part in (
                                "pointer",
                                "virtual_address",
                                "device_address",
                                "gpu_address",
                            )
                        ),
                        f"{label}: address key {location}.{key}",
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")
        elif isinstance(node, str):
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                f"{label}: address text {location}",
            )

    visit(value, label)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def generated_input() -> bytes:
    values = (0x0000, 0x2C00, 0xAC00, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800)
    seed = 0x94D049BB133111EB
    step = 0xD1B54A32D192ED03
    result = bytearray(160 * 96 * 32 * 2)
    mask = (1 << 64) - 1
    for index in range(len(result) // 2):
        struct.pack_into(
            "<H",
            result,
            index * 2,
            values[splitmix64((seed + index * step) & mask) % len(values)],
        )
    return bytes(result)


INPUT_SHA256 = sha256_bytes(generated_input())


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    outputs: list[bytes] = []
    receipts: list[dict[str, Any]] = []
    for poison in (0xA5, 0x5A):
        for ordinal in ("a", "b"):
            run_id = f"live-model-poison-{poison:02x}-process-{ordinal}"
            run_root = root / run_id
            receipt = load_json(run_root / "run-receipt.json")
            output = (run_root / "output.bin").read_bytes()
            require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
            require(receipt.get("backend") == expected_backend, f"{run_id}: backend")
            require(receipt.get("run_id") == run_id, f"{run_id}: identity")
            require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
            require(receipt.get("module_token") == 52, f"{run_id}: module token")
            require(receipt.get("module_bytes") == 3912, f"{run_id}: module bytes")
            require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_id}: module hash")
            require(receipt.get("trace_launch_token") == "launch-000051", f"{run_id}: launch")
            require(receipt.get("packed_parameter_bytes") == 40, f"{run_id}: params")
            require(receipt.get("abi_scalars") == [160, 96, 160, 96], f"{run_id}: ABI")
            require(receipt.get("input_shape_nhwc") == [1, 96, 160, 32], f"{run_id}: input shape")
            require(receipt.get("output_shape_nhwc") == [1, 96, 160, 16], f"{run_id}: output shape")
            require(receipt.get("weight_source_token") == 95, f"{run_id}: weight token")
            require(receipt.get("weight_capture_point") == "post_evaluate_0_fence_pre_evaluate_1", f"{run_id}: capture")
            require(receipt.get("weight_capture_manifest_sha256") == MANIFEST_SHA256, f"{run_id}: manifest")
            require(receipt.get("weight_pre_sha256") == WEIGHT_SHA256, f"{run_id}: weights")
            require(receipt.get("weight_post_sha256") == WEIGHT_SHA256, f"{run_id}: weight mutation")
            require(receipt.get("input_generator") == "splitmix64-fg-conv-l51-live-v1", f"{run_id}: generator")
            require(receipt.get("input_pre_sha256") == INPUT_SHA256, f"{run_id}: input")
            require(receipt.get("input_post_sha256") == INPUT_SHA256, f"{run_id}: input mutation")
            require(len(output) == OUTPUT_BYTES, f"{run_id}: output size")
            require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_id}: receipt size")
            require(receipt.get("output_sha256") == sha256_bytes(output), f"{run_id}: output hash")
            require(receipt.get("input_mismatched_bytes") == 0, f"{run_id}: immutable")
            require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: canary")
            require(receipt.get("per_run_write_coverage_conclusive") is False, f"{run_id}: coverage overclaim")
            require(receipt.get("external_weight_file_opened") is True, f"{run_id}: weights not consumed")
            require(receipt.get("expected_output_file_opened") is False, f"{run_id}: expected output consumed")
            require(receipt.get("address_fields_recorded") is False, f"{run_id}: address flag")
            require(receipt.get("result") == "pass", f"{run_id}: failed")
            if backend == "rtx":
                require(receipt.get("grid") == [10, 24, 1], f"{run_id}: grid")
                require(receipt.get("block") == [32, 4, 1], f"{run_id}: block")
                require(receipt.get("function_name") == "k_conv_fp16_nhwc", f"{run_id}: function")
            else:
                require(receipt.get("selected_rtx_grid") == [10, 24, 1], f"{run_id}: selected grid")
                require(receipt.get("selected_rtx_block") == [32, 4, 1], f"{run_id}: selected block")
                require(receipt.get("metal_dispatch_threads") == [160, 96, 16], f"{run_id}: dispatch")
                require(receipt.get("metal_threads_per_threadgroup") == [8, 4, 1], f"{run_id}: threadgroup")
                require(receipt.get("metal_function") == "fg_conv_l51", f"{run_id}: Metal function")
            require_address_free(receipt, run_id)
            receipts.append(receipt)
            outputs.append(output)
    require(len({r["process_id"] for r in receipts}) == 4, "fresh process gate")
    require(len({sha256_bytes(o) for o in outputs}) == 1, "two poison/process invariance")
    mutated = bytearray(outputs[0]); mutated[51] ^= 1
    require(sha256_bytes(mutated) != receipts[0]["output_sha256"], "mutation undetected")
    receipt = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "selected_unit": "evaluate-1/launch-000051/k_conv_fp16_nhwc/module-52",
        "module_sha256": MODULE_SHA256,
        "weight_source_token": 95,
        "weight_sha256": WEIGHT_SHA256,
        "weight_capture_manifest_sha256": MANIFEST_SHA256,
        "input_generator": "splitmix64-fg-conv-l51-live-v1",
        "input_sha256": INPUT_SHA256,
        "output_sha256": sha256_bytes(outputs[0]),
        "output_bytes": OUTPUT_BYTES,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_weight_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "output_mutation_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_final(rtx_path: Path, mac_path: Path) -> dict[str, Any]:
    rtx = load_json(rtx_path); mac = load_json(mac_path)
    for receipt, backend in ((rtx, "rtx-cuda-driver"), (mac, "mac-metal")):
        require(receipt.get("schema") == BACKEND_SCHEMA, f"{backend}: schema")
        require(receipt.get("backend") == backend, f"{backend}: identity")
        without_root = dict(receipt); root = without_root.pop("receipt_root_sha256", None)
        require(root == canonical_sha256(without_root), f"{backend}: root")
        require_address_free(receipt, backend)
    require(rtx["output_sha256"] == mac["output_sha256"], "RTX/Mac mismatch")
    receipt = {
        "schema": FINAL_SCHEMA,
        "selected_unit": rtx["selected_unit"],
        "module_sha256": MODULE_SHA256,
        "weight_sha256": WEIGHT_SHA256,
        "input_sha256": INPUT_SHA256,
        "output_sha256": rtx["output_sha256"],
        "output_bytes": OUTPUT_BYTES,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_differing_bytes": 0,
        "rtx_mac_exact": True,
        "scope": "second complete selected convolution, live weights and independent dense input; no multi-kernel group claim",
        "address_fields_recorded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest="command", required=True)
    backend = subs.add_parser("backend"); backend.add_argument("--backend", choices=("rtx", "mac"), required=True); backend.add_argument("--root", type=Path, required=True); backend.add_argument("--output", type=Path, required=True)
    final = subs.add_parser("final"); final.add_argument("--rtx", type=Path, required=True); final.add_argument("--mac", type=Path, required=True); final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_backend(args.backend, args.root) if args.command == "backend" else validate_final(args.rtx, args.mac)
        write_json(args.output, receipt)
        print(f"receipt_root_sha256={receipt['receipt_root_sha256']}"); print("result=pass")
        return 0
    except (OSError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
