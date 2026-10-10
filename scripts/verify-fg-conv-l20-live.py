#!/usr/bin/env python3
"""Verify dense live-weight RTX/Metal parity for selected FG launch 20."""

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


MODULE_SHA256 = "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d"
WEIGHT_SHA256 = "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"
MANIFEST_SHA256 = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
INPUT_SHA256 = "c3ace7c82216c975b11fb39f997418ec6380d907cd34c99b55df6b62729831f4"
OUTPUT_BYTES = 160 * 96 * 32 * 2
RUN_SCHEMA = "neuralstudio.fg.conv-l20-live-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.conv-l20-live-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.conv-l20-live-verification.v1"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = key.lower()
                if key != "address_fields_recorded":
                    require(
                        not any(
                            part in lowered
                            for part in (
                                "pointer",
                                "virtual_address",
                                "device_address",
                                "gpu_address",
                            )
                        ),
                        f"{label}: address-bearing key at {location}.{key}",
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")
        elif isinstance(node, str):
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                f"{label}: raw address-like text at {location}",
            )

    visit(value, label)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def generated_input() -> bytes:
    values = (0x0000, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800)
    seed = 0xD1B54A32D192ED03
    step = 0x9E3779B97F4A7C15
    result = bytearray(160 * 96 * 48 * 2)
    for index in range(len(result) // 2):
        struct.pack_into(
            "<H",
            result,
            index * 2,
            values[splitmix64((seed + index * step) & ((1 << 64) - 1)) % 7],
        )
    return bytes(result)


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    generated = generated_input()
    require(sha256_bytes(generated) == INPUT_SHA256, "input generator drift")
    receipts: list[dict[str, Any]] = []
    outputs: list[bytes] = []
    for poison in (0xA5, 0x5A):
        for ordinal in ("a", "b"):
            run_id = f"live-model-poison-{poison:02x}-process-{ordinal}"
            run_root = root / run_id
            receipt = load_json(run_root / "run-receipt.json")
            output = (run_root / "output.bin").read_bytes()
            require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
            require(receipt.get("backend") == expected_backend, f"{run_id}: backend")
            require(receipt.get("run_id") == run_id, f"{run_id}: run id")
            require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
            require(isinstance(receipt.get("process_id"), int), f"{run_id}: pid")
            if backend == "rtx":
                require(receipt.get("grid") == [10, 8, 1], f"{run_id}: grid")
                require(receipt.get("block") == [32, 4, 1], f"{run_id}: block")
            else:
                require(receipt.get("selected_rtx_grid") == [10, 8, 1], f"{run_id}: selected grid")
                require(receipt.get("selected_rtx_block") == [32, 4, 1], f"{run_id}: selected block")
                require(receipt.get("metal_dispatch_threads") == [160, 96, 32], f"{run_id}: dispatch")
                require(receipt.get("metal_threads_per_threadgroup") == [8, 4, 1], f"{run_id}: threadgroup")
            require(receipt.get("abi_scalars") == [160, 96, 162, 98, 160, 96], f"{run_id}: ABI")
            require(receipt.get("weight_source_token") == 63, f"{run_id}: token")
            require(receipt.get("weight_capture_point") == "post_evaluate_0_fence_pre_evaluate_1", f"{run_id}: capture point")
            require(receipt.get("weight_capture_manifest_sha256") == MANIFEST_SHA256, f"{run_id}: manifest")
            require(receipt.get("weight_pre_sha256") == WEIGHT_SHA256, f"{run_id}: weight hash")
            require(receipt.get("weight_post_sha256") == WEIGHT_SHA256, f"{run_id}: weight mutation")
            require(receipt.get("input_generator") == "splitmix64-fg-conv-l20-live-v1", f"{run_id}: generator")
            require(receipt.get("input_pre_sha256") == INPUT_SHA256, f"{run_id}: input hash")
            require(receipt.get("input_post_sha256") == INPUT_SHA256, f"{run_id}: input mutation")
            require(len(output) == OUTPUT_BYTES, f"{run_id}: output bytes")
            require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_id}: receipt bytes")
            require(receipt.get("output_sha256") == sha256_bytes(output), f"{run_id}: output hash")
            require(receipt.get("input_mismatched_bytes") == 0, f"{run_id}: input mismatch")
            require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: guard mismatch")
            require(receipt.get("per_run_write_coverage_conclusive") is False, f"{run_id}: false coverage claim")
            require(receipt.get("external_weight_file_opened") is True, f"{run_id}: missing model input")
            require(receipt.get("expected_output_file_opened") is False, f"{run_id}: oracle output consumed")
            require(receipt.get("address_fields_recorded") is False, f"{run_id}: addresses")
            require(receipt.get("result") == "pass", f"{run_id}: failed")
            if backend == "rtx":
                require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_id}: module")
                require(receipt.get("module_token") == 28, f"{run_id}: module token")
                require(receipt.get("launch_order") == 20, f"{run_id}: launch")
            else:
                require(receipt.get("metal_function") == "fg_conv_l20", f"{run_id}: function")
            require_address_free(receipt, run_id)
            receipts.append(receipt)
            outputs.append(output)
    require(len({receipt["process_id"] for receipt in receipts}) == 4, "fresh-process gate")
    require(len({sha256_bytes(output) for output in outputs}) == 1, "two-process/two-poison invariance")
    mutated = bytearray(outputs[0])
    mutated[101] ^= 1
    require(sha256_bytes(mutated) != receipts[0]["output_sha256"], "mutation was not detected")
    receipt = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "selected_unit": "evaluate-1/L20/k_conv_fp16_nhwc/module-28",
        "module_sha256": MODULE_SHA256,
        "weight_source_token": 63,
        "weight_sha256": WEIGHT_SHA256,
        "weight_capture_manifest_sha256": MANIFEST_SHA256,
        "weight_capture_point": "post_evaluate_0_fence_pre_evaluate_1",
        "input_generator": "splitmix64-fg-conv-l20-live-v1",
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
    rtx = load_json(rtx_path)
    mac = load_json(mac_path)
    for receipt, backend in ((rtx, "rtx-cuda-driver"), (mac, "mac-metal")):
        require(receipt.get("schema") == BACKEND_SCHEMA, f"{backend}: schema")
        require(receipt.get("backend") == backend, f"{backend}: identity")
        require(receipt.get("result") == "pass", f"{backend}: result")
        root = receipt.get("receipt_root_sha256")
        without_root = dict(receipt)
        without_root.pop("receipt_root_sha256", None)
        require(root == canonical_sha256(without_root), f"{backend}: root")
        require_address_free(receipt, backend)
    require(rtx["output_sha256"] == mac["output_sha256"], "dense output mismatch")
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
        "scope": "one complete live-weight launch-20 convolution on an independently generated dense input; no multi-kernel group or full-frame claim",
        "address_fields_recorded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--rtx", type=Path, required=True)
    final.add_argument("--mac", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = (
            validate_backend(args.backend, args.root)
            if args.command == "backend"
            else validate_final(args.rtx, args.mac)
        )
        write_json(args.output, receipt)
        print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
