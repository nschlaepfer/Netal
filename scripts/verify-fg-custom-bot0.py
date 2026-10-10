#!/usr/bin/env python3
"""Verify launch 68 custom block0 bottom convolution on RTX and Metal."""

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

MODULE_SHA256 = "3acb96881f20a50e575e2b2959f0aa433e03e3da20fbf624ae6bb183d77a3e48"
WEIGHT_SHA256 = "6b59f16f1be43617ebf0c94ff29cf7c8b7a2f2c7b7bfb41df8f9ede585e70684"
BIAS_SHA256 = "c3e301df2e15c48e0c2fb35fc636b944653c6f91071a101d71b256b740224d29"
MANIFEST_SHA256 = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
OUTPUT_BYTES = 3 * 92 * 160 * 32 * 2
RUN_SCHEMA = "neuralstudio.fg.custom-bot0-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.custom-bot0-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.custom-bot0-verification.v1"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def address_free(value: Any, label: str) -> None:
    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key != "address_fields_recorded":
                    require(
                        not any(part in key.lower() for part in ("pointer", "virtual_address", "device_address", "gpu_address")),
                        f"{label}: address key {path}.{key}",
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
        elif isinstance(node, str):
            require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None, f"{label}: address text")

    visit(value, label)


def splitmix(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def generated_input() -> bytes:
    values = (0x0000, 0x2800, 0xA800, 0x2C00, 0xAC00, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800)
    result = bytearray(46 * 80 * 64 * 2)
    for index in range(len(result) // 2):
        bits = values[splitmix((0xC0AC29B7C97C50DD + index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)) % len(values)]
        struct.pack_into("<H", result, index * 2, bits)
    return bytes(result)


INPUT_SHA256 = sha(generated_input())


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    backend_name = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    receipts: list[dict[str, Any]] = []
    outputs: list[bytes] = []
    for poison in (0xA5, 0x5A):
        for ordinal in ("a", "b"):
            run_id = f"bot0-poison-{poison:02x}-process-{ordinal}"
            receipt = load(root / run_id / "run-receipt.json")
            output = (root / run_id / "output.bin").read_bytes()
            require(receipt.get("schema") == RUN_SCHEMA and receipt.get("backend") == backend_name, f"{run_id}: identity")
            require(receipt.get("run_id") == run_id and receipt.get("poison_byte") == poison, f"{run_id}: run identity")
            require(isinstance(receipt.get("process_id"), int), f"{run_id}: process id")
            require(receipt.get("trace_launch_token") == "launch-000068", f"{run_id}: launch")
            require(receipt.get("module_token") == 58 and receipt.get("module_bytes") == 4840, f"{run_id}: module size")
            require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_id}: module hash")
            require(receipt.get("weight_source_token") == 150 and receipt.get("bias_source_token") == 152, f"{run_id}: model tokens")
            require(receipt.get("model_manifest_sha256") == MANIFEST_SHA256, f"{run_id}: manifest")
            require(receipt.get("weight_pre_sha256") == WEIGHT_SHA256 == receipt.get("weight_post_sha256"), f"{run_id}: weights")
            require(receipt.get("bias_pre_sha256") == BIAS_SHA256 == receipt.get("bias_post_sha256"), f"{run_id}: bias")
            require(receipt.get("input_generator") == "splitmix64-fg-custom-bot0-v1", f"{run_id}: generator")
            require(receipt.get("input_pre_sha256") == INPUT_SHA256 == receipt.get("input_post_sha256"), f"{run_id}: input")
            require(len(output) == OUTPUT_BYTES and receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_id}: output size")
            require(receipt.get("output_sha256") == sha(output), f"{run_id}: output hash")
            require(receipt.get("input_mismatched_bytes") == 0 and receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: guards")
            require(receipt.get("external_model_files_opened") is True, f"{run_id}: model provenance")
            require(receipt.get("expected_output_file_opened") is False, f"{run_id}: expected output isolation")
            require(receipt.get("address_fields_recorded") is False and receipt.get("result") == "pass", f"{run_id}: result")
            if backend == "rtx":
                require(receipt.get("grid") == [20, 23, 3] and receipt.get("block") == [32, 2, 4], f"{run_id}: RTX launch")
                require(receipt.get("function_name") == "custom_block0_conv_bot0_hf_kernel", f"{run_id}: RTX function")
                require(receipt.get("packed_parameter_bytes") == 96, f"{run_id}: ABI")
            else:
                require(receipt.get("selected_rtx_grid") == [20, 23, 3] and receipt.get("selected_rtx_block") == [32, 2, 4], f"{run_id}: traced launch")
                require(receipt.get("metal_dispatch_threads") == [160, 92, 96], f"{run_id}: Metal dispatch")
                require(receipt.get("metal_function") == "fg_custom_block0_bot0", f"{run_id}: Metal function")
            address_free(receipt, run_id)
            receipts.append(receipt)
            outputs.append(output)
    require(len({item["process_id"] for item in receipts}) == 4, "fresh-process gate")
    require(len({sha(item) for item in outputs}) == 1, "two-poison output invariance")
    mutated = bytearray(outputs[0])
    mutated[57] ^= 1
    require(sha(mutated) != receipts[0]["output_sha256"], "mutation rejection")
    result = {
        "schema": BACKEND_SCHEMA,
        "backend": backend_name,
        "selected_unit": "evaluate-1/launch-000068/custom_block0_conv_bot0_hf_kernel/module-58",
        "module_sha256": MODULE_SHA256,
        "weight_sha256": WEIGHT_SHA256,
        "bias_sha256": BIAS_SHA256,
        "model_manifest_sha256": MANIFEST_SHA256,
        "input_sha256": INPUT_SHA256,
        "output_sha256": sha(outputs[0]),
        "output_bytes": OUTPUT_BYTES,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_model_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "output_mutation_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_hash(result)
    return result


def validate_final(rtx_path: Path, mac_path: Path) -> dict[str, Any]:
    rtx = load(rtx_path)
    mac = load(mac_path)
    for receipt, backend in ((rtx, "rtx-cuda-driver"), (mac, "mac-metal")):
        require(receipt.get("schema") == BACKEND_SCHEMA and receipt.get("backend") == backend, f"{backend}: identity")
        body = dict(receipt)
        root = body.pop("receipt_root_sha256", None)
        require(root == canonical_hash(body), f"{backend}: receipt root")
        address_free(receipt, backend)
    require(rtx["output_sha256"] == mac["output_sha256"], "RTX/Metal parity")
    result = {
        "schema": FINAL_SCHEMA,
        "selected_unit": rtx["selected_unit"],
        "module_sha256": MODULE_SHA256,
        "weight_sha256": WEIGHT_SHA256,
        "bias_sha256": BIAS_SHA256,
        "input_sha256": INPUT_SHA256,
        "output_sha256": rtx["output_sha256"],
        "output_bytes": OUTPUT_BYTES,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_differing_bytes": 0,
        "rtx_mac_exact": True,
        "scope": "one custom fused upsample-convolution launch; no custom-family or connected-group claim",
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_hash(result)
    return result


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
        result = validate_backend(args.backend, args.root) if args.command == "backend" else validate_final(args.rtx, args.mac)
        write(args.output, result)
        print(f"receipt_root_sha256={result['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
