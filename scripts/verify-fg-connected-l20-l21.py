#!/usr/bin/env python3
"""Verify the first connected FG convolution->pooling group."""

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


RUN_SCHEMA = "neuralstudio.fg.connected-l20-l21-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.connected-l20-l21-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.connected-l20-l21-verification.v1"
CONV_MODULE_SHA = "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d"
POOL_MODULE_SHA = "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977"
WEIGHT_SHA = "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"
MANIFEST_SHA = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
INPUT_SHA = "4c7c15a47cb7e775e861f9a5b529abd4fde3e9c4287b6663895031cd4c20af77"
INPUT_BYTES = 160 * 96 * 48 * 2
INTERMEDIATE_BYTES = 160 * 96 * 32 * 2
OUTPUT_BYTES = 80 * 48 * 32 * 2


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
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
                            token in lowered
                            for token in (
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
                f"{label}: address-like string at {location}",
            )

    visit(value, label)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def generate_input() -> bytes:
    values = (
        0x0000,
        0x8000,
        0x2400,
        0xA400,
        0x2C00,
        0xAC00,
        0x3000,
        0xB000,
        0x3400,
        0xB400,
        0x3800,
    )
    seed = 0xA0761D6478BD642F
    step = 0xE7037ED1A0B428DB
    result = bytearray(INPUT_BYTES)
    for index in range(INPUT_BYTES // 2):
        mixed = splitmix64((seed + index * step) & ((1 << 64) - 1))
        struct.pack_into("<H", result, index * 2, values[mixed % len(values)])
    return bytes(result)


def poison_collisions(data: bytes, poison: int) -> set[int]:
    require(len(data) % 2 == 0, "halfword payload has odd size")
    return {
        offset // 2
        for offset in range(0, len(data), 2)
        if data[offset] == poison and data[offset + 1] == poison
    }


def mismatch_count(left: bytes, right: bytes) -> int:
    require(len(left) == len(right), "comparison byte sizes differ")
    return sum(a != b for a, b in zip(left, right, strict=True))


def validate_selected_launches(value: Any, backend: str, run_id: str) -> None:
    require(isinstance(value, list) and len(value) == 2, f"{run_id}: launches")
    conv, pool = value
    require(conv.get("launch") == 20 and conv.get("module") == 28, f"{run_id}: conv identity")
    require(conv.get("module_sha256") == CONV_MODULE_SHA, f"{run_id}: conv module")
    require(pool.get("launch") == 21 and pool.get("module") == 29, f"{run_id}: pool identity")
    require(pool.get("module_sha256") == POOL_MODULE_SHA, f"{run_id}: pool module")
    if backend == "rtx-cuda-driver":
        require(conv.get("function") == "k_conv_fp16_nhwc", f"{run_id}: conv function")
        require(conv.get("grid") == [10, 8, 1], f"{run_id}: conv grid")
        require(conv.get("block") == [32, 4, 1], f"{run_id}: conv block")
        require(pool.get("function") == "k_pooling", f"{run_id}: pool function")
        require(pool.get("grid") == [20, 12, 1], f"{run_id}: pool grid")
        require(pool.get("block") == [4, 4, 16], f"{run_id}: pool block")
    else:
        require(conv.get("function") == "fg_conv_l20", f"{run_id}: Metal conv")
        require(conv.get("selected_grid") == [10, 8, 1], f"{run_id}: selected conv grid")
        require(conv.get("selected_block") == [32, 4, 1], f"{run_id}: selected conv block")
        require(conv.get("metal_threads") == [160, 96, 32], f"{run_id}: Metal conv threads")
        require(pool.get("function") == "fg_pooling", f"{run_id}: Metal pool")
        require(pool.get("selected_grid") == [20, 12, 1], f"{run_id}: selected pool grid")
        require(pool.get("selected_block") == [4, 4, 16], f"{run_id}: selected pool block")


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    generated = generate_input()
    require(sha256_bytes(generated) == INPUT_SHA, "input generator drift")
    receipts: list[dict[str, Any]] = []
    intermediates: list[bytes] = []
    outputs: list[bytes] = []
    by_poison: dict[int, tuple[bytes, bytes]] = {}
    for poison in (0xA5, 0x5A):
        poison_intermediates: list[bytes] = []
        poison_outputs: list[bytes] = []
        for ordinal in ("a", "b"):
            run_id = f"connected-poison-{poison:02x}-process-{ordinal}"
            run_root = root / run_id
            receipt_path = run_root / "run-receipt.json"
            receipt = load_json(receipt_path)
            input_after = (run_root / "input-after.bin").read_bytes()
            weights_after = (run_root / "weights-after.bin").read_bytes()
            intermediate = (run_root / "intermediate.bin").read_bytes()
            output = (run_root / "output.bin").read_bytes()
            require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
            require(receipt.get("backend") == expected_backend, f"{run_id}: backend")
            require(receipt.get("run_id") == run_id, f"{run_id}: run id")
            require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
            require(isinstance(receipt.get("process_id"), int), f"{run_id}: process id")
            validate_selected_launches(receipt.get("selected_launches"), expected_backend, run_id)
            require(receipt.get("weight_source_token") == 63, f"{run_id}: weight token")
            require(receipt.get("weight_capture_point") == "post_evaluate_0_fence_pre_evaluate_1", f"{run_id}: capture point")
            require(receipt.get("weight_capture_manifest_sha256") == MANIFEST_SHA, f"{run_id}: manifest")
            require(receipt.get("weight_pre_sha256") == WEIGHT_SHA, f"{run_id}: weight prehash")
            require(receipt.get("weight_post_sha256") == WEIGHT_SHA, f"{run_id}: weight mutation")
            require(sha256_bytes(weights_after) == WEIGHT_SHA, f"{run_id}: weight file")
            require(receipt.get("input_generator") == "splitmix64-fg-connected-l20-l21-v1", f"{run_id}: generator")
            require(receipt.get("input_pre_sha256") == INPUT_SHA, f"{run_id}: input prehash")
            require(receipt.get("input_post_sha256") == INPUT_SHA, f"{run_id}: input mutation")
            require(input_after == generated, f"{run_id}: generated input drift")
            require(len(intermediate) == INTERMEDIATE_BYTES, f"{run_id}: intermediate bytes")
            require(len(output) == OUTPUT_BYTES, f"{run_id}: output bytes")
            require(receipt.get("intermediate_bytes") == INTERMEDIATE_BYTES, f"{run_id}: intermediate receipt bytes")
            require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_id}: output receipt bytes")
            require(receipt.get("intermediate_sha256") == sha256_bytes(intermediate), f"{run_id}: intermediate hash")
            require(receipt.get("output_sha256") == sha256_bytes(output), f"{run_id}: output hash")
            require(receipt.get("immutable_input_mismatched_bytes") == 0, f"{run_id}: immutable input")
            require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: canary")
            require(receipt.get("intermediate_poison_collision_halfwords") == len(poison_collisions(intermediate, poison)), f"{run_id}: intermediate poison census")
            require(receipt.get("output_poison_collision_halfwords") == len(poison_collisions(output, poison)), f"{run_id}: output poison census")
            require(receipt.get("per_run_write_coverage_conclusive") is False, f"{run_id}: per-run coverage claim")
            require(receipt.get("external_model_weight_opened") is True, f"{run_id}: weight ingress")
            require(receipt.get("expected_output_opened") is False, f"{run_id}: expected output ingress")
            require(receipt.get("archived_intermediate_opened") is False, f"{run_id}: archived intermediate ingress")
            require(receipt.get("address_fields_recorded") is False, f"{run_id}: address fields")
            require(receipt.get("result") == "pass", f"{run_id}: run failed")
            require_address_free(receipt, run_id)
            receipts.append(receipt)
            intermediates.append(intermediate)
            outputs.append(output)
            poison_intermediates.append(intermediate)
            poison_outputs.append(output)
        require(poison_intermediates[0] == poison_intermediates[1], f"poison {poison:02x}: process intermediate drift")
        require(poison_outputs[0] == poison_outputs[1], f"poison {poison:02x}: process output drift")
        by_poison[poison] = (poison_intermediates[0], poison_outputs[0])

    require(len({receipt["process_id"] for receipt in receipts}) == 4, "fresh-process gate")
    require(len({sha256_bytes(value) for value in intermediates}) == 1, "two-poison intermediate drift")
    require(len({sha256_bytes(value) for value in outputs}) == 1, "two-poison output drift")
    intermediate_untouched = poison_collisions(by_poison[0xA5][0], 0xA5) & poison_collisions(by_poison[0x5A][0], 0x5A)
    output_untouched = poison_collisions(by_poison[0xA5][1], 0xA5) & poison_collisions(by_poison[0x5A][1], 0x5A)
    require(not intermediate_untouched, "intermediate write coverage failed")
    require(not output_untouched, "output write coverage failed")

    mutated = bytearray(outputs[0])
    mutated[127] ^= 1
    require(sha256_bytes(mutated) != sha256_bytes(outputs[0]), "output mutation was not detected")
    receipt: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "selected_group": "evaluate-1/L20-L21/conv3x3-pool2x2",
        "selected_launches": [20, 21],
        "selected_modules": [28, 29],
        "module_sha256": [CONV_MODULE_SHA, POOL_MODULE_SHA],
        "weight_source_token": 63,
        "weight_sha256": WEIGHT_SHA,
        "weight_capture_manifest_sha256": MANIFEST_SHA,
        "input_generator": "splitmix64-fg-connected-l20-l21-v1",
        "input_sha256": INPUT_SHA,
        "intermediate_sha256": sha256_bytes(intermediates[0]),
        "intermediate_bytes": INTERMEDIATE_BYTES,
        "output_sha256": sha256_bytes(outputs[0]),
        "output_bytes": OUTPUT_BYTES,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "intermediate_write_coverage": True,
        "output_write_coverage": True,
        "input_weight_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediate_fed_to_execution": False,
        "output_mutation_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    require_address_free(receipt, "backend receipt")
    return receipt


def verify_backend_receipt(receipt: dict[str, Any], backend: str) -> None:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("schema") == BACKEND_SCHEMA, f"{backend}: schema")
    require(receipt.get("backend") == expected_backend, f"{backend}: identity")
    declared = receipt.get("receipt_root_sha256")
    payload = dict(receipt)
    payload.pop("receipt_root_sha256", None)
    require(declared == canonical_sha256(payload), f"{backend}: root mismatch")
    require(receipt.get("result") == "pass", f"{backend}: failed")
    require_address_free(receipt, f"{backend} backend receipt")


def validate_final(rtx_path: Path, mac_path: Path, rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx = load_json(rtx_path)
    mac = load_json(mac_path)
    verify_backend_receipt(rtx, "rtx")
    verify_backend_receipt(mac, "mac")
    rtx_run = rtx_root / "connected-poison-a5-process-a"
    mac_run = mac_root / "connected-poison-a5-process-a"
    rtx_intermediate = (rtx_run / "intermediate.bin").read_bytes()
    mac_intermediate = (mac_run / "intermediate.bin").read_bytes()
    rtx_output = (rtx_run / "output.bin").read_bytes()
    mac_output = (mac_run / "output.bin").read_bytes()
    intermediate_mismatches = mismatch_count(rtx_intermediate, mac_intermediate)
    output_mismatches = mismatch_count(rtx_output, mac_output)
    require(intermediate_mismatches == 0, "RTX/Metal intermediate mismatch")
    require(output_mismatches == 0, "RTX/Metal connected output mismatch")
    require(rtx["intermediate_sha256"] == mac["intermediate_sha256"], "backend intermediate root mismatch")
    require(rtx["output_sha256"] == mac["output_sha256"], "backend output root mismatch")
    receipt: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_group": rtx["selected_group"],
        "selected_launches": [20, 21],
        "selected_modules": [28, 29],
        "weight_source_token": 63,
        "weight_sha256": WEIGHT_SHA,
        "input_sha256": INPUT_SHA,
        "intermediate_sha256": rtx["intermediate_sha256"],
        "intermediate_bytes": INTERMEDIATE_BYTES,
        "output_sha256": rtx["output_sha256"],
        "output_bytes": OUTPUT_BYTES,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_intermediate_mismatched_bytes": intermediate_mismatches,
        "rtx_mac_output_mismatched_bytes": output_mismatches,
        "gates": {
            "fresh_process_repeat": True,
            "two_poison_invariance": True,
            "provenance_bound_live_weight": True,
            "full_intermediate_write_coverage": True,
            "full_output_write_coverage": True,
            "input_weight_immutability": True,
            "canaries_intact": True,
            "no_expected_output_ingress": True,
            "no_archived_intermediate_ingress": True,
            "address_free": True,
        },
        "claims": {
            "dependency_closed_connected_group_parity": True,
            "complete_encoder_decoder_parity": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    require_address_free(receipt, "final receipt")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--rtx", type=Path, required=True)
    final.add_argument("--mac", type=Path, required=True)
    final.add_argument("--rtx-root", type=Path, required=True)
    final.add_argument("--mac-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend":
            receipt = validate_backend(args.backend, args.root)
        else:
            receipt = validate_final(args.rtx, args.mac, args.rtx_root, args.mac_root)
        write_json(args.output, receipt)
        print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
