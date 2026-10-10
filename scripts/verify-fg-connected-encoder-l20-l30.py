#!/usr/bin/env python3
"""Verify connected FG encoder launch prefix L20-L30."""

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


RUN_SCHEMA = "neuralstudio.fg.connected-encoder-l20-l30-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.connected-encoder-l20-l30-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.connected-encoder-l20-l30-verification.v1"
MANIFEST_SHA = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
INPUT_SHA = "8d3380b000a1770fb9143345d3716b7c28abe38e083296b566607b4e04d269bc"
INPUT_BYTES = 160 * 96 * 48 * 2
LAUNCHES = list(range(20, 31))
POOL_MODULE_SHA = "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977"
WEIGHTS = [
    (20, 28, "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d", 63, 56320, "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"),
    (24, 31, "e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63", 67, 299008, "4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55"),
    (26, 32, "c74961f6f71f4e7bac5b216cc704b10a0fdd6f0987352fdd315f371467ca7187", 69, 593920, "c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404"),
    (28, 33, "6f2ffbe7faacde8119719d92dacde40f21ddd562878381b81654dad7a66775dd", 71, 1187840, "fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93"),
    (22, 30, "31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785", 65, 10240, "eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60"),
    (30, 34, "2f490857c6f7aa22f2435057706ec10b28b6b8d3081bd967ebe1d61550ac6419", 73, 540672, "e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9"),
]
TENSORS = [
    ("l20", 20, 983040),
    ("l21", 21, 245760),
    ("l22", 22, 491520),
    ("l23", 23, 122880),
    ("l24", 24, 245760),
    ("l25", 25, 61440),
    ("l26", 26, 61440),
    ("l27", 27, 15360),
    ("l28", 28, 30720),
    ("l29", 29, 7680),
    ("l30", 30, 15360),
]


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        0x2000,
        0xA000,
        0x2400,
        0xA400,
        0x2800,
        0xA800,
        0x2C00,
        0xAC00,
        0x3000,
        0xB000,
        0x3400,
    )
    seed = 0x243F6A8885A308D3
    step = 0x9E3779B97F4A7C15
    result = bytearray(INPUT_BYTES)
    for index in range(INPUT_BYTES // 2):
        mixed = splitmix64((seed + index * step) & ((1 << 64) - 1))
        struct.pack_into("<H", result, index * 2, values[mixed % len(values)])
    return bytes(result)


def collisions(data: bytes, poison: int) -> set[int]:
    require(len(data) % 2 == 0, "odd tensor byte count")
    return {
        offset // 2
        for offset in range(0, len(data), 2)
        if data[offset] == poison and data[offset + 1] == poison
    }


def mismatch_count(left: bytes, right: bytes) -> int:
    require(len(left) == len(right), "comparison size mismatch")
    return sum(a != b for a, b in zip(left, right, strict=True))


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    backend_name = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    generated = generate_input()
    require(sha256_bytes(generated) == INPUT_SHA, "input generator drift")
    receipts: list[dict[str, Any]] = []
    all_tensors: dict[str, list[bytes]] = {tensor[0]: [] for tensor in TENSORS}
    poison_first: dict[int, dict[str, bytes]] = {}
    for poison in (0xA5, 0x5A):
        per_poison: dict[str, list[bytes]] = {tensor[0]: [] for tensor in TENSORS}
        for ordinal in ("a", "b"):
            run_id = f"encoder-poison-{poison:02x}-process-{ordinal}"
            run_root = root / run_id
            receipt = load_json(run_root / "run-receipt.json")
            require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
            require(receipt.get("backend") == backend_name, f"{run_id}: backend")
            require(receipt.get("run_id") == run_id, f"{run_id}: run id")
            require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
            require(isinstance(receipt.get("process_id"), int), f"{run_id}: process")
            require(receipt.get("selected_launches") == LAUNCHES, f"{run_id}: launch sequence")
            require(receipt.get("pool_module") == {"token": 29, "sha256": POOL_MODULE_SHA}, f"{run_id}: pool module")
            require(receipt.get("model_capture_manifest_sha256") == MANIFEST_SHA, f"{run_id}: manifest")
            observed_weights = receipt.get("weights")
            require(isinstance(observed_weights, list) and len(observed_weights) == len(WEIGHTS), f"{run_id}: weights")
            for observed, expected in zip(observed_weights, WEIGHTS, strict=True):
                launch, module, module_sha, token, size, weight_sha = expected
                require(observed.get("launch") == launch, f"{run_id}: weight launch")
                require(observed.get("module") == module, f"{run_id}: weight module")
                require(observed.get("module_sha256") == module_sha, f"{run_id}: module hash")
                require(observed.get("source_token") == token, f"{run_id}: weight token")
                require(observed.get("bytes") == size, f"{run_id}: weight bytes")
                require(observed.get("pre_sha256") == weight_sha, f"{run_id}: weight prehash")
                require(observed.get("post_sha256") == weight_sha, f"{run_id}: weight mutation")
                weight_after = (run_root / f"weight-token-{token}-after.bin").read_bytes()
                require(len(weight_after) == size and sha256_bytes(weight_after) == weight_sha, f"{run_id}: weight artifact")
            require(receipt.get("input_generator") == "splitmix64-fg-connected-encoder-l20-l30-v1", f"{run_id}: generator")
            require(receipt.get("input_bytes") == INPUT_BYTES, f"{run_id}: input bytes")
            require(receipt.get("input_pre_sha256") == INPUT_SHA, f"{run_id}: input prehash")
            require(receipt.get("input_post_sha256") == INPUT_SHA, f"{run_id}: input mutation")
            require((run_root / "input-after.bin").read_bytes() == generated, f"{run_id}: input artifact")
            observed_tensors = receipt.get("tensors")
            require(isinstance(observed_tensors, list) and len(observed_tensors) == len(TENSORS), f"{run_id}: tensors")
            for observed, (tensor_id, launch, size) in zip(observed_tensors, TENSORS, strict=True):
                data = (run_root / f"{tensor_id}.bin").read_bytes()
                require(observed.get("id") == tensor_id, f"{run_id}: tensor id")
                require(observed.get("launch") == launch, f"{run_id}: tensor launch")
                require(observed.get("bytes") == size and len(data) == size, f"{run_id}: tensor bytes")
                require(observed.get("sha256") == sha256_bytes(data), f"{run_id}: tensor hash")
                require(observed.get("poison_collision_halfwords") == len(collisions(data, poison)), f"{run_id}: collision census")
                per_poison[tensor_id].append(data)
                all_tensors[tensor_id].append(data)
            require(receipt.get("immutable_input_mismatched_bytes") == 0, f"{run_id}: immutable input")
            require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: canary")
            require(receipt.get("per_run_write_coverage_conclusive") is False, f"{run_id}: per-run coverage claim")
            require(receipt.get("external_model_weights_opened") is True, f"{run_id}: model ingress")
            require(receipt.get("expected_output_opened") is False, f"{run_id}: expected output ingress")
            require(receipt.get("archived_intermediate_opened") is False, f"{run_id}: archived intermediate ingress")
            require(receipt.get("address_fields_recorded") is False, f"{run_id}: address fields")
            require(receipt.get("result") == "pass", f"{run_id}: failed")
            require_address_free(receipt, run_id)
            receipts.append(receipt)
        for tensor_id, values in per_poison.items():
            require(values[0] == values[1], f"{tensor_id}: process repeat drift")
        poison_first[poison] = {key: values[0] for key, values in per_poison.items()}
    require(len({receipt["process_id"] for receipt in receipts}) == 4, "fresh-process gate")
    tensor_summary = []
    for tensor_id, launch, size in TENSORS:
        values = all_tensors[tensor_id]
        require(len({sha256_bytes(value) for value in values}) == 1, f"{tensor_id}: poison drift")
        untouched = collisions(poison_first[0xA5][tensor_id], 0xA5) & collisions(poison_first[0x5A][tensor_id], 0x5A)
        require(not untouched, f"{tensor_id}: write coverage failed")
        tensor_summary.append(
            {
                "id": tensor_id,
                "launch": launch,
                "bytes": size,
                "sha256": sha256_bytes(values[0]),
                "write_coverage": True,
            }
        )
    mutated = bytearray(all_tensors["l30"][0])
    mutated[31] ^= 1
    require(sha256_bytes(mutated) != tensor_summary[-1]["sha256"], "output mutation undetected")
    receipt: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": backend_name,
        "selected_group": "evaluate-1/connected-encoder/L20-L30",
        "selected_launches": LAUNCHES,
        "model_capture_manifest_sha256": MANIFEST_SHA,
        "weight_tokens": [item[3] for item in WEIGHTS],
        "weight_sha256": [item[5] for item in WEIGHTS],
        "input_generator": "splitmix64-fg-connected-encoder-l20-l30-v1",
        "input_sha256": INPUT_SHA,
        "tensors": tensor_summary,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "all_tensor_write_coverage": True,
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


def verify_backend(receipt: dict[str, Any], backend: str) -> None:
    backend_name = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("schema") == BACKEND_SCHEMA, f"{backend}: schema")
    require(receipt.get("backend") == backend_name, f"{backend}: identity")
    declared = receipt.get("receipt_root_sha256")
    payload = dict(receipt)
    payload.pop("receipt_root_sha256", None)
    require(declared == canonical_sha256(payload), f"{backend}: root")
    require(receipt.get("result") == "pass", f"{backend}: failed")
    require_address_free(receipt, f"{backend} receipt")


def validate_final(rtx_path: Path, mac_path: Path, rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx = load_json(rtx_path)
    mac = load_json(mac_path)
    verify_backend(rtx, "rtx")
    verify_backend(mac, "mac")
    require(rtx["weight_tokens"] == mac["weight_tokens"], "weight token drift")
    require(rtx["weight_sha256"] == mac["weight_sha256"], "weight hash drift")
    rtx_run = rtx_root / "encoder-poison-a5-process-a"
    mac_run = mac_root / "encoder-poison-a5-process-a"
    tensors = []
    for tensor_id, launch, size in TENSORS:
        rtx_data = (rtx_run / f"{tensor_id}.bin").read_bytes()
        mac_data = (mac_run / f"{tensor_id}.bin").read_bytes()
        mismatches = mismatch_count(rtx_data, mac_data)
        require(mismatches == 0, f"{tensor_id}: RTX/Metal mismatch")
        rtx_entry = next(item for item in rtx["tensors"] if item["id"] == tensor_id)
        mac_entry = next(item for item in mac["tensors"] if item["id"] == tensor_id)
        require(rtx_entry["sha256"] == mac_entry["sha256"] == sha256_bytes(rtx_data), f"{tensor_id}: backend hash drift")
        tensors.append(
            {
                "id": tensor_id,
                "launch": launch,
                "bytes": size,
                "sha256": sha256_bytes(rtx_data),
                "rtx_mac_mismatched_bytes": mismatches,
            }
        )
    receipt: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_group": "evaluate-1/connected-encoder/L20-L30",
        "selected_launches": LAUNCHES,
        "weight_tokens": rtx["weight_tokens"],
        "input_sha256": INPUT_SHA,
        "tensors": tensors,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "all_tensors_rtx_mac_exact": True,
        "gates": {
            "fresh_process_repeat": True,
            "two_poison_invariance": True,
            "all_tensor_write_coverage": True,
            "model_state_provenance": True,
            "input_weight_immutability": True,
            "canaries_intact": True,
            "no_expected_output_ingress": True,
            "no_archived_intermediate_ingress": True,
            "address_free": True,
        },
        "claims": {
            "connected_encoder_l20_l30_parity": True,
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
