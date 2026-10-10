#!/usr/bin/env python3
"""Fail-closed verifier for canonical FG L9/module14 on RTX and Metal."""

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


RUN_SCHEMA = "neuralstudio.fg.early-l9-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.early-l9-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.early-l9-verification.v1"
RIG_SCHEMA = "neuralstudio.fg.early-l9-rig.v1"
MODULE_SHA = "540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
CONTRACT_SHA = "c3c8b79fb59f657f080a9e8a80d7b6aed2d756a784f775526cd893708f88bd0a"
RTX_SOURCE_SHA = "d054074394bbde914a4cf7da102f18233630961390ec26b69820aca3bdcbd5cd"
RTX_HARNESS_SHA = "90b2caaa3a687e191994ad5dfb305dd638ee1efbf540359d2343424d28d50c41"
METAL_SOURCE_SHA = "0bc45ef672f5cb69d945c8f5002da2bab52be13ef98de27c4cfd8d45ed890b8e"
MAC_HOST_SHA = "99434e41232e225d6ae5d8eb033f1baa9170bd52388e6cc6c6e98e9620cabf89"
MAC_HARNESS_SHA = "f3979716dba769945d951f7f1da1ace2a60523655b9fd7fe898597ab13dbebe9"
INPUT_SHA = "64786511c30d43cd9299bc6f8bffb265e19105e8bc68748b928e59a219d057b6"
DEAD_SHA = {
    0xA5: "132e8894e5af4f981aa9d77b671647bf1eeea6abcde444037ae5b9f9af1735be",
    0x5A: "bba37529ba3e0d6f439ac24a54e163db1817712ff441e5c1878515d856c3d6b8",
}
OUTPUT_SHA = "ae8629392d1cc725a17522cd2a71197c3daddab02d36931f7b4da333f0ed62b1"
WIDTH = 320
HEIGHT = 180
INPUT_BYTES = WIDTH * HEIGHT * 8
DEAD_BYTES = WIDTH * HEIGHT
OUTPUT_BYTES = WIDTH * HEIGHT * 4
MASK64 = (1 << 64) - 1
CASES = {
    "l9-poison-a5-process-a": 0xA5,
    "l9-poison-a5-process-b": 0xA5,
    "l9-poison-5a-process-a": 0x5A,
    "l9-poison-5a-process-b": 0x5A,
}
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True,
                          separators=(",", ":")).encode())


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    address_free(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def address_free(value: Any, label: str) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                if lowered != "address_fields_recorded":
                    require(not any(marker in lowered for marker in (
                        "pointer", "gpu_va", "virtual_address",
                        "device_address", "gpu_address")),
                        f"{label}: address field at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            require(ADDRESS_PATTERN.search(item) is None,
                    f"{label}: address-like value at {path}")
    visit(value, "$")


def mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def generated_input() -> bytes:
    values = (
        0x0000, 0x8000, 0x0001, 0x8001, 0x1400, 0x9400, 0x2000,
        0xA000, 0x2800, 0xA800, 0x3000, 0xB000, 0x3400, 0xB400,
        0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4200,
    )
    result = bytearray(INPUT_BYTES)
    for index in range(WIDTH * HEIGHT * 4):
        value = values[mix((0x243F6A8885A308D3 +
                            index * 0xD1B54A32D192ED03) & MASK64) % len(values)]
        struct.pack_into("<H", result, index * 2, value)
    return bytes(result)


def generated_dead(poison: int) -> bytes:
    return bytes(
        poison ^ ((mix((0x13198A2E03707344 + index) & MASK64) >> 56) & 0xFF)
        for index in range(DEAD_BYTES)
    )


INPUT = generated_input()
require(sha(INPUT) == INPUT_SHA, "embedded input generator drift")


def verify_sources(backend: str) -> dict[str, str]:
    repository = Path(__file__).resolve().parent.parent
    require(file_sha(repository / "src/fg_early_l9/contract.h") == CONTRACT_SHA,
            "contract source drift")
    if backend == "rtx":
        require(file_sha(repository / "src/fg_early_l9/rtx_oracle.cpp") == RTX_SOURCE_SHA,
                "RTX source drift")
        require(file_sha(repository / "scripts/build-fg-early-l9-rtx.ps1") == RTX_HARNESS_SHA,
                "RTX harness drift")
        return {"contract": CONTRACT_SHA, "source": RTX_SOURCE_SHA,
                "harness": RTX_HARNESS_SHA}
    require(file_sha(repository / "src/fg_early_l9/early_l9.metal") == METAL_SOURCE_SHA,
            "Metal source drift")
    require(file_sha(repository / "src/fg_early_l9/macos_probe.mm") == MAC_HOST_SHA,
            "Mac host source drift")
    require(file_sha(repository / "scripts/build-macos-fg-early-l9.sh") == MAC_HARNESS_SHA,
            "Mac harness drift")
    return {"contract": CONTRACT_SHA, "metal": METAL_SOURCE_SHA,
            "host": MAC_HOST_SHA, "harness": MAC_HARNESS_SHA}


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    identity = verify_sources(backend)
    backend_name = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    outputs = []
    receipts = []
    receipt_hashes = {}
    dead_hashes = set()
    for run_id, poison in CASES.items():
        run_root = root / run_id
        receipt_path = run_root / "run-receipt.json"
        receipt = load(receipt_path)
        input_after = (run_root / "input-after.bin").read_bytes()
        dead_after = (run_root / "dead-mask-after.bin").read_bytes()
        output = (run_root / "output.bin").read_bytes()
        dead = generated_dead(poison)
        require(sha(dead) == DEAD_SHA[poison], f"{run_id}: dead generator drift")
        require(receipt.get("schema") == RUN_SCHEMA and
                receipt.get("backend") == backend_name, f"{run_id}: identity drift")
        require(receipt.get("run_id") == run_id and
                receipt.get("poison_byte") == poison, f"{run_id}: run drift")
        require(isinstance(receipt.get("process_id"), int), f"{run_id}: process id")
        require(receipt.get("selected_launch") == 9 and
                receipt.get("module_token") == 14, f"{run_id}: launch drift")
        require(receipt.get("module_sha256") == MODULE_SHA,
                f"{run_id}: module hash drift")
        require(receipt.get("packed_parameter_bytes") == 32 and
                receipt.get("dead_parameter_offset") == 8,
                f"{run_id}: ABI drift")
        require(receipt.get("input_sha256") == [INPUT_SHA, DEAD_SHA[poison]],
                f"{run_id}: input receipt drift")
        require(receipt.get("input_post_sha256") == [INPUT_SHA, DEAD_SHA[poison]],
                f"{run_id}: input post receipt drift")
        require(input_after == INPUT and dead_after == dead,
                f"{run_id}: generated input drift")
        require(len(output) == OUTPUT_BYTES and
                receipt.get("output_bytes") == OUTPUT_BYTES,
                f"{run_id}: output size drift")
        require(receipt.get("output_sha256") == sha(output) == OUTPUT_SHA,
                f"{run_id}: output receipt hash drift")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("canary_mismatched_bytes") == 0,
                f"{run_id}: integrity gate failed")
        require(receipt.get("expected_output_opened") is False and
                receipt.get("archived_intermediate_opened") is False,
                f"{run_id}: forbidden ingress")
        require(receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass", f"{run_id}: result drift")
        if backend == "rtx":
            require(receipt.get("function_name") == "main_kernel" and
                    receipt.get("grid") == [20, 12, 1] and
                    receipt.get("block") == [16, 16, 1],
                    f"{run_id}: RTX geometry drift")
        else:
            require(receipt.get("metal_function") == "fg_early_l9" and
                    receipt.get("selected_grid") == [20, 12, 1] and
                    receipt.get("selected_block") == [16, 16, 1],
                    f"{run_id}: Metal geometry drift")
            require(receipt.get("architecture_rom_sha256") == ROM_SHA,
                    f"{run_id}: RCP ROM drift")
            require(receipt.get("surface_half_rounding") == "toward-zero",
                    f"{run_id}: surface rounding drift")
        address_free(receipt, run_id)
        outputs.append(output)
        receipts.append(receipt)
        receipt_hashes[run_id] = file_sha(receipt_path)
        dead_hashes.add(DEAD_SHA[poison])
    require(len({row["process_id"] for row in receipts}) == 4,
            "fresh-process gate failed")
    require(len({sha(output) for output in outputs}) == 1,
            "two-poison output invariance failed")
    require(len(dead_hashes) == 2, "dead-resource poison diversity failed")
    result = {
        "schema": BACKEND_SCHEMA,
        "backend": backend_name,
        "selected_unit": "canonical-L9/module-14/main_kernel",
        "module_sha256": MODULE_SHA,
        "source_identity": identity,
        "input_sha256": INPUT_SHA,
        "dead_resource_sha256_by_poison": {
            "a5": DEAD_SHA[0xA5], "5a": DEAD_SHA[0x5A]},
        "output_bytes": OUTPUT_BYTES,
        "output_sha256": OUTPUT_SHA,
        "run_receipt_sha256": receipt_hashes,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "dead_parameter_proven_output_invariant": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediate_fed_to_execution": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical(result)
    return result


def validate_root(receipt: dict[str, Any], backend: str) -> None:
    require(receipt.get("schema") == BACKEND_SCHEMA and
            receipt.get("backend") == backend, f"{backend}: receipt identity")
    body = dict(receipt)
    observed = body.pop("receipt_root_sha256", None)
    require(observed == canonical(body), f"{backend}: receipt root drift")
    address_free(receipt, backend)


def validate_final(rtx_path: Path, mac_path: Path,
                   rig_path: Path) -> dict[str, Any]:
    rtx = load(rtx_path)
    mac = load(mac_path)
    rig = load(rig_path)
    validate_root(rtx, "rtx-cuda-driver")
    validate_root(mac, "mac-metal")
    require(rtx["output_sha256"] == mac["output_sha256"] == OUTPUT_SHA,
            "RTX/Metal parity failed")
    require(rig.get("schema") == RIG_SCHEMA and rig.get("result") == "pass",
            "rig receipt identity drift")
    require(rig.get("module_sha256") == MODULE_SHA and
            rig.get("source_sha256") == RTX_SOURCE_SHA and
            rig.get("contract_sha256") == CONTRACT_SHA and
            rig.get("harness_sha256") == RTX_HARNESS_SHA,
            "rig source identity drift")
    require(rig.get("fresh_process_count") == 4 and
            rig.get("output_sha256") == OUTPUT_SHA,
            "rig repeat/output drift")
    address_free(rig, "rig receipt")
    result = {
        "schema": FINAL_SCHEMA,
        "selected_unit": rtx["selected_unit"],
        "module_sha256": MODULE_SHA,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "output_bytes": OUTPUT_BYTES,
        "output_sha256": OUTPUT_SHA,
        "rtx_mac_differing_bytes": 0,
        "rtx_mac_exact": True,
        "scope": "canonical reset-phase L9 arithmetic unit only",
        "claims": {
            "standalone_launch_exact": True,
            "connected_early_path_exact": False,
            "generated_frame_exact": False,
            "netal_integration": False,
        },
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--rtx", type=Path, required=True)
    final.add_argument("--mac", type=Path, required=True)
    final.add_argument("--rig", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (validate_backend(args.backend, args.root)
                  if args.command == "backend"
                  else validate_final(args.rtx, args.mac, args.rig))
        write(args.output, result)
        print(f"receipt_root_sha256={result['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
