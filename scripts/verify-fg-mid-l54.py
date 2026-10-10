#!/usr/bin/env python3
"""Fail-closed verifier for selected FG launch 54 on RTX and Metal."""

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


RUN_SCHEMA = "neuralstudio.fg.mid-l54-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.mid-l54-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.mid-l54-verification.v1"
RIG_SCHEMA = "neuralstudio.fg.mid-l54-rig.v1"
MODULE_SHA256 = "d8d41b87895a3119f24db4a8d4cb1ec4f73613b507e3d1d4de5dea0f36fe5dc5"
RCP_ROM_SHA256 = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
CONTRACT_SHA256 = "7be38a96208f41c7a4f76fd611b0ebd0d319f2be86107e4d3dd0ebe99ddc654f"
RTX_SOURCE_SHA256 = "70a4e48de43e6f29c66f27905ad2604ed6348f2207b5ae2f466e9e5c70c29bad"
RTX_HARNESS_SHA256 = "c07cabf67adb817d13ee0dfa7dd62a148eb910dd1b8c32622ce34da9dd30e9f0"
METAL_SOURCE_SHA256 = "6d0b9ca56ddca998a5063149db4a2638b71c6c7eda709cff18f8a1b8830af209"
MAC_HOST_SHA256 = "a5676faca10d997918198a59a33f14905c598a686bc7d568b4b417b6aa4e3552"
MAC_HARNESS_SHA256 = "13fa4546e8443ebed05849221a9a3c1c176a76b529b7b89be3990fa3d8064179"
OUTPUT_SHA256 = "b2747b5f0d605921d3b7471b792fa883f7d240b4f083d2757c077648f378a9ad"
WIDTH = 320
HEIGHT = 180
TEXTURE_BYTES = WIDTH * HEIGHT * 4
OFFSET_BYTES = WIDTH * HEIGHT * 8
OUTPUT_BYTES = TEXTURE_BYTES
MASK64 = (1 << 64) - 1
CASES = {
    "l54-poison-a5-process-a": 0xA5,
    "l54-poison-a5-process-b": 0xA5,
    "l54-poison-5a-process-a": 0x5A,
    "l54-poison-5a-process-b": 0x5A,
}
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    require_no_addresses(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_no_addresses(value: Any, label: str) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                if lowered != "address_fields_recorded":
                    require(
                        not any(
                            marker in lowered
                            for marker in (
                                "pointer",
                                "gpu_va",
                                "virtual_address",
                                "device_address",
                                "gpu_address",
                            )
                        ),
                        f"{label} contains address field at {path}.{key}",
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            require(
                ADDRESS_PATTERN.search(item) is None,
                f"{label} contains address-like value at {path}",
            )

    visit(value, "$")


def mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def texture(seed: int) -> bytes:
    values = (
        0x0000, 0x8000, 0x2000, 0xA000, 0x2400, 0xA400,
        0x2800, 0xA800, 0x2C00, 0xAC00, 0x3000, 0xB000,
        0x3400, 0xB400, 0x3800, 0xB800, 0x3C00,
    )
    result = bytearray(TEXTURE_BYTES)
    for index in range(WIDTH * HEIGHT * 2):
        bits = values[
            mix((seed + index * 0xD1B54A32D192ED03) & MASK64) % len(values)
        ]
        struct.pack_into("<H", result, index * 2, bits)
    return bytes(result)


def offset_buffer(seed: int, poison: int) -> bytes:
    offsets = (
        0x0000, 0x8000, 0x1400, 0x9400, 0x1800,
        0x9800, 0x1C00, 0x9C00, 0x2000,
    )
    result = bytearray([poison]) * OFFSET_BYTES
    for pixel in range(WIDTH * HEIGHT):
        mixed = mix((seed + pixel * 0x9E3779B97F4A7C15) & MASK64)
        struct.pack_into("<H", result, pixel * 8, offsets[mixed % len(offsets)])
        struct.pack_into(
            "<H", result, pixel * 8 + 2, offsets[(mixed >> 8) % len(offsets)]
        )
    return bytes(result)


TEXTURE_A = texture(0x243F6A8885A308D3)
TEXTURE_B = texture(0x13198A2E03707344)


def verify_repository_sources(repository: Path, backend: str) -> dict[str, str]:
    common = repository / "src/fg_mid_l54/contract.h"
    require(sha256_file(common) == CONTRACT_SHA256, "L54 contract source drift")
    if backend == "rtx":
        source = repository / "src/fg_mid_l54/rtx_oracle.cpp"
        harness = repository / "scripts/build-fg-mid-l54-rtx.ps1"
        require(sha256_file(source) == RTX_SOURCE_SHA256, "RTX source drift")
        require(sha256_file(harness) == RTX_HARNESS_SHA256, "RTX harness drift")
        return {
            "contract_sha256": CONTRACT_SHA256,
            "source_sha256": RTX_SOURCE_SHA256,
            "harness_sha256": RTX_HARNESS_SHA256,
        }
    metal = repository / "src/fg_mid_l54/mid_l54.metal"
    host = repository / "src/fg_mid_l54/macos_probe.mm"
    harness = repository / "scripts/build-macos-fg-mid-l54.sh"
    require(sha256_file(metal) == METAL_SOURCE_SHA256, "Metal source drift")
    require(sha256_file(host) == MAC_HOST_SHA256, "Mac host source drift")
    require(sha256_file(harness) == MAC_HARNESS_SHA256, "Mac harness drift")
    return {
        "contract_sha256": CONTRACT_SHA256,
        "metal_source_sha256": METAL_SOURCE_SHA256,
        "host_source_sha256": MAC_HOST_SHA256,
        "harness_sha256": MAC_HARNESS_SHA256,
    }


def validate_backend(backend: str, root: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parent.parent
    source_identity = verify_repository_sources(repository, backend)
    backend_name = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    receipts: list[dict[str, Any]] = []
    outputs: list[bytes] = []
    receipt_hashes: dict[str, str] = {}
    for run_id, poison in CASES.items():
        run_root = root / run_id
        receipt_path = run_root / "run-receipt.json"
        receipt = load_json(receipt_path)
        output = (run_root / "output.bin").read_bytes()
        expected_offset_a = offset_buffer(0xA4093822299F31D0, poison)
        expected_offset_b = offset_buffer(0x082EFA98EC4E6C89, poison)
        texture_a_after = (run_root / "texture-a-after.bin").read_bytes()
        texture_b_after = (run_root / "texture-b-after.bin").read_bytes()
        offset_a_after = (run_root / "offset-a-after.bin").read_bytes()
        offset_b_after = (run_root / "offset-b-after.bin").read_bytes()

        require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema drift")
        require(receipt.get("backend") == backend_name, f"{run_id}: backend drift")
        require(receipt.get("run_id") == run_id, f"{run_id}: run identity drift")
        require(receipt.get("poison_byte") == poison, f"{run_id}: poison drift")
        require(isinstance(receipt.get("process_id"), int), f"{run_id}: process id")
        require(receipt.get("selected_launch") == 54, f"{run_id}: launch drift")
        require(receipt.get("module_token") == 26, f"{run_id}: module token drift")
        require(receipt.get("module_sha256") == MODULE_SHA256,
                f"{run_id}: module hash drift")
        require(receipt.get("packed_parameter_bytes") == 88,
                f"{run_id}: packed ABI drift")
        require(receipt.get("dimensions") == [320, 180, 320, 180],
                f"{run_id}: dimensions drift")
        require(receipt.get("texture_pre_sha256") == [
                    sha256_bytes(TEXTURE_A), sha256_bytes(TEXTURE_B)],
                f"{run_id}: texture preimage drift")
        require(receipt.get("texture_post_sha256") == [
                    sha256_bytes(TEXTURE_A), sha256_bytes(TEXTURE_B)],
                f"{run_id}: texture receipt drift")
        require(texture_a_after == TEXTURE_A and texture_b_after == TEXTURE_B,
                f"{run_id}: generated texture drift")
        expected_offset_hashes = [
            sha256_bytes(expected_offset_a), sha256_bytes(expected_offset_b)
        ]
        require(receipt.get("offset_pre_sha256") == expected_offset_hashes,
                f"{run_id}: offset preimage drift")
        require(receipt.get("offset_post_sha256") == expected_offset_hashes,
                f"{run_id}: offset receipt drift")
        require(offset_a_after == expected_offset_a and
                offset_b_after == expected_offset_b,
                f"{run_id}: generated offset drift")
        require(len(output) == OUTPUT_BYTES and
                receipt.get("output_bytes") == OUTPUT_BYTES,
                f"{run_id}: output size drift")
        require(receipt.get("output_sha256") == sha256_bytes(output),
                f"{run_id}: output receipt hash drift")
        require(receipt.get("output_sha256") == OUTPUT_SHA256,
                f"{run_id}: selected output drift")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("canary_mismatched_bytes") == 0,
                f"{run_id}: input/canary integrity failed")
        require(receipt.get("unused_u64_upper_halves_poisoned") is True,
                f"{run_id}: poisoned unused halves gate failed")
        require(receipt.get("expected_output_opened") is False,
                f"{run_id}: expected output was opened")
        require(receipt.get("archived_intermediate_opened") is False,
                f"{run_id}: archived intermediate was opened")
        require(receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass", f"{run_id}: result drift")
        if backend == "rtx":
            require(receipt.get("function_name") == "main_kernel",
                    f"{run_id}: RTX function drift")
            require(receipt.get("grid") == [20, 12, 1] and
                    receipt.get("block") == [16, 16, 1],
                    f"{run_id}: RTX geometry drift")
        else:
            require(receipt.get("metal_function") == "fg_mid_l54",
                    f"{run_id}: Metal function drift")
            require(receipt.get("selected_grid") == [20, 12, 1] and
                    receipt.get("selected_block") == [16, 16, 1],
                    f"{run_id}: selected geometry drift")
            require(receipt.get("architecture_rom_sha256") == RCP_ROM_SHA256,
                    f"{run_id}: RCP ROM drift")
            require(receipt.get("normalized_point_coordinate_fraction_bits") == 12,
                    f"{run_id}: point-coordinate rule drift")
            require(receipt.get("surface_half_rounding") == "toward-zero",
                    f"{run_id}: half-store rule drift")
        require_no_addresses(receipt, run_id)
        receipts.append(receipt)
        outputs.append(output)
        receipt_hashes[run_id] = sha256_file(receipt_path)

    require(len({item["process_id"] for item in receipts}) == 4,
            "fresh-process gate failed")
    require(len({sha256_bytes(item) for item in outputs}) == 1,
            "two-poison output invariance failed")
    require(outputs[0] == outputs[1] == outputs[2] == outputs[3],
            "byte equality gate failed")
    result: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": backend_name,
        "selected_unit": "evaluate-0/launch-000054/main_kernel/module-26",
        "module_sha256": MODULE_SHA256,
        "source_identity": source_identity,
        "input": {
            "generator": "splitmix64-fg-mid-l54-v1",
            "texture_sha256": [sha256_bytes(TEXTURE_A), sha256_bytes(TEXTURE_B)],
            "offset_sha256_by_poison": {
                "a5": [
                    sha256_bytes(offset_buffer(0xA4093822299F31D0, 0xA5)),
                    sha256_bytes(offset_buffer(0x082EFA98EC4E6C89, 0xA5)),
                ],
                "5a": [
                    sha256_bytes(offset_buffer(0xA4093822299F31D0, 0x5A)),
                    sha256_bytes(offset_buffer(0x082EFA98EC4E6C89, 0x5A)),
                ],
            },
        },
        "output_bytes": OUTPUT_BYTES,
        "output_sha256": OUTPUT_SHA256,
        "run_receipt_sha256": receipt_hashes,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediate_fed_to_execution": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_hash(result)
    return result


def validate_backend_receipt(receipt: dict[str, Any], backend: str) -> None:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("schema") == BACKEND_SCHEMA and
            receipt.get("backend") == expected_backend,
            f"{backend}: backend receipt identity drift")
    body = dict(receipt)
    root = body.pop("receipt_root_sha256", None)
    require(root == canonical_hash(body), f"{backend}: backend receipt root drift")
    require(receipt.get("fresh_process_count") == 4 and
            receipt.get("two_process_two_poison_invariant") is True,
            f"{backend}: repeat gate drift")
    require(receipt.get("full_write_coverage_from_two_poison_invariance") is True,
            f"{backend}: write-coverage gate drift")
    require(receipt.get("expected_output_fed_to_execution") is False and
            receipt.get("archived_intermediate_fed_to_execution") is False,
            f"{backend}: forbidden ingress drift")
    require_no_addresses(receipt, backend)


def validate_final(rtx_path: Path, mac_path: Path,
                   rig_path: Path) -> dict[str, Any]:
    rtx = load_json(rtx_path)
    mac = load_json(mac_path)
    rig = load_json(rig_path)
    validate_backend_receipt(rtx, "rtx")
    validate_backend_receipt(mac, "mac")
    require(rtx.get("output_sha256") == mac.get("output_sha256") == OUTPUT_SHA256,
            "RTX/Metal parity failed")
    require(rtx.get("output_bytes") == mac.get("output_bytes") == OUTPUT_BYTES,
            "RTX/Metal output size drift")
    require(rig.get("schema") == RIG_SCHEMA and rig.get("result") == "pass",
            "RTX rig receipt identity drift")
    require(rig.get("module_sha256") == MODULE_SHA256,
            "RTX rig module hash drift")
    require(rig.get("source_sha256") == RTX_SOURCE_SHA256 and
            rig.get("contract_sha256") == CONTRACT_SHA256 and
            rig.get("harness_sha256") == RTX_HARNESS_SHA256,
            "RTX rig source identity drift")
    require(rig.get("fresh_process_count") == 4 and
            rig.get("output_sha256") == OUTPUT_SHA256,
            "RTX rig repeat/output drift")
    require_no_addresses(rig, "RTX rig receipt")
    result: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_unit": rtx["selected_unit"],
        "module_sha256": MODULE_SHA256,
        "packed_parameter_bytes": 88,
        "dimensions": [320, 180, 320, 180],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": sha256_file(rig_path),
        "output_bytes": OUTPUT_BYTES,
        "output_sha256": OUTPUT_SHA256,
        "rtx_mac_differing_bytes": 0,
        "rtx_mac_exact": True,
        "normalized_point_coordinate_fraction_bits": 12,
        "surface_half_rounding": "toward-zero",
        "scope": "one selected preprocessing launch; no connected-middle or complete-frame claim",
        "claims": {
            "standalone_launch_exact": True,
            "connected_middle_exact": False,
            "complete_selected_model_path_exact": False,
            "complete_generated_frame_exact": False,
            "netal_integration": False,
        },
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
    final.add_argument("--rig", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend":
            result = validate_backend(args.backend, args.root)
        else:
            result = validate_final(args.rtx, args.mac, args.rig)
        write_json(args.output, result)
        print(f"receipt_root_sha256={result['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
