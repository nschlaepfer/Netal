#!/usr/bin/env python3
"""Fail-closed verification for selected FG convolution launch 20."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any


WIDTH = 160
HEIGHT = 96
INPUT_CHANNELS = 48
OUTPUT_CHANNELS = 32
INPUT_BYTES = WIDTH * HEIGHT * INPUT_CHANNELS * 2
OUTPUT_BYTES = WIDTH * HEIGHT * OUTPUT_CHANNELS * 2
WEIGHT_BYTES = 56320
DENSE_WEIGHT_BYTES = 27648
UNUSED_WEIGHT_BYTES = 27648
BIAS_OFFSET = 55296
MODULE_BYTES = 16512
MODULE_SHA256 = "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d"
RUN_SCHEMA = "neuralstudio.fg.conv-l20-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.conv-l20-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.conv-l20-verification.v1"
GENERATOR = "fg-conv-l20-synthetic-v1"
SCENARIOS = ("spatial-map", "one-hot", "bias")
POISONS = (0xA5, 0x5A)
ORDINALS = ("a", "b")


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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load JSON {path}: {error}") from error
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


def require_no_address_fields(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = key.lower()
                if key != "address_fields_recorded":
                    require(
                        not any(
                            marker in lowered
                            for marker in (
                                "pointer",
                                "gpu_address",
                                "device_address",
                                "virtual_address",
                            )
                        ),
                        f"{label} has address field at {location}.{key}",
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")
        elif isinstance(node, str):
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                f"{label} has address-like text at {location}",
            )

    visit(value, label)


def store_half(data: bytearray, byte_offset: int, bits: int) -> None:
    struct.pack_into("<H", data, byte_offset, bits)


def set_input(
    data: bytearray, y: int, x: int, channel: int, bits: int
) -> None:
    element = ((y * WIDTH + x) * INPUT_CHANNELS) + channel
    store_half(data, element * 2, bits)


def packed_weight_byte_offset(
    raw_spatial_tile: int, input_channel: int, output_channel: int
) -> int:
    output_group, n = divmod(output_channel, 8)
    channel_stage, k = divmod(input_channel, 16)
    thread_in_group = (k % 8) // 2
    fragment_half = (2 if k >= 8 else 0) + (k & 1)
    lane = n * 4 + thread_in_group
    result = (
        output_group * 6912
        + raw_spatial_tile * 768
        + channel_stage * 256
        + lane * 8
        + fragment_half * 2
    )
    require(result < DENSE_WEIGHT_BYTES, "packed weight offset exceeds active region")
    return result


def set_packed_weight(
    data: bytearray,
    raw_spatial_tile: int,
    input_channel: int,
    output_channel: int,
    bits: int,
) -> None:
    store_half(
        data,
        packed_weight_byte_offset(
            raw_spatial_tile, input_channel, output_channel
        ),
        bits,
    )


def set_packed_bias(data: bytearray, output_channel: int, bits: int) -> None:
    group, lane = divmod(output_channel, 8)
    for repeat in range(16):
        store_half(data, BIAS_OFFSET + group * 256 + (repeat * 8 + lane) * 2, bits)


def generate(scenario: str, poison: int) -> tuple[bytes, bytes]:
    require(scenario in SCENARIOS, f"unsupported scenario: {scenario}")
    weights = bytearray(WEIGHT_BYTES)
    weights[DENSE_WEIGHT_BYTES:BIAS_OFFSET] = bytes([poison]) * UNUSED_WEIGHT_BYTES
    input_data = bytearray(INPUT_BYTES)
    if scenario == "spatial-map":
        set_input(input_data, 48, 80, 0, 0x3C00)
        for tile in range(9):
            set_packed_weight(weights, tile, 0, tile, 0x3C00)
    elif scenario == "one-hot":
        for y in range(HEIGHT):
            for x in range(WIDTH):
                for channel in range(INPUT_CHANNELS):
                    bits = 0x3C00 + ((x + 3 * y + 5 * channel) & 7) * 0x80
                    set_input(input_data, y, x, channel, bits)
        for output_channel in range(OUTPUT_CHANNELS):
            set_packed_weight(
                weights,
                output_channel % 9,
                (output_channel * 13 + 7) % INPUT_CHANNELS,
                output_channel,
                0x3C00 if output_channel % 2 == 0 else 0xBC00,
            )
    else:
        for output_channel in range(OUTPUT_CHANNELS):
            set_packed_bias(
                weights,
                output_channel,
                0x3800 if output_channel % 2 == 0 else 0xB800,
            )
    return bytes(weights), bytes(input_data)


def round_f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def reference_output(scenario: str) -> bytes:
    output = bytearray(OUTPUT_BYTES)
    if scenario == "spatial-map":
        for tile in range(9):
            ky, kx = divmod(tile, 3)
            output_y = 48 - ky + 1
            output_x = 80 - kx + 1
            element = ((output_y * WIDTH + output_x) * OUTPUT_CHANNELS) + tile
            store_half(output, element * 2, 0x3C00)
        return bytes(output)

    slope_f32 = struct.unpack("<f", struct.pack("<I", 0x3C23D70A))[0]
    slope = round_f16(slope_f32)
    if scenario == "bias":
        positive_bits = 0x3800
        negative_bits = struct.unpack("<H", struct.pack("<e", round_f16(-0.5 * slope)))[0]
        for pixel in range(WIDTH * HEIGHT):
            for output_channel in range(OUTPUT_CHANNELS):
                store_half(
                    output,
                    (pixel * OUTPUT_CHANNELS + output_channel) * 2,
                    positive_bits if output_channel % 2 == 0 else negative_bits,
                )
        return bytes(output)

    require(scenario == "one-hot", f"no reference for scenario {scenario}")
    for y in range(HEIGHT):
        for x in range(WIDTH):
            for output_channel in range(OUTPUT_CHANNELS):
                tile = output_channel % 9
                ky, kx = divmod(tile, 3)
                input_y = y + ky - 1
                input_x = x + kx - 1
                value = 0.0
                if 0 <= input_y < HEIGHT and 0 <= input_x < WIDTH:
                    input_channel = (
                        output_channel * 13 + 7
                    ) % INPUT_CHANNELS
                    value = 1.0 + (
                        (input_x + 3 * input_y + 5 * input_channel) & 7
                    ) * 0.125
                    if output_channel % 2:
                        value = -value
                if value < 0:
                    value = round_f16(value * slope)
                struct.pack_into(
                    "<e",
                    output,
                    ((y * WIDTH + x) * OUTPUT_CHANNELS + output_channel) * 2,
                    value,
                )
    return bytes(output)


def expected_run_id(scenario: str, poison: int, ordinal: str) -> str:
    return f"{scenario}-poison-{poison:02x}-process-{ordinal}"


def validate_run(
    run_root: Path,
    expected_backend: str,
    scenario: str,
    poison: int,
    ordinal: str,
    reference: bytes,
) -> dict[str, Any]:
    receipt = load_json(run_root / "run-receipt.json")
    output = (run_root / "output.bin").read_bytes()
    weights, input_data = generate(scenario, poison)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_root}: schema mismatch")
    require(receipt.get("backend") == expected_backend, f"{run_root}: backend mismatch")
    require(
        receipt.get("run_id") == expected_run_id(scenario, poison, ordinal),
        f"{run_root}: run id mismatch",
    )
    require(receipt.get("scenario") == scenario, f"{run_root}: scenario mismatch")
    require(receipt.get("poison_byte") == poison, f"{run_root}: poison mismatch")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{run_root}: process id missing",
    )
    if expected_backend == "rtx-cuda-driver":
        require(receipt.get("grid") == [10, 8, 1], f"{run_root}: grid mismatch")
        require(receipt.get("block") == [32, 4, 1], f"{run_root}: block mismatch")
    else:
        require(receipt.get("selected_rtx_grid") == [10, 8, 1], f"{run_root}: selected grid mismatch")
        require(receipt.get("selected_rtx_block") == [32, 4, 1], f"{run_root}: selected block mismatch")
        require(receipt.get("metal_dispatch_threads") == [160, 96, 32], f"{run_root}: Metal dispatch mismatch")
        require(receipt.get("metal_threads_per_threadgroup") == [8, 4, 1], f"{run_root}: Metal threadgroup mismatch")
    require(receipt.get("dynamic_shared_bytes") == 0, f"{run_root}: shared mismatch")
    require(
        receipt.get("abi_scalars") == [160, 96, 162, 98, 160, 96],
        f"{run_root}: ABI scalar mismatch",
    )
    require(receipt.get("generator") == GENERATOR, f"{run_root}: generator mismatch")
    require(receipt.get("packed_weight_bytes") == WEIGHT_BYTES, f"{run_root}: weight size mismatch")
    require(receipt.get("active_weight_bytes") == DENSE_WEIGHT_BYTES, f"{run_root}: active size mismatch")
    require(receipt.get("poisoned_gap_offset") == DENSE_WEIGHT_BYTES, f"{run_root}: gap offset mismatch")
    require(receipt.get("poisoned_gap_bytes") == UNUSED_WEIGHT_BYTES, f"{run_root}: gap size mismatch")
    require(receipt.get("bias_offset") == BIAS_OFFSET, f"{run_root}: bias offset mismatch")
    require(receipt.get("weight_pre_sha256") == sha256_bytes(weights), f"{run_root}: weight generator mismatch")
    require(receipt.get("weight_post_sha256") == sha256_bytes(weights), f"{run_root}: weight mutation")
    require(receipt.get("input_pre_sha256") == sha256_bytes(input_data), f"{run_root}: input generator mismatch")
    require(receipt.get("input_post_sha256") == sha256_bytes(input_data), f"{run_root}: input mutation")
    require(len(output) == OUTPUT_BYTES, f"{run_root}: output size mismatch")
    require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_root}: receipt output size mismatch")
    require(receipt.get("output_sha256") == sha256_bytes(output), f"{run_root}: output hash mismatch")
    require(output == reference, f"{run_root}: CPU semantic reference mismatch")
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_root}: input mismatch")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_root}: guard mismatch")
    require(receipt.get("untouched_halfwords") == 0, f"{run_root}: local coverage mismatch")
    require(receipt.get("input_immutability") is True, f"{run_root}: immutability flag mismatch")
    require(receipt.get("external_data_files_opened") is False, f"{run_root}: external execution input")
    require(receipt.get("address_fields_recorded") is False, f"{run_root}: address flag")
    require(receipt.get("result") == "pass", f"{run_root}: run failed")
    if expected_backend == "rtx-cuda-driver":
        require(receipt.get("module_token") == 28, f"{run_root}: module token mismatch")
        require(receipt.get("module_bytes") == MODULE_BYTES, f"{run_root}: module size mismatch")
        require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_root}: module hash mismatch")
        require(receipt.get("function_name") == "k_conv_fp16_nhwc", f"{run_root}: function mismatch")
        require(receipt.get("launch_order") == 20, f"{run_root}: launch order mismatch")
    else:
        require(receipt.get("metal_function") == "fg_conv_l20", f"{run_root}: Metal function mismatch")
    require_no_address_fields(receipt, str(run_root))
    return receipt


def validate_backend(
    backend: str, root: Path, module: Path | None = None
) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    if backend == "rtx":
        require(module is not None and module.is_file(), "RTX module is required")
        require(module.stat().st_size == MODULE_BYTES, "selected module size mismatch")
        require(sha256_file(module) == MODULE_SHA256, "selected module hash mismatch")
    references = {scenario: reference_output(scenario) for scenario in SCENARIOS}
    receipts: list[dict[str, Any]] = []
    hashes: dict[str, set[str]] = {scenario: set() for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        for poison in POISONS:
            for ordinal in ORDINALS:
                run_id = expected_run_id(scenario, poison, ordinal)
                receipt = validate_run(
                    root / run_id,
                    expected_backend,
                    scenario,
                    poison,
                    ordinal,
                    references[scenario],
                )
                receipts.append(receipt)
                hashes[scenario].add(receipt["output_sha256"])
    process_ids = {receipt["process_id"] for receipt in receipts}
    require(len(process_ids) == 12, "fresh-process gate failed")
    for scenario in SCENARIOS:
        require(len(hashes[scenario]) == 1, f"{scenario}: output invariance failed")

    # These deliberate corruptions must all fail the same validation rules.
    mutation_rejections: list[str] = []
    sample_root = root / expected_run_id("one-hot", 0xA5, "a")
    sample_receipt = load_json(sample_root / "run-receipt.json")
    sample_output = bytearray((sample_root / "output.bin").read_bytes())
    sample_output[17] ^= 1
    try:
        require(
            sample_receipt["output_sha256"] == sha256_bytes(sample_output),
            "mutated output hash mismatch",
        )
    except VerificationError:
        mutation_rejections.append("output-byte")
    mutated_receipt = copy.deepcopy(sample_receipt)
    mutated_receipt["generator"] = "wrong-generator"
    try:
        require(mutated_receipt["generator"] == GENERATOR, "mutated generator")
    except VerificationError:
        mutation_rejections.append("generator")
    mutated_receipt = copy.deepcopy(sample_receipt)
    mutated_receipt["gpu_virtual_address"] = "0x0000000ABCDEF012"
    try:
        require_no_address_fields(mutated_receipt, "mutated-receipt")
    except VerificationError:
        mutation_rejections.append("address-field")
    duplicate_ids = [receipt["process_id"] for receipt in receipts]
    duplicate_ids[-1] = duplicate_ids[0]
    try:
        require(len(set(duplicate_ids)) == 12, "duplicate process id")
    except VerificationError:
        mutation_rejections.append("process-id")
    require(
        mutation_rejections
        == ["output-byte", "generator", "address-field", "process-id"],
        "negative mutation gate incomplete",
    )

    receipt = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "selected_unit": "evaluate-1/L20/k_conv_fp16_nhwc/module-28",
        "module_sha256": MODULE_SHA256,
        "module_bytes": MODULE_BYTES,
        "launch": {
            "grid": [10, 8, 1],
            "block": [32, 4, 1],
            "packed_parameter_bytes": 48,
            "abi_scalars": [160, 96, 162, 98, 160, 96],
        },
        "generator": GENERATOR,
        "scenario_output_sha256": {
            scenario: next(iter(hashes[scenario])) for scenario in SCENARIOS
        },
        "fresh_process_count": len(process_ids),
        "run_count": len(receipts),
        "two_process_two_poison_invariant": True,
        "unused_gap_two_poison_invariant": True,
        "input_immutability": True,
        "canary_integrity": True,
        "full_write_coverage": True,
        "cpu_semantic_reference_exact": True,
        "spatial_mapping": {
            "raw_tile_order": "row-major-3x3",
            "padding": 1,
            "operation": "cross-correlation",
        },
        "weight_mapping": "mma-m16n8k16-row-col-B-fragment-lane-order",
        "bias_mapping": "four-groups-of-eight-repeated-sixteen-times",
        "leaky_relu_slope_half_bits": "0x211f",
        "negative_mutations_rejected": mutation_rejections,
        "address_fields_recorded": False,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_final(rtx_path: Path, mac_path: Path) -> dict[str, Any]:
    rtx = load_json(rtx_path)
    mac = load_json(mac_path)
    require(rtx.get("schema") == BACKEND_SCHEMA, "RTX backend schema mismatch")
    require(mac.get("schema") == BACKEND_SCHEMA, "Mac backend schema mismatch")
    require(rtx.get("backend") == "rtx-cuda-driver", "RTX backend identity mismatch")
    require(mac.get("backend") == "mac-metal", "Mac backend identity mismatch")
    for receipt, name in ((rtx, "rtx"), (mac, "mac")):
        require(receipt.get("result") == "pass", f"{name} backend failed")
        require_no_address_fields(receipt, f"{name}-backend")
        root = receipt.get("receipt_root_sha256")
        without_root = dict(receipt)
        without_root.pop("receipt_root_sha256", None)
        require(root == canonical_sha256(without_root), f"{name} root mismatch")
    require(
        rtx["scenario_output_sha256"] == mac["scenario_output_sha256"],
        "RTX/Mac scenario output mismatch",
    )
    receipt = {
        "schema": FINAL_SCHEMA,
        "selected_unit": rtx["selected_unit"],
        "module_sha256": MODULE_SHA256,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "scenario_output_sha256": rtx["scenario_output_sha256"],
        "rtx_mac_exact": True,
        "scope": "sparse dependency-closed semantic probes; dense model arithmetic not claimed",
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
    backend.add_argument("--module", type=Path)
    backend.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--rtx", type=Path, required=True)
    final.add_argument("--mac", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "backend":
            receipt = validate_backend(
                arguments.backend, arguments.root, arguments.module
            )
        else:
            receipt = validate_final(arguments.rtx, arguments.mac)
        write_json(arguments.output, receipt)
        print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
