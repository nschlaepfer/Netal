#!/usr/bin/env python3
"""Verify the dependency-closed Frame Generation central-block unit.

CUDA and Metal generate their tensor inputs independently.  This verifier is
the third implementation of that generator and consumes only post-execution
snapshots.  Expected/oracle tensors are never inputs to live execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
WIDTH = 640
HEIGHT = 368
PIXELS = WIDTH * HEIGHT
INPUT_BYTES = (
    PIXELS * 8,
    PIXELS * 8,
    PIXELS * 2,
    PIXELS * 8,
    PIXELS * 2,
    PIXELS * 6,
)
OUTPUT_BYTES = PIXELS * 36
MODULE_BYTES = 5752
MODULE_SHA256 = "184542107e98be3dc6229251c19cae4ecc44a7cc1b78cbf741b7fb3b8d81f060"
RAW_PARAMETER_SHA256 = "9e52540566596e311b3466c42d6903c49811655c510248a98b264e5864c9a2a7"
ADDRESS_FREE_PARAMETER_SHA256 = "71112a8cfc99a78d21bd18c8b848974aa94009bb2c688bd2fe51a714f5b39f49"
GENERATOR = "splitmix64-fg-central-block-signed-v1"
RUN_SCHEMA = "neuralstudio.fg.central-block-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.central-block-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.central-block-verification.v1"
CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}


class VerificationError(RuntimeError):
    """An evidence or parity gate failed closed."""


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


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


SPECIAL_HALF_BITS = (
    0x0000, 0x8000, 0x2C00, 0xAC00,
    0x3400, 0xB400, 0x3555, 0xB555,
    0x3800, 0xB800, 0x3A00, 0xBA00,
    0x3C00, 0xBC00, 0x3E00, 0xBE00,
)
MOTION_HALF_BITS = (
    0x0000, 0x3400, 0xB400, 0x3800,
    0xB800, 0x3A00, 0xBA00, 0x3D00,
    0xBD00, 0x3E00, 0xBE00, 0x4080,
    0xC080, 0x3555, 0xB555, 0xBC00,
)


def tensor_half_bits(lane: int, index: int) -> int:
    mixed = splitmix64(
        0xD6E8FEB86659FD93
        ^ (((lane + 1) * 0xA0761D6478BD642F) & ((1 << 64) - 1))
        ^ ((index * 0xE7037ED1A0B428DB) & ((1 << 64) - 1))
    )
    if mixed & 7 == 0:
        return SPECIAL_HALF_BITS[(mixed >> 8) & 15]
    sign = 0x8000 if mixed >> 63 else 0
    exponent = 11 + ((mixed >> 12) & 5)
    return sign | (exponent << 10) | ((mixed >> 19) & 0x03FF)


def motion_half_bits(pixel: int, component: int) -> int:
    x = pixel % WIDTH
    y = pixel // WIDTH
    if component == 0 and x in (0, 1):
        return 0xBA00
    if component == 1 and y == 0:
        return 0xB555
    if component == 1 and y == 1:
        return 0xBE00
    if component == 2 and x == 0:
        return 0xBD00
    if component == 3 and y == 0:
        return 0xBA00
    if component in (0, 2) and x + 1 == WIDTH:
        return 0x3A00
    if component in (1, 3) and y + 1 == HEIGHT:
        return 0x3A00
    mixed = splitmix64(
        0x8EBC6AF09C88C6E3
        ^ ((pixel * 0x589965CC75374CC3) & ((1 << 64) - 1))
        ^ (((component + 1) * 0x1D8E4E27C47D124F) & ((1 << 64) - 1))
    )
    return MOTION_HALF_BITS[mixed & 15]


def generate_inputs() -> tuple[bytes, ...]:
    lanes = tuple(bytearray(size) for size in INPUT_BYTES)
    for pixel in range(PIXELS):
        for component in range(4):
            struct.pack_into(
                "<H", lanes[0], (pixel * 4 + component) * 2,
                tensor_half_bits(0, pixel * 4 + component),
            )
            struct.pack_into(
                "<H", lanes[1], (pixel * 4 + component) * 2,
                tensor_half_bits(1, pixel * 4 + component),
            )
            struct.pack_into(
                "<H", lanes[3], (pixel * 4 + component) * 2,
                motion_half_bits(pixel, component),
            )
        struct.pack_into("<H", lanes[2], pixel * 2, tensor_half_bits(2, pixel))
        struct.pack_into("<H", lanes[4], pixel * 2, tensor_half_bits(4, pixel))
        for component in range(3):
            struct.pack_into(
                "<H", lanes[5], (pixel * 3 + component) * 2,
                tensor_half_bits(5 + component, pixel * 3 + component),
            )
    return tuple(bytes(lane) for lane in lanes)


def half_from_bits(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def quantize_half(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def numerical_probe_counts(inputs: tuple[bytes, ...]) -> dict[str, int]:
    motion = inputs[3]
    negative_fractional = 0
    clamped_low = 0
    clamped_high = 0
    for pixel in range(PIXELS):
        x = pixel % WIDTH
        y = pixel // WIDTH
        for component in range(4):
            axis = x if component in (0, 2) else y
            extent = WIDTH if component in (0, 2) else HEIGHT
            bits = struct.unpack_from("<H", motion, (pixel * 4 + component) * 2)[0]
            offset = half_from_bits(bits)
            scaled = quantize_half(quantize_half(2.0) * offset)
            coordinate = quantize_half(quantize_half(float(axis)) + scaled)
            base = math.floor(coordinate)
            base_half = quantize_half(float(base))
            fraction = quantize_half(coordinate - base_half)
            negative_fractional += coordinate < 0.0 and fraction != 0.0
            clamped_low += base < 0
            clamped_high += base + 1 >= extent

    # Show that the chosen signed tensor domain distinguishes a separately
    # rounded half multiply/add sequence from an illicit fused replacement.
    contraction_distinguishers = 0
    sample = inputs[0]
    for pixel in range(min(8192, PIXELS)):
        motion_x = half_from_bits(
            struct.unpack_from("<H", motion, pixel * 8)[0]
        )
        fraction = quantize_half(abs(motion_x) - math.floor(abs(motion_x)))
        weight = quantize_half(quantize_half(1.0) - fraction)
        for component in range(4):
            value = half_from_bits(
                struct.unpack_from("<H", sample, (pixel * 4 + component) * 2)[0]
            )
            addend = half_from_bits(
                struct.unpack_from(
                    "<H", sample,
                    (((pixel + 1) % PIXELS) * 4 + component) * 2,
                )[0]
            )
            separate = quantize_half(quantize_half(weight * value) + addend)
            fused = quantize_half(weight * value + addend)
            contraction_distinguishers += separate != fused
    return {
        "negative_fractional_coordinate_count": negative_fractional,
        "low_clamp_coordinate_count": clamped_low,
        "high_clamp_coordinate_count": clamped_high,
        "half_contraction_distinguishing_sample_count": contraction_distinguishers,
    }


def require_exact_bytes(actual: bytes, expected: bytes, label: str) -> None:
    if actual == expected:
        return
    differing = sum(left != right for left, right in zip(actual, expected))
    differing += abs(len(actual) - len(expected))
    first = next(
        (
            index
            for index, (left, right) in enumerate(zip(actual, expected))
            if left != right
        ),
        min(len(actual), len(expected)),
    )
    raise VerificationError(
        f"{label} mismatch: differing_bytes={differing} first_offset={first}"
    )


def untouched_pixels(output: bytes, poison: int) -> int:
    require(len(output) == OUTPUT_BYTES, "write coverage output size mismatch")
    marker = bytes([poison]) * 36
    return sum(
        output[pixel * 36 : pixel * 36 + 36] == marker
        for pixel in range(PIXELS)
    )


def require_no_address_fields(receipt: dict[str, Any], label: str) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.lower()
                if key != "address_fields_recorded":
                    require(
                        not any(fragment in lowered for fragment in (
                            "pointer", "gpu_address", "device_address"
                        )),
                        f"{label} contains address-bearing field {path}.{key}",
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str):
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", value) is None,
                f"{label} contains a raw address-like value at {path}",
            )

    visit(receipt, label)


def validate_run(
    run_root: Path,
    backend: str,
    run_id: str,
    poison: int,
    generated_inputs: tuple[bytes, ...],
) -> tuple[dict[str, Any], bytes]:
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_root}: schema mismatch")
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("backend") == expected_backend, f"{run_root}: backend")
    require(receipt.get("run_id") == run_id, f"{run_root}: run id")
    require(receipt.get("poison_byte") == poison, f"{run_root}: poison")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{run_root}: process id",
    )
    require(receipt.get("grid") == [40, 23, 1], f"{run_root}: grid")
    require(receipt.get("block") == [16, 16, 1], f"{run_root}: block")
    require(receipt.get("dynamic_shared_bytes") == 0, f"{run_root}: shared")
    require(receipt.get("width") == WIDTH, f"{run_root}: width")
    require(receipt.get("height") == HEIGHT, f"{run_root}: height")
    require(receipt.get("factor_f32") == 0.5, f"{run_root}: factor")
    require(receipt.get("factor_f16_bits") == 0x3800, f"{run_root}: factor bits")
    require(receipt.get("input_generator") == GENERATOR, f"{run_root}: generator")
    require(receipt.get("input_bytes") == list(INPUT_BYTES), f"{run_root}: sizes")
    require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_root}: output size")
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_root}: input")
    require(receipt.get("input_immutability") is True, f"{run_root}: immutability")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_root}: canary")
    require(receipt.get("full_write_coverage") is True, f"{run_root}: coverage")
    require(receipt.get("external_tensor_inputs_opened") is False, f"{run_root}: external")
    require(receipt.get("address_fields_recorded") is False, f"{run_root}: address")
    require(receipt.get("result") == "pass", f"{run_root}: result")
    require_no_address_fields(receipt, str(receipt_path))

    expected_hashes = [sha256_bytes(value) for value in generated_inputs]
    require(receipt.get("input_pre_sha256") == expected_hashes, f"{run_root}: pre hashes")
    require(receipt.get("input_post_sha256") == expected_hashes, f"{run_root}: post hashes")
    for lane, expected in enumerate(generated_inputs):
        actual = (run_root / f"input-{lane}-after.bin").read_bytes()
        require_exact_bytes(actual, expected, f"{run_root}/input-{lane}-after.bin")

    if backend == "rtx":
        require(receipt.get("module_token") == 61, f"{run_root}: module token")
        require(receipt.get("module_bytes") == MODULE_BYTES, f"{run_root}: module bytes")
        require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_root}: module hash")
        require(receipt.get("launch_order") == 71, f"{run_root}: launch order")
        require(
            receipt.get("function_names") == [
                "k_central_block_original", "k_central_block"
            ],
            f"{run_root}: function names",
        )
        require(receipt.get("entry_mismatched_bytes") == 0, f"{run_root}: entries")
        require(receipt.get("entry_byte_equality") is True, f"{run_root}: equality")
        require(receipt.get("original_untouched_pixels") == 0, f"{run_root}: original coverage")
        require(receipt.get("optimized_untouched_pixels") == 0, f"{run_root}: optimized coverage")
        original = (run_root / "output-original.bin").read_bytes()
        optimized = (run_root / "output-optimized.bin").read_bytes()
        require_exact_bytes(original, optimized, f"{run_root}: NVIDIA entry equality")
        require(receipt.get("original_output_sha256") == sha256_bytes(original), f"{run_root}: original hash")
        require(receipt.get("optimized_output_sha256") == sha256_bytes(optimized), f"{run_root}: optimized hash")
        output = optimized
    else:
        require(receipt.get("reconstruction_unit") == "L71/k_central_block", f"{run_root}: unit")
        require(receipt.get("metal_function") == "fg_central_block", f"{run_root}: function")
        require(receipt.get("store_width_contract") == "nine-u32-per-pixel", f"{run_root}: stores")
        require(receipt.get("untouched_pixels") == 0, f"{run_root}: coverage count")
        output = (run_root / "output-metal.bin").read_bytes()
        require(receipt.get("output_sha256") == sha256_bytes(output), f"{run_root}: output hash")

    require(len(output) == OUTPUT_BYTES, f"{run_root}: output artifact size")
    require(untouched_pixels(output, poison) == 0, f"{run_root}: recomputed coverage")
    return receipt, output


def verify_backend(arguments: argparse.Namespace) -> int:
    root = Path(arguments.root).resolve()
    generated_inputs = generate_inputs()
    probes = numerical_probe_counts(generated_inputs)
    require(probes["negative_fractional_coordinate_count"] > 0, "negative floor domain absent")
    require(probes["low_clamp_coordinate_count"] > 0, "low clamp domain absent")
    require(probes["high_clamp_coordinate_count"] > 0, "high clamp domain absent")
    require(
        probes["half_contraction_distinguishing_sample_count"] > 0,
        "input domain does not distinguish half contraction",
    )
    receipts: list[dict[str, Any]] = []
    outputs: list[bytes] = []
    for run_id, poison in CASES.items():
        receipt, output = validate_run(
            root / run_id, arguments.backend, run_id, poison, generated_inputs
        )
        receipts.append(receipt)
        outputs.append(output)
    process_ids = {receipt["process_id"] for receipt in receipts}
    require(len(process_ids) == 4, "fresh-process gate failed")
    for index, output in enumerate(outputs[1:], 1):
        require_exact_bytes(output, outputs[0], f"fresh process output {index}")

    module_identity: dict[str, Any] | None = None
    entry_equality: bool | None = None
    if arguments.backend == "rtx":
        require(arguments.module is not None, "RTX verification requires --module")
        module_path = Path(arguments.module).resolve()
        require(module_path.stat().st_size == MODULE_BYTES, "module byte size mismatch")
        require(sha256_file(module_path) == MODULE_SHA256, "module hash mismatch")
        module_identity = {
            "token": 61,
            "bytes": MODULE_BYTES,
            "sha256": MODULE_SHA256,
            "functions": ["k_central_block_original", "k_central_block"],
        }
        entry_equality = True

    result: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": arguments.backend,
        "reconstruction_unit": "L71/k_central_block",
        "input_generator": GENERATOR,
        "input_sha256": [sha256_bytes(value) for value in generated_inputs],
        "input_bytes": list(INPUT_BYTES),
        "output_bytes": OUTPUT_BYTES,
        "output_sha256": sha256_bytes(outputs[0]),
        "run_ids": list(CASES),
        "fresh_process_count": 4,
        "poison_values": [0x5A, 0xA5],
        "poison_output_invariance": True,
        "input_immutability": True,
        "canary_integrity": True,
        "full_write_coverage": True,
        "numerical_probe_counts": probes,
        "nvidia_entry_byte_equality": entry_equality,
        "selected_module": module_identity,
        "raw_parameter_sha256": RAW_PARAMETER_SHA256,
        "address_free_parameter_sha256": ADDRESS_FREE_PARAMETER_SHA256,
        "external_expected_tensors_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    require_no_address_fields(result, "backend receipt")
    result["canonical_sha256_without_self"] = canonical_sha256(result)
    write_json(Path(arguments.output), result)
    print(f"backend={arguments.backend}")
    print(f"output_sha256={result['output_sha256']}")
    print("result=pass")
    return 0


def verify_final(arguments: argparse.Namespace) -> int:
    rtx_root = Path(arguments.rtx_root).resolve()
    mac_root = Path(arguments.mac_root).resolve()
    rtx = load_json(rtx_root / "backend-receipt.json")
    mac = load_json(mac_root / "backend-receipt.json")
    rig = load_json(rtx_root / "rig-receipt.json")
    mac_run = load_json(mac_root / "poison-a5-process-a" / "run-receipt.json")
    require(rtx.get("schema") == BACKEND_SCHEMA and rtx.get("backend") == "rtx", "RTX backend receipt mismatch")
    require(mac.get("schema") == BACKEND_SCHEMA and mac.get("backend") == "mac", "Mac backend receipt mismatch")
    require(rtx.get("result") == "pass" and mac.get("result") == "pass", "backend did not pass")
    require(rtx.get("input_generator") == mac.get("input_generator") == GENERATOR, "generator mismatch")
    require(rtx.get("input_sha256") == mac.get("input_sha256"), "input provenance mismatch")
    require(rtx.get("output_sha256") == mac.get("output_sha256"), "RTX/Metal output hash mismatch")
    rtx_output = rtx_root / "poison-a5-process-a" / "output-optimized.bin"
    mac_output = mac_root / "poison-a5-process-a" / "output-metal.bin"
    require_exact_bytes(rtx_output.read_bytes(), mac_output.read_bytes(), "RTX/Metal output")

    source_paths = [
        REPOSITORY / "src/fg_central_block/rtx_oracle.cpp",
        REPOSITORY / "src/fg_central_block/macos_probe.mm",
        REPOSITORY / "src/fg_central_block/central_block.metal",
        REPOSITORY / "scripts/build-fg-central-block-rtx.ps1",
        REPOSITORY / "scripts/build-macos-fg-central-block.sh",
        REPOSITORY / "scripts/verify-fg-central-block.py",
    ]
    for path in source_paths:
        require(path.is_file(), f"source artifact missing: {path}")
    result: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "reconstruction_unit": "L71/k_central_block",
        "function_name": "k_central_block",
        "source_module": {
            "token": 61,
            "bytes": MODULE_BYTES,
            "sha256": MODULE_SHA256,
            "functions_compared": ["k_central_block_original", "k_central_block"],
        },
        "live_launch": {
            "order": 71,
            "grid": [40, 23, 1],
            "block": [16, 16, 1],
            "dynamic_shared_bytes": 0,
            "width": WIDTH,
            "height": HEIGHT,
            "factor_f32": 0.5,
            "raw_parameter_sha256": RAW_PARAMETER_SHA256,
            "address_free_parameter_sha256": ADDRESS_FREE_PARAMETER_SHA256,
        },
        "input_generator": GENERATOR,
        "input_bytes": list(INPUT_BYTES),
        "output_bytes": OUTPUT_BYTES,
        "output_sha256": rtx["output_sha256"],
        "rtx_backend_receipt_sha256": sha256_file(rtx_root / "backend-receipt.json"),
        "mac_backend_receipt_sha256": sha256_file(mac_root / "backend-receipt.json"),
        "rtx_metal_byte_exact": True,
        "nvidia_original_optimized_byte_exact": True,
        "nvidia_shared_staging_remap_validated": True,
        "fresh_processes_per_backend": 4,
        "fresh_process_output_equality": True,
        "poison_values": [0x5A, 0xA5],
        "poison_output_invariance": True,
        "input_immutability": True,
        "canary_integrity": True,
        "full_write_coverage": True,
        "numerical_probe_counts": rtx["numerical_probe_counts"],
        "hardware_attestations": {
            "rtx": {
                "job_id": rig["job_id"],
                "gpu": rig["gpu"],
                "cuda_toolkit": rig["cuda_toolkit"],
                "fresh_process_count": rig["fresh_process_count"],
                "rig_receipt_sha256": sha256_file(rtx_root / "rig-receipt.json"),
            },
            "mac": {
                "device_name": mac_run["device_name"],
                "fresh_process_count": mac["fresh_process_count"],
            },
        },
        "selected_opcode_contract": {
            "coordinate_arithmetic": "explicit HFMA2(flow, two, pixel) matching selected SM120 driver lowering",
            "coordinate_integer_conversion": "floor/cvt.rmi.s32.f16 before signed clamp",
            "bilinear_arithmetic": "separately rounded FP16 weights and w10 sample product, then explicit fused w00, w01, w11 accumulation chain",
            "implicit_fp_contraction": "disabled; selected driver contractions are explicit fma calls",
            "optimized_shared_staging_bytes": 9216,
            "optimized_shared_remap": "256 threads stage nine u32 each then coalesced nine-u32 global remap",
            "metal_store_width": "nine explicit u32 stores per pixel",
        },
        "source_artifacts": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "external_expected_tensors_consumed": False,
        "address_fields_recorded": False,
        "accepted_frontier": {
            "central_block_rtx_metal_byte_exact": True,
            "dependency_closed_multi_family_fg_group": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    require_no_address_fields(result, "final receipt")
    write_json(Path(arguments.output), result)
    print(f"output_sha256={result['output_sha256']}")
    print("rtx_metal_byte_exact=true")
    print("result=pass")
    return 0


def self_test() -> int:
    inputs = generate_inputs()
    require(tuple(map(len, inputs)) == INPUT_BYTES, "generator sizes")
    probes = numerical_probe_counts(inputs)
    require(probes["negative_fractional_coordinate_count"] > 0, "negative floor probe")
    require(probes["low_clamp_coordinate_count"] > 0, "low clamp probe")
    require(probes["high_clamp_coordinate_count"] > 0, "high clamp probe")
    require(probes["half_contraction_distinguishing_sample_count"] > 0, "contraction probe")
    original = bytes(range(64))
    mutated = bytearray(original)
    mutated[31] ^= 1
    try:
        require_exact_bytes(bytes(mutated), original, "mutation sentinel")
    except VerificationError:
        pass
    else:
        raise VerificationError("output mutation was not rejected")
    try:
        require_no_address_fields({"gpu_address": "0x123456789abcdef0"}, "mutation")
    except VerificationError:
        pass
    else:
        raise VerificationError("address mutation was not rejected")
    print(json.dumps({"schema": "neuralstudio.fg.central-block-self-test.v1", "probes": probes, "result": "pass"}, sort_keys=True))
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", required=True)
    backend.add_argument("--module")
    backend.add_argument("--output", required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--rtx-root", required=True)
    final.add_argument("--mac-root", required=True)
    final.add_argument("--output", required=True)
    subparsers.add_parser("self-test")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.command == "backend":
        return verify_backend(arguments)
    if arguments.command == "final":
        return verify_final(arguments)
    return self_test()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, VerificationError, struct.error, OverflowError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        raise SystemExit(1)
