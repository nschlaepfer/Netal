#!/usr/bin/env python3
"""Verify exact RTX/Metal parity for selected FG elementwise launch 33."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parent.parent
WIDTH = 10
HEIGHT = 6
CHANNELS = 256
VALUES = WIDTH * HEIGHT * CHANNELS
TENSOR_BYTES = VALUES * 2
MODULE_BYTES = 2152
MODULE_SHA256 = "1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26"
RAW_PARAMETER_SHA256 = "0bb4c1c5e582ea1a7386318d294df1511fdc2e81287b7943f23f9a531a1690bc"
NORMALIZED_PARAMETER_SHA256 = "60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80"
SCALAR_PARAMETER_SHA256 = "0f2b3cefe270ff83b105bc583212a894fdaf8671acab257be993fcedab912822"
SCALARS = (10, 6, 256, 10, 6, 10, 6)
INPUT_A_SHA256 = "1a43703cd7415d819fc5b65c96b01321065c51f4a5eae6cf2a9872f05443f66b"
INPUT_B_SHA256 = "c52ae8cad86aafdc49e13bbd59da41248ed7f924c6145b1dabd760af72eb7bfd"
OUTPUT_SHA256 = "5690f94bfd5461417bf2e23bf25fb4cdbb2f2908344fa84fc52eac2557c43c0a"
RUN_SCHEMA = "neuralstudio.fg.elementwise-add-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.elementwise-add-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.elementwise-add-verification.v1"
GENERATOR = "splitmix64-fg-elementwise-add-v1"
CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}


class VerificationError(RuntimeError):
    """An evidence or numerical gate failed closed."""


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


def write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def input_half_bits(index: int, input_b: bool) -> int:
    special_a = (0x3C00, 0x3C01, 0xBC00, 0xBC01, 0x0400, 0x8400, 0x3555, 0xB555)
    special_b = (0x1000, 0x1000, 0x9000, 0x9000, 0x0400, 0x8400, 0x3000, 0xB000)
    if index < len(special_a):
        return special_b[index] if input_b else special_a[index]
    mask = (1 << 64) - 1
    seed = 0x13198A2E03707344 if input_b else 0x243F6A8885A308D3
    step = 0xBF58476D1CE4E5B9 if input_b else 0x9E3779B97F4A7C15
    value = splitmix64((seed + index * step) & mask)
    sign_value = splitmix64(
        (0xA4093822299F31D0 + index * 0x94D049BB133111EB) & mask
    )
    sign = ((sign_value >> 63) & 1) << 15
    exponent = 5 + ((value >> 10) % 19)
    return sign | (exponent << 10) | (value & 0x03FF)


def generate_inputs() -> tuple[bytes, bytes]:
    input_a = bytearray(TENSOR_BYTES)
    input_b = bytearray(TENSOR_BYTES)
    for index in range(VALUES):
        struct.pack_into("<H", input_a, index * 2, input_half_bits(index, False))
        struct.pack_into("<H", input_b, index * 2, input_half_bits(index, True))
    generated = (bytes(input_a), bytes(input_b))
    require(sha256_bytes(generated[0]) == INPUT_A_SHA256, "input A generator drift")
    require(sha256_bytes(generated[1]) == INPUT_B_SHA256, "input B generator drift")
    return generated


def build_reference_output(input_a: bytes, input_b: bytes) -> bytes:
    require(len(input_a) == TENSOR_BYTES, "input A reference size mismatch")
    require(len(input_b) == TENSOR_BYTES, "input B reference size mismatch")
    output = bytearray(TENSOR_BYTES)
    for index in range(VALUES):
        value_a = struct.unpack_from("<e", input_a, index * 2)[0]
        value_b = struct.unpack_from("<e", input_b, index * 2)[0]
        struct.pack_into("<e", output, index * 2, value_a + value_b)
    reference = bytes(output)
    require(sha256_bytes(reference) == OUTPUT_SHA256, "reference output drift")
    return reference


def untouched_half_words(output: bytes, poison: int) -> int:
    require(len(output) == TENSOR_BYTES, "write coverage output size mismatch")
    marker = poison | (poison << 8)
    return sum(
        struct.unpack_from("<H", output, index * 2)[0] == marker
        for index in range(VALUES)
    )


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


def require_no_address_fields(value: Any, label: str) -> None:
    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = key.lower()
                if key != "address_fields_recorded":
                    require(
                        not any(
                            fragment in lowered
                            for fragment in (
                                "gpu_address",
                                "device_address",
                                "raw_pointer",
                                "pointer_value",
                                "gpu_va",
                            )
                        ),
                        f"{label} contains address-bearing field {path}.{key}",
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
        elif isinstance(node, str):
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                f"{label} contains a raw address-like value at {path}",
            )

    visit(value, label)


def validate_module_bytes(module: bytes) -> None:
    require(len(module) == MODULE_BYTES, "selected module size mismatch")
    require(sha256_bytes(module) == MODULE_SHA256, "selected module identity mismatch")


def validate_parameter_bytes(parameters: bytes) -> dict[str, Any]:
    require(len(parameters) == 52, "live parameter size mismatch")
    require(
        sha256_bytes(parameters) == RAW_PARAMETER_SHA256,
        "captured live parameter identity mismatch",
    )
    scalars = struct.unpack_from("<7I", parameters, 24)
    require(scalars == SCALARS, "live scalar ABI mismatch")
    normalized = bytes(24) + parameters[24:]
    require(
        sha256_bytes(normalized) == NORMALIZED_PARAMETER_SHA256,
        "address-free parameter normalization mismatch",
    )
    require(
        sha256_bytes(parameters[24:]) == SCALAR_PARAMETER_SHA256,
        "scalar parameter hash mismatch",
    )
    return {
        "captured_raw_sha256": RAW_PARAMETER_SHA256,
        "raw_hash_is_trace_specific_not_portable_identity": True,
        "address_free_normalized_sha256": NORMALIZED_PARAMETER_SHA256,
        "scalar_bytes_sha256": SCALAR_PARAMETER_SHA256,
        "pointer_roles": ["input_a", "input_b", "output"],
        "scalars_u32": list(SCALARS),
        "bytes": 52,
    }


def validate_run(
    run_root: Path,
    expected_backend: str,
    expected_run_id: str,
    poison: int,
    generated_inputs: tuple[bytes, bytes],
    reference_output: bytes,
) -> dict[str, Any]:
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_root}: schema mismatch")
    require(receipt.get("backend") == expected_backend, f"{run_root}: backend mismatch")
    require(receipt.get("run_id") == expected_run_id, f"{run_root}: run id mismatch")
    require(receipt.get("poison_byte") == poison, f"{run_root}: poison mismatch")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{run_root}: missing process id",
    )
    require(receipt.get("grid") == [3, 2, 1], f"{run_root}: grid mismatch")
    require(receipt.get("block") == [4, 4, 8], f"{run_root}: block mismatch")
    require(receipt.get("dynamic_shared_bytes") == 0, f"{run_root}: shared mismatch")
    require(receipt.get("shape_nhwc") == [1, 6, 10, 256], f"{run_root}: shape mismatch")
    require(receipt.get("parameter_scalars_u32") == list(SCALARS), f"{run_root}: scalar mismatch")
    require(receipt.get("input_generator") == GENERATOR, f"{run_root}: generator mismatch")
    require(receipt.get("tensor_bytes") == TENSOR_BYTES, f"{run_root}: byte size mismatch")
    require(receipt.get("result") == "pass", f"{run_root}: run did not pass")
    require(
        receipt.get("input_mismatched_bytes") == 0
        and receipt.get("input_immutability") is True,
        f"{run_root}: input immutability failed",
    )
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_root}: canary failed")
    require(
        receipt.get("untouched_half_words") == 0
        and receipt.get("full_write_coverage") is True,
        f"{run_root}: write coverage failed",
    )
    require(
        receipt.get("external_tensor_input_files_opened") is False,
        f"{run_root}: external tensor input contamination",
    )
    require(receipt.get("address_fields_recorded") is False, f"{run_root}: address flag")
    require_no_address_fields(receipt, str(receipt_path))

    if expected_backend == "rtx-cuda-driver":
        require(receipt.get("module_token") == 37, f"{run_root}: module token mismatch")
        require(receipt.get("module_bytes") == MODULE_BYTES, f"{run_root}: module size mismatch")
        require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_root}: module hash mismatch")
        require(receipt.get("function_name") == "k_element_wise", f"{run_root}: function mismatch")
        require(receipt.get("launch_token") == 33, f"{run_root}: launch mismatch")
        require(receipt.get("evaluate_1_phase_order") == 22, f"{run_root}: phase order mismatch")
        require(receipt.get("global_launch_order") == 31, f"{run_root}: global order mismatch")
    else:
        require(
            receipt.get("selected_module_sha256") == MODULE_SHA256,
            f"{run_root}: Metal selected-module binding mismatch",
        )
        require(
            receipt.get("metal_function") == "fg_elementwise_add_256",
            f"{run_root}: Metal function mismatch",
        )

    actual_a = (run_root / "input-a-after.bin").read_bytes()
    actual_b = (run_root / "input-b-after.bin").read_bytes()
    output = (run_root / "output.bin").read_bytes()
    require_exact_bytes(actual_a, generated_inputs[0], f"{run_root} input A")
    require_exact_bytes(actual_b, generated_inputs[1], f"{run_root} input B")
    require_exact_bytes(output, reference_output, f"{run_root} output")
    require(untouched_half_words(output, poison) == 0, f"{run_root}: untouched output")
    require(receipt.get("input_a_pre_sha256") == INPUT_A_SHA256, f"{run_root}: input A pre hash")
    require(receipt.get("input_a_post_sha256") == INPUT_A_SHA256, f"{run_root}: input A post hash")
    require(receipt.get("input_b_pre_sha256") == INPUT_B_SHA256, f"{run_root}: input B pre hash")
    require(receipt.get("input_b_post_sha256") == INPUT_B_SHA256, f"{run_root}: input B post hash")
    require(receipt.get("output_sha256") == OUTPUT_SHA256, f"{run_root}: output hash")
    return {
        "run_id": expected_run_id,
        "poison_byte": poison,
        "process_id": receipt["process_id"],
        "receipt_relative_path": f"{expected_run_id}/run-receipt.json",
        "receipt_sha256": sha256_file(receipt_path),
        "artifacts": {
            "input-a-after.bin": sha256_file(run_root / "input-a-after.bin"),
            "input-b-after.bin": sha256_file(run_root / "input-b-after.bin"),
            "output.bin": sha256_file(run_root / "output.bin"),
        },
    }


def build_backend_receipt(backend: str, root: Path) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    generated_inputs = generate_inputs()
    reference_output = build_reference_output(*generated_inputs)
    runs = [
        validate_run(
            root / run_id,
            expected_backend,
            run_id,
            poison,
            generated_inputs,
            reference_output,
        )
        for run_id, poison in CASES.items()
    ]
    process_ids = {run["process_id"] for run in runs}
    require(len(process_ids) == 4, f"{backend}: fresh-process gate failed")
    output_hashes = {run["artifacts"]["output.bin"] for run in runs}
    require(output_hashes == {OUTPUT_SHA256}, f"{backend}: output invariance failed")
    receipt: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "selected_unit": "launch-000033/k_element_wise/module-000037",
        "contract": {
            "module_token": 37,
            "module_sha256": MODULE_SHA256,
            "launch_token": 33,
            "evaluate_1_phase_order": 22,
            "global_launch_order": 31,
            "function": "k_element_wise",
            "grid": [3, 2, 1],
            "block": [4, 4, 8],
            "shape_nhwc": [1, 6, 10, 256],
            "scalars_u32": list(SCALARS),
            "operation": "binary16 add.rn",
        },
        "input_generator": GENERATOR,
        "input_a_sha256": INPUT_A_SHA256,
        "input_b_sha256": INPUT_B_SHA256,
        "output_sha256": OUTPUT_SHA256,
        "fresh_process_count": 4,
        "two_processes_per_poison": True,
        "poison_values": [0xA5, 0x5A],
        "poison_output_invariance": True,
        "input_immutability": True,
        "full_write_coverage": True,
        "address_free": True,
        "runs": runs,
        "result": "pass",
    }
    root_value = canonical_sha256(receipt)
    receipt["receipt_root_sha256"] = root_value
    require_no_address_fields(receipt, f"{backend} backend receipt")
    return receipt


def expect_rejection(name: str, operation: Callable[[], None]) -> dict[str, Any]:
    try:
        operation()
    except VerificationError as error:
        return {"mutation": name, "rejected": True, "reason": str(error)}
    raise VerificationError(f"negative mutation was accepted: {name}")


def negative_audit(
    generated_inputs: tuple[bytes, bytes], reference_output: bytes,
    module: bytes, parameters: bytes,
) -> list[dict[str, Any]]:
    output_flip = bytearray(reference_output)
    output_flip[137] ^= 0x01
    input_flip = bytearray(generated_inputs[0])
    input_flip[41] ^= 0x80
    module_flip = bytearray(module)
    module_flip[17] ^= 0x01
    scalar_flip = bytearray(parameters)
    scalar_flip[24] ^= 0x01
    poisoned = bytearray(reference_output)
    poisoned[0:2] = b"\xA5\xA5"
    return [
        expect_rejection(
            "rtx-or-mac-output-byte-flip",
            lambda: require_exact_bytes(bytes(output_flip), reference_output, "mutated output"),
        ),
        expect_rejection(
            "input-provenance-byte-flip",
            lambda: require_exact_bytes(bytes(input_flip), generated_inputs[0], "mutated input A"),
        ),
        expect_rejection(
            "selected-module-byte-flip",
            lambda: validate_module_bytes(bytes(module_flip)),
        ),
        expect_rejection(
            "live-scalar-byte-flip",
            lambda: validate_parameter_bytes(bytes(scalar_flip)),
        ),
        expect_rejection(
            "untouched-output-half-word",
            lambda: require(
                untouched_half_words(bytes(poisoned), 0xA5) == 0,
                "untouched output half word detected",
            ),
        ),
        expect_rejection(
            "raw-address-in-receipt",
            lambda: require_no_address_fields(
                {"note": "volatile value 0x0000000123456789"}, "mutated receipt"
            ),
        ),
    ]


def source_artifacts() -> list[dict[str, Any]]:
    paths = [
        REPOSITORY / "src/fg_elementwise/rtx_oracle.cpp",
        REPOSITORY / "src/fg_elementwise/macos_probe.mm",
        REPOSITORY / "src/fg_elementwise/elementwise_add.metal",
        REPOSITORY / "scripts/build-fg-elementwise-rtx.ps1",
        REPOSITORY / "scripts/build-macos-fg-elementwise.sh",
        Path(__file__).resolve(),
    ]
    result = []
    for path in paths:
        require(path.is_file(), f"missing source artifact: {path}")
        result.append(
            {
                "relative_path": path.relative_to(REPOSITORY).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def build_final_receipt(
    rtx_root: Path,
    mac_root: Path,
    module_path: Path,
    parameter_path: Path,
) -> dict[str, Any]:
    module = module_path.read_bytes()
    parameters = parameter_path.read_bytes()
    validate_module_bytes(module)
    parameter_contract = validate_parameter_bytes(parameters)
    rtx = build_backend_receipt("rtx", rtx_root)
    mac = build_backend_receipt("mac", mac_root)
    require(rtx["output_sha256"] == mac["output_sha256"], "cross-backend output mismatch")
    for root, rebuilt, label in (
        (rtx_root, rtx, "rtx"),
        (mac_root, mac, "mac"),
    ):
        recorded_path = root / "backend-receipt.json"
        recorded = load_json(recorded_path)
        require(recorded == rebuilt, f"{label}: recorded backend receipt drift")
        sidecar_path = root / "backend-receipt.sha256"
        require(sidecar_path.is_file(), f"{label}: backend sidecar missing")
        sidecar_hash = sidecar_path.read_text(encoding="utf-8-sig").split()[0]
        require(
            sidecar_hash == sha256_file(recorded_path),
            f"{label}: backend sidecar mismatch",
        )
    generated_inputs = generate_inputs()
    reference_output = build_reference_output(*generated_inputs)

    rig_receipt_path = rtx_root / "rig-receipt.json"
    rig_receipt = load_json(rig_receipt_path)
    require(
        rig_receipt.get("schema") == "neuralstudio.fg.elementwise-add-rig.v1",
        "RTX rig receipt schema mismatch",
    )
    require(rig_receipt.get("result") == "pass", "RTX rig receipt did not pass")
    require(rig_receipt.get("fresh_process_count") == 4, "RTX rig process gate")
    require(
        rig_receipt.get("address_fields_recorded") is False,
        "RTX rig address flag",
    )
    require(
        rig_receipt.get("selected_module", {}).get("sha256") == MODULE_SHA256,
        "RTX rig module binding mismatch",
    )
    rig_sidecar = (rtx_root / "rig-receipt.sha256").read_text(
        encoding="utf-8-sig"
    ).split()[0]
    require(
        rig_sidecar == sha256_file(rig_receipt_path),
        "RTX rig receipt sidecar mismatch",
    )
    require_no_address_fields(rig_receipt, "RTX rig receipt")

    mac_run_receipts = [
        load_json(mac_root / run_id / "run-receipt.json") for run_id in CASES
    ]
    mac_devices = {item.get("device_name") for item in mac_run_receipts}
    mac_metallibs = {item.get("metallib_sha256") for item in mac_run_receipts}
    require(len(mac_devices) == 1 and None not in mac_devices, "Mac device drift")
    require(len(mac_metallibs) == 1 and None not in mac_metallibs, "Mac metallib drift")
    mac_binary_path = mac_root / "build/fg-elementwise-mac"
    mac_metallib_path = mac_root / "build/elementwise_add.metallib"
    require(mac_binary_path.is_file(), "Mac executable missing")
    require(mac_metallib_path.is_file(), "Mac metallib missing")

    def backend_attestation(
        backend: dict[str, Any], root: Path
    ) -> dict[str, Any]:
        return {
            "backend": backend["backend"],
            "selected_unit": backend["selected_unit"],
            "contract": backend["contract"],
            "input_a_sha256": backend["input_a_sha256"],
            "input_b_sha256": backend["input_b_sha256"],
            "output_sha256": backend["output_sha256"],
            "fresh_process_count": backend["fresh_process_count"],
            "two_processes_per_poison": backend["two_processes_per_poison"],
            "poison_output_invariance": backend["poison_output_invariance"],
            "input_immutability": backend["input_immutability"],
            "full_write_coverage": backend["full_write_coverage"],
            "address_free": backend["address_free"],
            "backend_receipt_sha256": sha256_file(root / "backend-receipt.json"),
            "backend_receipt_root_sha256": backend["receipt_root_sha256"],
            "result": backend["result"],
        }

    receipt: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_unit": {
            "phase": "evaluate_1",
            "launch_token": 33,
            "phase_order": 22,
            "global_order": 31,
            "module_token": 37,
            "module_bytes": MODULE_BYTES,
            "module_sha256": MODULE_SHA256,
            "function": "k_element_wise",
            "grid": [3, 2, 1],
            "block": [4, 4, 8],
            "dynamic_shared_bytes": 0,
            "shape_nhwc": [1, 6, 10, 256],
            "operation": "binary16 add.rn",
            "parameter_contract": parameter_contract,
        },
        "cross_backend": {
            "byte_exact_output": True,
            "mismatched_output_bytes": 0,
            "output_bytes": TENSOR_BYTES,
            "output_sha256": OUTPUT_SHA256,
            "input_a_sha256": INPUT_A_SHA256,
            "input_b_sha256": INPUT_B_SHA256,
            "independent_input_generation": True,
            "external_expected_or_intermediate_bytes_fed_to_execution": False,
            "four_fresh_processes_per_backend": True,
            "two_processes_per_poison": True,
            "two_poison_output_invariance": True,
            "zero_untouched_half_words": True,
            "input_immutability": True,
            "canaries_intact": True,
        },
        "execution_attestations": {
            "rtx": {
                "job_id": rig_receipt["job_id"],
                "attempt_id": rig_receipt["attempt_id"],
                "source_capture_job": rig_receipt["source_capture_job"],
                "started_utc": rig_receipt["started_utc"],
                "ended_utc": rig_receipt["ended_utc"],
                "gpu": rig_receipt["gpu"],
                "cuda_toolkit": rig_receipt["cuda_toolkit"],
                "visual_studio": rig_receipt["visual_studio"],
                "binary": rig_receipt["binary"],
                "rig_receipt_sha256": sha256_file(rig_receipt_path),
                "exit_code": 0,
                "result": "pass",
            },
            "mac": {
                "device_name": next(iter(mac_devices)),
                "metallib_sha256": next(iter(mac_metallibs)),
                "binary_sha256": sha256_file(mac_binary_path),
                "binary_bytes": mac_binary_path.stat().st_size,
                "exit_code": 0,
                "result": "pass",
            },
        },
        "rtx_backend": backend_attestation(rtx, rtx_root),
        "mac_backend": backend_attestation(mac, mac_root),
        "source_artifacts": source_artifacts(),
        "negative_mutations": negative_audit(
            generated_inputs, reference_output, module, parameters
        ),
        "family_map": [
            {
                "module_token": 37,
                "module_sha256": MODULE_SHA256,
                "launch_tokens": [33],
                "shapes_nhwc": [[1, 6, 10, 256]],
                "channel_vector_shift": 5,
                "accepted_exact": True,
            },
            {
                "module_token": 41,
                "module_sha256": "a717783407a1cea5e3d64b4b23edb451f2b791d8cb7980b374433b9f6ded17e1",
                "launch_tokens": [37, 41],
                "shapes_nhwc": [[1, 12, 20, 128], [1, 24, 40, 128]],
                "channel_vector_shift": 4,
                "accepted_exact": False,
            },
            {
                "module_token": 46,
                "module_sha256": "c733968c9cdca9ba170a9a453d2a831537f467de10e089325aabf8e8ed1cd58f",
                "launch_tokens": [45],
                "shapes_nhwc": [[1, 48, 80, 64]],
                "channel_vector_shift": 3,
                "accepted_exact": False,
            },
            {
                "module_token": 50,
                "module_sha256": "dc36c503f9fca1cb42944cab551c268b2d738912927f4aab9db4d53ae66766b6",
                "launch_tokens": [49],
                "shapes_nhwc": [[1, 96, 160, 32]],
                "channel_vector_shift": 2,
                "accepted_exact": False,
            },
        ],
        "accepted_frontier": {
            "exact_selected_module_executed_on_rtx": True,
            "one_elementwise_variant_on_metal": True,
            "rtx_mac_byte_exact_unit_parity": True,
            "all_elementwise_variants": False,
            "dependency_closed_fg_group": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    receipt["evidence_root_sha256"] = canonical_sha256(receipt)
    require_no_address_fields(receipt, "final receipt")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--rtx-root", type=Path, required=True)
    final.add_argument("--mac-root", type=Path, required=True)
    final.add_argument("--module", type=Path, required=True)
    final.add_argument("--live-parameters", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "backend":
            receipt = build_backend_receipt(args.backend, args.root)
        else:
            receipt = build_final_receipt(
                args.rtx_root, args.mac_root, args.module, args.live_parameters
            )
        write_json(args.output, receipt)
        write_sha256_sidecar(args.output)
        print(f"receipt={args.output}")
        print(f"sha256={sha256_file(args.output)}")
        print(f"evidence_root={receipt.get('evidence_root_sha256', receipt.get('receipt_root_sha256'))}")
        print("result=pass")
        return 0
    except (OSError, VerificationError, struct.error, OverflowError) as error:
        print(f"result=fail\nerror={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
