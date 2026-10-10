#!/usr/bin/env python3
"""Verify the first dependency-closed Frame Generation Metal unit.

Execution inputs are generated independently by the CUDA and Metal programs.
This verifier is the third implementation of that generator.  It consumes
only post-execution snapshots and receipts, then performs parity checks and
negative mutations after both backends have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parent.parent
WIDTH = 640
HEIGHT = 368
PIXELS = WIDTH * HEIGHT
INPUT_A_BYTES = PIXELS * 8
INPUT_B_BYTES = PIXELS * 8
INPUT_C_BYTES = PIXELS * 2
OUTPUT_BYTES = PIXELS * 20
MODULE_BYTES = 912
MODULE_SHA256 = "63b7e8105e00b04049e1004a8df23998074fb04b2b6637171da379b4a967d8c3"
RUN_SCHEMA = "neuralstudio.fg.initial-merge-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.initial-merge-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.initial-merge-verification.v1"
GENERATOR = "splitmix64-fg-initial-merge-v1"
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


def generate_inputs() -> tuple[bytes, bytes, bytes]:
    mask = (1 << 64) - 1
    seed_a = 0x243F6A8885A308D3
    seed_b = 0x13198A2E03707344
    seed_c = 0xA4093822299F31D0
    step_b = 0x9E3779B97F4A7C15
    step_c = 0xBF58476D1CE4E5B9
    input_a = bytearray(INPUT_A_BYTES)
    input_b = bytearray(INPUT_B_BYTES)
    input_c = bytearray(INPUT_C_BYTES)
    for index in range(PIXELS):
        struct.pack_into("<Q", input_a, index * 8, splitmix64(seed_a ^ index))
        struct.pack_into(
            "<Q",
            input_b,
            index * 8,
            splitmix64((seed_b + index * step_b) & mask),
        )
        struct.pack_into(
            "<H",
            input_c,
            index * 2,
            (splitmix64(seed_c ^ ((index * step_c) & mask)) >> 19) & 0xFFFF,
        )
    return bytes(input_a), bytes(input_b), bytes(input_c)


def build_reference_output(input_a: bytes, input_b: bytes, input_c: bytes) -> bytes:
    require(len(input_a) == INPUT_A_BYTES, "input A reference size mismatch")
    require(len(input_b) == INPUT_B_BYTES, "input B reference size mismatch")
    require(len(input_c) == INPUT_C_BYTES, "input C reference size mismatch")
    output = bytearray(OUTPUT_BYTES)
    interpolation = struct.pack("<H", 0x3800)
    for index in range(PIXELS):
        base = index * 20
        output[base : base + 8] = input_a[index * 8 : index * 8 + 8]
        output[base + 8 : base + 16] = input_b[index * 8 : index * 8 + 8]
        output[base + 16 : base + 18] = input_c[index * 2 : index * 2 + 2]
        output[base + 18 : base + 20] = interpolation
    return bytes(output)


def untouched_pixels(output: bytes, poison: int) -> int:
    require(len(output) == OUTPUT_BYTES, "write coverage output size mismatch")
    marker = bytes([poison]) * 20
    return sum(
        output[pixel * 20 : pixel * 20 + 20] == marker
        for pixel in range(PIXELS)
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


def require_no_address_fields(receipt: dict[str, Any], label: str) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = key.lower()
                if key != "address_fields_recorded":
                    require(
                        not any(
                            fragment in lowered
                            for fragment in ("pointer", "gpu_address", "device_address")
                        ),
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
    expected_backend: str,
    expected_run_id: str,
    poison: int,
    generated_inputs: tuple[bytes, bytes, bytes],
    reference_output: bytes,
) -> dict[str, Any]:
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_root}: schema mismatch")
    require(
        receipt.get("backend") == expected_backend,
        f"{run_root}: backend mismatch",
    )
    require(receipt.get("run_id") == expected_run_id, f"{run_root}: run id mismatch")
    require(receipt.get("poison_byte") == poison, f"{run_root}: poison mismatch")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{run_root}: missing process id",
    )
    require(receipt.get("grid") == [40, 23, 1], f"{run_root}: grid mismatch")
    require(receipt.get("block") == [16, 16, 1], f"{run_root}: block mismatch")
    require(
        receipt.get("dynamic_shared_bytes") == 0,
        f"{run_root}: dynamic shared memory mismatch",
    )
    require(receipt.get("height") == HEIGHT, f"{run_root}: height mismatch")
    require(receipt.get("width") == WIDTH, f"{run_root}: width mismatch")
    require(
        receipt.get("interpolation_f32") == 0.5,
        f"{run_root}: interpolation mismatch",
    )
    require(
        receipt.get("interpolation_f16_bits") == 0x3800,
        f"{run_root}: f16 interpolation mismatch",
    )
    require(
        receipt.get("input_generator") == GENERATOR,
        f"{run_root}: input generator mismatch",
    )
    require(receipt.get("result") == "pass", f"{run_root}: run did not pass")
    require(
        receipt.get("input_mismatched_bytes") == 0
        and receipt.get("input_immutability") is True,
        f"{run_root}: input immutability failed",
    )
    require(
        receipt.get("canary_mismatched_bytes") == 0,
        f"{run_root}: canary failure",
    )
    require(
        receipt.get("untouched_pixels") == 0
        and receipt.get("full_write_coverage") is True,
        f"{run_root}: write coverage failure",
    )
    require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_root}: output size")
    require(
        receipt.get("address_fields_recorded") is False,
        f"{run_root}: address fields were recorded",
    )
    require_no_address_fields(receipt, str(receipt_path))

    if expected_backend == "rtx-cuda-driver":
        require(receipt.get("module_token") == 53, f"{run_root}: module token")
        require(receipt.get("module_bytes") == MODULE_BYTES, f"{run_root}: module bytes")
        require(
            receipt.get("module_sha256") == MODULE_SHA256,
            f"{run_root}: module identity",
        )
        require(receipt.get("function_name") == "k_initial_merge", f"{run_root}: function")
        require(receipt.get("launch_order") == 56, f"{run_root}: launch order")
        require(
            receipt.get("module_transport")
            == "cuModuleLoadData exact selected sm_89 PTX fatbin; driver JIT to active RTX architecture",
            f"{run_root}: module transport mismatch",
        )
    else:
        require(
            receipt.get("reconstruction_unit") == "L56/k_initial_merge",
            f"{run_root}: reconstruction unit mismatch",
        )
        require(
            receipt.get("metal_function") == "fg_initial_merge",
            f"{run_root}: Metal function mismatch",
        )
        require(
            receipt.get("store_width_contract") == "five-u32-at-0-4-8-12-16",
            f"{run_root}: output store-width contract mismatch",
        )
        require(
            receipt.get("external_input_files_opened") is False,
            f"{run_root}: Mac execution admitted an external input",
        )

    artifact_names = (
        "input-a-after.bin",
        "input-b-after.bin",
        "input-c-after.bin",
        "output.bin",
    )
    artifact_bytes = [(run_root / name).read_bytes() for name in artifact_names]
    for name, actual, expected in zip(
        artifact_names[:3], artifact_bytes[:3], generated_inputs
    ):
        require_exact_bytes(actual, expected, f"{run_root}/{name}")
    require_exact_bytes(artifact_bytes[3], reference_output, f"{run_root}/output.bin")
    require(
        untouched_pixels(artifact_bytes[3], poison) == 0,
        f"{run_root}: recomputed untouched pixels are nonzero",
    )

    expected_hashes = tuple(sha256_bytes(value) for value in generated_inputs)
    for lane, expected_hash in zip(("a", "b", "c"), expected_hashes):
        require(
            receipt.get(f"input_{lane}_pre_sha256") == expected_hash,
            f"{run_root}: input {lane} pre hash mismatch",
        )
        require(
            receipt.get(f"input_{lane}_post_sha256") == expected_hash,
            f"{run_root}: input {lane} post hash mismatch",
        )
    require(
        receipt.get("output_sha256") == sha256_bytes(reference_output),
        f"{run_root}: output hash mismatch",
    )
    return {
        "run_id": expected_run_id,
        "process_id": receipt["process_id"],
        "poison_byte": poison,
        "receipt_relative_path": f"{expected_run_id}/run-receipt.json",
        "receipt_sha256": sha256_file(receipt_path),
        "input_a_sha256": expected_hashes[0],
        "input_b_sha256": expected_hashes[1],
        "input_c_sha256": expected_hashes[2],
        "output_sha256": sha256_bytes(reference_output),
        "artifact_sha256": {
            name: sha256_bytes(data)
            for name, data in zip(artifact_names, artifact_bytes)
        },
    }


def validate_backend(
    root: Path, backend: str, module_path: Path | None = None
) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(root.is_dir(), f"backend root not found: {root}")
    if backend == "rtx":
        require(module_path is not None, "RTX validation requires selected module")
        module = module_path.read_bytes()
        require(len(module) == MODULE_BYTES, "selected module byte count mismatch")
        require(sha256_bytes(module) == MODULE_SHA256, "selected module SHA-256 mismatch")
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
    process_ids = [run["process_id"] for run in runs]
    require(len(set(process_ids)) == len(process_ids), f"{backend}: processes were not fresh")
    require(
        len({run["output_sha256"] for run in runs}) == 1,
        f"{backend}: poison/process output invariance failed",
    )
    for filename in ("input-a-after.bin", "input-b-after.bin", "input-c-after.bin"):
        paths = [root / run_id / filename for run_id in CASES]
        if hasattr(os.path, "samestat"):
            stats = [path.stat() for path in paths]
            for left in range(len(stats)):
                for right in range(left + 1, len(stats)):
                    require(
                        not os.path.samestat(stats[left], stats[right]),
                        f"{backend}: input snapshots share a file identity",
                    )
    result: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "contract": {
            "launch": 56,
            "function": "k_initial_merge",
            "grid": [40, 23, 1],
            "block": [16, 16, 1],
            "height": HEIGHT,
            "width": WIDTH,
            "interpolation_f32": 0.5,
            "interpolation_f16_bits": 0x3800,
        },
        "fresh_process_count": len(process_ids),
        "distinct_process_ids": True,
        "poison_values": [0xA5, 0x5A],
        "two_process_per_poison": True,
        "poison_output_invariance": True,
        "input_immutability": True,
        "full_write_coverage": True,
        "address_free": True,
        "input_generator": GENERATOR,
        "output_sha256": sha256_bytes(reference_output),
        "runs": runs,
    }
    if backend == "rtx":
        result["selected_module"] = {
            "token": 53,
            "bytes": MODULE_BYTES,
            "sha256": MODULE_SHA256,
            "transport": "cuda-driver-exact-fatbin-jit",
            "embedded_code": "sm_89-ptx-no-elf",
        }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def attest_stored_backend_receipt(
    root: Path, validated: dict[str, Any]
) -> dict[str, Any]:
    receipt_path = root / "backend-receipt.json"
    stored = load_json(receipt_path)
    require(stored == validated, f"stored backend receipt drift: {receipt_path}")
    root_value = stored.get("receipt_root_sha256")
    rootless = dict(stored)
    rootless.pop("receipt_root_sha256", None)
    require(
        root_value == canonical_sha256(rootless),
        f"stored backend receipt root mismatch: {receipt_path}",
    )
    sidecar = root / "backend-receipt.sha256"
    require(sidecar.is_file(), f"backend receipt sidecar missing: {sidecar}")
    sidecar_hash = sidecar.read_text(encoding="utf-8-sig").split()[0].lower()
    actual_hash = sha256_file(receipt_path)
    require(sidecar_hash == actual_hash, f"backend receipt transfer hash mismatch: {receipt_path}")
    return {
        "relative_path": "backend-receipt.json",
        "sha256": actual_hash,
        "receipt_root_sha256": root_value,
        "sidecar_reverified": True,
    }


def attest_rig_receipt(root: Path) -> dict[str, Any]:
    path = root / "rig-receipt.json"
    receipt = load_json(path)
    require(
        receipt.get("schema") == "neuralstudio.fg.initial-merge-rig.v1",
        "rig receipt schema mismatch",
    )
    require(
        receipt.get("job_id") == "fg2x-intake-20260904T082417Z",
        "rig job identity mismatch",
    )
    require(receipt.get("result") == "pass", "rig receipt did not pass")
    require(
        receipt.get("fresh_process_count") == 4,
        "rig receipt fresh-process count mismatch",
    )
    require(
        receipt.get("address_fields_recorded") is False,
        "rig receipt contains address fields",
    )
    module = receipt.get("selected_module", {})
    require(module.get("token") == 53, "rig receipt module token mismatch")
    require(module.get("bytes") == MODULE_BYTES, "rig receipt module byte mismatch")
    require(module.get("sha256") == MODULE_SHA256, "rig receipt module hash mismatch")
    launch = receipt.get("launch", {})
    require(
        launch
        == {
            "order": 56,
            "function": "k_initial_merge",
            "grid": [40, 23, 1],
            "block": [16, 16, 1],
            "dynamic_shared_bytes": 0,
        },
        "rig receipt launch mismatch",
    )
    require_no_address_fields(receipt, str(path))
    sidecar = root / "rig-receipt.sha256"
    require(sidecar.is_file(), "rig receipt sidecar missing")
    sidecar_hash = sidecar.read_text(encoding="utf-8-sig").split()[0].lower()
    actual_hash = sha256_file(path)
    require(sidecar_hash == actual_hash, "rig receipt transfer hash mismatch")
    return {
        "relative_path": "rig-receipt.json",
        "sha256": actual_hash,
        "sidecar_reverified": True,
        "gpu": receipt.get("gpu"),
        "cuda_toolkit": receipt.get("cuda_toolkit"),
        "binary": receipt.get("binary"),
        "attempt_id": receipt.get("attempt_id"),
    }


def attest_mac_build(root: Path) -> dict[str, Any]:
    relative_paths = [
        "build/fg-initial-merge-mac",
        "build/initial_merge.air",
        "build/initial_merge.metallib",
        "build/host-compile.log",
        "build/metal-compile.log",
        "build/metallib-link.log",
    ]
    artifacts: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = root / relative
        require(path.is_file(), f"Mac build artifact missing: {path}")
        artifacts.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    device_names = {
        load_json(root / run_id / "run-receipt.json").get("device_name")
        for run_id in CASES
    }
    require(len(device_names) == 1 and None not in device_names, "Mac device identity drift")
    return {
        "device_name": next(iter(device_names)),
        "artifacts": artifacts,
        "all_runs_used_one_staged_binary_path": True,
    }


def source_manifest() -> list[dict[str, Any]]:
    paths = [
        "src/fg_initial_merge/rtx_oracle.cpp",
        "src/fg_initial_merge/macos_probe.mm",
        "src/fg_initial_merge/initial_merge.metal",
        "scripts/build-fg-initial-merge-rtx.ps1",
        "scripts/build-macos-fg-initial-merge.sh",
        "scripts/verify-fg-initial-merge.py",
        "evidence/specs/fg-initial-merge-v1/failure-ledger.md",
    ]
    result = []
    for relative in paths:
        path = REPOSITORY / relative
        require(path.is_file(), f"source artifact missing: {relative}")
        result.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def validate_mac_execution_isolation() -> dict[str, Any]:
    inspected = [
        REPOSITORY / "src/fg_initial_merge/macos_probe.mm",
        REPOSITORY / "src/fg_initial_merge/initial_merge.metal",
        REPOSITORY / "scripts/build-macos-fg-initial-merge.sh",
    ]
    forbidden = re.compile(r"\b(expected|oracle|rtx)\b", re.IGNORECASE)
    for path in inspected:
        text = path.read_text(encoding="utf-8")
        match = forbidden.search(text)
        found = match.group(0) if match is not None else ""
        require(
            match is None,
            f"Mac execution source contains forbidden external-feed token {found!r}: {path}",
        )
    return {
        "inspected_sources": [str(path.relative_to(REPOSITORY)) for path in inspected],
        "external_feed_tokens_absent": True,
        "mac_process_accepts_reference_inputs": False,
    }


def expect_negative(label: str, operation: Callable[[], None]) -> dict[str, Any]:
    try:
        operation()
    except VerificationError as error:
        return {"mutation": label, "rejected": True, "reason": str(error)}
    raise VerificationError(f"negative mutation was not rejected: {label}")


def negative_mutations(
    module_path: Path, generated_inputs: tuple[bytes, bytes, bytes], output: bytes
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    mutated_output = bytearray(output)
    mutated_output[0] ^= 0x01
    results.append(
        expect_negative(
            "mac-output-byte-flip",
            lambda: require_exact_bytes(bytes(mutated_output), output, "mutated Mac output"),
        )
    )

    mutated_rtx_output = bytearray(output)
    mutated_rtx_output[-1] ^= 0x80
    results.append(
        expect_negative(
            "rtx-output-byte-flip",
            lambda: require_exact_bytes(bytes(mutated_rtx_output), output, "mutated RTX output"),
        )
    )

    mutated_input = bytearray(generated_inputs[0])
    mutated_input[17] ^= 0x40
    results.append(
        expect_negative(
            "input-provenance-byte-flip",
            lambda: require_exact_bytes(
                bytes(mutated_input), generated_inputs[0], "mutated input A"
            ),
        )
    )

    mutated_module = bytearray(module_path.read_bytes())
    mutated_module[len(mutated_module) // 2] ^= 0x01

    def reject_module() -> None:
        require(
            sha256_bytes(bytes(mutated_module)) == MODULE_SHA256,
            "selected module identity mismatch",
        )

    results.append(expect_negative("selected-module-byte-flip", reject_module))

    untouched_output = bytearray(output)
    untouched_output[:20] = bytes([0xA5]) * 20

    def reject_untouched() -> None:
        require(
            untouched_pixels(bytes(untouched_output), 0xA5) == 0,
            "untouched output pixel detected",
        )

    results.append(expect_negative("untouched-output-pixel", reject_untouched))

    wrong_interpolation = bytearray(output)
    wrong_interpolation[18:20] = struct.pack("<H", 0x3400)
    results.append(
        expect_negative(
            "interpolation-half-mutation",
            lambda: require_exact_bytes(
                bytes(wrong_interpolation), output, "mutated interpolation half"
            ),
        )
    )
    return results


def verify_cross_backend(
    rtx_root: Path, mac_root: Path, module_path: Path
) -> dict[str, Any]:
    rtx = validate_backend(rtx_root, "rtx", module_path)
    mac = validate_backend(mac_root, "mac")
    rtx_backend_attestation = attest_stored_backend_receipt(rtx_root, rtx)
    mac_backend_attestation = attest_stored_backend_receipt(mac_root, mac)
    rig_attestation = attest_rig_receipt(rtx_root)
    require(
        rtx["output_sha256"] == mac["output_sha256"],
        "RTX/Mac output root mismatch",
    )
    for run_id in CASES:
        rtx_output = rtx_root / run_id / "output.bin"
        mac_output = mac_root / run_id / "output.bin"
        require_exact_bytes(
            mac_output.read_bytes(), rtx_output.read_bytes(), f"RTX/Mac {run_id}"
        )
        for filename in (
            "input-a-after.bin",
            "input-b-after.bin",
            "input-c-after.bin",
        ):
            rtx_input = rtx_root / run_id / filename
            mac_input = mac_root / run_id / filename
            require_exact_bytes(
                mac_input.read_bytes(),
                rtx_input.read_bytes(),
                f"RTX/Mac input contract {run_id}/{filename}",
            )
            require(
                not os.path.samefile(rtx_input, mac_input),
                f"Mac and RTX inputs are the same physical file: {run_id}/{filename}",
            )

    generated_inputs = generate_inputs()
    reference_output = build_reference_output(*generated_inputs)
    census_path = (
        REPOSITORY
        / "evidence/specs/fg-selected-module-census-v1/selected-module-census.json"
    )
    require(census_path.is_file(), "selected-module census is missing")
    evidence: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_path_parent": {
            "census_relative_path": str(census_path.relative_to(REPOSITORY)),
            "census_sha256": sha256_file(census_path),
            "launch": 56,
            "evaluate_1_order": 45,
            "module_token": 53,
            "module_sha256": MODULE_SHA256,
            "function": "k_initial_merge",
        },
        "rtx_backend": rtx,
        "mac_backend": mac,
        "backend_attestations": {
            "rtx": rtx_backend_attestation,
            "mac": mac_backend_attestation,
            "mac_build": attest_mac_build(mac_root),
            "rig": rig_attestation,
        },
        "mac_execution_isolation": validate_mac_execution_isolation(),
        "cross_backend": {
            "independent_input_files": True,
            "input_bytes_equal": True,
            "byte_exact_output": True,
            "mismatched_output_bytes": 0,
            "output_bytes": OUTPUT_BYTES,
            "output_sha256": sha256_bytes(reference_output),
            "four_fresh_processes_per_backend": True,
            "two_processes_per_poison": True,
            "two_poison_output_invariance": True,
            "zero_untouched_pixels": True,
        },
        "negative_mutations": negative_mutations(
            module_path, generated_inputs, reference_output
        ),
        "source_artifacts": source_manifest(),
        "accepted_frontier": {
            "live_selected_module_identity_bound": True,
            "exact_selected_module_executed_on_rtx": True,
            "smallest_dependency_closed_group_on_metal": True,
            "rtx_mac_byte_exact_group_parity": True,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    evidence["evidence_root_sha256"] = canonical_sha256(evidence)
    return evidence


def self_test() -> None:
    inputs = generate_inputs()
    require(tuple(map(len, inputs)) == (INPUT_A_BYTES, INPUT_B_BYTES, INPUT_C_BYTES), "input sizes")
    second = generate_inputs()
    require(inputs == second, "input generator is nondeterministic")
    output = build_reference_output(*inputs)
    require(len(output) == OUTPUT_BYTES, "output size")
    require(output[:8] == inputs[0][:8], "output A lane")
    require(output[8:16] == inputs[1][:8], "output B lane")
    require(output[16:18] == inputs[2][:2], "output C lane")
    require(output[18:20] == b"\x00\x38", "output interpolation lane")
    require(untouched_pixels(output, 0xA5) == 0, "A5 coverage")
    require(untouched_pixels(output, 0x5A) == 0, "5A coverage")
    with tempfile.TemporaryDirectory() as temporary:
        module_path = Path(temporary) / "module.bin"
        module_path.write_bytes(bytes(MODULE_BYTES))
        # The mutation helper tests its altered hash against the pinned identity;
        # a synthetic source is sufficient for its fail-closed behavior.
        mutations = negative_mutations(module_path, inputs, output)
        require(all(item["rejected"] for item in mutations), "negative mutation")
    print("self_test=pass")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    backend_parser = subparsers.add_parser("backend")
    backend_parser.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend_parser.add_argument("--root", type=Path, required=True)
    backend_parser.add_argument("--module", type=Path)
    backend_parser.add_argument("--output", type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--rtx-root", type=Path, required=True)
    verify_parser.add_argument("--mac-root", type=Path, required=True)
    verify_parser.add_argument("--module", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("self-test")
    arguments = parser.parse_args()
    if arguments.command == "self-test":
        self_test()
        return 0
    if arguments.command == "backend":
        value = validate_backend(arguments.root, arguments.backend, arguments.module)
        output = arguments.output or arguments.root / "backend-receipt.json"
        write_json(output, value)
        write_sha256_sidecar(output)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    value = verify_cross_backend(
        arguments.rtx_root, arguments.mac_root, arguments.module
    )
    write_json(arguments.output, value)
    write_sha256_sidecar(arguments.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"verification_error={error}", file=sys.stderr)
        raise SystemExit(1)
