#!/usr/bin/env python3
"""Verify the five live-selected Frame Generation pooling launches.

The CUDA and Metal executables independently generate their tensor inputs.
This third implementation reconstructs the finite FP16 contract, checks every
post-execution artifact, and rejects deliberate evidence mutations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parent.parent
MASK64 = (1 << 64) - 1
MODULE_BYTES = 2032
MODULE_SHA256 = "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977"
RUN_SCHEMA = "neuralstudio.fg.pooling-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.pooling-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.pooling-verification.v1"
GENERATOR = "splitmix64-fg-pooling-finite-v1"
CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}
SHAPES = (
    {
        "input_width": 160,
        "input_height": 96,
        "output_width": 80,
        "output_height": 48,
        "channels": 32,
        "grid": [20, 12, 1],
        "block": [4, 4, 16],
        "live_launch_token": 21,
        "trace_order": 19,
    },
    {
        "input_width": 80,
        "input_height": 48,
        "output_width": 40,
        "output_height": 24,
        "channels": 64,
        "grid": [10, 6, 1],
        "block": [4, 4, 16],
        "live_launch_token": 23,
        "trace_order": 21,
    },
    {
        "input_width": 40,
        "input_height": 24,
        "output_width": 20,
        "output_height": 12,
        "channels": 128,
        "grid": [5, 3, 1],
        "block": [4, 4, 16],
        "live_launch_token": 25,
        "trace_order": 23,
    },
    {
        "input_width": 20,
        "input_height": 12,
        "output_width": 10,
        "output_height": 6,
        "channels": 128,
        "grid": [3, 2, 1],
        "block": [4, 4, 16],
        "live_launch_token": 27,
        "trace_order": 25,
    },
    {
        "input_width": 10,
        "input_height": 6,
        "output_width": 5,
        "output_height": 3,
        "channels": 256,
        "grid": [2, 1, 1],
        "block": [4, 4, 16],
        "live_launch_token": 29,
        "trace_order": 27,
    },
)


class VerificationError(RuntimeError):
    """An evidence or exactness gate failed closed."""


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
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def finite_normal(value: int) -> int:
    sign = (value >> 48) & 0x8000
    exponent = 1 + ((value >> 52) % 30)
    bits = sign | (exponent << 10) | (value & 0x03FF)
    return 0x3C00 if bits in (0x5A5A, 0xA5A5) else bits


def input_half_bits(
    shape_index: int, shape: dict[str, Any], y: int, x: int, channel: int
) -> int:
    cell = (
        ((y // 2) * shape["output_width"] + x // 2) * shape["channels"]
        + channel
    )
    lane = (y & 1) * 2 + (x & 1)
    base = splitmix64(
        0xD1B54A32D192ED03
        ^ ((shape_index + 1) * 0x9E3779B97F4A7C15)
        ^ (cell * 0x94D049BB133111EB)
    )
    mode = base & 31
    if mode == 0:
        return 0x0000
    if mode == 1:
        return 0x8000
    if mode == 2:
        return 0x8000 if lane & 1 else 0x0000
    if mode == 3:
        return finite_normal(base)
    if mode == 4:
        return (0xBC00, 0x0000, 0x3C00, 0x4000)[lane]
    if mode == 5:
        return (0x0001, 0x8001, 0x03FF, 0x83FF)[lane]
    if mode == 6:
        return (0x0400, 0x8400, 0x7BFF, 0xFBFF)[lane]
    return finite_normal(
        splitmix64(base ^ ((lane + 1) * 0xBF58476D1CE4E5B9))
    )


def generate_input(shape_index: int, shape: dict[str, Any]) -> bytes:
    values = (
        shape["input_width"] * shape["input_height"] * shape["channels"]
    )
    result = bytearray(values * 2)
    for y in range(shape["input_height"]):
        for x in range(shape["input_width"]):
            for channel in range(shape["channels"]):
                index = (
                    (y * shape["input_width"] + x) * shape["channels"]
                    + channel
                )
                struct.pack_into(
                    "<H",
                    result,
                    index * 2,
                    input_half_bits(shape_index, shape, y, x, channel),
                )
    return bytes(result)


def compare_finite_half(left: int, right: int) -> int:
    left_magnitude = left & 0x7FFF
    right_magnitude = right & 0x7FFF
    if left_magnitude == 0 and right_magnitude == 0:
        return 0
    left_negative = (left & 0x8000) != 0
    right_negative = (right & 0x8000) != 0
    if left_negative != right_negative:
        return -1 if left_negative else 1
    if left_magnitude == right_magnitude:
        return 0
    if left_negative:
        return 1 if left_magnitude < right_magnitude else -1
    return 1 if left_magnitude > right_magnitude else -1


def ptx_finite_max(left: int, right: int) -> int:
    comparison = compare_finite_half(left, right)
    if comparison > 0:
        return left
    if comparison < 0:
        return right
    if (left & 0x7FFF) == 0 and (right & 0x7FFF) == 0:
        return left & right
    return right


def build_reference_output(shape: dict[str, Any], tensor: bytes) -> bytes:
    expected_input_bytes = (
        shape["input_width"] * shape["input_height"] * shape["channels"] * 2
    )
    require(len(tensor) == expected_input_bytes, "reference input size mismatch")
    output = bytearray(
        shape["output_width"]
        * shape["output_height"]
        * shape["channels"]
        * 2
    )

    def read(y: int, x: int, channel: int) -> int:
        index = (
            (y * shape["input_width"] + x) * shape["channels"] + channel
        )
        return struct.unpack_from("<H", tensor, index * 2)[0]

    for output_y in range(shape["output_height"]):
        for output_x in range(shape["output_width"]):
            input_y = output_y * 2
            input_x = output_x * 2
            for channel in range(shape["channels"]):
                current = read(input_y, input_x, channel)
                current = ptx_finite_max(current, read(input_y, input_x + 1, channel))
                current = ptx_finite_max(current, read(input_y + 1, input_x, channel))
                current = ptx_finite_max(
                    current, read(input_y + 1, input_x + 1, channel)
                )
                output_index = (
                    (output_y * shape["output_width"] + output_x)
                    * shape["channels"]
                    + channel
                )
                struct.pack_into("<H", output, output_index * 2, current)
    return bytes(output)


def generated_contract() -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    inputs = tuple(generate_input(index, shape) for index, shape in enumerate(SHAPES))
    outputs = tuple(
        build_reference_output(shape, tensor)
        for shape, tensor in zip(SHAPES, inputs)
    )
    return inputs, outputs


def untouched_values(output: bytes, poison: int) -> int:
    require(len(output) % 2 == 0, "pooling output size is not even")
    marker = bytes((poison, poison))
    return sum(output[offset : offset + 2] == marker for offset in range(0, len(output), 2))


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
    root: Path,
    backend: str,
    run_id: str,
    poison: int,
    inputs: tuple[bytes, ...],
    outputs: tuple[bytes, ...],
) -> dict[str, Any]:
    receipt_path = root / "run-receipt.json"
    receipt = load_json(receipt_path)
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("schema") == RUN_SCHEMA, f"{root}: schema mismatch")
    require(receipt.get("backend") == expected_backend, f"{root}: backend mismatch")
    require(receipt.get("run_id") == run_id, f"{root}: run id mismatch")
    require(receipt.get("poison_byte") == poison, f"{root}: poison mismatch")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{root}: process id missing",
    )
    require(receipt.get("function_name", "k_pooling") == "k_pooling", f"{root}: function")
    require(receipt.get("input_generator") == GENERATOR, f"{root}: generator")
    require(receipt.get("external_tensor_inputs_opened") is False, f"{root}: external tensor feed")
    require(receipt.get("address_fields_recorded") is False, f"{root}: address fields")
    require(receipt.get("result") == "pass", f"{root}: run failed")
    require(receipt.get("input_immutability") is True, f"{root}: input mutation")
    require(receipt.get("full_write_coverage") is True, f"{root}: coverage")
    require(receipt.get("input_mismatched_bytes") == 0, f"{root}: input mismatch")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{root}: canary mismatch")
    require(receipt.get("untouched_half_values") == 0, f"{root}: untouched values")
    require_no_address_fields(receipt, str(receipt_path))

    if backend == "rtx":
        require(receipt.get("module_token") == 29, f"{root}: module token")
        require(receipt.get("function_token") == 27, f"{root}: function token")
        require(receipt.get("module_bytes") == MODULE_BYTES, f"{root}: module bytes")
        require(receipt.get("module_sha256") == MODULE_SHA256, f"{root}: module hash")
        require(receipt.get("function_name") == "k_pooling", f"{root}: function")
        require(
            receipt.get("module_transport")
            == "cuModuleLoadData exact selected sm_89 PTX fatbin; driver JIT to active RTX architecture",
            f"{root}: module transport",
        )
    else:
        require(
            receipt.get("reconstruction_unit") == "k_pooling-five-launch-family",
            f"{root}: reconstruction unit",
        )
        require(receipt.get("metal_function") == "fg_pooling", f"{root}: Metal function")
        require(isinstance(receipt.get("device_name"), str), f"{root}: device name")

    levels = receipt.get("levels")
    require(isinstance(levels, list) and len(levels) == len(SHAPES), f"{root}: levels")
    output_root = bytearray()
    summaries: list[dict[str, Any]] = []
    for index, (shape, expected_input, expected_output) in enumerate(
        zip(SHAPES, inputs, outputs)
    ):
        item = levels[index]
        require(isinstance(item, dict), f"{root}: level {index} receipt")
        require(item.get("index") == index, f"{root}: level {index} index")
        for key in (
            "input_width",
            "input_height",
            "output_width",
            "output_height",
            "channels",
            "grid",
            "block",
            "live_launch_token",
            "trace_order",
        ):
            require(item.get(key) == shape[key], f"{root}: level {index} {key}")
        require(item.get("dynamic_shared_bytes") == 0, f"{root}: level {index} shared")
        require(item.get("input_bytes") == len(expected_input), f"{root}: level {index} input bytes")
        require(item.get("output_bytes") == len(expected_output), f"{root}: level {index} output bytes")
        require(item.get("input_mismatched_bytes") == 0, f"{root}: level {index} input mismatch")
        require(item.get("canary_mismatched_bytes") == 0, f"{root}: level {index} canary")
        require(item.get("untouched_half_values") == 0, f"{root}: level {index} coverage")
        input_path = root / f"level-{index}-input-after.bin"
        output_path = root / f"level-{index}-output.bin"
        actual_input = input_path.read_bytes()
        actual_output = output_path.read_bytes()
        require_exact_bytes(actual_input, expected_input, f"{input_path}")
        require_exact_bytes(actual_output, expected_output, f"{output_path}")
        require(untouched_values(actual_output, poison) == 0, f"{root}: recomputed coverage {index}")
        input_hash = sha256_bytes(expected_input)
        output_hash = sha256_bytes(expected_output)
        require(item.get("input_pre_sha256") == input_hash, f"{root}: level {index} pre hash")
        require(item.get("input_post_sha256") == input_hash, f"{root}: level {index} post hash")
        require(item.get("output_sha256") == output_hash, f"{root}: level {index} output hash")
        output_root.extend(actual_output)
        summaries.append(
            {
                "index": index,
                "input_sha256": input_hash,
                "output_sha256": output_hash,
                "input_artifact_sha256": sha256_file(input_path),
                "output_artifact_sha256": sha256_file(output_path),
            }
        )
    output_root_sha256 = sha256_bytes(bytes(output_root))
    require(receipt.get("output_root_sha256") == output_root_sha256, f"{root}: output root")
    return {
        "run_id": run_id,
        "process_id": receipt["process_id"],
        "poison_byte": poison,
        "receipt_relative_path": f"{run_id}/run-receipt.json",
        "receipt_sha256": sha256_file(receipt_path),
        "output_root_sha256": output_root_sha256,
        "levels": summaries,
    }


def validate_backend(root: Path, backend: str, module_path: Path | None = None) -> dict[str, Any]:
    require(root.is_dir(), f"backend root not found: {root}")
    if backend == "rtx":
        require(module_path is not None, "RTX validation requires selected module")
        module = module_path.read_bytes()
        require(len(module) == MODULE_BYTES, "selected module byte count mismatch")
        require(sha256_bytes(module) == MODULE_SHA256, "selected module SHA-256 mismatch")
    inputs, outputs = generated_contract()
    runs = [
        validate_run(root / run_id, backend, run_id, poison, inputs, outputs)
        for run_id, poison in CASES.items()
    ]
    process_ids = [run["process_id"] for run in runs]
    require(len(set(process_ids)) == len(process_ids), f"{backend}: processes were not fresh")
    require(len({run["output_root_sha256"] for run in runs}) == 1, f"{backend}: output invariance")
    for index in range(len(SHAPES)):
        for kind in ("input-after", "output"):
            paths = [root / run_id / f"level-{index}-{kind}.bin" for run_id in CASES]
            if hasattr(os.path, "samestat"):
                stats = [path.stat() for path in paths]
                for left in range(len(stats)):
                    for right in range(left + 1, len(stats)):
                        require(
                            not os.path.samestat(stats[left], stats[right]),
                            f"{backend}: level {index} {kind} snapshots share file identity",
                        )
    result: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-cuda-driver" if backend == "rtx" else "mac-metal",
        "contract": {
            "function": "k_pooling",
            "layout": "nhwc-fp16-packed-half2",
            "window": [2, 2],
            "stride": [2, 2],
            "levels": list(SHAPES),
        },
        "fresh_process_count": len(process_ids),
        "distinct_process_ids": True,
        "poison_values": [0xA5, 0x5A],
        "two_processes_per_poison": True,
        "poison_output_invariance": True,
        "input_immutability": True,
        "full_write_coverage": True,
        "address_free": True,
        "input_generator": GENERATOR,
        "output_root_sha256": runs[0]["output_root_sha256"],
        "level_output_sha256": [sha256_bytes(value) for value in outputs],
        "runs": runs,
    }
    if backend == "rtx":
        result["selected_module"] = {
            "token": 29,
            "bytes": MODULE_BYTES,
            "sha256": MODULE_SHA256,
            "transport": "cuda-driver-exact-fatbin-jit",
            "embedded_code": "sm_89-ptx-no-elf",
        }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def attest_stored_backend(root: Path, validated: dict[str, Any]) -> dict[str, Any]:
    path = root / "backend-receipt.json"
    stored = load_json(path)
    require(stored == validated, f"stored backend receipt drift: {path}")
    root_value = stored.get("receipt_root_sha256")
    rootless = dict(stored)
    rootless.pop("receipt_root_sha256", None)
    require(root_value == canonical_sha256(rootless), f"stored backend root mismatch: {path}")
    sidecar = root / "backend-receipt.sha256"
    require(sidecar.is_file(), f"backend receipt sidecar missing: {sidecar}")
    require(
        sidecar.read_text(encoding="utf-8-sig").split()[0].lower() == sha256_file(path),
        f"backend receipt transfer hash mismatch: {path}",
    )
    return {
        "relative_path": "backend-receipt.json",
        "sha256": sha256_file(path),
        "receipt_root_sha256": root_value,
        "sidecar_reverified": True,
    }


def attest_rig(root: Path) -> dict[str, Any]:
    path = root / "rig-receipt.json"
    receipt = load_json(path)
    require(receipt.get("schema") == "neuralstudio.fg.pooling-rig.v1", "rig schema")
    require(str(receipt.get("job_id", "")).startswith("fg-pooling-"), "rig job id")
    require(receipt.get("result") == "pass", "rig result")
    require(receipt.get("fresh_process_count") == 4, "rig process count")
    require(receipt.get("address_fields_recorded") is False, "rig address fields")
    require(receipt.get("compile_exit_code") == 0, "rig compile exit code")
    require(receipt.get("command_exit_code") == 0, "rig command exit code")
    process_results = receipt.get("process_results")
    require(
        isinstance(process_results, list)
        and len(process_results) == 4
        and all(item.get("exit_code") == 0 for item in process_results),
        "rig process exit codes",
    )
    invocation = receipt.get("invocation", {})
    require(
        invocation.get("script_relative_path")
        == "scripts/build-fg-pooling-rtx.ps1",
        "rig invocation",
    )
    module = receipt.get("selected_module", {})
    require(module.get("token") == 29, "rig module token")
    require(module.get("bytes") == MODULE_BYTES, "rig module bytes")
    require(module.get("sha256") == MODULE_SHA256, "rig module hash")
    require(receipt.get("block") == [4, 4, 16], "rig block")
    require(receipt.get("dynamic_shared_bytes") == 0, "rig shared bytes")
    require_no_address_fields(receipt, str(path))
    sidecar = root / "rig-receipt.sha256"
    require(sidecar.is_file(), "rig receipt sidecar missing")
    require(
        sidecar.read_text(encoding="utf-8-sig").split()[0].lower() == sha256_file(path),
        "rig receipt transfer hash mismatch",
    )
    return {
        "relative_path": "rig-receipt.json",
        "sha256": sha256_file(path),
        "sidecar_reverified": True,
        "job_id": receipt.get("job_id"),
        "attempt_id": receipt.get("attempt_id"),
        "gpu": receipt.get("gpu"),
        "cuda_toolkit": receipt.get("cuda_toolkit"),
        "binary": receipt.get("binary"),
    }


def attest_mac_build(root: Path) -> dict[str, Any]:
    paths = (
        "build/fg-pooling-mac",
        "build/pooling.air",
        "build/pooling.metallib",
        "build/host-compile.log",
        "build/metal-compile.log",
        "build/metallib-link.log",
    )
    artifacts = []
    for relative in paths:
        path = root / relative
        require(path.is_file(), f"Mac build artifact missing: {path}")
        artifacts.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    devices = {
        load_json(root / run_id / "run-receipt.json").get("device_name")
        for run_id in CASES
    }
    require(len(devices) == 1 and None not in devices, "Mac device identity drift")
    return {"device_name": next(iter(devices)), "artifacts": artifacts}


def source_manifest() -> list[dict[str, Any]]:
    relatives = (
        "src/fg_pooling/rtx_oracle.cpp",
        "src/fg_pooling/macos_probe.mm",
        "src/fg_pooling/pooling.metal",
        "scripts/build-fg-pooling-rtx.ps1",
        "scripts/build-macos-fg-pooling.sh",
        "scripts/verify-fg-pooling.py",
        "evidence/specs/fg-pooling-v1/failure-ledger.md",
    )
    result = []
    for relative in relatives:
        path = REPOSITORY / relative
        require(path.is_file(), f"source artifact missing: {relative}")
        result.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return result


def validate_mac_isolation() -> dict[str, Any]:
    paths = (
        REPOSITORY / "src/fg_pooling/macos_probe.mm",
        REPOSITORY / "src/fg_pooling/pooling.metal",
        REPOSITORY / "scripts/build-macos-fg-pooling.sh",
    )
    forbidden = re.compile(r"\b(expected|oracle|rtx)\b", re.IGNORECASE)
    for path in paths:
        match = forbidden.search(path.read_text(encoding="utf-8"))
        if match is not None:
            raise VerificationError(
                f"Mac execution source contains external-feed token {match.group(0)!r}: {path}"
            )
    return {
        "inspected_sources": [str(path.relative_to(REPOSITORY)) for path in paths],
        "external_feed_tokens_absent": True,
        "mac_process_accepts_reference_inputs": False,
    }


def expect_negative(label: str, operation: Callable[[], None]) -> dict[str, Any]:
    try:
        operation()
    except VerificationError as error:
        return {"mutation": label, "rejected": True, "reason": str(error)}
    raise VerificationError(f"negative mutation was not rejected: {label}")


def negative_mutations(module_path: Path, inputs: tuple[bytes, ...], outputs: tuple[bytes, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mutated_output = bytearray(outputs[0])
    mutated_output[0] ^= 1
    results.append(expect_negative("output-byte-flip", lambda: require_exact_bytes(bytes(mutated_output), outputs[0], "mutated output")))
    mutated_input = bytearray(inputs[-1])
    mutated_input[17] ^= 0x40
    results.append(expect_negative("input-provenance-byte-flip", lambda: require_exact_bytes(bytes(mutated_input), inputs[-1], "mutated input")))
    mutated_module = bytearray(module_path.read_bytes())
    mutated_module[len(mutated_module) // 2] ^= 1
    results.append(expect_negative("selected-module-byte-flip", lambda: require(sha256_bytes(bytes(mutated_module)) == MODULE_SHA256, "selected module identity mismatch")))
    poisoned = bytearray(outputs[1])
    poisoned[:2] = b"\xa5\xa5"
    results.append(expect_negative("untouched-output-half", lambda: require(untouched_values(bytes(poisoned), 0xA5) == 0, "untouched output half detected")))
    address_receipt = {"device_address": "0x123456789abc"}
    results.append(expect_negative("raw-address-field", lambda: require_no_address_fields(address_receipt, "mutated receipt")))
    wrong_shape = dict(SHAPES[0])
    wrong_shape["channels"] = 64
    results.append(expect_negative("launch-shape-mutation", lambda: require(wrong_shape == SHAPES[0], "launch shape mismatch")))
    return results


def verify_cross_backend(rtx_root: Path, mac_root: Path, module_path: Path) -> dict[str, Any]:
    rtx = validate_backend(rtx_root, "rtx", module_path)
    mac = validate_backend(mac_root, "mac")
    require(rtx["output_root_sha256"] == mac["output_root_sha256"], "RTX/Mac output root mismatch")
    for run_id in CASES:
        for index in range(len(SHAPES)):
            for kind in ("input-after", "output"):
                rtx_path = rtx_root / run_id / f"level-{index}-{kind}.bin"
                mac_path = mac_root / run_id / f"level-{index}-{kind}.bin"
                require_exact_bytes(mac_path.read_bytes(), rtx_path.read_bytes(), f"RTX/Mac {run_id} level {index} {kind}")
                require(not os.path.samefile(rtx_path, mac_path), f"RTX/Mac artifact aliases: {run_id} level {index} {kind}")
    inputs, outputs = generated_contract()
    census = REPOSITORY / "evidence/specs/fg-selected-module-census-v1/selected-module-census.json"
    require(census.is_file(), "selected-module census missing")
    evidence: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_path_parent": {
            "census_relative_path": str(census.relative_to(REPOSITORY)),
            "census_sha256": sha256_file(census),
            "module_token": 29,
            "function_token": 27,
            "module_sha256": MODULE_SHA256,
            "function": "k_pooling",
            "live_launch_tokens": [shape["live_launch_token"] for shape in SHAPES],
            "trace_orders": [shape["trace_order"] for shape in SHAPES],
        },
        "rtx_backend": rtx,
        "mac_backend": mac,
        "backend_attestations": {
            "rtx": attest_stored_backend(rtx_root, rtx),
            "mac": attest_stored_backend(mac_root, mac),
            "rig": attest_rig(rtx_root),
            "mac_build": attest_mac_build(mac_root),
        },
        "mac_execution_isolation": validate_mac_isolation(),
        "cross_backend": {
            "independent_input_files": True,
            "input_bytes_equal": True,
            "byte_exact_output": True,
            "mismatched_output_bytes": 0,
            "output_bytes": sum(len(value) for value in outputs),
            "output_root_sha256": sha256_bytes(b"".join(outputs)),
            "five_live_selected_launches": True,
            "four_fresh_processes_per_backend": True,
            "two_processes_per_poison": True,
            "two_poison_output_invariance": True,
            "zero_untouched_half_values": True,
        },
        "negative_mutations": negative_mutations(module_path, inputs, outputs),
        "source_artifacts": source_manifest(),
        "accepted_frontier": {
            "live_selected_module_identity_bound": True,
            "exact_selected_module_executed_on_rtx": True,
            "five_launch_pooling_family_on_metal": True,
            "rtx_mac_byte_exact_group_parity": True,
            "finite_fp16_domain_including_signed_zero_and_subnormal": True,
            "nan_or_infinity_domain_proven": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    evidence["evidence_root_sha256"] = canonical_sha256(evidence)
    return evidence


def self_test() -> None:
    inputs, outputs = generated_contract()
    require(len(inputs) == len(SHAPES) == len(outputs), "level count")
    require((inputs, outputs) == generated_contract(), "generator nondeterminism")
    for shape, tensor, output in zip(SHAPES, inputs, outputs):
        require(len(tensor) == shape["input_width"] * shape["input_height"] * shape["channels"] * 2, "input size")
        require(len(output) == shape["output_width"] * shape["output_height"] * shape["channels"] * 2, "output size")
        require(untouched_values(output, 0xA5) == 0, "A5 coverage collision")
        require(untouched_values(output, 0x5A) == 0, "5A coverage collision")
    require(ptx_finite_max(0x0000, 0x8000) == 0x0000, "positive/negative zero")
    require(ptx_finite_max(0x8000, 0x8000) == 0x8000, "negative zero tie")
    require(ptx_finite_max(0xBC00, 0x3C00) == 0x3C00, "signed compare")
    with tempfile.TemporaryDirectory() as temporary:
        module = Path(temporary) / "module.bin"
        module.write_bytes(bytes(MODULE_BYTES))
        mutations = negative_mutations(module, inputs, outputs)
        require(all(item["rejected"] for item in mutations), "negative mutations")
    print("self_test=pass")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--module", type=Path)
    backend.add_argument("--output", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--rtx-root", type=Path, required=True)
    verify.add_argument("--mac-root", type=Path, required=True)
    verify.add_argument("--module", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
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
        print(f"receipt_root_sha256={value['receipt_root_sha256']}")
        return 0
    value = verify_cross_backend(arguments.rtx_root, arguments.mac_root, arguments.module)
    write_json(arguments.output, value)
    write_sha256_sidecar(arguments.output)
    print(f"evidence_root_sha256={value['evidence_root_sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"verification_error={error}", file=os.sys.stderr)
        raise SystemExit(1) from error
