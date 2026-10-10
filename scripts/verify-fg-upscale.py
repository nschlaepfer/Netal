#!/usr/bin/env python3
"""Verify the five live-selected Frame Generation k_upscale launches.

The RTX CUDA Driver executable and native Metal executable independently
generate their inputs. This verifier recomputes that generator, validates all
artifacts and process/poison gates, and compares every output byte. It never
provides an expected tensor to either live backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
MASK64 = (1 << 64) - 1
RUN_SCHEMA = "neuralstudio.fg.upscale-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.upscale-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.upscale-verification.v1"
GENERATOR = "splitmix64-fg-upscale-signed-spatial-v3"
DOMAIN = "signed-finite-fp16-normal-or-signed-zero-spatially-mixed-below-four"
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")
CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}
MODULES = (
    {
        "token": 36,
        "bytes": 2296,
        "sha256": "d6962e058c46f6de9c6c5f8c75dafde64a34f094a49f1719af16c505f3b530fd",
    },
    {
        "token": 40,
        "bytes": 2296,
        "sha256": "070c7f4987251c28bcad097ac8a00e7b1e2644dae4267ae37b676e280478a2b8",
    },
    {
        "token": 45,
        "bytes": 2296,
        "sha256": "547a05ab275edcb0d413e4b7f89bada85f08035fa0517cd5267960fe695498fc",
    },
    {
        "token": 49,
        "bytes": 2296,
        "sha256": "2b91bf740602f074acbaa5efba1cd1115c5ee52653a96d087c0d52231de2a74a",
    },
)
SHAPES = (
    {
        "input_height": 3,
        "input_width": 5,
        "output_height": 6,
        "output_width": 10,
        "channels": 256,
        "grid": [1, 2, 2],
        "block": [32, 5, 5],
        "live_launch_token": 32,
        "trace_order": 30,
        "module_index": 0,
        "address_free_parameter_sha256": "c152f0af5bc347eb9fe551c9f10c05a52f8662cfb70df6675f3b52e586c1a0a0",
    },
    {
        "input_height": 6,
        "input_width": 10,
        "output_height": 12,
        "output_width": 20,
        "channels": 128,
        "grid": [1, 3, 2],
        "block": [16, 8, 8],
        "live_launch_token": 36,
        "trace_order": 34,
        "module_index": 1,
        "address_free_parameter_sha256": "cebc93f61861652d0e2d0b3996777cc5ead6142711841aa4edf895e4a3adf646",
    },
    {
        "input_height": 12,
        "input_width": 20,
        "output_height": 24,
        "output_width": 40,
        "channels": 128,
        "grid": [1, 5, 3],
        "block": [16, 8, 8],
        "live_launch_token": 40,
        "trace_order": 38,
        "module_index": 1,
        "address_free_parameter_sha256": "072dfaec950e6ca0be8b6a3bc7cd0b0904f79ec1e7f3b59d8044b8acf6bc8e37",
    },
    {
        "input_height": 24,
        "input_width": 40,
        "output_height": 48,
        "output_width": 80,
        "channels": 64,
        "grid": [1, 8, 5],
        "block": [8, 11, 11],
        "live_launch_token": 44,
        "trace_order": 42,
        "module_index": 2,
        "address_free_parameter_sha256": "9e9e5e0751feb2f521e0874b9b25d93c7ee040e6768ac9d3da9b7f104ea3b1d2",
    },
    {
        "input_height": 48,
        "input_width": 80,
        "output_height": 96,
        "output_width": 160,
        "channels": 32,
        "grid": [1, 10, 6],
        "block": [4, 16, 16],
        "live_launch_token": 48,
        "trace_order": 46,
        "module_index": 3,
        "address_free_parameter_sha256": "d92d55ba971164cc0e879eeba693a8a8ee71f8fd5da6daaee585e1f388dfcfeb",
    },
)
SCALE_BITS = {
    (3, 6): "3ecccccd",
    (5, 10): "3ee38e39",
    (6, 12): "3ee8ba2f",
    (10, 20): "3ef286bd",
    (12, 24): "3ef4de9c",
    (20, 40): "3ef96f97",
    (24, 48): "3efa8d9d",
    (40, 80): "3efcc26f",
    (48, 96): "3efd4e26",
    (80, 160): "3efe63d2",
}


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


def require_no_address_fields(value: Any, label: str) -> None:
    """Reject pointer-like fields or values from an address-free receipt."""

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                require(
                    not any(
                        marker in lowered
                        for marker in ("pointer", "device_address", "gpu_va")
                    ),
                    f"{label} contains address field at {path}.{key}",
                )
                walk(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
        elif isinstance(item, str):
            require(
                ADDRESS_PATTERN.search(item) is None,
                f"{label} contains pointer-like value at {path}",
            )

    walk(value, "$")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
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


def input_half_bits(shape_index: int, index: int, channels: int) -> int:
    channel = index % channels
    channel_code = splitmix64(
        0xA0761D6478BD642F
        ^ ((shape_index + 1) * 0xE7037ED1A0B428DB)
        ^ (channel * 0x8EBC6AF09C88C6E3)
    )
    if (channel_code & 31) == 0:
        return 0x0000
    if (channel_code & 31) == 1:
        return 0x8000
    mixed = splitmix64(
        0xD6E8FEB86659FD93
        ^ ((shape_index + 1) * 0x9E3779B97F4A7C15)
        ^ (index * 0x94D049BB133111EB)
    )
    sign = 0x8000 if (mixed >> 63) else 0
    mode = mixed & 15
    fixed = {
        0: 0x3400,
        1: 0x3800,
        2: 0x3C00,
        3: 0x4000,
        4: 0x3555,
        5: 0x3BFF,
        6: 0x3C01,
        7: 0x3AAA,
    }
    if mode in fixed:
        return sign | fixed[mode]
    exponent = 13 + ((mixed >> 12) & 3)
    return sign | (exponent << 10) | (mixed & 0x03FF)


def generate_input(shape_index: int, shape: dict[str, Any]) -> bytes:
    values = (
        shape["input_height"] * shape["input_width"] * shape["channels"]
    )
    result = bytearray(values * 2)
    for index in range(values):
        struct.pack_into(
            "<H",
            result,
            index * 2,
            input_half_bits(shape_index, index, shape["channels"]),
        )
    return bytes(result)


def expected_output_bytes(shape: dict[str, Any]) -> int:
    return (
        shape["output_height"]
        * shape["output_width"]
        * shape["channels"]
        * 2
    )


def validate_shape_fields(
    level: dict[str, Any], shape: dict[str, Any], backend: str, label: str
) -> None:
    module = MODULES[shape["module_index"]]
    scalar_fields = (
        "input_height",
        "input_width",
        "output_height",
        "output_width",
        "channels",
        "live_launch_token",
        "trace_order",
    )
    for field in scalar_fields:
        require(level.get(field) == shape[field], f"{label} {field} drift")
    require(
        level.get("physical_output_height") == shape["output_height"],
        f"{label} physical output height drift",
    )
    require(
        level.get("physical_output_width") == shape["output_width"],
        f"{label} physical output width drift",
    )
    if backend == "rtx":
        require(level.get("grid") == shape["grid"], f"{label} grid drift")
        require(level.get("block") == shape["block"], f"{label} block drift")
        require(level.get("module_token") == module["token"], f"{label} module token drift")
        require(level.get("module_bytes") == module["bytes"], f"{label} module bytes drift")
        require(level.get("module_sha256") == module["sha256"], f"{label} module hash drift")
        require(
            level.get("module_transport")
            == "cuModuleLoadDataEx-exact-selected-module",
            f"{label} module transport drift",
        )
        require(level.get("dynamic_shared_bytes") == 0, f"{label} shared memory drift")
    else:
        require(level.get("live_grid") == shape["grid"], f"{label} live grid drift")
        require(level.get("live_block") == shape["block"], f"{label} live block drift")
        require(
            level.get("source_module_token") == module["token"],
            f"{label} source module token drift",
        )
        require(
            level.get("source_module_bytes") == module["bytes"],
            f"{label} source module bytes drift",
        )
        require(
            level.get("source_module_sha256") == module["sha256"],
            f"{label} source module hash drift",
        )
        require(
            level.get("metal_threads")
            == [
                shape["channels"] // 2,
                shape["output_width"],
                shape["output_height"],
            ],
            f"{label} Metal thread extent drift",
        )


def validate_run(
    root: Path, run_id: str, backend: str
) -> tuple[dict[str, Any], list[bytes], list[bytes]]:
    run_root = root / run_id
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id} run schema drift")
    require(receipt.get("backend") == expected_backend, f"{run_id} backend drift")
    require(receipt.get("run_id") == run_id, f"{run_id} identifier drift")
    require(receipt.get("poison_byte") == CASES[run_id], f"{run_id} poison drift")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{run_id} invalid process id",
    )
    require(receipt.get("function_name") == "k_upscale", f"{run_id} function drift")
    require(
        receipt.get("reconstruction_unit") == "k_upscale-five-launch-family",
        f"{run_id} unit drift",
    )
    require(receipt.get("input_generator") == GENERATOR, f"{run_id} generator drift")
    require(receipt.get("numerical_domain") == DOMAIN, f"{run_id} domain drift")
    require(receipt.get("external_tensor_inputs_opened") is False, f"{run_id} consumed external tensor")
    require(receipt.get("address_fields_recorded") is False, f"{run_id} contains address fields")
    levels = receipt.get("levels")
    require(isinstance(levels, list) and len(levels) == len(SHAPES), f"{run_id} level count drift")

    inputs: list[bytes] = []
    outputs: list[bytes] = []
    input_mismatches = 0
    canary_mismatches = 0
    poison_matches = 0
    for index, (level, shape) in enumerate(zip(levels, SHAPES)):
        label = f"{run_id} level {index}"
        require(isinstance(level, dict), f"{label} is not an object")
        require(level.get("index") == index, f"{label} index drift")
        validate_shape_fields(level, shape, backend, label)
        expected_input = generate_input(index, shape)
        input_path = run_root / f"input-level-{index}-after.bin"
        output_path = run_root / f"output-level-{index}.bin"
        try:
            actual_input = input_path.read_bytes()
            output = output_path.read_bytes()
        except OSError as error:
            raise VerificationError(f"{label} artifact read failed: {error}") from error
        require(actual_input == expected_input, f"{label} input generator mismatch")
        require(
            len(output) == expected_output_bytes(shape),
            f"{label} output byte count drift",
        )
        require(level.get("input_bytes") == len(expected_input), f"{label} input bytes drift")
        require(level.get("output_bytes") == len(output), f"{label} output bytes drift")
        expected_input_hash = sha256_bytes(expected_input)
        require(level.get("input_pre_sha256") == expected_input_hash, f"{label} input pre-hash drift")
        require(level.get("input_post_sha256") == expected_input_hash, f"{label} input post-hash drift")
        require(level.get("output_sha256") == sha256_bytes(output), f"{label} output hash drift")
        require(level.get("input_mismatched_bytes") == 0, f"{label} input was modified")
        require(level.get("canary_mismatched_bytes") == 0, f"{label} canary was modified")
        poison_pair = bytes([CASES[run_id], CASES[run_id]])
        actual_poison_matches = sum(
            output[offset : offset + 2] == poison_pair
            for offset in range(0, len(output), 2)
        )
        require(
            level.get("poison_matching_half_values") == actual_poison_matches,
            f"{label} poison-match count drift",
        )
        input_mismatches += int(level["input_mismatched_bytes"])
        canary_mismatches += int(level["canary_mismatched_bytes"])
        poison_matches += actual_poison_matches
        inputs.append(actual_input)
        outputs.append(output)

    output_root = b"".join(outputs)
    require(receipt.get("output_root_sha256") == sha256_bytes(output_root), f"{run_id} output root drift")
    require(receipt.get("input_mismatched_bytes") == input_mismatches == 0, f"{run_id} input mismatch total")
    require(receipt.get("canary_mismatched_bytes") == canary_mismatches == 0, f"{run_id} canary mismatch total")
    require(
        receipt.get("poison_matching_half_values") == poison_matches,
        f"{run_id} poison-match total drift",
    )
    require(receipt.get("input_immutability") is True, f"{run_id} input immutability false")
    require(
        receipt.get("single_poison_zero_matching_values")
        is (poison_matches == 0),
        f"{run_id} single-poison match claim drift",
    )
    require(receipt.get("result") == "pass", f"{run_id} result is not pass")
    return receipt, inputs, outputs


def build_backend_receipt(root: Path, backend: str) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"backend root invalid: {root}")
    run_receipts: list[dict[str, Any]] = []
    output_sets: list[list[bytes]] = []
    for run_id in CASES:
        receipt, _inputs, outputs = validate_run(root, run_id, backend)
        run_receipts.append(receipt)
        output_sets.append(outputs)
    process_ids = {receipt["process_id"] for receipt in run_receipts}
    require(len(process_ids) == len(CASES), "fresh-process gate failed")
    baseline = output_sets[0]
    for run_id, outputs in zip(CASES, output_sets):
        for index, (left, right) in enumerate(zip(baseline, outputs)):
            require(left == right, f"backend output mismatch: {run_id} level {index}")
    a5_outputs = output_sets[0]
    five_a_outputs = output_sets[2]
    two_poison_unwritten = 0
    for a5_output, five_a_output in zip(a5_outputs, five_a_outputs):
        require(len(a5_output) == len(five_a_output), "poison output size mismatch")
        two_poison_unwritten += sum(
            a5_output[offset : offset + 2] == b"\xA5\xA5"
            and five_a_output[offset : offset + 2] == b"\x5A\x5A"
            for offset in range(0, len(a5_output), 2)
        )
    require(two_poison_unwritten == 0, "two-poison write coverage has untouched values")
    level_hashes = [sha256_bytes(output) for output in baseline]
    run_artifacts = []
    for run_id, receipt in zip(CASES, run_receipts):
        run_artifacts.append(
            {
                "run_id": run_id,
                "poison_byte": CASES[run_id],
                "run_receipt_sha256": sha256_file(root / run_id / "run-receipt.json"),
                "output_root_sha256": receipt["output_root_sha256"],
                "poison_matching_half_values": receipt[
                    "poison_matching_half_values"
                ],
            }
        )
    stable = {
        "backend": backend,
        "function_name": "k_upscale",
        "input_generator": GENERATOR,
        "numerical_domain": DOMAIN,
        "fresh_process_count": len(process_ids),
        "poison_values": sorted(set(CASES.values())),
        "live_launch_tokens": [shape["live_launch_token"] for shape in SHAPES],
        "level_output_sha256": level_hashes,
        "output_root_sha256": sha256_bytes(b"".join(baseline)),
        "run_artifacts": run_artifacts,
        "input_immutability": True,
        "canary_integrity": True,
        "two_poison_unwritten_half_values": two_poison_unwritten,
        "full_write_coverage": True,
        "fresh_process_output_equality": True,
        "two_poison_output_invariance": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document = {
        "schema": BACKEND_SCHEMA,
        **stable,
        "backend_evidence_root_sha256": canonical_sha256(stable),
    }
    require_no_address_fields(document, f"{backend} backend receipt")
    return document


def write_backend(root: Path, backend: str, output: Path) -> dict[str, Any]:
    document = build_backend_receipt(root, backend)
    write_json(output, document)
    return document


def verify_existing_backend(root: Path, backend: str) -> dict[str, Any]:
    expected = build_backend_receipt(root, backend)
    actual = load_json(root / "backend-receipt.json")
    require(actual == expected, f"{backend} backend receipt does not match artifacts")
    return actual


def validate_modules(paths: list[Path]) -> list[dict[str, Any]]:
    require(len(paths) == len(MODULES), "exactly four selected modules are required")
    result = []
    for index, (path, expected) in enumerate(zip(paths, MODULES)):
        require(path.is_file() and not path.is_symlink(), f"module {index} path invalid")
        require(path.stat().st_size == expected["bytes"], f"module {index} byte count drift")
        require(sha256_file(path) == expected["sha256"], f"module {index} hash drift")
        result.append(
            {
                "token": expected["token"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "name": path.name,
            }
        )
    return result


def validate_coordinate_probe(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "coordinate probe path invalid")
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        rows = list(csv.DictReader(raw.decode(encoding).splitlines()))
    except (UnicodeError, csv.Error) as error:
        raise VerificationError(f"coordinate probe parse failed: {error}") from error
    expected_columns = [
        "input_extent",
        "output_extent",
        "coordinate",
        "scale_bits",
        "source_bits",
        "lower",
        "upper",
        "fraction_bits",
        "fraction_half_bits",
        "inverse_half_bits",
        "scale",
        "source",
    ]
    require(rows and list(rows[0]) == expected_columns, "coordinate probe columns drift")
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        try:
            pair = (int(row["input_extent"]), int(row["output_extent"]))
            coordinate = int(row["coordinate"])
            lower = int(row["lower"])
            upper = int(row["upper"])
        except (KeyError, ValueError) as error:
            raise VerificationError(f"coordinate probe numeric field invalid: {error}") from error
        require(pair in SCALE_BITS, f"coordinate probe has unselected extent pair {pair}")
        require(row["scale_bits"].lower() == "0x" + SCALE_BITS[pair], f"coordinate probe scale drift for {pair}")
        require(0 <= coordinate < pair[1], f"coordinate probe coordinate out of range for {pair}")
        require(0 <= lower < pair[0] and 0 <= upper < pair[0], f"coordinate probe source index out of range for {pair}")
        for field, digits in (
            ("source_bits", 8),
            ("fraction_bits", 8),
            ("fraction_half_bits", 4),
            ("inverse_half_bits", 4),
        ):
            require(
                re.fullmatch(rf"0x[0-9a-fA-F]{{{digits}}}", row[field]) is not None,
                f"coordinate probe {field} encoding drift",
            )
        grouped.setdefault(pair, []).append(row)
    require(set(grouped) == set(SCALE_BITS), "coordinate probe extent coverage drift")
    for pair, pair_rows in grouped.items():
        require(len(pair_rows) == pair[1], f"coordinate probe row count drift for {pair}")
        require(
            [int(row["coordinate"]) for row in pair_rows] == list(range(pair[1])),
            f"coordinate probe order drift for {pair}",
        )
    return {
        "name": path.name,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "row_count": len(rows),
        "selected_extent_pair_count": len(grouped),
        "scale_bits": {f"{left}_to_{right}": bits for (left, right), bits in SCALE_BITS.items()},
        "purpose": "supporting div.approx opcode semantics; no tensor outputs consumed",
    }


def build_cross_receipt(
    rtx_root: Path,
    mac_root: Path,
    module_paths: list[Path],
    coordinate_probe_path: Path,
) -> dict[str, Any]:
    rtx_backend = verify_existing_backend(rtx_root, "rtx")
    mac_backend = verify_existing_backend(mac_root, "mac")
    modules = validate_modules(module_paths)
    coordinate_probe = validate_coordinate_probe(coordinate_probe_path)
    rig_receipt = load_json(rtx_root / "rig-receipt.json")
    require(
        rig_receipt.get("schema") == "neuralstudio.fg.upscale-rig.v1",
        "RTX rig receipt schema drift",
    )
    require(rig_receipt.get("result") == "pass", "RTX rig receipt did not pass")
    require(
        rig_receipt.get("fresh_process_count") == len(CASES),
        "RTX rig fresh-process count drift",
    )
    require(
        rig_receipt.get("external_expected_tensors_consumed") is False,
        "RTX rig receipt consumed expected tensors",
    )
    require(
        rig_receipt.get("address_fields_recorded") is False,
        "RTX rig receipt contains addresses",
    )
    rig_modules = rig_receipt.get("selected_modules")
    require(
        isinstance(rig_modules, list) and len(rig_modules) == len(MODULES),
        "RTX rig module attestation count drift",
    )
    for actual, expected in zip(rig_modules, MODULES):
        require(actual.get("token") == expected["token"], "RTX rig module token drift")
        require(actual.get("bytes") == expected["bytes"], "RTX rig module bytes drift")
        require(actual.get("sha256") == expected["sha256"], "RTX rig module hash drift")
    mac_run = load_json(mac_root / next(iter(CASES)) / "run-receipt.json")
    require(
        isinstance(mac_run.get("device_name"), str) and mac_run["device_name"],
        "Mac device attestation missing",
    )
    mismatch_counts: list[int] = []
    level_hashes: list[str] = []
    for index in range(len(SHAPES)):
        rtx_output = (rtx_root / next(iter(CASES)) / f"output-level-{index}.bin").read_bytes()
        mac_output = (mac_root / next(iter(CASES)) / f"output-level-{index}.bin").read_bytes()
        mismatch = sum(left != right for left, right in zip(rtx_output, mac_output))
        mismatch += abs(len(rtx_output) - len(mac_output))
        mismatch_counts.append(mismatch)
        require(mismatch == 0, f"RTX/Metal mismatch at level {index}: {mismatch} bytes")
        level_hashes.append(sha256_bytes(rtx_output))
    require(
        level_hashes == rtx_backend["level_output_sha256"]
        == mac_backend["level_output_sha256"],
        "backend level hash lists differ",
    )
    require(
        rtx_backend["two_poison_unwritten_half_values"]
        == mac_backend["two_poison_unwritten_half_values"]
        == 0,
        "backend two-poison write coverage differs",
    )
    source_paths = [
        REPOSITORY / "src/fg_upscale/rtx_oracle.cpp",
        REPOSITORY / "src/fg_upscale/macos_probe.mm",
        REPOSITORY / "src/fg_upscale/upscale.metal",
        REPOSITORY / "src/fg_upscale/rtx_coordinate_probe.cu",
        REPOSITORY / "scripts/build-fg-upscale-rtx.ps1",
        REPOSITORY / "scripts/build-macos-fg-upscale.sh",
        REPOSITORY / "scripts/verify-fg-upscale.py",
    ]
    source_artifacts = []
    for path in source_paths:
        require(path.is_file() and not path.is_symlink(), f"source artifact missing: {path.name}")
        source_artifacts.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    stable = {
        "function_name": "k_upscale",
        "reconstruction_unit": "k_upscale-five-launch-family",
        "input_generator": GENERATOR,
        "numerical_domain": DOMAIN,
        "selected_modules": modules,
        "live_launches": [
            {
                key: shape[key]
                for key in (
                    "live_launch_token",
                    "trace_order",
                    "input_height",
                    "input_width",
                    "output_height",
                    "output_width",
                    "channels",
                    "grid",
                    "block",
                    "module_index",
                    "address_free_parameter_sha256",
                )
            }
            for shape in SHAPES
        ],
        "rtx_backend_receipt_sha256": sha256_file(rtx_root / "backend-receipt.json"),
        "mac_backend_receipt_sha256": sha256_file(mac_root / "backend-receipt.json"),
        "level_output_sha256": level_hashes,
        "level_mismatched_bytes": mismatch_counts,
        "total_mismatched_bytes": sum(mismatch_counts),
        "fresh_processes_per_backend": len(CASES),
        "poison_values": sorted(set(CASES.values())),
        "source_artifacts": source_artifacts,
        "hardware_attestations": {
            "rtx": {
                "job_id": rig_receipt.get("job_id"),
                "gpu": rig_receipt.get("gpu"),
                "cuda_toolkit": rig_receipt.get("cuda_toolkit"),
                "rig_receipt_sha256": sha256_file(rtx_root / "rig-receipt.json"),
                "fresh_process_count": rig_receipt.get("fresh_process_count"),
            },
            "mac": {
                "device_name": mac_run["device_name"],
                "fresh_process_count": mac_backend["fresh_process_count"],
            },
        },
        "selected_opcode_contract": {
            "packed_parameter_abi": {
                "bytes": 40,
                "little_endian_layout": [
                    "input_device_pointer_u64",
                    "output_device_pointer_u64",
                    "input_height_u32",
                    "input_width_u32",
                    "output_height_u32",
                    "output_width_u32",
                    "physical_output_height_u32",
                    "physical_output_width_u32",
                ],
                "volatile_process_bytes_excluded_from_address_free_receipt": True,
            },
            "horizontal_lowering": "hfma2(left, one_minus_x, round_half2(right_times_x))",
            "vertical_lowering": "hfma2(top, one_minus_y, round_half2(bottom_times_y))",
            "div_approx_scale_bits": coordinate_probe["scale_bits"],
            "coordinate_probe": coordinate_probe,
        },
        "external_expected_tensors_consumed": False,
        "input_immutability": True,
        "canary_integrity": True,
        "two_poison_unwritten_half_values": 0,
        "full_write_coverage": True,
        "fresh_process_output_equality": True,
        "two_poison_output_invariance": True,
        "rtx_metal_byte_exact": True,
        "address_fields_recorded": False,
        "unsupported_numerical_domains": [
            "FP16 subnormal inputs or intermediates",
            "FP16 infinities",
            "FP16 NaNs",
            "physical output extents larger than logical output extents",
        ],
        "accepted_frontier": {
            "five_live_selected_upscale_launches_rtx_metal_byte_exact": True,
            "dependency_closed_multi_family_fg_group": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    document = {
        "schema": FINAL_SCHEMA,
        **stable,
        "verification_root_sha256": canonical_sha256(stable),
    }
    require_no_address_fields(document, "cross-backend receipt")
    return document


def write_cross(
    rtx_root: Path,
    mac_root: Path,
    module_paths: list[Path],
    coordinate_probe_path: Path,
    output: Path,
) -> dict[str, Any]:
    document = build_cross_receipt(
        rtx_root, mac_root, module_paths, coordinate_probe_path
    )
    write_json(output, document)
    write_sha256_sidecar(output)
    return document


def expect_rejection(label: str, operation: Any) -> str:
    try:
        operation()
    except (VerificationError, OSError, json.JSONDecodeError) as error:
        return f"{label}: {error}"
    raise VerificationError(f"mutation was incorrectly accepted: {label}")


def self_test(
    rtx_root: Path,
    mac_root: Path,
    module_paths: list[Path],
    coordinate_probe_path: Path,
) -> dict[str, Any]:
    baseline = build_cross_receipt(
        rtx_root, mac_root, module_paths, coordinate_probe_path
    )
    rejections: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fg-upscale-self-test-") as temporary:
        temp = Path(temporary)

        mutated_rtx = temp / "rtx-output"
        shutil.copytree(rtx_root, mutated_rtx)
        output_path = mutated_rtx / "poison-a5-process-a/output-level-0.bin"
        data = bytearray(output_path.read_bytes())
        data[0] ^= 1
        output_path.write_bytes(data)
        rejections.append(
            expect_rejection(
                "output-byte-flip",
                lambda: build_backend_receipt(mutated_rtx, "rtx"),
            )
        )

        mutated_mac = temp / "mac-process"
        shutil.copytree(mac_root, mutated_mac)
        receipt_path = mutated_mac / "poison-a5-process-b/run-receipt.json"
        receipt = load_json(receipt_path)
        receipt["process_id"] = load_json(
            mutated_mac / "poison-a5-process-a/run-receipt.json"
        )["process_id"]
        write_json(receipt_path, receipt)
        rejections.append(
            expect_rejection(
                "process-id-collision",
                lambda: build_backend_receipt(mutated_mac, "mac"),
            )
        )

        mutated_rtx = temp / "rtx-poison"
        shutil.copytree(rtx_root, mutated_rtx)
        receipt_path = mutated_rtx / "poison-5a-process-a/run-receipt.json"
        receipt = load_json(receipt_path)
        receipt["poison_byte"] = 0xA5
        write_json(receipt_path, receipt)
        rejections.append(
            expect_rejection(
                "poison-relabel",
                lambda: build_backend_receipt(mutated_rtx, "rtx"),
            )
        )

        mutated_mac = temp / "mac-input"
        shutil.copytree(mac_root, mutated_mac)
        input_path = mutated_mac / "poison-5a-process-b/input-level-4-after.bin"
        input_path.write_bytes(input_path.read_bytes()[:-2])
        rejections.append(
            expect_rejection(
                "input-truncation",
                lambda: build_backend_receipt(mutated_mac, "mac"),
            )
        )

        module_copies = []
        for path in module_paths:
            destination = temp / path.name
            shutil.copy2(path, destination)
            module_copies.append(destination)
        module_data = bytearray(module_copies[0].read_bytes())
        module_data[-1] ^= 1
        module_copies[0].write_bytes(module_data)
        rejections.append(
            expect_rejection(
                "module-byte-flip",
                lambda: validate_modules(module_copies),
            )
        )

        mutated_rtx = temp / "rtx-backend-receipt"
        shutil.copytree(rtx_root, mutated_rtx)
        backend_path = mutated_rtx / "backend-receipt.json"
        backend = load_json(backend_path)
        backend["result"] = "fail"
        write_json(backend_path, backend)
        rejections.append(
            expect_rejection(
                "backend-receipt-mutation",
                lambda: build_cross_receipt(
                    mutated_rtx, mac_root, module_paths, coordinate_probe_path
                ),
            )
        )

        coordinate_copy = temp / coordinate_probe_path.name
        shutil.copy2(coordinate_probe_path, coordinate_copy)
        coordinate_raw = coordinate_copy.read_bytes()
        coordinate_encoding = (
            "utf-16"
            if coordinate_raw.startswith((b"\xff\xfe", b"\xfe\xff"))
            else "utf-8-sig"
        )
        coordinate_text = coordinate_raw.decode(coordinate_encoding)
        require("3ecccccd" in coordinate_text, "coordinate mutation target missing")
        coordinate_copy.write_text(
            coordinate_text.replace("3ecccccd", "3ecccccc", 1),
            encoding=coordinate_encoding,
        )
        rejections.append(
            expect_rejection(
                "coordinate-probe-mutation",
                lambda: validate_coordinate_probe(coordinate_copy),
            )
        )

    stable = {
        "baseline_verification_root_sha256": baseline["verification_root_sha256"],
        "mutation_count": len(rejections),
        "mutations_rejected": True,
        "rejections": rejections,
        "result": "pass",
    }
    return {
        "schema": "neuralstudio.fg.upscale-self-test.v1",
        **stable,
        "self_test_root_sha256": canonical_sha256(stable),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)

    cross = subparsers.add_parser("cross")
    cross.add_argument("--rtx-root", type=Path, required=True)
    cross.add_argument("--mac-root", type=Path, required=True)
    cross.add_argument("--module", type=Path, action="append", required=True)
    cross.add_argument("--coordinate-probe", type=Path, required=True)
    cross.add_argument("--output", type=Path, required=True)

    test = subparsers.add_parser("self-test")
    test.add_argument("--rtx-root", type=Path, required=True)
    test.add_argument("--mac-root", type=Path, required=True)
    test.add_argument("--module", type=Path, action="append", required=True)
    test.add_argument("--coordinate-probe", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "backend":
            document = write_backend(args.root, args.backend, args.output)
            print(json.dumps(document, sort_keys=True))
        elif args.command == "cross":
            document = write_cross(
                args.rtx_root,
                args.mac_root,
                args.module,
                args.coordinate_probe,
                args.output,
            )
            print(json.dumps(document, sort_keys=True))
        else:
            document = self_test(
                args.rtx_root,
                args.mac_root,
                args.module,
                args.coordinate_probe,
            )
            write_json(args.output, document)
            write_sha256_sidecar(args.output)
            print(json.dumps(document, sort_keys=True))
        return 0
    except VerificationError as error:
        print(f"fg upscale verification failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
