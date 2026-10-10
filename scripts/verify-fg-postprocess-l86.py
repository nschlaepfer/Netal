#!/usr/bin/env python3
"""Fail-closed verifier for live-selected FG launch 86 mode 0 and Metal parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any


RUN_SCHEMA = "neuralstudio.fg.postprocess-l86-run.v1"
RTX_RIG_SCHEMA = "neuralstudio.fg.postprocess-l86-rig.v1"
MAC_BUILD_SCHEMA = "neuralstudio.fg.postprocess-l86-mac-build.v1"
BACKEND_SCHEMA = "neuralstudio.fg.postprocess-l86-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.postprocess-l86-verification.v1"
MODULE_SHA = "143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0"
RAW_PARAMETER_SHA = "faea46ab664868a3ad3bcdcf66dbb7350fddd82fb8d65031b2ea2cec12eb8925"
CANONICAL_PARAMETER_SHA = "8f2b991ee0a0288d0f79777881720354e691b920393d921ff53a30b7f27d5fc9"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
GENERATOR = "splitmix64-fg-postprocess-l86-independent-pyramids-v1"
WIDTHS = (320, 160, 80, 40, 20)
HEIGHTS = (180, 90, 45, 23, 12)
LEVEL_BYTES = tuple(width * height * 8 for width, height in zip(WIDTHS, HEIGHTS))
CASES = {
    "pyramid-a-poison-a5": ("pyramid-a", 0, 0xA5),
    "pyramid-b-poison-a5": ("pyramid-b", 1, 0xA5),
    "pyramid-a-poison-5a": ("pyramid-a", 0, 0x5A),
    "pyramid-b-poison-5a": ("pyramid-b", 1, 0x5A),
}
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")
MASK64 = (1 << 64) - 1


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def require_no_addresses(value: Any, label: str) -> None:
    def visit(item: Any, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                if not lowered.startswith("address_free"):
                    require(
                        not any(
                            marker in lowered
                            for marker in ("pointer", "gpu_va", "device_address")
                        ),
                        f"{label} contains address field at {location}.{key}",
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")
        elif isinstance(item, str):
            require(
                ADDRESS_PATTERN.search(item) is None,
                f"{label} contains address-like value at {location}",
            )

    visit(value, "$")


def with_root(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(canonical_bytes(result))
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    require_no_addresses(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def rgb_bits(fixture: int, level: int, x: int, y: int, channel: int) -> int:
    mixed = splitmix64(
        0xD6E8FEB86659FD93
        ^ ((fixture + 1) * 0xA0761D6478BD642F)
        ^ ((level + 1) * 0xE7037ED1A0B428DB)
        ^ (y * 0x8EBC6AF09C88C6E3)
        ^ (x * 0x589965CC75374CC3)
        ^ ((channel + 1) * 0x1D8E4E27C47D124F)
    )
    sign = 0x8000 if mixed >> 63 else 0
    exponent = 13 + ((mixed >> 19) & 3)
    return sign | (exponent << 10) | (mixed & 0x03FF)


def has_support(fixture: int, level: int, x: int, y: int) -> bool:
    if fixture == 0:
        return ((x * 3 + y * 5 + level * 7) % 11) < 4
    return ((x * 7 + y * 2 + level * 3 + 1) % 13) < 5


def alpha_bits(fixture: int, level: int, x: int, y: int) -> int:
    return (0x3000, 0x3400, 0x3800, 0x3C00)[
        (fixture + level + x * 3 + y * 5) & 3
    ]


def generate_pyramid(fixture: int, poison: int) -> tuple[bytes, ...]:
    result: list[bytes] = []
    for level, (width, height, size) in enumerate(
        zip(WIDTHS, HEIGHTS, LEVEL_BYTES)
    ):
        data = bytearray(size)
        for y in range(height):
            for x in range(width):
                support = has_support(fixture, level, x, y)
                rgb = tuple(
                    rgb_bits(fixture, level, x, y, channel)
                    if support
                    else poison * 0x0101
                    for channel in range(3)
                )
                alpha = alpha_bits(fixture, level, x, y)
                if not support:
                    alpha = (
                        0xB400
                        if level == 0 and ((x + y + fixture) & 1) != 0
                        else 0
                    )
                struct.pack_into("<4H", data, (y * width + x) * 8, *rgb, alpha)
        result.append(bytes(data))
    return tuple(result)


def mismatch_count(left: bytes, right: bytes) -> int:
    require(len(left) == len(right), "mismatch comparison length drift")
    return sum(a != b for a, b in zip(left, right))


def half_mismatch_count(left: bytes, right: bytes) -> int:
    require(len(left) == len(right) and len(left) % 2 == 0, "half comparison drift")
    return sum(
        left[index : index + 2] != right[index : index + 2]
        for index in range(0, len(left), 2)
    )


def coverage(before: bytes, after: bytes) -> dict[str, int]:
    require(len(before) == len(after) == LEVEL_BYTES[0], "level0 size drift")
    counts = {
        "preserved_pixels": 0,
        "preservation_mismatched_bytes": 0,
        "hole_pixels": 0,
        "zero_alpha_holes": 0,
        "negative_alpha_holes": 0,
        "changed_hole_pixels": 0,
        "positive_output_holes": 0,
    }
    for offset in range(0, len(before), 8):
        alpha = struct.unpack_from("<H", before, offset + 6)[0]
        positive = (alpha & 0x8000) == 0 and (alpha & 0x7FFF) != 0
        if positive:
            counts["preserved_pixels"] += 1
            counts["preservation_mismatched_bytes"] += mismatch_count(
                before[offset : offset + 8], after[offset : offset + 8]
            )
            continue
        counts["hole_pixels"] += 1
        if (alpha & 0x7FFF) == 0:
            counts["zero_alpha_holes"] += 1
        else:
            counts["negative_alpha_holes"] += 1
        counts["changed_hole_pixels"] += int(
            before[offset : offset + 8] != after[offset : offset + 8]
        )
        output_alpha = struct.unpack_from("<H", after, offset + 6)[0]
        counts["positive_output_holes"] += int(
            (output_alpha & 0x8000) == 0 and (output_alpha & 0x7FFF) != 0
        )
    return counts


def validate_run(
    root: Path, run_id: str, expected_backend: str
) -> tuple[dict[str, Any], bytes]:
    fixture_name, fixture, poison = CASES[run_id]
    run_root = root / run_id
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    require_no_addresses(receipt, f"{run_id} receipt")
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id} schema drift")
    require(receipt.get("backend") == expected_backend, f"{run_id} backend drift")
    require(receipt.get("run_id") == run_id, f"{run_id} run-id drift")
    require(receipt.get("fixture") == fixture_name, f"{run_id} fixture drift")
    require(receipt.get("poison_byte") == poison, f"{run_id} poison drift")
    require(receipt.get("result") == "pass", f"{run_id} did not pass")
    require(receipt.get("mode") == 0, f"{run_id} escaped mode 0")
    require(receipt.get("grid") == [20, 12, 1], f"{run_id} grid drift")
    require(receipt.get("block") == [256, 1, 1], f"{run_id} block drift")
    require(
        receipt.get("input_generator") == GENERATOR,
        f"{run_id} generator drift",
    )
    module_field = "module_sha256" if expected_backend == "rtx-cuda-driver" else "source_module_sha256"
    require(receipt.get(module_field) == MODULE_SHA, f"{run_id} module hash drift")
    require(
        receipt.get("raw_parameter_sha256") == RAW_PARAMETER_SHA,
        f"{run_id} raw parameter hash drift",
    )
    require(
        receipt.get("address_free_parameter_sha256") == CANONICAL_PARAMETER_SHA,
        f"{run_id} canonical parameter hash drift",
    )
    require(
        receipt.get("external_expected_tensors_consumed") is False,
        f"{run_id} expected-output ingress",
    )
    require(
        receipt.get("archived_intermediates_consumed") is False,
        f"{run_id} archived intermediate ingress",
    )
    require(
        receipt.get("address_fields_recorded") is False,
        f"{run_id} address recording drift",
    )
    expected = generate_pyramid(fixture, poison)
    actual: list[bytes] = []
    for level in range(5):
        before_path = run_root / f"level{level}-before.bin"
        after_path = run_root / f"level{level}-after.bin"
        before = before_path.read_bytes()
        after = after_path.read_bytes()
        require(len(before) == LEVEL_BYTES[level], f"{run_id} level{level} before size drift")
        require(len(after) == LEVEL_BYTES[level], f"{run_id} level{level} after size drift")
        require(before == expected[level], f"{run_id} generated level{level} input drift")
        if level != 0:
            require(after == before, f"{run_id} readonly level{level} was written")
        actual.append(after)
    require(
        receipt.get("level0_before_sha256") == sha256_bytes(expected[0]),
        f"{run_id} level0-before receipt hash drift",
    )
    require(
        receipt.get("level0_after_sha256") == sha256_bytes(actual[0]),
        f"{run_id} output receipt hash drift",
    )
    measured = coverage(expected[0], actual[0])
    for field, value in measured.items():
        require(receipt.get(field) == value, f"{run_id} {field} drift")
    require(measured["preserved_pixels"] > 0, f"{run_id} no preserved pixels")
    require(measured["hole_pixels"] > 0, f"{run_id} no replacement pixels")
    require(measured["zero_alpha_holes"] > 0, f"{run_id} no zero-alpha holes")
    require(measured["negative_alpha_holes"] > 0, f"{run_id} no negative-alpha holes")
    require(
        measured["preservation_mismatched_bytes"] == 0,
        f"{run_id} identity preservation failed",
    )
    require(
        measured["changed_hole_pixels"] == measured["hole_pixels"],
        f"{run_id} conditional write coverage failed",
    )
    require(
        measured["positive_output_holes"] == measured["hole_pixels"],
        f"{run_id} unresolved output holes",
    )
    require(
        receipt.get("readonly_level_mismatched_bytes") == 0,
        f"{run_id} readonly mismatch receipt drift",
    )
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id} canary drift")
    return receipt, actual[0]


def validate_backend(root: Path, backend: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    runs: list[dict[str, Any]] = []
    outputs: dict[str, bytes] = {}
    process_ids: list[int] = []
    for run_id in CASES:
        receipt, output = validate_run(root, run_id, expected_backend)
        process_id = receipt.get("process_id")
        require(isinstance(process_id, int) and process_id > 0, f"{run_id} process id drift")
        process_ids.append(process_id)
        outputs[run_id] = output
        runs.append(
            {
                "run_id": run_id,
                "fixture": receipt["fixture"],
                "poison_byte": receipt["poison_byte"],
                "input_sha256": receipt["level0_before_sha256"],
                "output_sha256": receipt["level0_after_sha256"],
                "run_receipt_sha256": sha256_file(root / run_id / "run-receipt.json"),
                "preserved_pixels": receipt["preserved_pixels"],
                "hole_pixels": receipt["hole_pixels"],
            }
        )
    require(len(set(process_ids)) == 4, f"{backend} fresh process gate failed")
    for fixture in ("pyramid-a", "pyramid-b"):
        matching = [run for run in runs if run["fixture"] == fixture]
        require(len(matching) == 2, f"{backend} fixture coverage drift: {fixture}")
        require(
            len({run["poison_byte"] for run in matching}) == 2,
            f"{backend} poison coverage drift: {fixture}",
        )
        require(
            len({run["input_sha256"] for run in matching}) == 2,
            f"{backend} poison did not reach fixture inputs: {fixture}",
        )
        require(
            len({run["output_sha256"] for run in matching}) == 1,
            f"{backend} two-poison output invariance failed: {fixture}",
        )

    common: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": backend,
        "selected_launch_token": 86,
        "module_token": 19,
        "module_sha256": MODULE_SHA,
        "raw_parameter_sha256": RAW_PARAMETER_SHA,
        "address_free_parameter_sha256": CANONICAL_PARAMETER_SHA,
        "mode": 0,
        "input_generator": GENERATOR,
        "fresh_process_count": 4,
        "independent_fixture_count": 2,
        "poison_values": [90, 165],
        "two_poison_output_invariance": True,
        "readonly_levels_immutable": True,
        "identity_preservation": True,
        "conditional_write_coverage": True,
        "canaries_intact": True,
        "runs": runs,
        "external_expected_tensors_consumed": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    if backend == "rtx":
        rig_path = root / "rig-receipt.json"
        rig = load_json(rig_path)
        require_no_addresses(rig, "RTX rig receipt")
        require(rig.get("schema") == RTX_RIG_SCHEMA, "RTX rig schema drift")
        require(rig.get("result") == "pass", "RTX rig did not pass")
        require(rig.get("fresh_process_count") == 4, "RTX rig fresh process drift")
        require(rig.get("mode_1_rejected") is True, "RTX mode-1 rejection drift")
        require(
            rig.get("bfind_constant_proof", {}).get("input_u32") == 0x20000000
            and rig.get("bfind_constant_proof", {}).get("output_u32") == 2
            and rig.get("bfind_constant_proof", {}).get("result") == "pass",
            "RTX bfind constant proof drift",
        )
        sidecar = (root / "rig-receipt.sha256").read_text().split()[0]
        require(sidecar == sha256_file(rig_path), "RTX rig receipt sidecar drift")
        common.update(
            {
                "rig_receipt_sha256": sidecar,
                "bfind_constant_input_u32": 0x20000000,
                "bfind_constant_output_u32": 2,
                "mode_1_rejected": True,
            }
        )
    else:
        build_path = root / "mac-build-receipt.json"
        build = load_json(build_path)
        require_no_addresses(build, "Mac build receipt")
        require(build.get("schema") == MAC_BUILD_SCHEMA, "Mac build schema drift")
        require(build.get("result") == "pass", "Mac build did not pass")
        require(build.get("mode_1_rejected") is True, "Mac mode-1 rejection drift")
        require(build.get("architecture_rom_sha256") == ROM_SHA, "Mac ROM hash drift")
        sidecar = (root / "mac-build-receipt.sha256").read_text().split()[0]
        require(sidecar == sha256_file(build_path), "Mac build receipt sidecar drift")
        negative = (root / "negative-precise" / "level0-after.bin").read_bytes()
        exact = outputs["pyramid-a-poison-a5"]
        negative_bytes = mismatch_count(negative, exact)
        negative_halves = half_mismatch_count(negative, exact)
        require(negative_bytes > 0, "ordinary Metal division counterexample vanished")
        require(negative_halves > 0, "ordinary Metal division half counterexample vanished")
        common.update(
            {
                "build_receipt_sha256": sidecar,
                "architecture_rom_sha256": ROM_SHA,
                "mode_1_rejected": True,
                "ordinary_division_output_sha256": sha256_bytes(negative),
                "ordinary_division_mismatched_bytes": negative_bytes,
                "ordinary_division_mismatched_half_values": negative_halves,
            }
        )
    receipt = with_root(common, "receipt_root_sha256")
    require_no_addresses(receipt, f"{backend} backend receipt")
    return receipt, outputs


def verify_final(rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx, rtx_outputs = validate_backend(rtx_root, "rtx")
    mac, mac_outputs = validate_backend(mac_root, "mac")
    total_mismatches = 0
    fixtures: list[dict[str, Any]] = []
    for fixture in ("pyramid-a", "pyramid-b"):
        run_ids = [run_id for run_id, values in CASES.items() if values[0] == fixture]
        fixture_mismatches = sum(
            mismatch_count(rtx_outputs[run_id], mac_outputs[run_id])
            for run_id in run_ids
        )
        total_mismatches += fixture_mismatches
        fixtures.append(
            {
                "fixture": fixture,
                "output_bytes": LEVEL_BYTES[0],
                "output_sha256": sha256_bytes(rtx_outputs[run_ids[0]]),
                "poison_values": [90, 165],
                "rtx_mac_mismatched_bytes": fixture_mismatches,
            }
        )
    require(total_mismatches == 0, "RTX/Mac output parity failed")
    value: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_launch_token": 86,
        "module_token": 19,
        "module_sha256": MODULE_SHA,
        "raw_parameter_sha256": RAW_PARAMETER_SHA,
        "address_free_parameter_sha256": CANONICAL_PARAMETER_SHA,
        "abi": {
            "packed_bytes": 80,
            "buffers": ["level0", "level1", "level2", "level3", "level4"],
            "level_dimensions": [[320, 180], [160, 90], [80, 45], [40, 23], [20, 12]],
            "width": 320,
            "height": 180,
            "mode": 0,
            "texture_handle": 0,
            "surface_handle": 0,
            "origin": [0, 0],
        },
        "input_generator": GENERATOR,
        "fixtures": fixtures,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_fresh_process_count": 4,
        "mac_fresh_process_count": 4,
        "two_poison_values": [90, 165],
        "rtx_mac_mismatched_bytes": total_mismatches,
        "ordinary_division_mismatched_bytes": mac["ordinary_division_mismatched_bytes"],
        "ordinary_division_mismatched_half_values": mac[
            "ordinary_division_mismatched_half_values"
        ],
        "architecture_rom": {
            "kind": "sm120-mufu-rcp-2bit-correction",
            "bytes": 2097152,
            "sha256": ROM_SHA,
            "model_values": 0,
            "archived_intermediates": 0,
        },
        "gates": {
            "exact_selected_module_execution": True,
            "bfind_constant_proven": True,
            "fresh_process_repeat": True,
            "two_poison_invariance": True,
            "independent_pyramids": True,
            "readonly_levels_immutable": True,
            "identity_preservation": True,
            "conditional_write_coverage": True,
            "zero_and_negative_alpha_replaced": True,
            "canaries_intact": True,
            "ordinary_division_rejected": True,
            "nonzero_modes_rejected": True,
            "independent_backend_inputs": True,
            "no_expected_output_ingress": True,
            "no_archived_intermediate_ingress": True,
            "address_free": True,
        },
        "claims": {
            "launch86_mode0_dependency_closed_parity": True,
            "other_modes": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    return with_root(value, "receipt_root_sha256")


def mutate_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    require(bool(data), f"cannot mutate empty file: {path}")
    data[len(data) // 2] ^= 1
    path.write_bytes(data)


def run_negative_audit(rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    mutations: list[dict[str, Any]] = []

    def exercise(name: str, target: str, action: Any, expected: str) -> None:
        with tempfile.TemporaryDirectory(prefix="fg-l86-negative-") as temporary:
            temp = Path(temporary)
            rtx_copy = temp / "rtx"
            mac_copy = temp / "mac"
            shutil.copytree(rtx_root, rtx_copy)
            shutil.copytree(mac_root, mac_copy)
            action(rtx_copy if target == "rtx" else mac_copy)
            try:
                verify_final(rtx_copy, mac_copy)
            except VerificationError as error:
                observed = str(error)
                require(expected in observed, f"{name} unexpected error: {observed}")
                mutations.append(
                    {
                        "mutation": name,
                        "expected_error_fragment": expected,
                        "observed_error": observed,
                        "receipt_created": False,
                        "result": "pass",
                    }
                )
                return
            raise VerificationError(f"{name} mutation was accepted")

    exercise(
        "rtx-output-byte",
        "rtx",
        lambda root: mutate_byte(root / "pyramid-a-poison-a5" / "level0-after.bin"),
        "output receipt hash drift",
    )
    exercise(
        "rtx-generated-input-byte",
        "rtx",
        lambda root: mutate_byte(root / "pyramid-b-poison-a5" / "level2-before.bin"),
        "generated level2 input drift",
    )
    exercise(
        "rtx-readonly-level-write",
        "rtx",
        lambda root: mutate_byte(root / "pyramid-a-poison-5a" / "level3-after.bin"),
        "readonly level3 was written",
    )
    exercise(
        "mac-output-byte",
        "mac",
        lambda root: mutate_byte(root / "pyramid-b-poison-5a" / "level0-after.bin"),
        "output receipt hash drift",
    )

    def duplicate_process(root: Path) -> None:
        first = load_json(root / "pyramid-a-poison-a5" / "run-receipt.json")
        second_path = root / "pyramid-b-poison-a5" / "run-receipt.json"
        second = load_json(second_path)
        second["process_id"] = first["process_id"]
        second_path.write_text(json.dumps(second), encoding="utf-8")

    exercise("rtx-duplicate-process", "rtx", duplicate_process, "fresh process gate failed")

    def module_identity(root: Path) -> None:
        path = root / "pyramid-a-poison-5a" / "run-receipt.json"
        value = load_json(path)
        value["module_sha256"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")

    exercise("rtx-module-identity", "rtx", module_identity, "module hash drift")

    def archived_ingress(root: Path) -> None:
        path = root / "pyramid-b-poison-5a" / "run-receipt.json"
        value = load_json(path)
        value["archived_intermediates_consumed"] = True
        path.write_text(json.dumps(value), encoding="utf-8")

    exercise("rtx-archived-ingress", "rtx", archived_ingress, "archived intermediate ingress")

    def address_field(root: Path) -> None:
        path = root / "pyramid-a-poison-a5" / "run-receipt.json"
        value = load_json(path)
        value["gpu_va"] = "0x123456789abcdef0"
        path.write_text(json.dumps(value), encoding="utf-8")

    exercise("address-field", "rtx", address_field, "contains address field")

    def erase_div_counterexample(root: Path) -> None:
        source = root / "pyramid-a-poison-a5" / "level0-after.bin"
        destination = root / "negative-precise" / "level0-after.bin"
        destination.write_bytes(source.read_bytes())

    exercise(
        "ordinary-division-counterexample",
        "mac",
        erase_div_counterexample,
        "ordinary Metal division counterexample vanished",
    )

    def mode_rejection(root: Path) -> None:
        path = root / "mac-build-receipt.json"
        value = load_json(path)
        value["mode_1_rejected"] = False
        path.write_text(json.dumps(value), encoding="utf-8")
        (root / "mac-build-receipt.sha256").write_text(
            sha256_file(path) + "  mac-build-receipt.json\n", encoding="utf-8"
        )

    exercise("mac-mode-rejection", "mac", mode_rejection, "mode-1 rejection drift")
    value = {
        "schema": "neuralstudio.fg.postprocess-l86-negative-audit.v1",
        "mutation_count": len(mutations),
        "mutations": mutations,
        "all_rejected": True,
    }
    return with_root(value, "audit_root_sha256")


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
    final.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--rtx-root", type=Path, required=True)
    audit.add_argument("--mac-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "backend":
            value, _ = validate_backend(args.root, args.backend)
        elif args.command == "final":
            value = verify_final(args.rtx_root, args.mac_root)
        else:
            value = run_negative_audit(args.rtx_root, args.mac_root)
        write_json(args.output, value)
        print(f"receipt={args.output}")
        print("result=pass")
        return 0
    except VerificationError as error:
        print(f"verification_error={error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
