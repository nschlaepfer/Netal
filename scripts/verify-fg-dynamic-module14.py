#!/usr/bin/env python3
"""Fail-closed receipts for target-resolution FG module14/L9."""

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


MODULE_SHA = "540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
JOB_ID = "fg-dynamic-module14-20260904T151149Z"
RUN_SCHEMA = "neuralstudio.fg.dynamic-module14-run.v1"
RESOLUTIONS = {
    "720p": {
        "extent": [640, 360], "grid": [40, 23, 1],
        "parameter_sha256": "e1f743baeb62a509d45537354a13acf4a8c3c61f6ea8f08d4469d46f70365c4f",
        "salt": 0x7207207207207207,
    },
    "1080p": {
        "extent": [960, 540], "grid": [60, 34, 1],
        "parameter_sha256": "36bfa4cc6d18f4fa17d4e0191bf4c4e6abdfebb07e865ff540d447319146fbfe",
        "salt": 0x1080108010801080,
    },
    "1440p": {
        "extent": [1280, 720], "grid": [80, 45, 1],
        "parameter_sha256": "7c9719e0b5d5ee388bc0d40de8c17f52a3cedd2c60125804b07ad279817e80d6",
        "salt": 0x1440144014401440,
    },
    "4k": {
        "extent": [960, 540], "grid": [60, 34, 1],
        "parameter_sha256": "36bfa4cc6d18f4fa17d4e0191bf4c4e6abdfebb07e865ff540d447319146fbfe",
        "salt": 0x4B4B4B4B4B4B4B4B,
    },
}
POISONS = {"a5": 0xA5, "5a": 0x5A}
ORDINALS = ("a", "b")
MASK64 = (1 << 64) - 1
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")
VALUES = (
    0x0000, 0x8000, 0x0001, 0x8001, 0x1400, 0x9400, 0x2000,
    0xA000, 0x2800, 0xA800, 0x3000, 0xB000, 0x3400, 0xB400,
    0x3555, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000,
    0x4200, 0x4400, 0x7BFF, 0xFBFF,
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def rooted(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["receipt_root_sha256"] = sha(canonical(result))
    return result


def verify_root(value: dict[str, Any], label: str) -> None:
    body = copy.deepcopy(value)
    observed = body.pop("receipt_root_sha256", None)
    require(observed == sha(canonical(body)), f"{label}: receipt root drift")
    address_free(value, label)


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


def mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def active_input(name: str) -> bytes:
    spec = RESOLUTIONS[name]
    width, height = spec["extent"]
    result = bytearray(width * height * 8)
    for index in range(width * height * 4):
        value = VALUES[mix((spec["salt"] +
                            index * 0xD1B54A32D192ED03) & MASK64) % len(VALUES)]
        struct.pack_into("<H", result, index * 2, value)
    struct.pack_into("<4H", result, 0, 0x0000, 0x0001, 0x1400, 0x3C00)
    return bytes(result)


def guarded_input(name: str, poison: int) -> bytes:
    spec = RESOLUTIONS[name]
    width, _ = spec["extent"]
    result = bytearray(active_input(name))
    start = len(result)
    result.extend(bytes(width * 8))
    for index in range(start, len(result)):
        result[index] = poison ^ (
            (mix((spec["salt"] + index) & MASK64) >> 56) & 0xFF)
    return bytes(result)


def dead_mask(name: str, poison: int) -> bytes:
    spec = RESOLUTIONS[name]
    width, height = spec["extent"]
    return bytes(
        poison ^ ((mix((0x13198A2E03707344 + spec["salt"] + index) & MASK64)
                   >> 56) & 0xFF)
        for index in range(width * (height + 1))
    )


def source_identity(repository: Path) -> dict[str, str]:
    paths = (
        "src/fg_dynamic_module14/contract.h",
        "src/fg_dynamic_module14/dynamic_module14.metal",
        "src/fg_dynamic_module14/macos_probe.mm",
        "src/fg_dynamic_module14/rtx_oracle.cpp",
        "scripts/build-fg-dynamic-module14-rtx.ps1",
        "scripts/build-macos-fg-dynamic-module14.sh",
        "scripts/verify-fg-dynamic-module14.py",
        "scripts/test-fg-dynamic-module14-mutations.py",
        "src/fg_conv/rtx_oracle.cpp",
        "src/fg_conv/macos_probe.mm",
    )
    return {path: file_sha(repository / path) for path in paths}


def run_id(name: str, poison: str, ordinal: str) -> str:
    return f"{name}-poison-{poison}-process-{ordinal}"


def validate_rig(rig: dict[str, Any], sources: dict[str, str]) -> None:
    require(rig.get("schema") == "neuralstudio.fg.dynamic-module14-rig.v1" and
            rig.get("job_id") == JOB_ID and rig.get("attempt_id") == "oracle-v2",
            "rig identity drift")
    require(rig.get("module_sha256") == MODULE_SHA and
            rig.get("source_sha256") ==
            sources["src/fg_dynamic_module14/rtx_oracle.cpp"] and
            rig.get("contract_sha256") ==
            sources["src/fg_dynamic_module14/contract.h"] and
            rig.get("base_source_sha256") ==
            sources["src/fg_conv/rtx_oracle.cpp"] and
            rig.get("harness_sha256") ==
            sources["scripts/build-fg-dynamic-module14-rtx.ps1"],
            "rig source identity drift")
    require(rig.get("raw_parameter_sha256") == {
        name: spec["parameter_sha256"] for name, spec in RESOLUTIONS.items()},
        "rig raw parameter identity drift")
    require(rig.get("run_count") == 16 and
            rig.get("fresh_process_count") == 16 and
            rig.get("true_typed_cuda_arrays") is True and
            rig.get("exit_code") == 0 and
            isinstance(rig.get("started_utc"), str) and
            isinstance(rig.get("completed_utc"), str),
            "rig execution receipt drift")
    require(rig.get("expected_or_captured_output_ingress") is False and
            rig.get("result") == "pass", "rig forbidden ingress")
    address_free(rig, "rig")


def validate_backend(kind: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = "rtx-cuda-driver-typed-arrays" if kind == "rtx" else "mac-metal"
    process_ids: set[int] = set()
    output_summary: dict[str, Any] = {}
    receipt_hashes: dict[str, str] = {}
    for name, spec in RESOLUTIONS.items():
        active = active_input(name)
        observed_hashes: set[str] = set()
        poison_resource_hashes: set[str] = set()
        poison_dead_hashes: set[str] = set()
        for poison_name, poison in POISONS.items():
            for ordinal in ORDINALS:
                identity = run_id(name, poison_name, ordinal)
                run_root = root / identity
                receipt_path = run_root / "run-receipt.json"
                receipt = load(receipt_path)
                require(receipt.get("schema") == RUN_SCHEMA and
                        receipt.get("backend") == backend and
                        receipt.get("run_id") == identity and
                        receipt.get("resolution") == name and
                        receipt.get("poison_byte") == poison,
                        f"{identity}: run identity drift")
                require(receipt.get("module_sha256") == MODULE_SHA and
                        receipt.get("extent") == spec["extent"] and
                        receipt.get("grid") == spec["grid"] and
                        receipt.get("block") == [16, 16, 1] and
                        receipt.get("packed_parameter_bytes") == 32 and
                        receipt.get("raw_parameter_sha256") ==
                        spec["parameter_sha256"], f"{identity}: ABI drift")
                if kind == "rtx":
                    require(receipt.get("function_name") == "main_kernel",
                            f"{identity}: function drift")
                else:
                    require(receipt.get("metal_function") ==
                            "fg_dynamic_module14" and
                            receipt.get("architecture_rom_sha256") == ROM_SHA and
                            receipt.get("surface_half_rounding") == "toward-zero",
                            f"{identity}: Metal identity drift")
                    require(receipt.get("buffer_canary_mismatched_bytes") == 0 and
                            receipt.get("approx_division_probe_bits") ==
                            0x39AAAD56 and
                            receipt.get("precise_division_probe_bits") ==
                            0x39AAAD55 and
                            receipt.get("rz_store_probe_half_bits") == 0x3C00 and
                            receipt.get("rne_store_probe_half_bits") == 0x3C01 and
                            receipt.get("numeric_distinctions_observed") is True,
                            f"{identity}: numeric probe drift")
                input_resource = guarded_input(name, poison)
                dead_resource = dead_mask(name, poison)
                require(receipt.get("input_active_sha256") == sha(active) and
                        receipt.get("input_resource_sha256") ==
                        sha(input_resource) and
                        receipt.get("input_post_sha256") ==
                        sha(input_resource) and
                        receipt.get("dead_resource_sha256") ==
                        sha(dead_resource) and
                        receipt.get("dead_post_sha256") == sha(dead_resource),
                        f"{identity}: input provenance drift")
                output_bytes = spec["extent"][0] * spec["extent"][1] * 4
                require(receipt.get("output_bytes") == output_bytes and
                        isinstance(receipt.get("output_sha256"), str),
                        f"{identity}: output receipt drift")
                require(receipt.get("input_mismatched_bytes") == 0 and
                        receipt.get("output_guard_mismatched_bytes") == 0,
                        f"{identity}: integrity gate failed")
                full = poison_name == "a5" and ordinal == "a"
                require(receipt.get("artifacts_written") is full,
                        f"{identity}: artifact policy drift")
                require(receipt.get("expected_or_captured_output_ingress") is False and
                        receipt.get("address_fields_recorded") is False and
                        receipt.get("result") == "pass",
                        f"{identity}: forbidden ingress")
                require(isinstance(receipt.get("process_id"), int),
                        f"{identity}: missing process identity")
                address_free(receipt, identity)
                process_ids.add(receipt["process_id"])
                observed_hashes.add(receipt["output_sha256"])
                poison_resource_hashes.add(receipt["input_resource_sha256"])
                poison_dead_hashes.add(receipt["dead_resource_sha256"])
                receipt_hashes[identity] = file_sha(receipt_path)
                if full:
                    output = (run_root / "output.bin").read_bytes()
                    input_after = (run_root / "input-after.bin").read_bytes()
                    dead_after = (run_root / "dead-mask-after.bin").read_bytes()
                    require(len(output) == output_bytes and
                            sha(output) == receipt["output_sha256"],
                            f"{identity}: output artifact drift")
                    require(input_after == input_resource and
                            dead_after == dead_resource,
                            f"{identity}: immutable artifact drift")
                    if kind == "mac":
                        numeric = (run_root / "numeric-probe-output.bin").read_bytes()
                        require(struct.unpack("<4I", numeric) ==
                                (0x39AAAD56, 0x39AAAD55, 0x3C00, 0x3C01),
                                f"{identity}: numeric artifact drift")
        require(len(observed_hashes) == 1,
                f"{name}: fresh-process/two-poison output drift")
        require(len(poison_resource_hashes) == 2 and len(poison_dead_hashes) == 2,
                f"{name}: poison diversity drift")
        output_summary[name] = {
            "extent": spec["extent"],
            "grid": spec["grid"],
            "bytes": spec["extent"][0] * spec["extent"][1] * 4,
            "sha256": next(iter(observed_hashes)),
        }
    require(len(process_ids) == 16, f"{kind.upper()} process count drift")
    result = rooted({
        "schema": f"neuralstudio.fg.dynamic-module14-{kind}-backend.v1",
        "backend": backend,
        "module_sha256": MODULE_SHA,
        "fresh_process_count": 16,
        "outputs": output_summary,
        "run_receipt_sha256": receipt_hashes,
        "repeat_and_dual_poison_exact": True,
        "full_write_coverage_from_dual_poison": True,
        "input_and_dead_resource_immutable": True,
        "guard_rows_intact": True,
        "true_typed_resources": True,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    return result, output_summary


def build(args: argparse.Namespace) -> None:
    repository = args.repository.resolve()
    sources = source_identity(repository)
    rig_path = args.rtx / "rig-receipt.json"
    rig = load(rig_path)
    validate_rig(rig, sources)
    rtx, rtx_outputs = validate_backend("rtx", args.rtx.resolve())
    mac, mac_outputs = validate_backend("mac", args.mac.resolve())
    require(rtx_outputs == mac_outputs, "backend output disagreement")
    compared_bytes = sum(item["bytes"] for item in rtx_outputs.values())
    shape_sources = {
        name: file_sha(repository /
                       f"evidence/specs/fg-resolution-shapes-v1/shape-{name}.json")
        for name in RESOLUTIONS
    }
    abi = rooted({
        "schema": "neuralstudio.fg.dynamic-module14-abi.v1",
        "canonical_launch": 9,
        "large_trace_launch": 11,
        "module_token": 14,
        "module_sha256": MODULE_SHA,
        "function_name": "main_kernel",
        "packed_parameter_bytes": 32,
        "fields": [
            {"offset": 0, "bytes": 8, "role": "RGBA16_FLOAT point-border input texture"},
            {"offset": 8, "bytes": 8, "role": "ABI-present dead R8_UNORM texture", "ptx_loaded": False},
            {"offset": 16, "bytes": 8, "role": "RG16_FLOAT output surface"},
            {"offset": 24, "bytes": 4, "role": "internal width"},
            {"offset": 28, "bytes": 4, "role": "internal height"},
        ],
        "block": [16, 16, 1],
        "resolutions": {
            name: {
                "extent": spec["extent"], "grid": spec["grid"],
                "raw_parameter_sha256": spec["parameter_sha256"],
                "shape_receipt_sha256": shape_sources[name],
            } for name, spec in RESOLUTIONS.items()
        },
        "four_k_internal_extent_is_not_presentation_extent": True,
        "opaque_live_handles_recorded": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    implementation = rooted({
        "schema": "neuralstudio.fg.dynamic-module14-implementation.v1",
        "source_sha256": sources,
        "architecture_rom_sha256": ROM_SHA,
        "selected_add_order": "B + (R + G)",
        "selected_division": "SM120 approximate reciprocal product by 3",
        "selected_saturation": "unit interval",
        "selected_surface_half_rounding": "toward-zero",
        "numeric_probe": {
            "approx_division_bits": 0x39AAAD56,
            "precise_division_bits": 0x39AAAD55,
            "rz_half_bits": 0x3C00,
            "rne_half_bits": 0x3C01,
        },
        "compile_policy": "Metal 3.1, fast math disabled, FP contraction disabled",
        "expected_output_embedded": False,
        "captured_output_embedded": False,
        "result": "pass",
    })
    counterexamples = rooted({
        "schema": "neuralstudio.fg.dynamic-module14-counterexamples.v1",
        "items": [
            {"name": "presentation-size substitution", "result": "rejected",
             "reason": "the live 4K trace selects a 960x540 internal surface"},
            {"name": "precise division substitution", "result": "rejected",
             "reason": "the device probe differs from reciprocal-product division by one FP32 ULP"},
            {"name": "default half conversion substitution", "result": "rejected",
             "reason": "the device probe distinguishes RNE from the selected RZ store"},
            {"name": "RTX oracle-v1 provenance", "result": "superseded",
             "reason": "it omitted the included base-source hash and job timing"},
            {"name": "quoted PowerShell file path", "result": "rejected",
             "reason": "the first invocation passed literal quotes to the file parameter"},
            {"name": "partial recursive retrieval", "result": "rejected",
             "reason": "large artifacts were re-fetched individually and hash verified"},
        ],
        "all_failures_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.dynamic-module14-accepted.v1",
        "selected_unit": "canonical L9, module14, main_kernel",
        "module_sha256": MODULE_SHA,
        "resolutions": list(RESOLUTIONS),
        "processes_per_backend": 16,
        "abi_receipt_root_sha256": abi["receipt_root_sha256"],
        "implementation_receipt_root_sha256":
            implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": counterexamples["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "rtx_mac_compared_bytes": compared_bytes,
        "rtx_mac_mismatched_bytes": 0,
        "output_sha256_by_resolution": {
            name: value["sha256"] for name, value in rtx_outputs.items()},
        "claims": {
            "module14_four_resolution_exact": True,
            "approximate_division_and_rz_store_preserved": True,
            "full_active_region_write_coverage": True,
            "input_and_dead_resource_immutable": True,
            "connected_early_graph": False,
            "complete_generated_frame": False,
            "Netal_integration": False,
        },
        "scope": "standalone reset projection L9/module14 at four live target configurations",
        "result": "pass",
    })
    outputs = {
        "abi-receipt.json": abi,
        "implementation-receipt.json": implementation,
        "counterexamples.json": counterexamples,
        "rtx-backend-receipt.json": rtx,
        "mac-backend-receipt.json": mac,
        "accepted-receipt.json": accepted,
    }
    for name, value in outputs.items():
        write(args.output / name, value)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_compared_bytes={compared_bytes}")
    print("verification_result=pass")


def verify_evidence(repository: Path, evidence: Path,
                    require_negative: bool) -> None:
    names = (
        "abi-receipt.json", "implementation-receipt.json",
        "counterexamples.json", "rtx-backend-receipt.json",
        "mac-backend-receipt.json", "accepted-receipt.json",
    )
    values = {name: load(evidence / name) for name in names}
    for name, value in values.items():
        verify_root(value, name)
    abi = values["abi-receipt.json"]
    implementation = values["implementation-receipt.json"]
    counterexamples = values["counterexamples.json"]
    rtx = values["rtx-backend-receipt.json"]
    mac = values["mac-backend-receipt.json"]
    accepted = values["accepted-receipt.json"]
    require(abi.get("packed_parameter_bytes") == 32 and
            abi.get("block") == [16, 16, 1], "packed ABI drift")
    require(abi.get("resolutions") == {
        name: {
            "extent": spec["extent"], "grid": spec["grid"],
            "raw_parameter_sha256": spec["parameter_sha256"],
            "shape_receipt_sha256": file_sha(
                repository / f"evidence/specs/fg-resolution-shapes-v1/shape-{name}.json"),
        } for name, spec in RESOLUTIONS.items()
    }, "resolution ABI drift")
    require(abi.get("four_k_internal_extent_is_not_presentation_extent") is True and
            abi.get("opaque_live_handles_recorded") is False and
            abi.get("address_fields_recorded") is False,
            "ABI address/scope drift")
    require(implementation.get("source_sha256") == source_identity(repository),
            "source identity drift")
    require(implementation.get("architecture_rom_sha256") == ROM_SHA and
            implementation.get("selected_division") ==
            "SM120 approximate reciprocal product by 3" and
            implementation.get("selected_surface_half_rounding") ==
            "toward-zero" and
            implementation.get("numeric_probe") == {
                "approx_division_bits": 0x39AAAD56,
                "precise_division_bits": 0x39AAAD55,
                "rz_half_bits": 0x3C00,
                "rne_half_bits": 0x3C01,
            }, "numeric probe drift")
    require(counterexamples.get("all_failures_preserved") is True and
            len(counterexamples.get("items", [])) == 6,
            "counterexample ledger drift")
    require(rtx.get("fresh_process_count") == 16,
            "RTX process count drift")
    require(mac.get("fresh_process_count") == 16,
            "Mac process count drift")
    for label, backend in (("RTX", rtx), ("Mac", mac)):
        require(backend.get("module_sha256") == MODULE_SHA and
                backend.get("repeat_and_dual_poison_exact") is True and
                backend.get("full_write_coverage_from_dual_poison") is True and
                backend.get("input_and_dead_resource_immutable") is True and
                backend.get("guard_rows_intact") is True and
                backend.get("true_typed_resources") is True and
                backend.get("expected_or_captured_output_ingress") is False and
                backend.get("result") == "pass", f"{label} forbidden ingress")
    require(rtx.get("outputs") == mac.get("outputs"),
            "backend output disagreement")
    compared = sum(value["bytes"] for value in rtx["outputs"].values())
    require(accepted.get("abi_receipt_root_sha256") ==
            abi["receipt_root_sha256"] and
            accepted.get("implementation_receipt_root_sha256") ==
            implementation["receipt_root_sha256"] and
            accepted.get("counterexamples_root_sha256") ==
            counterexamples["receipt_root_sha256"] and
            accepted.get("rtx_backend_root_sha256") ==
            rtx["receipt_root_sha256"] and
            accepted.get("mac_backend_root_sha256") ==
            mac["receipt_root_sha256"], "backend link drift")
    require(accepted.get("rtx_mac_compared_bytes") == compared and
            accepted.get("rtx_mac_mismatched_bytes") == 0,
            "accepted mismatch count drift")
    claims = accepted.get("claims", {})
    require(claims.get("module14_four_resolution_exact") is True and
            claims.get("approximate_division_and_rz_store_preserved") is True and
            claims.get("full_active_region_write_coverage") is True and
            claims.get("input_and_dead_resource_immutable") is True and
            claims.get("connected_early_graph") is False and
            claims.get("complete_generated_frame") is False and
            claims.get("Netal_integration") is False,
            "scope drift")
    if require_negative:
        negative = load(evidence / "negative-audit.json")
        verify_root(negative, "negative-audit.json")
        require(negative.get("mutation_count", 0) >= 14 and
                negative.get("all_mutations_rejected") is True and
                negative.get("verifier_sha256") ==
                file_sha(repository / "scripts/verify-fg-dynamic-module14.py"),
                "negative audit drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--repository", type=Path, required=True)
    build_parser.add_argument("--rtx", type=Path, required=True)
    build_parser.add_argument("--mac", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    receipt_parser = commands.add_parser("receipt")
    receipt_parser.add_argument("--repository", type=Path, required=True)
    receipt_parser.add_argument("--evidence", type=Path, required=True)
    receipt_parser.add_argument("--skip-negative", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        else:
            verify_evidence(args.repository.resolve(), args.evidence.resolve(),
                            not args.skip_negative)
            print("verification_result=pass")
        return 0
    except (OSError, ValueError, KeyError, struct.error,
            json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
