#!/usr/bin/env python3
"""Fail-closed receipts for target-resolution FG L19/module5."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


MODULE_SHA = "3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
JOB_ID = "fg-dynamic-module5-20260904T152305Z"
RUN_SCHEMA = "neuralstudio.fg.dynamic-module5-run.v1"
RESOLUTIONS = {
    "720p": {
        "motion": [640, 360], "work": [640, 360], "tensor": [320, 192, 48],
        "grid": [40, 24, 1], "origin": [0, 6],
        "raw": "31452fa6c5411ee2372faa3fd7846d81add56279812d45300b28f4871d752938",
        "address_free": "b0c7faa583b12174bf72f7c03665fff788bc8b7298b754913d60beac4ce93b1a",
    },
    "1080p": {
        "motion": [960, 540], "work": [960, 540], "tensor": [480, 288, 48],
        "grid": [60, 36, 1], "origin": [0, 9],
        "raw": "a0bb155954511a39e1f567ef798ac9265ec1b91e5cc8dc12fe25a32e91d33f59",
        "address_free": "b2910867274200a5ad634e4ec86b8804491e8f02888f18d04d87a3c0088d2419",
    },
    "1440p": {
        "motion": [1280, 720], "work": [1280, 720], "tensor": [640, 384, 48],
        "grid": [80, 48, 1], "origin": [0, 12],
        "raw": "f32b37ca6be9d175995b367f9475cf2c349776118985f49207ec6fe3957fcddb",
        "address_free": "6bd52da18317b2cf2b88ffe3e7132779a367c2e629d8a9fbc613ed2051e877e8",
    },
    "4k": {
        "motion": [1920, 1080], "work": [960, 540], "tensor": [480, 288, 48],
        "grid": [60, 36, 1], "origin": [0, 9],
        "raw": "32571c8e622d6628b61a1552653e927b239dc0ff11973d5b02d8cfe32479de75",
        "address_free": "e0f68a98f41f292f2f645d469cc0ef889463280a8975de34bd437ca94b93b5a4",
    },
}
FIXTURES = ("first-frame", "fan-in")
POISONS = {"a5": 0xA5, "5a": 0x5A}
ORDINALS = ("a", "b")
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


def source_identity(repository: Path) -> dict[str, str]:
    paths = (
        "src/fg_dynamic_module5/contract.h",
        "src/fg_dynamic_module5/dynamic_module5.metal",
        "src/fg_dynamic_module5/macos_probe.mm",
        "src/fg_dynamic_module5/rtx_nvapi_oracle.cpp",
        "scripts/build-fg-dynamic-module5-rtx.ps1",
        "scripts/build-macos-fg-dynamic-module5.sh",
        "scripts/verify-fg-dynamic-module5.py",
        "scripts/test-fg-dynamic-module5-mutations.py",
        "src/fg_early_l19/contract.h",
        "src/fg_early_l19/macos_probe.mm",
        "src/fg_early_l19/rtx_nvapi_oracle.cpp",
        "src/fg_compositor_l87/compositor_l87_contract.h",
        "src/fg_compositor_l87/rtx_nvapi_oracle.cpp",
    )
    return {path: file_sha(repository / path) for path in paths}


def run_id(resolution: str, fixture: str,
           poison: str, ordinal: str) -> str:
    return f"{resolution}-{fixture}-poison-{poison}-process-{ordinal}"


def output_sizes(spec: dict[str, Any]) -> dict[str, int]:
    work_pixels = spec["work"][0] * spec["work"][1]
    tensor = spec["tensor"][0] * spec["tensor"][1] * 48 * 2
    return {"tensor": tensor, "mask": work_pixels, "vector": work_pixels * 4}


def validate_rig(rig: dict[str, Any], sources: dict[str, str]) -> None:
    require(rig.get("schema") == "neuralstudio.fg.dynamic-module5-rig.v1" and
            rig.get("job_id") == JOB_ID and
            rig.get("attempt_id") == "oracle-v1" and
            rig.get("smoke") is False, "rig identity drift")
    expected = {
        "module_sha256": MODULE_SHA,
        "source_sha256": sources["src/fg_dynamic_module5/rtx_nvapi_oracle.cpp"],
        "contract_sha256": sources["src/fg_dynamic_module5/contract.h"],
        "early_source_sha256": sources["src/fg_early_l19/rtx_nvapi_oracle.cpp"],
        "early_contract_sha256": sources["src/fg_early_l19/contract.h"],
        "base_source_sha256": sources["src/fg_compositor_l87/rtx_nvapi_oracle.cpp"],
        "base_contract_sha256": sources["src/fg_compositor_l87/compositor_l87_contract.h"],
        "harness_sha256": sources["scripts/build-fg-dynamic-module5-rtx.ps1"],
    }
    require(all(rig.get(key) == value for key, value in expected.items()),
            "rig source identity drift")
    require(rig.get("run_count") == 32 and
            rig.get("fresh_process_count") == 32 and
            rig.get("true_typed_d3d12_resources") is True and
            rig.get("exit_code") == 0 and
            isinstance(rig.get("started_utc"), str) and
            isinstance(rig.get("completed_utc"), str),
            "rig execution drift")
    require(rig.get("expected_or_captured_output_ingress") is False and
            rig.get("address_fields_recorded") is False and
            rig.get("result") == "pass", "rig forbidden ingress")
    address_free(rig, "rig")


def validate_backend(kind: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = "rtx-d3d12-nvapi" if kind == "rtx" else "mac-metal"
    process_ids: set[int] = set()
    scenarios: dict[str, Any] = {}
    receipt_hashes: dict[str, str] = {}
    for resolution, spec in RESOLUTIONS.items():
        sizes = output_sizes(spec)
        for fixture in FIXTURES:
            key = f"{resolution}/{fixture}"
            receipts = []
            for poison_name, poison in POISONS.items():
                for ordinal in ORDINALS:
                    identity = run_id(resolution, fixture,
                                      poison_name, ordinal)
                    run_root = root / identity
                    receipt_path = run_root / "run-receipt.json"
                    receipt = load(receipt_path)
                    require(receipt.get("schema") == RUN_SCHEMA and
                            receipt.get("backend") == backend and
                            receipt.get("run_id") == identity and
                            receipt.get("resolution") == resolution and
                            receipt.get("phase_fixture") == fixture and
                            receipt.get("poison_byte") == poison,
                            f"{identity}: identity drift")
                    require(receipt.get("module_token") == 5 and
                            receipt.get("module_sha256") == MODULE_SHA and
                            receipt.get("grid") == spec["grid"] and
                            receipt.get("block") == [16, 16, 1] and
                            receipt.get("packed_parameter_bytes") == 192 and
                            receipt.get("motion_extent") == spec["motion"] and
                            receipt.get("work_extent") == spec["work"] and
                            receipt.get("tensor_extent") == spec["tensor"] and
                            receipt.get("logical_origin") == spec["origin"] and
                            receipt.get("raw_parameter_sha256") == spec["raw"] and
                            receipt.get("address_free_parameter_sha256") ==
                            spec["address_free"], f"{identity}: ABI drift")
                    inputs = receipt.get("input_pre_sha256")
                    require(isinstance(inputs, list) and len(inputs) == 10 and
                            inputs == receipt.get("input_post_sha256"),
                            f"{identity}: input immutability drift")
                    require(receipt.get("tensor_output_bytes") == sizes["tensor"] and
                            receipt.get("mask_output_bytes") == sizes["mask"] and
                            receipt.get("vector_output_bytes") == sizes["vector"],
                            f"{identity}: output size drift")
                    require(receipt.get("input_mismatched_bytes") == 0 and
                            receipt.get("buffer_canary_mismatched_bytes") == 0 and
                            receipt.get("texture_canary_mismatched_bytes") == 0,
                            f"{identity}: integrity gate failed")
                    full = poison_name == "a5" and ordinal == "a"
                    require(receipt.get("artifacts_written") is full,
                            f"{identity}: artifact policy drift")
                    require(receipt.get("complete_write_coverage_checked_by_dual_poison") is True and
                            receipt.get("expected_or_captured_output_ingress") is False and
                            receipt.get("address_fields_recorded") is False and
                            receipt.get("result") == "pass",
                            f"{identity}: forbidden ingress")
                    if kind == "rtx":
                        require(receipt.get("function_name") == "main_kernel" and
                                receipt.get("module_transport") ==
                                "NvAPI D3D12 selected fatbin",
                                f"{identity}: RTX transport drift")
                    else:
                        require(receipt.get("architecture_rom_sha256") == ROM_SHA and
                                receipt.get("normalized_point_fraction_bits") == 21 and
                                receipt.get("vector_surface_half_rounding") ==
                                "toward-zero",
                                f"{identity}: Metal numeric drift")
                    require(isinstance(receipt.get("process_id"), int),
                            f"{identity}: process identity missing")
                    address_free(receipt, identity)
                    process_ids.add(receipt["process_id"])
                    receipt_hashes[identity] = file_sha(receipt_path)
                    receipts.append(receipt)
                    if full:
                        for label, filename, bytes_field, hash_field in (
                            ("tensor", "tensor-output.fp16",
                             "tensor_output_bytes", "tensor_output_sha256"),
                            ("mask", "mask-output.r8",
                             "mask_output_bytes", "mask_output_sha256"),
                            ("vector", "vector-output.rg16f",
                             "vector_output_bytes", "vector_output_sha256"),
                        ):
                            payload = (run_root / filename).read_bytes()
                            require(len(payload) == receipt[bytes_field] and
                                    sha(payload) == receipt[hash_field],
                                    f"{identity}: {label} artifact drift")
            require(len({tuple(row["input_pre_sha256"]) for row in receipts}) == 1,
                    f"{key}: input repeat drift")
            fields = ("tensor_output_sha256", "mask_output_sha256",
                      "vector_output_sha256")
            require(all(len({row[field] for row in receipts}) == 1
                        for field in fields),
                    f"{key}: repeat/two-poison output drift")
            first = receipts[0]
            scenarios[key] = {
                "input_sha256": first["input_pre_sha256"],
                "tensor": {"bytes": sizes["tensor"],
                           "sha256": first["tensor_output_sha256"]},
                "mask": {"bytes": sizes["mask"],
                         "sha256": first["mask_output_sha256"]},
                "vector": {"bytes": sizes["vector"],
                           "sha256": first["vector_output_sha256"]},
            }
    require(len(process_ids) == 32, f"{kind.upper()} process count drift")
    for resolution in RESOLUTIONS:
        require(scenarios[f"{resolution}/first-frame"]["input_sha256"] !=
                scenarios[f"{resolution}/fan-in"]["input_sha256"],
                f"{resolution}: temporal fixture diversity drift")
    result = rooted({
        "schema": f"neuralstudio.fg.dynamic-module5-{kind}-backend.v1",
        "backend": backend,
        "module_sha256": MODULE_SHA,
        "fresh_process_count": 32,
        "scenarios": scenarios,
        "run_receipt_sha256": receipt_hashes,
        "repeat_and_dual_poison_exact": True,
        "full_write_coverage_from_dual_poison": True,
        "all_ten_inputs_immutable": True,
        "buffer_and_texture_canaries_intact": True,
        "true_typed_resources": True,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    return result, scenarios


def compare_outputs(rtx_root: Path, mac_root: Path) -> tuple[int, int, list[dict[str, Any]]]:
    compared = 0
    mismatched = 0
    rows = []
    for resolution in RESOLUTIONS:
        for fixture in FIXTURES:
            identity = run_id(resolution, fixture, "a5", "a")
            for label, filename in (
                ("tensor", "tensor-output.fp16"),
                ("mask", "mask-output.r8"),
                ("vector", "vector-output.rg16f"),
            ):
                rtx = (rtx_root / identity / filename).read_bytes()
                mac = (mac_root / identity / filename).read_bytes()
                require(len(rtx) == len(mac), f"{identity}/{label}: size drift")
                differences = sum(left != right for left, right in zip(rtx, mac))
                compared += len(rtx)
                mismatched += differences
                rows.append({
                    "scenario": f"{resolution}/{fixture}",
                    "output": label,
                    "bytes": len(rtx),
                    "sha256": sha(rtx),
                    "mismatched_bytes": differences,
                })
    return compared, mismatched, rows


def fixed12_counterexample(rtx_root: Path, rejected_root: Path) -> dict[str, Any]:
    identity = run_id("720p", "fan-in", "a5", "a")
    mismatches = {}
    for label, filename in (
        ("tensor", "tensor-output.fp16"),
        ("mask", "mask-output.r8"),
        ("vector", "vector-output.rg16f"),
    ):
        selected = (rtx_root / identity / filename).read_bytes()
        rejected = (rejected_root / filename).read_bytes()
        require(len(selected) == len(rejected), "fixed12 counterexample size drift")
        mismatches[label] = sum(a != b for a, b in zip(selected, rejected))
    require(mismatches == {"tensor": 4499, "mask": 0, "vector": 4499},
            "fixed12 counterexample census drift")
    return mismatches


def build(args: argparse.Namespace) -> None:
    repository = args.repository.resolve()
    sources = source_identity(repository)
    rig_path = args.rtx / "rig-receipt.json"
    rig = load(rig_path)
    validate_rig(rig, sources)
    rtx, rtx_scenarios = validate_backend("rtx", args.rtx.resolve())
    mac, mac_scenarios = validate_backend("mac", args.mac.resolve())
    require(rtx_scenarios == mac_scenarios, "backend scenario disagreement")
    compared, mismatched, comparisons = compare_outputs(
        args.rtx.resolve(), args.mac.resolve())
    require(mismatched == 0, "RTX/Metal output mismatch")
    fixed12 = fixed12_counterexample(
        args.rtx.resolve(), args.rejected_fixed12.resolve())
    shape_hashes = {
        name: file_sha(repository /
                       f"evidence/specs/fg-resolution-shapes-v1/shape-{name}.json")
        for name in RESOLUTIONS
    }
    abi = rooted({
        "schema": "neuralstudio.fg.dynamic-module5-abi.v1",
        "canonical_launch": 19,
        "large_trace_launch": 23,
        "phase": "evaluate_1 generated",
        "module_token": 5,
        "module_sha256": MODULE_SHA,
        "function_name": "main_kernel",
        "packed_parameter_bytes": 192,
        "handle_fields": {
            "input_textures": {"offsets": list(range(0, 80, 8)), "count": 10},
            "tensor_output": {"offset": 80, "type": "FP16 NHWC48 buffer"},
            "mask_output": {"offset": 88, "type": "R8_UNORM surface"},
            "vector_output": {"offset": 96, "type": "RG16_FLOAT surface"},
        },
        "tail_offsets": {
            "152_156": "motion normalization extent",
            "160_164": "work surface extent",
            "168_172": "tensor spatial extent",
            "176_180_184_188": "origin and paired controls",
        },
        "resolutions": {
            name: {
                "motion_extent": spec["motion"],
                "work_extent": spec["work"],
                "tensor_extent": spec["tensor"],
                "grid": spec["grid"],
                "block": [16, 16, 1],
                "logical_origin": spec["origin"],
                "raw_parameter_sha256": spec["raw"],
                "address_free_parameter_sha256": spec["address_free"],
                "shape_receipt_sha256": shape_hashes[name],
            } for name, spec in RESOLUTIONS.items()
        },
        "four_k_split_motion_and_work_extents": True,
        "opaque_live_handles_recorded": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    implementation = rooted({
        "schema": "neuralstudio.fg.dynamic-module5-implementation.v1",
        "source_sha256": sources,
        "architecture_rom_sha256": ROM_SHA,
        "normalized_point_fraction_bits": 21,
        "per_input_coordinate_patch_count": 0,
        "selected_approximate_division": True,
        "vector_surface_half_rounding": "toward-zero",
        "tensor_channel_layout":
            "10 features x four 2x2 subpixels, then eight zero channels",
        "compile_policy": "Metal 3.1, fast math disabled, FP contraction disabled",
        "expected_output_embedded": False,
        "captured_output_embedded": False,
        "result": "pass",
    })
    counterexamples = rooted({
        "schema": "neuralstudio.fg.dynamic-module5-counterexamples.v1",
        "items": [
            {"name": "fixed12 normalized point sampler", "result": "rejected",
             "scenario": "720p/fan-in", "mismatched_bytes": fixed12},
            {"name": "fixed 320x180 coordinate exception table",
             "result": "rejected",
             "reason": "resolution-specific rows are replaced by one 21-bit rule"},
            {"name": "4K equal motion and work extents", "result": "rejected",
             "reason": "live ABI uses 1920x1080 motion normalization and 960x540 work"},
            {"name": "oracle-smoke-v1", "result": "build-rejected",
             "reason": "nested source redefined main; direct compositor dependency fixed it"},
            {"name": "remote trace listing filter", "result": "rejected",
             "reason": "local shell consumed a PowerShell pipeline variable; a literal filter replaced it"},
            {"name": "first PTX extraction command", "result": "rejected",
             "reason": "nested shell parsing rejected the exit-code expression before remote execution"},
            {"name": "first full-artifact staging target", "result": "corrected",
             "reason": "two harmless job-scoped extra copies were ignored and correct paths restaged"},
        ],
        "fixed12_counterexample_fed_to_execution": False,
        "all_failures_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.dynamic-module5-accepted.v1",
        "selected_unit": "canonical L19, module5, main_kernel",
        "module_sha256": MODULE_SHA,
        "resolutions": list(RESOLUTIONS),
        "temporal_phase_fixtures": list(FIXTURES),
        "processes_per_backend": 32,
        "abi_receipt_root_sha256": abi["receipt_root_sha256"],
        "implementation_receipt_root_sha256":
            implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": counterexamples["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "rtx_mac_compared_bytes": compared,
        "rtx_mac_mismatched_bytes": mismatched,
        "comparisons": comparisons,
        "claims": {
            "module5_four_resolution_two_fixture_exact": True,
            "all_three_outputs_exact": True,
            "generic_21_bit_sampler_exact": True,
            "per_input_coordinate_patches_used": False,
            "full_output_write_coverage": True,
            "all_ten_inputs_immutable": True,
            "direct_L20_tensor_layout_exact": True,
            "connected_dynamic_early_graph": False,
            "complete_generated_frame": False,
            "Netal_integration": False,
        },
        "scope": "standalone generated-phase L19/module5 at four live target configurations",
        "result": "pass",
    })
    for name, value in (
        ("abi-receipt.json", abi),
        ("implementation-receipt.json", implementation),
        ("counterexamples.json", counterexamples),
        ("rtx-backend-receipt.json", rtx),
        ("mac-backend-receipt.json", mac),
        ("accepted-receipt.json", accepted),
    ):
        write(args.output / name, value)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_compared_bytes={compared}")
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
    require(abi.get("packed_parameter_bytes") == 192 and
            abi.get("four_k_split_motion_and_work_extents") is True and
            abi.get("resolutions", {}).get("4k", {}).get("motion_extent") ==
            [1920, 1080] and
            abi.get("resolutions", {}).get("4k", {}).get("work_extent") ==
            [960, 540], "ABI extent drift")
    require(implementation.get("source_sha256") == source_identity(repository),
            "source identity drift")
    require(implementation.get("architecture_rom_sha256") == ROM_SHA and
            implementation.get("normalized_point_fraction_bits") == 21 and
            implementation.get("per_input_coordinate_patch_count") == 0 and
            implementation.get("vector_surface_half_rounding") ==
            "toward-zero", "implementation numeric drift")
    require(counterexamples.get("all_failures_preserved") is True and
            len(counterexamples.get("items", [])) == 7 and
            counterexamples["items"][0].get("mismatched_bytes") ==
            {"tensor": 4499, "mask": 0, "vector": 4499},
            "counterexample ledger drift")
    require(rtx.get("fresh_process_count") == 32,
            "RTX process count drift")
    require(mac.get("fresh_process_count") == 32,
            "Mac process count drift")
    for label, backend in (("RTX", rtx), ("Mac", mac)):
        require(backend.get("module_sha256") == MODULE_SHA and
                backend.get("repeat_and_dual_poison_exact") is True and
                backend.get("full_write_coverage_from_dual_poison") is True and
                backend.get("all_ten_inputs_immutable") is True and
                backend.get("buffer_and_texture_canaries_intact") is True and
                backend.get("true_typed_resources") is True and
                backend.get("expected_or_captured_output_ingress") is False and
                backend.get("result") == "pass", f"{label} forbidden ingress")
    require(rtx.get("scenarios") == mac.get("scenarios"),
            "backend scenario disagreement")
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
    require(accepted.get("rtx_mac_compared_bytes") == 133_954_560 and
            accepted.get("rtx_mac_mismatched_bytes") == 0 and
            len(accepted.get("comparisons", [])) == 24 and
            all(row.get("mismatched_bytes") == 0
                for row in accepted.get("comparisons", [])),
            "accepted mismatch count drift")
    claims = accepted.get("claims", {})
    require(claims.get("module5_four_resolution_two_fixture_exact") is True and
            claims.get("all_three_outputs_exact") is True and
            claims.get("generic_21_bit_sampler_exact") is True and
            claims.get("per_input_coordinate_patches_used") is False and
            claims.get("full_output_write_coverage") is True and
            claims.get("all_ten_inputs_immutable") is True and
            claims.get("direct_L20_tensor_layout_exact") is True and
            claims.get("connected_dynamic_early_graph") is False and
            claims.get("complete_generated_frame") is False and
            claims.get("Netal_integration") is False,
            "scope drift")
    if require_negative:
        negative = load(evidence / "negative-audit.json")
        verify_root(negative, "negative-audit.json")
        require(negative.get("mutation_count", 0) >= 14 and
                negative.get("all_mutations_rejected") is True and
                negative.get("verifier_sha256") ==
                file_sha(repository / "scripts/verify-fg-dynamic-module5.py"),
                "negative audit drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--repository", type=Path, required=True)
    build_parser.add_argument("--rtx", type=Path, required=True)
    build_parser.add_argument("--mac", type=Path, required=True)
    build_parser.add_argument("--rejected-fixed12", type=Path, required=True)
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
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
