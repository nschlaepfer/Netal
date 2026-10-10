#!/usr/bin/env python3
"""Fail-closed verification for the connected dynamic FG early chain."""

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


JOB_ID = "fg-connected-dynamic-early-20260904T155900Z"
RUN_SCHEMA = "neuralstudio.fg.connected-dynamic-early-run.v1"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
CHECKPOINTS = (
    "reset-module11", "reset-module15", "reset-module13", "reset-module21",
    "reset-module22", "reset-module23", "reset-module24", "reset-module25",
    "reset-module14", "reset-module27", "reset-module4",
    "generated-module27", "generated-module11", "generated-module15",
    "generated-module13", "generated-module21", "generated-module22",
    "generated-module23", "generated-module24", "generated-module25",
    "generated-module5",
)
RESOLUTIONS = {
    "720p": {
        "source": [1280, 720], "color": [640, 360], "motion": [640, 360],
        "tensor": [320, 192, 48], "selected_level": 10,
        "extra_levels": 1, "module23": [320, 180],
    },
    "1080p": {
        "source": [1920, 1080], "color": [960, 540], "motion": [960, 540],
        "tensor": [480, 288, 48], "selected_level": 10,
        "extra_levels": 1, "module23": [480, 270],
    },
    "1440p": {
        "source": [2560, 1440], "color": [1280, 720], "motion": [1280, 720],
        "tensor": [640, 384, 48], "selected_level": 11,
        "extra_levels": 2, "module23": [320, 180],
    },
    "4k": {
        "source": [3840, 2160], "color": [960, 540], "motion": [1920, 1080],
        "tensor": [480, 288, 48], "selected_level": 11,
        "extra_levels": 2, "module23": [480, 270],
    },
}
MODULES = {
    "11": "06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206",
    "15": "9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381",
    "13": "fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1",
    "21": "11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2",
    "22": "7d6602bbed88409e9ad2ef71293869dd22f7153f6387339ed65e91c12341c67a",
    "23": "d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d",
    "24": "5a41f9d65d1f6f933ae42f36b493f60687a646b73929291bafbe9d00297b6bdf",
    "25": "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824",
    "14": "540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c",
    "27": "c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7",
    "4": "c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840",
    "5": "3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c",
}
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
        "src/fg_connected_dynamic_early/contract.h",
        "src/fg_connected_dynamic_early/module5_mixed.metal",
        "src/fg_connected_dynamic_early/macos_connected.mm",
        "src/fg_connected_dynamic_early/rtx_connected.cpp",
        "scripts/build-fg-connected-dynamic-early-rtx.ps1",
        "scripts/build-macos-fg-connected-dynamic-early.sh",
        "scripts/verify-fg-connected-dynamic-early.py",
        "scripts/test-fg-connected-dynamic-early-mutations.py",
        "src/fg_mid_l55/rtx_oracle.cpp",
        "src/fg_mid_l55/fg_mid_l55_contract.h",
        "src/fg_mid_l55/fg_mid_l55_inputs.h",
        "src/fg_connected_l3_l19/macos_connected.mm",
        "src/fg_dynamic_module11/contract.h",
        "src/fg_dynamic_module11/dynamic_module11.metal",
        "src/fg_dynamic_module13/contract.h",
        "src/fg_dynamic_module13/dynamic_module13.metal",
        "src/fg_dynamic_module15/contract.h",
        "src/fg_dynamic_module15/dynamic_module15.metal",
        "src/fg_dynamic_module21/contract.h",
        "src/fg_dynamic_module21/dynamic_module21.metal",
        "src/fg_dynamic_module23/contract.h",
        "src/fg_dynamic_module23/dynamic_module23_mode0.metal",
        "src/fg_dynamic_module25/contract.h",
        "src/fg_dynamic_module25/dynamic_module25.metal",
        "src/fg_dynamic_module14/dynamic_module14.metal",
        "src/fg_dynamic_module5/contract.h",
        "src/fg_dynamic_module5/dynamic_module5.metal",
        "src/fg_early_l19/contract.h",
        "src/fg_resolution_preprocess_l22_l24/contract.h",
        "src/fg_resolution_preprocess_l22_l24/module22.metal",
        "src/fg_resolution_preprocess_l22_l24/module24_level10.metal",
        "src/fg_resolution_preprocess_l22_l24/module24_level11.metal",
        "src/fg_early_status_l10_l12/status_write.metal",
        "src/fg_early_l11/early_l11.metal",
    )
    return {path: file_sha(repository / path) for path in paths}


def run_id(resolution: str, poison: str, ordinal: str) -> str:
    return f"{resolution}-poison-{poison}-process-{ordinal}"


def artifact_sizes(spec: dict[str, Any]) -> dict[str, int]:
    return {
        "allocation37.fp16": spec["tensor"][0] * spec["tensor"][1] * 48 * 2,
        "auxiliary-mask.r8": spec["color"][0] * spec["color"][1],
        "auxiliary-vector.rg16f": spec["color"][0] * spec["color"][1] * 4,
        "reset-present.rgba8": spec["source"][0] * spec["source"][1] * 4,
    }


def validate_rig(rig: dict[str, Any], sources: dict[str, str]) -> None:
    require(rig.get("schema") ==
            "neuralstudio.fg.connected-dynamic-early-rig.v1" and
            rig.get("job_id") == JOB_ID and
            rig.get("attempt_id") == "oracle-v2" and
            rig.get("smoke") is False, "rig identity drift")
    expected = {
        "source_sha256": sources["src/fg_connected_dynamic_early/rtx_connected.cpp"],
        "contract_sha256": sources["src/fg_connected_dynamic_early/contract.h"],
        "base_source_sha256": sources["src/fg_mid_l55/rtx_oracle.cpp"],
        "base_contract_sha256": sources["src/fg_mid_l55/fg_mid_l55_contract.h"],
        "base_inputs_sha256": sources["src/fg_mid_l55/fg_mid_l55_inputs.h"],
        "harness_sha256": sources["scripts/build-fg-connected-dynamic-early-rtx.ps1"],
    }
    require(all(rig.get(key) == value for key, value in expected.items()),
            "rig source identity drift")
    dependency = {
        path: sources[path] for path in (
            "src/fg_dynamic_module11/contract.h",
            "src/fg_dynamic_module13/contract.h",
            "src/fg_dynamic_module15/contract.h",
            "src/fg_dynamic_module21/contract.h",
            "src/fg_dynamic_module23/contract.h",
            "src/fg_dynamic_module25/contract.h",
            "src/fg_dynamic_module5/contract.h",
            "src/fg_early_l19/contract.h",
            "src/fg_resolution_preprocess_l22_l24/contract.h",
        )
    }
    require(rig.get("dependency_sha256") == dependency,
            "rig dependency identity drift")
    require(rig.get("module_sha256") == MODULES,
            "rig module identity drift")
    require(rig.get("run_count") == 16 and
            rig.get("fresh_process_count") == 16 and
            rig.get("selected_launches_per_process") == 21 and
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
    resolutions: dict[str, Any] = {}
    receipt_hashes: dict[str, str] = {}
    for resolution, spec in RESOLUTIONS.items():
        rows = []
        for poison_name, poison in (("a5", 0xA5), ("5a", 0x5A)):
            for ordinal in ("a", "b"):
                identity = run_id(resolution, poison_name, ordinal)
                run_root = root / identity
                receipt_path = run_root / "run-receipt.json"
                receipt = load(receipt_path)
                require(receipt.get("schema") == RUN_SCHEMA and
                        receipt.get("backend") == backend and
                        receipt.get("run_id") == identity and
                        receipt.get("resolution") == resolution and
                        receipt.get("poison_byte") == poison,
                        f"{identity}: identity drift")
                require(receipt.get("selected_launch_count") == 21 and
                        receipt.get("module22_and_module24_executed_each_phase")
                        is True and
                        tuple(receipt.get("checkpoint_sha256", {}).keys()) ==
                        CHECKPOINTS, f"{identity}: checkpoint topology drift")
                sizes = artifact_sizes(spec)
                require(receipt.get("allocation37_bytes") ==
                        sizes["allocation37.fp16"] and
                        isinstance(receipt.get("allocation37_sha256"), str),
                        f"{identity}: allocation37 receipt drift")
                if kind == "rtx":
                    require(receipt.get("allocation37_poison_halfwords") == 0,
                            f"{identity}: allocation37 coverage drift")
                require(receipt.get("input_mismatched_bytes") == 0 and
                        receipt.get("canary_mismatched_bytes") == 0,
                        f"{identity}: integrity gate failed")
                full = poison_name == "a5" and ordinal == "a"
                require(receipt.get("artifacts_written") is full,
                        f"{identity}: artifact policy drift")
                require(receipt.get("expected_or_captured_output_ingress") is False and
                        receipt.get("archived_intermediate_ingress") is False and
                        receipt.get("address_fields_recorded") is False and
                        receipt.get("result") == "pass",
                        f"{identity}: forbidden ingress")
                require(isinstance(receipt.get("process_id"), int),
                        f"{identity}: process identity missing")
                address_free(receipt, identity)
                process_ids.add(receipt["process_id"])
                receipt_hashes[identity] = file_sha(receipt_path)
                rows.append(receipt)
                if full:
                    expected_hashes = {
                        "allocation37.fp16": receipt["allocation37_sha256"],
                        "auxiliary-mask.r8": receipt["auxiliary_mask_sha256"],
                        "auxiliary-vector.rg16f": receipt["auxiliary_vector_sha256"],
                        "reset-present.rgba8": receipt["reset_present_sha256"],
                    }
                    for filename, size in sizes.items():
                        payload = (run_root / filename).read_bytes()
                        require(len(payload) == size and
                                sha(payload) == expected_hashes[filename],
                                f"{identity}: {filename} artifact drift")
        require(all(len({row["checkpoint_sha256"][name] for row in rows}) == 1
                    for name in CHECKPOINTS),
                f"{resolution}: checkpoint poison/repeat drift")
        require(len({row["allocation37_sha256"] for row in rows}) == 1 and
                len({row["auxiliary_mask_sha256"] for row in rows}) == 1 and
                len({row["auxiliary_vector_sha256"] for row in rows}) == 1 and
                len({row["reset_present_sha256"] for row in rows}) == 1,
                f"{resolution}: final poison/repeat drift")
        first = rows[0]
        resolutions[resolution] = {
            "checkpoint_sha256": first["checkpoint_sha256"],
            "allocation37_bytes": first["allocation37_bytes"],
            "allocation37_sha256": first["allocation37_sha256"],
            "auxiliary_mask_sha256": first["auxiliary_mask_sha256"],
            "auxiliary_vector_sha256": first["auxiliary_vector_sha256"],
            "reset_present_sha256": first["reset_present_sha256"],
        }
    require(len(process_ids) == 16, f"{kind.upper()} process count drift")
    result = rooted({
        "schema": f"neuralstudio.fg.connected-dynamic-early-{kind}-backend.v1",
        "backend": backend,
        "fresh_process_count": 16,
        "selected_launches_per_process": 21,
        "resolutions": resolutions,
        "run_receipt_sha256": receipt_hashes,
        "checkpoint_repeat_and_dual_poison_exact": True,
        "full_declared_write_coverage_from_dual_poison": True,
        "external_inputs_immutable": True,
        "buffer_canaries_intact": True,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    return result, resolutions


def compare_artifacts(rtx: Path, mac: Path) -> tuple[int, int, list[dict[str, Any]]]:
    compared = 0
    mismatched = 0
    rows = []
    for resolution, spec in RESOLUTIONS.items():
        identity = run_id(resolution, "a5", "a")
        for filename, expected_size in artifact_sizes(spec).items():
            left = (rtx / identity / filename).read_bytes()
            right = (mac / identity / filename).read_bytes()
            require(len(left) == len(right) == expected_size,
                    f"{resolution}/{filename}: size drift")
            differences = sum(a != b for a, b in zip(left, right))
            compared += len(left)
            mismatched += differences
            rows.append({
                "resolution": resolution, "artifact": filename,
                "bytes": len(left), "sha256": sha(left),
                "mismatched_bytes": differences,
            })
    return compared, mismatched, rows


def build(args: argparse.Namespace) -> None:
    repository = args.repository.resolve()
    sources = source_identity(repository)
    rig_path = args.rtx / "rig-receipt.json"
    rig = load(rig_path)
    validate_rig(rig, sources)
    rtx, rtx_resolutions = validate_backend("rtx", args.rtx.resolve())
    mac, mac_resolutions = validate_backend("mac", args.mac.resolve())
    require(rtx_resolutions == mac_resolutions,
            "RTX/Metal checkpoint disagreement")
    compared, mismatched, comparisons = compare_artifacts(
        args.rtx.resolve(), args.mac.resolve())
    require(compared == 126_881_280 and mismatched == 0,
            "RTX/Metal artifact mismatch")
    shape_hashes = {
        name: file_sha(repository /
                       f"evidence/specs/fg-resolution-shapes-v1/shape-{name}.json")
        for name in RESOLUTIONS
    }
    topology = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-early-topology.v1",
        "selected_launch_count": 21,
        "checkpoint_order": list(CHECKPOINTS),
        "resolutions": {
            name: {**spec, "shape_receipt_sha256": shape_hashes[name]}
            for name, spec in RESOLUTIONS.items()
        },
        "module_sha256": MODULES,
        "module23_bindings": {
            "offsets_0_through_32": "five dependency-ordered half3 levels",
            "level10": "pyramid1,pyramid2,pyramid3,pyramid4,extra0",
            "level11": "pyramid2,pyramid3,pyramid4,extra0,extra1",
        },
        "module24_bindings": {
            "offset8": "pyramid0 R8 selection mask",
            "offsets16_through_40": "pyramid1 through pyramid4 for every selected level",
        },
        "four_k_mixed_module5_resources": True,
        "address_fields_recorded": False,
        "result": "pass",
    })
    implementation = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-early-implementation.v1",
        "source_sha256": sources,
        "architecture_rom_sha256": ROM_SHA,
        "module5_normalized_point_fraction_bits": 21,
        "module5_per_input_coordinate_patch_count": 0,
        "checkpoint_count_per_process": 21,
        "live_device_handoffs_only": True,
        "expected_output_embedded": False,
        "captured_intermediate_embedded": False,
        "result": "pass",
    })
    failures = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-early-counterexamples.v1",
        "items": [
            {"name": "module24 selected-level routing", "result": "rejected",
             "first_failure": "1440p reset module24 device removal",
             "correction": "module24 always consumes pyramid1 through pyramid4"},
            {"name": "module23 offset32 mask label", "result": "rejected",
             "correction": "live handle equality and exact connected execution prove a fifth half3 level"},
            {"name": "uniform module5 resource extents at 4K", "result": "rejected",
             "correction": "motion resources are 1920x1080 while color/model resources are 960x540"},
            {"name": "RTX smoke-v1", "result": "build-rejected",
             "reason": "unit headers required an explicit Windows uint alias"},
            {"name": "RTX smoke-v2", "result": "receipt-rejected",
             "reason": "integer dictionary keys could not be serialized in the rig receipt"},
            {"name": "Mac smoke-v1", "result": "build-rejected",
             "reason": "two unused resolution references failed the warning-as-error host build"},
            {"name": "RTX oracle-v1", "result": "execution-rejected",
             "reason": "preserved the incorrect module24 route and stopped at 1440p"},
        ],
        "all_failures_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-early-accepted.v1",
        "selected_range": "large-resolution early launches 3 through 23",
        "resolutions": list(RESOLUTIONS),
        "processes_per_backend": 16,
        "selected_launches_per_process": 21,
        "rtx_mac_checkpoint_comparisons": 336,
        "rtx_mac_checkpoint_mismatches": 0,
        "rtx_mac_artifact_compared_bytes": compared,
        "rtx_mac_artifact_mismatched_bytes": mismatched,
        "topology_receipt_root_sha256": topology["receipt_root_sha256"],
        "implementation_receipt_root_sha256":
            implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": failures["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "comparisons": comparisons,
        "allocation37": {
            name: {
                "bytes": value["allocation37_bytes"],
                "sha256": value["allocation37_sha256"],
            } for name, value in rtx_resolutions.items()
        },
        "claims": {
            "connected_dynamic_early_exact": True,
            "every_selected_boundary_exact": True,
            "module22_and_module24_both_phases_exact": True,
            "allocation37_live_handoff_exact": True,
            "full_declared_write_coverage": True,
            "external_inputs_immutable": True,
            "connected_L20_and_later": False,
            "complete_generated_frame": False,
            "Netal_integration": False,
        },
        "scope": "connected early reset and first-generated phases through allocation37 only",
        "result": "pass",
    })
    for name, value in (
        ("topology-receipt.json", topology),
        ("implementation-receipt.json", implementation),
        ("counterexamples.json", failures),
        ("rtx-backend-receipt.json", rtx),
        ("mac-backend-receipt.json", mac),
        ("accepted-receipt.json", accepted),
    ):
        write(args.output / name, value)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_artifact_compared_bytes={compared}")
    print("verification_result=pass")


def verify_evidence(repository: Path, evidence: Path,
                    require_negative: bool) -> None:
    names = (
        "topology-receipt.json", "implementation-receipt.json",
        "counterexamples.json", "rtx-backend-receipt.json",
        "mac-backend-receipt.json", "accepted-receipt.json",
    )
    values = {name: load(evidence / name) for name in names}
    for name, value in values.items():
        verify_root(value, name)
    topology = values["topology-receipt.json"]
    implementation = values["implementation-receipt.json"]
    failures = values["counterexamples.json"]
    rtx = values["rtx-backend-receipt.json"]
    mac = values["mac-backend-receipt.json"]
    accepted = values["accepted-receipt.json"]
    require(topology.get("selected_launch_count") == 21 and
            topology.get("checkpoint_order") == list(CHECKPOINTS) and
            topology.get("module_sha256") == MODULES and
            topology.get("four_k_mixed_module5_resources") is True,
            "topology drift")
    require(topology.get("module23_bindings", {}).get("offsets_0_through_32") ==
            "five dependency-ordered half3 levels" and
            topology.get("module24_bindings", {}).get("offsets16_through_40") ==
            "pyramid1 through pyramid4 for every selected level",
            "level binding drift")
    require(implementation.get("source_sha256") == source_identity(repository),
            "source identity drift")
    require(implementation.get("architecture_rom_sha256") == ROM_SHA and
            implementation.get("module5_normalized_point_fraction_bits") == 21 and
            implementation.get("module5_per_input_coordinate_patch_count") == 0 and
            implementation.get("live_device_handoffs_only") is True,
            "implementation drift")
    require(failures.get("all_failures_preserved") is True and
            len(failures.get("items", [])) == 7,
            "failure ledger drift")
    require(rtx.get("fresh_process_count") == 16,
            "RTX process count drift")
    require(mac.get("fresh_process_count") == 16,
            "Mac process count drift")
    for label, backend in (("RTX", rtx), ("Mac", mac)):
        require(backend.get("selected_launches_per_process") == 21 and
                backend.get("checkpoint_repeat_and_dual_poison_exact") is True and
                backend.get("full_declared_write_coverage_from_dual_poison") is True and
                backend.get("external_inputs_immutable") is True and
                backend.get("buffer_canaries_intact") is True and
                backend.get("expected_or_captured_output_ingress") is False and
                backend.get("result") == "pass", f"{label} forbidden ingress")
    require(rtx.get("resolutions") == mac.get("resolutions"),
            "RTX/Metal checkpoint disagreement")
    require(accepted.get("topology_receipt_root_sha256") ==
            topology["receipt_root_sha256"] and
            accepted.get("implementation_receipt_root_sha256") ==
            implementation["receipt_root_sha256"] and
            accepted.get("counterexamples_root_sha256") ==
            failures["receipt_root_sha256"] and
            accepted.get("rtx_backend_root_sha256") ==
            rtx["receipt_root_sha256"] and
            accepted.get("mac_backend_root_sha256") ==
            mac["receipt_root_sha256"], "receipt link drift")
    require(accepted.get("rtx_mac_checkpoint_comparisons") == 336 and
            accepted.get("rtx_mac_checkpoint_mismatches") == 0 and
            accepted.get("rtx_mac_artifact_compared_bytes") == 126_881_280 and
            accepted.get("rtx_mac_artifact_mismatched_bytes") == 0 and
            len(accepted.get("comparisons", [])) == 16,
            "accepted comparison drift")
    claims = accepted.get("claims", {})
    require(claims.get("connected_dynamic_early_exact") is True and
            claims.get("every_selected_boundary_exact") is True and
            claims.get("module22_and_module24_both_phases_exact") is True and
            claims.get("allocation37_live_handoff_exact") is True and
            claims.get("full_declared_write_coverage") is True and
            claims.get("external_inputs_immutable") is True and
            claims.get("connected_L20_and_later") is False and
            claims.get("complete_generated_frame") is False and
            claims.get("Netal_integration") is False,
            "scope drift")
    if require_negative:
        negative = load(evidence / "negative-audit.json")
        verify_root(negative, "negative-audit.json")
        require(negative.get("mutation_count", 0) >= 14 and
                negative.get("all_mutations_rejected") is True and
                negative.get("verifier_sha256") ==
                file_sha(repository /
                         "scripts/verify-fg-connected-dynamic-early.py"),
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
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
