#!/usr/bin/env python3
"""Verify the exact 640x360 89-launch distortion-enabled FG profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

RUN_SCHEMA = "neuralstudio.fg.connected-distortion-l3-l91-run.v1"
EARLY_SCHEMA = "neuralstudio.fg.connected-distortion-l3-l23-run.v1"
TAIL_SCHEMA = "neuralstudio.fg.connected-l20-l87-run.v1"
RTX_BACKEND = "rtx-cuda-external-memory-d3d12-nvapi"
MAC_BACKEND = "mac-metal"
RUNS = (
    ("connected-distortion-l3-l91-poison-a5-process-a", 0xA5),
    ("connected-distortion-l3-l91-poison-a5-process-b", 0xA5),
    ("connected-distortion-l3-l91-poison-5a-process-a", 0x5A),
    ("connected-distortion-l3-l91-poison-5a-process-b", 0x5A),
)
CHECKPOINT_KEYS = [f"launch-{value:06d}" for value in range(3, 92)]
EARLY_KEYS = [f"launch-{value:06d}" for value in range(3, 24)]
TAIL_KEYS = [f"launch-{value:06d}" for value in range(20, 88)]

ALLOCATION37_SHA256 = (
    "957fdf9a9e5d06707f6c36101bd4c4b3b1a2dfb4a40c065a015f4f5d23628b2d"
)
FINAL_SHA256 = (
    "69ced2ef8314a4e4f0d72729b6ef6882afc113078cc6df93fc0bb5d5dadfd374"
)
ARTIFACTS = {
    "allocation37.fp16": (
        1_474_560,
        ALLOCATION37_SHA256,
    ),
    "allocation33.r8": (
        57_600,
        "c72274a22cba2dcef62e0f62834446ee39b0cd95ad0faa27a89b3b49f48aa390",
    ),
    "allocation27.rg16f": (
        230_400,
        "6c132b468890a01b9b0bf1396be1be79ea209e07d1321944ac5a11981be183d9",
    ),
    "distortion-reset-a.rg16f": (
        230_400,
        "85504adc3979478c78e660e883becaf231f883d2c25f2f3d5751cf364dcaf94a",
    ),
    "distortion-reset-b.rg16f": (
        230_400,
        "a4447d1a0989ac3c4164d213a0d606d804deb90b100a570c65a6983f492328e0",
    ),
    "distortion-reset-depth.r32u": (
        230_400,
        "69cb3da9653b5055e2fb5db04e177e5d8729f367a19a9f21a2ca2eb485fbe176",
    ),
    "distortion-generated-a.rg16f": (
        230_400,
        "02bc6d08c5f53535dd7055466daef1324643526fa9c5b7d4b76159abc190846b",
    ),
    "distortion-generated-b.rg16f": (
        230_400,
        "0980ab6224137356459cbcebad8c3c7343a978dad128a5133ee03aeb07f454bc",
    ),
    "distortion-generated-depth.r32u": (
        230_400,
        "412ecc3d4812ea946fd45ce71db08606f60ffca0cc2d5daa5ff337c438ce6ce9",
    ),
    "l20-l87/output.bin": (921_600, FINAL_SHA256),
}
COMPONENT_HASHES = {
    "reset": [ARTIFACTS[name][1] for name in (
        "distortion-reset-a.rg16f",
        "distortion-reset-b.rg16f",
        "distortion-reset-depth.r32u",
    )],
    "generated": [ARTIFACTS[name][1] for name in (
        "distortion-generated-a.rg16f",
        "distortion-generated-b.rg16f",
        "distortion-generated-depth.r32u",
    )],
}
TABLES = {
    "sm120_ex2_unit_reference": (
        67_108_868,
        "40088ff59b58c4018114b4a2dd1fb5b124488b91e7cad79bdf3bf143bf267b75",
    ),
    "sm120_ex2_fraction_reference": (
        33_554_436,
        "f7cfb2fd9df97ae01bc747b866e4928a79311e8443700f938859f43772f73d86",
    ),
    "sm120_rcp_correction_rom": (
        2_097_152,
        "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750",
    ),
    "sm120_half_logit_sigmoid_lut": (
        262_144,
        "394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850",
    ),
}
PROVENANCE_ROOTS = {
    "selected_module_census_root_sha256":
        "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26",
    "allocation_provenance_root_sha256":
        "8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c",
    "descriptor_provenance_root_sha256":
        "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733",
}
DISTORTION_ADAPTER = {
    "accepted_six_launch_receipt_root_sha256":
        "cd3c642042cfb996b6c1296caa2413c63190bc9e355f232b597df8bb92390f1d",
    "abi_receipt_root_sha256":
        "215a912e5b19bddcff115b6239a40c6efd1ac0ad185c662d5836de6762544122",
    "module_sha256": [
        "0f7627574281429873003bd700c909d09c895770d8464af0914559d866fe9d3c",
        "6c507cc85455aeaed386e293c77aa65c861e0093eebd4e2ca220fe57636319ed",
        "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824",
    ],
    "module_parameter_bytes": [56, 72, 48, 56, 72, 48],
    "scatter_write_coverage": {
        "reset": {
            "map_sha256":
                "9e1be38c4114040430f18747bf107ca102aa25f0cf9f569a5bfca34cd5a41dd7",
            "written_pixels": 57_527,
            "target_pixels": 57_600,
        },
        "generated": {
            "map_sha256":
                "8e5277d84b596265902e1437f9b9025dfbf7b76071c3d94b4c130313467dda02",
            "written_pixels": 57_422,
            "target_pixels": 57_600,
        },
    },
}
SOURCE_PATHS = (
    "src/fg_connected_distortion_l3_l87/contract.h",
    "src/fg_connected_distortion_l3_l87/l52_distortion_exact.metal",
    "src/fg_connected_distortion_l3_l87/l54_distortion_exact.metal",
    "src/fg_connected_distortion_l3_l87/macos_connected_distortion.mm",
    "src/fg_connected_distortion_l3_l87/macos_live_handoff.h",
    "src/fg_connected_distortion_l3_l87/macos_live_handoff.mm",
    "src/fg_connected_distortion_l3_l87/module9_debug.ptx",
    "src/fg_connected_distortion_l3_l87/module9_full_core.metal",
    "src/fg_connected_distortion_l3_l87/rtx_connected_distortion.cpp",
    "src/fg_connected_distortion_l3_l87/rtx_half_div_probe.cu",
    "src/fg_connected_distortion_l3_l87/rtx_half_texture_probe.cu",
    "src/fg_connected_distortion_l3_l87/rtx_live_prefix_bridge.cpp",
    "src/fg_connected_distortion_l3_l87/rtx_live_prefix_bridge.h",
    "src/fg_connected_distortion_l3_l87/rtx_module9_debug.cu",
    "scripts/build-fg-connected-distortion-l3-l87-div-probe.ps1",
    "scripts/build-fg-connected-distortion-l3-l87-half-texture-probe.ps1",
    "scripts/build-fg-connected-distortion-l3-l87-rtx.ps1",
    "scripts/build-macos-fg-connected-distortion-l3-l87.sh",
    "scripts/collect-fg-connected-distortion-l3-l87-rtx.ps1",
    "scripts/verify-fg-connected-distortion-l3-l87.py",
    "src/fg_distortion_path/connected_contract.h",
    "src/fg_distortion_path/module10_contract.h",
    "src/fg_distortion_path/module10.metal",
    "src/fg_connected_l20_l87/macos_connected_l20_l87.mm",
    "src/fg_connected_l20_l87/rtx_live_core_bridge.cpp",
    "src/fg_connected_l20_l87/rtx_live_core_bridge.h",
    "src/fg_connected_l20_l87/rtx_mid_l51_l56_smoke.cpp",
    "src/fg_math_ex2/sm120_ex2.metalh",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return sha256_bytes(encoded)


def reroot(value: dict[str, Any]) -> None:
    value.pop("receipt_root_sha256", None)
    value["receipt_root_sha256"] = canonical_sha256(value)


def rooted(value: dict[str, Any]) -> bool:
    candidate = copy.deepcopy(value)
    recorded = candidate.pop("receipt_root_sha256", None)
    return recorded == canonical_sha256(candidate)


def load(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot load {path}: {error}") from error
    require(isinstance(result, dict), f"object required: {path}")
    return result


def address_free(value: dict[str, Any], label: str) -> None:
    encoded = json.dumps(value, sort_keys=True).lower()
    for forbidden in (
        "gpu_virtual_address",
        "device_pointer",
        "host_pointer",
        "descriptor_pointer",
        "resource_address",
    ):
        require(forbidden not in encoded, f"{label}: address field {forbidden}")


def validate_tail(tail: dict[str, Any], backend: str, run_name: str) -> None:
    address_free(tail, f"{run_name}/tail")
    require(tail.get("schema") == TAIL_SCHEMA, f"{run_name}: tail schema")
    require(tail.get("backend") == backend, f"{run_name}: tail backend")
    require(tail.get("run_id") == run_name, f"{run_name}: tail run id")
    require(tail.get("launch_range") == [20, 87] and
            tail.get("launch_count") == 68, f"{run_name}: tail coverage")
    require(list(tail.get("checkpoint_sha256", {})) == TAIL_KEYS,
            f"{run_name}: tail checkpoint sequence")
    for key, expected in PROVENANCE_ROOTS.items():
        require(tail.get(key) == expected, f"{run_name}: {key}")
    require(tail.get("input_model_mismatched_bytes") == 0 and
            tail.get("canary_mismatched_bytes") == 0 and
            tail.get("readonly_tail_mismatched_bytes") == 0,
            f"{run_name}: tail integrity")
    require(tail.get("live_device_handoff_count") == 67 and
            tail.get("host_intermediate_ingress") is False and
            tail.get("expected_or_archived_tensor_ingress") is False and
            tail.get("address_fields_recorded") is False,
            f"{run_name}: tail handoff/ingress")
    require(tail.get("final_output_bytes") == 921_600 and
            tail.get("final_output_sha256") == FINAL_SHA256 and
            tail.get("result") == "pass", f"{run_name}: tail output")


def validate_early(early: dict[str, Any], backend: str, run_name: str) -> None:
    address_free(early, f"{run_name}/early")
    early_backend = "rtx-d3d12-nvapi" if backend == RTX_BACKEND else MAC_BACKEND
    require(early.get("schema") == EARLY_SCHEMA, f"{run_name}: early schema")
    require(early.get("backend") == early_backend,
            f"{run_name}: early backend")
    require(early.get("run_id") == run_name, f"{run_name}: early run id")
    require(early.get("launch_range") == [3, 23] and
            early.get("launch_count") == 21, f"{run_name}: early coverage")
    require(list(early.get("checkpoint_sha256", {})) == EARLY_KEYS,
            f"{run_name}: early checkpoint sequence")
    require(early.get("allocation29_runtime_source") == "generated-half-0.5" and
            early.get("allocation29_capture_bytes_consumed") is False,
            f"{run_name}: temporal input provenance")
    require(early.get("distortion_component_sha256") == COMPONENT_HASHES,
            f"{run_name}: distortion components")
    require(early.get("input_mismatched_bytes") == 0 and
            early.get("distortion_handoff_mismatched_bytes") == 0 and
            early.get("canary_mismatched_bytes") == 0 and
            early.get("allocation37_poison_halfwords") == 0,
            f"{run_name}: early integrity/write coverage")
    require(early.get("expected_or_archived_intermediate_ingress") is False and
            early.get("distortion_live_device_handoffs") is True and
            early.get("host_distortion_checkpoint_ingress") is False and
            early.get("address_fields_recorded") is False,
            f"{run_name}: distortion handoff/ingress")
    require(early.get("allocation37_bytes") == 1_474_560 and
            early.get("allocation37_sha256") == ALLOCATION37_SHA256 and
            early.get("l20_live_handoff") is True and
            early.get("result") == "pass", f"{run_name}: allocation37")


def derive_checkpoints(early: dict[str, Any], tail: dict[str, Any]) -> dict[str, str]:
    result = dict(early["checkpoint_sha256"])
    for name, digest in tail["checkpoint_sha256"].items():
        shifted = f"launch-{int(name.removeprefix('launch-')) + 4:06d}"
        require(shifted not in result, f"duplicate checkpoint {shifted}")
        result[shifted] = digest
    require(list(result) == CHECKPOINT_KEYS, "derived checkpoint sequence")
    return result


def validate_combined(
    combined: dict[str, Any], early: dict[str, Any], tail: dict[str, Any],
    backend: str, run_name: str, poison: int,
) -> dict[str, str]:
    address_free(combined, f"{run_name}/combined")
    validate_early(early, backend, run_name)
    validate_tail(tail, backend, run_name)
    require(combined.get("schema") == RUN_SCHEMA, f"{run_name}: schema")
    require(combined.get("backend") == backend, f"{run_name}: backend")
    require(combined.get("run_id") == run_name, f"{run_name}: run id")
    require(isinstance(combined.get("process_id"), int) and
            combined.get("process_id") == early.get("process_id") ==
            tail.get("process_id"), f"{run_name}: same process")
    require(combined.get("poison_byte") == poison == early.get("poison_byte") ==
            tail.get("poison_byte"), f"{run_name}: poison")
    require(combined.get("launch_range") == [3, 91] and
            combined.get("launch_count") == 89, f"{run_name}: coverage")
    derived = derive_checkpoints(early, tail)
    require(combined.get("checkpoint_sha256") == derived,
            f"{run_name}: checkpoint provenance")
    require(combined.get("allocation37_sha256") == ALLOCATION37_SHA256,
            f"{run_name}: combined allocation37")
    require(combined.get("live_device_handoff_l19_l20") is True and
            combined.get("live_distortion_adapter_handoffs") is True and
            combined.get("distortion_checkpoint_ingress") is False and
            combined.get("host_intermediate_ingress") is False and
            combined.get("expected_or_archived_tensor_ingress") is False and
            combined.get("address_fields_recorded") is False,
            f"{run_name}: combined handoff/ingress")
    require(combined.get("final_output_bytes") == 921_600 and
            combined.get("final_output_sha256") == FINAL_SHA256 and
            combined.get("result") == "pass", f"{run_name}: final output")
    return derived


def validate_artifacts(run: Path, run_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, (expected_bytes, expected_sha256) in ARTIFACTS.items():
        path = run / name
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise VerificationError(f"{run_name}: cannot read {name}: {error}") from error
        require(len(payload) == expected_bytes, f"{run_name}: {name} bytes")
        digest = sha256_bytes(payload)
        require(digest == expected_sha256, f"{run_name}: {name} hash")
        result[name] = digest
    return result


def validate_backend_receipt(
    receipt: dict[str, Any], backend: str, process_ids: list[int],
) -> None:
    address_free(receipt, f"{backend}/backend")
    expected_schema = (
        "neuralstudio.fg.connected-distortion-l3-l91-rig.v1"
        if backend == RTX_BACKEND
        else "neuralstudio.fg.connected-distortion-l3-l91-mac-backend.v1"
    )
    require(receipt.get("schema") == expected_schema, f"{backend}: backend schema")
    require(receipt.get("fresh_process_count") == 4 and
            receipt.get("poison_bytes") == [90, 165] and
            receipt.get("launch_range") == [3, 91] and
            receipt.get("launch_count") == 89,
            f"{backend}: backend coverage")
    require(receipt.get("allocation37_sha256") == ALLOCATION37_SHA256 and
            receipt.get("invariant_output_sha256") == FINAL_SHA256,
            f"{backend}: backend outputs")
    for field in (
        "all_checkpoint_two_poison_invariant",
        "full_write_coverage_from_two_poison_invariance",
        "live_device_handoff_l19_l20",
        "live_distortion_adapter_handoffs",
    ):
        require(receipt.get(field) is True, f"{backend}: {field}")
    for field in (
        "distortion_checkpoint_ingress",
        "host_intermediate_ingress",
        "expected_or_archived_tensor_ingress",
        "address_fields_recorded",
    ):
        require(receipt.get(field) is False, f"{backend}: {field}")
    require(receipt.get("result") == "pass", f"{backend}: backend result")
    if backend == RTX_BACKEND:
        rows = receipt.get("processes")
        require(isinstance(rows, list) and len(rows) == 4,
                "RTX backend processes")
        require([row.get("process_id") for row in rows] == process_ids and
                all(row.get("exit_code") == 0 for row in rows),
                "RTX backend process identity")


def validate_backend(root: Path, backend: str) -> dict[str, Any]:
    maps: list[dict[str, str]] = []
    pids: list[int] = []
    artifacts: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for run_name, poison in RUNS:
        run = root / run_name
        early_path = run / "run-receipt.json"
        tail_path = run / "l20-l87" / "run-receipt.json"
        combined_path = run / "combined-receipt.json"
        early, tail, combined = map(load, (early_path, tail_path, combined_path))
        checkpoints = validate_combined(
            combined, early, tail, backend, run_name, poison
        )
        artifact_hashes = validate_artifacts(run, run_name)
        maps.append(checkpoints)
        artifacts.append(artifact_hashes)
        pids.append(combined["process_id"])
        rows.append({
            "run_id": run_name,
            "poison_byte": poison,
            "process_id": combined["process_id"],
            "early_receipt_sha256": sha256_file(early_path),
            "tail_receipt_sha256": sha256_file(tail_path),
            "combined_receipt_sha256": sha256_file(combined_path),
            "artifact_bundle_root_sha256": canonical_sha256(artifact_hashes),
        })
    require(len(set(pids)) == 4, f"{backend}: four fresh process ids")
    require(all(value == maps[0] for value in maps[1:]),
            f"{backend}: checkpoint two-poison invariance")
    require(all(value == artifacts[0] for value in artifacts[1:]),
            f"{backend}: artifact two-poison invariance")
    receipt_path = root / "backend-receipt.json"
    receipt = load(receipt_path)
    validate_backend_receipt(receipt, backend, pids)
    result = {
        "schema": "neuralstudio.fg.connected-distortion-l3-l91-normalized-backend.v1",
        "backend": backend,
        "runs": rows,
        "fresh_process_count": 4,
        "two_process_repeat_per_poison": True,
        "poison_bytes": [90, 165],
        "launch_range": [3, 91],
        "launch_count": 89,
        "checkpoint_root_sha256": canonical_sha256(maps[0]),
        "artifact_bundle_root_sha256": canonical_sha256(artifacts[0]),
        "allocation37_sha256": ALLOCATION37_SHA256,
        "final_output_sha256": FINAL_SHA256,
        "all_checkpoint_two_poison_invariant": True,
        "all_artifact_two_poison_invariant": True,
        "complete_write_coverage": True,
        "canary_integrity": True,
        "input_model_immutability": True,
        "live_device_handoffs": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "raw_backend_receipt_sha256": sha256_file(receipt_path),
        "result": "pass",
    }
    reroot(result)
    return result


def validate_tables(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        expected_bytes, expected_sha256 = TABLES[name]
        require(path.stat().st_size == expected_bytes, f"{name}: bytes")
        require(sha256_file(path) == expected_sha256, f"{name}: hash")
        result[name] = {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "semantic_role": "architecture-math lookup, never model/frame bytes",
        }
    return result


def verify(
    mac_root: Path, rtx_root: Path, table_paths: dict[str, Path],
) -> dict[str, Any]:
    mac = validate_backend(mac_root, MAC_BACKEND)
    rtx = validate_backend(rtx_root, RTX_BACKEND)
    require(mac["checkpoint_root_sha256"] == rtx["checkpoint_root_sha256"],
            "RTX/Metal checkpoint root")
    require(mac["artifact_bundle_root_sha256"] ==
            rtx["artifact_bundle_root_sha256"],
            "RTX/Metal artifact root")
    for run_name, _ in RUNS:
        for artifact in ARTIFACTS:
            left = (rtx_root / run_name / artifact).read_bytes()
            right = (mac_root / run_name / artifact).read_bytes()
            require(left == right, f"{run_name}/{artifact}: RTX/Metal bytes")
    tables = validate_tables(table_paths)
    value = {
        "schema": "neuralstudio.fg.connected-distortion-l3-l91-verification.v1",
        "scope": "independent deterministic 640x360 reset-plus-one-generated 89-launch distortion profile",
        "launch_range": [3, 91],
        "launch_count": 89,
        "checkpoint_count": 89,
        "cross_backend_checkpoint_root_sha256":
            mac["checkpoint_root_sha256"],
        "cross_backend_artifact_root_sha256":
            mac["artifact_bundle_root_sha256"],
        "allocation37_sha256": ALLOCATION37_SHA256,
        "final_output_bytes": 921_600,
        "final_output_sha256": FINAL_SHA256,
        "distortion_component_sha256": COMPONENT_HASHES,
        "independent_distortion_adapter": DISTORTION_ADAPTER,
        "architecture_math_tables": tables,
        "rtx_backend_receipt_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_receipt_root_sha256": mac["receipt_root_sha256"],
        "fresh_processes_per_backend": 4,
        "two_process_repeat_per_poison": True,
        "two_poison_checkpoint_invariance": True,
        "full_declared_write_coverage": True,
        "partial_scatter_untouched_pixels_are_defined_initial_state": True,
        "launch9_checkpoint_exact": True,
        "shifted_l52_l54_tail_exact": True,
        "same_process_all_89_launches": True,
        "live_distortion_adapter_handoffs": True,
        "live_device_handoff_l19_l20": True,
        "input_model_table_immutability": True,
        "canary_integrity": True,
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "connected_full_core_distortion_fixture_exact": True,
            "complete_generated_frame_fixture_exact": True,
            "production_game_resources": False,
            "second_generated_frame_temporal_parity": False,
            "subrect_profile_exact": False,
            "hudless_ui_modes_exact": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    }
    reroot(value)
    return value


def validate_verification(value: dict[str, Any]) -> None:
    require(rooted(value), "verification root")
    require(value.get("schema") ==
            "neuralstudio.fg.connected-distortion-l3-l91-verification.v1",
            "verification schema")
    require(value.get("launch_count") == 89 and
            value.get("checkpoint_count") == 89,
            "verification coverage")
    require(value.get("launch9_checkpoint_exact") is True and
            value.get("shifted_l52_l54_tail_exact") is True,
            "verification exact checkpoints")
    require(value.get("independent_distortion_adapter") == DISTORTION_ADAPTER,
            "verification distortion adapter provenance")
    claims = value.get("claims", {})
    require(claims.get("connected_full_core_distortion_fixture_exact") is True and
            claims.get("complete_generated_frame_fixture_exact") is True,
            "verification accepted claim")
    for claim in (
        "production_game_resources",
        "second_generated_frame_temporal_parity",
        "subrect_profile_exact",
        "hudless_ui_modes_exact",
        "netal_integration",
        "performance",
    ):
        require(claims.get(claim) is False, f"scope overclaim: {claim}")
    require(value.get("result") == "pass", "verification result")


def negative_audit(
    mac_root: Path, rtx_root: Path, table_paths: dict[str, Path],
) -> dict[str, Any]:
    run_name, poison = RUNS[0]
    run = mac_root / run_name
    early = load(run / "run-receipt.json")
    tail = load(run / "l20-l87" / "run-receipt.json")
    combined = load(run / "combined-receipt.json")
    receipt_mutations: dict[str, tuple[str, Callable[[dict[str, Any]], None]]] = {
        "checkpoint-byte": ("combined", lambda value: value["checkpoint_sha256"].__setitem__("launch-000009", "0" * 64)),
        "checkpoint-coverage": ("combined", lambda value: value["checkpoint_sha256"].pop("launch-000056")),
        "launch-count": ("combined", lambda value: value.__setitem__("launch_count", 88)),
        "final-output": ("combined", lambda value: value.__setitem__("final_output_sha256", "f" * 64)),
        "same-process": ("combined", lambda value: value.__setitem__("process_id", value["process_id"] + 1)),
        "poison": ("combined", lambda value: value.__setitem__("poison_byte", 0)),
        "distortion-handoff": ("combined", lambda value: value.__setitem__("live_distortion_adapter_handoffs", False)),
        "expected-ingress": ("combined", lambda value: value.__setitem__("expected_or_archived_tensor_ingress", True)),
        "address-field": ("combined", lambda value: value.__setitem__("gpu_virtual_address", "0x1")),
        "early-canary": ("early", lambda value: value.__setitem__("canary_mismatched_bytes", 1)),
        "early-handoff": ("early", lambda value: value.__setitem__("distortion_handoff_mismatched_bytes", 1)),
        "early-poison": ("early", lambda value: value.__setitem__("allocation37_poison_halfwords", 1)),
        "capture-ingress": ("early", lambda value: value.__setitem__("allocation29_capture_bytes_consumed", True)),
        "tail-model": ("tail", lambda value: value.__setitem__("input_model_mismatched_bytes", 1)),
        "tail-canary": ("tail", lambda value: value.__setitem__("canary_mismatched_bytes", 1)),
        "tail-readonly": ("tail", lambda value: value.__setitem__("readonly_tail_mismatched_bytes", 1)),
    }
    rejected: list[dict[str, str]] = []
    for name, (target, mutate) in receipt_mutations.items():
        c, e, t = map(copy.deepcopy, (combined, early, tail))
        mutate({"combined": c, "early": e, "tail": t}[target])
        try:
            validate_combined(c, e, t, MAC_BACKEND, run_name, poison)
        except VerificationError as error:
            rejected.append({"mutation": name, "observed_error": str(error)})
        else:
            raise VerificationError(f"receipt mutation accepted: {name}")

    baseline = verify(mac_root, rtx_root, table_paths)
    overclaim = copy.deepcopy(baseline)
    overclaim.pop("receipt_root_sha256")
    overclaim["claims"]["subrect_profile_exact"] = True
    reroot(overclaim)
    try:
        validate_verification(overclaim)
    except VerificationError as error:
        rejected.append({
            "mutation": "scope-overclaim-subrect",
            "observed_error": str(error),
        })
    else:
        raise VerificationError("scope overclaim accepted")

    artifact = (run / "l20-l87" / "output.bin").read_bytes()
    mutated_artifact = bytearray(artifact)
    mutated_artifact[0] ^= 1
    require(sha256_bytes(mutated_artifact) != FINAL_SHA256,
            "output artifact mutation not detected")
    rejected.append({
        "mutation": "final-artifact-byte",
        "observed_error": "final artifact hash",
    })
    require(len(rejected) == len(receipt_mutations) + 2,
            "mutation rejection count")
    value = {
        "schema": "neuralstudio.fg.connected-distortion-l3-l91-negative-audit.v1",
        "baseline_verification_root_sha256":
            baseline["receipt_root_sha256"],
        "mutations": rejected,
        "mutation_count": len(rejected),
        "all_mutations_rejected": True,
        "preserved_execution_counterexamples": [
            "rtx/prototype-003 partial scatter targets retained poison",
            "macos/prototype-001 launch9 and shifted L52 first mismatches",
            "macos/prototype-004 module9 RCP-only hypothesis",
            "macos/prototype-005 exhaustive rounded half-div table hypothesis",
            "macos/prototype-006 texture conversion hypothesis",
        ],
        "result": "pass",
    }
    reroot(value)
    return value


def checkpoint_receipt(mac_root: Path, rtx_root: Path) -> dict[str, Any]:
    mac = validate_backend(mac_root, MAC_BACKEND)
    rtx = validate_backend(rtx_root, RTX_BACKEND)
    require(mac["checkpoint_root_sha256"] == rtx["checkpoint_root_sha256"],
            "checkpoint receipt cross-backend root")
    first_run = rtx_root / RUNS[0][0]
    checkpoints = load(first_run / "combined-receipt.json")["checkpoint_sha256"]
    require(canonical_sha256(checkpoints) == rtx["checkpoint_root_sha256"],
            "checkpoint receipt map root")
    value = {
        "schema": "neuralstudio.fg.connected-distortion-l3-l91-checkpoints.v1",
        "launch_range": [3, 91],
        "launch_count": 89,
        "checkpoint_sha256": checkpoints,
        "checkpoint_root_sha256": canonical_sha256(checkpoints),
        "rtx_mismatched_checkpoints": 0,
        "mac_mismatched_checkpoints": 0,
        "cross_backend_mismatched_checkpoints": 0,
        "two_poison_invariant": True,
        "result": "pass",
    }
    reroot(value)
    return value


def normalize_rig_receipt(path: Path) -> dict[str, Any]:
    raw = load(path)
    address_free(raw, "rig receipt")
    require(raw.get("schema") ==
            "neuralstudio.fg.connected-distortion-l3-l91-rig-receipt.v1",
            "rig schema")
    require(raw.get("job_id") ==
            "fg-connected-distortion-l3-l87-20260904T150406Z" and
            raw.get("attempt_id") == "prototype-010-final",
            "rig job identity")
    require(raw.get("host") == "NICO-RIG" and
            raw.get("gpu") == "NVIDIA GeForce RTX 5090" and
            raw.get("driver") == "616.56" and
            raw.get("vram_mib") == 32607 and
            "cuda_13.3" in raw.get("cuda_toolkit", ""),
            "rig environment")
    ids = raw.get("fresh_process_ids")
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == 4 and
            raw.get("exit_codes") == [0, 0, 0, 0] and
            raw.get("fresh_process_count") == 4,
            "rig fresh processes")
    require(raw.get("poison_bytes") == [90, 165] and
            raw.get("launch_range") == [3, 91] and
            raw.get("launch_count") == 89,
            "rig coverage")
    require(raw.get("binary_sha256") ==
            "e3331c6820c7bbce4d3fed9cfcb214b78f70305d44f4788f66b99fec3d0f5c1b" and
            raw.get("compile_log_sha256") ==
            "7fe6e97967dde782463aa404cddddf0d64ceedb8005cbc0dc0d2e39b49932156" and
            raw.get("backend_receipt_sha256") ==
            "a860d4d0d1bfd0b76d0f874c2214f6aff47cb02dc054d880d371dbace451b886",
            "rig binary/receipt pins")
    for key, expected in PROVENANCE_ROOTS.items():
        require(raw.get(key) == expected, f"rig {key}")
    require(raw.get("invariant_allocation37_sha256") == ALLOCATION37_SHA256 and
            raw.get("invariant_output_sha256") == FINAL_SHA256,
            "rig output pins")
    for field in (
        "all_checkpoint_two_poison_invariant",
        "canary_integrity",
        "input_model_immutability",
        "live_device_handoffs",
    ):
        require(raw.get(field) is True, f"rig {field}")
    require(raw.get("expected_or_archived_ingress") is False and
            raw.get("address_fields_recorded") is False and
            raw.get("result") == "pass", "rig ingress/result")
    value = {
        "schema": "neuralstudio.fg.connected-distortion-l3-l91-rig-normalized.v1",
        "job_id": raw["job_id"],
        "attempt_id": raw["attempt_id"],
        "remote_job_retained": raw["remote_job_retained"],
        "host": raw["host"],
        "os": raw["os"],
        "os_version": raw["os_version"],
        "gpu": raw["gpu"],
        "driver": raw["driver"],
        "vram_mib": raw["vram_mib"],
        "cuda_toolkit": raw["cuda_toolkit"],
        "build_command": raw["build_command"],
        "process_started_utc": raw["process_started_utc"],
        "receipt_completed_utc": raw["receipt_completed_utc"],
        "fresh_process_ids": ids,
        "fresh_process_count": 4,
        "poison_bytes": [90, 165],
        "launch_range": [3, 91],
        "launch_count": 89,
        "source_sha256": raw["source_sha256"],
        "input_manifests": raw["input_manifests"],
        **PROVENANCE_ROOTS,
        "binary_sha256": raw["binary_sha256"],
        "compile_log_bytes": raw["compile_log_bytes"],
        "compile_log_sha256": raw["compile_log_sha256"],
        "raw_receipt_sha256": sha256_file(path),
        "invariant_allocation37_sha256": ALLOCATION37_SHA256,
        "invariant_output_sha256": FINAL_SHA256,
        "all_checkpoint_two_poison_invariant": True,
        "canary_integrity": True,
        "input_model_immutability": True,
        "live_device_handoffs": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(value)
    return value


def source_pins(repository_root: Path) -> dict[str, Any]:
    rows = []
    for relative in SOURCE_PATHS:
        path = repository_root / relative
        require(path.is_file(), f"source pin missing: {relative}")
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    value = {
        "schema": "neuralstudio.fg.connected-distortion-l3-l91-source-pins.v1",
        "sources": rows,
        "source_manifest_root_sha256": canonical_sha256(rows),
        "reused_exact_contract_commits": {
            "production_16_frame_contract":
                "872bbcb0dc9aa86150a3a2d5cde18a6fb2de434f",
            "general_finite_nonpositive_sm120_ex2":
                "b18ce9fa6773c41bf9c3a1ceedecd7235591330f",
        },
        "reference_ex2_used_for_acceptance": True,
        "compact_ex2_promotion_in_this_receipt": False,
        "result": "pass",
    }
    reroot(value)
    return value


def table_arguments(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "sm120_ex2_unit_reference": args.ex2_unit,
        "sm120_ex2_fraction_reference": args.ex2_fraction,
        "sm120_rcp_correction_rom": args.rcp_rom,
        "sm120_half_logit_sigmoid_lut": args.sigmoid_lut,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "verify", "negative-audit", "backend", "checkpoints", "rig",
            "source-pins",
        ),
    )
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--backend", choices=(MAC_BACKEND, RTX_BACKEND))
    parser.add_argument("--ex2-unit", type=Path, required=True)
    parser.add_argument("--ex2-fraction", type=Path, required=True)
    parser.add_argument("--rcp-rom", type=Path, required=True)
    parser.add_argument("--sigmoid-lut", type=Path, required=True)
    parser.add_argument("--rig-receipt", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tables = table_arguments(args)
    if args.mode == "verify":
        result = verify(args.mac, args.rtx, tables)
    elif args.mode == "negative-audit":
        result = negative_audit(args.mac, args.rtx, tables)
    elif args.mode == "backend":
        require(args.backend is not None, "--backend required for backend mode")
        result = validate_backend(
            args.mac if args.backend == MAC_BACKEND else args.rtx,
            args.backend,
        )
    elif args.mode == "checkpoints":
        result = checkpoint_receipt(args.mac, args.rtx)
    elif args.mode == "rig":
        require(args.rig_receipt is not None, "--rig-receipt required")
        result = normalize_rig_receipt(args.rig_receipt)
    else:
        require(args.repository_root is not None, "--repository-root required")
        result = source_pins(args.repository_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise VerificationError(f"preserve existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, VerificationError) as error:
        raise SystemExit(f"verification_error={error}")
