#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Callable


RUN_SCHEMA = "neuralstudio.fg.early-l4-l14-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.early-l4-l14-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.early-l4-l14-verification.v1"
MODULE_SHA = "9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381"
GENERATOR = "fg-early-l4-l14-independent-inputs-v1"
VECTOR_BYTES = 320 * 180 * 8
DEPTH_BYTES = 320 * 180 * 4
PROBE_BYTES = 320 * 180 * 5 * 4
PROBE_SHA = "c5286036094506e212a8d04e0ac942032842d768e3c1eb4a26594caa4e3d28cf"
FIXTURES = {
    "branch-grid": {
        "vector_input": "6e0e17ca0515a0d8d39328cc943667eec77462e6506babbc8896552f16772929",
        "scalar_input": "4170cba10fad699dba1ebf749b6ff2022e4a67ddad2d9481c432401c3739ae89",
    },
    "center-tie": {
        "vector_input": "48cc01e4e20d6fd6438742963265cef88ce6fc3aef9f7c5a6163cdda4f37605e",
        "scalar_input": "452f1bb56b30aaecc0eec86edfdd8be71ad73ca6fa3e6cb6d6707290292bae3c",
    },
}
PHASES = {
    "reset": {
        "launch": 4,
        "raw": "2b822288e9427f60708f9e0f537f25484eb3c11309efc3755eccb693879249ce",
        "frozen_address_free": "59686bcaf449fd6d217aceb7c6faef7e1b2de50561eb504a9b1c8deed072ac20",
        "fixture_address_free": "e6ea4f8ae223c3b1683356c5168e4931ca00dc0f734acbfd678724a8b1d52c65",
    },
    "generated": {
        "launch": 14,
        "raw": "35284822ea5908b342267ed168ea193541f1b5d66a168182dfe0e26dcdb6a83c",
        "frozen_address_free": "c884fbfbf81a56cc50b3271754d228d2e2c3ba7eb1f05c5565318dbb291a21af",
        "fixture_address_free": "29144317759be7f820383ed92ea0191a1f157a02dcf3e74f3eea2b80c81123b6",
    },
}
OUTPUTS = {
    ("reset", "branch-grid"): {
        "vector": "22f26ff46068302d279b1ff2b45c40305a7013b453429463fe1a825671f9d4d5",
        "depth": "0cd62436334e36f1e446b4c99502e3a57ff12e4e63ae419f98cc0679a628298e",
        "depth_unique": 168,
    },
    ("generated", "branch-grid"): {
        "vector": "9fb02d2d9960fdbf71bcabf617ee7b5d80b9129de4068bc37d7df1747048b7c4",
        "depth": "f445fab79865825d2953fc3e1640a6a10a593615fcec81d41f76e3f68daa4b41",
        "depth_unique": 170,
    },
    ("reset", "center-tie"): {
        "vector": "f66176e2e53e8572c1e7b56fb9ec52598be72aa218ae5e77c205158d5bb25f01",
        "depth": "e67b2be09bf786ec9159b2b348528ee3a68bf1054d0bf4cdf6e8c7f756b7361f",
        "depth_unique": 1,
    },
    ("generated", "center-tie"): {
        "vector": "f66176e2e53e8572c1e7b56fb9ec52598be72aa218ae5e77c205158d5bb25f01",
        "depth": "e67b2be09bf786ec9159b2b348528ee3a68bf1054d0bf4cdf6e8c7f756b7361f",
        "depth_unique": 1,
    },
}
RUNS = tuple(
    (f"l{PHASES[phase]['launch']}-{fixture}-poison-{poison}-process-{ordinal}",
     phase, fixture, int(poison, 16))
    for fixture in ("branch-grid", "center-tie")
    for phase in ("reset", "generated")
    for poison in ("a5", "5a")
    for ordinal in ("a", "b")
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode())


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def rooted(value: dict[str, Any]) -> bool:
    copied = copy.deepcopy(value)
    actual = copied.pop("receipt_root_sha256", None)
    return isinstance(actual, str) and actual == canonical(copied)


def reroot(value: dict[str, Any]) -> None:
    value.pop("receipt_root_sha256", None)
    value["receipt_root_sha256"] = canonical(value)


def reject_addresses(value: Any, path: str = "root") -> None:
    forbidden = ("gpu_virtual_address", "device_pointer", "host_pointer",
                 "descriptor_ptr", "resource_identity")
    if isinstance(value, dict):
        for key, child in value.items():
            require(not any(item in key.lower() for item in forbidden),
                    f"{path}.{key}: volatile address")
            reject_addresses(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_addresses(child, f"{path}[{index}]")


def validate_run(root: Path, run_id: str, phase: str, fixture: str,
                 poison: int, backend: str) -> tuple[dict[str, Any], bytes, bytes]:
    receipt = load(root / run_id / "run-receipt.json")
    reject_addresses(receipt, run_id)
    phase_identity = PHASES[phase]
    fixture_identity = FIXTURES[fixture]
    output_identity = OUTPUTS[(phase, fixture)]
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
    require(receipt.get("backend") ==
            ("rtx-cuda-driver" if backend == "rtx" else "mac-metal"),
            f"{run_id}: backend")
    require(receipt.get("run_id") == run_id and
            receipt.get("phase") == phase and
            receipt.get("fixture") == fixture, f"{run_id}: identity")
    require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
    require(isinstance(receipt.get("process_id"), int), f"{run_id}: process")
    require(receipt.get("selected_launch") == phase_identity["launch"],
            f"{run_id}: launch")
    require((receipt.get("module_token") or
             receipt.get("source_module_token")) == 15, f"{run_id}: module")
    require((receipt.get("module_bytes") or
             receipt.get("source_module_bytes")) == 12632,
            f"{run_id}: module bytes")
    require((receipt.get("module_sha256") or
             receipt.get("source_module_sha256")) == MODULE_SHA,
            f"{run_id}: module hash")
    require((receipt.get("grid") or receipt.get("source_grid")) == [20, 12, 1]
            and (receipt.get("block") or
                 receipt.get("source_block")) == [16, 16, 1],
            f"{run_id}: dispatch")
    require(receipt.get("packed_parameter_bytes") == 160,
            f"{run_id}: parameter bytes")
    require(receipt.get("frozen_selected_raw_parameter_sha256") ==
            phase_identity["raw"], f"{run_id}: raw parameter provenance")
    require(receipt.get("frozen_selected_address_free_parameter_sha256") ==
            phase_identity["frozen_address_free"],
            f"{run_id}: selected address-free provenance")
    require(receipt.get("fixture_address_free_parameter_sha256") ==
            phase_identity["fixture_address_free"],
            f"{run_id}: fixture parameter provenance")
    require(receipt.get("input_generator") == GENERATOR,
            f"{run_id}: input generator")
    require(receipt.get("vector_input_pre_sha256") ==
            receipt.get("vector_input_post_sha256") ==
            fixture_identity["vector_input"], f"{run_id}: vector input")
    require(receipt.get("scalar_input_pre_sha256") ==
            receipt.get("scalar_input_post_sha256") ==
            fixture_identity["scalar_input"], f"{run_id}: scalar input")
    vector = (root / run_id / "vector-output.bin").read_bytes()
    depth = (root / run_id / "depth-output.bin").read_bytes()
    require(len(vector) == receipt.get("vector_output_bytes") == VECTOR_BYTES
            and sha(vector) == receipt.get("vector_output_sha256") ==
            output_identity["vector"], f"{run_id}: vector output")
    require(len(depth) == receipt.get("depth_output_bytes") == DEPTH_BYTES
            and sha(depth) == receipt.get("depth_output_sha256") ==
            output_identity["depth"], f"{run_id}: depth output")
    require(receipt.get("depth_unique_u32_values") ==
            output_identity["depth_unique"], f"{run_id}: depth branches")
    for name in ("input_mismatched_bytes", "vector_canary_mismatched_bytes",
                 "depth_canary_mismatched_bytes", "vector_untouched_pixels",
                 "depth_untouched_pixels"):
        require(receipt.get(name) == 0, f"{run_id}: {name}")
    require(receipt.get("expected_output_fed_to_execution") is False,
            f"{run_id}: expected ingress")
    require(receipt.get("archived_intermediates_consumed") is False,
            f"{run_id}: archive ingress")
    require(receipt.get("address_fields_recorded") is False,
            f"{run_id}: addresses")
    require(receipt.get("result") == "pass", f"{run_id}: result")
    return receipt, vector, depth


def build_backend(root: Path, backend: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    outputs: dict[str, tuple[bytes, bytes]] = {}
    for run_id, phase, fixture, poison in RUNS:
        receipt, vector, depth = validate_run(
            root, run_id, phase, fixture, poison, backend)
        outputs[run_id] = vector, depth
        rows.append({"run_id": run_id, "phase": phase, "fixture": fixture,
                     "poison_byte": poison,
                     "process_id": receipt["process_id"],
                     "vector_output_sha256": receipt["vector_output_sha256"],
                     "depth_output_sha256": receipt["depth_output_sha256"]})
    require(len({row["process_id"] for row in rows}) == 16,
            "fresh process gate")
    scenarios: dict[str, Any] = {}
    for phase in ("reset", "generated"):
        for fixture in ("branch-grid", "center-tie"):
            selected = [row for row in rows
                        if row["phase"] == phase and row["fixture"] == fixture]
            require(len(selected) == 4, f"{phase}/{fixture}: run count")
            first = outputs[selected[0]["run_id"]]
            require(all(outputs[row["run_id"]] == first
                        for row in selected[1:]),
                    f"{phase}/{fixture}: poison/process invariance")
            identity = OUTPUTS[(phase, fixture)]
            scenarios[f"{phase}/{fixture}"] = {
                "selected_launch": PHASES[phase]["launch"],
                "vector_input_sha256": FIXTURES[fixture]["vector_input"],
                "scalar_input_sha256": FIXTURES[fixture]["scalar_input"],
                "vector_output_sha256": identity["vector"],
                "depth_output_sha256": identity["depth"],
                "depth_unique_u32_values": identity["depth_unique"],
            }
    document: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-cuda-driver" if backend == "rtx" else "mac-metal",
        "selected_launches": [4, 14],
        "module_token": 15,
        "module_sha256": MODULE_SHA,
        "input_generator": GENERATOR,
        "fixture_count": 2,
        "runs": rows,
        "scenarios": scenarios,
        "fresh_process_count": 16,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "input_immutability": True,
        "full_surface_write_coverage": True,
        "surface_tail_canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(document)
    return document


def decode_coordinate(bits: int) -> tuple[int, int]:
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    require(value.is_integer() and 0 <= value < 360 * 1024 + 640,
            "coordinate probe value domain")
    integer = int(value)
    return integer % 1024, integer // 1024


def build_coordinate_receipt(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    require(len(data) == PROBE_BYTES and sha(data) == PROBE_SHA,
            "coordinate probe identity")
    labels = ("center", "lower_left", "lower_right", "upper_left",
              "upper_right")
    observed: dict[str, list[list[tuple[int, int]]]] = {
        label: [[(0, 0) for _ in range(320)] for _ in range(180)]
        for label in labels
    }
    for y in range(180):
        for x in range(320):
            words = struct.unpack_from("<5I", data, (y * 320 + x) * 20)
            for label, word in zip(labels, words):
                observed[label][y][x] = decode_coordinate(word)
    negative_x_collapses = list(range(260, 320, 5))
    negative_y_collapses = [135, 148]

    def negative_x(x: int) -> int:
        if x == 0 or x in negative_x_collapses:
            return 2 * x
        return 2 * x - 1

    def positive_x(x: int) -> int:
        return min(639, 2 * x + 1)

    def negative_y(y: int) -> int:
        if y == 0 or y in negative_y_collapses:
            return 2 * y
        return 2 * y - 1

    def positive_y(y: int) -> int:
        return min(359, 2 * y + 1)

    for y in range(180):
        for x in range(320):
            expected = {
                "center": (2 * x, 2 * y),
                "lower_left": (negative_x(x), negative_y(y)),
                "lower_right": (positive_x(x), negative_y(y)),
                "upper_left": (negative_x(x), positive_y(y)),
                "upper_right": (positive_x(x), positive_y(y)),
            }
            for label in labels:
                require(observed[label][y][x] == expected[label],
                        f"coordinate map mismatch {label}/{x}/{y}")
    document: dict[str, Any] = {
        "schema": "neuralstudio.fg.early-l4-l14-coordinate-map.v1",
        "backend": "rtx-sm120-cuda-texture-object",
        "probe_sha256": PROBE_SHA,
        "probe_bytes": PROBE_BYTES,
        "output_positions": 320 * 180,
        "candidate_samples": 320 * 180 * 5,
        "source_extent": [640, 360],
        "candidate_order": list(labels),
        "center_map": "source=(2*x,2*y)",
        "positive_maps": "source=(2*axis+1) clamped to final pixel",
        "negative_x_center_collapse_output_indices": negative_x_collapses,
        "negative_y_center_collapse_output_indices": negative_y_collapses,
        "all_candidate_coordinates_classified": True,
        "probe_data_fed_to_model_execution": False,
        "expected_or_archived_output_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(document)
    return document


def validate_coordinate(value: dict[str, Any]) -> None:
    reject_addresses(value)
    require(value.get("schema") ==
            "neuralstudio.fg.early-l4-l14-coordinate-map.v1" and
            rooted(value), "coordinate identity/root")
    require(value.get("backend") == "rtx-sm120-cuda-texture-object" and
            value.get("probe_sha256") == PROBE_SHA and
            value.get("probe_bytes") == PROBE_BYTES,
            "coordinate probe provenance")
    require(value.get("output_positions") == 320 * 180 and
            value.get("candidate_samples") == 320 * 180 * 5 and
            value.get("source_extent") == [640, 360],
            "coordinate probe domain")
    require(value.get("candidate_order") == [
        "center", "lower_left", "lower_right", "upper_left", "upper_right"
    ] and value.get("center_map") == "source=(2*x,2*y)" and
            value.get("positive_maps") ==
            "source=(2*axis+1) clamped to final pixel",
            "coordinate map")
    require(value.get("negative_x_center_collapse_output_indices") ==
            list(range(260, 320, 5)) and
            value.get("negative_y_center_collapse_output_indices") ==
            [135, 148], "coordinate exceptions")
    require(value.get("all_candidate_coordinates_classified") is True and
            value.get("probe_data_fed_to_model_execution") is False and
            value.get("expected_or_archived_output_consumed") is False and
            value.get("address_fields_recorded") is False and
            value.get("result") == "pass", "coordinate gates")


def build_final(rtx_root: Path, mac_root: Path,
                coordinate_receipt: dict[str, Any]) -> dict[str, Any]:
    validate_coordinate(coordinate_receipt)
    rtx = build_backend(rtx_root, "rtx")
    mac = build_backend(mac_root, "mac")
    comparisons = []
    for run_id, phase, fixture, poison in RUNS:
        vector_rtx = (rtx_root / run_id / "vector-output.bin").read_bytes()
        vector_mac = (mac_root / run_id / "vector-output.bin").read_bytes()
        depth_rtx = (rtx_root / run_id / "depth-output.bin").read_bytes()
        depth_mac = (mac_root / run_id / "depth-output.bin").read_bytes()
        vector_mismatch = sum(a != b for a, b in zip(vector_rtx, vector_mac))
        depth_mismatch = sum(a != b for a, b in zip(depth_rtx, depth_mac))
        require(len(vector_rtx) == len(vector_mac) == VECTOR_BYTES and
                vector_mismatch == 0, f"{run_id}: vector parity")
        require(len(depth_rtx) == len(depth_mac) == DEPTH_BYTES and
                depth_mismatch == 0, f"{run_id}: depth parity")
        comparisons.append({
            "run_id": run_id,
            "phase": phase,
            "fixture": fixture,
            "poison_byte": poison,
            "vector_bytes": VECTOR_BYTES,
            "vector_sha256": sha(vector_rtx),
            "vector_mismatched_bytes": 0,
            "depth_bytes": DEPTH_BYTES,
            "depth_sha256": sha(depth_rtx),
            "depth_mismatched_bytes": 0,
        })
    document: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_launches": [4, 14],
        "module_token": 15,
        "module_sha256": MODULE_SHA,
        "fixture_domain": ["branch-grid", "center-tie"],
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "coordinate_map_root_sha256":
            coordinate_receipt["receipt_root_sha256"],
        "fresh_process_count_per_backend": 16,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "input_immutability": True,
        "full_surface_write_coverage": True,
        "surface_tail_canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "claims": {
            "launch4_reset_standalone_exact_on_fixture_domain": True,
            "launch14_generated_standalone_exact_on_fixture_domain": True,
            "all_finite_r32f_inputs_proven": False,
            "connected_early_group_proven": False,
            "complete_frame_metal_parity_proven": False,
            "netal_integrated": False,
            "performance_advantage_proven": False,
        },
        "result": "pass",
    }
    reroot(document)
    return document


def validate_final(value: dict[str, Any], coordinate_root: str) -> None:
    reject_addresses(value)
    require(value.get("schema") == FINAL_SCHEMA and rooted(value),
            "final identity/root")
    require(value.get("selected_launches") == [4, 14] and
            value.get("module_token") == 15 and
            value.get("module_sha256") == MODULE_SHA, "selected identity")
    require(value.get("fixture_domain") == ["branch-grid", "center-tie"],
            "fixture domain")
    require(value.get("coordinate_map_root_sha256") == coordinate_root,
            "coordinate provenance")
    require(value.get("fresh_process_count_per_backend") == 16 and
            value.get("two_process_repeat_per_poison") is True and
            value.get("two_poison_output_invariance") is True,
            "process/poison gates")
    require(value.get("input_immutability") is True and
            value.get("full_surface_write_coverage") is True and
            value.get("surface_tail_canary_integrity") is True,
            "memory gates")
    require(value.get("expected_output_fed_to_execution") is False and
            value.get("archived_intermediates_consumed") is False and
            value.get("address_fields_recorded") is False,
            "ingress/address gates")
    expected_comparisons = [{
        "run_id": run_id,
        "phase": phase,
        "fixture": fixture,
        "poison_byte": poison,
        "vector_bytes": VECTOR_BYTES,
        "vector_sha256": OUTPUTS[(phase, fixture)]["vector"],
        "vector_mismatched_bytes": 0,
        "depth_bytes": DEPTH_BYTES,
        "depth_sha256": OUTPUTS[(phase, fixture)]["depth"],
        "depth_mismatched_bytes": 0,
    } for run_id, phase, fixture, poison in RUNS]
    require(value.get("comparisons") == expected_comparisons,
            "parity comparisons")
    for name in ("rtx_backend_root_sha256", "mac_backend_root_sha256"):
        require(isinstance(value.get(name), str) and len(value[name]) == 64,
                f"{name} missing")
    require(value.get("claims") == {
        "launch4_reset_standalone_exact_on_fixture_domain": True,
        "launch14_generated_standalone_exact_on_fixture_domain": True,
        "all_finite_r32f_inputs_proven": False,
        "connected_early_group_proven": False,
        "complete_frame_metal_parity_proven": False,
        "netal_integrated": False,
        "performance_advantage_proven": False,
    }, "claim boundary")
    require(value.get("result") == "pass", "result")


def build_audit(value: dict[str, Any], coordinate_root: str) -> dict[str, Any]:
    validate_final(value, coordinate_root)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("vector_output_mismatch", lambda x: x["comparisons"][0].__setitem__(
            "vector_mismatched_bytes", 1)),
        ("depth_output_mismatch", lambda x: x["comparisons"][0].__setitem__(
            "depth_mismatched_bytes", 1)),
        ("module_drift", lambda x: x.__setitem__("module_sha256", "0" * 64)),
        ("coordinate_provenance", lambda x: x.__setitem__(
            "coordinate_map_root_sha256", "0" * 64)),
        ("coverage_gate", lambda x: x.__setitem__(
            "full_surface_write_coverage", False)),
        ("canary_gate", lambda x: x.__setitem__(
            "surface_tail_canary_integrity", False)),
        ("poison_gate", lambda x: x.__setitem__(
            "two_poison_output_invariance", False)),
        ("expected_ingress", lambda x: x.__setitem__(
            "expected_output_fed_to_execution", True)),
        ("archive_ingress", lambda x: x.__setitem__(
            "archived_intermediates_consumed", True)),
        ("scope_expansion", lambda x: x["claims"].__setitem__(
            "connected_early_group_proven", True)),
    ]
    outcomes = []
    for name, mutation in mutations:
        changed = copy.deepcopy(value)
        mutation(changed)
        reroot(changed)
        try:
            validate_final(changed, coordinate_root)
        except VerificationError:
            outcomes.append({"mutation": name, "rejected": True})
            continue
        raise VerificationError(f"mutation accepted: {name}")
    document: dict[str, Any] = {
        "schema": "neuralstudio.fg.early-l4-l14-negative-audit.v1",
        "accepted_receipt_root_sha256": value["receipt_root_sha256"],
        "coordinate_map_root_sha256": coordinate_root,
        "mutations": outcomes,
        "mutation_count": len(outcomes),
        "all_mutations_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(document)
    return document


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise VerificationError(f"refusing to overwrite {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("backend")
    sub.add_argument("--backend", choices=("rtx", "mac"), required=True)
    sub.add_argument("--root", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("coordinate")
    sub.add_argument("--probe", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("final")
    sub.add_argument("--rtx-root", type=Path, required=True)
    sub.add_argument("--mac-root", type=Path, required=True)
    sub.add_argument("--coordinate-receipt", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("audit")
    sub.add_argument("--accepted", type=Path, required=True)
    sub.add_argument("--coordinate-receipt", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend":
            value = build_backend(args.root, args.backend)
        elif args.command == "coordinate":
            value = build_coordinate_receipt(args.probe)
        elif args.command == "final":
            value = build_final(args.rtx_root, args.mac_root,
                                load(args.coordinate_receipt))
        else:
            coordinate = load(args.coordinate_receipt)
            validate_coordinate(coordinate)
            value = build_audit(load(args.accepted),
                                coordinate["receipt_root_sha256"])
        write_exclusive(args.output, value)
        print(f"receipt_root_sha256={value['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (VerificationError, OSError) as error:
        print(f"verification_error={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
