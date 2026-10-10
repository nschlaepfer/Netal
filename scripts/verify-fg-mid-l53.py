#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Callable


RUN_SCHEMA = "neuralstudio.fg.mid-l53-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.mid-l53-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.mid-l53-verification.v1"
MODULE_SHA = "5a8e0284aab8ac14fc82b0504bbeef25d2fce1d13a1c11d8bdb3f91fee8145fc"
PARAMETER_SHA = "5a68951844e42596a4d20bc8d3d3db6ecff94927d7f6e96d66f7264c3fb895c7"
RCP_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
GENERATOR = "splitmix64-fg-mid-l53-two-flow-depth-v1"
SENTINEL = 0x7FFFFFFF7FFFFFFF
PIXELS = 320 * 180
OUTPUT_BYTES = PIXELS * 8
RUNS = tuple(
    (f"l53-{scenario}-poison-{poison}-process-{ordinal}", scenario,
     int(poison, 16))
    for scenario in ("small-flow", "pixel-flow")
    for poison in ("a5", "5a")
    for ordinal in ("a", "b")
)
EXPECTED = {
    "small-flow": {
        "output_a": "a73da0eb5ac98f26419bee1c8366f3f15f9f104787a69ecba10b9bfeee1d0f3a",
        "output_b": "8499aa682b30b88b05af3742f6de140bbc02dbd31dcc3a8969df18e2added737",
        "sentinels": [129, 0],
    },
    "pixel-flow": {
        "output_a": "a73da0eb5ac98f26419bee1c8366f3f15f9f104787a69ecba10b9bfeee1d0f3a",
        "output_b": "8d159c9aa4fc3696f3c3c66b29b2ab4069d5e88ef5843b163bb700d3d24aa950",
        "sentinels": [129, 11801],
    },
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode())


def rooted(value: dict[str, Any]) -> bool:
    copied = copy.deepcopy(value)
    actual = copied.pop("receipt_root_sha256", None)
    return isinstance(actual, str) and actual == canonical(copied)


def reject_addresses(value: Any, path: str = "root") -> None:
    forbidden = ("gpu_virtual_address", "resource_identity", "device_pointer",
                 "host_pointer", "descriptor_ptr")
    if isinstance(value, dict):
        for key, child in value.items():
            require(not any(item in key.lower() for item in forbidden),
                    f"{path}.{key}: volatile address")
            reject_addresses(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_addresses(child, f"{path}[{index}]")


def classify_output(data: bytes) -> dict[str, Any]:
    require(len(data) == OUTPUT_BYTES, "output size")
    values = struct.unpack(f"<{PIXELS}Q", data)
    sentinel_mask = bytes(value == SENTINEL for value in values)
    sentinel_count = sum(value == SENTINEL for value in values)
    partial_sentinel_high = sum(
        (value >> 32) == 0x7FFFFFFF and value != SENTINEL for value in values)
    signed_high = sum((value >> 63) != 0 for value in values)
    return {
        "sentinel_u64_count": sentinel_count,
        "non_sentinel_u64_count": PIXELS - sentinel_count,
        "sentinel_position_mask_sha256": sha(sentinel_mask),
        "partial_sentinel_high_count": partial_sentinel_high,
        "negative_signed_candidate_count": signed_high,
        "classified_u64_count": PIXELS,
        "unclassified_u64_count": 0,
    }


def validate_run(root: Path, run_id: str, scenario: str, poison: int,
                 backend: str) -> tuple[dict[str, Any], list[bytes], list[dict[str, Any]]]:
    receipt = load(root / run_id / "run-receipt.json")
    reject_addresses(receipt, run_id)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
    require(receipt.get("backend") ==
            ("rtx-cuda-driver" if backend == "rtx" else "mac-metal"),
            f"{run_id}: backend")
    require(receipt.get("run_id") == run_id, f"{run_id}: id")
    require(receipt.get("scenario") == scenario, f"{run_id}: scenario")
    require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
    require(isinstance(receipt.get("process_id"), int), f"{run_id}: process")
    require(receipt.get("selected_launch") == 53, f"{run_id}: launch")
    require(receipt.get("module_token") == 12, f"{run_id}: module")
    require(receipt.get("module_sha256") == MODULE_SHA, f"{run_id}: module hash")
    require(receipt.get("raw_live_parameter_sha256") == PARAMETER_SHA,
            f"{run_id}: parameter hash")
    require(receipt.get("input_generator") == GENERATOR,
            f"{run_id}: generator")
    require(receipt.get("packed_parameter_bytes") == 144, f"{run_id}: params")
    require((receipt.get("grid") or receipt.get("selected_grid")) == [20, 12, 1],
            f"{run_id}: grid")
    require((receipt.get("block") or receipt.get("selected_block")) == [324, 1, 1],
            f"{run_id}: block")
    require(receipt.get("dimensions") == [320, 180], f"{run_id}: dimensions")
    require(receipt.get("input_pre_sha256") == receipt.get("input_post_sha256"),
            f"{run_id}: input hashes")
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_id}: input")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: canary")
    require(receipt.get("expected_output_opened") is False,
            f"{run_id}: expected ingress")
    require(receipt.get("archived_intermediate_opened") is False,
            f"{run_id}: archived ingress")
    require(receipt.get("address_fields_recorded") is False,
            f"{run_id}: address declaration")
    require(receipt.get("result") == "pass", f"{run_id}: result")
    if backend == "mac":
        require(receipt.get("architecture_rom_sha256") == RCP_SHA,
                f"{run_id}: RCP ROM")
        require(receipt.get("atomic_lowering") ==
                "u32-high-min then equal-high-u32-low-min",
                f"{run_id}: atomic lowering")
    outputs = [(root / run_id / name).read_bytes()
               for name in ("output-a.bin", "output-b.bin")]
    require(all(len(value) == OUTPUT_BYTES for value in outputs),
            f"{run_id}: output sizes")
    hashes = [sha(value) for value in outputs]
    require(receipt.get("output_sha256") == hashes, f"{run_id}: output hashes")
    require(hashes == [EXPECTED[scenario]["output_a"],
                       EXPECTED[scenario]["output_b"]],
            f"{run_id}: frozen hashes")
    classifications = [classify_output(value) for value in outputs]
    require(receipt.get("sentinel_u64_count") ==
            [item["sentinel_u64_count"] for item in classifications],
            f"{run_id}: sentinel counts")
    require(receipt.get("non_sentinel_u64_count") ==
            [item["non_sentinel_u64_count"] for item in classifications],
            f"{run_id}: non-sentinel counts")
    require(all(item["partial_sentinel_high_count"] == 0 and
                item["negative_signed_candidate_count"] == 0 and
                item["unclassified_u64_count"] == 0
                for item in classifications), f"{run_id}: output classification")
    return receipt, outputs, classifications


def build_backend(root: Path, backend: str) -> dict[str, Any]:
    rows, output_map = [], {}
    for run_id, scenario, poison in RUNS:
        receipt, outputs, classifications = validate_run(
            root, run_id, scenario, poison, backend)
        output_map[run_id] = outputs
        rows.append({
            "run_id": run_id,
            "scenario": scenario,
            "poison_byte": poison,
            "process_id": receipt["process_id"],
            "input_sha256": receipt["input_pre_sha256"],
            "output_sha256": receipt["output_sha256"],
            "classification": classifications,
        })
    require(len({row["process_id"] for row in rows}) == 8,
            "fresh process gate")
    scenarios = {}
    for scenario in ("small-flow", "pixel-flow"):
        selected = [row for row in rows if row["scenario"] == scenario]
        require(len(selected) == 4, f"{scenario}: run count")
        first = output_map[selected[0]["run_id"]]
        require(all(output_map[row["run_id"]] == first for row in selected[1:]),
                f"{scenario}: repeat/poison invariance")
        classifications = selected[0]["classification"]
        scenarios[scenario] = {
            "output_sha256": selected[0]["output_sha256"],
            "sentinel_u64_count": [item["sentinel_u64_count"]
                                   for item in classifications],
            "non_sentinel_u64_count": [item["non_sentinel_u64_count"]
                                       for item in classifications],
            "sentinel_position_mask_sha256":
                [item["sentinel_position_mask_sha256"]
                 for item in classifications],
            "classified_u64_count": [PIXELS, PIXELS],
            "unclassified_u64_count": [0, 0],
        }
    document: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-cuda-driver" if backend == "rtx" else "mac-metal",
        "selected_unit": "evaluate-1/launch-000053/main_kernel/module-000012",
        "module_sha256": MODULE_SHA,
        "raw_live_parameter_sha256": PARAMETER_SHA,
        "input_generator": GENERATOR,
        "runs": rows,
        "scenarios": scenarios,
        "fresh_process_count": 8,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "complete_output_write_classification": True,
        "intentional_sentinel_positions_preserved": True,
        "input_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def build_final(rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx, mac = build_backend(rtx_root, "rtx"), build_backend(mac_root, "mac")
    comparisons = []
    for run_id, scenario, poison in RUNS:
        artifacts = []
        for name in ("output-a.bin", "output-b.bin"):
            left = (rtx_root / run_id / name).read_bytes()
            right = (mac_root / run_id / name).read_bytes()
            mismatch = sum(a != b for a, b in zip(left, right))
            require(len(left) == len(right) == OUTPUT_BYTES and mismatch == 0,
                    f"{run_id}/{name}: parity")
            artifacts.append({"artifact": name, "bytes": OUTPUT_BYTES,
                              "sha256": sha(left), "mismatched_bytes": 0})
        comparisons.append({"run_id": run_id, "scenario": scenario,
                            "poison_byte": poison, "artifacts": artifacts,
                            "mismatched_bytes": 0})
    document: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_launch_token": 53,
        "module_token": 12,
        "module_sha256": MODULE_SHA,
        "raw_live_parameter_sha256": PARAMETER_SHA,
        "input_generator": GENERATOR,
        "supported_domain": {
            "dimensions": [320, 180],
            "interpolation_factor": 0.5,
            "motion_threshold_bits": "0x414f5c29",
            "depth_threshold": 40.0,
            "depth_scale": 1.0,
            "invert_depth": False,
            "flow_scenarios": ["small-flow", "pixel-flow"],
            "atomic_candidate_high_sign": 0,
        },
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "sm120_rcp_rom_sha256": RCP_SHA,
        "fresh_process_count_per_backend": 8,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "complete_output_write_classification": True,
        "intentional_sentinel_positions_preserved": True,
        "all_outputs_rtx_metal_exact": True,
        "input_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "claims": {
            "launch53_standalone_exact": True,
            "connected_l52_to_l53_exact": False,
            "launch54_or_later_inferred": False,
            "complete_frame_exact": False,
            "netal_integrated": False,
            "performance_advantage_proven": False,
        },
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def validate_final(value: dict[str, Any]) -> None:
    reject_addresses(value)
    require(value.get("schema") == FINAL_SCHEMA, "final schema")
    require(rooted(value), "final root")
    require(value.get("all_outputs_rtx_metal_exact") is True, "final parity")
    require(value.get("complete_output_write_classification") is True,
            "final coverage")
    require(value.get("two_poison_output_invariance") is True, "final poison")
    require(value.get("expected_output_fed_to_execution") is False,
            "final expected ingress")
    require(value.get("archived_intermediates_consumed") is False,
            "final archived ingress")
    require(value.get("result") == "pass", "final result")
    require(len(value.get("comparisons", [])) == 8 and
            all(row.get("mismatched_bytes") == 0
                for row in value["comparisons"]), "final comparisons")


def build_audit(value: dict[str, Any]) -> dict[str, Any]:
    validate_final(value)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("output_mismatch", lambda x: x["comparisons"][0].__setitem__("mismatched_bytes", 1)),
        ("module_drift", lambda x: x.__setitem__("module_sha256", "0" * 64)),
        ("coverage_gate", lambda x: x.__setitem__("complete_output_write_classification", False)),
        ("poison_gate", lambda x: x.__setitem__("two_poison_output_invariance", False)),
        ("expected_ingress", lambda x: x.__setitem__("expected_output_fed_to_execution", True)),
        ("archived_ingress", lambda x: x.__setitem__("archived_intermediates_consumed", True)),
        ("address_leak", lambda x: x.__setitem__("gpu_virtual_address", "0x1")),
        ("result_drift", lambda x: x.__setitem__("result", "fail")),
    ]
    outcomes = []
    for name, mutation in mutations:
        changed = copy.deepcopy(value)
        mutation(changed)
        try:
            validate_final(changed)
        except VerificationError:
            outcomes.append({"mutation": name, "rejected": True})
            continue
        raise VerificationError(f"mutation accepted: {name}")
    document: dict[str, Any] = {
        "schema": "neuralstudio.fg.mid-l53-negative-audit.v1",
        "accepted_receipt_root_sha256": value["receipt_root_sha256"],
        "mutations": outcomes,
        "mutation_count": len(outcomes),
        "all_mutations_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
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
    p = commands.add_parser("backend")
    p.add_argument("--backend", choices=("rtx", "mac"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("final")
    p.add_argument("--rtx-root", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("audit")
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend":
            value = build_backend(args.root, args.backend)
        elif args.command == "final":
            value = build_final(args.rtx_root, args.mac_root)
        else:
            value = build_audit(load(args.accepted))
        write_exclusive(args.output, value)
        print(f"receipt_root_sha256={value['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (VerificationError, OSError) as error:
        print(f"verification_error={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
