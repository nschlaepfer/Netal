#!/usr/bin/env python3
"""Freeze the conditional OutputInterpolationStatus launch pair.

The pinned FG runtime loads module 27 in every process.  It launches that
module once per submitted frame only when the optional disable/status output
resource is supplied.  This builder compares two fresh processes for each
resource profile and proves that removing those two launches leaves the same
ordered 83-launch core graph.

No captured output or intermediate is executed or fed back to the runtime.
The inputs are immutable trace and run receipts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
INTAKE = REPOSITORY / "build" / "fg_rig" / "fg2x-intake-20260904T082417Z" / "output"
SENTINEL = REPOSITORY / "build" / "fg_rig" / "fg-opaque-sentinel-20260904T093500Z" / "output"
DEFAULT_OUTPUT = (
    REPOSITORY
    / "evidence"
    / "specs"
    / "fg-optional-disable-path-v1"
    / "optional-disable-path.json"
)
DEFAULT_RUNS = {
    "with_disable_a": INTAKE / "phase-core-5a-run-a",
    "with_disable_b": INTAKE / "phase-core-a5-run-b",
    "core_a": SENTINEL / "sentinel-core-5a-a",
    "core_b": SENTINEL / "sentinel-core-a5-b",
}
TRACE_SUFFIX = "-trace"
MODULE_TOKEN = "module-000027"
FUNCTION_NAME = "main_kernel"
EXPECTED_OUTPUT = "07a87435b2e7d79540dcec737b9abb6354e68e8595b4df29ca6f8d92cce54383"
SCHEMA = "neuralstudio.fg.optional-disable-path.v1"


class EvidenceError(RuntimeError):
    """Raised when the conditional path evidence fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return path.name


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_trace(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(path.is_file() and not path.is_symlink(), f"missing regular trace: {path}")
    launches: list[dict[str, Any]] = []
    modules: list[dict[str, Any]] = []
    phase = "runtime"
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvidenceError(f"cannot read trace {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"invalid trace JSON {path}:{line_number}: {error}") from error
        require(isinstance(row, dict), f"trace row is not an object: {path}:{line_number}")
        if row.get("event") == "phase_marker":
            phase = str(row.get("phase"))
        elif row.get("event") == "kernel_launch":
            launches.append(
                {
                    "phase": phase,
                    "module_token": row.get("module_token"),
                    "function_name": row.get("function_name"),
                    "grid": row.get("grid"),
                    "block": row.get("block"),
                    "dynamic_shared_bytes": row.get("dynamic_shared_bytes"),
                    "parameter_bytes": row.get("parameter_bytes"),
                    "parameter_file": row.get("parameter_file"),
                    "launch_token": row.get("launch_token"),
                }
            )
        elif (
            row.get("event") == "nvapi_call"
            and row.get("call") == "NvAPI_D3D12_CreateCuModule"
            and row.get("status") == 0
        ):
            modules.append(
                {
                    "module_token": row.get("module_token"),
                    "bytes": row.get("module_blob_bytes"),
                    "sha256": str(row.get("module_blob_sha256", "")).lower(),
                }
            )
    return launches, modules


def structural_launch(launch: dict[str, Any]) -> dict[str, Any]:
    """Drop process-local launch tokens and dump filenames."""

    return {
        key: launch[key]
        for key in (
            "phase",
            "module_token",
            "function_name",
            "grid",
            "block",
            "dynamic_shared_bytes",
            "parameter_bytes",
        )
    }


def compare_launch_sequences(
    with_disable: list[dict[str, Any]], core: list[dict[str, Any]]
) -> list[int]:
    conditional_indices = [
        index
        for index, launch in enumerate(with_disable)
        if launch["module_token"] == MODULE_TOKEN
    ]
    require(conditional_indices == [7, 9], "conditional module must occur at indices 7 and 9")
    stripped = [
        structural_launch(launch)
        for launch in with_disable
        if launch["module_token"] != MODULE_TOKEN
    ]
    projected_core = [structural_launch(launch) for launch in core]
    require(stripped == projected_core, "83-launch core structure differs after removal")
    return conditional_indices


def run_paths(root: Path) -> tuple[Path, Path]:
    receipt = root / "fg-run-receipt.json"
    trace = root.parent / f"{root.name}{TRACE_SUFFIX}" / "nvapi-trace.jsonl"
    return receipt, trace


def load_run(name: str, root: Path) -> dict[str, Any]:
    receipt_path, trace_path = run_paths(root)
    receipt = load_json(receipt_path)
    launches, modules = load_trace(trace_path)
    require(len(receipt.get("frames", [])) == 2, f"{name}: expected two frames")
    frames = receipt["frames"]
    require(frames[1]["output_interpolated_sha256"] == EXPECTED_OUTPUT, f"{name}: output drift")
    require(receipt.get("runtime_sha256") == "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b", f"{name}: runtime drift")
    return {
        "name": name,
        "root": root,
        "receipt_path": receipt_path,
        "trace_path": trace_path,
        "receipt": receipt,
        "launches": launches,
        "modules": modules,
    }


def build_receipt(run_roots: dict[str, Path] | None = None) -> dict[str, Any]:
    roots = run_roots or DEFAULT_RUNS
    require(set(roots) == set(DEFAULT_RUNS), "run set is incomplete")
    runs = {name: load_run(name, path) for name, path in roots.items()}

    with_a = runs["with_disable_a"]
    with_b = runs["with_disable_b"]
    core_a = runs["core_a"]
    core_b = runs["core_b"]

    with_structure = [structural_launch(item) for item in with_a["launches"]]
    require(with_structure == [structural_launch(item) for item in with_b["launches"]], "85-launch profile differs across processes")
    core_structure = [structural_launch(item) for item in core_a["launches"]]
    require(core_structure == [structural_launch(item) for item in core_b["launches"]], "83-launch profile differs across processes")
    require(len(with_structure) == 85, "disable-output profile must have 85 launches")
    require(len(core_structure) == 83, "core profile must have 83 launches")
    require(sum(item["phase"] == "evaluate_0" for item in with_structure) == 9, "disable profile evaluate_0 count drift")
    require(sum(item["phase"] == "evaluate_1" for item in with_structure) == 76, "disable profile evaluate_1 count drift")
    require(sum(item["phase"] == "evaluate_0" for item in core_structure) == 8, "core evaluate_0 count drift")
    require(sum(item["phase"] == "evaluate_1" for item in core_structure) == 75, "core evaluate_1 count drift")
    conditional_indices = compare_launch_sequences(with_a["launches"], core_a["launches"])

    expected_modules = with_a["modules"]
    require(len(expected_modules) == 64, "expected 64 selected modules")
    for run in runs.values():
        require(run["modules"] == expected_modules, f"{run['name']}: selected module set drift")

    profiles = {
        name: run["receipt"].get("resource_profile") for name, run in runs.items()
    }
    require(profiles["with_disable_a"] == profiles["with_disable_b"] == "core+disable_output", "disable profile label drift")
    require(profiles["core_a"] == profiles["core_b"] == "core", "core profile label drift")
    poisons = {name: run["receipt"].get("output_poison_byte") for name, run in runs.items()}
    require(poisons == {"with_disable_a": 0x5A, "with_disable_b": 0xA5, "core_a": 0x5A, "core_b": 0xA5}, "poison matrix drift")
    process_ids = [run["receipt"].get("process_id") for run in runs.values()]
    require(all(isinstance(value, int) and value > 0 for value in process_ids), "invalid process id")
    require(len(set(process_ids)) == 4, "runs are not four fresh processes")

    conditional_launches: list[dict[str, Any]] = []
    for run in (with_a, with_b):
        selected = [item for item in run["launches"] if item["module_token"] == MODULE_TOKEN]
        require(len(selected) == 2, f"{run['name']}: conditional launch count drift")
        parsed: list[dict[str, Any]] = []
        for launch in selected:
            dump_path = run["trace_path"].parent / "dumps" / str(launch["parameter_file"])
            require(dump_path.is_file() and not dump_path.is_symlink(), f"missing parameter dump: {dump_path}")
            data = dump_path.read_bytes()
            require(len(data) == 16, "conditional parameter ABI must be 16 bytes")
            pointer, element_count, value = struct.unpack("<QII", data)
            require(pointer != 0, "conditional output pointer must be bound")
            normalized = struct.pack("<QII", 0, element_count, value)
            parsed.append(
                {
                    "phase": launch["phase"],
                    "element_count": element_count,
                    "value": value,
                    "parameter_bytes": len(data),
                    "address_free_parameter_sha256": hashlib.sha256(normalized).hexdigest(),
                }
            )
        require([(item["phase"], item["element_count"], item["value"]) for item in parsed] == [("evaluate_0", 1, 1), ("evaluate_1", 1, 0)], f"{run['name']}: conditional parameter values drift")
        conditional_launches.append({"run": run["name"], "launches": parsed})
    require(conditional_launches[0]["launches"] == conditional_launches[1]["launches"], "conditional parameters differ across poisons")

    for run in (with_a, with_b):
        frames = run["receipt"]["frames"]
        require([frame["disable_output_provided"] for frame in frames] == [True, True], "disable output must be bound")
        require([frame["disable_interpolation"] for frame in frames] == [True, False], "disable values drift")
    for run in (core_a, core_b):
        frames = run["receipt"]["frames"]
        require([frame["disable_output_provided"] for frame in frames] == [False, False], "core must omit disable output")
        require([frame["disable_interpolation"] for frame in frames] == [False, False], "core disable state drift")

    sources = []
    for name, run in runs.items():
        sources.append(
            {
                "run": name,
                "receipt": relative(run["receipt_path"]),
                "receipt_sha256": sha256_file(run["receipt_path"]),
                "trace": relative(run["trace_path"]),
                "trace_sha256": sha256_file(run["trace_path"]),
                "poison": poisons[name],
            }
        )

    value: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "fresh_processes": 4,
            "source_files": sources,
            "runtime_sha256": with_a["receipt"]["runtime_sha256"],
            "selected_module_count": len(expected_modules),
            "selected_module_set_sha256": canonical_sha256(expected_modules),
        },
        "profiles": {
            "core_plus_disable_output": {
                "launches": 85,
                "evaluate_0_launches": 9,
                "evaluate_1_launches": 76,
                "two_process_structure_equal": True,
                "generated_frame_sha256": EXPECTED_OUTPUT,
            },
            "core": {
                "launches": 83,
                "evaluate_0_launches": 8,
                "evaluate_1_launches": 75,
                "two_process_structure_equal": True,
                "generated_frame_sha256": EXPECTED_OUTPUT,
            },
        },
        "conditional_path": {
            "module_token": MODULE_TOKEN,
            "module_sha256": next(item["sha256"] for item in expected_modules if item["module_token"] == MODULE_TOKEN),
            "function_name": FUNCTION_NAME,
            "indices_in_85_launch_sequence": conditional_indices,
            "launch_geometry": {"grid": [1, 1, 1], "block": [64, 1, 1]},
            "address_free_parameter_abi": ["u64 output_pointer", "u32 element_count", "u32 value"],
            "parameter_observations": conditional_launches[0]["launches"],
            "two_process_parameters_equal": True,
        },
        "comparison": {
            "selected_module_set_equal": True,
            "core_structure_after_removing_conditional_launches_equal": True,
            "common_core_launches": 83,
            "generated_frame_equal": True,
        },
        "claims": {
            "canonical_85_launch_count_is_resource_profile_specific": True,
            "module_27_is_loaded_even_when_not_launched": True,
            "conditional_pair_belongs_to_optional_disable_status_resource": True,
            "arithmetic_model_path_changed": False,
            "module_27_metal_parity_proven": False,
            "full_frame_metal_parity_proven": False,
        },
    }
    value["receipt_root_sha256"] = canonical_sha256(value)
    return value


def verify_receipt(value: dict[str, Any]) -> None:
    require(value.get("schema") == SCHEMA, "receipt schema drift")
    declared = value.get("receipt_root_sha256")
    require(isinstance(declared, str), "receipt root is missing")
    payload = dict(value)
    payload.pop("receipt_root_sha256", None)
    require(canonical_sha256(payload) == declared, "receipt root mismatch")
    require(value["comparison"]["common_core_launches"] == 83, "common core count drift")
    require(value["claims"]["full_frame_metal_parity_proven"] is False, "forbidden full-frame claim")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        value = build_receipt()
        verify_receipt(value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    except EvidenceError as error:
        print(f"error: {error}")
        return 1
    print(args.output)
    print(value["receipt_root_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
