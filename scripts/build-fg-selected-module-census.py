#!/usr/bin/env python3
"""Build an address-free census of the live-selected DLSS-FG CUDA path.

The builder consumes two phase-marked NvAPI traces plus job-scoped cuobjdump
metadata.  It deliberately selects only successful live CreateCuModule calls;
it never scans a runtime binary for dormant variants.  Module and launch
parameter dumps are rehashed, the two processes are compared structurally,
and every differing parameter byte is retained as an explicit offset/value
record rather than normalized away.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_JOB_OUTPUT = (
    REPOSITORY
    / "build"
    / "fg_rig"
    / "fg2x-intake-20260904T082417Z"
    / "output"
)
DEFAULT_RUN_A = DEFAULT_JOB_OUTPUT / "phase-core-5a-run-a-trace"
DEFAULT_RUN_B = DEFAULT_JOB_OUTPUT / "phase-core-a5-run-b-trace"
DEFAULT_RIG_INSPECTION = DEFAULT_JOB_OUTPUT / "fg-selected-module-census-v1"
DEFAULT_ABI_INSPECTION = DEFAULT_JOB_OUTPUT / "fg-selected-module-census-v1-abi"
DEFAULT_OUTPUT = (
    REPOSITORY
    / "evidence"
    / "specs"
    / "fg-selected-module-census-v1"
    / "selected-module-census.json"
)
SCHEMA_PATH = (
    REPOSITORY
    / "evidence"
    / "specs"
    / "fg-selected-module-census-v1"
    / "census.schema.json"
)

TRACE_SCHEMA = "neuralstudio.fg.nvapi-forward-trace.v1"
RIG_SCHEMA = "neuralstudio.fg.selected-module-rig-inspection.v1"
ABI_SCHEMA = "neuralstudio.fg.selected-module-abi-inspection.v1"
CENSUS_SCHEMA = "neuralstudio.fg.selected-module-census.v1"
CREATE_MODULE = "NvAPI_D3D12_CreateCuModule"
CREATE_FUNCTION = "NvAPI_D3D12_CreateCuFunction"
LAUNCH_CHAIN = "NvAPI_D3D12_LaunchCuKernelChain"
RESOURCE_CALLS = {
    "NvAPI_D3D12_GetCudaMergedTextureSamplerObject",
    "NvAPI_D3D12_GetCudaIndependentDescriptorObject",
}
EXPECTED_PHASES = [
    "pre_device",
    "requirements",
    "init",
    "post_init",
    "create",
    "create.end",
    "evaluate_0",
    "evaluate_0.end",
    "evaluate_1",
    "evaluate_1.end",
    "release",
    "release.end",
    "shutdown",
    "shutdown.end",
]


class CensusError(RuntimeError):
    """Raised when any census evidence fails closed."""


@dataclass(frozen=True)
class TraceRun:
    run_id: str
    root: Path
    trace_path: Path
    trace_sha256: str
    rows: list[dict[str, Any]]
    phases: list[str]
    modules: list[dict[str, Any]]
    functions: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    launches: list[dict[str, Any]]
    phase_event_counts: dict[str, dict[str, int]]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CensusError(f"cannot read JSON {path}: {error}") from error


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise CensusError(f"cannot read text {path}: {error}") from error


def relative_display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return path.name


def require_regular_file(path: Path, label: str) -> None:
    require(path.exists(), f"{label} missing: {path}")
    require(path.is_file(), f"{label} is not a regular file: {path}")
    require(not path.is_symlink(), f"{label} may not be a symlink: {path}")


def require_token(value: Any, prefix: str, label: str) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    require(
        re.fullmatch(rf"{re.escape(prefix)}-\d{{6}}", value) is not None,
        f"invalid {label}: {value}",
    )
    return value


def require_triplet(value: Any, label: str) -> list[int]:
    require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int) and item >= 0 for item in value),
        f"{label} must be a non-negative integer triplet",
    )
    return list(value)


def event_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Return the address-free structural portion of a trace row."""

    event = row.get("event")
    phase = row.get("_phase")
    if event == "runtime_ready":
        return {
            "event": event,
            "phase": row.get("_phase"),
            "schema": row.get("schema"),
            "real_nvapi_sha256": str(row.get("real_nvapi_sha256", "")).lower(),
        }
    if event == "phase_marker":
        return {
            "event": event,
            "phase": row.get("phase"),
            "truncated": row.get("truncated"),
        }
    if event == "interface_query":
        projected = {
            key: row.get(key)
            for key in (
                "event",
                "dispatcher",
                "interface_id",
                "interface",
                "real_resolved",
                "returned_wrapper",
                "real_dispatch_mismatch",
            )
            if key in row
        }
        projected["phase"] = row.get("_phase")
        return projected
    if event == "kernel_launch":
        projected = {
            key: row.get(key)
            for key in (
                "event",
                "launch_token",
                "kernel_index",
                "function_token",
                "module_token",
                "function_name",
                "grid",
                "block",
                "dynamic_shared_bytes",
                "parameter_mode",
                "parameter_captured",
                "parameter_bytes",
                "parameter_file",
            )
        }
        projected["phase"] = row.get("_phase")
        return projected
    if event == "nvapi_call":
        excluded = {"module_blob_sha256", "parameter_sha256"}
        return {
            "phase": phase,
            **{
                key: value
                for key, value in row.items()
                if key not in excluded and key not in {"_phase", "_event_index"}
            },
        }
    return {
        "phase": phase,
        **{
            key: value
            for key, value in row.items()
            if key not in {"_phase", "_event_index"}
        },
    }


def load_trace(run_id: str, root: Path) -> TraceRun:
    require(root.is_dir(), f"{run_id} trace root missing: {root}")
    require(not root.is_symlink(), f"{run_id} trace root may not be a symlink")
    root = root.resolve()
    trace_path = root / "nvapi-trace.jsonl"
    dumps_root = root / "dumps"
    require(dumps_root.is_dir(), f"{run_id} dumps root missing: {dumps_root}")
    require(not dumps_root.is_symlink(), f"{run_id} dumps root may not be a symlink")
    require_regular_file(trace_path, f"{run_id} trace")
    rows: list[dict[str, Any]] = []
    phases: list[str] = []
    current_phase = "runtime"
    phase_counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for line_number, line in enumerate(
        trace_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        require(line.strip() != "", f"{run_id} trace has blank line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CensusError(
                f"{run_id} trace line {line_number} is invalid JSON: {error}"
            ) from error
        require(isinstance(row, dict), f"{run_id} trace line {line_number} is not an object")
        require(isinstance(row.get("event"), str), f"{run_id} trace line {line_number} has no event")
        if row["event"] == "phase_marker":
            require(isinstance(row.get("phase"), str), f"invalid phase marker at line {line_number}")
            current_phase = row["phase"]
            phases.append(current_phase)
        normalized = copy.deepcopy(row)
        normalized["_phase"] = current_phase
        normalized["_event_index"] = line_number
        rows.append(normalized)
        phase_counts[current_phase][row["event"]] += 1

    require(rows, f"{run_id} trace is empty")
    require(rows[0].get("event") == "runtime_ready", f"{run_id} has no runtime_ready prefix")
    require(rows[0].get("schema") == TRACE_SCHEMA, f"{run_id} trace schema drift")
    require(phases == EXPECTED_PHASES, f"{run_id} phase order drift: {phases}")
    require(
        all(not bool(row.get("truncated")) for row in rows if row["event"] == "phase_marker"),
        f"{run_id} contains a truncated phase marker",
    )

    successful_modules = [
        row
        for row in rows
        if row["event"] == "nvapi_call"
        and row.get("call") == CREATE_MODULE
        and row.get("status") == 0
    ]
    successful_functions = [
        row
        for row in rows
        if row["event"] == "nvapi_call"
        and row.get("call") == CREATE_FUNCTION
        and row.get("status") == 0
    ]
    resources = [
        row
        for row in rows
        if row["event"] == "nvapi_call"
        and row.get("call") in RESOURCE_CALLS
        and row.get("status") == 0
    ]
    launches = [row for row in rows if row["event"] == "kernel_launch"]

    module_tokens: set[str] = set()
    modules: list[dict[str, Any]] = []
    for order, row in enumerate(successful_modules, 1):
        token = require_token(row.get("module_token"), "module", "module token")
        require(token not in module_tokens, f"{run_id} duplicate module token: {token}")
        module_tokens.add(token)
        require(row.get("_phase") == "create", f"{run_id} selected module outside create: {token}")
        filename = row.get("module_blob_file")
        require(
            isinstance(filename, str)
            and re.fullmatch(r"cu-module-\d{6}\.bin", filename) is not None,
            f"{run_id} invalid module filename for {token}",
        )
        path = dumps_root / filename
        require_regular_file(path, f"{run_id} module dump {token}")
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        declared_sha = str(row.get("module_blob_sha256", "")).lower()
        require(row.get("module_blob_captured") is True, f"{run_id} module not captured: {token}")
        require(row.get("module_blob_bytes") == actual_bytes, f"{run_id} module byte mismatch: {token}")
        require(declared_sha == actual_sha, f"{run_id} module hash mismatch: {token}")
        modules.append(
            {
                "order": order,
                "event_index": row["_event_index"],
                "phase": row["_phase"],
                "module_token": token,
                "dump_file": filename,
                "bytes": actual_bytes,
                "trace_declared_sha256": declared_sha,
                "rehashed_sha256": actual_sha,
                "rehash_matches": True,
            }
        )

    module_by_token = {item["module_token"]: item for item in modules}
    function_tokens: set[str] = set()
    functions: list[dict[str, Any]] = []
    for order, row in enumerate(successful_functions, 1):
        token = require_token(row.get("function_token"), "function", "function token")
        module_token = require_token(row.get("module_token"), "module", "function module token")
        require(token not in function_tokens, f"{run_id} duplicate function token: {token}")
        require(module_token in module_by_token, f"{run_id} function references unknown module: {token}")
        require(row.get("_phase") == "create", f"{run_id} selected function outside create: {token}")
        name = row.get("function_name")
        require(isinstance(name, str) and name, f"{run_id} empty function name: {token}")
        function_tokens.add(token)
        functions.append(
            {
                "order": order,
                "event_index": row["_event_index"],
                "phase": row["_phase"],
                "function_token": token,
                "module_token": module_token,
                "module_sha256": module_by_token[module_token]["rehashed_sha256"],
                "function_name": name,
            }
        )

    function_by_token = {item["function_token"]: item for item in functions}
    resource_records: list[dict[str, Any]] = []
    resource_phase_orders: collections.Counter[str] = collections.Counter()
    for order, row in enumerate(resources, 1):
        phase = row["_phase"]
        resource_phase_orders[phase] += 1
        token = require_token(row.get("resource_token"), "resource", "resource token")
        require(
            row.get("descriptor_values_recorded") is False,
            f"{run_id} unexpectedly records raw descriptor values: {token}",
        )
        resource_records.append(
            {
                "order": order,
                "phase_order": resource_phase_orders[phase],
                "event_index": row["_event_index"],
                "phase": phase,
                "call": row["call"],
                "resource_token": token,
                "resource_kind": row.get("resource_kind"),
                "descriptor_values_recorded": False,
                "status": 0,
            }
        )

    launch_tokens: set[str] = set()
    launch_records: list[dict[str, Any]] = []
    launch_phase_orders: collections.Counter[str] = collections.Counter()
    for order, row in enumerate(launches, 1):
        phase = row["_phase"]
        require(phase in {"evaluate_0", "evaluate_1"}, f"{run_id} launch outside evaluate: line {row['_event_index']}")
        launch_phase_orders[phase] += 1
        launch_token = require_token(row.get("launch_token"), "launch", "launch token")
        function_token = require_token(row.get("function_token"), "function", "launch function token")
        module_token = require_token(row.get("module_token"), "module", "launch module token")
        require(launch_token not in launch_tokens, f"{run_id} duplicate launch token: {launch_token}")
        require(function_token in function_by_token, f"{run_id} launch references unknown function: {launch_token}")
        function = function_by_token[function_token]
        require(function["module_token"] == module_token, f"{run_id} launch module/function mismatch: {launch_token}")
        require(function["function_name"] == row.get("function_name"), f"{run_id} launch function-name mismatch: {launch_token}")
        require(row.get("kernel_index") == 0, f"{run_id} multi-kernel chain not represented: {launch_token}")
        require(row.get("parameter_mode") == "packed", f"{run_id} non-packed parameters: {launch_token}")
        require(row.get("parameter_captured") is True, f"{run_id} parameters not captured: {launch_token}")
        filename = row.get("parameter_file")
        require(
            isinstance(filename, str)
            and re.fullmatch(r"launch-\d{6}-kernel-\d{4}\.params\.bin", filename),
            f"{run_id} invalid parameter filename: {launch_token}",
        )
        path = dumps_root / filename
        require_regular_file(path, f"{run_id} parameters {launch_token}")
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        declared_sha = str(row.get("parameter_sha256", "")).lower()
        require(row.get("parameter_bytes") == actual_bytes, f"{run_id} parameter byte mismatch: {launch_token}")
        require(declared_sha == actual_sha, f"{run_id} parameter hash mismatch: {launch_token}")
        launch_tokens.add(launch_token)
        launch_records.append(
            {
                "order": order,
                "phase_order": launch_phase_orders[phase],
                "event_index": row["_event_index"],
                "phase": phase,
                "launch_token": launch_token,
                "kernel_index": 0,
                "module_token": module_token,
                "module_sha256": module_by_token[module_token]["rehashed_sha256"],
                "function_token": function_token,
                "function_name": row["function_name"],
                "grid": require_triplet(row.get("grid"), f"{launch_token} grid"),
                "block": require_triplet(row.get("block"), f"{launch_token} block"),
                "dynamic_shared_bytes": row.get("dynamic_shared_bytes"),
                "parameter_mode": "packed",
                "parameter_file": filename,
                "parameter_bytes": actual_bytes,
                "trace_declared_parameter_sha256": declared_sha,
                "rehashed_parameter_sha256": actual_sha,
                "parameter_rehash_matches": True,
            }
        )

    return TraceRun(
        run_id=run_id,
        root=root,
        trace_path=trace_path,
        trace_sha256=sha256_file(trace_path),
        rows=rows,
        phases=phases,
        modules=modules,
        functions=functions,
        resources=resource_records,
        launches=launch_records,
        phase_event_counts={
            phase: dict(sorted(counts.items())) for phase, counts in phase_counts.items()
        },
    )


def module_structure(module: dict[str, Any]) -> dict[str, Any]:
    return {
        key: module[key]
        for key in ("order", "phase", "module_token", "dump_file", "bytes", "rehashed_sha256")
    }


def function_structure(function: dict[str, Any]) -> dict[str, Any]:
    return {
        key: function[key]
        for key in (
            "order",
            "phase",
            "function_token",
            "module_token",
            "module_sha256",
            "function_name",
        )
    }


def resource_structure(resource: dict[str, Any]) -> dict[str, Any]:
    return {
        key: resource[key]
        for key in (
            "order",
            "phase_order",
            "phase",
            "call",
            "resource_token",
            "resource_kind",
            "descriptor_values_recorded",
            "status",
        )
    }


def launch_structure(launch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: launch[key]
        for key in (
            "order",
            "phase_order",
            "phase",
            "launch_token",
            "kernel_index",
            "module_token",
            "module_sha256",
            "function_token",
            "function_name",
            "grid",
            "block",
            "dynamic_shared_bytes",
            "parameter_mode",
            "parameter_file",
            "parameter_bytes",
        )
    }


def byte_differences(first: bytes, second: bytes) -> list[dict[str, Any]]:
    require(len(first) == len(second), "cannot byte-diff unequal lengths")
    return [
        {"offset": offset, "run_a_hex": f"{left:02x}", "run_b_hex": f"{right:02x}"}
        for offset, (left, right) in enumerate(zip(first, second))
        if left != right
    ]


def parse_image_list(text: str, kind: str) -> list[dict[str, Any]]:
    suffix = "cubin" if kind == "ELF" else "ptx"
    pattern = re.compile(
        rf"(?m)^{kind} file\s+(\d+):\s+(\S+)\.(\d+)\.sm_(\d+)\.{suffix}\s*$"
    )
    return [
        {
            "ordinal": int(match.group(1)),
            "container_ordinal": int(match.group(3)),
            "architecture": f"sm_{match.group(4)}",
            "format": suffix,
        }
        for match in pattern.finditer(text)
    ]


def parse_fatbin_metadata(text: str) -> list[dict[str, Any]]:
    starts = list(re.finditer(r"(?m)^Fatbin (ptx|elf) code:\s*$", text))
    records: list[dict[str, Any]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[match.end() : end]
        fields: dict[str, Any] = {"format": match.group(1)}
        for key, value in re.findall(r"(?m)^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", body):
            normalized_key = {
                "arch": "architecture",
                "host": "host",
                "compile_size": "compile_size",
                "ptxasOptions": "ptxas_options",
            }.get(key, key)
            if key == "code version":
                normalized_key = "code_version"
            fields[normalized_key] = value
        code_match = re.search(r"(?m)^code version\s*=\s*\[(\d+),(\d+)\]", body)
        if code_match:
            fields["code_version"] = f"{code_match.group(1)}.{code_match.group(2)}"
        fields["compressed"] = bool(re.search(r"(?m)^compressed\s*$", body))
        if "architecture" in fields:
            fields["architecture"] = f"sm_{str(fields['architecture']).removeprefix('sm_')}"
        records.append(fields)
    return records


SECTION_PATTERN = re.compile(
    r"^\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+"
    r"([0-9a-f]+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+(\S+)\s*$",
    re.IGNORECASE,
)


def parse_elf_sections(text: str) -> list[dict[str, Any]]:
    match = re.search(r"(?m)^Sections:\s*$", text)
    if not match:
        return []
    records: list[dict[str, Any]] = []
    for line in text[match.end() :].splitlines()[1:]:
        if not line.strip():
            if records:
                break
            continue
        row = SECTION_PATTERN.match(line)
        if not row:
            if records:
                break
            continue
        values = row.groups()
        records.append(
            {
                "index": int(values[0], 16),
                "offset": int(values[1], 16),
                "size": int(values[2], 16),
                "entry_size": int(values[3], 16),
                "align": int(values[4], 16),
                "type": values[5],
                "flags": values[6].lower(),
                "link": int(values[7], 16),
                "info": values[8].lower(),
                "name": values[9],
            }
        )
    return records


def parse_elf_header(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"(?m)^64bit elf: type=(\d+), abi=(\d+), sm=(\d+), toolkit=(\d+), flags = (0x[0-9a-f]+)\s*$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "bits": 64,
        "type": int(match.group(1)),
        "abi": int(match.group(2)),
        "architecture": f"sm_{match.group(3)}",
        "toolkit": int(match.group(4)),
        "flags": match.group(5).lower(),
    }


def parse_elf_symbols(text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?m)^(STT_\S+)\s+(STB_\S+)\s+(STO_\S+)\s+(\S+)\s*$"
    )
    return [
        {
            "type": match.group(1),
            "binding": match.group(2),
            "other": match.group(3),
            "name": match.group(4),
        }
        for match in pattern.finditer(text)
    ]


def parse_resource_usage(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(?m)^ Function (\S+):\s*$\n^\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) "
        r"LOCAL:(\d+) CONSTANT\[0\]:(\d+) TEXTURE:(\d+) SURFACE:(\d+) SAMPLER:(\d+)\s*$"
    )
    return [
        {
            "function_name": match.group(1),
            "registers": int(match.group(2)),
            "stack_bytes": int(match.group(3)),
            "shared_bytes": int(match.group(4)),
            "local_bytes": int(match.group(5)),
            "constant0_bytes": int(match.group(6)),
            "textures": int(match.group(7)),
            "surfaces": int(match.group(8)),
            "samplers": int(match.group(9)),
        }
        for match in pattern.finditer(text)
    ]


def verified_inspection_outputs(
    inspection_root: Path, record: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    outputs = record.get("outputs")
    require(isinstance(outputs, list), "rig inspection module outputs must be an array")
    metadata: dict[str, dict[str, Any]] = {}
    text_by_kind: dict[str, str] = {}
    for output in outputs:
        require(isinstance(output, dict), "rig inspection output must be an object")
        kind = output.get("kind")
        require(kind in {"list-elf", "list-ptx", "elf", "symbols", "resource-usage"}, f"unknown rig output kind: {kind}")
        require(kind not in metadata, f"duplicate rig output kind: {kind}")
        relative = output.get("relative_path")
        require(isinstance(relative, str) and relative, f"invalid rig output path: {kind}")
        require("\\" not in relative, f"rig output path must use forward slashes: {relative}")
        pure = Path(relative)
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe rig output path: {relative}")
        lexical_path = inspection_root / pure
        require_regular_file(lexical_path, f"rig inspection {kind}")
        parent = lexical_path.parent
        while parent != inspection_root:
            require(not parent.is_symlink(), f"rig inspection path uses symlink: {relative}")
            parent = parent.parent
        path = lexical_path.resolve()
        try:
            path.relative_to(inspection_root.resolve())
        except ValueError as error:
            raise CensusError(f"rig output escapes inspection root: {relative}") from error
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        require(output.get("bytes") == actual_bytes, f"rig output byte mismatch: {relative}")
        require(str(output.get("sha256", "")).lower() == actual_sha, f"rig output hash mismatch: {relative}")
        metadata[kind] = {
            "relative_path": relative,
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "rehash_matches": True,
        }
        text_by_kind[kind] = read_text(path)
    require(
        set(metadata) == {"list-elf", "list-ptx", "elf", "symbols", "resource-usage"},
        "rig inspection output set is incomplete",
    )
    return metadata, text_by_kind


def load_rig_inspection(
    inspection_root: Path, selected_modules: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(inspection_root.is_dir(), f"rig inspection root missing: {inspection_root}")
    require(not inspection_root.is_symlink(), "rig inspection root may not be a symlink")
    inspection_root = inspection_root.resolve()
    receipt_path = inspection_root / "receipt.json"
    require_regular_file(receipt_path, "rig inspection receipt")
    receipt = read_json(receipt_path)
    require(isinstance(receipt, dict), "rig inspection receipt must be an object")
    require(receipt.get("schema") == RIG_SCHEMA, "rig inspection receipt schema drift")
    records = receipt.get("modules")
    require(isinstance(records, list), "rig inspection modules must be an array")
    require(receipt.get("module_count") == len(records), "rig inspection module count mismatch")
    selected_by_name = {module["dump_file"]: module for module in selected_modules}
    require(len(records) == len(selected_by_name), "rig inspection does not cover every selected module")
    by_hash: dict[str, dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "rig inspection module must be an object")
        source_name = record.get("source_name")
        require(source_name in selected_by_name, f"rig inspection includes unselected module: {source_name}")
        selected = selected_by_name[source_name]
        source_sha = str(record.get("source_sha256", "")).lower()
        require(source_sha == selected["rehashed_sha256"], f"rig inspection source hash drift: {source_name}")
        require(record.get("source_bytes") == selected["bytes"], f"rig inspection source size drift: {source_name}")
        require(source_sha not in by_hash, f"duplicate rig inspection code hash: {source_sha}")
        outputs, texts = verified_inspection_outputs(inspection_root, record)
        ptx_images = parse_image_list(texts["list-ptx"], "PTX")
        elf_images = parse_image_list(texts["list-elf"], "ELF")
        require(ptx_images, f"selected module has no PTX image: {source_name}")
        fatbin = parse_fatbin_metadata(texts["elf"])
        require(
            collections.Counter((item["format"], item.get("architecture")) for item in fatbin)
            == collections.Counter(
                [("ptx", item["architecture"]) for item in ptx_images]
                + [("elf", item["architecture"]) for item in elf_images]
            ),
            f"fatbin/list architecture mismatch: {source_name}",
        )
        symbols = parse_elf_symbols(texts["symbols"])
        sections = parse_elf_sections(texts["elf"])
        elf_header = parse_elf_header(texts["elf"])
        require(bool(elf_images) == bool(elf_header), f"ELF header presence mismatch: {source_name}")
        require(bool(elf_images) == bool(sections), f"ELF sections presence mismatch: {source_name}")
        attributes = sorted(set(re.findall(r"Attribute:\s+([A-Z0-9_]+)", texts["elf"])))
        resource_usage = parse_resource_usage(texts["resource-usage"])
        by_hash[source_sha] = {
            "inspection_outputs": outputs,
            "ptx_images": ptx_images,
            "elf_images": elf_images,
            "fatbin_metadata": fatbin,
            "elf_header": elf_header,
            "elf_sections": sections,
            "elf_symbols": symbols,
            "cuda_metadata_attributes": attributes,
            "resource_usage": resource_usage,
        }
    require(set(by_hash) == {module["rehashed_sha256"] for module in selected_modules}, "rig inspection hash set mismatch")
    receipt_summary = {
        "schema": receipt["schema"],
        "job_id": receipt.get("job_id"),
        "receipt_relative_path": relative_display(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
        "receipt_sha256": sha256_file(receipt_path),
        "started_utc": receipt.get("started_utc"),
        "ended_utc": receipt.get("ended_utc"),
        "exit_code": receipt.get("exit_code"),
        "cuobjdump_version": receipt.get("cuobjdump_version"),
        "gpu": receipt.get("gpu"),
        "cuda_path": receipt.get("cuda_path"),
        "command_kinds": receipt.get("command_kinds"),
        "module_count": receipt.get("module_count"),
        "all_output_hashes_reverified": True,
    }
    require(receipt_summary["exit_code"] == 0, "rig inspection did not complete successfully")
    return receipt_summary, by_hash


def load_initial_merge_abi(
    abi_root: Path, expected_module: dict[str, Any]
) -> dict[str, Any]:
    """Verify and reduce the job-scoped PTX ABI for the first closed FG group."""

    require(abi_root.is_dir(), f"ABI inspection root missing: {abi_root}")
    require(not abi_root.is_symlink(), "ABI inspection root may not be a symlink")
    abi_root = abi_root.resolve()
    receipt_path = abi_root / "receipt.json"
    require_regular_file(receipt_path, "ABI inspection receipt")
    receipt = read_json(receipt_path)
    require(isinstance(receipt, dict), "ABI inspection receipt must be an object")
    require(receipt.get("schema") == ABI_SCHEMA, "ABI inspection receipt schema drift")
    require(receipt.get("exit_code") == 0, "ABI inspection did not complete successfully")
    require(
        str(receipt.get("source_sha256", "")).lower()
        == expected_module["rehashed_sha256"],
        "ABI inspection source module hash mismatch",
    )
    require(
        receipt.get("source_bytes") == expected_module["bytes"],
        "ABI inspection source module size mismatch",
    )
    relative = receipt.get("output_relative_path")
    require(
        isinstance(relative, str)
        and relative
        and "\\" not in relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        "unsafe ABI inspection output path",
    )
    output_path = abi_root / relative
    require_regular_file(output_path, "ABI inspection PTX output")
    require(
        output_path.parent.resolve() == abi_root,
        "ABI inspection PTX output must be a direct child of its root",
    )
    output_bytes = output_path.stat().st_size
    output_sha = sha256_file(output_path)
    require(receipt.get("output_bytes") == output_bytes, "ABI inspection output size mismatch")
    require(
        str(receipt.get("output_sha256", "")).lower() == output_sha,
        "ABI inspection output hash mismatch",
    )
    text = read_text(output_path).replace("\r", "")
    entry = re.search(
        r"\.visible\s+\.entry\s+k_initial_merge\s*\((.*?)\)\s*\{(.*)\}\s*$",
        text,
        re.DOTALL,
    )
    require(entry is not None, "k_initial_merge PTX entry is missing")
    parameter_rows = re.findall(
        r"\.param\s+\.(u64|u32|f32)\s+(k_initial_merge_param_(\d+))",
        entry.group(1),
    )
    require(
        [(kind, name, int(index)) for kind, name, index in parameter_rows]
        == [
            ("u64", "k_initial_merge_param_0", 0),
            ("u64", "k_initial_merge_param_1", 1),
            ("u64", "k_initial_merge_param_2", 2),
            ("u64", "k_initial_merge_param_3", 3),
            ("u32", "k_initial_merge_param_4", 4),
            ("u32", "k_initial_merge_param_5", 5),
            ("f32", "k_initial_merge_param_6", 6),
        ],
        "k_initial_merge PTX parameter ABI drift",
    )
    body = entry.group(2)
    global_loads = re.findall(r"\bld\.global\.(u\d+)\b", body)
    global_stores = re.findall(r"\bst\.global\.(u\d+)\b", body)
    require(global_loads == ["u64", "u64", "u16"], "k_initial_merge load-width drift")
    require(global_stores == ["u32"] * 5, "k_initial_merge store-width drift")
    require(".shared" not in body, "k_initial_merge unexpectedly uses shared memory")
    require(
        not re.search(r"\b(?:tex|tld4|suld|sust|sured|suq)\.", body),
        "k_initial_merge unexpectedly uses texture/surface instructions",
    )
    required_fragments = [
        "mul.wide.s32 %rd6, %r12, 8;",
        "mul.wide.s32 %rd13, %r12, 2;",
        "mul.wide.s32 %rd16, %r12, 20;",
        "{ cvt.rn.f16.f32 %rs1, %f1;}",
    ]
    require(
        all(fragment in body for fragment in required_fragments),
        "k_initial_merge indexing/conversion semantics drift",
    )
    roles = [
        "output_struct20_buffer",
        "input_u64_a_buffer",
        "input_u64_b_buffer",
        "input_u16_buffer",
        "height",
        "width",
        "interpolation_factor",
    ]
    offsets = [0, 8, 16, 24, 32, 36, 40]
    sizes = {"u64": 8, "u32": 4, "f32": 4}
    typed_parameters = [
        {
            "index": int(index),
            "name": name,
            "ptx_type": kind,
            "offset": offsets[int(index)],
            "bytes": sizes[kind],
            "role": roles[int(index)],
            **({"raw_pointer_value_recorded": False} if kind == "u64" else {}),
        }
        for kind, name, index in parameter_rows
    ]
    return {
        "receipt_relative_path": relative_display(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "started_utc": receipt.get("started_utc"),
        "ended_utc": receipt.get("ended_utc"),
        "command": receipt.get("command"),
        "source_sha256": expected_module["rehashed_sha256"],
        "source_bytes": expected_module["bytes"],
        "ptx_output_relative_path": relative_display(output_path),
        "ptx_output_bytes": output_bytes,
        "ptx_output_sha256": output_sha,
        "ptx_payload_tracked": False,
        "entry": "k_initial_merge",
        "typed_parameters": typed_parameters,
        "parameter_struct_bytes": 44,
        "global_load_width_bits": [64, 64, 16],
        "global_store_width_bits": [32, 32, 32, 32, 32],
        "input_element_strides_bytes": [8, 8, 2],
        "output_element_stride_bytes": 20,
        "interpolation_conversion": "cvt.rn.f16.f32",
        "uses_texture_or_surface_instructions": False,
        "uses_shared_memory": False,
        "has_embedded_weight_dependency": False,
        "normalized_ptx_abi_sha256": canonical_sha256(
            {
                "entry": "k_initial_merge",
                "parameters": [(kind, name, int(index)) for kind, name, index in parameter_rows],
                "global_loads": global_loads,
                "global_stores": global_stores,
                "required_fragments": required_fragments,
            }
        ),
    }


def collection_root(records: Any) -> str:
    return canonical_sha256(records)


def architecture_summary(code_hashes: list[dict[str, Any]]) -> dict[str, Any]:
    ptx_architectures: collections.Counter[str] = collections.Counter()
    elf_architectures: collections.Counter[str] = collections.Counter()
    ptx_versions: collections.Counter[str] = collections.Counter()
    elf_versions: collections.Counter[str] = collections.Counter()
    section_types: collections.Counter[str] = collections.Counter()
    section_names: collections.Counter[str] = collections.Counter()
    symbol_names: collections.Counter[str] = collections.Counter()
    cuda_attributes: collections.Counter[str] = collections.Counter()
    modules_with_elf = 0
    modules_with_ptx = 0
    for module in code_hashes:
        inspection = module["inspection"]
        if inspection["ptx_images"]:
            modules_with_ptx += 1
        if inspection["elf_images"]:
            modules_with_elf += 1
        ptx_architectures.update(item["architecture"] for item in inspection["ptx_images"])
        elf_architectures.update(item["architecture"] for item in inspection["elf_images"])
        for record in inspection["fatbin_metadata"]:
            version = record.get("code_version", "unknown")
            (ptx_versions if record["format"] == "ptx" else elf_versions)[version] += 1
        section_types.update(section["type"] for section in inspection["elf_sections"])
        section_names.update(section["name"] for section in inspection["elf_sections"])
        symbol_names.update(symbol["name"] for symbol in inspection["elf_symbols"])
        cuda_attributes.update(inspection["cuda_metadata_attributes"])
    return {
        "modules_with_ptx": modules_with_ptx,
        "modules_with_elf": modules_with_elf,
        "ptx_image_count": sum(ptx_architectures.values()),
        "elf_image_count": sum(elf_architectures.values()),
        "ptx_architecture_image_counts": dict(sorted(ptx_architectures.items())),
        "elf_architecture_image_counts": dict(sorted(elf_architectures.items())),
        "ptx_code_version_counts": dict(sorted(ptx_versions.items())),
        "elf_code_version_counts": dict(sorted(elf_versions.items())),
        "elf_section_type_counts": dict(sorted(section_types.items())),
        "elf_section_name_counts": dict(sorted(section_names.items())),
        "elf_entry_symbol_counts": dict(sorted(symbol_names.items())),
        "cuda_metadata_attribute_module_counts": dict(sorted(cuda_attributes.items())),
        "static_runtime_scan_performed": False,
        "unselected_static_variants_included": False,
    }


def build_census(
    run_a_root: Path,
    run_b_root: Path,
    inspection_root: Path,
    abi_inspection_root: Path | None = None,
    *,
    enforce_production_shape: bool = True,
) -> dict[str, Any]:
    run_a = load_trace("run-a", run_a_root)
    run_b = load_trace("run-b", run_b_root)

    module_a = [module_structure(item) for item in run_a.modules]
    module_b = [module_structure(item) for item in run_b.modules]
    function_a = [function_structure(item) for item in run_a.functions]
    function_b = [function_structure(item) for item in run_b.functions]
    resource_a = [resource_structure(item) for item in run_a.resources]
    resource_b = [resource_structure(item) for item in run_b.resources]
    launch_a = [launch_structure(item) for item in run_a.launches]
    launch_b = [launch_structure(item) for item in run_b.launches]

    require(run_a.phases == run_b.phases, "fresh-process phase sequences differ")
    require(module_a == module_b, "fresh-process selected module structures differ")
    require(function_a == function_b, "fresh-process function structures differ")
    require(resource_a == resource_b, "fresh-process resource request structures differ")
    require(launch_a == launch_b, "fresh-process launch structures differ")
    projected_a = [event_projection(row) for row in run_a.rows]
    projected_b = [event_projection(row) for row in run_b.rows]
    require(projected_a == projected_b, "fresh-process address-free trace structures differ")

    receipt, inspection_by_hash = load_rig_inspection(inspection_root, run_a.modules)

    functions_by_module = {item["module_token"]: item for item in run_a.functions}
    selected_modules: list[dict[str, Any]] = []
    code_by_sha256: dict[str, dict[str, Any]] = {}
    for first, second in zip(run_a.modules, run_b.modules):
        require(first["rehashed_sha256"] == second["rehashed_sha256"], f"module bytes differ: {first['module_token']}")
        function = functions_by_module[first["module_token"]]
        selected_modules.append(
            {
                **module_structure(first),
                "trace_declared_sha256": first["trace_declared_sha256"],
                "rehash_matches": first["rehash_matches"],
                "function_token": function["function_token"],
                "function_name": function["function_name"],
                "observed_runs": [run_a.run_id, run_b.run_id],
                "module_bytes_equal": True,
            }
        )
        code = code_by_sha256.setdefault(
            first["rehashed_sha256"],
            {
                "code_order": first["order"],
                "sha256": first["rehashed_sha256"],
                "bytes": first["bytes"],
                "module_tokens": [],
                "function_tokens": [],
                "function_names": [],
                "inspection": inspection_by_hash[first["rehashed_sha256"]],
            },
        )
        require(code["bytes"] == first["bytes"], "same code hash reported with different byte size")
        code["module_tokens"].append(first["module_token"])
        code["function_tokens"].append(function["function_token"])
        if function["function_name"] not in code["function_names"]:
            code["function_names"].append(function["function_name"])

    code_hashes = sorted(code_by_sha256.values(), key=lambda item: item["code_order"])
    for code in code_hashes:
        code["module_tokens"].sort()
        code["function_tokens"].sort()
        code["function_names"].sort()

    selected_functions = function_a
    ordered_launches: list[dict[str, Any]] = []
    parameter_difference_records: list[dict[str, Any]] = []
    equal_parameter_count = 0
    for first, second in zip(run_a.launches, run_b.launches):
        first_path = run_a.root / "dumps" / first["parameter_file"]
        second_path = run_b.root / "dumps" / second["parameter_file"]
        first_bytes = first_path.read_bytes()
        second_bytes = second_path.read_bytes()
        differences = byte_differences(first_bytes, second_bytes)
        equal = not differences
        if equal:
            equal_parameter_count += 1
        else:
            parameter_difference_records.append(
                {
                    "launch_token": first["launch_token"],
                    "phase": first["phase"],
                    "parameter_bytes": len(first_bytes),
                    "run_a_sha256": first["rehashed_parameter_sha256"],
                    "run_b_sha256": second["rehashed_parameter_sha256"],
                    "differing_byte_count": len(differences),
                    "differing_bytes": differences,
                }
            )
        ordered_launches.append(
            {
                **launch_structure(first),
                "parameters": {
                    "run_a": {
                        "trace_declared_sha256": first["trace_declared_parameter_sha256"],
                        "rehashed_sha256": first["rehashed_parameter_sha256"],
                        "rehash_matches": True,
                    },
                    "run_b": {
                        "trace_declared_sha256": second["trace_declared_parameter_sha256"],
                        "rehashed_sha256": second["rehashed_parameter_sha256"],
                        "rehash_matches": True,
                    },
                    "bytes_equal": equal,
                    "differing_byte_count": len(differences),
                    "differing_bytes": differences,
                },
            }
        )

    kernel_groups: dict[str, dict[str, Any]] = {}
    for function in selected_functions:
        group = kernel_groups.setdefault(
            function["function_name"],
            {"function_name": function["function_name"], "module_sha256s": [], "function_tokens": []},
        )
        group["module_sha256s"].append(function["module_sha256"])
        group["function_tokens"].append(function["function_token"])
    launch_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for launch in ordered_launches:
        launch_counts[launch["function_name"]][launch["phase"]] += 1
    kernel_names: list[dict[str, Any]] = []
    for name, group in sorted(kernel_groups.items()):
        group["module_sha256s"] = sorted(group["module_sha256s"])
        group["function_tokens"] = sorted(group["function_tokens"])
        group["module_count"] = len(group["module_sha256s"])
        group["launch_counts_by_phase"] = dict(sorted(launch_counts[name].items()))
        group["launch_count"] = sum(launch_counts[name].values())
        kernel_names.append(group)

    if enforce_production_shape:
        require(len(selected_modules) == 64, f"expected 64 selected modules, found {len(selected_modules)}")
        require(len(code_hashes) == 64, f"expected 64 unique code hashes, found {len(code_hashes)}")
        require(len(selected_functions) == 64, f"expected 64 functions, found {len(selected_functions)}")
        require(len(kernel_names) == 19, f"expected 19 kernel names, found {len(kernel_names)}")
        require(len(ordered_launches) == 85, f"expected 85 launches, found {len(ordered_launches)}")
        require(sum(launch["phase"] == "evaluate_0" for launch in ordered_launches) == 9, "evaluate_0 launch count drift")
        require(sum(launch["phase"] == "evaluate_1" for launch in ordered_launches) == 76, "evaluate_1 launch count drift")
        require(equal_parameter_count == 83, f"expected 83 byte-identical parameter dumps, found {equal_parameter_count}")
        require(
            parameter_difference_records
            == [
                {
                    "launch_token": "launch-000006",
                    "phase": "evaluate_0",
                    "parameter_bytes": 224,
                    "run_a_sha256": "7327ed532cd5fc06b4016442fa778a78bdfbcff0d9773b8c94deaf76014c2498",
                    "run_b_sha256": "2a0f71e5d776feea8befeb4fd166d4d1d590102504970f691ac39e9ed01b50a3",
                    "differing_byte_count": 1,
                    "differing_bytes": [{"offset": 157, "run_a_hex": "02", "run_b_hex": "01"}],
                },
                {
                    "launch_token": "launch-000016",
                    "phase": "evaluate_1",
                    "parameter_bytes": 224,
                    "run_a_sha256": "f6c81a583eeff2f81206ec164357f49e53c7b4209c36ab602c641728c7375be8",
                    "run_b_sha256": "9d486489b98673d0d1a34fbf68105b3f38e4cd544515a92e451069894211570a",
                    "differing_byte_count": 1,
                    "differing_bytes": [{"offset": 157, "run_a_hex": "02", "run_b_hex": "01"}],
                },
            ],
            "parameter byte-difference frontier drift",
        )

    evaluate_1 = [launch for launch in ordered_launches if launch["phase"] == "evaluate_1"]
    require(evaluate_1, "no evaluate_1 launches")
    first_eval1 = evaluate_1[0]
    first_eval1_event_index = next(
        launch["event_index"]
        for launch in run_a.launches
        if launch["launch_token"] == first_eval1["launch_token"]
    )
    resource_before_first_eval1 = [
        resource_structure(resource)
        for resource in run_a.resources
        if resource["phase"] == "evaluate_1"
        and resource["event_index"] < first_eval1_event_index
    ]
    prefix_code = next(
        code for code in code_hashes if code["sha256"] == first_eval1["module_sha256"]
    )
    prior_same_function: list[dict[str, Any]] = []
    for launch in ordered_launches:
        if launch["order"] >= first_eval1["order"]:
            break
        if (
            launch["module_sha256"] != first_eval1["module_sha256"]
            or launch["function_name"] != first_eval1["function_name"]
        ):
            continue
        prior_parameter = (
            run_a.root / "dumps" / launch["parameter_file"]
        ).read_bytes()
        prefix_parameter = (
            run_a.root / "dumps" / first_eval1["parameter_file"]
        ).read_bytes()
        cross_phase_differences = byte_differences(prior_parameter, prefix_parameter)
        prior_same_function.append(
            {
                "launch_token": launch["launch_token"],
                "phase": launch["phase"],
                "parameter_sha256": launch["parameters"]["run_a"]["rehashed_sha256"],
                "parameter_bytes": launch["parameter_bytes"],
                "parameter_difference_to_prefix": [
                    {
                        "offset": difference["offset"],
                        "prior_hex": difference["run_a_hex"],
                        "prefix_hex": difference["run_b_hex"],
                    }
                    for difference in cross_phase_differences
                ],
                "relation": "same selected module and function; data dependency unproven",
            }
        )
    # The trace-local prefix is closed over its recorded module/function/packed
    # parameter/geometry records.  Descriptor values and memory access roles are
    # absent, so semantic data closure is explicitly *not* claimed.
    earliest_prefix = {
        "scope": "evaluate_1",
        "closure_kind": "trace_record_dependency_closure",
        "launch_count": 1,
        "first_launch_token": first_eval1["launch_token"],
        "last_launch_token": first_eval1["launch_token"],
        "launch": {
            key: first_eval1[key]
            for key in (
                "launch_token",
                "module_token",
                "module_sha256",
                "function_token",
                "function_name",
                "grid",
                "block",
                "dynamic_shared_bytes",
                "parameter_mode",
                "parameter_file",
                "parameter_bytes",
                "parameters",
            )
        },
        "module_dependencies": [
            {
                "module_token": first_eval1["module_token"],
                "sha256": first_eval1["module_sha256"],
                "bytes": next(
                    module["bytes"]
                    for module in selected_modules
                    if module["module_token"] == first_eval1["module_token"]
                ),
                "ptx_images": prefix_code["inspection"]["ptx_images"],
                "elf_images": prefix_code["inspection"]["elf_images"],
            }
        ],
        "function_dependencies": [
            {
                "function_token": first_eval1["function_token"],
                "function_name": first_eval1["function_name"],
            }
        ],
        "resource_descriptor_dependencies_before_launch": resource_before_first_eval1,
        "predecessor_launch_dependencies_within_evaluate_1": [],
        "prior_same_module_function_launches": prior_same_function,
        "trace_record_dependency_closed": True,
        "semantic_resource_dependency_closed": False,
        "semantic_closure_blocker": (
            "NvAPI descriptor values, packed-parameter field types, memory access roles, "
            "and live output bindings were not captured; the 16-byte parameter blob "
            "therefore cannot yet be rebound for an independent Metal execution."
        ),
    }

    smallest_group: dict[str, Any] = {
        "status": "no_semantically_closed_group_identified",
        "earliest_trace_record_prefix": earliest_prefix,
    }
    if enforce_production_shape:
        target = next(
            (launch for launch in ordered_launches if launch["launch_token"] == "launch-000056"),
            None,
        )
        require(target is not None, "k_initial_merge launch-000056 is missing")
        require(target["phase"] == "evaluate_1", "k_initial_merge is outside evaluate_1")
        require(target["phase_order"] == 45, "k_initial_merge evaluate_1 order drift")
        require(target["function_name"] == "k_initial_merge", "launch-000056 function drift")
        require(target["module_token"] == "module-000053", "launch-000056 module drift")
        target_module = next(
            module
            for module in run_a.modules
            if module["module_token"] == target["module_token"]
        )
        target_code = next(
            code for code in code_hashes if code["sha256"] == target["module_sha256"]
        )
        abi = load_initial_merge_abi(
            abi_inspection_root or DEFAULT_ABI_INSPECTION, target_module
        )
        parameter_a = (
            run_a.root / "dumps" / target["parameter_file"]
        ).read_bytes()
        parameter_b = (
            run_b.root / "dumps" / target["parameter_file"]
        ).read_bytes()
        require(len(parameter_a) == abi["parameter_struct_bytes"], "k_initial_merge parameter size drift")
        require(parameter_a == parameter_b, "k_initial_merge parameters differ between processes")
        unpacked_a = struct.unpack("<QQQQIIf", parameter_a)
        unpacked_b = struct.unpack("<QQQQIIf", parameter_b)
        require(unpacked_a[4:] == unpacked_b[4:], "k_initial_merge scalar parameters differ")
        height, width, interpolation = unpacked_a[4:]
        require((height, width, interpolation) == (368, 640, 0.5), "k_initial_merge scalar frontier drift")
        pixels = height * width
        external_buffers = [
            {
                "parameter_index": 1,
                "role": "input_u64_a_buffer",
                "element_bytes": 8,
                "minimum_extent_bytes": pixels * 8,
                "capture_requirement": "immutable live pre-launch snapshot",
            },
            {
                "parameter_index": 2,
                "role": "input_u64_b_buffer",
                "element_bytes": 8,
                "minimum_extent_bytes": pixels * 8,
                "capture_requirement": "immutable live pre-launch snapshot",
            },
            {
                "parameter_index": 3,
                "role": "input_u16_buffer",
                "element_bytes": 2,
                "minimum_extent_bytes": pixels * 2,
                "capture_requirement": "immutable live pre-launch snapshot",
            },
        ]
        output = {
            "parameter_index": 0,
            "role": "output_struct20_buffer",
            "element_bytes": 20,
            "minimum_extent_bytes": pixels * 20,
            "layout": [
                {"offset": 0, "type": "u64", "source": "input_a"},
                {"offset": 8, "type": "u64", "source": "input_b"},
                {"offset": 16, "type": "u16", "source": "input_c"},
                {"offset": 18, "type": "f16", "source": "round_to_nearest_even(interpolation_factor)"},
            ],
            "write_coverage_requirement": "every byte written under two independent output poisons",
        }
        smallest_group = {
            "status": "dependency_surface_identified_replay_evidence_pending",
            "scope": "evaluate_1",
            "selection_rule": (
                "first trace-ordered launch with a fully typed PTX ABI, no texture/surface/shared/"
                "embedded-weight dependency, and a standalone boundary over immutable pre-launch inputs"
            ),
            "launch_order": target["order"],
            "evaluate_1_order": target["phase_order"],
            "launch": {
                key: target[key]
                for key in (
                    "launch_token",
                    "module_token",
                    "module_sha256",
                    "function_token",
                    "function_name",
                    "grid",
                    "block",
                    "dynamic_shared_bytes",
                    "parameter_mode",
                    "parameter_file",
                    "parameter_bytes",
                    "parameters",
                )
            },
            "module_dependency": {
                "module_token": target["module_token"],
                "sha256": target["module_sha256"],
                "bytes": target_module["bytes"],
                "ptx_images": target_code["inspection"]["ptx_images"],
                "elf_images": target_code["inspection"]["elf_images"],
            },
            "function_dependency": {
                "function_token": target["function_token"],
                "function_name": target["function_name"],
            },
            "ptx_abi": abi,
            "address_free_parameter_values": {
                "height": height,
                "width": width,
                "interpolation_factor": interpolation,
                "pixel_count": pixels,
                "raw_pointer_values_recorded": False,
            },
            "external_buffer_dependencies": external_buffers,
            "output_dependency": output,
            "prior_launches_required_with_fresh_prelaunch_snapshots": [],
            "descriptor_object_dependencies": [],
            "embedded_weight_dependencies": [],
            "earlier_evaluate_1_launches_examined": 44,
            "earlier_launch_rejection_summary": [
                {
                    "launch_tokens": ["launch-000012", "launch-000019"],
                    "count": 8,
                    "reason": "opaque main-kernel control/preprocess ABIs and descriptor-bound resources",
                },
                {
                    "launch_tokens": ["launch-000020", "launch-000051"],
                    "count": 32,
                    "reason": "interdependent convolution/pooling/upscale/element-wise pyramid chain",
                },
                {
                    "launch_tokens": ["launch-000052", "launch-000055"],
                    "count": 4,
                    "reason": "descriptor-bound main-kernel preparation path",
                },
            ],
            "kernel_dependency_closed_with_fresh_input_snapshots": True,
            "fresh_input_snapshots_captured": False,
            "standalone_rtx_replay_proven": False,
            "metal_replay_proven": False,
            "metal_lowering_guard": {
                "required_store_width_bits": [32, 32, 32, 32, 32],
                "required_store_offsets_bytes": [0, 4, 8, 12, 16],
                "forbidden_lowering": "overlapping or unaligned widened u64 stores",
            },
            "next_acceptance_gate": [
                "immutable live pre-launch snapshots of all three inputs",
                "normalized allocation-plus-offset bindings",
                "two output poisons with complete write coverage",
                "unchanged input hashes",
                "two fresh RTX process repeats",
                "two fresh Mac process repeats",
                "byte-exact RTX-versus-Mac output equality",
            ],
            "earliest_trace_record_prefix": earliest_prefix,
        }

    summary = architecture_summary(code_hashes)
    result: dict[str, Any] = {
        "schema": CENSUS_SCHEMA,
        "source": {
            "selection_rule": "successful live NvAPI_D3D12_CreateCuModule calls in create phase",
            "tooling": {
                "builder_relative_path": relative_display(Path(__file__)),
                "builder_sha256": sha256_file(Path(__file__)),
                "schema_relative_path": relative_display(SCHEMA_PATH),
                "schema_sha256": sha256_file(SCHEMA_PATH),
            },
            "run_a": {
                "run_id": run_a.run_id,
                "trace_relative_path": relative_display(run_a.trace_path),
                "trace_sha256": run_a.trace_sha256,
                "phase_event_counts": run_a.phase_event_counts,
            },
            "run_b": {
                "run_id": run_b.run_id,
                "trace_relative_path": relative_display(run_b.trace_path),
                "trace_sha256": run_b.trace_sha256,
                "phase_event_counts": run_b.phase_event_counts,
            },
            "rig_inspection": receipt,
            "static_runtime_scan_performed": False,
            "unselected_static_variants_included": False,
        },
        "phase_order": run_a.phases,
        "counts": {
            "selected_module_instances": len(selected_modules),
            "unique_selected_code_hashes": len({item["sha256"] for item in code_hashes}),
            "selected_function_instances": len(selected_functions),
            "unique_kernel_names": len(kernel_names),
            "resource_descriptor_requests": len(resource_a),
            "launches": len(ordered_launches),
            "evaluate_0_launches": sum(launch["phase"] == "evaluate_0" for launch in ordered_launches),
            "evaluate_1_launches": sum(launch["phase"] == "evaluate_1" for launch in ordered_launches),
            "parameter_dumps_rehashed": len(ordered_launches) * 2,
            "module_dumps_rehashed": len(selected_modules) * 2,
            "byte_identical_parameter_pairs": equal_parameter_count,
            "byte_different_parameter_pairs": len(parameter_difference_records),
        },
        "two_process_equality": {
            "phase_sequence_equal": True,
            "module_structure_equal": True,
            "module_bytes_equal": True,
            "function_structure_equal": True,
            "resource_request_structure_equal": True,
            "launch_structure_equal": True,
            "address_free_trace_structure_equal": True,
            "raw_parameter_bytes_equal": not parameter_difference_records,
            "address_free_structure_sha256": collection_root(projected_a),
            "parameter_difference_records": parameter_difference_records,
        },
        "ordered_modules": selected_modules,
        "ordered_functions": selected_functions,
        "ordered_resource_requests": resource_a,
        "ordered_launches": ordered_launches,
        "unique_code_hashes": code_hashes,
        "kernel_names": kernel_names,
        "architecture_census": summary,
        "smallest_dependency_closed_eval1_group": smallest_group,
        "capture_limits": {
            "descriptor_values_recorded": False,
            "packed_parameter_field_types_recorded": False,
            "memory_access_roles_recorded": False,
            "live_outputs_captured_by_nvapi_trace": False,
            "module_recovery_proven": True,
            "model_recovery_proven": False,
            "metal_replay_proven": False,
            "rtx_mac_parity_proven": False,
        },
        "roots": {
            "ordered_modules_sha256": collection_root(selected_modules),
            "ordered_functions_sha256": collection_root(selected_functions),
            "ordered_resource_requests_sha256": collection_root(resource_a),
            "ordered_launches_sha256": collection_root(ordered_launches),
            "unique_code_hashes_sha256": collection_root(code_hashes),
            "kernel_names_sha256": collection_root(kernel_names),
        },
        "claims": {
            "live_selected_module_path_identified": True,
            "fresh_process_structural_module_equality": True,
            "all_module_and_parameter_dumps_rehashed": True,
            "parameter_bytes_fully_equal_across_processes": not parameter_difference_records,
            "l56_dependency_surface_identified": enforce_production_shape,
            "semantic_resource_dependency_closure": False,
            "complete_model_recovered": False,
            "mac_parity": False,
        },
    }
    result["census_root_sha256"] = canonical_sha256(result)
    return result


def verify_self_root(value: dict[str, Any]) -> None:
    claimed = value.get("census_root_sha256")
    require(isinstance(claimed, str), "census root missing")
    body = copy.deepcopy(value)
    body.pop("census_root_sha256", None)
    require(canonical_sha256(body) == claimed, "census self-root mismatch")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-root", type=Path, default=DEFAULT_RUN_A)
    parser.add_argument("--run-b-root", type=Path, default=DEFAULT_RUN_B)
    parser.add_argument("--rig-inspection-root", type=Path, default=DEFAULT_RIG_INSPECTION)
    parser.add_argument("--abi-inspection-root", type=Path, default=DEFAULT_ABI_INSPECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = build_census(
            args.run_a_root,
            args.run_b_root,
            args.rig_inspection_root,
            args.abi_inspection_root,
            enforce_production_shape=True,
        )
        verify_self_root(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except CensusError as error:
        print(f"fg selected-module census failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": CENSUS_SCHEMA,
                "result": "pass",
                "output": relative_display(args.output),
                "census_root_sha256": result["census_root_sha256"],
                "selected_modules": result["counts"]["selected_module_instances"],
                "unique_code_hashes": result["counts"]["unique_selected_code_hashes"],
                "kernel_names": result["counts"]["unique_kernel_names"],
                "launches": result["counts"]["launches"],
                "parameter_pairs_with_differences": result["counts"]["byte_different_parameter_pairs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
