#!/usr/bin/env python3
"""Build an address-free, two-process NvAPI selected-graph repeat receipt.

Raw launch parameters are compared byte-for-byte. Differences are reported at
their exact offsets and as contiguous runs; this builder never masks or
canonicalizes parameter bytes. Until an independent resource/GPU-VA provenance
receipt explains every difference, those launches remain blocked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any


SCHEMA = "neuralstudio.fg.nvapi-selected-graph-repeat.v2"
TRACE_SCHEMA = "neuralstudio.fg.nvapi-forward-trace.v1"
MODULE21_SHA256 = "11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"
HASH = re.compile(r"^[0-9a-fA-F]{64}$")
INTERFACE_ID = re.compile(r"^0x[0-9A-Fa-f]{8}$")
TOKEN = re.compile(r"^(?:module|function|shader|resource|launch|uav-info)-[0-9]{6}$|^untracked$")
KNOWN_EVENTS = {
    "runtime_ready",
    "phase_marker",
    "interface_query",
    "nvapi_call",
    "module_function_name",
    "kernel_launch",
    "capture_failure",
    "opaque_interface_call_summary",
}
RESOURCE_CALLS = {
    "NvAPI_D3D12_GetCudaMergedTextureSamplerObject",
    "NvAPI_D3D12_GetCudaIndependentDescriptorObject",
    "NvAPI_D3D12_GetCudaTextureObject",
    "NvAPI_D3D12_GetCudaSurfaceObject",
}
MODULE_CALLS = {
    "NvAPI_D3D12_CreateCubinComputeShaderExV2": "legacy_shader",
    "NvAPI_D3D12_CreateCubinComputeShader": "legacy_shader",
    "NvAPI_D3D12_CreateCubinComputeShaderEx": "legacy_shader",
    "NvAPI_D3D12_CreateCubinComputeShaderWithName": "legacy_shader",
    "NvAPI_D3D12_CreateCuModule": "cu_module",
}
LAUNCH_CALLS = {
    "NvAPI_D3D12_LaunchCubinShader",
    "NvAPI_D3D12_LaunchCuKernelChain",
    "NvAPI_D3D12_LaunchCuKernelChainEx",
}
REQUIRED_LIFECYCLE_PHASES = {"create", "release", "shutdown"}


class RepeatError(RuntimeError):
    """Malformed or incomplete evidence that cannot be safely represented."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rooted(value: dict[str, Any], field: str) -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return digest(body)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepeatError(message)


def is_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def normalized_hash(value: str) -> str:
    require(is_hash(value), "SHA-256 is invalid")
    return value.lower()


def validate_dim3(value: Any, label: str) -> list[int]:
    require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int) and item >= 0 for item in value),
        f"{label} must be an unsigned dim3",
    )
    return value


def validate_token(value: Any, label: str, allow_untracked: bool = True) -> str:
    require(
        isinstance(value, str) and TOKEN.fullmatch(value) is not None,
        f"{label} is not a normalized encounter token",
    )
    if not allow_untracked:
        require(value != "untracked", f"{label} is untracked")
    return value


def event_keys(event: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = sorted(set(event) - allowed)
    require(not extra, f"{label} contains unknown fields: {','.join(extra)}")


def load_trace(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"trace is missing or unsafe: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        require(bool(line.strip()), f"blank JSONL line at {path}:{line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RepeatError(f"invalid JSON at {path}:{line_number}: {error}") from error
        require(isinstance(event, dict), f"event is not an object at {path}:{line_number}")
        require(event.get("event") in KNOWN_EVENTS, f"unknown event at {path}:{line_number}")
        event["_line"] = line_number
        events.append(event)
    require(bool(events), f"trace is empty: {path}")
    return events


def safe_dump_file(root: Path, name: Any) -> Path:
    require(isinstance(name, str) and bool(name), "dump file name is missing")
    pure = PurePath(name)
    require(
        not pure.is_absolute()
        and len(pure.parts) == 1
        and pure.name == name
        and name not in {".", ".."}
        and "/" not in name
        and "\\" not in name,
        f"unsafe dump file name: {name}",
    )
    resolved_root = root.resolve()
    lexical = resolved_root / name
    require(not lexical.is_symlink(), f"dump symlink is forbidden: {name}")
    path = lexical.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise RepeatError(f"dump escapes root: {name}") from error
    require(path.is_file(), f"referenced dump is missing: {name}")
    return path


@dataclass
class DumpEvidence:
    name: str | None
    size: int
    sha256: str | None
    data: bytes | None


class DumpInventory:
    def __init__(self, root: Path) -> None:
        require(root.is_dir() and not root.is_symlink(), f"dump root is missing or unsafe: {root}")
        self.root = root.resolve()
        self.referenced: dict[str, dict[str, Any]] = {}

    def capture(self, event: dict[str, Any], prefix: str, label: str) -> DumpEvidence:
        captured = event.get(f"{prefix}_captured")
        size = event.get(f"{prefix}_bytes")
        require(isinstance(captured, bool), f"{label} captured flag is missing")
        require(isinstance(size, int) and size >= 0, f"{label} size is invalid")
        name = event.get(f"{prefix}_file")
        expected_hash = event.get(f"{prefix}_sha256")
        if not captured:
            require(size == 0 and name is None and expected_hash is None, f"{label} uncaptured dump has metadata")
            return DumpEvidence(None, 0, None, None)
        if size == 0:
            require(name is None and expected_hash is None, f"{label} empty dump unexpectedly names a file")
            return DumpEvidence(None, 0, hashlib.sha256(b"").hexdigest(), b"")
        require(is_hash(expected_hash), f"{label} SHA-256 is invalid")
        path = safe_dump_file(self.root, name)
        data = path.read_bytes()
        require(len(data) == size, f"{label} byte count mismatch")
        actual_hash = hashlib.sha256(data).hexdigest()
        require(actual_hash == expected_hash.lower(), f"{label} SHA-256 mismatch")
        require(name not in self.referenced, f"dump referenced more than once: {name}")
        self.referenced[name] = {
            "name": name,
            "bytes": size,
            "sha256": actual_hash,
        }
        return DumpEvidence(name, size, actual_hash, data)

    def finalize(self) -> dict[str, Any]:
        actual: set[str] = set()
        for path in self.root.iterdir():
            require(path.is_file() and not path.is_symlink(), f"unexpected dump entry: {path.name}")
            actual.add(path.name)
        expected = set(self.referenced)
        require(actual == expected, f"dump inventory mismatch: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")
        files = [self.referenced[name] for name in sorted(self.referenced)]
        return {
            "file_count": len(files),
            "files": files,
            "inventory_root_sha256": digest(files),
        }


@dataclass
class AnnotatedEvent:
    event: dict[str, Any]
    phase: str
    phase_segment: int
    ordinal: int


def annotate_phases(events: list[dict[str, Any]]) -> tuple[list[AnnotatedEvent], list[str], list[str]]:
    annotated: list[AnnotatedEvent] = []
    phase_sequence: list[str] = []
    blockers: list[str] = []
    current_phase = "__bootstrap__"
    phase_segment = 0
    runtime_ready_count = 0
    for ordinal, event in enumerate(events):
        kind = event.get("event")
        if kind == "runtime_ready":
            event_keys(event, {"event", "schema", "real_nvapi_sha256", "_line"}, f"line {event['_line']}")
            runtime_ready_count += 1
            require(event.get("schema") == TRACE_SCHEMA, "trace schema mismatch")
            require(is_hash(event.get("real_nvapi_sha256")), "real NvAPI hash is invalid")
        elif kind == "phase_marker":
            event_keys(event, {"event", "phase", "truncated", "_line"}, f"line {event['_line']}")
            phase = event.get("phase")
            require(isinstance(phase, str) and bool(phase), "phase name is missing")
            require(event.get("truncated") is False, f"phase marker is truncated: {phase}")
            current_phase = phase
            phase_sequence.append(phase)
            phase_segment += 1
        elif current_phase == "__bootstrap__":
            blockers.append(f"phase.unscoped_operational.line_{event['_line']}")
        annotated.append(AnnotatedEvent(event, current_phase, phase_segment, ordinal))
    require(runtime_ready_count == 1, "each trace must contain exactly one runtime_ready event")
    require(events[0].get("event") == "runtime_ready", "runtime_ready must be the first event")
    if not phase_sequence:
        blockers.append("phase.markers_missing")
    observed = set(phase_sequence)
    for required_phase in sorted(REQUIRED_LIFECYCLE_PHASES - observed):
        blockers.append(f"phase.required_missing.{required_phase}")
    if not any(
        re.fullmatch(r"evaluate[._][0-9]+", phase) is not None
        for phase in phase_sequence
    ):
        blockers.append("phase.required_missing.evaluate")
    return annotated, phase_sequence, blockers


def phase_counts(events: list[AnnotatedEvent]) -> list[dict[str, Any]]:
    by_segment: dict[tuple[int, str], dict[str, Counter[str]]] = {}
    for item in events:
        key = (item.phase_segment, item.phase)
        bucket = by_segment.setdefault(
            key,
            {"events": Counter(), "calls": Counter(), "statuses": Counter()},
        )
        kind = str(item.event.get("event"))
        bucket["events"][kind] += 1
        if kind == "nvapi_call":
            call = str(item.event.get("call"))
            status = item.event.get("status")
            bucket["calls"][call] += 1
            bucket["statuses"][f"{call}:{status}"] += 1
    result = []
    for (segment, phase), counts in sorted(by_segment.items()):
        result.append(
            {
                "phase_segment": segment,
                "phase": phase,
                "event_counts": dict(sorted(counts["events"].items())),
                "nvapi_call_counts": dict(sorted(counts["calls"].items())),
                "nvapi_status_counts": dict(sorted(counts["statuses"].items())),
            }
        )
    return result


def validate_interface_event(event: dict[str, Any]) -> None:
    event_keys(
        event,
        {
            "event",
            "dispatcher",
            "interface_id",
            "interface",
            "classification",
            "opaque_call_sentinel",
            "real_resolved",
            "returned_wrapper",
            "real_dispatch_mismatch",
            "_line",
        },
        f"interface line {event['_line']}",
    )
    require(event.get("dispatcher") in {"nvapi_QueryInterface", "nvapi_Direct_GetMethod"}, "unknown NvAPI dispatcher")
    require(isinstance(event.get("interface_id"), str) and INTERFACE_ID.fullmatch(event["interface_id"]) is not None, "invalid interface ID")
    for field in ("real_resolved", "returned_wrapper", "real_dispatch_mismatch"):
        require(isinstance(event.get(field), bool), f"interface {field} is invalid")
    if "interface" in event:
        require(isinstance(event["interface"], str) and bool(event["interface"]), "interface name is invalid")
    classification = interface_classification(event)
    require(
        classification
        in {"wrapped", "known_passthrough", "unknown", "opaque_call_sentinel"},
        "interface classification is invalid",
    )
    if classification == "wrapped":
        require("interface" in event, "wrapped interface has no name")
        require(
            event.get("real_resolved") is True
            and event.get("returned_wrapper") is True,
            "wrapped interface was not resolved through its wrapper",
        )
    elif classification == "known_passthrough":
        require("interface" in event, "known passthrough interface has no name")
        require(
            event.get("returned_wrapper") is False,
            "known passthrough unexpectedly returned a wrapper",
        )
    elif classification == "unknown":
        require("interface" not in event, "unknown interface unexpectedly has a name")
        require(
            event.get("returned_wrapper") is False,
            "unknown interface unexpectedly returned a wrapper",
        )
    else:
        require(
            event.get("opaque_call_sentinel") is True
            and event.get("real_resolved") is True
            and event.get("returned_wrapper") is True
            and event.get("interface")
            == f"opaque_{event.get('interface_id')}",
            "opaque call sentinel query is malformed",
        )


def interface_classification(event: dict[str, Any]) -> str:
    declared = event.get("classification")
    if declared is not None:
        require(isinstance(declared, str), "interface classification is not a string")
        return declared
    if "interface" in event and event.get("returned_wrapper") is True:
        return "wrapped"
    if "interface" in event:
        return "known_passthrough"
    return "unknown"


NVAPI_ALLOWED = {
    "event",
    "call",
    "status",
    "shader_token",
    "shader_name",
    "block",
    "dynamic_shared_bytes",
    "module_blob_captured",
    "module_blob_bytes",
    "module_blob_file",
    "module_blob_sha256",
    "input_struct_readable",
    "launch_token",
    "grid",
    "parameter_captured",
    "parameter_bytes",
    "parameter_file",
    "parameter_sha256",
    "resource_kind",
    "resource_token",
    "descriptor_values_recorded",
    "supported",
    "module_token",
    "requested_count",
    "returned_count",
    "function_token",
    "function_name",
    "kernel_count",
    "phase",
    "descriptor_type",
    "volatile_trace_written",
    "uav_info_token",
    "version",
    "gpu_va_size",
    "out_flags",
    "_line",
}


def validate_nvapi_event(event: dict[str, Any]) -> None:
    event_keys(event, NVAPI_ALLOWED, f"NvAPI line {event['_line']}")
    require(isinstance(event.get("call"), str) and bool(event["call"]), "NvAPI call name is missing")
    require(isinstance(event.get("status"), int), "NvAPI status is missing")
    for key, value in event.items():
        if key.endswith("_token"):
            validate_token(value, key)


def validate_kernel_event(event: dict[str, Any]) -> None:
    event_keys(
        event,
        {
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
            "parameter_sha256",
            "_line",
        },
        f"kernel line {event['_line']}",
    )
    validate_token(event.get("launch_token"), "launch_token", False)
    validate_token(event.get("function_token"), "function_token", False)
    validate_token(event.get("module_token"), "module_token", False)
    require(isinstance(event.get("kernel_index"), int) and event["kernel_index"] >= 0, "kernel index is invalid")
    require(isinstance(event.get("function_name"), str), "function name is invalid")
    validate_dim3(event.get("grid"), "kernel grid")
    validate_dim3(event.get("block"), "kernel block")
    require(isinstance(event.get("dynamic_shared_bytes"), int) and event["dynamic_shared_bytes"] >= 0, "kernel shared bytes are invalid")
    require(event.get("parameter_mode") in {"packed", "none", "pointer_array_unbounded_not_captured"}, "parameter mode is invalid")


def validate_named_event(event: dict[str, Any]) -> None:
    event_keys(
        event,
        {"event", "module_token", "index", "function_name", "_line"},
        f"function-name line {event['_line']}",
    )
    validate_token(event.get("module_token"), "module_token")
    require(isinstance(event.get("index"), int) and event["index"] >= 0, "function-name index is invalid")
    require(isinstance(event.get("function_name"), str), "enumerated function name is invalid")


def validate_capture_failure(event: dict[str, Any]) -> None:
    event_keys(event, {"event", "call", "reason", "_line"}, f"capture-failure line {event['_line']}")
    require(isinstance(event.get("call"), str) and bool(event["call"]), "capture failure call is missing")
    require(isinstance(event.get("reason"), str) and bool(event["reason"]), "capture failure reason is missing")


def validate_opaque_summary(event: dict[str, Any]) -> None:
    event_keys(
        event,
        {
            "event",
            "interface_id",
            "phase",
            "invocations_since_last",
            "total_invocations",
            "signature_inferred",
            "tail_jump_preserved_arguments",
            "raw_arguments_in_volatile_trace",
            "volatile_trace_written",
            "_line",
        },
        f"opaque sentinel line {event['_line']}",
    )
    require(
        isinstance(event.get("interface_id"), str)
        and INTERFACE_ID.fullmatch(event["interface_id"]) is not None,
        "opaque sentinel interface ID is invalid",
    )
    require(isinstance(event.get("phase"), str) and bool(event["phase"]), "opaque sentinel phase is invalid")
    require(
        isinstance(event.get("invocations_since_last"), int)
        and event["invocations_since_last"] >= 0
        and isinstance(event.get("total_invocations"), int)
        and event["total_invocations"] >= 0,
        "opaque sentinel invocation count is invalid",
    )
    for field in (
        "signature_inferred",
        "tail_jump_preserved_arguments",
        "raw_arguments_in_volatile_trace",
        "volatile_trace_written",
    ):
        require(isinstance(event.get(field), bool), f"opaque sentinel {field} is invalid")


def differing_offsets(a: bytes, b: bytes) -> list[int]:
    limit = max(len(a), len(b))
    return [
        index
        for index in range(limit)
        if index >= len(a) or index >= len(b) or a[index] != b[index]
    ]


def contiguous_runs(offsets: list[int]) -> list[dict[str, int]]:
    if not offsets:
        return []
    runs: list[dict[str, int]] = []
    start = offsets[0]
    previous = start
    for offset in offsets[1:]:
        if offset != previous + 1:
            runs.append(
                {
                    "start": start,
                    "end_exclusive": previous + 1,
                    "length": previous + 1 - start,
                }
            )
            start = offset
        previous = offset
    runs.append(
        {
            "start": start,
            "end_exclusive": previous + 1,
            "length": previous + 1 - start,
        }
    )
    return runs


def grouped_records(counter: Counter[tuple[Any, ...]], fields: list[str]) -> list[dict[str, Any]]:
    records = []
    for key, count in sorted(counter.items(), key=lambda item: tuple(str(value) for value in item[0])):
        record = {field: value for field, value in zip(fields, key)}
        record["count"] = count
        records.append(record)
    return records


@dataclass
class VolatileEvidence:
    input_receipt: dict[str, Any]
    resources: list[dict[str, Any]]
    uav_info: list[dict[str, Any]]
    handle_map_by_phase: dict[str, dict[bytes, set[str]]]
    source_ambiguities: list[dict[str, Any]]
    blockers: list[str]


def parse_fixed_hex(value: Any, digits: int, label: str) -> int:
    require(
        isinstance(value, str)
        and re.fullmatch(rf"0x[0-9A-Fa-f]{{{digits}}}", value) is not None,
        f"{label} is not a fixed-width hexadecimal value",
    )
    return int(value, 16)


def load_volatile(path: Path, phases: list[str]) -> VolatileEvidence:
    require(
        path.is_file() and not path.is_symlink(),
        f"volatile trace is missing or unsafe: {path}",
    )
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        require(bool(line.strip()), f"blank volatile JSONL line at {path}:{line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RepeatError(
                f"invalid volatile JSON at {path}:{line_number}: {error}"
            ) from error
        require(isinstance(event, dict), f"volatile event is not an object at line {line_number}")
        require(
            event.get("event") in {"resource_object_raw", "uav_info_raw"},
            f"unknown volatile event at {path}:{line_number}",
        )
        event["_line"] = line_number
        events.append(event)
    require(bool(events), f"volatile trace is empty: {path}")

    phase_set = set(phases)
    resources: list[dict[str, Any]] = []
    uav_info: list[dict[str, Any]] = []
    handle_map: dict[str, dict[bytes, set[str]]] = {}
    blockers: list[str] = []
    resource_fields_common = {
        "event",
        "call",
        "phase",
        "resource_kind",
        "resource_token",
        "status",
        "output_field",
        "byte_order",
        "cuda_object_handle_bytes",
        "cuda_object_handle",
        "_line",
    }
    pointer_field_sets = [
        {"descriptor_ptr"},
        {"texture_descriptor_ptr", "sampler_descriptor_ptr"},
    ]
    uav_fields = {
        "event",
        "phase",
        "uav_info_token",
        "status",
        "version",
        "gpu_va_start",
        "gpu_va_size",
        "surface_handle",
        "_line",
    }
    for event in events:
        phase_name = event.get("phase")
        require(
            isinstance(phase_name, str) and phase_name in phase_set,
            f"volatile event phase is not present in the main trace at line {event['_line']}",
        )
        if event["event"] == "resource_object_raw":
            observed_fields = set(event)
            require(
                any(
                    observed_fields == resource_fields_common | pointer_fields
                    for pointer_fields in pointer_field_sets
                ),
                f"volatile resource event fields drifted at line {event['_line']}",
            )
            require(event.get("call") in RESOURCE_CALLS, "volatile resource call is unknown")
            require(isinstance(event.get("resource_kind"), str) and bool(event["resource_kind"]), "volatile resource kind is invalid")
            validate_token(event.get("resource_token"), "volatile resource_token")
            require(isinstance(event.get("status"), int), "volatile resource status is invalid")
            require(isinstance(event.get("output_field"), str) and bool(event["output_field"]), "volatile output field is invalid")
            require(event.get("byte_order") == "little", "volatile handle byte order is not little-endian")
            width = event.get("cuda_object_handle_bytes")
            require(width in {4, 8}, "volatile CUDA object handle width is not 32 or 64 bits")
            handle = parse_fixed_hex(
                event.get("cuda_object_handle"),
                width * 2,
                "volatile CUDA object handle",
            )
            for pointer_field in (
                "descriptor_ptr",
                "texture_descriptor_ptr",
                "sampler_descriptor_ptr",
            ):
                if pointer_field in event:
                    parse_fixed_hex(event[pointer_field], 16, pointer_field)
            status = event["status"]
            token = event["resource_token"]
            if status == 0:
                validate_token(token, "volatile resource_token", False)
                require(handle != 0, "successful volatile resource has a zero handle")
                encoded = handle.to_bytes(width, "little")
                handle_map.setdefault(phase_name, {}).setdefault(encoded, set()).add(token)
            else:
                require(token == "untracked", "failed volatile resource unexpectedly has a token")
                require(handle == 0, "failed volatile resource unexpectedly has a handle")
            resources.append(
                {
                    "encounter_ordinal": len(resources),
                    "phase": phase_name,
                    "call": event["call"],
                    "resource_kind": event["resource_kind"],
                    "resource_token": token,
                    "status": status,
                    "output_field": event["output_field"],
                    "handle_width_bytes": width,
                    "handle_present": handle != 0,
                }
            )
        else:
            require(set(event) == uav_fields, f"volatile UAV event fields drifted at line {event['_line']}")
            validate_token(event.get("uav_info_token"), "uav_info_token", False)
            require(isinstance(event.get("status"), int), "volatile UAV status is invalid")
            require(isinstance(event.get("version"), int) and event["version"] >= 0, "volatile UAV version is invalid")
            gpu_va_start = parse_fixed_hex(event.get("gpu_va_start"), 16, "gpu_va_start")
            gpu_va_size = event.get("gpu_va_size")
            require(isinstance(gpu_va_size, int) and gpu_va_size >= 0, "volatile UAV span size is invalid")
            surface_handle = parse_fixed_hex(event.get("surface_handle"), 8, "surface_handle")
            if gpu_va_size == 0:
                provenance = "zero_span_no_provenance"
                require(gpu_va_start == 0, "zero-sized UAV span has a nonzero start")
            else:
                provenance = "nonzero_span_recorded_not_used_for_handle_normalization"
            uav_info.append(
                {
                    "encounter_ordinal": len(uav_info),
                    "phase": phase_name,
                    "uav_info_token": event["uav_info_token"],
                    "status": event["status"],
                    "version": event["version"],
                    "gpu_va_size": gpu_va_size,
                    "surface_handle_present": surface_handle != 0,
                    "provenance_result": provenance,
                }
            )

    source_ambiguities: list[dict[str, Any]] = []
    for phase_name, candidates in sorted(handle_map.items()):
        for candidate_index, (_encoded, tokens) in enumerate(
            sorted(candidates.items(), key=lambda item: (len(item[0]), sorted(item[1])))
        ):
            if len(tokens) > 1:
                source_ambiguities.append(
                    {
                        "phase": phase_name,
                        "candidate_ordinal": candidate_index,
                        "resource_tokens": sorted(tokens),
                        "result": "ambiguous_raw_handle_maps_to_multiple_tokens",
                    }
                )
    if source_ambiguities:
        blockers.append("volatile.resource_handle_source_ambiguity")
    return VolatileEvidence(
        input_receipt={
            "trace_sha256": digest_file(path),
            "trace_bytes": path.stat().st_size,
            "trace_line_count": len(events),
        },
        resources=resources,
        uav_info=uav_info,
        handle_map_by_phase=handle_map,
        source_ambiguities=source_ambiguities,
        blockers=blockers,
    )


@dataclass
class ParsedRun:
    label: str
    input_receipt: dict[str, Any]
    phases: list[str]
    phase_counts: list[dict[str, Any]]
    modules: list[dict[str, Any]]
    functions: list[dict[str, Any]]
    enumerated_names: list[dict[str, Any]]
    launches: list[dict[str, Any]]
    parameter_data: list[bytes | None]
    resources: list[dict[str, Any]]
    non_model_interfaces: list[dict[str, Any]]
    unknown_interfaces: list[dict[str, Any]]
    failed_calls: list[dict[str, Any]]
    capture_failures: list[dict[str, Any]]
    opaque_sentinel: dict[str, Any]
    volatile: VolatileEvidence | None
    blockers: list[str]


def parse_run(
    label: str,
    trace_path: Path,
    dump_root: Path,
    volatile_path: Path | None = None,
) -> ParsedRun:
    events = load_trace(trace_path)
    annotated, phases, blockers = annotate_phases(events)
    dumps = DumpInventory(dump_root)
    runtime_events = [item.event for item in annotated if item.event.get("event") == "runtime_ready"]
    real_nvapi_sha256 = normalized_hash(runtime_events[0]["real_nvapi_sha256"])

    launch_api_by_token: dict[str, str] = {}
    for item in annotated:
        event = item.event
        if event.get("event") == "nvapi_call":
            validate_nvapi_event(event)
            if "phase" in event:
                require(
                    event.get("phase") == item.phase,
                    f"NvAPI call phase disagrees with marker at line {event['_line']}",
                )
            if "volatile_trace_written" in event:
                require(
                    isinstance(event.get("volatile_trace_written"), bool),
                    "volatile trace flag is invalid",
                )
            if event.get("call") in LAUNCH_CALLS and isinstance(event.get("launch_token"), str):
                launch_api_by_token[event["launch_token"]] = event["call"]
        elif event.get("event") == "interface_query":
            validate_interface_event(event)
        elif event.get("event") == "kernel_launch":
            validate_kernel_event(event)
        elif event.get("event") == "module_function_name":
            validate_named_event(event)
        elif event.get("event") == "capture_failure":
            validate_capture_failure(event)
        elif event.get("event") == "opaque_interface_call_summary":
            validate_opaque_summary(event)

    modules: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    enumerated_names: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    parameter_data: list[bytes | None] = []
    resource_counter: Counter[tuple[Any, ...]] = Counter()
    unknown_counter: Counter[tuple[Any, ...]] = Counter()
    non_model_counter: Counter[tuple[Any, ...]] = Counter()
    failed_counter: Counter[tuple[Any, ...]] = Counter()
    failure_counter: Counter[tuple[Any, ...]] = Counter()
    opaque_queries: list[dict[str, Any]] = []
    opaque_summaries: list[dict[str, Any]] = []

    for item in annotated:
        event = item.event
        kind = event.get("event")
        if kind == "interface_query":
            classification = interface_classification(event)
            if classification == "known_passthrough":
                non_model_counter[
                    (
                        item.phase,
                        "known_passthrough",
                        event.get("dispatcher"),
                        event.get("interface_id"),
                        event.get("interface"),
                        event.get("real_resolved"),
                    )
                ] += 1
            elif classification == "unknown" and event.get("real_resolved") is False:
                non_model_counter[
                    (
                        item.phase,
                        "unresolved_false_probe",
                        event.get("dispatcher"),
                        event.get("interface_id"),
                        None,
                        False,
                    )
                ] += 1
            elif classification == "unknown" and event.get("real_resolved") is True:
                unknown_counter[
                    (
                        item.phase,
                        event.get("dispatcher"),
                        event.get("interface_id"),
                        event.get("real_resolved"),
                        event.get("returned_wrapper"),
                    )
                ] += 1
            elif classification == "opaque_call_sentinel":
                opaque_queries.append(
                    {
                        "phase": item.phase,
                        "dispatcher": event.get("dispatcher"),
                        "interface_id": event.get("interface_id"),
                        "interface": event.get("interface"),
                        "real_resolved": True,
                        "returned_wrapper": True,
                        "opaque_call_sentinel": True,
                    }
                )
            if event.get("real_dispatch_mismatch") is True:
                blockers.append(f"interface.real_dispatch_mismatch.line_{event['_line']}")
            if (
                classification == "wrapped"
                and event.get("real_resolved") is True
                and event.get("returned_wrapper") is not True
            ):
                blockers.append(
                    f"interface.wrapped_not_returned.line_{event['_line']}"
                )
        elif kind == "capture_failure":
            failure_counter[(item.phase, event.get("call"), event.get("reason"))] += 1
        elif kind == "opaque_interface_call_summary":
            require(
                event.get("phase") == item.phase,
                f"opaque sentinel phase disagrees with marker at line {event['_line']}",
            )
            opaque_summaries.append(
                {
                    key: event[key]
                    for key in (
                        "interface_id",
                        "phase",
                        "invocations_since_last",
                        "total_invocations",
                        "signature_inferred",
                        "tail_jump_preserved_arguments",
                        "raw_arguments_in_volatile_trace",
                        "volatile_trace_written",
                    )
                }
            )
        elif kind == "module_function_name":
            enumerated_names.append(
                {
                    "phase": item.phase,
                    "phase_segment": item.phase_segment,
                    "encounter_ordinal": len(enumerated_names),
                    "module_token": event["module_token"],
                    "index": event["index"],
                    "function_name": event["function_name"],
                }
            )
        elif kind == "kernel_launch":
            dump = dumps.capture(event, "parameter", f"{label} kernel line {event['_line']}")
            api = launch_api_by_token.get(event["launch_token"], "unmatched_launch_api")
            launch = {
                "phase": item.phase,
                "phase_segment": item.phase_segment,
                "launch_ordinal": len(launches),
                "launch_api": api,
                "launch_token": event["launch_token"],
                "kernel_index": event["kernel_index"],
                "object_kind": "cu_function",
                "module_token": event["module_token"],
                "function_token": event["function_token"],
                "function_name": event["function_name"],
                "grid": event["grid"],
                "block": event["block"],
                "dynamic_shared_bytes": event["dynamic_shared_bytes"],
                "parameter_mode": event["parameter_mode"],
                "parameter_captured": event["parameter_captured"],
                "parameter_bytes": event["parameter_bytes"],
                "parameter_sha256": dump.sha256,
                "parameter_dump_name": dump.name,
            }
            if api == "unmatched_launch_api":
                blockers.append(f"launch.unmatched_api.ordinal_{len(launches)}")
            if event["parameter_mode"] == "pointer_array_unbounded_not_captured":
                blockers.append(f"parameters.pointer_array_unbounded.ordinal_{len(launches)}")
            launches.append(launch)
            parameter_data.append(dump.data if event["parameter_mode"] != "pointer_array_unbounded_not_captured" else None)
        elif kind == "nvapi_call":
            call = event["call"]
            status = event["status"]
            if status != 0:
                failed_counter[(item.phase, call, status)] += 1
            if call in RESOURCE_CALLS:
                if "volatile_trace_written" in event and event["volatile_trace_written"] is not True:
                    blockers.append(f"volatile.resource_event_not_written.line_{event['_line']}")
                if status == 0:
                    validate_token(
                        event.get("resource_token"), "resource_token", False
                    )
                resource_counter[(item.phase, call, event.get("resource_kind"), status)] += 1
            if call in MODULE_CALLS:
                dump = dumps.capture(event, "module_blob", f"{label} module line {event['_line']}")
                object_kind = MODULE_CALLS[call]
                token_field = "module_token" if object_kind == "cu_module" else "shader_token"
                if status == 0:
                    validate_token(event.get(token_field), token_field, False)
                    require(dump.size > 0 and dump.data is not None, "successful module blob is empty")
                    modules.append(
                        {
                            "phase": item.phase,
                            "phase_segment": item.phase_segment,
                            "encounter_ordinal": len(modules),
                            "api": call,
                            "object_kind": object_kind,
                            "encounter_token": event.get(token_field, "untracked"),
                            "name": event.get("shader_name", ""),
                            "blob_bytes": dump.size,
                            "blob_sha256": dump.sha256,
                            "status": status,
                        }
                    )
            if call == "NvAPI_D3D12_CreateCuFunction":
                if status == 0:
                    validate_token(event.get("module_token"), "module_token", False)
                    validate_token(event.get("function_token"), "function_token", False)
                    functions.append(
                        {
                            "phase": item.phase,
                            "phase_segment": item.phase_segment,
                            "encounter_ordinal": len(functions),
                            "api": call,
                            "module_token": event.get("module_token"),
                            "function_token": event.get("function_token"),
                            "function_name": event.get("function_name", ""),
                            "status": status,
                        }
                    )
            if call == "NvAPI_D3D12_LaunchCubinShader":
                if status == 0:
                    validate_token(event.get("launch_token"), "launch_token", False)
                    validate_token(event.get("shader_token"), "shader_token", False)
                dump = dumps.capture(event, "parameter", f"{label} legacy launch line {event['_line']}")
                launches.append(
                    {
                        "phase": item.phase,
                        "phase_segment": item.phase_segment,
                        "launch_ordinal": len(launches),
                        "launch_api": call,
                        "launch_token": event.get("launch_token"),
                        "kernel_index": 0,
                        "object_kind": "legacy_shader",
                        "module_token": event.get("shader_token"),
                        "function_token": event.get("shader_token"),
                        "function_name": event.get("shader_name", ""),
                        "grid": validate_dim3(event.get("grid"), "legacy grid"),
                        "block": validate_dim3(event.get("block"), "legacy block"),
                        "dynamic_shared_bytes": event.get("dynamic_shared_bytes"),
                        "parameter_mode": "packed" if dump.size else "none",
                        "parameter_captured": event.get("parameter_captured"),
                        "parameter_bytes": event.get("parameter_bytes"),
                        "parameter_sha256": dump.sha256,
                        "parameter_dump_name": dump.name,
                    }
                )
                parameter_data.append(dump.data)

    inventory = dumps.finalize()
    modules_root = digest(modules)
    functions_root = digest(
        {"created": functions, "enumerated_names": enumerated_names}
    )
    resources = grouped_records(
        resource_counter, ["phase", "call", "resource_kind", "status"]
    )
    unknown_interfaces = grouped_records(
        unknown_counter,
        ["phase", "dispatcher", "interface_id", "real_resolved", "returned_wrapper"],
    )
    non_model_interfaces = grouped_records(
        non_model_counter,
        [
            "phase",
            "classification",
            "dispatcher",
            "interface_id",
            "interface",
            "real_resolved",
        ],
    )
    failed_calls = grouped_records(failed_counter, ["phase", "call", "status"])
    capture_failures = grouped_records(
        failure_counter, ["phase", "call", "reason"]
    )
    if unknown_interfaces:
        blockers.append("interfaces.unknown_ids_present")
    if any(
        re.fullmatch(r"evaluate[._][0-9]+", str(record.get("phase")))
        is not None
        for record in failed_calls
    ):
        blockers.append("calls.failed_during_evaluate")
    if capture_failures:
        blockers.append("capture.failures_present")
    if not launches:
        blockers.append("launches.none_captured")

    sentinel_ids = sorted({item["interface_id"] for item in opaque_queries})
    summary_ids = sorted({item["interface_id"] for item in opaque_summaries})
    sentinel_phase_coverage: list[dict[str, Any]] = []
    if sentinel_ids or summary_ids:
        if sentinel_ids != summary_ids:
            blockers.append("interfaces.opaque_sentinel_query_summary_id_mismatch")
        for interface_id in sorted(set(sentinel_ids) | set(summary_ids)):
            queries = [item for item in opaque_queries if item["interface_id"] == interface_id]
            summaries = [
                item for item in opaque_summaries if item["interface_id"] == interface_id
            ]
            first_query_phase = queries[0]["phase"] if queries else None
            expected_phases: list[str] = []
            if first_query_phase in phases:
                expected_phases = phases[phases.index(first_query_phase) :]
                if expected_phases and expected_phases[-1].endswith(".end"):
                    expected_phases = expected_phases[:-1]
            observed_phases = [item["phase"] for item in summaries]
            counts_zero = bool(summaries) and all(
                item["invocations_since_last"] == 0
                and item["total_invocations"] == 0
                for item in summaries
            )
            safe_sentinel = (
                bool(queries)
                and observed_phases == expected_phases
                and counts_zero
                and all(
                    item["signature_inferred"] is False
                    and item["tail_jump_preserved_arguments"] is True
                    and item["raw_arguments_in_volatile_trace"] is False
                    and item["volatile_trace_written"] is True
                    for item in summaries
                )
            )
            sentinel_phase_coverage.append(
                {
                    "interface_id": interface_id,
                    "query_count": len(queries),
                    "first_query_phase": first_query_phase,
                    "expected_summary_phases": expected_phases,
                    "observed_summary_phases": observed_phases,
                    "all_invocation_counts_zero": counts_zero,
                    "resolved_never_invoked_non_model_dependency": safe_sentinel,
                }
            )
            if safe_sentinel:
                non_model_counter[
                    (
                        first_query_phase,
                        "resolved_never_invoked_sentinel",
                        queries[0]["dispatcher"],
                        interface_id,
                        queries[0]["interface"],
                        True,
                    )
                ] += len(queries)
            else:
                blockers.append(f"interfaces.opaque_sentinel_unproven.{interface_id}")
            if not counts_zero:
                blockers.append(f"interfaces.opaque_sentinel_invoked.{interface_id}")
        non_model_interfaces = grouped_records(
            non_model_counter,
            [
                "phase",
                "classification",
                "dispatcher",
                "interface_id",
                "interface",
                "real_resolved",
            ],
        )

    volatile = load_volatile(volatile_path, phases) if volatile_path is not None else None
    if volatile is not None:
        blockers.extend(volatile.blockers)
        volatile_resource_counter: Counter[tuple[Any, ...]] = Counter()
        for record in volatile.resources:
            volatile_resource_counter[
                (
                    record["phase"],
                    record["call"],
                    record["resource_kind"],
                    record["status"],
                )
            ] += 1
        volatile_resource_counts = grouped_records(
            volatile_resource_counter,
            ["phase", "call", "resource_kind", "status"],
        )
        if volatile_resource_counts != resources:
            blockers.append("volatile.resource_call_census_mismatch")
        main_uav_counter: Counter[tuple[Any, ...]] = Counter()
        for item in annotated:
            if (
                item.event.get("event") == "nvapi_call"
                and item.event.get("call") == "NvAPI_D3D12_CaptureUAVInfo"
            ):
                if item.event.get("volatile_trace_written") is not True:
                    blockers.append(
                        f"volatile.uav_event_not_written.line_{item.event['_line']}"
                    )
                main_uav_counter[(item.phase, item.event.get("status"))] += 1
        volatile_uav_counter: Counter[tuple[Any, ...]] = Counter()
        for record in volatile.uav_info:
            volatile_uav_counter[(record["phase"], record["status"])] += 1
        if main_uav_counter != volatile_uav_counter:
            blockers.append("volatile.uav_info_census_mismatch")

    return ParsedRun(
        label=label,
        input_receipt={
            "trace_sha256": digest_file(trace_path),
            "trace_bytes": trace_path.stat().st_size,
            "trace_line_count": len(events),
            "real_nvapi_sha256": real_nvapi_sha256,
            "dump_inventory": inventory,
            "runtime_ready_first": events[0].get("event") == "runtime_ready",
            "runtime_ready_count": len(runtime_events),
            "volatile_trace": (
                copy.deepcopy(volatile.input_receipt) if volatile is not None else None
            ),
        },
        phases=phases,
        phase_counts=phase_counts(annotated),
        modules=modules,
        functions=functions,
        enumerated_names=enumerated_names,
        launches=launches,
        parameter_data=parameter_data,
        resources=resources,
        non_model_interfaces=non_model_interfaces,
        unknown_interfaces=unknown_interfaces,
        failed_calls=failed_calls,
        capture_failures=capture_failures,
        opaque_sentinel={
            "queries": opaque_queries,
            "summaries": opaque_summaries,
            "phase_coverage": sentinel_phase_coverage,
            "root_sha256": digest(
                {
                    "queries": opaque_queries,
                    "summaries": opaque_summaries,
                    "phase_coverage": sentinel_phase_coverage,
                }
            ),
        },
        volatile=volatile,
        blockers=sorted(set(blockers)),
    )


def launch_structure(launch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in launch.items()
        if key not in {"parameter_sha256", "parameter_dump_name"}
    }


def compare_parameters(a: ParsedRun, b: ParsedRun) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    exact = 0
    unresolved = 0
    count = max(len(a.launches), len(b.launches))
    for ordinal in range(count):
        launch_a = a.launches[ordinal] if ordinal < len(a.launches) else None
        launch_b = b.launches[ordinal] if ordinal < len(b.launches) else None
        data_a = a.parameter_data[ordinal] if ordinal < len(a.parameter_data) else None
        data_b = b.parameter_data[ordinal] if ordinal < len(b.parameter_data) else None
        base = {
            "launch_ordinal": ordinal,
            "phase_a": launch_a.get("phase") if launch_a else None,
            "phase_b": launch_b.get("phase") if launch_b else None,
            "launch_api_a": launch_a.get("launch_api") if launch_a else None,
            "launch_api_b": launch_b.get("launch_api") if launch_b else None,
            "parameter_bytes_a": launch_a.get("parameter_bytes") if launch_a else None,
            "parameter_bytes_b": launch_b.get("parameter_bytes") if launch_b else None,
            "parameter_sha256_a": launch_a.get("parameter_sha256") if launch_a else None,
            "parameter_sha256_b": launch_b.get("parameter_sha256") if launch_b else None,
            "differing_byte_offsets": [],
            "differing_byte_runs": [],
            "provenance_resolution": "not_needed_exact",
        }
        if launch_a is None or launch_b is None:
            base["comparison"] = "missing_launch"
            base["provenance_resolution"] = "unresolved_no_resource_gpu_va_provenance"
            unresolved += 1
        elif data_a is None or data_b is None:
            base["comparison"] = "raw_parameters_not_captured"
            base["provenance_resolution"] = "unresolved_no_resource_gpu_va_provenance"
            unresolved += 1
        elif data_a == data_b:
            base["comparison"] = "exact_raw_equal"
            exact += 1
        else:
            offsets = differing_offsets(data_a, data_b)
            base["comparison"] = "raw_bytes_differ"
            base["differing_byte_offsets"] = offsets
            base["differing_byte_runs"] = contiguous_runs(offsets)
            base["provenance_resolution"] = "unresolved_no_resource_gpu_va_provenance"
            unresolved += 1
        comparisons.append(base)
    return {
        "launch_comparison_count": count,
        "exact_raw_parameter_equality_count": exact,
        "differing_or_unresolved_launch_count": unresolved,
        "all_raw_parameters_exact": count > 0 and exact == count,
        "provenance_canonicalization_applied": False,
        "speculative_masking_applied": False,
        "comparisons": comparisons,
        "comparison_root_sha256": digest(comparisons),
    }


def canonical_resource_token_bytes(resource_token: str, width: int) -> bytes:
    require(width in {4, 8}, "canonical resource-token width is invalid")
    payload = (
        b"neuralstudio.fg.resource-token.v1\0"
        + resource_token.encode("ascii")
        + b"\0"
        + str(width).encode("ascii")
    )
    return hashlib.sha256(payload).digest()[:width]


def normalize_parameter_handles(
    data: bytes | None,
    phase: str | None,
    volatile: VolatileEvidence | None,
) -> dict[str, Any]:
    if data is None:
        return {
            "normalized_data": None,
            "normalized_sha256": None,
            "substitutions": [],
            "ambiguities": [],
            "result": "parameter_bytes_unavailable",
        }
    if volatile is None:
        return {
            "normalized_data": data,
            "normalized_sha256": hashlib.sha256(data).hexdigest(),
            "substitutions": [],
            "ambiguities": [],
            "result": "no_volatile_provenance_supplied",
        }
    candidates = volatile.handle_map_by_phase.get(str(phase), {})
    matches: list[dict[str, Any]] = []
    for raw_bytes, tokens in candidates.items():
        if len(tokens) != 1:
            continue
        token = next(iter(tokens))
        start = 0
        while True:
            offset = data.find(raw_bytes, start)
            if offset < 0:
                break
            matches.append(
                {
                    "offset": offset,
                    "end_exclusive": offset + len(raw_bytes),
                    "width": len(raw_bytes),
                    "resource_token": token,
                }
            )
            start = offset + 1
    matches.sort(
        key=lambda item: (
            item["offset"],
            item["end_exclusive"],
            item["resource_token"],
        )
    )
    groups: list[list[dict[str, Any]]] = []
    for match in matches:
        if not groups or match["offset"] >= max(
            item["end_exclusive"] for item in groups[-1]
        ):
            groups.append([match])
        else:
            groups[-1].append(match)
    normalized = bytearray(data)
    substitutions: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for group in groups:
        if len(group) != 1:
            ambiguities.append(
                {
                    "start": min(item["offset"] for item in group),
                    "end_exclusive": max(item["end_exclusive"] for item in group),
                    "candidates": [
                        {
                            "offset": item["offset"],
                            "width": item["width"],
                            "resource_token": item["resource_token"],
                        }
                        for item in group
                    ],
                    "result": "overlapping_handle_matches_not_normalized",
                }
            )
            continue
        match = group[0]
        replacement = canonical_resource_token_bytes(
            match["resource_token"], match["width"]
        )
        normalized[match["offset"] : match["end_exclusive"]] = replacement
        substitutions.append(
            {
                "offset": match["offset"],
                "width": match["width"],
                "resource_token": match["resource_token"],
                "canonical_replacement_sha256": hashlib.sha256(
                    replacement
                ).hexdigest(),
            }
        )
    normalized_bytes = bytes(normalized)
    return {
        "normalized_data": normalized_bytes,
        "normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "substitutions": substitutions,
        "ambiguities": ambiguities,
        "result": (
            "ambiguous_matches_preserved"
            if ambiguities
            else "matched_handles_normalized_unmatched_bytes_preserved"
        ),
    }


def compare_handle_normalized_parameters(
    a: ParsedRun, b: ParsedRun
) -> dict[str, Any]:
    enabled = a.volatile is not None and b.volatile is not None
    count = max(len(a.launches), len(b.launches))
    comparisons: list[dict[str, Any]] = []
    exact = 0
    unresolved = 0
    total_substitutions_a = 0
    total_substitutions_b = 0
    ambiguity_count = (
        len(a.volatile.source_ambiguities) if a.volatile is not None else 0
    ) + (len(b.volatile.source_ambiguities) if b.volatile is not None else 0)
    for ordinal in range(count):
        launch_a = a.launches[ordinal] if ordinal < len(a.launches) else None
        launch_b = b.launches[ordinal] if ordinal < len(b.launches) else None
        data_a = a.parameter_data[ordinal] if ordinal < len(a.parameter_data) else None
        data_b = b.parameter_data[ordinal] if ordinal < len(b.parameter_data) else None
        normalized_a = normalize_parameter_handles(
            data_a,
            launch_a.get("phase") if launch_a else None,
            a.volatile,
        )
        normalized_b = normalize_parameter_handles(
            data_b,
            launch_b.get("phase") if launch_b else None,
            b.volatile,
        )
        total_substitutions_a += len(normalized_a["substitutions"])
        total_substitutions_b += len(normalized_b["substitutions"])
        ambiguity_count += len(normalized_a["ambiguities"]) + len(
            normalized_b["ambiguities"]
        )
        result: dict[str, Any] = {
            "launch_ordinal": ordinal,
            "phase_a": launch_a.get("phase") if launch_a else None,
            "phase_b": launch_b.get("phase") if launch_b else None,
            "launch_api_a": launch_a.get("launch_api") if launch_a else None,
            "launch_api_b": launch_b.get("launch_api") if launch_b else None,
            "normalized_sha256_a": normalized_a["normalized_sha256"],
            "normalized_sha256_b": normalized_b["normalized_sha256"],
            "substitutions_a": normalized_a["substitutions"],
            "substitutions_b": normalized_b["substitutions"],
            "ambiguities_a": normalized_a["ambiguities"],
            "ambiguities_b": normalized_b["ambiguities"],
            "differing_byte_offsets_after_normalization": [],
            "differing_byte_runs_after_normalization": [],
            "unmatched_bytes_preserved": True,
        }
        normalized_data_a = normalized_a["normalized_data"]
        normalized_data_b = normalized_b["normalized_data"]
        if launch_a is None or launch_b is None:
            result["comparison"] = "missing_launch"
            unresolved += 1
        elif normalized_data_a is None or normalized_data_b is None:
            result["comparison"] = "parameter_bytes_unavailable"
            unresolved += 1
        elif normalized_a["ambiguities"] or normalized_b["ambiguities"]:
            offsets = differing_offsets(normalized_data_a, normalized_data_b)
            result["comparison"] = "ambiguous_handle_match"
            result["differing_byte_offsets_after_normalization"] = offsets
            result["differing_byte_runs_after_normalization"] = contiguous_runs(
                offsets
            )
            unresolved += 1
        elif normalized_data_a == normalized_data_b:
            result["comparison"] = "address_free_exact"
            exact += 1
        else:
            offsets = differing_offsets(normalized_data_a, normalized_data_b)
            result["comparison"] = "unmatched_bytes_differ"
            result["differing_byte_offsets_after_normalization"] = offsets
            result["differing_byte_runs_after_normalization"] = contiguous_runs(
                offsets
            )
            unresolved += 1
        comparisons.append(result)
    all_exact = enabled and count > 0 and exact == count and ambiguity_count == 0
    public_comparisons = copy.deepcopy(comparisons)
    return {
        "enabled": enabled,
        "launch_comparison_count": count,
        "address_free_exact_count": exact,
        "unresolved_launch_count": unresolved,
        "all_address_free_parameters_exact": all_exact,
        "matched_handle_occurrence_count_a": total_substitutions_a,
        "matched_handle_occurrence_count_b": total_substitutions_b,
        "ambiguity_count": ambiguity_count,
        "unmatched_bytes_preserved": True,
        "descriptor_pointer_normalization_applied": False,
        "gpu_va_normalization_applied": False,
        "comparisons": public_comparisons,
        "comparison_root_sha256": digest(public_comparisons),
    }


@dataclass
class PaddingEvidence:
    public: dict[str, Any]
    unused_by_all_variants: set[int]


PTX_TYPE_BYTES = {
    "u8": 1,
    "s8": 1,
    "b8": 1,
    "u16": 2,
    "s16": 2,
    "b16": 2,
    "f16": 2,
    "u32": 4,
    "s32": 4,
    "b32": 4,
    "f32": 4,
    "u64": 8,
    "s64": 8,
    "b64": 8,
    "f64": 8,
}


def load_module21_padding_proof(
    ptx_path: Path, receipt_path: Path
) -> PaddingEvidence:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    require(
        isinstance(receipt, dict)
        and receipt.get("schema")
        == "neuralstudio.fg.module21-param-access-input.v1"
        and receipt.get("source_sha256") == MODULE21_SHA256
        and receipt.get("source_bytes") == 38472
        and receipt.get("exit_code") == 0,
        "module 21 PTX input receipt identity mismatch",
    )
    require(
        ptx_path.is_file()
        and not ptx_path.is_symlink()
        and digest_file(ptx_path) == receipt.get("output_sha256"),
        "module 21 PTX hash mismatch",
    )
    text = ptx_path.read_text(encoding="utf-8-sig")
    blocks = [
        block
        for block in re.split(r"(?=Fatbin ptx code:)", text)
        if ".visible .entry main_kernel" in block
    ]
    require(len(blocks) == 2, "module 21 must expose exactly two PTX variants")
    load_pattern = re.compile(
        r"(?:\.pragma\s+\"used_bytes_mask\s+(?P<mask>[0-9]+)\";\s*)?"
        r"ld\.param(?:\.v(?P<vector>[0-9]+))?\."
        r"(?P<type>[usb][0-9]+|f(?:16|32|64))\s+[^\n]*"
        r"\[(?P<address>[^\]]+)\]"
    )
    variants: list[dict[str, Any]] = []
    union_accessed: set[int] = set()
    for block in blocks:
        architecture = re.search(r"arch\s*=\s*(sm_[0-9]+)", block)
        parameter = re.search(
            r"\.param\s+\.align\s+16\s+\.b8\s+main_kernel_param_0\[(\d+)\]",
            block,
        )
        require(
            architecture is not None
            and parameter is not None
            and int(parameter.group(1)) == 224
            and re.search(
                r"mov\.b64\s+%rd\d+,\s*main_kernel_param_0", block
            )
            is not None,
            "module 21 PTX parameter block identity mismatch",
        )
        matches = list(load_pattern.finditer(block))
        require(
            len(matches)
            == sum("ld.param" in line for line in block.splitlines()),
            "unsupported module 21 ld.param form",
        )
        accessed: set[int] = set()
        instructions: list[dict[str, Any]] = []
        for match in matches:
            scalar = match.group("type")
            require(scalar in PTX_TYPE_BYTES, "unknown module 21 PTX load type")
            vector = int(match.group("vector") or "1")
            width = PTX_TYPE_BYTES[scalar] * vector
            offset_match = re.search(
                r"\+(0x[0-9A-Fa-f]+|[0-9]+)\s*$", match.group("address")
            )
            offset = int(offset_match.group(1), 0) if offset_match else 0
            require(offset + width <= 224, "module 21 PTX access exceeds parameter block")
            used_mask = (
                int(match.group("mask"))
                if match.group("mask") is not None
                else (1 << width) - 1
            )
            require(0 <= used_mask < (1 << width), "module 21 used-bytes mask is invalid")
            used_offsets = [
                offset + byte
                for byte in range(width)
                if used_mask & (1 << byte)
            ]
            accessed.update(used_offsets)
            instructions.append(
                {
                    "offset": offset,
                    "bytes": width,
                    "type": scalar,
                    "vector": vector,
                    "used_bytes_mask": used_mask,
                    "used_byte_offsets": used_offsets,
                }
            )
        union_accessed.update(accessed)
        variants.append(
            {
                "architecture": architecture.group(1),
                "parameter_bytes": 224,
                "ld_param_instruction_count": len(matches),
                "accessed_byte_count": len(accessed),
                "accessed_ranges_root_sha256": digest(instructions),
                "byte_156_accessed": 156 in accessed,
                "bytes_157_159_accessed": any(
                    byte in accessed for byte in (157, 158, 159)
                ),
                "byte_160_accessed": 160 in accessed,
            }
        )
    require(
        {variant["architecture"] for variant in variants} == {"sm_89", "sm_120"}
        and all(
            variant["byte_156_accessed"]
            and not variant["bytes_157_159_accessed"]
            and variant["byte_160_accessed"]
            for variant in variants
        ),
        "module 21 padding is read by a selected architecture",
    )
    unused = set(range(224)) - union_accessed
    public = {
        "module_sha256": MODULE21_SHA256,
        "ptx_input_receipt_sha256": digest_file(receipt_path),
        "ptx_sha256": digest_file(ptx_path),
        "variants": variants,
        "unused_by_all_variants_root_sha256": digest(sorted(unused)),
        "padding_157_159_unused_by_all_variants": {157, 158, 159}.issubset(
            unused
        ),
    }
    public["proof_root_sha256"] = digest(public)
    return PaddingEvidence(public, unused)


def module_hashes(parsed: ParsedRun) -> dict[str, str]:
    return {
        module["encounter_token"]: module["blob_sha256"]
        for module in parsed.modules
        if module.get("object_kind") == "cu_module"
    }


def compare_module_bound_padding(
    run_a: ParsedRun, run_b: ParsedRun, proof: PaddingEvidence | None
) -> dict[str, Any]:
    count = max(len(run_a.launches), len(run_b.launches))
    hashes_a = module_hashes(run_a)
    hashes_b = module_hashes(run_b)
    comparisons: list[dict[str, Any]] = []
    exact_before = 0
    exact_after = 0
    substitutions = 0
    for ordinal in range(count):
        launch_a = run_a.launches[ordinal] if ordinal < len(run_a.launches) else None
        launch_b = run_b.launches[ordinal] if ordinal < len(run_b.launches) else None
        raw_a = run_a.parameter_data[ordinal] if ordinal < len(run_a.parameter_data) else None
        raw_b = run_b.parameter_data[ordinal] if ordinal < len(run_b.parameter_data) else None
        handle_a = normalize_parameter_handles(
            raw_a, launch_a.get("phase") if launch_a else None, run_a.volatile
        )
        handle_b = normalize_parameter_handles(
            raw_b, launch_b.get("phase") if launch_b else None, run_b.volatile
        )
        data_a = handle_a["normalized_data"]
        data_b = handle_b["normalized_data"]
        before = (
            differing_offsets(data_a, data_b)
            if data_a is not None and data_b is not None
            else []
        )
        if data_a is not None and data_a == data_b:
            exact_before += 1
        canonical_a = bytearray(data_a) if data_a is not None else None
        canonical_b = bytearray(data_b) if data_b is not None else None
        padding_substitutions: list[dict[str, Any]] = []
        if proof is not None and canonical_a is not None and canonical_b is not None and launch_a and launch_b:
            module_a = hashes_a.get(launch_a.get("module_token"))
            module_b = hashes_b.get(launch_b.get("module_token"))
            for offset in before:
                if (
                    module_a == module_b == MODULE21_SHA256
                    and len(canonical_a) == len(canonical_b) == 224
                    and offset in proof.unused_by_all_variants
                ):
                    canonical_a[offset] = 0
                    canonical_b[offset] = 0
                    substitutions += 1
                    padding_substitutions.append(
                        {
                            "offset": offset,
                            "module_sha256": MODULE21_SHA256,
                            "unused_by_all_selected_architectures": True,
                            "proof_root_sha256": proof.public[
                                "proof_root_sha256"
                            ],
                        }
                    )
        final_a = bytes(canonical_a) if canonical_a is not None else None
        final_b = bytes(canonical_b) if canonical_b is not None else None
        after = (
            differing_offsets(final_a, final_b)
            if final_a is not None and final_b is not None
            else []
        )
        if final_a is not None and final_a == final_b:
            exact_after += 1
        comparisons.append(
            {
                "launch_ordinal": ordinal,
                "phase": launch_a.get("phase") if launch_a else None,
                "module_token": launch_a.get("module_token") if launch_a else None,
                "function_token": launch_a.get("function_token") if launch_a else None,
                "function_name": launch_a.get("function_name") if launch_a else None,
                "differing_offsets_before_padding_proof": before,
                "differing_runs_before_padding_proof": contiguous_runs(before),
                "padding_substitutions": padding_substitutions,
                "differing_offsets_after_padding_proof": after,
                "differing_runs_after_padding_proof": contiguous_runs(after),
                "canonical_equal": final_a is not None and final_a == final_b,
                "unmatched_bytes_preserved": True,
            }
        )
    return {
        "enabled": proof is not None,
        "launch_comparison_count": count,
        "exact_before_padding_proof_count": exact_before,
        "exact_after_padding_proof_count": exact_after,
        "all_module_bound_canonical_parameters_exact": proof is not None
        and count > 0
        and exact_after == count,
        "padding_substitution_count": substitutions,
        "proof": copy.deepcopy(proof.public) if proof is not None else None,
        "comparisons": comparisons,
        "comparison_root_sha256": digest(comparisons),
    }


def run_projection(run: ParsedRun) -> dict[str, Any]:
    return {
        "phase_sequence": run.phases,
        "phase_scoped_counts": run.phase_counts,
        "modules": run.modules,
        "functions": run.functions,
        "enumerated_names": run.enumerated_names,
        "launches": [launch_structure(launch) for launch in run.launches],
        "resource_call_counts": run.resources,
        "non_model_interface_dependencies": run.non_model_interfaces,
        "unknown_interface_ids": run.unknown_interfaces,
        "failed_real_calls": run.failed_calls,
        "capture_failures": run.capture_failures,
        "opaque_interface_sentinel": run.opaque_sentinel,
    }


def build_repeat(
    trace_a: Path,
    dumps_a: Path,
    trace_b: Path,
    dumps_b: Path,
    volatile_a: Path | None = None,
    volatile_b: Path | None = None,
    module21_ptx: Path | None = None,
    module21_ptx_receipt: Path | None = None,
) -> tuple[dict[str, Any], int]:
    require(
        (volatile_a is None) == (volatile_b is None),
        "volatile provenance traces must be supplied as a pair",
    )
    require(
        (module21_ptx is None) == (module21_ptx_receipt is None),
        "module 21 PTX and its input receipt must be supplied as a pair",
    )
    require(
        trace_a.resolve() != trace_b.resolve(),
        "fresh-process trace paths must be distinct",
    )
    require(
        dumps_a.resolve() != dumps_b.resolve(),
        "fresh-process dump roots must be distinct",
    )
    if volatile_a is not None and volatile_b is not None:
        require(
            volatile_a.resolve() != volatile_b.resolve(),
            "fresh-process volatile trace paths must be distinct",
        )
    run_a = parse_run("run_a", trace_a, dumps_a, volatile_a)
    run_b = parse_run("run_b", trace_b, dumps_b, volatile_b)
    parameters = compare_parameters(run_a, run_b)
    normalized_parameters = compare_handle_normalized_parameters(run_a, run_b)
    padding_proof = (
        load_module21_padding_proof(module21_ptx, module21_ptx_receipt)
        if module21_ptx is not None and module21_ptx_receipt is not None
        else None
    )
    padding_parameters = compare_module_bound_padding(
        run_a, run_b, padding_proof
    )
    projection_a = run_projection(run_a)
    projection_b = run_projection(run_b)
    structural_root_a = digest(projection_a)
    structural_root_b = digest(projection_b)
    module_root_a = digest(run_a.modules)
    module_root_b = digest(run_b.modules)
    function_root_a = digest(
        {"created": run_a.functions, "enumerated_names": run_a.enumerated_names}
    )
    function_root_b = digest(
        {"created": run_b.functions, "enumerated_names": run_b.enumerated_names}
    )
    phase_root_a = digest(run_a.phase_counts)
    phase_root_b = digest(run_b.phase_counts)
    resource_root_a = digest(run_a.resources)
    resource_root_b = digest(run_b.resources)
    unknown_root_a = digest(run_a.unknown_interfaces)
    unknown_root_b = digest(run_b.unknown_interfaces)
    non_model_root_a = digest(run_a.non_model_interfaces)
    non_model_root_b = digest(run_b.non_model_interfaces)
    failed_root_a = digest(run_a.failed_calls)
    failed_root_b = digest(run_b.failed_calls)
    capture_failure_root_a = digest(run_a.capture_failures)
    capture_failure_root_b = digest(run_b.capture_failures)
    opaque_sentinel_root_a = run_a.opaque_sentinel["root_sha256"]
    opaque_sentinel_root_b = run_b.opaque_sentinel["root_sha256"]
    volatile_resource_root_a = digest(
        run_a.volatile.resources if run_a.volatile is not None else []
    )
    volatile_resource_root_b = digest(
        run_b.volatile.resources if run_b.volatile is not None else []
    )
    uav_info_root_a = digest(
        run_a.volatile.uav_info if run_a.volatile is not None else []
    )
    uav_info_root_b = digest(
        run_b.volatile.uav_info if run_b.volatile is not None else []
    )

    blockers = set(run_a.blockers + run_b.blockers)
    if run_a.phases != run_b.phases:
        blockers.add("repeat.phase_sequence_mismatch")
    if phase_root_a != phase_root_b:
        blockers.add("repeat.phase_scoped_counts_mismatch")
    if run_a.input_receipt["real_nvapi_sha256"] != run_b.input_receipt["real_nvapi_sha256"]:
        blockers.add("repeat.real_nvapi_identity_mismatch")
    if module_root_a != module_root_b:
        blockers.add("repeat.module_blob_root_mismatch")
    if function_root_a != function_root_b:
        blockers.add("repeat.function_encounters_mismatch")
    if resource_root_a != resource_root_b:
        blockers.add("repeat.resource_call_counts_mismatch")
    if unknown_root_a != unknown_root_b:
        blockers.add("repeat.unknown_interface_census_mismatch")
    if non_model_root_a != non_model_root_b:
        blockers.add("repeat.non_model_interface_census_mismatch")
    if failed_root_a != failed_root_b:
        blockers.add("repeat.failed_real_call_census_mismatch")
    if capture_failure_root_a != capture_failure_root_b:
        blockers.add("repeat.capture_failure_census_mismatch")
    if opaque_sentinel_root_a != opaque_sentinel_root_b:
        blockers.add("repeat.opaque_interface_sentinel_mismatch")
    if volatile_resource_root_a != volatile_resource_root_b:
        blockers.add("repeat.resource_handle_provenance_mismatch")
    if uav_info_root_a != uav_info_root_b:
        blockers.add("repeat.uav_info_census_mismatch")
    if structural_root_a != structural_root_b:
        blockers.add("repeat.structural_graph_mismatch")
    if padding_parameters["enabled"]:
        if not padding_parameters["all_module_bound_canonical_parameters_exact"]:
            blockers.add("parameters.unresolved_after_module_padding_proof")
    elif normalized_parameters["enabled"]:
        if normalized_parameters["ambiguity_count"]:
            blockers.add("parameters.handle_match_ambiguity")
        if not normalized_parameters["all_address_free_parameters_exact"]:
            blockers.add("parameters.unresolved_after_handle_normalization")
    elif not parameters["all_raw_parameters_exact"]:
        blockers.add("parameters.unresolved_raw_differences")

    blockers_list = sorted(blockers)
    selected_graph_repeat = not blockers_list
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "builder_sha256": digest_file(Path(__file__).resolve()),
        "inputs": {
            "trace_paths_distinct": True,
            "dump_roots_distinct": True,
            "fresh_process_precondition": "caller_supplied_disjoint_fresh_process_jobs",
            "run_a": run_a.input_receipt,
            "run_b": run_b.input_receipt,
        },
        "phase_sequence": {
            "run_a": run_a.phases,
            "run_b": run_b.phases,
            "repeat_equal": run_a.phases == run_b.phases,
        },
        "phase_scoped_counts": {
            "run_a": run_a.phase_counts,
            "run_b": run_b.phase_counts,
            "root_a_sha256": phase_root_a,
            "root_b_sha256": phase_root_b,
            "repeat_equal": phase_root_a == phase_root_b,
        },
        "module_blobs": {
            "run_a": run_a.modules,
            "run_b": run_b.modules,
            "root_a_sha256": module_root_a,
            "root_b_sha256": module_root_b,
            "repeat_equal": module_root_a == module_root_b,
        },
        "function_encounters": {
            "run_a": run_a.functions,
            "run_b": run_b.functions,
            "enumerated_names_a": run_a.enumerated_names,
            "enumerated_names_b": run_b.enumerated_names,
            "root_a_sha256": function_root_a,
            "root_b_sha256": function_root_b,
            "repeat_equal": function_root_a == function_root_b,
        },
        "selected_launches": {
            "run_a": run_a.launches,
            "run_b": run_b.launches,
            "structural_projection_a": [launch_structure(launch) for launch in run_a.launches],
            "structural_projection_b": [launch_structure(launch) for launch in run_b.launches],
        },
        "structural_graph": {
            "root_a_sha256": structural_root_a,
            "root_b_sha256": structural_root_b,
            "repeat_equal": structural_root_a == structural_root_b,
            "address_free": True,
        },
        "raw_parameters": parameters,
        "handle_normalized_parameters": normalized_parameters,
        "module_bound_padding_parameters": padding_parameters,
        "resource_handle_provenance": {
            "enabled": run_a.volatile is not None and run_b.volatile is not None,
            "run_a": (
                {
                    "input": run_a.volatile.input_receipt,
                    "resources": run_a.volatile.resources,
                    "source_ambiguities": run_a.volatile.source_ambiguities,
                    "uav_info": run_a.volatile.uav_info,
                }
                if run_a.volatile is not None
                else None
            ),
            "run_b": (
                {
                    "input": run_b.volatile.input_receipt,
                    "resources": run_b.volatile.resources,
                    "source_ambiguities": run_b.volatile.source_ambiguities,
                    "uav_info": run_b.volatile.uav_info,
                }
                if run_b.volatile is not None
                else None
            ),
            "resource_root_a_sha256": volatile_resource_root_a,
            "resource_root_b_sha256": volatile_resource_root_b,
            "resource_repeat_equal": volatile_resource_root_a
            == volatile_resource_root_b,
            "uav_info_root_a_sha256": uav_info_root_a,
            "uav_info_root_b_sha256": uav_info_root_b,
            "uav_info_repeat_equal": uav_info_root_a == uav_info_root_b,
            "uav_zero_span_no_provenance_count_a": (
                sum(
                    record["provenance_result"] == "zero_span_no_provenance"
                    for record in run_a.volatile.uav_info
                )
                if run_a.volatile is not None
                else 0
            ),
            "uav_zero_span_no_provenance_count_b": (
                sum(
                    record["provenance_result"] == "zero_span_no_provenance"
                    for record in run_b.volatile.uav_info
                )
                if run_b.volatile is not None
                else 0
            ),
            "capture_uav_info_zero_spans_are_provenance": False,
        },
        "resource_call_counts": {
            "run_a": run_a.resources,
            "run_b": run_b.resources,
            "root_a_sha256": resource_root_a,
            "root_b_sha256": resource_root_b,
            "repeat_equal": resource_root_a == resource_root_b,
        },
        "unknown_interface_ids": {
            "run_a": run_a.unknown_interfaces,
            "run_b": run_b.unknown_interfaces,
            "root_a_sha256": unknown_root_a,
            "root_b_sha256": unknown_root_b,
            "repeat_equal": unknown_root_a == unknown_root_b,
        },
        "non_model_interface_dependencies": {
            "run_a": run_a.non_model_interfaces,
            "run_b": run_b.non_model_interfaces,
            "root_a_sha256": non_model_root_a,
            "root_b_sha256": non_model_root_b,
            "repeat_equal": non_model_root_a == non_model_root_b,
        },
        "opaque_interface_sentinel": {
            "run_a": run_a.opaque_sentinel,
            "run_b": run_b.opaque_sentinel,
            "root_a_sha256": opaque_sentinel_root_a,
            "root_b_sha256": opaque_sentinel_root_b,
            "repeat_equal": opaque_sentinel_root_a == opaque_sentinel_root_b,
        },
        "failed_real_call_census": {
            "run_a": run_a.failed_calls,
            "run_b": run_b.failed_calls,
            "total": sum(record["count"] for record in run_a.failed_calls + run_b.failed_calls),
            "root_a_sha256": failed_root_a,
            "root_b_sha256": failed_root_b,
            "repeat_equal": failed_root_a == failed_root_b,
        },
        "capture_failure_census": {
            "run_a": run_a.capture_failures,
            "run_b": run_b.capture_failures,
            "total": sum(record["count"] for record in run_a.capture_failures + run_b.capture_failures),
            "root_a_sha256": capture_failure_root_a,
            "root_b_sha256": capture_failure_root_b,
            "repeat_equal": capture_failure_root_a == capture_failure_root_b,
        },
        "blockers": blockers_list,
        "claims": {
            "address_free": True,
            "phase_markers_present": bool(run_a.phases) and bool(run_b.phases),
            "structural_graph_repeat": structural_root_a == structural_root_b,
            "raw_parameters_exact": parameters["all_raw_parameters_exact"],
            "address_free_parameters_exact": normalized_parameters[
                "all_address_free_parameters_exact"
            ]
            or padding_parameters[
                "all_module_bound_canonical_parameters_exact"
            ],
            "parameter_differences_provenanced": (
                padding_parameters[
                    "all_module_bound_canonical_parameters_exact"
                ]
                if padding_parameters["enabled"]
                else normalized_parameters["all_address_free_parameters_exact"]
                if normalized_parameters["enabled"]
                else parameters["all_raw_parameters_exact"]
            ),
            "selected_graph_repeat_accepted": selected_graph_repeat,
            "intake_binding_eligible": selected_graph_repeat,
            "intake_acceptance": False,
        },
        "receipt_root_sha256": None,
    }
    receipt["receipt_root_sha256"] = rooted(receipt, "receipt_root_sha256")
    return receipt, 0 if selected_graph_repeat else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--dumps-a", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--dumps-b", type=Path, required=True)
    parser.add_argument("--volatile-a", type=Path)
    parser.add_argument("--volatile-b", type=Path)
    parser.add_argument("--module21-ptx", type=Path)
    parser.add_argument("--module21-ptx-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-receipt", action="store_true")
    args = parser.parse_args()
    try:
        receipt, status = build_repeat(
            args.trace_a,
            args.dumps_a,
            args.trace_b,
            args.dumps_b,
            args.volatile_a,
            args.volatile_b,
            args.module21_ptx,
            args.module21_ptx_receipt,
        )
    except (OSError, json.JSONDecodeError, RepeatError) as error:
        print(f"FG NvAPI repeat build failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        print(f"FG NvAPI repeat output already exists: {args.output}", file=sys.stderr)
        return 2
    if args.print_receipt:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "receipt_root_sha256": receipt["receipt_root_sha256"],
                    "structural_graph_repeat": receipt["claims"][
                        "structural_graph_repeat"
                    ],
                    "exact_raw_parameter_equality_count": receipt[
                        "raw_parameters"
                    ]["exact_raw_parameter_equality_count"],
                    "launch_comparison_count": receipt["raw_parameters"][
                        "launch_comparison_count"
                    ],
                    "address_free_exact_count": receipt[
                        "module_bound_padding_parameters"
                    ]["exact_after_padding_proof_count"],
                    "blockers": receipt["blockers"],
                },
                sort_keys=True,
            )
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
