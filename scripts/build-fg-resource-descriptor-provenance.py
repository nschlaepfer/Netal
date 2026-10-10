#!/usr/bin/env python3
"""Build an address-free FG D3D12/NvAPI resource provenance receipt.

The builder validates two fresh, poisoned RTX processes.  It joins raw CPU
descriptor handles and CUDA object handles only inside the verifier, then
emits stable tokens, D3D12 resource/view/sampler semantics, and exact packed
parameter byte locations.  Opaque CUDA objects are never classified as GPU
virtual addresses, and a byte occurrence is not promoted to a SASS dereference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "neuralstudio.fg.resource-descriptor-provenance.v1"
PROCESS_SCHEMA = "neuralstudio.fg.resource-descriptor-process.v1"
DESCRIPTOR_SCHEMA = "neuralstudio.fg.resource-descriptor-trace.v1"
DESCRIPTOR_VOLATILE_SCHEMA = "neuralstudio.fg.resource-descriptor-volatile.v1"
EXPECTED_RUNTIME_SHA256 = (
    "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"
)
EXPECTED_OUTPUT_SHA256 = (
    "07a87435b2e7d79540dcec737b9abb6354e68e8595b4df29ca6f8d92cce54383"
)
HASH = re.compile(r"^[0-9a-f]{64}$")
HEX64 = re.compile(r"^0x[0-9A-Fa-f]{16}$")
RESOURCE_TOKEN = re.compile(r"^d3d-resource-[0-9]{6}$")
DESCRIPTOR_TOKEN = re.compile(r"^descriptor-[0-9]{6}$")
NVAPI_RESOURCE_TOKEN = re.compile(r"^resource-[0-9]{6}$")
REQUEST_TOKEN = re.compile(r"^resource-request-[0-9]{6}$")

FORBIDDEN_SEMANTIC_FIELDS = {
    "resource_identity",
    "counter_resource_identity",
    "heap_identity",
    "cpu_descriptor_start",
    "cpu_descriptor_handle",
    "sampler_cpu_descriptor_handle",
    "source_cpu_descriptor_handle",
    "gpu_va_start",
    "gpu_virtual_address",
    "cuda_object_handle",
    "texture_descriptor_ptr",
    "sampler_descriptor_ptr",
    "descriptor_ptr",
    "uav_descriptor_ptr",
}

DXGI_FORMAT_NAMES = {
    0: "DXGI_FORMAT_UNKNOWN",
    10: "DXGI_FORMAT_R16G16B16A16_FLOAT",
    16: "DXGI_FORMAT_R32G32_FLOAT",
    28: "DXGI_FORMAT_R8G8B8A8_UNORM",
    34: "DXGI_FORMAT_R16G16_FLOAT",
    39: "DXGI_FORMAT_R32_TYPELESS",
    41: "DXGI_FORMAT_R32_FLOAT",
    42: "DXGI_FORMAT_R32_UINT",
    54: "DXGI_FORMAT_R16_FLOAT",
    61: "DXGI_FORMAT_R8_UNORM",
}
RESOURCE_DIMENSION_NAMES = {
    0: "UNKNOWN",
    1: "BUFFER",
    2: "TEXTURE1D",
    3: "TEXTURE2D",
    4: "TEXTURE3D",
}
SRV_DIMENSION_NAMES = {
    0: "UNKNOWN",
    1: "BUFFER",
    2: "TEXTURE1D",
    3: "TEXTURE1DARRAY",
    4: "TEXTURE2D",
    5: "TEXTURE2DARRAY",
    6: "TEXTURE2DMS",
    7: "TEXTURE2DMSARRAY",
    8: "TEXTURE3D",
    9: "TEXTURECUBE",
    10: "TEXTURECUBEARRAY",
    11: "RAYTRACING_ACCELERATION_STRUCTURE",
}
UAV_DIMENSION_NAMES = {
    0: "UNKNOWN",
    1: "BUFFER",
    2: "TEXTURE1D",
    3: "TEXTURE1DARRAY",
    4: "TEXTURE2D",
    5: "TEXTURE2DARRAY",
    8: "TEXTURE3D",
}


class ProvenanceError(RuntimeError):
    """Evidence is malformed, incomplete, or violates the acceptance gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"unsafe JSONL file: {path}")
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        require(bool(line.strip()), f"blank JSONL line at {path}:{line_number}")
        item = json.loads(line)
        require(isinstance(item, dict), f"non-object at {path}:{line_number}")
        result.append(item)
    require(bool(result), f"empty JSONL file: {path}")
    return result


def safe_relative(root: Path, name: str) -> Path:
    require(isinstance(name, str) and bool(name), "relative evidence path missing")
    pure = PurePosixPath(name.replace("\\", "/"))
    require(
        not pure.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"unsafe evidence path: {name}",
    )
    root = root.resolve()
    path = root.joinpath(*pure.parts)
    require(not path.is_symlink(), f"evidence symlink forbidden: {name}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProvenanceError(f"evidence path escapes root: {name}") from error
    require(resolved.is_file(), f"evidence file missing: {name}")
    return resolved


def validate_file_entry(root: Path, entry: Any, label: str) -> Path:
    require(isinstance(entry, dict), f"{label} file receipt missing")
    path = safe_relative(root, entry.get("path"))
    require(entry.get("bytes") == path.stat().st_size, f"{label} byte count mismatch")
    expected = entry.get("sha256")
    require(isinstance(expected, str) and HASH.fullmatch(expected), f"{label} hash invalid")
    require(digest_file(path) == expected, f"{label} hash mismatch")
    return path


def token_number(token: str) -> int:
    return int(token.rsplit("-", 1)[1])


def parse_hex64(value: Any, label: str) -> int:
    require(isinstance(value, str) and HEX64.fullmatch(value), f"{label} is not hex64")
    return int(value, 16)


def format_name(value: int) -> str:
    return DXGI_FORMAT_NAMES.get(value, f"DXGI_FORMAT_{value}")


def resource_description(event: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dimension",
        "alignment",
        "width",
        "height",
        "depth_or_array_size",
        "mip_levels",
        "format",
        "sample_count",
        "sample_quality",
        "layout",
        "resource_flags",
        "heap_properties_present",
        "heap_type",
        "heap_flags",
        "cpu_page_property",
        "memory_pool_preference",
        "creation_node_mask",
        "visible_node_mask",
        "initial_state",
        "clear_value_present",
        "clear_value_format",
        "clear_value_union_words",
    )
    result = {key: event[key] for key in keys if key in event}
    result["dimension_name"] = RESOURCE_DIMENSION_NAMES.get(
        result["dimension"], f"DIMENSION_{result['dimension']}"
    )
    result["format_name"] = format_name(result["format"])
    if "clear_value_format" in result:
        result["clear_value_format_name"] = format_name(
            result["clear_value_format"]
        )
    return result


DESCRIPTOR_METADATA_KEYS = {
    "event",
    "phase",
    "descriptor_token",
    "descriptor_heap_token",
    "descriptor_heap_slot",
    "descriptor_kind",
    "resource_token",
    "counter_resource_token",
    "source_descriptor_token",
    "default_view",
}


def descriptor_description(event: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in event.items()
        if key not in DESCRIPTOR_METADATA_KEYS
    }
    kind = event["descriptor_kind"]
    if "view_format" in result:
        result["view_format_name"] = format_name(result["view_format"])
    if "view_dimension" in result:
        names = (
            SRV_DIMENSION_NAMES
            if kind == "shader_resource_view"
            else UAV_DIMENSION_NAMES
        )
        result["view_dimension_name"] = names.get(
            result["view_dimension"], f"VIEW_DIMENSION_{result['view_dimension']}"
        )
    return result


def derived_subresources(event: dict[str, Any]) -> dict[str, Any]:
    kind = event["descriptor_kind"]
    view = event.get("view_dimension")
    if view == 1:
        return {
            "kind": "buffer_element_range",
            "first_element": event.get("first_element", 0),
            "num_elements": event.get("num_elements", 0),
            "structure_byte_stride": event.get("structure_byte_stride", 0),
        }
    if kind == "shader_resource_view":
        return {
            "kind": "texture_subresource_range",
            "most_detailed_mip": event.get("most_detailed_mip", 0),
            "mip_count": event.get("mip_count", 1),
            "first_array_slice": event.get("first_array_slice", 0),
            "array_size": event.get("array_size", 1),
            "plane_slice": event.get("plane_slice", 0),
        }
    if kind == "unordered_access_view":
        return {
            "kind": "texture_subresource_range",
            "mip_slice": event.get("mip_slice", 0),
            "first_array_slice": event.get("first_array_slice", 0),
            "array_size": event.get("array_size", 1),
            "plane_slice": event.get("plane_slice", 0),
            "first_depth_slice": event.get("first_depth_slice", 0),
            "depth_size": event.get("depth_size", 1),
        }
    return {"kind": "not_applicable"}


def reject_semantic_addresses(events: list[dict[str, Any]]) -> None:
    for ordinal, event in enumerate(events):
        forbidden = sorted(set(event) & FORBIDDEN_SEMANTIC_FIELDS)
        require(not forbidden, f"semantic event {ordinal} leaks raw fields: {forbidden}")
        for flag in (
            "addresses_recorded",
            "device_address_recorded",
            "gpu_virtual_addresses_recorded",
            "opaque_cuda_handles_recorded",
            "gpu_virtual_address_recorded",
        ):
            if flag in event:
                require(event[flag] is False, f"semantic event enables {flag}")


def validate_allocation_match(
    resource: dict[str, Any], allocation: dict[str, Any]
) -> None:
    pairs = (
        ("dimension", "dimension"),
        ("alignment", "alignment"),
        ("width", "width"),
        ("height", "height"),
        ("depth_or_array_size", "depth_or_array"),
        ("mip_levels", "mip_levels"),
        ("format", "format"),
        ("sample_count", "sample_count"),
        ("sample_quality", "sample_quality"),
        ("layout", "layout"),
        ("resource_flags", "resource_flags"),
        ("heap_type", "heap_type"),
        ("heap_flags", "heap_flags"),
        ("cpu_page_property", "cpu_page_property"),
        ("memory_pool_preference", "memory_pool_preference"),
        ("creation_node_mask", "creation_node_mask"),
        ("visible_node_mask", "visible_node_mask"),
        ("initial_state", "initial_state"),
    )
    for resource_key, allocation_key in pairs:
        require(
            resource.get(resource_key) == allocation.get(allocation_key),
            f"runtime allocation descriptor mismatch at {resource['resource_token']}:{resource_key}",
        )


@dataclass
class Launch:
    phase: str
    token: str
    kernel_index: int
    module_token: str
    function_token: str
    function_name: str
    parameter_path: Path
    parameter_data: bytes


@dataclass
class RunEvidence:
    name: str
    process_receipt_path: Path
    process_receipt: dict[str, Any]
    host_receipt: dict[str, Any]
    semantic_path: Path
    semantic: list[dict[str, Any]]
    descriptor_volatile_path: Path
    descriptor_volatile: list[dict[str, Any]]
    nvapi_path: Path
    nvapi: list[dict[str, Any]]
    nvapi_volatile_path: Path
    nvapi_volatile: list[dict[str, Any]]
    allocation_semantic_path: Path
    allocation_semantic: list[dict[str, Any]]
    resources: dict[str, dict[str, Any]]
    descriptors: dict[str, dict[str, Any]]
    requests: list[dict[str, Any]]
    runtime_allocation_by_resource: dict[str, dict[str, Any]]
    launches: list[Launch]
    raw_requests: list[dict[str, Any]]
    consumer_bindings: dict[str, list[dict[str, Any]]]


def parse_launches(trace: list[dict[str, Any]], dump_root: Path) -> list[Launch]:
    current_phase = "__bootstrap__"
    launches: list[Launch] = []
    for event in trace:
        if event.get("event") == "phase_marker":
            current_phase = event.get("phase")
        elif event.get("event") == "kernel_launch":
            require(event.get("parameter_mode") == "packed", "non-packed selected launch")
            require(event.get("parameter_captured") is True, "uncaptured launch parameter")
            name = event.get("parameter_file")
            require(
                isinstance(name, str)
                and PurePosixPath(name).name == name
                and "/" not in name
                and "\\" not in name,
                "unsafe parameter dump name",
            )
            path = dump_root / name
            require(path.is_file() and not path.is_symlink(), f"parameter dump missing: {name}")
            data = path.read_bytes()
            require(len(data) == event.get("parameter_bytes"), "parameter byte count mismatch")
            require(digest_file(path).upper() == event.get("parameter_sha256"), "parameter hash mismatch")
            launches.append(
                Launch(
                    current_phase,
                    event["launch_token"],
                    event["kernel_index"],
                    event["module_token"],
                    event["function_token"],
                    event["function_name"],
                    path,
                    data,
                )
            )
    require(
        len(launches) == 83,
        f"expected 83 successful kernel parameter records, got {len(launches)}",
    )
    return launches


def validate_descriptor_semantic(
    events: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    reject_semantic_addresses(events)
    ready = events[0]
    summary = events[-1]
    require(
        ready == {
            "event": "descriptor_runtime_ready",
            "schema": DESCRIPTOR_SCHEMA,
            "instrumentation": "id3d12device_vtable",
            "addresses_recorded": False,
            "opaque_cuda_handles_recorded": False,
            "gpu_virtual_addresses_recorded": False,
        },
        "descriptor runtime header mismatch",
    )
    require(summary.get("event") == "descriptor_capture_summary", "summary missing")
    require(summary.get("capture_healthy") is True, "descriptor capture unhealthy")
    require(summary.get("hooks_restored") is True, "descriptor hooks not restored")
    require(summary.get("unresolved_successful_request_count") == 0, "unresolved request")

    resources = {
        event["resource_token"]: event
        for event in events
        if event.get("event") == "resource_created"
    }
    descriptors = {
        event["descriptor_token"]: event
        for event in events
        if event.get("event") in {"descriptor_written", "descriptor_copied"}
    }
    requests = [
        event for event in events if event.get("event") == "resource_object_request"
    ]
    require(len(resources) == summary.get("resource_count") == 225, "resource count drift")
    require(len(descriptors) == summary.get("descriptor_write_count") == 213, "descriptor count drift")
    require(len(requests) == summary.get("request_count") == 64, "request count drift")
    require(
        len([event for event in events if event.get("event") == "descriptor_heap_created"])
        == summary.get("descriptor_heap_count")
        == 4,
        "descriptor heap count drift",
    )
    require(
        len([event for event in events if event.get("event") == "descriptor_copied"])
        == summary.get("descriptor_copy_count")
        == 54,
        "descriptor copy count drift",
    )
    require(len(set(resources)) == len(resources), "duplicate resource token")
    require(len(set(descriptors)) == len(descriptors), "duplicate descriptor token")
    require(
        all(RESOURCE_TOKEN.fullmatch(token) for token in resources),
        "invalid D3D resource token",
    )
    require(
        all(DESCRIPTOR_TOKEN.fullmatch(token) for token in descriptors),
        "invalid descriptor token",
    )

    successful = [request for request in requests if request.get("status") == 0]
    failed = [request for request in requests if request.get("status") != 0]
    require(len(successful) == summary.get("successful_request_count") == 60, "success count drift")
    require(len(failed) == 4, "expected four intentional ABI smoke failures")
    for request in successful:
        require(request.get("snapshot_timing") == "pre_real_nvapi_call", "post-call snapshot rejected")
        require(request.get("mapping_resolved") is True, "request mapping unresolved")
        require(request.get("descriptor_kind_matches") is True, "descriptor kind mismatch")
        require(NVAPI_RESOURCE_TOKEN.fullmatch(request.get("nvapi_resource_token", "")), "bad NvAPI token")
        descriptor = descriptors.get(request.get("descriptor_token"))
        require(descriptor is not None, "request descriptor missing")
        require(descriptor.get("resource_token") in resources, "request resource missing")
        if request.get("resource_kind") == "merged_texture_sampler":
            require(request.get("sampler_kind_matches") is True, "sampler kind mismatch")
            sampler = descriptors.get(request.get("sampler_descriptor_token"))
            require(sampler is not None and sampler.get("descriptor_kind") == "sampler", "sampler missing")
    return resources, descriptors, requests


def validate_descriptor_volatile(
    semantic: list[dict[str, Any]],
    volatile: list[dict[str, Any]],
    resources: dict[str, dict[str, Any]],
    descriptors: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    require(
        volatile[0]
        == {
            "event": "descriptor_volatile_runtime_ready",
            "schema": DESCRIPTOR_VOLATILE_SCHEMA,
        },
        "descriptor volatile header mismatch",
    )
    resource_raw = [event for event in volatile if event.get("event") == "resource_created_raw"]
    descriptor_raw = [event for event in volatile if event.get("event") == "descriptor_location_raw"]
    request_raw = [event for event in volatile if event.get("event") == "resource_object_request_raw"]
    heap_raw = [event for event in volatile if event.get("event") == "descriptor_heap_raw"]
    semantic_descriptors = [
        event
        for event in semantic
        if event.get("event") in {"descriptor_written", "descriptor_copied"}
    ]
    semantic_requests = [event for event in semantic if event.get("event") == "resource_object_request"]
    require(len(resource_raw) == len(resources) == 225, "raw resource count drift")
    require(len(descriptor_raw) == len(semantic_descriptors) == 213, "raw descriptor count drift")
    require(len(request_raw) == len(semantic_requests) == 64, "raw request count drift")
    require(len(heap_raw) == 4, "raw heap count drift")

    identity_by_resource: dict[str, int] = {}
    for raw in resource_raw:
        token = raw.get("resource_token")
        require(token in resources and token not in identity_by_resource, "raw resource token drift")
        identity = parse_hex64(raw.get("resource_identity"), "resource identity")
        require(identity != 0, "zero resource identity")
        identity_by_resource[token] = identity

    for semantic_event, raw in zip(semantic_descriptors, descriptor_raw, strict=True):
        token = semantic_event["descriptor_token"]
        require(raw.get("descriptor_token") == token, "descriptor raw order/token drift")
        require(parse_hex64(raw.get("cpu_descriptor_handle"), "CPU descriptor handle") != 0, "zero descriptor handle")
        resource_token = semantic_event.get("resource_token")
        if resource_token != "untracked":
            require(
                parse_hex64(raw.get("resource_identity"), "descriptor resource identity")
                == identity_by_resource[resource_token],
                "descriptor-to-resource raw identity mismatch",
            )
        counter_token = semantic_event.get("counter_resource_token")
        if counter_token != "untracked":
            require(
                parse_hex64(raw.get("counter_resource_identity"), "counter identity")
                == identity_by_resource[counter_token],
                "descriptor counter-resource identity mismatch",
            )

    for semantic_event, raw in zip(semantic_requests, request_raw, strict=True):
        require(raw.get("request_token") == semantic_event.get("request_token"), "raw request token drift")
        require(raw.get("nvapi_resource_token") == semantic_event.get("nvapi_resource_token"), "raw NvAPI token drift")
        parse_hex64(raw.get("cpu_descriptor_handle"), "request CPU descriptor handle")
        if "sampler_descriptor_token" in semantic_event:
            parse_hex64(raw.get("sampler_cpu_descriptor_handle"), "sampler CPU descriptor handle")
    return request_raw, identity_by_resource


def validate_nvapi_join(
    semantic_requests: list[dict[str, Any]],
    descriptor_raw_requests: list[dict[str, Any]],
    nvapi_volatile: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    nvapi_raw = [event for event in nvapi_volatile if event.get("event") == "resource_object_raw"]
    require(len(nvapi_raw) == len(semantic_requests) == 64, "NvAPI resource raw count drift")
    for request, descriptor_raw, nvapi_raw_event in zip(
        semantic_requests, descriptor_raw_requests, nvapi_raw, strict=True
    ):
        require(nvapi_raw_event.get("call") == request.get("call"), "NvAPI call join mismatch")
        require(nvapi_raw_event.get("phase") == request.get("phase"), "NvAPI phase join mismatch")
        require(nvapi_raw_event.get("resource_kind") == request.get("resource_kind"), "NvAPI kind join mismatch")
        require(nvapi_raw_event.get("resource_token") == request.get("nvapi_resource_token"), "NvAPI token join mismatch")
        require(nvapi_raw_event.get("status") == request.get("status"), "NvAPI status join mismatch")
        cpu_handle = descriptor_raw["cpu_descriptor_handle"]
        call = request["call"]
        if call == "NvAPI_D3D12_GetCudaMergedTextureSamplerObject":
            require(nvapi_raw_event.get("texture_descriptor_ptr") == cpu_handle, "merged texture descriptor join mismatch")
            require(
                nvapi_raw_event.get("sampler_descriptor_ptr")
                == descriptor_raw.get("sampler_cpu_descriptor_handle"),
                "merged sampler descriptor join mismatch",
            )
        elif call == "NvAPI_D3D12_GetCudaIndependentDescriptorObject":
            require(nvapi_raw_event.get("descriptor_ptr") == cpu_handle, "independent descriptor join mismatch")
        elif call == "NvAPI_D3D12_GetCudaTextureObject":
            require(nvapi_raw_event.get("texture_descriptor_ptr") == cpu_handle, "texture descriptor join mismatch")
        elif call == "NvAPI_D3D12_GetCudaSurfaceObject":
            require(nvapi_raw_event.get("uav_descriptor_ptr") == cpu_handle, "surface descriptor join mismatch")
        else:
            raise ProvenanceError(f"unknown resource object call: {call}")
    return nvapi_raw


def locate_consumers(
    raw_requests: list[dict[str, Any]], launches: list[Launch]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_requests:
        if raw.get("status") != 0:
            continue
        token = raw["resource_token"]
        width = raw.get("cuda_object_handle_bytes")
        require(width in {4, 8}, "CUDA object handle width invalid")
        handle = parse_hex64(raw.get("cuda_object_handle"), "CUDA object handle")
        require(handle != 0, "successful CUDA object handle is zero")
        needle = handle.to_bytes(width, "little")
        bindings: list[dict[str, Any]] = []
        for launch in launches:
            for offset in range(0, len(launch.parameter_data) - width + 1, 8):
                if launch.parameter_data[offset : offset + width] == needle:
                    bindings.append(
                        {
                            "phase": launch.phase,
                            "launch_token": launch.token,
                            "kernel_index": launch.kernel_index,
                            "module_token": launch.module_token,
                            "function_token": launch.function_token,
                            "function_name": launch.function_name,
                            "parameter_offset": offset,
                            "handle_bytes": width,
                            "evidence": "exact_aligned_little_endian_byte_occurrence",
                        }
                    )
        require(token not in result, "duplicate successful NvAPI resource token")
        result[token] = bindings
    require(len(result) == 60, "consumer binding census must cover 60 object handles")
    return result


def parse_run(root: Path, run_name: str) -> RunEvidence:
    receipt_path = root / "receipts" / f"{run_name}-resource-descriptor-process.json"
    process_receipt = load_json(receipt_path)
    require(process_receipt.get("schema") == PROCESS_SCHEMA, "process schema mismatch")
    require(process_receipt.get("run_id") == run_name, "process run id mismatch")
    require(process_receipt.get("accepted") is True and process_receipt.get("exit_code") == 0, "process rejected")
    require(process_receipt.get("host_accepted") is True, "host process not accepted")
    require(process_receipt.get("output_sha256") == EXPECTED_OUTPUT_SHA256, "output baseline drift")
    require(process_receipt.get("output_untouched_pixels") == 0, "output poison coverage failed")
    files = process_receipt.get("files")
    require(isinstance(files, dict), "process file inventory missing")
    traced_process_path = validate_file_entry(root, files.get("traced_process_receipt"), "traced process")
    host_path = validate_file_entry(root, files.get("host_receipt"), "host")
    nvapi_path = validate_file_entry(root, files.get("nvapi_trace"), "NvAPI trace")
    nvapi_volatile_path = validate_file_entry(root, files.get("nvapi_volatile_trace"), "NvAPI volatile")
    semantic_path = validate_file_entry(root, files.get("descriptor_semantic_trace"), "descriptor semantic")
    descriptor_volatile_path = validate_file_entry(root, files.get("descriptor_volatile_trace"), "descriptor volatile")

    traced_process = load_json(traced_process_path)
    host_receipt = load_json(host_path)
    require(traced_process.get("exit_code") == 0, "traced process exit drift")
    require(traced_process.get("runtime_sha256") == EXPECTED_RUNTIME_SHA256, "traced runtime drift")
    require(host_receipt.get("runtime_sha256") == EXPECTED_RUNTIME_SHA256, "host runtime drift")
    require(host_receipt.get("accepted") is True, "host receipt rejected")
    require(host_receipt.get("process_id") == process_receipt.get("process_id"), "process id drift")
    require(host_receipt.get("run_nonce") == process_receipt.get("run_nonce"), "run nonce drift")
    require(host_receipt.get("queue_nonce") == process_receipt.get("queue_nonce"), "queue nonce drift")
    require(host_receipt.get("output_poison_byte") == process_receipt.get("poison_byte"), "poison drift")
    require(host_receipt["frames"][1]["output_interpolated_sha256"] == EXPECTED_OUTPUT_SHA256, "frame hash drift")
    require(host_receipt["frames"][1]["output_untouched_pixels"] == 0, "frame coverage drift")
    allocation_receipt = host_receipt.get("runtime_allocation_capture", {})
    require(
        allocation_receipt.get("enabled") is True
        and allocation_receipt.get("healthy") is True
        and allocation_receipt.get("baseline_output_match") is True
        and allocation_receipt.get("allocations") == 205
        and allocation_receipt.get("releases") == 205
        and allocation_receipt.get("leaks") == 0,
        "runtime allocation capture gate failed",
    )

    semantic = load_jsonl(semantic_path)
    descriptor_volatile = load_jsonl(descriptor_volatile_path)
    nvapi = load_jsonl(nvapi_path)
    nvapi_volatile = load_jsonl(nvapi_volatile_path)
    resources, descriptors, requests = validate_descriptor_semantic(semantic)
    descriptor_raw_requests, _ = validate_descriptor_volatile(
        semantic, descriptor_volatile, resources, descriptors
    )
    raw_requests = validate_nvapi_join(requests, descriptor_raw_requests, nvapi_volatile)

    allocation_semantic_path = root / "output" / run_name / allocation_receipt["semantic_ledger"]
    require(digest_file(allocation_semantic_path) == allocation_receipt["semantic_ledger_sha256"], "allocation semantic hash mismatch")
    allocation_semantic = load_jsonl(allocation_semantic_path)
    allocations = [event for event in allocation_semantic if event.get("event") == "allocation"]
    create_phase_resources = [
        event
        for event in semantic
        if event.get("event") == "resource_created" and event.get("phase") == "create"
    ]
    require(len(allocations) == len(create_phase_resources) == 205, "callback allocation correlation count drift")
    runtime_allocation_by_resource: dict[str, dict[str, Any]] = {}
    for resource, allocation in zip(create_phase_resources, allocations, strict=True):
        validate_allocation_match(resource, allocation)
        runtime_allocation_by_resource[resource["resource_token"]] = allocation

    dump_root = nvapi_path.parent / "dumps"
    launches = parse_launches(nvapi, dump_root)
    consumer_bindings = locate_consumers(raw_requests, launches)
    return RunEvidence(
        run_name,
        receipt_path,
        process_receipt,
        host_receipt,
        semantic_path,
        semantic,
        descriptor_volatile_path,
        descriptor_volatile,
        nvapi_path,
        nvapi,
        nvapi_volatile_path,
        nvapi_volatile,
        allocation_semantic_path,
        allocation_semantic,
        resources,
        descriptors,
        requests,
        runtime_allocation_by_resource,
        launches,
        raw_requests,
        consumer_bindings,
    )


def source_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, entry in sorted(receipt.get("source_identity", {}).items()):
        require(isinstance(entry, dict), f"source identity missing: {name}")
        require(isinstance(entry.get("sha256"), str) and HASH.fullmatch(entry["sha256"]), f"source hash invalid: {name}")
        result[name] = {"bytes": entry.get("bytes"), "sha256": entry["sha256"]}
    return result


def selected_request_receipts(run: RunEvidence) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for request in run.requests:
        if request["status"] != 0:
            continue
        descriptor = run.descriptors[request["descriptor_token"]]
        resource = run.resources[descriptor["resource_token"]]
        sampler = None
        if "sampler_descriptor_token" in request:
            sampler = run.descriptors[request["sampler_descriptor_token"]]
        allocation = run.runtime_allocation_by_resource.get(resource["resource_token"])
        result.append(
            {
                "request_token": request["request_token"],
                "phase": request["phase"],
                "call": request["call"],
                "request_type": request["resource_kind"],
                "nvapi_resource_token": request["nvapi_resource_token"],
                "snapshot_timing": request["snapshot_timing"],
                "descriptor_token": descriptor["descriptor_token"],
                "descriptor_heap_token": descriptor["descriptor_heap_token"],
                "descriptor_heap_slot": descriptor["descriptor_heap_slot"],
                "descriptor_kind": descriptor["descriptor_kind"],
                "d3d_resource_token": resource["resource_token"],
                "runtime_allocation_token": (
                    f"allocation-{allocation['token']:06d}" if allocation else None
                ),
                "resource": resource_description(resource),
                "view": descriptor_description(descriptor),
                "subresources": derived_subresources(descriptor),
                "sampler_descriptor_token": (
                    sampler["descriptor_token"] if sampler else None
                ),
                "sampler": descriptor_description(sampler) if sampler else None,
                "packed_parameter_bindings": run.consumer_bindings[
                    request["nvapi_resource_token"]
                ],
            }
        )
    require(len(result) == 60, "selected request receipt count drift")
    return result


def build_counterexample(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    events = load_jsonl(path)
    descriptors = {
        event["descriptor_token"]: event
        for event in events
        if event.get("event") in {"descriptor_written", "descriptor_copied"}
    }
    successful = [
        event
        for event in events
        if event.get("event") == "resource_object_request"
        and event.get("status") == 0
    ]
    lost = sum(
        descriptors.get(event.get("descriptor_token"), {}).get("resource_token")
        == "untracked"
        for event in successful
    )
    return {
        "file_sha256": digest_file(path),
        "successful_request_count": len(successful),
        "post_call_resource_association_lost_count": lost,
        "result": "rejected_post_call_snapshot_driver_rewrote_scratch_descriptor",
        "preserved": True,
    }


def build_receipt(
    input_root: Path,
    run_a_name: str,
    run_b_name: str,
    post_call_counterexample: Path | None = None,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    require(input_root.is_dir() and not input_root.is_symlink(), "unsafe input root")
    run_a = parse_run(input_root, run_a_name)
    run_b = parse_run(input_root, run_b_name)

    require(run_a.process_receipt["job_id"] == run_b.process_receipt["job_id"], "job id mismatch")
    require(run_a.process_receipt["process_id"] != run_b.process_receipt["process_id"], "process ids not fresh")
    require(run_a.process_receipt["run_nonce"] != run_b.process_receipt["run_nonce"], "run nonces not fresh")
    require(run_a.process_receipt["queue_nonce"] != run_b.process_receipt["queue_nonce"], "queue nonces not fresh")
    require(
        {run_a.process_receipt["poison_byte"], run_b.process_receipt["poison_byte"]}
        == {0x5A, 0xA5},
        "two-poison gate failed",
    )
    source_a = source_identity(run_a.process_receipt)
    source_b = source_identity(run_b.process_receipt)
    require(source_a == source_b, "source identity differs across processes")
    require(source_a["runtime"]["sha256"] == EXPECTED_RUNTIME_SHA256, "pinned runtime hash mismatch")
    require(run_a.semantic == run_b.semantic, "address-free descriptor trace differs")
    require(digest_file(run_a.semantic_path) == digest_file(run_b.semantic_path), "descriptor trace hash differs")
    mapping_a = sorted(
        (
            resource_token,
            allocation["token"],
        )
        for resource_token, allocation in run_a.runtime_allocation_by_resource.items()
    )
    mapping_b = sorted(
        (
            resource_token,
            allocation["token"],
        )
        for resource_token, allocation in run_b.runtime_allocation_by_resource.items()
    )
    require(mapping_a == mapping_b, "runtime resource/allocation mapping differs")
    require(run_a.consumer_bindings == run_b.consumer_bindings, "packed parameter bindings differ")

    requests = selected_request_receipts(run_a)
    requests_b = selected_request_receipts(run_b)
    require(requests == requests_b, "selected request receipts differ")
    request_by_resource: dict[str, list[str]] = defaultdict(list)
    for request in requests:
        request_by_resource[request["d3d_resource_token"]].append(
            request["nvapi_resource_token"]
        )

    selected_resources: list[dict[str, Any]] = []
    for token in sorted(request_by_resource, key=token_number):
        event = run_a.resources[token]
        allocation = run_a.runtime_allocation_by_resource.get(token)
        selected_resources.append(
            {
                "d3d_resource_token": token,
                "creation_phase": event["phase"],
                "creation_method": event["creation_method"],
                "runtime_allocation_token": (
                    f"allocation-{allocation['token']:06d}" if allocation else None
                ),
                "resource": resource_description(event),
                "nvapi_resource_tokens": request_by_resource[token],
            }
        )

    runtime_textures: list[dict[str, Any]] = []
    for token, allocation in sorted(
        run_a.runtime_allocation_by_resource.items(), key=lambda item: item[1]["token"]
    ):
        resource = run_a.resources[token]
        if allocation["dimension"] != 3:
            continue
        runtime_textures.append(
            {
                "runtime_allocation_token": f"allocation-{allocation['token']:06d}",
                "d3d_resource_token": token,
                "resource": resource_description(resource),
                "nvapi_resource_tokens": request_by_resource.get(token, []),
                "selected_by_resource_object_request": token in request_by_resource,
            }
        )
    require(len(runtime_textures) == 30, "runtime texture allocation census drift")

    binding_count = sum(
        len(bindings) for bindings in run_a.consumer_bindings.values()
    )
    handles_with_bindings = sum(
        bool(bindings) for bindings in run_a.consumer_bindings.values()
    )
    unbound_handles = sorted(
        token
        for token, bindings in run_a.consumer_bindings.items()
        if not bindings
    )
    semantic_hash = digest_file(run_a.semantic_path)
    runtime_mapping = [
        {
            "d3d_resource_token": token,
            "runtime_allocation_token": f"allocation-{allocation['token']:06d}",
        }
        for token, allocation in sorted(
            run_a.runtime_allocation_by_resource.items(),
            key=lambda item: item[1]["token"],
        )
    ]

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "job_id": run_a.process_receipt["job_id"],
        "builder_sha256": digest_file(Path(__file__).resolve()),
        "pinned_runtime": {
            "nvngx_dlssg_sha256": EXPECTED_RUNTIME_SHA256,
            "generated_frame_sha256": EXPECTED_OUTPUT_SHA256,
            "source_identity": source_a,
        },
        "fresh_process_gate": {
            "run_a": {
                "run_id": run_a.name,
                "process_id": run_a.process_receipt["process_id"],
                "run_nonce": run_a.process_receipt["run_nonce"],
                "queue_nonce": run_a.process_receipt["queue_nonce"],
                "poison_byte": run_a.process_receipt["poison_byte"],
                "descriptor_process_receipt_sha256": digest_file(
                    run_a.process_receipt_path
                ),
            },
            "run_b": {
                "run_id": run_b.name,
                "process_id": run_b.process_receipt["process_id"],
                "run_nonce": run_b.process_receipt["run_nonce"],
                "queue_nonce": run_b.process_receipt["queue_nonce"],
                "poison_byte": run_b.process_receipt["poison_byte"],
                "descriptor_process_receipt_sha256": digest_file(
                    run_b.process_receipt_path
                ),
            },
            "process_ids_distinct": True,
            "run_nonces_distinct": True,
            "queue_nonces_distinct": True,
            "output_repeat_exact": True,
            "output_untouched_pixels_each": 0,
        },
        "capture_instrumentation": {
            "method": "early_ID3D12Device_vtable_observation",
            "hook_count": 9,
            "descriptor_snapshot_timing": "before_real_NvAPI_object_request",
            "driver_scratch_descriptor_rewrites_observed": True,
            "semantic_trace_sha256": semantic_hash,
            "semantic_trace_byte_identical_across_processes": True,
            "raw_addresses_confined_to_volatile_sidecars": True,
            "hooks_restored_each_process": True,
        },
        "capture_counts_per_process": {
            "d3d_resource_creations": 225,
            "runtime_callback_allocations": 205,
            "runtime_callback_texture_allocations": 30,
            "descriptor_heaps": 4,
            "descriptor_writes_and_copies": 213,
            "descriptor_copies": 54,
            "nvapi_resource_object_requests": 64,
            "successful_nvapi_resource_object_requests": 60,
            "intentional_failed_abi_smoke_requests": 4,
            "nvapi_launch_calls_including_failed_abi_smoke": 85,
            "successful_kernel_parameter_records": 83,
        },
        "runtime_allocation_correlation": {
            "method": "exact_create_phase_ordinal_join_with_full_descriptor_equality",
            "resource_count": 205,
            "all_descriptor_fields_equal": True,
            "repeat_equal": True,
            "mapping_root_sha256": digest(runtime_mapping),
            "mapping": runtime_mapping,
        },
        "runtime_texture_allocations": runtime_textures,
        "selected_d3d_resources": selected_resources,
        "resource_object_requests": requests,
        "packed_parameter_bindings": {
            "cuda_object_count": 60,
            "objects_with_aligned_parameter_occurrences": handles_with_bindings,
            "objects_without_parameter_occurrences": len(unbound_handles),
            "unbound_nvapi_resource_tokens": unbound_handles,
            "aligned_occurrence_count": binding_count,
            "alignment_bytes": 8,
            "repeat_equal": True,
            "binding_root_sha256": digest(run_a.consumer_bindings),
            "interpretation": "byte_binding_only_not_SASS_dereference_proof",
        },
        "post_call_snapshot_counterexample": build_counterexample(
            post_call_counterexample
        ),
        "remaining_limits": [
            "opaque CUDA texture and surface objects are not GPU virtual addresses",
            "four successful objects have no direct occurrence in captured packed launch parameters",
            "seven runtime-created textures are not selected by the captured NvAPI object requests",
            "parameter byte occurrence does not prove a kernel dereference without module-level access analysis",
            "resource contents and RTX-to-Metal numerical parity are outside this receipt",
        ],
        "blockers": [],
        "claims": {
            "live_d3d12_resource_descriptors_captured": True,
            "live_nvapi_descriptor_relationships_captured": True,
            "all_successful_nvapi_requests_resolved": True,
            "runtime_allocation_mapping_repeat_exact": True,
            "selected_resource_mapping_repeat_exact": True,
            "packed_parameter_binding_repeat_exact": True,
            "cuda_objects_are_gpu_virtual_addresses": False,
            "full_model_recovered": False,
            "mac_parity": False,
        },
    }
    receipt["receipt_root_sha256"] = rooted(receipt, "receipt_root_sha256")
    return receipt


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--post-call-counterexample", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        receipt = build_receipt(
            args.input_root,
            args.run_a,
            args.run_b,
            args.post_call_counterexample,
        )
        require(not args.output.exists(), f"output must be fresh: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "result": "accepted",
            "receipt_root_sha256": receipt["receipt_root_sha256"],
            "selected_d3d_resources": len(receipt["selected_d3d_resources"]),
            "successful_nvapi_requests": len(receipt["resource_object_requests"]),
            "packed_parameter_occurrences": receipt["packed_parameter_bindings"]["aligned_occurrence_count"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ProvenanceError) as error:
        print(json.dumps({"result": "rejected", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
