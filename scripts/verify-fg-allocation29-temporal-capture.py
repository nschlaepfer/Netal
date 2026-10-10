#!/usr/bin/env python3
"""Verify allocation29's generated-write to next-frame-read provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


RUNTIME_SHA256 = "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"
FIRST_GENERATED_SHA256 = "07a87435b2e7d79540dcec737b9abb6354e68e8595b4df29ca6f8d92cce54383"
SECOND_GENERATED_SHA256 = "1abac274f224b4e726b2ce3c824f1012f3aca701ed8075f18f446fa3e041f073"
STATE_SHA256 = "7933f2db8534defdbb941615e544f1ab86ccad2c9f8cc3bcc72852b71bacdd81"
DESCRIPTOR_SEMANTIC_SHA256 = "e4d195c6b8b891cac32bbb4c274499a043a9c60c82b21b987a627e390b406b1c"
PROXY_SHA256 = "08f1b60155300bfbc7647d5e360503229077a4482336daa5e97714044e6d9efe"
HOST_SOURCE_SHA256 = "7bf8e2d78181f4ffb3c62f8f47bee1aa270fb107a2e787620698153789b0b7a0"
CAPTURE_SOURCE_SHA256 = "b9029a61aa8b351ffcb64719e374a9a8fa57c2cff157df23b29e25fef69a61dc"
ORCHESTRATION_SHA256 = "7a820e5328ac54076299847ebbc5630c4b83c4cca87c91d2f3b75db245cb3f88"
CAPTURE_RECEIPT_SHA256 = "89221c5e49d61e3658855128b53f5aab2d4f11084c60d461b5cc6cfbf1b94dc8"
MODULE5_SHA256 = "3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"
MODULE6_SHA256 = "d45333f43f343d1f38bd304fe76214dc7064f2a7d24ce1ea06bc791867783264"
CAPTURE_POINT = "post_evaluate_1_fence_pre_evaluate_2"
JOB_ID = "fg-allocation29-temporal-provenance-v2-20260904T131049Z"

RUNS = {
    "allocation29-post-generated-poison-a5-process-a": 165,
    "allocation29-post-generated-poison-a5-process-b": 165,
    "allocation29-post-generated-poison-5a-process-a": 90,
    "allocation29-post-generated-poison-5a-process-b": 90,
}

TARGET_LAUNCHES = {
    "launch-000019": ("module-000005", 192, 48, "generated_1_read"),
    "launch-000052": ("module-000006", 168, 64, "generated_1_write"),
    "launch-000095": ("module-000005", 192, 48, "generated_2_read"),
    "launch-000128": ("module-000006", 168, 64, "generated_2_write"),
}

FORBIDDEN_SEMANTIC_FIELDS = {
    "address",
    "cpu_descriptor_handle",
    "cpu_descriptor_start",
    "cuda_object_handle",
    "descriptor_ptr",
    "gpu_va_size",
    "gpu_va_start",
    "heap_identity",
    "resource_identity",
    "sampler_descriptor_ptr",
    "texture_descriptor_ptr",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for ordinal, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL object required: {path}:{ordinal + 1}")
        values.append(value)
    require(values, f"empty JSONL input: {path}")
    return values


def validate_semantic_trace(events: list[dict[str, Any]]) -> None:
    for ordinal, event in enumerate(events):
        leaked = sorted(FORBIDDEN_SEMANTIC_FIELDS.intersection(event))
        require(not leaked, f"semantic descriptor event {ordinal} leaks {leaked}")
        for flag in (
            "addresses_recorded",
            "device_address_recorded",
            "gpu_virtual_addresses_recorded",
            "opaque_cuda_handles_recorded",
            "gpu_virtual_address_recorded",
        ):
            if flag in event:
                require(event[flag] is False, f"semantic trace enables {flag}")
    require(events[0].get("event") == "descriptor_runtime_ready", "descriptor header missing")
    summary = events[-1]
    require(summary.get("event") == "descriptor_capture_summary", "descriptor summary missing")
    require(summary.get("capture_healthy") is True, "descriptor trace unhealthy")
    require(summary.get("hooks_restored") is True, "descriptor hooks were not restored")
    require(summary.get("unresolved_successful_request_count") == 0, "unresolved descriptor request")
    require(summary.get("resource_count") == 244, "three-frame resource count drift")
    require(summary.get("successful_request_count") == 64, "resource-object count drift")


def resource_matches_allocation(resource: dict[str, Any], allocation: dict[str, Any]) -> bool:
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
    return all(resource.get(left) == allocation.get(right) for left, right in pairs)


def parse_handle(value: str, width: int) -> bytes:
    require(isinstance(value, str) and value.startswith("0x"), "raw handle encoding drift")
    number = int(value[2:], 16)
    return number.to_bytes(width, byteorder="little", signed=False)


def validate_target_launch(
    run_root: Path,
    event: dict[str, Any],
    handle: bytes,
    expected: tuple[str, int, int, str],
) -> dict[str, Any]:
    module, parameter_bytes, offset, role = expected
    require(event.get("kernel_index") == 0, f"{role} kernel index drift")
    require(event.get("module_token") == module, f"{role} module drift")
    require(event.get("function_name") == "main_kernel", f"{role} function drift")
    require(event.get("grid") == [20, 12, 1], f"{role} grid drift")
    require(event.get("block") == [16, 16, 1], f"{role} block drift")
    require(event.get("parameter_mode") == "packed", f"{role} parameter mode drift")
    require(event.get("parameter_bytes") == parameter_bytes, f"{role} parameter size drift")
    name = event.get("parameter_file")
    require(isinstance(name, str) and Path(name).name == name, f"{role} parameter path unsafe")
    path = run_root / name
    data = path.read_bytes()
    require(len(data) == parameter_bytes, f"{role} parameter file size drift")
    require(sha_file(path).upper() == event.get("parameter_sha256"), f"{role} parameter hash drift")
    require(data[offset:offset + len(handle)] == handle, f"{role} allocation29 binding drift")
    return {
        "launch_token": event["launch_token"],
        "module_token": module,
        "parameter_bytes": parameter_bytes,
        "allocation29_parameter_offset": offset,
        "role": role,
    }


def state_census(data: bytes) -> dict[str, Any]:
    require(len(data) == 115200 and len(data) % 2 == 0, "allocation29 state extent drift")
    bits = [value[0] for value in struct.iter_unpack("<H", data)]
    values = [struct.unpack("<e", struct.pack("<H", value))[0] for value in bits]
    require(all(math.isfinite(value) for value in values), "allocation29 contains non-finite state")
    unique = sorted(set(bits))
    return {
        "halfword_count": len(bits),
        "unique_half_bit_pattern_count": len(unique),
        "zero_count": sum(value == 0x0000 for value in bits),
        "half_0_5_count": sum(value == 0x3800 for value in bits),
        "minimum": min(values),
        "maximum": max(values),
        "nonzero_count": sum(value != 0x0000 for value in bits),
    }


def validate_run(root: Path, run_id: str, poison: int, row: dict[str, Any]) -> tuple[dict[str, Any], tuple[Any, Any, Any]]:
    run_root = root / run_id
    require(run_root.is_dir(), f"run directory missing: {run_id}")

    process_path = run_root / f"{run_id}-process.json"
    host_path = run_root / "fg-run-receipt.json"
    manifest_path = run_root / "runtime-default-buffers-post-evaluate-1.manifest.json"
    state_path = run_root / "default-token-000029.bin"
    semantic_path = run_root / "descriptor-semantic.jsonl"
    volatile_path = run_root / "descriptor-volatile.jsonl"
    nvapi_path = run_root / "nvapi-trace.jsonl"
    nvapi_volatile_path = run_root / "nvapi-volatile.jsonl"
    allocation_path = run_root / "runtime-allocations-semantic.jsonl"
    for path in (
        process_path, host_path, manifest_path, state_path, semantic_path,
        volatile_path, nvapi_path, nvapi_volatile_path, allocation_path,
    ):
        require(path.is_file() and not path.is_symlink(), f"required run file missing: {path}")

    require(row.get("run_id") == run_id, f"run-row identity drift: {run_id}")
    require(row.get("poison_byte") == poison, f"poison drift: {run_id}")
    require(row.get("rendered_frames") == 3, f"frame-count drift: {run_id}")
    require(row.get("capture_point") == CAPTURE_POINT, f"capture-point drift: {run_id}")
    require(row.get("dump_bytes") == 115200, f"state byte-count drift: {run_id}")
    require(row.get("dump_sha256") == sha_file(state_path) == STATE_SHA256, f"state hash drift: {run_id}")
    require(row.get("manifest_sha256") == sha_file(manifest_path), f"manifest hash drift: {run_id}")
    require(row.get("host_receipt_sha256") == sha_file(host_path), f"host hash drift: {run_id}")
    require(row.get("process_receipt_sha256") == sha_file(process_path), f"process hash drift: {run_id}")
    require(row.get("descriptor_semantic_sha256") == sha_file(semantic_path) == DESCRIPTOR_SEMANTIC_SHA256, f"semantic trace hash drift: {run_id}")
    require(row.get("descriptor_volatile_sha256") == sha_file(volatile_path), f"volatile descriptor hash drift: {run_id}")

    process = load_json(process_path)
    host = load_json(host_path)
    require(process.get("schema") == "neuralstudio.dlss_fg.traced_process.v1", "process schema drift")
    require(process.get("exit_code") == 0 and process.get("error") is None, "process did not succeed")
    require(process.get("run_id") == run_id and process.get("output_poison_byte") == poison, "process identity drift")
    require(process.get("rendered_frames") == 3, "process frame count drift")
    require(process.get("runtime_sha256") == RUNTIME_SHA256, "process runtime drift")
    require(process.get("proxy_sha256") == PROXY_SHA256, "descriptor-aware proxy drift")
    require(process.get("host_receipt_sha256") == sha_file(host_path), "process-to-host hash drift")
    require(process.get("trace_sha256") == sha_file(nvapi_path), "process-to-NvAPI trace hash drift")
    require(process.get("volatile_trace_sha256") == sha_file(nvapi_volatile_path), "process-to-volatile trace hash drift")
    require(host.get("runtime_sha256") == RUNTIME_SHA256 and host.get("accepted") is True, "host rejected")
    require(host.get("output_poison_byte") == poison, "host poison drift")
    frames = host.get("frames")
    require(isinstance(frames, list) and len(frames) == 3, "host frame receipt drift")
    require(frames[1].get("output_interpolated_sha256") == FIRST_GENERATED_SHA256, "first generated frame drift")
    require(frames[2].get("output_interpolated_sha256") == SECOND_GENERATED_SHA256, "second generated frame drift")
    require(frames[1].get("output_untouched_pixels") == 0 and frames[2].get("output_untouched_pixels") == 0, "generated output poison remained")
    require(frames[1].get("generated_output_accepted") is True and frames[2].get("generated_output_accepted") is True, "generated frame rejected")
    require(frames[1].get("input_sha256") != frames[2].get("input_sha256"), "temporal inputs collapsed")
    require(row.get("first_generated_output_sha256") == FIRST_GENERATED_SHA256, "row first output drift")
    require(row.get("second_generated_output_sha256") == SECOND_GENERATED_SHA256, "row second output drift")

    capture = host.get("runtime_allocation_capture", {})
    require(capture.get("enabled") is True and capture.get("healthy") is True, "allocation capture unhealthy")
    require(capture.get("allocations") == 205 and capture.get("releases") == 205 and capture.get("leaks") == 0, "allocation lifecycle drift")
    require(capture.get("baseline_output_match") is True, "baseline output changed under capture")
    require(capture.get("default_capture_enabled") is True and capture.get("default_capture_point") == CAPTURE_POINT, "default capture boundary drift")
    require(capture.get("default_dump_count") == 1 and capture.get("default_dump_bytes") == 115200, "default capture extent drift")
    require(capture.get("semantic_ledger_sha256") == sha_file(allocation_path), "allocation ledger hash drift")

    manifest = load_json(manifest_path)
    require(manifest.get("schema") == "neuralstudio.dlss_fg.default_resource_readback.v2", "manifest schema drift")
    require(manifest.get("capture_point") == CAPTURE_POINT, "manifest boundary drift")
    require(manifest.get("source_state_strategy") == "texture_original_to_copy_source_and_restore", "texture transition contract drift")
    require(manifest.get("bytes_fed_back_to_execution") is False and manifest.get("direct_default_map") is False, "captured state entered execution")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == 1, "manifest entry count drift")
    entry = entries[0]
    require(entry.get("source_token") == 29 and entry.get("resource_kind") == "texture2d", "allocation29 manifest identity drift")
    require(entry.get("descriptor_width") == 320 and entry.get("row_count") == 180 and entry.get("row_bytes") == 640, "allocation29 texture extent drift")
    require(entry.get("logical_bytes") == 115200 and entry.get("sha256") == STATE_SHA256, "allocation29 manifest payload drift")

    semantic = load_jsonl(semantic_path)
    validate_semantic_trace(semantic)
    resources = {
        event["resource_token"]: event
        for event in semantic if event.get("event") == "resource_created"
    }
    descriptors = {
        event["descriptor_token"]: event
        for event in semantic
        if event.get("event") in {"descriptor_written", "descriptor_copied"}
    }
    allocation_events = load_jsonl(allocation_path)
    allocations = [event for event in allocation_events if event.get("event") == "allocation"]
    release_tokens = sorted(
        event["token"] for event in allocation_events if event.get("event") == "release"
    )
    create_resources = [
        event for event in semantic
        if event.get("event") == "resource_created" and event.get("phase") == "create"
    ]
    require(len(allocations) == len(create_resources) == 205, "allocation-to-D3D correlation count drift")
    target_resource = None
    for resource, allocation in zip(create_resources, allocations, strict=True):
        require(resource_matches_allocation(resource, allocation), f"D3D allocation descriptor drift at token {allocation.get('token')}")
        if allocation.get("token") == 29:
            target_resource = resource["resource_token"]
    require(target_resource is not None, "allocation29 D3D resource missing")
    allocation29 = allocations[28]
    require(
        allocation29.get("token") == 29
        and allocation29.get("dimension") == 3
        and allocation29.get("width") == 320
        and allocation29.get("height") == 180
        and allocation29.get("format") == 54
        and allocation29.get("initial_state") == 8,
        "allocation29 descriptor identity drift",
    )

    target_requests = []
    for request in semantic:
        if request.get("event") != "resource_object_request" or request.get("status") != 0:
            continue
        descriptor = descriptors.get(request.get("descriptor_token"))
        if descriptor and descriptor.get("resource_token") == target_resource:
            target_requests.append(request)
    require(len(target_requests) == 2, "allocation29 must have exactly texture and surface objects")
    by_kind = {request["resource_kind"]: request for request in target_requests}
    require(set(by_kind) == {"independent_surface", "merged_texture_sampler"}, "allocation29 object kinds drift")

    raw_objects = {
        event["resource_token"]: event
        for event in load_jsonl(nvapi_volatile_path)
        if event.get("event") == "resource_object_raw" and event.get("status") == 0
    }
    surface_request = by_kind["independent_surface"]
    texture_request = by_kind["merged_texture_sampler"]
    surface_raw = raw_objects.get(surface_request["nvapi_resource_token"])
    texture_raw = raw_objects.get(texture_request["nvapi_resource_token"])
    require(surface_raw is not None and texture_raw is not None, "allocation29 raw handle join failed")
    require(surface_raw.get("cuda_object_handle_bytes") == 8 and texture_raw.get("cuda_object_handle_bytes") == 8, "allocation29 handle width drift")
    surface_handle = parse_handle(surface_raw["cuda_object_handle"], 8)
    texture_handle = parse_handle(texture_raw["cuda_object_handle"], 8)

    trace = load_jsonl(nvapi_path)
    launches = {
        event["launch_token"]: event
        for event in trace if event.get("event") == "kernel_launch"
    }
    require(len(launches) == 161, "three-frame launch count drift")
    module_hashes = {
        event["module_token"]: str(event.get("module_blob_sha256", "")).lower()
        for event in trace
        if event.get("call") == "NvAPI_D3D12_CreateCuModule" and event.get("status") == 0
    }
    require(module_hashes.get("module-000005") == MODULE5_SHA256, "L19 module identity drift")
    require(module_hashes.get("module-000006") == MODULE6_SHA256, "L52 module identity drift")

    bindings = []
    for token, expected in TARGET_LAUNCHES.items():
        require(token in launches, f"target launch missing: {token}")
        handle = texture_handle if expected[3].endswith("read") else surface_handle
        bindings.append(validate_target_launch(run_root, launches[token], handle, expected))

    positions = {
        event.get("launch_token", event.get("phase")): ordinal
        for ordinal, event in enumerate(trace)
        if event.get("event") in {"kernel_launch", "phase_marker"}
    }
    start = positions.get("capture_default_buffers.post-evaluate-1")
    complete = positions.get("capture_default_buffers.post-evaluate-1.copy_complete")
    require(
        isinstance(start, int) and isinstance(complete, int)
        and positions["launch-000052"] < start < complete < positions["launch-000095"],
        "capture boundary is not ordered between L52 write and next-frame L19 read",
    )

    run_receipt = {
        "run_id": run_id,
        "poison_byte": poison,
        "descriptor_semantic_sha256": sha_file(semantic_path),
        "allocation_descriptor_root_sha256": canonical_sha({
            "allocations": allocations,
            "released_tokens": release_tokens,
        }),
        "manifest_sha256": sha_file(manifest_path),
        "state_sha256": sha_file(state_path),
        "allocation29_d3d_resource_token": target_resource,
        "allocation29_object_kinds": sorted(by_kind),
        "bindings": bindings,
        "capture_order": [
            "generated_1_L52_write",
            "fenced_texture_readback",
            "generated_2_L19_read",
            "generated_2_L52_write",
        ],
        "first_generated_output_sha256": FIRST_GENERATED_SHA256,
        "second_generated_output_sha256": SECOND_GENERATED_SHA256,
        "generated_output_untouched_pixels": [0, 0],
        "raw_process_and_volatile_evidence_validated": True,
        "raw_addresses_or_handles_in_normalized_receipt": False,
    }
    identity = (host.get("process_id"), host.get("run_nonce"), host.get("queue_nonce"))
    return run_receipt, identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.input_root.resolve()
    repository = arguments.repository.resolve()
    capture_receipt = load_json(arguments.receipt.resolve())
    require(sha_file(arguments.receipt.resolve()) == CAPTURE_RECEIPT_SHA256, "capture receipt file drift")
    require(capture_receipt.get("schema") == "neuralstudio.fg.allocation29-temporal-capture.v1", "capture receipt schema drift")
    require(capture_receipt.get("job_id") == JOB_ID, "capture job identity drift")
    require(capture_receipt.get("runtime_sha256") == RUNTIME_SHA256, "capture runtime drift")
    require(capture_receipt.get("host_source_sha256") == HOST_SOURCE_SHA256, "host source drift")
    require(capture_receipt.get("allocation_capture_source_sha256") == CAPTURE_SOURCE_SHA256, "capture source drift")
    require(capture_receipt.get("orchestration_source_sha256") == ORCHESTRATION_SHA256, "orchestration source drift")
    require(capture_receipt.get("fresh_processes") == 4, "fresh process count drift")
    require(capture_receipt.get("poison_values") == [165, 90], "poison matrix drift")
    require(capture_receipt.get("capture_between_generated_evaluations") is True, "temporal boundary flag missing")
    require(capture_receipt.get("d3d12_descriptor_capture") is True, "descriptor capture flag missing")
    require(capture_receipt.get("expected_frame_bytes_fed_to_capture") is False, "expected frame bytes entered capture")
    require(capture_receipt.get("captured_bytes_fed_back_to_execution") is False, "captured state entered execution")
    require(capture_receipt.get("address_fields_recorded") is False, "capture receipt contains addresses")
    require(capture_receipt.get("result") == "pass", "capture receipt rejected")

    local_sources = {
        "host": repository / "src/fg_host/main.cpp",
        "capture": repository / "src/fg_host/runtime_allocation_capture.cpp",
        "orchestration": repository / "scripts/run-fg-allocation29-temporal-capture.ps1",
    }
    expected_sources = {
        "host": HOST_SOURCE_SHA256,
        "capture": CAPTURE_SOURCE_SHA256,
        "orchestration": ORCHESTRATION_SHA256,
    }
    for name, path in local_sources.items():
        require(sha_file(path) == expected_sources[name], f"local {name} source drift")

    rows = capture_receipt.get("runs")
    require(isinstance(rows, list) and len(rows) == 4, "capture run matrix drift")
    row_by_id = {row.get("run_id"): row for row in rows if isinstance(row, dict)}
    require(set(row_by_id) == set(RUNS), "capture run IDs drift")

    run_receipts = []
    identities = []
    for run_id, poison in RUNS.items():
        run_receipt, identity = validate_run(root, run_id, poison, row_by_id[run_id])
        run_receipts.append(run_receipt)
        identities.append(identity)
    require(len(set(identities)) == 4, "fresh process identities collapsed")

    state_data = (root / next(iter(RUNS)) / "default-token-000029.bin").read_bytes()
    census = state_census(state_data)
    require(census["unique_half_bit_pattern_count"] > 1, "post-L52 state remained uniform")
    require(census["half_0_5_count"] < census["halfword_count"], "post-L52 state remained reset half 0.5")

    evidence = {
        "schema": "neuralstudio.fg.allocation29-temporal-provenance.v1",
        "job_id": JOB_ID,
        "pinned_runtime_sha256": RUNTIME_SHA256,
        "source_sha256": {
            "host": HOST_SOURCE_SHA256,
            "runtime_capture": CAPTURE_SOURCE_SHA256,
            "orchestration": ORCHESTRATION_SHA256,
        },
        "raw_capture_receipt_validated": True,
        "fresh_process_count": 4,
        "fresh_process_identities_distinct": True,
        "poison_values": [165, 90],
        "state_repeat_exact": True,
        "state_sha256": STATE_SHA256,
        "state_bytes": 115200,
        "state_census": census,
        "allocation29_descriptor": {
            "format": "DXGI_FORMAT_R16_FLOAT",
            "width": 320,
            "height": 180,
            "source_token": 29,
        },
        "temporal_chain": {
            "producer": "generated_1 canonical L52 module6 surface offset64",
            "capture": CAPTURE_POINT,
            "consumer": "generated_2 canonical L19 module5 texture offset48",
            "next_producer": "generated_2 canonical L52 module6 surface offset64",
            "producer_module_sha256": MODULE6_SHA256,
            "consumer_module_sha256": MODULE5_SHA256,
            "same_live_d3d12_resource": True,
            "capture_order_verified": True,
            "captured_bytes_fed_back_to_execution": False,
        },
        "output_invariance": {
            "first_generated_sha256": FIRST_GENERATED_SHA256,
            "second_generated_sha256": SECOND_GENERATED_SHA256,
            "untouched_pixels": 0,
        },
        "runs": run_receipts,
        "address_fields_recorded": False,
        "claims": {
            "post_l52_allocation29_state_captured": True,
            "same_process_next_generated_l19_consumes_allocation29": True,
            "complete_temporal_metal_parity": False,
            "complete_frame_parity": False,
            "netal_integration": False,
            "performance": False,
        },
        "remaining_limits": [
            "This receipt proves RTX temporal provenance, not Metal temporal parity.",
            "The readback is observational evidence and is forbidden as runtime input.",
            "Only the pinned 640x360 synthetic three-frame sequence is covered.",
        ],
        "result": "pass",
    }
    evidence["receipt_root_sha256"] = canonical_sha(evidence)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"state_sha256={STATE_SHA256}")
    print(f"receipt_root_sha256={evidence['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
