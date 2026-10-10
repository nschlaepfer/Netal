#!/usr/bin/env python3
"""Build a two-process FG runtime-allocation and GPU-VA provenance receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
NVAPI_REPEAT_PATH = REPOSITORY / "scripts" / "build-fg-nvapi-repeat.py"
SCHEMA = "neuralstudio.fg.allocation-provenance-repeat.v1"
HASH = re.compile(r"^[0-9a-f]{64}$")
MODULE21_SHA256 = "11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"


class AllocationError(RuntimeError):
    pass


def load_nvapi_repeat() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fg_nvapi_repeat_for_allocations", NVAPI_REPEAT_PATH
    )
    if spec is None or spec.loader is None:
        raise AllocationError("cannot load NvAPI repeat dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nvapi = load_nvapi_repeat()


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
        raise AllocationError(message)


def is_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise AllocationError(f"invalid JSON in {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSONL: {path}")
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        require(bool(line.strip()), f"blank JSONL line at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AllocationError(f"invalid JSON at {path}:{line_number}: {error}") from error
        require(isinstance(value, dict), f"JSONL value is not an object at {path}:{line_number}")
        result.append(value)
    require(bool(result), f"empty JSONL: {path}")
    return result


ALLOCATION_FIELDS = {
    "event",
    "token",
    "dimension",
    "alignment",
    "width",
    "height",
    "depth_or_array",
    "mip_levels",
    "format",
    "sample_count",
    "sample_quality",
    "layout",
    "resource_flags",
    "heap_type",
    "heap_flags",
    "cpu_page_property",
    "memory_pool_preference",
    "creation_node_mask",
    "visible_node_mask",
    "initial_state",
    "allocation_size",
    "allocation_alignment",
}


@dataclass
class AllocationRun:
    semantic_input: dict[str, Any]
    volatile_input: dict[str, Any]
    descriptors: list[dict[str, Any]]
    descriptor_root_sha256: str
    release_order: list[str]
    release_order_root_sha256: str
    release_set_root_sha256: str
    buffer_ranges: list[tuple[int, int, str]]
    zero_gpu_va_resources: list[dict[str, Any]]
    range_ambiguities: list[dict[str, Any]]
    lifetime_contained: bool


def allocation_token(value: int) -> str:
    return f"allocation-{value:06d}"


def parse_allocation_run(
    semantic_path: Path, volatile_path: Path
) -> AllocationRun:
    semantic = load_jsonl(semantic_path)
    volatile = load_jsonl(volatile_path)
    semantic_allocations = [item for item in semantic if item.get("event") == "allocation"]
    semantic_releases = [item for item in semantic if item.get("event") == "release"]
    semantic_summaries = [item for item in semantic if item.get("event") == "summary"]
    require(
        len(semantic_summaries) == 1 and semantic[-1] is semantic_summaries[0],
        "allocation semantic ledger must end in one summary",
    )
    require(
        not any(item.get("event") not in {"allocation", "release", "summary"} for item in semantic),
        "unknown allocation semantic event",
    )
    count = len(semantic_allocations)
    require(count > 0 and len(semantic_releases) == count, "allocation/release count mismatch")
    require(
        semantic[:count] == semantic_allocations
        and semantic[count : count * 2] == semantic_releases,
        "allocations must all precede releases for launch-lifetime containment",
    )
    summary = semantic_summaries[0]
    require(
        set(summary) == {"event", "allocations", "releases", "leaks", "healthy"},
        "allocation summary fields drifted",
    )
    require(
        summary == {
            "event": "summary",
            "allocations": count,
            "releases": count,
            "leaks": 0,
            "healthy": True,
        },
        "allocation summary is not healthy",
    )
    tokens: list[int] = []
    descriptors: list[dict[str, Any]] = []
    semantic_by_token: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(semantic_allocations):
        require(set(item) == ALLOCATION_FIELDS, f"allocation fields drifted at index {index}")
        token = item.get("token")
        require(isinstance(token, int) and token > 0, "allocation token is invalid")
        require(token not in semantic_by_token, "duplicate allocation token")
        require(
            all(
                isinstance(item.get(field), int) and item[field] >= 0
                for field in ALLOCATION_FIELDS - {"event"}
            ),
            f"allocation descriptor contains a non-unsigned field: {token}",
        )
        require(item["allocation_size"] > 0, "allocation size is zero")
        semantic_by_token[token] = item
        tokens.append(token)
        descriptor = {key: item[key] for key in sorted(ALLOCATION_FIELDS - {"event", "token"})}
        descriptor["allocation_token"] = allocation_token(token)
        descriptor["resource_class"] = (
            "buffer" if item["dimension"] == 1 else "texture"
            if item["dimension"] in {2, 3}
            else "other"
        )
        descriptors.append(descriptor)
    require(tokens == list(range(1, count + 1)), "allocation tokens are not contiguous encounter IDs")
    release_tokens = [item.get("token") for item in semantic_releases]
    require(
        all(set(item) == {"event", "token"} for item in semantic_releases),
        "release fields drifted",
    )
    require(
        len(set(release_tokens)) == count and set(release_tokens) == set(tokens),
        "release tokens are incomplete or duplicated",
    )

    volatile_allocations = [item for item in volatile if item.get("event") == "allocation"]
    volatile_releases = [item for item in volatile if item.get("event") == "release"]
    require(
        not any(item.get("event") not in {"allocation", "release"} for item in volatile),
        "unknown allocation volatile event",
    )
    require(
        len(volatile_allocations) == count and len(volatile_releases) == count,
        "volatile allocation/release count mismatch",
    )
    require(
        volatile[:count] == volatile_allocations
        and volatile[count:] == volatile_releases,
        "volatile allocations must precede releases",
    )
    volatile_by_token: dict[int, dict[str, Any]] = {}
    for item in volatile_allocations:
        require(
            set(item) == {"event", "token", "gpu_virtual_address", "allocation_size"},
            "volatile allocation fields drifted",
        )
        token = item.get("token")
        address = item.get("gpu_virtual_address")
        size = item.get("allocation_size")
        require(
            isinstance(token, int)
            and token in semantic_by_token
            and token not in volatile_by_token,
            "volatile allocation token is invalid",
        )
        require(isinstance(address, int) and address >= 0, "GPU virtual address is invalid")
        require(size == semantic_by_token[token]["allocation_size"], "volatile allocation size drift")
        volatile_by_token[token] = item
    volatile_release_tokens = [item.get("token") for item in volatile_releases]
    require(
        all(set(item) == {"event", "token"} for item in volatile_releases)
        and volatile_release_tokens == release_tokens,
        "semantic and volatile release sequences disagree",
    )

    buffer_ranges: list[tuple[int, int, str]] = []
    zero_gpu_va_resources: list[dict[str, Any]] = []
    for token in tokens:
        semantic_item = semantic_by_token[token]
        volatile_item = volatile_by_token[token]
        address = volatile_item["gpu_virtual_address"]
        dimension = semantic_item["dimension"]
        if dimension == 1:
            require(address != 0, "buffer allocation has no GPU virtual address")
            logical_bytes = semantic_item["width"]
            require(
                logical_bytes > 0 and logical_bytes <= semantic_item["allocation_size"],
                "buffer logical GPU-VA span exceeds its allocation",
            )
            buffer_ranges.append(
                (address, address + logical_bytes, allocation_token(token))
            )
        else:
            require(
                dimension in {2, 3} and address == 0,
                "non-buffer allocation unexpectedly exposes a GPU virtual address",
            )
            zero_gpu_va_resources.append(
                {
                    "allocation_token": allocation_token(token),
                    "dimension": dimension,
                    "width": semantic_item["width"],
                    "height": semantic_item["height"],
                    "format": semantic_item["format"],
                    "provenance_result": "texture_resource_has_no_buffer_gpu_va_span",
                }
            )
    sorted_ranges = sorted(buffer_ranges)
    ambiguities: list[dict[str, Any]] = []
    for index, (_start, end, token) in enumerate(sorted_ranges):
        for other_start, _other_end, other_token in sorted_ranges[index + 1 :]:
            if other_start >= end:
                break
            ambiguities.append(
                {
                    "allocation_token_a": token,
                    "allocation_token_b": other_token,
                    "result": "overlapping_live_buffer_gpu_va_spans",
                }
            )
    return AllocationRun(
        semantic_input={
            "sha256": digest_file(semantic_path),
            "bytes": semantic_path.stat().st_size,
            "line_count": len(semantic),
        },
        volatile_input={
            "sha256": digest_file(volatile_path),
            "bytes": volatile_path.stat().st_size,
            "line_count": len(volatile),
        },
        descriptors=descriptors,
        descriptor_root_sha256=digest(descriptors),
        release_order=[allocation_token(token) for token in release_tokens],
        release_order_root_sha256=digest(
            [allocation_token(token) for token in release_tokens]
        ),
        release_set_root_sha256=digest(
            sorted(allocation_token(token) for token in release_tokens)
        ),
        buffer_ranges=buffer_ranges,
        zero_gpu_va_resources=zero_gpu_va_resources,
        range_ambiguities=ambiguities,
        lifetime_contained=True,
    )


@dataclass
class HostRun:
    input_receipt: dict[str, Any]
    process_identity: dict[str, Any]
    runtime_identity: dict[str, Any]
    output_sequence: list[dict[str, Any]]
    output_root_sha256: str
    expected_baseline_sha256: str


def validate_file_hash(path: Path, expected: Any, label: str) -> str:
    require(is_hash(expected), f"{label} expected hash is invalid")
    require(path.is_file() and not path.is_symlink(), f"{label} file is missing or unsafe")
    actual = digest_file(path)
    require(actual == expected, f"{label} file hash mismatch")
    return actual


def parse_host_run(run_dir: Path) -> HostRun:
    receipt_path = run_dir / "fg-run-receipt.json"
    receipt = load_json(receipt_path)
    require(isinstance(receipt, dict), "host receipt is not an object")
    require(
        receipt.get("schema") == "neuralstudio.dlss_fg.standalone_d3d12.v1"
        and receipt.get("accepted") is True,
        "host FG run was not accepted",
    )
    capture = receipt.get("runtime_allocation_capture")
    require(isinstance(capture, dict), "runtime allocation capture receipt is missing")
    require(
        capture.get("enabled") is True
        and capture.get("healthy") is True
        and capture.get("baseline_output_match") is True
        and capture.get("allocations") == capture.get("releases")
        and capture.get("allocations", 0) > 0
        and capture.get("leaks") == 0,
        "runtime allocation capture was not healthy",
    )
    semantic_path = run_dir / str(capture.get("semantic_ledger"))
    volatile_path = run_dir / str(capture.get("volatile_ledger"))
    validate_file_hash(
        semantic_path, capture.get("semantic_ledger_sha256"), "semantic ledger"
    )
    validate_file_hash(
        volatile_path, capture.get("volatile_ledger_sha256"), "volatile ledger"
    )
    frames = receipt.get("frames")
    require(isinstance(frames, list) and len(frames) >= 2, "host receipt needs two frames")
    output_sequence: list[dict[str, Any]] = []
    for frame in frames:
        require(isinstance(frame, dict), "host frame receipt is invalid")
        frame_id = frame.get("frame_id")
        require(isinstance(frame_id, int) and frame_id >= 0, "frame ID is invalid")
        input_hash = frame.get("input_sha256")
        output_hash = frame.get("output_interpolated_sha256")
        disable_hash = frame.get("disable_sha256")
        validate_file_hash(
            run_dir / f"input-backbuffer-{frame_id}.rgba8",
            input_hash,
            f"frame {frame_id} input",
        )
        validate_file_hash(
            run_dir / f"output-interpolated-{frame_id}.rgba8",
            output_hash,
            f"frame {frame_id} output",
        )
        validate_file_hash(
            run_dir / f"output-disable-{frame_id}.bin",
            disable_hash,
            f"frame {frame_id} disable output",
        )
        require(
            frame.get("generated_output_accepted") is True
            and frame.get("output_untouched_pixels") == 0,
            "host frame output was not accepted",
        )
        output_sequence.append(
            {
                "frame_id": frame_id,
                "reset": frame.get("reset"),
                "input_sha256": input_hash,
                "output_interpolated_sha256": output_hash,
                "disable_sha256": disable_hash,
                "generated_output_accepted": True,
            }
        )
    expected_baseline = capture.get("expected_baseline_output_sha256")
    require(
        is_hash(expected_baseline)
        and output_sequence[-1]["output_interpolated_sha256"] == expected_baseline,
        "final output does not match the pinned baseline",
    )
    process_id = receipt.get("process_id")
    require(isinstance(process_id, int) and process_id > 0, "process ID is invalid")
    run_nonce = receipt.get("run_nonce")
    queue_nonce = receipt.get("queue_nonce")
    require(
        isinstance(run_nonce, str)
        and len(run_nonce) >= 16
        and isinstance(queue_nonce, str)
        and len(queue_nonce) >= 16,
        "run/queue nonce is invalid",
    )
    runtime_hash = receipt.get("runtime_sha256")
    proxy_hash = receipt.get("nvapi_proxy_sha256")
    thunk_hash = receipt.get("thunk_sha256")
    require(
        is_hash(runtime_hash) and is_hash(proxy_hash) and is_hash(thunk_hash),
        "runtime identity hash is invalid",
    )
    return HostRun(
        input_receipt={
            "sha256": digest_file(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "allocation_capture": copy.deepcopy(capture),
        },
        process_identity={
            "process_id": process_id,
            "run_nonce": run_nonce,
            "queue_nonce": queue_nonce,
            "output_poison_byte": receipt.get("output_poison_byte"),
        },
        runtime_identity={
            "runtime_sha256": runtime_hash,
            "nvapi_proxy_sha256": proxy_hash,
            "thunk_sha256": thunk_hash,
        },
        output_sequence=output_sequence,
        output_root_sha256=digest(output_sequence),
        expected_baseline_sha256=expected_baseline,
    )


def parse_runner_receipt(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(isinstance(value, list) and len(value) >= 5, "runner receipt is invalid")
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        require(
            isinstance(item, dict) and set(item) == {"name", "sha256"},
            "runner receipt fields drifted",
        )
        name = item.get("name")
        sha256 = item.get("sha256")
        require(isinstance(name, str) and name not in seen and is_hash(sha256), "runner receipt item is invalid")
        seen.add(name)
        files.append({"name": name, "sha256": sha256})
    required = {
        "neuralstudio-fg-standalone.exe",
        "nvapi64.dll",
        "nvapi64.real.dll",
        "nvngx.dll",
        "nvngx_dlssg.dll",
    }
    require(required.issubset(seen), "runner receipt is incomplete")
    return {
        "receipt_sha256": digest_file(path),
        "files": files,
        "files_root_sha256": digest(files),
    }


@dataclass
class PaddingProof:
    public: dict[str, Any]
    unused_by_all_variants: set[int]


TYPE_BYTES = {
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


def parse_padding_proof(ptx_path: Path, receipt_path: Path) -> PaddingProof:
    receipt = load_json(receipt_path)
    require(isinstance(receipt, dict), "PTX proof receipt is invalid")
    require(
        receipt.get("schema") == "neuralstudio.fg.module21-param-access-input.v1"
        and receipt.get("source_sha256") == MODULE21_SHA256
        and receipt.get("source_bytes") == 38472
        and receipt.get("exit_code") == 0,
        "PTX proof receipt identity mismatch",
    )
    validate_file_hash(ptx_path, receipt.get("output_sha256"), "module 21 PTX")
    text = ptx_path.read_text(encoding="utf-8-sig")
    blocks = [
        block
        for block in re.split(r"(?=Fatbin ptx code:)", text)
        if ".visible .entry main_kernel" in block
    ]
    require(len(blocks) == 2, "module 21 must contain exactly two selected PTX variants")
    variants: list[dict[str, Any]] = []
    union_accessed: set[int] = set()
    load_pattern = re.compile(
        r"(?:\.pragma\s+\"used_bytes_mask\s+(?P<mask>[0-9]+)\";\s*)?"
        r"ld\.param(?:\.v(?P<vector>[0-9]+))?\."
        r"(?P<type>[usb][0-9]+|f(?:16|32|64))\s+[^\n]*"
        r"\[(?P<address>[^\]]+)\]"
    )
    for block in blocks:
        architecture_match = re.search(r"arch\s*=\s*(sm_[0-9]+)", block)
        parameter_match = re.search(
            r"\.param\s+\.align\s+16\s+\.b8\s+main_kernel_param_0\[(\d+)\]",
            block,
        )
        require(architecture_match is not None, "PTX architecture is missing")
        require(parameter_match is not None and int(parameter_match.group(1)) == 224, "module 21 parameter block is not 224 bytes")
        require(
            re.search(r"mov\.b64\s+%rd\d+,\s*main_kernel_param_0", block)
            is not None,
            "PTX parameter base materialization is missing",
        )
        ld_lines = [line for line in block.splitlines() if "ld.param" in line]
        matches = list(load_pattern.finditer(block))
        require(len(matches) == len(ld_lines), "unsupported or dynamic ld.param access")
        accessed: set[int] = set()
        instructions: list[dict[str, Any]] = []
        for match in matches:
            scalar_type = match.group("type")
            require(scalar_type in TYPE_BYTES, f"unknown PTX load type: {scalar_type}")
            vector = int(match.group("vector") or "1")
            address = match.group("address")
            offset_match = re.search(r"\+(0x[0-9A-Fa-f]+|[0-9]+)\s*$", address)
            offset = int(offset_match.group(1), 0) if offset_match else 0
            width = TYPE_BYTES[scalar_type] * vector
            require(offset + width <= 224, "PTX parameter access exceeds 224 bytes")
            used_mask = (
                int(match.group("mask"))
                if match.group("mask") is not None
                else (1 << width) - 1
            )
            require(
                used_mask >= 0 and used_mask < (1 << width),
                "PTX used-bytes mask exceeds the load width",
            )
            used_byte_offsets = [
                offset + byte_index
                for byte_index in range(width)
                if used_mask & (1 << byte_index)
            ]
            accessed.update(used_byte_offsets)
            instructions.append(
                {
                    "offset": offset,
                    "bytes": width,
                    "type": scalar_type,
                    "vector": vector,
                    "used_bytes_mask": used_mask,
                    "used_byte_offsets": used_byte_offsets,
                }
            )
        union_accessed.update(accessed)
        variants.append(
            {
                "architecture": architecture_match.group(1),
                "parameter_bytes": 224,
                "ld_param_instruction_count": len(matches),
                "accessed_byte_count": len(accessed),
                "accessed_ranges_root_sha256": digest(instructions),
                "byte_156_accessed": 156 in accessed,
                "bytes_157_159_accessed": any(offset in accessed for offset in (157, 158, 159)),
                "byte_160_accessed": 160 in accessed,
            }
        )
    require(
        {variant["architecture"] for variant in variants} == {"sm_89", "sm_120"},
        "module 21 PTX architecture set drifted",
    )
    require(
        all(
            variant["byte_156_accessed"]
            and not variant["bytes_157_159_accessed"]
            and variant["byte_160_accessed"]
            for variant in variants
        ),
        "module 21 padding boundary is not unused on every selected architecture",
    )
    unused = set(range(224)) - union_accessed
    public = {
        "module_sha256": MODULE21_SHA256,
        "ptx_input_receipt_sha256": digest_file(receipt_path),
        "ptx_sha256": digest_file(ptx_path),
        "ptx_bytes": ptx_path.stat().st_size,
        "variants": variants,
        "unused_by_all_variants_count": len(unused),
        "unused_by_all_variants_root_sha256": digest(sorted(unused)),
        "padding_157_159_unused_by_all_variants": {157, 158, 159}.issubset(unused),
    }
    public["proof_root_sha256"] = digest(public)
    return PaddingProof(public=public, unused_by_all_variants=unused)


def canonical_allocation_pointer_bytes(token: str, offset: int) -> bytes:
    return hashlib.sha256(
        b"neuralstudio.fg.allocation-pointer.v1\0"
        + token.encode("ascii")
        + b"\0"
        + str(offset).encode("ascii")
    ).digest()[:8]


def normalize_allocation_pointers(
    data: bytes | None, allocation: AllocationRun
) -> dict[str, Any]:
    if data is None:
        return {
            "data": None,
            "sha256": None,
            "substitutions": [],
            "ambiguities": [],
            "unaligned_candidates": [],
        }
    substitutions: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    unaligned: list[dict[str, Any]] = []
    normalized = bytearray(data)
    for parameter_offset in range(0, max(0, len(data) - 7)):
        pointer = int.from_bytes(data[parameter_offset : parameter_offset + 8], "little")
        hits = [
            (token, pointer - start)
            for start, end, token in allocation.buffer_ranges
            if start <= pointer < end
        ]
        if not hits:
            continue
        if parameter_offset % 8 != 0:
            unaligned.append(
                {
                    "parameter_offset": parameter_offset,
                    "candidate_count": len(hits),
                    "result": "not_normalized_without_8_byte_cuda_abi_alignment",
                }
            )
            continue
        if len(hits) != 1:
            ambiguities.append(
                {
                    "parameter_offset": parameter_offset,
                    "candidate_allocation_tokens": sorted(token for token, _ in hits),
                    "result": "multiple_live_allocations_contain_pointer",
                }
            )
            continue
        token, allocation_offset = hits[0]
        replacement = canonical_allocation_pointer_bytes(token, allocation_offset)
        normalized[parameter_offset : parameter_offset + 8] = replacement
        substitutions.append(
            {
                "parameter_offset": parameter_offset,
                "width": 8,
                "allocation_token": token,
                "allocation_offset": allocation_offset,
                "lifetime_contained": allocation.lifetime_contained,
                "canonical_replacement_sha256": hashlib.sha256(replacement).hexdigest(),
            }
        )
    normalized_bytes = bytes(normalized)
    return {
        "data": normalized_bytes,
        "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "substitutions": substitutions,
        "ambiguities": ambiguities,
        "unaligned_candidates": unaligned,
    }


def module_sha_by_token(parsed: Any) -> dict[str, str]:
    return {
        module["encounter_token"]: module["blob_sha256"]
        for module in parsed.modules
        if module.get("object_kind") == "cu_module"
    }


def compare_canonical_parameters(
    parsed_a: Any,
    allocation_a: AllocationRun,
    parsed_b: Any,
    allocation_b: AllocationRun,
    padding: PaddingProof,
) -> dict[str, Any]:
    count = max(len(parsed_a.launches), len(parsed_b.launches))
    modules_a = module_sha_by_token(parsed_a)
    modules_b = module_sha_by_token(parsed_b)
    comparisons: list[dict[str, Any]] = []
    exact_before_padding = 0
    exact_after_padding = 0
    pointer_substitutions_a = 0
    pointer_substitutions_b = 0
    pointer_ambiguities = 0
    unaligned_candidates_a = 0
    unaligned_candidates_b = 0
    padding_substitution_count = 0
    for ordinal in range(count):
        launch_a = parsed_a.launches[ordinal] if ordinal < len(parsed_a.launches) else None
        launch_b = parsed_b.launches[ordinal] if ordinal < len(parsed_b.launches) else None
        raw_a = parsed_a.parameter_data[ordinal] if ordinal < len(parsed_a.parameter_data) else None
        raw_b = parsed_b.parameter_data[ordinal] if ordinal < len(parsed_b.parameter_data) else None
        handle_a = nvapi.normalize_parameter_handles(
            raw_a,
            launch_a.get("phase") if launch_a else None,
            parsed_a.volatile,
        )
        handle_b = nvapi.normalize_parameter_handles(
            raw_b,
            launch_b.get("phase") if launch_b else None,
            parsed_b.volatile,
        )
        pointer_a = normalize_allocation_pointers(handle_a["normalized_data"], allocation_a)
        pointer_b = normalize_allocation_pointers(handle_b["normalized_data"], allocation_b)
        pointer_substitutions_a += len(pointer_a["substitutions"])
        pointer_substitutions_b += len(pointer_b["substitutions"])
        pointer_ambiguities += len(pointer_a["ambiguities"]) + len(pointer_b["ambiguities"])
        unaligned_candidates_a += len(pointer_a["unaligned_candidates"])
        unaligned_candidates_b += len(pointer_b["unaligned_candidates"])
        data_a = pointer_a["data"]
        data_b = pointer_b["data"]
        before_offsets = (
            nvapi.differing_offsets(data_a, data_b)
            if data_a is not None and data_b is not None
            else []
        )
        if data_a is not None and data_a == data_b:
            exact_before_padding += 1
        canonical_a = bytearray(data_a) if data_a is not None else None
        canonical_b = bytearray(data_b) if data_b is not None else None
        padding_substitutions: list[dict[str, Any]] = []
        if canonical_a is not None and canonical_b is not None and launch_a and launch_b:
            module_a = modules_a.get(launch_a.get("module_token"))
            module_b = modules_b.get(launch_b.get("module_token"))
            for offset in before_offsets:
                if (
                    module_a == module_b == MODULE21_SHA256
                    and len(canonical_a) == len(canonical_b) == 224
                    and offset in padding.unused_by_all_variants
                ):
                    canonical_a[offset] = 0
                    canonical_b[offset] = 0
                    padding_substitutions.append(
                        {
                            "offset": offset,
                            "module_sha256": MODULE21_SHA256,
                            "padding_proof_root_sha256": padding.public[
                                "proof_root_sha256"
                            ],
                            "unused_by_all_selected_architectures": True,
                        }
                    )
                    padding_substitution_count += 1
        final_a = bytes(canonical_a) if canonical_a is not None else None
        final_b = bytes(canonical_b) if canonical_b is not None else None
        after_offsets = (
            nvapi.differing_offsets(final_a, final_b)
            if final_a is not None and final_b is not None
            else []
        )
        if final_a is not None and final_a == final_b:
            exact_after_padding += 1
        comparisons.append(
            {
                "launch_ordinal": ordinal,
                "phase": launch_a.get("phase") if launch_a else None,
                "module_token": launch_a.get("module_token") if launch_a else None,
                "function_token": launch_a.get("function_token") if launch_a else None,
                "function_name": launch_a.get("function_name") if launch_a else None,
                "parameter_bytes_a": len(raw_a) if raw_a is not None else None,
                "parameter_bytes_b": len(raw_b) if raw_b is not None else None,
                "raw_sha256_a": hashlib.sha256(raw_a).hexdigest() if raw_a is not None else None,
                "raw_sha256_b": hashlib.sha256(raw_b).hexdigest() if raw_b is not None else None,
                "cuda_handle_substitutions_a": handle_a["substitutions"],
                "cuda_handle_substitutions_b": handle_b["substitutions"],
                "allocation_pointer_substitutions_a": pointer_a["substitutions"],
                "allocation_pointer_substitutions_b": pointer_b["substitutions"],
                "pointer_ambiguities_a": pointer_a["ambiguities"],
                "pointer_ambiguities_b": pointer_b["ambiguities"],
                "unaligned_gpu_va_candidates_a": pointer_a["unaligned_candidates"],
                "unaligned_gpu_va_candidates_b": pointer_b["unaligned_candidates"],
                "differing_offsets_before_padding_proof": before_offsets,
                "differing_runs_before_padding_proof": nvapi.contiguous_runs(before_offsets),
                "padding_substitutions": padding_substitutions,
                "differing_offsets_after_all_provenance": after_offsets,
                "differing_runs_after_all_provenance": nvapi.contiguous_runs(after_offsets),
                "canonical_sha256_a": hashlib.sha256(final_a).hexdigest() if final_a is not None else None,
                "canonical_sha256_b": hashlib.sha256(final_b).hexdigest() if final_b is not None else None,
                "canonical_equal": final_a is not None and final_a == final_b,
                "unmatched_bytes_preserved": True,
            }
        )
    public = {
        "launch_comparison_count": count,
        "exact_after_handle_and_allocation_normalization_count": exact_before_padding,
        "exact_after_padding_proof_count": exact_after_padding,
        "all_address_free_canonical_parameters_exact": count > 0
        and exact_after_padding == count,
        "allocation_pointer_substitution_count_a": pointer_substitutions_a,
        "allocation_pointer_substitution_count_b": pointer_substitutions_b,
        "pointer_ambiguity_count": pointer_ambiguities,
        "unaligned_gpu_va_candidate_count_a": unaligned_candidates_a,
        "unaligned_gpu_va_candidate_count_b": unaligned_candidates_b,
        "padding_substitution_count": padding_substitution_count,
        "allocation_pointer_bindings_repeat": all(
            comparison["allocation_pointer_substitutions_a"]
            == comparison["allocation_pointer_substitutions_b"]
            for comparison in comparisons
        ),
        "pointer_alignment_policy_bytes": 8,
        "unmatched_bytes_preserved": True,
        "comparisons": comparisons,
    }
    public["comparison_root_sha256"] = digest(comparisons)
    return public


def non_gpu_va_resource_census(parsed: Any) -> list[dict[str, Any]]:
    if parsed.volatile is None:
        return []
    return [
        {
            "phase": record["phase"],
            "resource_token": record["resource_token"],
            "resource_kind": record["resource_kind"],
            "call": record["call"],
            "provenance_result": "cuda_texture_or_surface_object_not_proven_by_d3d12_buffer_gpu_va",
        }
        for record in parsed.volatile.resources
        if record["status"] == 0 and record["resource_token"] != "untracked"
    ]


def require_launches_precede_release(parsed: Any) -> None:
    phases = parsed.phases
    require("release" in phases, "trace release phase is missing")
    release_index = phases.index("release")
    evaluate_indices = [
        index
        for index, phase in enumerate(phases)
        if re.fullmatch(r"evaluate[._][0-9]+", phase) is not None
    ]
    require(
        bool(evaluate_indices)
        and all(index < release_index for index in evaluate_indices),
        "trace evaluate phases do not precede release",
    )
    phase_index = {phase: index for index, phase in enumerate(phases)}
    require(
        all(
            launch.get("phase") in phase_index
            and phase_index[launch["phase"]] < release_index
            for launch in parsed.launches
        ),
        "a selected launch is not lifetime-contained before release",
    )


def parse_opaque_sentinel_receipt(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(isinstance(value, dict), "opaque sentinel receipt is not an object")
    require(
        value.get("schema") == "neuralstudio.fg.nvapi-selected-graph-repeat.v2"
        and nvapi.rooted(value, "receipt_root_sha256")
        == value.get("receipt_root_sha256")
        and value.get("builder_sha256") == digest_file(NVAPI_REPEAT_PATH),
        "opaque sentinel receipt identity/root mismatch",
    )
    require(value.get("blockers") == [], "opaque sentinel receipt is blocked")
    claims = value.get("claims", {})
    require(
        claims.get("selected_graph_repeat_accepted") is True
        and claims.get("parameter_differences_provenanced") is True,
        "opaque sentinel selected graph was not accepted",
    )
    unknown = value.get("unknown_interface_ids", {})
    require(
        unknown.get("run_a") == []
        and unknown.get("run_b") == []
        and unknown.get("repeat_equal") is True,
        "opaque sentinel still contains undocumented interfaces",
    )
    sentinel = value.get("opaque_interface_sentinel", {})
    require(sentinel.get("repeat_equal") is True, "opaque sentinel roots differ")
    coverage_a = sentinel.get("run_a", {}).get("phase_coverage", [])
    coverage_b = sentinel.get("run_b", {}).get("phase_coverage", [])
    require(coverage_a == coverage_b and bool(coverage_a), "opaque sentinel phase coverage is incomplete")
    require(
        all(
            item.get("all_invocation_counts_zero") is True
            and item.get("resolved_never_invoked_non_model_dependency") is True
            for item in coverage_a
        ),
        "opaque sentinel observed an invocation or missing phase",
    )
    interface_ids = sorted(item["interface_id"] for item in coverage_a)
    return {
        "receipt_file_sha256": digest_file(path),
        "receipt_root_sha256": value["receipt_root_sha256"],
        "structural_graph_root_sha256": value["structural_graph"][
            "root_a_sha256"
        ],
        "interface_ids": interface_ids,
        "phase_coverage": coverage_a,
        "zero_invocation_repeat": True,
        "classification": "resolved_but_never_invoked_non_model_dependency",
    }


def build_receipt(
    run_a_dir: Path,
    trace_a_dir: Path,
    run_b_dir: Path,
    trace_b_dir: Path,
    runner_receipt_path: Path,
    ptx_path: Path,
    ptx_receipt_path: Path,
    opaque_sentinel_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    require(run_a_dir.resolve() != run_b_dir.resolve(), "run directories must be distinct")
    require(trace_a_dir.resolve() != trace_b_dir.resolve(), "trace directories must be distinct")
    host_a = parse_host_run(run_a_dir)
    host_b = parse_host_run(run_b_dir)
    runner = parse_runner_receipt(runner_receipt_path)
    runner_by_name = {item["name"]: item["sha256"] for item in runner["files"]}
    for host in (host_a, host_b):
        require(
            host.runtime_identity["runtime_sha256"]
            == runner_by_name["nvngx_dlssg.dll"]
            and host.runtime_identity["nvapi_proxy_sha256"]
            == runner_by_name["nvapi64.dll"]
            and host.runtime_identity["thunk_sha256"] == runner_by_name["nvngx.dll"],
            "host runtime identity does not match immutable runner",
        )
    require(
        len(
            {
                host_a.process_identity["process_id"],
                host_b.process_identity["process_id"],
            }
        )
        == 2
        and host_a.process_identity["run_nonce"] != host_b.process_identity["run_nonce"]
        and host_a.process_identity["queue_nonce"] != host_b.process_identity["queue_nonce"],
        "fresh-process identity did not change",
    )
    require(
        host_a.process_identity["output_poison_byte"]
        != host_b.process_identity["output_poison_byte"],
        "poison bytes must differ across the fresh-process pair",
    )
    require(
        host_a.output_root_sha256 == host_b.output_root_sha256
        and host_a.expected_baseline_sha256 == host_b.expected_baseline_sha256,
        "output baseline changed across allocation capture",
    )
    allocation_a = parse_allocation_run(
        run_a_dir / "runtime-allocations-semantic.jsonl",
        run_a_dir / "runtime-allocations-volatile.jsonl",
    )
    allocation_b = parse_allocation_run(
        run_b_dir / "runtime-allocations-semantic.jsonl",
        run_b_dir / "runtime-allocations-volatile.jsonl",
    )
    require(
        allocation_a.descriptor_root_sha256 == allocation_b.descriptor_root_sha256,
        "allocation descriptor roots differ",
    )
    require(
        allocation_a.release_set_root_sha256 == allocation_b.release_set_root_sha256,
        "allocation release sets differ",
    )
    parsed_a = nvapi.parse_run(
        "run_a",
        trace_a_dir / "nvapi-trace.jsonl",
        trace_a_dir / "dumps",
        trace_a_dir / "nvapi-volatile.jsonl",
    )
    parsed_b = nvapi.parse_run(
        "run_b",
        trace_b_dir / "nvapi-trace.jsonl",
        trace_b_dir / "dumps",
        trace_b_dir / "nvapi-volatile.jsonl",
    )
    require_launches_precede_release(parsed_a)
    require_launches_precede_release(parsed_b)
    padding = parse_padding_proof(ptx_path, ptx_receipt_path)
    opaque_sentinel = (
        parse_opaque_sentinel_receipt(opaque_sentinel_receipt_path)
        if opaque_sentinel_receipt_path is not None
        else None
    )
    parameters = compare_canonical_parameters(
        parsed_a, allocation_a, parsed_b, allocation_b, padding
    )
    non_gpu_va_a = non_gpu_va_resource_census(parsed_a)
    non_gpu_va_b = non_gpu_va_resource_census(parsed_b)
    non_gpu_va_root_a = digest(non_gpu_va_a)
    non_gpu_va_root_b = digest(non_gpu_va_b)

    blockers = set(parsed_a.blockers + parsed_b.blockers)
    if opaque_sentinel is not None:
        unresolved_interface_ids = sorted(
            {
                record["interface_id"]
                for record in parsed_a.unknown_interfaces
                + parsed_b.unknown_interfaces
            }
        )
        require(
            unresolved_interface_ids == opaque_sentinel["interface_ids"],
            "opaque sentinel interface set does not match allocation traces",
        )
        blockers.discard("interfaces.unknown_ids_present")
    if allocation_a.range_ambiguities or allocation_b.range_ambiguities:
        blockers.add("allocation.overlapping_live_gpu_va_spans")
    if parameters["pointer_ambiguity_count"]:
        blockers.add("allocation.pointer_match_ambiguity")
    if not parameters["all_address_free_canonical_parameters_exact"]:
        blockers.add("parameters.unresolved_after_all_provenance")
    if parameters["allocation_pointer_substitution_count_a"] != parameters[
        "allocation_pointer_substitution_count_b"
    ]:
        blockers.add("allocation.pointer_substitution_count_mismatch")
    if not parameters["allocation_pointer_bindings_repeat"]:
        blockers.add("allocation.pointer_bindings_mismatch")
    if non_gpu_va_root_a != non_gpu_va_root_b:
        blockers.add("resources.non_gpu_va_census_mismatch")
    allocation_gate = (
        not allocation_a.range_ambiguities
        and not allocation_b.range_ambiguities
        and parameters["pointer_ambiguity_count"] == 0
        and parameters["allocation_pointer_substitution_count_a"]
        == parameters["allocation_pointer_substitution_count_b"]
        and parameters["allocation_pointer_bindings_repeat"]
        and allocation_a.descriptor_root_sha256
        == allocation_b.descriptor_root_sha256
        and allocation_a.release_set_root_sha256
        == allocation_b.release_set_root_sha256
        and host_a.output_root_sha256 == host_b.output_root_sha256
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "builder_sha256": digest_file(Path(__file__).resolve()),
        "nvapi_repeat_dependency_sha256": digest_file(NVAPI_REPEAT_PATH),
        "immutable_runner": runner,
        "fresh_process": {
            "run_a": host_a.process_identity,
            "run_b": host_b.process_identity,
            "process_ids_distinct": True,
            "run_nonces_distinct": True,
            "queue_nonces_distinct": True,
            "poison_bytes_distinct": True,
        },
        "inputs": {
            "run_a": {
                "host_receipt": host_a.input_receipt,
                "semantic_allocation_ledger": allocation_a.semantic_input,
                "volatile_allocation_ledger": allocation_a.volatile_input,
                "nvapi_trace_sha256": parsed_a.input_receipt["trace_sha256"],
                "nvapi_volatile_trace_sha256": parsed_a.input_receipt[
                    "volatile_trace"
                ]["trace_sha256"],
                "dump_inventory_root_sha256": parsed_a.input_receipt[
                    "dump_inventory"
                ]["inventory_root_sha256"],
            },
            "run_b": {
                "host_receipt": host_b.input_receipt,
                "semantic_allocation_ledger": allocation_b.semantic_input,
                "volatile_allocation_ledger": allocation_b.volatile_input,
                "nvapi_trace_sha256": parsed_b.input_receipt["trace_sha256"],
                "nvapi_volatile_trace_sha256": parsed_b.input_receipt[
                    "volatile_trace"
                ]["trace_sha256"],
                "dump_inventory_root_sha256": parsed_b.input_receipt[
                    "dump_inventory"
                ]["inventory_root_sha256"],
            },
        },
        "output_baseline": {
            "run_a": host_a.output_sequence,
            "run_b": host_b.output_sequence,
            "root_a_sha256": host_a.output_root_sha256,
            "root_b_sha256": host_b.output_root_sha256,
            "expected_baseline_sha256": host_a.expected_baseline_sha256,
            "repeat_equal": True,
            "baseline_unchanged": True,
        },
        "allocation_descriptors": {
            "run_a": allocation_a.descriptors,
            "run_b": allocation_b.descriptors,
            "root_a_sha256": allocation_a.descriptor_root_sha256,
            "root_b_sha256": allocation_b.descriptor_root_sha256,
            "repeat_equal": True,
            "allocation_count": len(allocation_a.descriptors),
            "buffer_gpu_va_span_count": len(allocation_a.buffer_ranges),
            "texture_zero_gpu_va_count": len(allocation_a.zero_gpu_va_resources),
        },
        "lifetime": {
            "model": "all_callback_allocations_precede_all_callback_releases_and_trace_launches_execute_before_feature_release",
            "run_a_contained": allocation_a.lifetime_contained,
            "run_b_contained": allocation_b.lifetime_contained,
            "release_set_root_a_sha256": allocation_a.release_set_root_sha256,
            "release_set_root_b_sha256": allocation_b.release_set_root_sha256,
            "release_sets_equal": True,
            "release_order_root_a_sha256": allocation_a.release_order_root_sha256,
            "release_order_root_b_sha256": allocation_b.release_order_root_sha256,
            "release_order_equal": allocation_a.release_order
            == allocation_b.release_order,
            "release_order_not_required_for_pointer_identity": True,
            "range_ambiguities_a": allocation_a.range_ambiguities,
            "range_ambiguities_b": allocation_b.range_ambiguities,
        },
        "zero_gpu_va_resources": {
            "run_a": allocation_a.zero_gpu_va_resources,
            "run_b": allocation_b.zero_gpu_va_resources,
            "result": "texture allocations cannot be normalized as D3D12 buffer GPU virtual addresses",
        },
        "cuda_object_handles_not_proven_by_gpu_va": {
            "run_a": non_gpu_va_a,
            "run_b": non_gpu_va_b,
            "root_a_sha256": non_gpu_va_root_a,
            "root_b_sha256": non_gpu_va_root_b,
            "repeat_equal": non_gpu_va_root_a == non_gpu_va_root_b,
            "capture_uav_info_zero_span_count_a": sum(
                record["provenance_result"] == "zero_span_no_provenance"
                for record in parsed_a.volatile.uav_info
            ),
            "capture_uav_info_zero_span_count_b": sum(
                record["provenance_result"] == "zero_span_no_provenance"
                for record in parsed_b.volatile.uav_info
            ),
            "capture_uav_info_zero_spans_are_provenance": False,
        },
        "module21_unused_padding_proof": padding.public,
        "opaque_interface_noninvocation_proof": opaque_sentinel,
        "canonical_parameters": parameters,
        "upstream_nvapi_blockers": sorted(
            set(parsed_a.blockers + parsed_b.blockers)
        ),
        "blockers": sorted(blockers),
        "claims": {
            "output_baseline_unchanged": True,
            "allocation_descriptor_repeat": True,
            "allocation_lifetime_containment": True,
            "allocation_gpu_va_provenance_accepted": allocation_gate,
            "all_address_free_canonical_parameters_exact": parameters[
                "all_address_free_canonical_parameters_exact"
            ],
            "texture_cuda_handles_proven_by_gpu_va": False,
            "whole_intake_acceptance": False,
        },
        "receipt_root_sha256": None,
    }
    receipt["receipt_root_sha256"] = rooted(receipt, "receipt_root_sha256")
    status = (
        0
        if allocation_gate
        and parameters["all_address_free_canonical_parameters_exact"]
        and not blockers
        else 1
    )
    return receipt, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a-dir", type=Path, required=True)
    parser.add_argument("--trace-a-dir", type=Path, required=True)
    parser.add_argument("--run-b-dir", type=Path, required=True)
    parser.add_argument("--trace-b-dir", type=Path, required=True)
    parser.add_argument("--runner-receipt", type=Path, required=True)
    parser.add_argument("--module21-ptx", type=Path, required=True)
    parser.add_argument("--module21-ptx-receipt", type=Path, required=True)
    parser.add_argument("--opaque-sentinel-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-receipt", action="store_true")
    args = parser.parse_args()
    try:
        receipt, status = build_receipt(
            args.run_a_dir,
            args.trace_a_dir,
            args.run_b_dir,
            args.trace_b_dir,
            args.runner_receipt,
            args.module21_ptx,
            args.module21_ptx_receipt,
            args.opaque_sentinel_receipt,
        )
    except (OSError, json.JSONDecodeError, AllocationError, nvapi.RepeatError) as error:
        print(f"FG allocation provenance build failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        print(f"FG allocation provenance output already exists: {args.output}", file=sys.stderr)
        return 2
    if args.print_receipt:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "receipt_root_sha256": receipt["receipt_root_sha256"],
                    "allocation_gpu_va_provenance_accepted": receipt["claims"][
                        "allocation_gpu_va_provenance_accepted"
                    ],
                    "canonical_parameter_exact_count": receipt[
                        "canonical_parameters"
                    ]["exact_after_padding_proof_count"],
                    "launch_comparison_count": receipt["canonical_parameters"][
                        "launch_comparison_count"
                    ],
                    "allocation_pointer_substitutions": receipt[
                        "canonical_parameters"
                    ]["allocation_pointer_substitution_count_a"],
                    "blockers": receipt["blockers"],
                },
                sort_keys=True,
            )
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
