#!/usr/bin/env python3
"""Verify repeat-exact FG runtime buffer extraction without weight overclaim."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
ALLOCATION_VERIFIER = REPOSITORY / "scripts" / "build-fg-allocation-provenance.py"
RECEIPT_SCHEMA = (
    REPOSITORY
    / "evidence"
    / "specs"
    / "fg-runtime-buffer-extraction-v1"
    / "receipt.schema.json"
)
SCHEMA = "neuralstudio.fg.runtime-buffer-extraction.v1"
EXPECTED_DEFAULT_TOKENS = list(range(63, 96, 2)) + list(range(106, 205, 2))
EXPECTED_UPLOAD_TOKENS = [token + 1 for token in EXPECTED_DEFAULT_TOKENS]
EXPECTED_BUFFER_COUNT = 67
EXPECTED_LOGICAL_BYTES = 4_868_576
EXPECTED_ALLOCATION_COUNT = 205
EXPECTED_ALLOCATION_RECEIPT_FILE_SHA256 = (
    "77befd00cd65064400b1a813f56a9f0ab0a2657f77d35c77c21190673b33de54"
)
EXPECTED_ALLOCATION_RECEIPT_ROOT_SHA256 = (
    "8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c"
)
EXPECTED_ALLOCATION_DESCRIPTOR_ROOT_SHA256 = (
    "60826d12af43c5830c7525b0cba6512253c64334006d4344d4b4c3bfeb086865"
)
EXPECTED_RUNTIME_SHA256 = (
    "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"
)
EXPECTED_BASELINE_OUTPUT_SHA256 = (
    "07a87435b2e7d79540dcec737b9abb6354e68e8595b4df29ca6f8d92cce54383"
)
EXPECTED_UPLOAD_COUNTEREXAMPLE_FILE_SHA256 = (
    "fd34f5ee8511c9a6485f534d3fd6dd59c8a3a95e2cf61b6ee5bc406674e7a275"
)
EXPECTED_UPLOAD_COUNTEREXAMPLE_ROOT_SHA256 = (
    "c5d0b367fd2163a5441a042792579edafe3439cde8071675e8cb36d9f58f8c27"
)
EXPECTED_DEFAULT_RUNNER_MANIFEST_SHA256 = (
    "bf0f92dc678f5d31afe6487a092a6a302f5523fe539836510dbea776b5f2c6b5"
)
EXPECTED_DEFAULT_SOURCE_MANIFEST_SHA256 = (
    "7ee66537ec7da379cb73e1ecfb3ee420b12c2de427d5802f667b0e2550a9e235"
)
EXPECTED_DEFAULT_JOB_ID = "fg-default-readback-20260904T095705Z"
EXPECTED_UPLOAD_JOB_ID = "fg-upload-capture-20260904T094658Z"
EXPECTED_DEFAULT_LOCAL_COMMIT = "6c9f9934a978823f33b0b0df8d5bbd7a23040513"


class ExtractionError(RuntimeError):
    pass


def load_allocation_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fg_allocation_for_extraction", ALLOCATION_VERIFIER
    )
    if spec is None or spec.loader is None:
        raise ExtractionError("cannot load allocation provenance dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


allocation = load_allocation_module()


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


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def rooted(value: dict[str, Any], field: str) -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return digest(body)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionError(message)


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ExtractionError(f"invalid JSON in {path}: {error}") from error


def safe_dump(root: Path, name: Any) -> Path:
    require(isinstance(name, str) and bool(name), "dump name is missing")
    pure = PurePath(name)
    require(
        not pure.is_absolute()
        and len(pure.parts) == 1
        and pure.name == name
        and "/" not in name
        and "\\" not in name,
        f"unsafe dump name: {name}",
    )
    path = root / name
    require(path.is_file() and not path.is_symlink(), f"dump is missing or unsafe: {name}")
    return path


def descriptor_map(allocation_receipt: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for descriptor in allocation_receipt["allocation_descriptors"]["run_a"]:
        token_text = descriptor.get("allocation_token")
        require(
            isinstance(token_text, str) and token_text.startswith("allocation-"),
            "frozen allocation token is invalid",
        )
        token = int(token_text.split("-")[-1])
        require(token not in result, "duplicate frozen allocation token")
        result[token] = descriptor
    return result


def load_frozen_allocation_receipt(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    value = load_json(path)
    require(isinstance(value, dict), "allocation provenance receipt is invalid")
    file_sha256 = digest_file(path)
    require(
        value.get("schema") == "neuralstudio.fg.allocation-provenance-repeat.v1"
        and file_sha256 == EXPECTED_ALLOCATION_RECEIPT_FILE_SHA256
        and value.get("receipt_root_sha256")
        == EXPECTED_ALLOCATION_RECEIPT_ROOT_SHA256
        and allocation.rooted(value, "receipt_root_sha256")
        == value.get("receipt_root_sha256")
        and value.get("builder_sha256") == digest_file(ALLOCATION_VERIFIER)
        and value.get("blockers") == []
        and value.get("claims", {}).get("allocation_gpu_va_provenance_accepted")
        is True,
        "frozen allocation provenance receipt is not accepted/current",
    )
    output_baseline = value.get("output_baseline")
    require(
        isinstance(output_baseline, dict)
        and output_baseline.get("baseline_unchanged") is True
        and output_baseline.get("repeat_equal") is True
        and is_sha256(output_baseline.get("expected_baseline_sha256")),
        "frozen allocation output baseline is not accepted",
    )
    require(
        value["allocation_descriptors"].get("root_a_sha256")
        == value["allocation_descriptors"].get("root_b_sha256")
        == EXPECTED_ALLOCATION_DESCRIPTOR_ROOT_SHA256
        and next(
            item.get("sha256")
            for item in value["immutable_runner"]["files"]
            if item.get("name") == "nvngx_dlssg.dll"
        )
        == EXPECTED_RUNTIME_SHA256
        and output_baseline["expected_baseline_sha256"]
        == EXPECTED_BASELINE_OUTPUT_SHA256,
        "frozen allocation identity drift",
    )
    return (
        {
            "file_sha256": file_sha256,
            "receipt_root_sha256": value["receipt_root_sha256"],
            "descriptor_root_sha256": value["allocation_descriptors"][
                "root_a_sha256"
            ],
            "runtime_sha256": next(
                item["sha256"]
                for item in value["immutable_runner"]["files"]
                if item["name"] == "nvngx_dlssg.dll"
            ),
            "allocation_count": value["allocation_descriptors"][
                "allocation_count"
            ],
            "expected_baseline_output_sha256": output_baseline[
                "expected_baseline_sha256"
            ],
        },
        descriptor_map(value),
    )


@dataclass
class BufferCapture:
    input: dict[str, Any]
    capture_point: str
    entries: list[dict[str, Any]]
    entries_root_sha256: str
    total_nonzero_bytes: int
    zero_tokens: list[int]
    nonzero_tokens: list[int]
    all_zero: bool
    all_nonzero: bool


UPLOAD_FIELDS = {
    "token",
    "dump_file",
    "heap_class",
    "heap_type",
    "cpu_page_property",
    "memory_pool_preference",
    "descriptor_dimension",
    "descriptor_width",
    "initial_state",
    "logical_bytes",
    "allocation_bytes",
    "sha256",
}
DEFAULT_FIELDS = {
    "source_token",
    "dump_file",
    "descriptor_dimension",
    "descriptor_width",
    "source_heap_type",
    "source_initial_state",
    "logical_bytes",
    "sha256",
}


def parse_buffer_capture(
    manifest_path: Path,
    dump_dir: Path,
    kind: str,
    descriptors: dict[int, dict[str, Any]],
    expected_capture_point: str,
) -> BufferCapture:
    require(
        dump_dir.is_dir() and not dump_dir.is_symlink(),
        f"buffer dump directory is missing or unsafe: {dump_dir}",
    )
    manifest = load_json(manifest_path)
    require(isinstance(manifest, dict), "buffer manifest is not an object")
    if kind == "upload":
        require(
            set(manifest)
            == {
                "schema",
                "capture_point",
                "captured_buffers",
                "captured_logical_bytes",
                "default_heap_read",
                "entries",
                "privacy",
            }
            and manifest.get("schema")
            == "neuralstudio.dlss_fg.runtime_uploads.v1"
            and manifest.get("default_heap_read") is False,
            "upload manifest contract drift",
        )
        expected_tokens = EXPECTED_UPLOAD_TOKENS
        fields = UPLOAD_FIELDS
    else:
        require(
            set(manifest)
            == {
                "schema",
                "capture_point",
                "captured_buffers",
                "captured_logical_bytes",
                "source_heap",
                "direct_default_map",
                "copy_submission",
                "source_state_strategy",
                "explicit_source_transition",
                "bytes_fed_back_to_execution",
                "entries",
                "privacy",
            }
            and manifest.get("schema")
            == "neuralstudio.dlss_fg.default_buffer_readback.v1"
            and manifest.get("source_heap") == "default"
            and manifest.get("direct_default_map") is False
            and manifest.get("copy_submission") == "dedicated"
            and manifest.get("source_state_strategy")
            == "implicit_common_to_copy_source_promotion_and_decay"
            and manifest.get("explicit_source_transition") is False
            and manifest.get("bytes_fed_back_to_execution") is False,
            "DEFAULT readback manifest contract drift",
        )
        expected_tokens = EXPECTED_DEFAULT_TOKENS
        fields = DEFAULT_FIELDS
    require(
        manifest.get("capture_point") == expected_capture_point,
        "buffer capture point mismatch",
    )
    require(
        manifest.get("captured_buffers") == EXPECTED_BUFFER_COUNT
        and manifest.get("captured_logical_bytes") == EXPECTED_LOGICAL_BYTES
        and manifest.get("privacy") == "private_local_evidence",
        "buffer manifest count/size/privacy mismatch",
    )
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == EXPECTED_BUFFER_COUNT, "buffer entries are incomplete")
    public_entries: list[dict[str, Any]] = []
    actual_files: set[str] = set()
    tokens: list[int] = []
    total_nonzero = 0
    zero_tokens: list[int] = []
    nonzero_tokens: list[int] = []
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict) and set(entry) == fields, f"buffer entry fields drifted: {index}")
        token = entry["token"] if kind == "upload" else entry["source_token"]
        require(isinstance(token, int), "buffer token is invalid")
        tokens.append(token)
        dump_path = safe_dump(dump_dir, entry["dump_file"])
        expected_name = (
            f"upload-token-{token:06d}.bin"
            if kind == "upload"
            else f"default-token-{token:06d}.bin"
        )
        require(entry["dump_file"] == expected_name, "buffer dump/token naming mismatch")
        actual_files.add(entry["dump_file"])
        data = dump_path.read_bytes()
        require(
            isinstance(entry["logical_bytes"], int)
            and entry["logical_bytes"] > 0
            and len(data) == entry["logical_bytes"]
            and is_sha256(entry["sha256"])
            and digest_file(dump_path) == entry["sha256"],
            "buffer dump size/hash mismatch",
        )
        default_token = token - 1 if kind == "upload" else token
        require(default_token in descriptors, "buffer token is absent from frozen allocation descriptors")
        descriptor = descriptors[default_token]
        require(
            descriptor.get("resource_class") == "buffer"
            and descriptor.get("dimension") == entry["descriptor_dimension"] == 1
            and descriptor.get("width")
            == entry["descriptor_width"]
            == entry["logical_bytes"],
            "buffer token/descriptor pairing mismatch",
        )
        if kind == "upload":
            require(
                token in descriptors,
                "upload token is absent from frozen allocation descriptors",
            )
            upload_descriptor = descriptors[token]
            require(
                entry["heap_class"] == "upload"
                and entry["heap_type"] == 2
                and entry["cpu_page_property"]
                == upload_descriptor.get("cpu_page_property")
                and entry["memory_pool_preference"]
                == upload_descriptor.get("memory_pool_preference")
                and entry["initial_state"]
                == upload_descriptor.get("initial_state")
                and entry["allocation_bytes"]
                == upload_descriptor.get("allocation_size")
                and upload_descriptor.get("resource_class") == "buffer"
                and upload_descriptor.get("dimension")
                == entry["descriptor_dimension"]
                and upload_descriptor.get("width")
                == entry["descriptor_width"]
                and upload_descriptor.get("heap_type") == 2
                and descriptor.get("heap_type") == 1,
                "upload/default adjacency pairing mismatch",
            )
        else:
            require(
                entry["source_heap_type"] == descriptor.get("heap_type") == 1
                and entry["source_initial_state"]
                == descriptor.get("initial_state")
                == 0,
                "DEFAULT source descriptor mismatch",
            )
        nonzero = sum(byte != 0 for byte in data)
        total_nonzero += nonzero
        (nonzero_tokens if nonzero else zero_tokens).append(token)
        public_entries.append(
            {
                "token": token,
                "paired_default_token": default_token,
                "logical_bytes": len(data),
                "sha256": entry["sha256"],
                "nonzero_bytes": nonzero,
                "all_bytes_zero": nonzero == 0,
                "capture_class": kind,
            }
        )
    require(tokens == expected_tokens, "buffer token set/order mismatch")
    require(
        sum(entry["logical_bytes"] for entry in public_entries)
        == EXPECTED_LOGICAL_BYTES,
        "buffer entry logical-byte total mismatch",
    )
    inventory = {
        path.name
        for path in dump_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    require(inventory == actual_files, "buffer dump inventory mismatch")
    return BufferCapture(
        input={
            "manifest_sha256": digest_file(manifest_path),
            "manifest_bytes": manifest_path.stat().st_size,
            "dump_inventory_root_sha256": digest(public_entries),
        },
        capture_point=expected_capture_point,
        entries=public_entries,
        entries_root_sha256=digest(public_entries),
        total_nonzero_bytes=total_nonzero,
        zero_tokens=zero_tokens,
        nonzero_tokens=nonzero_tokens,
        all_zero=len(zero_tokens) == EXPECTED_BUFFER_COUNT,
        all_nonzero=len(nonzero_tokens) == EXPECTED_BUFFER_COUNT,
    )


@dataclass
class HostRun:
    input: dict[str, Any]
    identity: dict[str, Any]
    runtime_identity: dict[str, Any]
    output: dict[str, Any]
    allocation_root_sha256: str


def parse_host_run(
    run_dir: Path,
    manifest_filename: str,
    manifest_sha_field: str,
    frozen: dict[str, Any],
    require_local_allocation_ledgers: bool = True,
    require_local_output_artifact: bool = True,
) -> HostRun:
    receipt_path = run_dir / "fg-run-receipt.json"
    receipt = load_json(receipt_path)
    require(
        isinstance(receipt, dict)
        and receipt.get("schema")
        == "neuralstudio.dlss_fg.standalone_d3d12.v1"
        and receipt.get("accepted") is True
        and receipt.get("d3d12_severe_messages") == 0,
        "host receipt is rejected or has D3D12 severe messages",
    )
    process_id = receipt.get("process_id")
    run_nonce = receipt.get("run_nonce")
    queue_nonce = receipt.get("queue_nonce")
    require(
        isinstance(process_id, int)
        and process_id > 0
        and isinstance(run_nonce, str)
        and len(run_nonce) >= 16
        and isinstance(queue_nonce, str)
        and len(queue_nonce) >= 16
        and receipt.get("output_poison_byte") in {0x5A, 0xA5},
        "host process identity is invalid",
    )
    require(receipt.get("runtime_sha256") == frozen["runtime_sha256"], "runtime hash drift")
    require(
        is_sha256(receipt.get("runtime_sha256"))
        and is_sha256(receipt.get("nvapi_proxy_sha256"))
        and is_sha256(receipt.get("thunk_sha256")),
        "host runtime identity hash is invalid",
    )
    capture = receipt.get("runtime_allocation_capture")
    require(
        isinstance(capture, dict)
        and capture.get("enabled") is True
        and capture.get("healthy") is True
        and capture.get("baseline_output_match") is True
        and capture.get("allocations")
        == capture.get("releases")
        == EXPECTED_ALLOCATION_COUNT
        and capture.get("leaks") == 0,
        "host allocation capture is unhealthy",
    )
    manifest_path = run_dir / manifest_filename
    require(
        capture.get(manifest_sha_field) == digest_file(manifest_path),
        "host buffer-manifest hash mismatch",
    )
    require(
        capture.get("semantic_ledger") == "runtime-allocations-semantic.jsonl"
        and capture.get("volatile_ledger")
        == "runtime-allocations-volatile.jsonl",
        "host allocation-ledger paths drifted",
    )
    semantic = run_dir / "runtime-allocations-semantic.jsonl"
    volatile = run_dir / "runtime-allocations-volatile.jsonl"
    if require_local_allocation_ledgers:
        require(
            digest_file(semantic) == capture["semantic_ledger_sha256"]
            and digest_file(volatile) == capture["volatile_ledger_sha256"],
            "host allocation-ledger hash mismatch",
        )
        allocation_run = allocation.parse_allocation_run(semantic, volatile)
        require(
            allocation_run.descriptor_root_sha256
            == frozen["descriptor_root_sha256"],
            "host allocation descriptors drift from frozen provenance",
        )
        allocation_root = allocation_run.descriptor_root_sha256
    else:
        require(
            isinstance(capture.get("semantic_ledger_sha256"), str)
            and isinstance(capture.get("volatile_ledger_sha256"), str),
            "host allocation ledger identities are missing",
        )
        allocation_root = frozen["descriptor_root_sha256"]
    frames = receipt.get("frames")
    require(isinstance(frames, list) and len(frames) >= 2, "host frame receipt is incomplete")
    final = frames[-1]
    require(
        isinstance(final, dict)
        and isinstance(final.get("frame_id"), int)
        and final["frame_id"] >= 1
        and is_sha256(final.get("output_interpolated_sha256")),
        "host final frame identity is invalid",
    )
    baseline = capture.get("expected_baseline_output_sha256")
    require(
        final.get("output_interpolated_sha256") == baseline
        and baseline == frozen["expected_baseline_output_sha256"]
        and final.get("output_untouched_pixels") == 0
        and final.get("generated_output_accepted") is True,
        "host output baseline changed",
    )
    output_path = run_dir / f"output-interpolated-{final['frame_id']}.rgba8"
    if require_local_output_artifact:
        require(
            output_path.is_file()
            and digest_file(output_path)
            == final["output_interpolated_sha256"],
            "host output artifact hash mismatch",
        )
    return HostRun(
        input={
            "receipt_sha256": digest_file(receipt_path),
            "receipt_bytes": receipt_path.stat().st_size,
            "buffer_manifest_sha256": digest_file(manifest_path),
            "semantic_allocation_ledger_sha256": capture[
                "semantic_ledger_sha256"
            ],
            "volatile_allocation_ledger_sha256": capture[
                "volatile_ledger_sha256"
            ],
            "allocation_ledgers_rehashed_locally": require_local_allocation_ledgers,
            "output_artifact_rehashed_locally": require_local_output_artifact,
        },
        identity={
            "process_id": process_id,
            "run_nonce": run_nonce,
            "queue_nonce": queue_nonce,
            "output_poison_byte": receipt.get("output_poison_byte"),
        },
        runtime_identity={
            "runtime_sha256": receipt.get("runtime_sha256"),
            "nvapi_proxy_sha256": receipt.get("nvapi_proxy_sha256"),
            "thunk_sha256": receipt.get("thunk_sha256"),
        },
        output={
            "baseline_sha256": baseline,
            "final_frame_id": final["frame_id"],
            "output_untouched_pixels": 0,
            "generated_output_accepted": True,
        },
        allocation_root_sha256=allocation_root,
    )


def require_fresh_pair(a: HostRun, b: HostRun) -> None:
    require(
        a.identity["process_id"] != b.identity["process_id"]
        and a.identity["run_nonce"] != b.identity["run_nonce"]
        and a.identity["queue_nonce"] != b.identity["queue_nonce"]
        and a.identity["output_poison_byte"] != b.identity["output_poison_byte"],
        "capture pair is not two fresh processes with distinct poisons",
    )
    require(
        {a.identity["output_poison_byte"], b.identity["output_poison_byte"]}
        == {0x5A, 0xA5},
        "capture pair does not use the required complementary poisons",
    )
    require(a.output == b.output, "capture pair output baseline differs")


def verify_upload_counterexample(
    root: Path,
    counterexample_path: Path,
    descriptors: dict[int, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    counterexample = load_json(counterexample_path)
    require(
        isinstance(counterexample, dict)
        and digest_file(counterexample_path)
        == EXPECTED_UPLOAD_COUNTEREXAMPLE_FILE_SHA256
        and counterexample.get("schema")
        == "neuralstudio.dlss_fg.runtime_upload_zero_counterexample.v1"
        and counterexample.get("receipt_root_sha256")
        == EXPECTED_UPLOAD_COUNTEREXAMPLE_ROOT_SHA256
        and rooted(counterexample, "receipt_root_sha256")
        == counterexample.get("receipt_root_sha256")
        and counterexample.get("classification")
        == "failed_weight_recovery_counterexample"
        and counterexample.get("candidate_weights_recovered") is False
        and counterexample.get("capture_point")
        == "post_create_fence_pre_evaluate"
        and counterexample.get("default_heap_read") is False,
        "upload counterexample receipt drift",
    )
    for name, expected in (
        ("build-manifest.json", counterexample.get("build_manifest_sha256")),
        ("runner-manifest.json", counterexample.get("runner_manifest_sha256")),
        ("source-manifest-v2.json", counterexample.get("source_manifest_sha256")),
    ):
        require(
            digest_file(root / "manifests" / name) == expected,
            f"upload counterexample {name} hash mismatch",
        )
    source_manifest = load_json(root / "manifests" / "source-manifest-v2.json")
    upload_runner = load_json(root / "manifests" / "runner-manifest.json")
    require(
        isinstance(source_manifest, dict)
        and source_manifest.get("schema")
        == "neuralstudio.fg_upload_capture.source_stage.v2"
        and source_manifest.get("job_id") == EXPECTED_UPLOAD_JOB_ID
        and source_manifest.get("local_commit") == EXPECTED_DEFAULT_LOCAL_COMMIT
        and isinstance(source_manifest.get("files"), list)
        and bool(source_manifest["files"])
        and isinstance(upload_runner, dict)
        and upload_runner.get("schema")
        == "neuralstudio.fg_upload_capture.immutable_runner.v2"
        and upload_runner.get("source_manifest_sha256")
        == counterexample["source_manifest_sha256"]
        and isinstance(upload_runner.get("files"), list),
        "upload source/runner manifest identity drift",
    )
    upload_runner_by_name = {
        item["name"]: item["sha256"] for item in upload_runner["files"]
    }
    require(
        len(upload_runner_by_name) == len(upload_runner["files"])
        and {
            "nvngx_dlssg.dll",
            "nvapi64.dll",
            "nvngx.dll",
        }.issubset(upload_runner_by_name),
        "upload immutable runner is incomplete",
    )
    parsed: list[BufferCapture] = []
    hosts: list[HostRun] = []
    runs = counterexample.get("runs")
    require(isinstance(runs, list) and len(runs) == 2, "upload counterexample run list is invalid")
    for run_record in runs:
        run_id = run_record["run_id"]
        run_dir = root / run_id
        manifest_name = "runtime-upload-buffers-manifest.json"
        capture = parse_buffer_capture(
            run_dir / manifest_name,
            run_dir / "runtime-upload-buffers",
            "upload",
            descriptors,
            "post_create_fence_pre_evaluate",
        )
        host = parse_host_run(
            run_dir, manifest_name, "upload_manifest_sha256", frozen
        )
        require(
            capture.all_zero
            and not capture.nonzero_tokens
            and run_record["all_bytes_zero"] is True
            and run_record["captured_buffers"] == EXPECTED_BUFFER_COUNT
            and run_record["captured_logical_bytes"]
            == EXPECTED_LOGICAL_BYTES
            and run_record["upload_manifest_sha256"]
            == capture.input["manifest_sha256"]
            and run_record["host_receipt_sha256"]
            == host.input["receipt_sha256"]
            and run_record["output_sha256"]
            == host.output["baseline_sha256"]
            == frozen["expected_baseline_output_sha256"]
            and run_record["output_untouched_pixels"] == 0
            and run_record["poison_byte"]
            == host.identity["output_poison_byte"],
            "upload all-zero counterexample does not match files",
        )
        process_receipt = root / "receipts" / f"{run_id}-process.json"
        process = load_json(process_receipt)
        require(
            digest_file(process_receipt) == run_record["process_receipt_sha256"]
            and isinstance(process, dict)
            and process.get("job_id") == EXPECTED_UPLOAD_JOB_ID
            and process.get("run_id") == run_id
            and process.get("exit_code") == 0,
            "upload process receipt hash mismatch",
        )
        parsed.append(capture)
        hosts.append(host)
    require_fresh_pair(hosts[0], hosts[1])
    for host in hosts:
        require(
            host.runtime_identity
            == {
                "runtime_sha256": upload_runner_by_name["nvngx_dlssg.dll"],
                "nvapi_proxy_sha256": upload_runner_by_name["nvapi64.dll"],
                "thunk_sha256": upload_runner_by_name["nvngx.dll"],
            },
            "upload host runtime components do not match immutable runner",
        )
    require(
        parsed[0].entries == parsed[1].entries
        and counterexample.get("expected_output_sha256")
        == frozen["expected_baseline_output_sha256"]
        and counterexample["summary"]
        == {
            "buffers_per_run": EXPECTED_BUFFER_COUNT,
            "logical_bytes_per_run": EXPECTED_LOGICAL_BYTES,
            "nonzero_tokens": 0,
            "stable_tokens": EXPECTED_BUFFER_COUNT,
            "varying_tokens": [],
            "zero_tokens": EXPECTED_BUFFER_COUNT,
        },
        "upload counterexample manifests/summary drift",
    )
    expected_buffers = [
        {
            "upload_token": entry["token"],
            "adjacent_default_token": entry["paired_default_token"],
            "logical_bytes": entry["logical_bytes"],
            "sha256": entry["sha256"],
            "all_bytes_zero": True,
            "cross_process_stable": True,
            "candidate_weight": False,
        }
        for entry in parsed[0].entries
    ]
    require(
        counterexample.get("buffers") == expected_buffers,
        "upload counterexample buffer ledger drift",
    )
    return {
        "counterexample_file_sha256": digest_file(counterexample_path),
        "counterexample_root_sha256": counterexample["receipt_root_sha256"],
        "capture_point": counterexample["capture_point"],
        "source_identity": {
            "job_id": source_manifest["job_id"],
            "local_commit": source_manifest["local_commit"],
            "source_manifest_sha256": counterexample["source_manifest_sha256"],
            "runner_manifest_sha256": counterexample["runner_manifest_sha256"],
            "build_manifest_sha256": counterexample["build_manifest_sha256"],
        },
        "run_a": {
            "host": hosts[0].input,
            "identity": hosts[0].identity,
            "runtime_identity": hosts[0].runtime_identity,
            "buffers": parsed[0].input,
        },
        "run_b": {
            "host": hosts[1].input,
            "identity": hosts[1].identity,
            "runtime_identity": hosts[1].runtime_identity,
            "buffers": parsed[1].input,
        },
        "buffer_count": EXPECTED_BUFFER_COUNT,
        "logical_bytes": EXPECTED_LOGICAL_BYTES,
        "nonzero_bytes": 0,
        "all_bytes_zero": True,
        "classification": "rejected_weight_recovery_counterexample",
        "candidate_model_weights": False,
    }


def verify_default_capture(
    run_dir: Path,
    manifest_name: str,
    dump_name: str,
    capture_point: str,
    descriptors: dict[int, dict[str, Any]],
    frozen: dict[str, Any],
    require_local_allocation_ledgers: bool = True,
    require_local_output_artifact: bool = True,
) -> tuple[BufferCapture, HostRun]:
    capture = parse_buffer_capture(
        run_dir / manifest_name,
        run_dir / dump_name,
        "default",
        descriptors,
        capture_point,
    )
    host = parse_host_run(
        run_dir,
        manifest_name,
        "default_manifest_sha256",
        frozen,
        require_local_allocation_ledgers,
        require_local_output_artifact,
    )
    return capture, host


def parse_runner_manifest(path: Path, hosts: list[HostRun]) -> dict[str, Any]:
    value = load_json(path)
    require(
        isinstance(value, dict)
        and digest_file(path) == EXPECTED_DEFAULT_RUNNER_MANIFEST_SHA256
        and set(value)
        == {
            "schema",
            "name",
            "created_utc",
            "source_manifest_sha256",
            "files",
        }
        and value.get("schema") == "neuralstudio.fg_default_readback.immutable_runner.v2"
        and isinstance(value.get("name"), str)
        and bool(value["name"])
        and isinstance(value.get("created_utc"), str)
        and value["created_utc"].endswith("Z")
        and isinstance(value.get("files"), list),
        "default readback runner manifest drift",
    )
    files = value["files"]
    require(len(files) >= 5, "runner manifest is incomplete")
    for item in files:
        require(
            isinstance(item, dict)
            and set(item) == {"name", "bytes", "sha256"}
            and isinstance(item["name"], str)
            and bool(item["name"])
            and isinstance(item["bytes"], int)
            and item["bytes"] > 0
            and is_sha256(item["sha256"]),
            "runner manifest file entry drift",
        )
    by_name = {item["name"]: item["sha256"] for item in files}
    require(len(by_name) == len(files), "runner manifest contains duplicate files")
    # Runtime identity is already checked through the frozen allocation receipt.
    require(
        {"nvngx_dlssg.dll", "nvapi64.dll", "nvngx.dll"}.issubset(by_name),
        "runner manifest lacks required runtime components",
    )
    for host in hosts:
        require(
            host.runtime_identity
            == {
                "runtime_sha256": by_name["nvngx_dlssg.dll"],
                "nvapi_proxy_sha256": by_name["nvapi64.dll"],
                "thunk_sha256": by_name["nvngx.dll"],
            },
            "host runtime components do not match immutable runner manifest",
        )
    source_manifest_path = path.parent / "source-manifest-v2.json"
    source_manifest = load_json(source_manifest_path)
    source_manifest_sha256 = value.get("source_manifest_sha256")
    require(
        is_sha256(source_manifest_sha256)
        and source_manifest_sha256 == EXPECTED_DEFAULT_SOURCE_MANIFEST_SHA256
        and digest_file(source_manifest_path) == source_manifest_sha256
        and isinstance(source_manifest, dict)
        and set(source_manifest) == {"schema", "job_id", "local_commit", "files"}
        and source_manifest.get("schema")
        == "neuralstudio.fg_default_readback.source_stage.v2"
        and isinstance(source_manifest.get("job_id"), str)
        and source_manifest["job_id"] == EXPECTED_DEFAULT_JOB_ID
        and isinstance(source_manifest.get("local_commit"), str)
        and source_manifest["local_commit"] == EXPECTED_DEFAULT_LOCAL_COMMIT
        and isinstance(source_manifest.get("files"), list)
        and bool(source_manifest["files"]),
        "default readback source manifest drift",
    )
    return {
        "manifest_sha256": digest_file(path),
        "name": value.get("name"),
        "job_id": source_manifest["job_id"],
        "local_commit": source_manifest["local_commit"],
        "source_manifest_sha256": source_manifest_sha256,
        "files": files,
        "files_root_sha256": digest(files),
    }


def build_receipt(
    allocation_receipt_path: Path,
    upload_root: Path,
    upload_counterexample_path: Path,
    post_create_run: Path,
    eval_run_a: Path,
    eval_run_b: Path,
    runner_manifest_path: Path,
) -> tuple[dict[str, Any], int]:
    frozen, descriptors = load_frozen_allocation_receipt(
        allocation_receipt_path
    )
    require(
        frozen["allocation_count"] == EXPECTED_ALLOCATION_COUNT
        and len(descriptors) == EXPECTED_ALLOCATION_COUNT,
        "frozen allocation descriptor count drift",
    )
    upload = verify_upload_counterexample(
        upload_root, upload_counterexample_path, descriptors, frozen
    )
    post_create, post_host = verify_default_capture(
        post_create_run,
        "runtime-default-buffers-manifest.json",
        "runtime-default-buffers-post-create",
        "post_create_fence_pre_evaluate",
        descriptors,
        frozen,
        False,
        False,
    )
    require(
        post_create.all_zero
        and post_create.entries[0]["token"] == EXPECTED_DEFAULT_TOKENS[0]
        and post_create.entries[0]["all_bytes_zero"] is True,
        "post-create DEFAULT token 63 is not the frozen zero counterexample",
    )
    eval_a, host_a = verify_default_capture(
        eval_run_a,
        "runtime-default-buffers-post-evaluate-0.manifest.json",
        "runtime-default-buffers-post-evaluate-0",
        "post_evaluate_0_fence_pre_evaluate_1",
        descriptors,
        frozen,
    )
    eval_b, host_b = verify_default_capture(
        eval_run_b,
        "runtime-default-buffers-post-evaluate-0.manifest.json",
        "runtime-default-buffers-post-evaluate-0",
        "post_evaluate_0_fence_pre_evaluate_1",
        descriptors,
        frozen,
    )
    require_fresh_pair(host_a, host_b)
    require(
        post_host.identity["process_id"]
        not in {host_a.identity["process_id"], host_b.identity["process_id"]}
        and post_host.identity["run_nonce"]
        not in {host_a.identity["run_nonce"], host_b.identity["run_nonce"]}
        and post_host.identity["queue_nonce"]
        not in {host_a.identity["queue_nonce"], host_b.identity["queue_nonce"]},
        "post-create counterexample was not captured in a distinct process",
    )
    require(
        eval_a.entries == eval_b.entries
        and eval_a.entries_root_sha256 == eval_b.entries_root_sha256,
        "post-evaluate DEFAULT buffer manifests do not repeat exactly",
    )
    require(
        eval_a.all_nonzero
        and eval_b.all_nonzero
        and not eval_a.zero_tokens
        and not eval_b.zero_tokens,
        "post-evaluate DEFAULT buffers are all-zero or partially zero",
    )
    runner = parse_runner_manifest(runner_manifest_path, [host_a, host_b])
    token63_post_create = post_create.entries[0]
    token63_post_eval = eval_a.entries[0]
    require(
        token63_post_create["sha256"] != token63_post_eval["sha256"]
        and token63_post_create["all_bytes_zero"] is True
        and token63_post_eval["all_bytes_zero"] is False,
        "token 63 capture points were conflated",
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "verifier_sha256": digest_file(Path(__file__).resolve()),
        "schema_sha256": digest_file(RECEIPT_SCHEMA),
        "allocation_verifier_dependency_sha256": digest_file(
            ALLOCATION_VERIFIER
        ),
        "frozen_allocation_provenance": frozen,
        "immutable_runner": runner,
        "upload_zero_counterexample": upload,
        "post_create_default_zero_counterexample": {
            "host": post_host.input,
            "identity": post_host.identity,
            "runtime_identity": post_host.runtime_identity,
            "capture": post_create.input,
            "capture_point": post_create.capture_point,
            "entries_root_sha256": post_create.entries_root_sha256,
            "buffer_count": len(post_create.entries),
            "logical_bytes": sum(
                entry["logical_bytes"] for entry in post_create.entries
            ),
            "nonzero_bytes": 0,
            "all_bytes_zero": True,
            "token63": token63_post_create,
            "classification": "post_create_default_zero_counterexample",
            "positive_eval_runner_identity_match_claimed": False,
            "candidate_model_weights": False,
        },
        "post_evaluate_0_default_repeat": {
            "run_a": {
                "host": host_a.input,
                "identity": host_a.identity,
                "capture": eval_a.input,
            },
            "run_b": {
                "host": host_b.input,
                "identity": host_b.identity,
                "capture": eval_b.input,
            },
            "capture_point": eval_a.capture_point,
            "entries": eval_a.entries,
            "entries_root_sha256": eval_a.entries_root_sha256,
            "buffer_count": len(eval_a.entries),
            "logical_bytes": sum(
                entry["logical_bytes"] for entry in eval_a.entries
            ),
            "nonzero_bytes": eval_a.total_nonzero_bytes,
            "all_buffers_nonzero": True,
            "all_dump_hashes_repeat_exact": True,
            "bytes_fed_back_to_execution": False,
            "output_baseline_sha256": host_a.output["baseline_sha256"],
            "output_baseline_unchanged": host_a.output == host_b.output,
            "classification": "repeat_exact_nonzero_runtime_buffers_unclassified",
        },
        "token63_capture_point_separation": {
            "post_create": token63_post_create,
            "post_evaluate_0": token63_post_eval,
            "hash_changed": True,
            "zero_to_nonzero": True,
            "capture_points_not_interchangeable": True,
        },
        "promotion_gate": {
            "nonzero_default_data": True,
            "two_process_repeat_exact": True,
            "exact_selected_module_consumption_proven": False,
            "contents_classified_as_model_weights": False,
            "required_next_evidence": "bind each candidate buffer token and byte range to an exact selected module load/consumption group",
        },
        "classification": "repeat_exact_nonzero_runtime_buffers_not_yet_model_weights",
        "promotion_blockers": [
            "exact_selected_module_buffer_consumption_not_proven"
        ],
        "claims": {
            "address_free_receipt": True,
            "upload_weight_recovery_rejected": True,
            "post_create_default_weight_recovery_rejected": True,
            "post_evaluate_runtime_buffer_extraction_accepted": True,
            "runtime_buffers_repeat_exact": True,
            "runtime_buffers_nonzero": True,
            "bytes_fed_back_to_execution": False,
            "model_weights_recovered": False,
            "whole_model_consumption_proven": False,
        },
        "blockers": [],
        "receipt_root_sha256": None,
    }
    receipt["receipt_root_sha256"] = rooted(receipt, "receipt_root_sha256")
    return receipt, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-provenance-receipt", type=Path, required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--upload-counterexample", type=Path, required=True)
    parser.add_argument("--post-create-run", type=Path, required=True)
    parser.add_argument("--eval-run-a", type=Path, required=True)
    parser.add_argument("--eval-run-b", type=Path, required=True)
    parser.add_argument("--runner-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-receipt", action="store_true")
    args = parser.parse_args()
    try:
        receipt, status = build_receipt(
            args.allocation_provenance_receipt,
            args.upload_root,
            args.upload_counterexample,
            args.post_create_run,
            args.eval_run_a,
            args.eval_run_b,
            args.runner_manifest,
        )
    except (OSError, json.JSONDecodeError, ExtractionError, allocation.AllocationError) as error:
        print(f"FG runtime buffer extraction verification failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        print(f"FG runtime buffer extraction output already exists: {args.output}", file=sys.stderr)
        return 2
    if args.print_receipt:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "receipt_root_sha256": receipt["receipt_root_sha256"],
                    "buffer_count": receipt["post_evaluate_0_default_repeat"][
                        "buffer_count"
                    ],
                    "logical_bytes": receipt["post_evaluate_0_default_repeat"][
                        "logical_bytes"
                    ],
                    "nonzero_bytes": receipt["post_evaluate_0_default_repeat"][
                        "nonzero_bytes"
                    ],
                    "repeat_exact": receipt["claims"][
                        "runtime_buffers_repeat_exact"
                    ],
                    "model_weights_recovered": False,
                },
                sort_keys=True,
            )
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
