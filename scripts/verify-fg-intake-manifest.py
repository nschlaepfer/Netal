#!/usr/bin/env python3
"""Fail-closed verifier for two-process DLSS Frame Generation intake."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
BUILDER = REPOSITORY / "scripts" / "build-fg-intake-manifest.py"
SPEC = REPOSITORY / "evidence" / "specs" / "fg-intake-v1" / "manifest.schema.json"
MANIFEST_SCHEMA = "neuralstudio.fg.intake-manifest.v1"
AUDIT_SCHEMA = "neuralstudio.fg.intake-audit.v1"

HASH = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
COMMIT = {"sha1": re.compile(r"^[0-9a-f]{40}$"), "sha256": HASH}

ARTIFACT_KINDS = {
    "runtime_binary",
    "module_binary",
    "source_snapshot",
    "instrumentation_binary",
    "environment_report",
    "command_log",
    "stdout",
    "stderr",
    "parameter_ledger",
    "resource_ledger",
    "allocation_ledger",
    "transfer_ledger",
    "launch_trace",
    "input_frame",
    "real_frame",
    "interpolated_frame",
    "temporal_state",
    "process_receipt",
    "driver_binary",
    "nvapi_binary",
    "ngx_cache_snapshot",
    "scenario_directory_snapshot",
    "control_byte",
    "module_identity_ledger",
}
ALLOWED_ORIGINS = {
    "live_renderer_input",
    "fresh_generated_input",
    "runtime_internal",
    "immutable_runtime_asset",
    "fresh_zero_initialization",
    "live_prior_temporal_state",
}
FORBIDDEN_ORIGINS = {
    "expected_output",
    "oracle_output",
    "captured_runtime_output",
    "captured_intermediate",
    "comparison_fixture",
    "replayed_intermediate",
    "archived_temporal_state",
}
REQUIRED_CAPABILITIES = {
    "minimal_core",
    "hudless_color",
    "ui_color_alpha",
    "ui_alpha",
    "distortion_field",
    "output_real",
    "disable_interpolation",
    "hdr",
    "sdr",
    "subrects",
    "reset_menu_detection",
    "multi_frame_count_index",
    "presentation_reflex",
}


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


def is_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def is_id(value: Any) -> bool:
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


def valid_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.tzinfo == timezone.utc else None


def add(failures: list[str], condition: bool, code: str) -> None:
    if not condition:
        failures.append(code)


def exact(
    failures: list[str], value: Any, fields: set[str], code: str
) -> bool:
    if not isinstance(value, dict):
        failures.append(f"{code}.object")
        return False
    missing = sorted(fields - set(value))
    extra = sorted(set(value) - fields)
    failures.extend(f"{code}.missing.{field}" for field in missing)
    failures.extend(f"{code}.unknown.{field}" for field in extra)
    return not missing and not extra


def safe_bundle_file(bundle_root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return None
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        return None
    root = bundle_root.resolve()
    lexical = root / Path(*pure.parts)
    parent = lexical.parent
    while parent != root:
        if parent.is_symlink():
            return None
        parent = parent.parent
    if lexical.is_symlink():
        return None
    path = lexical.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


class ArtifactIndex:
    def __init__(
        self,
        entries: Any,
        bundle_root: Path,
        failures: list[str],
        verify_files: bool,
    ) -> None:
        self.by_id: dict[str, dict[str, Any]] = {}
        self.used: set[str] = set()
        self.failures = failures
        self.bundle_root = bundle_root
        if not isinstance(entries, list) or not entries:
            failures.append("artifacts.array")
            return
        paths: set[str] = set()
        for index, entry in enumerate(entries):
            code = f"artifacts.{index}"
            if not exact(
                failures,
                entry,
                {"id", "relative_path", "kind", "bytes", "sha256"},
                code,
            ):
                continue
            artifact_id = entry.get("id")
            relative = entry.get("relative_path")
            add(failures, is_id(artifact_id), f"{code}.id")
            add(
                failures,
                entry.get("kind") in ARTIFACT_KINDS,
                f"{code}.kind",
            )
            add(
                failures,
                isinstance(entry.get("bytes"), int) and entry["bytes"] >= 0,
                f"{code}.bytes",
            )
            add(failures, is_hash(entry.get("sha256")), f"{code}.sha256")
            add(
                failures,
                isinstance(relative, str) and relative not in paths,
                f"{code}.path_unique",
            )
            if isinstance(relative, str):
                paths.add(relative)
            if is_id(artifact_id):
                add(failures, artifact_id not in self.by_id, f"{code}.id_unique")
                self.by_id[artifact_id] = entry
            if verify_files:
                path = safe_bundle_file(bundle_root, relative)
                add(failures, path is not None, f"{code}.file")
                if path is not None:
                    add(
                        failures,
                        path.stat().st_size == entry.get("bytes"),
                        f"{code}.file_bytes",
                    )
                    add(
                        failures,
                        digest_file(path) == entry.get("sha256"),
                        f"{code}.file_sha256",
                    )

    def reference(
        self,
        artifact_id: Any,
        code: str,
        kinds: set[str] | None = None,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        nullable: bool = False,
    ) -> dict[str, Any] | None:
        if artifact_id is None and nullable:
            return None
        add(self.failures, is_id(artifact_id), f"{code}.id")
        entry = self.by_id.get(artifact_id) if isinstance(artifact_id, str) else None
        add(self.failures, entry is not None, f"{code}.missing")
        if entry is None:
            return None
        self.used.add(artifact_id)
        if kinds is not None:
            add(self.failures, entry.get("kind") in kinds, f"{code}.kind")
        if expected_sha256 is not None:
            add(
                self.failures,
                entry.get("sha256") == expected_sha256,
                f"{code}.sha256",
            )
        if expected_bytes is not None:
            add(
                self.failures,
                entry.get("bytes") == expected_bytes,
                f"{code}.bytes",
            )
        return entry


def validate_job(
    value: Any, artifacts: ArtifactIndex, failures: list[str]
) -> None:
    fields = {
        "job_id",
        "remote_job_root_sha256",
        "repository",
        "branch",
        "commit_algorithm",
        "commit_sha",
        "worktree_clean",
        "source_snapshot_artifact_id",
        "source_snapshot_sha256",
        "instrumentation_snapshot_artifact_id",
        "instrumentation_snapshot_sha256",
        "capture_protocol_sha256",
    }
    if not exact(failures, value, fields, "job"):
        return
    add(failures, is_id(value.get("job_id")), "job.id")
    add(failures, is_hash(value.get("remote_job_root_sha256")), "job.remote_root")
    add(
        failures,
        isinstance(value.get("repository"), str) and bool(value["repository"]),
        "job.repository",
    )
    add(
        failures,
        isinstance(value.get("branch"), str) and bool(value["branch"]),
        "job.branch",
    )
    algorithm = value.get("commit_algorithm")
    add(failures, algorithm in COMMIT, "job.commit_algorithm")
    add(
        failures,
        algorithm in COMMIT
        and isinstance(value.get("commit_sha"), str)
        and COMMIT[algorithm].fullmatch(value["commit_sha"]) is not None,
        "job.commit_sha",
    )
    add(failures, isinstance(value.get("worktree_clean"), bool), "job.worktree_clean")
    add(failures, is_hash(value.get("source_snapshot_sha256")), "job.source_sha")
    add(
        failures,
        is_hash(value.get("instrumentation_snapshot_sha256")),
        "job.instrumentation_sha",
    )
    add(failures, is_hash(value.get("capture_protocol_sha256")), "job.protocol_sha")
    artifacts.reference(
        value.get("source_snapshot_artifact_id"),
        "job.source_snapshot",
        {"source_snapshot"},
        value.get("source_snapshot_sha256"),
    )
    artifacts.reference(
        value.get("instrumentation_snapshot_artifact_id"),
        "job.instrumentation_snapshot",
        {"source_snapshot", "instrumentation_binary"},
        value.get("instrumentation_snapshot_sha256"),
    )


def validate_runtime(
    value: Any, artifacts: ArtifactIndex, failures: list[str]
) -> None:
    fields = {
        "feature",
        "feature_id",
        "version",
        "mode",
        "multiplication_factor",
        "primary_runtime_artifact_id",
        "primary_runtime_sha256",
        "component_artifact_ids",
        "configuration_sha256",
        "nvapi_artifact_id",
        "nvapi_sha256",
        "driver_artifact_id",
        "driver_sha256",
    }
    if not exact(failures, value, fields, "runtime"):
        return
    add(failures, value.get("feature") == "DLSS Frame Generation", "runtime.feature")
    add(
        failures,
        isinstance(value.get("feature_id"), int) and value["feature_id"] > 0,
        "runtime.feature_id",
    )
    add(
        failures,
        isinstance(value.get("version"), str) and bool(value["version"]),
        "runtime.version",
    )
    add(failures, value.get("mode") == "2x", "runtime.mode")
    add(failures, value.get("multiplication_factor") == 2, "runtime.factor")
    add(failures, is_hash(value.get("primary_runtime_sha256")), "runtime.primary_sha")
    primary_id = value.get("primary_runtime_artifact_id")
    artifacts.reference(
        primary_id,
        "runtime.primary",
        {"runtime_binary"},
        value.get("primary_runtime_sha256"),
    )
    add(failures, is_hash(value.get("nvapi_sha256")), "runtime.nvapi_sha")
    nvapi_artifact = artifacts.reference(
        value.get("nvapi_artifact_id"),
        "runtime.nvapi",
        {"nvapi_binary", "module_identity_ledger"},
    )
    if nvapi_artifact is not None and nvapi_artifact.get("kind") == "nvapi_binary":
        add(
            failures,
            nvapi_artifact.get("sha256") == value.get("nvapi_sha256"),
            "runtime.nvapi.sha256",
        )
    add(failures, is_hash(value.get("driver_sha256")), "runtime.driver_sha")
    driver_artifact = artifacts.reference(
        value.get("driver_artifact_id"),
        "runtime.driver",
        {"driver_binary", "module_identity_ledger"},
    )
    if driver_artifact is not None and driver_artifact.get("kind") == "driver_binary":
        add(
            failures,
            driver_artifact.get("sha256") == value.get("driver_sha256"),
            "runtime.driver.sha256",
        )
    components = value.get("component_artifact_ids")
    add(
        failures,
        isinstance(components, list) and bool(components),
        "runtime.components",
    )
    if isinstance(components, list):
        add(failures, len(components) == len(set(components)), "runtime.components_unique")
        add(failures, primary_id in components, "runtime.primary_component")
        for index, artifact_id in enumerate(components):
            artifacts.reference(
                artifact_id,
                f"runtime.components.{index}",
                {"runtime_binary", "module_binary"},
            )
    add(failures, is_hash(value.get("configuration_sha256")), "runtime.configuration")


def validate_policy(value: Any, failures: list[str]) -> None:
    fields = {
        "fresh_process_count",
        "network_access_during_capture",
        "all_runtime_inputs_provenanced",
        "expected_oracle_bytes_runtime_access",
        "comparison_started_after_process_exit",
        "comparison_may_construct_runtime_values",
        "raw_pointer_values_recorded",
        "selected_launch_scope",
        "allowed_runtime_origins",
        "forbidden_runtime_origins",
        "required_output_pattern",
    }
    if not exact(failures, value, fields, "capture_policy"):
        return
    add(failures, value.get("fresh_process_count") == 2, "policy.fresh_processes")
    add(failures, value.get("network_access_during_capture") is False, "policy.offline")
    add(
        failures,
        value.get("all_runtime_inputs_provenanced") is True,
        "policy.input_provenance",
    )
    add(
        failures,
        value.get("expected_oracle_bytes_runtime_access") is False,
        "policy.no_expected_oracle_runtime_access",
    )
    add(
        failures,
        value.get("comparison_started_after_process_exit") is True,
        "policy.post_exit_comparison",
    )
    add(
        failures,
        value.get("comparison_may_construct_runtime_values") is False,
        "policy.comparison_no_construction",
    )
    add(
        failures,
        value.get("raw_pointer_values_recorded") is False,
        "policy.no_raw_pointers",
    )
    add(
        failures,
        value.get("selected_launch_scope") == "all_launches_reachable_from_fg_evaluate",
        "policy.launch_scope",
    )
    add(
        failures,
        set(value.get("allowed_runtime_origins", [])) == ALLOWED_ORIGINS,
        "policy.allowed_origins",
    )
    add(
        failures,
        set(value.get("forbidden_runtime_origins", [])) == FORBIDDEN_ORIGINS,
        "policy.forbidden_origins",
    )
    add(
        failures,
        value.get("required_output_pattern") == "real-interpolated-alternating",
        "policy.output_pattern",
    )


def validate_environment(
    value: Any, artifacts: ArtifactIndex, failures: list[str], code: str
) -> None:
    fields = {
        "host_id_sha256",
        "os_build",
        "gpu_name",
        "gpu_uuid",
        "pci_device_id",
        "compute_capability",
        "driver_version",
        "cuda_driver_version",
        "cuda_runtime_version",
        "graphics_api",
        "graphics_api_version",
        "environment_report_artifact_id",
    }
    if not exact(failures, value, fields, code):
        return
    for field in ("host_id_sha256",):
        add(failures, is_hash(value.get(field)), f"{code}.{field}")
    for field in (
        "os_build",
        "gpu_name",
        "gpu_uuid",
        "pci_device_id",
        "compute_capability",
        "driver_version",
        "cuda_driver_version",
        "cuda_runtime_version",
        "graphics_api_version",
    ):
        add(
            failures,
            isinstance(value.get(field), str) and bool(value[field]),
            f"{code}.{field}",
        )
    add(failures, value.get("graphics_api") == "D3D12", f"{code}.graphics_api")
    artifacts.reference(
        value.get("environment_report_artifact_id"),
        f"{code}.report",
        {"environment_report"},
    )


def validate_process(
    value: Any, artifacts: ArtifactIndex, failures: list[str], code: str
) -> None:
    fields = {
        "pid",
        "process_start_token_sha256",
        "nonce_sha256",
        "executable_artifact_id",
        "executable_sha256",
        "fresh_process",
        "started_utc",
        "ended_utc",
        "device_luid_sha256",
        "queue_identity_sha256",
        "peak_gpu_memory_bytes",
    }
    if not exact(failures, value, fields, code):
        return
    add(
        failures,
        isinstance(value.get("pid"), int) and value["pid"] > 0,
        f"{code}.pid",
    )
    add(
        failures,
        is_hash(value.get("process_start_token_sha256")),
        f"{code}.start_token",
    )
    add(failures, is_hash(value.get("nonce_sha256")), f"{code}.nonce")
    add(failures, is_hash(value.get("device_luid_sha256")), f"{code}.device_luid")
    add(failures, is_hash(value.get("queue_identity_sha256")), f"{code}.queue_identity")
    add(
        failures,
        isinstance(value.get("peak_gpu_memory_bytes"), int)
        and value["peak_gpu_memory_bytes"] >= 0,
        f"{code}.peak_gpu_memory",
    )
    add(failures, is_hash(value.get("executable_sha256")), f"{code}.executable_sha")
    artifacts.reference(
        value.get("executable_artifact_id"),
        f"{code}.executable",
        {"instrumentation_binary"},
        value.get("executable_sha256"),
    )
    add(failures, value.get("fresh_process") is True, f"{code}.fresh")
    started = valid_utc(value.get("started_utc"))
    ended = valid_utc(value.get("ended_utc"))
    add(failures, started is not None, f"{code}.started")
    add(failures, ended is not None, f"{code}.ended")
    add(
        failures,
        started is not None and ended is not None and ended >= started,
        f"{code}.time_order",
    )


def validate_command(
    value: Any, artifacts: ArtifactIndex, failures: list[str], code: str
) -> None:
    fields = {
        "argv_sha256",
        "environment_sha256",
        "working_directory_sha256",
        "command_log_artifact_id",
    }
    if not exact(failures, value, fields, code):
        return
    for field in ("argv_sha256", "environment_sha256", "working_directory_sha256"):
        add(failures, is_hash(value.get(field)), f"{code}.{field}")
    artifacts.reference(
        value.get("command_log_artifact_id"),
        f"{code}.log",
        {"command_log"},
    )


def validate_modules(
    values: Any,
    runtime: dict[str, Any],
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_graph: bool,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(values, list) or not values:
        failures.append(f"{code}.array")
        return result
    orders: list[int] = []
    primary_seen = False
    cuda_seen = False
    nvapi_seen = False
    driver_seen = False
    fields = {
        "module_id",
        "load_order",
        "kind",
        "artifact_id",
        "sha256",
        "bytes",
        "load_event_sha256",
        "exports_root_sha256",
        "creation_api",
        "module_descriptor_sha256",
    }
    for index, module in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, module, fields, item_code):
            continue
        module_id = module.get("module_id")
        add(failures, is_id(module_id), f"{item_code}.id")
        add(failures, module_id not in result, f"{item_code}.id_unique")
        if is_id(module_id):
            result[module_id] = module
        order = module.get("load_order")
        add(failures, isinstance(order, int) and order > 0, f"{item_code}.order")
        if isinstance(order, int):
            orders.append(order)
        add(
            failures,
            module.get("kind")
            in {"runtime_library", "cuda_module", "dxil_module", "optical_flow_module", "driver_library", "nvapi_library", "opaque_module"},
            f"{item_code}.kind",
        )
        cuda_seen = cuda_seen or module.get("kind") == "cuda_module"
        nvapi_seen = nvapi_seen or (
            module.get("kind") == "nvapi_library"
            and module.get("sha256") == runtime.get("nvapi_sha256")
        )
        driver_seen = driver_seen or (
            module.get("kind") == "driver_library"
            and module.get("sha256") == runtime.get("driver_sha256")
        )
        add(
            failures,
            module.get("creation_api")
            in {
                "LoadLibrary",
                "NvAPI_CreateCuModule",
                "NvAPI_CreateCuFunction",
                "NvAPI_D3D12_CreateCuModule",
                "NvAPI_D3D12_CreateCuModuleEx",
                "NvAPI_D3D12_CreateCuFunction",
                "cuModuleLoadData",
                "D3D12_CreatePipelineState",
                "driver_load",
            },
            f"{item_code}.creation_api",
        )
        add(failures, is_hash(module.get("sha256")), f"{item_code}.sha")
        add(
            failures,
            isinstance(module.get("bytes"), int) and module["bytes"] > 0,
            f"{item_code}.bytes",
        )
        module_artifact = artifacts.reference(
            module.get("artifact_id"),
            f"{item_code}.artifact",
            {"runtime_binary", "module_binary", "driver_binary", "nvapi_binary", "module_identity_ledger"},
        )
        if module_artifact is not None and module_artifact.get("kind") != "module_identity_ledger":
            add(
                failures,
                module_artifact.get("sha256") == module.get("sha256"),
                f"{item_code}.artifact.sha256",
            )
            add(
                failures,
                module_artifact.get("bytes") == module.get("bytes"),
                f"{item_code}.artifact.bytes",
            )
        primary_seen = primary_seen or module.get("sha256") == runtime.get(
            "primary_runtime_sha256"
        )
        add(failures, is_hash(module.get("load_event_sha256")), f"{item_code}.load_event")
        add(failures, is_hash(module.get("exports_root_sha256")), f"{item_code}.exports")
        add(
            failures,
            rooted(module, "module_descriptor_sha256")
            == module.get("module_descriptor_sha256"),
            f"{item_code}.root",
        )
    add(failures, sorted(orders) == list(range(1, len(values) + 1)), f"{code}.orders")
    add(failures, primary_seen, f"{code}.primary_runtime_loaded")
    if require_graph:
        add(failures, cuda_seen, f"{code}.cuda_module_loaded")
    add(failures, nvapi_seen, f"{code}.nvapi_loaded")
    add(failures, driver_seen, f"{code}.driver_loaded")
    return result


def validate_allocations(
    values: Any, failures: list[str], code: str, require_complete: bool
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(values, list):
        failures.append(f"{code}.array")
        return result
    if not values:
        if require_complete:
            failures.append(f"{code}.array")
        return result
    fields = {
        "allocation_id",
        "creation_order",
        "size_bytes",
        "memory_domain",
        "heap_kind",
        "flags",
        "created_by",
        "alias_group",
        "initial_content_sha256",
        "initial_origin",
        "input_to_runtime",
        "expected_or_oracle",
        "destroyed_after_run",
        "allocation_descriptor_sha256",
    }
    orders: list[int] = []
    for index, allocation in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, allocation, fields, item_code):
            continue
        allocation_id = allocation.get("allocation_id")
        add(failures, is_id(allocation_id), f"{item_code}.id")
        add(failures, allocation_id not in result, f"{item_code}.id_unique")
        if is_id(allocation_id):
            result[allocation_id] = allocation
        order = allocation.get("creation_order")
        add(failures, isinstance(order, int) and order > 0, f"{item_code}.order")
        if isinstance(order, int):
            orders.append(order)
        add(
            failures,
            isinstance(allocation.get("size_bytes"), int)
            and allocation["size_bytes"] > 0,
            f"{item_code}.size",
        )
        add(
            failures,
            allocation.get("memory_domain")
            in {"device_local", "host_visible", "managed", "readback", "upload"},
            f"{item_code}.memory_domain",
        )
        add(
            failures,
            isinstance(allocation.get("heap_kind"), str) and bool(allocation["heap_kind"]),
            f"{item_code}.heap_kind",
        )
        add(
            failures,
            isinstance(allocation.get("flags"), list)
            and all(isinstance(flag, str) for flag in allocation["flags"])
            and allocation["flags"] == sorted(set(allocation["flags"])),
            f"{item_code}.flags",
        )
        add(
            failures,
            allocation.get("created_by") in {"application", "runtime", "driver"},
            f"{item_code}.created_by",
        )
        add(
            failures,
            allocation.get("alias_group") is None or is_id(allocation.get("alias_group")),
            f"{item_code}.alias_group",
        )
        add(
            failures,
            is_hash(allocation.get("initial_content_sha256")),
            f"{item_code}.initial_content",
        )
        add(
            failures,
            allocation.get("initial_origin") in ALLOWED_ORIGINS,
            f"{item_code}.initial_origin",
        )
        add(
            failures,
            isinstance(allocation.get("input_to_runtime"), bool),
            f"{item_code}.input_flag",
        )
        add(
            failures,
            allocation.get("expected_or_oracle") is False,
            f"{item_code}.no_expected_oracle",
        )
        add(
            failures,
            allocation.get("destroyed_after_run") is True,
            f"{item_code}.destroyed",
        )
        add(
            failures,
            rooted(allocation, "allocation_descriptor_sha256")
            == allocation.get("allocation_descriptor_sha256"),
            f"{item_code}.root",
        )
    add(failures, sorted(orders) == list(range(1, len(values) + 1)), f"{code}.orders")
    return result


def validate_resources(
    values: Any,
    allocations: dict[str, dict[str, Any]],
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_complete: bool,
    require_state_transitions: bool,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(values, list):
        failures.append(f"{code}.array")
        return result
    if not values:
        if require_complete:
            failures.append(f"{code}.array")
        return result
    fields = {
        "resource_id",
        "frame_index",
        "semantic_role",
        "access",
        "allocation_id",
        "offset_bytes",
        "size_bytes",
        "format",
        "width",
        "height",
        "depth",
        "row_pitch_bytes",
        "mip_levels",
        "array_layers",
        "sample_count",
        "subresource_index",
        "content_sha256",
        "artifact_id",
        "com_resource_token_sha256",
        "initial_state",
        "final_state",
        "state_transitions",
        "provenance",
        "resource_descriptor_sha256",
    }
    provenance_fields = {
        "origin",
        "created_in_process",
        "source_artifact_id",
        "producer_event_id",
        "expected_or_oracle",
        "fed_to_runtime",
    }
    roles: set[str] = set()
    for index, resource in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, resource, fields, item_code):
            continue
        resource_id = resource.get("resource_id")
        add(failures, is_id(resource_id), f"{item_code}.id")
        add(failures, resource_id not in result, f"{item_code}.id_unique")
        if is_id(resource_id):
            result[resource_id] = resource
        add(
            failures,
            resource.get("frame_index") is None
            or isinstance(resource.get("frame_index"), int)
            and resource["frame_index"] >= 0,
            f"{item_code}.frame_index",
        )
        role = resource.get("semantic_role")
        add(failures, is_id(role), f"{item_code}.role")
        if isinstance(role, str):
            roles.add(role)
        add(
            failures,
            resource.get("access") in {"read", "write", "read_write"},
            f"{item_code}.access",
        )
        allocation = allocations.get(resource.get("allocation_id"))
        add(failures, allocation is not None, f"{item_code}.allocation")
        for field, minimum in (
            ("offset_bytes", 0),
            ("size_bytes", 1),
            ("width", 1),
            ("height", 1),
            ("depth", 1),
            ("row_pitch_bytes", 1),
            ("mip_levels", 1),
            ("array_layers", 1),
            ("sample_count", 1),
            ("subresource_index", 0),
        ):
            add(
                failures,
                isinstance(resource.get(field), int) and resource[field] >= minimum,
                f"{item_code}.{field}",
            )
        if allocation is not None and isinstance(resource.get("offset_bytes"), int) and isinstance(resource.get("size_bytes"), int):
            add(
                failures,
                resource["offset_bytes"] + resource["size_bytes"]
                <= allocation.get("size_bytes", -1),
                f"{item_code}.bounds",
            )
        add(
            failures,
            isinstance(resource.get("format"), str) and bool(resource["format"]),
            f"{item_code}.format",
        )
        add(failures, is_hash(resource.get("content_sha256")), f"{item_code}.content")
        add(
            failures,
            is_hash(resource.get("com_resource_token_sha256")),
            f"{item_code}.com_token",
        )
        add(
            failures,
            isinstance(resource.get("initial_state"), str)
            and bool(resource["initial_state"]),
            f"{item_code}.initial_state",
        )
        add(
            failures,
            isinstance(resource.get("final_state"), str)
            and bool(resource["final_state"]),
            f"{item_code}.final_state",
        )
        transitions = resource.get("state_transitions")
        add(
            failures,
            isinstance(transitions, list)
            and (bool(transitions) or not require_state_transitions),
            f"{item_code}.state_transitions",
        )
        previous_state = resource.get("initial_state")
        if isinstance(transitions, list):
            transition_fields = {"order", "before", "after", "event_sha256"}
            for transition_index, transition in enumerate(transitions):
                transition_code = f"{item_code}.state_transitions.{transition_index}"
                if not exact(failures, transition, transition_fields, transition_code):
                    continue
                add(
                    failures,
                    transition.get("order") == transition_index + 1,
                    f"{transition_code}.order",
                )
                add(
                    failures,
                    transition.get("before") == previous_state,
                    f"{transition_code}.continuity",
                )
                add(
                    failures,
                    isinstance(transition.get("after"), str)
                    and bool(transition["after"]),
                    f"{transition_code}.after",
                )
                add(
                    failures,
                    is_hash(transition.get("event_sha256")),
                    f"{transition_code}.event",
                )
                previous_state = transition.get("after")
            if transitions or require_state_transitions:
                add(
                    failures,
                    previous_state == resource.get("final_state"),
                    f"{item_code}.final_state_binding",
                )
        artifact = artifacts.reference(
            resource.get("artifact_id"),
            f"{item_code}.artifact",
            {"input_frame", "real_frame", "interpolated_frame", "temporal_state", "resource_ledger"},
            resource.get("content_sha256") if resource.get("artifact_id") is not None else None,
            nullable=True,
        )
        provenance = resource.get("provenance")
        if exact(failures, provenance, provenance_fields, f"{item_code}.provenance"):
            origin = provenance.get("origin")
            add(failures, origin in ALLOWED_ORIGINS, f"{item_code}.provenance.origin")
            add(failures, origin not in FORBIDDEN_ORIGINS, f"{item_code}.provenance.forbidden")
            add(
                failures,
                isinstance(provenance.get("created_in_process"), bool),
                f"{item_code}.provenance.created",
            )
            add(
                failures,
                provenance.get("expected_or_oracle") is False,
                f"{item_code}.provenance.no_expected_oracle",
            )
            add(
                failures,
                provenance.get("fed_to_runtime") is True,
                f"{item_code}.provenance.fed",
            )
            source_id = provenance.get("source_artifact_id")
            source = artifacts.reference(
                source_id,
                f"{item_code}.provenance.source",
                {"input_frame", "real_frame", "temporal_state", "source_snapshot", "module_binary", "runtime_binary"},
                nullable=True,
            )
            if source is not None:
                add(
                    failures,
                    source.get("sha256") == resource.get("content_sha256"),
                    f"{item_code}.provenance.source_content",
                )
            add(
                failures,
                provenance.get("producer_event_id") is None
                or is_id(provenance.get("producer_event_id")),
                f"{item_code}.provenance.producer",
            )
        add(
            failures,
            rooted(resource, "resource_descriptor_sha256")
            == resource.get("resource_descriptor_sha256"),
            f"{item_code}.root",
        )
    if require_complete:
        add(
            failures,
            {"input_color", "depth", "motion_vectors", "output_interpolated"}.issubset(roles),
            f"{code}.core_roles",
        )
    return result


def validate_transfers(
    values: Any,
    allocations: dict[str, dict[str, Any]],
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_complete: bool,
) -> None:
    if not isinstance(values, list):
        failures.append(f"{code}.array")
        return
    if not values:
        if require_complete:
            failures.append(f"{code}.array")
        return
    fields = {
        "transfer_id",
        "order",
        "frame_index",
        "direction",
        "src_allocation_id",
        "dst_allocation_id",
        "offset_bytes",
        "size_bytes",
        "content_sha256",
        "origin",
        "source_artifact_id",
        "runtime_input",
        "expected_or_oracle",
        "transfer_descriptor_sha256",
    }
    ids: set[str] = set()
    orders: list[int] = []
    for index, transfer in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, transfer, fields, item_code):
            continue
        transfer_id = transfer.get("transfer_id")
        add(failures, is_id(transfer_id), f"{item_code}.id")
        add(failures, transfer_id not in ids, f"{item_code}.id_unique")
        if isinstance(transfer_id, str):
            ids.add(transfer_id)
        order = transfer.get("order")
        add(failures, isinstance(order, int) and order > 0, f"{item_code}.order")
        if isinstance(order, int):
            orders.append(order)
        add(
            failures,
            transfer.get("frame_index") is None
            or isinstance(transfer.get("frame_index"), int)
            and transfer["frame_index"] >= 0,
            f"{item_code}.frame",
        )
        direction = transfer.get("direction")
        add(
            failures,
            direction in {"host_to_device", "device_to_host", "device_to_device"},
            f"{item_code}.direction",
        )
        source_allocation = transfer.get("src_allocation_id")
        destination_allocation = transfer.get("dst_allocation_id")
        add(
            failures,
            source_allocation is None or source_allocation in allocations,
            f"{item_code}.src",
        )
        add(
            failures,
            destination_allocation is None or destination_allocation in allocations,
            f"{item_code}.dst",
        )
        if direction == "host_to_device":
            add(failures, source_allocation is None and destination_allocation in allocations, f"{item_code}.host_to_device")
        elif direction == "device_to_host":
            add(failures, source_allocation in allocations and destination_allocation is None, f"{item_code}.device_to_host")
        elif direction == "device_to_device":
            add(failures, source_allocation in allocations and destination_allocation in allocations, f"{item_code}.device_to_device")
        for field, minimum in (("offset_bytes", 0), ("size_bytes", 1)):
            add(
                failures,
                isinstance(transfer.get(field), int) and transfer[field] >= minimum,
                f"{item_code}.{field}",
            )
        target_id = destination_allocation if direction == "host_to_device" else source_allocation
        target = allocations.get(target_id)
        if target is not None and isinstance(transfer.get("offset_bytes"), int) and isinstance(transfer.get("size_bytes"), int):
            add(
                failures,
                transfer["offset_bytes"] + transfer["size_bytes"]
                <= target.get("size_bytes", -1),
                f"{item_code}.bounds",
            )
        add(failures, is_hash(transfer.get("content_sha256")), f"{item_code}.content")
        add(failures, transfer.get("origin") in ALLOWED_ORIGINS, f"{item_code}.origin")
        add(failures, transfer.get("origin") not in FORBIDDEN_ORIGINS, f"{item_code}.forbidden")
        add(
            failures,
            isinstance(transfer.get("runtime_input"), bool),
            f"{item_code}.runtime_input",
        )
        add(
            failures,
            transfer.get("expected_or_oracle") is False,
            f"{item_code}.no_expected_oracle",
        )
        source = artifacts.reference(
            transfer.get("source_artifact_id"),
            f"{item_code}.source_artifact",
            {"input_frame", "real_frame", "interpolated_frame", "temporal_state", "resource_ledger"},
            nullable=True,
        )
        if source is not None:
            add(
                failures,
                source.get("sha256") == transfer.get("content_sha256"),
                f"{item_code}.source_content",
            )
        add(
            failures,
            rooted(transfer, "transfer_descriptor_sha256")
            == transfer.get("transfer_descriptor_sha256"),
            f"{item_code}.root",
        )
    add(failures, sorted(orders) == list(range(1, len(values) + 1)), f"{code}.orders")


def parameter_value_valid(kind: Any, value: Any, resource_ids: set[str]) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind in {"i32", "i64"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in {"u32", "u64"}:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if kind in {"f32", "f64"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "sha256":
        return is_hash(value)
    if kind == "resource_id":
        return value in resource_ids
    if kind == "null":
        return value is None
    if kind in {"u32x2", "f32x2", "f32x4"}:
        length = {"u32x2": 2, "f32x2": 2, "f32x4": 4}[kind]
        return isinstance(value, list) and len(value) == length and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and (kind != "u32x2" or isinstance(item, int) and item >= 0)
            for item in value
        )
    return False


def validate_parameters(
    values: Any,
    resource_ids: set[str],
    evaluation_count: int,
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_complete: bool,
) -> dict[int, str]:
    roots: dict[int, str] = {}
    if not isinstance(values, list):
        failures.append(f"{code}.array")
        return roots
    if not values:
        if require_complete:
            failures.append(f"{code}.array")
        return roots
    create_count = 0
    evaluate_indices: list[int] = []
    fields = {"phase", "evaluation_index", "ledger_artifact_id", "parameters", "parameter_set_sha256"}
    parameter_fields = {"ordinal", "operation", "name", "type", "value", "status"}
    for index, parameter_set in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, parameter_set, fields, item_code):
            continue
        phase = parameter_set.get("phase")
        evaluation_index = parameter_set.get("evaluation_index")
        add(failures, phase in {"create", "evaluate"}, f"{item_code}.phase")
        if phase == "create":
            create_count += 1
            add(failures, evaluation_index is None, f"{item_code}.create_index")
        if phase == "evaluate":
            add(
                failures,
                isinstance(evaluation_index, int) and 0 <= evaluation_index < evaluation_count,
                f"{item_code}.evaluate_index",
            )
            if isinstance(evaluation_index, int):
                evaluate_indices.append(evaluation_index)
        artifacts.reference(
            parameter_set.get("ledger_artifact_id"),
            f"{item_code}.ledger",
            {"parameter_ledger"},
        )
        parameters = parameter_set.get("parameters")
        add(failures, isinstance(parameters, list) and bool(parameters), f"{item_code}.parameters")
        ordinals: list[int] = []
        names: set[tuple[str, str]] = set()
        operations: set[str] = set()
        if isinstance(parameters, list):
            for parameter_index, parameter in enumerate(parameters):
                parameter_code = f"{item_code}.parameters.{parameter_index}"
                if not exact(failures, parameter, parameter_fields, parameter_code):
                    continue
                ordinal = parameter.get("ordinal")
                add(
                    failures,
                    isinstance(ordinal, int) and ordinal >= 0,
                    f"{parameter_code}.ordinal",
                )
                if isinstance(ordinal, int):
                    ordinals.append(ordinal)
                operation = parameter.get("operation")
                add(
                    failures,
                    operation in {"set", "get"},
                    f"{parameter_code}.operation",
                )
                if isinstance(operation, str):
                    operations.add(operation)
                name = parameter.get("name")
                name_key = (str(operation), str(name))
                add(
                    failures,
                    isinstance(name, str) and bool(name) and name_key not in names,
                    f"{parameter_code}.name",
                )
                if isinstance(name, str):
                    names.add(name_key)
                add(
                    failures,
                    parameter_value_valid(parameter.get("type"), parameter.get("value"), resource_ids),
                    f"{parameter_code}.value",
                )
                add(
                    failures,
                    parameter.get("status") == "success",
                    f"{parameter_code}.status",
                )
            add(
                failures,
                sorted(ordinals) == list(range(len(parameters))),
                f"{item_code}.ordinals",
            )
            add(
                failures,
                operations == {"set", "get"},
                f"{item_code}.set_get_coverage",
            )
        add(
            failures,
            rooted(parameter_set, "parameter_set_sha256")
            == parameter_set.get("parameter_set_sha256"),
            f"{item_code}.root",
        )
        if phase == "evaluate" and isinstance(evaluation_index, int):
            add(failures, evaluation_index not in roots, f"{item_code}.evaluation_unique")
            roots[evaluation_index] = parameter_set.get("parameter_set_sha256")
    if require_complete:
        add(failures, create_count == 1, f"{code}.one_create")
        add(
            failures,
            sorted(evaluate_indices) == list(range(evaluation_count)),
            f"{code}.evaluate_coverage",
        )
    return roots


def validate_launches(
    values: Any,
    modules: dict[str, dict[str, Any]],
    allocations: dict[str, dict[str, Any]],
    parameter_roots: dict[int, str],
    evaluation_count: int,
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_graph: bool,
) -> None:
    if not isinstance(values, list):
        failures.append(f"{code}.array")
        return
    if not values:
        if require_graph:
            failures.append(f"{code}.array")
        return
    fields = {
        "launch_id",
        "evaluation_index",
        "order",
        "module_id",
        "function",
        "kernel_code_sha256",
        "grid",
        "block",
        "dynamic_shared_bytes",
        "stream_token_sha256",
        "argument_buffer_sha256",
        "argument_extent_bytes",
        "argument_extents_captured",
        "parameter_set_sha256",
        "input_allocation_ids",
        "output_allocation_ids",
        "trace_artifact_id",
        "selection_reason",
        "dispatch_api",
        "module_create_event_sha256",
        "function_create_event_sha256",
        "launch_descriptor_sha256",
    }
    ids: set[str] = set()
    orders: dict[int, list[int]] = {index: [] for index in range(evaluation_count)}
    for index, launch in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, launch, fields, item_code):
            continue
        launch_id = launch.get("launch_id")
        add(failures, is_id(launch_id), f"{item_code}.id")
        add(failures, launch_id not in ids, f"{item_code}.id_unique")
        if isinstance(launch_id, str):
            ids.add(launch_id)
        evaluation_index = launch.get("evaluation_index")
        add(
            failures,
            isinstance(evaluation_index, int) and 0 <= evaluation_index < evaluation_count,
            f"{item_code}.evaluation",
        )
        order = launch.get("order")
        add(failures, isinstance(order, int) and order > 0, f"{item_code}.order")
        if isinstance(evaluation_index, int) and evaluation_index in orders and isinstance(order, int):
            orders[evaluation_index].append(order)
        add(failures, launch.get("module_id") in modules, f"{item_code}.module")
        add(
            failures,
            isinstance(launch.get("function"), str) and bool(launch["function"]),
            f"{item_code}.function",
        )
        for field in ("kernel_code_sha256", "stream_token_sha256", "argument_buffer_sha256"):
            add(failures, is_hash(launch.get(field)), f"{item_code}.{field}")
        add(
            failures,
            isinstance(launch.get("argument_extent_bytes"), int)
            and launch["argument_extent_bytes"] > 0,
            f"{item_code}.argument_extent",
        )
        add(
            failures,
            launch.get("argument_extents_captured") is True,
            f"{item_code}.argument_extents_captured",
        )
        for field in ("grid", "block"):
            add(
                failures,
                isinstance(launch.get(field), list)
                and len(launch[field]) == 3
                and all(isinstance(item, int) and item > 0 for item in launch[field]),
                f"{item_code}.{field}",
            )
        add(
            failures,
            isinstance(launch.get("dynamic_shared_bytes"), int)
            and launch["dynamic_shared_bytes"] >= 0,
            f"{item_code}.shared",
        )
        add(
            failures,
            parameter_roots.get(evaluation_index) == launch.get("parameter_set_sha256"),
            f"{item_code}.parameter_binding",
        )
        for field in ("input_allocation_ids", "output_allocation_ids"):
            entries = launch.get(field)
            add(
                failures,
                isinstance(entries, list)
                and bool(entries)
                and len(entries) == len(set(entries))
                and all(entry in allocations for entry in entries),
                f"{item_code}.{field}",
            )
        artifacts.reference(
            launch.get("trace_artifact_id"),
            f"{item_code}.trace",
            {"launch_trace"},
        )
        add(
            failures,
            launch.get("selection_reason") == "reachable_from_fg_evaluate",
            f"{item_code}.selection",
        )
        add(
            failures,
            launch.get("dispatch_api")
            in {
                "NvAPI_LaunchCuKernel",
                "NvAPI_LaunchCuKernelChain",
                "NvAPI_D3D12_LaunchCuKernel",
                "NvAPI_D3D12_LaunchCuKernelChain",
                "NvAPI_D3D12_LaunchCuKernelChainEx",
                "cuLaunchKernel",
                "D3D12_Dispatch",
            },
            f"{item_code}.dispatch_api",
        )
        add(
            failures,
            is_hash(launch.get("module_create_event_sha256")),
            f"{item_code}.module_create_event",
        )
        add(
            failures,
            is_hash(launch.get("function_create_event_sha256")),
            f"{item_code}.function_create_event",
        )
        add(
            failures,
            rooted(launch, "launch_descriptor_sha256")
            == launch.get("launch_descriptor_sha256"),
            f"{item_code}.root",
        )
    if require_graph:
        for evaluation_index, observed in orders.items():
            add(
                failures,
                sorted(observed) == list(range(1, len(observed) + 1)) and bool(observed),
                f"{code}.evaluation.{evaluation_index}.orders",
            )


def validate_outputs(
    values: Any,
    resources: dict[str, dict[str, Any]],
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_complete: bool,
) -> tuple[int, dict[str, dict[str, Any]]]:
    output_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(values, list):
        failures.append(f"{code}.minimum_sequence")
        return 0, output_by_id
    if not values:
        if require_complete:
            failures.append(f"{code}.minimum_sequence")
        return 0, output_by_id
    if len(values) < 3 or len(values) % 2 == 0:
        failures.append(f"{code}.sequence_shape")
        return 0, output_by_id
    if require_complete and len(values) < 5:
        failures.append(f"{code}.minimum_sequence")
    fields = {
        "output_id",
        "sequence_index",
        "evaluation_index",
        "kind",
        "frame_artifact_id",
        "resource_id",
        "content_sha256",
        "width",
        "height",
        "format",
        "presentation_time_ns",
        "source_real_output_ids",
        "classification_method",
        "presented",
        "readback_complete",
    }
    sequence: list[int] = []
    for index, output in enumerate(values):
        item_code = f"{code}.{index}"
        if not exact(failures, output, fields, item_code):
            continue
        output_id = output.get("output_id")
        add(failures, is_id(output_id), f"{item_code}.id")
        add(failures, output_id not in output_by_id, f"{item_code}.id_unique")
        if is_id(output_id):
            output_by_id[output_id] = output
        sequence_index = output.get("sequence_index")
        add(
            failures,
            isinstance(sequence_index, int) and sequence_index >= 0,
            f"{item_code}.sequence",
        )
        if isinstance(sequence_index, int):
            sequence.append(sequence_index)
        kind = output.get("kind")
        add(failures, kind in {"real", "interpolated"}, f"{item_code}.kind")
        expected_artifact_kind = {"real": "real_frame", "interpolated": "interpolated_frame"}.get(kind)
        add(failures, is_hash(output.get("content_sha256")), f"{item_code}.content")
        artifacts.reference(
            output.get("frame_artifact_id"),
            f"{item_code}.frame",
            {expected_artifact_kind} if expected_artifact_kind else set(),
            output.get("content_sha256"),
        )
        resource = resources.get(output.get("resource_id"))
        add(failures, resource is not None, f"{item_code}.resource")
        if resource is not None:
            add(
                failures,
                resource.get("content_sha256") == output.get("content_sha256"),
                f"{item_code}.resource_content",
            )
            for field in ("width", "height", "format"):
                add(
                    failures,
                    resource.get(field) == output.get(field),
                    f"{item_code}.resource_{field}",
                )
        for field in ("width", "height"):
            add(
                failures,
                isinstance(output.get(field), int) and output[field] > 0,
                f"{item_code}.{field}",
            )
        add(
            failures,
            isinstance(output.get("format"), str) and bool(output["format"]),
            f"{item_code}.format",
        )
        add(
            failures,
            isinstance(output.get("presentation_time_ns"), int)
            and output["presentation_time_ns"] >= 0,
            f"{item_code}.presentation_time",
        )
        sources = output.get("source_real_output_ids")
        if kind == "real":
            add(failures, output.get("evaluation_index") is None, f"{item_code}.real_evaluation")
            add(failures, sources == [], f"{item_code}.real_sources")
        elif kind == "interpolated":
            add(
                failures,
                isinstance(output.get("evaluation_index"), int)
                and output["evaluation_index"] >= 0,
                f"{item_code}.interpolated_evaluation",
            )
            add(
                failures,
                isinstance(sources, list) and len(sources) == 2 and len(set(sources)) == 2,
                f"{item_code}.interpolated_sources",
            )
        add(
            failures,
            output.get("classification_method")
            == "present_chain_and_fg_evaluate_correlation",
            f"{item_code}.classification",
        )
        add(failures, output.get("presented") is True, f"{item_code}.presented")
        add(
            failures,
            output.get("readback_complete") is True,
            f"{item_code}.readback",
        )
    add(failures, sequence == list(range(len(values))), f"{code}.sequence_order")
    kinds = [entry.get("kind") if isinstance(entry, dict) else None for entry in values]
    expected_kinds = ["real" if index % 2 == 0 else "interpolated" for index in range(len(values))]
    add(failures, kinds == expected_kinds, f"{code}.alternating")
    evaluation_indices: list[int] = []
    for index in range(1, len(values), 2):
        interpolated = values[index]
        left = values[index - 1]
        right = values[index + 1] if index + 1 < len(values) else {}
        add(
            failures,
            interpolated.get("source_real_output_ids")
            == [left.get("output_id"), right.get("output_id")],
            f"{code}.{index}.bracketing_sources",
        )
        if isinstance(interpolated.get("evaluation_index"), int):
            evaluation_indices.append(interpolated["evaluation_index"])
    evaluation_count = len(values) // 2
    add(
        failures,
        evaluation_indices == list(range(evaluation_count)),
        f"{code}.evaluation_order",
    )
    return evaluation_count, output_by_id


def validate_temporal(
    value: Any,
    evaluation_count: int,
    outputs: dict[str, dict[str, Any]],
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_complete: bool,
) -> None:
    fields = {"history_epoch_sha256", "state_visibility", "evaluations"}
    if not exact(failures, value, fields, code):
        return
    add(failures, is_hash(value.get("history_epoch_sha256")), f"{code}.epoch")
    add(
        failures,
        value.get("state_visibility") in {"fully_captured", "opaque", "partial"},
        f"{code}.visibility",
    )
    evaluations = value.get("evaluations")
    add(failures, isinstance(evaluations, list), f"{code}.evaluations")
    if not isinstance(evaluations, list):
        return
    if require_complete:
        add(
            failures,
            len(evaluations) == evaluation_count,
            f"{code}.evaluation_count",
        )
    elif not evaluations:
        return
    fields_evaluation = {
        "evaluation_index",
        "left_real_output_id",
        "right_real_output_id",
        "reset",
        "state_before_artifact_id",
        "state_before_sha256",
        "state_after_artifact_id",
        "state_after_sha256",
        "state_origin",
        "continuity_from_evaluation_index",
        "temporal_descriptor_sha256",
    }
    previous: dict[str, Any] | None = None
    for index, evaluation in enumerate(evaluations):
        item_code = f"{code}.{index}"
        if not exact(failures, evaluation, fields_evaluation, item_code):
            continue
        add(
            failures,
            evaluation.get("evaluation_index") == index,
            f"{item_code}.index",
        )
        left = outputs.get(evaluation.get("left_real_output_id"))
        right = outputs.get(evaluation.get("right_real_output_id"))
        add(failures, left is not None and left.get("kind") == "real", f"{item_code}.left")
        add(failures, right is not None and right.get("kind") == "real", f"{item_code}.right")
        expected_interpolated = next(
            (
                output
                for output in outputs.values()
                if output.get("kind") == "interpolated"
                and output.get("evaluation_index") == index
            ),
            None,
        )
        if expected_interpolated is not None:
            add(
                failures,
                expected_interpolated.get("source_real_output_ids")
                == [evaluation.get("left_real_output_id"), evaluation.get("right_real_output_id")],
                f"{item_code}.output_binding",
            )
        if require_complete:
            add(failures, is_hash(evaluation.get("state_before_sha256")), f"{item_code}.before")
            add(failures, is_hash(evaluation.get("state_after_sha256")), f"{item_code}.after")
            artifacts.reference(
                evaluation.get("state_before_artifact_id"),
                f"{item_code}.before_artifact",
                {"temporal_state"},
                evaluation.get("state_before_sha256"),
            )
            artifacts.reference(
                evaluation.get("state_after_artifact_id"),
                f"{item_code}.after_artifact",
                {"temporal_state"},
                evaluation.get("state_after_sha256"),
            )
        else:
            add(
                failures,
                evaluation.get("state_before_sha256") is None
                and evaluation.get("state_after_sha256") is None
                and evaluation.get("state_before_artifact_id") is None
                and evaluation.get("state_after_artifact_id") is None,
                f"{item_code}.opaque_state_must_be_null",
            )
        if index == 0:
            add(failures, evaluation.get("reset") is True, f"{item_code}.reset")
            add(
                failures,
                evaluation.get("state_origin") == "fresh_zero_initialization",
                f"{item_code}.origin",
            )
            add(
                failures,
                evaluation.get("continuity_from_evaluation_index") is None,
                f"{item_code}.continuity_parent",
            )
        else:
            add(failures, evaluation.get("reset") is False, f"{item_code}.reset")
            add(
                failures,
                evaluation.get("state_origin") == "live_prior_temporal_state",
                f"{item_code}.origin",
            )
            add(
                failures,
                evaluation.get("continuity_from_evaluation_index") == index - 1,
                f"{item_code}.continuity_parent",
            )
            if require_complete:
                add(
                    failures,
                    previous is not None
                    and previous.get("state_after_sha256")
                    == evaluation.get("state_before_sha256"),
                    f"{item_code}.continuity_hash",
                )
        add(
            failures,
            rooted(evaluation, "temporal_descriptor_sha256")
            == evaluation.get("temporal_descriptor_sha256"),
            f"{item_code}.root",
        )
        previous = evaluation


def validate_logs(
    value: Any, artifacts: ArtifactIndex, failures: list[str], code: str
) -> bool:
    fields = {
        "stdout_artifact_id",
        "stderr_artifact_id",
        "process_receipt_artifact_id",
        "exit_code",
        "timed_out",
        "termination_signal",
    }
    if not exact(failures, value, fields, code):
        return False
    artifacts.reference(value.get("stdout_artifact_id"), f"{code}.stdout", {"stdout"})
    artifacts.reference(value.get("stderr_artifact_id"), f"{code}.stderr", {"stderr"})
    artifacts.reference(
        value.get("process_receipt_artifact_id"),
        f"{code}.receipt",
        {"process_receipt"},
    )
    add(failures, value.get("exit_code") == 0, f"{code}.exit")
    add(failures, value.get("timed_out") is False, f"{code}.timeout")
    add(failures, value.get("termination_signal") is None, f"{code}.signal")
    return (
        value.get("exit_code") == 0
        and value.get("timed_out") is False
        and value.get("termination_signal") is None
    )


def validate_run_isolation(
    value: Any,
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
    require_complete: bool,
) -> None:
    fields = {
        "scenario_nonce_sha256",
        "scenario_directory_initial_artifact_id",
        "scenario_directory_initial_sha256",
        "scenario_directory_initial_entries",
        "scenario_directory_final_artifact_id",
        "scenario_directory_final_sha256",
        "scenario_directory_final_entries",
        "ngx_cache_initial_artifact_id",
        "ngx_cache_initial_sha256",
        "ngx_cache_initial_entries",
        "ngx_cache_final_artifact_id",
        "ngx_cache_final_sha256",
        "ngx_cache_final_entries",
        "isolation_descriptor_sha256",
    }
    if not exact(failures, value, fields, code):
        return
    if not require_complete:
        nullable_fields = (
            "scenario_nonce_sha256",
            "scenario_directory_initial_artifact_id",
            "scenario_directory_initial_sha256",
            "scenario_directory_initial_entries",
            "scenario_directory_final_artifact_id",
            "scenario_directory_final_sha256",
            "scenario_directory_final_entries",
            "ngx_cache_initial_artifact_id",
            "ngx_cache_initial_sha256",
            "ngx_cache_initial_entries",
            "ngx_cache_final_artifact_id",
            "ngx_cache_final_sha256",
            "ngx_cache_final_entries",
        )
        add(
            failures,
            all(value.get(field) is None for field in nullable_fields),
            f"{code}.unavailable_must_be_null",
        )
        add(
            failures,
            rooted(value, "isolation_descriptor_sha256")
            == value.get("isolation_descriptor_sha256"),
            f"{code}.root",
        )
        return
    add(failures, is_hash(value.get("scenario_nonce_sha256")), f"{code}.nonce")
    add(
        failures,
        is_hash(value.get("scenario_directory_initial_sha256")),
        f"{code}.scenario_sha",
    )
    artifacts.reference(
        value.get("scenario_directory_initial_artifact_id"),
        f"{code}.scenario",
        {"scenario_directory_snapshot"},
        value.get("scenario_directory_initial_sha256"),
    )
    add(
        failures,
        value.get("scenario_directory_initial_entries") == 0,
        f"{code}.scenario_not_empty",
    )
    add(
        failures,
        is_hash(value.get("scenario_directory_final_sha256")),
        f"{code}.scenario_final_sha",
    )
    artifacts.reference(
        value.get("scenario_directory_final_artifact_id"),
        f"{code}.scenario_final",
        {"scenario_directory_snapshot"},
        value.get("scenario_directory_final_sha256"),
    )
    add(
        failures,
        isinstance(value.get("scenario_directory_final_entries"), int)
        and value["scenario_directory_final_entries"] >= 0,
        f"{code}.scenario_final_entries",
    )
    add(failures, is_hash(value.get("ngx_cache_initial_sha256")), f"{code}.cache_initial_sha")
    add(failures, is_hash(value.get("ngx_cache_final_sha256")), f"{code}.cache_final_sha")
    artifacts.reference(
        value.get("ngx_cache_initial_artifact_id"),
        f"{code}.cache_initial",
        {"ngx_cache_snapshot"},
        value.get("ngx_cache_initial_sha256"),
    )
    artifacts.reference(
        value.get("ngx_cache_final_artifact_id"),
        f"{code}.cache_final",
        {"ngx_cache_snapshot"},
        value.get("ngx_cache_final_sha256"),
    )
    add(
        failures,
        value.get("ngx_cache_initial_entries") == 0,
        f"{code}.cache_not_empty",
    )
    add(
        failures,
        isinstance(value.get("ngx_cache_final_entries"), int)
        and value["ngx_cache_final_entries"] >= 0,
        f"{code}.cache_final_entries",
    )
    add(
        failures,
        rooted(value, "isolation_descriptor_sha256")
        == value.get("isolation_descriptor_sha256"),
        f"{code}.root",
    )


def validate_control_evidence(
    value: Any,
    outputs: dict[str, dict[str, Any]],
    resources: dict[str, dict[str, Any]],
    artifacts: ArtifactIndex,
    failures: list[str],
    code: str,
) -> None:
    fields = {
        "frame_a_output_id",
        "frame_b_output_id",
        "interpolated_output_ids",
        "output_real_resource_ids",
        "disable_interpolation_artifact_id",
        "disable_interpolation_sha256",
        "disable_interpolation_value",
        "captured_from_live_parameter",
        "control_descriptor_sha256",
    }
    if not exact(failures, value, fields, code):
        return
    frame_a = outputs.get(value.get("frame_a_output_id"))
    frame_b = outputs.get(value.get("frame_b_output_id"))
    add(failures, frame_a is not None and frame_a.get("kind") == "real", f"{code}.frame_a")
    add(failures, frame_b is not None and frame_b.get("kind") == "real", f"{code}.frame_b")
    interpolated_ids = value.get("interpolated_output_ids")
    output_real_ids = value.get("output_real_resource_ids")
    expected_interpolated = [
        item_id for item_id, item in outputs.items() if item.get("kind") == "interpolated"
    ]
    expected_output_real = [
        resource_id
        for resource_id, resource in resources.items()
        if resource.get("semantic_role") == "output_real"
    ]
    add(
        failures,
        isinstance(interpolated_ids, list)
        and interpolated_ids == expected_interpolated,
        f"{code}.interpolated_outputs",
    )
    add(
        failures,
        isinstance(output_real_ids, list)
        and output_real_ids == expected_output_real,
        f"{code}.output_real_resources",
    )
    add(
        failures,
        is_hash(value.get("disable_interpolation_sha256")),
        f"{code}.disable_sha",
    )
    artifacts.reference(
        value.get("disable_interpolation_artifact_id"),
        f"{code}.disable_artifact",
        {"control_byte"},
        value.get("disable_interpolation_sha256"),
        1,
    )
    add(
        failures,
        value.get("disable_interpolation_value") in {0, 1},
        f"{code}.disable_value",
    )
    add(
        failures,
        value.get("captured_from_live_parameter") is True,
        f"{code}.live_parameter",
    )
    add(
        failures,
        rooted(value, "control_descriptor_sha256")
        == value.get("control_descriptor_sha256"),
        f"{code}.root",
    )


COMPLETENESS_FIELDS = {
    "all_parameters_recorded",
    "all_resource_descriptors_recorded",
    "all_modules_recorded",
    "all_allocations_recorded",
    "all_transfers_recorded",
    "all_fg_launches_selected",
    "all_presented_outputs_recorded",
    "temporal_state_captured",
    "stdout_stderr_captured",
    "exit_status_captured",
    "kernel_graph_evidence_captured",
    "resource_state_transitions_captured",
    "scenario_cache_isolation_captured",
}
GUARD_FIELDS = {
    "runtime_input_origin_audit_complete",
    "forbidden_origin_hits",
    "expected_oracle_artifacts_opened_before_exit",
    "expected_oracle_bytes_bound_to_gpu",
    "comparison_started_after_process_exit",
    "comparison_used_to_construct_runtime_values",
}


def validate_completeness_and_guard(
    run: dict[str, Any], failures: list[str], blockers: list[str], code: str
) -> None:
    completeness = run.get("capture_completeness")
    if exact(failures, completeness, COMPLETENESS_FIELDS, f"{code}.completeness"):
        for field in sorted(COMPLETENESS_FIELDS):
            add(
                failures,
                isinstance(completeness.get(field), bool),
                f"{code}.completeness.{field}.boolean",
            )
            if completeness.get(field) is not True:
                blockers.append(f"{code}.incomplete.{field}")
    guard = run.get("contamination_guard")
    if exact(failures, guard, GUARD_FIELDS, f"{code}.guard"):
        add(
            failures,
            guard.get("runtime_input_origin_audit_complete") is True,
            f"{code}.guard.origin_audit",
        )
        add(failures, guard.get("forbidden_origin_hits") == [], f"{code}.guard.forbidden_hits")
        add(
            failures,
            guard.get("expected_oracle_artifacts_opened_before_exit") is False,
            f"{code}.guard.expected_oracle_open",
        )
        add(
            failures,
            guard.get("expected_oracle_bytes_bound_to_gpu") is False,
            f"{code}.guard.expected_oracle_bound",
        )
        add(
            failures,
            guard.get("comparison_started_after_process_exit") is True,
            f"{code}.guard.post_exit",
        )
        add(
            failures,
            guard.get("comparison_used_to_construct_runtime_values") is False,
            f"{code}.guard.no_construct",
        )


def validate_issue_catalogs(
    manifest: dict[str, Any], artifacts: ArtifactIndex, failures: list[str], blockers: list[str]
) -> tuple[set[str], set[str]]:
    opaque_ids: set[str] = set()
    access_ids: set[str] = set()
    opaque = manifest.get("opaque_dependencies")
    access = manifest.get("access_issues")
    if not isinstance(opaque, list):
        failures.append("opaque_dependencies.array")
        opaque = []
    if not isinstance(access, list):
        failures.append("access_issues.array")
        access = []
    opaque_fields = {"id", "category", "component", "description_sha256", "blocking", "related_artifact_id"}
    for index, issue in enumerate(opaque):
        code = f"opaque_dependencies.{index}"
        if not exact(failures, issue, opaque_fields, code):
            continue
        issue_id = issue.get("id")
        add(failures, is_id(issue_id), f"{code}.id")
        add(failures, issue_id not in opaque_ids, f"{code}.id_unique")
        if isinstance(issue_id, str):
            opaque_ids.add(issue_id)
        add(
            failures,
            issue.get("category") in {"runtime", "module", "optical_flow", "present_chain", "temporal_state", "driver", "other"},
            f"{code}.category",
        )
        add(
            failures,
            isinstance(issue.get("component"), str) and bool(issue["component"]),
            f"{code}.component",
        )
        add(failures, is_hash(issue.get("description_sha256")), f"{code}.description")
        add(failures, isinstance(issue.get("blocking"), bool), f"{code}.blocking_boolean")
        artifacts.reference(
            issue.get("related_artifact_id"),
            f"{code}.artifact",
            nullable=True,
        )
        if issue.get("blocking") is True:
            blockers.append(f"opaque.{issue_id}")
    access_fields = {"id", "surface", "operation", "result", "error_code_sha256", "details_artifact_id", "blocking"}
    for index, issue in enumerate(access):
        code = f"access_issues.{index}"
        if not exact(failures, issue, access_fields, code):
            continue
        issue_id = issue.get("id")
        add(failures, is_id(issue_id), f"{code}.id")
        add(failures, issue_id not in access_ids, f"{code}.id_unique")
        if isinstance(issue_id, str):
            access_ids.add(issue_id)
        add(
            failures,
            isinstance(issue.get("surface"), str) and bool(issue["surface"]),
            f"{code}.surface",
        )
        add(
            failures,
            isinstance(issue.get("operation"), str) and bool(issue["operation"]),
            f"{code}.operation",
        )
        add(
            failures,
            issue.get("result") in {"granted", "denied", "unavailable", "partial"},
            f"{code}.result",
        )
        add(failures, is_hash(issue.get("error_code_sha256")), f"{code}.error")
        artifacts.reference(
            issue.get("details_artifact_id"),
            f"{code}.details",
            {"environment_report", "stdout", "stderr", "process_receipt"},
            nullable=True,
        )
        add(failures, isinstance(issue.get("blocking"), bool), f"{code}.blocking_boolean")
        if issue.get("blocking") is True or issue.get("result") != "granted":
            blockers.append(f"access.{issue_id}")
    return opaque_ids, access_ids


def validate_feature_coverage(
    value: Any,
    artifacts: ArtifactIndex,
    opaque_ids: set[str],
    access_ids: set[str],
    failures: list[str],
    blockers: list[str],
) -> None:
    if not exact(
        failures,
        value,
        {"entries", "coverage_root_sha256"},
        "feature_coverage",
    ):
        return
    entries = value.get("entries")
    add(
        failures,
        isinstance(entries, list) and len(entries) == len(REQUIRED_CAPABILITIES),
        "feature_coverage.entry_count",
    )
    if not isinstance(entries, list):
        return
    seen: set[str] = set()
    fields = {
        "capability",
        "status",
        "evidence_artifact_ids",
        "opaque_dependency_ids",
        "access_issue_ids",
        "observation_sha256",
    }
    for index, entry in enumerate(entries):
        code = f"feature_coverage.entries.{index}"
        if not exact(failures, entry, fields, code):
            continue
        capability = entry.get("capability")
        add(failures, capability in REQUIRED_CAPABILITIES, f"{code}.capability")
        add(failures, capability not in seen, f"{code}.capability_unique")
        if isinstance(capability, str):
            seen.add(capability)
        status = entry.get("status")
        add(
            failures,
            status in {"observed", "unsupported", "not_tested"},
            f"{code}.status",
        )
        evidence = entry.get("evidence_artifact_ids")
        opaque = entry.get("opaque_dependency_ids")
        access = entry.get("access_issue_ids")
        add(
            failures,
            isinstance(evidence, list) and len(evidence) == len(set(evidence)),
            f"{code}.evidence",
        )
        if isinstance(evidence, list):
            for evidence_index, artifact_id in enumerate(evidence):
                artifacts.reference(
                    artifact_id, f"{code}.evidence.{evidence_index}"
                )
        add(
            failures,
            isinstance(opaque, list)
            and len(opaque) == len(set(opaque))
            and set(opaque).issubset(opaque_ids),
            f"{code}.opaque_refs",
        )
        add(
            failures,
            isinstance(access, list)
            and len(access) == len(set(access))
            and set(access).issubset(access_ids),
            f"{code}.access_refs",
        )
        if status == "observed":
            add(failures, bool(evidence), f"{code}.observed_evidence")
        elif status in {"unsupported", "not_tested"}:
            add(
                failures,
                bool(opaque) or bool(access),
                f"{code}.explicit_gap_evidence",
            )
        add(
            failures,
            is_hash(entry.get("observation_sha256")),
            f"{code}.observation",
        )
        if capability == "minimal_core" and status != "observed":
            blockers.append("feature.minimal_core_not_observed")
    add(failures, seen == REQUIRED_CAPABILITIES, "feature_coverage.complete_catalog")
    add(
        failures,
        rooted(value, "coverage_root_sha256") == value.get("coverage_root_sha256"),
        "feature_coverage.root",
    )


def validate_run(
    run: Any,
    run_index: int,
    runtime: dict[str, Any],
    artifacts: ArtifactIndex,
    opaque_ids: set[str],
    access_ids: set[str],
    failures: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    code = f"runs.{run_index}"
    fields = {
        "run_id",
        "process",
        "environment",
        "command",
        "loaded_modules",
        "resources",
        "allocations",
        "transfers",
        "parameter_sets",
        "selected_launches",
        "outputs",
        "temporal",
        "logs",
        "run_isolation",
        "control_evidence",
        "capture_completeness",
        "contamination_guard",
        "opaque_dependency_ids",
        "access_issue_ids",
        "run_root_sha256",
    }
    if not exact(failures, run, fields, code):
        return {}
    add(failures, is_id(run.get("run_id")), f"{code}.id")
    completeness = (
        run.get("capture_completeness")
        if isinstance(run.get("capture_completeness"), dict)
        else {}
    )
    parameters_complete = completeness.get("all_parameters_recorded") is True
    resources_complete = completeness.get("all_resource_descriptors_recorded") is True
    allocations_complete = completeness.get("all_allocations_recorded") is True
    transfers_complete = completeness.get("all_transfers_recorded") is True
    outputs_complete = completeness.get("all_presented_outputs_recorded") is True
    temporal_complete = completeness.get("temporal_state_captured") is True
    graph_complete = completeness.get("kernel_graph_evidence_captured") is True
    transitions_complete = (
        completeness.get("resource_state_transitions_captured") is True
    )
    isolation_complete = (
        completeness.get("scenario_cache_isolation_captured") is True
    )
    validate_completeness_and_guard(run, failures, blockers, code)
    validate_process(run.get("process"), artifacts, failures, f"{code}.process")
    validate_environment(run.get("environment"), artifacts, failures, f"{code}.environment")
    validate_command(run.get("command"), artifacts, failures, f"{code}.command")
    modules = validate_modules(
        run.get("loaded_modules"),
        runtime,
        artifacts,
        failures,
        f"{code}.modules",
        graph_complete,
    )
    allocations = validate_allocations(
        run.get("allocations"), failures, f"{code}.allocations", allocations_complete
    )
    resources = validate_resources(
        run.get("resources"),
        allocations,
        artifacts,
        failures,
        f"{code}.resources",
        resources_complete,
        transitions_complete,
    )
    evaluation_count, outputs = validate_outputs(
        run.get("outputs"),
        resources,
        artifacts,
        failures,
        f"{code}.outputs",
        outputs_complete,
    )
    validate_temporal(
        run.get("temporal"),
        evaluation_count,
        outputs,
        artifacts,
        failures,
        f"{code}.temporal",
        temporal_complete,
    )
    validate_run_isolation(
        run.get("run_isolation"),
        artifacts,
        failures,
        f"{code}.isolation",
        isolation_complete,
    )
    validate_control_evidence(
        run.get("control_evidence"),
        outputs,
        resources,
        artifacts,
        failures,
        f"{code}.controls",
    )
    parameter_roots = validate_parameters(
        run.get("parameter_sets"),
        set(resources),
        evaluation_count,
        artifacts,
        failures,
        f"{code}.parameters",
        parameters_complete,
    )
    validate_transfers(
        run.get("transfers"),
        allocations,
        artifacts,
        failures,
        f"{code}.transfers",
        transfers_complete,
    )
    validate_launches(
        run.get("selected_launches"),
        modules,
        allocations,
        parameter_roots,
        evaluation_count,
        artifacts,
        failures,
        f"{code}.launches",
        graph_complete,
    )
    logs_ok = validate_logs(run.get("logs"), artifacts, failures, f"{code}.logs")
    opaque_refs = run.get("opaque_dependency_ids")
    access_refs = run.get("access_issue_ids")
    add(
        failures,
        isinstance(opaque_refs, list)
        and len(opaque_refs) == len(set(opaque_refs))
        and set(opaque_refs).issubset(opaque_ids),
        f"{code}.opaque_refs",
    )
    add(
        failures,
        isinstance(access_refs, list)
        and len(access_refs) == len(set(access_refs))
        and set(access_refs).issubset(access_ids),
        f"{code}.access_refs",
    )
    if any(completeness.get(field) is not True for field in COMPLETENESS_FIELDS):
        add(
            failures,
            bool(opaque_refs) or bool(access_refs),
            f"{code}.incomplete_capture_issue_unreported",
        )
    if any(module.get("kind") == "opaque_module" for module in modules.values()):
        add(failures, bool(opaque_refs), f"{code}.opaque_module_unreported")
        blockers.append(f"{code}.opaque_module")
    temporal = run.get("temporal")
    if isinstance(temporal, dict) and temporal.get("state_visibility") != "fully_captured":
        add(failures, bool(opaque_refs) or bool(access_refs), f"{code}.temporal_visibility_unreported")
        blockers.append(f"{code}.temporal_state_not_fully_captured")
    add(
        failures,
        rooted(run, "run_root_sha256") == run.get("run_root_sha256"),
        f"{code}.root",
    )
    return {
        "logs_ok": logs_ok,
        "evaluation_count": evaluation_count,
        "output_count": len(run.get("outputs", [])) if isinstance(run.get("outputs"), list) else 0,
        "kernel_graph_captured": isinstance(run.get("capture_completeness"), dict)
        and run["capture_completeness"].get("kernel_graph_evidence_captured") is True,
    }


def _normalize_artifact_references(value: Any, artifact_hashes: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_normalize_artifact_references(item, artifact_hashes) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key.endswith("_descriptor_sha256") or key in {
            "run_root_sha256",
            "stream_token_sha256",
            "presentation_time_ns",
            "com_resource_token_sha256",
            "scenario_nonce_sha256",
            "parameter_set_sha256",
        }:
            continue
        if key == "artifact_id" or key.endswith("_artifact_id"):
            normalized_key = "artifact_sha256" if key == "artifact_id" else key.replace("_artifact_id", "_artifact_sha256")
            result[normalized_key] = (
                None if item is None else artifact_hashes.get(item, f"missing:{item}")
            )
        elif key.endswith("_artifact_ids") and isinstance(item, list):
            result[key.replace("_artifact_ids", "_artifact_sha256s")] = [
                artifact_hashes.get(entry, f"missing:{entry}") for entry in item
            ]
        else:
            result[key] = _normalize_artifact_references(item, artifact_hashes)
    return result


def semantic_projection(run: dict[str, Any], artifact_hashes: dict[str, str]) -> dict[str, Any]:
    selected = {
        key: copy.deepcopy(run.get(key))
        for key in (
            "environment",
            "loaded_modules",
            "resources",
            "allocations",
            "transfers",
            "parameter_sets",
            "selected_launches",
            "outputs",
            "temporal",
            "capture_completeness",
            "contamination_guard",
            "run_isolation",
            "control_evidence",
        )
    }
    return _normalize_artifact_references(selected, artifact_hashes)


def validate_derived(manifest: dict[str, Any], failures: list[str]) -> None:
    derived = manifest.get("derived")
    fields = {
        "source_descriptor_sha256",
        "artifact_index_sha256",
        "job_descriptor_sha256",
        "runtime_descriptor_sha256",
        "run_semantic_sha256s",
        "cross_run_semantic_sha256",
        "semantic_repeat_equal",
        "output_sequence_sha256",
    }
    if not exact(failures, derived, fields, "derived"):
        return
    add(failures, is_hash(derived.get("source_descriptor_sha256")), "derived.source")
    add(
        failures,
        derived.get("artifact_index_sha256") == digest(manifest.get("artifacts")),
        "derived.artifacts",
    )
    add(
        failures,
        derived.get("job_descriptor_sha256") == digest(manifest.get("job")),
        "derived.job",
    )
    add(
        failures,
        derived.get("runtime_descriptor_sha256") == digest(manifest.get("runtime")),
        "derived.runtime",
    )
    artifact_hashes = {
        entry.get("id"): entry.get("sha256")
        for entry in manifest.get("artifacts", [])
        if isinstance(entry, dict)
    }
    runs = manifest.get("runs", [])
    semantic_roots = [
        digest(semantic_projection(run, artifact_hashes))
        for run in runs
        if isinstance(run, dict)
    ] if isinstance(runs, list) else []
    add(
        failures,
        derived.get("run_semantic_sha256s") == semantic_roots,
        "derived.run_semantics",
    )
    semantic_equal = len(semantic_roots) == 2 and len(set(semantic_roots)) == 1
    add(
        failures,
        derived.get("semantic_repeat_equal") is semantic_equal,
        "derived.semantic_repeat",
    )
    add(
        failures,
        derived.get("cross_run_semantic_sha256")
        == (semantic_roots[0] if semantic_equal else None),
        "derived.cross_run_root",
    )
    output_sequence = []
    if isinstance(runs, list) and runs and isinstance(runs[0], dict):
        for output in runs[0].get("outputs", []):
            if isinstance(output, dict):
                output_sequence.append(
                    {
                        "sequence_index": output.get("sequence_index"),
                        "kind": output.get("kind"),
                        "content_sha256": output.get("content_sha256"),
                        "source_real_output_ids": output.get("source_real_output_ids"),
                    }
                )
    add(
        failures,
        derived.get("output_sequence_sha256") == digest(output_sequence),
        "derived.output_sequence",
    )


def verify_manifest(
    manifest: dict[str, Any], bundle_root: Path, verify_files: bool = True
) -> tuple[list[str], list[str], dict[str, Any]]:
    failures: list[str] = []
    blockers: list[str] = []
    top_fields = {
        "schema",
        "tooling",
        "job",
        "runtime",
        "capture_policy",
        "artifacts",
        "runs",
        "opaque_dependencies",
        "access_issues",
        "claims",
        "feature_coverage",
        "derived",
        "manifest_root_sha256",
    }
    if not exact(failures, manifest, top_fields, "manifest"):
        report = {
            "schema": AUDIT_SCHEMA,
            "structurally_valid": False,
            "intake_accepted": False,
            "failures": sorted(set(failures)),
            "blockers": [],
            "audit_root_sha256": None,
        }
        report["audit_root_sha256"] = rooted(report, "audit_root_sha256")
        return report["failures"], [], report
    add(failures, manifest.get("schema") == MANIFEST_SCHEMA, "manifest.schema")
    add(
        failures,
        rooted(manifest, "manifest_root_sha256") == manifest.get("manifest_root_sha256"),
        "manifest.root",
    )
    tooling = manifest.get("tooling")
    if exact(
        failures,
        tooling,
        {"builder_sha256", "verifier_sha256", "schema_sha256"},
        "tooling",
    ):
        add(
            failures,
            BUILDER.is_file() and tooling.get("builder_sha256") == digest_file(BUILDER),
            "tooling.builder",
        )
        add(
            failures,
            tooling.get("verifier_sha256") == digest_file(Path(__file__).resolve()),
            "tooling.verifier",
        )
        add(
            failures,
            SPEC.is_file() and tooling.get("schema_sha256") == digest_file(SPEC),
            "tooling.schema",
        )
    artifacts = ArtifactIndex(
        manifest.get("artifacts"), bundle_root, failures, verify_files
    )
    validate_job(manifest.get("job"), artifacts, failures)
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    validate_runtime(runtime, artifacts, failures)
    validate_policy(manifest.get("capture_policy"), failures)
    opaque_ids, access_ids = validate_issue_catalogs(
        manifest, artifacts, failures, blockers
    )
    validate_feature_coverage(
        manifest.get("feature_coverage"),
        artifacts,
        opaque_ids,
        access_ids,
        failures,
        blockers,
    )
    runs = manifest.get("runs")
    add(failures, isinstance(runs, list) and len(runs) == 2, "runs.count")
    run_results: list[dict[str, Any]] = []
    if isinstance(runs, list):
        for index, run in enumerate(runs):
            run_results.append(
                validate_run(
                    run,
                    index,
                    runtime,
                    artifacts,
                    opaque_ids,
                    access_ids,
                    failures,
                    blockers,
                )
            )
    if isinstance(runs, list) and len(runs) == 2 and all(isinstance(run, dict) for run in runs):
        processes = [run.get("process", {}) for run in runs]
        for field in (
            "pid",
            "process_start_token_sha256",
            "nonce_sha256",
            "queue_identity_sha256",
        ):
            add(
                failures,
                len({process.get(field) for process in processes}) == 2,
                f"runs.fresh_process_independence.{field}",
            )
        add(
            failures,
            len({process.get("device_luid_sha256") for process in processes}) == 1,
            "runs.device_luid_repeat",
        )
        add(
            failures,
            len({run.get("run_id") for run in runs}) == 2,
            "runs.run_ids_unique",
        )
        add(
            failures,
            len({run.get("run_root_sha256") for run in runs}) == 2,
            "runs.roots_distinct",
        )
    validate_derived(manifest, failures)
    derived = manifest.get("derived", {})
    if derived.get("semantic_repeat_equal") is not True:
        blockers.append("cross_run.semantic_repeat")
    if isinstance(runs, list) and len(runs) == 2:
        for field in ("environment",):
            add(
                failures,
                semantic_projection({field: runs[0].get(field)}, {entry.get("id"): entry.get("sha256") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)})
                == semantic_projection({field: runs[1].get(field)}, {entry.get("id"): entry.get("sha256") for entry in manifest.get("artifacts", []) if isinstance(entry, dict)}),
                f"runs.{field}_repeat",
            )
    unused = sorted(set(artifacts.by_id) - artifacts.used)
    failures.extend(f"artifacts.unused.{artifact_id}" for artifact_id in unused)
    claims = manifest.get("claims")
    claim_fields = {
        "intake_complete",
        "fg_runtime_executed",
        "two_process_semantic_repeat",
        "interpolated_outputs_captured",
        "create_evaluate_transport_captured",
        "kernel_graph_captured",
        "reengineering_complete",
        "mac_port_complete",
        "optimized",
    }
    declared_blocking = bool(blockers)
    if exact(failures, claims, claim_fields, "claims"):
        add(
            failures,
            claims.get("intake_complete") is (not declared_blocking),
            "claims.intake_complete",
        )
        logs_ok = len(run_results) == 2 and all(result.get("logs_ok") for result in run_results)
        add(
            failures,
            claims.get("fg_runtime_executed") is logs_ok,
            "claims.runtime_executed",
        )
        add(
            failures,
            claims.get("two_process_semantic_repeat")
            is (derived.get("semantic_repeat_equal") is True),
            "claims.semantic_repeat",
        )
        interpolated_captured = len(run_results) == 2 and all(
            result.get("evaluation_count", 0) >= 2 for result in run_results
        )
        add(
            failures,
            claims.get("interpolated_outputs_captured") is interpolated_captured,
            "claims.interpolated_outputs",
        )
        add(
            failures,
            claims.get("create_evaluate_transport_captured")
            is (logs_ok and interpolated_captured),
            "claims.transport",
        )
        graph_captured = len(run_results) == 2 and all(
            result.get("kernel_graph_captured") for result in run_results
        )
        add(
            failures,
            claims.get("kernel_graph_captured") is graph_captured,
            "claims.kernel_graph",
        )
        for field in ("reengineering_complete", "mac_port_complete", "optimized"):
            add(failures, claims.get(field) is False, f"claims.no_overclaim.{field}")
    failures = sorted(set(failures))
    blockers = sorted(set(blockers))
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "manifest_root_sha256": manifest.get("manifest_root_sha256"),
        "artifact_count": len(artifacts.by_id),
        "run_count": len(runs) if isinstance(runs, list) else 0,
        "cross_run_semantic_sha256": derived.get("cross_run_semantic_sha256"),
        "claims": copy.deepcopy(claims),
        "structurally_valid": not failures,
        "intake_accepted": not failures and not blockers,
        "failures": failures,
        "blockers": blockers,
        "audit_root_sha256": None,
    }
    report["audit_root_sha256"] = rooted(report, "audit_root_sha256")
    return failures, blockers, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--emit", type=Path)
    parser.add_argument(
        "--skip-file-rehash",
        action="store_true",
        help="structural diagnostics only; never use for acceptance",
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        failures, blockers, report = verify_manifest(
            manifest, args.bundle_root.resolve(), verify_files=not args.skip_file_rehash
        )
    except (OSError, json.JSONDecodeError) as error:
        print(f"FG intake verification failed: {error}", file=sys.stderr)
        return 2
    if args.skip_file_rehash:
        report["intake_accepted"] = False
        report["blockers"] = sorted(set(report["blockers"] + ["file_rehash.skipped"]))
        report["audit_root_sha256"] = rooted(report, "audit_root_sha256")
        blockers = report["blockers"]
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures or blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
