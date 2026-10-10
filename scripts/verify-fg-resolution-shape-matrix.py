#!/usr/bin/env python3
"""Verify repeat-exact RTX FG shape selection at benchmark resolutions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_SHA256 = "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"
HOST_SHA256 = "c3a9e699381c3edaa6d142c880b9177d73b8bad0bfc8e9dc5cd508aa0655fa56"
PROXY_SHA256 = "08f1b60155300bfbc7647d5e360503229077a4482336daa5e97714044e6d9efe"
ORCHESTRATION_SHA256 = "89a7dd87042c3208dd57d7a096cb13502da58699cb0d57cd8064cbf840ead74a"
RAW_RECEIPT_SHA256 = "8b2f70c28775d9da31767131eeb781442bbc67d075e4545cf5fb7e03ef649665"
JOB_ID = "fg-resolution-shapes-v2-20260904T132417Z"

RESOLUTIONS = {
    "720p": {
        "width": 1280,
        "height": 720,
        "output_sha256": "d4643a9f5a6b8d20603223fa4ae7e046494387475305371edee95596a796acaa",
        "estimated_vram_bytes": 121149386,
        "gpu_memory_max_observed_bytes": 333725696,
    },
    "1080p": {
        "width": 1920,
        "height": 1080,
        "output_sha256": "3caff1261646ff9f90807f2401444be535fe006e9abfbf0f8b14b856353d5dcf",
        "estimated_vram_bytes": 207395206,
        "gpu_memory_max_observed_bytes": 448237568,
    },
    "1440p": {
        "width": 2560,
        "height": 1440,
        "output_sha256": "077bb40bdcf2e0398ffc19c7ca8c45cd48d7679fa4e8c4fd9c758548985d022b",
        "estimated_vram_bytes": 365419256,
        "gpu_memory_max_observed_bytes": 633442304,
    },
    "4k": {
        "width": 3840,
        "height": 2160,
        "output_sha256": "516c2a5542cc3b8de84a9bb64d389973d9ac5420bfe3be6861da85723ae31a72",
        "estimated_vram_bytes": 544354646,
        "gpu_memory_max_observed_bytes": 884305920,
    },
}

POISONS = {"a5": 165, "5a": 90}


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL object required: {path}")
            result.append(value)
    require(result, f"empty JSONL input: {path}")
    return result


def normalized_shape(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase = "__bootstrap__"
    shape = []
    for event in trace:
        if event.get("event") == "phase_marker":
            phase = event.get("phase")
        elif event.get("event") == "kernel_launch":
            require(event.get("parameter_mode") == "packed", "selected launch is not packed")
            require(event.get("parameter_captured") is True, "selected parameters were not captured")
            shape.append({
                "phase": phase,
                "launch_token": event["launch_token"],
                "kernel_index": event["kernel_index"],
                "module_token": event["module_token"],
                "function_token": event["function_token"],
                "function_name": event["function_name"],
                "grid": event["grid"],
                "block": event["block"],
                "dynamic_shared_bytes": event["dynamic_shared_bytes"],
                "parameter_bytes": event["parameter_bytes"],
            })
    return shape


def topology(shape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": ordinal,
            "phase": row["phase"],
            "module_token": row["module_token"],
            "function_token": row["function_token"],
            "function_name": row["function_name"],
            "block": row["block"],
            "dynamic_shared_bytes": row["dynamic_shared_bytes"],
            "parameter_bytes": row["parameter_bytes"],
        }
        for ordinal, row in enumerate(shape)
    ]


def added_resolution_launches(shape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in shape:
        if row["module_token"] not in {"module-000022", "module-000024"}:
            continue
        result.append({
            "phase": row["phase"],
            "launch_token": row["launch_token"],
            "module_token": row["module_token"],
            "grid": row["grid"],
            "block": row["block"],
            "parameter_bytes": row["parameter_bytes"],
        })
    require(
        [(row["phase"], row["module_token"]) for row in result] == [
            ("evaluate_0", "module-000022"),
            ("evaluate_0", "module-000024"),
            ("evaluate_1", "module-000022"),
            ("evaluate_1", "module-000024"),
        ],
        "resolution-dependent launch sequence drift",
    )
    require(all(row["block"] == [256, 1, 1] for row in result), "resolution launch block drift")
    require(
        [row["parameter_bytes"] for row in result] == [56, 64, 56, 64],
        "resolution launch ABI extent drift",
    )
    return result


def validate_run(
    root: Path,
    row: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[Any, Any, Any]]:
    run_id = row.get("run_id")
    require(isinstance(run_id, str), "run ID missing")
    run_root = root / run_id
    process_path = run_root / f"{run_id}-process.json"
    host_path = run_root / "fg-run-receipt.json"
    trace_path = run_root / "nvapi-trace.jsonl"
    shape_path = run_root / "selected-shape.json"
    for path in (process_path, host_path, trace_path, shape_path):
        require(path.is_file() and not path.is_symlink(), f"required run artifact missing: {path}")

    process = load_json(process_path)
    host = load_json(host_path)
    require(isinstance(process, dict) and isinstance(host, dict), "run receipts must be objects")
    require(process.get("schema") == "neuralstudio.dlss_fg.traced_process.v1", "process schema drift")
    require(process.get("exit_code") == 0 and process.get("error") is None, "process failed")
    require(process.get("run_id") == run_id, "process run ID drift")
    require(process.get("runtime_sha256") == RUNTIME_SHA256, "process runtime drift")
    require(process.get("proxy_sha256") == PROXY_SHA256, "process proxy drift")
    require(process.get("host_receipt_sha256") == sha_file(host_path), "process host hash drift")
    require(process.get("trace_sha256") == sha_file(trace_path), "process trace hash drift")
    require(row.get("process_receipt_sha256") == sha_file(process_path), "matrix process hash drift")
    require(row.get("host_receipt_sha256") == sha_file(host_path), "matrix host hash drift")
    require(row.get("nvapi_trace_sha256") == sha_file(trace_path), "matrix trace hash drift")

    require(host.get("accepted") is True and host.get("runtime_sha256") == RUNTIME_SHA256, "host rejected")
    require(host.get("width") == expected["width"] and host.get("height") == expected["height"], "host dimensions drift")
    require(host.get("resource_profile") == "core+disable_output", "resource profile drift")
    require(host.get("multiframe_count") == 1 and host.get("multiframe_index") == 1, "multiframe contract drift")
    require(host.get("d3d12_severe_messages") == 0, "D3D12 validation error")
    frames = host.get("frames")
    require(isinstance(frames, list) and len(frames) == 2, "frame receipt count drift")
    generated = frames[1]
    require(generated.get("generated_output_accepted") is True, "generated output rejected")
    require(generated.get("output_untouched_pixels") == 0, "generated output left poison pixels")
    require(generated.get("output_interpolated_sha256") == expected["output_sha256"], "generated output hash drift")

    trace = load_jsonl(trace_path)
    actual_shape = normalized_shape(trace)
    saved_shape = load_json(shape_path)
    require(isinstance(saved_shape, list) and saved_shape == actual_shape, "saved shape disagrees with trace")
    require(row.get("shape_sha256") == sha_file(shape_path), "saved shape hash drift")
    require(len(actual_shape) == 89, "large-resolution selected launch count drift")
    reset_count = sum(item["phase"] == "evaluate_0" for item in actual_shape)
    generated_count = sum(item["phase"] == "evaluate_1" for item in actual_shape)
    require(reset_count == row.get("reset_launches") == 11, "reset launch count drift")
    require(generated_count == row.get("generated_launches") == 78, "generated launch count drift")
    added_resolution_launches(actual_shape)

    require(row.get("width") == expected["width"] and row.get("height") == expected["height"], "matrix dimensions drift")
    require(row.get("output_bytes") == expected["width"] * expected["height"] * 4, "matrix output extent drift")
    require(row.get("generated_output_sha256") == expected["output_sha256"], "matrix output hash drift")
    require(row.get("generated_output_untouched_pixels") == 0, "matrix poison coverage drift")
    require(row.get("estimated_vram_bytes") == expected["estimated_vram_bytes"], "VRAM estimate drift")
    require(row.get("gpu_memory_max_observed_bytes") == expected["gpu_memory_max_observed_bytes"], "observed memory drift")
    identity = (host.get("process_id"), host.get("run_nonce"), host.get("queue_nonce"))
    return actual_shape, identity


def validate_normalized(result: dict[str, Any]) -> None:
    require(result.get("fresh_process_count") == 16, "normalized process count drift")
    require(result.get("poison_values") == [165, 90], "normalized poison matrix drift")
    require(result.get("all_outputs_poison_process_invariant") is True, "output invariance missing")
    require(result.get("all_shapes_repeat_exact") is True, "shape repeat gate missing")
    require(result.get("large_resolution_topology_repeat_exact") is True, "topology repeat gate missing")
    entries = result.get("resolutions")
    require(isinstance(entries, list) and len(entries) == 4, "normalized resolution count drift")
    for entry in entries:
        expected = RESOLUTIONS.get(entry.get("name"))
        require(expected is not None, "unknown normalized resolution")
        require(entry.get("width") == expected["width"] and entry.get("height") == expected["height"], "normalized dimensions drift")
        require(entry.get("reset_launches") == 11 and entry.get("generated_launches") == 78, "normalized launch count drift")
        require(entry.get("generated_output_sha256") == expected["output_sha256"], "normalized output drift")
        require(entry.get("output_untouched_pixels") == 0, "normalized coverage drift")
        require(len(entry.get("added_resolution_launches", [])) == 4, "normalized added launch census drift")
    require(result.get("claims", {}).get("performance") is False, "trace timing promoted to performance")
    require(result.get("claims", {}).get("complete_metal_parity") is False, "RTX shape receipt promoted to Metal parity")
    require(result.get("address_fields_recorded") is False, "normalized receipt contains addresses")


def negative_audit(result: dict[str, Any]) -> dict[str, Any]:
    cases = []

    def reject(name: str, mutation: Any) -> None:
        value = copy.deepcopy(result)
        mutation(value)
        accepted = True
        try:
            validate_normalized(value)
        except VerificationError:
            accepted = False
        require(not accepted, f"normalized mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})

    reject("process-count", lambda value: value.__setitem__("fresh_process_count", 15))
    reject("poison-matrix", lambda value: value.__setitem__("poison_values", [165]))
    reject("output-invariance", lambda value: value.__setitem__("all_outputs_poison_process_invariant", False))
    reject("shape-repeat", lambda value: value.__setitem__("all_shapes_repeat_exact", False))
    reject("topology-repeat", lambda value: value.__setitem__("large_resolution_topology_repeat_exact", False))
    reject("dimensions", lambda value: value["resolutions"][0].__setitem__("width", 640))
    reject("launch-count", lambda value: value["resolutions"][0].__setitem__("generated_launches", 76))
    reject("output-hash", lambda value: value["resolutions"][1].__setitem__("generated_output_sha256", "0" * 64))
    reject("added-module", lambda value: value["resolutions"][2]["added_resolution_launches"].pop())
    reject("performance-overclaim", lambda value: value["claims"].__setitem__("performance", True))
    reject("metal-overclaim", lambda value: value["claims"].__setitem__("complete_metal_parity", True))
    reject("address-overclaim", lambda value: value.__setitem__("address_fields_recorded", True))
    audit = {
        "schema": "neuralstudio.fg.resolution-shape-negative-audit.v1",
        "case_count": len(cases),
        "rejected_count": len(cases),
        "cases": cases,
        "result": "pass",
    }
    audit["receipt_root_sha256"] = canonical_sha(audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.input_root.resolve()
    receipt_path = arguments.receipt.resolve()
    repository = arguments.repository.resolve()
    raw = load_json(receipt_path)
    require(isinstance(raw, dict), "raw matrix receipt must be an object")
    require(sha_file(receipt_path) == RAW_RECEIPT_SHA256, "raw matrix receipt drift")
    require(raw.get("schema") == "neuralstudio.fg.resolution-shape-matrix.v1", "raw matrix schema drift")
    require(raw.get("job_id") == JOB_ID, "raw matrix job drift")
    require(raw.get("runtime_sha256") == RUNTIME_SHA256, "raw runtime drift")
    require(raw.get("host_binary_sha256") == HOST_SHA256, "raw host drift")
    require(raw.get("proxy_sha256") == PROXY_SHA256, "raw proxy drift")
    require(raw.get("orchestration_source_sha256") == ORCHESTRATION_SHA256, "raw orchestration drift")
    require(sha_file(repository / "scripts/run-fg-resolution-shape-matrix.ps1") == ORCHESTRATION_SHA256, "local orchestration drift")
    require(raw.get("fresh_processes") == 16 and raw.get("poison_values") == [165, 90], "raw run matrix drift")
    require(raw.get("trace_instrumented_timings_are_performance_evidence") is False, "trace timing overclaim")
    require(raw.get("expected_or_archived_tensor_ingress") is False, "forbidden tensor ingress")
    require(raw.get("address_fields_recorded") is False and raw.get("result") == "pass", "raw receipt rejected")

    rows = raw.get("runs")
    require(isinstance(rows, list) and len(rows) == 16, "raw run count drift")
    rows_by_resolution: dict[str, list[dict[str, Any]]] = {name: [] for name in RESOLUTIONS}
    identities = []
    shapes: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in RESOLUTIONS}
    for row in rows:
        require(isinstance(row, dict), "run row must be an object")
        name = row.get("resolution")
        expected = RESOLUTIONS.get(name)
        require(expected is not None, "unknown raw resolution")
        poison = row.get("poison_byte")
        ordinal = row.get("process_ordinal")
        require(poison in POISONS.values() and ordinal in {"a", "b"}, "raw poison/process cell drift")
        expected_run = f"shape-{name}-poison-{'a5' if poison == 165 else '5a'}-process-{ordinal}"
        require(row.get("run_id") == expected_run, "raw run ID drift")
        shape, identity = validate_run(root, row, expected)
        rows_by_resolution[name].append(row)
        shapes[name].append(shape)
        identities.append(identity)
    require(len(set(identities)) == 16, "fresh process identities collapsed")

    resolution_receipts = []
    topology_roots = []
    for name, expected in RESOLUTIONS.items():
        group = rows_by_resolution[name]
        require(len(group) == 4, f"resolution cell count drift: {name}")
        require({row["poison_byte"] for row in group} == {165, 90}, f"poison coverage drift: {name}")
        require(len({row["generated_output_sha256"] for row in group}) == 1, f"output repeat drift: {name}")
        require(all(shape == shapes[name][0] for shape in shapes[name][1:]), f"shape repeat drift: {name}")
        shape = shapes[name][0]
        topology_root = canonical_sha(topology(shape))
        topology_roots.append(topology_root)
        resolution_receipts.append({
            "name": name,
            "width": expected["width"],
            "height": expected["height"],
            "reset_launches": 11,
            "generated_launches": 78,
            "shape_root_sha256": canonical_sha(shape),
            "topology_root_sha256": topology_root,
            "added_resolution_launches": added_resolution_launches(shape),
            "generated_output_bytes": expected["width"] * expected["height"] * 4,
            "generated_output_sha256": expected["output_sha256"],
            "output_untouched_pixels": 0,
            "estimated_vram_bytes": expected["estimated_vram_bytes"],
            "gpu_memory_max_observed_bytes": expected["gpu_memory_max_observed_bytes"],
            "fresh_processes": 4,
            "poison_values": [165, 90],
        })
    require(len(set(topology_roots)) == 1, "large-resolution topology differs")

    result = {
        "schema": "neuralstudio.fg.resolution-shape-verification.v1",
        "job_id": JOB_ID,
        "pinned_runtime_sha256": RUNTIME_SHA256,
        "host_binary_sha256": HOST_SHA256,
        "proxy_sha256": PROXY_SHA256,
        "orchestration_source_sha256": ORCHESTRATION_SHA256,
        "fresh_process_count": 16,
        "fresh_process_identities_distinct": True,
        "poison_values": [165, 90],
        "all_outputs_poison_process_invariant": True,
        "all_shapes_repeat_exact": True,
        "large_resolution_topology_repeat_exact": True,
        "large_resolution_topology_root_sha256": topology_roots[0],
        "resolutions": resolution_receipts,
        "new_selected_modules_above_640x360": ["module-000022", "module-000024"],
        "trace_instrumented_timings_are_performance_evidence": False,
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "rtx_resolution_graph_mapped": True,
            "rtx_outputs_repeat_exact": True,
            "complete_metal_parity": False,
            "complete_resolution_correctness": False,
            "netal_integration": False,
            "performance": False,
        },
        "remaining_limits": [
            "Module22 and module24 require exact Metal reconstruction before large-resolution parity.",
            "These are trace-instrumented correctness runs, not latency or frame-pacing benchmarks.",
            "The observed memory samples are planning data and require production-path confirmation.",
        ],
        "result": "pass",
    }
    validate_normalized(result)
    result["receipt_root_sha256"] = canonical_sha(result)
    audit = negative_audit(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.negative_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"negative_root_sha256={audit['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
