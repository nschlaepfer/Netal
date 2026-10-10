#!/usr/bin/env python3
"""Fail-closed audit of stable FG Metal sources against live resolution shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RESOLUTIONS = ("720p", "1080p", "1440p", "4k")
MODULE_OWNERS = {
    3: "src/fg_mid_l55", 4: "src/fg_early_l11",
    5: "src/fg_early_l19", 6: "src/fg_mid_l52",
    11: "src/fg_early_l3_l13", 12: "src/fg_mid_l53",
    13: "src/fg_early_l5_l15", 14: "src/fg_early_l9",
    15: "src/fg_early_l4_l14", 17: "src/fg_postprocess_l84",
    18: "src/fg_postprocess_l85", 19: "src/fg_postprocess_l86",
    20: "src/fg_compositor_l87", 21: "src/fg_early_l6_l16",
    22: "src/fg_resolution_preprocess_l22_l24",
    23: "src/fg_early_l7_l17",
    24: "src/fg_resolution_preprocess_l22_l24",
    25: "src/fg_early_l8_l18", 26: "src/fg_mid_l54",
    27: "src/fg_early_status_l10_l12", 28: "src/fg_conv",
    29: "src/fg_pooling", 30: "src/fg_conv", 31: "src/fg_conv",
    32: "src/fg_conv", 33: "src/fg_conv", 34: "src/fg_conv",
    35: "src/fg_conv", 36: "src/fg_upscale",
    37: "src/fg_elementwise", 38: "src/fg_conv", 39: "src/fg_conv",
    40: "src/fg_upscale", 41: "src/fg_elementwise",
    42: "src/fg_conv", 43: "src/fg_conv", 44: "src/fg_conv",
    45: "src/fg_upscale", 46: "src/fg_elementwise",
    47: "src/fg_conv", 48: "src/fg_conv", 49: "src/fg_upscale",
    50: "src/fg_elementwise", 51: "src/fg_conv", 52: "src/fg_conv",
    53: "src/fg_initial_merge", 54: "src/fg_conv", 55: "src/fg_conv",
    56: "src/fg_conv", 57: "src/fg_conv", 58: "src/fg_conv",
    59: "src/fg_conv", 60: "src/fg_custom_upsample",
    61: "src/fg_central_block", 62: "src/fg_conv", 63: "src/fg_conv",
    64: "src/fg_conv", 65: "src/fg_conv", 66: "src/fg_conv",
}
DIAGNOSTICS = {
    "src/fg_mid_l52/ex2_probe.metal",
    "src/fg_mid_l52/macos_ex2_probe.mm",
    "src/fg_mid_l54/semantics_probe.metal",
    "src/fg_mid_l54/macos_semantics_probe.mm",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def rooted(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value["receipt_root_sha256"] = sha(canonical(value))
    return value


def tracked_metal_sources(repository: Path) -> list[str]:
    process = subprocess.run(
        ["git", "ls-files", "src/fg_*/*.metal", "src/fg_*/macos*.mm"],
        cwd=repository, check=True, capture_output=True, text=True)
    return sorted(line for line in process.stdout.splitlines() if line)


def source_class(path: str) -> str:
    if path in DIAGNOSTICS:
        return "diagnostic-not-selected-graph"
    if path.startswith("src/fg_resolution_preprocess_l22_l24/"):
        return "proven-720p-1080p-1440p-4k-standalone"
    if path.startswith("src/fg_early_status_l10_l12/"):
        return "shape-invariant-control-unit"
    if path.startswith("src/fg_connected_"):
        return "connected-host-fixed-720p-contract"
    return "standalone-receipt-fixed-720p-contract"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    shapes: dict[str, list[dict[str, Any]]] = {}
    shape_hashes: dict[str, str] = {}
    for resolution in RESOLUTIONS:
        path = repository / f"evidence/specs/fg-resolution-shapes-v1/shape-{resolution}.json"
        shapes[resolution] = json.loads(path.read_text())
        shape_hashes[resolution] = sha(path.read_bytes())
        if len(shapes[resolution]) != 89:
            raise SystemExit(f"{resolution} launch count drift")
    launch_rows = []
    varying_count = 0
    proven_dynamic_count = 0
    invariant_control_count = 0
    for index in range(89):
        rows = [shapes[resolution][index] for resolution in RESOLUTIONS]
        launch = rows[0]["launch_token"]
        module = int(rows[0]["module_token"].split("-")[-1])
        if any(row["launch_token"] != launch or
               int(row["module_token"].split("-")[-1]) != module
               for row in rows):
            raise SystemExit(f"launch/module alignment drift at index {index}")
        grids = {resolution: row["grid"]
                 for resolution, row in zip(RESOLUTIONS, rows)}
        varies = len({tuple(grid) for grid in grids.values()}) != 1
        varying_count += int(varies)
        if module in (22, 24):
            support = "proven-four-resolution-standalone"
            proven_dynamic_count += 1
        elif module == 27:
            support = "shape-invariant-control-unit"
            invariant_control_count += 1
        else:
            support = "large-resolution-unproven-fail-closed"
        launch_rows.append({
            "launch": launch,
            "phase": rows[0]["phase"],
            "module_token": module,
            "function_token": int(rows[0]["function_token"].split("-")[-1]),
            "owner": MODULE_OWNERS[module],
            "parameter_bytes": rows[0]["parameter_bytes"],
            "block": rows[0]["block"],
            "grids": grids,
            "grid_varies": varies,
            "large_resolution_support": support,
        })
    if varying_count != 87 or proven_dynamic_count != 4 or invariant_control_count != 2:
        raise SystemExit("resolution support count drift")

    source_rows = []
    class_counts: dict[str, int] = {}
    for relative in tracked_metal_sources(repository):
        path = repository / relative
        data = path.read_bytes()
        text = data.decode(errors="replace")
        classification = source_class(relative)
        class_counts[classification] = class_counts.get(classification, 0) + 1
        source_rows.append({
            "path": relative,
            "bytes": len(data),
            "sha256": sha(data),
            "kind": "metal-kernel" if relative.endswith(".metal") else "macos-host",
            "shape_classification": classification,
            "contains_640_literal": bool(re.search(r"(?<![0-9])640(?![0-9])", text)),
            "contains_360_literal": bool(re.search(r"(?<![0-9])360(?![0-9])", text)),
        })
    source_manifest_root = sha(canonical([
        {key: row[key] for key in ("path", "bytes", "sha256")}
        for row in source_rows
    ]))
    first = launch_rows[0]
    if (first["launch"] != "launch-000003" or first["module_token"] != 11 or
        first["grids"]["720p"] != [80, 45, 1]):
        raise SystemExit("first parity break evidence drift")
    contract_path = repository / "src/fg_early_l3_l13/contract.h"
    host_path = repository / "src/fg_early_l3_l13/macos_probe.mm"
    first_break = {
        "launch": "launch-000003",
        "canonical_85_launch": 3,
        "module_token": 11,
        "function_token": 9,
        "first_unsupported_resolution": "720p",
        "kind": "resource-extent-and-dispatch-contract-mismatch-before-arithmetic",
        "accepted_original_grid": [40, 23, 1],
        "required_720p_grid": [80, 45, 1],
        "accepted_source_extent": [640, 360],
        "required_720p_source_extent": [1280, 720],
        "accepted_output_extent": [320, 180],
        "required_720p_output_extent": [640, 360],
        "accepted_output_pixels": 57600,
        "required_720p_output_pixels": 230400,
        "uncovered_output_pixels_if_current_host_reused": 172800,
        "accepted_original_abi_receipt_root_sha256":
            "52fe85735eb053ada59e26846dd661019d6f792ccbdc93014d41721aeabed929",
        "contract_source": {
            "path": "src/fg_early_l3_l13/contract.h",
            "sha256": sha(contract_path.read_bytes()),
            "anchors": ["kWidth = 640", "kHeight = 360",
                        "kHalfWidth = 320", "kHalfHeight = 180"],
        },
        "host_source": {
            "path": "src/fg_early_l3_l13/macos_probe.mm",
            "sha256": sha(host_path.read_bytes()),
            "anchor": "dispatchThreads(contract::kHalfWidth, contract::kHalfHeight)",
        },
        "output_bytes_compared": 0,
        "reason_no_numeric_compare":
            "the accepted host cannot allocate or dispatch the required extent; parity fails closed before a valid large-resolution run exists",
        "large_resolution_parity": False,
    }
    ranking = [
        {"rank": 1, "module": 11, "canonical_launches": [3, 13],
         "large_trace_launches": [3, 15], "owner": "src/fg_early_l3_l13",
         "reason": "first dependency; selected kernel already consumes dimensions, while the contract and host are fixed"},
        {"rank": 2, "module": 15, "canonical_launches": [4, 14],
         "large_trace_launches": [4, 16], "owner": "src/fg_early_l4_l14",
         "reason": "next typed preprocessing dependency; extend extents and branch-grid fixtures"},
        {"rank": 3, "module": 13, "canonical_launches": [5, 15],
         "large_trace_launches": [5, 17], "owner": "src/fg_early_l5_l15",
         "reason": "next dependency; resize deterministic collision fixtures and preserve the exact multi-pass atomic lowering"},
        {"rank": 4, "module": 21, "canonical_launches": [6, 16],
         "large_trace_launches": [6, 18], "owner": "src/fg_early_l6_l16",
         "reason": "feeds newly exact module22; resize typed resources and five output banks"},
        {"rank": 5, "module": 23, "canonical_launches": [7, 17],
         "large_trace_launches": [8, 20], "owner": "src/fg_early_l7_l17",
         "reason": "repair stage after module22; preserve selected mode, mask coverage, and reciprocal behavior"},
    ]
    receipt = rooted({
        "schema": "neuralstudio.fg.dynamic-shape-audit.v1",
        "audit_basis": "tracked stable Metal kernels and macOS hosts at the audit commit",
        "shape_matrix_receipt_root_sha256":
            "361d224f422bf44e18d8db4d9f94f74a862d115251402d8ebb1cfc05fd58af9a",
        "shape_evidence_sha256": shape_hashes,
        "launch_count": 89,
        "grid_varying_launch_count": varying_count,
        "shape_invariant_control_launch_count": invariant_control_count,
        "proven_dynamic_standalone_launch_count": proven_dynamic_count,
        "unresolved_dynamic_launch_count": 83,
        "large_resolution_connected_parity": False,
        "launches": launch_rows,
        "tracked_metal_source_count": len(source_rows),
        "tracked_metal_source_manifest_sha256": source_manifest_root,
        "source_class_counts": class_counts,
        "sources": source_rows,
        "first_fail_closed_parity_break": first_break,
        "next_dependency_closed_units": ranking,
        "new_rig_job_required": False,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"tracked_metal_source_count={len(source_rows)}")
    print(f"unresolved_dynamic_launch_count={receipt['unresolved_dynamic_launch_count']}")
    print(f"first_fail_closed_launch={first_break['launch']}")
    print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
