#!/usr/bin/env python3
"""Freeze the address-free selected early-path dependency graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tracked(repo: Path, path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{path}"], cwd=repo,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    inventory_path = repo / "evidence/specs/fg-early-launch-inventory-v1/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    launches = [item for item in inventory["launches"]
                if 3 <= item["canonical_launch"] <= 19]
    if len(launches) != 17:
        raise RuntimeError("selected early launch census drift")
    dependencies = {
        "l3_l13": tracked(repo, "src/fg_early_l3_l13/early_l3_l13.metal"),
        "l4_l14": tracked(repo, "src/fg_early_l4_l14/early_l4_l14.metal"),
        "l5_l15": tracked(repo, "src/fg_early_l5_l15/early_l5_l15.metal"),
        "l6_l16": tracked(repo, "src/fg_early_l6_l16/early_l6_l16.metal"),
        "l7_l17": tracked(repo, "src/fg_early_l7_l17/early_l7_l17.metal"),
        "l8_l18": tracked(repo, "src/fg_early_l8_l18/early_l8_l18.metal"),
        "l9": tracked(repo, "src/fg_early_l9/early_l9.metal"),
        "l10_l12": tracked(repo, "src/fg_early_status_l10_l12/status_write.metal"),
        "l11": tracked(repo, "src/fg_early_l11/early_l11.metal"),
        "l19": tracked(repo, "src/fg_early_l19/early_l19.metal"),
    }
    edges = [
        [3,9,"allocation-000024"], [3,13,"allocation-000004-history-bank-b"],
        [3,19,"allocation-000024"],
        [4,5,"allocation-000007"], [4,5,"allocation-000013"],
        [4,6,"allocation-000007"], [4,16,"allocation-000013"],
        [5,6,"allocation-000020"], [5,16,"allocation-000016"],
        [6,7,"allocation-000022"], [6,7,"allocation-000023"],
        [6,7,"allocation-000043"], [6,7,"allocation-000044"],
        [6,7,"allocation-000040"], [6,7,"allocation-000008"],
        [7,8,"allocation-000008"], [8,9,"allocation-000031"],
        [9,19,"allocation-000026"],
        [13,19,"allocation-000025"], [13,19,"allocation-000030"],
        [14,15,"allocation-000007"], [14,15,"allocation-000012"],
        [14,16,"allocation-000007"], [14,19,"allocation-000007"],
        [15,16,"allocation-000020"], [15,16,"allocation-000015"],
        [16,17,"allocation-000022"], [16,17,"allocation-000023"],
        [16,17,"allocation-000043"], [16,17,"allocation-000044"],
        [16,17,"allocation-000040"], [16,17,"allocation-000008"],
        [16,19,"allocation-000017"], [17,18,"allocation-000008"],
        [17,19,"allocation-000008"], [18,19,"allocation-000032"],
        [19,20,"allocation-000037"],
    ]
    result = {
        "schema": "neuralstudio.fg.connected-l3-l19-edge-map.v1",
        "source_roots": {
            "early_inventory": inventory["receipt_root_sha256"],
            "selected_graph":
                "fba92f31c5c91c85d842394cffd8bb3d09216a52ec8db13c82db5d27fedb7fe4",
            "allocation_provenance":
                "8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c",
            "descriptor_provenance":
                "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733",
            "allocation29_first_generated":
                "c1435760ec194050aa1dd4dfd2d82f71e9aa7874ce550328a4eb08cd3d2a0018",
            "allocation29_temporal_reuse":
                "ebb4c6a84a4cf0e607ef62b39a55dc46ec0ec059f3d02bc1b77e4d084ef9a79b",
        },
        "launches": launches,
        "live_edges": [
            {"producer": p, "consumer": c, "resource": r}
            for p, c, r in edges
        ],
        "persistent_state": {
            "history_ping_pong": {
                "l3": {"read": "allocation-000003", "write": "allocation-000004"},
                "l13": {"read": "allocation-000004", "write": "allocation-000003"},
            },
            "depth_mask_cross_bank": {
                "l6": ["allocation-000012", "allocation-000015"],
                "l16": ["allocation-000013", "allocation-000016"],
            },
            "allocation29_first_generated_initialization": "generate half 0.5 live",
            "allocation29_evidence_bytes_runtime_ingress": False,
        },
        "dependencies_committed": dependencies,
        "full_executor_ready": all(dependencies.values()),
        "fail_closed_missing": [name for name, present in dependencies.items()
                                if not present],
        "expected_or_archived_intermediate_ingress": False,
        "addresses_recorded": False,
        "result": "pass" if all(dependencies.values()) else "blocked-on-commits",
    }
    result["receipt_root_sha256"] = canonical(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"result={result['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
