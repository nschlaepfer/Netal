#!/usr/bin/env python3
"""Build the address-free live resource-edge contract for FG L20-L87."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    allocation = json.loads((repo /
        "build/fg_rig/fg2x-intake-20260904T082417Z/"
        "fg-allocation-provenance-frozen-a.json").read_text())
    descriptors = json.loads((repo /
        "evidence/specs/fg-resource-descriptor-provenance-v1/"
        "real-resource-descriptor-provenance.json").read_text())
    census = json.loads((repo /
        "evidence/specs/fg-selected-module-census-v1/"
        "selected-module-census.json").read_text())
    require(allocation["receipt_root_sha256"] ==
            "8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c",
            "allocation provenance root drift")
    require(descriptors["receipt_root_sha256"] ==
            "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733",
            "descriptor provenance root drift")
    require(census["census_root_sha256"] ==
            "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26",
            "selected census root drift")

    comparisons = allocation["canonical_parameters"]["comparisons"]
    by_module: dict[int, dict[str, Any]] = {}
    for item in comparisons:
        if item["phase"] != "evaluate_1":
            continue
        token = int(item["module_token"].split("-")[-1])
        if token in {52, 6, 12, 26, 3, 53}:
            by_module[token] = item
    require(set(by_module) == {52, 6, 12, 26, 3, 53},
            "middle launch allocation records missing")

    def allocations(module: int) -> dict[int, str]:
        return {
            item["parameter_offset"]: item["allocation_token"]
            for item in by_module[module]["allocation_pointer_substitutions_a"]
        }

    expected_allocations = {
        52: {24: "allocation-000036"},
        6: {0: "allocation-000036"},
        12: {40: "allocation-000020", 48: "allocation-000021"},
        26: {32: "allocation-000020", 40: "allocation-000021"},
        3: {24: "allocation-000020", 32: "allocation-000021",
            72: "allocation-000038", 80: "allocation-000039",
            88: "allocation-000041"},
        53: {8: "allocation-000038", 16: "allocation-000039",
             24: "allocation-000041"},
    }
    for module, expected in expected_allocations.items():
        actual = allocations(module)
        for offset, token in expected.items():
            require(actual.get(offset) == token,
                    f"module {module} allocation binding drift at {offset}")

    descriptor_bindings: dict[tuple[int, int], dict[str, Any]] = {}
    for request in descriptors["resource_object_requests"]:
        for binding in request.get("packed_parameter_bindings", []):
            if binding["phase"] != "evaluate_1":
                continue
            module = int(binding["module_token"].split("-")[-1])
            descriptor_bindings[(module, binding["parameter_offset"])] = {
                "descriptor_trace_launch_alias":
                    int(binding["launch_token"].split("-")[-1]),
                "d3d_resource_token": request["d3d_resource_token"],
                "nvapi_resource_token": request["nvapi_resource_token"],
                "runtime_allocation_token": request.get(
                    "runtime_allocation_token"),
                "request_type": request["request_type"],
                "format": request["resource"]["format_name"],
            }

    descriptor_expectations = {
        (6, 48): "d3d-resource-000030",
        (6, 56): "d3d-resource-000028",
        (26, 16): "d3d-resource-000030",
        (26, 24): "d3d-resource-000028",
        (26, 48): "d3d-resource-000010",
        (3, 40): "d3d-resource-000010",
        (3, 48): "d3d-resource-000007",
        (3, 56): "d3d-resource-000008",
        (17, 16): "d3d-resource-000007",
        (17, 24): "d3d-resource-000008",
    }
    for key, resource in descriptor_expectations.items():
        require(descriptor_bindings.get(key, {}).get("d3d_resource_token") ==
                resource, f"descriptor edge drift: {key}")

    edges = [
        {"producer": 51, "consumer": 52, "kind": "buffer",
         "identity": "allocation-000036", "producer_parameter_offset": 24,
         "consumer_parameter_offset": 0, "bytes": 491520},
        {"producer": 52, "consumer": 54, "kind": "texture-rg16f-a",
         "identity": "d3d-resource-000030", "producer_parameter_offset": 48,
         "consumer_parameter_offset": 16, "bytes": 230400},
        {"producer": 52, "consumer": 54, "kind": "texture-rg16f-b",
         "identity": "d3d-resource-000028", "producer_parameter_offset": 56,
         "consumer_parameter_offset": 24, "bytes": 230400},
        {"producer": 53, "consumer": 54, "kind": "candidate-buffer-a",
         "identity": "allocation-000020", "producer_parameter_offset": 40,
         "consumer_parameter_offset": 32, "bytes": 460800},
        {"producer": 53, "consumer": 54, "kind": "candidate-buffer-b",
         "identity": "allocation-000021", "producer_parameter_offset": 48,
         "consumer_parameter_offset": 40, "bytes": 460800},
        {"producer": 53, "consumer": 55, "kind": "candidate-buffer-a",
         "identity": "allocation-000020", "producer_parameter_offset": 40,
         "consumer_parameter_offset": 24, "bytes": 460800},
        {"producer": 53, "consumer": 55, "kind": "candidate-buffer-b",
         "identity": "allocation-000021", "producer_parameter_offset": 48,
         "consumer_parameter_offset": 32, "bytes": 460800},
        {"producer": 54, "consumer": 55, "kind": "texture-rg16f",
         "identity": "d3d-resource-000010", "producer_parameter_offset": 48,
         "consumer_parameter_offset": 40, "bytes": 230400},
        {"producer": 55, "consumer": 56, "kind": "linear-half4-a",
         "identity": "allocation-000038", "producer_parameter_offset": 72,
         "consumer_parameter_offset": 8, "bytes": 1884160},
        {"producer": 55, "consumer": 56, "kind": "linear-half4-b",
         "identity": "allocation-000039", "producer_parameter_offset": 80,
         "consumer_parameter_offset": 16, "bytes": 1884160},
        {"producer": 55, "consumer": 56, "kind": "linear-half-mask",
         "identity": "allocation-000041", "producer_parameter_offset": 88,
         "consumer_parameter_offset": 24, "bytes": 471040},
        {"producer": 55, "consumer": 84, "kind": "surface-rgba16f-a",
         "identity": "d3d-resource-000007", "producer_parameter_offset": 48,
         "consumer_parameter_offset": 16, "bytes": 1843200},
        {"producer": 55, "consumer": 84, "kind": "surface-rgba16f-b",
         "identity": "d3d-resource-000008", "producer_parameter_offset": 56,
         "consumer_parameter_offset": 24, "bytes": 1843200},
    ]
    result = {
        "schema": "neuralstudio.fg.connected-l20-l87-edge-map.v1",
        "source_roots": {
            "selected_module_census": census["census_root_sha256"],
            "allocation_provenance": allocation["receipt_root_sha256"],
            "descriptor_provenance": descriptors["receipt_root_sha256"],
        },
        "numbering_reconciliation": {
            "canonical_selected_launch_count": 85,
            "descriptor_trace_launch_count": 83,
            "allocation_launch_ordinal_is_zero_based": True,
            "selected_launch_minus_allocation_ordinal": 3,
            "selected_launch_minus_descriptor_alias": 2,
            "examples": [
                {"selected_launch": 51, "allocation_ordinal": 48,
                 "module_token": 52},
                {"selected_launch": 55, "allocation_ordinal": 52,
                 "descriptor_trace_alias": 53, "module_token": 3},
                {"selected_launch": 84, "descriptor_trace_alias": 82,
                 "module_token": 17},
            ],
        },
        "edges": edges,
        "claims": {
            "l51_l56_dependency_closed": True,
            "l55_l84_surface_edge_proven": True,
            "host_intermediate_required": False,
        },
        "addresses_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
