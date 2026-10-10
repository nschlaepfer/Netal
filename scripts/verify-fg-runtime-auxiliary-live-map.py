#!/usr/bin/env python3
"""Build and verify the live L52/L53/L55 runtime resource alias map."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PINS = {
    "l52": ("evidence/specs/fg-mid-l52-v1/abi-receipt.json", "b4dfa4d7f79c81d468ea3a7ca4cf6973c2ef3b7b9e7efad68f0efed6bc8afc64"),
    "l53": ("evidence/specs/fg-mid-l53-v1/abi-receipt.json", "67cfdb6534b01b2bb61291eda3d8c970fec4b900c1a24e7460e6c54eba48d9b9"),
    "l55": ("evidence/specs/fg-mid-l55-v1/abi-receipt.json", "35b6dc9e095cda1e2fbbd68e96405ccffff78a0d645852df2a62c9795670cc78"),
    "descriptors": ("evidence/specs/fg-resource-descriptor-provenance-v1/real-resource-descriptor-provenance.json", "0a1361eba69a21234bbbec31a88a1d646a7653e0d70dcb00d8c2323856d66fd0"),
    "edges": ("evidence/specs/fg-connected-l20-l87-v1/edge-map.json", "f5ac4cd9216f9d3e63df952fcbf6d33be9f4148662be83b32b3d2c1948c7fe79"),
}


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode())


def load_pinned(root: Path, relative: str, expected: str) -> dict[str, Any]:
    path = root / relative
    require(path.is_file() and not path.is_symlink(), f"missing input: {relative}")
    data = path.read_bytes()
    require(sha_bytes(data) == expected, f"input hash drift: {relative}")
    value = json.loads(data)
    require(isinstance(value, dict), f"input receipt failed: {relative}")
    return value


def has_binding(request: dict[str, Any], launch: str, offset: int,
                request_type: str, phase: str) -> bool:
    return request.get("request_type") == request_type and any(
        item.get("launch_token") == launch and
        item.get("parameter_offset") == offset and
        item.get("phase") == phase
        for item in request.get("packed_parameter_bindings", []))


def validate_result(value: dict[str, Any], expected_rows: list[dict[str, Any]]) -> None:
    require(value.get("rows") == expected_rows, "resource alias map drift")
    require(value.get("row_count") == 9, "row count drift")
    require(value.get("descriptor_alias_to_canonical_delta") == 2,
            "launch numbering drift")
    require(value.get("l53_reads_working_vector_before_l54_overwrite") is True,
            "working-vector ordering drift")
    require(value.get("l55_history_order") == ["previous_history", "next_history"],
            "history bank order drift")
    require(value.get("auxiliary_mask_requires_shader_read") is True,
            "L52 usage drift")
    claims = value.get("claims", {})
    require(claims.get("descriptor_identity_proven") is True,
            "descriptor proof missing")
    require(claims.get("additional_game_input_required") is False,
            "caller API overclaim")
    require(claims.get("live_runtime_binding_executed") is False,
            "execution overclaim")
    require(claims.get("netal_integration") is False, "Netal overclaim")
    require(value.get("expected_or_archived_tensor_ingress") is False,
            "expected ingress")
    require(value.get("address_fields_recorded") is False, "address field")
    require(value.get("result") == "pass", "result failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    loaded = {name: load_pinned(repository, *pin) for name, pin in PINS.items()}
    for name in ("l52", "l53", "l55", "edges"):
        require(loaded[name].get("result") == "pass",
                f"input receipt failed: {name}")
    descriptor_claims = loaded["descriptors"].get("claims", {})
    require(loaded["descriptors"].get("schema") ==
            "neuralstudio.fg.resource-descriptor-provenance.v1" and
            descriptor_claims.get("live_d3d12_resource_descriptors_captured") is True and
            descriptor_claims.get("live_nvapi_descriptor_relationships_captured") is True and
            descriptor_claims.get("packed_parameter_binding_repeat_exact") is True,
            "descriptor provenance receipt failed")

    l52_depth = next(item for item in loaded["l52"]["bindings"]
                     if item.get("role") == "depth_texture_sampler")
    require(l52_depth.get("d3d_resource") == "d3d-resource-000035" and
            l52_depth.get("format") == "DXGI_FORMAT_R8_UNORM",
            "L52 live mask identity drift")
    l53_roles = {item.get("role"): item for item in loaded["l53"]["bindings"]}
    expected_l53 = {
        "curr_to_prev_flow_texture": ("d3d-resource-000009", "DXGI_FORMAT_R32G32_FLOAT"),
        "prev_to_curr_flow_texture": ("d3d-resource-000010", "DXGI_FORMAT_R16G16_FLOAT"),
        "curr_depth_texture": ("d3d-resource-000014", "DXGI_FORMAT_R32_UINT"),
        "prev_depth_texture": ("d3d-resource-000015", "DXGI_FORMAT_R32_UINT"),
    }
    for role, (resource, pixel_format) in expected_l53.items():
        require(l53_roles[role].get("d3d_resource") == resource and
                l53_roles[role].get("format") == pixel_format,
                f"L53 binding drift: {role}")
    l55_meanings = {item["offset"]: item["meaning"]
                    for item in loaded["l55"]["bindings"]}
    require(l55_meanings[0].startswith("resource-000030 RGBA16F source A") and
            l55_meanings[8].startswith("resource-000006 RGBA16F source B") and
            l55_meanings[16].startswith("resource-000050 RG16F motion") and
            l55_meanings[40].startswith("resource-000051 RG16F blend"),
            "L55 resource token order drift")

    requests = {item["nvapi_resource_token"]: item
                for item in loaded["descriptors"]["resource_object_requests"]}
    checks = [
        ("resource-000044", "d3d-resource-000035", "launch-000017", 88, "independent_surface"),
        ("resource-000046", "d3d-resource-000035", "launch-000050", 16, "merged_texture_sampler"),
        ("resource-000011", "d3d-resource-000009", "launch-000012", 32, "independent_surface"),
        ("resource-000015", "d3d-resource-000009", "launch-000051", 0, "merged_texture_sampler"),
        ("resource-000020", "d3d-resource-000010", "launch-000015", 56, "independent_surface"),
        ("resource-000023", "d3d-resource-000010", "launch-000051", 8, "merged_texture_sampler"),
        ("resource-000020", "d3d-resource-000010", "launch-000052", 48, "independent_surface"),
        ("resource-000051", "d3d-resource-000010", "launch-000053", 40, "merged_texture_sampler"),
        ("resource-000003", "d3d-resource-000014", "launch-000012", 88, "independent_surface"),
        ("resource-000018", "d3d-resource-000014", "launch-000051", 16, "merged_texture_sampler"),
        ("resource-000013", "d3d-resource-000015", "launch-000004", 88, "independent_surface", "evaluate_0"),
        ("resource-000016", "d3d-resource-000015", "launch-000051", 24, "merged_texture_sampler"),
        ("resource-000030", "d3d-resource-000006", "launch-000011", 80, "merged_texture_sampler"),
        ("resource-000030", "d3d-resource-000006", "launch-000053", 0, "merged_texture_sampler"),
        ("resource-000004", "d3d-resource-000005", "launch-000011", 120, "independent_surface"),
        ("resource-000006", "d3d-resource-000005", "launch-000053", 8, "merged_texture_sampler"),
        ("resource-000045", "d3d-resource-000029", "launch-000017", 96, "independent_surface"),
        ("resource-000050", "d3d-resource-000029", "launch-000053", 16, "merged_texture_sampler"),
    ]
    normalized_checks = [item if len(item) == 6 else (*item, "evaluate_1")
                         for item in checks]
    for token, d3d, launch, offset, request_type, phase in normalized_checks:
        request = requests[token]
        require(request.get("d3d_resource_token") == d3d and
                has_binding(request, launch, offset, request_type, phase),
                f"descriptor binding drift: {token}/{launch}/{offset}")
    require(loaded["edges"].get("numbering_reconciliation", {}).get(
        "selected_launch_minus_descriptor_alias") == 2,
        "descriptor numbering reconciliation drift")

    rows = [
        {"consumer": "L52", "role": "depth_mask", "d3d_resource": "d3d-resource-000035", "consumer_token": "resource-000046", "live_source": "L19 auxiliary mask", "runtime_binding": "auxiliary_mask_"},
        {"consumer": "L53", "role": "current_to_previous_flow", "d3d_resource": "d3d-resource-000009", "consumer_token": "resource-000015", "live_source": "L14 flow output", "runtime_binding": "flow_"},
        {"consumer": "L53", "role": "previous_to_current_flow", "d3d_resource": "d3d-resource-000010", "consumer_token": "resource-000023", "live_source": "L16/L17 working vector before L54", "runtime_binding": "working_vector_"},
        {"consumer": "L53", "role": "current_depth", "d3d_resource": "d3d-resource-000014", "consumer_token": "resource-000018", "live_source": "current depth bank", "runtime_binding": "current_depth"},
        {"consumer": "L53", "role": "previous_depth", "d3d_resource": "d3d-resource-000015", "consumer_token": "resource-000016", "live_source": "previous depth bank", "runtime_binding": "previous_depth"},
        {"consumer": "L55", "role": "source_a", "d3d_resource": "d3d-resource-000006", "consumer_token": "resource-000030", "live_source": "generated L13 input history", "runtime_binding": "previous_history"},
        {"consumer": "L55", "role": "source_b", "d3d_resource": "d3d-resource-000005", "consumer_token": "resource-000006", "live_source": "generated L13 output history", "runtime_binding": "next_history"},
        {"consumer": "L55", "role": "projection_motion", "d3d_resource": "d3d-resource-000029", "consumer_token": "resource-000050", "live_source": "L19 current projection", "runtime_binding": "current_projection"},
        {"consumer": "L55", "role": "blend", "d3d_resource": "d3d-resource-000010", "consumer_token": "resource-000051", "live_source": "L54 overwrite after L53 read", "runtime_binding": "working_vector_"},
    ]
    result = {
        "schema": "neuralstudio.fg.runtime-auxiliary-live-map.v1",
        "source_sha256": {name: pin[1] for name, pin in PINS.items()},
        "descriptor_alias_to_canonical_delta": 2,
        "row_count": len(rows),
        "rows": rows,
        "l53_reads_working_vector_before_l54_overwrite": True,
        "l55_history_order": ["previous_history", "next_history"],
        "auxiliary_mask_requires_shader_read": True,
        "claims": {
            "descriptor_identity_proven": True,
            "additional_game_input_required": False,
            "live_runtime_binding_executed": False,
            "netal_integration": False,
        },
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    validate_result(result, rows)
    result["receipt_root_sha256"] = canonical(result)

    cases = []
    mutations = {
        "history-order": lambda value: value.__setitem__("l55_history_order", ["next_history", "previous_history"]),
        "working-vector-order": lambda value: value.__setitem__("l53_reads_working_vector_before_l54_overwrite", False),
        "mask-usage": lambda value: value.__setitem__("auxiliary_mask_requires_shader_read", False),
        "resource-alias": lambda value: value["rows"][0].__setitem__("d3d_resource", "d3d-resource-000034"),
        "execution-overclaim": lambda value: value["claims"].__setitem__("live_runtime_binding_executed", True),
        "netal-overclaim": lambda value: value["claims"].__setitem__("netal_integration", True),
        "expected-ingress": lambda value: value.__setitem__("expected_or_archived_tensor_ingress", True),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(result)
        mutate(changed)
        rejected = False
        try:
            validate_result(changed, rows)
        except VerificationError:
            rejected = True
        require(rejected, f"mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})
    audit = {
        "schema": "neuralstudio.fg.runtime-auxiliary-live-map-negative.v1",
        "case_count": len(cases),
        "rejected_count": len(cases),
        "cases": cases,
        "result": "pass",
    }
    audit["receipt_root_sha256"] = canonical(audit)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    require(not arguments.output.exists() and not arguments.negative_output.exists(),
            "preserve existing output")
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.negative_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"negative_root_sha256={audit['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
