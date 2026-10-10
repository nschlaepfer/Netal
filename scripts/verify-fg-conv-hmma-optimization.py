#!/usr/bin/env python3
"""Verify the exact CLZ HMMA optimization and retained fast failures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


BACKEND_SHA256 = "b8a6abe37769259b939de30845a51980dc5539ea9561d2de6d0f296952ae011f"
BACKEND_ROOT = "8065f8c1ba10b39f531c29e50222135f76616145f250d265ca0b76aca148e867"
PARENT_SHA256 = "d2b636e25e2a1492616c43bc9e10f7e7098d79744477082ea09f078f44968839"
PARENT_ROOT = "972a6888b346a87423949b6cfa33c6dc91e3f93c303ebfc9e420a355c14550b3"
EXPECTED_OUTPUTS = {
    "l20": "12787ff635c08e8b1e35aaa887d30f3122dd3e9a5ef97743089f598cf3c9b0fc",
    "l24": "fc476f13ecfe443a28635b875bb50bd60f2447a4e77e3044bc7b7b7ba9811153",
    "l26": "730dbc79332d922707239cc7d855aa81000220a74baf0de529c44066ec049656",
    "l28": "67327484347a5b6119565282d607fa51202673b8fe9921f2d0d0cf0a27e3fc61",
}
EXPECTED_COUNTEREXAMPLES = {"simd": 117, "dual": 100, "multi": 88}


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


def load(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"input missing: {path}")
    data = path.read_bytes()
    require(sha_bytes(data) == expected_sha256, f"input hash drift: {path}")
    value = json.loads(data)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def validate(value: dict[str, Any]) -> None:
    require(value.get("backend_root_sha256") == BACKEND_ROOT,
            "backend root drift")
    require(value.get("parent_root_sha256") == PARENT_ROOT,
            "parent root drift")
    require(value.get("layer_count") == 4, "layer count drift")
    require(value.get("fresh_process_count") == 16,
            "fresh process count drift")
    require(value.get("poison_bytes") == [90, 165], "poison coverage drift")
    require(value.get("total_mismatched_bytes") == 0,
            "CLZ output mismatch")
    require(value.get("all_candidate_gpu_speedups_above_one") is True,
            "candidate timing regression")
    require(value.get("counterexample_mismatched_bytes") ==
            EXPECTED_COUNTEREXAMPLES, "counterexample drift")
    claims = value.get("claims", {})
    require(claims.get("exact_clz_3x3_family") is True,
            "CLZ exactness claim missing")
    require(claims.get("raw_simd_exact") is False,
            "SIMD exactness overclaim")
    require(claims.get("full_runtime_promoted") is False,
            "runtime overclaim")
    require(claims.get("netal_integration") is False, "Netal overclaim")
    require(value.get("expected_output_bound_to_gpu") is False,
            "expected output ingress")
    require(value.get("address_fields_recorded") is False, "address field")
    require(value.get("result") == "pass", "verification rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    backend = load(arguments.backend.resolve(), BACKEND_SHA256)
    parent = load(arguments.parent.resolve(), PARENT_SHA256)
    require(backend.get("receipt_root_sha256") == BACKEND_ROOT and
            backend.get("result") == "pass", "backend failed")
    require(parent.get("receipt_root_sha256") == PARENT_ROOT and
            parent.get("every_layer_rtx_mac_exact") is True,
            "parent exactness failed")
    require({item["layer_id"]: item["output_sha256"]
             for item in parent["layers"]} == EXPECTED_OUTPUTS,
            "parent output identity drift")
    for relative, expected in backend.get("source_sha256", {}).items():
        path = repository / relative
        require(path.is_file() and sha_bytes(path.read_bytes()) == expected,
                f"source drift: {relative}")
    family = backend.get("exact_clz_family", {})
    require(set(family) == set(EXPECTED_OUTPUTS), "CLZ family coverage drift")
    for layer, expected in EXPECTED_OUTPUTS.items():
        row = family[layer]
        require(row.get("output_sha256") == expected and
                row.get("fresh_process_count") == 4 and
                row.get("poison_bytes") == [90, 165] and
                row.get("mismatched_bytes") == 0 and
                row.get("candidate_gpu_speedup", 0) > 1.0,
                f"CLZ layer rejected: {layer}")
    counterexamples = backend.get("counterexamples", {})
    require(set(counterexamples) == set(EXPECTED_COUNTEREXAMPLES),
            "counterexample coverage drift")
    for name, mismatch in EXPECTED_COUNTEREXAMPLES.items():
        row = counterexamples[name]
        require(row.get("mismatched_bytes") == mismatch and
                row.get("result") == "rejected" and
                row.get("candidate_gpu_speedup", 0) > 1.0,
                f"counterexample not preserved: {name}")

    result = {
        "schema": "neuralstudio.fg.conv-hmma-clz-verification.v1",
        "backend_root_sha256": BACKEND_ROOT,
        "parent_root_sha256": PARENT_ROOT,
        "layer_count": 4,
        "fresh_process_count": 16,
        "poison_bytes": [90, 165],
        "total_compared_bytes": sum(row["compared_bytes"] for row in family.values()),
        "total_mismatched_bytes": 0,
        "output_sha256": EXPECTED_OUTPUTS,
        "gpu_speedup_by_layer": {
            layer: family[layer]["candidate_gpu_speedup"]
            for layer in EXPECTED_OUTPUTS
        },
        "all_candidate_gpu_speedups_above_one": True,
        "counterexample_mismatched_bytes": EXPECTED_COUNTEREXAMPLES,
        "expected_output_bound_to_gpu": False,
        "address_fields_recorded": False,
        "claims": {
            "exact_clz_3x3_family": True,
            "raw_simd_exact": False,
            "full_runtime_promoted": False,
            "netal_integration": False,
        },
        "result": "pass",
    }
    validate(result)
    result["receipt_root_sha256"] = canonical(result)

    cases = []
    mutations = {
        "backend-root": lambda value: value.__setitem__("backend_root_sha256", "0" * 64),
        "process-count": lambda value: value.__setitem__("fresh_process_count", 15),
        "poison": lambda value: value.__setitem__("poison_bytes", [165]),
        "clz-output": lambda value: value.__setitem__("total_mismatched_bytes", 1),
        "timing": lambda value: value.__setitem__("all_candidate_gpu_speedups_above_one", False),
        "counterexample": lambda value: value["counterexample_mismatched_bytes"].__setitem__("simd", 0),
        "simd-overclaim": lambda value: value["claims"].__setitem__("raw_simd_exact", True),
        "runtime-overclaim": lambda value: value["claims"].__setitem__("full_runtime_promoted", True),
        "netal-overclaim": lambda value: value["claims"].__setitem__("netal_integration", True),
        "expected-ingress": lambda value: value.__setitem__("expected_output_bound_to_gpu", True),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(result)
        mutate(changed)
        rejected = False
        try:
            validate(changed)
        except VerificationError:
            rejected = True
        require(rejected, f"mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})
    audit = {
        "schema": "neuralstudio.fg.conv-hmma-clz-negative.v1",
        "case_count": len(cases),
        "rejected_count": len(cases),
        "cases": cases,
        "result": "pass",
    }
    audit["receipt_root_sha256"] = canonical(audit)
    require(not arguments.output.exists() and not arguments.negative_output.exists(),
            "preserve existing output")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
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
