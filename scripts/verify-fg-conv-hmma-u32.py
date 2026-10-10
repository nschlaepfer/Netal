#!/usr/bin/env python3
"""Verify the exact uint32 HMMA optimization evidence package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


BACKEND_FILE_SHA256 = (
    "fe7982df459b2be02caf7e0efe0f86237bebcae27b8030ce82ba8f45830a44ae"
)
BACKEND_ROOT_SHA256 = (
    "1178aa27b4200fbae1009860827981a840653ec0ec2a4c98ec89ea8782d9c0d0"
)
PARENT_FILE_SHA256 = (
    "d2b636e25e2a1492616c43bc9e10f7e7098d79744477082ea09f078f44968839"
)
PARENT_ROOT_SHA256 = (
    "972a6888b346a87423949b6cfa33c6dc91e3f93c303ebfc9e420a355c14550b3"
)
EXPECTED_OUTPUTS = {
    "l20": "12787ff635c08e8b1e35aaa887d30f3122dd3e9a5ef97743089f598cf3c9b0fc",
    "l24": "fc476f13ecfe443a28635b875bb50bd60f2447a4e77e3044bc7b7b7ba9811153",
    "l26": "730dbc79332d922707239cc7d855aa81000220a74baf0de529c44066ec049656",
    "l28": "67327484347a5b6119565282d607fa51202673b8fe9921f2d0d0cf0a27e3fc61",
}
SCHEDULES = {"l20": False, "l24": False, "l26": False, "l28": True}
SUPERSEDED_ROOT_SHA256 = (
    "1c16d641147d37443ae42e35aa546610f28213cc35f2f0aa2a5bb80a7049cbfc"
)


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


def validate_backend(value: dict[str, Any]) -> None:
    without_root = copy.deepcopy(value)
    root = without_root.pop("receipt_root_sha256", None)
    require(root == BACKEND_ROOT_SHA256, "backend root drift")
    require(canonical(without_root) == root, "backend canonical root mismatch")
    require(value.get("schema") == "neuralstudio.fg.conv-hmma-u32-mac.v2",
            "backend schema drift")
    require(value.get("accepted_parent_root_sha256") == PARENT_ROOT_SHA256,
            "parent root drift")
    require(value.get("supersedes_receipt_root_sha256") ==
            SUPERSEDED_ROOT_SHA256, "superseded root drift")
    require(value.get("fresh_process_count") == 16, "process count drift")
    require(value.get("two_poison") is True, "two-poison proof absent")
    require(value.get("total_compared_bytes") == 5_283_840,
            "compared-byte count drift")
    require(value.get("total_mismatched_bytes") == 0,
            "candidate mismatch")
    bound = value.get("accumulator_bound", {})
    require(bound.get("maximum_unsigned_total") == 2_212_463_104 and
            bound.get("uint32_limit_exclusive") == 2**32 and
            bound.get("overflow_proven_absent") is True,
            "uint32 bound absent")
    require(16 * bound.get("maximum_aligned_product", 0) +
            bound.get("maximum_aligned_prior", 0) ==
            bound.get("maximum_unsigned_total"), "bound arithmetic drift")
    cpu = value.get("cpu_equivalence", {})
    require(cpu.get("finite_random_records") == 4_194_304 and
            cpu.get("mismatches") == 0 and
            cpu.get("proven_maximum_unsigned_total") == 2_212_463_104,
            "CPU equivalence drift")
    family = value.get("exact_u32_family", {})
    require(set(family) == set(EXPECTED_OUTPUTS), "layer coverage drift")
    for layer, output_sha256 in EXPECTED_OUTPUTS.items():
        row = family[layer]
        require(row.get("output_sha256") == output_sha256 and
                row.get("column_stage_schedule") == SCHEDULES[layer] and
                row.get("fresh_process_count") == 4 and
                row.get("poison_bytes") == [90, 165] and
                row.get("two_poison_write_coverage_conclusive") is True and
                row.get("mismatched_bytes") == 0 and
                row.get("candidate_gpu_speedup", 0.0) > 1.0,
                f"layer rejected: {layer}")
    require(value.get("expected_output_bound_to_gpu") is False,
            "expected output ingress")
    require(value.get("archived_intermediate_bound_to_gpu") is False,
            "archived intermediate ingress")
    require(value.get("address_fields_recorded") is False, "address field")
    claims = value.get("claims", {})
    require(claims.get(
        "separate_u32_accumulation_exact_for_finite_hmma_inputs") is True,
        "primitive exactness claim absent")
    require(claims.get("exact_u32_3x3_family") is True,
            "family exactness claim absent")
    require(claims.get("schedule_specific_timing_validated") is True,
            "schedule timing claim absent")
    require(claims.get("full_runtime_promoted") is False,
            "runtime overclaim")
    require(claims.get("netal_integration") is False,
            "Netal overclaim")
    require(value.get("result") == "pass", "backend rejected")


def validate_summary(value: dict[str, Any]) -> None:
    require(value.get("backend_root_sha256") == BACKEND_ROOT_SHA256,
            "summary backend drift")
    require(value.get("parent_root_sha256") == PARENT_ROOT_SHA256,
            "summary parent drift")
    require(value.get("supersedes_receipt_root_sha256") ==
            SUPERSEDED_ROOT_SHA256, "summary superseded root drift")
    require(value.get("layer_count") == 4, "summary layer count")
    require(value.get("fresh_process_count") == 16,
            "summary process count")
    require(value.get("poison_bytes") == [90, 165], "summary poison")
    require(value.get("total_mismatched_bytes") == 0,
            "summary mismatch")
    require(value.get("cpu_equivalence_mismatches") == 0,
            "summary CPU mismatch")
    require(value.get("proven_maximum_unsigned_total") == 2_212_463_104,
            "summary bound")
    require(value.get("column_stage_schedule") == SCHEDULES,
            "summary schedule map")
    require(value.get("all_gpu_speedups_above_one") is True,
            "summary timing regression")
    require(value.get("expected_output_bound_to_gpu") is False,
            "summary expected ingress")
    require(value.get("archived_intermediate_bound_to_gpu") is False,
            "summary archive ingress")
    claims = value.get("claims", {})
    require(claims.get("exact_u32_hmma") is True, "summary HMMA claim")
    require(claims.get("exact_u32_3x3_family") is True,
            "summary family claim")
    require(claims.get("schedule_specific_timing_validated") is True,
            "summary schedule claim")
    require(claims.get("full_runtime_promoted") is False,
            "summary runtime overclaim")
    require(claims.get("netal_integration") is False,
            "summary Netal overclaim")
    require(value.get("result") == "pass", "summary rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--backend", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--negative-output", required=True, type=Path)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    backend = load(arguments.backend.resolve(), BACKEND_FILE_SHA256)
    parent = load(arguments.parent.resolve(), PARENT_FILE_SHA256)
    validate_backend(backend)
    require(parent.get("receipt_root_sha256") == PARENT_ROOT_SHA256 and
            parent.get("every_layer_rtx_mac_exact") is True,
            "parent parity rejected")
    require({row["layer_id"]: row["output_sha256"]
             for row in parent["layers"]} == EXPECTED_OUTPUTS,
            "parent output identity drift")
    for relative, expected in backend["source_sha256"].items():
        path = repository / relative
        require(path.is_file() and sha_bytes(path.read_bytes()) == expected,
                f"source drift: {relative}")

    family = backend["exact_u32_family"]
    result = {
        "schema": "neuralstudio.fg.conv-hmma-u32-verification.v2",
        "backend_root_sha256": BACKEND_ROOT_SHA256,
        "parent_root_sha256": PARENT_ROOT_SHA256,
        "supersedes_receipt_root_sha256": SUPERSEDED_ROOT_SHA256,
        "layer_count": 4,
        "fresh_process_count": 16,
        "poison_bytes": [90, 165],
        "total_compared_bytes": 5_283_840,
        "total_mismatched_bytes": 0,
        "cpu_equivalence_records": 4_194_304,
        "cpu_equivalence_mismatches": 0,
        "proven_maximum_unsigned_total": 2_212_463_104,
        "uint32_limit_exclusive": 2**32,
        "output_sha256": EXPECTED_OUTPUTS,
        "column_stage_schedule": SCHEDULES,
        "gpu_speedup_by_layer": {
            layer: family[layer]["candidate_gpu_speedup"]
            for layer in EXPECTED_OUTPUTS
        },
        "all_gpu_speedups_above_one": True,
        "expected_output_bound_to_gpu": False,
        "archived_intermediate_bound_to_gpu": False,
        "address_fields_recorded": False,
        "claims": {
            "exact_u32_hmma": True,
            "exact_u32_3x3_family": True,
            "schedule_specific_timing_validated": True,
            "full_runtime_promoted": False,
            "netal_integration": False,
        },
        "result": "pass",
    }
    validate_summary(result)
    result["receipt_root_sha256"] = canonical(result)

    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "backend-root": lambda value: value.__setitem__(
            "backend_root_sha256", "0" * 64),
        "parent-root": lambda value: value.__setitem__(
            "parent_root_sha256", "0" * 64),
        "superseded-root": lambda value: value.__setitem__(
            "supersedes_receipt_root_sha256", "0" * 64),
        "process-count": lambda value: value.__setitem__(
            "fresh_process_count", 15),
        "poison": lambda value: value.__setitem__("poison_bytes", [165]),
        "output": lambda value: value.__setitem__(
            "total_mismatched_bytes", 1),
        "cpu-equivalence": lambda value: value.__setitem__(
            "cpu_equivalence_mismatches", 1),
        "bound": lambda value: value.__setitem__(
            "proven_maximum_unsigned_total", 2**32),
        "timing": lambda value: value.__setitem__(
            "all_gpu_speedups_above_one", False),
        "schedule": lambda value: value["claims"].__setitem__(
            "schedule_specific_timing_validated", False),
        "expected-ingress": lambda value: value.__setitem__(
            "expected_output_bound_to_gpu", True),
        "archive-ingress": lambda value: value.__setitem__(
            "archived_intermediate_bound_to_gpu", True),
        "runtime-overclaim": lambda value: value["claims"].__setitem__(
            "full_runtime_promoted", True),
        "netal-overclaim": lambda value: value["claims"].__setitem__(
            "netal_integration", True),
    }
    cases = []
    for name, mutate in mutations.items():
        changed = copy.deepcopy(result)
        mutate(changed)
        rejected = False
        try:
            validate_summary(changed)
        except VerificationError:
            rejected = True
        require(rejected, f"mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})
    negative = {
        "schema": "neuralstudio.fg.conv-hmma-u32-negative.v2",
        "case_count": len(cases),
        "rejected_count": len(cases),
        "cases": cases,
        "result": "pass",
    }
    negative["receipt_root_sha256"] = canonical(negative)

    require(not arguments.output.exists() and
            not arguments.negative_output.exists(), "preserve existing output")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.negative_output.write_text(
        json.dumps(negative, indent=2, sort_keys=True) + "\n"
    )
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"negative_root_sha256={negative['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
