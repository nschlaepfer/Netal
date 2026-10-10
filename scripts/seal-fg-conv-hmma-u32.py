#!/usr/bin/env python3
"""Seal one exact uint32 HMMA 3x3-family experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
SOURCE_PATHS = (
    "src/fg_conv/hmma_exact.metalh",
    "src/fg_conv/conv_3x3.metal",
    "src/fg_conv/macos_3x3_probe.mm",
    "src/fg_conv_simd_l20/analyze_truncation.cc",
    "src/fg_conv_simd_l20/test_u32_equivalence.cc",
    "src/fg_conv_simd_l20/hmma_exact_u32.metalh",
    "src/fg_conv_simd_l20/hmma_exact_u32_override.metalh",
    "src/fg_conv_simd_l20/benchmark.mm",
    "scripts/build-macos-fg-conv-hmma-u32.sh",
    "scripts/seal-fg-conv-hmma-u32.py",
)


class SealError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing file: {path}")
    return sha_bytes(path.read_bytes())


def canonical(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def parse_equivalence(path: Path) -> dict[str, int]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    expected = {
        "finite_random_records": 4_194_304,
        "mismatches": 0,
        "proven_maximum_unsigned_total": 2_212_463_104,
    }
    require(fields.get("result") == "pass", "CPU equivalence rejected")
    for key, value in expected.items():
        require(int(fields.get(key, "-1")) == value,
                f"CPU equivalence drift: {key}")
    return {
        **expected,
        "observed_maximum_positive": int(fields["observed_maximum_positive"]),
        "observed_maximum_negative": int(fields["observed_maximum_negative"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    attempt = arguments.attempt.resolve()
    output = arguments.output.resolve()
    require(attempt.is_dir() and not attempt.is_symlink(), "attempt missing")
    require(not output.exists(), "preserve existing output")

    parent_path = repository / (
        "evidence/specs/fg-conv-3x3-family-v1/family-verification.json"
    )
    require(sha_file(parent_path) == PARENT_FILE_SHA256, "parent file drift")
    parent = load_json(parent_path)
    require(parent.get("receipt_root_sha256") == PARENT_ROOT_SHA256,
            "parent root drift")
    require(parent.get("every_layer_rtx_mac_exact") is True,
            "parent RTX/Mac parity absent")
    require({row["layer_id"]: row["output_sha256"]
             for row in parent["layers"]} == EXPECTED_OUTPUTS,
            "parent output identity drift")

    cpu_equivalence = parse_equivalence(
        attempt / "u32-equivalence.stdout.log"
    )
    require(16 * 134_086_688 + 67_076_096 == 2_212_463_104,
            "accumulator bound arithmetic drift")
    require(cpu_equivalence["proven_maximum_unsigned_total"] < 2**32,
            "uint32 accumulator may overflow")

    process_ids: list[int] = []
    family: dict[str, Any] = {}
    for layer, expected_output in EXPECTED_OUTPUTS.items():
        paths = sorted((attempt / "runs").glob(
            f"{layer}-*/run-receipt.json"
        ))
        require(len(paths) == 4, f"fresh-process coverage: {layer}")
        runs = [load_json(path) for path in paths]
        require(len({run["process_id"] for run in runs}) == 4,
                f"PID reuse: {layer}")
        require({run["poison_byte"] for run in runs} == {90, 165},
                f"poison coverage: {layer}")
        require({run["output_sha256"] for run in runs} == {expected_output},
                f"RTX identity mismatch: {layer}")
        for run in runs:
            require(run.get("result") == "pass", f"run failed: {layer}")
            require(run.get("expected_output_file_opened") is False,
                    f"expected output ingress: {layer}")
            require(run.get("address_fields_recorded") is False,
                    f"address field: {layer}")
            require(run.get("input_pre_sha256") == run.get("input_post_sha256"),
                    f"input mutated: {layer}")
            require(run.get("weight_pre_sha256") == run.get("weight_post_sha256"),
                    f"weight mutated: {layer}")
            require(run.get("input_mismatched_bytes") == 0 and
                    run.get("canary_mismatched_bytes") == 0,
                    f"guard failure: {layer}")
        process_ids.extend(run["process_id"] for run in runs)

        benchmark = load_json(
            attempt / "runs" / f"{layer}-benchmark.json"
        )
        require(benchmark.get("correctness_result") == "pass" and
                benchmark.get("scalar_candidate_mismatched_bytes") == 0,
                f"benchmark mismatch: {layer}")
        require(benchmark.get("scalar_output_sha256") == expected_output and
                benchmark.get("candidate_output_sha256") == expected_output,
                f"benchmark schedule/output identity mismatch: {layer}")
        require(benchmark.get("immutable_input_mismatched_bytes") == 0 and
                benchmark.get("canary_mismatched_bytes") == 0,
                f"benchmark guard failure: {layer}")
        require(benchmark.get("candidate_gpu_speedup", 0.0) > 1.0,
                f"benchmark regression: {layer}")
        require(benchmark.get("expected_output_bound_to_gpu") is False and
                benchmark.get("address_fields_recorded") is False,
                f"benchmark provenance failure: {layer}")
        family[layer] = {
            "output_sha256": expected_output,
            "column_stage_schedule": SCHEDULES[layer],
            "fresh_process_count": 4,
            "poison_bytes": [90, 165],
            "two_poison_write_coverage_conclusive": True,
            "compared_bytes": runs[0]["output_bytes"] * 4,
            "mismatched_bytes": 0,
            "scalar_median_gpu_ms": benchmark["scalar_median_gpu_ms"],
            "candidate_median_gpu_ms": benchmark["candidate_median_gpu_ms"],
            "candidate_gpu_speedup": benchmark["candidate_gpu_speedup"],
        }
    require(len(set(process_ids)) == 16, "fresh process count drift")

    result = {
        "schema": "neuralstudio.fg.conv-hmma-u32-mac.v2",
        "device": "Apple M3 Max",
        "accepted_parent_root_sha256": PARENT_ROOT_SHA256,
        "supersedes_receipt_root_sha256": SUPERSEDED_ROOT_SHA256,
        "supersession_reason": (
            "v1 L28 timing used schedule 0; v2 binds every benchmark output "
            "to its accepted schedule-specific RTX identity"
        ),
        "cpu_equivalence": cpu_equivalence,
        "accumulator_bound": {
            "maximum_aligned_product": 134_086_688,
            "maximum_aligned_prior": 67_076_096,
            "term_count": 16,
            "maximum_unsigned_total": 2_212_463_104,
            "uint32_limit_exclusive": 2**32,
            "overflow_proven_absent": True,
        },
        "exact_u32_family": family,
        "fresh_process_count": 16,
        "two_poison": True,
        "total_compared_bytes": sum(
            row["compared_bytes"] for row in family.values()
        ),
        "total_mismatched_bytes": 0,
        "source_sha256": {
            relative: sha_file(repository / relative)
            for relative in SOURCE_PATHS
        },
        "expected_output_bound_to_gpu": False,
        "archived_intermediate_bound_to_gpu": False,
        "address_fields_recorded": False,
        "claims": {
            "separate_u32_accumulation_exact_for_finite_hmma_inputs": True,
            "exact_u32_3x3_family": True,
            "schedule_specific_timing_validated": True,
            "full_runtime_promoted": False,
            "netal_integration": False,
        },
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SealError as error:
        raise SystemExit(f"seal_error={error}")
