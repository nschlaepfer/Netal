#!/usr/bin/env python3
"""Verify the interleaved reference/compact FG runtime diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"read failed: {path}: {error}") from error


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def validate_mode(attempt: Path, mode: str, summary: dict[str, Any],
                  expected: list[str]) -> list[dict[str, Any]]:
    paths = sorted((attempt / mode).glob("run-*.json"))
    require(len(paths) == 3, f"{mode}: three runs")
    receipts = [load(path) for path in paths]
    require(len({receipt["process_id"] for receipt in receipts}) == 3,
            f"{mode}: fresh processes")
    require(summary.get("raw_receipt_sha256") == [sha(path) for path in paths],
            f"{mode}: raw receipt pins")
    require(summary.get("process_ids") == [r["process_id"] for r in receipts],
            f"{mode}: process provenance")
    all_times: list[float] = []
    per_run_medians: list[float] = []
    for receipt in receipts:
        require(receipt.get("schema") ==
                "neuralstudio.fg.runtime-benchmark-run.v1" and
                receipt.get("ex2_mode") == mode and
                receipt.get("timing_class") ==
                "synchronous-evaluate-wall-not-input-latency" and
                receipt.get("counter_attachment") is False and
                receipt.get("warmup_generated_frames") == 15 and
                receipt.get("measured_generated_frames") == 15 and
                receipt.get("postcheck_generated_frames") == 15 and
                receipt.get("output_hashes_comparable") is True and
                receipt.get("pre_output_sha256") == expected and
                receipt.get("measured_output_sha256") == expected and
                receipt.get("post_output_sha256") == expected and
                receipt.get("caller_owned_input_mismatches") == 0 and
                receipt.get("expected_or_archived_intermediate_ingress") is False and
                receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass", f"{mode}: run contract")
        times = receipt.get("measured_evaluate_wall_ms")
        allocations = receipt.get("measured_metal_resource_allocations")
        require(isinstance(times, list) and len(times) == 15 and
                all(isinstance(value, (int, float)) and value > 0 and
                    math.isfinite(value) for value in times),
                f"{mode}: frame timings")
        require(allocations == [0] * 15, f"{mode}: measured allocations")
        all_times.extend(times)
        per_run_medians.append(statistics.median(times))
    mean = statistics.mean(all_times)
    stddev = statistics.pstdev(all_times)
    expected_stats = {
        "per_run_median_evaluate_wall_ms": per_run_medians,
        "p50_evaluate_wall_ms": percentile(all_times, 0.50),
        "p95_evaluate_wall_ms": percentile(all_times, 0.95),
        "p99_evaluate_wall_ms": percentile(all_times, 0.99),
        "max_evaluate_wall_ms": max(all_times),
        "mean_evaluate_wall_ms": mean,
        "stddev_evaluate_wall_ms": stddev,
        "coefficient_of_variation": stddev / mean,
    }
    require(summary.get("evaluate_wall_ms") == all_times,
            f"{mode}: retained frame timings")
    for key, value in expected_stats.items():
        actual = summary.get(key)
        if isinstance(value, list):
            require(actual == value, f"{mode}: {key}")
        else:
            require(isinstance(actual, (int, float)) and close(actual, value),
                    f"{mode}: {key}")
    return receipts


def validate(attempt: Path, reference: Path,
             override: dict[str, Any] | None = None) -> dict[str, Any]:
    comparison = override or load(attempt / "comparison.json")
    expected = load(reference)["final_output_sha256"]
    require(comparison.get("schema") ==
            "neuralstudio.fg.runtime-benchmark-comparison.v1" and
            comparison.get("diagnostic_component_only") is True and
            comparison.get("production_benchmark") is False and
            comparison.get("timing_class") ==
            "synchronous-evaluate-wall-not-input-latency" and
            comparison.get("counter_attachment") is False and
            comparison.get("untimed_correctness_before_and_after") is True and
            comparison.get("output_hashes_comparable") is True and
            comparison.get("accepted_output_sha256") == expected and
            comparison.get("execution_order") == [
                "reference-1", "compact-1", "compact-2",
                "reference-2", "reference-3", "compact-3"] and
            comparison.get("result") == "pass", "comparison policy")
    reference_runs = validate_mode(attempt, "reference",
                                   comparison["reference"], expected)
    compact_runs = validate_mode(attempt, "compact",
                                 comparison["compact"], expected)
    require(len({r["process_id"] for r in reference_runs + compact_runs}) == 6,
            "six fresh benchmark processes")
    require(comparison.get("reference_ex2_bytes") == 100663304 and
            comparison.get("compact_ex2_bytes") == 6291458 and
            comparison.get("ex2_saved_bytes") == 94371846,
            "EX2 memory comparison")
    expected_speedup = (comparison["reference"]["p50_evaluate_wall_ms"] /
                        comparison["compact"]["p50_evaluate_wall_ms"])
    require(close(comparison.get("compact_vs_reference_p50_speedup"),
                  expected_speedup), "speedup arithmetic")
    for prefix in ("reference", "compact"):
        configuration = comparison[f"{prefix}_configuration"]
        require(canonical(configuration) ==
                comparison[f"{prefix}_configuration_root_sha256"],
                f"{prefix} configuration root")
    normalized = {key: value for key, value in comparison.items()
                  if key not in ("reference", "compact",
                                 "normalized_root_sha256")}
    require(canonical(normalized) == comparison.get("normalized_root_sha256"),
            "normalized comparison root")
    result = {
        "schema": "neuralstudio.fg.runtime-benchmark-verification.v1",
        "comparison_sha256": sha(attempt / "comparison.json"),
        "normalized_root_sha256": comparison["normalized_root_sha256"],
        "execution_order": comparison["execution_order"],
        "independent_runs_per_mode": 3,
        "measured_frames_per_run": 15,
        "reference_p50_evaluate_wall_ms":
            comparison["reference"]["p50_evaluate_wall_ms"],
        "compact_p50_evaluate_wall_ms":
            comparison["compact"]["p50_evaluate_wall_ms"],
        "compact_vs_reference_p50_speedup": expected_speedup,
        "reference_ex2_bytes": 100663304,
        "compact_ex2_bytes": 6291458,
        "ex2_saved_bytes": 94371846,
        "output_hashes_comparable": True,
        "production_benchmark": False,
        "timing_class": "synchronous-evaluate-wall-not-input-latency",
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def negative(attempt: Path, reference: Path) -> dict[str, Any]:
    original = load(attempt / "comparison.json")
    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "order": lambda value: value["execution_order"].reverse(),
        "production_claim": lambda value: value.__setitem__(
            "production_benchmark", True),
        "timing_label": lambda value: value.__setitem__(
            "timing_class", "input-latency"),
        "output": lambda value: value["accepted_output_sha256"].__setitem__(
            0, "0" * 64),
        "raw_receipt": lambda value: value["compact"][
            "raw_receipt_sha256"].__setitem__(1, "f" * 64),
        "p50": lambda value: value["reference"].__setitem__(
            "p50_evaluate_wall_ms", 1.0),
        "speedup": lambda value: value.__setitem__(
            "compact_vs_reference_p50_speedup", 9.0),
        "configuration": lambda value: value.__setitem__(
            "compact_configuration_root_sha256", "0" * 64),
        "normalized_root": lambda value: value.__setitem__(
            "normalized_root_sha256", "0" * 64),
    }
    rejected = []
    for name, mutate in mutations.items():
        changed = copy.deepcopy(original)
        mutate(changed)
        try:
            validate(attempt, reference, changed)
        except VerificationError as error:
            rejected.append({"mutation": name, "observed_error": str(error)})
        else:
            raise VerificationError(f"mutation accepted: {name}")
    baseline = validate(attempt, reference)
    return {
        "schema": "neuralstudio.fg.runtime-benchmark-negative-audit.v1",
        "baseline_verification_root_sha256":
            baseline["verification_root_sha256"],
        "mutations": rejected,
        "all_mutations_rejected": True,
        "result": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "negative-audit"))
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = (validate(arguments.attempt, arguments.reference)
              if arguments.mode == "verify"
              else negative(arguments.attempt, arguments.reference))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        if arguments.output.exists():
            raise VerificationError(f"preserve existing output: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
