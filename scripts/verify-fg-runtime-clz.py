#!/usr/bin/env python3
"""Verify exact CLZ promotion, interleaved timing, and boundary profiling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

CHECKPOINT_ROOT = "2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156"
HMMA_LIBS = (
    "conv_3x3", "conv_1x1", "conv_pre", "conv0", "conv1", "conv2",
    "bot0", "bot1", "block1_conv0", "block1_conv1", "block1_conv2",
    "block1_bot0", "block1_bot1",
)
SPAN_LABELS = (
    "status", "L3", "L4", "L5-high", "L5-low", "L5-pack", "L6",
    "L7-texture-to-buffer", "L7-kernel", "L7-buffer-to-texture", "L8",
    "L19", "L20-L51", "L52", "L53", "L54", "L55",
    "L55-to-L84-blit", "L56-L83", "L84", "L85", "L86", "L87",
    "caller-output-copy",
)
SPAN_DOMAINS = [0] * 12 + [1] * 11 + [2]


class VerificationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def expected_outputs(repo: Path) -> list[str]:
    return load(repo / "evidence/specs/fg-runtime-v1/live-alias/verification.json")[
        "output_sha256"
    ]


def validate_correctness(
    repo: Path,
    mac: Path,
    rtx: Path,
    backend_override: dict[str, Any] | None = None,
    promotion_override: dict[str, Any] | None = None,
    mac_runs_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = expected_outputs(repo)
    backend = backend_override or load(mac / "backend-receipt.json")
    promotion = promotion_override or load(mac / "clz-promotion-receipt.json")
    require(
        backend.get("schema") == "neuralstudio.fg.runtime-backend.v1"
        and backend.get("hmma_mode") == "clz"
        and backend.get("auxiliary_mode") == "live"
        and backend.get("ex2_mode") == "compact"
        and backend.get("checkpoint_root_sha256") == CHECKPOINT_ROOT
        and backend.get("output_sha256") == expected
        and backend.get("fresh_process_count") == 4
        and backend.get("poison_bytes") == [90, 165]
        and backend.get("metal_resource_allocations_per_generated_frame") == [210] + [0] * 14
        and backend.get("expected_or_archived_intermediate_ingress") is False
        and backend.get("result") == "pass",
        "backend policy",
    )
    require(
        promotion.get("schema") == "neuralstudio.fg.runtime-clz-promotion.v1"
        and promotion.get("hmma_mode") == "clz"
        and promotion.get("hmma_source_count") == len(HMMA_LIBS)
        and len(promotion.get("compiled_hmma_source_manifest", [])) == len(HMMA_LIBS)
        and promotion.get("checkpoint_root_sha256") == CHECKPOINT_ROOT
        and promotion.get("output_sha256") == expected
        and promotion.get("fresh_process_count") == 4
        and promotion.get("poison_bytes") == [90, 165]
        and promotion.get("expected_or_archived_intermediate_ingress") is False
        and promotion.get("result") == "pass",
        "promotion policy",
    )
    for relative, expected_sha in promotion["hmma_source_sha256"].items():
        require(sha(repo / relative) == expected_sha, "HMMA source pin: " + relative)
    require(
        sha(repo / "src/fg_conv_simd_l20/hmma_exact_clz.metalh")
        == promotion["clz_source_sha256"]
        and sha(repo / "src/fg_conv_simd_l20/hmma_exact_clz_override.metalh")
        == promotion["override_source_sha256"],
        "CLZ helper pins",
    )
    overlay = mac.parent / (mac.name + "-overlay")
    for name, expected_sha in promotion["compiled_overlay_sha256"].items():
        require(sha(overlay / name) == expected_sha, "overlay pin: " + name)
    for name, expected_sha in promotion["hmma_metallib_sha256"].items():
        require(name in HMMA_LIBS and sha(mac / "libs" / f"{name}.metallib") == expected_sha,
                "CLZ metallib pin: " + name)
    require(set(promotion["hmma_metallib_sha256"]) == set(HMMA_LIBS),
            "CLZ metallib coverage")

    mac_dirs = sorted(path for path in mac.glob("fg-runtime-poison-*-process-*") if path.is_dir())
    rtx_dirs = sorted(path for path in rtx.glob("fg-production-contract-l3-l87-poison-*-process-*")
                      if path.is_dir())
    mac_runs = mac_runs_override or [load(path / "run-receipt.json") for path in mac_dirs]
    rtx_runs = [load(path / "temporal-receipt.json") for path in rtx_dirs]
    require(len(mac_dirs) == len(rtx_dirs) == len(mac_runs) == 4, "process count")
    require(len({run.get("process_id") for run in mac_runs}) == 4
            and len({run.get("process_id") for run in rtx_runs}) == 4,
            "fresh process IDs")
    require(sorted(run.get("poison_byte") for run in mac_runs) == [90, 90, 165, 165]
            and sorted(run.get("poison_byte") for run in rtx_runs) == [90, 90, 165, 165],
            "poison coverage")
    for mac_run, rtx_run in zip(mac_runs, rtx_runs):
        checkpoints = mac_run.get("checkpoint_sha256")
        require(
            mac_run.get("schema") == "neuralstudio.fg.runtime-run.v1"
            and mac_run.get("ex2_mode") == "compact"
            and mac_run.get("auxiliary_mode") == "live"
            and mac_run.get("runtime_input_mismatched_bytes") == 0
            and mac_run.get("runtime_canary_mismatched_bytes") == 0
            and mac_run.get("resource_allocations_per_generated_frame") == [210] + [0] * 14
            and mac_run.get("expected_or_archived_intermediate_ingress") is False
            and mac_run.get("result") == "pass",
            "Mac process policy",
        )
        require(
            isinstance(checkpoints, dict) and len(checkpoints) == 1149
            and canonical(checkpoints) == CHECKPOINT_ROOT
            and checkpoints == rtx_run.get("checkpoint_sha256")
            and mac_run.get("output_sha256") == expected
            and mac_run.get("output_sha256") == rtx_run.get("final_output_sha256"),
            "RTX/Mac checkpoint parity",
        )
    representative_mac = mac / "fg-runtime-poison-a5-process-a"
    representative_rtx = rtx / "fg-production-contract-l3-l87-poison-a5-process-a"
    for frame, expected_sha in enumerate(expected, 1):
        mac_bytes = (representative_mac / f"generated-{frame}.rgba8").read_bytes()
        rtx_bytes = (representative_rtx / f"generated-{frame}/l20-l87/output.bin").read_bytes()
        require(mac_bytes == rtx_bytes and len(mac_bytes) == 921600
                and hashlib.sha256(mac_bytes).hexdigest() == expected_sha,
                f"retrieved frame {frame}")
    result = {
        "schema": "neuralstudio.fg.runtime-clz-correctness-verification.v1",
        "checkpoint_root_sha256": CHECKPOINT_ROOT,
        "output_sha256": expected,
        "hmma_source_count": len(HMMA_LIBS),
        "fresh_processes_per_backend": 4,
        "poison_bytes": [90, 165],
        "retrieved_outputs_byte_exact": True,
        "metal_resource_allocations_per_generated_frame": [210] + [0] * 14,
        "expected_or_archived_intermediate_ingress": False,
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def validate_benchmark_run(run: dict[str, Any], expected: list[str], run_id: str) -> None:
    require(
        run.get("schema") == "neuralstudio.fg.runtime-benchmark-run.v1"
        and run.get("run_id") == run_id
        and run.get("ex2_mode") == "compact"
        and run.get("auxiliary_mode") == "live"
        and run.get("poison_byte") == 165
        and run.get("profiling_enabled") is False
        and run.get("counter_attachment") is False
        and run.get("pre_output_sha256") == expected
        and run.get("measured_output_sha256") == expected
        and run.get("post_output_sha256") == expected
        and run.get("output_hashes_comparable") is True
        and run.get("measured_metal_resource_allocations") == [0] * 15
        and run.get("caller_owned_input_mismatches") == 0
        and run.get("expected_or_archived_intermediate_ingress") is False
        and run.get("address_fields_recorded") is False
        and run.get("result") == "pass",
        "benchmark raw run",
    )


def validate_benchmark(
    repo: Path,
    benchmark: Path,
    loop_attempt: Path,
    clz_attempt: Path,
    override: dict[str, Any] | None = None,
    runs_override: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    receipt = override or load(benchmark / "comparison.json")
    expected = expected_outputs(repo)
    require(
        receipt.get("schema") == "neuralstudio.fg.runtime-clz-benchmark-comparison.v1"
        and receipt.get("profile") == "640x360-compact-live-early-aliases"
        and receipt.get("diagnostic_component_only") is True
        and receipt.get("production_benchmark") is False
        and receipt.get("profiling_instrumentation") is False
        and receipt.get("execution_order") == ["loop-1", "clz-1", "clz-2", "loop-2", "loop-3", "clz-3"]
        and receipt.get("accepted_checkpoint_root_sha256") == CHECKPOINT_ROOT
        and receipt.get("accepted_output_sha256") == expected
        and receipt.get("expected_or_archived_intermediate_ingress") is False
        and receipt.get("address_fields_recorded") is False
        and receipt.get("result") == "pass",
        "benchmark policy",
    )
    attempts = {"loop": loop_attempt, "clz": clz_attempt}
    all_processes: list[int] = []
    for mode in ("loop", "clz"):
        paths = sorted((benchmark / mode).glob("run-*.json"))
        runs = (runs_override or {}).get(mode) or [load(path) for path in paths]
        require(len(paths) == len(runs) == 3, mode + ": run count")
        values: list[float] = []
        for ordinal, run in enumerate(runs, 1):
            validate_benchmark_run(run, expected, f"fg-runtime-clz-{mode}-{ordinal}")
            values.extend(run["measured_evaluate_wall_ms"])
            all_processes.append(run["process_id"])
        block = receipt[mode]
        stats = {
            "p50_evaluate_wall_ms": percentile(values, .5),
            "p95_evaluate_wall_ms": percentile(values, .95),
            "p99_evaluate_wall_ms": percentile(values, .99),
            "mean_evaluate_wall_ms": statistics.mean(values),
            "stddev_evaluate_wall_ms": statistics.pstdev(values),
            "max_evaluate_wall_ms": max(values),
        }
        require(block.get("hmma_mode") == mode
                and block.get("process_ids") == [run["process_id"] for run in runs]
                and block.get("raw_receipt_sha256") == [sha(path) for path in paths]
                and block.get("evaluate_wall_ms") == values
                and block.get("binary_sha256") == sha(attempts[mode] / "fg-runtime-benchmark")
                and block.get("runtime_tail_sha256") == sha(attempts[mode] / "build/runtime_tail.mm"),
                mode + ": benchmark pins")
        for key, value in stats.items():
            require(close(block.get(key), value), mode + ": " + key)
        for name in HMMA_LIBS:
            require(block["hmma_metallib_sha256"][name]
                    == sha(attempts[mode] / "libs" / f"{name}.metallib"),
                    mode + ": metallib " + name)
    require(len(set(all_processes)) == 6, "benchmark process freshness")
    loop_libs = {path.name: sha(path) for path in loop_attempt.joinpath("libs").glob("*.metallib")}
    clz_libs = {path.name: sha(path) for path in clz_attempt.joinpath("libs").glob("*.metallib")}
    require(set(loop_libs) == set(clz_libs), "metallib inventory")
    changed = {name.removesuffix(".metallib") for name in loop_libs
               if loop_libs[name] != clz_libs[name]}
    require(changed == set(HMMA_LIBS), "only HMMA metallibs changed")
    require(close(receipt["clz_vs_interleaved_loop_p50_speedup"],
                  receipt["loop"]["p50_evaluate_wall_ms"] / receipt["clz"]["p50_evaluate_wall_ms"])
            and close(receipt["clz_vs_previous_live_hotpath_p50_speedup"],
                      receipt["previous_live_hotpath_p50_ms"] / receipt["clz"]["p50_evaluate_wall_ms"]),
            "benchmark speedup")
    normalized = {key: value for key, value in receipt.items()
                  if key not in ("loop", "clz", "normalized_root_sha256")}
    require(canonical(normalized) == receipt.get("normalized_root_sha256"),
            "benchmark normalized root")
    result = {
        "schema": "neuralstudio.fg.runtime-clz-benchmark-verification.v1",
        "comparison_sha256": sha(benchmark / "comparison.json"),
        "normalized_root_sha256": receipt["normalized_root_sha256"],
        "loop_p50_evaluate_wall_ms": receipt["loop"]["p50_evaluate_wall_ms"],
        "clz_p50_evaluate_wall_ms": receipt["clz"]["p50_evaluate_wall_ms"],
        "clz_vs_loop_p50_speedup": receipt["clz_vs_interleaved_loop_p50_speedup"],
        "changed_metallib_count": len(changed),
        "unchanged_metallib_count": len(loop_libs) - len(changed),
        "output_hashes_comparable": True,
        "production_benchmark": False,
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def validate_profile_run(run: dict[str, Any], expected: list[str], run_id: str) -> None:
    validate_benchmark_run({**run, "profiling_enabled": False}, expected, run_id)
    require(run.get("profiling_enabled") is True
            and len(run.get("measured_command_spans", [])) == 15,
            "profile raw run")


def validate_profile(
    repo: Path,
    profile: Path,
    override: dict[str, Any] | None = None,
    runs_override: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    receipt = override or load(profile / "profile.json")
    expected = expected_outputs(repo)
    require(
        receipt.get("schema") == "neuralstudio.fg.runtime-clz-boundary-profile.v1"
        and receipt.get("profile") == "640x360-compact-live-early-aliases"
        and receipt.get("diagnostic_instrumentation") is True
        and receipt.get("production_benchmark") is False
        and receipt.get("command_spans_per_frame") == 24
        and receipt.get("selected_wait_boundaries") == 23
        and receipt.get("accepted_checkpoint_root_sha256") == CHECKPOINT_ROOT
        and receipt.get("accepted_output_sha256") == expected
        and receipt.get("expected_or_archived_intermediate_ingress") is False
        and receipt.get("address_fields_recorded") is False
        and receipt.get("result") == "pass",
        "profile policy",
    )
    all_processes: list[int] = []
    for mode in ("loop", "clz"):
        paths = sorted((profile / mode).glob("run-*.json"))
        runs = (runs_override or {}).get(mode) or [load(path) for path in paths]
        require(len(paths) == len(runs) == 3, mode + ": profile count")
        cpu = [[] for _ in SPAN_LABELS]
        gpu = [[] for _ in SPAN_LABELS]
        evaluate: list[float] = []
        cpu_sum: list[float] = []
        gpu_sum: list[float] = []
        unspanned: list[float] = []
        for ordinal, run in enumerate(runs, 1):
            validate_profile_run(run, expected, f"fg-runtime-clz-profile-{mode}-{ordinal}")
            all_processes.append(run["process_id"])
            for wall, spans in zip(run["measured_evaluate_wall_ms"], run["measured_command_spans"]):
                require(len(spans) == 24
                        and [span["ordinal"] for span in spans] == list(range(24))
                        and [span["domain"] for span in spans] == SPAN_DOMAINS,
                        mode + ": span identity")
                frame_cpu = sum(span["cpu_commit_wait_ms"] for span in spans)
                frame_gpu = sum(span["gpu_ms"] for span in spans)
                evaluate.append(wall); cpu_sum.append(frame_cpu); gpu_sum.append(frame_gpu)
                unspanned.append(wall - frame_cpu)
                for index, span in enumerate(spans):
                    cpu[index].append(span["cpu_commit_wait_ms"])
                    gpu[index].append(span["gpu_ms"])
        block = receipt[mode]
        require(block.get("hmma_mode") == mode
                and block.get("process_ids") == [run["process_id"] for run in runs]
                and block.get("raw_receipt_sha256") == [sha(path) for path in paths],
                mode + ": profile pins")
        aggregates = {
            "evaluate_wall_p50_ms": statistics.median(evaluate),
            "command_commit_wait_sum_p50_ms": statistics.median(cpu_sum),
            "gpu_span_sum_p50_ms": statistics.median(gpu_sum),
            "unspanned_host_work_p50_ms": statistics.median(unspanned),
        }
        for key, value in aggregates.items():
            require(close(block.get(key), value), mode + ": " + key)
        require(len(block.get("spans", [])) == 24, mode + ": span coverage")
        for index, label in enumerate(SPAN_LABELS):
            span = block["spans"][index]
            require(span.get("ordinal") == index and span.get("label") == label
                    and span.get("domain") == SPAN_DOMAINS[index]
                    and close(span.get("cpu_commit_wait_p50_ms"), statistics.median(cpu[index]))
                    and close(span.get("gpu_p50_ms"), statistics.median(gpu[index]))
                    and close(span.get("gpu_max_ms"), max(gpu[index])),
                    f"{mode}: span {index}")
    require(len(set(all_processes)) == 6, "profile process freshness")
    calculations = {
        "clz_vs_loop_evaluate_p50_speedup":
            receipt["loop"]["evaluate_wall_p50_ms"] / receipt["clz"]["evaluate_wall_p50_ms"],
        "clz_vs_loop_gpu_span_p50_speedup":
            receipt["loop"]["gpu_span_sum_p50_ms"] / receipt["clz"]["gpu_span_sum_p50_ms"],
        "generic_l20_l51_gpu_speedup":
            receipt["loop"]["spans"][12]["gpu_p50_ms"] / receipt["clz"]["spans"][12]["gpu_p50_ms"],
        "custom_l56_l83_gpu_speedup":
            receipt["loop"]["spans"][18]["gpu_p50_ms"] / receipt["clz"]["spans"][18]["gpu_p50_ms"],
    }
    for key, value in calculations.items():
        require(close(receipt.get(key), value), key)
    normalized = {key: value for key, value in receipt.items()
                  if key not in ("loop", "clz", "normalized_root_sha256")}
    require(canonical(normalized) == receipt.get("normalized_root_sha256"),
            "profile normalized root")
    result = {
        "schema": "neuralstudio.fg.runtime-clz-profile-verification.v1",
        "profile_sha256": sha(profile / "profile.json"),
        "normalized_root_sha256": receipt["normalized_root_sha256"],
        **calculations,
        "output_hashes_comparable": True,
        "production_benchmark": False,
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def validate_provenance(
    args: argparse.Namespace, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    require(args.provenance is not None, "provenance required")
    value = override or load(args.provenance)
    require(
        value.get("schema") == "neuralstudio.fg.runtime-clz-provenance.v1"
        and value.get("benchmark_sha256") == sha(args.benchmark / "comparison.json")
        and value.get("profile_sha256") == sha(args.profile / "profile.json")
        and value.get("rtx_archive_sha256")
        == sha(args.rtx.parent / (args.rtx.name + ".tar.gz"))
        and value.get("expected_or_archived_intermediate_ingress") is False
        and value.get("address_fields_recorded") is False
        and value.get("result") == "pass",
        "provenance policy",
    )
    for relative, expected_sha in value.get("source_sha256", {}).items():
        require((args.repo / relative).is_file()
                and sha(args.repo / relative) == expected_sha,
                "provenance source pin: " + relative)
    require(len(value.get("source_sha256", {})) >= 12, "provenance source coverage")
    for relative, expected_sha in value.get("build_artifact_sha256", {}).items():
        require((args.repo / relative).is_file()
                and sha(args.repo / relative) == expected_sha,
                "provenance build pin: " + relative)
    require(len(value.get("build_artifact_sha256", {})) >= 20,
            "provenance build coverage")
    result = {
        "schema": "neuralstudio.fg.runtime-clz-provenance-verification.v1",
        "source_pin_count": len(value["source_sha256"]),
        "build_pin_count": len(value["build_artifact_sha256"]),
        "benchmark_sha256": value["benchmark_sha256"],
        "profile_sha256": value["profile_sha256"],
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def verify(args: argparse.Namespace) -> dict[str, Any]:
    correctness = validate_correctness(args.repo, args.mac, args.rtx)
    benchmark = validate_benchmark(args.repo, args.benchmark, args.loop, args.mac)
    profile = validate_profile(args.repo, args.profile)
    provenance = validate_provenance(args) if args.provenance is not None else None
    result = {
        "schema": "neuralstudio.fg.runtime-clz-verification.v1",
        "correctness_verification_root_sha256": correctness["verification_root_sha256"],
        "benchmark_verification_root_sha256": benchmark["verification_root_sha256"],
        "profile_verification_root_sha256": profile["verification_root_sha256"],
        "provenance_verification_root_sha256":
            provenance["verification_root_sha256"] if provenance is not None else None,
        "checkpoint_root_sha256": CHECKPOINT_ROOT,
        "loop_p50_evaluate_wall_ms": benchmark["loop_p50_evaluate_wall_ms"],
        "clz_p50_evaluate_wall_ms": benchmark["clz_p50_evaluate_wall_ms"],
        "clz_vs_loop_p50_speedup": benchmark["clz_vs_loop_p50_speedup"],
        "generic_l20_l51_gpu_speedup": profile["generic_l20_l51_gpu_speedup"],
        "custom_l56_l83_gpu_speedup": profile["custom_l56_l83_gpu_speedup"],
        "expected_or_archived_intermediate_ingress": False,
        "claims": {"netal_integration": False, "production_performance": False},
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def negative(args: argparse.Namespace) -> dict[str, Any]:
    rejected: list[dict[str, str]] = []
    cases: list[tuple[str, Callable[[], None]]] = []
    backend = load(args.mac / "backend-receipt.json")
    promotion = load(args.mac / "clz-promotion-receipt.json")
    mac_runs = [load(path / "run-receipt.json") for path in sorted(
        path for path in args.mac.glob("fg-runtime-poison-*-process-*") if path.is_dir())]
    comparison = load(args.benchmark / "comparison.json")
    benchmark_runs = {mode: [load(path) for path in sorted((args.benchmark / mode).glob("run-*.json"))]
                      for mode in ("loop", "clz")}
    profile = load(args.profile / "profile.json")
    profile_runs = {mode: [load(path) for path in sorted((args.profile / mode).glob("run-*.json"))]
                    for mode in ("loop", "clz")}
    provenance = load(args.provenance) if args.provenance is not None else None

    def correctness_backend() -> None:
        value = copy.deepcopy(backend); value["hmma_mode"] = "loop"
        validate_correctness(args.repo, args.mac, args.rtx, backend_override=value)
    def correctness_source() -> None:
        value = copy.deepcopy(promotion)
        value["hmma_source_sha256"][next(iter(value["hmma_source_sha256"]))] = "0" * 64
        validate_correctness(args.repo, args.mac, args.rtx, promotion_override=value)
    def correctness_checkpoint() -> None:
        value = copy.deepcopy(mac_runs)
        value[0]["checkpoint_sha256"][next(iter(value[0]["checkpoint_sha256"]))] = "f" * 64
        validate_correctness(args.repo, args.mac, args.rtx, mac_runs_override=value)
    def correctness_process() -> None:
        value = copy.deepcopy(mac_runs); value[1]["process_id"] = value[0]["process_id"]
        validate_correctness(args.repo, args.mac, args.rtx, mac_runs_override=value)
    def correctness_poison() -> None:
        value = copy.deepcopy(mac_runs); value[0]["poison_byte"] = 17
        validate_correctness(args.repo, args.mac, args.rtx, mac_runs_override=value)
    def benchmark_production() -> None:
        value = copy.deepcopy(comparison); value["production_benchmark"] = True
        validate_benchmark(args.repo, args.benchmark, args.loop, args.mac, override=value)
    def benchmark_speedup() -> None:
        value = copy.deepcopy(comparison); value["clz_vs_interleaved_loop_p50_speedup"] = 9.0
        validate_benchmark(args.repo, args.benchmark, args.loop, args.mac, override=value)
    def benchmark_output() -> None:
        value = copy.deepcopy(benchmark_runs); value["clz"][0]["post_output_sha256"][0] = "0" * 64
        validate_benchmark(args.repo, args.benchmark, args.loop, args.mac, runs_override=value)
    def benchmark_process() -> None:
        value = copy.deepcopy(benchmark_runs); value["clz"][0]["process_id"] = value["loop"][0]["process_id"]
        validate_benchmark(args.repo, args.benchmark, args.loop, args.mac, runs_override=value)
    def profile_production() -> None:
        value = copy.deepcopy(profile); value["production_benchmark"] = True
        validate_profile(args.repo, args.profile, override=value)
    def profile_span() -> None:
        value = copy.deepcopy(profile); value["clz"]["spans"][18]["gpu_p50_ms"] = 1.0
        validate_profile(args.repo, args.profile, override=value)
    def profile_identity() -> None:
        value = copy.deepcopy(profile_runs); value["loop"][0]["measured_command_spans"][0][12]["ordinal"] = 99
        validate_profile(args.repo, args.profile, runs_override=value)
    def provenance_source() -> None:
        require(provenance is not None, "provenance mutation requires provenance")
        value = copy.deepcopy(provenance)
        value["source_sha256"][next(iter(value["source_sha256"]))] = "0" * 64
        validate_provenance(args, value)
    cases.extend([
        ("backend-mode", correctness_backend), ("source-pin", correctness_source),
        ("checkpoint", correctness_checkpoint), ("correctness-process", correctness_process),
        ("poison", correctness_poison), ("benchmark-production", benchmark_production),
        ("benchmark-speedup", benchmark_speedup), ("benchmark-output", benchmark_output),
        ("benchmark-process", benchmark_process), ("profile-production", profile_production),
        ("profile-span", profile_span), ("profile-identity", profile_identity),
    ])
    if provenance is not None:
        cases.append(("provenance-source", provenance_source))
    for name, function in cases:
        try:
            function()
        except VerificationError as error:
            rejected.append({"mutation": name, "observed_error": str(error)})
        else:
            raise VerificationError("mutation accepted: " + name)
    baseline = verify(args)
    return {
        "schema": "neuralstudio.fg.runtime-clz-negative-audit.v1",
        "baseline_verification_root_sha256": baseline["verification_root_sha256"],
        "mutations": rejected,
        "all_mutations_rejected": True,
        "result": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "negative-audit"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args) if args.mode == "verify" else negative(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise VerificationError("preserve existing output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, OSError, json.JSONDecodeError) as error:
        raise SystemExit("verification_error=" + str(error))
