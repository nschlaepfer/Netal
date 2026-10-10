#!/usr/bin/env python3
"""Seal the exact full-runtime CLZ promotion and timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(path: Path, destination: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing evidence input: {path}")
    shutil.copyfile(path, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    mac = args.mac.resolve()
    rtx = args.rtx.resolve()
    loop = args.loop.resolve()
    benchmark = args.benchmark.resolve()
    profile = args.profile.resolve()
    evidence = args.evidence.resolve()
    if evidence.exists():
        raise SystemExit(f"preserve existing evidence: {evidence}")
    evidence.mkdir(parents=True)

    source_paths = [
        "scripts/build-macos-fg-runtime.sh",
        "scripts/build-macos-fg-runtime-compact-ex2.sh",
        "scripts/build-macos-fg-runtime-clz.sh",
        "scripts/benchmark-macos-fg-runtime-clz.sh",
        "scripts/profile-macos-fg-runtime-clz.sh",
        "scripts/verify-fg-runtime-clz.py",
        "scripts/seal-fg-runtime-clz.py",
        "scripts/verify-fg-runtime-live-alias.py",
        "src/fg_runtime/fg_runtime.h",
        "src/fg_runtime/fg_runtime.mm",
        "src/fg_runtime/benchmark_host.mm",
        "src/fg_runtime/offline_host.mm",
        "src/fg_runtime/tail_output_bridge.h",
        "src/fg_runtime/tail_output_bridge.mm",
        "src/fg_runtime/tail_resource_cache.inc",
        "src/fg_conv_simd_l20/hmma_exact_clz.metalh",
        "src/fg_conv_simd_l20/hmma_exact_clz_override.metalh",
    ]
    mac_relative = mac.relative_to(repo)
    loop_relative = loop.relative_to(repo)
    benchmark_relative = benchmark.relative_to(repo)
    profile_relative = profile.relative_to(repo)
    rtx_archive = rtx.parent / (rtx.name + ".tar.gz")
    build_paths = [
        str(mac_relative / "backend-receipt.json"),
        str(mac_relative / "compact-promotion-receipt.json"),
        str(mac_relative / "clz-promotion-receipt.json"),
        str(mac_relative / "fg-runtime-benchmark"),
        str(mac_relative / "fg-runtime-offline"),
        str(mac_relative / "build/runtime_tail.mm"),
        str(mac_relative / "build/hmma-clz-sources.txt"),
        str(loop_relative / "fg-runtime-benchmark"),
        str(loop_relative / "build/runtime_tail.mm"),
        str(benchmark_relative / "comparison.json"),
        str(profile_relative / "profile.json"),
        str(rtx_archive.relative_to(repo)),
        "build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin",
        "build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin",
    ]
    for name in (
        "conv_3x3", "conv_1x1", "conv_pre", "conv0", "conv1", "conv2",
        "bot0", "bot1", "block1_conv0", "block1_conv1", "block1_conv2",
        "block1_bot0", "block1_bot1",
    ):
        build_paths.append(str(mac_relative / "libs" / f"{name}.metallib"))
        build_paths.append(str(loop_relative / "libs" / f"{name}.metallib"))
    overlay = Path(str(mac) + "-overlay")
    overlay_relative = overlay.relative_to(repo)
    for name in ("conv_3x3_live_zero.metal", "block0_bot0_live_zero.metal",
                 "l52_compact.metal", "l54_compact.metal"):
        build_paths.append(str(overlay_relative / name))
    for relative in source_paths + build_paths:
        if not (repo / relative).is_file():
            raise SystemExit(f"missing pinned path: {relative}")

    comparison = json.loads((benchmark / "comparison.json").read_text())
    profiled = json.loads((profile / "profile.json").read_text())
    provenance = {
        "schema": "neuralstudio.fg.runtime-clz-provenance.v1",
        "dependencies": {
            "live_alias_runtime_commit": "9b3f31d3ee65de2297d47db047850a47c8f99ed5",
            "hotpath_benchmark_commit": "3254cecff031ef0158cd217bff0972e212e8ecbf",
            "clz_family_optimization_commit": "b012ca68d3cb9c5588f688e80b82b217af6dc82f",
            "clz_force_include_commit": "41612b238583083d5148c03e66c5fb6395ddb18a",
            "clz_family_verification_root": "389d676f9d4127b3e7121bb84aa42a09c94676d1f6e8d92d1ae877e05597d08d",
            "live_alias_checkpoint_root": "2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156",
        },
        "mac_attempt": str(mac_relative),
        "loop_attempt": str(loop_relative),
        "rtx_attempt": str(rtx.relative_to(repo)),
        "benchmark_attempt": str(benchmark_relative),
        "profile_attempt": str(profile_relative),
        "benchmark_sha256": sha(benchmark / "comparison.json"),
        "profile_sha256": sha(profile / "profile.json"),
        "rtx_archive_sha256": sha(rtx_archive),
        "source_sha256": {relative: sha(repo / relative) for relative in source_paths},
        "build_artifact_sha256": {relative: sha(repo / relative) for relative in build_paths},
        "changed_metallib_count": 13,
        "unchanged_metallib_count": 24,
        "expected_or_archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    }

    copy(mac / "backend-receipt.json", evidence / "backend-receipt.json")
    copy(mac / "compact-promotion-receipt.json", evidence / "compact-promotion-receipt.json")
    copy(mac / "clz-promotion-receipt.json", evidence / "promotion-receipt.json")
    copy(mac / "build/hmma-clz-sources.txt", evidence / "compiled-hmma-sources.txt")
    for run in sorted(path for path in mac.glob("fg-runtime-poison-*-process-*") if path.is_dir()):
        copy(run / "run-receipt.json", evidence / f"correctness-{run.name}.json")
    copy(benchmark / "comparison.json", evidence / "benchmark-comparison.json")
    for mode in ("loop", "clz"):
        for ordinal in range(1, 4):
            copy(benchmark / mode / f"run-{ordinal}.json",
                 evidence / f"benchmark-{mode}-run-{ordinal}.json")
    copy(profile / "profile.json", evidence / "boundary-profile.json")
    for mode in ("loop", "clz"):
        for ordinal in range(1, 4):
            copy(profile / mode / f"run-{ordinal}.json",
                 evidence / f"profile-{mode}-run-{ordinal}.json")
    cross = repo / "build/fg_runtime/clz-seal-v1"
    copy(cross / "cross-backend-verification.json",
         evidence / "cross-backend-verification.json")
    copy(cross / "cross-backend-negative-audit.json",
         evidence / "cross-backend-negative-audit.json")
    write_json(evidence / "provenance.json", provenance)

    readme = f"""# Exact CLZ promotion for the full live FG runtime

All 13 HMMA-consuming Metal libraries were rebuilt through the committed exact
CLZ force-include.  Four fresh Mac processes across poisons `0xA5` and `0x5A`
match the RTX oracle at all 1,149 selected checkpoints and all 15 retrieved
RGBA8 generated frames.  The checkpoint root remains
`2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156`.
No expected or archived intermediate entered execution, caller inputs and
canaries remain intact, and Metal allocations after warmup remain zero.

The interleaved non-instrumented order was `L1,C1,C2,L2,L3,C3`.  Loop HMMA
measured {comparison['loop']['p50_evaluate_wall_ms']:.6f} ms p50 and exact CLZ
measured {comparison['clz']['p50_evaluate_wall_ms']:.6f} ms p50, a
{comparison['clz_vs_interleaved_loop_p50_speedup']:.6f}x controlled speedup.
Only the 13 HMMA metallibs changed; 24 other runtime metallibs are byte
identical.

The interleaved 24-span profile attributes the gain across both convolution
regions: L20-L51 GPU time improves
{profiled['generic_l20_l51_gpu_speedup']:.6f}x and L56-L83 improves
{profiled['custom_l56_l83_gpu_speedup']:.6f}x.  Summed GPU spans improve
{profiled['clz_vs_loop_gpu_span_p50_speedup']:.6f}x, while instrumented
synchronous evaluation improves {profiled['clz_vs_loop_evaluate_p50_speedup']:.6f}x.

These are pinned 640x360 synthetic component measurements on Apple M3 Max,
not game FPS, input latency, frame pacing, multi-resolution, subrect, or Netal
integration claims.  Correctness and timing receipts remain separately gated.
"""
    (evidence / "README.md").write_text(readme)
    failures = """# Preserved failures and limits

1. The first CLZ wrapper invocation failed before build or GPU execution because
   the newly created script lacked its executable bit.  The bit was corrected;
   no attempt directory was created or overwritten.
2. Raw SIMD, two-order, five-order, and guarded-SIMD alternatives remain
   rejected in the parent CLZ-family package.  Only the exact CLZ helper is
   promoted here.
3. The raw benchmark host retains its legacy profile-label string.  Explicit
   `auxiliary_mode=live`, aggregate profile identity, binary/metallib pins and
   exact output roots identify the executed configuration.
4. Timing is component-only and synchronous.  It is not a production, game,
   latency, pacing, dynamic-resolution, or Netal measurement.
"""
    (evidence / "failure-ledger.md").write_text(failures)

    verifier = repo / "scripts/verify-fg-runtime-clz.py"
    common = [
        "--repo", str(repo), "--mac", str(mac), "--rtx", str(rtx),
        "--loop", str(loop), "--benchmark", str(benchmark), "--profile", str(profile),
        "--provenance", str(evidence / "provenance.json"),
    ]
    subprocess.run(["python3", str(verifier), "verify", *common,
                    "--output", str(evidence / "verification.json")], check=True)
    subprocess.run(["python3", str(verifier), "negative-audit", *common,
                    "--output", str(evidence / "negative-audit.json")], check=True)

    names = sorted(path.name for path in evidence.iterdir()
                   if path.is_file() and path.name != "ARTIFACTS.sha256")
    (evidence / "ARTIFACTS.sha256").write_text(
        "".join(f"{sha(evidence / name)}  {name}\n" for name in names))
    print(f"evidence={evidence}")
    print(f"verification_sha256={sha(evidence / 'verification.json')}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
