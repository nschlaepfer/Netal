#!/usr/bin/env python3
"""Seal the accepted live-alias no-refill timing diagnostic."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    attempt = args.attempt.resolve()
    evidence = args.evidence.resolve()
    if evidence.exists():
        raise SystemExit(f"preserve existing evidence: {evidence}")
    evidence.mkdir(parents=True)

    source_paths = [
        "scripts/benchmark-macos-fg-runtime-live.sh",
        "scripts/verify-fg-runtime-live-benchmark.py",
        "scripts/seal-fg-runtime-live-benchmark.py",
        "scripts/build-macos-fg-runtime.sh",
        "scripts/build-macos-fg-runtime-compact-ex2.sh",
        "scripts/verify-fg-runtime-live-alias.py",
        "src/fg_runtime/fg_runtime.h",
        "src/fg_runtime/fg_runtime.mm",
        "src/fg_runtime/benchmark_host.mm",
        "src/fg_runtime/offline_host.mm",
        "src/fg_runtime/tail_output_bridge.h",
        "src/fg_runtime/tail_output_bridge.mm",
        "src/fg_runtime/tail_resource_cache.inc",
        "src/fg_runtime/rtx_live_aux_bridge.h",
        "src/fg_runtime/rtx_live_aux_bridge.cpp",
    ]
    build_paths = [
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/fg-runtime-benchmark",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/fg-runtime-offline",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/build/runtime_tail.mm",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/libs/conv_3x3.metallib",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/libs/block1_bot0.metallib",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/libs/compositor_l87.metallib",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/libs/mid_l52.metallib",
        "build/fg_runtime/macos/live-alias-v4-hmma-zero/libs/mid_l55.metallib",
        "build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin",
        "build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin",
    ]
    for relative in source_paths + build_paths:
        if not (repo / relative).is_file():
            raise SystemExit(f"missing pinned input: {relative}")

    benchmark = json.loads((attempt / "benchmark.json").read_text())
    provenance = {
        "schema": "neuralstudio.fg.runtime-live-hotpath-provenance.v1",
        "dependencies": {
            "compact_runtime_commit": "70ada913aeb95a547c7fc70beaab8a93c41c6867",
            "boundary_profile_commit": "90c7c33421419b79e3084be8032ceffed307b9d4",
            "live_alias_runtime_commit": "9b3f31d3ee65de2297d47db047850a47c8f99ed5",
            "live_alias_checkpoint_root": "2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156",
            "live_alias_verification_root": "ad2925933009862855e8711bd2c378e98667406b34f32abc614bc545768057f9",
        },
        "benchmark_attempt": "build/fg_runtime/benchmark/live-hotpath-v1",
        "benchmark_sha256": sha(attempt / "benchmark.json"),
        "raw_receipt_sha256": [sha(attempt / f"run-{i}.json") for i in range(1, 4)],
        "source_sha256": {relative: sha(repo / relative) for relative in source_paths},
        "build_artifact_sha256": {relative: sha(repo / relative) for relative in build_paths},
        "binary_sha256": benchmark["binary_sha256"],
        "runtime_tail_sha256": benchmark["runtime_tail_sha256"],
        "timing_poison_byte": 165,
        "dual_poison_correctness_inherited_from_live_alias_gate": True,
        "expected_or_archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    }

    shutil.copyfile(attempt / "benchmark.json", evidence / "benchmark.json")
    for ordinal in range(1, 4):
        shutil.copyfile(attempt / f"run-{ordinal}.json", evidence / f"run-{ordinal}.json")
    shutil.copyfile(attempt / "run-1.json", evidence / "representative-run.json")
    write_json(evidence / "provenance.json", provenance)
    readme = """# Live-alias no-refill runtime timing diagnostic

Three fresh processes retained 45 measured generated frames after warmup.  Each
process also executed untimed exact frames immediately before and after its
measured interval.  All 15 per-frame output roots stayed accepted in every
bracket, caller-owned inputs stayed immutable, no measured Metal allocation
occurred, and no expected or archived intermediate entered execution.

The median synchronous `evaluate` wall time is 120.519042 ms (p95
127.268042 ms).  This is a 4.040789x diagnostic ratio against the earlier
486.992 ms instrumented profile.  It demonstrates the value of caching model
bytes, retaining proven fully-overwritten resources, removing fixture
generation through traced live aliases, and suppressing audit-only readbacks
on the non-audit path.

This is a 640x360 synthetic component timing diagnostic, not a production or
game benchmark, input latency, frame pacing, Netal integration, or FPS claim.
The comparison baseline includes boundary instrumentation, so the ratio is not
a controlled single-variable speedup.  The absolute hot-path distribution and
its untimed exact brackets are the accepted result.

The timing series uses poison `0xA5`; the preceding live-alias correctness gate
covers four fresh processes across both `0xA5` and `0x5A`.  The representative
raw run retains a legacy profile label from the host, while its explicit
`auxiliary_mode=live` and the aggregate profile name identify the executed
configuration.
"""
    (evidence / "README.md").write_text(readme)
    failures = """# Preserved limits and counterexamples

- Reusing uninitialized or partially written scratch storage is forbidden.  The
  fast path only skips refill for resources whose complete overwrite is proven;
  audit mode retains poison, canaries, refill, and readback behavior.
- The 4.040789x figure is diagnostic because the historical 486.992 ms baseline
  was instrumented.  It is not used as a production performance claim.
- Timing used only poison `0xA5`; dual-poison correctness is inherited from the
  separately sealed live-alias gate, not inferred from timing.
- The raw host receipt's legacy profile label is descriptive metadata only;
  `auxiliary_mode=live`, exact output roots, binary pins, and aggregate profile
  identify the measured implementation.
"""
    (evidence / "failure-ledger.md").write_text(failures)

    verifier = repo / "scripts/verify-fg-runtime-live-benchmark.py"
    reference = repo / "evidence/specs/fg-runtime-v1/live-alias/verification.json"
    subprocess.run([
        "python3", str(verifier), "verify", "--attempt", str(attempt),
        "--reference", str(reference), "--output", str(evidence / "verification.json")
    ], check=True)
    subprocess.run([
        "python3", str(verifier), "verify-provenance", "--attempt", str(attempt),
        "--provenance", str(evidence / "provenance.json"), "--repo", str(repo),
        "--output", str(evidence / "provenance-verification.json")
    ], check=True)
    subprocess.run([
        "python3", str(verifier), "negative-audit", "--attempt", str(attempt),
        "--reference", str(reference), "--provenance", str(evidence / "provenance.json"),
        "--repo", str(repo), "--output", str(evidence / "negative-audit.json")
    ], check=True)

    artifact_names = sorted(path.name for path in evidence.iterdir()
                            if path.is_file() and path.name != "ARTIFACTS.sha256")
    manifest = "".join(f"{sha(evidence / name)}  {name}\n" for name in artifact_names)
    (evidence / "ARTIFACTS.sha256").write_text(manifest)
    print(f"evidence={evidence}")
    print(f"benchmark_sha256={sha(evidence / 'benchmark.json')}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
