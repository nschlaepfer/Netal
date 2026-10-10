#!/usr/bin/env python3
"""Confirm temporal allocation29 evidence mutations fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


RUN = "allocation29-post-generated-poison-a5-process-a"


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()).hexdigest()


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(data)


def edit_json(path: Path, mutate: Callable[[dict], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    mutate(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def edit_jsonl(path: Path, predicate: Callable[[dict], bool], mutate: Callable[[dict], None]) -> None:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    matches = [value for value in values if predicate(value)]
    if len(matches) != 1:
        raise ValueError(f"mutation expected one JSONL event, found {len(matches)}: {path}")
    mutate(matches[0])
    path.write_text("".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values))


def run_verifier(verifier: Path, root: Path, repository: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3", str(verifier),
            "--input-root", str(root),
            "--receipt", str(root / "allocation29-temporal-capture-receipt.json"),
            "--repository", str(repository),
            "--output", str(root / "verification.json"),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.input_root.resolve()
    repository = arguments.repository.resolve()
    verifier = repository / "scripts/verify-fg-allocation29-temporal-capture.py"

    baseline = run_verifier(verifier, source, repository)
    if baseline.returncode != 0:
        raise SystemExit(f"baseline temporal evidence rejected:\n{baseline.stdout}")

    run = Path(RUN)
    cases: list[tuple[str, Callable[[Path], None]]] = [
        ("state_byte", lambda root: flip(root / run / "default-token-000029.bin", 117)),
        ("manifest_source_token", lambda root: edit_json(
            root / run / "runtime-default-buffers-post-evaluate-1.manifest.json",
            lambda value: value["entries"][0].__setitem__("source_token", 30))),
        ("host_second_output", lambda root: edit_json(
            root / run / "fg-run-receipt.json",
            lambda value: value["frames"][2].__setitem__("output_interpolated_sha256", "0" * 64))),
        ("process_proxy", lambda root: edit_json(
            root / run / f"{RUN}-process.json",
            lambda value: value.__setitem__("proxy_sha256", "0" * 64))),
        ("descriptor_summary", lambda root: edit_jsonl(
            root / run / "descriptor-semantic.jsonl",
            lambda value: value.get("event") == "descriptor_capture_summary",
            lambda value: value.__setitem__("capture_healthy", False))),
        ("allocation29_format", lambda root: edit_jsonl(
            root / run / "runtime-allocations-semantic.jsonl",
            lambda value: value.get("event") == "allocation" and value.get("token") == 29,
            lambda value: value.__setitem__("format", 61))),
        ("l52_module", lambda root: edit_jsonl(
            root / run / "nvapi-trace.jsonl",
            lambda value: value.get("event") == "kernel_launch" and value.get("launch_token") == "launch-000052",
            lambda value: value.__setitem__("module_token", "module-000005"))),
        ("capture_phase", lambda root: edit_jsonl(
            root / run / "nvapi-trace.jsonl",
            lambda value: value.get("event") == "phase_marker" and value.get("phase") == "capture_default_buffers.post-evaluate-1",
            lambda value: value.__setitem__("phase", "capture_default_buffers.post-evaluate-0"))),
        ("l95_parameter_handle", lambda root: flip(
            root / run / "launch-000095-kernel-0000.params.bin", 48)),
        ("nvapi_volatile_handle", lambda root: edit_jsonl(
            root / run / "nvapi-volatile.jsonl",
            lambda value: value.get("event") == "resource_object_raw" and value.get("resource_token") == "resource-000042",
            lambda value: value.__setitem__("cuda_object_handle", "0x0000000000000001"))),
        ("descriptor_volatile", lambda root: flip(
            root / run / "descriptor-volatile.jsonl", 127)),
        ("capture_receipt_gate", lambda root: edit_json(
            root / "allocation29-temporal-capture-receipt.json",
            lambda value: value.__setitem__("captured_bytes_fed_back_to_execution", True))),
    ]

    results = []
    with tempfile.TemporaryDirectory(prefix="fg-allocation29-temporal-negative-") as temporary:
        temporary_root = Path(temporary)
        for name, mutation in cases:
            case_root = temporary_root / name
            shutil.copytree(source, case_root)
            mutation(case_root)
            completed = run_verifier(verifier, case_root, repository)
            if completed.returncode == 0:
                raise SystemExit(f"temporal mutation was accepted: {name}")
            results.append({
                "case": name,
                "rejected": True,
                "diagnostic_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            })

    audit = {
        "schema": "neuralstudio.fg.allocation29-temporal-negative-audit.v1",
        "case_count": len(results),
        "rejected_count": sum(result["rejected"] for result in results),
        "cases": results,
        "result": "pass",
    }
    audit["receipt_root_sha256"] = canonical_sha(audit)
    arguments.audit_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"rejected={audit['rejected_count']}/{audit['case_count']}")
    print(f"receipt_root_sha256={audit['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
