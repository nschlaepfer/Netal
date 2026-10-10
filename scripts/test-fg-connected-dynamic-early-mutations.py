#!/usr/bin/env python3
"""Mutation negatives for connected dynamic early receipts."""

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


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    body = dict(value)
    body.pop("receipt_root_sha256", None)
    body["receipt_root_sha256"] = digest(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def mutate_file(root: Path, name: str,
                change: Callable[[dict], None]) -> None:
    path = root / name
    value = load(path)
    change(value)
    save(path, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    verifier = repository / "scripts/verify-fg-connected-dynamic-early.py"

    def backend_disagreement(root: Path) -> None:
        path = root / "mac-backend-receipt.json"
        value = load(path)
        value["resolutions"]["720p"]["checkpoint_sha256"][
            "generated-module5"] = "0" * 64
        save(path, value)
        accepted_path = root / "accepted-receipt.json"
        accepted = load(accepted_path)
        accepted["mac_backend_root_sha256"] = load(path)["receipt_root_sha256"]
        save(accepted_path, accepted)

    cases: list[tuple[str, str, Callable[[Path], None]]] = [
        ("launch-count", "topology drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value.__setitem__(
                                      "selected_launch_count", 20))),
        ("module23-level5", "level binding drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value["module23_bindings"].__setitem__(
                                      "offsets_0_through_32", "four levels and mask"))),
        ("module24-level-route", "level binding drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value["module24_bindings"].__setitem__(
                                      "offsets16_through_40", "selected levels"))),
        ("mixed-four-k", "topology drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value.__setitem__(
                                      "four_k_mixed_module5_resources", False))),
        ("source-hash", "source identity drift",
         lambda root: mutate_file(root, "implementation-receipt.json",
                                  lambda value: value["source_sha256"].__setitem__(
                                      "src/fg_connected_dynamic_early/contract.h",
                                      "1" * 64))),
        ("point-bits", "implementation drift",
         lambda root: mutate_file(root, "implementation-receipt.json",
                                  lambda value: value.__setitem__(
                                      "module5_normalized_point_fraction_bits", 12))),
        ("archived-handoff", "implementation drift",
         lambda root: mutate_file(root, "implementation-receipt.json",
                                  lambda value: value.__setitem__(
                                      "live_device_handoffs_only", False))),
        ("failure-ledger", "failure ledger drift",
         lambda root: mutate_file(root, "counterexamples.json",
                                  lambda value: value.__setitem__(
                                      "all_failures_preserved", False))),
        ("rtx-process-count", "RTX process count drift",
         lambda root: mutate_file(root, "rtx-backend-receipt.json",
                                  lambda value: value.__setitem__(
                                      "fresh_process_count", 15))),
        ("mac-process-count", "Mac process count drift",
         lambda root: mutate_file(root, "mac-backend-receipt.json",
                                  lambda value: value.__setitem__(
                                      "fresh_process_count", 15))),
        ("forbidden-ingress", "RTX forbidden ingress",
         lambda root: mutate_file(root, "rtx-backend-receipt.json",
                                  lambda value: value.__setitem__(
                                      "expected_or_captured_output_ingress", True))),
        ("backend-checkpoint", "RTX/Metal checkpoint disagreement",
         backend_disagreement),
        ("accepted-byte-count", "accepted comparison drift",
         lambda root: mutate_file(root, "accepted-receipt.json",
                                  lambda value: value.__setitem__(
                                      "rtx_mac_artifact_compared_bytes", 1))),
        ("scope-escalation", "scope drift",
         lambda root: mutate_file(root, "accepted-receipt.json",
                                  lambda value: value["claims"].__setitem__(
                                      "complete_generated_frame", True))),
        ("address-like-value", "address-like value",
         lambda root: mutate_file(root, "accepted-receipt.json",
                                  lambda value: value.__setitem__(
                                      "debug_value", "0x1234567890abcdef"))),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    results = []
    with tempfile.TemporaryDirectory(
            prefix="fg-connected-dynamic-early-negative-") as temp:
        for name, expected, mutation in cases:
            case_root = Path(temp) / name
            shutil.copytree(args.evidence, case_root)
            mutation(case_root)
            process = subprocess.run(
                ["python3", str(verifier), "receipt", "--repository",
                 str(repository), "--evidence", str(case_root),
                 "--skip-negative"],
                cwd=repository, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            observed = process.stdout.strip()
            if process.returncode != 2 or expected not in observed:
                raise RuntimeError(
                    f"mutation {name} failed open: exit={process.returncode}, "
                    f"output={observed!r}")
            results.append({
                "mutation": name,
                "expected_error_fragment": expected,
                "observed_error": next(
                    (line for line in observed.splitlines()
                     if line.startswith("verification_error=")), observed),
                "exit_code": process.returncode,
                "result": "pass",
            })
    audit = {
        "schema": "neuralstudio.fg.connected-dynamic-early-negative-audit.v1",
        "verifier_sha256": digest(verifier.read_bytes()),
        "mutation_count": len(results),
        "mutations": results,
        "all_mutations_rejected": True,
        "result": "pass",
    }
    audit["receipt_root_sha256"] = digest(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"mutation_count={len(results)}")
    print(f"negative_audit_root_sha256={audit['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
