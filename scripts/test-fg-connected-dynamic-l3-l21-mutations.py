#!/usr/bin/env python3
"""Mutation negatives for connected dynamic L3-L21 receipts."""

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
    verifier = repository / "scripts/verify-fg-connected-dynamic-l3-l21.py"

    def backend_disagreement(root: Path) -> None:
        path = root / "mac-backend-receipt.json"
        value = load(path)
        value["resolutions"]["720p"]["l20_sha256"] = "0" * 64
        save(path, value)
        accepted_path = root / "accepted-receipt.json"
        accepted = load(accepted_path)
        accepted["mac_backend_root_sha256"] = load(path)["receipt_root_sha256"]
        save(accepted_path, accepted)

    cases: list[tuple[str, str, Callable[[Path], None]]] = [
        ("early-root", "topology drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value.__setitem__(
                                      "accepted_early_receipt_root_sha256", "0" * 64))),
        ("checkpoint-order", "topology drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value["checkpoint_order"].reverse())),
        ("weight-hash", "group topology drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value["l20"].__setitem__(
                                      "module_sha256", "1" * 64))),
        ("handoff-label", "group topology drift",
         lambda root: mutate_file(root, "topology-receipt.json",
                                  lambda value: value.__setitem__(
                                      "allocation37_handoff", "refilled"))),
        ("source-hash", "implementation drift",
         lambda root: mutate_file(root, "implementation-receipt.json",
                                  lambda value: value["source_sha256"].__setitem__(
                                      "src/fg_connected_dynamic_l3_l21/rtx_tail.inc",
                                      "2" * 64))),
        ("tail-count", "implementation drift",
         lambda root: mutate_file(root, "implementation-receipt.json",
                                  lambda value: value.__setitem__(
                                      "live_tail_injection_count", 1))),
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
        ("live-handoff", "RTX backend drift",
         lambda root: mutate_file(root, "rtx-backend-receipt.json",
                                  lambda value: value.__setitem__(
                                      "live_allocation37_device_handoff", False))),
        ("forbidden-ingress", "RTX backend drift",
         lambda root: mutate_file(root, "rtx-backend-receipt.json",
                                  lambda value: value.__setitem__(
                                      "expected_or_archived_intermediate_ingress", True))),
        ("backend-output", "RTX/Metal group disagreement", backend_disagreement),
        ("accepted-byte-count", "accepted comparison drift",
         lambda root: mutate_file(root, "accepted-receipt.json",
                                  lambda value: value.__setitem__(
                                      "l20_l21_full_artifact_compared_bytes", 1))),
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
            prefix="fg-connected-dynamic-l3-l21-negative-") as temp:
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
        "schema": "neuralstudio.fg.connected-dynamic-l3-l21-negative-audit.v1",
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
