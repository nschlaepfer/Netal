#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile
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


def mutate(root: Path, name: str, change: Callable[[dict], None]) -> None:
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
    verifier = repository / "scripts/verify-fg-connected-dynamic-l3-l25.py"

    def backend_disagreement(root: Path) -> None:
        path = root / "mac-backend-receipt.json"
        value = load(path)
        value["resolutions"]["1080p"]["l24_sha256"] = "0" * 64
        save(path, value)
        accepted_path = root / "accepted-receipt.json"
        accepted = load(accepted_path)
        accepted["mac_backend_root_sha256"] = load(path)["receipt_root_sha256"]
        save(accepted_path, accepted)

    cases: list[tuple[str, str, Callable[[Path], None]]] = [
        ("parent", "topology drift",
         lambda r: mutate(r, "topology-receipt.json",
                          lambda v: v.__setitem__(
                              "parent_receipt_root_sha256", "0" * 64))),
        ("order", "topology drift",
         lambda r: mutate(r, "topology-receipt.json",
                          lambda v: v["checkpoint_order"].reverse())),
        ("handoff", "topology drift",
         lambda r: mutate(r, "topology-receipt.json",
                          lambda v: v.__setitem__(
                              "live_l23_device_handoff", False))),
        ("model-state", "topology drift",
         lambda r: mutate(r, "topology-receipt.json",
                          lambda v: v["model_state"].__setitem__(
                              "capture_manifest_sha256", "2" * 64))),
        ("abi", "topology drift",
         lambda r: mutate(r, "topology-receipt.json",
                          lambda v: v["l24"].__setitem__(
                              "packed_parameter_bytes", 40))),
        ("source", "implementation drift",
         lambda r: mutate(r, "implementation-receipt.json",
                          lambda v: v["source_sha256"].__setitem__(
                              "src/fg_connected_dynamic_l3_l25/rtx_tail.inc",
                              "1" * 64))),
        ("patch", "implementation drift",
         lambda r: mutate(r, "implementation-receipt.json",
                          lambda v: v.__setitem__("per_input_patch_count", 1))),
        ("failures", "failure ledger drift",
         lambda r: mutate(r, "counterexamples.json",
                          lambda v: v.__setitem__(
                              "all_failures_preserved", False))),
        ("rtx-process", "RTX process count drift",
         lambda r: mutate(r, "rtx-backend-receipt.json",
                          lambda v: v.__setitem__("fresh_process_count", 15))),
        ("mac-process", "Mac process count drift",
         lambda r: mutate(r, "mac-backend-receipt.json",
                          lambda v: v.__setitem__("fresh_process_count", 15))),
        ("rtx-handoff", "receipt link drift",
         lambda r: mutate(r, "rtx-backend-receipt.json",
                          lambda v: v.__setitem__(
                              "live_l23_device_handoff", False))),
        ("rtx-ingress", "receipt link drift",
         lambda r: mutate(r, "rtx-backend-receipt.json",
                          lambda v: v.__setitem__(
                              "expected_or_archived_intermediate_ingress", True))),
        ("backend", "backend disagreement", backend_disagreement),
        ("bytes", "accepted comparison drift",
         lambda r: mutate(r, "accepted-receipt.json",
                          lambda v: v.__setitem__(
                              "l24_l25_full_artifact_compared_bytes", 1))),
        ("mismatch", "accepted comparison drift",
         lambda r: mutate(r, "accepted-receipt.json",
                          lambda v: v.__setitem__(
                              "l24_l25_full_artifact_mismatched_bytes", 1))),
        ("scope", "scope drift",
         lambda r: mutate(r, "accepted-receipt.json",
                          lambda v: v["claims"].__setitem__(
                              "complete_generated_frame", True))),
        ("address", "address-like value",
         lambda r: mutate(r, "accepted-receipt.json",
                          lambda v: v.__setitem__(
                              "debug", "0x1234567890abcdef"))),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    results = []
    with tempfile.TemporaryDirectory(
            prefix="fg-connected-dynamic-l3-l25-negative-") as temp:
        for name, expected, change in cases:
            root = Path(temp) / name
            shutil.copytree(args.evidence, root)
            change(root)
            process = subprocess.run(
                ["python3", str(verifier), "receipt", "--repository",
                 str(repository), "--evidence", str(root), "--skip-negative"],
                cwd=repository, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            observed = process.stdout.strip()
            if process.returncode != 2 or expected not in observed:
                raise RuntimeError(
                    f"mutation {name} failed open: {process.returncode} {observed}")
            results.append({
                "mutation": name, "expected_error_fragment": expected,
                "observed_error": next(
                    (line for line in observed.splitlines()
                     if line.startswith("verification_error=")), observed),
                "exit_code": process.returncode, "result": "pass"})
    audit = {
        "schema": "neuralstudio.fg.connected-dynamic-l3-l25-negative-audit.v1",
        "verifier_sha256": digest(verifier.read_bytes()),
        "mutation_count": len(results), "mutations": results,
        "all_mutations_rejected": True, "result": "pass"}
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
