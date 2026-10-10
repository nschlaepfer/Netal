#!/usr/bin/env python3
"""Exercise fail-closed allocation29 capture mutations."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)


def edit(path: Path, mutate: Callable[[dict], None]) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    verifier = repository / "scripts/verify-fg-allocation29-texture-capture.py"
    cases: list[tuple[str, str, Callable[[Path], None]]] = [
        ("create-byte", "dump hash drift",
         lambda root: flip(root / "oracle-post-create-allocation29.r16f", 17)),
        ("reset-byte", "dump hash drift",
         lambda root: flip(root / "oracle-post-reset-allocation29.r16f", 19)),
        ("create-token", "descriptor identity drift",
         lambda root: edit(root / "post-create-manifest.json",
                           lambda value: value["entries"][0].__setitem__("source_token", 28))),
        ("reset-state", "state-restore strategy drift",
         lambda root: edit(root / "post-reset-manifest.json",
                           lambda value: value.__setitem__("source_state_strategy", "none"))),
        ("row-pitch", "row-footprint drift",
         lambda root: edit(root / "post-reset-manifest.json",
                           lambda value: value["entries"][0].__setitem__("readback_row_pitch", 640))),
        ("feed-back", "capture isolation drift",
         lambda root: edit(root / "post-create-manifest.json",
                           lambda value: value.__setitem__("bytes_fed_back_to_execution", True))),
        ("runtime", "runtime identity drift",
         lambda root: edit(root / "rig-capture-receipt.json",
                           lambda value: value.__setitem__("runtime_sha256", "0" * 64))),
        ("capture-source", "capture source identity drift",
         lambda root: edit(root / "rig-capture-receipt.json",
                           lambda value: value.__setitem__("allocation_capture_source_sha256", "1" * 64))),
        ("receipt-feed-back", "capture isolation drift",
         lambda root: edit(root / "rig-capture-receipt.json",
                           lambda value: value.__setitem__("captured_bytes_fed_back_to_execution", True))),
        ("repeat-hash", "repeat mismatch",
         lambda root: edit(root / "rig-capture-receipt.json",
                           lambda value: value["runs"][1].__setitem__("dump_sha256", "2" * 64))),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    results = []
    with tempfile.TemporaryDirectory(prefix="fg-allocation29-negative-") as temp:
        for name, expected, mutate in cases:
            root = Path(temp) / name
            shutil.copytree(args.root, root)
            mutate(root)
            output = root / "should-not-exist.json"
            process = subprocess.run([
                "python3", str(verifier),
                "--capture-receipt", str(root / "rig-capture-receipt.json"),
                "--post-create-manifest", str(root / "post-create-manifest.json"),
                "--post-reset-manifest", str(root / "post-reset-manifest.json"),
                "--post-create", str(root / "oracle-post-create-allocation29.r16f"),
                "--post-reset", str(root / "oracle-post-reset-allocation29.r16f"),
                "--output", str(output),
            ], cwd=repository, env=environment, text=True,
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            observed = next((line for line in process.stdout.splitlines()
                             if line.startswith("verification_error=")),
                            process.stdout.strip())
            if process.returncode != 2 or expected not in observed or output.exists():
                raise RuntimeError(
                    f"mutation {name} did not fail closed: "
                    f"exit={process.returncode} output={process.stdout!r}"
                )
            results.append({
                "mutation": name,
                "expected_error_fragment": expected,
                "observed_error": observed,
                "exit_code": process.returncode,
                "receipt_created": output.exists(),
                "result": "pass",
            })
    audit = {
        "schema": "neuralstudio.fg.allocation29-texture-negative-audit.v1",
        "verifier_sha256": sha(verifier),
        "mutation_count": len(results),
        "all_rejected": True,
        "mutations": results,
        "result": "pass",
    }
    audit["audit_root_sha256"] = hashlib.sha256(json.dumps(
        audit, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"audit_root_sha256={audit['audit_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
