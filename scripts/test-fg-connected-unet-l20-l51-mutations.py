#!/usr/bin/env python3
"""Mutation audit for the connected generic FG U-Net L20-L51."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)


def edit(path: Path, mutation: Callable[[dict], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtx-root", type=Path, required=True)
    parser.add_argument("--mac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    verifier = repository / "scripts" / "verify-fg-connected-unet-l20-l51.py"
    first = Path("unet-poison-a5-process-a")
    second = Path("unet-poison-a5-process-b")
    cases: list[tuple[str, str, str, Callable[[Path], None]]] = [
        ("rtx-first-tensor", "rtx", "tensor hash", lambda root: flip(root / first / "l20.bin", 17)),
        ("rtx-final-tensor", "rtx", "tensor hash", lambda root: flip(root / first / "l51.bin", 31)),
        ("mac-weight-byte", "mac", "weight artifact", lambda root: flip(root / first / "weight-token-63-after.bin", 101)),
        ("mac-input-byte", "mac", "input artifact", lambda root: flip(root / first / "input-after.bin", 73)),
        (
            "rtx-duplicate-process",
            "rtx",
            "fresh-process gate",
            lambda root: edit(
                root / second / "run-receipt.json",
                lambda value: value.__setitem__(
                    "process_id",
                    json.loads((root / first / "run-receipt.json").read_text())["process_id"],
                ),
            ),
        ),
        (
            "mac-weight-token",
            "mac",
            "weight token",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value["weights"][0].__setitem__("source_token", 999),
            ),
        ),
        (
            "rtx-launch-count",
            "rtx",
            "launch count",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value.__setitem__("selected_launch_count", 31),
            ),
        ),
        (
            "mac-expected-output-ingress",
            "mac",
            "expected output ingress",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value.__setitem__("expected_output_opened", True),
            ),
        ),
        (
            "rtx-archived-intermediate-ingress",
            "rtx",
            "archived intermediate ingress",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value.__setitem__("archived_intermediate_opened", True),
            ),
        ),
        (
            "mac-address-injection",
            "mac",
            "address-bearing key",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value.__setitem__("gpu_pointer", "0x123456789abc"),
            ),
        ),
        (
            "rtx-canary",
            "rtx",
            "canary",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value.__setitem__("canary_mismatched_bytes", 1),
            ),
        ),
        (
            "mac-collision-census",
            "mac",
            "collision census",
            lambda root: edit(
                root / first / "run-receipt.json",
                lambda value: value["tensors"][2].__setitem__(
                    "poison_collision_halfwords", 0
                ),
            ),
        ),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="fg-connected-encoder-negative-") as temporary:
        temporary_root = Path(temporary)
        for name, backend, expected, mutation in cases:
            source = args.rtx_root if backend == "rtx" else args.mac_root
            root = temporary_root / name
            shutil.copytree(source, root)
            mutation(root)
            output = root / "should-not-exist.json"
            process = subprocess.run(
                [
                    "python3",
                    str(verifier),
                    "backend",
                    "--backend",
                    backend,
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            observed = process.stdout.strip()
            passed = process.returncode == 2 and expected in observed and not output.exists()
            if not passed:
                raise RuntimeError(
                    f"mutation {name} failed audit: exit={process.returncode} "
                    f"output={observed!r} receipt_created={output.exists()}"
                )
            results.append(
                {
                    "mutation": name,
                    "backend": backend,
                    "expected_error_fragment": expected,
                    "exit_code": process.returncode,
                    "receipt_created": output.exists(),
                    "result": "pass",
                }
            )
    audit = {
        "schema": "neuralstudio.fg.connected-unet-l20-l51-negative-audit.v1",
        "mutation_count": len(results),
        "all_rejected": True,
        "mutations": results,
        "result": "pass",
    }
    audit["audit_root_sha256"] = canonical_sha256(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"mutation_count={len(results)}")
    print(f"audit_root_sha256={audit['audit_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
